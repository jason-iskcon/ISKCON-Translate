"""
Background worker logic for transcription processing.

This module contains the main transcription worker that processes audio segments
from the queue and generates transcription results.
"""

import time
import queue
import threading
import collections
from typing import Optional, Any
from faster_whisper import WhisperModel

# Import with try-except to handle both direct execution and module import
try:
    from logging_utils import get_logger, TRACE
except ImportError:
    from ..logging_utils import get_logger, TRACE

from .config import (
    FAILURE_CHECK_INTERVAL, FAILURE_WINDOW_SIZE, MAX_RETRIES, 
    RETRY_BACKOFF_BASE, MAX_CONSECUTIVE_FAILURES
)
from .performance import PerformanceMonitor, log_worker_performance, log_failure_analysis
from .utils import log_audio_info

logger = get_logger('transcription.worker')


def run_transcription_worker(audio_queue: queue.Queue, result_queue: queue.Queue, 
                           model: WhisperModel, is_running_flag: threading.Event,
                           performance_monitor: PerformanceMonitor,
                           processed_chunks_counter: Any, avg_time_tracker: Any,
                           sampling_rate: int, warm_up_mode_flag: threading.Event,
                           drops_last_minute: collections.deque,
                           drop_stats_lock: threading.Lock,
                           worker_threads: list, device: str) -> None:
    """Main transcription worker function.
    
    Args:
        audio_queue: Queue containing audio segments to process
        result_queue: Queue to put transcription results
        model: WhisperModel instance for transcription
        is_running_flag: Flag indicating if transcription should continue
        performance_monitor: Performance monitoring instance
        processed_chunks_counter: Counter for processed chunks
        avg_time_tracker: Tracker for average processing time
        sampling_rate: Audio sampling rate
        warm_up_mode_flag: Flag indicating warm-up mode
        drops_last_minute: Deque tracking recent drops
        drop_stats_lock: Lock for drop statistics
        worker_threads: List of worker threads
        device: Processing device (cpu/cuda)
    """
    worker_name = threading.current_thread().name
    logger.info(f"[{worker_name}] Transcription worker started")
    
    # Performance and failure tracking
    failure_window = collections.deque(maxlen=FAILURE_WINDOW_SIZE)
    last_failure_check = time.time()
    consecutive_failures = 0
    
    try:
        while getattr(is_running_flag, 'is_set', lambda: True)():
            try:
                # Log performance periodically
                performance_monitor.log_system_performance(
                    audio_queue, result_queue, processed_chunks_counter.value,
                    avg_time_tracker.value, device, drops_last_minute,
                    drop_stats_lock, worker_threads
                )
                
                # Get audio data from queue with timeout to allow checking is_running
                try:
                    audio_segment = audio_queue.get(timeout=0.5)
                    failure_window.append(0)  # Record successful queue get
                    
                    # Unpack the segment
                    audio_data, timestamp = audio_segment
                    
                    # Skip empty segments
                    if audio_data is None or len(audio_data) == 0:
                        logger.warning("Skipping empty audio segment")
                        audio_queue.task_done()
                        continue
                        
                except queue.Empty:
                    failure_window.append(0)  # Queue empty is normal, not a failure
                    
                    # Check failure rate periodically
                    current_time = time.time()
                    last_failure_check = log_failure_analysis(
                        list(failure_window), processed_chunks_counter.value,
                        worker_name, current_time, last_failure_check, FAILURE_CHECK_INTERVAL
                    )
                    
                    # Reset consecutive failures if we successfully got an item
                    consecutive_failures = 0
                    
                    logger.trace("Audio queue empty, waiting for segments...")
                    time.sleep(0.1)
                    continue
                    
                except queue.Full:
                    # Queue full is NOT a failure, it's normal queue management
                    failure_window.append(0)  # Record as success, not failure
                    consecutive_failures = 0  # Reset consecutive failure counter
                    logger.trace("Audio queue full, waiting for space...")
                    time.sleep(0.1)
                    continue
                
                # Process the audio segment
                success = _process_audio_segment(
                    audio_data, timestamp, model, result_queue, 
                    worker_name, sampling_rate, processed_chunks_counter,
                    avg_time_tracker
                )
                
                if success:
                    consecutive_failures = 0
                    failure_window.append(0)
                else:
                    consecutive_failures += 1
                    failure_window.append(1)
                    
                    # Handle consecutive failures with backoff
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.warning(f"[{worker_name}] High failure rate, backing off")
                        time.sleep(RETRY_BACKOFF_BASE * consecutive_failures)
                        consecutive_failures = 0  # Reset after backoff
                
                # Mark task as done
                audio_queue.task_done()
                
            except Exception as e:
                logger.error(f"[{worker_name}] Unexpected error in worker loop: {e}", exc_info=True)
                consecutive_failures += 1
                failure_window.append(1)
                time.sleep(0.5)  # Prevent tight error loop
                
    except Exception as e:
        logger.critical(f"[{worker_name}] Fatal error in transcription worker: {e}", exc_info=True)
    finally:
        logger.info(f"[{worker_name}] Transcription worker stopped")


def _process_audio_segment(audio_data, timestamp: float, model: WhisperModel,
                         result_queue: queue.Queue, worker_name: str,
                         sampling_rate: int, processed_chunks_counter: Any,
                         avg_time_tracker: Any) -> bool:
    """Process a single audio segment.
    
    Args:
        audio_data: Audio data to process
        timestamp: Audio timestamp
        model: WhisperModel for transcription
        result_queue: Queue to put results
        worker_name: Name of the worker thread
        sampling_rate: Audio sampling rate
        processed_chunks_counter: Counter for processed chunks
        avg_time_tracker: Tracker for average processing time
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    try:
        # Log audio segment details
        log_audio_info(audio_data, timestamp, sampling_rate, level="trace")
        
        # Log model inference start
        logger.debug(f"[{worker_name}] Starting audio transcription...")
        start_process_time = time.time()
        
        # Transcribe the audio with retry logic
        segments, info = _transcribe_with_retry(model, audio_data, worker_name)
        
        if segments is None:
            return False
            
        # Log successful transcription
        processing_time = time.time() - start_process_time
        log_worker_performance(
            worker_name, processing_time, len(segments), 
            info.language, info.language_probability
        )
        
        # Update performance counters
        _update_performance_counters(
            processed_chunks_counter, avg_time_tracker, processing_time
        )
        
        # Process and queue results
        _queue_transcription_results(
            segments, result_queue, timestamp, processing_time, worker_name
        )
        
        return True
        
    except Exception as e:
        logger.error(f"[{worker_name}] Error processing audio segment: {e}", exc_info=True)
        return False


def _transcribe_with_retry(model: WhisperModel, audio_data, worker_name: str):
    """Transcribe audio with retry logic for transient errors.
    
    Args:
        model: WhisperModel for transcription
        audio_data: Audio data to transcribe
        worker_name: Name of the worker thread
        
    Returns:
        Tuple of (segments, info) or (None, None) on failure
    """
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            segments, info = model.transcribe(
                audio_data,
                language="en",
                beam_size=5,
                word_timestamps=True,
                temperature=0.0  # Disable sampling for more consistent results
            )
            segments = list(segments)  # Convert generator to list to catch errors early
            return segments, info
            
        except RuntimeError as e:
            last_error = e
            if "CUDA out of memory" in str(e) and attempt < MAX_RETRIES - 1:
                logger.warning(f"[{worker_name}] CUDA OOM on attempt {attempt + 1}, retrying...")
                time.sleep(RETRY_BACKOFF_BASE * (attempt + 1))  # Exponential backoff
                continue
            raise  # Re-raise if not a retryable error or out of retries
    
    # This block runs if the loop completes without breaking
    if last_error:
        raise last_error  # Re-raise the last error if all retries failed
    
    return None, None


def _update_performance_counters(processed_chunks_counter: Any, avg_time_tracker: Any,
                               processing_time: float) -> None:
    """Update performance tracking counters.
    
    Args:
        processed_chunks_counter: Counter for processed chunks
        avg_time_tracker: Tracker for average processing time
        processing_time: Time taken for this processing
    """
    # Update processing statistics with thread safety
    try:
        # Atomic increment of processed chunks
        with processed_chunks_counter.get_lock():
            processed_chunks_counter.value += 1
            current_count = processed_chunks_counter.value
        
        # Update average processing time with exponential moving average
        with avg_time_tracker.get_lock():
            if avg_time_tracker.value == 0:
                avg_time_tracker.value = processing_time
            else:
                # Exponential moving average with alpha=0.1
                alpha = 0.1
                avg_time_tracker.value = (alpha * processing_time + 
                                        (1 - alpha) * avg_time_tracker.value)
                                        
    except Exception as e:
        logger.warning(f"Error updating performance counters: {e}")


def _queue_transcription_results(segments: list, result_queue: queue.Queue,
                               timestamp: float, processing_time: float,
                               worker_name: str) -> None:
    """Queue transcription results for consumer.
    
    Args:
        segments: Transcription segments
        result_queue: Queue to put results
        timestamp: Audio timestamp
        processing_time: Processing time
        worker_name: Name of the worker thread
    """
    try:
        for segment in segments:
            # Create transcription result
            result = {
                'text': segment.text.strip(),
                'start': segment.start + timestamp,  # Add offset for absolute timing
                'end': segment.end + timestamp,
                'confidence': getattr(segment, 'avg_logprob', 0.0),
                'words': [],
                'processing_time': processing_time,
                'worker': worker_name
            }
            
            # Add word-level timestamps if available
            if hasattr(segment, 'words') and segment.words:
                result['words'] = [
                    {
                        'word': word.word,
                        'start': word.start + timestamp,
                        'end': word.end + timestamp,
                        'probability': word.probability
                    }
                    for word in segment.words
                ]
            
            # Queue the result
            try:
                result_queue.put(result, timeout=1.0)
                logger.trace(f"[{worker_name}] Queued result: {result['text'][:50]}...")
            except queue.Full:
                logger.warning(f"[{worker_name}] Result queue full, dropping result")
                
    except Exception as e:
        logger.error(f"[{worker_name}] Error queuing results: {e}", exc_info=True) 