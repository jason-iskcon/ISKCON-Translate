"""
Main transcription engine class.

This module contains the refactored TranscriptionEngine class that uses
the modularized components while maintaining the original public API.
"""

import numpy as np
import threading
import queue
from queue import Queue
import time
import collections
import torch
import multiprocessing
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor

# Configure logging for third-party libraries
import logging
for lib in ['numba', 'huggingface_hub', 'whisper', 'faster_whisper']:
    logging.getLogger(lib).setLevel(logging.WARNING)

# Suppress specific warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numba')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import with try-except to handle both direct execution and module import
try:
    from logging_utils import get_logger, setup_logging, TRACE
except ImportError:
    from ..logging_utils import get_logger, setup_logging, TRACE

# Import singleton clock
try:
    from clock import CLOCK
except ImportError:
    from ..clock import CLOCK

# Import modular components
from .config import CPU_PARAMS, GPU_PARAMS, SAMPLING_RATE, WORKER_THREAD_PREFIX
from .model_loader import init_whisper_model, validate_gpu_availability
from .audio_queue import add_audio_segment, get_transcription, clear_queue
from .worker import run_transcription_worker
from .performance import PerformanceMonitor
from .utils import validate_chunk_parameters, calculate_adaptive_sleep

# Configure logging for faster_whisper to reduce verbosity
logging.getLogger('faster_whisper').setLevel(logging.WARNING)

# Get logger instance with module name for better filtering
logger = get_logger('transcription.engine')


class TranscriptionEngine:
    """Modular transcription engine for real-time audio processing."""
    
    playback_start_time = 0.0  # Global playback time origin for sync
    
    def __init__(self, model_size: str = "small", device: str = "auto", 
                 compute_type: str = "auto", language: str = "en", 
                 warm_up_complete_event: Optional[threading.Event] = None):
        """Initialize the transcription engine.
        
        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
            device: Device to run the model on ('cuda', 'cpu', or 'auto' for automatic detection)
            compute_type: Computation type ('float16', 'int8', or 'auto' for automatic selection)
            language: Language code for transcription (e.g., 'en', 'fr', 'it', 'es', 'de')
            warm_up_complete_event: Event to signal when warm-up phase is complete
        """
        self.model_size = model_size
        self.language = language  # Store language for transcription
        self.sampling_rate = SAMPLING_RATE
        self.warm_up_complete_event = warm_up_complete_event
        
        # Initialize with default values, will be set in _init_model
        self.device = "cpu"
        self.compute_type = "int8"
        
        # Validate GPU availability for production deployment
        validate_gpu_availability(device)
        
        # Initialize the model first to determine actual device
        self.model, self.device, self.compute_type = init_whisper_model(
            model_size, device, compute_type
        )
        
        # Set device-specific parameters based on actual device
        params = CPU_PARAMS if self.device == "cpu" else GPU_PARAMS
        self.chunk_size = params['chunk_size']
        self.overlap = params['overlap']
        self.n_workers = params['n_workers']
        queue_maxsize = params['queue_maxsize']
        
        logger.info(f"🔧 Using {self.device.upper()} parameters: chunk_size={self.chunk_size}s, "
                   f"overlap={self.overlap}s, queue_maxsize={queue_maxsize}, n_workers={self.n_workers}")
        
        # Processing state
        self.is_running = False
        self._warm_up_mode = True
        
        # Shared counters for thread-safe statistics
        self.processed_chunks = multiprocessing.Value('i', 0)
        self.average_processing_time = multiprocessing.Value('d', 0.0)
        self.playback_start_time = 0.0
        
        # Rate limiting for log messages
        self._last_adaptive_drop_warning_time = 0
        
        # Track drop statistics separately from failures
        self.drops_last_minute = collections.deque(maxlen=60)  # Track drops per second for last minute
        self.drop_stats_lock = threading.Lock()
        
        # Threading and queues
        self.audio_queue = queue.Queue(maxsize=queue_maxsize)
        logger.info(f"🔧 CONSTRUCTOR: TranscriptionEngine audio_queue created with maxsize={self.audio_queue.maxsize}")
        self.result_queue = queue.Queue()
        self.worker_threads = []
        self.audio_processor_thread: Optional[threading.Thread] = None
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.n_workers,
            thread_name_prefix="TransWorker"
        ) if self.n_workers > 1 else None
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Rate limiting tracker
        self.rate_limit_tracker = {}
        
        logger.info(f"TranscriptionEngine initialized with {self.n_workers} workers on {self.device.upper()}")

    def _switch_to_cpu_model(self) -> bool:
        """Switch from GPU to CPU model when CUDA runtime fails.
        
        Returns:
            bool: True if switch was successful, False otherwise
        """
        if self.device == "cpu":
            logger.debug("Already using CPU model, no switch needed")
            return True
            
        try:
            logger.warning("🔄 Switching to CPU model due to CUDA runtime error...")
            
            # Initialize new CPU model
            new_model, new_device, new_compute_type = init_whisper_model(
                self.model_size, "cpu", "int8"
            )
            
            # Update model and device settings
            self.model = new_model
            self.device = new_device
            self.compute_type = new_compute_type
            
            # Update parameters to CPU-specific ones
            params = CPU_PARAMS
            self.chunk_size = params['chunk_size']
            self.overlap = params['overlap']
            self.n_workers = params['n_workers']
            
            logger.warning(f"✅ Successfully switched to CPU model (chunk_size={self.chunk_size}s, "
                         f"overlap={self.overlap}s, n_workers={self.n_workers})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to switch to CPU model: {e}", exc_info=True)
            return False

    def start_transcription(self) -> bool:
        """Start the transcription engine.
        
        Returns:
            bool: True if the transcription was started, False if it was already running
        """
        if self.is_running:
            logger.debug("Transcription already running, ignoring start request")
            return False
            
        try:
            logger.info("Starting transcription engine")
            self.is_running = True
            self._warm_up_mode = True  # Start in warm-up mode
            
            # Clear any existing queue state and reset counters
            clear_queue(self.audio_queue, "audio_queue")
            clear_queue(self.result_queue, "result_queue")
            
            # Reset processing state
            with self.processed_chunks.get_lock():
                self.processed_chunks.value = 0
            with self.average_processing_time.get_lock():
                self.average_processing_time.value = 0.0
            
            # Start worker threads based on n_workers
            self.worker_threads = []
            for i in range(self.n_workers):
                worker = threading.Thread(
                    target=run_transcription_worker,
                    args=(
                        self.audio_queue, self.result_queue, self.model,
                        self.is_running, self.performance_monitor,
                        self.processed_chunks, self.average_processing_time,
                        self.sampling_rate, self._warm_up_mode, self.drops_last_minute,
                        self.drop_stats_lock, self.worker_threads, self.device, 
                        self, self.language  # Pass language to worker
                    ),
                    name=f"{WORKER_THREAD_PREFIX}#{i}"
                )
                worker.daemon = True
                worker.start()
                self.worker_threads.append(worker)
                
                # Stagger worker starts to reduce initial load spikes
                if i < self.n_workers - 1:  # Don't sleep after the last worker
                    stagger_delay = self.chunk_size / 2  # Half chunk duration
                    logger.debug(f"Staggering worker start by {stagger_delay:.1f}s")
                    time.sleep(stagger_delay)
            
            logger.info(f"Started {self.n_workers} transcription worker(s) in warm-up mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start transcription: {e}", exc_info=True)
            self.is_running = False
            return False
    
    def stop_transcription(self, timeout: float = 5.0) -> bool:
        """Stop the transcription engine gracefully.
        
        Args:
            timeout: Maximum time to wait for the worker threads to stop (seconds)
            
        Returns:
            bool: True if stopped cleanly, False if timeout occurred or already stopped
        """
        if not self.is_running:
            logger.debug("Transcription not running, ignoring stop request")
            return True
            
        logger.info("Stopping transcription engine...")
        self.is_running = False
        self._warm_up_mode = False
        
        try:
            # Clear queues
            audio_cleared = clear_queue(self.audio_queue, "audio_queue")
            result_cleared = clear_queue(self.result_queue, "result_queue")
            
            # Signal and wait for all worker threads to finish
            if self.worker_threads:
                logger.debug(f"Waiting for {len(self.worker_threads)} worker threads to finish...")
                
                # Calculate timeout per worker
                worker_timeout = max(0.1, timeout / max(1, len(self.worker_threads)))
                all_stopped = True
                
                for i, worker in enumerate(self.worker_threads):
                    if worker.is_alive():
                        worker.join(timeout=worker_timeout)
                        if worker.is_alive():
                            logger.warning(f"Worker thread {i+1} did not stop gracefully")
                            all_stopped = False
                
                if not all_stopped:
                    logger.warning("Not all worker threads stopped gracefully")
                
                self.worker_threads = []
            
            # Shutdown thread pool if it exists
            if hasattr(self, 'thread_pool') and self.thread_pool is not None:
                self.thread_pool.shutdown(wait=False)
                self.thread_pool = None
            
            # Reset playback start time for next session
            self.playback_start_time = 0.0
            logger.info("Transcription engine stopped successfully")
            return all_stopped
            
        except Exception as e:
            logger.error(f"Error during transcription shutdown: {e}", exc_info=True)
            return False

    def _run_worker(self) -> None:
        """Run a transcription worker with the modular worker function."""
        # Create threading event that mimics the is_running flag
        is_running_event = threading.Event()
        is_running_event.set()  # Start as running
        
        # Create warm-up mode event
        warm_up_event = threading.Event()
        if self._warm_up_mode:
            warm_up_event.set()
        
        try:
            # Call the modular worker function
            run_transcription_worker(
                audio_queue=self.audio_queue,
                result_queue=self.result_queue,
                model=self.model,
                is_running_flag=self,  # Pass self to check is_running attribute
                performance_monitor=self.performance_monitor,
                processed_chunks_counter=self.processed_chunks,
                avg_time_tracker=self.average_processing_time,
                sampling_rate=self.sampling_rate,
                warm_up_mode_flag=warm_up_event,
                drops_last_minute=self.drops_last_minute,
                drop_stats_lock=self.drop_stats_lock,
                worker_threads=self.worker_threads,
                device=self.device,
                engine_instance=self,  # Pass self for CPU fallback
                language=self.language  # Pass language to worker
            )
        except Exception as e:
            logger.error(f"Worker thread failed: {e}", exc_info=True)
        finally:
            is_running_event.clear()

    def add_audio_segment(self, audio_segment: tuple) -> bool:
        """Add an audio segment to the processing queue.
        
        Args:
            audio_segment: Tuple of (audio_data, timestamp)
            
        Returns:
            bool: True if segment was added successfully, False otherwise
        """
        # Set playback_start_time to the timestamp of the first segment if not already set
        if self.playback_start_time == 0.0 and len(audio_segment) == 2:
            _, timestamp = audio_segment
            if timestamp > 0:
                self.playback_start_time = timestamp
                logger.debug(f"Set playback start time to {timestamp:.2f}s")
        
        return add_audio_segment(
            audio_queue=self.audio_queue,
            audio_segment=audio_segment,
            is_running=self.is_running,
            warm_up_mode=self._warm_up_mode,
            drop_stats_lock=self.drop_stats_lock,
            drops_last_minute=self.drops_last_minute,
            rate_limit_tracker=self.rate_limit_tracker
        )

    def get_transcription(self) -> Optional[dict]:
        """Get a transcription result from the result queue.
        
        Returns:
            Optional[dict]: Transcription result or None if none available
        """
        return get_transcription(self.result_queue, self.is_running)

    def process_audio(self, video_source, chunk_size: Optional[float] = None, 
                     overlap: Optional[float] = None) -> None:
        """Process audio from a video source in chunks with overlap.
        
        Args:
            video_source: Video source object that provides audio chunks
            chunk_size: Size of audio chunks in seconds (uses device-specific default if None)
            overlap: Overlap between chunks in seconds (uses device-specific default if None)
            
        Notes:
            - Uses a sliding window approach for better transcription continuity
            - Automatically adjusts queue behavior during warm-up vs. normal operation
            - Implements backpressure to prevent queue overflow
        """
        # Use device-specific parameters if not explicitly provided
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
            
        # Validate parameters
        if not validate_chunk_parameters(chunk_size, overlap):
            logger.error(f"Invalid chunk parameters: size={chunk_size}s, overlap={overlap}s")
            return
            
        logger.info(f"Starting audio processing (chunk_size={chunk_size}s, overlap={overlap}s, device={self.device})")
        
        # Wait for the main warm-up period to complete before active processing
        if self.warm_up_complete_event:
            logger.info("Audio processing thread waiting for warm-up barrier...")
            
            # Check if video_source has the new barrier system
            if hasattr(video_source, 'warm_up_barrier'):
                try:
                    video_source.warm_up_barrier.wait()
                    logger.info("Audio processing thread passed warm-up barrier.")
                except threading.BrokenBarrierError:
                    logger.error("Warm-up barrier was broken, falling back to event-based sync")
                    self.warm_up_complete_event.wait()
                    logger.info("Application warm-up complete, audio processing thread proceeding.")
            else:
                # Fall back to old event-based system
                logger.info("Audio processing thread waiting for application warm-up to complete...")
                self.warm_up_complete_event.wait()
                logger.info("Application warm-up complete, audio processing thread proceeding.")
                
            self._warm_up_mode = False # Explicitly transition out of warm-up mode for the engine
            
            # Clear any audio segments that might have been queued during app warm-up
            audio_cleared = clear_queue(self.audio_queue, "audio_queue")
            result_cleared = clear_queue(self.result_queue, "result_queue")
                    
            if audio_cleared > 0 or result_cleared > 0:
                logger.info(
                    f"Cleared {audio_cleared} audio segments and {result_cleared} results "
                    "from queue after warm-up."
                )
                
            # Reset processing state for normal operation
            with self.processed_chunks.get_lock():
                self.processed_chunks.value = 0
            with self.average_processing_time.get_lock():
                self.average_processing_time.value = 0.0
            self._warm_up_mode = False  # Ensure we're out of warm-up mode
                
            logger.info("Warm-up complete, transcription engine ready for normal operation")
            
        else:
            logger.warning("No warm_up_complete_event available to TranscriptionEngine. Audio processing may behave unexpectedly regarding warm-up state.")
            self._warm_up_mode = False # Default to not in warm-up if event is missing
        
        # Process audio continuously
        last_process_time = 0
        chunk_counter = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        # Track timing mode to handle transition correctly
        was_in_warmup = True
        
        # Debug timing tracking
        last_debug_log = 0
        debug_log_interval = 2.0  # Log every 2 seconds
        
        try:
            while (getattr(video_source, 'is_running', False) and 
                   not getattr(video_source, '_shutdown_event', threading.Event()).is_set() and 
                   self.is_running):  # Also check if transcription is still running
                try:
                    # Get current elapsed time (no media offset)
                    warm_up_complete = getattr(video_source, 'warm_up_complete', None)
                    warm_up_is_set = warm_up_complete.is_set() if warm_up_complete else False
                    
                    if warm_up_is_set:
                        # Use elapsed time since playback start (consistent with other timing)
                        current_time = time.time() - video_source.playback_start_time
                        in_warmup = False
                        
                        # Reset timing on transition from warm-up to normal operation
                        if was_in_warmup:
                            logger.debug("Transitioned from warm-up to normal operation, resetting timing")
                            last_process_time = 0  # Reset to start fresh timing
                            was_in_warmup = False  # This now persists across loop iterations
                    else:
                        with getattr(video_source, 'audio_position_lock', threading.Lock()):
                            # During warm-up, use audio position without seek offset
                            current_time = video_source.audio_position
                        in_warmup = True
                    
                    # Process audio in chunks with overlap
                    time_since_last = current_time - last_process_time
                    should_process = time_since_last >= (chunk_size - overlap)
                    
                    # Debug logging for timing calculations
                    current_debug_time = time.time()
                    if current_debug_time - last_debug_log >= debug_log_interval:
                        logger.debug(f"[AUDIO DEBUG] warmup={in_warmup}, current_time={current_time:.2f}s, "
                                   f"last_process_time={last_process_time:.2f}s, time_since_last={time_since_last:.2f}s, "
                                   f"should_process={should_process} (need >={chunk_size - overlap:.1f}s), "
                                   f"chunks_processed={chunk_counter}")
                        last_debug_log = current_debug_time
                    
                    if should_process:
                        chunk_counter += 1
                        chunk_id = f"#{chunk_counter} ({'warmup' if in_warmup else 'normal'})"
                        
                        # Get audio chunk from current position with specified size
                        try:
                            audio_chunk = video_source.get_audio_chunk(chunk_size=chunk_size)
                            
                            if audio_chunk is not None and len(audio_chunk[0]) > 0:
                                audio_data, timestamp = audio_chunk
                                
                                # Log chunk details at appropriate level
                                log_msg = (f"Chunk {chunk_id} at {current_time:.2f}s: "
                                         f"{len(audio_data)} samples, "
                                         f"duration={len(audio_data)/self.sampling_rate:.2f}s, "
                                         f"queue={self.audio_queue.qsize()}/{self.audio_queue.maxsize}")
                                
                                if in_warmup:
                                    logger.debug(log_msg)
                                else:
                                    logger.info(f"[NORMAL] {log_msg}")  # Make normal operation chunks visible
                                
                                # Add to transcription engine
                                success = self.add_audio_segment(audio_chunk)
                                if success:
                                    # Success - reset failure counter and continue
                                    last_process_time = current_time
                                    consecutive_failures = 0
                                    logger.debug(f"[AUDIO DEBUG] Added chunk successfully, updated last_process_time to {last_process_time:.2f}s")
                                else:
                                    # This is an ACTUAL error
                                    consecutive_failures += 1
                                    logger.warning(f"Failed to add chunk {chunk_id} (consecutive errors: {consecutive_failures})")
                                    
                                    # Adaptive back-off instead of stopping
                                    if consecutive_failures > max_consecutive_failures:
                                        logger.warning(f"Back-off due to {consecutive_failures} consecutive errors")
                                        time.sleep(chunk_size)  # Sleep for chunk duration to let workers catch up
                                        consecutive_failures = 0  # Clear failures after back-off
                                        continue
                            else:
                                logger.warning(f"Received empty audio chunk at {current_time:.2f}s")
                                
                        except Exception as e:
                            consecutive_failures += 1
                            logger.error(f"Error getting audio chunk: {e}", exc_info=consecutive_failures > 1)
                            
                            # Back-off strategy instead of accumulating failures
                            if consecutive_failures > max_consecutive_failures:
                                logger.warning(f"Consecutive audio errors ({consecutive_failures}), backing off")
                                time.sleep(chunk_size * 2)  # Sleep for two chunk durations on audio errors
                                consecutive_failures = 0  # Clear failures after back-off
                            else:
                                time.sleep(0.5)  # Shorter back-off for initial errors
                            continue
                    
                    # Adaptive sleep based on processing state
                    queue_ratio = self.audio_queue.qsize() / max(1, self.audio_queue.maxsize)
                    sleep_time = calculate_adaptive_sleep(
                        should_process, time_since_last, chunk_size, overlap, queue_ratio
                    )
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Unexpected error in audio processing loop: {e}", exc_info=True)
                    time.sleep(0.5)  # Prevent tight error loop
                    
        except Exception as e:
            logger.critical(f"Fatal error in audio processing: {e}", exc_info=True)
            raise
        finally:
            logger.info("Audio processing thread stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start_transcription()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_transcription() 