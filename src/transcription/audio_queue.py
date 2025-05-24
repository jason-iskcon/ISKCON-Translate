"""
Audio queue management for transcription engine.

This module handles adding audio segments to the processing queue and
retrieving transcription results with drop-oldest strategy for queue management.
"""

import queue
import time
import threading
from typing import Optional, Tuple, Any

# Import with try-except to handle both direct execution and module import  
try:
    from logging_utils import get_logger
except ImportError:
    from ..logging_utils import get_logger

from .config import DROP_OLDEST_THRESHOLD, LOG_RATE_LIMIT_INTERVAL
from .utils import is_valid_audio

logger = get_logger('transcription.audio_queue')


def add_audio_segment(audio_queue: queue.Queue, audio_segment: tuple, 
                     is_running: bool, warm_up_mode: bool = False,
                     drop_stats_lock: Optional[threading.Lock] = None,
                     drops_last_minute: Optional[list] = None,
                     rate_limit_tracker: Optional[dict] = None) -> bool:
    """Add an audio segment to the processing queue.
    
    This function implements a drop-oldest strategy when the queue is full to ensure
    the most recent audio data is always processed, preventing buffer overruns.
    
    Args:
        audio_queue: Queue for audio segments
        audio_segment: Tuple of (audio_data, timestamp)
        is_running: Whether transcription engine is running
        warm_up_mode: Whether in warm-up mode
        drop_stats_lock: Lock for drop statistics (optional)
        drops_last_minute: List to track drops (optional)
        rate_limit_tracker: Dict for rate limiting logs (optional)
        
    Returns:
        bool: True if segment was added successfully, False if queue was full or invalid input
        
    Raises:
        ValueError: If audio_segment is not a tuple of (audio_data, timestamp)
    """
    # Input validation
    if not is_valid_audio(audio_segment):
        raise ValueError("audio_segment must be a tuple of (audio_data, timestamp)")
        
    if not is_running:
        logger.debug("Transcription not running, ignoring audio segment")
        return False
    
    # Skip if we're in warm-up mode and queue is already full
    if warm_up_mode and audio_queue.full():
        logger.debug("Skipping audio segment during warm-up (queue full)")
        return True  # This is normal queue management, not a failure
    
    # Initialize rate limit tracker if not provided
    if rate_limit_tracker is None:
        rate_limit_tracker = {}
    
    # Persistent retry loop to ensure queue operations always succeed
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Try to add segment to queue directly
            audio_queue.put(audio_segment, block=False)
            return True  # Success!
            
        except queue.Full:
            # Queue is full - implement drop-oldest strategy
            try:
                # Remove oldest segment
                old_segment = audio_queue.get_nowait()
                audio_queue.task_done()
                
                # Add new segment
                audio_queue.put(audio_segment, block=False)
                
                # Track drop statistics
                if drop_stats_lock and drops_last_minute is not None:
                    with drop_stats_lock:
                        drops_last_minute.append(time.time())
                
                # Rate-limited warning
                current_time = time.time()
                last_warning_key = 'drop_oldest_warning'
                if (last_warning_key not in rate_limit_tracker or 
                    current_time - rate_limit_tracker[last_warning_key] >= LOG_RATE_LIMIT_INTERVAL):
                    
                    qsize = audio_queue.qsize()
                    maxsize = audio_queue.maxsize
                    logger.warning(f"discard-oldest (audio) engaged, dropped 1 "
                                 f"(queue: {qsize}/{maxsize})")
                    rate_limit_tracker[last_warning_key] = current_time
                
                return True  # Successfully added after dropping oldest
                
            except queue.Empty:
                # Queue became empty between checks, try direct add
                try:
                    audio_queue.put(audio_segment, block=False)
                    return True
                except queue.Full:
                    # Queue filled up again, continue to next attempt
                    continue
                    
        except Exception as e:
            logger.error(f"Unexpected error adding audio segment (attempt {attempt + 1}): {e}")
            if attempt == max_attempts - 1:
                return False
    
    return False


def get_transcription(result_queue: queue.Queue, is_running: bool, 
                     timeout: float = 0.1) -> Optional[dict]:
    """Get a transcription result from the result queue.
    
    Args:
        result_queue: Queue containing transcription results
        is_running: Whether transcription engine is running
        timeout: Timeout for queue get operation
        
    Returns:
        Optional[dict]: Transcription result or None if none available
    """
    if not is_running:
        return None
        
    try:
        result = result_queue.get(timeout=timeout)
        result_queue.task_done()
        return result
    except queue.Empty:
        return None
    except Exception as e:
        logger.error(f"Error getting transcription result: {e}")
        return None


def clear_queue(q: queue.Queue, queue_name: str = "queue") -> int:
    """Clear all items from a queue.
    
    Args:
        q: Queue to clear
        queue_name: Name for logging purposes
        
    Returns:
        int: Number of items cleared
    """
    cleared = 0
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
            cleared += 1
        except queue.Empty:
            break
    
    if cleared > 0:
        logger.debug(f"Cleared {cleared} items from {queue_name}")
    
    return cleared


def get_queue_stats(audio_queue: queue.Queue, result_queue: queue.Queue) -> dict:
    """Get statistics about queue usage.
    
    Args:
        audio_queue: Audio queue to check
        result_queue: Result queue to check
        
    Returns:
        dict: Queue statistics
    """
    audio_size = audio_queue.qsize()
    audio_maxsize = audio_queue.maxsize
    result_size = result_queue.qsize()
    
    return {
        'audio_queue_size': audio_size,
        'audio_queue_maxsize': audio_maxsize,
        'audio_queue_usage': audio_size / max(1, audio_maxsize),
        'result_queue_size': result_size
    }


def should_drop_oldest(audio_queue: queue.Queue, threshold: float = DROP_OLDEST_THRESHOLD) -> bool:
    """Check if queue usage exceeds drop-oldest threshold.
    
    Args:
        audio_queue: Audio queue to check
        threshold: Threshold ratio (0.0 to 1.0)
        
    Returns:
        bool: True if should implement drop-oldest strategy
    """
    queue_ratio = audio_queue.qsize() / max(1, audio_queue.maxsize)
    return queue_ratio >= threshold 