"""
Utility functions for transcription engine.

This module provides helper functions for logging, validation, 
and audio processing used throughout the transcription engine.
"""

import numpy as np
import logging
import time
from typing import Optional, Any, Tuple

# Import with try-except to handle both direct execution and module import
try:
    from logging_utils import get_logger, TRACE
except ImportError:
    from ..logging_utils import get_logger, TRACE

logger = get_logger('transcription.utils')


def is_valid_audio(audio_segment: Any) -> bool:
    """Validate audio segment data.
    
    Args:
        audio_segment: Audio segment to validate (should be tuple of (audio_data, timestamp))
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(audio_segment, tuple) or len(audio_segment) != 2:
        return False
        
    audio_data, timestamp = audio_segment
    
    if not isinstance(audio_data, np.ndarray):
        logger.error("Invalid audio data: expected numpy array")
        return False
        
    if len(audio_data) == 0:
        logger.warning("Received empty audio segment")
        return False
        
    if not isinstance(timestamp, (int, float)):
        logger.error("Invalid timestamp: expected numeric value")
        return False
        
    return True


def log_audio_info(audio_data: np.ndarray, timestamp: float, sampling_rate: int, level: str = "trace") -> None:
    """Log detailed information about an audio segment.
    
    Args:
        audio_data: Audio data array
        timestamp: Audio timestamp
        sampling_rate: Audio sampling rate
        level: Logging level ("trace", "debug", "info")
    """
    duration = len(audio_data) / sampling_rate
    log_func = getattr(logger, level.lower(), logger.debug)
    
    log_func(f"Audio segment details - "
             f"shape: {audio_data.shape}, "
             f"dtype: {audio_data.dtype}, "
             f"duration: {duration:.2f}s, "
             f"timestamp: {timestamp:.2f}s")


def clip_duration(duration: float, max_duration: float) -> float:
    """Clip duration to maximum value.
    
    Args:
        duration: Duration to clip
        max_duration: Maximum allowed duration
        
    Returns:
        float: Clipped duration
    """
    return min(duration, max_duration)


def calculate_adaptive_sleep(should_process: bool, time_since_last: float, 
                           chunk_size: float, overlap: float, 
                           queue_ratio: float, min_sleep: float = 0.1, 
                           max_sleep: float = 1.0) -> float:
    """Calculate adaptive sleep time based on processing state.
    
    Args:
        should_process: Whether a chunk should be processed
        time_since_last: Time since last processing
        chunk_size: Audio chunk size
        overlap: Audio overlap
        queue_ratio: Queue fullness ratio (0.0 to 1.0)
        min_sleep: Minimum sleep time
        max_sleep: Maximum sleep time
        
    Returns:
        float: Sleep time in seconds
    """
    if should_process:
        # Just processed a chunk, sleep less to be responsive
        sleep_time = min_sleep
    else:
        # Calculate how long until next chunk should be processed
        time_until_next = (chunk_size - overlap) - time_since_last
        if time_until_next > 0:
            # Sleep for most of the remaining time, but check periodically
            sleep_time = min(0.5, max(min_sleep, time_until_next * 0.8))
        else:
            sleep_time = min_sleep
    
    # If queue is getting full, slow down to let workers catch up
    if queue_ratio > 0.6:  # If queue is more than 60% full
        sleep_time = min(max_sleep, sleep_time * (1 + queue_ratio * 2))
    
    return sleep_time


def rate_limited_log(logger_instance: logging.Logger, level: str, message: str, 
                    rate_limit_key: str, interval: float, 
                    rate_limit_tracker: dict) -> bool:
    """Log a message with rate limiting.
    
    Args:
        logger_instance: Logger instance to use
        level: Log level ("debug", "info", "warning", "error")
        message: Message to log
        rate_limit_key: Unique key for this rate limit
        interval: Minimum interval between messages
        rate_limit_tracker: Dictionary to track rate limits
        
    Returns:
        bool: True if message was logged, False if rate limited
    """
    current_time = time.time()
    last_log_time = rate_limit_tracker.get(rate_limit_key, 0)
    
    if current_time - last_log_time >= interval:
        log_func = getattr(logger_instance, level.lower(), logger_instance.info)
        log_func(message)
        rate_limit_tracker[rate_limit_key] = current_time
        return True
    
    return False


def format_queue_info(audio_queue_size: int, audio_queue_maxsize: int, 
                     result_queue_size: int) -> str:
    """Format queue information for logging.
    
    Args:
        audio_queue_size: Current audio queue size
        audio_queue_maxsize: Maximum audio queue size
        result_queue_size: Current result queue size
        
    Returns:
        str: Formatted queue information
    """
    return f"Audio Q: {audio_queue_size}/{audio_queue_maxsize} | Result Q: {result_queue_size}"


def validate_chunk_parameters(chunk_size: float, overlap: float) -> bool:
    """Validate audio chunk processing parameters.
    
    Args:
        chunk_size: Size of audio chunks in seconds
        overlap: Overlap between chunks in seconds
        
    Returns:
        bool: True if parameters are valid
    """
    if chunk_size <= 0:
        logger.error(f"Invalid chunk_size: {chunk_size}s (must be > 0)")
        return False
        
    if overlap < 0:
        logger.error(f"Invalid overlap: {overlap}s (must be >= 0)")
        return False
        
    if overlap >= chunk_size:
        logger.error(f"Invalid overlap: {overlap}s >= chunk_size: {chunk_size}s")
        return False
        
    return True


def calculate_drop_rate(drops_last_minute: list, processed_chunks: int) -> Tuple[float, int]:
    """Calculate drop rate statistics.
    
    Args:
        drops_last_minute: List of drop timestamps from last minute
        processed_chunks: Total number of processed chunks
        
    Returns:
        Tuple[float, int]: (drop_rate, drops_per_minute)
    """
    current_time = time.time()
    recent_drops = [t for t in drops_last_minute if current_time - t <= 60.0]
    drops_per_minute = len(recent_drops)
    
    if processed_chunks > 0:
        drop_rate = drops_per_minute / processed_chunks
    else:
        drop_rate = 0.0
        
    return drop_rate, drops_per_minute


def get_chunk_id(chunk_counter: int, in_warmup: bool) -> str:
    """Generate a chunk ID string for logging.
    
    Args:
        chunk_counter: Current chunk counter
        in_warmup: Whether currently in warmup mode
        
    Returns:
        str: Formatted chunk ID
    """
    mode = 'warmup' if in_warmup else 'normal'
    return f"#{chunk_counter} ({mode})" 