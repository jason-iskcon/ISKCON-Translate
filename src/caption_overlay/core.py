"""Core state management for caption overlay functionality."""
import time
import threading
import logging
from typing import List, Dict, Any, Optional

# Import singleton clock
try:
    from ..clock import CLOCK
except ImportError:
    try:
        from clock import CLOCK
    except ImportError:
        from src.clock import CLOCK

# Import with try-except to handle both direct execution and module import
try:
    from ..logging_utils import get_logger, TRACE
    from .utils import (
        normalize_text, convert_timestamp, validate_duration,
        should_skip_similar_caption, adjust_timestamp_if_past
    )
except ImportError:
    from src.logging_utils import get_logger, TRACE
    from .utils import (
        normalize_text, convert_timestamp, validate_duration,
        should_skip_similar_caption, adjust_timestamp_if_past
    )

logger = get_logger(__name__)

class CaptionCore:
    """Core caption management functionality."""
    
    def __init__(self):
        """Initialize caption core."""
        self.captions = []
        self.caption_lock = threading.Lock()
        self.timing_buffer = 0.1  # Increased buffer for smoother transitions
        self._last_stats_log_time = 0.0
        self._caption_intervals = []
        self._last_caption_time = 0.0
        self.video_start_time = 0.0  # Initialize video start time
        
    def add_caption(self, text: str, timestamp: float, duration: float = 3.0, is_absolute: bool = False) -> dict:
        """Add a new caption with precise timing.
        
        Args:
            text: Caption text
            timestamp: Start time in seconds
            duration: Display duration in seconds
            is_absolute: Whether timestamp is absolute time
        
        Returns:
            dict: The added caption, or None if skipped
        """
        if not text:
            logger.warning("[CAPTION] Skipping empty caption.")
            return None
        with self.caption_lock:
            # Store original timestamp for logging
            original_timestamp = timestamp
            # Convert absolute timestamp to relative if needed
            if is_absolute:
                timestamp = timestamp - self.video_start_time
            # Negative timestamps are treated as 0
            if timestamp < 0:
                logger.warning(f"[CAPTION] Negative timestamp {timestamp:.2f}, treating as 0.")
                timestamp = 0.0
            # Calculate end time
            end_time = timestamp + duration
            # Add caption with precise timing
            caption_id = len(self.captions) + 1
            caption = {
                'id': caption_id,
                'text': text,
                'start_time': timestamp,
                'end_time': end_time,
                'duration': duration,
                'added_at': time.time(),
                'original_timestamp': original_timestamp,
                'was_absolute': is_absolute
            }
            self.captions.append(caption)
            # Log caption addition with precise timing
            logger.info(f"[CAPTION] Added caption #{caption_id}: '{text}' ({timestamp:.2f}s-{end_time:.2f}s)")
            # Track timing statistics
            current_time = time.time()
            if self._last_caption_time > 0:
                interval = current_time - self._last_caption_time
                self._caption_intervals.append(interval)
                if len(self._caption_intervals) > 10:
                    self._caption_intervals.pop(0)
            self._last_caption_time = current_time
            return caption
    
    def get_active_captions(self, current_time: float) -> List[Dict[str, Any]]:
        """Get currently active captions with precise timing.
        
        Args:
            current_time: Current time in seconds
            
        Returns:
            List of active captions
        """
        with self.caption_lock:
            # Calculate buffer times with a larger buffer for smoother transitions
            start_buffer = current_time - self.timing_buffer
            end_buffer = current_time + self.timing_buffer
            
            # Find active captions with precise timing
            active_captions = []
            for c in self.captions:  # Process in chronological order
                # Check if caption is active within buffer
                if (c['start_time'] <= end_buffer and 
                    c['end_time'] >= start_buffer):
                    # Calculate fade factor for smooth transitions
                    fade_start = max(0, (current_time - c['start_time']) / 0.2)  # 0.2s fade in
                    fade_end = max(0, (c['end_time'] - current_time) / 0.2)  # 0.2s fade out
                    fade_factor = min(fade_start, fade_end, 1.0)
                    
                    # Only include if not fully faded out
                    if fade_factor > 0:
                        c['fade_factor'] = fade_factor
                        active_captions.append(c)
            
            # Sort by start time to maintain proper order
            active_captions.sort(key=lambda x: x['start_time'])
            
            # Log timing stats periodically
            self._log_timing_stats(current_time, active_captions)
            
            return active_captions
    
    def _log_timing_stats(self, current_time: float, active_captions: List[Dict[str, Any]]) -> None:
        """Log timing statistics for debugging.
        
        Args:
            current_time: Current time in seconds
            active_captions: List of currently active captions
        """
        # Log stats every second
        if current_time - self._last_stats_log_time >= 1.0:
            logger.info("\n=== CAPTION TIMING STATS ===")
            logger.info(f"Current time: {current_time:.3f}s")
            logger.info(f"Active captions: {len(active_captions)}")
            
            if len(self._caption_intervals) >= 2:
                avg_interval = sum(self._caption_intervals) / len(self._caption_intervals)
                min_interval = min(self._caption_intervals)
                max_interval = max(self._caption_intervals)
                logger.info(f"Average caption interval: {avg_interval*1000:.2f}ms")
                logger.info(f"Min caption interval: {min_interval*1000:.2f}ms")
                logger.info(f"Max caption interval: {max_interval*1000:.2f}ms")
            
            logger.info(f"Timing buffer: {self.timing_buffer*1000:.2f}ms")
            self._last_stats_log_time = current_time
    
    def clear_captions(self) -> None:
        """Clear all captions."""
        with self.caption_lock:
            self.captions.clear()
            logger.info("[CAPTION] Cleared all captions")
    
    def prune_captions(self, current_time: float, buffer: float = 1.0) -> None:
        """Remove old captions to prevent memory growth.
        
        Args:
            current_time: Current time in seconds
            buffer: Buffer time in seconds for keeping expired captions
        """
        with self.caption_lock:
            # Keep captions that are either:
            # 1. Currently active
            # 2. Will be active in the future
            # 3. Ended within the buffer time (for debugging)
            self.captions = [
                c for c in self.captions
                if c['end_time'] > current_time - buffer
            ]
            logger.debug(f"Pruned captions. Remaining: {len(self.captions)}")
    
    def set_video_start_time(self, start_time):
        """Set the video's start time to handle offset captions.
        
        Args:
            start_time: The absolute timestamp where the video starts (in seconds)
        """
        # Always update when explicitly called - this is the authoritative source
        old_time = getattr(self, 'video_start_time', 'not set')
        self.video_start_time = start_time
        logger.info(f"Video start time updated from {old_time} to {start_time} (absolute)")
        
        logger.debug(f"[TIMING] Video start time set to {start_time}. Current offset: {time.time() - start_time:.2f}s")
        logger.trace(f"[TIMING] Video start time details - System time: {time.time()}, Offset: {time.time() - start_time:.6f}s")
    
    def get_caption_count(self):
        """Get the current number of captions in the queue.
        
        Returns:
            int: Number of captions
        """
        return len(self.captions) 