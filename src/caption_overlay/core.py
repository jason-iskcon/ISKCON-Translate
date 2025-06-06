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
        self.timing_buffer = 0.033  # 33ms (1 frame at 30fps) for good sync without breaking tests
        self._last_stats_log_time = 0.0
        self._caption_intervals = []
        self._last_caption_time = 0.0
        self.video_start_time = 0.0  # Initialize video start time
        
    def add_caption(self, text: str, timestamp: float, duration: float = 3.0, is_absolute: bool = False, language: str = 'en', is_primary: bool = True) -> dict:
        """Add a new caption with precise timing.
        
        Args:
            text: Caption text
            timestamp: Start time in seconds
            duration: Display duration in seconds (default 3.0s for better readability)
            is_absolute: Whether timestamp is absolute time
            language: Language code for the caption (e.g., 'en', 'fr', 'it')
            is_primary: Whether this is a primary language caption
        
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
            
            # Add caption with precise timing and language information
            caption_id = len(self.captions) + 1
            caption = {
                'id': caption_id,
                'text': text,
                'start_time': timestamp,
                'end_time': end_time,
                'duration': duration,
                'added_at': time.time(),
                'original_timestamp': original_timestamp,
                'was_absolute': is_absolute,
                'language': language,
                'is_primary': is_primary
            }
            
            self.captions.append(caption)
            
            # Log caption addition with precise timing (less frequently for performance)
            if len(self.captions) % 5 == 0:  # Only log every 5th caption
                logger.info(f"[CAPTION] Added caption #{caption_id} [{language}]: '{text}' ({timestamp:.2f}s-{end_time:.2f}s)")
            
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
            # Small timing buffer for good sync without breaking tests, with earlier appearance
            timing_buffer = 0.033  # 33ms (1 frame at 30fps)
            early_appearance_buffer = 0.100  # 100ms early appearance for better UX
            
            # Calculate buffer times with earlier appearance for better timing
            start_buffer = current_time - timing_buffer
            end_buffer = current_time + early_appearance_buffer  # Increased for earlier appearance
            
            # Find active captions with precise timing
            active_captions = []
            for c in self.captions:  # Process in chronological order
                # Check if caption is active within buffer
                if (c['start_time'] <= end_buffer and 
                    c['end_time'] >= start_buffer):
                    # Calculate fade factor for smooth but visible transitions
                    # Use smaller fade duration to ensure captions are visible at edges
                    fade_duration = 0.1  # 100ms fade duration
                    
                    fade_start = max(0, min(1, (current_time - c['start_time']) / fade_duration))
                    fade_end = max(0, min(1, (c['end_time'] - current_time) / fade_duration))
                    fade_factor = min(fade_start, fade_end)
                    
                    # Ensure captions are always visible when within their time range
                    if c['start_time'] <= current_time <= c['end_time']:
                        fade_factor = max(0.1, fade_factor)  # Minimum 10% opacity when active
                    
                    # Only include if fade factor is significant
                    if fade_factor > 0.05:  # Lowered threshold to prevent floating point edge cases
                        c['fade_factor'] = fade_factor
                        active_captions.append(c)
            
            # Sort by start time to maintain proper order
            active_captions.sort(key=lambda x: x['start_time'])
            
            # Log timing stats much less frequently (every 10 seconds instead of 3 seconds)
            current_sys_time = time.time()
            if current_sys_time - self._last_stats_log_time >= 10.0:  # Log every 10 seconds
                self._last_stats_log_time = current_sys_time
                self._log_timing_stats(current_time, active_captions)
            
            return active_captions
    
    def _log_timing_stats(self, current_time: float, active_captions: List[Dict[str, Any]]) -> None:
        """Log timing statistics for debugging.
        
        Args:
            current_time: Current time in seconds
            active_captions: List of currently active captions
        """
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
    
    def clear_captions(self) -> None:
        """Clear all captions."""
        with self.caption_lock:
            self.captions.clear()
            logger.info("[CAPTION] Cleared all captions")
    
    def remove_caption(self, caption_id: int) -> bool:
        """Remove a specific caption by ID.
        
        Args:
            caption_id: ID of the caption to remove
            
        Returns:
            bool: True if caption was found and removed, False otherwise
        """
        with self.caption_lock:
            for i, caption in enumerate(self.captions):
                if caption.get('id') == caption_id:
                    removed_caption = self.captions.pop(i)
                    logger.debug(f"[CAPTION] Removed caption #{caption_id}: '{removed_caption.get('text', '')[:30]}...'")
                    return True
            return False
    
    def remove_overlapping_captions(self, start_time: float, end_time: float) -> int:
        """Remove captions that overlap with the given time range.
        
        Args:
            start_time: Start time of the range
            end_time: End time of the range
            
        Returns:
            int: Number of captions removed
        """
        with self.caption_lock:
            original_count = len(self.captions)
            
            # Keep captions that don't overlap with the given range
            self.captions = [
                c for c in self.captions
                if not (c['start_time'] < end_time and c['end_time'] > start_time)
            ]
            
            removed_count = original_count - len(self.captions)
            if removed_count > 0:
                logger.debug(f"[CAPTION] Removed {removed_count} overlapping captions in range {start_time:.2f}-{end_time:.2f}s")
            
            return removed_count
    
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