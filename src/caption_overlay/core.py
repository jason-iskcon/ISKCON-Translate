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
    """Core state management for captions."""
    
    def __init__(self):
        """Initialize the caption core state."""
        self.captions = []  # List of dicts with text, start_time, end_time, and is_absolute flag
        self.video_start_time = time.time()  # Default to current time
        self.lock = threading.Lock()  # For thread-safe access to captions list
        logger.debug(f"CaptionCore initialized with start time: {self.video_start_time}")
    
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
    
    def add_caption(self, text, timestamp, duration=3.0, is_absolute=False, seamless=True):
        """Add a caption to be displayed.
        
        Args:
            text: Caption text to display
            timestamp: Timestamp for when to show the caption (relative to video start)
            duration: How long to display the caption in seconds
            is_absolute: If True, timestamp is treated as absolute system time
            seamless: If True, will try to merge with previous caption if similar
            
        Returns:
            dict: The added caption or None if skipped
        """
        with self.lock:
            # Store original values for logging
            original_timestamp = timestamp
            current_time = time.time()
            
            # Calculate current relative time
            current_relative_time = current_time - self.video_start_time
            
            # Convert timestamp using utility function ONLY if absolute
            if is_absolute:
                timestamp, was_converted = convert_timestamp(timestamp, self.video_start_time, is_absolute)
                logger.debug(f"[CAPTION] Converted absolute timestamp: {original_timestamp:.2f}s -> {timestamp:.2f}s")
            else:
                # For relative timestamps, use directly without conversion
                logger.trace(f"[CAPTION] Using relative timestamp directly: {timestamp:.2f}s")
                was_converted = False
            
            # Adjust timestamp if in the past
            timestamp = adjust_timestamp_if_past(timestamp, current_relative_time)
            
            # Create caption entry
            caption = {
                'text': text,
                'start_time': timestamp,
                'end_time': timestamp + duration,
                'added_at': time.time(),
                'was_absolute': is_absolute,
                'original_timestamp': original_timestamp,
                'display_count': 0  # Track how many times this caption has been displayed
            }
            
            # Add the new caption
            self.captions.append(caption)
            logger.info(f"[CAPTION] Added caption #{len(self.captions)}: '{text}' ({timestamp:.2f}s-{timestamp+duration:.2f}s)")
            
            return caption
    
    def prune_captions(self, current_time, buffer=1.0):
        """Remove captions whose end_time is more than `buffer` seconds before current_time.
        Also removes any captions that are too far in the future.
        
        Args:
            current_time: Current relative time from video start
            buffer: Buffer time in seconds for keeping expired captions
        """
        if not hasattr(self, 'last_prune_time') or (current_time - self.last_prune_time) > 1.0:
            before = len(self.captions)
            max_future_offset = 30  # 30 seconds in the future max
            
            removed_captions = [
                c for c in self.captions 
                if not ((c['end_time'] >= current_time - buffer) and 
                      (c['start_time'] <= current_time + max_future_offset))
            ]
            
            # Log removed captions
            for c in removed_captions:
                logger.debug(f"[CAPTION] Pruning caption (expired): '{c['text'][:50]}{'...' if len(c['text']) > 50 else ''}' "
                           f"(displayed {c.get('display_count', 0)} times, duration: {c['end_time']-c['start_time']:.1f}s)")
            
            # Keep only active captions
            self.captions = [
                c for c in self.captions 
                if (c['end_time'] >= current_time - buffer) and 
                   (c['start_time'] <= current_time + max_future_offset)
            ]
            
            # Also remove any captions that have been displayed for too long
            max_display_time = 10.0  # Maximum time a caption can be displayed (in seconds)
            self.captions = [
                c for c in self.captions 
                if (time.time() - c.get('added_at', 0)) < max_display_time
            ]
            
            after = len(self.captions)
            if before != after:
                logger.info(f"[CAPTION] Pruned {before-after} old/future captions at {current_time:.2f}s")
                
            self.last_prune_time = current_time
    
    def get_active_captions(self, current_time):
        """Get all captions that should be active at the current time.
        
        Args:
            current_time: Current relative time from video start
            
        Returns:
            list: List of active caption dictionaries
        """
        with self.lock:
            # First, remove any captions that have already ended (with a small buffer)
            before_prune = len(self.captions)
            self.captions = [c for c in self.captions if c['end_time'] > current_time - 1.0]
            after_prune = len(self.captions)
            
            if before_prune != after_prune:
                logger.debug(
                    f"[OVERLAY] Pruned {before_prune - after_prune} old captions | "
                    f"Current time: {current_time:.2f}s | "
                    f"Kept captions with end_time > {current_time - 1.0:.2f}s"
                )
            
            # Find all captions that should be active now (with small buffer for smooth transitions)
            active_captions = []
            
            for i, c in enumerate(self.captions):
                # Add small buffer for activation/deactivation to prevent flickering
                is_active = (c['start_time'] - 0.1) <= current_time <= (c['end_time'] + 0.1)
                
                if is_active:
                    # Update display count and log first display
                    if 'display_count' not in c:
                        c['display_count'] = 0
                    if c['display_count'] == 0:
                        logger.debug(
                            f"[CAPTION] Displaying: '{c['text'][:50]}{'...' if len(c['text']) > 50 else ''}' | "
                            f"At: {current_time:.2f}s | "
                            f"Duration: {c['end_time']-c['start_time']:.1f}s"
                        )
                    c['display_count'] += 1
                    
                    # Add trace logging for active caption timing
                    if c['display_count'] == 1:  # Only log first frame to reduce noise
                        logger.trace(
                            f"[TIMING] First frame timing - "
                            f"Caption: '{c['text'][:30]}...' | "
                            f"Start: {c['start_time']:.6f}s | "
                            f"Current: {current_time:.6f}s | "
                            f"End: {c['end_time']:.6f}s | "
                            f"Time in: {current_time - c['start_time']:.6f}s"
                        )
                    
                    active_captions.append(c)
            
            return sorted(active_captions, key=lambda x: x['start_time'])
    
    def get_caption_count(self):
        """Get the current number of captions in the queue.
        
        Returns:
            int: Number of captions
        """
        return len(self.captions)
    
    def clear_captions(self):
        """Clear all captions from the queue."""
        with self.lock:
            count = len(self.captions)
            self.captions.clear()
            logger.info(f"[CAPTION] Cleared {count} captions from queue") 