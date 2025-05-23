#!/usr/bin/env python3
"""Singleton playback clock for consistent A/V synchronization."""

import time
import logging

logger = logging.getLogger(__name__)

class PlaybackClock:
    """Singleton clock for coordinating timing across video and audio components."""
    
    def __init__(self):
        self.media_seek_pts = 0.0      # PTS where playback started (e.g., 325.0s)
        self.start_wall_time = None    # Wall-clock time when playback started
        self.video_source_created = False  # Track if VideoSource has been created
        
    def initialize(self, seek_pts: float):
        """Initialize the clock with media seek position.
        
        Args:
            seek_pts: Media timestamp where playback starts (in seconds)
        """
        if not self.video_source_created or self.media_seek_pts == 0.0:
            self.media_seek_pts = seek_pts
            self.video_source_created = True
            logger.info(f"🔧 Singleton clock initialized: media_seek_pts={seek_pts:.2f}s")
        else:
            logger.warning(f"🔧 Attempted to re-initialize singleton clock (seek_pts={seek_pts:.2f}s), ignoring")
        
    def get_video_relative_time(self) -> float:
        """Get current video time relative to seek position.
        
        Returns:
            float: Seconds elapsed since media seek position
        """
        if self.start_wall_time is None:
            return 0.0
        return time.time() - self.start_wall_time
        
    def get_elapsed_time(self) -> float:
        """Get elapsed wall-clock time since playback started.
        
        Returns:
            float: Seconds elapsed since playback start
        """
        if self.start_wall_time is None:
            return 0.0
        return time.time() - self.start_wall_time
        
    def rel_audio_time(self, abs_pts: float) -> float:
        """Convert absolute audio PTS to seek-relative time.
        
        Args:
            abs_pts: Absolute media timestamp (e.g., 359.5s)
            
        Returns:
            float: Time relative to seek position (e.g., 34.5s if seek_pts=325.0s)
        """
        return abs_pts - self.media_seek_pts
        
    def is_initialized(self) -> bool:
        """Check if the clock has been properly initialized.
        
        Returns:
            bool: True if both seek_pts and start_wall_time are set
        """
        return self.video_source_created and self.start_wall_time is not None
        
    def reset(self):
        """Reset clock for new session."""
        self.media_seek_pts = 0.0
        self.start_wall_time = None
        self.video_source_created = False

# Global singleton instance
CLOCK = PlaybackClock() 