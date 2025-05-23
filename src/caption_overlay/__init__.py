"""
ISKCON-Translate Caption Overlay Module

This module provides caption overlay functionality for video frames.
It has been decomposed into modular components for better maintainability:

- style_config.py: Configuration dataclass for styling
- utils.py: Text processing and utility functions  
- core.py: Core state management (add_caption, set_video_start_time, prune_captions)
- renderer.py: Frame rendering logic with fade effects and text layout
- overlay.py: Orchestration logic that coordinates core and rendering
- __init__.py: Main CaptionOverlay class for backward compatibility

Usage:
    from caption_overlay import CaptionOverlay
    
    overlay = CaptionOverlay(font_scale=1.5, font_color=(255, 255, 255))
    overlay.add_caption("Hello World", timestamp=0.0, duration=3.0)
    frame_with_captions = overlay.overlay_captions(frame, current_time=1.0)
"""

import cv2
import threading

# Import with try-except to handle both direct execution and module import
try:
    from ..logging_utils import get_logger
    from .style_config import CaptionStyleConfig
    from .overlay import CaptionOverlayOrchestrator
except ImportError:
    from src.logging_utils import get_logger
    from .style_config import CaptionStyleConfig
    from .overlay import CaptionOverlayOrchestrator

logger = get_logger(__name__)

class CaptionOverlay:
    """Main caption overlay class that provides the same interface as the original monolithic version.
    
    This class acts as a facade that delegates to the modular components internally,
    ensuring backward compatibility while providing improved maintainability.
    """
    
    def __init__(self, font_scale=1.0, font_thickness=2, font_color=(255, 255, 255),
                 bg_color=(0, 0, 0), padding=10, y_offset=50):
        """Initialize the caption overlay.
        
        Args:
            font_scale: Font size scale factor
            font_thickness: Thickness of the font
            font_color: Text color as BGR tuple
            bg_color: Background color as BGR tuple
            padding: Padding around text in pixels
            y_offset: Distance from bottom of frame in pixels
        """
        logger.debug("Initializing CaptionOverlay")
        logger.trace(f"Font scale: {font_scale}, Thickness: {font_thickness}")
        logger.trace(f"Text color: {font_color}, BG color: {bg_color}")
        logger.trace(f"Padding: {padding}px, Y-Offset: {y_offset}px")
        
        # Create style configuration
        self.style_config = CaptionStyleConfig(
            font_scale=font_scale,
            font_thickness=font_thickness,
            font_color=font_color,
            bg_color=bg_color,
            padding=padding,
            y_offset=y_offset
        )
        
        # Initialize the orchestrator with the style config
        self.orchestrator = CaptionOverlayOrchestrator(self.style_config)
        
        # Store individual style properties for backward compatibility
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font_color = font_color
        self.bg_color = bg_color
        self.padding = padding
        self.y_offset = y_offset
        
        logger.debug(f"CaptionOverlay initialized with start time: {self.orchestrator.core.video_start_time}")
    
    # Delegate core functionality to the orchestrator
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
        return self.orchestrator.add_caption(text, timestamp, duration, is_absolute, seamless)
    
    def set_video_start_time(self, start_time):
        """Set the video's start time to handle offset captions.
        
        Args:
            start_time: The absolute timestamp where the video starts (in seconds)
        """
        return self.orchestrator.set_video_start_time(start_time)
    
    def prune_captions(self, current_time, buffer=1.0):
        """Remove captions whose end_time is more than `buffer` seconds before current_time.
        Also removes any captions that are too far in the future.
        
        Args:
            current_time: Current relative time from video start
            buffer: Buffer time in seconds for keeping expired captions
        """
        return self.orchestrator.prune_captions(current_time, buffer)
    
    def overlay_captions(self, frame, current_time=None, frame_count=0):
        """Overlay all valid captions on frame.
        
        Args:
            frame: The frame to overlay captions on
            current_time: The current timestamp in seconds (relative to video start)
            frame_count: The current frame number (used for logging)
            
        Returns:
            Frame with captions drawn
        """
        return self.orchestrator.overlay_captions(frame, current_time, frame_count)
    
    # Properties for backward compatibility
    @property
    def captions(self):
        """Access to the captions list for backward compatibility."""
        return self.orchestrator.captions
    
    @property
    def video_start_time(self):
        """Access to video start time for backward compatibility."""
        return self.orchestrator.video_start_time
    
    @property
    def lock(self):
        """Access to the threading lock for backward compatibility."""
        return self.orchestrator.lock
    
    # Context manager methods for resource management
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass  # No resources to clean up

# Export the main class and configuration for easy importing
__all__ = ['CaptionOverlay', 'CaptionStyleConfig'] 