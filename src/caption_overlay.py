import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CaptionOverlay:
    def __init__(self, font_scale=0.6):
        """Bare minimum caption overlay."""
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = font_scale
        self.active_caption = None
        self.caption_timestamp = 0
        self.caption_duration = 5.0  # Show each caption for 5 seconds
        
    # Context manager methods for resource management
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # No resources to clean up
            
    def add_caption(self, text, timestamp=0.0, language="english"):
        """Add a new caption with timestamp."""
        if not text or not text.strip():
            return
            
        self.active_caption = text
        self.caption_timestamp = timestamp
        logger.info(f"Added caption: '{text}' at {timestamp:.2f}s")
        
    def overlay_captions(self, frame, current_time=None):
        """Overlay caption on frame."""
        if not self.active_caption:
            return frame
            
        # Check if caption should be shown based on time
        if current_time is not None:
            if current_time < self.caption_timestamp or current_time > self.caption_timestamp + self.caption_duration:
                return frame
                
        # Simple implementation - just draw text at bottom of frame
        result_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate text size and position
        (text_width, text_height), _ = cv2.getTextSize(
            self.active_caption, 
            self.font, 
            self.font_scale, 
            1  # Fixed thickness
        )
        
        # Position at bottom of frame
        x = (frame_width - text_width) // 2
        y = frame_height - 50
        
        # Draw simple black background rectangle
        cv2.rectangle(
            result_frame, 
            (x - 10, y - text_height - 10), 
            (x + text_width + 10, y + 10), 
            (0, 0, 0), 
            -1
        )
        
        # Draw caption text
        cv2.putText(
            result_frame,
            self.active_caption,
            (x, y),
            self.font,
            self.font_scale,
            (255, 255, 255),
            1,  # Fixed thickness
            cv2.LINE_AA
        )
        return result_frame
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
