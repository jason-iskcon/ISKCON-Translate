"""Side-by-side caption comparison display."""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

# Import with try-except to handle both direct execution and module import
try:
    from ..logging_utils import get_logger, TRACE
    from .renderer import CaptionRenderer
except ImportError:
    from src.logging_utils import get_logger, TRACE
    from .renderer import CaptionRenderer

logger = get_logger(__name__)

class ComparisonRenderer:
    """Handles side-by-side rendering of YouTube and Parakletos captions."""
    
    def __init__(self, style_config=None):
        """Initialize the comparison renderer.
        
        Args:
            style_config: Optional style configuration
        """
        self.renderer = CaptionRenderer(style_config)
        logger.debug("ComparisonRenderer initialized")
    
    def render_comparison(self, frame, youtube_caption: Dict[str, Any], parakletos_caption: Dict[str, Any], current_time: float) -> np.ndarray:
        """Render YouTube and Parakletos captions side by side.
        
        Args:
            frame: Video frame to render on
            youtube_caption: YouTube caption dict
            parakletos_caption: Parakletos caption dict
            current_time: Current relative time
            
        Returns:
            Frame with both captions rendered
        """
        try:
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Split frame into left and right halves
            left_frame = frame[:, :frame_width//2].copy()
            right_frame = frame[:, frame_width//2:].copy()
            
            # Render YouTube caption on left
            if youtube_caption:
                left_frame = self.renderer.render_caption(
                    left_frame,
                    youtube_caption,
                    current_time,
                    caption_index=0
                )
            
            # Render Parakletos caption on right
            if parakletos_caption:
                right_frame = self.renderer.render_caption(
                    right_frame,
                    parakletos_caption,
                    current_time,
                    caption_index=0
                )
            
            # Add divider line
            divider_x = frame_width // 2
            cv2.line(
                frame,
                (divider_x, 0),
                (divider_x, frame_height),
                (255, 255, 255),  # White line
                2  # Thickness
            )
            
            # Add labels
            label_y = 30
            cv2.putText(
                frame,
                "YouTube Captions",
                (10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),  # White text
                2,
                cv2.LINE_AA
            )
            
            cv2.putText(
                frame,
                "Parakletos",
                (divider_x + 10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),  # White text
                2,
                cv2.LINE_AA
            )
            
            # Combine frames
            frame[:, :frame_width//2] = left_frame
            frame[:, frame_width//2:] = right_frame
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering comparison: {e}", exc_info=True)
            return frame 