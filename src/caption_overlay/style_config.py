"""Configuration for caption styling and appearance."""
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class CaptionStyleConfig:
    """Configuration class for caption styling and appearance.
    
    This encapsulates all the visual styling options for captions,
    making it easier to manage and modify caption appearance.
    """
    font_scale: float = 1.0
    font_thickness: int = 2
    font_color: Tuple[int, int, int] = (255, 255, 255)  # White text (BGR)
    bg_color: Tuple[int, int, int] = (0, 0, 0)  # Black background (BGR)  
    padding: int = 10
    y_offset: int = 50
    
    # Video dimension properties for responsive sizing
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    base_video_width: int = 1920  # Reference resolution for font scaling
    base_video_height: int = 1080
    
    def __post_init__(self):
        """Validate configuration values after initialization."""
        if self.font_scale <= 0:
            raise ValueError("font_scale must be positive")
        if self.font_thickness <= 0:
            raise ValueError("font_thickness must be positive")
        if self.padding < 0:
            raise ValueError("padding must be non-negative")
        if self.y_offset < 0:
            raise ValueError("y_offset must be non-negative")
    
    def set_video_dimensions(self, width: int, height: int):
        """Set the video dimensions for responsive font scaling.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
        """
        self.video_width = width
        self.video_height = height
    
    def get_scaled_font_size(self, base_font_size: int = 30) -> int:
        """Calculate font size scaled based on video dimensions.
        
        Args:
            base_font_size: Base font size at reference resolution
            
        Returns:
            int: Scaled font size appropriate for current video dimensions
        """
        if self.video_width is None or self.video_height is None:
            # No video dimensions set, use original scaling
            return max(16, int(self.font_scale * base_font_size))
        
        # Calculate scaling factor based on video dimensions
        # Use area-based scaling for better proportional results
        base_area = self.base_video_width * self.base_video_height
        current_area = self.video_width * self.video_height
        area_scale = (current_area / base_area) ** 0.5  # Square root for linear scaling
        
        # Apply both user font_scale and video dimension scaling
        scaled_size = int(self.font_scale * base_font_size * area_scale)
        
        # Ensure minimum readable size and reasonable maximum
        return max(12, min(scaled_size, base_font_size * 2))
    
    def get_scaled_padding(self) -> int:
        """Calculate padding scaled based on video dimensions.
        
        Returns:
            int: Scaled padding appropriate for current video dimensions
        """
        if self.video_width is None or self.video_height is None:
            return self.padding
        
        # Scale padding proportionally to video size
        width_scale = self.video_width / self.base_video_width
        scaled_padding = int(self.padding * width_scale)
        
        # Ensure minimum padding
        return max(5, scaled_padding) 