"""Configuration for caption styling and appearance."""
from dataclasses import dataclass
from typing import Tuple

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