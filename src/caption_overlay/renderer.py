"""Frame rendering logic for caption overlay."""
import cv2
import numpy as np
import time
import logging

# Import with try-except to handle both direct execution and module import
try:
    from ..logging_utils import get_logger, TRACE
    from .utils import deduplicate_lines, wrap_text_lines
    from .style_config import CaptionStyleConfig
except ImportError:
    from src.logging_utils import get_logger, TRACE
    from .utils import deduplicate_lines, wrap_text_lines
    from .style_config import CaptionStyleConfig

logger = get_logger(__name__)

class CaptionRenderer:
    """Handles the visual rendering of captions on video frames."""
    
    def __init__(self, style_config=None):
        """Initialize the caption renderer.
        
        Args:
            style_config: CaptionStyleConfig instance or None for defaults
        """
        self.style = style_config or CaptionStyleConfig()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        logger.debug(f"CaptionRenderer initialized with style: {self.style}")
    
    def calculate_fade_factor(self, caption, current_time):
        """Calculate fade factor for smooth caption transitions.
        
        Args:
            caption: Caption dictionary with start_time and end_time
            current_time: Current relative time
            
        Returns:
            float: Fade factor between 0.1 and 1.0
        """
        caption_start = caption.get('start_time', 0)
        caption_end = caption.get('end_time', 0)
        caption_duration = caption_end - caption_start
        
        time_in_caption = current_time - caption_start
        time_until_end = caption_end - current_time
        
        # Calculate fade in/out effects - shorter durations for more precise timing
        fade_in_duration = min(0.15, caption_duration / 4)  # Reduced from 0.3s to 0.15s
        fade_out_duration = min(0.2, caption_duration / 4)  # Reduced from 0.5s to 0.2s
        
        fade_factor = 1.0
        
        # Fade in at start with smoother transition
        if time_in_caption < fade_in_duration and fade_in_duration > 0:
            # Use quadratic easing for smoother fade in
            progress = time_in_caption / fade_in_duration
            fade_factor = progress * progress  # Quadratic easing
        
        # Fade out at end with smoother transition
        elif time_until_end < fade_out_duration and fade_out_duration > 0:
            # Use quadratic easing for smoother fade out
            progress = time_until_end / fade_out_duration
            fade_factor = progress * progress  # Quadratic easing
        
        # Ensure fade factor is within valid range
        return max(0.1, min(1.0, fade_factor))
    
    def process_caption_text(self, caption_text):
        """Process caption text for optimal display.
        
        Args:
            caption_text: Raw caption text
            
        Returns:
            list: List of processed text lines ready for display
        """
        # Split the current caption text into lines and clean them up
        lines = [line.strip() for line in caption_text.split('\n') if line.strip()]
        
        # Remove duplicate lines while preserving order
        display_lines = deduplicate_lines(lines)
        
        # If there are no newlines in the original text and it's a single line, wrap long lines
        if len(display_lines) == 1 and '\n' not in caption_text:
            display_lines = wrap_text_lines(display_lines[0], max_chars_per_line=60)
        
        return display_lines
    
    def calculate_text_dimensions(self, display_lines):
        """Calculate dimensions for text block.
        
        Args:
            display_lines: List of text lines to measure
            
        Returns:
            tuple: (line_heights, line_widths, total_height, max_width)
        """
        line_heights = []
        line_widths = []
        
        for line in display_lines:
            (w, h), _ = cv2.getTextSize(
                line, self.font, self.style.font_scale, self.style.font_thickness
            )
            line_heights.append(h)
            line_widths.append(w)
        
        total_text_height = sum(line_heights) + (len(display_lines) - 1) * (self.style.font_thickness + 2)
        max_text_width = max(line_widths) if line_widths else 0
        
        return line_heights, line_widths, total_text_height, max_text_width
    
    def calculate_text_position(self, frame_width, frame_height, max_text_width, total_text_height, caption_index=0):
        """Calculate position for text block on frame.
        
        Args:
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            max_text_width: Maximum width of text block
            total_text_height: Total height of text block
            caption_index: Index for stacking multiple captions
            
        Returns:
            tuple: (text_x, text_y, bg_x1, bg_y1, bg_x2, bg_y2)
        """
        # Calculate text block position (centered at bottom with vertical offset for multiple captions)
        vertical_offset = caption_index * (total_text_height + 20)  # 20px spacing between captions
        
        # Center the text block horizontally and vertically within its background
        bg_padding = self.style.padding
        text_block_x = (frame_width - max_text_width) // 2
        text_block_y = frame_height - self.style.y_offset - total_text_height - vertical_offset
        
        # Calculate background rectangle first
        bg_x1 = text_block_x - bg_padding
        bg_y1 = text_block_y - bg_padding
        bg_x2 = text_block_x + max_text_width + bg_padding
        bg_y2 = text_block_y + total_text_height + bg_padding
        
        # Ensure background stays within frame bounds
        bg_x1 = max(0, min(bg_x1, frame_width - 1))
        bg_y1 = max(0, min(bg_y1, frame_height - 1))
        bg_x2 = max(0, min(bg_x2, frame_width - 1))
        bg_y2 = max(0, min(bg_y2, frame_height - 1))
        
        # Recalculate text position to be centered within the background
        text_block_x = (bg_x1 + bg_x2 - max_text_width) // 2
        text_block_y = (bg_y1 + bg_y2 - total_text_height) // 2
        
        return text_block_x, text_block_y, bg_x1, bg_y1, bg_x2, bg_y2
    
    def render_background(self, frame, bg_coords, fade_factor):
        """Render background rectangle for caption.
        
        Args:
            frame: Video frame to render on
            bg_coords: Tuple of (x1, y1, x2, y2) for background rectangle
            fade_factor: Fade factor for transparency
            
        Returns:
            numpy.ndarray: Frame with background rendered
        """
        bg_x1, bg_y1, bg_x2, bg_y2 = bg_coords
        
        # Create a semi-transparent background with fade effect
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            self.style.bg_color,
            -1
        )
        
        # Apply fade effect to the background
        alpha = 0.7 * fade_factor  # Base opacity * fade factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def render_text_line(self, frame, line, x, y, fade_factor):
        """Render a single line of text with shadow.
        
        Args:
            frame: Video frame to render on
            line: Text line to render
            x: X position for text
            y: Y position for text
            fade_factor: Fade factor for text opacity
        """
        if not line.strip():
            return
        
        # Draw text shadow (slightly offset)
        shadow_offset = 2
        shadow_color = (0, 0, 0)  # Black shadow
        cv2.putText(
            frame, 
            line, 
            (x + shadow_offset, y + shadow_offset),
            self.font, 
            self.style.font_scale, 
            shadow_color,
            self.style.font_thickness + 1, 
            cv2.LINE_AA
        )
        
        # Draw main text with fade effect applied
        text_color = list(self.style.font_color)
        if len(text_color) == 3:  # If no alpha channel, add one
            text_color = text_color + [255]  # Fully opaque by default
        
        # Apply fade factor to alpha channel
        text_color[3] = int(text_color[3] * fade_factor)
        
        # Convert to BGR for OpenCV
        bgr_color = tuple(text_color[2::-1])  # Convert RGB to BGR and remove alpha
        
        # Draw the text
        cv2.putText(
            frame, 
            line, 
            (x, y),
            self.font, 
            self.style.font_scale, 
            bgr_color,
            self.style.font_thickness, 
            cv2.LINE_AA
        )
    
    def render_caption(self, frame, caption, current_time, caption_index=0):
        """Render a single caption on the frame.
        
        Args:
            frame: Video frame to render on
            caption: Caption dictionary to render
            current_time: Current relative time
            caption_index: Index for stacking multiple captions
            
        Returns:
            numpy.ndarray: Frame with caption rendered
        """
        try:
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate fade factor
            fade_factor = self.calculate_fade_factor(caption, current_time)
            
            # Process text into display lines
            display_lines = self.process_caption_text(caption['text'])
            
            if not display_lines:
                return frame
            
            # Calculate text dimensions
            line_heights, line_widths, total_text_height, max_text_width = self.calculate_text_dimensions(display_lines)
            
            # Calculate positions
            text_x, text_y, bg_x1, bg_y1, bg_x2, bg_y2 = self.calculate_text_position(
                frame_width, frame_height, max_text_width, total_text_height, caption_index
            )
            
            # Render background
            frame = self.render_background(frame, (bg_x1, bg_y1, bg_x2, bg_y2), fade_factor)
            
            # Calculate the vertical center of the background
            bg_center_y = (bg_y1 + bg_y2) // 2
            
            # Calculate starting Y position to center all text lines within the background
            total_line_spacing = (len(display_lines) - 1) * 5  # 5px spacing between lines
            actual_text_height = sum(line_heights) + total_line_spacing
            start_y = bg_center_y - (actual_text_height // 2) + line_heights[0]  # Add first line height for baseline
            
            # Render each line of text
            y = start_y
            for line, h in zip(display_lines, line_heights):
                if not line.strip():
                    y += int(h * 1.5)  # Add extra space for empty lines
                    continue
                
                # Get text size for this line to center it horizontally
                (w, _), _ = cv2.getTextSize(
                    line, self.font, self.style.font_scale, self.style.font_thickness
                )
                x = (frame_width - w) // 2  # Center each line horizontally
                
                # Render the text line
                self.render_text_line(frame, line, x, y, fade_factor)
                
                # Move to next line position with spacing
                y += h + 5
            
            return frame
            
        except Exception as e:
            logger.error(f"Error rendering caption: {str(e)}", exc_info=True)
            return frame
    
    def render_multiple_captions(self, frame, active_captions, current_time):
        """Render multiple captions on the frame.
        
        Args:
            frame: Video frame to render on
            active_captions: List of active caption dictionaries
            current_time: Current relative time
            
        Returns:
            numpy.ndarray: Frame with all captions rendered
        """
        render_start = time.time()
        result_frame = frame.copy()
        
        # Sort captions by start time (most recent first)
        active_captions = sorted(active_captions, key=lambda x: x['start_time'], reverse=True)
        
        # Process each active caption
        for i, caption in enumerate(active_captions):
            # Only render if fade factor is significant
            fade_factor = self.calculate_fade_factor(caption, current_time)
            if fade_factor > 0.1:  # Only render if not too faded
                result_frame = self.render_caption(result_frame, caption, current_time, caption_index=i)
        
        # Log rendering performance
        render_time = (time.time() - render_start) * 1000  # Convert to milliseconds
        if render_time > 16:  # Log warning if rendering takes more than 16ms (~60fps)
            logger.warning(f"Slow frame rendering: {render_time:.2f}ms")
        elif len(active_captions) > 0:
            logger.trace(f"Rendered {len(active_captions)} captions in {render_time:.2f}ms")
        
        return result_frame 