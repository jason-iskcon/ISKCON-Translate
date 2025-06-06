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
        
        # Color mapping for different languages
        self.language_colors = {
            'en': (255, 255, 255),     # White for English (primary)
            'it': (150, 255, 150),     # Pale green for Italian
            'fr': (150, 255, 255),     # Pale blue for French
            'es': (255, 150, 255),     # Pale magenta for Spanish
            'de': (255, 200, 150),     # Pale orange for German
            'pt': (200, 150, 255),     # Pale purple for Portuguese
            'ru': (255, 180, 180),     # Pale red for Russian
            'uk': (255, 255, 150),     # Pale yellow for Ukrainian
            'hu': (180, 255, 180),     # Different shade of pale green for Hungarian
            'ja': (255, 150, 200),     # Pale pink for Japanese
            'zh': (150, 200, 255),     # Light blue for Chinese
            'ar': (255, 180, 180),     # Light red for Arabic
        }
        
        logger.debug(f"CaptionRenderer initialized with style: {self.style}")
    
    def get_language_color(self, language: str = 'en') -> tuple:
        """Get the color for a specific language.
        
        Args:
            language: Language code (e.g., 'en', 'it', 'fr')
            
        Returns:
            BGR color tuple for the language
        """
        return self.language_colors.get(language, (255, 255, 255))  # Default to white
    
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
        
        # Longer fade durations for better readability - don't fade out too early
        fade_in_duration = min(0.05, caption_duration / 8)  # Very quick 50ms fade in
        fade_out_duration = min(0.2, caption_duration / 6)   # Longer 200ms fade out for better readability
        
        fade_factor = 1.0
        
        # Quick fade in
        if time_in_caption < fade_in_duration and fade_in_duration > 0:
            # Linear fade for predictable timing
            progress = time_in_caption / fade_in_duration
            fade_factor = progress
        
        # Quick fade out
        elif time_until_end < fade_out_duration and fade_out_duration > 0:
            # Linear fade for predictable timing
            progress = time_until_end / fade_out_duration
            fade_factor = progress
        
        # Ensure fade factor is within valid range with minimum visibility
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
    
    def calculate_text_position(self, frame_width, frame_height, max_text_width, total_text_height, caption_index=0, active_captions=None):
        """Calculate text and background positions with intelligent overlap prevention.
        
        Args:
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            max_text_width: Maximum width of text lines
            total_text_height: Total height of all text lines
            caption_index: Index for stacking multiple captions
            active_captions: List of other active captions to avoid overlaps
            
        Returns:
            tuple: (text_x, text_y, bg_x1, bg_y1, bg_x2, bg_y2)
        """
        # Base position - very close to bottom with minimal margin
        base_margin = 20  # Reduced from 30 to 20
        
        # Calculate background dimensions with padding
        padding = self.style.padding  # Use existing padding attribute
        
        bg_width = max_text_width + (2 * padding)
        bg_height = total_text_height + (2 * padding)
        
        # Calculate dynamic spacing based on caption height and number of lines
        line_count = max(1, total_text_height // 25)  # Estimate lines (25px per line)
        
        # Smart spacing calculation - minimal gaps for tight display
        if line_count > 1:
            # Multi-line captions need minimal extra space
            base_spacing = bg_height + 2  # Reduced from 8 to 2
        else:
            # Single line captions should be very tight
            base_spacing = bg_height + 1  # Reduced from 4 to 1
            
        # Calculate vertical position with intelligent spacing
        if caption_index == 0:
            # Primary caption at bottom
            bg_y2 = frame_height - base_margin
            bg_y1 = bg_y2 - bg_height
        else:
            # Secondary captions stacked above with dynamic spacing
            # Check if previous captions were multi-line to adjust spacing
            accumulated_height = base_margin + bg_height
            
            for i in range(caption_index):
                # Add spacing for each previous caption
                prev_spacing = base_spacing
                
                # If we have active caption info, use actual heights
                if active_captions and i < len(active_captions):
                    prev_caption = active_captions[i]
                    prev_lines = self.process_caption_text(prev_caption.get('text', ''))
                    if len(prev_lines) > 1:
                        # Previous caption was multi-line, add minimal extra space
                        prev_spacing += 2  # Reduced from 5 to 2
                
                accumulated_height += prev_spacing
            
            bg_y2 = frame_height - accumulated_height
            bg_y1 = bg_y2 - bg_height
            
            # Ensure caption doesn't go off the top of the screen
            if bg_y1 < 20:
                bg_y1 = 20
                bg_y2 = bg_y1 + bg_height
        
        # Horizontal centering
        bg_x1 = (frame_width - bg_width) // 2
        bg_x2 = bg_x1 + bg_width
        
        # Text position (top-left corner of text area)
        text_x = bg_x1 + padding
        text_y = bg_y1 + padding
        
        return text_x, text_y, bg_x1, bg_y1, bg_x2, bg_y2
    
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
    
    def render_text_line(self, frame, line, x, y, fade_factor, language='en'):
        """Render a single line of text with shadow.
        
        Args:
            frame: Video frame to render on
            line: Text line to render
            x: X position for text
            y: Y position for text
            fade_factor: Fade factor for text opacity
            language: Language code for color selection
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
        
        # Get language-specific color
        text_color = list(self.get_language_color(language))
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
    
    def render_caption(self, frame, caption, current_time, caption_index=0, language='en', all_active_captions=None):
        """Render a single caption on the frame.
        
        Args:
            frame: Video frame to render on
            caption: Caption dictionary to render
            current_time: Current relative time
            caption_index: Index for stacking multiple captions
            language: Language code for color selection
            all_active_captions: List of all active captions for intelligent positioning
            
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
                logger.warning(f"[RENDER_CAPTION] No display lines generated for caption: '{caption['text']}'")
                return frame
            
            # Calculate text dimensions
            line_heights, line_widths, total_text_height, max_text_width = self.calculate_text_dimensions(display_lines)
            
            # Calculate positions
            text_x, text_y, bg_x1, bg_y1, bg_x2, bg_y2 = self.calculate_text_position(
                frame_width, frame_height, max_text_width, total_text_height, caption_index, all_active_captions
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
            for line_idx, (line, h) in enumerate(zip(display_lines, line_heights)):
                if not line.strip():
                    y += int(h * 1.5)  # Add extra space for empty lines
                    continue
                
                # Get text size for this line to center it horizontally
                (w, _), _ = cv2.getTextSize(
                    line, self.font, self.style.font_scale, self.style.font_thickness
                )
                x = (frame_width - w) // 2  # Center each line horizontally
                
                # Render the text line with language-specific color
                self.render_text_line(frame, line, x, y, fade_factor, language)
                
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
        
        # Reduce debug logging frequency to improve performance
        should_log = len(active_captions) > 0 and (int(current_time * 10) % 10 == 0)  # Log once per second
        
        if should_log:
            logger.debug(f"[RENDER] Rendering {len(active_captions)} captions at time {current_time:.3f}s")
        
        # Sort captions by start time (oldest first) to maintain proper stacking order
        active_captions = sorted(active_captions, key=lambda x: x['start_time'])
        
        # Separate primary and secondary language captions for proper layering
        primary_captions = [c for c in active_captions if c.get('is_primary', True)]
        secondary_captions = [c for c in active_captions if not c.get('is_primary', True)]
        
        # Process each active caption
        rendered_count = 0
        
        # Render primary language captions first (bottom layer)
        for i, caption in enumerate(primary_captions):
            # Only render if fade factor is significant
            fade_factor = self.calculate_fade_factor(caption, current_time)
            
            if fade_factor > 0.05:  # Lowered threshold to match core logic
                language = caption.get('language', 'en')
                result_frame = self.render_caption(result_frame, caption, current_time, caption_index=i, language=language, all_active_captions=active_captions)
                rendered_count += 1
        
        # Render secondary language captions on top
        for i, caption in enumerate(secondary_captions):
            # Only render if fade factor is significant
            fade_factor = self.calculate_fade_factor(caption, current_time)
            
            if fade_factor > 0.05:
                language = caption.get('language', 'en')
                # Offset secondary captions above primary ones
                caption_index = len(primary_captions) + i
                result_frame = self.render_caption(result_frame, caption, current_time, caption_index=caption_index, language=language, all_active_captions=active_captions)
                rendered_count += 1
        
        if should_log and rendered_count > 0:
            languages_rendered = set(c.get('language', 'en') for c in active_captions if self.calculate_fade_factor(c, current_time) > 0.05)
            logger.debug(f"[RENDER] Rendered {rendered_count}/{len(active_captions)} captions in languages: {languages_rendered}")
        
        # Log rendering performance warnings only
        render_time = (time.time() - render_start) * 1000  # Convert to milliseconds
        if render_time > 16:  # Log warning if rendering takes more than 16ms (~60fps)
            logger.warning(f"Slow frame rendering: {render_time:.2f}ms")
        
        return result_frame 