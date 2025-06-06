"""Frame rendering logic for caption overlay."""
import cv2
import numpy as np
import time
import logging
from PIL import Image, ImageDraw, ImageFont
import os

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
        
        # Color mapping for different languages - BGR format for OpenCV
        self.language_colors = {
            'en': (255, 255, 255),     # White for English (primary) - BGR
            'fr': (255, 200, 150),     # Pale blue for French - BGR
            'de': (0, 255, 255),       # Yellow for German - BGR (requested by user)
            'it': (0, 165, 255),       # Orange for Italian - BGR
            'hu': (0, 255, 0),         # Green for Hungarian - BGR
            'ru': (203, 192, 255),     # Pale pink for Russian (Cyrillic) - BGR
            'uk': (255, 0, 255),       # Magenta for Ukrainian (Cyrillic) - BGR
        }
        
        # SAFEGUARD: Yellow is now allowed for German specifically
        # Remove the old forbidden colors that conflict with our supported languages
        self.forbidden_colors = [
            (255, 255, 0),   # Pure yellow (different from German's yellow)
            (255, 255, 100), # Light yellow variants that could conflict
            # Note: (255, 255, 150) removed since it was causing issues
            (200, 255, 100),
            (150, 255, 100),
        ]
        
        logger.debug(f"CaptionRenderer initialized with style: {self.style}")
    
    def get_language_color(self, language: str = 'en') -> tuple:
        """Get the color for a specific language.
        
        Args:
            language: Language code (e.g., 'en', 'fr', 'de', 'it', 'hu', 'ru', 'uk')
            
        Returns:
            BGR color tuple for the language
        """
        # Support all configured languages, default everything else to white
        if language in self.language_colors:
            color = self.language_colors[language]
            
            # SAFEGUARD: Check for forbidden yellow colors (but allow German's specific yellow)
            if color in self.forbidden_colors:
                logger.error(f"FORBIDDEN COLOR DETECTED for language '{language}': {color}. Forcing to white!")
                color = (255, 255, 255)  # Force to white
            
            # DEBUG: Log all color assignments
            logger.debug(f"Color assignment: '{language}' -> {color}")
            return color
        else:
            # Log unsupported language and default to white
            logger.warning(f"UNSUPPORTED LANGUAGE CODE: '{language}' - defaulting to WHITE. Supported: {list(self.language_colors.keys())}")
            return (255, 255, 255)  # White for any unsupported language
    
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
        """Calculate dimensions for text block using PIL for consistency.
        
        Args:
            display_lines: List of text lines to measure
            
        Returns:
            tuple: (line_heights, line_widths, total_height, max_width)
        """
        line_heights = []
        line_widths = []
        
        # Use PIL for ALL text measurements for consistency
        try:
            # CRITICAL FIX: Use the SAME font size calculation as rendering
            pil_font_size = max(16, int(self.style.font_scale * 30))  # Match _render_unicode_text
            pil_font = self._get_unicode_font(pil_font_size)
            
            for line in display_lines:
                if not line.strip():
                    # Empty line - use a space for measurement
                    bbox = pil_font.getbbox(' ')
                    h = bbox[3] - bbox[1]
                    w = 0
                else:
                    # Use getlength for more accurate width measurement
                    w = int(pil_font.getlength(line))
                    bbox = pil_font.getbbox(line)
                    h = bbox[3] - bbox[1]
                    
                    # Account for text shadow (2px offset) that extends the rendered bounds
                    # But only add a smaller compensation since getlength is more accurate
                    w += 4  # Shadow extends text width by 2px, plus small buffer
                    h += 4  # Shadow extends text height by 2px, plus small buffer
                
                line_heights.append(h)
                line_widths.append(w)
            
            total_text_height = sum(line_heights) + (len(display_lines) - 1) * 5  # 5px spacing between lines
            max_text_width = max(line_widths) if line_widths else 0
            
            logger.debug(f"PIL text dimensions: {len(display_lines)} lines, max_width={max_text_width}, total_height={total_text_height}, font_size={pil_font_size}")
            
            return line_heights, line_widths, total_text_height, max_text_width
            
        except Exception as e:
            logger.error(f"Error calculating PIL text dimensions: {e}")
            # Fallback to approximate dimensions
            fallback_height = int(self.style.font_scale * 24)
            line_heights = [fallback_height] * len(display_lines)
            line_widths = [len(line) * int(fallback_height * 0.6) for line in display_lines]
            total_text_height = sum(line_heights) + (len(display_lines) - 1) * 5
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
        
        # Horizontal centering with frame boundary constraints
        # Ensure background never extends outside frame boundaries
        if bg_width > frame_width:
            # Text is wider than frame - constrain background to frame width
            bg_x1 = 0
            bg_x2 = frame_width
            # Recalculate effective text area width within frame constraints
            effective_text_area_width = frame_width - (2 * padding)
        else:
            # Normal case - center background on frame
            bg_x1 = (frame_width - bg_width) // 2
            bg_x2 = bg_x1 + bg_width
            effective_text_area_width = max_text_width
        
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
        """Render a single line of text using PIL for all languages.
        
        Args:
            frame: Video frame to render on
            line: Text line to render
            x: X position for text (left edge)
            y: Y position for text (top edge, NOT baseline)
            fade_factor: Fade factor for text opacity
            language: Language code for color selection
            
        Returns:
            numpy.ndarray: Frame with text rendered
        """
        if not line.strip():
            return frame
        
        # Get language-specific color
        text_color = self.get_language_color(language)
        
        # CRITICAL DEBUG: Log all color assignments to catch issues
        expected_colors = {
            'en': (255, 255, 255),
            'fr': (255, 200, 150), 
            'de': (0, 255, 255),
            'it': (0, 165, 255),
            'hu': (0, 255, 0),
            'ru': (203, 192, 255),
            'uk': (255, 0, 255)
        }
        
        if language in expected_colors and text_color != expected_colors[language]:
            logger.error(f"🚨 {language.upper()} COLOR WRONG! Expected {expected_colors[language]} but got {text_color}")
        
        # DETECT YELLOW COLORS - log any unexpected yellow-like colors (but allow German's yellow)
        r, g, b = text_color
        if g > 200 and b < 200 and language != 'de':  # Yellow-like colors, but allow German
            logger.error(f"🚨 UNEXPECTED YELLOW COLOR DETECTED for language '{language}': {text_color}")
        
        # DEBUG: Log color and language for debugging
        logger.debug(f"RENDER_TEXT_LINE: lang='{language}' color={text_color} text='{line[:15]}...'")
        if language not in expected_colors:
            logger.warning(f"DEBUG: Unexpected language '{language}' with color {text_color} for text: '{line[:30]}...'")
        
        # USE PIL FOR ALL TEXT RENDERING
        # Convert BGR to RGB for PIL (our colors are stored in BGR format)
        rgb_color = (text_color[2], text_color[1], text_color[0])
        
        # Apply fade factor to color
        faded_color = tuple(int(c * fade_factor) for c in rgb_color)
        
        # DEBUG: Log color conversion for debugging
        logger.debug(f"Color conversion for {language}: BGR{text_color} -> RGB{rgb_color} -> faded{faded_color}")
        
        # Use the same font scale as before for consistency
        pil_font_size = int(self.style.font_scale * 24)
        
        # Render using PIL at the given position (y is already top-left, not baseline)
        result_frame = self._render_unicode_text(frame, line, (x, y), faded_color, pil_font_size, language)
        return result_frame
    
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
            
            # Calculate text dimensions using PIL
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
            
            # For PIL text: start_y is the TOP position of the first line
            start_y = bg_center_y - (actual_text_height // 2)
            
            # Render each line of text
            y = start_y
            for line_idx, (line, h, w) in enumerate(zip(display_lines, line_heights, line_widths)):
                if not line.strip():
                    y += h + 5  # Add space for empty lines
                    continue
                
                # CRITICAL FIX: Center each line within the BACKGROUND text area, not the frame
                # Handle both normal and frame-constrained cases
                bg_width = bg_x2 - bg_x1
                text_area_width = bg_width - (2 * self.style.padding)
                text_area_left = bg_x1 + self.style.padding
                
                # For very wide text that exceeds frame width, ensure text fits within frame
                if text_area_width < w:
                    # Text is wider than available area - position at left edge and clip
                    x = text_area_left
                else:
                    # Normal case - center the line within the available text area
                    x = text_area_left + (text_area_width - w) // 2
                
                # Ensure text doesn't go outside the frame boundaries
                x = max(0, min(x, frame_width - w))
                
                # Render the text line with language-specific color
                frame = self.render_text_line(frame, line, x, y, fade_factor, language)
                
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
        
        # CRITICAL DEBUG: Log exactly what captions we're trying to render
        if len(active_captions) > 3:
            logger.error(f"🚨 PROBLEM: {len(active_captions)} captions detected! Expected only 3 (en, fr, ru)")
            for i, cap in enumerate(active_captions):
                lang = cap.get('language', 'UNKNOWN')
                text = cap.get('text', '')[:30]
                logger.error(f"   Caption {i+1}: lang='{lang}', text='{text}...'")
        
        # Reduce debug logging frequency to improve performance
        should_log = len(active_captions) > 0 and (int(current_time * 10) % 10 == 0)  # Log once per second
        
        if should_log:
            logger.debug(f"[RENDER] Rendering {len(active_captions)} captions at time {current_time:.3f}s")
            # Debug each caption
            for i, cap in enumerate(active_captions):
                lang = cap.get('language', 'UNKNOWN')
                text = cap.get('text', '')[:20]
                logger.debug(f"   Caption {i}: lang='{lang}', text='{text}...'")
        
        # Sort captions by start time (oldest first) to maintain proper stacking order
        active_captions = sorted(active_captions, key=lambda x: x['start_time'])
        
        # Separate primary and secondary language captions for proper layering
        primary_captions = [c for c in active_captions if c.get('is_primary', True)]
        secondary_captions = [c for c in active_captions if not c.get('is_primary', True)]
        
        # ADDITIONAL DEBUG: Check for duplicates
        lang_counts = {}
        for cap in active_captions:
            lang = cap.get('language', 'UNKNOWN')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        for lang, count in lang_counts.items():
            if count > 1:
                logger.error(f"🚨 DUPLICATE LANGUAGE DETECTED: '{lang}' appears {count} times!")
        
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
    
    def _get_unicode_font(self, size=24):
        """Get a font that supports Unicode characters.
        
        Args:
            size: Font size
            
        Returns:
            PIL ImageFont object
        """
        try:
            # Try to use system fonts that support Unicode
            font_paths = [
                # Windows fonts
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf", 
                "C:/Windows/Fonts/segoeui.ttf",
                # Common cross-platform fallbacks
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
                "/usr/share/fonts/TTF/arial.ttf",  # Some Linux distributions
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, size)
            
            # Fallback to default font
            return ImageFont.load_default()
            
        except Exception as e:
            logger.warning(f"Could not load Unicode font: {e}, using default")
            return ImageFont.load_default()
    
    def _render_unicode_text(self, frame, text, position, color, font_size=24, language='en'):
        """Render Unicode text on frame using PIL with exact positioning to match OpenCV background bounds.
        
        Args:
            frame: OpenCV frame (numpy array)
            text: Text to render (can contain Unicode characters)
            position: (x, y) position tuple - MUST match OpenCV text baseline
            color: RGB color tuple
            font_size: Font size
            language: Language code for font selection
            
        Returns:
            numpy.ndarray: Frame with text rendered
        """
        try:
            x, y = position
            
            # DEBUG: Log what we're trying to render
            logger.debug(f"Unicode rendering: lang='{language}', color={color}, text='{text[:20]}...', pos=({x},{y})")
            
            # Convert OpenCV frame (BGR) to PIL Image (RGB)
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Calculate font size to match OpenCV rendering exactly
            opencv_equivalent_size = max(16, int(self.style.font_scale * 30))
            
            # Get Unicode font (cache for better performance)
            if not hasattr(self, '_cached_fonts'):
                self._cached_fonts = {}
            
            font_key = f"{opencv_equivalent_size}_{language}"
            if font_key not in self._cached_fonts:
                self._cached_fonts[font_key] = self._get_unicode_font(opencv_equivalent_size)
            font = self._cached_fonts[font_key]
            
            # CRITICAL FIX: Use the SAME positioning logic as OpenCV for consistency
            # The key insight: text_y from calculate_text_position is the TOP of the text area
            # For PIL, we need to render at that exact top position (not baseline)
            
            # SIMPLIFIED: Use the text position directly for PIL top-left rendering
            # The position passed in is now already the correct top-left position
            pil_x, pil_y = position
            
            # DEBUG: Log positioning details
            logger.debug(f"{language.upper()} positioning: received position=({pil_x}, {pil_y})")
            
            # Ensure text doesn't go off screen
            pil_y = max(10, pil_y)  # At least 10 pixels from top
            
            # Draw text shadow first (for better readability)
            shadow_offset = 2
            shadow_color = (0, 0, 0)  # Black shadow
            draw.text((pil_x + shadow_offset, pil_y + shadow_offset), text, font=font, fill=shadow_color)
            
            # Draw main text using the calculated position
            draw.text((pil_x, pil_y), text, font=font, fill=color)
            
            # Convert back to OpenCV format (RGB to BGR)
            result_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # DEBUG: Log successful rendering
            logger.debug(f"{language.upper()} text rendered successfully at PIL position ({pil_x}, {pil_y})")
            
            return result_frame
            
        except Exception as e:
            logger.error(f"Error rendering Unicode text '{text}': {e}")
            # Fallback to original OpenCV rendering
            return frame 