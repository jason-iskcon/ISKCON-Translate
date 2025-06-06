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
            'fr': (255, 200, 150),     # Pale blue for French - BGR (changed from 255,255,150 to avoid yellow)
            'ru': (203, 192, 255),     # Pale pink for Russian - BGR
        }
        
        # SAFEGUARD: Absolutely NO yellow colors allowed anywhere
        self.forbidden_colors = [
            (255, 255, 0),   # Pure yellow
            (255, 255, 100), # Light yellow variants  
            (255, 255, 200), # Removed (255, 255, 150) as it conflicts with French
            (200, 255, 100),
            (150, 255, 100),
        ]
        
        logger.debug(f"CaptionRenderer initialized with style: {self.style}")
    
    def get_language_color(self, language: str = 'en') -> tuple:
        """Get the color for a specific language.
        
        Args:
            language: Language code (e.g., 'en', 'fr', 'ru')
            
        Returns:
            BGR color tuple for the language
        """
        # Only support the three specified languages, default everything else to white
        if language in self.language_colors:
            color = self.language_colors[language]
            
            # EXPLICIT COLOR ENFORCEMENT FIRST - before any validation
            if language == 'fr':
                color = (255, 200, 150)  # Force French to pale blue in BGR format (updated color)
                logger.debug(f"French color ENFORCED: {color} (BGR format)")
                return color  # Return immediately to avoid forbidden color check
            
            if language == 'ru':
                color = (203, 192, 255)  # Force Russian to pale pink in BGR format
                logger.debug(f"Russian color ENFORCED: {color} (BGR format)")
                return color  # Return immediately to avoid forbidden color check
            
            # SAFEGUARD: Check for forbidden yellow colors (only for English or unknown)
            if color in self.forbidden_colors:
                logger.error(f"FORBIDDEN YELLOW COLOR DETECTED for language '{language}': {color}. Forcing to white!")
                color = (255, 255, 255)  # Force to white
            
            # DEBUG: Log all color assignments
            logger.debug(f"Color assignment: '{language}' -> {color}")
            return color
        else:
            # Log unsupported language and default to white (NEVER yellow)
            logger.warning(f"UNSUPPORTED LANGUAGE CODE: '{language}' - defaulting to WHITE. Only 'en', 'fr', 'ru' are supported.")
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
        """Calculate dimensions for text block with proper Unicode support.
        
        Args:
            display_lines: List of text lines to measure
            
        Returns:
            tuple: (line_heights, line_widths, total_height, max_width)
        """
        line_heights = []
        line_widths = []
        
        for line in display_lines:
            # Check if line contains Unicode characters
            has_unicode = any(ord(char) > 127 for char in line)
            
            if has_unicode:
                # Use PIL for accurate Unicode text measurement
                try:
                    temp_img = Image.new('RGB', (100, 100))
                    temp_draw = ImageDraw.Draw(temp_img)
                    font = self._get_unicode_font(int(self.style.font_scale * 24))
                    bbox = temp_draw.textbbox((0, 0), line, font=font)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    line_heights.append(h)
                    line_widths.append(w)
                except Exception as e:
                    logger.warning(f"Failed PIL measurement for Unicode text, using OpenCV fallback: {e}")
                    # Fallback to OpenCV
                    (w, h), _ = cv2.getTextSize(
                        line, self.font, self.style.font_scale, self.style.font_thickness
                    )
                    line_heights.append(h)
                    line_widths.append(w)
            else:
                # Use OpenCV for ASCII text
                (w, h), _ = cv2.getTextSize(
                    line, self.font, self.style.font_scale, self.style.font_thickness
                )
                line_heights.append(h)
                line_widths.append(w)
        
        total_text_height = sum(line_heights) + (len(display_lines) - 1) * (self.style.font_thickness + 2)
        max_text_width = max(line_widths) if line_widths else 0
        
        logger.debug(f"Text dimensions: {len(display_lines)} lines, max_width={max_text_width}, total_height={total_text_height}")
        
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
        
        # Check if text contains Unicode characters (non-ASCII)
        has_unicode = any(ord(char) > 127 for char in line)
        
        # Only Russian uses Unicode in our simplified system
        uses_unicode_rendering = language == 'ru' or has_unicode
        
        # Get language-specific color
        text_color = self.get_language_color(language)
        
        # CRITICAL DEBUG: Log all color assignments to catch yellow colors
        if language == 'fr' and text_color != (255, 200, 150):
            logger.error(f"🚨 FRENCH COLOR WRONG! Expected (255,200,150) but got {text_color}")
        if language == 'ru' and text_color != (203, 192, 255):
            logger.error(f"🚨 RUSSIAN COLOR WRONG! Expected (203,192,255) but got {text_color}")
        
        # DETECT YELLOW COLORS - log any yellow-like colors
        r, g, b = text_color
        if g > 200 and b < 200:  # Yellow-like colors have high green, low blue
            logger.error(f"🚨 YELLOW COLOR DETECTED for language '{language}': {text_color}")
        
        # TEMPORARY DEBUG: Log color and language for debugging yellow text issue
        logger.debug(f"RENDER_TEXT_LINE: lang='{language}' color={text_color} text='{line[:15]}...'")
        if language not in ['en', 'fr', 'ru']:
            logger.warning(f"DEBUG: Unexpected language '{language}' with color {text_color} for text: '{line[:30]}...'")
        
        if uses_unicode_rendering:
            # Use PIL-based Unicode rendering
            # Convert BGR to RGB for PIL (our colors are stored in BGR format)
            rgb_color = (text_color[2], text_color[1], text_color[0])
            
            # Apply fade factor to color
            faded_color = tuple(int(c * fade_factor) for c in rgb_color)
            
            # DEBUG: Log color conversion for debugging
            logger.debug(f"Color conversion for {language}: BGR{text_color} -> RGB{rgb_color} -> faded{faded_color}")
            
            # Use the same font scale as OpenCV for consistency
            opencv_equivalent_size = int(self.style.font_scale * 24)
            
            # Render using PIL
            frame[:] = self._render_unicode_text(frame, line, (x, y), faded_color, opencv_equivalent_size, language)[:]
        else:
            # Use original OpenCV rendering for ASCII text
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
            
            # Apply fade factor directly to BGR color (no conversion needed for OpenCV)
            faded_color = tuple(int(c * fade_factor) for c in text_color)
            
            # DEBUG: Log OpenCV color application
            logger.debug(f"OpenCV rendering for {language}: original{text_color} -> faded{faded_color}")
            
            # Draw the text with faded color
            cv2.putText(
                frame, 
                line, 
                (x, y),
                self.font, 
                self.style.font_scale, 
                faded_color,  # Use BGR color directly
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
            # OpenCV uses font_scale directly, PIL needs pixel size
            # Use same calculation as in render_text_line for consistency
            opencv_equivalent_size = max(16, int(self.style.font_scale * 30))  # Increased base size for better matching
            
            # DEBUG: Log font size calculation
            logger.debug(f"Font size calculation: font_scale={self.style.font_scale}, opencv_size={opencv_equivalent_size}")
            
            # Get Unicode font (cache for better performance)
            if not hasattr(self, '_cached_fonts'):
                self._cached_fonts = {}
            
            font_key = f"{opencv_equivalent_size}_{language}"
            if font_key not in self._cached_fonts:
                self._cached_fonts[font_key] = self._get_unicode_font(opencv_equivalent_size)
            font = self._cached_fonts[font_key]
            
            # CRITICAL FIX: Ensure Russian text position matches OpenCV baseline exactly
            # OpenCV y is baseline, PIL y is top-left. Need to convert properly.
            # Get text metrics for accurate positioning
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # OpenCV baseline is approximately 80% from the top of the text height
            # But we need to account for descenders and ensure text fits in background
            descent = font.getmetrics()[1] if hasattr(font, 'getmetrics') else text_height * 0.2
            
            # Calculate PIL top position from OpenCV baseline position
            # y (OpenCV baseline) - text_height + descent = PIL top position
            adjusted_y = y - text_height + descent
            
            # ENSURE text stays within reasonable bounds (don't go off screen)
            adjusted_y = max(5, adjusted_y)  # At least 5 pixels from top
            
            # DEBUG: Log detailed positioning calculation
            logger.debug(f"Russian positioning: opencv_baseline={y}, text_h={text_height}, descent={descent:.1f}")
            logger.debug(f"Russian positioning: calculated_top={y - text_height + descent:.1f}, final_y={adjusted_y}")
            
            # Ensure text doesn't interfere with background calculation
            if adjusted_y < 0:
                adjusted_y = 5  # Force minimum distance from top
                logger.warning(f"Russian text positioned too high, moved to y={adjusted_y}")
            
            # Draw text shadow first (for better readability)
            shadow_offset = 2  # Same as OpenCV version
            shadow_color = (0, 0, 0)  # Black shadow
            draw.text((x + shadow_offset, adjusted_y + shadow_offset), text, font=font, fill=shadow_color)
            
            # Draw main text at calculated position
            draw.text((x, adjusted_y), text, font=font, fill=color)
            
            # Convert back to OpenCV format (RGB to BGR)
            result_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return result_frame
            
        except Exception as e:
            logger.error(f"Error rendering Unicode text '{text}': {e}")
            # Fallback to original OpenCV rendering
            return frame 