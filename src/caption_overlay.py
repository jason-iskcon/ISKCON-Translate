import cv2
import numpy as np
import time

# Import with try-except to handle both direct execution and module import
try:
    from logging_utils import get_logger
except ImportError:
    from .logging_utils import get_logger

# Get logger instance
logger = get_logger(__name__)

class CaptionOverlay:
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
        import threading
        
        self.captions = []  # List of dicts with text, start_time, end_time, and is_absolute flag
        self.video_start_time = 0  # Track when the video started
        self.lock = threading.Lock()  # For thread-safe access to captions list
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font_color = font_color
        self.bg_color = bg_color
        self.padding = padding
        self.y_offset = y_offset
        
    def set_video_start_time(self, start_time):
        """Set the video's start time to handle offset captions.
        
        Args:
            start_time: The absolute timestamp where the video starts (in seconds)
        """
        logger.info(f"[TIMING] Setting video start time to {start_time} (current time: {time.time()})")
        self.video_start_time = start_time
        logger.info(f"[TIMING] Video start time set. Current offset: {time.time() - start_time:.2f}s")
        
    # Context manager methods for resource management
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # No resources to clean up
        
    def add_caption(self, text, timestamp, duration=3.0, is_absolute=False, seamless=True):
        """Add a caption to be displayed.
        
        Args:
            text: Caption text to display
            timestamp: Timestamp for when to show the caption
            duration: How long to display the caption in seconds
            is_absolute: If True, timestamp is treated as absolute system time
            seamless: If True, will try to merge with previous caption if similar
            
        Returns:
            dict: The added caption or None if skipped
        """
        with self.lock:
            original_timestamp = timestamp
            
            # Log current state with timing details
            current_time = time.time()
            current_relative = current_time - (self.video_start_time if hasattr(self, 'video_start_time') else current_time)
            
            # Convert absolute timestamp to relative if needed
            if is_absolute:
                original_timestamp = timestamp
                timestamp = timestamp - self.video_start_time
                logger.info(f"[CAPTION] Adding absolute timestamp: '{text}' at {original_timestamp:.2f} (relative: {timestamp:.2f}s) for {duration:.1f}s")
            else:
                logger.info(f"[CAPTION] Adding relative caption: '{text}' at {timestamp:.2f}s for {duration:.1f}s")
            
            # Log current timing context
            logger.info(f"  - Current system time: {current_time:.2f}")
            logger.info(f"  - Video start time: {getattr(self, 'video_start_time', 0):.2f}")
            logger.info(f"  - Current relative time: {current_relative:.2f}s")
            
            # If the caption is in the past, adjust it to show immediately
            if timestamp < current_relative:
                logger.info(f"  - Adjusting timestamp from {timestamp:.2f}s to current time {current_relative:.2f}s")
                timestamp = current_relative
            
            # Log final timing
            logger.info(f"  - Will display at: {timestamp:.2f}s (in {max(0, timestamp - current_relative):.2f}s)")
            logger.info(f"  - Will end at: {(timestamp + duration):.2f}s")
                
            logger.info(f"[TIMING] Current relative time: {current_relative:.2f}s, "
                      f"Adding caption at: {timestamp:.2f}s, "
                      f"Duration: {duration:.1f}s")
            
            # Skip empty captions
            if not text.strip():
                logger.debug("Skipping empty caption")
                return None
                
            # Clean up the text (remove duplicate lines, extra spaces)
            text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
            
            # Create caption entry
            caption = {
                'text': text,
                'start_time': timestamp,
                'end_time': timestamp + duration,
                'added_at': time.time(),
                'was_absolute': is_absolute,
                'original_timestamp': original_timestamp,
                'display_count': 0  # Track how many times this caption has been displayed
            }
            
            # Check for duplicate/similar captions
            if self.captions and seamless:
                last_caption = self.captions[-1]
                similarity = self._text_similarity(text, last_caption['text'])
                if similarity > 0.8:  # 80% similar
                    logger.debug(f"[DUPLICATE] Skipping similar caption (similarity: {similarity:.2f}): '{text}'")
                    logger.debug(f"[DUPLICATE] Previous caption: '{last_caption['text']}'")
                    return None
            
            # Add the new caption
            self.captions.append(caption)
            
            # Log the addition
            logger.info(
                f"[CAPTION] Added caption: '{text[:50]}{'...' if len(text) > 50 else ''}'\n"
                f"          Start: {timestamp:.2f}s | End: {timestamp + duration:.2f}s | "
                f"Duration: {duration:.2f}s | Absolute: {is_absolute}"
            )
            
            # Log timing info if we have previous captions
            if len(self.captions) > 1:
                prev_caption = self.captions[-2]
                time_since_prev = timestamp - prev_caption['start_time']
                logger.debug(
                    f"[TIMING] Time since previous caption: {time_since_prev:.2f}s | "
                    f"Prev: {prev_caption['start_time']:.2f}s | Current: {timestamp:.2f}s"
                )
            
            # Sort captions by start time to ensure proper rendering order
            self.captions.sort(key=lambda x: x['start_time'])
            
            # Log current caption queue state
            logger.info(f"[CAPTION] Queue now has {len(self.captions)} captions")
            if self.captions:
                logger.debug("[CAPTION] Current caption queue:")
                for i, c in enumerate(self.captions):
                    logger.debug(
                        f"  {i}. '{c['text'][:30]}{'...' if len(c['text']) > 30 else ''}' | "
                        f"Start: {c['start_time']:.2f}s | End: {c['end_time']:.2f}s"
                    )
            
            return caption

    def _text_similarity(self, text1, text2):
        """Calculate a simple text similarity score between two strings.
        
        Args:
            text1 (str): First text to compare
            text2 (str): Second text to compare
            
        Returns:
            float: Similarity score between 0.0 (completely different) and 1.0 (identical)
        """
        if not text1 or not text2:
            return 0.0
            
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
            
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def prune_captions(self, current_time, buffer=1.0):
        """
        Remove captions whose end_time is more than `buffer` seconds before current_time.
        Also removes any captions that are too far in the future.
        """
        if not hasattr(self, 'last_prune_time') or (current_time - self.last_prune_time) > 1.0:
            before = len(self.captions)
            max_future_offset = 30  # 30 seconds in the future max
            
            self.captions = [
                c for c in self.captions 
                if (c['end_time'] >= current_time - buffer) and 
                   (c['start_time'] <= current_time + max_future_offset)
            ]
            
            # Also remove any captions that have been displayed for too long
            max_display_time = 10.0  # Maximum time a caption can be displayed (in seconds)
            self.captions = [
                c for c in self.captions 
                if (time.time() - c.get('added_at', 0)) < max_display_time
            ]
            
            after = len(self.captions)
            if before != after:
                logger.info(f"[CAPTION] Pruned {before-after} old/future captions at {current_time:.2f}s")
                
            self.last_prune_time = current_time

        
    def overlay_captions(self, frame, current_time=None, frame_count=0):
        """Overlay all valid captions on frame.
        
        Args:
            frame: The frame to overlay captions on
            current_time: The current timestamp in seconds (relative to video start)
            frame_count: The current frame number (used for logging)
            
        Returns:
            Frame with captions drawn
        """
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        result_frame = frame.copy()
        if current_time is None:
            logger.warning("[OVERLAY] No current_time provided, skipping caption overlay")
            return frame
            
        # Ensure current_time is relative to video start and non-negative
        current_relative_time = max(0, current_time)
            
        # Caption rendering is now handled by the main application
        # Test captions are added via add_caption() from main.py
        
        # Log timing info every second for debugging
        if frame_count % 30 == 0:  # Log every second at 30fps
            logger.info(f"[OVERLAY] Frame {frame_count} | Time: {current_relative_time:.2f}s | Captions in queue: {len(self.captions)}")
            if self.captions:
                logger.info("[OVERLAY] Next caption in queue:")
                for i, c in enumerate(sorted(self.captions, key=lambda x: x['start_time'])[:3]):  # Show next 3 captions by start time
                    time_until = c['start_time'] - current_relative_time
                    status = "ACTIVE " if c['start_time'] <= current_relative_time <= c['end_time'] else "PENDING"
                    logger.info(f"  {i+1}. [{status}] In {time_until:6.2f}s: '{c['text'][:50]}{'...' if len(c['text']) > 50 else ''}'")
    
        # Log caption state for debugging
        log_frequency = 15  # Log at ~2fps for debugging (every 15 frames at 30fps)
        should_log = frame_count % log_frequency == 0
        
        if should_log:
            logger.info(f"\n[OVERLAY] === Frame {frame_count} ===")
            logger.info(f"[OVERLAY] Current relative time: {current_relative_time:.2f}s")
            logger.info(f"[OVERLAY] Video start time: {self.video_start_time:.2f}")
            logger.info(f"[OVERLAY] System time: {time.time():.6f}")
            logger.info(f"[OVERLAY] Time since video start: {time.time() - self.video_start_time:.6f}s")
        
        with self.lock:
            # First, remove any captions that have already ended (with a small buffer)
            before_prune = len(self.captions)
            self.captions = [c for c in self.captions if c['end_time'] > current_time - 1.0]  # Keep captions that ended <1s ago
            after_prune = len(self.captions)
            
            if should_log and before_prune != after_prune:
                logger.info(f"[OVERLAY] Pruned {before_prune - after_prune} old captions")
                logger.debug(f"[OVERLAY] Current time: {current_time:.2f}s, Kept captions with end_time > {current_time - 1.0:.2f}s")
            
            # Find all captions that should be active now (with small buffer for smooth transitions)
            active_captions = []
            
            # Only log detailed caption info periodically to reduce log spam
            should_log_details = frame_count % 30 == 0  # Log details once per second
            
            if should_log_details:
                logger.info(f"[OVERLAY] Checking {len(self.captions)} captions against time {current_relative_time:.2f}")
            
            for i, c in enumerate(self.captions):
                # Calculate timing for this caption
                time_until_start = c['start_time'] - current_relative_time
                time_until_end = c['end_time'] - current_relative_time
                
                # Add small buffer for activation/deactivation to prevent flickering
                is_active = (c['start_time'] - 0.1) <= current_relative_time <= (c['end_time'] + 0.1)
                
                if should_log_details:
                    logger.info(
                        f"[OVERLAY] Caption {i}: '{c['text'][:30]}...' | "
                        f"Active: {is_active} | Start in: {time_until_start:.2f}s | "
                        f"End in: {time_until_end:.2f}s"
                    )
                
                if is_active:
                    # Update display count and log first display
                    if 'display_count' not in c:
                        c['display_count'] = 0
                    if c['display_count'] == 0:
                        logger.info(f"[CAPTION] Displaying caption for first time: '{c['text']}'")
                    c['display_count'] += 1
                    
                    active_captions.append(c)
                    if should_log_details:
                        logger.info(f"[OVERLAY]   - Active for {time_until_end:.2f}s more")
                elif should_log_details and time_until_start > 0 and time_until_start < 2.0:
                    logger.info(f"[OVERLAY]   - Will be active in {time_until_start:.2f}s")
            
            # If no captions, return the frame as-is
            if not active_captions:
                if should_log:
                    logger.info(f"[OVERLAY] No active captions at {current_relative_time:.2f}s")
                    if self.captions:
                        logger.info("[OVERLAY] Upcoming captions:")
                        for c in sorted(self.captions, key=lambda x: x['start_time']):
                            time_until = c['start_time'] - current_relative_time
                            if time_until > 0:
                                logger.info(f"  - In {time_until:.2f}s: '{c['text'][:50]}{'...' if len(c['text']) > 50 else ''}'")
                return frame
                
            # Process all active captions in order of their start time
            active_captions = sorted(active_captions, key=lambda x: x['start_time'])
            
            if should_log:
                logger.info(f"[OVERLAY] Found {len(active_captions)} active captions")
                for i, cap in enumerate(active_captions, 1):
                    logger.info(f"  {i}. '{cap['text']}' ({cap['start_time']:.2f}-{cap['end_time']:.2f}s)")
            
            # Process each active caption
            for current_caption in active_captions:
                try:
                    # Calculate timing for this caption
                    caption_start = current_caption.get('start_time', 0)
                    caption_end = current_caption.get('end_time', 0)
                    
                    # Skip if this caption is not ready to be displayed yet (with small buffer)
                    if current_time < caption_start - 0.1:
                        continue
                        
                    # Skip if caption has ended (with small buffer)
                    if current_time > caption_end + 0.1:
                        continue
                        
                    # Calculate time in caption and time until end
                    time_in_caption = current_time - caption_start
                    time_until_end = caption_end - current_time
                    
                    # Log timing information for debugging
                    logger.debug(f"[FADE] current_time: {current_time:.3f}, "
                               f"caption: {current_caption.get('start_time', 0):.3f}-{current_caption.get('end_time', 0):.3f}, "
                               f"time_in: {time_in_caption:.3f}, time_until_end: {time_until_end:.3f}")
                    
                    # Calculate fade in/out effects
                    fade_duration = 0.3  # seconds for fade in/out
                    if time_in_caption < fade_duration:
                        # Fade in
                        fade_factor = time_in_caption / fade_duration
                    elif time_until_end < fade_duration:
                        # Fade out
                        fade_factor = time_until_end / fade_duration
                    else:
                        # Fully visible
                        fade_factor = 1.0
                    
                    # Ensure smooth transitions with minimum visibility
                    fade_factor = max(0.05, min(1.0, fade_factor))
                    logger.debug(f"[FADE] Rendering caption - fade_factor: {fade_factor:.3f}")
                    
                    # Skip if fade factor is too low (caption not visible)
                    if fade_factor <= 0.05:
                        continue
                        
                    # Log caption timing info
                    logger.info(f"[OVERLAY]   Caption: '{current_caption.get('text', '')[:30]}...'")
                    logger.info(f"[OVERLAY]   Start: {current_caption.get('start_time', 0):.2f}s | "
                              f"End: {current_caption.get('end_time', 0):.2f}s | Now: {current_time:.2f}s | "
                              f"Time in: {time_in_caption:.2f}s | "
                              f"Remaining: {time_until_end:.1f}s")
                    
                    # Render the caption on the frame
                    caption_text = current_caption.get('text', '').strip()
                    if not caption_text:
                        continue
                        
                    # Split into lines and remove empty lines
                    lines = [line.strip() for line in caption_text.split('\n') if line.strip()]
                    if not lines:
                        continue
                        
                    # Calculate text size and position
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    font_thickness = 2
                    padding = 10
                    
                    # Calculate text block size
                    line_heights = []
                    line_widths = []
                    for line in lines:
                        (w, h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                        line_heights.append(h)
                        line_widths.append(w)
                    
                    total_text_height = sum(line_heights) + (len(lines) - 1) * (font_thickness + 2)
                    max_text_width = max(line_widths) if line_widths else 0
                    
                    # Calculate text block position (centered at bottom)
                    text_block_x = (frame_width - max_text_width) // 2
                    text_block_y = frame_height - self.y_offset - total_text_height
                    
                    # Draw background rectangle
                    bg_padding = padding
                    bg_x1 = max(0, text_block_x - bg_padding)
                    bg_y1 = max(0, text_block_y - bg_padding)
                    bg_x2 = min(frame_width, text_block_x + max_text_width + bg_padding)
                    bg_y2 = min(frame_height, text_block_y + total_text_height + bg_padding)
                    
                    # Create semi-transparent background
                    overlay = result_frame.copy()
                    cv2.rectangle(
                        overlay,
                        (bg_x1, bg_y1),
                        (bg_x2, bg_y2),
                        self.bg_color,
                        -1
                    )
                    
                    # Apply fade effect to background
                    alpha = 0.7 * fade_factor
                    cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
                    
                    # Draw each line of text with shadow
                    y = text_block_y
                    for line in lines:
                        if not line.strip():
                            y += int(h * 1.5)  # Add extra space for empty lines
                            continue
                            
                        # Get text size for this line
                        (w, h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                        x = (frame_width - w) // 2  # Center each line
                        
                        # Draw text shadow (slightly offset)
                        shadow_offset = 2
                        shadow_color = (0, 0, 0)  # Black shadow
                        cv2.putText(
                            result_frame, 
                            line, 
                            (x + shadow_offset, y + shadow_offset),
                            font, 
                            font_scale, 
                            shadow_color,
                            font_thickness + 1, 
                            cv2.LINE_AA
                        )
                        
                        # Draw main text with fade effect
                        text_color = list(self.font_color)
                        if len(text_color) == 3:  # If no alpha channel, add one
                            text_color = text_color + [255]  # Fully opaque by default
                        
                        # Apply fade factor to alpha channel
                        text_color[3] = int(text_color[3] * fade_factor)
                        
                        # Convert to BGR for OpenCV
                        bgr_color = tuple(text_color[2::-1])  # Convert RGB to BGR and remove alpha
                        
                        # Draw the text
                        cv2.putText(
                            result_frame, 
                            line, 
                            (x, y),
                            font, 
                            font_scale, 
                            bgr_color,
                            font_thickness, 
                            cv2.LINE_AA
                        )
                        
                        # Move to next line position with some spacing
                        y += h + 5
                    
                except Exception as e:
                    logger.error(f"Error processing caption: {e}", exc_info=True)
                    # Skip to next caption on error
                    continue
                
            # Log all captions in queue for debugging
            if self.captions and should_log:
                logger.info("[OVERLAY] Caption queue (by start time):")
                for i, c in enumerate(sorted(self.captions, key=lambda x: x['start_time'])):
                    time_until = c['start_time'] - current_relative_time
                    time_remaining = c['end_time'] - current_relative_time
                    
                    # Determine status and time info
                    if time_until > 0:
                        status = f"in {time_until:.1f}s"
                    elif time_remaining < 0:
                        status = "ended"
                    else:
                        status = f"active ({time_remaining:.1f}s remaining)"
                    
                    logger.info(f"  {i+1}. '{c['text']}' | Start: {c['start_time']:.1f}s | End: {c['end_time']:.1f}s | {status}")
            
            # Skip rendering if no active captions
            if not active_captions:
                return result_frame
            
            # Log caption timing info for debugging
            if should_log and self.captions:
                logger.info("[OVERLAY] Caption timing details:")
                for i, c in enumerate(sorted(self.captions, key=lambda x: x['start_time'])):
                    time_until = c['start_time'] - current_relative_time
                    time_remaining = c['end_time'] - current_relative_time
                    
                    # Determine status and time info
                    if time_until > 0:
                        status = "PENDING"
                        time_info = f"starts in {time_until:.1f}s"
                    elif time_remaining < 0:
                        status = "ENDED  "
                        time_info = f"ended {abs(time_remaining):.1f}s ago"
                    else:
                        status = "ACTIVE "
                        time_info = f"active, {time_remaining:.1f}s remaining"
                    
                    # Log caption info
                    logger.info(
                        f"  {i+1:2d}. [{status}] {time_info:>25s} | "
                        f"'{c['text'][:40]}{'...' if len(c['text']) > 40 else ''}'"
                    )
                    
                    # Log timing details
                    logger.debug(
                        f"      Start: {c['start_time']:.2f}s | "
                        f"End: {c['end_time']:.2f}s | "
                        f"Duration: {c['end_time']-c['start_time']:.2f}s"
                        f"{' | Added: ' + str(round(time.time() - c['added_at'], 1)) + 's ago' if 'added_at' in c else ''}"
                    )
                    
                    # Log timing mode if available
                    if 'was_absolute' in c:
                        abs_status = "ABSOLUTE" if c['was_absolute'] else "RELATIVE"
                        logger.debug(f"      Timing: {abs_status} | Original: {c.get('original_timestamp', 'N/A')}")

            # Start with a clean copy of the frame
            result_frame = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate timing for fade effects
            time_in_caption = current_relative_time - current_caption['start_time']
            time_until_end = current_caption['end_time'] - current_relative_time
            fade_duration = 0.3  # seconds for fade in/out
            
            # Log timing information for debugging
            logger.debug(f"[FADE] current_time: {current_relative_time:.3f}, "
                       f"caption: {current_caption['start_time']:.3f}-{current_caption['end_time']:.3f}, "
                       f"time_in: {time_in_caption:.3f}, time_until_end: {time_until_end:.3f}")
            
            # Check if we're outside the caption's time window (with 0.1s buffer)
            if time_in_caption < -0.1 or time_until_end < -0.1:
                # Skip rendering if we're clearly out of the window
                logger.debug(f"[FADE] Skipping caption: time_in={time_in_caption:.3f}, time_until_end={time_until_end:.3f}")
                return frame
                
            # Calculate fade in/out effects
            if time_in_caption < fade_duration:
                # Fade in
                fade_factor = time_in_caption / fade_duration
            elif time_until_end < fade_duration:
                # Fade out
                fade_factor = time_until_end / fade_duration
            else:
                # Fully visible
                fade_factor = 1.0
            
            # Log timing information for debugging
            logger.debug(f"[FADE] current_time: {current_relative_time:.3f}, "
                       f"caption: {current_caption['start_time']:.3f}-{current_caption['end_time']:.3f}, "
                       f"time_in: {time_in_caption:.3f}, time_until_end: {time_until_end:.3f}")
            
            # Skip rendering if we're clearly out of the window (with 0.1s buffer)
            if time_until_end < -0.1:
                logger.debug(f"[FADE] Skipping caption (ended): time_in={time_in_caption:.3f}, time_until_end={time_until_end:.3f}")
                return frame
            
            # Ensure smooth transitions with minimum visibility
            fade_factor = max(0.05, min(1.0, fade_factor))
            logger.debug(f"[FADE] Rendering caption - fade_factor: {fade_factor:.3f}")
            
            # Process each active caption
            for current_caption in active_captions:
                # Split the current caption text into lines and clean them up
                lines = [line.strip() for line in current_caption['text'].split('\n') if line.strip()]
        
                # Remove duplicate lines while preserving order and ignoring case differences
                seen = set()
                unique_lines = []
                for line in lines:
                    # Normalize whitespace and convert to lowercase for comparison
                    normalized = ' '.join(line.lower().split())
                    if normalized and normalized not in seen:
                        unique_lines.append(line)  # Keep original line with original case
                        seen.add(normalized)
        
                # Use the deduplicated lines for display
                display_lines = unique_lines
        
                # If there are no newlines in the original text and it's a single line, wrap long lines
                if len(display_lines) == 1 and '\n' not in current_caption['text']:
                    max_chars_per_line = 60
                    words = display_lines[0].split()
                    wrapped_lines = []
                    current_line = []
                    for word in words:
                        if current_line and len(' '.join(current_line + [word])) > max_chars_per_line:
                            wrapped_lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            current_line.append(word)
                    if current_line:  # Add the last line
                        wrapped_lines.append(' '.join(current_line))
                    display_lines = wrapped_lines
        
                # Calculate total text block size
                line_heights = []
                line_widths = []
                
                for line in display_lines:
                    (w, h), _ = cv2.getTextSize(
                        line, self.font, self.font_scale, self.font_thickness
                    )
                    line_heights.append(h)
                    line_widths.append(w)
                
                total_text_height = sum(line_heights) + (len(display_lines) - 1) * (self.font_thickness + 2)
                max_text_width = max(line_widths) if line_widths else 0
                
                # Calculate text block position (centered at bottom with vertical offset for multiple captions)
                caption_index = active_captions.index(current_caption)
                vertical_offset = caption_index * (total_text_height + 20)  # 20px spacing between captions
                
                text_block_x = (frame_width - max_text_width) // 2
                text_block_y = frame_height - self.y_offset - total_text_height - vertical_offset
                
                # Draw background rectangle for the entire text block
                bg_padding = self.padding
                bg_x1 = max(0, text_block_x - bg_padding)
                bg_y1 = max(0, text_block_y - bg_padding)
                bg_x2 = min(frame_width, text_block_x + max_text_width + bg_padding)
                bg_y2 = min(frame_height, text_block_y + total_text_height + bg_padding)
                
                # Create a semi-transparent background with fade effect
                overlay = result_frame.copy()
                cv2.rectangle(
                    overlay,
                    (bg_x1, bg_y1),
                    (bg_x2, bg_y2),
                    self.bg_color,
                    -1
                )
                
                # Apply fade effect to the background
                alpha = 0.7 * fade_factor  # Base opacity * fade factor
                cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
                
                # Draw each line of text with shadow for better visibility
                y = text_block_y
                for line in display_lines:
                    if not line.strip():
                        y += int(h * 1.5)  # Add extra space for empty lines
                        continue
                        
                    # Get text size for this line
                    (w, h), _ = cv2.getTextSize(
                        line, self.font, self.font_scale, self.font_thickness
                    )
                    x = (frame_width - w) // 2  # Center each line
                    
                    # Draw text shadow (slightly offset)
                    shadow_offset = 2
                    shadow_color = (0, 0, 0)  # Black shadow
                    cv2.putText(
                        result_frame, 
                        line, 
                        (x + shadow_offset, y + shadow_offset),
                        self.font, 
                        self.font_scale, 
                        shadow_color,
                        self.font_thickness + 1, 
                        cv2.LINE_AA
                    )
                    
                    # Draw main text with fade effect applied
                    text_color = list(self.font_color)
                    if len(text_color) == 3:  # If no alpha channel, add one
                        text_color = text_color + [255]  # Fully opaque by default
                    
                    # Apply fade factor to alpha channel
                    text_color[3] = int(text_color[3] * fade_factor)
                    
                    # Convert to BGR for OpenCV
                    bgr_color = tuple(text_color[2::-1])  # Convert RGB to BGR and remove alpha
                    
                    # Draw the text
                    cv2.putText(
                        result_frame, 
                        line, 
                        (x, y),
                        self.font, 
                        self.font_scale, 
                        bgr_color,
                        self.font_thickness, 
                        cv2.LINE_AA
                    )
                    
                    # Move to next line position with some spacing
                    y += h + 5
        
        logger.debug(f"Rendered captions at {current_time:.2f}s: {display_lines}")
            
        return result_frame
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
