import cv2
import numpy as np
import time

# Import with try-except to handle both direct execution and module import
try:
    from logging_utils import get_logger, TRACE
except ImportError:
    from .logging_utils import get_logger, TRACE

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
        
        logger.info("Initializing CaptionOverlay")
        logger.debug(f"Font scale: {font_scale}, Thickness: {font_thickness}")
        logger.debug(f"Text color: {font_color}, BG color: {bg_color}")
        logger.debug(f"Padding: {padding}px, Y-Offset: {y_offset}px")
        
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
        logger.info("CaptionOverlay initialized")
        
        logger.debug("CaptionOverlay initialization complete")
        
    def set_video_start_time(self, start_time):
        """Set the video's start time to handle offset captions.
        
        Args:
            start_time: The absolute timestamp where the video starts (in seconds)
        """
        logger.info(f"[TIMING] Setting video start time to {start_time} (current time: {time.time()})")
        logger.debug(f"[TIMING] Previous video start time: {getattr(self, 'video_start_time', 'not set')}")
        self.video_start_time = start_time
        logger.info(f"[TIMING] Video start time set to {start_time}. Current offset: {time.time() - start_time:.2f}s")
        logger.trace(f"[TIMING] Video start time details - System time: {time.time()}, Offset: {time.time() - start_time:.6f}s")
        
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
                logger.debug(f"[CAPTION] Adding absolute timestamp at {original_timestamp:.2f}s (relative: {timestamp:.2f}s) for {duration:.1f}s")
                logger.trace(f"[TIMING] Absolute: {original_timestamp:.6f}, Video start: {self.video_start_time:.6f}, Relative: {timestamp:.6f}")
            else:
                logger.debug(f"[CAPTION] Adding relative caption at {timestamp:.2f}s for {duration:.1f}s")
            
            # Move detailed timing to TRACE level
            logger.trace(f"[TIMING] System time: {current_time:.6f}")
            logger.trace(f"[TIMING] Video start: {getattr(self, 'video_start_time', 0):.6f}")
            logger.trace(f"[TIMING] Relative time: {current_relative:.6f}s")
            
            # If the caption is in the past, adjust it to show immediately
            if timestamp < current_relative:
                time_diff = current_relative - timestamp
                logger.debug(f"[CAPTION] Adjusting timestamp by {time_diff:.2f}s")
                logger.trace(
                    f"[TIMING] Adjustment details | "
                    f"Original: {timestamp:.6f}s | "
                    f"Current: {current_relative:.6f}s | "
                    f"Difference: {time_diff:.6f}s"
                )
                timestamp = current_relative
            
            # Move detailed timing to DEBUG level
            logger.debug(
                f"[CAPTION] Scheduling | "
                f"Starts in: {max(0, timestamp - current_relative):.2f}s | "
                f"Duration: {duration:.1f}s"
            )
            
            # Skip empty captions
            if not text.strip():
                logger.debug("Skipping empty caption")
                return None
                
            logger.trace(f"Adding caption: '{text[:50]}{'...' if len(text) > 50 else ''}' at {timestamp:.6f}s for {duration:.3f}s")
                
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
                
                # Log deduplication at appropriate levels
                if similarity > 0.8:  # 80% similar
                    logger.debug(f"[DEDUPE] Skipping similar caption (score: {similarity:.2f})")
                    logger.trace(
                        f"[DEDUPE] Details | "
                        f"New: '{text[:50]}{'...' if len(text) > 50 else ''}' | "
                        f"Prev: '{last_caption['text'][:50]}{'...' if len(last_caption['text']) > 50 else ''}'"
                    )
                    logger.trace(f"[DEDUPE] Timestamp diff: {timestamp - last_caption['start_time']:.6f}s")
                    return None
            
            # Add the new caption
            self.captions.append(caption)
            
            # Log caption addition at appropriate levels
            caption_preview = text.replace('\n', '\\n')
            logger.debug(
                f"[CAPTION] Added | "
                f"Text: '{caption_preview[:50]}{'...' if len(caption_preview) > 50 else ''}' | "
                f"Start: {timestamp:.2f}s | "
                f"Duration: {duration:.1f}s"
            )
            logger.trace(
                f"[CAPTION] Full details | "
                f"Start: {timestamp:.6f}s | "
                f"End: {timestamp + duration:.6f}s | "
                f"Absolute: {is_absolute} | "
                f"Text: '{caption_preview}'"
            )
            
            # Log timing info if we have previous captions
            if len(self.captions) > 1:
                prev_caption = self.captions[-2]
                time_since_prev = timestamp - prev_caption['start_time']
                logger.trace(
                    f"[TIMING] Caption interval | "
                    f"Since prev: {time_since_prev:.3f}s | "
                    f"Prev: {prev_caption['start_time']:.3f}s | "
                    f"Current: {timestamp:.3f}s"
                )
            
            # Sort captions by start time to ensure proper rendering order
            self.captions.sort(key=lambda x: x['start_time'])
            
            # Log queue state at appropriate levels
            logger.debug(f"[CAPTION] Queue size: {len(self.captions)}")
            if self.captions and logger.isEnabledFor(TRACE):
                logger.trace("[CAPTION] Queue contents:")
                for i, c in enumerate(self.captions[-5:]):  # Only show last 5 to avoid log spam
                    logger.trace(
                        f"  {i+1}. '{c['text'][:30]}{'...' if len(c['text']) > 30 else ''}' | "
                        f"Start: {c['start_time']:.3f}s | "
                        f"End: {c['end_time']:.3f}s"
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
            
            removed_captions = [
                c for c in self.captions 
                if not ((c['end_time'] >= current_time - buffer) and 
                      (c['start_time'] <= current_time + max_future_offset))
            ]
            
            # Log removed captions
            for c in removed_captions:
                logger.debug(f"[CAPTION] Pruning caption (expired): '{c['text'][:50]}{'...' if len(c['text']) > 50 else ''}' "
                           f"(displayed {c.get('display_count', 0)} times, duration: {c['end_time']-c['start_time']:.1f}s)")
            
            # Keep only active captions
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
        # Start timing the rendering operation
        render_start = time.time()
        
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
    
        # Log caption state for debugging - less frequent for production
        log_frequency = 15  # Log at ~2fps for debugging (every 15 frames at 30fps)
        should_log = frame_count % log_frequency == 0
        
        if should_log:
            # Keep basic frame info at DEBUG level
            logger.debug(f"[OVERLAY] Processing frame {frame_count}")
            
            # Move detailed timing to TRACE level
            logger.trace(
                f"[TIMING] Frame {frame_count} details - "
                f"Relative time: {current_relative_time:.6f}s | "
                f"Video start: {self.video_start_time:.6f} | "
                f"System time: {time.time():.6f} | "
                f"Time since start: {time.time() - self.video_start_time:.6f}s"
            )
        
        with self.lock:
            # First, remove any captions that have already ended (with a small buffer)
            before_prune = len(self.captions)
            self.captions = [c for c in self.captions if c['end_time'] > current_time - 1.0]  # Keep captions that ended <1s ago
            after_prune = len(self.captions)
            
            if before_prune != after_prune:
                # Move pruning details to DEBUG level
                logger.debug(
                    f"[OVERLAY] Pruned {before_prune - after_prune} old captions | "
                    f"Current time: {current_time:.2f}s | "
                    f"Kept captions with end_time > {current_time - 1.0:.2f}s"
                )
            
            # Find all captions that should be active now (with small buffer for smooth transitions)
            active_captions = []
            
            # Log levels for different frequencies
            should_log_details = frame_count % 30 == 0  # Log details once per second
            
            logger.trace(f"[OVERLAY] Processing {len(self.captions)} captions at time {current_relative_time:.6f}")
            
            for i, c in enumerate(self.captions):
                # Calculate timing for this caption
                time_until_start = c['start_time'] - current_relative_time
                time_until_end = c['end_time'] - current_relative_time
                
                # Add small buffer for activation/deactivation to prevent flickering
                is_active = (c['start_time'] - 0.1) <= current_relative_time <= (c['end_time'] + 0.1)
                
                # Log caption timing details at appropriate levels
                if should_log_details:
                    logger.debug(
                        f"[OVERLAY] Caption {i}: '{c['text'][:30]}...' | "
                        f"Active: {is_active} | Start in: {time_until_start:.3f}s | "
                        f"End in: {time_until_end:.3f}s"
                    )
                
                if is_active:
                    # Update display count and log first display
                    if 'display_count' not in c:
                        c['display_count'] = 0
                    if c['display_count'] == 0:
                        # Move first display to DEBUG level with less verbose format
                        logger.debug(
                            f"[CAPTION] Displaying: '{c['text'][:50]}{'...' if len(c['text']) > 50 else ''}' | "
                            f"At: {current_relative_time:.2f}s | "
                            f"Duration: {c['end_time']-c['start_time']:.1f}s"
                        )
                    c['display_count'] += 1
                    
                    # Add trace logging for active caption timing
                    if c['display_count'] == 1:  # Only log first frame to reduce noise
                        logger.trace(
                            f"[TIMING] First frame timing - "
                            f"Caption: '{c['text'][:30]}...' | "
                            f"Start: {c['start_time']:.6f}s | "
                            f"Current: {current_relative_time:.6f}s | "
                            f"End: {c['end_time']:.6f}s | "
                            f"Time in: {current_relative_time - c['start_time']:.6f}s"
                        )
                    
                    active_captions.append(c)
                    
                    if should_log_details:
                        logger.debug(f"[OVERLAY]   - Active for {time_until_end:.3f}s more")
                elif should_log_details and time_until_start > 0 and time_until_start < 2.0:
                    logger.debug(f"[OVERLAY]   - Will be active in {time_until_start:.3f}s")
            
            # If no captions, return the frame as-is
            if not active_captions:
                if should_log:
                    # Move queue status to TRACE level
                    logger.trace(f"[OVERLAY] No active captions at {current_relative_time:.3f}s")
                    if self.captions:
                        upcoming = [c for c in sorted(self.captions, key=lambda x: x['start_time']) 
                                  if c['start_time'] > current_relative_time]
                        if upcoming:
                            logger.trace("[OVERLAY] Upcoming captions:")
                            for c in upcoming[:3]:  # Show next 3 captions
                                time_until = c['start_time'] - current_relative_time
                                logger.trace(
                                    f"  - In {time_until:6.3f}s: "
                                    f"'{c['text'][:50]}{'...' if len(c['text']) > 50 else ''}'"
                                )
                            if len(upcoming) > 3:
                                logger.trace(f"  - ... and {len(upcoming) - 3} more captions")
                return frame
                
            # Process all active captions in order of their start time
            active_captions = sorted(active_captions, key=lambda x: x['start_time'])
            
            if should_log:
                logger.info(f"[OVERLAY] Found {len(active_captions)} active captions")
                for i, cap in enumerate(active_captions, 1):
                    logger.info(f"  {i}. '{cap['text']}' ({cap['start_time']:.2f}-{cap['end_time']:.2f}s)")
            
            # Process each active caption
            for i, current_caption in enumerate(active_captions, 1):
                try:
                    # Get caption timing info
                    caption_start = current_caption.get('start_time', 0)
                    caption_end = current_caption.get('end_time', 0)
                    caption_duration = caption_end - caption_start
                    
                    # Skip if this caption is not ready to be displayed yet (with small buffer)
                    if current_time < caption_start - 0.1:
                        logger.trace(
                            f"[RENDER] Skipping caption {i} - Not started yet | "
                            f"Current: {current_time:.6f}s | Start: {caption_start:.6f}s | "
                            f"Time until start: {caption_start - current_time:.6f}s"
                        )
                        continue
                        
                    # Skip if caption has ended (with small buffer)
                    if current_time > caption_end + 0.1:
                        logger.trace(
                            f"[RENDER] Skipping caption {i} - Already ended | "
                            f"Current: {current_time:.6f}s | End: {caption_end:.6f}s | "
                            f"Time since end: {current_time - caption_end:.6f}s"
                        )
                        continue
                        
                    # Calculate time in caption and time until end
                    time_in_caption = current_time - caption_start
                    time_until_end = caption_end - current_time
                    
                    # Log rendering details at appropriate levels
                    if should_log_details:
                        logger.debug(
                            f"[RENDER] Rendering caption {i}/{len(active_captions)} | "
                            f"{time_in_caption:.3f}s / {caption_duration:.3f}s | "
                            f"'{current_caption['text'][:50]}{'...' if len(current_caption['text']) > 50 else ''}'"
                        )
                    
                    logger.trace(
                        f"[RENDER] Caption {i} timing | "
                        f"Start: {caption_start:.6f}s | "
                        f"Current: {current_time:.6f}s | "
                        f"End: {caption_end:.6f}s | "
                        f"Time in: {time_in_caption:.6f}s | "
                        f"Remaining: {time_until_end:.6f}s"
                    )
                    
                    # Log timing information for debugging
                    if should_log:
                        logger.info(
                            f"[OVERLAY] Processing caption: '{current_caption['text'][:30]}...'\n"
                            f"          Time in caption: {time_in_caption:.2f}s | "
                            f"Time remaining: {time_until_end:.2f}s"
                        )
                    
                    # Calculate fade in/out effects
                    fade_in_duration = min(0.3, (caption_end - caption_start) / 3)  # Up to 0.3s fade in
                    fade_out_duration = min(0.5, (caption_end - caption_start) / 3)  # Up to 0.5s fade out
                    
                    fade_factor = 1.0
                    
                    # Fade in at start
                    if time_in_caption < fade_in_duration and fade_in_duration > 0:
                        fade_factor = time_in_caption / fade_in_duration
                    # Fade out at end
                    elif time_until_end < fade_out_duration and fade_out_duration > 0:
                        fade_factor = time_until_end / fade_out_duration
                    
                    # Ensure fade factor is within valid range
                    fade_factor = max(0.1, min(1.0, fade_factor))
                    
                    if should_log:
                        logger.info(f"[OVERLAY] Fade factor: {fade_factor:.2f}")
                    
                    # Split text into lines
                    lines = current_caption['text'].split('\n')
                    
                    # Calculate total text block height
                    total_text_height = 0
                    line_heights = []
                    
                    for line in lines:
                        if not line.strip():
                            continue
                        (w, h), _ = cv2.getTextSize(
                            line, self.font, self.font_scale, self.font_thickness
                        )
                        line_heights.append(h)
                        total_text_height += h + 5  # Add some spacing between lines
                    
                    if not line_heights:
                        continue  # Skip if no valid lines
                    
                    # Add padding
                    total_text_height += self.padding * 2
                    
                    # Calculate text block position
                    bg_x1 = (frame_width - frame_width // 2) // 2  # Center in the middle half of the frame
                    bg_x2 = frame_width - bg_x1
                    bg_y1 = frame_height - total_text_height - self.y_offset
                    bg_y2 = frame_height - self.y_offset
                    
                    # Ensure background is within frame bounds
                    bg_y1 = max(0, bg_y1)
                    bg_y2 = min(frame_height, bg_y2)
                    
                    # Create overlay for this caption
                    overlay = result_frame.copy()
                    
                    # Draw background rectangle with fade effect
                    bg_color = list(self.bg_color)
                    if len(bg_color) == 3:  # If no alpha channel, add one
                        bg_color = bg_color + [128]  # Semi-transparent by default
                    
                    # Apply fade factor to alpha channel
                    bg_color[3] = int(bg_color[3] * fade_factor)
                    
                    # Convert to BGRA for OpenCV
                    bgra_bg_color = (bg_color[2], bg_color[1], bg_color[0], bg_color[3])
                    
                    # Create a transparent overlay
                    overlay = np.zeros_like(overlay, dtype=np.uint8)
                    
                    # Draw the background rectangle
                    cv2.rectangle(
                        overlay, 
                        (bg_x1, bg_y1), 
                        (bg_x2, bg_y2), 
                        bgra_bg_color, 
                        -1
                    )
                    
                    # Add the overlay to the result with transparency
                    alpha = bg_color[3] / 255.0
                    cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
                    
                    # Draw text
                    y = bg_y1 + self.padding
                    
                    # Track the lines we're going to display
                    display_lines = []
                    
                    for line, h in zip(lines, line_heights):
                        if not line.strip():
                            y += h + 5  # Still account for empty lines in spacing
                            continue
                            
                        display_lines.append(line)
                        
                        # Get text size for centering
                        (w, _), _ = cv2.getTextSize(
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
                except Exception as e:
                    logger.error(f"Error rendering caption: {str(e)}", exc_info=True)
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
            
            # Start timing the rendering
            render_start = time.time()
            
            # Get frame dimensions
            try:
                frame_height, frame_width = frame.shape[:2]
            except Exception as e:
                logger.error(f"Error getting frame dimensions: {str(e)}")
                return frame
                
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
        
        # Log rendering time in debug mode
        render_time = (time.time() - render_start) * 1000  # Convert to milliseconds
        if render_time > 16:  # Log warning if rendering takes more than 16ms (~60fps)
            logger.warning(f"Slow frame rendering: {render_time:.2f}ms")
        else:
            logger.debug(f"Frame rendered in {render_time:.2f}ms")
            
        return result_frame
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
