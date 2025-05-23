"""Video playback and synchronization loop for ISKCON-Translate."""
import time
import cv2
import queue
from ..logging_utils import get_logger

logger = get_logger(__name__)

class VideoRunner:
    """Handles the main video playback and synchronization loop."""
    
    def __init__(self, video_source, transcriber, caption_overlay, window_name="Video with Captions"):
        """Initialize the video runner.
        
        Args:
            video_source: VideoSource instance for frame retrieval
            transcriber: TranscriptionEngine instance for caption processing
            caption_overlay: CaptionOverlay instance for displaying captions
            window_name: Name of the display window
        """
        self.video_source = video_source
        self.transcriber = transcriber
        self.caption_overlay = caption_overlay
        self.window_name = window_name
        self.frame_count = 0
        self.paused = False
        self.running = False
        
        # Get video information
        self.width, self.height, self.fps = video_source.get_video_info()
        self.target_frame_time = 1.0 / self.fps
        self.frame_buffer = []
        self.max_buffer_size = 30  # Increased buffer size for smoother playback
        
        # Initialize timing variables
        self.next_frame_time = time.time()
        self.last_frame_time = time.time()
        self.current_video_time = 0.0
        
        # Create display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        
        logger.info(f"Video runner initialized: {self.width}x{self.height} @ {self.fps}fps")
    
    def prebuffer_frames(self):
        """Pre-buffer frames before starting playback."""
        logger.info("Pre-buffering frames")
        while len(self.frame_buffer) < self.max_buffer_size // 2:
            frame_data = self.video_source.get_frame()
            if frame_data is not None:
                self.frame_buffer.append(frame_data)
            else:
                time.sleep(0.01)
        
        logger.info(f"Starting playback with {len(self.frame_buffer)} frames in buffer")
    
    def process_frame(self, frame_data):
        """Process a single frame and apply captions.
        
        Args:
            frame_data: Tuple of (frame, timestamp)
            
        Returns:
            Frame with captions applied
        """
        frame, frame_timestamp = frame_data
        
        # Get current audio position for synchronization
        with self.video_source.audio_position_lock:
            current_audio_time = self.video_source.audio_position
        
        # Use audio time for synchronization if available
        self.current_video_time = current_audio_time if self.video_source.audio_playing else frame_timestamp
        self.frame_count += 1
        
        # Calculate time delta and next frame time
        now = time.time()
        frame_delay = now - self.last_frame_time
        self.last_frame_time = now
        self.next_frame_time = now + max(0, self.target_frame_time - frame_delay)
        
        # Get any available transcriptions
        self._process_transcriptions()
        
        # Apply captions to frame using relative time and frame count
        frame_copy = frame.copy()
        relative_time = time.time() - self.video_source.playback_start_time
        frame_with_captions = self.caption_overlay.overlay_captions(
            frame=frame_copy,
            current_time=relative_time,
            frame_count=self.frame_count
        )
        
        # Add debug information
        self._add_debug_info(frame_with_captions, relative_time)
        
        return frame_with_captions
    
    def _process_transcriptions(self):
        """Process any available transcriptions from the queue."""
        while True:  # Process all available transcriptions
            try:
                transcription = self.transcriber.get_transcription()
                if not transcription:
                    logger.debug("No more transcriptions in queue")
                    break
                
                self._handle_transcription(transcription)
                
            except queue.Empty:
                break
    
    def _handle_transcription(self, transcription):
        """Handle a single transcription result."""
        try:
            text = transcription['text'].strip()
            if not text:
                return
                
            start_time = transcription['timestamp']
            end_time = transcription.get('end_time', start_time + 3.0)  # Default 3s duration
            duration = end_time - start_time
            
            # Convert to relative time if needed
            video_start_time = self.video_source.playback_start_time
            relative_start = start_time - video_start_time
            
            # Log the transcription with current video time for debugging
            logger.info(f"[TRANSCRIPTION] Received: '{text}' at {start_time:.2f}s (video: {self.video_source.get_current_time():.2f}s)")
            
            # Validate the timestamp is within reasonable bounds
            if relative_start < 0:
                logger.warning(f"[TIMING] Negative relative timestamp {relative_start:.2f}s, adjusting to 0")
                relative_start = 0
            
            # Add the caption with relative time
            logger.info(f"[CAPTION] Adding: {text!r} at {relative_start:.2f}s for {duration:.1f}s")
            try:
                self.caption_overlay.add_caption(
                    text,
                    timestamp=relative_start,
                    duration=duration,
                    is_absolute=False  # Using relative timestamps
                )
                logger.info("[CAPTION] Added successfully")
            except Exception as e:
                logger.error(f"Error adding caption: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Error processing transcription: {e}", exc_info=True)
    
    def _add_debug_info(self, frame, relative_time):
        """Add debug information to the frame."""
        # Frame number and relative time
        debug_text = f"Frame: {self.frame_count}, RelTime: {relative_time:.2f}s"
        cv2.putText(
            frame,
            debug_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),  # Red text for better visibility
            2,  # Thicker text
            cv2.LINE_AA
        )
        
        # Active caption count
        active_caption_count = len([
            c for c in self.caption_overlay.captions 
            if c['start_time'] <= relative_time <= c['end_time']
        ])
        debug_time_text = f"Captions: {active_caption_count}"
        cv2.putText(
            frame,
            debug_time_text,
            (10, 60),  # Positioned below the frame counter
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),  # Yellow text
            2,
            cv2.LINE_AA
        )
        
        # Log detailed debug info periodically
        if self.frame_count % 30 == 0:  # ~1 second at 30fps
            self._log_debug_info(relative_time)
    
    def _log_debug_info(self, relative_time):
        """Log detailed debug information."""
        logger.info("\n=== FRAME TIMING ===")
        logger.info(f"System time: {time.time()}")
        logger.info(f"Video start time: {self.video_source.playback_start_time}")
        logger.info(f"Current relative time: {relative_time:.2f}s")
        logger.info(f"Frame: {self.frame_count}")
        
        # Log all captions and their timing
        if hasattr(self.caption_overlay, 'captions') and self.caption_overlay.captions:
            logger.info("\n=== CAPTIONS ===")
            for i, cap in enumerate(self.caption_overlay.captions):
                active = "ACTIVE" if cap['start_time'] <= relative_time <= cap['end_time'] else "     "
                logger.info(f"{active} [{i}] '{cap['text'][:30]}...' | Start: {cap['start_time']:.2f}s | End: {cap['end_time']:.2f}s | Now: {relative_time:.2f}s")
        else:
            logger.info("\n=== NO CAPTIONS LOADED ===")
    
    def run(self):
        """Run the main video playback loop."""
        self.running = True
        logger.info("Starting video playback loop")
        
        try:
            while self.running:
                current_time = time.time()
                
                # Get more frames if buffer is low (in a non-blocking way)
                if len(self.frame_buffer) < self.max_buffer_size // 2:
                    try:
                        frame_data = self.video_source.get_frame()
                        if frame_data is not None:
                            self.frame_buffer.append(frame_data)
                    except Exception as e:
                        logger.warning(f"Error getting frame: {e}")
                
                # Process frame when it's time
                if not self.paused and self.frame_buffer and current_time >= self.next_frame_time:
                    frame_data = self.frame_buffer.pop(0)
                    
                    try:
                        frame_with_captions = self.process_frame(frame_data)
                        cv2.imshow(self.window_name, frame_with_captions)
                    except Exception as e:
                        logger.warning(f"Error processing frame: {e}")
                
                # Check for key presses (blocking with 1ms timeout)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # If a key was pressed
                    self._handle_key_press(key)
                
                # Adaptive sleep to reduce CPU usage
                sleep_time = max(0, self.next_frame_time - time.time() - 0.001)
                if sleep_time > 0.001:
                    time.sleep(min(0.01, sleep_time))
                    
        except KeyboardInterrupt:
            logger.info("Playback interrupted by user")
        except Exception as e:
            logger.error(f"Error in video playback: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def _handle_key_press(self, key):
        """Handle keyboard input."""
        if key == ord('q') or key == 27:  # 'q' or ESC
            logger.info("Quit requested by user")
            self.running = False
        elif key == ord('p') or key == 32:  # 'p' or SPACE
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            logger.info(f"Playback {status}")
            if not self.paused:
                self.next_frame_time = time.time()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info(f"Playback ended after {self.frame_count} frames")
        cv2.destroyAllWindows()
        self.running = False
