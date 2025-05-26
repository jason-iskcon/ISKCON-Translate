"""Video playback and synchronization loop for ISKCON-Translate."""
import time
import cv2
import queue
import threading
from logging_utils import get_logger

# Import singleton clock
try:
    from clock import CLOCK
except ImportError:
    from clock import CLOCK

# Import translation and comparison modules
try:
    from ..translation.translator import Translator
    from ..caption_overlay.comparison import ComparisonRenderer
except ImportError:
    from src.translation.translator import Translator
    from src.caption_overlay.comparison import ComparisonRenderer

logger = get_logger(__name__)

class VideoRunner:
    """Handles the main video playback and synchronization loop."""
    
    def __init__(self, video_source, transcriber, caption_overlay, window_name="Video with Captions", comparison_mode=False, youtube_url=None, headless=False):
        """Initialize the video runner.
        
        Args:
            video_source: VideoSource instance for frame retrieval
            transcriber: TranscriptionEngine instance for caption processing
            caption_overlay: CaptionOverlay instance for displaying captions
            window_name: Name of the display window
            comparison_mode: Whether to show side-by-side comparison
            youtube_url: Optional YouTube URL for comparison
            headless: Whether to run in headless mode (no window display)
        """
        self.video_source = video_source
        self.transcriber = transcriber
        self.caption_overlay = caption_overlay
        self.window_name = window_name
        self.frame_count = 0
        self.paused = False
        self.running = False
        self.comparison_mode = comparison_mode
        self.youtube_url = youtube_url
        self.headless = headless
        
        # Initialize translator and comparison renderer if in comparison mode
        if comparison_mode:
            self.translator = Translator()
            self.comparison_renderer = ComparisonRenderer()
            logger.info("Initialized comparison mode with translator and comparison renderer")
        
        # Get video information
        self.width, self.height, self.fps = video_source.get_video_info()
        self.target_frame_time = 1.0 / self.fps
        self.frame_buffer = []
        self.max_buffer_size = 30  # Increased buffer size for smoother playback
        
        # Initialize timing variables with high precision
        self.next_frame_time = time.perf_counter()
        self.last_frame_time = time.perf_counter()
        self.current_video_time = 0.0
        self.frame_times = []  # For frame timing analysis
        self.audio_sync_threshold = 0.033  # 33ms threshold for audio sync
        
        # For rate-limiting certain logs
        self._last_no_transcription_log_time = 0
        self._frames_since_last_slow_render_log = 0
        self._last_stats_log_time = 0  # For 5-second heartbeat stats
        
        # Create display window only if not in headless mode
        if not self.headless:
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
    
    def _sync_with_audio(self, frame_time: float) -> float:
        """Synchronize video frame time with audio playback.
        
        Args:
            frame_time: Current frame timestamp
            
        Returns:
            Adjusted frame time
        """
        with self.video_source.audio_position_lock:
            audio_time = self.video_source.audio_position
            
        if audio_time > 0 and self.video_source.audio_playing:
            time_diff = frame_time - audio_time
            # Only adjust if the difference is significant
            if abs(time_diff) > self.audio_sync_threshold:
                # Use audio time directly for better sync
                logger.debug(f"Audio sync adjustment: {time_diff:.3f}s")
                return audio_time
        return frame_time
    
    def _process_frame(self, frame_data):
        """Process a single frame and apply captions."""
        frame, frame_timestamp = frame_data
        
        # Get current audio position for synchronization
        with self.video_source.audio_position_lock:
            current_audio_time = self.video_source.audio_position
        
        # Use audio time for synchronization if available
        self.current_video_time = self._sync_with_audio(current_audio_time if self.video_source.audio_playing else frame_timestamp)
        
        # Calculate time delta and next frame time using high-precision timer
        now = time.perf_counter()
        frame_delay = now - self.last_frame_time
        self.last_frame_time = now
        
        # Simple frame timing without complex adjustments
        self.next_frame_time = now + self.target_frame_time
        
        # Track frame timing for analysis (keep smaller buffer)
        self.frame_times.append(frame_delay)
        if len(self.frame_times) > 30:  # Keep last 30 frames only
            self.frame_times.pop(0)
        
        # Get any available transcriptions (less frequently)
        if self.frame_count % 3 == 0:  # Only check every 3rd frame
            self._process_transcriptions()
        
        # Calculate relative time with seek offset
        relative_time = self.current_video_time - self.video_source.start_time
        
        # Process frame based on mode
        if self.comparison_mode:
            frame = self._process_comparison_frame(frame, relative_time)
        else:
            frame = self._process_normal_frame(frame, relative_time)
        
        return frame
    
    def _process_transcriptions(self):
        """Process any available transcriptions from the queue."""
        while True:  # Process all available transcriptions
            try:
                transcription = self.transcriber.get_transcription()
                if not transcription:
                    current_time = time.time()
                    if current_time - self._last_no_transcription_log_time > 0.5:
                        logger.trace("No more transcriptions in queue")
                        self._last_no_transcription_log_time = current_time
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
                
            start_time = transcription['start']
            end_time = transcription.get('end', start_time + 3.0)  # Default 3s duration
            duration = end_time - start_time
            
            # Calculate relative timestamps by subtracting seek offset
            rel_start = start_time - self.video_source.start_time
            rel_end = end_time - self.video_source.start_time
            
            # Track latest audio time for heartbeat calculation
            self._latest_audio_rel = rel_start
            
            # Get current video relative time for validation
            current_video_time = self.current_video_time - self.video_source.start_time
            logger.info(f"[TRANSCRIPTION] Received: '{text}' at rel_start={rel_start:.2f}s")
            logger.info(f"[TIMING] Caption timing: rel_start={rel_start:.2f}s, current_video_time={current_video_time:.2f}s")
            
            # Validate the timestamp is within reasonable bounds (allow captions from recent past and near future)
            time_diff = rel_start - current_video_time
            if time_diff < -30.0 or time_diff > 60.0:  # Allow 30s past, 60s future
                logger.warning(f"[TIMING] Caption timing out of bounds ({time_diff:.2f}s difference), dropping")
                return
            
            # Add the caption with relative timestamps
            logger.info(f"[CAPTION] Adding: {text!r} at rel_start={rel_start:.2f}s for {duration:.1f}s")
            try:
                self.caption_overlay.add_caption(text, rel_start, rel_end)
                logger.info("[CAPTION] Added successfully")
            except Exception as e:
                logger.error(f"Error adding caption: {e}")
                
        except Exception as e:
            logger.error(f"Error handling transcription: {e}")
    
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
    
    def _process_comparison_frame(self, frame, relative_time):
        """Process frame in comparison mode.
        
        Args:
            frame: Video frame to process
            relative_time: Current relative time
            
        Returns:
            Processed frame with side-by-side captions
        """
        # Get active captions
        active_captions = self.caption_overlay.get_active_captions(relative_time)
        
        if not active_captions:
            return frame
        
        # Get the most recent caption
        current_caption = active_captions[-1]
        
        # Get both YouTube and Parakletos captions
        youtube_text, parakletos_text = self.translator.translate(
            current_caption['text'],
            timestamp=relative_time,
            video_url=self.youtube_url
        )
        
        # Create caption dictionaries for comparison
        youtube_caption = {
            'text': youtube_text,
            'start_time': current_caption['start_time'],
            'end_time': current_caption['end_time']
        }
        
        parakletos_caption = {
            'text': parakletos_text,
            'start_time': current_caption['start_time'],
            'end_time': current_caption['end_time']
        }
        
        # Render side-by-side comparison
        return self.comparison_renderer.render_comparison(
            frame,
            youtube_caption,
            parakletos_caption,
            relative_time
        )
    
    def _process_normal_frame(self, frame, relative_time):
        """Process frame in normal mode.
        
        Args:
            frame: Video frame to process
            relative_time: Current relative time
            
        Returns:
            Processed frame with captions
        """
        return self.caption_overlay.overlay_captions(
            frame=frame,
            current_time=relative_time,
            frame_count=self.frame_count
        )
    
    def _log_timing_stats(self):
        """Log detailed timing statistics."""
        if not self.frame_times:
            return
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        max_frame_time = max(self.frame_times)
        min_frame_time = min(self.frame_times)
        
        logger.info("\n=== TIMING STATS ===")
        logger.info(f"Average frame time: {avg_frame_time*1000:.2f}ms")
        logger.info(f"Min frame time: {min_frame_time*1000:.2f}ms")
        logger.info(f"Max frame time: {max_frame_time*1000:.2f}ms")
        logger.info(f"Target frame time: {self.target_frame_time*1000:.2f}ms")
        
        # Log audio sync status
        with self.video_source.audio_position_lock:
            audio_time = self.video_source.audio_position
        logger.info(f"Current audio time: {audio_time:.3f}s")
        logger.info(f"Current video time: {self.current_video_time:.3f}s")
        logger.info(f"Sync difference: {abs(audio_time - self.current_video_time)*1000:.2f}ms")
    
    def run(self):
        """Run the main video playback and synchronization loop."""
        try:
            # Initialize timing variables
            start_time = time.perf_counter()
            frame_count = 0
            target_fps = self.video_source.fps
            frame_duration = 1.0 / target_fps
            
            while True:
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
                expected_frame = int(elapsed_time * target_fps)
                
                # Skip frames if we're behind
                if frame_count < expected_frame:
                    # Get the next frame
                    frame_data = self.video_source.get_frame()
                    if frame_data is None:
                        break
                    
                    # Process the frame
                    frame = self._process_frame(frame_data)
                    
                    # Display the frame only if not in headless mode
                    if not self.headless:
                        cv2.imshow(self.window_name, frame)
                    
                    frame_count += 1
                    self.frame_count = frame_count
                    
                    # Log timing info less frequently
                    if frame_count % 90 == 0:  # Every 3 seconds at 30fps
                        self._log_timing_stats()
                    
                    # Handle key press only if not in headless mode
                    if not self.headless and cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Sleep for a short time to avoid busy waiting
                    time.sleep(0.001)  # 1ms sleep
            
        finally:
            if not self.headless:
                cv2.destroyAllWindows()
            self.video_source.release()
