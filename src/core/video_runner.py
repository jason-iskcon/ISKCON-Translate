"""Video playback and synchronization loop for ISKCON-Translate."""
import time
import cv2
import queue
from ..logging_utils import get_logger

# Import singleton clock
try:
    from ..clock import CLOCK
except ImportError:
    from clock import CLOCK

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
        
        # For rate-limiting certain logs
        self._last_no_transcription_log_time = 0
        self._frames_since_last_slow_render_log = 0
        self._last_stats_log_time = 0  # For 5-second heartbeat stats
        
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
        
        # Debug log for playback_start_time
        if self.frame_count == 1:
            logger.info(f"🔧 FIRST FRAME: playback_start_time = {self.video_source.playback_start_time}")
        
        # Calculate relative time with safeguard
        if self.video_source.playback_start_time > 0:
            # Use singleton clock for consistent timing
            relative_time = CLOCK.get_video_relative_time()
            
            # Debug log for seek-aware timing
            if self.frame_count == 1:
                elapsed_time = CLOCK.get_elapsed_time()
                logger.info(f"🔧 SINGLETON TIMING: media_seek_pts={CLOCK.media_seek_pts:.2f}s, elapsed={elapsed_time:.2f}s, video_rel_time={relative_time:.2f}s")
        else:
            # Fallback: initialize playback_start_time if it's still 0
            logger.warning("playback_start_time is 0, using singleton clock fallback")
            if CLOCK.is_initialized():
                relative_time = CLOCK.get_video_relative_time()
            else:
                relative_time = 0.0
        
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
                
            start_time = transcription['timestamp']
            end_time = transcription.get('end_time', start_time + 3.0)  # Default 3s duration
            duration = end_time - start_time
            
            # Timestamps are now elapsed-time relative (0-N seconds since playback started)
            # Use directly for caption overlay without conversion
            rel_start = start_time
            rel_end = end_time
            
            # Track latest audio time for heartbeat calculation
            self._latest_audio_rel = rel_start
            
            # Get current elapsed time for validation using singleton clock
            current_elapsed = CLOCK.get_elapsed_time()
            logger.info(f"[TRANSCRIPTION] Received: '{text}' at rel_start={rel_start:.2f}s")
            logger.info(f"[TIMING] Caption timing: rel_start={rel_start:.2f}s, current_elapsed={current_elapsed:.2f}s")
            
            # Validate the timestamp is within reasonable bounds
            time_diff = abs(rel_start - current_elapsed)
            if time_diff > 15.0:
                logger.warning(f"[TIMING] Caption timing mismatch ({time_diff:.2f}s difference), dropping")
                return
            
            # Add the caption with elapsed-time relative timestamps (no conversion needed)
            logger.info(f"[CAPTION] Adding: {text!r} at rel_start={rel_start:.2f}s for {duration:.1f}s")
            try:
                self.caption_overlay.add_caption(
                    text,
                    timestamp=rel_start,  # Use relative time directly
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
        frame_render_times = []
        max_render_time_samples = 100  # Sample last 100 frames for avg render time
        
        try:
            while self.running and self.video_source.is_running:
                loop_start_time = time.time()
                
                # Handle pause state
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Get frame from buffer or video source
                frame_data = None
                if self.frame_buffer:
                    frame_data = self.frame_buffer.pop(0)
                elif self.video_source.is_running:
                    frame_data = self.video_source.get_frame()
                
                if frame_data:
                    processed_frame = self.process_frame(frame_data)
                    cv2.imshow(self.window_name, processed_frame)
                    
                    # Maintain frame buffer
                    if len(self.frame_buffer) < self.max_buffer_size and self.video_source.is_running:
                        new_frame = self.video_source.get_frame()
                        if new_frame:
                            self.frame_buffer.append(new_frame)
                            
                    # Frame rendering time calculation
                    render_duration = time.time() - loop_start_time
                    frame_render_times.append(render_duration)
                    if len(frame_render_times) > max_render_time_samples:
                        frame_render_times.pop(0)
                    
                    avg_render_time = sum(frame_render_times) / len(frame_render_times) if frame_render_times else 0
                    
                    # Log slow frame rendering
                    if render_duration > 0.033:  # Threshold to 33ms (approx 30 FPS)
                        self._frames_since_last_slow_render_log += 1
                        if self._frames_since_last_slow_render_log >= 100:
                            logger.warning(
                                f"Slow frame rendering: {render_duration*1000:.0f}ms "
                                f"(avg: {avg_render_time*1000:.0f}ms, target: {self.target_frame_time*1000:.0f}ms)"
                            )
                            self._frames_since_last_slow_render_log = 0
                    else:
                        # Reset counter if rendering is fast enough
                        self._frames_since_last_slow_render_log = 0
                else:
                    # No frame available, might be end of video or buffering issue
                    logger.debug("No frame available from video source.")
                    if not self.video_source.is_running and not self.frame_buffer:
                        logger.info("End of video or video source stopped.")
                        break # Exit loop if source stopped and buffer empty
                    time.sleep(0.005) # Brief pause if no frame but still running

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # If a key was pressed
                    self._handle_key_press(key)
                
                # Adaptive sleep to reduce CPU usage
                sleep_time = max(0, self.next_frame_time - time.time() - 0.001)
                if sleep_time > 0.001:
                    time.sleep(min(0.01, sleep_time))
                
                # Log total loop iteration time periodically
                loop_duration = time.time() - loop_start_time
                if self.frame_count % 30 == 0: # Log every second at 30fps
                    logger.debug(f"[TIMING] VideoRunner.run loop iteration took {loop_duration*1000:.2f}ms for frame {self.frame_count}")
                
                # 5-second heartbeat stats logging
                current_time = time.time()
                if current_time - self._last_stats_log_time >= 2.0:  # Reduced to 2s for testing
                    # Calculate current FPS
                    elapsed_time = current_time - self._last_stats_log_time if self._last_stats_log_time > 0 else 2.0
                    frames_in_period = 30 * 2 if self._last_stats_log_time > 0 else self.frame_count  # Rough estimate
                    current_fps = frames_in_period / elapsed_time if elapsed_time > 0 else 0
                    
                    # Get queue sizes
                    frame_q_size = self.video_source.frames_queue.qsize()
                    frame_q_max = self.video_source.frames_queue.maxsize
                    audio_q_size = self.transcriber.audio_queue.qsize()
                    audio_q_max = self.transcriber.audio_queue.maxsize
                    
                    # Calculate A/V drift - compare like-with-like elapsed times
                    # Video time: elapsed since playback start using singleton clock
                    video_rel_time = CLOCK.get_elapsed_time()
                    
                    # Audio time: convert transcription timestamp to elapsed time for proper comparison
                    # _latest_audio_rel contains media-relative timestamps, need to convert to elapsed
                    if hasattr(self, '_latest_audio_rel') and self._latest_audio_rel >= 0:
                        # Convert audio media-relative time to elapsed time
                        # Since transcriptions are now using elapsed time directly, use as-is
                        audio_rel_time = self._latest_audio_rel
                    else:
                        # No valid audio data yet, assume audio matches video to avoid scary drift
                        audio_rel_time = video_rel_time
                        if not hasattr(self, '_logged_audio_init'):
                            logger.debug("[HEARTBEAT] No audio transcriptions yet, using video_rel for drift calculation")
                            self._logged_audio_init = True
                    
                    # Calculate drift: positive means audio ahead of video, negative means video ahead
                    av_drift = audio_rel_time - video_rel_time
                    # Clamp to reasonable values to avoid display issues
                    av_drift = max(-999.0, min(999.0, av_drift))
                    
                    # Get consecutive drops from video source
                    consecutive_drops = getattr(self.video_source, '_consecutive_drops', 0)
                    
                    # Enhanced heartbeat with corrected drift calculation
                    logger.info(f"📊 [HB] fps={current_fps:.1f} | frame_q={frame_q_size}/{frame_q_max} | audio_q={audio_q_size}/{audio_q_max} | v_rel={video_rel_time:.1f} | a_rel={audio_rel_time:.1f} | drift={av_drift:+.2f}s | drops={consecutive_drops}")
                    self._last_stats_log_time = current_time
                    
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
