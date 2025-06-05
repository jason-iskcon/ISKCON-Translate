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
    from ..text_processing import CrossSegmentDetector, ProfanityFilter, RepetitionDetector
except ImportError:
    from src.translation.translator import Translator
    from src.caption_overlay.comparison import ComparisonRenderer
    from src.text_processing import CrossSegmentDetector, ProfanityFilter, RepetitionDetector

logger = get_logger(__name__)

class VideoRunner:
    """Handles the main video playback and synchronization loop."""
    
    def __init__(self, video_source, transcriber, caption_overlay, window_name="Video with Captions", comparison_mode=False, youtube_url=None, headless=False):
        """Initialize video runner.
        
        Args:
            video_source: VideoSource instance
            transcriber: TranscriptionEngine instance  
            caption_overlay: CaptionOverlayOrchestrator instance
            window_name: Name for OpenCV window
            comparison_mode: Whether to show side-by-side translation comparison
            youtube_url: YouTube URL for comparison mode
            headless: Whether to run without GUI
        """
        self.video_source = video_source
        self.transcriber = transcriber
        self.caption_overlay = caption_overlay
        self.window_name = window_name
        self.comparison_mode = comparison_mode
        self.youtube_url = youtube_url
        self.headless = headless
        
        # Initialize comparison mode components if needed
        if self.comparison_mode:
            self.translator = Translator()
            self.comparison_renderer = ComparisonRenderer()
        
        # Frame timing and statistics
        self.frame_count = 0
        self.frame_times = []
        self.target_frame_time = 1.0 / self.video_source.fps
        self.last_frame_time = time.perf_counter()
        self.next_frame_time = 0
        
        # Current timing
        self.current_video_time = 0.0
        
        # Audio sync threshold - make it tighter for better sync
        self.audio_sync_threshold = 0.016  # ~1 frame at 60fps for tighter sync
        
        # Performance tracking
        self._latest_audio_rel = 0.0
        self._last_no_transcription_log_time = 0.0
        
        # Caption timing offset for better sync - reduce early timing for better audio sync
        self.caption_timing_offset = -0.050  # 50ms early for better lip sync (reduced from 100ms)
        
        # Initialize text processing components
        try:
            # Import text processing classes
            from ..text_processing.cross_segment_detector import CrossSegmentDetector
            from ..text_processing.profanity_filter import ProfanityFilter  
            from ..text_processing.repetition_detector import RepetitionDetector
            
            # Initialize text processing pipeline
            self.cross_segment_detector = CrossSegmentDetector(
                window_size=5,
                overlap_threshold=0.3,
                similarity_threshold=0.7
            )
            
            self.profanity_filter = ProfanityFilter(
                level='moderate',
                strategy='beep'
            )
            
            self.repetition_detector = RepetitionDetector(
                max_word_reps=3,
                max_phrase_len=5,
                threshold=0.8
            )
            
            logger.info("Initialized text processing components for caption enhancement")
            
        except ImportError as e:
            logger.warning(f"Could not import text processing components: {e}")
            # Use dummy processors that just pass through text
            class DummyProcessor:
                def process_segment(self, text, timestamp):
                    from dataclasses import dataclass
                    from typing import List, Dict
                    @dataclass
                    class Result:
                        cleaned_text: str
                        duplications_found: List[Dict] = None
                        action_taken: str = "no processing"
                    return Result(cleaned_text=text, duplications_found=[])
                
                def detect_and_remove_repetitions(self, text):
                    from dataclasses import dataclass
                    from typing import List
                    @dataclass
                    class Result:
                        cleaned_text: str
                        repetitions_found: List = None
                    return Result(cleaned_text=text, repetitions_found=[])
                
                def filter_text(self, text):
                    from dataclasses import dataclass
                    from typing import List
                    @dataclass
                    class Result:
                        filtered_text: str
                        detections: List = None
                    return Result(filtered_text=text, detections=[])
            
            self.cross_segment_detector = DummyProcessor()
            self.profanity_filter = DummyProcessor()
            self.repetition_detector = DummyProcessor()
        
        # Get video information and log initialization
        width, height, fps = video_source.get_video_info()
        
        # Create display window with proper sizing if not in headless mode
        if not self.headless:
            # Calculate display size to fit screen (max 1280x720 for reasonable viewing)
            max_display_width = 1280
            max_display_height = 720
            
            # Calculate scale factor to fit within max display size
            scale_width = max_display_width / width
            scale_height = max_display_height / height
            scale_factor = min(scale_width, scale_height, 1.0)  # Don't upscale
            
            self.display_width = int(width * scale_factor)
            self.display_height = int(height * scale_factor)
            
            # Create named window that can be resized
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
            
            logger.info(f"Video display: {width}x{height} -> {self.display_width}x{self.display_height} (scale: {scale_factor:.2f})")
        else:
            self.display_width = width
            self.display_height = height
        
        logger.info(f"Video runner initialized: {width}x{height} @ {self.video_source.fps}fps")
    
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
            # Use tighter sync threshold for better lip sync
            if abs(time_diff) > self.audio_sync_threshold:
                # Use audio time directly for better sync
                logger.debug(f"Audio sync adjustment: {time_diff:.3f}s -> using audio time")
                return audio_time
        return frame_time
    
    def _process_frame(self, frame_data):
        """Process a single frame and apply captions."""
        frame, frame_timestamp = frame_data
        
        # Get current audio position for synchronization
        with self.video_source.audio_position_lock:
            current_audio_time = self.video_source.audio_position
        
        # Use audio time for better synchronization
        if self.video_source.audio_playing and current_audio_time > 0:
            # Use audio time directly for most accurate sync
            self.current_video_time = current_audio_time
        else:
            # Fall back to frame timestamp if no audio
            self.current_video_time = frame_timestamp
        
        # Calculate frame timing for performance monitoring
        now = time.perf_counter()
        frame_delay = now - self.last_frame_time
        self.last_frame_time = now
        
        # Track frame timing for analysis (keep smaller buffer)
        self.frame_times.append(frame_delay)
        if len(self.frame_times) > 30:  # Keep last 30 frames only
            self.frame_times.pop(0)
        
        # Process transcriptions less frequently to improve performance
        if self.frame_count % 5 == 0:  # Check every 5th frame for much better performance
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
            end_time = transcription.get('end', start_time + 2.0)
            duration = end_time - start_time
            
            # Ensure reasonable duration for readability but cap for responsiveness
            duration = max(1.0, min(duration, 3.0))  # Between 1.0s and 3.0s for balance
            
            # Calculate relative timestamps by subtracting seek offset
            rel_start = start_time - self.video_source.start_time
            
            # Apply caption timing offset for better lip sync
            rel_start_adjusted = rel_start + self.caption_timing_offset
            
            # Track latest audio time for heartbeat calculation
            self._latest_audio_rel = rel_start
            
            # Get current video relative time for validation
            current_video_time = self.current_video_time - self.video_source.start_time
            
            # Reduce logging frequency for better performance
            if self.frame_count % 30 == 0:  # Only log transcription details every 30 frames
                logger.info(f"[TRANSCRIPTION] Received: '{text}' at rel_start={rel_start:.2f}s")
                logger.info(f"[TIMING] Caption timing: rel_start={rel_start:.2f}s -> adjusted={rel_start_adjusted:.2f}s, current_video_time={current_video_time:.2f}s")
            
            # Validate the timestamp is within reasonable bounds (allow captions from recent past and near future)
            time_diff = rel_start - current_video_time
            if time_diff < -30.0 or time_diff > 60.0:  # Allow 30s past, 60s future
                logger.warning(f"[TIMING] Caption timing out of bounds ({time_diff:.2f}s difference), dropping")
                return
            
            # Apply text processing to enhance caption quality
            processed_text = self._process_caption_text(text, rel_start)
            
            # Skip empty captions after processing
            if not processed_text.strip():
                logger.debug(f"[TEXT_PROCESSING] Caption removed after processing: '{text}'")
                return
            
            # Add the caption with adjusted timing for better lip sync
            if self.frame_count % 30 == 0:  # Only log caption addition details every 30 frames
                logger.info(f"[CAPTION] Adding: {processed_text!r} at rel_start={rel_start_adjusted:.2f}s for {duration:.1f}s")
                if processed_text != text:
                    logger.debug(f"[TEXT_PROCESSING] Original: '{text}' -> Processed: '{processed_text}'")
            
            try:
                # Use adjusted timing for better lip sync
                self.caption_overlay.add_caption(processed_text, rel_start_adjusted, duration, is_absolute=False)
                if self.frame_count % 30 == 0:  # Only log success every 30 frames
                    logger.info("[CAPTION] Added successfully")
            except Exception as e:
                logger.error(f"Error adding caption: {e}")
                
        except Exception as e:
            logger.error(f"Error handling transcription: {e}")
    
    def _process_caption_text(self, text: str, timestamp: float) -> str:
        """
        Process caption text through the text enhancement pipeline.
        
        Args:
            text: Original caption text
            timestamp: Caption timestamp
            
        Returns:
            Processed and enhanced caption text
        """
        try:
            # Step 1: Cross-segment duplication detection (most important for the current issue)
            cross_segment_result = self.cross_segment_detector.process_segment(text, timestamp)
            processed_text = cross_segment_result.cleaned_text
            
            if cross_segment_result.duplications_found:
                logger.info(f"[CROSS_SEGMENT] {cross_segment_result.action_taken}: "
                           f"{len(cross_segment_result.duplications_found)} duplications removed")
                for dup in cross_segment_result.duplications_found:
                    logger.debug(f"  - {dup['type']}: confidence={dup['confidence']:.2f}")
            
            # Step 2: Remove repetitions within the text
            if processed_text.strip():
                rep_result = self.repetition_detector.detect_and_remove_repetitions(processed_text)
                processed_text = rep_result.cleaned_text
                
                if rep_result.repetitions_found:
                    logger.debug(f"[REPETITION] Removed {len(rep_result.repetitions_found)} repetitions")
            
            # Step 3: Filter profanity (for family-friendly ISKCON content)
            if processed_text.strip():
                prof_result = self.profanity_filter.filter_text(processed_text)
                processed_text = prof_result.filtered_text
                
                if prof_result.detections:
                    logger.debug(f"[PROFANITY] Filtered {len(prof_result.detections)} items")
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error processing caption text: {e}")
            return text  # Return original text if processing fails
    
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
            
            # Optimize frame timing for smoother playback
            skip_threshold = 0  # Skip frames only when absolutely necessary for smooth playback
            max_skip_per_loop = 2  # Limit skips to maintain smooth video quality
            
            while True:
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
                expected_frame = int(elapsed_time * target_fps)
                
                # Calculate how many frames we're behind
                frames_behind = expected_frame - frame_count
                
                # Only skip frames if we're significantly behind (more conservative)
                if frames_behind > skip_threshold:
                    # Skip frames by getting multiple frames quickly
                    skipped = 0
                    while skipped < frames_behind and skipped < max_skip_per_loop:  # Limit skips for smooth video
                        frame_data = self.video_source.get_frame()
                        if frame_data is None:
                            break
                        skipped += 1
                        frame_count += 1
                    
                    if skipped > 0:
                        logger.debug(f"Skipped {skipped} frames for smooth playback")
                
                # Process the current frame
                if frame_count <= expected_frame:
                    # Get the next frame
                    frame_data = self.video_source.get_frame()
                    if frame_data is None:
                        break
                    
                    # Process the frame
                    frame = self._process_frame(frame_data)
                    
                    # Display the frame only if not in headless mode
                    if not self.headless:
                        # Resize frame for display if needed (optimize by checking less frequently)
                        if hasattr(self, 'display_width') and hasattr(self, 'display_height'):
                            frame_height, frame_width = frame.shape[:2]
                            if frame_width != self.display_width or frame_height != self.display_height:
                                # Use faster interpolation for better performance
                                frame = cv2.resize(frame, (self.display_width, self.display_height), interpolation=cv2.INTER_LINEAR)
                        
                        cv2.imshow(self.window_name, frame)
                    
                    frame_count += 1
                    self.frame_count = frame_count
                    
                    # Log timing info much less frequently for better performance
                    if frame_count % 300 == 0:  # Every 10 seconds at 30fps instead of every 3 seconds
                        self._log_timing_stats()
                    
                    # Handle key press only if not in headless mode
                    if not self.headless and cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Sleep for a very short time to avoid busy waiting
                    time.sleep(0.001)  # 1ms sleep
            
        finally:
            if not self.headless:
                cv2.destroyAllWindows()
            self.video_source.release()
