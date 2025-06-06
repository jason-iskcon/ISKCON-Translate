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
    
    def __init__(self, video_source, transcriber, caption_overlay, window_name="Video with Captions", comparison_mode=False, youtube_url=None, headless=False, secondary_languages=None, primary_language="en"):
        """Initialize video runner.
        
        Args:
            video_source: VideoSource instance
            transcriber: TranscriptionEngine instance  
            caption_overlay: CaptionOverlayOrchestrator instance
            window_name: Name for OpenCV window
            comparison_mode: Whether to show side-by-side translation comparison
            youtube_url: YouTube URL for comparison mode
            headless: Whether to run without GUI
            secondary_languages: List of secondary language codes for real-time translation
            primary_language: Primary language specified by user (will be translated from English)
        """
        self.video_source = video_source
        self.transcriber = transcriber
        self.caption_overlay = caption_overlay
        self.window_name = window_name
        self.comparison_mode = comparison_mode
        self.youtube_url = youtube_url
        self.headless = headless
        self.secondary_languages = secondary_languages or []
        self.primary_language = primary_language
        
        # Playback control
        self.running = True
        self.paused = False
        self.pause_start_time = 0.0
        self.total_pause_duration = 0.0
        
        # Restore translation functionality with performance optimizations
        # Re-enable target languages but keep processing optimized
        
        # Create list of all target languages for translation
        self.target_languages = []
        
        # Add primary language if not English (and not already in the list)
        if primary_language != "en":
            self.target_languages.append(primary_language)
        
        # Add secondary languages (only if not already in the list)
        if self.secondary_languages:
            for sec_lang in self.secondary_languages:
                if sec_lang not in self.target_languages and sec_lang != "en":
                    self.target_languages.append(sec_lang)
        
        # Log exactly what target languages we have
        logger.info(f"🎯 TARGET LANGUAGES SETUP: primary='{primary_language}', secondary={self.secondary_languages}")
        logger.info(f"🎯 FINAL TARGET LANGUAGES: {self.target_languages} (total: {len(set(self.target_languages))} unique languages)")
        
        # Additional safety check for duplicates
        if len(self.target_languages) != len(set(self.target_languages)):
            logger.error(f"🚨 DUPLICATE LANGUAGES DETECTED in target_languages: {self.target_languages}")
            self.target_languages = list(set(self.target_languages))  # Remove duplicates
            logger.info(f"🚨 DEDUPLICATED TARGET LANGUAGES: {self.target_languages}")
        
        # Initialize comparison mode components if needed
        if self.comparison_mode:
            self.translator = Translator()
            self.comparison_renderer = ComparisonRenderer()
        
        # Initialize Google Translator for multi-language support
        if self.target_languages:
            try:
                from deep_translator import GoogleTranslator
                self._translator = GoogleTranslator(source='en', target='it')  # We'll change target dynamically
                
                # Translation cache for performance
                self._translation_cache = {}
                
                # Translation buffer for concurrent display (eliminates delay)
                self._translation_buffer = {}
                self._buffer_lock = threading.Lock()
                
                logger.info(f"Initialized Google Translator for target languages: {self.target_languages}")
            except ImportError:
                logger.warning("Deep Translator not available. Install with: pip install deep-translator")
                self.target_languages = []  # Disable translations if no translator
        
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
        self.caption_timing_offset = 0.0  # Immediate captions (reduced from -0.025 for instant startup)
        
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
        
        # Set video dimensions in caption overlay for responsive font scaling
        caption_overlay.set_video_dimensions(width, height)
        logger.info(f"Set caption overlay video dimensions: {width}x{height}")
        
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
        
        # 🚨 NUCLEAR EMERGENCY MODE: ONLY English captions for smooth video
        # Add ONLY primary English caption - no translations whatsoever
        
        # Log the emergency mode activation
        logger.warning("🚨 NUCLEAR EMERGENCY MODE ACTIVATED: English-only captions for smooth video")
        logger.warning("🚨 All translations disabled until performance issues are resolved")
        
        # Disable translator completely
        self._translator = None
    
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
        """Process a single frame and apply captions with optimized timing."""
        frame, frame_timestamp = frame_data
        
        # Get current audio position for synchronization
        with self.video_source.audio_position_lock:
            current_audio_time = self.video_source.audio_position
        
        # Use audio time for most accurate synchronization
        if self.video_source.audio_playing and current_audio_time > 0:
            self.current_video_time = current_audio_time
        else:
            self.current_video_time = frame_timestamp
        
        # Process transcriptions much less frequently for better performance
        if self.frame_count % 60 == 0:  # Check every 60th frame for nuclear emergency performance (reduced from 30)
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
            
            # Ensure reasonable duration - extend for better readability
            # CRITICAL FIX: Prevent overlapping captions by using shorter, non-overlapping durations
            duration = max(1.5, duration)  # Minimum 1.5 seconds (reduced from 2.0)
            if duration > 4.0:  # Shorter max duration to prevent overlaps (reduced from 8.0)
                duration = 4.0
            
            # Calculate relative timestamps by subtracting seek offset
            rel_start = start_time - self.video_source.start_time
            
            # Apply caption timing offset for better lip sync
            rel_start_adjusted = rel_start + self.caption_timing_offset
            
            # Track latest audio time for heartbeat calculation
            self._latest_audio_rel = rel_start
            
            # Get current video relative time for validation
            current_video_time = self.current_video_time - self.video_source.start_time
            
            # CRITICAL: Smart caption deduplication instead of clearing all
            # Only clear captions that would create exact duplicates, not all captions
            try:
                # Get current active captions
                current_video_time_rel = self.current_video_time - self.video_source.start_time
                active_captions = self.caption_overlay.get_active_captions(current_video_time_rel)
                
                # AGGRESSIVE DEDUPLICATION: Check for any similar captions in ANY language within timing window
                # This prevents multiple caption sets from being created
                duplicate_found = False
                timing_window = 3.0  # 3 second window to check for duplicates
                
                for active_cap in active_captions:
                    if abs(active_cap.get('start_time', 0) - rel_start_adjusted) < timing_window:
                        # Check for similar text content regardless of language
                        active_words = set(active_cap.get('text', '').lower().split())
                        new_words = set(text.lower().split())
                        word_overlap = len(active_words & new_words)
                        
                        # If significant word overlap, this is likely a duplicate transcription
                        if word_overlap > 2:
                            duplicate_found = True
                            logger.debug(f"🚨 DUPLICATE DETECTED: Skipping transcription - {word_overlap} overlapping words with existing caption")
                            break
                
                if duplicate_found:
                    return  # Skip adding this caption AND all its translations
                
                # AGGRESSIVE CLEANUP: Remove ALL overlapping captions before adding new ones
                # This ensures we never have multiple sets active at once
                overlap_start = rel_start_adjusted - 1.0  # Remove captions 1s before
                overlap_end = rel_start_adjusted + duration + 1.0  # Remove captions 1s after
                removed_count = self.caption_overlay.remove_overlapping_captions(overlap_start, overlap_end)
                if removed_count > 0:
                    logger.debug(f"🧹 REMOVED {removed_count} overlapping captions to prevent duplicates")
                
                # Additional cleanup: remove very old captions
                pruned_count = self.caption_overlay.prune_captions(current_video_time_rel, buffer=1.0)
                if pruned_count > 0:
                    logger.debug(f"🧹 PRUNED {pruned_count} old captions")
                                
            except Exception as e:
                logger.warning(f"Failed to perform smart caption deduplication: {e}")
            
            logger.debug(f"🚨 CAPTION TIMING: rel_start={rel_start_adjusted:.2f}s, duration={duration:.2f}s, end={rel_start_adjusted + duration:.2f}s, current_video={current_video_time:.2f}s")
            
            # Reduce logging frequency for better performance
            if self.frame_count % 30 == 0:  # Only log transcription details every 30 frames
                logger.info(f"[TRANSCRIPTION] Received: '{text}' at rel_start={rel_start:.2f}s")
                logger.info(f"[TIMING] Caption timing: rel_start={rel_start:.2f}s -> adjusted={rel_start_adjusted:.2f}s, current_video_time={current_video_time:.2f}s")
            
            # Validate the timestamp is within reasonable bounds (allow captions from recent past and near future)
            time_diff = rel_start - current_video_time
            if time_diff < -10.0 or time_diff > 30.0:  # Reduced from -30s/60s to -10s/30s for more responsive captions
                logger.warning(f"[TIMING] Caption timing out of bounds ({time_diff:.2f}s difference), dropping")
                return
            
            # Filter out bogus transcriptions and hallucinations
            filtered_text = self._filter_bogus_transcriptions(text)
            
            if not filtered_text:
                return
            
            # Apply Sanskrit vocabulary corrections for Chapter 6 terms
            corrected_text = self._correct_sanskrit_vocabulary(filtered_text)
            
            # Apply text processing to enhance caption quality
            processed_text = self._process_caption_text(corrected_text, rel_start)
            
            # Skip empty captions after processing
            if not processed_text.strip():
                logger.debug(f"[TEXT_PROCESSING] Caption removed after processing: '{corrected_text}'")
                return
            
            # Check if we have target languages to translate to
            target_languages = getattr(self, 'target_languages', [])
            
            try:
                # Clean the text for proper display (handle Unicode characters)
                processed_text = self._clean_text_for_display(processed_text)
                
                # Add ONLY English caption - no translations
                primary_caption = self.caption_overlay.add_caption(
                    processed_text, 
                    rel_start_adjusted, 
                    duration, 
                    is_absolute=False, 
                    language='en', 
                    is_primary=True
                )
                
                # EMERGENCY: Skip ALL translation processing completely
                logger.debug(f"🚨 EMERGENCY MODE: Added English-only caption: '{processed_text[:30]}...'")
                
                # VALIDATION: Ensure only 1 caption was added
                current_video_time_rel = self.current_video_time - self.video_source.start_time
                active_captions = self.caption_overlay.get_active_captions(current_video_time_rel)
                
                if len(active_captions) > 1:
                    logger.error(f"🚨 EMERGENCY MODE FAILURE: Found {len(active_captions)} captions, expected only 1!")
                    for i, cap in enumerate(active_captions):
                        logger.error(f"   Caption {i+1}: lang='{cap.get('language', 'UNKNOWN')}', text='{cap.get('text', '')[:30]}...'")
                elif self.frame_count % 30 == 0:
                    logger.info(f"[EMERGENCY] ✓ English-only caption OK: {len(active_captions)}/1 expected")
                
            except Exception as e:
                logger.error(f"Error adding caption: {e}")
                
        except Exception as e:
            logger.error(f"Error handling transcription: {e}")
    
    def _filter_bogus_transcriptions(self, text: str) -> str:
        """Filter out bogus transcriptions and hallucinations.
        
        Args:
            text: Transcribed text to filter
            
        Returns:
            Filtered text or empty string if bogus
        """
        text_lower = text.lower().strip()
        
        # Skip empty or very short text
        if len(text_lower) < 2:
            return ""
        
        # Filter out obvious bogus content
        bogus_patterns = [
            "osho", "www.osho.com", "copyright", "© osho",
            "transcribed by", "subtitles by", "captions by",
            "www.", ".com", ".org", ".net",
            "thank you for watching", "subscribe",
            "like and subscribe", "bell icon"
        ]
        
        # Check for bogus patterns
        for pattern in bogus_patterns:
            if pattern in text_lower:
                logger.debug(f"Filtered bogus transcription: '{text}'")
                return ""
        
        # Filter out single Sanskrit words that appear during silence (likely hallucinations)
        words = text_lower.split()
        if len(words) <= 2:  # 1-2 words only
            sanskrit_words = [
                "krishna", "arjuna", "dharma", "karma", "yoga", "gita",
                "bhagavad", "prabhu", "swami", "guru", "ashram", "mantra"
            ]
            # If it's just Sanskrit words and very short, likely hallucination during silence
            if all(word in sanskrit_words for word in words):
                logger.debug(f"Filtered likely Sanskrit hallucination: '{text}'")
                return ""
        
        return text
    
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
    
    def _clean_text_for_display(self, text: str) -> str:
        """Clean text for proper display while preserving Unicode characters for languages like Russian, Ukrainian, Hungarian.
        
        Args:
            text: Input text that may contain Unicode characters
            
        Returns:
            Cleaned text preserving Unicode characters for proper display
        """
        try:
            # First, ensure the text is properly decoded
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            
            # Only replace problematic characters that could cause rendering issues
            # But preserve all Unicode characters for proper multi-language support
            replacements = {
                # Problematic quotation marks and dashes that might cause rendering issues
                ''': "'", ''': "'", '"': '"', '"': '"',
                '–': '-', '—': '-', '…': '...',
                
                # Currency symbols that might not render well
                '€': 'EUR', '£': 'GBP', '¥': 'JPY',
                '°': 'deg', '±': '+/-', '×': 'x', '÷': '/',
            }
            
            # Apply minimal replacements only for known problematic characters
            cleaned_text = text
            for unicode_char, replacement in replacements.items():
                cleaned_text = cleaned_text.replace(unicode_char, replacement)
            
            # DO NOT remove Unicode characters - we now have proper Unicode rendering support!
            # Keep all Cyrillic, Ukrainian, Hungarian, and other language characters intact
            # Our PIL-based rendering can handle them properly
            
            return cleaned_text.strip()
            
        except Exception as e:
            logger.warning(f"Error cleaning text for display: {e}")
            # Fallback: return the original text to preserve Unicode
            return str(text).strip()
    
    def _translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language using Google Translate with caching.
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'it', 'es', 'fr')
            
        Returns:
            Translated text or original text if translation fails
        """
        try:
            # Create cache key
            cache_key = f"{text}:{target_language}"
            
            # Check cache first
            if hasattr(self, '_translation_cache') and cache_key in self._translation_cache:
                return self._translation_cache[cache_key]
            
            # Create a new translator instance for the specific target language
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='auto', target=target_language)
            result = translator.translate(text)
            
            # Cache the result
            if hasattr(self, '_translation_cache'):
                self._translation_cache[cache_key] = result
                
                # Limit cache size to prevent memory issues
                if len(self._translation_cache) > 1000:
                    # Remove oldest entries (simple FIFO)
                    keys_to_remove = list(self._translation_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self._translation_cache[key]
            
            return result if result and result.strip() else text
            
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text
    
    def _translate_and_add_caption_fast(self, text: str, lang_code: str, start_time: float, duration: float):
        """Fast translation with immediate caption addition for concurrent display.
        
        Args:
            text: Text to translate
            lang_code: Target language code
            start_time: Caption start time
            duration: Caption duration
        """
        try:
            # Use fast translation with caching
            translated_text = self._translate_text(text, target_language=lang_code)
            
            if translated_text and translated_text.strip() and translated_text != text:
                # Clean the translated text for proper display
                cleaned_translated_text = self._clean_text_for_display(translated_text)
                
                # Add translated caption with SAME timing as primary for concurrent display
                secondary_caption = self.caption_overlay.add_caption(
                    cleaned_translated_text, 
                    start_time, 
                    duration, 
                    is_absolute=False, 
                    language=lang_code, 
                    is_primary=False
                )
                    
                if self.frame_count % 30 == 0:
                    logger.info(f"[CAPTION] Added {lang_code}: {cleaned_translated_text!r}")
                return True
            else:
                if self.frame_count % 30 == 0:
                    logger.warning(f"[CAPTION] Translation failed or identical for {lang_code}: {translated_text!r}")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to translate to {lang_code}: {e}")
            return False
    
    def _buffer_translations(self, text: str, timestamp: float):
        """Pre-translate text and buffer for concurrent display.
        
        Args:
            text: Text to translate
            timestamp: Timestamp for the translation
        """
        if not hasattr(self, '_translator') or not self.target_languages:
            return
            
        def translate_and_buffer():
            try:
                with self._buffer_lock:
                    # Pre-translate to all target languages
                    for lang_code in self.target_languages:
                        translated_text = self._translate_text(text, target_language=lang_code)
                        if translated_text and translated_text.strip():
                            # Store in buffer with timestamp key
                            buffer_key = f"{timestamp}:{lang_code}"
                            self._translation_buffer[buffer_key] = {
                                'text': translated_text,
                                'timestamp': timestamp,
                                'language': lang_code,
                                'ready': True
                            }
                            
                    # Clean old buffer entries (older than 30 seconds)
                    current_time = timestamp
                    keys_to_remove = [
                        key for key, value in self._translation_buffer.items()
                        if current_time - value['timestamp'] > 30.0
                    ]
                    for key in keys_to_remove:
                        del self._translation_buffer[key]
                        
            except Exception as e:
                logger.warning(f"Translation buffering failed: {e}")
        
        # Start buffering in background thread to avoid blocking
        buffer_thread = threading.Thread(target=translate_and_buffer, daemon=True)
        buffer_thread.start()

    def _get_buffered_translation(self, text: str, timestamp: float, lang_code: str) -> str:
        """Get buffered translation if available, otherwise translate immediately.
        
        Args:
            text: Original text
            timestamp: Timestamp for the translation
            lang_code: Target language code
            
        Returns:
            Translated text
        """
        try:
            with self._buffer_lock:
                buffer_key = f"{timestamp}:{lang_code}"
                if buffer_key in self._translation_buffer:
                    buffered = self._translation_buffer[buffer_key]
                    if buffered['ready']:
                        return buffered['text']
        except Exception as e:
            logger.warning(f"Buffer lookup failed: {e}")
        
        # Fallback to immediate translation
        return self._translate_text(text, target_language=lang_code)
    
    def _correct_sanskrit_vocabulary(self, text: str) -> str:
        """Correct common mishearings of Sanskrit terms using Chapter 6 vocabulary.
        
        Args:
            text: Transcribed text to correct
            
        Returns:
            Text with corrected Sanskrit terms
        """
        # Sanskrit term corrections for common Whisper mishearings
        # This fixes terms AFTER transcription to avoid hallucinations
        corrections = {
            # Chapter 6 Dhyana Yoga terms
            'diana': 'dhyāna', 'dianna': 'dhyāna', 'dhana': 'dhyāna', 'dyana': 'dhyāna', 'dhyan': 'dhyāna',
            'kusha': 'kuśa', 'kusa': 'kuśa', 'kosha': 'kuśa',
            'abhyasa': 'abhyāsa', 'abhiasa': 'abhyāsa', 'abhyassa': 'abhyāsa',
            'vairagya': 'vairāgya', 'vairagya': 'vairāgya', 'vairaga': 'vairāgya',
            'pratyahara': 'pratyāhāra', 'pratyaahara': 'pratyāhāra', 'pratiahara': 'pratyāhāra',
            'pranayama': 'prāṇāyāma', 'pranayaama': 'prāṇāyāma', 'pranaayama': 'prāṇāyāma',
            'asana': 'āsana', 'aasana': 'āsana', 'aasanna': 'āsana',
            'dhaarana': 'dhāraṇā', 'dharana': 'dhāraṇā', 'dhaaranaa': 'dhāraṇā',
            'samadhi': 'samādhi', 'samaadhi': 'samādhi', 'samaadhii': 'samādhi',
            
            # Common names and terms
            'krishna': 'Krishna', 'krsna': 'Krsna', 'krisna': 'Krishna',
            'arjuna': 'Arjuna', 'arjun': 'Arjuna', 'arjunaa': 'Arjuna',
            'bhagavad': 'Bhagavad', 'bhagavat': 'Bhagavad', 'bhagwad': 'Bhagavad',
            'prabhupaada': 'Prabhupāda', 'prabhupada': 'Prabhupāda', 'prabhupad': 'Prabhupāda',
            'madhusudana': 'Madhusūdana', 'madhusudan': 'Madhusūdana',
            'janardana': 'Janārdana', 'janardan': 'Janārdana',
            'govinda': 'Govinda', 'govind': 'Govinda',
            'keshava': 'Keśava', 'keshav': 'Keśava', 'kesava': 'Keśava',
            
            # Philosophy terms
            'atma': 'ātmā', 'aatma': 'ātmā', 'aatmaa': 'ātmā',
            'paramatma': 'paramātmā', 'paramaatma': 'paramātmā', 'paramaatmaa': 'paramātmā',
            'moksha': 'mokṣa', 'moksh': 'mokṣa', 'mokshaa': 'mokṣa',
            'samsara': 'samsāra', 'samsaar': 'samsāra', 'samsaara': 'samsāra',
            'brahmacaari': 'brahmacārī', 'brahmachari': 'brahmacārī', 'brahmacari': 'brahmacārī',
            'varnasrama': 'varṇāśrama', 'varnashrama': 'varṇāśrama', 'varna ashrama': 'varṇāśrama',
        }
        
        corrected_text = text
        
        # Apply corrections (case-insensitive matching, case-preserving replacement)
        for wrong_spelling, correct_spelling in corrections.items():
            # Replace whole words only to avoid partial matches
            import re
            pattern = r'\b' + re.escape(wrong_spelling) + r'\b'
            
            # Case-insensitive replacement
            def replace_func(match):
                matched_text = match.group(0)
                # Preserve original case pattern
                if matched_text.isupper():
                    return correct_spelling.upper()
                elif matched_text.islower():
                    return correct_spelling.lower()
                elif matched_text.istitle():
                    return correct_spelling.title()
                else:
                    return correct_spelling
            
            corrected_text = re.sub(pattern, replace_func, corrected_text, flags=re.IGNORECASE)
        
        # Log corrections if any were made
        if corrected_text != text:
            logger.info(f"[SANSKRIT] Corrected: '{text}' -> '{corrected_text}'")
        
        return corrected_text
    
    def _handle_key_press(self, key):
        """Handle keyboard input for playback control.
        
        Args:
            key: Key code from cv2.waitKey()
        """
        if key == ord('q') or key == 27:  # 'q' or ESC
            logger.info("Quit key pressed - stopping playback")
            self.running = False
        elif key == ord('p') or key == 32:  # 'p' or SPACE
            self.paused = not self.paused
            if self.paused:
                self.pause_start_time = time.perf_counter()
                logger.info("Playback paused")
            else:
                pause_duration = time.perf_counter() - self.pause_start_time
                self.total_pause_duration += pause_duration
                logger.info(f"Playback resumed (paused for {pause_duration:.2f}s)")
    
    def run(self):
        """Run the main video playback and synchronization loop."""
        try:
            # Initialize high-precision timing
            start_time = time.perf_counter()
            frame_count = 0
            target_fps = self.video_source.fps
            frame_duration = 1.0 / target_fps
            
            logger.info(f"Starting high-performance playback at {target_fps}fps")
            
            # Frame processing optimization
            max_frame_processing_time = frame_duration * 0.6  # Use 60% of frame time budget (reduced from 80%)
            frame_skip_count = 0
            
            while self.running:
                loop_start = time.perf_counter()
                
                # Handle pause efficiently
                if self.paused:
                    if not self.headless:
                        key = cv2.waitKey(50) & 0xFF
                        if key != 255:
                            self._handle_key_press(key)
                    time.sleep(0.05)
                    continue
                
                # Get audio time for synchronization
                with self.video_source.audio_position_lock:
                    audio_time = self.video_source.audio_position
                
                # Calculate target frame time based on audio
                if self.video_source.audio_playing and audio_time > 0:
                    # Use audio time as the authoritative source
                    adjusted_audio_time = audio_time - self.video_source.start_time
                    target_frame_number = int(adjusted_audio_time * target_fps)
                else:
                    # Fallback to elapsed time
                    elapsed_time = loop_start - start_time - self.total_pause_duration
                    target_frame_number = int(elapsed_time * target_fps)
                
                # Skip frames only if significantly behind (aggressive sync)
                frames_behind = target_frame_number - frame_count
                
                if frames_behind > 3:  # If more than 3 frames behind
                    # Skip frames to catch up
                    while frame_count < target_frame_number - 1 and frames_behind > 1:
                        frame_data = self.video_source.get_frame()
                        if frame_data is None:
                            self.running = False
                            break
                        frame_count += 1
                        frames_behind -= 1
                        frame_skip_count += 1
                    
                    if frame_skip_count > 0 and frame_skip_count % 10 == 0:
                        logger.info(f"Skipped {frame_skip_count} frames for audio sync")
                
                # Process current frame with timing budget
                if self.running:
                    frame_processing_start = time.perf_counter()
                    
                    # Get the next frame
                    frame_data = self.video_source.get_frame()
                    if frame_data is None:
                        break
                    
                    # Process frame with timing check
                    frame = self._process_frame(frame_data)
                    
                    # Display frame only if not in headless mode
                    if not self.headless:
                        # Efficient frame resizing
                        if hasattr(self, 'display_width') and hasattr(self, 'display_height'):
                            frame_height, frame_width = frame.shape[:2]
                            if frame_width != self.display_width or frame_height != self.display_height:
                                frame = cv2.resize(frame, (self.display_width, self.display_height), 
                                                 interpolation=cv2.INTER_LINEAR)
                        
                        cv2.imshow(self.window_name, frame)
                        
                        # Fast key handling
                        key = cv2.waitKey(1) & 0xFF
                        if key != 255:
                            self._handle_key_press(key)
                            if not self.running:
                                break
                    
                    frame_count += 1
                    self.frame_count = frame_count
                    
                    # Check processing time budget
                    frame_processing_time = time.perf_counter() - frame_processing_start
                    if frame_processing_time > max_frame_processing_time:
                        logger.warning(f"Frame processing exceeded budget: {frame_processing_time*1000:.1f}ms")
                    
                    # 🚨 EMERGENCY: Disable all logging for maximum performance
                    # if frame_count % 1800 == 0:  # Disabled for nuclear emergency mode
                    #     self._log_timing_stats()
                
                # Precise timing control
                loop_duration = time.perf_counter() - loop_start
                sleep_time = frame_duration - loop_duration
                
                if sleep_time > 0.001:  # Only sleep if worthwhile
                    time.sleep(sleep_time)
                elif sleep_time < -frame_duration:  # If severely behind
                    logger.debug(f"Severe frame timing delay: {-sleep_time*1000:.1f}ms")
            
        finally:
            if not self.headless:
                cv2.destroyAllWindows()
            self.video_source.release()
            logger.info(f"Playback ended. Skipped {frame_skip_count} frames total for sync.")
