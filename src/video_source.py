import cv2
import numpy as np
import os
import time
import threading
import queue
from queue import Queue
import logging
import sounddevice as sd
import subprocess
import soundfile as sf
# Import with try-except to handle both direct execution and module import
try:
    from logging_utils import TRACE, setup_logging, get_logger
except ImportError:
    from .logging_utils import TRACE, setup_logging, get_logger

# Import singleton clock
try:
    from clock import CLOCK
except ImportError:
    from .clock import CLOCK

# Configure logging with our custom utilities
logger = get_logger(__name__)

class VideoSource:
    def __init__(self, source, start_time=0.0):
        """Initialize video source with audio handling capabilities.
        
        Args:
            source: Path to video file (should be in ~/.video_cache)
            start_time: Time position (in seconds) to start playback from
        """
        self.logger = get_logger(f"{__name__}.VideoSource")
        self.logger.info(f"Initializing VideoSource with source: {source}, start_time: {start_time}")
        
        # Guard against duplicate VideoSource instances
        if CLOCK.video_source_created:
            logger.warning("🚨 Duplicate VideoSource detected - using singleton clock to prevent timing conflicts")
        else:
            CLOCK.video_source_created = True
            logger.info("🔧 First VideoSource instance - will initialize singleton clock")
        
        # Use source directly if it's an absolute path, otherwise assume it's in cache
        self.source = source
        if not os.path.isabs(source):
            cache_dir = os.path.expanduser("~/.video_cache")
            self.source = os.path.join(cache_dir, source)
            self.logger.debug(f"Using cached video path: {self.source}")
            
        # Initialize queue for synchronized playback with larger buffer
        self.frames_queue = Queue(maxsize=240)  # 240 frames (8 seconds at 30fps)
        logger.info(f"🔧 CONSTRUCTOR: VideoSource frames_queue created with maxsize={self.frames_queue.maxsize}")
        self.last_queue_warning = 0
        self._last_queue_warning = 0
        self._consecutive_drops = 0
        self._last_queue_size = 0
        
        # Thread management
        self.is_running = False
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        
        # Video information
        self._video_fps = 0
        self._frame_count = 0
        
        # Audio playback
        self.audio_playing = False
        self.audio_thread = None
        self._cached_audio_file = None  # Cache the audio file path
        
        # Starting position (in seconds)
        self.start_time = start_time
        
        # Media timing - use singleton clock instead of instance variables
        # Note: actual initialization happens in start() after video is opened
        
        # Synchronization objects
        self.audio_position = 0.0  # Current audio position in seconds
        self.audio_position_lock = threading.RLock()
        self.playback_started = threading.Event()
        self.warm_up_complete = threading.Event()
        self.start_sync_event = threading.Event()  # For exact start synchronization
        
        # Compatibility properties for existing code
        self.playback_start_time = 0.0  # Will be set from CLOCK when initialized
        self.media_seek_pts = 0.0       # Will be set from CLOCK when initialized
        
        # Thread control
        self._stop_event = threading.Event()
        self._frame_available_cv = threading.Condition()
        self._started = False
        self._error_occurred = False
        self._error = None
        self._capture_thread = None
        self._start_cv = threading.Condition()
        
    def start(self):
        """Start video capture and processing."""
        with self._lock:
            if self.is_running:
                self.logger.warning("Video source is already running")
                return
                
            # Reset state
            self._stop_event.clear()
            self._error_occurred = False
            self._error = None
            self._started = False
            self.logger.info("Starting video source")
            
            try:
                # Start video capture directly from source but don't open window yet
                self.logger.info(f"Opening video source: {self.source}")
                start_time = time.time()
                self._cap = cv2.VideoCapture(self.source)
                
                if not self._cap.isOpened():
                    error_msg = f"Failed to open video: {self.source}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
                open_time = time.time() - start_time
                self.logger.debug(f"Video opened in {open_time:.3f} seconds")
                    
                # Get and log video properties
                self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
                width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / self._video_fps if self._video_fps > 0 else 0
                
                self.logger.info(
                    f"Video properties - Resolution: {width}x{height}, "
                    f"FPS: {self._video_fps:.2f}, "
                    f"Duration: {duration:.2f}s, "
                    f"Frames: {frame_count}"
                )
                self.logger.debug(f"Video backend: {self._cap.getBackendName()}")
                
                # Log detailed codec information at TRACE level
                if logger.isEnabledFor(TRACE):
                    codec = int(self._cap.get(cv2.CAP_PROP_FOURCC))
                    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
                    self.logger.log(TRACE, f"Video codec: {codec_str} (0x{codec:08X})")
                
                # Set initial frame count based on start_time if specified
                if self.start_time > 0:
                    self._frame_count = int(self.start_time * self._video_fps)
                    # Seek to the specified start position
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_count)
                    logger.info(f"Seeking video to {self.start_time:.2f}s (frame {self._frame_count})")
                    # Initialize singleton clock with seek position
                    if CLOCK.initialize(self.start_time):
                        logger.info(f"🔧 Initialized singleton clock: media_seek_pts={CLOCK.media_seek_pts:.2f}s")
                    else:
                        logger.info(f"🔧 Singleton clock already initialized: media_seek_pts={CLOCK.media_seek_pts:.2f}s")
                else:
                    # Initialize clock with no seek
                    if CLOCK.initialize(0.0):
                        logger.info(f"🔧 Initialized singleton clock: media_seek_pts={CLOCK.media_seek_pts:.2f}s")
                    else:
                        logger.info(f"🔧 Singleton clock already initialized: media_seek_pts={CLOCK.media_seek_pts:.2f}s")
                
                # Set compatibility properties from clock
                self.media_seek_pts = CLOCK.media_seek_pts
                
                # Check for existing audio file in cache but don't start playback yet
                self.audio_file = self._get_audio_file_path()
                
                # Initialize is_running before starting threads
                self.is_running = True
                
                # Clear sync events before starting threads
                self.start_sync_event.clear()
                self.warm_up_complete.clear()
                
                # Start video thread
                logger.info("Starting video capture thread")
                self._capture_thread = threading.Thread(
                    target=self._capture_frames,
                    name="VideoCaptureThread"
                )
                self._capture_thread.daemon = True
                self._capture_thread.start()
                
                # Wait for thread to initialize
                with self._start_cv:
                    if not self._started:
                        self._start_cv.wait(5.0)  # 5 second timeout
                
                if not self._started:
                    raise RuntimeError("Failed to start video capture thread")
                
                # Don't start audio thread yet - it will be started after warm-up
                has_audio = bool(self.audio_file)
                if not has_audio:
                    logger.warning("No audio file available - video timing will be used")
                
                # Prebuffering is now handled by _capture_frames after warm_up_complete.
                # logger.info("Pre-buffering frames for smooth start")
                # start_buffer_time = time.time()
                # while self.frames_queue.qsize() < 5 and time.time() - start_buffer_time < 2.0:
                #     time.sleep(0.1)
                
                # logger.info(f"Buffered {self.frames_queue.qsize()} frames, ready to start playback")
                
                # The actual playback_start_time will be set by main.py or VideoRunner just before display loop
                
                # Signal threads to start playback at exactly the same time
                logger.info(f"Signaling synchronized playback start at time {self.playback_start_time}")
                self.start_sync_event.set()
                
                # Wait a moment to ensure threads have picked up the signal
                time.sleep(0.1)
                logger.info("VideoSource started successfully")
                
            except Exception as e:
                logger.error(f"Error starting video source: {e}")
                self.is_running = False
                raise
                
    def _get_audio_file_path(self):
        """Get path to audio file that corresponds to the video file."""
        # Return cached path if available
        if self._cached_audio_file is not None:
            return self._cached_audio_file
            
        # First, check if there's a matching WAV file in the same folder
        video_basename = os.path.splitext(self.source)[0]
        audio_file = f"{video_basename}.wav"
        
        if os.path.exists(audio_file):
            logger.info(f"Found existing audio file: {audio_file}")
            self._cached_audio_file = audio_file  # Cache the result
            return audio_file
            
        # If no WAV file exists, extract it from the video using FFmpeg
        audio_file = f"{video_basename}_audio.wav"
        if not os.path.exists(audio_file):
            logger.info(f"Extracting audio to {audio_file}")
            
            try:
                # Use FFmpeg to extract audio
                cmd = [
                    'ffmpeg',
                    '-i', self.source,
                    '-f', 'wav',
                    '-vn',  # Skip video
                    '-acodec', 'pcm_s16le',  # Convert to WAV PCM
                    '-ar', '44100',  # Set sample rate
                    '-ac', '2',  # Set channels
                    '-y',  # Overwrite output file
                    audio_file
                ]
                
                process = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                if process.returncode != 0:
                    logger.warning(f"FFmpeg error: {process.stderr.decode()}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error extracting audio: {e}")
                return None
                
        if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
            logger.info(f"Audio file ready: {audio_file}")
            self._cached_audio_file = audio_file  # Cache the result
            return audio_file
        else:
            logger.warning("Audio file extraction failed or file is empty")
            return None
            
    def _capture_frames(self):
        """Capture frames from video and put them into the queue."""
        self.logger.info("Video capture thread started.")
        self._started = True
        with self._start_cv:
            self._start_cv.notify_all()

        frame_time = 1.0 / self._video_fps if self._video_fps > 0 else 0.033
        consecutive_empty_frames = 0
        max_empty_frames = int(self._video_fps * 2) # 2 seconds worth of empty frames

        # Wait for the main application warm-up to complete before starting capture loop
        self.logger.info("Capture thread waiting for application warm-up...")
        self.warm_up_complete.wait() # Blocks until main.py sets this event
        self.logger.info("Application warm-up complete. Capture thread starting frame acquisition.")

        # Pre-buffer a small number of frames *after* warm-up, without pacing
        initial_buffer_target = 5
        self.logger.info(f"Pre-buffering initial {initial_buffer_target} frames...")
        for _ in range(initial_buffer_target):
            if not self.is_running or self._stop_event.is_set() or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if ret:
                current_frame_pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
                timestamp = current_frame_pos / self._video_fps if self._video_fps > 0 else time.time() - self.playback_start_time
                try:
                    self.frames_queue.put_nowait((frame, timestamp))
                except queue.Full:
                    self.logger.warning("Frame queue full during initial pre-buffer, discarding oldest frame.")
                    try:
                        self.frames_queue.get_nowait() # Discard oldest
                        self.frames_queue.put_nowait((frame, timestamp)) # Try adding new one
                    except queue.Full:
                        self.logger.error("Frame queue still full after discarding one, cannot pre-buffer.")
                        break # Unable to pre-buffer
            else:
                self.logger.warning("Failed to read frame during initial pre-buffer.")
                break
        self.logger.info(f"Initial pre-buffering complete. {self.frames_queue.qsize()} frames in queue.")

        last_frame_capture_time = time.time()

        try:
            while self.is_running and not self._stop_event.is_set() and self._cap.isOpened():
                capture_start_time = time.time()
                ret, frame = self._cap.read()

                if not ret:
                    consecutive_empty_frames += 1
                    self.logger.debug(f"No frame returned by cap.read(). Consecutive empty: {consecutive_empty_frames}")
                    if consecutive_empty_frames > max_empty_frames:
                        self.logger.warning(f"Reached {max_empty_frames} consecutive empty frames. Assuming end of video or error.")
                        self.is_running = False
                        break
                    time.sleep(0.01) # Brief pause if no frame
                    continue
                
                consecutive_empty_frames = 0 # Reset on successful frame read
                current_frame_pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
                timestamp = current_frame_pos / self._video_fps if self._video_fps > 0 else time.time() - self.playback_start_time
                frame_data = (frame, timestamp)

                try:
                    if self.frames_queue.qsize() >= self.frames_queue.maxsize * 0.9:
                        # Queue is > 90% full, discard oldest frame
                        try:
                            discarded_frame_data = self.frames_queue.get_nowait()
                            self._consecutive_drops +=1
                            logger.debug("discard-oldest engaged")  # Requested debug log
                            if self._consecutive_drops % int(self._video_fps) == 0: # Log every second of continuous drops
                                logger.warning(f"Frames queue >90% full. Discarded oldest frame. (Dropped: {self._consecutive_drops} consecutive)")
                        except queue.Empty:
                            pass # Should not happen if qsize > 0
                    else:
                        self._consecutive_drops = 0 # Reset drop counter if queue has space

                    self.frames_queue.put(frame_data, timeout=0.1)  # Add with a small timeout
                except queue.Full:
                    # This path should ideally be hit less now due to the 90% check above.
                    # If still full after discarding, log a more severe warning or handle as critical.
                    self.logger.error(f"Frames queue critically full despite adaptive drop. Timestamp: {timestamp:.2f}s. Queue size: {self.frames_queue.qsize()}")
                    # As a fallback, try to make space again if this happens
                    try:
                        self.frames_queue.get_nowait() # Discard oldest
                        self.frames_queue.put_nowait(frame_data) # Try adding current frame
                    except queue.Full:
                        self.logger.critical("CRITICAL: Frame queue completely blocked. Dropping current frame.")
                        self._consecutive_drops += 1 # Count this as a drop too

                # Frame pacing
                processing_time = time.time() - capture_start_time
                sleep_duration = frame_time - processing_time
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                last_frame_capture_time = time.time()

        except Exception as e:
            self.logger.error(f"Exception in video capture thread: {e}", exc_info=True)
            self._error = e
            self._error_occurred = True
        finally:
            self.logger.info("Video capture thread stopping.")
            if self._cap.isOpened():
                self._cap.release()
            self.is_running = False
            # Signal any waiting threads that capture is done
            with self._frame_available_cv:
                self._frame_available_cv.notify_all()

    def _play_audio(self, audio_file):
        """Play audio using sounddevice with proper stream handling to avoid distortion."""
        try:
            self.logger.info(f"Preparing audio playback from {audio_file}")
            
            # Load audio file with timing
            load_start = time.time()
            data, sample_rate = sf.read(audio_file)
            load_time = time.time() - load_start
            
            self.logger.debug(
                f"Audio loaded: {len(data)/sample_rate:.2f}s duration, "
                f"{sample_rate}Hz, {data.shape[1] if len(data.shape) > 1 else 1} channels, "
                f"loaded in {load_time*1000:.1f}ms"
            )
            
            # Convert audio to float32 if not already in that format
            if data.dtype != np.float32:
                data = data.astype(np.float32)
                
            # Wait for the sync signal
            logger.info("Audio thread ready, waiting for sync signal")
            self.start_sync_event.wait()
            
            # Wait until the exact start time
            sleep_time = max(0, self.playback_start_time - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            logger.info(f"Starting audio playback at {self.start_time:.2f}s")
            
            # Tracking variables - initialize to start position
            frames_played = int(self.start_time * sample_rate)
            
            # Create a callback that plays audio sequentially without repositioning
            def audio_callback(outdata, frames, time_info, status):
                nonlocal frames_played
                
                if status:
                    self.logger.warning(f"Audio callback status: {status}")
                
                # Log detailed audio callback info at TRACE level
                if self.logger.isEnabledFor(TRACE):
                    self.logger.log(
                        TRACE,
                        f"Audio callback: frames={frames}, time_info={time_info}, "
                        f"position={frames_played/sample_rate:.3f}s"
                    )
                
                # Check if we've reached the end of the audio
                if frames_played + frames > len(data):
                    # Partial frame at the end
                    remaining = len(data) - frames_played
                    if remaining > 0:
                        outdata[:remaining] = data[frames_played:frames_played+remaining]
                        outdata[remaining:].fill(0)
                        self.logger.debug(f"Playing final audio chunk: {remaining} samples")
                    else:
                        outdata.fill(0)
                    
                    # Update position before stopping
                    current_time = frames_played / sample_rate
                    with self.audio_position_lock:
                        self.audio_position = current_time
                        
                    self.logger.info(f"Audio playback completed at {current_time:.2f}s")
                    return sd.CallbackStop
                
                # Copy the data
                outdata[:] = data[frames_played:frames_played+frames]
                
                # Update frames played
                frames_played += frames
                
                # Update position for video synchronization
                current_time = frames_played / sample_rate
                with self.audio_position_lock:
                    self.audio_position = current_time
                
                return None
            
            # Start audio stream with callback
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=data.shape[1] if len(data.shape) > 1 else 1,
                callback=audio_callback,
                blocksize=1024  # Use a reasonable block size for smooth playback
            ) as stream:
                # Wait until audio is done playing or shutdown is requested
                while self.is_running and self.audio_playing and not self._shutdown_event.is_set():
                    if not stream.active:
                        break
                    time.sleep(0.1)
                    
            logger.info("Audio playback ended")
                
        except Exception as e:
            logger.error(f"Error in audio playback: {e}")
        finally:
            # Just ensure we've stopped the stream
            sd.stop()
            
    def get_frame(self):
        """Get the next video frame with its timestamp."""
        if not self.is_running:
            return None
            
        try:
            return self.frames_queue.get(timeout=0.1)
        except:
            return None
            
    def get_audio_chunk(self, chunk_size=3.0):
        """Get the next audio chunk with its timestamp.
        
        During warm-up, this will process audio chunks without playing them.
        After warm-up, it will return chunks for actual playback.
        
        Args:
            chunk_size: Length of audio chunk in seconds (default: 3.0)
            
        Returns:
            tuple: (audio_data, start_time) where audio_data is a numpy array of samples
                   and start_time is the timestamp in seconds where the chunk begins.
        """
        if not self.is_running:
            return None
            
        try:
            with self.audio_position_lock:
                # Use singleton clock for proper timing
                if CLOCK.is_initialized():
                    # Calculate elapsed time since warm-up completion
                    elapsed_time = time.time() - CLOCK.start_wall_time
                    # Audio chunks should be timestamped relative to the seek position
                    current_time = CLOCK.media_seek_pts + elapsed_time
                else:
                    # Fallback if clock not available - use old method
                    current_time = self.start_time + self.audio_position
                    
                # If we're in warm-up, increment the position for processing
                if not self.warm_up_complete.is_set():
                    self.audio_position += chunk_size
            
            # Extract actual audio data
            audio_file = self._get_audio_file_path()
            if audio_file and os.path.exists(audio_file):
                try:
                    with sf.SoundFile(audio_file) as f:
                        sample_rate = f.samplerate
                        # Calculate the start sample from the beginning of the file
                        start_sample = int(current_time * sample_rate)
                        chunk_samples = int(chunk_size * sample_rate)
                        
                        # Don't try to read before the start or past the end of the file
                        if 0 <= start_sample < len(f):
                            f.seek(start_sample)
                            # Read chunk_size seconds of audio or remaining audio
                            audio_chunk = f.read(
                                min(chunk_samples, len(f) - start_sample), 
                                dtype='float32'
                            )
                            
                            # If we got less than chunk_size seconds, pad with zeros
                            if len(audio_chunk) < chunk_samples:
                                padding = np.zeros(
                                    (chunk_samples - len(audio_chunk),) + audio_chunk.shape[1:], 
                                    dtype=audio_chunk.dtype
                                )
                                audio_chunk = np.concatenate([audio_chunk, padding])
                            
                            # Convert to mono if stereo
                            if len(audio_chunk.shape) > 1 and audio_chunk.shape[1] > 1:
                                audio_chunk = audio_chunk.mean(axis=1)
                            
                            # Resample to 16kHz if needed (Whisper's expected sample rate)
                            if sample_rate != 16000:
                                import librosa
                                audio_chunk = librosa.resample(
                                    audio_chunk, 
                                    orig_sr=sample_rate, 
                                    target_sr=16000
                                )
                            
                            logger.debug(
                                f"Processing audio chunk {len(audio_chunk)/16000:.2f}s "
                                f"at {current_time:.2f}s (warm-up: {not self.warm_up_complete.is_set()})"
                            )
                            
                            # Return the chunk with its absolute timestamp
                            return (audio_chunk, current_time)
                            
                except Exception as e:
                    logger.error(f"Error extracting audio: {e}", exc_info=True)
            
            # Fallback to silent audio at 16kHz if extraction fails
            audio_data = np.zeros(int(chunk_size * 16000), dtype=np.float32)
            return (audio_data, current_time)
            
        except Exception as e:
            logger.error(f"Error in get_audio_chunk: {e}", exc_info=True)
            return None
            
    def get_current_time(self):
        """Get the current playback time in seconds.
        
        Returns:
            float: Current playback time in seconds, or 0.0 if not available
        """
        if not hasattr(self, 'playback_start_time') or not hasattr(self, 'audio_position'):
            return 0.0
            
        # Calculate current time based on when playback started and current audio position
        return self.audio_position
        
    def get_video_info(self):
        """Get video dimensions and FPS."""
        if not hasattr(self, '_cap'):
            return (640, 480, 30.0)  # Default values
            
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return (width, height, fps)
            
    def stop(self):
        """Stop all threads and release resources."""
        with self._lock:
            if not self.is_running:
                self.logger.debug("Video source already stopped")
                return
                
            self.logger.info("Stopping video source...")
            
            try:
                # Signal threads to stop
                self.is_running = False
                self._shutdown_event.set()
                
                # Stop audio if playing
                self.audio_playing = False
                
                # Log queue status before cleanup
                qsize = self.frames_queue.qsize()
                if qsize > 0:
                    self.logger.debug(f"Clearing frame queue with {qsize} pending frames")
                
                # Wait for threads to finish
                if hasattr(self, 'audio_thread') and self.audio_thread and self.audio_thread.is_alive():
                    self.logger.debug("Waiting for audio thread to finish...")
                    self.audio_thread.join(timeout=1.0)
                    if self.audio_thread.is_alive():
                        self.logger.warning("Audio thread did not stop gracefully")
                
                # Release video capture
                if hasattr(self, '_cap') and self._cap is not None:
                    try:
                        self._cap.release()
                        self.logger.debug("Video capture released")
                    except Exception as e:
                        self.logger.error(f"Error releasing video capture: {e}")
                
                # Clear frame queue
                cleared_frames = 0
                while not self.frames_queue.empty():
                    try:
                        self.frames_queue.get_nowait()
                        cleared_frames += 1
                    except queue.Empty:
                        break
                        
                if cleared_frames > 0:
                    self.logger.debug(f"Cleared {cleared_frames} frames from queue")
                
                self.logger.info("Video source stopped successfully")
                
            except Exception as e:
                self.logger.error(f"Error during video source shutdown: {e}", exc_info=True)
                raise
            
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
