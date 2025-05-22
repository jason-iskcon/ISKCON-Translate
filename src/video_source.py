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

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoSource:
    def __init__(self, source, start_time=0.0):
        """Initialize video source with audio handling capabilities.
        
        Args:
            source: Path to video file (should be in ~/.video_cache)
            start_time: Time position (in seconds) to start playback from
        """
        # Use source directly if it's an absolute path, otherwise assume it's in cache
        self.source = source
        if not os.path.isabs(source):
            self.source = os.path.join(os.path.expanduser("~/.video_cache"), source)
            
        # Initialize queue for synchronized playback with larger buffer
        self.frames_queue = Queue(maxsize=120)  # Increased from 30 to 120 frames (about 4 seconds at 30fps)
        self.last_queue_warning = 0
        
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
        
        # Synchronization objects
        self.audio_position = 0.0  # Current audio position in seconds
        self.audio_position_lock = threading.RLock()
        self.playback_started = threading.Event()
        self.warm_up_complete = threading.Event()
        self.video_start_time = 0.0
        self.start_sync_event = threading.Event()  # For exact start synchronization
        self.playback_start_time = 0.0  # Common reference point for sync
        
    def start(self):
        """Start video capture and processing."""
        with self._lock:
            if self.is_running:
                return
            
            try:
                # Start video capture directly from source but don't open window yet
                logger.info(f"Opening video source: {self.source}")
                self._cap = cv2.VideoCapture(self.source)
                
                if not self._cap.isOpened():
                    raise RuntimeError(f"Failed to open video: {self.source}")
                    
                # Get video info
                self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
                
                # Set initial frame count based on start_time if specified
                if self.start_time > 0:
                    self._frame_count = int(self.start_time * self._video_fps)
                    # Seek to the specified start position
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_count)
                    logger.info(f"Seeking video to {self.start_time:.2f}s (frame {self._frame_count})")
                else:
                    self._frame_count = 0
                
                # Check for existing audio file in cache but don't start playback yet
                self.audio_file = self._get_audio_file_path()
                
                # Initialize is_running before starting threads
                self.is_running = True
                
                # Clear sync events before starting threads
                self.start_sync_event.clear()
                self.warm_up_complete.clear()
                
                # Start video thread (will wait for warm-up)
                logger.info("Starting video thread (waiting for warm-up to complete)")
                self.video_thread = threading.Thread(target=self._capture_frames)
                self.video_thread.daemon = True
                self.video_thread.start()
                
                # Don't start audio thread yet - it will be started after warm-up
                has_audio = bool(self.audio_file)
                if not has_audio:
                    logger.warning("No audio file available - video timing will be used")
                
                # Prebuffer some frames to ensure smooth start
                logger.info("Pre-buffering frames for smooth start")
                start_buffer_time = time.time()
                while self.frames_queue.qsize() < 5 and time.time() - start_buffer_time < 2.0:
                    time.sleep(0.1)
                
                logger.info(f"Buffered {self.frames_queue.qsize()} frames, ready to start playback")
                
                # Set the common start time for both audio and video
                self.playback_start_time = time.time() + 0.2  # Small delay to ensure both threads are ready
                
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
        """Capture frames with synchronization to audio."""
        logger.info("Video thread ready, waiting for warm-up to complete")
        
        # Calculate frame time based on FPS
        frame_time = 1.0 / self._video_fps if self._video_fps > 0 else 0.033  # Default to ~30fps if fps is 0
        
        # Wait for warm-up to complete before doing anything
        while not self.warm_up_complete.is_set() and not self._shutdown_event.is_set():
            # Just keep the thread alive during warm-up
            time.sleep(0.1)
        
        if self._shutdown_event.is_set():
            return
            
        # Now that warm-up is complete, initialize video playback
        logger.info(f"Starting video playback at {self.start_time:.2f}s")
        
        # Reset video capture to the beginning
        self._cap.release()
        self._cap = cv2.VideoCapture(self.source)
        self._frame_count = int(self.start_time * self._video_fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_count)
        
        # Clear any existing frames in the queue
        while not self.frames_queue.empty():
            try:
                self.frames_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for the sync signal
        self.start_sync_event.wait()
        
        # Set common start time for sync
        self.playback_start_time = time.time()
        logger.info(f"Video playback started at {self.playback_start_time}")
        
        try:
            while self.is_running and not self._shutdown_event.is_set():
                start_time = time.time()
                
                # Read the next frame
                ret, frame = self._cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                # Calculate timestamp for this frame based on frame number
                current_pts = (self._frame_count - int(self.start_time * self._video_fps)) * frame_time
                self._frame_count += 1
                
                # Queue frame with timestamp
                current_time = time.time()
                qsize = self.frames_queue.qsize()
                
                # Only log queue status if it's getting full (over 80%)
                if qsize > self.frames_queue.maxsize * 0.8:
                    if current_time - getattr(self, 'last_queue_warning', 0) > 1.0:  # Rate limit warnings
                        logger.warning(f"Frame queue {qsize}/{self.frames_queue.maxsize} ({(qsize/self.frames_queue.maxsize*100):.1f}% full)")
                        self.last_queue_warning = current_time
                
                # Try to put frame without blocking
                try:
                    self.frames_queue.put_nowait((frame, current_pts))
                    logger.debug(f"Queued frame at {current_pts:.2f}s (qsize: {qsize+1})")
                except queue.Full:
                    if current_time - getattr(self, 'last_queue_warning', 0) > 1.0:  # Rate limit warnings
                        logger.warning(f"Frame queue full ({qsize}), dropping frame at {current_pts:.2f}s")
                        self.last_queue_warning = current_time
                
                # Calculate time to sleep to maintain frame rate
                process_time = time.time() - start_time
                sleep_time = max(0, frame_time - process_time)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in frame capture: {e}")
        finally:
            self._cap.release()
    
    def _play_audio(self, audio_file):
        """Play audio using sounddevice with proper stream handling to avoid distortion."""
        try:
            logger.info(f"Preparing audio playback from {audio_file}")
            
            # Load audio file
            data, sample_rate = sf.read(audio_file)
            
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
                    logger.warning(f"Audio callback status: {status}")
                
                # Check if we've reached the end of the audio
                if frames_played + frames > len(data):
                    # Partial frame at the end
                    remaining = len(data) - frames_played
                    if remaining > 0:
                        outdata[:remaining] = data[frames_played:frames_played+remaining]
                        outdata[remaining:].fill(0)
                    else:
                        outdata.fill(0)
                    
                    # Update position before stopping
                    current_time = frames_played / sample_rate
                    with self.audio_position_lock:
                        self.audio_position = current_time
                        
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
                # Calculate the current time including the start time offset
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
        logger.info("Stopping video source")
        self.is_running = False
        self.audio_playing = False
        self._shutdown_event.set()
        
        if hasattr(self, 'video_thread') and self.video_thread:
            self.video_thread.join(timeout=1.0)
            
        if hasattr(self, 'audio_thread') and self.audio_thread:
            self.audio_thread.join(timeout=1.0)
            
        if hasattr(self, '_cap') and self._cap:
            self._cap.release()
            
        # Stop any playing audio
        sd.stop()
            
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
