import cv2
import numpy as np
import os
import time
import threading
from queue import Queue
import logging
import sounddevice as sd
import subprocess
import soundfile as sf

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoSource:
    def __init__(self, source):
        """Initialize video source with audio handling capabilities.
        
        Args:
            source: Path to video file (should be in ~/.video_cache)
        """
        # Use source directly if it's an absolute path, otherwise assume it's in cache
        self.source = source
        if not os.path.isabs(source):
            self.source = os.path.join(os.path.expanduser("~/.video_cache"), source)
            
        # Initialize queue for synchronized playback
        self.frames_queue = Queue(maxsize=30)
        
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
        
        # Synchronization objects
        self.audio_position = 0.0  # Current audio position in seconds
        self.audio_position_lock = threading.RLock()
        self.start_sync_event = threading.Event()  # For exact start synchronization
        self.playback_start_time = 0.0  # Common reference point for sync
        
    def start(self):
        """Start video capture and processing."""
        with self._lock:
            if self.is_running:
                return
            
            try:
                # Start video capture directly from source
                logger.info(f"Opening video source: {self.source}")
                self._cap = cv2.VideoCapture(self.source)
                
                if not self._cap.isOpened():
                    raise RuntimeError(f"Failed to open video: {self.source}")
                    
                # Get video info
                self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
                self._frame_count = 0
                
                # Check for existing audio file in cache
                audio_file = self._get_audio_file_path()
                
                # Initialize is_running before starting threads
                self.is_running = True
                
                # Clear sync event before starting threads
                self.start_sync_event.clear()
                
                # Start both threads first, but they'll wait for the sync event
                logger.info("Starting media threads (waiting for sync)")
                
                # Start video thread
                self.video_thread = threading.Thread(target=self._capture_frames)
                self.video_thread.daemon = True
                self.video_thread.start()
                
                # Start audio playback thread if audio file exists
                has_audio = False
                if audio_file:
                    has_audio = True
                    self.audio_playing = True
                    self.audio_thread = threading.Thread(target=self._play_audio, args=(audio_file,))
                    self.audio_thread.daemon = True
                    self.audio_thread.start()
                else:
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
        # First, check if there's a matching WAV file in the same folder
        video_basename = os.path.splitext(self.source)[0]
        audio_file = f"{video_basename}.wav"
        
        if os.path.exists(audio_file):
            logger.info(f"Found existing audio file: {audio_file}")
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
            return audio_file
        else:
            logger.warning("Audio file extraction failed or file is empty")
            return None
            
    def _capture_frames(self):
        """Capture frames with synchronization to audio."""
        logger.info("Video thread ready, waiting for sync signal")
        
        # Wait for the sync signal
        self.start_sync_event.wait()
        
        # Wait until the exact start time
        sleep_time = max(0, self.playback_start_time - time.time())
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        logger.info("Starting video playback")
        
        frame_time = 1.0 / self._video_fps
        
        try:
            while self.is_running and not self._shutdown_event.is_set():
                # Get current audio position (this is our master clock)
                with self.audio_position_lock:
                    current_audio_time = self.audio_position
                
                # Determine which frame should be displayed at this time
                target_frame = int(current_audio_time * self._video_fps)
                
                # If we're ahead of where we should be, skip frames
                if self._frame_count < target_frame:
                    # Skip ahead to catch up with audio
                    frames_to_skip = target_frame - self._frame_count
                    if frames_to_skip > 1:
                        logger.debug(f"Skipping {frames_to_skip} frames to sync with audio")
                        # Skip frames to catch up
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        self._frame_count = target_frame
                
                # Read the next frame
                ret, frame = self._cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                # Calculate timestamp for this frame based on frame number
                current_pts = self._frame_count * frame_time
                self._frame_count += 1
                
                # Queue frame with timestamp
                try:
                    self.frames_queue.put((frame, current_pts))
                except:
                    pass
                
                # If no audio, use video timing
                if not self.audio_playing:
                    # Calculate when this frame should be displayed
                    elapsed_time = time.time() - self.playback_start_time
                    target_time = self.playback_start_time + current_pts
                    now = time.time()
                    sleep_time = max(0, target_time - now)
                    
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 0.1))  # Cap at 100ms to stay responsive
                else:
                    # Yield to other threads briefly
                    time.sleep(0.001)
                
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
                
            logger.info("Starting audio playback")
            
            # Tracking variables
            frames_played = 0
            
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
            
    def get_audio_chunk(self):
        """Get the next audio chunk with its timestamp.
        For transcription purposes, we use the audio position which is our master clock.
        """
        if not self.is_running:
            return None
            
        try:
            # Use the current audio position for precise timing
            with self.audio_position_lock:
                current_time = self.audio_position
                
            # Return dummy audio data with accurate current timestamp
            audio_data = np.zeros(1600, dtype=np.float32)
            return (audio_data, current_time)
        except Exception as e:
            logger.error(f"Error getting audio chunk: {e}")
            return None
            
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
