import numpy as np
import threading
import queue
from queue import Queue
import time
import os
from faster_whisper import WhisperModel
import torch

# Import with try-except to handle both direct execution and module import
try:
    from logging_utils import get_logger, setup_logging
except ImportError:
    from .logging_utils import get_logger, setup_logging

# Get logger instance
logger = get_logger(__name__)

# Set faster_whisper logger to WARNING level to reduce verbosity
import logging
logging.getLogger('faster_whisper').setLevel(logging.WARNING)

class TranscriptionEngine:
    playback_start_time = 0.0  # Global playback time origin for sync
    def __init__(self, config=None):
        """Initialize the transcription engine.
        
        Args:
            config: Optional configuration dictionary (not currently used)
        """
        self.sampling_rate = 16000
        self.chunk_size = 10.0  # Process 10-second chunks for better context
        self.audio_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=20)
        self.is_running = False
        self.latest_timestamp = 0.0
        self.last_emitted_caption_time = 0.0
        self.transcription_thread = None
        self.processing_start_time = 0.0  # Track when processing starts
        self.processed_chunks = 0  # Count of processed chunks
        self.average_processing_time = 0.0  # Track average processing time
        
        # Initialize faster-whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model_dir = os.path.expanduser("~/.cache/faster-whisper")
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Loading faster-whisper model on {device} with {compute_type} precision")
        self.model = WhisperModel(
            "small",  # Using small for better speed/accuracy balance
            device="cpu",
            compute_type="int8",
            download_root=model_dir
        )
        logger.info("Model loaded successfully")

        
    # No model initialization needed for MVP
            
    def start_transcription(self):
        """Start the transcription thread."""
        if self.is_running:
            return
            
        self.is_running = True
        self.transcription_thread = threading.Thread(target=self._transcription_worker)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        logger.info("Transcription thread started")
        
    def stop_transcription(self):
        """Stop transcription thread."""
        self.is_running = False
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join(timeout=1.0)
        logger.info("Transcription stopped")
            
    def add_audio_segment(self, audio_data):
        """Add audio segment to processing queue."""
        if not self.is_running:
            return
            
        try:
            # Expecting a tuple of (audio_data, timestamp)
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                # Extract audio data and timestamp
                audio_segment, timestamp = audio_data
                
                # Store in queue for processing
                self.audio_queue.put((audio_segment, timestamp))
                
                # Update the latest timestamp directly
                self.latest_timestamp = max(self.latest_timestamp, timestamp)
            else:
                logger.warning("Received malformed audio data")
        except Exception as e:
            logger.error(f"Error adding audio segment: {e}")
                
    def get_transcription(self):
        """Get the next available transcription result."""
        if self.result_queue.empty():
            return None
            
        try:
            result = self.result_queue.get_nowait()
            return result
        except:
            return None
            
    def _transcription_worker(self):
        """Worker thread that processes audio segments and generates transcriptions."""
        logger.info("Transcription worker started")
        
        while self.is_running:
            try:
                # Get next audio segment
                try:
                    audio_data, start_time = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Skip empty segments
                if audio_data is None or len(audio_data) == 0:
                    logger.warning("Skipping empty audio segment")
                    self.audio_queue.task_done()
                    continue
                
                try:
                    # Get current audio position for timing comparison
                    current_time = getattr(self, 'current_audio_time', 0)
                    
                    # Transcribe the audio
                    segments, _ = self.model.transcribe(
                        audio_data,
                        language="en",
                        beam_size=5,
                        word_timestamps=True
                    )
                    
                    # Process each transcribed segment
                    for segment in segments:
                        if not segment.text.strip():
                            continue
                            
                        # Calculate absolute timestamps
                        # Use global playback_start_time as the timestamp origin for all captions
                        start = self.playback_start_time + segment.start
                        end = self.playback_start_time + segment.end
                        logger.info(f"[TIMING] Caption segment: '{segment.text.strip()}' start={start:.2f} end={end:.2f} (origin={self.playback_start_time:.2f})")
                        
                        # Calculate timing difference from current audio position
                        time_diff = start - current_time
                        
                        # Format the timestamp with color coding
                        if time_diff > 1.0:  # More than 1 second ahead
                            time_str = f"\033[32m+{time_diff:.2f}s\033[0m"  # Green
                        elif time_diff < -1.0:  # More than 1 second behind
                            time_str = f"\033[31m{time_diff:+.2f}s\033[0m"  # Red
                        else:  # Within 1 second (on time)
                            time_str = f"\033[33m{time_diff:+.2f}s\033[0m"  # Yellow
                        
                        # Log the transcription with relative timing
                        logger.info(f"Transcribed {time_str}: '{segment.text.strip()}' (abs: {start:.2f}s-{end:.2f}s)")
                        
                        # Add to result queue for display
                        try:
                            self.result_queue.put_nowait({
                                'text': segment.text.strip(),
                                'timestamp': start,
                                'duration': end - start
                            })
                        except queue.Full:
                            logger.warning("Result queue full, dropping transcription")
                        

                
                except Exception as e:
                    logger.error(f"Error in transcription: {e}", exc_info=True)
                
                finally:
                    self.audio_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("Transcription worker stopped")

    def process_audio(self, video_source):
        """Process audio from video source and send to transcriber.
        
        This method processes audio in chunks during warm-up and continues
        processing during playback for real-time transcription.
        
        Args:
            video_source: The video source to get audio chunks from
        """
        logger.info("Starting audio processing thread")
        chunk_size = 3.0  # Process 3-second chunks
        overlap = 1.0     # 1-second overlap for better continuity
        last_process_time = 0
        video_start_time = video_source.start_time if hasattr(video_source, 'start_time') else 0.0
        
        try:
            while video_source.is_running and not video_source._shutdown_event.is_set():
                try:
                    # Get current time or position
                    if video_source.warm_up_complete.is_set():
                        current_time = time.time() - video_source.playback_start_time + video_start_time
                    else:
                        with video_source.audio_position_lock:
                            current_time = video_source.audio_position + video_start_time
                    
                    # Process audio in chunks with overlap
                    if current_time >= last_process_time + (chunk_size - overlap):
                        # Get audio chunk from current position with specified size
                        audio_chunk = video_source.get_audio_chunk(chunk_size=chunk_size)
                        if audio_chunk is not None and len(audio_chunk[0]) > 0:
                            # Add to transcription engine using the proper method
                            self.add_audio_segment(audio_chunk)
                            last_process_time = current_time
                            
                            # Log progress with absolute timestamp
                            if video_source.warm_up_complete.is_set():
                                logger.debug(f"Added audio chunk at {current_time:.2f}s to queue")
                            else:
                                logger.debug(f"Warm-up: Added chunk at {current_time:.2f}s to queue")
                    
                    # Small delay to prevent busy waiting
                    time.sleep(0.05)
                    
                except Exception as e:
                    logger.error(f"Error in audio processing: {e}", exc_info=True)
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Fatal error in audio processing: {e}", exc_info=True)
        finally:
            logger.info("Audio processing thread stopped")
    
    def __enter__(self):
        self.start_transcription()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_transcription()
