import numpy as np
import threading
import queue
from queue import Queue
import time
import os
import torch
from faster_whisper import WhisperModel

# Import with try-except to handle both direct execution and module import
try:
    from logging_utils import get_logger, setup_logging, TRACE
except ImportError:
    from .logging_utils import get_logger, setup_logging, TRACE

# Get logger instance with module name for better filtering
logger = get_logger('transcription')

# Configure logging for faster_whisper to reduce verbosity
import logging
logging.getLogger('faster_whisper').setLevel(logging.WARNING)

class TranscriptionEngine:
    playback_start_time = 0.0  # Global playback time origin for sync
    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "auto"):
        """Initialize the transcription engine.
        
        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
            device: Device to run the model on ('cuda', 'cpu', or 'auto' for automatic detection)
            compute_type: Computation type ('float16', 'int8', or 'auto' for automatic selection)
        """
        self.model_size = model_size
        self.sampling_rate = 16000  # Whisper's native sampling rate
        
        # Initialize with default values, will be set in _init_model
        self.device = "cpu"
        self.compute_type = "int8"
        
        # Processing state
        self.is_running = False
        self._warm_up_mode = True
        self.processed_chunks = 0
        self.average_processing_time = 0.0
        self.playback_start_time = 0.0
        
        # Threading and queues
        self.audio_queue = queue.Queue(maxsize=15)  # Increased queue size for warm-up
        self.result_queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.audio_processor_thread: Optional[threading.Thread] = None
        
        # Initialize the model with automatic fallback
        self._init_model(device, compute_type)

    def _init_model(self, device_pref: str = "auto", compute_type_pref: str = "auto"):
        """Initialize the Whisper model with automatic fallback for device and compute type.
        
        Args:
            device_pref: Preferred device ('cuda', 'cpu', or 'auto')
            compute_type_pref: Preferred compute type ('float16', 'int8', or 'auto')
        """
        model_dir = os.path.expanduser("~/.cache/faster-whisper")
        
        # Determine device to use
        use_cuda = False
        if device_pref == "auto":
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                try:
                    # Test if CUDA is actually working
                    torch.zeros(1).cuda()
                    self.device = "cuda"
                except Exception as e:
                    logger.warning(f"CUDA is available but not working: {e}. Falling back to CPU.")
                    use_cuda = False
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device_pref.lower()
            use_cuda = self.device == "cuda"
        
        # Determine compute type based on device
        if compute_type_pref == "auto":
            self.compute_type = "float16" if use_cuda else "int8"
        else:
            self.compute_type = compute_type_pref
        
        logger.info(f"Initializing Whisper model (size={self.model_size}, device={self.device}, "
                  f"compute_type={self.compute_type})")
        
        try:
            os.makedirs(model_dir, exist_ok=True)
            logger.debug(f"Model cache directory: {model_dir}")
            
            # Try to initialize with preferred settings
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=model_dir
                )
                logger.info(f"Successfully initialized Whisper model on {self.device.upper()}")
                
            except Exception as e:
                if use_cuda:  # If CUDA failed, try falling back to CPU
                    logger.warning(f"Failed to initialize with CUDA: {e}. Falling back to CPU.")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                        download_root=model_dir
                    )
                    logger.info("Successfully initialized Whisper model on CPU")
                else:
                    raise  # Re-raise if we're already on CPU
                    
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            logger.debug("Model initialization failed", exc_info=True)
            raise

        
    # No model initialization needed for MVP
            
    def start_transcription(self) -> bool:
        """Start the transcription engine.
        
        Returns:
            bool: True if the transcription was started, False if it was already running
        """
        if self.is_running:
            logger.debug("Transcription already running, ignoring start request")
            return False
            
        try:
            logger.info("Starting transcription engine")
            self.is_running = True
            self._warm_up_mode = True  # Start in warm-up mode
            
            # Clear any existing queue state
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except queue.Empty:
                    break
            
            # Start the worker thread
            self.thread = threading.Thread(
                target=self._transcription_worker,
                name="TranscriptionWorker",
                daemon=True
            )
            self.thread.start()
            
            logger.info("Transcription engine started in warm-up mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start transcription: {e}", exc_info=True)
            self.is_running = False
            return False
        
    def stop_transcription(self, timeout: float = 5.0) -> bool:
        """Stop the transcription engine gracefully.
        
        Args:
            timeout: Maximum time to wait for the worker thread to stop (seconds)
            
        Returns:
            bool: True if stopped cleanly, False if timeout occurred or already stopped
        """
        if not self.is_running:
            logger.debug("Transcription not running, ignoring stop request")
            return True
            
        logger.info("Stopping transcription engine...")
        self.is_running = False
        self._warm_up_mode = False
        
        try:
            # Clear the audio queue
            cleared = 0
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                    cleared += 1
                except queue.Empty:
                    break
            
            if cleared > 0:
                logger.debug(f"Cleared {cleared} pending audio segments from queue")
            
            # Signal and wait for the worker thread to finish
            if self.thread and self.thread.is_alive():
                logger.debug("Waiting for transcription worker to finish...")
                self.thread.join(timeout=timeout)
                
                if self.thread.is_alive():
                    logger.warning("Transcription worker did not stop gracefully")
                    return False
            
            # Clear the result queue
            cleared = 0
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.task_done()
                    cleared += 1
                except queue.Empty:
                    break
                    
            if cleared > 0:
                logger.debug(f"Cleared {cleared} pending results from queue")
            
            logger.info("Transcription engine stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during transcription shutdown: {e}", exc_info=True)
            return False

    def add_audio_segment(self, audio_segment: tuple) -> bool:
        """Add an audio segment to the processing queue.
        
        Args:
            audio_segment: Tuple of (audio_data, timestamp)
            
        Returns:
            bool: True if segment was added successfully, False if queue was full
        """
        if not self.is_running:
            logger.warning("Transcription not running, ignoring audio segment")
            return False
            
        try:
            if self._warm_up_mode:
                # During warm-up, wait a bit if queue is full to avoid dropping chunks
                try:
                    self.audio_queue.put(audio_segment, timeout=0.5)
                    return True
                except queue.Full:
                    logger.debug("Queue full during warm-up, waiting for space...")
                    return False
            else:
                # During normal operation, use non-blocking put
                self.audio_queue.put_nowait(audio_segment)
                return True
        except queue.Full:
            if not self._warm_up_mode:  # Only log warning if not in warm-up
                logger.warning("Audio queue full, dropping segment")
            return False
            
    def get_transcription(self):
        """Get the next available transcription result.
        
        Returns:
            dict or None: Transcription result with keys 'text', 'timestamp', 'duration',
                        or None if no result is available.
        """
        if not self.is_running:
            logger.trace("Transcription service not running, no results available")
            return None
            
        if self.result_queue.empty():
            logger.trace("No transcription results available in queue")
            return None
            
        try:
            result = self.result_queue.get_nowait()
            
            # Log at TRACE level to avoid log spam in production
            logger.trace(f"Retrieved transcription: '{result.get('text', '')}' "
                       f"at {result.get('timestamp', 0):.2f}s "
                       f"(queue size: {self.result_queue.qsize()})")
            
            return result
            
        except queue.Empty:
            logger.trace("No transcription results available (queue empty)")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving transcription: {e}", exc_info=True)
            return None
            
    def _transcription_worker(self):
        """Worker thread that processes audio segments and generates transcriptions.
        
        Logging Levels:
        - INFO: Major operations, transcription results
        - DEBUG: Processing decisions, buffer management
        - TRACE: Detailed audio analysis, timing calculations
        """
        logger.info("Starting transcription worker thread")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                # Get next audio segment
                try:
                    audio_data, start_time = self.audio_queue.get(timeout=0.5)
                    logger.debug(f"Processing audio segment at {start_time:.3f}s, "
                                f"queue size: {self.audio_queue.qsize()}")
                    
                    # Log audio format details at DEBUG level
                    if audio_data is not None:
                        audio_format = audio_data.dtype if hasattr(audio_data, 'dtype') else type(audio_data[0]).__name__
                        audio_shape = audio_data.shape if hasattr(audio_data, 'shape') else 'N/A'
                        logger.debug(f"Audio format: {audio_format}, shape: {audio_shape}")
                    
                    consecutive_errors = 0  # Reset error counter on successful get
                    
                except queue.Empty:
                    logger.trace("Audio queue empty, waiting for segments...")
                    time.sleep(0.1)
                    continue
                
                # Skip empty segments
                if audio_data is None or len(audio_data) == 0:
                    logger.warning("Skipping empty audio segment")
                    self.audio_queue.task_done()
                    continue
                
                try:
                    # Get current audio position for timing comparison
                    current_time = getattr(self, 'current_audio_time', 0)
                    
                    # Log audio segment details at TRACE level
                    logger.trace(f"Audio segment details - "
                               f"shape: {audio_data.shape}, "
                               f"dtype: {audio_data.dtype}, "
                               f"duration: {len(audio_data)/self.sampling_rate:.2f}s")
                    
                    # Log model inference start
                    logger.debug("Starting audio transcription...")
                    start_process_time = time.time()
                    
                    # Transcribe the audio
                    segments = []
                    try:
                        segments, _ = self.model.transcribe(
                            audio_data,
                            language="en",
                            beam_size=5,
                            word_timestamps=True
                        )
                        segments = list(segments)  # Convert generator to list to catch errors early
                        
                        # Log successful transcription
                        processing_time = time.time() - start_process_time
                        logger.debug(f"Transcription completed in {processing_time:.2f}s, "
                                   f"found {len(segments)} segments")
                        
                    except Exception as e:
                        logger.error(f"Transcription failed: {e}", exc_info=True)
                        raise
                    
                    # Process each transcribed segment
                    for segment in segments:
                        if not segment.text.strip():
                            logger.debug("Skipping empty transcription segment")
                            continue
                            
                        # Calculate absolute timestamps
                        segment_start = self.playback_start_time + segment.start
                        segment_end = self.playback_start_time + segment.end
                        duration = segment_end - segment_start
                        
                        # Log transcription result at INFO level
                        confidence = getattr(segment, 'confidence', None)
                        confidence_str = f", confidence: {confidence:.2f}" if confidence is not None else ""
                        logger.info(f"Transcribed: '{segment.text.strip()}' "
                                  f"({segment_start:.2f}s-{segment_end:.2f}s{confidence_str})")
                        
                        # Add to result queue for display
                        try:
                            result = {
                                'text': segment.text.strip(),
                                'timestamp': segment_start,
                                'duration': duration
                            }
                            # Only add confidence if available
                            if confidence is not None:
                                result['confidence'] = confidence
                            self.result_queue.put_nowait(result)
                            logger.trace(f"Added to result queue: {result}")
                            
                        except queue.Full:
                            logger.warning("Result queue full, dropping transcription")
                    
                    # Update processing statistics
                    self.processed_chunks += 1
                    processing_time = time.time() - start_process_time
                    self.average_processing_time = (
                        (self.average_processing_time * (self.processed_chunks - 1) + processing_time) 
                        / self.processed_chunks
                    )
                    
                    logger.debug(f"Processing stats - "
                               f"chunks: {self.processed_chunks}, "
                               f"avg time: {self.average_processing_time:.2f}s")
                    
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error in transcription: {e}", exc_info=True)
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical(f"Reached {consecutive_errors} consecutive errors, stopping worker")
                        self.is_running = False
                        break
                        
                finally:
                    self.audio_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                time.sleep(0.5)  # Prevent tight error loop
                
        logger.info("Transcription worker stopped")

    def process_audio(self, video_source, chunk_size: float = 3.0, overlap: float = 1.0):
        """Process audio from video source and send to transcriber.
        
        This method processes audio in chunks during warm-up and continues
        processing during playback for real-time transcription.
        
        Args:
            video_source: The video source to get audio chunks from
            chunk_size: Size of audio chunks in seconds (default: 3.0s)
            overlap: Overlap between chunks in seconds (default: 1.0s)
            
        Notes:
            - Uses a sliding window approach for better transcription continuity
            - Automatically adjusts queue behavior during warm-up vs. normal operation
            - Implements backpressure to prevent queue overflow
        """
        logger.info(f"Starting audio processing (chunk_size={chunk_size}s, overlap={overlap}s)")
        
        # Validate parameters
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            logger.error(f"Invalid chunk parameters: size={chunk_size}s, overlap={overlap}s")
            return
            
        last_process_time = 0
        video_start_time = getattr(video_source, 'start_time', 0.0)
        chunk_counter = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            while getattr(video_source, 'is_running', False) and not getattr(video_source, '_shutdown_event', None).is_set():
                try:
                    # Get current time or position
                    if getattr(video_source, 'warm_up_complete', None) and video_source.warm_up_complete.is_set():
                        current_time = time.time() - video_source.playback_start_time + video_start_time
                        in_warmup = False
                    else:
                        with getattr(video_source, 'audio_position_lock', threading.Lock()):
                            current_time = video_source.audio_position + video_start_time
                        in_warmup = True
                    
                    # Process audio in chunks with overlap
                    time_since_last = current_time - last_process_time
                    if time_since_last >= (chunk_size - overlap):
                        chunk_counter += 1
                        chunk_id = f"#{chunk_counter} ({'warmup' if in_warmup else 'normal'})"
                        
                        # Get audio chunk from current position with specified size
                        try:
                            audio_chunk = video_source.get_audio_chunk(chunk_size=chunk_size)
                            
                            if audio_chunk is not None and len(audio_chunk[0]) > 0:
                                audio_data, timestamp = audio_chunk
                                
                                # Log chunk details at appropriate level
                                log_msg = (f"Chunk {chunk_id} at {current_time:.2f}s: "
                                         f"{len(audio_data)} samples, "
                                         f"duration={len(audio_data)/self.sampling_rate:.2f}s, "
                                         f"queue={self.audio_queue.qsize()}/{self.audio_queue.maxsize}")
                                
                                if in_warmup:
                                    logger.debug(log_msg)
                                else:
                                    logger.trace(log_msg)
                                
                                # Add to transcription engine with backpressure
                                if self.add_audio_segment(audio_chunk):
                                    last_process_time = current_time
                                    consecutive_failures = 0  # Reset failure counter on success
                                else:
                                    consecutive_failures += 1
                                    logger.warning(f"Failed to add chunk {chunk_id} to queue (failures: {consecutive_failures})")
                                    
                                    # If we're failing too much, slow down processing
                                    if consecutive_failures > max_consecutive_failures:
                                        logger.warning("Too many consecutive failures, pausing to clear queue...")
                                        time.sleep(0.5)
                                        consecutive_failures = 0  # Reset after pause
                                        continue
                            else:
                                logger.warning(f"Received empty audio chunk at {current_time:.2f}s")
                                
                        except Exception as e:
                            consecutive_failures += 1
                            logger.error(f"Error getting audio chunk: {e}", exc_info=consecutive_failures > 1)
                            
                            if consecutive_failures > max_consecutive_failures:
                                logger.critical("Too many errors, stopping audio processing")
                                break
                                
                            time.sleep(0.5)  # Back off on errors
                            continue
                    
                    # Adaptive sleep to balance between responsiveness and CPU usage
                    sleep_time = 0.05  # Base sleep time (50ms)
                    
                    # If queue is getting full, slow down to let the worker catch up
                    queue_ratio = self.audio_queue.qsize() / max(1, self.audio_queue.maxsize)
                    if queue_ratio > 0.7:  # If queue is more than 70% full
                        sleep_time = min(0.5, sleep_time * (1 + queue_ratio * 2))  # Up to 0.5s
                    
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Unexpected error in audio processing loop: {e}", exc_info=True)
                    time.sleep(0.5)  # Prevent tight error loop
                    
        except Exception as e:
            logger.critical(f"Fatal error in audio processing: {e}", exc_info=True)
            raise
            
        finally:
            logger.info("Audio processing thread stopped")
    
    def __enter__(self):
        self.start_transcription()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_transcription()
