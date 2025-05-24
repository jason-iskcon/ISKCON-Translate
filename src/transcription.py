import numpy as np
import threading
import queue
from queue import Queue
import time
import os
import collections
import torch
import psutil
import logging
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

# Configure logging for third-party libraries
for lib in ['numba', 'huggingface_hub', 'whisper', 'faster_whisper']:
    logging.getLogger(lib).setLevel(logging.WARNING)

# Suppress specific warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numba')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import with try-except to handle both direct execution and module import
try:
    from logging_utils import get_logger, setup_logging, TRACE
except ImportError:
    from .logging_utils import get_logger, setup_logging, TRACE

# Import singleton clock
try:
    from clock import CLOCK
except ImportError:
    from .clock import CLOCK

# Get logger instance with module name for better filtering
logger = get_logger('transcription')

# Configure logging for faster_whisper to reduce verbosity
import logging
logging.getLogger('faster_whisper').setLevel(logging.WARNING)

# Device-specific configuration
CPU_PARAMS = {
    'chunk_size': 1.0,      # Smaller chunks for faster CPU processing
    'overlap': 0.2,         # Smaller overlap
    'queue_maxsize': 20,    # Larger queue to handle 4 workers (was 15)
    'n_workers': 4          # Use 4 workers for CPU to improve throughput (was 3)
}

GPU_PARAMS = {
    'chunk_size': 3.0,      # Larger chunks for GPU (better context for Whisper)
    'overlap': 1.0,         # Larger overlap for better continuity
    'queue_maxsize': 10,    # Smaller queue since GPU processes much faster
    'n_workers': 1          # Single worker for GPU (80ms per 3s chunk = massive headroom)
}

class TranscriptionEngine:
    playback_start_time = 0.0  # Global playback time origin for sync
    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "auto", warm_up_complete_event: threading.Event = None):
        """Initialize the transcription engine.
        
        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
            device: Device to run the model on ('cuda', 'cpu', or 'auto' for automatic detection)
            compute_type: Computation type ('float16', 'int8', or 'auto' for automatic selection)
            warm_up_complete_event: Event to signal when warm-up phase is complete
        """
        self.model_size = model_size
        self.sampling_rate = 16000  # Whisper's native sampling rate
        self.warm_up_complete_event = warm_up_complete_event
        
        # Initialize with default values, will be set in _init_model
        self.device = "cpu"
        self.compute_type = "int8"
        
        # Validate GPU availability for production deployment
        if device == "auto":
            if not torch.cuda.is_available():
                logger.error("🚨 PRODUCTION ERROR: CUDA not available! GPU mode required for production.")
                logger.error("   → Check NVIDIA driver installation")
                logger.error("   → Verify CUDA runtime compatibility") 
                logger.error("   → Restart application after driver update")
                raise RuntimeError("GPU acceleration required but CUDA not available")
            else:
                logger.info("✅ CUDA available - will attempt GPU initialization")
        
        # Initialize the model first to determine actual device
        self._init_model(device, compute_type)
        
        # Set device-specific parameters based on actual device
        params = CPU_PARAMS if self.device == "cpu" else GPU_PARAMS
        self.chunk_size = params['chunk_size']
        self.overlap = params['overlap']
        self.n_workers = params['n_workers']
        queue_maxsize = params['queue_maxsize']
        
        logger.info(f"🔧 Using {self.device.upper()} parameters: chunk_size={self.chunk_size}s, overlap={self.overlap}s, queue_maxsize={queue_maxsize}, n_workers={self.n_workers}")
        
        # Processing state
        self.is_running = False
        self._warm_up_mode = True
        self.processed_chunks = 0
        self.average_processing_time = 0.0
        self.playback_start_time = 0.0
        
        # Rate limiting for log messages
        self._last_adaptive_drop_warning_time = 0
        
        # Track drop statistics separately from failures
        self.drops_last_minute = collections.deque(maxlen=60)  # Track drops per second for last minute
        self.drop_stats_lock = threading.Lock()
        
        # Threading and queues
        self.audio_queue = queue.Queue(maxsize=queue_maxsize)
        logger.info(f"🔧 CONSTRUCTOR: TranscriptionEngine audio_queue created with maxsize={self.audio_queue.maxsize}")
        self.result_queue = queue.Queue()
        self.worker_threads = []
        self.audio_processor_thread: Optional[threading.Thread] = None
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.n_workers,
            thread_name_prefix="TransWorker"
        ) if self.n_workers > 1 else None

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
            
            # Clear any existing queue state and reset counters
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except queue.Empty:
                    break
            
            # Clear result queue
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.task_done()
                except queue.Empty:
                    break
            
            # Reset processing state
            self.processed_chunks = 0
            self.average_processing_time = 0.0
            
            # Start worker threads based on n_workers
            self.worker_threads = []
            for i in range(self.n_workers):
                worker = threading.Thread(
                    target=self._transcription_worker,
                    name=f"TranscriptionWorker-{i+1}",
                    daemon=True
                )
                worker.start()
                self.worker_threads.append(worker)
                
                # Stagger worker starts to reduce initial load spikes
                if i < self.n_workers - 1:  # Don't sleep after the last worker
                    stagger_delay = self.chunk_size / 2  # Half chunk duration
                    logger.debug(f"Staggering worker start by {stagger_delay:.1f}s")
                    time.sleep(stagger_delay)
            
            logger.info(f"Started {self.n_workers} transcription worker(s) in warm-up mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start transcription: {e}", exc_info=True)
            self.is_running = False
            return False
        
    def stop_transcription(self, timeout: float = 5.0) -> bool:
        """Stop the transcription engine gracefully.
        
        Args:
            timeout: Maximum time to wait for the worker threads to stop (seconds)
            
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
            
            # Signal and wait for all worker threads to finish
            if self.worker_threads:
                logger.debug(f"Waiting for {len(self.worker_threads)} worker threads to finish...")
                
                # Calculate timeout per worker
                worker_timeout = max(0.1, timeout / max(1, len(self.worker_threads)))
                all_stopped = True
                
                for i, worker in enumerate(self.worker_threads):
                    if worker.is_alive():
                        worker.join(timeout=worker_timeout)
                        if worker.is_alive():
                            logger.warning(f"Worker thread {i+1} did not stop gracefully")
                            all_stopped = False
                
                if not all_stopped:
                    logger.warning("Not all worker threads stopped gracefully")
                
                self.worker_threads = []
            
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
            
            # Shutdown thread pool if it exists
            if hasattr(self, 'thread_pool') and self.thread_pool is not None:
                self.thread_pool.shutdown(wait=False)
                self.thread_pool = None
            
            # Reset playback start time for next session
            self.playback_start_time = 0.0
            logger.info("Transcription engine stopped successfully")
            return all_stopped
            
        except Exception as e:
            logger.error(f"Error during transcription shutdown: {e}", exc_info=True)
            return False

    def add_audio_segment(self, audio_segment: tuple) -> bool:
        """Add an audio segment to the processing queue.
        
        This method implements a drop-oldest strategy when the queue is full to ensure
        the most recent audio data is always processed, preventing buffer overruns.
        
        Args:
            audio_segment: Tuple of (audio_data, timestamp)
            
        Returns:
            bool: True if segment was added successfully, False if queue was full or invalid input
            
        Raises:
            ValueError: If audio_segment is not a tuple of (audio_data, timestamp)
        """
        # Input validation
        if not isinstance(audio_segment, tuple) or len(audio_segment) != 2:
            raise ValueError("audio_segment must be a tuple of (audio_data, timestamp)")
            
        audio_data, timestamp = audio_segment
        
        # Check if audio data is valid
        if not isinstance(audio_data, np.ndarray):
            logger.error("Invalid audio data: expected numpy array")
            return False
            
        if len(audio_data) == 0:
            logger.warning("Received empty audio segment")
            return False
            
        if not self.is_running:
            logger.debug("Transcription not running, ignoring audio segment")
            return False
            
        # Set playback_start_time to the timestamp of the first segment if not already set
        if self.playback_start_time == 0.0 and timestamp > 0:
            self.playback_start_time = timestamp
            logger.debug(f"Set playback start time to {timestamp:.2f}s")
        
        # Skip if we're in warm-up mode and queue is already full
        if self._warm_up_mode and self.audio_queue.full():
            logger.debug("Skipping audio segment during warm-up (queue full)")
            return True  # This is normal queue management, not a failure
            
        # Persistent retry loop to ensure queue operations always succeed
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Try to add segment to queue directly
                self.audio_queue.put(audio_segment, block=False)
                return True  # Success!
                
            except queue.Full:
                # Queue is full - implement drop-oldest strategy
                try:
                    # Remove oldest segment
                    old_segment = self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                    
                    # Add new segment
                    self.audio_queue.put(audio_segment, block=False)
                    
                    # Track drop statistics (not as failure)
                    with self.drop_stats_lock:
                        self.drops_last_minute.append(time.time())
                    
                    # Rate-limited warning
                    current_time = time.time()
                    if current_time - self._last_adaptive_drop_warning_time >= 1.0:
                        qsize = self.audio_queue.qsize()
                        maxsize = self.audio_queue.maxsize
                        logger.warning(
                            f"discard-oldest (audio) engaged, dropped 1 "
                            f"(queue: {qsize}/{maxsize}, "
                            f"warmup: {'yes' if self._warm_up_mode else 'no'})"
                        )
                        self._last_adaptive_drop_warning_time = current_time
                    
                    # Drop-oldest successful!
                    return True
                    
                except queue.Empty:
                    # Queue became empty between checks - try adding directly
                    try:
                        self.audio_queue.put(audio_segment, block=False)
                        return True
                    except queue.Full:
                        # Queue filled up again, retry the whole drop-oldest cycle
                        logger.debug(f"Queue filled again during drop-oldest, attempt {attempt + 1}/{max_attempts}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error in add_audio_segment attempt {attempt + 1}: {e}")
                if attempt == max_attempts - 1:
                    # Only fail on the last attempt
                    return False
                time.sleep(0.01)  # Brief pause before retry
                
        # This should never be reached, but just in case
        logger.error("Failed to add audio segment after all attempts")
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
        - DEBUG: Processing decisions, buffer management, performance metrics
        - TRACE: Detailed audio analysis, timing calculations
        """
        worker_name = threading.current_thread().name
        logger.info(f"Transcription worker started ({worker_name})")
        
        # Sliding window failure tracking with larger window for stability
        import collections
        failure_window = collections.deque(maxlen=50)  # Increased from 20 to 50 for more stable tracking
        last_failure_check = time.time()
        failure_check_interval = 5.0  # Check failure rate every 5 seconds
        consecutive_failures = 0  # Track consecutive failures for immediate response
        
        process = psutil.Process(os.getpid())
        last_perf_log = time.time()
        perf_log_interval = 5.0  # Log performance every 5 seconds
        
        while self.is_running or not self.audio_queue.empty():
            try:
                # Performance monitoring
                current_time = time.time()
                if current_time - last_perf_log >= perf_log_interval:
                    # Memory usage
                    mem_info = process.memory_info()
                    mem_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
                    
                    # Queue sizes
                    audio_qsize = self.audio_queue.qsize()
                    result_qsize = self.result_queue.qsize()
                    
                    # System load
                    cpu_percent = psutil.cpu_percent()
                    
                    # Calculate drop rate in last minute
                    drop_rate = 0.0
                    with self.drop_stats_lock:
                        # Count drops in last 60 seconds
                        now = time.time()
                        recent_drops = [t for t in self.drops_last_minute if now - t <= 60.0]
                        total_chunks = self.processed_chunks
                        if total_chunks > 0:
                            drop_rate = len(recent_drops) / total_chunks
                    
                    # Calculate drops per minute
                    drops_per_min = len(recent_drops) if recent_drops else 0
                    
                    # Enhanced telemetry with device and performance
                    logger.debug(
                        f"Performance - "
                        f"Device: {self.device.upper()} | "
                        f"Memory: {mem_mb:.1f}MB | "
                        f"CPU: {cpu_percent}% | "
                        f"Audio Q: {audio_qsize}/{self.audio_queue.maxsize} | "
                        f"Result Q: {result_qsize} | "
                        f"Chunks: {self.processed_chunks} | "
                        f"Proc avg: {self.average_processing_time:.2f}s | "
                        f"Drops/min: {drops_per_min} | "
                        f"Drop rate: {drop_rate:.1%}"
                    )
                    
                    # PRODUCTION ALERT: Non-CUDA mode detected
                    if self.device != "cuda" and self.processed_chunks > 0:
                        if not hasattr(self, '_last_non_cuda_warning') or current_time - self._last_non_cuda_warning >= 60.0:
                            logger.error(f"🚨 PRODUCTION ALERT: Running on {self.device.upper()}, not CUDA! Performance will be degraded.")
                            logger.error(f"   → Expected: ~0.1s per chunk | Actual: {self.average_processing_time:.2f}s per chunk")
                            self._last_non_cuda_warning = current_time
                    
                    # TELEMETRY: Alert on high drop rate with backlog stats
                    if drops_per_min > 80:  # More than 80 drops per minute (raised threshold)
                        if not hasattr(self, '_last_high_drop_warning') or current_time - self._last_high_drop_warning >= 10.0:
                            # Calculate drop percentage and show backlog context
                            drop_percentage = (drops_per_min / (self.processed_chunks + drops_per_min)) * 100 if self.processed_chunks > 0 else 0
                            drift = getattr(self, '_audio_drift', 0.0)  # Will add this tracking
                            logger.warning(f"🚨 HIGH DROP RATE {drop_percentage:.0f}% | audio_q {audio_qsize}/{self.audio_queue.maxsize} | drift {drift:.1f}s")
                            self._last_high_drop_warning = current_time
                    elif drops_per_min > 40:  # More than 40 drops per minute  
                        if not hasattr(self, '_last_elevated_drop_warning') or current_time - self._last_elevated_drop_warning >= 30.0:
                            logger.info(f"⚠️  Elevated drop rate: {drops_per_min} drops/min | audio_q {audio_qsize}/{self.audio_queue.maxsize}")
                            self._last_elevated_drop_warning = current_time
                    
                    # P1 Fix: Auto-spawn 4th worker when queue >11/15 for >10s
                    if (self.device == "cpu" and 
                        audio_qsize > 11 and 
                        len(self.worker_threads) < 4 and
                        hasattr(self, '_high_queue_start_time')):
                        
                        high_queue_duration = current_time - self._high_queue_start_time
                        if high_queue_duration > 10.0:  # Queue high for >10 seconds
                            logger.warning(f"🔧 Auto-spawning 4th worker (queue={audio_qsize}/15 for {high_queue_duration:.1f}s)")
                            worker = threading.Thread(
                                target=self._transcription_worker,
                                name=f"TranscriptionWorker-4",
                                daemon=True
                            )
                            worker.start()
                            self.worker_threads.append(worker)
                            delattr(self, '_high_queue_start_time')  # Reset timer
                    
                    elif audio_qsize > 11 and not hasattr(self, '_high_queue_start_time'):
                        # Start tracking high queue time
                        self._high_queue_start_time = current_time
                        
                    elif audio_qsize <= 8:
                        # Queue back to normal, reset timer
                        if hasattr(self, '_high_queue_start_time'):
                            delattr(self, '_high_queue_start_time')
                    
                    # Alert if drop rate is high
                    if drop_rate > 0.3 and self.processed_chunks > 10:  # > 30% and enough samples
                        logger.warning(f"High drop rate detected: {drop_rate:.1%} of chunks dropped in last minute")
                    last_perf_log = current_time
                
                # Get audio data from queue with timeout to allow checking is_running
                try:
                    audio_segment = self.audio_queue.get(timeout=0.5)
                    failure_window.append(0)  # Record successful queue get
                    
                    # Unpack the segment (removed CPU chunk merging - it caused 11-12s latency)
                    audio_data, timestamp = audio_segment
                    
                    # Skip empty segments
                    if audio_data is None or len(audio_data) == 0:
                        logger.warning("Skipping empty audio segment")
                        self.audio_queue.task_done()
                        continue
                        
                except queue.Empty:
                    failure_window.append(0)  # Queue empty is normal, not a failure
                    
                    # Check failure rate periodically - implement exact Fix #2
                    current_time = time.time()
                    if current_time - last_failure_check >= failure_check_interval and len(failure_window) >= 10:
                        failure_rate = sum(failure_window) / len(failure_window)
                        
                        # Critical fix: back-off instead of stopping when fail_rate > 0.6
                        if failure_rate > 0.6:
                            logger.warning("High drop rate – backing off")
                            time.sleep(self.chunk_size)  # let queue drain
                            failure_window.clear()
                            consecutive_failures = 0  # Reset consecutive failures too
                            last_failure_check = current_time
                            continue  # Back to queue processing
                        
                        # Only warn about actual transcription failure rate (not queue management)
                        actual_failures = sum(1 for x in failure_window if x == 1)
                        if actual_failures > 3 and self.processed_chunks > 0:  # 3+ actual failures
                            logger.warning(
                                f"Transcription failure rate: {actual_failures} actual failures in last {len(failure_window)} attempts. "
                                f"Queue size: {self.audio_queue.qsize()}/{self.audio_queue.maxsize}. "
                                f"Processed chunks: {self.processed_chunks}"
                            )
                            # Don't clear the window, let it naturally slide
                        elif self.processed_chunks == 0:
                            # During warm-up, just log at debug level occasionally
                            logger.debug(f"[{worker_name}] Warm-up phase: waiting for audio segments ({len(failure_window)} queue checks)")
                        last_failure_check = current_time
                        
                    # Reset consecutive failures if we successfully got an item
                    consecutive_failures = 0
                    
                    logger.trace("Audio queue empty, waiting for segments...")
                    time.sleep(0.1)
                    continue
                    
                except queue.Full:
                    # P0 Fix #1: Queue full is NOT a failure, it's normal queue management
                    failure_window.append(0)  # Record as success, not failure
                    consecutive_failures = 0  # Reset consecutive failure counter
                    logger.trace("Audio queue full, waiting for space...")
                    time.sleep(0.1)
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
                    # Add retry logic for transient errors
                    max_retries = 3
                    last_error = None
                    for attempt in range(max_retries):
                        try:
                            segments, info = self.model.transcribe(
                                audio_data,
                                language="en",
                                beam_size=5,
                                word_timestamps=True,
                                temperature=0.0  # Disable sampling for more consistent results
                            )
                            segments = list(segments)  # Convert generator to list to catch errors early
                            break  # Success, exit retry loop
                        except RuntimeError as e:
                            last_error = e
                            if "CUDA out of memory" in str(e) and attempt < max_retries - 1:
                                logger.warning(f"CUDA OOM on attempt {attempt + 1}, retrying...")
                                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                                continue
                            raise  # Re-raise if not a retryable error or out of retries
                    else:
                        # This block runs if the loop completes without breaking
                        if last_error:
                            raise last_error  # Re-raise the last error if all retries failed
                        
                    # Log successful transcription and language
                    processing_time = time.time() - start_process_time
                    logger.debug(
                        f"Transcription completed in {processing_time:.2f}s, "
                        f"found {len(segments)} segments. Language: {info.language} (Prob: {info.language_probability:.2f})"
                    )
                    
                    try:
                        # Process each transcribed segment
                        for segment in segments:
                            if not segment.text.strip():
                                logger.debug("Skipping empty transcription segment")
                                continue
                                
                            # Calculate timestamps relative to when the audio chunk was captured
                            if CLOCK.is_initialized():
                                # Use the original audio chunk timestamp + segment offset within chunk
                                rel_start = timestamp + segment.start  # Add segment offset within the 3s chunk
                                rel_end = timestamp + segment.end      # Add segment end offset within chunk
                            else:
                                # Fallback if clock not initialized
                                logger.warning("Singleton clock not initialized, using fallback timestamps")
                                rel_start = segment.start
                                rel_end = segment.end
                            
                            duration = rel_end - rel_start
                            
                            # Log transcription result at INFO level
                            confidence = getattr(segment, 'confidence', None)
                            confidence_str = f", confidence: {confidence:.2f}" if confidence is not None else ""
                            logger.info(f"Transcribed: '{segment.text.strip()}' "
                                      f"(rel_start={rel_start:.2f}s-{rel_end:.2f}s{confidence_str})")
                            
                            # Add to result queue for display
                            try:
                                result = {
                                    'text': segment.text.strip(),
                                    'timestamp': rel_start,  # Now using elapsed-time relative timestamps
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
                        self.average_processing_time = (
                            (self.average_processing_time * (self.processed_chunks - 1) + processing_time) 
                            / max(1, self.processed_chunks)  # Avoid division by zero
                        )
                        
                        logger.debug(f"Processing stats - "
                                   f"chunks: {self.processed_chunks}, "
                                   f"avg time: {self.average_processing_time:.2f}s")
                        
                    except Exception as e:
                        consecutive_failures += 1
                        failure_window.append(1)  # Record processing error as failure
                        logger.error(f"Error in transcription (attempt {consecutive_failures}): {e}", exc_info=True)
                        
                        # Check if we should continue or pause briefly
                        current_time = time.time()
                        if consecutive_failures >= 3:  # 3 consecutive failures
                            logger.warning(f"{consecutive_failures} consecutive failures, pausing briefly")
                            time.sleep(min(1.0 * consecutive_failures, 5.0))  # Cap at 5 seconds
                            
                        # Check overall transcription failure rate  
                        if len(failure_window) >= 10:  # Only check if we have enough samples
                            actual_failures = sum(1 for x in failure_window if x == 1)
                            if actual_failures > 5:  # More than 5 actual transcription failures
                                logger.warning(f"Critical transcription failure count ({actual_failures} failures), resetting model")
                                try:
                                    self._init_model()  # Try to reinitialize the model
                                    time.sleep(1.0)
                                    failure_window.clear()
                                    consecutive_failures = 0
                                except Exception as reset_error:
                                    logger.error(f"Failed to reset model: {reset_error}")
                                    time.sleep(2.0)  # Longer pause if reset fails
                                    
                except Exception as e:
                    logger.error(f"Error processing audio segment: {e}", exc_info=True)
                    consecutive_failures += 1
                    failure_window.append(1)
                        
                finally:
                    self.audio_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                time.sleep(0.5)  # Prevent tight error loop
                
        logger.info(f"Transcription worker stopped ({worker_name})")

    def process_audio(self, video_source, chunk_size: float = None, overlap: float = None):
        """Process audio from video source and send to transcriber.
        
        This method processes audio in chunks during warm-up and continues
        processing during playback for real-time transcription.
        
        Args:
            video_source: The video source to get audio chunks from
            chunk_size: Size of audio chunks in seconds (uses device-specific default if None)
            overlap: Overlap between chunks in seconds (uses device-specific default if None)
            
        Notes:
            - Uses a sliding window approach for better transcription continuity
            - Automatically adjusts queue behavior during warm-up vs. normal operation
            - Implements backpressure to prevent queue overflow
        """
        # Use device-specific parameters if not explicitly provided
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
            
        # Dynamic chunk sizing for CPU mode based on processing performance
        original_chunk_size = chunk_size
        last_chunk_resize_time = time.time()
        chunk_resize_interval = 10.0  # Check every 10 seconds
        min_chunk_size = 0.6  # Never go below 0.6s chunks
            
        logger.info(f"Starting audio processing (chunk_size={chunk_size}s, overlap={overlap}s, device={self.device})")
        
        # Validate parameters
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            logger.error(f"Invalid chunk parameters: size={chunk_size}s, overlap={overlap}s")
            return
            
        # Wait for the main warm-up period to complete before active processing
        if self.warm_up_complete_event:
            logger.info("Audio processing thread waiting for warm-up barrier...")
            
            # Check if video_source has the new barrier system
            if hasattr(video_source, 'warm_up_barrier'):
                try:
                    video_source.warm_up_barrier.wait()
                    logger.info("Audio processing thread passed warm-up barrier.")
                except threading.BrokenBarrierError:
                    logger.error("Warm-up barrier was broken, falling back to event-based sync")
                    self.warm_up_complete_event.wait()
                    logger.info("Application warm-up complete, audio processing thread proceeding.")
            else:
                # Fall back to old event-based system
                logger.info("Audio processing thread waiting for application warm-up to complete...")
                self.warm_up_complete_event.wait()
                logger.info("Application warm-up complete, audio processing thread proceeding.")
                
            self._warm_up_mode = False # Explicitly transition out of warm-up mode for the engine
            
            # Clear any audio segments that might have been queued during app warm-up
            # or rapidly queued from VideoSource's own backlog immediately after warm_up_complete.set()
            cleared_count = 0
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                    cleared_count += 1
                except queue.Empty:
                    break
                    
            # Clear any pending results as well
            result_cleared = 0
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.task_done()
                    result_cleared += 1
                except queue.Empty:
                    break
                    
            if cleared_count > 0 or result_cleared > 0:
                logger.info(
                    f"Cleared {cleared_count} audio segments and {result_cleared} results "
                    "from queue after warm-up."
                )
                
            # Reset processing state for normal operation
            self.processed_chunks = 0
            self.average_processing_time = 0.0
            self._warm_up_mode = False  # Ensure we're out of warm-up mode
            
            # Reset the thread pool for normal operation
            if hasattr(self, 'thread_pool') and self.thread_pool is not None:
                self.thread_pool.shutdown(wait=False)
            # Initialize thread pool
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.n_workers,
                thread_name_prefix="TranscriptionWorker"
            ) if self.n_workers > 1 else None
                
            logger.info("Warm-up complete, transcription engine ready for normal operation")
            
        else:
            logger.warning("No warm_up_complete_event available to TranscriptionEngine. Audio processing may behave unexpectedly regarding warm-up state.")
            self._warm_up_mode = False # Default to not in warm-up if event is missing
            
        last_process_time = 0
        video_start_time = getattr(video_source, 'start_time', 0.0)
        chunk_counter = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        # Track timing mode to handle transition correctly
        was_in_warmup = True
        warm_up_chunks_processed = 0
        WARM_UP_CHUNKS_NEEDED = 3  # Number of chunks to process during warm-up
        
        # Debug timing tracking
        last_debug_log = 0
        debug_log_interval = 2.0  # Log every 2 seconds
        
        try:
            while (getattr(video_source, 'is_running', False) and 
                   not getattr(video_source, '_shutdown_event', None).is_set() and 
                   self.is_running):  # Also check if transcription is still running
                try:
                    # Get current elapsed time (no media offset)
                    warm_up_complete = getattr(video_source, 'warm_up_complete', None)
                    warm_up_is_set = warm_up_complete.is_set() if warm_up_complete else False
                    
                    if warm_up_is_set:
                        # Use elapsed time since playback start (consistent with other timing)
                        current_time = time.time() - video_source.playback_start_time
                        in_warmup = False
                        
                        # Reset timing on transition from warm-up to normal operation
                        if was_in_warmup:
                            logger.debug("Transitioned from warm-up to normal operation, resetting timing")
                            last_process_time = 0  # Reset to start fresh timing
                            was_in_warmup = False  # This now persists across loop iterations
                    else:
                        with getattr(video_source, 'audio_position_lock', threading.Lock()):
                            # During warm-up, use audio position without seek offset
                            current_time = video_source.audio_position
                        in_warmup = True
                    
                    # Process audio in chunks with overlap
                    time_since_last = current_time - last_process_time
                    should_process = time_since_last >= (chunk_size - overlap)
                    
                    # Dynamic chunk size adjustment for CPU mode - Fix #3
                    current_debug_time = time.time()
                    if (current_debug_time - last_chunk_resize_time >= chunk_resize_interval and 
                        self.device == "cpu" and 
                        self.processed_chunks > 5):  # Need some samples
                        
                        # Check if processing is keeping up
                        target_processing_time = 0.8 * chunk_size * self.n_workers  # 80% of theoretical capacity
                        if self.average_processing_time > target_processing_time and chunk_size > min_chunk_size:
                            # Processing is too slow, reduce chunk size
                            new_chunk_size = max(min_chunk_size, chunk_size / 1.5)
                            new_overlap = max(0.1, overlap / 1.5)
                            
                            if new_chunk_size != chunk_size:
                                logger.warning(f"🔧 Dynamic resize: chunk_size {chunk_size:.1f}s → {new_chunk_size:.1f}s "
                                             f"(avg_proc={self.average_processing_time:.1f}s > target={target_processing_time:.1f}s)")
                                chunk_size = new_chunk_size
                                overlap = new_overlap
                                
                        elif (self.average_processing_time < target_processing_time * 0.6 and 
                              chunk_size < original_chunk_size and
                              self.audio_queue.qsize() < self.audio_queue.maxsize * 0.3):  # Queue is light
                            # Processing is fast and queue is light, can increase chunk size
                            new_chunk_size = min(original_chunk_size, chunk_size * 1.2)
                            new_overlap = min(overlap * 1.2, new_chunk_size * 0.2)
                            
                            if new_chunk_size != chunk_size:
                                logger.info(f"🔧 Dynamic resize: chunk_size {chunk_size:.1f}s → {new_chunk_size:.1f}s "
                                          f"(avg_proc={self.average_processing_time:.1f}s, queue={self.audio_queue.qsize()}/{self.audio_queue.maxsize})")
                                chunk_size = new_chunk_size
                                overlap = new_overlap
                                
                        last_chunk_resize_time = current_debug_time
                    
                    # Debug logging for timing calculations
                    if current_debug_time - last_debug_log >= debug_log_interval:
                        logger.debug(f"[AUDIO DEBUG] warmup={in_warmup}, current_time={current_time:.2f}s, "
                                   f"last_process_time={last_process_time:.2f}s, time_since_last={time_since_last:.2f}s, "
                                   f"should_process={should_process} (need >={chunk_size - overlap:.1f}s), "
                                   f"chunks_processed={chunk_counter}")
                        if hasattr(video_source, 'playback_start_time'):
                            wall_time = time.time()
                            logger.debug(f"[AUDIO DEBUG] wall_time={wall_time:.2f}, playback_start_time={video_source.playback_start_time:.2f}, "
                                       f"calculated_current_time={wall_time - video_source.playback_start_time:.2f}")
                        last_debug_log = current_debug_time
                    
                    if should_process:
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
                                    logger.info(f"[NORMAL] {log_msg}")  # Make normal operation chunks visible
                                
                                # Try to add to transcription engine - P0 Fix #2: use 60% threshold for drop-oldest
                                try:
                                    # P0 Fix #2: Start drop-oldest at 60% (12/20) instead of 100% (20/20)
                                    queue_ratio = self.audio_queue.qsize() / self.audio_queue.maxsize
                                    if queue_ratio < 0.6:  # Less than 60% full (< 12/20)
                                        self.audio_queue.put(audio_chunk, block=False)
                                        # Success - reset failure counter and continue
                                        last_process_time = current_time
                                        consecutive_failures = 0
                                        logger.debug(f"[AUDIO DEBUG] Added chunk directly, updated last_process_time to {last_process_time:.2f}s")
                                    else:
                                        # Queue is ≥60% full - implement drop-oldest manually
                                        try:
                                            old_chunk = self.audio_queue.get_nowait()
                                            self.audio_queue.task_done()
                                            self.audio_queue.put(audio_chunk, block=False)
                                            
                                            # Track drop statistics (NOT as failures)
                                            with self.drop_stats_lock:
                                                self.drops_last_minute.append(time.time())
                                            
                                            # Rate-limited warning
                                            current_time_check = time.time()
                                            if current_time_check - self._last_adaptive_drop_warning_time >= 1.0:
                                                qsize = self.audio_queue.qsize()
                                                maxsize = self.audio_queue.maxsize
                                                logger.warning(
                                                    f"discard-oldest (audio) engaged, dropped 1 "
                                                    f"(queue: {qsize}/{maxsize}, chunk: {chunk_id})"
                                                )
                                                self._last_adaptive_drop_warning_time = current_time_check
                                            
                                            # P0 Fix: Drop-oldest is NOT a failure - reset timing and continue
                                            last_process_time = current_time
                                            consecutive_failures = 0  # Explicit: drops are NOT failures
                                            logger.debug(f"[AUDIO DEBUG] Drop-oldest successful, updated last_process_time to {last_process_time:.2f}s")
                                            
                                        except queue.Empty:
                                            # Queue became empty, try direct add
                                            self.audio_queue.put(audio_chunk, block=False)
                                            last_process_time = current_time
                                            consecutive_failures = 0
                                            logger.debug(f"[AUDIO DEBUG] Queue became empty, added directly")
                                    
                                    # Debug log for audio processing progress
                                    if not in_warmup and chunk_counter % 5 == 0:  # Every 5th chunk during normal operation
                                        logger.debug(f"Audio processing progress: {chunk_counter} chunks processed, "
                                                   f"current_time={current_time:.1f}s")
                                        
                                except queue.Full:
                                    # This should be rare after drop-oldest, but handle gracefully
                                    logger.debug(f"Queue still full after drop-oldest for chunk {chunk_id}, skipping")
                                    # P0 Fix: Do NOT increment consecutive_failures - this is queue management, not an error
                                    consecutive_failures = 0  # Explicit reset
                                    
                                except Exception as e:
                                    # This is an ACTUAL error (not queue management)
                                    consecutive_failures += 1
                                    log_level = logger.debug if not self.is_running else logger.warning
                                    log_level(f"Real error adding chunk {chunk_id}: {e} (consecutive errors: {consecutive_failures})")
                                    
                                    # Adaptive back-off instead of stopping - Fix #2
                                    if consecutive_failures > max_consecutive_failures:
                                        logger.warning(f"Back-off due to {consecutive_failures} consecutive errors")
                                        time.sleep(chunk_size)  # Sleep for chunk duration to let workers catch up
                                        consecutive_failures = 0  # Clear failures after back-off
                                        continue
                            else:
                                logger.warning(f"Received empty audio chunk at {current_time:.2f}s")
                                
                        except Exception as e:
                            consecutive_failures += 1
                            logger.error(f"Error getting audio chunk: {e}", exc_info=consecutive_failures > 1)
                            
                            # Back-off strategy instead of accumulating failures
                            if consecutive_failures > max_consecutive_failures:
                                logger.warning(f"Consecutive audio errors ({consecutive_failures}), backing off")
                                time.sleep(chunk_size * 2)  # Sleep for two chunk durations on audio errors
                                consecutive_failures = 0  # Clear failures after back-off
                            else:
                                time.sleep(0.5)  # Shorter back-off for initial errors
                            continue
                    
                    # Adaptive sleep based on chunk processing timing
                    if should_process:
                        # Just processed a chunk, sleep less to be responsive
                        sleep_time = 0.1  # 100ms
                    else:
                        # Calculate how long until next chunk should be processed
                        time_until_next = (chunk_size - overlap) - time_since_last
                        if time_until_next > 0:
                            # Sleep for most of the remaining time, but check periodically
                            sleep_time = min(0.5, max(0.1, time_until_next * 0.8))
                        else:
                            sleep_time = 0.1  # Fallback
                    
                    # If queue is getting full, slow down to let the worker catch up
                    queue_ratio = self.audio_queue.qsize() / max(1, self.audio_queue.maxsize)
                    if queue_ratio > 0.6:  # If queue is more than 60% full (9/15 for CPU)
                        sleep_time = min(1.0, sleep_time * (1 + queue_ratio * 2))  # Up to 1.0s
                    
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
