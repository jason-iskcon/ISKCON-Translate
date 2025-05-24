"""
Configuration constants for the transcription engine.

This module centralizes all configurable parameters for the TranscriptionEngine
to make it easier to maintain and modify settings.
"""

# Audio processing constants
SAMPLING_RATE = 16000  # Whisper's native sampling rate
MAX_AUDIO_QUEUE_SIZE = 15  # Default max queue size (will be overridden by device-specific params)
PERF_LOG_INTERVAL = 5.0  # Performance logging interval in seconds
DEFAULT_DURATION = 3.0  # Default audio chunk duration
WARM_UP_ENABLED = True  # Enable warm-up phase

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

# Worker and threading constants
MAX_CONSECUTIVE_FAILURES = 5  # Maximum consecutive failures before back-off
WORKER_TIMEOUT_MULTIPLIER = 0.1  # Timeout per worker calculation multiplier
THREAD_POOL_PREFIX = "TransWorker"  # Prefix for thread pool workers
WORKER_THREAD_PREFIX = "TranscriptionWorker"  # Prefix for worker threads

# Performance monitoring
FAILURE_CHECK_INTERVAL = 2.0  # How often to check failure rates
FAILURE_WINDOW_SIZE = 20  # Size of failure tracking window
CHUNK_RESIZE_INTERVAL = 10.0  # Interval for dynamic chunk resizing (CPU mode)
MIN_CHUNK_SIZE = 0.6  # Minimum chunk size for dynamic resizing

# Queue management
DROP_OLDEST_THRESHOLD = 0.6  # Queue usage threshold for drop-oldest strategy
QUEUE_LIGHT_THRESHOLD = 0.3  # Queue threshold for considering it "light"

# Logging rate limits
LOG_RATE_LIMIT_INTERVAL = 1.0  # Rate limit for repeated log messages
DEBUG_LOG_INTERVAL = 2.0  # Debug logging interval
HIGH_DROP_WARNING_INTERVAL = 10.0  # Interval for high drop rate warnings
ELEVATED_DROP_WARNING_INTERVAL = 30.0  # Interval for elevated drop rate warnings

# Retry and backoff settings
MAX_RETRIES = 3  # Maximum retries for transient errors
RETRY_BACKOFF_BASE = 0.5  # Base backoff time for retries
AUTO_SPAWN_QUEUE_THRESHOLD = 11  # Queue size threshold for auto-spawning workers
AUTO_SPAWN_DURATION_THRESHOLD = 10.0  # Duration threshold for auto-spawning workers

# Sleep timing
ADAPTIVE_SLEEP_MIN = 0.1  # Minimum adaptive sleep time
ADAPTIVE_SLEEP_MAX = 1.0  # Maximum adaptive sleep time
QUEUE_FULL_SLEEP_MULTIPLIER = 2.0  # Multiplier for sleep when queue is full 