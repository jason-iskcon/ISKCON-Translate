"""
Configuration constants for the transcription engine.

This module centralizes all configurable parameters for the TranscriptionEngine
to make it easier to maintain and modify settings.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

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
    'chunk_size': 5.0,      # Increased from 3.0s to 5.0s for better context
    'overlap': 1.5,         # Increased from 1.0s to 1.5s for better continuity
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


@dataclass
class WhisperConfig:
    """Configuration for Whisper model and inference."""
    
    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    language: Optional[str] = None
    task: str = "transcribe"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    prefix: Optional[str] = None
    suppress_tokens: Optional[list] = None
    suppress_numerals: bool = False
    word_timestamps: bool = True
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for Whisper model."""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'language': self.language,
            'task': self.task,
            'beam_size': self.beam_size,
            'best_of': self.best_of,
            'patience': self.patience,
            'length_penalty': self.length_penalty,
            'temperature': self.temperature,
            'compression_ratio_threshold': self.compression_ratio_threshold,
            'log_prob_threshold': self.log_prob_threshold,
            'no_speech_threshold': self.no_speech_threshold,
            'condition_on_previous_text': self.condition_on_previous_text,
            'initial_prompt': self.initial_prompt,
            'prefix': self.prefix,
            'suppress_tokens': self.suppress_tokens,
            'suppress_numerals': self.suppress_numerals,
            'word_timestamps': self.word_timestamps,
            'repetition_penalty': self.repetition_penalty,
            'no_repeat_ngram_size': self.no_repeat_ngram_size
        } 