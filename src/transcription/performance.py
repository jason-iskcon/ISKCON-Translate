"""
Performance monitoring and logging for transcription engine.

This module handles system performance tracking, memory usage monitoring,
and diagnostic logging for the transcription engine.
"""

import time
import psutil
import threading
import json
from typing import Optional, Dict, Any
from collections import deque

# Import with try-except to handle both direct execution and module import
try:
    from logging_utils import get_logger
except ImportError:
    from ..logging_utils import get_logger

from .config import (
    PERF_LOG_INTERVAL, HIGH_DROP_WARNING_INTERVAL, 
    ELEVATED_DROP_WARNING_INTERVAL, AUTO_SPAWN_QUEUE_THRESHOLD,
    AUTO_SPAWN_DURATION_THRESHOLD
)
from .utils import calculate_drop_rate, format_queue_info

logger = get_logger('transcription.performance')


class PerformanceMonitor:
    """Performance monitoring class for transcription engine."""
    
    def __init__(self):
        self.last_perf_log = 0
        self.last_json_heartbeat = 0
        self.rate_limit_tracker = {}
        self.process = psutil.Process()
        self.inference_failures = deque(maxlen=60)  # Track failures for last minute
        self.json_heartbeat_interval = 60.0  # JSON heartbeat every minute
        
    def log_system_performance(self, audio_queue, result_queue, processed_chunks: int,
                             avg_time: float, device: str, drops_last_minute: deque,
                             drop_stats_lock: threading.Lock, worker_threads: list) -> None:
        """Log comprehensive system performance metrics.
        
        Args:
            audio_queue: Audio processing queue
            result_queue: Results queue
            processed_chunks: Number of processed chunks
            avg_time: Average processing time
            device: Processing device (cpu/cuda)
            drops_last_minute: Deque of drop timestamps
            drop_stats_lock: Lock for drop statistics
            worker_threads: List of worker threads
        """
        current_time = time.time()
        
        if current_time - self.last_perf_log >= PERF_LOG_INTERVAL:
            # Memory usage
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
            
            # Queue sizes
            audio_qsize = audio_queue.qsize()
            result_qsize = result_queue.qsize()
            
            # System load
            cpu_percent = psutil.cpu_percent()
            
            # Calculate drop rate in last minute
            drop_rate, drops_per_min = calculate_drop_rate(list(drops_last_minute), processed_chunks)
            
            # Enhanced telemetry with device and performance
            queue_info = format_queue_info(audio_qsize, audio_queue.maxsize, result_qsize)
            logger.debug(
                f"Performance - "
                f"Device: {device.upper()} | "
                f"Memory: {mem_mb:.1f}MB | "
                f"CPU: {cpu_percent}% | "
                f"{queue_info} | "
                f"Chunks: {processed_chunks} | "
                f"Proc avg: {avg_time:.2f}s | "
                f"Drops/min: {drops_per_min} | "
                f"Drop rate: {drop_rate:.1%}"
            )
            
            # PRODUCTION ALERT: Non-CUDA mode detected
            self._check_device_performance(device, processed_chunks, avg_time, current_time)
            
            # TELEMETRY: Alert on high drop rate with backlog stats
            self._check_drop_rates(drops_per_min, processed_chunks, audio_qsize, 
                                 audio_queue.maxsize, current_time)
            
            # Auto-spawn worker logic for CPU
            self._check_auto_spawn_workers(device, audio_qsize, worker_threads, current_time)
            
            # JSON heartbeat logging for structured monitoring
            if current_time - self.last_json_heartbeat >= self.json_heartbeat_interval:
                inference_failures_per_min = self.get_inference_failures_per_minute()
                heartbeat_data = {
                    "timestamp": current_time,
                    "device": device,
                    "proc_avg": round(avg_time, 3),
                    "inference_failures_per_min": inference_failures_per_min,
                    "audio_queue": f"{audio_qsize}/{audio_queue.maxsize}",
                    "result_queue": result_qsize,
                    "processed_chunks": processed_chunks,
                    "drops_per_min": drops_per_min,
                    "memory_mb": round(mem_mb, 1),
                    "cpu_percent": cpu_percent,
                    "worker_count": len(worker_threads)
                }
                logger.info(f"HEARTBEAT: {json.dumps(heartbeat_data)}")
                self.last_json_heartbeat = current_time
            
            self.last_perf_log = current_time
    
    def _check_device_performance(self, device: str, processed_chunks: int, 
                                avg_time: float, current_time: float) -> None:
        """Check and warn about device performance issues."""
        if device != "cuda" and processed_chunks > 0:
            warning_key = 'non_cuda_warning'
            if (warning_key not in self.rate_limit_tracker or 
                current_time - self.rate_limit_tracker[warning_key] >= 60.0):
                
                logger.error(f"🚨 PRODUCTION ALERT: Running on {device.upper()}, not CUDA! Performance will be degraded.")
                logger.error(f"   → Expected: ~0.1s per chunk | Actual: {avg_time:.2f}s per chunk")
                self.rate_limit_tracker[warning_key] = current_time
    
    def _check_drop_rates(self, drops_per_min: int, processed_chunks: int,
                         audio_qsize: int, audio_maxsize: int, current_time: float) -> None:
        """Check and warn about high drop rates."""
        if drops_per_min > 80:  # More than 80 drops per minute (raised threshold)
            warning_key = 'high_drop_warning'
            if (warning_key not in self.rate_limit_tracker or 
                current_time - self.rate_limit_tracker[warning_key] >= HIGH_DROP_WARNING_INTERVAL):
                
                # Calculate drop percentage and show backlog context
                drop_percentage = (drops_per_min / (processed_chunks + drops_per_min)) * 100 if processed_chunks > 0 else 0
                drift = getattr(self, '_audio_drift', 0.0)  # Will add this tracking
                logger.warning(f"🚨 HIGH DROP RATE {drop_percentage:.0f}% | audio_q {audio_qsize}/{audio_maxsize} | drift {drift:.1f}s")
                self.rate_limit_tracker[warning_key] = current_time
                
        elif drops_per_min > 40:  # More than 40 drops per minute  
            warning_key = 'elevated_drop_warning'
            if (warning_key not in self.rate_limit_tracker or 
                current_time - self.rate_limit_tracker[warning_key] >= ELEVATED_DROP_WARNING_INTERVAL):
                
                logger.info(f"⚠️  Elevated drop rate: {drops_per_min} drops/min | audio_q {audio_qsize}/{audio_maxsize}")
                self.rate_limit_tracker[warning_key] = current_time
    
    def _check_auto_spawn_workers(self, device: str, audio_qsize: int, 
                                worker_threads: list, current_time: float) -> None:
        """Check if additional workers should be spawned automatically."""
        if (device == "cpu" and 
            audio_qsize > AUTO_SPAWN_QUEUE_THRESHOLD and 
            len(worker_threads) < 4 and
            hasattr(self, '_high_queue_start_time')):
            
            high_queue_duration = current_time - self._high_queue_start_time
            if high_queue_duration > AUTO_SPAWN_DURATION_THRESHOLD:  # Queue high for >10 seconds
                logger.warning(f"🔧 Auto-spawning 4th worker (queue={audio_qsize}/15 for {high_queue_duration:.1f}s)")
                # Note: The actual worker spawning should be done by the caller
                # This just logs the condition
                delattr(self, '_high_queue_start_time')  # Reset timer
        
        elif audio_qsize > AUTO_SPAWN_QUEUE_THRESHOLD and not hasattr(self, '_high_queue_start_time'):
            # Start tracking high queue time
            self._high_queue_start_time = current_time
            
        elif audio_qsize <= 8:
            # Queue back to normal, reset timer
            if hasattr(self, '_high_queue_start_time'):
                delattr(self, '_high_queue_start_time')

    def record_inference_failure(self, failure_type: str = "general") -> None:
        """Record an inference failure for tracking.
        
        Args:
            failure_type: Type of failure (e.g., 'cuda_runtime', 'general', 'oom')
        """
        current_time = time.time()
        self.inference_failures.append({
            'timestamp': current_time,
            'type': failure_type
        })
        
    def get_inference_failures_per_minute(self) -> int:
        """Calculate inference failures in the last minute.
        
        Returns:
            int: Number of inference failures in the last minute
        """
        current_time = time.time()
        one_minute_ago = current_time - 60.0
        
        # Count failures in the last minute
        recent_failures = [f for f in self.inference_failures if f['timestamp'] >= one_minute_ago]
        return len(recent_failures)


def log_worker_performance(worker_name: str, processing_time: float, 
                         segments_count: int, language: str, 
                         language_prob: float) -> None:
    """Log individual worker performance metrics.
    
    Args:
        worker_name: Name of the worker thread
        processing_time: Time taken to process
        segments_count: Number of segments found
        language: Detected language
        language_prob: Language detection probability
    """
    logger.debug(
        f"[{worker_name}] Transcription completed in {processing_time:.2f}s, "
        f"found {segments_count} segments. Language: {language} (Prob: {language_prob:.2f})"
    )


def log_failure_analysis(failure_window: list, processed_chunks: int, 
                       worker_name: str, current_time: float,
                       last_failure_check: float, check_interval: float) -> float:
    """Log failure rate analysis.
    
    Args:
        failure_window: Window of recent failures
        processed_chunks: Total processed chunks
        worker_name: Name of the worker
        current_time: Current timestamp
        last_failure_check: Last time failures were checked
        check_interval: Check interval
        
    Returns:
        float: Updated last failure check time
    """
    if current_time - last_failure_check >= check_interval and len(failure_window) >= 10:
        failure_rate = sum(failure_window) / len(failure_window)
        
        # Only warn about actual transcription failure rate (not queue management)
        actual_failures = sum(1 for x in failure_window if x == 1)
        if actual_failures > 3 and processed_chunks > 0:  # 3+ actual failures
            logger.warning(
                f"Transcription failure rate: {actual_failures} actual failures in last {len(failure_window)} attempts. "
                f"Processed chunks: {processed_chunks}"
            )
        elif processed_chunks == 0:
            # During warm-up, just log at debug level occasionally
            logger.debug(f"[{worker_name}] Warm-up phase: waiting for audio segments ({len(failure_window)} queue checks)")
        
        return current_time
    
    return last_failure_check


def get_system_metrics() -> Dict[str, Any]:
    """Get current system performance metrics.
    
    Returns:
        Dict[str, Any]: System metrics including CPU, memory, etc.
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_mb': mem_info.rss / (1024 * 1024),
        'memory_percent': process.memory_percent(),
        'num_threads': process.num_threads(),
        'timestamp': time.time()
    } 