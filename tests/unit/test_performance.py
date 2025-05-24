import pytest
import time
import queue
import threading
import psutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from collections import deque

# Add src directory to path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.transcription.performance import (
    PerformanceMonitor, log_worker_performance, 
    log_failure_analysis, get_system_metrics
)


@pytest.fixture
def performance_monitor():
    """Create a fresh PerformanceMonitor instance."""
    return PerformanceMonitor()


@pytest.fixture
def mock_queues():
    """Create mock queues for testing."""
    audio_queue = MagicMock()
    audio_queue.qsize.return_value = 3
    audio_queue.maxsize = 15
    
    result_queue = MagicMock()
    result_queue.qsize.return_value = 1
    
    return audio_queue, result_queue


@pytest.fixture
def drop_stats():
    """Create drop statistics objects."""
    drops_last_minute = deque([time.time() - 30, time.time() - 20, time.time() - 10])
    drop_stats_lock = threading.Lock()
    return drops_last_minute, drop_stats_lock


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor class."""
    
    @patch('src.transcription.performance.logger')
    @patch('src.transcription.performance.psutil')
    def test_log_system_performance_logs_basic_metrics(self, mock_psutil, mock_logger, 
                                                      performance_monitor, mock_queues, drop_stats):
        """Test that system performance logging includes all basic metrics."""
        audio_queue, result_queue = mock_queues
        drops_last_minute, drop_stats_lock = drop_stats
        
        # Mock psutil process
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 134217728  # 128MB in bytes
        performance_monitor.process = mock_process
        
        # Mock psutil.cpu_percent
        mock_psutil.cpu_percent.return_value = 45.5
        
        worker_threads = [MagicMock(), MagicMock(), MagicMock()]
        
        performance_monitor.log_system_performance(
            audio_queue=audio_queue,
            result_queue=result_queue,
            processed_chunks=100,
            avg_time=0.15,
            device="cuda",
            drops_last_minute=drops_last_minute,
            drop_stats_lock=drop_stats_lock,
            worker_threads=worker_threads
        )
        
        # Verify debug log was called
        mock_logger.debug.assert_called()
        log_message = mock_logger.debug.call_args[0][0]
        
        # Check that all expected metrics are in the log
        assert "Device: CUDA" in log_message
        assert "Memory: 128.0MB" in log_message
        assert "CPU: 45.5%" in log_message
        assert "Chunks: 100" in log_message
        assert "Proc avg: 0.15s" in log_message
        
    @patch('src.transcription.performance.logger')
    def test_check_device_performance_warns_non_cuda(self, mock_logger, performance_monitor, 
                                                    mock_queues, drop_stats):
        """Test that non-CUDA device usage triggers warnings."""
        audio_queue, result_queue = mock_queues
        drops_last_minute, drop_stats_lock = drop_stats
        
        # Reset last_perf_log to ensure logging happens
        performance_monitor.last_perf_log = 0
        
        performance_monitor.log_system_performance(
            audio_queue=audio_queue,
            result_queue=result_queue,
            processed_chunks=50,
            avg_time=2.5,  # Slow processing time
            device="cpu",
            drops_last_minute=drops_last_minute,
            drop_stats_lock=drop_stats_lock,
            worker_threads=[]
        )
        
        # Should log error about non-CUDA usage
        error_calls = [call for call in mock_logger.error.call_args_list 
                      if "PRODUCTION ALERT" in str(call)]
        assert len(error_calls) >= 1
        
        error_message = str(error_calls[0])
        assert "Running on CPU, not CUDA" in error_message
        assert "Performance will be degraded" in error_message
        
    @patch('src.transcription.performance.logger')
    def test_check_drop_rates_warns_high_drops(self, mock_logger, performance_monitor, 
                                              mock_queues, drop_stats):
        """Test that high drop rates trigger warnings."""
        audio_queue, result_queue = mock_queues
        drops_last_minute, drop_stats_lock = drop_stats
        
        # Create a scenario with high drop rate
        performance_monitor.last_perf_log = 0
        
        # Mock high drops per minute calculation
        with patch('src.transcription.performance.calculate_drop_rate') as mock_calc_drop:
            mock_calc_drop.return_value = (0.85, 120)  # 85% drop rate, 120 drops/min
            
            performance_monitor.log_system_performance(
                audio_queue=audio_queue,
                result_queue=result_queue,
                processed_chunks=20,
                avg_time=0.1,
                device="cuda",
                drops_last_minute=drops_last_minute,
                drop_stats_lock=drop_stats_lock,
                worker_threads=[]
            )
            
        # Should log warning about high drop rate
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if "HIGH DROP RATE" in str(call)]
        assert len(warning_calls) >= 1
        
    @patch('src.transcription.performance.logger')
    def test_check_auto_spawn_workers_logic(self, mock_logger, performance_monitor, 
                                          mock_queues, drop_stats):
        """Test auto-spawn worker detection logic."""
        audio_queue, result_queue = mock_queues
        audio_queue.qsize.return_value = 14  # High queue size
        drops_last_minute, drop_stats_lock = drop_stats
        
        performance_monitor.last_perf_log = 0
        worker_threads = [MagicMock(), MagicMock()]  # Only 2 workers
        
        # Set up high queue start time
        performance_monitor._high_queue_start_time = time.time() - 15  # 15 seconds ago
        
        performance_monitor.log_system_performance(
            audio_queue=audio_queue,
            result_queue=result_queue,
            processed_chunks=100,
            avg_time=0.2,
            device="cpu",
            drops_last_minute=drops_last_minute,
            drop_stats_lock=drop_stats_lock,
            worker_threads=worker_threads
        )
        
        # Should log auto-spawn recommendation
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if "Auto-spawning" in str(call)]
        assert len(warning_calls) >= 1
        
    def test_rate_limiting_prevents_spam(self, performance_monitor, mock_queues, drop_stats):
        """Test that rate limiting prevents log spam."""
        audio_queue, result_queue = mock_queues
        drops_last_minute, drop_stats_lock = drop_stats
        
        with patch('src.transcription.performance.logger') as mock_logger:
            # Force first log
            performance_monitor.last_perf_log = 0
            
            # Log multiple times quickly
            for i in range(3):
                performance_monitor.log_system_performance(
                    audio_queue=audio_queue,
                    result_queue=result_queue,
                    processed_chunks=50,
                    avg_time=2.0,
                    device="cpu",
                    drops_last_minute=drops_last_minute,
                    drop_stats_lock=drop_stats_lock,
                    worker_threads=[]
                )
                
        # Should only log once due to rate limiting
        error_calls = [call for call in mock_logger.error.call_args_list 
                      if "PRODUCTION ALERT" in str(call)]
        assert len(error_calls) == 1


class TestLogWorkerPerformance:
    """Test suite for log_worker_performance function."""
    
    @patch('src.transcription.performance.logger')
    def test_log_worker_performance_logs_system_metrics(self, mock_logger):
        """Test that worker performance logging includes all metrics."""
        log_worker_performance(
            worker_name="Worker-1",
            processing_time=1.25,
            segments_count=3,
            language="english",
            language_prob=0.95
        )
        
        mock_logger.debug.assert_called_once()
        log_message = mock_logger.debug.call_args[0][0]
        
        # Verify all expected information is logged
        assert "Worker-1" in log_message
        assert "1.25s" in log_message
        assert "3 segments" in log_message
        assert "Language: english" in log_message
        assert "Prob: 0.95" in log_message
        
    @patch('src.transcription.performance.logger')
    def test_log_worker_performance_handles_edge_cases(self, mock_logger):
        """Test worker performance logging with edge case values."""
        # Test with zero segments
        log_worker_performance(
            worker_name="Worker-Edge",
            processing_time=0.001,
            segments_count=0,
            language="unknown",
            language_prob=0.1
        )
        
        mock_logger.debug.assert_called()
        log_message = mock_logger.debug.call_args[0][0]
        
        assert "0 segments" in log_message
        assert "unknown" in log_message


class TestLogFailureAnalysis:
    """Test suite for log_failure_analysis function."""
    
    @patch('src.transcription.performance.logger')
    def test_log_failure_analysis_warns_on_high_failure_rate(self, mock_logger):
        """Test that high failure rates trigger warnings."""
        failure_window = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 4 failures out of 10
        
        new_last_check = log_failure_analysis(
            failure_window=failure_window,
            processed_chunks=100,
            worker_name="Worker-1",
            current_time=time.time(),
            last_failure_check=0,  # Force check
            check_interval=5.0
        )
        
        # Should update the last check time
        assert new_last_check > 0
        
        # Should log warning about failure rate
        mock_logger.warning.assert_called()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "failure rate" in warning_message.lower()
        assert "4 actual failures" in warning_message
        
    @patch('src.transcription.performance.logger')
    def test_log_failure_analysis_skips_during_warmup(self, mock_logger):
        """Test that failure analysis is skipped during warm-up."""
        failure_window = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # All failures
        
        log_failure_analysis(
            failure_window=failure_window,
            processed_chunks=0,  # No processed chunks = warm-up
            worker_name="Worker-1",
            current_time=time.time(),
            last_failure_check=0,
            check_interval=5.0
        )
        
        # Should not log warning during warm-up
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if "failure rate" in str(call).lower()]
        assert len(warning_calls) == 0
        
    def test_log_failure_analysis_respects_check_interval(self):
        """Test that failure analysis respects the check interval."""
        failure_window = [1] * 15  # Many failures
        current_time = time.time()
        last_check = current_time - 2.0  # 2 seconds ago
        
        with patch('src.transcription.performance.logger') as mock_logger:
            new_last_check = log_failure_analysis(
                failure_window=failure_window,
                processed_chunks=100,
                worker_name="Worker-1",
                current_time=current_time,
                last_failure_check=last_check,
                check_interval=5.0  # 5 second interval
            )
            
        # Should not have updated last check time (interval not reached)
        assert new_last_check == last_check
        
        # Should not have logged anything
        assert not mock_logger.warning.called


class TestGetSystemMetrics:
    """Test suite for get_system_metrics function."""
    
    @patch('src.transcription.performance.psutil')
    def test_get_system_metrics_returns_correct_data(self, mock_psutil):
        """Test that get_system_metrics returns comprehensive system data."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 65.2
        mock_psutil.virtual_memory.return_value.percent = 78.5
        mock_psutil.virtual_memory.return_value.available = 4294967296  # 4GB
        mock_psutil.disk_usage.return_value.percent = 45.0
        
        metrics = get_system_metrics()
        
        expected_keys = [
            'cpu_percent', 'memory_percent', 'memory_available_gb', 
            'disk_percent', 'timestamp'
        ]
        
        for key in expected_keys:
            assert key in metrics
            
        assert metrics['cpu_percent'] == 65.2
        assert metrics['memory_percent'] == 78.5
        assert metrics['memory_available_gb'] == 4.0
        assert metrics['disk_percent'] == 45.0
        assert isinstance(metrics['timestamp'], float)
        
    @patch('src.transcription.performance.psutil')
    def test_get_system_metrics_handles_errors(self, mock_psutil):
        """Test that get_system_metrics handles psutil errors gracefully."""
        # Mock psutil to raise an exception
        mock_psutil.cpu_percent.side_effect = Exception("CPU error")
        
        with patch('src.transcription.performance.logger') as mock_logger:
            metrics = get_system_metrics()
            
        # Should return empty dict on error
        assert metrics == {}
        
        # Should log the error
        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "Error getting system metrics" in error_message
        
    def test_get_system_metrics_timestamp_accuracy(self):
        """Test that get_system_metrics timestamp is accurate."""
        start_time = time.time()
        
        with patch('src.transcription.performance.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.virtual_memory.return_value.percent = 60.0
            mock_psutil.virtual_memory.return_value.available = 2147483648
            mock_psutil.disk_usage.return_value.percent = 30.0
            
            metrics = get_system_metrics()
            
        end_time = time.time()
        
        # Timestamp should be between start and end time
        assert start_time <= metrics['timestamp'] <= end_time


class TestPerformanceIntegration:
    """Integration tests for performance monitoring components."""
    
    def test_performance_monitor_full_workflow(self, performance_monitor):
        """Test a complete performance monitoring workflow."""
        # Create realistic test scenario
        audio_queue = queue.Queue(maxsize=15)
        result_queue = queue.Queue()
        drops_last_minute = deque()
        drop_stats_lock = threading.Lock()
        worker_threads = [MagicMock() for _ in range(3)]
        
        # Add some items to queues
        for i in range(8):
            audio_queue.put((f"audio_data_{i}", float(i)))
            
        for i in range(2):
            result_queue.put({"text": f"result_{i}", "timestamp": float(i)})
            
        # Add some recent drops
        current_time = time.time()
        drops_last_minute.extend([current_time - 30, current_time - 20, current_time - 10])
        
        with patch('src.transcription.performance.logger') as mock_logger:
            # Force logging by resetting last_perf_log
            performance_monitor.last_perf_log = 0
            
            performance_monitor.log_system_performance(
                audio_queue=audio_queue,
                result_queue=result_queue,
                processed_chunks=75,
                avg_time=0.12,
                device="cuda",
                drops_last_minute=drops_last_minute,
                drop_stats_lock=drop_stats_lock,
                worker_threads=worker_threads
            )
            
        # Verify comprehensive logging occurred
        mock_logger.debug.assert_called()
        log_message = mock_logger.debug.call_args[0][0]
        
        # Check for key performance indicators
        assert "Device: CUDA" in log_message
        assert "Chunks: 75" in log_message
        assert "8/15" in log_message  # Queue status
        
    def test_concurrent_performance_monitoring(self, performance_monitor):
        """Test performance monitoring under concurrent access."""
        results = []
        
        def monitor_worker(worker_id):
            try:
                audio_queue = queue.Queue(maxsize=10)
                result_queue = queue.Queue()
                drops_last_minute = deque()
                drop_stats_lock = threading.Lock()
                
                performance_monitor.log_system_performance(
                    audio_queue=audio_queue,
                    result_queue=result_queue,
                    processed_chunks=worker_id * 10,
                    avg_time=0.1 + worker_id * 0.01,
                    device="cuda",
                    drops_last_minute=drops_last_minute,
                    drop_stats_lock=drop_stats_lock,
                    worker_threads=[]
                )
                results.append(True)
            except Exception as e:
                results.append(False)
                
        # Start multiple monitoring threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=monitor_worker, args=(i,))
            threads.append(t)
            t.start()
            
        # Wait for all threads
        for t in threads:
            t.join()
            
        # All monitoring should complete successfully
        assert all(results)
        assert len(results) == 5 