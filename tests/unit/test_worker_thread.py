import pytest
import sys
import time
import queue
import threading
import collections
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add src directory to path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.transcription.worker import (
    run_transcription_worker, _process_audio_segment, 
    _transcribe_with_retry, _update_performance_counters,
    _queue_transcription_results
)


class TestTranscriptionWorker:
    """Test suite for transcription worker functionality."""
    
    @pytest.fixture
    def worker_setup(self):
        """Set up common worker test dependencies."""
        audio_queue = queue.Queue(maxsize=10)
        result_queue = queue.Queue(maxsize=10)
        model = MagicMock()
        is_running_flag = threading.Event()
        is_running_flag.set()
        performance_monitor = MagicMock()
        processed_chunks_counter = MagicMock()
        processed_chunks_counter.value = 0
        processed_chunks_counter.get_lock.return_value.__enter__ = MagicMock()
        processed_chunks_counter.get_lock.return_value.__exit__ = MagicMock()
        avg_time_tracker = MagicMock()
        avg_time_tracker.value = 0.0
        avg_time_tracker.get_lock.return_value.__enter__ = MagicMock()
        avg_time_tracker.get_lock.return_value.__exit__ = MagicMock()
        sampling_rate = 16000
        warm_up_mode_flag = threading.Event()
        drops_last_minute = collections.deque()
        drop_stats_lock = threading.Lock()
        worker_threads = []
        device = "cpu"
        
        return {
            'audio_queue': audio_queue,
            'result_queue': result_queue,
            'model': model,
            'is_running_flag': is_running_flag,
            'performance_monitor': performance_monitor,
            'processed_chunks_counter': processed_chunks_counter,
            'avg_time_tracker': avg_time_tracker,
            'sampling_rate': sampling_rate,
            'warm_up_mode_flag': warm_up_mode_flag,
            'drops_last_minute': drops_last_minute,
            'drop_stats_lock': drop_stats_lock,
            'worker_threads': worker_threads,
            'device': device
        }
    
    @patch('src.transcription.worker.logger')
    @patch('src.transcription.worker.time')
    def test_worker_initialization_and_startup(self, mock_time, mock_logger, worker_setup):
        """Test worker initialization and startup logging."""
        mock_time.time.return_value = 1000.0
        
        # Set up to exit immediately
        worker_setup['is_running_flag'].clear()
        
        run_transcription_worker(**worker_setup)
        
        # Should log startup
        startup_calls = [call for call in mock_logger.info.call_args_list 
                        if 'Transcription worker started' in str(call)]
        assert len(startup_calls) > 0
        
        # Should log shutdown
        shutdown_calls = [call for call in mock_logger.info.call_args_list 
                         if 'Transcription worker stopped' in str(call)]
        assert len(shutdown_calls) > 0
    
    @patch('src.transcription.worker.logger')
    @patch('src.transcription.worker.time')
    def test_worker_processes_audio_segments(self, mock_time, mock_logger, worker_setup):
        """Test worker processes audio segments successfully."""
        mock_time.time.side_effect = [1000.0, 1000.1, 1000.2]
        
        # Set up audio data
        audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of audio
        timestamp = 5.0
        worker_setup['audio_queue'].put((audio_data, timestamp))
        
        # Mock successful transcription
        mock_segment = MagicMock()
        mock_segment.text = "Test transcription"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -0.5
        mock_segment.words = []
        
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        
        worker_setup['model'].transcribe.return_value = ([mock_segment], mock_info)
        
        # Run worker for short time
        def stop_worker():
            time.sleep(0.1)
            worker_setup['is_running_flag'].clear()
        
        stop_thread = threading.Thread(target=stop_worker)
        stop_thread.start()
        
        run_transcription_worker(**worker_setup)
        stop_thread.join()
        
        # Should have processed the audio
        assert worker_setup['result_queue'].qsize() > 0
        result = worker_setup['result_queue'].get()
        assert result['text'] == 'Test transcription'
        assert result['start'] == 5.0  # timestamp + segment.start
        assert result['end'] == 6.0    # timestamp + segment.end
    
    @patch('src.transcription.worker.logger')
    def test_worker_handles_empty_queue(self, mock_logger, worker_setup):
        """Test worker handles empty audio queue gracefully."""
        # Empty queue - worker should wait and continue
        def stop_worker():
            time.sleep(0.1)
            worker_setup['is_running_flag'].clear()
        
        stop_thread = threading.Thread(target=stop_worker)
        stop_thread.start()
        
        run_transcription_worker(**worker_setup)
        stop_thread.join()
        
        # Should log trace messages about empty queue
        trace_calls = [call for call in mock_logger.trace.call_args_list 
                      if 'queue empty' in str(call).lower()]
        assert len(trace_calls) > 0
    
    @patch('src.transcription.worker.logger')
    def test_worker_handles_empty_audio_segment(self, mock_logger, worker_setup):
        """Test worker handles empty audio segments."""
        # Add empty audio segment
        worker_setup['audio_queue'].put((None, 1.0))
        worker_setup['audio_queue'].put((np.array([]), 2.0))
        
        def stop_worker():
            time.sleep(0.1)
            worker_setup['is_running_flag'].clear()
        
        stop_thread = threading.Thread(target=stop_worker)
        stop_thread.start()
        
        run_transcription_worker(**worker_setup)
        stop_thread.join()
        
        # Should log warnings about empty segments
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if 'empty audio segment' in str(call).lower()]
        assert len(warning_calls) >= 2
    
    @patch('src.transcription.worker.logger')
    def test_worker_handles_processing_failures(self, mock_logger, worker_setup):
        """Test worker handles processing failures with backoff."""
        # Set up audio data
        audio_data = np.zeros(16000, dtype=np.float32)
        worker_setup['audio_queue'].put((audio_data, 1.0))
        
        # Mock transcription failure
        worker_setup['model'].transcribe.side_effect = RuntimeError("Test error")
        
        def stop_worker():
            time.sleep(0.2)
            worker_setup['is_running_flag'].clear()
        
        stop_thread = threading.Thread(target=stop_worker)
        stop_thread.start()
        
        run_transcription_worker(**worker_setup)
        stop_thread.join()
        
        # Should log errors about processing failures
        error_calls = [call for call in mock_logger.error.call_args_list 
                      if 'Error processing audio segment' in str(call)]
        assert len(error_calls) > 0


class TestProcessAudioSegment:
    """Test suite for _process_audio_segment function."""
    
    @pytest.fixture
    def segment_setup(self):
        """Set up for audio segment processing tests."""
        audio_data = np.zeros(16000, dtype=np.float32)
        timestamp = 5.0
        model = MagicMock()
        result_queue = queue.Queue()
        worker_name = "TestWorker"
        sampling_rate = 16000
        processed_chunks_counter = MagicMock()
        processed_chunks_counter.value = 0
        processed_chunks_counter.get_lock.return_value.__enter__ = MagicMock()
        processed_chunks_counter.get_lock.return_value.__exit__ = MagicMock()
        avg_time_tracker = MagicMock()
        avg_time_tracker.value = 0.0
        avg_time_tracker.get_lock.return_value.__enter__ = MagicMock()
        avg_time_tracker.get_lock.return_value.__exit__ = MagicMock()
        
        return {
            'audio_data': audio_data,
            'timestamp': timestamp,
            'model': model,
            'result_queue': result_queue,
            'worker_name': worker_name,
            'sampling_rate': sampling_rate,
            'processed_chunks_counter': processed_chunks_counter,
            'avg_time_tracker': avg_time_tracker
        }
    
    @patch('src.transcription.worker.log_audio_info')
    @patch('src.transcription.worker.log_worker_performance')
    @patch('src.transcription.worker.time')
    def test_process_audio_segment_success(self, mock_time, mock_log_perf, mock_log_audio, segment_setup):
        """Test successful audio segment processing."""
        mock_time.time.side_effect = [1000.0, 1000.5]  # 0.5s processing time
        
        # Mock successful transcription
        mock_segment = MagicMock()
        mock_segment.text = "Test transcription"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -0.3
        mock_segment.words = []
        
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        
        segment_setup['model'].transcribe.return_value = ([mock_segment], mock_info)
        
        result = _process_audio_segment(**segment_setup)
        
        assert result is True
        assert segment_setup['result_queue'].qsize() == 1
        
        # Check result content
        queued_result = segment_setup['result_queue'].get()
        assert queued_result['text'] == 'Test transcription'
        assert queued_result['start'] == 5.0  # timestamp + segment.start
        assert queued_result['end'] == 6.0    # timestamp + segment.end
        assert queued_result['worker'] == 'TestWorker'
        
        # Should log performance
        mock_log_perf.assert_called_once()
        mock_log_audio.assert_called_once()
    
    @patch('src.transcription.worker.logger')
    def test_process_audio_segment_transcription_failure(self, mock_logger, segment_setup):
        """Test audio segment processing with transcription failure."""
        # Mock transcription failure
        segment_setup['model'].transcribe.side_effect = RuntimeError("Transcription failed")
        
        result = _process_audio_segment(**segment_setup)
        
        assert result is False
        assert segment_setup['result_queue'].empty()
        
        # Should log error
        error_calls = [call for call in mock_logger.error.call_args_list 
                      if 'Error processing audio segment' in str(call)]
        assert len(error_calls) > 0


class TestTranscribeWithRetry:
    """Test suite for _transcribe_with_retry function."""
    
    def test_transcribe_with_retry_success_first_attempt(self):
        """Test successful transcription on first attempt."""
        model = MagicMock()
        audio_data = np.zeros(16000, dtype=np.float32)
        worker_name = "TestWorker"
        
        # Mock successful transcription
        mock_segment = MagicMock()
        mock_segment.text = "Success"
        mock_info = MagicMock()
        
        model.transcribe.return_value = ([mock_segment], mock_info)
        
        segments, info = _transcribe_with_retry(model, audio_data, worker_name)
        
        assert segments == [mock_segment]
        assert info == mock_info
        assert model.transcribe.call_count == 1
    
    @patch('src.transcription.worker.logger')
    @patch('src.transcription.worker.time')
    def test_transcribe_with_retry_cuda_oom_retry(self, mock_time, mock_logger):
        """Test retry logic for CUDA out of memory errors."""
        model = MagicMock()
        audio_data = np.zeros(16000, dtype=np.float32)
        worker_name = "TestWorker"
        
        # Mock CUDA OOM on first attempt, success on second
        mock_segment = MagicMock()
        mock_info = MagicMock()
        
        model.transcribe.side_effect = [
            RuntimeError("CUDA out of memory"),
            ([mock_segment], mock_info)
        ]
        
        segments, info = _transcribe_with_retry(model, audio_data, worker_name)
        
        assert segments == [mock_segment]
        assert info == mock_info
        assert model.transcribe.call_count == 2
        
        # Should log retry warning
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if 'CUDA OOM' in str(call)]
        assert len(warning_calls) > 0
        
        # Should have slept for backoff
        mock_time.sleep.assert_called()
    
    def test_transcribe_with_retry_non_retryable_error(self):
        """Test non-retryable errors are raised immediately."""
        model = MagicMock()
        audio_data = np.zeros(16000, dtype=np.float32)
        worker_name = "TestWorker"
        
        # Mock non-retryable error
        model.transcribe.side_effect = RuntimeError("Non-retryable error")
        
        with pytest.raises(RuntimeError, match="Non-retryable error"):
            _transcribe_with_retry(model, audio_data, worker_name)
        
        assert model.transcribe.call_count == 1
    
    @patch('src.transcription.worker.time')
    def test_transcribe_with_retry_max_retries_exceeded(self, mock_time):
        """Test behavior when max retries are exceeded."""
        model = MagicMock()
        audio_data = np.zeros(16000, dtype=np.float32)
        worker_name = "TestWorker"
        
        # Mock CUDA OOM on all attempts
        model.transcribe.side_effect = RuntimeError("CUDA out of memory")
        
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            _transcribe_with_retry(model, audio_data, worker_name)
        
        # Should have tried MAX_RETRIES times (3 by default)
        assert model.transcribe.call_count == 3


class TestUpdatePerformanceCounters:
    """Test suite for _update_performance_counters function."""
    
    def test_update_performance_counters_first_update(self):
        """Test performance counter update with no previous data."""
        processed_chunks_counter = MagicMock()
        processed_chunks_counter.value = 0
        processed_chunks_counter.get_lock.return_value.__enter__ = MagicMock()
        processed_chunks_counter.get_lock.return_value.__exit__ = MagicMock()
        
        avg_time_tracker = MagicMock()
        avg_time_tracker.value = 0.0
        avg_time_tracker.get_lock.return_value.__enter__ = MagicMock()
        avg_time_tracker.get_lock.return_value.__exit__ = MagicMock()
        
        processing_time = 0.5
        
        _update_performance_counters(processed_chunks_counter, avg_time_tracker, processing_time)
        
        # Should increment counter
        assert processed_chunks_counter.value == 1
        
        # Should set average to current processing time
        assert avg_time_tracker.value == 0.5
    
    def test_update_performance_counters_exponential_moving_average(self):
        """Test exponential moving average calculation."""
        processed_chunks_counter = MagicMock()
        processed_chunks_counter.value = 5
        processed_chunks_counter.get_lock.return_value.__enter__ = MagicMock()
        processed_chunks_counter.get_lock.return_value.__exit__ = MagicMock()
        
        avg_time_tracker = MagicMock()
        avg_time_tracker.value = 1.0  # Previous average
        avg_time_tracker.get_lock.return_value.__enter__ = MagicMock()
        avg_time_tracker.get_lock.return_value.__exit__ = MagicMock()
        
        processing_time = 0.5  # New time
        
        _update_performance_counters(processed_chunks_counter, avg_time_tracker, processing_time)
        
        # Should calculate exponential moving average
        # alpha = 0.1, so: 0.1 * 0.5 + 0.9 * 1.0 = 0.95
        expected_avg = 0.1 * 0.5 + 0.9 * 1.0
        assert abs(avg_time_tracker.value - expected_avg) < 0.001
    
    @patch('src.transcription.worker.logger')
    def test_update_performance_counters_exception_handling(self, mock_logger):
        """Test exception handling in performance counter update."""
        processed_chunks_counter = MagicMock()
        processed_chunks_counter.get_lock.side_effect = Exception("Lock error")
        
        avg_time_tracker = MagicMock()
        processing_time = 0.5
        
        # Should not raise exception
        _update_performance_counters(processed_chunks_counter, avg_time_tracker, processing_time)
        
        # Should log warning
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if 'Error updating performance counters' in str(call)]
        assert len(warning_calls) > 0


class TestQueueTranscriptionResults:
    """Test suite for _queue_transcription_results function."""
    
    def test_queue_transcription_results_basic(self):
        """Test basic transcription result queuing."""
        # Create mock segments
        mock_segment1 = MagicMock()
        mock_segment1.text = "Hello world"
        mock_segment1.start = 0.0
        mock_segment1.end = 1.0
        mock_segment1.avg_logprob = -0.2
        mock_segment1.words = []
        
        mock_segment2 = MagicMock()
        mock_segment2.text = "How are you?"
        mock_segment2.start = 1.0
        mock_segment2.end = 2.0
        mock_segment2.avg_logprob = -0.3
        mock_segment2.words = []
        
        segments = [mock_segment1, mock_segment2]
        result_queue = queue.Queue()
        timestamp = 5.0
        processing_time = 0.5
        worker_name = "TestWorker"
        
        _queue_transcription_results(segments, result_queue, timestamp, processing_time, worker_name)
        
        assert result_queue.qsize() == 2
        
        # Check first result
        result1 = result_queue.get()
        assert result1['text'] == 'Hello world'
        assert result1['start'] == 5.0  # timestamp + segment.start
        assert result1['end'] == 6.0    # timestamp + segment.end
        assert result1['worker'] == 'TestWorker'
        assert result1['processing_time'] == 0.5
        
        # Check second result
        result2 = result_queue.get()
        assert result2['text'] == 'How are you?'
        assert result2['start'] == 6.0
        assert result2['end'] == 7.0
    
    def test_queue_transcription_results_with_words(self):
        """Test queuing results with word-level timestamps."""
        # Create mock word
        mock_word = MagicMock()
        mock_word.word = "hello"
        mock_word.start = 0.1
        mock_word.end = 0.5
        mock_word.probability = 0.95
        
        # Create mock segment with words
        mock_segment = MagicMock()
        mock_segment.text = "hello"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -0.2
        mock_segment.words = [mock_word]
        
        segments = [mock_segment]
        result_queue = queue.Queue()
        timestamp = 2.0
        processing_time = 0.3
        worker_name = "TestWorker"
        
        _queue_transcription_results(segments, result_queue, timestamp, processing_time, worker_name)
        
        result = result_queue.get()
        assert len(result['words']) == 1
        
        word_result = result['words'][0]
        assert word_result['word'] == 'hello'
        assert word_result['start'] == 2.1  # timestamp + word.start
        assert word_result['end'] == 2.5    # timestamp + word.end
        assert word_result['probability'] == 0.95
    
    @patch('src.transcription.worker.logger')
    def test_queue_transcription_results_queue_full(self, mock_logger):
        """Test handling of full result queue."""
        mock_segment = MagicMock()
        mock_segment.text = "Test"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -0.2
        mock_segment.words = []
        
        segments = [mock_segment]
        result_queue = queue.Queue(maxsize=1)
        result_queue.put("blocking_item")  # Fill the queue
        
        timestamp = 0.0
        processing_time = 0.5
        worker_name = "TestWorker"
        
        _queue_transcription_results(segments, result_queue, timestamp, processing_time, worker_name)
        
        # Should log warning about full queue
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if 'Result queue full' in str(call)]
        assert len(warning_calls) > 0
    
    @patch('src.transcription.worker.logger')
    def test_queue_transcription_results_exception_handling(self, mock_logger):
        """Test exception handling in result queuing."""
        # Create problematic segment that causes exception
        mock_segment = MagicMock()
        mock_segment.text = "Test"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -0.2
        mock_segment.words = []
        
        segments = [mock_segment]
        
        # Mock queue that raises exception
        result_queue = MagicMock()
        result_queue.put.side_effect = Exception("Queue error")
        
        timestamp = 0.0
        processing_time = 0.5
        worker_name = "TestWorker"
        
        # Should not raise exception
        _queue_transcription_results(segments, result_queue, timestamp, processing_time, worker_name)
        
        # Should log error
        error_calls = [call for call in mock_logger.error.call_args_list 
                      if 'Error queuing results' in str(call)]
        assert len(error_calls) > 0 