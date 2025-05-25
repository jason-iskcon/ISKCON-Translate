import pytest
import sys
import time
import queue
import threading
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add src directory to path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.core.video_runner import VideoRunner


class TestVideoRunner:
    """Test suite for VideoRunner class."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mocks for VideoRunner dependencies."""
        mock_video_source = MagicMock()
        mock_video_source.get_video_info.return_value = (1920, 1080, 30.0)
        mock_video_source.is_running = True
        mock_video_source.audio_playing = True
        mock_video_source.audio_position = 0.0
        mock_video_source.audio_position_lock = threading.Lock()
        mock_video_source.playback_start_time = 1000.0
        mock_video_source.frames_queue.qsize.return_value = 5
        mock_video_source.frames_queue.maxsize = 100
        
        mock_transcriber = MagicMock()
        mock_transcriber.get_transcription.return_value = None
        mock_transcriber.audio_queue.qsize.return_value = 3
        mock_transcriber.audio_queue.maxsize = 50
        
        mock_caption_overlay = MagicMock()
        mock_caption_overlay.captions = []
        mock_caption_overlay.overlay_captions.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        return {
            'video_source': mock_video_source,
            'transcriber': mock_transcriber,
            'caption_overlay': mock_caption_overlay
        }
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_video_runner_initialization(self, mock_clock, mock_cv2, mock_dependencies):
        """Test VideoRunner initialization."""
        mock_clock.get_video_relative_time.return_value = 0.0
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Verify initialization
        assert runner.video_source == mock_dependencies['video_source']
        assert runner.transcriber == mock_dependencies['transcriber']
        assert runner.caption_overlay == mock_dependencies['caption_overlay']
        assert runner.width == 1920
        assert runner.height == 1080
        assert runner.fps == 30.0
        assert runner.target_frame_time == 1.0 / 30.0
        assert runner.frame_count == 0
        assert runner.paused is False
        assert runner.running is False
        
        # Verify window creation
        mock_cv2.namedWindow.assert_called_once_with("Video with Captions", mock_cv2.WINDOW_NORMAL)
        mock_cv2.resizeWindow.assert_called_once_with("Video with Captions", 1920, 1080)
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_prebuffer_frames(self, mock_clock, mock_cv2, mock_dependencies):
        """Test frame prebuffering logic."""
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Mock frame data
        frame_data = (np.zeros((1080, 1920, 3), dtype=np.uint8), 1.0)
        mock_dependencies['video_source'].get_frame.side_effect = [frame_data] * 20
        
        runner.prebuffer_frames()
        
        # Should prebuffer half of max buffer size
        assert len(runner.frame_buffer) == runner.max_buffer_size // 2
        
        # Should call get_frame multiple times
        assert mock_dependencies['video_source'].get_frame.call_count >= 15
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    @patch('src.core.video_runner.time')
    def test_process_frame(self, mock_time, mock_clock, mock_cv2, mock_dependencies):
        """Test frame processing logic."""
        mock_time.time.return_value = 1001.0
        mock_clock.get_video_relative_time.return_value = 1.0
        mock_clock.get_elapsed_time.return_value = 1.0
        mock_clock.is_initialized.return_value = True
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Mock frame data
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_data = (frame, 1.0)
        
        result = runner.process_frame(frame_data)
        
        # Verify frame processing
        assert runner.frame_count == 1
        assert runner.current_video_time == 0.0  # Uses audio position
        
        # Verify caption overlay was called
        mock_dependencies['caption_overlay'].overlay_captions.assert_called_once()
        overlay_call = mock_dependencies['caption_overlay'].overlay_captions.call_args
        assert overlay_call[1]['current_time'] == 1.0
        assert overlay_call[1]['frame_count'] == 1
        
        # Verify result
        assert result is not None
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_process_transcriptions(self, mock_clock, mock_cv2, mock_dependencies):
        """Test transcription processing."""
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Mock transcription data
        transcription = {
            'text': 'Test transcription',
            'timestamp': 1.0,
            'end_time': 4.0
        }
        mock_dependencies['transcriber'].get_transcription.side_effect = [transcription, None]
        
        runner._process_transcriptions()
        
        # Verify transcription was processed
        mock_dependencies['transcriber'].get_transcription.assert_called()
        
        # Verify caption was added
        mock_dependencies['caption_overlay'].add_caption.assert_called_once_with(
            'Test transcription',
            timestamp=1.0,
            duration=3.0,
            is_absolute=False
        )
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_handle_transcription(self, mock_clock, mock_cv2, mock_dependencies):
        """Test individual transcription handling."""
        mock_clock.get_elapsed_time.return_value = 2.0
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Test valid transcription
        transcription = {
            'text': 'Valid transcription',
            'timestamp': 2.5,
            'end_time': 5.0
        }
        
        runner._handle_transcription(transcription)
        
        # Should add caption
        mock_dependencies['caption_overlay'].add_caption.assert_called_once_with(
            'Valid transcription',
            timestamp=2.5,
            duration=2.5,
            is_absolute=False
        )
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_handle_transcription_timing_mismatch(self, mock_clock, mock_cv2, mock_dependencies):
        """Test transcription handling with timing mismatch."""
        mock_clock.get_elapsed_time.return_value = 2.0
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Test transcription with large time difference
        transcription = {
            'text': 'Old transcription',
            'timestamp': 20.0,  # 18 seconds ahead
            'end_time': 23.0
        }
        
        runner._handle_transcription(transcription)
        
        # Should not add caption due to timing mismatch
        mock_dependencies['caption_overlay'].add_caption.assert_not_called()
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_handle_transcription_empty_text(self, mock_clock, mock_cv2, mock_dependencies):
        """Test transcription handling with empty text."""
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Test transcription with empty text
        transcription = {
            'text': '   ',  # Whitespace only
            'timestamp': 1.0,
            'end_time': 4.0
        }
        
        runner._handle_transcription(transcription)
        
        # Should not add caption for empty text
        mock_dependencies['caption_overlay'].add_caption.assert_not_called()
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_add_debug_info(self, mock_clock, mock_cv2, mock_dependencies):
        """Test debug information overlay."""
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Set up mock captions
        mock_dependencies['caption_overlay'].captions = [
            {'start_time': 0.5, 'end_time': 3.5, 'text': 'Active caption'},
            {'start_time': 5.0, 'end_time': 8.0, 'text': 'Future caption'}
        ]
        
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        runner.frame_count = 30  # Trigger detailed logging
        
        runner._add_debug_info(frame, 2.0)
        
        # Verify cv2.putText was called for debug overlay
        assert mock_cv2.putText.call_count >= 2  # At least 2 debug text overlays
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    @patch('src.core.video_runner.time')
    def test_run_main_loop(self, mock_time, mock_clock, mock_cv2, mock_dependencies):
        """Test the main video runner loop."""
        # Mock time progression
        mock_time.time.side_effect = [1000.0, 1000.033, 1000.066, 1000.099]
        mock_clock.get_video_relative_time.return_value = 1.0
        mock_clock.get_elapsed_time.return_value = 1.0
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Mock frame data
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_data = (frame, 1.0)
        
        # Set up limited loop
        mock_dependencies['video_source'].get_frame.side_effect = [frame_data, frame_data, None]
        mock_dependencies['video_source'].is_running = True
        
        # Mock keyboard input to stop after a few frames
        mock_cv2.waitKey.side_effect = [255, 255, ord('q')]  # No key, no key, then 'q'
        
        runner.run()
        
        # Verify the loop executed
        assert runner.frame_count >= 2
        assert not runner.running  # Should have stopped
        
        # Verify cleanup was called
        mock_cv2.destroyAllWindows.assert_called()
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_handle_key_press_quit(self, mock_clock, mock_cv2, mock_dependencies):
        """Test quit key handling."""
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        runner.running = True
        
        # Test 'q' key
        runner._handle_key_press(ord('q'))
        assert not runner.running
        
        # Test ESC key
        runner.running = True
        runner._handle_key_press(27)
        assert not runner.running
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    @patch('src.core.video_runner.time')
    def test_handle_key_press_pause(self, mock_time, mock_clock, mock_cv2, mock_dependencies):
        """Test pause key handling."""
        mock_time.time.return_value = 1000.0
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        assert not runner.paused
        
        # Test 'p' key
        runner._handle_key_press(ord('p'))
        assert runner.paused
        
        # Test 'p' key again
        runner._handle_key_press(ord('p'))
        assert not runner.paused
        
        # Test SPACE key
        runner._handle_key_press(32)
        assert runner.paused
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_cleanup(self, mock_clock, mock_cv2, mock_dependencies):
        """Test cleanup functionality."""
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        runner.frame_count = 100
        runner.running = True
        
        runner.cleanup()
        
        assert not runner.running
        mock_cv2.destroyAllWindows.assert_called_once()


class TestVideoRunnerEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mocks for VideoRunner dependencies."""
        mock_video_source = MagicMock()
        mock_video_source.get_video_info.return_value = (1920, 1080, 30.0)
        mock_video_source.is_running = True
        mock_video_source.audio_playing = False  # No audio
        mock_video_source.audio_position = 0.0
        mock_video_source.audio_position_lock = threading.Lock()
        mock_video_source.playback_start_time = 0.0  # Uninitialized
        mock_video_source.frames_queue.qsize.return_value = 0
        mock_video_source.frames_queue.maxsize = 100
        
        mock_transcriber = MagicMock()
        mock_transcriber.get_transcription.return_value = None
        mock_transcriber.audio_queue.qsize.return_value = 0
        mock_transcriber.audio_queue.maxsize = 50
        
        mock_caption_overlay = MagicMock()
        mock_caption_overlay.captions = []
        
        return {
            'video_source': mock_video_source,
            'transcriber': mock_transcriber,
            'caption_overlay': mock_caption_overlay
        }
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_process_frame_uninitialized_playback_time(self, mock_clock, mock_cv2, mock_dependencies):
        """Test frame processing when playback time is uninitialized."""
        mock_clock.is_initialized.return_value = True
        mock_clock.get_video_relative_time.return_value = 0.0
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_data = (frame, 1.0)
        
        # Should handle uninitialized playback time gracefully
        result = runner.process_frame(frame_data)
        assert result is not None
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_process_frame_clock_not_initialized(self, mock_clock, mock_cv2, mock_dependencies):
        """Test frame processing when clock is not initialized."""
        mock_clock.is_initialized.return_value = False
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_data = (frame, 1.0)
        
        # Should fallback to relative_time = 0.0
        result = runner.process_frame(frame_data)
        assert result is not None
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_handle_transcription_exception(self, mock_clock, mock_cv2, mock_dependencies):
        """Test transcription handling when caption overlay raises exception."""
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Mock caption overlay to raise exception
        mock_dependencies['caption_overlay'].add_caption.side_effect = ValueError("Test error")
        
        transcription = {
            'text': 'Test transcription',
            'timestamp': 1.0,
            'end_time': 4.0
        }
        
        # Should handle exception gracefully
        runner._handle_transcription(transcription)
        
        # Exception should be caught and logged
        mock_dependencies['caption_overlay'].add_caption.assert_called_once()
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    @patch('src.core.video_runner.time')
    def test_run_with_no_frames(self, mock_time, mock_clock, mock_cv2, mock_dependencies):
        """Test run loop when no frames are available."""
        mock_time.time.return_value = 1000.0
        mock_time.sleep = MagicMock()
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Mock no frames available
        mock_dependencies['video_source'].get_frame.return_value = None
        mock_dependencies['video_source'].is_running = False
        
        # Mock keyboard input to quit immediately
        mock_cv2.waitKey.return_value = ord('q')
        
        runner.run()
        
        # Should handle gracefully and exit
        assert not runner.running
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    @patch('src.core.video_runner.time')
    def test_run_with_slow_rendering(self, mock_time, mock_clock, mock_cv2, mock_dependencies):
        """Test run loop with slow frame rendering."""
        # Mock slow time progression
        mock_time.time.side_effect = [1000.0, 1000.05, 1000.1, 1000.15]  # 50ms per frame
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_data = (frame, 1.0)
        
        # Set up limited frames to avoid infinite loop
        mock_dependencies['video_source'].get_frame.side_effect = [frame_data, None]
        mock_dependencies['video_source'].is_running = False
        
        mock_cv2.waitKey.return_value = 255  # No key pressed
        
        runner.run()
        
        # Should complete despite slow rendering
        assert not runner.running


class TestVideoRunnerPerformance:
    """Performance and stress testing for VideoRunner."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create performance test mocks."""
        mock_video_source = MagicMock()
        mock_video_source.get_video_info.return_value = (1920, 1080, 60.0)  # High FPS
        mock_video_source.is_running = True
        mock_video_source.audio_playing = True
        mock_video_source.audio_position = 0.0
        mock_video_source.audio_position_lock = threading.Lock()
        mock_video_source.playback_start_time = 1000.0
        mock_video_source.frames_queue.qsize.return_value = 50
        mock_video_source.frames_queue.maxsize = 100
        mock_video_source._consecutive_drops = 0
        
        mock_transcriber = MagicMock()
        mock_transcriber.audio_queue.qsize.return_value = 25
        mock_transcriber.audio_queue.maxsize = 50
        
        mock_caption_overlay = MagicMock()
        mock_caption_overlay.captions = []
        
        return {
            'video_source': mock_video_source,
            'transcriber': mock_transcriber,
            'caption_overlay': mock_caption_overlay
        }
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    def test_high_fps_handling(self, mock_clock, mock_cv2, mock_dependencies):
        """Test VideoRunner with high FPS video."""
        mock_clock.get_video_relative_time.return_value = 1.0
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Should handle 60 FPS correctly
        assert runner.fps == 60.0
        assert runner.target_frame_time == 1.0 / 60.0
        
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_data = (frame, 1.0)
        
        # Process multiple frames quickly
        for _ in range(10):
            result = runner.process_frame(frame_data)
            assert result is not None
        
        assert runner.frame_count == 10
    
    @patch('src.core.video_runner.cv2')
    @patch('src.core.video_runner.CLOCK')
    @patch('src.core.video_runner.time')
    def test_heartbeat_logging(self, mock_time, mock_clock, mock_cv2, mock_dependencies):
        """Test periodic heartbeat logging."""
        # Mock time to trigger heartbeat
        mock_time.time.side_effect = [1000.0, 1002.1]  # 2.1 seconds apart
        mock_clock.get_elapsed_time.return_value = 2.0
        
        runner = VideoRunner(
            mock_dependencies['video_source'],
            mock_dependencies['transcriber'],
            mock_dependencies['caption_overlay']
        )
        
        # Set up for heartbeat trigger
        runner._last_stats_log_time = 1000.0
        runner._latest_audio_rel = 2.0
        
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_data = (frame, 1.0)
        
        # Mock single frame to avoid infinite loop
        mock_dependencies['video_source'].get_frame.side_effect = [frame_data, None]
        mock_dependencies['video_source'].is_running = False
        mock_cv2.waitKey.return_value = 255
        
        with patch('src.core.video_runner.logger') as mock_logger:
            runner.run()
            
            # Should log heartbeat information
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            heartbeat_logs = [log for log in info_calls if '📊 [HB]' in log]
            assert len(heartbeat_logs) > 0 