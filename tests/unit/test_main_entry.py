import pytest
import sys
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call, mock_open

# Add src directory to path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import main function and dependencies
from src.main import main


class TestMainEntry:
    """Test suite for main entry point orchestration."""
    
    @pytest.fixture
    def mock_components(self):
        """Create comprehensive mocks for all main components."""
        with patch('src.main.parse_arguments') as mock_parse_args, \
             patch('src.main.initialize_app') as mock_init_app, \
             patch('src.main.VideoSource') as mock_video_source, \
             patch('src.main.TranscriptionEngine') as mock_transcriber, \
             patch('src.main.CaptionOverlay') as mock_caption_overlay, \
             patch('src.main.VideoRunner') as mock_video_runner, \
             patch('src.main.CLOCK') as mock_clock, \
             patch('src.main.cv2') as mock_cv2, \
             patch('src.main.time') as mock_time, \
             patch('src.main.threading') as mock_threading, \
             patch('src.main.logger') as mock_logger, \
             patch.dict('os.environ', {'GIT_COMMIT': 'test-commit-123'}):
            
            # Configure argument parser mock
            mock_args = MagicMock()
            mock_args.source = None
            mock_args.source_arg = None
            mock_args.log_level = 'INFO'
            mock_parse_args.return_value = mock_args
            
            # Configure app initializer mock
            mock_init_app.return_value = ('/tmp/logs', '/tmp/logs/app.log')
            
            # Configure video source mock
            mock_video_instance = MagicMock()
            mock_video_instance.frames_queue.maxsize = 100
            mock_video_instance.warm_up_complete = threading.Event()
            mock_video_instance.warm_up_barrier = MagicMock()
            mock_video_instance.playback_start_time = 1000.0
            mock_video_instance.get_video_info.return_value = (1920, 1080, 30.0)
            mock_video_instance.audio_file = '/tmp/audio.wav'
            mock_video_instance.audio_playing = False
            mock_video_instance.audio_thread = None
            mock_video_instance.__enter__ = MagicMock(return_value=mock_video_instance)
            mock_video_instance.__exit__ = MagicMock(return_value=None)
            mock_video_source.return_value = mock_video_instance
            
            # Configure transcription engine mock
            mock_transcriber_instance = MagicMock()
            mock_transcriber_instance.audio_queue.maxsize = 50
            mock_transcriber_instance.caption_overlay = None
            mock_transcriber_instance.playback_start_time = None
            mock_transcriber_instance.__enter__ = MagicMock(return_value=mock_transcriber_instance)
            mock_transcriber_instance.__exit__ = MagicMock(return_value=None)
            mock_transcriber.return_value = mock_transcriber_instance
            
            # Configure caption overlay mock
            mock_caption_instance = MagicMock()
            mock_caption_instance.__enter__ = MagicMock(return_value=mock_caption_instance)
            mock_caption_instance.__exit__ = MagicMock(return_value=None)
            mock_caption_overlay.return_value = mock_caption_instance
            
            # Configure video runner mock
            mock_runner_instance = MagicMock()
            mock_video_runner.return_value = mock_runner_instance
            
            # Configure clock mock
            mock_clock.start_wall_time = None
            mock_clock.media_seek_pts = 0.0
            
            # Configure time mock to simulate progression
            mock_time.time.side_effect = [
                1000.0,  # Initial time
                1030.0,  # After warm-up
                1030.5,  # After timing stabilization
                1030.5,  # Current playback time calculation
                1031.5,  # Caption scheduling
                1032.5,  # More caption scheduling
                1033.5,  # Final caption scheduling
            ]
            
            # Configure threading mock
            mock_thread = MagicMock()
            mock_threading.Thread.return_value = mock_thread
            
            yield {
                'parse_args': mock_parse_args,
                'init_app': mock_init_app,
                'video_source': mock_video_source,
                'video_instance': mock_video_instance,
                'transcriber': mock_transcriber,
                'transcriber_instance': mock_transcriber_instance,
                'caption_overlay': mock_caption_overlay,
                'caption_instance': mock_caption_instance,
                'video_runner': mock_video_runner,
                'runner_instance': mock_runner_instance,
                'clock': mock_clock,
                'cv2': mock_cv2,
                'time': mock_time,
                'threading': mock_threading,
                'thread': mock_thread,
                'logger': mock_logger,
                'args': mock_args
            }
    
    def test_main_initializes_all_components(self, mock_components):
        """Test that main properly initializes all required components."""
        # Run main function
        main(video_file='/test/video.mp4')
        
        # Verify argument parsing
        mock_components['parse_args'].assert_called_once()
        
        # Verify app initialization
        mock_components['init_app'].assert_called_once_with(log_level='INFO')
        
        # Verify component initialization with correct parameters
        mock_components['video_source'].assert_called_once_with('/test/video.mp4', start_time=325.0)
        mock_components['transcriber'].assert_called_once_with(
            warm_up_complete_event=mock_components['video_instance'].warm_up_complete
        )
        mock_components['caption_overlay'].assert_called_once_with(
            font_scale=1.5,
            font_thickness=2,
            font_color=(255, 255, 255),
            bg_color=(0, 0, 0, 180),
            padding=15,
            y_offset=50
        )
        
        # Verify video runner initialization
        mock_components['video_runner'].assert_called_once_with(
            mock_components['video_instance'],
            mock_components['transcriber_instance'],
            mock_components['caption_instance']
        )
    
    def test_main_handles_missing_video_file(self, mock_components):
        """Test main handles missing video file with fallback."""
        with patch('os.path.expanduser') as mock_expanduser:
            mock_expanduser.return_value = '/home/user'
            
            # Run main without video file
            main()
            
            # Should use fallback path
            expected_path = '/home/user/.video_cache/3YhGU6vVBg8.mp4'
            mock_components['video_source'].assert_called_once_with(expected_path, start_time=325.0)
    
    def test_main_argument_precedence(self, mock_components):
        """Test that command line arguments take precedence correctly."""
        # Configure args with both source options
        mock_components['args'].source = '/arg/source.mp4'
        mock_components['args'].source_arg = '/arg/source_arg.mp4'
        
        main(video_file='/param/video.mp4')
        
        # Should use --source first
        mock_components['video_source'].assert_called_once_with('/arg/source.mp4', start_time=325.0)
    
    def test_main_component_orchestration_sequence(self, mock_components):
        """Test the correct sequence of component initialization and startup."""
        main(video_file='/test/video.mp4')
        
        # Verify transcriber is connected to caption overlay
        assert mock_components['transcriber_instance'].caption_overlay == mock_components['caption_instance']
        
        # Verify transcription is started
        mock_components['transcriber_instance'].start_transcription.assert_called_once()
        
        # Verify audio thread is created and started
        mock_components['threading'].Thread.assert_called()
        mock_components['thread'].start.assert_called()
        
        # Verify warm-up barrier is joined
        mock_components['video_instance'].warm_up_barrier.wait.assert_called_once()
        
        # Verify clock is initialized
        assert mock_components['clock'].start_wall_time == 1030.0
        
        # Verify caption overlay video start time is set
        mock_components['caption_instance'].set_video_start_time.assert_called_once_with(1030.0)
        
        # Verify transcriber playback start time is set
        assert mock_components['transcriber_instance'].playback_start_time == mock_components['video_instance'].playback_start_time
    
    def test_main_test_caption_scheduling(self, mock_components):
        """Test that test captions are properly scheduled."""
        main(video_file='/test/video.mp4')
        
        # Verify test captions are scheduled
        caption_calls = mock_components['caption_instance'].add_caption.call_args_list
        assert len(caption_calls) == 3
        
        # Check first caption
        first_call = caption_calls[0]
        assert first_call[1]['text'] == 'TEST 1: Overlay working'
        assert first_call[1]['duration'] == 10.0
        assert first_call[1]['is_absolute'] is False
        
        # Check second caption
        second_call = caption_calls[1]
        assert second_call[1]['text'] == 'TEST 2: Still alive!'
        
        # Check third caption
        third_call = caption_calls[2]
        assert third_call[1]['text'] == 'TEST 3: Quit with \'q\''
    
    def test_main_audio_playback_setup(self, mock_components):
        """Test that audio playback is properly configured."""
        main(video_file='/test/video.mp4')
        
        # Verify audio playing flag is set
        assert mock_components['video_instance'].audio_playing is True
        
        # Verify audio thread is created with correct target
        thread_calls = mock_components['threading'].Thread.call_args_list
        audio_thread_call = None
        for call in thread_calls:
            if 'target' in call[1] and hasattr(call[1]['target'], '__name__'):
                if call[1]['target'].__name__ == '_play_audio':
                    audio_thread_call = call
                    break
        
        assert audio_thread_call is not None
        assert call[1]['args'] == ('/tmp/audio.wav',)
    
    def test_main_video_runner_execution(self, mock_components):
        """Test that video runner is properly executed."""
        main(video_file='/test/video.mp4')
        
        # Verify video runner prebuffering and execution
        mock_components['runner_instance'].prebuffer_frames.assert_called_once()
        mock_components['runner_instance'].run.assert_called_once()
    
    def test_main_cv2_window_management(self, mock_components):
        """Test OpenCV window creation and cleanup."""
        main(video_file='/test/video.mp4')
        
        # Verify window creation
        mock_components['cv2'].namedWindow.assert_called_once_with(
            "Video with Captions", mock_components['cv2'].WINDOW_NORMAL
        )
        mock_components['cv2'].resizeWindow.assert_called_once_with(
            "Video with Captions", 1920, 1080
        )
        
        # Verify cleanup
        mock_components['cv2'].destroyAllWindows.assert_called_once()
    
    def test_main_build_verification_logging(self, mock_components):
        """Test that build verification information is logged."""
        main(video_file='/test/video.mp4')
        
        # Check for build verification logs
        log_calls = [call[0][0] for call in mock_components['logger'].info.call_args_list]
        
        # Should log build tag with environment variable
        build_log = next((log for log in log_calls if '🚀 BUILD TAG: test-commit-123' in log), None)
        assert build_log is not None
        assert 'frame_q=100' in build_log
        assert 'audio_q=50' in build_log
        
        # Should log runtime verification
        runtime_log = next((log for log in log_calls if '🔧 RUNTIME VERIFICATION' in log), None)
        assert runtime_log is not None
    
    def test_main_warm_up_barrier_error_handling(self, mock_components):
        """Test handling of warm-up barrier errors."""
        # Configure barrier to raise exception
        mock_components['video_instance'].warm_up_barrier.wait.side_effect = threading.BrokenBarrierError()
        
        main(video_file='/test/video.mp4')
        
        # Should handle error gracefully and fall back to event
        mock_components['logger'].error.assert_called_once_with(
            "Warm-up barrier was broken, falling back to event-based sync"
        )
        mock_components['video_instance'].warm_up_complete.set.assert_called_once()
    
    def test_main_exception_handling(self, mock_components):
        """Test that main handles exceptions gracefully."""
        # Configure video source to raise exception
        mock_components['video_source'].side_effect = RuntimeError("Test error")
        
        main(video_file='/test/video.mp4')
        
        # Should log error and clean up
        mock_components['logger'].error.assert_called_once_with("Error: Test error")
        mock_components['cv2'].destroyAllWindows.assert_called_once()
    
    @patch('src.main.os.path.exists')
    def test_main_with_different_log_levels(self, mock_exists, mock_components):
        """Test main with different log levels."""
        mock_exists.return_value = True
        mock_components['args'].log_level = 'DEBUG'
        
        main(video_file='/test/video.mp4')
        
        mock_components['init_app'].assert_called_once_with(log_level='DEBUG')
    
    def test_main_context_manager_cleanup(self, mock_components):
        """Test that context managers are properly used for cleanup."""
        main(video_file='/test/video.mp4')
        
        # Verify all components use context managers
        mock_components['video_instance'].__enter__.assert_called_once()
        mock_components['video_instance'].__exit__.assert_called_once()
        
        mock_components['transcriber_instance'].__enter__.assert_called_once()
        mock_components['transcriber_instance'].__exit__.assert_called_once()
        
        mock_components['caption_instance'].__enter__.assert_called_once()
        mock_components['caption_instance'].__exit__.assert_called_once()


class TestMainEntryIntegration:
    """Integration tests for main entry point with minimal mocking."""
    
    @patch('src.main.cv2')
    @patch('src.main.VideoRunner')
    def test_main_component_interactions(self, mock_video_runner, mock_cv2):
        """Test real component interactions with minimal mocking."""
        with patch('src.main.VideoSource') as mock_vs, \
             patch('src.main.TranscriptionEngine') as mock_te, \
             patch('src.main.CaptionOverlay') as mock_co:
            
            # Configure minimal mocks for context managers
            for mock_comp in [mock_vs, mock_te, mock_co]:
                instance = MagicMock()
                instance.__enter__ = MagicMock(return_value=instance)
                instance.__exit__ = MagicMock(return_value=None)
                mock_comp.return_value = instance
                
            # Configure required attributes
            video_instance = mock_vs.return_value
            video_instance.frames_queue.maxsize = 100
            video_instance.warm_up_complete = threading.Event()
            video_instance.warm_up_barrier = MagicMock()
            video_instance.get_video_info.return_value = (1920, 1080, 30.0)
            video_instance.audio_file = None  # No audio file
            
            transcriber_instance = mock_te.return_value
            transcriber_instance.audio_queue.maxsize = 50
            
            with patch('src.main.time.time', side_effect=[1000.0, 1030.0, 1030.5]):
                main(video_file='/test/video.mp4')
            
            # Verify components are properly connected
            assert transcriber_instance.caption_overlay is not None
            assert transcriber_instance.playback_start_time is not None


class TestMainEntryEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_main_with_missing_environment_variables(self, mock_components):
        """Test main behavior with missing environment variables."""
        with patch.dict('os.environ', {}, clear=True):
            main(video_file='/test/video.mp4')
            
            # Should use default 'dev' for git commit
            log_calls = [call[0][0] for call in mock_components['logger'].info.call_args_list]
            build_log = next((log for log in log_calls if '🚀 BUILD TAG: dev' in log), None)
            assert build_log is not None
    
    def test_main_clock_already_initialized(self, mock_components):
        """Test main when clock is already initialized."""
        mock_components['clock'].start_wall_time = 999.0  # Already set
        
        main(video_file='/test/video.mp4')
        
        # Should not override existing clock time
        assert mock_components['clock'].start_wall_time == 999.0 