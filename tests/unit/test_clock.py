import pytest
import time
import sys
from pathlib import Path
from unittest.mock import patch

# Add src directory to path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.clock import PlaybackClock, CLOCK


@pytest.fixture
def clock():
    """Create a fresh PlaybackClock instance for each test."""
    clock = PlaybackClock()
    yield clock
    # Cleanup after each test
    clock.reset()


class TestPlaybackClock:
    """Test suite for PlaybackClock functionality."""
    
    def test_initialize_sets_start_time_correctly(self, clock):
        """Test that initialize sets the media seek position correctly."""
        seek_pts = 325.0
        
        clock.initialize(seek_pts)
        
        assert clock.media_seek_pts == seek_pts
        assert clock.video_source_created is True
        
    def test_initialize_prevents_reinitialization(self, clock):
        """Test that initialize prevents multiple initializations."""
        seek_pts_1 = 325.0
        seek_pts_2 = 400.0
        
        # First initialization should succeed
        clock.initialize(seek_pts_1)
        assert clock.media_seek_pts == seek_pts_1
        
        # Second initialization should be ignored
        clock.initialize(seek_pts_2)
        assert clock.media_seek_pts == seek_pts_1  # Should remain unchanged
        
    def test_get_video_relative_time_increases_monotonically(self, clock):
        """Test that get_video_relative_time increases monotonically over time."""
        clock.initialize(100.0)
        clock.start_wall_time = time.time()
        
        # Get initial time
        time1 = clock.get_video_relative_time()
        
        # Wait a small amount
        time.sleep(0.01)
        
        # Get second time
        time2 = clock.get_video_relative_time()
        
        # Should be monotonically increasing
        assert time2 > time1
        assert (time2 - time1) >= 0.01  # At least the sleep duration
        
    def test_get_video_relative_time_returns_zero_when_not_started(self, clock):
        """Test that get_video_relative_time returns 0 when not started."""
        clock.initialize(100.0)
        # Don't set start_wall_time
        
        assert clock.get_video_relative_time() == 0.0
        
    def test_get_elapsed_time_resets_after_reset_call(self, clock):
        """Test that get_elapsed_time resets properly after reset."""
        clock.initialize(100.0)
        clock.start_wall_time = time.time()
        
        # Wait and verify time elapsed
        time.sleep(0.01)
        elapsed_before = clock.get_elapsed_time()
        assert elapsed_before > 0
        
        # Reset and verify time is back to 0
        clock.reset()
        elapsed_after = clock.get_elapsed_time()
        assert elapsed_after == 0.0
        
    def test_rel_audio_time_synchronizes_to_reference_clock(self, clock):
        """Test that rel_audio_time correctly converts absolute to relative time."""
        seek_pts = 325.0
        clock.initialize(seek_pts)
        
        # Test various absolute timestamps
        test_cases = [
            (325.0, 0.0),    # Start position
            (359.5, 34.5),   # 34.5 seconds into playback
            (300.0, -25.0),  # Before start (negative relative time)
            (400.0, 75.0),   # 75 seconds into playback
        ]
        
        for abs_pts, expected_rel in test_cases:
            actual_rel = clock.rel_audio_time(abs_pts)
            assert actual_rel == expected_rel, f"For abs_pts={abs_pts}, expected {expected_rel}, got {actual_rel}"
            
    def test_is_initialized_returns_correct_state(self, clock):
        """Test that is_initialized correctly reports initialization state."""
        # Initially not initialized
        assert not clock.is_initialized()
        
        # After initialize() but before start_wall_time
        clock.initialize(100.0)
        assert not clock.is_initialized()  # start_wall_time still None
        
        # After both are set
        clock.start_wall_time = time.time()
        assert clock.is_initialized()
        
        # After reset
        clock.reset()
        assert not clock.is_initialized()
        
    def test_reset_clears_all_state(self, clock):
        """Test that reset completely clears clock state."""
        # Set up clock with all values
        clock.initialize(325.0)
        clock.start_wall_time = time.time()
        
        # Verify initialized state
        assert clock.media_seek_pts == 325.0
        assert clock.video_source_created is True
        assert clock.start_wall_time is not None
        assert clock.is_initialized()
        
        # Reset
        clock.reset()
        
        # Verify all state is cleared
        assert clock.media_seek_pts == 0.0
        assert clock.video_source_created is False
        assert clock.start_wall_time is None
        assert not clock.is_initialized()
        
    def test_wall_time_consistency(self, clock):
        """Test that wall time tracking is consistent."""
        clock.initialize(100.0)
        start_time = time.time()
        clock.start_wall_time = start_time
        
        # Both methods should return similar values
        relative_time = clock.get_video_relative_time()
        elapsed_time = clock.get_elapsed_time()
        
        # They should be essentially the same (within small tolerance)
        assert abs(relative_time - elapsed_time) < 0.001
        
        # Both should be close to actual elapsed time
        actual_elapsed = time.time() - start_time
        assert abs(relative_time - actual_elapsed) < 0.01
        
    @patch('src.clock.logger')
    def test_initialization_logging(self, mock_logger, clock):
        """Test that initialization logs appropriate messages."""
        seek_pts = 325.0
        
        clock.initialize(seek_pts)
        
        # Check that info log was called with correct message
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "clock initialized" in log_message.lower()
        assert f"media_seek_pts={seek_pts:.2f}s" in log_message
        
    @patch('src.clock.logger')
    def test_reinitialization_warning(self, mock_logger, clock):
        """Test that reinitialization attempts log warnings."""
        # First initialization
        clock.initialize(325.0)
        mock_logger.reset_mock()
        
        # Second initialization should log warning
        clock.initialize(400.0)
        
        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "re-initialize" in warning_message.lower()
        assert "ignoring" in warning_message.lower()


class TestGlobalClockSingleton:
    """Test the global CLOCK singleton behavior."""
    
    def setup_method(self):
        """Reset global clock before each test."""
        CLOCK.reset()
        
    def teardown_method(self):
        """Clean up global clock after each test."""
        CLOCK.reset()
        
    def test_global_clock_is_singleton(self):
        """Test that CLOCK is a singleton instance."""
        from src.clock import CLOCK as clock1
        from src.clock import CLOCK as clock2
        
        assert clock1 is clock2
        assert id(clock1) == id(clock2)
        
    def test_global_clock_state_persistence(self):
        """Test that global clock state persists across imports."""
        # Initialize global clock
        CLOCK.initialize(500.0)
        CLOCK.start_wall_time = time.time()
        
        # Import again and verify state persists
        from src.clock import CLOCK as imported_clock
        
        assert imported_clock.media_seek_pts == 500.0
        assert imported_clock.start_wall_time is not None
        assert imported_clock.is_initialized()
        
    def test_global_clock_concurrent_access(self):
        """Test that global clock handles concurrent access safely."""
        import threading
        
        results = []
        
        def worker(seek_pts):
            CLOCK.initialize(seek_pts)
            results.append(CLOCK.media_seek_pts)
            
        # Start multiple threads trying to initialize
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(100.0 + i,))
            threads.append(t)
            t.start()
            
        # Wait for all threads
        for t in threads:
            t.join()
            
        # All results should be the same (first one wins)
        assert all(result == results[0] for result in results)
        assert CLOCK.media_seek_pts == results[0] 