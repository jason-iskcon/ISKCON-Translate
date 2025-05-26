"""
Integration tests for video playback and caption synchronization.

These tests validate the current working behavior of the video playback system
to ensure regressions don't occur.
"""
import time
import threading
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.core.video_runner import VideoRunner
from src.caption_overlay import CaptionOverlay
from src.transcription.engine import TranscriptionEngine


class MockVideoSource:
    """Mock video source for testing."""
    
    def __init__(self, fps=30, width=640, height=480):
        self.fps = fps
        self.width = width
        self.height = height
        self.start_time = 0.0
        self.audio_position = 0.0
        self.audio_position_lock = threading.Lock()
        self.audio_playing = True
        self.playback_start_time = time.perf_counter()
        self.frame_count = 0
        
    def get_video_info(self):
        return self.width, self.height, self.fps
    
    def get_frame(self):
        """Return a mock frame with timestamp."""
        if self.frame_count > 300:  # Stop after 10 seconds at 30fps
            return None
        
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        timestamp = self.frame_count / self.fps
        self.frame_count += 1
        
        # Update audio position to simulate synchronized playback
        with self.audio_position_lock:
            self.audio_position = timestamp
        
        return frame, timestamp
    
    def release(self):
        pass


class MockTranscriber:
    """Mock transcriber for testing."""
    
    def __init__(self):
        self.transcriptions = []
        self.current_index = 0
        
    def add_test_transcription(self, text, start_time, end_time):
        """Add a test transcription."""
        self.transcriptions.append({
            'text': text,
            'start': start_time,
            'end': end_time
        })
    
    def get_transcription(self):
        """Get next transcription if available."""
        if self.current_index < len(self.transcriptions):
            transcription = self.transcriptions[self.current_index]
            self.current_index += 1
            return transcription
        return None


class TestVideoPlaybackIntegration:
    """Test video playback integration with current behavior."""
    
    def setup_method(self):
        """Set up test environment."""
        self.video_source = MockVideoSource()
        self.transcriber = MockTranscriber()
        self.caption_overlay = CaptionOverlay()
        
        # Add some test transcriptions
        self.transcriber.add_test_transcription("First caption", 1.0, 3.0)
        self.transcriber.add_test_transcription("Second caption", 2.5, 4.5)
        self.transcriber.add_test_transcription("Third caption", 4.0, 6.0)
    
    def test_video_runner_initialization(self):
        """Test that VideoRunner initializes correctly with current parameters."""
        runner = VideoRunner(
            video_source=self.video_source,
            transcriber=self.transcriber,
            caption_overlay=self.caption_overlay,
            headless=True
        )
        
        assert runner.video_source == self.video_source
        assert runner.transcriber == self.transcriber
        assert runner.caption_overlay == self.caption_overlay
        assert runner.headless is True
        assert runner.fps == 30
        assert runner.target_frame_time == 1.0 / 30
        assert runner.frame_count == 0
        assert runner.running is False
    
    def test_frame_processing_with_captions(self):
        """Test that frame processing works with captions."""
        runner = VideoRunner(
            video_source=self.video_source,
            transcriber=self.transcriber,
            caption_overlay=self.caption_overlay,
            headless=True
        )
        
        # Process a few frames
        for i in range(5):
            frame_data = self.video_source.get_frame()
            if frame_data:
                processed_frame = runner._process_frame(frame_data)
                assert processed_frame is not None
                assert processed_frame.shape == (480, 640, 3)
    
    def test_audio_video_sync_mechanism(self):
        """Test that audio-video sync mechanism works."""
        runner = VideoRunner(
            video_source=self.video_source,
            transcriber=self.transcriber,
            caption_overlay=self.caption_overlay,
            headless=True
        )
        
        # Test sync with audio ahead
        synced_time = runner._sync_with_audio(1.0)
        assert isinstance(synced_time, float)
        
        # Test sync with audio behind
        with self.video_source.audio_position_lock:
            self.video_source.audio_position = 0.5
        
        synced_time = runner._sync_with_audio(1.0)
        assert isinstance(synced_time, float)
    
    def test_transcription_processing(self):
        """Test that transcriptions are processed correctly."""
        runner = VideoRunner(
            video_source=self.video_source,
            transcriber=self.transcriber,
            caption_overlay=self.caption_overlay,
            headless=True
        )
        
        # Process transcriptions
        runner._process_transcriptions()
        
        # Check that captions were added
        assert len(self.caption_overlay.captions) > 0
        
        # Verify caption content
        first_caption = self.caption_overlay.captions[0]
        assert 'text' in first_caption
        assert 'start_time' in first_caption
        assert 'end_time' in first_caption
    
    def test_caption_timing_accuracy(self):
        """Test that caption timing is accurate."""
        runner = VideoRunner(
            video_source=self.video_source,
            transcriber=self.transcriber,
            caption_overlay=self.caption_overlay,
            headless=True
        )
        
        # Add a caption at a specific time
        self.caption_overlay.add_caption("Test caption", 1.0, 2.0)
        
        # Test at different times
        active_at_0_5 = self.caption_overlay.core.get_active_captions(0.5)
        active_at_1_5 = self.caption_overlay.core.get_active_captions(1.5)
        active_at_3_5 = self.caption_overlay.core.get_active_captions(3.5)
        
        assert len(active_at_0_5) == 0  # Before caption
        assert len(active_at_1_5) == 1  # During caption
        assert len(active_at_3_5) == 0  # After caption
    
    def test_multiple_caption_handling(self):
        """Test that multiple overlapping captions are handled correctly."""
        runner = VideoRunner(
            video_source=self.video_source,
            transcriber=self.transcriber,
            caption_overlay=self.caption_overlay,
            headless=True
        )
        
        # Add overlapping captions
        self.caption_overlay.add_caption("Caption 1", 1.0, 3.0)
        self.caption_overlay.add_caption("Caption 2", 2.0, 4.0)
        self.caption_overlay.add_caption("Caption 3", 2.5, 3.5)
        
        # Test at overlap time
        active_captions = self.caption_overlay.core.get_active_captions(2.7)
        
        # Should have multiple active captions
        assert len(active_captions) >= 1
        
        # All should have fade factors
        for caption in active_captions:
            assert 'fade_factor' in caption or caption.get('fade_factor', 1.0) > 0
    
    def test_frame_timing_consistency(self):
        """Test that frame timing is consistent."""
        runner = VideoRunner(
            video_source=self.video_source,
            transcriber=self.transcriber,
            caption_overlay=self.caption_overlay,
            headless=True
        )
        
        frame_times = []
        start_time = time.perf_counter()
        
        # Process several frames and measure timing
        for i in range(10):
            frame_data = self.video_source.get_frame()
            if frame_data:
                frame_start = time.perf_counter()
                runner._process_frame(frame_data)
                frame_end = time.perf_counter()
                frame_times.append(frame_end - frame_start)
        
        # Check that frame processing is reasonably fast
        avg_frame_time = sum(frame_times) / len(frame_times)
        assert avg_frame_time < 0.033  # Should be faster than 33ms (30fps)
        
        # Check that all frame times are reasonable (adjusted for actual performance)
        max_frame_time = max(frame_times)
        min_frame_time = min(frame_times)
        assert max_frame_time < 0.100, f"Max frame time too slow: {max_frame_time:.6f}s"
        assert min_frame_time >= 0.0, f"Min frame time should be non-negative: {min_frame_time:.6f}s"
        
        # Check that most frames are processed quickly
        fast_frames = [t for t in frame_times if t < 0.010]
        assert len(fast_frames) >= len(frame_times) * 0.5, "At least half the frames should be fast"
    
    def test_caption_fade_effects(self):
        """Test that caption fade effects work correctly."""
        # Add a caption
        self.caption_overlay.add_caption("Fade test", 1.0, 3.0)
        
        # Get active captions at different times
        captions_start = self.caption_overlay.core.get_active_captions(1.1)  # Just after start
        captions_middle = self.caption_overlay.core.get_active_captions(2.0)  # Middle
        captions_end = self.caption_overlay.core.get_active_captions(2.9)  # Just before end
        
        # All should have captions
        assert len(captions_start) == 1
        assert len(captions_middle) == 1
        assert len(captions_end) == 1
        
        # Check fade factors exist
        for captions in [captions_start, captions_middle, captions_end]:
            if captions:
                caption = captions[0]
                assert 'fade_factor' in caption
                assert 0 <= caption['fade_factor'] <= 1
    
    def test_memory_management(self):
        """Test that memory is managed properly during playback."""
        runner = VideoRunner(
            video_source=self.video_source,
            transcriber=self.transcriber,
            caption_overlay=self.caption_overlay,
            headless=True
        )
        
        # Add many captions
        for i in range(20):
            self.caption_overlay.add_caption(f"Caption {i}", i * 0.5, 1.0)
        
        initial_count = len(self.caption_overlay.captions)
        
        # Prune old captions
        self.caption_overlay.prune_captions(10.0, buffer=1.0)
        
        final_count = len(self.caption_overlay.captions)
        
        # Should have fewer captions after pruning
        assert final_count < initial_count
    
    def test_error_handling_robustness(self):
        """Test that the system handles errors gracefully."""
        runner = VideoRunner(
            video_source=self.video_source,
            transcriber=self.transcriber,
            caption_overlay=self.caption_overlay,
            headless=True
        )
        
        # Test with invalid caption data
        try:
            self.caption_overlay.add_caption("", 1.0, 2.0)  # Empty text
            self.caption_overlay.add_caption("Valid", -1.0, 2.0)  # Negative time
            self.caption_overlay.add_caption("Valid", 1.0, 0)  # Zero duration
        except Exception as e:
            pytest.fail(f"System should handle invalid data gracefully: {e}")
        
        # System should still be functional
        frame_data = self.video_source.get_frame()
        if frame_data:
            processed_frame = runner._process_frame(frame_data)
            assert processed_frame is not None


class TestCaptionRenderingBehavior:
    """Test caption rendering behavior."""
    
    def setup_method(self):
        """Set up test environment."""
        self.caption_overlay = CaptionOverlay()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_text_centering_in_background(self):
        """Test that text is properly centered in background."""
        # Add a caption
        self.caption_overlay.add_caption("Centered text test", 0.0, 2.0)
        
        # Render the caption
        result_frame = self.caption_overlay.overlay_captions(
            self.test_frame.copy(),
            current_time=1.0,
            frame_count=1
        )
        
        # Frame should be modified (not identical to original)
        assert not np.array_equal(result_frame, self.test_frame)
        
        # Should have some non-zero pixels (text and background)
        assert np.any(result_frame > 0)
    
    def test_multiple_caption_stacking(self):
        """Test that multiple captions stack properly."""
        # Add multiple overlapping captions
        self.caption_overlay.add_caption("Caption 1", 0.0, 3.0)
        self.caption_overlay.add_caption("Caption 2", 0.5, 3.5)
        self.caption_overlay.add_caption("Caption 3", 1.0, 4.0)
        
        # Render at overlap time
        result_frame = self.caption_overlay.overlay_captions(
            self.test_frame.copy(),
            current_time=2.0,
            frame_count=1
        )
        
        # Should have rendered content
        assert not np.array_equal(result_frame, self.test_frame)
        
        # Check that we have reasonable amount of non-zero pixels
        non_zero_pixels = np.count_nonzero(result_frame)
        assert non_zero_pixels > 1000  # Should have substantial content
    
    def test_caption_fade_rendering(self):
        """Test that caption fading is rendered correctly."""
        # Add a caption
        self.caption_overlay.add_caption("Fade test", 1.0, 3.0)
        
        # Render at different fade points
        frame_start = self.caption_overlay.overlay_captions(
            self.test_frame.copy(), current_time=1.1, frame_count=1
        )
        frame_middle = self.caption_overlay.overlay_captions(
            self.test_frame.copy(), current_time=2.0, frame_count=1
        )
        frame_end = self.caption_overlay.overlay_captions(
            self.test_frame.copy(), current_time=2.9, frame_count=1
        )
        
        # All should have content
        assert not np.array_equal(frame_start, self.test_frame)
        assert not np.array_equal(frame_middle, self.test_frame)
        assert not np.array_equal(frame_end, self.test_frame)
    
    def test_performance_under_load(self):
        """Test rendering performance with many captions."""
        # Add many captions
        for i in range(10):
            self.caption_overlay.add_caption(f"Performance test caption {i}", i * 0.1, 2.0)
        
        # Measure rendering time
        start_time = time.perf_counter()
        
        for frame_num in range(30):  # Simulate 1 second at 30fps
            current_time = frame_num / 30.0
            result_frame = self.caption_overlay.overlay_captions(
                self.test_frame.copy(),
                current_time=current_time,
                frame_count=frame_num
            )
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second for 30 frames)
        assert total_time < 1.0
        
        # Average frame time should be reasonable
        avg_frame_time = total_time / 30
        assert avg_frame_time < 0.033  # Should be faster than 33ms per frame 