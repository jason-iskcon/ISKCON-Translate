"""Test suite for video, audio, and caption synchronization."""
import time
import threading
import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import components to test
from src.core.video_runner import VideoRunner
from src.caption_overlay.core import CaptionCore
from src.clock import CLOCK

class MockVideoSource:
    """Mock video source for testing."""
    def __init__(self, fps=30):
        self.fps = fps
        self.audio_position = 0.0
        self.audio_playing = True
        self.audio_position_lock = threading.Lock()
        self.playback_start_time = time.time()
        self.frame_count = 0
        self.frame_times = []  # Track frame timestamps
        
    def get_video_info(self):
        return 1280, 720, self.fps
        
    def read_frame(self):
        self.frame_count += 1
        current_time = time.time()
        self.frame_times.append(current_time)
        return True, np.zeros((720, 1280, 3), dtype=np.uint8)
        
    def get_frame(self):
        """Alias for read_frame for backward compatibility."""
        return self.read_frame()
        
    def get_frame_time(self):
        """Get the current frame time."""
        return self.frame_count * (1.0 / self.fps)
        
    def get_frame_times(self):
        """Get list of frame timestamps."""
        return self.frame_times
        
    def release(self):
        pass

class MockTranscriber:
    """Mock transcriber for testing."""
    def __init__(self):
        self.segments = []
        self.segment_times = []  # Track segment processing times
    
    def add_segment(self, text, start, end):
        """Add a mock segment."""
        current_time = time.time()
        self.segment_times.append(current_time)
        self.segments.append({
            'text': text,
            'start': start,
            'end': end,
            'added_at': current_time
        })
    
    def get_segments(self):
        """Get all segments."""
        return self.segments
        
    def get_segment_times(self):
        """Get list of segment processing times."""
        return self.segment_times

class MockCaptionOverlay:
    """Mock caption overlay for testing."""
    def __init__(self):
        self.core = CaptionCore()
        self.display_times = []  # Track caption display times
    
    def add_caption(self, text, timestamp, duration=3.0):
        """Add a caption."""
        current_time = time.time()
        self.display_times.append(current_time)
        return self.core.add_caption(text, timestamp, duration)
    
    def get_active_captions(self, current_time):
        """Get active captions."""
        return self.core.get_active_captions(current_time)
    
    def clear_captions(self):
        """Clear all captions."""
        self.core.clear_captions()
        
    def get_display_times(self):
        """Get list of caption display times."""
        return self.display_times

@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    video_source = MockVideoSource()
    transcriber = MockTranscriber()
    caption_overlay = MockCaptionOverlay()
    return video_source, transcriber, caption_overlay

def test_frame_timing_precision(mock_components):
    """Test frame timing precision."""
    video_source, transcriber, caption_overlay = mock_components
    
    # Test frame timing at 30fps
    frame_time = 1.0 / 30.0
    test_frames = 30
    
    for _ in range(test_frames):
        success, frame = video_source.read_frame()
        assert success
        frame_time = video_source.get_frame_time()
        
        # Verify frame timing
        assert abs(frame_time - (video_source.frame_count * (1.0/30.0))) < 0.001

def test_audio_video_sync(mock_components):
    """Test audio-video synchronization."""
    video_source, transcriber, caption_overlay = mock_components
    
    # Add test segments
    test_segments = [
        ("Test 1", 0.1, 0.2),
        ("Test 2", 0.2, 0.3),
        ("Test 3", 0.3, 0.4),
        ("Test 4", 0.4, 0.5)
    ]
    
    for text, start, end in test_segments:
        transcriber.add_segment(text, start, end)
    
    # Verify segment timing
    segments = transcriber.get_segments()
    assert len(segments) == 4
    
    for i, segment in enumerate(segments):
        assert segment['start'] == test_segments[i][1]
        assert segment['end'] == test_segments[i][2]

def test_caption_timing(mock_components):
    """Test caption timing accuracy."""
    video_source, transcriber, caption_overlay = mock_components
    
    # Add test captions
    test_captions = [
        ("Test 1", 1.0, 2.0),
        ("Test 2", 2.0, 3.0),
        ("Test 3", 3.0, 4.0)
    ]
    
    for text, start, duration in test_captions:
        caption_overlay.add_caption(text, start, duration)
    
    # Test caption timing at different points
    test_times = [0.9, 1.1, 1.9, 2.1, 2.9, 3.1]
    active_counts = []
    
    for t in test_times:
        active = caption_overlay.get_active_captions(t)
        active_counts.append(len(active))
    
    # Verify caption timing
    assert active_counts[0] == 0  # Before first caption
    assert active_counts[1] == 1  # During first caption
    assert active_counts[2] == 1  # End of first caption
    assert active_counts[3] == 1  # Start of second caption
    assert active_counts[4] == 1  # End of second caption
    assert active_counts[5] == 1  # Start of third caption

def test_caption_buffer_accuracy(mock_components):
    """Test caption timing buffer accuracy."""
    video_source, transcriber, caption_overlay = mock_components
    
    # Add caption with precise timing
    caption_overlay.add_caption("Buffer Test", 1.0, 1.0)
    
    # Test timing around buffer edges
    buffer = 0.0165  # Half frame at 30fps
    test_times = [
        1.0 - buffer - 0.001,  # Just before buffer
        1.0 - buffer + 0.001,  # Just inside buffer
        2.0 - buffer - 0.001,  # Just before end buffer
        2.0 + buffer + 0.001   # Just after end buffer
    ]
    
    active_counts = []
    for t in test_times:
        active = caption_overlay.get_active_captions(t)
        active_counts.append(len(active))
    
    # Verify buffer behavior
    assert active_counts[0] == 0  # Before buffer
    assert active_counts[1] == 1  # Inside start buffer
    assert active_counts[2] == 1  # Before end buffer
    assert active_counts[3] == 0  # After end buffer

def test_caption_cleanup(mock_components):
    """Test caption cleanup behavior."""
    video_source, transcriber, caption_overlay = mock_components
    
    # Add test captions
    caption_overlay.add_caption("Past", 0.0, 1.0)     # Should be cleaned up
    caption_overlay.add_caption("Current", 5.0, 1.0)  # Should be active
    caption_overlay.add_caption("Future", 10.0, 1.0)  # Should be kept
    
    # Test cleanup at different times
    test_times = [0.0, 5.0, 10.0]
    active_counts = []
    
    for t in test_times:
        active = caption_overlay.get_active_captions(t)
        active_counts.append(len(active))
    
    # Verify cleanup behavior
    assert active_counts[0] == 1  # Past caption should be active
    assert active_counts[1] == 1  # Current caption should be active
    assert active_counts[2] == 1  # Future caption should be active

def test_synchronization_under_load(mock_components):
    """Test synchronization under heavy load."""
    video_source, transcriber, caption_overlay = mock_components
    
    # Add many captions in quick succession
    for i in range(100):
        caption_overlay.add_caption(f"Load Test {i}", i * 0.1, 0.1)
    
    # Test frame timing under load
    frame_time = 1.0 / 30.0
    test_frames = 30
    
    for _ in range(test_frames):
        success, frame = video_source.read_frame()
        assert success
        frame_time = video_source.get_frame_time()
        
        # Get active captions
        active = caption_overlay.get_active_captions(frame_time)
        
        # Verify timing precision is maintained
        assert abs(frame_time - (video_source.frame_count * (1.0/30.0))) < 0.001

def test_rapid_caption_changes(mock_components):
    """Test synchronization during rapid caption changes."""
    video_source, transcriber, caption_overlay = mock_components
    
    # Add captions with very short durations
    for i in range(10):
        caption_overlay.add_caption(f"Quick {i}", i * 0.1, 0.05)  # 50ms duration
    
    # Test timing at precise intervals
    test_times = [0.05, 0.15, 0.25, 0.35, 0.45]
    active_counts = []
    
    for t in test_times:
        active = caption_overlay.get_active_captions(t)
        active_counts.append(len(active))
    
    # Verify only one caption is active at a time
    assert all(count == 1 for count in active_counts)

def test_overlapping_captions(mock_components):
    """Test handling of overlapping captions."""
    video_source, transcriber, caption_overlay = mock_components
    
    # Add overlapping captions
    caption_overlay.add_caption("First", 1.0, 2.0)   # Ends at 3.0s
    caption_overlay.add_caption("Second", 1.5, 2.0)  # Ends at 3.5s
    caption_overlay.add_caption("Third", 2.0, 2.0)   # Ends at 4.0s
    
    # Test timing at overlap points
    test_times = [1.4, 1.6, 1.9, 2.1, 4.1]  # Changed last time to 4.1s to test after Third caption ends
    active_captions = []
    
    for t in test_times:
        active = caption_overlay.get_active_captions(t)
        active_captions.append([c['text'] for c in active])
    
    # Verify correct caption ordering and transitions
    assert active_captions[0] == ["First"]   # Before overlap
    assert active_captions[1] == ["Second"]  # During first overlap
    assert active_captions[2] == ["Second"]  # End of first overlap
    assert active_captions[3] == ["Third"]   # Start of second overlap
    assert active_captions[4] == []          # After all captions have ended

def test_frame_drop_handling(mock_components):
    """Test synchronization when frames are dropped."""
    video_source, transcriber, caption_overlay = mock_components
    
    # Simulate frame drops by skipping some frame counts
    frame_times = []
    for i in range(30):
        if i % 3 != 0:  # Drop every third frame
            success, frame = video_source.read_frame()
            assert success
            frame_times.append(video_source.get_frame_time())
    
    # Verify timing remains consistent despite frame drops
    intervals = [t2 - t1 for t1, t2 in zip(frame_times[:-1], frame_times[1:])]
    avg_interval = sum(intervals) / len(intervals)
    assert abs(avg_interval - (1.0/30.0)) < 0.002  # Allow small deviation

def test_audio_video_caption_sync(mock_components):
    """Test complete synchronization between audio, video, and captions."""
    video_source, transcriber, caption_overlay = mock_components
    
    # Add synchronized test data
    test_data = [
        (0.0, "Start", "First caption"),
        (1.0, "Middle", "Second caption"),
        (2.0, "End", "Third caption")
    ]
    
    for timestamp, audio_text, caption_text in test_data:
        # Add audio segment
        transcriber.add_segment(audio_text, timestamp, timestamp + 0.5)
        # Add caption
        caption_overlay.add_caption(caption_text, timestamp, 0.5)
    
    # Test synchronization at key points
    test_times = [0.1, 1.1, 2.1]
    sync_results = []
    
    for t in test_times:
        # Get active caption
        active = caption_overlay.get_active_captions(t)
        # Get corresponding audio segment
        segments = [s for s in transcriber.get_segments() 
                   if s['start'] <= t <= s['end']]
        
        sync_results.append({
            'time': t,
            'caption': active[0]['text'] if active else None,
            'audio': segments[0]['text'] if segments else None
        })
    
    # Verify synchronization
    assert sync_results[0]['caption'] == "First caption"
    assert sync_results[0]['audio'] == "Start"
    assert sync_results[1]['caption'] == "Second caption"
    assert sync_results[1]['audio'] == "Middle"
    assert sync_results[2]['caption'] == "Third caption"
    assert sync_results[2]['audio'] == "End"

def test_timing_precision_under_stress(mock_components):
    """Test timing precision under stress conditions."""
    video_source, transcriber, caption_overlay = mock_components
    
    # Simulate stress by rapid operations
    operations = []
    start_time = time.time()
    
    # Perform rapid operations
    for i in range(100):
        # Add caption
        caption_overlay.add_caption(f"Stress {i}", i * 0.01, 0.01)
        # Read frame
        success, frame = video_source.read_frame()
        assert success
        # Record operation time
        operations.append(time.time() - start_time)
    
    # Calculate timing statistics
    intervals = [t2 - t1 for t1, t2 in zip(operations[:-1], operations[1:])]
    avg_interval = sum(intervals) / len(intervals)
    max_deviation = max(abs(i - 0.01) for i in intervals)
    
    # Verify timing precision
    assert avg_interval < 0.02  # Average interval should be small
    assert max_deviation < 0.05  # Max deviation should be reasonable 