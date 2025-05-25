"""Shared test configuration and fixtures for ISKCON-Translate test suite."""

import os
import sys
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import cv2

# Add src directory to Python path for all tests
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide a temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp(prefix="iskcon_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.debug = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.trace = MagicMock()
    return logger


@pytest.fixture
def mock_video_frame():
    """Provide a mock video frame for testing."""
    # Create a simple test frame (blue background with white text)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = [255, 0, 0]  # Blue background
    
    # Add some text
    cv2.putText(frame, "Test Frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                2, (255, 255, 255), 3, cv2.LINE_AA)
    
    return frame


@pytest.fixture
def mock_audio_data():
    """Provide mock audio data for testing."""
    # Generate 1 second of sine wave at 440Hz (A note)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    return audio.astype(np.float32)


@pytest.fixture
def mock_transcription_result():
    """Provide a mock transcription result."""
    return {
        'text': 'This is a test transcription',
        'start': 0.0,
        'end': 3.0,
        'confidence': 0.95,
        'words': [
            {'word': 'This', 'start': 0.0, 'end': 0.3, 'probability': 0.98},
            {'word': 'is', 'start': 0.3, 'end': 0.5, 'probability': 0.97},
            {'word': 'a', 'start': 0.5, 'end': 0.6, 'probability': 0.95},
            {'word': 'test', 'start': 0.6, 'end': 1.0, 'probability': 0.99},
            {'word': 'transcription', 'start': 1.0, 'end': 3.0, 'probability': 0.94}
        ],
        'processing_time': 0.5,
        'worker': 'TestWorker'
    }


@pytest.fixture
def mock_whisper_model():
    """Provide a mock Whisper model for testing."""
    model = MagicMock()
    
    # Mock transcribe method
    mock_segment = MagicMock()
    mock_segment.text = "Test transcription"
    mock_segment.start = 0.0
    mock_segment.end = 3.0
    mock_segment.avg_logprob = -0.5
    mock_segment.words = []
    
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95
    
    model.transcribe.return_value = ([mock_segment], mock_info)
    
    return model


@pytest.fixture
def mock_video_source():
    """Provide a comprehensive mock VideoSource for testing."""
    video_source = MagicMock()
    
    # Video properties
    video_source.get_video_info.return_value = (1920, 1080, 30.0)
    video_source.is_running = True
    video_source.playback_start_time = 1000.0
    video_source.audio_playing = False
    video_source.audio_position = 0.0
    
    # Queue properties
    video_source.frames_queue.qsize.return_value = 10
    video_source.frames_queue.maxsize = 100
    
    # Frame generation
    def mock_get_frame():
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        return (frame, 1.0)  # frame, timestamp
    
    video_source.get_frame = mock_get_frame
    
    # Context manager support
    video_source.__enter__ = MagicMock(return_value=video_source)
    video_source.__exit__ = MagicMock(return_value=None)
    
    return video_source


@pytest.fixture
def mock_transcription_engine():
    """Provide a comprehensive mock TranscriptionEngine for testing."""
    engine = MagicMock()
    
    # Queue properties
    engine.audio_queue.qsize.return_value = 5
    engine.audio_queue.maxsize = 50
    
    # Transcription methods
    engine.get_transcription.return_value = None
    engine.start_transcription.return_value = None
    engine.stop_transcription.return_value = None
    
    # Properties
    engine.playback_start_time = None
    engine.caption_overlay = None
    
    # Context manager support
    engine.__enter__ = MagicMock(return_value=engine)
    engine.__exit__ = MagicMock(return_value=None)
    
    return engine


@pytest.fixture
def mock_caption_overlay():
    """Provide a comprehensive mock CaptionOverlay for testing."""
    overlay = MagicMock()
    
    # Properties
    overlay.captions = []
    overlay.font_scale = 1.5
    overlay.font_thickness = 2
    overlay.font_color = (255, 255, 255)
    
    # Methods
    overlay.add_caption.return_value = None
    overlay.overlay_captions.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    overlay.set_video_start_time.return_value = None
    
    # Context manager support
    overlay.__enter__ = MagicMock(return_value=overlay)
    overlay.__exit__ = MagicMock(return_value=None)
    
    return overlay


@pytest.fixture
def mock_clock():
    """Provide a mock singleton clock for testing."""
    clock = MagicMock()
    clock.start_wall_time = None
    clock.media_seek_pts = 0.0
    clock.get_elapsed_time.return_value = 1.0
    clock.get_video_relative_time.return_value = 1.0
    clock.is_initialized.return_value = True
    clock.reset.return_value = None
    return clock


@pytest.fixture
def disable_logging():
    """Disable logging for tests that don't need it."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def temp_video_file(test_data_dir):
    """Create a temporary test video file."""
    video_path = test_data_dir / "test_video.mp4"
    
    # Create a simple test video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    # Write 30 frames (1 second at 30fps)
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [100 + i * 5, 50, 200 - i * 3]  # Gradually changing colors
        
        # Add frame number text
        cv2.putText(frame, f"Frame {i}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
    
    out.release()
    
    yield video_path
    
    # Cleanup handled by test_data_dir fixture


@pytest.fixture
def environment_variables():
    """Provide controlled environment variables for testing."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        'GIT_COMMIT': 'test-commit-abc123',
        'LOG_LEVEL': 'DEBUG',
        'PYTHONPATH': src_path
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def performance_counters():
    """Provide mock performance counters for testing."""
    counters = {
        'processed_chunks': MagicMock(),
        'avg_time_tracker': MagicMock()
    }
    
    # Set up counter behavior
    counters['processed_chunks'].value = 0
    counters['processed_chunks'].get_lock.return_value.__enter__ = MagicMock()
    counters['processed_chunks'].get_lock.return_value.__exit__ = MagicMock()
    
    counters['avg_time_tracker'].value = 0.0
    counters['avg_time_tracker'].get_lock.return_value.__enter__ = MagicMock()
    counters['avg_time_tracker'].get_lock.return_value.__exit__ = MagicMock()
    
    return counters


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "network: mark test as requiring network access")


# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to all tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that take longer
        if "performance" in item.name or "stress" in item.name:
            item.add_marker(pytest.mark.slow)


# Cleanup fixture for global state
@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Automatically clean up global state after each test."""
    yield
    
    # Reset any global state if needed
    # This is particularly important for singleton patterns like CLOCK
    try:
        from src.clock import CLOCK
        CLOCK.reset()
    except ImportError:
        pass  # Clock not available in this test
    
    # Clear any OpenCV windows that might be open
    try:
        cv2.destroyAllWindows()
    except:
        pass 