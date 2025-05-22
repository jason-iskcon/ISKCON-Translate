"""
System tests for video source loading and logging.

These tests verify that the VideoSource class correctly loads videos
and produces the expected log output at different log levels.
"""
import os
import time
import shutil
import pytest
import cv2
import logging
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.video_source import VideoSource

# Add src to path so we can import from it
import sys
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import test utilities
sys.path.append(str(Path(__file__).parent.parent))
from test_utils import TestVideo

from src.video_source import VideoSource
from src.logging_utils import TRACE, get_logger

# Configure test logger
test_logger = get_logger('test_video_loading')

class TestVideoSourceLoading:
    """Test video source loading functionality and logging."""
    
    @classmethod
    def create_test_video(cls, duration=2.0, fps=30, width=640, height=480, 
                         output_format='mp4', codec='libx264'):
        """Create a test video file with the specified parameters.
        
        Args:
            duration: Video duration in seconds
            fps: Frames per second
            width: Video width in pixels
            height: Video height in pixels
            output_format: Output format (mp4, avi, mkv)
            codec: Video codec to use
            
        Returns:
            Path to the created video file
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'test_data')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        output_file = os.path.join(
            output_dir, 
            f'test_video_{width}x{height}_{fps}fps_{duration}s_{codec}.{output_format}'
        )
        
        # If file already exists, return the path
        if os.path.exists(output_file):
            return output_file
            
        # Create a simple video with FFmpeg
        try:
            import subprocess
            
            # FFmpeg command to generate color bars test pattern
            cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'color=c=red:s={width}x{height}:r={fps}:d={duration}',
                '-c:v', codec,
                '-pix_fmt', 'yuv420p',
                '-y',  # Overwrite output file if it exists
                output_file
            ]
            
            # Run FFmpeg command
            subprocess.run(cmd, 
                         check=True, 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
            
            return output_file
            
        except Exception as e:
            test_logger.error(f"Failed to create test video: {e}")
            raise
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before any tests are run."""
        # Create a test video file
        cls.test_video = cls.create_test_video()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files after all tests are done."""
        # Clean up test video files
        test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'test_data')
        if os.path.exists(test_data_dir):
            for file in os.listdir(test_data_dir):
                if file.startswith('test_video_') and (file.endswith('.mp4') or 
                                                     file.endswith('.avi') or 
                                                     file.endswith('.mkv')):
                    try:
                        os.remove(os.path.join(test_data_dir, file))
                    except Exception as e:
                        test_logger.warning(f"Failed to remove test video {file}: {e}")
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path):
        """Setup and teardown for each test."""
        # Create a test video for this test session
        with TestVideo(tmp_path, "test_video.mp4", with_audio=True, duration=2.0, fps=30) as test_video_path:
            self.test_video = str(test_video_path)
            self.log_file = tmp_path / 'video_test.log'
            self.vs = None
            yield
            # Cleanup - VideoSource uses a context manager for cleanup
            if self.vs and hasattr(self.vs, '_cap') and self.vs._cap.isOpened():
                self.vs._cap.release()
    
    def test_video_loads_correctly(self):
        """Test that a video loads correctly and basic properties are set."""
        self.vs = VideoSource(self.test_video)
        self.vs.start()  # Initialize the video capture
        
        # Check if video capture was initialized
        assert hasattr(self.vs, '_cap'), "VideoCapture not initialized"
        assert self.vs._cap.isOpened(), "Video source failed to open"
        
        # Get video info
        width, height, fps = self.vs.get_video_info()
        
        # Basic validation of video properties
        assert fps > 0, f"Invalid FPS: {fps}"
        assert width > 0 and height > 0, f"Invalid frame size: {width}x{height}"
    
    def test_video_loading_logs(self, caplog):
        """Test that video loading produces the expected log messages."""
        with caplog.at_level(logging.INFO):
            self.vs = VideoSource(self.test_video)
            self.vs.start()  # Initialize the video capture
            
            # Check for expected log messages
            log_text = caplog.text
            assert "Initializing VideoSource with source:" in log_text, "Missing initialization log"
            assert "Opening video source:" in log_text, "Missing video source opening log"
    
    def test_video_properties_logged_at_debug(self, caplog):
        """Test that video properties are logged at DEBUG level."""
        with caplog.at_level(logging.DEBUG):
            self.vs = VideoSource(self.test_video)
            self.vs.start()  # Initialize the video capture
            
            log_text = caplog.text
            assert "Video properties - Resolution:" in log_text, "Missing video properties log"
            assert "FPS:" in log_text, "Missing FPS in logs"
            assert "Duration:" in log_text, "Missing duration in logs"
    
    @pytest.mark.parametrize("format_info", [
        {"format": "mp4", "codec": "libx264"},
        {"format": "avi", "codec": "mpeg4"},
        {"format": "mkv", "codec": "libx265"}
    ])
    def test_video_loading(self, caplog, format_info):
        """Test video loading with different formats and validate properties."""
        # Skip if FFmpeg is not available for format conversion
        if not shutil.which('ffmpeg'):
            pytest.skip("FFmpeg not available for format conversion")
            
        # Create test video in the specified format
        test_video = self.create_test_video(
            duration=2.0,
            fps=30,
            width=640,
            height=480,
            output_format=format_info["format"],
            codec=format_info["codec"]
        )
        self.test_video = test_video  # Make available for cleanup
        # Log the test video path and format for debugging
        test_logger.info(f"Testing {format_info['format'].upper()} format with {format_info['codec']} codec")
        test_logger.info(f"Test video path: {test_video}")
        
        # Verify the test video exists and has content
        assert os.path.exists(test_video), f"Test video not found at {test_video}"
        video_size = os.path.getsize(test_video)
        assert video_size > 0, f"Test video is empty: {test_video}"
        test_logger.info(f"Video size: {video_size / (1024*1024):.2f} MB")
        
        # Get video properties using OpenCV for validation
        cap = cv2.VideoCapture(test_video)
        expected_fps = cap.get(cv2.CAP_PROP_FPS)
        expected_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        expected_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_duration = expected_frame_count / expected_fps if expected_fps > 0 else 0
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = ''.join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()
        
        test_logger.info(
            f"Video properties - Resolution: {expected_width}x{expected_height}, "
            f"FPS: {expected_fps:.2f}, Frames: {expected_frame_count}, "
            f"Duration: {expected_duration:.2f}s, Codec: {codec_str}"
        )
        
        # Validate basic properties
        assert expected_width == 640, f"Unexpected width: {expected_width}"
        assert expected_height == 480, f"Unexpected height: {expected_height}"
        assert abs(expected_fps - 30) < 0.1, f"Unexpected FPS: {expected_fps}"
        assert expected_frame_count > 0, "No frames in video"
        
        # Skip test if video is too short
        if expected_frame_count < 10:
            pytest.skip("Test video is too short for accurate frame counting")
        
        # First, try to read the video directly with OpenCV
        cap = cv2.VideoCapture(test_video)
        assert cap.isOpened(), f"Failed to open {test_video} directly with OpenCV"
        
        # Read a few frames directly
        direct_frames = 0
        for _ in range(10):  # Try to read 10 frames
            ret, frame = cap.read()
            if not ret:
                break
            direct_frames += 1
            assert frame is not None, "Direct frame read returned None"
            assert frame.shape[0] > 0 and frame.shape[1] > 0, f"Invalid frame dimensions: {frame.shape}"
        
        cap.release()
        test_logger.info(f"Successfully read {direct_frames} frames directly from video")
        assert direct_frames > 0, "Failed to read any frames directly from video"
        
        # Now test with VideoSource
        test_logger.info("Testing with VideoSource...")
        self.vs = VideoSource(test_video)
        
        # _cap should not be initialized until start() is called
        assert not hasattr(self.vs, '_cap'), "VideoCapture should not be initialized yet"
        
        # Start the video processing with debug logging
        with caplog.at_level(logging.DEBUG):
            # Start the video source
            self.vs.start()
            
            # Now _cap should be initialized
            assert hasattr(self.vs, '_cap'), "VideoCapture should be initialized after start()"
            assert self.vs._cap.isOpened(), "VideoCapture should be opened"
            
            # Log video properties for debugging
            video_info = self.vs.get_video_info()
            width, height, fps = video_info
            test_logger.info(f"Video info: {width}x{height} @ {fps}fps")
            
            # Validate video properties
            assert width > 0 and height > 0, f"Invalid video dimensions: {width}x{height}"
            assert fps > 0, f"Invalid FPS: {fps}"
            
            # Get all frames with retries
            frames_received = 0
            max_attempts = expected_frame_count * 2  # Allow up to 2x frame count for retries
            frame_timestamps = []
            frame_shapes = set()
            
            test_logger.info(f"Attempting to read {expected_frame_count} frames...")

            for attempt in range(max_attempts):
                try:
                    frame_data = self.vs.get_frame()
                    if frame_data is None:
                        if attempt % 10 == 0:  # Only log every 10th attempt
                            test_logger.debug(f"Attempt {attempt + 1}/{max_attempts}: No frame received yet")
                        time.sleep(0.1)
                        continue

                    frame, timestamp = frame_data
                    frames_received += 1
                    frame_timestamps.append(timestamp)
                    frame_shapes.add(frame.shape)
                    
                    if frames_received % 10 == 0:  # Log progress every 10 frames
                        test_logger.debug(
                            f"Received frame {frames_received}/{expected_frame_count} "
                            f"({frames_received/expected_frame_count*100:.1f}%)"
                        )

                    # Basic frame validation
                    assert frame is not None, "Frame is None"
                    assert timestamp >= 0, f"Invalid timestamp: {timestamp}"
                    assert len(frame.shape) == 3, f"Invalid frame dimensions: {frame.shape}"
                    assert frame.shape[0] > 0 and frame.shape[1] > 0, \
                        f"Invalid frame dimensions: {frame.shape}"

                    if frames_received >= expected_frame_count:
                        test_logger.info("Successfully read all expected frames")
                        break
                        
                except Exception as e:
                    test_logger.error(f"Error on frame {frames_received}: {str(e)}")
                    raise
                    
            # Frame count validation
            test_logger.info(
                f"Frame count: {frames_received} (expected: {expected_frame_count}, "
                f"{frames_received/expected_frame_count*100:.1f}%)"
            )
            
            # Allow for up to 5% frame drops in test environment
            min_expected_frames = int(expected_frame_count * 0.95)
            assert frames_received >= min_expected_frames, \
                f"Expected at least {min_expected_frames} frames, got {frames_received}"
                
            # Validate timestamps are increasing
            for i in range(1, len(frame_timestamps)):
                assert frame_timestamps[i] >= frame_timestamps[i-1], \
                    f"Timestamps not monotonically increasing at frame {i}: " \
                    f"{frame_timestamps[i-1]:.3f} -> {frame_timestamps[i]:.3f}"
                    
            # Validate consistent frame dimensions
            assert len(frame_shapes) == 1, f"Inconsistent frame dimensions: {frame_shapes}"
            
            # Log frame rate statistics
            if len(frame_timestamps) > 1:
                actual_duration = frame_timestamps[-1] - frame_timestamps[0]
                actual_fps = (len(frame_timestamps) - 1) / actual_duration if actual_duration > 0 else 0
                test_logger.info(
                    f"Frame rate: {actual_fps:.2f} fps (expected: {expected_fps:.2f}), "
                    f"Duration: {actual_duration:.2f}s (expected: {expected_duration:.2f})"
                )
                
                # Allow 10% FPS deviation from expected
                assert abs(actual_fps - expected_fps) / expected_fps < 0.1, \
                    f"Frame rate deviates more than 10%: {actual_fps:.2f} vs {expected_fps:.2f} expected"
                        
            # Log final status
            test_logger.info(f"Successfully received {frames_received} frames")
            # Check if capture thread is running if we didn't get frames
            if frames_received == 0 and hasattr(self.vs, '_capture_thread'):
                capture_thread = self.vs._capture_thread
                is_alive = capture_thread.is_alive() if capture_thread else False
                test_logger.error(f"Capture thread state: is_alive={is_alive}")
                
                # Check queue status if available
                if hasattr(self.vs, 'frames_queue'):
                    try:
                        qsize = self.vs.frames_queue.qsize()
                        test_logger.error(f"Frame queue size: {qsize}")
                    except Exception as e:
                        test_logger.error(f"Error getting queue size: {e}")
            
            # Dump more debug info if no frames received
            if frames_received == 0:
                test_logger.error("=== Debug Info ===")
                test_logger.error(f"Video source: {self.vs.source}")
                test_logger.error(f"Video capture is open: {self.vs._cap.isOpened() if hasattr(self.vs, '_cap') else 'No _cap'}")
                if hasattr(self.vs, '_cap'):
                    test_logger.error(f"Capture properties: {self.vs._cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames, "
                                    f"{self.vs._cap.get(cv2.CAP_PROP_FPS)} fps")
            
            # Assert that we received at least one frame
            assert frames_received > 0, (
                f"Failed to receive any frames after {max_attempts} attempts. "
                f"Video info: {width}x{height} @ {fps}fps. "
                f"Check logs for more details."
            )
    
    @pytest.mark.parametrize("video_path", [
        "nonexistent.mp4",
        "",
        None
    ])
    def test_error_handling(self, video_path, caplog):
        """Test error handling for invalid video paths."""
        with pytest.raises(Exception):
            with caplog.at_level(logging.ERROR):
                self.vs = VideoSource(video_path)
                assert "Failed to open video" in caplog.text

if __name__ == "__main__":
    # For manual testing
    vs = VideoSource(TEST_VIDEOS['mp4'])
    try:
        print(f"Video opened: {vs.is_opened()}")
        print(f"FPS: {vs.get_fps()}")
        print(f"Frame size: {vs.get_frame_size()}")
        print(f"Frame count: {vs.get_frame_count()}")
    finally:
        vs.release()
