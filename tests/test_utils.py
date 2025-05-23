"""
Test utilities for video processing tests.
"""
import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List


class TextDetector:
    """Helper class for detecting text in frames."""
    
    def __init__(self, debug_dir: Optional[str] = None):
        """Initialize the text detector.
        
        Args:
            debug_dir: Directory to save debug frames (optional). If None, uses test_data/debug_frames.
        """
        if debug_dir is None:
            # Use test_data/debug_frames in the tests directory
            test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
            self.debug_dir = os.path.join(test_data_dir, 'debug_frames')
        else:
            self.debug_dir = debug_dir
            
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def has_visible_text(
        self, 
        frame: np.ndarray, 
        expected_text: Optional[str] = None, 
        threshold: float = 0.5,
        min_text_height: int = 20
    ) -> bool:
        """Check if the frame contains visible text.
        
        Args:
            frame: Input frame (BGR or grayscale)
            expected_text: Optional expected text for debugging
            threshold: Minimum percentage of pixels to consider as text (0-1)
            min_text_height: Minimum height of text in pixels to detect
            
        Returns:
            bool: True if text is detected, False otherwise
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Apply adaptive thresholding to better detect text
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Look at center region where text is likely to be
        height, width = gray.shape
        y_start, y_end = int(height * 0.3), int(height * 0.7)
        x_start, x_end = int(width * 0.1), int(width * 0.9)
        
        # Calculate the percentage of white pixels in the center region
        center_region = binary[y_start:y_end, x_start:x_end]
        white_pixels = np.sum(center_region == 255)
        total_pixels = center_region.size
        white_ratio = white_pixels / total_pixels
        
        # Save debug information if debug_dir is set
        if self.debug_dir:
            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            debug_text = f"Text: {white_ratio:.2f}%"
            cv2.putText(debug_frame, debug_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if expected_text:
                cv2.putText(debug_frame, f"Exp: {expected_text}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            timestamp = int(time.time() * 1000)
            debug_path = os.path.join(self.debug_dir, f"debug_{timestamp}.png")
            cv2.imwrite(debug_path, debug_frame)
        
        return white_ratio >= threshold


def create_test_video(output_path, duration=2.0, fps=30, width=640, height=480, with_audio=True):
    """
    Create a test video with random content and optional audio.
    
    Args:
        output_path: Path where the video will be saved
        duration: Duration in seconds
        fps: Frames per second
        width: Video width in pixels
        height: Video height in pixels
        with_audio: Whether to include an audio stream
    """
    import tempfile
    import subprocess
    import numpy as np
    import cv2
    
    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # First create a video file with FFmpeg directly to ensure it has the right format
        video_path = os.path.join(temp_dir, 'video.mp4')
        
        # Generate frames using FFmpeg's test pattern
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'color=c=red:size={width}x{height}:rate={fps}:duration={duration}',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            video_path,
            '-y'  # Overwrite output file if it exists
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating test video: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            raise
            
        # If audio is requested, create a silent audio track and mux it with the video
        if with_audio:
            audio_path = os.path.join(temp_dir, 'audio.wav')
            output_with_audio = os.path.join(temp_dir, 'output.mp4')
            
            # Create a silent audio track
            cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'aevalsrc=0:d={duration}:s=44100',
                '-ar', '44100',
                '-ac', '2',
                audio_path,
                '-y'
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Mux video and audio
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    '-shortest',
                    output_with_audio,
                    '-y'
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Use the version with audio
                video_path = output_with_audio
                
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not create audio track: {e}")
                print(f"FFmpeg stderr: {e.stderr}")
                # Continue with video-only if audio creation fails
        
        # Copy the final file to the requested output path
        import shutil
        shutil.copy2(video_path, output_path)
    
    return output_path


class TestVideo:
    """Context manager for creating temporary test videos."""
    
    def __init__(self, tmp_path, name="test_video.mp4", duration=2.0, fps=30, with_audio=True):
        self.tmp_path = tmp_path
        self.name = name
        self.duration = duration
        self.fps = fps
        self.with_audio = with_audio
        self.path = None
    
    def __enter__(self):
        self.path = create_test_video(
            self.tmp_path / self.name,
            duration=self.duration,
            fps=self.fps,
            with_audio=self.with_audio
        )
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup is handled by pytest's tmp_path fixture
        pass
