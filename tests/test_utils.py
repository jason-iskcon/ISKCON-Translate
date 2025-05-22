"""
Test utilities for video processing tests.
"""
import os
import cv2
import numpy as np
from pathlib import Path


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
