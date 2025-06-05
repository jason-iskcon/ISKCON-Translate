"""
Audio utility functions for transcription engine.
"""

import numpy as np
import soundfile as sf
from typing import Optional, Tuple
from .utils import logger


def get_audio_duration(audio_path: str) -> float:
    """Get the duration of an audio file in seconds.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        float: Duration in seconds
    """
    try:
        with sf.SoundFile(audio_path) as audio_file:
            return len(audio_file) / audio_file.samplerate
    except Exception as e:
        logger.error(f"Error getting audio duration for {audio_path}: {e}")
        return 0.0


def load_audio(audio_path: str, start: float = 0.0, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """Load audio data from a file.
    
    Args:
        audio_path: Path to the audio file
        start: Start time in seconds
        duration: Duration to load in seconds (None for entire file)
        
    Returns:
        Tuple[np.ndarray, int]: (audio_data, sample_rate)
    """
    try:
        with sf.SoundFile(audio_path) as audio_file:
            if start > 0:
                audio_file.seek(int(start * audio_file.samplerate))
            
            if duration is not None:
                samples = int(duration * audio_file.samplerate)
                audio_data = audio_file.read(samples)
            else:
                audio_data = audio_file.read()
                
            return audio_data, audio_file.samplerate
            
    except Exception as e:
        logger.error(f"Error loading audio from {audio_path}: {e}")
        return np.array([]), 0
