import os
import sys
import time
import pytest
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# Add src directory to path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.transcription import TranscriptionEngine

# Test fixtures
@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file for testing."""
    # Create a simple 1kHz sine wave for 1 second at 16kHz
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1kHz sine wave
    
    # Save to a temporary file
    temp_file = tmp_path / "test_audio.wav"
    sf.write(temp_file, audio_data, sample_rate)
    
    return str(temp_file)

def load_audio_segment(file_path, start_time=0, duration=30, target_sample_rate=16000):
    """Load an audio segment from a file and resample if needed.
    
    Args:
        file_path: Path to the audio file
        start_time: Start time in seconds
        duration: Duration in seconds
        target_sample_rate: Target sample rate in Hz
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    try:
        # Read the audio file
        with sf.SoundFile(file_path) as sf_file:
            sample_rate = sf_file.samplerate
            start_frame = int(start_time * sample_rate)
            end_frame = int((start_time + duration) * sample_rate)
            sf_file.seek(start_frame)
            audio_data = sf_file.read(frames=end_frame - start_frame, dtype='float32')
        
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Resample if needed
        if sample_rate != target_sample_rate:
            from scipy import signal
            if audio_data.size > 0:
                num_samples = int(audio_data.shape[0] * target_sample_rate / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
            sample_rate = target_sample_rate
            
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

def test_transcription_engine_initialization():
    """Test that the transcription engine initializes and starts correctly."""
    # Initialize and start the engine
    engine = TranscriptionEngine()
    
    # Verify basic attributes exist
    assert hasattr(engine, 'start_transcription'), "Engine should have start_transcription method"
    assert hasattr(engine, 'stop_transcription'), "Engine should have stop_transcription method"
    assert hasattr(engine, 'add_audio_segment'), "Engine should have add_audio_segment method"
    assert hasattr(engine, 'get_transcription'), "Engine should have get_transcription method"
    
    # Start the engine
    engine.start_transcription()
    
    try:
        # Add a small silent audio segment
        sample_rate = 16000
        silent_audio = np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence
        engine.add_audio_segment((silent_audio, 0.0))
        
        # Try to get a result (might be None since we're not testing actual transcription)
        result = engine.get_transcription()
        assert result is None or isinstance(result, dict), "Result should be None or a dict"
        
    finally:
        # Always stop the engine to clean up threads
        engine.stop_transcription()

def test_audio_loading(sample_audio_file):
    """Test that audio files can be loaded correctly."""
    # Load the test audio file
    audio_data, sample_rate = load_audio_segment(sample_audio_file)
    
    # Verify the loaded audio data
    assert audio_data is not None, "Failed to load audio data"
    assert len(audio_data) > 0, "Loaded audio data is empty"
    assert sample_rate == 16000, f"Expected sample rate 16000, got {sample_rate}"
    assert isinstance(audio_data, np.ndarray), "Audio data should be a numpy array"
    assert audio_data.dtype == np.float32, "Audio data should be float32"

if __name__ == "__main__":
    # This block is kept for manual testing if needed
    import tempfile
    
    # Run the test with a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_audio.wav")
        
        # Create a test audio file
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1kHz sine wave
        sf.write(test_file, audio_data, sample_rate)
        
        # Run the test
        test_transcription_quality(test_file, os.path.join(temp_dir, "output.txt"))
