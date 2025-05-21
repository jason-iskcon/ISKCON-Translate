import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from transcription import TranscriptionEngine

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

def test_transcription_quality(audio_file, output_file=None, duration=30, chunk_size=30):
    """Test transcription quality on an audio file.
    
    Args:
        audio_file: Path to the audio file
        output_file: Optional path to save transcription results
        duration: Total duration to transcribe in seconds
        chunk_size: Size of audio chunks to process at once (in seconds)
    """
    print(f"Testing transcription on {duration}-second segment of {audio_file}")
    
    # Initialize transcription engine
    engine = TranscriptionEngine()
    engine.start_transcription()
    
    try:
        # Process audio in chunks
        total_chunks = int(duration // chunk_size) + (1 if duration % chunk_size > 0 else 0)
        results = []
        
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_duration = min(chunk_size, duration - chunk_start)
            
            if chunk_duration <= 0:
                break
                
            print(f"\nProcessing chunk {chunk_idx + 1}/{total_chunks} "
                  f"({chunk_start:.1f}s - {chunk_start + chunk_duration:.1f}s)")
            
            # Load the audio chunk
            audio_data, sample_rate = load_audio_segment(
                audio_file, 
                start_time=chunk_start, 
                duration=chunk_duration
            )
            
            if audio_data is None:
                print(f"Failed to load audio chunk starting at {chunk_start}s")
                continue
                
            print(f"  Loaded {len(audio_data)/sample_rate:.1f}s of audio at {sample_rate}Hz")
            
            # Add audio chunk to the engine
            engine.add_audio_segment((audio_data, chunk_start))
            
            # Wait a bit for processing
            time.sleep(5)  # Adjust based on your system's processing speed
            
            # Collect any available results
            while not engine.result_queue.empty():
                result = engine.result_queue.get()
                results.append(result)
                print(f"  [{result['timestamp']:.2f}s] {result['text']}")
        
        # Wait for final processing
        print("\nFinalizing transcription...")
        time.sleep(5)
        
        # Get any remaining results
        while not engine.result_queue.empty():
            result = engine.result_queue.get()
            results.append(result)
            print(f"  [{result['timestamp']:.2f}s] {result['text']}")
        
        # Sort results by timestamp
        results.sort(key=lambda x: x['timestamp'])
        
        # Save results to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"[{result['timestamp']:.2f}s] {result['text']}\n")
            print(f"\nTranscription saved to {output_file}")
        
        # Print summary
        print("\nTranscription complete!")
        print(f"Processed {total_chunks} chunks")
        print(f"Total segments transcribed: {len(results)}")
        
        # Print first and last few segments as a sample
        if results:
            print("\nSample transcription:")
            for i, result in enumerate(results[:3]):
                print(f"  [{result['timestamp']:.2f}s] {result['text']}")
            if len(results) > 6:
                print("  ...")
                for i in range(max(3, len(results)-3), len(results)):
                    print(f"  [{results[i]['timestamp']:.2f}s] {results[i]['text']}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        engine.stop_transcription()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test transcription quality on an audio file')
    parser.add_argument('audio_file', help='Path to the audio file to transcribe')
    parser.add_argument('--output', '-o', help='Output file for transcription results')
    parser.add_argument('--duration', '-d', type=float, default=30.0,
                       help='Duration of audio to transcribe in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    test_transcription_quality(args.audio_file, args.output, args.duration)
