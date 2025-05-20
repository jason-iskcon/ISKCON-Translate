import numpy as np
import threading
from queue import Queue
import time
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriptionEngine:
    def __init__(self, config=None):
        """Transcription engine using Whisper small model."""
        import os
        self.sampling_rate = 16000
        self.audio_queue = Queue(maxsize=20)
        self.result_queue = Queue(maxsize=20)
        self.is_running = False
        self.latest_timestamp = 0.0
        self.last_emitted_caption_time = 0.0
        self.transcription_thread = None
        # Load Whisper small model from cache (do not download)
        import whisper
        model_dir = os.path.expanduser(r"C:/Users/Jason/.cache/whisper")
        self.model = whisper.load_model("small", download_root=model_dir)
        self.model_dir = model_dir
        import numpy as np
        self.np = np

        
    # No model initialization needed for MVP
            
    def start_transcription(self):
        """Start the transcription thread."""
        if self.is_running:
            return
            
        self.is_running = True
        self.transcription_thread = threading.Thread(target=self._transcription_worker)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        logger.info("Transcription thread started")
        
    def stop_transcription(self):
        """Stop transcription thread."""
        self.is_running = False
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join(timeout=1.0)
        logger.info("Transcription stopped")
            
    def add_audio_segment(self, audio_data):
        """Add audio segment to processing queue."""
        if not self.is_running:
            return
            
        try:
            # Expecting a tuple of (audio_data, timestamp)
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                # Extract audio data and timestamp
                audio_segment, timestamp = audio_data
                
                # Store in queue for processing
                self.audio_queue.put((audio_segment, timestamp))
                
                # Update the latest timestamp directly
                self.latest_timestamp = max(self.latest_timestamp, timestamp)
            else:
                logger.warning("Received malformed audio data")
        except Exception as e:
            logger.error(f"Error adding audio segment: {e}")
                
    def get_transcription(self):
        """Get the next available transcription result."""
        if self.result_queue.empty():
            return None
            
        try:
            result = self.result_queue.get_nowait()
            return result
        except:
            return None
            
    def _transcription_worker(self):
        """Process audio segments and generate Whisper transcriptions with timestamps."""
        logger.info("Transcription worker started (Whisper model)")
        
        try:
            while self.is_running:
                if not self.audio_queue.empty():
                    try:
                        # Get audio data from queue
                        audio_data, timestamp = self.audio_queue.get()
                        
                        # Prepare audio for Whisper (must be float32, mono, 16kHz)
                        if not isinstance(audio_data, self.np.ndarray):
                            audio_data = self.np.array(audio_data, dtype=self.np.float32)
                        if audio_data.dtype != self.np.float32:
                            audio_data = audio_data.astype(self.np.float32)
                        
                        # Convert to mono if needed
                        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                            audio_data = audio_data.mean(axis=1)
                        
                        # Ensure we have enough audio for Whisper (at least 1 second)
                        if len(audio_data) < self.sampling_rate:
                            audio_data = self.np.pad(audio_data, (0, self.sampling_rate - len(audio_data)))
                        else:
                            audio_data = audio_data[:self.sampling_rate]  # Limit to 1 second
                        
                        # Transcribe with Whisper
                        result = self.model.transcribe(audio_data, language='en', fp16=False, verbose=False)
                        text = result.get('text', '').strip()
                        
                        # If we got text, add it to the result queue
                        if text:
                            transcription = {
                                "text": text,
                                "timestamp": timestamp,
                                "language": result.get('language', 'english')
                            }
                            self.result_queue.put(transcription)
                            logger.info(f"Transcribed at {timestamp:.2f}s: '{text}'")
                            self.last_emitted_caption_time = timestamp
                    except Exception as e:
                        logger.error(f"Error in transcription: {e}")
                # Small sleep to prevent 100% CPU usage
                time.sleep(0.05)
        except Exception as e:
            logger.error(f"Error in transcription worker: {e}")
            
    def __enter__(self):
        self.start_transcription()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_transcription()
