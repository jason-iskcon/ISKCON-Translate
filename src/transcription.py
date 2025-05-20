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
        """Minimalist transcription engine - just returns hardcoded captions for MVP."""
        # Fixed sampling rate
        self.sampling_rate = 16000
        
        # Audio queue for processing
        self.audio_queue = Queue(maxsize=20)
        self.result_queue = Queue(maxsize=20)
        self.is_running = False
        
        # Pre-made captions for demo (with timestamps)
        self.demo_captions = [
            ("Welcome to the video.", 1.0), 
            ("This is a synchronized captioning system.", 5.0),
            ("Notice how captions appear at the right time.", 10.0),
            ("Synchronization is working correctly.", 15.0),
            ("Audio and video are properly aligned.", 20.0),
            ("This demonstrates timestamp-based synchronization.", 25.0),
            ("The system uses presentation timestamps for all components.", 30.0),
            ("Caption overlay is synchronized with video frames.", 35.0),
            ("This simple system ensures proper audio-video sync.", 40.0),
            ("Thank you for watching this demo.", 45.0)
        ]
        self.caption_index = 0
        self.latest_timestamp = 0.0
        self.last_emitted_caption_time = 0.0
        
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
                logger.debug(f"Updated timestamp: {self.latest_timestamp:.2f}s")
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
        """Process audio segments and generate demo captions with timestamps."""
        logger.info("Transcription worker started")
        
        try:
            while self.is_running:
                # Process any audio data in queue
                if not self.audio_queue.empty():
                    try:
                        audio_data, timestamp = self.audio_queue.get()
                        # We've already updated the timestamp in add_audio_segment
                        # but we could do additional processing here if needed
                    except Exception as e:
                        logger.error(f"Error processing audio data: {e}")
                
                # Check if we should emit the next caption based on timestamp
                if self.caption_index < len(self.demo_captions):
                    caption_text, caption_time = self.demo_captions[self.caption_index]
                    
                    # Emit caption when we reach its timestamp
                    if self.latest_timestamp >= caption_time and caption_time > self.last_emitted_caption_time:
                        # Create result with timestamp for synchronization
                        result = {
                            "text": caption_text,
                            "timestamp": caption_time,
                            "language": "english"
                        }
                        
                        # Add to result queue
                        self.result_queue.put(result)
                        logger.info(f"Caption generated: '{result['text']}' at time {caption_time:.2f}s")
                        
                        # Update tracking variables
                        self.last_emitted_caption_time = caption_time
                        self.caption_index += 1
                
                # Don't burn CPU
                time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error in transcription worker: {e}")
            
    def __enter__(self):
        self.start_transcription()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_transcription()
