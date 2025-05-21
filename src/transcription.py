import numpy as np
import threading
import queue
from queue import Queue
import time
import logging
import os
from faster_whisper import WhisperModel
import torch

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set faster_whisper logger to WARNING level to reduce verbosity
logging.getLogger('faster_whisper').setLevel(logging.WARNING)

class TranscriptionEngine:
    def __init__(self, config=None):
        """Transcription engine using faster-whisper with medium model."""
        self.sampling_rate = 16000
        self.chunk_size = 30.0  # Process 30-second chunks for better context
        self.audio_queue = Queue(maxsize=5)  # Reduced queue size for larger chunks
        self.result_queue = Queue(maxsize=20)
        self.is_running = False
        self.latest_timestamp = 0.0
        self.last_emitted_caption_time = 0.0
        self.transcription_thread = None
        
        # Initialize faster-whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model_dir = os.path.expanduser("~/.cache/faster-whisper")
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Loading faster-whisper model on {device} with {compute_type} precision")
        self.model = WhisperModel(
            "small",  # Using small for better speed/accuracy balance
            device="cpu",
            compute_type="int8",
            download_root=model_dir
        )
        logger.info("Model loaded successfully")

        
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
        """Process audio segments and generate transcriptions with timestamps at sentence level.
        
        This worker processes audio chunks using a sliding window approach with overlap
        to ensure smooth, continuous transcription. It handles word-level timestamps
        and assembles complete sentences before emitting them.
        """
        logger.info("Starting transcription worker")
        
        # Buffer to hold partial words between chunks
        partial_word = ""
        partial_word_start = None
        
        try:
            while self.is_running:
                try:
                    # Get audio data from queue
                    audio_data, chunk_start_time = self.audio_queue.get(timeout=0.5)
                    if audio_data is None:  # Sentinel value to stop
                        break
                    
                    logger.debug(f"Processing audio chunk starting at {chunk_start_time:.2f}s")
                    
                    try:
                        # Process audio with Whisper using word-level timestamps
                        segments, info = self.model.transcribe(
                            audio_data,
                            language='en',
                            word_timestamps=True,
                            vad_filter=True,
                            vad_parameters=dict(
                                threshold=0.5,
                                min_silence_duration_ms=500
                            )
                        )
                        # Convert generator to list to allow indexing
                        segments = list(segments) if segments else []
                        
                        # Process segments and emit complete sentences
                        current_sentence = []
                        sentence_start = None
                        
                        # Handle any partial word from previous chunk
                        if partial_word:
                            current_sentence.append(partial_word)
                            if partial_word_start is not None:
                                sentence_start = partial_word_start
                            partial_word = ""
                        
                        for segment in segments:
                            for word in segment.words:
                                word_text = word.word.strip()
                                if not word_text:
                                    continue
                                
                                # Calculate absolute timestamps
                                word_start = chunk_start_time + (word.start or 0)
                                word_end = chunk_start_time + (word.end or 0)
                                
                                # Check if this is a continuation of a word from previous chunk
                                if not current_sentence and not word_text[0].isspace() and word_start > chunk_start_time + 0.1:
                                    # This might be the continuation of a word from previous chunk
                                    if not partial_word:
                                        partial_word = word_text
                                        partial_word_start = word_start
                                        continue
                                
                                # Start new sentence if needed
                                if not current_sentence:
                                    sentence_start = word_start
                                
                                # Add word to current sentence
                                current_sentence.append(word_text)
                                
                                # Check for sentence end (period, question mark, exclamation mark)
                                if word_text.endswith(('.', '?', '!')):
                                    if current_sentence:
                                        # Emit complete sentence
                                        sentence_text = ' '.join(current_sentence)
                                        self.result_queue.put({
                                            'text': sentence_text,
                                            'timestamp': sentence_start,
                                            'end_time': word_end,
                                            'language': 'english',
                                            'is_final': False
                                        })
                                        logger.debug(f"Emitted sentence: '{sentence_text}' at {sentence_start:.2f}-{word_end:.2f}s")
                                        current_sentence = []
                                        sentence_start = None
                        
                        # Handle any partial sentence at the end of the chunk
                        if current_sentence:
                            # Keep the last partial word for the next chunk
                            if current_sentence and current_sentence[-1] and not current_sentence[-1][-1] in '.!?':
                                partial_word = current_sentence.pop()
                                last_segment_end = segments[-1].end if segments and hasattr(segments[-1], 'end') else 0
                                partial_word_start = chunk_start_time + last_segment_end - 0.1  # Approximate
                            
                            # Emit any complete sentences we have
                            if current_sentence:
                                sentence_text = ' '.join(current_sentence)
                                sentence_end = chunk_start_time + (segments[-1].end if segments else 0) - (len(partial_word) * 0.1 if partial_word else 0)
                                self.result_queue.put({
                                    'text': sentence_text,
                                    'timestamp': sentence_start or chunk_start_time,
                                    'end_time': sentence_end,
                                    'language': 'english',
                                    'is_final': False
                                })
                                logger.debug(f"Emitted partial sentence: '{sentence_text}...' at {sentence_start or chunk_start_time:.2f}-{sentence_end:.2f}s")
                        
                        # Update latest timestamp if we have segments
                        if segments and hasattr(segments[-1], 'end'):
                            self.latest_timestamp = chunk_start_time + segments[-1].end
                        
                    except Exception as e:
                        logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                    
                    # Mark task as done
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    continue
                    
                except Exception as e:
                    logger.error(f"Error in transcription worker: {e}", exc_info=True)
                    time.sleep(0.1)
            
            # Emit any remaining partial sentence before stopping
            if partial_word:
                self.result_queue.put({
                    'text': partial_word,
                    'timestamp': partial_word_start or self.latest_timestamp,
                    'end_time': (self.latest_timestamp + 0.5) if hasattr(self, 'latest_timestamp') else 0.5,
                    'language': 'english',
                    'is_final': False
                })
                    
        except Exception as e:
            logger.error(f"Fatal error in transcription worker: {e}", exc_info=True)
        finally:
            logger.info("Transcription worker stopped")

    def __enter__(self):
        self.start_transcription()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_transcription()
