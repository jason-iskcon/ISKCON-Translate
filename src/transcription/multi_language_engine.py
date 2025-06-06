"""
Multi-language transcription engine for concurrent language processing.

This module handles simultaneous transcription in multiple languages with
synchronized timing and color-coded output.
"""

import threading
import queue
import time
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# Import with try-except to handle both direct execution and module import
try:
    from ..logging_utils import get_logger, TRACE
except ImportError:
    from src.logging_utils import get_logger, TRACE

from .engine import TranscriptionEngine

logger = get_logger('transcription.multi_language')


class MultiLanguageTranscriptionEngine:
    """Manages multiple transcription engines for concurrent multi-language processing."""
    
    def __init__(self, primary_language: str = "en", secondary_languages: List[str] = None, 
                 model_size: str = "small", device: str = "auto", compute_type: str = "auto"):
        """Initialize the multi-language transcription engine.
        
        Args:
            primary_language: Primary language code (e.g., 'en')
            secondary_languages: List of secondary language codes (e.g., ['it', 'es'])
            model_size: Size of the Whisper model
            device: Device to run the models on
            compute_type: Computation type
        """
        self.primary_language = primary_language
        self.secondary_languages = secondary_languages or []
        self.all_languages = [primary_language] + self.secondary_languages
        
        # Create transcription engines for each language
        self.engines: Dict[str, TranscriptionEngine] = {}
        
        # Combined result queue for all languages
        self.combined_result_queue = queue.Queue()
        
        # Tracking
        self.is_running = False
        
        logger.info(f"Multi-language engine initialized for languages: {self.all_languages}")
        
        # Initialize engines for each language
        for language in self.all_languages:
            try:
                engine = TranscriptionEngine(
                    model_size=model_size,
                    device=device,
                    compute_type=compute_type,
                    language=language
                )
                self.engines[language] = engine
                logger.info(f"Initialized transcription engine for {language}")
            except Exception as e:
                logger.error(f"Failed to initialize engine for {language}: {e}")
    
    def start_transcription(self) -> bool:
        """Start transcription for all languages.
        
        Returns:
            bool: True if all engines started successfully
        """
        if self.is_running:
            logger.warning("Multi-language transcription already running")
            return False
        
        success_count = 0
        for language, engine in self.engines.items():
            try:
                if engine.start_transcription():
                    success_count += 1
                    logger.info(f"Started transcription for {language}")
                else:
                    logger.error(f"Failed to start transcription for {language}")
            except Exception as e:
                logger.error(f"Error starting transcription for {language}: {e}")
        
        if success_count > 0:
            self.is_running = True
            # Start result aggregation thread
            self._start_result_aggregation()
            logger.info(f"Multi-language transcription started for {success_count}/{len(self.engines)} languages")
            return True
        else:
            logger.error("Failed to start any transcription engines")
            return False
    
    def stop_transcription(self, timeout: float = 5.0) -> bool:
        """Stop transcription for all languages.
        
        Args:
            timeout: Maximum time to wait for shutdown
            
        Returns:
            bool: True if all engines stopped successfully
        """
        if not self.is_running:
            logger.warning("Multi-language transcription not running")
            return True
        
        self.is_running = False
        
        success_count = 0
        for language, engine in self.engines.items():
            try:
                if engine.stop_transcription(timeout):
                    success_count += 1
                    logger.info(f"Stopped transcription for {language}")
                else:
                    logger.warning(f"Timeout stopping transcription for {language}")
            except Exception as e:
                logger.error(f"Error stopping transcription for {language}: {e}")
        
        logger.info(f"Multi-language transcription stopped for {success_count}/{len(self.engines)} languages")
        return success_count == len(self.engines)
    
    def add_audio_segment(self, audio_segment: tuple) -> bool:
        """Add audio segment to all transcription engines.
        
        Args:
            audio_segment: Tuple of (audio_data, timestamp)
            
        Returns:
            bool: True if added to at least one engine
        """
        success_count = 0
        for language, engine in self.engines.items():
            try:
                if engine.add_audio_segment(audio_segment):
                    success_count += 1
            except Exception as e:
                logger.warning(f"Failed to add audio segment to {language} engine: {e}")
        
        return success_count > 0
    
    def get_transcription(self) -> Optional[dict]:
        """Get a transcription result from any language.
        
        Returns:
            Optional[dict]: Transcription result with language information
        """
        try:
            return self.combined_result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _start_result_aggregation(self):
        """Start the result aggregation thread that collects results from all engines."""
        def aggregate_results():
            """Aggregate results from all language engines."""
            while self.is_running:
                # Check each engine for results
                for language, engine in self.engines.items():
                    try:
                        result = engine.get_transcription()
                        if result:
                            # Add language information to the result
                            result['language'] = language
                            result['is_primary'] = (language == self.primary_language)
                            
                            # Put in combined queue
                            try:
                                self.combined_result_queue.put(result, timeout=0.1)
                                logger.debug(f"Aggregated result from {language}: {result['text'][:50]}...")
                            except queue.Full:
                                logger.warning(f"Combined result queue full, dropping {language} result")
                    except Exception as e:
                        logger.warning(f"Error getting result from {language} engine: {e}")
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
        
        # Start aggregation thread
        aggregation_thread = threading.Thread(
            target=aggregate_results,
            name="MultiLangResultAggregator",
            daemon=True
        )
        aggregation_thread.start()
        logger.debug("Started result aggregation thread")
    
    def process_audio(self, video_source, chunk_size: Optional[float] = None, 
                     overlap: Optional[float] = None) -> None:
        """Process audio from video source for all languages.
        
        Args:
            video_source: Video source object
            chunk_size: Size of audio chunks in seconds
            overlap: Overlap between chunks in seconds
        """
        # Use the primary language engine to drive audio processing
        primary_engine = self.engines.get(self.primary_language)
        if primary_engine:
            # Start audio processing for primary engine - it will feed audio to all engines
            primary_engine.process_audio(video_source, chunk_size, overlap)
        else:
            logger.error("No primary language engine available for audio processing")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_transcription() 