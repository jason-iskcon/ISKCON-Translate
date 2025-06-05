"""
Abstract decode wrapper for enhanced Whisper inference.

This module provides a high-level interface for Whisper transcription with
glossary support and context-aware prompting.
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

try:
    from faster_whisper import WhisperModel
    from ..logging_utils import get_logger
except ImportError:
    from faster_whisper import WhisperModel
    from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Result of transcription with metadata."""
    text: str
    segments: List[Dict[str, Any]]
    language: str
    language_probability: float
    processing_time: float
    glossary_matches: int
    context_used: bool


class DecodeWrapper:
    """
    Abstract decode wrapper for enhanced Whisper inference.
    
    Features:
    - Context-aware prompting from previous transcript
    - Glossary integration for domain-specific terms
    - Configurable prompt building strategies
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 model: WhisperModel,
                 context_window: int = 32,
                 max_prompt_length: int = 224):
        """
        Initialize the decode wrapper.
        
        Args:
            model: Initialized WhisperModel instance
            context_window: Number of words to use from previous transcript
            max_prompt_length: Maximum length of the initial prompt
        """
        self.model = model
        self.context_window = context_window
        self.max_prompt_length = max_prompt_length
        
        # Context management
        self.previous_transcript = []
        self.glossary_text = ""
        
        # Performance tracking
        self.transcription_count = 0
        self.total_processing_time = 0.0
        
        logger.info(f"DecodeWrapper initialized with context_window={context_window}, "
                   f"max_prompt_length={max_prompt_length}")
    
    def load_glossary(self, glossary_path: Optional[Union[str, Path]]) -> bool:
        """
        Load glossary from file.
        
        Args:
            glossary_path: Path to glossary file (text file with terms)
            
        Returns:
            bool: True if glossary loaded successfully, False otherwise
        """
        if not glossary_path:
            self.glossary_text = ""
            logger.debug("No glossary path provided")
            return True
            
        try:
            glossary_path = Path(glossary_path)
            if not glossary_path.exists():
                logger.warning(f"Glossary file not found: {glossary_path}")
                self.glossary_text = ""
                return False
                
            with open(glossary_path, 'r', encoding='utf-8') as f:
                self.glossary_text = f.read().strip()
                
            logger.info(f"Loaded glossary with {len(self.glossary_text)} characters from {glossary_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load glossary from {glossary_path}: {e}")
            self.glossary_text = ""
            return False
    
    def update_context(self, new_text: str) -> None:
        """
        Update the context with new transcribed text.
        
        Args:
            new_text: New text to add to context
        """
        if not new_text.strip():
            return
            
        # Add words to context
        words = new_text.strip().split()
        self.previous_transcript.extend(words)
        
        # Keep only the last context_window words
        if len(self.previous_transcript) > self.context_window:
            self.previous_transcript = self.previous_transcript[-self.context_window:]
        
        logger.debug(f"Updated context: {len(self.previous_transcript)} words")
    
    def build_initial_prompt(self) -> str:
        """
        Build initial prompt from context and glossary.
        
        Returns:
            str: Initial prompt for Whisper
        """
        prompt_parts = []
        
        # Add context from previous transcript
        if self.previous_transcript:
            context_text = " ".join(self.previous_transcript[-self.context_window:])
            prompt_parts.append(context_text)
            logger.debug(f"Added context: {len(context_text)} chars")
        
        # Add glossary terms
        if self.glossary_text:
            # Truncate glossary if needed to fit within prompt limit
            available_space = self.max_prompt_length - sum(len(part) for part in prompt_parts) - 10
            if available_space > 0:
                glossary_part = self.glossary_text[:available_space]
                prompt_parts.append(glossary_part)
                logger.debug(f"Added glossary: {len(glossary_part)} chars")
        
        # Combine parts
        initial_prompt = " ".join(prompt_parts)
        
        # Ensure we don't exceed max length
        if len(initial_prompt) > self.max_prompt_length:
            initial_prompt = initial_prompt[:self.max_prompt_length]
            logger.debug(f"Truncated prompt to {self.max_prompt_length} chars")
        
        logger.debug(f"Built initial prompt: {len(initial_prompt)} chars")
        return initial_prompt
    
    def transcribe(self, 
                  audio_path: Union[str, Path, np.ndarray],
                  language: Optional[str] = None,
                  task: str = "transcribe",
                  temperature: float = 0.0,
                  **kwargs) -> TranscriptionResult:
        """
        Transcribe audio with context and glossary support.
        
        Args:
            audio_path: Path to audio file or numpy array of audio data
            language: Language code (e.g., 'en', 'hi', 'sa')
            task: Task type ('transcribe' or 'translate')
            temperature: Sampling temperature
            **kwargs: Additional arguments for Whisper
            
        Returns:
            TranscriptionResult: Transcription result with metadata
        """
        start_time = time.perf_counter()
        
        try:
            # Build initial prompt
            initial_prompt = self.build_initial_prompt()
            context_used = bool(initial_prompt.strip())
            
            # Prepare transcription parameters
            transcribe_params = {
                'task': task,
                'language': language,
                'temperature': temperature,
                'initial_prompt': initial_prompt if context_used else None,
                **kwargs
            }
            
            # Remove None values
            transcribe_params = {k: v for k, v in transcribe_params.items() if v is not None}
            
            logger.debug(f"Transcribing with params: {list(transcribe_params.keys())}")
            
            # Perform transcription
            segments, info = self.model.transcribe(audio_path, **transcribe_params)
            
            # Convert segments to list and extract text
            segment_list = list(segments)
            full_text = " ".join(segment.text.strip() for segment in segment_list)
            
            # Count glossary matches (simple word matching)
            glossary_matches = self._count_glossary_matches(full_text)
            
            # Update context with new text
            self.update_context(full_text)
            
            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            
            # Update performance stats
            self.transcription_count += 1
            self.total_processing_time += processing_time
            
            # Create result
            result = TranscriptionResult(
                text=full_text,
                segments=[{
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text,
                    'words': getattr(seg, 'words', [])
                } for seg in segment_list],
                language=info.language,
                language_probability=info.language_probability,
                processing_time=processing_time,
                glossary_matches=glossary_matches,
                context_used=context_used
            )
            
            logger.info(f"Transcribed {len(full_text)} chars in {processing_time:.3f}s "
                       f"(lang={info.language}, matches={glossary_matches})")
            
            return result
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            logger.error(f"Transcription failed after {processing_time:.3f}s: {e}")
            
            # Return empty result on failure
            return TranscriptionResult(
                text="",
                segments=[],
                language="unknown",
                language_probability=0.0,
                processing_time=processing_time,
                glossary_matches=0,
                context_used=False
            )
    
    def _count_glossary_matches(self, text: str) -> int:
        """
        Count how many glossary terms appear in the transcribed text.
        
        Args:
            text: Transcribed text
            
        Returns:
            int: Number of glossary term matches
        """
        if not self.glossary_text or not text:
            return 0
            
        # Simple word-based matching
        text_words = set(text.lower().split())
        glossary_words = set(self.glossary_text.lower().split())
        
        matches = len(text_words & glossary_words)
        return matches
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        avg_time = (self.total_processing_time / self.transcription_count 
                   if self.transcription_count > 0 else 0.0)
        
        return {
            'transcription_count': self.transcription_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_time,
            'context_words': len(self.previous_transcript),
            'glossary_length': len(self.glossary_text)
        }
    
    def reset_context(self) -> None:
        """Reset the context buffer."""
        self.previous_transcript.clear()
        logger.debug("Context reset")
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.transcription_count = 0
        self.total_processing_time = 0.0
        logger.debug("Performance stats reset")


# Convenience function for quick transcription
def transcribe_with_context(model: WhisperModel,
                          audio_path: Union[str, Path, np.ndarray],
                          glossary_path: Optional[Union[str, Path]] = None,
                          context_window: int = 32,
                          **kwargs) -> TranscriptionResult:
    """
    Quick transcription with context and glossary support.
    
    Args:
        model: WhisperModel instance
        audio_path: Path to audio file or audio data
        glossary_path: Optional path to glossary file
        context_window: Number of context words to use
        **kwargs: Additional transcription parameters
        
    Returns:
        TranscriptionResult: Transcription result
    """
    wrapper = DecodeWrapper(model, context_window=context_window)
    
    if glossary_path:
        wrapper.load_glossary(glossary_path)
    
    return wrapper.transcribe(audio_path, **kwargs) 