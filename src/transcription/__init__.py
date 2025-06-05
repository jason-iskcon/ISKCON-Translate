"""
Transcription engine package for ISKCON-Translate.

This package provides a modular transcription engine built on top of faster-whisper
with automatic device detection, queue management, and performance monitoring.

New in v2.0: Enhanced Whisper inference with glossary support and fuzzy post-correction.
"""

from .engine import TranscriptionEngine
from .decode_wrapper import DecodeWrapper, TranscriptionResult, transcribe_with_context
from .glossary_policy import (
    GlossaryPolicySelector, GlossaryStrategy, GlossaryResult,
    create_glossary_selector
)
from .post_processor import (
    FuzzyPostProcessor, CorrectionResult, correct_transcription,
    create_post_processor
)
from .gt_whisper import GTWhisper

__all__ = [
    # Original engine
    'TranscriptionEngine',
    
    # Enhanced inference components
    'DecodeWrapper',
    'TranscriptionResult', 
    'transcribe_with_context',
    
    # Glossary system
    'GlossaryPolicySelector',
    'GlossaryStrategy',
    'GlossaryResult',
    'create_glossary_selector',
    
    # Post-processing
    'FuzzyPostProcessor',
    'CorrectionResult',
    'correct_transcription',
    'create_post_processor',
    
    # Main GT-Whisper interface
    'GTWhisper'
]

# Version information
__version__ = "2.0.0"
__author__ = "ISKCON-Translate Team"
__description__ = "Enhanced transcription engine with glossary support and fuzzy post-correction" 