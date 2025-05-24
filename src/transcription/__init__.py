"""
Transcription engine package for ISKCON-Translate.

This package provides a modular transcription engine built on top of faster-whisper
with automatic device detection, queue management, and performance monitoring.
"""

from .engine import TranscriptionEngine

__all__ = ['TranscriptionEngine']

# Version information
__version__ = "1.0.0"
__author__ = "ISKCON-Translate Team"
__description__ = "Modular transcription engine with automatic device detection and performance monitoring" 