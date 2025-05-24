"""
Legacy transcription module for backward compatibility.

This module provides backward compatibility by importing the refactored
TranscriptionEngine from the new modular transcription package.
"""

# Import the refactored TranscriptionEngine
from .transcription import TranscriptionEngine

# Maintain backward compatibility with any direct imports from the old module
__all__ = ['TranscriptionEngine'] 