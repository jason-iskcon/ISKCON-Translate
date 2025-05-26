"""
Advanced text processing module for intelligent caption handling.

This module provides sophisticated text analysis capabilities including:
- Sentence boundary detection and completion
- Intelligent word repetition detection and removal
- Cross-segment duplication detection and removal
- Context-aware text processing
- Semantic similarity analysis
"""

from .sentence_analyzer import SentenceAnalyzer
from .repetition_detector import RepetitionDetector, remove_repetitions
from .profanity_filter import ProfanityFilter, FilterLevel, ReplacementStrategy, filter_profanity
from .cross_segment_detector import CrossSegmentDetector, detect_cross_segment_duplications

__all__ = [
    'SentenceAnalyzer',
    'RepetitionDetector',
    'remove_repetitions',
    'ProfanityFilter',
    'FilterLevel',
    'ReplacementStrategy',
    'filter_profanity',
    'CrossSegmentDetector',
    'detect_cross_segment_duplications'
] 