"""
Intelligent sentence analysis for caption processing.

This module provides advanced sentence boundary detection, completion analysis,
and intelligent text segmentation for real-time transcription.
"""

import re
import string
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

try:
    from ..logging_utils import get_logger
except ImportError:
    from src.logging_utils import get_logger

logger = get_logger(__name__)


class SentenceState(Enum):
    """Enumeration of possible sentence states."""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FRAGMENT = "fragment"
    CONTINUATION = "continuation"


@dataclass
class SentenceInfo:
    """Information about a sentence or text segment."""
    text: str
    state: SentenceState
    confidence: float
    start_pos: int
    end_pos: int
    has_subject: bool = False
    has_predicate: bool = False
    word_count: int = 0
    
    def __post_init__(self):
        self.word_count = len(self.text.split())


class SentenceAnalyzer:
    """Advanced sentence analysis for intelligent caption processing."""
    
    def __init__(self):
        """Initialize the sentence analyzer."""
        # Sentence ending patterns
        self.sentence_endings = r'[.!?]'
        self.strong_endings = r'[.!?](?:\s|$)'
        
        # Common sentence starters (capitalized words that often start sentences)
        self.sentence_starters = {
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'and', 'but', 'or', 'so', 'yet', 'for', 'nor',
            'when', 'where', 'why', 'how', 'what', 'who',
            'if', 'unless', 'because', 'since', 'although',
            'however', 'therefore', 'moreover', 'furthermore'
        }
        
        # Common incomplete patterns
        self.incomplete_patterns = [
            r'\b(and|but|or|so|because|since|when|where|while|if|unless|although)\s*$',
            r'\b(the|a|an)\s*$',
            r'\b(is|are|was|were|will|would|could|should|might|may)\s*$',
            r'\b(to|for|with|by|from|in|on|at|of)\s*$',
            r'\b(very|really|quite|rather|somewhat|extremely)\s*$'
        ]
        
        # Subject indicators (simplified)
        self.subject_indicators = {
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'this', 'that', 'these', 'those', 'there'
        }
        
        # Predicate indicators (simplified)
        self.predicate_indicators = {
            'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'might', 'may', 'must', 'shall'
        }
    
    def analyze_text(self, text: str) -> List[SentenceInfo]:
        """Analyze text and return sentence information.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of SentenceInfo objects
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        sentences = []
        
        # Split by strong sentence boundaries first
        segments = re.split(self.strong_endings, text)
        
        current_pos = 0
        for i, segment in enumerate(segments):
            if not segment.strip():
                continue
                
            segment = segment.strip()
            
            # Check if this segment has a sentence ending (except for last segment)
            has_ending = i < len(segments) - 1
            
            # Analyze the segment
            sentence_info = self._analyze_segment(
                segment, current_pos, has_ending
            )
            
            sentences.append(sentence_info)
            current_pos += len(segment) + (1 if has_ending else 0)
        
        return sentences
    
    def _analyze_segment(self, segment: str, start_pos: int, has_ending: bool) -> SentenceInfo:
        """Analyze a single text segment.
        
        Args:
            segment: Text segment to analyze
            start_pos: Starting position in original text
            has_ending: Whether segment has sentence ending punctuation
            
        Returns:
            SentenceInfo object
        """
        words = segment.lower().split()
        word_count = len(words)
        
        # Check for subject and predicate
        has_subject = any(word in self.subject_indicators for word in words)
        has_predicate = any(word in self.predicate_indicators for word in words)
        
        # Determine sentence state
        state = self._determine_sentence_state(
            segment, words, has_ending, has_subject, has_predicate
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            segment, words, state, has_ending, has_subject, has_predicate
        )
        
        return SentenceInfo(
            text=segment,
            state=state,
            confidence=confidence,
            start_pos=start_pos,
            end_pos=start_pos + len(segment),
            has_subject=has_subject,
            has_predicate=has_predicate,
            word_count=word_count
        )
    
    def _determine_sentence_state(self, segment: str, words: List[str], 
                                has_ending: bool, has_subject: bool, 
                                has_predicate: bool) -> SentenceState:
        """Determine the state of a sentence segment.
        
        Args:
            segment: Original text segment
            words: Lowercase words
            has_ending: Whether segment has ending punctuation
            has_subject: Whether segment has a subject
            has_predicate: Whether segment has a predicate
            
        Returns:
            SentenceState enum value
        """
        # Check for incomplete patterns
        for pattern in self.incomplete_patterns:
            if re.search(pattern, segment.lower()):
                return SentenceState.INCOMPLETE
        
        # Very short segments are likely fragments
        if len(words) < 2:
            return SentenceState.FRAGMENT
        
        # Has ending punctuation and basic sentence structure
        if has_ending and has_subject and has_predicate:
            return SentenceState.COMPLETE
        
        # Has ending but missing structure
        if has_ending:
            if len(words) >= 3:
                return SentenceState.COMPLETE  # Assume complete if reasonable length
            else:
                return SentenceState.FRAGMENT
        
        # No ending punctuation
        if has_subject and has_predicate and len(words) >= 4:
            return SentenceState.INCOMPLETE  # Likely incomplete sentence
        
        # Check if it looks like a continuation
        first_word = words[0] if words else ""
        if first_word in {'and', 'but', 'or', 'so', 'because', 'since', 'when'}:
            return SentenceState.CONTINUATION
        
        # Default to fragment for short segments
        if len(words) < 3:
            return SentenceState.FRAGMENT
        
        return SentenceState.INCOMPLETE
    
    def _calculate_confidence(self, segment: str, words: List[str], 
                            state: SentenceState, has_ending: bool,
                            has_subject: bool, has_predicate: bool) -> float:
        """Calculate confidence score for sentence analysis.
        
        Args:
            segment: Original text segment
            words: Lowercase words
            state: Determined sentence state
            has_ending: Whether segment has ending punctuation
            has_subject: Whether segment has a subject
            has_predicate: Whether segment has a predicate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear indicators
        if has_ending:
            confidence += 0.2
        
        if has_subject:
            confidence += 0.1
        
        if has_predicate:
            confidence += 0.1
        
        # Adjust based on word count
        word_count = len(words)
        if word_count >= 5:
            confidence += 0.1
        elif word_count <= 2:
            confidence -= 0.2
        
        # Adjust based on state
        if state == SentenceState.COMPLETE:
            confidence += 0.1
        elif state == SentenceState.FRAGMENT:
            confidence -= 0.2
        
        # Check for common complete phrases
        segment_lower = segment.lower()
        complete_phrases = [
            'thank you', 'excuse me', 'i see', 'oh no', 'of course',
            'not yet', 'right now', 'over there', 'come on'
        ]
        
        if any(phrase in segment_lower for phrase in complete_phrases):
            confidence += 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def is_sentence_complete(self, text: str, min_confidence: float = 0.7) -> bool:
        """Check if text represents a complete sentence.
        
        Args:
            text: Text to check
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if sentence appears complete
        """
        sentences = self.analyze_text(text)
        
        if not sentences:
            return False
        
        # Check if we have at least one complete sentence with high confidence
        for sentence in sentences:
            if (sentence.state == SentenceState.COMPLETE and 
                sentence.confidence >= min_confidence):
                return True
        
        return False
    
    def find_sentence_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """Find sentence boundaries in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of (start, end) positions for each sentence
        """
        sentences = self.analyze_text(text)
        boundaries = []
        
        for sentence in sentences:
            if sentence.state in [SentenceState.COMPLETE, SentenceState.INCOMPLETE]:
                boundaries.append((sentence.start_pos, sentence.end_pos))
        
        return boundaries
    
    def get_incomplete_ending(self, text: str) -> Optional[str]:
        """Get the incomplete ending of text that should be continued.
        
        Args:
            text: Text to analyze
            
        Returns:
            Incomplete ending text or None
        """
        sentences = self.analyze_text(text)
        
        if not sentences:
            return None
        
        last_sentence = sentences[-1]
        
        # Return incomplete or continuation sentences
        if last_sentence.state in [SentenceState.INCOMPLETE, SentenceState.CONTINUATION]:
            return last_sentence.text
        
        # Return fragments if they're substantial
        if (last_sentence.state == SentenceState.FRAGMENT and 
            last_sentence.word_count >= 2):
            return last_sentence.text
        
        return None 