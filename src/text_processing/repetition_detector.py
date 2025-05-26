"""
Intelligent repetition detection and removal for real-time transcription.

This module identifies and removes various types of repetitions that commonly
occur in speech-to-text systems, while preserving intentional repetitions.
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque, Counter
import difflib

try:
    from ..logging_utils import get_logger
except ImportError:
    from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RepetitionResult:
    """Result of repetition detection and removal."""
    original_text: str
    cleaned_text: str
    repetitions_found: List[Dict]
    confidence: float
    processing_time: float


class RepetitionDetector:
    """
    Intelligent repetition detection and removal system.
    
    Features:
    - Word-level repetition detection ("the the the" → "the")
    - Phrase-level repetition detection ("I think I think" → "I think")
    - Sentence-level repetition detection
    - Context-aware filtering (preserves intentional repetitions)
    - Real-time processing optimized
    - Confidence scoring for each detection
    """
    
    def __init__(self, 
                 max_word_repetitions: int = 3,
                 max_phrase_length: int = 5,
                 similarity_threshold: float = 0.8,
                 context_window: int = 10):
        """
        Initialize the repetition detector.
        
        Args:
            max_word_repetitions: Maximum allowed consecutive word repetitions
            max_phrase_length: Maximum phrase length to check for repetitions
            similarity_threshold: Similarity threshold for fuzzy matching
            context_window: Number of previous segments to consider for context
        """
        self.max_word_repetitions = max_word_repetitions
        self.max_phrase_length = max_phrase_length
        self.similarity_threshold = similarity_threshold
        self.context_window = context_window
        
        # Context buffer for cross-segment repetition detection
        self.context_buffer = deque(maxlen=context_window)
        
        # Intentional repetition patterns (words that are commonly repeated)
        self.intentional_patterns = self._load_intentional_patterns()
        
        # Common filler words that might be repeated
        self.filler_words = self._load_filler_words()
        
        logger.info(f"Repetition detector initialized: max_word_reps={max_word_repetitions}, "
                   f"max_phrase_len={max_phrase_length}, threshold={similarity_threshold}")
    
    def _load_intentional_patterns(self) -> Set[str]:
        """Load patterns that represent intentional repetitions."""
        return {
            # Emphasis patterns
            "very", "really", "so", "quite", "extremely",
            "absolutely", "totally", "completely", "definitely",
            
            # Religious/spiritual repetitions (for ISKCON context)
            "hare", "krishna", "rama", "om", "amen",
            "hallelujah", "praise", "glory", "holy",
            "hare krishna", "thank you",
            
            # Common intentional repetitions (only for strong emphasis)
            "yes", "no", "please", "thank", "thanks", "sorry",
            
            # Exclamations (only strong ones)
            "wow", "oh", "ah"
        }
    
    def _load_filler_words(self) -> Set[str]:
        """Load common filler words that are often repeated unintentionally."""
        return {
            "um", "uh", "er", "ah", "eh", "hmm", "mm",
            "like", "you", "know", "i", "mean", "well",
            "so", "and", "but", "the", "a", "an", "is", "are",
            "was", "were", "have", "has", "had", "will", "would",
            "could", "should", "can", "may", "might", "must"
        }
    
    def detect_and_remove_repetitions(self, text: str, context: Optional[str] = None) -> RepetitionResult:
        """
        Detect and remove repetitions from text.
        
        Args:
            text: Text to process
            context: Additional context for better detection
            
        Returns:
            RepetitionResult with cleaned text and metadata
        """
        import time
        start_time = time.perf_counter()
        
        repetitions_found = []
        cleaned_text = text
        
        # 1. Remove word-level repetitions
        cleaned_text, word_reps = self._remove_word_repetitions(cleaned_text)
        repetitions_found.extend(word_reps)
        
        # 2. Remove phrase-level repetitions
        cleaned_text, phrase_reps = self._remove_phrase_repetitions(cleaned_text)
        repetitions_found.extend(phrase_reps)
        
        # 3. Check for cross-segment repetitions using context buffer
        if self.context_buffer:
            cleaned_text, context_reps = self._remove_context_repetitions(cleaned_text)
            repetitions_found.extend(context_reps)
        
        # 4. Clean up formatting
        cleaned_text = self._clean_text(cleaned_text)
        
        # 5. Update context buffer
        self.context_buffer.append(cleaned_text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(text, cleaned_text, repetitions_found)
        
        processing_time = time.perf_counter() - start_time
        
        if repetitions_found:
            logger.debug(f"Removed {len(repetitions_found)} repetitions from: {text[:50]}...")
        
        return RepetitionResult(
            original_text=text,
            cleaned_text=cleaned_text,
            repetitions_found=repetitions_found,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def _remove_word_repetitions(self, text: str) -> Tuple[str, List[Dict]]:
        """Remove consecutive word repetitions."""
        repetitions = []
        words = text.split()
        cleaned_words = []
        
        i = 0
        while i < len(words):
            current_word = words[i].lower().strip('.,!?;:"')
            
            # Count consecutive repetitions
            repetition_count = 1
            j = i + 1
            
            while (j < len(words) and 
                   words[j].lower().strip('.,!?;:"') == current_word):
                repetition_count += 1
                j += 1
            
            # Check if this is an intentional repetition
            is_intentional = self._is_intentional_repetition(current_word, repetition_count)
            
            if repetition_count > 1 and not is_intentional:
                # Keep only one instance (or allowed number for intentional)
                keep_count = 1
                if current_word in self.intentional_patterns and repetition_count <= 3:
                    keep_count = min(2, repetition_count)  # Allow up to 2 for emphasis
                
                # Add the kept instances
                for k in range(keep_count):
                    cleaned_words.append(words[i + k])
                
                # Record the repetition
                repetitions.append({
                    'type': 'word_repetition',
                    'word': current_word,
                    'original_count': repetition_count,
                    'kept_count': keep_count,
                    'position': (i, j),
                    'confidence': 0.9 if repetition_count > 2 else 0.7
                })
                
                i = j  # Skip all repetitions
            else:
                cleaned_words.append(words[i])
                i += 1
        
        return ' '.join(cleaned_words), repetitions
    
    def _remove_phrase_repetitions(self, text: str) -> Tuple[str, List[Dict]]:
        """Remove phrase-level repetitions."""
        repetitions = []
        words = text.split()
        
        if len(words) < 4:  # Too short for phrase repetitions
            return text, repetitions
        
        cleaned_words = words.copy()
        
        # Check for phrase repetitions of different lengths
        for phrase_len in range(2, min(self.max_phrase_length + 1, len(words) // 2)):
            i = 0
            while i <= len(cleaned_words) - (phrase_len * 2):
                phrase1 = cleaned_words[i:i + phrase_len]
                phrase2 = cleaned_words[i + phrase_len:i + (phrase_len * 2)]
                
                # Check if phrases are similar
                similarity = self._calculate_phrase_similarity(phrase1, phrase2)
                
                if similarity >= self.similarity_threshold:
                    # Check if this is intentional (like "thank you thank you")
                    phrase_text = ' '.join(phrase1).lower()
                    # Only consider it intentional if the ENTIRE phrase is in intentional patterns
                    is_intentional = phrase_text in self.intentional_patterns
                    
                    if not is_intentional:
                        # Remove the second phrase
                        repetitions.append({
                            'type': 'phrase_repetition',
                            'phrase': ' '.join(phrase1),
                            'similarity': similarity,
                            'position': (i, i + phrase_len * 2),
                            'confidence': similarity
                        })
                        
                        # Remove the repeated phrase
                        cleaned_words = cleaned_words[:i + phrase_len] + cleaned_words[i + phrase_len * 2:]
                        continue
                
                i += 1
        
        return ' '.join(cleaned_words), repetitions
    
    def _remove_context_repetitions(self, text: str) -> Tuple[str, List[Dict]]:
        """Remove repetitions that span across segments using context buffer."""
        repetitions = []
        
        if not self.context_buffer:
            return text, repetitions
        
        # Check if current text is similar to recent context
        current_words = text.split()
        
        for i, previous_text in enumerate(reversed(self.context_buffer)):
            previous_words = previous_text.split()
            
            # Check for exact matches or high similarity
            similarity = difflib.SequenceMatcher(None, current_words, previous_words).ratio()
            
            if similarity >= self.similarity_threshold:
                # Check if this might be intentional (like repeated prayers)
                is_intentional = any(word.lower() in self.intentional_patterns 
                                   for word in current_words[:3])  # Check first few words
                
                if not is_intentional and similarity > 0.9:
                    # Very high similarity suggests unintentional repetition
                    repetitions.append({
                        'type': 'context_repetition',
                        'similarity': similarity,
                        'context_distance': i + 1,
                        'confidence': similarity
                    })
                    
                    # Return empty string to remove the repetition
                    return "", repetitions
        
        return text, repetitions
    
    def _is_intentional_repetition(self, word: str, count: int) -> bool:
        """Check if a word repetition is likely intentional."""
        # Very short words are often unintentional repetitions
        if len(word) <= 2 and count > 2:
            return False
        
        # Filler words are usually unintentional when repeated
        if word in self.filler_words and count > 1:
            return False
        
        # Words in intentional patterns might be repeated for emphasis
        if word in self.intentional_patterns:
            return count <= 2  # Allow up to 2 repetitions for emphasis
        
        # Default: more than 1 repetition is usually unintentional
        return count <= 1
    
    def _calculate_phrase_similarity(self, phrase1: List[str], phrase2: List[str]) -> float:
        """Calculate similarity between two phrases."""
        if len(phrase1) != len(phrase2):
            return 0.0
        
        # Exact match
        if phrase1 == phrase2:
            return 1.0
        
        # Case-insensitive match
        phrase1_lower = [word.lower().strip('.,!?;:"') for word in phrase1]
        phrase2_lower = [word.lower().strip('.,!?;:"') for word in phrase2]
        
        if phrase1_lower == phrase2_lower:
            return 0.95
        
        # Fuzzy similarity
        text1 = ' '.join(phrase1_lower)
        text2 = ' '.join(phrase2_lower)
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def _clean_text(self, text: str) -> str:
        """Clean up text formatting after repetition removal."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        return text
    
    def _calculate_confidence(self, original: str, cleaned: str, repetitions: List[Dict]) -> float:
        """Calculate confidence score for repetition removal."""
        if not repetitions:
            return 1.0
        
        # Base confidence on detection types and patterns
        total_confidence = 0.0
        for repetition in repetitions:
            total_confidence += repetition['confidence']
        
        # Average confidence
        avg_confidence = total_confidence / len(repetitions) if repetitions else 1.0
        
        # Adjust based on text length change - be more lenient
        original_len = len(original.split())
        cleaned_len = len(cleaned.split())
        
        if original_len > 0:
            reduction_ratio = (original_len - cleaned_len) / original_len
            
            # Only penalize if we removed a LOT of text
            if reduction_ratio > 0.7:
                avg_confidence *= 0.8
            elif reduction_ratio > 0.5:
                avg_confidence *= 0.95
        
        return min(1.0, avg_confidence)
    
    def reset_context(self):
        """Reset the context buffer."""
        self.context_buffer.clear()
        logger.debug("Repetition detector context buffer reset")
    
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            'max_word_repetitions': self.max_word_repetitions,
            'max_phrase_length': self.max_phrase_length,
            'similarity_threshold': self.similarity_threshold,
            'context_window': self.context_window,
            'context_buffer_size': len(self.context_buffer),
            'intentional_patterns_count': len(self.intentional_patterns),
            'filler_words_count': len(self.filler_words)
        }


# Convenience function for quick repetition removal
def remove_repetitions(text: str, 
                      max_word_reps: int = 3,
                      similarity_threshold: float = 0.8) -> str:
    """
    Quick repetition removal function.
    
    Args:
        text: Text to process
        max_word_reps: Maximum allowed word repetitions
        similarity_threshold: Similarity threshold for phrase detection
        
    Returns:
        Cleaned text
    """
    detector = RepetitionDetector(
        max_word_repetitions=max_word_reps,
        similarity_threshold=similarity_threshold
    )
    result = detector.detect_and_remove_repetitions(text)
    return result.cleaned_text 