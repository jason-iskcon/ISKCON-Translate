"""
Cross-segment repetition detection for real-time transcription.

This module specifically handles text duplications that cross caption boundaries,
which is a common issue with chunked audio processing in Whisper AI.
"""

import re
import difflib
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque
import time

try:
    from ..logging_utils import get_logger
except ImportError:
    from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CrossSegmentResult:
    """Result of cross-segment repetition detection."""
    original_text: str
    cleaned_text: str
    duplications_found: List[Dict]
    confidence: float
    processing_time: float
    action_taken: str  # "kept", "merged", "removed", "cleaned"


class CrossSegmentDetector:
    """
    Detects and removes text duplications that cross caption boundaries.
    
    This addresses the core issue where Whisper AI with short audio chunks
    creates overlapping captions with repeated text content.
    
    Features:
    - Detects exact text overlaps between consecutive segments
    - Identifies partial sentence continuations
    - Removes redundant text while preserving meaning
    - Handles word-level and phrase-level overlaps
    - Real-time processing for live captions
    """
    
    def __init__(self, 
                 context_window: int = 5,
                 overlap_threshold: float = 0.3,
                 similarity_threshold: float = 0.7,
                 min_overlap_words: int = 2):
        """
        Initialize the cross-segment detector.
        
        Args:
            context_window: Number of previous segments to consider
            overlap_threshold: Minimum overlap ratio to consider duplication
            similarity_threshold: Similarity threshold for fuzzy matching
            min_overlap_words: Minimum words needed for overlap detection
        """
        self.context_window = context_window
        self.overlap_threshold = overlap_threshold
        self.similarity_threshold = similarity_threshold
        self.min_overlap_words = min_overlap_words
        
        # Context buffer for recent segments
        self.segment_buffer = deque(maxlen=context_window)
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'duplications_found': 0,
            'text_removed': 0,
            'segments_merged': 0
        }
        
        logger.info(f"Cross-segment detector initialized: window={context_window}, "
                   f"overlap_threshold={overlap_threshold}, similarity_threshold={similarity_threshold}")
    
    def process_segment(self, text: str, timestamp: float, context: Optional[str] = None) -> CrossSegmentResult:
        """
        Process a new text segment and detect cross-segment duplications.
        
        Args:
            text: New text segment to process
            timestamp: Timestamp of the segment
            context: Additional context information
            
        Returns:
            CrossSegmentResult with cleaned text and metadata
        """
        start_time = time.perf_counter()
        self.stats['total_processed'] += 1
        
        if not text.strip():
            return CrossSegmentResult(
                original_text=text,
                cleaned_text=text,
                duplications_found=[],
                confidence=1.0,
                processing_time=time.perf_counter() - start_time,
                action_taken="kept"
            )
        
        duplications_found = []
        cleaned_text = text.strip()
        action_taken = "kept"
        
        # Check against recent segments
        if self.segment_buffer:
            cleaned_text, duplications, action = self._detect_and_remove_duplications(cleaned_text, timestamp)
            duplications_found.extend(duplications)
            if duplications:
                action_taken = action
                self.stats['duplications_found'] += len(duplications)
        
        # Add current segment to buffer
        self.segment_buffer.append({
            'text': cleaned_text,
            'original_text': text,
            'timestamp': timestamp,
            'words': self._extract_words(cleaned_text)
        })
        
        # Calculate confidence
        confidence = self._calculate_confidence(text, cleaned_text, duplications_found)
        
        processing_time = time.perf_counter() - start_time
        
        if duplications_found:
            logger.debug(f"Cross-segment: Found {len(duplications_found)} duplications in: {text[:50]}...")
            logger.debug(f"Action taken: {action_taken}")
        
        return CrossSegmentResult(
            original_text=text,
            cleaned_text=cleaned_text,
            duplications_found=duplications_found,
            confidence=confidence,
            processing_time=processing_time,
            action_taken=action_taken
        )
    
    def _detect_and_remove_duplications(self, text: str, timestamp: float) -> Tuple[str, List[Dict], str]:
        """Detect and remove duplications with previous segments."""
        duplications = []
        cleaned_text = text
        action_taken = "kept"
        
        current_words = self._extract_words(text)
        
        # Check each previous segment (most recent first)
        for i, prev_segment in enumerate(reversed(self.segment_buffer)):
            prev_words = prev_segment['words']
            
            # 1. Check for exact text overlap
            exact_overlap = self._find_exact_overlap(current_words, prev_words)
            if exact_overlap:
                cleaned_text, overlap_info = self._remove_exact_overlap(cleaned_text, exact_overlap)
                duplications.append({
                    'type': 'exact_overlap',
                    'previous_segment': prev_segment['text'],
                    'overlap': overlap_info,
                    'confidence': 0.95,
                    'segment_distance': i + 1
                })
                action_taken = "cleaned"
                continue
            
            # 2. Check for partial sentence continuation
            continuation = self._find_sentence_continuation(current_words, prev_words)
            if continuation:
                cleaned_text, cont_info = self._handle_sentence_continuation(cleaned_text, continuation)
                duplications.append({
                    'type': 'sentence_continuation',
                    'previous_segment': prev_segment['text'],
                    'continuation': cont_info,
                    'confidence': 0.85,
                    'segment_distance': i + 1
                })
                action_taken = "merged"
                continue
            
            # 3. Check for fuzzy similarity (high overlap)
            similarity = self._calculate_segment_similarity(current_words, prev_words)
            if similarity > self.similarity_threshold:
                # High similarity - likely duplicate segment
                if similarity > 0.9:
                    # Very high similarity - remove entirely
                    cleaned_text = ""
                    action_taken = "removed"
                else:
                    # Moderate similarity - remove overlapping parts
                    cleaned_text = self._remove_similar_parts(cleaned_text, prev_segment['text'])
                    action_taken = "cleaned"
                
                duplications.append({
                    'type': 'fuzzy_similarity',
                    'previous_segment': prev_segment['text'],
                    'similarity': similarity,
                    'confidence': similarity,
                    'segment_distance': i + 1
                })
                break  # Stop after finding high similarity
            
            # 4. Check for substantial word overlap (even if not exact)
            common_words = set(current_words) & set(prev_words)
            if len(common_words) >= 3:  # At least 3 common words
                overlap_ratio = len(common_words) / len(current_words)
                if overlap_ratio > 0.4:  # More than 40% overlap
                    # Remove common words from current text
                    original_words = text.split()
                    filtered_words = []
                    for word in original_words:
                        word_normalized = self._extract_words(word)
                        if word_normalized and word_normalized[0] not in common_words:
                            filtered_words.append(word)
                    if filtered_words:
                        cleaned_text = ' '.join(filtered_words)
                        action_taken = "cleaned"
                    else:
                        cleaned_text = ""
                        action_taken = "removed"
                    
                    duplications.append({
                        'type': 'word_overlap',
                        'previous_segment': prev_segment['text'],
                        'common_words': len(common_words),
                        'overlap_ratio': overlap_ratio,
                        'confidence': overlap_ratio,
                        'segment_distance': i + 1
                    })
                    break
        
        return cleaned_text, duplications, action_taken
    
    def _find_exact_overlap(self, current_words: List[str], prev_words: List[str]) -> Optional[Dict]:
        """Find exact word overlap between segments."""
        if len(current_words) < self.min_overlap_words or len(prev_words) < self.min_overlap_words:
            return None
        
        # Check for overlap at the beginning of current with end of previous
        max_overlap = min(len(current_words), len(prev_words))
        
        for overlap_len in range(max_overlap, self.min_overlap_words - 1, -1):
            # Check if first N words of current match last N words of previous
            current_start = current_words[:overlap_len]
            prev_end = prev_words[-overlap_len:]
            
            if current_start == prev_end:
                overlap_ratio = overlap_len / len(current_words)
                # Be more lenient with overlap threshold for exact matches
                if overlap_ratio >= max(0.2, self.overlap_threshold * 0.7):
                    return {
                        'overlap_words': overlap_len,
                        'overlap_ratio': overlap_ratio,
                        'overlapping_text': ' '.join(current_start),
                        'position': 'start'
                    }
        
        # Check for overlap at the end of current with beginning of previous
        for overlap_len in range(max_overlap, self.min_overlap_words - 1, -1):
            current_end = current_words[-overlap_len:]
            prev_start = prev_words[:overlap_len]
            
            if current_end == prev_start:
                overlap_ratio = overlap_len / len(current_words)
                # Be more lenient with overlap threshold for exact matches
                if overlap_ratio >= max(0.2, self.overlap_threshold * 0.7):
                    return {
                        'overlap_words': overlap_len,
                        'overlap_ratio': overlap_ratio,
                        'overlapping_text': ' '.join(current_end),
                        'position': 'end'
                    }
        
        return None
    
    def _find_sentence_continuation(self, current_words: List[str], prev_words: List[str]) -> Optional[Dict]:
        """Find if current segment continues a sentence from previous segment."""
        if len(current_words) < 2 or len(prev_words) < 2:
            return None
        
        # Check if previous segment ends mid-sentence and current continues it
        prev_text = ' '.join(prev_words)
        current_text = ' '.join(current_words)
        
        # Look for incomplete sentences (no ending punctuation)
        if not re.search(r'[.!?]$', prev_text.strip()):
            # Previous doesn't end with punctuation - might be incomplete
            
            # Check if combining them creates a more complete sentence
            combined = prev_text + ' ' + current_text
            
            # Simple heuristic: if combined text has better sentence structure
            if self._has_better_sentence_structure(combined, current_text):
                return {
                    'type': 'incomplete_sentence',
                    'previous_incomplete': True,
                    'combined_text': combined
                }
        
        return None
    
    def _remove_exact_overlap(self, text: str, overlap_info: Dict) -> Tuple[str, Dict]:
        """Remove exact overlap from text."""
        # Work with original text to preserve punctuation and spacing
        original_words = text.split()
        overlap_words = overlap_info['overlap_words']
        
        if overlap_info['position'] == 'start':
            # Remove overlapping words from the beginning
            remaining_words = original_words[overlap_words:]
            cleaned_text = ' '.join(remaining_words)
        else:  # position == 'end'
            # Remove overlapping words from the end
            remaining_words = original_words[:-overlap_words]
            cleaned_text = ' '.join(remaining_words)
        
        return cleaned_text, overlap_info
    
    def _handle_sentence_continuation(self, text: str, continuation_info: Dict) -> Tuple[str, Dict]:
        """Handle sentence continuation by merging appropriately."""
        # For now, keep the current text as-is but mark it as a continuation
        # In a more advanced implementation, we might merge with the previous segment
        return text, continuation_info
    
    def _calculate_segment_similarity(self, words1: List[str], words2: List[str]) -> float:
        """Calculate similarity between two word lists."""
        if not words1 or not words2:
            return 0.0
        
        # Use difflib for sequence similarity
        matcher = difflib.SequenceMatcher(None, words1, words2)
        return matcher.ratio()
    
    def _remove_similar_parts(self, current_text: str, prev_text: str) -> str:
        """Remove similar parts from current text based on previous text."""
        current_words = self._extract_words(current_text)
        prev_words = self._extract_words(prev_text)
        
        # Find the longest common subsequence and remove it
        matcher = difflib.SequenceMatcher(None, prev_words, current_words)
        matching_blocks = matcher.get_matching_blocks()
        
        # Remove the largest matching block from current text
        if matching_blocks:
            largest_block = max(matching_blocks, key=lambda x: x.size)
            if largest_block.size >= self.min_overlap_words:
                # Remove the matching part from current text
                start_idx = largest_block.b
                end_idx = start_idx + largest_block.size
                remaining_words = current_words[:start_idx] + current_words[end_idx:]
                return ' '.join(remaining_words)
        
        return current_text
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text, normalized for comparison."""
        # Remove punctuation and convert to lowercase for comparison
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _has_better_sentence_structure(self, combined: str, current: str) -> bool:
        """Check if combined text has better sentence structure than current alone."""
        # Simple heuristics for better sentence structure
        combined_sentences = len(re.findall(r'[.!?]', combined))
        current_sentences = len(re.findall(r'[.!?]', current))
        
        # If combined has complete sentences and current doesn't, it's better
        return combined_sentences > current_sentences
    
    def _calculate_confidence(self, original: str, cleaned: str, duplications: List[Dict]) -> float:
        """Calculate confidence score for the detection."""
        if not duplications:
            return 1.0
        
        # Base confidence on detection types and overlap ratios
        total_confidence = 0.0
        for dup in duplications:
            total_confidence += dup['confidence']
        
        avg_confidence = total_confidence / len(duplications) if duplications else 1.0
        
        # Adjust based on text length change
        if original and cleaned:
            length_ratio = len(cleaned) / len(original)
            if length_ratio < 0.1:  # Removed too much text
                avg_confidence *= 0.7
            elif length_ratio < 0.3:  # Removed significant text
                avg_confidence *= 0.9
        
        return min(1.0, avg_confidence)
    
    def reset_context(self):
        """Reset the context buffer."""
        self.segment_buffer.clear()
        logger.debug("Cross-segment detector context reset")
    
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            **self.stats,
            'context_window': self.context_window,
            'overlap_threshold': self.overlap_threshold,
            'similarity_threshold': self.similarity_threshold,
            'segments_in_buffer': len(self.segment_buffer)
        }


# Convenience function for quick cross-segment detection
def detect_cross_segment_duplications(text: str, 
                                    timestamp: float,
                                    detector: Optional[CrossSegmentDetector] = None) -> str:
    """
    Quick cross-segment duplication detection function.
    
    Args:
        text: Text to process
        timestamp: Timestamp of the segment
        detector: Existing detector instance or None to create new one
        
    Returns:
        Cleaned text
    """
    if detector is None:
        detector = CrossSegmentDetector()
    
    result = detector.process_segment(text, timestamp)
    return result.cleaned_text 