"""
Tests for cross-segment duplication detection.

These tests validate the cross-segment detector's ability to identify and remove
text duplications that cross caption boundaries, which is the core issue causing
multiple overlapping captions.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.text_processing.cross_segment_detector import (
    CrossSegmentDetector, detect_cross_segment_duplications
)


class TestCrossSegmentDetector:
    """Test cases for the cross-segment duplication detector."""
    
    def test_exact_overlap_detection(self):
        """Test detection of exact text overlaps between segments."""
        detector = CrossSegmentDetector()
        
        # Add first segment
        result1 = detector.process_segment("learn to do is instead of identifying with the", 1.0)
        assert result1.action_taken == "kept"
        assert result1.cleaned_text == "learn to do is instead of identifying with the"
        
        # Add overlapping segment (matches the log pattern)
        result2 = detector.process_segment("identifying with the mind, they learn to identify", 2.0)
        
        # Should detect and remove the overlap
        assert len(result2.duplications_found) > 0
        assert result2.action_taken in ["cleaned", "removed"]
        # Should remove "identifying with the" from the beginning
        assert "mind" in result2.cleaned_text and "learn" in result2.cleaned_text
        assert result2.cleaned_text != result2.original_text
    
    def test_high_similarity_removal(self):
        """Test removal of highly similar segments."""
        detector = CrossSegmentDetector()
        
        # Add first segment
        detector.process_segment("They learn to identify the mind", 1.0)
        
        # Add very similar segment
        result = detector.process_segment("They learn to identify the mind.", 2.0)
        
        # Should detect high similarity and remove/clean
        assert len(result.duplications_found) > 0
        assert result.action_taken in ["removed", "cleaned"]
        assert result.confidence > 0.8
    
    def test_sentence_continuation_detection(self):
        """Test detection of sentence continuations."""
        detector = CrossSegmentDetector()
        
        # Add incomplete sentence
        detector.process_segment("Krishna says, but they remain steady because there's", 1.0)
        
        # Add continuation
        result = detector.process_segment("no attachment to the results", 2.0)
        
        # Should be kept as it's a valid continuation
        assert result.action_taken in ["kept", "merged"]
        assert result.cleaned_text.strip() != ""
    
    def test_no_false_positives_different_content(self):
        """Test that different content is not flagged as duplication."""
        detector = CrossSegmentDetector()
        
        # Add first segment
        detector.process_segment("Krishna speaks of divine wisdom", 1.0)
        
        # Add completely different segment
        result = detector.process_segment("The devotees chant with devotion", 2.0)
        
        # Should be kept without modification
        assert result.action_taken == "kept"
        assert len(result.duplications_found) == 0
        assert result.cleaned_text == result.original_text
    
    def test_partial_overlap_cleaning(self):
        """Test cleaning of partial overlaps."""
        detector = CrossSegmentDetector()
        
        # Add first segment
        detector.process_segment("So many things can be happening in their life", 1.0)
        
        # Add segment with partial overlap
        result = detector.process_segment("So many things can be happening around them", 2.0)
        
        # Should detect overlap and clean
        assert len(result.duplications_found) > 0
        assert result.action_taken in ["cleaned", "removed"]
    
    def test_context_window_limit(self):
        """Test that context window limits are respected."""
        detector = CrossSegmentDetector(context_window=2)
        
        # Add multiple segments
        detector.process_segment("First segment", 1.0)
        detector.process_segment("Second segment", 2.0)
        detector.process_segment("Third segment", 3.0)
        
        # Add segment similar to first (should not detect due to window limit)
        result = detector.process_segment("First segment again", 4.0)
        
        # Should not detect similarity with first segment (outside window)
        assert result.action_taken == "kept"
    
    def test_minimum_overlap_threshold(self):
        """Test minimum overlap threshold requirements."""
        detector = CrossSegmentDetector(min_overlap_words=3)
        
        # Add first segment
        detector.process_segment("Krishna teaches divine wisdom", 1.0)
        
        # Add segment with small overlap (below threshold)
        result = detector.process_segment("divine love and compassion", 2.0)
        
        # Should not detect overlap (only 1 word overlap)
        assert result.action_taken == "kept"
        assert len(result.duplications_found) == 0
    
    def test_real_world_log_pattern(self):
        """Test with actual patterns from the logs."""
        detector = CrossSegmentDetector()
        
        # Simulate the exact pattern from logs
        segments = [
            ("learn to do is instead of identifying with the", 45.98),
            ("identifying with the mind, they learn to identify with the mind.", 47.97),
            ("They learn to identify the mind.", 50.02)
        ]
        
        results = []
        for text, timestamp in segments:
            result = detector.process_segment(text, timestamp)
            results.append(result)
        
        # First segment should be kept
        assert results[0].action_taken == "kept"
        
        # Second segment should detect overlap with first
        assert len(results[1].duplications_found) > 0
        assert results[1].action_taken in ["cleaned", "removed"]
        
        # Third segment should detect overlap with previous
        assert len(results[2].duplications_found) > 0
        assert results[2].action_taken in ["cleaned", "removed"]
    
    def test_performance_timing(self):
        """Test that processing is fast enough for real-time use."""
        detector = CrossSegmentDetector()
        
        # Add some context
        for i in range(5):
            detector.process_segment(f"Context segment {i} with some text", float(i))
        
        # Test processing time
        result = detector.process_segment("New segment to process for timing test", 6.0)
        
        # Should process in under 10ms for real-time use
        assert result.processing_time < 0.01
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        detector = CrossSegmentDetector()
        
        # Test empty string
        result1 = detector.process_segment("", 1.0)
        assert result1.action_taken == "kept"
        assert result1.cleaned_text == ""
        
        # Test whitespace only
        result2 = detector.process_segment("   \n\t  ", 2.0)
        assert result2.action_taken == "kept"
        assert result2.cleaned_text == "   \n\t  "
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        detector = CrossSegmentDetector()
        
        # Process some segments
        detector.process_segment("First segment", 1.0)
        detector.process_segment("First segment duplicate", 2.0)  # Should be detected
        detector.process_segment("Different content", 3.0)
        
        stats = detector.get_stats()
        
        assert stats['total_processed'] == 3
        assert stats['duplications_found'] >= 1
        assert stats['segments_in_buffer'] <= 3
    
    def test_context_reset(self):
        """Test context buffer reset functionality."""
        detector = CrossSegmentDetector()
        
        # Add some segments
        detector.process_segment("Segment 1", 1.0)
        detector.process_segment("Segment 2", 2.0)
        
        assert len(detector.segment_buffer) == 2
        
        # Reset context
        detector.reset_context()
        
        assert len(detector.segment_buffer) == 0
        
        # New segments should not detect duplications with reset context
        result = detector.process_segment("Segment 1", 3.0)
        assert result.action_taken == "kept"
    
    def test_convenience_function(self):
        """Test the convenience function for quick processing."""
        # Test without existing detector
        cleaned1 = detect_cross_segment_duplications("Test text", 1.0)
        assert cleaned1 == "Test text"
        
        # Test with existing detector
        detector = CrossSegmentDetector()
        detector.process_segment("Previous text", 1.0)
        
        cleaned2 = detect_cross_segment_duplications("Previous text again", 2.0, detector)
        # Should detect similarity and clean/remove
        assert cleaned2 != "Previous text again" or cleaned2 == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 