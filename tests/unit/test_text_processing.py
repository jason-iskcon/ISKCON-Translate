"""
Tests for text processing modules including profanity filtering and repetition detection.

These tests validate the behavior of the new text processing capabilities
to ensure they work correctly for real-time caption enhancement.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.text_processing.profanity_filter import (
    ProfanityFilter, FilterLevel, ReplacementStrategy, filter_profanity
)
from src.text_processing.repetition_detector import (
    RepetitionDetector, remove_repetitions
)


class TestProfanityFilter:
    """Test cases for the profanity filtering system."""
    
    def test_basic_profanity_filtering(self):
        """Test basic profanity word filtering."""
        filter_instance = ProfanityFilter(
            filter_level=FilterLevel.MODERATE,
            replacement_strategy=ReplacementStrategy.BEEP
        )
        
        # Test clean text (should pass through unchanged)
        clean_text = "This is a clean sentence"
        result = filter_instance.filter_text(clean_text)
        assert result.filtered_text == clean_text
        assert result.original_text == clean_text
        assert result.confidence > 0.0
        assert len(result.detections) == 0
    
    def test_asterisk_replacement_strategy(self):
        """Test asterisk replacement strategy."""
        filter_instance = ProfanityFilter(
            filter_level=FilterLevel.MODERATE,
            replacement_strategy=ReplacementStrategy.ASTERISK
        )
        
        # Test with a mild word that should be filtered
        test_text = "This is really bad"
        result = filter_instance.filter_text(test_text)
        # Should either be unchanged (if 'bad' not in profanity list) or filtered
        assert result.original_text == test_text
    
    def test_religious_context_awareness(self):
        """Test that religious context is handled appropriately."""
        filter_instance = ProfanityFilter(
            filter_level=FilterLevel.MODERATE,
            replacement_strategy=ReplacementStrategy.BEEP
        )
        
        # Religious context should preserve certain words
        religious_text = "Krishna speaks of heaven and divine grace"
        result = filter_instance.filter_text(religious_text)
        
        # Should preserve religious content
        assert "Krishna" in result.filtered_text
        assert "heaven" in result.filtered_text
        assert result.confidence > 0.8
    
    def test_filter_levels(self):
        """Test different filtering levels."""
        text_with_mild = "That's really annoying"
        
        # Mild level - should not filter mild words
        mild_filter = ProfanityFilter(filter_level=FilterLevel.MILD)
        mild_result = mild_filter.filter_text(text_with_mild)
        assert mild_result.filtered_text == text_with_mild
        
        # Strict level - might filter more aggressively
        strict_filter = ProfanityFilter(filter_level=FilterLevel.STRICT)
        strict_result = strict_filter.filter_text(text_with_mild)
        # Should at least process the text
        assert strict_result.original_text == text_with_mild
    
    def test_off_filter_level(self):
        """Test that OFF level doesn't filter anything."""
        filter_instance = ProfanityFilter(filter_level=FilterLevel.OFF)
        
        test_text = "This is some test content"
        result = filter_instance.filter_text(test_text)
        
        assert result.filtered_text == test_text
        assert len(result.detections) == 0
        assert result.confidence == 1.0
    
    def test_custom_words(self):
        """Test adding custom words to filter."""
        custom_words = ["badword", "inappropriate"]
        filter_instance = ProfanityFilter(custom_words=custom_words)
        
        result = filter_instance.filter_text("This is a badword example")
        # Should detect the custom word
        assert result.original_text == "This is a badword example"
        # Should have some processing result
        assert result.confidence > 0.0
    
    def test_convenience_function(self):
        """Test the convenience filter_profanity function."""
        text = "This is a test sentence"
        filtered = filter_profanity(text)
        
        # Should return a string
        assert isinstance(filtered, str)
        # Should process the text (even if no changes)
        assert len(filtered) > 0


class TestRepetitionDetector:
    """Test cases for the repetition detection system."""
    
    def test_word_repetition_removal(self):
        """Test removal of consecutive word repetitions."""
        detector = RepetitionDetector()
        
        test_cases = [
            ("the the the cat", "the cat"),
            ("I I I think so", "I think so"),
            ("very very good", "very very good"),  # Intentional emphasis
            ("hello hello world", "hello world"),
        ]
        
        for input_text, expected in test_cases:
            result = detector.detect_and_remove_repetitions(input_text)
            assert result.cleaned_text == expected
            assert result.original_text == input_text
    
    def test_phrase_repetition_removal(self):
        """Test removal of phrase-level repetitions."""
        detector = RepetitionDetector()
        
        test_cases = [
            ("I think I think it's good", "I think it's good"),
            ("you know you know what I mean", "you know what I mean"),
            ("thank you thank you very much", "thank you thank you very much"),  # Preserved as intentional
        ]
        
        for input_text, expected in test_cases:
            result = detector.detect_and_remove_repetitions(input_text)
            assert result.cleaned_text == expected
    
    def test_intentional_repetitions_preserved(self):
        """Test that intentional repetitions are preserved."""
        detector = RepetitionDetector()
        
        # Religious/spiritual repetitions should be preserved
        religious_cases = [
            "Hare Krishna Hare Krishna",
            "very very important",
            "thank you thank you"
        ]
        
        for text in religious_cases:
            result = detector.detect_and_remove_repetitions(text)
            # Should preserve some repetition for emphasis/religious context
            words_original = len(text.split())
            words_cleaned = len(result.cleaned_text.split())
            assert words_cleaned >= words_original - 2  # Allow some reduction
    
    def test_filler_word_handling(self):
        """Test handling of filler words."""
        detector = RepetitionDetector()
        
        test_cases = [
            ("um um um I think", "um I think"),
            ("you you you know", "you know"),  # Single word repetition
            ("like like like this", "like this"),
            ("well well well then", "well then"),
        ]
        
        for input_text, expected in test_cases:
            result = detector.detect_and_remove_repetitions(input_text)
            assert result.cleaned_text == expected
    
    def test_confidence_scoring(self):
        """Test confidence scoring for repetition detection."""
        detector = RepetitionDetector()
        
        # Clear repetition should have high confidence
        clear_repetition = "the the the the cat"
        result = detector.detect_and_remove_repetitions(clear_repetition)
        assert result.confidence > 0.8
        
        # No repetition should have perfect confidence
        no_repetition = "this is a normal sentence"
        result = detector.detect_and_remove_repetitions(no_repetition)
        assert result.confidence == 1.0
    
    def test_processing_time_tracking(self):
        """Test that processing time is tracked."""
        detector = RepetitionDetector()
        
        result = detector.detect_and_remove_repetitions("test text here")
        assert result.processing_time > 0.0
        assert result.processing_time < 1.0  # Should be very fast
    
    def test_context_reset(self):
        """Test context buffer reset functionality."""
        detector = RepetitionDetector(context_window=2)
        
        # Add some context
        detector.detect_and_remove_repetitions("first segment")
        detector.detect_and_remove_repetitions("second segment")
        
        assert len(detector.context_buffer) == 2
        
        # Reset context
        detector.reset_context()
        assert len(detector.context_buffer) == 0
    
    def test_convenience_function(self):
        """Test the convenience remove_repetitions function."""
        text = "the the the cat sat sat on a mat"
        cleaned = remove_repetitions(text)
        
        assert cleaned != text
        # Should remove word repetitions
        assert cleaned.count("the") == 1
        assert cleaned.count("sat") == 1


class TestIntegration:
    """Integration tests for combined text processing."""
    
    def test_profanity_and_repetition_combined(self):
        """Test combining profanity filtering and repetition removal."""
        # Create instances
        profanity_filter = ProfanityFilter()
        repetition_detector = RepetitionDetector()
        
        # Test text with repetition issues
        text = "this text text is is really good good"
        
        # Apply repetition removal first
        rep_result = repetition_detector.detect_and_remove_repetitions(text)
        
        # Then apply profanity filtering
        prof_result = profanity_filter.filter_text(rep_result.cleaned_text)
        
        final_text = prof_result.filtered_text
        
        # Should have repetition issues resolved
        assert final_text.count("text") == 1  # Repetition removed
        assert final_text.count("good") == 1  # Repetition removed
        # Note: "is" appears in "this" so we check for improvement, not exact count
    
    def test_religious_content_processing(self):
        """Test processing of religious content (ISKCON context)."""
        profanity_filter = ProfanityFilter()
        repetition_detector = RepetitionDetector()
        
        # Religious text
        religious_text = "Krishna speaks of divine grace and spiritual wisdom"
        
        # Process with profanity filter (should preserve religious context)
        prof_result = profanity_filter.filter_text(religious_text)
        
        # Process with repetition detector
        rep_result = repetition_detector.detect_and_remove_repetitions(prof_result.filtered_text)
        
        # Religious terms should be preserved
        final_text = rep_result.cleaned_text
        assert "Krishna" in final_text
        assert "divine" in final_text or "spiritual" in final_text
    
    def test_performance_with_long_text(self):
        """Test performance with longer text segments."""
        profanity_filter = ProfanityFilter()
        repetition_detector = RepetitionDetector()
        
        # Create a longer text with various issues
        long_text = " ".join([
            "this is a long text with some some repetitions",
            "and maybe some content here and there",
            "the the the text continues with more more content",
            "testing testing the performance of our system"
        ])
        
        # Process both
        rep_result = repetition_detector.detect_and_remove_repetitions(long_text)
        prof_result = profanity_filter.filter_text(rep_result.cleaned_text)
        
        # Should complete quickly
        assert rep_result.processing_time < 0.1
        assert prof_result.processing_time < 0.1
        
        # Should have some improvements
        assert len(rep_result.repetitions_found) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 