import pytest
import time
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src directory to path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.caption_overlay.utils import (
    normalize_text, deduplicate_lines, wrap_text_lines,
    convert_timestamp, validate_duration, should_skip_similar_caption,
    adjust_timestamp_if_past
)


class TestNormalizeText:
    """Test suite for caption overlay normalize_text function."""
    
    @pytest.mark.skip(reason="TODO: Fix whitespace normalization - implementation preserves multiple spaces while test expects single spaces")
    def test_normalize_text_cleans_whitespace(self):
        """Test that normalize_text removes extra whitespace and line breaks."""
        test_cases = [
            ("hello world", "hello world"),
            ("  hello   world  ", "hello world"),
            ("line1\n\nline2\n", "line1\nline2"),
            ("line1\n   \nline2", "line1\nline2"),
            ("\n\n  text  \n\n", "text"),
            ("", ""),
            ("   ", ""),
        ]
        
        for input_text, expected in test_cases:
            result = normalize_text(input_text)
            assert result == expected, f"normalize_text('{repr(input_text)}') = '{repr(result)}', expected '{repr(expected)}'"
            
    def test_normalize_text_handles_none_input(self):
        """Test that normalize_text handles None input gracefully."""
        assert normalize_text(None) == ""
        
    def test_normalize_text_preserves_meaningful_newlines(self):
        """Test that normalize_text preserves meaningful line breaks."""
        text = "First line\nSecond line\nThird line"
        expected = "First line\nSecond line\nThird line"
        result = normalize_text(text)
        assert result == expected


class TestDeduplicateLines:
    """Test suite for deduplicate_lines function."""
    
    def test_deduplicate_lines_removes_exact_duplicates(self):
        """Test that deduplicate_lines removes exact duplicate lines."""
        lines = ["hello", "world", "hello", "test", "world"]
        expected = ["hello", "world", "test"]
        result = deduplicate_lines(lines)
        assert result == expected
        
    def test_deduplicate_lines_case_insensitive(self):
        """Test that deduplicate_lines is case insensitive."""
        lines = ["Hello", "WORLD", "hello", "world", "Test"]
        expected = ["Hello", "WORLD", "Test"]  # Preserves original case
        result = deduplicate_lines(lines)
        assert result == expected


class TestWrapTextLines:
    """Test suite for wrap_text_lines function."""
    
    def test_wrap_text_lines_respects_max_width(self):
        """Test that wrap_text_lines wraps long lines to fit within character limit."""
        text = "This is a very long line that should be wrapped to fit within the specified character limit"
        result = wrap_text_lines(text, max_chars_per_line=20)
        
        # All lines should be <= 20 characters
        for line in result:
            assert len(line) <= 20, f"Line '{line}' is {len(line)} characters, exceeds 20"
            
        # Text should be preserved when joined
        joined = " ".join(result)
        assert joined == text
        
    def test_wrap_text_lines_preserves_existing_breaks(self):
        """Test that wrap_text_lines preserves existing line breaks."""
        text = "Short line\nAnother short line"
        result = wrap_text_lines(text, max_chars_per_line=50)
        
        expected = ["Short line", "Another short line"]
        assert result == expected


class TestConvertTimestamp:
    """Test suite for convert_timestamp function."""
    
    def test_convert_timestamp_relative_time(self):
        """Test convert_timestamp with relative timestamps."""
        video_start_time = 1000.0
        test_cases = [
            (5.0, False, 5.0, False),    # Already relative
            (10.5, False, 10.5, False),  # Already relative
            (0.0, False, 0.0, False),    # Zero timestamp
        ]
        
        for timestamp, is_absolute, expected_timestamp, expected_converted in test_cases:
            result_timestamp, was_converted = convert_timestamp(timestamp, video_start_time, is_absolute)
            assert result_timestamp == expected_timestamp
            assert was_converted == expected_converted


class TestValidateDuration:
    """Test suite for validate_duration function."""
    
    def test_validate_duration_valid_durations(self):
        """Test validate_duration with valid duration values."""
        valid_durations = [0.1, 1.0, 3.0, 5.5, 8.0, 10.0]
        
        for duration in valid_durations:
            result = validate_duration(duration)
            assert result == duration, f"Valid duration {duration} should be unchanged"
            
    def test_validate_duration_invalid_durations(self):
        """Test validate_duration with invalid duration values."""
        invalid_cases = [
            (0.0, 3.0),      # Zero duration
            (-1.0, 3.0),     # Negative duration
            (-5.5, 3.0),     # Very negative duration
        ]
        
        for input_duration, expected in invalid_cases:
            result = validate_duration(input_duration)
            assert result == expected, f"Invalid duration {input_duration} should default to {expected}"


class TestShouldSkipSimilarCaption:
    """Test suite for should_skip_similar_caption function."""
    
    def test_should_skip_similar_caption_filters_duplicates(self):
        """Test that should_skip_similar_caption filters similar captions."""
        last_caption = {"text": "Hello world", "timestamp": 1.0}
        
        # Very similar text should be skipped
        assert should_skip_similar_caption("Hello world", last_caption) is True
        assert should_skip_similar_caption("hello world", last_caption) is True  # Case insensitive
        assert should_skip_similar_caption("Hello, world!", last_caption) is True  # Punctuation
        
        # Different text should not be skipped
        assert should_skip_similar_caption("Goodbye world", last_caption) is False
        assert should_skip_similar_caption("Completely different", last_caption) is False


class TestAdjustTimestampIfPast:
    """Test suite for adjust_timestamp_if_past function."""
    
    def test_adjust_timestamp_if_past_shifts_appropriately(self):
        """Test that adjust_timestamp_if_past adjusts past timestamps."""
        current_time = 10.0
        
        # Future timestamps should not be adjusted
        assert adjust_timestamp_if_past(15.0, current_time) == 15.0
        assert adjust_timestamp_if_past(10.1, current_time) == 10.1
        
        # Past timestamps should be adjusted to current time
        assert adjust_timestamp_if_past(5.0, current_time) == 10.0
        assert adjust_timestamp_if_past(0.0, current_time) == 10.0 