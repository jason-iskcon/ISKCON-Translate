"""Utility functions for caption processing and text handling."""
import time
import logging

# Import with try-except to handle both direct execution and module import
try:
    from ..logging_utils import get_logger
    from ..text_utils import text_similarity
except ImportError:
    from src.logging_utils import get_logger
    from src.text_utils import text_similarity

logger = get_logger(__name__)

def normalize_text(text):
    """Normalize text by cleaning up whitespace and line breaks.
    
    Args:
        text: The text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Clean up the text (remove duplicate lines, extra spaces)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def deduplicate_lines(lines):
    """Remove duplicate lines while preserving order and ignoring case differences.
    
    Args:
        lines: List of text lines
        
    Returns:
        list: Deduplicated lines preserving original case
    """
    seen = set()
    unique_lines = []
    
    for line in lines:
        # Normalize whitespace and convert to lowercase for comparison
        normalized = ' '.join(line.lower().split())
        if normalized and normalized not in seen:
            unique_lines.append(line)  # Keep original line with original case
            seen.add(normalized)
    
    return unique_lines

def wrap_text_lines(text, max_chars_per_line=60):
    """Wrap long lines of text to fit within character limit.
    
    Args:
        text: The text to wrap
        max_chars_per_line: Maximum characters per line
        
    Returns:
        list: List of wrapped text lines
    """
    # Split into existing lines first
    lines = text.split('\n')
    wrapped_lines = []
    
    for line in lines:
        if len(line) <= max_chars_per_line:
            wrapped_lines.append(line)
        else:
            # Wrap long lines
            words = line.split()
            current_line = []
            
            for word in words:
                if current_line and len(' '.join(current_line + [word])) > max_chars_per_line:
                    wrapped_lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)
            
            if current_line:  # Add the last line
                wrapped_lines.append(' '.join(current_line))
    
    return wrapped_lines

def convert_timestamp(timestamp, video_start_time, is_absolute=False):
    """Convert timestamp to relative time from video start.
    
    Args:
        timestamp: The timestamp to convert
        video_start_time: The video's start time
        is_absolute: Whether the timestamp is absolute system time
        
    Returns:
        tuple: (converted_timestamp, was_converted)
    """
    original_timestamp = timestamp
    was_converted = False
    
    if is_absolute:
        # Add tolerance window to distinguish between legitimate absolute wall-clock times
        # and elapsed-time timestamps. Use a much larger threshold for real absolute times.
        tolerance_window = 1000000  # 1M seconds (way in future) - for real absolute wall-clock times
        
        # Check if this is a real absolute wall-clock time (very large number like 1732321234.5)
        current_wall_time = time.time()
        if timestamp > current_wall_time - 86400 and timestamp < current_wall_time + 3600:
            # This looks like a real wall-clock timestamp (within last 24h to next 1h)
            relative_time = timestamp - video_start_time
            logger.debug(f"[TIMING] Converted wall-clock absolute {timestamp:.2f} to relative {relative_time:.2f}s")
            timestamp = relative_time
            was_converted = True
        else:
            # This is likely already an elapsed time or media PTS, treat as relative
            logger.debug(f"[TIMING] Timestamp {timestamp:.2f} appears to be elapsed time, keeping as-is")
            is_absolute = False
    
    # Ensure timestamp is reasonable (but allow small negative values for near-real-time captions)
    if timestamp < -1.0:  # Allow up to 1 second before start for timing tolerance
        logger.warning(f"[TIMING] Timestamp {timestamp:.2f}s is too far in past, adjusting to 0")
        timestamp = 0
        was_converted = True
    
    return timestamp, was_converted

def validate_duration(duration):
    """Validate and normalize caption duration.
    
    Args:
        duration: Duration in seconds
        
    Returns:
        float: Validated duration
    """
    if duration <= 0:
        logger.warning(f"Invalid duration {duration:.1f}s, using default 3.0s")
        return 3.0
    elif duration > 10.0:  # Arbitrary max duration
        logger.warning(f"Unusually long duration {duration:.1f}s, capping at 10.0s")
        return 10.0
    
    return duration

def should_skip_similar_caption(text, last_caption, similarity_threshold=0.8):
    """Check if a caption should be skipped due to similarity with the last one.
    
    Args:
        text: New caption text
        last_caption: Last caption dictionary
        similarity_threshold: Similarity threshold (0.0-1.0)
        
    Returns:
        bool: True if caption should be skipped
    """
    if not last_caption:
        return False
    
    similarity = text_similarity(text, last_caption['text'])
    
    if similarity > similarity_threshold:
        logger.debug(f"[DEDUPE] Skipping similar caption (score: {similarity:.2f})")
        logger.trace(
            f"[DEDUPE] Details | "
            f"New: '{text[:50]}{'...' if len(text) > 50 else ''}' | "
            f"Prev: '{last_caption['text'][:50]}{'...' if len(last_caption['text']) > 50 else ''}'"
        )
        return True
    
    return False

def adjust_timestamp_if_past(timestamp, current_relative_time):
    """Adjust timestamp if it's in the past to show immediately.
    
    Args:
        timestamp: The caption timestamp
        current_relative_time: Current relative time
        
    Returns:
        float: Adjusted timestamp
    """
    if timestamp < current_relative_time:
        time_diff = current_relative_time - timestamp
        if time_diff > 1.0:  # Only log if more than 1 second adjustment
            logger.debug(f"[CAPTION] Adjusting timestamp by {time_diff:.2f}s (was in the past)")
        return current_relative_time  # Show immediately
    
    return timestamp 