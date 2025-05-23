"""Test caption scheduling functionality for ISKCON-Translate."""
import time
from ..logging_utils import get_logger

logger = get_logger(__name__)

def get_test_captions():
    """Return a list of test captions with their display times.
    
    Returns:
        list: List of tuples (caption_text, display_time_offset)
    """
    return [
        ("TEST 1: Overlay working", 1.0),
        ("TEST 2: Still alive!", 3.0),
        ("TEST 3: Quit with 'q'", 5.0),
    ]

def schedule_test_captions(caption_overlay, video_start_time):
    """Schedule test captions to be displayed at specific times.
    
    Args:
        caption_overlay: Instance of CaptionOverlay to add captions to
        video_start_time: Timestamp when the video started playing
        
    Returns:
        None
    """
    current_playback_time = time.time() - video_start_time
    test_offsets = [1.0, 3.0, 5.0]  # 1s, 3s, and 5s from now
    test_captions = get_test_captions()
    
    logger.info("\n=== SCHEDULING TEST CAPTIONS ===")
    logger.info(f"System time: {time.time()}")
    logger.info(f"Video start time: {video_start_time}")
    logger.info(f"Current playback time: {current_playback_time:.2f}s")
    
    # Schedule test captions relative to current playback time
    for (caption_text, _), offset in zip(test_captions, test_offsets):
        caption_time = current_playback_time + offset
        logger.info(f"\nAdding caption: '{caption_text}' at {offset:.1f}s from now")
        logger.info(f"  - Will appear at system time: {time.time() + offset:.2f}")
        logger.info(f"  - Relative to video start: {caption_time:.2f}s")
        
        caption_overlay.add_caption(
            text=caption_text,
            timestamp=caption_time,  # Absolute time when caption should appear
            duration=10.0,  # 10 seconds duration
            is_absolute=False  # Using relative timestamp from video start
        )
        logger.info(f"[TEST] Scheduled caption: '{caption_text}' for {caption_time:.2f}s from video start")
