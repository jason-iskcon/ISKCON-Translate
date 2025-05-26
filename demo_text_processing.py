#!/usr/bin/env python3
"""
Demo script for the new intelligent text processing capabilities.

This script demonstrates the profanity filtering and repetition detection
features that can be integrated into the real-time caption system.
"""

import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.text_processing import (
    ProfanityFilter, FilterLevel, ReplacementStrategy,
    RepetitionDetector, filter_profanity, remove_repetitions
)


def demo_profanity_filtering():
    """Demonstrate profanity filtering capabilities."""
    print("🔒 PROFANITY FILTERING DEMO")
    print("=" * 50)
    
    # Create filter with different settings
    filter_moderate = ProfanityFilter(
        filter_level=FilterLevel.MODERATE,
        replacement_strategy=ReplacementStrategy.BEEP
    )
    
    filter_asterisk = ProfanityFilter(
        filter_level=FilterLevel.MODERATE,
        replacement_strategy=ReplacementStrategy.ASTERISK
    )
    
    # Test cases
    test_texts = [
        "This is a clean sentence about Krishna's teachings",
        "What the hell is going on here",
        "That's some fucking bullshit right there",
        "Krishna speaks of hell and heaven in religious context",
        "This damn thing is broken again"
    ]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        
        # BEEP strategy
        result_beep = filter_moderate.filter_text(text)
        print(f"BEEP:     {result_beep.filtered_text}")
        
        # Asterisk strategy
        result_asterisk = filter_asterisk.filter_text(text)
        print(f"Asterisk: {result_asterisk.filtered_text}")
        
        if result_beep.detections:
            print(f"Detected: {len(result_beep.detections)} issues")


def demo_repetition_detection():
    """Demonstrate repetition detection capabilities."""
    print("\n\n🔄 REPETITION DETECTION DEMO")
    print("=" * 50)
    
    detector = RepetitionDetector()
    
    test_texts = [
        "the the the cat sat on the mat",
        "I think I think this is good",
        "um um um well you know",
        "Hare Krishna Hare Krishna",  # Religious - should be preserved
        "very very important message",
        "you know you know what I mean",
        "thank you thank you very much"
    ]
    
    for text in test_texts:
        result = detector.detect_and_remove_repetitions(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned:  {result.cleaned_text}")
        if result.repetitions_found:
            print(f"Removed:  {len(result.repetitions_found)} repetitions")
            for rep in result.repetitions_found:
                print(f"  - {rep['type']}: {rep.get('word', rep.get('phrase', 'context'))}")


def demo_combined_processing():
    """Demonstrate combined profanity filtering and repetition detection."""
    print("\n\n🔧 COMBINED PROCESSING DEMO")
    print("=" * 50)
    
    profanity_filter = ProfanityFilter()
    repetition_detector = RepetitionDetector()
    
    test_texts = [
        "this fucking fucking text has has repetitions",
        "the the damn thing is is broken broken",
        "Krishna Krishna teaches about divine divine love",
        "um um this shit shit is really really bad"
    ]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        
        # Step 1: Remove repetitions
        rep_result = repetition_detector.detect_and_remove_repetitions(text)
        print(f"Step 1:   {rep_result.cleaned_text}")
        
        # Step 2: Filter profanity
        prof_result = profanity_filter.filter_text(rep_result.cleaned_text)
        print(f"Final:    {prof_result.filtered_text}")
        
        # Show processing stats
        total_time = rep_result.processing_time + prof_result.processing_time
        print(f"Time:     {total_time*1000:.2f}ms")


def demo_convenience_functions():
    """Demonstrate convenience functions for quick processing."""
    print("\n\n⚡ CONVENIENCE FUNCTIONS DEMO")
    print("=" * 50)
    
    test_text = "this fucking fucking text has has many many repetitions"
    
    print(f"Original: {test_text}")
    
    # Quick profanity filtering
    clean_profanity = filter_profanity(test_text)
    print(f"No profanity: {clean_profanity}")
    
    # Quick repetition removal
    clean_repetitions = remove_repetitions(test_text)
    print(f"No repetitions: {clean_repetitions}")
    
    # Combined (manual)
    combined = filter_profanity(remove_repetitions(test_text))
    print(f"Both cleaned: {combined}")


def demo_religious_context():
    """Demonstrate religious context awareness."""
    print("\n\n🕉️  RELIGIOUS CONTEXT DEMO")
    print("=" * 50)
    
    filter_instance = ProfanityFilter()
    
    religious_texts = [
        "Krishna speaks of hell and heaven",
        "Damn the ignorance, seek divine wisdom",
        "In hell or heaven, God's love prevails",
        "Regular text with damn profanity"
    ]
    
    for text in religious_texts:
        result = filter_instance.filter_text(text)
        print(f"\nOriginal: {text}")
        print(f"Filtered: {result.filtered_text}")
        if result.detections:
            print(f"Filtered: {len(result.detections)} items")
        else:
            print("Preserved: Religious context detected")


if __name__ == "__main__":
    print("🎯 INTELLIGENT TEXT PROCESSING DEMO")
    print("This demo showcases the new text processing capabilities")
    print("for real-time caption enhancement in ISKCON content.\n")
    
    try:
        demo_profanity_filtering()
        demo_repetition_detection()
        demo_combined_processing()
        demo_convenience_functions()
        demo_religious_context()
        
        print("\n\n✅ DEMO COMPLETE!")
        print("These features are now ready for integration into the")
        print("real-time caption system for immediate improvement.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 