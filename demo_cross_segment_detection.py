#!/usr/bin/env python3
"""
Demo script for cross-segment duplication detection.

This script demonstrates how the cross-segment detector handles the exact
text duplication patterns found in the ISKCON video transcription logs.
"""

import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.text_processing import CrossSegmentDetector


def demo_real_world_patterns():
    """Demonstrate with actual patterns from the logs."""
    print("🔍 CROSS-SEGMENT DUPLICATION DETECTION DEMO")
    print("=" * 60)
    print("Demonstrating with ACTUAL patterns from your logs:\n")
    
    detector = CrossSegmentDetector(
        context_window=5,
        overlap_threshold=0.3,
        similarity_threshold=0.7
    )
    
    # Exact patterns from the logs that were causing 18 active captions
    log_segments = [
        ("learn to do is instead of identifying with the", 45.98),
        ("identifying with the mind, they learn to identify with the mind.", 47.97),
        ("They learn to identify the mind.", 50.02),
        ("And therefore,", 53.13),
        ("And therefore, so many things can be happened.", 54.06),
        ("Many things can be happening around them.", 56.05),
        ("So many things can be happening in their life.", 58.10),
        ("So many.", 58.39),
        ("in their lives, speed the procure.", 60.14),
        ("Pragya, Pratisthi, Pa, Krishna says.", 62.14),
        ("Krishna says, but they remain steady because there's", 64.18)
    ]
    
    print("BEFORE Cross-Segment Detection:")
    print("(This is what was causing 18 active captions)")
    for i, (text, timestamp) in enumerate(log_segments, 1):
        print(f"  {i:2d}. [{timestamp:5.2f}s] '{text}'")
    
    print("\nAFTER Cross-Segment Detection:")
    print("(Duplications removed, clean captions)")
    
    kept_captions = []
    total_duplications = 0
    
    for i, (text, timestamp) in enumerate(log_segments, 1):
        result = detector.process_segment(text, timestamp)
        
        if result.cleaned_text.strip():
            kept_captions.append((result.cleaned_text, timestamp, result.action_taken))
            status = f"✅ {result.action_taken.upper()}"
        else:
            status = "❌ REMOVED"
        
        total_duplications += len(result.duplications_found)
        
        print(f"  {i:2d}. [{timestamp:5.2f}s] {status}")
        print(f"      Original: '{text}'")
        if result.cleaned_text.strip():
            print(f"      Cleaned:  '{result.cleaned_text}'")
        else:
            print(f"      Cleaned:  [REMOVED - duplicate content]")
        
        if result.duplications_found:
            for dup in result.duplications_found:
                print(f"      🔍 Detected: {dup['type']} (confidence: {dup['confidence']:.2f})")
        print()
    
    print("=" * 60)
    print("SUMMARY:")
    print(f"📊 Original captions: {len(log_segments)}")
    print(f"📊 Kept captions: {len(kept_captions)}")
    print(f"📊 Duplications removed: {total_duplications}")
    print(f"📊 Reduction: {(1 - len(kept_captions)/len(log_segments))*100:.1f}%")
    
    print("\nFINAL CLEAN CAPTIONS:")
    for i, (text, timestamp, action) in enumerate(kept_captions, 1):
        print(f"  {i}. [{timestamp:5.2f}s] '{text}' ({action})")


def demo_overlap_types():
    """Demonstrate different types of overlaps detected."""
    print("\n\n🎯 OVERLAP DETECTION TYPES")
    print("=" * 60)
    
    detector = CrossSegmentDetector()
    
    test_cases = [
        # Exact word overlap
        ("Exact Word Overlap", [
            ("Krishna teaches divine wisdom and love", 1.0),
            ("divine wisdom and love brings peace", 2.0)
        ]),
        
        # High similarity
        ("High Similarity", [
            ("The devotees chant with devotion", 3.0),
            ("The devotees chant with great devotion", 4.0)
        ]),
        
        # Substantial word overlap
        ("Word Overlap", [
            ("Many things can be happening in their life", 5.0),
            ("So many things can be happening around them", 6.0)
        ]),
        
        # No overlap (should be kept)
        ("No Overlap", [
            ("Krishna speaks of transcendence", 7.0),
            ("Devotees practice meditation daily", 8.0)
        ])
    ]
    
    for test_name, segments in test_cases:
        print(f"\n{test_name}:")
        detector.reset_context()  # Reset for each test
        
        for text, timestamp in segments:
            result = detector.process_segment(text, timestamp)
            print(f"  [{timestamp:.1f}s] '{text}'")
            print(f"    → {result.action_taken}: '{result.cleaned_text}'")
            if result.duplications_found:
                for dup in result.duplications_found:
                    print(f"    🔍 {dup['type']}: {dup.get('overlap_ratio', dup.get('similarity', 0)):.2f}")


def demo_performance():
    """Demonstrate processing performance."""
    print("\n\n⚡ PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    
    detector = CrossSegmentDetector()
    
    # Process many segments to show performance
    import time
    start_time = time.perf_counter()
    
    for i in range(100):
        text = f"This is test segment number {i} with some content"
        result = detector.process_segment(text, float(i))
    
    total_time = time.perf_counter() - start_time
    avg_time = total_time / 100
    
    print(f"📊 Processed 100 segments in {total_time*1000:.2f}ms")
    print(f"📊 Average per segment: {avg_time*1000:.3f}ms")
    print(f"📊 Real-time capable: {'✅ YES' if avg_time < 0.01 else '❌ NO'}")
    
    # Show stats
    stats = detector.get_stats()
    print(f"📊 Total processed: {stats['total_processed']}")
    print(f"📊 Duplications found: {stats['duplications_found']}")
    print(f"📊 Context buffer size: {stats['segments_in_buffer']}")


if __name__ == "__main__":
    print("🎯 ISKCON CROSS-SEGMENT DUPLICATION DETECTION")
    print("Solving the caption boundary duplication problem")
    print()
    
    try:
        demo_real_world_patterns()
        demo_overlap_types()
        demo_performance()
        
        print("\n\n✅ SOLUTION READY!")
        print("The cross-segment detector is now integrated into the video runner")
        print("and will automatically clean up text duplications in real-time.")
        print("\nThis should solve the issue of 18 active captions with duplicate content!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 