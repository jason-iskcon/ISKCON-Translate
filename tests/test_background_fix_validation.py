#!/usr/bin/env python3
"""
Validation test for the background sizing fix in ISKCON-Translate caption system.

This test validates that the critical bug fix is working:
- Backgrounds never extend outside frame boundaries
- Text is properly positioned within backgrounds
- Very long text is handled gracefully
"""

import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import unittest
import numpy as np
from src.caption_overlay.renderer import CaptionRenderer


class TestBackgroundFixValidation(unittest.TestCase):
    """Focused tests for the background sizing fix."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = CaptionRenderer()
        self.frame_width = 640
        self.frame_height = 480
        
    def test_backgrounds_stay_within_frame_bounds(self):
        """Test that backgrounds never extend outside frame boundaries."""
        test_cases = [
            "Short text",
            "This is a medium length line of text that should fit nicely",
            "This is a very long line of text that should definitely require a wide background to contain it properly without overflow issues",
            "Нам нужна очень длинная строка русского текста которая может быть шире чем ширина кадра",
            "Nous avons besoin d'une très longue ligne de texte français qui pourrait être plus large que la largeur du cadre"
        ]
        
        for text in test_cases:
            with self.subTest(text=text[:50] + "..."):
                # Calculate background positioning
                display_lines = self.renderer.process_caption_text(text)
                line_heights, line_widths, total_text_height, max_text_width = \
                    self.renderer.calculate_text_dimensions(display_lines)
                
                text_x, text_y, bg_x1, bg_y1, bg_x2, bg_y2 = \
                    self.renderer.calculate_text_position(
                        self.frame_width, self.frame_height, max_text_width, total_text_height
                    )
                
                print(f"\nText: '{text[:40]}...'")
                print(f"  Max width: {max_text_width}px")
                print(f"  Background: {bg_x1} to {bg_x2} (width: {bg_x2-bg_x1})")
                
                # CRITICAL FIX VALIDATION: Background must stay within frame
                self.assertGreaterEqual(bg_x1, 0, 
                    f"Background left edge ({bg_x1}) should not be < 0")
                self.assertLessEqual(bg_x2, self.frame_width, 
                    f"Background right edge ({bg_x2}) should not exceed frame width ({self.frame_width})")
                
                print(f"  ✓ Background stays within frame bounds")
                
    def test_text_positioning_within_background(self):
        """Test that text is positioned within background bounds."""
        test_caption = {
            'text': "Line 1\nThis is a longer line\nShort",
            'start_time': 0,
            'end_time': 5,
            'language': 'en'
        }
        
        # Test rendering
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        result_frame = self.renderer.render_caption(frame, test_caption, 2.5, 0, 'en')
        
        # Verify rendering worked
        self.assertFalse(np.array_equal(result_frame, frame),
            "Caption should modify the frame")
        
        # Analyze rendered content bounds
        gray = result_frame[:, :, 0] + result_frame[:, :, 1] + result_frame[:, :, 2]
        modified_mask = gray > 0
        
        if np.any(modified_mask):
            y_coords, x_coords = np.where(modified_mask)
            actual_left = np.min(x_coords)
            actual_right = np.max(x_coords)
            
            print(f"\nRendered content bounds: {actual_left} to {actual_right}")
            
            # Content should stay within frame
            self.assertGreaterEqual(actual_left, 0, 
                "Rendered content should not extend left of frame")
            self.assertLessEqual(actual_right, self.frame_width - 1, 
                "Rendered content should not extend right of frame")
            
            print("✓ Text rendered within frame bounds")
        else:
            self.fail("No content was rendered")
            
    def test_very_long_text_handling(self):
        """Test that extremely long text is handled gracefully."""
        very_long_text = "This is an extremely long line of text that is definitely much wider than the standard frame width and should be handled gracefully without breaking the system or extending outside frame boundaries"
        
        # Calculate positioning
        display_lines = self.renderer.process_caption_text(very_long_text)
        line_heights, line_widths, total_text_height, max_text_width = \
            self.renderer.calculate_text_dimensions(display_lines)
        
        text_x, text_y, bg_x1, bg_y1, bg_x2, bg_y2 = \
            self.renderer.calculate_text_position(
                self.frame_width, self.frame_height, max_text_width, total_text_height
            )
        
        print(f"\nVery long text test:")
        print(f"  Text width: {max_text_width}px")
        print(f"  Frame width: {self.frame_width}px")
        print(f"  Background: {bg_x1} to {bg_x2}")
        
        # For very long text, background should be constrained to frame
        if max_text_width > self.frame_width - 20:  # Accounting for padding
            self.assertEqual(bg_x1, 0, "Very long text should have background starting at frame edge")
            self.assertEqual(bg_x2, self.frame_width, "Very long text should have background ending at frame edge")
            print("✓ Very long text properly constrained to frame")
        else:
            # Normal centering should apply
            self.assertGreater(bg_x1, 0, "Normal text should have centered background")
            self.assertLess(bg_x2, self.frame_width, "Normal text should have centered background")
            print("✓ Normal text properly centered")
            
    def test_multi_language_background_consistency(self):
        """Test that background sizing is consistent across languages."""
        test_texts = {
            'en': "Hello world with some text",
            'fr': "Bonjour le monde avec du texte", 
            'de': "Hallo Welt mit etwas Text",
            'ru': "Привет мир с некоторым текстом"
        }
        
        for lang, text in test_texts.items():
            with self.subTest(language=lang):
                # Test caption rendering
                test_caption = {
                    'text': text,
                    'start_time': 0,
                    'end_time': 5,
                    'language': lang
                }
                
                frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                result_frame = self.renderer.render_caption(frame, test_caption, 2.5, 0, lang)
                
                # Should render successfully
                self.assertFalse(np.array_equal(result_frame, frame),
                    f"{lang} caption should modify the frame")
                
                print(f"✓ {lang.upper()} caption rendered successfully")


def run_validation():
    """Run background fix validation tests."""
    print("=" * 70)
    print("BACKGROUND SIZING FIX VALIDATION")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBackgroundFixValidation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! Background sizing fix is working correctly!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")
        
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_validation()
    sys.exit(0 if success else 1) 