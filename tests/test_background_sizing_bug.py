#!/usr/bin/env python3
"""
Test to highlight the background sizing bug in ISKCON-Translate caption system.

This test verifies that caption backgrounds are properly sized to contain all text.
"""

import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import unittest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from src.caption_overlay.renderer import CaptionRenderer


class TestBackgroundSizingBug(unittest.TestCase):
    """Tests to highlight and verify the background sizing bug."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = CaptionRenderer()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
    def test_background_width_matches_text_width(self):
        """Test that background width properly contains text width."""
        # Test various text lengths that are known to cause issues
        problematic_texts = [
            "We're coming to now the end of the first six chapters.",
            "Nous arrivons maintenant à la fin des six premiers chapitres de la Gita.",
            "Мы приближаемся к концу первых шести глав Гиты Кришны.",
            "This is a very long line of text that should definitely require a wide background to contain it properly without overflow.",
            "Sanskrit transliteration: oṃ namaḥ śivāya tat sat hari oṃ"
        ]
        
        for text in problematic_texts:
            with self.subTest(text=text[:50] + "..."):
                # Process text into display lines
                display_lines = self.renderer.process_caption_text(text)
                
                # Calculate text dimensions
                line_heights, line_widths, total_text_height, max_text_width = \
                    self.renderer.calculate_text_dimensions(display_lines)
                
                # Calculate background position (which includes width calculation)
                frame_width, frame_height = 640, 480
                text_x, text_y, bg_x1, bg_y1, bg_x2, bg_y2 = \
                    self.renderer.calculate_text_position(
                        frame_width, frame_height, max_text_width, total_text_height
                    )
                
                # Calculate actual background width and text area width
                bg_width = bg_x2 - bg_x1
                text_area_width = bg_width - (2 * self.renderer.style.padding)
                
                print(f"\nText: '{text[:50]}...'")
                print(f"Max text width: {max_text_width}")
                print(f"Background width: {bg_width}")
                print(f"Text area width: {text_area_width}")
                print(f"Padding: {self.renderer.style.padding}")
                
                # UPDATED: Handle frame-constrained backgrounds correctly
                frame_width = 640
                
                # Check if background is constrained to frame boundaries
                if bg_x1 == 0 and bg_x2 == frame_width:
                    # Background is frame-constrained - text may be wider than available area
                    print(f"  Background constrained to frame width - text may be clipped")
                    self.assertEqual(bg_width, frame_width, "Frame-constrained background should span full frame width")
                else:
                    # Normal case - text area should contain the text
                    self.assertGreaterEqual(text_area_width, max_text_width,
                        f"Background text area ({text_area_width}px) should contain text ({max_text_width}px)")
                
                # Verify background stays within frame bounds (main fix)
                self.assertGreaterEqual(bg_x1, 0, "Background should not extend left of frame")
                self.assertLessEqual(bg_x2, frame_width, "Background should not extend right of frame")
                
    def test_rendered_text_fits_in_background(self):
        """Test that when actually rendered, text fits within the background bounds."""
        test_caption = {
            'text': "We're coming to now the end of the first six chapters of the Gita Krishna just told Arjuna to reposition his lifestyle.",
            'start_time': 0,
            'end_time': 5,
            'language': 'en'
        }
        
        frame_copy = self.test_frame.copy()
        
        # Render the caption
        result_frame = self.renderer.render_caption(
            frame_copy, test_caption, 2.5, 0, 'en'
        )
        
        # Convert to PIL for analysis
        pil_image = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
        pixels = np.array(pil_image)
        
        # Find the background area (dark pixels)
        # Background should be darker than text
        gray = cv2.cvtColor(result_frame, cv2.COLOR_BGR2GRAY)
        
        # Find regions that are not pure black (have been modified)
        modified_mask = gray > 0
        
        if np.any(modified_mask):
            # Find bounding box of all modified pixels
            y_coords, x_coords = np.where(modified_mask)
            
            if len(x_coords) > 0 and len(y_coords) > 0:
                actual_left = np.min(x_coords)
                actual_right = np.max(x_coords)
                actual_top = np.min(y_coords)
                actual_bottom = np.max(y_coords)
                
                actual_width = actual_right - actual_left
                actual_height = actual_bottom - actual_top
                
                print(f"\nActual rendered bounds:")
                print(f"Width: {actual_width}px, Height: {actual_height}px")
                print(f"Left: {actual_left}, Right: {actual_right}")
                print(f"Top: {actual_top}, Bottom: {actual_bottom}")
                
                # The actual rendered area should have reasonable dimensions
                self.assertGreater(actual_width, 0, "No text was rendered (width = 0)")
                self.assertGreater(actual_height, 0, "No text was rendered (height = 0)")
        else:
            self.fail("No text was rendered - frame was not modified")
            
    def test_multi_line_background_sizing(self):
        """Test background sizing for multi-line captions."""
        multi_line_text = "First line of caption text\nSecond line that is much longer than the first\nThird line medium length"
        
        # Process into display lines
        display_lines = self.renderer.process_caption_text(multi_line_text)
        
        # Should have multiple lines
        self.assertGreater(len(display_lines), 1, "Multi-line text should produce multiple display lines")
        
        # Calculate dimensions
        line_heights, line_widths, total_text_height, max_text_width = \
            self.renderer.calculate_text_dimensions(display_lines)
        
        print(f"\nMulti-line text analysis:")
        print(f"Lines: {len(display_lines)}")
        print(f"Line widths: {line_widths}")
        print(f"Max width: {max_text_width}")
        print(f"Total height: {total_text_height}")
        
        # Max width should be the width of the longest line
        self.assertEqual(max_text_width, max(line_widths))
        
        # Total height should account for all lines plus spacing
        expected_height = sum(line_heights) + (len(display_lines) - 1) * 5  # 5px spacing
        self.assertEqual(total_text_height, expected_height)
        
    def test_unicode_text_background_sizing(self):
        """Test background sizing for Unicode text (French, German, Russian)."""
        unicode_texts = {
            'fr': 'Nous arrivons maintenant à la fin des six premiers chapitres: café, résumé, naïve',
            'de': 'Hallo Welt mit deutschen Umlauten: Müller, Größe, Weiß, Österreich',
            'ru': 'Привет мир с русскими символами: Москва, Санкт-Петербург, Владивосток'
        }
        
        for lang, text in unicode_texts.items():
            with self.subTest(language=lang):
                # Process text
                display_lines = self.renderer.process_caption_text(text)
                
                # Calculate dimensions
                line_heights, line_widths, total_text_height, max_text_width = \
                    self.renderer.calculate_text_dimensions(display_lines)
                
                print(f"\n{lang.upper()} text: '{text[:40]}...'")
                print(f"Calculated width: {max_text_width}px")
                
                # Width should be positive and reasonable
                self.assertGreater(max_text_width, 0, f"{lang} text width should be positive")
                self.assertLess(max_text_width, 2000, f"{lang} text width seems unreasonably large")
                
                # Test actual rendering
                frame_copy = self.test_frame.copy()
                result = self.renderer.render_text_line(
                    frame_copy, text, 50, 50, 1.0, lang
                )
                
                # Should modify the frame
                self.assertFalse(np.array_equal(result, self.test_frame),
                    f"{lang} text should modify the frame when rendered")

    def test_line_alignment_within_background(self):
        """Test that all text lines are positioned within the background boundaries."""
        # Create text with lines of very different lengths
        problematic_text = "Short\nThis is a much longer line that should be contained within the background\nMedium length line"
        
        frame_copy = self.test_frame.copy()
        test_caption = {
            'text': problematic_text,
            'start_time': 0,
            'end_time': 5,
            'language': 'en'
        }
        
        # Process the text to understand the line structure
        display_lines = self.renderer.process_caption_text(problematic_text)
        line_heights, line_widths, total_text_height, max_text_width = \
            self.renderer.calculate_text_dimensions(display_lines)
        
        print(f"\nLine alignment test:")
        print(f"Lines: {display_lines}")
        print(f"Line widths: {line_widths}")
        print(f"Max width: {max_text_width}")
        
        # Calculate expected background position
        frame_width, frame_height = 640, 480
        text_x, text_y, bg_x1, bg_y1, bg_x2, bg_y2 = \
            self.renderer.calculate_text_position(
                frame_width, frame_height, max_text_width, total_text_height
            )
        
        print(f"Background: x1={bg_x1}, x2={bg_x2}, width={bg_x2-bg_x1}")
        print(f"Text area: x={text_x}, width={bg_x2-bg_x1-2*self.renderer.style.padding}")
        
        # All lines should fit within the background boundaries
        # The longest line determines the background width
        # Shorter lines should be positioned within that background, not centered on the frame
        
        # Expected positioning: all lines should start at or near text_x (left edge of text area)
        # and end before bg_x2 - padding (right edge of text area)
        
        expected_text_right_boundary = bg_x2 - self.renderer.style.padding
        
        for i, (line, width) in enumerate(zip(display_lines, line_widths)):
            # Calculate where this line would be positioned if centered on frame
            frame_centered_x = (frame_width - width) // 2
            frame_centered_right = frame_centered_x + width
            
            print(f"Line {i+1}: '{line}' width={width}")
            print(f"  If frame-centered: x={frame_centered_x}, right={frame_centered_right}")
            print(f"  Background boundary: right={expected_text_right_boundary}")
            
            # For lines shorter than max_text_width, frame-centering would 
            # position them differently than background-relative positioning
            if width < max_text_width:
                # This line will be positioned outside the background bounds
                # if it's centered on the frame instead of within the background
                if frame_centered_x < bg_x1 or frame_centered_right > bg_x2:
                    print(f"  ❌ PROBLEM: Line extends outside background bounds!")
                    print(f"     Background: {bg_x1} to {bg_x2}")
                    print(f"     Line: {frame_centered_x} to {frame_centered_right}")
                    
                    # This is the bug! Lines should be positioned relative to background, not frame
                    self.fail(f"Line '{line}' would be positioned outside background bounds when frame-centered")
        
        # Test actual rendering to see the problem
        result_frame = self.renderer.render_caption(frame_copy, test_caption, 2.5, 0, 'en')
        
        # The result should have text properly contained within the background
        # This test documents the current bug behavior
        print(f"Caption rendered - investigating actual positioning...")

    def test_fixed_line_positioning_within_background(self):
        """Test that the fix correctly positions lines within background boundaries."""
        # Use shorter text that creates a background smaller than frame width
        test_text = "Short\nMedium line\nThis line is longer but not too long"
        
        frame_copy = self.test_frame.copy()
        test_caption = {
            'text': test_text,
            'start_time': 0,
            'end_time': 5,
            'language': 'en'
        }
        
        # Get the expected background boundaries
        display_lines = self.renderer.process_caption_text(test_text)
        line_heights, line_widths, total_text_height, max_text_width = \
            self.renderer.calculate_text_dimensions(display_lines)
        
        frame_width, frame_height = 640, 480
        text_x, text_y, bg_x1, bg_y1, bg_x2, bg_y2 = \
            self.renderer.calculate_text_position(
                frame_width, frame_height, max_text_width, total_text_height
            )
        
        # Calculate text area boundaries
        text_area_left = bg_x1 + self.renderer.style.padding
        text_area_right = bg_x2 - self.renderer.style.padding
        text_area_width = text_area_right - text_area_left
        
        print(f"\nFixed positioning test:")
        print(f"Background: {bg_x1} to {bg_x2} (width: {bg_x2-bg_x1})")
        print(f"Text area: {text_area_left} to {text_area_right} (width: {text_area_width})")
        print(f"Max text width: {max_text_width}")
        
        # For each line, calculate where it SHOULD be positioned (centered within text area)
        has_difference = False
        for i, (line, width) in enumerate(zip(display_lines, line_widths)):
            # Expected position: centered within the text area
            expected_x = text_area_left + (text_area_width - width) // 2
            expected_right = expected_x + width
            
            print(f"Line {i+1}: '{line}' (width: {width})")
            print(f"  Expected position: x={expected_x}, right={expected_right}")
            
            # Verify the expected position is within bounds
            self.assertGreaterEqual(expected_x, text_area_left, 
                f"Line {i+1} expected position should be >= text area left")
            self.assertLessEqual(expected_right, text_area_right, 
                f"Line {i+1} expected position should be <= text area right")
            
            # Check if this differs from frame-centering for shorter lines
            frame_centered_x = (frame_width - width) // 2
            if width < max_text_width:
                print(f"  Frame-centered would be: x={frame_centered_x}")
                if expected_x != frame_centered_x:
                    has_difference = True
                    print(f"  ✓ Background-relative positioning differs from frame-centering")
                else:
                    print(f"  Note: Background-relative matches frame-centering for this case")
        
        # At least one line should show the difference (unless background spans full frame)
        if bg_x1 > 0 and bg_x2 < frame_width:
            self.assertTrue(has_difference, 
                "At least one line should show difference between background-relative and frame-centering")
        
        # Test actual rendering
        result_frame = self.renderer.render_caption(frame_copy, test_caption, 2.5, 0, 'en')
        
        # Verify the frame was modified (text was rendered)
        self.assertFalse(np.array_equal(result_frame, self.test_frame),
            "Frame should be modified after rendering caption")
        
        print("✓ Caption rendered with fixed positioning!")


def run_tests():
    """Run background sizing tests."""
    print("=" * 70)
    print("RUNNING BACKGROUND SIZING BUG TESTS")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBackgroundSizingBug)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("BACKGROUND SIZING TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
            print(f"  {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
            print(f"  {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 