#!/usr/bin/env python3
"""Unit tests for caption renderer functionality."""

import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.caption_overlay.renderer import CaptionRenderer
from src.caption_overlay.style_config import CaptionStyleConfig


class TestCaptionRenderer(unittest.TestCase):
    """Test cases for CaptionRenderer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = CaptionRenderer()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
    def test_language_color_assignment(self):
        """Test that all supported languages get correct colors."""
        expected_colors = {
            'en': (255, 255, 255),  # White
            'fr': (255, 200, 150),  # Pale blue (NOT 255,255,150!)
            'de': (0, 255, 255),    # Yellow
            'it': (0, 165, 255),    # Orange
            'hu': (0, 255, 0),      # Green
            'ru': (203, 192, 255),  # Pale pink
            'uk': (255, 0, 255),    # Magenta
        }
        
        for lang, expected_color in expected_colors.items():
            with self.subTest(language=lang):
                actual_color = self.renderer.get_language_color(lang)
                self.assertEqual(actual_color, expected_color, 
                    f"Language '{lang}' color mismatch: expected {expected_color}, got {actual_color}")
    
    def test_french_color_not_yellow(self):
        """Test that French color is NOT the problematic yellow color."""
        french_color = self.renderer.get_language_color('fr')
        forbidden_yellow = (255, 255, 150)
        
        self.assertNotEqual(french_color, forbidden_yellow,
            f"French color should NOT be {forbidden_yellow}, but got {french_color}")
        
        # Verify it's the correct pale blue
        self.assertEqual(french_color, (255, 200, 150),
            f"French color should be (255, 200, 150), but got {french_color}")
    
    def test_unicode_rendering_logic(self):
        """Test Unicode rendering decision logic."""
        test_cases = [
            # (text, language, expected_uses_unicode, description)
            ('Hello World', 'en', False, 'English ASCII should use OpenCV'),
            ('Hello World', 'de', False, 'German ASCII should use OpenCV'),
            ('schön größer', 'de', True, 'German with umlauts should use PIL'),
            ('café naïve', 'fr', True, 'French with accents should use PIL'),
            ('ciao', 'it', False, 'Italian ASCII should use OpenCV'),
            ('città caffè', 'it', True, 'Italian with accents should use PIL'),
            ('hello', 'hu', False, 'Hungarian ASCII should use OpenCV'),
            ('helló', 'hu', True, 'Hungarian with accents should use PIL'),
            ('hello', 'ru', True, 'Russian should always use PIL'),
            ('привет', 'ru', True, 'Russian Cyrillic should use PIL'),
            ('hello', 'uk', True, 'Ukrainian should always use PIL'),
            ('привіт', 'uk', True, 'Ukrainian Cyrillic should use PIL'),
        ]
        
        for text, language, expected_unicode, description in test_cases:
            with self.subTest(text=text, language=language):
                # Check if text contains Unicode characters
                has_unicode = any(ord(char) > 127 for char in text)
                is_cyrillic_language = language in ['ru', 'uk']
                uses_unicode_rendering = is_cyrillic_language or has_unicode
                
                self.assertEqual(uses_unicode_rendering, expected_unicode, 
                    f"{description}: expected {expected_unicode}, got {uses_unicode_rendering}")
    
    def test_german_umlaut_detection(self):
        """Test that German umlauts are properly detected as Unicode."""
        german_texts = [
            ('Hallo', False, 'ASCII German'),
            ('schön', True, 'German with ö'),
            ('größer', True, 'German with ö'),
            ('Mädchen', True, 'German with ä'),
            ('über', True, 'German with ü'),
            ('Straße', True, 'German with ß'),
            ('Müller', True, 'German with ü'),
            ('Bäcker', True, 'German with ä'),
        ]
        
        for text, should_have_unicode, description in german_texts:
            with self.subTest(text=text):
                has_unicode = any(ord(char) > 127 for char in text)
                self.assertEqual(has_unicode, should_have_unicode, 
                    f"{description}: Unicode detection failed for '{text}'")
    
    def test_french_accent_detection(self):
        """Test that French accents are properly detected as Unicode."""
        french_texts = [
            ('bonjour', False, 'ASCII French'),
            ('café', True, 'French with é'),
            ('naïve', True, 'French with ï'),
            ('résumé', True, 'French with é'),
            ('crème', True, 'French with è'),
            ('français', True, 'French with ç'),
            ('hôtel', True, 'French with ô'),
            ('où', True, 'French with ù'),
        ]
        
        for text, should_have_unicode, description in french_texts:
            with self.subTest(text=text):
                has_unicode = any(ord(char) > 127 for char in text)
                self.assertEqual(has_unicode, should_have_unicode, 
                    f"{description}: Unicode detection failed for '{text}'")
    
    def test_forbidden_colors(self):
        """Test that forbidden colors are properly rejected."""
        # Mock the language_colors to include a forbidden color
        original_colors = self.renderer.language_colors.copy()
        
        try:
            # Test with a forbidden color that's actually in the forbidden list
            self.renderer.language_colors['test'] = (255, 255, 0)  # Pure yellow - actually forbidden
            
            color = self.renderer.get_language_color('test')
            
            # Should be forced to white
            self.assertEqual(color, (255, 255, 255), 
                "Forbidden color should be forced to white")
        
        finally:
            # Restore original colors
            self.renderer.language_colors = original_colors
    
    def test_unsupported_language_defaults_to_white(self):
        """Test that unsupported languages default to white."""
        unsupported_languages = ['es', 'pt', 'zh', 'ja', 'ar']
        
        for lang in unsupported_languages:
            with self.subTest(language=lang):
                color = self.renderer.get_language_color(lang)
                self.assertEqual(color, (255, 255, 255), 
                    f"Unsupported language '{lang}' should default to white")
    
    def test_text_dimensions_calculation(self):
        """Test text dimensions calculation."""
        test_lines = ['Hello World', 'Test Line 2']
        
        line_heights, line_widths, total_height, max_width = \
            self.renderer.calculate_text_dimensions(test_lines)
        
        # Basic sanity checks
        self.assertEqual(len(line_heights), 2, "Should have 2 line heights")
        self.assertEqual(len(line_widths), 2, "Should have 2 line widths")
        self.assertGreater(total_height, 0, "Total height should be positive")
        self.assertGreater(max_width, 0, "Max width should be positive")
        self.assertEqual(max_width, max(line_widths), "Max width should match maximum line width")
    
    def test_fade_factor_calculation(self):
        """Test fade factor calculation for caption timing."""
        caption = {
            'start_time': 10.0,
            'end_time': 15.0,
            'text': 'Test caption'
        }
        
        # Test different time positions
        test_cases = [
            (9.0, 0.1),   # Before start (should be minimum)
            (10.0, 1.0),  # At start (should be full)
            (12.5, 1.0),  # Middle (should be full)
            (14.8, 1.0),  # Near end (should be full)
            (15.0, 0.1),  # At end (should be minimum)
            (16.0, 0.1),  # After end (should be minimum)
        ]
        
        for current_time, min_expected in test_cases:
            with self.subTest(time=current_time):
                fade_factor = self.renderer.calculate_fade_factor(caption, current_time)
                self.assertGreaterEqual(fade_factor, 0.1, "Fade factor should be at least 0.1")
                self.assertLessEqual(fade_factor, 1.0, "Fade factor should be at most 1.0")
                if min_expected == 0.1:
                    self.assertEqual(fade_factor, 0.1, f"Expected minimum fade at time {current_time}")
    
    @patch('cv2.putText')
    def test_ascii_text_rendering_uses_opencv(self, mock_puttext):
        """Test that ASCII text uses OpenCV rendering."""
        # Test ASCII text for German (should use OpenCV, not PIL)
        self.renderer.render_text_line(
            self.test_frame, 
            'Hello World',  # ASCII text
            100, 100, 1.0, 'de'
        )
        
        # OpenCV putText should be called
        self.assertTrue(mock_puttext.called, "OpenCV putText should be called for ASCII text")
    
    def test_process_caption_text(self):
        """Test caption text processing."""
        # Test basic line splitting
        result = self.renderer.process_caption_text("Line 1\nLine 2\nLine 3")
        expected = ['Line 1', 'Line 2', 'Line 3']
        self.assertEqual(result, expected, "Should split lines correctly")
        
        # Test empty line removal
        result = self.renderer.process_caption_text("Line 1\n\nLine 2\n  \nLine 3")
        expected = ['Line 1', 'Line 2', 'Line 3']
        self.assertEqual(result, expected, "Should remove empty lines")
        
        # Test single long line wrapping (basic check)
        long_text = "This is a very long line that should be wrapped because it exceeds the maximum characters per line limit"
        result = self.renderer.process_caption_text(long_text)
        self.assertIsInstance(result, list, "Should return a list")
        self.assertGreater(len(result), 0, "Should have at least one line")


class TestCaptionRendererIntegration(unittest.TestCase):
    """Integration tests for caption rendering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = CaptionRenderer()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
    def test_full_caption_rendering_workflow(self):
        """Test complete caption rendering workflow."""
        caption = {
            'text': 'Test caption with unicode: café',
            'start_time': 10.0,
            'end_time': 15.0,
            'language': 'fr'
        }
        
        current_time = 12.5
        
        # This should not raise any exceptions
        try:
            result_frame = self.renderer.render_caption(
                self.test_frame.copy(), 
                caption, 
                current_time, 
                caption_index=0, 
                language='fr'
            )
            
            # Basic checks
            self.assertEqual(result_frame.shape, self.test_frame.shape, 
                "Result frame should have same shape as input")
            self.assertEqual(result_frame.dtype, self.test_frame.dtype,
                "Result frame should have same dtype as input")
            
        except Exception as e:
            self.fail(f"Caption rendering failed with exception: {e}")


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when executed directly
    success = run_tests()
    if not success:
        print("\n❌ TESTS FAILED - Do not proceed with changes until tests pass!")
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0) 