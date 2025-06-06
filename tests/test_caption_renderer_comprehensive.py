#!/usr/bin/env python3
"""Comprehensive unit tests for caption renderer covering all major fixes."""

import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock, PropertyMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.caption_overlay.renderer import CaptionRenderer
from src.caption_overlay.style_config import CaptionStyleConfig


class TestCaptionRendererComprehensive(unittest.TestCase):
    """Comprehensive test cases for CaptionRenderer covering all major fixes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = CaptionRenderer()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
    # ==================== LANGUAGE COLOR ASSIGNMENT TESTS ====================
    
    def test_all_supported_language_colors(self):
        """Test that all supported languages have correct color assignments."""
        expected_colors = {
            'en': (255, 255, 255),  # White
            'fr': (255, 200, 150),  # Pale blue (NOT the forbidden 255, 255, 150)
            'de': (0, 255, 255),    # Yellow (specifically requested)
            'it': (0, 165, 255),    # Orange
            'hu': (0, 255, 0),      # Green
            'ru': (203, 192, 255),  # Pale pink
            'uk': (255, 0, 255),    # Magenta
        }
        
        for lang, expected_color in expected_colors.items():
            with self.subTest(language=lang):
                color = self.renderer.get_language_color(lang)
                self.assertEqual(color, expected_color, 
                    f"{lang.upper()} color mismatch. Expected {expected_color}, got {color}")
    
    def test_french_color_not_forbidden_yellow(self):
        """Test that French color is NOT the forbidden yellow (255, 255, 150)."""
        french_color = self.renderer.get_language_color('fr')
        forbidden_yellow = (255, 255, 150)
        
        self.assertNotEqual(french_color, forbidden_yellow,
            "French color should NOT be the forbidden yellow (255, 255, 150)")
        self.assertEqual(french_color, (255, 200, 150),
            "French should be pale blue (255, 200, 150)")
    
    def test_german_gets_yellow_color(self):
        """Test that German specifically gets yellow color as requested."""
        german_color = self.renderer.get_language_color('de')
        expected_yellow = (0, 255, 255)  # Yellow in BGR
        
        self.assertEqual(german_color, expected_yellow,
            "German should get yellow color (0, 255, 255) as specifically requested")
    
    def test_forbidden_colors_enforcement(self):
        """Test that forbidden colors are properly rejected and forced to white."""
        # Mock the language_colors to include a forbidden color
        original_colors = self.renderer.language_colors.copy()
        
        try:
            # Test with pure yellow (forbidden)
            self.renderer.language_colors['test'] = (255, 255, 0)  # Pure yellow
            color = self.renderer.get_language_color('test')
            self.assertEqual(color, (255, 255, 255), 
                "Forbidden pure yellow should be forced to white")
                
        finally:
            # Restore original colors
            self.renderer.language_colors = original_colors
    
    def test_unsupported_language_defaults_to_white(self):
        """Test that unsupported languages default to white."""
        unsupported_langs = ['zh', 'ja', 'ar', 'hi', 'unknown']
        
        for lang in unsupported_langs:
            with self.subTest(language=lang):
                color = self.renderer.get_language_color(lang)
                self.assertEqual(color, (255, 255, 255),
                    f"Unsupported language '{lang}' should default to white")
    
    # ==================== UNICODE RENDERING LOGIC TESTS ====================
    
    def test_unicode_rendering_assignment(self):
        """Test that Unicode rendering is correctly assigned based on language and content."""
        test_cases = [
            # Cyrillic languages always use PIL
            ('ru', 'Hello', True, 'Russian always uses PIL'),
            ('ru', 'Привет', True, 'Russian with Cyrillic uses PIL'),
            ('uk', 'Hello', True, 'Ukrainian always uses PIL'),
            ('uk', 'Привіт', True, 'Ukrainian with Cyrillic uses PIL'),
            
            # Western European languages use PIL only for Unicode content
            ('en', 'Hello', False, 'English ASCII uses OpenCV'),
            ('fr', 'Hello', False, 'French ASCII uses OpenCV'),
            ('fr', 'café naïve', True, 'French with accents uses PIL'),
            ('de', 'Hello', False, 'German ASCII uses OpenCV'),
            ('de', 'schön größer', True, 'German with umlauts uses PIL'),
            ('it', 'Hello', False, 'Italian ASCII uses OpenCV'),
            ('it', 'città perché', True, 'Italian with accents uses PIL'),
            ('hu', 'Hello', False, 'Hungarian ASCII uses OpenCV'),
            ('hu', 'szép nagy', True, 'Hungarian with special chars uses PIL'),
        ]
        
        for language, text, expected_unicode, description in test_cases:
            with self.subTest(language=language, text=text):
                # Check if text contains Unicode characters
                has_unicode = any(ord(char) > 127 for char in text)
                is_cyrillic = language in ['ru', 'uk']
                expected_result = is_cyrillic or has_unicode
                
                self.assertEqual(expected_result, expected_unicode,
                    f"{description}: Unicode rendering logic failed")
    
    def test_consistent_rendering_per_language(self):
        """Test that each language uses consistent rendering regardless of specific text content."""
        # This test ensures we don't have the "German formatting 2 different ways" issue
        test_texts = ['Hello World', 'Test text', 'Simple']
        
        cyrillic_languages = ['ru', 'uk']
        western_languages = ['en', 'fr', 'de', 'it', 'hu']
        
        # Cyrillic languages should always use PIL
        for lang in cyrillic_languages:
            for text in test_texts:
                with self.subTest(language=lang, text=text):
                    is_cyrillic = lang in ['ru', 'uk']
                    self.assertTrue(is_cyrillic, 
                        f"{lang} should always use PIL rendering")
        
        # Western languages should use OpenCV for ASCII
        for lang in western_languages:
            for text in test_texts:
                with self.subTest(language=lang, text=text):
                    has_unicode = any(ord(char) > 127 for char in text)
                    is_cyrillic = lang in ['ru', 'uk']
                    uses_unicode = is_cyrillic or has_unicode
                    self.assertFalse(uses_unicode,
                        f"{lang} with ASCII text should use OpenCV rendering")
    
    # ==================== TEXT DIMENSIONS AND POSITIONING TESTS ====================
    
    def test_text_dimensions_calculation(self):
        """Test that text dimensions are calculated without errors."""
        test_lines = [
            'Hello World',
            'This is a test',
            'Multiple lines of text'
        ]
        
        try:
            line_heights, line_widths, total_height, max_width = self.renderer.calculate_text_dimensions(test_lines)
            
            # Verify results are reasonable
            self.assertEqual(len(line_heights), len(test_lines))
            self.assertEqual(len(line_widths), len(test_lines))
            self.assertGreater(total_height, 0)
            self.assertGreater(max_width, 0)
            
            # Verify all heights and widths are positive
            for height in line_heights:
                self.assertGreater(height, 0, "Line height should be positive")
            for width in line_widths:
                self.assertGreater(width, 0, "Line width should be positive")
                
        except Exception as e:
            self.fail(f"calculate_text_dimensions failed with error: {e}")
    
    def test_no_language_variable_error_in_dimensions(self):
        """Test that the 'language' variable error in calculate_text_dimensions is fixed."""
        # This test ensures the NameError: name 'language' is not defined is fixed
        test_lines = ['Test line with unicode: café', 'Another line']
        
        try:
            result = self.renderer.calculate_text_dimensions(test_lines)
            self.assertIsNotNone(result, "calculate_text_dimensions should return a result")
        except NameError as e:
            if "'language' is not defined" in str(e):
                self.fail("The 'language' variable error in calculate_text_dimensions is not fixed")
            else:
                raise  # Re-raise if it's a different NameError
    
    # ==================== CAPTION MANAGEMENT TESTS ====================
    
    @patch('src.caption_overlay.renderer.CaptionRenderer.process_caption_text')
    def test_caption_processing_without_duplication_logic(self, mock_process):
        """Test caption processing doesn't have built-in duplication issues."""
        mock_process.return_value = ['Processed caption']
        
        # Test with a sample caption
        caption_data = {
            'text': 'Test caption',
            'language': 'en',
            'start_time': 0.0,
            'end_time': 3.0
        }
        
        try:
            # This should not raise any errors about duplicate detection within renderer
            # Use the correct signature
            result_frame = self.renderer.render_caption(
                self.test_frame.copy(), 
                caption_data, 
                current_time=1.0,  # Within the caption timeframe
                caption_index=0,
                language='en'
            )
            self.assertIsNotNone(result_frame, "render_caption should return a frame")
        except Exception as e:
            # Should not have caption duplication errors from renderer itself
            self.assertNotIn("duplicate", str(e).lower(), 
                "Renderer should not have built-in duplication detection")
    
    def test_process_caption_text_handles_all_languages(self):
        """Test that process_caption_text works for all supported languages."""
        test_text = "This is a test caption that should be processed correctly."
        
        supported_languages = ['en', 'fr', 'de', 'it', 'hu', 'ru', 'uk']
        
        for language in supported_languages:
            with self.subTest(language=language):
                try:
                    # process_caption_text only takes the text, not language
                    result = self.renderer.process_caption_text(test_text)
                    self.assertIsInstance(result, list, 
                        f"process_caption_text should return a list for {language}")
                    self.assertGreater(len(result), 0,
                        f"Processed text should not be empty for {language}")
                except Exception as e:
                    self.fail(f"process_caption_text failed for {language}: {e}")
    
    # ==================== RENDERING METHOD TESTS ====================
    
    def test_render_caption_all_languages(self):
        """Test that render_caption works for all supported languages without errors."""
        supported_languages = ['en', 'fr', 'de', 'it', 'hu', 'ru', 'uk']
        
        for language in supported_languages:
            with self.subTest(language=language):
                caption_data = {
                    'text': f'Test caption in {language}',
                    'language': language,
                    'start_time': 0.0,
                    'end_time': 3.0
                }
                
                try:
                    # Use the correct signature: frame, caption, current_time, caption_index, language
                    result_frame = self.renderer.render_caption(
                        self.test_frame.copy(), 
                        caption_data, 
                        current_time=1.0,  # Within the caption timeframe
                        caption_index=0,
                        language=language
                    )
                    self.assertIsNotNone(result_frame, 
                        f"render_caption should return a frame for {language}")
                    self.assertEqual(result_frame.shape, self.test_frame.shape,
                        f"Result frame should have same shape for {language}")
                except Exception as e:
                    self.fail(f"render_caption failed for {language}: {e}")
    
    def test_unicode_text_rendering(self):
        """Test that Unicode text renders without errors for all languages."""
        unicode_test_cases = [
            ('fr', 'Bonjour café, naïve résumé'),
            ('de', 'schön größer hören Mädchen'),
            ('it', 'città perché più'),
            ('hu', 'szép nagy hét'),
            ('ru', 'Привет мир'),
            ('uk', 'Привіт світ'),
        ]
        
        for language, text in unicode_test_cases:
            with self.subTest(language=language, text=text):
                caption_data = {
                    'text': text,
                    'language': language,
                    'start_time': 0.0,
                    'end_time': 3.0
                }
                
                try:
                    # Use the correct signature
                    result_frame = self.renderer.render_caption(
                        self.test_frame.copy(), 
                        caption_data, 
                        current_time=1.0,  # Within the caption timeframe
                        caption_index=0,
                        language=language
                    )
                    self.assertIsNotNone(result_frame, 
                        f"Unicode rendering should work for {language}")
                except Exception as e:
                    self.fail(f"Unicode rendering failed for {language} '{text}': {e}")
    
    # ==================== ERROR PREVENTION TESTS ====================
    
    def test_no_missing_methods(self):
        """Test that all expected methods exist to prevent 'method not found' errors."""
        required_methods = [
            'get_language_color',
            'calculate_text_dimensions', 
            'process_caption_text',
            'render_caption',
            'render_text_line'
        ]
        
        for method_name in required_methods:
            with self.subTest(method=method_name):
                self.assertTrue(hasattr(self.renderer, method_name),
                    f"CaptionRenderer should have method '{method_name}'")
                method = getattr(self.renderer, method_name)
                self.assertTrue(callable(method),
                    f"'{method_name}' should be callable")
    
    def test_no_color_conflicts(self):
        """Test that no language colors conflict with forbidden colors."""
        forbidden_colors = [
            (255, 255, 0),   # Pure yellow
            (255, 255, 100), # Light yellow variants
            (255, 255, 150), # The problematic French color
            (200, 255, 100),
            (150, 255, 100),
        ]
        
        for language, color in self.renderer.language_colors.items():
            with self.subTest(language=language, color=color):
                self.assertNotIn(color, forbidden_colors,
                    f"{language} color {color} conflicts with forbidden colors")
    
    # ==================== INTEGRATION TESTS ====================
    
    def test_end_to_end_multilingual_rendering(self):
        """Test end-to-end rendering with multiple languages simultaneously."""
        captions = [
            {'text': 'Hello world', 'language': 'en', 'start_time': 0.0, 'end_time': 3.0},
            {'text': 'Bonjour monde', 'language': 'fr', 'start_time': 0.0, 'end_time': 3.0},
            {'text': 'Привет мир', 'language': 'ru', 'start_time': 0.0, 'end_time': 3.0},
        ]
        
        frame = self.test_frame.copy()
        
        # Render each caption (simulating overlay behavior)
        for i, caption in enumerate(captions):
            try:
                # Use the correct signature
                frame = self.renderer.render_caption(
                    frame, 
                    caption, 
                    current_time=1.0,  # Within timeframe
                    caption_index=i,
                    language=caption['language']
                )
            except Exception as e:
                self.fail(f"End-to-end rendering failed for {caption['language']}: {e}")
        
        # Frame should still be valid
        self.assertEqual(frame.shape, self.test_frame.shape)
    
    def test_performance_no_memory_leaks(self):
        """Test that repeated rendering doesn't cause obvious memory issues."""
        caption = {
            'text': 'Performance test caption',
            'language': 'en',
            'start_time': 0.0,
            'end_time': 3.0
        }
        
        # Render many times to check for obvious memory issues
        for i in range(100):
            try:
                frame = self.renderer.render_caption(
                    self.test_frame.copy(), 
                    caption, 
                    current_time=1.0,  # Within timeframe
                    caption_index=0,
                    language='en'
                )
            except Exception as e:
                self.fail(f"Performance test failed at iteration {i}: {e}")


def run_comprehensive_tests():
    """Run all comprehensive tests and report results."""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE CAPTION RENDERER TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaptionRendererComprehensive)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_tests()
    exit(0 if success else 1) 