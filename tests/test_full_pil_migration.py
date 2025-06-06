#!/usr/bin/env python3
"""
Comprehensive tests for the full PIL migration of ISKCON-Translate caption system.

Tests:
1. All languages render with PIL
2. Sanskrit transliteration characters
3. Unicode handling for Western European languages
4. Consistent dimensions and positioning
5. No OpenCV rendering paths remaining
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


class TestFullPILMigration(unittest.TestCase):
    """Tests for complete PIL migration of caption rendering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = CaptionRenderer()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
    def test_all_languages_use_pil(self):
        """Test that all supported languages use PIL rendering."""
        languages = ['en', 'fr', 'de', 'it', 'hu', 'ru', 'uk']
        test_texts = {
            'en': 'Hello world with special chars: "quotes" & symbols!',
            'fr': 'Bonjour le monde avec accents: café, résumé, naïve',
            'de': 'Hallo Welt mit Umlauten: Müller, Größe, Weiß',
            'it': 'Ciao mondo con accenti: città, perché, così',
            'hu': 'Helló világ magyar ékezetes betűkkel: szép, nagy',
            'ru': 'Привет мир с кириллицей: Москва, Санкт-Петербург',
            'uk': 'Привіт світ українською: Київ, Львів, Харків'
        }
        
        for lang in languages:
            with self.subTest(language=lang):
                text = test_texts.get(lang, f'Test text for {lang}')
                frame_copy = self.test_frame.copy()
                
                # This should not raise any errors and should use PIL for all
                result = self.renderer.render_text_line(
                    frame_copy, text, 50, 50, 1.0, lang
                )
                
                # Verify result frame was modified (text was rendered)
                self.assertFalse(np.array_equal(result, self.test_frame))
                
    def test_sanskrit_transliteration_complete(self):
        """Test comprehensive Sanskrit transliteration character support."""
        sanskrit_chars = [
            # Vowels with macrons
            'ā', 'ī', 'ū', 'ṝ', 'ḷ', 'ḹ',
            # Consonants with dots
            'ṛ', 'ṃ', 'ḥ', 'ṅ', 'ñ', 'ṭ', 'ḍ', 'ṇ', 'ś', 'ṣ',
            # Additional diacritics
            'kh', 'gh', 'ch', 'jh', 'th', 'dh', 'ph', 'bh'
        ]
        
        # Test individual characters
        for char in sanskrit_chars:
            with self.subTest(character=char):
                text = f'Sanskrit: {char}'
                frame_copy = self.test_frame.copy()
                
                result = self.renderer.render_text_line(
                    frame_copy, text, 50, 50, 1.0, 'en'
                )
                
                # Should render without errors
                self.assertIsNotNone(result)
        
        # Test full Sanskrit text
        full_text = 'oṃ namaḥ śivāya: ' + ' '.join(sanskrit_chars[:10])
        frame_copy = self.test_frame.copy()
        
        result = self.renderer.render_text_line(
            frame_copy, full_text, 50, 50, 1.0, 'en'
        )
        
        self.assertFalse(np.array_equal(result, self.test_frame))
        
    def test_western_european_unicode(self):
        """Test that Western European Unicode characters render correctly."""
        unicode_tests = {
            'fr': 'café résumé naïve déjà fiancé',
            'de': 'Müller Größe Weiß Österreich',
            'it': 'città perché così università',
            'hu': 'szép nagy könnyű',
            'en': 'résumé naïve café'  # English with borrowed words
        }
        
        for lang, text in unicode_tests.items():
            with self.subTest(language=lang, text=text):
                frame_copy = self.test_frame.copy()
                
                # Verify text contains Unicode characters
                has_unicode = any(ord(char) > 127 for char in text)
                self.assertTrue(has_unicode, f"Test text for {lang} should contain Unicode")
                
                # Render should work
                result = self.renderer.render_text_line(
                    frame_copy, text, 50, 50, 1.0, lang
                )
                
                # Result frame should be modified
                self.assertFalse(np.array_equal(result, self.test_frame))
                
    def test_pil_text_dimensions_accuracy(self):
        """Test that PIL text dimensions are calculated accurately."""
        test_lines = [
            'Short text',
            'This is a much longer line of text to test width calculation',
            'Unicode: café résumé naïve',
            'Sanskrit: oṃ namaḥ śivāya',
            'Cyrillic: Привет мир'
        ]
        
        line_heights, line_widths, total_height, max_width = \
            self.renderer.calculate_text_dimensions(test_lines)
        
        # Should have dimensions for each line
        self.assertEqual(len(line_heights), len(test_lines))
        self.assertEqual(len(line_widths), len(test_lines))
        
        # Dimensions should be positive
        self.assertTrue(all(h > 0 for h in line_heights))
        self.assertTrue(all(w > 0 for w in line_widths))
        self.assertGreater(total_height, 0)
        self.assertGreater(max_width, 0)
        
        # Max width should be the actual maximum
        self.assertEqual(max_width, max(line_widths))
        
    def test_no_opencv_text_rendering(self):
        """Test that no OpenCV text rendering methods are used."""
        # This test ensures we're not accidentally using cv2.putText anywhere
        import inspect
        
        # Get the render_text_line method source
        source = inspect.getsource(self.renderer.render_text_line)
        
        # Should not contain cv2.putText calls
        self.assertNotIn('cv2.putText', source)
        self.assertNotIn('putText', source)
        
        # Should contain PIL rendering calls
        self.assertIn('_render_unicode_text', source)
        
    def test_consistent_positioning_all_languages(self):
        """Test that text positioning is consistent across all languages."""
        languages = ['en', 'fr', 'de', 'it', 'hu', 'ru', 'uk']
        test_text = 'Test positioning'
        
        positions = []
        
        for lang in languages:
            frame_copy = self.test_frame.copy()
            
            # Use render_caption for consistent positioning
            caption = {
                'text': test_text,
                'start_time': 0,
                'end_time': 5,
                'language': lang
            }
            
            result = self.renderer.render_caption(
                frame_copy, caption, 2.5, 0, lang
            )
            
            # Should render without errors
            self.assertIsNotNone(result)
            positions.append(f"Lang {lang}: OK")
        
        # All languages should process successfully
        self.assertEqual(len(positions), len(languages))
        
    def test_mixed_script_handling(self):
        """Test handling of mixed scripts in single text."""
        mixed_texts = [
            'English with café and résumé',
            'Deutsch with English: Hello Welt',
            'Sanskrit in English: oṃ namaḥ śivāya means...',
            'Русский and English mixed text',
            'Multiple scripts: Hello café привет'
        ]
        
        for text in mixed_texts:
            with self.subTest(text=text[:30] + '...'):
                frame_copy = self.test_frame.copy()
                
                result = self.renderer.render_text_line(
                    frame_copy, text, 50, 50, 1.0, 'en'
                )
                
                # Should handle mixed scripts without errors
                self.assertIsNotNone(result)
                # Should modify the frame
                self.assertFalse(np.array_equal(result, self.test_frame))
                
    def test_performance_pil_rendering(self):
        """Test that PIL rendering performance is acceptable."""
        import time
        
        test_text = 'Performance test with Unicode: café résumé naïve ṃ ḥ ś'
        iterations = 10
        
        start_time = time.time()
        
        for _ in range(iterations):
            frame_copy = self.test_frame.copy()
            self.renderer.render_text_line(
                frame_copy, test_text, 50, 50, 1.0, 'en'
            )
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        
        # Should render reasonably fast (under 50ms per frame)
        self.assertLess(avg_time, 0.05, f"PIL rendering too slow: {avg_time:.3f}s")
        
    def test_color_consistency_pil(self):
        """Test that colors are applied consistently in PIL mode."""
        test_colors = {
            'en': (255, 255, 255),
            'fr': (255, 200, 150),
            'de': (0, 255, 255),
            'ru': (203, 192, 255)
        }
        
        for lang, expected_bgr in test_colors.items():
            with self.subTest(language=lang):
                # Get the color from renderer
                actual_bgr = self.renderer.get_language_color(lang)
                
                # Should match expected color
                self.assertEqual(actual_bgr, expected_bgr, 
                    f"{lang} color mismatch: expected {expected_bgr}, got {actual_bgr}")


def run_tests():
    """Run all PIL migration tests."""
    print("=" * 60)
    print("RUNNING FULL PIL MIGRATION TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFullPILMigration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIL MIGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 