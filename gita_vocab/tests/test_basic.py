"""
Basic tests for gita_vocab components.
"""

import pytest
from unittest.mock import Mock, patch

from gita_vocab.scraper import GitaScraper, ScrapedContent
from gita_vocab.normalizer import TextNormalizer, TokenInfo
from gita_vocab.generator import GlossaryGenerator


class TestGitaScraper:
    """Test the GitaScraper class."""
    
    def test_scraper_initialization(self):
        """Test scraper initializes correctly."""
        scraper = GitaScraper(base_delay=2.0, max_retries=3)
        
        assert scraper.base_delay == 2.0
        assert scraper.max_retries == 3
        assert len(scraper.sources) == 3
        assert 'vedabase' in scraper.sources
        assert 'asitis' in scraper.sources
        assert 'gitasupersite' in scraper.sources
    
    def test_scraped_content_model(self):
        """Test ScrapedContent model."""
        content = ScrapedContent(
            url="https://example.com",
            title="Test Chapter",
            text="Krishna teaches Arjuna about dharma and karma",
            chapter=2,
            verse="1",
            source="test",
            word_count=8
        )
        
        assert content.url == "https://example.com"
        assert content.chapter == 2
        assert content.word_count == 8
        assert content.source == "test"


class TestTextNormalizer:
    """Test the TextNormalizer class."""
    
    def test_normalizer_initialization(self):
        """Test normalizer initializes correctly."""
        normalizer = TextNormalizer(min_token_length=3, max_token_length=40)
        
        assert normalizer.min_token_length == 3
        assert normalizer.max_token_length == 40
        assert len(normalizer.stop_words) > 0
        assert 'the' in normalizer.stop_words
        assert 'Krishna' not in normalizer.stop_words
    
    def test_token_info(self):
        """Test TokenInfo class."""
        token_info = TokenInfo("Krishna", "krishna")
        
        assert token_info.token == "Krishna"
        assert token_info.ascii_form == "krishna"
        assert token_info.count == 0
        assert len(token_info.chapters) == 0
        
        # Add occurrence
        token_info.add_occurrence(2, "vedabase", "Krishna teaches Arjuna")
        
        assert token_info.count == 1
        assert 2 in token_info.chapters
        assert "vedabase" in token_info.sources
        assert len(token_info.contexts) == 1
    
    def test_should_keep_token(self):
        """Test token filtering logic."""
        normalizer = TextNormalizer(min_token_length=2, max_token_length=50)
        
        # Should keep
        assert normalizer._should_keep_token("Krishna") == True
        assert normalizer._should_keep_token("dharma") == True
        assert normalizer._should_keep_token("Bhagavad") == True
        
        # Should filter out
        assert normalizer._should_keep_token("a") == False  # Too short
        assert normalizer._should_keep_token("the") == False  # Stop word
        assert normalizer._should_keep_token("123") == False  # All digits
        assert normalizer._should_keep_token("...") == False  # All punctuation
    
    def test_ascii_fallback_generation(self):
        """Test ASCII fallback generation."""
        normalizer = TextNormalizer()
        
        # Test diacritic removal
        ascii_form = normalizer._generate_ascii_fallback("Kṛṣṇa")
        assert ascii_form == "krsna"
        
        ascii_form = normalizer._generate_ascii_fallback("Bhagavān")
        assert ascii_form == "bhagavan"
        
        # Test regular text
        ascii_form = normalizer._generate_ascii_fallback("Krishna")
        assert ascii_form == "krishna"


class TestGlossaryGenerator:
    """Test the GlossaryGenerator class."""
    
    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        normalizer = TextNormalizer()
        generator = GlossaryGenerator(normalizer)
        
        assert generator.normalizer == normalizer
        assert generator.output_dir.name == "glossaries"
    
    def test_validation_empty(self):
        """Test validation with no files."""
        normalizer = TextNormalizer()
        generator = GlossaryGenerator(normalizer)
        
        # Should fail validation when no files exist
        results = generator.validate_glossaries()
        assert results['common_glossary'] == False
        assert results['all_valid'] == False


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_sample_content_processing(self):
        """Test processing sample content through the pipeline."""
        # Create sample content
        sample_content = [
            ScrapedContent(
                url="https://example.com/bg/2/1",
                title="Bhagavad Gita 2.1",
                text="Krishna teaches Arjuna about dharma and karma in this verse",
                chapter=2,
                verse="1",
                source="test",
                word_count=10
            ),
            ScrapedContent(
                url="https://example.com/bg/2/2",
                title="Bhagavad Gita 2.2",
                text="Arjuna asks Krishna about the nature of dharma and duty",
                chapter=2,
                verse="2",
                source="test",
                word_count=9
            )
        ]
        
        # Normalize content
        normalizer = TextNormalizer()
        normalizer.normalize_content(sample_content)
        
        # Check results
        assert len(normalizer.tokens) > 0
        assert "Krishna" in normalizer.tokens
        assert "Arjuna" in normalizer.tokens
        assert "dharma" in normalizer.tokens
        
        # Check token info
        krishna_token = normalizer.tokens["Krishna"]
        assert krishna_token.count == 2  # Appears in both verses
        assert 2 in krishna_token.chapters
        assert "test" in krishna_token.sources
        
        # Test generator
        generator = GlossaryGenerator(normalizer)
        top_tokens = normalizer.get_top_tokens(5)
        
        assert len(top_tokens) > 0
        # Should have some tokens with counts
        assert all(count > 0 for _, count in top_tokens)


def test_cli_import():
    """Test that CLI module can be imported."""
    from gita_vocab.cli import cli
    assert cli is not None


def test_package_import():
    """Test that main package imports work."""
    from gita_vocab import GitaScraper, TextNormalizer, GlossaryGenerator
    
    assert GitaScraper is not None
    assert TextNormalizer is not None
    assert GlossaryGenerator is not None 