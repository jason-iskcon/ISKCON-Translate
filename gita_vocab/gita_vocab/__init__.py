"""
Gita Vocab: Corpus scraper and glossary generator for Bhagavad Gita content.

This package provides tools for scraping ISKCON and Bhagavad Gita content,
normalizing text, and generating frequency-based glossaries for enhanced
transcription accuracy.
"""

from .scraper import GitaScraper
from .normalizer import TextNormalizer
from .generator import GlossaryGenerator

__version__ = "0.1.0"
__author__ = "ISKCON-Translate Team"
__description__ = "Corpus scraper and glossary generator for Bhagavad Gita content"

__all__ = [
    "GitaScraper",
    "TextNormalizer", 
    "GlossaryGenerator",
] 