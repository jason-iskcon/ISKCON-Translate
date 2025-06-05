"""
Text normalizer for Bhagavad Gita content.

This module handles text normalization, deduplication, and transliteration
while preserving UTF-8 diacritics and generating ASCII fallbacks.
"""

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from .scraper import ScrapedContent

logger = logging.getLogger(__name__)


class TokenInfo:
    """Information about a normalized token."""
    
    def __init__(self, token: str, ascii_form: str):
        self.token = token
        self.ascii_form = ascii_form
        self.count = 0
        self.chapters: Set[int] = set()
        self.sources: Set[str] = set()
        self.contexts: List[str] = []
    
    def add_occurrence(self, chapter: Optional[int], source: str, context: str = "") -> None:
        """Add an occurrence of this token."""
        self.count += 1
        if chapter:
            self.chapters.add(chapter)
        self.sources.add(source)
        if context and len(self.contexts) < 5:  # Keep up to 5 example contexts
            self.contexts.append(context[:100])  # Truncate long contexts
    
    def to_dict(self) -> Dict[str, Union[str, int, List]]:
        """Convert to dictionary for CSV export."""
        return {
            'token': self.token,
            'ascii': self.ascii_form,
            'count': self.count,
            'chapters': ','.join(map(str, sorted(self.chapters))),
            'sources': ','.join(sorted(self.sources)),
            'contexts': ' | '.join(self.contexts)
        }


class TextNormalizer:
    """
    Text normalizer for Bhagavad Gita content.
    
    Features:
    - Preserves UTF-8 diacritics in original tokens
    - Generates ASCII fallbacks using indic_transliteration
    - Deduplicates content while tracking occurrences
    - Handles Sanskrit, Hindi, and English text
    - Maintains chapter and source information
    """
    
    def __init__(self, min_token_length: int = 2, max_token_length: int = 50):
        """
        Initialize the normalizer.
        
        Args:
            min_token_length: Minimum length for tokens to keep
            max_token_length: Maximum length for tokens to keep
        """
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        
        # Token storage
        self.tokens: Dict[str, TokenInfo] = {}
        self.duplicate_content: Set[str] = set()
        
        # Sanskrit/Devanagari patterns
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        self.sanskrit_pattern = re.compile(r'[a-zA-Zāīūṛṝḷḹēōṃḥṅñṭḍṇśṣḻ]+')
        
        # Common stop words to filter out
        self.stop_words = {
            # English
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'a', 'an', 'as', 'if',
            'when', 'where', 'why', 'how', 'what', 'who', 'which', 'whose',
            
            # Common Sanskrit/spiritual terms that are too common
            'ca', 'tu', 'hi', 'vai', 'iti', 'eva', 'api', 'atha', 'tatha',
            'yatha', 'tena', 'saha', 'cha', 'va', 'na', 'sa', 'te', 'me',
            'tvam', 'aham', 'mama', 'tava', 'asya', 'tasya', 'etasya',
        }
        
        logger.info(f"TextNormalizer initialized with {len(self.stop_words)} stop words")
    
    def normalize_content(self, scraped_content: List[ScrapedContent]) -> None:
        """
        Normalize and deduplicate scraped content.
        
        Args:
            scraped_content: List of scraped content items
        """
        logger.info(f"Normalizing {len(scraped_content)} content items")
        
        content_hashes = set()
        processed_items = 0
        duplicates_found = 0
        
        for item in scraped_content:
            # Check for duplicate content
            content_hash = self._hash_content(item.text)
            if content_hash in content_hashes:
                self.duplicate_content.add(content_hash)
                duplicates_found += 1
                continue
            
            content_hashes.add(content_hash)
            
            # Extract and normalize tokens
            tokens = self._extract_tokens(item.text)
            
            for token in tokens:
                normalized_token, ascii_form = self._normalize_token(token)
                
                if self._should_keep_token(normalized_token):
                    if normalized_token not in self.tokens:
                        self.tokens[normalized_token] = TokenInfo(normalized_token, ascii_form)
                    
                    # Get context around the token
                    context = self._get_token_context(item.text, token)
                    
                    self.tokens[normalized_token].add_occurrence(
                        item.chapter, item.source, context
                    )
            
            processed_items += 1
            
            if processed_items % 100 == 0:
                logger.debug(f"Processed {processed_items} items, found {len(self.tokens)} unique tokens")
        
        logger.info(f"Normalization complete: {processed_items} items processed, "
                   f"{duplicates_found} duplicates found, {len(self.tokens)} unique tokens")
    
    def _hash_content(self, text: str) -> str:
        """Create a hash for content deduplication."""
        # Normalize whitespace and case for comparison
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return str(hash(normalized))
    
    def _extract_tokens(self, text: str) -> List[str]:
        """Extract tokens from text."""
        # Split on various delimiters while preserving diacritics
        tokens = []
        
        # Split on whitespace and punctuation, but preserve word boundaries
        words = re.findall(r'\b[\w\u0900-\u097Fāīūṛṝḷḹēōṃḥṅñṭḍṇśṣḻ]+\b', text)
        
        for word in words:
            # Further split compound words if needed
            sub_tokens = self._split_compound_words(word)
            tokens.extend(sub_tokens)
        
        return tokens
    
    def _split_compound_words(self, word: str) -> List[str]:
        """Split compound words while preserving meaningful units."""
        # For now, just return the word as-is
        # This could be enhanced with more sophisticated compound word detection
        return [word]
    
    def _normalize_token(self, token: str) -> Tuple[str, str]:
        """
        Normalize a token and generate ASCII fallback.
        
        Args:
            token: Original token
            
        Returns:
            Tuple of (normalized_token, ascii_form)
        """
        # Preserve the original token with diacritics
        normalized = token.strip()
        
        # Generate ASCII fallback
        ascii_form = self._generate_ascii_fallback(normalized)
        
        return normalized, ascii_form
    
    def _generate_ascii_fallback(self, token: str) -> str:
        """Generate ASCII fallback for a token."""
        # Check if token contains Devanagari script
        if self.devanagari_pattern.search(token):
            try:
                # Transliterate from Devanagari to IAST (International Alphabet of Sanskrit Transliteration)
                ascii_form = transliterate(token, sanscript.DEVANAGARI, sanscript.IAST)
            except Exception as e:
                logger.debug(f"Failed to transliterate '{token}': {e}")
                ascii_form = self._fallback_ascii_conversion(token)
        else:
            # For non-Devanagari text, remove diacritics
            ascii_form = self._remove_diacritics(token)
        
        return ascii_form.lower()
    
    def _remove_diacritics(self, text: str) -> str:
        """Remove diacritics from text to create ASCII version."""
        # Normalize to NFD (decomposed form)
        nfd = unicodedata.normalize('NFD', text)
        
        # Remove combining characters (diacritics)
        ascii_text = ''.join(
            char for char in nfd
            if unicodedata.category(char) != 'Mn'
        )
        
        return ascii_text
    
    def _fallback_ascii_conversion(self, token: str) -> str:
        """Fallback ASCII conversion for problematic tokens."""
        # Simple character-by-character conversion
        ascii_chars = []
        for char in token:
            if ord(char) < 128:
                ascii_chars.append(char)
            else:
                # Try to find a reasonable ASCII equivalent
                ascii_equiv = self._get_ascii_equivalent(char)
                if ascii_equiv:
                    ascii_chars.append(ascii_equiv)
        
        return ''.join(ascii_chars)
    
    def _get_ascii_equivalent(self, char: str) -> Optional[str]:
        """Get ASCII equivalent for a Unicode character."""
        # Common Sanskrit/Devanagari to ASCII mappings
        mappings = {
            'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṛ': 'r', 'ṝ': 'r',
            'ḷ': 'l', 'ḹ': 'l', 'ē': 'e', 'ō': 'o', 'ṃ': 'm',
            'ḥ': 'h', 'ṅ': 'n', 'ñ': 'n', 'ṭ': 't', 'ḍ': 'd',
            'ṇ': 'n', 'ś': 's', 'ṣ': 's', 'ḻ': 'l',
            # Add more mappings as needed
        }
        
        return mappings.get(char)
    
    def _should_keep_token(self, token: str) -> bool:
        """Determine if a token should be kept."""
        # Check length
        if len(token) < self.min_token_length or len(token) > self.max_token_length:
            return False
        
        # Check if it's a stop word
        if token.lower() in self.stop_words:
            return False
        
        # Check if it's all digits
        if token.isdigit():
            return False
        
        # Check if it's all punctuation
        if all(not char.isalnum() and not self._is_sanskrit_char(char) for char in token):
            return False
        
        return True
    
    def _is_sanskrit_char(self, char: str) -> bool:
        """Check if character is a Sanskrit/Devanagari character."""
        return (
            '\u0900' <= char <= '\u097F' or  # Devanagari
            char in 'āīūṛṝḷḹēōṃḥṅñṭḍṇśṣḻ'  # Common Sanskrit diacritics
        )
    
    def _get_token_context(self, text: str, token: str, context_size: int = 30) -> str:
        """Get context around a token in the text."""
        token_pos = text.lower().find(token.lower())
        if token_pos == -1:
            return ""
        
        start = max(0, token_pos - context_size)
        end = min(len(text), token_pos + len(token) + context_size)
        
        context = text[start:end].strip()
        return context
    
    def export_to_csv(self, output_file: Union[str, Path] = "gita_tokens.csv") -> None:
        """
        Export normalized tokens to CSV file.
        
        Args:
            output_file: Path to output CSV file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tokens to list of dictionaries
        token_data = []
        for token_info in self.tokens.values():
            token_data.append(token_info.to_dict())
        
        # Sort by count (descending)
        token_data.sort(key=lambda x: x['count'], reverse=True)
        
        # Create DataFrame and save
        df = pd.DataFrame(token_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Exported {len(token_data)} tokens to {output_path}")
    
    def get_top_tokens(self, n: int = 200) -> List[Tuple[str, int]]:
        """
        Get top N tokens by frequency.
        
        Args:
            n: Number of top tokens to return
            
        Returns:
            List of (token, count) tuples
        """
        sorted_tokens = sorted(
            self.tokens.items(),
            key=lambda x: x[1].count,
            reverse=True
        )
        
        return [(token, info.count) for token, info in sorted_tokens[:n]]
    
    def get_tokens_by_chapter(self, chapter: int) -> List[Tuple[str, int]]:
        """
        Get tokens that appear in a specific chapter.
        
        Args:
            chapter: Chapter number
            
        Returns:
            List of (token, count) tuples for that chapter
        """
        chapter_tokens = []
        
        for token, info in self.tokens.items():
            if chapter in info.chapters:
                # Count occurrences in this chapter (approximation)
                chapter_count = info.count // len(info.chapters) if info.chapters else 0
                chapter_tokens.append((token, chapter_count))
        
        # Sort by count
        chapter_tokens.sort(key=lambda x: x[1], reverse=True)
        
        return chapter_tokens
    
    def get_stats(self) -> Dict[str, Union[int, float, List]]:
        """Get normalization statistics."""
        if not self.tokens:
            return {}
        
        total_occurrences = sum(info.count for info in self.tokens.values())
        avg_count = total_occurrences / len(self.tokens)
        
        # Get chapter distribution
        all_chapters = set()
        for info in self.tokens.values():
            all_chapters.update(info.chapters)
        
        # Get source distribution
        all_sources = set()
        for info in self.tokens.values():
            all_sources.update(info.sources)
        
        return {
            'unique_tokens': len(self.tokens),
            'total_occurrences': total_occurrences,
            'average_count_per_token': avg_count,
            'chapters_covered': sorted(list(all_chapters)),
            'sources_covered': sorted(list(all_sources)),
            'duplicates_found': len(self.duplicate_content),
            'tokens_with_diacritics': sum(
                1 for info in self.tokens.values()
                if info.token != info.ascii_form
            )
        } 