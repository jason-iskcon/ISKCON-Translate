"""
Glossary generator for Bhagavad Gita content.

This module generates frequency-based glossaries including common terms
and chapter-specific vocabularies ready for inference.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class GlossaryGenerator:
    """
    Generator for frequency-based glossaries.
    
    Features:
    - Generates common vocabulary lists (top N terms)
    - Creates chapter-specific glossaries
    - Sorts terms by frequency (descending)
    - Outputs ready-to-use glossary files for inference
    - Supports both UTF-8 and ASCII formats
    """
    
    def __init__(self, normalizer: TextNormalizer):
        """
        Initialize the generator.
        
        Args:
            normalizer: Initialized TextNormalizer with processed tokens
        """
        self.normalizer = normalizer
        self.output_dir = Path("glossaries")
        
        logger.info("GlossaryGenerator initialized")
    
    def generate_all_glossaries(
        self,
        output_dir: Union[str, Path] = "glossaries",
        common_count: int = 200,
        chapter_count: int = 100,
        include_ascii: bool = True
    ) -> None:
        """
        Generate all glossary files.
        
        Args:
            output_dir: Directory to save glossary files
            common_count: Number of terms in common glossary
            chapter_count: Number of terms per chapter glossary
            include_ascii: Whether to include ASCII versions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating glossaries in {self.output_dir}")
        
        # Generate common glossary
        self.generate_common_glossary(common_count, include_ascii)
        
        # Generate chapter-specific glossaries
        self.generate_chapter_glossaries(chapter_count, include_ascii)
        
        # Generate summary statistics
        self.generate_summary_stats()
        
        logger.info("All glossaries generated successfully")
    
    def generate_common_glossary(
        self,
        count: int = 200,
        include_ascii: bool = True
    ) -> None:
        """
        Generate common vocabulary glossary.
        
        Args:
            count: Number of top terms to include
            include_ascii: Whether to include ASCII versions
        """
        logger.info(f"Generating common glossary with top {count} terms")
        
        # Get top tokens
        top_tokens = self.normalizer.get_top_tokens(count)
        
        # Generate UTF-8 version
        utf8_file = self.output_dir / f"common_{count}.txt"
        self._write_glossary_file(utf8_file, top_tokens, format_type="utf8")
        
        # Generate ASCII version if requested
        if include_ascii:
            ascii_file = self.output_dir / f"common_{count}_ascii.txt"
            ascii_tokens = [
                (self.normalizer.tokens[token].ascii_form, freq)
                for token, freq in top_tokens
            ]
            self._write_glossary_file(ascii_file, ascii_tokens, format_type="ascii")
        
        logger.info(f"Common glossary saved: {utf8_file}")
    
    def generate_chapter_glossaries(
        self,
        count: int = 100,
        include_ascii: bool = True
    ) -> None:
        """
        Generate chapter-specific glossaries.
        
        Args:
            count: Number of terms per chapter
            include_ascii: Whether to include ASCII versions
        """
        logger.info(f"Generating chapter glossaries with {count} terms each")
        
        # Create by_chapter directory
        chapter_dir = self.output_dir / "by_chapter"
        chapter_dir.mkdir(exist_ok=True)
        
        # Get all chapters that have content
        stats = self.normalizer.get_stats()
        chapters = stats.get('chapters_covered', [])
        
        for chapter in chapters:
            if 1 <= chapter <= 18:  # Bhagavad Gita has 18 chapters
                self._generate_single_chapter_glossary(
                    chapter, count, include_ascii, chapter_dir
                )
        
        logger.info(f"Generated glossaries for {len(chapters)} chapters")
    
    def _generate_single_chapter_glossary(
        self,
        chapter: int,
        count: int,
        include_ascii: bool,
        chapter_dir: Path
    ) -> None:
        """Generate glossary for a single chapter."""
        logger.debug(f"Generating glossary for chapter {chapter}")
        
        # Get tokens for this chapter
        chapter_tokens = self.normalizer.get_tokens_by_chapter(chapter)
        
        if not chapter_tokens:
            logger.warning(f"No tokens found for chapter {chapter}")
            return
        
        # Limit to requested count
        chapter_tokens = chapter_tokens[:count]
        
        # Generate UTF-8 version
        utf8_file = chapter_dir / f"{chapter:02d}.txt"
        self._write_glossary_file(utf8_file, chapter_tokens, format_type="utf8")
        
        # Generate ASCII version if requested
        if include_ascii:
            ascii_file = chapter_dir / f"{chapter:02d}_ascii.txt"
            ascii_tokens = [
                (self.normalizer.tokens[token].ascii_form, freq)
                for token, freq in chapter_tokens
                if token in self.normalizer.tokens
            ]
            self._write_glossary_file(ascii_file, ascii_tokens, format_type="ascii")
        
        logger.debug(f"Chapter {chapter} glossary saved: {utf8_file}")
    
    def _write_glossary_file(
        self,
        file_path: Path,
        tokens: List[Tuple[str, int]],
        format_type: str = "utf8"
    ) -> None:
        """
        Write glossary file.
        
        Args:
            file_path: Path to output file
            tokens: List of (token, frequency) tuples
            format_type: Format type ("utf8" or "ascii")
        """
        # Reverse the order so most frequent terms are at the bottom
        # This is important for Whisper which processes glossaries from bottom to top
        reversed_tokens = list(reversed(tokens))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            # Write header comment
            f.write(f"# Bhagavad Gita Glossary ({format_type.upper()})\n")
            f.write(f"# Generated from corpus analysis\n")
            f.write(f"# Terms sorted by frequency (ascending - most frequent at bottom)\n")
            f.write(f"# This order optimizes Whisper transcription accuracy\n")
            f.write(f"# Total terms: {len(tokens)}\n")
            f.write("#\n")
            
            # Write tokens (least frequent first, most frequent last)
            for token, frequency in reversed_tokens:
                f.write(f"{token}\n")
        
        logger.debug(f"Wrote {len(tokens)} terms to {file_path}")
    
    def generate_summary_stats(self) -> None:
        """Generate summary statistics file."""
        stats_file = self.output_dir / "README.md"
        
        stats = self.normalizer.get_stats()
        top_tokens = self.normalizer.get_top_tokens(20)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("# Bhagavad Gita Glossaries\n\n")
            f.write("Generated from comprehensive corpus analysis.\n\n")
            
            f.write("## Statistics\n\n")
            f.write(f"- **Unique tokens**: {stats.get('unique_tokens', 0):,}\n")
            f.write(f"- **Total occurrences**: {stats.get('total_occurrences', 0):,}\n")
            f.write(f"- **Chapters covered**: {len(stats.get('chapters_covered', []))}\n")
            f.write(f"- **Sources**: {', '.join(stats.get('sources_covered', []))}\n\n")
            
            f.write("## Files\n\n")
            f.write("- `common_200.txt` - Top 200 most frequent terms\n")
            f.write("- `by_chapter/01.txt` through `18.txt` - Chapter-specific terms\n\n")
            
            f.write("## Top 20 Terms\n\n")
            for i, (token, count) in enumerate(top_tokens, 1):
                f.write(f"{i:2d}. **{token}** - {count:,} occurrences\n")
        
        logger.info(f"Summary statistics saved to {stats_file}")
    
    def validate_glossaries(self) -> Dict[str, bool]:
        """Validate generated glossary files."""
        results = {}
        
        # Check common glossary
        common_file = self.output_dir / "common_200.txt"
        results['common_glossary'] = common_file.exists()
        
        # Check chapter glossaries
        chapter_dir = self.output_dir / "by_chapter"
        chapter_files = 0
        
        if chapter_dir.exists():
            for chapter in range(1, 19):
                chapter_file = chapter_dir / f"{chapter:02d}.txt"
                if chapter_file.exists():
                    chapter_files += 1
        
        results['chapter_glossaries'] = chapter_files >= 10
        results['all_valid'] = results['common_glossary'] and results['chapter_glossaries']
        
        return results 