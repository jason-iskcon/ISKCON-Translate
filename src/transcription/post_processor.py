"""
Fast fuzzy post-correction for transcription output.

This module provides rapid fuzzy matching and correction of transcribed text
using Levenshtein distance to replace near-matches with canonical terms.
"""

import time
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict

try:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import Levenshtein
except ImportError:
    raise ImportError("rapidfuzz is required for post-processing. Install with: pip install rapidfuzz>=3.0.0")

try:
    from ..logging_utils import get_logger
except ImportError:
    from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CorrectionResult:
    """Result of post-correction processing."""
    original_text: str
    corrected_text: str
    corrections_made: List[Dict[str, str]]
    processing_time: float
    confidence_score: float


@dataclass
class CorrectionMatch:
    """A single correction match."""
    original: str
    canonical: str
    distance: int
    confidence: float
    position: int


class FuzzyPostProcessor:
    """
    Fast fuzzy post-processor for transcription correction.
    
    Uses rapidfuzz for efficient Levenshtein distance matching to correct
    common transcription errors with canonical terms.
    """
    
    def __init__(self, 
                 max_distance: int = 2,
                 min_confidence: float = 80.0,
                 min_word_length: int = 3):
        """
        Initialize the fuzzy post-processor.
        
        Args:
            max_distance: Maximum Levenshtein distance for matches
            min_confidence: Minimum confidence score for corrections
            min_word_length: Minimum word length to consider for correction
        """
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        self.min_word_length = min_word_length
        
        # Canonical terms dictionary
        self.canonical_terms: Dict[str, str] = {}
        self.term_frequencies: Dict[str, int] = defaultdict(int)
        
        # Performance tracking
        self.correction_count = 0
        self.total_processing_time = 0.0
        self.cache: Dict[str, str] = {}  # Cache for frequent corrections
        
        # Default ISKCON/spiritual terms
        self._load_default_terms()
        
        logger.info(f"FuzzyPostProcessor initialized: max_distance={max_distance}, "
                   f"min_confidence={min_confidence}, terms={len(self.canonical_terms)}")
    
    def _load_default_terms(self) -> None:
        """Load default canonical terms for ISKCON content."""
        default_terms = {
            # Core names and concepts
            "krishna": "Krishna",
            "krsna": "Krishna", 
            "krisna": "Krishna",
            "arjuna": "Arjuna",
            "arjun": "Arjuna",
            "prabhupada": "Prabhupada",
            "prabhupad": "Prabhupada",
            "bhagavad": "Bhagavad",
            "bhagwad": "Bhagavad",
            "gita": "Gita",
            "geeta": "Gita",
            "srimad": "Srimad",
            "shrimad": "Srimad",
            "bhagavatam": "Bhagavatam",
            "bhagwatam": "Bhagavatam",
            
            # Spiritual concepts
            "dharma": "dharma",
            "dharm": "dharma",
            "karma": "karma",
            "karm": "karma",
            "bhakti": "bhakti",
            "bhakty": "bhakti",
            "yoga": "yoga",
            "yog": "yoga",
            "moksha": "moksha",
            "moksh": "moksha",
            "samsara": "samsara",
            "samsar": "samsara",
            
            # Practices and terms
            "mantra": "mantra",
            "mantr": "mantra",
            "kirtan": "kirtan",
            "kirtana": "kirtan",
            "prasadam": "prasadam",
            "prasad": "prasadam",
            "devotee": "devotee",
            "devote": "devotee",
            "devotees": "devotees",
            "temple": "temple",
            "templ": "temple",
            "deity": "deity",
            "diety": "deity",
            
            # Places
            "vrindavan": "Vrindavan",
            "vrindaban": "Vrindavan",
            "vrndavan": "Vrindavan",
            "mayapur": "Mayapur",
            "mayapura": "Mayapur",
            "iskcon": "ISKCON",
            "iskon": "ISKCON",
            
            # Sanskrit terms
            "vedic": "Vedic",
            "vaidic": "Vedic",
            "sanskrit": "Sanskrit",
            "sanskrt": "Sanskrit",
            "guru": "guru",
            "gur": "guru",
            "consciousness": "consciousness",
            "consciousnes": "consciousness",
            "transcendental": "transcendental",
            "transcendentl": "transcendental",
            "spiritual": "spiritual",
            "spiritul": "spiritual",
            "meditation": "meditation",
            "meditaton": "meditation",
            "chanting": "chanting",
            "chantng": "chanting",
            "philosophy": "philosophy",
            "philosphy": "philosophy",
            "scripture": "scripture",
            "scriptur": "scripture",
            "verse": "verse",
            "vers": "verse"
        }
        
        # Convert to lowercase keys for case-insensitive matching
        self.canonical_terms = {k.lower(): v for k, v in default_terms.items()}
        
        logger.debug(f"Loaded {len(self.canonical_terms)} default canonical terms")
    
    def load_terms_from_file(self, file_path: Union[str, Path]) -> bool:
        """
        Load canonical terms from a file.
        
        Args:
            file_path: Path to file containing canonical terms
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"Terms file not found: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Support format: "incorrect_term -> correct_term" or just "correct_term"
                    if '->' in line:
                        incorrect, correct = line.split('->', 1)
                        self.canonical_terms[incorrect.strip().lower()] = correct.strip()
                    else:
                        # Just a canonical term - add common variations
                        term = line.strip()
                        self.canonical_terms[term.lower()] = term
            
            logger.info(f"Loaded {len(self.canonical_terms)} terms from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load terms from {file_path}: {e}")
            return False
    
    def add_canonical_term(self, canonical: str, variations: Optional[List[str]] = None) -> None:
        """
        Add a canonical term with optional variations.
        
        Args:
            canonical: The correct canonical form
            variations: List of common misspellings/variations
        """
        # Add the canonical term itself
        self.canonical_terms[canonical.lower()] = canonical
        
        # Add variations if provided
        if variations:
            for variation in variations:
                self.canonical_terms[variation.lower()] = canonical
        
        logger.debug(f"Added canonical term: {canonical} with {len(variations or [])} variations")
    
    def correct_text(self, text: str) -> CorrectionResult:
        """
        Correct text using fuzzy matching.
        
        Args:
            text: Input text to correct
            
        Returns:
            CorrectionResult: Correction result with metadata
        """
        start_time = time.perf_counter()
        
        if not text.strip():
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                corrections_made=[],
                processing_time=0.0,
                confidence_score=1.0
            )
        
        # Split text into words while preserving punctuation and spacing
        words = self._tokenize_text(text)
        corrected_words = []
        corrections_made = []
        total_confidence = 0.0
        word_count = 0
        
        for i, word_info in enumerate(words):
            word = word_info['word']
            is_word = word_info['is_word']
            
            if is_word and len(word) >= self.min_word_length:
                # Try to correct this word
                correction = self._find_correction(word, i)
                if correction:
                    corrected_words.append(correction.canonical)
                    corrections_made.append({
                        'original': correction.original,
                        'corrected': correction.canonical,
                        'position': correction.position,
                        'distance': correction.distance,
                        'confidence': correction.confidence
                    })
                    total_confidence += correction.confidence
                else:
                    corrected_words.append(word)
                    total_confidence += 100.0  # Perfect confidence for unchanged words
                word_count += 1
            else:
                # Keep punctuation and short words as-is
                corrected_words.append(word)
        
        # Calculate overall confidence
        overall_confidence = total_confidence / word_count if word_count > 0 else 100.0
        
        # Reconstruct text
        corrected_text = ''.join(corrected_words)
        
        # Update statistics
        self.correction_count += 1
        processing_time = time.perf_counter() - start_time
        self.total_processing_time += processing_time
        
        result = CorrectionResult(
            original_text=text,
            corrected_text=corrected_text,
            corrections_made=corrections_made,
            processing_time=processing_time,
            confidence_score=overall_confidence
        )
        
        if corrections_made:
            logger.debug(f"Made {len(corrections_made)} corrections in {processing_time*1000:.2f}ms")
        
        return result
    
    def _tokenize_text(self, text: str) -> List[Dict[str, Union[str, bool]]]:
        """
        Tokenize text while preserving spacing and punctuation.
        
        Args:
            text: Input text
            
        Returns:
            List of token dictionaries with word and is_word flag
        """
        tokens = []
        current_pos = 0
        
        # Find all word boundaries
        for match in re.finditer(r'\b\w+\b', text):
            start, end = match.span()
            
            # Add any non-word characters before this word
            if start > current_pos:
                tokens.append({
                    'word': text[current_pos:start],
                    'is_word': False
                })
            
            # Add the word
            tokens.append({
                'word': text[start:end],
                'is_word': True
            })
            
            current_pos = end
        
        # Add any remaining non-word characters
        if current_pos < len(text):
            tokens.append({
                'word': text[current_pos:],
                'is_word': False
            })
        
        return tokens
    
    def _find_correction(self, word: str, position: int) -> Optional[CorrectionMatch]:
        """
        Find the best correction for a word.
        
        Args:
            word: Word to correct
            position: Position in text
            
        Returns:
            CorrectionMatch if correction found, None otherwise
        """
        word_lower = word.lower()
        
        # Check cache first
        if word_lower in self.cache:
            cached_result = self.cache[word_lower]
            if cached_result == word:
                return None  # No correction needed
            return CorrectionMatch(
                original=word,
                canonical=cached_result,
                distance=0,  # Cached, so assume good
                confidence=95.0,
                position=position
            )
        
        # Check for exact match first
        if word_lower in self.canonical_terms:
            canonical = self.canonical_terms[word_lower]
            if canonical != word:  # Case correction needed
                self.cache[word_lower] = canonical
                return CorrectionMatch(
                    original=word,
                    canonical=canonical,
                    distance=0,
                    confidence=100.0,
                    position=position
                )
            return None
        
        # Find fuzzy matches
        best_match = None
        best_score = 0.0
        
        # Use rapidfuzz for efficient fuzzy matching
        matches = process.extract(
            word_lower,
            self.canonical_terms.keys(),
            scorer=fuzz.ratio,
            limit=5,
            score_cutoff=self.min_confidence
        )
        
        for match_word, score, _ in matches:
            # Calculate Levenshtein distance
            distance = Levenshtein.distance(word_lower, match_word)
            
            if distance <= self.max_distance and score > best_score:
                best_match = match_word
                best_score = score
        
        if best_match:
            canonical = self.canonical_terms[best_match]
            distance = Levenshtein.distance(word_lower, best_match)
            
            # Cache the result
            self.cache[word_lower] = canonical
            
            return CorrectionMatch(
                original=word,
                canonical=canonical,
                distance=distance,
                confidence=best_score,
                position=position
            )
        
        # Cache negative result
        self.cache[word_lower] = word
        return None
    
    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """Get performance statistics."""
        avg_time = (self.total_processing_time / self.correction_count 
                   if self.correction_count > 0 else 0.0)
        
        return {
            'correction_count': self.correction_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_time,
            'canonical_terms': len(self.canonical_terms),
            'cache_size': len(self.cache),
            'cache_hit_rate': len(self.cache) / max(1, self.correction_count)
        }
    
    def clear_cache(self) -> None:
        """Clear the correction cache."""
        self.cache.clear()
        logger.debug("Correction cache cleared")
    
    def get_correction_suggestions(self, word: str, limit: int = 5) -> List[Tuple[str, float, int]]:
        """
        Get correction suggestions for a word.
        
        Args:
            word: Word to get suggestions for
            limit: Maximum number of suggestions
            
        Returns:
            List of (canonical_term, confidence, distance) tuples
        """
        word_lower = word.lower()
        suggestions = []
        
        # Use rapidfuzz to find matches
        matches = process.extract(
            word_lower,
            self.canonical_terms.keys(),
            scorer=fuzz.ratio,
            limit=limit,
            score_cutoff=50.0  # Lower threshold for suggestions
        )
        
        for match_word, score, _ in matches:
            distance = Levenshtein.distance(word_lower, match_word)
            canonical = self.canonical_terms[match_word]
            suggestions.append((canonical, score, distance))
        
        # Sort by confidence (score) descending
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return suggestions


# Convenience functions
def correct_transcription(text: str, 
                         max_distance: int = 2,
                         min_confidence: float = 80.0,
                         terms_file: Optional[str] = None) -> CorrectionResult:
    """
    Quick transcription correction function.
    
    Args:
        text: Text to correct
        max_distance: Maximum Levenshtein distance
        min_confidence: Minimum confidence for corrections
        terms_file: Optional file with additional canonical terms
        
    Returns:
        CorrectionResult: Correction result
    """
    processor = FuzzyPostProcessor(max_distance, min_confidence)
    
    if terms_file:
        processor.load_terms_from_file(terms_file)
    
    return processor.correct_text(text)


def create_post_processor(config: Optional[Dict[str, Union[str, int, float]]] = None) -> FuzzyPostProcessor:
    """
    Create a configured post-processor.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FuzzyPostProcessor: Configured processor
    """
    config = config or {}
    
    processor = FuzzyPostProcessor(
        max_distance=config.get('max_distance', 2),
        min_confidence=config.get('min_confidence', 80.0),
        min_word_length=config.get('min_word_length', 3)
    )
    
    # Load additional terms if specified
    terms_file = config.get('terms_file')
    if terms_file:
        processor.load_terms_from_file(terms_file)
    
    return processor 