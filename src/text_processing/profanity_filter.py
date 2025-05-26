"""
Advanced profanity filtering system for real-time caption processing.

This module provides multi-layer profanity detection with context-aware filtering,
smart replacement strategies, and minimal false positives.
"""

import re
import string
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from ..logging_utils import get_logger
except ImportError:
    from src.logging_utils import get_logger

logger = get_logger(__name__)


class FilterLevel(Enum):
    """Profanity filtering levels."""
    STRICT = "strict"      # Filter all potentially offensive content
    MODERATE = "moderate"  # Filter clear profanity, allow mild language
    MILD = "mild"         # Filter only strong profanity
    OFF = "off"           # No filtering


class ReplacementStrategy(Enum):
    """Strategies for replacing filtered content."""
    BEEP = "beep"         # Replace with [BEEP]
    ASTERISK = "asterisk" # Replace with f***
    REMOVE = "remove"     # Remove completely
    SYNONYM = "synonym"   # Replace with appropriate synonym


@dataclass
class FilterResult:
    """Result of profanity filtering operation."""
    original_text: str
    filtered_text: str
    detections: List[Dict]
    confidence: float
    processing_time: float


class ProfanityFilter:
    """
    Advanced profanity filtering with context awareness and smart replacement.
    
    Features:
    - Multi-level filtering (strict, moderate, mild)
    - Context-aware detection to avoid false positives
    - Multiple replacement strategies
    - Real-time processing optimized
    - Religious/spiritual context consideration
    """
    
    def __init__(self, 
                 filter_level: FilterLevel = FilterLevel.MODERATE,
                 replacement_strategy: ReplacementStrategy = ReplacementStrategy.BEEP,
                 custom_words: Optional[List[str]] = None):
        """
        Initialize the profanity filter.
        
        Args:
            filter_level: Level of filtering to apply
            replacement_strategy: How to replace detected profanity
            custom_words: Additional words to filter
        """
        self.filter_level = filter_level
        self.replacement_strategy = replacement_strategy
        
        # Load word lists
        self.profanity_words = self._load_profanity_words()
        self.mild_words = self._load_mild_words()
        self.context_exceptions = self._load_context_exceptions()
        self.religious_terms = self._load_religious_terms()
        
        # Add custom words if provided
        if custom_words:
            self.profanity_words.update(word.lower() for word in custom_words)
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info(f"Profanity filter initialized: level={filter_level.value}, "
                   f"strategy={replacement_strategy.value}, "
                   f"words_loaded={len(self.profanity_words)}")
    
    def _load_profanity_words(self) -> Set[str]:
        """Load the main profanity word list."""
        # Core profanity words (keeping this family-friendly in the code)
        words = {
            # Strong profanity
            "fuck", "fucking", "fucked", "fucker", "fucks",
            "shit", "shitting", "shitted", "shits",
            "bitch", "bitching", "bitches",
            "damn", "damned", "damning",
            "hell", "hells",
            "ass", "asses", "asshole", "assholes",
            "crap", "crappy", "craps",
            "piss", "pissed", "pissing",
            "bastard", "bastards",
            "whore", "whores",
            "slut", "sluts",
            
            # Variations and common misspellings
            "f*ck", "f**k", "f***", "fck", "fuk",
            "sh*t", "sh**", "sht",
            "b*tch", "b**ch", "btch",
            "d*mn", "d**n", "dmn",
            "h*ll", "h**l", "hll",
            "a*s", "a**", "azz",
            "cr*p", "cr**",
            "p*ss", "p**s",
            
            # Numbers/symbols substitutions
            "f4ck", "sh1t", "b1tch", "d4mn", "h3ll",
            "fvck", "shyt", "bytch", "dayum",
        }
        
        return words
    
    def _load_mild_words(self) -> Set[str]:
        """Load mild profanity that might be acceptable in some contexts."""
        return {
            "stupid", "idiot", "moron", "dumb", "dumbass",
            "crazy", "insane", "nuts", "mental",
            "suck", "sucks", "sucked", "sucking",
            "hate", "hated", "hating", "hates",
        }
    
    def _load_context_exceptions(self) -> Set[str]:
        """Load words that might trigger false positives in certain contexts."""
        return {
            # Place names that contain profanity substrings
            "scunthorpe", "penistone", "cockburn", "fucking", "hell",
            # Technical terms
            "class", "assignment", "assumption",
            # Religious terms that might be misinterpreted
            "hell", "damn", "god", "jesus", "christ",
        }
    
    def _load_religious_terms(self) -> Set[str]:
        """Load religious/spiritual terms that should be handled carefully."""
        return {
            # ISKCON/Hindu terms
            "krishna", "rama", "vishnu", "shiva", "brahma",
            "hare", "mantra", "bhakti", "yoga", "dharma",
            "karma", "samsara", "moksha", "guru", "swami",
            "temple", "deity", "prasadam", "kirtan", "bhajan",
            
            # General religious terms
            "god", "lord", "divine", "sacred", "holy",
            "prayer", "worship", "faith", "spirit", "soul",
            "heaven", "paradise", "blessed", "grace",
            
            # Terms that might be misfiltered
            "hell", "damn", "jesus", "christ", "allah",
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # Create word boundary patterns for exact matches
        profanity_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in self.profanity_words) + r')\b'
        self.profanity_regex = re.compile(profanity_pattern, re.IGNORECASE)
        
        # Pattern for detecting leetspeak and symbol substitutions
        self.leetspeak_patterns = [
            (r'f[u4@]ck', 'f***'),
            (r'sh[i1!]t', 's***'),
            (r'b[i1!]tch', 'b****'),
            (r'd[a@4]mn', 'd***'),
            (r'h[e3]ll', 'h***'),
            (r'[a@4]ss', 'a**'),
        ]
        
        self.leetspeak_regex = [(re.compile(pattern, re.IGNORECASE), replacement) 
                               for pattern, replacement in self.leetspeak_patterns]
    
    def filter_text(self, text: str, context: Optional[str] = None) -> FilterResult:
        """
        Filter profanity from text with context awareness.
        
        Args:
            text: Text to filter
            context: Additional context for better filtering decisions
            
        Returns:
            FilterResult with original text, filtered text, and metadata
        """
        import time
        start_time = time.perf_counter()
        
        if self.filter_level == FilterLevel.OFF:
            return FilterResult(
                original_text=text,
                filtered_text=text,
                detections=[],
                confidence=1.0,
                processing_time=time.perf_counter() - start_time
            )
        
        detections = []
        filtered_text = text
        
        # 1. Check for religious/spiritual context
        is_religious_context = self._is_religious_context(text, context)
        
        # 2. Detect and filter profanity
        filtered_text, word_detections = self._filter_profanity_words(filtered_text, is_religious_context)
        detections.extend(word_detections)
        
        # 3. Handle leetspeak and symbol substitutions
        filtered_text, leetspeak_detections = self._filter_leetspeak(filtered_text)
        detections.extend(leetspeak_detections)
        
        # 4. Apply mild word filtering based on level
        if self.filter_level == FilterLevel.STRICT:
            filtered_text, mild_detections = self._filter_mild_words(filtered_text)
            detections.extend(mild_detections)
        
        # 5. Clean up multiple spaces and formatting
        filtered_text = self._clean_text(filtered_text)
        
        # Calculate confidence based on detection accuracy
        confidence = self._calculate_confidence(text, filtered_text, detections)
        
        processing_time = time.perf_counter() - start_time
        
        if detections:
            logger.debug(f"Filtered {len(detections)} items from text: {text[:50]}...")
        
        return FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            detections=detections,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def _is_religious_context(self, text: str, context: Optional[str] = None) -> bool:
        """Check if the text is in a religious/spiritual context."""
        combined_text = (text + " " + (context or "")).lower()
        
        # Check for religious terms
        religious_count = sum(1 for term in self.religious_terms 
                            if term in combined_text)
        
        # If we find multiple religious terms, likely religious context
        return religious_count >= 2
    
    def _filter_profanity_words(self, text: str, is_religious_context: bool) -> Tuple[str, List[Dict]]:
        """Filter main profanity words with context awareness."""
        detections = []
        
        def replace_match(match):
            word = match.group().lower()
            original_word = match.group()
            
            # Check for context exceptions
            if is_religious_context and word in self.context_exceptions:
                # In religious context, be more lenient with certain words
                if word in ["hell", "damn"]:
                    return original_word  # Keep religious references
            
            # Apply replacement strategy
            replacement = self._get_replacement(original_word, word)
            
            detections.append({
                'type': 'profanity',
                'original': original_word,
                'replacement': replacement,
                'position': match.span(),
                'confidence': 0.95
            })
            
            return replacement
        
        filtered_text = self.profanity_regex.sub(replace_match, text)
        return filtered_text, detections
    
    def _filter_leetspeak(self, text: str) -> Tuple[str, List[Dict]]:
        """Filter leetspeak and symbol-substituted profanity."""
        detections = []
        filtered_text = text
        
        for pattern, replacement in self.leetspeak_regex:
            def replace_match(match):
                detections.append({
                    'type': 'leetspeak',
                    'original': match.group(),
                    'replacement': replacement,
                    'position': match.span(),
                    'confidence': 0.85
                })
                return replacement
            
            filtered_text = pattern.sub(replace_match, filtered_text)
        
        return filtered_text, detections
    
    def _filter_mild_words(self, text: str) -> Tuple[str, List[Dict]]:
        """Filter mild profanity (only in strict mode)."""
        detections = []
        words = text.split()
        filtered_words = []
        
        for i, word in enumerate(words):
            clean_word = word.lower().strip(string.punctuation)
            
            if clean_word in self.mild_words:
                replacement = self._get_replacement(word, clean_word)
                detections.append({
                    'type': 'mild_profanity',
                    'original': word,
                    'replacement': replacement,
                    'position': (i, i+1),
                    'confidence': 0.75
                })
                filtered_words.append(replacement)
            else:
                filtered_words.append(word)
        
        return ' '.join(filtered_words), detections
    
    def _get_replacement(self, original_word: str, clean_word: str) -> str:
        """Get replacement text based on strategy."""
        if self.replacement_strategy == ReplacementStrategy.BEEP:
            return "[BEEP]"
        
        elif self.replacement_strategy == ReplacementStrategy.ASTERISK:
            if len(clean_word) <= 2:
                return "*" * len(clean_word)
            else:
                return clean_word[0] + "*" * (len(clean_word) - 2) + clean_word[-1]
        
        elif self.replacement_strategy == ReplacementStrategy.REMOVE:
            return ""
        
        elif self.replacement_strategy == ReplacementStrategy.SYNONYM:
            # Simple synonym replacement
            synonyms = {
                "fuck": "darn", "shit": "crud", "damn": "darn",
                "bitch": "person", "hell": "heck", "ass": "butt",
                "stupid": "silly", "idiot": "person", "hate": "dislike"
            }
            return synonyms.get(clean_word, "[FILTERED]")
        
        return "[FILTERED]"
    
    def _clean_text(self, text: str) -> str:
        """Clean up text formatting after filtering."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        return text
    
    def _calculate_confidence(self, original: str, filtered: str, detections: List[Dict]) -> float:
        """Calculate confidence score for the filtering operation."""
        if not detections:
            return 1.0
        
        # Base confidence on detection types and patterns
        total_confidence = 0.0
        for detection in detections:
            total_confidence += detection['confidence']
        
        # Average confidence, weighted by number of detections
        avg_confidence = total_confidence / len(detections) if detections else 1.0
        
        # Adjust based on text length and detection ratio
        detection_ratio = len(detections) / max(1, len(original.split()))
        if detection_ratio > 0.5:  # Too many detections might indicate false positives
            avg_confidence *= 0.8
        
        return min(1.0, avg_confidence)
    
    def update_filter_level(self, level: FilterLevel):
        """Update the filtering level."""
        self.filter_level = level
        logger.info(f"Profanity filter level updated to: {level.value}")
    
    def update_replacement_strategy(self, strategy: ReplacementStrategy):
        """Update the replacement strategy."""
        self.replacement_strategy = strategy
        logger.info(f"Profanity filter replacement strategy updated to: {strategy.value}")
    
    def add_custom_words(self, words: List[str]):
        """Add custom words to the filter list."""
        for word in words:
            self.profanity_words.add(word.lower())
        
        # Recompile patterns
        self._compile_patterns()
        logger.info(f"Added {len(words)} custom words to profanity filter")
    
    def get_stats(self) -> Dict:
        """Get filter statistics."""
        return {
            'filter_level': self.filter_level.value,
            'replacement_strategy': self.replacement_strategy.value,
            'profanity_words_count': len(self.profanity_words),
            'mild_words_count': len(self.mild_words),
            'religious_terms_count': len(self.religious_terms),
            'context_exceptions_count': len(self.context_exceptions)
        }


# Convenience function for quick filtering
def filter_profanity(text: str, 
                    level: FilterLevel = FilterLevel.MODERATE,
                    strategy: ReplacementStrategy = ReplacementStrategy.BEEP) -> str:
    """
    Quick profanity filtering function.
    
    Args:
        text: Text to filter
        level: Filtering level
        strategy: Replacement strategy
        
    Returns:
        Filtered text
    """
    filter_instance = ProfanityFilter(level, strategy)
    result = filter_instance.filter_text(text)
    return result.filtered_text 