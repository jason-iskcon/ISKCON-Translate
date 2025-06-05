"""
Pluggable glossary policy system for domain-specific transcription enhancement.

This module provides different strategies for selecting and applying glossaries
based on content, context, or configuration.
"""

import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

try:
    from ..logging_utils import get_logger
except ImportError:
    from src.logging_utils import get_logger

logger = get_logger(__name__)


class GlossaryStrategy(Enum):
    """Available glossary strategies."""
    STATIC_COMMON = "static_common"
    CHAPTER_GUESS = "chapter_guess"
    EMPTY = "empty"


@dataclass
class GlossaryResult:
    """Result of glossary policy application."""
    glossary_text: str
    strategy_used: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


class BaseGlossaryPolicy(ABC):
    """Base class for glossary policies."""
    
    def __init__(self, name: str):
        self.name = name
        self.application_count = 0
        self.total_processing_time = 0.0
    
    @abstractmethod
    def get_glossary(self, context: Dict[str, Any]) -> GlossaryResult:
        """
        Get glossary text based on context.
        
        Args:
            context: Context information (audio_path, previous_text, metadata, etc.)
            
        Returns:
            GlossaryResult: Glossary text and metadata
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get policy statistics."""
        avg_time = (self.total_processing_time / self.application_count 
                   if self.application_count > 0 else 0.0)
        
        return {
            'name': self.name,
            'application_count': self.application_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_time
        }


class StaticCommonPolicy(BaseGlossaryPolicy):
    """
    Static common glossary policy.
    
    Uses a predefined set of common domain-specific terms that are always applied.
    Ideal for consistent terminology across all transcriptions.
    """
    
    def __init__(self, glossary_path: Optional[Union[str, Path]] = None):
        super().__init__("static_common")
        self.glossary_path = glossary_path
        self.glossary_text = ""
        self.load_time = 0.0
        
        # Default ISKCON/spiritual terms if no file provided
        self.default_terms = [
            "Krishna", "Bhagavad Gita", "Srimad Bhagavatam", "Prabhupada",
            "ISKCON", "devotee", "devotees", "bhakti", "yoga", "dharma",
            "karma", "moksha", "samsara", "guru", "mantra", "kirtan",
            "prasadam", "temple", "deity", "Radha", "Govinda", "Vrindavan",
            "Mayapur", "spiritual", "consciousness", "transcendental",
            "meditation", "chanting", "Hare Krishna", "maha-mantra",
            "Vedic", "Sanskrit", "philosophy", "scripture", "verse"
        ]
        
        self._load_glossary()
    
    def _load_glossary(self) -> None:
        """Load glossary from file or use defaults."""
        start_time = time.perf_counter()
        
        try:
            if self.glossary_path and Path(self.glossary_path).exists():
                with open(self.glossary_path, 'r', encoding='utf-8') as f:
                    self.glossary_text = f.read().strip()
                logger.info(f"Loaded static glossary from {self.glossary_path}: "
                           f"{len(self.glossary_text)} characters")
            else:
                # Use default terms
                self.glossary_text = " ".join(self.default_terms)
                logger.info(f"Using default ISKCON glossary: {len(self.default_terms)} terms")
                
        except Exception as e:
            logger.error(f"Failed to load glossary, using defaults: {e}")
            self.glossary_text = " ".join(self.default_terms)
        
        self.load_time = time.perf_counter() - start_time
        logger.debug(f"Glossary loaded in {self.load_time:.3f}s")
    
    def get_glossary(self, context: Dict[str, Any]) -> GlossaryResult:
        """Get static common glossary."""
        start_time = time.perf_counter()
        
        self.application_count += 1
        processing_time = time.perf_counter() - start_time
        self.total_processing_time += processing_time
        
        return GlossaryResult(
            glossary_text=self.glossary_text,
            strategy_used=self.name,
            confidence=1.0,  # Always confident in static terms
            processing_time=processing_time,
            metadata={
                'source': str(self.glossary_path) if self.glossary_path else 'default',
                'term_count': len(self.glossary_text.split()),
                'load_time': self.load_time
            }
        )


class ChapterGuessPolicy(BaseGlossaryPolicy):
    """
    Chapter-based glossary policy.
    
    Attempts to guess the chapter or topic from audio filename, previous context,
    or content analysis, then applies chapter-specific terminology.
    """
    
    def __init__(self, glossary_dir: Optional[Union[str, Path]] = None):
        super().__init__("chapter_guess")
        self.glossary_dir = Path(glossary_dir) if glossary_dir else None
        self.chapter_glossaries = {}
        self.fallback_terms = [
            "Krishna", "Arjuna", "Bhagavad Gita", "dharma", "karma", "yoga"
        ]
        
        self._load_chapter_glossaries()
    
    def _load_chapter_glossaries(self) -> None:
        """Load chapter-specific glossaries from directory."""
        if not self.glossary_dir or not self.glossary_dir.exists():
            logger.warning(f"Glossary directory not found: {self.glossary_dir}")
            return
        
        try:
            for glossary_file in self.glossary_dir.glob("*.txt"):
                chapter_name = glossary_file.stem.lower()
                with open(glossary_file, 'r', encoding='utf-8') as f:
                    self.chapter_glossaries[chapter_name] = f.read().strip()
                
            logger.info(f"Loaded {len(self.chapter_glossaries)} chapter glossaries")
            
        except Exception as e:
            logger.error(f"Failed to load chapter glossaries: {e}")
    
    def _guess_chapter(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Guess chapter from context.
        
        Args:
            context: Context information
            
        Returns:
            Optional[str]: Guessed chapter name or None
        """
        # Try to extract from audio filename
        audio_path = context.get('audio_path', '')
        if audio_path:
            filename = Path(audio_path).stem.lower()
            
            # Look for chapter patterns
            chapter_patterns = [
                r'chapter[_\s]*(\d+)',
                r'ch[_\s]*(\d+)',
                r'bg[_\s]*(\d+)',  # Bhagavad Gita
                r'sb[_\s]*(\d+)',  # Srimad Bhagavatam
            ]
            
            for pattern in chapter_patterns:
                match = re.search(pattern, filename)
                if match:
                    chapter_num = match.group(1)
                    potential_names = [
                        f"chapter{chapter_num}",
                        f"ch{chapter_num}",
                        f"bg{chapter_num}",
                        f"sb{chapter_num}"
                    ]
                    
                    for name in potential_names:
                        if name in self.chapter_glossaries:
                            return name
        
        # Try to guess from previous text content
        previous_text = context.get('previous_text', '').lower()
        if previous_text:
            # Look for chapter mentions
            for chapter_name in self.chapter_glossaries.keys():
                if chapter_name in previous_text:
                    return chapter_name
        
        return None
    
    def get_glossary(self, context: Dict[str, Any]) -> GlossaryResult:
        """Get chapter-specific glossary based on context."""
        start_time = time.perf_counter()
        
        self.application_count += 1
        
        # Guess the chapter
        guessed_chapter = self._guess_chapter(context)
        
        if guessed_chapter and guessed_chapter in self.chapter_glossaries:
            glossary_text = self.chapter_glossaries[guessed_chapter]
            confidence = 0.8  # High confidence when chapter is identified
            metadata = {
                'guessed_chapter': guessed_chapter,
                'available_chapters': list(self.chapter_glossaries.keys()),
                'guess_method': 'filename_or_context'
            }
        else:
            # Fallback to common terms
            glossary_text = " ".join(self.fallback_terms)
            confidence = 0.3  # Low confidence when guessing fails
            metadata = {
                'guessed_chapter': None,
                'available_chapters': list(self.chapter_glossaries.keys()),
                'guess_method': 'fallback'
            }
        
        processing_time = time.perf_counter() - start_time
        self.total_processing_time += processing_time
        
        return GlossaryResult(
            glossary_text=glossary_text,
            strategy_used=self.name,
            confidence=confidence,
            processing_time=processing_time,
            metadata=metadata
        )


class EmptyPolicy(BaseGlossaryPolicy):
    """
    Empty glossary policy (control).
    
    Returns no glossary terms, used as a control for testing the impact
    of glossary-enhanced transcription.
    """
    
    def __init__(self):
        super().__init__("empty")
    
    def get_glossary(self, context: Dict[str, Any]) -> GlossaryResult:
        """Return empty glossary."""
        start_time = time.perf_counter()
        
        self.application_count += 1
        processing_time = time.perf_counter() - start_time
        self.total_processing_time += processing_time
        
        return GlossaryResult(
            glossary_text="",
            strategy_used=self.name,
            confidence=1.0,  # Always confident in returning nothing
            processing_time=processing_time,
            metadata={'reason': 'control_policy'}
        )


class GlossaryPolicySelector:
    """
    Strategy selector for glossary policies.
    
    Manages multiple glossary policies and selects the appropriate one
    based on configuration or context.
    """
    
    def __init__(self, default_strategy: GlossaryStrategy = GlossaryStrategy.STATIC_COMMON):
        self.default_strategy = default_strategy
        self.policies: Dict[GlossaryStrategy, BaseGlossaryPolicy] = {}
        self.current_strategy = default_strategy
        
        # Initialize default policies
        self._initialize_policies()
        
        logger.info(f"GlossaryPolicySelector initialized with default: {default_strategy.value}")
    
    def _initialize_policies(self) -> None:
        """Initialize all available policies."""
        try:
            # Static common policy
            self.policies[GlossaryStrategy.STATIC_COMMON] = StaticCommonPolicy()
            
            # Chapter guess policy
            self.policies[GlossaryStrategy.CHAPTER_GUESS] = ChapterGuessPolicy()
            
            # Empty policy
            self.policies[GlossaryStrategy.EMPTY] = EmptyPolicy()
            
            logger.debug(f"Initialized {len(self.policies)} glossary policies")
            
        except Exception as e:
            logger.error(f"Failed to initialize policies: {e}")
    
    def set_strategy(self, strategy: Union[GlossaryStrategy, str]) -> bool:
        """
        Set the current glossary strategy.
        
        Args:
            strategy: Strategy to use
            
        Returns:
            bool: True if strategy was set successfully
        """
        try:
            if isinstance(strategy, str):
                strategy = GlossaryStrategy(strategy)
            
            if strategy in self.policies:
                self.current_strategy = strategy
                logger.info(f"Switched to glossary strategy: {strategy.value}")
                return True
            else:
                logger.error(f"Unknown glossary strategy: {strategy}")
                return False
                
        except ValueError as e:
            logger.error(f"Invalid glossary strategy: {strategy}")
            return False
    
    def get_glossary(self, context: Dict[str, Any]) -> GlossaryResult:
        """
        Get glossary using current strategy.
        
        Args:
            context: Context information for glossary selection
            
        Returns:
            GlossaryResult: Glossary result from current policy
        """
        try:
            policy = self.policies[self.current_strategy]
            return policy.get_glossary(context)
            
        except Exception as e:
            logger.error(f"Failed to get glossary with {self.current_strategy.value}: {e}")
            
            # Fallback to empty policy
            if self.current_strategy != GlossaryStrategy.EMPTY:
                return self.policies[GlossaryStrategy.EMPTY].get_glossary(context)
            else:
                # Last resort: return empty result
                return GlossaryResult(
                    glossary_text="",
                    strategy_used="error_fallback",
                    confidence=0.0,
                    processing_time=0.0,
                    metadata={'error': str(e)}
                )
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return [strategy.value for strategy in self.policies.keys()]
    
    def get_policy_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all policies."""
        return {
            strategy.value: policy.get_stats()
            for strategy, policy in self.policies.items()
        }
    
    def configure_policy(self, strategy: GlossaryStrategy, **kwargs) -> bool:
        """
        Configure a specific policy with custom parameters.
        
        Args:
            strategy: Strategy to configure
            **kwargs: Configuration parameters
            
        Returns:
            bool: True if configuration was successful
        """
        try:
            if strategy == GlossaryStrategy.STATIC_COMMON:
                glossary_path = kwargs.get('glossary_path')
                if glossary_path:
                    self.policies[strategy] = StaticCommonPolicy(glossary_path)
                    
            elif strategy == GlossaryStrategy.CHAPTER_GUESS:
                glossary_dir = kwargs.get('glossary_dir')
                if glossary_dir:
                    self.policies[strategy] = ChapterGuessPolicy(glossary_dir)
            
            logger.info(f"Configured {strategy.value} policy with {kwargs}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure {strategy.value} policy: {e}")
            return False


# Convenience functions
def create_glossary_selector(strategy: str = "static_common",
                           glossary_path: Optional[str] = None,
                           glossary_dir: Optional[str] = None) -> GlossaryPolicySelector:
    """
    Create and configure a glossary policy selector.
    
    Args:
        strategy: Default strategy name
        glossary_path: Path to static glossary file
        glossary_dir: Directory containing chapter glossaries
        
    Returns:
        GlossaryPolicySelector: Configured selector
    """
    try:
        strategy_enum = GlossaryStrategy(strategy)
    except ValueError:
        logger.warning(f"Unknown strategy '{strategy}', using static_common")
        strategy_enum = GlossaryStrategy.STATIC_COMMON
    
    selector = GlossaryPolicySelector(strategy_enum)
    
    # Configure policies if paths provided
    if glossary_path:
        selector.configure_policy(GlossaryStrategy.STATIC_COMMON, glossary_path=glossary_path)
    
    if glossary_dir:
        selector.configure_policy(GlossaryStrategy.CHAPTER_GUESS, glossary_dir=glossary_dir)
    
    return selector 