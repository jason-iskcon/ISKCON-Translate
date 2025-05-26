"""Translation module for ISKCON-Translate."""
import time
import logging
from typing import Dict, Optional, Tuple
import json
import os
import yt_dlp

# Import with try-except to handle both direct execution and module import
try:
    from ..logging_utils import get_logger, TRACE
except ImportError:
    from src.logging_utils import get_logger, TRACE

logger = get_logger(__name__)

class Translator:
    """Handles comparison of YouTube and Parakletos captions."""
    
    def __init__(self, cache_dir: str = ".translation_cache"):
        """Initialize the translator.
        
        Args:
            cache_dir: Directory to store translation cache
        """
        self.cache_dir = cache_dir
        self.cache: Dict[str, Dict[str, str]] = {}
        self._load_cache()
        
        # Initialize yt-dlp
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
        }
        
        logger.info("Translator initialized for YouTube vs Parakletos comparison")
    
    def _load_cache(self):
        """Load translation cache from disk."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, "translations.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached translations")
        except Exception as e:
            logger.error(f"Error loading translation cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save translation cache to disk."""
        try:
            cache_file = os.path.join(self.cache_dir, "translations.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self.cache)} translations to cache")
        except Exception as e:
            logger.error(f"Error saving translation cache: {e}")
    
    def get_youtube_captions(self, video_url: str) -> Dict[str, str]:
        """Get YouTube captions for a video.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Dictionary mapping timestamps to caption text
        """
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                # Get automatic captions
                if 'automatic_captions' in info and 'en' in info['automatic_captions']:
                    captions = info['automatic_captions']['en']
                    if captions:
                        # Convert to our format
                        return {
                            str(entry['start']): entry['text']
                            for entry in captions
                        }
                
                logger.warning(f"No automatic captions found for {video_url}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting YouTube captions: {e}")
            return {}
    
    def translate(self, text: str, timestamp: float, video_url: Optional[str] = None) -> Tuple[str, str]:
        """Get both YouTube and Parakletos captions for comparison.
        
        Args:
            text: Original text to translate
            timestamp: Current video timestamp
            video_url: Optional YouTube video URL for comparison
            
        Returns:
            Tuple of (youtube_text, parakletos_text)
        """
        if not text.strip():
            return text, text
            
        # Check cache first
        cache_key = f"{text}_{timestamp}"
        if cache_key in self.cache:
            logger.debug(f"Using cached translations for: {text[:50]}...")
            return self.cache[cache_key]['youtube'], self.cache[cache_key]['parakletos']
        
        try:
            # Get YouTube caption if URL provided
            youtube_text = text
            if video_url:
                youtube_captions = self.get_youtube_captions(video_url)
                youtube_text = youtube_captions.get(str(timestamp), text)
            
            # Get Parakletos translation
            parakletos_text = self._translate_parakletos(text)
            
            # Cache the results
            self.cache[cache_key] = {
                'youtube': youtube_text,
                'parakletos': parakletos_text,
                'timestamp': time.time()
            }
            self._save_cache()
            
            return youtube_text, parakletos_text
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text, text
    
    def _translate_parakletos(self, text: str) -> str:
        """Translate text using Parakletos.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        # TODO: Implement Parakletos translation
        # For now, return a placeholder
        return f"[Parakletos] {text}" 