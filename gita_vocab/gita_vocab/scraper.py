"""
Resilient web scraper for Bhagavad Gita and ISKCON content.

This module provides a robust scraper with exponential backoff, rate limiting,
and error handling for scraping spiritual content from various sources.
"""

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ScrapedContent(BaseModel):
    """Model for scraped content."""
    
    url: str
    title: str
    text: str
    chapter: Optional[int] = None
    verse: Optional[str] = None
    source: str
    timestamp: float = Field(default_factory=time.time)
    word_count: int = 0
    
    def __post_init__(self) -> None:
        """Calculate word count after initialization."""
        self.word_count = len(self.text.split())


class GitaScraper:
    """
    Resilient scraper for Bhagavad Gita and ISKCON content.
    
    Features:
    - Exponential backoff for rate limiting (429, 503)
    - Configurable delays and retry logic
    - Multiple source support
    - Content deduplication
    - Progress tracking
    """
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
        user_agent: Optional[str] = None,
    ):
        """
        Initialize the scraper.
        
        Args:
            base_delay: Base delay between requests (seconds)
            max_delay: Maximum delay for exponential backoff
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Multiplier for exponential backoff
            user_agent: Custom user agent string
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Setup session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent or (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Content tracking
        self.scraped_urls: Set[str] = set()
        self.scraped_content: List[ScrapedContent] = []
        self.failed_urls: List[str] = []
        
        # Known sources for Bhagavad Gita content
        self.sources = {
            'vedabase': {
                'base_url': 'https://vedabase.io/en/library/bg/',
                'chapters': list(range(1, 19)),  # 18 chapters
                'parser': self._parse_vedabase
            }
        }
        
        logger.info(f"GitaScraper initialized with {len(self.sources)} sources")
    
    def scrape_all_sources(self, output_file: Union[str, Path] = "raw_synonyms.jsonl") -> None:
        """
        Scrape content from all configured sources.
        
        Args:
            output_file: Path to output JSONL file
        """
        logger.info("Starting comprehensive scraping of all sources")
        
        total_chapters = sum(len(source['chapters']) for source in self.sources.values())
        
        with tqdm(total=total_chapters, desc="Scraping chapters") as pbar:
            for source_name, source_config in self.sources.items():
                logger.info(f"Scraping from {source_name}")
                
                for chapter in source_config['chapters']:
                    try:
                        self._scrape_chapter(source_name, chapter)
                        pbar.update(1)
                        
                        # Random delay between chapters
                        delay = self.base_delay + random.uniform(0, 1)
                        time.sleep(delay)
                        
                    except Exception as e:
                        logger.error(f"Failed to scrape {source_name} chapter {chapter}: {e}")
                        pbar.update(1)
        
        # Save results
        self.save_content(output_file)
        
        logger.info(f"Scraping completed: {len(self.scraped_content)} items, "
                   f"{len(self.failed_urls)} failures")
    
    def _scrape_chapter(self, source_name: str, chapter: int) -> None:
        """Scrape a specific chapter from a source."""
        source_config = self.sources[source_name]
        
        # Build chapter URL
        if source_name == 'vedabase':
            url = f"{source_config['base_url']}{chapter}/"
        elif source_name == 'bhagavadgitaasitis':
            url = f"{source_config['base_url']}{chapter}/"
        elif source_name == 'holy_bhagavad_gita':
            url = f"{source_config['base_url']}{chapter}"
        else:
            logger.warning(f"Unknown source: {source_name}")
            return
        
        # Skip if already scraped
        if url in self.scraped_urls:
            logger.debug(f"Skipping already scraped URL: {url}")
            return
        
        # Fetch and parse content
        html_content = self._fetch_with_retry(url)
        if html_content:
            try:
                parser = source_config['parser']
                content_items = parser(html_content, url, chapter)
                
                for item in content_items:
                    self.scraped_content.append(item)
                
                self.scraped_urls.add(url)
                logger.debug(f"Successfully scraped {len(content_items)} items from {url}")
                
            except Exception as e:
                logger.error(f"Failed to parse content from {url}: {e}")
                self.failed_urls.append(url)
        else:
            self.failed_urls.append(url)
    
    def _fetch_with_retry(self, url: str) -> Optional[str]:
        """
        Fetch URL with exponential backoff retry logic.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        delay = self.base_delay
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching {url} (attempt {attempt + 1})")
                
                response = self.session.get(url, timeout=30)
                
                # Handle rate limiting
                if response.status_code in [429, 503]:
                    logger.warning(f"Rate limited ({response.status_code}) for {url}, "
                                 f"waiting {delay:.1f}s")
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                    continue
                
                # Handle other errors
                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    if attempt < self.max_retries - 1:
                        time.sleep(delay)
                        delay = min(delay * self.backoff_factor, self.max_delay)
                        continue
                    else:
                        return None
                
                # Success
                return response.text
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                else:
                    logger.error(f"Max retries exceeded for {url}")
                    return None
        
        return None
    
    def _parse_vedabase(self, html: str, url: str, chapter: int) -> List[ScrapedContent]:
        """Parse content from vedabase.io to extract Sanskrit terms ONLY from verse text."""
        soup = BeautifulSoup(html, 'lxml')
        content_items = []
        
        # Get all text content for verse extraction
        all_text = soup.get_text()
        
        # STRATEGY: Extract ONLY from verse content, not entire page
        # Step 1: Extract individual verses using TEXT pattern
        verse_pattern = r'TEXT\s+(\d+):\s*(.*?)(?=TEXT\s+\d+|$)'
        verse_matches = re.findall(verse_pattern, all_text, re.DOTALL | re.IGNORECASE)
        
        if not verse_matches:
            logger.warning(f"No verses found in {url}")
            return content_items
        
        logger.info(f"Found {len(verse_matches)} verses in chapter {chapter}")
        
        # Step 2: Extract Sanskrit terms ONLY from verse content
        verse_text_only = ""
        for verse_num, verse_content in verse_matches:
            # Clean the verse content and add to our verse-only text
            clean_verse = verse_content.strip()
            verse_text_only += f" {clean_verse}"
        
        # Step 3: Extract Sanskrit terms from verse-only content
        sanskrit_terms = set()
        
        # Pattern 1: Words with diacritics (transliterated Sanskrit)
        diacritic_pattern = r'\b[A-Za-z]*[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣḻĀĪŪṚṜḶḸĒŌṂḤṄÑṬḌṆŚṢḺ][A-Za-z]*\b'
        diacritic_matches = re.findall(diacritic_pattern, verse_text_only)
        sanskrit_terms.update(diacritic_matches)
        
        # Pattern 2: Known Sanskrit vocabulary (in verse context only)
        common_sanskrit = [
            'Krishna', 'Krsna', 'Arjuna', 'Bhagavan', 'Bhagavad', 'dharma', 'karma', 'yoga',
            'Kurukshetra', 'Pandava', 'Kauravas', 'Sanjaya', 'Dhritarashtra', 'Duryodhana',
            'Bhima', 'Yudhishthira', 'Nakula', 'Sahadeva', 'Draupadi', 'Drona', 'Bhishma',
            'moksha', 'samsara', 'atman', 'Brahman', 'Paramatma', 'jiva', 'maya', 'lila',
            'bhakti', 'jnana', 'vairagya', 'tapas', 'yajna', 'mantra', 'ashrama', 'varna',
            'guru', 'shishya', 'sadhu', 'sant', 'rishi', 'muni', 'acharya',
            'pranayama', 'asana', 'dhyana', 'samadhi', 'pratyahara', 'dharana',
            'Vishnu', 'Shiva', 'Brahma', 'Devi', 'Lakshmi', 'Saraswati', 'Ganga',
            'puja', 'aarti', 'prasadam', 'kirtan', 'bhajan', 'satsang', 'darshan',
            'vrindavan', 'mathura', 'dvaraka', 'mayapur', 'kashi', 'vraja',
            'Govinda', 'Madhusudana', 'Janardana', 'Vasudeva', 'Keshava', 'Hrishikesha',
            'Achyuta', 'Ananta', 'Garuda', 'Hanuman', 'Ganesha', 'Indra', 'Varuna',
            'Agni', 'Vayu', 'Surya', 'Chandra', 'Prithvi', 'Akasha', 'Kala', 'Maha',
            'Prabhu', 'Dasa', 'Mata', 'Pitamaha'
        ]
        
        for term in common_sanskrit:
            # Case-insensitive search in verse text only
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.findall(pattern, verse_text_only, re.IGNORECASE)
            sanskrit_terms.update(matches)
        
        # Step 4: Aggressive filtering to remove extraneous terms
        # Filter out English words that commonly appear even in verse context
        english_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'a', 'an', 
            'as', 'if', 'when', 'where', 'why', 'how', 'what', 'who', 'which', 'whose', 'all', 'any', 'each', 
            'every', 'some', 'many', 'much', 'more', 'most', 'other', 'another', 'such', 'only', 'own', 'same', 
            'so', 'than', 'too', 'very', 'just', 'now', 'here', 'there', 'then', 'them', 'they', 'their', 
            'theirs', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours', 
            'you', 'your', 'yours', 'me', 'my', 'mine', 'one', 'two', 'three', 'first', 'second', 'third',
            'said', 'says', 'tell', 'told', 'speak', 'spoke', 'word', 'words', 'great', 'like', 'also',
            'thus', 'also', 'such', 'who', 'whom', 'those', 'see', 'seen', 'look', 'looked', 'take', 'taken',
            'give', 'given', 'come', 'came', 'go', 'went', 'know', 'known', 'think', 'thought', 'find', 'found',
            'make', 'made', 'become', 'became', 'call', 'called', 'get', 'got', 'use', 'used', 'work', 'worked',
            'son', 'sons', 'father', 'fathers', 'king', 'teacher', 'army', 'battle', 'fight', 'fighting',
            'warrior', 'warriors', 'bow', 'arrow', 'arrows', 'chariot', 'horse', 'horses', 'sound', 'voice',
            'earth', 'sky', 'water', 'fire', 'air', 'mind', 'body', 'heart', 'soul', 'life', 'death',
            'good', 'bad', 'right', 'wrong', 'true', 'false', 'well', 'better', 'best', 'always', 'never',
            'english', 'language', 'chapter', 'text', 'verse', 'book', 'page', 'library', 'site', 'website',
            'swami', 'prabhupada', 'prabhupāda', 'bhaktivedanta', 'ashram', 'gita', 'temple', 'mandir',
            'international', 'society', 'consciousness', 'foundation', 'trust', 'book'
        }
        
        # Common donor names to exclude
        donor_names = {
            'john', 'james', 'robert', 'michael', 'william', 'david', 'richard', 'joseph', 'thomas', 'charles',
            'christopher', 'daniel', 'matthew', 'anthony', 'mark', 'donald', 'steven', 'paul', 'andrew', 'joshua',
            'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara', 'susan', 'jessica', 'sarah', 'karen',
            'douglas', 'mario', 'peter', 'alex', 'jordan', 'taylor', 'casey', 'jamie', 'morgan',
            'arun', 'raj', 'ravi', 'sunil', 'amit', 'rohit', 'vikash', 'anita', 'nisha', 'priya', 'kavya',
            'divya', 'sneha', 'pooja', 'riya', 'tanya', 'shreya', 'hari', 'krishna', 'rama', 'sita'
        }
        
        # Step 5: Filter to keep only Sanskrit terms from verses
        filtered_sanskrit = set()
        for term in sanskrit_terms:
            term_lower = term.lower()
            
            # Skip common English words
            if term_lower in english_words:
                continue
                
            # Skip common donor names
            if term_lower in donor_names:
                continue
                
            # Skip very short terms
            if len(term) < 3:
                continue
                
            # Skip terms that are all uppercase (likely abbreviations)
            if term.isupper() and len(term) > 1:
                continue
                
            # Keep terms with diacritics (definitely Sanskrit)
            if re.search(r'[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣḻĀĪŪṚṜḶḸĒŌṂḤṄÑṬḌṆŚṢḺ]', term):
                filtered_sanskrit.add(term)
                continue
                
            # Keep terms from our known Sanskrit list (if they appear in verses)
            if any(term.lower() == known.lower() for known in common_sanskrit):
                filtered_sanskrit.add(term)
                continue
                
            # Keep terms that end in common Sanskrit suffixes and are capitalized
            sanskrit_suffixes = ['a', 'am', 'an', 'as', 'i', 'u', 'ya', 'va', 'na', 'ra', 'la', 'ta', 'da', 'ma', 'sa']
            if (term[0].isupper() and 
                any(term.lower().endswith(suffix) for suffix in sanskrit_suffixes) and 
                len(term) > 4):
                filtered_sanskrit.add(term)
        
        # Step 6: Create content item with filtered Sanskrit terms from verses only
        if filtered_sanskrit:
            sanskrit_text = ' '.join(sorted(filtered_sanskrit))
            content_items.append(ScrapedContent(
                url=url,
                title=f"Bhagavad Gita Chapter {chapter} Sanskrit Terms (Verse Content Only)",
                text=sanskrit_text,
                chapter=chapter,
                verse="verses_only",
                source='vedabase_verses',
                word_count=len(filtered_sanskrit)
            ))
            
            logger.info(f"Extracted {len(filtered_sanskrit)} Sanskrit terms from {len(verse_matches)} verses in chapter {chapter}")
        else:
            logger.warning(f"No Sanskrit terms found in verses for chapter {chapter}")
        
        return content_items
    
    def _parse_bhagavadgitaasitis(self, html: str, url: str, chapter: int) -> List[ScrapedContent]:
        """Parse content from bhagavadgitaasitis.com."""
        soup = BeautifulSoup(html, 'lxml')
        content_items = []
        
        # Try to find verse content
        verses = soup.find_all(['div', 'p'], class_=lambda x: x and ('verse' in x.lower() or 'text' in x.lower()))
        
        if not verses:
            # Fallback: look for any content with spiritual keywords
            all_text = soup.get_text()
            if any(word in all_text.lower() for word in ['krishna', 'arjuna', 'dharma', 'karma']):
                # Extract meaningful sentences
                sentences = re.split(r'[.!?]+', all_text)
                meaningful_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if (len(sentence) > 30 and 
                        any(word in sentence.lower() for word in ['krishna', 'arjuna', 'dharma', 'karma', 'yoga', 'soul', 'supreme'])):
                        meaningful_sentences.append(sentence)
                
                if meaningful_sentences:
                    combined_text = '. '.join(meaningful_sentences[:5])  # Take first 5 meaningful sentences
                    content_items.append(ScrapedContent(
                        url=url,
                        title=f"Bhagavad Gita Chapter {chapter}",
                        text=combined_text,
                        chapter=chapter,
                        verse="general",
                        source='bhagavadgitaasitis',
                        word_count=len(combined_text.split())
                    ))
        
        return content_items
    
    def _parse_holy_bhagavad_gita(self, html: str, url: str, chapter: int) -> List[ScrapedContent]:
        """Parse content from holy-bhagavad-gita.org."""
        soup = BeautifulSoup(html, 'lxml')
        content_items = []
        
        # Try to extract any meaningful content
        all_text = soup.get_text()
        
        # Look for spiritual content
        if any(word in all_text.lower() for word in ['krishna', 'arjuna', 'dharma', 'karma']):
            # Split into paragraphs and find meaningful ones
            paragraphs = all_text.split('\n')
            meaningful_paragraphs = []
            
            for para in paragraphs:
                para = para.strip()
                if (len(para) > 50 and 
                    any(word in para.lower() for word in ['krishna', 'arjuna', 'dharma', 'karma', 'yoga', 'soul', 'supreme', 'lord'])):
                    meaningful_paragraphs.append(para)
            
            if meaningful_paragraphs:
                # Combine first few meaningful paragraphs
                combined_text = ' '.join(meaningful_paragraphs[:3])
                content_items.append(ScrapedContent(
                    url=url,
                    title=f"Bhagavad Gita Chapter {chapter}",
                    text=combined_text,
                    chapter=chapter,
                    verse="general",
                    source='holy_bhagavad_gita',
                    word_count=len(combined_text.split())
                ))
        
        return content_items
    
    def save_content(self, output_file: Union[str, Path]) -> None:
        """
        Save scraped content to JSONL file.
        
        Args:
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.scraped_content:
                f.write(item.model_dump_json() + '\n')
        
        logger.info(f"Saved {len(self.scraped_content)} items to {output_path}")
    
    def load_content(self, input_file: Union[str, Path]) -> None:
        """
        Load previously scraped content from JSONL file.
        
        Args:
            input_file: Path to input file
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            return
        
        self.scraped_content.clear()
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item_data = json.loads(line)
                        item = ScrapedContent(**item_data)
                        self.scraped_content.append(item)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse line: {e}")
        
        logger.info(f"Loaded {len(self.scraped_content)} items from {input_path}")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get scraping statistics."""
        if not self.scraped_content:
            return {}
        
        total_words = sum(item.word_count for item in self.scraped_content)
        sources = set(item.source for item in self.scraped_content)
        chapters = set(item.chapter for item in self.scraped_content if item.chapter)
        
        return {
            'total_items': len(self.scraped_content),
            'total_words': total_words,
            'average_words_per_item': total_words / len(self.scraped_content),
            'unique_sources': len(sources),
            'unique_chapters': len(chapters),
            'failed_urls': len(self.failed_urls),
            'sources': list(sources),
            'chapters': sorted(list(chapters)) if chapters else []
        } 