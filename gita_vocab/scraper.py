"""
Scraper for Bhagavad Gita Sanskrit terms.
"""

import re
import time
import logging
from typing import Dict, List, Set, Tuple
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitaScraper:
    """Scraper for Bhagavad Gita Sanskrit terms."""
    
    BASE_URL = "https://www.holy-bhagavad-gita.org"
    CHAPTER_URL = "https://www.holy-bhagavad-gita.org/chapter/{}/"
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_chapter_url(self, chapter: int) -> str:
        """Get the URL for a specific chapter."""
        return self.CHAPTER_URL.format(chapter)
    
    def fetch_page(self, url: str) -> str:
        """Fetch a page with retries."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(self.RETRY_DELAY)
    
    def extract_sanskrit_terms(self, html: str) -> List[str]:
        """Extract Sanskrit terms from HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        terms = []
        
        # Find all verse containers
        verse_containers = soup.find_all('div', class_='verse-container')
        
        for container in verse_containers:
            # Find the Sanskrit text element
            sanskrit_elem = container.find('div', class_='sanskrit')
            if not sanskrit_elem:
                continue
                
            # Get the Sanskrit text
            sanskrit_text = sanskrit_elem.get_text(strip=True)
            if not sanskrit_text:
                continue
                
            # Split into words and clean
            words = sanskrit_text.split()
            for word in words:
                # Remove punctuation and clean
                word = re.sub(r'[^\u0900-\u097F]', '', word)
                if word and len(word) > 1:  # Only keep words longer than 1 character
                    terms.append(word)
        
        return terms
    
    def scrape_chapter(self, chapter: int) -> List[str]:
        """Scrape Sanskrit terms from a specific chapter."""
        url = self.get_chapter_url(chapter)
        logger.info(f"Scraping chapter {chapter} from {url}")
        
        try:
            html = self.fetch_page(url)
            terms = self.extract_sanskrit_terms(html)
            logger.info(f"Found {len(terms)} terms in chapter {chapter}")
            return terms
        except Exception as e:
            logger.error(f"Error scraping chapter {chapter}: {e}")
            return []
    
    def scrape_all_chapters(self) -> Dict[int, List[str]]:
        """Scrape all chapters and return terms by chapter."""
        all_terms = {}
        
        # Scrape all 18 chapters
        for chapter in range(1, 19):
            terms = self.scrape_chapter(chapter)
            if terms:
                all_terms[chapter] = terms
            time.sleep(1)  # Be nice to the server
        
        return all_terms
    
    def save_terms(self, terms_by_chapter: Dict[int, List[str]], output_file: str):
        """Save terms to a file, maintaining chapter order."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write terms in chapter order
            for chapter in sorted(terms_by_chapter.keys()):
                f.write(f"\n# Chapter {chapter}\n")
                for term in terms_by_chapter[chapter]:
                    f.write(f"{term}\n")
    
    def run(self, output_file: str = "sanskrit_terms.txt"):
        """Run the scraper and save results."""
        logger.info("Starting scraper...")
        terms_by_chapter = self.scrape_all_chapters()
        
        if not terms_by_chapter:
            logger.error("No terms found!")
            return
        
        total_terms = sum(len(terms) for terms in terms_by_chapter.values())
        logger.info(f"Found {total_terms} total terms across {len(terms_by_chapter)} chapters")
        
        self.save_terms(terms_by_chapter, output_file)
        logger.info(f"Saved terms to {output_file}")

def main():
    """Main entry point."""
    scraper = GitaScraper()
    scraper.run()

if __name__ == "__main__":
    main() 