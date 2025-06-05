#!/usr/bin/env python3
"""
Test the Sanskrit-focused scraper.
"""

from gita_vocab.scraper import GitaScraper

def test_sanskrit_scraper():
    """Test the Sanskrit scraper with one chapter."""
    print("Testing Sanskrit scraper...")
    
    scraper = GitaScraper()
    # Only test vedabase for now
    scraper.sources = {
        'vedabase': {
            'base_url': 'https://vedabase.io/en/library/bg/',
            'chapters': [1],  # Just test chapter 1
            'parser': scraper._parse_vedabase
        }
    }
    
    scraper.scrape_all_sources('sanskrit_test.jsonl')
    
    stats = scraper.get_stats()
    print(f"Scraped {stats.get('total_items', 0)} items")
    print(f"Total words: {stats.get('total_words', 0)}")
    
    # Show first few items
    if scraper.scraped_content:
        print("\nFirst item:")
        first_item = scraper.scraped_content[0]
        print(f"Title: {first_item.title}")
        print(f"Text: {first_item.text[:200]}...")
        print(f"Word count: {first_item.word_count}")

if __name__ == "__main__":
    test_sanskrit_scraper() 