#!/usr/bin/env python3
"""Test the improved scraper to check verse-only extraction."""

from gita_vocab.scraper import GitaScraper
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_verse_only_scraper():
    """Test the improved scraper that extracts only from verse content."""
    print("=== TESTING VERSE-ONLY SCRAPER ===\n")
    
    # Create scraper instance
    scraper = GitaScraper()
    
    # Test with just chapter 1
    print("Scraping Chapter 1 from vedabase.io...")
    scraper._scrape_chapter('vedabase', 1)
    
    print(f"\nScraping completed:")
    print(f"  Total items scraped: {len(scraper.scraped_content)}")
    print(f"  Failed URLs: {len(scraper.failed_urls)}")
    
    if scraper.scraped_content:
        item = scraper.scraped_content[0]
        print(f"\n=== EXTRACTED CONTENT ===")
        print(f"Title: {item.title}")
        print(f"Source: {item.source}")
        print(f"Chapter: {item.chapter}")
        print(f"Verse: {item.verse}")
        print(f"Word count: {item.word_count}")
        print(f"\nExtracted Sanskrit terms:")
        
        # Split the text to show individual terms
        terms = item.text.split()
        print(f"Total terms: {len(terms)}")
        
        # Show terms in columns
        for i in range(0, len(terms), 5):
            batch = terms[i:i+5]
            print("  " + " | ".join(f"{term:<15}" for term in batch))
        
        # Check for problematic terms
        print(f"\n=== QUALITY CHECK ===")
        problematic_terms = []
        generic_terms = ['Prabhupada', 'Prabhupāda', 'Swami', 'Ashram', 'Gita', 'International', 'Society', 'Foundation']
        
        for term in terms:
            if any(generic.lower() in term.lower() for generic in generic_terms):
                problematic_terms.append(term)
        
        if problematic_terms:
            print(f"⚠️  Found {len(problematic_terms)} potentially generic terms:")
            for term in problematic_terms:
                print(f"    - {term}")
        else:
            print("✅ No generic/extraneous terms detected!")
        
        # Check for proper Sanskrit terms
        sanskrit_indicators = ['ṛ', 'ṇ', 'ā', 'ī', 'ū', 'ṃ', 'ḥ']
        sanskrit_terms = [term for term in terms if any(char in term for char in sanskrit_indicators)]
        
        print(f"\n✅ Terms with Sanskrit diacritics ({len(sanskrit_terms)}):")
        for term in sanskrit_terms[:10]:  # Show first 10
            print(f"    - {term}")
        if len(sanskrit_terms) > 10:
            print(f"    ... and {len(sanskrit_terms) - 10} more")
            
    else:
        print("❌ No content was scraped!")
        if scraper.failed_urls:
            print("Failed URLs:")
            for url in scraper.failed_urls:
                print(f"  - {url}")

if __name__ == "__main__":
    test_verse_only_scraper() 