#!/usr/bin/env python3
"""Debug the scraper to see what's happening."""

import requests
from bs4 import BeautifulSoup
from gita_vocab.scraper import GitaScraper
import logging
import re

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def debug_scraper():
    """Debug the scraper step by step."""
    print("=== DEBUGGING SCRAPER ===\n")
    
    # Test 1: Direct URL fetch
    url = "https://vedabase.io/en/library/bg/1/"
    print(f"1. Testing direct fetch of {url}")
    
    try:
        response = requests.get(url, timeout=30)
        print(f"   Status: {response.status_code}")
        print(f"   Content length: {len(response.text)}")
        
        if response.status_code == 200:
            # Test 2: Parse for verses
            soup = BeautifulSoup(response.text, 'lxml')
            all_text = soup.get_text()
            
            print(f"\n2. Testing verse pattern extraction")
            verse_pattern = r'TEXT\s+(\d+):\s*([^T]*?)(?=TEXT\s+\d+:|$)'
            verse_matches = re.findall(verse_pattern, all_text, re.DOTALL | re.IGNORECASE)
            
            print(f"   Found {len(verse_matches)} verses")
            
            if verse_matches:
                print("   Sample verses:")
                for i, (num, content) in enumerate(verse_matches[:3]):
                    clean_content = content.strip()[:100].replace('\n', ' ')
                    print(f"     Verse {num}: {clean_content}...")
            
            # Test 3: Extract Sanskrit terms from verse content
            print(f"\n3. Testing Sanskrit extraction from verses")
            
            verse_text_only = ""
            for verse_num, verse_content in verse_matches:
                clean_verse = verse_content.strip()
                verse_text_only += f" {clean_verse}"
            
            print(f"   Combined verse text length: {len(verse_text_only)}")
            
            # Pattern 1: Diacritical terms
            diacritic_pattern = r'\b[A-Za-z]*[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣḻĀĪŪṚṜḶḸĒŌṂḤṄÑṬḌṆŚṢḺ][A-Za-z]*\b'
            diacritic_matches = re.findall(diacritic_pattern, verse_text_only)
            print(f"   Diacritical terms found: {len(diacritic_matches)}")
            print(f"   Sample: {diacritic_matches[:10]}")
            
        else:
            print(f"   ❌ Failed to fetch URL: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error fetching URL: {e}")
    
    # Test 4: GitaScraper instance
    print(f"\n4. Testing GitaScraper instance")
    
    try:
        scraper = GitaScraper()
        print(f"   Scraper created successfully")
        print(f"   Available sources: {list(scraper.sources.keys())}")
        
        # Test the specific method
        print(f"\n5. Testing _parse_vedabase method")
        if response.status_code == 200:
            content_items = scraper._parse_vedabase(response.text, url, 1)
            print(f"   Content items returned: {len(content_items)}")
            
            if content_items:
                item = content_items[0]
                print(f"   Title: {item.title}")
                print(f"   Source: {item.source}")
                print(f"   Word count: {item.word_count}")
                print(f"   Text preview: {item.text[:200]}...")
            else:
                print(f"   ❌ No content items returned")
        
    except Exception as e:
        print(f"   ❌ Error with GitaScraper: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_scraper() 