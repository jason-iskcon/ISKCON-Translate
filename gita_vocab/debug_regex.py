#!/usr/bin/env python3
"""Debug the specific regex pattern for verse extraction."""

import requests
from bs4 import BeautifulSoup
import re

def debug_regex_pattern():
    """Debug the specific regex pattern."""
    url = "https://vedabase.io/en/library/bg/1/"
    
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.text, 'lxml')
    all_text = soup.get_text()
    
    print("=== DEBUGGING REGEX PATTERN ===\n")
    
    # The current pattern from the scraper
    current_pattern = r'TEXT\s+(\d+):\s*([^T]*?)(?=TEXT\s+\d+:|$)'
    
    print(f"1. Testing current pattern: {current_pattern}")
    
    current_matches = re.findall(current_pattern, all_text, re.DOTALL | re.IGNORECASE)
    print(f"   Current pattern matches: {len(current_matches)}")
    
    if current_matches:
        print("   Sample matches:")
        for i, (num, content) in enumerate(current_matches[:3]):
            clean_content = content.strip()[:100].replace('\n', ' ')
            print(f"     Verse {num}: {clean_content}...")
    else:
        print("   ❌ No matches with current pattern")
    
    # Let's try simpler patterns
    print(f"\n2. Testing simpler patterns...")
    
    patterns = [
        (r'TEXT\s+(\d+):\s*([^T]*?)(?=TEXT\s+\d+|$)', "Pattern A: Stop at next TEXT"),
        (r'TEXT\s+(\d+):\s*([^T]*?)(?=TEXT|\Z)', "Pattern B: Stop at any TEXT or end"),
        (r'TEXT\s+(\d+):\s*([^T]+?)(?=TEXT\s+\d+|\Z)', "Pattern C: Non-greedy until next TEXT"),
        (r'TEXT\s+(\d+):\s*(.*?)(?=TEXT\s+\d+|$)', "Pattern D: Any chars until next TEXT"),
    ]
    
    for pattern, description in patterns:
        matches = re.findall(pattern, all_text, re.DOTALL | re.IGNORECASE)
        print(f"   {description}: {len(matches)} matches")
        
        if matches and len(matches) > 0:
            sample_num, sample_content = matches[0]
            clean_sample = sample_content.strip()[:80].replace('\n', ' ')
            print(f"     Sample: Verse {sample_num} -> {clean_sample}...")
    
    # Let's manually find a TEXT pattern and see what's around it
    print(f"\n3. Manual pattern inspection...")
    
    # Find first occurrence of "TEXT 1:"
    text_1_pos = all_text.find("TEXT 1:")
    if text_1_pos >= 0:
        print(f"   Found 'TEXT 1:' at position {text_1_pos}")
        
        # Show 200 characters around it
        start = max(0, text_1_pos - 50)
        end = min(len(all_text), text_1_pos + 200)
        context = all_text[start:end]
        print(f"   Context: ...{repr(context)}...")
        
        # Find next TEXT
        next_text_pos = all_text.find("TEXT", text_1_pos + 1)
        if next_text_pos >= 0:
            verse_1_content = all_text[text_1_pos:next_text_pos]
            print(f"   Verse 1 full content length: {len(verse_1_content)}")
            print(f"   Verse 1 preview: {repr(verse_1_content[:100])}...")
    
    # Test a very simple pattern
    print(f"\n4. Testing very simple pattern...")
    simple_pattern = r'TEXT\s+(\d+):\s*(.*?)(?=TEXT\s+\d+)'
    simple_matches = re.findall(simple_pattern, all_text, re.DOTALL)
    print(f"   Simple pattern matches: {len(simple_matches)}")
    
    if simple_matches:
        for i, (num, content) in enumerate(simple_matches[:2]):
            print(f"     Verse {num}: Length={len(content)}, Preview={repr(content[:60])}...")

if __name__ == "__main__":
    debug_regex_pattern() 