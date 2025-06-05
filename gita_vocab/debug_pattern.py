#!/usr/bin/env python3
"""Debug the verse pattern matching."""

import requests
from bs4 import BeautifulSoup
import re

def analyze_verse_patterns():
    """Analyze what the actual verse patterns look like."""
    url = "https://vedabase.io/en/library/bg/1/"
    
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.text, 'lxml')
    all_text = soup.get_text()
    
    print("=== ANALYZING VERSE PATTERNS ===\n")
    
    # Look for any mention of "TEXT" in the content
    print("1. Searching for any 'TEXT' occurrences...")
    text_matches = re.findall(r'TEXT[^.]*', all_text, re.IGNORECASE)
    print(f"   Found {len(text_matches)} TEXT occurrences")
    for i, match in enumerate(text_matches[:5]):
        print(f"     {i+1}: {match[:80]}...")
    
    # Look for numbered patterns
    print(f"\n2. Searching for numbered patterns...")
    
    # Pattern A: "TEXT 1:", "TEXT 2:", etc.
    pattern_a = r'TEXT\s+(\d+):'
    matches_a = re.findall(pattern_a, all_text, re.IGNORECASE)
    print(f"   Pattern 'TEXT [number]:': {len(matches_a)} matches")
    print(f"   Numbers found: {matches_a[:10]}")
    
    # Pattern B: Just numbers followed by colon
    pattern_b = r'\b(\d+):'
    matches_b = re.findall(pattern_b, all_text)
    print(f"   Pattern '[number]:': {len(matches_b)} matches")
    print(f"   Numbers found: {matches_b[:10]}")
    
    # Pattern C: Look for Sanskrit names that we know should be there
    sanskrit_names = ['Dhṛtarāṣṭra', 'Sañjaya', 'Kurukṣetra', 'Pāṇḍu', 'Arjuna', 'Kṛṣṇa']
    print(f"\n3. Searching for known Sanskrit names...")
    
    for name in sanskrit_names:
        name_matches = re.findall(re.escape(name), all_text)
        print(f"   '{name}': {len(name_matches)} occurrences")
    
    # Pattern D: Look for verses with context
    print(f"\n4. Searching for verses with context...")
    
    # Look for lines that contain Sanskrit names
    lines = all_text.split('\n')
    verse_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if any(name in line for name in sanskrit_names) and len(line) > 20:
            verse_lines.append((i, line))
    
    print(f"   Found {len(verse_lines)} lines with Sanskrit names")
    for i, (line_num, line) in enumerate(verse_lines[:5]):
        print(f"     Line {line_num}: {line[:80]}...")
    
    # Pattern E: Try different verse patterns
    print(f"\n5. Trying alternative verse patterns...")
    
    patterns = [
        (r'(\d+)\.\s*([^0-9]*?)(?=\d+\.|$)', "Number. Content"),
        (r'TEXT\s*(\d+)\s*:?\s*([^T]*?)(?=TEXT\s*\d+|$)', "TEXT Number: Content"),
        (r'(\d+):([^0-9]*?)(?=\d+:|$)', "Number: Content"),
        (r'verse\s+(\d+)[:\.]?\s*([^v]*?)(?=verse\s+\d+|$)', "verse Number Content"),
    ]
    
    for pattern, description in patterns:
        matches = re.findall(pattern, all_text, re.DOTALL | re.IGNORECASE)
        print(f"   {description}: {len(matches)} matches")
        if matches:
            sample = matches[0]
            if len(sample) == 2:
                num, content = sample
                print(f"     Sample: {num} -> {content.strip()[:60]}...")
    
    # Pattern F: Show actual text structure around known Sanskrit terms
    print(f"\n6. Context around Sanskrit terms...")
    
    for name in sanskrit_names[:3]:
        pos = all_text.find(name)
        if pos >= 0:
            start = max(0, pos - 50)
            end = min(len(all_text), pos + 100)
            context = all_text[start:end].replace('\n', ' ')
            print(f"   Around '{name}': ...{context}...")
            break

if __name__ == "__main__":
    analyze_verse_patterns() 