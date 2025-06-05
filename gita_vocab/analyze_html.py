#!/usr/bin/env python3
"""Analyze HTML structure to identify verse content containers."""

import requests
from bs4 import BeautifulSoup
import re

def analyze_vedabase_structure():
    """Analyze HTML structure of vedabase.io to find verse containers."""
    url = "https://vedabase.io/en/library/bg/1/"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        print("=== ANALYZING VEDABASE.IO HTML STRUCTURE ===\n")
        
        # Look for elements that might contain verse text
        print("1. Looking for elements with 'TEXT' pattern...")
        text_elements = soup.find_all(text=re.compile(r'TEXT\s+\d+:', re.IGNORECASE))
        print(f"Found {len(text_elements)} elements with TEXT pattern")
        
        for i, elem in enumerate(text_elements[:3]):  # Show first 3
            parent = elem.parent
            print(f"  Element {i+1}: Parent tag = {parent.name}, classes = {parent.get('class', [])}")
            print(f"    Content preview: {str(elem)[:100]}...")
        
        print("\n2. Looking for div/p elements with verse-like content...")
        # Look for divs or paragraphs that contain Sanskrit names
        sanskrit_pattern = r'(Kṛṣṇa|Arjuna|Dhṛtarāṣṭra|Kurukṣetra|Bhīṣma|Pāṇḍu)'
        
        verse_containers = []
        for tag in soup.find_all(['div', 'p', 'span']):
            text_content = tag.get_text()
            if re.search(sanskrit_pattern, text_content) and len(text_content) > 50:
                verse_containers.append(tag)
        
        print(f"Found {len(verse_containers)} potential verse containers")
        
        for i, container in enumerate(verse_containers[:5]):  # Show first 5
            print(f"  Container {i+1}: Tag = {container.name}, classes = {container.get('class', [])}")
            text_preview = container.get_text()[:150].replace('\n', ' ')
            print(f"    Content: {text_preview}...")
        
        print("\n3. Looking for main content area...")
        # Common patterns for main content
        main_selectors = [
            'main',
            '.content',
            '.main-content', 
            '#content',
            '.article',
            '.post',
            '.text-content',
            '.verse-content'
        ]
        
        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                print(f"  Found {len(elements)} elements with selector '{selector}'")
                
        print("\n4. Analyzing page structure...")
        # Look at the overall structure
        body = soup.find('body')
        if body:
            print("Body structure (first level children):")
            for child in body.find_all(recursive=False):
                if child.name:
                    print(f"  <{child.name}> with classes: {child.get('class', [])}")
        
        print("\n5. Looking for navigation/footer patterns to exclude...")
        # Identify elements to exclude
        exclude_patterns = [
            ('nav', 'navigation'),
            ('.footer', 'footer'),
            ('.header', 'header'),
            ('.sidebar', 'sidebar'),
            ('.donation', 'donations'),
            ('.thanks', 'acknowledgments')
        ]
        
        for pattern, description in exclude_patterns:
            elements = soup.select(pattern)
            if elements:
                print(f"  Found {len(elements)} {description} elements to potentially exclude")
        
        # Try to identify the actual verse content
        print("\n6. ATTEMPTING TO EXTRACT CLEAN VERSE CONTENT...")
        
        # Strategy 1: Find elements containing "TEXT [number]:" followed by verse
        verse_pattern = r'TEXT\s+(\d+):\s*([^T]*?)(?=TEXT\s+\d+:|$)'
        full_text = soup.get_text()
        verses = re.findall(verse_pattern, full_text, re.DOTALL)
        
        print(f"Strategy 1: Found {len(verses)} verses using regex")
        if verses:
            print("Sample verses:")
            for i, (num, content) in enumerate(verses[:3]):
                clean_content = content.strip()[:100].replace('\n', ' ')
                print(f"  Verse {num}: {clean_content}...")
        
        # Strategy 2: Look for specific container classes/ids
        print(f"\nStrategy 2: Looking for specific verse containers...")
        
        # Try to find the actual content div by looking for Sanskrit content
        potential_containers = soup.find_all(['div', 'section', 'article'])
        
        verse_rich_containers = []
        for container in potential_containers:
            text = container.get_text()
            sanskrit_count = len(re.findall(sanskrit_pattern, text))
            verse_count = len(re.findall(r'TEXT\s+\d+:', text))
            
            if sanskrit_count >= 3 and verse_count >= 5:  # Threshold for verse-rich content
                verse_rich_containers.append((container, sanskrit_count, verse_count))
        
        verse_rich_containers.sort(key=lambda x: x[1] + x[2], reverse=True)
        
        print(f"Found {len(verse_rich_containers)} verse-rich containers")
        for i, (container, s_count, v_count) in enumerate(verse_rich_containers[:3]):
            print(f"  Container {i+1}: {container.name} with {s_count} Sanskrit terms, {v_count} verses")
            print(f"    Classes: {container.get('class', [])}")
            print(f"    ID: {container.get('id', 'None')}")
            
            # Sample content
            content_preview = container.get_text()[:200].replace('\n', ' ')
            print(f"    Preview: {content_preview}...")
            
        return verse_rich_containers[0][0] if verse_rich_containers else None
        
    except Exception as e:
        print(f"Error analyzing structure: {e}")
        return None

if __name__ == "__main__":
    best_container = analyze_vedabase_structure()
    if best_container:
        print(f"\n=== RECOMMENDED CONTAINER ===")
        print(f"Tag: {best_container.name}")
        print(f"Classes: {best_container.get('class', [])}")
        print(f"ID: {best_container.get('id', 'None')}") 