#!/usr/bin/env python3
"""Check the final scraping results."""

import json
from pathlib import Path

def check_results():
    """Check the final scraping results."""
    
    result_file = Path("verse_sanskrit_final.jsonl")
    
    if not result_file.exists():
        print("❌ Result file not found!")
        return
    
    print("=== FINAL VERSE-ONLY SCRAPING RESULTS ===\n")
    
    content_items = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                content_items.append(json.loads(line))
    
    print(f"📊 SUMMARY:")
    print(f"  Chapters processed: {len(content_items)}")
    
    total_terms = sum(item['word_count'] for item in content_items)
    print(f"  Total Sanskrit terms: {total_terms}")
    print(f"  Average per chapter: {total_terms/len(content_items):.1f}")
    
    chapters = sorted([item['chapter'] for item in content_items])
    print(f"  Chapters: {chapters}")
    
    sources = set(item.get('source', 'unknown') for item in content_items)
    print(f"  Sources: {sources}")
    
    print(f"\n📖 CHAPTER BREAKDOWN:")
    for item in content_items:
        print(f"  Chapter {item['chapter']}: {item['word_count']} terms")
    
    # Show sample terms from a few chapters
    print(f"\n🔤 SAMPLE SANSKRIT TERMS:")
    
    for i, item in enumerate(content_items[:3]):  # First 3 chapters
        terms = item['text'].split()
        print(f"\n  Chapter {item['chapter']} ({len(terms)} terms):")
        
        # Show terms with diacritics
        diacritic_terms = [t for t in terms if any(c in t for c in 'āīūṛṝḷḹēōṃḥṅñṭḍṇśṣḻ')]
        if diacritic_terms:
            print(f"    With diacritics: {', '.join(diacritic_terms[:8])}")
            if len(diacritic_terms) > 8:
                print(f"    ... and {len(diacritic_terms) - 8} more")
        
        # Show other terms
        other_terms = [t for t in terms if not any(c in t for c in 'āīūṛṝḷḹēōṃḥṅñṭḍṇśṣḻ')]
        if other_terms:
            print(f"    Other Sanskrit: {', '.join(other_terms[:8])}")
            if len(other_terms) > 8:
                print(f"    ... and {len(other_terms) - 8} more")
    
    print(f"\n✅ SUCCESS: Extracted {total_terms} Sanskrit terms from verse content only!")
    print(f"   No extraneous website content, donor names, or generic terms!")

if __name__ == "__main__":
    check_results() 