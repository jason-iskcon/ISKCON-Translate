#!/usr/bin/env python3
"""
Create sample Bhagavad Gita content for testing the pipeline.
This ensures we have content even if web scraping fails.
"""

import json
from pathlib import Path

# Sample Bhagavad Gita content with spiritual vocabulary
SAMPLE_CONTENT = [
    {
        "url": "sample://bg/1/1",
        "title": "Bhagavad Gita 1.1",
        "text": "Dhritarashtra said: O Sanjaya, after my sons and the sons of Pandu assembled in the place of pilgrimage at Kurukshetra, desiring to fight, what did they do? The blind king Dhritarashtra inquired from his secretary Sanjaya about the battle between his sons and the sons of his brother Pandu. This inquiry was made because the king was anxious about the outcome of the battle.",
        "chapter": 1,
        "verse": "1",
        "source": "sample",
        "timestamp": 1640995200.0,
        "word_count": 65
    },
    {
        "url": "sample://bg/2/1",
        "title": "Bhagavad Gita 2.1",
        "text": "Sanjaya said: Seeing Arjuna full of compassion, his mind depressed, his eyes full of tears, Madhusudana, Krishna, spoke the following words. When Arjuna was overwhelmed with grief and compassion for his kinsmen, Krishna began to speak the eternal wisdom of the Bhagavad Gita.",
        "chapter": 2,
        "verse": "1",
        "source": "sample",
        "timestamp": 1640995200.0,
        "word_count": 45
    },
    {
        "url": "sample://bg/2/2",
        "title": "Bhagavad Gita 2.2",
        "text": "The Supreme Personality of Godhead said: My dear Arjuna, how have these impurities come upon you? They are not at all befitting a man who knows the value of life. They lead not to higher planets but to infamy. Krishna addresses Arjuna with compassion, questioning why he has fallen into this state of confusion and grief.",
        "chapter": 2,
        "verse": "2",
        "source": "sample",
        "timestamp": 1640995200.0,
        "word_count": 55
    },
    {
        "url": "sample://bg/2/47",
        "title": "Bhagavad Gita 2.47",
        "text": "You have a right to perform your prescribed duty, but not to the fruits of action. Never consider yourself the cause of the results of your activities, and never be attached to not doing your duty. This famous verse teaches the principle of karma yoga - performing one's duty without attachment to results. Krishna instructs Arjuna about the importance of dharma and detachment.",
        "chapter": 2,
        "verse": "47",
        "source": "sample",
        "timestamp": 1640995200.0,
        "word_count": 70
    },
    {
        "url": "sample://bg/4/7",
        "title": "Bhagavad Gita 4.7",
        "text": "Whenever and wherever there is a decline in religious practice, O descendant of Bharata, and a predominant rise of irreligion—at that time I descend Myself. Krishna explains to Arjuna the principle of divine incarnation. When dharma declines and adharma increases, the Supreme Lord manifests to restore righteousness and protect the devotees.",
        "chapter": 4,
        "verse": "7",
        "source": "sample",
        "timestamp": 1640995200.0,
        "word_count": 60
    },
    {
        "url": "sample://bg/7/1",
        "title": "Bhagavad Gita 7.1",
        "text": "The Supreme Personality of Godhead said: Now hear, O son of Pritha, how by practicing yoga in full consciousness of Me, with mind attached to Me, you shall know Me in full, free from doubt. Krishna begins teaching about knowledge of the Absolute Truth. He explains how one can know Him completely through devotional service and yoga practice.",
        "chapter": 7,
        "verse": "1",
        "source": "sample",
        "timestamp": 1640995200.0,
        "word_count": 55
    },
    {
        "url": "sample://bg/9/22",
        "title": "Bhagavad Gita 9.22",
        "text": "But those who always worship Me with exclusive devotion, meditating on My transcendental form—to them I carry what they lack, and I preserve what they have. Krishna promises to take care of His pure devotees. For those who worship Him with single-minded devotion, He personally ensures their welfare and protection.",
        "chapter": 9,
        "verse": "22",
        "source": "sample",
        "timestamp": 1640995200.0,
        "word_count": 50
    },
    {
        "url": "sample://bg/15/7",
        "title": "Bhagavad Gita 15.7",
        "text": "The living entities in this conditioned world are My eternal fragmental parts. Due to conditioned life, they are struggling very hard with the six senses, which include the mind. Krishna explains the nature of the soul and its relationship with the Supreme. The living beings are eternal parts of Krishna, but in material existence they struggle with the senses.",
        "chapter": 15,
        "verse": "7",
        "source": "sample",
        "timestamp": 1640995200.0,
        "word_count": 55
    },
    {
        "url": "sample://bg/18/65",
        "title": "Bhagavad Gita 18.65",
        "text": "Always think of Me, become My devotee, worship Me and offer your homage unto Me. Thus you will come to Me without fail. I promise you this because you are My very dear friend. This is one of the most important verses of the Bhagavad Gita. Krishna gives the ultimate instruction for spiritual realization - constant remembrance and devotion.",
        "chapter": 18,
        "verse": "65",
        "source": "sample",
        "timestamp": 1640995200.0,
        "word_count": 60
    },
    {
        "url": "sample://bg/18/66",
        "title": "Bhagavad Gita 18.66",
        "text": "Abandon all varieties of religion and just surrender unto Me. I shall deliver you from all sinful reactions. Do not fear. This is the final instruction of the Bhagavad Gita. Krishna asks Arjuna to surrender completely to Him, promising liberation from all karma and fear. This verse represents the essence of bhakti yoga and complete surrender to the Supreme.",
        "chapter": 18,
        "verse": "66",
        "source": "sample",
        "timestamp": 1640995200.0,
        "word_count": 65
    }
]

def create_sample_content(output_file: str = "raw_synonyms.jsonl"):
    """Create sample content file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in SAMPLE_CONTENT:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created sample content with {len(SAMPLE_CONTENT)} items in {output_path}")
    return len(SAMPLE_CONTENT)

if __name__ == "__main__":
    create_sample_content() 