"""Utility functions for text processing and comparison."""
import re
from typing import Tuple, List, Optional

def normalize_text(t: str) -> str:
    """Normalize text for comparison by removing punctuation and normalizing whitespace.
    
    Args:
        t: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if not t:
        return ""
    # Convert to lowercase
    t = t.lower()
    # Remove all punctuation except apostrophes
    t = re.sub(r'[^\w\s\']', ' ', t)
    # Normalize whitespace and apostrophes
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'\'\s+', '\'', t)  # Fix spaces after apostrophes
    t = re.sub(r'\s+\'', '\'', t)  # Fix spaces before apostrophes
    return t

def levenshtein_ratio(s1: str, s2: str) -> float:
    """Calculate the Levenshtein distance ratio between two strings.
    
    The ratio is computed as (1 - distance/max_length), where distance is the
    Levenshtein distance between the two strings and max_length is the length
    of the longer string.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        float: Similarity ratio between 0.0 (completely different) and 1.0 (identical)
    """
    if not s1 or not s2:
        return 0.0
        
    if s1 == s2:
        return 1.0
        
    # Convert to lowercase for case-insensitive comparison
    s1 = s1.lower()
    s2 = s2.lower()
    
    # If one string is empty, the distance is the length of the other string
    if len(s1) == 0:
        return 1.0 - (len(s2) / max(len(s1), len(s2)))
    if len(s2) == 0:
        return 1.0 - (len(s1) / max(len(s1), len(s2)))
    
    # Initialize matrix of zeros
    rows = len(s1) + 1
    cols = len(s2) + 1
    distance = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Populate matrix of zeros with the indices of each character of both strings
    for i in range(1, rows):
        distance[i][0] = i
    for j in range(1, cols):
        distance[0][j] = j
    
    # Iterate over the matrix to compute the cost of operations
    for col in range(1, cols):
        for row in range(1, rows):
            if s1[row-1] == s2[col-1]:
                cost = 0  # If the characters are the same in the two strings, cost is 0
            else:
                cost = 1  # If not, cost is 1
            
            # Find the minimum cost for deletion, insertion, and substitution
            distance[row][col] = min(
                distance[row-1][col] + 1,      # Deletion
                distance[row][col-1] + 1,      # Insertion
                distance[row-1][col-1] + cost   # Substitution
            )
    
    # Calculate the ratio of the Levenshtein distance to the maximum possible distance
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance[-1][-1] / max_len)

def word_order_similarity(w1: List[str], w2: List[str]) -> float:
    """Calculate word order similarity between two lists of words.
    
    Args:
        w1: First list of words
        w2: Second list of words
        
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    if not w1 or not w2:
        return 0.0
        
    # Convert to sets for comparison
    set1 = set(w1)
    set2 = set(w2)
    
    # If either set is empty, return 0.0
    if not set1 or not set2:
        return 0.0
        
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
        
    return intersection / union

def text_similarity(text1: str, text2: str) -> float:
    """Calculate a text similarity score between two strings.
    
    This function combines multiple similarity metrics to determine how similar
    two pieces of text are. It's useful for detecting duplicate or near-duplicate
    captions.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        float: Similarity score between 0.0 (completely different) and 1.0 (identical)
    """
    if not text1 or not text2:
        return 0.0
        
    # If either string is empty after stripping, return 0.0
    text1 = text1.strip()
    text2 = text2.strip()
    if not text1 or not text2:
        return 0.0
        
    # If strings are exactly the same, return 1.0 immediately
    if text1 == text2:
        return 1.0
        
    # Normalize both texts
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    # If normalized strings are identical, return 1.0
    if norm1 == norm2:
        return 1.0
        
    # If one normalized string is a substring of the other, return high similarity
    if norm1 in norm2 or norm2 in norm1:
        shorter, longer = (norm1, norm2) if len(norm1) < len(norm2) else (norm2, norm1)
        similarity = len(shorter) / len(longer)
        # Only consider it a match if the shorter string is at least 60% of the longer one
        return similarity if similarity >= 0.6 else 0.0
    
    # Calculate Levenshtein ratio
    lev_ratio = levenshtein_ratio(norm1, norm2)
    
    # Calculate word order similarity
    words1 = [w for w in norm1.split() if w.strip()]
    words2 = [w for w in norm2.split() if w.strip()]
    word_sim = word_order_similarity(words1, words2)
    
    # Combine the scores (simple average for now, could be weighted)
    combined_score = (lev_ratio + word_sim) / 2.0
    
    return combined_score
