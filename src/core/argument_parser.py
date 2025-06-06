"""Command-line argument parsing for ISKCON-Translate."""
import argparse
from typing import Optional

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="ISKCON-Translate - Synchronized Video Captioning")
    
    # Video file argument
    parser.add_argument("video_file", nargs="?", help="Path to video file")
    
    # Language support for transcription
    parser.add_argument("--language", "--lang", default="en", 
                       help="Language code for transcription (e.g., 'en', 'fr', 'it', 'es', 'de')")
    
    # Secondary language support for concurrent display
    parser.add_argument("--secondary-languages", "--sec-lang", nargs='+', default=[], 
                       help="Additional language codes for concurrent display (e.g., 'it', 'es')")
    
    # Comparison mode
    parser.add_argument("--comparison", action="store_true", help="Enable comparison mode")
    
    # YouTube URL for comparison
    parser.add_argument("--youtube", help="YouTube URL for comparison")
    
    # Logging level
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    # Headless mode
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no window display)")
    
    # Seek time
    parser.add_argument("--seek", type=float, default=0.0, help="Start time in seconds")
    
    return parser.parse_args()
