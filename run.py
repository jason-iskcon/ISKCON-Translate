"""
Synchronized Video Captioning System - Ultra Minimal MVP

Super simple launcher script for the application.
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import main function
from src.main import main

if __name__ == "__main__":
    # If video file provided as command line argument, use it
    video_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run with that file or let main use the default
    main(video_file)
