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
    # Run with proper argument parsing
    main()
