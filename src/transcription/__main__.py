"""
Entry point for running GT-Whisper as a module.

This allows the tool to be run with:
    python -m src.transcription audio.mp3 --strategy static_common
"""

from .gt_whisper import main

if __name__ == "__main__":
    main() 