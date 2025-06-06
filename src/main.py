import cv2
import sys
import os
import time
import threading
import logging
import argparse
from typing import Optional

# Add src directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components
from video_source import VideoSource
from transcription import TranscriptionEngine
from caption_overlay import CaptionOverlay
from logging_utils import get_logger, TRACE, setup_logging
from core.argument_parser import parse_arguments
from core.app_initializer import initialize_app
from core.video_runner import VideoRunner
from clock import CLOCK

# Get logger instance
logger = get_logger(__name__)

# Process audio moved to TranscriptionEngine class

def main():
    """Main entry point for the application."""
    try:
        # Parse command line arguments using the core argument parser
        args = parse_arguments()
        
        # Initialize application
        video_source, transcriber, caption_overlay = initialize_app(args)
        
        # Run the application
        run_app(video_source, transcriber, caption_overlay, args)
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        sys.exit(1)

def run_app(video_source, transcriber, caption_overlay, args):
    """Run the application.
    
    Args:
        video_source: Initialized video source
        transcriber: Initialized transcription engine
        caption_overlay: Initialized caption overlay
        args: Command line arguments
    """
    try:
        # Start video source first
        logger.info("Starting video source...")
        video_source.start()
        
        # Start transcription engine
        logger.info("Starting transcription engine...")
        transcriber.start_transcription()
        
        # Start audio processing thread
        logger.info("Starting audio processing thread...")
        audio_thread = threading.Thread(
            target=transcriber.process_audio,
            args=(video_source,),
            name="AudioProcessingThread"
        )
        audio_thread.daemon = True
        audio_thread.start()
        
        # Create video runner with comparison mode if requested
        runner = VideoRunner(
            video_source=video_source,
            transcriber=transcriber,
            caption_overlay=caption_overlay,
            window_name="ISKCON-Translate Comparison" if args.comparison else "ISKCON-Translate",
            comparison_mode=args.comparison,
            youtube_url=args.youtube,
            headless=args.headless,
            secondary_languages=args.secondary_languages,
            primary_language=args.language
        )
        
        # Start video playback
        runner.run()
        
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Error in run_app: {e}", exc_info=True)
    finally:
        # Cleanup
        if 'video_source' in locals():
            video_source.release()
        if 'transcriber' in locals():
            transcriber.stop_transcription()
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
