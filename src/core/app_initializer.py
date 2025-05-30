"""Application initialization and logging setup for ISKCON-Translate."""
import os
import logging
from logging.handlers import RotatingFileHandler
from logging_utils import get_logger, TRACE, setup_logging
from video_source import VideoSource
from transcription import TranscriptionEngine
from caption_overlay import CaptionOverlay

# Get logger instance
logger = get_logger(__name__)

def ensure_logs_dir():
    """Ensure the logs directory exists.
    
    Returns:
        str: Path to the logs directory
    """
    logs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'logs'
    )
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def initialize_video_source(args):
    """Initialize the video source.
    
    Args:
        args: Command line arguments
        
    Returns:
        VideoSource: Initialized video source
    """
    try:
        video_source = VideoSource(args.video_file, start_time=args.seek)
        logger.info(f"Initialized video source with file: {args.video_file}, start_time: {args.seek}")
        return video_source
    except Exception as e:
        logger.error(f"Error initializing video source: {e}", exc_info=True)
        raise

def initialize_transcriber():
    """Initialize the transcription engine.
    
    Returns:
        TranscriptionEngine: Initialized transcription engine
    """
    try:
        transcriber = TranscriptionEngine()
        logger.info("Initialized transcription engine")
        return transcriber
    except Exception as e:
        logger.error(f"Error initializing transcription engine: {e}", exc_info=True)
        raise

def initialize_caption_overlay(args):
    """Initialize the caption overlay.
    
    Args:
        args: Command line arguments
        
    Returns:
        CaptionOverlay: Initialized caption overlay
    """
    try:
        caption_overlay = CaptionOverlay()
        # Set initial video start time from args.seek
        caption_overlay.orchestrator.core.set_video_start_time(args.seek)
        logger.info(f"Initialized caption overlay with start time: {args.seek}")
        return caption_overlay
    except Exception as e:
        logger.error(f"Error initializing caption overlay: {e}", exc_info=True)
        raise

def initialize_app(args):
    """Initialize the application with the given arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (video_source, transcriber, caption_overlay)
    """
    try:
        # Setup logging first
        log_file = os.path.join('logs', 'app.log')
        setup_logging(level=args.log_level, log_file=log_file)
        
        # Initialize components
        video_source = initialize_video_source(args)
        transcriber = initialize_transcriber()
        caption_overlay = initialize_caption_overlay(args)
        
        return video_source, transcriber, caption_overlay
        
    except Exception as e:
        logger.error(f"Error initializing application: {e}", exc_info=True)
        raise
