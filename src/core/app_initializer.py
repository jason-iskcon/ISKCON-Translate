"""Application initialization and logging setup for ISKCON-Translate."""
import os
import logging
from logging.handlers import RotatingFileHandler
from logging_utils import get_logger, TRACE, setup_logging
from video_source import VideoSource
from transcription import TranscriptionEngine
from caption_overlay import CaptionOverlay
from core.youtube_downloader import YouTubeDownloader

# Import multi-language engine
try:
    from transcription.multi_language_engine import MultiLanguageTranscriptionEngine
except ImportError:
    from src.transcription.multi_language_engine import MultiLanguageTranscriptionEngine

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

def resolve_video_path(video_input: str, cache_dir: str = None) -> str:
    """Resolve video input to a local file path.
    
    If the input is a YouTube URL, download it first.
    If it's a local file path, return as-is.
    
    Args:
        video_input: Video file path or YouTube URL
        cache_dir: Directory to store downloaded videos (defaults to ~/.video_cache)
        
    Returns:
        str: Path to local video file
        
    Raises:
        ValueError: If input is neither a valid file nor YouTube URL
        FileNotFoundError: If local file doesn't exist
    """
    if not video_input:
        raise ValueError("No video input provided")
    
    # Check if it's a YouTube URL
    downloader = YouTubeDownloader(cache_dir)
    if downloader.is_youtube_url(video_input):
        logger.info(f"Detected YouTube URL: {video_input}")
        return downloader.download_video(video_input)
    
    # Assume it's a local file path
    if not os.path.exists(video_input):
        raise FileNotFoundError(f"Video file not found: {video_input}")
    
    logger.info(f"Using local video file: {video_input}")
    return video_input

def initialize_video_source(args):
    """Initialize the video source.
    
    Args:
        args: Command line arguments
        
    Returns:
        VideoSource: Initialized video source
    """
    try:
        # Resolve video path (download if YouTube URL)
        video_path = resolve_video_path(args.video_file, args.cache_dir)
        
        video_source = VideoSource(video_path, start_time=args.seek)
        logger.info(f"Initialized video source with file: {video_path}, start_time: {args.seek}")
        return video_source
    except Exception as e:
        logger.error(f"Error initializing video source: {e}", exc_info=True)
        raise

def initialize_transcriber(args):
    """Initialize the transcription engine.
    
    Args:
        args: Command line arguments
        
    Returns:
        TranscriptionEngine: Initialized transcription engine (always English for source)
    """
    try:
        # Always transcribe in English first to get the source language
        # Then translate to both primary and secondary languages for concurrent display
        transcriber = TranscriptionEngine(language="en")  # Always English source
        logger.info(f"Initialized transcription engine with source language: en")
        
        # Collect all target languages for translation
        target_languages = []
        if args.language != "en":  # If primary language is not English, add it as translation target
            target_languages.append(args.language)
        if args.secondary_languages:
            target_languages.extend(args.secondary_languages)
            
        if target_languages:
            logger.info(f"Target languages for concurrent display: {target_languages}")
        
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
        transcriber = initialize_transcriber(args)
        caption_overlay = initialize_caption_overlay(args)
        
        return video_source, transcriber, caption_overlay
        
    except Exception as e:
        logger.error(f"Error initializing application: {e}", exc_info=True)
        raise
