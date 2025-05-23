"""Application initialization and logging setup for ISKCON-Translate."""
import os
import logging
from logging.handlers import RotatingFileHandler
from ..logging_utils import get_logger, TRACE, setup_logging

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

def initialize_app(log_level='INFO'):
    """Initialize the application with logging configuration.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        tuple: (logs_dir, log_file) paths
    """
    # Ensure logs directory exists
    logs_dir = ensure_logs_dir()
    log_file = os.path.join(logs_dir, 'iskcon_translate.log')
    
    # Set up logging with the specified level and log file
    setup_logging(level=log_level, log_file=log_file)
    
    # Get logger instance
    logger = get_logger(__name__)
    logger.info("Application initialized with log level: %s", log_level)
    logger.info("Log file: %s", os.path.abspath(log_file))
    
    return logs_dir, log_file
