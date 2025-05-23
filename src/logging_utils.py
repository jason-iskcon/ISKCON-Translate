"""
Logging utilities for ISKCON-Translate project.

This module provides enhanced logging functionality including:
- Custom TRACE log level
- Consistent log formatting
- Helper functions for logging
"""

import logging
import os
import sys
from typing import Optional, Union
from logging.handlers import RotatingFileHandler

# Define custom log levels
TRACE = logging.DEBUG - 5
logging.addLevelName(TRACE, "TRACE")

# Add trace method to Logger class
def _trace(self, message: str, *args, **kwargs) -> None:
    """Log 'message % args' with severity 'TRACE'.
    
    Args:
        message: The message to log
        *args: Format arguments for the message
        **kwargs: Additional arguments for the logger
    """
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)

# Add trace method to Logger class
logging.Logger.trace = _trace

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    # ANSI color codes
    COLORS = {
        'TRACE': '\033[0;36m',  # Cyan
        'DEBUG': '\033[0;32m',  # Green
        'INFO': '\033[0;37m',   # White
        'WARNING': '\033[1;33m', # Yellow
        'ERROR': '\033[1;31m',   # Red
        'CRITICAL': '\033[1;41m', # Red background
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format the specified record as text with colors."""
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname:8}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logging(level: Union[str, int] = logging.INFO, log_file: Optional[str] = None) -> None:
    """Set up logging with the specified log level and optional log file.
    
    Args:
        level: Logging level (string or int)
        log_file: Optional path to a log file. If None, logs only to console.
    """
    if isinstance(level, str):
        level = level.upper()
        if level == 'TRACE':
            level = TRACE
        else:
            level = getattr(logging, level, logging.INFO)
    
    # Create formatters
    colored_formatter = ColoredFormatter(
        '%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(colored_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Set up rotating file handler (10MB per file, keep 5 backup files)
        max_bytes = 10 * 1024 * 1024  # 10MB
        backup_count = 5
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        logger = logging.getLogger(__name__)
        logger.info(f"Logging to rotating file: {os.path.abspath(log_file)} (max {max_bytes/1024/1024}MB, {backup_count} backups)")
    
    # Log the logging level and output location
    logger = logging.getLogger(__name__)
    if log_file:
        logger.info("Logging initialized at level %s to %s", 
                   logging.getLevelName(level), os.path.abspath(log_file))
    else:
        logger.info("Logging initialized at level %s (console only)", 
                   logging.getLevelName(level))

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

# Set up default logging when module is imported
setup_logging()
