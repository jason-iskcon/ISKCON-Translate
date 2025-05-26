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

def setup_logging(level='INFO', log_file=None):
    """Setup logging configuration.
    
    Args:
        level (str): Logging level (default: 'INFO')
        log_file (str): Optional path to log file
    """
    try:
        # Convert string level to numeric value
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {level}')
            
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler if log file specified
        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            # Create rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            root_logger.info(f"Logging initialized at level {level} (console and file)")
        else:
            root_logger.info(f"Logging initialized at level {level} (console only)")
            
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
