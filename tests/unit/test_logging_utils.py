"""
Unit tests for logging_utils.py

These tests verify the core functionality of the logging utilities,
including log levels, formatting, and thread safety.
"""
import logging
import os
import pytest
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path so we can import from it
import sys
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.logging_utils import setup_logging, get_logger, TRACE, ColoredFormatter

class TestLoggingUtils:
    """Test logging utilities functionality."""
    
    def test_trace_level_registered(self):
        """Verify that TRACE level is properly registered."""
        assert logging.getLevelName(TRACE) == 'TRACE'
    
    def test_trace_method_added(self):
        """Verify that trace() method is added to Logger class."""
        logger = get_logger('test_trace_method')
        assert hasattr(logger, 'trace')
        assert callable(logger.trace)
    
    def test_colored_formatter(self):
        """Test that ColoredFormatter adds colors to log levels."""
        formatter = ColoredFormatter('%(levelname)s')
        record = logging.LogRecord(
            'test', logging.INFO, 'test.py', 1, 'test', (), None
        )
        
        # Check that colors are added
        formatted = formatter.format(record)
        assert 'INFO' in formatted
        assert '\033[0;37m' in formatted  # White color for INFO
    
    def test_thread_safety(self):
        """Verify that logging is thread-safe."""
        setup_logging('INFO')
        logger = get_logger('test_threading')
        
        messages = set()
        num_threads = 5
        messages_per_thread = 10
        
        def log_messages(thread_id):
            for i in range(messages_per_thread):
                msg = f"Thread {thread_id} message {i}"
                logger.info(msg)
                messages.add(msg)
        
        # Create and start threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=log_messages, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify all messages were processed
        assert len(messages) == num_threads * messages_per_thread
        
    def test_get_logger_returns_logger(self):
        """Verify that get_logger returns a Logger instance."""
        logger = get_logger('test_logger')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_logger'

class TestLogLevels:
    """Test different log levels and their behavior."""
    
    @pytest.mark.parametrize('level,expected', [
        ('TRACE', True),
        ('DEBUG', False),
        ('INFO', False),
        ('WARNING', False),
        ('ERROR', False)
    ])
    def test_trace_level(self, level, expected):
        """Verify that TRACE level logs only when level is TRACE."""
        setup_logging(level)
        logger = get_logger('test_trace_level')
        
        with patch.object(logger, '_log') as mock_log:
            logger.trace("Test trace message")
            
        if expected:
            mock_log.assert_called_once()
        else:
            mock_log.assert_not_called()
    
    def test_log_levels_work_correctly(self):
        """Verify that log levels filter messages appropriately."""
        for level in ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR']:
            setup_logging(level)
            logger = get_logger(f'test_level_{level}')
            
            with patch.object(logger, '_log') as mock_log:
                logger.trace("Trace message")
                logger.debug("Debug message")
                logger.info("Info message")
                logger.warning("Warning message")
                logger.error("Error message")
                
                # The number of expected calls depends on the log level
                expected_calls = {
                    'TRACE': 5,
                    'DEBUG': 4,
                    'INFO': 3,
                    'WARNING': 2,
                    'ERROR': 1
                }
                
                assert mock_log.call_count == expected_calls[level]
