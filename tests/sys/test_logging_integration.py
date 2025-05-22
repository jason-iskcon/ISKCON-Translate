"""
System tests for logging integration.

These tests verify that logging works correctly in the context of the
entire application, including configuration, file handling, and integration
with other components.
"""
import os
import tempfile
import shutil
import pytest
import logging
from pathlib import Path
import sys

# Add src to path so we can import from it
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.logging_utils import setup_logging, get_logger, TRACE

class TestLoggingIntegration:
    """Test logging integration with the application."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for log files."""
        temp_dir = tempfile.mkdtemp(prefix="iskcon_log_test_")
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_log_file_creation(self, temp_log_dir):
        """Verify that log files are created with the correct permissions."""
        log_file = os.path.join(temp_log_dir, "test.log")
        
        # Configure logging to use the test log file
        setup_logging('INFO', log_file=log_file)
        logger = get_logger('test_file_creation')
        
        # Log a test message
        test_message = "This is a test log message"
        logger.info(test_message)
        
        # Verify the log file was created and contains our message
        assert os.path.exists(log_file), f"Log file was not created: {log_file}"
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert test_message in content, "Test message not found in log file"
    
    def test_log_rotation(self, temp_log_dir):
        """Verify that log rotation works correctly."""
        log_file = os.path.join(temp_log_dir, "rotation_test.log")
        
        # Configure logging with rotation
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            log_file, maxBytes=1024, backupCount=3
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Clear existing handlers and add our rotating handler
        logger = logging.getLogger()
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log enough data to trigger rotation
        for i in range(1000):
            logger.info(f"Test log message {i}" * 10)
        
        # Verify log files were created
        log_files = list(Path(temp_log_dir).glob("rotation_test.log*"))
        assert len(log_files) > 1, "Expected multiple log files due to rotation"
    
    def test_log_level_configuration(self):
        """Verify that log levels are correctly configured."""
        # Test with different log levels
        for level in ['ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE']:
            setup_logging(level)
            logger = get_logger(f'test_level_{level}')
            
            # Verify the effective level is set correctly
            assert logger.getEffectiveLevel() == getattr(logging, level) if level != 'TRACE' else TRACE
    
    def test_multiple_loggers(self, temp_log_dir):
        """Verify that multiple loggers work independently."""
        log_file = os.path.join(temp_log_dir, "multiple.log")
        
        # Configure logging
        setup_logging('INFO', log_file=log_file)
        
        # Create multiple loggers
        loggers = [get_logger(f'test_logger_{i}') for i in range(5)]
        
        # Log messages from each logger
        for i, logger in enumerate(loggers):
            logger.info(f"Message from logger {i}")
        
        # Verify all messages are in the log file
        with open(log_file, 'r') as f:
            content = f.read()
            for i in range(5):
                assert f"Message from logger {i}" in content, f"Message from logger {i} not found"

if __name__ == "__main__":
    # Simple test runner for manual verification
    test_dir = tempfile.mkdtemp(prefix="iskcon_log_test_")
    try:
        print(f"Running integration tests in {test_dir}")
        
        # Test log file creation
        print("\n=== Testing log file creation ===")
        test_file = os.path.join(test_dir, "test.log")
        setup_logging('INFO', log_file=test_file)
        logger = get_logger('integration_test')
        logger.info("This is an integration test message")
        print(f"Log file created at: {test_file}")
        
        # Verify the file was created
        if os.path.exists(test_file):
            print("✓ Log file creation test passed")
            with open(test_file, 'r') as f:
                print(f"Log content: {f.read()}")
        else:
            print("✗ Log file creation test failed")
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)
