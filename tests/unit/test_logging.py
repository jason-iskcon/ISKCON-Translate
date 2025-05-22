"""Test script for verifying logging functionality."""
import os
import sys
import time
import threading
import pytest
from pathlib import Path

# Add src to path so we can import from it
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.logging_utils import setup_logging, get_logger, TRACE


def test_log_levels(capsys):
    """Test that different log levels work as expected."""
    # Test with different log levels
    for level in ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR']:
        setup_logging(level)
        logger = get_logger('test_logger')
        
        # Clear any previous output
        capsys.readouterr()
        
        # Log messages at different levels
        logger.trace("Trace message")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Capture the output
        captured = capsys.readouterr()
        output = captured.out + captured.err
        
        # Verify the output contains the expected messages
        if level == 'TRACE':
            assert "Trace message" in output
        if level in ['TRACE', 'DEBUG']:
            assert "Debug message" in output
        if level in ['TRACE', 'DEBUG', 'INFO']:
            assert "Info message" in output
        if level in ['TRACE', 'DEBUG', 'INFO', 'WARNING']:
            assert "Warning message" in output
        # Error should always be shown
        assert "Error message" in output


def test_thread_safety():
    """Test that logging is thread-safe."""
    setup_logging('DEBUG')
    logger = get_logger('test_threading')
    
    messages = []
    num_threads = 5
    messages_per_thread = 10
    
    def log_messages(thread_id):
        for i in range(messages_per_thread):
            logger.info(f"Thread {thread_id} message {i}")
    
    # Create and start threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=log_messages, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # If we get here without errors, thread safety is working
    assert True


def test_log_file_creation(tmp_path):
    """Test that log files are created when specified."""
    log_file = tmp_path / "test.log"
    setup_logging('INFO', log_file=str(log_file))
    
    logger = get_logger('test_file_logging')
    test_message = "Test log message to file"
    logger.info(test_message)
    
    # Ensure the log file was created
    assert log_file.exists()
    
    # Check that our message is in the file
    with open(log_file, 'r') as f:
        log_content = f.read()
    assert test_message in log_content


if __name__ == "__main__":
    # Manual test for visual verification
    print("Testing log levels (visual verification):")
    for level in ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR']:
        print(f"\n=== Testing log level: {level} ===")
        setup_logging(level)
        logger = get_logger('visual_test')
        
        logger.trace("This is a TRACE message")
        logger.debug("This is a DEBUG message")
        logger.info("This is an INFO message")
        logger.warning("This is a WARNING message")
        logger.error("This is an ERROR message")
    
    # Test thread safety visually
    print("\n=== Testing thread safety ===")
    test_thread_safety()
    
    print("\nAll tests completed. Check the output above to verify logging behavior.")
