"""
Performance tests for logging utilities.

These tests measure the performance impact of logging at different levels
and verify that logging overhead is acceptable in production.
"""
import timeit
import pytest
import logging
from pathlib import Path
import sys

# Add src to path so we can import from it
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.logging_utils import setup_logging, get_logger

# Number of iterations for performance tests
PERF_ITERATIONS = 10000

class TestLoggingPerformance:
    """Test the performance impact of logging at different levels."""
    
    @pytest.mark.parametrize('level', ['ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE'])
    def test_logging_overhead(self, level, benchmark):
        """Measure the overhead of logging at different levels."""
        setup_logging(level)
        logger = get_logger(f'perf_{level}')
        
        def log_messages():
            logger.trace("Trace message")
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
        
        # Use benchmark fixture if available (pytest-benchmark)
        if hasattr(benchmark, 'pedantic'):
            benchmark.pedantic(log_messages, iterations=100, rounds=100)
        else:
            # Fallback to timeit
            time_taken = timeit.timeit(log_messages, number=PERF_ITERATIONS)
            print(f"\n{level}: {time_taken:.6f} seconds for {PERF_ITERATIONS} iterations")
            print(f"Average: {time_taken/PERF_ITERATIONS*1e6:.2f} μs per log call")
    
    def test_no_logging_performance(self, benchmark):
        """Establish a baseline with no logging."""
        logger = logging.getLogger('no_logging')
        logger.disabled = True
        
        def no_log():
            logger.trace("Trace message")
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
        
        if hasattr(benchmark, 'pedantic'):
            benchmark.pedantic(no_log, iterations=100, rounds=100)
    
    def test_file_logging_performance(self, tmp_path):
        """Measure the performance impact of file logging."""
        log_file = tmp_path / "performance_test.log"
        setup_logging('INFO', log_file=str(log_file))
        logger = get_logger('file_perf')
        
        def log_to_file():
            logger.info("This is a performance test message")
        
        # Time file logging
        time_taken = timeit.timeit(log_to_file, number=PERF_ITERATIONS)
        
        print(f"\nFile logging: {time_taken:.6f} seconds for {PERF_ITERATIONS} iterations")
        print(f"Average: {time_taken/PERF_ITERATIONS*1e6:.2f} μs per file log call")
        
        # Verify the log file was created and has content
        assert log_file.exists()
        with open(log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 1  # At least one line should be logged

if __name__ == "__main__":
    # Simple test runner for quick performance checks
    print("=== Logging Performance Tests ===\n")
    
    # Test with different log levels
    for level in ['ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE']:
        setup_logging(level)
        logger = get_logger(f'perf_{level}')
        
        def log_messages():
            logger.trace("Trace message")
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
        
        # Time the logging operations
        time_taken = timeit.timeit(log_messages, number=PERF_ITERATIONS)
        
        print(f"{level}: {time_taken:.6f} seconds for {PERF_ITERATIONS} iterations")
        print(f"Average: {time_taken/PERF_ITERATIONS*1e6:.2f} μs per log call\n")
