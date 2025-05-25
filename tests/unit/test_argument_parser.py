import pytest
import sys
import argparse
from pathlib import Path
from unittest.mock import patch

# Add src directory to path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.core.argument_parser import parse_arguments


class TestArgumentParser:
    """Test suite for command line argument parsing."""
    
    def test_parse_arguments_no_args(self):
        """Test parsing with no command line arguments."""
        with patch('sys.argv', ['program']):
            args = parse_arguments()
            
            assert args.source is None
            assert args.source_arg is None
            assert args.log_level == 'INFO'  # Default
    
    def test_parse_arguments_positional_source(self):
        """Test parsing with positional source argument."""
        with patch('sys.argv', ['program', '/path/to/video.mp4']):
            args = parse_arguments()
            
            assert args.source == '/path/to/video.mp4'
            assert args.source_arg is None
            assert args.log_level == 'INFO'
    
    def test_parse_arguments_source_flag(self):
        """Test parsing with --source flag."""
        with patch('sys.argv', ['program', '--source', '/path/to/video.mp4']):
            args = parse_arguments()
            
            assert args.source is None  # Positional is None
            assert args.source_arg == '/path/to/video.mp4'
            assert args.log_level == 'INFO'
    
    def test_parse_arguments_both_source_args(self):
        """Test parsing with both positional and --source arguments."""
        with patch('sys.argv', ['program', '/pos/video.mp4', '--source', '/flag/video.mp4']):
            args = parse_arguments()
            
            assert args.source == '/pos/video.mp4'
            assert args.source_arg == '/flag/video.mp4'
            assert args.log_level == 'INFO'
    
    def test_parse_arguments_log_level_debug(self):
        """Test parsing with DEBUG log level."""
        with patch('sys.argv', ['program', '--log-level', 'DEBUG']):
            args = parse_arguments()
            
            assert args.source is None
            assert args.source_arg is None
            assert args.log_level == 'DEBUG'
    
    def test_parse_arguments_log_level_short_flag(self):
        """Test parsing with short -l flag for log level."""
        with patch('sys.argv', ['program', '-l', 'ERROR']):
            args = parse_arguments()
            
            assert args.log_level == 'ERROR'
    
    def test_parse_arguments_all_log_levels(self):
        """Test all valid log levels."""
        valid_levels = ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR']
        
        for level in valid_levels:
            with patch('sys.argv', ['program', '--log-level', level]):
                args = parse_arguments()
                assert args.log_level == level
    
    def test_parse_arguments_invalid_log_level(self):
        """Test parsing with invalid log level."""
        with patch('sys.argv', ['program', '--log-level', 'INVALID']):
            with pytest.raises(SystemExit):  # argparse exits on invalid choice
                parse_arguments()
    
    def test_parse_arguments_complex_combination(self):
        """Test complex combination of arguments."""
        with patch('sys.argv', [
            'program', 
            '/path/to/video.mp4',  # Positional source
            '--source', '/another/video.mp4',  # Flag source
            '--log-level', 'WARNING'
        ]):
            args = parse_arguments()
            
            assert args.source == '/path/to/video.mp4'
            assert args.source_arg == '/another/video.mp4'
            assert args.log_level == 'WARNING'
    
    def test_parse_arguments_help_flag(self):
        """Test help flag causes system exit."""
        with patch('sys.argv', ['program', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            # Help should exit with code 0
            assert exc_info.value.code == 0
    
    def test_parse_arguments_help_flag_short(self):
        """Test short help flag causes system exit."""
        with patch('sys.argv', ['program', '-h']):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            # Help should exit with code 0
            assert exc_info.value.code == 0
    
    def test_parse_arguments_file_paths_with_spaces(self):
        """Test parsing file paths with spaces."""
        with patch('sys.argv', ['program', '/path/to/video file.mp4']):
            args = parse_arguments()
            
            assert args.source == '/path/to/video file.mp4'
    
    def test_parse_arguments_relative_paths(self):
        """Test parsing with relative file paths."""
        with patch('sys.argv', ['program', './video.mp4', '--source', '../other.mp4']):
            args = parse_arguments()
            
            assert args.source == './video.mp4'
            assert args.source_arg == '../other.mp4'
    
    def test_parse_arguments_windows_paths(self):
        """Test parsing with Windows-style paths."""
        with patch('sys.argv', ['program', 'C:\\Videos\\test.mp4']):
            args = parse_arguments()
            
            assert args.source == 'C:\\Videos\\test.mp4'
    
    def test_parse_arguments_url_source(self):
        """Test parsing with URL as source."""
        with patch('sys.argv', ['program', 'https://example.com/video.mp4']):
            args = parse_arguments()
            
            assert args.source == 'https://example.com/video.mp4'
    
    def test_parse_arguments_special_characters(self):
        """Test parsing with special characters in paths."""
        special_path = '/path/with/special-chars_123.mp4'
        with patch('sys.argv', ['program', special_path]):
            args = parse_arguments()
            
            assert args.source == special_path
    
    def test_parse_arguments_empty_string_source(self):
        """Test parsing with empty string as source."""
        with patch('sys.argv', ['program', '']):
            args = parse_arguments()
            
            assert args.source == ''


class TestArgumentParserEdgeCases:
    """Test edge cases and error scenarios for argument parser."""
    
    def test_parse_arguments_unknown_flag(self):
        """Test parsing with unknown command line flag."""
        with patch('sys.argv', ['program', '--unknown-flag', 'value']):
            with pytest.raises(SystemExit):  # Should exit on unknown argument
                parse_arguments()
    
    def test_parse_arguments_missing_log_level_value(self):
        """Test parsing when log level flag is missing its value."""
        with patch('sys.argv', ['program', '--log-level']):
            with pytest.raises(SystemExit):  # Should exit on missing required value
                parse_arguments()
    
    def test_parse_arguments_case_sensitive_log_level(self):
        """Test that log levels are case sensitive."""
        with patch('sys.argv', ['program', '--log-level', 'debug']):  # lowercase
            with pytest.raises(SystemExit):  # Should exit on invalid choice
                parse_arguments()
    
    def test_parse_arguments_multiple_positional_sources(self):
        """Test behavior with multiple positional arguments."""
        with patch('sys.argv', ['program', 'video1.mp4', 'video2.mp4']):
            with pytest.raises(SystemExit):  # Should exit on too many positional args
                parse_arguments()
    
    def test_parse_arguments_log_level_with_equals(self):
        """Test log level parsing with equals sign."""
        with patch('sys.argv', ['program', '--log-level=DEBUG']):
            args = parse_arguments()
            
            assert args.log_level == 'DEBUG'
    
    def test_parse_arguments_source_with_equals(self):
        """Test source parsing with equals sign."""
        with patch('sys.argv', ['program', '--source=/path/to/video.mp4']):
            args = parse_arguments()
            
            assert args.source_arg == '/path/to/video.mp4'
    
    def test_parse_arguments_mixed_flags_and_positional(self):
        """Test mixing flags and positional arguments in different orders."""
        # Test flags before positional
        with patch('sys.argv', ['program', '--log-level', 'WARNING', 'video.mp4']):
            args = parse_arguments()
            assert args.source == 'video.mp4'
            assert args.log_level == 'WARNING'
        
        # Test flags after positional
        with patch('sys.argv', ['program', 'video.mp4', '--log-level', 'DEBUG']):
            args = parse_arguments()
            assert args.source == 'video.mp4'
            assert args.log_level == 'DEBUG'
    
    def test_parse_arguments_duplicate_log_level_flags(self):
        """Test behavior with duplicate log level flags."""
        with patch('sys.argv', ['program', '--log-level', 'DEBUG', '--log-level', 'INFO']):
            args = parse_arguments()
            # Last one should win
            assert args.log_level == 'INFO'
    
    def test_parse_arguments_duplicate_source_flags(self):
        """Test behavior with duplicate source flags."""
        with patch('sys.argv', ['program', '--source', 'video1.mp4', '--source', 'video2.mp4']):
            args = parse_arguments()
            # Last one should win
            assert args.source_arg == 'video2.mp4'


class TestArgumentParserIntegration:
    """Integration tests for argument parser with realistic scenarios."""
    
    def test_typical_usage_scenarios(self):
        """Test typical usage scenarios."""
        scenarios = [
            # Basic usage with just a video file
            (['program', 'video.mp4'], {'source': 'video.mp4', 'log_level': 'INFO'}),
            
            # Debug mode
            (['program', 'video.mp4', '-l', 'DEBUG'], {'source': 'video.mp4', 'log_level': 'DEBUG'}),
            
            # Using --source flag instead of positional
            (['program', '--source', 'video.mp4'], {'source_arg': 'video.mp4', 'log_level': 'INFO'}),
            
            # Production mode with minimal logging
            (['program', 'video.mp4', '--log-level', 'ERROR'], {'source': 'video.mp4', 'log_level': 'ERROR'}),
            
            # Development mode with verbose logging
            (['program', 'test.mp4', '--log-level', 'TRACE'], {'source': 'test.mp4', 'log_level': 'TRACE'}),
        ]
        
        for argv, expected in scenarios:
            with patch('sys.argv', argv):
                args = parse_arguments()
                
                for key, value in expected.items():
                    assert getattr(args, key) == value
    
    def test_command_line_compatibility(self):
        """Test compatibility with different command line styles."""
        # Unix-style long options
        with patch('sys.argv', ['program', '--log-level', 'DEBUG']):
            args = parse_arguments()
            assert args.log_level == 'DEBUG'
        
        # Unix-style short options
        with patch('sys.argv', ['program', '-l', 'DEBUG']):
            args = parse_arguments()
            assert args.log_level == 'DEBUG'
        
        # GNU-style with equals
        with patch('sys.argv', ['program', '--log-level=DEBUG']):
            args = parse_arguments()
            assert args.log_level == 'DEBUG'
    
    def test_realistic_file_paths(self):
        """Test with realistic file paths."""
        realistic_paths = [
            '/home/user/Videos/iskcon_lecture.mp4',
            'C:\\Users\\User\\Videos\\krishna_consciousness.mp4',
            './videos/bhagavad_gita_class.mp4',
            '../shared/iskcon_kirtan.mp4',
            '~/Downloads/spiritual_discourse.mp4',
            '/mnt/nas/video_archive/2023/lecture_001.mp4'
        ]
        
        for path in realistic_paths:
            with patch('sys.argv', ['program', path]):
                args = parse_arguments()
                assert args.source == path
    
    def test_argument_namespace_completeness(self):
        """Test that parsed arguments contain all expected attributes."""
        with patch('sys.argv', ['program', 'video.mp4', '--log-level', 'DEBUG']):
            args = parse_arguments()
            
            # Check that all expected attributes exist
            assert hasattr(args, 'source')
            assert hasattr(args, 'source_arg')
            assert hasattr(args, 'log_level')
            
            # Check types
            assert isinstance(args.source, str)
            assert args.source_arg is None
            assert isinstance(args.log_level, str)
    
    def test_parser_configuration(self):
        """Test that the parser is configured correctly."""
        # This test ensures the parser function creates the expected parser configuration
        with patch('sys.argv', ['program']):
            args = parse_arguments()
            
            # Should have default values
            assert args.log_level == 'INFO'
            assert args.source is None
            assert args.source_arg is None 