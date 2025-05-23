"""Command-line argument parsing for ISKCON-Translate."""
import argparse

def parse_arguments():
    """Parse and return command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='ISKCON-Translate Video Captioning',
        add_help=False  # We'll add help manually to control formatting
    )
    
    # Add arguments with proper formatting
    parser.add_argument(
        'source',
        nargs='?',  # Make it optional
        default=None,
        help='Source video file path (positional or use --source)'
    )
    
    # For backward compatibility with --source
    parser.add_argument(
        '--source', 
        dest='source_arg',
        help=argparse.SUPPRESS  # Hide from help
    )
    
    parser.add_argument(
        '--log-level', 
        '-l',
        default='INFO',
        choices=['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set the logging level (default: %(default)s)'
    )
    
    parser.add_argument(
        '--help',
        '-h',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit'
    )
    
    return parser.parse_args()
