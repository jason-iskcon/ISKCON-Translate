"""
GT-Whisper: Enhanced Whisper transcription with glossary support.

Command-line interface and minimal REST API for the enhanced Whisper
transcription system with context-aware prompting and fuzzy post-correction.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import os
import logging
import torch
import numpy as np

# Fix relative imports
from .decode_wrapper import DecodeWrapper, TranscriptionResult
from .glossary_policy import create_glossary_selector, GlossaryStrategy
from .post_processor import FuzzyPostProcessor
from .model_loader import init_whisper_model
from .audio_utils import get_audio_duration, load_audio
from .config import WhisperConfig
from src.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


class GTWhisper:
    """
    GT-Whisper: Enhanced transcription engine.
    
    Combines Whisper with glossary-enhanced prompting and fuzzy post-correction
    for improved accuracy on domain-specific content.
    """
    
    def __init__(self,
                 model_size: str = "small",
                 device: str = "auto",
                 strategy: str = "static_common",
                 glossary_path: Optional[str] = None,
                 glossary_dir: Optional[str] = None,
                 enable_post_correction: bool = True,
                 context_window: int = 32):
        """
        Initialize GT-Whisper.
        
        Args:
            model_size: Whisper model size
            device: Device to use (auto, cpu, cuda)
            strategy: Glossary strategy (static_common, chapter_guess, empty)
            glossary_path: Path to static glossary file
            glossary_dir: Directory with chapter-specific glossaries
            enable_post_correction: Whether to enable fuzzy post-correction
            context_window: Number of context words to use
        """
        self.model_size = model_size
        self.device = device
        self.strategy = strategy
        self.enable_post_correction = enable_post_correction
        self.context_window = context_window
        
        # Initialize components
        logger.info(f"Initializing GT-Whisper: model={model_size}, device={device}, strategy={strategy}")
        
        # Initialize Whisper model
        self.model, self.actual_device, self.compute_type = init_whisper_model(
            model_size, device, "auto"
        )
        
        # Initialize decode wrapper
        self.decoder = DecodeWrapper(
            model=self.model,
            context_window=context_window
        )
        
        # Initialize glossary policy
        self.glossary_selector = create_glossary_selector(
            strategy=strategy,
            glossary_path=glossary_path,
            glossary_dir=glossary_dir
        )
        
        # Initialize post-processor
        if enable_post_correction:
            self.post_processor = FuzzyPostProcessor()
            if glossary_path:
                self.post_processor.load_terms_from_file(glossary_path)
        else:
            self.post_processor = None
        
        logger.info(f"GT-Whisper initialized successfully on {self.actual_device}")
    
    def transcribe_file(self,
                       audio_path: str,
                       language: Optional[str] = None,
                       output_format: str = "text",
                       output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional)
            output_format: Output format (text, json, srt)
            output_file: Output file path (optional)
            
        Returns:
            dict: Transcription result with metadata
        """
        start_time = time.perf_counter()
        
        try:
            # Get glossary for this context
            context = {
                'audio_path': audio_path,
                'previous_text': " ".join(self.decoder.previous_transcript)
            }
            
            glossary_result = self.glossary_selector.get_glossary(context)
            
            # Load glossary into decoder
            self.decoder.glossary_text = glossary_result.glossary_text
            
            logger.info(f"Transcribing {audio_path} with {glossary_result.strategy_used} strategy")
            
            # Perform transcription
            result = self.decoder.transcribe(
                audio_path=audio_path,
                language=language
            )
            
            # Apply post-correction if enabled
            if self.post_processor and result.text:
                correction_result = self.post_processor.correct_text(result.text)
                corrected_text = correction_result.corrected_text
                corrections_made = len(correction_result.corrections_made)
            else:
                corrected_text = result.text
                corrections_made = 0
            
            # Calculate total processing time
            total_time = time.perf_counter() - start_time
            
            # Prepare output
            output_data = {
                'text': corrected_text,
                'original_text': result.text,
                'language': result.language,
                'language_probability': result.language_probability,
                'processing_time': total_time,
                'transcription_time': result.processing_time,
                'glossary_strategy': glossary_result.strategy_used,
                'glossary_matches': result.glossary_matches,
                'corrections_made': corrections_made,
                'context_used': result.context_used,
                'segments': result.segments,
                'metadata': {
                    'model_size': self.model_size,
                    'device': self.actual_device,
                    'compute_type': self.compute_type,
                    'context_window': self.context_window,
                    'post_correction_enabled': self.enable_post_correction
                }
            }
            
            # Save output if requested
            if output_file:
                self._save_output(output_data, output_file, output_format)
            
            logger.info(f"Transcription completed in {total_time:.2f}s: {len(corrected_text)} chars")
            
            return output_data
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'processing_time': time.perf_counter() - start_time
            }
    
    def _save_output(self, data: Dict[str, Any], output_file: str, format_type: str) -> None:
        """Save output to file in specified format."""
        output_path = Path(output_file)
        
        try:
            if format_type == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
            elif format_type == "srt":
                self._save_srt(data, output_path)
                
            else:  # text format
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(data['text'])
            
            logger.info(f"Output saved to {output_path} ({format_type} format)")
            
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
    
    def _save_srt(self, data: Dict[str, Any], output_path: Path) -> None:
        """Save output in SRT subtitle format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(data['segments'], 1):
                start_time = self._format_srt_time(segment['start'])
                end_time = self._format_srt_time(segment['end'])
                text = segment['text'].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'decoder': self.decoder.get_performance_stats(),
            'glossary': self.glossary_selector.get_policy_stats(),
        }
        
        if self.post_processor:
            stats['post_processor'] = self.post_processor.get_performance_stats()
        
        return stats


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="GT-Whisper: Enhanced Whisper transcription with glossary support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m gt_whisper audio.mp3
  python -m gt_whisper audio.mp3 --strategy static_common --output result.txt
  python -m gt_whisper audio.mp3 --strategy chapter_guess --glossary-dir ./glossaries
  python -m gt_whisper audio.mp3 --format json --output result.json
  python -m gt_whisper audio.mp3 --model medium --device cuda --language hi
        """
    )
    
    # Required arguments
    parser.add_argument(
        "audio_file",
        help="Path to audio file to transcribe"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", "-m",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)"
    )
    
    parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)"
    )
    
    parser.add_argument(
        "--language", "-l",
        help="Language code (e.g., en, hi, sa)"
    )
    
    # Glossary configuration
    parser.add_argument(
        "--strategy", "-s",
        default="static_common",
        choices=["static_common", "chapter_guess", "empty"],
        help="Glossary strategy (default: static_common)"
    )
    
    parser.add_argument(
        "--glossary-path",
        help="Path to static glossary file"
    )
    
    parser.add_argument(
        "--glossary-dir",
        help="Directory containing chapter-specific glossaries"
    )
    
    # Processing options
    parser.add_argument(
        "--no-post-correction",
        action="store_true",
        help="Disable fuzzy post-correction"
    )
    
    parser.add_argument(
        "--context-window",
        type=int,
        default=32,
        help="Number of context words to use (default: 32)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output file path"
    )
    
    parser.add_argument(
        "--format", "-f",
        default="text",
        choices=["text", "json", "srt"],
        help="Output format (default: text)"
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except errors"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        log_level = "ERROR"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    setup_logging(level=log_level)
    
    # Validate input file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)
    
    try:
        # Initialize GT-Whisper
        gt_whisper = GTWhisper(
            model_size=args.model,
            device=args.device,
            strategy=args.strategy,
            glossary_path=args.glossary_path,
            glossary_dir=args.glossary_dir,
            enable_post_correction=not args.no_post_correction,
            context_window=args.context_window
        )
        
        # Perform transcription
        result = gt_whisper.transcribe_file(
            audio_path=str(audio_path),
            language=args.language,
            output_format=args.format,
            output_file=args.output
        )
        
        # Handle errors
        if 'error' in result:
            logger.error(f"Transcription failed: {result['error']}")
            sys.exit(1)
        
        # Print result to stdout if no output file specified
        if not args.output:
            if args.format == "json":
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(result['text'])
        
        # Print stats if verbose
        if args.verbose:
            stats = gt_whisper.get_stats()
            logger.info(f"Performance stats: {json.dumps(stats, indent=2)}")
        
    except KeyboardInterrupt:
        logger.info("Transcription interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


# Minimal REST API (optional)
def create_rest_api():
    """Create a minimal REST API for GT-Whisper."""
    try:
        from flask import Flask, request, jsonify
        from werkzeug.utils import secure_filename
        import tempfile
        import os
    except ImportError:
        logger.error("Flask is required for REST API. Install with: pip install flask")
        return None
    
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    
    # Initialize GT-Whisper instance
    gt_whisper = GTWhisper()
    
    @app.route('/transcribe', methods=['POST'])
    def transcribe():
        """Transcribe uploaded audio file."""
        try:
            # Check if file was uploaded
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            file = request.files['audio']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Get parameters
            language = request.form.get('language')
            strategy = request.form.get('strategy', 'static_common')
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                file.save(tmp_file.name)
                temp_path = tmp_file.name
            
            try:
                # Update strategy if requested
                if strategy != gt_whisper.strategy:
                    gt_whisper.glossary_selector.set_strategy(strategy)
                    gt_whisper.strategy = strategy
                
                # Transcribe
                result = gt_whisper.transcribe_file(
                    audio_path=temp_path,
                    language=language,
                    output_format="json"
                )
                
                return jsonify(result)
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"REST API error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'model': gt_whisper.model_size,
            'device': gt_whisper.actual_device,
            'strategy': gt_whisper.strategy
        })
    
    @app.route('/stats', methods=['GET'])
    def stats():
        """Get performance statistics."""
        return jsonify(gt_whisper.get_stats())
    
    return app


if __name__ == "__main__":
    main() 