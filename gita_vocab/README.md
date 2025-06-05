# Gita Vocab: Corpus Scraper & Glossary Generator

A comprehensive corpus scraper and glossary generator for Bhagavad Gita and ISKCON content, designed to enhance transcription accuracy through domain-specific vocabulary.

## Features

### 🕷️ A1: Bootstrap Repo
- Python 3.11+ with Poetry dependency management
- Strict linting with Black, isort, flake8, and mypy
- Comprehensive test suite with pytest
- Pre-commit hooks for code quality

### 🌐 A2: Resilient Scraper
- Multi-source scraping from vedabase.io, asitis.com, and gitasupersite.iitk.ac.in
- Exponential backoff for rate limiting (429/503 errors)
- Configurable delays and retry logic
- Content deduplication and progress tracking
- Outputs to `raw_synonyms.jsonl`

### 🔧 A3: Normalize & Dedupe
- Preserves UTF-8 diacritics in original tokens
- Generates ASCII fallbacks using `indic_transliteration`
- Handles Sanskrit, Hindi, and English text
- Deduplicates content while tracking occurrences
- Exports to `gita_tokens.csv` with metadata

### 📚 A4: Generate Frequency Lists
- Creates `common_200.txt` with top 200 most frequent terms
- Generates chapter-specific glossaries (`by_chapter/01.txt` through `18.txt`)
- Terms sorted by frequency (descending)
- Ready-to-use glossary files for inference systems

## Installation

### Prerequisites
- Python 3.11+
- Poetry (recommended) or pip

### Using Poetry
```bash
cd gita_vocab
poetry install
poetry shell
```

### Using pip
```bash
cd gita_vocab
pip install -e .
```

## Usage

### Quick Start (Complete Pipeline)
```bash
# Run the complete pipeline
gita-scraper pipeline --output-dir ./output

# This will create:
# ./output/raw_synonyms.jsonl      - Raw scraped content
# ./output/gita_tokens.csv         - Normalized tokens
# ./output/glossaries/common_200.txt - Top 200 terms
# ./output/glossaries/by_chapter/  - Chapter-specific glossaries
```

### Individual Commands

#### Scraping Only
```bash
gita-scraper scrape --output raw_synonyms.jsonl --delay 1.5
```

#### Full Pipeline with Options
```bash
gita-scraper pipeline \
  --output-dir ./my_output \
  --common-count 300 \
  --chapter-count 150 \
  --delay 2.0
```

### Command Options

| Command | Option | Description | Default |
|---------|--------|-------------|---------|
| `scrape` | `--output, -o` | Output JSONL file | `raw_synonyms.jsonl` |
| | `--delay` | Base delay between requests (seconds) | `1.0` |
| `pipeline` | `--output-dir, -o` | Output directory | `gita_vocab_output` |
| | `--common-count` | Terms in common glossary | `200` |
| | `--chapter-count` | Terms per chapter glossary | `100` |
| | `--delay` | Base delay between requests | `1.0` |

## Output Files

### Raw Data
- **`raw_synonyms.jsonl`** - Scraped content in JSONL format
  - Each line contains: url, title, text, chapter, verse, source, timestamp

### Processed Data
- **`gita_tokens.csv`** - Normalized tokens with metadata
  - Columns: token, ascii, count, chapters, sources, contexts

### Glossaries
- **`common_200.txt`** - Top 200 most frequent terms (UTF-8)
- **`common_200_ascii.txt`** - ASCII transliteration version
- **`by_chapter/01.txt`** through **`18.txt`** - Chapter-specific terms
- **`by_chapter/*_ascii.txt`** - ASCII versions of chapter glossaries
- **`README.md`** - Statistics and usage guide

## Integration with GT-Whisper

The generated glossaries are designed to work seamlessly with the GT-Whisper enhanced transcription system:

```bash
# Use common glossary
python -m src.transcription audio.mp3 \
  --strategy static_common \
  --glossary-path ./output/glossaries/common_200.txt

# Use chapter-specific glossaries
python -m src.transcription bg_chapter_2.mp3 \
  --strategy chapter_guess \
  --glossary-dir ./output/glossaries/by_chapter/
```

## Architecture

### Data Flow
```
Web Sources → Scraper → Raw JSONL → Normalizer → CSV → Generator → Glossaries
```

### Components

1. **GitaScraper** - Resilient web scraper with exponential backoff
2. **TextNormalizer** - Unicode-aware text processing and deduplication
3. **GlossaryGenerator** - Frequency-based glossary creation
4. **CLI** - Command-line interface for all operations

### Sources
- **vedabase.io** - Comprehensive Bhagavad Gita with purports
- **asitis.com** - Alternative translations and commentaries
- **gitasupersite.iitk.ac.in** - Academic Sanskrit resources

## Development

### Setup Development Environment
```bash
git clone <repository>
cd gita_vocab
poetry install --with dev
poetry shell
pre-commit install
```

### Running Tests
```bash
pytest
pytest --cov=gita_vocab  # With coverage
```

### Code Quality
```bash
black gita_vocab/
isort gita_vocab/
flake8 gita_vocab/
mypy gita_vocab/
```

### Pre-commit Hooks
```bash
pre-commit run --all-files
```

## Configuration

### Scraper Settings
- **Base delay**: 1.0 seconds (configurable)
- **Max retries**: 5 attempts
- **Backoff factor**: 2.0x exponential
- **Max delay**: 60 seconds
- **Timeout**: 30 seconds per request

### Normalizer Settings
- **Min token length**: 2 characters
- **Max token length**: 50 characters
- **Stop words**: English + common Sanskrit particles
- **Diacritic preservation**: Full UTF-8 support

### Generator Settings
- **Common glossary**: 200 terms (configurable)
- **Chapter glossaries**: 100 terms each (configurable)
- **Sorting**: Frequency descending
- **Formats**: UTF-8 + ASCII transliteration

## Performance

### Benchmarks
- **Scraping**: ~54 chapters across 3 sources in ~5-10 minutes
- **Normalization**: ~1000 items/second
- **Generation**: <1 second for all glossaries
- **Memory usage**: <100MB peak

### Rate Limiting
- Respectful 1-2 second delays between requests
- Exponential backoff for 429/503 errors
- User-agent rotation and session management

## Troubleshooting

### Common Issues

1. **Network timeouts**
   ```bash
   gita-scraper scrape --delay 3.0  # Increase delay
   ```

2. **Memory issues with large corpora**
   ```bash
   # Process in smaller batches or increase system memory
   ```

3. **Unicode encoding errors**
   ```bash
   # Ensure UTF-8 locale: export LANG=en_US.UTF-8
   ```

4. **Missing dependencies**
   ```bash
   poetry install  # Reinstall all dependencies
   ```

### Debug Mode
```bash
gita-scraper --verbose pipeline  # Enable debug logging
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks: `pre-commit run --all-files`
5. Submit a pull request

## License

Same as ISKCON-Translate project.

## Acknowledgments

- Built for the ISKCON-Translate project
- Uses `indic_transliteration` for Sanskrit processing
- Respects website terms of service and rate limits
- Designed for spiritual content transcription enhancement 