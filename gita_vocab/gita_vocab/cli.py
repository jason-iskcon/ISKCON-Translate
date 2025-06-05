"""
Command-line interface for Gita Vocab scraper and glossary generator.
"""

import logging
import sys
from pathlib import Path

import click

from .scraper import GitaScraper
from .normalizer import TextNormalizer
from .generator import GlossaryGenerator


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def create_sample_content(output_file: str) -> int:
    """Create sample Bhagavad Gita content as fallback."""
    import json
    
    # Sample content with Sanskrit vocabulary
    sample_content = [
        {
            "url": "sample://bg/1/1",
            "title": "Bhagavad Gita 1.1 Sanskrit Terms",
            "text": "Dhṛtarāṣṭra Sañjaya Pāṇḍu Kurukṣetra dharma-kṣetre kuru-kṣetre samavetā yuyutsavaḥ māmakāḥ pāṇḍavāś caiva kim akurvata",
            "chapter": 1,
            "verse": "1",
            "source": "sample_sanskrit",
            "timestamp": 1640995200.0,
            "word_count": 12
        },
        {
            "url": "sample://bg/2/1",
            "title": "Bhagavad Gita 2.1 Sanskrit Terms", 
            "text": "Sañjaya Arjuna kṛpayā paripūrṇa aśru-pūrṇa īkṣaṇa viṣīdantam idaṁ vākyam uvāca Madhusūdana",
            "chapter": 2,
            "verse": "1", 
            "source": "sample_sanskrit",
            "timestamp": 1640995200.0,
            "word_count": 10
        },
        {
            "url": "sample://bg/2/47",
            "title": "Bhagavad Gita 2.47 Sanskrit Terms",
            "text": "karmaṇy evādhikāras te mā phaleṣu kadācana mā karma-phala-hetur bhūr mā te saṅgo 'stv akarmaṇi karma yoga dharma",
            "chapter": 2,
            "verse": "47",
            "source": "sample_sanskrit", 
            "timestamp": 1640995200.0,
            "word_count": 15
        },
        {
            "url": "sample://bg/4/7",
            "title": "Bhagavad Gita 4.7 Sanskrit Terms",
            "text": "yadā yadā hi dharmasya glānir bhavati bhārata abhyutthānam adharmasya tadātmānaṁ sṛjāmy aham avatāra dharma adharma",
            "chapter": 4,
            "verse": "7",
            "source": "sample_sanskrit",
            "timestamp": 1640995200.0,
            "word_count": 13
        },
        {
            "url": "sample://bg/7/1",
            "title": "Bhagavad Gita 7.1 Sanskrit Terms",
            "text": "Śrī Bhagavān uvāca mayy āsakta-manāḥ pārtha yogaṁ yuñjan mad-āśrayaḥ asaṁśayaṁ samagraṁ māṁ yathā jñāsyasi tac chṛṇu yoga bhakti jñāna",
            "chapter": 7,
            "verse": "1",
            "source": "sample_sanskrit",
            "timestamp": 1640995200.0,
            "word_count": 16
        },
        {
            "url": "sample://bg/9/22",
            "title": "Bhagavad Gita 9.22 Sanskrit Terms",
            "text": "ananyāś cintayanto māṁ ye janāḥ paryupāsate teṣāṁ nityābhiyuktānāṁ yoga-kṣemaṁ vahāmy aham bhakti yoga kṣema",
            "chapter": 9,
            "verse": "22",
            "source": "sample_sanskrit",
            "timestamp": 1640995200.0,
            "word_count": 13
        },
        {
            "url": "sample://bg/15/7",
            "title": "Bhagavad Gita 15.7 Sanskrit Terms",
            "text": "mamaivāṁśo jīva-loke jīva-bhūtaḥ sanātanaḥ manaḥ-ṣaṣṭhānīndriyāṇi prakṛti-sthāni karṣati jīva ātman prakṛti indriya manas",
            "chapter": 15,
            "verse": "7",
            "source": "sample_sanskrit",
            "timestamp": 1640995200.0,
            "word_count": 14
        },
        {
            "url": "sample://bg/18/65",
            "title": "Bhagavad Gita 18.65 Sanskrit Terms",
            "text": "man-manā bhava mad-bhakto mad-yājī māṁ namaskuru mām evaiṣyasi satyaṁ te pratijāne priyo 'si me bhakti namaskāra priya satya",
            "chapter": 18,
            "verse": "65",
            "source": "sample_sanskrit",
            "timestamp": 1640995200.0,
            "word_count": 14
        },
        {
            "url": "sample://bg/18/66",
            "title": "Bhagavad Gita 18.66 Sanskrit Terms",
            "text": "sarva-dharmān parityajya mām ekaṁ śaraṇaṁ vraja ahaṁ tvāṁ sarva-pāpebhyo mokṣayiṣyāmi mā śucaḥ śaraṇāgati mokṣa dharma pāpa",
            "chapter": 18,
            "verse": "66",
            "source": "sample_sanskrit",
            "timestamp": 1640995200.0,
            "word_count": 15
        },
        {
            "url": "sample://general/sanskrit",
            "title": "Common ISKCON Sanskrit Terms",
            "text": "Kṛṣṇa Rādhā Govinda Gopāla Vāsudeva Viṣṇu Nārāyaṇa Śiva Brahmā Gaṅgā Yamunā Vṛndāvana Mathurā Dvārakā Jagannātha Baladeva Subhadrā Caitanya Nityānanda Advaita Gadādhara Śrīvāsa guru paramparā sādhu saṅga kīrtana bhajana ārati maṅgala-ārati prasādam mahā-prasādam ekādaśī parikramā darśana sevā pūjā abhiṣeka śṛṅgāra bhoga rāga-mārga vaidhi-bhakti rāgānugā-bhakti prema bhāva rati āsakti bhāva-bhakti prema-bhakti",
            "chapter": None,
            "verse": "general",
            "source": "sample_sanskrit",
            "timestamp": 1640995200.0,
            "word_count": 45
        }
    ]
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample_content:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(sample_content)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Gita Vocab: Corpus scraper and glossary generator."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(verbose)


@cli.command()
@click.option('--output', '-o', default='raw_synonyms.jsonl')
@click.option('--delay', default=1.0, type=float)
@click.option('--use-sample', is_flag=True, help='Use sample content instead of web scraping')
@click.pass_context
def scrape(ctx: click.Context, output: str, delay: float, use_sample: bool) -> None:
    """Scrape Bhagavad Gita content from multiple sources."""
    if use_sample:
        click.echo("Using sample content...")
        count = create_sample_content(output)
        click.echo(f"Created sample content with {count} items")
        return
    
    click.echo("Starting corpus scraping...")
    
    try:
        scraper = GitaScraper(base_delay=delay)
        scraper.scrape_all_sources(output_file=output)
        
        stats = scraper.get_stats()
        total_items = stats.get('total_items', 0)
        
        # If scraping produced very little content, offer to use sample content
        if total_items < 5:
            click.echo(f"Warning: Only scraped {total_items} items. This may not be enough for good glossaries.")
            if click.confirm("Would you like to use sample content instead?"):
                count = create_sample_content(output)
                click.echo(f"Created sample content with {count} items")
                return
        
        click.echo(f"Scraping completed: {total_items} items")
        
    except Exception as e:
        click.echo(f"Scraping failed: {e}", err=True)
        if click.confirm("Would you like to use sample content as fallback?"):
            count = create_sample_content(output)
            click.echo(f"Created sample content with {count} items")
        else:
            sys.exit(1)


@cli.command()
@click.option('--output-dir', '-o', default='gita_vocab_output')
@click.option('--common-count', default=200, type=int)
@click.option('--delay', default=1.0, type=float)
@click.option('--use-sample', is_flag=True, help='Use sample content instead of web scraping')
@click.pass_context
def pipeline(ctx: click.Context, output_dir: str, common_count: int, delay: float, use_sample: bool) -> None:
    """Run the complete pipeline: scrape → normalize → generate."""
    click.echo("Starting complete Gita Vocab pipeline...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    raw_file = output_path / "raw_synonyms.jsonl"
    tokens_file = output_path / "gita_tokens.csv"
    glossaries_dir = output_path / "glossaries"
    
    try:
        # Step 1: Scraping
        click.echo("Step 1/3: Scraping content...")
        
        if use_sample:
            click.echo("Using sample content...")
            count = create_sample_content(str(raw_file))
            click.echo(f"Created sample content with {count} items")
            
            # Load the sample content into scraper for stats
            scraper = GitaScraper(base_delay=delay)
            scraper.load_content(raw_file)
        else:
            scraper = GitaScraper(base_delay=delay)
            scraper.scrape_all_sources(output_file=raw_file)
            
            stats = scraper.get_stats()
            total_items = stats.get('total_items', 0)
            
            # If scraping produced very little content, offer to use sample content
            if total_items < 5:
                click.echo(f"Warning: Only scraped {total_items} items.")
                if click.confirm("Would you like to use sample content instead?"):
                    count = create_sample_content(str(raw_file))
                    click.echo(f"Created sample content with {count} items")
                    scraper.load_content(raw_file)
        
        # Step 2: Normalization
        click.echo("Step 2/3: Normalizing content...")
        normalizer = TextNormalizer()
        normalizer.normalize_content(scraper.scraped_content)
        normalizer.export_to_csv(tokens_file)
        
        # Step 3: Glossary Generation
        click.echo("Step 3/3: Generating glossaries...")
        generator = GlossaryGenerator(normalizer)
        generator.generate_all_glossaries(
            output_dir=glossaries_dir,
            common_count=common_count
        )
        
        click.echo(f"Pipeline completed! Output in: {output_path}")
        
        # Show summary
        stats = normalizer.get_stats()
        click.echo(f"\nSummary:")
        click.echo(f"- Unique tokens: {stats.get('unique_tokens', 0)}")
        click.echo(f"- Total occurrences: {stats.get('total_occurrences', 0)}")
        click.echo(f"- Chapters covered: {len(stats.get('chapters_covered', []))}")
        
    except Exception as e:
        click.echo(f"Pipeline failed: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main() 