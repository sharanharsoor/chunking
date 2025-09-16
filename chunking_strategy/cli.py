"""
Command-line interface for the chunking strategy library.

This module provides a comprehensive CLI for chunking files, running benchmarks,
configuring strategies, and other operations.
"""

import click
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chunking_strategy import (
    ChunkerOrchestrator,
    create_chunker,
    list_chunkers,
    get_chunker_metadata,
    EmbeddingModel,
    OutputFormat,
    EmbeddingConfig,
    embed_chunking_result,
    print_embedding_summary,
    export_for_vector_db,
    __version__
)
from chunking_strategy.core.registry import get_registry
from chunking_strategy.core.custom_algorithm_loader import (
    get_custom_loader,
    load_custom_algorithm,
    load_custom_algorithms_directory,
    list_custom_algorithms,
    get_custom_algorithm_info
)
from chunking_strategy.core.custom_config_integration import (
    load_config_with_custom_algorithms,
    validate_custom_config_file
)
from chunking_strategy.core.custom_validation import (
    validate_custom_algorithm_file,
    run_comprehensive_validation,
    batch_validate_algorithms
)
from chunking_strategy.utils.benchmarking import BenchmarkRunner
from chunking_strategy.utils.validation import ChunkValidator
from chunking_strategy.core.metrics import ChunkingQualityEvaluator


def safe_content_display(content, max_length=100, binary_placeholder="[Binary Content]"):
    """Safely display content, handling binary data appropriately."""
    if isinstance(content, bytes):
        # For binary content, show a placeholder with size info
        return f"{binary_placeholder} ({len(content)} bytes)"
    elif isinstance(content, str):
        # For text content, truncate if needed
        if len(content) > max_length:
            return content[:max_length] + "..."
        return content
    else:
        # For other types, convert to string safely
        str_content = str(content)
        if len(str_content) > max_length:
            return str_content[:max_length] + "..."
        return str_content


def json_encoder(obj):
    """Custom JSON encoder to handle binary content and other non-serializable objects."""
    if isinstance(obj, bytes):
        return f"[Binary Content] ({len(obj)} bytes)"
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


# Import centralized logging
from chunking_strategy.logging_config import (
    configure_logging, LogLevel, get_logger,
    enable_debug_mode, collect_debug_info, create_debug_archive,
    user_info, user_success, user_warning, user_error,
    debug_operation, performance_log, metrics_log
)

logger = get_logger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output except errors')
@click.option('--debug', is_flag=True, help='Enable debug mode with detailed logging')
@click.option('--log-level', type=click.Choice(['silent', 'minimal', 'normal', 'verbose', 'debug', 'trace']),
              help='Set specific log level')
@click.option('--log-file', type=click.Path(path_type=Path), help='Write logs to file')
@click.pass_context
def main(ctx: click.Context, verbose: bool, quiet: bool, debug: bool,
         log_level: Optional[str], log_file: Optional[Path]) -> None:
    """
    Chunking Strategy Library CLI

    A comprehensive toolkit for chunking text, documents, audio, video, and data streams.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Determine logging configuration
    if debug:
        level = LogLevel.DEBUG
        if log_file is None:
            # Default debug log file
            log_file = Path("chunking_debug.log")
    elif log_level:
        level = LogLevel(log_level.lower())
    elif quiet:
        level = LogLevel.MINIMAL
    elif verbose:
        level = LogLevel.VERBOSE
    else:
        level = LogLevel.NORMAL

    # Configure centralized logging
    configure_logging(
        level=level,
        file_output=bool(log_file),
        log_file=log_file,
        console_output=not quiet,
        collect_performance=debug,
        collect_metrics=debug
    )

    ctx.obj['verbose'] = verbose or debug
    ctx.obj['quiet'] = quiet
    ctx.obj['debug'] = debug
    ctx.obj['log_file'] = log_file


@main.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--strategy', '-s', help='Chunking strategy to use')
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path), help='Configuration file')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file for chunks')
@click.option('--format', 'output_format', type=click.Choice(['json', 'text', 'yaml']), default='json', help='Output format')
@click.option('--no-output', is_flag=True, help='Suppress all output file generation (summary only)')
@click.option('--summary-only', is_flag=True, help='Show only processing summary, no chunk content')
@click.option('--skip-large-output', type=int, help='Skip output file if result size exceeds N chunks (default: no limit)')
@click.option('--chunk-size', type=int, help='Chunk size (for fixed-size strategy)')
@click.option('--max-sentences', type=int, help='Max sentences per chunk (for sentence-based strategy)')
@click.option('--validate', is_flag=True, help='Validate chunks after creation')
@click.option('--quality-report', is_flag=True, help='Generate quality report')
@click.pass_context
def chunk(
    ctx: click.Context,
    input_file: Path,
    strategy: Optional[str],
    config: Optional[Path],
    output: Optional[Path],
    output_format: str,
    no_output: bool,
    summary_only: bool,
    skip_large_output: Optional[int],
    chunk_size: Optional[int],
    max_sentences: Optional[int],
    validate: bool,
    quality_report: bool
) -> None:
    """Chunk a file using specified strategy or configuration."""
    try:
        # Create orchestrator
        if config:
            orchestrator = ChunkerOrchestrator(config_path=config)
        else:
            orchestrator = ChunkerOrchestrator()

        # Build strategy parameters
        strategy_params = {}
        if chunk_size:
            strategy_params['chunk_size'] = chunk_size
        if max_sentences:
            strategy_params['max_sentences'] = max_sentences

        # Override chunker configuration if parameters provided
        if strategy and strategy_params:
            chunker = create_chunker(strategy, **strategy_params)
            if not chunker:
                click.echo(f"Error: Strategy '{strategy}' not found", err=True)
                sys.exit(1)
            result = chunker.chunk(input_file)
        else:
            # Use orchestrator for automatic strategy selection
            result = orchestrator.chunk_file(input_file, strategy_override=strategy)

        # Validate if requested
        if validate:
            validator = ChunkValidator()
            issues = validator.validate_result(result)
            if issues:
                click.echo(f"Validation issues found: {len(issues)}", err=True)
                for issue in issues[:5]:  # Show first 5 issues
                    click.echo(f"  - {issue}", err=True)
                if len(issues) > 5:
                    click.echo(f"  ... and {len(issues) - 5} more", err=True)

        # Generate quality report if requested
        if quality_report:
            evaluator = ChunkingQualityEvaluator()
            original_content = input_file.read_text(encoding='utf-8')
            metrics = evaluator.evaluate(result, original_content)

            click.echo("\nQuality Report:")
            click.echo(f"  Overall Score: {metrics.overall_score:.3f}")
            click.echo(f"  Chunk Count: {len(result.chunks)}")
            click.echo(f"  Avg Chunk Size: {metrics.avg_chunk_size:.1f}")
            click.echo(f"  Size Consistency: {metrics.size_consistency:.3f}")
            click.echo(f"  Coherence: {metrics.coherence:.3f}")
            click.echo(f"  Processing Time: {result.processing_time:.3f}s")

        # Output results based on user preferences
        should_save_output = False
        should_display_chunks = False

        # Determine output behavior
        if no_output:
            # User explicitly requested no output
            pass
        elif summary_only:
            # User only wants summary, no chunk content
            pass
        elif skip_large_output and len(result.chunks) > skip_large_output:
            # Too many chunks, skip output
            click.echo(f"Skipping output: {len(result.chunks)} chunks exceeds limit of {skip_large_output}")
        elif output:
            # User specified output file
            should_save_output = True
        else:
            # Default: display chunks
            should_display_chunks = True

        # Execute output actions
        if should_save_output:
            _save_chunks(result, output, output_format)
            click.echo(f"Chunks saved to {output}")
        elif should_display_chunks:
            _display_chunks(result, output_format)

        if not ctx.obj.get('quiet'):
            user_success(f"Processing complete: {len(result.chunks)} chunks generated in {result.processing_time:.3f}s")
            user_info(f"Strategy used: {result.strategy_used}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--category', help='Filter by category')
@click.option('--modality', help='Filter by modality')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'simple']), default='table', help='Output format')
@click.option('--show-details', is_flag=True, help='Show detailed information')
def list_strategies(
    category: Optional[str],
    modality: Optional[str],
    output_format: str,
    show_details: bool
) -> None:
    """List available chunking strategies."""
    try:
        # Get filter parameters
        filter_params = {}
        if category:
            filter_params['category'] = category

        # Get strategies
        strategies = list_chunkers(**filter_params)

        if not strategies:
            click.echo("No strategies found matching criteria")
            return

        if output_format == 'simple':
            for strategy in strategies:
                click.echo(strategy)

        elif output_format == 'json':
            strategy_data = []
            for strategy in strategies:
                metadata = get_chunker_metadata(strategy)
                if metadata:
                    strategy_data.append(metadata.to_dict())
            click.echo(json.dumps(strategy_data, indent=2))

        else:  # table format
            click.echo(f"{'Name':<20} {'Category':<12} {'Complexity':<10} {'Quality':<7} {'Description'}")
            click.echo("-" * 80)

            for strategy in strategies:
                metadata = get_chunker_metadata(strategy)
                if metadata:
                    description = metadata.description[:30] + "..." if len(metadata.description) > 30 else metadata.description
                    click.echo(f"{strategy:<20} {metadata.category:<12} {metadata.complexity.value:<10} {metadata.quality:<7.1f} {description}")

                    if show_details:
                        click.echo(f"  Supported formats: {', '.join(metadata.supported_formats)}")
                        click.echo(f"  Use cases: {', '.join(metadata.use_cases)}")
                        if metadata.dependencies:
                            click.echo(f"  Dependencies: {', '.join(metadata.dependencies)}")
                        click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('strategy_name')
@click.option('--test-file', type=click.Path(exists=True, path_type=Path), help='Test file to use')
@click.option('--test-text', help='Test text to use')
@click.option('--chunk-size', type=int, help='Chunk size parameter')
@click.option('--max-sentences', type=int, help='Max sentences parameter')
@click.option('--validate', is_flag=True, help='Validate output')
def test_strategy(
    strategy_name: str,
    test_file: Optional[Path],
    test_text: Optional[str],
    chunk_size: Optional[int],
    max_sentences: Optional[int],
    validate: bool
) -> None:
    """Test a specific chunking strategy."""
    try:
        # Get test content
        if test_file:
            content = test_file.read_text(encoding='utf-8')
        elif test_text:
            content = test_text
        else:
            content = "This is a test sentence. This is another test sentence. And here is a third one for good measure."

        # Build parameters
        params = {}
        if chunk_size:
            params['chunk_size'] = chunk_size
        if max_sentences:
            params['max_sentences'] = max_sentences

        # Create and test chunker
        chunker = create_chunker(strategy_name, **params)
        if not chunker:
            click.echo(f"Error: Strategy '{strategy_name}' not found", err=True)
            sys.exit(1)

        click.echo(f"Testing strategy: {strategy_name}")
        click.echo(f"Content length: {len(content)} characters")
        click.echo()

        # Run chunking
        start_time = time.time()
        result = chunker.chunk(content)
        processing_time = time.time() - start_time

        # Display results
        click.echo(f"Generated {len(result.chunks)} chunks in {processing_time:.3f}s")
        click.echo()

        for i, chunk in enumerate(result.chunks[:5]):  # Show first 5 chunks
            content_preview = safe_content_display(chunk.content)
            click.echo(f"Chunk {i+1}: {content_preview}")

        if len(result.chunks) > 5:
            click.echo(f"... and {len(result.chunks) - 5} more chunks")

        # Validate if requested
        if validate:
            validator = ChunkValidator()
            issues = validator.validate_result(result)
            if issues:
                click.echo(f"\nValidation issues: {len(issues)}")
                for issue in issues:
                    click.echo(f"  - {issue}")
            else:
                click.echo("\nValidation: PASSED")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_files', nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option('--strategies', help='Comma-separated list of strategies to benchmark')
@click.option('--output-dir', type=click.Path(path_type=Path), help='Output directory for results')
@click.option('--runs', type=int, default=3, help='Number of runs per strategy')
@click.option('--quick', is_flag=True, help='Quick benchmark with default strategies')
@click.option('--no-console', is_flag=True, help='Skip console summary output')
@click.option('--no-json', is_flag=True, help='Skip JSON output file')
@click.option('--no-csv', is_flag=True, help='Skip CSV output file')
@click.option('--custom-algorithms', type=click.Path(exists=True, path_type=Path), multiple=True,
              help='Include custom algorithms from specified files (can use multiple times)')
def benchmark(
    input_files: tuple[Path, ...],
    strategies: Optional[str],
    output_dir: Optional[Path],
    runs: int,
    quick: bool,
    no_console: bool,
    no_json: bool,
    no_csv: bool,
    custom_algorithms: tuple[Path, ...]
) -> None:
    """Benchmark chunking strategies on given files using production-ready benchmarking system."""
    try:
        from chunking_strategy.core.production_benchmark import ProductionBenchmarkRunner, ProductionBenchmarkConfig

        # Create configuration
        config = ProductionBenchmarkConfig(
            output_dir=output_dir,
            console_summary=not no_console,
            save_json=not no_json,
            save_csv=not no_csv,
            save_report=True,
            runs_per_strategy=runs,
            custom_algorithm_paths=list(custom_algorithms)
        )

        runner = ProductionBenchmarkRunner(config)

        # Prepare datasets
        datasets = {}
        for file_path in input_files:
            datasets[file_path.name] = file_path

        if not datasets:
            # Use default test content
            datasets["default_test"] = runner._get_default_test_content()

        # Prepare strategies
        if strategies:
            strategy_list = [s.strip() for s in strategies.split(',')]
        else:
            strategy_list = None  # Use defaults from runner

        if quick:
            # Quick benchmark with single dataset
            content = next(iter(datasets.values()))
            if isinstance(content, Path):
                content = content.read_text(encoding='utf-8')

            single_dataset = {"quick_test": content}
            suite = runner.run_comprehensive_benchmark(
                strategies=strategy_list[:5] if strategy_list else None,  # Limit to 5 for quick
                datasets=single_dataset,
                suite_name="quick_cli_benchmark"
            )
        else:
            # Full comprehensive benchmark
            suite = runner.run_comprehensive_benchmark(
                strategies=strategy_list,
                datasets=datasets,
                suite_name="cli_benchmark"
            )

        # Success message
        click.echo(f"✅ Benchmark completed successfully!")
        click.echo(f"📁 Results saved to: {config.output_dir}")
        click.echo(f"📊 Total strategies tested: {suite.summary_stats.get('strategies_tested', 0)}")
        click.echo(f"🎯 Success rate: {suite.summary_stats.get('success_rate', 0):.1%}")

        if custom_algorithms:
            custom_count = suite.summary_stats.get('custom_algorithm_count', 0)
            click.echo(f"🔧 Custom algorithms included: {custom_count}")

    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}", err=True)
        sys.exit(1)


@main.command("batch-directory")
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--output-dir', type=click.Path(path_type=Path), help='Output directory')
@click.option('--strategy', help='Strategy to use for all files')
@click.option('--config', type=click.Path(exists=True, path_type=Path), help='Configuration file')
@click.option('--pattern', default='*', help='File pattern to match (default: all files)')
@click.option('--recursive', is_flag=True, help='Process files recursively')
@click.option('--parallel', is_flag=True, help='Process files in parallel')
@click.option('--no-output-files', is_flag=True, help='Process files but do not save output files (summary only)')
@click.option('--skip-large-files', type=int, help='Skip saving output for files with more than N chunks')
def batch_directory(
    input_dir: Path,
    output_dir: Optional[Path],
    strategy: Optional[str],
    config: Optional[Path],
    pattern: str,
    recursive: bool,
    parallel: bool,
    no_output_files: bool,
    skip_large_files: Optional[int]
) -> None:
    """Process multiple files in batch mode."""
    try:
        # Setup output directory (only if we'll be saving files)
        if not no_output_files:
            if not output_dir:
                output_dir = input_dir / "chunks"
            output_dir.mkdir(parents=True, exist_ok=True)

        # Find files to process
        if recursive:
            files = list(input_dir.rglob(pattern))
        else:
            files = list(input_dir.glob(pattern))

        if not files:
            click.echo(f"No files found matching pattern '{pattern}'")
            return

        click.echo(f"Found {len(files)} files to process")

        # Create orchestrator
        if config:
            orchestrator = ChunkerOrchestrator(config_path=config)
        else:
            orchestrator = ChunkerOrchestrator()

        # Process files
        processed = 0
        errors = 0

        for file_path in files:
            try:
                # Print full file path and name before starting chunking process
                click.echo(f"\n📄 Processing file: {file_path}")
                click.echo(f"   File name: {file_path.name}")
                click.echo(f"   File size: {file_path.stat().st_size:,} bytes")
                click.echo(f"   Extension: {file_path.suffix}")

                # Chunk file
                result = orchestrator.chunk_file(file_path, strategy_override=strategy)

                # Save results based on output preferences
                should_save = True
                skip_reason = ""

                if no_output_files:
                    should_save = False
                    skip_reason = "(no output files requested)"
                elif skip_large_files and len(result.chunks) > skip_large_files:
                    should_save = False
                    skip_reason = f"(skipped: {len(result.chunks)} chunks > {skip_large_files} limit)"

                if should_save:
                    output_file = output_dir / f"{file_path.stem}_chunks.json"
                    _save_chunks(result, output_file, 'json')

                # Show processing summary
                chunk_info = f"{len(result.chunks)} chunks"
                if skip_reason:
                    chunk_info += f" {skip_reason}"
                click.echo(f"  → {chunk_info} in {result.processing_time:.3f}s")

                processed += 1

            except Exception as e:
                click.echo(f"Error processing {file_path.name}: {e}", err=True)
                errors += 1

        click.echo(f"\nBatch processing complete:")
        click.echo(f"  Processed: {processed}")
        click.echo(f"  Errors: {errors}")
        click.echo(f"  Output directory: {output_dir}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command("process-directory")
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), help='Output directory for chunked files')
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path), help='Configuration file')
@click.option('--extensions', help='Comma-separated list of file extensions (e.g., .txt,.md,.py)')
@click.option('--recursive/--no-recursive', default=True, help='Process files recursively (default: yes)')
@click.option('--parallel-mode', type=click.Choice(['auto', 'sequential', 'thread', 'process']), default='auto', help='Parallel processing mode')
@click.option('--workers', type=int, help='Number of workers (auto-detected if not specified)')
@click.option('--show-preview', is_flag=True, help='Show preview of chunks for each file')
@click.option('--max-preview-chunks', type=int, default=3, help='Maximum chunks to preview per file')
@click.pass_context
def process_directory(
    ctx: click.Context,
    directory: Path,
    output_dir: Optional[Path],
    config: Optional[Path],
    extensions: Optional[str],
    recursive: bool,
    parallel_mode: str,
    workers: Optional[int],
    show_preview: bool,
    max_preview_chunks: int
) -> None:
    """
    Process all files in a directory with comprehensive chunking.

    This command provides a comprehensive directory processing experience with:
    - Full file path printing before processing
    - Support for all file types with auto-strategy selection
    - Parallel processing with progress tracking
    - Integration with config files and CLI options

    Examples:
        chunking-strategy process-directory /path/to/docs
        chunking-strategy process-directory /path/to/docs --extensions .txt,.md,.py
        chunking-strategy process-directory /path/to/docs --config my_config.yaml --parallel-mode thread
    """
    try:
        from chunking_strategy.core.batch import BatchProcessor

        click.echo(f"🗂️  Processing directory: {directory}")
        click.echo(f"   Recursive: {'Yes' if recursive else 'No'}")
        click.echo(f"   Parallel mode: {parallel_mode}")
        if extensions:
            click.echo(f"   File extensions: {extensions}")
        else:
            click.echo(f"   File extensions: All supported types")
        click.echo()

        # Find files to process
        if extensions:
            ext_list = [ext.strip() for ext in extensions.split(',')]
            # Ensure extensions start with dot
            ext_list = [ext if ext.startswith('.') else f'.{ext}' for ext in ext_list]

            files = []
            for ext in ext_list:
                if recursive:
                    files.extend(list(directory.rglob(f"*{ext}")))
                else:
                    files.extend(list(directory.glob(f"*{ext}")))
        else:
            # Process all files
            if recursive:
                files = [f for f in directory.rglob("*") if f.is_file()]
            else:
                files = [f for f in directory.glob("*") if f.is_file()]

        if not files:
            click.echo("❌ No files found matching the criteria.")
            return

        click.echo(f"📋 Found {len(files)} files to process:")

        # Group files by extension for summary
        files_by_ext = {}
        for file in files:
            ext = file.suffix.lower() or 'no extension'
            if ext not in files_by_ext:
                files_by_ext[ext] = []
            files_by_ext[ext].append(file)

        for ext, file_list in files_by_ext.items():
            click.echo(f"   {ext}: {len(file_list)} files")
        click.echo()

        # Create orchestrator
        if config:
            orchestrator = ChunkerOrchestrator(config_path=config)
            click.echo(f"✅ Using configuration: {config}")
        else:
            orchestrator = ChunkerOrchestrator()
            click.echo("✅ Using default configuration")

        # Setup output directory
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            click.echo(f"📁 Output directory: {output_dir}")
        else:
            click.echo("📄 Output: Console only (no files will be saved)")

        click.echo("\n" + "="*60)
        click.echo("STARTING DIRECTORY PROCESSING")
        click.echo("="*60)

        # Process files using orchestrator batch processing
        start_time = time.time()

        try:
            results = orchestrator.chunk_files_batch(
                file_paths=files,
                parallel_mode=parallel_mode,
                max_workers=workers
            )

            end_time = time.time()

            # Show results
            click.echo("\n" + "="*60)
            click.echo("PROCESSING COMPLETED")
            click.echo("="*60)

            successful = 0
            failed = 0
            total_chunks = 0

            for i, result in enumerate(results):
                file_path = files[i]

                if result and result.chunks:
                    successful += 1
                    chunk_count = len(result.chunks)
                    total_chunks += chunk_count

                    click.echo(f"✅ {file_path}")
                    click.echo(f"   └─ {chunk_count} chunks using {result.strategy_used}")

                    # Show preview if requested
                    if show_preview and chunk_count > 0:
                        preview_count = min(max_preview_chunks, chunk_count)
                        click.echo(f"   └─ Preview ({preview_count} of {chunk_count} chunks):")
                        for j in range(preview_count):
                            chunk = result.chunks[j]
                            preview_content = safe_content_display(chunk.content, max_length=100)
                            click.echo(f"      Chunk {j+1}: {preview_content}")

                    # Save output if requested
                    if output_dir:
                        output_file = output_dir / f"{file_path.stem}_chunks.json"
                        _save_chunks(result, output_file, 'json')
                        click.echo(f"   └─ Saved: {output_file}")

                else:
                    failed += 1
                    click.echo(f"❌ {file_path}: No chunks generated")

                click.echo()

            # Final summary
            click.echo(f"📊 SUMMARY:")
            click.echo(f"   Total files: {len(files)}")
            click.echo(f"   Successful: {successful}")
            click.echo(f"   Failed: {failed}")
            click.echo(f"   Total chunks: {total_chunks}")
            click.echo(f"   Processing time: {end_time - start_time:.2f}s")
            click.echo(f"   Average: {len(files)/(end_time - start_time):.1f} files/sec")

        except Exception as processing_error:
            click.echo(f"❌ Processing failed: {processing_error}", err=True)
            if ctx.obj.get('verbose'):
                import traceback
                traceback.print_exc()
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output configuration file')
@click.option('--profile', type=click.Choice(['rag', 'summarization', 'search', 'balanced']), default='balanced', help='Configuration profile')
def init_config(output: Optional[Path], profile: str) -> None:
    """Generate a configuration file template."""
    try:
        if not output:
            output = Path("chunking_config.yaml")

        # Create configuration based on profile
        if profile == 'rag':
            config = {
                'profile_name': 'rag_optimized',
                'strategies': {
                    'primary': 'semantic_chunking',
                    'fallbacks': ['sentence_based', 'paragraph_based'],
                    'configs': {
                        'semantic_chunking': {'similarity_threshold': 0.7},
                        'sentence_based': {'max_sentences': 3},
                        'paragraph_based': {'max_paragraphs': 2}
                    }
                },
                'preprocessing': {
                    'enabled': True,
                    'normalize_whitespace': True,
                    'remove_headers_footers': False
                },
                'postprocessing': {
                    'enabled': True,
                    'merge_short_chunks': True,
                    'min_chunk_size': 100,
                    'remove_empty_chunks': True
                }
            }
        elif profile == 'summarization':
            config = {
                'profile_name': 'summarization',
                'strategies': {
                    'primary': 'paragraph_based',
                    'fallbacks': ['sentence_based', 'fixed_size'],
                    'configs': {
                        'paragraph_based': {'max_paragraphs': 5},
                        'sentence_based': {'max_sentences': 10},
                        'fixed_size': {'chunk_size': 2048}
                    }
                },
                'preprocessing': {
                    'enabled': True,
                    'normalize_whitespace': True,
                    'remove_headers_footers': True
                },
                'postprocessing': {
                    'enabled': True,
                    'merge_short_chunks': False
                }
            }
        else:  # balanced
            config = {
                'profile_name': 'balanced',
                'strategies': {
                    'primary': 'sentence_based',
                    'fallbacks': ['paragraph_based', 'fixed_size'],
                    'configs': {
                        'sentence_based': {'max_sentences': 5},
                        'paragraph_based': {'max_paragraphs': 3},
                        'fixed_size': {'chunk_size': 1024}
                    }
                },
                'preprocessing': {
                    'enabled': False
                },
                'postprocessing': {
                    'enabled': True,
                    'merge_short_chunks': True,
                    'min_chunk_size': 50
                }
            }

        # Save configuration
        import yaml
        with open(output, 'w') as f:
            yaml.safe_dump(config, f, indent=2)

        click.echo(f"Configuration file created: {output}")
        click.echo(f"Profile: {profile}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _save_chunks(result: Any, output_path: Path, format: str) -> None:
    """Save chunks to file in specified format."""
    if format == 'json':
        # Custom JSON encoder to handle enums and other non-serializable objects
        def json_encoder(obj):
            if hasattr(obj, 'value'):  # Handle enums
                return obj.value
            return str(obj)  # Fallback to string representation

        data = {
            'metadata': {
                'strategy_used': result.strategy_used,
                'total_chunks': len(result.chunks),
                'processing_time': result.processing_time,
                'source_info': result.source_info
            },
            'chunks': [
                {
                    'id': chunk.id,
                    'content': safe_content_display(chunk.content, max_length=1000, binary_placeholder="[Binary Content]"),
                    'modality': chunk.modality.value if hasattr(chunk.modality, 'value') else str(chunk.modality),
                    'metadata': chunk.metadata.__dict__ if hasattr(chunk.metadata, '__dict__') else chunk.metadata
                }
                for chunk in result.chunks
            ]
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=json_encoder)

    elif format == 'text':
        with open(output_path, 'w') as f:
            for i, chunk in enumerate(result.chunks):
                f.write(f"=== Chunk {i+1} ===\n")
                if isinstance(chunk.content, bytes):
                    f.write(f"[Binary Content] ({len(chunk.content)} bytes)\n")
                else:
                    f.write(str(chunk.content))
                f.write("\n\n")

    elif format == 'yaml':
        import yaml
        data = {
            'metadata': {
                'strategy_used': result.strategy_used,
                'total_chunks': len(result.chunks),
                'processing_time': result.processing_time
            },
            'chunks': [{'id': chunk.id, 'content': safe_content_display(chunk.content, max_length=1000, binary_placeholder="[Binary Content]")} for chunk in result.chunks]
        }
        with open(output_path, 'w') as f:
            yaml.safe_dump(data, f, indent=2)


def _display_chunks(result: Any, format: str) -> None:
    """Display chunks in specified format."""
    if format == 'json':
        # Custom JSON encoder to handle enums and other non-serializable objects
        def json_encoder(obj):
            if hasattr(obj, 'value'):  # Handle enums
                return obj.value
            return str(obj)  # Fallback to string representation

        data = {
            'chunks': [
                {
                    'id': chunk.id,
                    'content': safe_content_display(chunk.content, max_length=1000, binary_placeholder="[Binary Content]"),
                    'modality': chunk.modality.value if hasattr(chunk.modality, 'value') else str(chunk.modality),
                    'metadata': chunk.metadata.__dict__ if hasattr(chunk.metadata, '__dict__') else chunk.metadata
                }
                for chunk in result.chunks
            ],
            'metadata': {
                'total_chunks': len(result.chunks),
                'strategy_used': result.strategy_used
            }
        }
        click.echo(json.dumps(data, indent=2, default=json_encoder))

    elif format == 'text':
        for i, chunk in enumerate(result.chunks):
            click.echo(f"=== Chunk {i+1} ===")
            click.echo(safe_content_display(chunk.content, max_length=1000))
            click.echo()

    elif format == 'yaml':
        import yaml
        data = {'chunks': [{'id': chunk.id, 'content': safe_content_display(chunk.content, max_length=1000, binary_placeholder="[Binary Content]")} for chunk in result.chunks]}
        click.echo(yaml.safe_dump(data, indent=2))


@main.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed hardware information')
@click.option('--recommendations', '-r', is_flag=True, help='Show optimization recommendations')
def hardware(detailed: bool, recommendations: bool) -> None:
    """Detect and display hardware information."""
    try:
        from chunking_strategy.core.hardware import get_hardware_info

        hardware_info = get_hardware_info()

        click.echo("🖥️  Hardware Information")
        click.echo("=" * 50)

        # Basic info
        click.echo(f"Platform: {hardware_info.platform} ({hardware_info.architecture})")
        click.echo(f"Python: {hardware_info.python_version}")
        click.echo()

        # CPU info
        click.echo("🔧 CPU Information:")
        click.echo(f"  Logical cores: {hardware_info.cpu_count}")
        if hardware_info.cpu_count_physical:
            click.echo(f"  Physical cores: {hardware_info.cpu_count_physical}")
        if hardware_info.cpu_freq:
            click.echo(f"  Frequency: {hardware_info.cpu_freq:.1f} MHz")
        if hardware_info.cpu_usage is not None:
            click.echo(f"  Current usage: {hardware_info.cpu_usage:.1f}%")
        click.echo()

        # Memory info
        if hardware_info.memory_total_gb:
            click.echo("💾 Memory Information:")
            click.echo(f"  Total: {hardware_info.memory_total_gb:.1f} GB")
            if hardware_info.memory_available_gb:
                click.echo(f"  Available: {hardware_info.memory_available_gb:.1f} GB")
            if hardware_info.memory_usage_percent:
                click.echo(f"  Usage: {hardware_info.memory_usage_percent:.1f}%")
            click.echo()

        # GPU info
        if hardware_info.gpu_count > 0:
            click.echo("🎮 GPU Information:")
            click.echo(f"  Count: {hardware_info.gpu_count}")
            for i, (name, total_mem, free_mem, util) in enumerate(zip(
                hardware_info.gpu_names,
                hardware_info.gpu_memory_total,
                hardware_info.gpu_memory_free,
                hardware_info.gpu_utilization
            )):
                click.echo(f"  GPU {i}: {name}")
                if detailed:
                    click.echo(f"    Memory: {free_mem:.1f}/{total_mem:.1f} GB")
                    click.echo(f"    Utilization: {util:.1f}%")
            click.echo()
        else:
            click.echo("🎮 GPU Information: No GPUs detected")
            click.echo()

        # Recommendations
        if recommendations:
            click.echo("💡 Optimization Recommendations:")
            click.echo(f"  Recommended batch size: {hardware_info.recommended_batch_size}")
            click.echo(f"  Recommended workers: {hardware_info.recommended_workers}")
            click.echo(f"  Use GPU: {'Yes' if hardware_info.use_gpu else 'No'}")
            click.echo()

    except ImportError:
        click.echo("Hardware detection requires additional dependencies.", err=True)
        click.echo("Install with: pip install chunking-strategy[hardware]", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error detecting hardware: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('files', nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option('--strategy', '-s', default='fixed_size', help='Chunking strategy to use')
@click.option('--batch-size', '-b', type=int, help='Batch size (auto-detected if not specified)')
@click.option('--workers', '-w', type=int, help='Number of workers (auto-detected if not specified)')
@click.option('--mode', type=click.Choice(['process', 'thread', 'sequential']), default='process',
              help='Parallel processing mode')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), help='Output directory for results')
@click.option('--format', type=click.Choice(['json', 'text', 'summary']), default='summary',
              help='Output format')
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path), help='Configuration file')
@click.option('--no-gpu', is_flag=True, help='Force CPU-only processing')
@click.option('--chunk-size', type=int, help='Chunk size parameter for strategy')
@click.option('--progress/--no-progress', default=True, help='Show progress bar')
def batch(
    files: List[Path],
    strategy: str,
    batch_size: Optional[int],
    workers: Optional[int],
    mode: str,
    output_dir: Optional[Path],
    format: str,
    config: Optional[Path],
    no_gpu: bool,
    chunk_size: Optional[int],
    progress: bool
) -> None:
    """Process multiple files in batch mode with automatic optimization."""
    try:
        from chunking_strategy.core.batch import BatchProcessor
        from chunking_strategy.core.hardware import get_optimal_batch_config

        # Prepare strategy parameters
        strategy_params = {}
        if chunk_size:
            strategy_params['chunk_size'] = chunk_size

        # Set up progress callback
        def progress_callback(current: int, total: int, status: str) -> None:
            if progress:
                percent = (current / total) * 100
                click.echo(f"Progress: {current}/{total} ({percent:.1f}%) - {status}")

        # Set up error callback
        def error_callback(file_path: Path, error: Exception) -> None:
            click.echo(f"Error processing {file_path}: {error}", err=True)

        # Create batch processor
        processor = BatchProcessor(
            progress_callback=progress_callback if progress else None,
            error_callback=error_callback
        )

        # Get optimal configuration preview
        if format != 'summary':
            avg_size = sum(f.stat().st_size for f in files) / len(files) / (1024 * 1024)
            config_info = get_optimal_batch_config(
                total_files=len(files),
                avg_file_size_mb=avg_size,
                user_batch_size=batch_size,
                user_workers=workers,
                force_cpu=no_gpu
            )
            click.echo(f"🔧 Batch Configuration:")
            click.echo(f"  Files: {len(files)}")
            click.echo(f"  Batch size: {config_info['batch_size']}")
            click.echo(f"  Workers: {config_info['workers']}")
            click.echo(f"  Mode: {mode}")
            click.echo(f"  GPU: {'Disabled' if no_gpu else 'Auto'}")
            click.echo()

        # Process files
        start_time = time.time()
        result = processor.process_files(
            files=files,
            default_strategy=strategy,
            default_params=strategy_params,
            batch_size=batch_size,
            workers=workers,
            use_gpu=None if no_gpu else None,
            parallel_mode=mode
        )

        # Display results
        if format == 'json':
            output_data = {
                'total_files': result.total_files,
                'successful_files': [str(f) for f in result.successful_files],
                'failed_files': [(str(f), err) for f, err in result.failed_files],
                'total_chunks': result.total_chunks,
                'processing_time': result.total_processing_time,
                'performance': {
                    'files_per_second': result.files_per_second,
                    'chunks_per_second': result.chunks_per_second,
                    'mb_per_second': result.mb_per_second
                }
            }
            click.echo(json.dumps(output_data, indent=2))

        elif format == 'text':
            click.echo("📋 Detailed Results:")
            click.echo("=" * 50)
            for file_path, chunk_result in result.chunk_results.items():
                click.echo(f"\n📄 {file_path}:")
                click.echo(f"  Chunks: {len(chunk_result.chunks)}")
                click.echo(f"  Processing time: {chunk_result.processing_time:.3f}s")

        else:  # summary
            click.echo("\n📊 Batch Processing Summary:")
            click.echo("=" * 50)
            click.echo(f"✅ Successful files: {len(result.successful_files)}")
            click.echo(f"❌ Failed files: {len(result.failed_files)}")
            click.echo(f"📦 Total chunks: {result.total_chunks}")
            click.echo(f"⏱️  Processing time: {result.total_processing_time:.2f}s")
            click.echo(f"🚀 Performance:")
            click.echo(f"   Files/sec: {result.files_per_second:.1f}")
            click.echo(f"   Chunks/sec: {result.chunks_per_second:.1f}")
            click.echo(f"   MB/sec: {result.mb_per_second:.1f}")

            if result.failed_files:
                click.echo("\n❌ Failed files:")
                for file_path, error in result.failed_files:
                    click.echo(f"   {file_path}: {error}")

        # Save results if requested
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save individual chunk results
            for file_path, chunk_result in result.chunk_results.items():
                output_file = output_dir / f"{Path(file_path).stem}_chunks.json"
                _save_chunks(chunk_result, output_file)

            # Save summary
            summary_file = output_dir / "batch_summary.json"
            with open(summary_file, 'w') as f:
                summary = {
                    'timestamp': time.time(),
                    'total_files': result.total_files,
                    'successful_files': len(result.successful_files),
                    'failed_files': len(result.failed_files),
                    'total_chunks': result.total_chunks,
                    'processing_time': result.total_processing_time,
                    'performance': {
                        'files_per_second': result.files_per_second,
                        'chunks_per_second': result.chunks_per_second,
                        'mb_per_second': result.mb_per_second
                    }
                }
                json.dump(summary, f, indent=2)

            click.echo(f"\n💾 Results saved to: {output_dir}")

    except ImportError as e:
        click.echo(f"Batch processing requires additional dependencies: {e}", err=True)
        click.echo("Install with: pip install chunking-strategy[hardware]", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during batch processing: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--strategy', '-s', help='Chunking strategy to use before embedding')
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path), help='Configuration file')
@click.option('--model', '-m', type=click.Choice([model.value for model in EmbeddingModel]),
              default=EmbeddingModel.ALL_MINILM_L6_V2.value, help='Embedding model to use')
@click.option('--output-format', type=click.Choice([fmt.value for fmt in OutputFormat]),
              default=OutputFormat.FULL_METADATA.value, help='Output format for embeddings')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file for embeddings')
@click.option('--export-format', type=click.Choice(['dict', 'json']), default='json',
              help='Export format for vector database')
@click.option('--batch-size', type=int, default=32, help='Batch size for embedding generation')
@click.option('--normalize', is_flag=True, default=True, help='Normalize embeddings')
@click.option('--no-normalize', is_flag=True, help='Disable embedding normalization')
@click.option('--max-chunks', type=int, default=5, help='Maximum chunks to display in summary')
@click.option('--device', help='Device to use (cuda/cpu), auto-detect if not specified')
@click.pass_context
def embed(
    ctx: click.Context,
    input_file: Path,
    strategy: Optional[str],
    config: Optional[Path],
    model: str,
    output_format: str,
    output: Optional[Path],
    export_format: str,
    batch_size: int,
    normalize: bool,
    no_normalize: bool,
    max_chunks: int,
    device: Optional[str]
) -> None:
    """Generate embeddings from chunked content."""
    try:
        # Step 1: Chunk the content
        click.echo(f"📝 Chunking content from: {input_file}")

        if config:
            orchestrator = ChunkerOrchestrator(config_path=config)
        else:
            orchestrator = ChunkerOrchestrator()

        chunking_result = orchestrator.chunk_file(input_file, strategy_override=strategy)
        click.echo(f"✅ Generated {len(chunking_result.chunks)} chunks using {chunking_result.strategy_used}")

        # Step 2: Configure embedding generation
        embedding_model = EmbeddingModel(model)
        output_fmt = OutputFormat(output_format)

        # Handle normalization flags
        normalize_embeddings = normalize and not no_normalize

        embedding_config = EmbeddingConfig(
            model=embedding_model,
            output_format=output_fmt,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            device=device
        )

        # Step 3: Generate embeddings
        click.echo(f"🔮 Generating embeddings with model: {model}")
        if device:
            click.echo(f"   Using device: {device}")

        embedding_result = embed_chunking_result(chunking_result, embedding_config)

        # Step 4: Display summary
        if not ctx.obj.get('quiet'):
            print_embedding_summary(embedding_result, max_chunks=max_chunks)

        # Step 5: Export for vector database
        click.echo(f"\n💾 Exporting embeddings for vector database...")
        vector_db_data = export_for_vector_db(embedding_result, format=export_format)

        # Step 6: Save results
        if output:
            click.echo(f"💾 Saving embeddings to: {output}")

            if export_format == 'json':
                output.write_text(vector_db_data)
            else:
                import json
                output.write_text(json.dumps(vector_db_data, indent=2))

            click.echo(f"✅ Embeddings saved successfully")
        else:
            click.echo(f"📊 Generated {len(vector_db_data)} embedding entries")
            click.echo(f"   Use --output to save embeddings for vector database integration")

        # Show vector database integration hint
        if not ctx.obj.get('quiet'):
            click.echo(f"\n💡 Vector Database Integration:")
            click.echo(f"   Each embedding entry contains:")
            click.echo(f"   - id: unique chunk identifier")
            click.echo(f"   - vector: {embedding_result.embedding_dim}D embedding")
            click.echo(f"   - payload: metadata and content (if included)")
            click.echo(f"   Ready for insertion into Qdrant, Weaviate, Pinecone, ChromaDB, etc.")

    except ImportError as e:
        click.echo(f"Embedding requires additional dependencies: {e}", err=True)
        click.echo("Install with: pip install 'chunking-strategy[text,ml]'", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error generating embeddings: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('input_files', nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option('--strategy', '-s', help='Chunking strategy to use for all files')
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path), help='Configuration file')
@click.option('--model', '-m', type=click.Choice([model.value for model in EmbeddingModel]),
              default=EmbeddingModel.ALL_MINILM_L6_V2.value, help='Embedding model to use')
@click.option('--output-format', type=click.Choice([fmt.value for fmt in OutputFormat]),
              default=OutputFormat.FULL_METADATA.value, help='Output format for embeddings')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), help='Output directory for batch embeddings')
@click.option('--batch-size', type=int, default=32, help='Batch size for embedding generation')
@click.option('--normalize', is_flag=True, default=True, help='Normalize embeddings')
@click.option('--parallel', is_flag=True, help='Process files in parallel')
@click.option('--device', help='Device to use (cuda/cpu), auto-detect if not specified')
@click.pass_context
def embed_batch(
    ctx: click.Context,
    input_files: tuple,
    strategy: Optional[str],
    config: Optional[Path],
    model: str,
    output_format: str,
    output_dir: Optional[Path],
    batch_size: int,
    normalize: bool,
    parallel: bool,
    device: Optional[str]
) -> None:
    """Generate embeddings for multiple files in batch."""
    try:
        click.echo(f"📁 Batch embedding for {len(input_files)} files...")

        # Setup configuration
        if config:
            orchestrator = ChunkerOrchestrator(config_path=config)
        else:
            orchestrator = ChunkerOrchestrator()

        embedding_model = EmbeddingModel(model)
        output_fmt = OutputFormat(output_format)

        embedding_config = EmbeddingConfig(
            model=embedding_model,
            output_format=output_fmt,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            device=device
        )

        all_embeddings = []
        total_chunks = 0
        successful_files = 0
        failed_files = []

        start_time = time.time()

        for i, input_file in enumerate(input_files):
            click.echo(f"\n📝 Processing file {i+1}/{len(input_files)}: {input_file}")

            try:
                # Chunk the file
                chunking_result = orchestrator.chunk_file(input_file, strategy_override=strategy)
                click.echo(f"   Generated {len(chunking_result.chunks)} chunks")

                # Generate embeddings
                embedding_result = embed_chunking_result(chunking_result, embedding_config)
                click.echo(f"   Generated {embedding_result.total_chunks} embeddings")

                # Add file metadata
                for embedded_chunk in embedding_result.embedded_chunks:
                    if embedded_chunk.metadata:
                        embedded_chunk.metadata['source_file'] = str(input_file)
                        embedded_chunk.metadata['file_index'] = i
                    else:
                        embedded_chunk.metadata = {
                            'source_file': str(input_file),
                            'file_index': i
                        }

                all_embeddings.extend(embedding_result.embedded_chunks)
                total_chunks += embedding_result.total_chunks
                successful_files += 1

            except Exception as e:
                click.echo(f"   ❌ Failed: {e}", err=True)
                failed_files.append((str(input_file), str(e)))

        end_time = time.time()
        processing_time = end_time - start_time

        # Display summary
        click.echo(f"\n📊 Batch Embedding Summary:")
        click.echo(f"   Files processed: {successful_files}/{len(input_files)}")
        click.echo(f"   Total embeddings: {len(all_embeddings)}")
        click.echo(f"   Processing time: {processing_time:.2f}s")
        click.echo(f"   Speed: {len(all_embeddings)/processing_time:.1f} embeddings/sec")

        if failed_files:
            click.echo(f"\n❌ Failed files:")
            for file_path, error in failed_files:
                click.echo(f"   {file_path}: {error}")

        # Export batch results
        if all_embeddings:
            vector_db_data = []
            for embedded_chunk in all_embeddings:
                item = {
                    "id": embedded_chunk.chunk_id,
                    "vector": embedded_chunk.embedding,
                    "payload": embedded_chunk.metadata or {}
                }
                if embedded_chunk.content:
                    item["payload"]["content"] = embedded_chunk.content
                vector_db_data.append(item)

            # Save results
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save all embeddings
                output_file = output_dir / "batch_embeddings.json"
                output_file.write_text(json.dumps(vector_db_data, indent=2))

                # Save summary
                summary_file = output_dir / "batch_summary.json"
                summary_data = {
                    'timestamp': time.time(),
                    'total_files': len(input_files),
                    'successful_files': successful_files,
                    'failed_files': len(failed_files),
                    'total_embeddings': len(all_embeddings),
                    'processing_time': processing_time,
                    'model_used': model,
                    'embedding_dimension': all_embeddings[0].embedding_dim if all_embeddings else 0
                }
                summary_file.write_text(json.dumps(summary_data, indent=2))

                click.echo(f"\n💾 Results saved to: {output_dir}")
                click.echo(f"   Embeddings: {output_file}")
                click.echo(f"   Summary: {summary_file}")
            else:
                click.echo(f"\n💡 Use --output-dir to save batch embeddings for vector database")

    except ImportError as e:
        click.echo(f"Batch embedding requires additional dependencies: {e}", err=True)
        click.echo("Install with: pip install 'chunking-strategy[text,ml]'", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error in batch embedding: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command("list-models")
@click.pass_context
def list_models(ctx: click.Context) -> None:
    """List available embedding models."""

    click.echo("🔮 Available Embedding Models:")
    click.echo("=" * 50)

    # Group models by type
    text_models = [
        (EmbeddingModel.ALL_MINILM_L6_V2, "Fast, lightweight, 384D"),
        (EmbeddingModel.ALL_MINILM_L12_V2, "Balanced speed/quality, 384D"),
        (EmbeddingModel.ALL_MPNET_BASE_V2, "High quality, 768D"),
        (EmbeddingModel.ALL_DISTILROBERTA_V1, "RoBERTa-based, 768D"),
        (EmbeddingModel.PARAPHRASE_MULTILINGUAL_MINILM, "Multilingual, 384D")
    ]

    multimodal_models = [
        (EmbeddingModel.CLIP_VIT_B_32, "Text + Images, 512D"),
        (EmbeddingModel.CLIP_VIT_B_16, "Text + Images, 512D, higher quality"),
        (EmbeddingModel.CLIP_VIT_L_14, "Text + Images, 768D, best quality")
    ]

    click.echo("\n📝 Text-only Models (sentence-transformers):")
    for model, description in text_models:
        click.echo(f"  {model.value:<35} - {description}")

    click.echo("\n🎨 Multimodal Models (CLIP):")
    for model, description in multimodal_models:
        click.echo(f"  {model.value:<35} - {description}")

    click.echo("\n💡 Model Selection Tips:")
    click.echo("  • all-MiniLM-L6-v2: Best for speed and general use")
    click.echo("  • all-mpnet-base-v2: Best for quality on English text")
    click.echo("  • CLIP models: Use for text-image cross-modal search")
    click.echo("  • Multilingual: Use for non-English content")

    click.echo("\n📦 Dependencies:")
    click.echo("  Text models: pip install 'chunking-strategy[text]'")
    click.echo("  CLIP models: pip install 'chunking-strategy[ml]'")


# Custom Algorithm Management Commands
@main.group("custom")
@click.pass_context
def custom(ctx: click.Context) -> None:
    """Manage custom chunking algorithms."""
    pass


@custom.command("load")
@click.argument('algorithm_path', type=click.Path(exists=True, path_type=Path))
@click.option('--algorithm-name', '-n', help='Specific algorithm name to load (if file contains multiple)')
@click.option('--force-reload', '-f', is_flag=True, help='Force reload even if already loaded')
@click.option('--validate', is_flag=True, default=True, help='Validate algorithm before loading')
@click.pass_context
def load_custom_algorithm_cmd(
    ctx: click.Context,
    algorithm_path: Path,
    algorithm_name: Optional[str],
    force_reload: bool,
    validate: bool
) -> None:
    """Load a custom chunking algorithm from a Python file."""

    try:
        loader = get_custom_loader()
        loader.strict_validation = validate

        click.echo(f"📦 Loading custom algorithm from: {algorithm_path}")

        algo_info = loader.load_algorithm(
            algorithm_path,
            algorithm_name=algorithm_name,
            force_reload=force_reload
        )

        if algo_info:
            click.echo(f"✅ Successfully loaded algorithm: {algo_info.name}")
            click.echo(f"   Source: {algo_info.source_file}")
            click.echo(f"   Class: {algo_info.chunker_class.__name__}")

            if algo_info.metadata:
                click.echo(f"   Category: {algo_info.metadata.category}")
                click.echo(f"   Description: {algo_info.metadata.description}")
                if algo_info.metadata.use_cases:
                    click.echo(f"   Use cases: {', '.join(algo_info.metadata.use_cases)}")

            if algo_info.loading_warnings:
                click.echo("⚠️  Warnings:")
                for warning in algo_info.loading_warnings:
                    click.echo(f"   {warning}")

        else:
            click.echo("❌ Failed to load algorithm", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Error loading custom algorithm: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@custom.command("load-dir")
@click.argument('directory_path', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--recursive', '-r', is_flag=True, help='Search subdirectories recursively')
@click.option('--pattern', default='*.py', help='File pattern to match (default: *.py)')
@click.pass_context
def load_custom_directory_cmd(
    ctx: click.Context,
    directory_path: Path,
    recursive: bool,
    pattern: str
) -> None:
    """Load custom algorithms from a directory."""

    try:
        click.echo(f"📂 Loading custom algorithms from: {directory_path}")
        if recursive:
            click.echo("   (searching recursively)")

        algorithms = load_custom_algorithms_directory(
            directory_path,
            recursive=recursive,
            pattern=pattern
        )

        if algorithms:
            click.echo(f"✅ Successfully loaded {len(algorithms)} custom algorithms:")
            for algo_info in algorithms:
                click.echo(f"   • {algo_info.name} ({algo_info.source_file.name})")
        else:
            click.echo("⚠️  No custom algorithms found in directory")

    except Exception as e:
        click.echo(f"❌ Error loading custom algorithms: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@custom.command("list")
@click.option('--detailed', '-d', is_flag=True, help='Show detailed information')
@click.pass_context
def list_custom_algorithms_cmd(ctx: click.Context, detailed: bool) -> None:
    """List all loaded custom algorithms."""

    algorithms = list_custom_algorithms()

    if not algorithms:
        click.echo("📭 No custom algorithms currently loaded")
        click.echo("\nUse 'chunking-strategy custom load <path>' to load custom algorithms")
        return

    click.echo(f"📋 Loaded Custom Algorithms ({len(algorithms)}):")
    click.echo("=" * 50)

    for name in algorithms:
        algo_info = get_custom_algorithm_info(name)
        if not algo_info:
            continue

        click.echo(f"\n🔧 {name}")
        click.echo(f"   Source: {algo_info.source_file}")
        click.echo(f"   Class: {algo_info.chunker_class.__name__}")

        if detailed and algo_info.metadata:
            click.echo(f"   Category: {algo_info.metadata.category}")
            click.echo(f"   Description: {algo_info.metadata.description}")
            click.echo(f"   Quality: {algo_info.metadata.quality:.2f}")
            click.echo(f"   Complexity: {algo_info.metadata.complexity.value}")
            click.echo(f"   Speed: {algo_info.metadata.speed.value}")
            click.echo(f"   Memory: {algo_info.metadata.memory.value}")

            if algo_info.metadata.use_cases:
                click.echo(f"   Use cases: {', '.join(algo_info.metadata.use_cases)}")

        if detailed and (algo_info.loading_errors or algo_info.loading_warnings):
            if algo_info.loading_errors:
                click.echo("   ❌ Errors:")
                for error in algo_info.loading_errors:
                    click.echo(f"      {error}")
            if algo_info.loading_warnings:
                click.echo("   ⚠️  Warnings:")
                for warning in algo_info.loading_warnings:
                    click.echo(f"      {warning}")


@custom.command("validate-config")
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate_custom_config_cmd(ctx: click.Context, config_file: Path) -> None:
    """Validate a configuration file that uses custom algorithms."""

    try:
        click.echo(f"🔍 Validating configuration: {config_file}")

        errors = validate_custom_config_file(config_file)

        if not errors:
            click.echo("✅ Configuration is valid!")

            # Try to load the config to test custom algorithm loading
            try:
                config = load_config_with_custom_algorithms(config_file)
                click.echo("✅ Custom algorithms successfully loaded from config")

                # Show loaded algorithms
                if 'custom_algorithms' in config:
                    custom_algos = list_custom_algorithms()
                    if custom_algos:
                        click.echo(f"   Loaded algorithms: {', '.join(custom_algos)}")

            except Exception as e:
                click.echo(f"⚠️  Configuration valid but loading failed: {e}")

        else:
            click.echo("❌ Configuration validation failed:")
            for error in errors:
                click.echo(f"   • {error}")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Error validating configuration: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@custom.command("create-template")
@click.argument('output_file', type=click.Path(path_type=Path))
@click.option('--algorithm-name', default='my_custom_algorithm', help='Name for the example algorithm')
@click.option('--config-template', is_flag=True, help='Create config template instead of algorithm template')
@click.pass_context
def create_template_cmd(
    ctx: click.Context,
    output_file: Path,
    algorithm_name: str,
    config_template: bool
) -> None:
    """Create a template for custom algorithms or configurations."""

    if config_template:
        # Create configuration template
        from chunking_strategy.core.custom_config_integration import create_custom_algorithm_config_template
        template = create_custom_algorithm_config_template()

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(template, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            click.echo(f"✅ Configuration template created: {output_file}")
            click.echo("   Edit the file to specify paths to your custom algorithms")

        except Exception as e:
            click.echo(f"❌ Error creating config template: {e}", err=True)
            sys.exit(1)

    else:
        # Create algorithm template
        template_code = f'''"""
Custom Chunking Algorithm: {algorithm_name}

This is a template for creating custom chunking algorithms.
Modify this file to implement your own chunking logic.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import time

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType
)
from chunking_strategy.core.registry import (
    register_chunker,
    ComplexityLevel,
    SpeedLevel,
    MemoryUsage
)

logger = logging.getLogger(__name__)


@register_chunker(
    name="{algorithm_name}",
    category="custom",  # Change to appropriate category
    description="Custom chunking algorithm - replace with your description",
    supported_modalities=[ModalityType.TEXT],  # Add other modalities as needed
    supported_formats=["txt", "md"],  # Add supported file formats
    complexity=ComplexityLevel.MEDIUM,  # LOW, MEDIUM, HIGH, VERY_HIGH
    speed=SpeedLevel.MEDIUM,  # VERY_FAST, FAST, MEDIUM, SLOW, VERY_SLOW
    memory=MemoryUsage.LOW,  # VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
    quality=0.7,  # Quality score between 0.0 and 1.0
    use_cases=["example", "custom"],  # List your use cases
    best_for=["specific scenario"],  # What this algorithm is best for
    parameters_schema={{
        "chunk_size": {{
            "type": "integer",
            "minimum": 1,
            "default": 1000,
            "description": "Size of each chunk"
        }},
        "custom_param": {{
            "type": "string",
            "default": "default_value",
            "description": "Your custom parameter"
        }}
    }},
    default_parameters={{
        "chunk_size": 1000,
        "custom_param": "default_value"
    }}
)
class {algorithm_name.replace('_', ' ').title().replace(' ', '')}Chunker(BaseChunker):
    """
    Custom chunking algorithm.

    Replace this docstring with a description of your algorithm,
    how it works, when to use it, and any special considerations.
    """

    def __init__(self, chunk_size: int = 1000, custom_param: str = "default_value", **kwargs):
        """
        Initialize the custom chunker.

        Args:
            chunk_size: Size of each chunk
            custom_param: Your custom parameter
            **kwargs: Additional parameters
        """
        super().__init__(
            name="{algorithm_name}",
            category="custom",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )
        self.chunk_size = chunk_size
        self.custom_param = custom_param

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk the input content.

        Replace this method with your custom chunking logic.

        Args:
            content: Input content to chunk
            source_info: Optional source information
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with generated chunks
        """
        start_time = time.time()

        # Handle different input types
        if isinstance(content, Path):
            with open(content, 'r', encoding='utf-8') as f:
                text_content = f.read()
            source_name = str(content)
        elif isinstance(content, bytes):
            text_content = content.decode('utf-8')
            source_name = source_info.get('source', 'unknown') if source_info else 'unknown'
        else:
            text_content = str(content)
            source_name = source_info.get('source', 'unknown') if source_info else 'unknown'

        # Validate input
        self.validate_input(text_content, ModalityType.TEXT)

        # TODO: Replace this with your custom chunking logic
        chunks = self._create_custom_chunks(text_content, source_name)

        processing_time = time.time() - start_time

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used="{algorithm_name}",
            source_info=source_info
        )

    def _create_custom_chunks(self, content: str, source_name: str) -> List[Chunk]:
        """
        Custom chunking logic - replace with your implementation.

        This is just a simple example that splits by chunk_size.
        Replace with your actual algorithm.
        """
        chunks = []

        # Simple example: split by chunk_size characters
        for i in range(0, len(content), self.chunk_size):
            chunk_content = content[i:i + self.chunk_size]

            # Create metadata
            metadata = ChunkMetadata(
                source=source_name,
                position=f"chars {i}-{i + len(chunk_content)}",
                offset=i,
                length=len(chunk_content),
                chunker_used="{algorithm_name}",
                # Add any custom metadata fields
                extra={{"custom_param": self.custom_param}}
            )

            # Create chunk
            chunk = Chunk(
                id=f"{algorithm_name}_{{i // self.chunk_size:04d}}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=metadata
            )

            chunks.append(chunk)

        logger.info(f"Generated {{len(chunks)}} chunks using {algorithm_name}")
        return chunks


# You can define multiple algorithms in the same file
# Just add more @register_chunker decorated classes

"""
Example usage:

# Save this file as my_custom_chunker.py
# Then load it with:

from chunking_strategy.core.custom_algorithm_loader import load_custom_algorithm

# Load the algorithm
algo_info = load_custom_algorithm("my_custom_chunker.py")

# Use it
from chunking_strategy import create_chunker
chunker = create_chunker("{algorithm_name}", chunk_size=500, custom_param="my_value")
result = chunker.chunk("Your text content here...")
"""
'''

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(template_code)

            click.echo(f"✅ Algorithm template created: {output_file}")
            click.echo("   Edit the file to implement your custom chunking logic")
            click.echo("   Key areas to modify:")
            click.echo("   • Algorithm metadata in @register_chunker decorator")
            click.echo("   • Constructor parameters")
            click.echo("   • _create_custom_chunks() method - your main logic")

        except Exception as e:
            click.echo(f"❌ Error creating algorithm template: {e}", err=True)
            sys.exit(1)


@custom.command("validate")
@click.argument('algorithm_path', type=click.Path(exists=True, path_type=Path))
@click.option('--comprehensive', '-c', is_flag=True, help='Run comprehensive validation including performance tests')
@click.option('--include-performance', is_flag=True, default=True, help='Include performance tests')
@click.option('--include-quality', is_flag=True, default=True, help='Include quality assessment')
@click.option('--generate-report', is_flag=True, help='Generate detailed report file')
@click.option('--strict', is_flag=True, help='Enable strict validation mode')
@click.pass_context
def validate_custom_algorithm_cmd(
    ctx: click.Context,
    algorithm_path: Path,
    comprehensive: bool,
    include_performance: bool,
    include_quality: bool,
    generate_report: bool,
    strict: bool
) -> None:
    """Validate a custom chunking algorithm."""

    try:
        click.echo(f"🔍 Validating custom algorithm: {algorithm_path}")

        if comprehensive:
            report = run_comprehensive_validation(
                algorithm_path,
                include_performance=include_performance,
                include_quality_tests=include_quality,
                generate_report_file=generate_report
            )
        else:
            report = validate_custom_algorithm_file(
                algorithm_path,
                strict_mode=strict,
                include_performance_tests=include_performance,
                include_quality_tests=include_quality
            )

        # Display results
        status_icon = "✅" if report.is_valid else "❌"
        click.echo(f"\n{status_icon} Validation Results for '{report.algorithm_name}'")
        click.echo("=" * 50)

        # Overall score
        click.echo(f"Overall Score: {report.overall_score:.2f}/1.0")

        # Score breakdown
        if comprehensive:
            click.echo("\nScore Breakdown:")
            click.echo(f"  Interface:     {report.interface_score:.2f}")
            click.echo(f"  Functionality: {report.functionality_score:.2f}")
            click.echo(f"  Performance:   {report.performance_score:.2f}")
            click.echo(f"  Quality:       {report.quality_score:.2f}")
            click.echo(f"  Security:      {report.security_score:.2f}")
            click.echo(f"  Integration:   {report.integration_score:.2f}")
            click.echo(f"  Metadata:      {report.metadata_score:.2f}")

        # Test results
        if report.test_cases_run > 0:
            success_rate = report.test_cases_passed / report.test_cases_run * 100
            click.echo(f"\nTest Results: {report.test_cases_passed}/{report.test_cases_run} passed ({success_rate:.1f}%)")

        # Performance metrics
        if report.avg_processing_time:
            click.echo(f"\nPerformance:")
            click.echo(f"  Avg Processing Time: {report.avg_processing_time:.3f}s")
            if report.chunks_per_second:
                click.echo(f"  Chunks per Second: {report.chunks_per_second:.1f}")

        # Quality metrics
        if report.avg_chunk_size:
            click.echo(f"\nQuality Metrics:")
            click.echo(f"  Avg Chunk Size: {report.avg_chunk_size:.0f} chars")
            if report.chunk_size_variance:
                click.echo(f"  Size Variance: {report.chunk_size_variance:.0f}")

        # Issues
        all_issues = report.get_all_issues()
        if all_issues:
            click.echo(f"\nIssues ({len(all_issues)}):")

            for issue in all_issues:
                severity_colors = {
                    "critical": "red",
                    "error": "red",
                    "warning": "yellow",
                    "info": "blue"
                }
                color = severity_colors.get(issue.severity.value, "white")

                click.echo(f"  {click.style(issue.severity.value.upper(), fg=color)}: {issue.message}")
                if issue.suggestion and ctx.obj.get('verbose'):
                    click.echo(f"    💡 {issue.suggestion}")

        else:
            click.echo("\n✨ No issues found!")

        if generate_report:
            click.echo(f"\n📄 Detailed report saved to validation report file")

        # Exit with error code if validation failed
        if not report.is_valid:
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@custom.command("validate-batch")
@click.argument('directory_path', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--recursive', '-r', is_flag=True, help='Search subdirectories recursively')
@click.option('--summary-only', is_flag=True, help='Show only summary results')
@click.option('--fail-on-invalid', is_flag=True, help='Exit with error if any algorithm is invalid')
@click.pass_context
def validate_batch_cmd(
    ctx: click.Context,
    directory_path: Path,
    recursive: bool,
    summary_only: bool,
    fail_on_invalid: bool
) -> None:
    """Validate all custom algorithms in a directory."""

    try:
        click.echo(f"🔍 Batch validating algorithms in: {directory_path}")
        if recursive:
            click.echo("   (searching recursively)")

        results = batch_validate_algorithms(
            directory_path,
            recursive=recursive,
            strict_mode=False,
            include_performance_tests=True,
            include_quality_tests=True
        )

        if not results:
            click.echo("⚠️  No Python files found in directory")
            return

        # Summary statistics
        total_algorithms = len(results)
        valid_algorithms = sum(1 for r in results.values() if r.is_valid)
        invalid_algorithms = total_algorithms - valid_algorithms

        click.echo(f"\n📊 Batch Validation Summary:")
        click.echo("=" * 40)
        click.echo(f"Total algorithms: {total_algorithms}")
        click.echo(f"Valid algorithms: {valid_algorithms}")
        click.echo(f"Invalid algorithms: {invalid_algorithms}")

        if not summary_only:
            click.echo(f"\nDetailed Results:")
            click.echo("-" * 40)

            for file_path, report in results.items():
                status_icon = "✅" if report.is_valid else "❌"
                file_name = Path(file_path).name

                click.echo(f"{status_icon} {file_name:<30} Score: {report.overall_score:.2f}")

                if not report.is_valid:
                    critical_count = len(report.critical_issues)
                    error_count = len(report.errors)
                    warning_count = len(report.warnings)

                    issue_summary = []
                    if critical_count > 0:
                        issue_summary.append(f"{critical_count} critical")
                    if error_count > 0:
                        issue_summary.append(f"{error_count} errors")
                    if warning_count > 0:
                        issue_summary.append(f"{warning_count} warnings")

                    if issue_summary:
                        click.echo(f"   Issues: {', '.join(issue_summary)}")

        # Top performers
        if len(results) > 1 and not summary_only:
            sorted_results = sorted(results.items(), key=lambda x: x[1].overall_score, reverse=True)

            click.echo(f"\n🏆 Top Performers:")
            for i, (file_path, report) in enumerate(sorted_results[:3]):
                if report.is_valid:
                    file_name = Path(file_path).name
                    click.echo(f"   {i+1}. {file_name} (Score: {report.overall_score:.2f})")

        # Exit with error if requested and any invalid
        if fail_on_invalid and invalid_algorithms > 0:
            click.echo(f"\n❌ Batch validation failed: {invalid_algorithms} invalid algorithms")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Batch validation failed: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@custom.command("benchmark")
@click.argument('algorithm_path', type=click.Path(exists=True, path_type=Path))
@click.option('--compare-with', multiple=True, help='Compare with these built-in algorithms (can specify multiple)')
@click.option('--output-dir', type=click.Path(path_type=Path), help='Output directory for results')
@click.option('--test-content', help='Custom test content (uses defaults if not specified)')
@click.option('--runs', type=int, default=3, help='Number of benchmark runs per algorithm')
@click.option('--no-console', is_flag=True, help='Skip console summary output')
@click.pass_context
def benchmark_custom_algorithm_cmd(
    ctx: click.Context,
    algorithm_path: Path,
    compare_with: Tuple[str],
    output_dir: Optional[Path],
    test_content: Optional[str],
    runs: int,
    no_console: bool
) -> None:
    """Benchmark a custom algorithm against built-in strategies using production system."""

    try:
        from chunking_strategy.core.production_benchmark import run_custom_algorithm_benchmark

        click.echo(f"🏁 Benchmarking custom algorithm: {algorithm_path}")

        # Prepare comparison strategies
        compare_strategies = list(compare_with) if compare_with else None

        # Run benchmark using production system
        suite = run_custom_algorithm_benchmark(
            custom_algorithm_path=algorithm_path,
            compare_with=compare_strategies,
            test_content=test_content,
            output_dir=output_dir
        )

        # If console output is disabled, show minimal success message
        if no_console:
            click.echo(f"✅ Benchmark completed!")
            click.echo(f"📁 Results saved to: {output_dir or Path.cwd() / 'chunking_benchmarks'}")

        # Show summary stats
        stats = suite.summary_stats
        click.echo(f"📊 Summary:")
        click.echo(f"  • Strategies tested: {stats.get('strategies_tested', 0)}")
        click.echo(f"  • Success rate: {stats.get('success_rate', 0):.1%}")
        click.echo(f"  • Custom algorithms: {stats.get('custom_algorithm_count', 0)}")

        if stats.get('successful_results', 0) > 0:
            click.echo(f"  • Fastest time: {stats.get('min_processing_time', 0):.3f}s")
            click.echo(f"  • Best quality: {stats.get('max_quality_score', 0):.3f}")

    except Exception as e:
        click.echo(f"❌ Custom algorithm benchmark failed: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Debug and logging commands
@main.group("debug")
@click.pass_context
def debug_commands(ctx: click.Context) -> None:
    """Debug and logging utilities for troubleshooting issues."""
    pass


@debug_commands.command("enable")
@click.option('--log-file', type=click.Path(path_type=Path),
              help='File to write debug logs to (default: chunking_debug.log)')
@click.pass_context
def enable_debug(ctx: click.Context, log_file: Optional[Path]) -> None:
    """Enable comprehensive debug logging for troubleshooting."""
    try:
        if not log_file:
            log_file = Path("chunking_debug.log")

        debug_dir = enable_debug_mode(log_file)

        user_success("Debug mode enabled!")
        user_info(f"Debug logs will be written to: {log_file}")
        user_info(f"Debug data collection directory: {debug_dir}")
        user_info("Use 'chunking-strategy debug collect' to create a debug archive for bug reports")

    except Exception as e:
        user_error(f"Failed to enable debug mode: {e}")
        sys.exit(1)


@debug_commands.command("collect")
@click.option('--description', '-d', help='Description of the issue you\'re experiencing')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output path for debug archive')
@click.pass_context
def collect_debug(ctx: click.Context, description: Optional[str], output: Optional[Path]) -> None:
    """Collect debug information for bug reporting."""
    try:
        user_info("Collecting debug information...")

        debug_info = create_debug_archive(description or "")

        # Move to specified output location if requested
        debug_path = Path(debug_info['debug_archive'])
        if output:
            final_path = output
            if output.is_dir():
                final_path = output / debug_path.name
            debug_path.rename(final_path)
            debug_path = final_path

        user_success("Debug information collected successfully!")
        user_info(f"Debug archive created: {debug_path}")
        user_info(f"Session ID: {debug_info['session_id']}")

        click.echo("\n📋 Next steps for bug reporting:")
        for step in debug_info['next_steps']:
            click.echo(f"  {step}")

        click.echo("\n💡 Instructions:")
        for instruction in debug_info['instructions']:
            click.echo(f"  • {instruction}")

        if description:
            click.echo(f"\n📝 Issue description: {description}")

    except Exception as e:
        user_error(f"Failed to collect debug information: {e}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@debug_commands.command("archive")
@click.argument('description')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output path for debug archive')
@click.pass_context
def create_debug_archive_cmd(ctx: click.Context, description: str, output: Optional[Path]) -> None:
    """Create debug archive with description - shorthand for collect with description."""
    # Enable debug mode temporarily
    original_debug = ctx.obj.get('debug', False)
    if not original_debug:
        user_info("Temporarily enabling debug mode to collect information...")
        enable_debug_mode()

    try:
        ctx.invoke(collect_debug, description=description, output=output)
    finally:
        if not original_debug:
            # Could restore original logging level here if needed
            pass


@debug_commands.command("test-logging")
@click.pass_context
def test_logging(ctx: click.Context) -> None:
    """Test all logging levels and functionality."""
    user_info("Testing logging functionality...")

    # Test different user message types
    user_info("This is an info message for users")
    user_success("This is a success message for users")
    user_warning("This is a warning message for users")

    # Test debug logging
    debug_operation("test_operation", {
        "parameter1": "value1",
        "parameter2": 42,
        "nested": {"key": "value"}
    })

    # Test performance logging
    import time
    start_time = time.time()
    time.sleep(0.001)  # Simulate work
    performance_log("test_operation", time.time() - start_time,
                   chunks_processed=10, file_size=1024)

    # Test metrics logging
    metrics_log({
        "chunks_created": 5,
        "avg_chunk_size": 512,
        "processing_time": 0.123,
        "quality_score": 0.85
    })

    user_success("Logging test completed!")
    user_info("Check your configured log outputs to see all message types")


if __name__ == '__main__':
    main()
