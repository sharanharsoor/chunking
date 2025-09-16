"""
Production-ready benchmarking system for chunking strategies.

This module provides a robust benchmarking framework designed for pip-installed
environments with flexible output configuration, custom algorithm support,
and comprehensive error handling.

Key Features:
- Configurable output directories (working directory by default)
- Seamless custom algorithm integration
- Multiple output formats (JSON, CSV, console)
- Robust error handling and directory management
- No hardcoded paths or development assumptions
"""

import json
import csv
import logging
import statistics
import time
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import tempfile

from chunking_strategy.core.base import ChunkingResult, BaseChunker
from chunking_strategy.core.metrics import ChunkingQualityEvaluator, QualityMetrics
from chunking_strategy.core.registry import create_chunker, list_chunkers

logger = logging.getLogger(__name__)


@dataclass
class ProductionBenchmarkConfig:
    """Configuration for production benchmarking."""

    output_dir: Optional[Path] = None
    console_summary: bool = True
    save_json: bool = True
    save_csv: bool = True
    save_report: bool = True
    runs_per_strategy: int = 3
    include_system_info: bool = True
    custom_algorithm_paths: List[Path] = None

    def __post_init__(self):
        """Set default values and validate configuration."""
        if self.output_dir is None:
            # Default to current working directory + chunking_benchmarks
            self.output_dir = Path.cwd() / "chunking_benchmarks"

        if self.custom_algorithm_paths is None:
            self.custom_algorithm_paths = []

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProductionBenchmarkConfig':
        """Create config from dictionary."""
        # Convert string paths to Path objects
        if 'output_dir' in config_dict and config_dict['output_dir']:
            config_dict['output_dir'] = Path(config_dict['output_dir'])

        if 'custom_algorithm_paths' in config_dict:
            config_dict['custom_algorithm_paths'] = [
                Path(p) for p in config_dict['custom_algorithm_paths']
            ]

        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> 'ProductionBenchmarkConfig':
        """Create config from environment variables."""
        config = {}

        # Read environment variables
        if os.getenv('CHUNKING_BENCHMARK_OUTPUT_DIR'):
            config['output_dir'] = Path(os.getenv('CHUNKING_BENCHMARK_OUTPUT_DIR'))

        if os.getenv('CHUNKING_BENCHMARK_RUNS'):
            config['runs_per_strategy'] = int(os.getenv('CHUNKING_BENCHMARK_RUNS'))

        if os.getenv('CHUNKING_BENCHMARK_CONSOLE') in ['false', 'False', '0']:
            config['console_summary'] = False

        if os.getenv('CHUNKING_BENCHMARK_JSON') in ['false', 'False', '0']:
            config['save_json'] = False

        return cls.from_dict(config)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    strategy_name: str
    dataset_name: str
    content_size: int
    processing_time: float
    memory_usage: Optional[float]
    chunk_count: int
    avg_chunk_size: float
    quality_metrics: Dict[str, float]
    parameters: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    is_custom_algorithm: bool = False
    custom_algorithm_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    description: str
    timestamp: float
    config: ProductionBenchmarkConfig
    system_info: Dict[str, Any]
    results: List[BenchmarkResult]
    summary_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "config": {
                "output_dir": str(self.config.output_dir),
                "runs_per_strategy": self.config.runs_per_strategy,
                "custom_algorithm_count": len(self.config.custom_algorithm_paths)
            },
            "system_info": self.system_info,
            "results": [result.to_dict() for result in self.results],
            "summary_stats": self.summary_stats
        }


class ProductionBenchmarkRunner:
    """
    Production-ready benchmark runner for chunking strategies.

    Designed for pip-installed environments with no hardcoded paths,
    robust error handling, and seamless custom algorithm integration.
    """

    def __init__(self, config: Optional[ProductionBenchmarkConfig] = None):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration (uses defaults if None)
        """
        self.config = config or ProductionBenchmarkConfig()
        self.quality_evaluator = ChunkingQualityEvaluator()
        self.logger = logging.getLogger(f"{__name__}.ProductionBenchmarkRunner")

        # Ensure output directory exists
        self._setup_output_directory()

        # Load custom algorithms if specified
        self._load_custom_algorithms()

    def _setup_output_directory(self) -> None:
        """Setup and validate output directory."""
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

            # Test write permissions
            test_file = self.config.output_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()

            self.logger.info(f"Output directory ready: {self.config.output_dir}")

        except PermissionError:
            # Fallback to temp directory
            fallback_dir = Path(tempfile.gettempdir()) / "chunking_benchmarks"
            self.logger.warning(
                f"Cannot write to {self.config.output_dir}, using fallback: {fallback_dir}"
            )
            self.config.output_dir = fallback_dir
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            self.logger.error(f"Failed to setup output directory: {e}")
            raise RuntimeError(f"Cannot setup benchmark output directory: {e}")

    def _load_custom_algorithms(self) -> None:
        """Load custom algorithms from specified paths."""
        self.custom_algorithms = {}

        for algo_path in self.config.custom_algorithm_paths:
            try:
                from chunking_strategy.core.custom_algorithm_loader import load_custom_algorithm

                algo_info = load_custom_algorithm(algo_path)
                if algo_info:
                    self.custom_algorithms[algo_info.name] = {
                        'info': algo_info,
                        'path': algo_path
                    }
                    self.logger.info(f"Loaded custom algorithm: {algo_info.name}")

            except Exception as e:
                self.logger.warning(f"Failed to load custom algorithm from {algo_path}: {e}")

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        try:
            import platform
            import sys

            info = {
                "timestamp": datetime.now().isoformat(),
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": sys.version,
                "working_directory": str(Path.cwd()),
                "output_directory": str(self.config.output_dir)
            }

            # Try to get additional system info
            try:
                import psutil
                info.update({
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3)
                })
            except ImportError:
                info["psutil_available"] = False

            return info

        except Exception as e:
            self.logger.warning(f"Failed to get system info: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def benchmark_strategy(
        self,
        strategy_name: str,
        content: Union[str, bytes, Path],
        dataset_name: str = "unknown",
        parameters: Optional[Dict[str, Any]] = None,
        runs: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Benchmark a single strategy on given content.

        Args:
            strategy_name: Name of strategy to benchmark
            content: Content to chunk (string, bytes, or file path)
            dataset_name: Name of dataset being tested
            parameters: Strategy parameters
            runs: Number of runs to average (uses config default if None)

        Returns:
            Benchmark result
        """
        runs = runs or self.config.runs_per_strategy
        parameters = parameters or {}

        # Handle content loading
        if isinstance(content, Path):
            try:
                content_str = content.read_text(encoding='utf-8')
                dataset_name = content.name
            except Exception as e:
                return self._create_error_result(
                    strategy_name, dataset_name, 0, f"Failed to read file: {e}"
                )
        elif isinstance(content, bytes):
            try:
                content_str = content.decode('utf-8')
            except Exception as e:
                return self._create_error_result(
                    strategy_name, dataset_name, len(content), f"Failed to decode bytes: {e}"
                )
        else:
            content_str = str(content)

        # Check if this is a custom algorithm
        is_custom = strategy_name in self.custom_algorithms
        custom_path = None
        if is_custom:
            custom_path = str(self.custom_algorithms[strategy_name]['path'])

        try:
            # Create chunker
            chunker = create_chunker(strategy_name, **parameters)
            if not chunker:
                return self._create_error_result(
                    strategy_name, dataset_name, len(content_str),
                    f"Strategy {strategy_name} not found",
                    is_custom, custom_path
                )

            # Run multiple times and collect metrics
            times = []
            memory_usages = []
            results = []

            for run in range(runs):
                try:
                    memory_before = self._get_memory_usage()
                    start_time = time.time()

                    result = chunker.chunk(content_str)

                    processing_time = time.time() - start_time
                    memory_after = self._get_memory_usage()

                    times.append(processing_time)
                    if memory_before and memory_after:
                        memory_usages.append(memory_after - memory_before)
                    results.append(result)

                except Exception as e:
                    self.logger.warning(f"Run {run+1} failed for {strategy_name}: {e}")

            if not times:
                return self._create_error_result(
                    strategy_name, dataset_name, len(content_str),
                    "All benchmark runs failed", is_custom, custom_path
                )

            # Use best result for quality metrics
            best_result = max(results, key=lambda r: len(r.chunks) if r.chunks else 0)

            # Calculate quality metrics
            try:
                quality_metrics = self.quality_evaluator.evaluate(best_result, content_str)
                quality_dict = quality_metrics.to_dict()
            except Exception as e:
                self.logger.warning(f"Quality evaluation failed for {strategy_name}: {e}")
                quality_dict = {"overall_score": 0.0}

            return BenchmarkResult(
                strategy_name=strategy_name,
                dataset_name=dataset_name,
                content_size=len(content_str),
                processing_time=statistics.mean(times),
                memory_usage=statistics.mean(memory_usages) if memory_usages else None,
                chunk_count=len(best_result.chunks) if best_result.chunks else 0,
                avg_chunk_size=best_result.avg_chunk_size or 0.0,
                quality_metrics=quality_dict,
                parameters=parameters,
                success=True,
                is_custom_algorithm=is_custom,
                custom_algorithm_path=custom_path
            )

        except Exception as e:
            self.logger.error(f"Benchmark failed for {strategy_name}: {e}")
            return self._create_error_result(
                strategy_name, dataset_name, len(content_str), str(e), is_custom, custom_path
            )

    def benchmark_multiple_strategies(
        self,
        strategies: List[Union[str, tuple[str, Dict[str, Any]]]],
        content: Union[str, bytes, Path],
        dataset_name: str = "unknown"
    ) -> List[BenchmarkResult]:
        """
        Benchmark multiple strategies on the same content.

        Args:
            strategies: List of strategy names or (name, parameters) tuples
            content: Content to chunk
            dataset_name: Name of dataset

        Returns:
            List of benchmark results
        """
        results = []

        for strategy_spec in strategies:
            if isinstance(strategy_spec, str):
                strategy_name = strategy_spec
                parameters = {}
            else:
                strategy_name, parameters = strategy_spec

            self.logger.info(f"Benchmarking {strategy_name} on {dataset_name}")

            result = self.benchmark_strategy(
                strategy_name=strategy_name,
                content=content,
                dataset_name=dataset_name,
                parameters=parameters
            )

            results.append(result)

        return results

    def run_comprehensive_benchmark(
        self,
        strategies: Optional[List[str]] = None,
        datasets: Optional[Dict[str, Union[str, Path]]] = None,
        suite_name: str = "comprehensive_benchmark"
    ) -> BenchmarkSuite:
        """
        Run a comprehensive benchmark suite.

        Args:
            strategies: List of strategies to test (uses defaults if None)
            datasets: Dictionary mapping dataset names to content (uses defaults if None)
            suite_name: Name for the benchmark suite

        Returns:
            Complete benchmark suite results
        """
        start_time = time.time()

        # Default strategies
        if strategies is None:
            strategies = ["fixed_size", "sentence_based", "paragraph"]
            # Add available custom algorithms
            strategies.extend(self.custom_algorithms.keys())

        # Default datasets
        if datasets is None:
            datasets = {
                "sample_text": self._get_default_test_content(),
                "short_text": "This is a short test. It has few sentences.",
                "long_text": self._get_default_test_content() * 10
            }

        self.logger.info(f"Starting comprehensive benchmark: {suite_name}")
        self.logger.info(f"Strategies: {strategies}")
        self.logger.info(f"Datasets: {list(datasets.keys())}")

        all_results = []

        # Run benchmarks
        for dataset_name, content in datasets.items():
            dataset_results = self.benchmark_multiple_strategies(
                strategies=strategies,
                content=content,
                dataset_name=dataset_name
            )
            all_results.extend(dataset_results)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(all_results)

        # Create benchmark suite
        suite = BenchmarkSuite(
            name=suite_name,
            description=f"Comprehensive benchmark of {len(strategies)} strategies on {len(datasets)} datasets",
            timestamp=start_time,
            config=self.config,
            system_info=self.get_system_info(),
            results=all_results,
            summary_stats=summary_stats
        )

        # Save results
        self._save_benchmark_suite(suite)

        # Display console summary if enabled
        if self.config.console_summary:
            self._display_console_summary(suite)

        total_time = time.time() - start_time
        self.logger.info(f"Comprehensive benchmark completed in {total_time:.2f}s")

        return suite

    def _save_benchmark_suite(self, suite: BenchmarkSuite) -> None:
        """Save benchmark suite in multiple formats."""
        timestamp_str = datetime.fromtimestamp(suite.timestamp).strftime("%Y%m%d_%H%M%S")
        base_name = f"{suite.name}_{timestamp_str}"

        try:
            # Save JSON
            if self.config.save_json:
                json_path = self.config.output_dir / f"{base_name}.json"
                with open(json_path, 'w') as f:
                    json.dump(suite.to_dict(), f, indent=2, default=str)
                self.logger.info(f"Saved JSON results: {json_path}")

            # Save CSV
            if self.config.save_csv:
                csv_path = self.config.output_dir / f"{base_name}.csv"
                self._save_csv_results(suite.results, csv_path)
                self.logger.info(f"Saved CSV results: {csv_path}")

            # Save human-readable report
            if self.config.save_report:
                report_path = self.config.output_dir / f"{base_name}_report.txt"
                with open(report_path, 'w') as f:
                    f.write(self._generate_text_report(suite))
                self.logger.info(f"Saved text report: {report_path}")

        except Exception as e:
            self.logger.error(f"Failed to save benchmark results: {e}")

    def _save_csv_results(self, results: List[BenchmarkResult], csv_path: Path) -> None:
        """Save results to CSV format."""
        if not results:
            return

        fieldnames = [
            'strategy_name', 'dataset_name', 'content_size', 'processing_time',
            'memory_usage', 'chunk_count', 'avg_chunk_size', 'success',
            'is_custom_algorithm', 'overall_quality_score'
        ]

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    'strategy_name': result.strategy_name,
                    'dataset_name': result.dataset_name,
                    'content_size': result.content_size,
                    'processing_time': result.processing_time,
                    'memory_usage': result.memory_usage,
                    'chunk_count': result.chunk_count,
                    'avg_chunk_size': result.avg_chunk_size,
                    'success': result.success,
                    'is_custom_algorithm': result.is_custom_algorithm,
                    'overall_quality_score': result.quality_metrics.get('overall_score', 0.0)
                }
                writer.writerow(row)

    def _display_console_summary(self, suite: BenchmarkSuite) -> None:
        """Display benchmark results summary to console."""
        print("\n" + "="*80)
        print(f"🏁 BENCHMARK RESULTS: {suite.name}")
        print("="*80)

        print(f"⏰ Completed: {datetime.fromtimestamp(suite.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Results saved to: {self.config.output_dir}")
        print(f"📊 Total results: {len(suite.results)}")

        # Success rate
        successful = [r for r in suite.results if r.success]
        success_rate = len(successful) / len(suite.results) if suite.results else 0
        print(f"✅ Success rate: {success_rate:.1%} ({len(successful)}/{len(suite.results)})")

        if not successful:
            print("❌ No successful benchmark runs")
            return

        print("\n" + "-"*80)
        print("🚀 TOP PERFORMERS")
        print("-"*80)

        # Fastest strategy
        fastest = min(successful, key=lambda r: r.processing_time)
        print(f"⚡ Fastest: {fastest.strategy_name} ({fastest.processing_time:.3f}s)")
        if fastest.is_custom_algorithm:
            print(f"   🔧 Custom algorithm from: {fastest.custom_algorithm_path}")

        # Best quality
        best_quality = max(successful, key=lambda r: r.quality_metrics.get('overall_score', 0))
        quality_score = best_quality.quality_metrics.get('overall_score', 0)
        print(f"🏆 Best quality: {best_quality.strategy_name} (score: {quality_score:.3f})")
        if best_quality.is_custom_algorithm:
            print(f"   🔧 Custom algorithm from: {best_quality.custom_algorithm_path}")

        # Strategy comparison
        print(f"\n{'-'*80}")
        print("📈 STRATEGY COMPARISON")
        print(f"{'-'*80}")

        # Group by strategy for comparison
        strategy_groups = {}
        for result in successful:
            if result.strategy_name not in strategy_groups:
                strategy_groups[result.strategy_name] = []
            strategy_groups[result.strategy_name].append(result)

        print(f"{'Strategy':<25} {'Avg Time (s)':<12} {'Quality':<10} {'Custom':<8}")
        print("-" * 65)

        for strategy_name, results in sorted(strategy_groups.items()):
            avg_time = statistics.mean(r.processing_time for r in results)
            avg_quality = statistics.mean(r.quality_metrics.get('overall_score', 0) for r in results)
            is_custom = "🔧" if results[0].is_custom_algorithm else ""

            print(f"{strategy_name:<25} {avg_time:<12.3f} {avg_quality:<10.3f} {is_custom:<8}")

        # Custom algorithms section
        custom_results = [r for r in successful if r.is_custom_algorithm]
        if custom_results:
            print(f"\n{'-'*80}")
            print("🔧 CUSTOM ALGORITHMS")
            print(f"{'-'*80}")

            for result in custom_results:
                print(f"Algorithm: {result.strategy_name}")
                print(f"  📁 Path: {result.custom_algorithm_path}")
                print(f"  ⚡ Performance: {result.processing_time:.3f}s")
                print(f"  🏆 Quality: {result.quality_metrics.get('overall_score', 0):.3f}")
                print()

        print("="*80)
        print(f"📁 Detailed results saved to: {self.config.output_dir}")
        print("="*80)

    def _generate_text_report(self, suite: BenchmarkSuite) -> str:
        """Generate a comprehensive text report."""
        lines = []
        lines.append("CHUNKING STRATEGY BENCHMARK REPORT")
        lines.append("=" * 50)
        lines.append(f"Suite: {suite.name}")
        lines.append(f"Description: {suite.description}")
        lines.append(f"Timestamp: {datetime.fromtimestamp(suite.timestamp).isoformat()}")
        lines.append(f"Total Results: {len(suite.results)}")
        lines.append("")

        # System info
        lines.append("SYSTEM INFORMATION")
        lines.append("-" * 30)
        for key, value in suite.system_info.items():
            lines.append(f"{key}: {value}")
        lines.append("")

        # Configuration
        lines.append("CONFIGURATION")
        lines.append("-" * 30)
        lines.append(f"Output Directory: {suite.config.output_dir}")
        lines.append(f"Runs Per Strategy: {suite.config.runs_per_strategy}")
        lines.append(f"Custom Algorithms: {len(suite.config.custom_algorithm_paths)}")
        lines.append("")

        # Results summary
        successful = [r for r in suite.results if r.success]
        if successful:
            lines.append("RESULTS SUMMARY")
            lines.append("-" * 30)
            lines.append(f"Success Rate: {len(successful)/len(suite.results):.1%}")

            # Performance metrics
            times = [r.processing_time for r in successful]
            qualities = [r.quality_metrics.get('overall_score', 0) for r in successful]

            lines.append(f"Avg Processing Time: {statistics.mean(times):.3f}s")
            lines.append(f"Avg Quality Score: {statistics.mean(qualities):.3f}")
            lines.append("")

            # Detailed results
            lines.append("DETAILED RESULTS")
            lines.append("-" * 30)

            for result in suite.results:
                lines.append(f"Strategy: {result.strategy_name}")
                lines.append(f"  Dataset: {result.dataset_name}")
                lines.append(f"  Success: {result.success}")
                if result.success:
                    lines.append(f"  Processing Time: {result.processing_time:.3f}s")
                    lines.append(f"  Chunk Count: {result.chunk_count}")
                    lines.append(f"  Quality Score: {result.quality_metrics.get('overall_score', 0):.3f}")
                    lines.append(f"  Custom Algorithm: {result.is_custom_algorithm}")
                    if result.is_custom_algorithm:
                        lines.append(f"  Algorithm Path: {result.custom_algorithm_path}")
                else:
                    lines.append(f"  Error: {result.error_message}")
                lines.append("")

        return "\n".join(lines)

    def _create_error_result(
        self,
        strategy_name: str,
        dataset_name: str,
        content_size: int,
        error_message: str,
        is_custom: bool = False,
        custom_path: Optional[str] = None
    ) -> BenchmarkResult:
        """Create a benchmark result for a failed run."""
        return BenchmarkResult(
            strategy_name=strategy_name,
            dataset_name=dataset_name,
            content_size=content_size,
            processing_time=0.0,
            memory_usage=None,
            chunk_count=0,
            avg_chunk_size=0.0,
            quality_metrics={},
            parameters={},
            success=False,
            error_message=error_message,
            is_custom_algorithm=is_custom,
            custom_algorithm_path=custom_path
        )

    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return None
        except Exception:
            return None

    def _get_default_test_content(self) -> str:
        """Get default test content for benchmarking."""
        return """
        Machine learning is a subset of artificial intelligence that enables computers to learn and make
        decisions from data without being explicitly programmed for every scenario. It involves algorithms
        that can identify patterns in data and use those patterns to make predictions or decisions about
        new, unseen data.

        There are three main types of machine learning: supervised learning, where the algorithm learns
        from labeled examples; unsupervised learning, where it finds hidden patterns in data without labels;
        and reinforcement learning, where it learns through interaction with an environment and receives
        rewards or penalties.

        The applications of machine learning are vast and growing. From recommendation systems on streaming
        platforms to autonomous vehicles, from medical diagnosis to financial fraud detection, machine
        learning is transforming industries and creating new possibilities for innovation and efficiency.

        Deep learning, a subset of machine learning, uses artificial neural networks with multiple layers
        to model and understand complex patterns in data. This approach has revolutionized fields such as
        computer vision, natural language processing, and speech recognition, enabling breakthroughs like
        image classification, language translation, and voice assistants.
        """.strip()

    def _calculate_summary_stats(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics for benchmark results."""
        if not results:
            return {}

        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {
                "success_rate": 0.0,
                "total_results": len(results),
                "custom_algorithm_count": sum(1 for r in results if r.is_custom_algorithm)
            }

        processing_times = [r.processing_time for r in successful_results]
        chunk_counts = [r.chunk_count for r in successful_results]
        quality_scores = [r.quality_metrics.get('overall_score', 0.0) for r in successful_results]

        return {
            "success_rate": len(successful_results) / len(results),
            "total_results": len(results),
            "successful_results": len(successful_results),
            "custom_algorithm_count": sum(1 for r in results if r.is_custom_algorithm),
            "avg_processing_time": statistics.mean(processing_times),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "avg_chunk_count": statistics.mean(chunk_counts),
            "avg_quality_score": statistics.mean(quality_scores),
            "max_quality_score": max(quality_scores),
            "strategies_tested": len(set(r.strategy_name for r in results)),
            "datasets_tested": len(set(r.dataset_name for r in results))
        }


# Convenience functions for easy usage
def run_quick_benchmark(
    content: str,
    strategies: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> BenchmarkSuite:
    """
    Run a quick benchmark on given content.

    Args:
        content: Text content to benchmark
        strategies: List of strategies to test (uses defaults if None)
        output_dir: Output directory (uses current directory if None)

    Returns:
        Benchmark suite results
    """
    config = ProductionBenchmarkConfig(output_dir=output_dir)
    runner = ProductionBenchmarkRunner(config)

    datasets = {"quick_test": content}

    return runner.run_comprehensive_benchmark(
        strategies=strategies,
        datasets=datasets,
        suite_name="quick_benchmark"
    )


def run_custom_algorithm_benchmark(
    custom_algorithm_path: Path,
    compare_with: Optional[List[str]] = None,
    test_content: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> BenchmarkSuite:
    """
    Benchmark a custom algorithm against built-in strategies.

    Args:
        custom_algorithm_path: Path to custom algorithm file
        compare_with: List of built-in strategies to compare with
        test_content: Test content (uses default if None)
        output_dir: Output directory (uses current directory if None)

    Returns:
        Benchmark suite results
    """
    config = ProductionBenchmarkConfig(
        output_dir=output_dir,
        custom_algorithm_paths=[custom_algorithm_path]
    )
    runner = ProductionBenchmarkRunner(config)

    # Get custom algorithm name
    try:
        from chunking_strategy.core.custom_algorithm_loader import load_custom_algorithm
        algo_info = load_custom_algorithm(custom_algorithm_path)
    except ImportError as e:
        raise ImportError(f"Could not import custom algorithm loader: {e}")

    if not algo_info:
        raise ValueError(f"Failed to load custom algorithm from {custom_algorithm_path}")

    strategies = [algo_info.name]
    if compare_with:
        strategies.extend(compare_with)
    else:
        strategies.extend(["fixed_size", "sentence_based", "paragraph"])

    datasets = {"test_content": test_content or runner._get_default_test_content()}

    return runner.run_comprehensive_benchmark(
        strategies=strategies,
        datasets=datasets,
        suite_name=f"custom_benchmark_{algo_info.name}"
    )
