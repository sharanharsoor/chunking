"""
Benchmarking utilities for chunking strategies.

This module provides benchmarking capabilities to measure and compare
the performance of different chunking strategies.
"""

import logging
import time
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict

from chunking_strategy.core.base import ChunkingResult, BaseChunker
from chunking_strategy.core.metrics import ChunkingQualityEvaluator, QualityMetrics
from chunking_strategy.core.registry import create_chunker, list_chunkers

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    description: str
    timestamp: float
    results: List[BenchmarkResult]
    summary_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "results": [result.to_dict() for result in self.results],
            "summary_stats": self.summary_stats
        }


class BenchmarkRunner:
    """
    Runner for benchmarking chunking strategies.

    Provides comprehensive benchmarking capabilities including performance
    measurement, quality assessment, and comparative analysis.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize benchmark runner.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("benchmarks/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.quality_evaluator = ChunkingQualityEvaluator()
        self.logger = logging.getLogger(f"{__name__}.BenchmarkRunner")

    def benchmark_strategy(
        self,
        strategy_name: str,
        content: Union[str, bytes],
        dataset_name: str = "unknown",
        parameters: Optional[Dict[str, Any]] = None,
        runs: int = 1
    ) -> BenchmarkResult:
        """
        Benchmark a single strategy on given content.

        Args:
            strategy_name: Name of strategy to benchmark
            content: Content to chunk
            dataset_name: Name of dataset being tested
            parameters: Strategy parameters
            runs: Number of runs to average

        Returns:
            Benchmark result
        """
        parameters = parameters or {}

        try:
            # Create chunker
            chunker = create_chunker(strategy_name, **parameters)
            if not chunker:
                return BenchmarkResult(
                    strategy_name=strategy_name,
                    dataset_name=dataset_name,
                    content_size=len(content),
                    processing_time=0.0,
                    memory_usage=None,
                    chunk_count=0,
                    avg_chunk_size=0.0,
                    quality_metrics={},
                    parameters=parameters,
                    success=False,
                    error_message=f"Strategy {strategy_name} not found"
                )

            # Run multiple times and average
            times = []
            results = []

            for run in range(runs):
                start_time = time.time()
                memory_before = self._get_memory_usage()

                try:
                    result = chunker.chunk(content)
                    processing_time = time.time() - start_time
                    memory_after = self._get_memory_usage()

                    times.append(processing_time)
                    results.append(result)

                except Exception as e:
                    return BenchmarkResult(
                        strategy_name=strategy_name,
                        dataset_name=dataset_name,
                        content_size=len(content),
                        processing_time=0.0,
                        memory_usage=None,
                        chunk_count=0,
                        avg_chunk_size=0.0,
                        quality_metrics={},
                        parameters=parameters,
                        success=False,
                        error_message=str(e)
                    )

            # Use best result for quality metrics
            best_result = max(results, key=lambda r: len(r.chunks))

            # Calculate quality metrics
            quality_metrics = self.quality_evaluator.evaluate(best_result, content)

            # Calculate memory usage
            memory_usage = memory_after - memory_before if memory_after and memory_before else None

            return BenchmarkResult(
                strategy_name=strategy_name,
                dataset_name=dataset_name,
                content_size=len(content),
                processing_time=statistics.mean(times),
                memory_usage=memory_usage,
                chunk_count=len(best_result.chunks),
                avg_chunk_size=best_result.avg_chunk_size or 0.0,
                quality_metrics=quality_metrics.to_dict(),
                parameters=parameters,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Benchmark failed for {strategy_name}: {e}")
            return BenchmarkResult(
                strategy_name=strategy_name,
                dataset_name=dataset_name,
                content_size=len(content),
                processing_time=0.0,
                memory_usage=None,
                chunk_count=0,
                avg_chunk_size=0.0,
                quality_metrics={},
                parameters=parameters,
                success=False,
                error_message=str(e)
            )

    def benchmark_multiple_strategies(
        self,
        strategies: List[Union[str, tuple[str, Dict[str, Any]]]],
        content: Union[str, bytes],
        dataset_name: str = "unknown",
        runs: int = 1
    ) -> List[BenchmarkResult]:
        """
        Benchmark multiple strategies on the same content.

        Args:
            strategies: List of strategy names or (name, parameters) tuples
            content: Content to chunk
            dataset_name: Name of dataset
            runs: Number of runs per strategy

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
                parameters=parameters,
                runs=runs
            )

            results.append(result)

        return results

    def run_benchmark_suite(
        self,
        suite_name: str,
        strategies: List[Union[str, tuple[str, Dict[str, Any]]]],
        datasets: Dict[str, Union[str, bytes, Path]],
        runs: int = 3,
        save_results: bool = True
    ) -> BenchmarkSuite:
        """
        Run a comprehensive benchmark suite.

        Args:
            suite_name: Name of benchmark suite
            strategies: List of strategies to test
            datasets: Dictionary mapping dataset names to content
            runs: Number of runs per strategy per dataset
            save_results: Whether to save results to disk

        Returns:
            Complete benchmark suite results
        """
        self.logger.info(f"Starting benchmark suite: {suite_name}")
        start_time = time.time()

        all_results = []

        for dataset_name, dataset_content in datasets.items():
            # Load content if it's a file path
            if isinstance(dataset_content, Path):
                try:
                    if dataset_content.suffix.lower() in ['.txt', '.md']:
                        content = dataset_content.read_text(encoding='utf-8')
                    else:
                        content = dataset_content.read_bytes()
                except Exception as e:
                    self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
                    continue
            else:
                content = dataset_content

            self.logger.info(f"Testing dataset: {dataset_name} ({len(content)} chars/bytes)")

            # Benchmark all strategies on this dataset
            dataset_results = self.benchmark_multiple_strategies(
                strategies=strategies,
                content=content,
                dataset_name=dataset_name,
                runs=runs
            )

            all_results.extend(dataset_results)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(all_results)

        # Create benchmark suite
        suite = BenchmarkSuite(
            name=suite_name,
            description=f"Benchmark of {len(strategies)} strategies on {len(datasets)} datasets",
            timestamp=start_time,
            results=all_results,
            summary_stats=summary_stats
        )

        # Save results
        if save_results:
            self.save_benchmark_suite(suite)

        total_time = time.time() - start_time
        self.logger.info(f"Benchmark suite completed in {total_time:.2f}s")

        return suite

    def quick_benchmark(
        self,
        content: str,
        strategies: Optional[List[str]] = None,
        max_strategies: int = 5
    ) -> List[BenchmarkResult]:
        """
        Quick benchmark of top strategies on given content.

        Args:
            content: Content to test
            strategies: Specific strategies to test (uses top if None)
            max_strategies: Maximum number of strategies to test

        Returns:
            List of benchmark results sorted by performance
        """
        if strategies is None:
            # Get top strategies for text content
            available_strategies = list_chunkers(modality=None)  # Get all available
            strategies = available_strategies[:max_strategies]

        results = self.benchmark_multiple_strategies(
            strategies=strategies,
            content=content,
            dataset_name="quick_test",
            runs=1
        )

        # Sort by processing speed (fastest first)
        results.sort(key=lambda r: r.processing_time if r.success else float('inf'))

        return results

    def save_benchmark_suite(self, suite: BenchmarkSuite) -> Path:
        """
        Save benchmark suite to disk.

        Args:
            suite: Benchmark suite to save

        Returns:
            Path to saved file
        """
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(suite.timestamp))
        filename = f"{suite.name}_{timestamp_str}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2)

        self.logger.info(f"Benchmark results saved to {filepath}")
        return filepath

    def load_benchmark_suite(self, filepath: Union[str, Path]) -> BenchmarkSuite:
        """
        Load benchmark suite from disk.

        Args:
            filepath: Path to benchmark file

        Returns:
            Loaded benchmark suite
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        results = [BenchmarkResult(**result_data) for result_data in data['results']]

        return BenchmarkSuite(
            name=data['name'],
            description=data['description'],
            timestamp=data['timestamp'],
            results=results,
            summary_stats=data['summary_stats']
        )

    def compare_strategies(
        self,
        results: List[BenchmarkResult],
        metric: str = "processing_time"
    ) -> Dict[str, Any]:
        """
        Compare strategies based on a specific metric.

        Args:
            results: Benchmark results to compare
            metric: Metric to compare ("processing_time", "quality_score", etc.)

        Returns:
            Comparison results
        """
        if not results:
            return {}

        # Group results by strategy
        strategy_groups = {}
        for result in results:
            if result.success:
                if result.strategy_name not in strategy_groups:
                    strategy_groups[result.strategy_name] = []
                strategy_groups[result.strategy_name].append(result)

        # Calculate statistics for each strategy
        comparison = {}
        for strategy_name, strategy_results in strategy_groups.items():
            if metric == "processing_time":
                values = [r.processing_time for r in strategy_results]
            elif metric == "quality_score":
                values = [r.quality_metrics.get('overall_score', 0.0) for r in strategy_results]
            elif metric == "chunk_count":
                values = [r.chunk_count for r in strategy_results]
            elif metric == "avg_chunk_size":
                values = [r.avg_chunk_size for r in strategy_results]
            else:
                values = [r.quality_metrics.get(metric, 0.0) for r in strategy_results]

            if values:
                comparison[strategy_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "count": len(values)
                }

        return comparison

    def generate_report(self, suite: BenchmarkSuite) -> str:
        """
        Generate a human-readable report from benchmark suite.

        Args:
            suite: Benchmark suite to report on

        Returns:
            Formatted report string
        """
        report = []
        report.append(f"Benchmark Suite: {suite.name}")
        report.append(f"Description: {suite.description}")
        report.append(f"Timestamp: {time.ctime(suite.timestamp)}")
        report.append(f"Total Results: {len(suite.results)}")
        report.append("")

        # Success rate
        successful = sum(1 for r in suite.results if r.success)
        success_rate = successful / len(suite.results) if suite.results else 0
        report.append(f"Success Rate: {success_rate:.1%} ({successful}/{len(suite.results)})")
        report.append("")

        # Top performers by speed
        successful_results = [r for r in suite.results if r.success]
        if successful_results:
            fastest = min(successful_results, key=lambda r: r.processing_time)
            report.append(f"Fastest Strategy: {fastest.strategy_name} ({fastest.processing_time:.3f}s)")

            # Top performers by quality
            best_quality = max(successful_results, key=lambda r: r.quality_metrics.get('overall_score', 0))
            report.append(f"Best Quality: {best_quality.strategy_name} (score: {best_quality.quality_metrics.get('overall_score', 0):.3f})")
            report.append("")

        # Strategy comparison
        speed_comparison = self.compare_strategies(successful_results, "processing_time")
        if speed_comparison:
            report.append("Processing Time Comparison:")
            for strategy, stats in sorted(speed_comparison.items(), key=lambda x: x[1]['mean']):
                report.append(f"  {strategy}: {stats['mean']:.3f}s (±{stats['std']:.3f})")
            report.append("")

        quality_comparison = self.compare_strategies(successful_results, "quality_score")
        if quality_comparison:
            report.append("Quality Score Comparison:")
            for strategy, stats in sorted(quality_comparison.items(), key=lambda x: x[1]['mean'], reverse=True):
                report.append(f"  {strategy}: {stats['mean']:.3f} (±{stats['std']:.3f})")

        return "\n".join(report)

    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return None

    def _calculate_summary_stats(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics for benchmark results."""
        if not results:
            return {}

        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {"success_rate": 0.0, "total_results": len(results)}

        processing_times = [r.processing_time for r in successful_results]
        chunk_counts = [r.chunk_count for r in successful_results]
        quality_scores = [r.quality_metrics.get('overall_score', 0.0) for r in successful_results]

        return {
            "success_rate": len(successful_results) / len(results),
            "total_results": len(results),
            "avg_processing_time": statistics.mean(processing_times),
            "avg_chunk_count": statistics.mean(chunk_counts),
            "avg_quality_score": statistics.mean(quality_scores),
            "fastest_time": min(processing_times),
            "slowest_time": max(processing_times),
            "best_quality": max(quality_scores),
            "strategies_tested": len(set(r.strategy_name for r in results)),
            "datasets_tested": len(set(r.dataset_name for r in results))
        }
