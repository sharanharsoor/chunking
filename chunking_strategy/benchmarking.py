"""
Comprehensive Benchmarking Framework for Chunking Strategies.

This module provides comprehensive performance benchmarking and analysis
for all chunking strategies, including advanced chunkers (FastCDC, Adaptive, Context-Enriched).
"""

import time
import gc
import psutil
import logging
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
import yaml

from chunking_strategy.core.base import BaseChunker, ChunkingResult, ModalityType
from chunking_strategy.core.registry import list_chunkers, create_chunker
from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.enhanced_orchestrator import EnhancedOrchestrator


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single chunking operation."""
    strategy_name: str
    file_path: str
    file_size: int
    processing_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    chunks_generated: int
    avg_chunk_size: float
    quality_score: float
    throughput_mb_per_sec: float
    cpu_usage_percent: float
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Results from a complete benchmark run."""
    test_name: str
    timestamp: str
    system_info: Dict[str, Any]
    file_metrics: List[PerformanceMetrics]
    strategy_summaries: Dict[str, Dict[str, Any]]
    comparative_analysis: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "file_metrics": [metric.to_dict() for metric in self.file_metrics],
            "strategy_summaries": self.strategy_summaries,
            "comparative_analysis": self.comparative_analysis
        }


class ChunkingBenchmark:
    """
    Comprehensive benchmarking suite for chunking strategies.

    Features:
    - Performance profiling (time, memory, CPU)
    - Quality assessment (chunk distribution, coherence)
    - Comparative analysis across strategies
    - System resource monitoring
    - Detailed reporting and visualization
    """

    def __init__(
        self,
        enable_memory_profiling: bool = True,
        enable_cpu_monitoring: bool = True,
        enable_quality_analysis: bool = True,
        warmup_iterations: int = 1,
        benchmark_iterations: int = 3
    ):
        """
        Initialize benchmarking suite.

        Args:
            enable_memory_profiling: Track memory usage during chunking
            enable_cpu_monitoring: Monitor CPU usage
            enable_quality_analysis: Analyze chunk quality metrics
            warmup_iterations: Number of warmup runs before measurement
            benchmark_iterations: Number of measurement iterations
        """
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_cpu_monitoring = enable_cpu_monitoring
        self.enable_quality_analysis = enable_quality_analysis
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            import platform
            import sys
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "platform": platform.system(),
                "python_version": sys.version,
                "chunking_library_version": "1.0.0"  # Update as needed
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system info: {e}")
            return {"error": str(e)}

    def benchmark_strategy(
        self,
        strategy_name: str,
        file_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None
    ) -> PerformanceMetrics:
        """
        Benchmark a single strategy on a single file.

        Args:
            strategy_name: Name of the chunking strategy
            file_path: Path to the file to chunk
            config: Optional configuration for the strategy

        Returns:
            Performance metrics for the operation
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        errors = []
        warnings = []

        # Warmup runs
        for _ in range(self.warmup_iterations):
            try:
                self._single_chunk_operation(strategy_name, file_path, config)
            except Exception as e:
                self.logger.warning(f"Warmup failed for {strategy_name}: {e}")

        # Measurement runs
        times = []
        memory_usages = []
        peak_memories = []
        cpu_usages = []
        results = []

        for iteration in range(self.benchmark_iterations):
            try:
                metrics = self._measure_single_run(strategy_name, file_path, config)
                times.append(metrics['processing_time'])
                memory_usages.append(metrics['memory_usage_mb'])
                peak_memories.append(metrics['peak_memory_mb'])
                cpu_usages.append(metrics['cpu_usage_percent'])
                results.append(metrics['result'])

            except Exception as e:
                error_msg = f"Iteration {iteration} failed: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)

        if not times:
            # All iterations failed
            return PerformanceMetrics(
                strategy_name=strategy_name,
                file_path=str(file_path),
                file_size=file_size,
                processing_time=0.0,
                memory_usage_mb=0.0,
                peak_memory_mb=0.0,
                chunks_generated=0,
                avg_chunk_size=0.0,
                quality_score=0.0,
                throughput_mb_per_sec=0.0,
                cpu_usage_percent=0.0,
                errors=errors,
                warnings=warnings
            )

        # Calculate averages
        avg_time = sum(times) / len(times)
        avg_memory = sum(memory_usages) / len(memory_usages)
        avg_peak_memory = sum(peak_memories) / len(peak_memories)
        avg_cpu = sum(cpu_usages) / len(cpu_usages)

        # Use the best result for chunk analysis
        best_result = results[0] if results else None

        chunks_generated = len(best_result.chunks) if best_result else 0
        avg_chunk_size = best_result.avg_chunk_size if best_result else 0.0
        quality_score = best_result.quality_score if best_result else 0.0

        # Calculate throughput
        throughput = (file_size / (1024*1024)) / avg_time if avg_time > 0 else 0.0

        return PerformanceMetrics(
            strategy_name=strategy_name,
            file_path=str(file_path),
            file_size=file_size,
            processing_time=avg_time,
            memory_usage_mb=avg_memory,
            peak_memory_mb=avg_peak_memory,
            chunks_generated=chunks_generated,
            avg_chunk_size=avg_chunk_size,
            quality_score=quality_score,
            throughput_mb_per_sec=throughput,
            cpu_usage_percent=avg_cpu,
            errors=errors,
            warnings=warnings
        )

    def _measure_single_run(
        self,
        strategy_name: str,
        file_path: Path,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Measure a single chunking run with full profiling."""

        # Start monitoring
        if self.enable_memory_profiling:
            tracemalloc.start()

        initial_memory = self.process.memory_info().rss / (1024*1024)
        initial_cpu_time = self.process.cpu_times()

        start_time = time.perf_counter()

        # Perform chunking
        result = self._single_chunk_operation(strategy_name, file_path, config)

        end_time = time.perf_counter()

        # Calculate metrics
        processing_time = end_time - start_time

        final_memory = self.process.memory_info().rss / (1024*1024)
        memory_usage = final_memory - initial_memory

        peak_memory = memory_usage
        if self.enable_memory_profiling:
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak / (1024*1024)
            tracemalloc.stop()

        # CPU usage calculation
        final_cpu_time = self.process.cpu_times()
        cpu_time_used = (final_cpu_time.user + final_cpu_time.system) - \
                       (initial_cpu_time.user + initial_cpu_time.system)
        cpu_usage = (cpu_time_used / processing_time * 100) if processing_time > 0 else 0

        # Force garbage collection
        gc.collect()

        return {
            'processing_time': processing_time,
            'memory_usage_mb': memory_usage,
            'peak_memory_mb': peak_memory,
            'cpu_usage_percent': cpu_usage,
            'result': result
        }

    def _single_chunk_operation(
        self,
        strategy_name: str,
        file_path: Path,
        config: Optional[Dict[str, Any]]
    ) -> ChunkingResult:
        """Perform a single chunking operation."""

        try:
            # Try to create chunker directly
            chunker = create_chunker(strategy_name)

            # Apply config if provided
            if config and hasattr(chunker, 'update_config'):
                chunker.update_config(config)

            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()

            # Perform chunking
            if hasattr(chunker, 'chunk_file'):
                result = chunker.chunk_file(file_path)
            else:
                # Determine content type
                try:
                    text_content = content.decode('utf-8')
                    result = chunker.chunk(text_content)
                except UnicodeDecodeError:
                    result = chunker.chunk(content)

            return result

        except Exception as e:
            # Fallback to orchestrator
            self.logger.warning(f"Direct chunker creation failed for {strategy_name}, using orchestrator: {e}")

            orchestrator = ChunkerOrchestrator()
            result = orchestrator.chunk_file(file_path, strategy=strategy_name)
            return result

    def benchmark_multiple_strategies(
        self,
        strategies: List[str],
        file_paths: List[Union[str, Path]],
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> BenchmarkResult:
        """
        Benchmark multiple strategies on multiple files.

        Args:
            strategies: List of strategy names to benchmark
            file_paths: List of file paths to test
            configs: Optional configs for each strategy

        Returns:
            Complete benchmark results
        """
        from datetime import datetime

        test_name = f"Multi-Strategy Benchmark {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()

        self.logger.info(f"Starting benchmark: {test_name}")
        self.logger.info(f"Strategies: {strategies}")
        self.logger.info(f"Files: {[str(p) for p in file_paths]}")

        # Get system information
        system_info = self.get_system_info()

        # Run benchmarks
        file_metrics = []

        for file_path in file_paths:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.warning(f"Skipping non-existent file: {file_path}")
                continue

            self.logger.info(f"Benchmarking file: {file_path}")

            for strategy in strategies:
                self.logger.info(f"  Testing strategy: {strategy}")

                config = configs.get(strategy) if configs else None

                try:
                    metrics = self.benchmark_strategy(strategy, file_path, config)
                    file_metrics.append(metrics)

                    self.logger.info(f"  {strategy}: {metrics.processing_time:.3f}s, "
                                   f"{metrics.chunks_generated} chunks, "
                                   f"{metrics.throughput_mb_per_sec:.2f} MB/s")

                except Exception as e:
                    error_msg = f"Benchmark failed for {strategy} on {file_path}: {e}"
                    self.logger.error(error_msg)

                    # Create error metrics
                    error_metrics = PerformanceMetrics(
                        strategy_name=strategy,
                        file_path=str(file_path),
                        file_size=file_path.stat().st_size if file_path.exists() else 0,
                        processing_time=0.0,
                        memory_usage_mb=0.0,
                        peak_memory_mb=0.0,
                        chunks_generated=0,
                        avg_chunk_size=0.0,
                        quality_score=0.0,
                        throughput_mb_per_sec=0.0,
                        cpu_usage_percent=0.0,
                        errors=[error_msg],
                        warnings=[]
                    )
                    file_metrics.append(error_metrics)

        # Generate summaries and analysis
        strategy_summaries = self._generate_strategy_summaries(file_metrics)
        comparative_analysis = self._generate_comparative_analysis(file_metrics)

        return BenchmarkResult(
            test_name=test_name,
            timestamp=timestamp,
            system_info=system_info,
            file_metrics=file_metrics,
            strategy_summaries=strategy_summaries,
            comparative_analysis=comparative_analysis
        )

    def _generate_strategy_summaries(self, metrics: List[PerformanceMetrics]) -> Dict[str, Dict[str, Any]]:
        """Generate summary statistics for each strategy."""
        strategy_data = defaultdict(list)

        # Group metrics by strategy
        for metric in metrics:
            if metric.processing_time > 0:  # Skip failed runs
                strategy_data[metric.strategy_name].append(metric)

        summaries = {}

        for strategy, strategy_metrics in strategy_data.items():
            if not strategy_metrics:
                continue

            # Calculate aggregated statistics
            times = [m.processing_time for m in strategy_metrics]
            throughputs = [m.throughput_mb_per_sec for m in strategy_metrics]
            memory_usages = [m.memory_usage_mb for m in strategy_metrics]
            quality_scores = [m.quality_score for m in strategy_metrics if m.quality_score is not None and m.quality_score > 0]

            summaries[strategy] = {
                "files_processed": len(strategy_metrics),
                "avg_processing_time": sum(times) / len(times),
                "min_processing_time": min(times),
                "max_processing_time": max(times),
                "avg_throughput_mb_s": sum(throughputs) / len(throughputs),
                "avg_memory_usage_mb": sum(memory_usages) / len(memory_usages),
                "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                "total_errors": sum(len(m.errors) for m in strategy_metrics),
                "total_warnings": sum(len(m.warnings) for m in strategy_metrics),
                "success_rate": len([m for m in strategy_metrics if not m.errors]) / len(strategy_metrics)
            }

        return summaries

    def _generate_comparative_analysis(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Generate comparative analysis across strategies."""
        if not metrics:
            return {}

        # Filter successful runs
        successful_metrics = [m for m in metrics if m.processing_time > 0 and not m.errors]

        if not successful_metrics:
            return {"error": "No successful runs to analyze"}

        # Find best performers
        fastest_strategy = min(successful_metrics, key=lambda m: m.processing_time)
        highest_throughput = max(successful_metrics, key=lambda m: m.throughput_mb_per_sec)
        lowest_memory = min(successful_metrics, key=lambda m: m.memory_usage_mb)
        # Filter out None quality scores before finding max
        metrics_with_quality = [m for m in successful_metrics if m.quality_score is not None]
        highest_quality = max(metrics_with_quality, key=lambda m: m.quality_score) if metrics_with_quality else None

        # Calculate overall statistics
        all_times = [m.processing_time for m in successful_metrics]
        all_throughputs = [m.throughput_mb_per_sec for m in successful_metrics]
        all_memory = [m.memory_usage_mb for m in successful_metrics]

        return {
            "fastest_strategy": {
                "name": fastest_strategy.strategy_name,
                "time": fastest_strategy.processing_time,
                "file": fastest_strategy.file_path
            },
            "highest_throughput": {
                "name": highest_throughput.strategy_name,
                "throughput": highest_throughput.throughput_mb_per_sec,
                "file": highest_throughput.file_path
            },
            "lowest_memory": {
                "name": lowest_memory.strategy_name,
                "memory": lowest_memory.memory_usage_mb,
                "file": lowest_memory.file_path
            },
            "highest_quality": {
                "name": highest_quality.strategy_name if highest_quality else "N/A",
                "quality": highest_quality.quality_score if highest_quality else 0.0,
                "file": highest_quality.file_path if highest_quality else "N/A"
            },
            "overall_stats": {
                "avg_processing_time": sum(all_times) / len(all_times),
                "avg_throughput": sum(all_throughputs) / len(all_throughputs),
                "avg_memory_usage": sum(all_memory) / len(all_memory),
                "total_successful_runs": len(successful_metrics),
                "total_runs": len(metrics)
            }
        }

    def save_results(
        self,
        results: BenchmarkResult,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save benchmark results to file.

        Args:
            results: Benchmark results to save
            output_path: Output file path
            format: Output format ('json' or 'yaml')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = results.to_dict()

        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format.lower() == "yaml":
            with open(output_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Benchmark results saved to: {output_path}")

    def generate_report(self, results: BenchmarkResult) -> str:
        """Generate a human-readable benchmark report."""
        report = []

        # Header
        report.append("="*80)
        report.append(f"CHUNKING STRATEGY BENCHMARK REPORT")
        report.append("="*80)
        report.append(f"Test: {results.test_name}")
        report.append(f"Timestamp: {results.timestamp}")
        report.append("")

        # System Info
        report.append("SYSTEM INFORMATION:")
        report.append("-" * 40)
        for key, value in results.system_info.items():
            report.append(f"{key:20}: {value}")
        report.append("")

        # Strategy Summaries
        report.append("STRATEGY PERFORMANCE SUMMARY:")
        report.append("-" * 40)

        for strategy, summary in results.strategy_summaries.items():
            report.append(f"\n{strategy.upper()}:")
            report.append(f"  Files processed: {summary['files_processed']}")
            report.append(f"  Avg time: {summary['avg_processing_time']:.3f}s")
            report.append(f"  Avg throughput: {summary['avg_throughput_mb_s']:.2f} MB/s")
            report.append(f"  Avg memory: {summary['avg_memory_usage_mb']:.2f} MB")
            report.append(f"  Avg quality: {summary['avg_quality_score']:.3f}")
            report.append(f"  Success rate: {summary['success_rate']*100:.1f}%")

        # Comparative Analysis
        if "error" not in results.comparative_analysis:
            report.append("\nCOMPARATIVE ANALYSIS:")
            report.append("-" * 40)

            analysis = results.comparative_analysis
            report.append(f"🏆 Fastest: {analysis['fastest_strategy']['name']} "
                         f"({analysis['fastest_strategy']['time']:.3f}s)")
            report.append(f"🚀 Highest throughput: {analysis['highest_throughput']['name']} "
                         f"({analysis['highest_throughput']['throughput']:.2f} MB/s)")
            report.append(f"💾 Lowest memory: {analysis['lowest_memory']['name']} "
                         f"({analysis['lowest_memory']['memory']:.2f} MB)")
            report.append(f"⭐ Highest quality: {analysis['highest_quality']['name']} "
                         f"({analysis['highest_quality']['quality']:.3f})")

        report.append("\n" + "="*80)

        return "\n".join(report)


def run_comprehensive_benchmark(
    output_dir: Optional[Union[str, Path]] = None,
    test_files: Optional[List[Union[str, Path]]] = None,
    strategies: Optional[List[str]] = None
) -> BenchmarkResult:
    """
    Run a comprehensive benchmark of all chunking strategies.

    Args:
        output_dir: Directory to save results (default: current dir)
        test_files: List of test files (default: use available test data)
        strategies: List of strategies to test (default: all available)

    Returns:
        Complete benchmark results
    """
    from datetime import datetime

    # Setup
    benchmark = ChunkingBenchmark(
        enable_memory_profiling=True,
        enable_cpu_monitoring=True,
        enable_quality_analysis=True,
        warmup_iterations=1,
        benchmark_iterations=3
    )

    # Default strategies to test (including advanced ones)
    if strategies is None:
        strategies = [
            # Advanced chunkers
            'fastcdc',
            'adaptive',
            'context_enriched',
            # Traditional chunkers
            'fixed_size',
            'sentence_based',
            'paragraph',
            # Format-specific chunkers
            'python',
            'javascript',
            'json',
            'markdown',
            'csv'
        ]

    # Default test files
    if test_files is None:
        test_files = [
            'test_data/sample_semantic_text.txt',
            'test_data/sample_adaptive_data.json',
            'test_data/sample_entity_narrative.txt',
            'test_data/sample_topic_transitions.txt'
        ]
        # Filter to existing files
        test_files = [f for f in test_files if Path(f).exists()]

    # Run benchmark
    results = benchmark.benchmark_multiple_strategies(strategies, test_files)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON results
        json_path = output_dir / f"benchmark_results_{timestamp}.json"
        benchmark.save_results(results, json_path, format="json")

        # Save human-readable report
        report_path = output_dir / f"benchmark_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(benchmark.generate_report(results))

        print(f"Benchmark results saved to: {output_dir}")
        print(f"  JSON: {json_path}")
        print(f"  Report: {report_path}")

    return results


if __name__ == "__main__":
    # Run comprehensive benchmark when executed directly
    results = run_comprehensive_benchmark(
        output_dir="benchmark_results",
        test_files=None,  # Use defaults
        strategies=None   # Test all strategies
    )

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)

    # Print summary
    if results.strategy_summaries:
        print("\nTOP PERFORMERS:")

        # Sort by average processing time
        sorted_strategies = sorted(
            results.strategy_summaries.items(),
            key=lambda x: x[1]['avg_processing_time']
        )

        for i, (strategy, summary) in enumerate(sorted_strategies[:5], 1):
            print(f"{i}. {strategy}: {summary['avg_processing_time']:.3f}s avg, "
                  f"{summary['avg_throughput_mb_s']:.2f} MB/s")
