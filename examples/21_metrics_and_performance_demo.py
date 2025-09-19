#!/usr/bin/env python3
"""
Chunking Strategy - Metrics and Performance Demo

This script demonstrates comprehensive performance monitoring and metrics collection
for chunking operations. It shows how to:

1. Monitor performance in real-time
2. Collect detailed metrics at various levels
3. Compare different chunking strategies
4. Export metrics for analysis
5. Create performance visualizations

Usage:
    python examples/21_metrics_and_performance_demo.py
"""

import os
import sys
from pathlib import Path
import time
import json
import statistics
from typing import Dict, List, Any
import psutil

# Add parent directory to path for local development
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from chunking_strategy import create_chunker, list_strategies

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("📊 Optional: Install matplotlib and seaborn for visualizations: pip install matplotlib seaborn")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("📊 Optional: Install pandas for advanced analysis: pip install pandas")


class PerformanceMonitor:
    """Simple performance monitoring for chunking operations."""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_peak = 0
        self.chunks_processed = 0
        self.input_size = 0
        self.chunk_sizes = []
        self.processing_stages = {}

    def start(self, content_size: int = 0):
        """Start monitoring."""
        self.start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_peak = self.memory_start
        self.input_size = content_size
        print(f"⏱️  Starting performance monitoring for {self.strategy_name}")

    def update_memory(self):
        """Update peak memory usage."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        if current_memory > self.memory_peak:
            self.memory_peak = current_memory

    def add_chunk(self, chunk):
        """Record a processed chunk."""
        self.chunks_processed += 1

        # Calculate chunk size
        if hasattr(chunk, 'content'):
            size = len(str(chunk.content))
        elif isinstance(chunk, (str, bytes)):
            size = len(chunk)
        else:
            size = len(str(chunk))

        self.chunk_sizes.append(size)
        self.update_memory()

    def add_stage_time(self, stage_name: str, duration: float):
        """Record time for a processing stage."""
        self.processing_stages[stage_name] = duration

    def finish(self):
        """Finish monitoring and calculate final metrics."""
        self.end_time = time.time()
        processing_time = self.end_time - self.start_time

        print(f"✅ Monitoring complete for {self.strategy_name}")
        print(f"   📊 Processing time: {processing_time:.3f}s")
        print(f"   📊 Chunks created: {self.chunks_processed}")
        print(f"   📊 Memory usage: {self.memory_start:.1f}MB → {self.memory_peak:.1f}MB")

        return self.get_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics dictionary."""
        processing_time = (self.end_time or time.time()) - (self.start_time or 0)

        metrics = {
            "strategy": self.strategy_name,
            "timestamp": time.time(),
            "performance": {
                "processing_time_seconds": processing_time,
                "chunks_per_second": self.chunks_processed / processing_time if processing_time > 0 else 0,
                "bytes_per_second": self.input_size / processing_time if processing_time > 0 else 0,
                "mb_per_second": (self.input_size / 1024 / 1024) / processing_time if processing_time > 0 else 0,
            },
            "memory": {
                "start_mb": self.memory_start or 0,
                "peak_mb": self.memory_peak,
                "delta_mb": self.memory_peak - (self.memory_start or 0),
            },
            "chunks": {
                "total_count": self.chunks_processed,
                "avg_size": statistics.mean(self.chunk_sizes) if self.chunk_sizes else 0,
                "median_size": statistics.median(self.chunk_sizes) if self.chunk_sizes else 0,
                "min_size": min(self.chunk_sizes) if self.chunk_sizes else 0,
                "max_size": max(self.chunk_sizes) if self.chunk_sizes else 0,
                "size_std": statistics.stdev(self.chunk_sizes) if len(self.chunk_sizes) > 1 else 0,
            },
            "input": {
                "size_bytes": self.input_size,
                "size_mb": self.input_size / 1024 / 1024,
            },
            "stages": self.processing_stages
        }

        # Quality metrics
        if self.chunk_sizes:
            consistency = 1 - (metrics["chunks"]["size_std"] / metrics["chunks"]["avg_size"]) if metrics["chunks"]["avg_size"] > 0 else 0
            metrics["quality"] = {
                "size_consistency": max(0, consistency),  # 1 = perfectly consistent, 0 = highly variable
                "coverage_ratio": sum(self.chunk_sizes) / self.input_size if self.input_size > 0 else 0,
            }

        return metrics


def monitor_chunking_operation(strategy_name: str, content: str, **strategy_kwargs) -> Dict[str, Any]:
    """Monitor a complete chunking operation and return metrics."""

    monitor = PerformanceMonitor(strategy_name)
    monitor.start(len(content.encode()))

    try:
        # Create chunker
        stage_start = time.time()
        chunker = create_chunker(strategy_name, **strategy_kwargs)
        if chunker is None:
            raise Exception(f"Failed to create chunker for strategy: {strategy_name}")
        monitor.add_stage_time("chunker_creation", time.time() - stage_start)

        # Perform chunking
        stage_start = time.time()
        result = chunker.chunk(content)

        # Process chunks - handle ChunkingResult properly
        if hasattr(result, 'chunks'):
            chunks_list = list(result.chunks)
        elif hasattr(result, '__iter__'):
            chunks_list = list(result)
        else:
            chunks_list = [result]

        for chunk in chunks_list:
            monitor.add_chunk(chunk)

        monitor.add_stage_time("chunking_execution", time.time() - stage_start)

    except Exception as e:
        print(f"❌ Error during chunking with {strategy_name}: {e}")

    return monitor.finish()


def benchmark_strategies(content: str, strategies: List[str], iterations: int = 3) -> Dict[str, Any]:
    """Benchmark multiple strategies on the same content."""

    print(f"\n🏁 Starting benchmark with {len(strategies)} strategies, {iterations} iterations each")
    print(f"📝 Content size: {len(content)} characters ({len(content.encode())} bytes)")
    print("=" * 70)

    results = {}

    for strategy_name in strategies:
        print(f"\n🧪 Testing strategy: {strategy_name}")
        strategy_results = []

        for i in range(iterations):
            print(f"  🔄 Iteration {i+1}/{iterations}")
            metrics = monitor_chunking_operation(strategy_name, content)
            strategy_results.append(metrics)
            time.sleep(0.1)  # Brief pause between iterations

        # Aggregate results
        results[strategy_name] = {
            "iterations": strategy_results,
            "aggregate": _aggregate_results(strategy_results)
        }

        # Print summary
        agg = results[strategy_name]["aggregate"]
        print(f"  📊 Average processing time: {agg['avg_processing_time']:.3f}s")
        print(f"  📊 Average throughput: {agg['avg_chunks_per_second']:.1f} chunks/s")
        print(f"  📊 Average memory delta: {agg['avg_memory_delta']:.1f}MB")

    return {
        "content_size": len(content),
        "iterations_per_strategy": iterations,
        "timestamp": time.time(),
        "results": results,
        "ranking": _rank_strategies(results)
    }


def _aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics from multiple benchmark iterations."""

    if not results:
        return {}

    # Extract metrics
    processing_times = [r["performance"]["processing_time_seconds"] for r in results]
    chunks_per_second = [r["performance"]["chunks_per_second"] for r in results]
    memory_deltas = [r["memory"]["delta_mb"] for r in results]
    total_chunks = [r["chunks"]["total_count"] for r in results]
    avg_chunk_sizes = [r["chunks"]["avg_size"] for r in results]

    return {
        "avg_processing_time": statistics.mean(processing_times),
        "std_processing_time": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
        "avg_chunks_per_second": statistics.mean(chunks_per_second),
        "avg_memory_delta": statistics.mean(memory_deltas),
        "avg_total_chunks": statistics.mean(total_chunks),
        "avg_chunk_size": statistics.mean(avg_chunk_sizes),
        "reliability_score": len(results)  # All completed successfully
    }


def _rank_strategies(results: Dict[str, Any]) -> Dict[str, List[str]]:
    """Rank strategies by different performance criteria."""

    strategies = list(results.keys())

    return {
        "fastest_processing": sorted(strategies,
            key=lambda s: results[s]["aggregate"]["avg_processing_time"]),
        "highest_throughput": sorted(strategies,
            key=lambda s: results[s]["aggregate"]["avg_chunks_per_second"], reverse=True),
        "most_memory_efficient": sorted(strategies,
            key=lambda s: results[s]["aggregate"]["avg_memory_delta"]),
        "most_consistent_chunks": sorted(strategies,
            key=lambda s: results[s]["aggregate"]["avg_chunk_size"])
    }


def create_performance_visualization(benchmark_results: Dict[str, Any]):
    """Create visualizations of benchmark results."""

    if not HAS_PLOTTING:
        print("📊 Skipping visualizations (matplotlib not available)")
        return

    results = benchmark_results["results"]
    strategies = list(results.keys())

    # Extract data for plotting
    processing_times = [results[s]["aggregate"]["avg_processing_time"] for s in strategies]
    throughputs = [results[s]["aggregate"]["avg_chunks_per_second"] for s in strategies]
    memory_usage = [results[s]["aggregate"]["avg_memory_delta"] for s in strategies]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Chunking Strategy Performance Comparison', fontsize=16)

    # Processing time comparison
    axes[0, 0].bar(strategies, processing_times, color='skyblue')
    axes[0, 0].set_title('Processing Time')
    axes[0, 0].set_ylabel('Seconds')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Throughput comparison
    axes[0, 1].bar(strategies, throughputs, color='lightgreen')
    axes[0, 1].set_title('Throughput')
    axes[0, 1].set_ylabel('Chunks/Second')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Memory usage comparison
    axes[1, 0].bar(strategies, memory_usage, color='salmon')
    axes[1, 0].set_title('Memory Usage')
    axes[1, 0].set_ylabel('MB Delta')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Rankings summary
    rankings = benchmark_results["ranking"]
    axes[1, 1].axis('off')
    ranking_text = "🏆 Performance Rankings:\n\n"
    ranking_text += f"⚡ Fastest: {rankings['fastest_processing'][0]}\n"
    ranking_text += f"🚀 Highest Throughput: {rankings['highest_throughput'][0]}\n"
    ranking_text += f"💾 Most Memory Efficient: {rankings['most_memory_efficient'][0]}\n"

    axes[1, 1].text(0.1, 0.7, ranking_text, fontsize=12, verticalalignment='top')

    plt.tight_layout()

    # Save plot
    output_file = "performance_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Performance visualization saved as: {output_file}")

    return output_file


def save_metrics_report(benchmark_results: Dict[str, Any], filename: str = "chunking_metrics_report.json"):
    """Save comprehensive metrics report to JSON file."""

    with open(filename, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"📄 Metrics report saved as: {filename}")
    return filename


def demonstrate_real_time_monitoring():
    """Demonstrate real-time performance monitoring during chunking."""

    print("\n" + "="*70)
    print("🔍 REAL-TIME MONITORING DEMONSTRATION")
    print("="*70)

    # Sample content
    sample_content = """
    This is a comprehensive demonstration of real-time performance monitoring
    for chunking operations. We can track various metrics including processing
    time, memory usage, chunk statistics, and quality measures.

    The system provides detailed insights into how different chunking strategies
    perform with various types of content. This information helps users choose
    the optimal strategy for their specific use case.

    Performance monitoring includes memory usage tracking, processing time
    measurement, throughput calculation, and quality assessment metrics.
    """ * 50  # Make it larger for meaningful metrics

    strategies_to_test = [
        "sentence_based",
        "fixed_length_word",
        "fixed_size"
    ]

    print(f"📝 Sample content: {len(sample_content)} characters")
    print(f"🧪 Testing strategies: {', '.join(strategies_to_test)}")

    individual_results = {}

    for strategy in strategies_to_test:
        print(f"\n🔬 Monitoring {strategy} strategy...")
        metrics = monitor_chunking_operation(strategy, sample_content, max_chunk_size=200)
        individual_results[strategy] = metrics

        print(f"   ⚡ Throughput: {metrics['performance']['chunks_per_second']:.1f} chunks/s")
        print(f"   💾 Memory delta: {metrics['memory']['delta_mb']:.1f}MB")
        print(f"   📊 Avg chunk size: {metrics['chunks']['avg_size']:.1f} characters")

    return individual_results


def main():
    """Main demonstration function."""

    print("🎯 CHUNKING STRATEGY - METRICS & PERFORMANCE DEMO")
    print("="*70)

    # Show available strategies
    available_strategies = list_strategies()
    print(f"📋 Available strategies: {len(available_strategies)}")
    for i, strategy in enumerate(available_strategies[:10], 1):  # Show first 10
        print(f"   {i:2d}. {strategy}")
    if len(available_strategies) > 10:
        print(f"   ... and {len(available_strategies) - 10} more")

    # Demonstrate individual monitoring
    individual_results = demonstrate_real_time_monitoring()

    # Demonstrate benchmarking
    print("\n" + "="*70)
    print("🏁 BENCHMARKING DEMONSTRATION")
    print("="*70)

    # Create test content
    test_content = """
    Artificial Intelligence and Machine Learning have revolutionized the way we process
    and analyze data. Natural Language Processing, a subset of AI, focuses on enabling
    computers to understand, interpret, and generate human language.

    Text chunking is a fundamental preprocessing step in many NLP pipelines. It involves
    breaking down large documents into smaller, manageable pieces while preserving
    semantic coherence and context.

    Different chunking strategies offer various trade-offs between processing speed,
    memory efficiency, and output quality. Some prioritize semantic coherence, while
    others focus on computational efficiency.

    Performance monitoring and metrics collection help users understand these trade-offs
    and choose the optimal strategy for their specific use case and requirements.
    """ * 20  # Make it substantial for benchmarking

    # Select strategies for benchmarking
    benchmark_strategies_list = [
        "sentence_based",
        "fixed_length_word",
        "fixed_size",
        "paragraph_based"
    ]

    # Run benchmark
    benchmark_results = benchmark_strategies(test_content, benchmark_strategies_list, iterations=3)

    # Show results
    print("\n📊 BENCHMARK RESULTS SUMMARY:")
    print("-" * 50)

    rankings = benchmark_results["ranking"]
    print(f"⚡ Fastest Processing: {rankings['fastest_processing'][0]}")
    print(f"🚀 Highest Throughput: {rankings['highest_throughput'][0]}")
    print(f"💾 Most Memory Efficient: {rankings['most_memory_efficient'][0]}")

    # Save results
    report_file = save_metrics_report(benchmark_results)

    # Create visualizations
    viz_file = create_performance_visualization(benchmark_results)

    # Summary
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    print("📋 What was demonstrated:")
    print("   • Real-time performance monitoring")
    print("   • Comprehensive metrics collection")
    print("   • Multi-strategy benchmarking")
    print("   • Performance visualization")
    print("   • Metrics export and reporting")
    print()
    print("📁 Generated files:")
    print(f"   • {report_file} - Detailed metrics report")
    if viz_file:
        print(f"   • {viz_file} - Performance visualization")

    print("\n💡 Use these metrics to:")
    print("   • Choose optimal chunking strategies")
    print("   • Monitor production performance")
    print("   • Identify performance bottlenecks")
    print("   • Compare different configurations")


if __name__ == "__main__":
    main()
