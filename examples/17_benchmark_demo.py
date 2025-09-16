#!/usr/bin/env python3
"""
Benchmarking Demo Script

This script demonstrates how to use the production benchmarking system
programmatically. It shows various benchmarking scenarios including
custom algorithms, different configurations, and result analysis.

Run this script to see the benchmarking system in action.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chunking_strategy.core.production_benchmark import (
    ProductionBenchmarkRunner,
    ProductionBenchmarkConfig,
    run_quick_benchmark,
    run_custom_algorithm_benchmark
)


def demo_quick_benchmark():
    """Demonstrate quick benchmarking functionality."""
    print("=" * 60)
    print("🚀 QUICK BENCHMARK DEMO")
    print("=" * 60)

    test_content = """
    Machine learning is revolutionizing how we process and understand data.
    From natural language processing to computer vision, AI systems are
    becoming increasingly sophisticated. Deep learning models can now
    perform tasks that were once thought to be exclusively human domains.

    The field of artificial intelligence continues to evolve rapidly,
    with new breakthroughs in areas like transformer architectures,
    reinforcement learning, and neural network optimization. These
    advances are enabling more powerful and efficient AI applications.

    As we look to the future, the integration of AI into everyday life
    will likely accelerate, bringing both opportunities and challenges
    that society must thoughtfully address.
    """

    print("Testing content length:", len(test_content), "characters")
    print("Strategies: Using defaults (built-in algorithms)")
    print()

    try:
        # Run quick benchmark
        suite = run_quick_benchmark(
            content=test_content,
            strategies=None,  # Use defaults
            output_dir=Path("./demo_results/quick_benchmark")
        )

        print(f"✅ Quick benchmark completed!")
        print(f"📊 Strategies tested: {suite.summary_stats.get('strategies_tested', 0)}")
        print(f"🎯 Success rate: {suite.summary_stats.get('success_rate', 0):.1%}")

        # Show top performer
        successful = [r for r in suite.results if r.success]
        if successful:
            fastest = min(successful, key=lambda r: r.processing_time)
            print(f"⚡ Fastest: {fastest.strategy_name} ({fastest.processing_time:.3f}s)")

            best_quality = max(successful, key=lambda r: r.quality_metrics.get('overall_score', 0))
            quality_score = best_quality.quality_metrics.get('overall_score', 0)
            print(f"🏆 Best quality: {best_quality.strategy_name} (score: {quality_score:.3f})")

        print(f"📁 Detailed results saved to: demo_results/quick_benchmark/")

    except Exception as e:
        print(f"❌ Quick benchmark failed: {e}")
        return False

    return True


def demo_custom_algorithm_benchmark():
    """Demonstrate custom algorithm benchmarking."""
    print("\n" + "=" * 60)
    print("🔧 CUSTOM ALGORITHM BENCHMARK DEMO")
    print("=" * 60)

    # Check if custom algorithms exist
    custom_algos = [
        project_root / "examples/custom_algorithms/balanced_length_chunker.py",
        project_root / "examples/custom_algorithms/sentiment_based_chunker.py",
        project_root / "examples/custom_algorithms/regex_pattern_chunker.py"
    ]

    available_algos = [algo for algo in custom_algos if algo.exists()]

    if not available_algos:
        print("⚠️  No custom algorithms found in examples/custom_algorithms/")
        print("Run this from the project root with custom algorithms present.")
        return False

    print(f"Found {len(available_algos)} custom algorithms:")
    for algo in available_algos:
        print(f"  • {algo.name}")
    print()

    # Test first available algorithm
    test_algo = available_algos[0]

    test_content = """
    This is a comprehensive test document for evaluating custom chunking algorithms.
    It contains multiple paragraphs with varying sentence lengths and complexity.

    The first paragraph introduces the topic and sets the context for the analysis.
    Each sentence is crafted to provide meaningful content for chunking evaluation.

    The second paragraph delves deeper into technical aspects. Machine learning
    algorithms process text by breaking it into manageable segments. The quality
    of chunking directly impacts downstream processing effectiveness.

    Finally, the third paragraph concludes with practical considerations. Real-world
    applications require robust and efficient chunking strategies that can handle
    diverse content types while maintaining semantic coherence.
    """

    print(f"Testing: {test_algo.name}")
    print(f"Content length: {len(test_content)} characters")
    print("Comparing with: fixed_size, sentence_based")
    print()

    try:
        suite = run_custom_algorithm_benchmark(
            custom_algorithm_path=test_algo,
            compare_with=["fixed_size", "sentence_based"],
            test_content=test_content,
            output_dir=Path(f"./demo_results/custom_{test_algo.stem}")
        )

        print(f"✅ Custom algorithm benchmark completed!")

        # Analyze results
        successful = [r for r in suite.results if r.success]
        custom_results = [r for r in successful if r.is_custom_algorithm]
        builtin_results = [r for r in successful if not r.is_custom_algorithm]

        if custom_results and builtin_results:
            custom_result = custom_results[0]
            avg_builtin_time = sum(r.processing_time for r in builtin_results) / len(builtin_results)

            performance_ratio = custom_result.processing_time / avg_builtin_time
            if performance_ratio < 1.0:
                print(f"🚀 Custom algorithm is {1/performance_ratio:.1f}x faster than built-ins!")
            else:
                print(f"📊 Custom algorithm is {performance_ratio:.1f}x slower than built-ins")

            custom_quality = custom_result.quality_metrics.get('overall_score', 0)
            avg_builtin_quality = sum(r.quality_metrics.get('overall_score', 0) for r in builtin_results) / len(builtin_results)

            if custom_quality > avg_builtin_quality:
                print(f"🏆 Custom algorithm has better quality: {custom_quality:.3f} vs {avg_builtin_quality:.3f}")
            else:
                print(f"📈 Built-in algorithms have better average quality: {avg_builtin_quality:.3f} vs {custom_quality:.3f}")

        print(f"📁 Detailed results saved to: demo_results/custom_{test_algo.stem}/")

    except Exception as e:
        print(f"❌ Custom algorithm benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def demo_comprehensive_benchmark():
    """Demonstrate comprehensive benchmarking with custom configuration."""
    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE BENCHMARK DEMO")
    print("=" * 60)

    # Custom configuration
    config = ProductionBenchmarkConfig(
        output_dir=Path("./demo_results/comprehensive"),
        console_summary=True,
        save_json=True,
        save_csv=True,
        save_report=True,
        runs_per_strategy=2,  # Fast for demo
        custom_algorithm_paths=[
            project_root / "examples/custom_algorithms/balanced_length_chunker.py"
        ] if (project_root / "examples/custom_algorithms/balanced_length_chunker.py").exists() else []
    )

    print("Configuration:")
    print(f"  • Output directory: {config.output_dir}")
    print(f"  • Runs per strategy: {config.runs_per_strategy}")
    print(f"  • Custom algorithms: {len(config.custom_algorithm_paths)}")
    print()

    # Multiple datasets
    datasets = {
        "short_text": "Brief content for testing chunking algorithms.",
        "medium_text": " ".join([
            "This is a medium-length text document.",
            "It contains several sentences across multiple paragraphs.",
            "The purpose is to test how different chunking strategies handle moderately complex content.",
            "Each strategy should produce meaningful chunks that preserve semantic coherence.",
            "The results will help evaluate the relative performance and quality of each approach."
        ]),
        "structured_text": """
        # Document Title

        ## Introduction
        This section introduces the main concepts and provides background information.

        ## Methods
        The methodology section describes the approach taken in this analysis.
        - First step: Data collection
        - Second step: Processing
        - Third step: Analysis

        ## Results
        The results section presents the findings from the analysis.

        ## Conclusion
        This section summarizes the key insights and implications.
        """
    }

    strategies = ["fixed_size", "sentence_based"]
    if config.custom_algorithm_paths:
        strategies.append("balanced_length")  # Assuming this is the name from the custom algorithm

    print(f"Testing {len(strategies)} strategies on {len(datasets)} datasets:")
    print(f"  • Strategies: {', '.join(strategies)}")
    print(f"  • Datasets: {', '.join(datasets.keys())}")
    print()

    try:
        runner = ProductionBenchmarkRunner(config)

        suite = runner.run_comprehensive_benchmark(
            strategies=strategies,
            datasets=datasets,
            suite_name="comprehensive_demo"
        )

        print(f"\n✅ Comprehensive benchmark completed!")

        # Detailed analysis
        stats = suite.summary_stats
        print(f"📊 Summary Statistics:")
        print(f"  • Total results: {stats.get('total_results', 0)}")
        print(f"  • Successful results: {stats.get('successful_results', 0)}")
        print(f"  • Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"  • Strategies tested: {stats.get('strategies_tested', 0)}")
        print(f"  • Datasets tested: {stats.get('datasets_tested', 0)}")
        print(f"  • Custom algorithms: {stats.get('custom_algorithm_count', 0)}")

        if stats.get('successful_results', 0) > 0:
            print(f"  • Average processing time: {stats.get('avg_processing_time', 0):.3f}s")
            print(f"  • Fastest time: {stats.get('min_processing_time', 0):.3f}s")
            print(f"  • Average quality score: {stats.get('avg_quality_score', 0):.3f}")
            print(f"  • Best quality score: {stats.get('max_quality_score', 0):.3f}")

        # Output files created
        output_files = list(config.output_dir.glob("*"))
        print(f"\n📁 Output files created ({len(output_files)}):")
        for file in output_files:
            print(f"  • {file.name} ({file.stat().st_size} bytes)")

    except Exception as e:
        print(f"❌ Comprehensive benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def analyze_demo_results():
    """Analyze and summarize all demo results."""
    print("\n" + "=" * 60)
    print("📈 DEMO RESULTS ANALYSIS")
    print("=" * 60)

    results_dir = Path("./demo_results")
    if not results_dir.exists():
        print("No demo results found.")
        return

    # Find all JSON result files
    json_files = list(results_dir.rglob("*.json"))

    if not json_files:
        print("No JSON result files found.")
        return

    print(f"Found {len(json_files)} result files:")

    import json

    all_strategies = set()
    all_results = []

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            suite_name = data.get('name', 'unknown')
            print(f"\n📊 {suite_name} ({json_file.parent.name}):")

            results = data.get('results', [])
            successful = [r for r in results if r.get('success', False)]

            print(f"  • Results: {len(successful)}/{len(results)} successful")

            if successful:
                avg_time = sum(r.get('processing_time', 0) for r in successful) / len(successful)
                avg_quality = sum(r.get('quality_metrics', {}).get('overall_score', 0) for r in successful) / len(successful)

                print(f"  • Avg time: {avg_time:.3f}s")
                print(f"  • Avg quality: {avg_quality:.3f}")

                # Track strategies
                for result in successful:
                    all_strategies.add(result.get('strategy_name', 'unknown'))
                    all_results.append(result)

        except Exception as e:
            print(f"  ❌ Failed to analyze {json_file}: {e}")

    if all_results:
        print(f"\n🎯 OVERALL SUMMARY:")
        print(f"  • Total successful benchmarks: {len(all_results)}")
        print(f"  • Unique strategies tested: {len(all_strategies)}")
        print(f"  • Strategies: {', '.join(sorted(all_strategies))}")

        # Find best performers
        fastest = min(all_results, key=lambda r: r.get('processing_time', float('inf')))
        best_quality = max(all_results, key=lambda r: r.get('quality_metrics', {}).get('overall_score', 0))

        print(f"  • Fastest overall: {fastest.get('strategy_name')} ({fastest.get('processing_time', 0):.3f}s)")
        quality_score = best_quality.get('quality_metrics', {}).get('overall_score', 0)
        print(f"  • Best quality: {best_quality.get('strategy_name')} (score: {quality_score:.3f})")

        # Custom algorithm performance
        custom_results = [r for r in all_results if r.get('is_custom_algorithm', False)]
        if custom_results:
            print(f"  • Custom algorithms tested: {len(custom_results)}")
            avg_custom_time = sum(r.get('processing_time', 0) for r in custom_results) / len(custom_results)
            builtin_results = [r for r in all_results if not r.get('is_custom_algorithm', False)]
            if builtin_results:
                avg_builtin_time = sum(r.get('processing_time', 0) for r in builtin_results) / len(builtin_results)
                ratio = avg_custom_time / avg_builtin_time
                print(f"  • Custom vs built-in performance: {ratio:.2f}x")


def main():
    """Run all benchmarking demos."""
    print("🏁 CHUNKING STRATEGY BENCHMARKING DEMO")
    print("This demo showcases the production-ready benchmarking system")
    print()

    # Ensure demo results directory exists
    Path("./demo_results").mkdir(exist_ok=True)

    demos = [
        ("Quick Benchmark", demo_quick_benchmark),
        ("Custom Algorithm Benchmark", demo_custom_algorithm_benchmark),
        ("Comprehensive Benchmark", demo_comprehensive_benchmark)
    ]

    results = {}

    for demo_name, demo_func in demos:
        try:
            print(f"\nRunning {demo_name}...")
            success = demo_func()
            results[demo_name] = success

            if success:
                print(f"✅ {demo_name} completed successfully")
            else:
                print(f"❌ {demo_name} failed")

        except KeyboardInterrupt:
            print(f"\n⏹️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"❌ {demo_name} failed with exception: {e}")
            results[demo_name] = False

    # Analyze results
    try:
        analyze_demo_results()
    except Exception as e:
        print(f"⚠️  Results analysis failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("🏁 DEMO SUMMARY")
    print(f"{'='*60}")

    successful_demos = sum(1 for success in results.values() if success)
    total_demos = len(results)

    print(f"Completed: {successful_demos}/{total_demos} demos")

    for demo_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {demo_name}")

    if successful_demos > 0:
        print(f"\n📁 Results saved to: ./demo_results/")
        print("You can examine the JSON, CSV, and report files for detailed analysis.")

    if successful_demos == total_demos:
        print("\n🎉 All demos completed successfully!")
        print("The production benchmarking system is working correctly.")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demos failed.")
        print("Check the error messages above for troubleshooting information.")


if __name__ == "__main__":
    main()
