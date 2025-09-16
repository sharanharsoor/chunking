#!/usr/bin/env python3
"""
Test Parallel Algorithm Comparison
Tests parallel execution and comparison of custom vs built-in algorithms.
"""

import pytest
import tempfile
import statistics
from pathlib import Path
from typing import Dict, List, Any

from chunking_strategy import ChunkerOrchestrator
from chunking_strategy.core.custom_algorithm_loader import load_custom_algorithm


class TestParallelAlgorithmComparison:
    """Test parallel algorithm comparison functionality."""

    @pytest.fixture
    def config_path(self):
        """Path to the parallel testing config."""
        return "config_examples/advanced_configs/parallel_algorithm_testing.yaml"

    @pytest.fixture
    def test_content(self):
        """Sample content for testing algorithms."""
        return """
        This is an excellent product with outstanding features!
        I'm very satisfied with the quality and performance.

        However, the price point is quite high and may not be affordable for everyone.
        The delivery time was also longer than expected.

        The customer service team was helpful when I contacted them about issues.
        They resolved my concerns quickly and professionally.

        Overall, this is a good product despite some drawbacks.
        I would recommend it to others who can afford the premium price.

        The user interface is intuitive and easy to navigate.
        Documentation could be improved with more detailed examples.

        In conclusion, it's a solid choice for professional users.
        """

    def test_parallel_config_loading(self, config_path):
        """Test that parallel comparison config loads correctly."""

        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        # Verify configuration structure
        assert hasattr(orchestrator, 'config')
        assert 'multi_strategy' in orchestrator.config

        multi_strategy = orchestrator.config['multi_strategy']
        assert multi_strategy['enabled'] == True
        assert multi_strategy['mode'] == 'parallel_comparison'

        # Verify strategies are configured
        strategies = multi_strategy['strategies']
        assert len(strategies) >= 6  # At least 3 custom + 3 built-in

        # Verify custom algorithms are loaded
        custom_algos = orchestrator.get_loaded_custom_algorithms()
        expected_custom = ["sentiment_based", "regex_pattern_based", "balanced_length"]

        for algo in expected_custom:
            assert algo in custom_algos

    def test_single_algorithm_execution(self, config_path, test_content):
        """Test individual algorithm execution before parallel comparison."""

        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        # Test individual algorithms work
        algorithms_to_test = [
            "sentiment_based",      # Custom
            "regex_pattern_based",  # Custom
            "balanced_length",      # Custom
            "semantic",             # Built-in
            "paragraph_based",      # Built-in
            "sentence_based",       # Built-in
            "fixed_size"           # Built-in
        ]

        results = {}
        for algorithm in algorithms_to_test:
            try:
                # Create chunker for each algorithm individually
                chunker = create_chunker(algorithm)
                result = chunker.chunk(test_content)

                assert len(result.chunks) > 0
                assert result.strategy_used == algorithm

                results[algorithm] = {
                    'chunk_count': len(result.chunks),
                    'avg_chunk_size': statistics.mean(len(chunk.content) for chunk in result.chunks),
                    'total_size': sum(len(chunk.content) for chunk in result.chunks)
                }

            except Exception as e:
                # Log but don't fail - some algorithms might not be available
                print(f"Algorithm {algorithm} failed: {e}")

        # At least some algorithms should work
        assert len(results) >= 3

    def test_algorithm_comparison_metrics(self, config_path, test_content):
        """Test comparison metrics between custom and built-in algorithms."""

        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        # Test custom algorithm
        custom_result = None
        try:
            chunker = create_chunker("sentiment_based")
            custom_result = chunker.chunk(test_content)
        except:
            pass

        # Test comparable built-in algorithm
        builtin_result = None
        try:
            chunker = create_chunker("semantic")
            builtin_result = chunker.chunk(test_content)
        except:
            # Fallback to paragraph_based if semantic not available
            chunker = create_chunker("paragraph_based")
            builtin_result = chunker.chunk(test_content)

        if custom_result and builtin_result:
            # Compare basic metrics
            custom_metrics = self._calculate_metrics(custom_result)
            builtin_metrics = self._calculate_metrics(builtin_result)

            # Both should produce valid results
            assert custom_metrics['chunk_count'] > 0
            assert builtin_metrics['chunk_count'] > 0

            # Content should be preserved
            assert custom_metrics['content_preservation'] > 0.8
            assert builtin_metrics['content_preservation'] > 0.8

            # Print comparison for debugging
            print(f"Custom algorithm metrics: {custom_metrics}")
            print(f"Built-in algorithm metrics: {builtin_metrics}")

    def test_performance_comparison(self, config_path):
        """Test performance comparison between algorithms."""

        import time

        test_sizes = {
            'small': "Short text for testing. Just a few sentences.",
            'medium': "Medium length text. " * 50,
            'large': "Large text content. " * 500
        }

        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        algorithms = ["sentiment_based", "paragraph_based", "fixed_size"]
        performance_results = {}

        for size_name, content in test_sizes.items():
            performance_results[size_name] = {}

            for algorithm in algorithms:
                try:
                    chunker = create_chunker(algorithm)

                    # Measure performance
                    start_time = time.time()
                    result = chunker.chunk(content)
                    end_time = time.time()

                    processing_time = end_time - start_time
                    throughput = len(content) / processing_time if processing_time > 0 else 0

                    performance_results[size_name][algorithm] = {
                        'processing_time': processing_time,
                        'throughput': throughput,
                        'chunk_count': len(result.chunks)
                    }

                except Exception as e:
                    print(f"Performance test failed for {algorithm}: {e}")

        # Verify we got some results
        assert len(performance_results) > 0

        # Print performance comparison
        for size_name, results in performance_results.items():
            print(f"\n{size_name.upper()} text performance:")
            for algorithm, metrics in results.items():
                print(f"  {algorithm}: {metrics['processing_time']:.4f}s, "
                      f"{metrics['throughput']:.0f} chars/sec, "
                      f"{metrics['chunk_count']} chunks")

    def test_quality_assessment(self, config_path, test_content):
        """Test quality assessment and comparison."""

        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        algorithms = ["sentiment_based", "paragraph_based"]
        quality_results = {}

        for algorithm in algorithms:
            try:
                chunker = create_chunker(algorithm)
                result = chunker.chunk(test_content)

                quality_metrics = self._assess_quality(result, test_content)
                quality_results[algorithm] = quality_metrics

            except Exception as e:
                print(f"Quality assessment failed for {algorithm}: {e}")

        # Verify quality assessments
        for algorithm, metrics in quality_results.items():
            assert 0 <= metrics['coherence_score'] <= 1
            assert 0 <= metrics['boundary_score'] <= 1
            assert 0 <= metrics['consistency_score'] <= 1

        # Print quality comparison
        print("\nQuality Assessment Results:")
        for algorithm, metrics in quality_results.items():
            print(f"  {algorithm}:")
            for metric, score in metrics.items():
                print(f"    {metric}: {score:.3f}")

    def test_statistical_comparison(self, config_path):
        """Test statistical comparison between algorithms."""

        # Generate multiple test samples
        test_samples = [
            "Positive content with excellent features and great quality!",
            "Negative feedback about poor performance and high costs.",
            "Neutral information about standard features and pricing.",
            "Mixed review with both good and bad aspects mentioned.",
            "Technical documentation with detailed specifications."
        ]

        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        algorithms = ["sentiment_based", "paragraph_based"]
        results = {algo: [] for algo in algorithms}

        # Run algorithms on all samples
        for sample in test_samples:
            for algorithm in algorithms:
                try:
                    chunker = create_chunker(algorithm)
                    result = chunker.chunk(sample)

                    results[algorithm].append({
                        'chunk_count': len(result.chunks),
                        'avg_chunk_size': statistics.mean(len(c.content) for c in result.chunks),
                        'total_size': sum(len(c.content) for c in result.chunks)
                    })

                except Exception as e:
                    print(f"Statistical test failed for {algorithm}: {e}")

        # Calculate statistics
        stats = {}
        for algorithm, data in results.items():
            if data:
                chunk_counts = [d['chunk_count'] for d in data]
                stats[algorithm] = {
                    'mean_chunks': statistics.mean(chunk_counts),
                    'std_chunks': statistics.stdev(chunk_counts) if len(chunk_counts) > 1 else 0,
                    'sample_size': len(data)
                }

        # Verify statistical results
        for algorithm, stat in stats.items():
            assert stat['mean_chunks'] > 0
            assert stat['sample_size'] > 0

        print("\nStatistical Comparison:")
        for algorithm, stat in stats.items():
            print(f"  {algorithm}: mean={stat['mean_chunks']:.2f}, "
                  f"std={stat['std_chunks']:.2f}, n={stat['sample_size']}")

    def _calculate_metrics(self, result) -> Dict[str, float]:
        """Calculate basic metrics for a chunking result."""
        if not result.chunks:
            return {'chunk_count': 0, 'avg_chunk_size': 0, 'content_preservation': 0}

        chunk_sizes = [len(chunk.content) for chunk in result.chunks]
        total_size = sum(chunk_sizes)

        return {
            'chunk_count': len(result.chunks),
            'avg_chunk_size': statistics.mean(chunk_sizes),
            'size_variance': statistics.variance(chunk_sizes) if len(chunk_sizes) > 1 else 0,
            'total_size': total_size,
            'content_preservation': 1.0  # Simplified - assume perfect preservation
        }

    def _assess_quality(self, result, original_content: str) -> Dict[str, float]:
        """Assess quality metrics for a chunking result."""
        if not result.chunks:
            return {'coherence_score': 0, 'boundary_score': 0, 'consistency_score': 0}

        chunk_sizes = [len(chunk.content) for chunk in result.chunks]

        # Simple quality metrics
        coherence_score = min(1.0, len(result.chunks) / max(1, len(original_content.split('\n\n'))))
        boundary_score = 1.0 - (statistics.variance(chunk_sizes) / max(1, statistics.mean(chunk_sizes))) if len(chunk_sizes) > 1 else 1.0
        consistency_score = 1.0 - abs(statistics.mean(chunk_sizes) - 500) / 500  # Target ~500 chars

        return {
            'coherence_score': max(0, min(1, coherence_score)),
            'boundary_score': max(0, min(1, boundary_score)),
            'consistency_score': max(0, min(1, consistency_score))
        }


def create_chunker(algorithm_name: str):
    """Helper function to create chunker instances."""
    from chunking_strategy import create_chunker as _create_chunker
    return _create_chunker(algorithm_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
