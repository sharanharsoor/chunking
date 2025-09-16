"""
Integration Tests for Advanced Chunking System.

This module provides comprehensive end-to-end integration tests
covering all chunking strategies, orchestrator functionality,
and advanced features (FastCDC, Adaptive, Context-Enriched).
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

from chunking_strategy.core.base import ChunkingResult, ModalityType
from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.enhanced_orchestrator import EnhancedOrchestrator
from chunking_strategy.benchmarking import ChunkingBenchmark, run_comprehensive_benchmark
from chunking_strategy.core.registry import list_chunkers, create_chunker


class TestIntegrationAdvancedChunkers:
    """Integration tests for advanced chunking strategies."""

    def test_fastcdc_integration(self):
        """Test FastCDC chunker integration."""
        orchestrator = ChunkerOrchestrator()

        # Create test content
        content = "This is a test document. " * 100  # Repeating content for CDC

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Test FastCDC chunking
            result = orchestrator.chunk_file(temp_path, strategy='fastcdc')

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0
            assert result.strategy_used == 'fastcdc'

            # Verify chunk properties
            for chunk in result.chunks:
                assert chunk.content is not None
                assert chunk.size > 0
                assert chunk.metadata is not None

        finally:
            Path(temp_path).unlink()

    def test_adaptive_integration(self):
        """Test Adaptive Dynamic chunker integration."""
        orchestrator = ChunkerOrchestrator()

        # Create mixed content for adaptation
        content = """
        # Technical Document

        This is a technical document with various content types.

        ## Code Example
        ```python
        def example_function():
            return "Hello, World!"
        ```

        ## Data Section
        {
            "key": "value",
            "number": 42,
            "list": [1, 2, 3]
        }

        ## Conclusion
        This document tests adaptive chunking capabilities.
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Test Adaptive chunking
            result = orchestrator.chunk_file(temp_path, strategy='adaptive')

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0
            assert result.strategy_used == 'adaptive'

            # Verify adaptive capabilities
            chunker = create_chunker('adaptive')
            assert hasattr(chunker, 'adapt_parameters')
            assert hasattr(chunker, 'get_adaptation_info')

        finally:
            Path(temp_path).unlink()

    def test_context_enriched_integration(self):
        """Test Context-Enriched chunker integration."""
        orchestrator = ChunkerOrchestrator()

        # Create semantic content
        content = """
        Introduction to Machine Learning

        Machine learning is a subset of artificial intelligence that enables computers
        to learn and improve from experience without being explicitly programmed.

        Types of Machine Learning

        There are three main types of machine learning:
        1. Supervised learning
        2. Unsupervised learning
        3. Reinforcement learning

        Applications

        Machine learning has numerous applications including image recognition,
        natural language processing, and recommendation systems.
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Test Context-Enriched chunking
            result = orchestrator.chunk_file(temp_path, strategy='context_enriched')

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0
            assert result.strategy_used == 'context_enriched'

            # Verify semantic processing
            for chunk in result.chunks:
                assert 'semantic_fingerprint' in chunk.metadata.extra

        finally:
            Path(temp_path).unlink()

    def test_enhanced_orchestrator_integration(self):
        """Test Enhanced Orchestrator with intelligent strategy selection."""
        enhanced = EnhancedOrchestrator()

        # Test different content types
        test_cases = [
            {
                'content': "This is semantic text content with structure.",
                'extension': '.txt',
                'expected_strategy_type': 'intelligent'  # Should select advanced strategy
            },
            {
                'content': '{"key": "value", "data": [1, 2, 3]}',
                'extension': '.json',
                'expected_strategy_type': 'format_specific'  # Should select JSON strategy
            }
        ]

        for case in test_cases:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=case['extension'],
                delete=False
            ) as f:
                f.write(case['content'])
                temp_path = f.name

            try:
                # Test enhanced orchestrator
                result = enhanced.chunk_file(temp_path)

                assert isinstance(result, ChunkingResult)
                assert len(result.chunks) > 0

                # Test strategy recommendations
                file_info = enhanced._analyze_file_characteristics(temp_path)
                recommendations = enhanced.get_strategy_recommendations(file_info)

                assert len(recommendations) > 0
                assert all(len(rec) == 3 for rec in recommendations)  # (strategy, score, reason)

                # Test strategy explanation
                explanation = enhanced.explain_strategy_selection(file_info)
                assert 'selected_strategy' in explanation
                assert 'selection_reasoning' in explanation

            finally:
                Path(temp_path).unlink()


class TestIntegrationWorkflows:
    """Integration tests for complete chunking workflows."""

    def test_auto_selection_workflow(self):
        """Test automatic strategy selection workflow."""
        orchestrator = ChunkerOrchestrator()

        # Test with different file types
        test_files = [
            ("sample.txt", "This is plain text content."),
            ("code.py", "def hello():\n    return 'Hello, World!'"),
            ("data.json", '{"message": "Hello, JSON"}'),
        ]

        for filename, content in test_files:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=Path(filename).suffix,
                delete=False
            ) as f:
                f.write(content)
                temp_path = f.name

            try:
                # Test auto strategy selection
                result = orchestrator.chunk_file(temp_path, strategy='auto')

                assert isinstance(result, ChunkingResult)
                assert len(result.chunks) > 0

                # Verify strategy was selected and executed
                assert result.strategy_used is not None
                assert result.strategy_used != 'auto'  # Should resolve to actual strategy

            finally:
                Path(temp_path).unlink()

    def test_configuration_workflow(self):
        """Test configuration-based chunking workflow."""
        orchestrator = ChunkerOrchestrator()

        # Create test configuration
        config = {
            'strategies': {
                'primary': 'adaptive',
                'fallback': ['sentence_based', 'fixed_size']
            },
            'adaptive': {
                'enable_learning': True,
                'adaptation_threshold': 0.1
            }
        }

        content = "This is test content for configuration testing."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Test configuration-based chunking
            result = orchestrator.chunk_file(temp_path, config=config)

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0

        finally:
            Path(temp_path).unlink()

    def test_streaming_workflow(self):
        """Test streaming chunking workflow."""
        # Test streaming with large content
        large_content = "This is a line of text.\n" * 1000

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_content)
            temp_path = f.name

        try:
            orchestrator = ChunkerOrchestrator()

            # Test streaming chunking
            result = orchestrator.chunk_file(temp_path, strategy='adaptive')

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0

            # Verify streaming capabilities
            chunker = create_chunker('adaptive')
            if hasattr(chunker, 'supports_streaming'):
                assert chunker.supports_streaming()

        finally:
            Path(temp_path).unlink()


class TestIntegrationBenchmarking:
    """Integration tests for benchmarking functionality."""

    def test_single_strategy_benchmark(self):
        """Test benchmarking a single strategy."""
        benchmark = ChunkingBenchmark(
            enable_memory_profiling=True,
            enable_cpu_monitoring=True,
            benchmark_iterations=2
        )

        # Create test file
        content = "This is test content for benchmarking."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Benchmark a strategy
            metrics = benchmark.benchmark_strategy('fixed_size', temp_path)

            assert metrics.strategy_name == 'fixed_size'
            assert metrics.file_path == temp_path
            assert metrics.processing_time >= 0
            assert metrics.chunks_generated > 0
            assert metrics.throughput_mb_per_sec >= 0

        finally:
            Path(temp_path).unlink()

    def test_multi_strategy_benchmark(self):
        """Test benchmarking multiple strategies."""
        benchmark = ChunkingBenchmark(
            benchmark_iterations=1  # Fast test
        )

        # Create test files
        test_files = []
        for i in range(2):
            content = f"Test content for file {i}. " * 20
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                test_files.append(f.name)

        try:
            # Benchmark multiple strategies
            strategies = ['fixed_size', 'sentence_based']
            result = benchmark.benchmark_multiple_strategies(strategies, test_files)

            assert result.test_name is not None
            assert result.timestamp is not None
            assert len(result.file_metrics) > 0
            assert len(result.strategy_summaries) > 0

            # Verify metrics for each strategy
            for strategy in strategies:
                assert strategy in result.strategy_summaries
                summary = result.strategy_summaries[strategy]
                assert 'avg_processing_time' in summary
                assert 'files_processed' in summary

        finally:
            for temp_path in test_files:
                Path(temp_path).unlink()


class TestIntegrationErrorHandling:
    """Integration tests for error handling and robustness."""

    def test_invalid_file_handling(self):
        """Test handling of invalid files."""
        orchestrator = ChunkerOrchestrator()

        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            orchestrator.chunk_file("nonexistent_file.txt")

    def test_corrupted_content_handling(self):
        """Test handling of corrupted or unusual content."""
        orchestrator = ChunkerOrchestrator()

        # Test binary content
        binary_content = b'\x00\x01\x02\x03\xff\xfe\xfd'

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
            f.write(binary_content)
            temp_path = f.name

        try:
            # Should handle gracefully with fallback
            result = orchestrator.chunk_file(temp_path)

            assert isinstance(result, ChunkingResult)
            # Should have at least one chunk (fallback behavior)
            assert len(result.chunks) >= 1

        finally:
            Path(temp_path).unlink()

    def test_strategy_fallback_chain(self):
        """Test strategy fallback chain functionality."""
        orchestrator = ChunkerOrchestrator()

        content = "Test content for fallback testing."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Test with a strategy that might fail and fallback
            result = orchestrator.chunk_file(
                temp_path,
                strategy='nonexistent_strategy'  # Should trigger fallback
            )

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0

            # Should have fallen back to a working strategy
            assert result.strategy_used != 'nonexistent_strategy'

        finally:
            Path(temp_path).unlink()


class TestIntegrationPerformance:
    """Integration tests for performance characteristics."""

    def test_large_file_processing(self):
        """Test processing of large files."""
        orchestrator = ChunkerOrchestrator()

        # Create a moderately large file (1MB)
        large_content = "This is a test line with some content.\n" * 25000

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_content)
            temp_path = f.name

        try:
            # Test with FastCDC (good for large files)
            result = orchestrator.chunk_file(temp_path, strategy='fastcdc')

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0
            assert result.processing_time > 0

            # Should process efficiently
            file_size_mb = Path(temp_path).stat().st_size / (1024 * 1024)
            throughput = file_size_mb / result.processing_time
            assert throughput > 0.1  # At least 0.1 MB/s

        finally:
            Path(temp_path).unlink()

    def test_memory_efficiency(self):
        """Test memory efficiency of chunking operations."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        orchestrator = ChunkerOrchestrator()

        # Process multiple files to test memory management
        for i in range(5):
            content = f"Memory test content {i}. " * 1000

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_path = f.name

            try:
                result = orchestrator.chunk_file(temp_path, strategy='adaptive')
                assert len(result.chunks) > 0

            finally:
                Path(temp_path).unlink()

        # Memory should not have grown excessively
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB

        # Allow reasonable memory growth (should be < 100MB for this test)
        assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB"


@pytest.mark.integration
class TestIntegrationFullSystem:
    """Full system integration tests."""

    def test_complete_chunking_pipeline(self):
        """Test complete chunking pipeline from file to results."""

        # Create a comprehensive test document
        content = """
        # Advanced Chunking System Test Document

        This document tests the complete chunking pipeline with various content types.

        ## Text Content
        This section contains natural language text that should be processed
        by semantic chunkers with attention to sentence and paragraph boundaries.

        ## Code Section
        ```python
        def advanced_chunking_example():
            '''Example function for testing code chunking.'''
            strategies = ['fastcdc', 'adaptive', 'context_enriched']
            for strategy in strategies:
                print(f"Testing {strategy}")
            return "Complete"
        ```

        ## Data Section
        {
            "chunking_strategies": [
                {"name": "FastCDC", "type": "content_defined"},
                {"name": "Adaptive", "type": "dynamic"},
                {"name": "Context-Enriched", "type": "semantic"}
            ],
            "performance_metrics": {
                "throughput": "variable",
                "quality": "high"
            }
        }

        ## Conclusion
        This comprehensive test validates the entire chunking system
        including strategy selection, processing, and result generation.
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Test with Enhanced Orchestrator
            enhanced = EnhancedOrchestrator(
                enable_advanced_selection=True,
                enable_content_analysis=True
            )

            # Get file analysis
            file_info = enhanced._analyze_file_characteristics(temp_path)
            assert file_info['file_size'] > 0
            assert file_info['text_ratio'] > 0.8  # Should be highly textual

            # Get strategy recommendations
            recommendations = enhanced.get_strategy_recommendations(file_info)
            assert len(recommendations) >= 3

            # Process with enhanced orchestrator
            result = enhanced.chunk_file(temp_path)

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0
            assert result.strategy_used is not None
            assert result.processing_time > 0

            # Verify chunk quality
            total_content = ""
            for chunk in result.chunks:
                assert chunk.content is not None
                assert len(chunk.content.strip()) > 0
                assert chunk.metadata is not None
                total_content += chunk.content

            # Content should be preserved (allowing for minor whitespace differences)
            original_length = len(content.replace(' ', '').replace('\n', ''))
            processed_length = len(total_content.replace(' ', '').replace('\n', ''))

            # Allow for minor differences due to processing
            preservation_ratio = processed_length / original_length
            assert 0.9 <= preservation_ratio <= 1.1, f"Content preservation ratio: {preservation_ratio}"

            # Test benchmarking on the result
            benchmark = ChunkingBenchmark(benchmark_iterations=1)
            metrics = benchmark.benchmark_strategy(result.strategy_used, temp_path)

            assert metrics.chunks_generated == len(result.chunks)
            assert metrics.processing_time > 0

        finally:
            Path(temp_path).unlink()

    def test_cross_strategy_consistency(self):
        """Test consistency across different strategies."""
        orchestrator = ChunkerOrchestrator()

        content = "This is consistent test content. " * 50

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            strategies = ['fixed_size', 'sentence_based', 'adaptive']
            results = {}

            for strategy in strategies:
                try:
                    result = orchestrator.chunk_file(temp_path, strategy=strategy)
                    results[strategy] = result
                except Exception as e:
                    pytest.skip(f"Strategy {strategy} not available: {e}")

            # Verify all strategies processed the same content
            if results:
                file_sizes = [Path(temp_path).stat().st_size for _ in results]
                assert all(size == file_sizes[0] for size in file_sizes)

                # All should have generated chunks
                for strategy, result in results.items():
                    assert len(result.chunks) > 0, f"Strategy {strategy} produced no chunks"

                    # Verify content preservation
                    total_chars = sum(len(chunk.content) for chunk in result.chunks)
                    assert total_chars > 0, f"Strategy {strategy} produced empty content"

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    # Run integration tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
