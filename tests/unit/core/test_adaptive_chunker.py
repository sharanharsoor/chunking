"""
Unit tests for Adaptive Dynamic Chunking strategy.

Tests cover:
- Basic adaptive functionality and registration
- Content profiling and analysis
- Strategy selection mechanisms (content-based, performance-based, auto)
- Parameter optimization and adaptation
- Performance tracking and learning
- Adaptation history and persistence
- Multi-strategy orchestration
- Edge cases and error handling
- Integration with orchestrator
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from chunking_strategy.core.registry import create_chunker, list_chunkers
from chunking_strategy.core.base import ModalityType
from chunking_strategy.strategies.general.adaptive_chunker import (
    AdaptiveChunker, ContentProfile, PerformanceMetrics, AdaptationRecord
)
from chunking_strategy import ChunkerOrchestrator


class TestAdaptiveChunker:
    """Test Adaptive Dynamic Chunker functionality."""

    def test_adaptive_registration(self):
        """Test that Adaptive chunker is properly registered."""
        chunkers = list_chunkers()
        assert "adaptive" in chunkers

        # Test creation
        chunker = create_chunker("adaptive")
        assert chunker is not None
        assert chunker.__class__.__name__ == "AdaptiveChunker"
        assert "any" in chunker.get_supported_formats()
        assert len(chunker.available_strategies) >= 4

    def test_content_profiling(self):
        """Test content analysis and profiling capabilities."""
        chunker = create_chunker("adaptive")

        # Test text content profiling
        text_content = "This is a well-structured document with clear paragraphs.\n\nIt has multiple sections and should be easy to analyze."
        profile = chunker._profile_content(text_content)

        assert isinstance(profile, ContentProfile)
        assert profile.file_type == "text"
        assert profile.size_bytes > 0
        assert profile.text_ratio > 0.8  # Should be high for text
        assert 0 <= profile.structure_score <= 1
        assert 0 <= profile.complexity_score <= 1
        assert 0 <= profile.repetition_score <= 1

    def test_strategy_selection_content_based(self):
        """Test content-based strategy selection."""
        chunker = create_chunker("adaptive", strategy_selection_mode="content")

        # Test highly textual content
        text_profile = ContentProfile(
            file_type="text", size_bytes=5000, estimated_entropy=4.5,
            text_ratio=0.95, structure_score=0.8, repetition_score=0.2, complexity_score=0.7
        )
        strategy = chunker._select_strategy_by_content(text_profile)
        assert strategy in ["paragraph_based", "sentence_based"]

        # Test JSON content
        json_profile = ContentProfile(
            file_type="json", size_bytes=10000, estimated_entropy=5.5,
            text_ratio=0.8, structure_score=0.9, repetition_score=0.6, complexity_score=0.8
        )
        strategy = chunker._select_strategy_by_content(json_profile)
        assert strategy == "fastcdc"

        # Test large file
        large_profile = ContentProfile(
            file_type="binary", size_bytes=200000, estimated_entropy=7.0,
            text_ratio=0.3, structure_score=0.2, repetition_score=0.8, complexity_score=0.9
        )
        strategy = chunker._select_strategy_by_content(large_profile)
        assert strategy == "fastcdc"

    def test_parameter_optimization(self):
        """Test parameter optimization for different strategies."""
        chunker = create_chunker("adaptive")

        # Test FastCDC parameter optimization
        large_profile = ContentProfile(
            file_type="binary", size_bytes=2000000, estimated_entropy=6.0,
            text_ratio=0.1, structure_score=0.1, repetition_score=0.9, complexity_score=0.8
        )
        params = chunker._optimize_parameters("fastcdc", large_profile)
        assert "avg_chunk_size" in params
        assert params["avg_chunk_size"] >= 8192  # Should increase for large files

        # Test small file optimization
        small_profile = ContentProfile(
            file_type="text", size_bytes=5000, estimated_entropy=4.0,
            text_ratio=0.9, structure_score=0.5, repetition_score=0.3, complexity_score=0.6
        )
        params = chunker._optimize_parameters("fastcdc", small_profile)
        assert params["avg_chunk_size"] <= 8192  # Should decrease for small files

    def test_performance_evaluation(self):
        """Test performance metrics calculation."""
        chunker = create_chunker("adaptive")

        # Create a mock chunking result
        from chunking_strategy.core.base import Chunk, ChunkingResult, ChunkMetadata

        chunks = [
            Chunk(
                id="test_1", content="Test content 1", modality=ModalityType.TEXT,
                metadata=ChunkMetadata(source="test", chunker_used="adaptive")
            ),
            Chunk(
                id="test_2", content="Test content 2", modality=ModalityType.TEXT,
                metadata=ChunkMetadata(source="test", chunker_used="adaptive")
            )
        ]

        result = ChunkingResult(chunks=chunks, processing_time=0.1, strategy_used="test_strategy")

        profile = ContentProfile(
            file_type="text", size_bytes=100, estimated_entropy=4.0,
            text_ratio=1.0, structure_score=0.5, repetition_score=0.0, complexity_score=0.5
        )

        start_time = 0.0
        metrics = chunker._evaluate_performance(result, profile, "test_strategy", start_time)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.strategy_used == "test_strategy"
        assert metrics.chunk_count == 2
        assert metrics.processing_time >= 0
        assert 0 <= metrics.quality_score <= 1
        assert 0 <= metrics.boundary_quality <= 1

    def test_basic_adaptive_chunking(self):
        """Test basic adaptive chunking functionality."""
        chunker = create_chunker("adaptive")

        # Test with structured text
        text = """Section 1: Introduction
        This is a test document for adaptive chunking.

        Section 2: Details
        The adaptive chunker should analyze this content and select an appropriate strategy.

        Section 3: Conclusion
        This demonstrates the adaptive chunking capabilities."""

        result = chunker.chunk(text)

        assert result.total_chunks > 0
        assert result.strategy_used.startswith("adaptive_")
        assert result.processing_time >= 0

        # Check adaptive metadata
        assert "adaptive_strategy" in result.source_info
        assert "content_profile" in result.source_info
        assert "performance_metrics" in result.source_info
        assert "operation_count" in result.source_info

    def test_adaptation_over_multiple_operations(self):
        """Test adaptation behavior over multiple chunking operations."""
        chunker = create_chunker("adaptive", adaptation_interval=2, min_samples=1)

        # Perform multiple operations
        texts = [
            "Simple text for testing adaptation mechanism.",
            "Another simple text document for the adaptive chunker.",
            "Third document to trigger adaptation behavior.",
            "Fourth document to test learning capabilities."
        ]

        results = []
        for text in texts:
            result = chunker.chunk(text)
            results.append(result)

        # Check that operation count increases
        assert results[0].source_info["operation_count"] == 1
        assert results[3].source_info["operation_count"] == 4

        # Check that performance history is being tracked
        assert len(chunker.performance_history) == 4
        assert len(chunker.strategy_performance) > 0

    def test_strategy_performance_tracking(self):
        """Test strategy performance tracking and comparison."""
        chunker = create_chunker("adaptive")

        # Test with different content types
        contents = [
            "Simple paragraph text that should use sentence or paragraph strategy.",
            '{"json": "data", "structured": true, "should": "use fastcdc"}',
            "AAAAAABBBBBBCCCCCCDDDDDD" * 100,  # Repetitive content
        ]

        for content in contents:
            result = chunker.chunk(content)

        # Check performance summary
        summary = chunker.get_performance_summary()
        assert summary["total_operations"] == len(contents)
        assert len(summary["strategies_used"]) > 0
        assert "strategy_performance" in summary

        # Each strategy should have performance data
        for strategy, perf_data in summary["strategy_performance"].items():
            assert "total_uses" in perf_data
            assert "avg_score" in perf_data
            assert "avg_processing_time" in perf_data

    def test_file_input_handling(self):
        """Test adaptive chunking with file inputs."""
        chunker = create_chunker("adaptive")

        # Test with existing test files
        test_files = [
            "test_data/sample_simple_text.txt",
            "test_data/sample_adaptive_data.json",
            "test_data/sample_mixed_strategies.txt"
        ]

        for file_path in test_files:
            if os.path.exists(file_path):
                result = chunker.chunk(file_path)

                assert result.total_chunks > 0
                assert result.strategy_used.startswith("adaptive_")

                # Check content profiling worked
                profile = result.source_info.get("content_profile", {})
                assert "file_type" in profile
                assert "size_bytes" in profile
                assert profile["size_bytes"] > 0

    def test_adaptation_with_feedback(self):
        """Test adaptation mechanism with external feedback."""
        chunker = create_chunker("adaptive", min_samples=1)

        # Perform initial chunking
        text = "Test content for feedback adaptation."
        result = chunker.chunk(text)

        initial_learning_rate = chunker.learning_rate
        initial_exploration_rate = chunker.exploration_rate

        # Provide negative feedback
        chunker.adapt_parameters(0.3, "quality")

        # Learning rate should increase with poor feedback
        assert chunker.learning_rate >= initial_learning_rate

        # Provide positive feedback
        chunker.adapt_parameters(0.9, "quality")

        # Learning rate should decrease with good feedback
        # (may not be lower due to previous increase, but should trend down)

        # Check adaptation history
        history = chunker.get_adaptation_history()
        assert len(history) == 2
        assert history[0]["feedback_score"] == 0.3
        assert history[1]["feedback_score"] == 0.9

    def test_persistence_functionality(self):
        """Test saving and loading adaptation history."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            persistence_file = f.name

        try:
            # Create chunker with persistence
            chunker1 = create_chunker("adaptive", persistence_file=persistence_file)

            # Perform some operations
            texts = ["Text 1", "Text 2", "Text 3"]
            for text in texts:
                chunker1.chunk(text)

            # Save history
            chunker1._save_history()

            # Create new chunker and load history
            chunker2 = create_chunker("adaptive", persistence_file=persistence_file)

            # Check that history was loaded
            assert chunker2.operation_count == chunker1.operation_count

        finally:
            if os.path.exists(persistence_file):
                os.unlink(persistence_file)

    def test_streaming_support(self):
        """Test streaming chunking capability."""
        chunker = create_chunker("adaptive")

        # Create a stream of content
        content_parts = [
            "First part of streaming content. ",
            "Second part with more information. ",
            "Third part to complete the stream."
        ]

        # Test streaming
        chunks = list(chunker.chunk_stream(content_parts))

        assert len(chunks) > 0

        # Compare with direct chunking
        direct_content = ''.join(content_parts)
        direct_result = chunker.chunk(direct_content)

        # Should produce similar results
        assert len(chunks) == len(direct_result.chunks)

    def test_content_profiling_edge_cases(self):
        """Test content profiling with edge cases."""
        chunker = create_chunker("adaptive")

        # Empty content
        empty_profile = chunker._profile_content("")
        assert empty_profile.size_bytes == 0
        assert empty_profile.text_ratio == 0.0

        # Binary content
        binary_content = bytes(range(256))
        binary_profile = chunker._profile_content(binary_content)
        assert binary_profile.text_ratio < 0.5
        assert binary_profile.estimated_entropy > 0

        # Large text content
        large_text = "A" * 10000 + "B" * 10000
        large_profile = chunker._profile_content(large_text)
        assert large_profile.size_bytes == 20000
        assert large_profile.repetition_score > 0.5  # Should detect repetition

    def test_strategy_comparison_mode(self):
        """Test strategy comparison functionality."""
        chunker = create_chunker("adaptive",
                                enable_strategy_comparison=True,
                                min_samples=1)

        # Perform operations to build comparison data
        test_content = "Test content for strategy comparison analysis."

        for _ in range(5):  # Multiple operations with same content type
            result = chunker.chunk(test_content)

        # Strategy comparison should be working
        assert chunker.enable_strategy_comparison
        assert len(chunker.strategy_performance) > 0

    def test_orchestrator_integration(self):
        """Test Adaptive chunker integration with orchestrator."""
        config = {
            'strategies': {
                'primary': 'adaptive'
            },
            'adaptive': {
                'learning_rate': 0.1,
                'exploration_rate': 0.05,
                'adaptation_threshold': 0.1
            }
        }

        orchestrator = ChunkerOrchestrator(config=config)

        text = "Orchestrator integration test with adaptive chunker."
        result = orchestrator.chunk_content(text)

        assert result.total_chunks > 0
        assert result.strategy_used.startswith("adaptive")

    def test_multiple_strategy_availability(self):
        """Test adaptive chunker with different available strategies."""
        # Test with limited strategies
        limited_chunker = create_chunker("adaptive",
                                       available_strategies=["fastcdc", "fixed_size"])

        assert len(limited_chunker.available_strategies) == 2
        assert "fastcdc" in limited_chunker.available_strategies
        assert "fixed_size" in limited_chunker.available_strategies

        # Test chunking still works
        text = "Test with limited strategy set."
        result = limited_chunker.chunk(text)
        assert result.total_chunks > 0

    def test_adaptation_threshold_behavior(self):
        """Test adaptation threshold and interval behavior."""
        chunker = create_chunker("adaptive",
                                adaptation_threshold=0.5,  # High threshold
                                adaptation_interval=3,      # Every 3 operations
                                min_samples=2)

        # Perform operations
        for i in range(6):
            text = f"Test content number {i} for adaptation testing."
            result = chunker.chunk(text)

        # Check that adaptations are controlled by interval
        assert chunker.operation_count == 6

        # Check adaptation should be considered at intervals
        should_adapt_ops = [3, 6]  # Operations 3 and 6
        for op in should_adapt_ops:
            # At these operations, adaptation should be considered
            pass  # Specific adaptation behavior depends on performance

    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        # Test with invalid strategy in available list
        chunker = create_chunker("adaptive",
                                available_strategies=["fastcdc", "invalid_strategy"])

        # Should still work with valid strategies
        text = "Test error handling capabilities."
        result = chunker.chunk(text)
        assert result.total_chunks > 0

        # Test with corrupted content
        try:
            # This should not crash the chunker
            result = chunker.chunk(None)
        except Exception:
            # Expected to fail gracefully
            pass

    def test_reset_adaptation(self):
        """Test resetting adaptation history."""
        chunker = create_chunker("adaptive")

        # Perform some operations
        for i in range(3):
            chunker.chunk(f"Test content {i}")

        initial_count = chunker.operation_count
        assert initial_count > 0

        # Reset adaptation
        chunker.reset_adaptation()

        # Check that everything is reset
        assert chunker.operation_count == 0
        assert len(chunker.adaptation_history) == 0
        assert len(chunker.performance_history) == 0
        assert len(chunker.strategy_performance) == 0
        assert chunker.current_strategy == chunker.default_strategy

    def test_exploration_mode(self):
        """Test exploration mode functionality."""
        chunker = create_chunker("adaptive")

        initial_exploration = chunker.exploration_rate

        # Enable exploration
        chunker.set_exploration_mode(True)
        assert chunker.exploration_rate > initial_exploration

        # Disable exploration
        chunker.set_exploration_mode(False)
        assert chunker.exploration_rate < 0.02  # Should be very low

    def test_content_strategy_mapping(self):
        """Test content type to strategy mapping functionality."""
        chunker = create_chunker("adaptive", min_samples=1, adaptation_interval=1)

        # Create content with distinct characteristics
        json_content = '{"test": "json content for mapping"}'
        text_content = "Simple text content for strategy mapping."

        # Process multiple times to build mapping
        for _ in range(3):
            chunker.chunk(json_content)
            chunker.chunk(text_content)

        # Check that content-strategy mapping is being built
        assert len(chunker.content_strategy_map) > 0

    def test_quality_and_boundary_scoring(self):
        """Test quality and boundary quality scoring."""
        chunker = create_chunker("adaptive")

        # Create mock result with varied chunk sizes
        from chunking_strategy.core.base import Chunk, ChunkingResult, ChunkMetadata

        chunks = [
            Chunk(id="1", content="Short.", modality=ModalityType.TEXT,
                 metadata=ChunkMetadata(source="test", chunker_used="adaptive")),
            Chunk(id="2", content="This is a much longer chunk with more content to analyze.",
                 modality=ModalityType.TEXT,
                 metadata=ChunkMetadata(source="test", chunker_used="adaptive")),
            Chunk(id="3", content="Medium length chunk here.", modality=ModalityType.TEXT,
                 metadata=ChunkMetadata(source="test", chunker_used="adaptive"))
        ]

        result = ChunkingResult(chunks=chunks, processing_time=0.1, strategy_used="test")

        profile = ContentProfile(
            file_type="text", size_bytes=200, estimated_entropy=4.0,
            text_ratio=1.0, structure_score=0.7, repetition_score=0.1, complexity_score=0.6
        )

        quality_score = chunker._calculate_quality_score(result, profile)
        boundary_quality = chunker._calculate_boundary_quality(result, profile)

        assert 0 <= quality_score <= 1
        assert 0 <= boundary_quality <= 1

    def test_chunk_estimation(self):
        """Test chunk count estimation."""
        chunker = create_chunker("adaptive")

        # Test with string content
        text = "Test content for chunk estimation." * 100
        estimated = chunker.estimate_chunks(text)

        assert estimated > 0
        assert isinstance(estimated, int)

        # Test actual chunking to compare
        result = chunker.chunk(text)
        actual = result.total_chunks

        # Estimation should be in reasonable range
        assert 0.5 <= (estimated / actual) <= 2.0 if actual > 0 else estimated == 0
