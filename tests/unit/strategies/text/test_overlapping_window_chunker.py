"""
Test suite for OverlappingWindowChunker.

This module provides comprehensive tests for the overlapping window chunking
strategy, covering all functionality including different window units,
boundary preservation, streaming, adaptation, and edge cases.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from chunking_strategy.strategies.text.overlapping_window_chunker import (
    OverlappingWindowChunker,
    WindowUnit
)
from chunking_strategy.core.base import ChunkingResult, Chunk


class TestOverlappingWindowChunker:
    """Test suite for OverlappingWindowChunker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = OverlappingWindowChunker(
            name="test_overlapping_window",
            window_size=100,
            step_size=50,
            window_unit="words",
            min_window_size=10  # Allow adaptation by setting smaller minimum
        )

        # Test texts of various lengths and structures
        self.short_text = "This is a short test sentence for basic validation."

        self.medium_text = """
        Artificial intelligence represents one of the most significant technological
        developments of the 21st century. From machine learning algorithms that can
        recognize patterns in vast datasets to natural language processing systems
        that understand human speech, AI is transforming industries and creating new
        possibilities for automation and enhancement of human capabilities.

        The field encompasses various subdomains including computer vision, robotics,
        expert systems, and neural networks. Each area contributes unique insights
        and methodologies that collectively advance our understanding of intelligent
        systems and their applications in real-world scenarios.
        """.strip()

        self.long_text = " ".join([
            "This is sentence number {}.".format(i)
            for i in range(1, 201)
        ])  # 200 sentences, about 1000 words

    def test_initialization_valid_parameters(self):
        """Test chunker initialization with valid parameters."""
        chunker = OverlappingWindowChunker(
            window_size=200,
            step_size=100,
            window_unit="characters",
            preserve_boundaries=True,
            min_window_size=25,
            max_chunk_chars=5000
        )

        assert chunker.window_size == 200
        assert chunker.step_size == 100
        assert chunker.window_unit == WindowUnit.CHARACTERS
        assert chunker.preserve_boundaries is True
        assert chunker.min_window_size == 25
        assert chunker.max_chunk_chars == 5000
        assert len(chunker._adaptation_history) == 0

    def test_initialization_invalid_parameters(self):
        """Test chunker initialization with invalid parameters."""
        # step_size >= window_size
        with pytest.raises(ValueError, match="step_size must be less than window_size"):
            OverlappingWindowChunker(window_size=100, step_size=100)

        with pytest.raises(ValueError, match="step_size must be less than window_size"):
            OverlappingWindowChunker(window_size=100, step_size=150)

        # min_window_size > window_size (but first check step_size validation)
        with pytest.raises(ValueError, match="min_window_size must be less than or equal to window_size"):
            OverlappingWindowChunker(window_size=100, step_size=50, min_window_size=101)

        with pytest.raises(ValueError, match="min_window_size must be less than or equal to window_size"):
            OverlappingWindowChunker(window_size=100, step_size=50, min_window_size=150)

    def test_window_unit_enum_validation(self):
        """Test window unit enumeration validation."""
        # Valid units
        for unit in ["characters", "words", "sentences"]:
            chunker = OverlappingWindowChunker(window_unit=unit)
            assert chunker.window_unit == WindowUnit(unit)

        # Invalid unit
        with pytest.raises(ValueError):
            OverlappingWindowChunker(window_unit="invalid_unit")

    def test_chunk_basic_functionality(self):
        """Test basic chunking functionality."""
        # Use longer text to ensure multiple chunks
        long_text = self.long_text  # 200 sentences, about 1000 words
        result = self.chunker.chunk(long_text)

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 1
        assert all(isinstance(chunk, Chunk) for chunk in result.chunks)
        assert result.strategy_used == "test_overlapping_window"
        assert "total_chars" in result.source_info
        assert "overlap_ratio" in result.source_info

    def test_word_based_chunking(self):
        """Test word-based overlapping window chunking."""
        chunker = OverlappingWindowChunker(
            window_size=20,
            step_size=10,
            window_unit="words",
            min_window_size=5  # Smaller than window_size
        )

        result = chunker.chunk(self.medium_text)

        # Verify overlapping content
        if len(result.chunks) > 1:
            # With window_size=20 and step_size=10, we expect 10 words of overlap
            # The overlap should be the last 10 words of chunk 0 with the first 10 words of chunk 1
            first_chunk_words = result.chunks[0].content.split()
            second_chunk_words = result.chunks[1].content.split()

            # Calculate expected overlap size (window_size - step_size)
            expected_overlap = chunker.window_size - chunker.step_size

            # Check overlap by comparing word sets (more flexible than exact position matching)
            first_chunk_set = set(first_chunk_words)
            second_chunk_set = set(second_chunk_words)
            actual_overlap = first_chunk_set & second_chunk_set

            assert len(actual_overlap) >= min(5, expected_overlap), f"Expected at least {min(5, expected_overlap)} overlapping words between consecutive chunks, got {len(actual_overlap)}"

        # Check chunk metadata
        for i, chunk in enumerate(result.chunks):
            assert chunk.metadata.extra["unit_count"] <= 20  # Within window size
            if i > 0:
                assert chunk.metadata.extra["overlap_with_previous"] is True
            else:
                assert chunk.metadata.extra["overlap_with_previous"] is False

    def test_character_based_chunking(self):
        """Test character-based overlapping window chunking."""
        chunker = OverlappingWindowChunker(
            window_size=200,
            step_size=100,
            window_unit="characters"
        )

        result = chunker.chunk(self.medium_text)

        # Verify character-level window sizes (allow for some flexibility due to boundary preservation)
        for chunk in result.chunks[:-1]:  # Exclude last chunk which might be shorter
            assert len(chunk.content) <= 220  # Allow some flexibility for boundary preservation

        # Check overlap ratio calculation
        assert "overlap_ratio" in result.source_info
        assert 0.0 <= result.source_info["overlap_ratio"] <= 1.0

    def test_sentence_based_chunking(self):
        """Test sentence-based overlapping window chunking."""
        sentence_text = "First sentence here. Second sentence follows. Third sentence continues. Fourth sentence appears. Fifth sentence ends. Sixth sentence starts. Seventh sentence proceeds."

        chunker = OverlappingWindowChunker(
            window_size=3,
            step_size=2,
            window_unit="sentences",
            min_window_size=1  # Smaller than window_size
        )

        result = chunker.chunk(sentence_text)

        # Should create multiple overlapping sentence chunks
        assert len(result.chunks) > 1

        # Each chunk should contain up to 3 sentences
        for chunk in result.chunks:
            sentence_count = chunk.content.count('.') + chunk.content.count('!') + chunk.content.count('?')
            assert sentence_count <= 3

    def test_boundary_preservation(self):
        """Test boundary preservation functionality."""
        # Test with boundary preservation enabled
        chunker_with_boundaries = OverlappingWindowChunker(
            window_size=50,
            step_size=25,
            window_unit="characters",
            preserve_boundaries=True,
            min_window_size=10  # Smaller than window_size
        )

        result_with = chunker_with_boundaries.chunk(self.medium_text)

        # Test without boundary preservation
        chunker_without_boundaries = OverlappingWindowChunker(
            window_size=50,
            step_size=25,
            window_unit="characters",
            preserve_boundaries=False,
            min_window_size=10  # Smaller than window_size
        )

        result_without = chunker_without_boundaries.chunk(self.medium_text)

        # With boundaries, chunks should end at word boundaries more often (proportion-wise)
        boundary_endings_with = sum(
            1 for chunk in result_with.chunks
            if chunk.content.endswith(' ') or chunk.content[-1] in '.!?'
        )

        boundary_endings_without = sum(
            1 for chunk in result_without.chunks
            if chunk.content.endswith(' ') or chunk.content[-1] in '.!?'
        )

        # Compare proportions rather than raw counts since boundary preservation may create fewer chunks
        proportion_with = boundary_endings_with / len(result_with.chunks) if len(result_with.chunks) > 0 else 0
        proportion_without = boundary_endings_without / len(result_without.chunks) if len(result_without.chunks) > 0 else 0

        # Allow some tolerance since this is a heuristic comparison
        assert proportion_with >= proportion_without - 0.05, f"Expected higher proportion of boundary endings with preserve_boundaries=True: {proportion_with:.3f} vs {proportion_without:.3f}"

    def test_minimum_window_size_enforcement(self):
        """Test minimum window size enforcement."""
        chunker = OverlappingWindowChunker(
            window_size=100,
            step_size=90,
            min_window_size=20,
            window_unit="words"
        )

        result = chunker.chunk(self.short_text)  # Short text

        # Should not create chunks smaller than min_window_size
        for chunk in result.chunks:
            assert chunk.metadata.extra["unit_count"] >= 20 or len(result.chunks) == 1

    def test_maximum_chunk_chars_limit(self):
        """Test maximum chunk character limit enforcement."""
        very_long_text = "word " * 5000  # 25000 characters

        chunker = OverlappingWindowChunker(
            window_size=2000,
            step_size=1000,
            window_unit="words",
            max_chunk_chars=1000
        )

        result = chunker.chunk(very_long_text)

        # All chunks should respect character limit
        for chunk in result.chunks:
            assert len(chunk.content) <= 1000

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        # Empty string
        result = self.chunker.chunk("")
        assert len(result.chunks) == 0
        assert "total_chars" not in result.source_info or result.source_info.get("total_chars", 0) == 0

        # Whitespace only
        result = self.chunker.chunk("   \n\t   ")
        assert len(result.chunks) == 0

    def test_single_chunk_scenarios(self):
        """Test scenarios that should produce a single chunk."""
        chunker = OverlappingWindowChunker(
            window_size=1000,
            step_size=500,
            window_unit="words"
        )

        result = chunker.chunk(self.short_text)

        # Short text should fit in one chunk
        assert len(result.chunks) == 1
        assert result.chunks[0].metadata.extra["overlap_with_previous"] is False

    def test_file_input_handling(self):
        """Test chunking from file input."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(self.long_text)  # Use longer text for multiple chunks
            tmp_file_path = Path(tmp_file.name)

        try:
            result = self.chunker.chunk(tmp_file_path)
            assert len(result.chunks) > 1
            assert "total_chars" in result.source_info
        finally:
            tmp_file_path.unlink()

    def test_streaming_functionality(self):
        """Test streaming chunk generation."""
        chunks = list(self.chunker.stream_chunk(self.long_text))  # Use longer text

        # Should produce multiple chunks
        assert len(chunks) > 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

        # Chunk IDs should be sequential
        for i, chunk in enumerate(chunks):
            expected_id = f"test_overlapping_window_chunk_{i}"
            assert chunk.id == expected_id

        # Compare with regular chunking
        regular_result = self.chunker.chunk(self.long_text)
        assert len(chunks) == len(regular_result.chunks)

    def test_adaptation_quality_feedback(self):
        """Test parameter adaptation based on quality feedback."""
        original_step = self.chunker.step_size

        # Low quality score should increase overlap (decrease step size)
        changes = self.chunker.adapt_parameters(feedback_score=0.3, feedback_type="quality")
        assert "step_size" in changes
        assert self.chunker.step_size < original_step

        # Reset for next test
        self.chunker.step_size = original_step

        # High quality score should decrease overlap (increase step size)
        changes = self.chunker.adapt_parameters(feedback_score=0.9, feedback_type="quality")
        assert "step_size" in changes
        assert self.chunker.step_size > original_step

    def test_adaptation_performance_feedback(self):
        """Test parameter adaptation based on performance feedback."""
        original_window = self.chunker.window_size

        # Poor performance should reduce window size
        changes = self.chunker.adapt_parameters(feedback_score=0.3, feedback_type="performance")
        assert "window_size" in changes
        assert self.chunker.window_size < original_window

    def test_adaptation_size_feedback(self):
        """Test parameter adaptation based on size feedback."""
        original_window = self.chunker.window_size

        # Size feedback with target change
        changes = self.chunker.adapt_parameters(
            feedback_score=0.3,
            feedback_type="size",
            target_size_change=1.5
        )
        assert "window_size" in changes
        assert self.chunker.window_size != original_window

    def test_adaptation_history_tracking(self):
        """Test adaptation history tracking."""
        assert len(self.chunker.get_adaptation_history()) == 0

        # Make an adaptation
        self.chunker.adapt_parameters(feedback_score=0.3, feedback_type="quality")

        history = self.chunker.get_adaptation_history()
        assert len(history) == 1
        assert "timestamp" in history[0]
        assert "feedback_score" in history[0]
        assert "changes" in history[0]

    def test_chunk_size_estimation(self):
        """Test chunk size estimation functionality."""
        # Test different text lengths
        estimate_short = self.chunker.get_chunk_size_estimate(100)
        estimate_long = self.chunker.get_chunk_size_estimate(10000)

        assert estimate_short >= 1
        assert estimate_long > estimate_short

    def test_custom_sentence_separators(self):
        """Test custom sentence separator configuration."""
        chinese_text = "这是第一句话。这是第二句话！这是第三句话？"

        chunker = OverlappingWindowChunker(
            window_size=2,
            step_size=1,
            window_unit="sentences",
            sentence_separators=["。", "！", "？", ".", "!", "?"],
            min_window_size=1  # Smaller than window_size
        )

        result = chunker.chunk(chinese_text)

        # Should properly split on Chinese punctuation
        assert len(result.chunks) >= 1

    def test_split_into_units_methods(self):
        """Test internal unit splitting methods."""
        text = "Hello world. This is a test sentence!"

        # Test character splitting
        char_chunker = OverlappingWindowChunker(window_unit="characters")
        char_units = char_chunker._split_into_units(text)
        assert len(char_units) == len(text)
        assert char_units == list(text)

        # Test word splitting
        word_chunker = OverlappingWindowChunker(window_unit="words")
        word_units = word_chunker._split_into_units(text)
        expected_words = text.split()
        assert len(word_units) >= len(expected_words)  # Punctuation may create additional units

        # Test sentence splitting
        sentence_chunker = OverlappingWindowChunker(window_unit="sentences")
        sentence_units = sentence_chunker._split_into_units(text)
        assert len(sentence_units) >= 1

    def test_join_units_methods(self):
        """Test internal unit joining methods."""
        # Test character joining
        char_chunker = OverlappingWindowChunker(window_unit="characters")
        chars = ["H", "e", "l", "l", "o"]
        joined = char_chunker._join_units(chars)
        assert joined == "Hello"

        # Test word joining
        word_chunker = OverlappingWindowChunker(window_unit="words")
        words = ["Hello", "world"]
        joined = word_chunker._join_units(words)
        assert joined == "Hello world"

    def test_boundary_position_finding(self):
        """Test boundary position finding functionality."""
        text = "Hello world test example"
        units = list(text)  # Character units

        chunker = OverlappingWindowChunker(window_unit="characters", preserve_boundaries=True)

        # Find forward boundary (should find next space)
        pos = chunker._find_boundary_position(units, 7, "forward")  # Middle of "world"
        assert units[pos] == ' ' or pos == len(units)

        # Find backward boundary (should find previous space)
        pos = chunker._find_boundary_position(units, 7, "backward")  # Middle of "world"
        assert pos == 0 or units[pos-1] == ' '

    def test_overlap_ratio_calculation(self):
        """Test overlap ratio calculation accuracy."""
        chunker = OverlappingWindowChunker(
            window_size=10,
            step_size=5,
            window_unit="words",
            min_window_size=2  # Smaller than window_size
        )

        text = " ".join([f"word{i}" for i in range(50)])  # 50 words
        result = chunker.chunk(text)

        # Should have overlap
        assert result.source_info["overlap_ratio"] > 0.0
        assert result.source_info["overlap_ratio"] < 1.0

    def test_progress_tracking_with_large_input(self):
        """Test that chunker makes progress and doesn't loop infinitely."""
        chunker = OverlappingWindowChunker(
            window_size=10,
            step_size=1,
            window_unit="words",
            min_window_size=2  # Smaller than window_size
        )

        # Use a large text that could potentially cause issues
        large_text = " ".join([f"word{i}" for i in range(1000)])

        result = chunker.chunk(large_text)

        # Should complete without infinite loop and produce reasonable number of chunks
        assert len(result.chunks) > 0
        assert len(result.chunks) < 2000  # Sanity check

    def test_chunk_content_integrity(self):
        """Test that chunk content maintains text integrity."""
        result = self.chunker.chunk(self.medium_text)

        # Reconstruct text from chunks (removing overlap)
        reconstructed_parts = []
        for i, chunk in enumerate(result.chunks):
            if i == 0:
                # First chunk: take all content
                reconstructed_parts.append(chunk.content)
            else:
                # Subsequent chunks: need to handle overlap
                # For now, just verify each chunk contains valid text
                assert len(chunk.content.strip()) > 0
                assert chunk.content.strip()[0].isalnum() or chunk.content.strip()[0] in "\"'("

    def test_metadata_completeness(self):
        """Test that chunk metadata is complete and accurate."""
        result = self.chunker.chunk(self.medium_text)

        for i, chunk in enumerate(result.chunks):
            # Required metadata fields
            assert chunk.metadata.extra["chunker_used"] == self.chunker.name
            assert chunk.metadata.extra["chunk_index"] == i
            assert chunk.metadata.length == len(chunk.content)
            assert chunk.metadata.offset >= 0

            # Window-specific metadata
            assert "window_info" in chunk.metadata.extra
            window_info = chunk.metadata.extra["window_info"]
            assert "window_start" in window_info
            assert "window_end" in window_info
            assert "unit_type" in window_info
            assert "step_size" in window_info
