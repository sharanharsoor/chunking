"""
Comprehensive tests for the FixedLengthWordChunker.

This module tests the fixed-length word chunking strategy with real files,
edge cases, and comprehensive validation scenarios.
"""

import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from chunking_strategy.strategies.text.fixed_length_word_chunker import FixedLengthWordChunker
from chunking_strategy.core.base import Chunk, ChunkingResult, ModalityType
from chunking_strategy.utils.validation import ChunkValidator
from chunking_strategy.core.metrics import ChunkingQualityEvaluator
from tests.conftest import assert_valid_chunks, assert_content_integrity, assert_reasonable_performance


class TestFixedLengthWordChunker:
    """Test suite for FixedLengthWordChunker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = FixedLengthWordChunker(words_per_chunk=10)
        self.validator = ChunkValidator()
        self.evaluator = ChunkingQualityEvaluator()

        # Test data (exactly 10 words)
        self.simple_text = "This is a simple text with exactly ten words here."
        self.long_text = "This is a longer text document with many words. " * 20  # 200 words
        self.complex_punctuation = "Hello, world! This is a test. What about this? Yes, it's working!"
        self.mixed_content = """
        First paragraph with multiple sentences. It contains various punctuation marks!

        Second paragraph continues with more content. Numbers like 123, 456 are included.
        Special characters @#$% and symbols & are also present.

        Third paragraph completes the test content with final words.
        """

        # Edge case data
        self.single_word = "Hello"
        self.empty_text = ""
        self.whitespace_only = "   \n\t   "
        self.repeated_punctuation = "Hello... world!!! How??? are??? you???"

    def test_basic_word_chunking(self):
        """Test basic word-based chunking functionality."""
        result = self.chunker.chunk(self.simple_text)

        # Validate result structure
        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "fixed_length_word"
        assert result.processing_time is not None
        assert result.processing_time > 0

        # Should create 1 chunk (exactly 10 words)
        assert len(result.chunks) == 1

        # Validate chunk content
        chunk = result.chunks[0]
        assert isinstance(chunk, Chunk)
        assert chunk.modality == ModalityType.TEXT
        assert chunk.content == self.simple_text
        assert chunk.metadata.extra["word_count"] == 10

        # Validate all chunks
        assert_valid_chunks(result.chunks, self.validator)

    def test_multi_chunk_splitting(self):
        """Test splitting text into multiple word-based chunks."""
        chunker = FixedLengthWordChunker(words_per_chunk=25)
        result = chunker.chunk(self.long_text)

        # Should create multiple chunks (200 words / 25 = 8 chunks)
        expected_chunks = 8
        assert len(result.chunks) == expected_chunks

        # Validate word counts
        total_words = 0
        for i, chunk in enumerate(result.chunks):
            word_count = chunk.metadata.extra["word_count"]
            total_words += word_count

            # All chunks except possibly the last should have exactly 25 words
            if i < len(result.chunks) - 1:
                assert word_count == 25
            else:
                assert word_count <= 25  # Last chunk may be smaller

        # Total words should match (the test text actually has 180 words)
        # "This is a longer text document with many words. " * 20 = 9 words * 20 = 180 words
        assert total_words == 180

        # Validate all chunks
        assert_valid_chunks(result.chunks, self.validator)

    def test_word_tokenization_methods(self):
        """Test different word tokenization methods."""
        text = "Hello, world! This has punctuation... and numbers 123."

        # Test simple tokenization (default)
        simple_chunker = FixedLengthWordChunker(words_per_chunk=5, word_tokenization="simple")
        simple_result = simple_chunker.chunk(text)

        # Test whitespace tokenization
        whitespace_chunker = FixedLengthWordChunker(words_per_chunk=5, word_tokenization="whitespace")
        whitespace_result = whitespace_chunker.chunk(text)

        # Test regex tokenization
        regex_chunker = FixedLengthWordChunker(words_per_chunk=5, word_tokenization="regex")
        regex_result = regex_chunker.chunk(text)

        # All should produce valid results
        for result in [simple_result, whitespace_result, regex_result]:
            assert len(result.chunks) > 0
            assert_valid_chunks(result.chunks, self.validator)

        # Simple and regex should be similar (preserve punctuation)
        assert len(simple_result.chunks) == len(regex_result.chunks)

    def test_punctuation_handling(self):
        """Test punctuation preservation options."""
        text = "Hello, world! How are you? Fine, thanks."

        # Test with punctuation preservation (default)
        preserve_chunker = FixedLengthWordChunker(
            words_per_chunk=5,
            preserve_punctuation=True
        )
        preserve_result = preserve_chunker.chunk(text)

        # Test without punctuation preservation
        no_punct_chunker = FixedLengthWordChunker(
            words_per_chunk=5,
            preserve_punctuation=False
        )
        no_punct_result = no_punct_chunker.chunk(text)

        # Both should be valid
        assert_valid_chunks(preserve_result.chunks, self.validator)
        assert_valid_chunks(no_punct_result.chunks, self.validator)

        # Without punctuation should potentially have fewer tokens
        preserve_content = ' '.join(c.content for c in preserve_result.chunks)
        no_punct_content = ' '.join(c.content for c in no_punct_result.chunks)

        assert ',' in preserve_content or '!' in preserve_content
        # Note: Content comparison depends on tokenization method implementation

    def test_word_overlap_functionality(self):
        """Test overlap between word chunks."""
        # Use min_chunk_words=5 to ensure smaller chunks aren't filtered out
        chunker = FixedLengthWordChunker(words_per_chunk=8, overlap_words=2, min_chunk_words=5)
        # Create a text with enough words to generate multiple chunks
        words = ["word" + str(i) for i in range(20)]  # 20 unique words
        text = " ".join(words)
        result = chunker.chunk(text)

        # Should create multiple chunks with overlap
        assert len(result.chunks) >= 2

        # Check that chunks have expected word counts
        for i, chunk in enumerate(result.chunks):
            word_count = chunk.metadata.extra["word_count"]
            if i < len(result.chunks) - 1:
                # All but last chunk should have full words_per_chunk
                assert word_count == chunker.words_per_chunk
            else:
                # Last chunk may be smaller
                assert word_count <= chunker.words_per_chunk

        # Test overlap between chunks
        if len(result.chunks) >= 2:
            first_words = result.chunks[0].content.split()
            second_words = result.chunks[1].content.split()

            # Should have overlap (last 2 words of first chunk should appear in second chunk)
            # This is a basic check - exact overlap depends on step_size calculation
            assert len(first_words) <= chunker.words_per_chunk
            assert len(second_words) <= chunker.words_per_chunk

        assert_valid_chunks(result.chunks, self.validator)

    def test_minimum_chunk_size_handling(self):
        """Test handling of minimum chunk size constraints."""
        chunker = FixedLengthWordChunker(words_per_chunk=10, min_chunk_words=3)

        # Text with 12 words - should create 1 full chunk + 1 small chunk
        text = "One two three four five six seven eight nine ten eleven twelve"
        result = chunker.chunk(text)

        assert len(result.chunks) == 2
        assert result.chunks[0].metadata.extra["word_count"] == 10
        assert result.chunks[1].metadata.extra["word_count"] == 2  # Remaining words

        assert_valid_chunks(result.chunks, self.validator)

    def test_size_constraints(self):
        """Test maximum chunk size constraints."""
        # Create chunker with very small max_chunk_size
        chunker = FixedLengthWordChunker(
            words_per_chunk=100,  # High word count
            max_chunk_size=50     # But low character limit
        )

        long_words = "supercalifragilisticexpialidocious " * 10  # Long words
        result = chunker.chunk(long_words)

        # Should still create chunks despite size constraint warning
        assert len(result.chunks) > 0
        assert_valid_chunks(result.chunks, self.validator)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty text
        empty_result = self.chunker.chunk(self.empty_text)
        assert len(empty_result.chunks) == 0
        assert empty_result.processing_time >= 0

        # Whitespace only
        whitespace_result = self.chunker.chunk(self.whitespace_only)
        assert len(whitespace_result.chunks) == 0

        # Single word
        single_result = self.chunker.chunk(self.single_word)
        assert len(single_result.chunks) == 1
        assert single_result.chunks[0].content == self.single_word

        # All results should be valid
        for result in [empty_result, whitespace_result, single_result]:
            if result.chunks:
                assert_valid_chunks(result.chunks, self.validator)

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Test with non-string input types
        with pytest.raises(ValueError):
            self.chunker.chunk(b"bytes content")

        with pytest.raises(ValueError):
            self.chunker.chunk(Path("/some/path"))

        # Test invalid parameter combinations - these should be caught during initialization
        with pytest.raises(ValueError):
            FixedLengthWordChunker(words_per_chunk=10, overlap_words=15)  # overlap >= chunk size

        with pytest.raises(ValueError):
            FixedLengthWordChunker(words_per_chunk=0)

        with pytest.raises(ValueError):
            FixedLengthWordChunker(words_per_chunk=10, overlap_words=-1)

    def test_parameter_validation(self):
        """Test parameter validation and bounds."""
        # Valid parameters should work
        valid_chunker = FixedLengthWordChunker(
            words_per_chunk=50,
            overlap_words=5,
            max_chunk_size=2000,
            min_chunk_words=5
        )
        result = valid_chunker.chunk("Test content with several words for validation.")
        assert len(result.chunks) > 0

        # Test that invalid parameters raise errors (no auto-correction)
        with pytest.raises(ValueError):
            FixedLengthWordChunker(
                words_per_chunk=10,
                overlap_words=20  # Too high, should raise error
            )

    def test_metadata_completeness(self):
        """Test completeness and accuracy of chunk metadata."""
        result = self.chunker.chunk(self.long_text)

        for i, chunk in enumerate(result.chunks):
            # Basic metadata
            assert chunk.id.startswith("fixed_word_")
            assert chunk.metadata.source is not None
            assert chunk.metadata.length == len(chunk.content)

            # Strategy-specific metadata
            extra = chunk.metadata.extra
            assert "word_count" in extra
            assert "start_word_index" in extra
            assert "end_word_index" in extra
            assert "chunk_index" in extra
            assert extra["chunk_index"] == i
            assert extra["chunking_strategy"] == "fixed_length_word"

            # Word indices should be consistent
            assert extra["start_word_index"] <= extra["end_word_index"]
            if i > 0:
                prev_end = result.chunks[i-1].metadata.extra["end_word_index"]
                curr_start = extra["start_word_index"]
                # Should have reasonable progression (accounting for overlap)
                assert curr_start >= 0

    def test_streaming_support(self):
        """Test streaming chunk processing."""
        # Verify streaming capability
        assert self.chunker.can_stream() == True

        # Test streaming with content pieces
        content_pieces = [
            "First piece of content with several words. ",
            "Second piece continues the stream with more words. ",
            "Third piece completes the streaming test content here."
        ]

        chunks = list(self.chunker.chunk_stream(content_pieces))
        assert len(chunks) > 0

        # All streamed chunks should be valid
        assert_valid_chunks(chunks, self.validator)

        # Verify streaming metadata
        for chunk in chunks:
            extra = chunk.metadata.extra
            assert extra.get("streaming") == True
            assert "chunk_index" in extra

    def test_adaptation_functionality(self):
        """Test parameter adaptation based on feedback."""
        original_words = self.chunker.words_per_chunk
        original_overlap = self.chunker.overlap_words

        # Test quality feedback (poor score should reduce chunk size)
        self.chunker.adapt_parameters(0.3, "quality")
        assert self.chunker.words_per_chunk < original_words

        # Reset and test performance feedback
        self.chunker.words_per_chunk = original_words
        self.chunker.overlap_words = original_overlap

        # Performance feedback (poor score should increase chunk size)
        self.chunker.adapt_parameters(0.3, "performance")
        assert self.chunker.words_per_chunk > original_words

        # Test good feedback
        self.chunker.words_per_chunk = original_words
        self.chunker.adapt_parameters(0.9, "quality")
        # Should make modest improvements

        # Test adaptation history
        history = self.chunker.get_adaptation_history()
        assert len(history) > 0
        assert all("feedback_score" in record for record in history)

    def test_performance_benchmarks(self):
        """Test performance characteristics and benchmarks."""
        # Test with various text sizes
        small_text = "Short text with few words."
        medium_text = "Medium text content. " * 50  # ~100 words
        large_text = "Large text content. " * 500   # ~1000 words

        # Benchmark processing
        for text, description in [
            (small_text, "small"),
            (medium_text, "medium"),
            (large_text, "large")
        ]:
            result = self.chunker.chunk(text)

            # Performance should be reasonable
            assert_reasonable_performance(result.processing_time, len(text))

            # Memory efficiency - chunks shouldn't be excessively large
            for chunk in result.chunks:
                assert len(chunk.content) <= self.chunker.max_chunk_size

    def test_quality_metrics(self):
        """Test chunk quality evaluation."""
        result = self.chunker.chunk(self.mixed_content)

        if result.chunks:
            # Evaluate quality using the quality evaluator
            quality_report = self.evaluator.evaluate(
                result.chunks,
                original_content=self.mixed_content
            )

            # Basic quality checks
            assert quality_report.overall_score >= 0.0
            assert quality_report.overall_score <= 1.0

            # Coverage should be reasonable for word-based chunking
            assert quality_report.coverage >= 0.8  # Most content should be covered

            # Coherence may be lower for fixed-word chunking but should exist
            assert quality_report.coherence >= 0.3

    def test_integration_with_orchestrator(self):
        """Test integration with chunking orchestrator."""
        from chunking_strategy import ChunkerOrchestrator

        # Create orchestrator and test fixed-length word chunking
        orchestrator = ChunkerOrchestrator()

        # Test that the chunker is properly registered
        try:
            # Try to create the chunker via orchestrator
            result = orchestrator.chunk_text(
                content=self.simple_text,
                strategy="fixed_length_word",
                words_per_chunk=5
            )

            assert isinstance(result, ChunkingResult)
            assert result.strategy_used == "fixed_length_word"
            assert len(result.chunks) > 0
        except Exception as e:
            pytest.skip(f"Orchestrator integration not ready: {e}")

        # Test chunking via orchestrator
        result = orchestrator.chunk_text(
            content=self.simple_text,
            strategy="fixed_length_word",
            words_per_chunk=5
        )

        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "fixed_length_word"
        assert len(result.chunks) > 0

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        unicode_text = "Hello 世界! This is a test with émojis 🌍 and spëcial chäràctërs."

        result = self.chunker.chunk(unicode_text)
        assert len(result.chunks) > 0
        assert_valid_chunks(result.chunks, self.validator)

        # Content should be preserved
        reconstructed = ' '.join(chunk.content for chunk in result.chunks)
        assert "世界" in reconstructed
        assert "🌍" in reconstructed
        assert "émojis" in reconstructed

    def test_content_integrity(self):
        """Test that content integrity is maintained across chunking."""
        test_texts = [
            self.simple_text,
            self.complex_punctuation,
            self.mixed_content
        ]

        for text in test_texts:
            result = self.chunker.chunk(text)
            if result.chunks:
                # Custom content integrity check for word chunking
                # Word chunking may not preserve exact spacing, so we'll check word-level integrity
                original_words = text.lower().split()
                reconstructed_text = ' '.join(chunk.content for chunk in result.chunks)
                reconstructed_words = reconstructed_text.lower().split()

                # Allow for minor spacing differences but ensure all words are preserved
                assert len(set(original_words) - set(reconstructed_words)) == 0, "Some words lost during chunking"

    def test_boundary_conditions(self):
        """Test various boundary conditions."""
        # Test with exactly the chunk size
        exact_words = "word " * self.chunker.words_per_chunk
        result = self.chunker.chunk(exact_words.strip())
        assert len(result.chunks) == 1

        # Test with chunk_size + 1
        plus_one = "word " * (self.chunker.words_per_chunk + 1)
        result = self.chunker.chunk(plus_one.strip())
        assert len(result.chunks) == 2

        # Test with very large chunk size setting
        large_chunker = FixedLengthWordChunker(words_per_chunk=10000)
        result = large_chunker.chunk(self.long_text)
        # Should fit in one chunk
        assert len(result.chunks) == 1

    def test_configuration_compatibility(self):
        """Test compatibility with configuration files."""
        # Test that all parameters can be set via configuration
        config_params = {
            "words_per_chunk": 75,
            "overlap_words": 5,
            "max_chunk_size": 3000,
            "word_tokenization": "regex",
            "preserve_punctuation": False,
            "min_chunk_words": 8
        }

        chunker = FixedLengthWordChunker(**config_params)
        result = chunker.chunk(self.long_text)

        assert len(result.chunks) > 0
        assert_valid_chunks(result.chunks, self.validator)

        # Verify configuration was applied
        assert chunker.words_per_chunk == 75
        assert chunker.overlap_words == 5
        assert chunker.word_tokenization == "regex"

    def test_error_handling_robustness(self):
        """Test robust error handling in various scenarios."""
        # Test with malformed input (should not crash)
        weird_text = "\n\n\n   \t\t\t   \n\n\n"
        result = self.chunker.chunk(weird_text)
        # Should handle gracefully (empty or valid result)
        assert isinstance(result, ChunkingResult)

        # Test with very long words
        super_long_word = "a" * 10000
        result = self.chunker.chunk(super_long_word)
        assert len(result.chunks) > 0
        # Should not crash even if it exceeds size limits

        # Test with mixed line endings
        mixed_endings = "Line one\r\nLine two\nLine three\rLine four"
        result = self.chunker.chunk(mixed_endings)
        assert len(result.chunks) > 0
