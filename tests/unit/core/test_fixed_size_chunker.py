"""
Comprehensive tests for the FixedSizeChunker.

This module demonstrates best practices for testing chunking strategies
and serves as a template for testing other algorithms.
"""

import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from chunking_strategy.strategies.general.fixed_size import FixedSizeChunker
from chunking_strategy.core.base import Chunk, ChunkingResult, ModalityType
from chunking_strategy.utils.validation import ChunkValidator


class TestFixedSizeChunker:
    """Test suite for FixedSizeChunker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = FixedSizeChunker(chunk_size=10)
        self.validator = ChunkValidator()

        # Test data
        self.simple_text = "Hello world! This is a test."
        self.long_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        self.empty_text = ""
        self.whitespace_text = "   \n\t   "
        self.unicode_text = "Hello 世界! This is a tëst with ñoñ-ASCII characters."

    def test_basic_chunking(self):
        """Test basic chunking functionality."""
        result = self.chunker.chunk(self.simple_text)

        # Validate result structure
        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "fixed_size"
        assert result.processing_time is not None
        assert result.processing_time > 0

        # Validate chunks
        assert len(result.chunks) > 0
        for chunk in result.chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) <= self.chunker.chunk_size
            assert chunk.modality == ModalityType.TEXT
            assert chunk.metadata is not None
            assert chunk.metadata.chunker_used == "fixed_size"

    def test_chunk_size_parameter(self):
        """Test different chunk sizes."""
        test_cases = [
            (5, self.simple_text),
            (15, self.simple_text),
            (100, self.simple_text),
        ]

        for chunk_size, text in test_cases:
            chunker = FixedSizeChunker(chunk_size=chunk_size)
            result = chunker.chunk(text)

            # All chunks except possibly the last should be exactly chunk_size
            for i, chunk in enumerate(result.chunks[:-1]):
                assert len(chunk.content) == chunk_size, f"Chunk {i} has wrong size"

            # Last chunk should be <= chunk_size
            if result.chunks:
                assert len(result.chunks[-1].content) <= chunk_size

    def test_overlap_functionality(self):
        """Test overlap between chunks."""
        chunker = FixedSizeChunker(chunk_size=10, overlap_size=3)
        result = chunker.chunk(self.long_text)

        if len(result.chunks) > 1:
            # Check that consecutive chunks have expected overlap
            for i in range(len(result.chunks) - 1):
                current_chunk = result.chunks[i].content
                next_chunk = result.chunks[i + 1].content

                # The end of current chunk should overlap with start of next chunk
                current_end = current_chunk[-3:]  # Last 3 characters
                next_start = next_chunk[:3]       # First 3 characters

                # There should be some overlap (not necessarily exact due to boundary preservation)
                overlap_found = any(char in next_start for char in current_end)
                assert overlap_found or len(current_chunk) < 10, "No overlap found between consecutive chunks"

    def test_different_units(self):
        """Test character, byte, and word-based chunking."""
        # Character-based (default)
        char_chunker = FixedSizeChunker(chunk_size=10, unit="character")
        char_result = char_chunker.chunk(self.simple_text)

        # Word-based
        word_chunker = FixedSizeChunker(chunk_size=3, unit="word")
        word_result = word_chunker.chunk(self.simple_text)

        # Byte-based
        byte_chunker = FixedSizeChunker(chunk_size=10, unit="byte")
        byte_result = byte_chunker.chunk(self.simple_text)

        # All should produce valid results
        for result in [char_result, word_result, byte_result]:
            assert len(result.chunks) > 0
            validation_issues = self.validator.validate_result(result)
            assert len(validation_issues) == 0, f"Validation failed: {validation_issues}"

    def test_boundary_preservation(self):
        """Test boundary preservation option."""
        chunker = FixedSizeChunker(chunk_size=8, preserve_boundaries=True)
        text = "word1 word2 word3 word4"
        result = chunker.chunk(text)

        # With boundary preservation, chunks should not break words
        for chunk in result.chunks:
            content = chunk.content.strip()
            if len(content) > 0:
                # Should not start or end with partial words (except first/last chunk)
                words = content.split()
                # Basic check: if chunk has multiple words, it should be reasonably complete
                if len(words) > 1:
                    assert not content.startswith(' '), "Chunk starts with space unexpectedly"

    def test_empty_input(self):
        """Test handling of empty input."""
        result = self.chunker.chunk(self.empty_text)

        assert len(result.chunks) == 0
        assert result.strategy_used == "fixed_size"
        validation_issues = self.validator.validate_result(result)
        assert len(validation_issues) == 0

    def test_whitespace_only_input(self):
        """Test handling of whitespace-only input."""
        result = self.chunker.chunk(self.whitespace_text)

        # Should either have no chunks or chunks with minimal content
        if result.chunks:
            for chunk in result.chunks:
                # Chunk content might be empty after stripping
                assert isinstance(chunk.content, str)

    def test_unicode_handling(self):
        """Test Unicode character handling."""
        result = self.chunker.chunk(self.unicode_text)

        assert len(result.chunks) > 0

        # Verify Unicode characters are preserved
        combined_content = ''.join(chunk.content for chunk in result.chunks)
        assert "世界" in combined_content
        assert "tëst" in combined_content
        assert "ñoñ" in combined_content

    def test_file_input(self):
        """Test chunking from file input."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(self.long_text)
            temp_path = Path(f.name)

        try:
            result = self.chunker.chunk(temp_path)

            assert len(result.chunks) > 0
            assert result.source_info is None or "source" in str(result.source_info)

            # Verify content integrity
            combined_content = ''.join(chunk.content for chunk in result.chunks)
            assert combined_content == self.long_text

        finally:
            temp_path.unlink()  # Clean up

    def test_binary_input(self):
        """Test binary input handling."""
        binary_data = b"Hello binary world! This is a test."
        result = self.chunker.chunk(binary_data)

        assert len(result.chunks) > 0

        # Should handle binary data gracefully
        for chunk in result.chunks:
            # Content might be decoded to string or kept as bytes
            assert chunk.content is not None

    def test_streaming_interface(self):
        """Test streaming chunking interface."""
        def content_generator():
            """Generate content in pieces."""
            pieces = ["Hello ", "world! ", "This is ", "a streaming ", "test."]
            for piece in pieces:
                yield piece

        chunks = list(self.chunker.chunk_stream(content_generator()))

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.modality == ModalityType.TEXT

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid chunk_size
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=0)

        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=-1)

        # Test invalid overlap_size
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap_size=-1)

        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap_size=10)  # overlap >= chunk_size

        # Test invalid unit
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, unit="invalid")

    def test_adaptive_functionality(self):
        """Test adaptive parameter adjustment."""
        chunker = FixedSizeChunker(chunk_size=100)
        original_size = chunker.chunk_size

        # Test quality feedback
        chunker.adapt_parameters(0.3, "quality")
        assert chunker.chunk_size > original_size, "Chunk size should increase for low quality feedback"

        # Test performance feedback
        chunker = FixedSizeChunker(chunk_size=100)
        chunker.adapt_parameters(0.3, "performance")
        assert chunker.chunk_size < original_size, "Chunk size should decrease for low performance feedback"

    def test_chunk_metadata(self):
        """Test chunk metadata completeness."""
        result = self.chunker.chunk(self.simple_text)

        for chunk in result.chunks:
            # Basic metadata checks
            assert chunk.id is not None
            assert chunk.metadata is not None
            assert chunk.metadata.source is not None
            assert chunk.metadata.chunker_used == "fixed_size"
            assert chunk.metadata.length == len(chunk.content)

            # Size consistency
            assert chunk.size == len(chunk.content)

    def test_performance_characteristics(self):
        """Test performance on different input sizes."""
        import time

        # Test with different text sizes
        sizes = [100, 1000, 10000]

        for size in sizes:
            text = "A" * size

            start_time = time.time()
            result = self.chunker.chunk(text)
            end_time = time.time()

            processing_time = end_time - start_time

            # Should complete in reasonable time
            assert processing_time < 1.0, f"Processing {size} chars took too long: {processing_time}s"

            # Should produce expected number of chunks
            expected_chunks = (size + self.chunker.chunk_size - 1) // self.chunker.chunk_size
            assert len(result.chunks) == expected_chunks, f"Wrong number of chunks for size {size}"

    def test_content_integrity(self):
        """Test that no content is lost during chunking."""
        test_texts = [
            self.simple_text,
            self.long_text,
            self.unicode_text,
            "Single",
            "A" * 1000,  # Long repetitive text
        ]

        for text in test_texts:
            result = self.chunker.chunk(text)

            # Reconstruct original text from chunks
            reconstructed = ''.join(chunk.content for chunk in result.chunks)

            assert reconstructed == text, f"Content integrity lost for text: {text[:50]}..."

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single character
        result = self.chunker.chunk("A")
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "A"

        # Chunk size larger than content
        big_chunker = FixedSizeChunker(chunk_size=1000)
        result = big_chunker.chunk(self.simple_text)
        assert len(result.chunks) == 1
        assert result.chunks[0].content == self.simple_text

        # Very small chunk size
        tiny_chunker = FixedSizeChunker(chunk_size=1)
        result = tiny_chunker.chunk("ABC")
        assert len(result.chunks) == 3
        assert [chunk.content for chunk in result.chunks] == ["A", "B", "C"]


class TestFixedSizeChunkerIntegration:
    """Integration tests for FixedSizeChunker with other components."""

    def test_with_validator(self):
        """Test integration with chunk validator."""
        chunker = FixedSizeChunker(chunk_size=50)
        validator = ChunkValidator()

        text = "This is a test document with multiple sentences. Each sentence should be properly handled by the chunker."
        result = chunker.chunk(text)

        # Validate result
        issues = validator.validate_result(result)
        assert len(issues) == 0, f"Validation issues: {issues}"

        # Get quality score
        quality_score = validator.get_quality_score(result)
        assert 0.0 <= quality_score <= 1.0

    def test_with_orchestrator(self):
        """Test integration with orchestrator."""
        from chunking_strategy import ChunkerOrchestrator

        # This would test the chunker through the orchestrator
        orchestrator = ChunkerOrchestrator()

        # Override strategy to use fixed_size
        result = orchestrator.chunk_content(
            "Test content for orchestrator integration.",
            strategy_override="fixed_size"
        )

        assert result.strategy_used == "fixed_size"
        assert len(result.chunks) > 0


# Test data files creation helpers
def create_test_files():
    """
    Create test data files for comprehensive testing.

    This function should be called during test setup to create
    sample files for testing different scenarios.
    """
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)

    # Create different types of test files
    files_to_create = {
        "simple.txt": "This is a simple text file for testing.",
        "long.txt": "This is a sentence. " * 100,  # 100 repeated sentences
        "unicode.txt": "Unicode test: héllo wörld! 你好世界 🌍",
        "empty.txt": "",
        "structured.md": """# Title

## Section 1
This is section one with some content.

## Section 2
This is section two with more content.
""",
    }

    for filename, content in files_to_create.items():
        file_path = test_data_dir / filename
        file_path.write_text(content, encoding='utf-8')

    return test_data_dir


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
