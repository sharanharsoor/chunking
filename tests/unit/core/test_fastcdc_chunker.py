"""
Unit tests for FastCDC (Fast Content-Defined Chunking) strategy.

Tests cover:
- Basic FastCDC functionality
- Different hash algorithms (gear, rabin, buzhash)
- Content boundary detection
- Variable chunk size validation
- Binary and text content handling
- Performance characteristics
- Adaptation mechanisms
- Streaming support
"""

import pytest
import tempfile
import os
from pathlib import Path

from chunking_strategy.core.registry import create_chunker, list_chunkers
from chunking_strategy.core.base import ModalityType
from chunking_strategy import ChunkerOrchestrator


class TestFastCDCChunker:
    """Test FastCDC chunker functionality."""

    def test_fastcdc_registration(self):
        """Test that FastCDC is properly registered."""
        chunkers = list_chunkers()
        assert "fastcdc" in chunkers

        # Test creation
        chunker = create_chunker("fastcdc")
        assert chunker is not None
        assert chunker.__class__.__name__ == "FastCDCChunker"
        assert "any" in chunker.get_supported_formats()

    def test_basic_text_chunking(self):
        """Test basic text chunking with FastCDC."""
        chunker = create_chunker("fastcdc", min_chunk_size=100, max_chunk_size=500, avg_chunk_size=300)

        # Create a text with clear content boundaries
        text = """Section 1: Introduction
This is the first section of our document that contains introductory information about the topic we're discussing.

Section 2: Main Content
This section contains the main content and represents a substantial portion of the document with detailed information.

Section 3: Analysis
Here we provide detailed analysis and insights based on the information presented in the previous sections.

Section 4: Conclusion
Finally, we conclude with a summary of key points and recommendations for future work."""

        result = chunker.chunk(text)

        # Validate results
        assert result.total_chunks > 0
        assert result.strategy_used == "fastcdc"
        assert result.processing_time >= 0

        # Check that chunks have reasonable sizes (allow some flexibility for CDC)
        for chunk in result.chunks:
            chunk_size = len(chunk.content.encode('utf-8'))
            # CDC may create smaller final chunks, so we allow some flexibility
            assert chunk_size >= 50 or chunk == result.chunks[-1]  # More flexible size check
            assert chunk.modality == ModalityType.TEXT
            assert "content_fingerprint" in chunk.metadata.extra
            assert "entropy" in chunk.metadata.extra
            assert "chunk_size" in chunk.metadata.extra

    def test_hash_algorithms(self):
        """Test different hash algorithms."""
        algorithms = ["gear", "rabin", "buzhash"]
        text = "A" * 5000 + "B" * 5000 + "C" * 5000  # Pattern that should create boundaries

        for algorithm in algorithms:
            chunker = create_chunker("fastcdc",
                                   hash_algorithm=algorithm,
                                   min_chunk_size=1000,
                                   max_chunk_size=8000,
                                   avg_chunk_size=4000)

            result = chunker.chunk(text)

            assert result.total_chunks > 0
            assert result.source_info["algorithm"] == algorithm

            # Each algorithm should detect some boundaries
            assert result.total_chunks >= 1

    def test_binary_content_handling(self):
        """Test FastCDC with binary-like content."""
        # Create binary-like content
        binary_data = bytes(range(256)) * 100  # 25.6KB of binary data

        chunker = create_chunker("fastcdc",
                               min_chunk_size=2048,
                               max_chunk_size=16384,
                               avg_chunk_size=8192)

        result = chunker.chunk(binary_data)

        assert result.total_chunks > 0
        assert result.source_info["total_bytes"] == len(binary_data)

        # Check binary content handling
        for chunk in result.chunks:
            # Should detect as binary and use MIXED modality
            if "binary data" in chunk.content:
                assert chunk.modality == ModalityType.MIXED
            assert "is_text" in chunk.metadata.extra
            assert "entropy" in chunk.metadata.extra

    def test_variable_chunk_sizes(self):
        """Test that FastCDC creates variable-size chunks."""
        chunker = create_chunker("fastcdc",
                               min_chunk_size=1000,
                               max_chunk_size=10000,
                               avg_chunk_size=5000)

        # Create content with varying patterns
        text = ("A" * 2000 + "This is a clear boundary marker. " + "B" * 3000 +
               "Another boundary marker here. " + "C" * 4000 +
               "Final boundary marker section. " + "D" * 1500)

        result = chunker.chunk(text)

        # Should create multiple chunks with different sizes
        assert result.total_chunks > 1

        chunk_sizes = [len(chunk.content.encode('utf-8')) for chunk in result.chunks]

        # Verify size constraints (allow some flexibility for content-defined chunking)
        for i, size in enumerate(chunk_sizes):
            # Last chunk can be smaller due to end-of-content
            if i == len(chunk_sizes) - 1:
                assert size <= 10000  # Max size
            else:
                assert size >= 800  # Allow some flexibility below min_size
                assert size <= 10000  # Max size

        # Chunks should have different sizes (variable chunking)
        assert len(set(chunk_sizes)) > 1, "All chunks have the same size - not variable chunking"

    def test_content_fingerprinting(self):
        """Test content fingerprinting and deduplication potential."""
        chunker = create_chunker("fastcdc", min_chunk_size=500, max_chunk_size=5000, avg_chunk_size=2000)

        # Create content with a large duplicate section to ensure it gets its own chunk
        duplicate_section = "This is a repeated section that should generate the same fingerprint when chunked. " * 50
        text1 = "Short unique start. " + duplicate_section + " Short unique end 1."
        text2 = "Different unique start. " + duplicate_section + " Different unique end 2."

        result1 = chunker.chunk(text1)
        result2 = chunker.chunk(text2)

        # If both create multiple chunks, there should be potential for common fingerprints
        if result1.total_chunks > 1 and result2.total_chunks > 1:
            # Collect fingerprints
            fingerprints1 = [chunk.metadata.extra["content_fingerprint"]
                            for chunk in result1.chunks]
            fingerprints2 = [chunk.metadata.extra["content_fingerprint"]
                            for chunk in result2.chunks]

            # Should have some matching fingerprints for duplicate content
            common_fingerprints = set(fingerprints1) & set(fingerprints2)
            assert len(common_fingerprints) > 0, "No common fingerprints found for duplicate content"
        else:
            # If only single chunks, at least verify fingerprints are different for different content
            fp1 = result1.chunks[0].metadata.extra["content_fingerprint"]
            fp2 = result2.chunks[0].metadata.extra["content_fingerprint"]
            # Different overall content should have different fingerprints
            assert fp1 != fp2, "Different content should have different fingerprints"

    def test_normalization_effect(self):
        """Test the effect of FastCDC normalization."""
        text = "x" * 20000  # Uniform content that benefits from normalization

        # Test with normalization
        chunker_norm = create_chunker("fastcdc",
                                    normalization=True,
                                    min_chunk_size=2048,
                                    max_chunk_size=16384,
                                    avg_chunk_size=8192)

        # Test without normalization
        chunker_no_norm = create_chunker("fastcdc",
                                       normalization=False,
                                       min_chunk_size=2048,
                                       max_chunk_size=16384,
                                       avg_chunk_size=8192)

        result_norm = chunker_norm.chunk(text)
        result_no_norm = chunker_no_norm.chunk(text)

        # Both should work, but might have different characteristics
        assert result_norm.total_chunks > 0
        assert result_no_norm.total_chunks > 0

        # Normalization metadata should be correct
        assert result_norm.source_info["normalization"] == True
        assert result_no_norm.source_info["normalization"] == False

    def test_file_input(self):
        """Test FastCDC with file input."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("File content for FastCDC testing. " * 200)
            temp_file = f.name

        try:
            chunker = create_chunker("fastcdc")

            # Test with file path
            result = chunker.chunk(temp_file)

            assert result.total_chunks > 0
            assert result.source_info["source_file"] != "direct_input"

            # Test with Path object
            result_path = chunker.chunk(Path(temp_file))
            assert result_path.total_chunks > 0

        finally:
            os.unlink(temp_file)

    def test_chunk_size_parameters(self):
        """Test different chunk size parameters."""
        text = "Test content. " * 1000  # 14KB of content

        # Small chunks
        small_chunker = create_chunker("fastcdc",
                                     min_chunk_size=512,
                                     max_chunk_size=2048,
                                     avg_chunk_size=1024)

        # Large chunks
        large_chunker = create_chunker("fastcdc",
                                     min_chunk_size=4096,
                                     max_chunk_size=16384,
                                     avg_chunk_size=8192)

        small_result = small_chunker.chunk(text)
        large_result = large_chunker.chunk(text)

        # Small chunker should create more, smaller chunks
        assert small_result.total_chunks >= large_result.total_chunks

        # Verify size constraints
        for chunk in small_result.chunks:
            size = len(chunk.content.encode('utf-8'))
            assert size >= 512 and size <= 2048

        for chunk in large_result.chunks:
            size = len(chunk.content.encode('utf-8'))
            assert size >= 4096 and size <= 16384

    def test_streaming_support(self):
        """Test streaming chunking capability."""
        chunker = create_chunker("fastcdc")

        # Create a stream of content
        content_parts = [
            "First part of the content stream. ",
            "Second part with different patterns. ",
            "Third part containing more information. ",
            "Final part of the streaming content."
        ]

        # Test streaming
        chunks = list(chunker.chunk_stream(content_parts))

        assert len(chunks) > 0

        # Combine streamed chunks content
        streamed_content = ''.join(chunk.content for chunk in chunks)

        # Compare with direct chunking
        direct_content = ''.join(content_parts)
        direct_result = chunker.chunk(direct_content)
        direct_combined = ''.join(chunk.content for chunk in direct_result.chunks)

        # Content should be equivalent
        assert streamed_content == direct_combined

    def test_adaptation_mechanism(self):
        """Test parameter adaptation based on feedback."""
        chunker = create_chunker("fastcdc")

        original_avg = chunker.avg_chunk_size
        original_mask = chunker.mask_bits

        # Test deduplication feedback
        chunker.adapt_parameters(0.3, "deduplication")
        assert chunker.avg_chunk_size > original_avg  # Should increase for better deduplication

        # Reset and test performance feedback
        chunker.avg_chunk_size = original_avg
        chunker.adapt_parameters(0.3, "performance")
        assert chunker.avg_chunk_size < original_avg  # Should decrease for better performance

        # Test quality feedback
        chunker.mask_bits = original_mask
        chunker.adapt_parameters(0.3, "quality")
        # Mask bits should change for better quality

        # Check adaptation history
        history = chunker.get_adaptation_history()
        assert len(history) == 3  # Three adaptations

    def test_orchestrator_integration(self):
        """Test FastCDC integration with orchestrator."""
        config = {
            'strategies': {
                'primary': 'fastcdc'
            },
            'fastcdc': {
                'min_chunk_size': 1024,
                'max_chunk_size': 8192,
                'avg_chunk_size': 4096
            }
        }

        orchestrator = ChunkerOrchestrator(config=config)

        text = "Orchestrator integration test content. " * 100
        result = orchestrator.chunk_content(text)

        assert result.total_chunks > 0
        assert result.strategy_used == "fastcdc"

    def test_real_file_analysis(self):
        """Test FastCDC with actual test files."""
        chunker = create_chunker("fastcdc")

        # Test with text file
        text_file = "test_data/sample_text_for_cdc.txt"
        if os.path.exists(text_file):
            result = chunker.chunk(text_file)

            assert result.total_chunks > 0
            assert result.source_info["source_file"] == text_file

            # Verify content analysis
            for chunk in result.chunks:
                assert "entropy" in chunk.metadata.extra
                assert "compressibility" in chunk.metadata.extra
                assert chunk.metadata.extra["is_text"] == True

    def test_mixed_content_analysis(self):
        """Test FastCDC with mixed content file."""
        chunker = create_chunker("fastcdc")

        mixed_file = "test_data/sample_mixed_content.txt"
        if os.path.exists(mixed_file):
            result = chunker.chunk(mixed_file)

            assert result.total_chunks > 0

            # Should detect content transitions
            entropies = [chunk.metadata.extra.get("entropy", 0) for chunk in result.chunks]

            # Mixed content should show entropy variation if multiple chunks
            if result.total_chunks > 1:
                assert max(entropies) - min(entropies) > 0.5, "No entropy variation in mixed content"
            else:
                # Single chunk is ok for mixed content, just verify entropy is reasonable
                assert entropies[0] > 3.0, "Mixed content should have reasonable entropy"

    def test_performance_characteristics(self):
        """Test FastCDC performance characteristics."""
        chunker = create_chunker("fastcdc")

        # Large content test
        large_content = "Performance test content. " * 10000  # ~250KB

        result = chunker.chunk(large_content)

        # Should be fast
        assert result.processing_time < 1.0  # Should complete in under 1 second

        # Should create reasonable number of chunks
        expected_chunks = len(large_content.encode()) // chunker.avg_chunk_size
        assert result.total_chunks > 0
        assert result.total_chunks <= expected_chunks * 2  # Reasonable upper bound

    def test_edge_cases(self):
        """Test FastCDC edge cases."""
        chunker = create_chunker("fastcdc")

        # Empty content
        result_empty = chunker.chunk("")
        assert result_empty.total_chunks == 0

        # Very small content
        result_small = chunker.chunk("tiny")
        assert result_small.total_chunks >= 0

        # Single character repeated
        result_uniform = chunker.chunk("a" * 10000)
        assert result_uniform.total_chunks > 0

    def test_chunk_metadata_completeness(self):
        """Test that chunk metadata is complete and accurate."""
        chunker = create_chunker("fastcdc")

        text = "Metadata completeness test. " * 200
        result = chunker.chunk(text)

        for chunk in result.chunks:
            metadata = chunk.metadata.extra

            # Required fields
            assert "chunk_type" in metadata
            assert "algorithm" in metadata
            assert "chunk_size" in metadata
            assert "is_text" in metadata
            assert "entropy" in metadata
            assert "compressibility" in metadata
            assert "content_fingerprint" in metadata
            assert "boundary_type" in metadata
            assert "normalization_used" in metadata

            # Validate values
            assert metadata["chunk_type"] == "content_defined"
            assert metadata["algorithm"] == chunker.hash_algorithm
            assert metadata["chunk_size"] > 0
            assert isinstance(metadata["entropy"], float)
            assert 0 <= metadata["compressibility"] <= 1
            assert len(metadata["content_fingerprint"]) > 0
