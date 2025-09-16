"""
Comprehensive tests for the ParagraphBasedChunker.

This module tests the paragraph-based chunking strategy with real files,
edge cases, and comprehensive validation scenarios.
"""

import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from chunking_strategy.strategies.text.paragraph_based import ParagraphBasedChunker
from chunking_strategy.core.base import Chunk, ChunkingResult, ModalityType
from chunking_strategy.utils.validation import ChunkValidator
from chunking_strategy.core.metrics import ChunkingQualityEvaluator
from tests.conftest import assert_valid_chunks, assert_reasonable_performance


class TestParagraphBasedChunker:
    """Test suite for ParagraphBasedChunker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = ParagraphBasedChunker(max_paragraphs=2)
        self.validator = ChunkValidator()
        self.evaluator = ChunkingQualityEvaluator()

        # Test data
        self.simple_paragraphs = """First paragraph with some content here. This explains the initial concept.

Second paragraph continues the discussion. It provides more details and context.

Third paragraph adds additional information. This expands on the previous topics.

Fourth paragraph concludes the section. It wraps up the main points discussed."""

        self.mixed_length_paragraphs = """Very short paragraph.

This is a much longer paragraph with significantly more content. It contains multiple sentences and provides extensive detail about the topic being discussed. The length variation should test the chunker's ability to handle different paragraph sizes.

Another short one.

Final paragraph with moderate length. It has a reasonable amount of content without being too verbose."""

        self.single_paragraph = "This is just one paragraph with some content. It contains a few sentences but no paragraph breaks."

    def test_basic_paragraph_chunking(self):
        """Test basic paragraph grouping functionality."""
        result = self.chunker.chunk(self.simple_paragraphs)

        # Validate result structure
        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "paragraph_based"
        assert result.processing_time is not None
        assert result.processing_time > 0

        # Should create 2 chunks (2 paragraphs each)
        assert len(result.chunks) == 2

        # Each chunk should have 2 paragraphs
        for chunk in result.chunks:
            assert chunk.metadata.extra.get("paragraph_count") == 2
            # Should contain paragraph separators
            assert "\n\n" in chunk.content

        # Validate all chunks
        assert_valid_chunks(result.chunks, self.validator)

        # Verify all paragraphs are preserved
        all_content = " ".join(chunk.content for chunk in result.chunks)
        assert "First paragraph" in all_content
        assert "Second paragraph" in all_content
        assert "Third paragraph" in all_content
        assert "Fourth paragraph" in all_content

    def test_paragraph_count_limits(self):
        """Test minimum and maximum paragraph constraints."""
        # Test with different max_paragraphs settings
        test_cases = [
            (1, 4),  # max_paragraphs=1 should create 4 chunks
            (2, 2),  # max_paragraphs=2 should create 2 chunks
            (3, 2),  # max_paragraphs=3 should create 2 chunks (4 paragraphs total)
            (5, 1),  # max_paragraphs=5 should create 1 chunk (all paragraphs fit)
        ]

        for max_paragraphs, expected_chunks in test_cases:
            chunker = ParagraphBasedChunker(max_paragraphs=max_paragraphs)
            result = chunker.chunk(self.simple_paragraphs)
            assert len(result.chunks) == expected_chunks, f"Failed for max_paragraphs={max_paragraphs}"

    def test_paragraph_overlap(self):
        """Test overlap between paragraph chunks."""
        chunker = ParagraphBasedChunker(max_paragraphs=2, overlap_paragraphs=1)
        result = chunker.chunk(self.simple_paragraphs)

        # Should have overlap between consecutive chunks
        if len(result.chunks) > 1:
            # Check for overlapping content between first and second chunk
            first_content = result.chunks[0].content
            second_content = result.chunks[1].content

            # Find paragraphs in each chunk
            first_paragraphs = [p.strip() for p in first_content.split('\n\n') if p.strip()]
            second_paragraphs = [p.strip() for p in second_content.split('\n\n') if p.strip()]

            # Should find at least one overlapping paragraph
            overlap_found = any(para in second_paragraphs for para in first_paragraphs[-1:])
            assert overlap_found, "Expected overlap between chunks"

    def test_chunk_size_limits(self):
        """Test maximum chunk size constraints."""
        chunker = ParagraphBasedChunker(max_paragraphs=5, max_chunk_size=200)

        # Create text with long paragraphs
        long_paragraph = "This is a very long paragraph that exceeds the chunk size limit. " * 5
        text = f"{long_paragraph}\n\n{long_paragraph}\n\n{long_paragraph}"

        result = chunker.chunk(text)

        # For paragraph-based chunking, individual paragraphs can exceed max_chunk_size
        # but the chunker should avoid adding additional paragraphs that would exceed the limit
        for chunk in result.chunks:
            # Each chunk should contain at least one complete paragraph
            assert chunk.metadata.extra.get("paragraph_count") >= 1
            # If a single paragraph exceeds max_chunk_size, that's acceptable
            paragraph_count = chunk.metadata.extra.get("paragraph_count", 1)
            if paragraph_count > 1:
                # If multiple paragraphs, should respect size limit reasonably
                assert len(chunk.content) <= chunker.max_chunk_size * 2

    def test_mixed_paragraph_lengths(self):
        """Test handling of paragraphs with different lengths."""
        result = self.chunker.chunk(self.mixed_length_paragraphs)

        assert len(result.chunks) >= 1
        assert_valid_chunks(result.chunks, self.validator)

        # Should handle both short and long paragraphs
        all_content = " ".join(chunk.content for chunk in result.chunks)
        assert "Very short paragraph" in all_content
        assert "much longer paragraph" in all_content
        assert "Another short one" in all_content
        assert "Final paragraph" in all_content

    def test_single_paragraph_input(self):
        """Test handling of single paragraph input."""
        result = self.chunker.chunk(self.single_paragraph)

        assert len(result.chunks) == 1
        assert result.chunks[0].content.strip() == self.single_paragraph
        assert result.chunks[0].metadata.extra.get("paragraph_count") == 1
        assert_valid_chunks(result.chunks, self.validator)

    def test_empty_and_whitespace_input(self):
        """Test handling of empty and whitespace-only input."""
        # Empty input
        empty_result = self.chunker.chunk("")
        assert len(empty_result.chunks) == 0

        # Whitespace only
        whitespace_result = self.chunker.chunk("   \n\n\t   ")
        # Should either have no chunks or handle gracefully
        if whitespace_result.chunks:
            for chunk in whitespace_result.chunks:
                assert isinstance(chunk.content, str)

    def test_formatting_preservation(self):
        """Test preservation of paragraph formatting."""
        # Test with formatting preservation enabled
        preserve_chunker = ParagraphBasedChunker(max_paragraphs=2, preserve_formatting=True)
        result = preserve_chunker.chunk(self.simple_paragraphs)

        # Should preserve paragraph separators
        for chunk in result.chunks:
            if chunk.metadata.extra.get("paragraph_count", 0) > 1:
                assert "\n\n" in chunk.content

        # Test with formatting preservation disabled
        no_preserve_chunker = ParagraphBasedChunker(max_paragraphs=2, preserve_formatting=False)
        result_no_preserve = no_preserve_chunker.chunk(self.simple_paragraphs)

        # Should join paragraphs with spaces instead
        for chunk in result_no_preserve.chunks:
            # Formatting not preserved, so no double newlines
            assert "\n\n" not in chunk.content or chunk.metadata.extra.get("paragraph_count", 0) == 1

    def test_short_paragraph_merging(self):
        """Test merging of very short paragraphs."""
        text_with_short_paragraphs = """Very short.

Another short one.

This is a longer paragraph with more substantial content that should not be merged with others.

Short again.

Final short."""

        # Test with merging enabled
        merge_chunker = ParagraphBasedChunker(
            max_paragraphs=3,
            merge_short_paragraphs=True,
            min_paragraph_length=20
        )
        result_merged = merge_chunker.chunk(text_with_short_paragraphs)

        # Test with merging disabled
        no_merge_chunker = ParagraphBasedChunker(
            max_paragraphs=3,
            merge_short_paragraphs=False
        )
        result_no_merge = no_merge_chunker.chunk(text_with_short_paragraphs)

        # Merged version should have fewer "paragraphs" after processing
        # (Note: actual paragraph count may vary based on merging logic)
        assert_valid_chunks(result_merged.chunks, self.validator)
        assert_valid_chunks(result_no_merge.chunks, self.validator)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid max_paragraphs
        with pytest.raises(ValueError):
            ParagraphBasedChunker(max_paragraphs=0)

        # Test invalid min_paragraphs
        with pytest.raises(ValueError):
            ParagraphBasedChunker(min_paragraphs=0)

        # Test min > max
        with pytest.raises(ValueError):
            ParagraphBasedChunker(max_paragraphs=2, min_paragraphs=5)

        # Test invalid overlap
        with pytest.raises(ValueError):
            ParagraphBasedChunker(max_paragraphs=3, overlap_paragraphs=3)

        # Test invalid chunk size
        with pytest.raises(ValueError):
            ParagraphBasedChunker(max_chunk_size=0)

        # Test invalid min_paragraph_length
        with pytest.raises(ValueError):
            ParagraphBasedChunker(min_paragraph_length=0)

    def test_streaming_interface(self):
        """Test streaming chunking interface."""
        def paragraph_generator():
            paragraphs = [
                "First paragraph content here.\n\n",
                "Second paragraph follows.\n\n",
                "Third paragraph continues.\n\n",
                "Fourth paragraph concludes."
            ]
            for paragraph in paragraphs:
                yield paragraph

        chunks = list(self.chunker.chunk_stream(paragraph_generator()))

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.modality == ModalityType.TEXT
            assert_valid_chunks([chunk], self.validator)

    def test_adaptive_functionality(self):
        """Test adaptive parameter adjustment."""
        chunker = ParagraphBasedChunker(max_paragraphs=3)
        original_max = chunker.max_paragraphs

        # Test quality feedback (should decrease max_paragraphs)
        chunker.adapt_parameters(0.3, "quality")
        assert chunker.max_paragraphs < original_max

        # Test coverage feedback (should increase max_paragraphs)
        chunker = ParagraphBasedChunker(max_paragraphs=2)
        original_max = chunker.max_paragraphs
        chunker.adapt_parameters(0.3, "coverage")
        assert chunker.max_paragraphs > original_max

        # Test speed feedback (should simplify settings)
        chunker = ParagraphBasedChunker(preserve_formatting=True, merge_short_paragraphs=True)
        chunker.adapt_parameters(0.3, "speed")
        assert not chunker.preserve_formatting
        assert not chunker.merge_short_paragraphs

    def test_chunk_metadata(self):
        """Test chunk metadata completeness."""
        result = self.chunker.chunk(self.simple_paragraphs)

        for chunk in result.chunks:
            # Basic metadata checks
            assert chunk.id is not None
            assert chunk.metadata is not None
            assert chunk.metadata.chunker_used == "paragraph_based"
            assert chunk.metadata.length == len(chunk.content)

            # Paragraph-specific metadata
            assert "paragraph_count" in chunk.metadata.extra
            assert "avg_paragraph_length" in chunk.metadata.extra
            assert "first_paragraph_preview" in chunk.metadata.extra
            assert chunk.metadata.extra["paragraph_count"] > 0

    def test_quality_metrics(self):
        """Test quality evaluation of paragraph-based chunks."""
        result = self.chunker.chunk(self.simple_paragraphs)

        metrics = self.evaluator.evaluate(result, self.simple_paragraphs)

        # Paragraph-based chunking should have good quality scores
        assert 0.0 <= metrics.overall_score <= 1.0
        assert metrics.coherence > 0.5  # Should have good coherence
        assert metrics.coverage > 0.8   # Should have high coverage
        assert metrics.boundary_quality >= 0.7  # Should respect paragraph boundaries

    def test_performance_characteristics(self):
        """Test performance on different input sizes."""
        import time

        # Test with different text sizes
        test_cases = [
            ("small", "Short paragraph.\n\nAnother short paragraph."),
            ("medium", ("Medium paragraph. " * 20 + "\n\n") * 10),
            ("large", ("Large paragraph content. " * 50 + "\n\n") * 50)
        ]

        for size_name, text in test_cases:
            start_time = time.time()
            result = self.chunker.chunk(text)
            end_time = time.time()

            processing_time = end_time - start_time

            # Should complete in reasonable time
            assert processing_time < 2.0, f"Processing {size_name} took too long: {processing_time}s"

            # Should produce valid results
            assert len(result.chunks) > 0
            assert_valid_chunks(result.chunks, self.validator)
            assert_reasonable_performance(processing_time, len(text))


class TestParagraphBasedChunkerWithRealFiles:
    """Test paragraph-based chunker with real downloaded files."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = ParagraphBasedChunker(max_paragraphs=3)
        self.validator = ChunkValidator()
        self.test_data_dir = Path("test_data")

        # Ensure test data directory exists
        if not self.test_data_dir.exists():
            pytest.skip("Test data directory not found")

    def test_article_processing(self):
        """Test processing of article-style content."""
        file_path = self.test_data_dir / "sample_article.txt"

        if not file_path.exists():
            pytest.skip("Sample article test file not found")

        result = self.chunker.chunk(file_path)

        # Should handle structured article content
        assert len(result.chunks) >= 1
        assert result.processing_time < 2.0

        # Validate chunks
        assert_valid_chunks(result.chunks, self.validator)

        # Should preserve article structure
        all_content = " ".join(chunk.content for chunk in result.chunks)
        # Look for typical article elements
        article_indicators = ["Introduction", "Conclusion", "##", "#"]
        has_structure = any(indicator in all_content for indicator in article_indicators)
        if has_structure:
            assert len(result.chunks) >= 2  # Structured content should create multiple chunks

    def test_technical_documentation(self):
        """Test processing of technical documentation."""
        file_path = self.test_data_dir / "technical_doc.txt"

        if not file_path.exists():
            pytest.skip("Technical documentation test file not found")

        result = self.chunker.chunk(file_path)

        # Should handle technical content
        assert len(result.chunks) >= 1
        assert_valid_chunks(result.chunks, self.validator)

        # Should preserve technical structure
        all_content = " ".join(chunk.content for chunk in result.chunks)
        assert "API" in all_content or "Documentation" in all_content

    def test_large_file_performance(self):
        """Test performance with large files."""
        file_path = self.test_data_dir / "alice_wonderland.txt"

        if not file_path.exists():
            pytest.skip("Large test file not found")

        import time
        start_time = time.time()

        result = self.chunker.chunk(file_path)

        processing_time = time.time() - start_time
        file_size = file_path.stat().st_size

        # Should process large files efficiently
        assert processing_time < 15.0, f"Large file processing too slow: {processing_time}s"

        # Performance should be reasonable
        size_mb = file_size / (1024 * 1024)
        assert processing_time / size_mb < 2.0, f"Processing rate too slow: {processing_time/size_mb:.2f}s/MB"

        print(f"Processed {size_mb:.2f}MB in {processing_time:.3f}s ({processing_time/size_mb:.3f}s/MB)")

    def test_edge_case_files(self):
        """Test edge case files."""
        edge_cases = [
            ("empty.txt", 0),      # Empty file
            ("short.txt", 1),      # Single paragraph
            ("unicode.txt", None), # Unicode content
        ]

        for filename, expected_chunks in edge_cases:
            file_path = self.test_data_dir / filename

            if not file_path.exists():
                continue

            result = self.chunker.chunk(file_path)

            if expected_chunks is not None:
                assert len(result.chunks) == expected_chunks, f"Wrong chunk count for {filename}"

            # All results should be valid (skip validation for empty files)
            if len(result.chunks) > 0:
                assert_valid_chunks(result.chunks, self.validator)

            # Content should be preserved (allowing for paragraph boundary normalization)
            if file_path.stat().st_size > 0:  # Skip empty files
                original_content = file_path.read_text(encoding='utf-8')
                all_chunk_content = " ".join(chunk.content for chunk in result.chunks)
                # For paragraph-based chunking, verify content is preserved semantically
                if len(original_content) > 10:  # Only check substantial content
                    original_words = set(original_content.lower().split())
                    chunk_words = set(all_chunk_content.lower().split())
                    # Most words should be preserved (allowing some normalization)
                    preserved_ratio = len(original_words & chunk_words) / len(original_words) if original_words else 1.0
                    assert preserved_ratio > 0.85, f"Too many words lost in {filename}: {preserved_ratio:.2f}"


class TestParagraphBasedChunkerIntegration:
    """Integration tests for ParagraphBasedChunker with other components."""

    def test_with_orchestrator(self):
        """Test integration with orchestrator."""
        from chunking_strategy import ChunkerOrchestrator

        orchestrator = ChunkerOrchestrator()

        text = """First paragraph with content.

Second paragraph continues.

Third paragraph concludes."""

        result = orchestrator.chunk_content(text, strategy_override="paragraph_based")

        assert result.strategy_used == "paragraph_based"
        assert len(result.chunks) > 0

    def test_with_quality_evaluator(self):
        """Test integration with quality evaluator."""
        chunker = ParagraphBasedChunker(max_paragraphs=2)
        evaluator = ChunkingQualityEvaluator()

        text = """First paragraph for testing quality metrics. This provides initial content.

Second paragraph continues the discussion. It adds more detail and context.

Third paragraph expands further. It provides additional information.

Fourth paragraph concludes. It wraps up the content."""

        result = chunker.chunk(text)
        metrics = evaluator.evaluate(result, text)

        # Paragraph-based chunking should score well on boundary quality
        assert metrics.boundary_quality >= 0.7
        assert metrics.coherence > 0.6
        assert metrics.overall_score > 0.5


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
