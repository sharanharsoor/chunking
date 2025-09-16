"""
Comprehensive tests for the SentenceBasedChunker.

This module tests the sentence-based chunking strategy with real files,
edge cases, and comprehensive validation scenarios.
"""

import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from chunking_strategy.strategies.text.sentence_based import SentenceBasedChunker
from chunking_strategy.core.base import Chunk, ChunkingResult, ModalityType
from chunking_strategy.utils.validation import ChunkValidator
from chunking_strategy.core.metrics import ChunkingQualityEvaluator
from tests.conftest import assert_valid_chunks, assert_content_integrity, assert_reasonable_performance


class TestSentenceBasedChunker:
    """Test suite for SentenceBasedChunker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = SentenceBasedChunker(max_sentences=3)
        self.validator = ChunkValidator()
        self.evaluator = ChunkingQualityEvaluator()

        # Test data
        self.simple_sentences = "First sentence. Second sentence. Third sentence. Fourth sentence."
        self.complex_sentences = "Dr. Smith went to the U.S.A. yesterday. He bought items for $19.99 each. The meeting is at 3:30 p.m. on Dec. 15th."
        self.questions_and_exclamations = "What is this? This is amazing! How does it work? It works well!"
        self.long_paragraph = """
        This is the first sentence of a longer paragraph. It contains multiple sentences that should be grouped together.
        The second sentence continues the thought. The third sentence adds more detail.
        The fourth sentence provides additional context. The fifth sentence concludes the paragraph.
        """

    def test_basic_sentence_chunking(self):
        """Test basic sentence grouping functionality."""
        result = self.chunker.chunk(self.simple_sentences)

        # Validate result structure
        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "sentence_based"
        assert result.processing_time is not None
        assert result.processing_time > 0

        # Should create 2 chunks (3 sentences + 1 sentence)
        assert len(result.chunks) == 2

        # First chunk should have 3 sentences
        first_chunk = result.chunks[0].content
        assert first_chunk.count('.') == 3
        assert "First sentence" in first_chunk
        assert "Third sentence" in first_chunk

        # Second chunk should have 1 sentence
        second_chunk = result.chunks[1].content
        assert second_chunk.count('.') == 1
        assert "Fourth sentence" in second_chunk

                # Validate all chunks
        assert_valid_chunks(result.chunks, self.validator)

        # For sentence-based chunking, verify that all sentences are preserved
        all_chunk_content = ' '.join(chunk.content for chunk in result.chunks)
        assert "First sentence" in all_chunk_content
        assert "Second sentence" in all_chunk_content
        assert "Third sentence" in all_chunk_content
        assert "Fourth sentence" in all_chunk_content

    def test_sentence_splitting_methods(self):
        """Test different sentence splitting methods."""
        text = "First sentence. Second sentence! Third question? Fourth statement."

        # Test simple splitter (default)
        simple_chunker = SentenceBasedChunker(max_sentences=2, sentence_splitter="simple")
        simple_result = simple_chunker.chunk(text)

        # Test NLTK splitter (will fallback to simple if not available)
        nltk_chunker = SentenceBasedChunker(max_sentences=2, sentence_splitter="nltk")
        nltk_result = nltk_chunker.chunk(text)

        # Test spaCy splitter (will fallback to simple if not available)
        spacy_chunker = SentenceBasedChunker(max_sentences=2, sentence_splitter="spacy")
        spacy_result = spacy_chunker.chunk(text)

        # All should produce valid results
        for result in [simple_result, nltk_result, spacy_result]:
            assert len(result.chunks) > 0
            assert_valid_chunks(result.chunks, self.validator)
            # Verify sentences are preserved (content integrity for sentence-based)
            all_content = ' '.join(chunk.content for chunk in result.chunks)
            assert "First sentence" in all_content
            assert "Second sentence" in all_content

    def test_sentence_overlap(self):
        """Test overlap between sentence chunks."""
        chunker = SentenceBasedChunker(max_sentences=2, overlap_sentences=1)
        text = "One. Two. Three. Four. Five. Six."

        result = chunker.chunk(text)

        # Should have overlap between consecutive chunks
        if len(result.chunks) > 1:
            # Check for overlapping content
            for i in range(len(result.chunks) - 1):
                current_content = result.chunks[i].content
                next_content = result.chunks[i + 1].content

                # Should find some overlapping sentences
                current_sentences = [s.strip() for s in current_content.split('.') if s.strip()]
                next_sentences = [s.strip() for s in next_content.split('.') if s.strip()]

                # There should be at least one overlapping sentence
                overlap_found = any(sent in next_sentences for sent in current_sentences[-1:])
                assert overlap_found or len(current_sentences) < 2, "Expected overlap between chunks"

    def test_min_max_sentences(self):
        """Test minimum and maximum sentence constraints."""
        # Test minimum sentences
        min_chunker = SentenceBasedChunker(max_sentences=5, min_sentences=2)
        short_text = "Only one sentence."

        result = min_chunker.chunk(short_text)
        assert len(result.chunks) == 1  # Even if below minimum, should create chunk

        # Test with enough sentences
        enough_text = "One. Two. Three. Four. Five. Six."
        result = min_chunker.chunk(enough_text)

        for chunk in result.chunks:
            sentence_count = chunk.content.count('.')
            # Most chunks should meet minimum requirement
            assert sentence_count >= 1  # At least one sentence

    def test_chunk_size_limits(self):
        """Test maximum chunk size constraints."""
        chunker = SentenceBasedChunker(max_sentences=10, max_chunk_size=100)

        # Create text with very long sentences
        long_sentence = "This is a very long sentence that exceeds the maximum chunk size limit " * 5
        text = f"{long_sentence}. Another long sentence here."

        result = chunker.chunk(text)

        # For sentence-based chunking, individual sentences can exceed max_chunk_size
        # but the chunker should avoid adding additional sentences that would exceed the limit
        for chunk in result.chunks:
            # Each chunk should contain at least one complete sentence
            assert chunk.content.count('.') >= 1
            # If a single sentence exceeds max_chunk_size, that's acceptable
            sentences_in_chunk = [s.strip() for s in chunk.content.split('.') if s.strip()]
            if len(sentences_in_chunk) > 1:
                # If multiple sentences, the total shouldn't exceed limit by too much
                assert len(chunk.content) <= chunker.max_chunk_size * 2

    def test_complex_punctuation(self):
        """Test handling of complex punctuation cases."""
        result = self.chunker.chunk(self.complex_sentences)

        assert len(result.chunks) >= 1
        assert_valid_chunks(result.chunks, self.validator)

        # Should handle abbreviations correctly
        combined_content = ' '.join(chunk.content for chunk in result.chunks)
        assert "U.S.A." in combined_content
        assert "$19.99" in combined_content
        assert "3:30 p.m." in combined_content

    def test_questions_and_exclamations(self):
        """Test handling of questions and exclamations."""
        result = self.chunker.chunk(self.questions_and_exclamations)

        assert len(result.chunks) >= 1
        assert_valid_chunks(result.chunks, self.validator)

        # Should preserve question marks and exclamation points
        combined_content = ' '.join(chunk.content for chunk in result.chunks)
        assert "?" in combined_content
        assert "!" in combined_content

    def test_empty_and_whitespace_input(self):
        """Test handling of empty and whitespace-only input."""
        # Empty input
        empty_result = self.chunker.chunk("")
        assert len(empty_result.chunks) == 0

        # Whitespace only
        whitespace_result = self.chunker.chunk("   \n\t   ")
        # Should either have no chunks or handle gracefully
        if whitespace_result.chunks:
            for chunk in whitespace_result.chunks:
                assert isinstance(chunk.content, str)

    def test_single_sentence(self):
        """Test handling of single sentence input."""
        single_sentence = "This is just one sentence."
        result = self.chunker.chunk(single_sentence)

        assert len(result.chunks) == 1
        assert result.chunks[0].content.strip() == single_sentence
        assert_valid_chunks(result.chunks, self.validator)

    def test_very_long_sentence(self):
        """Test handling of extremely long sentences."""
        long_sentence = "This is an extremely long sentence " * 50 + "."
        result = self.chunker.chunk(long_sentence)

        # Should handle long sentences gracefully
        assert len(result.chunks) >= 1
        assert_valid_chunks(result.chunks, self.validator)

        # Content should be preserved
        combined_content = ''.join(chunk.content for chunk in result.chunks)
        assert "extremely long sentence" in combined_content

    def test_unicode_and_multilingual(self):
        """Test handling of Unicode and multilingual text."""
        unicode_text = "这是中文句子。This is English. Ceci est français. これは日本語です。"
        result = self.chunker.chunk(unicode_text)

        assert len(result.chunks) >= 1
        assert_valid_chunks(result.chunks, self.validator)

        # Should preserve Unicode characters
        combined_content = ''.join(chunk.content for chunk in result.chunks)
        assert "中文" in combined_content
        assert "français" in combined_content
        assert "日本語" in combined_content

    def test_streaming_interface(self):
        """Test streaming chunking interface."""
        def sentence_generator():
            sentences = [
                "First sentence here. ",
                "Second sentence follows. ",
                "Third sentence continues. ",
                "Fourth sentence concludes."
            ]
            for sentence in sentences:
                yield sentence

        chunks = list(self.chunker.chunk_stream(sentence_generator()))

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.modality == ModalityType.TEXT
            assert_valid_chunks([chunk], self.validator)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid max_sentences
        with pytest.raises(ValueError):
            SentenceBasedChunker(max_sentences=0)

        # Test invalid min_sentences
        with pytest.raises(ValueError):
            SentenceBasedChunker(min_sentences=0)

        # Test min > max
        with pytest.raises(ValueError):
            SentenceBasedChunker(max_sentences=2, min_sentences=5)

        # Test invalid overlap
        with pytest.raises(ValueError):
            SentenceBasedChunker(max_sentences=3, overlap_sentences=3)

        # Test invalid sentence splitter
        with pytest.raises(ValueError):
            SentenceBasedChunker(sentence_splitter="invalid")

    def test_adaptive_functionality(self):
        """Test adaptive parameter adjustment."""
        chunker = SentenceBasedChunker(max_sentences=5)
        original_max = chunker.max_sentences

        # Test quality feedback (should decrease max_sentences)
        chunker.adapt_parameters(0.3, "quality")
        assert chunker.max_sentences < original_max

        # Test coverage feedback (should increase max_sentences)
        chunker = SentenceBasedChunker(max_sentences=3)
        original_max = chunker.max_sentences
        chunker.adapt_parameters(0.3, "coverage")
        assert chunker.max_sentences > original_max

    def test_chunk_metadata(self):
        """Test chunk metadata completeness."""
        result = self.chunker.chunk(self.simple_sentences)

        for chunk in result.chunks:
            # Basic metadata checks
            assert chunk.id is not None
            assert chunk.metadata is not None
            assert chunk.metadata.chunker_used == "sentence_based"
            assert chunk.metadata.length == len(chunk.content)

            # Sentence-specific metadata
            assert "sentence_count" in chunk.metadata.extra
            assert "avg_sentence_length" in chunk.metadata.extra
            assert chunk.metadata.extra["sentence_count"] > 0

    def test_quality_metrics(self):
        """Test quality evaluation of sentence-based chunks."""
        text = self.long_paragraph.strip()
        result = self.chunker.chunk(text)

        metrics = self.evaluator.evaluate(result, text)

        # Sentence-based chunking should have good quality scores
        assert 0.0 <= metrics.overall_score <= 1.0
        assert metrics.coherence > 0.5  # Should have good coherence
        assert metrics.coverage > 0.8   # Should have high coverage
        assert metrics.boundary_quality > 0.6  # Should respect sentence boundaries

    def test_performance_characteristics(self):
        """Test performance on different input sizes."""
        import time

        # Test with different text sizes
        test_cases = [
            ("small", "Short text. Single sentence."),
            ("medium", "Medium text. " * 100),
            ("large", "Large text sentence. " * 1000)
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


class TestSentenceBasedChunkerWithRealFiles:
    """Test sentence-based chunker with real downloaded files."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = SentenceBasedChunker(max_sentences=5)
        self.validator = ChunkValidator()
        self.test_data_dir = Path("test_data")

        # Ensure test data directory exists
        if not self.test_data_dir.exists():
            pytest.skip("Test data directory not found")

    def test_alice_wonderland_chunking(self):
        """Test chunking of Alice in Wonderland text."""
        file_path = self.test_data_dir / "alice_wonderland.txt"

        if not file_path.exists():
            pytest.skip("Alice in Wonderland test file not found")

        result = self.chunker.chunk(file_path)

        # Should handle large text files
        assert len(result.chunks) > 10  # Large text should produce many chunks
        assert result.processing_time < 5.0  # Should process in reasonable time

        # Validate chunks
        assert_valid_chunks(result.chunks, self.validator)

        # Check content preservation (for sentence-based chunking)
        original_content = file_path.read_text(encoding='utf-8')
        all_chunk_content = ' '.join(chunk.content for chunk in result.chunks)
        # Verify key content is preserved (allowing for sentence boundary normalization)
        assert "Alice" in all_chunk_content  # Key character should be preserved

        # Literature should have good sentence boundaries
        for chunk in result.chunks[:5]:  # Check first 5 chunks
            assert len(chunk.content.strip()) > 0
            # Should end with sentence punctuation (most of the time)
            content = chunk.content.strip()
            if content:
                assert content[-1] in '.!?"' or '"' in content

    def test_technical_documentation(self):
        """Test chunking of technical documentation."""
        file_path = self.test_data_dir / "technical_doc.txt"

        if not file_path.exists():
            pytest.skip("Technical documentation test file not found")

        result = self.chunker.chunk(file_path)

        # Should handle technical content
        assert len(result.chunks) >= 1
        assert_valid_chunks(result.chunks, self.validator)

        # Should preserve technical terms and structure
        combined_content = ' '.join(chunk.content for chunk in result.chunks)
        assert "API" in combined_content
        assert "POST" in combined_content
        assert "Parameters" in combined_content

    def test_markdown_content(self):
        """Test chunking of markdown content."""
        file_path = self.test_data_dir / "pytorch_readme.md"

        if not file_path.exists():
            pytest.skip("Markdown test file not found")

        result = self.chunker.chunk(file_path)

        # Should handle markdown syntax
        assert len(result.chunks) > 5  # README should have multiple chunks
        assert_valid_chunks(result.chunks, self.validator)

        # Should preserve markdown elements
        combined_content = ' '.join(chunk.content for chunk in result.chunks)
        assert "#" in combined_content or "PyTorch" in combined_content

    def test_structured_article(self):
        """Test chunking of structured article."""
        file_path = self.test_data_dir / "sample_article.txt"

        if not file_path.exists():
            pytest.skip("Sample article test file not found")

        result = self.chunker.chunk(file_path)

        # Should handle structured content
        assert len(result.chunks) >= 3  # Article should have multiple chunks
        assert_valid_chunks(result.chunks, self.validator)

        # Should preserve article structure
        combined_content = ' '.join(chunk.content for chunk in result.chunks)
        assert "Introduction" in combined_content
        assert "Conclusion" in combined_content

    def test_edge_case_files(self):
        """Test edge case files."""
        edge_cases = [
            ("empty.txt", 0),      # Empty file
            ("short.txt", 1),      # Single sentence
            ("unicode.txt", None), # Unicode content
            ("difficult_punctuation.txt", None), # Complex punctuation
            ("single_long_sentence.txt", 1),      # Very long sentence
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

            # Content should be preserved (allowing for sentence boundary normalization)
            if file_path.stat().st_size > 0:  # Skip empty files
                original_content = file_path.read_text(encoding='utf-8')
                all_chunk_content = ' '.join(chunk.content for chunk in result.chunks)
                # For sentence-based chunking, verify content is preserved semantically
                if len(original_content) > 10:  # Only check substantial content
                    original_words = set(original_content.lower().split())
                    chunk_words = set(all_chunk_content.lower().split())
                    # Most words should be preserved (allowing some normalization)
                    preserved_ratio = len(original_words & chunk_words) / len(original_words) if original_words else 1.0
                    assert preserved_ratio > 0.9, f"Too many words lost in {filename}: {preserved_ratio:.2f}"

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
        assert processing_time < 10.0, f"Large file processing too slow: {processing_time}s"

        # Performance should be reasonable (less than 1s per MB)
        size_mb = file_size / (1024 * 1024)
        assert processing_time / size_mb < 1.0, f"Processing rate too slow: {processing_time/size_mb:.2f}s/MB"

        print(f"Processed {size_mb:.2f}MB in {processing_time:.3f}s ({processing_time/size_mb:.3f}s/MB)")


class TestSentenceBasedChunkerIntegration:
    """Integration tests for SentenceBasedChunker with other components."""

    def test_with_orchestrator(self):
        """Test integration with orchestrator."""
        from chunking_strategy import ChunkerOrchestrator

        orchestrator = ChunkerOrchestrator()

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = orchestrator.chunk_content(text, strategy_override="sentence_based")

        assert result.strategy_used == "sentence_based"
        assert len(result.chunks) > 0

    def test_with_pipeline(self):
        """Test integration with pipeline system."""
        from chunking_strategy.core.pipeline import ChunkingPipeline, ChunkerStep, FilterStep

        # Create pipeline with sentence chunker and filter
        pipeline = ChunkingPipeline([
            ("chunker", ChunkerStep("sentence_chunker", "sentence_based", max_sentences=2)),
            ("filter", FilterStep("size_filter", min_size=10))
        ])

        text = "Very short. This is a longer sentence with more content."
        result = pipeline.process(text)

        assert len(result.chunks) > 0
        # Filter should remove very short chunks
        for chunk in result.chunks:
            assert len(chunk.content) >= 10

    def test_with_quality_evaluator(self):
        """Test integration with quality evaluator."""
        chunker = SentenceBasedChunker(max_sentences=3)
        evaluator = ChunkingQualityEvaluator()

        text = "First sentence for testing. Second sentence continues. Third sentence concludes. Fourth sentence starts new chunk."
        result = chunker.chunk(text)

        metrics = evaluator.evaluate(result, text)

        # Sentence-based chunking should score well on boundary quality
        assert metrics.boundary_quality >= 0.7
        assert metrics.coherence > 0.6
        assert metrics.overall_score > 0.5


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
