"""
Basic tests for ContextEnrichedChunker to validate core functionality.
"""

import pytest
from chunking_strategy.strategies.general.context_enriched_chunker import ContextEnrichedChunker
from chunking_strategy.core.base import ModalityType, ChunkingResult


class TestContextEnrichedBasic:
    """Basic test suite for ContextEnrichedChunker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = ContextEnrichedChunker(
            target_chunk_size=1000,
            min_chunk_size=200,
            max_chunk_size=2000
        )

        self.sample_text = """
        Machine Learning Applications. Machine learning transforms how we process data.
        Neural networks learn patterns from large datasets automatically.

        Natural Language Processing. NLP helps computers understand human language.
        Chatbots use NLP to generate conversational responses.

        Computer Vision Systems. Computer vision interprets visual information.
        Image recognition identifies objects in photographs accurately.
        """

    def test_initialization(self):
        """Test basic initialization."""
        chunker = ContextEnrichedChunker()
        assert chunker.target_chunk_size == 2000
        assert chunker.enable_ner == True
        assert chunker.enable_topic_modeling == True

    def test_basic_chunking(self):
        """Test basic chunking functionality."""
        result = self.chunker.chunk(self.sample_text)

        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "context_enriched"
        assert len(result.chunks) >= 1

        # Check first chunk has required metadata
        chunk = result.chunks[0]
        assert chunk.modality == ModalityType.TEXT
        assert len(chunk.content) > 0
        assert hasattr(chunk.metadata, 'extra')
        assert 'entities' in chunk.metadata.extra
        assert 'topics' in chunk.metadata.extra

    def test_empty_content(self):
        """Test handling of empty content."""
        result = self.chunker.chunk("")
        assert len(result.chunks) == 0
        assert result.processing_time >= 0

    def test_short_content(self):
        """Test handling of short content."""
        result = self.chunker.chunk("Short text.")
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "Short text."

    def test_sentence_segmentation(self):
        """Test sentence segmentation."""
        sentences = self.chunker._segment_sentences(self.sample_text)
        assert isinstance(sentences, list)
        assert len(sentences) >= 6
        assert all(len(s.strip()) > 0 for s in sentences)

    def test_supported_formats(self):
        """Test supported formats method."""
        formats = self.chunker.get_supported_formats()
        assert isinstance(formats, list)
        assert "txt" in formats
        assert "md" in formats

    def test_chunk_estimation(self):
        """Test chunk estimation."""
        estimate = self.chunker.estimate_chunks(self.sample_text)
        assert isinstance(estimate, int)
        assert estimate >= 1
