"""
Unit tests for DOC/DOCX/ODT/RTF document chunking strategy.

Tests cover various chunking strategies, multiple backends,
edge cases, and integration with the orchestrator.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from chunking_strategy.strategies.document.doc_chunker import DocChunker
from chunking_strategy.core.base import Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import get_registry, create_chunker
from chunking_strategy import ChunkerOrchestrator


class TestDocChunkerRegistration:
    """Test chunker registration and metadata."""

    def test_chunker_is_registered(self):
        """Test that DocChunker is properly registered."""
        from chunking_strategy.core.registry import list_chunkers
        chunkers = list_chunkers()
        assert "doc_chunker" in chunkers

        # Test that create_chunker works (may return None if no backends available)
        chunker = create_chunker("doc_chunker")
        if chunker is not None:
            assert isinstance(chunker, DocChunker)
            assert chunker.name == "doc_chunker"
            assert chunker.category == "document"
        else:
            # Expected if no backends are available
            pytest.skip("No document processing backends available")

    def test_create_chunker_factory(self):
        """Test creating chunker via factory function."""
        chunker = create_chunker("doc_chunker")
        if chunker is not None:
            assert isinstance(chunker, DocChunker)
            assert chunker.name == "doc_chunker"
            assert chunker.category == "document"
        else:
            pytest.skip("No document processing backends available")


class TestDocChunkerInitialization:
    """Test chunker initialization and configuration."""

    def create_test_chunker(self):
        """Create a test chunker with fallback backend."""
        class TestDocChunker(DocChunker):
            def _setup_backend(self):
                self.backend = 'fallback'
        return TestDocChunker()

    def test_default_initialization(self):
        """Test default initialization parameters."""
        chunker = self.create_test_chunker()
        assert chunker.name == "doc_chunker"
        assert chunker.category == "document"
        assert chunker.chunk_by == "paragraphs"
        assert chunker.paragraphs_per_chunk == 5
        assert chunker.preserve_formatting is True
        assert chunker.extract_tables is True
        assert chunker.min_chunk_size == 100
        assert chunker.max_chunk_size == 10000
        assert chunker.chunk_overlap == 50

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        class TestDocChunker(DocChunker):
            def _setup_backend(self):
                self.backend = 'fallback'

        chunker = TestDocChunker(
            chunk_by="sections",
            paragraphs_per_chunk=3,
            preserve_formatting=False,
            extract_tables=False,
            min_chunk_size=50,
            max_chunk_size=5000,
            chunk_overlap=25
        )
        assert chunker.chunk_by == "sections"
        assert chunker.paragraphs_per_chunk == 3
        assert chunker.preserve_formatting is False
        assert chunker.extract_tables is False
        assert chunker.min_chunk_size == 50
        assert chunker.max_chunk_size == 5000
        assert chunker.chunk_overlap == 25


class TestDocChunkerFormats:
    """Test format detection and handling."""

    def create_test_chunker(self):
        class TestDocChunker(DocChunker):
            def _setup_backend(self):
                self.backend = 'fallback'
        return TestDocChunker()

    def test_format_detection(self):
        """Test document format detection."""
        chunker = self.create_test_chunker()

        assert chunker._detect_document_format(Path("test.docx")) == "docx"
        assert chunker._detect_document_format(Path("test.doc")) == "doc"
        assert chunker._detect_document_format(Path("test.odt")) == "odt"
        assert chunker._detect_document_format(Path("test.rtf")) == "rtf"
        assert chunker._detect_document_format(Path("test.unknown")) == "unknown"


class TestDocChunkerStrategies:
    """Test different chunking strategies."""

    def create_test_chunker(self, **kwargs):
        class TestDocChunker(DocChunker):
            def _setup_backend(self):
                self.backend = 'fallback'
        return TestDocChunker(**kwargs)

    def create_test_document_data(self):
        """Create test document data structure."""
        return {
            "paragraphs": [
                {"text": "First paragraph content", "style": "Normal", "is_heading": False},
                {"text": "Second paragraph content", "style": "Normal", "is_heading": False},
                {"text": "Main Heading", "style": "Heading 1", "is_heading": True},
                {"text": "Third paragraph under heading", "style": "Normal", "is_heading": False},
                {"text": "Fourth paragraph content", "style": "Normal", "is_heading": False},
                {"text": "Subheading", "style": "Heading 2", "is_heading": True},
                {"text": "Fifth paragraph content", "style": "Normal", "is_heading": False},
            ],
            "tables": [
                {
                    "index": 0,
                    "data": [["Header 1", "Header 2"], ["Row 1 Col 1", "Row 1 Col 2"]],
                    "text": "Header 1 | Header 2\nRow 1 Col 1 | Row 1 Col 2"
                }
            ],
            "headings": [
                {"text": "Main Heading", "level": 1, "paragraph_index": 2},
                {"text": "Subheading", "level": 2, "paragraph_index": 5}
            ],
            "paragraph_count": 7,
            "backend": "test"
        }

    def test_paragraph_chunking(self):
        """Test paragraph-based chunking."""
        chunker = self.create_test_chunker(chunk_by="paragraphs", paragraphs_per_chunk=2, min_chunk_size=10)
        document_data = self.create_test_document_data()
        source_info = {"source": "test", "source_type": "content"}

        chunks = chunker._chunk_by_paragraphs(document_data, source_info)

        assert len(chunks) >= 4  # 7 paragraphs / 2 per chunk + 1 table = 4+ chunks

        # Check chunk properties
        for chunk in chunks[:3]:  # Check text chunks
            assert isinstance(chunk, Chunk)
            assert chunk.modality == ModalityType.TEXT
            assert "paragraph_count" in chunk.metadata.extra
            assert chunk.metadata.extra["chunk_type"] == "paragraphs"

    def test_section_chunking(self):
        """Test section-based chunking."""
        chunker = self.create_test_chunker(chunk_by="sections", heading_levels=[1, 2], min_chunk_size=10)
        document_data = self.create_test_document_data()
        source_info = {"source": "test", "source_type": "content"}

        chunks = chunker._chunk_by_sections(document_data, source_info)

        assert len(chunks) >= 2  # Should have multiple sections + table

        # Check that sections are created
        section_chunks = [c for c in chunks if c.metadata.extra.get("chunk_type") == "section"]
        assert len(section_chunks) > 0

        for chunk in section_chunks:
            assert "section_heading" in chunk.metadata.extra
            assert "heading_level" in chunk.metadata.extra

    def test_content_size_chunking(self):
        """Test content size-based chunking."""
        chunker = self.create_test_chunker(chunk_by="content_size", max_chunk_size=100, min_chunk_size=10)
        document_data = self.create_test_document_data()
        source_info = {"source": "test", "source_type": "content"}

        chunks = chunker._chunk_by_content_size(document_data, source_info)

        assert len(chunks) > 0

        # Check size constraints
        for chunk in chunks:
            if chunk.metadata.extra.get("chunk_type") == "content_size":
                assert len(chunk.content) <= chunker.max_chunk_size
                assert len(chunk.content) >= chunker.min_chunk_size

    def test_table_chunk_creation(self):
        """Test table chunk creation."""
        chunker = self.create_test_chunker()
        table_data = {
            "index": 0,
            "data": [["Name", "Age"], ["John", "25"], ["Jane", "30"]],
            "text": "Name | Age\nJohn | 25\nJane | 30"
        }
        source_info = {"source": "test", "source_type": "content"}

        chunk = chunker._create_table_chunk(table_data, source_info)

        assert isinstance(chunk, Chunk)
        assert chunk.modality == ModalityType.TABLE
        assert chunk.metadata.extra["chunk_type"] == "table"
        assert chunk.metadata.extra["table_index"] == 0
        assert chunk.metadata.extra["row_count"] == 3
        assert chunk.metadata.extra["column_count"] == 2


class TestDocChunkerRealDocFile:
    """Test DocChunker with real .doc file from test_data."""

    def create_test_chunker(self):
        class TestDocChunker(DocChunker):
            def _setup_backend(self):
                self.backend = 'fallback'
        return TestDocChunker(min_chunk_size=10)

    def get_test_doc_file(self):
        """Get the test .doc file path if it exists."""
        from pathlib import Path
        test_data_dir = Path(__file__).parent.parent / "test_data"

        # Look for any .doc file in test_data
        for doc_file in test_data_dir.glob("*.doc"):
            return doc_file

        return None

    def test_real_doc_file_chunking(self):
        """Test chunking a real .doc file."""
        doc_file = self.get_test_doc_file()
        if doc_file is None:
            pytest.skip("No .doc file found in test_data directory")

        chunker = self.create_test_chunker()

        # Test chunking the real .doc file
        result = chunker.chunk(str(doc_file))

        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "doc_chunker"
        assert result.total_chunks > 0

        # Validate source info contains file information
        assert result.source_info["source_type"] == "file"
        assert str(doc_file) in result.source_info["source"]

        # Check that chunks contain actual content
        total_content = 0
        for chunk in result.chunks:
            assert chunk.content.strip(), "Chunks should contain non-empty content"
            total_content += len(chunk.content.strip())

        # Should have extracted reasonable amount of content
        assert total_content > 100, f"Total content too short: {total_content} characters"

    def test_real_doc_file_different_strategies(self):
        """Test real .doc file with different chunking strategies."""
        doc_file = self.get_test_doc_file()
        if doc_file is None:
            pytest.skip("No .doc file found in test_data directory")

        strategies = [
            {"chunk_by": "paragraphs", "paragraphs_per_chunk": 2},
            {"chunk_by": "sections", "heading_levels": [1, 2]},
            {"chunk_by": "content_size", "max_chunk_size": 500, "min_chunk_size": 100}
        ]

        for i, strategy_config in enumerate(strategies):
            chunker = self.create_test_chunker()
            # Update chunker configuration
            for key, value in strategy_config.items():
                setattr(chunker, key, value)

            result = chunker.chunk(str(doc_file))

            assert isinstance(result, ChunkingResult)
            assert result.total_chunks > 0, f"Strategy {i} produced no chunks"

            # Check chunks have content
            for chunk in result.chunks:
                assert chunk.content.strip(), f"Empty chunk in strategy {i}"

    def test_real_doc_file_format_detection(self):
        """Test format detection with real .doc file."""
        doc_file = self.get_test_doc_file()
        if doc_file is None:
            pytest.skip("No .doc file found in test_data directory")

        chunker = self.create_test_chunker()

        # Test format detection
        detected_format = chunker._detect_document_format(doc_file)
        assert detected_format == "doc", f"Expected 'doc', got '{detected_format}'"

    def test_real_doc_file_metadata_extraction(self):
        """Test metadata extraction from real .doc file."""
        doc_file = self.get_test_doc_file()
        if doc_file is None:
            pytest.skip("No .doc file found in test_data directory")

        chunker = self.create_test_chunker()
        result = chunker.chunk(str(doc_file))

        assert isinstance(result, ChunkingResult)

        # Check that chunks have proper metadata
        for chunk in result.chunks:
            assert isinstance(chunk.metadata, ChunkMetadata)
            # source_type can be "text", "table" (normal extraction) or "file" (fallback extraction)
            assert chunk.metadata.source_type in ["text", "table", "file"]
            assert "chunk_type" in chunk.metadata.extra

        # Check source info has backend information
        assert "backend_used" in result.source_info

    def test_real_doc_file_performance(self):
        """Test performance characteristics with real .doc file."""
        doc_file = self.get_test_doc_file()
        if doc_file is None:
            pytest.skip("No .doc file found in test_data directory")

        chunker = self.create_test_chunker()

        import time
        start_time = time.time()
        result = chunker.chunk(str(doc_file))
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete in reasonable time (less than 30 seconds for any .doc file)
        assert processing_time < 30, f"Processing took too long: {processing_time:.2f}s"

        # Should have reasonable processing time recorded
        assert result.processing_time > 0
        assert abs(result.processing_time - processing_time) < 1  # Should be close


class TestDocChunkerInputTypes:
    """Test different input types."""

    def create_test_chunker(self):
        class TestDocChunker(DocChunker):
            def _setup_backend(self):
                self.backend = 'fallback'
        return TestDocChunker(min_chunk_size=10)

    def test_string_content(self):
        """Test chunking string content."""
        chunker = self.create_test_chunker()
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        result = chunker.chunk(content)
        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "doc_chunker"
        assert result.total_chunks > 0

    def test_bytes_content(self):
        """Test chunking bytes content."""
        chunker = self.create_test_chunker()
        content = b"First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        result = chunker.chunk(content)
        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "doc_chunker"

    def test_file_path_string(self):
        """Test chunking with file path as string."""
        chunker = self.create_test_chunker()

        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content.\n\nSecond paragraph.\n\nThird paragraph.")
            temp_path = f.name

        try:
            result = chunker.chunk(temp_path)
            assert isinstance(result, ChunkingResult)
            assert result.source_info["source_type"] == "file"
            assert temp_path in result.source_info["source"]
        finally:
            Path(temp_path).unlink()

    def test_path_object(self):
        """Test chunking with Path object."""
        chunker = self.create_test_chunker()

        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test path content.\n\nSecond paragraph.\n\nThird paragraph.")
            temp_path = Path(f.name)

        try:
            result = chunker.chunk(temp_path)
            assert isinstance(result, ChunkingResult)
            assert result.source_info["source_type"] == "file"
        finally:
            temp_path.unlink()

    def test_invalid_content_type(self):
        """Test error handling for invalid content type."""
        chunker = self.create_test_chunker()

        with pytest.raises(ValueError, match="Unsupported content type"):
            chunker.chunk(123)  # Invalid type


class TestDocChunkerEdgeCases:
    """Test edge cases and error handling."""

    def create_test_chunker(self):
        class TestDocChunker(DocChunker):
            def _setup_backend(self):
                self.backend = 'fallback'
        return TestDocChunker(min_chunk_size=10)

    def test_empty_content(self):
        """Test handling of empty content."""
        chunker = self.create_test_chunker()

        result = chunker.chunk("")
        assert isinstance(result, ChunkingResult)
        # Should have 0 or very few chunks for empty content
        assert result.total_chunks <= 1

    def test_very_short_content(self):
        """Test handling of very short content."""
        chunker = self.create_test_chunker()

        result = chunker.chunk("Short.")
        assert isinstance(result, ChunkingResult)
        # Should handle short content gracefully

    def test_large_content(self):
        """Test handling of large content."""
        chunker = self.create_test_chunker()

        # Create large content with multiple paragraphs
        large_content = "\n\n".join([f"Paragraph {i}. " + "Content " * 50 for i in range(20)])
        result = chunker.chunk(large_content)

        assert isinstance(result, ChunkingResult)
        assert result.total_chunks > 1

    def test_invalid_file_path(self):
        """Test error handling for non-existent file."""
        chunker = self.create_test_chunker()

        # This should be treated as content, not file path
        result = chunker.chunk("nonexistent_file.docx")
        assert isinstance(result, ChunkingResult)


class TestDocChunkerStreaming:
    """Test streaming capabilities."""

    def create_test_chunker(self):
        class TestDocChunker(DocChunker):
            def _setup_backend(self):
                self.backend = 'fallback'
        return TestDocChunker(min_chunk_size=10)

    def test_can_stream(self):
        """Test that chunker supports streaming."""
        chunker = self.create_test_chunker()
        assert chunker.can_stream() is True

    def test_chunk_stream(self):
        """Test streaming chunk processing."""
        chunker = self.create_test_chunker()

        # Simulate streaming content
        content_parts = [
            "First paragraph content.\n\n",
            "Second paragraph content.\n\n",
            "Third paragraph content."
        ]

        chunks = list(chunker.chunk_stream(content_parts))
        assert len(chunks) > 0

        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.modality in [ModalityType.TEXT, ModalityType.TABLE]


class TestDocChunkerAdaptation:
    """Test adaptive behavior."""

    def create_test_chunker(self):
        class TestDocChunker(DocChunker):
            def _setup_backend(self):
                self.backend = 'fallback'
        return TestDocChunker()

    def test_adapt_parameters_quality(self):
        """Test parameter adaptation based on quality feedback."""
        chunker = self.create_test_chunker()

        original_paragraphs = chunker.paragraphs_per_chunk
        original_max_size = chunker.max_chunk_size

        # Low quality score should reduce sizes
        chunker.adapt_parameters(0.3, "quality")
        assert chunker.paragraphs_per_chunk < original_paragraphs
        assert chunker.max_chunk_size < original_max_size

    def test_adapt_parameters_performance(self):
        """Test parameter adaptation based on performance feedback."""
        chunker = self.create_test_chunker()

        original_paragraphs = chunker.paragraphs_per_chunk
        original_max_size = chunker.max_chunk_size

        # Poor performance should increase sizes
        chunker.adapt_parameters(0.2, "performance")
        assert chunker.paragraphs_per_chunk >= original_paragraphs
        assert chunker.max_chunk_size >= original_max_size

    def test_adaptation_history(self):
        """Test adaptation history tracking."""
        chunker = self.create_test_chunker()

        # Initially no history
        history = chunker.get_adaptation_history()
        assert len(history) == 0

        # After adaptation, should have history
        chunker.adapt_parameters(0.5, "quality")
        history = chunker.get_adaptation_history()
        assert len(history) == 1
        assert history[0]["feedback_score"] == 0.5
        assert history[0]["feedback_type"] == "quality"

    def test_specific_feedback_adaptation(self):
        """Test adaptation to specific feedback."""
        chunker = self.create_test_chunker()

        original_paragraphs = chunker.paragraphs_per_chunk
        original_max_size = chunker.max_chunk_size

        # Chunks too large feedback
        chunker.adapt_parameters(0.5, "quality", chunks_too_large=True)
        assert chunker.paragraphs_per_chunk < original_paragraphs
        assert chunker.max_chunk_size < original_max_size


class TestDocChunkerBackends:
    """Test different backend handling."""

    def test_fallback_backend(self):
        """Test fallback backend works."""
        class TestDocChunker(DocChunker):
            def _setup_backend(self):
                self.backend = 'fallback'

        chunker = TestDocChunker(min_chunk_size=10)
        content = "Test content.\n\nSecond paragraph.\n\nThird paragraph."

        result = chunker.chunk(content)
        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "doc_chunker"
        assert result.source_info["backend_used"] == "fallback"

    def test_table_formatting(self):
        """Test table formatting functionality."""
        class TestDocChunker(DocChunker):
            def _setup_backend(self):
                self.backend = 'fallback'

        chunker = TestDocChunker()
        table_data = [
            ["Name", "Age", "City"],
            ["John", "25", "New York"],
            ["Jane", "30", "Boston"]
        ]

        formatted = chunker._format_table_as_text(table_data)
        assert "Name" in formatted
        assert "John" in formatted
        assert "|" in formatted  # Should have column separators


class TestDocChunkerOrchestrator:
    """Test integration with ChunkerOrchestrator."""

    def test_orchestrator_auto_selection_docx(self):
        """Test orchestrator auto-selects DOC chunker for DOCX files."""
        config = {"strategies": {"primary": "auto"}}
        orchestrator = ChunkerOrchestrator(config=config)

        # Create temporary DOCX file with substantial content
        docx_content = "Test Document\n\n" + "Paragraph content. " * 50

        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False) as f:
            f.write(docx_content)
            temp_path = f.name

        try:
            # Test auto-selection logic
            docx_info = {
                'file_extension': '.docx',
                'file_size': len(docx_content.encode('utf-8'))
            }
            primary, fallbacks = orchestrator._auto_select_strategy(docx_info)
            assert primary == "doc"
            from chunking_strategy.orchestrator import STRATEGY_NAME_MAPPING
            assert "doc_chunker" in STRATEGY_NAME_MAPPING.get("doc", "")
        finally:
            Path(temp_path).unlink()

    def test_orchestrator_explicit_strategy(self):
        """Test orchestrator with explicit DOC chunker strategy."""
        config = {"strategies": {"primary": "doc_chunker"}}
        orchestrator = ChunkerOrchestrator(config=config)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for explicit strategy.\n\nSecond paragraph.")
            temp_path = f.name

        try:
            # This should attempt to use doc_chunker but may fall back due to missing backends
            result = orchestrator.chunk_file(temp_path)
            # The actual strategy used depends on whether backends are available
            assert isinstance(result, ChunkingResult)
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
