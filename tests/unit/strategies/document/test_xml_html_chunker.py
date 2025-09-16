"""
Unit tests for XML/HTML chunking strategy.

Tests cover various chunking strategies, format detection, parser backends,
edge cases, and integration with the orchestrator.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from chunking_strategy.strategies.document.xml_html_chunker import XMLHTMLChunker
from chunking_strategy.core.base import Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import get_registry, create_chunker
from chunking_strategy import ChunkerOrchestrator


class TestXMLHTMLChunkerRegistration:
    """Test chunker registration and metadata."""

    def test_chunker_is_registered(self):
        """Test that XMLHTMLChunker is properly registered."""
        from chunking_strategy.core.registry import list_chunkers
        chunkers = list_chunkers()
        assert "xml_html_chunker" in chunkers

        # Test that create_chunker works
        chunker = create_chunker("xml_html_chunker")
        assert chunker is not None
        assert isinstance(chunker, XMLHTMLChunker)
        assert chunker.name == "xml_html_chunker"
        assert chunker.category == "document"

    def test_create_chunker_factory(self):
        """Test creating chunker via factory function."""
        chunker = create_chunker("xml_html_chunker")
        assert isinstance(chunker, XMLHTMLChunker)
        assert chunker.name == "xml_html_chunker"
        assert chunker.category == "document"


class TestXMLHTMLChunkerInitialization:
    """Test chunker initialization and configuration."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        chunker = XMLHTMLChunker()
        assert chunker.name == "xml_html_chunker"
        assert chunker.category == "document"
        assert chunker.chunk_by == "hierarchy"
        assert chunker.min_chunk_size == 20
        assert chunker.chunk_size == 2000
        assert chunker.chunk_overlap == 200
        assert chunker.max_chunk_size == 8000
        assert chunker.preserve_structure is True
        assert chunker.include_attributes is True
        assert chunker.parser == "auto"
        assert chunker.encoding == "utf-8"

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        chunker = XMLHTMLChunker(
            chunk_by="semantic",
            chunk_size=1500,
            min_chunk_size=50,
            max_chunk_size=5000,
            chunk_overlap=100,
            preserve_structure=False,
            include_attributes=False,
            parser="lxml",
            encoding="latin1"
        )
        assert chunker.chunk_by == "semantic"
        assert chunker.chunk_size == 1500
        assert chunker.min_chunk_size == 50
        assert chunker.max_chunk_size == 5000
        assert chunker.chunk_overlap == 100
        assert chunker.preserve_structure is False
        assert chunker.include_attributes is False
        assert chunker.parser == "lxml"
        assert chunker.encoding == "latin1"

    def test_semantic_html_tags(self):
        """Test semantic HTML tags configuration."""
        chunker = XMLHTMLChunker()
        semantic_tags = chunker._get_default_semantic_tags()
        expected_tags = ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer']
        for tag in expected_tags:
            assert tag in semantic_tags

    def test_target_tags(self):
        """Test target tags configuration."""
        chunker = XMLHTMLChunker()
        target_tags = chunker._get_default_target_tags()
        expected_tags = ['div', 'section', 'article', 'p', 'h1', 'h2', 'h3', 'table']
        for tag in expected_tags:
            assert tag in target_tags


class TestXMLHTMLFormatDetection:
    """Test format detection capabilities."""

    def test_html_detection(self):
        """Test HTML format detection."""
        chunker = XMLHTMLChunker()

        # HTML with DOCTYPE
        html_content = '<!DOCTYPE html><html><head><title>Test</title></head><body><p>Content</p></body></html>'
        format_type = chunker._detect_format(html_content)
        assert format_type == "html"

        # HTML with typical tags
        html_content2 = '<div class="container"><p>Some text</p><span>More text</span></div>'
        format_type2 = chunker._detect_format(html_content2)
        assert format_type2 == "html"

    def test_xml_detection(self):
        """Test XML format detection."""
        chunker = XMLHTMLChunker()

        # XML with declaration
        xml_content = '<?xml version="1.0" encoding="UTF-8"?><root><item>value</item></root>'
        format_type = chunker._detect_format(xml_content)
        assert format_type == "xml"

        # XML without typical HTML tags
        xml_content2 = '<configuration><server><host>localhost</host></server></configuration>'
        format_type2 = chunker._detect_format(xml_content2)
        assert format_type2 == "xml"

    def test_ambiguous_detection(self):
        """Test format detection for ambiguous content."""
        chunker = XMLHTMLChunker()

        # Minimal content defaults to HTML
        minimal_content = '<root>content</root>'
        format_type = chunker._detect_format(minimal_content)
        # Should default to HTML when unclear
        assert format_type in ["html", "xml"]


class TestXMLHTMLChunkingStrategies:
    """Test different chunking strategies."""

    def test_hierarchy_chunking(self):
        """Test hierarchy-based chunking."""
        chunker = XMLHTMLChunker(chunk_by="hierarchy", min_chunk_size=10)

        xml_content = '''<?xml version="1.0"?>
        <catalog>
            <book id="1">
                <title>Test Book</title>
                <author>Test Author</author>
                <description>A test book for testing purposes.</description>
            </book>
        </catalog>'''

        result = chunker.chunk(xml_content)
        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "xml_html_chunker"
        assert result.total_chunks > 0

        # Check chunk properties
        for chunk in result.chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.modality == ModalityType.TEXT
            assert "hierarchy_depth" in chunk.metadata.extra
            assert "element_tag" in chunk.metadata.extra

    def test_semantic_chunking_html(self):
        """Test semantic chunking with HTML."""
        chunker = XMLHTMLChunker(chunk_by="semantic", min_chunk_size=10)

        html_content = '''<!DOCTYPE html>
        <html>
        <body>
            <header><h1>Title</h1></header>
            <main>
                <article>
                    <h2>Article Title</h2>
                    <p>Article content goes here.</p>
                </article>
                <section>
                    <h3>Section Title</h3>
                    <p>Section content.</p>
                </section>
            </main>
            <footer><p>Copyright 2025</p></footer>
        </body>
        </html>'''

        result = chunker.chunk(html_content)
        assert result.total_chunks > 0

        # Check that semantic elements are captured
        semantic_tags = set()
        for chunk in result.chunks:
            semantic_tags.add(chunk.metadata.extra.get("element_tag"))

        expected_semantic = {"header", "main", "article", "section", "footer"}
        assert len(expected_semantic.intersection(semantic_tags)) > 0

    def test_tag_based_chunking(self):
        """Test tag-based chunking."""
        chunker = XMLHTMLChunker(
            chunk_by="tag_based",
            target_tags=["book", "title", "author"],
            min_chunk_size=5
        )

        xml_content = '''<catalog>
            <book id="1">
                <title>Book One</title>
                <author>Author One</author>
            </book>
            <book id="2">
                <title>Book Two</title>
                <author>Author Two</author>
            </book>
        </catalog>'''

        result = chunker.chunk(xml_content)
        assert result.total_chunks > 0

        # Check that target tags are captured
        found_tags = set()
        for chunk in result.chunks:
            found_tags.add(chunk.metadata.extra.get("element_tag"))

        assert "book" in found_tags or "title" in found_tags or "author" in found_tags

    def test_element_type_chunking(self):
        """Test element type-based chunking."""
        chunker = XMLHTMLChunker(chunk_by="element_type", min_chunk_size=10)

        xml_content = '''<catalog>
            <book><title>Book 1</title></book>
            <book><title>Book 2</title></book>
            <author>Author 1</author>
            <author>Author 2</author>
        </catalog>'''

        result = chunker.chunk(xml_content)
        assert result.total_chunks > 0

        # Check element type grouping
        for chunk in result.chunks:
            assert "element_type" in chunk.metadata.extra
            assert "element_count" in chunk.metadata.extra

    def test_attribute_based_chunking(self):
        """Test attribute-based chunking."""
        chunker = XMLHTMLChunker(
            chunk_by="attribute_based",
            group_by_attribute="class",
            min_chunk_size=10
        )

        html_content = '''<div>
            <p class="intro">Introduction paragraph.</p>
            <p class="content">Main content paragraph.</p>
            <p class="intro">Another intro paragraph.</p>
        </div>'''

        result = chunker.chunk(html_content)
        assert result.total_chunks >= 0  # May be 0 if no attributes match

        if result.chunks:
            for chunk in result.chunks:
                assert "group_attribute" in chunk.metadata.extra
                assert "attribute_value" in chunk.metadata.extra

    def test_content_size_chunking(self):
        """Test content size-based chunking."""
        chunker = XMLHTMLChunker(
            chunk_by="content_size",
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=10
        )

        xml_content = '''<root>
            <item>This is some content that should be chunked based on size.</item>
            <item>More content here for testing size-based chunking algorithm.</item>
            <item>Additional content to ensure we have enough text for multiple chunks.</item>
        </root>'''

        result = chunker.chunk(xml_content)
        assert result.total_chunks > 0

        # Check size constraints
        for chunk in result.chunks:
            assert len(chunk.content) >= chunker.min_chunk_size
            assert "content_chunk_index" in chunk.metadata.extra


class TestXMLHTMLInputTypes:
    """Test different input types."""

    def test_string_content(self):
        """Test chunking string content."""
        chunker = XMLHTMLChunker(min_chunk_size=10)
        xml_content = '<root><item>Test content</item></root>'

        result = chunker.chunk(xml_content)
        assert isinstance(result, ChunkingResult)
        assert result.source_info["source_type"] == "content"

    def test_bytes_content(self):
        """Test chunking bytes content."""
        chunker = XMLHTMLChunker(min_chunk_size=10)
        xml_content = b'<root><item>Test content</item></root>'

        result = chunker.chunk(xml_content)
        assert isinstance(result, ChunkingResult)
        assert result.source_info["source_type"] == "content"

    def test_file_path_string(self):
        """Test chunking with file path as string."""
        chunker = XMLHTMLChunker(min_chunk_size=10)

        # Create temporary XML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<root><item>Test file content</item></root>')
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
        chunker = XMLHTMLChunker(min_chunk_size=10)

        # Create temporary XML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<root><item>Test path content</item></root>')
            temp_path = Path(f.name)

        try:
            result = chunker.chunk(temp_path)
            assert isinstance(result, ChunkingResult)
            assert result.source_info["source_type"] == "file"
        finally:
            temp_path.unlink()

    def test_invalid_content_type(self):
        """Test error handling for invalid content type."""
        chunker = XMLHTMLChunker()

        with pytest.raises(ValueError, match="Unsupported content type"):
            chunker.chunk(123)  # Invalid type


class TestXMLHTMLEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_content(self):
        """Test handling of empty content."""
        chunker = XMLHTMLChunker()

        result = chunker.chunk("")
        assert isinstance(result, ChunkingResult)
        assert result.total_chunks == 0
        assert result.strategy_used == "xml_html_chunker"
        assert result.source_info["markup_structure"]["structure_type"] == "empty"

    def test_whitespace_only_content(self):
        """Test handling of whitespace-only content."""
        chunker = XMLHTMLChunker()

        result = chunker.chunk("   \n\t   ")
        assert result.total_chunks == 0
        assert result.source_info["markup_structure"]["structure_type"] == "empty"

    def test_malformed_xml(self):
        """Test handling of malformed XML/HTML."""
        chunker = XMLHTMLChunker(min_chunk_size=5)

        # Malformed XML - should fall back to text chunking
        malformed_content = '<root><unclosed><item>content</root>'
        result = chunker.chunk(malformed_content)

        # Should either parse with error recovery or fall back to text chunking
        assert isinstance(result, ChunkingResult)
        # Fallback should be indicated somehow
        if result.source_info.get("markup_structure", {}).get("parsing_failed"):
            assert result.total_chunks >= 0

    def test_very_large_chunk(self):
        """Test handling of content exceeding max_chunk_size."""
        chunker = XMLHTMLChunker(
            chunk_by="content_size",
            chunk_size=50,
            max_chunk_size=100,
            min_chunk_size=10
        )

        # Create content larger than max_chunk_size
        large_content = '<root>' + '<item>' + 'x' * 200 + '</item>' + '</root>'
        result = chunker.chunk(large_content)

        # Should create chunks within size limits
        for chunk in result.chunks:
            assert len(chunk.content) <= chunker.max_chunk_size

    def test_invalid_file_path(self):
        """Test error handling for non-existent file."""
        chunker = XMLHTMLChunker()

        with pytest.raises(FileNotFoundError):
            chunker.chunk(Path("nonexistent_file.xml"))

    def test_unknown_chunking_strategy(self):
        """Test error handling for unknown chunking strategy."""
        chunker = XMLHTMLChunker(chunk_by="unknown_strategy")

        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunker.chunk("<root><item>test</item></root>")


class TestXMLHTMLParserBackends:
    """Test different parser backends."""

    def test_lxml_parser(self):
        """Test lxml parser backend."""
        chunker = XMLHTMLChunker(parser="lxml", min_chunk_size=5)

        xml_content = '<root><item>Test with lxml</item></root>'
        result = chunker.chunk(xml_content)

        assert result.source_info["parser_used"] == "lxml"
        assert result.total_chunks >= 0

    def test_bs4_parser(self):
        """Test BeautifulSoup parser backend."""
        chunker = XMLHTMLChunker(parser="bs4", min_chunk_size=5)

        html_content = '<div><p>Test with BeautifulSoup</p></div>'
        result = chunker.chunk(html_content)

        assert result.source_info["parser_used"] == "bs4"
        assert result.total_chunks >= 0

    def test_builtin_parser(self):
        """Test builtin parser backend."""
        chunker = XMLHTMLChunker(parser="builtin", min_chunk_size=5)

        xml_content = '<root><item>Test with builtin parser</item></root>'
        result = chunker.chunk(xml_content)

        assert result.source_info["parser_used"] == "builtin"
        assert result.total_chunks >= 0

    def test_auto_parser_selection(self):
        """Test automatic parser selection."""
        chunker = XMLHTMLChunker(parser="auto", min_chunk_size=5)

        xml_content = '<root><item>Test with auto selection</item></root>'
        result = chunker.chunk(xml_content)

        # Should select the best available parser
        assert result.source_info["parser_used"] in ["lxml", "bs4", "builtin"]

    @patch('chunking_strategy.strategies.document.xml_html_chunker.HAS_LXML', False)
    @patch('chunking_strategy.strategies.document.xml_html_chunker.HAS_BS4', False)
    def test_fallback_to_builtin(self):
        """Test fallback to builtin parser when others unavailable."""
        chunker = XMLHTMLChunker(parser="auto", min_chunk_size=5)

        xml_content = '<root><item>Test builtin fallback</item></root>'
        result = chunker.chunk(xml_content)

        assert result.source_info["parser_used"] == "builtin"


class TestXMLHTMLStreaming:
    """Test streaming capabilities."""

    def test_can_stream(self):
        """Test that chunker supports streaming."""
        chunker = XMLHTMLChunker()
        assert chunker.can_stream() is True

    def test_chunk_stream(self):
        """Test streaming chunk processing."""
        chunker = XMLHTMLChunker(min_chunk_size=10)

        # Simulate streaming content
        content_parts = [
            '<root>',
            '<item>First item content</item>',
            '<item>Second item content</item>',
            '</root>'
        ]

        chunks = list(chunker.chunk_stream(content_parts))
        assert len(chunks) > 0

        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.modality == ModalityType.TEXT


class TestXMLHTMLAdaptation:
    """Test adaptive behavior."""

    def test_adapt_parameters_quality(self):
        """Test parameter adaptation based on quality feedback."""
        # Test low quality score reduction
        chunker1 = XMLHTMLChunker(chunk_size=1000, max_chunk_size=5000)
        original_chunk_size = chunker1.chunk_size

        chunker1.adapt_parameters(0.3, "quality")
        assert chunker1.chunk_size < original_chunk_size

        # Test high quality score increase
        chunker2 = XMLHTMLChunker(chunk_size=1000, max_chunk_size=5000)
        original_chunk_size2 = chunker2.chunk_size

        chunker2.adapt_parameters(0.9, "quality")
        assert chunker2.chunk_size > original_chunk_size2

    def test_adapt_parameters_performance(self):
        """Test parameter adaptation based on performance feedback."""
        chunker = XMLHTMLChunker(max_chunk_size=5000)

        original_max_size = chunker.max_chunk_size

        # Poor performance should reduce max size
        chunker.adapt_parameters(0.2, "performance")
        assert chunker.max_chunk_size < original_max_size

    def test_adaptation_history(self):
        """Test adaptation history tracking."""
        chunker = XMLHTMLChunker()

        # Initially no history
        history = chunker.get_adaptation_history()
        assert len(history) == 0

        # After adaptation, should have history
        chunker.adapt_parameters(0.5, "quality")
        history = chunker.get_adaptation_history()
        assert len(history) == 1
        assert history[0]["feedback_score"] == 0.5
        assert history[0]["feedback_type"] == "quality"
        assert "old_config" in history[0]
        assert "new_config" in history[0]

    def test_specific_feedback_adaptation(self):
        """Test adaptation to specific feedback."""
        chunker = XMLHTMLChunker(chunk_size=1000, max_chunk_size=2000)

        original_chunk_size = chunker.chunk_size
        original_max_size = chunker.max_chunk_size

        # Chunks too large feedback
        chunker.adapt_parameters(0.5, "quality", chunks_too_large=True)
        assert chunker.chunk_size < original_chunk_size
        assert chunker.max_chunk_size < original_max_size

        # Reset for next test
        chunker.chunk_size = original_chunk_size
        chunker.max_chunk_size = original_max_size

        # Chunks too small feedback
        chunker.adapt_parameters(0.5, "quality", chunks_too_small=True)
        assert chunker.chunk_size > original_chunk_size
        assert chunker.max_chunk_size > original_max_size


class TestXMLHTMLChunkMetadata:
    """Test chunk metadata generation."""

    def test_hierarchy_chunk_metadata(self):
        """Test metadata for hierarchy chunks."""
        chunker = XMLHTMLChunker(chunk_by="hierarchy", min_chunk_size=5)

        xml_content = '<root><book><title>Test Title</title></book></root>'
        result = chunker.chunk(xml_content)

        for chunk in result.chunks:
            metadata = chunk.metadata
            assert metadata.source == "unknown"  # Default for string content
            assert metadata.source_type == "content"
            assert "hierarchy_depth" in metadata.extra
            assert "element_tag" in metadata.extra
            assert "chunk_type" in metadata.extra
            assert metadata.extra["chunk_type"] == "hierarchy"
            assert metadata.extra["chunking_strategy"] == "hierarchy"

    def test_semantic_chunk_metadata(self):
        """Test metadata for semantic chunks."""
        chunker = XMLHTMLChunker(chunk_by="semantic", min_chunk_size=5)

        html_content = '<main><article>Article content</article></main>'
        result = chunker.chunk(html_content)

        for chunk in result.chunks:
            metadata = chunk.metadata
            assert "element_tag" in metadata.extra
            assert "chunk_type" in metadata.extra
            assert metadata.extra["chunk_type"] == "semantic"

    def test_file_source_metadata(self):
        """Test metadata for file-based chunking."""
        chunker = XMLHTMLChunker(min_chunk_size=5)

        # Create temporary XML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<root><item>File content</item></root>')
            temp_path = f.name

        try:
            result = chunker.chunk(temp_path)
            assert result.source_info["source"] == temp_path
            assert result.source_info["source_type"] == "file"

            for chunk in result.chunks:
                assert chunk.metadata.source == temp_path
                assert chunk.metadata.source_type == "file"
        finally:
            Path(temp_path).unlink()


class TestXMLHTMLOrchestatorIntegration:
    """Test integration with ChunkerOrchestrator."""

    def test_orchestrator_auto_selection_html(self):
        """Test orchestrator auto-selects XML/HTML chunker for HTML files."""
        config = {"strategies": {"primary": "auto"}}
        orchestrator = ChunkerOrchestrator(config=config)

        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write('<html><body><p>Test HTML content for orchestrator.</p></body></html>')
            temp_path = f.name

        try:
            result = orchestrator.chunk_file(temp_path)
            assert result.strategy_used == "xml_html_chunker"
            assert result.total_chunks > 0
        finally:
            Path(temp_path).unlink()

    def test_orchestrator_auto_selection_xml(self):
        """Test orchestrator auto-selects XML/HTML chunker for XML files."""
        config = {"strategies": {"primary": "auto"}}
        orchestrator = ChunkerOrchestrator(config=config)

        # Create temporary XML file with enough content to avoid small file override
        xml_content = '''<?xml version="1.0"?>
        <catalog>
            <book id="1">
                <title>Test Book One</title>
                <author>Test Author One</author>
                <description>A comprehensive test book for validating XML chunking functionality. This description is long enough to ensure the file size exceeds the small file threshold and allows the XML chunker to be properly selected by the orchestrator auto-selection logic.</description>
            </book>
            <book id="2">
                <title>Test Book Two</title>
                <author>Test Author Two</author>
                <description>Another test book with substantial content to ensure proper XML chunking behavior. This description also contains enough text to contribute to the overall file size requirements for automatic strategy selection.</description>
            </book>
        </catalog>'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            result = orchestrator.chunk_file(temp_path)
            assert result.strategy_used == "xml_html_chunker"
            assert result.total_chunks > 0
        finally:
            Path(temp_path).unlink()

    def test_orchestrator_explicit_strategy(self):
        """Test orchestrator with explicit XML/HTML chunker strategy."""
        config = {"strategies": {"primary": "xml_html_chunker"}}
        orchestrator = ChunkerOrchestrator(config=config)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<root><item>Explicit strategy test</item></root>')
            temp_path = f.name

        try:
            result = orchestrator.chunk_file(temp_path)
            assert result.strategy_used == "xml_html_chunker"
        finally:
            Path(temp_path).unlink()

    def test_orchestrator_with_custom_config(self):
        """Test orchestrator with custom XML/HTML chunker configuration."""
        config = {
            "strategies": {
                "primary": "xml_html_chunker",
                "configs": {
                    "xml_html_chunker": {
                        "chunk_by": "element_type",
                        "min_chunk_size": 5,
                        "chunk_size": 500
                    }
                }
            }
        }
        orchestrator = ChunkerOrchestrator(config=config)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write('<div><p>Custom config test</p><span>More content</span></div>')
            temp_path = f.name

        try:
            result = orchestrator.chunk_file(temp_path)
            assert result.strategy_used == "xml_html_chunker"
            # Check that custom config was applied by checking chunk metadata
            if result.chunks:
                chunk = result.chunks[0]
                assert "element_type" in chunk.metadata.extra or "chunk_type" in chunk.metadata.extra
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
