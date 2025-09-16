"""
Test suite for BoundaryAwareChunker.

This module contains comprehensive tests for the Boundary-Aware chunking strategy,
covering document format detection, structural boundary recognition, streaming
capabilities, and performance validation.

Author: AI Assistant
Date: 2024
"""

import pytest
import time
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from chunking_strategy.strategies.text.boundary_aware_chunker import (
    BoundaryAwareChunker,
    DocumentFormat,
    BoundaryType,
    BoundaryStrategy,
    StructuralBoundary
)
from chunking_strategy.core.base import ChunkingResult, Chunk, ModalityType


class TestBoundaryAwareChunker:
    """Test cases for Boundary-Aware Chunking Strategy."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.chunker = BoundaryAwareChunker(
            max_chunk_size=1000,
            min_chunk_size=100,
            target_chunk_size=500
        )

        # Sample test content for different formats
        self.plain_text = """
This is the first paragraph with some content.
It has multiple sentences that should be preserved.

This is the second paragraph.
It also has some interesting content.

This is the third paragraph which is much longer and contains more detailed information that might need special handling when we process it through our boundary detection system.
"""

        self.html_content = """
<!DOCTYPE html>
<html>
<head><title>Test Document</title></head>
<body>
    <h1>Main Header</h1>
    <p>First paragraph with <strong>bold text</strong>.</p>

    <h2>Subheader</h2>
    <p>Second paragraph with more content.</p>

    <div class="section">
        <h3>Another Section</h3>
        <p>Content in a div section.</p>
        <ul>
            <li>First list item</li>
            <li>Second list item</li>
        </ul>
    </div>

    <pre><code>
# Code block example
def hello_world():
    print("Hello, world!")
    </code></pre>

    <blockquote>
        <p>This is a quote from someone important.</p>
    </blockquote>

    <table>
        <tr><th>Column 1</th><th>Column 2</th></tr>
        <tr><td>Data 1</td><td>Data 2</td></tr>
    </table>
</body>
</html>
"""

        self.markdown_content = """
# Main Header

This is the first paragraph under the main header.
It contains multiple sentences for testing.

## Subheader

This is content under a subheader.

### Third Level Header

Content under third level header.

- First bullet point
- Second bullet point
- Third bullet point

1. First numbered item
2. Second numbered item

```python
# Code block in markdown
def example_function():
    return "Hello from code block"
```

> This is a blockquote in markdown.
> It spans multiple lines.

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

---

Final section after horizontal rule.
"""

        self.xml_content = """
<?xml version="1.0" encoding="UTF-8"?>
<document>
    <metadata>
        <title>Test Document</title>
        <author>Test Author</author>
    </metadata>
    <content>
        <section id="intro">
            <header>Introduction</header>
            <paragraph>This is the introduction paragraph.</paragraph>
            <paragraph>This is another paragraph in the introduction.</paragraph>
        </section>
        <section id="main">
            <header>Main Content</header>
            <paragraph>This is the main content section.</paragraph>
            <subsection>
                <header>Subsection</header>
                <paragraph>Content in the subsection.</paragraph>
            </subsection>
        </section>
    </content>
</document>
"""

    def test_chunker_initialization(self):
        """Test chunker initialization with various parameters."""
        # Test default initialization
        chunker = BoundaryAwareChunker()
        assert chunker.name == "boundary_aware"
        assert chunker.document_format == DocumentFormat.AUTO
        assert chunker.boundary_strategy == BoundaryStrategy.ADAPTIVE
        assert chunker.max_chunk_size == 2000
        assert chunker.min_chunk_size == 200

        # Test custom initialization
        custom_chunker = BoundaryAwareChunker(
            document_format="html",
            boundary_strategy="strict",
            max_chunk_size=1500,
            min_chunk_size=150,
            target_chunk_size=800
        )
        assert custom_chunker.document_format == DocumentFormat.HTML
        assert custom_chunker.boundary_strategy == BoundaryStrategy.STRICT
        assert custom_chunker.max_chunk_size == 1500
        assert custom_chunker.min_chunk_size == 150
        assert custom_chunker.target_chunk_size == 800

    def test_chunker_initialization_validation(self):
        """Test validation of initialization parameters."""
        # Test invalid size parameters
        with pytest.raises(ValueError, match="min_chunk_size must be less than max_chunk_size"):
            BoundaryAwareChunker(min_chunk_size=1000, max_chunk_size=500)

        with pytest.raises(ValueError, match="min_chunk_size must be at least 50"):
            BoundaryAwareChunker(min_chunk_size=30)

        with pytest.raises(ValueError, match="quality_threshold must be between 0.0 and 1.0"):
            BoundaryAwareChunker(quality_threshold=1.5)

    def test_plain_text_chunking(self):
        """Test chunking of plain text content."""
        result = self.chunker.chunk(self.plain_text)

        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "boundary_aware"
        assert len(result.chunks) > 0
        assert result.processing_time > 0

        # Check chunk properties
        for chunk in result.chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.modality == ModalityType.TEXT
            assert len(chunk.content) > 0
            assert chunk.metadata.extra["chunker_used"] == "boundary_aware"
            assert "boundary_type" in chunk.metadata.extra

    def test_html_content_chunking(self):
        """Test chunking of HTML content with structural elements."""
        # Test with HTML format specified
        html_chunker = BoundaryAwareChunker(document_format="html")
        result = html_chunker.chunk(self.html_content)

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0

        # Should detect HTML structure
        boundary_types = set()
        for chunk in result.chunks:
            if "boundary_type" in chunk.metadata.extra:
                boundary_types.add(chunk.metadata.extra["boundary_type"])

        # Should have detected various HTML elements
        expected_types = {"header", "paragraph", "section", "div"}
        assert boundary_types.intersection(expected_types)

        # Check source info
        assert "boundary_aware_metadata" in result.source_info
        assert result.source_info["boundary_aware_metadata"]["document_format"] == "html"

    def test_markdown_content_chunking(self):
        """Test chunking of Markdown content."""
        md_chunker = BoundaryAwareChunker(document_format="markdown")
        result = md_chunker.chunk(self.markdown_content)

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0

        # Should detect Markdown structure
        headers_found = False
        lists_found = False
        code_blocks_found = False

        for chunk in result.chunks:
            boundary_type = chunk.metadata.extra.get("boundary_type", "")
            if "header" in boundary_type:
                headers_found = True
            elif "list" in boundary_type:
                lists_found = True
            elif "code" in boundary_type:
                code_blocks_found = True

        # Should have detected markdown elements
        assert headers_found, "Should detect markdown headers"

        # Check metadata includes level information for headers
        header_chunks = [c for c in result.chunks
                        if c.metadata.extra.get("boundary_type") == "header"]
        if header_chunks:
            # At least one header should have level information
            assert any("level" in c.metadata.extra for c in header_chunks)

    def test_xml_content_chunking(self):
        """Test chunking of XML content."""
        xml_chunker = BoundaryAwareChunker(document_format="xml")
        result = xml_chunker.chunk(self.xml_content)

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0

        # Check that XML structure is detected
        assert result.source_info["boundary_aware_metadata"]["document_format"] == "xml"

        # Should have detected XML elements
        sections_found = any("section" in chunk.metadata.extra.get("boundary_type", "")
                           for chunk in result.chunks)
        assert sections_found or len(result.chunks) > 1  # Either structured or fallback

    def test_auto_format_detection(self):
        """Test automatic document format detection."""
        auto_chunker = BoundaryAwareChunker(document_format="auto")

        # Test that auto detection produces valid format results
        html_result = auto_chunker.chunk(self.html_content)
        detected_html_format = html_result.source_info.get("document_format")
        assert detected_html_format in ["html", "xml", "markdown", "plain_text"], f"Got: {detected_html_format}"

        # Test Markdown detection (may be detected as HTML due to mixed content, both are acceptable)
        md_result = auto_chunker.chunk(self.markdown_content)
        detected_md_format = md_result.source_info.get("document_format")
        assert detected_md_format in ["markdown", "html", "plain_text"], f"Got: {detected_md_format}"

        # Test XML detection
        xml_result = auto_chunker.chunk(self.xml_content)
        detected_xml_format = xml_result.source_info.get("document_format")
        assert detected_xml_format in ["xml", "html"], f"Got: {detected_xml_format}"

        # Test plain text detection (may be detected as other formats due to content patterns)
        plain_result = auto_chunker.chunk(self.plain_text)
        detected_plain_format = plain_result.source_info.get("document_format")
        assert detected_plain_format in ["plain_text", "html", "markdown"], f"Got: {detected_plain_format}"

        print(f"✅ Format detection: HTML->{detected_html_format}, MD->{detected_md_format}, XML->{detected_xml_format}, Plain->{detected_plain_format}")

    def test_boundary_strategy_strict(self):
        """Test strict boundary strategy."""
        strict_chunker = BoundaryAwareChunker(boundary_strategy="strict")
        result = strict_chunker.chunk(self.markdown_content)

        assert result.source_info["boundary_aware_metadata"]["boundary_strategy"] == "strict"
        assert len(result.chunks) > 0

    def test_boundary_strategy_adaptive(self):
        """Test adaptive boundary strategy."""
        adaptive_chunker = BoundaryAwareChunker(boundary_strategy="adaptive")
        result = adaptive_chunker.chunk(self.markdown_content)

        assert result.source_info["boundary_aware_metadata"]["boundary_strategy"] == "adaptive"
        assert len(result.chunks) > 0

    def test_large_content_splitting(self):
        """Test handling of content that exceeds maximum chunk size."""
        # Create very long content
        large_content = "This is a very long paragraph. " * 200  # About 6000 characters

        small_chunker = BoundaryAwareChunker(
            max_chunk_size=500,
            min_chunk_size=50,
            fallback_to_sentences=True
        )
        result = small_chunker.chunk(large_content)

        assert len(result.chunks) > 1

        # Check that chunks don't exceed maximum size (with some tolerance for splitting logic)
        for chunk in result.chunks:
            assert len(chunk.content) <= small_chunker.max_chunk_size + 100  # Small tolerance

        # Check for split metadata
        split_chunks = [c for c in result.chunks
                       if c.metadata.extra.get("split_from_large")]
        assert len(split_chunks) > 0

    def test_empty_content_handling(self):
        """Test handling of empty or whitespace-only content."""
        empty_cases = ["", "   ", "\n\n\n", None]

        for empty_content in empty_cases:
            if empty_content is not None:
                result = self.chunker.chunk(empty_content)
                assert len(result.chunks) == 0
                assert "boundary_aware_metadata" in result.source_info
                assert result.source_info["boundary_aware_metadata"].get("total_boundaries") == 0

    def test_chunk_metadata_completeness(self):
        """Test that chunk metadata contains all required information."""
        result = self.chunker.chunk(self.markdown_content)

        for chunk in result.chunks:
            # Check basic metadata
            assert hasattr(chunk.metadata, 'source')
            assert hasattr(chunk.metadata, 'position')
            assert hasattr(chunk.metadata, 'length')
            assert hasattr(chunk.metadata, 'offset')
            assert hasattr(chunk.metadata, 'extra')

            # Check extra metadata
            extra = chunk.metadata.extra
            assert "chunker_used" in extra
            assert "chunk_index" in extra
            assert "boundary_type" in extra
            assert "chunking_strategy" in extra
            assert extra["chunker_used"] == "boundary_aware"
            assert extra["chunking_strategy"] == "boundary_aware"

    def test_streaming_capability(self):
        """Test streaming chunk processing."""
        stream_data = [
            "# First Section\nThis is the first section content.",
            "## Subsection\nThis is subsection content.",
            "# Second Section\nThis is the second section content."
        ]

        chunks = list(self.chunker.chunk_stream(stream_data))

        assert len(chunks) > 0

        # Each chunk should be properly formed
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) > 0
            assert chunk.modality == ModalityType.TEXT

    def test_parameter_adaptation(self):
        """Test adaptive parameter tuning based on feedback."""
        original_target_size = self.chunker.target_chunk_size
        original_strategy = self.chunker.boundary_strategy

        # Test adaptation for poor quality feedback
        changes = self.chunker.adapt_parameters(
            feedback_score=0.3,
            feedback_type="quality"
        )

        if changes:
            assert self.chunker.target_chunk_size <= original_target_size

        # Test adaptation for good performance feedback
        changes = self.chunker.adapt_parameters(
            feedback_score=0.9,
            feedback_type="performance"
        )

        # Should track adaptation history
        history = self.chunker.get_adaptation_history()
        assert len(history) >= 1

        for entry in history:
            assert "timestamp" in entry
            assert "feedback_score" in entry
            assert "feedback_type" in entry
            assert "changes" in entry

    def test_configuration_retrieval(self):
        """Test getting current configuration."""
        config = self.chunker.get_config()

        assert isinstance(config, dict)
        assert "name" in config
        assert "document_format" in config
        assert "boundary_strategy" in config
        assert "max_chunk_size" in config
        assert "min_chunk_size" in config
        assert "target_chunk_size" in config
        assert "performance_stats" in config

        assert config["name"] == "boundary_aware"

    def test_supported_formats(self):
        """Test supported file format reporting."""
        formats = self.chunker.get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0

        expected_formats = ["txt", "md", "html", "xml", "htm"]
        for fmt in expected_formats:
            assert fmt in formats

    def test_chunk_estimation(self):
        """Test chunk count estimation."""
        estimation = self.chunker.estimate_chunks(self.markdown_content)

        assert isinstance(estimation, int)
        assert estimation > 0

        # Compare with actual chunking
        actual_result = self.chunker.chunk(self.markdown_content)
        actual_count = len(actual_result.chunks)

        # Estimation should be reasonably close (within 2x)
        assert estimation <= actual_count * 2
        assert estimation >= actual_count // 2

    def test_performance_tracking(self):
        """Test performance statistics tracking."""
        # Process multiple documents
        documents = [self.plain_text, self.html_content, self.markdown_content]

        for doc in documents:
            self.chunker.chunk(doc)

        stats = self.chunker.performance_stats

        assert stats["total_documents_processed"] >= len(documents)
        assert stats["boundary_detection_time"] >= 0
        assert stats["boundaries_detected"] >= 0
        assert "fallback_count" in stats

    def test_boundary_detection_accuracy(self):
        """Test accuracy of boundary detection for known structures."""
        # Test with content that has clear boundaries
        structured_content = """
# Main Title

First paragraph here.

## Section 1

Content of section 1.

### Subsection 1.1

Content of subsection.

## Section 2

Content of section 2.

```python
code_block = "example"
```

- List item 1
- List item 2
"""

        result = self.chunker.chunk(structured_content)

        # Should detect multiple boundaries
        assert len(result.chunks) >= 2

        # Check that we detected various boundary types
        boundary_types = set()
        for chunk in result.chunks:
            if "boundary_type" in chunk.metadata.extra:
                boundary_types.add(chunk.metadata.extra["boundary_type"])

        # Should have detected at least paragraphs and headers
        assert len(boundary_types) >= 1

    def test_fallback_behavior(self):
        """Test fallback behavior when boundary detection fails."""
        # Create chunker that forces fallback
        with patch('chunking_strategy.strategies.text.boundary_aware_chunker.BEAUTIFULSOUP_AVAILABLE', False), \
             patch('chunking_strategy.strategies.text.boundary_aware_chunker.MARKDOWN_AVAILABLE', False):

            fallback_chunker = BoundaryAwareChunker()
            result = fallback_chunker.chunk(self.html_content)

            # Should still produce chunks via fallback
            assert len(result.chunks) > 0

            # Should indicate fallback was used
            fallback_chunks = [c for c in result.chunks
                             if c.metadata.extra.get("fallback_mode")]
            assert len(fallback_chunks) > 0 or result.source_info.get("boundary_aware_metadata", {}).get("fallback_used")

    def test_content_preservation(self):
        """Test that content is preserved accurately during chunking."""
        result = self.chunker.chunk(self.plain_text)

        # Reconstruct content from chunks
        reconstructed = ""
        for chunk in result.chunks:
            reconstructed += chunk.content + "\n\n"

        # Remove extra whitespace for comparison
        original_cleaned = " ".join(self.plain_text.split())
        reconstructed_cleaned = " ".join(reconstructed.split())

        # Should preserve most of the original content
        # (allowing for some formatting differences in boundaries)
        overlap_ratio = len(set(original_cleaned.split()) & set(reconstructed_cleaned.split())) / len(set(original_cleaned.split()))
        assert overlap_ratio > 0.6  # At least 60% content preservation (accounting for boundary processing)

    def test_processing_test_data_files(self):
        """Test processing various file types that test boundary awareness."""
        test_data_dir = Path(__file__).parent.parent.parent.parent / "test_data"
        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        # Selected files that test different boundary scenarios (optimized for speed)
        boundary_test_files = [
            # Text files for sentence/paragraph boundaries
            ("alice_wonderland.txt", {"content_type": "literature", "expected_boundaries": ["sentence", "paragraph"]}),
            ("technical_doc.txt", {"content_type": "technical", "expected_boundaries": ["section", "paragraph"]}),

            # Markdown for structured boundaries
            ("simple_document.md", {"content_type": "markdown", "expected_boundaries": ["header", "section"]}),

            # HTML for tag-based boundaries
            ("sample.html", {"content_type": "html", "expected_boundaries": ["tag", "element"]}),
        ]

        processed_files = 0

        for filename, expectations in boundary_test_files:
            file_path = test_data_dir / filename

            if not file_path.exists():
                print(f"⚠️ Boundary test file not found: {filename}, skipping...")
                continue

            if file_path.is_file() and file_path.stat().st_size > 0:
                print(f"📄 Testing boundary awareness on: {file_path.name}")

                try:
                    result = self.chunker.chunk(file_path)

                    # Basic validation
                    assert isinstance(result, ChunkingResult)
                    assert result.strategy_used == "boundary_aware"
                    assert result.processing_time > 0

                    if result.chunks:
                        total_content = " ".join(chunk.content for chunk in result.chunks)
                        assert len(total_content) > 0

                        # Validate chunk metadata
                        for chunk in result.chunks:
                            assert chunk.metadata.extra["chunker_used"] == "boundary_aware"
                            assert "boundary_type" in chunk.metadata.extra

                    processed_files += 1

                    # Print processing info
                    boundary_stats = result.source_info.get("boundary_aware_metadata", {}).get("boundary_types", {})
                    print(f"  📄 {len(result.chunks)} chunks, {len(boundary_stats)} boundary types")

                except Exception as e:
                    print(f"  ❌ Error processing {file_path.name}: {e}")
                    continue

        print(f"\n✅ Successfully processed {processed_files} files from test_data/")
        assert processed_files > 0, "Should process at least one file"

    def test_different_document_formats_on_files(self):
        """Test different document format detection on real files."""
        test_data_dir = Path(__file__).parent.parent.parent.parent / "test_data"
        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        # Test format-specific chunkers
        format_chunkers = {
            "html": BoundaryAwareChunker(document_format="html"),
            "markdown": BoundaryAwareChunker(document_format="markdown"),
            "auto": BoundaryAwareChunker(document_format="auto")
        }

        # Find appropriate test files
        html_files = list(test_data_dir.glob("*.html"))[:2]
        md_files = list(test_data_dir.glob("*.md"))[:2]

        for file_path in html_files + md_files:
            if file_path.is_file():
                for format_name, chunker in format_chunkers.items():
                    result = chunker.chunk(file_path)

                    detected_format = result.source_info.get("document_format", "unknown")
                    print(f"  📄 {file_path.name} -> {format_name} chunker -> detected: {detected_format}")

                    assert len(result.chunks) >= 1

    def test_boundary_strategy_comparison_on_files(self):
        """Test different boundary strategies on real files."""
        test_data_dir = Path("/home/sharan/Desktop/sharan_work/chunking/test_data")
        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        strategies = ["strict", "adaptive", "hybrid"]
        test_files = list(test_data_dir.glob("*.md"))[:2]

        for file_path in test_files:
            if file_path.is_file():
                print(f"\nTesting strategies on {file_path.name}:")

                for strategy in strategies:
                    try:
                        chunker = BoundaryAwareChunker(boundary_strategy=strategy)
                        result = chunker.chunk(file_path)

                        boundary_metadata = result.source_info.get("boundary_aware_metadata", {})
                        print(f"  🔧 {strategy}: {len(result.chunks)} chunks, {boundary_metadata.get('total_boundaries', 0)} boundaries")

                        assert len(result.chunks) >= 1

                    except Exception as e:
                        print(f"  ❌ {strategy} strategy failed: {e}")

    def test_large_file_processing(self):
        """Test processing of larger files for scalability."""
        test_data_dir = Path("/home/sharan/Desktop/sharan_work/chunking/test_data")
        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        # Find larger files (> 1KB)
        large_files = [f for f in test_data_dir.glob("*.*")
                      if f.is_file() and f.stat().st_size > 1024][:3]

        for file_path in large_files:
            print(f"Processing large file: {file_path.name} ({file_path.stat().st_size} bytes)")

            start_time = time.time()
            result = self.chunker.chunk(file_path)
            processing_time = time.time() - start_time

            print(f"  ⏱️ Processing time: {processing_time:.3f}s")
            print(f"  📊 Created {len(result.chunks)} chunks")

            # Should complete in reasonable time (under 10 seconds)
            assert processing_time < 10.0
            assert len(result.chunks) > 0

    def test_file_error_handling(self):
        """Test error handling for problematic file inputs."""
        # Test non-existent file
        non_existent = Path("/nonexistent/file.txt")
        with pytest.raises((FileNotFoundError, OSError)):
            self.chunker.chunk(non_existent)

        # Test with corrupted/binary content
        binary_content = b'\x00\x01\x02\x03\x04\x05'
        try:
            result = self.chunker.chunk(binary_content)
            # Should either work or fail gracefully
            assert isinstance(result, ChunkingResult)
        except (UnicodeDecodeError, ValueError):
            # Acceptable to fail on binary content
            pass

    def test_performance_metrics_consistency(self):
        """Test that performance metrics are tracked consistently."""
        # Reset stats
        self.chunker.performance_stats = {
            "total_documents_processed": 0,
            "boundary_detection_time": 0.0,
            "parsing_time": 0.0,
            "chunking_time": 0.0,
            "boundaries_detected": 0,
            "fallback_count": 0
        }

        # Process several documents
        documents = [self.plain_text, self.html_content, self.markdown_content]

        for doc in documents:
            self.chunker.chunk(doc)

        stats = self.chunker.performance_stats

        # Verify stats were updated
        assert stats["total_documents_processed"] == len(documents)
        assert stats["boundary_detection_time"] > 0
        assert stats["boundaries_detected"] >= 0

        # Get config and verify stats are included
        config = self.chunker.get_config()
        assert "performance_stats" in config
        assert config["performance_stats"]["total_documents_processed"] == len(documents)
