"""
Tests for the MarkdownChunker class.

This module contains comprehensive tests for Markdown chunking functionality,
including header-based chunking, different strategies, frontmatter handling,
and various edge cases.
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, List

from chunking_strategy.strategies.document.markdown_chunker import MarkdownChunker
from chunking_strategy.core.base import Chunk, ChunkingResult, ModalityType
from chunking_strategy.core.registry import create_chunker


class TestMarkdownChunkerRegistration:
    """Test MarkdownChunker registration and initialization."""

    def test_chunker_registration(self):
        """Test that MarkdownChunker is properly registered."""
        chunker = create_chunker("markdown_chunker")
        assert chunker is not None
        assert isinstance(chunker, MarkdownChunker)
        assert chunker.name == "markdown_chunker"

    def test_initialization_defaults(self):
        """Test MarkdownChunker initialization with default parameters."""
        chunker = MarkdownChunker()
        assert chunker.chunk_by == "headers"
        assert chunker.header_level == 2
        assert chunker.preserve_code_blocks is True
        assert chunker.preserve_tables is True
        assert chunker.preserve_lists is True
        assert chunker.include_frontmatter is True
        assert chunker.chunk_size == 2000
        assert chunker.min_chunk_size == 100

    def test_initialization_custom_params(self):
        """Test MarkdownChunker initialization with custom parameters."""
        chunker = MarkdownChunker(
            chunk_by="sections",
            header_level=3,
            preserve_code_blocks=False,
            chunk_size=1500,
            min_chunk_size=50
        )
        assert chunker.chunk_by == "sections"
        assert chunker.header_level == 3
        assert chunker.preserve_code_blocks is False
        assert chunker.chunk_size == 1500
        assert chunker.min_chunk_size == 50


class TestMarkdownChunkerBasic:
    """Test basic Markdown chunking functionality."""

    def test_simple_header_chunking(self):
        """Test chunking with simple header structure."""
        content = """# Main Title

Content for main section.

## Section 1

Content for section 1.

## Section 2

Content for section 2."""

        chunker = MarkdownChunker(min_chunk_size=20)
        result = chunker.chunk(content)

        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "markdown_chunker"
        assert result.total_chunks == 3

        # Check first chunk
        first_chunk = result.chunks[0]
        assert "# Main Title" in first_chunk.content
        assert first_chunk.metadata.extra["markdown_header"] == "Main Title"
        assert first_chunk.metadata.extra["markdown_level"] == 1

    def test_nested_headers(self):
        """Test chunking with nested header structure."""
        content = """# Main Title

Main content.

## Section A

Section A content.

### Subsection A.1

Subsection content.

### Subsection A.2

More subsection content.

## Section B

Section B content."""

        chunker = MarkdownChunker(header_level=3, min_chunk_size=10)
        result = chunker.chunk(content)

        assert result.total_chunks == 5

        # Check header levels in metadata
        header_levels = [chunk.metadata.extra["markdown_level"] for chunk in result.chunks]
        assert 1 in header_levels  # H1
        assert 2 in header_levels  # H2
        assert 3 in header_levels  # H3

    def test_no_headers(self):
        """Test chunking content without headers."""
        content = """This is just plain markdown content without any headers.

It has multiple paragraphs and should be treated as a single chunk when no headers are present.

This tests the fallback behavior when header-based chunking finds no headers."""

        chunker = MarkdownChunker(min_chunk_size=20)
        result = chunker.chunk(content)

        assert result.total_chunks == 1
        assert result.chunks[0].metadata.extra["no_headers_found"] is True
        assert result.chunks[0].metadata.extra["chunk_type"] == "single"


class TestMarkdownChunkerStrategies:
    """Test different chunking strategies."""

    def test_header_based_strategy(self):
        """Test header-based chunking strategy."""
        content = """# Title

Content.

## Section 1

Section content.

## Section 2

More content."""

        chunker = MarkdownChunker(chunk_by="headers", min_chunk_size=10)
        result = chunker.chunk(content)

        assert result.total_chunks == 3
        for chunk in result.chunks:
            assert chunk.metadata.extra["chunk_type"] == "header_based"

    def test_section_based_strategy(self):
        """Test section-based chunking strategy."""
        content = """# Title

Content here.

## Section 1

Section content.

## Section 2

More content."""

        chunker = MarkdownChunker(chunk_by="sections", min_chunk_size=10)
        result = chunker.chunk(content)

        # Should create fewer, larger chunks
        assert result.total_chunks >= 1
        for chunk in result.chunks:
            assert chunk.metadata.extra["chunk_type"] == "section_based"

    def test_content_blocks_strategy(self):
        """Test content blocks chunking strategy."""
        content = """# Title

Content paragraph.

```python
def example():
    pass
```

Another paragraph.

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |"""

        chunker = MarkdownChunker(chunk_by="content_blocks", min_chunk_size=10)
        result = chunker.chunk(content)

        assert result.total_chunks >= 1
        for chunk in result.chunks:
            assert chunk.metadata.extra["chunk_type"] == "content_blocks"

    def test_fixed_size_strategy(self):
        """Test fixed-size chunking strategy."""
        content = """# Title

This is a longer piece of content that should be split into multiple chunks when using fixed-size chunking. It needs to be long enough to trigger the splitting behavior.

This is another paragraph that continues the content and should help ensure we have enough text to split across multiple chunks when using a small chunk size."""

        chunker = MarkdownChunker(chunk_by="fixed_size", chunk_size=100, min_chunk_size=10)
        result = chunker.chunk(content)

        assert result.total_chunks >= 2
        for chunk in result.chunks:
            assert chunk.metadata.extra["chunk_type"] == "fixed_size"


class TestMarkdownChunkerFrontmatter:
    """Test frontmatter handling."""

    def test_yaml_frontmatter(self):
        """Test handling of YAML frontmatter."""
        content = """---
title: "Test Document"
author: "Test Author"
date: "2025-01-01"
---

# Main Title

Content goes here."""

        chunker = MarkdownChunker(min_chunk_size=10)
        result = chunker.chunk(content)

        assert result.source_info["frontmatter_present"] is True
        assert result.chunks[0].metadata.extra["has_frontmatter"] is True
        assert "---" in result.chunks[0].content

    def test_no_frontmatter(self):
        """Test handling when no frontmatter is present."""
        content = """# Main Title

Content goes here."""

        chunker = MarkdownChunker(min_chunk_size=10)
        result = chunker.chunk(content)

        assert result.source_info["frontmatter_present"] is False
        assert result.chunks[0].metadata.extra["has_frontmatter"] is False

    def test_frontmatter_disabled(self):
        """Test disabling frontmatter inclusion."""
        content = """---
title: "Test Document"
---

# Main Title

Content goes here."""

        chunker = MarkdownChunker(include_frontmatter=False, min_chunk_size=10)
        result = chunker.chunk(content)

        # Frontmatter should not be included in chunks
        assert "title:" not in result.chunks[0].content


class TestMarkdownChunkerCodeBlocks:
    """Test code block handling."""

    def test_preserve_code_blocks(self):
        """Test that code blocks are preserved intact."""
        content = """# Code Example

Here's some code:

```python
def hello_world():
    print("Hello, world!")
    return True
```

End of example."""

        chunker = MarkdownChunker(preserve_code_blocks=True, min_chunk_size=10)
        result = chunker.chunk(content)

        # Code block should be in the content
        chunk_content = result.chunks[0].content
        assert "```python" in chunk_content
        assert "def hello_world():" in chunk_content
        assert "```" in chunk_content

    def test_inline_code(self):
        """Test handling of inline code."""
        content = """# Code Reference

Use the `print()` function to output text.

Also try `len()` for length."""

        chunker = MarkdownChunker(min_chunk_size=10)
        result = chunker.chunk(content)

        chunk_content = result.chunks[0].content
        assert "`print()`" in chunk_content
        assert "`len()`" in chunk_content


class TestMarkdownChunkerTables:
    """Test table handling."""

    def test_simple_table(self):
        """Test handling of simple tables."""
        content = """# Data Table

| Name | Age | City |
|------|-----|------|
| Alice | 25 | NYC |
| Bob | 30 | LA |

End of table."""

        chunker = MarkdownChunker(preserve_tables=True, min_chunk_size=10)
        result = chunker.chunk(content)

        chunk_content = result.chunks[0].content
        assert "| Name | Age | City |" in chunk_content
        assert "| Alice | 25 | NYC |" in chunk_content


class TestMarkdownChunkerLists:
    """Test list handling."""

    def test_unordered_list(self):
        """Test handling of unordered lists."""
        content = """# Features

- Feature A
- Feature B
- Feature C

End of list."""

        chunker = MarkdownChunker(preserve_lists=True, min_chunk_size=10)
        result = chunker.chunk(content)

        chunk_content = result.chunks[0].content
        assert "- Feature A" in chunk_content
        assert "- Feature B" in chunk_content

    def test_ordered_list(self):
        """Test handling of ordered lists."""
        content = """# Steps

1. First step
2. Second step
3. Third step

End of steps."""

        chunker = MarkdownChunker(preserve_lists=True, min_chunk_size=10)
        result = chunker.chunk(content)

        chunk_content = result.chunks[0].content
        assert "1. First step" in chunk_content
        assert "2. Second step" in chunk_content


class TestMarkdownChunkerFileInput:
    """Test file input handling."""

    def test_file_path_input(self):
        """Test chunking from file path."""
        content = """# Test Document

This is test content for file input testing.

## Section 1

Content for section 1."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            chunker = MarkdownChunker(min_chunk_size=10)
            result = chunker.chunk(temp_path)

            assert result.total_chunks >= 1
            assert result.source_info["source_type"] == "file"
            assert temp_path in result.source_info["source"]
        finally:
            os.unlink(temp_path)

    def test_path_object_input(self):
        """Test chunking from Path object."""
        content = """# Path Test

Content for path object testing."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            chunker = MarkdownChunker(min_chunk_size=10)
            result = chunker.chunk(temp_path)

            assert result.total_chunks >= 1
            assert result.source_info["source_type"] == "file"
        finally:
            temp_path.unlink()

    def test_bytes_input(self):
        """Test chunking from bytes input."""
        content = """# Bytes Test

Content for bytes testing."""

        chunker = MarkdownChunker(min_chunk_size=10)
        result = chunker.chunk(content.encode('utf-8'))

        assert result.total_chunks >= 1

    def test_invalid_file_path(self):
        """Test handling of invalid file path."""
        chunker = MarkdownChunker()

        # Use Path object to force file path interpretation
        with pytest.raises(FileNotFoundError):
            chunker.chunk(Path("nonexistent_file.md"))


class TestMarkdownChunkerEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_content(self):
        """Test chunking empty content."""
        chunker = MarkdownChunker()
        result = chunker.chunk("")

        assert result.total_chunks == 0

    def test_whitespace_only(self):
        """Test chunking whitespace-only content."""
        chunker = MarkdownChunker()
        result = chunker.chunk("   \n\n   \t   ")

        assert result.total_chunks == 0

    def test_headers_only(self):
        """Test content with only headers."""
        content = """# Title 1

## Title 2

### Title 3"""

        chunker = MarkdownChunker(min_chunk_size=5)
        result = chunker.chunk(content)

        assert result.total_chunks >= 1

    def test_very_large_chunk(self):
        """Test handling of very large chunks."""
        # Create content larger than max_chunk_size
        large_content = "# Large Document\n\n" + "This is a very long line. " * 500

        chunker = MarkdownChunker(max_chunk_size=1000, min_chunk_size=10)
        result = chunker.chunk(large_content)

        # Should split large chunks
        assert result.total_chunks >= 1

    def test_unsupported_content_type(self):
        """Test error handling for unsupported content types."""
        chunker = MarkdownChunker()

        with pytest.raises(ValueError):
            chunker.chunk(123)  # Invalid type


class TestMarkdownChunkerStreaming:
    """Test streaming functionality."""

    def test_can_stream(self):
        """Test streaming capability check."""
        chunker = MarkdownChunker()
        assert chunker.can_stream() is True

    def test_chunk_stream(self):
        """Test streaming chunking."""
        content_parts = [
            "# Title\n\n",
            "Content part 1.\n\n",
            "## Section\n\n",
            "Content part 2."
        ]

        chunker = MarkdownChunker(min_chunk_size=10)
        chunks = list(chunker.chunk_stream(content_parts))

        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)


class TestMarkdownChunkerAdaptation:
    """Test adaptive functionality."""

    def test_adapt_parameters_quality(self):
        """Test parameter adaptation based on quality feedback."""
        chunker = MarkdownChunker()
        initial_header_level = chunker.header_level
        initial_chunk_size = chunker.chunk_size

        # Poor quality feedback should make chunks more granular
        chunker.adapt_parameters(0.3, "quality")

        assert chunker.header_level <= initial_header_level
        assert chunker.chunk_size <= initial_chunk_size

    def test_adapt_parameters_performance(self):
        """Test parameter adaptation based on performance feedback."""
        chunker = MarkdownChunker()
        initial_max_chunk_size = chunker.max_chunk_size

        # Poor performance feedback should reduce max chunk size
        chunker.adapt_parameters(0.3, "performance")

        assert chunker.max_chunk_size <= initial_max_chunk_size

    def test_adaptation_history(self):
        """Test adaptation history tracking."""
        chunker = MarkdownChunker()

        # Initially no history
        history = chunker.get_adaptation_history()
        assert len(history) == 0

        # Add some adaptations
        chunker.adapt_parameters(0.4, "quality")
        chunker.adapt_parameters(0.6, "performance")

        history = chunker.get_adaptation_history()
        assert len(history) == 2

        # Check history structure
        for record in history:
            assert "timestamp" in record
            assert "feedback_score" in record
            assert "feedback_type" in record
            assert "old_config" in record
            assert "new_config" in record

    def test_custom_feedback(self):
        """Test adaptation with custom feedback parameters."""
        chunker = MarkdownChunker()
        initial_chunk_size = chunker.chunk_size

        chunker.adapt_parameters(0.5, "quality", chunks_too_large=True)

        assert chunker.chunk_size < initial_chunk_size


class TestMarkdownChunkerIntegration:
    """Test integration with the chunker registry and orchestrator."""

    def test_registry_integration(self):
        """Test integration with chunker registry."""
        from chunking_strategy.core.registry import list_chunkers

        chunkers = list_chunkers()
        assert "markdown_chunker" in chunkers

    def test_create_chunker_integration(self):
        """Test creating chunker through registry."""
        chunker = create_chunker("markdown_chunker", chunk_by="sections", header_level=3)

        assert isinstance(chunker, MarkdownChunker)
        assert chunker.chunk_by == "sections"
        assert chunker.header_level == 3

    def test_orchestrator_integration(self):
        """Test integration with orchestrator auto-selection."""
        from chunking_strategy import ChunkerOrchestrator

        # Create temporary markdown file with sufficient content to avoid small file override
        content = """# Test Document

This is a comprehensive test document with enough content to avoid being treated as a small file that gets overridden to use sentence-based chunking.

## Section 1

This section contains substantial content to ensure the file is large enough for the markdown chunker to be selected automatically by the orchestrator.

## Section 2

Additional content to make the file larger and more substantial for proper testing of the auto-selection mechanism.

## Section 3

Final section with more content to ensure we have sufficient text for proper markdown chunking strategy selection."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            config = {'strategies': {'primary': 'auto'}}
            orchestrator = ChunkerOrchestrator(config=config)
            result = orchestrator.chunk_file(temp_path)

            assert result.strategy_used == "markdown_chunker"
        finally:
            os.unlink(temp_path)


class TestMarkdownChunkerMetadata:
    """Test metadata generation and structure analysis."""

    def test_structure_analysis(self):
        """Test document structure analysis."""
        content = """# Main Title

Content.

## Section 1

Content.

### Subsection

Content.

```python
code
```

| table |
|-------|
| data  |"""

        chunker = MarkdownChunker(min_chunk_size=10)
        result = chunker.chunk(content)

        structure = result.source_info["markdown_structure"]
        assert structure["total_headers"] == 3
        assert structure["header_distribution"]["h1"] == 1
        assert structure["header_distribution"]["h2"] == 1
        assert structure["header_distribution"]["h3"] == 1
        assert structure["code_blocks"] >= 1
        assert structure["structure_type"] == "hierarchical"

    def test_chunk_metadata(self):
        """Test individual chunk metadata."""
        content = """# Main Title

Content for main title.

## Section 1

Content for section 1."""

        chunker = MarkdownChunker(min_chunk_size=10)
        result = chunker.chunk(content)

        for chunk in result.chunks:
            metadata = chunk.metadata
            assert metadata.source_type == "content"
            assert chunk.modality == ModalityType.TEXT
            assert "markdown_header" in metadata.extra
            assert "markdown_level" in metadata.extra
            assert "section_index" in metadata.extra


if __name__ == "__main__":
    pytest.main([__file__])
