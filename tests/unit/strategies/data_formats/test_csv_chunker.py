"""
Comprehensive tests for CSV chunking strategy.

This module tests all aspects of CSV chunking including:
- Row-based chunking with various sizes
- Logical grouping by column values
- Memory-based chunking
- Header section detection
- Edge cases and error handling
- Streaming capabilities
- Adaptive parameter adjustment
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import List

from chunking_strategy.strategies.data_formats.csv_chunker import CSVChunker
from chunking_strategy.core.base import Chunk, ChunkingResult, ModalityType
from chunking_strategy.core.registry import create_chunker, get_chunker


class TestCSVChunkerBasics:
    """Test basic CSV chunker functionality."""

    def test_chunker_registration(self):
        """Test that CSV chunker is properly registered."""
        chunker = create_chunker("csv_chunker")
        assert chunker is not None
        assert isinstance(chunker, CSVChunker)

    def test_chunker_initialization(self):
        """Test CSV chunker initialization with various parameters."""
        # Default initialization
        chunker = CSVChunker()
        assert chunker.chunk_by == "rows"
        assert chunker.rows_per_chunk == 1000
        assert chunker.preserve_headers is True

        # Custom initialization
        chunker = CSVChunker(
            chunk_by="logical_groups",
            group_by_column="category",
            preserve_headers=False
        )
        assert chunker.chunk_by == "logical_groups"
        assert chunker.group_by_column == "category"
        assert chunker.preserve_headers is False

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid chunk_by method
        with pytest.raises(ValueError):
            CSVChunker(chunk_by="invalid_method")

        # Missing group_by_column for logical_groups
        with pytest.raises(ValueError):
            CSVChunker(chunk_by="logical_groups")

        # Invalid rows_per_chunk
        with pytest.raises(ValueError):
            CSVChunker(rows_per_chunk=-1)

        # Invalid memory_limit_mb
        with pytest.raises(ValueError):
            CSVChunker(memory_limit_mb=-5)


class TestRowBasedChunking:
    """Test row-based chunking strategy."""

    def test_simple_row_chunking(self):
        """Test basic row-based chunking."""
        csv_content = """id,name,value
1,Alice,100
2,Bob,200
3,Charlie,300
4,Diana,400
5,Eve,500"""

        chunker = CSVChunker(chunk_by="rows", rows_per_chunk=2)
        result = chunker.chunk(csv_content)

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 3  # 5 rows / 2 = 3 chunks (2, 2, 1)
        assert result.strategy_used == "csv_chunker"

        # Check first chunk
        first_chunk = result.chunks[0]
        assert "id,name,value" in first_chunk.content  # Headers preserved
        assert "1,Alice,100" in first_chunk.content
        assert "2,Bob,200" in first_chunk.content
        assert first_chunk.metadata.extra["csv_row_count"] == 2

        # Check last chunk
        last_chunk = result.chunks[-1]
        assert "5,Eve,500" in last_chunk.content
        assert last_chunk.metadata.extra["csv_row_count"] == 1

    def test_chunk_overlap(self):
        """Test row-based chunking with overlap."""
        csv_content = """id,name,value
1,Alice,100
2,Bob,200
3,Charlie,300
4,Diana,400
5,Eve,500"""

        chunker = CSVChunker(
            chunk_by="rows",
            rows_per_chunk=2,
            chunk_overlap_rows=1
        )
        result = chunker.chunk(csv_content)

        # With overlap, chunks should share rows
        assert len(result.chunks) >= 2

        # Check that overlap is preserved in metadata
        for chunk in result.chunks:
            assert 'csv_start_row' in chunk.metadata.extra
            assert 'csv_end_row' in chunk.metadata.extra

    def test_headers_preservation(self):
        """Test header preservation options."""
        csv_content = """id,name,value
1,Alice,100
2,Bob,200"""

        # With headers
        chunker = CSVChunker(preserve_headers=True, rows_per_chunk=1)
        result = chunker.chunk(csv_content)
        for chunk in result.chunks:
            assert "id,name,value" in chunk.content

        # Without headers
        chunker = CSVChunker(preserve_headers=False, rows_per_chunk=1)
        result = chunker.chunk(csv_content)
        # Only first chunk should not have headers repeated
        assert "id,name,value" not in result.chunks[0].content

    def test_file_input(self):
        """Test chunking from file path."""
        # Use existing test file
        test_file = Path("test_data/simple_data.csv")

        chunker = CSVChunker(chunk_by="rows", rows_per_chunk=3)
        result = chunker.chunk(test_file)

        assert result.total_chunks > 0
        assert result.source_info["csv_rows"] == 10  # simple_data.csv has 10 data rows

        # Check source information
        first_chunk = result.chunks[0]
        assert str(test_file) in first_chunk.metadata.source


class TestLogicalGrouping:
    """Test logical grouping chunking strategy."""

    def test_group_by_column(self):
        """Test grouping by a specific column."""
        csv_content = """id,category,value
1,A,100
2,B,200
3,A,300
4,C,400
5,B,500
6,A,600"""

        chunker = CSVChunker(
            chunk_by="logical_groups",
            group_by_column="category"
        )
        result = chunker.chunk(csv_content)

        # Should have 3 groups: A, B, C
        assert len(result.chunks) == 3

        # Check that each chunk contains only one category
        categories = set()
        for chunk in result.chunks:
            lines = chunk.content.strip().split('\n')
            data_lines = lines[1:]  # Skip header
            chunk_categories = {line.split(',')[1] for line in data_lines if line.strip()}
            assert len(chunk_categories) == 1  # Each chunk should have only one category
            categories.update(chunk_categories)

        assert categories == {'A', 'B', 'C'}

    def test_group_by_missing_column(self):
        """Test behavior when group column doesn't exist."""
        csv_content = """id,name,value
1,Alice,100
2,Bob,200"""

        chunker = CSVChunker(
            chunk_by="logical_groups",
            group_by_column="nonexistent"
        )

        # Should fallback to row-based chunking
        result = chunker.chunk(csv_content)
        assert len(result.chunks) > 0  # Should still produce chunks

    def test_group_metadata(self):
        """Test metadata for grouped chunks."""
        csv_content = """id,region,sales
1,North,100
2,South,200
3,North,300"""

        chunker = CSVChunker(
            chunk_by="logical_groups",
            group_by_column="region"
        )
        result = chunker.chunk(csv_content)

        for chunk in result.chunks:
            assert 'csv_group_value' in chunk.metadata.extra
            assert 'csv_group_column' in chunk.metadata.extra
            assert chunk.metadata.extra['csv_group_column'] == "region"
            assert chunk.metadata.extra['csv_group_value'] in ["North", "South"]


class TestMemoryBasedChunking:
    """Test memory-based chunking strategy."""

    def test_memory_limit_chunking(self):
        """Test chunking based on memory limits."""
        # Create content that will exceed memory limit
        large_content = "id,data,description\n"
        for i in range(100):
            large_content += f"{i},{'x' * 100},{'description ' * 10}\n"

        chunker = CSVChunker(
            chunk_by="memory_size",
            memory_limit_mb=0.01  # Very small limit to force multiple chunks
        )
        result = chunker.chunk(large_content)

        assert len(result.chunks) > 1

        # Check that each chunk respects memory limits (approximately)
        for chunk in result.chunks:
            assert 'csv_memory_size_mb' in chunk.metadata.extra
            # Size should be reasonable (allowing some overhead)
            assert chunk.metadata.extra['csv_memory_size_mb'] <= 0.02

    def test_single_large_row(self):
        """Test handling of single very large row."""
        csv_content = f"id,data\n1,{'x' * 10000}"

        chunker = CSVChunker(
            chunk_by="memory_size",
            memory_limit_mb=0.001  # Smaller than the row
        )
        result = chunker.chunk(csv_content)

        # Should still create a chunk even if it exceeds limit
        assert len(result.chunks) == 1


class TestHeaderSectionChunking:
    """Test header section detection chunking."""

    def test_header_section_detection(self):
        """Test detection of header-like rows in data."""
        # Use the multi_section test file
        test_file = Path("test_data/multi_section.csv")

        chunker = CSVChunker(chunk_by="header_sections")
        result = chunker.chunk(test_file)

        # Should detect multiple sections
        assert len(result.chunks) >= 1

        # Each chunk should have appropriate metadata
        for chunk in result.chunks:
            assert 'csv_headers' in chunk.metadata.extra
            assert len(chunk.metadata.extra['csv_headers']) > 0


class TestDialectDetection:
    """Test CSV dialect detection and handling."""

    def test_auto_dialect_detection(self):
        """Test automatic dialect detection."""
        # Tab-separated content
        tsv_content = "id\tname\tvalue\n1\tAlice\t100\n2\tBob\t200"

        chunker = CSVChunker(dialect="auto")
        result = chunker.chunk(tsv_content)

        assert len(result.chunks) > 0
        # Should properly parse TSV format
        assert result.source_info.get("dialect") is not None

    def test_explicit_dialect(self):
        """Test explicit dialect specification."""
        csv_content = "id,name,value\n1,Alice,100\n2,Bob,200"

        chunker = CSVChunker(dialect="excel")
        result = chunker.chunk(csv_content)

        assert len(result.chunks) > 0
        assert result.source_info.get("dialect") == "Dialect"  # Based on csv.excel


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_csv(self):
        """Test handling of empty CSV."""
        chunker = CSVChunker()
        result = chunker.chunk("")

        assert result.total_chunks == 0
        assert len(result.chunks) == 0
        assert result.source_info["csv_rows"] == 0

    def test_headers_only_csv(self):
        """Test CSV with only headers."""
        csv_content = "id,name,value"

        chunker = CSVChunker()
        result = chunker.chunk(csv_content)

        assert result.total_chunks == 0
        assert result.source_info["headers"] == ["id", "name", "value"]

    def test_special_characters(self):
        """Test handling of special characters."""
        test_file = Path("test_data/special_characters.csv")

        chunker = CSVChunker(rows_per_chunk=5)
        result = chunker.chunk(test_file)

        assert len(result.chunks) > 0

        # Check that special characters are preserved
        content = ''.join(chunk.content for chunk in result.chunks)
        assert "O'Connor" in content
        assert "François" in content
        assert "Multi\nLine" in content

    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        chunker = CSVChunker()

        # Non-existent file should be treated as string content
        result = chunker.chunk("nonexistent.csv")
        # Should process as string content, not throw error
        assert isinstance(result, ChunkingResult)

    def test_skip_empty_lines(self):
        """Test skipping empty lines."""
        csv_content = """id,name,value
1,Alice,100

2,Bob,200

3,Charlie,300"""

        chunker = CSVChunker(skip_empty_lines=True)
        result = chunker.chunk(csv_content)

        # Should have 3 data rows, not 5
        assert result.source_info["csv_rows"] == 3

        chunker = CSVChunker(skip_empty_lines=False)
        result = chunker.chunk(csv_content)

        # Should include empty rows
        assert result.source_info["csv_rows"] == 5


class TestStreaming:
    """Test streaming capabilities."""

    def test_chunk_stream(self):
        """Test streaming chunk processing."""
        csv_content = """id,name,value
1,Alice,100
2,Bob,200
3,Charlie,300"""

        chunker = CSVChunker(rows_per_chunk=1)

        # Create a simple content stream
        content_stream = [csv_content]
        chunks = list(chunker.chunk_stream(content_stream))

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_byte_stream_handling(self):
        """Test handling of byte streams."""
        csv_content = """id,name,value
1,Alice,100
2,Bob,200"""

        chunker = CSVChunker(rows_per_chunk=1)

        # Create byte stream
        byte_stream = [csv_content.encode('utf-8')]
        chunks = list(chunker.chunk_stream(byte_stream))

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)


class TestAdaptation:
    """Test adaptive parameter adjustment."""

    def test_quality_feedback_adaptation(self):
        """Test adaptation based on quality feedback."""
        chunker = CSVChunker(chunk_by="rows", rows_per_chunk=1000)
        original_size = chunker.rows_per_chunk

        # Poor quality feedback should reduce chunk size
        chunker.adapt_parameters(feedback_score=0.3, feedback_type="quality")
        assert chunker.rows_per_chunk < original_size

    def test_performance_feedback_adaptation(self):
        """Test adaptation based on performance feedback."""
        chunker = CSVChunker(chunk_by="rows", rows_per_chunk=100)
        original_size = chunker.rows_per_chunk

        # Poor performance feedback should increase chunk size
        chunker.adapt_parameters(feedback_score=0.3, feedback_type="performance")
        assert chunker.rows_per_chunk > original_size

    def test_memory_adaptation(self):
        """Test adaptation for memory-based chunking."""
        chunker = CSVChunker(chunk_by="memory_size", memory_limit_mb=10)
        original_limit = chunker.memory_limit_mb

        # Poor quality should reduce memory limit
        chunker.adapt_parameters(feedback_score=0.2, feedback_type="quality")
        assert chunker.memory_limit_mb < original_limit

    def test_adaptation_bounds(self):
        """Test that adaptation respects bounds."""
        chunker = CSVChunker(chunk_by="rows", rows_per_chunk=100)

        # Multiple poor quality feedbacks should not go below minimum
        for _ in range(10):
            chunker.adapt_parameters(feedback_score=0.1, feedback_type="quality")

        assert chunker.rows_per_chunk >= 100  # Should respect minimum

    def test_adaptation_history(self):
        """Test adaptation history tracking."""
        chunker = CSVChunker()
        history = chunker.get_adaptation_history()

        assert isinstance(history, list)
        # Currently returns empty list, but structure should be correct


class TestIntegration:
    """Integration tests with other system components."""

    def test_registry_integration(self):
        """Test integration with chunker registry."""
        from chunking_strategy.core.registry import get_chunker_metadata, list_chunkers

        # Check that CSV chunker is in registry
        chunkers = list_chunkers(category="data_formats")
        assert "csv_chunker" in chunkers

        # Check metadata
        metadata = get_chunker_metadata("csv_chunker")
        assert metadata is not None
        assert metadata.category == "data_formats"
        assert "csv" in metadata.supported_formats

    def test_create_chunker_with_params(self):
        """Test creating chunker through registry with parameters."""
        chunker = create_chunker(
            "csv_chunker",
            chunk_by="logical_groups",
            group_by_column="region",
            rows_per_chunk=500
        )

        assert isinstance(chunker, CSVChunker)
        assert chunker.chunk_by == "logical_groups"
        assert chunker.group_by_column == "region"
        assert chunker.rows_per_chunk == 500

    def test_real_world_files(self):
        """Test with various real-world CSV files."""
        test_files = [
            "test_data/simple_data.csv",
            "test_data/complex_data.csv",
            "test_data/sales_by_region.csv",
            "test_data/large_dataset.csv"
        ]

        for test_file in test_files:
            if Path(test_file).exists():
                chunker = CSVChunker(rows_per_chunk=5)
                result = chunker.chunk(test_file)

                assert result.total_chunks > 0
                assert all(chunk.modality == ModalityType.TEXT for chunk in result.chunks)
                assert result.strategy_used == "csv_chunker"


class TestPerformance:
    """Performance and scalability tests."""

    def test_large_file_performance(self):
        """Test performance with large CSV files."""
        # Create a temporary large CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,data,timestamp,value\n")
            for i in range(10000):
                f.write(f"{i},data_{i},2023-01-01T{i%24:02d}:00:00,{i*1.5}\n")
            temp_file = f.name

        try:
            chunker = CSVChunker(chunk_by="rows", rows_per_chunk=1000)

            import time
            start_time = time.time()
            result = chunker.chunk(temp_file)
            processing_time = time.time() - start_time

            assert result.total_chunks == 10  # 10,000 / 1,000
            assert processing_time < 10.0  # Should complete within 10 seconds
            assert result.source_info["csv_rows"] == 10000

        finally:
            os.unlink(temp_file)

    def test_memory_efficiency(self):
        """Test memory efficiency with streaming."""
        # This test would need memory profiling tools for proper validation
        # For now, just ensure streaming doesn't fail
        csv_content = "id,value\n" + "\n".join(f"{i},{i*2}" for i in range(1000))

        chunker = CSVChunker(chunk_by="memory_size", memory_limit_mb=0.005)  # 5KB limit
        content_stream = [csv_content]

        chunks = list(chunker.chunk_stream(content_stream))
        assert len(chunks) > 1  # Should create multiple chunks


if __name__ == "__main__":
    pytest.main([__file__])
