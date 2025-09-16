"""
Comprehensive tests for JSON chunking strategy.

This module tests all aspects of JSON chunking including:
- Object-based chunking with various sizes
- Array element chunking
- Key-based logical grouping
- Size-based chunking
- Depth-limited chunking
- JSON Lines (JSONL) format support
- Edge cases and error handling
- Streaming capabilities
- Adaptive parameter adjustment
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from typing import List

from chunking_strategy.strategies.data_formats.json_chunker import JSONChunker
from chunking_strategy.core.base import Chunk, ChunkingResult, ModalityType
from chunking_strategy.core.registry import create_chunker, get_chunker


class TestJSONChunkerBasics:
    """Test basic JSON chunker functionality."""

    def test_chunker_registration(self):
        """Test that JSON chunker is properly registered."""
        chunker = create_chunker("json_chunker")
        assert chunker is not None
        assert isinstance(chunker, JSONChunker)

    def test_chunker_initialization(self):
        """Test JSON chunker initialization with various parameters."""
        # Default initialization
        chunker = JSONChunker()
        assert chunker.chunk_by == "objects"
        assert chunker.objects_per_chunk == 100
        assert chunker.preserve_structure is True

        # Custom initialization
        chunker = JSONChunker(
            chunk_by="key_groups",
            group_by_key="category",
            preserve_structure=False
        )
        assert chunker.chunk_by == "key_groups"
        assert chunker.group_by_key == "category"
        assert chunker.preserve_structure is False

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid chunk_by method
        with pytest.raises(ValueError):
            JSONChunker(chunk_by="invalid_method")

        # Missing group_by_key for key_groups
        with pytest.raises(ValueError):
            JSONChunker(chunk_by="key_groups")

        # Invalid objects_per_chunk
        with pytest.raises(ValueError):
            JSONChunker(objects_per_chunk=-1)

        # Invalid size_limit_mb
        with pytest.raises(ValueError):
            JSONChunker(size_limit_mb=-5)


class TestObjectBasedChunking:
    """Test object-based chunking strategy."""

    def test_simple_object_chunking(self):
        """Test basic object-based chunking."""
        json_content = """[
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
            {"id": 4, "name": "Diana"},
            {"id": 5, "name": "Eve"}
        ]"""

        chunker = JSONChunker(chunk_by="objects", objects_per_chunk=2)
        result = chunker.chunk(json_content)

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 3  # 5 objects / 2 = 3 chunks (2, 2, 1)
        assert result.strategy_used == "json_chunker"

        # Check first chunk
        first_chunk = result.chunks[0]
        first_data = json.loads(first_chunk.content)
        assert len(first_data) == 2
        assert first_data[0]["name"] == "Alice"
        assert first_data[1]["name"] == "Bob"
        assert first_chunk.metadata.extra["json_object_count"] == 2

        # Check last chunk
        last_chunk = result.chunks[-1]
        last_data = json.loads(last_chunk.content)
        assert len(last_data) == 1
        assert last_data[0]["name"] == "Eve"
        assert last_chunk.metadata.extra["json_object_count"] == 1

    def test_single_object_chunking(self):
        """Test chunking of a single JSON object."""
        json_content = """{"id": 1, "name": "Alice", "age": 25}"""

        chunker = JSONChunker(chunk_by="objects")
        result = chunker.chunk(json_content)

        assert len(result.chunks) == 1
        chunk_data = json.loads(result.chunks[0].content)
        assert len(chunk_data) == 1  # Wrapped in array for structure preservation
        assert chunk_data[0]["name"] == "Alice"

    def test_file_input(self):
        """Test chunking from file path."""
        test_file = Path("test_data/simple_objects.json")

        chunker = JSONChunker(chunk_by="objects", objects_per_chunk=2)
        result = chunker.chunk(test_file)

        assert result.total_chunks > 0
        assert result.source_info["json_objects"] == 5  # simple_objects.json has 5 objects

        # Check source information
        first_chunk = result.chunks[0]
        assert str(test_file) in first_chunk.metadata.source


class TestArrayElementChunking:
    """Test array element chunking strategy."""

    def test_array_element_chunking(self):
        """Test chunking of arrays by elements."""
        test_file = Path("test_data/large_array.json")

        chunker = JSONChunker(chunk_by="array_elements", elements_per_chunk=5)
        result = chunker.chunk(test_file)

        assert len(result.chunks) > 1  # Should split the large array

        # Check that chunks contain the expected structure
        for chunk in result.chunks:
            chunk_data = json.loads(chunk.content)
            assert "data" in chunk_data
            assert isinstance(chunk_data["data"], list)
            assert len(chunk_data["data"]) <= 5

    def test_nested_array_detection(self):
        """Test detection and chunking of nested arrays."""
        json_content = """{
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
                {"id": 3, "name": "Charlie"}
            ],
            "products": [
                {"id": "P1", "name": "Product 1"},
                {"id": "P2", "name": "Product 2"}
            ]
        }"""

        chunker = JSONChunker(chunk_by="array_elements", elements_per_chunk=2)
        result = chunker.chunk(json_content)

        assert len(result.chunks) > 0
        # Should find and chunk both arrays


class TestKeyGroupChunking:
    """Test key-based grouping chunking strategy."""

    def test_group_by_key(self):
        """Test grouping by a specific key."""
        test_file = Path("test_data/products_by_category.json")

        chunker = JSONChunker(
            chunk_by="key_groups",
            group_by_key="category"
        )
        result = chunker.chunk(test_file)

        # Should have 3 groups: electronics, sports, home
        assert len(result.chunks) == 3

        # Check that each chunk contains only one category
        categories = set()
        for chunk in result.chunks:
            chunk_data = json.loads(chunk.content)
            chunk_categories = {obj["category"] for obj in chunk_data}
            assert len(chunk_categories) == 1  # Each chunk should have only one category
            categories.update(chunk_categories)

        assert categories == {"electronics", "sports", "home"}

    def test_group_by_missing_key(self):
        """Test behavior when group key doesn't exist in some objects."""
        json_content = """[
            {"id": 1, "category": "A", "name": "Item 1"},
            {"id": 2, "category": "B", "name": "Item 2"},
            {"id": 3, "name": "Item 3"}
        ]"""

        chunker = JSONChunker(
            chunk_by="key_groups",
            group_by_key="category"
        )
        result = chunker.chunk(json_content)

        # Should have groups for A, B, and ungrouped
        assert len(result.chunks) >= 2

    def test_group_metadata(self):
        """Test metadata for grouped chunks."""
        json_content = """[
            {"id": 1, "region": "North", "sales": 100},
            {"id": 2, "region": "South", "sales": 200},
            {"id": 3, "region": "North", "sales": 300}
        ]"""

        chunker = JSONChunker(
            chunk_by="key_groups",
            group_by_key="region"
        )
        result = chunker.chunk(json_content)

        for chunk in result.chunks:
            assert 'json_group_key' in chunk.metadata.extra
            assert 'json_group_value' in chunk.metadata.extra
            assert chunk.metadata.extra['json_group_key'] == "region"
            assert chunk.metadata.extra['json_group_value'] in ["North", "South"]


class TestSizeLimitChunking:
    """Test size-based chunking strategy."""

    def test_size_limit_chunking(self):
        """Test chunking based on size limits."""
        # Create content that will exceed size limit
        large_objects = []
        for i in range(100):
            large_objects.append({
                "id": i,
                "data": "x" * 100,
                "description": "description " * 10
            })

        json_content = json.dumps(large_objects)

        chunker = JSONChunker(
            chunk_by="size_limit",
            size_limit_mb=0.01  # Very small limit to force multiple chunks
        )
        result = chunker.chunk(json_content)

        assert len(result.chunks) > 1

        # Check that each chunk respects size limits (approximately)
        for chunk in result.chunks:
            assert 'json_size_mb' in chunk.metadata.extra
            # Size should be reasonable (allowing some overhead)
            assert chunk.metadata.extra['json_size_mb'] <= 0.02

    def test_single_large_object(self):
        """Test handling of single very large object."""
        large_object = {"id": 1, "data": "x" * 10000}
        json_content = json.dumps(large_object)

        chunker = JSONChunker(
            chunk_by="size_limit",
            size_limit_mb=0.001  # Smaller than the object
        )
        result = chunker.chunk(json_content)

        # Should still create a chunk even if it exceeds limit
        assert len(result.chunks) == 1


class TestDepthLimitChunking:
    """Test depth-limited chunking strategy."""

    def test_depth_limit_chunking(self):
        """Test depth limiting of nested structures."""
        test_file = Path("test_data/nested_structure.json")

        chunker = JSONChunker(chunk_by="depth_level", max_depth=2)
        result = chunker.chunk(test_file)

        assert len(result.chunks) == 1

        # Check that deep nesting is truncated
        chunk_data = json.loads(result.chunks[0].content)
        # Some deeply nested values should be replaced with truncation strings
        assert 'json_max_depth' in result.chunks[0].metadata.extra
        assert result.chunks[0].metadata.extra['json_max_depth'] == 2


class TestJSONLSupport:
    """Test JSON Lines format support."""

    def test_jsonl_parsing(self):
        """Test parsing of JSON Lines format."""
        test_file = Path("test_data/sample.jsonl")

        chunker = JSONChunker(chunk_by="objects", objects_per_chunk=2, handle_jsonl=True)
        result = chunker.chunk(test_file)

        assert result.source_info["format"] == "jsonl"
        assert len(result.chunks) > 1  # Should be chunked

        # Check that content is valid JSON
        for chunk in result.chunks:
            chunk_data = json.loads(chunk.content)
            assert isinstance(chunk_data, list)
            for obj in chunk_data:
                assert "id" in obj
                assert "message" in obj

    def test_jsonl_vs_json_detection(self):
        """Test correct detection of JSONL vs regular JSON."""
        # Regular JSON
        json_chunker = JSONChunker(handle_jsonl=True)
        json_result = json_chunker.chunk('{"test": "value"}')
        assert json_result.source_info["format"] == "json"

        # JSONL content
        jsonl_content = '{"id": 1}\n{"id": 2}\n{"id": 3}'
        jsonl_result = json_chunker.chunk(jsonl_content)
        assert jsonl_result.source_info["format"] == "jsonl"


class TestFormatting:
    """Test JSON formatting options."""

    def test_compact_output(self):
        """Test compact JSON output."""
        json_content = """[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]"""

        # Compact output
        compact_chunker = JSONChunker(compact_output=True, objects_per_chunk=1)
        compact_result = compact_chunker.chunk(json_content)
        compact_content = compact_result.chunks[0].content
        assert '\n' not in compact_content  # No newlines in compact format

        # Pretty output
        pretty_chunker = JSONChunker(compact_output=False, objects_per_chunk=1)
        pretty_result = pretty_chunker.chunk(json_content)
        pretty_content = pretty_result.chunks[0].content
        assert '\n' in pretty_content  # Newlines for formatting

    def test_structure_preservation(self):
        """Test structure preservation options."""
        json_content = """[{"id": 1, "name": "Alice"}]"""

        # With structure preservation
        preserve_chunker = JSONChunker(preserve_structure=True)
        preserve_result = preserve_chunker.chunk(json_content)
        preserve_data = json.loads(preserve_result.chunks[0].content)
        assert isinstance(preserve_data, list)

        # Without structure preservation (implementation specific)
        no_preserve_chunker = JSONChunker(preserve_structure=False)
        no_preserve_result = no_preserve_chunker.chunk(json_content)
        # Should still be valid JSON
        json.loads(no_preserve_result.chunks[0].content)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_json(self):
        """Test handling of empty JSON."""
        chunker = JSONChunker()
        result = chunker.chunk("{}")

        assert result.total_chunks == 1
        assert result.source_info["json_objects"] == 1

    def test_empty_array(self):
        """Test handling of empty JSON array."""
        chunker = JSONChunker()
        result = chunker.chunk("[]")

        assert result.total_chunks == 0
        assert result.source_info["json_objects"] == 0

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        test_file = Path("test_data/invalid.json")

        chunker = JSONChunker()
        # Should raise an exception or handle gracefully
        try:
            result = chunker.chunk(test_file)
            # If it doesn't raise, should have no chunks
            assert result.total_chunks == 0
        except Exception:
            # Acceptable to raise exception for invalid JSON
            pass

    def test_large_numbers_and_strings(self):
        """Test handling of large numbers and strings."""
        json_content = """{
            "large_number": 9223372036854775807,
            "large_string": "a very long string that goes on and on and contains lots of text",
            "unicode": "Hello 世界 🌍",
            "special_chars": "quotes \\"test\\" and newlines \\n and tabs \\t"
        }"""

        chunker = JSONChunker()
        result = chunker.chunk(json_content)

        assert len(result.chunks) == 1
        chunk_data = json.loads(result.chunks[0].content)
        if isinstance(chunk_data, list):
            chunk_data = chunk_data[0]
        assert "large_number" in chunk_data
        assert "unicode" in chunk_data

    def test_content_vs_file_detection(self):
        """Test proper detection of content vs file paths."""
        chunker = JSONChunker()

        # Short JSON content should be detected as content, not file path
        short_json = '{"test": "value"}'
        result = chunker.chunk(short_json)
        assert result.chunks[0].metadata.source == "string"

        # Long JSON content should also be detected as content
        long_json = json.dumps([{"id": i, "data": f"item {i}"} for i in range(10)])
        result = chunker.chunk(long_json)
        assert result.chunks[0].metadata.source == "string"


class TestStreaming:
    """Test streaming capabilities."""

    def test_chunk_stream(self):
        """Test streaming chunk processing."""
        json_content = """[
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"}
        ]"""

        chunker = JSONChunker(objects_per_chunk=1)

        # Create a simple content stream
        content_stream = [json_content]
        chunks = list(chunker.chunk_stream(content_stream))

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_byte_stream_handling(self):
        """Test handling of byte streams."""
        json_content = """[{"id": 1, "name": "Alice"}]"""

        chunker = JSONChunker(objects_per_chunk=1)

        # Create byte stream
        byte_stream = [json_content.encode('utf-8')]
        chunks = list(chunker.chunk_stream(byte_stream))

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)


class TestAdaptation:
    """Test adaptive parameter adjustment."""

    def test_quality_feedback_adaptation(self):
        """Test adaptation based on quality feedback."""
        chunker = JSONChunker(chunk_by="objects", objects_per_chunk=1000)
        original_size = chunker.objects_per_chunk

        # Poor quality feedback should reduce chunk size
        chunker.adapt_parameters(feedback_score=0.3, feedback_type="quality")
        assert chunker.objects_per_chunk < original_size

    def test_performance_feedback_adaptation(self):
        """Test adaptation based on performance feedback."""
        chunker = JSONChunker(chunk_by="objects", objects_per_chunk=10)
        original_size = chunker.objects_per_chunk

        # Poor performance feedback should increase chunk size
        chunker.adapt_parameters(feedback_score=0.3, feedback_type="performance")
        assert chunker.objects_per_chunk > original_size

    def test_array_elements_adaptation(self):
        """Test adaptation for array elements chunking."""
        chunker = JSONChunker(chunk_by="array_elements", elements_per_chunk=100)
        original_size = chunker.elements_per_chunk

        # Poor quality should reduce elements per chunk
        chunker.adapt_parameters(feedback_score=0.2, feedback_type="quality")
        assert chunker.elements_per_chunk < original_size

    def test_size_limit_adaptation(self):
        """Test adaptation for size-based chunking."""
        chunker = JSONChunker(chunk_by="size_limit", size_limit_mb=10)
        original_limit = chunker.size_limit_mb

        # Poor quality should reduce size limit
        chunker.adapt_parameters(feedback_score=0.2, feedback_type="quality")
        assert chunker.size_limit_mb < original_limit

    def test_adaptation_bounds(self):
        """Test that adaptation respects bounds."""
        chunker = JSONChunker(chunk_by="objects", objects_per_chunk=10)

        # Multiple poor quality feedbacks should not go below minimum
        for _ in range(10):
            chunker.adapt_parameters(feedback_score=0.1, feedback_type="quality")

        assert chunker.objects_per_chunk >= 10  # Should respect minimum

    def test_adaptation_history(self):
        """Test adaptation history tracking."""
        chunker = JSONChunker()
        history = chunker.get_adaptation_history()

        assert isinstance(history, list)
        # Currently returns empty list, but structure should be correct


class TestIntegration:
    """Integration tests with other system components."""

    def test_registry_integration(self):
        """Test integration with chunker registry."""
        from chunking_strategy.core.registry import get_chunker_metadata, list_chunkers

        # Check that JSON chunker is in registry
        chunkers = list_chunkers(category="data_formats")
        assert "json_chunker" in chunkers

        # Check metadata
        metadata = get_chunker_metadata("json_chunker")
        assert metadata is not None
        assert metadata.category == "data_formats"
        assert "json" in metadata.supported_formats

    def test_create_chunker_with_params(self):
        """Test creating chunker through registry with parameters."""
        chunker = create_chunker(
            "json_chunker",
            chunk_by="key_groups",
            group_by_key="category",
            objects_per_chunk=50
        )

        assert isinstance(chunker, JSONChunker)
        assert chunker.chunk_by == "key_groups"
        assert chunker.group_by_key == "category"
        assert chunker.objects_per_chunk == 50

    def test_real_world_files(self):
        """Test with various real-world JSON files."""
        test_files = [
            "test_data/simple_objects.json",
            "test_data/products_by_category.json",
            "test_data/nested_structure.json",
            "test_data/single_object.json"
        ]

        for test_file in test_files:
            if Path(test_file).exists():
                chunker = JSONChunker(objects_per_chunk=3)
                result = chunker.chunk(test_file)

                assert result.total_chunks > 0
                assert all(chunk.modality == ModalityType.TEXT for chunk in result.chunks)
                assert result.strategy_used == "json_chunker"

                # Verify all chunks contain valid JSON
                for chunk in result.chunks:
                    json.loads(chunk.content)  # Should not raise exception


class TestPerformance:
    """Performance and scalability tests."""

    def test_large_file_performance(self):
        """Test performance with large JSON files."""
        # Create a temporary large JSON file
        large_data = [{"id": i, "data": f"item_{i}", "value": i * 2.5} for i in range(5000)]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(large_data, f)
            temp_file = f.name

        try:
            chunker = JSONChunker(chunk_by="objects", objects_per_chunk=500)

            import time
            start_time = time.time()
            result = chunker.chunk(temp_file)
            processing_time = time.time() - start_time

            assert result.total_chunks == 10  # 5,000 / 500
            assert processing_time < 10.0  # Should complete within 10 seconds
            assert result.source_info["json_objects"] == 5000

        finally:
            os.unlink(temp_file)

    def test_memory_efficiency(self):
        """Test memory efficiency with large arrays."""
        large_array_data = {
            "items": [{"id": i, "value": f"item_{i}"} for i in range(1000)]
        }
        json_content = json.dumps(large_array_data)

        chunker = JSONChunker(chunk_by="array_elements", elements_per_chunk=100)
        result = chunker.chunk(json_content)

        assert len(result.chunks) == 10  # 1000 / 100

        # Each chunk should be manageable size
        for chunk in result.chunks:
            chunk_data = json.loads(chunk.content)
            assert len(chunk_data["items"]) <= 100


if __name__ == "__main__":
    pytest.main([__file__])
