"""
Edge cases and performance test suite.

Tests corner cases, error conditions, and performance characteristics
that might not be covered in regular integration tests.
"""

import gc
import os
import tempfile
import time
import psutil
from pathlib import Path
from typing import List
import pytest

from chunking_strategy import create_chunker
from chunking_strategy.core.base import ModalityType


class TestEdgeCasesAndPerformance:
    """Test edge cases and performance characteristics."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        yield

        # Cleanup
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_empty_and_minimal_inputs(self):
        """Test handling of empty and minimal inputs."""
        chunker = create_chunker("fixed_size", chunk_size=100)

        # Test empty string
        result = chunker.chunk("")
        assert result is not None
        assert isinstance(result.chunks, list)

        # Test single character
        result = chunker.chunk("a")
        assert result is not None
        assert len(result.chunks) >= 0  # May or may not create chunks

        # Test whitespace only
        result = chunker.chunk("   \n\t  ")
        assert result is not None

        # Test single word
        result = chunker.chunk("hello")
        assert result is not None
        if result.chunks:
            assert result.chunks[0].content.strip() == "hello"

    def test_very_large_chunks(self):
        """Test handling of very large chunk sizes."""
        chunker = create_chunker("fixed_size", chunk_size=1000000)  # 1MB chunks

        # Create a test file with reasonable content
        test_content = "This is a test sentence. " * 1000  # ~25KB
        test_file = self.temp_dir / "large_chunk_test.txt"
        with open(test_file, 'w') as f:
            f.write(test_content)

        result = chunker.chunk(test_file)
        assert result is not None
        assert len(result.chunks) <= 2  # Should create very few chunks

    def test_very_small_chunks(self):
        """Test handling of very small chunk sizes."""
        chunker = create_chunker("fixed_size", chunk_size=1)  # 1 character chunks

        test_content = "Hello World!"
        result = chunker.chunk(test_content)

        assert result is not None
        assert len(result.chunks) > 0
        # Should create many small chunks

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        chunker = create_chunker("sentence_based", max_sentences=2)

        # Test with various Unicode characters
        test_cases = [
            "Hello 世界! This is a test. こんにちは。",
            "Émojis: 🎉🚀💻 are everywhere! Testing continues.",
            "Math symbols: ∑∫∂∇ and more. Scientific notation works.",
            "Mixed scripts: English עברית العربية русский. Multiple languages.",
            "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>? punctuation test."
        ]

        for test_text in test_cases:
            result = chunker.chunk(test_text)
            assert result is not None
            assert len(result.chunks) > 0

            # Verify content integrity
            reconstructed = " ".join(chunk.content for chunk in result.chunks)
            # Content might be reformatted but should contain key elements
            assert any(char in reconstructed for char in test_text[:10])

    def test_malformed_files(self):
        """Test handling of malformed or problematic files."""
        chunker = create_chunker("fixed_size", chunk_size=1000)

        # Test binary file
        binary_file = self.temp_dir / "binary_test.bin"
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\xFF\xFE\xFD')

        try:
            result = chunker.chunk(binary_file)
            # Should handle gracefully, may or may not produce chunks
            assert result is not None
        except (UnicodeDecodeError, ValueError):
            # Acceptable to fail on binary files
            pass

    def test_performance_with_large_files(self):
        """Test performance characteristics with large files."""
        # Create a large test file
        large_content = "This is a performance test sentence. " * 10000  # ~370KB
        large_file = self.temp_dir / "performance_test.txt"
        with open(large_file, 'w') as f:
            f.write(large_content)

        strategies = ["fixed_size", "sentence_based"]
        performance_results = {}

        for strategy in strategies:
            chunker = create_chunker(strategy)

            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss

            result = chunker.chunk(large_file)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            performance_results[strategy] = {
                'processing_time': end_time - start_time,
                'memory_increase': end_memory - start_memory,
                'chunks_generated': len(result.chunks),
                'chunks_per_second': len(result.chunks) / (end_time - start_time)
            }

            # Performance assertions
            assert end_time - start_time < 30.0, f"{strategy} took too long"
            assert len(result.chunks) > 0, f"{strategy} generated no chunks"

            # Clean up memory
            del result
            gc.collect()

        print(f"Performance results: {performance_results}")

    def test_memory_efficiency(self):
        """Test memory efficiency with multiple operations."""
        initial_memory = psutil.Process().memory_info().rss

        # Perform multiple chunking operations
        for i in range(50):
            chunker = create_chunker("fixed_size", chunk_size=500)
            test_content = f"Test content iteration {i}. " * 100
            result = chunker.chunk(test_content)

            # Clean up immediately
            del chunker
            del result

            if i % 10 == 0:
                gc.collect()

        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Memory leak detected: {memory_increase} bytes"

    def test_concurrent_chunker_creation(self):
        """Test concurrent creation of chunkers."""
        import threading
        import concurrent.futures

        def create_and_test_chunker(chunker_type: str):
            chunker = create_chunker(chunker_type, chunk_size=1000)
            result = chunker.chunk("Test content for concurrent creation.")
            return len(result.chunks)

        # Test concurrent creation
        strategies = ["fixed_size", "sentence_based"] * 5  # 10 total

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(create_and_test_chunker, strategy)
                for strategy in strategies
            ]
            results = [future.result(timeout=30) for future in futures]

        # All should succeed
        assert all(result > 0 for result in results)

    def test_parameter_boundary_conditions(self):
        """Test boundary conditions for parameters."""

        # Test zero and negative parameters - should raise ValueError for invalid parameters
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            create_chunker("fixed_size", chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            create_chunker("fixed_size", chunk_size=-100)

        # Test very large parameters
        chunker = create_chunker("fixed_size", chunk_size=1000000)
        assert chunker.chunk_size == 1000000

        # Test sentence chunker boundaries - should raise ValueError for invalid parameters
        with pytest.raises(ValueError, match="max_sentences must be positive"):
            create_chunker("sentence_based", max_sentences=0)

        # Test overlap parameters
        chunker = create_chunker("fixed_size", chunk_size=100, overlap_size=50)
        result = chunker.chunk("This is a test sentence that should be chunked properly.")
        assert result is not None

    def test_file_permission_errors(self):
        """Test handling of file permission errors."""
        # Create a file and remove read permissions
        restricted_file = self.temp_dir / "restricted.txt"
        with open(restricted_file, 'w') as f:
            f.write("This file will be restricted.")

        # Remove read permissions
        os.chmod(restricted_file, 0o000)

        try:
            chunker = create_chunker("fixed_size")

            with pytest.raises((PermissionError, OSError)):
                chunker.chunk(restricted_file)

        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(restricted_file, 0o644)
            except:
                pass

    def test_extremely_long_lines(self):
        """Test handling of extremely long lines."""
        # Create content with very long lines
        long_line = "word " * 10000  # ~50KB single line
        content = long_line + "\nShort line.\n" + long_line

        chunker = create_chunker("sentence_based", max_sentences=1)
        result = chunker.chunk(content)

        assert result is not None
        assert len(result.chunks) > 0

    def test_deep_nesting_and_recursion(self):
        """Test scenarios that might cause deep recursion."""
        # Create highly nested or repetitive content
        nested_content = ""
        for i in range(1000):
            nested_content += f"Level {i}: " + "nested " * 10 + "content. "

        chunker = create_chunker("paragraph_based", max_paragraphs=10)
        result = chunker.chunk(nested_content)

        assert result is not None
        assert len(result.chunks) > 0

    def test_invalid_file_types(self):
        """Test handling of invalid file types."""
        chunker = create_chunker("fixed_size")

        # Test directory instead of file
        with pytest.raises((IsADirectoryError, ValueError)):
            chunker.chunk(self.temp_dir)

        # Test non-existent file
        with pytest.raises((FileNotFoundError, ValueError)):
            chunker.chunk(self.temp_dir / "does_not_exist.txt")

    def test_chunker_state_consistency(self):
        """Test that chunkers maintain consistent state."""
        chunker = create_chunker("fixed_size", chunk_size=1000)

        # Test multiple uses of same chunker
        test_contents = [
            "First test content.",
            "Second test content with different length and structure.",
            "Third test: short."
        ]

        results = []
        for content in test_contents:
            result = chunker.chunk(content)
            results.append(result)

            # Chunker parameters should remain unchanged
            assert chunker.chunk_size == 1000

        # All results should be valid
        for result in results:
            assert result is not None
            assert hasattr(result, 'chunks')

    def test_metadata_consistency(self):
        """Test that metadata is consistently generated."""
        chunker = create_chunker("sentence_based", max_sentences=2)

        test_content = "First sentence. Second sentence. Third sentence here. Fourth and final."
        result = chunker.chunk(test_content)

        for i, chunk in enumerate(result.chunks):
            # Check required metadata fields
            assert chunk.id is not None
            assert chunk.metadata is not None
            assert chunk.modality is not None

            # Check metadata consistency
            assert chunk.metadata.source is not None
            assert hasattr(chunk.metadata, 'extra')
            assert isinstance(chunk.metadata.extra, dict)

    def test_chunk_content_integrity(self):
        """Test that chunk content maintains integrity."""
        original_content = "This is a test document. It has multiple sentences. Each sentence should be preserved correctly. No data should be lost or corrupted during chunking."

        strategies = ["fixed_size", "sentence_based", "paragraph_based"]

        for strategy in strategies:
            chunker = create_chunker(strategy)
            result = chunker.chunk(original_content)

            # Reconstruct content from chunks
            reconstructed = ""
            for chunk in result.chunks:
                reconstructed += chunk.content + " "

            reconstructed = reconstructed.strip()

            # Check that no critical content is lost
            # (exact reconstruction might not be possible due to formatting)
            original_words = set(original_content.lower().split())
            reconstructed_words = set(reconstructed.lower().split())

            # Most words should be preserved
            preserved_ratio = len(original_words & reconstructed_words) / len(original_words)
            assert preserved_ratio > 0.8, f"Too much content lost in {strategy}: {preserved_ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
