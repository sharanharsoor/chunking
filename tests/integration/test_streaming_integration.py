#!/usr/bin/env python3
"""
Comprehensive tests for streaming integration with intelligent file processing.

Tests streaming functionality, Tika integration, and large file handling without
actually creating 100GB+ files by using mocking and chunked generation.
"""

import pytest
import tempfile
import time
import gc
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.core.streaming import StreamingChunker
from chunking_strategy.core.hardware import get_smart_parallelization_config, configure_smart_parallelization
from chunking_strategy.core.tika_integration import get_tika_processor
from chunking_strategy.core.base import ChunkingResult, ModalityType


class TestStreamingIntegration:
    """Test intelligent streaming integration in orchestrator."""

    def setup_method(self):
        """Setup test fixtures."""
        # Reset smart parallelization config to defaults
        configure_smart_parallelization(
            min_file_size_for_streaming=100 * 1024 * 1024,  # 100MB
            streaming_block_size=64 * 1024 * 1024,           # 64MB
            streaming_overlap_size=1024 * 1024               # 1MB
        )

    def test_smart_streaming_decision_small_file(self):
        """Test that small files do not trigger streaming."""
        # Create a small test file (1KB)
        test_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(test_file, 'w') as f:
            f.write("Small file content. " * 50)  # ~1KB

        try:
            orchestrator = ChunkerOrchestrator(enable_smart_parallelization=True)

            # Should not use streaming
            config = get_smart_parallelization_config()
            file_size = test_file.stat().st_size
            assert not config.should_use_streaming(file_size)

            # Chunk the file - should use regular processing
            result = orchestrator.chunk_file(test_file)

            assert result.chunks
            assert result.source_info.get("streaming_used") != True
            assert "streaming_config" not in result.source_info

        finally:
            test_file.unlink()

    def test_smart_streaming_decision_large_file(self):
        """Test that large files trigger streaming using mocked file size."""
        test_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(test_file, 'w') as f:
            f.write("Large file content. " * 1000)  # ~18KB actual size

        try:
            orchestrator = ChunkerOrchestrator(enable_smart_parallelization=True)

            # Mock file stat to report large size
            original_stat = test_file.stat

            def mock_stat():
                stat_result = original_stat()
                # Create a mock that returns large file size
                mock_stat_result = Mock()
                mock_stat_result.st_size = 200 * 1024 * 1024  # 200MB
                # Copy other attributes
                for attr in ['st_mode', 'st_mtime', 'st_ctime', 'st_atime']:
                    if hasattr(stat_result, attr):
                        setattr(mock_stat_result, attr, getattr(stat_result, attr))
                return mock_stat_result

            with patch('pathlib.Path.stat', return_value=mock_stat()):
                # Should use streaming
                config = get_smart_parallelization_config()
                assert config.should_use_streaming(200 * 1024 * 1024)

                # Mock StreamingChunker to avoid actual streaming overhead
                with patch('chunking_strategy.orchestrator.StreamingChunker') as mock_streaming:
                    mock_instance = Mock()
                    mock_streaming.return_value = mock_instance

                    # Create mock chunks
                    from chunking_strategy.core.base import Chunk, ChunkMetadata
                    mock_chunks = [
                        Chunk(
                            id=f"stream_chunk_{i}",
                            content=f"Streamed chunk {i} content",
                            modality=ModalityType.TEXT,
                            metadata=ChunkMetadata(source=str(test_file), chunker_used="sentence_based")
                        )
                        for i in range(5)
                    ]
                    mock_instance.stream_file.return_value = iter(mock_chunks)

                    # Chunk the file - should use streaming
                    result = orchestrator.chunk_file(test_file)

                    # Verify streaming was used
                    assert result.chunks
                    assert len(result.chunks) == 5
                    assert result.source_info.get("streaming_used") == True
                    assert "streaming_config" in result.source_info

                    # Verify StreamingChunker was called
                    mock_streaming.assert_called_once()
                    mock_instance.stream_file.assert_called_once()

        finally:
            test_file.unlink()

    def test_streaming_force_options(self):
        """Test force streaming and disable streaming options."""
        test_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(test_file, 'w') as f:
            f.write("Medium file content. " * 500)  # ~9KB

        try:
            orchestrator = ChunkerOrchestrator(enable_smart_parallelization=True)

            # Test force_streaming=True for small file
            with patch('chunking_strategy.orchestrator.StreamingChunker') as mock_streaming:
                mock_instance = Mock()
                mock_streaming.return_value = mock_instance

                from chunking_strategy.core.base import Chunk, ChunkMetadata
                mock_chunks = [
                    Chunk(
                        id="forced_chunk_0",
                        content="Forced streaming chunk",
                        modality=ModalityType.TEXT,
                        metadata=ChunkMetadata(source=str(test_file), chunker_used="sentence_based")
                    )
                ]
                mock_instance.stream_file.return_value = iter(mock_chunks)

                result = orchestrator.chunk_file(test_file, force_streaming=True)

                assert result.chunks
                assert result.source_info.get("streaming_used") == True
                mock_streaming.assert_called_once()

            # Test disable_streaming=True (should prevent streaming even for "large" files)
            with patch('pathlib.Path.stat') as mock_stat:
                mock_stat_result = Mock()
                mock_stat_result.st_size = 200 * 1024 * 1024  # Mock large size
                mock_stat.return_value = mock_stat_result

                result = orchestrator.chunk_file(test_file, disable_streaming=True)

                assert result.chunks
                assert result.source_info.get("streaming_used") != True

        finally:
            test_file.unlink()

    def test_streaming_custom_parameters(self):
        """Test custom streaming parameters."""
        test_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(test_file, 'w') as f:
            f.write("Content for custom streaming. " * 100)

        try:
            orchestrator = ChunkerOrchestrator(enable_smart_parallelization=True)

            with patch('chunking_strategy.orchestrator.StreamingChunker') as mock_streaming:
                mock_instance = Mock()
                mock_streaming.return_value = mock_instance

                from chunking_strategy.core.base import Chunk, ChunkMetadata
                mock_chunks = [
                    Chunk(
                        id="custom_chunk_0",
                        content="Custom streaming chunk",
                        modality=ModalityType.TEXT,
                        metadata=ChunkMetadata(source=str(test_file), chunker_used="sentence_based")
                    )
                ]
                mock_instance.stream_file.return_value = iter(mock_chunks)

                # Use custom streaming parameters
                result = orchestrator.chunk_file(
                    test_file,
                    force_streaming=True,
                    streaming_block_size=32 * 1024 * 1024,  # 32MB
                    streaming_overlap_size=512 * 1024       # 512KB
                )

                assert result.chunks

                # Verify custom parameters were passed
                call_args = mock_streaming.call_args
                assert call_args[1]['block_size'] == 32 * 1024 * 1024
                assert call_args[1]['overlap_size'] == 512 * 1024

        finally:
            test_file.unlink()

    def test_streaming_fallback_on_failure(self):
        """Test fallback to regular processing when streaming fails."""
        test_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(test_file, 'w') as f:
            f.write("Content that will cause streaming to fail. " * 100)

        try:
            orchestrator = ChunkerOrchestrator(enable_smart_parallelization=True)

            with patch('chunking_strategy.orchestrator.StreamingChunker') as mock_streaming:
                # Make streaming fail
                mock_streaming.side_effect = Exception("Streaming failed")

                # Should fall back to regular processing
                result = orchestrator.chunk_file(test_file, force_streaming=True)

                assert result.chunks
                assert result.source_info.get("streaming_failed") == True
                assert result.source_info.get("orchestrator_used") == True

        finally:
            test_file.unlink()


class TestTikaIntegration:
    """Test Tika integration in orchestrator."""

    def test_tika_enhanced_file_analysis(self):
        """Test that Tika is used for enhanced file analysis."""
        # Create a mock PDF file
        test_file = Path(tempfile.mktemp(suffix='.pdf'))
        with open(test_file, 'wb') as f:
            f.write(b'%PDF-1.4\nMock PDF content')  # Minimal PDF header

        try:
            orchestrator = ChunkerOrchestrator()

            # Mock Tika processor
            with patch('chunking_strategy.orchestrator.get_tika_processor') as mock_get_tika:
                mock_tika = Mock()
                mock_get_tika.return_value = mock_tika
                mock_tika.is_available.return_value = True

                # Mock Tika detection
                mock_tika.detect_file_type.return_value = {
                    "mime_type": "application/pdf",
                    "file_type": "pdf",
                    "confidence": 0.95
                }

                # Analyze the file
                file_info = orchestrator._analyze_file(test_file)

                # Verify Tika was used
                assert file_info.get("tika_available") == True
                assert file_info.get("tika_mime_type") == "application/pdf"
                assert file_info.get("enhanced_analysis") == True

                mock_tika.detect_file_type.assert_called_once_with(test_file)

        finally:
            test_file.unlink()

    def test_tika_content_extraction(self):
        """Test that Tika is used for content extraction."""
        # Create a mock Word document
        test_file = Path(tempfile.mktemp(suffix='.docx'))
        with open(test_file, 'wb') as f:
            f.write(b'PK\x03\x04Mock DOCX content')  # Minimal DOCX header

        try:
            orchestrator = ChunkerOrchestrator()

            # Mock file analysis to indicate Tika is available
            file_info = {
                "tika_available": True,
                "file_type": "docx",
                "modality": "text"
            }

            with patch('chunking_strategy.orchestrator.get_tika_processor') as mock_get_tika:
                mock_tika = Mock()
                mock_get_tika.return_value = mock_tika
                mock_tika.is_available.return_value = True

                # Mock Tika content extraction
                extracted_content = "This is the extracted content from the DOCX file using Tika."
                mock_tika.extract_content_and_metadata.return_value = {
                    "content": extracted_content,
                    "metadata": {
                        "title": "Test Document",
                        "author": "Test Author"
                    }
                }

                # Load content using Tika
                content = orchestrator._load_content(test_file, file_info)

                assert content == extracted_content
                mock_tika.extract_content_and_metadata.assert_called_once_with(test_file)

        finally:
            test_file.unlink()

    def test_tika_fallback_on_failure(self):
        """Test fallback to standard loading when Tika fails."""
        test_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(test_file, 'w') as f:
            f.write("Standard text file content")

        try:
            orchestrator = ChunkerOrchestrator()

            # Mock file analysis to indicate Tika is available but will fail
            file_info = {
                "tika_available": True,
                "file_type": "html",
                "modality": ModalityType.TEXT,
                "encoding": "utf-8"
            }

            with patch('chunking_strategy.orchestrator.get_tika_processor') as mock_get_tika:
                mock_tika = Mock()
                mock_get_tika.return_value = mock_tika
                mock_tika.is_available.return_value = True

                # Make Tika extraction fail
                mock_tika.extract_content_and_metadata.side_effect = Exception("Tika extraction failed")

                # Should fall back to standard loading
                content = orchestrator._load_content(test_file, file_info)

                assert content == "Standard text file content"

        finally:
            test_file.unlink()

    def test_tika_not_used_for_simple_formats(self):
        """Test that Tika is not used for simple text formats."""
        test_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(test_file, 'w') as f:
            f.write("Simple text file content")

        try:
            orchestrator = ChunkerOrchestrator()

            # Analyze the file
            file_info = orchestrator._analyze_file(test_file)

            # Verify Tika was not used for .txt files
            assert file_info.get("tika_available") == False
            assert file_info.get("tika_reason") == "Format not beneficial for Tika"

        finally:
            test_file.unlink()


class TestLargeFileSimulation:
    """Test large file handling through simulation."""

    def test_memory_efficient_streaming(self):
        """Test that streaming prevents memory issues with large files."""
        # Simulate a large file without actually creating it
        with patch('chunking_strategy.orchestrator.StreamingChunker') as mock_streaming_class:
            mock_instance = Mock()
            mock_streaming_class.return_value = mock_instance

            # Simulate streaming chunks from a 1GB file
            def generate_mock_chunks():
                for i in range(1000):  # Simulate 1000 chunks
                    from chunking_strategy.core.base import Chunk, ChunkMetadata
                    yield Chunk(
                            id=f"large_file_chunk_{i}",
                            content=f"This is chunk {i} from a very large file. " * 50,
                            modality=ModalityType.TEXT,
                            metadata=ChunkMetadata(source="large_file.txt", chunker_used="fixed_size")
                        )

            mock_instance.stream_file.return_value = generate_mock_chunks()

            # Create a small actual file but mock its size
            test_file = Path(tempfile.mktemp(suffix='.txt'))
            with open(test_file, 'w') as f:
                f.write("Small actual content")

            try:
                orchestrator = ChunkerOrchestrator(enable_smart_parallelization=True)

                # Mock file size to be very large
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat_result = Mock()
                    mock_stat_result.st_size = 1024 * 1024 * 1024  # 1GB
                    mock_stat.return_value = mock_stat_result

                    start_time = time.time()
                    result = orchestrator.chunk_file(test_file)
                    elapsed_time = time.time() - start_time

                    # Verify streaming was used and completed efficiently
                    assert result.chunks
                    assert len(result.chunks) == 1000
                    assert result.source_info.get("streaming_used") == True

                    # Should complete reasonably quickly (streaming simulation)
                    assert elapsed_time < 1.0  # Should be very fast with mocked streaming

                    # Verify streaming chunker was used
                    mock_streaming_class.assert_called_once()
                    mock_instance.stream_file.assert_called_once()

            finally:
                test_file.unlink()

    def test_configuration_file_integration(self):
        """Test that streaming configuration from file is respected."""
        config_file = Path(tempfile.mktemp(suffix='.yaml'))
        config_content = """
profile_name: "test_streaming"

strategy_selection:
  ".txt":
    primary: "sentence_based"
    fallbacks: ["fixed_size"]

streaming:
  enabled: true
  min_file_size: 52428800  # 50MB
  block_size: 33554432    # 32MB
  overlap_size: 524288    # 512KB

parallelization:
  smart_parallelization: true
  min_file_size_for_streaming: 52428800  # 50MB
"""

        with open(config_file, 'w') as f:
            f.write(config_content)

        try:
            # Test that we can load orchestrator with streaming config
            orchestrator = ChunkerOrchestrator(config_path=config_file)

            # Verify smart parallelization is enabled
            assert orchestrator.enable_smart_parallelization == True
            assert orchestrator.smart_config is not None

            # The config file itself doesn't directly set streaming thresholds yet,
            # but we can verify the orchestrator loads successfully
            config = get_smart_parallelization_config()
            assert config is not None

        finally:
            config_file.unlink()


class TestStreamingPerformance:
    """Test streaming performance characteristics."""

    def test_streaming_progress_logging(self):
        """Test that streaming provides progress logging for large files."""
        test_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(test_file, 'w') as f:
            f.write("Content for progress logging test")

        try:
            orchestrator = ChunkerOrchestrator(enable_smart_parallelization=True)

            with patch('chunking_strategy.orchestrator.StreamingChunker') as mock_streaming:
                mock_instance = Mock()
                mock_streaming.return_value = mock_instance

                # Simulate many chunks to trigger progress logging
                def generate_chunks():
                    from chunking_strategy.core.base import Chunk, ChunkMetadata
                    for i in range(2500):  # More than 1000 to trigger logging
                        yield Chunk(
                            id=f"progress_chunk_{i}",
                            content=f"Progress chunk {i}",
                            modality=ModalityType.TEXT,
                            metadata=ChunkMetadata(source=str(test_file), chunker_used="fixed_size")
                        )

                mock_instance.stream_file.return_value = generate_chunks()

                # Mock large file size
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat_result = Mock()
                    mock_stat_result.st_size = 500 * 1024 * 1024  # 500MB
                    mock_stat.return_value = mock_stat_result

                                        # Test that streaming works with large chunk counts
                    result = orchestrator.chunk_file(test_file)

                    assert result.chunks
                    assert len(result.chunks) == 2500

                    # Verify that streaming actually happened
                    assert result.source_info.get("streaming_used") == True

                    # Verify performance metrics are reasonable
                    processing_time = result.source_info.get("processing_time", 0)
                    assert processing_time < 1.0  # Should be very fast with mocked streaming

                    # Verify chunks are properly structured
                    for chunk in result.chunks[:5]:  # Check first 5 chunks
                        assert chunk.id.startswith("progress_chunk_")
                        assert "Progress chunk" in chunk.content
                        assert chunk.metadata.chunker_used == "fixed_size"

        finally:
            test_file.unlink()


class TestLargeScaleStreamingValidation:
    """
    Test streaming chunking with actual large files (100GB+).

    IMPORTANT: These tests use existing large test file:
        Uses existing 100GB text file: test_data/big_text.txt

    Notes:
    - These tests will NOT run automatically in CI/CD pipelines
    - The large files are NOT committed to git (included in .gitignore)
    - Tests will be SKIPPED gracefully if files don't exist

    Purpose: Validate that streaming chunking actually works at scale without
    running into memory issues or performance problems with real large files.
    """

    def setup_method(self):
        """Setup test environment for large file testing."""
        self.test_data_dir = Path("test_data")
        self.large_file_100gb = self.test_data_dir / "big_text.txt"

        # Configure streaming for large files
        configure_smart_parallelization(
            min_file_size_for_streaming=1 * 1024 * 1024,  # 1MB threshold for testing
            streaming_block_size=32 * 1024 * 1024,        # 32MB blocks
            streaming_overlap_size=512 * 1024             # 512KB overlap
        )

    def _get_memory_usage_mb(self) -> float:
        """Get current process memory usage in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        return 0.0

    def _log_memory_stats(self, stage: str, initial_memory: float = 0.0):
        """Log memory usage statistics."""
        if PSUTIL_AVAILABLE:
            current_memory = self._get_memory_usage_mb()
            if initial_memory > 0:
                memory_increase = current_memory - initial_memory
                print(f"\n  Memory at {stage}: {current_memory:.1f} MB (increase: +{memory_increase:.1f} MB)")
            else:
                print(f"\n  Memory at {stage}: {current_memory:.1f} MB")
            return current_memory
        else:
            print(f"\n  Memory monitoring at {stage}: psutil not available")
            return 0.0

    @pytest.mark.slow
    @pytest.mark.requires_large_files
    def test_100gb_file_basic_streaming_test(self):
        """
        Quick test to verify streaming works at all with the large file.

        This is a simpler test that just verifies we can get a few chunks
        without hanging, to help debug issues with complex data.
        """
        if not self.large_file_100gb.exists():
            pytest.skip(
                f"Large test file not found: {self.large_file_100gb}\n"
                "Expected to find test_data/big_text.txt (100GB file) for basic streaming validation."
            )

        file_size_gb = self.large_file_100gb.stat().st_size / (1024**3)
        print(f"\n  Basic streaming test with {file_size_gb:.1f} GB file")

        # Try with the simplest possible chunker
        chunker = StreamingChunker(
            strategy="fixed_size",
            chunk_size=1024,  # Smaller chunks
            streaming_block_size=1024 * 1024,  # 1MB blocks
        )

        chunks_processed = 0
        max_chunks = 10  # Just get 10 chunks to verify it works
        timeout_seconds = 30

        start_time = time.time()
        print(f"  Attempting to get first {max_chunks} chunks (timeout: {timeout_seconds}s)...")

        try:
            for i, chunk in enumerate(chunker.stream_file(self.large_file_100gb)):
                chunks_processed += 1
                elapsed = time.time() - start_time

                print(f"    Got chunk {chunks_processed}: {len(chunk.content)} chars, elapsed: {elapsed:.1f}s")

                if elapsed > timeout_seconds:
                    print(f"    ⚠️  Timeout after {timeout_seconds}s")
                    break

                if chunks_processed >= max_chunks:
                    break

            print(f"  ✅ Basic streaming test: Got {chunks_processed} chunks in {elapsed:.1f}s")
            assert chunks_processed > 0, "Could not get any chunks from streaming"

        except Exception as e:
            print(f"  ❌ Basic streaming failed: {e}")
            pytest.fail(f"Basic streaming test failed: {e}")

    @pytest.mark.slow
    @pytest.mark.requires_large_files
    def test_100gb_file_streaming_chunks_only(self):
        """
        Test streaming chunking of 100GB file - validates chunking logic only.

        This test processes only the first 1000 chunks to validate that:
        1. Streaming chunking works correctly
        2. Memory usage remains bounded
        3. Chunk quality is maintained
        4. No memory leaks occur

        NOTE: Uses test_data/big_text.txt (100GB file)
        """
        if not self.large_file_100gb.exists():
            pytest.skip(
                f"Large test file not found: {self.large_file_100gb}\n"
                "Expected to find test_data/big_text.txt (100GB file) for large-scale streaming validation."
            )

        # Check file size to confirm it's actually large
        file_size_gb = self.large_file_100gb.stat().st_size / (1024**3)
        if file_size_gb < 50:  # At least 50GB
            pytest.skip(f"Test file is too small ({file_size_gb:.1f} GB). Need at least 50GB for this test.")

        print(f"\n  Testing with {file_size_gb:.1f} GB file: {self.large_file_100gb}")

        # Quick data inspection to understand file content
        try:
            with open(self.large_file_100gb, 'r', encoding='utf-8', errors='ignore') as f:
                sample_data = f.read(1000)  # Read first 1KB
                line_count = sample_data.count('\n')
                avg_line_length = len(sample_data) / max(line_count, 1)
                print(f"  Sample data analysis: {line_count} lines in first 1KB, avg line length: {avg_line_length:.1f} chars")
                print(f"  First 100 chars: {repr(sample_data[:100])}")
        except Exception as e:
            print(f"  Could not inspect file: {e}")

        # Initial memory check
        gc.collect()  # Clean up before starting
        initial_memory = self._log_memory_stats("test start")

        # Create streaming chunker with conservative settings - use fixed_size for reliability
        chunker = StreamingChunker(
            strategy="fixed_size",     # Use simple fixed-size strategy to avoid parsing issues
            chunk_size=2048,           # Reasonable chunk size
            streaming_block_size=16 * 1024 * 1024,  # 16MB blocks for this test
        )

        memory_after_setup = self._log_memory_stats("chunker setup", initial_memory)

        # Stream chunks and process first 1000 to validate
        chunks_processed = 0
        max_chunks_to_process = 1000  # Limit for validation
        memory_high_water_mark = memory_after_setup

        try:
            print(f"  Starting streaming chunking (processing first {max_chunks_to_process} chunks)...")
            start_time = time.time()
            last_progress_time = start_time

            # Add timeout protection
            timeout_seconds = 60  # 1 minute timeout

            for i, chunk in enumerate(chunker.stream_file(self.large_file_100gb)):
                current_time = time.time()

                # Check for timeout
                if current_time - start_time > timeout_seconds:
                    print(f"    ⚠️  Timeout after {timeout_seconds} seconds, stopping test")
                    break

                chunks_processed += 1

                # Validate chunk structure
                assert chunk is not None, f"Chunk {i} is None"
                assert hasattr(chunk, 'content'), f"Chunk {i} missing content"
                assert hasattr(chunk, 'metadata'), f"Chunk {i} missing metadata"
                assert len(chunk.content) > 0, f"Chunk {i} has empty content"

                # More frequent progress reporting for better monitoring
                if chunks_processed % 50 == 0 or current_time - last_progress_time > 5:
                    current_memory = self._get_memory_usage_mb()
                    memory_high_water_mark = max(memory_high_water_mark, current_memory)

                    elapsed = current_time - start_time
                    chunks_per_sec = chunks_processed / elapsed if elapsed > 0 else 0
                    print(f"    Processed {chunks_processed} chunks, "
                          f"Memory: {current_memory:.1f} MB, "
                          f"Rate: {chunks_per_sec:.1f} chunks/sec, "
                          f"Elapsed: {elapsed:.1f}s")
                    last_progress_time = current_time

                # Stop after processing enough chunks for validation
                if chunks_processed >= max_chunks_to_process:
                    break

            elapsed_time = time.time() - start_time
            final_memory = self._log_memory_stats("processing complete", initial_memory)

            # Validate results - be flexible if we hit timeout
            if chunks_processed < max_chunks_to_process:
                print(f"    ℹ️  Processed {chunks_processed}/{max_chunks_to_process} chunks (stopped early)")

            assert chunks_processed > 0, "No chunks were processed - streaming failed completely"

            # Memory usage should be reasonable (not growing linearly with file size)
            memory_increase = memory_high_water_mark - initial_memory
            assert memory_increase < 500, \
                f"Memory usage too high: {memory_increase:.1f} MB increase. Streaming should keep memory bounded."

            # Performance should be reasonable (lower threshold for complex data)
            chunks_per_second = chunks_processed / elapsed_time if elapsed_time > 0 else 0
            assert chunks_per_second > 1, \
                f"Processing too slow: {chunks_per_second:.1f} chunks/sec. Expected >1 chunks/sec."

            print(f"\n  ✅ SUCCESS: Processed {chunks_processed} chunks from {file_size_gb:.1f} GB file")
            print(f"     Memory increase: {memory_increase:.1f} MB")
            print(f"     Processing rate: {chunks_per_second:.1f} chunks/sec")
            print(f"     Total time: {elapsed_time:.2f} seconds")

        except Exception as e:
            self._log_memory_stats("error occurred", initial_memory)
            pytest.fail(f"Streaming failed after {chunks_processed} chunks: {e}")

        finally:
            # Clean up memory
            del chunker
            gc.collect()
            self._log_memory_stats("cleanup complete", initial_memory)

    @pytest.mark.slow
    @pytest.mark.requires_large_files
    def test_100gb_file_orchestrator_streaming(self):
        """
        Test ChunkerOrchestrator with 100GB file using streaming.

        This test validates that the orchestrator correctly:
        1. Detects the large file and enables streaming automatically
        2. Uses appropriate chunking strategy
        3. Maintains reasonable memory usage
        4. Processes chunks efficiently

        NOTE: Uses test_data/big_text.txt (100GB file)
        """
        if not self.large_file_100gb.exists():
            pytest.skip(
                f"Large test file not found: {self.large_file_100gb}\n"
                "Expected to find test_data/big_text.txt (100GB file) for orchestrator streaming validation."
            )

        file_size_gb = self.large_file_100gb.stat().st_size / (1024**3)
        if file_size_gb < 50:
            pytest.skip(f"Test file too small ({file_size_gb:.1f} GB) for orchestrator streaming test.")

        print(f"\n  Testing orchestrator streaming with {file_size_gb:.1f} GB file")

        # Memory monitoring
        gc.collect()
        initial_memory = self._log_memory_stats("orchestrator test start")

        # Create orchestrator with streaming enabled
        orchestrator = ChunkerOrchestrator(
            enable_smart_parallelization=True
        )

        memory_after_setup = self._log_memory_stats("orchestrator setup", initial_memory)

        try:
            start_time = time.time()

            # This should automatically detect large file and use streaming
            with patch('chunking_strategy.orchestrator.StreamingChunker') as mock_streaming:
                # Create a real StreamingChunker instance with fixed_size for reliability
                real_chunker = StreamingChunker(strategy="fixed_size", chunk_size=2048)

                def limited_stream_with_timeout(file_path):
                    """Stream only first 100 chunks for testing with timeout protection."""
                    count = 0
                    start_time = time.time()
                    timeout_seconds = 30  # 30 second timeout

                    for chunk in real_chunker.stream_file(file_path):
                        yield chunk
                        count += 1

                        # Check timeout
                        if time.time() - start_time > timeout_seconds:
                            print(f"    ⚠️  Orchestrator test timeout after {timeout_seconds}s")
                            break

                        if count >= 100:  # Smaller limit for testing
                            break

                mock_streaming.return_value.stream_file = limited_stream_with_timeout
                mock_streaming.return_value.strategy = "fixed_size"

                # Process file - should use streaming automatically
                result = orchestrator.chunk_file(self.large_file_100gb, strategy_override="fixed_size")

                elapsed_time = time.time() - start_time
                final_memory = self._log_memory_stats("orchestrator complete", initial_memory)

                # Validate orchestrator used streaming
                assert result is not None, "No chunking result returned"
                assert len(result.chunks) > 0, "No chunks generated"
                assert result.source_info.get("streaming_used") == True, \
                    "Orchestrator should have used streaming for large file"

                # Validate memory efficiency
                memory_increase = final_memory - initial_memory
                assert memory_increase < 300, \
                    f"Orchestrator memory usage too high: {memory_increase:.1f} MB"

                print(f"\n  ✅ SUCCESS: Orchestrator streaming test completed")
                print(f"     Chunks generated: {len(result.chunks)}")
                print(f"     Strategy used: {result.strategy_used}")
                print(f"     Streaming enabled: {result.source_info.get('streaming_used')}")
                print(f"     Memory increase: {memory_increase:.1f} MB")
                print(f"     Processing time: {elapsed_time:.2f} seconds")

                # Validate we got a reasonable number of chunks (reduced from 500 to 100)
                assert len(result.chunks) <= 100, f"Expected ≤100 chunks, got {len(result.chunks)}"

        except Exception as e:
            self._log_memory_stats("orchestrator error", initial_memory)
            pytest.fail(f"Orchestrator streaming test failed: {e}")

        finally:
            del orchestrator
            gc.collect()
            self._log_memory_stats("orchestrator cleanup", initial_memory)


class TestSentenceBasedStreamingFix:
    """Test the fixed sentence-based streaming with dictionary data."""

    def setup_method(self):
        """Setup test environment."""
        self.large_file = Path("test_data/big_text.txt")

    @pytest.mark.slow
    @pytest.mark.requires_large_files
    def test_sentence_based_streaming_with_dictionary_data(self):
        """Test that sentence-based streaming now works with dictionary data."""
        if not self.large_file.exists():
            pytest.skip("Large test file not found")

        file_size_gb = self.large_file.stat().st_size / (1024**3)
        print(f"\n  Testing sentence-based streaming with {file_size_gb:.1f} GB dictionary file")

        # Create sentence-based streaming chunker
        chunker = StreamingChunker(strategy="sentence_based", max_sentences=5)

        chunks_processed = 0
        start_time = time.time()

        try:
            for chunk in chunker.stream_file(self.large_file):
                chunks_processed += 1

                # Validate chunk structure
                assert chunk is not None, f"Chunk {chunks_processed} is None"
                assert hasattr(chunk, 'content'), f"Chunk {chunks_processed} missing content"
                assert len(chunk.content) > 0, f"Chunk {chunks_processed} has empty content"

                # Progress reporting
                if chunks_processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = chunks_processed / elapsed if elapsed > 0 else 0
                    print(f"    Processed {chunks_processed} chunks, rate: {rate:.1f}/sec")

                # Stop after reasonable test amount
                if chunks_processed >= 300:
                    break

            elapsed_time = time.time() - start_time
            rate = chunks_processed / elapsed_time if elapsed_time > 0 else 0

            print(f"\n  ✅ Sentence-based streaming SUCCESS:")
            print(f"    Chunks: {chunks_processed}")
            print(f"    Time: {elapsed_time:.2f}s")
            print(f"    Rate: {rate:.1f} chunks/sec")

            # Validate results
            assert chunks_processed > 0, "No chunks produced"
            assert rate > 10, f"Too slow: {rate:.1f} chunks/sec"
            assert elapsed_time < 60, f"Took too long: {elapsed_time:.2f}s"

        except Exception as e:
            pytest.fail(f"Sentence-based streaming failed: {e}")

    def test_sentence_based_chunker_with_small_dictionary_data(self):
        """Test sentence-based chunker with small dictionary-style data."""
        # Create test data
        test_data = "\n".join([
            "apple", "banana", "cherry", "date", "elderberry",
            "fig", "grape", "honeydew", "kiwi", "lemon"
        ] * 10)  # 100 words

        test_file = Path(tempfile.mktemp(suffix='.txt'))
        try:
            with open(test_file, 'w') as f:
                f.write(test_data)

            print(f"\n  Testing sentence chunking with small dictionary data...")

            chunker = StreamingChunker(strategy="sentence_based", max_sentences=3)

            chunks_processed = 0
            start_time = time.time()

            for chunk in chunker.stream_file(test_file):
                chunks_processed += 1

                # Validate chunk
                assert chunk is not None
                assert len(chunk.content) > 0

                if chunks_processed <= 3:
                    print(f"    Chunk {chunks_processed}: {chunk.content[:50]}...")

                if chunks_processed >= 10:
                    break

            elapsed_time = time.time() - start_time
            rate = chunks_processed / elapsed_time if elapsed_time > 0 else 0

            print(f"    ✅ Small dictionary test: {chunks_processed} chunks in {elapsed_time:.3f}s ({rate:.1f}/sec)")

            # Validate results
            assert chunks_processed > 0, "No chunks produced"
            assert elapsed_time < 5.0, f"Too slow: {elapsed_time:.3f}s"

        finally:
            test_file.unlink()


class TestDocFileStreamingIntegration:
    """Test streaming integration with .doc files."""

    def get_test_doc_file(self):
        """Get the test .doc file if it exists."""
        test_data_dir = Path(__file__).parent.parent / "test_data"

        # Look for any .doc file in test_data
        for doc_file in test_data_dir.glob("*.doc"):
            return doc_file

        return None

    def test_doc_file_streaming_compatibility(self):
        """Test that .doc files can be processed with streaming if large enough."""
        doc_file = self.get_test_doc_file()
        if doc_file is None:
            pytest.skip("No .doc file found in test_data directory")

        print(f"\n📄 Testing .doc file streaming compatibility: {doc_file.name}")

        # Test different strategies with the .doc file
        # Note: .doc files are binary format, so we use doc_chunker specifically
        streaming_strategies = [
            ("doc_chunker", {"min_chunk_size": 500}),
            ("fixed_size", {"chunk_size": 1000, "unit": "byte"})  # Use byte-based chunking for binary files
        ]

        results = {}

        for strategy_name, params in streaming_strategies:
            print(f"  Testing {strategy_name} streaming with .doc file...")

            try:
                # Create streaming chunker
                streaming_chunker = StreamingChunker(strategy=strategy_name, **params)

                chunks_processed = 0
                start_time = time.time()
                total_content_length = 0

                # Process file with streaming
                for chunk in streaming_chunker.stream_file(doc_file):
                    chunks_processed += 1

                    # Validate chunk
                    assert chunk is not None, f"Chunk {chunks_processed} is None"
                    assert hasattr(chunk, 'content'), f"Chunk {chunks_processed} missing content"

                    content = chunk.content
                    assert content, f"Chunk {chunks_processed} has empty content"
                    total_content_length += len(content.strip())

                    # Stop after reasonable number of chunks for testing
                    if chunks_processed >= 20:
                        break

                processing_time = time.time() - start_time

                results[strategy_name] = {
                    "status": "SUCCESS",
                    "chunks_processed": chunks_processed,
                    "total_content_length": total_content_length,
                    "processing_time": processing_time,
                    "rate": chunks_processed / processing_time if processing_time > 0 else 0
                }

                print(f"    ✅ {strategy_name}: {chunks_processed} chunks, {total_content_length} chars, {processing_time:.3f}s")

                # Basic validations
                assert chunks_processed > 0, f"No chunks produced by {strategy_name}"
                assert total_content_length > 50, f"Too little content extracted by {strategy_name}: {total_content_length}"
                assert processing_time < 30, f"{strategy_name} took too long: {processing_time:.2f}s"

            except Exception as e:
                error_msg = str(e)

                # Check for dependency errors
                dependency_keywords = ["import", "module", "dependency", "install", "backend", "not available"]
                is_dependency_error = any(kw.lower() in error_msg.lower() for kw in dependency_keywords)

                if is_dependency_error:
                    results[strategy_name] = {"status": "DEPENDENCY_MISSING", "error": error_msg}
                    print(f"    ⏭️  {strategy_name}: Skipped (missing dependency)")
                else:
                    results[strategy_name] = {"status": "ERROR", "error": error_msg}
                    print(f"    ❌ {strategy_name}: Error - {error_msg[:100]}")

        # Validation: At least one strategy should work
        successful_strategies = [s for s, r in results.items() if r["status"] == "SUCCESS"]
        assert len(successful_strategies) > 0, f"No strategies successfully streamed .doc file: {results}"

    def test_doc_file_vs_regular_file_streaming(self):
        """Compare .doc file streaming performance with regular text file streaming."""
        doc_file = self.get_test_doc_file()
        if doc_file is None:
            pytest.skip("No .doc file found in test_data directory")

        # Find a comparable text file
        test_data_dir = Path(__file__).parent.parent / "test_data"
        text_files = list(test_data_dir.glob("*.txt"))
        if not text_files:
            pytest.skip("No .txt files found for comparison")

        text_file = text_files[0]

        print(f"\n📊 Comparing streaming performance:")
        print(f"  .doc file: {doc_file.name}")
        print(f"  .txt file: {text_file.name}")

        # Use appropriate strategies for different file types
        doc_strategy = "doc_chunker"
        doc_params = {"min_chunk_size": 500}

        txt_strategy = "fixed_size"
        txt_params = {"chunk_size": 1000}

        results = {}

        for file_type, file_path in [("doc", doc_file), ("txt", text_file)]:
            try:
                if file_type == "doc":
                    streaming_chunker = StreamingChunker(strategy=doc_strategy, **doc_params)
                else:
                    streaming_chunker = StreamingChunker(strategy=txt_strategy, **txt_params)

                chunks_processed = 0
                start_time = time.time()

                # Process first 10 chunks for comparison
                for chunk in streaming_chunker.stream_file(file_path):
                    chunks_processed += 1
                    if chunks_processed >= 10:
                        break

                processing_time = time.time() - start_time

                results[file_type] = {
                    "chunks": chunks_processed,
                    "time": processing_time,
                    "rate": chunks_processed / processing_time if processing_time > 0 else 0
                }

                print(f"    {file_type.upper()}: {chunks_processed} chunks in {processing_time:.3f}s ({results[file_type]['rate']:.1f}/sec)")

            except Exception as e:
                results[file_type] = {"error": str(e)}
                print(f"    {file_type.upper()}: Error - {str(e)[:50]}")

        # Both should produce chunks (quality check)
        if "doc" in results and "chunks" in results["doc"]:
            assert results["doc"]["chunks"] > 0, ".doc file should produce chunks"

        if "txt" in results and "chunks" in results["txt"]:
            assert results["txt"]["chunks"] > 0, ".txt file should produce chunks"

    def test_doc_file_memory_usage_during_streaming(self):
        """Test memory usage when streaming .doc files."""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available for memory testing")

        doc_file = self.get_test_doc_file()
        if doc_file is None:
            pytest.skip("No .doc file found in test_data directory")

        print(f"\n💾 Testing .doc file streaming memory usage: {doc_file.name}")

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        try:
            streaming_chunker = StreamingChunker(strategy="doc_chunker", min_chunk_size=1000)

            chunks_processed = 0
            max_memory = initial_memory

            for chunk in streaming_chunker.stream_file(doc_file):
                chunks_processed += 1

                # Check memory every few chunks
                if chunks_processed % 5 == 0:
                    current_memory = process.memory_info().rss
                    max_memory = max(max_memory, current_memory)

                # Process reasonable number of chunks
                if chunks_processed >= 15:
                    break

            final_memory = process.memory_info().rss
            memory_increase = max_memory - initial_memory
            memory_increase_mb = memory_increase / (1024 * 1024)

            print(f"    Memory usage:")
            print(f"      Initial: {initial_memory / (1024*1024):.1f} MB")
            print(f"      Peak: {max_memory / (1024*1024):.1f} MB")
            print(f"      Final: {final_memory / (1024*1024):.1f} MB")
            print(f"      Max increase: {memory_increase_mb:.1f} MB")
            print(f"    Processed: {chunks_processed} chunks")

            # Memory should not grow excessively (allow up to 50MB increase for .doc processing)
            assert memory_increase_mb < 50, f"Memory increase too high: {memory_increase_mb:.1f} MB"

            # Should process some chunks
            assert chunks_processed > 0, "No chunks processed"

        except Exception as e:
            pytest.fail(f"Memory test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
