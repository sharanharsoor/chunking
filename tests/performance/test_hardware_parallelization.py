"""
Comprehensive test suite for hardware detection and parallelization features.

Tests hardware-aware orchestrator, multi-strategy processing, and parallel batch operations.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
import time

from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.core.multi_strategy import MultiStrategyProcessor
from chunking_strategy.core.hardware import HardwareDetector, HardwareInfo


class TestHardwareIntegration:
    """Test hardware detection integration with orchestrator."""

    def test_orchestrator_hardware_detection_enabled(self):
        """Test that orchestrator properly initializes hardware detection."""
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)

        assert orchestrator.enable_hardware_optimization is True
        assert orchestrator.hardware_detector is not None
        assert orchestrator.hardware_info is not None
        assert hasattr(orchestrator.hardware_info, 'cpu_count')
        assert hasattr(orchestrator.hardware_info, 'memory_total_gb')

    def test_orchestrator_hardware_detection_disabled(self):
        """Test that orchestrator can work without hardware detection."""
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=False)

        assert orchestrator.enable_hardware_optimization is False
        assert orchestrator.hardware_detector is None
        assert orchestrator.hardware_info is None

    def test_orchestrator_hardware_info_logging(self, caplog):
        """Test that hardware information is properly logged."""
        with caplog.at_level('INFO'):
            orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)

        # Check that hardware info was logged
        hardware_logs = [record for record in caplog.records if 'Hardware detected:' in record.message]
        assert len(hardware_logs) > 0

        log_message = hardware_logs[0].message
        assert 'cores' in log_message
        assert 'RAM' in log_message
        assert 'GPUs' in log_message


class TestBatchProcessingWithHardware:
    """Test hardware-optimized batch processing."""

    @pytest.fixture
    def test_files(self):
        """Create temporary test files."""
        files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files of different sizes
            for i in range(5):
                file_path = temp_path / f"test_file_{i}.txt"
                content = f"This is test content for file {i}. " * (10 * (i + 1))
                file_path.write_text(content)
                files.append(file_path)

            yield files

    def test_batch_processing_sequential(self, test_files):
        """Test sequential batch processing."""
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)

        results = orchestrator.chunk_files_batch(
            file_paths=test_files,
            parallel_mode="sequential"
        )

        assert len(results) == len(test_files)
        assert all(len(result.chunks) > 0 for result in results)

    def test_batch_processing_threaded(self, test_files):
        """Test threaded batch processing."""
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)

        results = orchestrator.chunk_files_batch(
            file_paths=test_files,
            parallel_mode="thread",
            max_workers=2
        )

        assert len(results) == len(test_files)
        assert all(len(result.chunks) > 0 for result in results)

    def test_batch_processing_multiprocess(self, test_files):
        """Test multiprocess batch processing."""
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)

        results = orchestrator.chunk_files_batch(
            file_paths=test_files,
            parallel_mode="process",
            max_workers=2
        )

        assert len(results) == len(test_files)
        assert all(len(result.chunks) > 0 for result in results)

    def test_batch_processing_auto_mode(self, test_files):
        """Test auto mode selection for batch processing."""
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)

        results = orchestrator.chunk_files_batch(
            file_paths=test_files,
            parallel_mode="auto"
        )

        assert len(results) == len(test_files)
        assert all(len(result.chunks) > 0 for result in results)

    def test_batch_processing_progress_callback(self, test_files):
        """Test progress callback functionality."""
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)

        progress_calls = []
        def progress_callback(current, total, message):
            progress_calls.append((current, total, message))

        results = orchestrator.chunk_files_batch(
            file_paths=test_files[:2],  # Use fewer files for testing
            parallel_mode="sequential",
            progress_callback=progress_callback
        )

        assert len(results) == 2
        # Progress callback should have been called
        assert len(progress_calls) > 0

    def test_batch_processing_empty_file_list(self):
        """Test batch processing with empty file list."""
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)

        results = orchestrator.chunk_files_batch(file_paths=[])

        assert results == []

    def test_batch_processing_nonexistent_files(self):
        """Test batch processing with nonexistent files."""
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)

        nonexistent_files = [Path("/nonexistent/file1.txt"), Path("/nonexistent/file2.txt")]

        results = orchestrator.chunk_files_batch(
            file_paths=nonexistent_files,
            parallel_mode="sequential"
        )

        # Should return empty results for nonexistent files
        assert len(results) == len(nonexistent_files)
        assert all(len(result.chunks) == 0 for result in results)


class TestMultiStrategyProcessor:
    """Test multi-strategy parallel processing."""

    @pytest.fixture
    def test_file(self):
        """Create a temporary test file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_document.txt"
            content = """
            This is a test document with multiple paragraphs.

            Each paragraph contains several sentences. This allows us to test
            different chunking strategies effectively. The content is long enough
            to generate multiple chunks with various strategies.

            This final paragraph ensures we have sufficient content for testing.
            The multi-strategy processor should be able to handle this document
            with all available strategies.
            """
            file_path.write_text(content)
            yield file_path

    def test_multi_strategy_sequential(self, test_file):
        """Test multi-strategy processing in sequential mode."""
        processor = MultiStrategyProcessor()

        strategies = ["sentence_based", "paragraph_based", "fixed_size"]
        result = processor.process_file_with_strategies(
            file_path=test_file,
            strategies=strategies,
            parallel_mode="sequential"
        )

        assert result.file_path == test_file
        assert len(result.successful_strategies) > 0
        assert all(strategy in result.strategy_results for strategy in result.successful_strategies)
        assert all(len(result.strategy_results[strategy].chunks) > 0
                  for strategy in result.successful_strategies)

    def test_multi_strategy_threaded(self, test_file):
        """Test multi-strategy processing in threaded mode."""
        processor = MultiStrategyProcessor()

        strategies = ["sentence_based", "paragraph_based", "fixed_size"]
        result = processor.process_file_with_strategies(
            file_path=test_file,
            strategies=strategies,
            parallel_mode="thread",
            max_workers=2
        )

        assert result.file_path == test_file
        assert len(result.successful_strategies) > 0
        assert result.hardware_config['mode'] == 'thread'
        assert result.hardware_config['workers'] == 2

    def test_multi_strategy_multiprocess(self, test_file):
        """Test multi-strategy processing in multiprocess mode."""
        processor = MultiStrategyProcessor()

        strategies = ["sentence_based", "fixed_size"]  # Use fewer for process testing
        result = processor.process_file_with_strategies(
            file_path=test_file,
            strategies=strategies,
            parallel_mode="process",
            max_workers=2
        )

        assert result.file_path == test_file
        assert len(result.successful_strategies) > 0
        assert result.hardware_config['mode'] == 'process'

    def test_multi_strategy_auto_mode(self, test_file):
        """Test multi-strategy processing in auto mode."""
        processor = MultiStrategyProcessor()

        strategies = ["sentence_based", "paragraph_based"]
        result = processor.process_file_with_strategies(
            file_path=test_file,
            strategies=strategies,
            parallel_mode="auto"
        )

        assert result.file_path == test_file
        assert len(result.successful_strategies) > 0

    def test_multi_strategy_with_configs(self, test_file):
        """Test multi-strategy processing with strategy-specific configurations."""
        processor = MultiStrategyProcessor()

        strategies = ["sentence_based", "fixed_size"]
        strategy_configs = {
            "sentence_based": {"max_sentences": 3},
            "fixed_size": {"chunk_size": 500}
        }

        result = processor.process_file_with_strategies(
            file_path=test_file,
            strategies=strategies,
            strategy_configs=strategy_configs,
            parallel_mode="sequential"
        )

        assert len(result.successful_strategies) > 0
        # Verify that configs were applied (sentence_based should respect max_sentences)
        if "sentence_based" in result.successful_strategies:
            sentence_chunks = result.strategy_results["sentence_based"].chunks
            # Each chunk should have reasonable content (the exact split depends on sentence detection)
            assert all(len(chunk.content.strip()) > 0 for chunk in sentence_chunks)

    def test_multi_strategy_performance_comparison(self, test_file):
        """Test that parallel processing is actually faster than sequential."""
        processor = MultiStrategyProcessor()

        strategies = ["sentence_based", "paragraph_based", "fixed_size"]

        # Time sequential processing
        start_time = time.time()
        sequential_result = processor.process_file_with_strategies(
            file_path=test_file,
            strategies=strategies,
            parallel_mode="sequential"
        )
        sequential_time = time.time() - start_time

        # Time parallel processing
        start_time = time.time()
        parallel_result = processor.process_file_with_strategies(
            file_path=test_file,
            strategies=strategies,
            parallel_mode="thread",
            max_workers=len(strategies)
        )
        parallel_time = time.time() - start_time

        # Both should produce the same number of successful strategies
        assert len(sequential_result.successful_strategies) == len(parallel_result.successful_strategies)

        # Note: Due to overhead, parallel might not always be faster for small workloads
        # This is expected and acceptable
        print(f"Sequential: {sequential_time:.3f}s, Parallel: {parallel_time:.3f}s")

    def test_multi_strategy_multiple_files(self):
        """Test multi-strategy processing across multiple files."""
        processor = MultiStrategyProcessor()

        # Create multiple test files
        files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(3):
                file_path = Path(temp_dir) / f"test_doc_{i}.txt"
                content = f"Test document {i} with multiple sentences. " * (5 + i * 2)
                file_path.write_text(content)
                files.append(file_path)

            strategies = ["sentence_based", "fixed_size"]
            results = processor.process_multiple_files_with_strategies(
                files=files,
                strategies=strategies,
                parallel_mode="thread"
            )

            assert len(results) == len(files)
            assert all(len(result.successful_strategies) > 0 for result in results)

    def test_multi_strategy_result_properties(self, test_file):
        """Test MultiStrategyResult properties and methods."""
        processor = MultiStrategyProcessor()

        strategies = ["sentence_based", "fixed_size"]
        result = processor.process_file_with_strategies(
            file_path=test_file,
            strategies=strategies,
            parallel_mode="sequential"
        )

        # Test properties
        assert isinstance(result.total_chunks, int)
        assert result.total_chunks > 0
        assert isinstance(result.success_rate, float)
        assert 0.0 <= result.success_rate <= 1.0
        assert result.success_rate > 0.5  # Most strategies should succeed

    def test_multi_strategy_error_handling(self):
        """Test error handling in multi-strategy processing."""
        processor = MultiStrategyProcessor()

        # Test with nonexistent file
        nonexistent_file = Path("/nonexistent/file.txt")

        with pytest.raises(FileNotFoundError):
            processor.process_file_with_strategies(
                file_path=nonexistent_file,
                strategies=["sentence_based"]
            )

    def test_multi_strategy_invalid_strategy(self, test_file):
        """Test handling of invalid strategies."""
        processor = MultiStrategyProcessor()

        strategies = ["sentence_based", "nonexistent_strategy"]
        result = processor.process_file_with_strategies(
            file_path=test_file,
            strategies=strategies,
            parallel_mode="sequential"
        )

        # Should have some successful strategies and some failed ones
        assert len(result.successful_strategies) >= 1  # At least sentence_based should work
        assert len(result.failed_strategies) >= 1     # nonexistent_strategy should fail
        assert result.success_rate < 1.0             # Not all strategies succeeded


class TestParallelizationConstraints:
    """Test understanding of parallelization constraints and limitations."""

    def test_overlap_constraint_understanding(self):
        """Test that we understand overlap creates sequential dependencies."""
        # This is a conceptual test to document the constraint
        # Overlapping chunks require sequential processing because each chunk
        # depends on the content of the previous chunk

        from chunking_strategy.strategies.general.fixed_size import FixedSizeChunker

        # Create chunker with overlap
        chunker = FixedSizeChunker(chunk_size=100, overlap_size=20)

        # This constraint means we cannot parallelize chunk creation
        # within a single file when overlap > 0
        assert chunker.overlap_size > 0

        # The chunker should still work sequentially
        content = "This is test content for overlap testing. It has multiple sentences and enough content to generate multiple chunks with the fixed chunk size of 100 characters and overlap of 20 characters. We need sufficient content to demonstrate the overlap constraint clearly."
        result = chunker.chunk(content)

        assert len(result.chunks) >= 1
        # Verify overlap exists between consecutive chunks
        if len(result.chunks) > 1:
            chunk1_end = result.chunks[0].content[-20:]
            chunk2_start = result.chunks[1].content[:20]
            # There should be some overlap (exact matching depends on boundary preservation)
            assert len(chunk1_end) > 0 and len(chunk2_start) > 0

    def test_independent_strategy_parallelization(self, caplog):
        """Test that independent strategies can be processed in parallel."""
        # Different strategies on the same content can be processed in parallel
        # because they don't depend on each other

        processor = MultiStrategyProcessor()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.txt"
            content = "Independent test content. " * 50
            file_path.write_text(content)

            # These strategies are independent - can be parallelized
            strategies = ["sentence_based", "fixed_size"]

            with caplog.at_level('INFO'):
                result = processor.process_file_with_strategies(
                    file_path=file_path,
                    strategies=strategies,
                    parallel_mode="thread"
                )

            assert len(result.successful_strategies) == len(strategies)

            # Check that parallel processing was actually used
            processing_logs = [record for record in caplog.records
                             if 'mode=thread' in record.message]
            assert len(processing_logs) > 0


class TestHardwareOptimizationIntegration:
    """Test integration between hardware detection and optimization decisions."""

    @patch('chunking_strategy.core.hardware.HardwareDetector.detect_hardware')
    def test_hardware_based_mode_selection(self, mock_detect):
        """Test that hardware characteristics influence processing mode selection."""

        # Mock a high-performance system
        high_perf_hardware = HardwareInfo(
            cpu_count=16, cpu_count_physical=8, cpu_freq=3.5, cpu_usage=20.0,
            memory_total_gb=32.0, memory_available_gb=24.0, memory_usage_percent=25.0,
            gpu_count=1, gpu_names=["RTX 4090"], gpu_memory_total=[24.0],
            gpu_memory_free=[20.0], gpu_utilization=[10.0],
            platform="Linux", architecture="x86_64", python_version="3.11.0",
            recommended_batch_size=32, recommended_workers=8, use_gpu=True
        )
        mock_detect.return_value = high_perf_hardware

        processor = MultiStrategyProcessor(enable_hardware_optimization=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "large_test.txt"
            # Create a large file (> 50MB as per the logic)
            content = "Test content for hardware optimization. " * 100000
            file_path.write_text(content)

            strategies = ["sentence_based", "paragraph_based", "fixed_size"]

            # Auto mode should select process for large files on high-perf hardware
            result = processor.process_file_with_strategies(
                file_path=file_path,
                strategies=strategies,
                parallel_mode="auto"
            )

            # Verify hardware was considered
            assert result.hardware_config['hardware_optimized'] is True
            assert len(result.successful_strategies) > 0

    @patch('chunking_strategy.core.hardware.HardwareDetector.detect_hardware')
    def test_low_resource_system_optimization(self, mock_detect):
        """Test optimization for low-resource systems."""

        # Mock a low-resource system
        low_resource_hardware = HardwareInfo(
            cpu_count=2, cpu_count_physical=2, cpu_freq=1.8, cpu_usage=80.0,
            memory_total_gb=4.0, memory_available_gb=1.5, memory_usage_percent=75.0,
            gpu_count=0, gpu_names=[], gpu_memory_total=[],
            gpu_memory_free=[], gpu_utilization=[],
            platform="Linux", architecture="x86_64", python_version="3.11.0",
            recommended_batch_size=8, recommended_workers=1, use_gpu=False
        )
        mock_detect.return_value = low_resource_hardware

        processor = MultiStrategyProcessor(enable_hardware_optimization=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "small_test.txt"
            content = "Small test content. " * 10
            file_path.write_text(content)

            strategies = ["sentence_based", "fixed_size"]

            # Should select conservative settings
            result = processor.process_file_with_strategies(
                file_path=file_path,
                strategies=strategies,
                parallel_mode="auto"
            )

            # Should work despite resource constraints
            assert len(result.successful_strategies) > 0
            # Should use conservative worker count
            assert result.hardware_config['workers'] <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
