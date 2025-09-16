#!/usr/bin/env python3
"""
Unit tests for smart parallelization functionality.

Tests the smart parallelization configuration and its integration with
the orchestrator and multi-strategy processor.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from chunking_strategy.core.hardware import (
    SmartParallelizationConfig,
    get_smart_parallelization_config,
    configure_smart_parallelization
)
from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.core.multi_strategy import MultiStrategyProcessor


class TestSmartParallelizationConfig:
    """Test cases for SmartParallelizationConfig class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = SmartParallelizationConfig()

        assert config.min_file_size_for_hw_optimization == 50000
        assert config.min_total_size_for_threading == 100000
        assert config.min_files_for_batch == 10
        assert config.min_strategies_for_parallel == 3
        assert config.max_hw_detection_overhead == 0.2
        assert config.max_thread_setup_overhead == 0.3

    def test_hardware_optimization_decision(self):
        """Test hardware optimization decisions based on file size."""
        config = SmartParallelizationConfig()

        # Small files should not use hardware optimization
        assert not config.should_use_hardware_optimization(1000)   # 1KB
        assert not config.should_use_hardware_optimization(10000)  # 10KB
        assert not config.should_use_hardware_optimization(49999)  # Just under threshold

        # Large files should use hardware optimization
        assert config.should_use_hardware_optimization(50000)      # Exactly at threshold
        assert config.should_use_hardware_optimization(100000)     # 100KB
        assert config.should_use_hardware_optimization(1000000)    # 1MB

    def test_threading_decision(self):
        """Test threading decisions based on total size and file count."""
        config = SmartParallelizationConfig()

        # Small total size and few files - no threading
        assert not config.should_use_threading(50000, 5)   # 50KB, 5 files
        assert not config.should_use_threading(80000, 8)   # 80KB, 8 files

        # Large total size - should use threading
        assert config.should_use_threading(150000, 5)      # 150KB, 5 files
        assert config.should_use_threading(200000, 3)      # 200KB, 3 files

        # Many files - should use threading regardless of size
        assert config.should_use_threading(30000, 10)      # 30KB, 10 files
        assert config.should_use_threading(40000, 15)      # 40KB, 15 files

        # Edge cases
        assert config.should_use_threading(100000, 10)     # Both thresholds met
        assert not config.should_use_threading(99999, 9)   # Just under both thresholds

    def test_parallel_strategies_decision(self):
        """Test parallel strategy processing decisions."""
        config = SmartParallelizationConfig()

        # Few strategies - no parallel processing
        assert not config.should_use_parallel_strategies(1, 100000)  # 1 strategy, 100KB
        assert not config.should_use_parallel_strategies(2, 100000)  # 2 strategies, 100KB

        # Many strategies but small file - no parallel processing
        assert not config.should_use_parallel_strategies(5, 30000)   # 5 strategies, 30KB
        assert not config.should_use_parallel_strategies(4, 49999)   # 4 strategies, just under threshold

        # Many strategies and large file - use parallel processing
        assert config.should_use_parallel_strategies(3, 50000)       # 3 strategies, 50KB
        assert config.should_use_parallel_strategies(5, 100000)      # 5 strategies, 100KB
        assert config.should_use_parallel_strategies(10, 200000)     # 10 strategies, 200KB

    def test_hardware_info_caching(self):
        """Test hardware info caching functionality."""
        config = SmartParallelizationConfig()

        # Initially no cached info
        assert config._cached_hardware_info is None
        assert config._hardware_detection_time is None

        # First call should detect and cache
        with patch('chunking_strategy.core.hardware.HardwareDetector') as mock_detector:
            mock_instance = Mock()
            mock_detector.return_value = mock_instance
            mock_info = {'cpu_count': 4, 'memory_total_gb': 16.0}
            mock_instance.detect_hardware.return_value = mock_info

            # Get hardware info (should detect)
            result1 = config.get_cached_hardware_info()
            assert result1 == mock_info
            assert config._cached_hardware_info == mock_info
            assert config._hardware_detection_time is not None
            assert mock_instance.detect_hardware.call_count == 1

            # Second call should use cache
            result2 = config.get_cached_hardware_info()
            assert result2 == mock_info
            assert mock_instance.detect_hardware.call_count == 1  # No additional calls

    def test_global_configuration_access(self):
        """Test global configuration access functions."""
        # Test getting global config
        global_config = get_smart_parallelization_config()
        assert isinstance(global_config, SmartParallelizationConfig)

        # Test configuring thresholds
        original_threshold = global_config.min_file_size_for_hw_optimization
        configure_smart_parallelization(min_file_size_for_hw_optimization=75000)

        updated_config = get_smart_parallelization_config()
        assert updated_config.min_file_size_for_hw_optimization == 75000

        # Reset to original
        configure_smart_parallelization(min_file_size_for_hw_optimization=original_threshold)


class TestSmartParallelizationIntegration:
    """Test integration of smart parallelization with main components."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create test files of different sizes
        self.temp_files = {}

        # Small file (~1KB)
        small_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(small_file, 'w') as f:
            f.write("Small file content. " * 50)  # ~1KB
        self.temp_files['small'] = small_file

        # Large file (~60KB)
        large_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(large_file, 'w') as f:
            f.write("Large file content. " * 3000)  # ~60KB
        self.temp_files['large'] = large_file

    def teardown_method(self):
        """Clean up test fixtures."""
        for file_path in self.temp_files.values():
            if file_path.exists():
                file_path.unlink()

    def test_orchestrator_smart_parallelization_enabled(self):
        """Test orchestrator with smart parallelization enabled."""
        # Test with smart parallelization enabled (default)
        orch = ChunkerOrchestrator(enable_smart_parallelization=True)

        assert orch.enable_smart_parallelization is True
        assert orch.smart_config is not None

        # Test chunking a small file - should be fast (no HW overhead)
        start_time = time.time()
        result = orch.chunk_file(self.temp_files['small'])
        small_time = time.time() - start_time

        assert result.chunks
        assert result.strategy_used  # Should have selected a strategy

        # Test chunking a large file - may use HW optimization
        start_time = time.time()
        result = orch.chunk_file(self.temp_files['large'])
        large_time = time.time() - start_time

        assert result.chunks
        assert result.strategy_used

    def test_orchestrator_smart_parallelization_disabled(self):
        """Test orchestrator with smart parallelization disabled."""
        orch = ChunkerOrchestrator(enable_smart_parallelization=False)

        assert orch.enable_smart_parallelization is False
        assert orch.smart_config is None

        # Should still work, but without smart decisions
        result = orch.chunk_file(self.temp_files['small'])
        assert result.chunks

    def test_orchestrator_batch_processing_smart_decisions(self):
        """Test batch processing with smart parallelization decisions."""
        orch = ChunkerOrchestrator(enable_smart_parallelization=True)

        # Small batch - should use sequential processing
        small_files = [self.temp_files['small']] * 3
        results = orch.chunk_files_batch(small_files, strategies=["sentence_based"])
        assert len(results) == 3
        assert all(r.chunks for r in results)

        # Large batch - might use parallel processing
        large_files = [self.temp_files['large']] * 3
        results = orch.chunk_files_batch(large_files, strategies=["sentence_based"])
        assert len(results) == 3
        assert all(r.chunks for r in results)

    def test_multi_strategy_processor_smart_decisions(self):
        """Test multi-strategy processor with smart parallelization."""
        processor = MultiStrategyProcessor(enable_smart_parallelization=True)

        assert processor.enable_smart_parallelization is True
        assert processor.smart_config is not None

        # Few strategies with small file - should use sequential
        strategies = ["sentence_based", "paragraph_based"]
        result = processor.process_file_with_strategies(
            self.temp_files['small'], strategies
        )
        assert len(result.successful_strategies) >= 1

        # Many strategies with large file - might use parallel
        strategies = ["sentence_based", "paragraph_based", "fixed_size"]
        result = processor.process_file_with_strategies(
            self.temp_files['large'], strategies
        )
        assert len(result.successful_strategies) >= 1

    def test_hardware_detection_caching(self):
        """Test that hardware detection is cached across components."""
        # Create multiple components with smart parallelization
        orch1 = ChunkerOrchestrator(enable_smart_parallelization=True)
        orch2 = ChunkerOrchestrator(enable_smart_parallelization=True)
        processor = MultiStrategyProcessor(enable_smart_parallelization=True)

        # They should all use the same cached hardware info
        assert orch1.smart_config is orch2.smart_config
        assert orch1.smart_config is processor.smart_config

    def test_performance_regression_prevention(self):
        """Test that smart parallelization prevents performance regressions."""
        # Create orchestrators with and without smart parallelization
        orch_smart = ChunkerOrchestrator(enable_smart_parallelization=True)
        orch_basic = ChunkerOrchestrator(enable_hardware_optimization=False)

        # For small files, smart parallelization should be as fast as basic
        start_time = time.time()
        orch_basic.chunk_file(self.temp_files['small'])
        basic_time = time.time() - start_time

        start_time = time.time()
        orch_smart.chunk_file(self.temp_files['small'])
        smart_time = time.time() - start_time

        # Smart should not be significantly slower (within 50% is acceptable for test)
        assert smart_time < basic_time * 1.5, f"Smart parallelization too slow: {smart_time:.4f}s vs {basic_time:.4f}s"


class TestSmartParallelizationEdgeCases:
    """Test edge cases for smart parallelization."""

    def test_nonexistent_file_handling(self):
        """Test handling of nonexistent files."""
        orch = ChunkerOrchestrator(enable_smart_parallelization=True)

        with pytest.raises(FileNotFoundError):
            orch.chunk_file("nonexistent_file.txt")

    def test_empty_file_handling(self):
        """Test handling of empty files."""
        empty_file = Path(tempfile.mktemp(suffix='.txt'))
        empty_file.touch()  # Create empty file

        try:
            orch = ChunkerOrchestrator(enable_smart_parallelization=True)
            result = orch.chunk_file(empty_file)
            # Should handle empty file gracefully
            assert isinstance(result.chunks, list)  # May be empty but should be a list
        finally:
            empty_file.unlink()

    def test_custom_thresholds(self):
        """Test smart parallelization with custom thresholds."""
        # Configure custom thresholds
        configure_smart_parallelization(
            min_file_size_for_hw_optimization=10000,  # 10KB
            min_strategies_for_parallel=2
        )

        try:
            config = get_smart_parallelization_config()

            # Test with new thresholds
            assert config.should_use_hardware_optimization(15000)  # 15KB > 10KB
            assert not config.should_use_hardware_optimization(8000)  # 8KB < 10KB

            assert config.should_use_parallel_strategies(2, 15000)  # 2 strategies >= 2, 15KB >= 10KB
            assert not config.should_use_parallel_strategies(1, 15000)  # 1 strategy < 2

        finally:
            # Reset to defaults
            configure_smart_parallelization(
                min_file_size_for_hw_optimization=50000,
                min_strategies_for_parallel=3
            )

    def test_mixed_workloads(self):
        """Test smart parallelization with mixed file sizes in batch processing."""
        # Create files of different sizes
        small_file = Path(tempfile.mktemp(suffix='.txt'))
        medium_file = Path(tempfile.mktemp(suffix='.txt'))
        large_file = Path(tempfile.mktemp(suffix='.txt'))

        with open(small_file, 'w') as f:
            f.write("Small content. " * 20)  # ~300B
        with open(medium_file, 'w') as f:
            f.write("Medium content. " * 500)  # ~7KB
        with open(large_file, 'w') as f:
            f.write("Large content. " * 2000)  # ~26KB

        try:
            orch = ChunkerOrchestrator(enable_smart_parallelization=True)

            # Mixed batch - decision should be based on total size and file count
            mixed_files = [small_file, medium_file, large_file] * 2  # 6 files total
            results = orch.chunk_files_batch(mixed_files, strategies=["sentence_based"])

            assert len(results) == 6
            assert all(r.chunks for r in results if r.chunks is not None)

        finally:
            for f in [small_file, medium_file, large_file]:
                if f.exists():
                    f.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
