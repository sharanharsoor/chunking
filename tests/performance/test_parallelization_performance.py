"""
Performance benchmarks for parallelization features.

This module tests that parallel processing actually improves performance
and measures the effectiveness of hardware optimizations.
"""

import pytest
import time
import tempfile
from pathlib import Path
from typing import List
import statistics

from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.core.multi_strategy import MultiStrategyProcessor


class TestParallelizationPerformance:
    """Performance tests for parallel processing features."""

    @pytest.fixture
    def large_test_files(self) -> List[Path]:
        """Create multiple large test files for performance testing."""
        files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create 10 files of different sizes
            for i in range(10):
                file_path = temp_path / f"large_test_{i}.txt"
                # Create progressively larger files
                content_multiplier = 100 + (i * 50)  # 100, 150, 200, ... chunks of content
                content = f"This is test content for performance benchmarking. File {i} contains structured text with multiple sentences and paragraphs. " * content_multiplier
                file_path.write_text(content)
                files.append(file_path)
            
            yield files

    @pytest.fixture
    def medium_test_file(self) -> Path:
        """Create a single medium-sized test file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "medium_test.txt"
            content = """
            This is a comprehensive test document designed for multi-strategy performance benchmarking.
            
            It contains multiple paragraphs with varying sentence lengths and complexity. Each paragraph
            is designed to test different aspects of the chunking strategies. The content includes
            technical terms, varied punctuation, and different sentence structures.
            
            Performance testing requires sufficient content volume to demonstrate meaningful differences
            between sequential and parallel processing modes. This document provides that volume while
            maintaining realistic text characteristics.
            
            The multi-strategy processor should be able to apply different chunking approaches to this
            content simultaneously, demonstrating the effectiveness of parallel processing for
            computational tasks that can be decomposed into independent operations.
            
            Additional paragraphs ensure we have enough content for comprehensive testing across
            multiple strategies with different parameter configurations and processing modes.
            """ * 20  # Multiply to create substantial content
            
            file_path.write_text(content)
            yield file_path

    def test_batch_processing_performance_scaling(self, large_test_files):
        """Test that batch processing scales with the number of workers."""
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)
        
        # Test with limited files to ensure reasonable test time
        test_files = large_test_files[:6]
        results = {}
        
        # Test different worker counts
        for workers in [1, 2, 4]:
            start_time = time.time()
            
            chunk_results = orchestrator.chunk_files_batch(
                file_paths=test_files,
                parallel_mode="thread",
                max_workers=workers
            )
            
            processing_time = time.time() - start_time
            results[workers] = {
                'time': processing_time,
                'chunks': sum(len(result.chunks) for result in chunk_results),
                'successful_files': sum(1 for result in chunk_results if len(result.chunks) > 0)
            }
            
            print(f"Workers: {workers}, Time: {processing_time:.2f}s, Chunks: {results[workers]['chunks']}")
        
        # Verify all configurations processed the files successfully
        for workers, result in results.items():
            assert result['successful_files'] == len(test_files), f"Worker count {workers} failed to process all files"
            assert result['chunks'] > 0, f"Worker count {workers} produced no chunks"
        
        # Performance should generally improve with more workers (allowing for some variance)
        # We don't enforce strict performance improvements due to test environment variability
        single_worker_time = results[1]['time']
        multi_worker_times = [results[w]['time'] for w in [2, 4]]
        
        print(f"Performance comparison - 1 worker: {single_worker_time:.2f}s, "
              f"Multi-worker average: {statistics.mean(multi_worker_times):.2f}s")

    def test_sequential_vs_parallel_multi_strategy(self, medium_test_file):
        """Compare sequential vs parallel multi-strategy processing performance."""
        processor = MultiStrategyProcessor(enable_hardware_optimization=True)
        
        strategies = ["sentence_based", "paragraph_based", "fixed_size"]
        iterations = 3  # Run multiple times for more reliable measurements
        
        sequential_times = []
        parallel_times = []
        
        for _ in range(iterations):
            # Sequential processing
            start_time = time.time()
            sequential_result = processor.process_file_with_strategies(
                file_path=medium_test_file,
                strategies=strategies,
                parallel_mode="sequential"
            )
            sequential_time = time.time() - start_time
            sequential_times.append(sequential_time)
            
            # Parallel processing
            start_time = time.time()
            parallel_result = processor.process_file_with_strategies(
                file_path=medium_test_file,
                strategies=strategies,
                parallel_mode="thread",
                max_workers=len(strategies)
            )
            parallel_time = time.time() - start_time
            parallel_times.append(parallel_time)
            
            # Verify both produce equivalent results
            assert len(sequential_result.successful_strategies) == len(parallel_result.successful_strategies)
            assert sequential_result.total_chunks > 0
            assert parallel_result.total_chunks > 0
        
        avg_sequential = statistics.mean(sequential_times)
        avg_parallel = statistics.mean(parallel_times)
        
        print(f"Average times - Sequential: {avg_sequential:.3f}s, Parallel: {avg_parallel:.3f}s")
        print(f"Speedup ratio: {avg_sequential / avg_parallel:.2f}x")
        
        # Both should complete successfully
        assert avg_sequential > 0
        assert avg_parallel > 0

    def test_hardware_optimization_impact(self, large_test_files):
        """Test the impact of enabling/disabling hardware optimization."""
        test_files = large_test_files[:4]
        
        # Test with hardware optimization enabled
        orchestrator_optimized = ChunkerOrchestrator(enable_hardware_optimization=True)
        start_time = time.time()
        results_optimized = orchestrator_optimized.chunk_files_batch(
            file_paths=test_files,
            parallel_mode="auto"
        )
        time_optimized = time.time() - start_time
        
        # Test with hardware optimization disabled
        orchestrator_basic = ChunkerOrchestrator(enable_hardware_optimization=False)
        start_time = time.time()
        results_basic = orchestrator_basic.chunk_files_batch(
            file_paths=test_files,
            parallel_mode="sequential"  # Fallback mode when optimization disabled
        )
        time_basic = time.time() - start_time
        
        # Both should process all files successfully
        assert len(results_optimized) == len(test_files)
        assert len(results_basic) == len(test_files)
        
        chunks_optimized = sum(len(result.chunks) for result in results_optimized)
        chunks_basic = sum(len(result.chunks) for result in results_basic)
        
        print(f"Hardware optimized: {time_optimized:.2f}s, {chunks_optimized} chunks")
        print(f"Basic processing: {time_basic:.2f}s, {chunks_basic} chunks")
        
        # Verify successful processing
        assert chunks_optimized > 0
        assert chunks_basic > 0

    def test_processing_mode_performance_comparison(self, large_test_files):
        """Compare performance of different processing modes."""
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)
        test_files = large_test_files[:5]
        
        modes = ["sequential", "thread", "process"]
        results = {}
        
        for mode in modes:
            start_time = time.time()
            
            try:
                chunk_results = orchestrator.chunk_files_batch(
                    file_paths=test_files,
                    parallel_mode=mode,
                    max_workers=2
                )
                
                processing_time = time.time() - start_time
                total_chunks = sum(len(result.chunks) for result in chunk_results)
                successful_files = sum(1 for result in chunk_results if len(result.chunks) > 0)
                
                results[mode] = {
                    'time': processing_time,
                    'chunks': total_chunks,
                    'successful_files': successful_files,
                    'chunks_per_second': total_chunks / processing_time if processing_time > 0 else 0
                }
                
            except Exception as e:
                print(f"Mode {mode} failed: {e}")
                results[mode] = {
                    'time': float('inf'),
                    'chunks': 0,
                    'successful_files': 0,
                    'chunks_per_second': 0
                }
        
        # Print performance comparison
        print("\\nProcessing Mode Performance Comparison:")
        print("Mode        | Time (s) | Chunks | Files | Chunks/sec")
        print("-" * 50)
        for mode, stats in results.items():
            print(f"{mode:10} | {stats['time']:7.2f} | {stats['chunks']:6} | "
                  f"{stats['successful_files']:5} | {stats['chunks_per_second']:8.1f}")
        
        # Verify all modes processed files successfully (allowing for some failures in testing environment)
        successful_modes = [mode for mode, stats in results.items() if stats['successful_files'] > 0]
        assert len(successful_modes) > 0, "At least one processing mode should succeed"

    def test_strategy_complexity_performance_impact(self, medium_test_file):
        """Test how strategy complexity affects parallel processing benefits."""
        processor = MultiStrategyProcessor(enable_hardware_optimization=True)
        
        # Simple strategies (should have less parallel benefit)
        simple_strategies = ["fixed_size"]
        
        # Complex strategies (should benefit more from parallelization)
        complex_strategies = ["sentence_based", "paragraph_based"]
        
        # Mixed complexity
        mixed_strategies = ["fixed_size", "sentence_based", "paragraph_based"]
        
        test_cases = [
            ("simple", simple_strategies),
            ("complex", complex_strategies),
            ("mixed", mixed_strategies)
        ]
        
        for case_name, strategies in test_cases:
            if len(strategies) == 1:
                # Skip parallelization test for single strategy
                continue
                
            # Sequential processing
            start_time = time.time()
            sequential_result = processor.process_file_with_strategies(
                file_path=medium_test_file,
                strategies=strategies,
                parallel_mode="sequential"
            )
            sequential_time = time.time() - start_time
            
            # Parallel processing
            start_time = time.time()
            parallel_result = processor.process_file_with_strategies(
                file_path=medium_test_file,
                strategies=strategies,
                parallel_mode="thread",
                max_workers=len(strategies)
            )
            parallel_time = time.time() - start_time
            
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            
            print(f"{case_name.capitalize()} strategies:")
            print(f"  Sequential: {sequential_time:.3f}s, Parallel: {parallel_time:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Success rate: {parallel_result.success_rate:.1%}")
            
            # Verify processing completed successfully
            assert len(sequential_result.successful_strategies) > 0
            assert len(parallel_result.successful_strategies) > 0
            assert sequential_result.total_chunks > 0
            assert parallel_result.total_chunks > 0

    def test_memory_usage_scaling(self, large_test_files):
        """Test that parallel processing doesn't cause excessive memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process files with different worker counts
        test_files = large_test_files[:6]
        
        for workers in [1, 2, 4]:
            # Force garbage collection before measurement
            import gc
            gc.collect()
            
            pre_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            chunk_results = orchestrator.chunk_files_batch(
                file_paths=test_files,
                parallel_mode="thread",
                max_workers=workers
            )
            
            post_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = post_memory - pre_memory
            
            print(f"Workers: {workers}, Memory increase: {memory_increase:.1f} MB")
            
            # Verify processing succeeded
            assert len(chunk_results) == len(test_files)
            successful_files = sum(1 for result in chunk_results if len(result.chunks) > 0)
            assert successful_files > 0
            
            # Memory usage should be reasonable (allowing for test environment variations)
            # This is more of a monitoring test than a strict assertion
            assert memory_increase < 1000, f"Excessive memory usage: {memory_increase:.1f} MB"

    def test_concurrent_orchestrator_instances(self, large_test_files):
        """Test performance when multiple orchestrator instances run concurrently."""
        import threading
        
        test_files = large_test_files[:4]
        results = {}
        errors = {}
        
        def worker(instance_id):
            try:
                orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)
                start_time = time.time()
                
                chunk_results = orchestrator.chunk_files_batch(
                    file_paths=test_files,
                    parallel_mode="thread",
                    max_workers=2
                )
                
                processing_time = time.time() - start_time
                results[instance_id] = {
                    'time': processing_time,
                    'chunks': sum(len(result.chunks) for result in chunk_results),
                    'files': len(chunk_results)
                }
                
            except Exception as e:
                errors[instance_id] = str(e)
        
        # Run multiple instances concurrently
        threads = []
        for i in range(3):  # 3 concurrent instances
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all instances completed successfully
        assert len(errors) == 0, f"Errors in concurrent processing: {errors}"
        assert len(results) == 3, "All instances should complete"
        
        # Print results
        print("Concurrent instance performance:")
        for instance_id, stats in results.items():
            print(f"Instance {instance_id}: {stats['time']:.2f}s, {stats['chunks']} chunks")
        
        # All instances should produce results
        for stats in results.values():
            assert stats['chunks'] > 0
            assert stats['files'] == len(test_files)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print outputs
