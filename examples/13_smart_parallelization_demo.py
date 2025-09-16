#!/usr/bin/env python3
"""
Smart Parallelization Demo

Demonstrates the new smart parallelization feature that automatically
chooses the best processing strategy based on workload characteristics.
"""

import time
import tempfile
from pathlib import Path

from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.core.multi_strategy import MultiStrategyProcessor
from chunking_strategy.core.hardware import get_smart_parallelization_config


def create_test_files():
    """Create test files of different sizes."""
    files = {}
    
    # Small file (~1KB)
    small_file = Path(tempfile.mktemp(suffix='.txt'))
    with open(small_file, 'w') as f:
        f.write("This is a small test file. " * 40)
    files['small'] = small_file
    
    # Large file (~60KB) 
    large_file = Path(tempfile.mktemp(suffix='.txt'))
    with open(large_file, 'w') as f:
        f.write("This is a large test file with much more content. " * 1200)
    files['large'] = large_file
    
    return files


def demonstrate_smart_vs_basic():
    """Demonstrate smart parallelization vs basic approaches."""
    print("🧠 SMART PARALLELIZATION DEMONSTRATION")
    print("=" * 60)
    
    files = create_test_files()
    
    try:
        # Show file sizes
        print(f"\n📁 Test Files:")
        for name, path in files.items():
            size = path.stat().st_size
            print(f"   {name.capitalize()} file: {size:,} bytes ({size/1024:.1f} KB)")
        
        # Create different orchestrator configurations
        basic_orch = ChunkerOrchestrator(enable_hardware_optimization=False)
        hw_orch = ChunkerOrchestrator(enable_hardware_optimization=True, enable_smart_parallelization=False)  
        smart_orch = ChunkerOrchestrator(enable_smart_parallelization=True)
        
        print(f"\n⚡ Single File Processing Comparison:")
        print("-" * 50)
        
        # Test small file
        print(f"\n📄 Small File ({files['small'].stat().st_size:,} bytes):")
        
        start = time.time()
        basic_orch.chunk_file(files['small'])
        basic_time = time.time() - start
        print(f"   Basic (no HW):     {basic_time:.4f}s")
        
        start = time.time()
        hw_orch.chunk_file(files['small'])
        hw_time = time.time() - start
        print(f"   HW-optimized:      {hw_time:.4f}s ({hw_time/basic_time:.1f}x vs basic)")
        
        start = time.time()
        smart_orch.chunk_file(files['small'])
        smart_time = time.time() - start
        print(f"   Smart:             {smart_time:.4f}s ({smart_time/basic_time:.1f}x vs basic)")
        
        # Test large file
        print(f"\n📚 Large File ({files['large'].stat().st_size:,} bytes):")
        
        start = time.time()
        basic_orch.chunk_file(files['large'])
        basic_time = time.time() - start
        print(f"   Basic (no HW):     {basic_time:.4f}s")
        
        start = time.time()
        hw_orch.chunk_file(files['large'])
        hw_time = time.time() - start
        print(f"   HW-optimized:      {hw_time:.4f}s ({basic_time/hw_time:.1f}x speedup)")
        
        start = time.time()
        smart_orch.chunk_file(files['large'])
        smart_time = time.time() - start
        print(f"   Smart:             {smart_time:.4f}s ({basic_time/smart_time:.1f}x speedup)")
        
    finally:
        # Clean up
        for path in files.values():
            if path.exists():
                path.unlink()


def demonstrate_batch_processing():
    """Demonstrate smart batch processing decisions."""
    print(f"\n📦 BATCH PROCESSING DEMONSTRATION")
    print("=" * 50)
    
    # Create different batch scenarios
    small_files = []
    large_files = []
    
    # Small batch
    for i in range(3):
        small_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(small_file, 'w') as f:
            f.write(f"Small file {i} content. " * 30)
        small_files.append(small_file)
    
    # Large batch 
    for i in range(3):
        large_file = Path(tempfile.mktemp(suffix='.txt'))  
        with open(large_file, 'w') as f:
            f.write(f"Large file {i} content. " * 1000)
        large_files.append(large_file)
    
    try:
        smart_orch = ChunkerOrchestrator(enable_smart_parallelization=True)
        
        # Small batch processing
        total_size = sum(f.stat().st_size for f in small_files)
        print(f"\n📦 Small Batch ({len(small_files)} files, {total_size:,} bytes total):")
        
        start = time.time()
        results = smart_orch.chunk_files_batch(small_files, strategies=["sentence_based"])
        elapsed = time.time() - start
        
        config = get_smart_parallelization_config()
        should_use_threading = config.should_use_threading(total_size, len(small_files))
        processing_mode = "parallel" if should_use_threading else "sequential"
        
        print(f"   Processing mode:   {processing_mode}")
        print(f"   Time:              {elapsed:.4f}s")
        print(f"   Files processed:   {len(results)}")
        
        # Large batch processing
        total_size = sum(f.stat().st_size for f in large_files)
        print(f"\n📚 Large Batch ({len(large_files)} files, {total_size:,} bytes total):")
        
        start = time.time()
        results = smart_orch.chunk_files_batch(large_files, strategies=["sentence_based"])
        elapsed = time.time() - start
        
        should_use_threading = config.should_use_threading(total_size, len(large_files))
        processing_mode = "parallel" if should_use_threading else "sequential"
        
        print(f"   Processing mode:   {processing_mode}")
        print(f"   Time:              {elapsed:.4f}s")
        print(f"   Files processed:   {len(results)}")
        
    finally:
        # Clean up
        for f in small_files + large_files:
            if f.exists():
                f.unlink()


def demonstrate_multi_strategy():
    """Demonstrate smart multi-strategy processing."""
    print(f"\n🎯 MULTI-STRATEGY PROCESSING DEMONSTRATION")
    print("=" * 50)
    
    # Create test files
    small_file = Path(tempfile.mktemp(suffix='.txt'))
    large_file = Path(tempfile.mktemp(suffix='.txt'))
    
    with open(small_file, 'w') as f:
        f.write("Small content for multi-strategy test. " * 30)
    
    with open(large_file, 'w') as f:
        f.write("Large content for multi-strategy test. " * 1000)
    
    try:
        processor = MultiStrategyProcessor(enable_smart_parallelization=True)
        strategies = ["sentence_based", "paragraph_based", "fixed_size"]
        
        config = get_smart_parallelization_config()
        
        # Small file with multiple strategies
        file_size = small_file.stat().st_size
        should_parallel = config.should_use_parallel_strategies(len(strategies), file_size)
        processing_mode = "parallel" if should_parallel else "sequential"
        
        print(f"\n📄 Small File ({file_size:,} bytes, {len(strategies)} strategies):")
        print(f"   Expected mode:     {processing_mode}")
        
        start = time.time()
        result = processor.process_file_with_strategies(small_file, strategies)
        elapsed = time.time() - start
        
        print(f"   Time:              {elapsed:.4f}s")
        print(f"   Successful:        {len(result.successful_strategies)}/{len(strategies)}")
        
        # Large file with multiple strategies
        file_size = large_file.stat().st_size
        should_parallel = config.should_use_parallel_strategies(len(strategies), file_size)
        processing_mode = "parallel" if should_parallel else "sequential"
        
        print(f"\n📚 Large File ({file_size:,} bytes, {len(strategies)} strategies):")
        print(f"   Expected mode:     {processing_mode}")
        
        start = time.time()
        result = processor.process_file_with_strategies(large_file, strategies)
        elapsed = time.time() - start
        
        print(f"   Time:              {elapsed:.4f}s")
        print(f"   Successful:        {len(result.successful_strategies)}/{len(strategies)}")
        
    finally:
        # Clean up
        small_file.unlink()
        large_file.unlink()


def show_configuration():
    """Show the current smart parallelization configuration."""
    print(f"\n⚙️  SMART PARALLELIZATION CONFIGURATION")
    print("=" * 50)
    
    config = get_smart_parallelization_config()
    
    print(f"   HW optimization threshold:  {config.min_file_size_for_hw_optimization:,} bytes")
    print(f"   Threading threshold:        {config.min_total_size_for_threading:,} bytes total")
    print(f"   Batch threshold:            {config.min_files_for_batch} files")
    print(f"   Parallel strategies thresh: {config.min_strategies_for_parallel} strategies")
    print(f"   Max HW detection overhead:  {config.max_hw_detection_overhead}s")
    print(f"   Max thread setup overhead:  {config.max_thread_setup_overhead}s")


def main():
    """Run the complete demonstration."""
    show_configuration()
    demonstrate_smart_vs_basic()
    demonstrate_batch_processing()
    demonstrate_multi_strategy()
    
    print(f"\n🎯 KEY BENEFITS OF SMART PARALLELIZATION:")
    print("=" * 50)
    print("   ✅ No performance regression for small files")
    print("   ✅ Optimal performance for large files")
    print("   ✅ Intelligent workload-based decisions")
    print("   ✅ Hardware detection caching")
    print("   ✅ Automatic threshold-based optimization")
    print("   ✅ Backward compatible (enabled by default)")
    
    print(f"\n✨ Smart parallelization makes your chunking faster AND more reliable! ✨")


if __name__ == "__main__":
    main()
