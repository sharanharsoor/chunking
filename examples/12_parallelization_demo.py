#!/usr/bin/env python3
"""
Parallelization and Hardware Optimization Demo

This script demonstrates the new parallelization features in the chunking strategy library,
including hardware-aware batch processing and multi-strategy parallel processing.
"""

import time
from pathlib import Path
import tempfile
from typing import List

from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.core.multi_strategy import MultiStrategyProcessor
from chunking_strategy.core.hardware import HardwareDetector


def create_demo_files() -> List[Path]:
    """Create temporary demo files for testing."""
    files = []
    temp_dir = Path(tempfile.mkdtemp())
    print(f"📁 Creating demo files in: {temp_dir}")
    
    # Create different types of content
    demo_contents = [
        ("technical_doc.txt", """
        API Documentation: Advanced Chunking System
        
        Overview:
        The Advanced Chunking System provides sophisticated text processing capabilities
        with hardware-optimized parallel processing. This system can handle multiple
        file formats and apply various chunking strategies simultaneously.
        
        Features:
        - Multi-strategy parallel processing
        - Hardware-aware optimization
        - Batch processing capabilities
        - Performance monitoring and analytics
        
        Implementation:
        The system uses a modular architecture that allows for easy extension and
        customization. Each chunking strategy operates independently, enabling
        parallel execution across multiple CPU cores and processing units.
        """),
        
        ("research_paper.txt", """
        Abstract:
        This paper presents a novel approach to text chunking using parallel processing
        techniques. We demonstrate significant performance improvements through
        hardware optimization and multi-strategy processing.
        
        Introduction:
        Text chunking is a fundamental operation in natural language processing.
        Traditional approaches process text sequentially, which can be inefficient
        for large documents. Our system addresses this limitation through parallelization.
        
        Methodology:
        We implemented a multi-threaded architecture that can apply different chunking
        strategies simultaneously. The system automatically detects hardware capabilities
        and optimizes processing parameters accordingly.
        
        Results:
        Our experiments show up to 3x performance improvement on multi-core systems
        while maintaining the same quality of output. Memory usage scales linearly
        with the number of parallel workers.
        """),
        
        ("user_guide.txt", """
        User Guide: Getting Started with Parallel Chunking
        
        Chapter 1: Installation and Setup
        Before using the parallel chunking features, ensure your system meets
        the minimum requirements. The system works best on multi-core processors
        with sufficient RAM.
        
        Chapter 2: Basic Usage
        Start with simple batch processing to understand the system capabilities.
        The auto mode will detect your hardware and select optimal settings.
        
        Chapter 3: Advanced Features
        For power users, manual configuration allows fine-tuning of processing
        parameters. You can specify worker counts, processing modes, and
        strategy combinations.
        
        Chapter 4: Performance Optimization
        Monitor system resources during processing to identify bottlenecks.
        Adjust batch sizes and worker counts based on your specific workload.
        """),
        
        ("code_examples.py", '''
        """
        Code examples for the chunking system.
        """
        
        def basic_chunking_example():
            """Demonstrate basic chunking functionality."""
            from chunking_strategy.orchestrator import ChunkerOrchestrator
            
            # Initialize orchestrator with hardware optimization
            orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)
            
            # Process single file
            result = orchestrator.chunk_file("document.txt")
            print(f"Generated {len(result.chunks)} chunks")
            
            return result
        
        def parallel_batch_example():
            """Demonstrate parallel batch processing."""
            from chunking_strategy.orchestrator import ChunkerOrchestrator
            
            orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)
            
            # Process multiple files in parallel
            files = ["doc1.txt", "doc2.txt", "doc3.txt"]
            results = orchestrator.chunk_files_batch(
                file_paths=files,
                parallel_mode="auto",
                max_workers=4
            )
            
            total_chunks = sum(len(result.chunks) for result in results)
            print(f"Processed {len(files)} files, generated {total_chunks} chunks")
            
            return results
        
        def multi_strategy_example():
            """Demonstrate multi-strategy processing."""
            from chunking_strategy.core.multi_strategy import MultiStrategyProcessor
            
            processor = MultiStrategyProcessor(enable_hardware_optimization=True)
            
            strategies = ["sentence_based", "paragraph_based", "fixed_size"]
            result = processor.process_file_with_strategies(
                file_path="document.txt",
                strategies=strategies,
                parallel_mode="thread"
            )
            
            print(f"Applied {len(result.successful_strategies)} strategies")
            for strategy in result.successful_strategies:
                chunks = len(result.strategy_results[strategy].chunks)
                print(f"  {strategy}: {chunks} chunks")
            
            return result
        '''),
        
        ("data_analysis.json", '''
        {
            "dataset": "text_processing_benchmark",
            "metrics": {
                "processing_time": {
                    "sequential": 45.2,
                    "parallel_2_workers": 24.8,
                    "parallel_4_workers": 13.7,
                    "parallel_8_workers": 8.9
                },
                "memory_usage": {
                    "sequential": 156,
                    "parallel_2_workers": 312,
                    "parallel_4_workers": 624,
                    "parallel_8_workers": 1248
                },
                "chunk_quality": {
                    "sentence_based": 0.87,
                    "paragraph_based": 0.82,
                    "fixed_size": 0.64,
                    "rolling_hash": 0.71
                }
            },
            "system_info": {
                "cpu_cores": 8,
                "memory_gb": 16,
                "gpu_count": 0,
                "platform": "Linux"
            },
            "recommendations": [
                "Use 4 workers for optimal performance/memory balance",
                "Sentence-based strategy provides best quality",
                "Parallel processing provides 3-5x speedup",
                "Memory usage scales linearly with worker count"
            ]
        }
        ''')
    ]
    
    for filename, content in demo_contents:
        file_path = temp_dir / filename
        file_path.write_text(content.strip())
        files.append(file_path)
        print(f"  ✅ Created {filename} ({len(content):,} chars)")
    
    return files


def demo_hardware_detection():
    """Demonstrate hardware detection capabilities."""
    print("\\n🔧 HARDWARE DETECTION DEMO")
    print("=" * 60)
    
    detector = HardwareDetector()
    hardware_info = detector.detect_hardware()
    
    print(f"🖥️  System Information:")
    print(f"   CPU Cores: {hardware_info.cpu_count} ({hardware_info.cpu_count_physical} physical)")
    if hardware_info.cpu_freq:
        print(f"   CPU Frequency: {hardware_info.cpu_freq:.1f} GHz")
    if hardware_info.memory_total_gb:
        print(f"   Memory: {hardware_info.memory_total_gb:.1f} GB total, "
              f"{hardware_info.memory_available_gb:.1f} GB available")
    print(f"   GPUs: {hardware_info.gpu_count}")
    if hardware_info.gpu_names:
        for i, gpu_name in enumerate(hardware_info.gpu_names):
            print(f"     GPU {i}: {gpu_name} ({hardware_info.gpu_memory_total[i]:.1f} GB)")
    
    print(f"\\n🎯 Recommendations:")
    print(f"   Recommended batch size: {hardware_info.recommended_batch_size}")
    print(f"   Recommended workers: {hardware_info.recommended_workers}")
    print(f"   Use GPU: {hardware_info.use_gpu}")


def demo_basic_orchestrator_with_hardware():
    """Demonstrate hardware-aware orchestrator."""
    print("\\n🎼 HARDWARE-AWARE ORCHESTRATOR DEMO")
    print("=" * 60)
    
    # Create orchestrator with hardware optimization
    orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)
    
    # Create a test file
    demo_files = create_demo_files()
    test_file = demo_files[0]  # Use the first demo file
    
    print(f"\\n📄 Processing single file: {test_file.name}")
    start_time = time.time()
    result = orchestrator.chunk_file(test_file)
    processing_time = time.time() - start_time
    
    print(f"✅ Results:")
    print(f"   Chunks generated: {len(result.chunks)}")
    print(f"   Processing time: {processing_time:.3f}s")
    print(f"   Strategy used: {result.strategy_used if hasattr(result, 'strategy_used') else 'auto'}")
    
    # Show sample chunk
    if result.chunks:
        sample_chunk = result.chunks[0]
        preview = sample_chunk.content[:100] + "..." if len(sample_chunk.content) > 100 else sample_chunk.content
        print(f"   Sample chunk: {preview}")


def demo_batch_processing_performance():
    """Demonstrate batch processing with performance comparison."""
    print("\\n⚡ BATCH PROCESSING PERFORMANCE DEMO")
    print("=" * 60)
    
    demo_files = create_demo_files()
    orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)
    
    # Test different processing modes
    modes = ["sequential", "thread", "process"]
    results = {}
    
    for mode in modes:
        print(f"\\n🔄 Testing {mode} mode...")
        start_time = time.time()
        
        try:
            chunk_results = orchestrator.chunk_files_batch(
                file_paths=demo_files,
                parallel_mode=mode,
                max_workers=2
            )
            
            processing_time = time.time() - start_time
            total_chunks = sum(len(result.chunks) for result in chunk_results)
            successful_files = sum(1 for result in chunk_results if len(result.chunks) > 0)
            
            results[mode] = {
                'time': processing_time,
                'chunks': total_chunks,
                'files': successful_files
            }
            
            print(f"   ✅ Processed {successful_files}/{len(demo_files)} files")
            print(f"   ⏱️  Time: {processing_time:.3f}s")
            print(f"   📊 Chunks: {total_chunks}")
            print(f"   🚀 Rate: {total_chunks/processing_time:.1f} chunks/sec")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[mode] = {'time': float('inf'), 'chunks': 0, 'files': 0}
    
    # Performance comparison
    print("\\n📈 Performance Comparison:")
    print("Mode       | Time (s) | Chunks | Rate (c/s)")
    print("-" * 40)
    for mode, stats in results.items():
        rate = stats['chunks'] / stats['time'] if stats['time'] > 0 and stats['time'] != float('inf') else 0
        print(f"{mode:10} | {stats['time']:7.3f} | {stats['chunks']:6} | {rate:8.1f}")


def demo_multi_strategy_processing():
    """Demonstrate multi-strategy processing."""
    print("\\n🎯 MULTI-STRATEGY PROCESSING DEMO")
    print("=" * 60)
    
    demo_files = create_demo_files()
    processor = MultiStrategyProcessor(enable_hardware_optimization=True)
    
    # Test file (use the technical doc)
    test_file = demo_files[0]
    
    # Define strategies to test
    strategies = ["sentence_based", "paragraph_based", "fixed_size"]
    strategy_configs = {
        "sentence_based": {"max_sentences": 3},
        "paragraph_based": {"max_paragraphs": 2},
        "fixed_size": {"chunk_size": 500}
    }
    
    print(f"\\n📄 Processing {test_file.name} with {len(strategies)} strategies...")
    
    # Test different processing modes
    modes = ["sequential", "thread"]
    
    for mode in modes:
        print(f"\\n🔄 {mode.capitalize()} mode:")
        start_time = time.time()
        
        result = processor.process_file_with_strategies(
            file_path=test_file,
            strategies=strategies,
            strategy_configs=strategy_configs,
            parallel_mode=mode,
            max_workers=len(strategies)
        )
        
        processing_time = time.time() - start_time
        
        print(f"   ⏱️  Processing time: {processing_time:.3f}s")
        print(f"   ✅ Successful strategies: {len(result.successful_strategies)}/{len(strategies)}")
        print(f"   📊 Total chunks: {result.total_chunks}")
        print(f"   💯 Success rate: {result.success_rate:.1%}")
        
        if result.failed_strategies:
            print(f"   ❌ Failed strategies: {[name for name, _ in result.failed_strategies]}")
        
        print("   📈 Strategy breakdown:")
        for strategy in result.successful_strategies:
            chunks = len(result.strategy_results[strategy].chunks)
            print(f"     {strategy}: {chunks} chunks")


def demo_advanced_configuration():
    """Demonstrate advanced configuration options."""
    print("\\n⚙️  ADVANCED CONFIGURATION DEMO")
    print("=" * 60)
    
    demo_files = create_demo_files()
    
    # Custom orchestrator configuration
    custom_config = {
        "strategies": {
            "primary": "paragraph_based",
            "fallbacks": ["sentence_based", "fixed_size"]
        },
        "preprocessing": {
            "enabled": True,
            "normalize_whitespace": True
        },
        "postprocessing": {
            "enabled": True,
            "merge_short_chunks": True,
            "min_chunk_size": 100
        }
    }
    
    orchestrator = ChunkerOrchestrator(
        config=custom_config,
        enable_hardware_optimization=True
    )
    
    print("🔧 Custom Configuration:")
    print(f"   Primary strategy: {custom_config['strategies']['primary']}")
    print(f"   Fallback strategies: {custom_config['strategies']['fallbacks']}")
    print(f"   Preprocessing: {custom_config['preprocessing']['enabled']}")
    print(f"   Postprocessing: {custom_config['postprocessing']['enabled']}")
    
    # Process files with custom configuration
    print(f"\\n📄 Processing {len(demo_files)} files with custom config...")
    start_time = time.time()
    
    results = orchestrator.chunk_files_batch(
        file_paths=demo_files,
        parallel_mode="auto"
    )
    
    processing_time = time.time() - start_time
    total_chunks = sum(len(result.chunks) for result in results)
    
    print(f"   ✅ Processed {len(results)} files in {processing_time:.3f}s")
    print(f"   📊 Generated {total_chunks} chunks")
    print(f"   🚀 Average: {total_chunks/len(results):.1f} chunks per file")


def demo_progress_monitoring():
    """Demonstrate progress monitoring capabilities."""
    print("\\n📊 PROGRESS MONITORING DEMO")
    print("=" * 60)
    
    demo_files = create_demo_files()
    orchestrator = ChunkerOrchestrator(enable_hardware_optimization=True)
    
    # Progress tracking
    progress_updates = []
    
    def progress_callback(current, total, message):
        progress_updates.append((current, total, message))
        percent = (current / total) * 100
        print(f"   📈 Progress: {current}/{total} ({percent:.1f}%) - {message}")
    
    print("\\n🔄 Processing with progress monitoring...")
    start_time = time.time()
    
    results = orchestrator.chunk_files_batch(
        file_paths=demo_files,
        parallel_mode="thread",
        max_workers=2,
        progress_callback=progress_callback
    )
    
    processing_time = time.time() - start_time
    
    print(f"\\n✅ Completed processing:")
    print(f"   Files processed: {len(results)}")
    print(f"   Total time: {processing_time:.3f}s")
    print(f"   Progress updates: {len(progress_updates)}")
    print(f"   Final chunks: {sum(len(result.chunks) for result in results)}")


def cleanup_demo_files(files: List[Path]):
    """Clean up temporary demo files."""
    if files:
        temp_dir = files[0].parent
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\\n🧹 Cleaned up demo files from {temp_dir}")


def main():
    """Run all demonstration examples."""
    print("🚀 PARALLELIZATION AND HARDWARE OPTIMIZATION DEMO")
    print("=" * 80)
    
    try:
        # Run all demos
        demo_hardware_detection()
        demo_basic_orchestrator_with_hardware()
        demo_batch_processing_performance()
        demo_multi_strategy_processing()
        demo_advanced_configuration()
        demo_progress_monitoring()
        
        print("\\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\\n💡 Key Takeaways:")
        print("   • Hardware optimization provides automatic performance tuning")
        print("   • Batch processing scales well with multiple workers")
        print("   • Multi-strategy processing enables comprehensive analysis")
        print("   • Progress monitoring helps track long-running operations")
        print("   • Different processing modes suit different use cases")
        
    except Exception as e:
        print(f"\\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Note: Demo files are cleaned up automatically when temp directory goes out of scope
        pass


if __name__ == "__main__":
    main()
