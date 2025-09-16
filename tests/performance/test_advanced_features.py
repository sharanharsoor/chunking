#!/usr/bin/env python3
"""
Comprehensive test of advanced chunking features:
- PDF processing with images/tables
- Hardware detection and optimization
- Batch processing with multiple files
- Performance benchmarking
"""

import time
from pathlib import Path


def test_hardware_detection():
    """Test hardware detection capabilities."""
    print("🖥️  Testing Hardware Detection...")
    print("=" * 60)

    try:
        from chunking_strategy.core.hardware import get_hardware_info, get_optimal_batch_config

        # Get hardware info
        hardware = get_hardware_info()

        print(f"✅ Hardware detected successfully!")
        print(f"   CPU cores: {hardware.cpu_count} logical, {hardware.cpu_count_physical} physical")
        print(f"   Memory: {hardware.memory_total_gb:.1f}GB total, {hardware.memory_available_gb:.1f}GB available")
        print(f"   GPUs: {hardware.gpu_count}")
        if hardware.gpu_count > 0:
            for i, name in enumerate(hardware.gpu_names):
                print(f"     GPU {i}: {name}")

        print(f"\n💡 Recommendations:")
        print(f"   Batch size: {hardware.recommended_batch_size}")
        print(f"   Workers: {hardware.recommended_workers}")
        print(f"   Use GPU: {hardware.use_gpu}")

        # Test batch configuration
        config = get_optimal_batch_config(
            total_files=100,
            avg_file_size_mb=2.5
        )
        print(f"\n🔧 Optimal config for 100 files (2.5MB each):")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Workers: {config['workers']}")
        print(f"   Estimated memory: {config['total_estimated_memory_mb']:.1f}MB")

        return True

    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        return False


def test_pdf_processing():
    """Test PDF processing with multiple backends."""
    print("\n📄 Testing PDF Processing...")
    print("=" * 60)

    pdf_file = Path("test_data/example.pdf")
    if not pdf_file.exists():
        print(f"❌ PDF file not found: {pdf_file}")
        return False

    try:
        import chunking_strategy

        results = {}
        backends = ["pymupdf", "pypdf2", "pdfminer"]

        for backend in backends:
            print(f"\n🔄 Testing {backend} backend...")
            try:
                chunker = chunking_strategy.create_chunker(
                    "pdf_chunker",
                    backend=backend,
                    pages_per_chunk=1,
                    extract_images=(backend == "pymupdf"),  # Only PyMuPDF supports images
                    extract_tables=(backend == "pymupdf")
                )

                start_time = time.time()
                result = chunker.chunk(pdf_file)
                processing_time = time.time() - start_time

                # Analyze results
                chunk_types = {}
                for chunk in result.chunks:
                    chunk_type = chunk.metadata.extra.get('chunk_type', 'text')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

                results[backend] = {
                    'chunks': len(result.chunks),
                    'time': processing_time,
                    'types': chunk_types
                }

                print(f"   ✅ {backend}: {len(result.chunks)} chunks in {processing_time:.3f}s")
                print(f"      Types: {chunk_types}")

            except Exception as e:
                print(f"   ❌ {backend}: {e}")
                results[backend] = {'error': str(e)}

        # Summary
        print(f"\n📊 PDF Processing Summary:")
        for backend, data in results.items():
            if 'error' not in data:
                print(f"   {backend:10}: {data['chunks']:2d} chunks, {data['time']:.3f}s")

        return len([r for r in results.values() if 'error' not in r]) > 0

    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing with multiple files."""
    print("\n📦 Testing Batch Processing...")
    print("=" * 60)

    # Collect test files
    test_files = []
    test_data_dir = Path("test_data")

    if test_data_dir.exists():
        for file_path in test_data_dir.glob("*.txt"):
            test_files.append(file_path)

        # Add PDF if it exists
        pdf_file = test_data_dir / "example.pdf"
        if pdf_file.exists():
            test_files.append(pdf_file)

    if not test_files:
        print("❌ No test files found")
        return False

    print(f"📁 Found {len(test_files)} test files:")
    for f in test_files[:5]:  # Show first 5
        print(f"   {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    if len(test_files) > 5:
        print(f"   ... and {len(test_files) - 5} more")

    try:
        from chunking_strategy.core.batch import BatchProcessor

        # Test different modes
        modes = ["sequential", "thread", "process"]
        results = {}

        for mode in modes:
            print(f"\n🔄 Testing {mode} mode...")

            processor = BatchProcessor()

            start_time = time.time()
            result = processor.process_files(
                files=test_files[:min(5, len(test_files))],  # Limit to 5 files for testing
                default_strategy="fixed_size",
                default_params={"chunk_size": 1000},
                parallel_mode=mode,
                workers=2 if mode != "sequential" else 1
            )

            results[mode] = {
                'files': len(result.successful_files),
                'chunks': result.total_chunks,
                'time': result.total_processing_time,
                'speed': result.files_per_second
            }

            print(f"   ✅ {mode}: {result.total_files} files → {result.total_chunks} chunks")
            print(f"      Time: {result.total_processing_time:.3f}s, Speed: {result.files_per_second:.1f} files/s")

            if result.failed_files:
                print(f"      Failed: {len(result.failed_files)} files")

        # Performance comparison
        print(f"\n🚀 Performance Comparison:")
        for mode, data in results.items():
            print(f"   {mode:10}: {data['speed']:6.1f} files/s, {data['time']:.3f}s total")

        return True

    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_commands():
    """Test CLI commands with the new features."""
    print("\n💻 Testing CLI Commands...")
    print("=" * 60)

    try:
        import subprocess
        import sys

        # Test hardware command
        print("🔄 Testing hardware detection command...")
        result = subprocess.run([
            sys.executable, "-m", "chunking_strategy.cli", "hardware", "--recommendations"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("   ✅ Hardware command works")
            # Show first few lines
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"      {line}")
        else:
            print(f"   ❌ Hardware command failed: {result.stderr}")

        # Test list strategies command
        print("\n🔄 Testing list strategies command...")
        result = subprocess.run([
            sys.executable, "-m", "chunking_strategy.cli", "list-strategies"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("   ✅ List strategies command works")
            strategies = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            print(f"      Found {len(strategies)} strategies")
        else:
            print(f"   ❌ List strategies failed: {result.stderr}")

        return True

    except Exception as e:
        print(f"❌ CLI testing failed: {e}")
        return False


def run_performance_benchmark():
    """Run a comprehensive performance benchmark."""
    print("\n🏃 Running Performance Benchmark...")
    print("=" * 60)

    try:
        from chunking_strategy.core.hardware import get_hardware_info
        import chunking_strategy

        hardware = get_hardware_info()

        # Test file
        test_file = Path("test_data/alice_wonderland.txt")
        if not test_file.exists():
            print("❌ Benchmark file not found")
            return False

        file_size_mb = test_file.stat().st_size / (1024 * 1024)
        print(f"📄 Benchmark file: {test_file.name} ({file_size_mb:.1f} MB)")

        # Test different strategies
        strategies = [
            ("fixed_size", {"chunk_size": 1000}),
            ("sentence_based", {"max_sentences": 3}),
            ("paragraph_based", {"max_paragraphs": 1}),
        ]

        results = []

        for strategy_name, params in strategies:
            print(f"\n🔄 Benchmarking {strategy_name}...")

            try:
                chunker = chunking_strategy.create_chunker(strategy_name, **params)

                start_time = time.time()
                result = chunker.chunk(test_file)
                processing_time = time.time() - start_time

                mb_per_sec = file_size_mb / processing_time
                chunks_per_sec = len(result.chunks) / processing_time

                results.append({
                    'strategy': strategy_name,
                    'chunks': len(result.chunks),
                    'time': processing_time,
                    'mb_per_sec': mb_per_sec,
                    'chunks_per_sec': chunks_per_sec
                })

                print(f"   ✅ {strategy_name}: {len(result.chunks)} chunks in {processing_time:.3f}s")
                print(f"      Speed: {mb_per_sec:.1f} MB/s, {chunks_per_sec:.1f} chunks/s")

            except Exception as e:
                print(f"   ❌ {strategy_name}: {e}")

        # Summary
        print(f"\n🏆 Benchmark Results:")
        print(f"   Hardware: {hardware.cpu_count} cores, {hardware.memory_total_gb:.1f}GB RAM")
        print(f"   {'Strategy':<15} {'Chunks':<8} {'Time(s)':<8} {'MB/s':<8} {'Chunks/s':<10}")
        print(f"   {'-'*55}")

        for r in results:
            print(f"   {r['strategy']:<15} {r['chunks']:<8} {r['time']:<8.3f} {r['mb_per_sec']:<8.1f} {r['chunks_per_sec']:<10.1f}")

        return True

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def main():
    """Run comprehensive test suite."""
    print("🚀 Comprehensive Chunking Strategy Test Suite")
    print("=" * 80)

    tests = [
        ("Hardware Detection", test_hardware_detection),
        ("PDF Processing", test_pdf_processing),
        ("Batch Processing", test_batch_processing),
        ("CLI Commands", test_cli_commands),
        ("Performance Benchmark", run_performance_benchmark),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Final summary
    print("\n" + "=" * 80)
    print("📊 Test Suite Summary:")
    print("=" * 80)

    passed = 0
    total = len(tests)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:<25}: {status}")
        if success:
            passed += 1

    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The chunking library is working perfectly!")
        return 0
    else:
        print("💥 Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())
