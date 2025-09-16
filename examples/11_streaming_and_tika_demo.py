#!/usr/bin/env python3
"""
Comprehensive demonstration of streaming and Tika integration.

This demo shows:
1. Intelligent streaming decisions based on file size
2. Tika integration for enhanced document processing
3. Configuration-driven chunking with streaming options
4. Performance comparison between regular and streaming processing
5. Large file simulation without creating actual large files
"""

import tempfile
import time
from pathlib import Path
import logging

from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.core.hardware import get_smart_parallelization_config, configure_smart_parallelization

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_files():
    """Create various test files for demonstration."""
    files = {}

    # Small text file (will NOT use streaming)
    small_file = Path(tempfile.mktemp(suffix='.txt'))
    with open(small_file, 'w') as f:
        f.write("This is a small text file. " * 50)  # ~1.2KB
    files['small_text'] = small_file

    # Medium text file (might use streaming based on configuration)
    medium_file = Path(tempfile.mktemp(suffix='.txt'))
    with open(medium_file, 'w') as f:
        f.write("This is a medium-sized text file with more content. " * 2000)  # ~110KB
    files['medium_text'] = medium_file

    # Mock PDF file (will use Tika if available)
    pdf_file = Path(tempfile.mktemp(suffix='.pdf'))
    with open(pdf_file, 'wb') as f:
        f.write(b'%PDF-1.4\n')  # Minimal PDF header
        f.write(b'Mock PDF content for demonstration. ' * 1000)
    files['mock_pdf'] = pdf_file

    # Mock Word document (will use Tika if available)
    docx_file = Path(tempfile.mktemp(suffix='.docx'))
    with open(docx_file, 'wb') as f:
        f.write(b'PK\x03\x04')  # Minimal DOCX/ZIP header
        f.write(b'Mock DOCX content for demonstration. ' * 800)
    files['mock_docx'] = docx_file

    return files


def demo_smart_streaming_decisions():
    """Demonstrate intelligent streaming decisions."""
    print("\n🌊 === SMART STREAMING DECISIONS DEMO ===")

    files = create_test_files()
    orchestrator = ChunkerOrchestrator(enable_smart_parallelization=True)
    config = get_smart_parallelization_config()

    print(f"Current streaming threshold: {config.min_file_size_for_streaming:,} bytes (100MB)")

    for name, file_path in files.items():
        file_size = file_path.stat().st_size
        should_stream = config.should_use_streaming(file_size)

        print(f"\n📁 {name}: {file_size:,} bytes")
        print(f"   Should use streaming: {'✅ YES' if should_stream else '❌ NO'}")

        # Chunk the file
        start_time = time.time()
        result = orchestrator.chunk_file(file_path)
        elapsed_time = time.time() - start_time

        print(f"   Chunks created: {len(result.chunks)}")
        print(f"   Processing time: {elapsed_time:.3f}s")
        print(f"   Strategy used: {result.strategy_used}")
        print(f"   Streaming used: {'✅ YES' if result.source_info.get('streaming_used') else '❌ NO'}")

        if result.source_info.get('tika_available'):
            print(f"   Tika available: ✅ YES")

        # Cleanup
        file_path.unlink()


def demo_tika_integration():
    """Demonstrate Tika integration for document processing."""
    print("\n📄 === TIKA INTEGRATION DEMO ===")

    files = create_test_files()
    orchestrator = ChunkerOrchestrator()

    # Test different document types
    document_types = ['mock_pdf', 'mock_docx']

    for doc_type in document_types:
        if doc_type not in files:
            continue

        file_path = files[doc_type]
        print(f"\n📋 Testing {doc_type}: {file_path.suffix}")

        # Analyze file characteristics
        file_info = orchestrator._analyze_file(file_path)
        print(f"   File type detected: {file_info.get('file_type')}")
        print(f"   Modality: {file_info.get('modality')}")
        print(f"   Tika available: {'✅ YES' if file_info.get('tika_available') else '❌ NO'}")

        if file_info.get('tika_mime_type'):
            print(f"   Tika MIME type: {file_info['tika_mime_type']}")

        if file_info.get('tika_reason'):
            print(f"   Tika reason: {file_info['tika_reason']}")

        # Chunk the document
        result = orchestrator.chunk_file(file_path)
        print(f"   Chunks created: {len(result.chunks)}")
        print(f"   Strategy used: {result.strategy_used}")

        # Cleanup
        file_path.unlink()


def demo_force_streaming():
    """Demonstrate forcing streaming for small files."""
    print("\n⚡ === FORCE STREAMING DEMO ===")

    # Create a small file
    small_file = Path(tempfile.mktemp(suffix='.txt'))
    with open(small_file, 'w') as f:
        f.write("Small file content for forced streaming demo. " * 100)  # ~4.4KB

    try:
        orchestrator = ChunkerOrchestrator(enable_smart_parallelization=True)
        file_size = small_file.stat().st_size

        print(f"📁 Small file: {file_size:,} bytes")

        # Normal processing (should NOT use streaming)
        print("\n🔄 Normal processing:")
        result1 = orchestrator.chunk_file(small_file)
        print(f"   Chunks: {len(result1.chunks)}")
        print(f"   Streaming used: {'✅ YES' if result1.source_info.get('streaming_used') else '❌ NO'}")

        # Force streaming (WILL use streaming)
        print("\n⚡ Forced streaming:")
        result2 = orchestrator.chunk_file(small_file, force_streaming=True)
        print(f"   Chunks: {len(result2.chunks)}")
        print(f"   Streaming used: {'✅ YES' if result2.source_info.get('streaming_used') else '❌ NO'}")

        if result2.source_info.get('streaming_config'):
            config = result2.source_info['streaming_config']
            print(f"   Streaming block size: {config['block_size']:,} bytes")
            print(f"   Streaming overlap: {config['overlap_size']:,} bytes")

    finally:
        small_file.unlink()


def demo_custom_streaming_parameters():
    """Demonstrate custom streaming parameters."""
    print("\n⚙️ === CUSTOM STREAMING PARAMETERS DEMO ===")

    # Create a test file
    test_file = Path(tempfile.mktemp(suffix='.txt'))
    with open(test_file, 'w') as f:
        f.write("Content for custom streaming parameters demo. " * 200)  # ~8.8KB

    try:
        orchestrator = ChunkerOrchestrator()

        print(f"📁 Test file: {test_file.stat().st_size:,} bytes")

        # Use custom streaming parameters
        result = orchestrator.chunk_file(
            test_file,
            force_streaming=True,
            streaming_block_size=32 * 1024,  # 32KB blocks
            streaming_overlap_size=4 * 1024  # 4KB overlap
        )

        print(f"   Chunks created: {len(result.chunks)}")
        print(f"   Streaming used: {'✅ YES' if result.source_info.get('streaming_used') else '❌ NO'}")

        if result.source_info.get('streaming_config'):
            config = result.source_info['streaming_config']
            print(f"   Custom block size: {config['block_size']:,} bytes")
            print(f"   Custom overlap: {config['overlap_size']:,} bytes")

    finally:
        test_file.unlink()


def demo_configuration_file():
    """Demonstrate using configuration file with streaming settings."""
    print("\n📋 === CONFIGURATION FILE DEMO ===")

    # Create a test configuration file
    config_file = Path(tempfile.mktemp(suffix='.yaml'))
    config_content = """
profile_name: "streaming_demo"

strategy_selection:
  ".txt":
    primary: "sentence_based"
    fallbacks: ["paragraph_based", "fixed_size"]

# Smart parallelization with custom streaming thresholds
parallelization:
  smart_parallelization: true
  min_file_size_for_streaming: 50000  # 50KB (lower threshold for demo)

# Tika integration
tika_integration:
  enabled: true
  auto_tika_formats: [".pdf", ".docx", ".doc"]
"""

    with open(config_file, 'w') as f:
        f.write(config_content)

    try:
        # Load orchestrator with custom configuration
        orchestrator = ChunkerOrchestrator(config_path=config_file)
        print(f"✅ Loaded configuration from: {config_file}")

        # Create test files
        files = create_test_files()

        # Test with the custom configuration
        for name, file_path in files.items():
            file_size = file_path.stat().st_size

            result = orchestrator.chunk_file(file_path)

            print(f"\n📁 {name}: {file_size:,} bytes")
            print(f"   Chunks: {len(result.chunks)}")
            print(f"   Strategy: {result.strategy_used}")
            print(f"   Streaming: {'✅ YES' if result.source_info.get('streaming_used') else '❌ NO'}")

            file_path.unlink()

    finally:
        config_file.unlink()


def demo_performance_comparison():
    """Compare performance between regular and streaming processing."""
    print("\n🏁 === PERFORMANCE COMPARISON DEMO ===")

    # Create a larger test file
    large_file = Path(tempfile.mktemp(suffix='.txt'))
    content = "This is a larger file for performance comparison. " * 5000  # ~265KB
    with open(large_file, 'w') as f:
        f.write(content)

    try:
        orchestrator = ChunkerOrchestrator(enable_smart_parallelization=True)
        file_size = large_file.stat().st_size

        print(f"📁 Test file: {file_size:,} bytes")

        # Regular processing with streaming disabled
        print("\n⏱️  Regular processing:")
        start_time = time.time()
        result1 = orchestrator.chunk_file(large_file, disable_streaming=True)
        regular_time = time.time() - start_time

        print(f"   Chunks: {len(result1.chunks)}")
        print(f"   Time: {regular_time:.3f}s")
        print(f"   Streaming: {'✅ YES' if result1.source_info.get('streaming_used') else '❌ NO'}")

        # Streaming processing
        print("\n⚡ Streaming processing:")
        start_time = time.time()
        result2 = orchestrator.chunk_file(large_file, force_streaming=True)
        streaming_time = time.time() - start_time

        print(f"   Chunks: {len(result2.chunks)}")
        print(f"   Time: {streaming_time:.3f}s")
        print(f"   Streaming: {'✅ YES' if result2.source_info.get('streaming_used') else '❌ NO'}")

        # Compare
        if regular_time > 0 and streaming_time > 0:
            speedup = regular_time / streaming_time
            print(f"\n📊 Performance comparison:")
            print(f"   Regular: {regular_time:.3f}s")
            print(f"   Streaming: {streaming_time:.3f}s")
            print(f"   Speedup: {speedup:.2f}x {'(streaming faster)' if speedup > 1 else '(regular faster)'}")

    finally:
        large_file.unlink()


def main():
    """Run all streaming and Tika integration demos."""
    print("🚀 === STREAMING AND TIKA INTEGRATION COMPREHENSIVE DEMO ===")
    print("This demo showcases intelligent file processing with streaming and Tika integration.")

    try:
        # Demo 1: Smart streaming decisions
        demo_smart_streaming_decisions()

        # Demo 2: Tika integration
        demo_tika_integration()

        # Demo 3: Force streaming
        demo_force_streaming()

        # Demo 4: Custom streaming parameters
        demo_custom_streaming_parameters()

        # Demo 5: Configuration file
        demo_configuration_file()

        # Demo 6: Performance comparison
        demo_performance_comparison()

        print("\n🎉 === DEMO COMPLETED SUCCESSFULLY! ===")
        print("\n📋 Summary of Features Demonstrated:")
        print("   ✅ Smart streaming decisions based on file size")
        print("   ✅ Tika integration for enhanced document processing")
        print("   ✅ Force streaming and disable streaming options")
        print("   ✅ Custom streaming parameters (block size, overlap)")
        print("   ✅ Configuration file support")
        print("   ✅ Performance comparison between regular and streaming")

        print("\n💡 Key Benefits:")
        print("   🔹 Memory efficient processing of large files (100GB+)")
        print("   🔹 Intelligent decisions prevent overhead on small files")
        print("   🔹 Enhanced document extraction with Apache Tika")
        print("   🔹 Configurable thresholds and parameters")
        print("   🔹 Graceful fallbacks when components fail")
        print("   🔹 Comprehensive metadata and progress tracking")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
