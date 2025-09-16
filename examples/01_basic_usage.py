#!/usr/bin/env python3
"""
Basic Usage Examples - Getting Started with Chunking Strategy

This script demonstrates the fundamental usage patterns of the chunking library.
Perfect for new users who want to understand the core concepts.

Run with: python examples/01_basic_usage.py
"""

from pathlib import Path
from chunking_strategy import ChunkerOrchestrator, create_chunker

def example_1_simple_text_chunking():
    """Most basic chunking example."""
    print("\n🎯 Example 1: Simple Text Chunking")
    print("=" * 50)

    orchestrator = ChunkerOrchestrator()

    # Chunk a text file
    result = orchestrator.chunk_file("test_data/short.txt")

    print(f"📝 Generated {len(result.chunks)} chunks using {result.strategy_used}")
    print(f"📄 First chunk: {result.chunks[0].content[:100]}...")


def example_2_different_strategies():
    """Show different chunking strategies."""
    print("\n🎯 Example 2: Different Strategies")
    print("=" * 50)

    # Use orchestrator to properly handle different strategies
    orchestrator = ChunkerOrchestrator()
    strategies = ["fixed_size", "sentence_based", "paragraph_based"]

    for strategy in strategies:
        try:
            result = orchestrator.chunk_file("test_data/alice_wonderland.txt", strategy=strategy)
            print(f"📊 {strategy}: {len(result.chunks)} chunks")
        except Exception as e:
            print(f"❌ {strategy}: Failed ({e})")


def example_3_pdf_chunking():
    """PDF chunking example."""
    print("\n🎯 Example 3: PDF Chunking")
    print("=" * 50)

    orchestrator = ChunkerOrchestrator()
    pdf_path = "test_data/example.pdf"

    if Path(pdf_path).exists():
        result = orchestrator.chunk_file(pdf_path)
        print(f"📑 PDF chunks: {len(result.chunks)}")
        print(f"📄 Sample chunk: {result.chunks[0].content[:100]}...")
    else:
        print("❌ PDF file not found - skipping PDF example")


def example_4_batch_processing():
    """Batch processing multiple files."""
    print("\n🎯 Example 4: Batch Processing")
    print("=" * 50)

    orchestrator = ChunkerOrchestrator()
    test_files = list(Path("test_data").glob("*.txt"))[:3]  # First 3 txt files

    for file_path in test_files:
        result = orchestrator.chunk_file(str(file_path))
        print(f"📁 {file_path.name}: {len(result.chunks)} chunks")


def main():
    """Run all basic examples."""
    print("🚀 BASIC USAGE EXAMPLES")
    print("=" * 60)

    example_1_simple_text_chunking()
    example_2_different_strategies()
    example_3_pdf_chunking()
    example_4_batch_processing()

    print("\n✅ All basic examples completed!")
    print("💡 Next: Try advanced_usage.py for more complex scenarios")


if __name__ == "__main__":
    main()
