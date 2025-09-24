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
    """Document chunking example (PDF or text)."""
    print("\n🎯 Example 3: Document Chunking")
    print("=" * 50)

    orchestrator = ChunkerOrchestrator()

    # Try different document files
    test_documents = [
        "test_data/example.pdf",
        "test_data/sample_article.txt",
        "test_data/business_report.txt",
        "test_data/technical_doc.txt"
    ]

    document_found = False
    for doc_path in test_documents:
        if Path(doc_path).exists():
            try:
                result = orchestrator.chunk_file(doc_path)
                print(f"📑 Document: {Path(doc_path).name}")
                print(f"📊 Strategy used: {result.strategy_used}")
                print(f"📄 Chunks created: {len(result.chunks)}")
                print(f"📝 Sample chunk: {result.chunks[0].content[:100]}...")
                document_found = True
                break
            except Exception as e:
                print(f"⚠️  Could not process {Path(doc_path).name}: {e}")
                continue

    if not document_found:
        print("ℹ️  No suitable document files found for this example")


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
