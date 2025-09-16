#!/usr/bin/env python3
"""
Advanced Usage Examples - Complex Chunking Scenarios

This script demonstrates advanced features like custom configurations,
specialized chunkers, and complex workflows.

Run with: python examples/02_advanced_usage.py
"""

import yaml
from pathlib import Path
from chunking_strategy import (
    ChunkerOrchestrator,
    create_chunker,
    apply_universal_strategy,
    get_universal_strategy_registry,
)

def example_1_custom_configuration():
    """Using custom YAML configuration."""
    print("\n🎯 Example 1: Custom Configuration")
    print("=" * 50)

    config_path = "config_examples/quality_focused.yaml"
    if Path(config_path).exists():
        orchestrator = ChunkerOrchestrator(config_path=config_path)
        result = orchestrator.chunk_file("test_data/technical_doc.txt")
        print(f"📊 Quality-focused chunking: {len(result.chunks)} chunks")
    else:
        print("❌ Config file not found")


def example_2_code_chunking():
    """Specialized code chunking."""
    print("\n🎯 Example 2: Code Chunking")
    print("=" * 50)

    code_files = {
        "test_data/sample_code.py": "python_code",
        "test_data/sample_code.cpp": "c_cpp_code",
    }

    for file_path, strategy in code_files.items():
        if Path(file_path).exists():
            try:
                # For code chunking, use orchestrator which handles API properly
                orchestrator = ChunkerOrchestrator()
                result = orchestrator.chunk_file(file_path, strategy=strategy)
                print(f"💻 {file_path}: {len(result.chunks)} code chunks")
            except Exception as e:
                print(f"❌ {file_path}: Failed ({e})")


def example_3_universal_strategies():
    """Universal strategy application."""
    print("\n🎯 Example 3: Universal Strategies")
    print("=" * 50)

    # Show available universal strategies
    registry = get_universal_strategy_registry()
    print(f"🔧 Available strategies: {registry.list_strategies()}")

    # Apply paragraph strategy to different file types
    files = ["test_data/technical_doc.txt", "test_data/sample_code.py"]

    for file_path in files:
        if Path(file_path).exists():
            result = apply_universal_strategy("paragraph", file_path)
            print(f"📄 {Path(file_path).name}: {len(result.chunks)} paragraphs")


def example_4_auto_strategy_selection():
    """Automatic strategy selection."""
    print("\n🎯 Example 4: Auto Strategy Selection")
    print("=" * 50)

    # Use auto strategy with different files
    orchestrator = ChunkerOrchestrator()

    test_files = [
        "test_data/technical_doc.txt",
        "test_data/sample_code.py",
        "test_data/example.pdf"
    ]

    for file_path in test_files:
        if Path(file_path).exists():
            result = orchestrator.chunk_file(file_path, strategy="auto")
            print(f"🤖 {Path(file_path).name}: {result.strategy_used} → {len(result.chunks)} chunks")


def example_5_complex_pipeline():
    """Complex processing pipeline."""
    print("\n🎯 Example 5: Complex Pipeline")
    print("=" * 50)

    # Process multiple files with different strategies
    orchestrator = ChunkerOrchestrator()

    pipeline_config = {
        "*.txt": "sentence_based",
        "*.py": "python_code",
        "*.pdf": "pdf_chunker"
    }

    results = {}
    for pattern, strategy in pipeline_config.items():
        files = list(Path("test_data").glob(pattern))[:2]  # Limit for demo

        for file_path in files:
            try:
                result = orchestrator.chunk_file(str(file_path), strategy=strategy)
                results[file_path.name] = {
                    "strategy": strategy,
                    "chunks": len(result.chunks)
                }
            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")

    print("📊 Pipeline Results:")
    for filename, info in results.items():
        print(f"   {filename}: {info['strategy']} → {info['chunks']} chunks")


def main():
    """Run all advanced examples."""
    print("🚀 ADVANCED USAGE EXAMPLES")
    print("=" * 60)

    example_1_custom_configuration()
    example_2_code_chunking()
    example_3_universal_strategies()
    example_4_auto_strategy_selection()
    example_5_complex_pipeline()

    print("\n✅ All advanced examples completed!")
    print("💡 Next: Try embedding_workflows.py for ML integration")


if __name__ == "__main__":
    main()
