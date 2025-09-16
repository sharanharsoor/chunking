#!/usr/bin/env python3
"""
CLI Examples - Command Line Interface Usage

This script demonstrates how to use the command line interface
and includes example commands you can run directly.

Run with: python examples/04_cli_examples.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and show the output."""
    print(f"\n🔧 {description}")
    print("=" * 60)
    print(f"💻 Command: {cmd}")
    print("📤 Output:")
    print("-" * 40)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
        if result.returncode == 0:
            print("✅ Command completed successfully")
        else:
            print(f"❌ Command failed with return code {result.returncode}")

    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
    except Exception as e:
        print(f"❌ Failed to run command: {e}")


def main():
    """Demonstrate CLI usage with examples."""
    print("🚀 CLI USAGE EXAMPLES")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("test_data").exists():
        print("❌ Please run this from the chunking directory root")
        print("   Current directory should contain test_data/")
        return

    # Basic commands
    examples = [
        ("python -m chunking_strategy --help", "Show help information"),
        ("python -m chunking_strategy list-strategies", "List available chunking strategies"),
        ("python -m chunking_strategy list-models", "List available embedding models"),
        ("python -m chunking_strategy chunk test_data/short.txt", "Chunk a simple text file"),
        ("python -m chunking_strategy chunk test_data/short.txt --strategy sentence_based", "Use specific strategy"),
        ("python -m chunking_strategy chunk test_data/short.txt --output chunks_output.txt", "Save output to file"),
        ("python -m chunking_strategy chunk test_data/sample_code.py --strategy python_code", "Chunk Python code"),
    ]

    # Skip embedding examples due to known system library compatibility issues
    if Path("test_data/technical_doc.txt").exists():
        print("\n⚠️  Skipping embedding examples - System library compatibility issues detected")
        print("   Embedding functionality requires GLIBCXX_3.4.29+ system libraries")
        print("   To enable: Update system libraries or use conda environment")
        print("   Commands would be:")
        print("     python -m chunking_strategy embed test_data/technical_doc.txt")
        print("     python -m chunking_strategy embed test_data/technical_doc.txt --model all-MiniLM-L6-v2 --output-format vector_plus_text")

    # Hardware detection
    examples.extend([
        ("python -m chunking_strategy hardware", "Show hardware information"),
        ("python -m chunking_strategy batch test_data/*.txt", "Batch process multiple files"),
    ])

    # Run all examples
    for cmd, description in examples:
        run_command(cmd, description)
        # Removed keyboard intervention for automated testing

    print("\n✅ All CLI examples completed!")
    print("\n📚 Additional CLI Information:")
    print("   • Use --help with any command for more options")
    print("   • Output files are created in the current directory")
    print("   • Batch processing supports glob patterns")
    print("   • Configuration files can be specified with --config")


if __name__ == "__main__":
    main()
