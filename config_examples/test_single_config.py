#!/usr/bin/env python3
"""
Simple Single Config Tester

This tool tests one configuration file quickly and simply.

Usage:
    python test_single_config.py [--config path/to/config.yaml]

If no config is specified, it picks a random one from available configs.
User can easily modify the TARGET_CONFIG_FILE below to test their own config.
"""

import sys
import random
import tempfile
from pathlib import Path

# Add the parent directory to sys.path to import chunking_strategy
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from chunking_strategy import ChunkerOrchestrator
except ImportError as e:
    print(f"❌ Error importing chunking_strategy: {e}")
    print("Make sure you're running this from the project root.")
    sys.exit(1)

# 🎯 TARGET CONFIG FILE - EDIT THIS TO TEST YOUR CONFIG
TARGET_CONFIG_FILE = "basic_configs/basic_example.yaml"


def get_all_config_files():
    """Get all available config files."""
    config_dir = Path(__file__).parent
    config_files = []

    # Search in subdirectories
    for subdir in ["basic_configs", "strategy_configs", "format_specific_configs",
                   "use_case_configs", "advanced_configs", "custom_algorithms"]:
        subdir_path = config_dir / subdir
        if subdir_path.exists():
            config_files.extend(list(subdir_path.glob("*.yaml")))

    # Also check root directory
    config_files.extend(list(config_dir.glob("*.yaml")))

    return config_files


def create_simple_test_file():
    """Create a simple test file."""
    temp_dir = Path(tempfile.mkdtemp())
    test_file = temp_dir / "test.txt"

    content = """
This is a simple test document for chunking validation.

It contains multiple paragraphs to test how the chunking strategy works.
Each paragraph has different content to help evaluate the chunking behavior.

The first paragraph introduces the document and explains its purpose.
This helps test how the chunker handles introductory content.

The second paragraph contains technical information about chunking strategies.
Modern chunking approaches use various techniques like fixed-size, semantic,
and boundary-aware methods to divide text into meaningful segments.

The third paragraph discusses practical applications.
These include information retrieval, document analysis, and natural language processing tasks.
The effectiveness of chunking directly impacts downstream processing quality.

This final paragraph concludes the test document.
It provides a natural ending point for the chunking process.
"""

    test_file.write_text(content.strip())
    return test_file, temp_dir


def test_config(config_path):
    """Test a single configuration file."""
    print(f"🎯 Testing config: {config_path.name}")
    print(f"📁 Full path: {config_path}")
    print()

    # Create test file
    test_file, temp_dir = create_simple_test_file()

    try:
        # Initialize orchestrator
        print("🏗️  Initializing orchestrator...")
        orchestrator = ChunkerOrchestrator(config_path=config_path)
        print("✅ Orchestrator initialized successfully")

        # Process file
        print("📄 Processing test file...")
        result = orchestrator.chunk_file(str(test_file))

        # Show results
        if result and result.chunks:
            chunk_count = len(result.chunks)
            total_chars = sum(len(chunk.content) for chunk in result.chunks)
            avg_chunk_size = total_chars / chunk_count if chunk_count > 0 else 0

            print(f"✅ Successfully created {chunk_count} chunks")
            print(f"📊 Average chunk size: {avg_chunk_size:.0f} characters")

            if chunk_count <= 5:  # Show chunks if there aren't too many
                print("\n📝 Generated chunks:")
                for i, chunk in enumerate(result.chunks, 1):
                    preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                    print(f"   {i}. {preview}")

            print("\n🎉 Config test PASSED!")
            return True
        else:
            print("❌ No chunks generated")
            return False

    except Exception as e:
        print(f"❌ Config test FAILED: {e}")
        return False

    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test a single configuration file")
    parser.add_argument("--config", "-c", type=str, help="Path to config file")
    args = parser.parse_args()

    config_dir = Path(__file__).parent

    # Determine which config to test
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = config_dir / config_path
    else:
        # Use target config or pick random if target doesn't exist
        target_path = config_dir / TARGET_CONFIG_FILE
        if target_path.exists():
            config_path = target_path
        else:
            # Pick a random config
            available_configs = get_all_config_files()
            if not available_configs:
                print("❌ No config files found!")
                sys.exit(1)
            config_path = random.choice(available_configs)
            print(f"🎲 No config specified, picked random: {config_path.relative_to(config_dir)}")

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print(f"💡 Available configs:")
        for config in get_all_config_files()[:5]:  # Show first 5
            print(f"   - {config.relative_to(config_dir)}")
        sys.exit(1)

    print("🧪 Simple Config Tester")
    print("=" * 40)
    print(f"💡 To test your own config: Edit TARGET_CONFIG_FILE in this script")
    print(f"💡 Or use: python {Path(__file__).name} --config your_config.yaml")
    print()

    success = test_config(config_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
