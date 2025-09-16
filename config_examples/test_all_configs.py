#!/usr/bin/env python3
"""
Comprehensive Config Tester for All Example Configurations

This standalone program tests all configuration files in the config_examples
directory and its subdirectories, exactly like the integration test does.

Usage:
    python test_all_configs.py [--verbose] [--timeout SECONDS]

Options:
    --verbose    Show detailed output for each config test
    --timeout    Set timeout in seconds for each config test (default: 30)

This tool helps validate that all example configurations work correctly
without running the full test suite.
"""

import sys
import os
import yaml
import tempfile
import signal
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the parent directory to sys.path to import chunking_strategy
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from chunking_strategy import ChunkerOrchestrator
    from chunking_strategy.core.base import ModalityType
except ImportError as e:
    print(f"❌ Error importing chunking_strategy: {e}")
    print("Make sure you're running this from the project root or chunking_strategy is installed.")
    sys.exit(1)


class ConfigTester:
    """Test all configuration files comprehensively."""

    def __init__(self, verbose: bool = False, timeout: int = 30):
        self.verbose = verbose
        self.timeout = timeout
        self.config_dir = Path(__file__).parent
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_files = self._create_test_files()
        self.config_files = self._get_all_config_files()

        print(f"🔍 Found {len(self.config_files)} configuration files")
        if self.verbose:
            print("📂 Config files to test:")
            for config_file in self.config_files:
                print(f"   - {config_file.relative_to(self.config_dir)}")

    def _get_all_config_files(self) -> List[Path]:
        """Get all config files from subdirectories."""
        all_config_files = []

        # Search in subdirectories
        for subdir in ["basic_configs", "strategy_configs", "format_specific_configs",
                       "use_case_configs", "advanced_configs", "custom_algorithms"]:
            subdir_path = self.config_dir / subdir
            if subdir_path.exists():
                all_config_files.extend(list(subdir_path.glob("*.yaml")))

        # Also check for any remaining files in the root directory
        all_config_files.extend(list(self.config_dir.glob("*.yaml")))

        return sorted(all_config_files)

    def _create_test_files(self) -> Dict[str, Path]:
        """Create test files of various types."""
        files = {}

        # Python file
        python_content = '''
def process_data(data):
    """Process input data."""
    if not data:
        return []

    results = []
    for item in data:
        processed = item.strip().upper()
        if processed:
            results.append(processed)

    return results

class DataProcessor:
    """A simple data processor class."""

    def __init__(self):
        self.processed_count = 0

    def process(self, items):
        """Process a list of items."""
        self.processed_count += len(items)
        return [item.lower() for item in items if item]
'''
        files['python'] = self.temp_dir / "test.py"
        files['python'].write_text(python_content)

        # Text file
        text_content = """
This is a sample text file for testing various chunking strategies.
It contains multiple paragraphs to test different chunking approaches.

The first paragraph discusses the importance of text processing in natural language processing.
Modern NLP systems rely heavily on effective text chunking strategies.

The second paragraph explores different chunking methodologies.
Fixed-size chunking, semantic chunking, and boundary-aware chunking each have their merits.

The third paragraph concludes with practical applications.
These techniques are widely used in information retrieval, document analysis, and AI systems.
"""
        files['text'] = self.temp_dir / "test.txt"
        files['text'].write_text(text_content)

        # Markdown file
        markdown_content = """
# Sample Markdown Document

## Introduction
This is a **markdown** document for testing purposes.

## Features
- Supports *italics* and **bold** text
- Has multiple sections
- Contains code blocks

```python
def hello_world():
    print("Hello, World!")
```

## Conclusion
This document tests markdown parsing capabilities.
"""
        files['markdown'] = self.temp_dir / "test.md"
        files['markdown'].write_text(markdown_content)

        # JSON file
        json_content = """{
    "users": [
        {
            "id": 1,
            "name": "Alice",
            "email": "alice@example.com",
            "preferences": {
                "theme": "dark",
                "notifications": true
            }
        },
        {
            "id": 2,
            "name": "Bob",
            "email": "bob@example.com",
            "preferences": {
                "theme": "light",
                "notifications": false
            }
        }
    ],
    "settings": {
        "version": "1.0",
        "last_updated": "2024-01-01"
    }
}"""
        files['json'] = self.temp_dir / "test.json"
        files['json'].write_text(json_content)

        return files

    def test_config_validity(self) -> Dict[str, Any]:
        """Test that all config files are valid YAML."""
        results = {'passed': 0, 'failed': 0, 'errors': []}

        print("🧪 Testing config file validity...")

        for i, config_file in enumerate(self.config_files, 1):
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load_all(f)
                results['passed'] += 1
                print(f"  [{i:2d}/{len(self.config_files)}] ✅ {config_file.relative_to(self.config_dir)} - Valid YAML")
            except yaml.YAMLError as e:
                results['failed'] += 1
                error_msg = f"Invalid YAML in {config_file.name}: {e}"
                results['errors'].append(error_msg)
                print(f"  [{i:2d}/{len(self.config_files)}] ❌ {config_file.relative_to(self.config_dir)} - {error_msg}")

        return results

    def test_orchestrator_initialization(self) -> Dict[str, Any]:
        """Test that orchestrator can be initialized with all configs."""
        results = {'passed': 0, 'failed': 0, 'errors': []}

        print("🏗️  Testing orchestrator initialization...")

        for i, config_file in enumerate(self.config_files, 1):
            try:
                orchestrator = ChunkerOrchestrator(config_path=config_file)
                assert orchestrator is not None
                assert orchestrator.config is not None

                results['passed'] += 1
                print(f"  [{i:2d}/{len(self.config_files)}] ✅ {config_file.relative_to(self.config_dir)} - Orchestrator initialized")

            except Exception as e:
                results['failed'] += 1
                error_msg = f"Failed to initialize orchestrator with {config_file.name}: {e}"
                results['errors'].append(error_msg)
                print(f"  [{i:2d}/{len(self.config_files)}] ❌ {config_file.relative_to(self.config_dir)} - {error_msg}")

        return results

    def test_config_with_sample_files(self) -> Dict[str, Any]:
        """Test each config with sample files."""
        results = {'passed': 0, 'failed': 0, 'errors': []}

        print("📄 Testing configs with sample files...")

        for i, config_file in enumerate(self.config_files, 1):
            config_passed = True
            config_errors = []

            try:
                orchestrator = ChunkerOrchestrator(config_path=config_file)

                # Test with text file (most universal)
                try:
                    def test_with_timeout():
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Config {config_file.stem} timed out after {self.timeout} seconds")

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(self.timeout)

                        try:
                            result = orchestrator.chunk_file(str(self.test_files['text']))
                            assert result is not None
                            assert len(result.chunks) > 0
                        finally:
                            signal.alarm(0)  # Cancel alarm

                    test_with_timeout()

                    print(f"  [{i:2d}/{len(self.config_files)}] ✅ {config_file.relative_to(self.config_dir)} - Text file processing")

                except Exception as e:
                    config_passed = False
                    error_msg = f"Text file processing failed: {e}"
                    config_errors.append(error_msg)
                    print(f"  [{i:2d}/{len(self.config_files)}] ⚠️  {config_file.relative_to(self.config_dir)} - {error_msg}")

            except Exception as e:
                config_passed = False
                error_msg = f"Orchestrator initialization failed: {e}"
                config_errors.append(error_msg)
                print(f"  [{i:2d}/{len(self.config_files)}] ❌ {config_file.relative_to(self.config_dir)} - {error_msg}")

            if config_passed:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['errors'].extend([f"{config_file.name}: {err}" for err in config_errors])

        return results

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        print(f"🚀 Starting comprehensive config testing with {len(self.config_files)} configs...")
        print(f"⏱️  Timeout set to {self.timeout} seconds per config")
        print()

        start_time = time.time()

        # Run all test categories
        validity_results = self.test_config_validity()
        print()

        init_results = self.test_orchestrator_initialization()
        print()

        file_results = self.test_config_with_sample_files()
        print()

        total_time = time.time() - start_time

        # Compile overall results
        overall_results = {
            'total_configs': len(self.config_files),
            'validity': validity_results,
            'initialization': init_results,
            'file_processing': file_results,
            'total_time': total_time,
            'all_errors': []
        }

        # Collect all errors
        for test_results in [validity_results, init_results, file_results]:
            overall_results['all_errors'].extend(test_results.get('errors', []))

        return overall_results

    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temp directory: {e}")


def print_summary(results: Dict[str, Any]):
    """Print a comprehensive summary of test results."""
    print("=" * 80)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)

    total_configs = results['total_configs']

    print(f"📁 Total configurations tested: {total_configs}")
    print(f"⏱️  Total execution time: {results['total_time']:.2f} seconds")
    print()

    # Test category results
    categories = [
        ('YAML Validity', results['validity']),
        ('Orchestrator Initialization', results['initialization']),
        ('File Processing', results['file_processing'])
    ]

    overall_passed = 0
    overall_failed = 0

    for category_name, category_results in categories:
        passed = category_results['passed']
        failed = category_results['failed']
        overall_passed += passed
        overall_failed += failed

        status_icon = "✅" if failed == 0 else "❌"
        print(f"{status_icon} {category_name}: {passed} passed, {failed} failed")

    print()
    print(f"🎯 OVERALL: {overall_passed} passed, {overall_failed} failed")

    if results['all_errors']:
        print("\n❌ DETAILED ERRORS:")
        for i, error in enumerate(results['all_errors'], 1):
            print(f"   {i}. {error}")

    success_rate = (overall_passed / (overall_passed + overall_failed)) * 100 if (overall_passed + overall_failed) > 0 else 0
    print(f"\n📈 Success Rate: {success_rate:.1f}%")

    if overall_failed == 0:
        print("🎉 All tests passed! All configurations are working correctly.")
    else:
        print(f"⚠️  {overall_failed} test(s) failed. Please review the errors above.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test all example configurations comprehensively")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--timeout", "-t", type=int, default=30, help="Timeout in seconds per config (default: 30)")

    args = parser.parse_args()

    print("🧪 Comprehensive Configuration Tester")
    print("=" * 50)

    tester = ConfigTester(verbose=args.verbose, timeout=args.timeout)

    try:
        results = tester.run_all_tests()
        print_summary(results)

        # Exit with appropriate code
        if results['all_errors']:
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
