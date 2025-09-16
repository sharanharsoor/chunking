#!/usr/bin/env python3
"""
Integration tests for directory processing with configs and CLI.

These tests verify the complete directory processing workflow including:
- Config-driven directory processing
- CLI integration with process-directory command
- End-to-end directory processing workflows
- Integration between components
"""

import pytest
import tempfile
import json
import yaml
import subprocess
import time
from pathlib import Path
from click.testing import CliRunner

from chunking_strategy import ChunkerOrchestrator
from chunking_strategy.cli import main, process_directory


class TestConfigDrivenDirectoryProcessing:
    """Test directory processing with configuration files."""

    @pytest.fixture
    def test_directory_with_config(self):
        """Create test directory with config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            files_data = [
                ("doc1.txt", "This is a simple document. It contains multiple sentences for testing."),
                ("doc2.md", "# Markdown Document\n\nThis has **formatting** and structure.\n\n## Section\n\nMore content."),
                ("script.py", "def hello():\n    print('Hello World')\n\nclass Test:\n    def method(self):\n        pass"),
                ("data.json", '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'),
                ("styles.css", "body { margin: 0; padding: 20px; } .header { color: blue; }"),
            ]

            for filename, content in files_data:
                (temp_path / filename).write_text(content)

            # Create subdirectory with nested files
            subdir = temp_path / "nested"
            subdir.mkdir()
            (subdir / "nested_doc.txt").write_text("This is nested content.")
            (subdir / "nested_code.py").write_text("print('nested')")

            # Create comprehensive config file
            config = {
                'profile_name': 'integration_test',
                'strategies': {
                    'primary': 'auto',
                    'fallbacks': ['sentence_based', 'paragraph_based', 'fixed_size'],
                    'configs': {
                        'sentence_based': {
                            'max_sentences': 2,
                            'overlap': 0
                        },
                        'paragraph_based': {
                            'max_paragraphs': 1,
                            'preserve_structure': True
                        },
                        'fixed_size': {
                            'chunk_size': 500,
                            'overlap_size': 50
                        },
                        'python_code': {
                            'preserve_functions': True,
                            'preserve_classes': True
                        },
                        'markdown_chunker': {
                            'preserve_headers': True,
                            'preserve_structure': True
                        }
                    }
                },
                'strategy_selection': {
                    '.txt': 'sentence_based',
                    '.md': 'markdown_chunker',
                    '.py': 'python_code',
                    '.json': 'json_chunker',
                    '.css': 'fixed_size'
                },
                'preprocessing': {
                    'enabled': True,
                    'normalize_whitespace': True
                },
                'postprocessing': {
                    'enabled': True,
                    'merge_short_chunks': True,
                    'min_chunk_size': 20
                }
            }

            config_path = temp_path / "test_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            yield temp_path, config_path

    def test_orchestrator_config_driven_processing(self, test_directory_with_config):
        """Test orchestrator with config file for directory processing."""
        temp_path, config_path = test_directory_with_config

        # Create orchestrator with config
        orchestrator = ChunkerOrchestrator(config_path=config_path)

        # Get all files (excluding config)
        files = [f for f in temp_path.rglob("*") if f.is_file() and f.name != "test_config.yaml"]

        # Process files
        results = orchestrator.chunk_files_batch(
            file_paths=files,
            parallel_mode="sequential"
        )

        assert len(results) == len(files)

        # Verify results by file type
        results_by_ext = {}
        for i, result in enumerate(results):
            if result and result.chunks:
                file_path = files[i]
                ext = file_path.suffix or 'no_ext'

                if ext not in results_by_ext:
                    results_by_ext[ext] = []

                results_by_ext[ext].append({
                    'file': file_path,
                    'result': result,
                    'chunks': len(result.chunks),
                    'strategy': result.strategy_used
                })

        # Verify strategy selection based on config
        assert '.txt' in results_by_ext
        assert '.md' in results_by_ext
        assert '.py' in results_by_ext

        # Verify that appropriate strategies were used
        for ext_results in results_by_ext.values():
            for item in ext_results:
                assert item['chunks'] > 0
                assert item['strategy'] is not None

    def test_config_validation_and_error_handling(self, test_directory_with_config):
        """Test config validation and error handling."""
        temp_path, _ = test_directory_with_config

        # Test with invalid config
        invalid_config = {
            'strategies': {
                'primary': 'nonexistent_strategy',
                'fallbacks': ['also_nonexistent']
            }
        }

        invalid_config_path = temp_path / "invalid_config.yaml"
        with open(invalid_config_path, 'w') as f:
            yaml.dump(invalid_config, f)

        # Should handle gracefully
        try:
            orchestrator = ChunkerOrchestrator(config_path=invalid_config_path)
            files = [f for f in temp_path.glob("*.txt")]
            if files:
                results = orchestrator.chunk_files_batch(files)
                # Should fall back to working strategies
                assert len(results) > 0
        except Exception:
            # Some validation might catch this early, which is also acceptable
            pass

    def test_config_parameter_override(self, test_directory_with_config):
        """Test that config parameters are properly applied."""
        temp_path, config_path = test_directory_with_config

        orchestrator = ChunkerOrchestrator(config_path=config_path)

        # Process a specific file type to verify config application
        txt_files = list(temp_path.glob("*.txt"))
        if txt_files:
            txt_file = txt_files[0]
            result = orchestrator.chunk_file(txt_file)

            # Based on our config, txt files should use sentence_based with max_sentences=2
            assert result.chunks is not None
            assert len(result.chunks) > 0

            # The actual strategy used might be different due to fallbacks,
            # but processing should succeed
            assert result.strategy_used is not None


class TestCLIIntegration:
    """Test CLI integration for directory processing."""

    @pytest.fixture
    def cli_test_directory(self):
        """Create test directory for CLI testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "document1.txt").write_text("This is the first document. It has multiple sentences.")
            (temp_path / "document2.md").write_text("# Title\n\nMarkdown content with **bold** text.")
            (temp_path / "script1.py").write_text("def main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()")
            (temp_path / "data.json").write_text('{"key": "value", "list": [1, 2, 3]}')

            # Create subdirectory
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("Nested document content.")
            (subdir / "nested.py").write_text("print('nested script')")

            yield temp_path

    def test_process_directory_command_basic(self, cli_test_directory):
        """Test basic process-directory command functionality."""
        runner = CliRunner()

        result = runner.invoke(main, [
            'process-directory', str(cli_test_directory)
        ])

        assert result.exit_code == 0, f"Command failed: {result.output}"

        output = result.output
        # Should show directory processing info
        assert "🗂️  Processing directory:" in output
        assert "Found" in output and "files to process" in output
        assert "PROCESSING COMPLETED" in output
        assert "SUMMARY:" in output

    def test_process_directory_with_extensions(self, cli_test_directory):
        """Test process-directory with extension filtering."""
        runner = CliRunner()

        result = runner.invoke(main, [
            'process-directory', str(cli_test_directory),
            '--extensions', '.txt,.py'
        ])

        assert result.exit_code == 0

        output = result.output
        assert "File extensions: .txt,.py" in output
        # Should process only txt and py files
        assert ".txt:" in output
        assert ".py:" in output

    def test_process_directory_with_preview(self, cli_test_directory):
        """Test process-directory with chunk preview."""
        runner = CliRunner()

        result = runner.invoke(main, [
            'process-directory', str(cli_test_directory),
            '--show-preview',
            '--max-preview-chunks', '2',
            '--extensions', '.txt'  # Limit to txt files for cleaner test
        ])

        assert result.exit_code == 0

        output = result.output
        # Should show preview
        assert "Preview" in output
        assert "Chunk 1:" in output

    def test_process_directory_non_recursive(self, cli_test_directory):
        """Test process-directory without recursion."""
        runner = CliRunner()

        result = runner.invoke(main, [
            'process-directory', str(cli_test_directory),
            '--no-recursive'
        ])

        assert result.exit_code == 0

        output = result.output
        assert "Recursive: No" in output
        # Should not process nested files
        assert "nested.txt" not in output
        assert "nested.py" not in output

    def test_process_directory_with_output_dir(self, cli_test_directory):
        """Test process-directory with output directory."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as output_dir:
            result = runner.invoke(main, [
                'process-directory', str(cli_test_directory),
                '--output-dir', output_dir,
                '--extensions', '.txt'  # Limit files for testing
            ])

            assert result.exit_code == 0

            # Check that output files were created
            output_path = Path(output_dir)
            json_files = list(output_path.glob("*.json"))
            assert len(json_files) > 0, f"No output files created in {output_dir}"

    def test_process_directory_with_config(self, cli_test_directory):
        """Test process-directory with config file."""
        runner = CliRunner()

        # Create config file
        config = {
            'strategies': {
                'primary': 'fixed_size',
                'configs': {
                    'fixed_size': {'chunk_size': 200}
                }
            }
        }

        config_path = cli_test_directory / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = runner.invoke(main, [
            'process-directory', str(cli_test_directory),
            '--config', str(config_path),
            '--extensions', '.txt'
        ])

        assert result.exit_code == 0

        output = result.output
        assert f"Using configuration: {config_path}" in output

    def test_process_directory_parallel_modes(self, cli_test_directory):
        """Test process-directory with different parallel modes."""
        runner = CliRunner()

        parallel_modes = ['sequential', 'thread', 'auto']

        for mode in parallel_modes:
            result = runner.invoke(main, [
                'process-directory', str(cli_test_directory),
                '--parallel-mode', mode,
                '--extensions', '.txt'  # Limit for faster testing
            ])

            assert result.exit_code == 0, f"Failed with mode {mode}: {result.output}"

            output = result.output
            assert f"Parallel mode: {mode}" in output

    def test_enhanced_batch_command(self, cli_test_directory):
        """Test enhanced batch command with file path printing."""
        runner = CliRunner()

        result = runner.invoke(main, [
            'batch-directory', str(cli_test_directory),
            '--pattern', '*.txt',
            '--no-output-files'
        ])

        assert result.exit_code == 0

        output = result.output
        # Should show enhanced file information
        assert "📄 Processing file:" in output
        assert "File name:" in output
        assert "File size:" in output
        assert "Extension:" in output

    def test_cli_error_handling(self):
        """Test CLI error handling for invalid inputs."""
        runner = CliRunner()

        # Test with nonexistent directory
        result = runner.invoke(main, [
            'process-directory', '/nonexistent/directory'
        ])

        assert result.exit_code != 0
        # Should show error message
        assert "does not exist" in result.output or "Error" in result.output

    def test_cli_help_integration(self):
        """Test that CLI help includes directory processing commands."""
        runner = CliRunner()

        # Test main help
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'process-directory' in result.output

        # Test process-directory help
        result = runner.invoke(main, ['process-directory', '--help'])
        assert result.exit_code == 0
        assert 'comprehensive directory processing' in result.output.lower()


class TestEndToEndWorkflows:
    """Test complete end-to-end directory processing workflows."""

    @pytest.fixture
    def comprehensive_test_setup(self):
        """Create comprehensive test setup with multiple scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create realistic directory structure
            directories = [
                "documents", "code/src", "code/tests", "data", "configs", "output"
            ]

            for dir_name in directories:
                (temp_path / dir_name).mkdir(parents=True)

            # Create realistic files
            files_data = [
                # Documents
                ("documents/user_manual.txt", "User Manual\n\nThis document explains how to use the system. " * 20),
                ("documents/api_docs.md", "# API Documentation\n\n## Overview\n\nThe API provides access to all system features.\n\n## Endpoints\n\n### GET /users\n\nRetrieve user list.\n\n### POST /users\n\nCreate new user."),
                ("documents/release_notes.md", "# Release Notes v2.1\n\n## New Features\n- Added directory processing\n- Improved parallel processing\n- Enhanced CLI commands\n\n## Bug Fixes\n- Fixed memory leaks\n- Improved error handling"),

                # Code
                ("code/src/main.py", """
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Application:
    def __init__(self):
        self.config = {}

    def load_config(self, path):
        '''Load configuration from file'''
        try:
            with open(path) as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Config load failed: {e}")

    def run(self):
        '''Run the application'''
        logger.info("Application started")
        # Main application logic
        return True

def main():
    app = Application()
    app.run()

if __name__ == '__main__':
    main()
                """),

                ("code/src/utils.py", """
def process_text(text):
    '''Process text input'''
    return text.strip().lower()

def validate_input(data):
    '''Validate input data'''
    if not data:
        return False
    return True

class DataProcessor:
    def __init__(self):
        self.cache = {}

    def process(self, data):
        if data in self.cache:
            return self.cache[data]
        result = self._process_internal(data)
        self.cache[data] = result
        return result

    def _process_internal(self, data):
        return data.upper()
                """),

                ("code/tests/test_main.py", """
import unittest
from src.main import Application

class TestApplication(unittest.TestCase):
    def setUp(self):
        self.app = Application()

    def test_initialization(self):
        self.assertIsInstance(self.app.config, dict)

    def test_run(self):
        result = self.app.run()
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
                """),

                # Data files
                ("data/users.json", json.dumps({
                    "users": [
                        {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "admin"},
                        {"id": 2, "name": "Bob", "email": "bob@example.com", "role": "user"},
                        {"id": 3, "name": "Carol", "email": "carol@example.com", "role": "moderator"}
                    ]
                }, indent=2)),

                ("data/metrics.csv", """date,users,sessions,revenue
2024-01-01,150,450,1250.75
2024-01-02,160,480,1380.25
2024-01-03,155,465,1315.50
2024-01-04,170,510,1445.00
2024-01-05,165,495,1402.75"""),

                # Config file
                ("configs/processing_config.yaml", yaml.dump({
                    'profile_name': 'comprehensive_test',
                    'strategies': {
                        'primary': 'auto',
                        'fallbacks': ['sentence_based', 'paragraph_based', 'fixed_size'],
                        'configs': {
                            'sentence_based': {'max_sentences': 3},
                            'paragraph_based': {'max_paragraphs': 2},
                            'fixed_size': {'chunk_size': 800}
                        }
                    },
                    'preprocessing': {'enabled': True},
                    'postprocessing': {'enabled': True}
                }))
            ]

            # Write all files
            for file_path, content in files_data:
                full_path = temp_path / file_path
                full_path.write_text(content.strip())

            yield temp_path

    def test_complete_directory_processing_workflow(self, comprehensive_test_setup):
        """Test complete directory processing workflow."""
        test_dir = comprehensive_test_setup
        config_path = test_dir / "configs" / "processing_config.yaml"

        # Create orchestrator with config
        orchestrator = ChunkerOrchestrator(config_path=config_path)

        # Get all files (excluding config files)
        all_files = [
            f for f in test_dir.rglob("*")
            if f.is_file() and not f.name.endswith('.yaml')
        ]

        assert len(all_files) >= 8  # Should have multiple files

        # Process with different parallel modes
        modes = ["sequential", "thread"]

        for mode in modes:
            start_time = time.time()

            results = orchestrator.chunk_files_batch(
                file_paths=all_files,
                parallel_mode=mode,
                max_workers=2 if mode != "sequential" else None
            )

            end_time = time.time()
            processing_time = end_time - start_time

            # Verify results
            assert len(results) == len(all_files)

            successful_files = []
            total_chunks = 0

            for i, result in enumerate(results):
                file_path = all_files[i]

                if result and result.chunks:
                    successful_files.append(file_path)
                    total_chunks += len(result.chunks)

                    # Verify result properties
                    assert result.strategy_used is not None
                    assert result.processing_time >= 0
                    assert len(result.chunks) > 0

                    # Verify chunk properties
                    for chunk in result.chunks:
                        assert hasattr(chunk, 'content')
                        assert hasattr(chunk, 'metadata')
                        assert len(chunk.content.strip()) > 0

            # Should have high success rate
            success_rate = len(successful_files) / len(all_files)
            assert success_rate >= 0.8, f"Low success rate: {success_rate}"

            # Should have generated reasonable number of chunks
            assert total_chunks > 0

            print(f"Mode {mode}: {len(successful_files)}/{len(all_files)} files, "
                  f"{total_chunks} chunks, {processing_time:.2f}s")

    def test_mixed_file_type_processing(self, comprehensive_test_setup):
        """Test processing of mixed file types."""
        test_dir = comprehensive_test_setup

        orchestrator = ChunkerOrchestrator()

        # Group files by type
        file_groups = {
            'text': list(test_dir.rglob("*.txt")),
            'markdown': list(test_dir.rglob("*.md")),
            'python': list(test_dir.rglob("*.py")),
            'json': list(test_dir.rglob("*.json")),
            'csv': list(test_dir.rglob("*.csv"))
        }

        # Process each file type
        results_by_type = {}

        for file_type, files in file_groups.items():
            if files:
                results = orchestrator.chunk_files_batch(files, parallel_mode="sequential")

                successful_results = [r for r in results if r and r.chunks]
                results_by_type[file_type] = {
                    'total_files': len(files),
                    'successful': len(successful_results),
                    'total_chunks': sum(len(r.chunks) for r in successful_results),
                    'strategies_used': set(r.strategy_used for r in successful_results)
                }

        # Verify each file type was processed successfully
        for file_type, stats in results_by_type.items():
            assert stats['successful'] > 0, f"No successful processing for {file_type}"
            assert stats['total_chunks'] > 0, f"No chunks generated for {file_type}"
            assert len(stats['strategies_used']) > 0, f"No strategies used for {file_type}"

    def test_performance_and_scalability(self, comprehensive_test_setup):
        """Test performance and scalability characteristics."""
        test_dir = comprehensive_test_setup

        orchestrator = ChunkerOrchestrator()
        all_files = [f for f in test_dir.rglob("*") if f.is_file() and not f.name.endswith('.yaml')]

        # Test with different worker counts
        performance_results = {}

        for workers in [1, 2]:
            start_time = time.time()

            results = orchestrator.chunk_files_batch(
                file_paths=all_files,
                parallel_mode="thread",
                max_workers=workers
            )

            end_time = time.time()
            processing_time = end_time - start_time

            successful_count = sum(1 for r in results if r and r.chunks)
            total_chunks = sum(len(r.chunks) for r in results if r and r.chunks)

            performance_results[workers] = {
                'time': processing_time,
                'files_per_sec': successful_count / processing_time if processing_time > 0 else 0,
                'chunks_per_sec': total_chunks / processing_time if processing_time > 0 else 0,
                'successful_files': successful_count,
                'total_chunks': total_chunks
            }

        # Verify performance characteristics
        for workers, stats in performance_results.items():
            assert stats['files_per_sec'] > 0, f"No throughput with {workers} workers"
            assert stats['successful_files'] > 0, f"No successful files with {workers} workers"

        print("\nPerformance Results:")
        for workers, stats in performance_results.items():
            print(f"  {workers} workers: {stats['files_per_sec']:.1f} files/sec, "
                  f"{stats['chunks_per_sec']:.1f} chunks/sec")

    def test_error_recovery_and_robustness(self, comprehensive_test_setup):
        """Test error recovery and robustness."""
        test_dir = comprehensive_test_setup

        # Add some problematic files
        problematic_files = [
            (test_dir / "empty_file.txt", ""),  # Empty file
            (test_dir / "binary_file.bin", b'\x00\x01\x02\x03\x04\x05'),  # Binary file
        ]

        for file_path, content in problematic_files:
            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content)

        orchestrator = ChunkerOrchestrator()

        # Include both good and problematic files
        all_files = [f for f in test_dir.rglob("*") if f.is_file() and not f.name.endswith('.yaml')]

        # Should handle errors gracefully
        results = orchestrator.chunk_files_batch(
            file_paths=all_files,
            parallel_mode="sequential"
        )

        assert len(results) == len(all_files)

        # Count successful vs failed processing
        successful = sum(1 for r in results if r and r.chunks)
        failed = len(results) - successful

        # Should have some successful processing despite problematic files
        assert successful > 0, "No files processed successfully"

        # System should be robust enough to handle some failures
        success_rate = successful / len(all_files)
        assert success_rate >= 0.6, f"Success rate too low: {success_rate}"

        print(f"Robustness test: {successful}/{len(all_files)} successful ({success_rate:.1%})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
