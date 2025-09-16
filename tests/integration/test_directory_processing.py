#!/usr/bin/env python3
"""
Comprehensive unit tests for directory processing functionality.

Tests directory-level processing, file path printing, parallel processing,
and integration with configs and CLI.
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, call
from click.testing import CliRunner

from chunking_strategy import ChunkerOrchestrator
from chunking_strategy.core.batch import BatchProcessor, BatchFile, BatchResult
from chunking_strategy.cli import main, process_directory, batch


class TestDirectoryDiscovery:
    """Test directory file discovery and filtering."""

    @pytest.fixture
    def test_directory(self):
        """Create a test directory with multiple file types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files with different extensions
            files_to_create = [
                ("document1.txt", "This is a text document with multiple sentences. Each sentence provides content for testing."),
                ("document2.md", "# Markdown Document\n\nThis is **markdown** content."),
                ("code.py", "def hello():\n    print('Hello, World!')"),
                ("data.json", '{"key": "value", "numbers": [1, 2, 3]}'),
                ("styles.css", "body { font-family: Arial; }"),
                ("README", "This is a README file without extension."),
            ]

            for filename, content in files_to_create:
                file_path = temp_path / filename
                file_path.write_text(content)

            # Create subdirectory with nested files
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("This is a nested file.")
            (subdir / "deep.py").write_text("print('deep file')")

            yield temp_path

    def test_file_discovery_non_recursive(self, test_directory):
        """Test file discovery without recursion."""
        files = [f for f in test_directory.glob("*") if f.is_file()]

        assert len(files) == 6  # Only top-level files
        extensions = {f.suffix for f in files}
        expected_extensions = {'.txt', '.md', '.py', '.json', '.css', ''}
        assert extensions == expected_extensions

    def test_file_discovery_recursive(self, test_directory):
        """Test recursive file discovery."""
        files = [f for f in test_directory.rglob("*") if f.is_file()]

        assert len(files) == 8  # All files including nested ones
        filenames = {f.name for f in files}
        assert 'nested.txt' in filenames
        assert 'deep.py' in filenames

    def test_extension_filtering(self, test_directory):
        """Test filtering files by extension."""
        # Filter for text files
        txt_files = list(test_directory.rglob("*.txt"))
        assert len(txt_files) == 2  # document1.txt and nested.txt

        # Filter for Python files
        py_files = list(test_directory.rglob("*.py"))
        assert len(py_files) == 2  # code.py and deep.py

    def test_multiple_extension_filtering(self, test_directory):
        """Test filtering for multiple extensions."""
        extensions = ['.txt', '.py']
        files = []

        for ext in extensions:
            files.extend(list(test_directory.rglob(f"*{ext}")))

        assert len(files) == 4  # 2 txt + 2 py files

        # Verify all files have correct extensions
        for file in files:
            assert file.suffix in extensions


class TestBatchProcessorEnhancements:
    """Test enhancements to BatchProcessor for directory processing."""

    @pytest.fixture
    def test_files(self):
        """Create temporary test files."""
        files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for i in range(3):
                file_path = temp_path / f"test_file_{i}.txt"
                content = f"This is test content for file {i}. " * (5 + i)
                file_path.write_text(content)
                files.append(file_path)

            yield files

    def test_batch_processor_with_progress_callback(self, test_files):
        """Test BatchProcessor with progress callback showing file paths."""
        progress_calls = []

        def progress_callback(current, total, message):
            progress_calls.append((current, total, message))

        processor = BatchProcessor(progress_callback=progress_callback)

        result = processor.process_files(
            files=test_files,
            default_strategy="sentence_based",
            parallel_mode="sequential"
        )

        assert result.total_files == len(test_files)
        assert len(result.successful_files) == len(test_files)

        # Verify progress callback was called with full file paths
        assert len(progress_calls) == len(test_files)
        for i, (current, total, message) in enumerate(progress_calls):
            assert current == i + 1
            assert total == len(test_files)
            assert str(test_files[i]) in message  # Full path should be in message

    @patch('logging.getLogger')
    def test_batch_processor_logging_file_paths(self, mock_get_logger, test_files):
        """Test that BatchProcessor logs full file paths during processing."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        processor = BatchProcessor()

        result = processor.process_files(
            files=test_files,
            default_strategy="sentence_based",
            parallel_mode="sequential"
        )

        assert result.total_files == len(test_files)

        # Verify that full file paths were logged
        mock_logger.info.assert_any_call(f"📄 Processing file: {test_files[0]}")
        mock_logger.info.assert_any_call(f"📄 Processing file: {test_files[1]}")
        mock_logger.info.assert_any_call(f"📄 Processing file: {test_files[2]}")

    def test_batch_processor_parallel_modes(self, test_files):
        """Test different parallel modes with file path logging."""
        processor = BatchProcessor()

        modes = ["sequential", "thread"]  # Skip process mode for unit tests

        for mode in modes:
            result = processor.process_files(
                files=test_files,
                default_strategy="sentence_based",
                parallel_mode=mode,
                workers=2 if mode != "sequential" else None
            )

            assert result.total_files == len(test_files)
            assert len(result.successful_files) == len(test_files)
            assert result.total_chunks > 0

            # All files should be processed successfully
            for file_path in test_files:
                assert file_path in result.successful_files
                assert str(file_path) in result.chunk_results

    def test_batch_file_creation(self, test_files):
        """Test BatchFile creation with file metadata."""
        processor = BatchProcessor()
        batch_files = processor._prepare_batch_files(
            files=test_files,
            default_strategy="sentence_based",
            default_params={}
        )

        assert len(batch_files) == len(test_files)

        for i, batch_file in enumerate(batch_files):
            assert isinstance(batch_file, BatchFile)
            assert batch_file.path == test_files[i]
            assert batch_file.size_mb > 0
            assert batch_file.chunker_strategy == "sentence_based"


class TestOrchestratorBatchProcessing:
    """Test Orchestrator batch processing with directory support."""

    @pytest.fixture
    def test_directory_with_configs(self):
        """Create test directory with files and a config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "doc1.txt").write_text("This is a simple document. It has multiple sentences.")
            (temp_path / "doc2.md").write_text("# Title\n\nMarkdown content here.")
            (temp_path / "code.py").write_text("def func():\n    return 'hello'")

            # Create config file
            config = {
                'profile_name': 'test_profile',
                'strategies': {
                    'primary': 'sentence_based',
                    'fallbacks': ['paragraph_based', 'fixed_size'],
                    'configs': {
                        'sentence_based': {'max_sentences': 2},
                        'fixed_size': {'chunk_size': 500}
                    }
                },
                'preprocessing': {'enabled': False},
                'postprocessing': {'enabled': True, 'merge_short_chunks': True}
            }

            config_path = temp_path / "test_config.yaml"
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(config, f)

            yield temp_path, config_path

    def test_orchestrator_directory_processing_with_config(self, test_directory_with_configs):
        """Test orchestrator processing a directory with config file."""
        temp_path, config_path = test_directory_with_configs

        # Create orchestrator with config
        orchestrator = ChunkerOrchestrator(config_path=config_path)

        # Get all files
        files = [f for f in temp_path.rglob("*") if f.is_file() and f.name != "test_config.yaml"]

        # Process files
        results = orchestrator.chunk_files_batch(
            file_paths=files,
            parallel_mode="sequential"
        )

        assert len(results) == len(files)

        # All results should be successful
        for result in results:
            assert result is not None
            assert len(result.chunks) > 0
            assert result.strategy_used is not None

    def test_orchestrator_auto_strategy_selection(self, test_directory_with_configs):
        """Test that orchestrator selects appropriate strategies for different file types."""
        temp_path, config_path = test_directory_with_configs

        orchestrator = ChunkerOrchestrator(config_path=config_path)

        # Process individual files to see strategy selection
        txt_file = temp_path / "doc1.txt"
        md_file = temp_path / "doc2.md"
        py_file = temp_path / "code.py"

        txt_result = orchestrator.chunk_file(txt_file)
        md_result = orchestrator.chunk_file(md_file)
        py_result = orchestrator.chunk_file(py_file)

        # All should succeed
        assert len(txt_result.chunks) > 0
        assert len(md_result.chunks) > 0
        assert len(py_result.chunks) > 0

        # Strategy should be appropriate for each file type
        assert txt_result.strategy_used is not None
        assert md_result.strategy_used is not None
        assert py_result.strategy_used is not None

    def test_orchestrator_batch_performance_tracking(self, test_directory_with_configs):
        """Test performance tracking in batch processing."""
        temp_path, config_path = test_directory_with_configs

        orchestrator = ChunkerOrchestrator(config_path=config_path)

        files = [f for f in temp_path.rglob("*") if f.is_file() and f.name != "test_config.yaml"]

        start_time = time.time()
        results = orchestrator.chunk_files_batch(file_paths=files)
        end_time = time.time()

        processing_time = end_time - start_time

        # Verify timing
        assert processing_time > 0
        assert len(results) == len(files)

        # Each result should have timing information
        for result in results:
            if result:
                assert hasattr(result, 'processing_time')
                assert result.processing_time >= 0


class TestCLIDirectoryProcessing:
    """Test CLI commands for directory processing."""

    @pytest.fixture
    def cli_test_directory(self):
        """Create test directory for CLI testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "doc1.txt").write_text("Simple text document.")
            (temp_path / "doc2.md").write_text("# Markdown\n\nContent here.")
            (temp_path / "script.py").write_text("print('hello')")

            # Create subdirectory
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("Nested file content.")

            yield temp_path

    def test_cli_batch_command_enhanced(self, cli_test_directory):
        """Test enhanced batch command with file path printing."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as output_dir:
            result = runner.invoke(main, [
                'batch-directory', str(cli_test_directory),
                '--output-dir', output_dir,
                '--pattern', '*.txt',
                '--no-output-files'  # Don't save files for testing
            ])

            assert result.exit_code == 0

            # Output should contain file paths
            output = result.output
            assert "📄 Processing file:" in output
            assert "doc1.txt" in output
            assert "File name:" in output
            assert "File size:" in output
            assert "Extension:" in output

    def test_cli_process_directory_command(self, cli_test_directory):
        """Test new process-directory command."""
        runner = CliRunner()

        result = runner.invoke(main, [
            'process-directory', str(cli_test_directory),
            '--extensions', '.txt,.py',
            '--no-recursive'  # Test non-recursive mode
        ])

        assert result.exit_code == 0

        output = result.output
        # Should show directory processing info
        assert "🗂️  Processing directory:" in output
        assert str(cli_test_directory) in output
        assert "File extensions:" in output
        assert ".txt,.py" in output
        assert "Recursive: No" in output

    def test_cli_process_directory_with_preview(self, cli_test_directory):
        """Test process-directory command with chunk preview."""
        runner = CliRunner()

        result = runner.invoke(main, [
            'process-directory', str(cli_test_directory),
            '--extensions', '.txt',
            '--show-preview',
            '--max-preview-chunks', '2'
        ])

        assert result.exit_code == 0

        output = result.output
        # Should show preview information
        assert "Preview" in output
        assert "chunks using" in output
        assert "Chunk 1:" in output

    def test_cli_process_directory_with_config(self, cli_test_directory):
        """Test process-directory command with config file."""
        runner = CliRunner()

        # Create a simple config file
        config = {
            'strategies': {
                'primary': 'fixed_size',
                'configs': {
                    'fixed_size': {'chunk_size': 100}
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            import yaml
            yaml.dump(config, config_file)
            config_path = config_file.name

        try:
            result = runner.invoke(main, [
                'process-directory', str(cli_test_directory),
                '--config', config_path,
                '--extensions', '.txt'
            ])

            assert result.exit_code == 0

            output = result.output
            assert f"Using configuration: {config_path}" in output

        finally:
            Path(config_path).unlink()  # Clean up

    def test_cli_help_messages(self):
        """Test that help messages include directory processing info."""
        runner = CliRunner()

        # Test main help
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'process-directory' in result.output
        assert 'batch' in result.output

        # Test process-directory help
        result = runner.invoke(main, ['process-directory', '--help'])
        assert result.exit_code == 0
        assert 'Process all files in a directory' in result.output
        assert 'Full file path printing' in result.output
        assert '--extensions' in result.output
        assert '--parallel-mode' in result.output


class TestDirectoryProcessingIntegration:
    """Integration tests for directory processing with various scenarios."""

    @pytest.fixture
    def complex_directory(self):
        """Create a complex directory structure for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple file types
            files_data = [
                ("documents/report.txt", "Executive summary of quarterly results. Key metrics improved."),
                ("documents/notes.md", "# Meeting Notes\n\n- Action item 1\n- Action item 2"),
                ("code/main.py", "import os\n\ndef main():\n    print('Application started')\n\nif __name__ == '__main__':\n    main()"),
                ("code/utils.py", "def helper_function():\n    return 'helper result'"),
                ("data/config.json", '{"database": {"host": "localhost", "port": 5432}}'),
                ("data/sample.csv", "name,age,city\nAlice,30,NYC\nBob,25,SF"),
                ("styles/main.css", "body { margin: 0; padding: 20px; font-family: Arial; }"),
                ("README.txt", "This is the project README file with instructions."),
            ]

            for file_path, content in files_data:
                full_path = temp_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            yield temp_path

    def test_full_directory_processing_workflow(self, complex_directory):
        """Test complete directory processing workflow."""
        # Create orchestrator
        orchestrator = ChunkerOrchestrator()

        # Get all files
        all_files = [f for f in complex_directory.rglob("*") if f.is_file()]

        # Process with different parallel modes
        for mode in ["sequential", "thread"]:
            results = orchestrator.chunk_files_batch(
                file_paths=all_files,
                parallel_mode=mode
            )

            assert len(results) == len(all_files)

            # Group results by file type
            results_by_ext = {}
            for i, result in enumerate(results):
                if result and result.chunks:
                    file_path = all_files[i]
                    ext = file_path.suffix or 'no_ext'
                    if ext not in results_by_ext:
                        results_by_ext[ext] = []
                    results_by_ext[ext].append((file_path, result))

            # Verify different file types were processed
            assert len(results_by_ext) > 5  # Should have multiple file types

            # Verify strategies were selected appropriately
            for ext, file_results in results_by_ext.items():
                for file_path, result in file_results:
                    assert result.strategy_used is not None
                    assert len(result.chunks) > 0

    def test_directory_processing_with_filtering(self, complex_directory):
        """Test directory processing with file type filtering."""
        # Process only Python files
        py_files = list(complex_directory.rglob("*.py"))
        assert len(py_files) == 2

        orchestrator = ChunkerOrchestrator()
        results = orchestrator.chunk_files_batch(file_paths=py_files)

        assert len(results) == len(py_files)

        for result in results:
            assert result is not None
            assert len(result.chunks) > 0
            # Python files should use code-specific strategies
            assert result.strategy_used is not None

    def test_directory_processing_error_handling(self, complex_directory):
        """Test error handling in directory processing."""
        # Create a list with some non-existent files
        files = [f for f in complex_directory.rglob("*.txt")]
        files.append(Path("/nonexistent/file.txt"))

        orchestrator = ChunkerOrchestrator()

        # This should handle the error gracefully
        results = orchestrator.chunk_files_batch(file_paths=files)

        # Should have results for existing files and empty result for non-existent
        assert len(results) == len(files)

        # Count successful vs failed
        successful = sum(1 for r in results if r and r.chunks)
        failed = len(results) - successful

        assert successful > 0  # Some files should succeed
        assert failed == 1     # One file should fail (the non-existent one)

    def test_directory_processing_performance_scaling(self, complex_directory):
        """Test that directory processing scales appropriately."""
        orchestrator = ChunkerOrchestrator()
        all_files = [f for f in complex_directory.rglob("*") if f.is_file()]

        # Test with different worker counts
        for workers in [1, 2]:
            start_time = time.time()

            results = orchestrator.chunk_files_batch(
                file_paths=all_files,
                parallel_mode="thread",
                max_workers=workers
            )

            end_time = time.time()
            processing_time = end_time - start_time

            assert len(results) == len(all_files)
            assert processing_time > 0

            # Verify all successful
            successful_results = [r for r in results if r and r.chunks]
            assert len(successful_results) >= len(all_files) * 0.8  # At least 80% success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
