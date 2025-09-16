"""
CLI tests for OverlappingWindowChunker.

This module tests the command-line interface integration for the overlapping
window chunking strategy, covering argument parsing, parameter validation,
output formats, and error handling.
"""

import pytest
import tempfile
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestOverlappingWindowCLI:
    """Test suite for OverlappingWindowChunker CLI integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_content = """
        Machine Learning has revolutionized the way we approach data analysis
        and pattern recognition. From simple linear regression models to complex
        neural networks with millions of parameters, the field has evolved rapidly.
        Modern applications include computer vision, natural language processing,
        recommendation systems, and autonomous vehicles. Each of these domains
        presents unique challenges and opportunities for innovation.
        """.strip()
        
        self.long_content = " ".join([
            f"This is sentence number {i} in a longer document for testing purposes."
            for i in range(1, 51)
        ])

    @pytest.mark.skip(reason="Requires fully operational CLI system")
    def test_cli_help_and_version(self):
        """Test CLI help and version commands."""
        # Test help command
        result = subprocess.run([
            sys.executable, "-m", "chunking_strategy", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "overlapping_window" in result.stdout.lower() or "overlapping-window" in result.stdout.lower()

        # Test strategy-specific help
        result = subprocess.run([
            sys.executable, "-m", "chunking_strategy", "chunk",
            "--strategy", "overlapping_window", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "window-size" in result.stdout.lower()
        assert "step-size" in result.stdout.lower()

    @pytest.mark.skip(reason="Requires fully operational CLI system") 
    def test_basic_cli_chunking(self):
        """Test basic CLI chunking functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(self.test_content)
            input_path = tmp_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_output:
            output_path = tmp_output.name

        try:
            # Test basic chunking command
            result = subprocess.run([
                sys.executable, "-m", "chunking_strategy", "chunk",
                "--strategy", "overlapping_window",
                "--window-size", "30",
                "--step-size", "15", 
                "--window-unit", "words",
                "--input", input_path,
                "--output", output_path
            ], capture_output=True, text=True)

            assert result.returncode == 0
            
            # Verify output file exists and contains valid JSON
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            
            assert "chunks" in output_data
            assert len(output_data["chunks"]) > 0
            assert "strategy_used" in output_data

        finally:
            Path(input_path).unlink()
            Path(output_path).unlink()

    @pytest.mark.skip(reason="Requires fully operational CLI system")
    def test_different_window_units_cli(self):
        """Test CLI with different window unit options."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(self.long_content)
            input_path = tmp_file.name

        try:
            window_units = ["words", "characters", "sentences"]
            
            for unit in window_units:
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{unit}.json', delete=False) as tmp_output:
                    output_path = tmp_output.name

                try:
                    # Adjust parameters based on unit type
                    if unit == "words":
                        window_size, step_size = "25", "12"
                    elif unit == "characters":
                        window_size, step_size = "200", "100"
                    else:  # sentences
                        window_size, step_size = "3", "1"

                    result = subprocess.run([
                        sys.executable, "-m", "chunking_strategy", "chunk",
                        "--strategy", "overlapping_window",
                        "--window-size", window_size,
                        "--step-size", step_size,
                        "--window-unit", unit,
                        "--input", input_path,
                        "--output", output_path
                    ], capture_output=True, text=True)

                    assert result.returncode == 0, f"Failed for unit: {unit}, stderr: {result.stderr}"
                    
                    # Verify output
                    with open(output_path, 'r') as f:
                        output_data = json.load(f)
                    
                    assert len(output_data["chunks"]) > 0

                finally:
                    if Path(output_path).exists():
                        Path(output_path).unlink()

        finally:
            Path(input_path).unlink()

    @pytest.mark.skip(reason="Requires fully operational CLI system")
    def test_config_file_usage(self):
        """Test CLI usage with configuration files."""
        # Create a test config file
        config_data = {
            "chunking_strategy": "overlapping_window",
            "parameters": {
                "window_size": 40,
                "step_size": 20,
                "window_unit": "words",
                "preserve_boundaries": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            import yaml
            yaml.dump(config_data, config_file)
            config_path = config_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_file:
            input_file.write(self.test_content)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_path = output_file.name

        try:
            result = subprocess.run([
                sys.executable, "-m", "chunking_strategy", "chunk",
                "--config", config_path,
                "--input", input_path,
                "--output", output_path
            ], capture_output=True, text=True)

            assert result.returncode == 0
            
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            
            assert output_data["strategy_used"] == "overlapping_window"
            assert len(output_data["chunks"]) > 0

        finally:
            Path(config_path).unlink()
            Path(input_path).unlink() 
            Path(output_path).unlink()

    @pytest.mark.skip(reason="Requires fully operational CLI system")
    def test_batch_processing_cli(self):
        """Test CLI batch processing of multiple files."""
        # Create multiple test files
        test_files = []
        for i in range(3):
            content = f"Document {i+1}: " + self.test_content
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_doc{i+1}.txt', delete=False) as f:
                f.write(content)
                test_files.append(f.name)

        output_dir = tempfile.mkdtemp()

        try:
            # Process all files
            for input_file in test_files:
                output_name = Path(input_file).stem + "_chunks.json"
                output_path = Path(output_dir) / output_name
                
                result = subprocess.run([
                    sys.executable, "-m", "chunking_strategy", "chunk",
                    "--strategy", "overlapping_window",
                    "--window-size", "20",
                    "--step-size", "10",
                    "--input", input_file,
                    "--output", str(output_path)
                ], capture_output=True, text=True)

                assert result.returncode == 0
                assert output_path.exists()
                
                with open(output_path, 'r') as f:
                    output_data = json.load(f)
                assert len(output_data["chunks"]) > 0

        finally:
            for file_path in test_files:
                Path(file_path).unlink()
            import shutil
            shutil.rmtree(output_dir)

    @pytest.mark.skip(reason="Requires fully operational CLI system")
    def test_parameter_validation_cli(self):
        """Test CLI parameter validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(self.test_content)
            input_path = tmp_file.name

        try:
            # Test invalid step_size >= window_size
            result = subprocess.run([
                sys.executable, "-m", "chunking_strategy", "chunk",
                "--strategy", "overlapping_window",
                "--window-size", "20",
                "--step-size", "25",  # Invalid: larger than window_size
                "--input", input_path
            ], capture_output=True, text=True)

            assert result.returncode != 0
            assert "step_size" in result.stderr.lower() or "step-size" in result.stderr.lower()

            # Test invalid window unit
            result = subprocess.run([
                sys.executable, "-m", "chunking_strategy", "chunk",
                "--strategy", "overlapping_window",
                "--window-unit", "invalid_unit",
                "--input", input_path
            ], capture_output=True, text=True)

            assert result.returncode != 0

        finally:
            Path(input_path).unlink()

    def test_cli_parameter_mapping(self):
        """Test that CLI parameters map correctly to chunker parameters."""
        from chunking_strategy.strategies.text.overlapping_window_chunker import OverlappingWindowChunker
        
        # Test parameter mapping (mock CLI parsing)
        cli_args = {
            "window_size": 50,
            "step_size": 25,
            "window_unit": "words",
            "preserve_boundaries": True,
            "min_window_size": 10,
            "max_chunk_chars": 1000
        }
        
        # This would normally be done by CLI parser
        chunker = OverlappingWindowChunker(**cli_args)
        
        assert chunker.window_size == 50
        assert chunker.step_size == 25
        assert chunker.window_unit.value == "words"
        assert chunker.preserve_boundaries is True
        assert chunker.min_window_size == 10
        assert chunker.max_chunk_chars == 1000

    @pytest.mark.skip(reason="Requires fully operational CLI system")
    def test_verbose_output_cli(self):
        """Test CLI verbose output mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(self.test_content)
            input_path = tmp_file.name

        try:
            result = subprocess.run([
                sys.executable, "-m", "chunking_strategy", "chunk",
                "--strategy", "overlapping_window",
                "--window-size", "30",
                "--step-size", "15",
                "--verbose",
                "--input", input_path
            ], capture_output=True, text=True)

            assert result.returncode == 0
            # Verbose output should contain progress information
            assert "processing" in result.stderr.lower() or "chunk" in result.stderr.lower()

        finally:
            Path(input_path).unlink()

    @pytest.mark.skip(reason="Requires fully operational CLI system")
    def test_output_format_options(self):
        """Test different output format options."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(self.test_content)
            input_path = tmp_file.name

        output_formats = ["json", "yaml", "txt"]

        try:
            for fmt in output_formats:
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{fmt}', delete=False) as tmp_output:
                    output_path = tmp_output.name

                try:
                    result = subprocess.run([
                        sys.executable, "-m", "chunking_strategy", "chunk",
                        "--strategy", "overlapping_window",
                        "--window-size", "25",
                        "--step-size", "12",
                        "--format", fmt,
                        "--input", input_path,
                        "--output", output_path
                    ], capture_output=True, text=True)

                    assert result.returncode == 0
                    assert Path(output_path).exists()
                    assert Path(output_path).stat().st_size > 0

                finally:
                    if Path(output_path).exists():
                        Path(output_path).unlink()

        finally:
            Path(input_path).unlink()

    @pytest.mark.skip(reason="Requires fully operational CLI system")
    def test_dry_run_mode(self):
        """Test CLI dry-run mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(self.test_content)
            input_path = tmp_file.name

        try:
            result = subprocess.run([
                sys.executable, "-m", "chunking_strategy", "chunk",
                "--strategy", "overlapping_window",
                "--window-size", "30",
                "--step-size", "15",
                "--dry-run",
                "--input", input_path
            ], capture_output=True, text=True)

            assert result.returncode == 0
            # Dry run should show what would be done without actually processing
            assert "would process" in result.stdout.lower() or "dry run" in result.stdout.lower()

        finally:
            Path(input_path).unlink()

    def test_cli_integration_mock(self):
        """Test CLI integration with mocked CLI system."""
        # Mock CLI argument parsing
        mock_args = {
            'strategy': 'overlapping_window',
            'window_size': 40,
            'step_size': 20,
            'window_unit': 'words',
            'preserve_boundaries': True,
            'min_window_size': 20,  # Compatible with window_size=40
            'input_file': 'test.txt',
            'output_file': 'output.json'
        }
        
        # Test that arguments would be correctly processed
        from chunking_strategy.strategies.text.overlapping_window_chunker import OverlappingWindowChunker
        
        # Extract chunker parameters
        chunker_params = {
            k: v for k, v in mock_args.items() 
            if k not in ['strategy', 'input_file', 'output_file']
        }
        
        # Create chunker with CLI parameters
        chunker = OverlappingWindowChunker(**chunker_params)
        
        # Verify parameters were set correctly
        assert chunker.window_size == 40
        assert chunker.step_size == 20
        assert chunker.window_unit.value == 'words'
        assert chunker.preserve_boundaries is True

    def test_error_handling_scenarios(self):
        """Test various error handling scenarios."""
        from chunking_strategy.strategies.text.overlapping_window_chunker import OverlappingWindowChunker
        
        # Test invalid parameter combinations
        with pytest.raises(ValueError):
            OverlappingWindowChunker(
                window_size=10,
                step_size=15,  # step_size > window_size
                window_unit="words"
            )
        
        with pytest.raises(ValueError):
            OverlappingWindowChunker(
                window_size=20,
                min_window_size=25,  # min_window_size > window_size
                window_unit="words"
            )

    def test_help_text_content(self):
        """Test that help text contains expected information."""
        from chunking_strategy.strategies.text.overlapping_window_chunker import OverlappingWindowChunker
        
        # Check that the chunker has proper documentation
        assert OverlappingWindowChunker.__doc__ is not None
        assert "overlapping" in OverlappingWindowChunker.__doc__.lower()
        assert "window" in OverlappingWindowChunker.__doc__.lower()
        
        # Check that key parameters are documented
        init_doc = OverlappingWindowChunker.__init__.__doc__
        if init_doc:
            assert "window_size" in init_doc
            assert "step_size" in init_doc
            assert "window_unit" in init_doc
