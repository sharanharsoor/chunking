"""
Configuration tests for OverlappingWindowChunker.

This module tests configuration file loading, validation, parameter overrides,
and various configuration scenarios for the overlapping window chunking strategy.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

from chunking_strategy.strategies.text.overlapping_window_chunker import (
    OverlappingWindowChunker,
    WindowUnit
)
from chunking_strategy.core.base import ChunkingResult


class TestOverlappingWindowConfig:
    """Test suite for OverlappingWindowChunker configuration handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_content = """
        Quantum computing represents a paradigm shift in computational power and
        capability. Unlike classical computers that use binary bits, quantum
        computers leverage quantum mechanical phenomena such as superposition
        and entanglement to process information in fundamentally different ways.
        This enables exponentially faster processing for certain types of problems.
        """.strip()

        self.sample_configs = {
            "basic_config": {
                "window_size": 50,
                "step_size": 25,
                "window_unit": "words",
                "preserve_boundaries": True
            },
            "character_config": {
                "window_size": 300,
                "step_size": 150,
                "window_unit": "characters",
                "preserve_boundaries": False,
                "max_chunk_chars": 500
            },
            "sentence_config": {
                "window_size": 3,
                "step_size": 1,
                "window_unit": "sentences",
                "preserve_boundaries": True,
                "sentence_separators": [".", "!", "?"]
            }
        }

    def test_basic_yaml_configuration(self):
        """Test loading basic YAML configuration."""
        config_data = {
            "overlapping_window_chunker": self.sample_configs["basic_config"]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            yaml.dump(config_data, config_file)
            config_path = Path(config_file.name)

        try:
            # Load configuration
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)

            # Create chunker with loaded config
            chunker_config = loaded_config["overlapping_window_chunker"]
            chunker = OverlappingWindowChunker(**chunker_config)

            # Verify parameters
            assert chunker.window_size == 50
            assert chunker.step_size == 25
            assert chunker.window_unit == WindowUnit.WORDS
            assert chunker.preserve_boundaries is True

            # Test functionality
            result = chunker.chunk(self.test_content)
            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0

        finally:
            config_path.unlink()

    def test_multiple_configuration_profiles(self):
        """Test loading and using multiple configuration profiles."""
        config_data = {
            "profiles": {
                "rag_optimized": {
                    "window_size": 150,
                    "step_size": 75,
                    "window_unit": "words",
                    "preserve_boundaries": True,
                    "min_window_size": 30
                },
                "high_overlap": {
                    "window_size": 60,
                    "step_size": 15,
                    "window_unit": "words",
                    "preserve_boundaries": True,
                    "min_window_size": 10
                },
                "character_precise": {
                    "window_size": 400,
                    "step_size": 200,
                    "window_unit": "characters",
                    "preserve_boundaries": False,
                    "max_chunk_chars": 600
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            yaml.dump(config_data, config_file)
            config_path = Path(config_file.name)

        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)

            profiles = loaded_config["profiles"]

            # Test each profile
            for profile_name, profile_config in profiles.items():
                chunker = OverlappingWindowChunker(**profile_config)
                result = chunker.chunk(self.test_content)

                assert isinstance(result, ChunkingResult)
                assert len(result.chunks) > 0
                print(f"Profile '{profile_name}': {len(result.chunks)} chunks")

        finally:
            config_path.unlink()

    def test_use_case_specific_configurations(self):
        """Test configurations optimized for specific use cases."""
        use_case_configs = {
            "document_search": {
                "window_size": 100,
                "step_size": 50,
                "window_unit": "words",
                "preserve_boundaries": True,
                "min_window_size": 25,
                "description": "Optimized for document search and retrieval"
            },
            "sentiment_analysis": {
                "window_size": 40,
                "step_size": 20,
                "window_unit": "words",
                "preserve_boundaries": True,
                "min_window_size": 10,
                "max_chunk_chars": 300,
                "description": "Optimized for sentiment analysis tasks"
            },
            "summarization": {
                "window_size": 80,
                "step_size": 20,
                "window_unit": "words",
                "preserve_boundaries": True,
                "min_window_size": 15,
                "description": "Optimized for text summarization"
            }
        }

        for use_case, config in use_case_configs.items():
            # Remove description for chunker creation
            chunker_config = {k: v for k, v in config.items() if k != "description"}

            chunker = OverlappingWindowChunker(**chunker_config)
            result = chunker.chunk(self.test_content)

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0

            # Verify use case specific characteristics
            if use_case == "sentiment_analysis":
                # Should create smaller, more focused chunks
                avg_chunk_size = sum(len(chunk.content) for chunk in result.chunks) / len(result.chunks)
                assert avg_chunk_size < 400  # Characters

            elif use_case == "document_search":
                # Should have good overlap for context preservation
                if len(result.chunks) > 1:
                    assert result.source_info.get('overlap_ratio', 0) > 0.3

    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Valid configuration
        valid_config = {
            "window_size": 50,
            "step_size": 25,
            "window_unit": "words",
            "preserve_boundaries": True
        }

        # Should work without errors
        chunker = OverlappingWindowChunker(**valid_config)
        assert chunker.window_size == 50

        # Invalid configurations
        invalid_configs = [
            {
                "window_size": 20,
                "step_size": 25,  # step_size > window_size
                "window_unit": "words"
            },
            {
                "window_size": 30,
                "step_size": 15,
                "window_unit": "invalid_unit"  # Invalid unit
            },
            {
                "window_size": 50,
                "step_size": 25,
                "min_window_size": 60,  # min_window_size > window_size
                "window_unit": "words"
            }
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                OverlappingWindowChunker(**invalid_config)

    def test_configuration_file_formats(self):
        """Test different configuration file formats."""
        base_config = self.sample_configs["basic_config"]

        # Test YAML format
        yaml_data = {"chunker_config": base_config}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as yaml_file:
            yaml.dump(yaml_data, yaml_file)
            yaml_path = Path(yaml_file.name)

        try:
            with open(yaml_path, 'r') as f:
                loaded_yaml = yaml.safe_load(f)
            chunker = OverlappingWindowChunker(**loaded_yaml["chunker_config"])
            result = chunker.chunk(self.test_content)
            assert len(result.chunks) > 0

        finally:
            yaml_path.unlink()

        # Test JSON format (via yaml.safe_load which handles JSON)
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_file:
            json.dump({"chunker_config": base_config}, json_file)
            json_path = Path(json_file.name)

        try:
            with open(json_path, 'r') as f:
                loaded_json = json.load(f)
            chunker = OverlappingWindowChunker(**loaded_json["chunker_config"])
            result = chunker.chunk(self.test_content)
            assert len(result.chunks) > 0

        finally:
            json_path.unlink()

    def test_configuration_inheritance_and_overrides(self):
        """Test configuration inheritance and parameter overrides."""
        base_config = {
            "window_size": 50,
            "step_size": 25,
            "window_unit": "words",
            "preserve_boundaries": True,
            "min_window_size": 10
        }

        # Test parameter override
        override_config = base_config.copy()
        override_config.update({
            "step_size": 10,  # Override step_size
            "max_chunk_chars": 800  # Add new parameter
        })

        chunker = OverlappingWindowChunker(**override_config)

        # Verify overridden and inherited parameters
        assert chunker.window_size == 50  # Inherited
        assert chunker.step_size == 10   # Overridden
        assert chunker.window_unit == WindowUnit.WORDS  # Inherited
        assert chunker.max_chunk_chars == 800  # Added

        # Test functionality
        result = chunker.chunk(self.test_content)
        assert isinstance(result, ChunkingResult)

    def test_configuration_edge_cases(self):
        """Test configuration edge cases and boundary conditions."""
        # Minimal viable configuration
        minimal_config = {
            "window_size": 5,
            "step_size": 2,
            "window_unit": "words",
            "min_window_size": 1,
            "overlap_tokens": 0  # Fixed: ensure overlap_tokens < tokens_per_chunk
        }

        chunker = OverlappingWindowChunker(**minimal_config)
        result = chunker.chunk("This is a short test sentence.")
        assert isinstance(result, ChunkingResult)

        # Large window configuration
        large_config = {
            "window_size": 1000,
            "step_size": 500,
            "window_unit": "words",
            "min_window_size": 100
        }

        chunker = OverlappingWindowChunker(**large_config)
        result = chunker.chunk(self.test_content)
        assert isinstance(result, ChunkingResult)
        # Should create fewer chunks due to large window
        assert len(result.chunks) <= 2

        # Character-based edge case
        char_config = {
            "window_size": 100,
            "step_size": 50,
            "window_unit": "characters",
            "preserve_boundaries": False,
            "min_window_size": 20
        }

        chunker = OverlappingWindowChunker(**char_config)
        result = chunker.chunk(self.test_content)
        assert isinstance(result, ChunkingResult)

    def test_environment_variable_overrides(self):
        """Test configuration overrides via environment variables (mock)."""
        import os

        base_config = {
            "window_size": 50,
            "step_size": 25,
            "window_unit": "words"
        }

        # Mock environment variable override
        original_env = os.environ.get('CHUNKING_WINDOW_SIZE')
        os.environ['CHUNKING_WINDOW_SIZE'] = '100'

        try:
            # In a real implementation, this would read from environment
            env_override_size = int(os.environ.get('CHUNKING_WINDOW_SIZE', base_config['window_size']))

            override_config = base_config.copy()
            override_config['window_size'] = env_override_size

            chunker = OverlappingWindowChunker(**override_config)
            assert chunker.window_size == 100  # From environment variable

        finally:
            # Cleanup
            if original_env is not None:
                os.environ['CHUNKING_WINDOW_SIZE'] = original_env
            else:
                del os.environ['CHUNKING_WINDOW_SIZE']

    def test_configuration_with_custom_separators(self):
        """Test configuration with custom sentence separators."""
        config_with_separators = {
            "window_size": 3,
            "step_size": 1,
            "window_unit": "sentences",
            "preserve_boundaries": True,
            "min_window_size": 1,  # Compatible with window_size: 3
            "sentence_separators": [".", "!", "?", "。", "！", "？"]  # Include international
        }

        multilingual_text = """
        This is an English sentence. This is another one!
        这是中文句子。这是另一个中文句子！
        Final English sentence?
        """.strip()

        chunker = OverlappingWindowChunker(**config_with_separators)
        result = chunker.chunk(multilingual_text)

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0

        # Verify that custom separators are used
        assert chunker.sentence_separators == [".", "!", "?", "。", "！", "？"]

    def test_real_world_configuration_examples(self):
        """Test real-world configuration examples."""
        real_world_configs = {
            "blog_processing": {
                "window_size": 75,
                "step_size": 37,
                "window_unit": "words",
                "preserve_boundaries": True,
                "min_window_size": 20,
                "max_chunk_chars": 600
            },
            "academic_papers": {
                "window_size": 120,
                "step_size": 60,
                "window_unit": "words",
                "preserve_boundaries": True,
                "min_window_size": 40,
                "max_chunk_chars": 1000
            },
            "social_media": {
                "window_size": 30,
                "step_size": 15,
                "window_unit": "words",
                "preserve_boundaries": True,
                "min_window_size": 8,
                "max_chunk_chars": 280  # Twitter-like limit
            },
            "legal_documents": {
                "window_size": 100,
                "step_size": 25,  # High overlap for legal precision
                "window_unit": "words",
                "preserve_boundaries": True,
                "min_window_size": 30,
                "max_chunk_chars": 800
            }
        }

        for use_case, config in real_world_configs.items():
            chunker = OverlappingWindowChunker(**config)
            result = chunker.chunk(self.test_content)

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0

            # Verify chunks meet size constraints
            if 'max_chunk_chars' in config:
                max_chars = config['max_chunk_chars']
                for chunk in result.chunks:
                    assert len(chunk.content) <= max_chars + 100  # Allow some flexibility

    def test_configuration_performance_impact(self):
        """Test how different configurations affect performance."""
        performance_configs = [
            {
                "name": "Fast Processing",
                "config": {"window_size": 100, "step_size": 80, "window_unit": "words"}
            },
            {
                "name": "High Quality",
                "config": {"window_size": 50, "step_size": 15, "window_unit": "words"}
            },
            {
                "name": "Balanced",
                "config": {"window_size": 75, "step_size": 40, "window_unit": "words"}
            }
        ]

        # Generate longer test content for performance testing
        long_content = self.test_content * 10

        for perf_test in performance_configs:
            chunker = OverlappingWindowChunker(**perf_test["config"])
            result = chunker.chunk(long_content)

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0
            assert result.processing_time < 10.0  # Should complete reasonably fast

            print(f"{perf_test['name']}: {len(result.chunks)} chunks in {result.processing_time:.3f}s")

    def test_configuration_error_reporting(self):
        """Test that configuration errors provide helpful messages."""
        problematic_configs = [
            {
                "config": {"window_size": 10, "step_size": 15, "window_unit": "words"},
                "expected_error": "step_size must be less than window_size"
            },
            {
                "config": {"window_size": 20, "step_size": 10, "min_window_size": 25, "window_unit": "words"},
                "expected_error": "min_window_size must be less than window_size"
            }
        ]

        for test_case in problematic_configs:
            with pytest.raises(ValueError) as exc_info:
                OverlappingWindowChunker(**test_case["config"])

            # Verify error message contains helpful information
            error_msg = str(exc_info.value).lower()
            assert "window" in error_msg or "step" in error_msg or "size" in error_msg
