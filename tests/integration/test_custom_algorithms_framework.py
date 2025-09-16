"""
Comprehensive Tests for Custom Algorithms Framework

This test suite covers all combinations and scenarios for the custom chunking
algorithms framework, including:

- Running only user-defined algorithms
- Running only existing algorithms
- Running both user-defined and existing algorithms together
- Running multiple user-defined algorithms at once
- Validation and error handling
- Configuration integration
- CLI integration
- Performance and reliability testing
"""

import pytest
import tempfile
import shutil
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

from chunking_strategy.core.base import BaseChunker, ChunkingResult, Chunk, ModalityType, ChunkMetadata
from chunking_strategy.core.registry import (
    get_registry,
    clear_custom_algorithms,
    unregister_chunker,
    create_chunker,
    list_chunkers
)
from chunking_strategy.core.custom_algorithm_loader import (
    CustomAlgorithmLoader,
    CustomAlgorithmError,
    ValidationError,
    LoadingError,
    get_custom_loader
)
from chunking_strategy.core.custom_config_integration import (
    CustomConfigProcessor,
    load_config_with_custom_algorithms,
    validate_custom_config_file,
    CustomConfigError
)
from chunking_strategy import ChunkerOrchestrator


class TestCustomAlgorithmsFramework:
    """Comprehensive test suite for custom algorithms framework."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment before each test."""
        # Create temporary directory for test files
        self.temp_dir = Path(tempfile.mkdtemp())

        # Store original registry state
        self.original_registry_state = get_registry().export_registry()

        # Clear any existing custom algorithms
        clear_custom_algorithms("custom")

        # Create test content
        self.test_content = """
        This is a test document with multiple paragraphs and sentences.

        It contains various types of content that can be used to test different
        chunking algorithms and their behavior.

        Some sentences are positive and happy! Others might be negative or sad.
        Most are neutral and informational like this one.

        The document has consistent formatting that can be parsed with patterns.
        Each paragraph serves a different purpose in testing.
        """

        yield

        # Cleanup after test
        try:
            # Clear custom algorithms
            clear_custom_algorithms("custom")

            # Remove temporary directory
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Cleanup warning: {e}")

    @pytest.fixture
    def sample_custom_chunker_code(self):
        """Generate sample custom chunker code for testing."""
        return '''
from chunking_strategy.core.base import BaseChunker, ChunkingResult, Chunk, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
import time

@register_chunker(
    name="test_custom_chunker",
    category="custom",
    description="Test custom chunker for framework testing",
    complexity=ComplexityLevel.LOW,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    quality=0.7,
    use_cases=["testing"],
    default_parameters={"chunk_size": 100}
)
class TestCustomChunker(BaseChunker):
    def __init__(self, chunk_size=100, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size

    def chunk(self, content, source_info=None, **kwargs):
        if hasattr(content, 'read'):
            content = content.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        content = str(content)

        chunks = []
        for i in range(0, len(content), self.chunk_size):
            chunk_content = content[i:i + self.chunk_size]
            metadata = ChunkMetadata(
                source=source_info.get('source', 'test') if source_info else 'test',
                chunker_used="test_custom_chunker"
            )
            chunk = Chunk(
                id=f"test_{i // self.chunk_size}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=metadata
            )
            chunks.append(chunk)

        return ChunkingResult(
            chunks=chunks,
            strategy_used="test_custom_chunker"
        )
'''

    @pytest.fixture
    def invalid_custom_chunker_code(self):
        """Generate invalid custom chunker code for error testing."""
        return '''
# Invalid custom chunker - missing required methods
from chunking_strategy.core.base import BaseChunker
from chunking_strategy.core.registry import register_chunker

@register_chunker(
    name="invalid_custom_chunker",
    category="custom"
)
class InvalidCustomChunker(BaseChunker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Missing required chunk() method
    pass
'''

    @pytest.fixture
    def multiple_custom_chunkers_code(self):
        """Generate code with multiple custom chunkers."""
        return '''
from chunking_strategy.core.base import BaseChunker, ChunkingResult, Chunk, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel

@register_chunker(
    name="test_chunker_alpha",
    category="custom",
    description="First test chunker"
)
class TestChunkerAlpha(BaseChunker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def chunk(self, content, source_info=None, **kwargs):
        content = str(content)
        chunks = [Chunk(
            id="alpha_0",
            content=content[:50],
            modality=ModalityType.TEXT,
            metadata=ChunkMetadata(source="test", chunker_used="test_chunker_alpha")
        )]
        return ChunkingResult(chunks=chunks, strategy_used="test_chunker_alpha")

@register_chunker(
    name="test_chunker_beta",
    category="custom",
    description="Second test chunker"
)
class TestChunkerBeta(BaseChunker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def chunk(self, content, source_info=None, **kwargs):
        content = str(content)
        chunks = [Chunk(
            id="beta_0",
            content=content[-50:],
            modality=ModalityType.TEXT,
            metadata=ChunkMetadata(source="test", chunker_used="test_chunker_beta")
        )]
        return ChunkingResult(chunks=chunks, strategy_used="test_chunker_beta")
'''

    # Tests for Custom Algorithm Loading

    def test_load_single_custom_algorithm(self, sample_custom_chunker_code):
        """Test loading a single custom algorithm."""
        # Create custom chunker file
        custom_file = self.temp_dir / "test_chunker.py"
        custom_file.write_text(sample_custom_chunker_code)

        # Load the custom algorithm
        loader = CustomAlgorithmLoader()
        algo_info = loader.load_algorithm(custom_file)

        assert algo_info is not None
        assert algo_info.name == "test_custom_chunker"
        assert algo_info.is_registered
        assert "test_custom_chunker" in list_chunkers()

        # Test using the custom algorithm
        chunker = create_chunker("test_custom_chunker", chunk_size=50)
        assert chunker is not None

        result = chunker.chunk(self.test_content)
        assert result.chunks
        assert result.strategy_used == "test_custom_chunker"

    def test_load_multiple_custom_algorithms(self, multiple_custom_chunkers_code):
        """Test loading multiple custom algorithms from one file."""
        # Create file with multiple chunkers
        custom_file = self.temp_dir / "multiple_chunkers.py"
        custom_file.write_text(multiple_custom_chunkers_code)

        # Load algorithms
        loader = CustomAlgorithmLoader()
        loaded_algorithms = loader.load_directory(self.temp_dir)

        # Should load both algorithms
        loaded_names = [algo.name for algo in loaded_algorithms]
        assert "test_chunker_alpha" in loaded_names
        assert "test_chunker_beta" in loaded_names

        # Both should be registered
        available_chunkers = list_chunkers()
        assert "test_chunker_alpha" in available_chunkers
        assert "test_chunker_beta" in available_chunkers

    def test_load_custom_algorithm_validation_error(self, invalid_custom_chunker_code):
        """Test loading invalid custom algorithm with strict validation."""
        # Create invalid custom chunker file
        invalid_file = self.temp_dir / "invalid_chunker.py"
        invalid_file.write_text(invalid_custom_chunker_code)

        # Load with strict validation - should fail
        loader = CustomAlgorithmLoader(strict_validation=True)

        with pytest.raises((ValidationError, CustomAlgorithmError)):
            loader.load_algorithm(invalid_file)

        # Load with lenient validation - should succeed but with warnings
        loader_lenient = CustomAlgorithmLoader(strict_validation=False)
        algo_info = loader_lenient.load_algorithm(invalid_file)

        # Should load but have validation errors
        assert algo_info is not None
        assert algo_info.loading_errors or algo_info.loading_warnings

    def test_load_directory_recursive(self, sample_custom_chunker_code):
        """Test loading custom algorithms from directory recursively."""
        # Create subdirectories with custom algorithms
        subdir1 = self.temp_dir / "subdir1"
        subdir2 = self.temp_dir / "subdir1" / "nested"
        subdir1.mkdir()
        subdir2.mkdir()

        # Create custom chunker files in different directories
        (subdir1 / "chunker1.py").write_text(sample_custom_chunker_code.replace(
            "test_custom_chunker", "dir_chunker_1"
        ))
        (subdir2 / "chunker2.py").write_text(sample_custom_chunker_code.replace(
            "test_custom_chunker", "dir_chunker_2"
        ))

        # Load recursively
        loader = CustomAlgorithmLoader()
        loaded_algorithms = loader.load_directory(self.temp_dir, recursive=True)

        loaded_names = [algo.name for algo in loaded_algorithms]
        assert "dir_chunker_1" in loaded_names
        assert "dir_chunker_2" in loaded_names

    def test_custom_algorithm_unloading(self, sample_custom_chunker_code):
        """Test unloading custom algorithms."""
        # Load custom algorithm
        custom_file = self.temp_dir / "test_chunker.py"
        custom_file.write_text(sample_custom_chunker_code)

        loader = CustomAlgorithmLoader()
        algo_info = loader.load_algorithm(custom_file)

        assert "test_custom_chunker" in list_chunkers()

        # Unload the algorithm
        success = loader.unload_algorithm("test_custom_chunker")
        assert success

        # Should no longer be in loaded algorithms
        assert "test_custom_chunker" not in loader.list_loaded_algorithms()

    # Tests for Configuration Integration

    def test_config_with_custom_algorithms(self, sample_custom_chunker_code):
        """Test configuration file with custom algorithms."""
        # Create custom algorithm file
        custom_file = self.temp_dir / "test_chunker.py"
        custom_file.write_text(sample_custom_chunker_code)

        # Create configuration with custom algorithm
        config = {
            "custom_algorithms": [
                {"path": str(custom_file)}
            ],
            "strategies": {
                "primary": "test_custom_chunker"
            },
            "parameters": {
                "test_custom_chunker": {
                    "chunk_size": 75
                }
            }
        }

        config_file = self.temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Load config with custom algorithms
        processor = CustomConfigProcessor()
        processed_config = processor.process_config(config, config_file)

        assert "custom_algorithms" in processed_config

        # Verify custom algorithm was loaded
        loaded_algos = processor.get_loaded_algorithms()
        assert "test_custom_chunker" in loaded_algos

    def test_orchestrator_with_custom_algorithms(self, sample_custom_chunker_code):
        """Test orchestrator using custom algorithms."""
        # Create custom algorithm and config
        custom_file = self.temp_dir / "test_chunker.py"
        custom_file.write_text(sample_custom_chunker_code)

        config = {
            "custom_algorithms": [{"path": str(custom_file)}],
            "strategies": {"primary": "test_custom_chunker"}
        }

        # Create orchestrator with custom algorithms
        orchestrator = ChunkerOrchestrator(config=config, enable_custom_algorithms=True)

        # Verify custom algorithm is loaded
        custom_algos = orchestrator.get_loaded_custom_algorithms()
        assert "test_custom_chunker" in custom_algos

        # Test chunking with custom algorithm
        test_file = self.temp_dir / "test.txt"
        test_file.write_text(self.test_content)

        result = orchestrator.chunk_file(test_file)
        assert result.strategy_used == "test_custom_chunker"
        assert result.chunks

    def test_mixed_custom_and_builtin_strategies(self, sample_custom_chunker_code):
        """Test using both custom and built-in algorithms together."""
        # Create custom algorithm
        custom_file = self.temp_dir / "test_chunker.py"
        custom_file.write_text(sample_custom_chunker_code)

        config = {
            "custom_algorithms": [{"path": str(custom_file)}],
            "strategies": {
                "primary": "test_custom_chunker",
                "fallbacks": ["fixed_size", "paragraph_based"]  # Built-in algorithms
            }
        }

        orchestrator = ChunkerOrchestrator(config=config, enable_custom_algorithms=True)

        # Test that both custom and built-in are available
        all_strategies = orchestrator.list_all_available_strategies()
        assert "test_custom_chunker" in all_strategies  # Custom
        assert "fixed_size" in all_strategies           # Built-in
        assert "paragraph_based" in all_strategies       # Built-in

        # Test chunking - should use custom algorithm first
        test_file = self.temp_dir / "test.txt"
        test_file.write_text(self.test_content)

        result = orchestrator.chunk_file(test_file)
        assert result.strategy_used == "test_custom_chunker"

    def test_multi_strategy_with_custom_algorithms(self, sample_custom_chunker_code):
        """Test multi-strategy mode with custom algorithms."""
        # Create custom algorithm
        custom_file = self.temp_dir / "test_chunker.py"
        custom_file.write_text(sample_custom_chunker_code)

        config = {
            "custom_algorithms": [{"path": str(custom_file)}],
            "multi_strategy": {
                "enabled": True,
                "strategies": [
                    {"name": "test_custom_chunker", "weight": 0.5},
                    {"name": "fixed_size", "weight": 0.5}
                ]
            }
        }

        orchestrator = ChunkerOrchestrator(config=config, enable_custom_algorithms=True)

        test_file = self.temp_dir / "test.txt"
        test_file.write_text(self.test_content)

        result = orchestrator.chunk_file(test_file)

        # Should run multiple strategies
        assert result.chunks
        # Multi-strategy should be indicated in result
        assert "multi_strategy" in result.strategy_used or len(result.fallback_strategies or []) > 0

    # Tests for Validation Framework

    def test_config_validation_valid(self, sample_custom_chunker_code):
        """Test validation of valid custom algorithm configuration."""
        custom_file = self.temp_dir / "test_chunker.py"
        custom_file.write_text(sample_custom_chunker_code)

        config = {
            "custom_algorithms": [{"path": str(custom_file)}],
            "strategies": {"primary": "test_custom_chunker"}
        }

        config_file = self.temp_dir / "valid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Validation should pass
        errors = validate_custom_config_file(config_file)
        assert len(errors) == 0

    def test_config_validation_invalid(self):
        """Test validation of invalid custom algorithm configuration."""
        # Invalid config - missing required 'path' field
        invalid_config = {
            "custom_algorithms": [
                {"algorithms": ["some_algorithm"]}  # Missing 'path'
            ]
        }

        config_file = self.temp_dir / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)

        # Validation should fail
        errors = validate_custom_config_file(config_file)
        assert len(errors) > 0
        assert any("path" in error.lower() for error in errors)

    def test_custom_algorithm_parameter_validation(self, sample_custom_chunker_code):
        """Test parameter validation for custom algorithms."""
        # Modify sample code to add parameter validation
        custom_code = sample_custom_chunker_code.replace(
            "def __init__(self, chunk_size=100, **kwargs):",
            """def __init__(self, chunk_size=100, **kwargs):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")"""
        )

        custom_file = self.temp_dir / "test_chunker.py"
        custom_file.write_text(custom_code)

        # Load the custom algorithm
        loader = CustomAlgorithmLoader()
        loader.load_algorithm(custom_file)

        # Valid parameters should work
        chunker = create_chunker("test_custom_chunker", chunk_size=50)
        assert chunker is not None

        # Invalid parameters should raise error
        with pytest.raises(ValueError):
            create_chunker("test_custom_chunker", chunk_size=-10)

    # Tests for Error Handling and Edge Cases

    def test_nonexistent_custom_algorithm_file(self):
        """Test loading non-existent custom algorithm file."""
        loader = CustomAlgorithmLoader()

        with pytest.raises(LoadingError):
            loader.load_algorithm("/nonexistent/path/to/chunker.py")

    def test_corrupted_custom_algorithm_file(self):
        """Test loading corrupted custom algorithm file."""
        # Create file with invalid Python syntax
        corrupted_file = self.temp_dir / "corrupted.py"
        corrupted_file.write_text("This is not valid Python syntax !@#$%")

        loader = CustomAlgorithmLoader(strict_validation=True)

        with pytest.raises(LoadingError):
            loader.load_algorithm(corrupted_file)

    def test_custom_algorithm_runtime_error(self, sample_custom_chunker_code):
        """Test custom algorithm that raises runtime errors."""
        # Modify sample code to raise error during chunking
        error_code = sample_custom_chunker_code.replace(
            "chunks = []",
            "raise RuntimeError('Test runtime error')"
        )

        custom_file = self.temp_dir / "error_chunker.py"
        custom_file.write_text(error_code)

        # Load the algorithm
        loader = CustomAlgorithmLoader()
        algo_info = loader.load_algorithm(custom_file)

        # Should load successfully
        assert algo_info is not None

        # But should fail during chunking
        chunker = create_chunker("test_custom_chunker")
        with pytest.raises(RuntimeError):
            chunker.chunk(self.test_content)

    # Tests for Performance and Reliability

    def test_large_number_of_custom_algorithms(self, sample_custom_chunker_code):
        """Test loading a large number of custom algorithms."""
        num_algorithms = 20

        # Create multiple custom algorithm files
        for i in range(num_algorithms):
            custom_code = sample_custom_chunker_code.replace(
                "test_custom_chunker", f"test_chunker_{i}"
            )
            custom_file = self.temp_dir / f"chunker_{i}.py"
            custom_file.write_text(custom_code)

        # Load all algorithms
        loader = CustomAlgorithmLoader()
        loaded_algorithms = loader.load_directory(self.temp_dir)

        # Should load all algorithms
        assert len(loaded_algorithms) == num_algorithms

        # All should be registered
        available_chunkers = list_chunkers()
        for i in range(num_algorithms):
            assert f"test_chunker_{i}" in available_chunkers

    def test_custom_algorithm_memory_cleanup(self, sample_custom_chunker_code):
        """Test that custom algorithms are properly cleaned up."""
        import gc
        import sys

        # Record initial module count
        initial_modules = len(sys.modules)

        # Load and unload custom algorithm multiple times
        for i in range(5):
            custom_file = self.temp_dir / f"temp_chunker_{i}.py"
            custom_file.write_text(
                sample_custom_chunker_code.replace(
                    "test_custom_chunker", f"temp_chunker_{i}"
                )
            )

            loader = CustomAlgorithmLoader()
            loader.load_algorithm(custom_file)
            loader.unload_algorithm(f"temp_chunker_{i}")

        # Force garbage collection
        gc.collect()

        # Module count should not grow excessively
        final_modules = len(sys.modules)
        # Allow some growth but not unlimited
        assert final_modules < initial_modules + 50

    def test_concurrent_custom_algorithm_loading(self, sample_custom_chunker_code):
        """Test loading custom algorithms concurrently (basic thread safety)."""
        import threading
        import time

        # Create multiple custom algorithm files
        for i in range(5):
            custom_code = sample_custom_chunker_code.replace(
                "test_custom_chunker", f"concurrent_chunker_{i}"
            )
            custom_file = self.temp_dir / f"concurrent_{i}.py"
            custom_file.write_text(custom_code)

        results = {}
        errors = {}

        def load_algorithm(index):
            try:
                loader = CustomAlgorithmLoader()
                custom_file = self.temp_dir / f"concurrent_{index}.py"
                algo_info = loader.load_algorithm(custom_file)
                results[index] = algo_info
            except Exception as e:
                errors[index] = e

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=load_algorithm, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout

        # Check results
        assert len(errors) == 0, f"Concurrent loading errors: {errors}"
        assert len(results) == 5

    # Integration Tests

    def test_end_to_end_custom_algorithm_workflow(self, sample_custom_chunker_code):
        """Test complete end-to-end workflow with custom algorithms."""
        # Step 1: Create custom algorithm
        custom_file = self.temp_dir / "e2e_chunker.py"
        custom_file.write_text(sample_custom_chunker_code.replace(
            "test_custom_chunker", "e2e_chunker"
        ))

        # Step 2: Create configuration
        config = {
            "custom_algorithms": [{"path": str(custom_file)}],
            "strategies": {
                "primary": "e2e_chunker",
                "fallbacks": ["fixed_size"]
            },
            "parameters": {
                "e2e_chunker": {"chunk_size": 80}
            },
            "output": {"format": "json", "include_metadata": True}
        }

        config_file = self.temp_dir / "e2e_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Step 3: Create test content file
        test_file = self.temp_dir / "e2e_test.txt"
        test_file.write_text(self.test_content)

        # Step 4: Process with orchestrator
        orchestrator = ChunkerOrchestrator(config_path=config_file)
        result = orchestrator.chunk_file(test_file)

        # Step 5: Verify results
        assert result.strategy_used == "e2e_chunker"
        assert result.chunks
        assert all(len(chunk.content) <= 80 for chunk in result.chunks)

        # Verify metadata
        for chunk in result.chunks:
            assert chunk.metadata is not None
            assert chunk.metadata.chunker_used == "e2e_chunker"

    def test_compare_custom_vs_builtin_algorithms(self, sample_custom_chunker_code):
        """Test comparing custom algorithms against built-in ones."""
        # Create custom algorithm
        custom_file = self.temp_dir / "comparison_chunker.py"
        custom_file.write_text(sample_custom_chunker_code.replace(
            "test_custom_chunker", "comparison_chunker"
        ))

        # Load custom algorithm
        loader = CustomAlgorithmLoader()
        loader.load_algorithm(custom_file)

        # Test both custom and built-in chunkers on same content
        custom_chunker = create_chunker("comparison_chunker", chunk_size=100)
        builtin_chunker = create_chunker("fixed_size", chunk_size=100)

        custom_result = custom_chunker.chunk(self.test_content)
        builtin_result = builtin_chunker.chunk(self.test_content)

        # Both should produce chunks
        assert custom_result.chunks
        assert builtin_result.chunks

        # Results should be comparable in structure
        assert len(custom_result.chunks) > 0
        assert len(builtin_result.chunks) > 0

        # Metadata should indicate different chunkers
        assert custom_result.strategy_used == "comparison_chunker"
        assert builtin_result.strategy_used == "fixed_size"

    # Test Running Only Custom Algorithms

    def test_run_only_custom_algorithms(self, multiple_custom_chunkers_code):
        """Test running only user-defined algorithms without built-in ones."""
        # Create multiple custom algorithms
        custom_file = self.temp_dir / "only_custom.py"
        custom_file.write_text(multiple_custom_chunkers_code)

        # Load custom algorithms
        loader = CustomAlgorithmLoader()
        loaded_algorithms = loader.load_directory(self.temp_dir)

        # Get only custom algorithm names
        custom_names = [algo.name for algo in loaded_algorithms]
        assert len(custom_names) >= 2

        # Create config using only custom algorithms
        config = {
            "custom_algorithms": [{"path": str(custom_file)}],
            "strategies": {
                "primary": custom_names[0],
                "fallbacks": custom_names[1:]
            }
        }

        orchestrator = ChunkerOrchestrator(config=config, enable_custom_algorithms=True)

        # Verify only custom algorithms are being used
        assert orchestrator.is_custom_algorithm(custom_names[0])

        # Test chunking with only custom algorithms
        test_file = self.temp_dir / "custom_only_test.txt"
        test_file.write_text(self.test_content)

        result = orchestrator.chunk_file(test_file)
        assert result.strategy_used in custom_names

    # Test Running Only Built-in Algorithms (baseline)

    def test_run_only_builtin_algorithms(self):
        """Test running only existing built-in algorithms."""
        # Create config with only built-in algorithms
        config = {
            "strategies": {
                "primary": "fixed_size",
                "fallbacks": ["paragraph_based", "sentence_based"]
            },
            "parameters": {
                "fixed_size": {"chunk_size": 100}
            }
        }

        # Disable custom algorithms
        orchestrator = ChunkerOrchestrator(config=config, enable_custom_algorithms=False)

        # Verify no custom algorithms are loaded
        assert len(orchestrator.get_loaded_custom_algorithms()) == 0

        # Test chunking with only built-in algorithms
        test_file = self.temp_dir / "builtin_only_test.txt"
        test_file.write_text(self.test_content)

        result = orchestrator.chunk_file(test_file)
        assert result.strategy_used == "fixed_size"
        assert result.chunks

    def test_algorithm_priority_and_fallbacks(self, sample_custom_chunker_code):
        """Test algorithm priority and fallback behavior with mixed custom/built-in."""
        # Create a custom algorithm that can fail
        failing_code = sample_custom_chunker_code.replace(
            "return ChunkingResult(",
            "raise RuntimeError('Custom algorithm failed'); return ChunkingResult("
        ).replace("test_custom_chunker", "failing_chunker")

        custom_file = self.temp_dir / "failing_chunker.py"
        custom_file.write_text(failing_code)

        # Also create a working custom algorithm
        working_file = self.temp_dir / "working_chunker.py"
        working_file.write_text(sample_custom_chunker_code.replace(
            "test_custom_chunker", "working_chunker"
        ))

        # Load both algorithms
        loader = CustomAlgorithmLoader(strict_validation=False)
        loader.load_algorithm(custom_file)
        loader.load_algorithm(working_file)

        # Create config with failing primary, working custom fallback, and built-in fallback
        config = {
            "custom_algorithms": [
                {"path": str(custom_file)},
                {"path": str(working_file)}
            ],
            "strategies": {
                "primary": "failing_chunker",
                "fallbacks": ["working_chunker", "fixed_size"]
            }
        }

        orchestrator = ChunkerOrchestrator(config=config, enable_custom_algorithms=True)

        test_file = self.temp_dir / "fallback_test.txt"
        test_file.write_text(self.test_content)

        # Should fall back to working_chunker when primary fails
        result = orchestrator.chunk_file(test_file)
        assert result.strategy_used == "working_chunker"
        assert result.chunks

    @pytest.mark.parametrize("algorithm_combination", [
        {"custom_only": ["test_chunker_alpha", "test_chunker_beta"]},
        {"builtin_only": ["fixed_size", "paragraph_based"]},
        {"mixed": ["test_chunker_alpha", "fixed_size", "paragraph_based"]},
        {"custom_primary": ["test_chunker_alpha", "fixed_size"]},
        {"builtin_primary": ["fixed_size", "test_chunker_alpha"]}
    ])
    def test_all_algorithm_combinations(self, multiple_custom_chunkers_code, algorithm_combination):
        """Test all possible combinations of custom and built-in algorithms."""
        # Load custom algorithms if needed
        if any("test_chunker" in alg for alg in sum(algorithm_combination.values(), [])):
            custom_file = self.temp_dir / "test_chunkers.py"
            custom_file.write_text(multiple_custom_chunkers_code)
            loader = CustomAlgorithmLoader()
            loader.load_directory(self.temp_dir)

        for combo_type, algorithms in algorithm_combination.items():
            if not algorithms:
                continue

            # Create config for this combination
            config = {
                "strategies": {
                    "primary": algorithms[0],
                    "fallbacks": algorithms[1:] if len(algorithms) > 1 else []
                }
            }

            # Add custom algorithms config if needed
            if any("test_chunker" in alg for alg in algorithms):
                config["custom_algorithms"] = [{"path": str(custom_file)}]

            orchestrator = ChunkerOrchestrator(config=config, enable_custom_algorithms=True)

            # Test chunking
            test_content = f"Test content for {combo_type} combination."
            result = orchestrator.chunk_content(test_content)

            assert result.chunks, f"No chunks produced for {combo_type} combination"
            assert result.strategy_used == algorithms[0], f"Wrong strategy used for {combo_type}"

    def test_stress_test_framework(self, sample_custom_chunker_code):
        """Stress test the custom algorithms framework."""
        import time
        import gc

        start_time = time.time()

        # Test with many custom algorithms and large content
        num_algorithms = 10
        large_content = self.test_content * 100  # Much larger content

        # Create many custom algorithms
        for i in range(num_algorithms):
            custom_code = sample_custom_chunker_code.replace(
                "test_custom_chunker", f"stress_chunker_{i}"
            ).replace("chunk_size=100", f"chunk_size={50 + i * 10}")

            custom_file = self.temp_dir / f"stress_chunker_{i}.py"
            custom_file.write_text(custom_code)

        # Load all algorithms
        loader = CustomAlgorithmLoader()
        loaded_algorithms = loader.load_directory(self.temp_dir)
        assert len(loaded_algorithms) == num_algorithms

        # Test each algorithm with large content
        for i in range(num_algorithms):
            chunker = create_chunker(f"stress_chunker_{i}")
            result = chunker.chunk(large_content)
            assert result.chunks
            assert len(result.chunks) > 10  # Should create many chunks

        # Cleanup and measure time
        for i in range(num_algorithms):
            loader.unload_algorithm(f"stress_chunker_{i}")

        gc.collect()

        elapsed_time = time.time() - start_time
        assert elapsed_time < 30, f"Stress test took too long: {elapsed_time}s"

    def test_framework_backwards_compatibility(self):
        """Test that the framework doesn't break existing functionality."""
        # Test that all existing chunkers still work
        existing_chunkers = ["fixed_size", "paragraph_based", "sentence_based"]

        for chunker_name in existing_chunkers:
            if chunker_name in list_chunkers():
                chunker = create_chunker(chunker_name)
                result = chunker.chunk(self.test_content)
                assert result.chunks, f"Existing chunker {chunker_name} failed"

        # Test that orchestrator still works without custom algorithms
        config = {
            "strategies": {"primary": "fixed_size"},
            "parameters": {"fixed_size": {"chunk_size": 200}}
        }

        orchestrator = ChunkerOrchestrator(config=config, enable_custom_algorithms=False)
        result = orchestrator.chunk_content(self.test_content)
        assert result.chunks
        assert result.strategy_used == "fixed_size"

    def tearDown(self):
        """Clean up after all tests."""
        # Clear any remaining custom algorithms
        clear_custom_algorithms("custom")

        # Force garbage collection
        import gc
        gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
