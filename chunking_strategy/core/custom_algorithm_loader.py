"""
Custom Algorithm Loader for User-Defined Chunking Strategies.

This module provides dynamic loading and registration capabilities for user-defined
chunking algorithms. It allows users to provide their own algorithms in Python files
and seamlessly integrate them with the existing chunking library infrastructure.

Key Features:
- Dynamic loading of Python files containing custom chunkers
- Automatic discovery and validation of BaseChunker subclasses
- Integration with existing registry system
- Comprehensive error handling and validation
- Support for custom metadata and configuration schemas
- Hot-reloading capabilities for development

Example Usage:
    ```python
    from chunking_strategy.core.custom_algorithm_loader import CustomAlgorithmLoader

    loader = CustomAlgorithmLoader()

    # Load a single custom algorithm
    loader.load_algorithm("my_custom_chunker.py")

    # Load all algorithms from a directory
    loader.load_directory("custom_algorithms/")

    # Get loaded custom algorithms
    custom_algorithms = loader.list_loaded_algorithms()
    ```

Custom Algorithm File Example:
    ```python
    # my_custom_chunker.py
    from chunking_strategy.core.base import BaseChunker, ChunkingResult
    from chunking_strategy.core.registry import register_chunker, ComplexityLevel

    @register_chunker(
        name="my_custom_algorithm",
        category="text",
        description="My custom chunking algorithm",
        complexity=ComplexityLevel.MEDIUM,
        quality=0.8
    )
    class MyCustomChunker(BaseChunker):
        def __init__(self, custom_param=100, **kwargs):
            super().__init__(name="my_custom_algorithm", **kwargs)
            self.custom_param = custom_param

        def chunk(self, content, source_info=None, **kwargs):
            # Custom chunking logic here
            chunks = self._my_custom_logic(content)
            return ChunkingResult(
                chunks=chunks,
                strategy_used="my_custom_algorithm"
            )

        def _my_custom_logic(self, content):
            # Implementation details...
            pass
    ```
"""

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Tuple
import uuid
import traceback

from chunking_strategy.core.base import BaseChunker
from chunking_strategy.core.registry import (
    get_registry,
    unregister_chunker,
    ChunkerMetadata,
    ComplexityLevel,
    SpeedLevel,
    MemoryUsage
)

logger = logging.getLogger(__name__)


class CustomAlgorithmError(Exception):
    """Base exception for custom algorithm loading errors."""
    pass


class ValidationError(CustomAlgorithmError):
    """Raised when a custom algorithm fails validation."""
    pass


class LoadingError(CustomAlgorithmError):
    """Raised when a custom algorithm file cannot be loaded."""
    pass


class CustomAlgorithmInfo:
    """Information about a loaded custom algorithm."""

    def __init__(
        self,
        name: str,
        chunker_class: Type[BaseChunker],
        source_file: Path,
        metadata: Optional[ChunkerMetadata] = None,
        loading_errors: Optional[List[str]] = None,
        loading_warnings: Optional[List[str]] = None
    ):
        self.name = name
        self.chunker_class = chunker_class
        self.source_file = source_file
        self.metadata = metadata
        self.loading_errors = loading_errors or []
        self.loading_warnings = loading_warnings or []
        self.load_time = None
        self.is_registered = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "class_name": self.chunker_class.__name__,
            "source_file": str(self.source_file),
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "loading_errors": self.loading_errors,
            "loading_warnings": self.loading_warnings,
            "load_time": self.load_time,
            "is_registered": self.is_registered
        }


class CustomAlgorithmLoader:
    """
    Loader for user-defined chunking algorithms.

    This class handles the dynamic loading, validation, and registration of
    custom chunking algorithms provided by users in Python files.
    """

    def __init__(self, auto_register: bool = True, strict_validation: bool = True):
        """
        Initialize the custom algorithm loader.

        Args:
            auto_register: Automatically register loaded algorithms with global registry
            strict_validation: Enable strict validation of custom algorithms
        """
        self.auto_register = auto_register
        self.strict_validation = strict_validation
        self.loaded_algorithms: Dict[str, CustomAlgorithmInfo] = {}
        self.loaded_modules: Dict[str, Any] = {}  # Track loaded modules for hot-reloading
        self.validation_errors: List[str] = []
        self.loading_stats = {
            "total_attempted": 0,
            "successful_loads": 0,
            "validation_failures": 0,
            "loading_failures": 0
        }

    def load_algorithm(
        self,
        file_path: Union[str, Path],
        algorithm_name: Optional[str] = None,
        force_reload: bool = False,
        return_all: bool = False
    ) -> Optional[Union[CustomAlgorithmInfo, List[CustomAlgorithmInfo]]]:
        """
        Load a custom algorithm from a Python file.

        Args:
            file_path: Path to Python file containing custom algorithm
            algorithm_name: Optional specific algorithm name to load (if file contains multiple)
            force_reload: Force reload even if already loaded
            return_all: Return all algorithms found (List[CustomAlgorithmInfo]) instead of just one

        Returns:
            CustomAlgorithmInfo if successful (single), List[CustomAlgorithmInfo] if return_all=True, or None otherwise

        Raises:
            LoadingError: If the file cannot be loaded
            ValidationError: If validation fails in strict mode
        """
        file_path = Path(file_path)
        self.loading_stats["total_attempted"] += 1

        if not file_path.exists():
            error_msg = f"Algorithm file not found: {file_path}"
            logger.error(error_msg)
            if self.strict_validation:
                raise LoadingError(error_msg)
            return None

        if not file_path.suffix == ".py":
            error_msg = f"Algorithm file must be a Python file (.py): {file_path}"
            logger.error(error_msg)
            if self.strict_validation:
                raise LoadingError(error_msg)
            return None

        # Check if already loaded and handle reload
        file_key = str(file_path.absolute())
        if file_key in self.loaded_modules and not force_reload:
            logger.info(f"Algorithm file already loaded: {file_path}")
            # Return existing algorithm info if available
            for algo_info in self.loaded_algorithms.values():
                if str(algo_info.source_file.absolute()) == file_key:
                    return algo_info

        logger.info(f"Loading custom algorithm from: {file_path}")

        try:
            # Generate unique module name to avoid conflicts
            module_name = f"custom_algorithm_{uuid.uuid4().hex[:8]}"

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise LoadingError(f"Cannot create module spec for: {file_path}")

            module = importlib.util.module_from_spec(spec)

            # Add to sys.modules before execution to handle relative imports
            sys.modules[module_name] = module

            try:
                spec.loader.exec_module(module)
                self.loaded_modules[file_key] = module

            except Exception as e:
                # Clean up sys.modules on failure
                sys.modules.pop(module_name, None)
                raise LoadingError(f"Error executing module {file_path}: {str(e)}")

        except Exception as e:
            self.loading_stats["loading_failures"] += 1
            error_msg = f"Failed to load algorithm file {file_path}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            if self.strict_validation:
                raise LoadingError(error_msg)
            return None

        # Discover chunker classes in the module
        algorithms_found = self._discover_chunkers_in_module(module, file_path)

        if not algorithms_found:
            self.loading_stats["validation_failures"] += 1
            error_msg = f"No valid BaseChunker subclasses found in {file_path}"
            logger.warning(error_msg)
            if self.strict_validation:
                raise ValidationError(error_msg)
            return None

        # If specific algorithm requested, return only that one
        if algorithm_name:
            for algo_info in algorithms_found:
                if algo_info.name == algorithm_name:
                    self.loading_stats["successful_loads"] += 1
                    return self._process_loaded_algorithm(algo_info)

            error_msg = f"Algorithm '{algorithm_name}' not found in {file_path}"
            logger.error(error_msg)
            if self.strict_validation:
                raise ValidationError(error_msg)
            return None

        # Process all found algorithms
        processed_algorithms = []
        for algo_info in algorithms_found:
            processed = self._process_loaded_algorithm(algo_info)
            if processed:
                processed_algorithms.append(processed)

        if processed_algorithms:
            self.loading_stats["successful_loads"] += 1
            # Return all algorithms if requested, otherwise return the first one (primary)
            if return_all:
                return processed_algorithms
            else:
                return processed_algorithms[0]
        else:
            self.loading_stats["validation_failures"] += 1
            return None

    def load_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        pattern: str = "*.py"
    ) -> List[CustomAlgorithmInfo]:
        """
        Load all custom algorithms from a directory.

        Args:
            directory_path: Directory containing Python files with custom algorithms
            recursive: Search subdirectories recursively
            pattern: File pattern to match (default: "*.py")

        Returns:
            List of successfully loaded CustomAlgorithmInfo objects
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            error_msg = f"Directory not found: {directory_path}"
            logger.error(error_msg)
            if self.strict_validation:
                raise LoadingError(error_msg)
            return []

        if not directory_path.is_dir():
            error_msg = f"Path is not a directory: {directory_path}"
            logger.error(error_msg)
            if self.strict_validation:
                raise LoadingError(error_msg)
            return []

        logger.info(f"Loading custom algorithms from directory: {directory_path}")

        # Find Python files
        if recursive:
            py_files = list(directory_path.rglob(pattern))
        else:
            py_files = list(directory_path.glob(pattern))

        if not py_files:
            logger.warning(f"No Python files found in {directory_path}")
            return []

        loaded_algorithms = []

        for py_file in py_files:
            try:
                algo_results = self.load_algorithm(py_file, return_all=True)
                if algo_results:
                    # Handle both single algorithm and list of algorithms
                    if isinstance(algo_results, list):
                        loaded_algorithms.extend(algo_results)
                    else:
                        loaded_algorithms.append(algo_results)
            except Exception as e:
                logger.error(f"Failed to load {py_file}: {str(e)}")
                if self.strict_validation:
                    raise

        logger.info(f"Successfully loaded {len(loaded_algorithms)} custom algorithms from {directory_path}")
        return loaded_algorithms

    def _discover_chunkers_in_module(
        self,
        module: Any,
        source_file: Path
    ) -> List[CustomAlgorithmInfo]:
        """
        Discover BaseChunker subclasses in a loaded module.

        Args:
            module: Loaded Python module
            source_file: Path to the source file

        Returns:
            List of discovered algorithm info objects
        """
        discovered = []

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip non-BaseChunker classes
            if not issubclass(obj, BaseChunker) or obj is BaseChunker:
                continue

            # Skip imported classes (not defined in this module)
            if obj.__module__ != module.__name__:
                continue

            logger.debug(f"Found potential chunker class: {name}")

            # Validate the class
            validation_errors, validation_warnings = self._validate_chunker_class(obj)

            # Create algorithm info
            algo_info = CustomAlgorithmInfo(
                name=self._extract_algorithm_name(obj, name),
                chunker_class=obj,
                source_file=source_file,
                loading_errors=validation_errors,
                loading_warnings=validation_warnings
            )

            # Try to get metadata from registry if registered
            metadata = self._extract_metadata(obj)
            algo_info.metadata = metadata

            if validation_errors and self.strict_validation:
                error_msg = f"Validation failed for {name}: {'; '.join(validation_errors)}"
                logger.error(error_msg)
                raise ValidationError(error_msg)

            if validation_warnings:
                logger.warning(f"Validation warnings for {name}: {'; '.join(validation_warnings)}")

            discovered.append(algo_info)

        return discovered

    def _validate_chunker_class(self, chunker_class: Type) -> Tuple[List[str], List[str]]:
        """
        Validate a chunker class for compliance with BaseChunker interface.

        Args:
            chunker_class: Class to validate

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Check required methods
        if not hasattr(chunker_class, 'chunk'):
            errors.append("Missing required 'chunk' method")
        else:
            # Check chunk method signature
            chunk_method = getattr(chunker_class, 'chunk')
            if not callable(chunk_method):
                errors.append("'chunk' attribute is not callable")
            else:
                # Check method signature
                sig = inspect.signature(chunk_method)
                params = list(sig.parameters.keys())

                # Should have at least 'self' and 'content'
                if len(params) < 2:
                    errors.append("'chunk' method must accept at least 'content' parameter")
                elif 'content' not in params:
                    warnings.append("'chunk' method should have 'content' parameter")

        # Check constructor
        try:
            sig = inspect.signature(chunker_class.__init__)
            # Should accept **kwargs for flexibility
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_kwargs:
                warnings.append("Constructor should accept **kwargs for maximum compatibility")
        except Exception:
            warnings.append("Could not inspect constructor signature")

        # Check if class has proper inheritance
        if not issubclass(chunker_class, BaseChunker):
            errors.append("Class must inherit from BaseChunker")

        # Check for abstract method implementation
        try:
            # Try to instantiate with minimal parameters
            instance = chunker_class(name="test")
            if hasattr(instance, '__abstractmethods__') and instance.__abstractmethods__:
                errors.append(f"Abstract methods not implemented: {instance.__abstractmethods__}")
        except Exception as e:
            error_str = str(e)
            # Check if it's an abstract class error - this should be treated as an error
            if "abstract" in error_str.lower():
                errors.append(f"Cannot instantiate abstract class: {error_str}")
            else:
                warnings.append(f"Could not instantiate class for validation: {error_str}")

        return errors, warnings

    def _extract_algorithm_name(self, chunker_class: Type, class_name: str) -> str:
        """
        Extract algorithm name from class or metadata.

        Args:
            chunker_class: The chunker class
            class_name: Default class name

        Returns:
            Algorithm name
        """
        # Try to get name from registry metadata first
        registry = get_registry()
        for name, cls in registry._strategies.items():
            if cls is chunker_class:
                return name

        # Try to get from class attributes
        if hasattr(chunker_class, 'name'):
            return chunker_class.name

        # Fallback to class name, converted to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

        # Remove common suffixes
        name = name.replace('_chunker', '').replace('_algorithm', '').replace('chunker', '')

        return name or class_name.lower()

    def _extract_metadata(self, chunker_class: Type) -> Optional[ChunkerMetadata]:
        """
        Extract metadata from a chunker class.

        Args:
            chunker_class: The chunker class

        Returns:
            ChunkerMetadata if found, None otherwise
        """
        registry = get_registry()

        # Check if already registered
        for name, cls in registry._strategies.items():
            if cls is chunker_class:
                return registry.get_metadata(name)

        # Try to extract from class attributes
        try:
            name = self._extract_algorithm_name(chunker_class, chunker_class.__name__)

            metadata = ChunkerMetadata(
                name=name,
                category=getattr(chunker_class, 'category', 'custom'),
                description=getattr(chunker_class, '__doc__', '').strip() or f"Custom algorithm: {name}",
                complexity=getattr(chunker_class, 'complexity', ComplexityLevel.MEDIUM),
                speed=getattr(chunker_class, 'speed', SpeedLevel.MEDIUM),
                memory=getattr(chunker_class, 'memory', MemoryUsage.MEDIUM),
                quality=getattr(chunker_class, 'quality', 0.5),
                use_cases=getattr(chunker_class, 'use_cases', ['custom']),
                author="Custom Algorithm",
                maintainer="User"
            )

            return metadata

        except Exception as e:
            logger.debug(f"Could not extract metadata from {chunker_class.__name__}: {str(e)}")
            return None

    def _process_loaded_algorithm(self, algo_info: CustomAlgorithmInfo) -> Optional[CustomAlgorithmInfo]:
        """
        Process a loaded algorithm (register, validate, etc.).

        Args:
            algo_info: Algorithm information object

        Returns:
            Processed algorithm info or None if processing failed
        """
        try:
            # Store in loaded algorithms
            self.loaded_algorithms[algo_info.name] = algo_info

            # Auto-register if enabled
            if self.auto_register:
                self._register_algorithm(algo_info)

            logger.info(f"Successfully processed custom algorithm: {algo_info.name}")
            return algo_info

        except Exception as e:
            logger.error(f"Failed to process algorithm {algo_info.name}: {str(e)}")
            return None

    def _register_algorithm(self, algo_info: CustomAlgorithmInfo) -> bool:
        """
        Register algorithm with the global registry.

        Args:
            algo_info: Algorithm information

        Returns:
            True if registration successful
        """
        try:
            registry = get_registry()

            # Check if already registered (e.g., by @register_chunker decorator)
            if registry.get(algo_info.name):
                algo_info.is_registered = True
                logger.info(f"Custom algorithm already registered: {algo_info.name}")
                return True

            # If not already registered, register it now
            if algo_info.metadata:
                registry.register(algo_info.chunker_class, algo_info.metadata)
                algo_info.is_registered = True
                logger.info(f"Registered custom algorithm: {algo_info.name}")
                return True
            else:
                logger.warning(f"No metadata available for {algo_info.name}, skipping registration")

        except Exception as e:
            logger.error(f"Failed to register algorithm {algo_info.name}: {str(e)}")

        return False

    def unload_algorithm(self, name: str) -> bool:
        """
        Unload a custom algorithm.

        Args:
            name: Algorithm name to unload

        Returns:
            True if successfully unloaded
        """
        if name not in self.loaded_algorithms:
            logger.warning(f"Algorithm {name} not found in loaded algorithms")
            return False

        algo_info = self.loaded_algorithms[name]

        try:
            # Remove from registry if registered
            if algo_info.is_registered:
                success = unregister_chunker(name)
                if success:
                    logger.debug(f"Unregistered algorithm from registry: {name}")
                else:
                    logger.warning(f"Failed to unregister algorithm from registry: {name}")

            # Remove from loaded algorithms
            del self.loaded_algorithms[name]

            logger.info(f"Unloaded custom algorithm: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload algorithm {name}: {str(e)}")
            return False

    def reload_algorithm(self, name: str) -> Optional[CustomAlgorithmInfo]:
        """
        Reload a custom algorithm.

        Args:
            name: Algorithm name to reload

        Returns:
            Reloaded algorithm info or None if failed
        """
        if name not in self.loaded_algorithms:
            logger.error(f"Algorithm {name} not found in loaded algorithms")
            return None

        algo_info = self.loaded_algorithms[name]
        source_file = algo_info.source_file

        # Unload first
        self.unload_algorithm(name)

        # Reload
        return self.load_algorithm(source_file, force_reload=True)

    def list_loaded_algorithms(self) -> List[str]:
        """List names of all loaded custom algorithms."""
        return list(self.loaded_algorithms.keys())

    def get_algorithm_info(self, name: str) -> Optional[CustomAlgorithmInfo]:
        """Get information about a loaded algorithm."""
        return self.loaded_algorithms.get(name)

    def get_loading_stats(self) -> Dict[str, Any]:
        """Get statistics about algorithm loading."""
        stats = self.loading_stats.copy()
        stats["currently_loaded"] = len(self.loaded_algorithms)
        stats["success_rate"] = (
            stats["successful_loads"] / max(stats["total_attempted"], 1) * 100
        )
        return stats

    def validate_all_loaded(self) -> Dict[str, List[str]]:
        """
        Re-validate all loaded algorithms.

        Returns:
            Dictionary mapping algorithm names to validation errors
        """
        validation_results = {}

        for name, algo_info in self.loaded_algorithms.items():
            errors, warnings = self._validate_chunker_class(algo_info.chunker_class)
            if errors or warnings:
                validation_results[name] = {
                    "errors": errors,
                    "warnings": warnings
                }

        return validation_results

    def export_loaded_algorithms_info(self) -> Dict[str, Any]:
        """Export information about all loaded algorithms."""
        return {
            "algorithms": {
                name: info.to_dict()
                for name, info in self.loaded_algorithms.items()
            },
            "stats": self.get_loading_stats()
        }


# Global custom algorithm loader instance
_global_custom_loader = CustomAlgorithmLoader()


def get_custom_loader() -> CustomAlgorithmLoader:
    """Get the global custom algorithm loader instance."""
    return _global_custom_loader


def load_custom_algorithm(file_path: Union[str, Path], **kwargs) -> Optional[CustomAlgorithmInfo]:
    """Load a custom algorithm using the global loader."""
    return _global_custom_loader.load_algorithm(file_path, **kwargs)


def load_custom_algorithms_directory(directory_path: Union[str, Path], **kwargs) -> List[CustomAlgorithmInfo]:
    """Load custom algorithms from a directory using the global loader."""
    return _global_custom_loader.load_directory(directory_path, **kwargs)


def list_custom_algorithms() -> List[str]:
    """List all loaded custom algorithms."""
    return _global_custom_loader.list_loaded_algorithms()


def get_custom_algorithm_info(name: str) -> Optional[CustomAlgorithmInfo]:
    """Get information about a loaded custom algorithm."""
    return _global_custom_loader.get_algorithm_info(name)
