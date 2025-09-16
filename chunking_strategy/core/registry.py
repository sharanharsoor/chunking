"""
Registry system for chunking strategies.

This module provides a decorator-based registry system that allows chunking
strategies to register themselves with metadata including capabilities,
dependencies, performance characteristics, and usage information.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Callable, Union
from enum import Enum
import importlib
import pkg_resources

from chunking_strategy.core.base import BaseChunker, ModalityType

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Complexity levels for chunking strategies."""

    LOW = "low"          # Simple, fast, minimal dependencies
    MEDIUM = "medium"    # Moderate complexity, some dependencies
    HIGH = "high"        # Complex algorithms, heavy dependencies
    VERY_HIGH = "very_high"  # ML/AI models, significant resources


class SpeedLevel(Enum):
    """Performance speed classifications."""

    VERY_FAST = "very_fast"    # < 1ms per chunk
    FAST = "fast"              # 1-10ms per chunk
    MEDIUM = "medium"          # 10-100ms per chunk
    SLOW = "slow"              # 100ms-1s per chunk
    VERY_SLOW = "very_slow"    # > 1s per chunk


class MemoryUsage(Enum):
    """Memory usage classifications."""

    VERY_LOW = "very_low"      # < 10MB
    LOW = "low"                # 10-100MB
    MEDIUM = "medium"          # 100MB-1GB
    HIGH = "high"              # 1-10GB
    VERY_HIGH = "very_high"    # > 10GB


@dataclass
class ChunkerMetadata:
    """
    Comprehensive metadata for chunking strategies.

    This metadata helps users select appropriate strategies and understand
    their characteristics, dependencies, and performance implications.

    Example:
        ```python
        @register_chunker(
            name="semantic_chunking",
            category="text",
            complexity=ComplexityLevel.HIGH,
            supported_formats=["txt", "pdf", "docx"],
            dependencies=["sentence-transformers", "torch"],
            speed=SpeedLevel.SLOW,
            memory=MemoryUsage.HIGH,
            quality=0.9,
            parameters_schema={
                "model_name": {"type": "string", "default": "all-MiniLM-L6-v2"},
                "similarity_threshold": {"type": "number", "default": 0.7}
            },
            use_cases=["RAG", "semantic search", "content analysis"]
        )
        class SemanticChunker(BaseChunker):
            pass
        ```
    """

    # Basic information
    name: str
    category: str  # text, multimedia, document, general, etc.
    description: str = ""
    version: str = "1.0.0"

    # Capabilities
    supported_modalities: List[ModalityType] = field(default_factory=lambda: [ModalityType.TEXT])
    supported_formats: List[str] = field(default_factory=list)  # File extensions

    # Complexity and dependencies
    complexity: ComplexityLevel = ComplexityLevel.LOW
    dependencies: List[str] = field(default_factory=list)  # Required packages
    optional_dependencies: List[str] = field(default_factory=list)

    # Performance characteristics
    speed: SpeedLevel = SpeedLevel.MEDIUM
    memory: MemoryUsage = MemoryUsage.LOW
    quality: float = 0.5  # Quality score 0.0-1.0

    # Configuration
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    default_parameters: Dict[str, Any] = field(default_factory=dict)

    # Usage information
    use_cases: List[str] = field(default_factory=list)
    best_for: List[str] = field(default_factory=list)  # Specific scenarios
    limitations: List[str] = field(default_factory=list)

    # Technical details
    streaming_support: bool = False
    adaptive_support: bool = False
    hierarchical_support: bool = False

    # Maintenance info
    author: str = ""
    maintainer: str = ""
    license: str = "MIT"
    docs_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                result[key] = [item.value for item in value]
            else:
                result[key] = value
        return result


class ChunkerRegistry:
    """
    Global registry for chunking strategies.

    Manages registration, discovery, and instantiation of chunking strategies
    with comprehensive metadata support.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._strategies: Dict[str, Type[BaseChunker]] = {}
        self._metadata: Dict[str, ChunkerMetadata] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(
        self,
        chunker_class: Type[BaseChunker],
        metadata: ChunkerMetadata,
        allow_override: bool = False
    ) -> None:
        """
        Register a chunker class with metadata.

        Args:
            chunker_class: The chunker class to register
            metadata: Associated metadata
            allow_override: Allow overriding existing chunkers

        Raises:
            ValueError: If chunker name already exists and override not allowed
        """
        name = metadata.name

        if name in self._strategies:
            if allow_override:
                logger.warning(f"Overriding existing chunker: {name}")
            else:
                logger.warning(f"Chunker already registered: {name}, use allow_override=True to override")

        self._strategies[name] = chunker_class
        self._metadata[name] = metadata

        # Update category index
        category = metadata.category
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

        logger.debug(f"Registered chunker: {name} (category: {category})")

    def unregister(self, name: str) -> bool:
        """
        Unregister a chunker.

        Args:
            name: Name of the chunker to unregister

        Returns:
            True if successfully unregistered, False if not found
        """
        if name not in self._strategies:
            logger.warning(f"Chunker not found for unregistration: {name}")
            return False

        # Get metadata for category cleanup
        metadata = self._metadata.get(name)

        # Remove from strategies and metadata
        del self._strategies[name]
        del self._metadata[name]

        # Clean up category index
        if metadata:
            category = metadata.category
            if category in self._categories and name in self._categories[category]:
                self._categories[category].remove(name)
                # Remove empty category
                if not self._categories[category]:
                    del self._categories[category]

        logger.debug(f"Unregistered chunker: {name}")
        return True

    def is_registered(self, name: str) -> bool:
        """
        Check if a chunker is registered.

        Args:
            name: Name of the chunker to check

        Returns:
            True if registered, False otherwise
        """
        return name in self._strategies

    def get_registered_names(self) -> List[str]:
        """Get list of all registered chunker names."""
        return list(self._strategies.keys())

    def clear_custom_algorithms(self, category_filter: str = "custom") -> int:
        """
        Clear all custom algorithms (useful for testing and cleanup).

        Args:
            category_filter: Category to filter by (default: "custom")

        Returns:
            Number of algorithms removed
        """
        to_remove = []

        for name, metadata in self._metadata.items():
            if metadata.category == category_filter:
                to_remove.append(name)

        removed_count = 0
        for name in to_remove:
            if self.unregister(name):
                removed_count += 1

        logger.info(f"Cleared {removed_count} custom algorithms")
        return removed_count

    def get(self, name: str) -> Optional[Type[BaseChunker]]:
        """
        Get a chunker class by name.

        Args:
            name: Name of the chunker

        Returns:
            Chunker class or None if not found
        """
        return self._strategies.get(name)

    def get_metadata(self, name: str) -> Optional[ChunkerMetadata]:
        """
        Get metadata for a chunker.

        Args:
            name: Name of the chunker

        Returns:
            Chunker metadata or None if not found
        """
        return self._metadata.get(name)

    def list_chunkers(
        self,
        category: Optional[str] = None,
        modality: Optional[ModalityType] = None,
        complexity: Optional[ComplexityLevel] = None,
        dependencies_available: bool = True
    ) -> List[str]:
        """
        List available chunkers with optional filtering.

        Args:
            category: Filter by category
            modality: Filter by supported modality
            complexity: Filter by complexity level
            dependencies_available: Only include chunkers with available dependencies

        Returns:
            List of chunker names matching criteria
        """
        candidates = list(self._strategies.keys())

        if category:
            candidates = [name for name in candidates
                         if self._metadata[name].category == category]

        if modality:
            candidates = [name for name in candidates
                         if modality in self._metadata[name].supported_modalities]

        if complexity:
            candidates = [name for name in candidates
                         if self._metadata[name].complexity == complexity]

        if dependencies_available:
            candidates = [name for name in candidates
                         if self._check_dependencies(name)]

        return sorted(candidates)

    def list_categories(self) -> List[str]:
        """List all available categories."""
        return sorted(self._categories.keys())

    def get_category_chunkers(self, category: str) -> List[str]:
        """Get all chunkers in a specific category."""
        return self._categories.get(category, [])

    def create_chunker(
        self,
        name: str,
        **kwargs
    ) -> Optional[BaseChunker]:
        """
        Create an instance of a chunker.

        Args:
            name: Name of the chunker
            **kwargs: Parameters to pass to chunker constructor

        Returns:
            Chunker instance or None if not found/failed
        """
        chunker_class = self.get(name)
        if not chunker_class:
            logger.error(f"Chunker not found: {name}")
            return None

        try:
            metadata = self._metadata[name]

            # Merge default parameters with provided ones
            params = metadata.default_parameters.copy()
            params.update(kwargs)

            # Only add name parameter if not already explicitly handled by chunker class
            # Built-in chunkers handle name in their __init__, custom ones may need it injected
            try:
                # First try without injecting name (for built-in chunkers)
                return chunker_class(**params)
            except TypeError as e:
                if "missing 1 required positional argument: 'name'" in str(e):
                    # Custom chunker needs name parameter injected
                    params['name'] = name
                    return chunker_class(**params)
                else:
                    # Some other TypeError, re-raise
                    raise

        except ValueError as e:
            # Re-raise validation errors (like invalid parameter values)
            logger.error(f"Failed to create chunker {name}: {e}")
            raise e
        except Exception as e:
            # Log and return None for other errors (missing dependencies, etc.)
            logger.error(f"Failed to create chunker {name}: {e}")
            return None

    def check_dependencies(self, name: str) -> Dict[str, bool]:
        """
        Check availability of dependencies for a chunker.

        Args:
            name: Name of the chunker

        Returns:
            Dictionary mapping dependency names to availability status
        """
        metadata = self._metadata.get(name)
        if not metadata:
            return {}

        results = {}

        for dep in metadata.dependencies:
            results[dep] = self._is_package_available(dep)

        for dep in metadata.optional_dependencies:
            results[f"{dep} (optional)"] = self._is_package_available(dep)

        return results

    def _check_dependencies(self, name: str) -> bool:
        """Check if all required dependencies are available."""
        metadata = self._metadata.get(name)
        if not metadata:
            return False

        for dep in metadata.dependencies:
            if not self._is_package_available(dep):
                return False

        return True

    def _is_package_available(self, package_name: str) -> bool:
        """Check if a package is available for import."""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            try:
                # Try with pkg_resources for packages with different import names
                pkg_resources.get_distribution(package_name)
                return True
            except (pkg_resources.DistributionNotFound, Exception):
                return False

    def get_recommendations(
        self,
        modality: ModalityType,
        use_case: str,
        performance_priority: str = "balanced"  # speed, memory, quality, balanced
    ) -> List[str]:
        """
        Get recommended chunkers for specific requirements.

        Args:
            modality: Required modality support
            use_case: Target use case
            performance_priority: Performance optimization preference

        Returns:
            List of recommended chunker names in priority order
        """
        candidates = self.list_chunkers(modality=modality)

        # Filter by use case
        relevant = []
        for name in candidates:
            metadata = self._metadata[name]
            if (use_case.lower() in [uc.lower() for uc in metadata.use_cases] or
                use_case.lower() in [bf.lower() for bf in metadata.best_for]):
                relevant.append(name)

        if not relevant:
            relevant = candidates  # Fallback to all compatible

        # Sort by performance priority
        def score_chunker(name: str) -> float:
            metadata = self._metadata[name]

            if performance_priority == "speed":
                speed_scores = {
                    SpeedLevel.VERY_FAST: 1.0,
                    SpeedLevel.FAST: 0.8,
                    SpeedLevel.MEDIUM: 0.6,
                    SpeedLevel.SLOW: 0.4,
                    SpeedLevel.VERY_SLOW: 0.2
                }
                return speed_scores.get(metadata.speed, 0.5)

            elif performance_priority == "memory":
                memory_scores = {
                    MemoryUsage.VERY_LOW: 1.0,
                    MemoryUsage.LOW: 0.8,
                    MemoryUsage.MEDIUM: 0.6,
                    MemoryUsage.HIGH: 0.4,
                    MemoryUsage.VERY_HIGH: 0.2
                }
                return memory_scores.get(metadata.memory, 0.5)

            elif performance_priority == "quality":
                return metadata.quality

            else:  # balanced
                speed_score = score_chunker(name) if performance_priority != "speed" else 0.8
                memory_score = score_chunker(name) if performance_priority != "memory" else 0.8
                quality_score = metadata.quality
                return (speed_score + memory_score + quality_score) / 3.0

        return sorted(relevant, key=score_chunker, reverse=True)

    def export_registry(self) -> Dict[str, Any]:
        """Export complete registry data for serialization."""
        return {
            "strategies": list(self._strategies.keys()),
            "metadata": {name: meta.to_dict() for name, meta in self._metadata.items()},
            "categories": self._categories
        }


# Global registry instance
_global_registry = ChunkerRegistry()


def register_chunker(
    name: str,
    category: str = "general",
    description: str = "",
    supported_modalities: Optional[List[ModalityType]] = None,
    supported_formats: Optional[List[str]] = None,
    complexity: ComplexityLevel = ComplexityLevel.LOW,
    dependencies: Optional[List[str]] = None,
    optional_dependencies: Optional[List[str]] = None,
    speed: SpeedLevel = SpeedLevel.MEDIUM,
    memory: MemoryUsage = MemoryUsage.LOW,
    quality: float = 0.5,
    parameters_schema: Optional[Dict[str, Any]] = None,
    default_parameters: Optional[Dict[str, Any]] = None,
    use_cases: Optional[List[str]] = None,
    best_for: Optional[List[str]] = None,
    limitations: Optional[List[str]] = None,
    streaming_support: bool = False,
    adaptive_support: bool = False,
    hierarchical_support: bool = False,
    **kwargs
) -> Callable[[Type[BaseChunker]], Type[BaseChunker]]:
    """
    Decorator to register a chunker with metadata.

    Example:
        ```python
        @register_chunker(
            name="semantic_chunking",
            category="text",
            complexity=ComplexityLevel.HIGH,
            dependencies=["sentence-transformers"],
            quality=0.9,
            use_cases=["RAG", "semantic search"]
        )
        class SemanticChunker(BaseChunker):
            def chunk(self, content, **kwargs):
                # Implementation here
                pass
        ```

    Args:
        name: Unique name for the chunker
        category: Category classification
        description: Human-readable description
        supported_modalities: List of supported modality types
        supported_formats: List of supported file formats
        complexity: Algorithmic complexity level
        dependencies: Required package dependencies
        optional_dependencies: Optional package dependencies
        speed: Performance speed classification
        memory: Memory usage classification
        quality: Quality score (0.0 to 1.0)
        parameters_schema: JSON schema for parameters
        default_parameters: Default parameter values
        use_cases: List of primary use cases
        best_for: List of specific scenarios this chunker excels at
        limitations: List of known limitations
        streaming_support: Whether chunker supports streaming
        adaptive_support: Whether chunker supports adaptation
        hierarchical_support: Whether chunker supports hierarchical chunking
        **kwargs: Additional metadata fields

    Returns:
        Decorator function
    """
    def decorator(chunker_class: Type[BaseChunker]) -> Type[BaseChunker]:
        metadata = ChunkerMetadata(
            name=name,
            category=category,
            description=description,
            supported_modalities=supported_modalities or [ModalityType.TEXT],
            supported_formats=supported_formats or [],
            complexity=complexity,
            dependencies=dependencies or [],
            optional_dependencies=optional_dependencies or [],
            speed=speed,
            memory=memory,
            quality=quality,
            parameters_schema=parameters_schema or {},
            default_parameters=default_parameters or {},
            use_cases=use_cases or [],
            best_for=best_for or [],
            limitations=limitations or [],
            streaming_support=streaming_support,
            adaptive_support=adaptive_support,
            hierarchical_support=hierarchical_support,
            **kwargs
        )

        _global_registry.register(chunker_class, metadata)
        return chunker_class

    return decorator


def get_chunker(name: str) -> Optional[Type[BaseChunker]]:
    """Get a chunker class by name."""
    return _global_registry.get(name)


def list_chunkers(**kwargs) -> List[str]:
    """List available chunkers with optional filtering."""
    return _global_registry.list_chunkers(**kwargs)


def get_chunker_metadata(name: str) -> Optional[ChunkerMetadata]:
    """Get metadata for a chunker."""
    return _global_registry.get_metadata(name)


def create_chunker(name: str, **kwargs) -> Optional[BaseChunker]:
    """Create an instance of a chunker."""
    return _global_registry.create_chunker(name, **kwargs)


def unregister_chunker(name: str) -> bool:
    """Unregister a chunker from the global registry."""
    return _global_registry.unregister(name)


def is_chunker_registered(name: str) -> bool:
    """Check if a chunker is registered in the global registry."""
    return _global_registry.is_registered(name)


def clear_custom_algorithms(category_filter: str = "custom") -> int:
    """Clear custom algorithms from the global registry."""
    return _global_registry.clear_custom_algorithms(category_filter)


def get_registry() -> ChunkerRegistry:
    """Get the global registry instance."""
    return _global_registry
