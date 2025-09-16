"""
Universal chunking framework - core architecture.

This module provides the base classes and registry for universal strategies
that can work with any file type through the content extraction layer.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from chunking_strategy.core.base import (
    BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
)
from chunking_strategy.core.extractors import (
    extract_content, ExtractedContent
)


class UniversalStrategy(ABC):
    """Base class for universal chunking strategies."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize universal strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def apply(
        self, 
        extracted_content: ExtractedContent, 
        config: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Apply universal strategy to extracted content.
        
        Args:
            extracted_content: Content extracted from any file type
            config: Strategy configuration
            
        Returns:
            ChunkingResult with universal chunks
        """
        pass


class UniversalStrategyRegistry:
    """Registry for universal chunking strategies."""
    
    def __init__(self):
        """Initialize empty registry."""
        self.strategies: Dict[str, UniversalStrategy] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, strategy: UniversalStrategy):
        """Register a universal strategy."""
        self.strategies[strategy.name] = strategy
        self.logger.debug(f"Registered universal strategy: {strategy.name}")
    
    def get(self, name: str) -> Optional[UniversalStrategy]:
        """Get strategy by name."""
        return self.strategies.get(name)
    
    def get_strategy(self, name: str) -> Optional[UniversalStrategy]:
        """Get strategy by name."""
        return self.strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self.strategies.keys())


# Global registry instance
_universal_registry = UniversalStrategyRegistry()


def get_universal_strategy_registry() -> UniversalStrategyRegistry:
    """Get the global universal strategy registry."""
    return _universal_registry


def apply_universal_strategy(
    strategy_name: str,
    file_path: Union[str, Path] = None,
    config: Optional[Dict[str, Any]] = None,
    content: Optional[str] = None,
    **kwargs
) -> ChunkingResult:
    """
    Apply a universal strategy to any file type or content.
    
    Args:
        strategy_name: Name of universal strategy to use
        file_path: Path to file to chunk (if chunking from file)
        config: Strategy configuration
        content: Raw text content to chunk (if chunking from string)
        **kwargs: Additional configuration parameters (merged with config)
        
    Returns:
        ChunkingResult with universal chunks
        
    Raises:
        ValueError: If strategy not found or content extraction fails
    """
    # Get the strategy
    registry = get_universal_strategy_registry()
    strategy = registry.get(strategy_name)
    
    if not strategy:
        available = registry.list_strategies()
        raise ValueError(f"Universal strategy '{strategy_name}' not found. Available: {available}")
    
    # Merge kwargs into config
    final_config = config or {}
    final_config.update(kwargs)
    
    # Handle content vs file_path
    if content is not None:
        # Create ExtractedContent from raw text
        from chunking_strategy.core.extractors import ExtractedContent, ModalityType
        extracted_content = ExtractedContent(
            text_content=content,
            modality=ModalityType.TEXT,
            metadata={"source": "string_input"}
        )
    elif file_path is not None:
        # Extract content from file
        try:
            extracted_content = extract_content(file_path)
        except Exception as e:
            raise ValueError(f"Failed to extract content from {file_path}: {e}")
    else:
        raise ValueError("Either file_path or content must be provided")
    
    # Apply strategy
    return strategy.apply(extracted_content, final_config)
