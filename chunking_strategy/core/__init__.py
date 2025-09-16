"""
Core components for the chunking strategy library.

This module contains the fundamental building blocks including:
- Base chunker interfaces
- Universal chunk schema
- Registry system
- Pipeline and streaming capabilities
- Adaptive chunking framework
- Quality metrics
"""

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkMetadata,
    ChunkingResult,
    ModalityType,
)
from chunking_strategy.core.registry import (
    ChunkerRegistry,
    register_chunker,
    get_chunker,
    list_chunkers,
    ChunkerMetadata,
)

__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkMetadata",
    "ChunkingResult",
    "ModalityType",
    "ChunkerRegistry",
    "register_chunker",
    "get_chunker",
    "list_chunkers",
    "ChunkerMetadata",
]
