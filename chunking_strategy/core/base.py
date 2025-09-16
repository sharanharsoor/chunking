"""
Base classes and universal chunk schema for the chunking strategy library.

This module defines the fundamental interfaces and data structures used throughout
the library, including the universal chunk schema that supports text, images,
audio, video, tables, and mixed content.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Supported modality types for chunks."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABLE = "table"
    MIXED = "mixed"


@dataclass
class ChunkMetadata:
    """
    Universal metadata schema for chunks across all modalities.

    This schema is designed to work with embedding models and downstream
    processing while maintaining source information and positional context.
    """

    # Source information
    source: str  # filename, URL, or identifier
    source_type: str = "file"  # file, url, stream, etc.

    # Positional information (varies by modality)
    page: Optional[int] = None  # Document page number
    position: Optional[str] = None  # "paragraph 3", "line 45", etc.
    offset: Optional[int] = None  # Byte or character offset
    length: Optional[int] = None  # Length in appropriate units

    # Temporal information (audio/video)
    timestamp: Optional[Tuple[float, float]] = None  # (start, end) in seconds
    frame_range: Optional[Tuple[int, int]] = None  # (start_frame, end_frame)

    # Spatial information (images/video)
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
    coordinates: Optional[Dict[str, float]] = None  # Flexible coordinate system

    # Content information
    speaker: Optional[str] = None  # Audio/video speaker identification
    language: Optional[str] = None  # ISO language code
    encoding: Optional[str] = None  # Character encoding
    mime_type: Optional[str] = None  # MIME type of original content

    # Processing information
    chunker_used: Optional[str] = None  # Strategy that created this chunk
    processing_time: Optional[float] = None  # Time taken to create chunk

    # Quality metrics
    confidence: Optional[float] = None  # Confidence score (0.0 to 1.0)
    quality_score: Optional[float] = None  # Quality assessment score

    # Extensible metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class Chunk:
    """
    Universal chunk representation supporting all modalities.

    This is the primary output format for all chunking strategies and is designed
    to be compatible with embedding models and vector databases.

    Examples:
        Text chunk:
        ```python
        chunk = Chunk(
            id="doc1_para3",
            content="This is a paragraph of text...",
            modality=ModalityType.TEXT,
            metadata=ChunkMetadata(
                source="document.pdf",
                page=1,
                position="paragraph 3"
            )
        )
        ```

        Audio chunk:
        ```python
        chunk = Chunk(
            id="audio1_segment2",
            content=audio_bytes,
            modality=ModalityType.AUDIO,
            metadata=ChunkMetadata(
                source="speech.wav",
                timestamp=(30.5, 45.2),
                speaker="Speaker 1"
            )
        )
        ```
    """

    # Core fields
    id: str  # Unique identifier for the chunk
    content: Union[str, bytes, Any]  # The actual chunk content
    modality: ModalityType  # Type of content
    metadata: ChunkMetadata  # Associated metadata

    # Optional fields
    size: Optional[int] = None  # Size in appropriate units (chars, bytes, etc.)
    hash: Optional[str] = None  # Content hash for deduplication
    parent_id: Optional[str] = None  # Parent chunk for hierarchical chunking
    children_ids: Optional[List[str]] = None  # Child chunks

    def __post_init__(self):
        """Initialize computed fields after object creation."""
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.children_ids is None:
            self.children_ids = []
        if self.size is None and self.content is not None:
            if isinstance(self.content, str):
                self.size = len(self.content)
            elif isinstance(self.content, bytes):
                self.size = len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chunk to dictionary format for serialization.

        Returns:
            Dictionary representation compatible with JSON serialization.
        """
        return {
            "id": self.id,
            "content": self.content,
            "modality": self.modality.value,
            "metadata": self.metadata.to_dict(),
            "size": self.size,
            "hash": self.hash,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary representation."""
        metadata = ChunkMetadata(**data.get("metadata", {}))
        modality = ModalityType(data["modality"])

        return cls(
            id=data["id"],
            content=data["content"],
            modality=modality,
            metadata=metadata,
            size=data.get("size"),
            hash=data.get("hash"),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
        )


@dataclass
class ChunkingResult:
    """
    Complete result from a chunking operation.

    Contains the generated chunks along with processing metadata and statistics.
    """

    chunks: List[Chunk]  # Generated chunks
    total_chunks: int = field(init=False)  # Total number of chunks
    processing_time: Optional[float] = None  # Time taken for chunking
    strategy_used: Optional[str] = None  # Primary strategy used
    fallback_strategies: Optional[List[str]] = None  # Fallback strategies tried
    source_info: Optional[Dict[str, Any]] = None  # Source file information

    # Quality metrics
    avg_chunk_size: Optional[float] = None
    size_variance: Optional[float] = None
    quality_score: Optional[float] = None

    # Processing statistics
    memory_usage: Optional[float] = None  # Peak memory usage in MB
    errors: Optional[List[str]] = None  # Non-fatal errors encountered
    warnings: Optional[List[str]] = None  # Warnings during processing

    def __post_init__(self):
        """Compute derived fields after initialization."""
        self.total_chunks = len(self.chunks)
        if self.fallback_strategies is None:
            self.fallback_strategies = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

        # Compute basic statistics
        if self.chunks:
            sizes = [chunk.size for chunk in self.chunks if chunk.size is not None]
            if sizes:
                self.avg_chunk_size = sum(sizes) / len(sizes)
                if len(sizes) > 1:
                    variance = sum((s - self.avg_chunk_size) ** 2 for s in sizes) / len(sizes)
                    self.size_variance = variance ** 0.5

    def get_chunks_by_modality(self, modality: ModalityType) -> List[Chunk]:
        """Get all chunks of a specific modality."""
        return [chunk for chunk in self.chunks if chunk.modality == modality]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the chunking result."""
        modality_counts = {}
        for chunk in self.chunks:
            modality = chunk.modality.value
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        return {
            "total_chunks": self.total_chunks,
            "modality_distribution": modality_counts,
            "avg_chunk_size": self.avg_chunk_size,
            "size_variance": self.size_variance,
            "processing_time": self.processing_time,
            "strategy_used": self.strategy_used,
            "quality_score": self.quality_score,
        }


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.

    All chunking strategies must inherit from this class and implement the
    chunk() method. This ensures a consistent interface across all strategies.

    Attributes:
        name: Human-readable name for the chunker
        category: Category of the chunker (text, multimedia, etc.)
        supported_modalities: List of modalities this chunker supports
    """

    def __init__(
        self,
        name: str,
        category: str = "general",
        supported_modalities: Optional[List[ModalityType]] = None,
        **kwargs
    ):
        """
        Initialize the base chunker.

        Args:
            name: Human-readable name for the chunker
            category: Category classification
            supported_modalities: List of supported modality types
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.category = category
        self.supported_modalities = supported_modalities or [ModalityType.TEXT]
        self.config = kwargs
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    @abstractmethod
    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk the input content.

        Args:
            content: Input content to chunk (text, bytes, or file path)
            source_info: Information about the source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult containing the generated chunks and metadata

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the chunk() method")

    def supports_modality(self, modality: ModalityType) -> bool:
        """Check if this chunker supports a specific modality."""
        return modality in self.supported_modalities

    def validate_input(
        self,
        content: Union[str, bytes, Path],
        expected_modality: Optional[ModalityType] = None
    ) -> None:
        """
        Validate input content before processing.

        Args:
            content: Input content to validate
            expected_modality: Expected modality type

        Raises:
            ValueError: If input is invalid
            TypeError: If input type is not supported
        """
        if content is None:
            raise ValueError("Content cannot be None")

        if expected_modality and not self.supports_modality(expected_modality):
            raise ValueError(
                f"Chunker {self.name} does not support modality {expected_modality.value}"
            )

    def create_chunk(
        self,
        content: Union[str, bytes, Any],
        modality: ModalityType,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_id: Optional[str] = None
    ) -> Chunk:
        """
        Helper method to create a standardized chunk.

        Args:
            content: Chunk content
            modality: Content modality
            metadata: Additional metadata
            chunk_id: Optional custom chunk ID

        Returns:
            Properly formatted Chunk object
        """
        # Create metadata object
        meta_dict = metadata or {}
        meta_dict.setdefault("chunker_used", self.name)
        chunk_metadata = ChunkMetadata(**meta_dict)

        # Generate ID if not provided
        if chunk_id is None:
            chunk_id = str(uuid.uuid4())

        return Chunk(
            id=chunk_id,
            content=content,
            modality=modality,
            metadata=chunk_metadata
        )

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        self.config.update(kwargs)

    def __repr__(self) -> str:
        """String representation of the chunker."""
        return f"{self.__class__.__name__}(name='{self.name}', category='{self.category}')"


class StreamableChunker(BaseChunker):
    """
    Base class for chunkers that support streaming processing.

    Streaming chunkers can process large files without loading them entirely
    into memory, yielding chunks as they are generated.
    """

    @abstractmethod
    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Chunk content from a stream.

        Args:
            content_stream: Iterator yielding content pieces
            source_info: Information about the source
            **kwargs: Additional chunking parameters

        Yields:
            Individual chunks as they are generated
        """
        raise NotImplementedError("Streaming chunkers must implement chunk_stream()")


class AdaptableChunker(BaseChunker):
    """
    Base class for chunkers that support adaptive behavior.

    Adaptive chunkers can modify their behavior based on feedback signals
    such as retrieval accuracy, response quality, or performance metrics.
    """

    @abstractmethod
    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """
        Adapt chunker parameters based on feedback.

        Args:
            feedback_score: Numeric feedback score (0.0 to 1.0)
            feedback_type: Type of feedback (quality, performance, etc.)
            **kwargs: Additional feedback information
        """
        raise NotImplementedError("Adaptive chunkers must implement adapt_parameters()")

    @abstractmethod
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of adaptations made.

        Returns:
            List of adaptation records with timestamps and changes
        """
        raise NotImplementedError("Adaptive chunkers must implement get_adaptation_history()")


class HierarchicalChunker(BaseChunker):
    """
    Base class for chunkers that create hierarchical chunk structures.

    These chunkers create parent-child relationships between chunks,
    useful for multi-level document analysis and navigation.
    """

    @abstractmethod
    def chunk_hierarchical(
        self,
        content: Union[str, bytes, Path],
        max_levels: int = 3,
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Create hierarchical chunks with parent-child relationships.

        Args:
            content: Input content to chunk
            max_levels: Maximum hierarchy levels
            source_info: Information about the source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with hierarchical chunk structure
        """
        raise NotImplementedError("Hierarchical chunkers must implement chunk_hierarchical()")
