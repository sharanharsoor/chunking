"""
Fixed size chunking strategy.

This module implements fixed-size chunking, the most basic chunking strategy
that divides content into chunks of a specified size. This is the first
algorithm implementation and serves as a template for other strategies.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType,
    StreamableChunker
)
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage

logger = logging.getLogger(__name__)


@register_chunker(
    name="fixed_size",
    category="general",
    description="Divides content into fixed-size chunks with optional overlap",
    supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
    supported_formats=["txt", "md", "json", "csv", "*"],
    complexity=ComplexityLevel.LOW,
    dependencies=[],
    speed=SpeedLevel.VERY_FAST,
    memory=MemoryUsage.VERY_LOW,
    quality=0.3,  # Low quality due to no semantic awareness
    parameters_schema={
        "chunk_size": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000000,
            "default": 1024,
            "description": "Size of each chunk in characters/bytes"
        },
        "overlap_size": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10000,
            "default": 0,
            "description": "Number of characters/bytes to overlap between chunks"
        },
        "unit": {
            "type": "string",
            "enum": ["character", "byte", "word"],
            "default": "character",
            "description": "Unit for measuring chunk size"
        },
        "preserve_boundaries": {
            "type": "boolean",
            "default": False,
            "description": "Attempt to preserve word/line boundaries when possible"
        }
    },
    default_parameters={
        "chunk_size": 1024,
        "overlap_size": 0,
        "unit": "character",
        "preserve_boundaries": False
    },
    use_cases=["baseline", "benchmarking", "simple processing", "testing"],
    best_for=["uniform chunks", "predictable sizes", "fast processing"],
    limitations=["no semantic awareness", "may break words/sentences", "poor boundary quality"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=False
)
class FixedSizeChunker(StreamableChunker):
    """
    Fixed-size chunking strategy.

    Divides content into chunks of a specified size with optional overlap.
    This is the simplest and fastest chunking strategy, useful as a baseline
    and for scenarios where uniform chunk sizes are more important than
    semantic boundaries.

    Features:
    - Configurable chunk size and overlap
    - Support for character, byte, or word-based chunking
    - Optional boundary preservation
    - Streaming support for large files
    - Adaptive parameter adjustment

    Examples:
        Basic usage:
        ```python
        chunker = FixedSizeChunker(chunk_size=512)
        result = chunker.chunk("This is some text to chunk...")
        ```

        With overlap:
        ```python
        chunker = FixedSizeChunker(chunk_size=1024, overlap_size=100)
        result = chunker.chunk("Long document content...")
        ```

        Word-based chunking:
        ```python
        chunker = FixedSizeChunker(chunk_size=50, unit="word", preserve_boundaries=True)
        result = chunker.chunk("Document with word-based chunking...")
        ```
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        overlap_size: int = 0,
        unit: str = "character",
        preserve_boundaries: bool = False,
        **kwargs
    ):
        """
        Initialize the fixed-size chunker.

        Args:
            chunk_size: Size of each chunk in specified units
            overlap_size: Number of units to overlap between chunks
            unit: Unit for measuring size ("character", "byte", "word")
            preserve_boundaries: Whether to preserve word/line boundaries
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="fixed_size",
            category="general",
            supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
            **kwargs
        )

        # Validate parameters
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap_size < 0:
            raise ValueError("overlap_size cannot be negative")
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be less than chunk_size")
        if unit not in ["character", "byte", "word"]:
            raise ValueError("unit must be 'character', 'byte', or 'word'")

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.unit = unit
        self.preserve_boundaries = preserve_boundaries

        self.logger.info(
            f"Initialized FixedSizeChunker: size={chunk_size} {unit}, "
            f"overlap={overlap_size}, preserve_boundaries={preserve_boundaries}"
        )

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk content into fixed-size pieces.

        Args:
            content: Content to chunk (text, bytes, or file path)
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with generated chunks
        """
        start_time = time.time()

        # Handle different input types
        if isinstance(content, Path):
            file_path = Path(content)
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            actual_source = str(file_path)
        elif isinstance(content, str) and len(content) > 0 and len(content) < 500 and '\n' not in content:
            # Only treat as file path if it's short, has no newlines, actually exists, and is a file (not directory)
            try:
                if Path(content).exists() and Path(content).is_file():
                    file_path = Path(content)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    actual_source = str(file_path)
                else:
                    # Treat as direct text content
                    text_content = content
                    actual_source = "direct_input"
            except (OSError, ValueError):
                # Handle cases where content looks like a path but causes OS errors (e.g., too long)
                text_content = content
                actual_source = "direct_input"
        elif isinstance(content, bytes):
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                # Handle binary content
                return self._chunk_binary_content(content, source_info, start_time)
            actual_source = source_info.get("source", "bytes_input") if source_info else "bytes_input"
        else:
            text_content = str(content)
            actual_source = source_info.get("source", "text_input") if source_info else "text_input"

        # Validate input
        self.validate_input(text_content, ModalityType.TEXT)

        # Perform chunking based on unit
        if self.unit == "word":
            chunks = self._chunk_by_words(text_content, actual_source)
        elif self.unit == "byte":
            chunks = self._chunk_by_bytes(text_content, actual_source)
        else:  # character
            chunks = self._chunk_by_characters(text_content, actual_source)

        processing_time = time.time() - start_time

        # Create result
        result = ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used=self.name,
            source_info=source_info
        )

        self.logger.info(
            f"Fixed-size chunking completed: {len(chunks)} chunks in {processing_time:.3f}s"
        )

        return result

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream-process content into fixed-size chunks.

        Args:
            content_stream: Iterator yielding content pieces
            source_info: Information about the content source
            **kwargs: Additional parameters

        Yields:
            Individual chunks as they are generated
        """
        buffer = ""
        overlap_buffer = ""
        chunk_counter = 0

        for piece in content_stream:
            if isinstance(piece, bytes):
                try:
                    piece = piece.decode('utf-8')
                except UnicodeDecodeError:
                    self.logger.warning("Skipping non-UTF-8 content in stream")
                    continue

            buffer += piece

            # Process complete chunks from buffer
            while self._has_complete_chunk(buffer):
                chunk_content, remaining_buffer = self._extract_chunk_from_buffer(
                    overlap_buffer + buffer
                )

                # Create chunk
                chunk = self._create_chunk_from_content(
                    chunk_content,
                    chunk_counter,
                    source_info
                )

                yield chunk
                chunk_counter += 1

                # Setup overlap for next chunk
                if self.overlap_size > 0:
                    overlap_buffer = self._get_overlap_content(chunk_content)
                else:
                    overlap_buffer = ""

                buffer = remaining_buffer

        # Process any remaining content
        if buffer.strip():
            final_content = overlap_buffer + buffer
            if final_content.strip():
                chunk = self._create_chunk_from_content(
                    final_content,
                    chunk_counter,
                    source_info
                )
                yield chunk

    def _chunk_by_characters(self, content: str, source: str) -> List[Chunk]:
        """Chunk content by character count."""
        chunks = []
        content_length = len(content)

        if content_length == 0:
            return chunks

        start_pos = 0
        chunk_counter = 0

        while start_pos < content_length:
            # Calculate chunk boundaries
            chunk_start = max(0, start_pos - self.overlap_size)
            chunk_end = min(content_length, start_pos + self.chunk_size)

            # Extract chunk content
            chunk_content = content[chunk_start:chunk_end]

            # Apply boundary preservation if enabled
            if self.preserve_boundaries and chunk_end < content_length:
                chunk_content = self._preserve_boundaries(chunk_content, content, chunk_end)

            # Create chunk
            chunk = self._create_chunk_from_content(
                chunk_content,
                chunk_counter,
                {"source": source, "start_pos": chunk_start, "end_pos": chunk_start + len(chunk_content)}
            )

            chunks.append(chunk)
            chunk_counter += 1

            # Move to next position
            start_pos = chunk_end if self.overlap_size == 0 else start_pos + self.chunk_size

        return chunks

    def _chunk_by_words(self, content: str, source: str) -> List[Chunk]:
        """Chunk content by word count."""
        import re
        words = re.findall(r'\S+', content)

        if not words:
            return []

        chunks = []
        chunk_counter = 0
        start_idx = 0

        while start_idx < len(words):
            # Calculate word boundaries
            end_idx = min(len(words), start_idx + self.chunk_size)
            chunk_words = words[start_idx:end_idx]

            # Add overlap if needed
            if self.overlap_size > 0 and start_idx > 0:
                overlap_start = max(0, start_idx - self.overlap_size)
                overlap_words = words[overlap_start:start_idx]
                chunk_words = overlap_words + chunk_words

            # Join words back to text
            chunk_content = ' '.join(chunk_words)

            # Create chunk
            chunk = self._create_chunk_from_content(
                chunk_content,
                chunk_counter,
                {"source": source, "word_start": start_idx, "word_end": end_idx}
            )

            chunks.append(chunk)
            chunk_counter += 1

            # Move to next position
            start_idx = end_idx

        return chunks

    def _chunk_by_bytes(self, content: str, source: str) -> List[Chunk]:
        """Chunk content by byte count."""
        content_bytes = content.encode('utf-8')
        byte_length = len(content_bytes)

        if byte_length == 0:
            return []

        chunks = []
        chunk_counter = 0
        start_pos = 0

        while start_pos < byte_length:
            # Calculate byte boundaries
            chunk_start = max(0, start_pos - self.overlap_size)
            chunk_end = min(byte_length, start_pos + self.chunk_size)

            # Extract chunk bytes
            chunk_bytes = content_bytes[chunk_start:chunk_end]

            # Convert back to string, handling potential UTF-8 boundary issues
            try:
                chunk_content = chunk_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Find safe UTF-8 boundary
                chunk_content = self._find_safe_utf8_boundary(chunk_bytes)

            # Create chunk
            chunk = self._create_chunk_from_content(
                chunk_content,
                chunk_counter,
                {"source": source, "byte_start": chunk_start, "byte_end": chunk_start + len(chunk_bytes)}
            )

            chunks.append(chunk)
            chunk_counter += 1

            # Move to next position
            start_pos = chunk_end if self.overlap_size == 0 else start_pos + self.chunk_size

        return chunks

    def _chunk_binary_content(
        self,
        content: bytes,
        source_info: Optional[Dict[str, Any]],
        start_time: float
    ) -> ChunkingResult:
        """Handle binary content chunking."""
        chunks = []
        content_length = len(content)

        if content_length == 0:
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used=self.name,
                source_info=source_info
            )

        start_pos = 0
        chunk_counter = 0
        source = source_info.get("source", "binary_input") if source_info else "binary_input"

        while start_pos < content_length:
            # Calculate chunk boundaries
            chunk_start = max(0, start_pos - self.overlap_size)
            chunk_end = min(content_length, start_pos + self.chunk_size)

            # Extract chunk content
            chunk_content = content[chunk_start:chunk_end]

            # Create metadata
            metadata = ChunkMetadata(
                source=source,
                offset=chunk_start,
                length=len(chunk_content),
                chunker_used=self.name
            )

            # Create chunk
            chunk = Chunk(
                id=f"fixed_size_{chunk_counter}",
                content=chunk_content,
                modality=ModalityType.MIXED,
                metadata=metadata
            )

            chunks.append(chunk)
            chunk_counter += 1

            # Move to next position
            start_pos = chunk_end if self.overlap_size == 0 else start_pos + self.chunk_size

        return ChunkingResult(
            chunks=chunks,
            processing_time=time.time() - start_time,
            strategy_used=self.name,
            source_info=source_info
        )

    def _create_chunk_from_content(
        self,
        content: str,
        chunk_id: int,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """Create a chunk from content and metadata."""
        # Create metadata
        metadata = ChunkMetadata(
            source=extra_metadata.get("source", "unknown") if extra_metadata else "unknown",
            chunker_used=self.name,
            offset=extra_metadata.get("start_pos") if extra_metadata else None,
            length=len(content)
        )

        # Add extra metadata
        if extra_metadata:
            for key, value in extra_metadata.items():
                if key not in ["source", "start_pos"]:
                    metadata.extra[key] = value

        # Create chunk
        return Chunk(
            id=f"fixed_size_{chunk_id}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=metadata
        )

    def _preserve_boundaries(self, chunk_content: str, full_content: str, chunk_end: int) -> str:
        """Attempt to preserve word/sentence boundaries."""
        # Don't break words
        if chunk_end < len(full_content) and not full_content[chunk_end].isspace():
            # Find the last word boundary
            last_space = chunk_content.rfind(' ')
            if last_space > len(chunk_content) * 0.8:  # Only if we don't lose too much
                chunk_content = chunk_content[:last_space]

        return chunk_content

    def _find_safe_utf8_boundary(self, chunk_bytes: bytes) -> str:
        """Find a safe UTF-8 boundary to avoid decoding errors."""
        # Try progressively smaller chunks until we find valid UTF-8
        for i in range(len(chunk_bytes), 0, -1):
            try:
                return chunk_bytes[:i].decode('utf-8')
            except UnicodeDecodeError:
                continue

        # Fallback to empty string if nothing works
        return ""

    def _has_complete_chunk(self, buffer: str) -> bool:
        """Check if buffer has enough content for a complete chunk."""
        if self.unit == "word":
            import re
            words = re.findall(r'\S+', buffer)
            return len(words) >= self.chunk_size
        else:  # character or byte
            return len(buffer) >= self.chunk_size

    def _extract_chunk_from_buffer(self, buffer: str) -> tuple[str, str]:
        """Extract one chunk from buffer and return remaining content."""
        if self.unit == "word":
            import re
            words = re.findall(r'\S+', buffer)
            if len(words) >= self.chunk_size:
                chunk_words = words[:self.chunk_size]
                chunk_content = ' '.join(chunk_words)

                # Find where the chunk ends in the original buffer
                remaining_start = 0
                for word in chunk_words:
                    remaining_start = buffer.find(word, remaining_start) + len(word)

                remaining_buffer = buffer[remaining_start:].lstrip()
                return chunk_content, remaining_buffer
        else:  # character
            if len(buffer) >= self.chunk_size:
                chunk_content = buffer[:self.chunk_size]
                remaining_buffer = buffer[self.chunk_size:]
                return chunk_content, remaining_buffer

        return buffer, ""

    def _get_overlap_content(self, chunk_content: str) -> str:
        """Get overlap content from the end of a chunk."""
        if self.overlap_size <= 0:
            return ""

        if self.unit == "word":
            import re
            words = re.findall(r'\S+', chunk_content)
            if len(words) > self.overlap_size:
                overlap_words = words[-self.overlap_size:]
                return ' '.join(overlap_words) + ' '
        else:  # character
            if len(chunk_content) > self.overlap_size:
                return chunk_content[-self.overlap_size:]

        return chunk_content

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """
        Adapt chunker parameters based on feedback.

        Args:
            feedback_score: Feedback score (0.0 to 1.0)
            feedback_type: Type of feedback
            **kwargs: Additional feedback information
        """
        if feedback_type == "quality" and feedback_score < 0.5:
            # Increase chunk size for better quality
            old_size = self.chunk_size
            self.chunk_size = min(int(self.chunk_size * 1.2), 4096)
            self.logger.info(f"Adapted chunk_size: {old_size} -> {self.chunk_size} (quality feedback)")

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Decrease chunk size for better performance
            old_size = self.chunk_size
            # Use a more reasonable minimum - either 50 or 20% of original size, whichever is smaller
            min_size = min(50, int(self.chunk_size * 0.2))
            self.chunk_size = max(int(self.chunk_size * 0.8), min_size)
            self.logger.info(f"Adapted chunk_size: {old_size} -> {self.chunk_size} (performance feedback)")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        # For now, return empty list - could be enhanced to track changes
        return []
