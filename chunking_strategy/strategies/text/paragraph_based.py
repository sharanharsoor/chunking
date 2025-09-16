"""
Paragraph-based chunking strategy.

This module implements paragraph-aware chunking that respects paragraph boundaries
and can group multiple paragraphs into coherent chunks based on structure and content.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator

# Import centralized logging system
from chunking_strategy.logging_config import get_logger, user_info, performance_log

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType,
    StreamableChunker
)
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage

logger = get_logger(__name__)


@register_chunker(
    name="paragraph_based",
    category="text",
    description="Chunks text by grouping paragraphs with respect to paragraph boundaries",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "html", "json", "rtf"],
    complexity=ComplexityLevel.LOW,
    dependencies=[],
    optional_dependencies=[],
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    quality=0.8,  # Higher quality due to paragraph awareness
    parameters_schema={
        "max_paragraphs": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "default": 3,
            "description": "Maximum number of paragraphs per chunk"
        },
        "min_paragraphs": {
            "type": "integer",
            "minimum": 1,
            "maximum": 25,
            "default": 1,
            "description": "Minimum number of paragraphs per chunk"
        },
        "max_chunk_size": {
            "type": "integer",
            "minimum": 100,
            "maximum": 50000,
            "default": 2000,
            "description": "Maximum size of each chunk in characters"
        },
        "overlap_paragraphs": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "default": 0,
            "description": "Number of paragraphs to overlap between chunks"
        },
        "preserve_formatting": {
            "type": "boolean",
            "default": True,
            "description": "Whether to preserve original paragraph formatting"
        },
        "merge_short_paragraphs": {
            "type": "boolean",
            "default": True,
            "description": "Whether to merge very short paragraphs"
        },
        "min_paragraph_length": {
            "type": "integer",
            "minimum": 1,
            "maximum": 500,
            "default": 50,
            "description": "Minimum length for standalone paragraphs"
        }
    },
    use_cases=[
        "document_processing",
        "content_analysis",
        "structured_text_chunking",
        "academic_paper_processing",
        "book_chunking",
        "article_segmentation"
    ]
)
class ParagraphBasedChunker(StreamableChunker):
    """
    A chunker that splits text into paragraph-based chunks.

    This strategy is ideal for structured documents where paragraph boundaries
    represent logical content divisions. It maintains document structure while
    creating manageable chunks for processing.

    Example:
        >>> from chunking_strategy.strategies.text.paragraph_based import ParagraphBasedChunker
        >>> chunker = ParagraphBasedChunker(max_paragraphs=2)
        >>> text = '''First paragraph here.
        ...
        ... Second paragraph follows.
        ...
        ... Third paragraph continues.'''
        >>> result = chunker.chunk(text)
        >>> for chunk in result.chunks:
        ...     print(f"Content: {chunk.content[:50]}...")
        Content: First paragraph here....
        Content: Second paragraph follows....
    """

    def __init__(
        self,
        max_paragraphs: int = 3,
        min_paragraphs: int = 1,
        max_chunk_size: int = 2000,
        overlap_paragraphs: int = 0,
        preserve_formatting: bool = True,
        merge_short_paragraphs: bool = True,
        min_paragraph_length: int = 50,
        **kwargs
    ):
        """
        Initialize the ParagraphBasedChunker.

        Args:
            max_paragraphs: Maximum number of paragraphs per chunk
            min_paragraphs: Minimum number of paragraphs per chunk
            max_chunk_size: Maximum size of each chunk in characters
            overlap_paragraphs: Number of paragraphs to overlap between chunks
            preserve_formatting: Whether to preserve original formatting
            merge_short_paragraphs: Whether to merge very short paragraphs
            min_paragraph_length: Minimum length for standalone paragraphs
            **kwargs: Additional parameters
        """
        super().__init__(
            name="paragraph_based",
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        # Validate parameters
        if max_paragraphs <= 0:
            raise ValueError("max_paragraphs must be positive")
        if min_paragraphs <= 0:
            raise ValueError("min_paragraphs must be positive")
        if min_paragraphs > max_paragraphs:
            raise ValueError("min_paragraphs cannot be greater than max_paragraphs")
        if overlap_paragraphs >= max_paragraphs:
            raise ValueError("overlap_paragraphs must be less than max_paragraphs")
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if min_paragraph_length <= 0:
            raise ValueError("min_paragraph_length must be positive")

        self.max_paragraphs = max_paragraphs
        self.min_paragraphs = min_paragraphs
        self.max_chunk_size = max_chunk_size
        self.overlap_paragraphs = overlap_paragraphs
        self.preserve_formatting = preserve_formatting
        self.merge_short_paragraphs = merge_short_paragraphs
        self.min_paragraph_length = min_paragraph_length

        self.logger = logger.getChild(self.__class__.__name__)

        # Log initialization with key configuration
        logger.info(f"Initialized ParagraphBasedChunker: max_paragraphs={max_paragraphs}, "
                   f"overlap={overlap_paragraphs}, max_size={max_chunk_size} chars")

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk content into paragraph-based pieces.

        Args:
            content: Content to chunk
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with generated chunks
        """
        start_time = time.time()

        # Handle different input types
        if isinstance(content, Path):
            file_path = content
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            actual_source = str(file_path)
        elif isinstance(content, str) and len(content) > 0 and len(content) < 260 and '\n' not in content:
            # Check if string might be a file path (short, no newlines)
            try:
                file_path = Path(content)
                if file_path.is_file():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    actual_source = str(file_path)
                else:
                    # Treat as text content
                    text_content = content
                    actual_source = source_info.get("source", "string") if source_info else "string"
            except (OSError, IOError):
                # If any filesystem error, treat as text content
                text_content = content
                actual_source = source_info.get("source", "string") if source_info else "string"
        elif isinstance(content, bytes):
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                raise ValueError("Cannot decode bytes as UTF-8 for paragraph-based chunking")
            actual_source = source_info.get("source", "bytes") if source_info else "bytes"
        else:
            # Treat as string content
            text_content = str(content)
            actual_source = source_info.get("source", "string") if source_info else "string"

        # Split text into paragraphs
        paragraphs = self._split_paragraphs(text_content)

        # Merge short paragraphs if enabled
        if self.merge_short_paragraphs:
            paragraphs = self._merge_short_paragraphs(paragraphs)

        # Group paragraphs into chunks
        chunks = self._group_paragraphs_into_chunks(paragraphs, actual_source)

        processing_time = time.time() - start_time

        # Log completion with performance details
        logger.info(f"Paragraph-based chunking completed: {len(paragraphs)} paragraphs -> {len(chunks)} chunks in {processing_time:.3f}s")

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used=self.name,
            source_info=source_info,
            avg_chunk_size=sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0,
            size_variance=0.0  # Will be calculated by metrics if needed
        )

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream-process content into paragraph-based chunks.

        Args:
            content_stream: Iterator yielding content pieces
            source_info: Information about the content source
            **kwargs: Additional parameters

        Yields:
            Individual chunks as they are generated
        """
        paragraph_buffer = []
        overlap_buffer = []
        chunk_counter = 0
        text_buffer = ""

        for piece in content_stream:
            if isinstance(piece, bytes):
                try:
                    piece = piece.decode('utf-8')
                except UnicodeDecodeError:
                    self.logger.warning("Skipping non-UTF-8 content in stream")
                    continue

            text_buffer += piece

            # Look for paragraph boundaries in the buffer
            while "\n\n" in text_buffer or "\r\n\r\n" in text_buffer:
                # Find the earliest paragraph boundary
                double_newline_pos = text_buffer.find("\n\n")
                double_crlf_pos = text_buffer.find("\r\n\r\n")

                if double_newline_pos == -1:
                    boundary_pos = double_crlf_pos
                    boundary_len = 4
                elif double_crlf_pos == -1:
                    boundary_pos = double_newline_pos
                    boundary_len = 2
                else:
                    if double_newline_pos < double_crlf_pos:
                        boundary_pos = double_newline_pos
                        boundary_len = 2
                    else:
                        boundary_pos = double_crlf_pos
                        boundary_len = 4

                # Extract the paragraph
                paragraph_text = text_buffer[:boundary_pos].strip()
                text_buffer = text_buffer[boundary_pos + boundary_len:]

                if paragraph_text:
                    paragraph_buffer.append(paragraph_text)

                    # Check if we have enough paragraphs for a chunk
                    if len(paragraph_buffer) >= self.max_paragraphs:
                        # Create chunk with overlap from previous chunk
                        chunk_paragraphs = overlap_buffer + paragraph_buffer[:self.max_paragraphs]
                        chunk = self._create_chunk_from_paragraphs(
                            chunk_paragraphs,
                            chunk_counter,
                            {"source": source_info.get("source", "stream") if source_info else "stream"}
                        )
                        yield chunk
                        chunk_counter += 1

                        # Set up overlap for next chunk
                        if self.overlap_paragraphs > 0:
                            overlap_start = max(0, len(chunk_paragraphs) - self.overlap_paragraphs)
                            overlap_buffer = chunk_paragraphs[overlap_start:]
                        else:
                            overlap_buffer = []

                        # Keep remaining paragraphs for next chunk
                        paragraph_buffer = paragraph_buffer[self.max_paragraphs:]

        # Process remaining content
        if text_buffer.strip():
            paragraph_buffer.append(text_buffer.strip())

        # Yield final chunk if we have paragraphs left
        if paragraph_buffer:
            chunk_paragraphs = overlap_buffer + paragraph_buffer
            chunk = self._create_chunk_from_paragraphs(
                chunk_paragraphs,
                chunk_counter,
                {"source": source_info.get("source", "stream") if source_info else "stream"}
            )
            yield chunk

    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Input text to split

        Returns:
            List of paragraph strings
        """
        if not text.strip():
            return []

        # Split on double newlines (common paragraph separator)
        # Handle both \n\n and \r\n\r\n
        paragraphs = re.split(r'\r?\n\s*\r?\n', text.strip())

        # Clean up paragraphs
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            # Remove extra whitespace but preserve structure if requested
            if self.preserve_formatting:
                # Keep internal line breaks but normalize spacing
                cleaned = re.sub(r'\s+', ' ', paragraph.strip())
            else:
                # Normalize all whitespace
                cleaned = ' '.join(paragraph.split())

            if cleaned:
                cleaned_paragraphs.append(cleaned)

        return cleaned_paragraphs

    def _merge_short_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Merge very short paragraphs with adjacent ones.

        Args:
            paragraphs: List of paragraph strings

        Returns:
            List of paragraphs with short ones merged
        """
        if not paragraphs:
            return paragraphs

        merged = []
        current_paragraph = ""

        for paragraph in paragraphs:
            if len(paragraph) < self.min_paragraph_length:
                # Short paragraph - merge with current
                if current_paragraph:
                    current_paragraph += "\n\n" + paragraph
                else:
                    current_paragraph = paragraph
            else:
                # Normal length paragraph
                if current_paragraph:
                    # Finish the current merged paragraph
                    merged.append(current_paragraph)
                    current_paragraph = paragraph
                else:
                    # Start new paragraph
                    current_paragraph = paragraph

        # Add the final paragraph
        if current_paragraph:
            merged.append(current_paragraph)

        return merged

    def _group_paragraphs_into_chunks(self, paragraphs: List[str], source: str) -> List[Chunk]:
        """Group paragraphs into chunks according to configuration."""
        if not paragraphs:
            return []

        chunks = []
        chunk_counter = 0
        i = 0

        while i < len(paragraphs):
            chunk_paragraphs = []
            chunk_size = 0
            paragraphs_added = 0

            # Add paragraphs until we reach limits
            while (i < len(paragraphs) and
                   paragraphs_added < self.max_paragraphs and
                   chunk_size < self.max_chunk_size):

                paragraph = paragraphs[i]
                paragraph_size = len(paragraph)

                # Check if adding this paragraph would exceed size limit
                if chunk_size + paragraph_size > self.max_chunk_size and chunk_paragraphs:
                    break

                chunk_paragraphs.append(paragraph)
                chunk_size += paragraph_size
                paragraphs_added += 1
                i += 1

            # Ensure we meet minimum requirements
            if chunk_paragraphs and paragraphs_added >= self.min_paragraphs:
                chunk = self._create_chunk_from_paragraphs(chunk_paragraphs, chunk_counter, {"source": source})
                chunks.append(chunk)
                chunk_counter += 1

                # Handle overlap
                if self.overlap_paragraphs > 0 and i < len(paragraphs):
                    overlap_start = max(0, len(chunk_paragraphs) - self.overlap_paragraphs)
                    i -= len(chunk_paragraphs) - overlap_start
            elif chunk_paragraphs:
                # If we don't meet minimum, still create chunk (edge case)
                chunk = self._create_chunk_from_paragraphs(chunk_paragraphs, chunk_counter, {"source": source})
                chunks.append(chunk)
                chunk_counter += 1

        return chunks

    def _create_chunk_from_paragraphs(
        self,
        paragraphs: List[str],
        chunk_id: int,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """Create a chunk from a list of paragraphs."""
        # Join paragraphs with double newlines to preserve structure
        if self.preserve_formatting:
            content = '\n\n'.join(paragraphs)
        else:
            content = ' '.join(paragraphs)

        # Create metadata
        metadata = ChunkMetadata(
            source=extra_metadata.get("source", "unknown") if extra_metadata else "unknown",
            chunker_used=self.name,
            length=len(content),
            extra={
                "paragraph_count": len(paragraphs),
                "avg_paragraph_length": sum(len(p) for p in paragraphs) / len(paragraphs),
                "first_paragraph_preview": paragraphs[0][:100] + "..." if paragraphs and len(paragraphs[0]) > 100 else paragraphs[0] if paragraphs else "",
                "formatting_preserved": self.preserve_formatting,
                "merged_short_paragraphs": self.merge_short_paragraphs
            }
        )

        return Chunk(
            id=f"{self.name}_{chunk_id}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=metadata,
            size=len(content)
        )

    def adapt_parameters(self, feedback_score: float, feedback_type: str) -> None:
        """
        Adapt chunking parameters based on feedback.

        Args:
            feedback_score: Score indicating quality (0.0 to 1.0)
            feedback_type: Type of feedback ("quality", "coverage", "speed", etc.)
        """
        if feedback_type == "quality" and feedback_score < 0.5:
            # Poor quality - try smaller chunks
            if self.max_paragraphs > 1:
                self.max_paragraphs = max(1, self.max_paragraphs - 1)
                self.logger.info(f"Reduced max_paragraphs to {self.max_paragraphs} due to quality feedback")

        elif feedback_type == "coverage" and feedback_score < 0.5:
            # Poor coverage - try larger chunks
            if self.max_paragraphs < 10:
                self.max_paragraphs += 1
                self.logger.info(f"Increased max_paragraphs to {self.max_paragraphs} due to coverage feedback")

        elif feedback_type == "speed" and feedback_score < 0.5:
            # Slow processing - try simpler settings
            self.merge_short_paragraphs = False
            self.preserve_formatting = False
            self.logger.info("Simplified settings for better speed")

    def get_info(self) -> Dict[str, Any]:
        """Get information about the chunker configuration."""
        return {
            "name": self.name,
            "category": "text",
            "type": "paragraph_based",
            "parameters": {
                "max_paragraphs": self.max_paragraphs,
                "min_paragraphs": self.min_paragraphs,
                "max_chunk_size": self.max_chunk_size,
                "overlap_paragraphs": self.overlap_paragraphs,
                "preserve_formatting": self.preserve_formatting,
                "merge_short_paragraphs": self.merge_short_paragraphs,
                "min_paragraph_length": self.min_paragraph_length
            },
            "features": [
                "paragraph_boundary_detection",
                "smart_paragraph_merging",
                "formatting_preservation",
                "overlap_support",
                "size_limiting",
                "streaming_support",
                "adaptive_parameters"
            ],
            "use_cases": [
                "document_processing",
                "content_analysis",
                "structured_text_chunking",
                "academic_paper_processing",
                "book_chunking",
                "article_segmentation"
            ]
        }
