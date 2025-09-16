"""
Fixed-length word chunking strategy.

This module implements word-aware chunking that creates chunks with a specified
number of words while maintaining word boundaries for better semantic coherence.
"""

import logging
import re
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
    name="fixed_length_word",
    category="text",
    description="Chunks text by grouping a fixed number of words with word boundary awareness",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "html", "json", "rtf"],
    complexity=ComplexityLevel.LOW,
    dependencies=[],
    optional_dependencies=[],
    speed=SpeedLevel.VERY_FAST,
    memory=MemoryUsage.LOW,
    quality=0.6,  # Good basic quality due to word awareness
    parameters_schema={
        "words_per_chunk": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10000,
            "default": 100,
            "description": "Number of words per chunk"
        },
        "overlap_words": {
            "type": "integer",
            "minimum": 0,
            "maximum": 500,
            "default": 0,
            "description": "Number of words to overlap between chunks"
        },
        "max_chunk_size": {
            "type": "integer",
            "minimum": 100,
            "maximum": 100000,
            "default": 5000,
            "description": "Maximum chunk size in characters (safety limit)"
        },
        "word_tokenization": {
            "type": "string",
            "enum": ["simple", "whitespace", "regex"],
            "default": "simple",
            "description": "Method to use for word tokenization"
        },
        "preserve_punctuation": {
            "type": "boolean",
            "default": True,
            "description": "Whether to preserve punctuation with adjacent words"
        },
        "min_chunk_words": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000,
            "default": 5,
            "description": "Minimum number of words per chunk (for last chunk)"
        }
    },
    default_parameters={
        "words_per_chunk": 100,
        "overlap_words": 0,
        "max_chunk_size": 5000,
        "word_tokenization": "simple",
        "preserve_punctuation": True,
        "min_chunk_words": 5
    },
    use_cases=["RAG", "document indexing", "consistent chunk sizing", "text analysis", "NLP preprocessing"],
    best_for=["consistent processing", "batch operations", "simple text processing", "word-level analysis"],
    limitations=["ignores sentence boundaries", "may split mid-sentence", "not suitable for complex text"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=False
)
class FixedLengthWordChunker(StreamableChunker):
    """
    Fixed-length word chunking strategy.

    Chunks text into segments with a specified number of words while maintaining
    word boundaries. This approach provides consistent chunk sizes suitable for
    processing pipelines that require uniform input sizes.

    Features:
    - Fixed number of words per chunk
    - Word boundary preservation
    - Multiple tokenization methods
    - Optional overlap between chunks
    - Size constraints to prevent overly large chunks
    - Streaming support for large documents

    Examples:
        Basic usage:
        ```python
        chunker = FixedLengthWordChunker(words_per_chunk=50)
        result = chunker.chunk("This is a sample text with many words...")
        ```

        With overlap:
        ```python
        chunker = FixedLengthWordChunker(words_per_chunk=100, overlap_words=10)
        result = chunker.chunk("Long document with many words...")
        ```

        Custom tokenization:
        ```python
        chunker = FixedLengthWordChunker(
            words_per_chunk=75,
            word_tokenization="regex",
            preserve_punctuation=False
        )
        result = chunker.chunk("Complex text with various punctuation marks!")
        ```
    """

    def __init__(
        self,
        words_per_chunk: int = 100,
        overlap_words: int = 0,
        max_chunk_size: int = 5000,
        word_tokenization: str = "simple",
        preserve_punctuation: bool = True,
        min_chunk_words: int = 5,
        **kwargs
    ):
        """
        Initialize the fixed-length word chunker.

        Args:
            words_per_chunk: Number of words per chunk
            overlap_words: Number of words to overlap between chunks
            max_chunk_size: Maximum chunk size in characters (safety limit)
            word_tokenization: Method for word tokenization ("simple", "whitespace", "regex")
            preserve_punctuation: Whether to preserve punctuation with adjacent words
            min_chunk_words: Minimum words per chunk (for last chunk)
            **kwargs: Additional parameters
        """
        super().__init__(
            name="fixed_length_word",
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        self.words_per_chunk = words_per_chunk
        self.overlap_words = overlap_words
        self.max_chunk_size = max_chunk_size
        self.word_tokenization = word_tokenization
        self.preserve_punctuation = preserve_punctuation
        self.min_chunk_words = min_chunk_words

        self.logger = logging.getLogger(f"{__name__}.FixedLengthWordChunker")

        # Validate parameters
        if self.words_per_chunk <= 0:
            raise ValueError("words_per_chunk must be positive")
        if self.overlap_words < 0:
            raise ValueError("overlap_words must be non-negative")
        if self.overlap_words >= self.words_per_chunk:
            raise ValueError("overlap_words must be less than words_per_chunk")

        # Setup tokenization pattern
        self._setup_tokenization_pattern()

    def _setup_tokenization_pattern(self) -> None:
        """Setup the tokenization pattern based on the selected method."""
        if self.word_tokenization == "whitespace":
            # Simple whitespace splitting
            self.tokenization_pattern = None
        elif self.word_tokenization == "regex":
            # Advanced regex pattern for word boundaries
            if self.preserve_punctuation:
                # Keep punctuation with words
                self.tokenization_pattern = re.compile(r'\S+')
            else:
                # Split on word boundaries, exclude punctuation
                self.tokenization_pattern = re.compile(r'\b\w+\b')
        else:  # simple
            # Default simple tokenization with punctuation handling
            if self.preserve_punctuation:
                self.tokenization_pattern = re.compile(r'\S+')
            else:
                self.tokenization_pattern = re.compile(r'\b\w+\b')

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk the content into fixed-length word chunks.

        Args:
            content: Text content to chunk
            source_info: Source information metadata
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with word-based chunks
        """
        start_time = time.time()

        # Handle different input types - only accept strings
        if isinstance(content, bytes):
            raise ValueError(f"FixedLengthWordChunker only supports string content, got {type(content)}")
        elif not isinstance(content, str):
            # Reject Path objects and other non-string types as invalid input
            from pathlib import Path
            if isinstance(content, Path):
                raise ValueError(f"FixedLengthWordChunker only supports string content, got {type(content)}")
            # Convert other types to string as fallback
            content = str(content)

        if not isinstance(content, str):
            content = str(content)

        # Basic validation
        if not content or not content.strip():
            return ChunkingResult(
                chunks=[],
                processing_time=0.0,
                source_info=source_info or {},
                strategy_used="fixed_length_word"
            )

        source_info = source_info or {"source": "string", "source_type": "content"}

        # Tokenize text into words
        words = self._tokenize_text(content)

        if not words:
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                source_info=source_info,
                strategy_used="fixed_length_word"
            )

        # Create chunks from words
        chunks = self._create_word_chunks(words, source_info)

        processing_time = time.time() - start_time

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            source_info={
                **source_info,
                "total_words": len(words),
                "word_tokenization_method": self.word_tokenization,
                "words_per_chunk": self.words_per_chunk,
                "overlap_words": self.overlap_words
            },
            strategy_used="fixed_length_word"
        )

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words based on the selected method.

        Args:
            text: Input text to tokenize

        Returns:
            List of word tokens
        """
        if self.word_tokenization == "whitespace":
            # Simple whitespace splitting
            return text.split()
        else:
            # Use regex pattern
            return self.tokenization_pattern.findall(text)

    def _create_word_chunks(self, words: List[str], source_info: Dict[str, Any]) -> List[Chunk]:
        """
        Create chunks from the list of words.

        Args:
            words: List of word tokens
            source_info: Source information

        Returns:
            List of chunks
        """
        chunks = []
        total_words = len(words)
        chunk_index = 0

        # Calculate step size (words_per_chunk - overlap_words)
        step_size = max(1, self.words_per_chunk - self.overlap_words)

        for start_idx in range(0, total_words, step_size):
            end_idx = min(start_idx + self.words_per_chunk, total_words)
            chunk_words = words[start_idx:end_idx]

            # Skip chunks that are too small (except the last chunk)
            # Only skip if we're not at the end AND the chunk is too small
            if len(chunk_words) < self.min_chunk_words and end_idx < total_words:
                continue

            # Join words back to text with proper spacing
            chunk_text = " ".join(chunk_words)

            # Check size constraint
            if len(chunk_text) > self.max_chunk_size:
                self.logger.warning(
                    f"Chunk {chunk_index} exceeds max_chunk_size ({len(chunk_text)} > {self.max_chunk_size}). "
                    f"Consider reducing words_per_chunk."
                )

            # Create chunk metadata
            metadata = ChunkMetadata(
                source=source_info.get("source", "unknown"),
                source_type=source_info.get("source_type", "content"),
                position=f"words {start_idx}-{end_idx - 1}",
                length=len(chunk_text),
                extra={
                    "word_count": len(chunk_words),
                    "start_word_index": start_idx,
                    "end_word_index": end_idx - 1,
                    "chunk_index": chunk_index,
                    "chunking_strategy": "fixed_length_word"
                }
            )

            # Create chunk
            chunk = Chunk(
                id=f"fixed_word_{chunk_index}",
                content=chunk_text,
                metadata=metadata,
                modality=ModalityType.TEXT
            )

            chunks.append(chunk)
            chunk_index += 1

            # Break if we've processed all words
            if end_idx >= total_words:
                break

        return chunks

    # Streaming support methods
    def can_stream(self) -> bool:
        """Check if this chunker supports streaming."""
        return True

    def chunk_stream(
        self,
        content_stream: Iterator[str],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream chunks from content stream.

        Args:
            content_stream: Iterator of content strings
            source_info: Source information
            **kwargs: Additional parameters

        Yields:
            Individual chunks as they become available
        """
        words_buffer = []
        chunk_index = 0
        source_info = source_info or {"source": "stream", "source_type": "stream"}

        for content_piece in content_stream:
            if isinstance(content_piece, str):
                # Tokenize the new content piece
                piece_words = self._tokenize_text(content_piece)
                words_buffer.extend(piece_words)

                # Create chunks from buffer when we have enough words
                while len(words_buffer) >= self.words_per_chunk:
                    chunk_words = words_buffer[:self.words_per_chunk]
                    chunk_text = " ".join(chunk_words)

                    # Create chunk metadata
                    metadata = ChunkMetadata(
                        source=source_info.get("source", "stream"),
                        source_type=source_info.get("source_type", "stream"),
                        position=f"stream chunk {chunk_index}",
                        length=len(chunk_text),
                        extra={
                            "word_count": len(chunk_words),
                            "chunk_index": chunk_index,
                            "chunking_strategy": "fixed_length_word",
                            "streaming": True
                        }
                    )

                    # Create and yield chunk
                    chunk = Chunk(
                        id=f"fixed_word_stream_{chunk_index}",
                        content=chunk_text,
                        metadata=metadata,
                        modality=ModalityType.TEXT
                    )

                    yield chunk
                    chunk_index += 1

                    # Remove processed words from buffer, keeping overlap
                    step_size = max(1, self.words_per_chunk - self.overlap_words)
                    words_buffer = words_buffer[step_size:]

        # Process remaining words in buffer
        if len(words_buffer) >= self.min_chunk_words:
            chunk_text = " ".join(words_buffer)

            metadata = ChunkMetadata(
                source=source_info.get("source", "stream"),
                source_type=source_info.get("source_type", "stream"),
                position=f"stream chunk {chunk_index} (final)",
                length=len(chunk_text),
                extra={
                    "word_count": len(words_buffer),
                    "chunk_index": chunk_index,
                    "chunking_strategy": "fixed_length_word",
                    "streaming": True,
                    "final_chunk": True
                }
            )

            chunk = Chunk(
                id=f"fixed_word_stream_{chunk_index}",
                content=chunk_text,
                metadata=metadata,
                modality=ModalityType.TEXT
            )

            yield chunk

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Adapt chunking parameters based on feedback.

        Args:
            feedback_score: Feedback score (0.0 to 1.0)
            feedback_type: Type of feedback ("quality", "performance", etc.)
            **kwargs: Additional feedback parameters
        """
        adaptation_record = {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "old_config": {
                "words_per_chunk": self.words_per_chunk,
                "overlap_words": self.overlap_words
            }
        }

        # Apply adaptations based on feedback score
        if feedback_score < 0.5:  # Poor performance
            if feedback_type == "quality":
                # Reduce chunk size for better granularity
                self.words_per_chunk = max(5, int(self.words_per_chunk * 0.8))
                self.overlap_words = min(
                    int(self.overlap_words * 1.2),
                    self.words_per_chunk - 1
                )
            elif feedback_type == "performance":
                # Increase chunk size for better performance
                self.words_per_chunk = min(5000, int(self.words_per_chunk * 1.2))
                self.overlap_words = max(0, int(self.overlap_words * 0.8))

        elif feedback_score > 0.8:  # Good performance
            if feedback_type == "quality":
                # Slightly increase chunk size
                self.words_per_chunk = min(1000, int(self.words_per_chunk * 1.1))

        # Handle specific feedback
        if kwargs.get("chunks_too_large"):
            self.words_per_chunk = max(5, int(self.words_per_chunk * 0.7))
        elif kwargs.get("chunks_too_small"):
            self.words_per_chunk = min(5000, int(self.words_per_chunk * 1.3))

        # Ensure overlap doesn't exceed chunk size
        self.overlap_words = min(self.overlap_words, self.words_per_chunk - 1)

        # Record new config
        adaptation_record["new_config"] = {
            "words_per_chunk": self.words_per_chunk,
            "overlap_words": self.overlap_words
        }

        # Store adaptation history
        if not hasattr(self, '_adaptation_history'):
            self._adaptation_history = []
        self._adaptation_history.append(adaptation_record)

        self.logger.info(f"Adapted parameters based on {feedback_type} feedback (score: {feedback_score})")

        # Return changes made
        changes = {}
        if adaptation_record["old_config"] != adaptation_record["new_config"]:
            changes = {
                "words_per_chunk": {
                    "old": adaptation_record["old_config"]["words_per_chunk"],
                    "new": adaptation_record["new_config"]["words_per_chunk"]
                },
                "overlap_words": {
                    "old": adaptation_record["old_config"]["overlap_words"],
                    "new": adaptation_record["new_config"]["overlap_words"]
                }
            }
        return changes

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of adaptations made."""
        if not hasattr(self, '_adaptation_history'):
            self._adaptation_history = []
        return self._adaptation_history.copy()

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "name": "fixed_length_word",
            "words_per_chunk": self.words_per_chunk,
            "overlap_words": self.overlap_words,
            "max_chunk_size": self.max_chunk_size,
            "word_tokenization": self.word_tokenization,
            "preserve_punctuation": self.preserve_punctuation,
            "min_chunk_words": self.min_chunk_words,
            "tokenization_pattern": str(self.tokenization_pattern) if self.tokenization_pattern else None
        }
