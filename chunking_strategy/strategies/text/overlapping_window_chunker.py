"""
Overlapping window chunking strategy.

This module implements a sliding window approach to text chunking where consecutive
chunks have configurable overlap. This technique is particularly useful for:
- Maintaining context across chunk boundaries
- Information retrieval systems
- Embedding generation with context preservation
- Analysis requiring window-based processing

Key features:
- Configurable window size and step size
- Multiple granularities: characters, words, sentences
- Boundary-aware chunking (word/sentence boundaries)
- Adaptive overlap based on content structure
- Memory-efficient streaming support
- Integration with existing optimization infrastructure
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
from enum import Enum

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType,
    StreamableChunker
)
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.core.adaptive import AdaptableChunker

logger = logging.getLogger(__name__)


class WindowUnit(str, Enum):
    """Supported window units for overlapping chunking."""
    CHARACTERS = "characters"
    WORDS = "words"
    SENTENCES = "sentences"


@register_chunker(
    name="overlapping_window",
    category="text",
    description="Sliding window chunking with configurable overlap for context preservation",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "html", "json", "rtf"],
    complexity=ComplexityLevel.MEDIUM,
    dependencies=[],
    optional_dependencies=["spacy", "nltk"],
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.MEDIUM,
    quality=0.8,  # High quality due to context preservation
    parameters_schema={
        "window_size": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10000,
            "default": 500,
            "description": "Size of each window in specified units"
        },
        "step_size": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5000,
            "default": 250,
            "description": "Step size between windows (controls overlap)"
        },
        "window_unit": {
            "type": "string",
            "enum": ["characters", "words", "sentences"],
            "default": "words",
            "description": "Unit for window and step sizes"
        },
        "preserve_boundaries": {
            "type": "boolean",
            "default": True,
            "description": "Respect word/sentence boundaries when chunking"
        },
        "min_window_size": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000,
            "default": 50,
            "description": "Minimum window size to avoid tiny chunks"
        },
        "max_chunk_chars": {
            "type": "integer",
            "minimum": 100,
            "maximum": 100000,
            "default": 8000,
            "description": "Maximum characters per chunk for safety"
        },
        "sentence_separators": {
            "type": "array",
            "items": {"type": "string"},
            "default": [".", "!", "?", "。", "！", "？"],
            "description": "Custom sentence separators"
        }
    }
)
class OverlappingWindowChunker(StreamableChunker, AdaptableChunker):
    """
    Overlapping window chunker using sliding window approach.

    Creates chunks by sliding a window across text with configurable overlap,
    ensuring context preservation across chunk boundaries.
    """

    def __init__(
        self,
        window_size: int = 500,
        step_size: int = 250,
        window_unit: str = "words",
        preserve_boundaries: bool = True,
        min_window_size: int = 50,
        max_chunk_chars: int = 8000,
        sentence_separators: Optional[List[str]] = None,
        **kwargs
    ):
        # Extract name from kwargs or use default
        name = kwargs.pop("name", "overlapping_window")
        super().__init__(
            name=name,
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        # Core parameters
        self.window_size = window_size
        self.step_size = step_size
        self.window_unit = WindowUnit(window_unit)
        self.preserve_boundaries = preserve_boundaries
        self.min_window_size = min_window_size
        self.max_chunk_chars = max_chunk_chars
        self.sentence_separators = sentence_separators or [".", "!", "?", "。", "！", "？"]

        # Adaptation tracking
        self._adaptation_history: List[Dict[str, Any]] = []

        # Validate parameters
        if self.step_size >= self.window_size:
            raise ValueError("step_size must be less than window_size to create overlap")
        if self.min_window_size > self.window_size:
            raise ValueError("min_window_size must be less than or equal to window_size")

        # Compile patterns for efficiency
        self._sentence_pattern = re.compile(
            r'([.!?。！？]+(?:\s*["\']?\s*|\s+))',
            re.MULTILINE
        )
        self._word_pattern = re.compile(r'\S+')

        logger.debug(f"Initialized OverlappingWindowChunker: {self.window_size} {self.window_unit.value} window, {self.step_size} step")

    def _split_into_units(self, text: str) -> List[str]:
        """Split text into specified units (characters, words, sentences)."""
        if self.window_unit == WindowUnit.CHARACTERS:
            return list(text)
        elif self.window_unit == WindowUnit.WORDS:
            return self._word_pattern.findall(text)
        elif self.window_unit == WindowUnit.SENTENCES:
            return self._split_into_sentences(text)
        else:
            raise ValueError(f"Unsupported window unit: {self.window_unit}")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using configured separators."""
        # Use regex to split while preserving separators
        parts = self._sentence_pattern.split(text)

        sentences = []
        current_sentence = ""

        for i, part in enumerate(parts):
            if self._sentence_pattern.match(part):
                # This is a separator, add it to current sentence and finalize
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                # Regular text part
                current_sentence += part

        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return [s for s in sentences if s.strip()]

    def _join_units(self, units: List[str]) -> str:
        """Join units back into text based on unit type."""
        if self.window_unit == WindowUnit.CHARACTERS:
            return ''.join(units)
        elif self.window_unit == WindowUnit.WORDS:
            return ' '.join(units)
        elif self.window_unit == WindowUnit.SENTENCES:
            return ' '.join(units)
        else:
            raise ValueError(f"Unsupported window unit: {self.window_unit}")

    def _find_boundary_position(self, units: List[str], target_pos: int, direction: str = "forward") -> int:
        """
        Find appropriate boundary position when preserve_boundaries is True.

        Args:
            units: List of text units
            target_pos: Target position to adjust
            direction: "forward" or "backward" to find boundary

        Returns:
            Adjusted position respecting boundaries
        """
        if not self.preserve_boundaries or target_pos <= 0 or target_pos >= len(units):
            return min(max(target_pos, 0), len(units))

        # For character-level, find word boundaries
        if self.window_unit == WindowUnit.CHARACTERS:
            if direction == "forward":
                # Find next word boundary (space or punctuation)
                for i in range(target_pos, len(units)):
                    if units[i] in ' \t\n\r':
                        return i
                return len(units)
            else:  # backward
                # Find previous word boundary
                for i in range(target_pos - 1, -1, -1):
                    if units[i] in ' \t\n\r':
                        return i + 1
                return 0

        # For words and sentences, positions are already at boundaries
        return min(max(target_pos, 0), len(units))

    def chunk(self, content: Union[str, bytes, Path], source_info: Optional[Dict[str, Any]] = None, **kwargs) -> ChunkingResult:
        """
        Chunk text using overlapping sliding window approach.

        Args:
            content: Input content to chunk (text, bytes, or file path)
            source_info: Information about the source
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with overlapping chunks
        """
        start_time = time.time()

        # Handle different input types
        if isinstance(content, (bytes, Path)):
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            elif isinstance(content, Path):
                content = content.read_text(encoding='utf-8')
        elif not isinstance(content, str):
            content = str(content)

        if not content.strip():
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used=self.name,
                source_info=source_info or {}
            )

        source_info = source_info or {"source": "string", "source_type": "content"}

        # Split text into units
        units = self._split_into_units(content)
        total_units = len(units)

        if total_units == 0:
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used=self.name,
                source_info=source_info
            )

        chunks = []
        current_pos = 0
        chunk_id = 0

        while current_pos < total_units:
            # Calculate window end position
            window_end = current_pos + self.window_size

            # Adjust for boundaries if needed
            if self.preserve_boundaries:
                window_end = self._find_boundary_position(units, window_end, "forward")
            else:
                window_end = min(window_end, total_units)

            # Ensure minimum window size
            if window_end - current_pos < self.min_window_size and current_pos > 0:
                break

            # Extract window content
            window_units = units[current_pos:window_end]
            window_content = self._join_units(window_units)

            # Check character limit
            if len(window_content) > self.max_chunk_chars:
                # Truncate to character limit while preserving boundaries
                truncated_content = window_content[:self.max_chunk_chars]
                if self.preserve_boundaries and self.window_unit != WindowUnit.CHARACTERS:
                    # Find last complete unit boundary
                    last_space = truncated_content.rfind(' ')
                    if last_space > 0:
                        truncated_content = truncated_content[:last_space]
                window_content = truncated_content

            # Calculate character positions in original text
            if chunk_id == 0:
                start_char = 0
            else:
                # Find approximate start position in original text
                start_char = len(self._join_units(units[:current_pos]))

            end_char = start_char + len(window_content)

            # Create chunk
            chunk = Chunk(
                id=f"{self.name}_chunk_{chunk_id}",
                content=window_content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=source_info.get("source", "unknown"),
                    source_type=source_info.get("source_type", "content"),
                    position=f"window {chunk_id}",
                    offset=start_char,
                    length=len(window_content),
                    extra={
                        "chunk_index": chunk_id,
                        "unit_count": len(window_units),
                        "overlap_with_previous": current_pos > 0,
                        "chunker_used": self.name,
                        "window_info": {
                            "window_start": current_pos,
                            "window_end": window_end,
                            "unit_type": self.window_unit.value,
                            "step_size": self.step_size
                        },
                        "chunking_strategy": "overlapping_window"
                    }
                )
            )
            chunks.append(chunk)
            chunk_id += 1

            # Move to next position
            next_pos = current_pos + self.step_size

            # Adjust for boundaries if needed
            if self.preserve_boundaries:
                next_pos = self._find_boundary_position(units, next_pos, "forward")

            # Ensure we make progress
            if next_pos <= current_pos:
                next_pos = current_pos + 1

            current_pos = next_pos

            # Safety check to prevent infinite loops
            if current_pos >= total_units:
                break

        # Calculate overlap statistics
        overlap_ratio = 0.0
        if len(chunks) > 1:
            total_overlap_units = (len(chunks) - 1) * (self.window_size - self.step_size)
            overlap_ratio = total_overlap_units / total_units if total_units > 0 else 0.0

        processing_time = time.time() - start_time

        result = ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used=self.name,
            source_info={
                **source_info,
                "total_units": total_units,
                "total_chars": len(content),
                "overlap_ratio": round(overlap_ratio, 3),
                "avg_units_per_chunk": round(sum(chunk.metadata.extra["unit_count"] for chunk in chunks) / len(chunks), 1) if chunks else 0,
                "window_size": self.window_size,
                "step_size": self.step_size,
                "window_unit": self.window_unit.value,
                "preserve_boundaries": self.preserve_boundaries,
                "min_window_size": self.min_window_size,
                "max_chunk_chars": self.max_chunk_chars
            }
        )

        logger.info(f"Created {len(chunks)} overlapping chunks in {processing_time:.3f}s, {overlap_ratio:.1%} overlap")
        return result

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Chunk content from a stream using overlapping windows.

        Args:
            content_stream: Iterator yielding content pieces
            source_info: Information about the source
            **kwargs: Additional parameters

        Yields:
            Individual chunks as they're created
        """
        # Collect all content from stream
        collected_content = ""
        for content_piece in content_stream:
            if isinstance(content_piece, bytes):
                content_piece = content_piece.decode('utf-8', errors='ignore')
            collected_content += str(content_piece)

        # Use the streaming chunk generator with proper source_info
        yield from self.stream_chunk(collected_content, source_info=source_info, **kwargs)

    def stream_chunk(self, text: Union[str, Path], source_info: Optional[Dict[str, Any]] = None, **kwargs) -> Iterator[Chunk]:
        """
        Stream overlapping chunks one at a time for memory efficiency.

        Args:
            text: Input text or path to file
            **kwargs: Additional parameters

        Yields:
            Individual chunks as they're created
        """
        logger.debug("Starting streaming overlapping window chunking")

        # Handle file input
        if isinstance(text, Path):
            text = text.read_text(encoding='utf-8')
        elif not isinstance(text, str):
            text = str(text)

        source_info = source_info or {"source": "stream", "source_type": "stream"}

        if not text.strip():
            return

        # Split text into units
        units = self._split_into_units(text)
        total_units = len(units)

        if total_units == 0:
            return

        current_pos = 0
        chunk_id = 0

        while current_pos < total_units:
            # Calculate window end position
            window_end = current_pos + self.window_size

            # Adjust for boundaries if needed
            if self.preserve_boundaries:
                window_end = self._find_boundary_position(units, window_end, "forward")
            else:
                window_end = min(window_end, total_units)

            # Ensure minimum window size
            if window_end - current_pos < self.min_window_size and current_pos > 0:
                break

            # Extract window content
            window_units = units[current_pos:window_end]
            window_content = self._join_units(window_units)

            # Check character limit
            if len(window_content) > self.max_chunk_chars:
                truncated_content = window_content[:self.max_chunk_chars]
                if self.preserve_boundaries and self.window_unit != WindowUnit.CHARACTERS:
                    last_space = truncated_content.rfind(' ')
                    if last_space > 0:
                        truncated_content = truncated_content[:last_space]
                window_content = truncated_content

            # Calculate character positions
            if chunk_id == 0:
                start_char = 0
            else:
                start_char = len(self._join_units(units[:current_pos]))

            end_char = start_char + len(window_content)

            # Create and yield chunk
            chunk = Chunk(
                id=f"{self.name}_chunk_{chunk_id}",
                content=window_content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=source_info.get("source", "stream"),
                    source_type=source_info.get("source_type", "stream"),
                    position=f"window {chunk_id}",
                    offset=start_char,
                    length=len(window_content),
                    extra={
                        "chunk_index": chunk_id,
                        "unit_count": len(window_units),
                        "overlap_with_previous": current_pos > 0,
                        "chunker_used": self.name,
                        "window_info": {
                            "window_start": current_pos,
                            "window_end": window_end,
                            "unit_type": self.window_unit.value,
                            "step_size": self.step_size
                        },
                        "chunking_strategy": "overlapping_window",
                        "streaming": True
                    }
                )
            )

            yield chunk
            chunk_id += 1

            # Move to next position
            next_pos = current_pos + self.step_size

            if self.preserve_boundaries:
                next_pos = self._find_boundary_position(units, next_pos, "forward")

            if next_pos <= current_pos:
                next_pos = current_pos + 1

            current_pos = next_pos

            if current_pos >= total_units:
                break

    def adapt_parameters(self, feedback_score: float, feedback_type: str = "quality", **kwargs) -> Dict[str, Any]:
        """
        Adapt chunking parameters based on feedback.

        Args:
            feedback_score: Score from 0-1 (higher is better)
            feedback_type: Type of feedback ("quality", "performance", "size")
            **kwargs: Additional feedback context

        Returns:
            Dictionary of parameter changes made
        """
        original_window = self.window_size
        original_step = self.step_size
        changes = {}

        if feedback_type == "quality":
            if feedback_score < 0.6:
                # Increase overlap for better context preservation
                new_step = max(self.min_window_size, int(self.step_size * 0.8))
                if new_step != self.step_size:
                    self.step_size = new_step
                    changes["step_size"] = {"old": original_step, "new": new_step, "reason": "increased overlap for quality"}

            elif feedback_score > 0.8:
                # Can reduce overlap for efficiency
                new_step = min(self.window_size - self.min_window_size, int(self.step_size * 1.2))
                if new_step != self.step_size:
                    self.step_size = new_step
                    changes["step_size"] = {"old": original_step, "new": new_step, "reason": "reduced overlap for efficiency"}

        elif feedback_type == "performance":
            if feedback_score < 0.5:
                # Reduce window size for faster processing
                new_window = max(self.min_window_size * 2, int(self.window_size * 0.8))
                if new_window != self.window_size:
                    self.window_size = new_window
                    # Adjust step size proportionally
                    self.step_size = min(self.step_size, new_window - self.min_window_size)
                    changes["window_size"] = {"old": original_window, "new": new_window, "reason": "reduced for performance"}

        elif feedback_type == "size":
            if feedback_score < 0.5:
                # Chunks too large/small, adjust window size
                target_change = kwargs.get("target_size_change", 1.0)
                new_window = max(self.min_window_size * 2, int(self.window_size * target_change))
                if new_window != self.window_size:
                    self.window_size = new_window
                    self.step_size = min(self.step_size, new_window - self.min_window_size)
                    changes["window_size"] = {"old": original_window, "new": new_window, "reason": "size adjustment"}

        # Record adaptation
        if changes:
            adaptation_record = {
                "timestamp": time.time(),
                "feedback_score": feedback_score,
                "feedback_type": feedback_type,
                "changes": changes,
                "parameters_after": {
                    "window_size": self.window_size,
                    "step_size": self.step_size,
                    "overlap_ratio": 1.0 - (self.step_size / self.window_size)
                }
            }
            self._adaptation_history.append(adaptation_record)

            logger.info(f"Adapted parameters based on {feedback_type} feedback ({feedback_score:.2f}): {changes}")

        return changes

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self._adaptation_history.copy()

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "name": "overlapping_window",
            "window_size": self.window_size,
            "step_size": self.step_size,
            "window_unit": self.window_unit.value,
            "preserve_boundaries": self.preserve_boundaries,
            "min_window_size": self.min_window_size,
            "max_chunk_chars": self.max_chunk_chars,
            "sentence_separators": self.sentence_separators,
            "overlap_ratio": 1.0 - (self.step_size / self.window_size) if self.window_size > 0 else 0.0
        }

    def get_chunk_size_estimate(self, text_length: int) -> int:
        """
        Estimate number of chunks that will be created.

        Args:
            text_length: Length of input text

        Returns:
            Estimated number of chunks
        """
        if self.window_unit == WindowUnit.CHARACTERS:
            estimated_units = text_length
        elif self.window_unit == WindowUnit.WORDS:
            # Rough estimate: 5 chars per word on average
            estimated_units = text_length // 5
        else:  # sentences
            # Rough estimate: 20 words per sentence, 5 chars per word
            estimated_units = text_length // 100

        if estimated_units <= self.window_size:
            return 1

        # Calculate overlapping windows
        return max(1, ((estimated_units - self.window_size) // self.step_size) + 1)
