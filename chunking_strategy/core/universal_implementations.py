"""
Universal strategy implementations.

This module contains the concrete implementations of universal chunking strategies
that can work with any file type after content extraction.
"""

import re
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from chunking_strategy.core.base import (
    Chunk, ChunkingResult, ChunkMetadata, ModalityType
)
from chunking_strategy.core.universal_framework import UniversalStrategy
from chunking_strategy.core.extractors import ExtractedContent


class UniversalFixedSizeStrategy(UniversalStrategy):
    """Universal fixed-size chunking strategy."""

    def __init__(self):
        super().__init__(
            name="fixed_size",
            description="Split text into fixed-size chunks with overlap"
        )

    def apply(
        self,
        extracted_content: ExtractedContent,
        config: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """Apply fixed-size chunking to extracted content."""
        config = config or {}
        chunk_size = config.get("chunk_size", 1000)
        overlap_size = config.get("overlap_size", 200)

        chunks = []
        content = extracted_content.text_content

        if not content or not content.strip():
                    return ChunkingResult(
            chunks=[],
            strategy_used=self.name
        )

        # Split into fixed-size chunks
        start = 0
        chunk_id = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end]

            # Skip empty chunks
            if not chunk_text.strip():
                break

            chunk_id += 1
            chunks.append(
                Chunk(
                    id=f"chunk_{chunk_id}",
                    content=chunk_text,
                    modality=ModalityType.TEXT,
                    metadata=ChunkMetadata(
                        source=str(extracted_content.metadata.get('source', '')),
                        offset=start,
                        length=len(chunk_text),
                        chunker_used=self.name
                    )
                )
            )

            start = max(start + chunk_size - overlap_size, end)

        return ChunkingResult(
            chunks=chunks,
            strategy_used=self.name
        )


class UniversalSentenceStrategy(UniversalStrategy):
    """Universal sentence-based chunking strategy."""

    def __init__(self):
        super().__init__(
            name="sentence",
            description="Split text into sentence-based chunks"
        )

    def apply(
        self,
        extracted_content: ExtractedContent,
        config: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """Apply sentence-based chunking to extracted content."""
        config = config or {}
        max_chunk_size = config.get("max_chunk_size", 1000)
        max_sentences = config.get("max_sentences", None)

        chunks = []
        content = extracted_content.text_content

        if not content or not content.strip():
            return ChunkingResult(
                chunks=[],
                strategy_used=self.name
            )

        # Simple sentence splitting (can be enhanced with NLTK/spaCy)
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, content)

        current_chunk = ""
        current_sentence_count = 0
        chunk_id = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if we should create a new chunk
            should_create_new_chunk = False

            if max_sentences and current_sentence_count >= max_sentences:
                should_create_new_chunk = True
            elif current_chunk and len(current_chunk) + len(sentence) > max_chunk_size:
                should_create_new_chunk = True

            if should_create_new_chunk and current_chunk:
                # Save current chunk
                chunk_id += 1
                chunks.append(
                    Chunk(
                        id=f"sentence_chunk_{chunk_id}",
                        content=current_chunk.strip(),
                        modality=ModalityType.TEXT,
                        metadata=ChunkMetadata(
                            source=str(extracted_content.metadata.get('source', '')),
                            length=len(current_chunk),
                            chunker_used=self.name
                        )
                    )
                )
                current_chunk = sentence
                current_sentence_count = 1
            else:
                current_chunk += (" " if current_chunk else "") + sentence
                current_sentence_count += 1

        # Add final chunk
        if current_chunk.strip():
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=f"sentence_chunk_{chunk_id}",
                    content=current_chunk.strip(),
                    modality=ModalityType.TEXT,
                    metadata=ChunkMetadata(
                        source=str(extracted_content.metadata.get('source', '')),
                        length=len(current_chunk),
                        chunker_used=self.name
                    )
                )
            )

        return ChunkingResult(
            chunks=chunks,
            strategy_used=self.name
        )


class UniversalParagraphStrategy(UniversalStrategy):
    """Universal paragraph-based chunking strategy."""

    def __init__(self):
        super().__init__(
            name="paragraph",
            description="Split text into paragraph-based chunks"
        )

    def apply(
        self,
        extracted_content: ExtractedContent,
        config: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """Apply paragraph-based chunking to extracted content."""
        config = config or {}
        max_chunk_size = config.get("max_chunk_size", 2000)
        max_paragraphs = config.get("max_paragraphs", None)

        chunks = []
        content = extracted_content.text_content

        if not content or not content.strip():
            return ChunkingResult(
                chunks=[],
                strategy_used=self.name
            )

        # Split by double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', content)

        current_chunk = ""
        current_paragraph_count = 0
        chunk_id = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if we should create a new chunk
            should_create_new_chunk = False

            if max_paragraphs and current_paragraph_count >= max_paragraphs:
                should_create_new_chunk = True
            elif current_chunk and len(current_chunk) + len(paragraph) > max_chunk_size:
                should_create_new_chunk = True

            if should_create_new_chunk and current_chunk:
                # Save current chunk
                chunk_id += 1
                chunks.append(
                    Chunk(
                        id=f"para_chunk_{chunk_id}",
                        content=current_chunk.strip(),
                        modality=ModalityType.TEXT,
                        metadata=ChunkMetadata(
                            source=str(extracted_content.metadata.get('source', '')),
                            length=len(current_chunk),
                            chunker_used=self.name
                        )
                    )
                )
                current_chunk = paragraph
                current_paragraph_count = 1
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
                current_paragraph_count += 1

        # Add final chunk
        if current_chunk.strip():
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=f"para_chunk_{chunk_id}",
                    content=current_chunk.strip(),
                    modality=ModalityType.TEXT,
                    metadata=ChunkMetadata(
                        source=str(extracted_content.metadata.get('source', '')),
                        length=len(current_chunk),
                        chunker_used=self.name
                    )
                )
            )

        return ChunkingResult(
            chunks=chunks,
            strategy_used=self.name
        )


class UniversalOverlappingWindowStrategy(UniversalStrategy):
    """Universal overlapping window chunking strategy."""

    def __init__(self):
        super().__init__(
            name="overlapping_window",
            description="Split text using overlapping sliding windows"
        )

    def apply(
        self,
        extracted_content: ExtractedContent,
        config: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """Apply overlapping window chunking to extracted content."""
        config = config or {}
        window_size = config.get("window_size", 1000)
        overlap_size = config.get("overlap_size", 200)
        step_unit = config.get("step_unit", "char")

        # Calculate step size from overlap
        step_size = config.get("step_size", window_size - overlap_size)

        chunks = []
        content = extracted_content.text_content

        if not content or not content.strip():
            return ChunkingResult(
                chunks=[],
                strategy_used=self.name
            )

        # Handle different units
        if step_unit == "word":
            words = content.split()
            if len(words) <= window_size:
                # Content is smaller than window size, return as single chunk
                chunks.append(
                    Chunk(
                        id=f"window_1",
                        content=content,
                        modality=ModalityType.TEXT,
                        metadata=ChunkMetadata(
                            source=str(extracted_content.metadata.get('source', '')),
                            offset=0,
                            length=len(content),
                            chunker_used=self.name
                        )
                    )
                )
            else:
                chunk_id = 0
                start = 0
                while start < len(words):
                    end = min(start + window_size, len(words))
                    window_words = words[start:end]
                    window_text = " ".join(window_words)

                    chunk_id += 1
                    chunks.append(
                        Chunk(
                            id=f"window_{chunk_id}",
                            content=window_text,
                            modality=ModalityType.TEXT,
                            metadata=ChunkMetadata(
                                source=str(extracted_content.metadata.get('source', '')),
                                offset=start,
                                length=len(window_text),
                                chunker_used=self.name
                            )
                        )
                    )

                    start += step_size
                    if start >= len(words):
                        break
        else:
            # Character-based (default)
            if len(content) <= window_size:
                # Content is smaller than window size, return as single chunk
                chunks.append(
                    Chunk(
                        id=f"window_1",
                        content=content,
                        modality=ModalityType.TEXT,
                        metadata=ChunkMetadata(
                            source=str(extracted_content.metadata.get('source', '')),
                            offset=0,
                            length=len(content),
                            chunker_used=self.name
                        )
                    )
                )
            else:
                chunk_id = 0
                start = 0
                while start < len(content):
                    end = min(start + window_size, len(content))
                    window_text = content[start:end]

                    if not window_text.strip():
                        break

                    chunk_id += 1
                    chunks.append(
                        Chunk(
                            id=f"window_{chunk_id}",
                            content=window_text,
                            modality=ModalityType.TEXT,
                            metadata=ChunkMetadata(
                                source=str(extracted_content.metadata.get('source', '')),
                                offset=start,
                                length=len(window_text),
                                chunker_used=self.name
                            )
                        )
                    )

                    start += step_size
                    if start >= len(content):
                        break

        return ChunkingResult(
            chunks=chunks,
            strategy_used=self.name
        )


class UniversalRollingHashStrategy(UniversalStrategy):
    """Universal rolling hash-based chunking strategy."""

    def __init__(self):
        super().__init__(
            name="rolling_hash",
            description="Content-defined chunking using rolling hash"
        )

    def apply(
        self,
        extracted_content: ExtractedContent,
        config: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """Apply rolling hash chunking to extracted content."""
        config = config or {}
        target_size = config.get("target_chunk_size", config.get("target_size", 1000))
        min_size = config.get("min_chunk_size", config.get("min_size", 500))
        max_size = config.get("max_chunk_size", config.get("max_size", 2000))

        chunks = []
        content = extracted_content.text_content.encode('utf-8', errors='ignore')

        if not content:
                    return ChunkingResult(
            chunks=[],
            strategy_used=self.name
        )

        start = 0
        chunk_id = 0

        while start < len(content):
            # Find next chunk boundary using rolling hash
            end = self._find_chunk_boundary(
                content, start, target_size, min_size, max_size
            )

            chunk_bytes = content[start:end]
            chunk_text = chunk_bytes.decode('utf-8', errors='ignore')

            if chunk_text.strip():
                chunk_id += 1
                chunks.append(
                    Chunk(
                        id=f"hash_chunk_{chunk_id}",
                        content=chunk_text,
                        modality=ModalityType.TEXT,
                        metadata=ChunkMetadata(
                            source=str(extracted_content.metadata.get('source', '')),
                            offset=start,
                            length=len(chunk_text),
                            chunker_used=self.name
                        )
                    )
                )

            start = end

        return ChunkingResult(
            chunks=chunks,
            strategy_used=self.name
        )

    def _find_chunk_boundary(
        self,
        content: bytes,
        start: int,
        target_size: int,
        min_size: int,
        max_size: int
    ) -> int:
        """Find chunk boundary using rolling hash."""
        if start + min_size >= len(content):
            return len(content)

        # Simple rolling hash implementation
        window_size = min(32, (max_size - min_size) // 4)  # Smaller window for better boundaries
        target_hash = target_size % 256

        # Start looking for boundaries around target size
        ideal_pos = start + target_size
        search_start = max(start + min_size, ideal_pos - target_size // 2)
        search_end = min(start + max_size, len(content))

        # If we're close to target size, look for good boundaries
        if search_start < search_end:
            pos = search_start
            while pos + window_size <= search_end:
                # Calculate hash for current window
                window = content[pos:pos + window_size]
                hash_val = sum(window) % 256

                if hash_val == target_hash:
                    return pos

                pos += 1

        # If no good boundary found, return max allowed size but try to break at word boundary
        result_pos = min(start + max_size, len(content))

        # Try to find a space near the end to break cleanly
        search_back = min(100, result_pos - start - min_size)
        for i in range(search_back):
            pos = result_pos - i
            if pos > start and content[pos:pos+1] == b' ':
                return pos

        return result_pos
