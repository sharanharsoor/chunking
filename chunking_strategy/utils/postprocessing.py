"""
Postprocessing utilities for chunk refinement after chunking.

This module provides postprocessing capabilities to refine, filter, merge,
and enhance chunks after the initial chunking process.
"""

import logging
import statistics
from typing import Any, Dict, List, Optional

from chunking_strategy.core.base import Chunk, ChunkingResult, ModalityType

logger = logging.getLogger(__name__)


class PostprocessingPipeline:
    """
    Pipeline for postprocessing chunks after chunking.

    Provides various postprocessing operations to improve chunk quality,
    consistency, and usefulness.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize postprocessing pipeline.

        Args:
            config: Configuration for postprocessing steps
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.PostprocessingPipeline")

    def process(
        self,
        result: ChunkingResult,
        config: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Process chunking result through postprocessing pipeline.

        Args:
            result: Chunking result to postprocess
            config: Override configuration for this processing

        Returns:
            Postprocessed chunking result
        """
        active_config = {**self.config, **(config or {})}
        processed_chunks = result.chunks.copy()

        # Remove empty chunks
        if active_config.get("remove_empty_chunks", True):
            processed_chunks = self._remove_empty_chunks(processed_chunks)

        # Merge short chunks
        if active_config.get("merge_short_chunks", False):
            min_size = active_config.get("min_chunk_size", 100)
            max_size = active_config.get("max_merge_size", 2000)
            processed_chunks = self._merge_short_chunks(processed_chunks, min_size, max_size)

        # Filter by size
        if active_config.get("filter_by_size", False):
            min_size = active_config.get("absolute_min_size", 10)
            max_size = active_config.get("absolute_max_size", 10000)
            processed_chunks = self._filter_by_size(processed_chunks, min_size, max_size)

        # Deduplicate chunks
        if active_config.get("deduplicate", False):
            similarity_threshold = active_config.get("similarity_threshold", 0.9)
            processed_chunks = self._deduplicate_chunks(processed_chunks, similarity_threshold)

        # Add metadata enhancements
        if active_config.get("enhance_metadata", True):
            processed_chunks = self._enhance_metadata(processed_chunks)

        # Normalize content
        if active_config.get("normalize_content", False):
            processed_chunks = self._normalize_content(processed_chunks)

        # Custom postprocessors
        if "custom_processors" in active_config:
            for processor in active_config["custom_processors"]:
                processed_chunks = processor(processed_chunks)

        # Create new result
        new_result = ChunkingResult(
            chunks=processed_chunks,
            processing_time=result.processing_time,
            strategy_used=result.strategy_used,
            fallback_strategies=result.fallback_strategies,
            source_info=result.source_info,
            errors=result.errors or [],
            warnings=result.warnings or []
        )

        # Add postprocessing metadata
        if new_result.source_info is None:
            new_result.source_info = {}
        new_result.source_info["postprocessed"] = True
        new_result.source_info["original_chunk_count"] = len(result.chunks)
        new_result.source_info["final_chunk_count"] = len(processed_chunks)

        self.logger.info(
            f"Postprocessing completed: {len(result.chunks)} -> {len(processed_chunks)} chunks"
        )

        return new_result

    def _remove_empty_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remove empty or very small chunks."""
        filtered = []
        removed_count = 0

        for chunk in chunks:
            content = chunk.content
            if isinstance(content, str):
                if len(content.strip()) > 0:
                    filtered.append(chunk)
                else:
                    removed_count += 1
            elif isinstance(content, bytes):
                if len(content) > 0:
                    filtered.append(chunk)
                else:
                    removed_count += 1
            else:
                # Keep non-string/bytes content
                filtered.append(chunk)

        if removed_count > 0:
            self.logger.debug(f"Removed {removed_count} empty chunks")

        return filtered

    def _merge_short_chunks(
        self,
        chunks: List[Chunk],
        min_size: int,
        max_merge_size: int
    ) -> List[Chunk]:
        """Merge consecutive short chunks."""
        if not chunks:
            return chunks

        merged = []
        current_chunk = chunks[0]
        merged_count = 0

        for next_chunk in chunks[1:]:
            current_size = len(str(current_chunk.content))
            next_size = len(str(next_chunk.content))

            # Check if we should merge
            should_merge = (
                (current_size < min_size or next_size < min_size) and
                (current_size + next_size <= max_merge_size) and
                current_chunk.modality == next_chunk.modality == ModalityType.TEXT
            )

            if should_merge:
                # Merge chunks
                current_chunk = self._merge_two_chunks(current_chunk, next_chunk)
                merged_count += 1
            else:
                # Add current chunk and start new one
                merged.append(current_chunk)
                current_chunk = next_chunk

        # Add final chunk
        merged.append(current_chunk)

        if merged_count > 0:
            self.logger.debug(f"Merged {merged_count} short chunks")

        return merged

    def _merge_two_chunks(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        """Merge two chunks into one."""
        # Combine content
        if isinstance(chunk1.content, str) and isinstance(chunk2.content, str):
            merged_content = chunk1.content + " " + chunk2.content
        else:
            merged_content = str(chunk1.content) + " " + str(chunk2.content)

        # Create new metadata
        merged_metadata = chunk1.metadata
        merged_metadata.extra["merged_from"] = [chunk1.id, chunk2.id]
        merged_metadata.length = len(merged_content)

        # Create merged chunk
        return Chunk(
            id=f"merged_{chunk1.id}_{chunk2.id}",
            content=merged_content,
            modality=chunk1.modality,
            metadata=merged_metadata,
            parent_id=chunk1.parent_id,
            children_ids=(chunk1.children_ids or []) + (chunk2.children_ids or [])
        )

    def _filter_by_size(
        self,
        chunks: List[Chunk],
        min_size: int,
        max_size: int
    ) -> List[Chunk]:
        """Filter chunks by absolute size limits."""
        filtered = []
        removed_count = 0

        for chunk in chunks:
            size = len(str(chunk.content))
            if min_size <= size <= max_size:
                filtered.append(chunk)
            else:
                removed_count += 1

        if removed_count > 0:
            self.logger.debug(f"Filtered out {removed_count} chunks by size")

        return filtered

    def _deduplicate_chunks(
        self,
        chunks: List[Chunk],
        similarity_threshold: float
    ) -> List[Chunk]:
        """Remove duplicate or highly similar chunks."""
        if not chunks:
            return chunks

        unique_chunks = []
        removed_count = 0

        for chunk in chunks:
            is_duplicate = False
            chunk_content = str(chunk.content).lower().strip()

            for existing_chunk in unique_chunks:
                existing_content = str(existing_chunk.content).lower().strip()

                # Simple similarity check
                similarity = self._calculate_similarity(chunk_content, existing_content)
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_chunks.append(chunk)
            else:
                removed_count += 1

        if removed_count > 0:
            self.logger.debug(f"Removed {removed_count} duplicate chunks")

        return unique_chunks

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts."""
        if not text1 or not text2:
            return 0.0

        # Simple character-based similarity
        if text1 == text2:
            return 1.0

        # Jaccard similarity on words
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _enhance_metadata(self, chunks: List[Chunk]) -> List[Chunk]:
        """Enhance chunk metadata with computed statistics."""
        for i, chunk in enumerate(chunks):
            content = str(chunk.content)

            # Add position information
            chunk.metadata.extra["chunk_index"] = i
            chunk.metadata.extra["total_chunks"] = len(chunks)

            # Add content statistics
            if chunk.modality == ModalityType.TEXT:
                words = content.split()
                sentences = len([s for s in content.split('.') if s.strip()])

                chunk.metadata.extra.update({
                    "word_count": len(words),
                    "sentence_count": sentences,
                    "character_count": len(content),
                    "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
                })

            # Add quality indicators
            chunk.metadata.extra["has_meaningful_content"] = len(content.strip()) > 10
            chunk.metadata.extra["content_density"] = len(content.replace(' ', '')) / len(content) if content else 0

        return chunks

    def _normalize_content(self, chunks: List[Chunk]) -> List[Chunk]:
        """Normalize chunk content."""
        for chunk in chunks:
            if chunk.modality == ModalityType.TEXT and isinstance(chunk.content, str):
                # Remove extra whitespace
                normalized = ' '.join(chunk.content.split())

                # Ensure proper sentence spacing
                import re
                normalized = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', normalized)

                # Update chunk content
                chunk.content = normalized
                chunk.size = len(normalized)
                chunk.metadata.length = len(normalized)

        return chunks
