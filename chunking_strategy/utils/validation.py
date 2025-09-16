"""
Validation utilities for chunks and chunking results.

This module provides validation capabilities to check chunk quality,
consistency, and correctness.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import re

from chunking_strategy.core.base import Chunk, ChunkingResult, ModalityType

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ChunkValidator:
    """
    Validator for chunks and chunking results.

    Provides comprehensive validation checks to ensure chunk quality
    and consistency.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize chunk validator.

        Args:
            strict_mode: Whether to use strict validation rules
        """
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(f"{__name__}.ChunkValidator")

    def validate_chunk(self, chunk: Chunk) -> List[str]:
        """
        Validate a single chunk.

        Args:
            chunk: Chunk to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Basic structure validation
        if not chunk.id:
            issues.append("Chunk missing ID")
        elif not isinstance(chunk.id, str):
            issues.append("Chunk ID must be string")

        if chunk.content is None:
            issues.append("Chunk content is None")

        if not hasattr(chunk, 'modality') or chunk.modality is None:
            issues.append("Chunk missing modality")

        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
            issues.append("Chunk missing metadata")

        # Content validation
        if chunk.content is not None:
            content_issues = self._validate_content(chunk.content, chunk.modality)
            issues.extend(content_issues)

        # Size validation
        if chunk.size is not None:
            if chunk.size < 0:
                issues.append("Chunk size cannot be negative")

            # Check size consistency
            if chunk.content is not None:
                actual_size = len(chunk.content) if isinstance(chunk.content, (str, bytes)) else 0
                if chunk.size != actual_size:
                    issues.append(f"Chunk size mismatch: declared={chunk.size}, actual={actual_size}")

        # Metadata validation
        if chunk.metadata:
            metadata_issues = self._validate_metadata(chunk.metadata)
            issues.extend(metadata_issues)

        # ID uniqueness will be checked at result level

        return issues

    def validate_result(self, result: ChunkingResult) -> List[str]:
        """
        Validate a chunking result.

        Args:
            result: Chunking result to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Basic structure validation
        if not hasattr(result, 'chunks') or result.chunks is None:
            issues.append("Result missing chunks list")
            return issues

        if not isinstance(result.chunks, list):
            issues.append("Result chunks must be a list")
            return issues

        # Validate each chunk
        chunk_issues = []
        chunk_ids: Set[str] = set()

        for i, chunk in enumerate(result.chunks):
            if not isinstance(chunk, Chunk):
                chunk_issues.append(f"Chunk {i} is not a Chunk instance")
                continue

            # Validate individual chunk
            individual_issues = self.validate_chunk(chunk)
            for issue in individual_issues:
                chunk_issues.append(f"Chunk {i} ({chunk.id}): {issue}")

            # Check ID uniqueness
            if chunk.id in chunk_ids:
                chunk_issues.append(f"Duplicate chunk ID: {chunk.id}")
            else:
                chunk_ids.add(chunk.id)

        issues.extend(chunk_issues)

        # Validate result metadata
        if hasattr(result, 'total_chunks'):
            if result.total_chunks != len(result.chunks):
                issues.append(f"Total chunks mismatch: declared={result.total_chunks}, actual={len(result.chunks)}")

        # Validate processing time
        if hasattr(result, 'processing_time') and result.processing_time is not None:
            if result.processing_time < 0:
                issues.append("Processing time cannot be negative")

        # Validate strategy information
        if hasattr(result, 'strategy_used') and not result.strategy_used:
            if self.strict_mode:
                issues.append("Missing strategy_used information")

        # Check for empty result
        if not result.chunks and self.strict_mode:
            issues.append("Result contains no chunks")

        # Content coverage validation
        coverage_issues = self._validate_coverage(result)
        issues.extend(coverage_issues)

        # Ordering validation
        ordering_issues = self._validate_ordering(result)
        issues.extend(ordering_issues)

        return issues

    def validate_and_raise(self, obj: Any) -> None:
        """
        Validate and raise exception if invalid.

        Args:
            obj: Object to validate (Chunk or ChunkingResult)

        Raises:
            ValidationError: If validation fails
        """
        if isinstance(obj, Chunk):
            issues = self.validate_chunk(obj)
        elif isinstance(obj, ChunkingResult):
            issues = self.validate_result(obj)
        else:
            raise ValidationError(f"Cannot validate object of type {type(obj)}")

        if issues:
            raise ValidationError(f"Validation failed: {'; '.join(issues)}")

    def is_valid(self, obj: Any) -> bool:
        """
        Check if object is valid.

        Args:
            obj: Object to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            self.validate_and_raise(obj)
            return True
        except ValidationError:
            return False

    def _validate_content(self, content: Any, modality: ModalityType) -> List[str]:
        """Validate chunk content based on modality."""
        issues = []

        if modality == ModalityType.TEXT:
            if not isinstance(content, str):
                issues.append("Text modality content must be string")
            elif not content.strip():
                issues.append("Text content is empty or whitespace-only")
            elif len(content) > 1000000:  # 1MB text limit
                if self.strict_mode:
                    issues.append("Text content exceeds size limit (1MB)")

        elif modality == ModalityType.IMAGE:
            if not isinstance(content, (bytes, str)):
                issues.append("Image content must be bytes or string")

        elif modality == ModalityType.AUDIO:
            if not isinstance(content, (bytes, str)):
                issues.append("Audio content must be bytes or string")

        elif modality == ModalityType.VIDEO:
            if not isinstance(content, (bytes, str)):
                issues.append("Video content must be bytes or string")

        elif modality == ModalityType.MIXED:
            # Mixed content can be anything
            pass

        else:
            if self.strict_mode:
                issues.append(f"Unknown modality: {modality}")

        return issues

    def _validate_metadata(self, metadata: Any) -> List[str]:
        """Validate chunk metadata."""
        issues = []

        # Check required fields
        if not hasattr(metadata, 'source'):
            issues.append("Metadata missing source field")
        elif not metadata.source:
            if self.strict_mode:
                issues.append("Metadata source is empty")

        # Validate specific fields
        if hasattr(metadata, 'offset') and metadata.offset is not None:
            if metadata.offset < 0:
                issues.append("Metadata offset cannot be negative")

        if hasattr(metadata, 'length') and metadata.length is not None:
            if metadata.length < 0:
                issues.append("Metadata length cannot be negative")

        if hasattr(metadata, 'timestamp') and metadata.timestamp is not None:
            if isinstance(metadata.timestamp, tuple):
                if len(metadata.timestamp) != 2:
                    issues.append("Metadata timestamp tuple must have 2 elements")
                elif metadata.timestamp[0] > metadata.timestamp[1]:
                    issues.append("Metadata timestamp start > end")
                elif any(t < 0 for t in metadata.timestamp):
                    issues.append("Metadata timestamp values cannot be negative")

        if hasattr(metadata, 'bbox') and metadata.bbox is not None:
            if isinstance(metadata.bbox, tuple):
                if len(metadata.bbox) != 4:
                    issues.append("Metadata bbox must have 4 elements (x1, y1, x2, y2)")

        return issues

    def _validate_coverage(self, result: ChunkingResult) -> List[str]:
        """Validate content coverage in chunks."""
        issues = []

        if not result.chunks:
            return issues

        # Check for overlapping chunks (if position information available)
        positioned_chunks = []
        for chunk in result.chunks:
            if (hasattr(chunk.metadata, 'offset') and chunk.metadata.offset is not None and
                hasattr(chunk.metadata, 'length') and chunk.metadata.length is not None):
                start = chunk.metadata.offset
                end = start + chunk.metadata.length
                positioned_chunks.append((start, end, chunk.id))

        if len(positioned_chunks) > 1:
            # Sort by start position
            positioned_chunks.sort(key=lambda x: x[0])

            # Check for gaps or overlaps
            for i in range(len(positioned_chunks) - 1):
                current_end = positioned_chunks[i][1]
                next_start = positioned_chunks[i + 1][0]

                if current_end > next_start:
                    issues.append(
                        f"Overlapping chunks: {positioned_chunks[i][2]} and {positioned_chunks[i + 1][2]}"
                    )
                elif current_end < next_start and self.strict_mode:
                    gap_size = next_start - current_end
                    if gap_size > 100:  # Significant gap
                        issues.append(
                            f"Large gap ({gap_size} units) between chunks {positioned_chunks[i][2]} and {positioned_chunks[i + 1][2]}"
                        )

        return issues

    def _validate_ordering(self, result: ChunkingResult) -> List[str]:
        """Validate chunk ordering."""
        issues = []

        if not result.chunks:
            return issues

        # Check if chunks with position information are in order
        positioned_chunks = []
        for i, chunk in enumerate(result.chunks):
            if hasattr(chunk.metadata, 'offset') and chunk.metadata.offset is not None:
                positioned_chunks.append((chunk.metadata.offset, i, chunk.id))

        if len(positioned_chunks) > 1:
            # Check if positions are in ascending order
            positions = [pc[0] for pc in positioned_chunks]
            if positions != sorted(positions):
                issues.append("Chunks are not ordered by position")

        # Check hierarchical relationships
        for chunk in result.chunks:
            if chunk.parent_id:
                # Find parent chunk
                parent_found = any(c.id == chunk.parent_id for c in result.chunks)
                if not parent_found and self.strict_mode:
                    issues.append(f"Chunk {chunk.id} references missing parent {chunk.parent_id}")

            if chunk.children_ids:
                # Check if all children exist
                for child_id in chunk.children_ids:
                    child_found = any(c.id == child_id for c in result.chunks)
                    if not child_found and self.strict_mode:
                        issues.append(f"Chunk {chunk.id} references missing child {child_id}")

        return issues

    def get_quality_score(self, result: ChunkingResult) -> float:
        """
        Calculate a quality score for the chunking result.

        Args:
            result: Chunking result to score

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not result.chunks:
            return 0.0

        score = 1.0
        issues = self.validate_result(result)

        # Penalize validation issues
        critical_issues = sum(1 for issue in issues if any(word in issue.lower()
                             for word in ['missing', 'none', 'negative', 'mismatch']))
        warning_issues = len(issues) - critical_issues

        score -= critical_issues * 0.2  # 20% penalty per critical issue
        score -= warning_issues * 0.05   # 5% penalty per warning

        # Bonus for good characteristics
        if len(result.chunks) > 0:
            # Size consistency bonus
            sizes = [len(str(chunk.content)) for chunk in result.chunks]
            if len(set(sizes)) == 1:  # All same size
                score += 0.1
            elif len(sizes) > 1:
                cv = (max(sizes) - min(sizes)) / (sum(sizes) / len(sizes))
                if cv < 0.5:  # Low coefficient of variation
                    score += 0.05

            # Metadata completeness bonus
            complete_metadata = sum(1 for chunk in result.chunks
                                  if chunk.metadata and chunk.metadata.source)
            if complete_metadata == len(result.chunks):
                score += 0.1

        return max(0.0, min(1.0, score))
