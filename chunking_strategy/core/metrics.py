"""
Quality metrics and validation for chunking strategies.

This module provides metrics to evaluate the quality of chunking results,
including coherence, coverage, boundary quality, size consistency, and
information density measures.
"""

import logging
import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import re

from chunking_strategy.core.base import Chunk, ChunkingResult, ModalityType

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """
    Container for chunking quality metrics.

    All scores are normalized to 0.0-1.0 range where 1.0 is best.
    """

    # Size metrics
    size_consistency: float = 0.0      # How consistent chunk sizes are
    avg_chunk_size: float = 0.0        # Average chunk size
    size_variance: float = 0.0         # Variance in chunk sizes

    # Content quality
    coherence: float = 0.0             # Internal coherence of chunks
    coverage: float = 0.0              # Coverage of source content
    boundary_quality: float = 0.0      # Quality of chunk boundaries
    information_density: float = 0.0   # Information content per chunk

    # Structural metrics
    overlap_ratio: float = 0.0         # Ratio of overlapping content
    empty_chunk_ratio: float = 0.0     # Ratio of empty/very small chunks
    span_order_validity: float = 1.0   # Whether spans are in correct order

    # Performance metrics
    processing_efficiency: float = 0.0  # Chunks per processing time
    memory_efficiency: float = 0.0     # Memory usage per chunk

    # Overall score
    overall_score: float = 0.0         # Weighted combination of metrics

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "size_consistency": self.size_consistency,
            "avg_chunk_size": self.avg_chunk_size,
            "size_variance": self.size_variance,
            "coherence": self.coherence,
            "coverage": self.coverage,
            "boundary_quality": self.boundary_quality,
            "information_density": self.information_density,
            "overlap_ratio": self.overlap_ratio,
            "empty_chunk_ratio": self.empty_chunk_ratio,
            "span_order_validity": self.span_order_validity,
            "processing_efficiency": self.processing_efficiency,
            "memory_efficiency": self.memory_efficiency,
            "overall_score": self.overall_score,
        }

    def get_summary(self) -> str:
        """Get human-readable summary of metrics."""
        return (
            f"Quality Score: {self.overall_score:.3f}\n"
            f"  Size Consistency: {self.size_consistency:.3f}\n"
            f"  Coherence: {self.coherence:.3f}\n"
            f"  Coverage: {self.coverage:.3f}\n"
            f"  Boundary Quality: {self.boundary_quality:.3f}\n"
            f"  Information Density: {self.information_density:.3f}"
        )


class QualityMetric(ABC):
    """Abstract base class for quality metrics."""

    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize quality metric.

        Args:
            name: Name of the metric
            weight: Weight for overall score calculation
        """
        self.name = name
        self.weight = weight
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def compute(
        self,
        chunks: List[Chunk],
        original_content: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Compute the quality metric.

        Args:
            chunks: List of chunks to evaluate
            original_content: Original content before chunking
            **kwargs: Additional context

        Returns:
            Metric score (0.0 to 1.0)
        """
        raise NotImplementedError


class SizeConsistencyMetric(QualityMetric):
    """Measures consistency of chunk sizes."""

    def __init__(self, target_size: Optional[int] = None):
        """
        Initialize size consistency metric.

        Args:
            target_size: Target chunk size for evaluation
        """
        super().__init__("size_consistency")
        self.target_size = target_size

    def compute(
        self,
        chunks: List[Chunk],
        original_content: Optional[str] = None,
        **kwargs
    ) -> float:
        """Compute size consistency score."""
        if not chunks:
            return 0.0

        sizes = [chunk.size or len(str(chunk.content)) for chunk in chunks]

        if len(sizes) == 1:
            return 1.0  # Single chunk is perfectly consistent

        # Calculate coefficient of variation (lower is better)
        mean_size = statistics.mean(sizes)
        if mean_size == 0:
            return 0.0

        stdev_size = statistics.stdev(sizes)
        cv = stdev_size / mean_size

        # Convert to 0-1 score (lower CV is better)
        # CV of 0.5 or higher gets score of 0
        score = max(0.0, 1.0 - (cv / 0.5))

        return score


class CoherenceMetric(QualityMetric):
    """Measures internal coherence of chunks."""

    def __init__(self, modality: ModalityType = ModalityType.TEXT):
        """
        Initialize coherence metric.

        Args:
            modality: Modality type to evaluate
        """
        super().__init__("coherence")
        self.modality = modality

    def compute(
        self,
        chunks: List[Chunk],
        original_content: Optional[str] = None,
        **kwargs
    ) -> float:
        """Compute coherence score."""
        if not chunks:
            return 0.0

        text_chunks = [c for c in chunks if c.modality == ModalityType.TEXT]
        if not text_chunks:
            return 0.5  # Neutral score for non-text content

        coherence_scores = []

        for chunk in text_chunks:
            content = str(chunk.content).strip()
            if not content:
                coherence_scores.append(0.0)
                continue

            # Simple coherence heuristics
            score = 0.0

            # Sentence completeness (does it end with proper punctuation?)
            if content[-1] in '.!?':
                score += 0.3

            # No abrupt cuts in middle of words
            if not content.startswith(' ') or content.startswith(content[0].upper()):
                score += 0.2

            # Reasonable sentence structure (has some punctuation)
            sentences = re.split(r'[.!?]+', content)
            if len(sentences) > 1:
                score += 0.2

            # Not just fragments (has some connecting words)
            connecting_words = ['and', 'but', 'however', 'therefore', 'thus', 'also']
            if any(word in content.lower() for word in connecting_words):
                score += 0.1

            # Not too many line breaks (not fragmented)
            line_breaks = content.count('\n')
            if line_breaks < len(content) / 50:  # Less than 2% line breaks
                score += 0.2

            coherence_scores.append(min(1.0, score))

        return statistics.mean(coherence_scores) if coherence_scores else 0.0


class CoverageMetric(QualityMetric):
    """Measures how well chunks cover the original content."""

    def compute(
        self,
        chunks: List[Chunk],
        original_content: Optional[str] = None,
        **kwargs
    ) -> float:
        """Compute coverage score."""
        if not chunks or not original_content:
            return 0.0

        # For text content, measure character coverage
        original_chars = set(original_content.lower().replace(' ', ''))
        covered_chars = set()

        for chunk in chunks:
            if chunk.modality == ModalityType.TEXT:
                chunk_text = str(chunk.content).lower().replace(' ', '')
                covered_chars.update(chunk_text)

        if not original_chars:
            return 1.0

        coverage_ratio = len(covered_chars & original_chars) / len(original_chars)
        return coverage_ratio


class BoundaryQualityMetric(QualityMetric):
    """Measures quality of chunk boundaries."""

    def compute(
        self,
        chunks: List[Chunk],
        original_content: Optional[str] = None,
        **kwargs
    ) -> float:
        """Compute boundary quality score."""
        if len(chunks) <= 1:
            return 1.0  # Single chunk has perfect boundaries

        text_chunks = [c for c in chunks if c.modality == ModalityType.TEXT]
        if not text_chunks:
            return 0.5  # Neutral for non-text

        boundary_scores = []

        for i in range(len(text_chunks) - 1):
            current_chunk = str(text_chunks[i].content).strip()
            next_chunk = str(text_chunks[i + 1].content).strip()

            if not current_chunk or not next_chunk:
                boundary_scores.append(0.0)
                continue

            score = 0.0

            # Good: ends with sentence punctuation
            if current_chunk[-1] in '.!?':
                score += 0.4

            # Good: next chunk starts with capital letter
            if next_chunk[0].isupper():
                score += 0.3

            # Bad: ends mid-word (has hyphen or no space before next)
            if current_chunk.endswith('-'):
                score -= 0.2

            # Good: natural paragraph breaks
            if current_chunk.endswith('\n') or next_chunk.startswith('\n'):
                score += 0.2

            # Bad: breaks in middle of quotes
            if current_chunk.count('"') % 2 != 0:
                score -= 0.1

            boundary_scores.append(max(0.0, min(1.0, score)))

        return statistics.mean(boundary_scores) if boundary_scores else 0.0


class InformationDensityMetric(QualityMetric):
    """Measures information density of chunks."""

    def compute(
        self,
        chunks: List[Chunk],
        original_content: Optional[str] = None,
        **kwargs
    ) -> float:
        """Compute information density score."""
        if not chunks:
            return 0.0

        density_scores = []

        for chunk in chunks:
            if chunk.modality != ModalityType.TEXT:
                density_scores.append(0.5)  # Neutral for non-text
                continue

            content = str(chunk.content).strip()
            if not content:
                density_scores.append(0.0)
                continue

            # Simple information density heuristics
            total_chars = len(content)

            # Count meaningful characters (letters, numbers)
            meaningful_chars = sum(1 for c in content if c.isalnum())

            # Count unique words
            words = re.findall(r'\b\w+\b', content.lower())
            unique_words = len(set(words))
            total_words = len(words)

            # Calculate density components
            char_density = meaningful_chars / total_chars if total_chars > 0 else 0
            word_diversity = unique_words / total_words if total_words > 0 else 0

            # Penalize very short chunks
            length_factor = min(1.0, total_chars / 50)  # Penalty for < 50 chars

            density = (char_density * 0.4 + word_diversity * 0.4 + length_factor * 0.2)
            density_scores.append(density)

        return statistics.mean(density_scores)


class OverlapDetectionMetric(QualityMetric):
    """Detects and measures overlap between chunks."""

    def compute(
        self,
        chunks: List[Chunk],
        original_content: Optional[str] = None,
        **kwargs
    ) -> float:
        """Compute overlap ratio (lower is better for this metric)."""
        if len(chunks) <= 1:
            return 1.0  # No overlap possible

        text_chunks = [c for c in chunks if c.modality == ModalityType.TEXT]
        if not text_chunks:
            return 1.0

        total_content_length = 0
        overlapping_content = 0

        for i, chunk in enumerate(text_chunks):
            content = str(chunk.content)
            total_content_length += len(content)

            # Check overlap with other chunks
            for j, other_chunk in enumerate(text_chunks):
                if i >= j:  # Avoid double counting
                    continue

                other_content = str(other_chunk.content)

                # Find longest common substring
                overlap_length = self._longest_common_substring_length(content, other_content)
                if overlap_length > 10:  # Only count significant overlaps
                    overlapping_content += overlap_length

        if total_content_length == 0:
            return 1.0

        overlap_ratio = overlapping_content / total_content_length

        # Convert to score (lower overlap is better)
        # More than 20% overlap gets 0 score
        score = max(0.0, 1.0 - (overlap_ratio / 0.2))
        return score

    def _longest_common_substring_length(self, s1: str, s2: str) -> int:
        """Find length of longest common substring."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_length = max(max_length, dp[i][j])
                else:
                    dp[i][j] = 0

        return max_length


class ValidationMetric(QualityMetric):
    """Validates basic chunk properties."""

    def compute(
        self,
        chunks: List[Chunk],
        original_content: Optional[str] = None,
        **kwargs
    ) -> float:
        """Compute validation score."""
        if not chunks:
            return 0.0

        validation_scores = []

        for chunk in chunks:
            score = 1.0

            # Check for empty content
            if not chunk.content or (isinstance(chunk.content, str) and not chunk.content.strip()):
                score -= 0.5

            # Check for reasonable size
            size = chunk.size or len(str(chunk.content))
            if size < 5:  # Very small chunks
                score -= 0.3
            elif size > 10000:  # Very large chunks
                score -= 0.2

            # Check metadata presence
            if not chunk.metadata or not chunk.metadata.source:
                score -= 0.1

            # Check ID presence
            if not chunk.id:
                score -= 0.1

            validation_scores.append(max(0.0, score))

        return statistics.mean(validation_scores)


class ChunkingQualityEvaluator:
    """
    Comprehensive quality evaluator for chunking results.

    Combines multiple quality metrics to provide overall assessment
    of chunking performance and quality.
    """

    def __init__(
        self,
        metrics: Optional[List[QualityMetric]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize quality evaluator.

        Args:
            metrics: List of quality metrics to use
            weights: Weights for different metric categories
        """
        self.metrics = metrics or self._get_default_metrics()
        self.weights = weights or self._get_default_weights()
        self.logger = logging.getLogger(f"{__name__}.ChunkingQualityEvaluator")

    def evaluate(
        self,
        result: Union[ChunkingResult, List[Chunk]],
        original_content: Optional[str] = None,
        **kwargs
    ) -> QualityMetrics:
        """
        Evaluate quality of chunking result.

        Args:
            result: Chunking result or list of chunks
            original_content: Original content before chunking
            **kwargs: Additional evaluation context

        Returns:
            QualityMetrics with computed scores
        """
        if isinstance(result, ChunkingResult):
            chunks = result.chunks
            processing_time = result.processing_time
            memory_usage = getattr(result, 'memory_usage', None)
        else:
            chunks = result
            processing_time = None
            memory_usage = None

        if not chunks:
            self.logger.warning("No chunks to evaluate")
            return QualityMetrics()

        # Compute individual metrics
        metrics = QualityMetrics()

        # Size metrics
        sizes = [chunk.size or len(str(chunk.content)) for chunk in chunks]
        metrics.avg_chunk_size = statistics.mean(sizes)
        if len(sizes) > 1:
            metrics.size_variance = statistics.stdev(sizes)
        metrics.size_consistency = SizeConsistencyMetric("size_consistency").compute(chunks, original_content)

        # Content quality metrics
        metrics.coherence = CoherenceMetric("coherence").compute(chunks, original_content)
        metrics.coverage = CoverageMetric("coverage").compute(chunks, original_content)
        metrics.boundary_quality = BoundaryQualityMetric("boundary_quality").compute(chunks, original_content)
        metrics.information_density = InformationDensityMetric("information_density").compute(chunks, original_content)

        # Structural metrics
        metrics.overlap_ratio = 1.0 - OverlapDetectionMetric("overlap_detection").compute(chunks, original_content)

        # Empty chunk ratio
        empty_chunks = sum(1 for chunk in chunks
                          if not chunk.content or
                          (isinstance(chunk.content, str) and len(chunk.content.strip()) < 5))
        metrics.empty_chunk_ratio = empty_chunks / len(chunks)

        # Span order validation (simplified)
        metrics.span_order_validity = ValidationMetric("validation").compute(chunks, original_content)

        # Performance metrics
        if processing_time and processing_time > 0:
            metrics.processing_efficiency = len(chunks) / processing_time

        if memory_usage and memory_usage > 0:
            metrics.memory_efficiency = len(chunks) / memory_usage

        # Compute overall score
        metrics.overall_score = self._compute_overall_score(metrics)

        return metrics

    def _get_default_metrics(self) -> List[QualityMetric]:
        """Get default set of quality metrics."""
        return [
            SizeConsistencyMetric("size_consistency"),
            CoherenceMetric("coherence"),
            CoverageMetric("coverage"),
            BoundaryQualityMetric("boundary_quality"),
            InformationDensityMetric("information_density"),
            OverlapDetectionMetric("overlap_detection"),
            ValidationMetric("validation"),
        ]

    def _get_default_weights(self) -> Dict[str, float]:
        """Get default weights for metric categories."""
        return {
            "size_consistency": 0.15,
            "coherence": 0.25,
            "coverage": 0.20,
            "boundary_quality": 0.20,
            "information_density": 0.15,
            "validation": 0.05
        }

    def _compute_overall_score(self, metrics: QualityMetrics) -> float:
        """Compute weighted overall quality score."""
        scores = {
            "size_consistency": metrics.size_consistency,
            "coherence": metrics.coherence,
            "coverage": metrics.coverage,
            "boundary_quality": metrics.boundary_quality,
            "information_density": metrics.information_density,
            "validation": metrics.span_order_validity
        }

        # Apply penalties for structural issues
        penalties = 0.0
        if metrics.empty_chunk_ratio > 0.1:  # More than 10% empty chunks
            penalties += metrics.empty_chunk_ratio * 0.5

        if metrics.overlap_ratio > 0.1:  # More than 10% overlap
            penalties += metrics.overlap_ratio * 0.3

        # Compute weighted score
        weighted_sum = sum(scores[key] * self.weights.get(key, 0)
                          for key in scores if key in self.weights)
        total_weight = sum(self.weights.values())

        if total_weight == 0:
            return 0.0

        base_score = weighted_sum / total_weight
        final_score = max(0.0, base_score - penalties)

        return final_score

    def get_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """
        Get recommendations for improving chunking quality.

        Args:
            metrics: Computed quality metrics

        Returns:
            List of improvement recommendations
        """
        recommendations = []

        if metrics.size_consistency < 0.5:
            recommendations.append("Consider using fixed-size chunking for more consistent sizes")

        if metrics.coherence < 0.5:
            recommendations.append("Try sentence-based or paragraph-based chunking for better coherence")

        if metrics.coverage < 0.8:
            recommendations.append("Check for content loss during chunking process")

        if metrics.boundary_quality < 0.5:
            recommendations.append("Use semantic or structural boundaries for better chunk quality")

        if metrics.information_density < 0.4:
            recommendations.append("Consider filtering out low-content chunks or merging small chunks")

        if metrics.empty_chunk_ratio > 0.1:
            recommendations.append("Add validation to remove empty or very small chunks")

        if metrics.overlap_ratio > 0.2:
            recommendations.append("Reduce overlap between chunks or use non-overlapping strategy")

        if not recommendations:
            recommendations.append("Chunking quality looks good overall!")

        return recommendations
