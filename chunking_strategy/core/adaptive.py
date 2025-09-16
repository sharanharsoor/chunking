"""
Adaptive chunking framework that can modify behavior based on feedback.

This module provides adaptive chunking capabilities that can adjust parameters
or switch strategies based on feedback signals from retrieval accuracy,
response quality, latency, or other performance metrics.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import json
from pathlib import Path

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    AdaptableChunker
)
from chunking_strategy.core.registry import create_chunker, get_chunker_metadata

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback that can be used for adaptation."""

    QUALITY = "quality"              # Retrieval quality, answer accuracy
    PERFORMANCE = "performance"      # Speed, memory usage
    RELEVANCE = "relevance"         # Content relevance scores
    COVERAGE = "coverage"           # Information coverage metrics
    COHERENCE = "coherence"         # Chunk coherence scores
    USER_RATING = "user_rating"     # Direct user feedback
    CUSTOM = "custom"               # Custom feedback metrics


class AdaptationStrategy(Enum):
    """Strategies for how to adapt based on feedback."""

    PARAMETER_TUNING = "parameter_tuning"    # Adjust existing parameters
    STRATEGY_SWITCHING = "strategy_switching" # Switch to different strategy
    HYBRID_APPROACH = "hybrid_approach"      # Combine multiple strategies
    MULTI_LEVEL = "multi_level"             # Different strategies per level


@dataclass
class FeedbackSignal:
    """Represents a feedback signal for adaptation."""

    score: float                    # Numeric score (0.0 to 1.0)
    feedback_type: FeedbackType    # Type of feedback
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"        # Source of feedback
    context: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0           # Weight for this feedback

    def is_positive(self, threshold: float = 0.6) -> bool:
        """Check if feedback is positive above threshold."""
        return self.score >= threshold

    def is_negative(self, threshold: float = 0.4) -> bool:
        """Check if feedback is negative below threshold."""
        return self.score <= threshold


@dataclass
class AdaptationRecord:
    """Record of an adaptation that was made."""

    timestamp: float
    trigger_feedback: FeedbackSignal
    old_config: Dict[str, Any]
    new_config: Dict[str, Any]
    adaptation_type: str
    reason: str
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "trigger_score": self.trigger_feedback.score,
            "feedback_type": self.trigger_feedback.feedback_type.value,
            "old_config": self.old_config,
            "new_config": self.new_config,
            "adaptation_type": self.adaptation_type,
            "reason": self.reason,
            "success": self.success
        }


class AdaptationPolicy(ABC):
    """
    Abstract base class for adaptation policies.

    Adaptation policies determine how and when to adapt chunker behavior
    based on feedback signals.
    """

    @abstractmethod
    def should_adapt(
        self,
        feedback_history: List[FeedbackSignal],
        current_config: Dict[str, Any]
    ) -> bool:
        """
        Determine if adaptation should occur.

        Args:
            feedback_history: Recent feedback signals
            current_config: Current chunker configuration

        Returns:
            True if adaptation should occur
        """
        raise NotImplementedError

    @abstractmethod
    def adapt_config(
        self,
        feedback_history: List[FeedbackSignal],
        current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt configuration based on feedback.

        Args:
            feedback_history: Recent feedback signals
            current_config: Current chunker configuration

        Returns:
            New configuration
        """
        raise NotImplementedError


class ThresholdPolicy(AdaptationPolicy):
    """Simple threshold-based adaptation policy."""

    def __init__(
        self,
        adaptation_threshold: float = 0.4,
        min_feedback_count: int = 3,
        window_size: int = 10
    ):
        """
        Initialize threshold policy.

        Args:
            adaptation_threshold: Score threshold for triggering adaptation
            min_feedback_count: Minimum feedback signals before adapting
            window_size: Size of feedback window to consider
        """
        self.adaptation_threshold = adaptation_threshold
        self.min_feedback_count = min_feedback_count
        self.window_size = window_size

    def should_adapt(
        self,
        feedback_history: List[FeedbackSignal],
        current_config: Dict[str, Any]
    ) -> bool:
        """Adapt if recent feedback is consistently below threshold."""
        if len(feedback_history) < self.min_feedback_count:
            return False

        recent_feedback = feedback_history[-self.window_size:]
        avg_score = sum(f.score * f.weight for f in recent_feedback) / sum(f.weight for f in recent_feedback)

        return avg_score < self.adaptation_threshold

    def adapt_config(
        self,
        feedback_history: List[FeedbackSignal],
        current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt configuration based on feedback type."""
        recent_feedback = feedback_history[-self.window_size:]
        primary_feedback_type = max(
            set(f.feedback_type for f in recent_feedback),
            key=lambda t: sum(1 for f in recent_feedback if f.feedback_type == t)
        )

        new_config = current_config.copy()

        if primary_feedback_type == FeedbackType.QUALITY:
            # Increase chunk size for better quality
            if "chunk_size" in new_config:
                new_config["chunk_size"] = min(new_config["chunk_size"] * 1.2, 4096)

        elif primary_feedback_type == FeedbackType.PERFORMANCE:
            # Decrease chunk size for better performance
            if "chunk_size" in new_config:
                new_config["chunk_size"] = max(new_config["chunk_size"] * 0.8, 256)

        elif primary_feedback_type == FeedbackType.COHERENCE:
            # Adjust overlap for better coherence
            if "overlap_size" in new_config:
                new_config["overlap_size"] = min(new_config["overlap_size"] * 1.5, 512)

        return new_config


class AdaptiveChunker:
    """
    Adaptive chunker that can modify behavior based on feedback.

    This chunker wraps other chunking strategies and can adapt their parameters
    or switch strategies entirely based on feedback signals about performance,
    quality, or other metrics.

    Examples:
        Basic adaptive chunker:
        ```python
        chunker = AdaptiveChunker("sentence_based")

        # Initial chunking
        result = chunker.chunk("some text")

        # Provide feedback and re-chunk
        chunker.add_feedback(0.3, FeedbackType.QUALITY)
        result = chunker.chunk("more text")  # Will adapt if needed
        ```

        With custom policy:
        ```python
        policy = ThresholdPolicy(adaptation_threshold=0.5)
        chunker = AdaptiveChunker("fixed_size", adaptation_policy=policy)
        ```
    """

    def __init__(
        self,
        base_strategy: Union[str, BaseChunker],
        adaptation_policy: Optional[AdaptationPolicy] = None,
        fallback_strategies: Optional[List[str]] = None,
        max_adaptations: int = 5,
        adaptation_cooldown: float = 10.0,  # seconds
        **base_config
    ):
        """
        Initialize adaptive chunker.

        Args:
            base_strategy: Initial chunking strategy
            adaptation_policy: Policy for when/how to adapt
            fallback_strategies: Strategies to try if base fails
            max_adaptations: Maximum adaptations per session
            adaptation_cooldown: Minimum time between adaptations
            **base_config: Base configuration for chunker
        """
        self.base_strategy_name = base_strategy if isinstance(base_strategy, str) else base_strategy.name
        self.base_config = base_config
        self.adaptation_policy = adaptation_policy or ThresholdPolicy()
        self.fallback_strategies = fallback_strategies or []
        self.max_adaptations = max_adaptations
        self.adaptation_cooldown = adaptation_cooldown

        # State tracking
        self.feedback_history: List[FeedbackSignal] = []
        self.adaptation_history: List[AdaptationRecord] = []
        self.current_strategy_name = self.base_strategy_name
        self.current_config = base_config.copy()
        self.last_adaptation_time = 0.0
        self.adaptation_count = 0

        # Initialize current chunker
        self.current_chunker = self._create_chunker(self.current_strategy_name, self.current_config)

        self.logger = logging.getLogger(f"{__name__}.AdaptiveChunker")
        self.logger.info(f"Initialized with base strategy: {self.base_strategy_name}")

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        auto_adapt: bool = True,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk content with automatic adaptation based on feedback.

        Args:
            content: Content to chunk
            source_info: Information about the source
            auto_adapt: Whether to automatically adapt before chunking
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with adaptation metadata
        """
        # Check if we should adapt before chunking
        if auto_adapt:
            self._check_and_adapt()

        # Perform chunking
        try:
            result = self.current_chunker.chunk(content, source_info, **kwargs)

            # Add adaptation metadata
            result.strategy_used = self.current_strategy_name
            if hasattr(result, 'source_info') and result.source_info:
                result.source_info["adaptive_chunker"] = True
                result.source_info["adaptation_count"] = self.adaptation_count
                result.source_info["current_strategy"] = self.current_strategy_name

            return result

        except Exception as e:
            self.logger.error(f"Error with current strategy {self.current_strategy_name}: {e}")

            # Try fallback strategies
            for fallback in self.fallback_strategies:
                try:
                    self.logger.info(f"Trying fallback strategy: {fallback}")
                    fallback_chunker = self._create_chunker(fallback, self.base_config)
                    if fallback_chunker:
                        result = fallback_chunker.chunk(content, source_info, **kwargs)
                        result.strategy_used = fallback
                        result.fallback_strategies = [fallback]
                        return result
                except Exception as fe:
                    self.logger.error(f"Fallback {fallback} also failed: {fe}")

            # If all strategies fail, re-raise original error
            raise e

    def add_feedback(
        self,
        score: float,
        feedback_type: FeedbackType = FeedbackType.QUALITY,
        source: str = "user",
        context: Optional[Dict[str, Any]] = None,
        weight: float = 1.0
    ) -> None:
        """
        Add feedback signal for adaptation.

        Args:
            score: Feedback score (0.0 to 1.0)
            feedback_type: Type of feedback
            source: Source of the feedback
            context: Additional context information
            weight: Weight for this feedback signal
        """
        feedback = FeedbackSignal(
            score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
            feedback_type=feedback_type,
            source=source,
            context=context or {},
            weight=weight
        )

        self.feedback_history.append(feedback)
        self.logger.debug(f"Added feedback: {feedback_type.value}={score:.3f} from {source}")

        # Limit feedback history size
        max_history = 100
        if len(self.feedback_history) > max_history:
            self.feedback_history = self.feedback_history[-max_history:]

    def chunk_with_feedback(
        self,
        content: Union[str, bytes, Path],
        feedback_score: float,
        feedback_type: FeedbackType = FeedbackType.QUALITY,
        **kwargs
    ) -> ChunkingResult:
        """
        Convenience method to chunk with immediate feedback.

        Args:
            content: Content to chunk
            feedback_score: Feedback score for adaptation
            feedback_type: Type of feedback
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult after adaptation
        """
        # Add feedback first
        self.add_feedback(feedback_score, feedback_type)

        # Then chunk with adaptation
        return self.chunk(content, **kwargs)

    def force_adapt(self) -> bool:
        """
        Force adaptation regardless of policy.

        Returns:
            True if adaptation occurred
        """
        if not self.feedback_history:
            self.logger.warning("No feedback history available for adaptation")
            return False

        return self._perform_adaptation()

    def reset_adaptation(self) -> None:
        """Reset adaptation state to initial configuration."""
        self.current_strategy_name = self.base_strategy_name
        self.current_config = self.base_config.copy()
        self.current_chunker = self._create_chunker(self.current_strategy_name, self.current_config)
        self.adaptation_count = 0
        self.last_adaptation_time = 0.0

        self.logger.info("Reset to base strategy and configuration")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of adaptations made."""
        return [record.to_dict() for record in self.adaptation_history]

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback received."""
        if not self.feedback_history:
            return {"total_feedback": 0}

        by_type = {}
        for feedback in self.feedback_history:
            feedback_type = feedback.feedback_type.value
            if feedback_type not in by_type:
                by_type[feedback_type] = []
            by_type[feedback_type].append(feedback.score)

        summary = {"total_feedback": len(self.feedback_history)}
        for feedback_type, scores in by_type.items():
            summary[f"{feedback_type}_avg"] = sum(scores) / len(scores)
            summary[f"{feedback_type}_count"] = len(scores)

        return summary

    def save_adaptation_state(self, file_path: Union[str, Path]) -> None:
        """Save adaptation state to file."""
        state = {
            "base_strategy": self.base_strategy_name,
            "base_config": self.base_config,
            "current_strategy": self.current_strategy_name,
            "current_config": self.current_config,
            "adaptation_count": self.adaptation_count,
            "feedback_history": [
                {
                    "score": f.score,
                    "type": f.feedback_type.value,
                    "timestamp": f.timestamp,
                    "source": f.source,
                    "context": f.context,
                    "weight": f.weight
                }
                for f in self.feedback_history
            ],
            "adaptation_history": self.get_adaptation_history()
        }

        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Adaptation state saved to {file_path}")

    def load_adaptation_state(self, file_path: Union[str, Path]) -> None:
        """Load adaptation state from file."""
        with open(file_path, 'r') as f:
            state = json.load(f)

        self.base_strategy_name = state["base_strategy"]
        self.base_config = state["base_config"]
        self.current_strategy_name = state["current_strategy"]
        self.current_config = state["current_config"]
        self.adaptation_count = state["adaptation_count"]

        # Restore feedback history
        self.feedback_history = []
        for f_data in state.get("feedback_history", []):
            feedback = FeedbackSignal(
                score=f_data["score"],
                feedback_type=FeedbackType(f_data["type"]),
                timestamp=f_data["timestamp"],
                source=f_data["source"],
                context=f_data["context"],
                weight=f_data["weight"]
            )
            self.feedback_history.append(feedback)

        # Recreate current chunker
        self.current_chunker = self._create_chunker(self.current_strategy_name, self.current_config)

        self.logger.info(f"Adaptation state loaded from {file_path}")

    def _check_and_adapt(self) -> None:
        """Check if adaptation should occur and perform it."""
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_adaptation_time < self.adaptation_cooldown:
            return

        # Check if we've hit max adaptations
        if self.adaptation_count >= self.max_adaptations:
            self.logger.debug("Maximum adaptations reached")
            return

        # Check policy
        if self.adaptation_policy.should_adapt(self.feedback_history, self.current_config):
            self._perform_adaptation()

    def _perform_adaptation(self) -> bool:
        """Perform the actual adaptation."""
        if not self.feedback_history:
            return False

        old_config = self.current_config.copy()
        old_strategy = self.current_strategy_name

        try:
            # Get new configuration from policy
            new_config = self.adaptation_policy.adapt_config(self.feedback_history, self.current_config)

            # Check if configuration actually changed
            if new_config == old_config:
                self.logger.debug("Adaptation policy returned same configuration")
                return False

            # Create new chunker with adapted configuration
            new_chunker = self._create_chunker(self.current_strategy_name, new_config)
            if not new_chunker:
                self.logger.error("Failed to create adapted chunker")
                return False

            # Apply adaptation
            self.current_config = new_config
            self.current_chunker = new_chunker
            self.adaptation_count += 1
            self.last_adaptation_time = time.time()

            # Record adaptation
            trigger_feedback = self.feedback_history[-1] if self.feedback_history else None
            record = AdaptationRecord(
                timestamp=self.last_adaptation_time,
                trigger_feedback=trigger_feedback,
                old_config=old_config,
                new_config=new_config,
                adaptation_type="parameter_tuning",
                reason="policy_triggered",
                success=True
            )
            self.adaptation_history.append(record)

            self.logger.info(
                f"Adapted configuration: {old_strategy} "
                f"(adaptation #{self.adaptation_count})"
            )
            self.logger.debug(f"Config changes: {old_config} -> {new_config}")

            return True

        except Exception as e:
            self.logger.error(f"Error during adaptation: {e}")

            # Record failed adaptation
            if self.feedback_history:
                record = AdaptationRecord(
                    timestamp=time.time(),
                    trigger_feedback=self.feedback_history[-1],
                    old_config=old_config,
                    new_config={},
                    adaptation_type="parameter_tuning",
                    reason="adaptation_failed",
                    success=False
                )
                self.adaptation_history.append(record)

            return False

    def _create_chunker(self, strategy_name: str, config: Dict[str, Any]) -> Optional[BaseChunker]:
        """Create a chunker instance with given strategy and config."""
        try:
            return create_chunker(strategy_name, **config)
        except Exception as e:
            self.logger.error(f"Failed to create chunker {strategy_name}: {e}")
            return None

    def __repr__(self) -> str:
        return (
            f"AdaptiveChunker(current_strategy='{self.current_strategy_name}', "
            f"adaptations={self.adaptation_count}, "
            f"feedback_count={len(self.feedback_history)})"
        )
