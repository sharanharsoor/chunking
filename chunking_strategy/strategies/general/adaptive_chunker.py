"""
Adaptive Dynamic Chunking strategy.

The Adaptive Dynamic Chunker is an intelligent meta-chunker that automatically
optimizes chunking parameters and strategies based on content characteristics,
performance feedback, and historical data. It learns from past chunking operations
to continuously improve its performance.

Key features:
- Self-tuning parameters based on content analysis
- Performance learning and historical optimization
- Content-aware strategy selection
- Multi-strategy orchestration and comparison
- Intelligent feedback processing
- Real-time adaptation during processing
- Rich analytics and performance metrics
- Configurable adaptation algorithms
"""

import json
import logging
import math
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage, create_chunker
from chunking_strategy.core.streaming import StreamableChunker
from chunking_strategy.core.adaptive import AdaptableChunker


@dataclass
class ContentProfile:
    """Profile of content characteristics for adaptation."""
    file_type: str
    size_bytes: int
    estimated_entropy: float
    text_ratio: float
    structure_score: float  # How structured the content appears
    repetition_score: float  # Amount of repetitive content
    complexity_score: float  # Overall content complexity
    language: Optional[str] = None
    encoding: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a chunking operation."""
    strategy_used: str
    chunk_count: int
    avg_chunk_size: float
    processing_time: float
    memory_usage: float
    quality_score: float  # Subjective quality assessment
    user_satisfaction: Optional[float] = None  # User feedback
    deduplication_ratio: Optional[float] = None
    compression_ratio: Optional[float] = None
    boundary_quality: float = 0.0  # Quality of chunk boundaries

    def get_overall_score(self) -> float:
        """Calculate overall performance score."""
        # Weighted combination of metrics (higher is better)
        speed_score = max(0, 1.0 - (self.processing_time / 10.0))  # Normalize to 10s max
        quality_score = self.quality_score
        boundary_score = self.boundary_quality

        weights = {"speed": 0.3, "quality": 0.4, "boundary": 0.3}

        return (speed_score * weights["speed"] +
                quality_score * weights["quality"] +
                boundary_score * weights["boundary"])


@dataclass
class AdaptationRecord:
    """Record of an adaptation and its results."""
    timestamp: datetime
    content_profile: ContentProfile
    strategy_before: str
    strategy_after: str
    parameters_before: Dict[str, Any]
    parameters_after: Dict[str, Any]
    performance_before: Optional[PerformanceMetrics]
    performance_after: PerformanceMetrics
    adaptation_reason: str
    improvement_score: float


@register_chunker(
    name="adaptive",
    category="general",
    complexity=ComplexityLevel.HIGH,
    speed=SpeedLevel.MEDIUM,
    memory=MemoryUsage.MEDIUM,
    supported_formats=["any"],  # Universal - can adapt to any content
    dependencies=[],
    description="Adaptive Dynamic Chunker with self-tuning parameters and performance learning",
    use_cases=["mixed_content", "optimization", "learning_systems", "adaptive_processing", "meta_chunking"]
)
class AdaptiveChunker(StreamableChunker, AdaptableChunker):
    """
    Adaptive Dynamic Chunker that intelligently adjusts chunking strategies
    and parameters based on content characteristics and performance feedback.

    This is a meta-chunker that can utilize multiple underlying strategies,
    automatically selecting and tuning them for optimal performance.
    """

    def __init__(
        self,
        # Strategy selection
        available_strategies: List[str] = None,
        default_strategy: str = "fastcdc",
        strategy_selection_mode: str = "auto",  # "auto", "performance", "content"

        # Adaptation settings
        adaptation_threshold: float = 0.1,      # Minimum improvement needed to adapt
        learning_rate: float = 0.1,             # How quickly to adapt
        exploration_rate: float = 0.05,         # Rate of trying new strategies
        adaptation_interval: int = 10,          # Adapt every N files

        # Performance tracking
        history_size: int = 1000,               # Number of operations to remember
        performance_window: int = 50,           # Window for performance averaging
        min_samples: int = 5,                   # Minimum samples before adaptation

        # Content analysis
        enable_content_profiling: bool = True,
        enable_performance_learning: bool = True,
        enable_strategy_comparison: bool = True,

        # Storage
        persistence_file: Optional[str] = None,
        auto_save_interval: int = 100,          # Auto-save every N operations

        **kwargs
    ):
        """
        Initialize Adaptive Dynamic Chunker.

        Args:
            available_strategies: List of chunking strategies to choose from
            default_strategy: Default strategy to start with
            strategy_selection_mode: How to select strategies ("auto", "performance", "content")
            adaptation_threshold: Minimum improvement needed to trigger adaptation
            learning_rate: Rate of parameter adaptation
            exploration_rate: Rate of trying new strategies for learning
            adaptation_interval: How often to consider adaptations
            history_size: Number of historical records to maintain
            performance_window: Window size for performance averaging
            min_samples: Minimum samples needed before making adaptations
            enable_content_profiling: Whether to analyze content characteristics
            enable_performance_learning: Whether to learn from performance feedback
            enable_strategy_comparison: Whether to compare multiple strategies
            persistence_file: File to save/load adaptation history
            auto_save_interval: How often to auto-save history
            **kwargs: Additional parameters
        """
        super().__init__(
            name="adaptive",
            category="general",
            supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
            **kwargs
        )

        # Strategy management
        self.available_strategies = available_strategies or [
            "fastcdc", "fixed_size", "sentence_based", "paragraph_based"
        ]
        self.default_strategy = default_strategy
        self.current_strategy = default_strategy
        self.strategy_selection_mode = strategy_selection_mode

        # Adaptation parameters
        self.adaptation_threshold = adaptation_threshold
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.adaptation_interval = adaptation_interval

        # Performance tracking
        self.history_size = history_size
        self.performance_window = performance_window
        self.min_samples = min_samples

        # Feature flags
        self.enable_content_profiling = enable_content_profiling
        self.enable_performance_learning = enable_performance_learning
        self.enable_strategy_comparison = enable_strategy_comparison

        # Storage settings
        self.persistence_file = persistence_file
        self.auto_save_interval = auto_save_interval

        # Internal state
        self.operation_count = 0
        self.adaptation_history: deque = deque(maxlen=history_size)
        self.performance_history: deque = deque(maxlen=history_size)
        self.strategy_performance: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.content_strategy_map: Dict[str, str] = {}  # Content type -> best strategy
        self.parameter_cache: Dict[str, Dict[str, Any]] = {}  # Strategy -> parameters

        # Current chunker instance
        self._current_chunker = None
        self._chunker_cache: Dict[str, BaseChunker] = {}

        # Initialize strategy parameters for each available strategy
        self._initialize_strategy_parameters()

        # Load historical data if persistence is enabled
        if self.persistence_file:
            self._load_history()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Adaptive chunker initialized with {len(self.available_strategies)} strategies")

    def _initialize_strategy_parameters(self):
        """Initialize default parameters for each strategy."""
        # Default parameters for each strategy type
        strategy_defaults = {
            "fastcdc": {
                "min_chunk_size": 2048,
                "max_chunk_size": 65536,
                "avg_chunk_size": 8192,
                "hash_algorithm": "gear"
            },
            "fixed_size": {
                "chunk_size": 4096,
                "overlap": 0
            },
            "sentence_based": {
                "max_sentences": 5,
                "min_chunk_size": 500,
                "max_chunk_size": 10000
            },
            "paragraph_based": {
                "max_paragraphs": 3,
                "min_chunk_size": 1000,
                "max_chunk_size": 15000
            }
        }

        for strategy in self.available_strategies:
            if strategy in strategy_defaults:
                self.parameter_cache[strategy] = strategy_defaults[strategy].copy()
            else:
                # Generic defaults for unknown strategies
                self.parameter_cache[strategy] = {
                    "chunk_size": 4096,
                    "min_chunk_size": 1000,
                    "max_chunk_size": 10000
                }

    def _get_or_create_chunker(self, strategy: str, parameters: Dict[str, Any]) -> BaseChunker:
        """Get or create a chunker instance for the given strategy."""
        cache_key = f"{strategy}_{hash(frozenset(parameters.items()))}"

        if cache_key not in self._chunker_cache:
            try:
                self._chunker_cache[cache_key] = create_chunker(strategy, **parameters)
                self.logger.debug(f"Created {strategy} chunker with parameters: {parameters}")
            except Exception as e:
                self.logger.error(f"Failed to create {strategy} chunker: {e}")
                # Fallback to default strategy
                if strategy != self.default_strategy:
                    return self._get_or_create_chunker(self.default_strategy, self.parameter_cache[self.default_strategy])
                raise

        return self._chunker_cache[cache_key]

    def _profile_content(self, content: Union[str, bytes, Path]) -> ContentProfile:
        """Analyze content characteristics for adaptive chunking."""
        if not self.enable_content_profiling:
            return ContentProfile(
                file_type="unknown", size_bytes=0, estimated_entropy=4.0,
                text_ratio=0.5, structure_score=0.5, repetition_score=0.5, complexity_score=0.5
            )

        # Get content as bytes for analysis
        if isinstance(content, Path):
            if content.exists():
                content_bytes = content.read_bytes()
                file_type = content.suffix.lower().lstrip('.')
            else:
                return ContentProfile(
                    file_type="unknown", size_bytes=0, estimated_entropy=4.0,
                    text_ratio=0.0, structure_score=0.0, repetition_score=0.0, complexity_score=0.0
                )
        elif isinstance(content, str):
            content_bytes = content.encode('utf-8')
            file_type = "text"
        else:
            content_bytes = bytes(content)
            file_type = "binary"

        size_bytes = len(content_bytes)

        if size_bytes == 0:
            return ContentProfile(
                file_type=file_type, size_bytes=0, estimated_entropy=0.0,
                text_ratio=0.0, structure_score=0.0, repetition_score=0.0, complexity_score=0.0
            )

        # Calculate entropy
        byte_counts = [0] * 256
        for byte in content_bytes:
            byte_counts[byte] += 1

        entropy = 0.0
        for count in byte_counts:
            if count > 0:
                probability = count / size_bytes
                entropy -= probability * math.log2(probability)

        # Estimate text ratio
        try:
            if file_type == "binary":
                text_ratio = 0.0
            else:
                text = content_bytes.decode('utf-8', errors='ignore')
                printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
                text_ratio = printable_chars / len(text) if len(text) > 0 else 0.0
        except:
            text_ratio = 0.0

        # Analyze structure (simple heuristics)
        structure_score = 0.0
        if text_ratio > 0.7:  # Likely text content
            try:
                text = content_bytes.decode('utf-8', errors='ignore')
                # Look for structured patterns
                structure_indicators = [
                    '{', '}',  # JSON/code blocks
                    '<', '>',  # XML/HTML tags
                    '|',       # Tables
                    '\n\n',    # Paragraph breaks
                    '##',      # Markdown headers
                    '"""',     # Documentation blocks
                ]
                structure_count = sum(text.count(indicator) for indicator in structure_indicators)
                structure_score = min(1.0, structure_count / (len(text) / 1000))  # Normalize
            except:
                structure_score = 0.0

        # Analyze repetition
        repetition_score = 0.0
        if size_bytes > 100:
            # Simple repetition analysis - look for repeated byte patterns
            sample_size = min(1000, size_bytes)
            sample = content_bytes[:sample_size]
            unique_bigrams = len(set(sample[i:i+2] for i in range(len(sample)-1)))
            max_bigrams = len(sample) - 1
            repetition_score = 1.0 - (unique_bigrams / max_bigrams) if max_bigrams > 0 else 0.0

        # Calculate complexity score (combination of entropy and structure)
        complexity_score = (entropy / 8.0) * 0.6 + structure_score * 0.4

        return ContentProfile(
            file_type=file_type,
            size_bytes=size_bytes,
            estimated_entropy=entropy,
            text_ratio=text_ratio,
            structure_score=structure_score,
            repetition_score=repetition_score,
            complexity_score=complexity_score
        )

    def _select_strategy(self, content_profile: ContentProfile) -> str:
        """Select the best chunking strategy based on content profile."""
        if self.strategy_selection_mode == "content":
            return self._select_strategy_by_content(content_profile)
        elif self.strategy_selection_mode == "performance":
            return self._select_strategy_by_performance(content_profile)
        else:  # auto mode
            # Combine content and performance heuristics
            content_strategy = self._select_strategy_by_content(content_profile)
            if self.operation_count >= self.min_samples:
                performance_strategy = self._select_strategy_by_performance(content_profile)
                # Weight by confidence
                if len(self.strategy_performance[performance_strategy]) >= self.min_samples:
                    return performance_strategy
            return content_strategy

    def _select_strategy_by_content(self, content_profile: ContentProfile) -> str:
        """Select strategy based on content characteristics."""
        # Rule-based strategy selection
        if content_profile.file_type in ["json", "xml", "html"]:
            if "fastcdc" in self.available_strategies:
                return "fastcdc"

        if content_profile.text_ratio > 0.8:
            # Highly textual content
            if content_profile.structure_score > 0.5:
                if "paragraph_based" in self.available_strategies:
                    return "paragraph_based"
            else:
                if "sentence_based" in self.available_strategies:
                    return "sentence_based"

        if content_profile.repetition_score > 0.7:
            # Highly repetitive content - good for FastCDC
            if "fastcdc" in self.available_strategies:
                return "fastcdc"

        if content_profile.size_bytes > 100000:  # Large files
            if "fastcdc" in self.available_strategies:
                return "fastcdc"

        # Default strategy
        return self.current_strategy if self.current_strategy in self.available_strategies else self.default_strategy

    def _select_strategy_by_performance(self, content_profile: ContentProfile) -> str:
        """Select strategy based on historical performance."""
        # Look for similar content types in history
        content_key = f"{content_profile.file_type}_{int(content_profile.text_ratio*10)}_{int(content_profile.structure_score*10)}"

        if content_key in self.content_strategy_map:
            best_strategy = self.content_strategy_map[content_key]
            if best_strategy in self.available_strategies:
                return best_strategy

        # Find strategy with best average performance
        best_strategy = self.default_strategy
        best_score = 0.0

        for strategy, metrics_list in self.strategy_performance.items():
            if strategy in self.available_strategies and len(metrics_list) >= self.min_samples:
                recent_metrics = metrics_list[-self.performance_window:]
                avg_score = sum(m.get_overall_score() for m in recent_metrics) / len(recent_metrics)

                if avg_score > best_score:
                    best_score = avg_score
                    best_strategy = strategy

        return best_strategy

    def _optimize_parameters(self, strategy: str, content_profile: ContentProfile) -> Dict[str, Any]:
        """Optimize parameters for the given strategy based on content profile."""
        base_params = self.parameter_cache[strategy].copy()

        if not self.enable_performance_learning:
            return base_params

        # Content-aware parameter optimization
        if strategy == "fastcdc":
            # Adjust chunk sizes based on content characteristics
            if content_profile.size_bytes > 1000000:  # Large files
                base_params["avg_chunk_size"] = min(32768, base_params.get("avg_chunk_size", 8192) * 2)
            elif content_profile.size_bytes < 10000:  # Small files
                base_params["avg_chunk_size"] = max(2048, base_params.get("avg_chunk_size", 8192) // 2)

            # Adjust based on repetition
            if content_profile.repetition_score > 0.8:
                base_params["min_chunk_size"] = max(1024, base_params.get("min_chunk_size", 2048) // 2)

        elif strategy in ["sentence_based", "paragraph_based"]:
            # Adjust based on text characteristics
            if content_profile.structure_score > 0.7:
                if "max_sentences" in base_params:
                    base_params["max_sentences"] = max(3, base_params["max_sentences"] - 1)
                if "max_paragraphs" in base_params:
                    base_params["max_paragraphs"] = max(2, base_params["max_paragraphs"] - 1)

        elif strategy == "fixed_size":
            # Adjust chunk size based on content
            if content_profile.entropy > 6.0:  # High entropy content
                base_params["chunk_size"] = min(8192, base_params.get("chunk_size", 4096) * 2)
            elif content_profile.entropy < 3.0:  # Low entropy content
                base_params["chunk_size"] = max(2048, base_params.get("chunk_size", 4096) // 2)

        return base_params

    def _evaluate_performance(self, result: ChunkingResult, content_profile: ContentProfile,
                            strategy: str, start_time: float) -> PerformanceMetrics:
        """Evaluate the performance of a chunking operation."""
        processing_time = time.time() - start_time

        # Calculate basic metrics
        chunk_count = result.total_chunks
        total_content_size = sum(len(chunk.content.encode('utf-8')) for chunk in result.chunks)
        avg_chunk_size = total_content_size / chunk_count if chunk_count > 0 else 0

        # Estimate memory usage (simplified)
        memory_usage = total_content_size * 1.5  # Assume 1.5x overhead

        # Calculate quality score (heuristic-based)
        quality_score = self._calculate_quality_score(result, content_profile)

        # Calculate boundary quality
        boundary_quality = self._calculate_boundary_quality(result, content_profile)

        return PerformanceMetrics(
            strategy_used=strategy,
            chunk_count=chunk_count,
            avg_chunk_size=avg_chunk_size,
            processing_time=processing_time,
            memory_usage=memory_usage,
            quality_score=quality_score,
            boundary_quality=boundary_quality
        )

    def _calculate_quality_score(self, result: ChunkingResult, content_profile: ContentProfile) -> float:
        """Calculate a quality score for the chunking result."""
        if result.total_chunks == 0:
            return 0.0

        score = 0.8  # Base score

        # Penalize very small or very large chunks
        chunk_sizes = [len(chunk.content.encode('utf-8')) for chunk in result.chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        size_variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)
        size_cv = (size_variance ** 0.5) / avg_size if avg_size > 0 else 1.0

        # Lower coefficient of variation is better (more consistent chunk sizes)
        score += max(0, 0.2 - size_cv)

        # Bonus for reasonable chunk count
        if 5 <= result.total_chunks <= 100:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _calculate_boundary_quality(self, result: ChunkingResult, content_profile: ContentProfile) -> float:
        """Calculate the quality of chunk boundaries."""
        if result.total_chunks <= 1:
            return 1.0

        # Simple heuristic: prefer boundaries at natural breakpoints
        boundary_score = 0.5  # Base score

        # For text content, check if boundaries occur at sentence/paragraph breaks
        if content_profile.text_ratio > 0.7:
            natural_boundary_count = 0
            for chunk in result.chunks:
                content = chunk.content.strip()
                if content.endswith(('.', '!', '?', '\n\n')):
                    natural_boundary_count += 1

            boundary_ratio = natural_boundary_count / result.total_chunks
            boundary_score += boundary_ratio * 0.5

        return min(1.0, boundary_score)

    def _should_adapt(self) -> bool:
        """Determine if adaptation should be considered."""
        return (self.operation_count % self.adaptation_interval == 0 and
                self.operation_count >= self.min_samples and
                self.enable_performance_learning)

    def _perform_adaptation(self, content_profile: ContentProfile, current_performance: PerformanceMetrics):
        """Perform adaptation based on current performance and history."""
        if not self._should_adapt():
            return

        # Consider strategy changes
        if self.operation_count >= self.min_samples * 2:
            self._consider_strategy_adaptation(content_profile, current_performance)

        # Consider parameter changes
        self._consider_parameter_adaptation(content_profile, current_performance)

        # Update content-strategy mapping
        content_key = f"{content_profile.file_type}_{int(content_profile.text_ratio*10)}_{int(content_profile.structure_score*10)}"
        self.content_strategy_map[content_key] = current_performance.strategy_used

    def _consider_strategy_adaptation(self, content_profile: ContentProfile, current_performance: PerformanceMetrics):
        """Consider changing the current strategy."""
        current_strategy = current_performance.strategy_used
        current_score = current_performance.get_overall_score()

        # Compare with other strategies
        best_alternative = None
        best_score = current_score

        for strategy in self.available_strategies:
            if strategy != current_strategy and len(self.strategy_performance[strategy]) >= self.min_samples:
                recent_metrics = self.strategy_performance[strategy][-self.performance_window:]
                avg_score = sum(m.get_overall_score() for m in recent_metrics) / len(recent_metrics)

                if avg_score > best_score + self.adaptation_threshold:
                    best_alternative = strategy
                    best_score = avg_score

        if best_alternative:
            self.logger.info(f"Adapting strategy from {current_strategy} to {best_alternative} "
                           f"(score improvement: {best_score - current_score:.3f})")
            self.current_strategy = best_alternative

    def _consider_parameter_adaptation(self, content_profile: ContentProfile, current_performance: PerformanceMetrics):
        """Consider adjusting parameters for the current strategy."""
        strategy = current_performance.strategy_used
        current_params = self.parameter_cache[strategy].copy()

        # Simple parameter adaptation based on performance
        if current_performance.processing_time > 5.0:  # Too slow
            # Try to reduce chunk sizes or complexity
            if "avg_chunk_size" in current_params:
                current_params["avg_chunk_size"] = max(2048, int(current_params["avg_chunk_size"] * 0.8))
            elif "chunk_size" in current_params:
                current_params["chunk_size"] = max(1024, int(current_params["chunk_size"] * 0.8))

        elif current_performance.quality_score < 0.6:  # Poor quality
            # Try to improve quality
            if "min_chunk_size" in current_params:
                current_params["min_chunk_size"] = min(8192, int(current_params["min_chunk_size"] * 1.2))

        # Update cached parameters
        if current_params != self.parameter_cache[strategy]:
            self.parameter_cache[strategy] = current_params
            self.logger.debug(f"Adapted parameters for {strategy}: {current_params}")

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Perform adaptive chunking with intelligent strategy and parameter selection.

        Args:
            content: Content to chunk (string, bytes, or file path)
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with adaptively optimized chunks
        """
        start_time = time.time()
        self.operation_count += 1

        try:
            # Profile the content
            content_profile = self._profile_content(content)
            self.logger.debug(f"Content profile: {content_profile}")

            # Select optimal strategy
            selected_strategy = self._select_strategy(content_profile)

            # Optimize parameters for the strategy
            optimized_params = self._optimize_parameters(selected_strategy, content_profile)

            # Get or create the chunker
            chunker = self._get_or_create_chunker(selected_strategy, optimized_params)

            # Perform chunking
            result = chunker.chunk(content, source_info, **kwargs)

            # Evaluate performance
            performance = self._evaluate_performance(result, content_profile, selected_strategy, start_time)

            # Store performance data
            self.performance_history.append(performance)
            self.strategy_performance[selected_strategy].append(performance)

            # Consider adaptation
            self._perform_adaptation(content_profile, performance)

            # Update result metadata
            result.strategy_used = f"adaptive_{selected_strategy}"
            if result.source_info is None:
                result.source_info = {}

            result.source_info.update({
                "adaptive_strategy": selected_strategy,
                "content_profile": asdict(content_profile),
                "optimized_parameters": optimized_params,
                "performance_metrics": asdict(performance),
                "operation_count": self.operation_count,
                "adaptation_enabled": self.enable_performance_learning
            })

            # Auto-save if needed
            if self.persistence_file and self.operation_count % self.auto_save_interval == 0:
                self._save_history()

            self.logger.info(f"Adaptive chunking completed: {result.total_chunks} chunks using {selected_strategy} "
                           f"(score: {performance.get_overall_score():.3f})")

            return result

        except Exception as e:
            self.logger.error(f"Error in adaptive chunking: {e}")
            # Fallback to default strategy
            try:
                fallback_chunker = self._get_or_create_chunker(self.default_strategy,
                                                             self.parameter_cache[self.default_strategy])
                result = fallback_chunker.chunk(content, source_info, **kwargs)
                result.strategy_used = f"adaptive_fallback_{self.default_strategy}"
                return result
            except Exception as fallback_error:
                self.logger.error(f"Fallback chunking also failed: {fallback_error}")
                raise


    def get_adaptation_info(self) -> Dict[str, Any]:
        """Get current adaptation state and statistics."""
        # Calculate strategy performance summary
        strategy_stats = {}
        for strategy, metrics in self.strategy_performance.items():
            if metrics:
                scores = [m.get_overall_score() for m in metrics]
                strategy_stats[strategy] = {
                    'usage_count': len(metrics),
                    'avg_score': sum(scores) / len(scores) if scores else 0.0,
                    'best_score': max(scores) if scores else 0.0,
                    'recent_score': scores[-1] if scores else 0.0
                }

        return {
            'current_strategy': self.current_strategy,
            'operation_count': self.operation_count,
            'adaptation_threshold': self.adaptation_threshold,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'total_adaptations': len(self.adaptation_history),
            'strategy_performance': strategy_stats,
            'content_strategy_mappings': dict(self.content_strategy_map),
            'performance_learning_enabled': self.enable_performance_learning,
            'content_profiling_enabled': self.enable_content_profiling,
            'available_strategies': list(self.available_strategies),
            'history_size': len(self.performance_history)
        }

    def chunk_stream(self, content_stream, **kwargs):
        """Chunk data from a stream using adaptive strategy selection."""
        # For streaming, we'll use the current best strategy
        if not hasattr(self, '_stream_chunker'):
            strategy = self.current_strategy
            params = self.parameter_cache[strategy]
            self._stream_chunker = self._get_or_create_chunker(strategy, params)

        # Delegate to the selected chunker
        if hasattr(self._stream_chunker, 'chunk_stream'):
            for chunk in self._stream_chunker.chunk_stream(content_stream, **kwargs):
                yield chunk
        else:
            # Fallback: collect stream and chunk normally
            data_chunks = []
            for chunk in content_stream:
                data_chunks.append(chunk if isinstance(chunk, str) else chunk.decode('utf-8', errors='ignore'))

            result = self.chunk(''.join(data_chunks), **kwargs)
            for chunk in result.chunks:
                yield chunk

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats (union of all available strategies)."""
        return ["any"]  # Adaptive chunker supports any format through its strategies

    def estimate_chunks(self, content: Union[str, bytes, Path]) -> int:
        """Estimate number of chunks that will be generated."""
        content_profile = self._profile_content(content)
        strategy = self._select_strategy(content_profile)
        params = self._optimize_parameters(strategy, content_profile)
        chunker = self._get_or_create_chunker(strategy, params)

        if hasattr(chunker, 'estimate_chunks'):
            return chunker.estimate_chunks(content)
        else:
            # Fallback estimation
            if isinstance(content, Path) and content.exists():
                size = content.stat().st_size
            elif isinstance(content, str):
                size = len(content.encode('utf-8'))
            elif isinstance(content, bytes):
                size = len(content)
            else:
                size = len(str(content).encode('utf-8'))

            # Rough estimation based on average chunk size
            avg_chunk_size = params.get('avg_chunk_size', params.get('chunk_size', 4096))
            return max(1, size // avg_chunk_size)

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """Adapt parameters based on external feedback."""
        adaptation = {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "operation_count": self.operation_count,
            "current_strategy": self.current_strategy
        }

        # Update learning rate based on feedback
        if feedback_score < 0.5:
            self.learning_rate = min(0.5, self.learning_rate * 1.1)  # Learn faster from poor performance
        elif feedback_score > 0.8:
            self.learning_rate = max(0.01, self.learning_rate * 0.9)  # Learn slower when doing well

        # Update strategy performance if we have recent performance data
        if self.performance_history:
            latest_performance = self.performance_history[-1]
            latest_performance.user_satisfaction = feedback_score

            # Record adaptation if significant change is needed
            if abs(feedback_score - latest_performance.quality_score) > self.adaptation_threshold:
                if feedback_score < latest_performance.quality_score:
                    # Performance degraded, consider reverting or trying different strategy
                    self.exploration_rate = min(0.2, self.exploration_rate * 1.2)
                else:
                    # Performance improved, reduce exploration
                    self.exploration_rate = max(0.01, self.exploration_rate * 0.8)

        self.adaptation_history.append(adaptation)
        self.logger.info(f"Adapted based on {feedback_type} feedback: {feedback_score}")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return [dict(record) for record in self.adaptation_history]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance across all strategies."""
        summary = {
            "total_operations": self.operation_count,
            "strategies_used": list(self.strategy_performance.keys()),
            "current_strategy": self.current_strategy,
            "strategy_performance": {}
        }

        for strategy, metrics_list in self.strategy_performance.items():
            if metrics_list:
                recent_metrics = metrics_list[-self.performance_window:]
                summary["strategy_performance"][strategy] = {
                    "total_uses": len(metrics_list),
                    "recent_uses": len(recent_metrics),
                    "avg_score": sum(m.get_overall_score() for m in recent_metrics) / len(recent_metrics),
                    "avg_processing_time": sum(m.processing_time for m in recent_metrics) / len(recent_metrics),
                    "avg_quality": sum(m.quality_score for m in recent_metrics) / len(recent_metrics)
                }

        return summary

    def _save_history(self):
        """Save adaptation history to file."""
        if not self.persistence_file:
            return

        try:
            history_data = {
                "operation_count": self.operation_count,
                "current_strategy": self.current_strategy,
                "parameter_cache": self.parameter_cache,
                "content_strategy_map": self.content_strategy_map,
                "adaptation_history": [asdict(record) for record in self.adaptation_history],
                "performance_summary": self.get_performance_summary()
            }

            with open(self.persistence_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)

            self.logger.debug(f"Saved adaptation history to {self.persistence_file}")
        except Exception as e:
            self.logger.error(f"Failed to save adaptation history: {e}")

    def _load_history(self):
        """Load adaptation history from file."""
        if not self.persistence_file or not Path(self.persistence_file).exists():
            return

        try:
            with open(self.persistence_file, 'r') as f:
                history_data = json.load(f)

            self.operation_count = history_data.get("operation_count", 0)
            self.current_strategy = history_data.get("current_strategy", self.default_strategy)
            self.parameter_cache.update(history_data.get("parameter_cache", {}))
            self.content_strategy_map.update(history_data.get("content_strategy_map", {}))

            self.logger.info(f"Loaded adaptation history from {self.persistence_file} "
                           f"({self.operation_count} operations)")
        except Exception as e:
            self.logger.error(f"Failed to load adaptation history: {e}")

    def reset_adaptation(self):
        """Reset all adaptation history and return to default state."""
        self.operation_count = 0
        self.adaptation_history.clear()
        self.performance_history.clear()
        self.strategy_performance.clear()
        self.content_strategy_map.clear()
        self.current_strategy = self.default_strategy
        self._initialize_strategy_parameters()

        self.logger.info("Reset all adaptation history")

    def set_exploration_mode(self, enabled: bool):
        """Enable or disable exploration of new strategies."""
        if enabled:
            self.exploration_rate = 0.1
            self.logger.info("Enabled exploration mode")
        else:
            self.exploration_rate = 0.01
            self.logger.info("Disabled exploration mode")
