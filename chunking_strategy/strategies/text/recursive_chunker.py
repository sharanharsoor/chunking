"""
Recursive Chunking Strategy.

This module implements a recursive (hierarchical) text chunker that applies
different chunking strategies at multiple levels to create a hierarchical
structure of chunks. This enables fine-grained control over text division
with parent-child relationships preserved.

Key Features:
- Multi-level hierarchical chunking with configurable depth
- Different chunking strategies at each level (e.g., semantic -> sentence -> token)
- Parent-child relationships preserved in chunk metadata
- Configurable parameters for each hierarchical level
- Adaptive depth based on content complexity
- Support for mixed strategy combinations (structural + semantic + size-based)
- Hierarchical streaming with level-aware processing
- Quality metrics for hierarchical coherence
- Fallback strategies for each level

Author: AI Assistant
Date: 2024
"""

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
from pathlib import Path

from chunking_strategy.core.base import (
    BaseChunker, StreamableChunker, AdaptableChunker,
    Chunk, ChunkingResult, ChunkMetadata, ModalityType
)
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage


class LevelStrategy(Enum):
    """Available strategies for each hierarchical level."""
    SEMANTIC = "semantic"
    BOUNDARY_AWARE = "boundary_aware"
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    TOKEN = "token"
    OVERLAPPING = "overlapping"


@dataclass
class HierarchyLevel:
    """Configuration for a single hierarchy level."""
    name: str
    strategy: LevelStrategy
    parameters: Dict[str, Any]
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    target_chunk_size: int = 500
    max_children: int = 10
    quality_threshold: float = 0.7


@dataclass
class RecursiveChunk(Chunk):
    """Extended chunk with hierarchical information."""
    level: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    hierarchy_path: str = ""
    level_strategy: str = ""
    quality_score: float = 0.0

    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []


@register_chunker(
    name="recursive",
    category="text",
    description="Hierarchical text chunker that applies different chunking strategies at multiple levels",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "html", "xml", "json", "csv", "rtf"],
    complexity=ComplexityLevel.HIGH,
    dependencies=[],
    optional_dependencies=["sentence-transformers", "tiktoken", "transformers"],
    speed=SpeedLevel.MEDIUM,
    memory=MemoryUsage.MEDIUM,
    quality=0.85,
    use_cases=["hierarchical document analysis", "multi-level text processing", "structured content chunking"],
    best_for=["complex documents", "academic papers", "technical documentation", "structured content"],
    limitations=["may have deep recursion for very nested content", "performance depends on level-specific chunkers"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=True
)
class RecursiveChunker(StreamableChunker, AdaptableChunker):
    """
    Recursive hierarchical chunking strategy.

    Applies different chunking strategies at multiple levels to create
    a hierarchical structure with preserved parent-child relationships.
    """

    def __init__(
        self,
        hierarchy_levels: Optional[List[Dict[str, Any]]] = None,
        max_depth: int = 3,
        adaptive_depth: bool = True,
        preserve_hierarchy: bool = True,
        quality_threshold: float = 0.7,
        enable_streaming: bool = True,
        **kwargs
    ):
        """
        Initialize the Recursive Chunker.

        Args:
            hierarchy_levels: List of level configurations
            max_depth: Maximum recursion depth
            adaptive_depth: Adapt depth based on content
            preserve_hierarchy: Maintain parent-child relationships
            quality_threshold: Minimum quality for chunk acceptance
            enable_streaming: Enable streaming capabilities
        """
        super().__init__(
            name="recursive",
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        self.max_depth = max_depth
        self.adaptive_depth = adaptive_depth
        self.preserve_hierarchy = preserve_hierarchy
        self.quality_threshold = quality_threshold
        self.enable_streaming = enable_streaming

        # Initialize hierarchy levels
        self.hierarchy_levels = self._initialize_hierarchy_levels(hierarchy_levels)

        # Performance tracking
        self.performance_stats = {
            "total_levels_processed": 0,
            "total_recursive_calls": 0,
            "hierarchical_processing_time": 0.0,
            "chunk_relationship_time": 0.0,
            "quality_evaluation_time": 0.0,
            "adaptive_adjustments": 0,
            "fallback_count": 0
        }

        # Level-specific chunkers cache
        self._chunker_cache = {}

        # Adaptation history tracking
        self._adaptation_history = []

        logging.info(f"RecursiveChunker initialized with {len(self.hierarchy_levels)} levels, max_depth={max_depth}")

    def _initialize_hierarchy_levels(self, levels_config: Optional[List[Dict[str, Any]]]) -> List[HierarchyLevel]:
        """Initialize hierarchy level configurations."""
        if not levels_config:
            # Default 3-level hierarchy: Paragraph -> Sentence -> Fixed Size
            levels_config = [
                {
                    "name": "paragraph",
                    "strategy": "paragraph",
                    "parameters": {"min_paragraph_length": 50},
                    "target_chunk_size": 1200,
                    "max_chunk_size": 2500
                },
                {
                    "name": "sentence",
                    "strategy": "sentence",
                    "parameters": {"max_sentences": 4},
                    "target_chunk_size": 400,
                    "max_chunk_size": 800
                },
                {
                    "name": "fixed",
                    "strategy": "fixed_size",
                    "parameters": {"chunk_size": 200},
                    "target_chunk_size": 200,
                    "max_chunk_size": 300
                }
            ]

        levels = []
        for i, level_config in enumerate(levels_config):
            if i >= self.max_depth:
                break

            level = HierarchyLevel(
                name=level_config.get("name", f"level_{i}"),
                strategy=LevelStrategy(level_config["strategy"]),
                parameters=level_config.get("parameters", {}),
                min_chunk_size=level_config.get("min_chunk_size", 50),
                max_chunk_size=level_config.get("max_chunk_size", 2000),
                target_chunk_size=level_config.get("target_chunk_size", 500),
                max_children=level_config.get("max_children", 10),
                quality_threshold=level_config.get("quality_threshold", self.quality_threshold)
            )
            levels.append(level)

        return levels

    def _get_level_chunker(self, level: HierarchyLevel) -> BaseChunker:
        """Get or create a chunker for the specified level."""
        cache_key = f"{level.strategy.value}_{hash(str(level.parameters))}"

        if cache_key in self._chunker_cache:
            return self._chunker_cache[cache_key]

        # Import and create appropriate chunker
        chunker = None
        try:
            if level.strategy == LevelStrategy.SEMANTIC:
                from chunking_strategy.strategies.text.semantic_chunker import SemanticChunker
                chunker = SemanticChunker(**level.parameters)

            elif level.strategy == LevelStrategy.BOUNDARY_AWARE:
                from chunking_strategy.strategies.text.boundary_aware_chunker import BoundaryAwareChunker
                chunker = BoundaryAwareChunker(**level.parameters)

            elif level.strategy == LevelStrategy.SENTENCE:
                from chunking_strategy.strategies.text.sentence_based import SentenceBasedChunker
                chunker = SentenceBasedChunker(**level.parameters)

            elif level.strategy == LevelStrategy.PARAGRAPH:
                from chunking_strategy.strategies.text.paragraph_based import ParagraphBasedChunker
                chunker = ParagraphBasedChunker(**level.parameters)

            elif level.strategy == LevelStrategy.TOKEN:
                from chunking_strategy.strategies.text.token_based_chunker import TokenBasedChunker
                chunker = TokenBasedChunker(**level.parameters)

            elif level.strategy == LevelStrategy.OVERLAPPING:
                from chunking_strategy.strategies.text.overlapping_window_chunker import OverlappingWindowChunker
                chunker = OverlappingWindowChunker(**level.parameters)

            elif level.strategy == LevelStrategy.FIXED_SIZE:
                from chunking_strategy.strategies.general.fixed_size import FixedSizeChunker
                chunker = FixedSizeChunker(
                    chunk_size=level.parameters.get("chunk_size", level.target_chunk_size)
                )

            if chunker:
                self._chunker_cache[cache_key] = chunker
                return chunker

        except ImportError as e:
            logging.warning(f"Failed to import chunker for {level.strategy}: {e}")

        # Fallback to a simple sentence-based approach
        from chunking_strategy.strategies.text.sentence_based import SentenceBasedChunker
        fallback_chunker = SentenceBasedChunker()
        self._chunker_cache[cache_key] = fallback_chunker
        self.performance_stats["fallback_count"] += 1

        return fallback_chunker

    def _evaluate_chunk_quality(self, chunk: Chunk, level: HierarchyLevel) -> float:
        """Evaluate the quality of a chunk at a specific level."""
        start_time = time.time()

        # Basic quality metrics
        size_score = min(1.0, len(chunk.content) / level.target_chunk_size)
        if len(chunk.content) > level.max_chunk_size:
            size_score *= 0.5  # Penalty for oversized chunks

        # Content coherence (simple heuristic)
        sentences = chunk.content.split('. ')
        coherence_score = min(1.0, len(sentences) / 5.0)  # Optimal around 5 sentences

        # Boundary quality (check for incomplete sentences)
        boundary_score = 1.0
        if not chunk.content.strip().endswith(('.', '!', '?', '\n')):
            boundary_score = 0.8

        quality_score = (size_score * 0.4 + coherence_score * 0.4 + boundary_score * 0.2)

        self.performance_stats["quality_evaluation_time"] += time.time() - start_time
        return quality_score

    def _recursive_chunk(
        self,
        content: str,
        level: int = 0,
        parent_id: Optional[str] = None,
        hierarchy_path: str = ""
    ) -> List[RecursiveChunk]:
        """
        Recursively chunk content at the specified hierarchical level.

        Args:
            content: Text content to chunk
            level: Current hierarchy level (0-based)
            parent_id: ID of parent chunk
            hierarchy_path: Hierarchical path (e.g., "0.1.2")

        Returns:
            List of recursive chunks with hierarchical metadata
        """
        start_time = time.time()
        self.performance_stats["total_recursive_calls"] += 1

        # Check termination conditions
        if level >= len(self.hierarchy_levels) or level >= self.max_depth:
            return []

        if not content.strip():
            return []

        # Get current level configuration
        current_level = self.hierarchy_levels[level]

        # Adaptive depth: Skip levels if content is too small
        if self.adaptive_depth and len(content) < current_level.min_chunk_size:
            if level + 1 < len(self.hierarchy_levels):
                return self._recursive_chunk(content, level + 1, parent_id, hierarchy_path)
            else:
                return []

        # Get appropriate chunker for this level
        chunker = self._get_level_chunker(current_level)

        try:
            # Perform chunking at current level
            result = chunker.chunk(content)
            chunks = result.chunks

        except Exception as e:
            logging.warning(f"Chunking failed at level {level} ({current_level.strategy.value}): {e}")
            # Try fallback to next level
            if level + 1 < len(self.hierarchy_levels):
                return self._recursive_chunk(content, level + 1, parent_id, hierarchy_path)
            else:
                return []

        recursive_chunks = []

        for chunk_idx, chunk in enumerate(chunks):
            # Create hierarchical path
            current_path = f"{hierarchy_path}.{chunk_idx}" if hierarchy_path else str(chunk_idx)

            # Evaluate chunk quality
            quality_score = self._evaluate_chunk_quality(chunk, current_level)

            # Skip low-quality chunks if threshold is set
            if quality_score < current_level.quality_threshold:
                continue

            # Create recursive chunk
            recursive_chunk = RecursiveChunk(
                id=f"recursive_{level}_{chunk_idx}_{hash(chunk.content) % 10000}",
                content=chunk.content,
                modality=chunk.modality,
                metadata=ChunkMetadata(
                    source=chunk.metadata.source,
                    source_type=chunk.metadata.source_type,
                    position=f"level_{level}_chunk_{chunk_idx}",
                    offset=chunk.metadata.offset,
                    length=len(chunk.content),
                    extra={
                        "level": level,
                        "hierarchy_path": current_path,
                        "level_strategy": current_level.strategy.value,
                        "level_name": current_level.name,
                        "parent_id": parent_id,
                        "quality_score": quality_score,
                        "chunker_used": "recursive",
                        "chunking_strategy": "recursive"
                    }
                ),
                size=len(chunk.content),
                level=level,
                parent_id=parent_id,
                hierarchy_path=current_path,
                level_strategy=current_level.strategy.value,
                quality_score=quality_score
            )

            # Recursively process this chunk at the next level
            if level + 1 < len(self.hierarchy_levels):
                child_chunks = self._recursive_chunk(
                    chunk.content,
                    level + 1,
                    recursive_chunk.id,
                    current_path
                )

                # Update parent-child relationships
                if child_chunks:
                    recursive_chunk.children_ids = [child.id for child in child_chunks]
                    recursive_chunks.extend(child_chunks)

            recursive_chunks.append(recursive_chunk)

        self.performance_stats["hierarchical_processing_time"] += time.time() - start_time
        self.performance_stats["total_levels_processed"] += 1

        return recursive_chunks

    def _create_hierarchical_metadata(
        self,
        chunks: List[RecursiveChunk],
        processing_time: float
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for hierarchical chunking result."""
        # Analyze hierarchical structure
        levels = set(chunk.level for chunk in chunks)
        level_counts = {level: len([c for c in chunks if c.level == level]) for level in levels}

        # Calculate relationship statistics
        parent_child_pairs = sum(1 for chunk in chunks if chunk.children_ids)
        avg_children_per_parent = (
            sum(len(chunk.children_ids) for chunk in chunks if chunk.children_ids) /
            parent_child_pairs if parent_child_pairs > 0 else 0
        )

        # Quality statistics
        quality_scores = [chunk.quality_score for chunk in chunks if hasattr(chunk, 'quality_score')]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        return {
            "recursive_metadata": {
                "max_depth_configured": self.max_depth,
                "actual_levels_used": len(levels),
                "level_distribution": level_counts,
                "level_strategies": {
                    level.name: level.strategy.value
                    for level in self.hierarchy_levels
                },
                "total_hierarchical_relationships": parent_child_pairs,
                "avg_children_per_parent": avg_children_per_parent,
                "avg_chunk_quality": avg_quality,
                "adaptive_depth_enabled": self.adaptive_depth,
                "processing_time": processing_time,
                "performance_stats": self.performance_stats.copy(),
                "preserve_hierarchy": self.preserve_hierarchy
            },
            "chunking_strategy": "recursive",
            "total_levels": len(levels),
            "hierarchical_structure": True
        }

    def chunk(self, content: Union[str, Path], source_info: Optional[Dict[str, Any]] = None, **kwargs) -> ChunkingResult:
        """
        Perform hierarchical recursive chunking on the content.

        Args:
            content: Text content or file path to chunk
            source_info: Additional source information
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with hierarchical chunk structure
        """
        start_time = time.time()

        # Process input content
        if isinstance(content, Path):
            text_content = content.read_text(encoding='utf-8')
            source_type = "file"
            source_path = str(content)
        else:
            text_content = str(content)
            source_type = source_info.get("source_type", "content") if source_info else "content"
            source_path = source_info.get("source_path", "string") if source_info else "string"

        # Handle empty content
        if not text_content.strip():
            return self._create_empty_result(source_type, source_path)

        try:
            # Perform recursive chunking starting from level 0
            recursive_chunks = self._recursive_chunk(text_content, level=0)

            if not recursive_chunks:
                return self._create_fallback_result(text_content, source_type, source_path)

            # Convert RecursiveChunks to regular Chunks for result
            regular_chunks = []
            for rc in recursive_chunks:
                regular_chunk = Chunk(
                    id=rc.id,
                    content=rc.content,
                    modality=rc.modality,
                    metadata=rc.metadata,
                    size=rc.size,
                    parent_id=rc.parent_id,
                    children_ids=rc.children_ids
                )
                regular_chunks.append(regular_chunk)

            processing_time = time.time() - start_time

            # Create enhanced source info
            enhanced_source_info = source_info.copy() if source_info else {}
            enhanced_source_info.update({
                "source": source_path,
                "source_type": source_type,
                "total_recursive_chunks": len(regular_chunks)
            })
            enhanced_source_info.update(self._create_hierarchical_metadata(recursive_chunks, processing_time))

            return ChunkingResult(
                chunks=regular_chunks,
                processing_time=processing_time,
                strategy_used="recursive",
                source_info=enhanced_source_info
            )

        except Exception as e:
            logging.error(f"Recursive chunking failed: {e}")
            return self._create_fallback_result(text_content, source_type, source_path)

    def _create_empty_result(self, source_type: str, source_path: str) -> ChunkingResult:
        """Create empty chunking result."""
        return ChunkingResult(
            chunks=[],
            processing_time=0.0,
            strategy_used="recursive",
            source_info={
                "source": source_path,
                "source_type": source_type,
                "total_recursive_chunks": 0,
                "recursive_metadata": {
                    "error": "Empty content",
                    "levels_used": 0
                }
            }
        )

    def _create_fallback_result(self, content: str, source_type: str, source_path: str) -> ChunkingResult:
        """Create fallback single-chunk result."""
        fallback_chunk = Chunk(
            id="recursive_fallback_0",
            content=content,
            modality=ModalityType.TEXT,
            metadata=ChunkMetadata(
                source=source_path,
                source_type=source_type,
                position="fallback_chunk",
                offset=0,
                length=len(content),
                extra={
                    "chunker_used": "recursive",
                    "chunking_strategy": "recursive",
                    "fallback_mode": True,
                    "level": 0
                }
            ),
            size=len(content)
        )

        return ChunkingResult(
            chunks=[fallback_chunk],
            processing_time=0.001,
            strategy_used="recursive",
            source_info={
                "source": source_path,
                "source_type": source_type,
                "total_recursive_chunks": 1,
                "recursive_metadata": {
                    "fallback_used": True,
                    "reason": "Recursive processing failed"
                }
            }
        )

    def chunk_stream(self, content_stream: Iterator[str], **kwargs) -> Iterator[Chunk]:
        """
        Stream hierarchical chunks from content stream.

        Args:
            content_stream: Iterator of content strings
            **kwargs: Additional parameters

        Yields:
            Hierarchical chunks as they are processed
        """
        if not self.enable_streaming:
            # Fallback to batch processing
            full_content = "".join(content_stream)
            result = self.chunk(full_content, **kwargs)
            for chunk in result.chunks:
                yield chunk
            return

        buffer = ""
        chunk_counter = 0

        for content_piece in content_stream:
            buffer += content_piece

            # Process buffer when it reaches a reasonable size
            if len(buffer) >= 1000:  # Configurable threshold
                try:
                    recursive_chunks = self._recursive_chunk(buffer, level=0)

                    for rc in recursive_chunks:
                        chunk = Chunk(
                            id=f"stream_recursive_{chunk_counter}",
                            content=rc.content,
                            modality=rc.modality,
                            metadata=rc.metadata,
                            size=rc.size,
                            parent_id=rc.parent_id,
                            children_ids=rc.children_ids
                        )
                        yield chunk
                        chunk_counter += 1

                    buffer = ""  # Clear processed buffer

                except Exception as e:
                    logging.warning(f"Streaming recursive chunking error: {e}")
                    continue

        # Process remaining buffer
        if buffer.strip():
            try:
                recursive_chunks = self._recursive_chunk(buffer, level=0)
                for rc in recursive_chunks:
                    chunk = Chunk(
                        id=f"stream_recursive_final_{chunk_counter}",
                        content=rc.content,
                        modality=rc.modality,
                        metadata=rc.metadata,
                        size=rc.size,
                        parent_id=rc.parent_id,
                        children_ids=rc.children_ids
                    )
                    yield chunk
                    chunk_counter += 1
            except Exception as e:
                logging.warning(f"Final streaming chunk error: {e}")

    def adapt_parameters(self, feedback_score: float, feedback_type: str = "quality", **kwargs) -> Dict[str, Any]:
        """
        Adapt hierarchical parameters based on feedback.

        Args:
            feedback_score: Feedback score (0.0-1.0)
            feedback_type: Type of feedback ("quality", "performance", "structure")
            **kwargs: Additional feedback parameters

        Returns:
            Dictionary of parameter changes made
        """
        import time

        changes = {}
        adaptation_record = {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "changes": {},
            "kwargs": kwargs
        }

        if feedback_type == "quality" and feedback_score < 0.6:
            # Reduce quality thresholds to be less strict
            for level in self.hierarchy_levels:
                old_threshold = level.quality_threshold
                level.quality_threshold = max(0.3, level.quality_threshold - 0.1)
                changes[f"level_{level.name}_quality_threshold"] = {
                    "old": old_threshold,
                    "new": level.quality_threshold
                }

            self.performance_stats["adaptive_adjustments"] += 1

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Reduce max_depth for better performance
            if self.max_depth > 1:
                old_depth = self.max_depth
                self.max_depth = max(1, self.max_depth - 1)
                changes["max_depth"] = {"old": old_depth, "new": self.max_depth}

                # Update hierarchy levels
                self.hierarchy_levels = self.hierarchy_levels[:self.max_depth]
                changes["active_levels"] = len(self.hierarchy_levels)

        elif feedback_type == "structure" and feedback_score > 0.8:
            # Increase depth for better hierarchical structure
            if self.max_depth < 5 and len(self.hierarchy_levels) > self.max_depth:
                old_depth = self.max_depth
                self.max_depth = min(5, self.max_depth + 1)
                changes["max_depth"] = {"old": old_depth, "new": self.max_depth}

        # Record adaptation history
        adaptation_record["changes"] = changes
        self._adaptation_history.append(adaptation_record)

        # Keep only recent history (last 50 adaptations)
        if len(self._adaptation_history) > 50:
            self._adaptation_history = self._adaptation_history[-50:]

        return changes

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of adaptations made.

        Returns:
            List of adaptation records with timestamps and changes
        """
        return self._adaptation_history.copy()

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "chunker_name": "recursive",
            "max_depth": self.max_depth,
            "adaptive_depth": self.adaptive_depth,
            "preserve_hierarchy": self.preserve_hierarchy,
            "quality_threshold": self.quality_threshold,
            "enable_streaming": self.enable_streaming,
            "hierarchy_levels": [
                {
                    "name": level.name,
                    "strategy": level.strategy.value,
                    "parameters": level.parameters,
                    "min_chunk_size": level.min_chunk_size,
                    "max_chunk_size": level.max_chunk_size,
                    "target_chunk_size": level.target_chunk_size,
                    "quality_threshold": level.quality_threshold
                }
                for level in self.hierarchy_levels
            ],
            "performance_stats": self.performance_stats
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum recursion depth"
                },
                "adaptive_depth": {
                    "type": "boolean",
                    "description": "Enable adaptive depth based on content"
                },
                "preserve_hierarchy": {
                    "type": "boolean",
                    "description": "Maintain parent-child relationships"
                },
                "quality_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum quality score for chunk acceptance"
                },
                "hierarchy_levels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "strategy": {
                                "type": "string",
                                "enum": ["semantic", "boundary_aware", "sentence", "paragraph", "token", "overlapping", "fixed_size"]
                            },
                            "parameters": {"type": "object"},
                            "target_chunk_size": {"type": "integer", "minimum": 50}
                        },
                        "required": ["name", "strategy"]
                    }
                }
            }
        }

    def get_supported_formats(self) -> List[str]:
        """Get list of supported input formats."""
        return ["txt", "md", "html", "xml", "json", "csv", "rtf"]

    def estimate_chunks(self, content: Union[str, Path], **kwargs) -> int:
        """
        Estimate the number of chunks that would be produced.

        Args:
            content: Input content
            **kwargs: Additional parameters

        Returns:
            Estimated number of chunks
        """
        if isinstance(content, Path):
            text_content = content.read_text(encoding='utf-8')
        else:
            text_content = str(content)

        if not text_content.strip():
            return 0

        # Estimate based on hierarchy levels and target sizes
        total_estimate = 0
        current_content_size = len(text_content)

        for level in self.hierarchy_levels:
            level_estimate = max(1, current_content_size // level.target_chunk_size)
            total_estimate += level_estimate
            current_content_size = level_estimate * (level.target_chunk_size // 2)  # Estimate for next level

            if current_content_size < level.min_chunk_size:
                break

        return total_estimate
