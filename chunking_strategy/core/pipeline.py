"""
Pipeline system for chaining chunking operations and transformations.

This module provides a flexible pipeline framework that allows users to define
sequences of chunking operations, transformations, and post-processing steps
that can be executed in order and exported/imported as JSON configurations.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ModalityType
)
from chunking_strategy.core.registry import create_chunker

logger = logging.getLogger(__name__)


class PipelineStep(ABC):
    """
    Abstract base class for pipeline steps.

    Pipeline steps can be chunkers, transformers, filters, or any other
    operation that processes chunks or chunking results.
    """

    def __init__(self, name: str, **config):
        """
        Initialize pipeline step.

        Args:
            name: Human-readable name for the step
            **config: Configuration parameters for the step
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def process(
        self,
        input_data: Union[str, bytes, List[Chunk], ChunkingResult],
        **kwargs
    ) -> Union[List[Chunk], ChunkingResult]:
        """
        Process input data and return result.

        Args:
            input_data: Input to process
            **kwargs: Additional processing parameters

        Returns:
            Processed result (chunks or chunking result)
        """
        raise NotImplementedError("Pipeline steps must implement process()")

    def get_config(self) -> Dict[str, Any]:
        """Get step configuration."""
        return self.config.copy()

    def validate_input(self, input_data: Any) -> None:
        """Validate input data before processing."""
        pass  # Default: no validation

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ChunkerStep(PipelineStep):
    """Pipeline step that wraps a chunking strategy."""

    def __init__(self, name: str, strategy: Union[str, BaseChunker], **config):
        """
        Initialize chunker step.

        Args:
            name: Name for this step
            strategy: Chunker strategy name or instance
            **config: Chunker configuration
        """
        super().__init__(name, **config)

        if isinstance(strategy, str):
            self.chunker = create_chunker(strategy, **config)
            if not self.chunker:
                raise ValueError(f"Unknown chunker strategy: {strategy}")
        elif isinstance(strategy, BaseChunker):
            self.chunker = strategy
        else:
            raise TypeError("Strategy must be chunker name or BaseChunker instance")

        self.strategy_name = strategy if isinstance(strategy, str) else strategy.name

    def process(
        self,
        input_data: Union[str, bytes, List[Chunk], ChunkingResult],
        **kwargs
    ) -> ChunkingResult:
        """Process input through the chunker."""
        # Handle different input types
        if isinstance(input_data, ChunkingResult):
            # Re-chunk existing chunks
            content = self._combine_chunks(input_data.chunks)
        elif isinstance(input_data, list) and all(isinstance(c, Chunk) for c in input_data):
            # Re-chunk chunk list
            content = self._combine_chunks(input_data)
        else:
            # Direct content
            content = input_data

        return self.chunker.chunk(content, **kwargs)

    def _combine_chunks(self, chunks: List[Chunk]) -> str:
        """Combine chunks back into content for re-chunking."""
        combined = []
        for chunk in chunks:
            if isinstance(chunk.content, str):
                combined.append(chunk.content)
            elif isinstance(chunk.content, bytes):
                try:
                    combined.append(chunk.content.decode('utf-8'))
                except UnicodeDecodeError:
                    self.logger.warning(f"Could not decode chunk {chunk.id}")
        return "\n".join(combined)


class FilterStep(PipelineStep):
    """Pipeline step for filtering chunks based on criteria."""

    def __init__(
        self,
        name: str,
        filter_func: Optional[Callable[[Chunk], bool]] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        modalities: Optional[List[ModalityType]] = None,
        **config
    ):
        """
        Initialize filter step.

        Args:
            name: Name for this step
            filter_func: Custom filter function
            min_size: Minimum chunk size
            max_size: Maximum chunk size
            modalities: Allowed modalities
            **config: Additional filter configuration
        """
        super().__init__(name, **config)
        self.filter_func = filter_func
        self.min_size = min_size
        self.max_size = max_size
        self.modalities = modalities or []

    def process(
        self,
        input_data: Union[List[Chunk], ChunkingResult],
        **kwargs
    ) -> List[Chunk]:
        """Filter chunks based on configured criteria."""
        if isinstance(input_data, ChunkingResult):
            chunks = input_data.chunks
        else:
            chunks = input_data

        filtered = []
        for chunk in chunks:
            if self._should_keep_chunk(chunk):
                filtered.append(chunk)

        self.logger.info(f"Filtered {len(chunks)} -> {len(filtered)} chunks")
        return filtered

    def _should_keep_chunk(self, chunk: Chunk) -> bool:
        """Determine if chunk should be kept."""
        # Size filters
        if self.min_size is not None and (chunk.size or 0) < self.min_size:
            return False
        if self.max_size is not None and (chunk.size or 0) > self.max_size:
            return False

        # Modality filter
        if self.modalities and chunk.modality not in self.modalities:
            return False

        # Custom filter
        if self.filter_func and not self.filter_func(chunk):
            return False

        return True


class TransformStep(PipelineStep):
    """Pipeline step for transforming chunk content or metadata."""

    def __init__(
        self,
        name: str,
        transform_func: Callable[[Chunk], Chunk],
        **config
    ):
        """
        Initialize transform step.

        Args:
            name: Name for this step
            transform_func: Function to transform chunks
            **config: Additional configuration
        """
        super().__init__(name, **config)
        self.transform_func = transform_func

    def process(
        self,
        input_data: Union[List[Chunk], ChunkingResult],
        **kwargs
    ) -> List[Chunk]:
        """Transform chunks using the configured function."""
        if isinstance(input_data, ChunkingResult):
            chunks = input_data.chunks
        else:
            chunks = input_data

        transformed = []
        for chunk in chunks:
            try:
                transformed_chunk = self.transform_func(chunk)
                transformed.append(transformed_chunk)
            except Exception as e:
                self.logger.error(f"Error transforming chunk {chunk.id}: {e}")
                # Keep original chunk on error
                transformed.append(chunk)

        return transformed


class MergeStep(PipelineStep):
    """Pipeline step for merging small adjacent chunks."""

    def __init__(
        self,
        name: str,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        merge_separator: str = " ",
        **config
    ):
        """
        Initialize merge step.

        Args:
            name: Name for this step
            min_chunk_size: Minimum size before merging
            max_chunk_size: Maximum size after merging
            merge_separator: Separator when merging content
            **config: Additional configuration
        """
        super().__init__(name, **config)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.merge_separator = merge_separator

    def process(
        self,
        input_data: Union[List[Chunk], ChunkingResult],
        **kwargs
    ) -> List[Chunk]:
        """Merge small adjacent chunks."""
        if isinstance(input_data, ChunkingResult):
            chunks = input_data.chunks
        else:
            chunks = input_data

        if not chunks:
            return chunks

        merged = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            # Check if we should merge
            should_merge = (
                (current_chunk.size or 0) < self.min_chunk_size or
                (next_chunk.size or 0) < self.min_chunk_size
            ) and (
                (current_chunk.size or 0) + (next_chunk.size or 0) <= self.max_chunk_size
            ) and (
                current_chunk.modality == next_chunk.modality == ModalityType.TEXT
            )

            if should_merge:
                # Merge chunks
                current_chunk = self._merge_chunks(current_chunk, next_chunk)
            else:
                # Add current chunk and start new one
                merged.append(current_chunk)
                current_chunk = next_chunk

        # Add final chunk
        merged.append(current_chunk)

        self.logger.info(f"Merged {len(chunks)} -> {len(merged)} chunks")
        return merged

    def _merge_chunks(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        """Merge two chunks into one."""
        # Combine content
        if isinstance(chunk1.content, str) and isinstance(chunk2.content, str):
            merged_content = chunk1.content + self.merge_separator + chunk2.content
        else:
            merged_content = str(chunk1.content) + self.merge_separator + str(chunk2.content)

        # Create new metadata
        merged_metadata = chunk1.metadata
        merged_metadata.extra["merged_from"] = [chunk1.id, chunk2.id]

        # Create merged chunk
        return Chunk(
            id=f"merged_{chunk1.id}_{chunk2.id}",
            content=merged_content,
            modality=chunk1.modality,
            metadata=merged_metadata,
            parent_id=chunk1.parent_id,
            children_ids=chunk1.children_ids + chunk2.children_ids
        )


@dataclass
class PipelineResult:
    """Result from executing a chunking pipeline."""

    chunks: List[Chunk]
    total_processing_time: float
    step_times: Dict[str, float] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution."""
        return {
            "total_chunks": len(self.chunks),
            "total_time": self.total_processing_time,
            "step_times": self.step_times,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
        }


class ChunkingPipeline:
    """
    Configurable pipeline for chaining chunking operations.

    Supports multiple types of steps including chunkers, filters, transformers,
    and custom processing steps. Pipelines can be exported and imported as
    JSON configurations.

    Examples:
        Basic pipeline:
        ```python
        pipeline = ChunkingPipeline([
            ("chunker", ChunkerStep("text_chunker", "sentence_based")),
            ("filter", FilterStep("size_filter", min_size=50)),
            ("merge", MergeStep("merge_small", min_chunk_size=100))
        ])
        result = pipeline.process("text content")
        ```

        From configuration:
        ```python
        config = {
            "steps": [
                {"type": "chunker", "name": "chunker", "strategy": "fixed_size", "chunk_size": 1024},
                {"type": "filter", "name": "filter", "min_size": 50}
            ]
        }
        pipeline = ChunkingPipeline.from_config(config)
        ```
    """

    def __init__(
        self,
        steps: Optional[List[Tuple[str, PipelineStep]]] = None,
        name: str = "chunking_pipeline"
    ):
        """
        Initialize chunking pipeline.

        Args:
            steps: List of (name, step) tuples
            name: Name for the pipeline
        """
        self.name = name
        self.steps: List[Tuple[str, PipelineStep]] = steps or []
        self.logger = logging.getLogger(f"{__name__}.{name}")

    def add_step(self, name: str, step: PipelineStep) -> "ChunkingPipeline":
        """
        Add a step to the pipeline.

        Args:
            name: Name for the step
            step: Pipeline step to add

        Returns:
            Self for method chaining
        """
        self.steps.append((name, step))
        return self

    def remove_step(self, name: str) -> "ChunkingPipeline":
        """
        Remove a step from the pipeline.

        Args:
            name: Name of step to remove

        Returns:
            Self for method chaining
        """
        self.steps = [(n, s) for n, s in self.steps if n != name]
        return self

    def process(
        self,
        input_data: Union[str, bytes, Path, List[Chunk], ChunkingResult],
        **kwargs
    ) -> PipelineResult:
        """
        Execute the full pipeline on input data.

        Args:
            input_data: Input to process through pipeline
            **kwargs: Additional arguments passed to steps

        Returns:
            PipelineResult with final chunks and execution metadata
        """
        if not self.steps:
            raise ValueError("Pipeline has no steps")

        start_time = time.time()
        current_data = input_data
        step_times = {}
        step_results = {}
        errors = []
        warnings = []

        self.logger.info(f"Starting pipeline '{self.name}' with {len(self.steps)} steps")

        for step_name, step in self.steps:
            step_start = time.time()

            try:
                self.logger.debug(f"Executing step: {step_name}")
                step.validate_input(current_data)
                result = step.process(current_data, **kwargs)
                current_data = result

                step_time = time.time() - step_start
                step_times[step_name] = step_time
                step_results[step_name] = result

                self.logger.debug(f"Step '{step_name}' completed in {step_time:.3f}s")

            except Exception as e:
                error_msg = f"Error in step '{step_name}': {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)

                # Optionally continue with previous data or stop
                step_times[step_name] = time.time() - step_start
                continue  # Continue with previous data

        total_time = time.time() - start_time

        # Ensure we have chunks at the end
        if isinstance(current_data, ChunkingResult):
            final_chunks = current_data.chunks
        elif isinstance(current_data, list):
            final_chunks = current_data
        else:
            # Create a single chunk from remaining data
            from chunking_strategy.core.base import ChunkMetadata
            metadata = ChunkMetadata(source="pipeline_output")
            final_chunks = [Chunk(
                id="pipeline_final",
                content=current_data,
                modality=ModalityType.TEXT,
                metadata=metadata
            )]

        self.logger.info(
            f"Pipeline completed in {total_time:.3f}s, "
            f"produced {len(final_chunks)} chunks"
        )

        return PipelineResult(
            chunks=final_chunks,
            total_processing_time=total_time,
            step_times=step_times,
            step_results=step_results,
            errors=errors,
            warnings=warnings
        )

    def export_config(self) -> Dict[str, Any]:
        """
        Export pipeline configuration as dictionary.

        Returns:
            Dictionary representation of pipeline configuration
        """
        step_configs = []
        for name, step in self.steps:
            step_config = {
                "name": name,
                "type": step.__class__.__name__,
                "config": step.get_config()
            }

            # Add type-specific information
            if isinstance(step, ChunkerStep):
                step_config["strategy"] = step.strategy_name

            step_configs.append(step_config)

        return {
            "name": self.name,
            "steps": step_configs
        }

    def save_config(self, file_path: Union[str, Path]) -> None:
        """
        Save pipeline configuration to JSON file.

        Args:
            file_path: Path to save configuration
        """
        config = self.export_config()
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Pipeline configuration saved to {file_path}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ChunkingPipeline":
        """
        Create pipeline from configuration dictionary.

        Args:
            config: Pipeline configuration

        Returns:
            Configured pipeline instance
        """
        name = config.get("name", "loaded_pipeline")
        pipeline = cls(name=name)

        for step_config in config.get("steps", []):
            step_name = step_config["name"]
            step_type = step_config["type"]
            step_params = step_config.get("config", {})

            # Create step based on type
            if step_type == "ChunkerStep":
                strategy = step_config.get("strategy", "fixed_size")
                step = ChunkerStep(step_name, strategy, **step_params)

            elif step_type == "FilterStep":
                step = FilterStep(step_name, **step_params)

            elif step_type == "TransformStep":
                # Note: Custom transform functions can't be serialized easily
                # This would need extension for complex transforms
                if "transform_func" in step_params:
                    # Would need custom deserialization logic
                    pass
                step = TransformStep(step_name, lambda x: x, **step_params)

            elif step_type == "MergeStep":
                step = MergeStep(step_name, **step_params)

            else:
                logger.warning(f"Unknown step type: {step_type}, skipping")
                continue

            pipeline.add_step(step_name, step)

        return pipeline

    @classmethod
    def load_config(cls, file_path: Union[str, Path]) -> "ChunkingPipeline":
        """
        Load pipeline from JSON configuration file.

        Args:
            file_path: Path to configuration file

        Returns:
            Loaded pipeline instance
        """
        with open(file_path, 'r') as f:
            config = json.load(f)

        return cls.from_config(config)

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """Get a step by name."""
        for step_name, step in self.steps:
            if step_name == name:
                return step
        return None

    def validate(self) -> List[str]:
        """
        Validate pipeline configuration.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if not self.steps:
            issues.append("Pipeline has no steps")

        step_names = [name for name, _ in self.steps]
        if len(step_names) != len(set(step_names)):
            issues.append("Duplicate step names found")

        # Check step compatibility
        for i, (name, step) in enumerate(self.steps):
            if i == 0:
                # First step should accept raw input
                if isinstance(step, FilterStep):
                    issues.append(f"First step '{name}' is a filter, needs chunker first")

        return issues

    def __repr__(self) -> str:
        step_names = [name for name, _ in self.steps]
        return f"ChunkingPipeline(name='{self.name}', steps={step_names})"
