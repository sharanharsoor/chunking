"""
Multi-Level Content-Defined Chunking (ML-CDC) Implementation.

This module implements ML-CDC, a hierarchical chunking approach that uses
multiple levels of boundary detection for improved performance and features.
"""

import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
from dataclasses import dataclass
from enum import Enum

from chunking_strategy.core.base import (
    Chunk,
    ChunkingResult,
    ModalityType,
    ChunkMetadata
)
from chunking_strategy.core.adaptive import AdaptableChunker
from chunking_strategy.core.streaming import StreamableChunker
from chunking_strategy.core.registry import register_chunker


logger = logging.getLogger(__name__)


class MLCDCLevel(Enum):
    """ML-CDC hierarchy levels."""
    MICRO = "micro"       # Smallest granularity
    SMALL = "small"       # Small chunks
    MEDIUM = "medium"     # Medium chunks
    LARGE = "large"       # Large chunks
    SUPER = "super"       # Largest granularity


@dataclass
class MLCDCLevelConfig:
    """Configuration for a single ML-CDC level."""
    level: MLCDCLevel
    min_size: int
    max_size: int
    target_size: int
    boundary_mask: int
    hash_function: str = "sha1"  # sha1, md5, polynomial

    def __post_init__(self):
        """Validate level configuration."""
        if self.min_size <= 0:
            raise ValueError("min_size must be positive")
        if self.max_size <= self.min_size:
            raise ValueError("max_size must be greater than min_size")
        if self.target_size < self.min_size or self.target_size > self.max_size:
            raise ValueError("target_size must be between min and max sizes")


@dataclass
class MLCDCConfig:
    """Configuration for ML-CDC chunking."""
    window_size: int = 48
    enable_statistics: bool = True
    adaptive_threshold: bool = True
    deduplication_mode: bool = True
    compression_aware: bool = False

    # Default level configurations
    levels: List[MLCDCLevelConfig] = None

    def __post_init__(self):
        """Initialize default levels if not provided."""
        if self.levels is None:
            self.levels = [
                MLCDCLevelConfig(
                    level=MLCDCLevel.MICRO,
                    min_size=512,        # 512B
                    max_size=2048,       # 2KB
                    target_size=1024,    # 1KB
                    boundary_mask=0x3FF,  # 10-bit mask
                    hash_function="polynomial"
                ),
                MLCDCLevelConfig(
                    level=MLCDCLevel.SMALL,
                    min_size=2048,       # 2KB
                    max_size=8192,       # 8KB
                    target_size=4096,    # 4KB
                    boundary_mask=0x7FF,  # 11-bit mask
                    hash_function="polynomial"
                ),
                MLCDCLevelConfig(
                    level=MLCDCLevel.MEDIUM,
                    min_size=8192,       # 8KB
                    max_size=32768,      # 32KB
                    target_size=16384,   # 16KB
                    boundary_mask=0xFFF,  # 12-bit mask
                    hash_function="sha1"
                ),
                MLCDCLevelConfig(
                    level=MLCDCLevel.LARGE,
                    min_size=32768,      # 32KB
                    max_size=131072,     # 128KB
                    target_size=65536,   # 64KB
                    boundary_mask=0x1FFF, # 13-bit mask
                    hash_function="sha1"
                ),
                MLCDCLevelConfig(
                    level=MLCDCLevel.SUPER,
                    min_size=131072,     # 128KB
                    max_size=1048576,    # 1MB
                    target_size=524288,  # 512KB
                    boundary_mask=0x3FFF, # 14-bit mask
                    hash_function="sha1"
                )
            ]

        if self.window_size <= 0:
            raise ValueError("window_size must be positive")


class MLCDCHasher:
    """Multi-level hasher for ML-CDC algorithm."""

    def __init__(self, config: MLCDCConfig):
        self.config = config
        self.levels = {level.level: level for level in config.levels}

        # Initialize hash states for each level
        self.hash_states = {}
        self.windows = {}

        for level_config in config.levels:
            self.hash_states[level_config.level] = 0
            self.windows[level_config.level] = []

    def reset(self):
        """Reset all hash states."""
        for level in self.hash_states:
            self.hash_states[level] = 0
            self.windows[level] = []

    def roll_byte(self, byte_in: int) -> Dict[MLCDCLevel, int]:
        """
        Roll in a new byte and compute hashes for all levels.

        Args:
            byte_in: New byte to process

        Returns:
            Dictionary of hash values for each level
        """
        hash_values = {}

        for level, level_config in self.levels.items():
            # Update window for this level
            self.windows[level].append(byte_in)

            if len(self.windows[level]) > self.config.window_size:
                byte_out = self.windows[level].pop(0)
                # Update hash with rolling calculation
                self.hash_states[level] = self._update_hash(
                    level_config, self.hash_states[level], byte_in, byte_out
                )
            else:
                # Just add new byte
                self.hash_states[level] = self._update_hash(
                    level_config, self.hash_states[level], byte_in
                )

            hash_values[level] = self.hash_states[level]

        return hash_values

    def _update_hash(self, level_config: MLCDCLevelConfig, current_hash: int,
                    byte_in: int, byte_out: Optional[int] = None) -> int:
        """Update hash for a specific level."""
        if level_config.hash_function == "polynomial":
            # Simple polynomial rolling hash
            if byte_out is not None:
                # Remove old byte and add new byte
                current_hash = ((current_hash - byte_out * 256**(self.config.window_size-1)) * 256 + byte_in) % (10**9 + 7)
            else:
                # Just add new byte
                current_hash = (current_hash * 256 + byte_in) % (10**9 + 7)

        elif level_config.hash_function in ["sha1", "md5"]:
            # Cryptographic hash (not truly rolling, but computed over window)
            window_bytes = bytes(self.windows[level_config.level])
            if level_config.hash_function == "sha1":
                hash_obj = hashlib.sha1(window_bytes)
            else:
                hash_obj = hashlib.md5(window_bytes)

            # Convert to integer
            current_hash = int.from_bytes(hash_obj.digest()[:4], 'big')

        return current_hash

    def is_boundary(self, level: MLCDCLevel, hash_value: int) -> bool:
        """Check if hash indicates boundary for given level."""
        level_config = self.levels[level]
        return (hash_value & level_config.boundary_mask) == 0

    def get_hash(self, level: MLCDCLevel) -> int:
        """Get current hash for specific level."""
        return self.hash_states[level]


@register_chunker("ml_cdc")
class MLCDCChunker(StreamableChunker, AdaptableChunker):
    """
    Multi-Level Content-Defined Chunker (ML-CDC).

    This chunker implements a hierarchical approach to content-defined chunking
    using multiple levels of boundary detection. Each level operates at different
    granularities, providing flexibility for various use cases.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], MLCDCConfig]] = None,
        **kwargs
    ):
        """
        Initialize ML-CDC chunker.

        Args:
            config: Configuration for the chunker
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="ml_cdc",
            category="general",
            supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
            **kwargs
        )

        # Parse configuration
        if isinstance(config, MLCDCConfig):
            self.config = config
        else:
            config = config or {}
            config.update(kwargs)

            # Extract level configs if provided
            level_configs = config.pop('levels', None)

            self.config = MLCDCConfig(**{
                k: v for k, v in config.items()
                if k in MLCDCConfig.__dataclass_fields__
            })

            if level_configs:
                # Convert dictionary level configs to MLCDCLevelConfig objects
                converted_levels = []
                for level_dict in level_configs:
                    if isinstance(level_dict, dict):
                        # Convert level string to enum
                        level_value = level_dict.get('level', 'medium')
                        level_enum = MLCDCLevel(level_value) if isinstance(level_value, str) else level_value

                        converted_levels.append(MLCDCLevelConfig(
                            level=level_enum,
                            min_size=level_dict.get('min_size', 2048),
                            max_size=level_dict.get('max_size', 65536),
                            target_size=level_dict.get('target_size', 8192),
                            boundary_mask=level_dict.get('boundary_mask', 0x1FFF),
                            hash_function=level_dict.get('hash_function', 'sha1')
                        ))
                    else:
                        # Already a MLCDCLevelConfig object
                        converted_levels.append(level_dict)

                self.config.levels = converted_levels

        # Initialize hasher
        self.hasher = MLCDCHasher(self.config)

        # Statistics
        self.stats = {
            "chunks_created": 0,
            "bytes_processed": 0,
            "level_statistics": {level.level.value: {
                "boundaries_found": 0,
                "avg_chunk_size": 0,
                "chunks_created": 0
            } for level in self.config.levels},
            "hierarchy_usage": {},
            "deduplication_ratio": 0
        } if self.config.enable_statistics else None

        # Adaptation history
        self._adaptation_history = []

        # Current active level (can be adapted)
        self._active_level = MLCDCLevel.MEDIUM  # Default to medium level
        self._level_explicitly_set = False  # Track if level was explicitly set

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ML-CDC chunker initialized with {len(self.config.levels)} levels")

    @property
    def active_level(self) -> MLCDCLevel:
        """Get the current active level."""
        return self._active_level

    @active_level.setter
    def active_level(self, value: MLCDCLevel):
        """Set the active level explicitly."""
        self._active_level = value
        self._level_explicitly_set = True

    def chunk(
        self,
        content: Union[str, bytes],
        source_info: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Chunk content using ML-CDC algorithm.

        Args:
            content: Content to chunk
            source_info: Optional source information

        Returns:
            Chunking result with generated chunks
        """
        start_time = time.time()

        # Handle different input types
        from pathlib import Path
        if isinstance(content, Path):
            with open(content, 'rb') as f:
                content_bytes = f.read()
            # Try to decode as text to determine if it's text content
            try:
                content_bytes.decode('utf-8')
                is_text = True
            except UnicodeDecodeError:
                is_text = False
        elif isinstance(content, str):
            content_bytes = content.encode('utf-8')
            is_text = True
        else:
            content_bytes = content
            is_text = False

        if len(content_bytes) == 0:
            return ChunkingResult(
                chunks=[],
                strategy_used="ml_cdc",
                processing_time=time.time() - start_time,
                source_info=source_info or {}
            )

        # Determine optimal level based on content size and characteristics
        # Use active_level if it's been explicitly set, otherwise auto-select
        if self._level_explicitly_set:
            optimal_level = self.active_level
        else:
            optimal_level = self._select_optimal_level(content_bytes, source_info or {})

        # Perform ML-CDC chunking
        chunks = list(self._ml_cdc_chunk(content_bytes, is_text, optimal_level))

        processing_time = time.time() - start_time

        # Update statistics
        if self.stats:
            self.stats["chunks_created"] += len(chunks)
            self.stats["bytes_processed"] += len(content_bytes)

            level_stats = self.stats["level_statistics"][optimal_level.value]
            level_stats["chunks_created"] += len(chunks)
            if chunks:
                total_size = sum(chunk.metadata.extra.get("size", 0) for chunk in chunks)
                level_stats["avg_chunk_size"] = total_size / len(chunks)

        return ChunkingResult(
            chunks=chunks,
            strategy_used="ml_cdc",
            processing_time=processing_time,
            source_info=source_info or {}
        )

    def _select_optimal_level(self, content: bytes, source_info: Dict[str, Any]) -> MLCDCLevel:
        """
        Select optimal chunking level based on content characteristics.
        Only returns levels that are actually configured in the chunker.

        Args:
            content: Content to analyze
            source_info: Additional information about the content

        Returns:
            Optimal ML-CDC level from configured levels
        """
        content_size = len(content)

        # Get list of configured levels
        configured_levels = [lc.level for lc in self.config.levels]

        # Preference order based on content size
        size_preferences = []
        if content_size < 4096:      # < 4KB
            size_preferences = [MLCDCLevel.MICRO, MLCDCLevel.SMALL, MLCDCLevel.MEDIUM, MLCDCLevel.LARGE, MLCDCLevel.SUPER]
        elif content_size < 32768:   # < 32KB
            size_preferences = [MLCDCLevel.SMALL, MLCDCLevel.MICRO, MLCDCLevel.MEDIUM, MLCDCLevel.LARGE, MLCDCLevel.SUPER]
        elif content_size < 262144:  # < 256KB
            size_preferences = [MLCDCLevel.MEDIUM, MLCDCLevel.SMALL, MLCDCLevel.LARGE, MLCDCLevel.MICRO, MLCDCLevel.SUPER]
        elif content_size < 2097152: # < 2MB
            size_preferences = [MLCDCLevel.LARGE, MLCDCLevel.MEDIUM, MLCDCLevel.SUPER, MLCDCLevel.SMALL, MLCDCLevel.MICRO]
        else:
            size_preferences = [MLCDCLevel.SUPER, MLCDCLevel.LARGE, MLCDCLevel.MEDIUM, MLCDCLevel.SMALL, MLCDCLevel.MICRO]

        # Return the first preference that is actually configured
        for preferred_level in size_preferences:
            if preferred_level in configured_levels:
                return preferred_level

        # Fallback: return the first configured level if no preference matches
        return configured_levels[0]

    def _ml_cdc_chunk(self, content: bytes, is_text: bool, level: MLCDCLevel) -> Iterator[Chunk]:
        """
        Perform ML-CDC chunking at specified level.

        Args:
            content: Content bytes to chunk
            is_text: Whether content is text
            level: Chunking level to use

        Yields:
            Generated chunks
        """
        level_config = next(lc for lc in self.config.levels if lc.level == level)

        if len(content) <= level_config.min_size:
            # Content too small, return as single chunk
            chunk_content = content.decode('utf-8', errors='ignore') if is_text else content
            yield Chunk(
                id=f"ml_cdc_0",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=0,
                    length=len(content),
                    chunker_used="ml_cdc",
                    extra={
                        "chunk_index": 0,
                        "start_offset": 0,
                        "end_offset": len(content),
                        "algorithm": "ml_cdc",
                        "level": level.value,
                        "size": len(content),
                        "boundary_mask": hex(level_config.boundary_mask),
                        "hash_function": level_config.hash_function
                    }
                )
            )
            return

        # Initialize hasher
        self.hasher.reset()

        chunk_start = 0
        chunk_index = 0

        # Process each byte
        for position in range(len(content)):
            byte_val = content[position]

            # Compute hashes for all levels
            hash_values = self.hasher.roll_byte(byte_val)

            # Check boundary for current level
            chunk_size = position - chunk_start + 1

            is_boundary = (self.hasher.is_boundary(level, hash_values[level]) and
                          chunk_size >= level_config.min_size)
            is_max_size = chunk_size >= level_config.max_size

            if is_boundary or is_max_size:
                # Create chunk
                chunk_content = content[chunk_start:position + 1]
                if is_text:
                    chunk_content = chunk_content.decode('utf-8', errors='ignore')

                # Generate hierarchical fingerprint using multiple levels
                hierarchical_fingerprint = self._generate_hierarchical_fingerprint(
                    content[chunk_start:position + 1], hash_values
                )

                yield Chunk(
                    id=f"ml_cdc_{chunk_index}",
                    content=chunk_content,
                    modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                    metadata=ChunkMetadata(
                        source="unknown",
                        offset=chunk_start,
                        length=position - chunk_start + 1,
                        chunker_used="ml_cdc",
                        extra={
                            "chunk_index": chunk_index,
                            "start_offset": chunk_start,
                            "end_offset": position + 1,
                            "algorithm": "ml_cdc",
                            "level": level.value,
                            "size": chunk_size,
                            "boundary_type": "hash" if is_boundary else "size",
                            "hash_values": {lv.value: hv for lv, hv in hash_values.items()},
                            "hierarchical_fingerprint": hierarchical_fingerprint,
                            "boundary_mask": hex(level_config.boundary_mask),
                            "hash_function": level_config.hash_function
                        }
                    )
                )

                if self.stats:
                    if is_boundary:
                        self.stats["level_statistics"][level.value]["boundaries_found"] += 1

                chunk_start = position + 1
                chunk_index += 1
                self.hasher.reset()

        # Handle remaining content
        if chunk_start < len(content):
            chunk_content = content[chunk_start:]
            if is_text:
                chunk_content = chunk_content.decode('utf-8', errors='ignore')

            # Final hash values
            final_hash_values = {}
            for lv in self.hasher.hash_states:
                final_hash_values[lv] = self.hasher.get_hash(lv)

            hierarchical_fingerprint = self._generate_hierarchical_fingerprint(
                content[chunk_start:], final_hash_values
            )

            yield Chunk(
                id=f"ml_cdc_{chunk_index}",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=chunk_start,
                    length=len(content) - chunk_start,
                    chunker_used="ml_cdc",
                    extra={
                        "chunk_index": chunk_index,
                        "start_offset": chunk_start,
                        "end_offset": len(content),
                        "algorithm": "ml_cdc",
                        "level": level.value,
                        "size": len(content) - chunk_start,
                        "boundary_type": "end",
                        "hierarchical_fingerprint": hierarchical_fingerprint,
                        "hash_function": level_config.hash_function
                    }
                )
            )

    def _generate_hierarchical_fingerprint(self, content: bytes, hash_values: Dict[MLCDCLevel, int]) -> str:
        """Generate hierarchical fingerprint combining multiple levels."""
        # Combine hash values from different levels
        combined_hash = 0
        for level, hash_val in hash_values.items():
            combined_hash ^= hash_val

        # Create hierarchical fingerprint
        fingerprint = hashlib.sha256(content + combined_hash.to_bytes(8, 'big')).hexdigest()[:16]
        return fingerprint

    def supports_streaming(self) -> bool:
        """Check if chunker supports streaming."""
        return True

    def chunk_stream(
        self,
        stream: Iterator[Union[str, bytes]],
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Chunk data from a stream using ML-CDC.

        Args:
            stream: Iterator of content chunks
            **kwargs: Additional parameters

        Yields:
            Generated chunks
        """
        buffer = b""
        chunk_index = 0
        total_processed = 0

        # Use medium level for streaming by default
        stream_level = kwargs.get('level', MLCDCLevel.MEDIUM)
        level_config = next(lc for lc in self.config.levels if lc.level == stream_level)

        for data in stream:
            if isinstance(data, str):
                data = data.encode('utf-8')

            buffer += data

            # Process buffer when it's large enough
            if len(buffer) >= level_config.max_size * 2:
                # Find safe cut point
                safe_cut = len(buffer) - self.config.window_size

                if safe_cut > level_config.max_size:
                    # Chunk the buffer up to safe cut point
                    result = self.chunk(buffer[:safe_cut])

                    # Yield chunks
                    for chunk in result.chunks:
                        chunk.metadata.extra["chunk_index"] = chunk_index
                        chunk.metadata.extra["stream_offset"] = total_processed + chunk.metadata.extra["start_offset"]
                        yield chunk
                        chunk_index += 1

                    # Keep remaining buffer
                    total_processed += safe_cut
                    buffer = buffer[safe_cut:]

        # Process remaining buffer
        if buffer:
            result = self.chunk(buffer)
            for chunk in result.chunks:
                chunk.metadata.extra["chunk_index"] = chunk_index
                chunk.metadata.extra["stream_offset"] = total_processed + chunk.metadata.extra["start_offset"]
                yield chunk
                chunk_index += 1

    def adapt_parameters(self, feedback: float) -> bool:
        """
        Adapt chunker parameters based on feedback.

        Args:
            feedback: Performance feedback score (0.0-1.0)

        Returns:
            True if parameters were adapted
        """
        adapted = False
        adaptation_record = {
            "timestamp": time.time(),
            "feedback": feedback,
            "old_level": self.active_level.value,
            "new_level": None,
            "changes": []
        }

        # Simple adaptation based on feedback score
        if feedback < 0.5:  # Low score indicates need for adaptation
            # Try switching to a different level for better performance
            current_levels = list(MLCDCLevel)
            current_idx = current_levels.index(self.active_level)

            # Cycle to next level
            new_idx = (current_idx + 1) % len(current_levels)
            new_level = current_levels[new_idx]

            if new_level != self.active_level:
                adaptation_record["new_level"] = new_level.value
                adaptation_record["changes"].append("active_level")
                self._active_level = new_level  # Use internal field to avoid marking as explicitly set
                adapted = True

        if adapted:
            self._adaptation_history.append(adaptation_record)

        return adapted

    def _select_adaptive_level(self, avg_chunk_size: float, processing_time: float,
                              feedback: Dict[str, Any]) -> MLCDCLevel:
        """Select adaptive level based on performance feedback.
        Only returns levels that are actually configured in the chunker."""

        # Get list of configured levels
        configured_levels = [lc.level for lc in self.config.levels]

        # Performance-based level selection
        throughput = feedback.get("content_size", 0) / processing_time if processing_time > 0 else 0

        # Define preference orders based on performance characteristics
        performance_preferences = []

        if throughput < 1_000_000:  # < 1MB/s - use smaller level for speed
            if avg_chunk_size > 16384:  # Large chunks
                performance_preferences = [MLCDCLevel.SMALL, MLCDCLevel.MICRO, MLCDCLevel.MEDIUM, MLCDCLevel.LARGE, MLCDCLevel.SUPER]
            else:
                performance_preferences = [MLCDCLevel.MICRO, MLCDCLevel.SMALL, MLCDCLevel.MEDIUM, MLCDCLevel.LARGE, MLCDCLevel.SUPER]

        elif throughput > 10_000_000:  # > 10MB/s - can afford larger levels
            if avg_chunk_size < 32768:  # Small chunks
                performance_preferences = [MLCDCLevel.LARGE, MLCDCLevel.SUPER, MLCDCLevel.MEDIUM, MLCDCLevel.SMALL, MLCDCLevel.MICRO]
            else:
                performance_preferences = [MLCDCLevel.SUPER, MLCDCLevel.LARGE, MLCDCLevel.MEDIUM, MLCDCLevel.SMALL, MLCDCLevel.MICRO]

        else:  # Moderate performance
            performance_preferences = [MLCDCLevel.MEDIUM, MLCDCLevel.SMALL, MLCDCLevel.LARGE, MLCDCLevel.MICRO, MLCDCLevel.SUPER]

        # Return the first preference that is actually configured
        for preferred_level in performance_preferences:
            if preferred_level in configured_levels:
                return preferred_level

        # Fallback: return the first configured level if no preference matches
        return configured_levels[0]

    def get_adaptation_info(self) -> Dict[str, Any]:
        """Get information about current adaptations."""
        return {
            "active_level": self.active_level.value,
            "config": {
                "window_size": self.config.window_size,
                "levels": [{
                    "level": lc.level.value,
                    "min_size": lc.min_size,
                    "max_size": lc.max_size,
                    "target_size": lc.target_size,
                    "boundary_mask": hex(lc.boundary_mask),
                    "hash_function": lc.hash_function
                } for lc in self.config.levels],
                "adaptive_threshold": self.config.adaptive_threshold,
                "deduplication_mode": self.config.deduplication_mode
            },
            "statistics": self.stats,
            "adaptation_history": self.adaptation_history,
            "adaptation_count": len(self.adaptation_history)
        }

    def get_chunk_estimate(self, content_size: int) -> Tuple[int, int]:
        """Estimate number of chunks for given content size."""
        if content_size <= 1024:
            return (1, 1)

        # Use active level for estimation
        level_config = next(lc for lc in self.config.levels if lc.level == self.active_level)

        # Estimate based on target chunk size
        estimated = max(1, int(content_size / level_config.target_size))

        # Add variance for multi-level variability
        variance = max(1, int(estimated * 0.35))  # ±35% variance
        return (max(1, estimated - variance), estimated + variance)

    def get_quality_score(self, chunks: List[Chunk]) -> float:
        """Calculate quality score for generated chunks."""
        if not chunks:
            return 0.0

        # Multi-level size distribution quality
        chunk_sizes = [chunk.metadata.extra.get("size", len(chunk.content)) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)

        # Get target for active level
        level_config = next(lc for lc in self.config.levels if lc.level == self.active_level)
        target_ratio = avg_size / level_config.target_size
        size_score = max(0, 1 - abs(1 - target_ratio))

        # Hierarchical boundary quality
        hash_boundaries = sum(1 for chunk in chunks
                             if chunk.metadata.extra.get("boundary_type") == "hash")
        boundary_ratio = hash_boundaries / len(chunks) if chunks else 0
        boundary_score = boundary_ratio

        # Multi-level consistency
        levels_used = set(chunk.metadata.extra.get("level") for chunk in chunks)
        consistency_score = 1.0 if len(levels_used) == 1 else 0.8  # Slight penalty for mixed levels

        # Deduplication potential (based on hierarchical fingerprints)
        fingerprints = [chunk.metadata.extra.get("hierarchical_fingerprint") for chunk in chunks]
        unique_fingerprints = len(set(fp for fp in fingerprints if fp))
        dedup_score = unique_fingerprints / len(chunks) if chunks else 1.0

        # Combined quality score
        return (size_score * 0.3 + boundary_score * 0.25 +
                consistency_score * 0.2 + dedup_score * 0.25)

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self._adaptation_history.copy()

    def describe_algorithm(self) -> str:
        """Describe the ML-CDC algorithm."""
        levels_desc = "\n".join([
            f"    {lc.level.value.upper()}: {lc.min_size}-{lc.max_size} bytes (target: {lc.target_size})"
            for lc in self.config.levels
        ])

        return f"""
        Multi-Level Content-Defined Chunking (ML-CDC) Algorithm:

        Active Level: {self.active_level.value}
        Window Size: {self.config.window_size} bytes
        Levels Configured: {len(self.config.levels)}

        Level Hierarchy:
{levels_desc}

        ML-CDC uses a hierarchical approach to content-defined chunking,
        operating at multiple granularity levels simultaneously. Each level
        has its own boundary detection parameters and hash functions.

        Algorithm Features:
        1. Multi-level boundary detection
        2. Hierarchical fingerprinting for enhanced deduplication
        3. Adaptive level selection based on content characteristics
        4. Performance-aware optimization
        5. Configurable hash functions per level

        Advantages:
        - Flexible granularity control
        - Enhanced deduplication through hierarchy
        - Adaptive performance optimization
        - Multiple hash function support
        - Scalable for different content types

        Use Cases:
        - Advanced deduplication systems
        - Hierarchical storage management
        - Multi-tier caching systems
        - Research and development platforms
        - Large-scale content processing
        """
