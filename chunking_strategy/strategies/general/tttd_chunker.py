"""
TTTD (Two-Threshold Two-Divisor) Chunking Implementation.

This module implements TTTD chunking, which uses two thresholds and two divisors
for enhanced content-defined chunking with better size distribution control.
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


class TTTDThresholdType(Enum):
    """TTTD threshold types."""
    PRIMARY = "primary"       # Primary threshold (smaller)
    SECONDARY = "secondary"   # Secondary threshold (larger)


@dataclass
class TTTDConfig:
    """Configuration for TTTD chunking."""
    window_size: int = 48              # Rolling hash window size
    min_chunk_size: int = 2048         # 2KB minimum
    max_chunk_size: int = 65536        # 64KB maximum
    target_chunk_size: int = 8192      # 8KB target

    # Primary threshold (for smaller chunks)
    primary_divisor: int = 1024        # Primary divisor
    primary_threshold: int = 0         # Primary threshold value

    # Secondary threshold (for larger chunks)
    secondary_divisor: int = 4096      # Secondary divisor
    secondary_threshold: int = 0       # Secondary threshold value

    # Hash parameters
    hash_polynomial: int = 0x3DA3358B4DC173  # Polynomial for Rabin hash
    normalization: int = 2             # Normalization factor

    # Algorithm parameters
    enable_adaptive_thresholds: bool = True
    threshold_adjustment_rate: float = 0.1
    enable_statistics: bool = True

    def __post_init__(self):
        """Validate and initialize configuration."""
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.max_chunk_size <= self.min_chunk_size:
            raise ValueError("max_chunk_size must be greater than min_chunk_size")
        if self.target_chunk_size < self.min_chunk_size or self.target_chunk_size > self.max_chunk_size:
            raise ValueError("target_chunk_size must be between min and max chunk sizes")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.primary_divisor <= 0 or self.secondary_divisor <= 0:
            raise ValueError("divisors must be positive")
        if self.secondary_divisor <= self.primary_divisor:
            raise ValueError("secondary_divisor must be greater than primary_divisor")


class TTTDHasher:
    """
    TTTD hasher implementing two-threshold two-divisor algorithm.

    This hasher computes rolling hash and evaluates both primary and
    secondary thresholds for boundary detection.
    """

    def __init__(self, config: TTTDConfig):
        self.config = config

        # Initialize hash computation
        self.polynomial = config.hash_polynomial
        self.polynomial_degree = 53  # Degree of the polynomial

        # Build precomputation tables
        self.mod_table = self._build_mod_table()
        self.pow_table = self._build_pow_table()

        # Reset state
        self.reset()

    def _build_mod_table(self) -> List[int]:
        """Build modulo table for fast computation."""
        table = [0] * 256

        for i in range(256):
            value = i << (self.polynomial_degree - 8)

            for _ in range(8):
                if value & (1 << (self.polynomial_degree - 1)):
                    value = (value << 1) ^ self.polynomial
                else:
                    value <<= 1
                value &= (1 << self.polynomial_degree) - 1

            table[i] = value

        return table

    def _build_pow_table(self) -> List[int]:
        """Build power table for rolling hash."""
        table = [1]
        base = 256

        for i in range(1, self.config.window_size + 1):
            power = (table[-1] * base) % ((1 << self.polynomial_degree) - 1)
            if power >= (1 << (self.polynomial_degree - 1)):
                power ^= self.polynomial
                power &= (1 << self.polynomial_degree) - 1
            table.append(power)

        return table

    def reset(self):
        """Reset hasher state."""
        self.hash_value = 0
        self.window = []

    def roll_byte(self, byte_in: int) -> int:
        """
        Roll in a new byte and compute hash.

        Args:
            byte_in: New byte to process

        Returns:
            Current hash value
        """
        self.window.append(byte_in)

        if len(self.window) > self.config.window_size:
            # Remove oldest byte
            byte_out = self.window.pop(0)

            # Remove old byte contribution
            old_contribution = (byte_out * self.pow_table[self.config.window_size - 1]) % ((1 << self.polynomial_degree) - 1)
            self.hash_value = (self.hash_value - old_contribution) % ((1 << self.polynomial_degree) - 1)

        # Add new byte
        self.hash_value = (self.hash_value * 256 + byte_in) % ((1 << self.polynomial_degree) - 1)

        # Apply polynomial reduction
        if self.hash_value >= (1 << (self.polynomial_degree - 1)):
            self.hash_value ^= self.polynomial
            self.hash_value &= (1 << self.polynomial_degree) - 1

        return self.hash_value

    def get_hash(self) -> int:
        """Get current hash value."""
        return self.hash_value

    def check_thresholds(self, hash_value: int) -> Tuple[bool, bool]:
        """
        Check both primary and secondary thresholds.

        Args:
            hash_value: Current hash value

        Returns:
            Tuple of (primary_threshold_met, secondary_threshold_met)
        """
        # Primary threshold check (more frequent boundaries)
        primary_remainder = hash_value % self.config.primary_divisor
        primary_met = primary_remainder == self.config.primary_threshold

        # Secondary threshold check (less frequent boundaries)
        secondary_remainder = hash_value % self.config.secondary_divisor
        secondary_met = secondary_remainder == self.config.secondary_threshold

        return primary_met, secondary_met


@register_chunker("tttd")
class TTTDChunker(StreamableChunker, AdaptableChunker):
    """
    TTTD (Two-Threshold Two-Divisor) Chunker.

    This chunker implements the TTTD algorithm which uses two different
    threshold conditions for boundary detection, providing better control
    over chunk size distribution.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], TTTDConfig]] = None,
        **kwargs
    ):
        """
        Initialize TTTD chunker.

        Args:
            config: Configuration for the chunker
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="tttd",
            category="general",
            supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
            **kwargs
        )

        # Parse configuration
        if isinstance(config, TTTDConfig):
            self.config = config
        else:
            config = config or {}
            config.update(kwargs)
            self.config = TTTDConfig(**{
                k: v for k, v in config.items()
                if k in TTTDConfig.__dataclass_fields__
            })

        # Initialize hasher
        self.hasher = TTTDHasher(self.config)

        # Statistics
        self.stats = {
            "chunks_created": 0,
            "bytes_processed": 0,
            "hash_computations": 0,
            "primary_boundaries": 0,
            "secondary_boundaries": 0,
            "size_boundaries": 0,
            "avg_chunk_size": 0,
            "threshold_hit_rates": {
                "primary": 0.0,
                "secondary": 0.0
            }
        } if self.config.enable_statistics else None

        # Adaptation history
        self.adaptation_history = []

        # Adaptive threshold tracking
        self.chunk_size_history = []
        self.adaptation_counter = 0

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TTTD chunker initialized with primary divisor {self.config.primary_divisor}, "
                        f"secondary divisor {self.config.secondary_divisor}")

    def chunk(
        self,
        content: Union[str, bytes],
        source_info: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Chunk content using TTTD algorithm.

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
                strategy_used="tttd",
                processing_time=time.time() - start_time,
                source_info=source_info or {}
            )

        # Perform TTTD chunking
        chunks = list(self._tttd_chunk(content_bytes, is_text))

        processing_time = time.time() - start_time

        # Update statistics
        if self.stats:
            self.stats["chunks_created"] += len(chunks)
            self.stats["bytes_processed"] += len(content_bytes)
            if chunks:
                total_size = sum(chunk.metadata.extra.get("size", 0) for chunk in chunks)
                self.stats["avg_chunk_size"] = total_size / len(chunks)

                # Calculate threshold hit rates
                if self.stats["hash_computations"] > 0:
                    self.stats["threshold_hit_rates"]["primary"] = \
                        self.stats["primary_boundaries"] / self.stats["hash_computations"]
                    self.stats["threshold_hit_rates"]["secondary"] = \
                        self.stats["secondary_boundaries"] / self.stats["hash_computations"]

        # Perform adaptive threshold adjustment if enabled
        if self.config.enable_adaptive_thresholds and chunks:
            self._adapt_thresholds(chunks)

        return ChunkingResult(
            chunks=chunks,
            strategy_used="tttd",
            processing_time=processing_time,
            source_info=source_info or {}
        )

    def _tttd_chunk(self, content: bytes, is_text: bool) -> Iterator[Chunk]:
        """
        Perform TTTD chunking on content.

        Args:
            content: Content bytes to chunk
            is_text: Whether content is text

        Yields:
            Generated chunks
        """
        if len(content) <= self.config.min_chunk_size:
            # Content too small, return as single chunk
            chunk_content = content.decode('utf-8', errors='ignore') if is_text else content
            yield Chunk(
                id=f"tttd_0",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=0,
                    length=len(content),
                    chunker_used="tttd",
                    extra={
                        "chunk_index": 0,
                        "start_offset": 0,
                        "end_offset": len(content),
                        "algorithm": "tttd",
                        "size": len(content),
                        "boundary_type": "single",
                        "primary_divisor": self.config.primary_divisor,
                        "secondary_divisor": self.config.secondary_divisor
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

            # Compute hash
            hash_value = self.hasher.roll_byte(byte_val)

            if self.stats:
                self.stats["hash_computations"] += 1

            # Check thresholds
            primary_met, secondary_met = self.hasher.check_thresholds(hash_value)

            # Track threshold hits
            if self.stats:
                if primary_met:
                    self.stats["primary_boundaries"] += 1
                if secondary_met:
                    self.stats["secondary_boundaries"] += 1

            # Determine chunk boundary
            chunk_size = position - chunk_start + 1

            # TTTD boundary logic
            is_boundary = False
            boundary_type = None

            if chunk_size >= self.config.max_chunk_size:
                # Force boundary at max size
                is_boundary = True
                boundary_type = "size"
                if self.stats:
                    self.stats["size_boundaries"] += 1

            elif chunk_size >= self.config.min_chunk_size:
                # Check threshold conditions
                if secondary_met:
                    # Secondary threshold has priority (creates larger chunks)
                    is_boundary = True
                    boundary_type = "secondary"
                elif primary_met and chunk_size >= self.config.target_chunk_size // 2:
                    # Primary threshold (creates smaller chunks) with size condition
                    is_boundary = True
                    boundary_type = "primary"

            if is_boundary:
                # Create chunk
                chunk_content = content[chunk_start:position + 1]
                if is_text:
                    chunk_content = chunk_content.decode('utf-8', errors='ignore')

                yield Chunk(
                    id=f"tttd_{chunk_index}",
                    content=chunk_content,
                    modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                    metadata=ChunkMetadata(
                        source="unknown",
                        offset=chunk_start,
                        length=position - chunk_start + 1,
                        chunker_used="tttd",
                        extra={
                            "chunk_index": chunk_index,
                            "start_offset": chunk_start,
                            "end_offset": position + 1,
                            "algorithm": "tttd",
                            "hash_value": hash_value,
                            "size": chunk_size,
                            "boundary_type": boundary_type,
                            "primary_threshold_met": primary_met,
                            "secondary_threshold_met": secondary_met,
                            "primary_divisor": self.config.primary_divisor,
                            "secondary_divisor": self.config.secondary_divisor,
                            "primary_threshold": self.config.primary_threshold,
                            "secondary_threshold": self.config.secondary_threshold
                        }
                    )
                )

                chunk_start = position + 1
                chunk_index += 1
                self.hasher.reset()

        # Handle remaining content
        if chunk_start < len(content):
            chunk_content = content[chunk_start:]
            if is_text:
                chunk_content = chunk_content.decode('utf-8', errors='ignore')

            yield Chunk(
                id=f"tttd_{chunk_index}",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=chunk_start,
                    length=len(content) - chunk_start,
                    chunker_used="tttd",
                    extra={
                        "chunk_index": chunk_index,
                        "start_offset": chunk_start,
                        "end_offset": len(content),
                        "algorithm": "tttd",
                        "size": len(content) - chunk_start,
                        "boundary_type": "end",
                        "primary_divisor": self.config.primary_divisor,
                        "secondary_divisor": self.config.secondary_divisor
                    }
                )
            )

    def _adapt_thresholds(self, chunks: List[Chunk]):
        """
        Adapt thresholds based on chunk size distribution.

        Args:
            chunks: Generated chunks for analysis
        """
        # Collect chunk sizes
        chunk_sizes = [chunk.metadata.extra.get("size", 0) for chunk in chunks]
        self.chunk_size_history.extend(chunk_sizes)

        # Keep only recent history
        if len(self.chunk_size_history) > 1000:
            self.chunk_size_history = self.chunk_size_history[-1000:]

        self.adaptation_counter += 1

        # Adapt every 10 chunking operations
        if self.adaptation_counter % 10 == 0 and len(self.chunk_size_history) >= 20:
            avg_size = sum(self.chunk_size_history) / len(self.chunk_size_history)
            target = self.config.target_chunk_size

            # Adjust divisors based on average chunk size
            if abs(avg_size - target) > target * 0.2:  # 20% tolerance
                old_primary = self.config.primary_divisor
                old_secondary = self.config.secondary_divisor

                if avg_size > target:
                    # Chunks too large - increase boundary probability
                    self.config.primary_divisor = max(512, int(self.config.primary_divisor * 0.9))
                    self.config.secondary_divisor = max(1024, int(self.config.secondary_divisor * 0.9))
                else:
                    # Chunks too small - decrease boundary probability
                    self.config.primary_divisor = min(4096, int(self.config.primary_divisor * 1.1))
                    self.config.secondary_divisor = min(16384, int(self.config.secondary_divisor * 1.1))

                if (self.config.primary_divisor != old_primary or
                    self.config.secondary_divisor != old_secondary):

                    self.logger.info(f"Adapted TTTD divisors: primary {old_primary}->{self.config.primary_divisor}, "
                                   f"secondary {old_secondary}->{self.config.secondary_divisor}")

                    # Record adaptation
                    self.adaptation_history.append({
                        "timestamp": time.time(),
                        "old_primary_divisor": old_primary,
                        "new_primary_divisor": self.config.primary_divisor,
                        "old_secondary_divisor": old_secondary,
                        "new_secondary_divisor": self.config.secondary_divisor,
                        "avg_chunk_size": avg_size,
                        "target_size": target
                    })

                    # Recreate hasher with new config
                    self.hasher = TTTDHasher(self.config)

    def supports_streaming(self) -> bool:
        """Check if chunker supports streaming."""
        return True

    def chunk_stream(
        self,
        stream: Iterator[Union[str, bytes]],
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Chunk data from a stream using TTTD.

        Args:
            stream: Iterator of content chunks
            **kwargs: Additional parameters

        Yields:
            Generated chunks
        """
        buffer = b""
        chunk_index = 0
        total_processed = 0

        for data in stream:
            if isinstance(data, str):
                data = data.encode('utf-8')

            buffer += data

            # Process buffer when it's large enough
            if len(buffer) >= self.config.max_chunk_size * 2:
                # Find safe cut point
                safe_cut = len(buffer) - self.config.window_size

                if safe_cut > self.config.max_chunk_size:
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

    def adapt_parameters(self, feedback: Dict[str, Any]) -> bool:
        """
        Adapt chunker parameters based on feedback.

        Args:
            feedback: Performance feedback

        Returns:
            True if parameters were adapted
        """
        adapted = False
        adaptation_record = {
            "timestamp": time.time(),
            "feedback": feedback,
            "old_config": {},
            "new_config": {},
            "changes": []
        }

        # Adapt divisors based on chunk size feedback
        if "avg_chunk_size" in feedback:
            current_avg = feedback["avg_chunk_size"]
            target = self.config.target_chunk_size

            if abs(current_avg - target) > target * 0.15:  # 15% tolerance
                old_primary = self.config.primary_divisor
                old_secondary = self.config.secondary_divisor

                # Calculate adjustment factor
                size_ratio = current_avg / target
                adjustment = 1.0 / size_ratio  # Inverse relationship

                # Apply adjustment with limits
                new_primary = max(256, min(8192, int(self.config.primary_divisor * adjustment)))
                new_secondary = max(1024, min(32768, int(self.config.secondary_divisor * adjustment)))

                if new_primary != old_primary or new_secondary != old_secondary:
                    adaptation_record["old_config"] = {
                        "primary_divisor": old_primary,
                        "secondary_divisor": old_secondary
                    }

                    self.config.primary_divisor = new_primary
                    self.config.secondary_divisor = new_secondary

                    adaptation_record["new_config"] = {
                        "primary_divisor": new_primary,
                        "secondary_divisor": new_secondary
                    }
                    adaptation_record["changes"] = ["primary_divisor", "secondary_divisor"]
                    adapted = True

                    self.logger.info(f"Adapted TTTD divisors for target size {target}: "
                                   f"primary {old_primary}->{new_primary}, secondary {old_secondary}->{new_secondary}")

        # Adapt thresholds based on boundary hit rates
        if "primary_hit_rate" in feedback and "secondary_hit_rate" in feedback:
            primary_rate = feedback["primary_hit_rate"]
            secondary_rate = feedback["secondary_hit_rate"]

            # Adjust thresholds if hit rates are too extreme
            if primary_rate < 0.001:  # Too few primary boundaries
                if self.config.primary_threshold > 0:
                    adaptation_record["old_config"]["primary_threshold"] = self.config.primary_threshold
                    self.config.primary_threshold = max(0, self.config.primary_threshold - 1)
                    adaptation_record["new_config"]["primary_threshold"] = self.config.primary_threshold
                    adaptation_record["changes"].append("primary_threshold")
                    adapted = True

            elif primary_rate > 0.1:  # Too many primary boundaries
                adaptation_record["old_config"]["primary_threshold"] = self.config.primary_threshold
                self.config.primary_threshold = min(self.config.primary_divisor - 1, self.config.primary_threshold + 1)
                adaptation_record["new_config"]["primary_threshold"] = self.config.primary_threshold
                adaptation_record["changes"].append("primary_threshold")
                adapted = True

        if adapted:
            self.adaptation_history.append(adaptation_record)
            # Recreate hasher with new config
            self.hasher = TTTDHasher(self.config)

        return adapted

    def get_adaptation_info(self) -> Dict[str, Any]:
        """Get information about current adaptations."""
        return {
            "config": {
                "window_size": self.config.window_size,
                "min_chunk_size": self.config.min_chunk_size,
                "max_chunk_size": self.config.max_chunk_size,
                "target_chunk_size": self.config.target_chunk_size,
                "primary_divisor": self.config.primary_divisor,
                "primary_threshold": self.config.primary_threshold,
                "secondary_divisor": self.config.secondary_divisor,
                "secondary_threshold": self.config.secondary_threshold,
                "hash_polynomial": hex(self.config.hash_polynomial),
                "enable_adaptive_thresholds": self.config.enable_adaptive_thresholds,
                "threshold_adjustment_rate": self.config.threshold_adjustment_rate
            },
            "statistics": self.stats,
            "adaptation_history": self.adaptation_history,
            "adaptation_count": len(self.adaptation_history),
            "chunk_size_history_length": len(self.chunk_size_history)
        }

    def get_chunk_estimate(self, content_size: int) -> Tuple[int, int]:
        """Estimate number of chunks for given content size."""
        if content_size <= self.config.min_chunk_size:
            return (1, 1)

        # Estimate based on dual threshold probabilities
        primary_prob = 1 / self.config.primary_divisor
        secondary_prob = 1 / self.config.secondary_divisor

        # Combined probability (simplified model)
        combined_prob = primary_prob + secondary_prob - (primary_prob * secondary_prob)
        expected_chunk_size = max(self.config.min_chunk_size, 1 / combined_prob)

        estimated = max(1, int(content_size / expected_chunk_size))

        # Add variance for dual threshold variability
        variance = max(1, int(estimated * 0.4))  # ±40% variance
        return max(1, estimated - variance), estimated + variance

    def get_quality_score(self, chunks: List[Chunk]) -> float:
        """Calculate quality score for generated chunks."""
        if not chunks:
            return 0.0

        # Size distribution quality
        chunk_sizes = [chunk.metadata.extra.get("size", len(chunk.content)) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        target_ratio = avg_size / self.config.target_chunk_size
        size_score = max(0, 1 - abs(1 - target_ratio))

        # Boundary type distribution (good balance between primary/secondary)
        boundary_types = [chunk.metadata.extra.get("boundary_type") for chunk in chunks]
        primary_count = sum(1 for bt in boundary_types if bt == "primary")
        secondary_count = sum(1 for bt in boundary_types if bt == "secondary")
        total_boundaries = primary_count + secondary_count

        if total_boundaries > 0:
            balance_score = 1 - abs(primary_count - secondary_count) / total_boundaries
        else:
            balance_score = 0.5

        # TTTD efficiency (fewer size-forced boundaries is better)
        size_boundaries = sum(1 for bt in boundary_types if bt == "size")
        efficiency_score = max(0, 1 - (size_boundaries / len(chunks)))

        # Combined quality score
        return (size_score * 0.4 + balance_score * 0.3 + efficiency_score * 0.3)

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self.adaptation_history.copy()

    def describe_algorithm(self) -> str:
        """Describe the TTTD algorithm."""
        return f"""
        TTTD (Two-Threshold Two-Divisor) Chunking Algorithm:

        Window Size: {self.config.window_size} bytes
        Primary Divisor: {self.config.primary_divisor} (threshold: {self.config.primary_threshold})
        Secondary Divisor: {self.config.secondary_divisor} (threshold: {self.config.secondary_threshold})
        Target Chunk Size: {self.config.target_chunk_size} bytes
        Size Range: {self.config.min_chunk_size} - {self.config.max_chunk_size} bytes
        Adaptive Thresholds: {self.config.enable_adaptive_thresholds}

        TTTD uses two different threshold conditions for boundary detection:

        1. Primary Threshold: hash % {self.config.primary_divisor} == {self.config.primary_threshold}
           - Creates smaller, more frequent boundaries
           - Applied when chunk size >= target_size / 2

        2. Secondary Threshold: hash % {self.config.secondary_divisor} == {self.config.secondary_threshold}
           - Creates larger, less frequent boundaries
           - Has priority over primary threshold
           - Applied when chunk size >= min_size

        Algorithm Benefits:
        - Better control over chunk size distribution
        - Dual boundary conditions for flexibility
        - Adaptive threshold adjustment
        - Deterministic boundary detection
        - Balanced between small and large chunks

        Use Cases:
        - Systems requiring specific size distributions
        - Balanced deduplication and performance
        - Adaptive content processing
        - Research and experimental chunking
        """
