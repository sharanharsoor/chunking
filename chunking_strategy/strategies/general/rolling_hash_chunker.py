"""
Rolling Hash Chunking Implementation.

This module provides a foundational rolling hash chunking framework that can be
configured with different hash functions (Rabin, BuzHash, polynomial, etc.).
"""

import logging
import hashlib
import struct
import time
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
from dataclasses import dataclass
from pathlib import Path

from chunking_strategy.core.base import (
    Chunk,
    ChunkingResult,
    ModalityType,
    ChunkMetadata
)
from chunking_strategy.core.adaptive import AdaptableChunker
from chunking_strategy.core.streaming import StreamableChunker
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage


logger = logging.getLogger(__name__)


@dataclass
class RollingHashConfig:
    """Configuration for rolling hash chunking."""
    window_size: int = 64
    min_chunk_size: int = 2048      # 2KB
    max_chunk_size: int = 65536     # 64KB
    target_chunk_size: int = 8192   # 8KB
    hash_function: str = "polynomial"  # polynomial, rabin, buzhash
    normalization: int = 2
    polynomial: int = 0x3DA3358B4DC173  # Default polynomial for Rabin
    gear_mask: int = 0x003FFFFF      # 22-bit mask for gear
    enable_statistics: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.max_chunk_size <= self.min_chunk_size:
            raise ValueError("max_chunk_size must be greater than min_chunk_size")
        if self.target_chunk_size < self.min_chunk_size or self.target_chunk_size > self.max_chunk_size:
            raise ValueError("target_chunk_size must be between min and max chunk sizes")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")


class RollingHashFunction:
    """Base class for rolling hash functions."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset the hash state."""
        self.hash_value = 0
        self.window = []

    def roll(self, byte_in: int, byte_out: Optional[int] = None) -> int:
        """
        Roll the hash with a new byte.

        Args:
            byte_in: New byte to add
            byte_out: Old byte to remove (if window is full)

        Returns:
            Current hash value
        """
        raise NotImplementedError("Subclasses must implement roll method")

    def get_hash(self) -> int:
        """Get current hash value."""
        return self.hash_value


class PolynomialRollingHash(RollingHashFunction):
    """Simple polynomial rolling hash."""

    def __init__(self, window_size: int, base: int = 256, modulus: int = 10**9 + 7):
        self.base = base
        self.modulus = modulus
        self.base_power = pow(base, window_size - 1, modulus)
        super().__init__(window_size)

    def roll(self, byte_in: int, byte_out: Optional[int] = None) -> int:
        """Roll polynomial hash."""
        self.window.append(byte_in)

        if len(self.window) > self.window_size:
            byte_out = self.window.pop(0)
            # Remove old byte and add new byte
            self.hash_value = ((self.hash_value - byte_out * self.base_power) * self.base + byte_in) % self.modulus
        else:
            # Just add new byte
            self.hash_value = (self.hash_value * self.base + byte_in) % self.modulus

        return self.hash_value


class RabinRollingHash(RollingHashFunction):
    """Rabin fingerprinting rolling hash."""

    def __init__(self, window_size: int, polynomial: int = 0x3DA3358B4DC173):
        self.polynomial = polynomial
        self.poly_degree = polynomial.bit_length() - 1
        self.mod_table = self._build_mod_table()
        super().__init__(window_size)

    def _build_mod_table(self) -> List[int]:
        """Build precomputed modulo table for efficiency."""
        table = [0] * 256
        for i in range(256):
            value = i << (self.poly_degree - 8)
            for _ in range(8):
                if value & (1 << (self.poly_degree - 1)):
                    value = (value << 1) ^ self.polynomial
                else:
                    value <<= 1
                value &= (1 << self.poly_degree) - 1
            table[i] = value
        return table

    def roll(self, byte_in: int, byte_out: Optional[int] = None) -> int:
        """Roll Rabin hash."""
        self.window.append(byte_in)

        if len(self.window) > self.window_size:
            byte_out = self.window.pop(0)
            # Remove old byte contribution
            old_contribution = self.mod_table[byte_out] << ((self.window_size - 1) * 8 % self.poly_degree)
            old_contribution &= (1 << self.poly_degree) - 1
            self.hash_value ^= old_contribution

        # Add new byte
        self.hash_value = (self.hash_value << 8) ^ byte_in
        self.hash_value &= (1 << self.poly_degree) - 1

        return self.hash_value


class BuzHash(RollingHashFunction):
    """BuzHash rolling hash implementation."""

    def __init__(self, window_size: int):
        super().__init__(window_size)
        # Precomputed random table for BuzHash
        self.hash_table = self._generate_hash_table()
        self.rotations = [0] * 256
        for i in range(256):
            self.rotations[i] = self._left_rotate(self.hash_table[i], window_size % 32)

    def _generate_hash_table(self) -> List[int]:
        """Generate pseudo-random hash table."""
        import random
        random.seed(42)  # Deterministic seed for reproducibility
        return [random.getrandbits(32) for _ in range(256)]

    def _left_rotate(self, value: int, positions: int) -> int:
        """Left rotate 32-bit value."""
        positions %= 32
        return ((value << positions) | (value >> (32 - positions))) & 0xFFFFFFFF

    def roll(self, byte_in: int, byte_out: Optional[int] = None) -> int:
        """Roll BuzHash."""
        self.window.append(byte_in)

        if len(self.window) > self.window_size:
            byte_out = self.window.pop(0)
            # Remove old byte and add new byte
            self.hash_value ^= self.rotations[byte_out]

        self.hash_value = self._left_rotate(self.hash_value, 1) ^ self.hash_table[byte_in]
        return self.hash_value & 0xFFFFFFFF


@register_chunker("rolling_hash")
class RollingHashChunker(StreamableChunker, AdaptableChunker):
    """
    Rolling Hash Chunker using configurable hash functions.

    This chunker provides a foundation for content-defined chunking using
    rolling hash algorithms. It supports multiple hash functions including
    polynomial, Rabin, and BuzHash.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], RollingHashConfig]] = None,
        **kwargs
    ):
        """
        Initialize Rolling Hash chunker.

        Args:
            config: Configuration for the chunker
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="rolling_hash",
            category="general",
            supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
            **kwargs
        )

        # Parse configuration
        if isinstance(config, RollingHashConfig):
            self.config = config
        else:
            config = config or {}
            config.update(kwargs)
            self.config = RollingHashConfig(**{
                k: v for k, v in config.items()
                if k in RollingHashConfig.__dataclass_fields__
            })

        # Initialize hash function
        self.hash_function = self._create_hash_function()

        # Statistics
        self.stats = {
            "chunks_created": 0,
            "bytes_processed": 0,
            "hash_computations": 0,
            "boundary_hits": 0
        } if self.config.enable_statistics else None

        # Adaptation history
        self._adaptation_history = []

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Rolling hash chunker initialized with {self.config.hash_function} hash")

    def _create_hash_function(self) -> RollingHashFunction:
        """Create appropriate hash function based on configuration."""
        if self.config.hash_function == "polynomial":
            return PolynomialRollingHash(self.config.window_size)
        elif self.config.hash_function == "rabin":
            return RabinRollingHash(self.config.window_size, self.config.polynomial)
        elif self.config.hash_function == "buzhash":
            return BuzHash(self.config.window_size)
        else:
            raise ValueError(f"Unsupported hash function: {self.config.hash_function}")

    def chunk(
        self,
        content: Union[str, bytes],
        source_info: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Chunk content using rolling hash algorithm.

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
                strategy_used="rolling_hash",
                processing_time=time.time() - start_time,
                source_info=source_info or {}
            )

        # Perform rolling hash chunking
        chunks = list(self._rolling_hash_chunk(content_bytes, is_text))

        processing_time = time.time() - start_time

        # Update statistics
        if self.stats:
            self.stats["chunks_created"] += len(chunks)
            self.stats["bytes_processed"] += len(content_bytes)

        return ChunkingResult(
            chunks=chunks,
            strategy_used="rolling_hash",
            processing_time=processing_time,
            source_info=source_info or {}
        )

    def _rolling_hash_chunk(self, content: bytes, is_text: bool) -> Iterator[Chunk]:
        """
        Perform rolling hash chunking on content.

        Args:
            content: Content bytes to chunk
            is_text: Whether content is text

        Yields:
            Generated chunks
        """
        if len(content) <= self.config.min_chunk_size:
            # Content too small, return as single chunk
            chunk_content = content.decode('utf-8', errors='replace') if is_text else content
            yield Chunk(
                id=f"rolling_hash_0",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=0,
                    length=len(content),
                    chunker_used="rolling_hash",
                    extra={
                        "chunk_index": 0,
                        "start_offset": 0,
                        "end_offset": len(content),
                        "hash_function": self.config.hash_function,
                        "size": len(content),
                        "algorithm": "rolling_hash"
                    }
                )
            )
            return

        # Initialize rolling hash
        self.hash_function.reset()

        chunk_start = 0
        chunk_index = 0
        position = 0

        # Calculate threshold for boundary detection
        threshold = self._calculate_threshold()

        # Process each byte
        while position < len(content):
            byte_val = content[position]

            # Roll hash
            if position >= self.config.window_size:
                byte_out = content[position - self.config.window_size]
                hash_val = self.hash_function.roll(byte_val, byte_out)
            else:
                hash_val = self.hash_function.roll(byte_val)

            if self.stats:
                self.stats["hash_computations"] += 1

            # Check for chunk boundary
            chunk_size = position - chunk_start + 1

            if (self._is_boundary(hash_val, threshold, chunk_size) and
                chunk_size >= self.config.min_chunk_size) or \
               chunk_size >= self.config.max_chunk_size:

                # Create chunk
                chunk_content = content[chunk_start:position + 1]
                if is_text:
                    chunk_content = chunk_content.decode('utf-8', errors='replace')

                yield Chunk(
                    id=f"rolling_hash_{chunk_index}",
                    content=chunk_content,
                    modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                    metadata=ChunkMetadata(
                        source="unknown",
                        offset=chunk_start,
                        length=position - chunk_start + 1,
                        chunker_used="rolling_hash",
                        extra={
                            "chunk_index": chunk_index,
                            "start_offset": chunk_start,
                            "end_offset": position + 1,
                            "hash_function": self.config.hash_function,
                            "hash_value": hash_val,
                            "size": position - chunk_start + 1,
                            "algorithm": "rolling_hash"
                        }
                    )
                )

                if self.stats:
                    self.stats["boundary_hits"] += 1

                chunk_start = position + 1
                chunk_index += 1
                self.hash_function.reset()

            position += 1

        # Handle remaining content
        if chunk_start < len(content):
            chunk_content = content[chunk_start:]
            if is_text:
                chunk_content = chunk_content.decode('utf-8', errors='replace')

            yield Chunk(
                id=f"rolling_hash_{chunk_index}",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=chunk_start,
                    length=len(content) - chunk_start,
                    chunker_used="rolling_hash",
                    extra={
                        "chunk_index": chunk_index,
                        "start_offset": chunk_start,
                        "end_offset": len(content),
                        "hash_function": self.config.hash_function,
                        "size": len(content) - chunk_start,
                        "algorithm": "rolling_hash"
                    }
                )
            )

    def _calculate_threshold(self) -> int:
        """Calculate boundary threshold based on target chunk size."""
        # Simple threshold calculation - can be enhanced
        if self.config.hash_function == "buzhash":
            # For 32-bit BuzHash
            return (1 << 32) // self.config.target_chunk_size
        else:
            # For polynomial and Rabin
            return (1 << 20) // self.config.target_chunk_size  # Assuming 20-bit hash

    def _is_boundary(self, hash_val: int, threshold: int, chunk_size: int) -> bool:
        """
        Determine if current position is a chunk boundary.

        Args:
            hash_val: Current hash value
            threshold: Boundary threshold
            chunk_size: Current chunk size

        Returns:
            True if this is a boundary position
        """
        # Basic boundary detection - hash value below threshold
        return (hash_val % threshold) == 0

    def supports_streaming(self) -> bool:
        """Check if chunker supports streaming."""
        return True

    def chunk_stream(
        self,
        stream: Iterator[Union[str, bytes]],
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Chunk data from a stream.

        Args:
            stream: Iterator of content chunks
            **kwargs: Additional parameters

        Yields:
            Generated chunks
        """
        buffer = b""
        chunk_index = 0

        for data in stream:
            if isinstance(data, str):
                data = data.encode('utf-8')

            buffer += data

            # Process buffer when it's large enough
            if len(buffer) >= self.config.max_chunk_size * 2:
                # Chunk the buffer
                result = self.chunk(buffer)

                # Yield complete chunks (keep incomplete last chunk in buffer)
                for i, chunk in enumerate(result.chunks[:-1]):
                    chunk.metadata.extra["chunk_index"] = chunk_index
                    yield chunk
                    chunk_index += 1

                # Keep last incomplete chunk in buffer
                if result.chunks:
                    last_chunk = result.chunks[-1]
                    buffer = last_chunk.content.encode('utf-8') if isinstance(last_chunk.content, str) else last_chunk.content
                else:
                    buffer = b""

        # Process remaining buffer
        if buffer:
            result = self.chunk(buffer)
            for chunk in result.chunks:
                chunk.metadata.extra["chunk_index"] = chunk_index
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

        # Adapt target chunk size based on performance
        if "avg_chunk_size" in feedback and "target_chunk_size" in feedback:
            current_avg = feedback["avg_chunk_size"]
            target = feedback["target_chunk_size"]

            if abs(current_avg - target) > target * 0.2:  # 20% tolerance
                adjustment = (target - current_avg) * 0.1  # 10% adjustment
                new_target = max(
                    self.config.min_chunk_size,
                    min(self.config.max_chunk_size, int(self.config.target_chunk_size + adjustment))
                )

                if new_target != self.config.target_chunk_size:
                    self.config.target_chunk_size = new_target
                    adapted = True
                    self.logger.info(f"Adapted target chunk size to {new_target}")

        return adapted

    def get_adaptation_info(self) -> Dict[str, Any]:
        """Get information about current adaptations."""
        return {
            "config": {
                "window_size": self.config.window_size,
                "min_chunk_size": self.config.min_chunk_size,
                "max_chunk_size": self.config.max_chunk_size,
                "target_chunk_size": self.config.target_chunk_size,
                "hash_function": self.config.hash_function,
                "normalization": self.config.normalization
            },
            "statistics": self.stats,
            "hash_function_type": type(self.hash_function).__name__
        }

    def get_chunk_estimate(self, content_size: int) -> int:
        """Estimate number of chunks for given content size."""
        if content_size <= self.config.min_chunk_size:
            return 1

        # Estimate based on target chunk size
        estimated = content_size // self.config.target_chunk_size

        # Add some variance due to content-defined nature
        variance = max(1, int(estimated * 0.2))  # ±20% variance
        return max(1, estimated - variance), estimated + variance

    def get_quality_score(self, chunks: List[Chunk]) -> float:
        """Calculate quality score for generated chunks."""
        if not chunks:
            return 0.0

        chunk_sizes = [chunk.metadata.extra.get("size", len(chunk.content)) for chunk in chunks]

        # Calculate size distribution quality
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        size_variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)
        size_std = size_variance ** 0.5

        # Quality based on how close chunks are to target size
        size_score = max(0, 1 - (size_std / avg_size)) if avg_size > 0 else 0

        # Boundary consistency score (placeholder - could be enhanced)
        boundary_score = 0.8  # Assume good boundary detection

        # Combined quality score
        return (size_score * 0.7 + boundary_score * 0.3)

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """Adapt rolling hash parameters based on feedback."""
        adaptation = {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "old_target_chunk_size": self.config.target_chunk_size,
            "old_window_size": self.config.window_size
        }

        if feedback_type == "quality" and feedback_score < 0.5:
            # Adjust window size for better boundary detection
            old_window = self.config.window_size
            self.config.window_size = min(128, max(8, int(self.config.window_size * 1.2)))
            adaptation["new_window_size"] = self.config.window_size
            if hasattr(self, 'logger'):
                self.logger.info(f"Adapted window_size: {old_window} -> {self.config.window_size}")

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Reduce window size for better performance
            old_window = self.config.window_size
            self.config.window_size = max(8, int(self.config.window_size * 0.8))
            adaptation["new_window_size"] = self.config.window_size
            if hasattr(self, 'logger'):
                self.logger.info(f"Adapted window_size for performance: {old_window} -> {self.config.window_size}")

        self._adaptation_history.append(adaptation)

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self._adaptation_history.copy()

    def describe_algorithm(self) -> str:
        """Describe the rolling hash algorithm."""
        return f"""
        Rolling Hash Chunking Algorithm:

        Hash Function: {self.config.hash_function}
        Window Size: {self.config.window_size} bytes
        Target Chunk Size: {self.config.target_chunk_size} bytes
        Size Range: {self.config.min_chunk_size} - {self.config.max_chunk_size} bytes

        The algorithm uses a rolling hash to find content-defined boundaries.
        As the hash window slides over the content, chunk boundaries are
        determined when the hash value meets specific criteria.

        Advantages:
        - Content-defined boundaries for deduplication
        - Configurable hash functions for different use cases
        - Adjustable parameters for various content types
        """

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """
        Adapt chunker parameters based on feedback.

        Args:
            feedback_score: Feedback score (0.0 to 1.0)
            feedback_type: Type of feedback
            **kwargs: Additional feedback information
        """
        adaptation_entry = {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "old_config": {
                "target_chunk_size": self.config.target_chunk_size,
                "window_size": self.config.window_size,
                "hash_function": self.config.hash_function
            }
        }

        if feedback_type == "quality" and feedback_score < 0.5:
            # Increase target chunk size for better quality
            old_size = self.config.target_chunk_size
            self.config.target_chunk_size = min(
                int(self.config.target_chunk_size * 1.2),
                int(self.config.max_chunk_size * 0.9)  # Use 90% of max instead of subtracting fixed amount
            )
            self.logger.info(f"Adapted target_chunk_size: {old_size} -> {self.config.target_chunk_size} (quality feedback)")

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Decrease target chunk size for better performance
            old_size = self.config.target_chunk_size
            self.config.target_chunk_size = max(
                int(self.config.target_chunk_size * 0.8),
                int(self.config.min_chunk_size * 1.1)  # Use 110% of min instead of adding fixed amount
            )
            self.logger.info(f"Adapted target_chunk_size: {old_size} -> {self.config.target_chunk_size} (performance feedback)")

        adaptation_entry["new_config"] = {
            "target_chunk_size": self.config.target_chunk_size,
            "window_size": self.config.window_size,
            "hash_function": self.config.hash_function
        }

        self._adaptation_history.append(adaptation_entry)

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self._adaptation_history.copy()
