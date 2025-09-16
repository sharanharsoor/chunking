"""
Rabin Fingerprinting Chunking (RFC) Implementation.

This module implements the classic Rabin fingerprinting algorithm for
content-defined chunking, widely used in deduplication systems.
"""

import logging
import random
import time
from typing import List, Dict, Any, Optional, Union, Iterator
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
class RabinFingerprintingConfig:
    """Configuration for Rabin fingerprinting chunking."""
    window_size: int = 48               # Sliding window size
    min_chunk_size: int = 2048         # 2KB minimum
    max_chunk_size: int = 65536        # 64KB maximum
    target_chunk_size: int = 8192      # 8KB target
    polynomial: int = 0x3DA3358B4DC173 # Irreducible polynomial
    boundary_mask: int = 0x1FFF        # 13-bit mask (8KB avg)
    polynomial_degree: int = 53        # Degree of the polynomial
    enable_statistics: bool = True
    seed: int = 42                     # Random seed for reproducibility

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
        if self.polynomial <= 0:
            raise ValueError("polynomial must be positive")


class RabinFingerprinter:
    """
    Rabin fingerprinting implementation with rolling hash.

    This implementation uses the classic Rabin fingerprinting algorithm
    for efficient content-defined chunking.
    """

    def __init__(self, config: RabinFingerprintingConfig):
        self.config = config
        self.polynomial = config.polynomial
        self.polynomial_degree = config.polynomial_degree

        # Precompute tables for efficiency
        self.mod_table = self._build_mod_table()
        self.pow_table = self._build_pow_table()

        # Initialize state
        self.reset()

    def _build_mod_table(self) -> List[int]:
        """Build modulo table for fast Rabin computation."""
        table = [0] * 256

        # For each possible byte value
        for i in range(256):
            # Compute polynomial modulo for this byte shifted to MSB position
            value = i << (self.polynomial_degree - 8)

            # Reduce by polynomial
            for _ in range(8):
                if value & (1 << (self.polynomial_degree - 1)):
                    value = (value << 1) ^ self.polynomial
                else:
                    value <<= 1
                value &= (1 << self.polynomial_degree) - 1

            table[i] = value

        return table

    def _build_pow_table(self) -> List[int]:
        """Build power table for window operations."""
        table = [1]
        base = 256

        for i in range(1, self.config.window_size + 1):
            # Compute base^i mod polynomial
            power = (table[-1] * base) % ((1 << self.polynomial_degree) - 1)

            # Reduce by polynomial if needed
            if power >= (1 << (self.polynomial_degree - 1)):
                power ^= self.polynomial
                power &= (1 << self.polynomial_degree) - 1

            table.append(power)

        return table

    def reset(self):
        """Reset fingerprinter state."""
        self.fingerprint = 0
        self.window = []

    def roll_byte(self, byte_in: int) -> int:
        """
        Roll in a new byte and compute Rabin fingerprint.

        Args:
            byte_in: New byte to process

        Returns:
            Current Rabin fingerprint
        """
        self.window.append(byte_in)

        if len(self.window) > self.config.window_size:
            # Remove oldest byte
            byte_out = self.window.pop(0)

            # Remove old byte contribution: subtract byte_out * base^(window_size-1)
            old_contribution = (byte_out * self.pow_table[self.config.window_size - 1]) % ((1 << self.polynomial_degree) - 1)
            self.fingerprint = (self.fingerprint - old_contribution) % ((1 << self.polynomial_degree) - 1)

        # Add new byte: fingerprint = fingerprint * base + byte_in
        self.fingerprint = (self.fingerprint * 256 + byte_in) % ((1 << self.polynomial_degree) - 1)

        # Apply polynomial reduction if needed
        if self.fingerprint >= (1 << (self.polynomial_degree - 1)):
            self.fingerprint ^= self.polynomial
            self.fingerprint &= (1 << self.polynomial_degree) - 1

        return self.fingerprint

    def get_fingerprint(self) -> int:
        """Get current fingerprint value."""
        return self.fingerprint

    def is_boundary(self, fingerprint: int) -> bool:
        """
        Check if fingerprint indicates a chunk boundary.

        Args:
            fingerprint: Current fingerprint value

        Returns:
            True if this is a boundary position
        """
        return (fingerprint & self.config.boundary_mask) == 0


@register_chunker(
    name="rabin_fingerprinting",
    category="general",
    description="Classic Rabin fingerprinting algorithm for content-defined chunking",
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.MEDIUM,
    quality=0.8,
    supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
    supported_formats=["*"],
    parameters_schema={
        "window_size": {"type": "integer", "minimum": 8, "maximum": 256, "default": 48},
        "min_chunk_size": {"type": "integer", "minimum": 256, "maximum": 32768, "default": 2048},
        "max_chunk_size": {"type": "integer", "minimum": 4096, "maximum": 1048576, "default": 65536},
        "target_chunk_size": {"type": "integer", "minimum": 1024, "maximum": 131072, "default": 8192},
        "polynomial": {"type": "integer", "minimum": 1, "default": 0x3DA3358B4DC173},
        "boundary_mask": {"type": "integer", "minimum": 1, "default": 0x1FFF},
        "polynomial_degree": {"type": "integer", "minimum": 8, "maximum": 64, "default": 53}
    },
    default_parameters={
        "window_size": 48,
        "min_chunk_size": 2048,
        "max_chunk_size": 65536,
        "target_chunk_size": 8192,
        "polynomial": 0x3DA3358B4DC173,
        "boundary_mask": 0x1FFF,
        "polynomial_degree": 53
    },
    dependencies=[],
    use_cases=["deduplication", "backup systems", "content-defined chunking"],
    best_for=["variable content", "deduplication efficiency"],
    limitations=["polynomial parameter tuning", "computational overhead"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=False
)
class RabinFingerprintingChunker(StreamableChunker, AdaptableChunker):
    """
    Rabin Fingerprinting Chunker (RFC).

    This chunker implements the classic Rabin fingerprinting algorithm
    for content-defined chunking. It's widely used in deduplication
    systems and backup applications.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], RabinFingerprintingConfig]] = None,
        **kwargs
    ):
        """
        Initialize Rabin Fingerprinting chunker.

        Args:
            config: Configuration for the chunker
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="rabin_fingerprinting",
            category="general",
            supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
            **kwargs
        )

        # Parse configuration
        if isinstance(config, RabinFingerprintingConfig):
            self.config = config
        else:
            config = config or {}
            config.update(kwargs)
            self.config = RabinFingerprintingConfig(**{
                k: v for k, v in config.items()
                if k in RabinFingerprintingConfig.__dataclass_fields__
            })

        # Initialize fingerprinter
        self.fingerprinter = RabinFingerprinter(self.config)

        # Statistics
        self.stats = {
            "chunks_created": 0,
            "bytes_processed": 0,
            "fingerprint_computations": 0,
            "boundary_hits": 0,
            "avg_chunk_size": 0,
            "deduplication_ratio": 0
        } if self.config.enable_statistics else None

        # Adaptation history
        self._adaptation_history = []

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Rabin fingerprinting chunker initialized with polynomial {hex(self.config.polynomial)}")

    def chunk(
        self,
        content: Union[str, bytes],
        source_info: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Chunk content using Rabin fingerprinting algorithm.

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
                strategy_used="rabin_fingerprinting",
                processing_time=time.time() - start_time,
                source_info=source_info or {}
            )

        # Perform Rabin fingerprinting chunking
        chunks = list(self._rabin_chunk(content_bytes, is_text))

        processing_time = time.time() - start_time

        # Update statistics
        if self.stats:
            self.stats["chunks_created"] += len(chunks)
            self.stats["bytes_processed"] += len(content_bytes)
            if chunks:
                total_size = sum(chunk.metadata.extra.get("size", 0) for chunk in chunks)
                self.stats["avg_chunk_size"] = total_size / len(chunks)

        return ChunkingResult(
            chunks=chunks,
            strategy_used="rabin_fingerprinting",
            processing_time=processing_time,
            source_info=source_info or {}
        )

    def _rabin_chunk(self, content: bytes, is_text: bool) -> Iterator[Chunk]:
        """
        Perform Rabin fingerprinting chunking on content.

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
                id=f"rabin_0",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=0,
                    length=len(content),
                    chunker_used="rabin_fingerprinting",
                    extra={
                        "chunk_index": 0,
                        "start_offset": 0,
                        "end_offset": len(content),
                        "algorithm": "rabin_fingerprinting",
                        "size": len(content),
                        "polynomial": hex(self.config.polynomial)
                    }
                )
            )
            return

        # Initialize fingerprinter
        self.fingerprinter.reset()

        chunk_start = 0
        chunk_index = 0

        # Process each byte
        for position in range(len(content)):
            byte_val = content[position]

            # Compute Rabin fingerprint
            fingerprint = self.fingerprinter.roll_byte(byte_val)

            if self.stats:
                self.stats["fingerprint_computations"] += 1

            # Check for chunk boundary
            chunk_size = position - chunk_start + 1

            if ((self.fingerprinter.is_boundary(fingerprint) and
                 chunk_size >= self.config.min_chunk_size) or
                chunk_size >= self.config.max_chunk_size):

                # Create chunk
                chunk_content = content[chunk_start:position + 1]
                if is_text:
                    chunk_content = chunk_content.decode('utf-8', errors='ignore')

                yield Chunk(
                    id=f"rabin_{chunk_index}",
                    content=chunk_content,
                    modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                    metadata=ChunkMetadata(
                        source="unknown",
                        offset=chunk_start,
                        length=position - chunk_start + 1,
                        chunker_used="rabin_fingerprinting",
                        extra={
                            "chunk_index": chunk_index,
                            "start_offset": chunk_start,
                            "end_offset": position + 1,
                            "algorithm": "rabin_fingerprinting",
                            "fingerprint": fingerprint,
                            "size": chunk_size,
                            "polynomial": hex(self.config.polynomial),
                            "boundary_mask": hex(self.config.boundary_mask)
                        }
                    )
                )

                if self.stats:
                    self.stats["boundary_hits"] += 1

                chunk_start = position + 1
                chunk_index += 1
                self.fingerprinter.reset()

        # Handle remaining content
        if chunk_start < len(content):
            chunk_content = content[chunk_start:]
            if is_text:
                chunk_content = chunk_content.decode('utf-8', errors='ignore')

            yield Chunk(
                id=f"rabin_{chunk_index}",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=chunk_start,
                    length=len(content) - chunk_start,
                    chunker_used="rabin_fingerprinting",
                    extra={
                        "chunk_index": chunk_index,
                        "start_offset": chunk_start,
                        "end_offset": len(content),
                        "algorithm": "rabin_fingerprinting",
                        "size": len(content) - chunk_start,
                        "polynomial": hex(self.config.polynomial)
                    }
                )
            )

    def supports_streaming(self) -> bool:
        """Check if chunker supports streaming."""
        return True

    def chunk_stream(
        self,
        stream: Iterator[Union[str, bytes]],
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Chunk data from a stream using Rabin fingerprinting.

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
                # Find safe cut point (not in the middle of a potential boundary)
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

        # Adapt boundary mask based on actual vs target chunk size
        if "avg_chunk_size" in feedback:
            current_avg = feedback["avg_chunk_size"]
            target = self.config.target_chunk_size

            if abs(current_avg - target) > target * 0.25:  # 25% tolerance
                # Adjust boundary mask to get closer to target
                if current_avg > target:
                    # Chunks too large, make boundary condition more likely
                    new_mask = self.config.boundary_mask >> 1
                else:
                    # Chunks too small, make boundary condition less likely
                    new_mask = (self.config.boundary_mask << 1) | 1

                # Ensure mask stays within reasonable bounds
                new_mask = max(0x3FF, min(0x7FFF, new_mask))  # 10-bit to 15-bit range

                if new_mask != self.config.boundary_mask:
                    adaptation_record["old_config"]["boundary_mask"] = hex(self.config.boundary_mask)
                    self.config.boundary_mask = new_mask
                    adaptation_record["new_config"]["boundary_mask"] = hex(new_mask)
                    adaptation_record["changes"].append("boundary_mask")
                    adapted = True

                    self.logger.info(f"Adapted boundary mask to {hex(new_mask)} for target chunk size {target}")

        # Adapt window size based on performance
        if "processing_time" in feedback and "content_size" in feedback:
            throughput = feedback["content_size"] / feedback["processing_time"]

            # If processing is slow, consider smaller window
            if throughput < 1_000_000:  # Less than 1MB/s
                if self.config.window_size > 32:
                    adaptation_record["old_config"]["window_size"] = self.config.window_size
                    self.config.window_size = max(32, self.config.window_size - 8)
                    adaptation_record["new_config"]["window_size"] = self.config.window_size
                    adaptation_record["changes"].append("window_size")
                    adapted = True

                    self.logger.info(f"Adapted window size to {self.config.window_size} for better performance")

        if adapted:
            self.adaptation_history.append(adaptation_record)
            # Recreate fingerprinter with new config
            self.fingerprinter = RabinFingerprinter(self.config)

        return adapted

    def get_adaptation_info(self) -> Dict[str, Any]:
        """Get information about current adaptations."""
        return {
            "config": {
                "window_size": self.config.window_size,
                "min_chunk_size": self.config.min_chunk_size,
                "max_chunk_size": self.config.max_chunk_size,
                "target_chunk_size": self.config.target_chunk_size,
                "polynomial": hex(self.config.polynomial),
                "boundary_mask": hex(self.config.boundary_mask),
                "polynomial_degree": self.config.polynomial_degree
            },
            "statistics": self.stats,
            "adaptation_history": self.adaptation_history,
            "adaptation_count": len(self.adaptation_history)
        }

    def get_chunk_estimate(self, content_size: int) -> int:
        """Estimate number of chunks for given content size."""
        if content_size <= self.config.min_chunk_size:
            return 1

        # Estimate based on boundary probability
        boundary_probability = 1 / (self.config.boundary_mask + 1)
        expected_chunk_size = 1 / boundary_probability

        # Adjust for window size and minimum chunk constraints
        effective_chunk_size = max(self.config.min_chunk_size, expected_chunk_size)

        estimated = max(1, int(content_size / effective_chunk_size))

        # Add variance due to probabilistic nature
        variance = max(1, int(estimated * 0.3))  # ±30% variance
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

        # Boundary consistency (Rabin boundaries should be deterministic)
        boundary_score = 1.0  # Rabin boundaries are always consistent

        # Deduplication potential (based on size variance)
        if len(chunk_sizes) > 1:
            size_variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)
            size_std = size_variance ** 0.5
            dedup_score = max(0, 1 - (size_std / avg_size)) if avg_size > 0 else 0
        else:
            dedup_score = 1.0

        # Combined quality score
        return (size_score * 0.4 + boundary_score * 0.3 + dedup_score * 0.3)

    def calculate_fingerprint(self, content: bytes) -> int:
        """
        Calculate Rabin fingerprint for content.

        Args:
            content: Content to fingerprint

        Returns:
            Rabin fingerprint value
        """
        self.fingerprinter.reset()

        for byte_val in content:
            self.fingerprinter.roll_byte(byte_val)

        return self.fingerprinter.get_fingerprint()

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """Adapt Rabin fingerprinting parameters based on feedback."""
        adaptation = {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "old_boundary_mask": self.config.boundary_mask,
            "old_window_size": self.config.window_size
        }

        if feedback_type == "quality" and feedback_score < 0.5:
            # Adjust boundary mask for better boundary detection
            old_mask = self.config.boundary_mask
            # Reduce mask to create more boundaries (smaller chunks)
            self.config.boundary_mask = max(0x3FF, self.config.boundary_mask >> 1)
            adaptation["new_boundary_mask"] = self.config.boundary_mask
            if hasattr(self, 'logger'):
                self.logger.info(f"Adapted boundary_mask: {hex(old_mask)} -> {hex(self.config.boundary_mask)}")

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Increase mask for fewer boundaries (larger chunks, better performance)
            old_mask = self.config.boundary_mask
            self.config.boundary_mask = min(0xFFFF, self.config.boundary_mask << 1)
            adaptation["new_boundary_mask"] = self.config.boundary_mask
            if hasattr(self, 'logger'):
                self.logger.info(f"Adapted boundary_mask for performance: {hex(old_mask)} -> {hex(self.config.boundary_mask)}")

        self._adaptation_history.append(adaptation)

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self._adaptation_history.copy()

    def describe_algorithm(self) -> str:
        """Describe the Rabin fingerprinting algorithm."""
        return f"""
        Rabin Fingerprinting Chunking (RFC) Algorithm:

        Polynomial: {hex(self.config.polynomial)} (degree {self.config.polynomial_degree})
        Window Size: {self.config.window_size} bytes
        Boundary Mask: {hex(self.config.boundary_mask)}
        Target Chunk Size: {self.config.target_chunk_size} bytes
        Size Range: {self.config.min_chunk_size} - {self.config.max_chunk_size} bytes

        The Rabin fingerprinting algorithm uses polynomial arithmetic in
        finite fields to compute rolling hash values. Chunk boundaries
        are determined when the fingerprint value satisfies the boundary
        condition (fingerprint & mask == 0).

        Advantages:
        - Deterministic content-defined boundaries
        - Excellent deduplication properties
        - Widely studied and proven algorithm
        - Good distribution of chunk sizes

        Use Cases:
        - Backup and synchronization systems
        - Deduplication storage
        - Version control systems
        - Content distribution networks
        """
