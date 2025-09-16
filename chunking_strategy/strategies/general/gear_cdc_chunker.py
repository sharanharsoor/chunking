"""
Gear-based Content-Defined Chunking Implementation.

This module implements Gear-based CDC, an alternative approach to content-defined
chunking that uses gear shift operations for boundary detection.
"""

import logging
import random
import time
from typing import List, Dict, Any, Optional, Union, Iterator
from dataclasses import dataclass

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


@dataclass
class GearCDCConfig:
    """Configuration for Gear-based CDC chunking."""
    window_size: int = 32              # Gear window size
    min_chunk_size: int = 2048         # 2KB minimum
    max_chunk_size: int = 65536        # 64KB maximum
    target_chunk_size: int = 8192      # 8KB target
    gear_mask: int = 0x003FFFFF       # 22-bit mask for gear detection
    gear_threshold: int = 13           # Number of consecutive gear bits
    normalization: int = 2             # Normalization level
    enable_statistics: bool = True
    hash_seed: int = 42               # Seed for gear table generation

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
        if self.gear_threshold <= 0:
            raise ValueError("gear_threshold must be positive")


class GearHasher:
    """
    Gear-based hasher for content-defined chunking.

    The Gear approach uses a different method for detecting boundaries
    by looking for specific bit patterns (gears) in the hash values.
    """

    def __init__(self, config: GearCDCConfig):
        self.config = config

        # Generate gear lookup table
        self.gear_table = self._generate_gear_table()

        # Initialize state
        self.reset()

    def _generate_gear_table(self) -> List[int]:
        """Generate pseudo-random gear lookup table."""
        random.seed(self.config.hash_seed)

        # Generate 256 random 32-bit values for gear computation
        table = []
        for i in range(256):
            # Generate value with good bit distribution
            value = random.getrandbits(32)
            # Ensure some entropy in lower bits
            value |= (i << 24)  # Mix in byte value
            table.append(value)

        return table

    def reset(self):
        """Reset gear hasher state."""
        self.hash_value = 0
        self.window = []
        self.gear_state = 0

    def roll_byte(self, byte_in: int) -> int:
        """
        Process a new byte and update gear hash.

        Args:
            byte_in: New byte to process

        Returns:
            Current gear hash value
        """
        self.window.append(byte_in)

        if len(self.window) > self.config.window_size:
            # Remove oldest byte
            byte_out = self.window.pop(0)

            # Update hash by removing old byte contribution
            old_contribution = self.gear_table[byte_out]
            self.hash_value ^= old_contribution

        # Add new byte contribution
        new_contribution = self.gear_table[byte_in]
        self.hash_value ^= new_contribution

        # Update gear state for boundary detection
        self._update_gear_state()

        return self.hash_value

    def _update_gear_state(self):
        """Update gear state for boundary detection."""
        # Count consecutive bits in the gear pattern
        masked_hash = self.hash_value & self.config.gear_mask

        # Count leading zeros or ones (depending on implementation)
        self.gear_state = self._count_consecutive_bits(masked_hash)

    def _count_consecutive_bits(self, value: int) -> int:
        """Count consecutive bits in value."""
        if value == 0:
            return 32  # All zeros

        # Count leading zeros
        count = 0
        mask = 1 << 31

        while count < 32 and not (value & mask):
            count += 1
            mask >>= 1

        return count

    def get_hash(self) -> int:
        """Get current hash value."""
        return self.hash_value

    def is_gear_boundary(self) -> bool:
        """
        Check if current state indicates a gear boundary.

        Returns:
            True if gear pattern indicates boundary
        """
        return self.gear_state >= self.config.gear_threshold

    def get_gear_state(self) -> int:
        """Get current gear state."""
        return self.gear_state


@register_chunker("gear_cdc")
class GearCDCChunker(StreamableChunker, AdaptableChunker):
    """
    Gear-based Content-Defined Chunker.

    This chunker implements gear-based CDC, which uses gear shift
    operations and bit pattern matching for boundary detection.
    It provides an alternative to hash-based methods.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], GearCDCConfig]] = None,
        **kwargs
    ):
        """
        Initialize Gear-based CDC chunker.

        Args:
            config: Configuration for the chunker
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="gear_cdc",
            category="general",
            supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
            **kwargs
        )

        # Parse configuration
        if isinstance(config, GearCDCConfig):
            self.config = config
        else:
            config = config or {}
            config.update(kwargs)
            self.config = GearCDCConfig(**{
                k: v for k, v in config.items()
                if k in GearCDCConfig.__dataclass_fields__
            })

        # Initialize gear hasher
        self.hasher = GearHasher(self.config)

        # Statistics
        self.stats = {
            "chunks_created": 0,
            "bytes_processed": 0,
            "gear_computations": 0,
            "gear_boundaries": 0,
            "avg_chunk_size": 0,
            "gear_distribution": {},
            "boundary_types": {"gear": 0, "size": 0}
        } if self.config.enable_statistics else None

        # Adaptation history
        self._adaptation_history = []

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Gear-based CDC chunker initialized with gear threshold {self.config.gear_threshold}")

    def chunk(
        self,
        content: Union[str, bytes],
        source_info: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Chunk content using Gear-based CDC algorithm.

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
                strategy_used="gear_cdc",
                processing_time=time.time() - start_time,
                source_info=source_info or {}
            )

        # Perform Gear-based CDC chunking
        chunks = list(self._gear_cdc_chunk(content_bytes, is_text))

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
            strategy_used="gear_cdc",
            processing_time=processing_time,
            source_info=source_info or {}
        )

    def _gear_cdc_chunk(self, content: bytes, is_text: bool) -> Iterator[Chunk]:
        """
        Perform Gear-based CDC chunking on content.

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
                id=f"gear_cdc_0",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=0,
                    length=len(content),
                    chunker_used="gear_cdc",
                    extra={
                        "chunk_index": 0,
                        "start_offset": 0,
                        "end_offset": len(content),
                        "algorithm": "gear_cdc",
                        "size": len(content),
                        "window_size": self.config.window_size
                    }
                )
            )
            return

        # Initialize gear hasher
        self.hasher.reset()

        chunk_start = 0
        chunk_index = 0

        # Process each byte
        for position in range(len(content)):
            byte_val = content[position]

            # Compute gear hash
            hash_value = self.hasher.roll_byte(byte_val)
            gear_state = self.hasher.get_gear_state()

            if self.stats:
                self.stats["gear_computations"] += 1
                # Track gear state distribution
                self.stats["gear_distribution"][gear_state] = \
                    self.stats["gear_distribution"].get(gear_state, 0) + 1

            # Check for chunk boundary
            chunk_size = position - chunk_start + 1

            is_gear_boundary = self.hasher.is_gear_boundary()
            is_size_boundary = chunk_size >= self.config.max_chunk_size

            if ((is_gear_boundary and chunk_size >= self.config.min_chunk_size) or
                is_size_boundary):

                # Determine boundary type
                boundary_type = "gear" if is_gear_boundary else "size"

                # Create chunk
                chunk_content = content[chunk_start:position + 1]
                if is_text:
                    chunk_content = chunk_content.decode('utf-8', errors='ignore')

                yield Chunk(
                    id=f"gear_cdc_{chunk_index}",
                    content=chunk_content,
                    modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                    metadata=ChunkMetadata(
                        source="unknown",
                        offset=chunk_start,
                        length=position - chunk_start + 1,
                        chunker_used="gear_cdc",
                        extra={
                            "chunk_index": chunk_index,
                            "start_offset": chunk_start,
                            "end_offset": position + 1,
                            "algorithm": "gear_cdc",
                            "hash_value": hash_value,
                            "gear_state": gear_state,
                            "size": chunk_size,
                            "boundary_type": boundary_type,
                            "gear_mask": hex(self.config.gear_mask),
                            "gear_threshold": self.config.gear_threshold,
                            "window_size": self.config.window_size
                        }
                    )
                )

                if self.stats:
                    if boundary_type == "gear":
                        self.stats["gear_boundaries"] += 1
                    self.stats["boundary_types"][boundary_type] += 1

                chunk_start = position + 1
                chunk_index += 1
                self.hasher.reset()

        # Handle remaining content
        if chunk_start < len(content):
            chunk_content = content[chunk_start:]
            if is_text:
                chunk_content = chunk_content.decode('utf-8', errors='ignore')

            yield Chunk(
                id=f"gear_cdc_{chunk_index}",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=chunk_start,
                    length=len(content) - chunk_start,
                    chunker_used="gear_cdc",
                    extra={
                        "chunk_index": chunk_index,
                        "start_offset": chunk_start,
                        "end_offset": len(content),
                        "algorithm": "gear_cdc",
                        "size": len(content) - chunk_start,
                        "boundary_type": "end",
                        "window_size": self.config.window_size
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
        Chunk data from a stream using Gear-based CDC.

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
            "old_config": {},
            "new_config": {},
            "changes": []
        }

        # Adapt gear threshold based on feedback score
        if feedback < 0.5:
            # Low score indicates need for adaptation
            old_threshold = self.config.gear_threshold
            # Adjust threshold slightly to improve performance
            new_threshold = max(8, min(20, old_threshold + (-1 if feedback < 0.3 else 1)))

            if new_threshold != old_threshold:
                adaptation_record["old_config"]["gear_threshold"] = old_threshold
                self.config.gear_threshold = new_threshold
                adaptation_record["new_config"]["gear_threshold"] = new_threshold
                adaptation_record["changes"].append("gear_threshold")
                adapted = True

        # Record adaptation attempt
        self._adaptation_history.append(adaptation_record)
        return adapted

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get adaptation history."""
        return self._adaptation_history.copy()

    def supports_streaming(self) -> bool:
        """Check if chunker supports streaming."""
        return True

    def chunk_stream(self, stream: Iterator[bytes],
                     source_info: Optional[Dict[str, Any]] = None) -> Iterator[Chunk]:
        """Chunk streaming data."""
        total_processed = 0
        chunk_index = 0

        for chunk in self._gear_cdc_chunk(b''.join(stream), is_text=False):
            chunk.metadata.extra["chunk_index"] = chunk_index
            chunk.metadata.extra["stream_offset"] = total_processed + chunk.metadata.extra["start_offset"]
            yield chunk
            chunk_index += 1

    def get_adaptation_info(self) -> Dict[str, Any]:
        """Get information about current adaptations."""
        return {
            "config": {
                "window_size": self.config.window_size,
                "min_chunk_size": self.config.min_chunk_size,
                "max_chunk_size": self.config.max_chunk_size,
                "target_chunk_size": self.config.target_chunk_size,
                "gear_mask": hex(self.config.gear_mask),
                "gear_threshold": self.config.gear_threshold,
                "normalization": self.config.normalization,
                "hash_seed": self.config.hash_seed
            },
            "statistics": self.stats,
            "adaptation_history": self.adaptation_history,
            "adaptation_count": len(self.adaptation_history)
        }

    def get_chunk_estimate(self, content_size: int) -> int:
        """Estimate number of chunks for given content size."""
        if content_size <= self.config.min_chunk_size:
            return 1

        # Estimate based on gear threshold and mask
        gear_probability = 1 / (2 ** self.config.gear_threshold)
        expected_chunk_size = max(self.config.min_chunk_size, 1 / gear_probability)

        estimated = max(1, int(content_size / expected_chunk_size))

        # Add variance for gear-based variability
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

        # Gear boundary ratio (higher is better for content-defined chunking)
        gear_boundaries = sum(1 for chunk in chunks
                             if chunk.metadata.extra.get("boundary_type") == "gear")
        gear_ratio = gear_boundaries / len(chunks) if chunks else 0
        boundary_score = gear_ratio

        # Consistency score (gear boundaries should be deterministic)
        consistency_score = 0.95  # High baseline for gear determinism

        # Combined quality score
        return (size_score * 0.4 + boundary_score * 0.35 + consistency_score * 0.25)

    def calculate_gear_hash(self, content: bytes) -> int:
        """
        Calculate gear hash for content.

        Args:
            content: Content to hash

        Returns:
            Gear hash value
        """
        self.hasher.reset()

        for byte_val in content:
            self.hasher.roll_byte(byte_val)

        return self.hasher.get_hash()

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self._adaptation_history.copy()

    def describe_algorithm(self) -> str:
        """Describe the Gear-based CDC algorithm."""
        return f"""
        Gear-based Content-Defined Chunking Algorithm:

        Window Size: {self.config.window_size} bytes
        Gear Mask: {hex(self.config.gear_mask)}
        Gear Threshold: {self.config.gear_threshold} bits
        Target Chunk Size: {self.config.target_chunk_size} bytes
        Size Range: {self.config.min_chunk_size} - {self.config.max_chunk_size} bytes

        The Gear-based CDC algorithm uses a different approach from traditional
        rolling hash methods. Instead of checking hash values against a mask,
        it analyzes bit patterns (gears) in the hash to determine boundaries.

        Algorithm Steps:
        1. Maintain a sliding window and compute hash using gear table
        2. For each position, analyze the bit pattern in the hash
        3. Count consecutive bits matching the gear pattern
        4. Create boundary when gear state exceeds threshold

        Advantages:
        - Alternative to hash-mask boundary detection
        - Good distribution properties
        - Tunable boundary detection sensitivity
        - Deterministic content-defined boundaries

        Use Cases:
        - Systems requiring different boundary characteristics
        - Research and experimentation with CDC algorithms
        - Specialized deduplication requirements
        - Performance-sensitive applications
        """
