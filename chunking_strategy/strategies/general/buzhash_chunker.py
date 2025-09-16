"""
BuzHash Chunking Implementation.

This module implements BuzHash, a fast rolling hash algorithm that provides
good content-defined chunking with better performance than Rabin fingerprinting.
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
class BuzHashConfig:
    """Configuration for BuzHash chunking."""
    window_size: int = 64              # Sliding window size
    min_chunk_size: int = 2048         # 2KB minimum
    max_chunk_size: int = 65536        # 64KB maximum
    target_chunk_size: int = 8192      # 8KB target
    boundary_mask: int = 0x1FFF        # 13-bit mask (8KB avg)
    hash_table_seed: int = 42          # Seed for hash table generation
    enable_statistics: bool = True
    normalization: int = 2             # Boundary normalization factor

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


class BuzHasher:
    """
    BuzHash implementation for rolling hash computation.

    BuzHash is a rolling hash algorithm that's faster than Rabin fingerprinting
    while maintaining good distribution properties for content-defined chunking.
    """

    def __init__(self, config: BuzHashConfig):
        self.config = config
        self.window_size = config.window_size

        # Generate pseudo-random hash table
        self.hash_table = self._generate_hash_table()

        # Precompute rotation values for efficiency
        self.rotations = [0] * 256
        for i in range(256):
            self.rotations[i] = self._left_rotate(self.hash_table[i], self.window_size % 32)

        # Initialize state
        self.reset()

    def _generate_hash_table(self) -> List[int]:
        """Generate pseudo-random 32-bit hash table for BuzHash."""
        random.seed(self.config.hash_table_seed)
        return [random.getrandbits(32) for _ in range(256)]

    def _left_rotate(self, value: int, positions: int) -> int:
        """Left rotate a 32-bit value by specified positions."""
        positions %= 32
        return ((value << positions) | (value >> (32 - positions))) & 0xFFFFFFFF

    def _right_rotate(self, value: int, positions: int) -> int:
        """Right rotate a 32-bit value by specified positions."""
        positions %= 32
        return ((value >> positions) | (value << (32 - positions))) & 0xFFFFFFFF

    def reset(self):
        """Reset BuzHash state."""
        self.hash_value = 0
        self.window = []

    def roll_byte(self, byte_in: int) -> int:
        """
        Roll in a new byte and compute BuzHash.

        Args:
            byte_in: New byte to process (0-255)

        Returns:
            Current BuzHash value
        """
        self.window.append(byte_in)

        if len(self.window) > self.window_size:
            # Remove oldest byte from hash
            byte_out = self.window.pop(0)
            self.hash_value ^= self.rotations[byte_out]

        # Add new byte: left rotate current hash and XOR with new byte's hash
        self.hash_value = self._left_rotate(self.hash_value, 1) ^ self.hash_table[byte_in]
        self.hash_value &= 0xFFFFFFFF  # Keep as 32-bit

        return self.hash_value

    def get_hash(self) -> int:
        """Get current hash value."""
        return self.hash_value

    def is_boundary(self, hash_value: int) -> bool:
        """
        Check if hash value indicates a chunk boundary.

        Args:
            hash_value: Current hash value

        Returns:
            True if this is a boundary position
        """
        return (hash_value & self.config.boundary_mask) == 0


@register_chunker("buzhash")
class BuzHashChunker(StreamableChunker, AdaptableChunker):
    """
    BuzHash Chunker for fast content-defined chunking.

    This chunker implements the BuzHash algorithm, which provides
    fast rolling hash computation for content-defined chunking
    with good distribution properties.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], BuzHashConfig]] = None,
        **kwargs
    ):
        """
        Initialize BuzHash chunker.

        Args:
            config: Configuration for the chunker
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="buzhash",
            category="general",
            supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
            **kwargs
        )

        # Parse configuration
        if isinstance(config, BuzHashConfig):
            self.config = config
        else:
            config = config or {}
            config.update(kwargs)
            self.config = BuzHashConfig(**{
                k: v for k, v in config.items()
                if k in BuzHashConfig.__dataclass_fields__
            })

        # Initialize hasher
        self.hasher = BuzHasher(self.config)

        # Statistics
        self.stats = {
            "chunks_created": 0,
            "bytes_processed": 0,
            "hash_computations": 0,
            "boundary_hits": 0,
            "avg_chunk_size": 0,
            "hash_distribution": {}
        } if self.config.enable_statistics else None

        # Adaptation history
        self._adaptation_history = []

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"BuzHash chunker initialized with window size {self.config.window_size}")

    def chunk(
        self,
        content: Union[str, bytes],
        source_info: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Chunk content using BuzHash algorithm.

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
                strategy_used="buzhash",
                processing_time=time.time() - start_time,
                source_info=source_info or {}
            )

        # Perform BuzHash chunking
        chunks = list(self._buzhash_chunk(content_bytes, is_text))

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
            strategy_used="buzhash",
            processing_time=processing_time,
            source_info=source_info or {}
        )

    def _buzhash_chunk(self, content: bytes, is_text: bool) -> Iterator[Chunk]:
        """
        Perform BuzHash chunking on content.

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
                id=f"buzhash_0",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=0,
                    length=len(content),
                    chunker_used="buzhash",
                    extra={
                        "chunk_index": 0,
                        "start_offset": 0,
                        "end_offset": len(content),
                        "algorithm": "buzhash",
                        "size": len(content),
                        "window_size": self.config.window_size
                    }
                )
            )
            return

        # Initialize hasher
        self.hasher.reset()

        chunk_start = 0
        chunk_index = 0
        boundary_positions = []

        # Process each byte
        for position in range(len(content)):
            byte_val = content[position]

            # Compute BuzHash
            hash_value = self.hasher.roll_byte(byte_val)

            if self.stats:
                self.stats["hash_computations"] += 1
                # Track hash distribution for analysis
                hash_bucket = hash_value & 0xFF
                self.stats["hash_distribution"][hash_bucket] = \
                    self.stats["hash_distribution"].get(hash_bucket, 0) + 1

            # Check for chunk boundary
            chunk_size = position - chunk_start + 1

            if ((self.hasher.is_boundary(hash_value) and
                 chunk_size >= self.config.min_chunk_size) or
                chunk_size >= self.config.max_chunk_size):

                boundary_positions.append(position)

                # Create chunk
                chunk_content = content[chunk_start:position + 1]
                if is_text:
                    chunk_content = chunk_content.decode('utf-8', errors='ignore')

                yield Chunk(
                    id=f"buzhash_{chunk_index}",
                    content=chunk_content,
                    modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                    metadata=ChunkMetadata(
                        source="unknown",
                        offset=chunk_start,
                        length=position - chunk_start + 1,
                        chunker_used="buzhash",
                        extra={
                            "chunk_index": chunk_index,
                            "start_offset": chunk_start,
                            "end_offset": position + 1,
                            "algorithm": "buzhash",
                            "hash_value": hash_value,
                            "size": chunk_size,
                            "boundary_mask": hex(self.config.boundary_mask),
                            "window_size": self.config.window_size,
                            "boundary_type": "hash" if self.hasher.is_boundary(hash_value) else "size"
                        }
                    )
                )

                if self.stats:
                    self.stats["boundary_hits"] += 1

                chunk_start = position + 1
                chunk_index += 1
                self.hasher.reset()

        # Handle remaining content
        if chunk_start < len(content):
            chunk_content = content[chunk_start:]
            if is_text:
                chunk_content = chunk_content.decode('utf-8', errors='ignore')

            yield Chunk(
                id=f"buzhash_{chunk_index}",
                content=chunk_content,
                modality=ModalityType.TEXT if is_text else ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source="unknown",
                    offset=chunk_start,
                    length=len(content) - chunk_start,
                    chunker_used="buzhash",
                    extra={
                        "chunk_index": chunk_index,
                        "start_offset": chunk_start,
                        "end_offset": len(content),
                        "algorithm": "buzhash",
                        "size": len(content) - chunk_start,
                        "window_size": self.config.window_size,
                        "boundary_type": "end"
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
        Chunk data from a stream using BuzHash.

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

        # Adapt boundary mask based on chunk size performance
        if "avg_chunk_size" in feedback:
            current_avg = feedback["avg_chunk_size"]
            target = self.config.target_chunk_size

            if abs(current_avg - target) > target * 0.2:  # 20% tolerance
                # Calculate new boundary mask
                ratio = current_avg / target

                if ratio > 1.3:  # Chunks too large
                    # Increase boundary probability by reducing mask
                    new_mask = max(0x7FF, self.config.boundary_mask >> 1)  # 11-bit minimum
                elif ratio < 0.7:  # Chunks too small
                    # Decrease boundary probability by increasing mask
                    new_mask = min(0x3FFFF, (self.config.boundary_mask << 1) | 1)  # 18-bit maximum
                else:
                    new_mask = self.config.boundary_mask

                if new_mask != self.config.boundary_mask:
                    adaptation_record["old_config"]["boundary_mask"] = hex(self.config.boundary_mask)
                    self.config.boundary_mask = new_mask
                    adaptation_record["new_config"]["boundary_mask"] = hex(new_mask)
                    adaptation_record["changes"].append("boundary_mask")
                    adapted = True

                    self.logger.info(f"Adapted boundary mask to {hex(new_mask)} for target size {target}")

        # Adapt window size based on performance feedback
        if "processing_time" in feedback and "content_size" in feedback:
            throughput = feedback["content_size"] / feedback["processing_time"]

            # If throughput is low, consider adjusting window size
            if throughput < 5_000_000:  # Less than 5MB/s
                if self.config.window_size > 32:
                    adaptation_record["old_config"]["window_size"] = self.config.window_size
                    self.config.window_size = max(32, self.config.window_size - 8)
                    adaptation_record["new_config"]["window_size"] = self.config.window_size
                    adaptation_record["changes"].append("window_size")
                    adapted = True

                    self.logger.info(f"Adapted window size to {self.config.window_size} for better performance")
            elif throughput > 20_000_000:  # Very fast, can afford larger window
                if self.config.window_size < 128:
                    adaptation_record["old_config"]["window_size"] = self.config.window_size
                    self.config.window_size = min(128, self.config.window_size + 8)
                    adaptation_record["new_config"]["window_size"] = self.config.window_size
                    adaptation_record["changes"].append("window_size")
                    adapted = True

                    self.logger.info(f"Adapted window size to {self.config.window_size} for better quality")

        if adapted:
            self.adaptation_history.append(adaptation_record)
            # Recreate hasher with new config
            self.hasher = BuzHasher(self.config)

        return adapted

    def get_adaptation_info(self) -> Dict[str, Any]:
        """Get information about current adaptations."""
        return {
            "config": {
                "window_size": self.config.window_size,
                "min_chunk_size": self.config.min_chunk_size,
                "max_chunk_size": self.config.max_chunk_size,
                "target_chunk_size": self.config.target_chunk_size,
                "boundary_mask": hex(self.config.boundary_mask),
                "hash_table_seed": self.config.hash_table_seed,
                "normalization": self.config.normalization
            },
            "statistics": self.stats,
            "adaptation_history": self._adaptation_history,
            "adaptation_count": len(self._adaptation_history)
        }

    def get_chunk_estimate(self, content_size: int) -> int:
        """Estimate number of chunks for given content size."""
        if content_size <= self.config.min_chunk_size:
            return 1

        # Estimate based on boundary mask probability
        boundary_probability = 1 / (self.config.boundary_mask + 1)
        expected_chunk_size = max(self.config.min_chunk_size, 1 / boundary_probability)

        estimated = max(1, int(content_size / expected_chunk_size))

        # Add variance for BuzHash variability
        variance = max(1, int(estimated * 0.25))  # ±25% variance
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

        # Boundary distribution quality
        hash_boundaries = sum(1 for chunk in chunks
                             if chunk.metadata.extra.get("boundary_type") == "hash")
        boundary_ratio = hash_boundaries / len(chunks) if chunks else 0
        boundary_score = boundary_ratio  # Higher is better for content-defined chunking

        # Performance score (BuzHash should be fast)
        performance_score = 0.9  # High baseline for BuzHash efficiency

        # Combined quality score
        return (size_score * 0.4 + boundary_score * 0.3 + performance_score * 0.3)

    def calculate_hash(self, content: bytes) -> int:
        """
        Calculate BuzHash for content.

        Args:
            content: Content to hash

        Returns:
            BuzHash value
        """
        self.hasher.reset()

        for byte_val in content:
            self.hasher.roll_byte(byte_val)

        return self.hasher.get_hash()

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """Adapt BuzHash parameters based on feedback."""
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
            self.config.boundary_mask = max(0x3FF, self.config.boundary_mask >> 1)
            adaptation["new_boundary_mask"] = self.config.boundary_mask
            if hasattr(self, 'logger'):
                self.logger.info(f"Adapted boundary_mask: {hex(old_mask)} -> {hex(self.config.boundary_mask)}")

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Increase mask for better performance
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
        """Describe the BuzHash algorithm."""
        return f"""
        BuzHash Chunking Algorithm:

        Window Size: {self.config.window_size} bytes
        Boundary Mask: {hex(self.config.boundary_mask)}
        Target Chunk Size: {self.config.target_chunk_size} bytes
        Size Range: {self.config.min_chunk_size} - {self.config.max_chunk_size} bytes

        BuzHash is a fast rolling hash algorithm that uses a combination
        of left rotation and XOR operations with a precomputed hash table.
        It provides good distribution properties while being computationally
        efficient compared to polynomial-based methods like Rabin fingerprinting.

        Algorithm Steps:
        1. Maintain a sliding window of bytes
        2. For each new byte: left-rotate current hash and XOR with byte's hash
        3. When removing old byte: XOR with its rotated hash value
        4. Check if hash satisfies boundary condition

        Advantages:
        - Fast computation (simpler than Rabin)
        - Good chunk size distribution
        - Low memory overhead
        - Deterministic boundaries

        Use Cases:
        - High-performance deduplication
        - Real-time chunking applications
        - Large file processing
        - Network optimization
        """
