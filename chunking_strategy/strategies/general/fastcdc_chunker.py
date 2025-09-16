"""
FastCDC (Fast Content-Defined Chunking) strategy.

FastCDC is an advanced content-defined chunking algorithm that creates variable-size
chunks based on content boundaries rather than fixed sizes. It uses rolling hash
functions to detect natural breakpoints in data, making it excellent for:
- Data deduplication
- Backup optimization
- Version control systems
- Content distribution networks
- Large file processing

Key features:
- Content-defined boundaries (not fixed size)
- Rolling hash for boundary detection
- Variable chunk sizes with configurable min/max limits
- Excellent deduplication ratios
- Fast performance compared to traditional CDC algorithms
- Works on any file type (binary, text, images, documents)
"""

import hashlib
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.core.streaming import StreamableChunker
from chunking_strategy.core.adaptive import AdaptableChunker


@register_chunker(
    name="fastcdc",
    category="general",
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    supported_formats=["any"],  # Universal - works on any file type
    dependencies=[],
    description="Fast Content-Defined Chunking for variable-size chunks with natural boundaries",
    use_cases=["deduplication", "backup", "version_control", "large_files", "binary_data"]
)
class FastCDCChunker(StreamableChunker, AdaptableChunker):
    """
    FastCDC (Fast Content-Defined Chunking) implementation.

    Creates variable-size chunks by detecting natural content boundaries using
    rolling hash functions. Particularly effective for data deduplication and
    processing of large or binary files.

    Features:
    - Content-defined chunk boundaries
    - Configurable min/max chunk sizes
    - Multiple rolling hash algorithms
    - Normalization for better boundary detection
    - Rich metadata with content fingerprints
    - Adaptive parameter tuning
    """

    def __init__(
        self,
        min_chunk_size: int = 2048,      # 2KB minimum
        max_chunk_size: int = 65536,     # 64KB maximum
        avg_chunk_size: int = 8192,      # 8KB average target
        hash_algorithm: str = "gear",    # "gear", "rabin", "buzhash"
        normalization: bool = True,      # FastCDC normalization
        window_size: int = 48,           # Rolling hash window
        mask_bits: int = 13,             # Hash mask (affects avg chunk size)
        **kwargs
    ):
        """
        Initialize FastCDC chunker.

        Args:
            min_chunk_size: Minimum chunk size in bytes
            max_chunk_size: Maximum chunk size in bytes
            avg_chunk_size: Target average chunk size
            hash_algorithm: Rolling hash algorithm ("gear", "rabin", "buzhash")
            normalization: Enable FastCDC normalization for better boundaries
            window_size: Size of rolling hash window
            mask_bits: Number of bits in hash mask (affects chunk size distribution)
            **kwargs: Additional parameters
        """
        super().__init__(
            name="fastcdc",
            category="general",
            supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
            **kwargs
        )

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.avg_chunk_size = avg_chunk_size
        self.hash_algorithm = hash_algorithm
        self.normalization = normalization
        self.window_size = window_size
        self.mask_bits = mask_bits

        # Calculate hash mask for target average chunk size
        self.hash_mask = (1 << mask_bits) - 1

        # Initialize hash algorithm
        self._init_hash_algorithm()

        self.logger = logging.getLogger(__name__)
        self._adaptation_history = []

    def _init_hash_algorithm(self):
        """Initialize the rolling hash algorithm."""
        if self.hash_algorithm == "gear":
            self._gear_table = self._generate_gear_table()
        elif self.hash_algorithm == "rabin":
            self._rabin_poly = 0x3DA3358B4DC173  # Rabin polynomial
        elif self.hash_algorithm == "buzhash":
            self._buzz_table = self._generate_buzz_table()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")

    def _generate_gear_table(self) -> List[int]:
        """Generate gear hash table for fast lookups."""
        table = []
        for i in range(256):
            # Generate pseudo-random values for gear hash
            value = i
            for _ in range(8):
                if value & 1:
                    value = (value >> 1) ^ 0xEDB88320
                else:
                    value >>= 1
            table.append(value)
        return table

    def _generate_buzz_table(self) -> List[int]:
        """Generate BuzHash table for fast lookups."""
        table = []
        for i in range(256):
            # Simple pseudo-random generation for BuzHash
            value = i * 0x9E3779B9  # Golden ratio hash
            table.append(value & 0xFFFFFFFF)
        return table

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk content using FastCDC algorithm.

        Args:
            content: Content to chunk (string, bytes, or file path)
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with variable-size content-defined chunks
        """
        start_time = time.time()

        # Handle input types
        if isinstance(content, Path):
            source_path = content
            with open(content, 'rb') as f:
                data = f.read()
        elif isinstance(content, str) and len(content) > 0 and len(content) < 500 and '\n' not in content:
            try:
                if Path(content).exists() and Path(content).is_file():
                    source_path = Path(content)
                    with open(source_path, 'rb') as f:
                        data = f.read()
                else:
                    source_path = None
                    data = content.encode('utf-8')
            except (OSError, ValueError):
                source_path = None
                data = content.encode('utf-8')
        elif isinstance(content, str):
            source_path = None
            data = content.encode('utf-8')
        else:
            source_path = None
            data = bytes(content)

        try:
            # Perform FastCDC chunking
            chunks = self._fastcdc_chunk(data, source_path or "direct_input")

            # Create chunking result
            result = ChunkingResult(
                chunks=chunks,
                processing_time=time.time() - start_time,
                strategy_used="fastcdc",
                source_info={
                    "source_file": str(source_path) if source_path else "direct_input",
                    "algorithm": self.hash_algorithm,
                    "min_chunk_size": self.min_chunk_size,
                    "max_chunk_size": self.max_chunk_size,
                    "avg_chunk_size": self.avg_chunk_size,
                    "total_bytes": len(data),
                    "normalization": self.normalization
                }
            )

            self.logger.info(f"FastCDC chunking completed: {len(chunks)} chunks from {len(data)} bytes")
            return result

        except Exception as e:
            self.logger.error(f"Error in FastCDC chunking: {e}")
            # Fallback to fixed-size chunking
            return self._fallback_fixed_chunking(data, source_path, start_time)

    def _fastcdc_chunk(self, data: bytes, source: str) -> List[Chunk]:
        """
        Perform FastCDC chunking on data.

        Args:
            data: Bytes to chunk
            source: Source identifier

        Returns:
            List of chunks with content-defined boundaries
        """
        chunks = []
        data_len = len(data)

        if data_len == 0:
            return chunks

        chunk_start = 0
        chunk_id = 0

        while chunk_start < data_len:
            # Find next chunk boundary
            chunk_end = self._find_chunk_boundary(data, chunk_start, data_len)

            # Extract chunk data
            chunk_data = data[chunk_start:chunk_end]

            # Create chunk with metadata
            chunk = self._create_chunk(
                chunk_data, chunk_id, chunk_start, chunk_end, source
            )
            chunks.append(chunk)

            chunk_start = chunk_end
            chunk_id += 1

        return chunks

    def _find_chunk_boundary(self, data: bytes, start: int, data_len: int) -> int:
        """
        Find the next chunk boundary using FastCDC algorithm.

        Args:
            data: Full data bytes
            start: Starting position
            data_len: Total data length

        Returns:
            Position of next chunk boundary
        """
        pos = start
        min_pos = min(start + self.min_chunk_size, data_len)
        max_pos = min(start + self.max_chunk_size, data_len)

        # If we can't reach minimum size, return end
        if min_pos >= data_len:
            return data_len

        # Initialize rolling hash
        hash_val = 0
        if self.hash_algorithm == "gear":
            hash_val = self._init_gear_hash(data, pos, min_pos)
        elif self.hash_algorithm == "rabin":
            hash_val = self._init_rabin_hash(data, pos, min_pos)
        elif self.hash_algorithm == "buzhash":
            hash_val = self._init_buzz_hash(data, pos, min_pos)

        pos = min_pos

        # FastCDC normalization zones
        if self.normalization:
            normal_size = self.avg_chunk_size
            backup_pos = None

            # Look for boundary in normal zone first
            while pos < min(start + normal_size * 2, max_pos):
                if self._is_boundary(hash_val, pos - start):
                    return pos

                # Store backup boundary in backup zone
                if pos >= start + normal_size and backup_pos is None:
                    if self._is_backup_boundary(hash_val):
                        backup_pos = pos

                if pos < data_len - 1:
                    hash_val = self._update_hash(hash_val, data, pos)
                    pos += 1
                else:
                    break

            # Use backup boundary if found
            if backup_pos is not None:
                return backup_pos
        else:
            # Standard CDC without normalization
            while pos < max_pos:
                if self._is_boundary(hash_val, pos - start):
                    return pos

                if pos < data_len - 1:
                    hash_val = self._update_hash(hash_val, data, pos)
                    pos += 1
                else:
                    break

        # Force boundary at max size
        return max_pos

    def _init_gear_hash(self, data: bytes, start: int, min_pos: int) -> int:
        """Initialize gear hash for window."""
        hash_val = 0
        for i in range(start, min(min_pos, len(data))):
            hash_val = (hash_val << 1) + self._gear_table[data[i]]
        return hash_val & 0xFFFFFFFF

    def _init_rabin_hash(self, data: bytes, start: int, min_pos: int) -> int:
        """Initialize Rabin hash for window."""
        hash_val = 0
        for i in range(start, min(min_pos, len(data))):
            hash_val = (hash_val * 256 + data[i]) % self._rabin_poly
        return hash_val

    def _init_buzz_hash(self, data: bytes, start: int, min_pos: int) -> int:
        """Initialize BuzHash for window."""
        hash_val = 0
        for i in range(start, min(min_pos, len(data))):
            hash_val = ((hash_val << 1) | (hash_val >> 31)) ^ self._buzz_table[data[i]]
        return hash_val & 0xFFFFFFFF

    def _update_hash(self, hash_val: int, data: bytes, pos: int) -> int:
        """Update rolling hash with next byte."""
        if pos >= len(data):
            return hash_val

        if self.hash_algorithm == "gear":
            return ((hash_val << 1) + self._gear_table[data[pos]]) & 0xFFFFFFFF
        elif self.hash_algorithm == "rabin":
            return (hash_val * 256 + data[pos]) % self._rabin_poly
        elif self.hash_algorithm == "buzhash":
            return (((hash_val << 1) | (hash_val >> 31)) ^ self._buzz_table[data[pos]]) & 0xFFFFFFFF

        return hash_val

    def _is_boundary(self, hash_val: int, chunk_size: int) -> bool:
        """Check if current position is a chunk boundary."""
        return (hash_val & self.hash_mask) == 0

    def _is_backup_boundary(self, hash_val: int) -> bool:
        """Check for backup boundary (higher threshold)."""
        backup_mask = (1 << (self.mask_bits - 1)) - 1
        return (hash_val & backup_mask) == 0

    def _create_chunk(
        self,
        chunk_data: bytes,
        chunk_id: int,
        start_pos: int,
        end_pos: int,
        source: str
    ) -> Chunk:
        """Create a chunk with metadata."""

        # Calculate content fingerprints
        md5_hash = hashlib.md5(chunk_data).hexdigest()
        sha256_hash = hashlib.sha256(chunk_data).hexdigest()

        # Analyze content characteristics
        content_analysis = self._analyze_content(chunk_data)

        # Try to decode as text for text-like content
        try:
            if content_analysis['likely_text']:
                content_str = chunk_data.decode('utf-8', errors='ignore')
                modality = ModalityType.TEXT
            else:
                content_str = f"<binary data: {len(chunk_data)} bytes>"
                modality = ModalityType.MIXED
        except UnicodeDecodeError:
            content_str = f"<binary data: {len(chunk_data)} bytes>"
            modality = ModalityType.MIXED

        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_byte": start_pos, "end_byte": end_pos},
            chunker_used="fastcdc",
            extra={
                "chunk_type": "content_defined",
                "algorithm": self.hash_algorithm,
                "chunk_size": len(chunk_data),
                "is_text": content_analysis['likely_text'],
                "entropy": content_analysis['entropy'],
                "compressibility": content_analysis['compressibility'],
                "md5_hash": md5_hash,
                "sha256_hash": sha256_hash,
                "content_fingerprint": md5_hash[:16],  # Short fingerprint
                "boundary_type": "content_defined",
                "normalization_used": self.normalization,
                "min_size": self.min_chunk_size,
                "max_size": self.max_chunk_size,
                "avg_target": self.avg_chunk_size
            }
        )

        return Chunk(
            id=f"fastcdc_{chunk_id}_{start_pos}_{end_pos}",
            content=content_str,
            modality=modality,
            metadata=chunk_metadata
        )

    def _analyze_content(self, data: bytes) -> Dict[str, Any]:
        """Analyze content characteristics."""
        if len(data) == 0:
            return {
                'likely_text': False,
                'entropy': 0.0,
                'compressibility': 1.0
            }

        # Check if content is likely text
        try:
            text = data.decode('utf-8')
            printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
            text_ratio = printable_chars / len(text) if len(text) > 0 else 0
            likely_text = text_ratio > 0.8
        except UnicodeDecodeError:
            likely_text = False

        # Calculate entropy (measure of randomness)
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1

        entropy = 0.0
        data_len = len(data)
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability) if probability > 0 else 0

        # Estimate compressibility (inverse of entropy)
        max_entropy = 8.0  # Maximum entropy for 8-bit data
        compressibility = 1.0 - (entropy / max_entropy) if entropy > 0 else 1.0

        return {
            'likely_text': likely_text,
            'entropy': entropy,
            'compressibility': compressibility
        }

    def _fallback_fixed_chunking(self, data: bytes, source_path: Optional[Path], start_time: float) -> ChunkingResult:
        """Fallback to fixed-size chunking if FastCDC fails."""
        chunks = []
        chunk_size = self.avg_chunk_size

        for i in range(0, len(data), chunk_size):
            chunk_data = data[i:i + chunk_size]

            chunk = Chunk(
                id=f"fastcdc_fallback_{i // chunk_size}",
                content=chunk_data.decode('utf-8', errors='ignore') if len(chunk_data) > 0 else "",
                modality=ModalityType.MIXED,
                metadata=ChunkMetadata(
                    source=str(source_path) if source_path else "direct_input",
                    position={"start_byte": i, "end_byte": i + len(chunk_data)},
                    chunker_used="fastcdc",
                    extra={
                        "chunk_type": "fixed_size_fallback",
                        "chunk_size": len(chunk_data),
                        "fallback": True
                    }
                )
            )
            chunks.append(chunk)

        return ChunkingResult(
            chunks=chunks,
            processing_time=time.time() - start_time,
            strategy_used="fastcdc_fallback"
        )

    def chunk_stream(self, content_stream, **kwargs):
        """Chunk data from a stream using FastCDC."""
        # Collect stream data
        data_chunks = []
        for chunk in content_stream:
            if isinstance(chunk, str):
                data_chunks.append(chunk.encode('utf-8'))
            elif isinstance(chunk, bytes):
                data_chunks.append(chunk)

        data = b''.join(data_chunks)
        result = self.chunk(data, **kwargs)

        for chunk in result.chunks:
            yield chunk

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ["any"]  # FastCDC works on any file type

    def estimate_chunks(self, content: Union[str, bytes, Path]) -> int:
        """Estimate number of chunks that will be generated."""
        if isinstance(content, Path) and content.exists():
            size = content.stat().st_size
        elif isinstance(content, str):
            size = len(content.encode('utf-8'))
        elif isinstance(content, bytes):
            size = len(content)
        else:
            size = len(str(content).encode('utf-8'))

        # Estimate based on average chunk size
        return max(1, size // self.avg_chunk_size)

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """Adapt FastCDC parameters based on feedback."""
        adaptation = {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "old_avg_chunk_size": self.avg_chunk_size,
            "old_mask_bits": self.mask_bits
        }

        if feedback_type == "deduplication" and feedback_score < 0.5:
            # Increase chunk size for better deduplication
            old_avg = self.avg_chunk_size
            self.avg_chunk_size = min(32768, int(self.avg_chunk_size * 1.5))
            adaptation["new_avg_chunk_size"] = self.avg_chunk_size
            self.logger.info(f"Adapted avg_chunk_size for deduplication: {old_avg} -> {self.avg_chunk_size}")

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Decrease chunk size for better performance
            old_avg = self.avg_chunk_size
            self.avg_chunk_size = max(2048, int(self.avg_chunk_size * 0.8))
            adaptation["new_avg_chunk_size"] = self.avg_chunk_size
            self.logger.info(f"Adapted avg_chunk_size for performance: {old_avg} -> {self.avg_chunk_size}")

        elif feedback_type == "quality" and feedback_score < 0.5:
            # Adjust mask bits for better boundary detection
            old_mask = self.mask_bits
            if self.avg_chunk_size > 16384:
                self.mask_bits = min(15, self.mask_bits + 1)
            else:
                self.mask_bits = max(10, self.mask_bits - 1)
            self.hash_mask = (1 << self.mask_bits) - 1
            adaptation["new_mask_bits"] = self.mask_bits
            self.logger.info(f"Adapted mask_bits for quality: {old_mask} -> {self.mask_bits}")

        self._adaptation_history.append(adaptation)

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self._adaptation_history.copy()
