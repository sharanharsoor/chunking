"""
Streaming chunking capabilities for processing large files without loading them entirely into memory.

This module provides streaming interfaces that allow chunking of large files by
reading and processing them in blocks, yielding chunks as they are generated.
"""

import io
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union, BinaryIO, TextIO, Callable
import mmap
import os
import threading
from queue import Queue, Empty

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ModalityType,
    StreamableChunker
)
from chunking_strategy.core.registry import get_chunker, create_chunker

logger = logging.getLogger(__name__)


@dataclass
class StreamingProgress:
    """Detailed progress information for streaming operations."""
    file_path: str
    total_size: int
    processed_bytes: int
    chunks_generated: int
    current_chunk_id: Optional[str]
    throughput_mbps: float
    elapsed_time: float
    eta_seconds: Optional[float]
    status: str  # "processing", "paused", "completed", "error"
    error_message: Optional[str] = None

    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total_size == 0:
            return 100.0
        return min(100.0, (self.processed_bytes / self.total_size) * 100)

    @property
    def chunks_per_second(self) -> float:
        """Get chunks generated per second."""
        if self.elapsed_time == 0:
            return 0.0
        return self.chunks_generated / self.elapsed_time


@dataclass
class StreamingCheckpoint:
    """Checkpoint data for resumable streaming operations."""
    file_path: str
    file_size: int
    file_hash: str  # For integrity verification
    last_processed_offset: int
    chunks_generated: int
    strategy_name: str
    strategy_params: Dict[str, Any]
    streaming_config: Dict[str, Any]
    timestamp: float
    version: str = "1.0"

    def save_to_file(self, checkpoint_path: Path) -> None:
        """Save checkpoint to file."""
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, checkpoint_path: Path) -> 'StreamingCheckpoint':
        """Load checkpoint from file."""
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)

    def is_valid_for_file(self, file_path: Path) -> bool:
        """Check if checkpoint is valid for given file."""
        if not file_path.exists():
            return False

        current_size = file_path.stat().st_size
        if current_size != self.file_size:
            return False

        # Check file hash for integrity
        import hashlib
        hash_obj = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Sample first and last 64KB for quick verification
            hash_obj.update(f.read(65536))
            f.seek(max(0, current_size - 65536))
            hash_obj.update(f.read(65536))

        return hash_obj.hexdigest() == self.file_hash


@dataclass
class DistributedStreamingResult:
    """Result from distributed streaming across multiple files."""
    total_files: int
    completed_files: int
    failed_files: int
    total_chunks: int
    total_processing_time: float
    total_size_mb: float
    average_throughput_mbps: float
    file_results: Dict[str, ChunkingResult] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_files == 0:
            return 100.0
        return (self.completed_files / self.total_files) * 100


class StreamingChunker:
    """
    Wrapper that provides streaming capabilities for any chunker.

    This class can wrap existing chunkers and provide streaming processing
    for large files, reading content in configurable blocks and yielding
    chunks as they are generated.

    Examples:
        Basic streaming:
        ```python
        streamer = StreamingChunker("fixed_size", chunk_size=1024)
        for chunk in streamer.stream_file("large_file.txt"):
            process(chunk)
        ```

        With custom block size:
        ```python
        streamer = StreamingChunker(
            "sentence_based",
            block_size=64*1024,  # 64KB blocks
            overlap_size=1024     # 1KB overlap
        )
        chunks = list(streamer.stream_file("document.pdf"))
        ```
    """

    def __init__(
        self,
        strategy: Union[str, BaseChunker],
        block_size: int = 64 * 1024,  # 64KB default
        overlap_size: int = 0,         # No overlap by default
        buffer_size: int = 8 * 1024,   # 8KB read buffer
        progress_callback: Optional[Callable[[StreamingProgress], None]] = None,
        checkpoint_dir: Optional[Path] = None,
        enable_checkpointing: bool = True,
        progress_update_interval: int = 1000,  # Update every N chunks
        **chunker_kwargs
    ):
        """
        Initialize streaming chunker.

        Args:
            strategy: Chunker name or instance to use
            block_size: Size of blocks to read from file
            overlap_size: Overlap between consecutive blocks
            buffer_size: Buffer size for file reading
            progress_callback: Optional callback for progress updates
            checkpoint_dir: Directory for storing checkpoints (None = disabled)
            enable_checkpointing: Enable automatic checkpointing
            progress_update_interval: Update progress every N chunks
            **chunker_kwargs: Additional arguments for chunker creation
        """
        self.block_size = block_size
        self.overlap_size = overlap_size
        self.buffer_size = buffer_size
        self.progress_callback = progress_callback
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.enable_checkpointing = enable_checkpointing and checkpoint_dir is not None
        self.progress_update_interval = progress_update_interval

        # Progress tracking
        self._current_progress: Optional[StreamingProgress] = None
        self._progress_lock = threading.Lock()
        self._start_time = 0.0
        self._bytes_processed = 0
        self._chunks_generated = 0

        # Initialize chunker
        if isinstance(strategy, str):
            self.chunker = create_chunker(strategy, **chunker_kwargs)
            if not self.chunker:
                raise ValueError(f"Unknown chunker strategy: {strategy}")
            self.strategy_name = strategy
            self.strategy_params = chunker_kwargs
        elif isinstance(strategy, BaseChunker):
            self.chunker = strategy
            self.strategy_name = getattr(strategy, 'name', str(type(strategy).__name__))
            self.strategy_params = {}
        else:
            raise TypeError("Strategy must be chunker name (str) or BaseChunker instance")

        # Create checkpoint directory if needed
        if self.enable_checkpointing:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def stream_file(
        self,
        file_path: Union[str, Path],
        encoding: Optional[str] = None,
        resume_from_checkpoint: bool = True,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream chunks from a file with enhanced progress reporting and checkpointing.

        Args:
            file_path: Path to the file to process
            encoding: Text encoding (for text files)
            resume_from_checkpoint: Try to resume from existing checkpoint
            **kwargs: Additional arguments passed to chunker

        Yields:
            Individual chunks as they are generated

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.error(f"🚫 Streaming failed: File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        self.logger.info(f"🚀 Starting streaming: {file_path} ({file_size:,} bytes)")

        # Initialize progress tracking
        self._start_time = time.time()
        self._bytes_processed = 0
        self._chunks_generated = 0

        # Check for existing checkpoint
        checkpoint_path = None
        start_offset = 0

        if self.enable_checkpointing and resume_from_checkpoint:
            checkpoint_path = self._get_checkpoint_path(file_path)
            if checkpoint_path.exists():
                try:
                    checkpoint = StreamingCheckpoint.load_from_file(checkpoint_path)
                    if checkpoint.is_valid_for_file(file_path):
                        start_offset = checkpoint.last_processed_offset
                        self._chunks_generated = checkpoint.chunks_generated
                        self.logger.info(f"📍 Resuming from checkpoint at offset {start_offset:,} ({checkpoint.chunks_generated} chunks)")
                    else:
                        self.logger.warning("🔄 Checkpoint invalid, starting fresh")
                        checkpoint_path.unlink(missing_ok=True)
                except Exception as e:
                    self.logger.error(f"❌ Failed to load checkpoint: {e}")
                    checkpoint_path.unlink(missing_ok=True)

        # Initialize progress
        self._update_progress(
            file_path=str(file_path),
            total_size=file_size,
            status="processing"
        )

        try:
            # Determine if we should treat as text or binary
            is_text = self._should_treat_as_text(file_path, encoding)

            if is_text:
                yield from self._stream_text_file_enhanced(file_path, encoding, start_offset, **kwargs)
            else:
                yield from self._stream_binary_file_enhanced(file_path, start_offset, **kwargs)

            # Mark as completed
            self._update_progress(
                file_path=str(file_path),
                total_size=file_size,
                status="completed"
            )
            self.logger.info(f"✅ Completed streaming: {self._chunks_generated} chunks in {time.time() - self._start_time:.2f}s")

            # Clean up checkpoint on successful completion
            if checkpoint_path and checkpoint_path.exists():
                checkpoint_path.unlink(missing_ok=True)

        except Exception as e:
            self._update_progress(
                file_path=str(file_path),
                total_size=file_size,
                status="error",
                error_message=str(e)
            )
            self.logger.error(f"💥 Streaming failed: {e}")
            raise

    def stream_content(
        self,
        content: Union[str, bytes],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream chunks from in-memory content.

        Args:
            content: Content to process
            source_info: Information about the content source
            **kwargs: Additional arguments passed to chunker

        Yields:
            Individual chunks as they are generated
        """
        if isinstance(content, str):
            yield from self._stream_text_content(content, source_info, **kwargs)
        else:
            yield from self._stream_binary_content(content, source_info, **kwargs)

    def stream_from_iterator(
        self,
        content_iterator: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream chunks from a content iterator.

        This is useful for processing content that comes from streams,
        network connections, or other iterative sources.

        Args:
            content_iterator: Iterator yielding content pieces
            source_info: Information about the content source
            **kwargs: Additional arguments passed to chunker

        Yields:
            Individual chunks as they are generated
        """
        # Check if chunker supports native streaming
        if isinstance(self.chunker, StreamableChunker):
            yield from self.chunker.chunk_stream(content_iterator, source_info, **kwargs)
            return

        # Buffer content and process in blocks
        buffer = ""
        overlap_buffer = ""
        chunk_counter = 0

        for piece in content_iterator:
            if isinstance(piece, bytes):
                try:
                    piece = piece.decode('utf-8')
                except UnicodeDecodeError:
                    self.logger.warning("Failed to decode bytes to text, skipping piece")
                    continue

            buffer += piece

            # Process when buffer reaches block size
            while len(buffer) >= self.block_size:
                block = overlap_buffer + buffer[:self.block_size]

                # Process block
                try:
                    result = self.chunker.chunk(
                        block,
                        source_info=source_info,
                        **kwargs
                    )

                    for chunk in result.chunks:
                        chunk.id = f"stream_{chunk_counter}_{chunk.id}"
                        chunk_counter += 1
                        yield chunk

                except Exception as e:
                    self.logger.error(f"Error processing block: {e}")

                # Setup overlap for next block
                if self.overlap_size > 0:
                    overlap_buffer = buffer[self.block_size - self.overlap_size:self.block_size]
                else:
                    overlap_buffer = ""

                buffer = buffer[self.block_size:]

        # Process remaining content
        if buffer.strip():
            final_block = overlap_buffer + buffer
            try:
                result = self.chunker.chunk(
                    final_block,
                    source_info=source_info,
                    **kwargs
                )

                for chunk in result.chunks:
                    chunk.id = f"stream_{chunk_counter}_{chunk.id}"
                    chunk_counter += 1
                    yield chunk

            except Exception as e:
                self.logger.error(f"Error processing final block: {e}")

    def _stream_text_file(
        self,
        file_path: Path,
        encoding: Optional[str],
        **kwargs
    ) -> Iterator[Chunk]:
        """Stream process a text file."""
        source_info = {
            "source": str(file_path),
            "source_type": "file",
            "encoding": encoding or "utf-8"
        }

        try:
            with open(file_path, 'r', encoding=encoding or 'utf-8') as f:
                def read_blocks():
                    while True:
                        block = f.read(self.buffer_size)
                        if not block:
                            break
                        yield block

                yield from self.stream_from_iterator(read_blocks(), source_info, **kwargs)

        except UnicodeDecodeError as e:
            self.logger.error(f"Encoding error reading {file_path}: {e}")
            # Fallback to binary mode
            yield from self._stream_binary_file(file_path, **kwargs)

    def _stream_binary_file(
        self,
        file_path: Path,
        **kwargs
    ) -> Iterator[Chunk]:
        """Stream process a binary file."""
        source_info = {
            "source": str(file_path),
            "source_type": "file",
            "mime_type": self._guess_mime_type(file_path)
        }

        with open(file_path, 'rb') as f:
            def read_blocks():
                while True:
                    block = f.read(self.buffer_size)
                    if not block:
                        break
                    yield block

            yield from self.stream_from_iterator(read_blocks(), source_info, **kwargs)

    def _stream_text_content(
        self,
        content: str,
        source_info: Optional[Dict[str, Any]],
        **kwargs
    ) -> Iterator[Chunk]:
        """Stream process text content."""
        def content_blocks():
            for i in range(0, len(content), self.buffer_size):
                yield content[i:i + self.buffer_size]

        yield from self.stream_from_iterator(content_blocks(), source_info, **kwargs)

    def _stream_binary_content(
        self,
        content: bytes,
        source_info: Optional[Dict[str, Any]],
        **kwargs
    ) -> Iterator[Chunk]:
        """Stream process binary content."""
        def content_blocks():
            for i in range(0, len(content), self.buffer_size):
                yield content[i:i + self.buffer_size]

        yield from self.stream_from_iterator(content_blocks(), source_info, **kwargs)

    def _should_treat_as_text(self, file_path: Path, encoding: Optional[str]) -> bool:
        """Determine if file should be treated as text."""
        # If encoding is specified, treat as text
        if encoding:
            return True

        # Check file extension
        text_extensions = {
            '.txt', '.md', '.rst', '.csv', '.json', '.xml', '.html', '.htm',
            '.py', '.js', '.css', '.java', '.cpp', '.h', '.c', '.php', '.rb',
            '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.log'
        }

        if file_path.suffix.lower() in text_extensions:
            return True

        # Try to read a small sample to detect if it's text
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(1024)
                sample.decode('utf-8')
                return True
        except (UnicodeDecodeError, IOError):
            return False

    def _guess_mime_type(self, file_path: Path) -> str:
        """Guess MIME type from file extension."""
        extension_map = {
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.html': 'text/html',
            '.xml': 'application/xml',
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
        }

        return extension_map.get(file_path.suffix.lower(), 'application/octet-stream')

    def collect_all(self, *args, **kwargs) -> ChunkingResult:
        """
        Collect all streamed chunks into a ChunkingResult.

        This is a convenience method for when you want to use streaming
        internally but need a complete result.

        Args:
            *args: Arguments passed to streaming method
            **kwargs: Keyword arguments passed to streaming method

        Returns:
            ChunkingResult with all collected chunks
        """
        import time
        start_time = time.time()

        chunks = list(self.stream_file(*args, **kwargs))
        processing_time = time.time() - start_time

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used=self.chunker.name,
            source_info=kwargs.get('source_info')
        )

    def _update_progress(
        self,
        file_path: str = "",
        total_size: int = 0,
        status: str = "",
        error_message: Optional[str] = None
    ) -> None:
        """Update progress tracking and trigger callback if provided."""
        if not self.progress_callback:
            return

        with self._progress_lock:
            elapsed_time = time.time() - self._start_time

            # Calculate throughput
            throughput_mbps = 0.0
            if elapsed_time > 0 and self._bytes_processed > 0:
                throughput_mbps = (self._bytes_processed / (1024 * 1024)) / elapsed_time

            # Calculate ETA
            eta_seconds = None
            if throughput_mbps > 0 and total_size > 0 and self._bytes_processed > 0:
                remaining_bytes = total_size - self._bytes_processed
                remaining_mb = remaining_bytes / (1024 * 1024)
                eta_seconds = remaining_mb / throughput_mbps

            self._current_progress = StreamingProgress(
                file_path=file_path or self._current_progress.file_path if self._current_progress else "",
                total_size=total_size or self._current_progress.total_size if self._current_progress else 0,
                processed_bytes=self._bytes_processed,
                chunks_generated=self._chunks_generated,
                current_chunk_id=f"chunk_{self._chunks_generated}",
                throughput_mbps=throughput_mbps,
                elapsed_time=elapsed_time,
                eta_seconds=eta_seconds,
                status=status or self._current_progress.status if self._current_progress else "unknown",
                error_message=error_message
            )

            # Trigger callback
            try:
                self.progress_callback(self._current_progress)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    def _get_checkpoint_path(self, file_path: Path) -> Path:
        """Get checkpoint file path for given file."""
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory not configured")

        # Create a safe filename from the original file path
        safe_name = str(file_path).replace("/", "_").replace("\\", "_")
        checkpoint_filename = f"{safe_name}_{self.strategy_name}.checkpoint"
        return self.checkpoint_dir / checkpoint_filename

    def _create_checkpoint(self, file_path: Path, offset: int) -> None:
        """Create checkpoint at current position."""
        if not self.enable_checkpointing:
            return

        try:
            # Calculate file hash for integrity check
            import hashlib
            hash_obj = hashlib.md5()
            with open(file_path, 'rb') as f:
                # Sample first and last 64KB for quick verification
                hash_obj.update(f.read(65536))
                file_size = f.seek(0, 2)  # Seek to end
                f.seek(max(0, file_size - 65536))
                hash_obj.update(f.read(65536))

            checkpoint = StreamingCheckpoint(
                file_path=str(file_path),
                file_size=file_size,
                file_hash=hash_obj.hexdigest(),
                last_processed_offset=offset,
                chunks_generated=self._chunks_generated,
                strategy_name=self.strategy_name,
                strategy_params=self.strategy_params,
                streaming_config={
                    'block_size': self.block_size,
                    'overlap_size': self.overlap_size,
                    'buffer_size': self.buffer_size
                },
                timestamp=time.time()
            )

            checkpoint_path = self._get_checkpoint_path(file_path)
            checkpoint.save_to_file(checkpoint_path)
            self.logger.debug(f"💾 Checkpoint saved at offset {offset:,}")

        except Exception as e:
            self.logger.warning(f"Failed to create checkpoint: {e}")

    def _stream_text_file_enhanced(
        self,
        file_path: Path,
        encoding: Optional[str],
        start_offset: int = 0,
        **kwargs
    ) -> Iterator[Chunk]:
        """Enhanced streaming for text files with progress and checkpointing."""
        source_info = {
            "source": str(file_path),
            "source_type": "file",
            "encoding": encoding or "utf-8"
        }

        try:
            with open(file_path, 'r', encoding=encoding or 'utf-8') as f:
                # Skip to start position if resuming
                if start_offset > 0:
                    f.seek(start_offset)

                buffer = ""
                overlap_buffer = ""
                chunk_counter = self._chunks_generated

                while True:
                    # Read block
                    block = f.read(self.buffer_size)
                    if not block:
                        break

                    buffer += block
                    self._bytes_processed = f.tell()

                    # Process when buffer reaches block size
                    while len(buffer) >= self.block_size:
                        content_block = overlap_buffer + buffer[:self.block_size]

                        # Process block
                        try:
                            result = self.chunker.chunk(content_block, source_info=source_info, **kwargs)

                            for chunk in result.chunks:
                                chunk.id = f"stream_{chunk_counter}_{chunk.id}"
                                chunk.metadata.stream_offset = f.tell() - len(buffer) + len(content_block)
                                chunk_counter += 1
                                self._chunks_generated = chunk_counter
                                yield chunk

                                # Update progress periodically
                                if chunk_counter % self.progress_update_interval == 0:
                                    self._update_progress(
                                        file_path=str(file_path),
                                        total_size=file_path.stat().st_size,
                                        status="processing"
                                    )

                                    # Create checkpoint periodically
                                    if self.enable_checkpointing and chunk_counter % (self.progress_update_interval * 5) == 0:
                                        self._create_checkpoint(file_path, f.tell())

                        except Exception as e:
                            self.logger.error(f"Error processing block: {e}")

                        # Setup overlap for next block
                        if self.overlap_size > 0:
                            overlap_buffer = buffer[self.block_size - self.overlap_size:self.block_size]
                        else:
                            overlap_buffer = ""

                        buffer = buffer[self.block_size:]

                # Process remaining content
                if buffer.strip():
                    final_content = overlap_buffer + buffer
                    try:
                        result = self.chunker.chunk(final_content, source_info=source_info, **kwargs)

                        for chunk in result.chunks:
                            chunk.id = f"stream_{chunk_counter}_{chunk.id}"
                            chunk.metadata.stream_offset = f.tell()
                            chunk_counter += 1
                            self._chunks_generated = chunk_counter
                            yield chunk

                    except Exception as e:
                        self.logger.error(f"Error processing final block: {e}")

        except UnicodeDecodeError as e:
            self.logger.error(f"Encoding error reading {file_path}: {e}")
            # Fallback to binary mode
            yield from self._stream_binary_file_enhanced(file_path, start_offset, **kwargs)

    def _stream_binary_file_enhanced(
        self,
        file_path: Path,
        start_offset: int = 0,
        **kwargs
    ) -> Iterator[Chunk]:
        """Enhanced streaming for binary files with progress and checkpointing."""
        source_info = {
            "source": str(file_path),
            "source_type": "file",
            "mime_type": self._guess_mime_type(file_path)
        }

        with open(file_path, 'rb') as f:
            # Skip to start position if resuming
            if start_offset > 0:
                f.seek(start_offset)

            buffer = b""
            overlap_buffer = b""
            chunk_counter = self._chunks_generated

            while True:
                # Read block
                block = f.read(self.buffer_size)
                if not block:
                    break

                buffer += block
                self._bytes_processed = f.tell()

                # Process when buffer reaches block size
                while len(buffer) >= self.block_size:
                    content_block = overlap_buffer + buffer[:self.block_size]

                    # Try to decode as text for processing
                    try:
                        text_content = content_block.decode('utf-8', errors='ignore')
                        process_content = text_content
                    except:
                        process_content = content_block

                    # Process block
                    try:
                        result = self.chunker.chunk(process_content, source_info=source_info, **kwargs)

                        for chunk in result.chunks:
                            chunk.id = f"stream_{chunk_counter}_{chunk.id}"
                            chunk.metadata.stream_offset = f.tell() - len(buffer) + len(content_block)
                            chunk_counter += 1
                            self._chunks_generated = chunk_counter
                            yield chunk

                            # Update progress periodically
                            if chunk_counter % self.progress_update_interval == 0:
                                self._update_progress(
                                    file_path=str(file_path),
                                    total_size=file_path.stat().st_size,
                                    status="processing"
                                )

                                # Create checkpoint periodically
                                if self.enable_checkpointing and chunk_counter % (self.progress_update_interval * 5) == 0:
                                    self._create_checkpoint(file_path, f.tell())

                    except Exception as e:
                        self.logger.error(f"❌ Error processing chunk block at offset {f.tell()}: {e}")

                    # Setup overlap for next block
                    if self.overlap_size > 0:
                        overlap_buffer = buffer[self.block_size - self.overlap_size:self.block_size]
                    else:
                        overlap_buffer = b""

                    buffer = buffer[self.block_size:]

            # Process remaining content
            if buffer:
                try:
                    text_content = buffer.decode('utf-8', errors='ignore')
                    process_content = text_content
                except:
                    process_content = buffer

                final_content = overlap_buffer + buffer
                try:
                    result = self.chunker.chunk(process_content, source_info=source_info, **kwargs)

                    for chunk in result.chunks:
                        chunk.id = f"stream_{chunk_counter}_{chunk.id}"
                        chunk.metadata.stream_offset = f.tell()
                        chunk_counter += 1
                        self._chunks_generated = chunk_counter
                        yield chunk

                except Exception as e:
                    self.logger.error(f"❌ Error processing final chunk block: {e}")

    def get_current_progress(self) -> Optional[StreamingProgress]:
        """Get current progress information."""
        with self._progress_lock:
            return self._current_progress


class DistributedStreamingProcessor:
    """
    Distributed streaming processor for handling multiple large files in parallel.

    This class orchestrates streaming across multiple files, utilizing available
    CPU cores and providing aggregate progress reporting.
    """

    def __init__(
        self,
        strategy: Union[str, BaseChunker],
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[str, StreamingProgress], None]] = None,
        checkpoint_dir: Optional[Path] = None,
        **streaming_kwargs
    ):
        """
        Initialize distributed streaming processor.

        Args:
            strategy: Chunking strategy to use for all files
            max_workers: Maximum number of parallel workers (None = auto-detect)
            progress_callback: Callback for per-file progress updates (file_path, progress)
            checkpoint_dir: Directory for storing checkpoints
            **streaming_kwargs: Additional arguments for StreamingChunker
        """
        self.strategy = strategy
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.progress_callback = progress_callback
        self.checkpoint_dir = checkpoint_dir
        self.streaming_kwargs = streaming_kwargs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Progress tracking
        self._file_progress: Dict[str, StreamingProgress] = {}
        self._progress_lock = threading.Lock()
        self._total_files = 0
        self._completed_files = 0
        self._failed_files = 0

    def process_files(
        self,
        file_paths: List[Union[str, Path]],
        parallel_mode: str = "thread",  # "thread", "process", or "sequential"
        resume_all: bool = True
    ) -> DistributedStreamingResult:
        """
        Process multiple files in parallel with distributed streaming.

        Args:
            file_paths: List of file paths to process
            parallel_mode: Type of parallelization to use
            resume_all: Try to resume all files from checkpoints

        Returns:
            DistributedStreamingResult with comprehensive results
        """
        start_time = time.time()
        file_paths = [Path(p) for p in file_paths]

        self._total_files = len(file_paths)
        self._completed_files = 0
        self._failed_files = 0

        self.logger.info(f"🚀 Starting distributed streaming: {self._total_files} files")

        # Initialize result containers
        file_results: Dict[str, ChunkingResult] = {}
        errors: Dict[str, str] = {}

        # Calculate total size
        total_size = 0
        valid_files = []
        for file_path in file_paths:
            if file_path.exists():
                total_size += file_path.stat().st_size
                valid_files.append(file_path)
            else:
                error_msg = f"File not found: {file_path}"
                self.logger.warning(f"⚠️ {error_msg}")
                errors[str(file_path)] = error_msg
                self._failed_files += 1

        self.logger.info(f"📊 Total size: {total_size / (1024**3):.2f} GB across {len(valid_files)} files")
        total_chunks = 0

        if parallel_mode == "sequential":
            # Sequential processing
            for file_path in valid_files:
                try:
                    result = self._process_single_file(file_path, resume_all)
                    file_results[str(file_path)] = result
                    total_chunks += len(result.chunks)
                    self._completed_files += 1

                    self.logger.info(f"✅ Completed {file_path.name}: {len(result.chunks)} chunks")

                except Exception as e:
                    error_msg = str(e)
                    errors[str(file_path)] = error_msg
                    self._failed_files += 1
                    self.logger.error(f"❌ Failed {file_path.name}: {error_msg}")

        elif parallel_mode == "thread":
            # Thread-based parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, file_path, resume_all): file_path
                    for file_path in valid_files
                }

                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        file_results[str(file_path)] = result
                        total_chunks += len(result.chunks)
                        self._completed_files += 1

                        self.logger.info(f"✅ Completed {file_path.name}: {len(result.chunks)} chunks")

                    except Exception as e:
                        error_msg = str(e)
                        errors[str(file_path)] = error_msg
                        self._failed_files += 1
                        self.logger.error(f"❌ Failed {file_path.name}: {error_msg}")

        elif parallel_mode == "process":
            # Process-based parallel processing (for CPU-intensive tasks)
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        _process_file_worker,
                        str(file_path),
                        self.strategy,
                        self.streaming_kwargs,
                        resume_all
                    ): file_path
                    for file_path in valid_files
                }

                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        file_results[str(file_path)] = result
                        total_chunks += len(result.chunks)
                        self._completed_files += 1

                        self.logger.info(f"✅ Completed {file_path.name}: {len(result.chunks)} chunks")

                    except Exception as e:
                        error_msg = str(e)
                        errors[str(file_path)] = error_msg
                        self._failed_files += 1
                        self.logger.error(f"❌ Failed {file_path.name}: {error_msg}")

        else:
            raise ValueError(f"Unknown parallel_mode: {parallel_mode}")

        # Calculate final metrics
        total_processing_time = time.time() - start_time
        total_size_mb = total_size / (1024 * 1024)
        average_throughput_mbps = total_size_mb / total_processing_time if total_processing_time > 0 else 0

        result = DistributedStreamingResult(
            total_files=self._total_files,
            completed_files=self._completed_files,
            failed_files=self._failed_files,
            total_chunks=total_chunks,
            total_processing_time=total_processing_time,
            total_size_mb=total_size_mb,
            average_throughput_mbps=average_throughput_mbps,
            file_results=file_results,
            errors=errors
        )

        self.logger.info(
            f"🏁 Distributed streaming completed: "
            f"{self._completed_files}/{self._total_files} files, "
            f"{total_chunks:,} chunks, "
            f"{average_throughput_mbps:.2f} MB/s"
        )

        return result

    def _process_single_file(
        self,
        file_path: Path,
        resume_from_checkpoint: bool
    ) -> ChunkingResult:
        """Process a single file with streaming."""

        def progress_handler(progress: StreamingProgress) -> None:
            """Handle progress updates from individual file."""
            with self._progress_lock:
                self._file_progress[str(file_path)] = progress

                # Call external callback if provided
                if self.progress_callback:
                    try:
                        self.progress_callback(str(file_path), progress)
                    except Exception as e:
                        self.logger.warning(f"Progress callback failed for {file_path}: {e}")

        # Create dedicated streamer for this file
        streamer = StreamingChunker(
            self.strategy,
            progress_callback=progress_handler,
            checkpoint_dir=self.checkpoint_dir,
            **self.streaming_kwargs
        )

        # Collect all chunks from streaming
        start_time = time.time()
        chunks = list(streamer.stream_file(
            file_path,
            resume_from_checkpoint=resume_from_checkpoint
        ))
        processing_time = time.time() - start_time

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used=streamer.strategy_name,
            source_info={"source": str(file_path), "source_type": "file"}
        )

    def get_aggregate_progress(self) -> Dict[str, Any]:
        """Get aggregate progress across all files."""
        with self._progress_lock:
            if not self._file_progress:
                return {
                    "total_files": self._total_files,
                    "completed_files": self._completed_files,
                    "failed_files": self._failed_files,
                    "overall_progress_percentage": 0.0,
                    "total_chunks": 0,
                    "aggregate_throughput_mbps": 0.0
                }

            # Calculate aggregate metrics
            total_bytes = sum(p.total_size for p in self._file_progress.values())
            processed_bytes = sum(p.processed_bytes for p in self._file_progress.values())
            total_chunks = sum(p.chunks_generated for p in self._file_progress.values())

            overall_progress = (processed_bytes / total_bytes * 100) if total_bytes > 0 else 0

            # Calculate weighted average throughput
            active_files = [p for p in self._file_progress.values() if p.status == "processing"]
            avg_throughput = (
                sum(p.throughput_mbps for p in active_files) / len(active_files)
                if active_files else 0
            )

            return {
                "total_files": self._total_files,
                "completed_files": self._completed_files,
                "failed_files": self._failed_files,
                "processing_files": len(active_files),
                "overall_progress_percentage": overall_progress,
                "total_bytes": total_bytes,
                "processed_bytes": processed_bytes,
                "total_chunks": total_chunks,
                "aggregate_throughput_mbps": avg_throughput,
                "file_progress": dict(self._file_progress)
            }

    def cancel_all(self) -> None:
        """Cancel all ongoing processing (best effort)."""
        self.logger.warning("🛑 Cancellation requested - stopping after current chunks")
        # Note: Actual cancellation depends on the executor implementation
        # This is a placeholder for potential future cancellation logic


def _process_file_worker(
    file_path: str,
    strategy: Union[str, BaseChunker],
    streaming_kwargs: Dict[str, Any],
    resume_from_checkpoint: bool
) -> ChunkingResult:
    """
    Worker function for process-based parallel processing.

    This function is defined at module level to be pickle-able
    for ProcessPoolExecutor.
    """
    file_path = Path(file_path)

    # Create streamer in worker process
    streamer = StreamingChunker(strategy, **streaming_kwargs)

    # Process file
    start_time = time.time()
    chunks = list(streamer.stream_file(
        file_path,
        resume_from_checkpoint=resume_from_checkpoint
    ))
    processing_time = time.time() - start_time

    return ChunkingResult(
        chunks=chunks,
        processing_time=processing_time,
        strategy_used=streamer.strategy_name,
        source_info={"source": str(file_path), "source_type": "file"}
    )


class MemoryMappedStreamer:
    """
    Memory-mapped file streaming for very large files.

    Uses memory mapping to efficiently process files that are too large
    to fit in memory, providing random access while maintaining streaming
    performance characteristics.
    """

    def __init__(
        self,
        chunker: BaseChunker,
        window_size: int = 1024 * 1024,  # 1MB window
        overlap_size: int = 1024         # 1KB overlap
    ):
        """
        Initialize memory-mapped streamer.

        Args:
            chunker: Chunker to use for processing
            window_size: Size of processing window
            overlap_size: Overlap between windows
        """
        self.chunker = chunker
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def stream_file(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream chunks using memory mapping.

        Args:
            file_path: Path to the file to process
            **kwargs: Additional arguments passed to chunker

        Yields:
            Individual chunks as they are generated
        """
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.error(f"🚫 Memory-mapped streaming failed: File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        if file_size == 0:
            self.logger.warning(f"📄 Empty file detected: {file_path}")
            return

        self.logger.info(f"🗃️  Memory-mapped streaming: {file_path.name} ({file_size:,} bytes)")

        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                position = 0
                chunk_counter = 0

                while position < file_size:
                    # Calculate window boundaries
                    window_start = max(0, position - self.overlap_size)
                    window_end = min(file_size, position + self.window_size)

                    # Extract window content
                    window_content = mm[window_start:window_end]

                    # Try to decode as text, fallback to binary
                    try:
                        content = window_content.decode('utf-8')
                        modality = ModalityType.TEXT
                    except UnicodeDecodeError:
                        content = window_content
                        modality = ModalityType.MIXED  # Binary content

                    # Process window
                    try:
                        source_info = {
                            "source": str(file_path),
                            "source_type": "file",
                            "offset": window_start,
                            "length": window_end - window_start
                        }

                        result = self.chunker.chunk(content, source_info=source_info, **kwargs)

                        for chunk in result.chunks:
                            chunk.id = f"mmap_{chunk_counter}_{chunk.id}"
                            chunk.metadata.offset = window_start + (chunk.metadata.offset or 0)
                            chunk_counter += 1
                            yield chunk

                    except Exception as e:
                        self.logger.error(f"Error processing window at {position}: {e}")

                    # Move to next window
                    position += self.window_size

    def get_chunk_at_offset(
        self,
        file_path: Union[str, Path],
        offset: int,
        length: int,
        **kwargs
    ) -> Optional[Chunk]:
        """
        Get a specific chunk at a given file offset.

        Args:
            file_path: Path to the file
            offset: Byte offset in the file
            length: Length of content to extract
            **kwargs: Additional arguments passed to chunker

        Returns:
            Chunk at the specified location or None if error
        """
        file_path = Path(file_path)

        try:
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    content_bytes = mm[offset:offset + length]

                    # Try to decode as text
                    try:
                        content = content_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        content = content_bytes

                    source_info = {
                        "source": str(file_path),
                        "source_type": "file",
                        "offset": offset,
                        "length": length
                    }

                    result = self.chunker.chunk(content, source_info=source_info, **kwargs)

                    if result.chunks:
                        chunk = result.chunks[0]  # Return first chunk
                        chunk.metadata.offset = offset
                        return chunk

        except Exception as e:
            self.logger.error(f"Error getting chunk at offset {offset}: {e}")

        return None
