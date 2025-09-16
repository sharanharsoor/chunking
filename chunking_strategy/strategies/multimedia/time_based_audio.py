"""
Time-based audio chunking strategy.

This module implements a time-based audio chunker that splits audio files into
fixed-duration segments. This is useful for:
- Processing long audio files in manageable chunks
- Creating uniform segments for machine learning training
- Breaking up recordings for parallel processing
- Generating time-indexed segments for search/retrieval

Key features:
- Configurable segment duration
- Support for multiple audio formats (MP3, WAV, OGG, etc.)
- Overlap support for context preservation
- Metadata preservation including timestamps
- Memory-efficient streaming support
- Integration with existing chunking infrastructure

Author: AI Assistant
Date: 2024
"""

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator, TYPE_CHECKING
import tempfile
import os

# Type imports
if TYPE_CHECKING:
    try:
        from pydub import AudioSegment
    except ImportError:
        pass

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType,
    StreamableChunker
)
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage

logger = logging.getLogger(__name__)

# Try to import audio processing libraries
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None  # For type annotations
    logger.warning("pydub not available - audio chunking will be limited")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@register_chunker(
    name="time_based_audio",
    category="multimedia",
    description="Split audio files into fixed time duration segments",
    supported_modalities=[ModalityType.AUDIO],
    supported_formats=["mp3", "wav", "ogg", "flac", "m4a", "aac"],
    complexity=ComplexityLevel.MEDIUM,
    dependencies=["pydub"],
    optional_dependencies=["librosa", "ffmpeg"],
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.MEDIUM,
    quality=0.9,  # High quality for simple time-based chunking
    parameters_schema={
        "segment_duration": {
            "type": "number",
            "minimum": 0.1,
            "maximum": 3600.0,
            "default": 30.0,
            "description": "Duration of each audio segment in seconds"
        },
        "overlap_duration": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 300.0,
            "default": 0.0,
            "description": "Overlap between segments in seconds"
        },
        "min_segment_duration": {
            "type": "number",
            "minimum": 0.1,
            "maximum": 60.0,
            "default": 1.0,
            "description": "Minimum duration for the last segment"
        },
        "preserve_format": {
            "type": "boolean",
            "default": True,
            "description": "Preserve original audio format in chunks"
        },
        "sample_rate": {
            "type": "integer",
            "minimum": 8000,
            "maximum": 192000,
            "default": None,
            "description": "Target sample rate (None to preserve original)"
        },
        "channels": {
            "type": "integer",
            "minimum": 1,
            "maximum": 8,
            "default": None,
            "description": "Target number of channels (None to preserve original)"
        }
    }
)
class TimeBasedAudioChunker(StreamableChunker):
    """
    Time-based audio chunker that splits audio into fixed-duration segments.

    This chunker processes audio files by dividing them into segments of a
    specified duration. It supports various audio formats and can optionally
    include overlap between segments for better context preservation.
    """

    def __init__(
        self,
        segment_duration: float = 30.0,
        overlap_duration: float = 0.0,
        min_segment_duration: float = 1.0,
        preserve_format: bool = True,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the time-based audio chunker.

        Args:
            segment_duration: Duration of each segment in seconds
            overlap_duration: Overlap between segments in seconds
            min_segment_duration: Minimum duration for the last segment
            preserve_format: Whether to preserve original audio format
            sample_rate: Target sample rate (None to preserve original)
            channels: Target number of channels (None to preserve original)
            **kwargs: Additional configuration parameters
        """
        # Extract name from kwargs or use default
        name = kwargs.pop("name", "time_based_audio")
        super().__init__(
            name=name,
            category="multimedia",
            supported_modalities=[ModalityType.AUDIO],
            **kwargs
        )

        # Validate dependencies
        if not PYDUB_AVAILABLE:
            raise ImportError("pydub is required for audio chunking. Install with: pip install pydub")

        # Core parameters
        self.segment_duration = segment_duration
        self.overlap_duration = overlap_duration
        self.min_segment_duration = min_segment_duration
        self.preserve_format = preserve_format
        self.sample_rate = sample_rate
        self.channels = channels

        # Validation
        if self.overlap_duration >= self.segment_duration:
            raise ValueError("overlap_duration must be less than segment_duration")
        if self.min_segment_duration > self.segment_duration:
            raise ValueError("min_segment_duration must be less than or equal to segment_duration")

        # Convert durations to milliseconds for pydub
        self.segment_ms = int(self.segment_duration * 1000)
        self.overlap_ms = int(self.overlap_duration * 1000)
        self.min_segment_ms = int(self.min_segment_duration * 1000)

        logger.debug(f"Initialized TimeBasedAudioChunker: {self.segment_duration}s segments, {self.overlap_duration}s overlap")

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk audio content into time-based segments.

        Args:
            content: Input audio content (file path, bytes, or AudioSegment)
            source_info: Information about the source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult containing audio chunks with timing metadata
        """
        start_time = time.time()

        # Load audio file
        try:
            audio = self._load_audio(content)
            original_format = self._detect_format(content)
        except (FileNotFoundError, ValueError, TypeError) as e:
            # Let file-related errors bubble up for proper error handling
            raise
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used=self.name,
                source_info=source_info or {}
            )

        source_info = source_info or self._extract_source_info(content, audio, original_format)

        # Apply any audio preprocessing
        if self.sample_rate and audio.frame_rate != self.sample_rate:
            audio = audio.set_frame_rate(self.sample_rate)
        if self.channels and audio.channels != self.channels:
            audio = audio.set_channels(self.channels)

        # Create chunks
        chunks = []
        audio_duration_ms = len(audio)
        current_start = 0
        chunk_id = 0

        while current_start < audio_duration_ms:
            # Calculate segment boundaries
            segment_end = current_start + self.segment_ms

            # Handle last segment
            if segment_end > audio_duration_ms:
                segment_end = audio_duration_ms
                # Skip if remaining segment is too short
                if segment_end - current_start < self.min_segment_ms:
                    break

            # Extract audio segment
            audio_segment = audio[current_start:segment_end]

            # Convert to appropriate format
            chunk_content = self._segment_to_content(audio_segment, original_format)

            # Calculate timing
            start_seconds = current_start / 1000.0
            end_seconds = segment_end / 1000.0
            duration_seconds = (segment_end - current_start) / 1000.0

            # Create chunk
            chunk = Chunk(
                id=f"{self.name}_chunk_{chunk_id}",
                content=chunk_content,
                modality=ModalityType.AUDIO,
                size=len(chunk_content),
                metadata=ChunkMetadata(
                    source=source_info.get("source", "unknown"),
                    source_type=source_info.get("source_type", "audio"),
                    position=f"time {start_seconds:.2f}s-{end_seconds:.2f}s",
                    offset=current_start,
                    length=segment_end - current_start,
                    extra={
                        "chunk_index": chunk_id,
                        "start_time": start_seconds,
                        "end_time": end_seconds,
                        "duration": duration_seconds,
                        "sample_rate": audio.frame_rate,
                        "channels": audio.channels,
                        "format": original_format,
                        "chunker_used": self.name,
                        "overlap_with_previous": chunk_id > 0 and self.overlap_duration > 0,
                        "chunking_strategy": "time_based_audio"
                    }
                )
            )
            chunks.append(chunk)
            chunk_id += 1

            # Calculate next start position (with overlap)
            if self.overlap_ms > 0:
                current_start += self.segment_ms - self.overlap_ms
            else:
                current_start += self.segment_ms

            # Ensure we make progress
            if current_start >= segment_end and segment_end < audio_duration_ms:
                current_start = segment_end

        processing_time = time.time() - start_time

        result = ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used=self.name,
            source_info={
                **source_info,
                "total_duration": audio_duration_ms / 1000.0,
                "segment_duration": self.segment_duration,
                "overlap_duration": self.overlap_duration,
                "total_segments": len(chunks),
                "original_sample_rate": source_info.get("sample_rate"),
                "original_channels": source_info.get("channels"),
                "processed_sample_rate": audio.frame_rate,
                "processed_channels": audio.channels
            }
        )

        logger.info(f"Created {len(chunks)} audio segments from {audio_duration_ms/1000:.2f}s audio in {processing_time:.3f}s")
        return result

    def _load_audio(self, content: Union[str, bytes, Path]):
        """Load audio from various input types."""
        if isinstance(content, Path):
            return AudioSegment.from_file(str(content))
        elif isinstance(content, str):
            if os.path.exists(content):
                return AudioSegment.from_file(content)
            else:
                raise ValueError(f"Audio file not found: {content}")
        elif isinstance(content, bytes):
            # Create temporary file for bytes content
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                try:
                    audio = AudioSegment.from_file(temp_file.name)
                finally:
                    os.unlink(temp_file.name)
                return audio
        else:
            raise TypeError(f"Unsupported content type: {type(content)}")

    def _detect_format(self, content: Union[str, bytes, Path]) -> str:
        """Detect audio format from content."""
        if isinstance(content, (str, Path)):
            path = Path(content)
            return path.suffix.lower().lstrip('.')
        else:
            return "wav"  # Default format for bytes content

    def _extract_source_info(self, content: Union[str, bytes, Path], audio, format: str) -> Dict[str, Any]:
        """Extract source information from audio."""
        if isinstance(content, (str, Path)):
            source = str(content)
            source_type = "file"
        else:
            source = "bytes_content"
            source_type = "bytes"

        return {
            "source": source,
            "source_type": source_type,
            "format": format,
            "duration": len(audio) / 1000.0,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "frame_count": audio.frame_count(),
        }

    def _segment_to_content(self, segment, original_format: str) -> bytes:
        """Convert audio segment to bytes content."""
        if self.preserve_format and original_format in ["mp3", "wav", "ogg"]:
            return segment.export(format=original_format).read()
        else:
            # Default to WAV for universal compatibility
            return segment.export(format="wav").read()

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream audio chunks from input stream.

        Note: This collects all stream content first since audio needs to be
        processed as a complete file for accurate timing.
        """
        # Collect all content from stream
        collected_content = b""
        for content_piece in content_stream:
            if isinstance(content_piece, str):
                # Assume it's a file path
                with open(content_piece, 'rb') as f:
                    collected_content += f.read()
            elif isinstance(content_piece, bytes):
                collected_content += content_piece
            else:
                raise TypeError(f"Unsupported stream content type: {type(content_piece)}")

        # Process collected content and yield chunks
        result = self.chunk(collected_content, source_info=source_info, **kwargs)
        for chunk in result.chunks:
            yield chunk

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "name": self.name,
            "segment_duration": self.segment_duration,
            "overlap_duration": self.overlap_duration,
            "min_segment_duration": self.min_segment_duration,
            "preserve_format": self.preserve_format,
            "sample_rate": self.sample_rate,
            "channels": self.channels
        }

    def get_chunk_size_estimate(self, audio_duration: float) -> int:
        """
        Estimate number of chunks that will be created.

        Args:
            audio_duration: Duration of audio in seconds

        Returns:
            Estimated number of chunks
        """
        if audio_duration <= self.segment_duration:
            return 1

        # Calculate with overlap
        effective_segment_duration = self.segment_duration - self.overlap_duration
        if effective_segment_duration <= 0:
            return 1

        return max(1, math.ceil((audio_duration - self.overlap_duration) / effective_segment_duration))
