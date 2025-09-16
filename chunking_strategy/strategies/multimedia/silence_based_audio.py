"""
Silence-based audio chunking strategy.

This strategy segments audio files based on silence detection, creating natural
boundaries at quiet periods in the audio. Ideal for speech, podcasts, and
recordings with natural pauses.
"""

import logging
import time
import math
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Iterator, TYPE_CHECKING

from chunking_strategy.core.base import (
    StreamableChunker,
    ModalityType,
    ChunkingResult,
    Chunk,
    ChunkMetadata
)
from chunking_strategy.core.registry import register_chunker

if TYPE_CHECKING:
    from pydub import AudioSegment
    from pydub.silence import detect_silence

logger = logging.getLogger(__name__)


@register_chunker(
    name="silence_based_audio",
    category="multimedia",
    description="Segments audio based on silence detection for natural boundaries",
    supported_modalities=[ModalityType.AUDIO],
    supported_formats=["mp3", "wav", "ogg", "flac", "m4a", "aac", "wma", "opus"]
)
class SilenceBasedAudioChunker(StreamableChunker):
    """
    Chunks audio files based on silence detection.

    This chunker analyzes audio content to identify silent periods and uses them
    as natural boundaries for segmentation. Ideal for speech, podcasts, interviews,
    and other audio with natural pauses.
    """

    def __init__(
        self,
        silence_threshold_db: float = -40.0,
        min_silence_duration: float = 0.5,
        min_segment_duration: float = 5.0,
        max_segment_duration: float = 120.0,
        padding_duration: float = 0.1,
        preserve_format: bool = True,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the silence-based audio chunker.

        Args:
            silence_threshold_db: dB level below which audio is considered silent (default: -40.0)
            min_silence_duration: Minimum duration of silence to consider as boundary in seconds (default: 0.5)
            min_segment_duration: Minimum duration for a segment in seconds (default: 5.0)
            max_segment_duration: Maximum duration for a segment in seconds (default: 120.0)
            padding_duration: Padding to add around silence boundaries in seconds (default: 0.1)
            preserve_format: Whether to preserve the original audio format (default: True)
            sample_rate: Target sample rate in Hz, None to preserve original (default: None)
            channels: Target number of channels, None to preserve original (default: None)
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="silence_based_audio",
            category="multimedia",
            supported_modalities=[ModalityType.AUDIO],
            **kwargs
        )

        # Validate parameters
        if silence_threshold_db >= 0:
            raise ValueError("silence_threshold_db must be negative (dB)")
        if min_silence_duration <= 0:
            raise ValueError("min_silence_duration must be positive")
        if min_segment_duration <= 0:
            raise ValueError("min_segment_duration must be positive")
        if max_segment_duration <= min_segment_duration:
            raise ValueError("max_segment_duration must be greater than min_segment_duration")
        if padding_duration < 0:
            raise ValueError("padding_duration must be non-negative")
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels is not None and channels <= 0:
            raise ValueError("channels must be positive")

        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.padding_duration = padding_duration
        self.preserve_format = preserve_format
        self.sample_rate = sample_rate
        self.channels = channels

        logger.debug(f"Initialized SilenceBasedAudioChunker with threshold={silence_threshold_db}dB, "
                    f"min_silence={min_silence_duration}s, min_segment={min_segment_duration}s")

    def _load_audio(self, content: Union[str, bytes, Path]) -> "AudioSegment":
        """Load audio from various input types."""
        try:
            from pydub import AudioSegment
        except ImportError as e:
            raise ImportError(
                "pydub is required for audio chunking. "
                "Install with: pip install pydub"
            ) from e

        if isinstance(content, (str, Path)):
            audio = AudioSegment.from_file(str(content))
        elif isinstance(content, bytes):
            # Try to detect format from bytes and load accordingly
            from io import BytesIO
            audio = AudioSegment.from_file(BytesIO(content))
        else:
            raise TypeError(f"Unsupported content type: {type(content)}")

        return audio

    def _detect_format(self, content: Union[str, bytes, Path]) -> str:
        """Detect the original audio format."""
        if isinstance(content, (str, Path)):
            path = Path(content)
            return path.suffix.lower().lstrip('.')
        else:
            # For bytes, default to wav
            return "wav"

    def _extract_source_info(self, audio: "AudioSegment", original_format: str) -> Dict[str, Any]:
        """Extract metadata from audio."""
        return {
            "duration": len(audio) / 1000.0,  # Convert ms to seconds
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "frame_width": audio.frame_width,
            "format": original_format
        }

    def _detect_silence_boundaries(self, audio: "AudioSegment") -> List[tuple]:
        """
        Detect silence periods in audio that can serve as segment boundaries.

        Returns:
            List of (start_ms, end_ms) tuples for silence periods
        """
        try:
            from pydub.silence import detect_silence
        except ImportError as e:
            raise ImportError(
                "pydub is required for silence detection. "
                "Install with: pip install pydub"
            ) from e

        # Convert silence parameters to milliseconds
        min_silence_ms = int(self.min_silence_duration * 1000)

        # Detect silence periods
        silence_ranges = detect_silence(
            audio,
            min_silence_len=min_silence_ms,
            silence_thresh=self.silence_threshold_db
        )

        logger.debug(f"Detected {len(silence_ranges)} silence periods")
        return silence_ranges

    def _create_segments_from_silence(
        self,
        audio: "AudioSegment",
        silence_ranges: List[tuple]
    ) -> List["AudioSegment"]:
        """
        Create audio segments using silence boundaries.

        Args:
            audio: The original audio
            silence_ranges: List of (start_ms, end_ms) silence periods

        Returns:
            List of audio segments
        """
        segments = []
        audio_length = len(audio)
        padding_ms = int(self.padding_duration * 1000)
        min_segment_ms = int(self.min_segment_duration * 1000)
        max_segment_ms = int(self.max_segment_duration * 1000)

        if not silence_ranges:
            # No silence detected, split by max duration if needed
            if audio_length <= max_segment_ms:
                segments.append(audio)
            else:
                # Split into max-duration segments
                for start_ms in range(0, audio_length, max_segment_ms):
                    end_ms = min(start_ms + max_segment_ms, audio_length)
                    segments.append(audio[start_ms:end_ms])
            return segments

        # Create segments using silence boundaries
        last_end = 0

        for silence_start, silence_end in silence_ranges:
            # Calculate segment boundaries with padding
            segment_start = max(0, last_end)
            segment_end = min(silence_start + padding_ms, audio_length)

            # Check if segment meets minimum duration
            segment_duration = segment_end - segment_start
            if segment_duration >= min_segment_ms:
                segments.append(audio[segment_start:segment_end])
                last_end = max(0, silence_end - padding_ms)
            else:
                # Segment too short, extend to next silence or combine with next
                continue

        # Handle remaining audio after last silence
        if last_end < audio_length:
            remaining_duration = audio_length - last_end
            if remaining_duration >= min_segment_ms:
                segments.append(audio[last_end:audio_length])
            elif segments:
                # Combine with last segment if too short
                last_segment = segments.pop()
                last_segment_start = last_end - len(last_segment)
                segments.append(audio[last_segment_start:audio_length])
            else:
                # No previous segments, keep as is
                segments.append(audio[last_end:audio_length])

        # Merge segments that are too long
        final_segments = []
        for segment in segments:
            if len(segment) <= max_segment_ms:
                final_segments.append(segment)
            else:
                # Split long segment
                segment_len = len(segment)
                for start_ms in range(0, segment_len, max_segment_ms):
                    end_ms = min(start_ms + max_segment_ms, segment_len)
                    final_segments.append(segment[start_ms:end_ms])

        return final_segments

    def _segment_to_content(self, segment: "AudioSegment", original_format: str) -> bytes:
        """Convert audio segment back to bytes."""
        if self.preserve_format and original_format in ['mp3', 'wav', 'ogg', 'flac', 'm4a']:
            format_to_use = original_format
        else:
            format_to_use = "wav"  # Default fallback format

        return segment.export(format=format_to_use).read()

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk audio content based on silence detection.

        Args:
            content: Audio file path, audio bytes, or Path object
            source_info: Optional source information
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with audio chunks based on silence boundaries
        """
        start_time = time.time()

        try:
            # Load audio
            audio = self._load_audio(content)
            original_format = self._detect_format(content)

            # Apply audio transformations if specified
            if self.sample_rate and audio.frame_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)

            if self.channels and audio.channels != self.channels:
                if self.channels == 1:
                    audio = audio.set_channels(1)  # Convert to mono
                elif self.channels == 2:
                    audio = audio.set_channels(2)  # Convert to stereo

        except (FileNotFoundError, ValueError, TypeError, ImportError) as e:
            # Re-raise specific errors for proper handling
            raise e
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used=self.name,
                source_info=source_info or {}
            )

        try:
            # Detect silence boundaries
            silence_ranges = self._detect_silence_boundaries(audio)

            # Create segments based on silence
            segments = self._create_segments_from_silence(audio, silence_ranges)

            # Extract source information
            audio_info = self._extract_source_info(audio, original_format)

            # Create chunks
            chunks = []
            for i, segment in enumerate(segments):
                # Convert segment to bytes
                segment_content = self._segment_to_content(segment, original_format)

                # Calculate timing information
                segment_start_ms = sum(len(seg) for seg in segments[:i])
                segment_duration = len(segment) / 1000.0  # Convert to seconds

                # Create metadata
                metadata = ChunkMetadata(
                    source=(source_info or {}).get("source", "unknown"),
                    source_type="audio_file",
                    position=f"time {segment_start_ms/1000.0:.2f}s-{(segment_start_ms + len(segment))/1000.0:.2f}s",
                    offset=segment_start_ms,
                    length=len(segment_content),
                    extra={
                        "chunk_id": f"silence_audio_chunk_{i+1}",
                        "segment_index": i,
                        "duration_seconds": segment_duration,
                        "start_time_seconds": segment_start_ms / 1000.0,
                        "sample_rate": segment.frame_rate,
                        "channels": segment.channels,
                        "format": original_format,
                        "silence_threshold_db": self.silence_threshold_db,
                        "boundary_type": "silence_based",
                        "char_count": len(segment_content),
                        "word_count": 0,  # Not applicable for audio
                        "sentence_count": 0  # Not applicable for audio
                    }
                )

                # Create chunk
                chunk = Chunk(
                    id=f"chunk_{i+1}",
                    content=segment_content,
                    modality=ModalityType.AUDIO,
                    metadata=metadata
                )
                chunks.append(chunk)

            processing_time = time.time() - start_time

            logger.info(f"Created {len(chunks)} audio segments from {audio_info['duration']:.2f}s audio "
                       f"using silence detection in {processing_time:.3f}s")

            # Merge source info
            final_source_info = source_info or {}
            final_source_info.update(audio_info)

            return ChunkingResult(
                chunks=chunks,
                processing_time=processing_time,
                strategy_used=self.name,
                source_info=final_source_info
            )

        except Exception as e:
            logger.error(f"Failed to process audio with silence detection: {e}")
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used=self.name,
                source_info=source_info or {}
            )

    def get_chunk_size_estimate(self, audio_duration: float, typical_speech_rate: float = 0.7) -> int:
        """
        Estimate the number of chunks for given audio duration.

        Args:
            audio_duration: Duration of audio in seconds
            typical_speech_rate: Typical ratio of speech to silence (default: 0.7)

        Returns:
            Estimated number of chunks
        """
        if audio_duration <= 0:
            return 0

        # Estimate based on natural speech patterns
        # Assume natural breaks occur roughly every min_segment_duration to max_segment_duration
        avg_segment_duration = (self.min_segment_duration + self.max_segment_duration) / 2

        # Adjust for speech vs silence ratio
        effective_duration = audio_duration * typical_speech_rate

        estimated_chunks = max(1, math.ceil(effective_duration / avg_segment_duration))

        return estimated_chunks

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream audio content for chunking.

        Note: For audio files, streaming is typically not as useful as for text,
        since we need the complete audio to perform silence detection effectively.
        This implementation accumulates the stream and processes it as a whole.

        Args:
            content_stream: Iterator of audio content (typically bytes)
            source_info: Optional source information
            **kwargs: Additional parameters

        Yields:
            Chunk objects for each audio segment
        """
        # Accumulate all content from stream
        accumulated_content = b''
        for content_chunk in content_stream:
            if isinstance(content_chunk, str):
                accumulated_content += content_chunk.encode('utf-8')
            else:
                accumulated_content += content_chunk

        # Process accumulated content using regular chunk method
        result = self.chunk(accumulated_content, source_info=source_info, **kwargs)

        # Yield each chunk
        for chunk in result.chunks:
            yield chunk

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return {
            "silence_threshold_db": self.silence_threshold_db,
            "min_silence_duration": self.min_silence_duration,
            "min_segment_duration": self.min_segment_duration,
            "max_segment_duration": self.max_segment_duration,
            "padding_duration": self.padding_duration,
            "preserve_format": self.preserve_format,
            "sample_rate": self.sample_rate,
            "channels": self.channels
        }
