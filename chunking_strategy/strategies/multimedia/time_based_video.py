"""
Time-based video chunker that segments video files into fixed-duration chunks.

This chunker divides video content based on time intervals, similar to time-based audio chunking
but adapted for video files. It supports various video formats and preserves video properties.
"""

import logging
import time
import math
from pathlib import Path
from typing import Union, Dict, Any, Optional, Iterator, TYPE_CHECKING
from io import BytesIO

from chunking_strategy.core.base import StreamableChunker, Chunk, ChunkingResult, ChunkMetadata
from chunking_strategy.core.registry import register_chunker
from chunking_strategy.core.base import ModalityType

if TYPE_CHECKING:
    import cv2

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
except ImportError:
    VideoFileClip = None

logger = logging.getLogger(__name__)


@register_chunker(
    name="time_based_video",
    category="multimedia",
    description="Split video files into fixed time duration segments",
    supported_modalities=[ModalityType.VIDEO],
    supported_formats=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"]
)
class TimeBasedVideoChunker(StreamableChunker):
    """
    Chunks video content based on fixed time intervals.

    This chunker divides video files into segments of specified duration,
    with optional overlap between segments. It preserves video format
    and properties when possible.
    """

    def __init__(
        self,
        segment_duration: float = 30.0,
        overlap_duration: float = 0.0,
        preserve_format: bool = True,
        target_fps: Optional[int] = None,
        target_resolution: Optional[tuple] = None,
        **kwargs
    ):
        """
        Initialize the time-based video chunker.

        Args:
            segment_duration: Duration of each video segment in seconds (default: 30.0)
            overlap_duration: Overlap between consecutive segments in seconds (default: 0.0)
            preserve_format: Whether to preserve original video format (default: True)
            target_fps: Target FPS for output videos (None to preserve original)
            target_resolution: Target resolution as (width, height) tuple (None to preserve original)
            **kwargs: Additional parameters
        """
        name = kwargs.pop("name", "time_based_video")
        super().__init__(
            name=name,
            **kwargs
        )

        # Validate parameters
        if segment_duration <= 0:
            raise ValueError("segment_duration must be positive")
        if overlap_duration < 0:
            raise ValueError("overlap_duration cannot be negative")
        if overlap_duration >= segment_duration:
            raise ValueError("overlap_duration must be less than segment_duration")
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if target_resolution is not None:
            if not isinstance(target_resolution, (list, tuple)) or len(target_resolution) != 2:
                raise ValueError("target_resolution must be a (width, height) tuple")
            if target_resolution[0] <= 0 or target_resolution[1] <= 0:
                raise ValueError("target_resolution dimensions must be positive")

        self.segment_duration = float(segment_duration)
        self.overlap_duration = float(overlap_duration)
        self.preserve_format = preserve_format
        self.target_fps = target_fps
        self.target_resolution = target_resolution

        logger.debug(f"Initialized TimeBasedVideoChunker with segment_duration={segment_duration}s, "
                    f"overlap={overlap_duration}s")

    def _load_video(self, content: Union[str, bytes, Path]) -> "VideoFileClip":
        """Load video from various input types."""
        if VideoFileClip is None:
            raise ImportError(
                "moviepy is required for video chunking. "
                "Install with: pip install moviepy"
            )

        if isinstance(content, (str, Path)):
            video = VideoFileClip(str(content))
        elif isinstance(content, bytes):
            # Try to detect format from bytes and load accordingly
            from io import BytesIO
            import tempfile
            import os

            # Create temporary file for bytes content
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name

            try:
                video = VideoFileClip(tmp_path)
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
        else:
            raise TypeError(f"Unsupported content type: {type(content)}")

        return video

    def _detect_format(self, content: Union[str, bytes, Path]) -> str:
        """Detect the original video format."""
        if isinstance(content, (str, Path)):
            path = Path(content)
            return path.suffix.lower() or '.mp4'
        else:
            return '.mp4'  # Default for bytes content

    def _extract_source_info(self, video: "VideoFileClip", original_path: Union[str, bytes, Path]) -> Dict[str, Any]:
        """Extract video metadata and properties."""
        return {
            "duration": video.duration,
            "fps": video.fps,
            "size": video.size,  # (width, height)
            "format": self._detect_format(original_path),
            "total_frames": int(video.fps * video.duration) if video.fps else 0,
            "has_audio": video.audio is not None
        }

    def _segment_to_content(self, segment: "VideoFileClip", original_format: str) -> bytes:
        """Convert video segment to bytes content."""
        import tempfile
        import os

        # Determine output format
        if self.preserve_format and original_format in ['.mp4', '.avi', '.mov']:
            output_format = original_format.lstrip('.')
        else:
            output_format = 'mp4'  # Default format

        # Create temporary file for segment
        with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Apply transformations if specified
            processed_segment = segment

            if self.target_fps and segment.fps != self.target_fps:
                processed_segment = processed_segment.set_fps(self.target_fps)

            if self.target_resolution and segment.size != self.target_resolution:
                processed_segment = processed_segment.resize(self.target_resolution)

            # Write segment to temporary file
            processed_segment.write_videofile(
                tmp_path,
                audio_codec='aac' if processed_segment.audio is not None else None,
                verbose=False,
                logger=None
            )

            # Read bytes from temporary file
            with open(tmp_path, 'rb') as f:
                content_bytes = f.read()

            return content_bytes

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk video content into time-based segments.

        Args:
            content: Video content as file path, bytes, or Path object
            source_info: Optional source information
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with video chunks based on time intervals
        """
        start_time = time.time()

        try:
            # Load video
            video = self._load_video(content)
            original_format = self._detect_format(content)

            # Apply video transformations if specified
            if self.target_fps and video.fps != self.target_fps:
                video = video.set_fps(self.target_fps)

            if self.target_resolution and video.size != self.target_resolution:
                video = video.resize(self.target_resolution)

        except (FileNotFoundError, ValueError, TypeError, ImportError) as e:
            # Re-raise specific errors for proper handling
            raise e
        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                source_info={"error": f"Failed to load video: {e}"}
            )

        try:
            # Extract video metadata
            source_metadata = self._extract_source_info(video, content)
            total_duration = video.duration

            # Calculate segment parameters
            effective_step = self.segment_duration - self.overlap_duration
            segments = []

            current_start = 0.0
            segment_index = 0

            while current_start < total_duration:
                # Calculate segment end
                segment_end = min(current_start + self.segment_duration, total_duration)

                # Skip very short segments (less than 1 second)
                if segment_end - current_start < 1.0 and segment_index > 0:
                    break

                # Extract video segment
                segment = video.subclip(current_start, segment_end)
                segment_duration = segment_end - current_start

                # Convert segment to bytes
                segment_content = self._segment_to_content(segment, original_format)

                # Create metadata
                metadata = ChunkMetadata(
                    source=(source_info or {}).get("source", "unknown"),
                    source_type="video_file",
                    position=f"time {current_start:.2f}s-{segment_end:.2f}s",
                    offset=int(current_start * 1000),  # Convert to milliseconds
                    length=len(segment_content),
                    extra={
                        "chunk_id": f"time_video_chunk_{segment_index+1}",
                        "segment_index": segment_index,
                        "duration_seconds": segment_duration,
                        "start_time_seconds": current_start,
                        "end_time_seconds": segment_end,
                        "fps": segment.fps,
                        "size": segment.size,
                        "format": original_format,
                        "boundary_type": "time_based",
                        "char_count": 0,  # Not applicable for video
                        "word_count": 0,  # Not applicable for video
                        "sentence_count": 0  # Not applicable for video
                    }
                )

                chunk = Chunk(
                    id=f"video_segment_{segment_index+1}",
                    content=segment_content,
                    modality=ModalityType.VIDEO,
                    metadata=metadata
                )
                segments.append(chunk)

                # Move to next segment
                current_start += effective_step
                segment_index += 1

            # Close video file
            video.close()

            processing_time = time.time() - start_time

            logger.info(f"Created {len(segments)} video segments from {total_duration:.2f}s video "
                       f"using time-based chunking in {processing_time:.3f}s")

            return ChunkingResult(
                chunks=segments,
                processing_time=processing_time,
                source_info={
                    "total_segments": len(segments),
                    "source_duration": total_duration,
                    "segment_duration": self.segment_duration,
                    "overlap_duration": self.overlap_duration,
                    **source_metadata
                }
            )

        except Exception as e:
            video.close() if 'video' in locals() else None
            logger.error(f"Failed to process video with time-based chunking: {e}")
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                source_info={"error": f"Failed to process video: {e}"}
            )

    def get_chunk_size_estimate(self, content_length: int, **kwargs) -> int:
        """
        Estimate the number of chunks for given content.

        Args:
            content_length: Length of content (for video, this might be duration in seconds)
            **kwargs: Additional parameters

        Returns:
            Estimated number of chunks
        """
        if content_length <= 0:
            return 0

        # For video, content_length might represent duration in seconds
        # If not, we make a rough estimate based on typical video sizes
        estimated_duration = content_length  # Assume content_length is duration in seconds

        effective_step = self.segment_duration - self.overlap_duration
        estimated_chunks = math.ceil(estimated_duration / effective_step)

        return max(1, estimated_chunks)

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration parameters."""
        return {
            "segment_duration": self.segment_duration,
            "overlap_duration": self.overlap_duration,
            "preserve_format": self.preserve_format,
            "target_fps": self.target_fps,
            "target_resolution": self.target_resolution
        }

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream video content for chunking.

        Note: For video files, streaming is typically not as useful as for text,
        since we need the complete video to perform time-based segmentation effectively.
        This implementation accumulates the stream and processes it as a whole.

        Args:
            content_stream: Iterator of video content (typically bytes)
            source_info: Optional source information
            **kwargs: Additional parameters

        Yields:
            Chunk objects for each video segment
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
