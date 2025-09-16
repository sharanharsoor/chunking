"""
Scene-based video chunking implementation.

This module provides functionality to segment video files based on scene detection,
identifying natural boundaries where visual content changes significantly.
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import List, Union, Dict, Any, Iterator, Optional, Tuple
from io import BytesIO
import time
import math

from chunking_strategy.core.base import StreamableChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import cv2
    import numpy as np

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    import cv2
    import numpy as np
except ImportError:
    VideoFileClip = None
    cv2 = None
    np = None

logger = logging.getLogger(__name__)


@register_chunker(
    name="scene_based_video",
    category="multimedia",
    description="Split video files based on scene detection and visual content changes",
    supported_modalities=[ModalityType.VIDEO],
    supported_formats=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"]
)
class SceneBasedVideoChunker(StreamableChunker):
    """
    A chunker that segments video files based on scene detection.

    This chunker analyzes video content to identify scene boundaries where
    visual content changes significantly, creating semantically meaningful segments.

    Features:
    - Scene detection using histogram comparison and SSIM
    - Configurable sensitivity and thresholds
    - Support for minimum/maximum scene durations
    - Frame sampling optimization for performance
    - Graceful handling of missing dependencies
    """

    def __init__(
        self,
        scene_threshold: float = 30.0,
        min_scene_duration: float = 2.0,
        max_scene_duration: float = 120.0,
        sample_rate: float = 1.0,
        detection_method: str = "histogram",
        preserve_format: bool = True,
        target_fps: Optional[int] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        **kwargs
    ):
        """
        Initialize the scene-based video chunker.

        Args:
            scene_threshold: Threshold for scene change detection (0-100, higher = more sensitive)
            min_scene_duration: Minimum duration of a scene in seconds
            max_scene_duration: Maximum duration of a scene in seconds
            sample_rate: Rate at which to sample frames for analysis (frames per second)
            detection_method: Method for scene detection ("histogram", "ssim", "combined")
            preserve_format: Whether to preserve original video format
            target_fps: Target frame rate for output segments (None to preserve)
            target_resolution: Target resolution as (width, height) tuple (None to preserve)
            **kwargs: Additional arguments passed to parent class
        """
        name = kwargs.pop("name", "scene_based_video")
        super().__init__(name=name, **kwargs)

        # Validate parameters
        if not 0 < scene_threshold <= 100:
            raise ValueError("scene_threshold must be between 0 and 100")
        if min_scene_duration < 0:
            raise ValueError("min_scene_duration must be non-negative")
        if max_scene_duration <= min_scene_duration:
            raise ValueError("max_scene_duration must be greater than min_scene_duration")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if detection_method not in ["histogram", "ssim", "combined"]:
            raise ValueError("detection_method must be 'histogram', 'ssim', or 'combined'")

        self.scene_threshold = scene_threshold
        self.min_scene_duration = min_scene_duration
        self.max_scene_duration = max_scene_duration
        self.sample_rate = sample_rate
        self.detection_method = detection_method
        self.preserve_format = preserve_format
        self.target_fps = target_fps
        self.target_resolution = target_resolution

    def _load_video(self, content: Union[str, bytes, Path]) -> "VideoFileClip":
        """Load video from various input types."""
        if VideoFileClip is None:
            raise ImportError(
                "moviepy is required for video chunking. "
                "Install with: pip install moviepy"
            )

        if cv2 is None:
            raise ImportError(
                "opencv-python is required for scene detection. "
                "Install with: pip install opencv-python"
            )

        if isinstance(content, (str, Path)):
            video = VideoFileClip(str(content))
        elif isinstance(content, bytes):
            # Try to detect format from bytes and load accordingly
            video = VideoFileClip(BytesIO(content))
        else:
            raise TypeError(f"Unsupported content type: {type(content)}")
        return video

    def _extract_source_info(self, video: "VideoFileClip", format_ext: str = ".mp4") -> Dict[str, Any]:
        """Extract metadata from video."""
        total_frames = int(video.fps * video.duration) if video.fps else 0

        return {
            "duration": video.duration,
            "fps": video.fps,
            "size": video.size,
            "format": format_ext,
            "total_frames": total_frames,
            "has_audio": video.audio is not None
        }

    def _detect_scene_boundaries(self, video: "VideoFileClip") -> List[float]:
        """
        Detect scene boundaries in the video using the configured method.

        Returns:
            List of timestamps (in seconds) where scene changes occur
        """
        if cv2 is None or np is None:
            raise ImportError(
                "opencv-python and numpy are required for scene detection. "
                "Install with: pip install opencv-python numpy"
            )

        boundaries = []
        duration = video.duration
        sample_interval = 1.0 / self.sample_rate

        prev_frame = None
        prev_hist = None

        logger.info(f"Analyzing video for scene changes (duration: {duration:.2f}s)")

        # Sample frames throughout the video
        for t in range(0, int(duration), int(sample_interval)):
            if t >= duration:
                break

            try:
                # Extract frame at time t
                frame = video.get_frame(t)

                # Convert to OpenCV format (BGR)
                if len(frame.shape) == 3:
                    frame_cv = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_cv = frame

                # Detect scene change
                is_scene_change = False

                if self.detection_method in ["histogram", "combined"]:
                    is_scene_change |= self._detect_histogram_change(frame_cv, prev_hist)
                    prev_hist = self._compute_histogram(frame_cv)

                if self.detection_method in ["ssim", "combined"]:
                    if prev_frame is not None:
                        is_scene_change |= self._detect_ssim_change(frame_cv, prev_frame)

                if is_scene_change and len(boundaries) == 0 or (len(boundaries) > 0 and t - boundaries[-1] >= self.min_scene_duration):
                    boundaries.append(float(t))
                    logger.debug(f"Scene boundary detected at {t}s")

                prev_frame = frame_cv

            except Exception as e:
                logger.warning(f"Failed to process frame at {t}s: {e}")
                continue

        logger.info(f"Detected {len(boundaries)} scene boundaries")
        return boundaries

    def _compute_histogram(self, frame: "np.ndarray") -> "np.ndarray":
        """Compute color histogram for a frame."""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Compute 3D histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])

        # Normalize histogram
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return hist

    def _detect_histogram_change(self, frame: "np.ndarray", prev_hist: Optional["np.ndarray"]) -> bool:
        """Detect scene change based on histogram comparison."""
        if prev_hist is None:
            return False

        current_hist = self._compute_histogram(frame)

        # Compare histograms using correlation
        correlation = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)

        # Convert correlation to difference percentage
        difference = (1 - correlation) * 100

        return difference > self.scene_threshold

    def _detect_ssim_change(self, frame: "np.ndarray", prev_frame: "np.ndarray") -> bool:
        """Detect scene change based on Structural Similarity Index."""
        # Convert frames to grayscale for SSIM calculation
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize frames to consistent size for comparison
        target_size = (320, 240)
        gray1 = cv2.resize(gray1, target_size)
        gray2 = cv2.resize(gray2, target_size)

        # Compute SSIM
        score = self._compute_ssim(gray1, gray2)

        # Convert SSIM to difference percentage
        difference = (1 - score) * 100

        return difference > self.scene_threshold

    def _compute_ssim(self, img1: "np.ndarray", img2: "np.ndarray") -> float:
        """Compute Structural Similarity Index between two images."""
        # Constants for SSIM calculation
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim_map = numerator / denominator
        return ssim_map.mean()

    def _create_segments_from_boundaries(self, video: "VideoFileClip", boundaries: List[float]) -> List[Tuple[float, float]]:
        """Create segment ranges from scene boundaries."""
        if not boundaries:
            # No scenes detected, return single segment
            return [(0.0, video.duration)]

        segments = []
        start_time = 0.0

        # Add boundaries that don't start at 0
        if boundaries[0] > 0:
            boundaries.insert(0, 0.0)

        # Create segments between boundaries
        for i, boundary in enumerate(boundaries[1:], 1):
            segment_duration = boundary - start_time

            # Enforce maximum scene duration
            if segment_duration > self.max_scene_duration:
                # Split long scenes into smaller segments
                num_splits = math.ceil(segment_duration / self.max_scene_duration)
                split_duration = segment_duration / num_splits

                for j in range(num_splits):
                    split_start = start_time + j * split_duration
                    split_end = min(start_time + (j + 1) * split_duration, boundary)
                    segments.append((split_start, split_end))
            else:
                segments.append((start_time, boundary))

            start_time = boundary

        # Add final segment if needed
        if start_time < video.duration:
            segments.append((start_time, video.duration))

        return segments

    def _segment_to_content(self, video: "VideoFileClip", start_time: float, end_time: float) -> bytes:
        """Extract segment content as bytes."""
        try:
            # Create video segment
            segment = video.subclip(start_time, end_time)

            # Apply transformations if specified
            if self.target_fps:
                segment = segment.set_fps(self.target_fps)

            if self.target_resolution:
                segment = segment.resize(self.target_resolution)

            # Write to temporary file and read as bytes
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                segment.write_videofile(temp_file.name, verbose=False, logger=None)
                temp_file.flush()

                with open(temp_file.name, 'rb') as f:
                    content = f.read()

                # Clean up temporary file
                os.unlink(temp_file.name)

            segment.close()
            return content

        except Exception as e:
            logger.error(f"Failed to extract segment {start_time}-{end_time}: {e}")
            return b'error_extracting_segment'

    def chunk(self, content: Union[str, bytes, Path]) -> ChunkingResult:
        """
        Chunk video content based on scene detection.

        Args:
            content: Video file path, bytes, or Path object

        Returns:
            ChunkingResult containing video segments and metadata
        """
        start_time = time.time()
        segments = []

        try:
            # Load video
            video = self._load_video(content)
            total_duration = video.duration

            # Get format information
            if isinstance(content, (str, Path)):
                format_ext = Path(content).suffix.lower() or '.mp4'
            else:
                format_ext = '.mp4'  # Default for byte content

            # Extract source metadata
            source_metadata = self._extract_source_info(video, format_ext)

            # Detect scene boundaries
            boundaries = self._detect_scene_boundaries(video)

            # Create segments from boundaries
            segment_ranges = self._create_segments_from_boundaries(video, boundaries)

            logger.info(f"Creating {len(segment_ranges)} scene-based segments")

            # Create chunks for each segment
            for segment_index, (start_time_seg, end_time_seg) in enumerate(segment_ranges):
                duration = end_time_seg - start_time_seg

                # Skip segments that are too short
                if duration < self.min_scene_duration:
                    continue

                # Extract segment content
                segment_content = self._segment_to_content(video, start_time_seg, end_time_seg)

                # Create metadata for this chunk
                metadata = ChunkMetadata(
                    source="unknown" if isinstance(content, bytes) else str(content),
                    source_type="video_file",
                    position=f"scene {start_time_seg:.2f}s-{end_time_seg:.2f}s",
                    offset=int(start_time_seg * 1000),  # Convert to milliseconds
                    length=len(segment_content),
                    extra={
                        "chunk_id": f"scene_video_chunk_{segment_index+1}",
                        "segment_index": segment_index,
                        "duration_seconds": duration,
                        "start_time_seconds": start_time_seg,
                        "end_time_seconds": end_time_seg,
                        "scene_boundaries": len(boundaries),
                        "detection_method": self.detection_method,
                        "format": format_ext,
                        "boundary_type": "scene_based",
                        "char_count": 0,  # Not applicable for video
                        "word_count": 0,  # Not applicable for video
                        "sentence_count": 0  # Not applicable for video
                    }
                )

                chunk = Chunk(
                    id=f"scene_segment_{segment_index+1}",
                    content=segment_content,
                    modality=ModalityType.VIDEO,
                    metadata=metadata
                )
                segments.append(chunk)

            # Close video resource
            video.close()

        except (FileNotFoundError, ValueError, TypeError, ImportError) as e:
            # Re-raise specific errors for proper handling
            raise e
        except Exception as e:
            logger.error(f"Failed to process video with scene detection: {e}")
            # Return empty result on unexpected errors
            segments = []
            source_metadata = {"error": str(e)}

        processing_time = time.time() - start_time

        # Build source_info dictionary
        source_info = {
            "total_segments": len(segments),
            "source_duration": total_duration if 'total_duration' in locals() else 0,
            "scene_threshold": self.scene_threshold,
            "detection_method": self.detection_method,
            "min_scene_duration": self.min_scene_duration,
            "max_scene_duration": self.max_scene_duration
        }

        # Add source metadata if available
        if 'source_metadata' in locals():
            source_info.update(source_metadata)

        return ChunkingResult(
            chunks=segments,
            processing_time=processing_time,
            source_info=source_info
        )

    def get_chunk_size_estimate(self, content_size: int, **kwargs) -> int:
        """
        Estimate the number of chunks that will be created.

        For scene-based chunking, this is difficult to predict precisely,
        so we provide a rough estimate based on typical scene lengths.
        """
        # Rough estimate: assume average scene length of 10 seconds for typical video
        # and typical video bitrate to estimate duration from content size
        estimated_duration = content_size / (1024 * 1024)  # Very rough MB to minutes conversion
        estimated_scenes = max(1, int(estimated_duration * 60 / 10))  # ~10 seconds per scene average

        return min(estimated_scenes, int(estimated_duration * 60 / self.min_scene_duration))

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration parameters."""
        return {
            "scene_threshold": self.scene_threshold,
            "min_scene_duration": self.min_scene_duration,
            "max_scene_duration": self.max_scene_duration,
            "sample_rate": self.sample_rate,
            "detection_method": self.detection_method,
            "preserve_format": self.preserve_format,
            "target_fps": self.target_fps,
            "target_resolution": self.target_resolution
        }

    def chunk_stream(self, content_stream: Iterator[Union[str, bytes]], **kwargs) -> ChunkingResult:
        """
        Process streamed video content.

        For video content, we need to accumulate the full stream before processing
        since scene detection requires access to the complete video.
        """
        # Accumulate all streamed content
        accumulated_content = b""
        for chunk in content_stream:
            if isinstance(chunk, str):
                accumulated_content += chunk.encode('utf-8')
            else:
                accumulated_content += chunk

        # Process the accumulated content as a single unit
        return self.chunk(accumulated_content)
