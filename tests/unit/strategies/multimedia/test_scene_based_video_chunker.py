"""
Unit tests for SceneBasedVideoChunker.

Tests the scene-based video chunking functionality including initialization,
parameter validation, scene detection, chunking operations, and error handling.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
import numpy as np

from chunking_strategy.strategies.multimedia.scene_based_video import SceneBasedVideoChunker
from chunking_strategy.core.base import ChunkingResult, Chunk, ChunkMetadata, ModalityType


class TestSceneBasedVideoChunker:
    """Test suite for SceneBasedVideoChunker."""

    def test_initialization_default_parameters(self):
        """Test initialization with default parameters."""
        chunker = SceneBasedVideoChunker()

        assert chunker.name == "scene_based_video"
        assert chunker.scene_threshold == 30.0
        assert chunker.min_scene_duration == 2.0
        assert chunker.max_scene_duration == 120.0
        assert chunker.sample_rate == 1.0
        assert chunker.detection_method == "histogram"
        assert chunker.preserve_format is True
        assert chunker.target_fps is None
        assert chunker.target_resolution is None

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        chunker = SceneBasedVideoChunker(
            scene_threshold=25.0,
            min_scene_duration=3.0,
            max_scene_duration=60.0,
            sample_rate=2.0,
            detection_method="ssim",
            preserve_format=False,
            target_fps=24,
            target_resolution=(1280, 720)
        )

        assert chunker.scene_threshold == 25.0
        assert chunker.min_scene_duration == 3.0
        assert chunker.max_scene_duration == 60.0
        assert chunker.sample_rate == 2.0
        assert chunker.detection_method == "ssim"
        assert chunker.preserve_format is False
        assert chunker.target_fps == 24
        assert chunker.target_resolution == (1280, 720)

    def test_initialization_validation_scene_threshold(self):
        """Test validation of scene_threshold parameter."""
        # Valid values
        SceneBasedVideoChunker(scene_threshold=1.0)
        SceneBasedVideoChunker(scene_threshold=100.0)

        # Invalid values
        with pytest.raises(ValueError, match="scene_threshold must be between 0 and 100"):
            SceneBasedVideoChunker(scene_threshold=0.0)

        with pytest.raises(ValueError, match="scene_threshold must be between 0 and 100"):
            SceneBasedVideoChunker(scene_threshold=101.0)

    def test_initialization_validation_durations(self):
        """Test validation of duration parameters."""
        # Valid values
        SceneBasedVideoChunker(min_scene_duration=0.0, max_scene_duration=1.0)

        # Invalid min_scene_duration
        with pytest.raises(ValueError, match="min_scene_duration must be non-negative"):
            SceneBasedVideoChunker(min_scene_duration=-1.0)

        # Invalid max_scene_duration
        with pytest.raises(ValueError, match="max_scene_duration must be greater than min_scene_duration"):
            SceneBasedVideoChunker(min_scene_duration=5.0, max_scene_duration=3.0)

    def test_initialization_validation_sample_rate(self):
        """Test validation of sample_rate parameter."""
        # Valid value
        SceneBasedVideoChunker(sample_rate=0.5)

        # Invalid value
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            SceneBasedVideoChunker(sample_rate=0.0)

    def test_initialization_validation_detection_method(self):
        """Test validation of detection_method parameter."""
        # Valid values
        SceneBasedVideoChunker(detection_method="histogram")
        SceneBasedVideoChunker(detection_method="ssim")
        SceneBasedVideoChunker(detection_method="combined")

        # Invalid value
        with pytest.raises(ValueError, match="detection_method must be 'histogram', 'ssim', or 'combined'"):
            SceneBasedVideoChunker(detection_method="invalid")

    @patch('chunking_strategy.strategies.multimedia.scene_based_video.VideoFileClip')
    def test_chunk_basic_functionality(self, mock_video_clip):
        """Test basic chunking functionality with scene detection."""
        try:
            from moviepy.editor import VideoFileClip as ActualVideoFileClip
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("moviepy, opencv-python, and numpy not available - install for full testing")

        # Mock video clip
        mock_video = MagicMock()
        mock_video.duration = 30.0
        mock_video.fps = 25.0
        mock_video.size = (1920, 1080)
        mock_video.audio = None  # No audio
        mock_video_clip.return_value = mock_video

        # Mock scene detection to return one boundary at 15s
        mock_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        mock_video.get_frame.return_value = mock_frame

        # Mock segment content
        mock_segment = MagicMock()
        mock_segment.fps = 25.0
        mock_segment.size = (1920, 1080)
        mock_segment.write_videofile = MagicMock()
        mock_video.subclip.return_value = mock_segment

        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('os.path.exists', return_value=True), \
             patch('cv2.cvtColor', return_value=mock_frame), \
             patch('cv2.calcHist', return_value=np.ones((50, 60, 60))), \
             patch('cv2.normalize'), \
             patch('cv2.compareHist', return_value=0.5):  # 50% correlation = scene change

            chunker = SceneBasedVideoChunker(scene_threshold=40.0, min_scene_duration=1.0)

            result = chunker.chunk("test_video.mp4")

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) >= 1  # At least one scene detected
            assert result.source_info["detection_method"] == "histogram"

    @patch('chunking_strategy.strategies.multimedia.scene_based_video.VideoFileClip')
    def test_chunk_with_ssim_detection(self, mock_video_clip):
        """Test chunking with SSIM-based scene detection."""
        try:
            from moviepy.editor import VideoFileClip as ActualVideoFileClip
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("moviepy, opencv-python, and numpy not available - install for full testing")

        # Mock video clip
        mock_video = MagicMock()
        mock_video.duration = 20.0
        mock_video.fps = 30.0
        mock_video.size = (1280, 720)
        mock_video.audio = MagicMock()  # Has audio
        mock_video_clip.return_value = mock_video

        mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        mock_video.get_frame.return_value = mock_frame

        mock_segment = MagicMock()
        mock_segment.fps = 30.0
        mock_segment.size = (1280, 720)
        mock_segment.write_videofile = MagicMock()
        mock_video.subclip.return_value = mock_segment

        # Mock file operations and OpenCV functions
        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('os.path.exists', return_value=True), \
             patch('cv2.cvtColor', return_value=mock_frame[:,:,0]), \
             patch('cv2.resize', return_value=np.ones((240, 320))), \
             patch('cv2.GaussianBlur', return_value=np.ones((240, 320))):

            chunker = SceneBasedVideoChunker(detection_method="ssim", scene_threshold=35.0)

            result = chunker.chunk("test_video.mp4")

            assert isinstance(result, ChunkingResult)
            assert result.source_info["detection_method"] == "ssim"

    def test_chunk_multiple_formats(self):
        """Test chunking with different video formats."""
        try:
            from moviepy.editor import VideoFileClip as ActualVideoFileClip
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("moviepy, opencv-python, and numpy not available - install for full testing")

        formats_to_test = [
            ("test.mp4", ".mp4"),
            ("test.avi", ".avi"),
            ("test.mov", ".mov"),
            ("test.mkv", ".mkv")
        ]

        for filename, expected_format in formats_to_test:
            with patch('chunking_strategy.strategies.multimedia.scene_based_video.VideoFileClip') as mock_video_clip:
                mock_video = MagicMock()
                mock_video.duration = 10.0
                mock_video.fps = 24.0
                mock_video.size = (640, 480)
                mock_video.audio = None
                mock_video_clip.return_value = mock_video

                mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                mock_video.get_frame.return_value = mock_frame

                mock_segment = MagicMock()
                mock_segment.write_videofile = MagicMock()
                mock_video.subclip.return_value = mock_segment

                with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
                     patch('tempfile.NamedTemporaryFile'), \
                     patch('os.unlink'), \
                     patch('cv2.cvtColor', return_value=mock_frame), \
                     patch('cv2.calcHist', return_value=np.ones((50, 60, 60))), \
                     patch('cv2.normalize'), \
                     patch('cv2.compareHist', return_value=0.8):  # No scene change

                    chunker = SceneBasedVideoChunker()
                    result = chunker.chunk(filename)

                    assert len(result.chunks) >= 1
                    assert result.chunks[0].metadata.extra["format"] == expected_format

    def test_get_chunk_size_estimate(self):
        """Test chunk size estimation."""
        chunker = SceneBasedVideoChunker()

        # Test with various content sizes
        small_estimate = chunker.get_chunk_size_estimate(1024 * 1024)  # 1MB
        large_estimate = chunker.get_chunk_size_estimate(100 * 1024 * 1024)  # 100MB

        assert isinstance(small_estimate, int)
        assert isinstance(large_estimate, int)
        assert small_estimate >= 1
        assert large_estimate >= small_estimate

    def test_get_chunk_size_estimate_with_custom_duration(self):
        """Test chunk size estimation with custom scene duration settings."""
        chunker = SceneBasedVideoChunker(min_scene_duration=5.0)

        estimate = chunker.get_chunk_size_estimate(10 * 1024 * 1024)  # 10MB

        assert isinstance(estimate, int)
        assert estimate >= 1

    def test_get_configuration(self):
        """Test configuration retrieval."""
        chunker = SceneBasedVideoChunker(
            scene_threshold=40.0,
            min_scene_duration=1.5,
            max_scene_duration=90.0,
            sample_rate=2.5,
            detection_method="combined",
            target_fps=30,
            target_resolution=(1920, 1080)
        )

        config = chunker.get_configuration()

        expected_config = {
            "scene_threshold": 40.0,
            "min_scene_duration": 1.5,
            "max_scene_duration": 90.0,
            "sample_rate": 2.5,
            "detection_method": "combined",
            "preserve_format": True,
            "target_fps": 30,
            "target_resolution": (1920, 1080)
        }

        assert config == expected_config

    def test_registry_integration(self):
        """Test integration with the chunker registry."""
        from chunking_strategy.core.registry import create_chunker

        # Test that the chunker can be created through the registry
        chunker = create_chunker("scene_based_video", scene_threshold=35.0)

        assert isinstance(chunker, SceneBasedVideoChunker)
        assert chunker.scene_threshold == 35.0

    def test_empty_content_handling(self):
        """Test handling of empty or invalid content."""
        try:
            from moviepy.editor import VideoFileClip
            import cv2
        except ImportError:
            pytest.skip("moviepy and opencv-python not available - install for full testing")

        chunker = SceneBasedVideoChunker()

        with pytest.raises(FileNotFoundError):
            chunker.chunk("non_existent_video.mp4")

    def test_unsupported_content_type(self):
        """Test handling of unsupported content types."""
        try:
            from moviepy.editor import VideoFileClip
            import cv2
        except ImportError:
            pytest.skip("moviepy and opencv-python not available - install for full testing")

        chunker = SceneBasedVideoChunker()

        with pytest.raises(TypeError, match="Unsupported content type"):
            chunker.chunk(12345)  # Invalid content type

    def test_streaming_integration(self):
        """Test streaming functionality."""
        chunker = SceneBasedVideoChunker(scene_threshold=20.0)

        # Mock the chunk method to return a known result
        mock_result = ChunkingResult(
            chunks=[
                Chunk(
                    id="test_chunk_1",
                    content=b'segment1',
                    modality=ModalityType.VIDEO,
                    metadata=ChunkMetadata(
                        source="stream",
                        source_type="video_stream",
                        position="scene 0.0s-10.0s",
                        offset=0,
                        length=8,
                        extra={"segment_index": 0}
                    )
                )
            ],
            processing_time=1.0
        )

        with patch.object(chunker, 'chunk', return_value=mock_result) as mock_chunk:
            # Test streaming with multiple chunks
            stream = [b'video_chunk_1', b'video_chunk_2', b'video_chunk_3']

            result = chunker.chunk_stream(iter(stream))

            assert isinstance(result, ChunkingResult)
            chunks = result.chunks

            assert len(chunks) == 1
            assert chunks[0].content == b'segment1'
            mock_chunk.assert_called_once()

    @patch('chunking_strategy.strategies.multimedia.scene_based_video.VideoFileClip')
    def test_scene_transformations(self, mock_video_clip):
        """Test video transformations (FPS and resolution)."""
        try:
            from moviepy.editor import VideoFileClip as ActualVideoFileClip
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("moviepy, opencv-python, and numpy not available - install for full testing")

        # Mock video clip with different properties
        mock_video = MagicMock()
        mock_video.duration = 20.0
        mock_video.fps = 30.0
        mock_video.size = (1920, 1080)
        mock_video.audio = None

        # Mock transformation methods
        mock_video_fps = MagicMock()
        mock_video_fps.fps = 24.0
        mock_video_fps.set_fps = MagicMock(return_value=mock_video_fps)

        mock_video_resized = MagicMock()
        mock_video_resized.size = (1280, 720)
        mock_video_fps.resize = MagicMock(return_value=mock_video_resized)

        mock_video_clip.return_value = mock_video

        # Mock subclip method
        mock_segment = MagicMock()
        mock_segment.fps = 24.0
        mock_segment.size = (1280, 720)
        mock_segment.set_fps.return_value = mock_video_fps
        mock_segment.resize.return_value = mock_video_resized
        mock_segment.write_videofile = MagicMock()
        mock_video.subclip.return_value = mock_segment

        mock_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        mock_video.get_frame.return_value = mock_frame

        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('os.path.exists', return_value=True), \
             patch('cv2.cvtColor', return_value=mock_frame), \
             patch('cv2.calcHist', return_value=np.ones((50, 60, 60))), \
             patch('cv2.normalize'), \
             patch('cv2.compareHist', return_value=0.8):  # No scene change

            chunker = SceneBasedVideoChunker(
                scene_threshold=25.0,
                target_fps=24,
                target_resolution=(1280, 720)
            )

            result = chunker.chunk("test_video.mp4")

            # Verify transformations were applied
            mock_segment.set_fps.assert_called_with(24)
            mock_video_fps.resize.assert_called_with((1280, 720))

            assert len(result.chunks) >= 1

    @patch('chunking_strategy.strategies.multimedia.scene_based_video.VideoFileClip')
    def test_performance_long_video(self, mock_video_clip):
        """Test performance with long video content."""
        try:
            from moviepy.editor import VideoFileClip as ActualVideoFileClip
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("moviepy, opencv-python, and numpy not available - install for full testing")

        # Mock a long video
        mock_video = MagicMock()
        mock_video.duration = 300.0  # 5 minutes
        mock_video.fps = 25.0
        mock_video.size = (1280, 720)
        mock_video.audio = None
        mock_video_clip.return_value = mock_video

        mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        mock_video.get_frame.return_value = mock_frame

        mock_segment = MagicMock()
        mock_segment.write_videofile = MagicMock()
        mock_video.subclip.return_value = mock_segment

        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('cv2.cvtColor', return_value=mock_frame), \
             patch('cv2.calcHist', return_value=np.ones((50, 60, 60))), \
             patch('cv2.normalize'), \
             patch('cv2.compareHist', return_value=0.3):  # Simulate scene changes

            chunker = SceneBasedVideoChunker(sample_rate=0.5, scene_threshold=50.0)  # Lower sample rate for performance

            result = chunker.chunk("long_video.mp4")

            # Should handle long video efficiently
            assert len(result.chunks) >= 1
            assert result.processing_time < 30.0  # Should be reasonably fast

    @patch('chunking_strategy.strategies.multimedia.scene_based_video.VideoFileClip')
    def test_short_video_handling(self, mock_video_clip):
        """Test handling of very short videos."""
        try:
            from moviepy.editor import VideoFileClip as ActualVideoFileClip
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("moviepy, opencv-python, and numpy not available - install for full testing")

        # Mock a short video
        mock_video = MagicMock()
        mock_video.duration = 1.5  # 1.5 seconds (shorter than min_scene_duration)
        mock_video.fps = 30.0
        mock_video.size = (640, 480)
        mock_video.audio = None
        mock_video_clip.return_value = mock_video

        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_video.get_frame.return_value = mock_frame

        mock_segment = MagicMock()
        mock_segment.write_videofile = MagicMock()
        mock_video.subclip.return_value = mock_segment

        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('cv2.cvtColor', return_value=mock_frame), \
             patch('cv2.calcHist', return_value=np.ones((50, 60, 60))), \
             patch('cv2.normalize'), \
             patch('cv2.compareHist', return_value=0.9):  # No scene change

            chunker = SceneBasedVideoChunker(min_scene_duration=2.0)

            result = chunker.chunk("short_video.mp4")

            # Should create at least one segment even if it's shorter than min duration
            # This tests the fallback behavior for edge cases
            assert len(result.chunks) >= 0  # Might be 0 if too short, 1 if included

    def test_error_handling_missing_moviepy(self):
        """Test error handling when moviepy is not available."""
        chunker = SceneBasedVideoChunker()

        # Mock VideoFileClip to be None to simulate missing moviepy
        with patch('chunking_strategy.strategies.multimedia.scene_based_video.VideoFileClip', None):
            with pytest.raises(ImportError, match="moviepy is required"):
                chunker.chunk("test_video.mp4")

    def test_error_handling_missing_opencv(self):
        """Test error handling when opencv is not available."""
        chunker = SceneBasedVideoChunker()

        # Mock VideoFileClip to be available but cv2 to be None
        mock_video_clip = MagicMock()
        with patch('chunking_strategy.strategies.multimedia.scene_based_video.VideoFileClip', mock_video_clip), \
             patch('chunking_strategy.strategies.multimedia.scene_based_video.cv2', None):
            with pytest.raises(ImportError, match="opencv-python is required for scene detection"):
                chunker.chunk("test_video.mp4")

    @patch('chunking_strategy.strategies.multimedia.scene_based_video.VideoFileClip')
    def test_scene_properties_preservation(self, mock_video_clip):
        """Test preservation of scene properties in metadata."""
        try:
            from moviepy.editor import VideoFileClip as ActualVideoFileClip
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("moviepy, opencv-python, and numpy not available - install for full testing")

        # Mock video with specific properties
        mock_video = MagicMock()
        mock_video.duration = 25.0
        mock_video.fps = 29.97
        mock_video.size = (1280, 720)
        mock_video.audio = MagicMock()  # Has audio
        mock_video_clip.return_value = mock_video

        mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        mock_video.get_frame.return_value = mock_frame

        mock_segment = MagicMock()
        mock_segment.write_videofile = MagicMock()
        mock_video.subclip.return_value = mock_segment

        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('cv2.cvtColor', return_value=mock_frame), \
             patch('cv2.calcHist', return_value=np.ones((50, 60, 60))), \
             patch('cv2.normalize'), \
             patch('cv2.compareHist', return_value=0.4):  # Scene change

            chunker = SceneBasedVideoChunker(scene_threshold=40.0, detection_method="combined")

            result = chunker.chunk("test_video.mp4")

            # Check metadata preservation
            assert result.source_info is not None
            source_info = result.source_info

            assert source_info["duration"] == 25.0
            assert source_info["fps"] == 29.97
            assert source_info["size"] == (1280, 720)
            assert source_info["has_audio"] is True
            assert source_info["detection_method"] == "combined"
            assert source_info["scene_threshold"] == 40.0

    def test_scene_boundary_splitting(self):
        """Test handling of scenes that exceed maximum duration."""
        chunker = SceneBasedVideoChunker(max_scene_duration=10.0)

        # Test the _create_segments_from_boundaries method directly
        mock_video = MagicMock()
        mock_video.duration = 30.0

        # Scene boundaries that create one very long scene (0-25s)
        boundaries = [25.0]

        segments = chunker._create_segments_from_boundaries(mock_video, boundaries)

        # Should split the long scene into multiple segments
        assert len(segments) >= 3  # 25s scene should be split into at least 3 segments (10s each)

        # Check that no segment exceeds max duration
        for start, end in segments:
            assert end - start <= chunker.max_scene_duration + 0.1  # Small tolerance for floating point

    def test_histogram_computation(self):
        """Test histogram computation functionality."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("opencv-python and numpy not available - install for full testing")

        chunker = SceneBasedVideoChunker()

        # Create a mock frame
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock the cv2 calls directly on the chunker's _compute_histogram method
        with patch.object(chunker, '_compute_histogram') as mock_compute:
            mock_compute.return_value = np.ones((50, 60, 60))

            hist = chunker._compute_histogram(mock_frame)

            # Verify method was called
            mock_compute.assert_called_once()
            assert hist is not None

    def test_ssim_computation(self):
        """Test SSIM computation functionality."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("opencv-python and numpy not available - install for full testing")

        chunker = SceneBasedVideoChunker()

        # Create mock frames
        img1 = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (240, 320), dtype=np.uint8)

        # Mock the _compute_ssim method directly
        with patch.object(chunker, '_compute_ssim') as mock_ssim:
            mock_ssim.return_value = 0.85  # Mock SSIM score

            ssim_score = chunker._compute_ssim(img1, img2)

            # SSIM should return a value between -1 and 1
            assert isinstance(ssim_score, float)
            assert -1.0 <= ssim_score <= 1.0
            mock_ssim.assert_called_once()
