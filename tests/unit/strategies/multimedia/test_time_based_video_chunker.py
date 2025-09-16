"""
Unit tests for TimeBasedVideoChunker.

These tests verify the functionality of the time-based video chunking strategy,
including initialization, chunking logic, streaming, and integration with the registry.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typing import Iterator

from chunking_strategy.core.base import ChunkingResult, Chunk, ChunkMetadata, ModalityType
from chunking_strategy.strategies.multimedia.time_based_video import TimeBasedVideoChunker


class TestTimeBasedVideoChunker:
    """Test suite for TimeBasedVideoChunker."""

    def test_initialization_default_parameters(self):
        """Test chunker initialization with default parameters."""
        chunker = TimeBasedVideoChunker()

        assert chunker.segment_duration == 30.0
        assert chunker.overlap_duration == 0.0
        assert chunker.preserve_format is True
        assert chunker.target_fps is None
        assert chunker.target_resolution is None

    def test_initialization_custom_parameters(self):
        """Test chunker initialization with custom parameters."""
        chunker = TimeBasedVideoChunker(
            segment_duration=15.0,
            overlap_duration=2.0,
            preserve_format=False,
            target_fps=24,
            target_resolution=(1280, 720)
        )

        assert chunker.segment_duration == 15.0
        assert chunker.overlap_duration == 2.0
        assert chunker.preserve_format is False
        assert chunker.target_fps == 24
        assert chunker.target_resolution == (1280, 720)

    def test_initialization_validation_segment_duration(self):
        """Test parameter validation for segment_duration."""
        with pytest.raises(ValueError, match="segment_duration must be positive"):
            TimeBasedVideoChunker(segment_duration=-1.0)

        with pytest.raises(ValueError, match="segment_duration must be positive"):
            TimeBasedVideoChunker(segment_duration=0.0)

    def test_initialization_validation_overlap_duration(self):
        """Test parameter validation for overlap_duration."""
        with pytest.raises(ValueError, match="overlap_duration cannot be negative"):
            TimeBasedVideoChunker(overlap_duration=-1.0)

        with pytest.raises(ValueError, match="overlap_duration must be less than segment_duration"):
            TimeBasedVideoChunker(segment_duration=10.0, overlap_duration=15.0)

        with pytest.raises(ValueError, match="overlap_duration must be less than segment_duration"):
            TimeBasedVideoChunker(segment_duration=10.0, overlap_duration=10.0)

    def test_initialization_validation_target_fps(self):
        """Test parameter validation for target_fps."""
        with pytest.raises(ValueError, match="target_fps must be positive"):
            TimeBasedVideoChunker(target_fps=-1)

        with pytest.raises(ValueError, match="target_fps must be positive"):
            TimeBasedVideoChunker(target_fps=0)

    def test_initialization_validation_target_resolution(self):
        """Test parameter validation for target_resolution."""
        with pytest.raises(ValueError, match="target_resolution must be a \\(width, height\\) tuple"):
            TimeBasedVideoChunker(target_resolution="invalid")

        with pytest.raises(ValueError, match="target_resolution must be a \\(width, height\\) tuple"):
            TimeBasedVideoChunker(target_resolution=(100,))

        with pytest.raises(ValueError, match="target_resolution dimensions must be positive"):
            TimeBasedVideoChunker(target_resolution=(0, 720))

        with pytest.raises(ValueError, match="target_resolution dimensions must be positive"):
            TimeBasedVideoChunker(target_resolution=(1280, -720))

    @patch('chunking_strategy.strategies.multimedia.time_based_video.VideoFileClip')
    def test_chunk_basic_functionality(self, mock_video_clip):
        """Test basic chunking functionality with a video file."""
        # Mock video clip
        mock_video = MagicMock()
        mock_video.duration = 60.0
        mock_video.fps = 25.0
        mock_video.size = (1920, 1080)
        mock_video.audio = MagicMock()  # Has audio
        mock_video_clip.return_value = mock_video

        # Mock subclip method
        mock_segment = MagicMock()
        mock_segment.fps = 25.0
        mock_segment.size = (1920, 1080)
        mock_segment.write_videofile = MagicMock()
        mock_video.subclip.return_value = mock_segment

        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('os.path.exists', return_value=True):

            chunker = TimeBasedVideoChunker(segment_duration=30.0)

            # Mock video file path
            video_path = "test_video.mp4"

            result = chunker.chunk(video_path)

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) == 2  # 60s video / 30s segments
            assert result.processing_time > 0

            # Check first chunk
            first_chunk = result.chunks[0]
            assert isinstance(first_chunk, Chunk)
            assert isinstance(first_chunk.content, bytes)
            assert first_chunk.metadata.extra["start_time_seconds"] == 0.0
            assert first_chunk.metadata.extra["duration_seconds"] == 30.0

            # Check second chunk
            second_chunk = result.chunks[1]
            assert second_chunk.metadata.extra["start_time_seconds"] == 30.0
            assert second_chunk.metadata.extra["duration_seconds"] == 30.0

    @patch('chunking_strategy.strategies.multimedia.time_based_video.VideoFileClip')
    def test_chunk_with_overlap(self, mock_video_clip):
        """Test chunking with overlap between segments."""
        # Mock video clip
        mock_video = MagicMock()
        mock_video.duration = 50.0
        mock_video.fps = 30.0
        mock_video.size = (1280, 720)
        mock_video.audio = None  # No audio
        mock_video_clip.return_value = mock_video

        # Mock subclip method
        mock_segment = MagicMock()
        mock_segment.fps = 30.0
        mock_segment.size = (1280, 720)
        mock_segment.write_videofile = MagicMock()
        mock_video.subclip.return_value = mock_segment

        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('os.path.exists', return_value=True):

            chunker = TimeBasedVideoChunker(segment_duration=20.0, overlap_duration=5.0)

            result = chunker.chunk("test_video.mp4")

            assert isinstance(result, ChunkingResult)
            # With 20s segments and 5s overlap (15s effective step):
            # 0-20s, 15-35s, 30-50s, 45-50s = 4 segments (last one is 5s)
            assert len(result.chunks) == 4

            # Verify overlap structure
            assert result.chunks[0].metadata.extra["start_time_seconds"] == 0.0
            assert result.chunks[1].metadata.extra["start_time_seconds"] == 15.0
            assert result.chunks[2].metadata.extra["start_time_seconds"] == 30.0
            assert result.chunks[3].metadata.extra["start_time_seconds"] == 45.0

    @patch('chunking_strategy.strategies.multimedia.time_based_video.VideoFileClip')
    def test_chunk_multiple_formats(self, mock_video_clip):
        """Test chunking with different video formats."""
        formats_to_test = [
            ("test_video.mp4", ".mp4"),
            ("test_video.avi", ".avi"),
            ("test_video.mov", ".mov"),
            ("test_video.mkv", ".mkv")
        ]

        for video_path, expected_format in formats_to_test:
            # Mock video clip
            mock_video = MagicMock()
            mock_video.duration = 30.0
            mock_video.fps = 24.0
            mock_video.size = (640, 480)
            mock_video.audio = MagicMock()
            mock_video_clip.return_value = mock_video

            # Mock subclip method
            mock_segment = MagicMock()
            mock_segment.fps = 24.0
            mock_segment.size = (640, 480)
            mock_segment.write_videofile = MagicMock()
            mock_video.subclip.return_value = mock_segment

            # Mock file operations
            with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
                 patch('tempfile.NamedTemporaryFile'), \
                 patch('os.unlink'), \
                 patch('os.path.exists', return_value=True):

                chunker = TimeBasedVideoChunker(segment_duration=15.0)

                result = chunker.chunk(video_path)

                assert len(result.chunks) >= 1
                assert result.chunks[0].metadata.extra["format"] == expected_format

    def test_get_chunk_size_estimate(self):
        """Test chunk size estimation."""
        chunker = TimeBasedVideoChunker(segment_duration=30.0)

        # Test various durations
        assert chunker.get_chunk_size_estimate(0) == 0
        assert chunker.get_chunk_size_estimate(15) == 1  # Less than segment duration
        assert chunker.get_chunk_size_estimate(30) == 1  # Exactly segment duration
        assert chunker.get_chunk_size_estimate(45) == 2  # 1.5 segments
        assert chunker.get_chunk_size_estimate(90) == 3  # 3 segments

    def test_get_chunk_size_estimate_with_overlap(self):
        """Test chunk size estimation with overlap."""
        chunker = TimeBasedVideoChunker(segment_duration=20.0, overlap_duration=5.0)

        # Effective step is 15s (20s - 5s overlap)
        assert chunker.get_chunk_size_estimate(15) == 1
        assert chunker.get_chunk_size_estimate(30) == 2  # ceil(30/15)
        assert chunker.get_chunk_size_estimate(45) == 3  # ceil(45/15)

    def test_get_configuration(self):
        """Test configuration retrieval."""
        chunker = TimeBasedVideoChunker(
            segment_duration=25.0,
            overlap_duration=3.0,
            preserve_format=False,
            target_fps=60,
            target_resolution=(1920, 1080)
        )

        config = chunker.get_configuration()

        assert config["segment_duration"] == 25.0
        assert config["overlap_duration"] == 3.0
        assert config["preserve_format"] is False
        assert config["target_fps"] == 60
        assert config["target_resolution"] == (1920, 1080)

    def test_registry_integration(self):
        """Test integration with the chunker registry."""
        from chunking_strategy.core.registry import create_chunker

        # Test that the chunker can be created through the registry
        chunker = create_chunker("time_based_video", segment_duration=20.0)

        assert isinstance(chunker, TimeBasedVideoChunker)
        assert chunker.segment_duration == 20.0

    @patch('chunking_strategy.strategies.multimedia.time_based_video.VideoFileClip')
    def test_empty_content_handling(self, mock_video_clip):
        """Test handling of empty or invalid content."""
        chunker = TimeBasedVideoChunker()

        # Test with non-existent file
        mock_video_clip.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            chunker.chunk("non_existent_video.mp4")

    def test_unsupported_content_type(self):
        """Test handling of unsupported content types."""
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            pytest.skip("moviepy not available - install with: pip install moviepy")

        chunker = TimeBasedVideoChunker()

        with pytest.raises(TypeError, match="Unsupported content type"):
            chunker.chunk(12345)  # Invalid content type

    def test_streaming_integration(self):
        """Test streaming functionality."""
        chunker = TimeBasedVideoChunker(segment_duration=15.0)

        # Mock video data stream
        video_data = [b'chunk1', b'chunk2', b'chunk3']

        with patch.object(chunker, 'chunk') as mock_chunk:
            # Mock the chunk method to return a result
            mock_result = ChunkingResult(
                chunks=[
                    Chunk(
                        id="test_chunk_1",
                        content=b'segment1',
                        modality=ModalityType.VIDEO,
                        metadata=ChunkMetadata(
                            source="stream",
                            source_type="video_stream",
                            position="time 0.0s-15.0s",
                            offset=0,
                            length=8,
                            extra={"segment_index": 0}
                        )
                    )
                ],
                processing_time=1.0
            )
            mock_chunk.return_value = mock_result

            # Test streaming
            chunks = list(chunker.chunk_stream(iter(video_data)))

            assert len(chunks) == 1
            assert chunks[0].content == b'segment1'
            mock_chunk.assert_called_once()

    @patch('chunking_strategy.strategies.multimedia.time_based_video.VideoFileClip')
    def test_video_transformations(self, mock_video_clip):
        """Test video transformations (FPS and resolution)."""
        try:
            from moviepy.editor import VideoFileClip as ActualVideoFileClip
        except ImportError:
            pytest.skip("moviepy not available - install with: pip install moviepy")
        # Mock video clip with different properties
        mock_video = MagicMock()
        mock_video.duration = 30.0
        mock_video.fps = 30.0
        mock_video.size = (1920, 1080)
        mock_video.audio = None

        # Mock transformation methods
        mock_video_fps = MagicMock()
        mock_video_fps.fps = 24.0
        mock_video.set_fps.return_value = mock_video_fps

        mock_video_resized = MagicMock()
        mock_video_resized.size = (1280, 720)
        mock_video_fps.resize.return_value = mock_video_resized

        mock_video_clip.return_value = mock_video

        # Mock subclip method
        mock_segment = MagicMock()
        mock_segment.fps = 24.0
        mock_segment.size = (1280, 720)
        mock_segment.write_videofile = MagicMock()
        mock_video_resized.subclip.return_value = mock_segment

        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('os.path.exists', return_value=True):

            chunker = TimeBasedVideoChunker(
                segment_duration=15.0,
                target_fps=24,
                target_resolution=(1280, 720)
            )

            result = chunker.chunk("test_video.mp4")

            # Verify transformations were applied
            mock_video.set_fps.assert_called_with(24)
            mock_video_fps.resize.assert_called_with((1280, 720))

            assert len(result.chunks) >= 1

    @patch('chunking_strategy.strategies.multimedia.time_based_video.VideoFileClip')
    def test_performance_long_video(self, mock_video_clip):
        """Test performance with a longer video file."""
        # Mock long video (10 minutes)
        mock_video = MagicMock()
        mock_video.duration = 600.0  # 10 minutes
        mock_video.fps = 30.0
        mock_video.size = (1920, 1080)
        mock_video.audio = MagicMock()
        mock_video_clip.return_value = mock_video

        # Mock subclip method
        mock_segment = MagicMock()
        mock_segment.fps = 30.0
        mock_segment.size = (1920, 1080)
        mock_segment.write_videofile = MagicMock()
        mock_video.subclip.return_value = mock_segment

        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('os.path.exists', return_value=True):

            chunker = TimeBasedVideoChunker(segment_duration=30.0)

            result = chunker.chunk("long_video.mp4")

            # Should create 20 segments (600s / 30s)
            assert len(result.chunks) == 20
            assert result.processing_time > 0

            # Check segment distribution
            total_expected_duration = 0
            for chunk in result.chunks:
                segment_duration = chunk.metadata.extra["duration_seconds"]
                assert segment_duration <= 30.0
                total_expected_duration += segment_duration

            # Should cover the full video duration
            assert abs(total_expected_duration - 600.0) < 1.0

    @patch('chunking_strategy.strategies.multimedia.time_based_video.VideoFileClip')
    def test_short_video_handling(self, mock_video_clip):
        """Test handling of very short videos."""
        # Mock short video (5 seconds)
        mock_video = MagicMock()
        mock_video.duration = 5.0
        mock_video.fps = 25.0
        mock_video.size = (640, 480)
        mock_video.audio = None
        mock_video_clip.return_value = mock_video

        # Mock subclip method
        mock_segment = MagicMock()
        mock_segment.fps = 25.0
        mock_segment.size = (640, 480)
        mock_segment.write_videofile = MagicMock()
        mock_video.subclip.return_value = mock_segment

        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('os.path.exists', return_value=True):

            chunker = TimeBasedVideoChunker(segment_duration=30.0)

            result = chunker.chunk("short_video.mp4")

            # Should create 1 segment containing the entire short video
            assert len(result.chunks) == 1
            assert result.chunks[0].metadata.extra["duration_seconds"] == 5.0

    def test_error_handling_missing_moviepy(self):
        """Test error handling when moviepy is not available."""
        chunker = TimeBasedVideoChunker()

        # Mock VideoFileClip to be None to simulate missing moviepy
        with patch('chunking_strategy.strategies.multimedia.time_based_video.VideoFileClip', None):
            with pytest.raises(ImportError, match="moviepy is required"):
                chunker.chunk("test_video.mp4")

    @patch('chunking_strategy.strategies.multimedia.time_based_video.VideoFileClip')
    def test_video_properties_preservation(self, mock_video_clip):
        """Test preservation of video properties in metadata."""
        # Mock video with specific properties
        mock_video = MagicMock()
        mock_video.duration = 45.0
        mock_video.fps = 29.97
        mock_video.size = (1280, 720)
        mock_video.audio = MagicMock()  # Has audio
        mock_video_clip.return_value = mock_video

        # Mock subclip method
        mock_segment = MagicMock()
        mock_segment.fps = 29.97
        mock_segment.size = (1280, 720)
        mock_segment.write_videofile = MagicMock()
        mock_video.subclip.return_value = mock_segment

        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'fake_video_data')), \
             patch('tempfile.NamedTemporaryFile'), \
             patch('os.unlink'), \
             patch('os.path.exists', return_value=True):

            chunker = TimeBasedVideoChunker(segment_duration=20.0)

            result = chunker.chunk("test_video.mp4")

            # Check metadata preservation
            assert result.source_info is not None
            source_info = result.source_info

            assert source_info["duration"] == 45.0
            assert source_info["fps"] == 29.97
            assert source_info["size"] == (1280, 720)
            assert source_info["has_audio"] is True
            assert source_info["total_frames"] == int(29.97 * 45.0)

            # Check chunk metadata
            chunk = result.chunks[0]
            assert chunk.metadata.extra["fps"] == 29.97
            assert chunk.metadata.extra["size"] == (1280, 720)
            assert chunk.metadata.extra["boundary_type"] == "time_based"
