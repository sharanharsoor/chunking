"""
Tests for TimeBasedAudioChunker.

This test suite covers:
- Basic time-based chunking functionality
- Different audio formats (MP3, WAV, OGG)
- Edge cases and error handling
- Parameter validation
- Performance characteristics
- Integration with the chunking framework
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import List
import time

from chunking_strategy.strategies.multimedia.time_based_audio import TimeBasedAudioChunker, PYDUB_AVAILABLE
from chunking_strategy.core.base import ChunkingResult, ModalityType
from chunking_strategy import create_chunker

# Skip all tests if pydub is not available
pytestmark = pytest.mark.skipif(not PYDUB_AVAILABLE, reason="pydub not available for audio processing")


class TestTimeBasedAudioChunker:
    """Test suite for TimeBasedAudioChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a basic TimeBasedAudioChunker instance."""
        return TimeBasedAudioChunker(
            segment_duration=10.0,
            overlap_duration=1.0
        )

    @pytest.fixture
    def test_audio_files(self):
        """Return paths to test audio files."""
        audio_dir = Path("test_data/audio_files")
        files = {}
        if audio_dir.exists():
            for file_path in audio_dir.iterdir():
                if file_path.suffix.lower() in ['.mp3', '.wav', '.ogg']:
                    files[file_path.suffix.lower()[1:]] = file_path
        return files

    def test_initialization_default(self):
        """Test chunker initialization with default parameters."""
        chunker = TimeBasedAudioChunker()

        assert chunker.name == "time_based_audio"
        assert chunker.category == "multimedia"
        assert ModalityType.AUDIO in chunker.supported_modalities
        assert chunker.segment_duration == 30.0
        assert chunker.overlap_duration == 0.0
        assert chunker.min_segment_duration == 1.0
        assert chunker.preserve_format == True

    def test_initialization_custom_parameters(self):
        """Test chunker initialization with custom parameters."""
        chunker = TimeBasedAudioChunker(
            segment_duration=15.0,
            overlap_duration=2.0,
            min_segment_duration=0.5,
            preserve_format=False,
            sample_rate=22050,
            channels=1
        )

        assert chunker.segment_duration == 15.0
        assert chunker.overlap_duration == 2.0
        assert chunker.min_segment_duration == 0.5
        assert chunker.preserve_format == False
        assert chunker.sample_rate == 22050
        assert chunker.channels == 1

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Overlap duration >= segment duration
        with pytest.raises(ValueError, match="overlap_duration must be less than segment_duration"):
            TimeBasedAudioChunker(segment_duration=10.0, overlap_duration=10.0)

        with pytest.raises(ValueError, match="overlap_duration must be less than segment_duration"):
            TimeBasedAudioChunker(segment_duration=10.0, overlap_duration=15.0)

        # Min segment duration > segment duration
        with pytest.raises(ValueError, match="min_segment_duration must be less than or equal to segment_duration"):
            TimeBasedAudioChunker(segment_duration=5.0, min_segment_duration=10.0)

    @pytest.mark.skipif(not Path("test_data/audio_files").exists(), reason="Test audio files not available")
    def test_basic_audio_chunking_wav(self, test_audio_files):
        """Test basic chunking functionality with WAV file."""
        if 'wav' not in test_audio_files:
            pytest.skip("WAV test file not available")

        chunker = TimeBasedAudioChunker(segment_duration=10.0, overlap_duration=0.0)
        result = chunker.chunk(test_audio_files['wav'])

        # Basic validation
        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "time_based_audio"
        assert len(result.chunks) > 0
        assert result.processing_time > 0

        # Check chunks
        for i, chunk in enumerate(result.chunks):
            assert chunk.id == f"time_based_audio_chunk_{i}"
            assert chunk.modality == ModalityType.AUDIO
            assert isinstance(chunk.content, bytes)
            assert chunk.size > 0

            # Check metadata
            assert chunk.metadata.source == str(test_audio_files['wav'])
            assert chunk.metadata.source_type == "file"
            assert "start_time" in chunk.metadata.extra
            assert "end_time" in chunk.metadata.extra
            assert "duration" in chunk.metadata.extra
            assert "sample_rate" in chunk.metadata.extra
            assert "channels" in chunk.metadata.extra

    @pytest.mark.skipif(not Path("test_data/audio_files").exists(), reason="Test audio files not available")
    def test_chunking_with_overlap(self, test_audio_files):
        """Test chunking with overlap between segments."""
        if 'wav' not in test_audio_files:
            pytest.skip("WAV test file not available")

        chunker = TimeBasedAudioChunker(segment_duration=15.0, overlap_duration=3.0)
        result = chunker.chunk(test_audio_files['wav'])

        assert len(result.chunks) >= 1

        # Check overlap metadata
        if len(result.chunks) > 1:
            for i, chunk in enumerate(result.chunks[1:], 1):
                assert chunk.metadata.extra["overlap_with_previous"] == True

                # Check timing makes sense with overlap
                prev_chunk = result.chunks[i-1]
                curr_start = chunk.metadata.extra["start_time"]
                prev_end = prev_chunk.metadata.extra["end_time"]

                # Current chunk should start before previous chunk ends (due to overlap)
                assert curr_start < prev_end

    @pytest.mark.skipif(not Path("test_data/audio_files").exists(), reason="Test audio files not available")
    def test_multiple_audio_formats(self, test_audio_files):
        """Test chunking with different audio formats."""
        chunker = TimeBasedAudioChunker(segment_duration=20.0)

        formats_tested = []
        for format_name, file_path in test_audio_files.items():
            try:
                result = chunker.chunk(file_path)
                assert len(result.chunks) >= 1
                assert result.strategy_used == "time_based_audio"

                # Check format is preserved in metadata
                first_chunk = result.chunks[0]
                assert first_chunk.metadata.extra["format"] == format_name

                formats_tested.append(format_name)
            except Exception as e:
                pytest.fail(f"Failed to process {format_name} file: {e}")

        assert len(formats_tested) > 0, "No audio formats could be tested"

    def test_chunk_size_estimation(self):
        """Test chunk size estimation functionality."""
        chunker = TimeBasedAudioChunker(segment_duration=10.0, overlap_duration=0.0)

        # Short audio
        assert chunker.get_chunk_size_estimate(5.0) == 1

        # Exact fit
        assert chunker.get_chunk_size_estimate(10.0) == 1

        # Multiple segments
        assert chunker.get_chunk_size_estimate(25.0) == 3  # 0-10, 10-20, 20-25

        # With overlap
        chunker_overlap = TimeBasedAudioChunker(segment_duration=10.0, overlap_duration=2.0)
        estimate = chunker_overlap.get_chunk_size_estimate(30.0)
        assert estimate >= 3  # Should need more chunks due to overlap

    def test_configuration_retrieval(self):
        """Test configuration retrieval."""
        chunker = TimeBasedAudioChunker(
            segment_duration=25.0,
            overlap_duration=5.0,
            preserve_format=False
        )

        config = chunker.get_config()

        assert config["name"] == "time_based_audio"
        assert config["segment_duration"] == 25.0
        assert config["overlap_duration"] == 5.0
        assert config["preserve_format"] == False

    def test_registry_integration(self):
        """Test integration with the chunker registry."""
        # Test chunker can be created via registry
        chunker = create_chunker("time_based_audio", segment_duration=20.0)
        assert chunker is not None
        assert isinstance(chunker, TimeBasedAudioChunker)
        assert chunker.segment_duration == 20.0

    def test_empty_content_handling(self):
        """Test handling of empty or invalid content."""
        chunker = TimeBasedAudioChunker()

        # Non-existent file
        with pytest.raises((FileNotFoundError, ValueError)):
            chunker.chunk("nonexistent_audio.wav")

    def test_streaming_interface(self, test_audio_files):
        """Test streaming chunk interface."""
        if not test_audio_files:
            pytest.skip("No test audio files available")

        chunker = TimeBasedAudioChunker(segment_duration=10.0)
        file_path = next(iter(test_audio_files.values()))

        # Test streaming
        chunks = list(chunker.chunk_stream([str(file_path)]))
        assert len(chunks) > 0

        # Compare with regular chunking
        regular_result = chunker.chunk(file_path)
        assert len(chunks) == len(regular_result.chunks)

    def test_performance_characteristics(self, test_audio_files):
        """Test performance characteristics."""
        if not test_audio_files:
            pytest.skip("No test audio files available")

        chunker = TimeBasedAudioChunker(segment_duration=30.0)
        file_path = next(iter(test_audio_files.values()))

        start_time = time.time()
        result = chunker.chunk(file_path)
        processing_time = time.time() - start_time

        # Should process relatively quickly
        assert processing_time < 30.0, "Audio chunking took too long"
        assert result.processing_time > 0

        # Should have reasonable chunk distribution
        if len(result.chunks) > 1:
            durations = [chunk.metadata.extra["duration"] for chunk in result.chunks[:-1]]
            # All chunks except possibly the last should be close to target duration
            for duration in durations:
                assert abs(duration - 30.0) < 1.0, f"Chunk duration {duration} too far from target 30.0s"

    def test_audio_properties_preservation(self, test_audio_files):
        """Test that audio properties are correctly preserved in metadata."""
        if not test_audio_files:
            pytest.skip("No test audio files available")

        chunker = TimeBasedAudioChunker(segment_duration=20.0)
        file_path = next(iter(test_audio_files.values()))

        result = chunker.chunk(file_path)
        assert len(result.chunks) > 0

        # Check that audio properties are recorded
        first_chunk = result.chunks[0]
        extra = first_chunk.metadata.extra

        assert "sample_rate" in extra
        assert "channels" in extra
        assert "format" in extra
        assert extra["sample_rate"] > 0
        assert extra["channels"] > 0
        assert len(extra["format"]) > 0

    def test_min_segment_duration_handling(self, test_audio_files):
        """Test handling of minimum segment duration."""
        if not test_audio_files:
            pytest.skip("No test audio files available")

        # Use a large segment duration and small min duration
        chunker = TimeBasedAudioChunker(
            segment_duration=100.0,  # Longer than most test files
            min_segment_duration=2.0
        )
        file_path = next(iter(test_audio_files.values()))

        result = chunker.chunk(file_path)

        # Should still create chunks even if audio is shorter than segment_duration
        if result.source_info.get("total_duration", 0) >= 2.0:
            assert len(result.chunks) >= 1

        # Check that all chunks meet minimum duration (except possibly the last one)
        for chunk in result.chunks:
            duration = chunk.metadata.extra["duration"]
            # Allow some tolerance for the last chunk
            if chunk != result.chunks[-1]:
                assert duration >= chunker.min_segment_duration - 0.1


class TestTimeBasedAudioIntegration:
    """Integration tests for TimeBasedAudioChunker."""

    @pytest.mark.skipif(not PYDUB_AVAILABLE, reason="pydub not available")
    def test_chunker_supports_modality(self):
        """Test that chunker properly reports modality support."""
        chunker = TimeBasedAudioChunker()

        assert chunker.supports_modality(ModalityType.AUDIO)
        assert not chunker.supports_modality(ModalityType.TEXT)
        assert not chunker.supports_modality(ModalityType.IMAGE)

    @pytest.mark.skipif(not PYDUB_AVAILABLE, reason="pydub not available")
    def test_chunk_metadata_completeness(self, test_audio_files):
        """Test that chunk metadata is complete and correctly formatted."""
        if not test_audio_files:
            pytest.skip("No test audio files available")

        chunker = TimeBasedAudioChunker(segment_duration=15.0)
        file_path = next(iter(test_audio_files.values()))

        result = chunker.chunk(file_path)
        assert len(result.chunks) > 0

        for chunk in result.chunks:
            # Test required metadata fields
            assert chunk.id is not None
            assert chunk.content is not None
            assert chunk.modality == ModalityType.AUDIO
            assert chunk.size > 0

            # Test chunk metadata structure
            metadata = chunk.metadata
            assert metadata.source is not None
            assert metadata.source_type in ["file", "bytes"]
            assert metadata.position is not None
            assert metadata.offset >= 0
            assert metadata.length > 0

            # Test extra metadata
            extra = metadata.extra
            required_fields = ["start_time", "end_time", "duration", "sample_rate", "channels", "format", "chunker_used", "chunking_strategy"]
            for field in required_fields:
                assert field in extra, f"Missing required field: {field}"


@pytest.fixture
def test_audio_files():
    """Global fixture for test audio files."""
    audio_dir = Path("test_data/audio_files")
    files = {}
    if audio_dir.exists():
        for file_path in audio_dir.iterdir():
            if file_path.suffix.lower() in ['.mp3', '.wav', '.ogg']:
                files[file_path.suffix.lower()[1:]] = file_path
    return files
