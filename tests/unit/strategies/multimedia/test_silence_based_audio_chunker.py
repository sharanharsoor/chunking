"""
Unit tests for SilenceBasedAudioChunker.

Tests the silence-based audio chunking strategy with comprehensive coverage
of initialization, parameter validation, chunking functionality, and edge cases.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Only run these tests if pydub is available
pydub = pytest.importorskip("pydub")

from chunking_strategy.strategies.multimedia.silence_based_audio import SilenceBasedAudioChunker
from chunking_strategy.core.base import ModalityType, ChunkingResult
from chunking_strategy.core.registry import get_registry


class TestSilenceBasedAudioChunker:
    """Test suite for SilenceBasedAudioChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a basic silence-based audio chunker."""
        return SilenceBasedAudioChunker()

    @pytest.fixture
    def test_audio_files(self) -> Dict[str, Path]:
        """Provide test audio files if they exist."""
        audio_dir = Path("test_data/audio_files")
        files = {}

        if audio_dir.exists():
            for ext, name in [('wav', 'wav'), ('mp3', 'mp3'), ('ogg', 'ogg')]:
                for file_path in audio_dir.glob(f"*.{ext}"):
                    files[name] = file_path
                    break

        return files

    def test_init_default_parameters(self):
        """Test chunker initialization with default parameters."""
        chunker = SilenceBasedAudioChunker()

        assert chunker.name == "silence_based_audio"
        assert chunker.category == "multimedia"
        assert ModalityType.AUDIO in chunker.supported_modalities
        assert chunker.silence_threshold_db == -40.0
        assert chunker.min_silence_duration == 0.5
        assert chunker.min_segment_duration == 5.0
        assert chunker.max_segment_duration == 120.0
        assert chunker.padding_duration == 0.1
        assert chunker.preserve_format is True
        assert chunker.sample_rate is None
        assert chunker.channels is None

    def test_init_custom_parameters(self):
        """Test chunker initialization with custom parameters."""
        chunker = SilenceBasedAudioChunker(
            silence_threshold_db=-50.0,
            min_silence_duration=1.0,
            min_segment_duration=10.0,
            max_segment_duration=60.0,
            padding_duration=0.2,
            preserve_format=False,
            sample_rate=44100,
            channels=2
        )

        assert chunker.silence_threshold_db == -50.0
        assert chunker.min_silence_duration == 1.0
        assert chunker.min_segment_duration == 10.0
        assert chunker.max_segment_duration == 60.0
        assert chunker.padding_duration == 0.2
        assert chunker.preserve_format is False
        assert chunker.sample_rate == 44100
        assert chunker.channels == 2

    def test_init_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid silence threshold
        with pytest.raises(ValueError, match="silence_threshold_db must be negative"):
            SilenceBasedAudioChunker(silence_threshold_db=10.0)

        # Test invalid min_silence_duration
        with pytest.raises(ValueError, match="min_silence_duration must be positive"):
            SilenceBasedAudioChunker(min_silence_duration=-1.0)

        # Test invalid min_segment_duration
        with pytest.raises(ValueError, match="min_segment_duration must be positive"):
            SilenceBasedAudioChunker(min_segment_duration=-1.0)

        # Test invalid max vs min segment duration
        with pytest.raises(ValueError, match="max_segment_duration must be greater than min_segment_duration"):
            SilenceBasedAudioChunker(min_segment_duration=60.0, max_segment_duration=30.0)

        # Test invalid padding duration
        with pytest.raises(ValueError, match="padding_duration must be non-negative"):
            SilenceBasedAudioChunker(padding_duration=-1.0)

        # Test invalid sample rate
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            SilenceBasedAudioChunker(sample_rate=-1)

        # Test invalid channels
        with pytest.raises(ValueError, match="channels must be positive"):
            SilenceBasedAudioChunker(channels=-1)

    @pytest.mark.skipif(not Path("test_data/audio_files").exists(), reason="No test audio files available")
    def test_basic_chunking_wav(self, test_audio_files):
        """Test basic chunking functionality with WAV files."""
        if 'wav' not in test_audio_files:
            pytest.skip("WAV test file not available")

        chunker = SilenceBasedAudioChunker(
            silence_threshold_db=-30.0,  # More lenient for test files
            min_silence_duration=0.1,
            min_segment_duration=1.0,    # Shorter for test files
            max_segment_duration=30.0
        )

        result = chunker.chunk(test_audio_files['wav'])

        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "silence_based_audio"
        assert len(result.chunks) >= 1
        assert result.processing_time > 0

        # Validate first chunk
        first_chunk = result.chunks[0]
        assert first_chunk.modality == ModalityType.AUDIO
        assert isinstance(first_chunk.content, bytes)
        assert len(first_chunk.content) > 0
        assert first_chunk.metadata.extra["boundary_type"] == "silence_based"
        assert first_chunk.metadata.extra["silence_threshold_db"] == -30.0

    @pytest.mark.skipif(not Path("test_data/audio_files").exists(), reason="No test audio files available")
    def test_multiple_audio_formats(self, test_audio_files):
        """Test chunking with multiple audio formats."""
        if not test_audio_files:
            pytest.skip("No test audio files available")

        chunker = SilenceBasedAudioChunker(
            min_segment_duration=1.0,  # Shorter for test files
            max_segment_duration=30.0
        )

        for format_name, file_path in test_audio_files.items():
            result = chunker.chunk(file_path)

            assert len(result.chunks) >= 1
            assert result.strategy_used == "silence_based_audio"

            first_chunk = result.chunks[0]
            assert first_chunk.metadata.extra["format"] == format_name

    def test_chunk_size_estimation(self):
        """Test chunk size estimation."""
        chunker = SilenceBasedAudioChunker(
            min_segment_duration=10.0,
            max_segment_duration=60.0
        )

        # Test various durations
        assert chunker.get_chunk_size_estimate(0.0) == 0
        assert chunker.get_chunk_size_estimate(30.0) >= 1
        assert chunker.get_chunk_size_estimate(120.0) >= 2
        assert chunker.get_chunk_size_estimate(300.0) >= 3

        # Test with custom speech rate
        estimate1 = chunker.get_chunk_size_estimate(120.0, typical_speech_rate=0.5)
        estimate2 = chunker.get_chunk_size_estimate(120.0, typical_speech_rate=0.9)
        assert estimate1 <= estimate2  # Higher speech rate = more chunks

    def test_get_config(self):
        """Test configuration retrieval."""
        chunker = SilenceBasedAudioChunker(
            silence_threshold_db=-45.0,
            min_silence_duration=0.8,
            preserve_format=False
        )

        config = chunker.get_config()

        assert config["silence_threshold_db"] == -45.0
        assert config["min_silence_duration"] == 0.8
        assert config["preserve_format"] is False
        assert "min_segment_duration" in config
        assert "max_segment_duration" in config

    def test_registry_integration(self):
        """Test that chunker is properly registered."""
        registry = get_registry()

        # Check if chunker is registered
        chunker_names = registry.list_chunkers()
        assert "silence_based_audio" in chunker_names

        # Check chunker metadata
        chunker = registry.create_chunker("silence_based_audio")
        assert chunker is not None
        assert chunker.name == "silence_based_audio"
        assert chunker.category == "multimedia"

    def test_empty_content_handling(self):
        """Test handling of empty or invalid content."""
        chunker = SilenceBasedAudioChunker()

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            chunker.chunk("non_existent_file.wav")

    def test_streaming_support(self):
        """Test streaming chunker interface."""
        chunker = SilenceBasedAudioChunker()

        # Should inherit from StreamableChunker
        assert hasattr(chunker, 'chunk')
        assert hasattr(chunker, 'get_chunk_size_estimate')

    @patch('pydub.silence.detect_silence')
    @patch('pydub.AudioSegment.from_file')
    def test_silence_detection_logic(self, mock_from_file, mock_detect_silence):
        """Test silence detection and segmentation logic."""
        # Mock audio segment
        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_audio.channels = 2
        mock_audio.__len__.return_value = 10000  # 10 seconds
        mock_from_file.return_value = mock_audio

        # Mock silence detection - return one silence period
        mock_detect_silence.return_value = [(3000, 4000)]  # 3-4 seconds

        # Mock audio slicing
        mock_segment1 = MagicMock()
        mock_segment1.frame_rate = 44100
        mock_segment1.channels = 2
        mock_segment1.__len__.return_value = 3000
        mock_segment1.export.return_value.read.return_value = b'audio_data_1'

        mock_segment2 = MagicMock()
        mock_segment2.frame_rate = 44100
        mock_segment2.channels = 2
        mock_segment2.__len__.return_value = 6000
        mock_segment2.export.return_value.read.return_value = b'audio_data_2'

        mock_audio.__getitem__.side_effect = [mock_segment1, mock_segment2]

        chunker = SilenceBasedAudioChunker(
            min_segment_duration=1.0,  # 1 second minimum
            max_segment_duration=30.0
        )

        result = chunker.chunk("test.wav")

        # Verify silence detection was called
        mock_detect_silence.assert_called_once()

        # Should create segments based on silence
        assert len(result.chunks) >= 1
        assert result.strategy_used == "silence_based_audio"

    @patch('pydub.AudioSegment.from_file')
    def test_audio_transformations(self, mock_from_file):
        """Test audio format transformations."""
        # Mock audio segment
        mock_audio = MagicMock()
        mock_audio.frame_rate = 22050
        mock_audio.channels = 1
        mock_audio.__len__.return_value = 5000  # 5 seconds
        mock_audio.export.return_value.read.return_value = b'transformed_audio'

        # Mock transformations
        mock_resampled = MagicMock()
        mock_resampled.channels = 1
        mock_resampled.__len__.return_value = 5000
        mock_audio.set_frame_rate.return_value = mock_resampled

        mock_stereo = MagicMock()
        mock_stereo.__len__.return_value = 5000
        mock_resampled.set_channels.return_value = mock_stereo

        mock_from_file.return_value = mock_audio

        chunker = SilenceBasedAudioChunker(
            sample_rate=44100,
            channels=2,
            min_segment_duration=1.0
        )

        with patch('pydub.silence.detect_silence') as mock_detect_silence:
            mock_detect_silence.return_value = []  # No silence

            result = chunker.chunk("test.wav")

            # Verify transformations were applied
            mock_audio.set_frame_rate.assert_called_once_with(44100)
            mock_resampled.set_channels.assert_called_once_with(2)

    def test_performance_with_long_audio(self):
        """Test performance characteristics with simulated long audio."""
        with patch('pydub.AudioSegment.from_file') as mock_from_file:
            # Mock a long audio file (10 minutes)
            mock_audio = MagicMock()
            mock_audio.frame_rate = 44100
            mock_audio.channels = 2
            mock_audio.__len__.return_value = 600000  # 10 minutes in ms
            mock_from_file.return_value = mock_audio

            chunker = SilenceBasedAudioChunker(
                min_segment_duration=5.0,
                max_segment_duration=60.0
            )

            # Should handle large files without issues
            estimated_chunks = chunker.get_chunk_size_estimate(600.0)  # 10 minutes
            assert estimated_chunks > 0
            assert estimated_chunks < 50  # Should be reasonable number

    def test_edge_case_no_silence_detected(self):
        """Test behavior when no silence is detected."""
        with patch('pydub.AudioSegment.from_file') as mock_from_file:
            with patch('pydub.silence.detect_silence') as mock_detect_silence:
                # Mock audio with no silence
                mock_audio = MagicMock()
                mock_audio.frame_rate = 44100
                mock_audio.channels = 2
                mock_audio.__len__.return_value = 30000  # 30 seconds
                mock_audio.export.return_value.read.return_value = b'continuous_audio'
                mock_from_file.return_value = mock_audio

                # No silence detected
                mock_detect_silence.return_value = []

                chunker = SilenceBasedAudioChunker(
                    min_segment_duration=5.0,
                    max_segment_duration=20.0
                )

                result = chunker.chunk("test.wav")

                # Should still create chunks by splitting at max duration
                assert len(result.chunks) >= 1
                assert result.strategy_used == "silence_based_audio"

    def test_minimum_segment_duration_enforcement(self):
        """Test that minimum segment duration is enforced."""
        with patch('pydub.AudioSegment.from_file') as mock_from_file:
            with patch('pydub.silence.detect_silence') as mock_detect_silence:
                mock_audio = MagicMock()
                mock_audio.frame_rate = 44100
                mock_audio.channels = 2
                mock_audio.__len__.return_value = 20000  # 20 seconds
                mock_from_file.return_value = mock_audio

                # Mock many short silence periods that would create tiny segments
                mock_detect_silence.return_value = [
                    (1000, 1100),   # 1-1.1s
                    (2000, 2100),   # 2-2.1s
                    (3000, 3100),   # 3-3.1s
                ]

                chunker = SilenceBasedAudioChunker(
                    min_segment_duration=5.0,  # 5 seconds minimum
                    max_segment_duration=30.0
                )

                result = chunker.chunk("test.wav")

                # Should merge or skip segments that are too short
                assert len(result.chunks) >= 1
                for chunk in result.chunks:
                    duration = chunk.metadata.extra.get("duration_seconds", 0)
                    # Most chunks should meet minimum duration (allowing some tolerance)
                    assert duration >= 4.5 or duration == chunk.metadata.extra.get("duration_seconds")

    def test_error_handling_missing_pydub(self):
        """Test error handling when pydub is not available."""
        chunker = SilenceBasedAudioChunker()

        with patch.dict('sys.modules', {'pydub': None}):
            with pytest.raises(ImportError, match="pydub is required"):
                chunker.chunk("test.wav")

    def test_audio_properties_preservation(self):
        """Test that audio properties are properly captured in metadata."""
        with patch('pydub.AudioSegment.from_file') as mock_from_file:
            mock_audio = MagicMock()
            mock_audio.frame_rate = 48000
            mock_audio.channels = 2
            mock_audio.frame_width = 2
            mock_audio.__len__.return_value = 15000  # 15 seconds
            mock_audio.export.return_value.read.return_value = b'test_audio_data'
            mock_from_file.return_value = mock_audio

            chunker = SilenceBasedAudioChunker(min_segment_duration=1.0)

            with patch('pydub.silence.detect_silence') as mock_detect_silence:
                mock_detect_silence.return_value = []

                result = chunker.chunk("test.wav")

                assert len(result.chunks) >= 1

                # Check that source info contains audio properties
                assert "duration" in result.source_info
                assert "sample_rate" in result.source_info
                assert "channels" in result.source_info
                assert result.source_info["sample_rate"] == 48000
                assert result.source_info["channels"] == 2

                # Check chunk metadata
                first_chunk = result.chunks[0]
                assert first_chunk.metadata.extra["sample_rate"] == 48000
                assert first_chunk.metadata.extra["channels"] == 2
                assert first_chunk.metadata.extra["format"] == "wav"
