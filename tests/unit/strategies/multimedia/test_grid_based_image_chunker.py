"""
Comprehensive unit tests for GridBasedImageChunker.

This test suite verifies all functionality of the grid-based image chunking strategy,
including initialization, parameter validation, chunking logic, registry integration,
and error handling.
"""

import pytest
import io
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from chunking_strategy.core.base import ModalityType, ChunkingResult
from chunking_strategy.strategies.multimedia.grid_based_image import GridBasedImageChunker
from chunking_strategy.core.registry import list_chunkers, create_chunker


class TestGridBasedImageChunker:
    """Test suite for GridBasedImageChunker."""

    def test_chunker_initialization(self):
        """Test basic chunker initialization with default parameters."""
        chunker = GridBasedImageChunker()

        assert chunker.tile_width == 256
        assert chunker.tile_height == 256
        assert chunker.overlap_pixels == 0
        assert not chunker.preserve_aspect_ratio
        assert chunker.pad_incomplete_tiles
        assert chunker.output_format == "PNG"

    def test_chunker_initialization_with_parameters(self):
        """Test chunker initialization with custom parameters."""
        chunker = GridBasedImageChunker(
            tile_width=128,
            tile_height=64,
            overlap_pixels=16,
            preserve_aspect_ratio=True,
            pad_incomplete_tiles=False,
            output_format="JPEG"
        )

        assert chunker.tile_width == 128
        assert chunker.tile_height == 64
        assert chunker.overlap_pixels == 16
        assert chunker.preserve_aspect_ratio
        assert not chunker.pad_incomplete_tiles
        assert chunker.output_format == "JPEG"

    def test_parameter_validation(self):
        """Test parameter validation and constraints."""
        # Test minimum values
        chunker = GridBasedImageChunker(tile_width=0, tile_height=-1, overlap_pixels=-5)

        assert chunker.tile_width == 1  # Should be clamped to minimum
        assert chunker.tile_height == 1  # Should be clamped to minimum
        assert chunker.overlap_pixels == 0  # Should be clamped to minimum

    def test_invalid_output_format(self):
        """Test handling of invalid output formats."""
        chunker = GridBasedImageChunker(output_format="INVALID")

        # Should default to PNG for invalid format
        assert chunker.output_format == "PNG"

    @patch('chunking_strategy.strategies.multimedia.grid_based_image.Image')
    def test_basic_image_chunking(self, mock_image):
        """Test basic image chunking functionality."""
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.size = (400, 300)
        mock_img.width = 400
        mock_img.height = 300
        mock_img.mode = "RGB"
        mock_img.format = "PNG"

        # Mock crop method
        mock_tile = MagicMock()
        mock_tile.width = 200
        mock_tile.height = 150
        mock_img.crop.return_value = mock_tile

        # Mock save method
        mock_tile.save = MagicMock()

        mock_image.open.return_value = mock_img

        chunker = GridBasedImageChunker(tile_width=200, tile_height=150)
        result = chunker.chunk("test_image.png")

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 4  # 2x2 grid for 400x300 image with 200x150 tiles
        assert result.processing_time > 0

        # Check source info
        assert "total_tiles" in result.source_info
        assert result.source_info["total_tiles"] == 4

    @patch('chunking_strategy.strategies.multimedia.grid_based_image.Image')
    def test_chunking_with_padding(self, mock_image):
        """Test image chunking with incomplete tile padding."""
        # Mock PIL Image - size that requires padding
        mock_img = MagicMock()
        mock_img.size = (250, 250)
        mock_img.width = 250
        mock_img.height = 250
        mock_img.mode = "RGB"
        mock_img.format = "PNG"

        # Mock for incomplete tiles
        mock_incomplete_tile = MagicMock()
        mock_incomplete_tile.width = 50  # Partial tile
        mock_incomplete_tile.height = 50

        mock_padded_tile = MagicMock()
        mock_padded_tile.width = 100  # Full tile size after padding
        mock_padded_tile.height = 100
        mock_padded_tile.save = MagicMock()

        mock_img.crop.return_value = mock_incomplete_tile
        mock_image.new.return_value = mock_padded_tile
        mock_image.open.return_value = mock_img

        chunker = GridBasedImageChunker(tile_width=100, tile_height=100, pad_incomplete_tiles=True)
        result = chunker.chunk("test_image.png")

        assert len(result.chunks) == 9  # 3x3 grid
        # Verify padding was called for edge tiles
        mock_image.new.assert_called()

    @patch('chunking_strategy.strategies.multimedia.grid_based_image.Image')
    def test_chunking_with_overlap(self, mock_image):
        """Test image chunking with tile overlap."""
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.size = (300, 300)
        mock_img.width = 300
        mock_img.height = 300
        mock_img.mode = "RGB"
        mock_img.format = "PNG"

        mock_tile = MagicMock()
        mock_tile.width = 100
        mock_tile.height = 100
        mock_tile.save = MagicMock()
        mock_img.crop.return_value = mock_tile

        mock_image.open.return_value = mock_img

        # With 50 pixel overlap, step size is 50, so we get more tiles
        chunker = GridBasedImageChunker(tile_width=100, tile_height=100, overlap_pixels=50)
        result = chunker.chunk("test_image.png")

        # Should get more tiles due to overlap
        assert len(result.chunks) > 9  # More than 3x3 due to overlap

    def test_chunk_size_estimation(self):
        """Test chunk size estimation functionality."""
        chunker = GridBasedImageChunker(tile_width=128, tile_height=128)

        # Test various content lengths
        assert chunker.get_chunk_size_estimate(0) == 0
        assert chunker.get_chunk_size_estimate(1000) > 0
        assert chunker.get_chunk_size_estimate(10000) > chunker.get_chunk_size_estimate(1000)

    def test_get_config(self):
        """Test configuration retrieval."""
        chunker = GridBasedImageChunker(
            tile_width=512,
            tile_height=256,
            overlap_pixels=32,
            pad_incomplete_tiles=False,
            output_format="JPEG"
        )

        config = chunker.get_config()

        assert config["tile_width"] == 512
        assert config["tile_height"] == 256
        assert config["overlap_pixels"] == 32
        assert not config["pad_incomplete_tiles"]
        assert config["output_format"] == "JPEG"

    def test_registry_integration(self):
        """Test chunker registration with the registry system."""
        chunkers = list_chunkers()
        assert "grid_based_image" in chunkers

        # Test creation through registry
        chunker = create_chunker("grid_based_image", tile_width=64, tile_height=64)
        assert isinstance(chunker, GridBasedImageChunker)
        assert chunker.tile_width == 64
        assert chunker.tile_height == 64

    @patch('chunking_strategy.strategies.multimedia.grid_based_image.Image')
    def test_empty_content_handling(self, mock_image):
        """Test handling of empty or invalid content."""
        mock_image.open.side_effect = Exception("Invalid image")

        chunker = GridBasedImageChunker()
        result = chunker.chunk("invalid_image.png")

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 0
        assert "error" in result.source_info

    def test_unsupported_content_type(self):
        """Test handling of unsupported content types."""
        chunker = GridBasedImageChunker()

        # Test with invalid content type
        result = chunker.chunk(123)  # Invalid type

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 0
        assert "error" in result.source_info

    @patch('chunking_strategy.strategies.multimedia.grid_based_image.Image')
    def test_bytes_input(self, mock_image):
        """Test chunking with bytes input."""
        mock_img = MagicMock()
        mock_img.size = (200, 200)
        mock_img.width = 200
        mock_img.height = 200
        mock_img.mode = "RGB"
        mock_img.format = "PNG"

        mock_tile = MagicMock()
        mock_tile.width = 100
        mock_tile.height = 100
        mock_tile.save = MagicMock()
        mock_img.crop.return_value = mock_tile

        mock_image.open.return_value = mock_img

        chunker = GridBasedImageChunker(tile_width=100, tile_height=100)

        # Test with bytes input
        fake_image_bytes = b"fake_image_data"
        result = chunker.chunk(fake_image_bytes)

        # Should call Image.open with BytesIO
        mock_image.open.assert_called_once()
        args = mock_image.open.call_args[0]
        assert hasattr(args[0], 'read')  # Should be a BytesIO object

    @patch('chunking_strategy.strategies.multimedia.grid_based_image.Image')
    def test_streaming_functionality(self, mock_image):
        """Test streaming chunk functionality."""
        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_img.width = 100
        mock_img.height = 100
        mock_img.mode = "RGB"
        mock_img.format = "PNG"

        mock_tile = MagicMock()
        mock_tile.width = 100
        mock_tile.height = 100
        mock_tile.save = MagicMock()
        mock_img.crop.return_value = mock_tile

        mock_image.open.return_value = mock_img

        chunker = GridBasedImageChunker()

        # Simulate streaming content
        stream_data = [b"chunk1", b"chunk2", b"chunk3"]
        result = chunker.chunk_stream(stream_data)

        assert isinstance(result, ChunkingResult)
        # Should accumulate stream and process as single image

    def test_performance_metrics(self):
        """Test that performance metrics are captured."""
        chunker = GridBasedImageChunker()

        with patch('chunking_strategy.strategies.multimedia.grid_based_image.Image') as mock_image:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_img.width = 100
            mock_img.height = 100
            mock_img.mode = "RGB"
            mock_img.format = "PNG"

            mock_tile = MagicMock()
            mock_tile.width = 100
            mock_tile.height = 100
            mock_tile.save = MagicMock()
            mock_img.crop.return_value = mock_tile

            mock_image.open.return_value = mock_img

            start_time = time.time()
            result = chunker.chunk("test_image.png")
            end_time = time.time()

            assert result.processing_time > 0
            assert result.processing_time <= (end_time - start_time) + 0.1  # Small tolerance

    @patch('chunking_strategy.strategies.multimedia.grid_based_image.Image')
    def test_different_output_formats(self, mock_image):
        """Test different output formats (PNG, JPEG, etc.)."""
        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_img.width = 100
        mock_img.height = 100
        mock_img.mode = "RGB"
        mock_img.format = "PNG"

        mock_tile = MagicMock()
        mock_tile.width = 100
        mock_tile.height = 100
        mock_img.crop.return_value = mock_tile

        mock_image.open.return_value = mock_img

        # Test JPEG format
        chunker = GridBasedImageChunker(output_format="JPEG")

        # Mock the _tile_to_bytes method to return some bytes
        chunker._tile_to_bytes = MagicMock(return_value=b"mock_jpeg_bytes")

        result = chunker.chunk("test_image.png")

        # Verify results
        assert len(result.chunks) > 0
        assert chunker.output_format == "JPEG"

        # Verify that the output format is stored in tile config
        tile_config = result.source_info["tile_config"]
        assert tile_config["output_format"] == "JPEG"

        # Verify _tile_to_bytes was called
        assert chunker._tile_to_bytes.called

    @patch('chunking_strategy.strategies.multimedia.grid_based_image.Image')
    def test_rgba_to_rgb_conversion_for_jpeg(self, mock_image):
        """Test RGBA to RGB conversion when using JPEG format."""
        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_img.width = 100
        mock_img.height = 100
        mock_img.mode = "RGB"
        mock_img.format = "PNG"

        mock_tile = MagicMock()
        mock_tile.width = 100
        mock_tile.height = 100
        mock_tile.mode = "RGBA"  # Has alpha channel
        mock_tile.split.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_tile.save = MagicMock()

        # Mock background creation for RGBA->RGB conversion
        mock_background = MagicMock()
        mock_image.new.return_value = mock_background

        mock_img.crop.return_value = mock_tile
        mock_image.open.return_value = mock_img

        chunker = GridBasedImageChunker(output_format="JPEG")
        result = chunker.chunk("test_image.png")

        # Should handle RGBA->RGB conversion for JPEG
        assert len(result.chunks) > 0

    def test_error_handling_missing_pil(self):
        """Test error handling when PIL is not available."""
        with patch('chunking_strategy.strategies.multimedia.grid_based_image.Image', None):
            chunker = GridBasedImageChunker()

            with pytest.raises(ImportError) as excinfo:
                chunker.chunk("test_image.png")

            assert "PIL (Pillow) is required" in str(excinfo.value)

    @patch('chunking_strategy.strategies.multimedia.grid_based_image.Image')
    def test_grid_position_calculation(self, mock_image):
        """Test correct grid position calculation for tiles."""
        mock_img = MagicMock()
        mock_img.size = (300, 200)
        mock_img.width = 300
        mock_img.height = 200
        mock_img.mode = "RGB"
        mock_img.format = "PNG"

        mock_tile = MagicMock()
        mock_tile.width = 100
        mock_tile.height = 100
        mock_tile.save = MagicMock()
        mock_img.crop.return_value = mock_tile

        mock_image.open.return_value = mock_img

        chunker = GridBasedImageChunker(tile_width=100, tile_height=100)
        result = chunker.chunk("test_image.png")

        # Should have 3x2 = 6 tiles
        assert len(result.chunks) == 6

        # Check grid positions
        grid_positions = [chunk.metadata.extra["grid_position"] for chunk in result.chunks]
        expected_positions = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        assert grid_positions == expected_positions

    @patch('chunking_strategy.strategies.multimedia.grid_based_image.Image')
    def test_pixel_position_calculation(self, mock_image):
        """Test correct pixel position calculation for tiles."""
        mock_img = MagicMock()
        mock_img.size = (200, 200)
        mock_img.width = 200
        mock_img.height = 200
        mock_img.mode = "RGB"
        mock_img.format = "PNG"

        mock_tile = MagicMock()
        mock_tile.width = 100
        mock_tile.height = 100
        mock_tile.save = MagicMock()
        mock_img.crop.return_value = mock_tile

        mock_image.open.return_value = mock_img

        chunker = GridBasedImageChunker(tile_width=100, tile_height=100)
        result = chunker.chunk("test_image.png")

        # Check pixel positions and bounding boxes
        for chunk in result.chunks:
            pixel_pos = chunk.metadata.extra["pixel_position"]
            bbox = chunk.metadata.bbox

            # BBox should match pixel position + tile size
            assert bbox[0] == pixel_pos[0]  # x1
            assert bbox[1] == pixel_pos[1]  # y1
            assert bbox[2] == pixel_pos[0] + 100  # x2
            assert bbox[3] == pixel_pos[1] + 100  # y2

    @patch('chunking_strategy.strategies.multimedia.grid_based_image.Image')
    def test_edge_tile_detection(self, mock_image):
        """Test detection of edge tiles."""
        mock_img = MagicMock()
        mock_img.size = (250, 150)  # Size that creates edge tiles
        mock_img.width = 250
        mock_img.height = 150
        mock_img.mode = "RGB"
        mock_img.format = "PNG"

        mock_tile = MagicMock()
        mock_tile.width = 100
        mock_tile.height = 100
        mock_tile.save = MagicMock()
        mock_img.crop.return_value = mock_tile

        mock_image.open.return_value = mock_img

        chunker = GridBasedImageChunker(tile_width=100, tile_height=100)
        result = chunker.chunk("test_image.png")

        # Check edge tile detection
        edge_tiles = [chunk for chunk in result.chunks
                     if chunk.metadata.extra["is_edge_tile"]]
        assert len(edge_tiles) > 0  # Should have some edge tiles

    def test_chunk_id_format(self):
        """Test chunk ID formatting."""
        with patch('chunking_strategy.strategies.multimedia.grid_based_image.Image') as mock_image:
            mock_img = MagicMock()
            mock_img.size = (200, 100)
            mock_img.width = 200
            mock_img.height = 100
            mock_img.mode = "RGB"
            mock_img.format = "PNG"

            mock_tile = MagicMock()
            mock_tile.width = 100
            mock_tile.height = 100
            mock_tile.save = MagicMock()
            mock_img.crop.return_value = mock_tile

            mock_image.open.return_value = mock_img

            chunker = GridBasedImageChunker(tile_width=100, tile_height=100)
            result = chunker.chunk("test_image.png")

            # Check ID format: tile_{index}_{grid_x}x{grid_y}
            expected_ids = ["tile_0_0x0", "tile_1_1x0"]
            actual_ids = [chunk.id for chunk in result.chunks]
            assert actual_ids == expected_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
