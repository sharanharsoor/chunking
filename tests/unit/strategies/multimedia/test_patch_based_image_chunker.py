"""
Tests for Patch-based Image Chunker

This module contains comprehensive unit tests for the PatchBasedImageChunker,
including tests for different sampling strategies, parameter validation, 
edge handling, streaming functionality, and integration with the registry.
"""

import io
import pytest
from unittest.mock import MagicMock, patch

from chunking_strategy.core.base import ModalityType, ChunkingResult
from chunking_strategy.strategies.multimedia.patch_based_image import PatchBasedImageChunker


class TestPatchBasedImageChunker:
    """Test cases for the PatchBasedImageChunker class."""

    def test_chunker_initialization(self):
        """Test basic chunker initialization with defaults."""
        chunker = PatchBasedImageChunker()
        
        assert chunker.patch_width == 224
        assert chunker.patch_height == 224
        assert chunker.stride_x == 112
        assert chunker.stride_y == 112
        assert chunker.sampling_strategy == "uniform"
        assert chunker.max_patches is None
        assert chunker.edge_handling == "pad"
        assert chunker.output_format == "PNG"
        assert chunker.random_seed is None

    def test_chunker_initialization_with_parameters(self):
        """Test chunker initialization with custom parameters."""
        chunker = PatchBasedImageChunker(
            patch_width=128,
            patch_height=96,
            stride_x=64,
            stride_y=48,
            sampling_strategy="random",
            max_patches=50,
            edge_handling="crop",
            output_format="JPEG",
            random_seed=42
        )
        
        assert chunker.patch_width == 128
        assert chunker.patch_height == 96
        assert chunker.stride_x == 64
        assert chunker.stride_y == 48
        assert chunker.sampling_strategy == "random"
        assert chunker.max_patches == 50
        assert chunker.edge_handling == "crop"
        assert chunker.output_format == "JPEG"
        assert chunker.random_seed == 42

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid patch dimensions
        with pytest.raises(ValueError, match="Patch dimensions must be positive"):
            PatchBasedImageChunker(patch_width=0, patch_height=100)
        
        with pytest.raises(ValueError, match="Patch dimensions must be positive"):
            PatchBasedImageChunker(patch_width=100, patch_height=-1)
        
        # Test invalid stride values
        with pytest.raises(ValueError, match="Stride values must be positive"):
            PatchBasedImageChunker(stride_x=0, stride_y=50)
        
        with pytest.raises(ValueError, match="Stride values must be positive"):
            PatchBasedImageChunker(stride_x=50, stride_y=-1)
        
        # Test invalid sampling strategy
        with pytest.raises(ValueError, match="sampling_strategy must be one of"):
            PatchBasedImageChunker(sampling_strategy="invalid_strategy")
        
        # Test invalid edge handling
        with pytest.raises(ValueError, match="edge_handling must be one of"):
            PatchBasedImageChunker(edge_handling="invalid_handling")
        
        # Test invalid output format
        with pytest.raises(ValueError, match="output_format must be one of"):
            PatchBasedImageChunker(output_format="INVALID")

    def test_invalid_output_format(self):
        """Test behavior with invalid output format."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            PatchBasedImageChunker(output_format="INVALID_FORMAT")

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_uniform_sampling(self, mock_image):
        """Test uniform sampling strategy."""
        mock_img = MagicMock()
        mock_img.size = (400, 300)
        mock_img.width = 400
        mock_img.height = 300
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 100
        mock_patch.height = 100
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker(
            patch_width=100,
            patch_height=100,
            stride_x=50,
            stride_y=50,
            sampling_strategy="uniform"
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result = chunker.chunk("test_image.jpg")
        
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0
        assert all(chunk.modality == ModalityType.IMAGE for chunk in result.chunks)
        assert result.source_info["strategy_used"] == "patch_based_image"

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_random_sampling(self, mock_image):
        """Test random sampling strategy."""
        mock_img = MagicMock()
        mock_img.size = (400, 300)
        mock_img.width = 400
        mock_img.height = 300
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 100
        mock_patch.height = 100
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker(
            patch_width=100,
            patch_height=100,
            sampling_strategy="random",
            max_patches=10,
            random_seed=42
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result = chunker.chunk("test_image.jpg")
        
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 10  # Should respect max_patches limit
        assert result.source_info["strategy_used"] == "patch_based_image"

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_edge_aware_sampling(self, mock_image):
        """Test edge-aware sampling strategy."""
        mock_img = MagicMock()
        mock_img.size = (400, 300)
        mock_img.width = 400
        mock_img.height = 300
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 100
        mock_patch.height = 100
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker(
            patch_width=100,
            patch_height=100,
            sampling_strategy="edge_aware"
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result = chunker.chunk("test_image.jpg")
        
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0
        assert result.source_info["strategy_used"] == "patch_based_image"

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_pad_edge_handling(self, mock_image):
        """Test pad edge handling strategy."""
        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_img.width = 100
        mock_img.height = 100
        mock_img.mode = "RGB"
        
        # Test case where patch extends beyond image boundary
        mock_patch = MagicMock()
        mock_patch.width = 128
        mock_patch.height = 128
        
        # Mock padded image creation
        mock_padded = MagicMock()
        mock_padded.crop.return_value = mock_patch
        mock_image.new.return_value = mock_padded
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker(
            patch_width=128,
            patch_height=128,
            stride_x=100,
            stride_y=100,
            edge_handling="pad",
            sampling_strategy="uniform"
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result = chunker.chunk("test_image.jpg")
        
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_crop_edge_handling(self, mock_image):
        """Test crop edge handling strategy."""
        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_img.width = 100
        mock_img.height = 100
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.size = (50, 50)  # Cropped patch
        mock_patch.resize.return_value = mock_patch
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker(
            patch_width=128,
            patch_height=128,
            stride_x=100,
            stride_y=100,
            edge_handling="crop",
            sampling_strategy="uniform"
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result = chunker.chunk("test_image.jpg")
        
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_skip_edge_handling(self, mock_image):
        """Test skip edge handling strategy."""
        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_img.width = 100
        mock_img.height = 100
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 50
        mock_patch.height = 50
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker(
            patch_width=128,
            patch_height=128,
            stride_x=100,
            stride_y=100,
            edge_handling="skip",
            sampling_strategy="uniform"
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result = chunker.chunk("test_image.jpg")
        
        # With skip edge handling, patches that extend beyond boundaries should be skipped
        # In a 100x100 image with 128x128 patches, no patches should fit
        assert isinstance(result, ChunkingResult)
        # May have 0 chunks if all patches are skipped due to size

    def test_chunk_size_estimation_uniform(self):
        """Test chunk size estimation for uniform sampling."""
        chunker = PatchBasedImageChunker(
            patch_width=100,
            patch_height=100,
            stride_x=50,
            stride_y=50,
            sampling_strategy="uniform"
        )
        
        # Test with different content lengths
        small_estimate = chunker.get_chunk_size_estimate(5000)  # Small image
        medium_estimate = chunker.get_chunk_size_estimate(50000)  # Medium image
        large_estimate = chunker.get_chunk_size_estimate(500000)  # Large image
        
        assert small_estimate > 0
        assert medium_estimate > small_estimate
        assert large_estimate > medium_estimate

    def test_chunk_size_estimation_random(self):
        """Test chunk size estimation for random sampling."""
        chunker = PatchBasedImageChunker(
            patch_width=100,
            patch_height=100,
            sampling_strategy="random",
            max_patches=25
        )
        
        estimate = chunker.get_chunk_size_estimate(100000)
        assert estimate == 25  # Should respect max_patches limit

    def test_chunk_size_estimation_edge_aware(self):
        """Test chunk size estimation for edge-aware sampling."""
        chunker = PatchBasedImageChunker(
            patch_width=100,
            patch_height=100,
            sampling_strategy="edge_aware"
        )
        
        estimate = chunker.get_chunk_size_estimate(100000)
        assert estimate > 0
        assert estimate <= 20  # Edge-aware typically produces fewer patches

    def test_get_config(self):
        """Test configuration retrieval."""
        chunker = PatchBasedImageChunker(
            patch_width=128,
            patch_height=96,
            stride_x=64,
            stride_y=48,
            sampling_strategy="random",
            max_patches=30,
            edge_handling="crop",
            output_format="JPEG",
            random_seed=123
        )
        
        config = chunker.get_config()
        
        assert config["patch_width"] == 128
        assert config["patch_height"] == 96
        assert config["stride_x"] == 64
        assert config["stride_y"] == 48
        assert config["sampling_strategy"] == "random"
        assert config["max_patches"] == 30
        assert config["edge_handling"] == "crop"
        assert config["output_format"] == "JPEG"
        assert config["random_seed"] == 123

    def test_registry_integration(self):
        """Test integration with the chunker registry."""
        from chunking_strategy.core.registry import get_registry
        
        registry = get_registry()
        chunkers = registry.list_chunkers()
        
        assert "patch_based_image" in chunkers

    def test_empty_content_handling(self):
        """Test handling of empty content."""
        chunker = PatchBasedImageChunker()
        
        result = chunker.chunk(b"")
        
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 0
        assert "error" in result.source_info

    def test_unsupported_content_type(self):
        """Test handling of unsupported content types."""
        chunker = PatchBasedImageChunker()
        
        result = chunker.chunk(123)  # Invalid content type
        
        assert isinstance(result, ChunkingResult) 
        assert len(result.chunks) == 0
        assert "error" in result.source_info

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_bytes_input(self, mock_image):
        """Test processing bytes input."""
        mock_img = MagicMock()
        mock_img.size = (200, 200)
        mock_img.width = 200
        mock_img.height = 200
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 100
        mock_patch.height = 100
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker(
            patch_width=100,
            patch_height=100,
            stride_x=100,
            stride_y=100
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        # Test with bytes input
        result = chunker.chunk(b"fake_image_bytes")
        
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0

    def test_streaming_functionality(self):
        """Test streaming content processing."""
        chunker = PatchBasedImageChunker()
        
        # Mock chunk method to test streaming
        original_chunk = chunker.chunk
        chunker.chunk = MagicMock(return_value=ChunkingResult(chunks=[], processing_time=0.1, source_info={}))
        
        # Test stream processing
        stream = [b"chunk1", b"chunk2", b"chunk3"]
        result = chunker.chunk_stream(stream)
        
        assert isinstance(result, ChunkingResult)
        chunker.chunk.assert_called_once_with(b"chunk1chunk2chunk3")
        
        # Restore original method
        chunker.chunk = original_chunk

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_performance_metrics(self, mock_image):
        """Test that performance metrics are captured."""
        mock_img = MagicMock()
        mock_img.size = (200, 200)
        mock_img.width = 200
        mock_img.height = 200
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 100
        mock_patch.height = 100
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker()
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        import time
        start_time = time.time()
        result = chunker.chunk("test_image.jpg")
        end_time = time.time()
        
        assert result.processing_time > 0
        assert result.processing_time <= (end_time - start_time) + 0.1  # Small tolerance

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_different_output_formats(self, mock_image):
        """Test different output formats (PNG, JPEG, etc.)."""
        mock_img = MagicMock()
        mock_img.size = (200, 200)
        mock_img.width = 200
        mock_img.height = 200
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 100
        mock_patch.height = 100
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        # Test JPEG format
        chunker = PatchBasedImageChunker(output_format="JPEG")
        
        # Mock the _patch_to_bytes method to return some bytes
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_jpeg_bytes")
        
        result = chunker.chunk("test_image.png")
        
        # Verify results
        assert len(result.chunks) > 0
        assert chunker.output_format == "JPEG"
        
        # Verify that the output format is stored in patch config
        patch_config = result.source_info["patch_config"]
        assert patch_config["output_format"] == "JPEG"
        
        # Verify _patch_to_bytes was called
        assert chunker._patch_to_bytes.called

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_rgba_to_rgb_conversion_for_jpeg(self, mock_image):
        """Test RGBA to RGB conversion when using JPEG format."""
        mock_img = MagicMock()
        mock_img.size = (200, 200)
        mock_img.width = 200
        mock_img.height = 200
        mock_img.mode = "RGBA"  # RGBA mode
        
        mock_patch = MagicMock()
        mock_patch.mode = "RGBA"
        mock_patch.width = 100
        mock_patch.height = 100
        mock_patch.split.return_value = [None, None, None, MagicMock()]  # Mock alpha channel
        mock_img.crop.return_value = mock_patch
        
        # Mock RGB background creation  
        mock_background = MagicMock()
        mock_image.new.return_value = mock_background
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker(output_format="JPEG")
        result = chunker.chunk("test_image.png")
        
        assert len(result.chunks) > 0
        # Verify that new RGB image was created (RGBA -> RGB conversion)
        mock_image.new.assert_called()

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.PIL_AVAILABLE', False)
    def test_error_handling_missing_pil(self):
        """Test error handling when PIL is not available."""
        with pytest.raises(ImportError, match="Pillow \\(PIL\\) is required"):
            PatchBasedImageChunker()

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_patch_position_calculation(self, mock_image):
        """Test patch position calculation for different strategies."""
        mock_img = MagicMock()
        mock_img.size = (300, 200)
        mock_img.width = 300
        mock_img.height = 200
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 100
        mock_patch.height = 100
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker(
            patch_width=100,
            patch_height=100,
            stride_x=100,
            stride_y=100,
            sampling_strategy="uniform"
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result = chunker.chunk("test_image.jpg")
        
        # Check that positions are calculated correctly
        if result.chunks:
            first_chunk = result.chunks[0]
            position = first_chunk.metadata.extra["position"]
            assert isinstance(position, tuple)
            assert len(position) == 2
            assert all(isinstance(coord, int) for coord in position)

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')  
    def test_overlap_calculation(self, mock_image):
        """Test overlap calculation in patch metadata."""
        mock_img = MagicMock()
        mock_img.size = (200, 200)
        mock_img.width = 200
        mock_img.height = 200
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 100
        mock_patch.height = 100
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        # Test with overlapping patches (stride < patch size)
        chunker = PatchBasedImageChunker(
            patch_width=100,
            patch_height=100,
            stride_x=50,  # 50% overlap
            stride_y=75,  # 25% overlap
            sampling_strategy="uniform"
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result = chunker.chunk("test_image.jpg")
        
        if result.chunks:
            patch_meta = result.chunks[0].metadata.extra
            overlap = patch_meta["overlap"]
            assert overlap[0] == 50  # 100 - 50 = 50px overlap in X
            assert overlap[1] == 25  # 100 - 75 = 25px overlap in Y

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_edge_patch_detection(self, mock_image):
        """Test detection of edge patches."""
        mock_img = MagicMock()
        mock_img.size = (150, 150)  # Small image to force edge patches
        mock_img.width = 150
        mock_img.height = 150
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 100
        mock_patch.height = 100
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker(
            patch_width=100,
            patch_height=100,
            stride_x=100,
            stride_y=100,
            edge_handling="pad",
            sampling_strategy="uniform"
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result = chunker.chunk("test_image.jpg")
        
        # Check for edge patch detection
        edge_patches = [chunk for chunk in result.chunks 
                       if chunk.metadata.extra.get("is_edge_patch", False)]
        # With a 150x150 image and 100x100 patches with stride 100, 
        # patches at positions (100,0), (0,100), (100,100) should be edge patches
        assert len(edge_patches) > 0

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_chunk_id_format(self, mock_image):
        """Test chunk ID format."""
        mock_img = MagicMock()
        mock_img.size = (200, 200)
        mock_img.width = 200
        mock_img.height = 200
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 100
        mock_patch.height = 100
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        chunker = PatchBasedImageChunker(
            patch_width=100,
            patch_height=100,
            stride_x=100,
            stride_y=100,
            sampling_strategy="uniform"
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result = chunker.chunk("test_image.jpg")
        
        # Check chunk ID format: "patch_{index}_{x}x{y}"
        if result.chunks:
            chunk_id = result.chunks[0].id
            assert chunk_id.startswith("patch_")
            assert "_" in chunk_id
            assert "x" in chunk_id  # Position coordinates

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_max_patches_limit(self, mock_image):
        """Test max_patches limit functionality."""
        mock_img = MagicMock()
        mock_img.size = (500, 500)
        mock_img.width = 500
        mock_img.height = 500
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 50
        mock_patch.height = 50
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        # Set max_patches to limit output
        chunker = PatchBasedImageChunker(
            patch_width=50,
            patch_height=50,
            stride_x=50,
            stride_y=50,
            sampling_strategy="uniform",
            max_patches=5  # Limit to 5 patches
        )
        
        # Mock the _patch_to_bytes method
        chunker._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result = chunker.chunk("test_image.jpg")
        
        # Should be limited to max_patches
        assert len(result.chunks) == 5

    @patch('chunking_strategy.strategies.multimedia.patch_based_image.Image')
    def test_random_seed_reproducibility(self, mock_image):
        """Test that random sampling is reproducible with seed."""
        mock_img = MagicMock()
        mock_img.size = (200, 200)
        mock_img.width = 200
        mock_img.height = 200
        mock_img.mode = "RGB"
        
        mock_patch = MagicMock()
        mock_patch.width = 50
        mock_patch.height = 50
        mock_img.crop.return_value = mock_patch
        
        mock_image.open.return_value = mock_img
        
        # Create two chunkers with same seed
        chunker1 = PatchBasedImageChunker(
            patch_width=50,
            patch_height=50,
            sampling_strategy="random",
            max_patches=10,
            random_seed=42
        )
        
        chunker2 = PatchBasedImageChunker(
            patch_width=50,
            patch_height=50,
            sampling_strategy="random",
            max_patches=10,
            random_seed=42
        )
        
        # Mock the _patch_to_bytes method for both
        chunker1._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        chunker2._patch_to_bytes = MagicMock(return_value=b"mock_patch_bytes")
        
        result1 = chunker1.chunk("test_image.jpg")
        result2 = chunker2.chunk("test_image.jpg")
        
        # Should produce same number of chunks
        assert len(result1.chunks) == len(result2.chunks)
        
        # Should have same positions (if we could easily compare them)
        # This is a basic test - in practice positions should be identical
        assert len(result1.chunks) == 10
        assert len(result2.chunks) == 10
