"""
Grid-based Image Chunker

This chunker divides images into fixed-size grid tiles, useful for processing large images
in smaller, manageable chunks. Supports various image formats and provides flexible
grid sizing options.
"""

import io
import time
import logging
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Tuple, TYPE_CHECKING

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    if TYPE_CHECKING:
        from PIL import Image
    else:
        Image = None

try:
    import numpy as np
except ImportError:
    np = None

from chunking_strategy.core.base import StreamableChunker, Chunk, ChunkingResult, ModalityType, ChunkMetadata
from chunking_strategy.core.registry import register_chunker

logger = logging.getLogger(__name__)


@register_chunker(
    name="grid_based_image",
    category="multimedia",
    supported_modalities=[ModalityType.IMAGE],
    supported_formats=[".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".ico", ".svg"],
    description="Grid-based image chunking that divides images into fixed-size tiles"
)
class GridBasedImageChunker(StreamableChunker):
    """
    Grid-based image chunker that divides images into regular grid tiles.

    This chunker is ideal for:
    - Processing large images in smaller tiles
    - Creating uniform image patches for analysis
    - Preparing images for parallel processing
    - Geographic/satellite image processing
    """

    def __init__(self,
                 tile_width: int = 256,
                 tile_height: int = 256,
                 overlap_pixels: int = 0,
                 preserve_aspect_ratio: bool = False,
                 pad_incomplete_tiles: bool = True,
                 output_format: str = "PNG",
                 **kwargs):
        """
        Initialize the Grid-based Image Chunker.

        Args:
            tile_width: Width of each tile in pixels (default: 256)
            tile_height: Height of each tile in pixels (default: 256)
            overlap_pixels: Overlap between adjacent tiles in pixels (default: 0)
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing (default: False)
            pad_incomplete_tiles: Whether to pad incomplete edge tiles (default: True)
            output_format: Output image format for chunks (default: "PNG")
        """
        name = kwargs.pop("name", "grid_based_image")
        super().__init__(name=name, **kwargs)

        self.tile_width = max(1, tile_width)
        self.tile_height = max(1, tile_height)
        self.overlap_pixels = max(0, overlap_pixels)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.pad_incomplete_tiles = pad_incomplete_tiles
        self.output_format = output_format.upper()

        # Validate output format
        valid_formats = ["PNG", "JPEG", "BMP", "TIFF", "WEBP"]
        if self.output_format not in valid_formats:
            logger.warning(f"Invalid output format {self.output_format}, defaulting to PNG")
            self.output_format = "PNG"

    def _load_image(self, content: Union[str, bytes, Path]) -> "Image.Image":
        """Load image from various input types."""
        if Image is None:
            raise ImportError("PIL (Pillow) is required for image processing. Install with: pip install Pillow")

        try:
            if isinstance(content, bytes):
                return Image.open(io.BytesIO(content))
            elif isinstance(content, (str, Path)):
                # Check if it's obviously text data (very long string with newlines)
                path_str = str(content)
                if len(path_str) > 1000 or path_str.count('\n') > 5:
                    newline_count = path_str.count('\n')
                    raise ValueError(f"Content appears to be text data, not an image file path (length: {len(path_str)}, lines: {newline_count})")

                # Let PIL handle the file opening - it will give appropriate errors
                # This allows mocked tests and valid file paths to work
                return Image.open(content)
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        except Exception as e:
            # Re-raise the original exception from PIL for better error messages
            raise ValueError(f"Failed to load image: {e}")

    def _extract_source_info(self, image: "Image.Image", source_path: Optional[str] = None) -> Dict[str, Any]:
        """Extract metadata from the image."""
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format or "Unknown",
            "source_path": source_path or "Unknown"
        }

    def _create_grid_tiles(self, image: "Image.Image") -> List[Tuple["Image.Image", Dict[str, Any]]]:
        """
        Create grid tiles from the image.

        Returns:
            List of (tile_image, tile_metadata) tuples
        """
        tiles = []
        img_width, img_height = image.size

        # Calculate step size (accounting for overlap)
        step_width = self.tile_width - self.overlap_pixels
        step_height = self.tile_height - self.overlap_pixels

        # Ensure step size is at least 1
        step_width = max(1, step_width)
        step_height = max(1, step_height)

        tile_index = 0

        for y in range(0, img_height, step_height):
            for x in range(0, img_width, step_width):
                # Calculate tile boundaries
                x1 = x
                y1 = y
                x2 = min(x + self.tile_width, img_width)
                y2 = min(y + self.tile_height, img_height)

                # Extract tile
                tile = image.crop((x1, y1, x2, y2))

                # Handle incomplete tiles
                actual_width = x2 - x1
                actual_height = y2 - y1

                if self.pad_incomplete_tiles and (actual_width < self.tile_width or actual_height < self.tile_height):
                    # Create a new image with the desired tile size and paste the tile
                    padded_tile = Image.new(image.mode, (self.tile_width, self.tile_height), color=0)
                    padded_tile.paste(tile, (0, 0))
                    tile = padded_tile

                # Create tile metadata
                tile_metadata = {
                    "tile_index": tile_index,
                    "grid_position": (x // step_width, y // step_height),
                    "pixel_position": (x1, y1),
                    "original_size": (actual_width, actual_height),
                    "tile_size": (tile.width, tile.height),
                    "is_edge_tile": (x2 >= img_width or y2 >= img_height),
                    "overlap_pixels": self.overlap_pixels
                }

                tiles.append((tile, tile_metadata))
                tile_index += 1

        return tiles

    def _tile_to_bytes(self, tile: "Image.Image") -> bytes:
        """Convert PIL Image tile to bytes."""
        buffer = io.BytesIO()
        save_format = self.output_format

        # Handle format-specific options
        save_kwargs = {}
        if save_format == "JPEG":
            save_kwargs["quality"] = 95
            # Convert RGBA to RGB for JPEG
            if tile.mode == "RGBA":
                background = Image.new("RGB", tile.size, (255, 255, 255))
                background.paste(tile, mask=tile.split()[-1] if len(tile.split()) == 4 else None)
                tile = background

        tile.save(buffer, format=save_format, **save_kwargs)
        return buffer.getvalue()

    def chunk(self, content: Union[str, bytes, Path], source_info: Optional[Dict[str, Any]] = None, **kwargs) -> ChunkingResult:
        """
        Chunk an image into grid tiles.

        Args:
            content: Image content (file path, bytes, or Path object)
            source_info: Optional source information
            **kwargs: Additional parameters

        Returns:
            ChunkingResult containing image tiles as chunks
        """
        start_time = time.time()

        try:
            # Load image
            image = self._load_image(content)
            source_metadata = self._extract_source_info(image, str(content) if isinstance(content, (str, Path)) else None)

            # Create grid tiles
            tiles_data = self._create_grid_tiles(image)

            # Convert tiles to chunks
            chunks = []
            for tile_image, tile_metadata in tiles_data:
                # Convert tile to bytes
                tile_bytes = self._tile_to_bytes(tile_image)

                # Create chunk metadata
                chunk_metadata = ChunkMetadata(
                    source=source_metadata.get("source_path", "unknown"),
                    source_type="image",
                    position=f"tile_{tile_metadata['grid_position'][0]}x{tile_metadata['grid_position'][1]}",
                    bbox=(tile_metadata["pixel_position"][0], tile_metadata["pixel_position"][1],
                          tile_metadata["pixel_position"][0] + tile_metadata["tile_size"][0],
                          tile_metadata["pixel_position"][1] + tile_metadata["tile_size"][1]),
                    chunker_used="grid_based_image",
                    extra={
                        **tile_metadata,
                        "source_image": source_metadata,
                        "chunk_format": self.output_format
                    }
                )

                # Create chunk
                chunk = Chunk(
                    id=f"tile_{tile_metadata['tile_index']}_{tile_metadata['grid_position'][0]}x{tile_metadata['grid_position'][1]}",
                    content=tile_bytes,
                    modality=ModalityType.IMAGE,
                    metadata=chunk_metadata
                )

                chunks.append(chunk)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create source info
            result_source_info = {
                "total_tiles": len(chunks),
                "source_image": source_metadata,
                "tile_config": {
                    "tile_width": self.tile_width,
                    "tile_height": self.tile_height,
                    "overlap_pixels": self.overlap_pixels,
                    "output_format": self.output_format,
                    "pad_incomplete_tiles": self.pad_incomplete_tiles
                }
            }

            if source_info:
                result_source_info.update(source_info)

            return ChunkingResult(
                chunks=chunks,
                processing_time=processing_time,
                source_info=result_source_info
            )

        except (ImportError) as e:
            # Re-raise ImportError for missing dependencies
            raise e
        except (FileNotFoundError, ValueError, TypeError) as e:
            # Handle other specific errors gracefully
            logger.error(f"Failed to chunk image: {e}")
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                source_info={"error": f"Failed to chunk image: {e}"}
            )
        except Exception as e:
            logger.error(f"Failed to chunk image: {e}")
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                source_info={"error": f"Failed to chunk image: {e}"}
            )

    def chunk_stream(self, content_stream, source_info: Optional[Dict[str, Any]] = None, **kwargs) -> ChunkingResult:
        """
        Process streaming image content by accumulating and then chunking.

        Args:
            content_stream: Stream of image content
            source_info: Optional source information
            **kwargs: Additional parameters

        Returns:
            ChunkingResult containing image tiles
        """
        try:
            # Accumulate stream content
            accumulated_content = b""
            for chunk_data in content_stream:
                if isinstance(chunk_data, str):
                    chunk_data = chunk_data.encode('utf-8')
                accumulated_content += chunk_data

            # Process accumulated content
            return self.chunk(accumulated_content, source_info, **kwargs)

        except Exception as e:
            logger.error(f"Failed to process image stream: {e}")
            return ChunkingResult(
                chunks=[],
                processing_time=0.0,
                source_info={"error": f"Failed to process image stream: {e}"}
            )

    def get_chunk_size_estimate(self, content_length: int) -> int:
        """
        Estimate the number of chunks for given content length.
        This is approximate since it's based on average tile size.
        """
        if content_length <= 0:
            return 0

        # Rough estimate: assume average image density
        # Small files: ~100x100 pixels, Medium files: ~500x500, Large files: ~1000x1000+
        if content_length < 10000:  # Small image
            estimated_width, estimated_height = 100, 100
        elif content_length < 100000:  # Medium image
            estimated_width, estimated_height = 300, 200
        elif content_length < 1000000:  # Large image
            estimated_width, estimated_height = 800, 600
        else:  # Very large image
            # Scale based on content length
            estimated_pixels = content_length / 3  # Rough bytes per pixel estimate
            estimated_dimension = int(estimated_pixels ** 0.5)
            estimated_width = estimated_height = max(1000, estimated_dimension)

        # Calculate approximate number of tiles
        tiles_x = max(1, (estimated_width + self.tile_width - 1) // self.tile_width)
        tiles_y = max(1, (estimated_height + self.tile_height - 1) // self.tile_height)

        return tiles_x * tiles_y

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration of the chunker."""
        return {
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "overlap_pixels": self.overlap_pixels,
            "preserve_aspect_ratio": self.preserve_aspect_ratio,
            "pad_incomplete_tiles": self.pad_incomplete_tiles,
            "output_format": self.output_format
        }
