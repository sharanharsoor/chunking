"""
Patch-based Image Chunking Strategy

This module implements patch-based image chunking using a sliding window approach.
Unlike grid-based chunking, patch-based chunking extracts patches with flexible
positioning and overlap control through configurable stride parameters.

Key features:
- Sliding window patch extraction
- Configurable patch dimensions and stride
- Multiple sampling strategies (uniform, random, edge-aware)
- Support for overlapping patches
- Flexible edge handling
- Multiple output formats (PNG, JPEG, BMP, TIFF, WEBP)
"""

import io
import time
import logging
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any
import random

from chunking_strategy.core.base import StreamableChunker, ModalityType, Chunk, ChunkingResult, ChunkMetadata
from chunking_strategy.core.registry import register_chunker

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


@register_chunker(
    name="patch_based_image",
    category="multimedia",
    supported_modalities=[ModalityType.IMAGE],
    supported_formats=[".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".ico", ".svg"],
    description="Extracts patches from images using sliding window approach with configurable stride and sampling strategies"
)
class PatchBasedImageChunker(StreamableChunker):
    """
    Patch-based image chunker that extracts patches using sliding window approach.

    This chunker provides more flexible patch extraction compared to grid-based chunking,
    with support for overlapping patches, different sampling strategies, and configurable stride.
    """

    def __init__(self,
                 patch_width: int = 224,
                 patch_height: int = 224,
                 stride_x: int = 112,
                 stride_y: int = 112,
                 sampling_strategy: str = "uniform",
                 max_patches: Optional[int] = None,
                 edge_handling: str = "pad",
                 preserve_aspect_ratio: bool = False,
                 output_format: str = "PNG",
                 random_seed: Optional[int] = None,
                 **kwargs):
        """
        Initialize patch-based image chunker.

        Args:
            patch_width: Width of each patch in pixels (default: 224)
            patch_height: Height of each patch in pixels (default: 224)
            stride_x: Horizontal step size between patches (default: 112, 50% overlap)
            stride_y: Vertical step size between patches (default: 112, 50% overlap)
            sampling_strategy: Patch sampling strategy - "uniform", "random", "edge_aware" (default: "uniform")
            max_patches: Maximum number of patches to extract (None = no limit)
            edge_handling: How to handle edge patches - "pad", "crop", "skip" (default: "pad")
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing (default: False)
            output_format: Output image format - PNG, JPEG, BMP, TIFF, WEBP (default: PNG)
            random_seed: Seed for random sampling (default: None)
            **kwargs: Additional parameters
        """
        # Initialize base class
        name = kwargs.pop("name", "patch_based_image")
        super().__init__(
            name=name,
            **kwargs
        )

        if not PIL_AVAILABLE:
            raise ImportError(
                "Pillow (PIL) is required for patch-based image chunking. "
                "Install it with: pip install Pillow"
            )

        # Validate parameters
        if patch_width <= 0 or patch_height <= 0:
            raise ValueError("Patch dimensions must be positive")
        if stride_x <= 0 or stride_y <= 0:
            raise ValueError("Stride values must be positive")
        if sampling_strategy not in ["uniform", "random", "edge_aware"]:
            raise ValueError("sampling_strategy must be one of: uniform, random, edge_aware")
        if edge_handling not in ["pad", "crop", "skip"]:
            raise ValueError("edge_handling must be one of: pad, crop, skip")
        if output_format.upper() not in ["PNG", "JPEG", "BMP", "TIFF", "WEBP"]:
            raise ValueError("output_format must be one of: PNG, JPEG, BMP, TIFF, WEBP")

        self.patch_width = patch_width
        self.patch_height = patch_height
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.sampling_strategy = sampling_strategy
        self.max_patches = max_patches
        self.edge_handling = edge_handling
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.output_format = output_format.upper()
        self.random_seed = random_seed

        # Set random seed if provided
        if self.random_seed is not None:
            random.seed(self.random_seed)

        logger.debug(f"Initialized PatchBasedImageChunker: "
                    f"{patch_width}x{patch_height} patches, stride={stride_x}x{stride_y}, "
                    f"strategy={sampling_strategy}, format={output_format}")

    def _load_image(self, content: Union[str, Path, bytes]) -> Image.Image:
        """Load image from various input types."""
        try:
            if isinstance(content, (str, Path)):
                # Check if it's obviously text data (very long string with newlines)
                path_str = str(content)
                if len(path_str) > 1000 or path_str.count('\n') > 5:
                    newline_count = path_str.count('\n')
                    raise ValueError(f"Content appears to be text data, not an image file path (length: {len(path_str)}, lines: {newline_count})")

                # Let PIL handle the file opening - it will give appropriate errors
                # This allows mocked tests and valid file paths to work
                image = Image.open(content)
            elif isinstance(content, bytes):
                image = Image.open(io.BytesIO(content))
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")

            return image
        except Exception as e:
            # Re-raise the original exception from PIL for better error messages
            raise ValueError(f"Failed to load image: {e}")

    def _patch_to_bytes(self, patch: Image.Image) -> bytes:
        """Convert a patch image to bytes in the specified output format."""
        buffer = io.BytesIO()

        # Convert RGBA to RGB for JPEG format
        if self.output_format == "JPEG" and patch.mode == "RGBA":
            # Create white background
            background = Image.new("RGB", patch.size, (255, 255, 255))
            background.paste(patch, mask=patch.split()[-1] if patch.mode == "RGBA" else None)
            patch = background

        patch.save(buffer, format=self.output_format)
        return buffer.getvalue()

    def _calculate_patch_positions(self, image_width: int, image_height: int) -> List[Tuple[int, int]]:
        """Calculate patch positions based on sampling strategy."""
        positions = []

        if self.sampling_strategy == "uniform":
            # Uniform grid sampling with stride
            y = 0
            while y < image_height:
                x = 0
                while x < image_width:
                    positions.append((x, y))
                    x += self.stride_x
                y += self.stride_y

        elif self.sampling_strategy == "random":
            # Random sampling within image bounds
            if self.max_patches is None:
                # Default to approximate uniform density for random sampling
                approx_patches_x = max(1, (image_width - self.patch_width) // self.stride_x + 1)
                approx_patches_y = max(1, (image_height - self.patch_height) // self.stride_y + 1)
                num_patches = min(approx_patches_x * approx_patches_y, 100)  # Cap at 100 for random
            else:
                num_patches = self.max_patches

            max_x = max(0, image_width - self.patch_width)
            max_y = max(0, image_height - self.patch_height)

            for _ in range(num_patches):
                x = random.randint(0, max_x) if max_x > 0 else 0
                y = random.randint(0, max_y) if max_y > 0 else 0
                positions.append((x, y))

        elif self.sampling_strategy == "edge_aware":
            # Focus on edges and corners with some center patches
            positions = []

            # Corner patches
            corners = [(0, 0),
                      (max(0, image_width - self.patch_width), 0),
                      (0, max(0, image_height - self.patch_height)),
                      (max(0, image_width - self.patch_width), max(0, image_height - self.patch_height))]
            positions.extend(corners)

            # Edge patches (top, bottom, left, right)
            edge_stride = max(self.stride_x, self.stride_y) * 2

            # Top and bottom edges
            for x in range(0, image_width - self.patch_width + 1, edge_stride):
                positions.append((x, 0))  # Top edge
                positions.append((x, max(0, image_height - self.patch_height)))  # Bottom edge

            # Left and right edges
            for y in range(0, image_height - self.patch_height + 1, edge_stride):
                positions.append((0, y))  # Left edge
                positions.append((max(0, image_width - self.patch_width), y))  # Right edge

            # Center patches (sparser)
            center_x = image_width // 2 - self.patch_width // 2
            center_y = image_height // 2 - self.patch_height // 2
            if center_x >= 0 and center_y >= 0:
                positions.append((center_x, center_y))

            # Remove duplicates
            positions = list(set(positions))

        # Apply max_patches limit if specified
        if self.max_patches is not None and len(positions) > self.max_patches:
            if self.sampling_strategy == "random":
                # Already handled above
                pass
            else:
                # For uniform and edge_aware, take first N positions
                positions = positions[:self.max_patches]

        return positions

    def _extract_patch(self, image: Image.Image, x: int, y: int, patch_index: int) -> Tuple[bytes, Dict[str, Any]]:
        """Extract a single patch from the image at given position."""
        image_width, image_height = image.size

        # Handle edge cases based on edge_handling strategy
        if self.edge_handling == "skip":
            if x + self.patch_width > image_width or y + self.patch_height > image_height:
                return None, None

        # Calculate actual patch boundaries
        if self.edge_handling == "pad":
            # Pad the image if patch extends beyond boundaries
            if x + self.patch_width > image_width or y + self.patch_height > image_height:
                # Create padded image
                padded_width = max(image_width, x + self.patch_width)
                padded_height = max(image_height, y + self.patch_height)

                padded_image = Image.new(image.mode, (padded_width, padded_height),
                                       color=(0, 0, 0) if image.mode == "RGB" else 0)
                padded_image.paste(image, (0, 0))
                patch = padded_image.crop((x, y, x + self.patch_width, y + self.patch_height))
            else:
                patch = image.crop((x, y, x + self.patch_width, y + self.patch_height))

        elif self.edge_handling == "crop":
            # Crop to fit within image boundaries
            end_x = min(x + self.patch_width, image_width)
            end_y = min(y + self.patch_height, image_height)
            patch = image.crop((x, y, end_x, end_y))

            # Resize to target dimensions if needed
            if patch.size != (self.patch_width, self.patch_height):
                resize_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                patch = patch.resize((self.patch_width, self.patch_height), resize_method)

        # Convert patch to bytes
        patch_bytes = self._patch_to_bytes(patch)

        # Create patch metadata
        overlap_x = max(0, self.patch_width - self.stride_x) if self.stride_x < self.patch_width else 0
        overlap_y = max(0, self.patch_height - self.stride_y) if self.stride_y < self.patch_height else 0

        patch_metadata = {
            "patch_index": patch_index,
            "position": (x, y),
            "patch_size": (patch.width, patch.height),
            "target_size": (self.patch_width, self.patch_height),
            "stride": (self.stride_x, self.stride_y),
            "overlap": (overlap_x, overlap_y),
            "sampling_strategy": self.sampling_strategy,
            "edge_handling": self.edge_handling,
            "is_edge_patch": (x + self.patch_width > image_width or y + self.patch_height > image_height)
        }

        return patch_bytes, patch_metadata

    def chunk(self, content: Union[str, Path, bytes], source_info: Optional[Dict[str, Any]] = None, **kwargs) -> ChunkingResult:
        """
        Extract patches from an image using patch-based approach.

        Args:
            content: Image file path, Path object, or image bytes
            source_info: Optional source information dictionary
            **kwargs: Additional parameters

        Returns:
            ChunkingResult containing extracted patches as chunks
        """
        start_time = time.time()

        try:
            # Load the image
            image = self._load_image(content)
            image_width, image_height = image.size

            logger.debug(f"Processing image: {image_width}x{image_height}, "
                        f"mode={image.mode}, format={getattr(image, 'format', 'unknown')}")

            # Calculate patch positions based on sampling strategy
            positions = self._calculate_patch_positions(image_width, image_height)

            logger.debug(f"Extracting {len(positions)} patches using {self.sampling_strategy} strategy")

            # Extract patches
            chunks = []
            source_metadata = {
                "source_path": str(content) if isinstance(content, (str, Path)) else "bytes_input",
                "image_size": (image_width, image_height),
                "image_mode": image.mode,
                "image_format": getattr(image, 'format', 'unknown'),
                "total_patches": len(positions)
            }

            patch_config = {
                "patch_size": (self.patch_width, self.patch_height),
                "stride": (self.stride_x, self.stride_y),
                "sampling_strategy": self.sampling_strategy,
                "max_patches": self.max_patches,
                "edge_handling": self.edge_handling,
                "output_format": self.output_format
            }

            for patch_index, (x, y) in enumerate(positions):
                patch_result = self._extract_patch(image, x, y, patch_index)
                if patch_result[0] is None:  # Skip patches if edge_handling is "skip"
                    continue

                patch_bytes, patch_metadata = patch_result

                # Create chunk metadata
                chunk_metadata = ChunkMetadata(
                    source=source_metadata.get("source_path", "unknown"),
                    source_type="image",
                    position=f"patch_{patch_index}_at_{x}x{y}",
                    bbox=(x, y, x + self.patch_width, y + self.patch_height),
                    chunker_used="patch_based_image",
                    extra={
                        **patch_metadata,
                        "source_image": source_metadata,
                        "patch_config": patch_config,
                        "chunk_format": self.output_format
                    }
                )

                chunk = Chunk(
                    id=f"patch_{patch_index}_{x}x{y}",
                    content=patch_bytes,
                    modality=ModalityType.IMAGE,
                    size=len(patch_bytes),
                    metadata=chunk_metadata
                )

                chunks.append(chunk)

            processing_time = time.time() - start_time

            logger.info(f"Extracted {len(chunks)} patches in {processing_time:.3f}s using {self.sampling_strategy} strategy")

            return ChunkingResult(
                chunks=chunks,
                processing_time=processing_time,
                source_info={
                    **source_metadata,
                    "patch_config": patch_config,
                    "patches_extracted": len(chunks),
                    "strategy_used": "patch_based_image"
                }
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

    def chunk_stream(self, content_stream) -> ChunkingResult:
        """
        Process streaming content by accumulating it and calling chunk().

        Args:
            content_stream: Stream of content to process

        Returns:
            ChunkingResult from processing accumulated content
        """
        # Accumulate stream content
        accumulated_content = b""
        for chunk in content_stream:
            if isinstance(chunk, str):
                accumulated_content += chunk.encode('utf-8')
            else:
                accumulated_content += chunk

        return self.chunk(accumulated_content)

    def get_chunk_size_estimate(self, content_length: int) -> int:
        """
        Estimate the number of patches for given content length.
        This is approximate since it's based on estimated image dimensions.
        """
        if content_length <= 0:
            return 0

        # Rough estimate: assume average image density
        if content_length < 10000:  # Small image
            estimated_width, estimated_height = 200, 150
        elif content_length < 100000:  # Medium image
            estimated_width, estimated_height = 400, 300
        elif content_length < 1000000:  # Large image
            estimated_width, estimated_height = 800, 600
        else:  # Very large image
            # Scale based on content length
            estimated_pixels = content_length / 3  # Rough bytes per pixel estimate
            estimated_dimension = int(estimated_pixels ** 0.5)
            estimated_width = estimated_height = max(1000, estimated_dimension)

        # Calculate approximate number of patches based on sampling strategy
        if self.sampling_strategy == "uniform":
            patches_x = max(1, (estimated_width - self.patch_width) // self.stride_x + 1)
            patches_y = max(1, (estimated_height - self.patch_height) // self.stride_y + 1)
            estimated_patches = patches_x * patches_y
        elif self.sampling_strategy == "random":
            # Random sampling uses max_patches or default approximation
            if self.max_patches:
                estimated_patches = self.max_patches
            else:
                patches_x = max(1, estimated_width // self.stride_x)
                patches_y = max(1, estimated_height // self.stride_y)
                estimated_patches = min(patches_x * patches_y, 100)  # Cap at 100
        else:  # edge_aware
            # Edge-aware sampling produces fewer patches
            estimated_patches = min(20, max(10, estimated_width // self.patch_width))

        # Apply max_patches limit if set
        if self.max_patches is not None:
            estimated_patches = min(estimated_patches, self.max_patches)

        return max(1, estimated_patches)

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration of the chunker."""
        return {
            "patch_width": self.patch_width,
            "patch_height": self.patch_height,
            "stride_x": self.stride_x,
            "stride_y": self.stride_y,
            "sampling_strategy": self.sampling_strategy,
            "max_patches": self.max_patches,
            "edge_handling": self.edge_handling,
            "preserve_aspect_ratio": self.preserve_aspect_ratio,
            "output_format": self.output_format,
            "random_seed": self.random_seed
        }
