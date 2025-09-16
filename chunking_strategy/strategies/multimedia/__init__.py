"""
Multimedia chunking strategies.

This package contains chunking strategies for multimedia content including
audio, video, and image processing.

Audio Strategies:
- time_based: Fixed time duration segments
- silence_based: Split on silence detection

Video Strategies:
- time_based: Fixed time duration segments
- scene_based: Split on scene changes

Image Strategies:
- grid_based: Fixed-size grid tiles
- patch_based: Sliding window patches with configurable stride

Usage:
    # Direct import of specific strategies
    from chunking_strategy.strategies.multimedia.time_based_audio import TimeBasedAudioChunker
    from chunking_strategy.strategies.multimedia.silence_based_audio import SilenceBasedAudioChunker
    from chunking_strategy.strategies.multimedia.grid_based_image import GridBasedImageChunker

    # Using the registry system (recommended)
    from chunking_strategy import create_chunker
    chunker = create_chunker("time_based_audio", segment_duration=30.0)
    chunker = create_chunker("grid_based_image", tile_width=256, tile_height=256)
    chunker = create_chunker("patch_based_image", patch_width=224, stride_x=112)
"""

# Import multimedia chunkers for registration
from chunking_strategy.strategies.multimedia.time_based_audio import TimeBasedAudioChunker
from chunking_strategy.strategies.multimedia.silence_based_audio import SilenceBasedAudioChunker
from chunking_strategy.strategies.multimedia.time_based_video import TimeBasedVideoChunker
from chunking_strategy.strategies.multimedia.scene_based_video import SceneBasedVideoChunker
from chunking_strategy.strategies.multimedia.grid_based_image import GridBasedImageChunker
from chunking_strategy.strategies.multimedia.patch_based_image import PatchBasedImageChunker

__all__ = [
    'TimeBasedAudioChunker',
    'SilenceBasedAudioChunker',
    'TimeBasedVideoChunker',
    'SceneBasedVideoChunker',
    'GridBasedImageChunker',
    'PatchBasedImageChunker',
]
