"""
General-purpose chunking strategies.

This module contains general-purpose chunking algorithms that work across
different content types and modalities. These are typically the most
fundamental and widely-applicable chunking approaches.

Strategies included:
- Fixed size chunking
- Rolling hash algorithms (polynomial, Rabin, BuzHash)
- Content-defined chunking (CDC) variants
- Hash-based approaches (Gear, Multi-level, TTTD)
- Adaptive and hybrid methods
"""

from chunking_strategy.strategies.general.fixed_size import FixedSizeChunker
from chunking_strategy.strategies.general.fastcdc_chunker import FastCDCChunker
from chunking_strategy.strategies.general.adaptive_chunker import AdaptiveChunker
from chunking_strategy.strategies.general.context_enriched_chunker import ContextEnrichedChunker

# Hash-based chunkers
from chunking_strategy.strategies.general.rolling_hash_chunker import RollingHashChunker
from chunking_strategy.strategies.general.rabin_fingerprinting_chunker import RabinFingerprintingChunker
from chunking_strategy.strategies.general.buzhash_chunker import BuzHashChunker
from chunking_strategy.strategies.general.gear_cdc_chunker import GearCDCChunker
from chunking_strategy.strategies.general.ml_cdc_chunker import MLCDCChunker
from chunking_strategy.strategies.general.tttd_chunker import TTTDChunker

__all__ = [
    # Original chunkers
    "FixedSizeChunker",
    "FastCDCChunker",
    "AdaptiveChunker",
    "ContextEnrichedChunker",
    
    # Hash-based chunkers
    "RollingHashChunker",
    "RabinFingerprintingChunker",
    "BuzHashChunker",
    "GearCDCChunker",
    "MLCDCChunker",
    "TTTDChunker",
]
