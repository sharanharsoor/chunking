"""
Text-specific chunking strategies.

This module contains chunking strategies specifically designed for text content,
including natural language processing approaches and text structure awareness.

Strategies included:
- Sentence-based chunking
- Paragraph-based chunking
- Token-based chunking
- Semantic chunking variants
- Structure-aware chunking
- Language-specific chunkers
"""

from chunking_strategy.strategies.text.sentence_based import SentenceBasedChunker
from chunking_strategy.strategies.text.paragraph_based import ParagraphBasedChunker
from chunking_strategy.strategies.text.fixed_length_word_chunker import FixedLengthWordChunker
from chunking_strategy.strategies.text.token_based_chunker import TokenBasedChunker
from chunking_strategy.strategies.text.overlapping_window_chunker import OverlappingWindowChunker
from chunking_strategy.strategies.text.semantic_chunker import SemanticChunker
from chunking_strategy.strategies.text.embedding_based_chunker import EmbeddingBasedChunker
from chunking_strategy.strategies.text.boundary_aware_chunker import BoundaryAwareChunker
from chunking_strategy.strategies.text.recursive_chunker import RecursiveChunker

__all__ = [
    "SentenceBasedChunker",
    "ParagraphBasedChunker",
    "FixedLengthWordChunker",
    "TokenBasedChunker",
    "OverlappingWindowChunker",
    "SemanticChunker",
    "EmbeddingBasedChunker",
    "BoundaryAwareChunker",
    "RecursiveChunker",
]
