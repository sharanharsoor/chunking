"""token_based decode windows: offsets exist when preserve_word_boundaries is off."""

import pytest

tiktoken = pytest.importorskip("tiktoken")

from chunking_strategy.strategies.text.token_based_chunker import TokenBasedChunker


def test_decode_windows_have_offsets():
    text = "The quick brown fox jumps over the lazy dog. Token windows cut on cl100k_base, not characters."
    chunker = TokenBasedChunker(
        tokens_per_chunk=8,
        overlap_tokens=0,
        preserve_word_boundaries=False,
        min_chunk_tokens=1,
        max_chunk_chars=100000,
    )
    result = chunker.chunk(text)
    assert result.chunks
    assert all(c.start is not None and c.end is not None for c in result.chunks)
    assert result.chunks[0].start == 0
    assert all(c.token_count for c in result.chunks)
    first = result.chunks[0]
    assert text[first.start : first.end] == first.content
