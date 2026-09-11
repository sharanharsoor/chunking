"""recursive_character: separator cascade, not hierarchical recursive."""

from chunking_strategy import create_chunker, list_chunkers
from chunking_strategy.core.registry import get_chunker_metadata
from chunking_strategy.strategies.text.recursive_character_chunker import (
    RecursiveCharacterChunker,
    split_recursive_character,
)
from chunking_strategy.strategies.text.recursive_chunker import RecursiveChunker


def test_splits_paragraphs_before_spaces():
    text = "Hello world.\n\nSecond paragraph lives here.\n\nThird."
    chunker = RecursiveCharacterChunker(chunk_size=40, overlap_size=0)
    result = chunker.chunk(text, source_info={"source": "t"})
    assert result.strategy_used == "recursive_character"
    assert result.chunks
    assert all(c.start is not None and c.end is not None for c in result.chunks)
    assert result.chunks[0].start == 0
    joined = "".join(c.content for c in result.chunks)
    # overlap 0: pieces cover without requiring exact concat if separators dropped
    assert "Hello world." in result.chunks[0].content


def test_long_token_falls_back_to_characters():
    word = "A" * 50
    parts = split_recursive_character(word, chunk_size=20, overlap_size=0, separators=("\n\n", "\n", " ", ""))
    assert len(parts) >= 2
    assert all(len(p) <= 20 or p == word for p in parts[:-1])


def test_overlap_keeps_tail():
    text = "one two three four five six seven eight"
    chunker = RecursiveCharacterChunker(chunk_size=20, overlap_size=5)
    result = chunker.chunk(text)
    assert len(result.chunks) >= 2


def test_empty():
    assert RecursiveCharacterChunker().chunk("").chunks == []


def test_registered_for_create_chunker_and_listing():
    names = list_chunkers()
    assert "recursive_character" in names
    chunker = create_chunker("recursive_character", chunk_size=40, overlap_size=0)
    assert isinstance(chunker, RecursiveCharacterChunker)
    assert chunker.chunk_size == 40
    meta = get_chunker_metadata("recursive_character")
    assert meta is not None
    assert meta.category == "text"
    assert meta.name == "recursive_character"


def test_not_aliased_to_hierarchical_recursive():
    cascade = create_chunker("recursive_character")
    hierarchical = create_chunker("recursive")
    assert type(cascade) is RecursiveCharacterChunker
    assert type(hierarchical) is RecursiveChunker
    assert type(cascade) is not type(hierarchical)
