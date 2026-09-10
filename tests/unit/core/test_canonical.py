"""Canonical chunking.v1 JSON is stable for fixtures."""

from chunking_strategy.core.base import (
    Chunk,
    ChunkMetadata,
    ChunkingResult,
    ModalityType,
)


def _result():
    chunk = Chunk(
        id="will-be-rewritten",
        content="Hello.",
        modality=ModalityType.TEXT,
        metadata=ChunkMetadata(
            source="hello.txt",
            chunker_used="sentence_based",
            extra={"offset_unit": "char", "sentence_spec": "simple_v1"},
        ),
        start=0,
        end=6,
        hash="abc",
    )
    return ChunkingResult(chunks=[chunk], strategy_used="sentence_based")


def test_fixture_ids_are_stable():
    raw = _result().to_canonical_json(
        fixture=True,
        params={"sentence_splitter": "simple_v1"},
        source={"name": "hello.txt", "size": 6, "sha256": "deadbeef"},
    )
    assert '"id": "chunk-0000"' in raw
    assert "processing_time" not in raw
    assert '"schema": "chunking.v1"' in raw
    # sorted keys: children_ids before content
    assert raw.index('"children_ids"') < raw.index('"content"')


def test_roundtrip():
    original = _result()
    raw = original.to_canonical_json(fixture=True, params={"max_sentences": 1})
    restored = ChunkingResult.from_canonical_json(raw)
    assert restored.strategy_used == "sentence_based"
    assert restored.chunks[0].content == "Hello."
    assert restored.chunks[0].start == 0
    assert restored.chunks[0].end == 6
    assert restored.chunks[0].id == "chunk-0000"
