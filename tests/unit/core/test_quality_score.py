"""Result-level quality_score: size + boundary + coverage. No embeddings."""

from chunking_strategy.core.base import Chunk, ChunkMetadata, ChunkingResult, ModalityType
from chunking_strategy.core.metrics import compute_result_quality


def _chunk(text, start, end, name="sentence_based"):
    return Chunk(
        id="c",
        content=text,
        modality=ModalityType.TEXT,
        metadata=ChunkMetadata(source="t", chunker_used=name),
        start=start,
        end=end,
        size=len(text),
    )


def test_sentence_endings_score_high():
    result = ChunkingResult(
        chunks=[
            _chunk("Hello.", 0, 6),
            _chunk("World.", 7, 13),
        ],
        strategy_used="sentence_based",
    )
    assert result.quality_score is not None
    assert result.quality_score > 0.7


def test_json_skips_boundary_term():
    chunks = [
        _chunk('{"a":1}', 0, 7, "json_chunker"),
        _chunk('{"b":2}', 8, 15, "json_chunker"),
    ]
    scored = compute_result_quality(
        ChunkingResult(chunks=chunks, strategy_used="json_chunker")
    )
    # equal sizes + full coverage, no penalty for missing ".!?"
    assert scored > 0.9


def test_coverage_uses_span_union_not_sum():
    chunks = [
        _chunk("Hello world.", 0, 12),
        _chunk("world. Next.", 6, 18),
    ]
    result = ChunkingResult(chunks=chunks, strategy_used="overlapping_window")
    assert 0.0 < result.quality_score <= 1.0
