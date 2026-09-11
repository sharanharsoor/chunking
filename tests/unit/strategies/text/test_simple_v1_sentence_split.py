"""simple_v1 sentence splitter: abbreviations, Unicode, intentional lowercase non-split."""

from chunking_strategy.strategies.text.sentence_based import (
    SentenceBasedChunker,
    split_sentences_simple_v1,
)


UNICODE = """Dr. Müller said "3.14 is π." The U.K. agreed. See https://example.com...
he stopped. then left.
测试句子。日本語のテスト。
"""


def test_does_not_split_abbreviation_decimal_url():
    parts = split_sentences_simple_v1(UNICODE)
    joined = " ".join(parts)
    assert "Dr. Müller" in joined
    assert any("3.14" in p for p in parts)
    assert any("U.K." in p for p in parts)
    assert any("https://example.com" in p for p in parts)
    # First sentence keeps Dr. and 3.14 together
    assert parts[0].startswith("Dr. Müller")
    assert "3.14" in parts[0]
    assert any(p.startswith("The U.K.") for p in parts)


def test_lowercase_after_period_is_intentional_nonsplit():
    parts = split_sentences_simple_v1("he stopped. then left.")
    assert parts == ["he stopped. then left."]


def test_positive_split():
    parts = split_sentences_simple_v1("Hello. World. Done.")
    assert parts == ["Hello.", "World.", "Done."]


def test_cjk_ideographic_stop_is_not_ascii_terminator():
    parts = split_sentences_simple_v1("测试句子。日本語のテスト。")
    assert len(parts) == 1


def test_chunker_sets_char_offsets():
    chunker = SentenceBasedChunker(
        max_sentences=1,
        sentence_splitter="simple_v1",
        max_chunk_size=100000,
    )
    text = "Hello. World."
    result = chunker.chunk(text)
    assert len(result.chunks) == 2
    assert result.chunks[0].content == "Hello."
    assert result.chunks[0].start == 0
    assert result.chunks[0].end == 6
    assert result.chunks[1].content == "World."
    assert result.chunks[1].metadata.extra.get("sentence_spec") == "simple_v1"
    assert result.chunks[1].metadata.extra.get("offset_unit") == "char"


def test_default_splitter_is_simple_v1():
    assert SentenceBasedChunker().sentence_splitter == "simple_v1"
