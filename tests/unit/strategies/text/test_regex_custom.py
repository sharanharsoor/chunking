"""regex_custom: split at each regex match."""

from chunking_strategy import create_chunker, list_chunkers
from chunking_strategy.strategies.text.regex_custom_chunker import (
    RegexCustomChunker,
    split_regex_custom,
)


def test_splits_on_heading_marks():
    text = "# A\nhello\n\n# B\nworld\n"
    parts = split_regex_custom(text, r"^# ", multiline=True)
    assert len(parts) == 2
    assert parts[0][0] == 0
    assert parts[0][2].startswith("# A")
    assert parts[1][2].startswith("# B")


def test_no_match_is_one_chunk():
    text = "no delimiter here"
    parts = split_regex_custom(text, r"^# ", multiline=True)
    assert parts == [(0, len(text), text)]


def test_empty_pattern_is_whole_text():
    text = "keep me"
    parts = split_regex_custom(text, "", multiline=True)
    assert parts == [(0, len(text), text)]


def test_invalid_regex_raises():
    try:
        split_regex_custom("x", "(", True)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "invalid regex" in str(exc)


def test_create_chunker_and_listing():
    assert "regex_custom" in list_chunkers()
    chunker = create_chunker("regex_custom", pattern=r"\n\n", multiline=True)
    assert isinstance(chunker, RegexCustomChunker)
    result = chunker.chunk("one\n\ntwo")
    assert result.strategy_used == "regex_custom"
    assert len(result.chunks) == 2
