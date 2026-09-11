"""Optional extras. Keep the pip line in the exception, not a traceback."""

from typing import Optional

from chunking_strategy.exceptions import MissingExtraError

# Names the in-browser lab actually runs. Everything else is python_only.
LAB_STRATEGIES = frozenset(
    {
        "fixed_size",
        "recursive_character",
        "sentence_based",
        "paragraph_based",
        "overlapping_window",
        "fixed_length_word",
        "token_based",
        "regex_custom",
        "markdown_chunker",
        "xml_html_chunker",
        "json_chunker",
        "csv_chunker",
        "python_code",
        "javascript_code",
        "css_code",
        "go_code",
        "java_code",
        "c_cpp_code",
        "rolling_hash",
        "fastcdc",
    }
)

ML_STRATEGIES = frozenset(
    {
        "semantic",
        "semantic_chunker",
        "semantic_chunking",
        "embedding_based",
        "embedding_based_chunker",
    }
)

TIKA_STRATEGIES = frozenset({"universal_document", "tika_chunker"})


def extra_for_strategy(name: str) -> Optional[str]:
    key = (name or "").lower()
    if key in ML_STRATEGIES:
        return "ml"
    if key in TIKA_STRATEGIES:
        return "tika"
    return None


def ensure_strategy_extra(name: str) -> None:
    extra = extra_for_strategy(name)
    if extra == "ml":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise MissingExtraError("ml") from exc
    elif extra == "tika":
        try:
            import tika  # noqa: F401
        except ImportError as exc:
            raise MissingExtraError("tika") from exc
