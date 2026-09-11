"""Missing extras raise a one-line pip hint, not a traceback soup."""

import sys

from chunking_strategy import create_chunker
from chunking_strategy.core.token_packing import require_encoding
from chunking_strategy.exceptions import MissingExtraError


def test_import_does_not_load_optional_stacks():
    import chunking_strategy  # noqa: F401

    assert "torch" not in sys.modules
    assert "cv2" not in sys.modules
    assert "tika" not in sys.modules


def test_semantic_without_ml_is_missing_extra():
    try:
        import sentence_transformers  # noqa: F401
        return
    except ImportError:
        pass
    try:
        create_chunker("semantic")
        raise AssertionError("expected MissingExtraError")
    except MissingExtraError as exc:
        assert "chunking-strategy[ml]" in str(exc)
        assert exc.extra == "ml"


def test_max_tokens_without_tiktoken_is_missing_extra(monkeypatch):
    import builtins
    import sys

    monkeypatch.delitem(sys.modules, "tiktoken", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken" or name.startswith("tiktoken."):
            raise ImportError("no tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        require_encoding()
        raise AssertionError("expected MissingExtraError")
    except MissingExtraError as exc:
        assert exc.extra == "tiktoken"
        assert "chunking-strategy[tiktoken]" in str(exc)
