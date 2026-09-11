"""tiktoken packing for max_tokens / overlap_tokens.

Do not approximate tokens as chars/4. If tiktoken is missing, raise with an
install line. Callers that never set max_tokens must not import this module's
encoder path at runtime — use require_encoding() only when packing by tokens.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

INSTALL = "pip install chunking-strategy[tiktoken]"


def require_encoding(name: str = "cl100k_base"):
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            f"max_tokens requires tiktoken. Install with: {INSTALL}"
        ) from exc
    return tiktoken.get_encoding(name)


def count_tokens(text: str, enc) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def token_windows(
    n_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> List[Tuple[int, int]]:
    """Sliding [start, end) token-index windows."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be less than max_tokens")
    step = max(1, max_tokens - overlap_tokens)
    out: List[Tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + max_tokens, n_tokens)
        out.append((start, end))
        if end >= n_tokens:
            break
        start += step
    return out


def pack_unit_ranges(
    sizes: Sequence[int],
    max_tokens: int,
    overlap_tokens: int = 0,
    max_units: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Greedy [start, end) index ranges over units with token sizes.

    A unit larger than max_tokens is emitted alone. overlap_tokens walks back
    whole units from the end of the previous range.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    n = len(sizes)
    ranges: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        start = i
        used = 0
        units = 0
        while i < n:
            nxt = sizes[i]
            if units and (
                used + nxt > max_tokens
                or (max_units is not None and units >= max_units)
            ):
                break
            if units == 0 and nxt > max_tokens:
                i += 1
                units = 1
                used = nxt
                break
            used += nxt
            units += 1
            i += 1
        ranges.append((start, i))
        if i >= n:
            break
        if overlap_tokens > 0 and i > start:
            acc = 0
            k = i
            while k > start and acc < overlap_tokens:
                k -= 1
                acc += sizes[k]
            if k < i:
                i = k
            if i <= start:
                i = start + 1
        elif overlap_tokens == 0 and i == start:
            i = start + 1
    return ranges
