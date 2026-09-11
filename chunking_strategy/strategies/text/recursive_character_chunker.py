"""LangChain-style recursive character splitting.

Not the hierarchical ``recursive`` strategy. Separators try ``\\n\\n``, then
``\\n``, then space, then characters, merging pieces up to ``chunk_size``.
Separators are discarded (LangChain ``keep_separator=False``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from chunking_strategy.core.base import (
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType,
    StreamableChunker,
)
from chunking_strategy.core.canonical import sha256_text
from chunking_strategy.core.registry import (
    ComplexityLevel,
    MemoryUsage,
    SpeedLevel,
    register_chunker,
)

DEFAULT_SEPARATORS = ("\n\n", "\n", " ", "")


def _length(text: str) -> int:
    return len(text)


def _split_keep_empty_out(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)
    return [p for p in text.split(separator) if p != ""]


def _join(parts: Sequence[str], separator: str) -> Optional[str]:
    if not parts:
        return None
    text = separator.join(parts)
    return text if text else None


def _merge_splits(
    splits: Sequence[str],
    separator: str,
    chunk_size: int,
    overlap_size: int,
) -> List[str]:
    sep_len = _length(separator) if separator else 0
    docs: List[str] = []
    current: List[str] = []
    total = 0
    for piece in splits:
        plen = _length(piece)
        if total + plen + (sep_len if current else 0) > chunk_size:
            if current:
                joined = _join(current, separator)
                if joined is not None:
                    docs.append(joined)
                while current and (
                    total > overlap_size
                    or (
                        total + plen + (sep_len if current else 0) > chunk_size
                        and total > 0
                    )
                ):
                    total -= _length(current[0]) + (sep_len if len(current) > 1 else 0)
                    current = current[1:]
        current.append(piece)
        total += plen + (sep_len if len(current) > 1 else 0)
    joined = _join(current, separator)
    if joined is not None:
        docs.append(joined)
    return docs


def split_recursive_character(
    text: str,
    chunk_size: int,
    overlap_size: int,
    separators: Sequence[str],
) -> List[str]:
    if not text:
        return []
    seps = list(separators) if separators else list(DEFAULT_SEPARATORS)
    if seps[-1] != "":
        seps.append("")

    def split_with(piece: str, seps_here: List[str]) -> List[str]:
        separator = seps_here[-1]
        rest: List[str] = []
        for i, sep in enumerate(seps_here):
            if sep == "":
                separator = sep
                rest = []
                break
            if sep in piece:
                separator = sep
                rest = seps_here[i + 1 :]
                break
        splits = _split_keep_empty_out(piece, separator)
        out: List[str] = []
        good: List[str] = []
        for part in splits:
            if _length(part) < chunk_size:
                good.append(part)
                continue
            if good:
                out.extend(_merge_splits(good, separator, chunk_size, overlap_size))
                good = []
            if not rest:
                out.append(part)
            else:
                out.extend(split_with(part, rest))
        if good:
            out.extend(_merge_splits(good, separator, chunk_size, overlap_size))
        return out

    return split_with(text, seps)


def locate_pieces(
    text: str, pieces: Sequence[str], overlap_size: int = 0
) -> List[tuple[int, int, str]]:
    hint = 0
    found: List[tuple[int, int, str]] = []
    for piece in pieces:
        idx = text.find(piece, hint)
        if idx < 0:
            idx = text.find(piece)
        if idx < 0:
            idx = hint
        start = idx
        end = idx + len(piece)
        found.append((start, end, piece))
        if overlap_size:
            hint = max(start + 1, end - overlap_size)
        else:
            hint = end
    return found


@register_chunker(
    name="recursive_character",
    category="text",
    description="Split on paragraphs, then lines, then spaces, then characters, merging up to chunk_size",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "html", "json", "csv", "rtf"],
    complexity=ComplexityLevel.LOW,
    dependencies=[],
    optional_dependencies=[],
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    quality=0.75,
    parameters_schema={
        "chunk_size": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100000,
            "default": 1000,
            "description": "Max Unicode scalars per chunk",
        },
        "overlap_size": {
            "type": "integer",
            "minimum": 0,
            "maximum": 50000,
            "default": 0,
            "description": "Overlap in Unicode scalars when merging",
        },
        "separators": {
            "type": "array",
            "items": {"type": "string"},
            "default": ["\n\n", "\n", " ", ""],
            "description": "Tried biggest-first; last should be empty string",
        },
    },
    default_parameters={
        "chunk_size": 1000,
        "overlap_size": 0,
        "separators": ["\n\n", "\n", " ", ""],
    },
    use_cases=["RAG", "generic_text", "langchain_parity"],
    best_for=["prose", "markdown without relying on headers", "default RAG split"],
    limitations=["character sized, not tokens, unless packed later"],
    streaming_support=True,
    adaptive_support=False,
    hierarchical_support=False,
)
class RecursiveCharacterChunker(StreamableChunker):
    """Separator cascade. Distinct from hierarchical ``recursive``."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_size: int = 0,
        separators: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("name", None)
        super().__init__(
            name="recursive_character",
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs,
        )
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if overlap_size < 0:
            raise ValueError("overlap_size must be >= 0")
        if overlap_size >= chunk_size:
            overlap_size = max(0, chunk_size - 1)
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.separators = tuple(separators) if separators is not None else DEFAULT_SEPARATORS

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChunkingResult:
        started = time.time()
        if isinstance(content, Path):
            text = content.read_text(encoding="utf-8")
            source = str(content)
        elif isinstance(content, bytes):
            text = content.decode("utf-8")
            source = (source_info or {}).get("source", "bytes")
        else:
            text = content
            source = (source_info or {}).get("source", "string")

        pieces = split_recursive_character(
            text, self.chunk_size, self.overlap_size, self.separators
        )
        located = locate_pieces(text, pieces, self.overlap_size)
        chunks = []
        for i, (start, end, piece) in enumerate(located):
            meta = ChunkMetadata(
                source=source,
                chunker_used="recursive_character",
                offset=start,
                length=end - start,
                extra={
                    "offset_unit": "char",
                    "chunk_size": self.chunk_size,
                    "overlap_size": self.overlap_size,
                },
            )
            chunks.append(
                Chunk(
                    id=f"recursive_character_{i}",
                    content=piece,
                    modality=ModalityType.TEXT,
                    metadata=meta,
                    hash=sha256_text(piece),
                    start=start,
                    end=end,
                    size=len(piece),
                )
            )
        return ChunkingResult(
            chunks=chunks,
            processing_time=time.time() - started,
            strategy_used="recursive_character",
            source_info={"source": source},
        )

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Chunk]:
        parts: List[str] = []
        for piece in content_stream:
            if isinstance(piece, bytes):
                parts.append(piece.decode("utf-8"))
            else:
                parts.append(piece)
        yield from self.chunk("".join(parts), source_info=source_info).chunks
