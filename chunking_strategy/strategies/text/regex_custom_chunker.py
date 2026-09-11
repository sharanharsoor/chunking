"""Split text at each regex match. The delimiter stays with the following chunk.

Not a parser. Useful for speaker turns, log lines, or custom marks.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

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


def split_regex_custom(
    text: str,
    pattern: str,
    multiline: bool = True,
) -> List[tuple[int, int, str]]:
    if not text:
        return []
    if not pattern:
        return [(0, len(text), text)]
    flags = re.MULTILINE if multiline else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc
    starts = [m.start() for m in rx.finditer(text)]
    if not starts:
        return [(0, len(text), text)]
    if starts[0] != 0:
        starts = [0] + starts
    uniq: List[int] = []
    for pos in starts:
        if not uniq or pos > uniq[-1]:
            uniq.append(pos)
    out: List[tuple[int, int, str]] = []
    for i, start in enumerate(uniq):
        end = uniq[i + 1] if i + 1 < len(uniq) else len(text)
        if end > start:
            out.append((start, end, text[start:end]))
    return out


@register_chunker(
    name="regex_custom",
    category="text",
    description="Split at each regex match; the match starts the next chunk",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "log", "csv", "html"],
    complexity=ComplexityLevel.LOW,
    dependencies=[],
    optional_dependencies=[],
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    quality=0.7,
    parameters_schema={
        "pattern": {
            "type": "string",
            "default": "\\n\\n",
            "description": "Regex; each match starts a new chunk",
        },
        "multiline": {
            "type": "boolean",
            "default": True,
            "description": "re.MULTILINE so ^ and $ match line edges",
        },
    },
    default_parameters={"pattern": "\\n\\n", "multiline": True},
    use_cases=["transcripts", "logs", "custom_delimiters"],
    best_for=["speaker turns", "dated log lines", "user-defined cuts"],
    limitations=["catastrophic regex can hang; JS and Python regex dialects differ"],
    streaming_support=True,
    adaptive_support=False,
    hierarchical_support=False,
)
class RegexCustomChunker(StreamableChunker):
    """Split on a user regex. Distinct from sentence/paragraph heuristics."""

    def __init__(
        self,
        pattern: str = "\\n\\n",
        multiline: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("name", None)
        super().__init__(
            name="regex_custom",
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs,
        )
        self.pattern = pattern if pattern is not None else ""
        self.multiline = bool(multiline)

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

        pieces = split_regex_custom(text, self.pattern, self.multiline)
        chunks = []
        for i, (start, end, piece) in enumerate(pieces):
            meta = ChunkMetadata(
                source=source,
                chunker_used="regex_custom",
                offset=start,
                length=end - start,
                extra={
                    "offset_unit": "char",
                    "pattern": self.pattern,
                    "multiline": self.multiline,
                },
            )
            chunks.append(
                Chunk(
                    id=f"regex_custom_{i}",
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
            strategy_used="regex_custom",
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
