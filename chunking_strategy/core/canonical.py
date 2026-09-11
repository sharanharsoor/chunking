"""Canonical chunking.v1 JSON. Fixtures and the lab share this shape."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Union

from chunking_strategy.core.base import Chunk, ChunkMetadata, ChunkingResult, ModalityType

SCHEMA_VERSION = "chunking.v1"
PROMOTED_EXTRA = (
    "offset_unit",
    "line_start",
    "line_end",
    "breadcrumb",
    "header_path",
    "symbol_name",
    "symbol_kind",
    "sentence_spec",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    return obj


def _content_fields(chunk: Chunk) -> Dict[str, Any]:
    content = chunk.content
    if isinstance(content, bytes):
        return {"content_sha256": sha256_bytes(content)}
    if content is None:
        return {}
    text = content if isinstance(content, str) else str(content)
    return {"content": text, "hash": chunk.hash or sha256_text(text)}


def _start_end(chunk: Chunk) -> tuple:
    start = getattr(chunk, "start", None)
    end = getattr(chunk, "end", None)
    if start is None and chunk.metadata.offset is not None:
        start = chunk.metadata.offset
    if end is None and start is not None and chunk.metadata.length is not None:
        end = start + chunk.metadata.length
    extra = chunk.metadata.extra or {}
    if start is None:
        start = extra.get("start")
    if end is None:
        end = extra.get("end")
    return start, end


def chunk_to_canonical(chunk: Chunk, index: int, fixture: bool) -> Dict[str, Any]:
    extra = dict(chunk.metadata.extra or {})
    start, end = _start_end(chunk)
    offset_unit = extra.pop("offset_unit", None)
    if offset_unit is None and start is not None:
        offset_unit = "char"

    meta: Dict[str, Any] = {
        "source": chunk.metadata.source,
        "chunker_used": chunk.metadata.chunker_used or extra.get("chunker_used"),
    }
    if chunk.metadata.page is not None:
        meta["page"] = chunk.metadata.page
    if offset_unit is not None:
        meta["offset_unit"] = offset_unit

    for key in PROMOTED_EXTRA:
        if key == "offset_unit":
            continue
        if key in extra and extra[key] is not None:
            meta[key] = extra.pop(key)

    leftover = {k: v for k, v in extra.items() if k not in ("start", "end", "processing_time")}
    if fixture:
        leftover.pop("quality_score", None)
        leftover.pop("parent_id", None)
    if leftover:
        meta["extra"] = leftover

    chunk_id = f"chunk-{index:04d}" if fixture else chunk.id
    size = chunk.size
    if size is None and isinstance(chunk.content, (str, bytes)):
        size = len(chunk.content)

    out: Dict[str, Any] = {
        "id": chunk_id,
        "modality": chunk.modality.value if hasattr(chunk.modality, "value") else chunk.modality,
        "metadata": meta,
        "parent_id": chunk.parent_id,
        "children_ids": list(chunk.children_ids or []),
        "start": start,
        "end": end,
        "size": size,
    }
    token_count = getattr(chunk, "token_count", None)
    if token_count is not None:
        out["token_count"] = token_count
    out.update(_content_fields(chunk))
    return _round_floats(out)


def result_to_canonical_dict(
    result: ChunkingResult,
    *,
    fixture: bool = False,
    params: Optional[Dict[str, Any]] = None,
    source: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    src = dict(source or {})
    if result.source_info:
        src.setdefault("name", result.source_info.get("source") or result.source_info.get("name"))
        if "size" in result.source_info and "size" not in src:
            src["size"] = result.source_info["size"]
        if "sha256" in result.source_info and "sha256" not in src:
            src["sha256"] = result.source_info["sha256"]

    doc = {
        "schema": SCHEMA_VERSION,
        "strategy": result.strategy_used,
        "params": params if params is not None else {},
        "source": src,
        "chunks": [chunk_to_canonical(c, i, fixture) for i, c in enumerate(result.chunks)],
    }
    if not fixture and result.quality_score is not None:
        doc["quality_score"] = result.quality_score
    if fixture:
        id_map = {c.id: f"chunk-{i:04d}" for i, c in enumerate(result.chunks)}
        for row in doc["chunks"]:
            parent = row.get("parent_id")
            if parent in id_map:
                row["parent_id"] = id_map[parent]
            row["children_ids"] = [
                id_map[cid] for cid in (row.get("children_ids") or []) if cid in id_map
            ]
    return _round_floats(doc)


def result_to_canonical_json(
    result: ChunkingResult,
    *,
    fixture: bool = False,
    params: Optional[Dict[str, Any]] = None,
    source: Optional[Dict[str, Any]] = None,
) -> str:
    doc = result_to_canonical_dict(result, fixture=fixture, params=params, source=source)
    return json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def result_from_canonical_json(payload: Union[str, bytes, Dict[str, Any]]) -> ChunkingResult:
    if isinstance(payload, dict):
        doc = payload
    else:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        doc = json.loads(payload)

    chunks: List[Chunk] = []
    for item in doc.get("chunks") or []:
        raw_meta = dict(item.get("metadata") or {})
        extra = dict(raw_meta.pop("extra", None) or {})
        for key in PROMOTED_EXTRA:
            if key in raw_meta:
                extra[key] = raw_meta.pop(key)
        start = item.get("start")
        end = item.get("end")
        known = {
            "source",
            "source_type",
            "page",
            "position",
            "offset",
            "length",
            "timestamp",
            "frame_range",
            "bbox",
            "coordinates",
            "speaker",
            "language",
            "encoding",
            "mime_type",
            "chunker_used",
            "processing_time",
            "confidence",
            "quality_score",
        }
        meta_kwargs = {k: v for k, v in raw_meta.items() if k in known}
        if start is not None:
            meta_kwargs.setdefault("offset", start)
        if start is not None and end is not None:
            meta_kwargs.setdefault("length", end - start)
        meta_kwargs["extra"] = extra
        meta_kwargs.setdefault("source", "unknown")
        content = item.get("content")
        chunk = Chunk(
            id=item["id"],
            content="" if content is None else content,
            modality=ModalityType(item.get("modality", "text")),
            metadata=ChunkMetadata(**meta_kwargs),
            size=item.get("size"),
            hash=item.get("hash"),
            parent_id=item.get("parent_id"),
            children_ids=item.get("children_ids") or [],
            start=start,
            end=end,
            token_count=item.get("token_count"),
        )
        chunks.append(chunk)

    return ChunkingResult(
        chunks=chunks,
        strategy_used=doc.get("strategy"),
        source_info=doc.get("source"),
    )
