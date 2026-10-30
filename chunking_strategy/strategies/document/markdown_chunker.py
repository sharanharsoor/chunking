"""
Markdown-specific chunking strategy with header-based hierarchy support.

This module provides specialized chunking for Markdown documents, supporting:
- Header-based hierarchical chunking (H1, H2, H3, etc.)
- Section-based chunking with content preservation
- Code block aware chunking (preserves code blocks intact)
- List and table aware chunking
- Metadata preservation (frontmatter, links, etc.)
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.core.streaming import StreamableChunker
from chunking_strategy.core.adaptive import AdaptableChunker

_GFM_ROW = re.compile(r"^\s*\|.*\|\s*$")
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_SEP_CELL = re.compile(r":?-{3,}:?")
_SETEXT = re.compile(r"^(?P<title>.+)\n(?P<underline>=+|-+)[ \t]*$", re.MULTILINE)


def _is_gfm_separator(line: str) -> bool:
    if not _GFM_ROW.match(line):
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(_SEP_CELL.fullmatch(cell) for cell in cells)


def mask_fences(text: str) -> str:
    """Replace fenced-code interiors with spaces so `#` inside fences is not a header."""
    chars = list(text)
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("```", i) or text.startswith("~~~", i):
            delim = text[i : i + 3]
            nl = text.find("\n", i)
            if nl < 0:
                break
            close = text.find("\n" + delim, nl)
            end = n if close < 0 else close + 1 + len(delim)
            for p in range(i, min(end, n)):
                if chars[p] != "\n":
                    chars[p] = " "
            i = end
            continue
        i += 1
    return "".join(chars)


def atomic_spans(text: str) -> List[Tuple[int, int, str]]:
    """Non-overlapping spans: fenced code, HTML tables, GFM tables, then text."""
    if not text:
        return []
    lines: List[Tuple[int, int, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        lines.append((pos, pos + len(line), line))
        pos += len(line)
    if pos < len(text):
        lines.append((pos, len(text), text[pos:]))
    spans: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        start, end, line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            delim = stripped[:3]
            j = i + 1
            while j < len(lines) and not lines[j][2].lstrip().startswith(delim):
                j += 1
            if j < len(lines):
                j += 1
            spans.append((start, lines[j - 1][1], "code"))
            i = j
            continue
        if re.search(r"<table\b", line, re.I):
            j = i
            while j < len(lines) and not re.search(r"</table>", lines[j][2], re.I):
                j += 1
            if j < len(lines):
                j += 1
            spans.append((start, lines[max(i, j - 1)][1], "html_table"))
            i = max(j, i + 1)
            continue
        if _GFM_ROW.match(line.rstrip("\n")):
            j = i + 1
            while j < len(lines) and _GFM_ROW.match(lines[j][2].rstrip("\n")):
                j += 1
            spans.append((start, lines[j - 1][1], "gfm_table"))
            i = j
            continue
        j = i + 1
        while j < len(lines):
            nxt = lines[j][2]
            if not nxt.strip():
                j += 1
                break
            ns = nxt.lstrip()
            if ns.startswith("```") or ns.startswith("~~~") or re.search(r"<table\b", nxt, re.I) or _GFM_ROW.match(nxt.rstrip("\n")):
                break
            j += 1
        spans.append((start, lines[j - 1][1], "text"))
        i = j
    return spans


@register_chunker(
    name="markdown_chunker",
    category="document",
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    supported_formats=[".md", ".markdown", ".mdown", ".mkd", ".rst"],
    dependencies=[],
    description="Markdown chunker with header-based hierarchy and structure preservation",
    use_cases=["documentation", "blog_posts", "technical_writing", "knowledge_bases"],
    best_for=["structured_documents", "hierarchical_content", "technical_documentation"],
    limitations=["requires_well_structured_markdown"]
)
class MarkdownChunker(StreamableChunker, AdaptableChunker):
    """
    Specialized chunker for Markdown documents supporting:
    - Header-based hierarchical chunking (H1, H2, H3, etc.)
    - Section-based chunking with content preservation
    - Code block aware chunking (preserves code blocks intact)
    - List and table structure preservation
    - Frontmatter extraction and handling
    """

    def __init__(
        self,
        chunk_by: str = "headers",  # "headers", "sections", "content_blocks", "fixed_size"
        header_level: int = 2,  # Max header level to split on (1=H1, 2=H2, etc.)
        preserve_code_blocks: bool = True,
        preserve_tables: bool = True,
        preserve_lists: bool = True,
        include_frontmatter: bool = True,
        chunk_size: int = 2000,  # For fixed_size mode
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int = 8000,
        breadcrumb: bool = True,
        contextualize: bool = False,
        max_tokens: Optional[int] = None,
        overlap_tokens: int = 0,
        encoding: str = "cl100k_base",
        parent_chars: int = 7000,
        **kwargs
    ):
        """
        Initialize Markdown chunker.

        Args:
            chunk_by: Chunking strategy ("headers", "sections", "content_blocks", "fixed_size")
            header_level: Maximum header level to split on (1-6)
            preserve_code_blocks: Keep code blocks intact
            preserve_tables: Keep tables intact
            preserve_lists: Keep lists intact
            include_frontmatter: Include frontmatter in first chunk
            chunk_size: Target chunk size for fixed_size mode
            chunk_overlap: Overlap between chunks
            min_chunk_size: Sections shorter than this are merged into a neighbor, not dropped
            max_chunk_size: Maximum chunk size before forced split
            **kwargs: Additional parameters
        """
        super().__init__(
            name="markdown_chunker",
            category="document",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        self.chunk_by = chunk_by
        self.header_level = max(1, min(6, header_level))
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_tables = preserve_tables
        self.preserve_lists = preserve_lists
        self.include_frontmatter = include_frontmatter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.breadcrumb = breadcrumb
        self.contextualize = contextualize
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = encoding
        # Returned text around a split table. The embedded chunk stays under max_tokens.
        self.parent_chars = parent_chars
        self._enc = None

        self.logger = logging.getLogger(__name__)

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for Markdown parsing."""
        # Header patterns
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

        # Code block patterns
        self.code_block_pattern = re.compile(
            r'```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]+`',
            re.MULTILINE | re.DOTALL
        )

        # Table patterns
        self.table_pattern = re.compile(
            r'^\|.*\|\s*$\n^\|.*\|\s*$(?:\n^\|.*\|\s*$)*',
            re.MULTILINE
        )

        # List patterns
        self.list_pattern = re.compile(
            r'^(\s*)([-*+]|\d+\.)\s+(.+)(?:\n(?:\1\s+.+|\s*$))*',
            re.MULTILINE
        )

        # Frontmatter pattern
        self.frontmatter_pattern = re.compile(
            r'^---\s*\n(.*?)\n---\s*\n',
            re.DOTALL
        )

    def _tiktoken(self):
        if self._enc is None:
            from chunking_strategy.core.token_packing import require_encoding
            self._enc = require_encoding(self.encoding)
        return self._enc

    def _collect_headers(self, content: str) -> List[Dict[str, Any]]:
        masked = mask_fences(content)
        headers: List[Dict[str, Any]] = []
        for match in self.header_pattern.finditer(masked):
            if match.group(0).strip() == "":
                continue
            headers.append({
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "start": match.start(),
                "end": match.end(),
            })
        for match in _SETEXT.finditer(masked):
            underline = match.group("underline")
            title = match.group("title").strip()
            if "|" in title or "|" in underline:
                continue
            if not title or title.startswith("#"):
                continue
            headers.append({
                "level": 1 if underline.startswith("=") else 2,
                "title": title,
                "start": match.start(),
                "end": match.end(),
            })
        headers.sort(key=lambda h: h["start"])
        return headers

    def _header_path(self, headers: List[Dict[str, Any]], at: Dict[str, Any]) -> List[str]:
        stack: List[Dict[str, Any]] = []
        for header in headers:
            if header["start"] > at["start"]:
                break
            while stack and stack[-1]["level"] >= header["level"]:
                stack.pop()
            stack.append(header)
        return [h["title"] for h in stack]

    def _decorate(self, body: str, header_path: List[str]) -> Tuple[str, Dict[str, Any]]:
        extra: Dict[str, Any] = {}
        if self.breadcrumb and header_path:
            extra["header_path"] = list(header_path)
            extra["breadcrumb"] = " > ".join(header_path)
        content = body
        if self.contextualize and extra.get("breadcrumb"):
            extra["raw_content"] = body
            content = "[{}]\n\n{}".format(extra["breadcrumb"], body)
        return content, extra

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk Markdown content using specified strategy.

        Args:
            content: Markdown content (string, bytes, or file path)
            source_info: Source information metadata
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with chunks and metadata
        """
        start_time = time.time()

        # Handle different input types
        if isinstance(content, Path):
            file_path = content
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            source_info = source_info or {}
            source_info.update({
                "source": str(file_path),
                "source_type": "file"
            })
        elif isinstance(content, bytes):
            md_content = content.decode('utf-8')
        elif isinstance(content, str):
            # A short single line may be a path. Longer text, or a name the OS
            # rejects, is content. macOS raises ENAMETOOLONG around 255 bytes.
            file_path = None
            if (len(content) < 255 and "\n" not in content and
                    content.strip() and not content.isspace()):
                try:
                    candidate = Path(content)
                    if candidate.is_file():
                        file_path = candidate
                except OSError:
                    file_path = None
            if file_path is not None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                source_info = source_info or {}
                source_info.update({
                    "source": str(file_path),
                    "source_type": "file"
                })
            else:
                md_content = content
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        source_info = source_info or {"source": "unknown", "source_type": "content"}

        # Handle empty or whitespace-only content
        if not md_content or not md_content.strip():
            processing_time = time.time() - start_time
            return ChunkingResult(
                chunks=[],
                processing_time=processing_time,
                source_info={
                    "markdown_structure": {"total_headers": 0, "header_distribution": {}, "code_blocks": 0, "tables": 0, "structure_type": "empty"},
                    "frontmatter_present": False,
                    "chunking_method": self.chunk_by,
                    "header_level_limit": self.header_level,
                    **source_info
                },
                strategy_used="markdown_chunker"
            )

        # Extract frontmatter if present
        frontmatter, md_content = self._extract_frontmatter(md_content)

        # Choose chunking strategy
        if self.chunk_by == "headers":
            chunks = self._chunk_by_headers(md_content, frontmatter, source_info)
        elif self.chunk_by == "sections":
            chunks = self._chunk_by_sections(md_content, frontmatter, source_info)
        elif self.chunk_by == "content_blocks":
            chunks = self._chunk_by_content_blocks(md_content, frontmatter, source_info)
        elif self.chunk_by == "fixed_size":
            chunks = self._chunk_by_fixed_size(md_content, frontmatter, source_info)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunk_by}")

        processing_time = time.time() - start_time

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            source_info={
                "markdown_structure": self._analyze_structure(md_content),
                "frontmatter_present": frontmatter is not None,
                "chunking_method": self.chunk_by,
                "header_level_limit": self.header_level,
                **source_info
            },
            strategy_used="markdown_chunker"
        )

    def _extract_frontmatter(self, content: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Extract YAML frontmatter from Markdown content."""
        match = self.frontmatter_pattern.match(content)
        if match:
            frontmatter_text = match.group(1)
            content_without_frontmatter = content[match.end():]

            # Parse YAML frontmatter
            try:
                import yaml
                frontmatter_data = yaml.safe_load(frontmatter_text)
                return frontmatter_data, content_without_frontmatter
            except ImportError:
                self.logger.warning("PyYAML not available, frontmatter will be treated as text")
                return {"raw": frontmatter_text}, content_without_frontmatter
            except Exception as e:
                self.logger.warning(f"Failed to parse frontmatter: {e}")
                return {"raw": frontmatter_text}, content_without_frontmatter

        return None, content

    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structure of the Markdown document."""
        headers = self.header_pattern.findall(content)
        code_blocks = self.code_block_pattern.findall(content)
        tables = self.table_pattern.findall(content)

        header_counts = {}
        for level, _ in headers:
            header_counts[f"h{len(level)}"] = header_counts.get(f"h{len(level)}", 0) + 1

        return {
            "total_headers": len(headers),
            "header_distribution": header_counts,
            "code_blocks": len(code_blocks),
            "tables": len(tables),
            "structure_type": "hierarchical" if headers else "flat"
        }

    def _chunk_by_headers(
        self,
        content: str,
        frontmatter: Optional[Dict[str, Any]],
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by header hierarchy.

        A heading is a place we are allowed to cut, not a chunk by itself.
        With a token budget, adjacent sections pack until the next one would
        not fit. Without one, a section shorter than min_chunk_size merges
        into its neighbor. Short sections used to be dropped.
        """
        all_headers = self._collect_headers(content)
        headers = [h for h in all_headers if h["level"] <= self.header_level]

        if not headers:
            return self._finish_chunks(
                self._create_single_chunk(content, frontmatter, source_info, 0)
            )

        sections = self._header_sections(content, headers, all_headers, frontmatter)
        if not sections:
            return []
        if self.max_tokens is not None:
            sections = self._pack_header_sections(sections)
        else:
            sections = self._merge_short_header_sections(sections)

        chunks: List[Chunk] = []
        for i, section in enumerate(sections):
            chunks.extend(self._emit_block(
                section["text"],
                i,
                title=section["title"],
                level=section["level"],
                frontmatter=section["frontmatter"] if i == 0 else None,
                source_info=source_info,
                header_path=section["header_path"],
            ))
        return self._finish_chunks(chunks)

    def _header_sections(
        self,
        content: str,
        headers: List[Dict[str, Any]],
        all_headers: List[Dict[str, Any]],
        frontmatter: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """One record per heading section, plus any lead-in before the first."""
        sections: List[Dict[str, Any]] = []
        preamble = content[: headers[0]["start"]].strip()
        if preamble:
            sections.append({
                "text": preamble,
                "title": "",
                "level": 0,
                "header_path": [],
                "frontmatter": frontmatter if self.include_frontmatter else None,
            })
        for i, header in enumerate(headers):
            chunk_end = headers[i + 1]["start"] if i + 1 < len(headers) else len(content)
            text = content[header["start"]:chunk_end].strip()
            if not text:
                continue
            sections.append({
                "text": text,
                "title": header["title"],
                "level": header["level"],
                "header_path": self._header_path(all_headers, header),
                "frontmatter": None,
            })
        if sections and frontmatter and self.include_frontmatter and sections[0]["frontmatter"] is None:
            sections[0]["frontmatter"] = frontmatter
        return sections

    def _join_section_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        head = group[0]
        if len(group) == 1:
            return head
        text = "\n\n".join(section["text"].strip() for section in group if section["text"].strip())
        return {
            "text": text,
            "title": head["title"],
            "level": head["level"],
            "header_path": head["header_path"],
            "frontmatter": head["frontmatter"],
        }

    def _merge_short_header_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge a section under min_chunk_size into the next one, or the previous if it is last."""
        merged: List[Dict[str, Any]] = []
        buf = sections[0]
        for section in sections[1:]:
            if len(buf["text"]) < self.min_chunk_size:
                buf = self._join_section_group([buf, section])
            else:
                merged.append(buf)
                buf = section
        if merged and len(buf["text"]) < self.min_chunk_size:
            merged[-1] = self._join_section_group([merged[-1], buf])
        else:
            merged.append(buf)
        return merged

    _FOOTER_HEADINGS = (
        "about the author",
        "more like this",
        "product trial",
        "related articles",
        "you might also like",
    )

    def _pack_header_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add whole sections until the token cap is full.

        A heading starts a new chunk only when that section would pass
        max_tokens. A heading with no body still joins a neighbor. A section
        larger than the cap is emitted alone and split later. Stopping at
        every heading left chunks around 1,400 characters while the cap
        still allowed about 5,500.
        """
        from chunking_strategy.core.token_packing import count_tokens, pack_unit_ranges

        sections = self._stick_short_sections(sections)
        enc = self._tiktoken()
        sep = count_tokens("\n\n", enc)

        def tokens_of(parts: List[Dict[str, Any]]) -> int:
            if not parts:
                return 0
            total = sum(count_tokens(part["text"], enc) for part in parts)
            if len(parts) > 1:
                total += sep * (len(parts) - 1)
            return total

        # A bare heading prefixes the next section even when that section is
        # over the cap. The splitter keeps it on the first piece.
        units: List[Dict[str, Any]] = []
        frags: List[Dict[str, Any]] = []
        for section in sections:
            if len(section["text"].strip()) < 250:
                frags.append(section)
                continue
            if frags:
                units.append(self._join_section_group(frags + [section]))
                frags = []
            else:
                units.append(section)
        if frags and units and tokens_of([units[-1]] + frags) <= self.max_tokens:
            units[-1] = self._join_section_group([units[-1]] + frags)
            frags = []
        elif frags:
            frag_sizes = [
                count_tokens(part["text"], enc) + (sep if i else 0)
                for i, part in enumerate(frags)
            ]
            for start, end in pack_unit_ranges(frag_sizes, self.max_tokens, overlap_tokens=0):
                units.append(self._join_section_group(frags[start:end]))
        sizes = [
            count_tokens(unit["text"], enc) + (sep if i else 0)
            for i, unit in enumerate(units)
        ]
        ranges = pack_unit_ranges(sizes, self.max_tokens, overlap_tokens=0)
        groups = [self._join_section_group(units[start:end]) for start, end in ranges]
        return self._attach_footer_groups(groups)

    def _heading_text(self, text: str) -> str:
        line = ""
        for raw in text.splitlines():
            if raw.strip():
                line = raw.strip()
                break
        return line.lstrip("#").strip().lower()

    def _is_footer(self, text: str) -> bool:
        heading = self._heading_text(text)
        return any(heading == name or heading.startswith(name + " ") for name in self._FOOTER_HEADINGS)

    def _attach_footer_groups(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep a page footer on the previous chunk.

        "About the author" and "More like this" were their own search hits.
        Join the footer when it still fits. When the previous chunk is
        already at the cap, drop the footer instead of giving it a vector.
        """
        from chunking_strategy.core.token_packing import count_tokens

        if self.max_tokens is None:
            return groups
        enc = self._tiktoken()
        kept: List[Dict[str, Any]] = []
        for group in groups:
            if not self._is_footer(group["text"]):
                kept.append(group)
                continue
            if not kept:
                continue
            joined = self._join_section_group([kept[-1], group])
            if count_tokens(joined["text"], enc) <= self.max_tokens:
                kept[-1] = joined
        return kept

    def _stick_short_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prefix a short section onto the next one when they cannot pack together.

        A heading followed by a body larger than max_tokens was left as its own
        vector. The body is split later, so the heading belongs on that first piece.
        """
        from chunking_strategy.core.token_packing import count_tokens

        if self.max_tokens is None or len(sections) < 2:
            return sections
        enc = self._tiktoken()
        stuck: List[Dict[str, Any]] = []
        i = 0
        while i < len(sections):
            size = count_tokens(sections[i]["text"], enc)
            if i + 1 < len(sections):
                nxt = count_tokens(sections[i + 1]["text"], enc)
                # Only a heading-sized fragment prefixes an oversized body.
                # A finished section under max_tokens//3 used to be glued on too.
                if len(sections[i]["text"].strip()) < 250 and size + nxt > self.max_tokens:
                    sections[i + 1] = self._join_section_group([sections[i], sections[i + 1]])
                    i += 1
                    continue
            stuck.append(sections[i])
            i += 1
        return stuck

    def _finish_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        if self.max_tokens is None:
            return chunks
        chunks = self._merge_short_emitted(chunks)
        chunks = self._pack_fitting_neighbors(chunks)
        chunks = self._rebalance_short_chunks(chunks)
        return self._assign_table_parents(self._clamp_over(chunks))

    def _assign_table_parents(self, chunks: List[Chunk]) -> List[Chunk]:
        """Give each split table piece a window of neighboring rows.

        The piece itself stays under the embed cap. The window is what a
        search hit should return, sized like the old 7,000-character cut.
        A table that already fits has no separate parent.
        """
        if self.parent_chars <= 0 or len(chunks) < 2:
            return chunks
        keys: List[str] = []
        last = ""
        for chunk in chunks:
            text = chunk.content or ""
            key = self._table_group_key(text)
            if key:
                last = key
            elif last and any(_GFM_ROW.match(line) for line in text.splitlines()):
                key = last
            else:
                last = ""
            keys.append(key)
        start = 0
        while start < len(chunks):
            end = start + 1
            while end < len(chunks) and keys[end] and keys[end] == keys[start]:
                end += 1
            if keys[start] and end - start > 1:
                bodies = [chunks[i].content or "" for i in range(start, end)]
                for offset in range(end - start):
                    parent = self._parent_window(bodies, offset)
                    chunk = chunks[start + offset]
                    if (
                        chunk.metadata is not None
                        and len(parent) > len((chunk.content or "").strip()) + 40
                    ):
                        chunk.metadata.extra["parent"] = parent
            start = end
        return chunks

    def _table_group_key(self, text: str) -> str:
        if not any(_GFM_ROW.match(line) for line in text.splitlines()):
            return ""
        heading = ""
        for line in text.splitlines():
            if _HEADING_LINE.match(line):
                heading = line.strip()
        return heading

    def _parent_window(self, bodies: List[str], index: int) -> str:
        lo = hi = index
        current = self._join_table_pieces(bodies[lo:hi + 1])
        while len(current) < self.parent_chars:
            left = lo > 0
            right = hi + 1 < len(bodies)
            if not left and not right:
                break
            if right and (not left or (hi - index) <= (index - lo)):
                nlo, nhi = lo, hi + 1
            else:
                nlo, nhi = lo - 1, hi
            nxt = self._join_table_pieces(bodies[nlo:nhi + 1])
            # ponytail: stop if a sibling adds no new rows. Upgrade path: slice the
            # original table by row index instead of joining emitted pieces.
            if len(nxt) <= len(current):
                break
            lo, hi, current = nlo, nhi, nxt
        return current

    def _join_table_pieces(self, pieces: List[str]) -> str:
        """One heading and one header, then the data rows from each piece."""
        if not pieces:
            return ""
        preamble: List[str] = []
        header = ""
        separator = ""
        for line in pieces[0].splitlines():
            if _GFM_ROW.match(line):
                header = line
                break
            if line.strip():
                preamble.append(line.strip())
        if header:
            lines = pieces[0].splitlines()
            try:
                nxt = lines[lines.index(header) + 1]
            except (ValueError, IndexError):
                nxt = ""
            if nxt and _is_gfm_separator(nxt):
                separator = nxt
        skip = set(preamble)
        if header:
            skip.add(header.strip())
        if separator:
            skip.add(separator.strip())
        rows: List[str] = []
        for piece in pieces:
            for line in piece.splitlines():
                stripped = line.strip()
                if not stripped or stripped in skip or _HEADING_LINE.match(line):
                    continue
                rows.append(line)
        head = "\n".join(preamble)
        table_lines = [line for line in (header, separator) if line]
        table_lines.extend(rows)
        table = "\n".join(table_lines)
        if head and table:
            return f"{head}\n\n{table}"
        return head or table

    def _merge_short_emitted(self, chunks: List[Chunk]) -> List[Chunk]:
        """Keep a chunk under 250 characters from being its own vector.

        Join it to a neighbor when the two still fit. When the previous window
        is already full, re-cut that pair with overlap so the short text rides
        on the tail instead of standing alone.
        """
        from chunking_strategy.core.token_packing import count_tokens

        if len(chunks) < 2:
            return self._clamp_over(chunks)
        enc = self._tiktoken()
        out = list(chunks)
        toks = [count_tokens(chunk.content or "", enc) for chunk in out]
        i = 0
        while i < len(out):
            text = (out[i].content or "").strip()
            if len(text) >= 250:
                i += 1
                continue
            if i > 0 and toks[i - 1] + toks[i] <= self.max_tokens:
                merged = self._concat_chunks(out[i - 1], out[i])
                merged_tok = count_tokens(merged.content, enc)
                if merged_tok <= self.max_tokens:
                    out[i - 1] = merged
                    toks[i - 1] = merged_tok
                    out.pop(i)
                    toks.pop(i)
                    continue
            if i + 1 < len(out) and toks[i] + toks[i + 1] <= self.max_tokens:
                merged = self._concat_chunks(out[i], out[i + 1])
                merged_tok = count_tokens(merged.content, enc)
                if merged_tok <= self.max_tokens:
                    out[i] = merged
                    toks[i] = merged_tok
                    out.pop(i + 1)
                    toks.pop(i + 1)
                    continue
            if i > 0 and self._resplit_short_tail(out, toks, i, enc):
                continue
            i += 1
        return self._clamp_over(out)

    def _clamp_over(self, chunks: List[Chunk]) -> List[Chunk]:
        from chunking_strategy.core.token_packing import count_tokens

        if self.max_tokens is None:
            return chunks
        enc = self._tiktoken()
        clamped: List[Chunk] = []
        for chunk in chunks:
            if count_tokens(chunk.content or "", enc) <= self.max_tokens:
                clamped.append(chunk)
                continue
            for j, piece in enumerate(self._cut_to_fit(chunk.content or "")):
                piece = piece.strip()
                if piece:
                    clamped.append(self._with_content(chunk, piece, f"c{j}"))
        return clamped

    def _resplit_short_tail(self, out: List[Chunk], toks: List[int], i: int, enc) -> bool:
        from chunking_strategy.core.token_packing import count_tokens

        combined = (out[i - 1].content or "").rstrip() + "\n\n" + (out[i].content or "").lstrip()
        pieces = [piece.strip() for piece in self._cut_to_fit(combined) if piece and piece.strip()]
        if len(pieces) < 2 or len(pieces[-1]) < 250:
            return False
        counts = [count_tokens(piece, enc) for piece in pieces]
        if any(count > self.max_tokens for count in counts):
            return False
        made = [self._with_content(out[i - 1], piece, str(j)) for j, piece in enumerate(pieces)]
        out[i - 1:i + 1] = made
        toks[i - 1:i + 1] = counts
        return True

    def _with_content(self, template: Chunk, text: str, suffix: str) -> Chunk:
        meta = template.metadata
        return Chunk(
            id=f"{template.id}_{suffix}",
            content=text,
            metadata=ChunkMetadata(
                source=meta.source if meta else "unknown",
                source_type=meta.source_type if meta else "content",
                position=meta.position if meta else None,
                length=len(text),
                extra=dict(meta.extra) if meta and meta.extra else {},
            ),
            modality=template.modality,
        )

    def _concat_chunks(self, left: Chunk, right: Chunk) -> Chunk:
        text = (left.content or "").rstrip() + "\n\n" + (right.content or "").lstrip()
        left.content = text
        left.size = len(text)
        if left.metadata is not None:
            left.metadata.length = len(text)
        return left

    def _pack_fitting_neighbors(self, chunks: List[Chunk]) -> List[Chunk]:
        """Join adjacent chunks that still fit. A heading starts a new chunk only past the cap."""
        from chunking_strategy.core.token_packing import count_tokens

        if len(chunks) < 2 or self.max_tokens is None:
            return chunks
        enc = self._tiktoken()
        out: List[Chunk] = [chunks[0]]
        for chunk in chunks[1:]:
            text = (out[-1].content or "").rstrip() + "\n\n" + (chunk.content or "").lstrip()
            if len(text) <= self.max_chunk_size and count_tokens(text, enc) <= self.max_tokens:
                out[-1] = self._with_content(out[-1], text, "p")
            else:
                out.append(chunk)
        return out

    def _rebalance_short_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """A tail stuck between two full chunks borrows lines from the previous one.

        Both sides stay inside the token cap, and at least a third of the cap.
        The tail keeps that chunk's heading. A tail that opens a different
        section is left alone. ponytail: whole lines only. A single line wider
        than the gap stays put.
        """
        from chunking_strategy.core.token_packing import count_tokens

        if len(chunks) < 2 or self.max_tokens is None:
            return chunks
        enc = self._tiktoken()
        floor = max(1, self.max_tokens // 3)
        out = list(chunks)
        for i in range(1, len(out)):
            cur_text = out[i].content or ""
            if count_tokens(cur_text, enc) >= floor:
                continue
            prev_text = out[i - 1].content or ""
            cur_heading = self._opening_heading(cur_text)
            prev_heading = self._opening_heading(prev_text)
            if cur_heading and cur_heading != prev_heading:
                continue
            new_prev, new_cur = self._shift_short_boundary(prev_text, cur_text, floor, enc)
            if new_cur.strip() == cur_text.strip():
                continue
            if self._over_limit(new_prev) or self._over_limit(new_cur):
                continue
            out[i - 1] = self._with_content(out[i - 1], new_prev, "b")
            out[i] = self._with_content(out[i], new_cur, "b")
        return out

    def _shift_short_boundary(self, prev: str, cur: str, floor: int, enc) -> Tuple[str, str]:
        from chunking_strategy.core.token_packing import count_tokens

        lead = self._lead_lines(prev)
        lines = prev.splitlines()
        keep = self._min_keep(lines, lead)
        cur_text = cur
        while count_tokens(cur_text, enc) < floor and len(lines) > keep:
            moved = lines[-1]
            if not moved.strip():
                lines.pop()
                continue
            candidate_prev = "\n".join(lines[:-1]).strip()
            candidate_cur = self._ensure_lead(f"{moved}\n{cur_text}", lead)
            if count_tokens(candidate_prev, enc) < floor:
                break
            if count_tokens(candidate_prev, enc) > self.max_tokens:
                break
            if count_tokens(candidate_cur, enc) > self.max_tokens:
                break
            lines.pop()
            cur_text = candidate_cur
        new_prev = "\n".join(lines).strip()
        new_cur = self._ensure_lead(cur_text, lead)
        if count_tokens(new_prev, enc) > self.max_tokens or count_tokens(new_cur, enc) > self.max_tokens:
            return prev, cur
        if count_tokens(new_cur, enc) <= count_tokens(cur, enc):
            return prev, cur
        return new_prev, new_cur

    def _opening_heading(self, text: str) -> str:
        for line in text.splitlines():
            if not line.strip():
                continue
            if _HEADING_LINE.match(line):
                return line.strip()
            return ""
        return ""

    def _nearest_heading(self, text: str) -> str:
        found = ""
        for line in text.splitlines():
            if _HEADING_LINE.match(line):
                found = line.strip()
        return found

    def _lead_lines(self, text: str) -> List[str]:
        """Heading at the start of a chunk, plus the table header when the body is a table."""
        lines = text.splitlines()
        i = 0
        lead: List[str] = []
        while i < len(lines) and not lines[i].strip():
            i += 1
        while i < len(lines) and _HEADING_LINE.match(lines[i]):
            lead.append(lines[i].strip())
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
        if i < len(lines) and _GFM_ROW.match(lines[i]):
            lead.append(lines[i].strip())
            i += 1
            if i < len(lines) and _is_gfm_separator(lines[i]):
                lead.append(lines[i].strip())
        return lead

    def _min_keep(self, lines: List[str], lead: List[str]) -> int:
        if not lead:
            return 1 if any(line.strip() for line in lines) else 0
        seen = 0
        for i, line in enumerate(lines):
            if seen < len(lead) and line.strip() == lead[seen]:
                seen += 1
                if seen == len(lead):
                    return i + 1
        return min(len(lines), len(lead))

    def _ensure_lead(self, text: str, lead: List[str]) -> str:
        if not lead:
            return text.strip()
        use = lead
        if not any(_GFM_ROW.match(line) for line in text.splitlines()):
            use = [line for line in lead if _HEADING_LINE.match(line)]
        present = {line.strip() for line in text.splitlines()}
        missing = [line for line in use if line not in present]
        if not missing:
            return text.strip()
        body = [line for line in text.splitlines()]
        return "\n".join(missing + [""] + body).strip()

    def _over_limit(self, text: str) -> bool:
        if len(text) > self.max_chunk_size:
            return True
        if self.max_tokens is None:
            return False
        from chunking_strategy.core.token_packing import count_tokens
        return count_tokens(text, self._tiktoken()) > self.max_tokens

    def _emit_block(
        self,
        text: str,
        index: int,
        title: str,
        level: int,
        frontmatter: Optional[Dict[str, Any]],
        source_info: Dict[str, Any],
        header_path: List[str],
    ) -> List[Chunk]:
        if self._over_limit(text):
            return self._split_large_chunk(
                text, index, header_path=header_path, source_info=source_info
            )
        return [self._create_chunk(
            text,
            index,
            title,
            level,
            frontmatter,
            source_info,
            header_path=header_path,
        )]

    def _chunk_by_sections(
        self,
        content: str,
        frontmatter: Optional[Dict[str, Any]],
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by logical sections (preserving structure blocks)."""
        chunks = []

        # Split content into logical sections
        sections = self._identify_sections(content)

        current_section = ""
        section_index = 0

        for section in sections:
            # Check if adding this section would exceed max size
            if (len(current_section) + len(section) > self.max_chunk_size and
                current_section.strip()):

                chunk = self._create_section_chunk(
                    current_section.strip(),
                    section_index,
                    frontmatter if section_index == 0 and self.include_frontmatter else None,
                    source_info
                )
                chunks.append(chunk)

                current_section = section
                section_index += 1
            else:
                current_section += "\n\n" + section if current_section else section

        # Add final section
        if current_section.strip():
            chunk = self._create_section_chunk(
                current_section.strip(),
                section_index,
                frontmatter if section_index == 0 and self.include_frontmatter else None,
                source_info
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_content_blocks(
        self,
        content: str,
        frontmatter: Optional[Dict[str, Any]],
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by content blocks (paragraphs, code, tables, lists)."""
        chunks = []
        blocks = self._parse_content_blocks(content)

        current_chunk = ""
        chunk_index = 0

        for block in blocks:
            # Check if adding this block would exceed target size
            if (len(current_chunk) + len(block['content']) > self.chunk_size and
                current_chunk.strip()):

                chunk = self._create_block_chunk(
                    current_chunk.strip(),
                    chunk_index,
                    frontmatter if chunk_index == 0 and self.include_frontmatter else None,
                    source_info
                )
                chunks.append(chunk)

                # Start new chunk with overlap if needed
                if self.chunk_overlap > 0:
                    overlap_content = self._get_overlap_content(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_content + "\n\n" + block['content']
                else:
                    current_chunk = block['content']
                chunk_index += 1
            else:
                current_chunk += "\n\n" + block['content'] if current_chunk else block['content']

        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_block_chunk(
                current_chunk.strip(),
                chunk_index,
                frontmatter if chunk_index == 0 and self.include_frontmatter else None,
                source_info
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_fixed_size(
        self,
        content: str,
        frontmatter: Optional[Dict[str, Any]],
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by fixed size with overlap."""
        chunks = []

        # Include frontmatter in first chunk if enabled
        if frontmatter and self.include_frontmatter:
            content = f"---\n{frontmatter.get('raw', '')}\n---\n\n{content}"

        words = content.split()
        current_chunk = ""
        chunk_index = 0

        for word in words:
            if len(current_chunk) + len(word) + 1 > self.chunk_size and current_chunk:
                chunk = self._create_fixed_chunk(
                    current_chunk.strip(),
                    chunk_index,
                    source_info
                )
                chunks.append(chunk)

                # Create overlap
                if self.chunk_overlap > 0:
                    overlap_words = current_chunk.split()[-self.chunk_overlap:]
                    current_chunk = " ".join(overlap_words) + " " + word
                else:
                    current_chunk = word
                chunk_index += 1
            else:
                current_chunk += " " + word if current_chunk else word

        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_fixed_chunk(
                current_chunk.strip(),
                chunk_index,
                source_info
            )
            chunks.append(chunk)

        return chunks

    def _identify_sections(self, content: str) -> List[str]:
        """Identify logical sections in content."""
        # Split by double newlines and identify section boundaries
        paragraphs = content.split('\n\n')
        sections = []
        current_section = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if this starts a new section (header)
            if self.header_pattern.match(paragraph):
                if current_section:
                    sections.append('\n\n'.join(current_section))
                    current_section = []

            current_section.append(paragraph)

        # Add final section
        if current_section:
            sections.append('\n\n'.join(current_section))

        return sections

    def _parse_content_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Parse content into logical blocks (paragraphs, code, tables, lists)."""
        blocks = []
        lines = content.split('\n')
        current_block = []
        block_type = "paragraph"
        in_code_block = False

        i = 0
        while i < len(lines):
            line = lines[i].rstrip()

            # Handle code blocks
            if line.startswith('```') or line.startswith('~~~'):
                if current_block and block_type != "code":
                    blocks.append({
                        'type': block_type,
                        'content': '\n'.join(current_block).strip()
                    })
                    current_block = []

                # Find end of code block
                code_delimiter = line[:3]
                current_block.append(line)
                i += 1
                while i < len(lines):
                    current_block.append(lines[i].rstrip())
                    if lines[i].strip().startswith(code_delimiter):
                        break
                    i += 1

                blocks.append({
                    'type': 'code',
                    'content': '\n'.join(current_block).strip()
                })
                current_block = []
                block_type = "paragraph"

            # Handle tables
            elif '|' in line and line.strip().startswith('|'):
                if current_block and block_type != "table":
                    blocks.append({
                        'type': block_type,
                        'content': '\n'.join(current_block).strip()
                    })
                    current_block = []

                # Collect table lines
                block_type = "table"
                current_block.append(line)

            # Handle lists
            elif re.match(r'^\s*[-*+]|\d+\.', line):
                if current_block and block_type != "list":
                    blocks.append({
                        'type': block_type,
                        'content': '\n'.join(current_block).strip()
                    })
                    current_block = []

                block_type = "list"
                current_block.append(line)

            # Handle headers
            elif line.startswith('#'):
                if current_block:
                    blocks.append({
                        'type': block_type,
                        'content': '\n'.join(current_block).strip()
                    })
                    current_block = []

                current_block.append(line)
                blocks.append({
                    'type': 'header',
                    'content': line.strip()
                })
                current_block = []
                block_type = "paragraph"

            # Handle empty lines
            elif not line.strip():
                if current_block and block_type in ["table", "list"]:
                    blocks.append({
                        'type': block_type,
                        'content': '\n'.join(current_block).strip()
                    })
                    current_block = []
                    block_type = "paragraph"
                elif current_block:
                    current_block.append('')

            # Regular paragraph content
            else:
                if block_type in ["table", "list"]:
                    blocks.append({
                        'type': block_type,
                        'content': '\n'.join(current_block).strip()
                    })
                    current_block = []
                    block_type = "paragraph"
                current_block.append(line)

            i += 1

        # Add final block
        if current_block:
            blocks.append({
                'type': block_type,
                'content': '\n'.join(current_block).strip()
            })

        return [block for block in blocks if block['content'].strip()]

    def _create_chunk(
        self,
        content: str,
        index: int,
        header_title: str,
        header_level: int,
        frontmatter: Optional[Dict[str, Any]],
        source_info: Dict[str, Any],
        header_path: Optional[List[str]] = None,
    ) -> Chunk:
        """Create a header-based chunk."""
        # Include frontmatter if this is the first chunk
        if frontmatter and index == 0:
            if isinstance(frontmatter, dict) and 'raw' not in frontmatter:
                import yaml
                frontmatter_text = yaml.dump(frontmatter, default_flow_style=False)
                content = f"---\n{frontmatter_text}---\n\n{content}"
            else:
                frontmatter_text = frontmatter.get('raw', str(frontmatter))
                content = f"---\n{frontmatter_text}\n---\n\n{content}"

        path = header_path if header_path is not None else [header_title] if header_title else []
        content, crumb = self._decorate(content, path)
        extra = {
            "markdown_header": header_title,
            "markdown_level": header_level,
            "section_index": index,
            "has_frontmatter": frontmatter is not None and index == 0,
            "chunk_type": "header_based"
        }
        extra.update(crumb)

        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"section {index + 1}",
            length=len(content),
            extra=extra,
        )

        return Chunk(
            id=f"md_header_{index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _create_section_chunk(
        self,
        content: str,
        index: int,
        frontmatter: Optional[Dict[str, Any]],
        source_info: Dict[str, Any]
    ) -> Chunk:
        """Create a section-based chunk."""
        if frontmatter and index == 0:
            if isinstance(frontmatter, dict) and 'raw' not in frontmatter:
                import yaml
                frontmatter_text = yaml.dump(frontmatter, default_flow_style=False)
                content = f"---\n{frontmatter_text}---\n\n{content}"
            else:
                frontmatter_text = frontmatter.get('raw', str(frontmatter))
                content = f"---\n{frontmatter_text}\n---\n\n{content}"

        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"section {index + 1}",
            length=len(content),
            extra={
                "section_index": index,
                "has_frontmatter": frontmatter is not None and index == 0,
                "chunk_type": "section_based"
            }
        )

        return Chunk(
            id=f"md_section_{index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _create_block_chunk(
        self,
        content: str,
        index: int,
        frontmatter: Optional[Dict[str, Any]],
        source_info: Dict[str, Any]
    ) -> Chunk:
        """Create a content block-based chunk."""
        if frontmatter and index == 0:
            if isinstance(frontmatter, dict) and 'raw' not in frontmatter:
                import yaml
                frontmatter_text = yaml.dump(frontmatter, default_flow_style=False)
                content = f"---\n{frontmatter_text}---\n\n{content}"
            else:
                frontmatter_text = frontmatter.get('raw', str(frontmatter))
                content = f"---\n{frontmatter_text}\n---\n\n{content}"

        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"block {index + 1}",
            length=len(content),
            extra={
                "block_index": index,
                "has_frontmatter": frontmatter is not None and index == 0,
                "chunk_type": "content_blocks"
            }
        )

        return Chunk(
            id=f"md_block_{index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _create_fixed_chunk(
        self,
        content: str,
        index: int,
        source_info: Dict[str, Any]
    ) -> Chunk:
        """Create a fixed-size chunk."""
        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"chunk {index + 1}",
            length=len(content),
            extra={
                "chunk_index": index,
                "chunk_type": "fixed_size"
            }
        )

        return Chunk(
            id=f"md_fixed_{index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _create_single_chunk(
        self,
        content: str,
        frontmatter: Optional[Dict[str, Any]],
        source_info: Dict[str, Any],
        index: int
    ) -> List[Chunk]:
        """Create a single chunk for content without headers."""
        if frontmatter and self.include_frontmatter:
            if isinstance(frontmatter, dict) and 'raw' not in frontmatter:
                import yaml
                frontmatter_text = yaml.dump(frontmatter, default_flow_style=False)
                content = f"---\n{frontmatter_text}---\n\n{content}"
            else:
                frontmatter_text = frontmatter.get('raw', str(frontmatter))
                content = f"---\n{frontmatter_text}\n---\n\n{content}"

        if self._over_limit(content):
            return self._split_large_chunk(
                content, index, header_path=[], source_info=source_info
            )

        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position="full document",
            length=len(content),
            extra={
                "has_frontmatter": frontmatter is not None,
                "chunk_type": "single",
                "no_headers_found": True
            }
        )

        chunk = Chunk(
            id="md_single_0",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

        return [chunk]

    def _split_large_chunk(
        self,
        content: str,
        base_index: int,
        header_path: Optional[List[str]] = None,
        source_info: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split a large section without cutting GFM/HTML tables or fenced code."""
        from chunking_strategy.core.token_packing import count_tokens

        source_info = source_info or {}
        path = header_path or []
        spans = atomic_spans(content)
        if not spans:
            spans = [(0, len(content), "text")]
        enc = self._tiktoken() if self.max_tokens is not None else None
        ready: List[str] = []
        cur: List[Tuple[int, int, str]] = []
        cur_chars = 0
        cur_tok = 0

        def flush() -> None:
            nonlocal cur_chars, cur_tok
            if not cur:
                return
            ready.append(content[cur[0][0]:cur[-1][1]])
            cur.clear()
            cur_chars = 0
            cur_tok = 0

        for span in spans:
            piece = content[span[0]:span[1]]
            if span[2] == "gfm_table" and self.preserve_tables and self.max_tokens is not None:
                prefix = content[cur[0][0]:cur[-1][1]] if cur else ""
                cur.clear()
                cur_chars = 0
                cur_tok = 0
                self._append_gfm_table(ready, prefix, piece)
                continue
            plen = len(piece)
            ptok = count_tokens(piece, enc) if enc is not None else 0
            over = bool(cur) and (
                cur_chars + plen > self.max_chunk_size
                or (self.max_tokens is not None and cur_tok + ptok > self.max_tokens)
            )
            if over:
                floor = max(1, self.max_tokens // 3) if self.max_tokens else 0
                if self.max_tokens and cur_tok < floor:
                    prefix = content[cur[0][0]:cur[-1][1]]
                    cur.clear()
                    cur_chars = 0
                    cur_tok = 0
                    ready.extend(self._cut_to_fit(prefix + piece))
                    continue
                flush()
            cur.append(span)
            cur_chars += plen
            cur_tok += ptok
            piece_over = plen > self.max_chunk_size or (
                self.max_tokens is not None and ptok > self.max_tokens
            )
            if len(cur) == 1 and piece_over:
                cur.clear()
                cur_chars = 0
                cur_tok = 0
                ready.extend(self._cut_to_fit(piece))
        flush()

        chunks = []
        title = path[-1] if path else ""
        level = len(path) if path else 1
        sub_index = 0
        for body in ready:
            body = body.strip()
            if not body:
                continue
            text, crumb = self._decorate(body, path)
            parts = [text]
            if enc is not None and count_tokens(text, enc) > self.max_tokens:
                parts = self._cut_to_fit(text)
            for part in parts:
                if not part:
                    continue
                extra = {
                    "chunk_type": "oversized_split",
                    "original_index": base_index,
                    "split_index": sub_index,
                    "markdown_header": title,
                    "markdown_level": level,
                }
                extra.update(crumb)
                chunks.append(Chunk(
                    id=f"md_header_{base_index}_{sub_index}",
                    content=part,
                    metadata=ChunkMetadata(
                        source=source_info.get("source", "unknown"),
                        source_type=source_info.get("source_type", "content"),
                        position=f"split {sub_index + 1}",
                        length=len(part),
                        extra=extra,
                    ),
                    modality=ModalityType.TEXT,
                ))
                sub_index += 1
        return chunks

    def _detach_trailing_heading(self, text: str) -> Tuple[str, str]:
        """Take a trailing ATX heading off text so it can lead the next table."""
        lines = text.splitlines()
        end = len(lines)
        while end > 0 and not lines[end - 1].strip():
            end -= 1
        start = end
        while start > 0 and (_HEADING_LINE.match(lines[start - 1]) or not lines[start - 1].strip()):
            start -= 1
        region = lines[start:end]
        if not any(_HEADING_LINE.match(line) for line in region):
            return text, ""
        heading = "\n".join(line for line in region if line.strip())
        rest = "\n".join(lines[:start]).strip()
        return rest, heading

    def _append_gfm_table(self, ready: List[str], prefix: str, table: str) -> None:
        """Keep a GFM table whole when it fits. Split it on rows when it does not.

        The nearest ATX heading is copied onto the table even when a paragraph
        sits between that heading and the table. The paragraph keeps it too.
        A heading with no body is moved onto the table. Each row-split piece
        repeats that heading and the header row.
        """
        from chunking_strategy.core.token_packing import count_tokens

        rest, trailing = self._detach_trailing_heading(prefix) if prefix else ("", "")
        heading = trailing or self._nearest_heading(prefix)
        if not heading and ready and self.max_tokens is not None:
            enc = self._tiktoken()
            floor = max(1, self.max_tokens // 3)
            if count_tokens(ready[-1], enc) < floor:
                heading = self._nearest_heading(ready[-1])
        table_text = f"{heading}\n\n{table.strip()}".strip() if heading else table.strip()
        if rest.strip() and not self._over_limit(f"{rest.strip()}\n\n{table_text}"):
            ready.append(f"{rest.strip()}\n\n{table_text}")
            return
        if rest.strip():
            ready.append(rest.strip())
        if ready and not self._over_limit(f"{ready[-1].rstrip()}\n\n{table_text}"):
            ready[-1] = f"{ready[-1].rstrip()}\n\n{table_text}"
            return
        if not self._over_limit(table_text):
            ready.append(table_text)
            return
        ready.extend(part for part in self._cut_to_fit(table_text) if part.strip())

    def _split_oversized_tables(self, text: str) -> Optional[List[str]]:
        """Pack prose and GFM tables up to max_tokens.

        A table that fits stays one piece. A table that does not is cut
        between rows. Each later piece repeats that table's heading and
        header row. Returns None when the text has no GFM table.
        """
        if not self.preserve_tables or self.max_tokens is None:
            return None
        blocks = self._gfm_blocks(text)
        if not any(kind == "table" for kind, *_ in blocks):
            return None
        from chunking_strategy.core.token_packing import count_tokens

        enc = self._tiktoken()
        pieces: List[str] = []
        current: List[str] = []

        def packed() -> str:
            return "\n\n".join(current)

        def flush() -> None:
            if current:
                pieces.append(packed())
                current.clear()

        def add_unit(unit: str) -> None:
            unit = unit.strip()
            if not unit:
                return
            if count_tokens(unit, enc) > self.max_tokens:
                flush()
                pieces.extend(self._cut_token_windows(unit))
                return
            trial = unit if not current else f"{packed()}\n\n{unit}"
            if count_tokens(trial, enc) <= self.max_tokens:
                current.append(unit)
                return
            flush()
            current.append(unit)

        for block in blocks:
            if block[0] == "prose":
                add_unit("\n".join(block[1]))
                continue
            self._pack_gfm_rows(block[1], block[2], add_unit)
        flush()
        return pieces

    def _gfm_blocks(self, text: str) -> List[Tuple]:
        """Prose blocks and GFM tables. A heading immediately above a table belongs to that table."""
        lines = text.splitlines()
        blocks: List[Tuple] = []
        buf: List[str] = []
        index = 0

        def flush_buf() -> None:
            nonlocal buf
            if any(line.strip() for line in buf):
                blocks.append(("prose", buf))
            buf = []

        while index < len(lines):
            if not _GFM_ROW.match(lines[index]):
                buf.append(lines[index])
                index += 1
                continue
            table: List[str] = []
            while index < len(lines) and _GFM_ROW.match(lines[index]):
                table.append(lines[index])
                index += 1
            if len(table) < 2:
                buf.extend(table)
                continue
            preamble = self._pop_trailing_heading(buf)
            flush_buf()
            blocks.append(("table", preamble, table))
        flush_buf()
        return blocks

    def _pop_trailing_heading(self, buf: List[str]) -> List[str]:
        end = len(buf)
        while end > 0 and not buf[end - 1].strip():
            end -= 1
        start = end
        while start > 0 and (_HEADING_LINE.match(buf[start - 1]) or not buf[start - 1].strip()):
            start -= 1
        region = buf[start:end]
        if not any(_HEADING_LINE.match(line) for line in region):
            return []
        del buf[start:]
        return [line for line in region if line.strip()]

    def _pack_gfm_rows(self, preamble: List[str], rows: List[str], add_unit) -> None:
        """Fit a whole table in one piece, or cut between rows and repeat the header."""
        from chunking_strategy.core.token_packing import count_tokens

        enc = self._tiktoken()
        heading = "\n".join(preamble).strip()
        header = rows[0]
        separator = rows[1] if len(rows) > 1 and _is_gfm_separator(rows[1]) else ""
        data = rows[2:] if separator else rows[1:]
        prefix_lines = [line for line in (heading, header, separator) if line]
        prefix = "\n".join(prefix_lines)
        whole = "\n".join([heading, *rows] if heading else rows)

        def with_prefix(body_rows: List[str]) -> str:
            body = "\n".join(body_rows)
            return f"{prefix}\n{body}" if body else prefix

        if count_tokens(whole, enc) <= self.max_tokens:
            add_unit(whole)
            return
        if not data:
            add_unit(whole)
            return
        # ponytail: one GFM row wider than max_tokens is still token-cut.
        # Upgrade path: wrap that row's cells before cutting.
        held: List[str] = []
        for row in data:
            if count_tokens(with_prefix(held + [row]), enc) <= self.max_tokens:
                held.append(row)
                continue
            if held:
                add_unit(with_prefix(held))
                held = []
            if count_tokens(with_prefix([row]), enc) <= self.max_tokens:
                held = [row]
                continue
            add_unit(row)
        if held:
            add_unit(with_prefix(held))

    def _cut_to_fit(self, text: str) -> List[str]:
        """Last resort when one span is larger than the embedder accepts.

        A GFM table that fits stays whole. A table past the cap is cut on
        row boundaries, with its heading and header row repeated. Anything
        else is cut on a sentence when one falls inside the window.
        """
        if self.max_tokens is not None:
            tabled = self._split_oversized_tables(text)
            if tabled is not None:
                return tabled
        if self.max_tokens is None:
            size = max(1, self.max_chunk_size)
            step = max(1, size - min(size - 1, int(size * 0.1)))
            pieces = []
            start = 0
            while start < len(text):
                pieces.append(text[start:start + size])
                if start + size >= len(text):
                    break
                start += step
            return pieces
        return self._cut_token_windows(text)

    def _cut_token_windows(self, text: str) -> List[str]:
        """Cut prose on a sentence, then on token windows. Does not re-enter table splitting.

        Later windows repeat the section heading that opened this span, when
        that still fits. The first window already contains it.
        """
        from chunking_strategy.core.token_packing import count_tokens

        enc = self._tiktoken()
        ids = enc.encode(text)
        heading = self._opening_heading(text)
        lead = count_tokens(heading + "\n\n", enc) if heading else 0
        if lead >= self.max_tokens:
            heading = ""
            lead = 0
        pieces: List[str] = []
        start = 0
        guard = 0
        while start < len(ids) and guard < len(ids) + 2:
            guard += 1
            budget = self.max_tokens if start == 0 or not heading else max(1, self.max_tokens - lead)
            end = min(start + budget, len(ids))
            if end < len(ids):
                end = self._snap_sentence_end(enc, ids, start, end)
            piece = self._decode_within_cap(enc, ids, start, end)
            if not piece:
                break
            if start > 0 and heading and not piece.lstrip().startswith(heading):
                prefixed = f"{heading}\n\n{piece.lstrip()}"
                if count_tokens(prefixed, enc) <= self.max_tokens:
                    piece = prefixed
            pieces.append(piece)
            if end >= len(ids):
                break
            overlap = min(self.overlap_tokens, max(0, end - start - 1))
            nxt = end - overlap if overlap else end
            if nxt <= start:
                nxt = end
            start = nxt
        return pieces

    def _snap_sentence_end(self, enc, ids: List[int], start: int, end: int) -> int:
        """Move a window end back to the last sentence or line break that fits."""
        from chunking_strategy.core.token_packing import count_tokens

        raw = enc.decode(ids[start:end])
        min_chars = max(80, len(raw) // 2)
        best = -1
        for sep in ("\n", ". ", "? ", "! "):
            pos = raw.rfind(sep)
            if pos >= min_chars:
                best = max(best, pos + len(sep))
        if best <= 0:
            return end
        n = count_tokens(raw[:best], enc)
        if n < 1 or n > self.max_tokens or start + n >= end:
            return end
        return start + n

    def _decode_within_cap(self, enc, ids: List[int], start: int, end: int) -> str:
        """Decode a token window that still fits after strip().

        Dropping a leading space can make the same byte string encode as more
        tokens (`` LGPL`` is 1 token, ``LGPL`` is 2). Shrink the window until
        the stripped text is back under max_tokens. The next window overlaps,
        so the dropped tail is still emitted.
        """
        from chunking_strategy.core.token_packing import count_tokens
        hi = end
        while hi > start:
            piece = enc.decode(ids[start:hi]).strip()
            if piece and count_tokens(piece, enc) <= self.max_tokens:
                return piece
            hi -= 1
        return ""

    def _get_overlap_content(self, content: str, overlap_size: int) -> str:
        """Get overlap content from the end of a chunk."""
        words = content.split()
        if len(words) <= overlap_size:
            return content
        return " ".join(words[-overlap_size:])

        # Streaming and adaptation methods
    def can_stream(self) -> bool:
        """Check if this chunker supports streaming."""
        return True

    def chunk_stream(self, content_stream, source_info=None, **kwargs):
        """Stream chunks as content becomes available."""
        # For Markdown, we need the full content to properly identify headers
        # So we collect all content first then process
        full_content = ""
        for chunk in content_stream:
            if isinstance(chunk, bytes):
                chunk = chunk.decode('utf-8')
            full_content += chunk

        result = self.chunk(full_content, source_info=source_info, **kwargs)
        # Yield chunks one by one for streaming interface
        for chunk in result.chunks:
            yield chunk

    def adapt_parameters(self, feedback_score: float, feedback_type: str = "quality", **kwargs) -> None:
        """Adapt chunking parameters based on feedback."""
        # Track adaptation
        adaptation_record = {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "old_config": {
                "chunk_size": self.chunk_size,
                "header_level": self.header_level,
                "max_chunk_size": self.max_chunk_size
            }
        }

        # Apply adaptations based on feedback score
        if feedback_score < 0.5:  # Poor performance, make chunks smaller/more granular
            if feedback_type == "quality":
                self.header_level = max(1, self.header_level - 1)  # More granular splitting
                self.chunk_size = max(500, int(self.chunk_size * 0.8))
            elif feedback_type == "performance":
                self.max_chunk_size = max(1000, int(self.max_chunk_size * 0.7))
        elif feedback_score > 0.8:  # Good performance, can increase chunk sizes
            if feedback_type == "quality":
                self.header_level = min(6, self.header_level + 1)  # Less granular splitting
                self.chunk_size = min(5000, int(self.chunk_size * 1.2))
            elif feedback_type == "performance":
                self.max_chunk_size = min(10000, int(self.max_chunk_size * 1.3))

        # Handle specific feedback
        feedback_kwargs = kwargs
        if feedback_kwargs.get("chunks_too_large"):
            self.chunk_size = max(500, int(self.chunk_size * 0.8))
            self.max_chunk_size = max(1000, int(self.max_chunk_size * 0.8))
        elif feedback_kwargs.get("chunks_too_small"):
            self.chunk_size = min(5000, int(self.chunk_size * 1.2))
            self.max_chunk_size = min(10000, int(self.max_chunk_size * 1.2))

        if feedback_kwargs.get("prefer_smaller_headers"):
            self.header_level = max(1, self.header_level - 1)
        elif feedback_kwargs.get("prefer_larger_headers"):
            self.header_level = min(6, self.header_level + 1)

        # Record new config
        adaptation_record["new_config"] = {
            "chunk_size": self.chunk_size,
            "header_level": self.header_level,
            "max_chunk_size": self.max_chunk_size
        }

        # Store adaptation history
        if not hasattr(self, '_adaptation_history'):
            self._adaptation_history = []
        self._adaptation_history.append(adaptation_record)

        self.logger.info(f"Adapted parameters based on {feedback_type} feedback (score: {feedback_score})")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of adaptations made."""
        if not hasattr(self, '_adaptation_history'):
            self._adaptation_history = []
        return self._adaptation_history.copy()
