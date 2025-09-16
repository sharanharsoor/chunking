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
            min_chunk_size: Minimum chunk size
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
            # Check if it's a file path or actual content
            if (len(content) < 300 and '\n' not in content and
                content.strip() and not content.isspace() and
                Path(content).exists() and Path(content).is_file()):
                file_path = Path(content)
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
        """Chunk by header hierarchy."""
        chunks = []

        # Find all headers up to the specified level
        headers = []
        for match in self.header_pattern.finditer(content):
            level = len(match.group(1))
            if level <= self.header_level:
                headers.append({
                    'level': level,
                    'title': match.group(2).strip(),
                    'start': match.start(),
                    'end': match.end()
                })

        if not headers:
            # No headers found, treat as single chunk
            return self._create_single_chunk(content, frontmatter, source_info, 0)

        # Create chunks between headers
        for i, header in enumerate(headers):
            # Determine chunk boundaries
            chunk_start = header['start']
            chunk_end = headers[i + 1]['start'] if i + 1 < len(headers) else len(content)

            chunk_content = content[chunk_start:chunk_end].strip()

            # Skip empty chunks
            if not chunk_content or len(chunk_content) < self.min_chunk_size:
                continue

            # Handle oversized chunks
            if len(chunk_content) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk_content, i)
                chunks.extend(sub_chunks)
            else:
                chunk = self._create_chunk(
                    chunk_content,
                    i,
                    header['title'],
                    header['level'],
                    frontmatter if i == 0 and self.include_frontmatter else None,
                    source_info
                )
                chunks.append(chunk)

        return chunks

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
        source_info: Dict[str, Any]
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

        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"section {index + 1}",
            length=len(content),
            extra={
                "markdown_header": header_title,
                "markdown_level": header_level,
                "section_index": index,
                "has_frontmatter": frontmatter is not None and index == 0,
                "chunk_type": "header_based"
            }
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

    def _split_large_chunk(self, content: str, base_index: int) -> List[Chunk]:
        """Split a large chunk into smaller ones."""
        chunks = []
        words = content.split()
        current_chunk = ""
        sub_index = 0

        for word in words:
            if len(current_chunk) + len(word) + 1 > self.max_chunk_size and current_chunk:
                chunk = Chunk(
                    id=f"md_header_{base_index}_{sub_index}",
                    content=current_chunk.strip(),
                    metadata=ChunkMetadata(
                        source="large_chunk_split",
                        source_type="content",
                        position=f"split {sub_index + 1}",
                        length=len(current_chunk.strip()),
                        extra={
                            "chunk_type": "oversized_split",
                            "original_index": base_index,
                            "split_index": sub_index
                        }
                    ),
                    modality=ModalityType.TEXT
                )
                chunks.append(chunk)

                current_chunk = word
                sub_index += 1
            else:
                current_chunk += " " + word if current_chunk else word

        # Add final sub-chunk
        if current_chunk.strip():
            chunk = Chunk(
                id=f"md_header_{base_index}_{sub_index}",
                content=current_chunk.strip(),
                metadata=ChunkMetadata(
                    source="large_chunk_split",
                    source_type="content",
                    position=f"split {sub_index + 1}",
                    length=len(current_chunk.strip()),
                    extra={
                        "chunk_type": "oversized_split",
                        "original_index": base_index,
                        "split_index": sub_index
                    }
                ),
                modality=ModalityType.TEXT
            )
            chunks.append(chunk)

        return chunks

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
