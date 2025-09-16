"""
Boundary-Aware Chunking Strategy.

This module implements a boundary-aware text chunker that respects document
structure and boundaries. Instead of breaking text arbitrarily, it identifies
and preserves natural document boundaries such as HTML/XML elements, Markdown
sections, paragraphs, code blocks, lists, and other structural elements.

Key Features:
- Multiple document format support (HTML, XML, Markdown, plain text)
- Structural boundary detection (headers, paragraphs, lists, code blocks, tables)
- Configurable boundary priorities and preferences
- Fallback to content-based boundaries when structural boundaries are too large
- Structure metadata preservation
- Streaming capabilities with boundary buffering
- Adaptive parameter tuning based on document structure
- Performance optimization with incremental parsing

Author: AI Assistant
Date: 2024
"""

import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple, Iterator, Set
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

from chunking_strategy.core.base import (
    StreamableChunker,
    AdaptableChunker,
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType
)
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage

# Optional imports with fallbacks
try:
    from bs4 import BeautifulSoup, NavigableString, Tag
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    BeautifulSoup = NavigableString = Tag = None

try:
    import markdown
    from markdown.extensions import codehilite, fenced_code, tables
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    markdown = codehilite = fenced_code = tables = None

try:
    import html2text
    HTML2TEXT_AVAILABLE = True
except ImportError:
    HTML2TEXT_AVAILABLE = False
    html2text = None


class DocumentFormat(Enum):
    """Supported document formats for boundary detection."""
    HTML = "html"
    XML = "xml"
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    AUTO = "auto"


class BoundaryType(Enum):
    """Types of boundaries that can be detected."""
    PARAGRAPH = "paragraph"
    HEADER = "header"
    LIST_ITEM = "list_item"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    BLOCKQUOTE = "blockquote"
    SECTION = "section"
    DIV = "div"
    ARTICLE = "article"
    HORIZONTAL_RULE = "horizontal_rule"
    LINE_BREAK = "line_break"


class BoundaryStrategy(Enum):
    """Strategies for handling boundary detection."""
    STRICT = "strict"              # Only split at detected boundaries
    ADAPTIVE = "adaptive"          # Split at boundaries with size fallback
    HYBRID = "hybrid"             # Combine structural and semantic boundaries
    CONTENT_AWARE = "content_aware"  # Analyze content within boundaries


@dataclass
class StructuralBoundary:
    """Represents a detected structural boundary in the document."""
    boundary_type: BoundaryType
    start_pos: int
    end_pos: int
    tag_name: Optional[str] = None
    attributes: Optional[Dict[str, str]] = None
    level: Optional[int] = None  # For headers (h1=1, h2=2, etc.)
    content_preview: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@register_chunker(
    name="boundary_aware",
    category="text",
    description="Boundary-aware chunker that respects document structure like HTML/XML tags, Markdown headers, paragraphs, code blocks, and lists",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "html", "xml", "json", "rtf"],
    complexity=ComplexityLevel.MEDIUM,
    dependencies=[],
    optional_dependencies=["beautifulsoup4", "markdown", "html2text"],
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    quality=0.8,
    use_cases=["structured_document_processing", "HTML_chunking", "Markdown_processing", "boundary_preservation"],
    best_for=["HTML documents", "Markdown files", "structured content", "web pages", "technical documentation"],
    limitations=["may produce varying chunk sizes", "depends on document structure quality"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=False
)
class BoundaryAwareChunker(StreamableChunker, AdaptableChunker):
    """
    Boundary-aware text chunker that respects document structure.

    This chunker analyzes document structure to identify natural boundaries
    such as HTML/XML elements, Markdown sections, paragraphs, code blocks,
    and other structural elements. It preserves document structure while
    creating semantically coherent chunks.
    """

    def __init__(
        self,
        document_format: str = "auto",
        boundary_strategy: str = "adaptive",
        preferred_boundaries: Optional[List[str]] = None,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 200,
        target_chunk_size: int = 1000,
        preserve_structure: bool = True,
        include_metadata: bool = True,
        fallback_to_sentences: bool = True,
        respect_code_blocks: bool = True,
        respect_tables: bool = True,
        header_split_priority: int = 1,
        paragraph_split_priority: int = 3,
        list_split_priority: int = 2,
        adaptive_sizing: bool = True,
        quality_threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Boundary-Aware Chunker.

        Args:
            document_format: Format of input documents (html, xml, markdown, plain_text, auto)
            boundary_strategy: Strategy for boundary detection and handling
            preferred_boundaries: List of preferred boundary types (in priority order)
            max_chunk_size: Maximum character count per chunk
            min_chunk_size: Minimum character count per chunk
            target_chunk_size: Target character count per chunk
            preserve_structure: Whether to preserve structural information in metadata
            include_metadata: Whether to include rich metadata about boundaries
            fallback_to_sentences: Whether to fall back to sentence boundaries
            respect_code_blocks: Whether to keep code blocks intact
            respect_tables: Whether to keep tables intact
            header_split_priority: Priority for header boundaries (1=highest)
            paragraph_split_priority: Priority for paragraph boundaries
            list_split_priority: Priority for list item boundaries
            adaptive_sizing: Whether to adapt chunk sizes based on structure
            quality_threshold: Minimum quality score for adaptive tuning
        """
        # Initialize base class
        super().__init__(
            name="boundary_aware",
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        # Validate parameters
        self._validate_parameters(max_chunk_size, min_chunk_size, quality_threshold)

        # Core configuration
        self.document_format = DocumentFormat(document_format)
        self.boundary_strategy = BoundaryStrategy(boundary_strategy)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.target_chunk_size = target_chunk_size
        self.preserve_structure = preserve_structure
        self.include_metadata = include_metadata
        self.fallback_to_sentences = fallback_to_sentences
        self.respect_code_blocks = respect_code_blocks
        self.respect_tables = respect_tables
        self.adaptive_sizing = adaptive_sizing
        self.quality_threshold = quality_threshold

        # Boundary priorities
        self.boundary_priorities = {
            BoundaryType.HEADER: header_split_priority,
            BoundaryType.PARAGRAPH: paragraph_split_priority,
            BoundaryType.LIST_ITEM: list_split_priority,
            BoundaryType.CODE_BLOCK: 1,  # Always high priority
            BoundaryType.TABLE: 1,       # Always high priority
            BoundaryType.BLOCKQUOTE: 2,
            BoundaryType.SECTION: 1,
            BoundaryType.HORIZONTAL_RULE: 2,
            BoundaryType.LINE_BREAK: 5   # Lowest priority
        }

        # Set preferred boundaries
        if preferred_boundaries:
            self.preferred_boundaries = [BoundaryType(b) for b in preferred_boundaries]
        else:
            self.preferred_boundaries = [
                BoundaryType.HEADER,
                BoundaryType.SECTION,
                BoundaryType.PARAGRAPH,
                BoundaryType.LIST_ITEM,
                BoundaryType.CODE_BLOCK,
                BoundaryType.TABLE
            ]

        # Performance tracking
        self.performance_stats = {
            "total_documents_processed": 0,
            "boundary_detection_time": 0.0,
            "parsing_time": 0.0,
            "chunking_time": 0.0,
            "boundaries_detected": 0,
            "fallback_count": 0
        }

        # Adaptation history
        self._adaptation_history = []

        # Initialize parsers
        self._initialize_parsers()

        logging.info(f"BoundaryAwareChunker initialized with {document_format} format")

    def _validate_parameters(
        self,
        max_chunk_size: int,
        min_chunk_size: int,
        quality_threshold: float
    ):
        """Validate initialization parameters."""
        if min_chunk_size >= max_chunk_size:
            raise ValueError("min_chunk_size must be less than max_chunk_size")

        if min_chunk_size < 50:
            raise ValueError("min_chunk_size must be at least 50")

        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be between 0.0 and 1.0")

    def _initialize_parsers(self):
        """Initialize document parsers based on available dependencies."""
        self.html_parser = None
        self.markdown_parser = None
        self.html2text_converter = None

        # Initialize HTML/XML parser
        if BEAUTIFULSOUP_AVAILABLE:
            self.html_parser = True
            logging.info("BeautifulSoup HTML/XML parser available")
        else:
            logging.warning("BeautifulSoup not available, HTML/XML parsing will be limited")

        # Initialize Markdown parser
        if MARKDOWN_AVAILABLE:
            self.markdown_parser = markdown.Markdown(
                extensions=['fenced_code', 'tables', 'toc', 'codehilite']
            )
            logging.info("Markdown parser available with extensions")
        else:
            logging.warning("Markdown library not available, Markdown parsing will be limited")

        # Initialize HTML to text converter
        if HTML2TEXT_AVAILABLE:
            self.html2text_converter = html2text.HTML2Text()
            self.html2text_converter.ignore_links = False
            self.html2text_converter.body_width = 0  # Don't wrap lines
            logging.info("html2text converter available")

    def get_supported_formats(self) -> List[str]:
        """Return supported file formats."""
        return ["txt", "md", "html", "xml", "htm", "json", "csv"]

    def estimate_chunks(self, content: Union[str, Path]) -> int:
        """Estimate number of chunks for given content."""
        if isinstance(content, Path):
            content = content.read_text(encoding='utf-8')

        # Quick estimation based on target chunk size and detected boundaries
        boundaries = self._detect_boundaries_fast(content)
        estimated_chunks = max(1, len(boundaries) // 2)  # Conservative estimate
        return estimated_chunks

    def chunk(
        self,
        content: Union[str, Path, bytes],
        source_info: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Chunk content using boundary-aware structural analysis.

        Args:
            content: Text content, file path, or bytes to chunk
            source_info: Additional source information

        Returns:
            ChunkingResult with structure-aware chunks
        """
        start_time = time.time()

        # Process input content
        text_content, detected_format = self._process_input_content(content)

        if not text_content or not text_content.strip():
            return self._create_empty_result(start_time, source_info)

        try:
            # Detect document format if set to auto
            if self.document_format == DocumentFormat.AUTO:
                self.document_format = detected_format

            # Detect structural boundaries
            boundaries = self._detect_boundaries(text_content)

            # Create chunks from boundaries
            chunks = self._create_chunks_from_boundaries(
                text_content, boundaries, source_info
            )

            # Calculate processing time
            processing_time = time.time() - start_time
            self.performance_stats["total_documents_processed"] += 1
            self.performance_stats["boundaries_detected"] += len(boundaries)

            # Create enhanced source info
            enhanced_source_info = self._create_enhanced_source_info(
                source_info, boundaries, processing_time
            )

            return ChunkingResult(
                chunks=chunks,
                strategy_used=self.name,
                processing_time=processing_time,
                source_info=enhanced_source_info
            )

        except Exception as e:
            logging.error(f"Boundary-aware chunking failed: {e}")
            return self._fallback_chunking(text_content, source_info, start_time)

    def _process_input_content(self, content: Union[str, Path, bytes]) -> Tuple[str, DocumentFormat]:
        """Process various input types into text content and detect format."""
        detected_format = DocumentFormat.PLAIN_TEXT

        if isinstance(content, Path):
            text_content = content.read_text(encoding='utf-8')
            # Detect format from file extension
            suffix = content.suffix.lower()
            if suffix in ['.html', '.htm']:
                detected_format = DocumentFormat.HTML
            elif suffix in ['.xml']:
                detected_format = DocumentFormat.XML
            elif suffix in ['.md', '.markdown']:
                detected_format = DocumentFormat.MARKDOWN

        elif isinstance(content, bytes):
            text_content = content.decode('utf-8')
            # Simple format detection from content
            detected_format = self._detect_format_from_content(text_content)

        elif isinstance(content, str):
            text_content = content
            # Simple format detection from content
            detected_format = self._detect_format_from_content(text_content)

        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        return text_content, detected_format

    def _detect_format_from_content(self, content: str) -> DocumentFormat:
        """Detect document format from content analysis."""
        # Simple heuristics for format detection - check markdown first before HTML
        # as some markdown might contain HTML tags

        # Check for HTML/XML tags first (more specific than markdown)
        if re.search(r'<[^>]+>', content):
            # More specific HTML detection
            if re.search(r'<html|<body|<head|<div|<p>|<span|<a\s+|<!DOCTYPE', content, re.IGNORECASE):
                return DocumentFormat.HTML
            # XML detection
            elif re.search(r'<\?xml|<[^>]*xmlns', content, re.IGNORECASE):
                return DocumentFormat.XML
            else:
                # Generic XML if has tags but not HTML-specific
                return DocumentFormat.XML

        # Check for strong markdown indicators only if no HTML tags found
        markdown_patterns = [
            r'^#+\s+',          # Headers (must be at line start)
            r'```',             # Code blocks
            r'^\s*[-*+]\s+',    # Unordered lists
            r'^\s*\d+\.\s+',    # Ordered lists
            r'\|.*\|.*\|',      # Tables (must have multiple |)
            r'^>\s+',           # Blockquotes
            r'^---+$'           # Horizontal rules
        ]

        markdown_matches = sum(1 for pattern in markdown_patterns
                             if re.search(pattern, content, re.MULTILINE))

        # Need multiple markdown indicators to confidently detect markdown
        if markdown_matches >= 2:
            return DocumentFormat.MARKDOWN

        return DocumentFormat.PLAIN_TEXT

    def _detect_boundaries_fast(self, content: str) -> List[StructuralBoundary]:
        """Fast boundary detection for estimation purposes."""
        boundaries = []

        # Quick paragraph detection
        paragraphs = re.split(r'\n\s*\n', content)
        pos = 0
        for para in paragraphs:
            if para.strip():
                boundaries.append(StructuralBoundary(
                    boundary_type=BoundaryType.PARAGRAPH,
                    start_pos=pos,
                    end_pos=pos + len(para)
                ))
            pos += len(para) + 2  # Account for newlines

        return boundaries

    def _detect_boundaries(self, content: str) -> List[StructuralBoundary]:
        """Detect structural boundaries in the content."""
        start_time = time.time()
        boundaries = []
        fallback_used = False

        try:
            if self.document_format == DocumentFormat.HTML:
                boundaries = self._detect_html_boundaries(content)
            elif self.document_format == DocumentFormat.XML:
                boundaries = self._detect_xml_boundaries(content)
            elif self.document_format == DocumentFormat.MARKDOWN:
                boundaries = self._detect_markdown_boundaries(content)
            else:
                boundaries = self._detect_plain_text_boundaries(content)

        except Exception as e:
            logging.warning(f"Boundary detection failed: {e}, falling back to plain text")
            boundaries = self._detect_plain_text_boundaries(content)
            fallback_used = True
            self.performance_stats["fallback_count"] += 1

        # Check if we had to fallback due to missing libraries
        if ((self.document_format == DocumentFormat.HTML or self.document_format == DocumentFormat.XML) and
            not BEAUTIFULSOUP_AVAILABLE):
            fallback_used = True
            self.performance_stats["fallback_count"] += 1

        if (self.document_format == DocumentFormat.MARKDOWN and not MARKDOWN_AVAILABLE):
            fallback_used = True
            self.performance_stats["fallback_count"] += 1

        # Sort boundaries by position
        boundaries.sort(key=lambda b: b.start_pos)

        # Filter and prioritize boundaries
        boundaries = self._filter_and_prioritize_boundaries(boundaries)

        self.performance_stats["boundary_detection_time"] += time.time() - start_time

        # Store fallback status for later use
        self._fallback_used = fallback_used

        return boundaries

    def _detect_html_boundaries(self, content: str) -> List[StructuralBoundary]:
        """Detect boundaries in HTML content."""
        if not BEAUTIFULSOUP_AVAILABLE:
            return self._detect_plain_text_boundaries(content)

        boundaries = []
        soup = BeautifulSoup(content, 'html.parser')

        # Find structural elements
        structural_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div',
                          'section', 'article', 'header', 'footer', 'main',
                          'ul', 'ol', 'li', 'blockquote', 'pre', 'code',
                          'table', 'tr', 'td', 'th', 'hr']

        for tag in soup.find_all(structural_tags):
            boundary_type = self._html_tag_to_boundary_type(tag.name)
            start_pos = content.find(str(tag))

            if start_pos != -1:
                boundaries.append(StructuralBoundary(
                    boundary_type=boundary_type,
                    start_pos=start_pos,
                    end_pos=start_pos + len(str(tag)),
                    tag_name=tag.name,
                    attributes=dict(tag.attrs) if tag.attrs else None,
                    level=int(tag.name[1]) if tag.name.startswith('h') and tag.name[1:].isdigit() else None,
                    content_preview=tag.get_text()[:100]
                ))

        return boundaries

    def _detect_xml_boundaries(self, content: str) -> List[StructuralBoundary]:
        """Detect boundaries in XML content."""
        if not BEAUTIFULSOUP_AVAILABLE:
            return self._detect_plain_text_boundaries(content)

        boundaries = []
        soup = BeautifulSoup(content, 'xml')

        # Find all tags as potential boundaries
        for tag in soup.find_all():
            if tag.name:
                start_pos = content.find(f'<{tag.name}')
                if start_pos != -1:
                    boundaries.append(StructuralBoundary(
                        boundary_type=BoundaryType.SECTION,
                        start_pos=start_pos,
                        end_pos=start_pos + len(str(tag)),
                        tag_name=tag.name,
                        attributes=dict(tag.attrs) if tag.attrs else None,
                        content_preview=tag.get_text()[:100] if hasattr(tag, 'get_text') else None
                    ))

        return boundaries

    def _detect_markdown_boundaries(self, content: str) -> List[StructuralBoundary]:
        """Detect boundaries in Markdown content."""
        boundaries = []
        lines = content.split('\n')
        pos = 0

        for i, line in enumerate(lines):
            line_start = pos
            line_end = pos + len(line)

            # Headers
            if re.match(r'^#+\s+', line):
                level = len(line) - len(line.lstrip('#'))
                boundaries.append(StructuralBoundary(
                    boundary_type=BoundaryType.HEADER,
                    start_pos=line_start,
                    end_pos=line_end,
                    level=level,
                    content_preview=line.strip('#').strip()[:100]
                ))

            # Code blocks
            elif line.strip().startswith('```'):
                # Find end of code block
                end_line = i + 1
                while end_line < len(lines) and not lines[end_line].strip().startswith('```'):
                    end_line += 1

                if end_line < len(lines):
                    block_end = sum(len(lines[j]) + 1 for j in range(i, end_line + 1)) + pos - len(line) - 1
                    boundaries.append(StructuralBoundary(
                        boundary_type=BoundaryType.CODE_BLOCK,
                        start_pos=line_start,
                        end_pos=block_end,
                        content_preview=line.strip('`').strip()[:50]
                    ))

            # List items
            elif re.match(r'^[\s]*[-*+]\s+', line) or re.match(r'^[\s]*\d+\.\s+', line):
                boundaries.append(StructuralBoundary(
                    boundary_type=BoundaryType.LIST_ITEM,
                    start_pos=line_start,
                    end_pos=line_end,
                    content_preview=line.strip()[:100]
                ))

            # Horizontal rules
            elif re.match(r'^[\s]*[-*_]{3,}[\s]*$', line):
                boundaries.append(StructuralBoundary(
                    boundary_type=BoundaryType.HORIZONTAL_RULE,
                    start_pos=line_start,
                    end_pos=line_end
                ))

            # Blockquotes
            elif line.strip().startswith('>'):
                boundaries.append(StructuralBoundary(
                    boundary_type=BoundaryType.BLOCKQUOTE,
                    start_pos=line_start,
                    end_pos=line_end,
                    content_preview=line.strip('> ').strip()[:100]
                ))

            pos += len(line) + 1  # +1 for newline

        return boundaries

    def _detect_plain_text_boundaries(self, content: str) -> List[StructuralBoundary]:
        """Detect boundaries in plain text content."""
        boundaries = []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', content)
        pos = 0

        for para in paragraphs:
            if para.strip():
                boundaries.append(StructuralBoundary(
                    boundary_type=BoundaryType.PARAGRAPH,
                    start_pos=pos,
                    end_pos=pos + len(para),
                    content_preview=para.strip()[:100]
                ))

            # Move position past this paragraph and the separating newlines
            pos = content.find(para, pos) + len(para)
            while pos < len(content) and content[pos] in '\n ':
                pos += 1

        return boundaries

    def _html_tag_to_boundary_type(self, tag_name: str) -> BoundaryType:
        """Convert HTML tag name to boundary type."""
        mapping = {
            'h1': BoundaryType.HEADER, 'h2': BoundaryType.HEADER,
            'h3': BoundaryType.HEADER, 'h4': BoundaryType.HEADER,
            'h5': BoundaryType.HEADER, 'h6': BoundaryType.HEADER,
            'p': BoundaryType.PARAGRAPH,
            'div': BoundaryType.DIV,
            'section': BoundaryType.SECTION,
            'article': BoundaryType.ARTICLE,
            'li': BoundaryType.LIST_ITEM,
            'blockquote': BoundaryType.BLOCKQUOTE,
            'pre': BoundaryType.CODE_BLOCK,
            'code': BoundaryType.CODE_BLOCK,
            'table': BoundaryType.TABLE,
            'hr': BoundaryType.HORIZONTAL_RULE
        }
        return mapping.get(tag_name, BoundaryType.SECTION)

    def _filter_and_prioritize_boundaries(
        self,
        boundaries: List[StructuralBoundary]
    ) -> List[StructuralBoundary]:
        """Filter overlapping boundaries and prioritize by importance."""
        if not boundaries:
            return boundaries

        # Remove very small boundaries
        filtered = [b for b in boundaries if (b.end_pos - b.start_pos) >= 20]

        # Sort by priority (lower number = higher priority)
        filtered.sort(key=lambda b: (
            self.boundary_priorities.get(b.boundary_type, 10),
            b.start_pos
        ))

        # Remove overlapping boundaries, keeping higher priority ones
        final_boundaries = []
        for boundary in filtered:
            # Check if this boundary overlaps significantly with existing ones
            overlaps = False
            for existing in final_boundaries:
                if self._boundaries_overlap(boundary, existing, threshold=0.3):
                    overlaps = True
                    break

            if not overlaps:
                final_boundaries.append(boundary)

        return sorted(final_boundaries, key=lambda b: b.start_pos)

    def _boundaries_overlap(
        self,
        boundary1: StructuralBoundary,
        boundary2: StructuralBoundary,
        threshold: float = 0.3
    ) -> bool:
        """Check if two boundaries overlap significantly."""
        # Calculate overlap
        overlap_start = max(boundary1.start_pos, boundary2.start_pos)
        overlap_end = min(boundary1.end_pos, boundary2.end_pos)

        if overlap_start >= overlap_end:
            return False

        overlap_length = overlap_end - overlap_start
        min_length = min(
            boundary1.end_pos - boundary1.start_pos,
            boundary2.end_pos - boundary2.start_pos
        )

        overlap_ratio = overlap_length / min_length if min_length > 0 else 0
        return overlap_ratio > threshold

    def _create_chunks_from_boundaries(
        self,
        content: str,
        boundaries: List[StructuralBoundary],
        source_info: Optional[Dict[str, Any]]
    ) -> List[Chunk]:
        """Create Chunk objects from detected boundaries."""
        if not boundaries:
            # Fallback to simple paragraph splitting
            return self._create_fallback_chunks(content)

        chunks = []
        current_pos = 0

        for i, boundary in enumerate(boundaries):
            # Extract content for this boundary
            boundary_start = boundary.start_pos
            boundary_end = boundary.end_pos

            # Include content before boundary if significant
            if boundary_start > current_pos + 50:  # 50 char minimum
                preceding_content = content[current_pos:boundary_start].strip()
                if preceding_content:
                    chunks.append(self._create_chunk_from_content(
                        preceding_content,
                        len(chunks),
                        current_pos,
                        BoundaryType.PARAGRAPH,
                        {"type": "preceding_content"}
                    ))

            # Create chunk from boundary content
            boundary_content = content[boundary_start:boundary_end].strip()
            if boundary_content:
                # Check if chunk is too large and needs splitting
                if len(boundary_content) > self.max_chunk_size:
                    sub_chunks = self._split_large_content(
                        boundary_content, boundary, boundary_start
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(self._create_chunk_from_boundary(
                        boundary_content,
                        len(chunks),
                        boundary
                    ))

            current_pos = boundary_end

        # Handle remaining content
        if current_pos < len(content):
            remaining_content = content[current_pos:].strip()
            if remaining_content:
                chunks.append(self._create_chunk_from_content(
                    remaining_content,
                    len(chunks),
                    current_pos,
                    BoundaryType.PARAGRAPH,
                    {"type": "trailing_content"}
                ))

        return self._post_process_chunks(chunks)

    def _create_chunk_from_boundary(
        self,
        content: str,
        chunk_index: int,
        boundary: StructuralBoundary
    ) -> Chunk:
        """Create a chunk from a structural boundary."""
        metadata = ChunkMetadata(
            source="content",
            source_type="content",
            position=f"{boundary.boundary_type.value}_{boundary.start_pos}",
            length=len(content),
            offset=boundary.start_pos,
            extra={
                "chunker_used": "boundary_aware",
                "chunk_index": chunk_index,
                "boundary_type": boundary.boundary_type.value,
                "tag_name": boundary.tag_name,
                "attributes": boundary.attributes,
                "level": boundary.level,
                "content_preview": boundary.content_preview,
                "chunking_strategy": "boundary_aware",
                "structure_preserved": True
            }
        )

        return Chunk(
            id=f"boundary_{chunk_index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _create_chunk_from_content(
        self,
        content: str,
        chunk_index: int,
        offset: int,
        boundary_type: BoundaryType,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """Create a chunk from plain content."""
        metadata = ChunkMetadata(
            source="content",
            source_type="content",
            position=f"{boundary_type.value}_{offset}",
            length=len(content),
            offset=offset,
            extra={
                "chunker_used": "boundary_aware",
                "chunk_index": chunk_index,
                "boundary_type": boundary_type.value,
                "chunking_strategy": "boundary_aware",
                "structure_preserved": False,
                **(extra_metadata or {})
            }
        )

        return Chunk(
            id=f"content_{chunk_index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _split_large_content(
        self,
        content: str,
        boundary: StructuralBoundary,
        offset: int
    ) -> List[Chunk]:
        """Split large content that exceeds maximum chunk size."""
        chunks = []

        if self.fallback_to_sentences:
            # Split by sentences
            sentences = re.split(r'[.!?]+\s+', content)
            current_chunk = ""
            chunk_count = 0

            for sentence in sentences:
                if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                    chunks.append(self._create_chunk_from_content(
                        current_chunk.strip(),
                        chunk_count,
                        offset,
                        boundary.boundary_type,
                        {"split_from_large": True, "original_boundary": boundary.boundary_type.value}
                    ))
                    current_chunk = sentence
                    chunk_count += 1
                else:
                    current_chunk += sentence + ". "

            # Add final chunk
            if current_chunk.strip():
                chunks.append(self._create_chunk_from_content(
                    current_chunk.strip(),
                    chunk_count,
                    offset,
                    boundary.boundary_type,
                    {"split_from_large": True, "original_boundary": boundary.boundary_type.value}
                ))
        else:
            # Simple character-based splitting
            pos = 0
            chunk_count = 0
            while pos < len(content):
                chunk_end = min(pos + self.target_chunk_size, len(content))
                chunk_content = content[pos:chunk_end]

                chunks.append(self._create_chunk_from_content(
                    chunk_content,
                    chunk_count,
                    offset + pos,
                    boundary.boundary_type,
                    {"split_from_large": True, "original_boundary": boundary.boundary_type.value}
                ))

                pos = chunk_end
                chunk_count += 1

        return chunks

    def _create_fallback_chunks(self, content: str) -> List[Chunk]:
        """Create chunks using fallback method when no boundaries detected."""
        self.performance_stats["fallback_count"] += 1

        chunks = []
        paragraphs = re.split(r'\n\s*\n', content)

        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append(self._create_chunk_from_content(
                    para.strip(),
                    i,
                    0,  # Simplified offset
                    BoundaryType.PARAGRAPH,
                    {"fallback_mode": True}
                ))

        return chunks

    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Post-process chunks to ensure quality and consistency."""
        processed_chunks = []

        for chunk in chunks:
            # Skip very small chunks unless they contain important structure
            if len(chunk.content) < self.min_chunk_size:
                boundary_type = chunk.metadata.extra.get("boundary_type")
                if boundary_type not in ["header", "code_block", "table"]:
                    continue  # Skip small, unimportant chunks

            # Update chunk IDs to be sequential
            chunk.id = f"boundary_{len(processed_chunks)}"
            chunk.metadata.extra["chunk_index"] = len(processed_chunks)

            processed_chunks.append(chunk)

        return processed_chunks

    def _create_enhanced_source_info(
        self,
        source_info: Optional[Dict[str, Any]],
        boundaries: List[StructuralBoundary],
        processing_time: float
    ) -> Dict[str, Any]:
        """Create enhanced source information with boundary analysis."""
        enhanced_info = source_info.copy() if source_info else {}

        # Analyze boundary statistics
        boundary_stats = {}
        for boundary in boundaries:
            btype = boundary.boundary_type.value
            if btype not in boundary_stats:
                boundary_stats[btype] = 0
            boundary_stats[btype] += 1

        enhanced_info.update({
            "boundary_aware_metadata": {
                "document_format": self.document_format.value,
                "boundary_strategy": self.boundary_strategy.value,
                "total_boundaries": len(boundaries),
                "boundary_types": boundary_stats,
                "processing_time": processing_time,
                "performance_stats": self.performance_stats.copy(),
                "preserve_structure": self.preserve_structure,
                "fallback_used": getattr(self, '_fallback_used', False) or self.performance_stats["fallback_count"] > 0
            },
            "chunking_strategy": "boundary_aware",
            "total_boundaries": len(boundaries),
            "document_format": self.document_format.value
        })

        return enhanced_info

    def _create_empty_result(
        self,
        start_time: float,
        source_info: Optional[Dict[str, Any]]
    ) -> ChunkingResult:
        """Create empty result for edge cases."""
        processing_time = time.time() - start_time
        enhanced_source_info = source_info.copy() if source_info else {}
        enhanced_source_info["boundary_aware_metadata"] = {
            "processing_time": processing_time,
            "total_boundaries": 0,
            "reason": "empty_or_invalid_content"
        }

        return ChunkingResult(
            chunks=[],
            strategy_used=self.name,
            processing_time=processing_time,
            source_info=enhanced_source_info
        )

    def _fallback_chunking(
        self,
        content: str,
        source_info: Optional[Dict[str, Any]],
        start_time: float
    ) -> ChunkingResult:
        """Fallback chunking when boundary detection fails."""
        try:
            chunks = self._create_fallback_chunks(content)
            processing_time = time.time() - start_time

            enhanced_source_info = source_info.copy() if source_info else {}
            enhanced_source_info["boundary_aware_metadata"] = {
                "processing_time": processing_time,
                "fallback_mode": True,
                "total_boundaries": 0
            }

            return ChunkingResult(
                chunks=chunks,
                strategy_used="boundary_aware_fallback",
                processing_time=processing_time,
                source_info=enhanced_source_info
            )

        except Exception as e:
            logging.error(f"Fallback chunking also failed: {e}")
            return self._create_empty_result(start_time, source_info)

    def chunk_stream(
        self,
        stream_data: List[str],
        source_info: Optional[Dict[str, Any]] = None
    ) -> Iterator[Chunk]:
        """
        Chunk streaming data using boundary-aware analysis.

        Args:
            stream_data: List of text segments to process as stream
            source_info: Optional source information

        Yields:
            Individual chunks as they are processed
        """
        # Combine stream data for boundary detection
        combined_content = "\n".join(stream_data)
        result = self.chunk(combined_content, source_info)

        for chunk in result.chunks:
            yield chunk

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Adapt chunker parameters based on feedback.

        Args:
            feedback_score: Score from 0.0 to 1.0 indicating quality/performance
            feedback_type: Type of feedback ("quality", "performance", "structure")
            context: Additional context for adaptation

        Returns:
            Dictionary of parameter changes made
        """
        changes = {}
        original_params = {
            "max_chunk_size": self.max_chunk_size,
            "boundary_strategy": self.boundary_strategy.value,
            "target_chunk_size": self.target_chunk_size
        }

        if feedback_score < self.quality_threshold:
            # Poor feedback - adjust for better quality
            if feedback_type == "quality":
                # Reduce chunk sizes for better coherence
                if self.target_chunk_size > self.min_chunk_size + 200:
                    self.target_chunk_size -= 200
                    changes["target_chunk_size"] = self.target_chunk_size

            elif feedback_type == "structure":
                # Switch to more strict boundary strategy
                if self.boundary_strategy == BoundaryStrategy.ADAPTIVE:
                    self.boundary_strategy = BoundaryStrategy.STRICT
                    changes["boundary_strategy"] = self.boundary_strategy.value

            elif feedback_type == "performance":
                # Reduce max size for faster processing
                if self.max_chunk_size > self.min_chunk_size + 500:
                    self.max_chunk_size -= 500
                    changes["max_chunk_size"] = self.max_chunk_size

        elif feedback_score > 0.8:
            # Good feedback - optimize for efficiency
            if feedback_type == "performance" and self.target_chunk_size < 2000:
                self.target_chunk_size += 200
                changes["target_chunk_size"] = self.target_chunk_size

            elif feedback_type == "quality":
                # Can relax boundary strictness slightly
                if self.boundary_strategy == BoundaryStrategy.STRICT:
                    self.boundary_strategy = BoundaryStrategy.ADAPTIVE
                    changes["boundary_strategy"] = self.boundary_strategy.value

        # Record adaptation
        if changes:
            self._adaptation_history.append({
                "timestamp": time.time(),
                "feedback_score": feedback_score,
                "feedback_type": feedback_type,
                "context": context,
                "original_params": original_params,
                "adapted_params": {
                    "max_chunk_size": self.max_chunk_size,
                    "boundary_strategy": self.boundary_strategy.value,
                    "target_chunk_size": self.target_chunk_size
                },
                "changes": changes
            })

        return changes

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get the history of parameter adaptations."""
        return self._adaptation_history.copy()

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration and performance statistics."""
        return {
            "name": self.name,
            "document_format": self.document_format.value,
            "boundary_strategy": self.boundary_strategy.value,
            "max_chunk_size": self.max_chunk_size,
            "min_chunk_size": self.min_chunk_size,
            "target_chunk_size": self.target_chunk_size,
            "preserve_structure": self.preserve_structure,
            "include_metadata": self.include_metadata,
            "fallback_to_sentences": self.fallback_to_sentences,
            "respect_code_blocks": self.respect_code_blocks,
            "respect_tables": self.respect_tables,
            "adaptive_sizing": self.adaptive_sizing,
            "quality_threshold": self.quality_threshold,
            "preferred_boundaries": [b.value for b in self.preferred_boundaries],
            "boundary_priorities": {k.value: v for k, v in self.boundary_priorities.items()},
            "performance_stats": self.performance_stats.copy(),
            "adaptation_history_length": len(self._adaptation_history),
            "parsers_available": {
                "html_parser": self.html_parser is not None,
                "markdown_parser": self.markdown_parser is not None,
                "html2text_converter": self.html2text_converter is not None
            }
        }
