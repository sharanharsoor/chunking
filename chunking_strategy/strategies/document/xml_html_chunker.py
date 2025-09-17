"""
XML/HTML-specific chunking strategy with tag-aware processing.

This module provides specialized chunking for XML and HTML documents, supporting:
- Tag-based chunking (by specific elements like div, section, article)
- Semantic HTML5 chunking (header, main, article, section)
- Hierarchical chunking (respecting nested element structure)
- Attribute-based grouping (by CSS classes, IDs, custom attributes)
- Content preservation (code blocks, tables, lists, media)
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING
from xml.etree import ElementTree as ET
from html.parser import HTMLParser

try:
    from bs4 import BeautifulSoup, Tag, NavigableString
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    # Create dummy classes for type hints
    if TYPE_CHECKING:
        from bs4 import BeautifulSoup, Tag, NavigableString
    else:
        BeautifulSoup = Tag = NavigableString = None

try:
    from lxml import etree, html
    HAS_LXML = True
except ImportError:
    HAS_LXML = False

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.core.streaming import StreamableChunker
from chunking_strategy.core.adaptive import AdaptableChunker


class SimpleHTMLParser(HTMLParser):
    """Simple HTML parser for basic tag extraction when BeautifulSoup is not available."""

    def __init__(self):
        super().__init__()
        self.elements = []
        self.current_element = None
        self.tag_stack = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        element = {
            'tag': tag,
            'attrs': attrs_dict,
            'start_pos': self.getpos(),
            'content': '',
            'children': []
        }

        if self.tag_stack:
            self.tag_stack[-1]['children'].append(element)
        else:
            self.elements.append(element)

        self.tag_stack.append(element)
        self.current_element = element

    def handle_endtag(self, tag):
        if self.tag_stack and self.tag_stack[-1]['tag'] == tag:
            element = self.tag_stack.pop()
            element['end_pos'] = self.getpos()
            if self.tag_stack:
                self.current_element = self.tag_stack[-1]
            else:
                self.current_element = None

    def handle_data(self, data):
        if self.current_element is not None:
            self.current_element['content'] += data


@register_chunker(
    name="xml_html_chunker",
    category="document",
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.MEDIUM,
    supported_formats=[".xml", ".html", ".htm", ".xhtml", ".svg", ".xsd", ".wsdl"],
    dependencies=["beautifulsoup4", "lxml"],
    optional_dependencies=["beautifulsoup4", "lxml"],
    description="XML/HTML chunker with tag-aware and semantic chunking strategies",
    use_cases=["web_content", "xml_documents", "configuration_files", "markup_processing"],
    best_for=["structured_markup", "semantic_html", "web_scraping", "document_parsing"],
    limitations=["requires_well_formed_markup", "complex_nested_structures"]
)
class XMLHTMLChunker(StreamableChunker, AdaptableChunker):
    """
    Specialized chunker for XML and HTML documents supporting:
    - Tag-based chunking (by specific elements)
    - Semantic HTML5 chunking (header, main, article, section)
    - Hierarchical chunking (respecting nested structure)
    - Attribute-based grouping (by classes, IDs, attributes)
    - Content type preservation (code, tables, media)
    """

    def __init__(
        self,
        chunk_by: str = "hierarchy",  # "semantic", "tag_based", "hierarchy", "element_type", "attribute_based", "content_size"
        target_tags: Optional[List[str]] = None,  # Specific tags to chunk by
        semantic_tags: Optional[List[str]] = None,  # HTML5 semantic tags
        max_depth: Optional[int] = None,  # Maximum hierarchy depth
        group_by_attribute: Optional[str] = None,  # Attribute to group by (class, id, etc.)
        preserve_structure: bool = True,  # Preserve nested structure
        include_attributes: bool = True,  # Include element attributes in metadata
        chunk_size: int = 2000,  # Target chunk size for content_size mode
        chunk_overlap: int = 200,
        min_chunk_size: int = 20,  # Lower default for XML/HTML elements
        max_chunk_size: int = 8000,
        parser: str = "auto",  # "auto", "bs4", "lxml", "builtin"
        encoding: str = "utf-8",
        **kwargs
    ):
        """
        Initialize XML/HTML chunker.

        Args:
            chunk_by: Chunking strategy
            target_tags: Specific tags to split on
            semantic_tags: HTML5 semantic tags to use
            max_depth: Maximum nesting depth to consider
            group_by_attribute: Attribute name to group by
            preserve_structure: Whether to preserve nested structure
            include_attributes: Include element attributes in metadata
            chunk_size: Target chunk size for content_size mode
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            parser: Parser backend to use
            encoding: Character encoding
            **kwargs: Additional parameters
        """
        super().__init__(
            name="xml_html_chunker",
            category="document",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        self.chunk_by = chunk_by
        self.target_tags = target_tags or self._get_default_target_tags()
        self.semantic_tags = semantic_tags or self._get_default_semantic_tags()
        self.max_depth = max_depth
        self.group_by_attribute = group_by_attribute
        self.preserve_structure = preserve_structure
        self.include_attributes = include_attributes
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.parser = parser
        self.encoding = encoding

        self.logger = logging.getLogger(__name__)

        # Choose parser backend
        self._setup_parser()

    def _get_default_target_tags(self) -> List[str]:
        """Get default tags for tag-based chunking."""
        return [
            'div', 'section', 'article', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'pre', 'code', 'blockquote'
        ]

    def _get_default_semantic_tags(self) -> List[str]:
        """Get default HTML5 semantic tags."""
        return [
            'header', 'nav', 'main', 'section', 'article', 'aside', 'footer',
            'figure', 'figcaption', 'details', 'summary'
        ]

    def _setup_parser(self) -> None:
        """Setup the parser backend based on availability and preference."""
        if self.parser == "auto":
            if HAS_LXML:
                self.parser_backend = "lxml"
            elif HAS_BS4:
                self.parser_backend = "bs4"
            else:
                self.parser_backend = "builtin"
        else:
            self.parser_backend = self.parser

        self.logger.info(f"Using parser backend: {self.parser_backend}")

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk XML/HTML content using specified strategy.

        Args:
            content: XML/HTML content (string, bytes, or file path)
            source_info: Source information metadata
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with chunks and metadata
        """
        start_time = time.time()

        # Handle different input types
        if isinstance(content, Path):
            file_path = content
            with open(file_path, 'r', encoding=self.encoding) as f:
                markup_content = f.read()
            source_info = source_info or {}
            source_info.update({
                "source": str(file_path),
                "source_type": "file"
            })
        elif isinstance(content, bytes):
            markup_content = content.decode(self.encoding)
        elif isinstance(content, str):
            # Check if it's a file path or actual content
            if (len(content) < 300 and '\n' not in content and
                content.strip() and not content.isspace() and
                Path(content).exists() and Path(content).is_file()):
                file_path = Path(content)
                with open(file_path, 'r', encoding=self.encoding) as f:
                    markup_content = f.read()
                source_info = source_info or {}
                source_info.update({
                    "source": str(file_path),
                    "source_type": "file"
                })
            else:
                markup_content = content
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        source_info = source_info or {"source": "unknown", "source_type": "content"}

        # Handle empty or whitespace-only content
        if not markup_content or not markup_content.strip():
            processing_time = time.time() - start_time
            return ChunkingResult(
                chunks=[],
                processing_time=processing_time,
                source_info={
                    "markup_structure": {"total_elements": 0, "element_distribution": {}, "max_depth": 0, "structure_type": "empty"},
                    "format_detected": "unknown",
                    "chunking_method": self.chunk_by,
                    **source_info
                },
                strategy_used="xml_html_chunker"
            )

        # Detect if it's HTML or XML
        format_type = self._detect_format(markup_content)

        # Parse the markup
        parsed_tree = self._parse_markup(markup_content, format_type)
        if parsed_tree is None:
            # Fallback to text-based chunking if parsing fails
            return self._fallback_text_chunking(markup_content, source_info, start_time)

        # Choose chunking strategy
        if self.chunk_by == "semantic":
            chunks = self._chunk_by_semantic(parsed_tree, format_type, source_info)
        elif self.chunk_by == "tag_based":
            chunks = self._chunk_by_tags(parsed_tree, format_type, source_info)
        elif self.chunk_by == "hierarchy":
            chunks = self._chunk_by_hierarchy(parsed_tree, format_type, source_info)
        elif self.chunk_by == "element_type":
            chunks = self._chunk_by_element_type(parsed_tree, format_type, source_info)
        elif self.chunk_by == "attribute_based":
            chunks = self._chunk_by_attributes(parsed_tree, format_type, source_info)
        elif self.chunk_by == "content_size":
            chunks = self._chunk_by_content_size(parsed_tree, format_type, source_info)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunk_by}")

        processing_time = time.time() - start_time

        # Analyze structure
        structure_info = self._analyze_structure(parsed_tree)

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            source_info={
                "markup_structure": structure_info,
                "format_detected": format_type,
                "chunking_method": self.chunk_by,
                "parser_used": self.parser_backend,
                **source_info
            },
            strategy_used="xml_html_chunker"
        )

    def _detect_format(self, content: str) -> str:
        """Detect if content is HTML or XML."""
        content_lower = content.lower().strip()

        # Check for HTML doctype or common HTML patterns
        if (content_lower.startswith('<!doctype html') or
            '<html' in content_lower or
            '<head>' in content_lower or
            '<body>' in content_lower or
            re.search(r'<(div|p|span|a|img|table)\b', content_lower)):
            return "html"

        # Check for XML declaration
        if content_lower.startswith('<?xml'):
            return "xml"

        # Check for common XML patterns
        if (content_lower.startswith('<') and
            not any(tag in content_lower for tag in ['<html', '<head', '<body', '<div', '<p', '<span'])):
            return "xml"

        # Default to HTML if unclear
        return "html"

    def _parse_markup(self, content: str, format_type: str) -> Optional[Any]:
        """Parse markup content using the selected backend."""
        try:
            if self.parser_backend == "lxml":
                if format_type == "html":
                    return html.fromstring(content)
                else:
                    return etree.fromstring(content.encode(self.encoding))
            elif self.parser_backend == "bs4":
                parser = "html.parser" if format_type == "html" else "xml"
                return BeautifulSoup(content, parser)
            else:
                # Use builtin XML parser
                if format_type == "html":
                    # Use simple HTML parser
                    parser = SimpleHTMLParser()
                    parser.feed(content)
                    return parser.elements
                else:
                    return ET.fromstring(content)
        except Exception as e:
            self.logger.warning(f"Failed to parse markup with {self.parser_backend}: {e}")
            return None

    def _analyze_structure(self, parsed_tree: Any) -> Dict[str, Any]:
        """Analyze the structure of the markup document."""
        if self.parser_backend == "bs4":
            return self._analyze_bs4_structure(parsed_tree)
        elif self.parser_backend == "lxml":
            return self._analyze_lxml_structure(parsed_tree)
        else:
            return self._analyze_builtin_structure(parsed_tree)

    def _analyze_bs4_structure(self, soup: "BeautifulSoup") -> Dict[str, Any]:
        """Analyze structure using BeautifulSoup."""
        elements = soup.find_all()
        element_counts = {}
        max_depth = 0

        for element in elements:
            if hasattr(element, 'name') and element.name:
                element_counts[element.name] = element_counts.get(element.name, 0) + 1

                # Calculate depth
                depth = len(list(element.parents))
                max_depth = max(max_depth, depth)

        return {
            "total_elements": len(elements),
            "element_distribution": element_counts,
            "max_depth": max_depth,
            "structure_type": "hierarchical" if max_depth > 2 else "flat"
        }

    def _analyze_lxml_structure(self, tree) -> Dict[str, Any]:
        """Analyze structure using lxml."""
        elements = tree.xpath(".//*")
        element_counts = {}
        max_depth = 0

        for element in elements:
            tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            element_counts[tag_name] = element_counts.get(tag_name, 0) + 1

            # Calculate depth
            depth = len(list(element.iterancestors()))
            max_depth = max(max_depth, depth)

        return {
            "total_elements": len(elements),
            "element_distribution": element_counts,
            "max_depth": max_depth,
            "structure_type": "hierarchical" if max_depth > 2 else "flat"
        }

    def _analyze_builtin_structure(self, tree) -> Dict[str, Any]:
        """Analyze structure using builtin parser."""
        if isinstance(tree, list):
            # Simple HTML parser result
            total_elements = len(tree)
            element_counts = {}
            for element in tree:
                if isinstance(element, dict) and 'tag' in element:
                    tag = element['tag']
                    element_counts[tag] = element_counts.get(tag, 0) + 1

            return {
                "total_elements": total_elements,
                "element_distribution": element_counts,
                "max_depth": 1,
                "structure_type": "flat"
            }
        else:
            # ET result
            elements = list(tree.iter())
            element_counts = {}
            max_depth = 0

            for element in elements:
                tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
                element_counts[tag_name] = element_counts.get(tag_name, 0) + 1

                # Calculate depth (simplified)
                depth = len(element.tag.split('/'))
                max_depth = max(max_depth, depth)

            return {
                "total_elements": len(elements),
                "element_distribution": element_counts,
                "max_depth": max_depth,
                "structure_type": "hierarchical" if max_depth > 2 else "flat"
            }

    def _chunk_by_semantic(
        self,
        parsed_tree: Any,
        format_type: str,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by HTML5 semantic elements."""
        chunks = []

        if self.parser_backend == "bs4":
            elements = parsed_tree.find_all(self.semantic_tags)
        elif self.parser_backend == "lxml":
            xpath_query = " | ".join([f".//{tag}" for tag in self.semantic_tags])
            elements = parsed_tree.xpath(xpath_query)
        else:
            # Fallback for builtin parser
            elements = self._find_elements_builtin(parsed_tree, self.semantic_tags)

        for i, element in enumerate(elements):
            content = self._extract_element_content(element)
            if content and len(content.strip()) >= self.min_chunk_size:
                chunk = self._create_semantic_chunk(element, content, i, source_info)
                chunks.append(chunk)

        return chunks

    def _chunk_by_tags(
        self,
        parsed_tree: Any,
        format_type: str,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by specific target tags."""
        chunks = []

        if self.parser_backend == "bs4":
            elements = parsed_tree.find_all(self.target_tags)
        elif self.parser_backend == "lxml":
            xpath_query = " | ".join([f".//{tag}" for tag in self.target_tags])
            elements = parsed_tree.xpath(xpath_query)
        else:
            elements = self._find_elements_builtin(parsed_tree, self.target_tags)

        for i, element in enumerate(elements):
            content = self._extract_element_content(element)
            if content and len(content.strip()) >= self.min_chunk_size:
                chunk = self._create_tag_chunk(element, content, i, source_info)
                chunks.append(chunk)

        return chunks

    def _chunk_by_hierarchy(
        self,
        parsed_tree: Any,
        format_type: str,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by hierarchical structure."""
        chunks = []

        def process_element(element, depth=0, index=0):
            if self.max_depth and depth > self.max_depth:
                return index

            content = self._extract_element_content(element, recursive=False)
            if content and len(content.strip()) >= self.min_chunk_size:
                chunk = self._create_hierarchy_chunk(element, content, index, depth, source_info)
                chunks.append(chunk)
                index += 1

            # Process children
            children = self._get_element_children(element)
            for child in children:
                index = process_element(child, depth + 1, index)

            return index

        if self.parser_backend == "bs4":
            root_elements = [elem for elem in parsed_tree.children if hasattr(elem, 'name')]
        elif self.parser_backend == "lxml":
            root_elements = list(parsed_tree)
        else:
            root_elements = parsed_tree if isinstance(parsed_tree, list) else [parsed_tree]

        for root in root_elements:
            process_element(root)

        return chunks

    def _chunk_by_element_type(
        self,
        parsed_tree: Any,
        format_type: str,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by grouping similar element types."""
        chunks = []
        element_groups = {}

        # Group elements by type
        if self.parser_backend == "bs4":
            all_elements = parsed_tree.find_all()
        elif self.parser_backend == "lxml":
            all_elements = parsed_tree.xpath(".//*")
        else:
            all_elements = self._get_all_elements_builtin(parsed_tree)

        for element in all_elements:
            tag_name = self._get_element_tag(element)
            if tag_name not in element_groups:
                element_groups[tag_name] = []
            element_groups[tag_name].append(element)

        # Create chunks for each element type
        chunk_index = 0
        for element_type, elements in element_groups.items():
            combined_content = ""
            element_count = 0

            for element in elements:
                content = self._extract_element_content(element)
                if content and content.strip():
                    combined_content += content + "\n\n"
                    element_count += 1

                    # Check if we should create a chunk
                    if len(combined_content) >= self.chunk_size:
                        chunk = self._create_element_type_chunk(
                            element_type, combined_content.strip(), chunk_index,
                            element_count, source_info
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        combined_content = ""
                        element_count = 0

            # Add remaining content
            if combined_content.strip() and len(combined_content.strip()) >= self.min_chunk_size:
                chunk = self._create_element_type_chunk(
                    element_type, combined_content.strip(), chunk_index,
                    element_count, source_info
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _chunk_by_attributes(
        self,
        parsed_tree: Any,
        format_type: str,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by grouping elements with similar attributes."""
        if not self.group_by_attribute:
            # Fallback to tag-based chunking
            return self._chunk_by_tags(parsed_tree, format_type, source_info)

        chunks = []
        attribute_groups = {}

        # Group elements by attribute value
        if self.parser_backend == "bs4":
            all_elements = parsed_tree.find_all()
        elif self.parser_backend == "lxml":
            all_elements = parsed_tree.xpath(".//*")
        else:
            all_elements = self._get_all_elements_builtin(parsed_tree)

        for element in all_elements:
            attr_value = self._get_element_attribute(element, self.group_by_attribute)
            if attr_value:
                if attr_value not in attribute_groups:
                    attribute_groups[attr_value] = []
                attribute_groups[attr_value].append(element)

        # Create chunks for each attribute group
        chunk_index = 0
        for attr_value, elements in attribute_groups.items():
            combined_content = ""

            for element in elements:
                content = self._extract_element_content(element)
                if content and content.strip():
                    combined_content += content + "\n\n"

            if combined_content.strip() and len(combined_content.strip()) >= self.min_chunk_size:
                chunk = self._create_attribute_chunk(
                    attr_value, combined_content.strip(), chunk_index,
                    len(elements), source_info
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _chunk_by_content_size(
        self,
        parsed_tree: Any,
        format_type: str,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by content size while respecting tag boundaries."""
        chunks = []
        current_chunk = ""
        current_elements = []
        chunk_index = 0

        # Get all leaf elements (elements with text content)
        if self.parser_backend == "bs4":
            text_elements = [elem for elem in parsed_tree.find_all() if elem.string]
        elif self.parser_backend == "lxml":
            text_elements = [elem for elem in parsed_tree.xpath(".//*") if elem.text]
        else:
            text_elements = self._get_text_elements_builtin(parsed_tree)

        for element in text_elements:
            content = self._extract_element_content(element)
            if not content or not content.strip():
                continue

            # Truncate content if it exceeds max_chunk_size
            if len(content) > self.max_chunk_size:
                content = content[:self.max_chunk_size]

            # Check if adding this element would exceed chunk size
            if len(current_chunk) + len(content) > self.chunk_size and current_chunk:
                # Ensure current chunk doesn't exceed max size
                if len(current_chunk) > self.max_chunk_size:
                    current_chunk = current_chunk[:self.max_chunk_size]

                # Create chunk from current content
                chunk = self._create_content_size_chunk(
                    current_chunk.strip(), chunk_index, current_elements, source_info
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap_content = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_content + "\n\n" + content
                else:
                    current_chunk = content
                current_elements = [element]
                chunk_index += 1
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + content
                else:
                    current_chunk = content
                current_elements.append(element)

        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            # Ensure final chunk doesn't exceed max size
            final_content = current_chunk.strip()
            if len(final_content) > self.max_chunk_size:
                final_content = final_content[:self.max_chunk_size]

            chunk = self._create_content_size_chunk(
                final_content, chunk_index, current_elements, source_info
            )
            chunks.append(chunk)

        return chunks

    def _extract_element_content(self, element, recursive: bool = True) -> str:
        """Extract text content from an element."""
        if self.parser_backend == "bs4":
            if recursive:
                return element.get_text(separator='\n', strip=True)
            else:
                return ''.join(element.strings) if hasattr(element, 'strings') else str(element.string or '')
        elif self.parser_backend == "lxml":
            if recursive:
                return etree.tostring(element, method='text', encoding='unicode').strip()
            else:
                return element.text or ''
        else:
            # Builtin parser
            if isinstance(element, dict):
                return element.get('content', '')
            elif hasattr(element, 'text'):
                return element.text or ''
            else:
                return str(element)

    def _get_element_tag(self, element) -> str:
        """Get the tag name of an element."""
        if self.parser_backend == "bs4":
            return element.name if hasattr(element, 'name') else 'text'
        elif self.parser_backend == "lxml":
            return element.tag.split('}')[-1] if '}' in element.tag else element.tag
        else:
            if isinstance(element, dict):
                return element.get('tag', 'unknown')
            elif hasattr(element, 'tag'):
                return element.tag
            else:
                return 'text'

    def _get_element_attribute(self, element, attr_name: str) -> Optional[str]:
        """Get an attribute value from an element."""
        if self.parser_backend == "bs4":
            return element.get(attr_name) if hasattr(element, 'get') else None
        elif self.parser_backend == "lxml":
            return element.get(attr_name)
        else:
            if isinstance(element, dict) and 'attrs' in element:
                return element['attrs'].get(attr_name)
            elif hasattr(element, 'get'):
                return element.get(attr_name)
            else:
                return None

    def _get_element_children(self, element) -> List[Any]:
        """Get child elements."""
        if self.parser_backend == "bs4":
            # Check if element has children attribute (Tags do, NavigableStrings don't)
            if hasattr(element, 'children'):
                return [child for child in element.children if hasattr(child, 'name')]
            else:
                return []
        elif self.parser_backend == "lxml":
            return list(element)
        else:
            if isinstance(element, dict):
                return element.get('children', [])
            else:
                return list(element) if hasattr(element, '__iter__') else []

    def _find_elements_builtin(self, tree, tags: List[str]) -> List[Any]:
        """Find elements by tags using builtin parser."""
        found = []

        def search_recursive(elements):
            for element in elements:
                if isinstance(element, dict) and element.get('tag') in tags:
                    found.append(element)
                if isinstance(element, dict) and 'children' in element:
                    search_recursive(element['children'])

        if isinstance(tree, list):
            search_recursive(tree)
        else:
            search_recursive([tree])

        return found

    def _get_all_elements_builtin(self, tree) -> List[Any]:
        """Get all elements using builtin parser."""
        elements = []

        def collect_recursive(nodes):
            for node in nodes:
                if isinstance(node, dict):
                    elements.append(node)
                    if 'children' in node:
                        collect_recursive(node['children'])

        if isinstance(tree, list):
            collect_recursive(tree)
        else:
            collect_recursive([tree])

        return elements

    def _get_text_elements_builtin(self, tree) -> List[Any]:
        """Get elements with text content using builtin parser."""
        text_elements = []

        def collect_text_recursive(nodes):
            for node in nodes:
                if isinstance(node, dict):
                    if node.get('content', '').strip():
                        text_elements.append(node)
                    if 'children' in node:
                        collect_text_recursive(node['children'])

        if isinstance(tree, list):
            collect_text_recursive(tree)
        else:
            collect_text_recursive([tree])

        return text_elements

    def _create_semantic_chunk(
        self,
        element: Any,
        content: str,
        index: int,
        source_info: Dict[str, Any]
    ) -> Chunk:
        """Create a semantic-based chunk."""
        tag_name = self._get_element_tag(element)
        attributes = self._get_element_attributes(element) if self.include_attributes else {}

        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"semantic element {index + 1}",
            length=len(content),
            extra={
                "element_tag": tag_name,
                "element_attributes": attributes,
                "semantic_index": index,
                "chunk_type": "semantic",
                "chunking_strategy": "semantic"
            }
        )

        return Chunk(
            id=f"xml_html_semantic_{index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _create_tag_chunk(
        self,
        element: Any,
        content: str,
        index: int,
        source_info: Dict[str, Any]
    ) -> Chunk:
        """Create a tag-based chunk."""
        tag_name = self._get_element_tag(element)
        attributes = self._get_element_attributes(element) if self.include_attributes else {}

        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"tag element {index + 1}",
            length=len(content),
            extra={
                "element_tag": tag_name,
                "element_attributes": attributes,
                "tag_index": index,
                "chunk_type": "tag_based",
                "chunking_strategy": "tag_based"
            }
        )

        return Chunk(
            id=f"xml_html_tag_{index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _create_hierarchy_chunk(
        self,
        element: Any,
        content: str,
        index: int,
        depth: int,
        source_info: Dict[str, Any]
    ) -> Chunk:
        """Create a hierarchy-based chunk."""
        tag_name = self._get_element_tag(element)
        attributes = self._get_element_attributes(element) if self.include_attributes else {}

        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"hierarchy level {depth}, element {index + 1}",
            length=len(content),
            extra={
                "element_tag": tag_name,
                "element_attributes": attributes,
                "hierarchy_depth": depth,
                "hierarchy_index": index,
                "chunk_type": "hierarchy",
                "chunking_strategy": "hierarchy"
            }
        )

        return Chunk(
            id=f"xml_html_hierarchy_{depth}_{index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _create_element_type_chunk(
        self,
        element_type: str,
        content: str,
        index: int,
        element_count: int,
        source_info: Dict[str, Any]
    ) -> Chunk:
        """Create an element type-based chunk."""
        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"element type {element_type}, group {index + 1}",
            length=len(content),
            extra={
                "element_type": element_type,
                "element_count": element_count,
                "type_group_index": index,
                "chunk_type": "element_type",
                "chunking_strategy": "element_type"
            }
        )

        return Chunk(
            id=f"xml_html_type_{element_type}_{index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _create_attribute_chunk(
        self,
        attr_value: str,
        content: str,
        index: int,
        element_count: int,
        source_info: Dict[str, Any]
    ) -> Chunk:
        """Create an attribute-based chunk."""
        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"attribute {self.group_by_attribute}={attr_value}, group {index + 1}",
            length=len(content),
            extra={
                "group_attribute": self.group_by_attribute,
                "attribute_value": attr_value,
                "element_count": element_count,
                "attribute_group_index": index,
                "chunk_type": "attribute_based",
                "chunking_strategy": "attribute_based"
            }
        )

        return Chunk(
            id=f"xml_html_attr_{index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _create_content_size_chunk(
        self,
        content: str,
        index: int,
        elements: List[Any],
        source_info: Dict[str, Any]
    ) -> Chunk:
        """Create a content size-based chunk."""
        element_tags = [self._get_element_tag(elem) for elem in elements]

        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"content chunk {index + 1}",
            length=len(content),
            extra={
                "element_count": len(elements),
                "element_tags": element_tags,
                "content_chunk_index": index,
                "chunk_type": "content_size",
                "chunking_strategy": "content_size"
            }
        )

        return Chunk(
            id=f"xml_html_content_{index}",
            content=content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

    def _get_element_attributes(self, element) -> Dict[str, str]:
        """Get all attributes of an element."""
        if self.parser_backend == "bs4":
            return dict(element.attrs) if hasattr(element, 'attrs') else {}
        elif self.parser_backend == "lxml":
            return dict(element.attrib) if hasattr(element, 'attrib') else {}
        else:
            if isinstance(element, dict):
                return element.get('attrs', {})
            elif hasattr(element, 'attrib'):
                return dict(element.attrib)
            else:
                return {}

    def _fallback_text_chunking(
        self,
        content: str,
        source_info: Dict[str, Any],
        start_time: float
    ) -> ChunkingResult:
        """Fallback to simple text chunking when parsing fails."""
        # Simple text chunking by size
        chunks = []
        words = content.split()
        current_chunk = ""
        chunk_index = 0

        for word in words:
            if len(current_chunk) + len(word) + 1 > self.chunk_size and current_chunk:
                chunk = Chunk(
                    id=f"xml_html_fallback_{chunk_index}",
                    content=current_chunk.strip(),
                    metadata=ChunkMetadata(
                        source=source_info.get("source", "unknown"),
                        source_type=source_info.get("source_type", "content"),
                        position=f"fallback chunk {chunk_index + 1}",
                        length=len(current_chunk.strip()),
                        extra={
                            "chunk_type": "fallback_text",
                            "chunking_strategy": "fallback",
                            "parsing_failed": True
                        }
                    ),
                    modality=ModalityType.TEXT
                )
                chunks.append(chunk)

                current_chunk = word
                chunk_index += 1
            else:
                current_chunk += " " + word if current_chunk else word

        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                id=f"xml_html_fallback_{chunk_index}",
                content=current_chunk.strip(),
                metadata=ChunkMetadata(
                    source=source_info.get("source", "unknown"),
                    source_type=source_info.get("source_type", "content"),
                    position=f"fallback chunk {chunk_index + 1}",
                    length=len(current_chunk.strip()),
                    extra={
                        "chunk_type": "fallback_text",
                        "chunking_strategy": "fallback",
                        "parsing_failed": True
                    }
                ),
                modality=ModalityType.TEXT
            )
            chunks.append(chunk)

        processing_time = time.time() - start_time

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            source_info={
                "markup_structure": {"parsing_failed": True},
                "format_detected": "unknown",
                "chunking_method": "fallback",
                **source_info
            },
            strategy_used="xml_html_chunker"
        )

    # Streaming and adaptation methods
    def can_stream(self) -> bool:
        """Check if this chunker supports streaming."""
        return True

    def chunk_stream(self, content_stream, source_info=None, **kwargs):
        """Stream chunks as content becomes available."""
        # For XML/HTML, we need the full content to properly parse the structure
        # So we collect all content first then process
        full_content = ""
        for chunk in content_stream:
            if isinstance(chunk, bytes):
                chunk = chunk.decode(self.encoding)
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
                "max_depth": self.max_depth,
                "max_chunk_size": self.max_chunk_size
            }
        }

        # Apply adaptations based on feedback score
        if feedback_score < 0.5:  # Poor performance, adjust parameters
            if feedback_type == "quality":
                # Reduce chunk size for better granularity
                self.chunk_size = max(500, int(self.chunk_size * 0.8))
                if self.max_depth:
                    self.max_depth = max(1, self.max_depth - 1)
            elif feedback_type == "performance":
                # Reduce max chunk size for better performance
                self.max_chunk_size = max(1000, int(self.max_chunk_size * 0.7))
        elif feedback_score > 0.8:  # Good performance, can increase sizes
            if feedback_type == "quality":
                self.chunk_size = min(5000, int(self.chunk_size * 1.2))
                if self.max_depth:
                    self.max_depth = min(10, self.max_depth + 1)
            elif feedback_type == "performance":
                self.max_chunk_size = min(15000, int(self.max_chunk_size * 1.3))

        # Handle specific feedback
        feedback_kwargs = kwargs
        if feedback_kwargs.get("chunks_too_large"):
            self.chunk_size = max(500, int(self.chunk_size * 0.8))
            self.max_chunk_size = max(1000, int(self.max_chunk_size * 0.8))
        elif feedback_kwargs.get("chunks_too_small"):
            self.chunk_size = min(5000, int(self.chunk_size * 1.2))
            self.max_chunk_size = min(15000, int(self.max_chunk_size * 1.2))

        # Record new config
        adaptation_record["new_config"] = {
            "chunk_size": self.chunk_size,
            "max_depth": self.max_depth,
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
