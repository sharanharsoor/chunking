"""
CSS/SCSS code chunking strategy.

Provides intelligent chunking for CSS and SCSS/Sass stylesheets that preserves:
- CSS rules and selectors
- Media queries and responsive breakpoints
- At-rules (@import, @font-face, @keyframes, etc.)
- SCSS/Sass features (variables, mixins, nesting)
- Comment blocks and documentation
- Logical style groupings
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.core.streaming import StreamableChunker
from chunking_strategy.core.adaptive import AdaptableChunker


@register_chunker(
    name="css_code",
    category="code",
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    supported_formats=["css", "scss", "sass", "less"],
    dependencies=[],
    description="Intelligent CSS/SCSS chunking preserving rules, media queries, and at-rules",
    use_cases=["style_analysis", "css_optimization", "responsive_design", "sass_processing"]
)
class CSSChunker(StreamableChunker, AdaptableChunker):
    """
    Intelligent CSS/SCSS chunker that preserves stylistic boundaries.
    
    Features:
    - CSS rule and selector detection
    - Media query boundary preservation
    - At-rule identification (@import, @font-face, @keyframes)
    - SCSS/Sass feature support (variables, mixins, nesting)
    - Comment and documentation preservation
    - Responsive design pattern detection
    """

    def __init__(
        self,
        chunk_by: str = "rule",  # "rule", "media_query", "at_rule", "selector_type", "logical_block", "line_count"
        max_lines_per_chunk: int = 100,
        include_imports: bool = True,
        include_media_queries: bool = True,
        handle_scss: bool = True,
        preserve_comments: bool = True,
        group_responsive: bool = True,
        **kwargs
    ):
        """
        Initialize CSS/SCSS code chunker.

        Args:
            chunk_by: Chunking granularity ("rule", "media_query", "at_rule", "selector_type", "logical_block", "line_count")
            max_lines_per_chunk: Maximum lines per chunk for line_count mode
            include_imports: Whether to include @import statements
            include_media_queries: Whether to include @media blocks
            handle_scss: Whether to handle SCSS/Sass syntax
            preserve_comments: Whether to preserve comments
            group_responsive: Whether to group responsive design patterns
            **kwargs: Additional parameters
        """
        super().__init__(
            name="css_code",
            category="code",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )
        
        self.chunk_by = chunk_by
        self.max_lines_per_chunk = max_lines_per_chunk
        self.include_imports = include_imports
        self.include_media_queries = include_media_queries
        self.handle_scss = handle_scss
        self.preserve_comments = preserve_comments
        self.group_responsive = group_responsive
        
        self.logger = logging.getLogger(__name__)
        self._adaptation_history = []

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk CSS/SCSS code content preserving stylistic boundaries.

        Args:
            content: CSS/SCSS code content or file path
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with CSS/SCSS code chunks
        """
        start_time = time.time()
        
        # Handle input types
        if isinstance(content, Path):
            source_path = content
            with open(content, 'r', encoding='utf-8') as f:
                code_content = f.read()
        elif isinstance(content, str) and len(content) > 0 and len(content) < 500 and '\n' not in content:
            try:
                if Path(content).exists() and Path(content).is_file():
                    source_path = Path(content)
                    with open(source_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                else:
                    source_path = None
                    code_content = str(content)
            except (OSError, ValueError):
                source_path = None
                code_content = str(content)
        else:
            source_path = None
            code_content = str(content)

        # Detect file type
        is_scss = self._is_scss_file(source_path, code_content)
        is_sass = self._is_sass_file(source_path, code_content)
        
        try:
            # Parse CSS/SCSS code
            lines = code_content.split('\n')
            css_elements = self._extract_css_elements(lines, is_scss, is_sass)
            
            # Create chunks based on strategy
            chunks = self._create_chunks_from_elements(
                css_elements, 
                lines, 
                source_path or "direct_input",
                is_scss,
                is_sass
            )
            
            # Create chunking result
            result = ChunkingResult(
                chunks=chunks,
                processing_time=time.time() - start_time,
                strategy_used="css_code",
                source_info={
                    "source_file": str(source_path) if source_path else "direct_input",
                    "chunk_by": self.chunk_by,
                    "total_elements": len(css_elements),
                    "total_lines": len(lines),
                    "is_scss": is_scss,
                    "is_sass": is_sass
                }
            )
            
            self.logger.info(f"CSS/SCSS code chunking completed: {len(chunks)} chunks from {source_path or 'direct input'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing CSS/SCSS code: {e}")
            # Fallback to line-based chunking
            return self._fallback_line_chunking(code_content, source_path, start_time)

    def _is_scss_file(self, source_path: Optional[Path], content: str) -> bool:
        """Check if this is an SCSS file."""
        if source_path and source_path.suffix in ['.scss']:
            return True
        
        # Check for SCSS syntax patterns
        scss_patterns = [
            r'\$\w+\s*:',  # Variables: $variable: value
            r'@mixin\s+\w+',  # Mixins: @mixin name
            r'@include\s+\w+',  # Includes: @include mixin
            r'@extend\s+',  # Extends: @extend selector
            r'&:',  # Parent selector: &:hover
            r'@if\s+',  # Conditionals: @if condition
            r'@for\s+',  # Loops: @for $i from 1 through 10
        ]
        
        for pattern in scss_patterns:
            if re.search(pattern, content):
                return True
                
        return False

    def _is_sass_file(self, source_path: Optional[Path], content: str) -> bool:
        """Check if this is a Sass file (indented syntax)."""
        if source_path and source_path.suffix in ['.sass']:
            return True
        
        # Sass uses indentation instead of braces - basic detection
        lines = content.split('\n')
        indented_rules = 0
        total_rules = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not stripped.startswith('/*'):
                total_rules += 1
                if line.startswith('  ') or line.startswith('\t'):
                    indented_rules += 1
        
        # If more than 60% of rules are indented, likely Sass
        return total_rules > 0 and (indented_rules / total_rules) > 0.6

    def _extract_css_elements(self, lines: List[str], is_scss: bool, is_sass: bool) -> List[Dict[str, Any]]:
        """Extract CSS/SCSS elements."""
        elements = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Extract comments
            if line.startswith('/*') and self.preserve_comments:
                element = self._extract_comment_block(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue
            
            # Skip single-line comments
            if line.startswith('//'):
                i += 1
                continue
            
            # Extract @import statements
            if line.startswith('@import') and self.include_imports:
                element = self._extract_import(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue
            
            # Extract media queries
            if line.startswith('@media') and self.include_media_queries:
                element = self._extract_media_query(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue
            
            # Extract at-rules (@font-face, @keyframes, etc.)
            if line.startswith('@') and not line.startswith('@media') and not line.startswith('@import'):
                element = self._extract_at_rule(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue
            
            # Extract SCSS variables
            if is_scss and line.startswith('$'):
                element = self._extract_scss_variable(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue
            
            # Extract SCSS mixins
            if is_scss and line.startswith('@mixin'):
                element = self._extract_scss_mixin(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue
            
            # Extract CSS rules
            if self._is_css_rule_start(line):
                element = self._extract_css_rule(lines, i, is_scss)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue
            
            i += 1
        
        return elements

    def _extract_comment_block(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract CSS comment block."""
        comment_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            comment_lines.append(line)
            
            if '*/' in line:
                content = '\n'.join(comment_lines)
                return {
                    'type': 'comment',
                    'name': 'comment_block',
                    'content': content,
                    'lineno': start_idx + 1,
                    'end_lineno': i + 1
                }
            i += 1
        
        return None

    def _extract_import(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract @import statement."""
        line = lines[start_idx].strip()
        
        # Handle multi-line imports
        import_content = line
        i = start_idx
        
        while i < len(lines) and not import_content.rstrip().endswith(';'):
            i += 1
            if i < len(lines):
                import_content += '\n' + lines[i].strip()
        
        # Extract import URL/path
        import_match = re.search(r'@import\s+(["\']?)([^"\';\s]+)\1', import_content)
        import_url = import_match.group(2) if import_match else 'unknown'
        
        return {
            'type': 'import',
            'name': 'import',
            'content': import_content,
            'lineno': start_idx + 1,
            'end_lineno': i + 1,
            'import_url': import_url
        }

    def _extract_media_query(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract @media query block."""
        first_line = lines[start_idx].strip()
        
        # Extract media condition
        media_match = re.match(r'@media\s+(.+?)\s*{', first_line)
        if not media_match:
            return None
        
        media_condition = media_match.group(1).strip()
        
        # Find media query end
        brace_count = 0
        media_lines = []
        in_media = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            media_lines.append(line)
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_media = True
                elif char == '}':
                    brace_count -= 1
                    
                    if in_media and brace_count == 0:
                        content = '\n'.join(media_lines)
                        
                        # Extract rules within media query
                        inner_rules = self._extract_rules_from_content('\n'.join(media_lines[1:-1]))
                        
                        return {
                            'type': 'media_query',
                            'name': f'media_{media_condition}',
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'media_condition': media_condition,
                            'inner_rules': inner_rules
                        }
        
        return None

    def _extract_at_rule(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract at-rule (@font-face, @keyframes, etc.)."""
        first_line = lines[start_idx].strip()
        
        # Extract at-rule type
        at_rule_match = re.match(r'@(\w+)', first_line)
        if not at_rule_match:
            return None
        
        at_rule_type = at_rule_match.group(1)
        
        # Extract name if present
        name_match = re.match(r'@\w+\s+([^{;]+)', first_line)
        at_rule_name = name_match.group(1).strip() if name_match else at_rule_type
        
        # Single line at-rule (like @charset)
        if first_line.endswith(';'):
            return {
                'type': 'at_rule',
                'name': at_rule_name,
                'content': first_line,
                'lineno': start_idx + 1,
                'end_lineno': start_idx + 1,
                'at_rule_type': at_rule_type
            }
        
        # Multi-line at-rule
        brace_count = 0
        at_rule_lines = []
        in_at_rule = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            at_rule_lines.append(line)
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_at_rule = True
                elif char == '}':
                    brace_count -= 1
                    
                    if in_at_rule and brace_count == 0:
                        content = '\n'.join(at_rule_lines)
                        
                        return {
                            'type': 'at_rule',
                            'name': at_rule_name,
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'at_rule_type': at_rule_type
                        }
        
        return None

    def _extract_scss_variable(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract SCSS variable declaration."""
        line = lines[start_idx].strip()
        
        # Variable: $name: value;
        var_match = re.match(r'(\$\w+)\s*:\s*([^;]+);?', line)
        if var_match:
            var_name = var_match.group(1)
            var_value = var_match.group(2).strip()
            
            return {
                'type': 'scss_variable',
                'name': var_name,
                'content': line,
                'lineno': start_idx + 1,
                'end_lineno': start_idx + 1,
                'variable_value': var_value
            }
        
        return None

    def _extract_scss_mixin(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract SCSS mixin definition."""
        first_line = lines[start_idx].strip()
        
        # Extract mixin name and parameters
        mixin_match = re.match(r'@mixin\s+(\w+)(\([^)]*\))?\s*{?', first_line)
        if not mixin_match:
            return None
        
        mixin_name = mixin_match.group(1)
        mixin_params = mixin_match.group(2) or ''
        
        # Find mixin end
        brace_count = 0
        mixin_lines = []
        in_mixin = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            mixin_lines.append(line)
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_mixin = True
                elif char == '}':
                    brace_count -= 1
                    
                    if in_mixin and brace_count == 0:
                        content = '\n'.join(mixin_lines)
                        
                        return {
                            'type': 'scss_mixin',
                            'name': mixin_name,
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'mixin_params': mixin_params
                        }
        
        return None

    def _is_css_rule_start(self, line: str) -> bool:
        """Check if line starts a CSS rule."""
        # CSS selector patterns
        patterns = [
            r'^[.#]?[\w-]+',  # Class, ID, or element
            r'^\*',  # Universal selector
            r'^:',  # Pseudo-selector
            r'^\[',  # Attribute selector
            r'^@',  # At-rule (already handled above)
        ]
        
        # Skip at-rules and variables
        if line.startswith('@') or line.startswith('$'):
            return False
        
        # Look for selector pattern followed by { or ,
        return bool(re.search(r'[^{}]*\{', line) or 
                   re.search(r'[^{}]*,\s*$', line) or
                   ('{' not in line and '}' not in line and line.strip()))

    def _extract_css_rule(self, lines: List[str], start_idx: int, is_scss: bool) -> Optional[Dict[str, Any]]:
        """Extract CSS rule (selector + declarations)."""
        # Collect selector lines
        selector_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            selector_lines.append(line)
            
            if '{' in line:
                break
            i += 1
        
        if i >= len(lines):
            return None
        
        # Extract selector
        selector_text = '\n'.join(selector_lines)
        selector = self._clean_selector(selector_text.split('{')[0].strip())
        
        # Find rule end
        brace_count = 0
        rule_lines = selector_lines.copy()
        
        # Count braces in selector lines
        for line in selector_lines:
            brace_count += line.count('{')
        
        # Continue from where we left off
        while i < len(lines) and brace_count > 0:
            i += 1
            if i < len(lines):
                line = lines[i]
                rule_lines.append(line)
                brace_count += line.count('{') - line.count('}')
        
        content = '\n'.join(rule_lines)
        
        # Analyze selector
        selector_info = self._analyze_selector(selector)
        
        # Extract properties
        properties = self._extract_properties(rule_lines)
        
        return {
            'type': 'css_rule',
            'name': selector,
            'content': content,
            'lineno': start_idx + 1,
            'end_lineno': i + 1,
            'selector': selector,
            'selector_type': selector_info['type'],
            'selector_specificity': selector_info['specificity'],
            'properties': properties,
            'property_count': len(properties)
        }

    def _clean_selector(self, selector: str) -> str:
        """Clean and normalize selector text."""
        # Remove extra whitespace and normalize
        return ' '.join(selector.split())

    def _analyze_selector(self, selector: str) -> Dict[str, Any]:
        """Analyze CSS selector to determine type and specificity."""
        selector_types = []
        specificity = {'id': 0, 'class': 0, 'element': 0}
        
        # Count ID selectors
        id_count = len(re.findall(r'#[\w-]+', selector))
        specificity['id'] = id_count
        if id_count > 0:
            selector_types.append('id')
        
        # Count class selectors
        class_count = len(re.findall(r'\.[\w-]+', selector))
        specificity['class'] = class_count
        if class_count > 0:
            selector_types.append('class')
        
        # Count element selectors
        element_count = len(re.findall(r'\b[a-z][\w-]*\b', selector))
        specificity['element'] = element_count
        if element_count > 0:
            selector_types.append('element')
        
        # Check for pseudo selectors
        if ':' in selector:
            selector_types.append('pseudo')
        
        # Check for attribute selectors
        if '[' in selector:
            selector_types.append('attribute')
        
        # Check for universal selector
        if '*' in selector:
            selector_types.append('universal')
        
        return {
            'type': selector_types[0] if selector_types else 'unknown',
            'types': selector_types,
            'specificity': specificity
        }

    def _extract_properties(self, rule_lines: List[str]) -> List[str]:
        """Extract CSS properties from rule."""
        properties = []
        
        # Find content between { and }
        content = '\n'.join(rule_lines)
        
        # Extract declaration block
        match = re.search(r'\{([^}]*)\}', content, re.DOTALL)
        if match:
            declarations = match.group(1)
            
            # Split by semicolon and extract property names
            for declaration in declarations.split(';'):
                declaration = declaration.strip()
                if ':' in declaration:
                    prop_name = declaration.split(':')[0].strip()
                    if prop_name:
                        properties.append(prop_name)
        
        return properties

    def _extract_rules_from_content(self, content: str) -> List[str]:
        """Extract rule selectors from content."""
        rules = []
        
        # Simple extraction - could be enhanced
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('/*') and '{' in line:
                selector = line.split('{')[0].strip()
                if selector:
                    rules.append(selector)
        
        return rules

    def _create_chunks_from_elements(
        self, 
        elements: List[Dict[str, Any]], 
        lines: List[str], 
        source: str,
        is_scss: bool,
        is_sass: bool
    ) -> List[Chunk]:
        """Create chunks from extracted CSS elements."""
        chunks = []
        
        if self.chunk_by == "rule":
            # Create one chunk per CSS rule
            for element in elements:
                if element['type'] in ['css_rule', 'import', 'comment']:
                    chunk = self._create_chunk_from_element(element, source, is_scss, is_sass)
                    chunks.append(chunk)
        
        elif self.chunk_by == "media_query":
            # Create one chunk per media query
            for element in elements:
                if element['type'] in ['media_query', 'css_rule', 'import']:
                    chunk = self._create_chunk_from_element(element, source, is_scss, is_sass)
                    chunks.append(chunk)
        
        elif self.chunk_by == "at_rule":
            # Create one chunk per at-rule
            for element in elements:
                if element['type'] in ['at_rule', 'media_query', 'scss_mixin', 'import']:
                    chunk = self._create_chunk_from_element(element, source, is_scss, is_sass)
                    chunks.append(chunk)
        
        elif self.chunk_by == "selector_type":
            # Group by selector type
            chunks = self._create_selector_type_chunks(elements, source, is_scss, is_sass)
        
        elif self.chunk_by == "logical_block":
            # Group related elements together
            chunks = self._create_logical_blocks(elements, source, is_scss, is_sass)
        
        elif self.chunk_by == "line_count":
            # Chunk by line count while respecting boundaries
            chunks = self._create_line_count_chunks(elements, lines, source, is_scss, is_sass)
        
        return chunks

    def _create_chunk_from_element(self, element: Dict[str, Any], source: str, is_scss: bool, is_sass: bool) -> Chunk:
        """Create a chunk from a single CSS element."""
        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": element['lineno'], "end_line": element['end_lineno']},
            chunker_used="css_code",
            extra={
                "chunk_type": "code",
                "language": "scss" if is_scss else ("sass" if is_sass else "css"),
                "element_type": element['type'],
                "element_name": element.get('name', 'unnamed'),
                "selector": element.get('selector'),
                "selector_type": element.get('selector_type'),
                "selector_specificity": element.get('selector_specificity'),
                "properties": element.get('properties', []),
                "property_count": element.get('property_count', 0),
                "media_condition": element.get('media_condition'),
                "at_rule_type": element.get('at_rule_type'),
                "import_url": element.get('import_url'),
                "variable_value": element.get('variable_value'),
                "mixin_params": element.get('mixin_params'),
                "inner_rules": element.get('inner_rules', []),
                "is_scss": is_scss,
                "is_sass": is_sass
            }
        )
        
        chunk_id = f"css_{element['type']}_{element.get('name', 'unnamed')}_{element['lineno']}"
        
        return Chunk(
            id=chunk_id,
            content=element['content'],
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_selector_type_chunks(self, elements: List[Dict[str, Any]], source: str, is_scss: bool, is_sass: bool) -> List[Chunk]:
        """Create chunks grouped by selector type."""
        # Group elements by selector type
        type_groups = {}
        
        for element in elements:
            if element['type'] == 'css_rule':
                selector_type = element.get('selector_type', 'unknown')
                if selector_type not in type_groups:
                    type_groups[selector_type] = []
                type_groups[selector_type].append(element)
            else:
                # Non-rule elements go to their own groups
                element_type = element['type']
                if element_type not in type_groups:
                    type_groups[element_type] = []
                type_groups[element_type].append(element)
        
        chunks = []
        for group_type, group_elements in type_groups.items():
            chunk = self._create_group_chunk(group_elements, source, group_type, len(chunks), is_scss, is_sass)
            chunks.append(chunk)
        
        return chunks

    def _create_logical_blocks(self, elements: List[Dict[str, Any]], source: str, is_scss: bool, is_sass: bool) -> List[Chunk]:
        """Create chunks by grouping logically related elements."""
        chunks = []
        current_block = []
        current_lines = 0
        
        for element in elements:
            element_lines = element['end_lineno'] - element['lineno'] + 1
            
            if current_lines + element_lines > self.max_lines_per_chunk and current_block:
                chunk = self._create_block_chunk(current_block, source, len(chunks), is_scss, is_sass)
                chunks.append(chunk)
                current_block = [element]
                current_lines = element_lines
            else:
                current_block.append(element)
                current_lines += element_lines
        
        if current_block:
            chunk = self._create_block_chunk(current_block, source, len(chunks), is_scss, is_sass)
            chunks.append(chunk)
        
        return chunks

    def _create_group_chunk(self, elements: List[Dict[str, Any]], source: str, group_type: str, chunk_index: int, is_scss: bool, is_sass: bool) -> Chunk:
        """Create a chunk from grouped elements."""
        content = '\n\n'.join(element['content'] for element in elements)
        
        start_line = min(element['lineno'] for element in elements)
        end_line = max(element['end_lineno'] for element in elements)
        
        element_names = [element.get('name', 'unnamed') for element in elements]
        
        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": start_line, "end_line": end_line},
            chunker_used="css_code",
            extra={
                "chunk_type": "code",
                "language": "scss" if is_scss else ("sass" if is_sass else "css"),
                "group_type": group_type,
                "element_names": element_names,
                "group_size": len(elements),
                "is_scss": is_scss,
                "is_sass": is_sass
            }
        )
        
        return Chunk(
            id=f"css_group_{group_type}_{chunk_index}_{start_line}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_block_chunk(self, elements: List[Dict[str, Any]], source: str, chunk_index: int, is_scss: bool, is_sass: bool) -> Chunk:
        """Create a chunk from multiple elements."""
        content = '\n\n'.join(element['content'] for element in elements)
        
        start_line = min(element['lineno'] for element in elements)
        end_line = max(element['end_lineno'] for element in elements)
        
        element_names = [element.get('name', 'unnamed') for element in elements]
        element_types = [element['type'] for element in elements]
        
        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": start_line, "end_line": end_line},
            chunker_used="css_code",
            extra={
                "chunk_type": "code",
                "language": "scss" if is_scss else ("sass" if is_sass else "css"),
                "element_types": element_types,
                "element_names": element_names,
                "block_size": len(elements),
                "is_scss": is_scss,
                "is_sass": is_sass
            }
        )
        
        return Chunk(
            id=f"css_block_{chunk_index}_{start_line}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_line_count_chunks(
        self, 
        elements: List[Dict[str, Any]], 
        lines: List[str], 
        source: str,
        is_scss: bool,
        is_sass: bool
    ) -> List[Chunk]:
        """Create chunks based on line count while respecting element boundaries."""
        chunks = []
        current_lines = []
        current_elements = []
        
        for element in elements:
            element_lines = element['end_lineno'] - element['lineno'] + 1
            
            if len(current_lines) + element_lines > self.max_lines_per_chunk and current_lines:
                chunk = self._create_line_chunk(current_lines, current_elements, source, len(chunks), is_scss, is_sass)
                chunks.append(chunk)
                current_lines = []
                current_elements = []
            
            element_content_lines = element['content'].split('\n')
            current_lines.extend(element_content_lines)
            current_elements.append(element)
        
        if current_lines:
            chunk = self._create_line_chunk(current_lines, current_elements, source, len(chunks), is_scss, is_sass)
            chunks.append(chunk)
        
        return chunks

    def _create_line_chunk(
        self, 
        lines: List[str], 
        elements: List[Dict[str, Any]], 
        source: str, 
        chunk_index: int,
        is_scss: bool,
        is_sass: bool
    ) -> Chunk:
        """Create a chunk from lines and associated elements."""
        content = '\n'.join(lines)
        
        start_line = min(element['lineno'] for element in elements) if elements else 1
        end_line = max(element['end_lineno'] for element in elements) if elements else len(lines)
        
        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": start_line, "end_line": end_line},
            chunker_used="css_code",
            extra={
                "chunk_type": "code",
                "language": "scss" if is_scss else ("sass" if is_sass else "css"),
                "line_count": len(lines),
                "elements_included": len(elements),
                "is_scss": is_scss,
                "is_sass": is_sass
            }
        )
        
        return Chunk(
            id=f"css_lines_{chunk_index}_{start_line}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _fallback_line_chunking(self, content: str, source_path: Optional[Path], start_time: float) -> ChunkingResult:
        """Fallback to simple line-based chunking."""
        lines = content.split('\n')
        chunks = []
        
        for i in range(0, len(lines), self.max_lines_per_chunk):
            chunk_lines = lines[i:i + self.max_lines_per_chunk]
            chunk_content = '\n'.join(chunk_lines)
            
            chunk = Chunk(
                id=f"css_fallback_{i // self.max_lines_per_chunk}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=str(source_path) if source_path else "direct_input",
                    position={"start_line": i + 1, "end_line": min(i + len(chunk_lines), len(lines))},
                    chunker_used="css_code",
                    extra={
                        "chunk_type": "code",
                        "language": "css",
                        "fallback": True
                    }
                )
            )
            chunks.append(chunk)
        
        return ChunkingResult(
            chunks=chunks,
            processing_time=time.time() - start_time,
            strategy_used="css_code_fallback"
        )

    def chunk_stream(self, content_stream, **kwargs):
        """Chunk CSS/SCSS code from a stream."""
        # Collect stream content
        content_lines = []
        for chunk in content_stream:
            if isinstance(chunk, str):
                content_lines.extend(chunk.split('\n'))
            elif isinstance(chunk, bytes):
                content_lines.extend(chunk.decode('utf-8').split('\n'))
        
        content = '\n'.join(content_lines)
        result = self.chunk(content, **kwargs)
        
        for chunk in result.chunks:
            yield chunk

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ["css", "scss", "sass", "less"]

    def estimate_chunks(self, content: Union[str, Path]) -> int:
        """Estimate number of chunks that will be generated."""
        if isinstance(content, (str, Path)):
            if isinstance(content, Path) and content.exists():
                with open(content, 'r') as f:
                    lines = len(f.readlines())
            else:
                lines = len(str(content).split('\n'))
            
            if self.chunk_by in ["rule", "media_query", "at_rule"]:
                return max(1, lines // 15)  # Estimate rules
            else:
                return max(1, lines // self.max_lines_per_chunk)
        
        return 1

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """Adapt chunker parameters based on feedback."""
        adaptation = {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "old_max_lines": self.max_lines_per_chunk
        }
        
        if feedback_type == "quality" and feedback_score < 0.5:
            old_max = self.max_lines_per_chunk
            self.max_lines_per_chunk = max(20, int(self.max_lines_per_chunk * 0.8))
            adaptation["new_max_lines"] = self.max_lines_per_chunk
            self.logger.info(f"Adapted max_lines_per_chunk: {old_max} -> {self.max_lines_per_chunk}")
        elif feedback_type == "performance" and feedback_score < 0.5:
            old_max = self.max_lines_per_chunk
            self.max_lines_per_chunk = min(500, int(self.max_lines_per_chunk * 1.2))
            adaptation["new_max_lines"] = self.max_lines_per_chunk
            self.logger.info(f"Adapted max_lines_per_chunk: {old_max} -> {self.max_lines_per_chunk}")
        
        self._adaptation_history.append(adaptation)

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self._adaptation_history.copy()
