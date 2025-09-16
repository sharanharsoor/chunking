"""
JavaScript/TypeScript code chunking strategy.

Provides intelligent chunking for JavaScript and TypeScript source code that preserves:
- Function declarations, expressions, and arrow functions
- Class definitions and methods
- Import/export statements
- Object methods and properties
- JSX components (React)
- TypeScript interfaces and types
- Logical code blocks
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
    name="javascript_code",
    category="code",
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    supported_formats=["js", "jsx", "ts", "tsx", "mjs", "cjs"],
    dependencies=[],
    description="Intelligent JavaScript/TypeScript code chunking preserving function and class boundaries",
    use_cases=["code_analysis", "documentation_generation", "code_search", "react_analysis"]
)
class JavaScriptChunker(StreamableChunker, AdaptableChunker):
    """
    Intelligent JavaScript/TypeScript code chunker that preserves syntactic boundaries.

    Features:
    - Function declaration vs expression vs arrow function detection
    - Class and method boundary preservation
    - Import/export statement grouping
    - Object method detection
    - JSX component extraction
    - TypeScript interface and type detection
    - Template literal handling
    - Comment and documentation preservation
    """

    def __init__(
        self,
        chunk_by: str = "function",  # "function", "class", "component", "logical_block", "line_count"
        max_lines_per_chunk: int = 100,
        include_imports: bool = True,
        include_exports: bool = True,
        handle_jsx: bool = True,
        handle_typescript: bool = True,
        preserve_comments: bool = True,
        **kwargs
    ):
        """
        Initialize JavaScript/TypeScript code chunker.

        Args:
            chunk_by: Chunking granularity ("function", "class", "component", "logical_block", "line_count")
            max_lines_per_chunk: Maximum lines per chunk for line_count mode
            include_imports: Whether to include import statements
            include_exports: Whether to include export statements
            handle_jsx: Whether to detect JSX components
            handle_typescript: Whether to handle TypeScript syntax
            preserve_comments: Whether to preserve comments
            **kwargs: Additional parameters
        """
        super().__init__(
            name="javascript_code",
            category="code",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        self.chunk_by = chunk_by
        self.max_lines_per_chunk = max_lines_per_chunk
        self.include_imports = include_imports
        self.include_exports = include_exports
        self.handle_jsx = handle_jsx
        self.handle_typescript = handle_typescript
        self.preserve_comments = preserve_comments

        self.logger = logging.getLogger(__name__)
        self._adaptation_history = []

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk JavaScript/TypeScript code content preserving syntactic boundaries.

        Args:
            content: JavaScript/TypeScript code content or file path
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with JavaScript/TypeScript code chunks
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

        # Detect file type for TypeScript/JSX handling
        is_typescript = self._is_typescript_file(source_path, code_content)
        is_jsx = self._is_jsx_file(source_path, code_content)

        try:
            # Parse JavaScript/TypeScript code
            lines = code_content.split('\n')
            code_elements = self._extract_code_elements(lines, is_typescript, is_jsx)

            # Create chunks based on strategy
            chunks = self._create_chunks_from_elements(
                code_elements,
                lines,
                source_path or "direct_input",
                is_typescript,
                is_jsx
            )

            # Create chunking result
            result = ChunkingResult(
                chunks=chunks,
                processing_time=time.time() - start_time,
                strategy_used="javascript_code",
                source_info={
                    "source_file": str(source_path) if source_path else "direct_input",
                    "chunk_by": self.chunk_by,
                    "total_elements": len(code_elements),
                    "total_lines": len(lines),
                    "is_typescript": is_typescript,
                    "is_jsx": is_jsx
                }
            )

            self.logger.info(f"JavaScript/TypeScript code chunking completed: {len(chunks)} chunks from {source_path or 'direct input'}")
            return result

        except Exception as e:
            self.logger.error(f"Error parsing JavaScript/TypeScript code: {e}")
            # Fallback to line-based chunking
            return self._fallback_line_chunking(code_content, source_path, start_time)

    def _is_typescript_file(self, source_path: Optional[Path], content: str) -> bool:
        """Check if this is a TypeScript file."""
        if source_path and source_path.suffix in ['.ts', '.tsx']:
            return True

        # Check for TypeScript syntax patterns
        ts_patterns = [
            r'\b(interface|type)\s+\w+',
            r':\s*\w+(\[\]|\<\w+\>)?(\s*=|\s*;|\s*,)',
            r'\bas\s+\w+',
            r'\b(public|private|protected|readonly)\s+',
            r'\<[A-Z]\w*\>',
        ]

        for pattern in ts_patterns:
            if re.search(pattern, content):
                return True

        return False

    def _is_jsx_file(self, source_path: Optional[Path], content: str) -> bool:
        """Check if this contains JSX."""
        if source_path and source_path.suffix in ['.jsx', '.tsx']:
            return True

        # Check for JSX patterns
        jsx_patterns = [
            r'<[A-Z]\w*[^>]*>',  # Component tags
            r'<\w+[^>]*\s+\w+={[^}]+}',  # Props with expressions
            r'return\s*\(',  # Common JSX return pattern
        ]

        for pattern in jsx_patterns:
            if re.search(pattern, content):
                return True

        return False

    def _extract_code_elements(self, lines: List[str], is_typescript: bool, is_jsx: bool) -> List[Dict[str, Any]]:
        """Extract JavaScript/TypeScript code elements."""
        elements = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and single-line comments
            if not line or line.startswith('//') or line.startswith('/*'):
                i += 1
                continue

            # Extract imports/exports
            if self._is_import_export(line):
                element = self._extract_import_export(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue

            # Extract class definitions
            if self._is_class_start(line):
                element = self._extract_class(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']  # element line numbers are 1-based, i is 0-based
                else:
                    i += 1
                continue

            # Extract function definitions (declarations, expressions, arrows)
            if self._is_function_start(line, lines, i):
                element = self._extract_function(lines, i, is_jsx)
                if element:
                    elements.append(element)
                    i = element['end_lineno']  # element line numbers are 1-based, i is 0-based
                else:
                    i += 1
                continue

            # Extract TypeScript interfaces/types
            if is_typescript and self._is_type_definition(line):
                element = self._extract_type_definition(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']  # element line numbers are 1-based, i is 0-based
                else:
                    i += 1
                continue

            # Extract JSX components
            if is_jsx and self._is_jsx_component(line, lines, i):
                element = self._extract_jsx_component(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']  # element line numbers are 1-based, i is 0-based
                else:
                    i += 1
                continue

            i += 1

        return elements

    def _is_import_export(self, line: str) -> bool:
        """Check if line is an import or export statement."""
        return bool(re.match(r'^\s*(import|export)\s+', line))

    def _extract_import_export(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract import/export statement(s)."""
        if not (self.include_imports or self.include_exports):
            return None

        line = lines[start_idx].strip()
        statement_type = 'import' if line.startswith('import') else 'export'

        # Handle multi-line imports/exports
        end_idx = start_idx
        while end_idx < len(lines) and not lines[end_idx].rstrip().endswith(';') and not lines[end_idx].rstrip().endswith('}'):
            end_idx += 1

        content = '\n'.join(lines[start_idx:end_idx + 1])

        return {
            'type': statement_type,
            'content': content,
            'lineno': start_idx + 1,
            'end_lineno': end_idx + 1,
            'name': f'{statement_type}_statements'
        }

    def _is_class_start(self, line: str) -> bool:
        """Check if line starts a class definition."""
        return bool(re.match(r'^\s*(export\s+)?(default\s+)?class\s+\w+', line))

    def _extract_class(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract class definition."""
        brace_count = 0
        class_lines = []
        in_class = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            class_lines.append(line)

            # Count braces to find class end
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_class = True
                elif char == '}':
                    brace_count -= 1

                    if in_class and brace_count == 0:
                        # Class complete
                        content = '\n'.join(class_lines)

                        # Extract class name
                        first_line = class_lines[0].strip()
                        name_match = re.search(r'class\s+(\w+)', first_line)
                        class_name = name_match.group(1) if name_match else 'unnamed'

                        # Extract extends
                        extends_match = re.search(r'extends\s+(\w+)', first_line)
                        extends = extends_match.group(1) if extends_match else None

                        return {
                            'type': 'class',
                            'name': class_name,
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'extends': extends,
                            'methods': self._extract_class_methods(class_lines)
                        }

        return None

    def _is_function_start(self, line: str, lines: List[str], idx: int) -> bool:
        """Check if line starts a function definition."""
        # Function declaration (including async)
        if re.match(r'^\s*(export\s+)?(default\s+)?(async\s+)?function\s+\w+', line):
            return True

        # Function expression
        if re.match(r'^\s*(const|let|var)\s+\w+\s*=\s*(async\s+)?function', line):
            return True

        # Arrow function
        if re.match(r'^\s*(const|let|var)\s+\w+\s*=\s*(async\s+)?\([^)]*\)\s*=>', line):
            return True

        # Object method
        if re.match(r'^\s*\w+\s*\([^)]*\)\s*{', line):
            return True

        # Class method
        if re.match(r'^\s*(async\s+)?(static\s+)?\w+\s*\([^)]*\)\s*{', line):
            return True

        return False

    def _extract_function(self, lines: List[str], start_idx: int, is_jsx: bool) -> Optional[Dict[str, Any]]:
        """Extract function definition."""
        first_line = lines[start_idx].strip()

        # Determine function type
        function_type = self._determine_function_type(first_line)

        # Extract function name
        function_name = self._extract_function_name(first_line, function_type)

        # Extract parameters
        params = self._extract_function_params(first_line)

        # Extract function body
        if '=>' in first_line and not '{' in first_line:
            # Single-line arrow function
            return {
                'type': 'arrow_function',
                'name': function_name,
                'content': first_line,
                'lineno': start_idx + 1,
                'end_lineno': start_idx + 1,
                'params': params,
                'is_async': 'async' in first_line
            }

        # Multi-line function - find end by brace matching
        brace_count = 0
        function_lines = []
        in_function = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            function_lines.append(line)

            for char in line:
                if char == '{':
                    brace_count += 1
                    in_function = True
                elif char == '}':
                    brace_count -= 1

                    if in_function and brace_count == 0:
                        content = '\n'.join(function_lines)

                        return {
                            'type': function_type,
                            'name': function_name,
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'params': params,
                            'is_async': 'async' in first_line,
                            'is_jsx_component': is_jsx and self._contains_jsx_return(function_lines)
                        }

        return None

    def _is_type_definition(self, line: str) -> bool:
        """Check if line starts a TypeScript type definition."""
        return bool(re.match(r'^\s*(export\s+)?(interface|type)\s+\w+', line))

    def _extract_type_definition(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract TypeScript interface or type definition."""
        first_line = lines[start_idx].strip()

        # Determine if it's interface or type
        type_kind = 'interface' if 'interface' in first_line else 'type'

        # Extract name
        name_match = re.search(r'(interface|type)\s+(\w+)', first_line)
        type_name = name_match.group(2) if name_match else 'unnamed'

        if type_kind == 'interface':
            # Interface - find end by brace matching
            brace_count = 0
            type_lines = []
            in_interface = False

            for i in range(start_idx, len(lines)):
                line = lines[i]
                type_lines.append(line)

                for char in line:
                    if char == '{':
                        brace_count += 1
                        in_interface = True
                    elif char == '}':
                        brace_count -= 1

                        if in_interface and brace_count == 0:
                            content = '\n'.join(type_lines)

                            return {
                                'type': 'interface',
                                'name': type_name,
                                'content': content,
                                'lineno': start_idx + 1,
                                'end_lineno': i + 1
                            }
        else:
            # Type alias - usually single line
            return {
                'type': 'type_alias',
                'name': type_name,
                'content': first_line,
                'lineno': start_idx + 1,
                'end_lineno': start_idx + 1
            }

        return None

    def _is_jsx_component(self, line: str, lines: List[str], idx: int) -> bool:
        """Check if this might be a JSX component function."""
        # Look for React component patterns
        if re.match(r'^\s*(const|function)\s+[A-Z]\w*', line):
            # Check next few lines for JSX return
            for i in range(idx, min(idx + 10, len(lines))):
                if 'return' in lines[i] and ('<' in lines[i] or '<' in lines[i + 1] if i + 1 < len(lines) else False):
                    return True
        return False

    def _extract_jsx_component(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract JSX component."""
        # This is similar to function extraction but with JSX-specific metadata
        element = self._extract_function(lines, start_idx, True)
        if element:
            element['type'] = 'jsx_component'
            element['is_jsx_component'] = True
        return element

    def _determine_function_type(self, line: str) -> str:
        """Determine the type of function declaration."""
        # Check for function declaration (including async, export, default)
        if re.match(r'^\s*(export\s+)?(default\s+)?(async\s+)?function\s+\w+', line):
            return 'function_declaration'
        elif '=>' in line:
            return 'arrow_function'
        elif 'function' in line and '=' in line:
            return 'function_expression'
        else:
            return 'method'

    def _extract_function_name(self, line: str, function_type: str) -> str:
        """Extract function name from declaration."""
        if function_type == 'function_declaration':
            match = re.search(r'function\s+(\w+)', line)
            return match.group(1) if match else 'anonymous'
        elif function_type in ['arrow_function', 'function_expression']:
            match = re.search(r'(const|let|var)\s+(\w+)', line)
            return match.group(2) if match else 'anonymous'
        else:  # method
            match = re.search(r'^\s*(\w+)\s*\(', line)
            return match.group(1) if match else 'anonymous'

    def _extract_function_params(self, line: str) -> List[str]:
        """Extract function parameters."""
        # Find parameter list in parentheses
        match = re.search(r'\(([^)]*)\)', line)
        if match:
            params_str = match.group(1).strip()
            if not params_str:
                return []

            # Split by comma and clean up
            params = [p.strip().split('=')[0].strip() for p in params_str.split(',')]
            return [p for p in params if p and p != '...']

        return []

    def _extract_class_methods(self, class_lines: List[str]) -> List[str]:
        """Extract method names from class definition."""
        methods = []
        for line in class_lines:
            # Look for method definitions
            match = re.match(r'^\s*(async\s+)?(static\s+)?(\w+)\s*\([^)]*\)\s*{', line.strip())
            if match and match.group(3) not in ['constructor']:
                methods.append(match.group(3))
        return methods

    def _contains_jsx_return(self, function_lines: List[str]) -> bool:
        """Check if function contains JSX return statement."""
        found_return = False
        for i, line in enumerate(function_lines):
            # Check if this line contains 'return'
            if 'return' in line:
                # Check if JSX is on the same line
                if '<' in line:
                    return True
                found_return = True
            # Check if JSX is on a subsequent line after return (within a few lines)
            elif found_return and i < len(function_lines) - 1:
                if '<' in line and any(jsx_tag in line for jsx_tag in ['<div', '<span', '<p', '<h1', '<h2', '<h3', '<button', '<input', '<form', '<img', '<a']):
                    return True
                # Also check for generic JSX patterns
                if re.search(r'<[A-Z]\w*', line):  # Component tags like <UserCard
                    return True
        return False

    def _create_chunks_from_elements(
        self,
        elements: List[Dict[str, Any]],
        lines: List[str],
        source: str,
        is_typescript: bool,
        is_jsx: bool
    ) -> List[Chunk]:
        """Create chunks from extracted JavaScript/TypeScript elements."""
        chunks = []

        if self.chunk_by == "function":
            # Create one chunk per function, plus imports/exports if enabled, and TypeScript types
            for element in elements:
                element_type = element['type']
                should_include = (
                    element_type in ['function_declaration', 'function_expression', 'arrow_function', 'method'] or
                    (element_type == 'import' and self.include_imports) or
                    (element_type == 'export' and self.include_exports) or
                    (element_type in ['interface', 'type_alias'] and is_typescript)
                )
                if should_include:
                    chunk = self._create_chunk_from_element(element, source, is_typescript, is_jsx)
                    chunks.append(chunk)

        elif self.chunk_by == "class":
            # Create one chunk per class
            for element in elements:
                if element['type'] in ['class', 'import', 'export']:
                    chunk = self._create_chunk_from_element(element, source, is_typescript, is_jsx)
                    chunks.append(chunk)

        elif self.chunk_by == "component":
            # Create one chunk per JSX component
            for element in elements:
                if element['type'] in ['jsx_component', 'function_declaration', 'arrow_function'] and element.get('is_jsx_component'):
                    chunk = self._create_chunk_from_element(element, source, is_typescript, is_jsx)
                    chunks.append(chunk)

        elif self.chunk_by == "logical_block":
            # Group related elements together
            chunks = self._create_logical_blocks(elements, source, is_typescript, is_jsx)

        elif self.chunk_by == "line_count":
            # Chunk by line count while respecting boundaries
            chunks = self._create_line_count_chunks(elements, lines, source, is_typescript, is_jsx)

        return chunks

    def _create_chunk_from_element(self, element: Dict[str, Any], source: str, is_typescript: bool, is_jsx: bool) -> Chunk:
        """Create a chunk from a single code element."""
        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": element['lineno'], "end_line": element['end_lineno']},
            chunker_used="javascript_code",
            extra={
                "chunk_type": "code",
                "language": "typescript" if is_typescript else "javascript",
                "element_type": element['type'],
                "element_name": element.get('name', 'unnamed'),
                "is_jsx": is_jsx,
                "is_typescript": is_typescript,
                "is_async": element.get('is_async', False),
                "is_jsx_component": element.get('is_jsx_component', False),
                "params": element.get('params', []),
                "methods": element.get('methods', []),
                "extends": element.get('extends')
            }
        )

        chunk_id = f"js_{element['type']}_{element.get('name', 'unnamed')}_{element['lineno']}"

        return Chunk(
            id=chunk_id,
            content=element['content'],
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_logical_blocks(self, elements: List[Dict[str, Any]], source: str, is_typescript: bool, is_jsx: bool) -> List[Chunk]:
        """Create chunks by grouping logically related elements."""
        chunks = []
        current_block = []
        current_lines = 0

        for element in elements:
            element_lines = element['end_lineno'] - element['lineno'] + 1
            # Account for the extra newlines added between elements in _create_block_chunk
            separator_lines = 2 if current_block else 0  # '\n\n' adds 2 lines between elements
            total_new_lines = element_lines + separator_lines

            if current_lines + total_new_lines > self.max_lines_per_chunk and current_block:
                chunk = self._create_block_chunk(current_block, source, len(chunks), is_typescript, is_jsx)
                chunks.append(chunk)
                current_block = [element]
                current_lines = element_lines
            else:
                current_block.append(element)
                current_lines += total_new_lines

        if current_block:
            chunk = self._create_block_chunk(current_block, source, len(chunks), is_typescript, is_jsx)
            chunks.append(chunk)

        return chunks

    def _create_block_chunk(self, elements: List[Dict[str, Any]], source: str, chunk_index: int, is_typescript: bool, is_jsx: bool) -> Chunk:
        """Create a chunk from multiple elements."""
        content = '\n\n'.join(element['content'] for element in elements)

        start_line = min(element['lineno'] for element in elements)
        end_line = max(element['end_lineno'] for element in elements)

        element_names = [element.get('name', 'unnamed') for element in elements]
        element_types = [element['type'] for element in elements]

        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": start_line, "end_line": end_line},
            chunker_used="javascript_code",
            extra={
                "chunk_type": "code",
                "language": "typescript" if is_typescript else "javascript",
                "element_types": element_types,
                "element_names": element_names,
                "block_size": len(elements),
                "is_jsx": is_jsx,
                "is_typescript": is_typescript
            }
        )

        return Chunk(
            id=f"js_block_{chunk_index}_{start_line}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_line_count_chunks(
        self,
        elements: List[Dict[str, Any]],
        lines: List[str],
        source: str,
        is_typescript: bool,
        is_jsx: bool
    ) -> List[Chunk]:
        """Create chunks based on line count while respecting element boundaries."""
        chunks = []
        current_lines = []
        current_elements = []

        for element in elements:
            element_lines = element['end_lineno'] - element['lineno'] + 1

            if len(current_lines) + element_lines > self.max_lines_per_chunk and current_lines:
                chunk = self._create_line_chunk(current_lines, current_elements, source, len(chunks), is_typescript, is_jsx)
                chunks.append(chunk)
                current_lines = []
                current_elements = []

            element_content_lines = element['content'].split('\n')
            current_lines.extend(element_content_lines)
            current_elements.append(element)

        if current_lines:
            chunk = self._create_line_chunk(current_lines, current_elements, source, len(chunks), is_typescript, is_jsx)
            chunks.append(chunk)

        return chunks

    def _create_line_chunk(
        self,
        lines: List[str],
        elements: List[Dict[str, Any]],
        source: str,
        chunk_index: int,
        is_typescript: bool,
        is_jsx: bool
    ) -> Chunk:
        """Create a chunk from lines and associated elements."""
        content = '\n'.join(lines)

        start_line = min(element['lineno'] for element in elements) if elements else 1
        end_line = max(element['end_lineno'] for element in elements) if elements else len(lines)

        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": start_line, "end_line": end_line},
            chunker_used="javascript_code",
            extra={
                "chunk_type": "code",
                "language": "typescript" if is_typescript else "javascript",
                "line_count": len(lines),
                "elements_included": len(elements),
                "is_jsx": is_jsx,
                "is_typescript": is_typescript
            }
        )

        return Chunk(
            id=f"js_lines_{chunk_index}_{start_line}",
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
                id=f"js_fallback_{i // self.max_lines_per_chunk}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=str(source_path) if source_path else "direct_input",
                    position={"start_line": i + 1, "end_line": min(i + len(chunk_lines), len(lines))},
                    chunker_used="javascript_code",
                    extra={
                        "chunk_type": "code",
                        "language": "javascript",
                        "fallback": True
                    }
                )
            )
            chunks.append(chunk)

        return ChunkingResult(
            chunks=chunks,
            processing_time=time.time() - start_time,
            strategy_used="javascript_code_fallback"
        )

    def chunk_stream(self, content_stream, **kwargs):
        """Chunk JavaScript/TypeScript code from a stream."""
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
        return ["js", "jsx", "ts", "tsx", "mjs", "cjs"]

    def estimate_chunks(self, content: Union[str, Path]) -> int:
        """Estimate number of chunks that will be generated."""
        if isinstance(content, (str, Path)):
            if isinstance(content, Path) and content.exists():
                with open(content, 'r') as f:
                    lines = len(f.readlines())
            else:
                lines = len(str(content).split('\n'))

            if self.chunk_by in ["function", "class", "component"]:
                return max(1, lines // 15)  # Estimate functions/classes
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
