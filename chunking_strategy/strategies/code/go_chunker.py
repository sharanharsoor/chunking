"""
Go code chunking strategy.

Provides intelligent chunking for Go source code that preserves:
- Package declarations and imports
- Function definitions and methods
- Struct and interface definitions
- Type definitions
- Constants and variables
- Logical code blocks
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
    name="go_code",
    category="code",
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    supported_formats=["go"],
    dependencies=[],
    description="Intelligent Go code chunking preserving function, struct, and interface boundaries",
    use_cases=["code_analysis", "documentation_generation", "code_search", "microservice_analysis"]
)
class GoChunker(StreamableChunker, AdaptableChunker):
    """
    Intelligent Go code chunker that preserves syntactic boundaries.

    Features:
    - Package declaration and import grouping
    - Function and method boundary preservation
    - Struct and interface definition extraction
    - Type definition detection
    - Constant and variable grouping
    - Receiver method association
    - Comment and documentation preservation
    """

    def __init__(
        self,
        chunk_by: str = "function",  # "function", "struct", "interface", "type", "logical_block", "line_count"
        max_lines_per_chunk: int = 100,
        include_package: bool = True,
        include_imports: bool = True,
        include_types: bool = True,
        group_methods: bool = True,
        preserve_comments: bool = True,
        **kwargs
    ):
        """
        Initialize Go code chunker.

        Args:
            chunk_by: Chunking granularity ("function", "struct", "interface", "type", "logical_block", "line_count")
            max_lines_per_chunk: Maximum lines per chunk for line_count mode
            include_package: Whether to include package declaration
            include_imports: Whether to include import statements
            include_types: Whether to include type definitions
            group_methods: Whether to group methods with their receiver structs
            preserve_comments: Whether to preserve comments
            **kwargs: Additional parameters
        """
        super().__init__(
            name="go_code",
            category="code",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        self.chunk_by = chunk_by
        self.max_lines_per_chunk = max_lines_per_chunk
        self.include_package = include_package
        self.include_imports = include_imports
        self.include_types = include_types
        self.group_methods = group_methods
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
        Chunk Go code content preserving syntactic boundaries.

        Args:
            content: Go code content or file path
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with Go code chunks
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

        try:
            # Parse Go code
            lines = code_content.split('\n')
            code_elements = self._extract_code_elements(lines)

            # Create chunks based on strategy
            chunks = self._create_chunks_from_elements(
                code_elements,
                lines,
                source_path or "direct_input"
            )

            # Create chunking result
            result = ChunkingResult(
                chunks=chunks,
                processing_time=time.time() - start_time,
                strategy_used="go_code",
                source_info={
                    "source_file": str(source_path) if source_path else "direct_input",
                    "chunk_by": self.chunk_by,
                    "total_elements": len(code_elements),
                    "total_lines": len(lines),
                    "package_name": self._extract_package_name(lines)
                }
            )

            self.logger.info(f"Go code chunking completed: {len(chunks)} chunks from {source_path or 'direct input'}")
            return result

        except Exception as e:
            self.logger.error(f"Error parsing Go code: {e}")
            # Fallback to line-based chunking
            return self._fallback_line_chunking(code_content, source_path, start_time)

    def _extract_package_name(self, lines: List[str]) -> Optional[str]:
        """Extract package name from Go file."""
        for line in lines[:10]:  # Package should be near the top
            line = line.strip()
            if line.startswith('package '):
                return line.split()[1]
        return None

    def _extract_code_elements(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract Go code elements."""
        elements = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and single-line comments
            if not line or line.startswith('//'):
                i += 1
                continue

            # Extract package declaration
            if line.startswith('package ') and self.include_package:
                element = self._extract_package(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue

            # Extract import statements
            if line.startswith('import') and self.include_imports:
                element = self._extract_import_block(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue

            # Extract function definitions
            if self._is_function_start(line):
                element = self._extract_function(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue

            # Extract struct definitions
            if self._is_struct_start(line):
                element = self._extract_struct(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue

            # Extract interface definitions
            if self._is_interface_start(line):
                element = self._extract_interface(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue

            # Extract type definitions
            if self._is_type_definition(line) and self.include_types:
                element = self._extract_type_definition(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue

            # Extract const/var blocks
            if self._is_const_var_block(line):
                element = self._extract_const_var_block(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue

            i += 1

        return elements

    def _extract_package(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract package declaration."""
        line = lines[start_idx].strip()
        package_match = re.match(r'package\s+(\w+)', line)

        if package_match:
            package_name = package_match.group(1)
            return {
                'type': 'package',
                'name': package_name,
                'content': line,
                'lineno': start_idx + 1,
                'end_lineno': start_idx + 1
            }
        return None

    def _extract_import_block(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract import statements (single or block)."""
        line = lines[start_idx].strip()

        if line == 'import (':
            # Multi-line import block
            import_lines = [line]
            i = start_idx + 1

            while i < len(lines):
                import_line = lines[i]
                import_lines.append(import_line)
                if import_line.strip() == ')':
                    break
                i += 1

            content = '\n'.join(import_lines)
            imports = self._parse_import_content(content)

            return {
                'type': 'import_block',
                'name': 'imports',
                'content': content,
                'lineno': start_idx + 1,
                'end_lineno': i + 1,
                'imports': imports
            }

        elif line.startswith('import '):
            # Single import
            import_match = re.match(r'import\s+(.+)', line)
            if import_match:
                import_spec = import_match.group(1).strip()
                return {
                    'type': 'import',
                    'name': 'import',
                    'content': line,
                    'lineno': start_idx + 1,
                    'end_lineno': start_idx + 1,
                    'import_spec': import_spec
                }

        return None

    def _parse_import_content(self, content: str) -> List[str]:
        """Parse import specifications from import block."""
        imports = []
        lines = content.split('\n')

        for line in lines[1:-1]:  # Skip 'import (' and ')'
            line = line.strip()
            if line and not line.startswith('//'):
                imports.append(line)

        return imports

    def _is_function_start(self, line: str) -> bool:
        """Check if line starts a function definition."""
        # Function definition: func [receiver] name(params) [returns] {
        patterns = [
            r'^\s*func\s+\w+\s*\(',  # func name(
            r'^\s*func\s+\([^)]*\)\s*\w+\s*\(',  # func (receiver) name(
        ]

        for pattern in patterns:
            if re.match(pattern, line):
                return True
        return False

    def _extract_function(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract function definition."""
        first_line = lines[start_idx].strip()

        # Parse function signature
        func_info = self._parse_function_signature(first_line)
        if not func_info:
            return None

        # Find function body end
        brace_count = 0
        function_lines = []
        in_function = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            function_lines.append(line)

            # Count braces to find function end
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_function = True
                elif char == '}':
                    brace_count -= 1

                    if in_function and brace_count == 0:
                        # Function complete
                        content = '\n'.join(function_lines)

                        return {
                            'type': 'function',
                            'name': func_info['name'],
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'receiver': func_info.get('receiver'),
                            'params': func_info.get('params', []),
                            'returns': func_info.get('returns', [])
                        }

        return None

    def _parse_function_signature(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse function signature to extract name, receiver, params, returns."""
        # Handle receiver methods: func (r *Receiver) MethodName(params) returns
        receiver_match = re.match(r'^\s*func\s+\(([^)]+)\)\s*(\w+)\s*\(([^)]*)\)(.*)$', line)
        if receiver_match:
            receiver = receiver_match.group(1).strip()
            name = receiver_match.group(2)
            params_str = receiver_match.group(3).strip()
            returns_str = receiver_match.group(4).strip()

            return {
                'name': name,
                'receiver': receiver,
                'params': self._parse_params(params_str),
                'returns': self._parse_returns(returns_str)
            }

        # Handle regular functions: func FunctionName(params) returns
        func_match = re.match(r'^\s*func\s+(\w+)\s*\(([^)]*)\)(.*)$', line)
        if func_match:
            name = func_match.group(1)
            params_str = func_match.group(2).strip()
            returns_str = func_match.group(3).strip()

            return {
                'name': name,
                'params': self._parse_params(params_str),
                'returns': self._parse_returns(returns_str)
            }

        return None

    def _parse_params(self, params_str: str) -> List[str]:
        """Parse function parameters."""
        if not params_str.strip():
            return []

        # Simple parameter parsing - could be enhanced
        params = []
        for param in params_str.split(','):
            param = param.strip()
            if param:
                # Extract parameter name (before type)
                parts = param.split()
                if len(parts) >= 2:
                    params.append(parts[0])
                elif len(parts) == 1:
                    params.append(parts[0])

        return params

    def _parse_returns(self, returns_str: str) -> List[str]:
        """Parse function return types."""
        returns_str = returns_str.strip()
        if not returns_str:
            return []

        # Remove leading space and opening brace
        returns_str = returns_str.lstrip().rstrip(' {')

        if returns_str.startswith('(') and returns_str.endswith(')'):
            # Multiple returns: (type1, type2)
            returns_str = returns_str[1:-1]
            return [r.strip() for r in returns_str.split(',') if r.strip()]
        elif returns_str:
            # Single return
            return [returns_str]

        return []

    def _is_struct_start(self, line: str) -> bool:
        """Check if line starts a struct definition."""
        return bool(re.match(r'^\s*type\s+\w+\s+struct\s*{', line))

    def _extract_struct(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract struct definition."""
        first_line = lines[start_idx].strip()

        # Extract struct name
        struct_match = re.match(r'^\s*type\s+(\w+)\s+struct\s*{', first_line)
        if not struct_match:
            return None

        struct_name = struct_match.group(1)

        # Find struct end
        brace_count = 0
        struct_lines = []
        in_struct = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            struct_lines.append(line)

            for char in line:
                if char == '{':
                    brace_count += 1
                    in_struct = True
                elif char == '}':
                    brace_count -= 1

                    if in_struct and brace_count == 0:
                        content = '\n'.join(struct_lines)
                        fields = self._extract_struct_fields(struct_lines)

                        return {
                            'type': 'struct',
                            'name': struct_name,
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'fields': fields
                        }

        return None

    def _extract_struct_fields(self, struct_lines: List[str]) -> List[Dict[str, str]]:
        """Extract fields from struct definition."""
        fields = []

        for line in struct_lines[1:-1]:  # Skip opening and closing braces
            line = line.strip()
            if line and not line.startswith('//'):
                # Simple field parsing: FieldName FieldType `tags`
                parts = line.split()
                if len(parts) >= 2:
                    field_name = parts[0]
                    field_type = parts[1]
                    tags = ' '.join(parts[2:]) if len(parts) > 2 else ''

                    fields.append({
                        'name': field_name,
                        'type': field_type,
                        'tags': tags
                    })

        return fields

    def _is_interface_start(self, line: str) -> bool:
        """Check if line starts an interface definition."""
        return bool(re.match(r'^\s*type\s+\w+\s+interface\s*{', line))

    def _extract_interface(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract interface definition."""
        first_line = lines[start_idx].strip()

        # Extract interface name
        interface_match = re.match(r'^\s*type\s+(\w+)\s+interface\s*{', first_line)
        if not interface_match:
            return None

        interface_name = interface_match.group(1)

        # Find interface end
        brace_count = 0
        interface_lines = []
        in_interface = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            interface_lines.append(line)

            for char in line:
                if char == '{':
                    brace_count += 1
                    in_interface = True
                elif char == '}':
                    brace_count -= 1

                    if in_interface and brace_count == 0:
                        content = '\n'.join(interface_lines)
                        methods = self._extract_interface_methods(interface_lines)

                        return {
                            'type': 'interface',
                            'name': interface_name,
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'methods': methods
                        }

        return None

    def _extract_interface_methods(self, interface_lines: List[str]) -> List[str]:
        """Extract method signatures from interface definition."""
        methods = []

        for line in interface_lines[1:-1]:  # Skip opening and closing braces
            line = line.strip()
            if line and not line.startswith('//'):
                # Extract method name (first word)
                parts = line.split()
                if parts:
                    methods.append(parts[0])

        return methods

    def _is_type_definition(self, line: str) -> bool:
        """Check if line is a type definition (not struct or interface)."""
        return (line.strip().startswith('type ') and
                'struct' not in line and
                'interface' not in line)

    def _extract_type_definition(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract type definition."""
        line = lines[start_idx].strip()

        # Simple type alias: type NewType ExistingType
        type_match = re.match(r'^\s*type\s+(\w+)\s+(.+)$', line)
        if type_match:
            type_name = type_match.group(1)
            type_spec = type_match.group(2).strip()

            return {
                'type': 'type_definition',
                'name': type_name,
                'content': line,
                'lineno': start_idx + 1,
                'end_lineno': start_idx + 1,
                'type_spec': type_spec
            }

        return None

    def _is_const_var_block(self, line: str) -> bool:
        """Check if line starts a const or var block."""
        return bool(re.match(r'^\s*(const|var)\s*\(', line))

    def _extract_const_var_block(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract const or var block."""
        first_line = lines[start_idx].strip()

        # Determine if it's const or var
        block_type = 'const' if first_line.startswith('const') else 'var'

        # Find block end
        paren_count = 0
        block_lines = []
        in_block = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            block_lines.append(line)

            for char in line:
                if char == '(':
                    paren_count += 1
                    in_block = True
                elif char == ')':
                    paren_count -= 1

                    if in_block and paren_count == 0:
                        content = '\n'.join(block_lines)

                        return {
                            'type': f'{block_type}_block',
                            'name': f'{block_type}_declarations',
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1
                        }

        return None

    def _create_chunks_from_elements(
        self,
        elements: List[Dict[str, Any]],
        lines: List[str],
        source: str
    ) -> List[Chunk]:
        """Create chunks from extracted Go elements."""
        chunks = []

        if self.chunk_by == "function":
            # Create one chunk per function
            for element in elements:
                if element['type'] in ['function', 'package', 'import_block', 'import']:
                    chunk = self._create_chunk_from_element(element, source)
                    chunks.append(chunk)

        elif self.chunk_by == "struct":
            # Create one chunk per struct
            for element in elements:
                if element['type'] in ['struct', 'package', 'import_block']:
                    chunk = self._create_chunk_from_element(element, source)
                    chunks.append(chunk)

        elif self.chunk_by == "interface":
            # Create one chunk per interface
            for element in elements:
                if element['type'] in ['interface', 'package', 'import_block']:
                    chunk = self._create_chunk_from_element(element, source)
                    chunks.append(chunk)

        elif self.chunk_by == "type":
            # Create one chunk per type (struct, interface, type_definition)
            for element in elements:
                if element['type'] in ['struct', 'interface', 'type_definition', 'package', 'import_block']:
                    chunk = self._create_chunk_from_element(element, source)
                    chunks.append(chunk)

        elif self.chunk_by == "logical_block":
            # Group related elements together
            chunks = self._create_logical_blocks(elements, source)

        elif self.chunk_by == "line_count":
            # Chunk by line count while respecting boundaries
            chunks = self._create_line_count_chunks(elements, lines, source)

        return chunks

    def _create_chunk_from_element(self, element: Dict[str, Any], source: str) -> Chunk:
        """Create a chunk from a single code element."""
        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": element['lineno'], "end_line": element['end_lineno']},
            chunker_used="go_code",
            extra={
                "chunk_type": "code",
                "language": "go",
                "element_type": element['type'],
                "element_name": element.get('name', 'unnamed'),
                "receiver": element.get('receiver'),
                "params": element.get('params', []),
                "returns": element.get('returns', []),
                "fields": element.get('fields', []),
                "methods": element.get('methods', []),
                "imports": element.get('imports', []),
                "type_spec": element.get('type_spec')
            }
        )

        chunk_id = f"go_{element['type']}_{element.get('name', 'unnamed')}_{element['lineno']}"

        return Chunk(
            id=chunk_id,
            content=element['content'],
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_logical_blocks(self, elements: List[Dict[str, Any]], source: str) -> List[Chunk]:
        """Create chunks by grouping logically related elements."""
        chunks = []
        current_block = []
        current_lines = 0

        for element in elements:
            element_lines = element['end_lineno'] - element['lineno'] + 1

            if current_lines + element_lines > self.max_lines_per_chunk and current_block:
                chunk = self._create_block_chunk(current_block, source, len(chunks))
                chunks.append(chunk)
                current_block = [element]
                current_lines = element_lines
            else:
                current_block.append(element)
                current_lines += element_lines

        if current_block:
            chunk = self._create_block_chunk(current_block, source, len(chunks))
            chunks.append(chunk)

        return chunks

    def _create_block_chunk(self, elements: List[Dict[str, Any]], source: str, chunk_index: int) -> Chunk:
        """Create a chunk from multiple elements."""
        content = '\n\n'.join(element['content'] for element in elements)

        start_line = min(element['lineno'] for element in elements)
        end_line = max(element['end_lineno'] for element in elements)

        element_names = [element.get('name', 'unnamed') for element in elements]
        element_types = [element['type'] for element in elements]

        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": start_line, "end_line": end_line},
            chunker_used="go_code",
            extra={
                "chunk_type": "code",
                "language": "go",
                "element_types": element_types,
                "element_names": element_names,
                "block_size": len(elements)
            }
        )

        return Chunk(
            id=f"go_block_{chunk_index}_{start_line}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_line_count_chunks(
        self,
        elements: List[Dict[str, Any]],
        lines: List[str],
        source: str
    ) -> List[Chunk]:
        """Create chunks based on line count while respecting element boundaries."""
        chunks = []
        current_lines = []
        current_elements = []

        for element in elements:
            element_lines = element['end_lineno'] - element['lineno'] + 1

            if len(current_lines) + element_lines > self.max_lines_per_chunk and current_lines:
                chunk = self._create_line_chunk(current_lines, current_elements, source, len(chunks))
                chunks.append(chunk)
                current_lines = []
                current_elements = []

            element_content_lines = element['content'].split('\n')
            current_lines.extend(element_content_lines)
            current_elements.append(element)

        if current_lines:
            chunk = self._create_line_chunk(current_lines, current_elements, source, len(chunks))
            chunks.append(chunk)

        return chunks

    def _create_line_chunk(
        self,
        lines: List[str],
        elements: List[Dict[str, Any]],
        source: str,
        chunk_index: int
    ) -> Chunk:
        """Create a chunk from lines and associated elements."""
        content = '\n'.join(lines)

        start_line = min(element['lineno'] for element in elements) if elements else 1
        end_line = max(element['end_lineno'] for element in elements) if elements else len(lines)

        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": start_line, "end_line": end_line},
            chunker_used="go_code",
            extra={
                "chunk_type": "code",
                "language": "go",
                "line_count": len(lines),
                "elements_included": len(elements)
            }
        )

        return Chunk(
            id=f"go_lines_{chunk_index}_{start_line}",
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
                id=f"go_fallback_{i // self.max_lines_per_chunk}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=str(source_path) if source_path else "direct_input",
                    position={"start_line": i + 1, "end_line": min(i + len(chunk_lines), len(lines))},
                    chunker_used="go_code",
                    extra={
                        "chunk_type": "code",
                        "language": "go",
                        "fallback": True
                    }
                )
            )
            chunks.append(chunk)

        return ChunkingResult(
            chunks=chunks,
            processing_time=time.time() - start_time,
            strategy_used="go_code_fallback"
        )

    def chunk_stream(self, content_stream, **kwargs):
        """Chunk Go code from a stream."""
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
        return ["go"]

    def estimate_chunks(self, content: Union[str, Path]) -> int:
        """Estimate number of chunks that will be generated."""
        if isinstance(content, (str, Path)):
            if isinstance(content, Path) and content.exists():
                with open(content, 'r') as f:
                    lines = len(f.readlines())
            else:
                lines = len(str(content).split('\n'))

            if self.chunk_by in ["function", "struct", "interface", "type"]:
                return max(1, lines // 20)  # Estimate functions/types
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
