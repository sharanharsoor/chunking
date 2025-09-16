"""
Python code chunking strategy.

Provides intelligent chunking for Python source code that preserves:
- Function and class boundaries
- Import statements
- Documentation strings
- Logical code blocks
"""

import ast
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.core.streaming import StreamableChunker


@register_chunker(
    name="python_code",
    category="code",
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    supported_formats=["py", "pyx", "pyi"],
    dependencies=["ast"],
    description="Intelligent Python code chunking preserving function and class boundaries",
    use_cases=["code_analysis", "documentation_generation", "code_search"]
)
class PythonCodeChunker(StreamableChunker):
    """
    Intelligent Python code chunker that preserves syntactic boundaries.

    Features:
    - Function and class boundary preservation
    - Import statement grouping
    - Docstring extraction and association
    - Logical code block detection
    - Syntax-aware chunking
    """

    def __init__(
        self,
        chunk_by: str = "function",  # "function", "class", "logical_block", "line_count"
        max_lines_per_chunk: int = 100,
        include_imports: bool = True,
        include_docstrings: bool = True,
        preserve_structure: bool = True,
        **kwargs
    ):
        """
        Initialize Python code chunker.

        Args:
            chunk_by: Chunking granularity ("function", "class", "logical_block", "line_count")
            max_lines_per_chunk: Maximum lines per chunk for line_count mode
            include_imports: Whether to include imports in chunks
            include_docstrings: Whether to include docstrings
            preserve_structure: Whether to preserve Python structure
            **kwargs: Additional parameters
        """
        super().__init__(
            name="python_code",
            category="code",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        self.chunk_by = chunk_by
        self.max_lines_per_chunk = max_lines_per_chunk
        self.include_imports = include_imports
        self.include_docstrings = include_docstrings
        self.preserve_structure = preserve_structure

        self.logger = logging.getLogger(__name__)

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk Python code content preserving syntactic boundaries.

        Args:
            content: Python code content or file path
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with Python code chunks
        """
        start_time = time.time()

        # Handle input types
        if isinstance(content, Path):
            source_path = content
            with open(content, 'r', encoding='utf-8') as f:
                code_content = f.read()
        elif isinstance(content, str) and len(content) > 0 and len(content) < 500 and '\n' not in content and Path(content).exists() and Path(content).is_file():
            source_path = Path(content)
            with open(source_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        else:
            source_path = None
            code_content = str(content)

        try:
            # Parse Python code using AST
            tree = ast.parse(code_content)
            lines = code_content.split('\n')

            # Extract code elements
            code_elements = self._extract_code_elements(tree, lines)

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
                strategy_used="python_code",
                source_info={
                    "source_file": str(source_path) if source_path else "direct_input",
                    "chunk_by": self.chunk_by,
                    "total_elements": len(code_elements),
                    "total_lines": len(lines)
                }
            )

            self.logger.info(f"Python code chunking completed: {len(chunks)} chunks from {source_path or 'direct input'}")
            return result

        except SyntaxError as e:
            self.logger.error(f"Python syntax error: {e}")
            # Fallback to line-based chunking for invalid Python
            return self._fallback_line_chunking(code_content, source_path, start_time)
        except Exception as e:
            self.logger.error(f"Error parsing Python code: {e}")
            return self._fallback_line_chunking(code_content, source_path, start_time)

    def _extract_code_elements(self, tree: ast.AST, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract Python code elements (functions, classes, etc.) from AST."""
        elements = []

        # Extract module-level imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, 'lineno'):
                    import_text = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    imports.append({
                        'type': 'import',
                        'content': import_text,
                        'lineno': node.lineno,
                        'end_lineno': getattr(node, 'end_lineno', node.lineno)
                    })

        if imports and self.include_imports:
            elements.append({
                'type': 'imports',
                'content': '\n'.join(imp['content'] for imp in imports),
                'lineno': min(imp['lineno'] for imp in imports),
                'end_lineno': max(imp['end_lineno'] for imp in imports),
                'name': 'imports'
            })

        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                element = self._extract_function(node, lines)
                if element:
                    elements.append(element)
            elif isinstance(node, ast.ClassDef):
                element = self._extract_class(node, lines)
                if element:
                    elements.append(element)

        # Sort by line number
        elements.sort(key=lambda x: x['lineno'])

        return elements

    def _extract_function(self, node: ast.FunctionDef, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Extract function information from AST node."""
        if not hasattr(node, 'lineno'):
            return None

        start_line = node.lineno - 1  # Convert to 0-based
        end_line = getattr(node, 'end_lineno', node.lineno) - 1

        # Get function content
        if end_line < len(lines):
            content_lines = lines[start_line:end_line + 1]
            content = '\n'.join(content_lines)
        else:
            content = lines[start_line] if start_line < len(lines) else ""

        # Extract docstring if present
        docstring = ast.get_docstring(node) if self.include_docstrings else None

        return {
            'type': 'function',
            'name': node.name,
            'content': content,
            'lineno': node.lineno,
            'end_lineno': getattr(node, 'end_lineno', node.lineno),
            'docstring': docstring,
            'args': [arg.arg for arg in node.args.args],
            'decorators': [ast.unparse(dec) for dec in node.decorator_list] if hasattr(ast, 'unparse') else []
        }

    def _extract_class(self, node: ast.ClassDef, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Extract class information from AST node."""
        if not hasattr(node, 'lineno'):
            return None

        start_line = node.lineno - 1  # Convert to 0-based
        end_line = getattr(node, 'end_lineno', node.lineno) - 1

        # Get class content
        if end_line < len(lines):
            content_lines = lines[start_line:end_line + 1]
            content = '\n'.join(content_lines)
        else:
            content = lines[start_line] if start_line < len(lines) else ""

        # Extract docstring if present
        docstring = ast.get_docstring(node) if self.include_docstrings else None

        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)

        return {
            'type': 'class',
            'name': node.name,
            'content': content,
            'lineno': node.lineno,
            'end_lineno': getattr(node, 'end_lineno', node.lineno),
            'docstring': docstring,
            'methods': methods,
            'bases': [ast.unparse(base) for base in node.bases] if hasattr(ast, 'unparse') else []
        }

    def _create_chunks_from_elements(
        self,
        elements: List[Dict[str, Any]],
        lines: List[str],
        source: str
    ) -> List[Chunk]:
        """Create chunks from extracted Python elements."""
        chunks = []

        if self.chunk_by == "function":
            # Create one chunk per function
            for element in elements:
                if element['type'] in ['function', 'imports']:
                    chunk = self._create_chunk_from_element(element, source)
                    chunks.append(chunk)

        elif self.chunk_by == "class":
            # Create one chunk per class (including its methods)
            for element in elements:
                if element['type'] in ['class', 'imports']:
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
            chunker_used="python_code",
            extra={
                "chunk_type": "code",
                "language": "python",
                "element_type": element['type'],
                "element_name": element.get('name', 'unnamed'),
                "docstring": element.get('docstring'),
                "methods": element.get('methods', []),
                "args": element.get('args', []),
                "decorators": element.get('decorators', []),
                "bases": element.get('bases', [])
            }
        )

        chunk_id = f"python_{element['type']}_{element.get('name', 'unnamed')}_{element['lineno']}"

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

            # Start new block if current would be too large
            if current_lines + element_lines > self.max_lines_per_chunk and current_block:
                chunk = self._create_block_chunk(current_block, source, len(chunks))
                chunks.append(chunk)
                current_block = [element]
                current_lines = element_lines
            else:
                current_block.append(element)
                current_lines += element_lines

        # Add final block
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
            chunker_used="python_code",
            extra={
                "chunk_type": "code",
                "language": "python",
                "element_types": element_types,
                "element_names": element_names,
                "block_size": len(elements)
            }
        )

        return Chunk(
            id=f"python_block_{chunk_index}_{start_line}",
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

            # If adding this element would exceed max lines, finalize current chunk
            if len(current_lines) + element_lines > self.max_lines_per_chunk and current_lines:
                chunk = self._create_line_chunk(current_lines, current_elements, source, len(chunks))
                chunks.append(chunk)
                current_lines = []
                current_elements = []

            # Add element lines
            element_content_lines = element['content'].split('\n')
            current_lines.extend(element_content_lines)
            current_elements.append(element)

        # Add final chunk
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
            chunker_used="python_code",
            extra={
                "chunk_type": "code",
                "language": "python",
                "line_count": len(lines),
                "elements_included": len(elements)
            }
        )

        return Chunk(
            id=f"python_lines_{chunk_index}_{start_line}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _fallback_line_chunking(self, content: str, source_path: Optional[Path], start_time: float) -> ChunkingResult:
        """Fallback to simple line-based chunking for invalid Python code."""
        lines = content.split('\n')
        chunks = []

        for i in range(0, len(lines), self.max_lines_per_chunk):
            chunk_lines = lines[i:i + self.max_lines_per_chunk]
            chunk_content = '\n'.join(chunk_lines)

            chunk = Chunk(
                id=f"python_fallback_{i // self.max_lines_per_chunk}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=str(source_path) if source_path else "direct_input",
                    position={"start_line": i + 1, "end_line": min(i + len(chunk_lines), len(lines))},
                    chunker_used="python_code",
                    extra={
                        "chunk_type": "code",
                        "language": "python",
                        "fallback": True,
                        "reason": "syntax_error"
                    }
                )
            )
            chunks.append(chunk)

        return ChunkingResult(
            chunks=chunks,
            processing_time=time.time() - start_time,
            strategy_used="python_code_fallback"
        )

    def chunk_stream(self, content_stream, **kwargs):
        """
        Chunk Python code from a stream.
        Note: Python parsing requires full content, so we collect the stream first.
        """
        # Collect stream content
        content_lines = []
        for chunk in content_stream:
            if isinstance(chunk, str):
                content_lines.extend(chunk.split('\n'))
            elif isinstance(chunk, bytes):
                content_lines.extend(chunk.decode('utf-8').split('\n'))

        content = '\n'.join(content_lines)

        # Process as complete content
        result = self.chunk(content, **kwargs)

        # Yield chunks one by one
        for chunk in result.chunks:
            yield chunk

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ["py", "pyx", "pyi"]

    def estimate_chunks(self, content: Union[str, Path]) -> int:
        """Estimate number of chunks that will be generated."""
        if isinstance(content, (str, Path)):
            if isinstance(content, Path) and content.exists():
                with open(content, 'r') as f:
                    lines = len(f.readlines())
            else:
                lines = len(str(content).split('\n'))

            if self.chunk_by in ["function", "class"]:
                # Rough estimate: average 10-20 lines per function/class
                return max(1, lines // 15)
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
        if feedback_type == "quality" and feedback_score < 0.5:
            # Reduce chunk size for better granularity
            if self.chunk_by == "line_count":
                old_max = self.max_lines_per_chunk
                self.max_lines_per_chunk = max(20, int(self.max_lines_per_chunk * 0.8))
                self.logger.info(f"Adapted max_lines_per_chunk: {old_max} -> {self.max_lines_per_chunk}")

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Increase chunk size for better performance
            if self.chunk_by == "line_count":
                old_max = self.max_lines_per_chunk
                self.max_lines_per_chunk = min(500, int(self.max_lines_per_chunk * 1.2))
                self.logger.info(f"Adapted max_lines_per_chunk: {old_max} -> {self.max_lines_per_chunk}")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return []
