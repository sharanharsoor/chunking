"""
C/C++ code chunking strategy.

Provides intelligent chunking for C and C++ source code that preserves:
- Function definitions and declarations
- Class/struct definitions  
- Include statements
- Preprocessor directives
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


@register_chunker(
    name="c_cpp_code",
    category="code", 
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    supported_formats=["c", "cpp", "cc", "cxx", "h", "hpp", "hxx"],
    dependencies=[],
    description="Intelligent C/C++ code chunking preserving function and structure boundaries",
    use_cases=["code_analysis", "documentation_generation", "code_search"]
)
class CCppCodeChunker(StreamableChunker):
    """
    Intelligent C/C++ code chunker that preserves syntactic boundaries.
    
    Features:
    - Function and structure boundary preservation
    - Include statement grouping
    - Preprocessor directive handling
    - Class/struct definition extraction
    - Comment and documentation preservation
    """

    def __init__(
        self,
        chunk_by: str = "function",  # "function", "class", "logical_block", "line_count"
        max_lines_per_chunk: int = 100,
        include_headers: bool = True,
        include_comments: bool = True,
        preserve_structure: bool = True,
        **kwargs
    ):
        """
        Initialize C/C++ code chunker.

        Args:
            chunk_by: Chunking granularity ("function", "class", "logical_block", "line_count")
            max_lines_per_chunk: Maximum lines per chunk for line_count mode
            include_headers: Whether to include #include statements
            include_comments: Whether to include comments
            preserve_structure: Whether to preserve C/C++ structure
            **kwargs: Additional parameters
        """
        super().__init__(
            name="c_cpp_code",
            category="code",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )
        
        self.chunk_by = chunk_by
        self.max_lines_per_chunk = max_lines_per_chunk
        self.include_headers = include_headers
        self.include_comments = include_comments
        self.preserve_structure = preserve_structure
        
        self.logger = logging.getLogger(__name__)

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk C/C++ code content preserving syntactic boundaries.

        Args:
            content: C/C++ code content or file path
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with C/C++ code chunks
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
            # Parse C/C++ code using regex patterns
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
                strategy_used="c_cpp_code",
                source_info={
                    "source_file": str(source_path) if source_path else "direct_input",
                    "chunk_by": self.chunk_by,
                    "total_elements": len(code_elements),
                    "total_lines": len(lines)
                }
            )
            
            self.logger.info(f"C/C++ code chunking completed: {len(chunks)} chunks from {source_path or 'direct input'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing C/C++ code: {e}")
            # Fallback to line-based chunking
            return self._fallback_line_chunking(code_content, source_path, start_time)

    def _extract_code_elements(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract C/C++ code elements using regex patterns."""
        elements = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and single-line comments
            if not line or line.startswith('//'):
                i += 1
                continue
            
            # Extract includes/preprocessor directives
            if line.startswith('#'):
                element = self._extract_preprocessor(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue
            
            # Extract function definitions
            if self._is_function_start(line, lines, i):
                element = self._extract_function(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue
            
            # Extract class/struct definitions
            if self._is_class_struct_start(line):
                element = self._extract_class_struct(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue
            
            i += 1
        
        return elements

    def _extract_preprocessor(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract preprocessor directives."""
        if not self.include_headers:
            return None
        
        line = lines[start_idx].strip()
        
        return {
            'type': 'preprocessor',
            'content': line,
            'lineno': start_idx + 1,
            'end_lineno': start_idx + 1,
            'directive': line.split()[0] if line.split() else '#'
        }

    def _is_function_start(self, line: str, lines: List[str], idx: int) -> bool:
        """Check if line starts a function definition."""
        # Simple heuristic: look for patterns like "type name(" or "name("
        pattern = r'^\s*\w+\s+\w+\s*\([^)]*\)\s*\{?'
        return bool(re.match(pattern, line)) or '(' in line and '{' in line

    def _extract_function(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract function definition."""
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
                        
                        # Extract function name
                        first_line = function_lines[0].strip()
                        name_match = re.search(r'\b(\w+)\s*\(', first_line)
                        func_name = name_match.group(1) if name_match else 'unnamed'
                        
                        return {
                            'type': 'function',
                            'name': func_name,
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1
                        }
        
        # If we get here, function wasn't properly closed
        return None

    def _is_class_struct_start(self, line: str) -> bool:
        """Check if line starts a class or struct definition."""
        return bool(re.match(r'^\s*(class|struct)\s+\w+', line))

    def _extract_class_struct(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract class or struct definition."""
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
                        # Look for semicolon after closing brace
                        if i + 1 < len(lines) and ';' in lines[i + 1]:
                            class_lines.append(lines[i + 1])
                            i += 1
                        
                        content = '\n'.join(class_lines)
                        
                        # Extract class/struct name
                        first_line = class_lines[0].strip()
                        name_match = re.search(r'(class|struct)\s+(\w+)', first_line)
                        name = name_match.group(2) if name_match else 'unnamed'
                        class_type = name_match.group(1) if name_match else 'class'
                        
                        return {
                            'type': class_type,
                            'name': name,
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
        """Create chunks from extracted C/C++ elements."""
        chunks = []
        
        if self.chunk_by == "function":
            # Create one chunk per function
            for element in elements:
                if element['type'] in ['function', 'preprocessor']:
                    chunk = self._create_chunk_from_element(element, source)
                    chunks.append(chunk)
        
        elif self.chunk_by == "class":
            # Create one chunk per class/struct
            for element in elements:
                if element['type'] in ['class', 'struct', 'preprocessor']:
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
            chunker_used="c_cpp_code",
            extra={
                "chunk_type": "code",
                "language": "c_cpp",
                "element_type": element['type'],
                "element_name": element.get('name', 'unnamed'),
                "directive": element.get('directive')
            }
        )
        
        chunk_id = f"c_cpp_{element['type']}_{element.get('name', 'unnamed')}_{element['lineno']}"
        
        return Chunk(
            id=chunk_id,
            content=element['content'],
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_logical_blocks(self, elements: List[Dict[str, Any]], source: str) -> List[Chunk]:
        """Create chunks by grouping logically related elements."""
        # Implementation similar to Python chunker
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
            chunker_used="c_cpp_code",
            extra={
                "chunk_type": "code",
                "language": "c_cpp",
                "element_types": element_types,
                "element_names": element_names,
                "block_size": len(elements)
            }
        )
        
        return Chunk(
            id=f"c_cpp_block_{chunk_index}_{start_line}",
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
        # Implementation similar to Python chunker
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
            chunker_used="c_cpp_code",
            extra={
                "chunk_type": "code",
                "language": "c_cpp",
                "line_count": len(lines),
                "elements_included": len(elements)
            }
        )
        
        return Chunk(
            id=f"c_cpp_lines_{chunk_index}_{start_line}",
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
                id=f"c_cpp_fallback_{i // self.max_lines_per_chunk}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=str(source_path) if source_path else "direct_input",
                    position={"start_line": i + 1, "end_line": min(i + len(chunk_lines), len(lines))},
                    chunker_used="c_cpp_code",
                    extra={
                        "chunk_type": "code",
                        "language": "c_cpp",
                        "fallback": True
                    }
                )
            )
            chunks.append(chunk)
        
        return ChunkingResult(
            chunks=chunks,
            processing_time=time.time() - start_time,
            strategy_used="c_cpp_code_fallback"
        )

    def chunk_stream(self, content_stream, **kwargs):
        """Chunk C/C++ code from a stream."""
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
        return ["c", "cpp", "cc", "cxx", "h", "hpp", "hxx"]

    def estimate_chunks(self, content: Union[str, Path]) -> int:
        """Estimate number of chunks that will be generated."""
        if isinstance(content, (str, Path)):
            if isinstance(content, Path) and content.exists():
                with open(content, 'r') as f:
                    lines = len(f.readlines())
            else:
                lines = len(str(content).split('\n'))
            
            if self.chunk_by in ["function", "class"]:
                return max(1, lines // 20)  # Estimate functions/classes
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
            if self.chunk_by == "line_count":
                old_max = self.max_lines_per_chunk
                self.max_lines_per_chunk = max(20, int(self.max_lines_per_chunk * 0.8))
                self.logger.info(f"Adapted max_lines_per_chunk: {old_max} -> {self.max_lines_per_chunk}")
        elif feedback_type == "performance" and feedback_score < 0.5:
            if self.chunk_by == "line_count":
                old_max = self.max_lines_per_chunk
                self.max_lines_per_chunk = min(500, int(self.max_lines_per_chunk * 1.2))
                self.logger.info(f"Adapted max_lines_per_chunk: {old_max} -> {self.max_lines_per_chunk}")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return []
