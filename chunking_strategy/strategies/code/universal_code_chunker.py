"""
Universal code chunking strategy.

Provides intelligent chunking for any programming language using:
- Language detection
- Common syntax patterns
- Indentation-based structure detection
- Comment and string handling
- Configurable language-specific rules
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
    name="universal_code",
    category="code",
    complexity=ComplexityLevel.LOW,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    supported_formats=["*"],  # Supports all code formats
    dependencies=[],
    description="Universal code chunker for any programming language",
    use_cases=["code_analysis", "multi_language_processing", "generic_code_chunking"]
)
class UniversalCodeChunker(StreamableChunker):
    """
    Universal code chunker that works with any programming language.

    Features:
    - Automatic language detection from file extension
    - Indentation-based structure detection
    - Function/method boundary detection
    - Comment and documentation preservation
    - Configurable language-specific patterns
    - Fallback to generic chunking
    """

    # Language configuration
    LANGUAGE_CONFIGS = {
        'python': {
            'extensions': ['.py', '.pyx', '.pyi'],
            'comment_patterns': [r'#.*$'],
            'string_patterns': [r'""".*?"""', r"'''.*?'''", r'".*?"', r"'.*?'"],
            'function_patterns': [r'^\s*def\s+\w+\s*\(', r'^\s*class\s+\w+'],
            'indent_sensitive': True,
            'block_start': ':',
            'block_end': None
        },
        'javascript': {
            'extensions': ['.js', '.jsx', '.ts', '.tsx'],
            'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
            'string_patterns': [r'`.*?`', r'".*?"', r"'.*?'"],
            'function_patterns': [r'function\s+\w+\s*\(', r'^\s*\w+\s*:\s*function', r'^\s*\w+\s*=\s*\(.*?\)\s*=>'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'java': {
            'extensions': ['.java'],
            'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
            'string_patterns': [r'".*?"'],
            'function_patterns': [r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(', r'^\s*(public|private|protected)?\s*class\s+\w+'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'c_cpp': {
            'extensions': ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx'],
            'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
            'string_patterns': [r'".*?"'],
            'function_patterns': [r'^\s*\w+\s+\w+\s*\(', r'^\s*(class|struct)\s+\w+'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'go': {
            'extensions': ['.go'],
            'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
            'string_patterns': [r'`.*?`', r'".*?"'],
            'function_patterns': [r'^\s*func\s+\w+\s*\(', r'^\s*type\s+\w+\s+struct'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'rust': {
            'extensions': ['.rs'],
            'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
            'string_patterns': [r'".*?"'],
            'function_patterns': [r'^\s*fn\s+\w+\s*\(', r'^\s*struct\s+\w+', r'^\s*impl\s+'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'ruby': {
            'extensions': ['.rb', '.rbw'],
            'comment_patterns': [r'#.*$'],
            'string_patterns': [r'".*?"', r"'.*?'", r'%[qQ]\{.*?\}'],
            'function_patterns': [r'^\s*def\s+\w+', r'^\s*class\s+\w+', r'^\s*module\s+\w+'],
            'indent_sensitive': True,
            'block_start': None,
            'block_end': 'end'
        },
        'php': {
            'extensions': ['.php', '.phtml', '.php3', '.php4', '.php5'],
            'comment_patterns': [r'//.*$', r'/\*.*?\*/', r'#.*$'],
            'string_patterns': [r'".*?"', r"'.*?'"],
            'function_patterns': [r'^\s*function\s+\w+\s*\(', r'^\s*class\s+\w+'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'csharp': {
            'extensions': ['.cs'],
            'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
            'string_patterns': [r'".*?"', r"'.*?'", r'@".*?"'],
            'function_patterns': [r'^\s*(public|private|protected|internal)?\s*(static)?\s*\w+\s+\w+\s*\(', r'^\s*(public|private|protected|internal)?\s*class\s+\w+'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'swift': {
            'extensions': ['.swift'],
            'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
            'string_patterns': [r'".*?"', r"'.*?'"],
            'function_patterns': [r'^\s*func\s+\w+\s*\(', r'^\s*(class|struct|enum)\s+\w+'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'kotlin': {
            'extensions': ['.kt', '.kts'],
            'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
            'string_patterns': [r'".*?"', r"'.*?'", r'""".*?"""'],
            'function_patterns': [r'^\s*fun\s+\w+\s*\(', r'^\s*(class|object|interface)\s+\w+'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'scala': {
            'extensions': ['.scala', '.sc'],
            'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
            'string_patterns': [r'".*?"', r"'.*?'", r'""".*?"""'],
            'function_patterns': [r'^\s*def\s+\w+\s*\(', r'^\s*(class|object|trait)\s+\w+'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'r': {
            'extensions': ['.r', '.R'],
            'comment_patterns': [r'#.*$'],
            'string_patterns': [r'".*?"', r"'.*?'"],
            'function_patterns': [r'^\s*\w+\s*<-\s*function\s*\(', r'^\s*function\s*\('],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'shell': {
            'extensions': ['.sh', '.bash', '.zsh', '.fish'],
            'comment_patterns': [r'#.*$'],
            'string_patterns': [r'".*?"', r"'.*?'"],
            'function_patterns': [r'^\s*\w+\s*\(\s*\)\s*\{', r'^\s*function\s+\w+'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'perl': {
            'extensions': ['.pl', '.pm'],
            'comment_patterns': [r'#.*$'],
            'string_patterns': [r'".*?"', r"'.*?'"],
            'function_patterns': [r'^\s*sub\s+\w+'],
            'indent_sensitive': False,
            'block_start': '{',
            'block_end': '}'
        },
        'lua': {
            'extensions': ['.lua'],
            'comment_patterns': [r'--.*$', r'--\[\[.*?\]\]'],
            'string_patterns': [r'".*?"', r"'.*?'", r'\[\[.*?\]\]'],
            'function_patterns': [r'^\s*function\s+\w+', r'^\s*local\s+function\s+\w+'],
            'indent_sensitive': False,
            'block_start': None,
            'block_end': 'end'
        },
        'sql': {
            'extensions': ['.sql'],
            'comment_patterns': [r'--.*$', r'/\*.*?\*/'],
            'string_patterns': [r'".*?"', r"'.*?'"],
            'function_patterns': [r'^\s*CREATE\s+(FUNCTION|PROCEDURE)\s+\w+', r'^\s*DELIMITER'],
            'indent_sensitive': False,
            'block_start': 'BEGIN',
            'block_end': 'END'
        },
        'matlab': {
            'extensions': ['.m'],
            'comment_patterns': [r'%.*$'],
            'string_patterns': [r'".*?"', r"'.*?'"],
            'function_patterns': [r'^\s*function\s+.*\s*=\s*\w+\s*\(', r'^\s*function\s+\w+\s*\('],
            'indent_sensitive': False,
            'block_start': None,
            'block_end': 'end'
        },
        'generic': {
            'extensions': [],
            'comment_patterns': [r'#.*$', r'//.*$', r'/\*.*?\*/', r'--.*$'],
            'string_patterns': [r'".*?"', r"'.*?'"],
            'function_patterns': [],
            'indent_sensitive': False,
            'block_start': None,
            'block_end': None
        }
    }

    def __init__(
        self,
        chunk_by: str = "logical_block",  # "function", "logical_block", "line_count", "indentation"
        max_lines_per_chunk: int = 50,
        language: Optional[str] = None,  # Auto-detect if None
        preserve_comments: bool = True,
        preserve_structure: bool = True,
        **kwargs
    ):
        """
        Initialize universal code chunker.

        Args:
            chunk_by: Chunking strategy
            max_lines_per_chunk: Maximum lines per chunk
            language: Force specific language (auto-detect if None)
            preserve_comments: Whether to preserve comments
            preserve_structure: Whether to preserve code structure
            **kwargs: Additional parameters
        """
        super().__init__(
            name="universal_code",
            category="code",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        self.chunk_by = chunk_by
        self.max_lines_per_chunk = max_lines_per_chunk
        self.language = language
        self.preserve_comments = preserve_comments
        self.preserve_structure = preserve_structure

        self.logger = logging.getLogger(__name__)

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk code content using universal patterns.

        Args:
            content: Code content or file path
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with code chunks
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

        # Detect language
        detected_language = self._detect_language(source_path, code_content)
        lang_config = self.LANGUAGE_CONFIGS.get(detected_language, self.LANGUAGE_CONFIGS['generic'])

        self.logger.info(f"Detected language: {detected_language}")

        try:
            lines = code_content.split('\n')

            # Extract code elements based on language
            code_elements = self._extract_code_elements(lines, lang_config)

            # Create chunks based on strategy
            chunks = self._create_chunks_from_elements(
                code_elements,
                lines,
                source_path or "direct_input",
                detected_language
            )

            # Create chunking result
            result = ChunkingResult(
                chunks=chunks,
                processing_time=time.time() - start_time,
                strategy_used="universal_code",
                source_info={
                    "source_file": str(source_path) if source_path else "direct_input",
                    "detected_language": detected_language,
                    "chunk_by": self.chunk_by,
                    "total_elements": len(code_elements),
                    "total_lines": len(lines)
                }
            )

            self.logger.info(f"Universal code chunking completed: {len(chunks)} chunks")
            return result

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            # Fallback to simple line-based chunking
            return self._fallback_line_chunking(code_content, source_path, detected_language, start_time)

    def _detect_language(self, source_path: Optional[Path], content: str) -> str:
        """Detect programming language from file extension and content."""
        if self.language:
            return self.language

        # Try detection by file extension
        if source_path and source_path.suffix:
            extension = source_path.suffix.lower()
            for lang, config in self.LANGUAGE_CONFIGS.items():
                if extension in config['extensions']:
                    return lang

        # Try detection by content patterns
        if 'def ' in content and ':' in content and 'end' not in content:
            return 'python'
        elif ('def ' in content and 'end' in content) or 'class ' in content and '@' in content:
            return 'ruby'
        elif 'function ' in content or '=>' in content or 'const ' in content:
            return 'javascript'
        elif 'public class ' in content or 'import java.' in content or 'public static void main' in content:
            return 'java'
        elif '#include' in content or 'int main(' in content or 'using namespace' in content:
            return 'c_cpp'
        elif 'func ' in content and 'package ' in content:
            return 'go'
        elif 'fn ' in content and 'use ' in content and 'mod ' in content:
            return 'rust'
        elif '<?php' in content or 'function ' in content and '$' in content:
            return 'php'
        elif 'using System' in content or 'namespace ' in content and 'class ' in content:
            return 'csharp'
        elif 'import Foundation' in content or 'func ' in content and 'var ' in content:
            return 'swift'
        elif 'fun ' in content or 'val ' in content or 'var ' in content and 'kotlin' in content.lower():
            return 'kotlin'
        elif 'def ' in content and 'object ' in content or 'trait ' in content:
            return 'scala'
        elif 'function(' in content or '<-' in content and 'library(' in content:
            return 'r'
        elif '#!/bin/bash' in content or '#!/bin/sh' in content or 'echo ' in content:
            return 'shell'
        elif 'sub ' in content and '$' in content and '@' in content:
            return 'perl'
        elif 'function ' in content and 'end' in content and 'local ' in content:
            return 'lua'
        elif 'SELECT ' in content.upper() or 'CREATE TABLE' in content.upper():
            return 'sql'
        elif 'function ' in content and '=' in content and '%' in content:
            return 'matlab'

        return 'generic'

    def _extract_code_elements(self, lines: List[str], lang_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract code elements based on language configuration."""
        elements = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Skip comments if not preserving them
            if not self.preserve_comments and self._is_comment(line, lang_config):
                i += 1
                continue

            # Extract functions/methods
            if self._is_function_start(line, lang_config):
                element = self._extract_function_element(lines, i, lang_config)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue

            # Extract logical blocks (if indentation-sensitive)
            if lang_config['indent_sensitive'] and self.chunk_by == "indentation":
                element = self._extract_indentation_block(lines, i)
                if element and element['end_lineno'] > i:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue

            # Extract brace-delimited blocks
            if lang_config['block_start'] and lang_config['block_start'] in line:
                element = self._extract_brace_block(lines, i, lang_config)
                if element and element['end_lineno'] > i:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue

            i += 1

        return elements

    def _is_comment(self, line: str, lang_config: Dict[str, Any]) -> bool:
        """Check if line is a comment."""
        for pattern in lang_config['comment_patterns']:
            if re.search(pattern, line):
                return True
        return False

    def _is_function_start(self, line: str, lang_config: Dict[str, Any]) -> bool:
        """Check if line starts a function definition."""
        for pattern in lang_config['function_patterns']:
            if re.search(pattern, line):
                return True
        return False

    def _extract_function_element(self, lines: List[str], start_idx: int, lang_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract function/method element."""
        if lang_config['indent_sensitive']:
            return self._extract_indentation_block(lines, start_idx)
        elif lang_config['block_start']:
            return self._extract_brace_block(lines, start_idx, lang_config)
        else:
            # Single line function declaration
            return {
                'type': 'function',
                'content': lines[start_idx],
                'lineno': start_idx + 1,
                'end_lineno': start_idx + 1
            }

    def _extract_indentation_block(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract code block based on indentation (Python-style)."""
        if start_idx >= len(lines):
            return None

        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        block_lines = [lines[start_idx]]

        for i in range(start_idx + 1, len(lines)):
            line = lines[i]

            # Empty lines are part of the block
            if not line.strip():
                block_lines.append(line)
                continue

            # Check indentation
            current_indent = len(line) - len(line.lstrip())

            # If indentation is less than or equal to base, block ends
            if current_indent <= base_indent:
                break

            block_lines.append(line)

        if len(block_lines) > 1:  # Only return if it's a real block
            return {
                'type': 'indentation_block',
                'content': '\n'.join(block_lines),
                'lineno': start_idx + 1,
                'end_lineno': start_idx + len(block_lines)
            }

        return None

    def _extract_brace_block(self, lines: List[str], start_idx: int, lang_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract code block based on braces (C-style)."""
        if start_idx >= len(lines):
            return None

        block_start = lang_config['block_start']
        block_end = lang_config['block_end']

        brace_count = 0
        block_lines = []
        in_block = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            block_lines.append(line)

            # Count braces
            for char in line:
                if char == block_start:
                    brace_count += 1
                    in_block = True
                elif char == block_end:
                    brace_count -= 1

                    if in_block and brace_count == 0:
                        # Block complete
                        return {
                            'type': 'brace_block',
                            'content': '\n'.join(block_lines),
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1
                        }

        # If we get here, block wasn't properly closed
        return None

    def _create_chunks_from_elements(
        self,
        elements: List[Dict[str, Any]],
        lines: List[str],
        source: str,
        language: str
    ) -> List[Chunk]:
        """Create chunks from extracted code elements."""
        chunks = []

        if self.chunk_by == "function" and elements:
            # Create one chunk per function/block
            for element in elements:
                chunk = self._create_chunk_from_element(element, source, language)
                chunks.append(chunk)

        elif self.chunk_by == "logical_block" and elements:
            # Group related elements together
            chunks = self._create_logical_blocks(elements, source, language)

        else:
            # Fallback to line-based chunking
            chunks = self._create_line_based_chunks(lines, source, language)

        return chunks

    def _create_chunk_from_element(self, element: Dict[str, Any], source: str, language: str) -> Chunk:
        """Create a chunk from a single code element."""
        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": element['lineno'], "end_line": element['end_lineno']},
            extra={
                "chunk_type": "code",
                "language": language,
                "element_type": element['type'],
                "line_count": element['end_lineno'] - element['lineno'] + 1
            }
        )

        chunk_id = f"{language}_{element['type']}_{element['lineno']}"

        return Chunk(
            id=chunk_id,
            content=element['content'],
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_logical_blocks(self, elements: List[Dict[str, Any]], source: str, language: str) -> List[Chunk]:
        """Create chunks by grouping logically related elements."""
        chunks = []
        current_block = []
        current_lines = 0

        for element in elements:
            element_lines = element['end_lineno'] - element['lineno'] + 1

            # Start new block if current would be too large
            if current_lines + element_lines > self.max_lines_per_chunk and current_block:
                chunk = self._create_block_chunk(current_block, source, language, len(chunks))
                chunks.append(chunk)
                current_block = [element]
                current_lines = element_lines
            else:
                current_block.append(element)
                current_lines += element_lines

        # Add final block
        if current_block:
            chunk = self._create_block_chunk(current_block, source, language, len(chunks))
            chunks.append(chunk)

        return chunks

    def _create_block_chunk(self, elements: List[Dict[str, Any]], source: str, language: str, chunk_index: int) -> Chunk:
        """Create a chunk from multiple elements."""
        content = '\n\n'.join(element['content'] for element in elements)

        start_line = min(element['lineno'] for element in elements)
        end_line = max(element['end_lineno'] for element in elements)

        element_types = [element['type'] for element in elements]

        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": start_line, "end_line": end_line},
            chunker_used="universal_code",
            extra={
                "chunk_type": "code",
                "language": language,
                "element_types": element_types,
                "block_size": len(elements),
                "line_count": end_line - start_line + 1
            }
        )

        return Chunk(
            id=f"{language}_block_{chunk_index}_{start_line}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_line_based_chunks(self, lines: List[str], source: str, language: str) -> List[Chunk]:
        """Create chunks based on line count."""
        chunks = []

        for i in range(0, len(lines), self.max_lines_per_chunk):
            chunk_lines = lines[i:i + self.max_lines_per_chunk]
            chunk_content = '\n'.join(chunk_lines)

            chunk = Chunk(
                id=f"{language}_lines_{i // self.max_lines_per_chunk}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=source,
                    position={"start_line": i + 1, "end_line": min(i + len(chunk_lines), len(lines))},
                    chunker_used="universal_code",
                    extra={
                        "chunk_type": "code",
                        "language": language,
                        "line_count": len(chunk_lines),
                        "chunking_method": "line_based"
                    }
                )
            )
            chunks.append(chunk)

        return chunks

    def _fallback_line_chunking(self, content: str, source_path: Optional[Path], language: str, start_time: float) -> ChunkingResult:
        """Fallback to simple line-based chunking."""
        lines = content.split('\n')
        chunks = self._create_line_based_chunks(lines, str(source_path) if source_path else "direct_input", language)

        return ChunkingResult(
            chunks=chunks,
            processing_time=time.time() - start_time,
            strategy_used="universal_code_fallback",
            source_info={
                "detected_language": language,
                "fallback": True
            }
        )

    def chunk_stream(self, content_stream, **kwargs):
        """Chunk code from a stream."""
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
        formats = []
        for config in self.LANGUAGE_CONFIGS.values():
            formats.extend([ext.lstrip('.') for ext in config['extensions']])
        return list(set(formats))  # Remove duplicates

    def estimate_chunks(self, content: Union[str, Path]) -> int:
        """Estimate number of chunks that will be generated."""
        if isinstance(content, (str, Path)):
            if isinstance(content, Path) and content.exists():
                with open(content, 'r') as f:
                    lines = len(f.readlines())
            else:
                lines = len(str(content).split('\n'))

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
            old_max = self.max_lines_per_chunk
            self.max_lines_per_chunk = max(10, int(self.max_lines_per_chunk * 0.8))
            self.logger.info(f"Adapted max_lines_per_chunk: {old_max} -> {self.max_lines_per_chunk}")
        elif feedback_type == "performance" and feedback_score < 0.5:
            old_max = self.max_lines_per_chunk
            self.max_lines_per_chunk = min(200, int(self.max_lines_per_chunk * 1.2))
            self.logger.info(f"Adapted max_lines_per_chunk: {old_max} -> {self.max_lines_per_chunk}")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return []
