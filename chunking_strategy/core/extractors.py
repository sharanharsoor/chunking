"""
Content extraction layer for universal chunking strategies.

This module provides a unified interface for extracting content from different
file types, enabling any chunking strategy to work with any file format.
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from chunking_strategy.core.base import ModalityType


class ExtractedContent:
    """Container for extracted content with metadata."""
    
    def __init__(
        self,
        text_content: str,
        modality: ModalityType = ModalityType.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
        structured_content: Optional[List[Dict[str, Any]]] = None,
        binary_content: Optional[bytes] = None
    ):
        """
        Initialize extracted content.
        
        Args:
            text_content: Plain text extracted from the source
            modality: Primary modality of the content
            metadata: Extraction metadata (file info, processing details, etc.)
            structured_content: Structured elements (functions, tables, etc.)
            binary_content: Original binary content if needed
        """
        self.text_content = text_content
        self.modality = modality
        self.metadata = metadata or {}
        self.structured_content = structured_content or []
        self.binary_content = binary_content


class BaseContentExtractor(ABC):
    """Base class for content extractors."""
    
    def __init__(self, name: str, supported_extensions: List[str]):
        """
        Initialize base extractor.
        
        Args:
            name: Extractor name
            supported_extensions: List of file extensions this extractor supports
        """
        self.name = name
        self.supported_extensions = supported_extensions
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def extract(
        self, 
        content: Union[str, bytes, Path], 
        **kwargs
    ) -> ExtractedContent:
        """
        Extract content from input.
        
        Args:
            content: Input content (file path, text, or bytes)
            **kwargs: Extractor-specific options
            
        Returns:
            ExtractedContent with extracted text and metadata
        """
        pass
    
    def supports_extension(self, extension: str) -> bool:
        """Check if this extractor supports the given file extension."""
        return extension.lower() in [ext.lower() for ext in self.supported_extensions]


class TextContentExtractor(BaseContentExtractor):
    """Extractor for plain text files."""
    
    def __init__(self):
        super().__init__(
            name="text_extractor",
            supported_extensions=[".txt", ".md", ".rst", ".log"]
        )
    
    def extract(
        self, 
        content: Union[str, bytes, Path], 
        encoding: str = "utf-8",
        **kwargs
    ) -> ExtractedContent:
        """Extract content from text files."""
        start_time = time.time()
        
        if isinstance(content, Path):
            # Read from file
            with open(content, 'r', encoding=encoding) as f:
                text_content = f.read()
            source_info = str(content)
        elif isinstance(content, bytes):
            # Decode bytes
            text_content = content.decode(encoding)
            source_info = "bytes_input"
        else:
            # Direct string content
            text_content = str(content)
            source_info = "string_input"
        
        metadata = {
            "extractor": self.name,
            "source": source_info,
            "extraction_time": time.time() - start_time,
            "encoding": encoding,
            "content_length": len(text_content),
            "line_count": len(text_content.split('\n'))
        }
        
        return ExtractedContent(
            text_content=text_content,
            modality=ModalityType.TEXT,
            metadata=metadata
        )


class PDFContentExtractor(BaseContentExtractor):
    """Extractor for PDF files."""
    
    def __init__(self):
        super().__init__(
            name="pdf_extractor", 
            supported_extensions=[".pdf"]
        )
        
        # Check for PDF libraries
        self.has_pymupdf = self._check_import("fitz")
        self.has_pypdf2 = self._check_import("PyPDF2")
        self.has_pdfminer = self._check_import("pdfminer.high_level")
    
    def _check_import(self, module_name: str) -> bool:
        """Check if a module is available."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def extract(
        self, 
        content: Union[str, bytes, Path], 
        extract_images: bool = False,
        extract_tables: bool = False,
        backend: str = "auto",
        **kwargs
    ) -> ExtractedContent:
        """Extract content from PDF files."""
        start_time = time.time()
        
        # Determine the backend to use
        if backend == "auto":
            if self.has_pymupdf:
                backend = "pymupdf"
            elif self.has_pypdf2:
                backend = "pypdf2"
            elif self.has_pdfminer:
                backend = "pdfminer"
            else:
                raise ImportError("No PDF processing library available. Install PyMuPDF, PyPDF2, or pdfminer.six")
        
        # Extract text based on backend
        if backend == "pymupdf" and self.has_pymupdf:
            text_content, structured_content = self._extract_with_pymupdf(content, extract_images, extract_tables)
        elif backend == "pypdf2" and self.has_pypdf2:
            text_content, structured_content = self._extract_with_pypdf2(content)
        elif backend == "pdfminer" and self.has_pdfminer:
            text_content, structured_content = self._extract_with_pdfminer(content)
        else:
            raise ValueError(f"Backend '{backend}' not available or not supported")
        
        # Prepare metadata
        metadata = {
            "extractor": self.name,
            "backend": backend,
            "source": str(content) if isinstance(content, Path) else "direct_input",
            "extraction_time": time.time() - start_time,
            "content_length": len(text_content),
            "structured_elements": len(structured_content),
            "extract_images": extract_images,
            "extract_tables": extract_tables
        }
        
        return ExtractedContent(
            text_content=text_content,
            modality=ModalityType.TEXT,
            metadata=metadata,
            structured_content=structured_content
        )
    
    def _extract_with_pymupdf(
        self, 
        content: Union[str, bytes, Path], 
        extract_images: bool, 
        extract_tables: bool
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract using PyMuPDF."""
        import fitz
        
        # Open document
        if isinstance(content, Path):
            doc = fitz.open(str(content))
        else:
            # Handle bytes or string content
            doc = fitz.open(stream=content if isinstance(content, bytes) else content.encode())
        
        text_parts = []
        structured_content = []
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            # Extract text
            page_text = page.get_text()
            text_parts.append(page_text)
            
            # Extract images if requested
            if extract_images:
                images = page.get_images()
                for img_index, img in enumerate(images):
                    structured_content.append({
                        "type": "image",
                        "page": page_num + 1,
                        "index": img_index,
                        "bbox": None,  # Could extract bbox if needed
                        "content": f"[Image {img_index + 1} on page {page_num + 1}]"
                    })
            
            # Extract tables if requested (basic implementation)
            if extract_tables:
                tables = page.find_tables()
                for table_index, table in enumerate(tables):
                    try:
                        table_data = table.extract()
                        table_text = "\n".join(["\t".join(row) for row in table_data])
                        structured_content.append({
                            "type": "table",
                            "page": page_num + 1,
                            "index": table_index,
                            "content": table_text
                        })
                    except:
                        # Fallback if table extraction fails
                        structured_content.append({
                            "type": "table",
                            "page": page_num + 1,
                            "index": table_index,
                            "content": f"[Table {table_index + 1} on page {page_num + 1}]"
                        })
        
        doc.close()
        return "\n\n".join(text_parts), structured_content
    
    def _extract_with_pypdf2(self, content: Union[str, bytes, Path]) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract using PyPDF2."""
        from PyPDF2 import PdfReader
        
        if isinstance(content, Path):
            reader = PdfReader(str(content))
        else:
            # Handle bytes content
            from io import BytesIO
            content_bytes = content if isinstance(content, bytes) else content.encode()
            reader = PdfReader(BytesIO(content_bytes))
        
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text_parts.append(page_text)
        
        return "\n\n".join(text_parts), []
    
    def _extract_with_pdfminer(self, content: Union[str, bytes, Path]) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract using pdfminer."""
        from pdfminer.high_level import extract_text
        
        if isinstance(content, Path):
            text_content = extract_text(str(content))
        else:
            # Handle bytes content
            from io import BytesIO
            content_bytes = content if isinstance(content, bytes) else content.encode()
            text_content = extract_text(BytesIO(content_bytes))
        
        return text_content, []


class CodeContentExtractor(BaseContentExtractor):
    """Extractor for source code files."""
    
    def __init__(self):
        super().__init__(
            name="code_extractor",
            supported_extensions=[
                ".py", ".pyx", ".pyi",  # Python
                ".js", ".jsx", ".ts", ".tsx",  # JavaScript/TypeScript
                ".java",  # Java
                ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx",  # C/C++
                ".go",  # Go
                ".rs",  # Rust
                ".rb",  # Ruby
                ".php",  # PHP
                ".cs",  # C#
                ".swift",  # Swift
                ".kt",  # Kotlin
                ".scala",  # Scala
                ".r", ".R",  # R
                ".m",  # Objective-C/MATLAB
                ".sh", ".bash",  # Shell scripts
                ".sql",  # SQL
                ".xml", ".html", ".css",  # Markup/styling
                ".json", ".yaml", ".yml",  # Data formats
            ]
        )
    
    def extract(
        self, 
        content: Union[str, bytes, Path], 
        preserve_structure: bool = True,
        include_comments: bool = True,
        encoding: str = "utf-8",
        **kwargs
    ) -> ExtractedContent:
        """Extract content from source code files."""
        start_time = time.time()
        
        # Get the text content
        if isinstance(content, Path):
            with open(content, 'r', encoding=encoding) as f:
                text_content = f.read()
            source_info = str(content)
            file_extension = content.suffix.lower()
        elif isinstance(content, bytes):
            text_content = content.decode(encoding)
            source_info = "bytes_input"
            file_extension = kwargs.get('extension', '.txt')
        else:
            text_content = str(content)
            source_info = "string_input"
            file_extension = kwargs.get('extension', '.txt')
        
        structured_content = []
        
        # Basic structure extraction (can be enhanced)
        if preserve_structure:
            lines = text_content.split('\n')
            
            # Detect functions, classes, etc. (basic patterns)
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                # Function patterns for different languages
                if any(pattern in line_stripped for pattern in [
                    'def ', 'function ', 'func ', 'fn ', 'sub ', 'procedure '
                ]):
                    structured_content.append({
                        "type": "function",
                        "line": i + 1,
                        "content": line_stripped,
                        "language": self._detect_language(file_extension)
                    })
                
                # Class patterns
                elif any(pattern in line_stripped for pattern in [
                    'class ', 'interface ', 'struct ', 'type ', 'enum '
                ]):
                    structured_content.append({
                        "type": "class",
                        "line": i + 1,
                        "content": line_stripped,
                        "language": self._detect_language(file_extension)
                    })
        
        # Remove comments if requested
        if not include_comments:
            text_content = self._remove_comments(text_content, file_extension)
        
        metadata = {
            "extractor": self.name,
            "source": source_info,
            "extraction_time": time.time() - start_time,
            "encoding": encoding,
            "file_extension": file_extension,
            "language": self._detect_language(file_extension),
            "content_length": len(text_content),
            "line_count": len(text_content.split('\n')),
            "structured_elements": len(structured_content),
            "preserve_structure": preserve_structure,
            "include_comments": include_comments
        }
        
        return ExtractedContent(
            text_content=text_content,
            modality=ModalityType.TEXT,
            metadata=metadata,
            structured_content=structured_content
        )
    
    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension."""
        language_map = {
            '.py': 'python', '.pyx': 'python', '.pyi': 'python',
            '.js': 'javascript', '.jsx': 'javascript', '.ts': 'typescript', '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c', '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
            '.h': 'c', '.hpp': 'cpp', '.hxx': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r', '.R': 'r',
            '.m': 'objective-c',
            '.sh': 'shell', '.bash': 'shell',
            '.sql': 'sql',
            '.xml': 'xml', '.html': 'html', '.css': 'css',
            '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml'
        }
        return language_map.get(extension.lower(), 'unknown')
    
    def _remove_comments(self, text: str, extension: str) -> str:
        """Remove comments from code (basic implementation)."""
        import re
        
        if extension in ['.py', '.rb', '.sh', '.bash']:
            # Remove # comments
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                # Simple comment removal (doesn't handle strings properly)
                if '#' in line and not ('"' in line or "'" in line):
                    line = line.split('#')[0]
                cleaned_lines.append(line)
            return '\n'.join(cleaned_lines)
        
        elif extension in ['.js', '.java', '.c', '.cpp', '.go', '.rs', '.cs']:
            # Remove // and /* */ comments (basic)
            text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
            return text
        
        return text  # No comment removal for unknown formats


class ExtractorRegistry:
    """Registry for content extractors."""
    
    def __init__(self):
        self.extractors: List[BaseContentExtractor] = []
        self.logger = logging.getLogger(__name__)
        
        # Register default extractors
        self._register_default_extractors()
    
    def _register_default_extractors(self):
        """Register default content extractors."""
        self.register(TextContentExtractor())
        self.register(CodeContentExtractor())
        
        # Register PDF extractor if dependencies are available
        pdf_extractor = PDFContentExtractor()
        if pdf_extractor.has_pymupdf or pdf_extractor.has_pypdf2 or pdf_extractor.has_pdfminer:
            self.register(pdf_extractor)
    
    def register(self, extractor: BaseContentExtractor):
        """Register a content extractor."""
        self.extractors.append(extractor)
        self.logger.info(f"Registered extractor: {extractor.name}")
    
    def get_extractor(self, extension: str) -> Optional[BaseContentExtractor]:
        """Get the appropriate extractor for a file extension."""
        for extractor in self.extractors:
            if extractor.supports_extension(extension):
                return extractor
        return None
    
    def get_extractor_by_name(self, name: str) -> Optional[BaseContentExtractor]:
        """Get extractor by name."""
        for extractor in self.extractors:
            if extractor.name == name:
                return extractor
        return None
    
    def list_extractors(self) -> List[str]:
        """List all registered extractor names."""
        return [extractor.name for extractor in self.extractors]
    
    def list_supported_extensions(self) -> List[str]:
        """List all supported file extensions."""
        extensions = set()
        for extractor in self.extractors:
            extensions.update(extractor.supported_extensions)
        return sorted(list(extensions))


# Global extractor registry instance
_global_extractor_registry = ExtractorRegistry()


def get_extractor_registry() -> ExtractorRegistry:
    """Get the global extractor registry."""
    return _global_extractor_registry


def extract_content(
    content: Union[str, bytes, Path], 
    extractor_name: Optional[str] = None,
    **kwargs
) -> ExtractedContent:
    """
    Extract content using the appropriate extractor.
    
    Args:
        content: Input content (file path, text, or bytes)
        extractor_name: Specific extractor to use (auto-detect if None)
        **kwargs: Extractor-specific options
        
    Returns:
        ExtractedContent with extracted text and metadata
    """
    registry = get_extractor_registry()
    
    if extractor_name:
        # Use specific extractor
        extractor = registry.get_extractor_by_name(extractor_name)
        if not extractor:
            raise ValueError(f"Extractor '{extractor_name}' not found")
    else:
        # Auto-detect extractor
        if isinstance(content, Path):
            extension = content.suffix
        else:
            extension = kwargs.get('extension', '.txt')
        
        extractor = registry.get_extractor(extension)
        if not extractor:
            # Fallback to text extractor
            extractor = registry.get_extractor_by_name('text_extractor')
    
    if not extractor:
        raise ValueError("No suitable extractor found")
    
    return extractor.extract(content, **kwargs)
