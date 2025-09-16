"""
Apache Tika integration for enhanced file processing.

Provides unified content extraction, metadata analysis, and file type detection
using Apache Tika's powerful document processing capabilities.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

# Optional Tika dependencies
try:
    from tika import parser, detector
    HAS_TIKA = True
except ImportError:
    HAS_TIKA = False

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

from chunking_strategy.core.base import ChunkMetadata, ModalityType


class TikaProcessor:
    """
    Apache Tika integration for comprehensive document processing.
    
    Provides unified content extraction and metadata analysis across
    1,400+ file formats supported by Apache Tika.
    """
    
    def __init__(self, 
                 server_url: Optional[str] = None,
                 timeout: int = 30,
                 max_string_length: int = 100 * 1024 * 1024):  # 100MB
        """
        Initialize Tika processor.
        
        Args:
            server_url: Custom Tika server URL (uses local by default)
            timeout: Request timeout in seconds
            max_string_length: Maximum string length for extraction
        """
        self.logger = logging.getLogger(__name__)
        self.server_url = server_url
        self.timeout = timeout
        self.max_string_length = max_string_length
        
        if not HAS_TIKA:
            self.logger.warning("Apache Tika not available. Install with: pip install tika")
    
    def is_available(self) -> bool:
        """Check if Tika is available for use."""
        return HAS_TIKA
    
    def extract_content_and_metadata(self, 
                                   file_path: Union[str, Path]) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text content and metadata from a file using Tika.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
            
        Raises:
            ImportError: If Tika is not available
            Exception: If extraction fails
        """
        if not HAS_TIKA:
            raise ImportError("Apache Tika not available. Install with: pip install tika")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Parse document with Tika
            parsed = parser.from_file(
                str(file_path),
                serverEndpoint=self.server_url,
                requestOptions={'timeout': self.timeout}
            )
            
            # Extract content and metadata
            content = parsed.get('content', '') or ''
            metadata = parsed.get('metadata', {}) or {}
            
            # Limit content length to prevent memory issues
            if len(content) > self.max_string_length:
                self.logger.warning(f"Content truncated from {len(content)} to {self.max_string_length} chars")
                content = content[:self.max_string_length]
            
            # Clean and normalize content
            content = self._clean_content(content)
            
            # Enhance metadata
            metadata = self._enhance_metadata(metadata, file_path)
            
            self.logger.info(f"Extracted {len(content)} characters from {file_path.name}")
            return content, metadata
            
        except Exception as e:
            self.logger.error(f"Tika extraction failed for {file_path}: {e}")
            raise
    
    def detect_file_type(self, file_path: Union[str, Path]) -> Dict[str, str]:
        """
        Detect file type and MIME information using Tika.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file type information
        """
        if not HAS_TIKA:
            return self._fallback_file_detection(file_path)
        
        file_path = Path(file_path)
        
        try:
            # Use Tika detector
            mime_type = detector.from_file(str(file_path))
            
            # Get additional info
            file_info = {
                'mime_type': mime_type,
                'extension': file_path.suffix.lower(),
                'size_bytes': file_path.stat().st_size,
                'tika_detected': True
            }
            
            # Determine modality from MIME type
            file_info['modality'] = self._mime_to_modality(mime_type)
            
            return file_info
            
        except Exception as e:
            self.logger.warning(f"Tika detection failed for {file_path}: {e}")
            return self._fallback_file_detection(file_path)
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize extracted content."""
        if not content:
            return ""
        
        # Remove excessive whitespace
        import re
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Max 2 consecutive newlines
        content = re.sub(r'[ \t]+', ' ', content)           # Normalize spaces
        content = content.strip()
        
        return content
    
    def _enhance_metadata(self, metadata: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Enhance metadata with additional information."""
        enhanced = metadata.copy()
        
        # Add file information
        enhanced.update({
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower(),
            'extraction_method': 'apache_tika'
        })
        
        # Normalize common metadata fields
        if 'Content-Type' in enhanced:
            enhanced['mime_type'] = enhanced['Content-Type']
        
        # Extract language if available
        if 'Content-Language' in enhanced:
            enhanced['language'] = enhanced['Content-Language']
        elif 'language' in enhanced:
            enhanced['language'] = enhanced['language']
        
        # Extract creation date
        for date_field in ['Creation-Date', 'created', 'dcterms:created']:
            if date_field in enhanced:
                enhanced['created_date'] = enhanced[date_field]
                break
        
        # Extract author information
        for author_field in ['Author', 'creator', 'dc:creator']:
            if author_field in enhanced:
                enhanced['author'] = enhanced[author_field]
                break
        
        # Extract title
        for title_field in ['title', 'dc:title', 'Title']:
            if title_field in enhanced:
                enhanced['title'] = enhanced[title_field]
                break
        
        return enhanced
    
    def _mime_to_modality(self, mime_type: str) -> ModalityType:
        """Convert MIME type to modality type."""
        if not mime_type:
            return ModalityType.TEXT
        
        mime_lower = mime_type.lower()
        
        if mime_lower.startswith('text/'):
            return ModalityType.TEXT
        elif mime_lower.startswith('image/'):
            return ModalityType.IMAGE
        elif mime_lower.startswith('audio/'):
            return ModalityType.AUDIO
        elif mime_lower.startswith('video/'):
            return ModalityType.VIDEO
        elif 'pdf' in mime_lower:
            return ModalityType.TEXT  # PDFs are primarily text
        elif any(x in mime_lower for x in ['document', 'word', 'excel', 'spreadsheet']):
            return ModalityType.TEXT
        else:
            return ModalityType.TEXT  # Default to text
    
    def _fallback_file_detection(self, file_path: Union[str, Path]) -> Dict[str, str]:
        """Fallback file detection when Tika is not available."""
        file_path = Path(file_path)
        
        file_info = {
            'extension': file_path.suffix.lower(),
            'size_bytes': file_path.stat().st_size if file_path.exists() else 0,
            'tika_detected': False
        }
        
        # Basic MIME type detection
        if HAS_MAGIC:
            try:
                mime = magic.Magic(mime=True)
                file_info['mime_type'] = mime.from_file(str(file_path))
            except Exception:
                file_info['mime_type'] = self._guess_mime_from_extension(file_path.suffix)
        else:
            file_info['mime_type'] = self._guess_mime_from_extension(file_path.suffix)
        
        file_info['modality'] = self._mime_to_modality(file_info['mime_type'])
        
        return file_info
    
    def _guess_mime_from_extension(self, extension: str) -> str:
        """Basic MIME type guessing from file extension."""
        ext_to_mime = {
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.html': 'text/html',
            '.xml': 'text/xml',
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.py': 'text/x-python',
            '.cpp': 'text/x-c++src',
            '.c': 'text/x-csrc',
            '.java': 'text/x-java-source',
            '.js': 'text/javascript',
            '.md': 'text/markdown',
            '.jpg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo'
        }
        
        return ext_to_mime.get(extension.lower(), 'application/octet-stream')
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        if not HAS_TIKA:
            return ['txt', 'pdf', 'md', 'html', 'xml', 'json', 'csv']
        
        # Common formats supported by Tika
        return [
            # Text formats
            'txt', 'md', 'rst', 'tex',
            # Document formats
            'pdf', 'doc', 'docx', 'odt', 'rtf',
            # Spreadsheet formats
            'xls', 'xlsx', 'ods', 'csv',
            # Presentation formats
            'ppt', 'pptx', 'odp',
            # Web formats
            'html', 'xml', 'json',
            # Code formats
            'py', 'java', 'cpp', 'c', 'js', 'ts', 'go', 'rs',
            # Archive formats
            'zip', 'tar', 'gz', '7z',
            # Image formats (with OCR)
            'jpg', 'png', 'gif', 'tiff', 'bmp',
            # Audio formats
            'mp3', 'wav', 'flac', 'ogg',
            # Video formats
            'mp4', 'avi', 'mov', 'wmv'
        ]


# Global Tika processor instance
_tika_processor = None

def get_tika_processor(**kwargs) -> TikaProcessor:
    """Get global Tika processor instance."""
    global _tika_processor
    if _tika_processor is None:
        _tika_processor = TikaProcessor(**kwargs)
    return _tika_processor

def extract_with_tika(file_path: Union[str, Path]) -> Tuple[str, Dict[str, Any]]:
    """Convenience function for Tika extraction."""
    processor = get_tika_processor()
    return processor.extract_content_and_metadata(file_path)

def detect_file_type_with_tika(file_path: Union[str, Path]) -> Dict[str, str]:
    """Convenience function for file type detection."""
    processor = get_tika_processor()
    return processor.detect_file_type(file_path)
