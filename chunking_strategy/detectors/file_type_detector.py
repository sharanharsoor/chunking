"""
File type detection for chunking strategy selection.

This module provides file type detection capabilities to help the orchestrator
select appropriate chunking strategies based on file characteristics.
"""

import logging
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os

from chunking_strategy.core.base import ModalityType

logger = logging.getLogger(__name__)


class FileTypeDetector:
    """
    Detects file types and maps them to appropriate modalities and strategies.

    This detector uses file extensions, MIME types, and content inspection
    to determine the type of file and suggest appropriate chunking approaches.
    """

    def __init__(self):
        """Initialize file type detector."""
        self.logger = logging.getLogger(f"{__name__}.FileTypeDetector")

        # File extension to modality mapping
        self.extension_map = {
            # Text files
            '.txt': ('text/plain', ModalityType.TEXT),
            '.md': ('text/markdown', ModalityType.TEXT),
            '.rst': ('text/x-rst', ModalityType.TEXT),
            '.log': ('text/plain', ModalityType.TEXT),
            '.csv': ('text/csv', ModalityType.TEXT),
            '.tsv': ('text/tab-separated-values', ModalityType.TEXT),

            # Document files
            '.pdf': ('application/pdf', ModalityType.MIXED),
            '.doc': ('application/msword', ModalityType.TEXT),
            '.docx': ('application/vnd.openxmlformats-officedocument.wordprocessingml.document', ModalityType.TEXT),
            '.odt': ('application/vnd.oasis.opendocument.text', ModalityType.TEXT),
            '.rtf': ('application/rtf', ModalityType.TEXT),

            # Structured data
            '.xml': ('application/xml', ModalityType.TEXT),
            '.html': ('text/html', ModalityType.TEXT),
            '.htm': ('text/html', ModalityType.TEXT),
            '.json': ('application/json', ModalityType.TEXT),
            '.yaml': ('application/x-yaml', ModalityType.TEXT),
            '.yml': ('application/x-yaml', ModalityType.TEXT),
            '.toml': ('application/toml', ModalityType.TEXT),

            # Code files - Python
            '.py': ('text/x-python', ModalityType.TEXT),
            '.pyx': ('text/x-python', ModalityType.TEXT),
            '.pyi': ('text/x-python', ModalityType.TEXT),

            # Code files - JavaScript/TypeScript
            '.js': ('application/javascript', ModalityType.TEXT),
            '.jsx': ('application/javascript', ModalityType.TEXT),
            '.ts': ('application/typescript', ModalityType.TEXT),
            '.tsx': ('application/typescript', ModalityType.TEXT),

            # Code files - C/C++
            '.c': ('text/x-csrc', ModalityType.TEXT),
            '.cpp': ('text/x-c++src', ModalityType.TEXT),
            '.cc': ('text/x-c++src', ModalityType.TEXT),
            '.cxx': ('text/x-c++src', ModalityType.TEXT),
            '.h': ('text/x-chdr', ModalityType.TEXT),
            '.hpp': ('text/x-c++hdr', ModalityType.TEXT),
            '.hxx': ('text/x-c++hdr', ModalityType.TEXT),

            # Code files - JVM languages
            '.java': ('text/x-java-source', ModalityType.TEXT),
            '.kt': ('text/x-kotlin', ModalityType.TEXT),
            '.kts': ('text/x-kotlin', ModalityType.TEXT),
            '.scala': ('text/x-scala', ModalityType.TEXT),
            '.sc': ('text/x-scala', ModalityType.TEXT),

            # Code files - Other popular languages
            '.go': ('text/x-go', ModalityType.TEXT),
            '.rs': ('text/x-rust', ModalityType.TEXT),
            '.rb': ('application/x-ruby', ModalityType.TEXT),
            '.rbw': ('application/x-ruby', ModalityType.TEXT),
            '.php': ('application/x-httpd-php', ModalityType.TEXT),
            '.phtml': ('application/x-httpd-php', ModalityType.TEXT),
            '.cs': ('text/x-csharp', ModalityType.TEXT),
            '.swift': ('text/x-swift', ModalityType.TEXT),

            # Code files - Scripting and data languages
            '.r': ('text/x-r', ModalityType.TEXT),
            '.R': ('text/x-r', ModalityType.TEXT),
            '.m': ('text/x-matlab', ModalityType.TEXT),
            '.pl': ('text/x-perl', ModalityType.TEXT),
            '.pm': ('text/x-perl', ModalityType.TEXT),
            '.sh': ('application/x-sh', ModalityType.TEXT),
            '.bash': ('application/x-sh', ModalityType.TEXT),
            '.zsh': ('application/x-sh', ModalityType.TEXT),
            '.fish': ('application/x-sh', ModalityType.TEXT),
            '.lua': ('text/x-lua', ModalityType.TEXT),
            '.vim': ('text/x-vim', ModalityType.TEXT),
            '.sql': ('application/sql', ModalityType.TEXT),
            '.dockerfile': ('text/x-dockerfile', ModalityType.TEXT),

            # Image files
            '.jpg': ('image/jpeg', ModalityType.IMAGE),
            '.jpeg': ('image/jpeg', ModalityType.IMAGE),
            '.png': ('image/png', ModalityType.IMAGE),
            '.gif': ('image/gif', ModalityType.IMAGE),
            '.bmp': ('image/bmp', ModalityType.IMAGE),
            '.tiff': ('image/tiff', ModalityType.IMAGE),
            '.svg': ('image/svg+xml', ModalityType.IMAGE),
            '.webp': ('image/webp', ModalityType.IMAGE),

            # Audio files
            '.mp3': ('audio/mpeg', ModalityType.AUDIO),
            '.wav': ('audio/wav', ModalityType.AUDIO),
            '.flac': ('audio/flac', ModalityType.AUDIO),
            '.aac': ('audio/aac', ModalityType.AUDIO),
            '.ogg': ('audio/ogg', ModalityType.AUDIO),
            '.m4a': ('audio/mp4', ModalityType.AUDIO),

            # Video files
            '.mp4': ('video/mp4', ModalityType.VIDEO),
            '.avi': ('video/x-msvideo', ModalityType.VIDEO),
            '.mov': ('video/quicktime', ModalityType.VIDEO),
            '.wmv': ('video/x-ms-wmv', ModalityType.VIDEO),
            '.flv': ('video/x-flv', ModalityType.VIDEO),
            '.webm': ('video/webm', ModalityType.VIDEO),
            '.mkv': ('video/x-matroska', ModalityType.VIDEO),

            # Archive files
            '.zip': ('application/zip', ModalityType.MIXED),
            '.tar': ('application/x-tar', ModalityType.MIXED),
            '.gz': ('application/gzip', ModalityType.MIXED),
            '.rar': ('application/vnd.rar', ModalityType.MIXED),
            '.7z': ('application/x-7z-compressed', ModalityType.MIXED),
        }

        # Content type patterns for text detection
        self.text_indicators = [
            b'<?xml',           # XML files
            b'<!DOCTYPE',       # HTML files
            b'<html',           # HTML files
            b'{\n',             # JSON-like files
            b'[\n',             # JSON arrays
            b'---\n',           # YAML front matter
            b'#include',        # C/C++ files
            b'import ',         # Python/JS imports
            b'package ',        # Java/Go packages
            b'function ',       # Various languages
            b'class ',          # OOP languages
        ]

    def detect(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Detect file type and characteristics.

        Args:
            file_path: Path to the file to analyze

        Returns:
            Dictionary with file type information
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "file_extension": file_path.suffix.lower(),
        }

        # Detect by extension first
        extension = file_path.suffix.lower()
        if extension in self.extension_map:
            mime_type, modality = self.extension_map[extension]
            result.update({
                "file_type": self._get_file_category(extension),
                "mime_type": mime_type,
                "modality": modality,
                "detection_method": "extension"
            })
        else:
            # Try MIME type detection
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                modality = self._mime_to_modality(mime_type)
                result.update({
                    "file_type": self._mime_to_category(mime_type),
                    "mime_type": mime_type,
                    "modality": modality,
                    "detection_method": "mime"
                })
            else:
                # Fallback to content inspection
                content_result = self._detect_by_content(file_path)
                result.update(content_result)

        # Add additional metadata
        result.update(self._get_additional_metadata(file_path))

        self.logger.debug(f"Detected file type: {result}")
        return result

    def detect_modality(self, file_path: Union[str, Path]) -> ModalityType:
        """
        Quick modality detection for a file.

        Args:
            file_path: Path to the file

        Returns:
            Detected modality type
        """
        result = self.detect(file_path)
        return result.get("modality", ModalityType.TEXT)

    def is_text_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file is a text file.

        Args:
            file_path: Path to the file

        Returns:
            True if file appears to be text
        """
        try:
            modality = self.detect_modality(file_path)
            return modality == ModalityType.TEXT
        except Exception:
            return False

    def get_suggested_strategies(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get suggested chunking strategies for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with strategy suggestions
        """
        detection_result = self.detect(file_path)
        file_type = detection_result.get("file_type", "unknown")
        modality = detection_result.get("modality", ModalityType.TEXT)

        suggestions = {
            "primary_strategies": [],
            "fallback_strategies": [],
            "unsuitable_strategies": [],
            "special_considerations": []
        }

        if modality == ModalityType.TEXT:
            if file_type == "code":
                suggestions["primary_strategies"] = ["ast_based", "code_generic", "syntactic"]
                suggestions["fallback_strategies"] = ["fixed_size", "sentence_based"]
                suggestions["special_considerations"] = ["Preserve code structure", "Respect indentation"]

            elif file_type == "document":
                suggestions["primary_strategies"] = ["paragraph_based", "section_based", "semantic_basic"]
                suggestions["fallback_strategies"] = ["sentence_based", "fixed_size"]
                suggestions["special_considerations"] = ["Preserve document structure", "Handle formatting"]

            elif file_type == "structured":
                suggestions["primary_strategies"] = ["structure_aware", "regex_based"]
                suggestions["fallback_strategies"] = ["fixed_size", "boundary_aware"]
                suggestions["special_considerations"] = ["Respect data structure", "Preserve syntax"]

            else:  # plain text
                suggestions["primary_strategies"] = ["sentence_based", "paragraph_based"]
                suggestions["fallback_strategies"] = ["fixed_size", "word_fixed_length"]

        elif modality == ModalityType.IMAGE:
            suggestions["primary_strategies"] = ["tile_based", "patch_based", "region_based"]
            suggestions["fallback_strategies"] = ["image_block_based", "fixed_size"]
            suggestions["special_considerations"] = ["Maintain spatial relationships", "Consider image size"]

        elif modality == ModalityType.AUDIO:
            suggestions["primary_strategies"] = ["audio_silence_based", "audio_frame_based"]
            suggestions["fallback_strategies"] = ["audio_sample_based", "fixed_size"]
            suggestions["special_considerations"] = ["Preserve temporal structure", "Consider sample rate"]

        elif modality == ModalityType.VIDEO:
            suggestions["primary_strategies"] = ["video_scene_based", "video_keyframe"]
            suggestions["fallback_strategies"] = ["video_frame_based", "fixed_size"]
            suggestions["special_considerations"] = ["Maintain temporal coherence", "Consider frame rate"]

        else:  # MIXED or unknown
            suggestions["primary_strategies"] = ["fixed_size", "boundary_aware"]
            suggestions["fallback_strategies"] = ["rolling_hash", "rabin_cdc"]
            suggestions["special_considerations"] = ["Generic approach needed", "Consider content structure"]

        return suggestions

    def _get_file_category(self, extension: str) -> str:
        """Get file category from extension."""
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.sql'}
        doc_extensions = {'.pdf', '.doc', '.docx', '.odt', '.rtf'}
        structured_extensions = {'.xml', '.html', '.htm', '.json', '.yaml', '.yml', '.toml', '.csv', '.tsv'}

        if extension in code_extensions:
            return "code"
        elif extension in doc_extensions:
            return "document"
        elif extension in structured_extensions:
            return "structured"
        elif extension in {'.txt', '.md', '.rst', '.log'}:
            return "text"
        else:
            return "unknown"

    def _mime_to_modality(self, mime_type: str) -> ModalityType:
        """Convert MIME type to modality."""
        if mime_type.startswith('text/'):
            return ModalityType.TEXT
        elif mime_type.startswith('image/'):
            return ModalityType.IMAGE
        elif mime_type.startswith('audio/'):
            return ModalityType.AUDIO
        elif mime_type.startswith('video/'):
            return ModalityType.VIDEO
        elif mime_type in ['application/json', 'application/xml', 'application/x-yaml']:
            return ModalityType.TEXT
        else:
            return ModalityType.MIXED

    def _mime_to_category(self, mime_type: str) -> str:
        """Convert MIME type to file category."""
        if mime_type.startswith('text/'):
            return "text"
        elif mime_type.startswith('application/') and any(x in mime_type for x in ['json', 'xml', 'yaml']):
            return "structured"
        elif mime_type.startswith('image/'):
            return "image"
        elif mime_type.startswith('audio/'):
            return "audio"
        elif mime_type.startswith('video/'):
            return "video"
        else:
            return "binary"

    def _detect_by_content(self, file_path: Path) -> Dict[str, Any]:
        """Detect file type by examining content."""
        try:
            # Read first few KB to analyze
            with open(file_path, 'rb') as f:
                header = f.read(8192)  # 8KB should be enough

            if not header:
                return {
                    "file_type": "empty",
                    "mime_type": "application/octet-stream",
                    "modality": ModalityType.TEXT,
                    "detection_method": "content"
                }

            # Check for text content
            if self._is_text_content(header):
                return {
                    "file_type": "text",
                    "mime_type": "text/plain",
                    "modality": ModalityType.TEXT,
                    "detection_method": "content"
                }

            # Check for specific binary formats
            if header.startswith(b'\x89PNG'):
                return {"file_type": "image", "mime_type": "image/png", "modality": ModalityType.IMAGE, "detection_method": "content"}
            elif header.startswith(b'\xFF\xD8\xFF'):
                return {"file_type": "image", "mime_type": "image/jpeg", "modality": ModalityType.IMAGE, "detection_method": "content"}
            elif header.startswith(b'%PDF'):
                return {"file_type": "document", "mime_type": "application/pdf", "modality": ModalityType.MIXED, "detection_method": "content"}
            elif header.startswith(b'PK\x03\x04'):
                return {"file_type": "archive", "mime_type": "application/zip", "modality": ModalityType.MIXED, "detection_method": "content"}

            # Default for binary content
            return {
                "file_type": "binary",
                "mime_type": "application/octet-stream",
                "modality": ModalityType.MIXED,
                "detection_method": "content"
            }

        except Exception as e:
            self.logger.error(f"Error reading file content: {e}")
            return {
                "file_type": "unknown",
                "mime_type": "application/octet-stream",
                "modality": ModalityType.MIXED,
                "detection_method": "error"
            }

    def _is_text_content(self, content: bytes) -> bool:
        """Check if content appears to be text."""
        try:
            # Try to decode as UTF-8
            content.decode('utf-8')

            # Check for text indicators
            if any(indicator in content for indicator in self.text_indicators):
                return True

            # Check character distribution
            printable_chars = sum(1 for b in content if 32 <= b <= 126 or b in [9, 10, 13])
            total_chars = len(content)

            if total_chars == 0:
                return True

            printable_ratio = printable_chars / total_chars
            return printable_ratio > 0.7  # 70% printable characters

        except UnicodeDecodeError:
            return False

    def _get_additional_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get additional file metadata."""
        try:
            stat = file_path.stat()
            return {
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime,
                "permissions": oct(stat.st_mode)[-3:],
                "is_readable": os.access(file_path, os.R_OK),
                "is_writable": os.access(file_path, os.W_OK),
            }
        except Exception as e:
            self.logger.warning(f"Could not get additional metadata: {e}")
            return {}
