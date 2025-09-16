"""
Preprocessing utilities for content preparation before chunking.

This module provides preprocessing capabilities to clean, normalize, and
prepare content for optimal chunking results.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Pipeline for preprocessing content before chunking.

    Provides various preprocessing operations to clean and normalize
    content for better chunking results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessing pipeline.

        Args:
            config: Configuration for preprocessing steps
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.PreprocessingPipeline")

    def process(
        self,
        content: Union[str, bytes],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Union[str, bytes]:
        """
        Process content through preprocessing pipeline.

        Args:
            content: Content to preprocess
            metadata: Metadata about the content

        Returns:
            Preprocessed content
        """
        if isinstance(content, bytes):
            return self._process_binary_content(content, metadata)
        else:
            return self._process_text_content(content, metadata)

    def _process_text_content(self, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Process text content."""
        processed = content

        # Normalize whitespace
        if self.config.get("normalize_whitespace", True):
            processed = self._normalize_whitespace(processed)

        # Remove headers and footers
        if self.config.get("remove_headers_footers", False):
            processed = self._remove_headers_footers(processed)

        # Clean markup
        if self.config.get("clean_markup", False):
            processed = self._clean_markup(processed)

        # Normalize unicode
        if self.config.get("normalize_unicode", False):
            processed = self._normalize_unicode(processed)

        # Custom preprocessing
        if "custom_processors" in self.config:
            for processor in self.config["custom_processors"]:
                processed = processor(processed)

        return processed

    def _process_binary_content(self, content: bytes, metadata: Optional[Dict[str, Any]]) -> bytes:
        """Process binary content (placeholder for future implementation)."""
        # For now, return content unchanged
        # Future: Could add binary preprocessing like decompression, etc.
        return content

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with double newline (paragraph separation)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        return text.strip()

    def _remove_headers_footers(self, text: str) -> str:
        """Remove common headers and footers."""
        # This is a simplified implementation
        # Real implementation would need more sophisticated detection

        lines = text.split('\n')
        if not lines:
            return text

        # Simple heuristics for header/footer detection
        processed_lines = []
        skip_start = 0
        skip_end = 0

        # Skip short lines at the beginning (potential headers)
        for i, line in enumerate(lines[:5]):  # Check first 5 lines
            if len(line.strip()) < 20 and any(word in line.lower() for word in ['page', 'chapter', 'header']):
                skip_start = i + 1
            else:
                break

        # Skip short lines at the end (potential footers)
        for i, line in enumerate(reversed(lines[-5:])):  # Check last 5 lines
            if len(line.strip()) < 20 and any(word in line.lower() for word in ['page', 'footer', 'copyright']):
                skip_end = i + 1
            else:
                break

        end_index = len(lines) - skip_end if skip_end > 0 else len(lines)
        processed_lines = lines[skip_start:end_index]

        return '\n'.join(processed_lines)

    def _clean_markup(self, text: str) -> str:
        """Clean markup and formatting from text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Inline code
        text = re.sub(r'#{1,6}\s*', '', text)         # Headers
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links

        # Remove extra markup characters
        text = re.sub(r'[_~`]', '', text)

        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata

        # Normalize to NFKC form
        text = unicodedata.normalize('NFKC', text)

        # Replace various unicode spaces with regular space
        text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]', ' ', text)

        # Replace various unicode dashes with regular dash
        text = re.sub(r'[\u2010-\u2015]', '-', text)

        # Replace various unicode quotes with regular quotes
        text = re.sub(r'[\u2018\u2019]', "'", text)
        text = re.sub(r'[\u201C\u201D]', '"', text)

        return text
