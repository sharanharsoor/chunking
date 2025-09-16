"""
Character encoding detection for text files.

This module provides encoding detection capabilities to ensure proper
text decoding before chunking.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)


class EncodingDetector:
    """
    Detects character encoding of text files.

    Uses various methods to detect file encoding including BOM detection,
    statistical analysis, and external libraries when available.
    """

    def __init__(self):
        """Initialize encoding detector."""
        self.logger = logging.getLogger(f"{__name__}.EncodingDetector")

        # Byte Order Mark patterns
        self.bom_patterns = {
            b'\xEF\xBB\xBF': 'utf-8-sig',
            b'\xFF\xFE': 'utf-16-le',
            b'\xFE\xFF': 'utf-16-be',
            b'\xFF\xFE\x00\x00': 'utf-32-le',
            b'\x00\x00\xFE\xFF': 'utf-32-be',
        }

        # Common encodings to try
        self.common_encodings = [
            'utf-8', 'utf-16', 'utf-32',
            'ascii', 'latin1', 'cp1252',
            'iso-8859-1', 'iso-8859-15',
        ]

    def detect(self, file_path: Union[str, Path], sample_size: int = 8192) -> Dict[str, Any]:
        """
        Detect encoding of a file.

        Args:
            file_path: Path to the file
            sample_size: Bytes to read for detection

        Returns:
            Dictionary with encoding information
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                sample = f.read(sample_size)

            if not sample:
                return {
                    'encoding': 'utf-8',
                    'confidence': 1.0,
                    'method': 'default_empty'
                }

            result = self.detect_bytes(sample)
            result['file_path'] = str(file_path)
            return result

        except Exception as e:
            self.logger.error(f"Error detecting encoding for {file_path}: {e}")
            return {
                'encoding': 'utf-8',
                'confidence': 0.0,
                'method': 'error_fallback',
                'error': str(e)
            }

    def detect_bytes(self, data: bytes) -> Dict[str, Any]:
        """
        Detect encoding from byte data.

        Args:
            data: Byte data to analyze

        Returns:
            Dictionary with encoding detection results
        """
        if not data:
            return {
                'encoding': 'utf-8',
                'confidence': 1.0,
                'method': 'default_empty'
            }

        # Check for BOM first
        bom_result = self._detect_bom(data)
        if bom_result:
            return bom_result

        # Try external library detection
        external_result = self._try_external_detection(data)
        if external_result and external_result['confidence'] > 0.8:
            return external_result

        # Try common encodings
        encoding_result = self._try_common_encodings(data)
        if encoding_result:
            return encoding_result

        # Fallback to UTF-8
        return {
            'encoding': 'utf-8',
            'confidence': 0.5,
            'method': 'fallback'
        }

    def validate_encoding(self, file_path: Union[str, Path], encoding: str) -> bool:
        """
        Validate that a file can be decoded with the specified encoding.

        Args:
            file_path: Path to the file
            encoding: Encoding to validate

        Returns:
            True if encoding is valid for the file
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)  # Try to read first KB
            return True
        except (UnicodeDecodeError, LookupError):
            return False

    def get_suggested_encodings(self, file_path: Union[str, Path]) -> List[str]:
        """
        Get list of suggested encodings for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of encoding names in order of likelihood
        """
        detection_result = self.detect(file_path)
        detected_encoding = detection_result['encoding']

        # Start with detected encoding
        suggestions = [detected_encoding]

        # Add common alternatives
        for encoding in self.common_encodings:
            if encoding not in suggestions and self.validate_encoding(file_path, encoding):
                suggestions.append(encoding)

        return suggestions

    def _detect_bom(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Detect encoding from Byte Order Mark."""
        for bom, encoding in self.bom_patterns.items():
            if data.startswith(bom):
                return {
                    'encoding': encoding,
                    'confidence': 1.0,
                    'method': 'bom',
                    'bom_detected': True
                }
        return None

    def _try_external_detection(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Try external encoding detection library."""
        try:
            import chardet
            result = chardet.detect(data)
            if result and result['encoding']:
                return {
                    'encoding': result['encoding'].lower(),
                    'confidence': result['confidence'],
                    'method': 'chardet'
                }
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"chardet detection failed: {e}")

        try:
            import charset_normalizer
            results = charset_normalizer.from_bytes(data)
            if results:
                best = results.best()
                if best:
                    return {
                        'encoding': str(best.encoding).lower(),
                        'confidence': best.coherence,
                        'method': 'charset_normalizer'
                    }
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"charset_normalizer detection failed: {e}")

        return None

    def _try_common_encodings(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Try decoding with common encodings."""
        for encoding in self.common_encodings:
            try:
                data.decode(encoding)
                # If successful, estimate confidence based on encoding type
                confidence = 0.9 if encoding == 'utf-8' else 0.7
                return {
                    'encoding': encoding,
                    'confidence': confidence,
                    'method': 'trial_decode'
                }
            except UnicodeDecodeError:
                continue

        return None
