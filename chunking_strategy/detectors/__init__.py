"""
Detection modules for file types, languages, encodings, and content analysis.

This package provides various detectors that help the orchestrator make
intelligent decisions about which chunking strategies to use based on
the characteristics of the input content.
"""

from chunking_strategy.detectors.file_type_detector import FileTypeDetector
from chunking_strategy.detectors.language_detector import LanguageDetector
from chunking_strategy.detectors.encoding_detector import EncodingDetector
from chunking_strategy.detectors.content_analyzer import ContentAnalyzer

__all__ = [
    "FileTypeDetector",
    "LanguageDetector",
    "EncodingDetector",
    "ContentAnalyzer",
]
