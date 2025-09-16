"""
Shared utilities for text chunkers.

This module provides common functionality used across different text chunking
strategies to reduce code duplication and ensure consistency.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from chunking_strategy.core.base import ChunkMetadata, ModalityType

logger = logging.getLogger(__name__)


class TextChunkerUtils:
    """Shared utilities for text chunking strategies."""

    @staticmethod
    def validate_common_parameters(
        chunk_size: int,
        overlap: int,
        min_chunk_size: int,
        max_chunk_chars: int,
        size_parameter_name: str = "chunk_size",
        overlap_parameter_name: str = "overlap"
    ) -> None:
        """
        Validate common chunking parameters.

        Args:
            chunk_size: Primary chunk size parameter
            overlap: Overlap parameter
            min_chunk_size: Minimum chunk size
            max_chunk_chars: Maximum characters per chunk
            size_parameter_name: Name of the size parameter for error messages
            overlap_parameter_name: Name of the overlap parameter for error messages

        Raises:
            ValueError: If parameters are invalid
        """
        if chunk_size <= 0:
            raise ValueError(f"{size_parameter_name} must be positive")
        if overlap < 0:
            raise ValueError(f"{overlap_parameter_name} must be non-negative")
        if overlap >= chunk_size:
            raise ValueError(f"{overlap_parameter_name} must be less than {size_parameter_name}")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if min_chunk_size >= chunk_size:
            raise ValueError("min_chunk_size must be less than chunk_size")
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")

    @staticmethod
    def handle_input_content(content: Union[str, bytes, Path]) -> tuple[str, str, Optional[str]]:
        """
        Handle different input content types.

        Args:
            content: Input content (string, bytes, or file path)

        Returns:
            Tuple of (text_content, source_type, source_path)
        """
        if isinstance(content, Path):
            text_content = content.read_text(encoding='utf-8', errors='ignore')
            return text_content, "file", str(content)
        elif isinstance(content, bytes):
            text_content = content.decode('utf-8', errors='ignore')
            return text_content, "content", None
        else:
            return str(content), "content", None

    @staticmethod
    def create_standard_chunk_metadata(
        source: str,
        source_type: str,
        position: str,
        content_length: int,
        chunk_index: int,
        chunking_strategy: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        offset: Optional[int] = None
    ) -> ChunkMetadata:
        """
        Create standardized chunk metadata.

        Args:
            source: Source identifier
            source_type: Type of source
            position: Position description
            content_length: Length of chunk content
            chunk_index: Index of the chunk
            chunking_strategy: Name of chunking strategy
            extra_metadata: Additional metadata fields
            offset: Character offset in original text

        Returns:
            ChunkMetadata object
        """
        base_extra = {
            "chunk_index": chunk_index,
            "chunking_strategy": chunking_strategy
        }
        
        if extra_metadata:
            base_extra.update(extra_metadata)

        return ChunkMetadata(
            source=source,
            source_type=source_type,
            position=position,
            offset=offset,
            length=content_length,
            extra=base_extra
        )

    @staticmethod
    def calculate_adaptation_changes(
        old_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate changes between old and new configurations.

        Args:
            old_config: Original configuration
            new_config: New configuration

        Returns:
            Dictionary of changes with old/new values
        """
        changes = {}
        for key, new_value in new_config.items():
            old_value = old_config.get(key)
            if old_value != new_value:
                changes[key] = {"old": old_value, "new": new_value}
        return changes

    @staticmethod
    def create_adaptation_record(
        feedback_score: float,
        feedback_type: str,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create standardized adaptation record.

        Args:
            feedback_score: Feedback score received
            feedback_type: Type of feedback
            old_config: Configuration before adaptation
            new_config: Configuration after adaptation
            **kwargs: Additional context

        Returns:
            Adaptation record dictionary
        """
        changes = TextChunkerUtils.calculate_adaptation_changes(old_config, new_config)
        
        return {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "changes": changes,
            "old_config": old_config,
            "new_config": new_config,
            "context": kwargs
        }

    @staticmethod
    def truncate_to_word_boundary(text: str, max_length: int) -> str:
        """
        Truncate text to word boundary within max_length.

        Args:
            text: Text to truncate
            max_length: Maximum character length

        Returns:
            Truncated text ending at word boundary
        """
        if len(text) <= max_length:
            return text

        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        # Only truncate at word boundary if we don't lose too much content
        if last_space > max_length * 0.8:
            return truncated[:last_space]
        
        return truncated

    @staticmethod
    def estimate_processing_time(
        text_length: int,
        chunk_size: int,
        base_time_per_char: float = 0.000001
    ) -> float:
        """
        Estimate processing time for chunking operation.

        Args:
            text_length: Length of input text
            chunk_size: Size of chunks
            base_time_per_char: Base processing time per character

        Returns:
            Estimated processing time in seconds
        """
        # Basic estimation: linear time with chunk overhead
        num_chunks = max(1, text_length // chunk_size)
        return text_length * base_time_per_char + num_chunks * 0.0001


class TokenizerPatterns:
    """Pre-compiled regex patterns for common tokenization tasks."""

    def __init__(self):
        # Word tokenization patterns
        self.simple_word = re.compile(r'\S+')
        self.word_with_punctuation = re.compile(r'\S+')
        self.word_boundary = re.compile(r'\b\w+\b')
        
        # Sentence patterns
        self.sentence_basic = re.compile(r'([.!?。！？]+(?:\s*["\']?\s*|\s+))', re.MULTILINE)
        self.sentence_advanced = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[。！？])\s*', 
            re.MULTILINE | re.UNICODE
        )
        
        # Whitespace patterns
        self.whitespace = re.compile(r'\s+')
        self.line_breaks = re.compile(r'\n+')
        
        # Special characters
        self.punctuation = re.compile(r'[^\w\s]')

    def tokenize_words(self, text: str, preserve_punctuation: bool = True) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text
            preserve_punctuation: Whether to keep punctuation with words

        Returns:
            List of word tokens
        """
        if preserve_punctuation:
            return self.simple_word.findall(text)
        else:
            return self.word_boundary.findall(text)

    def split_sentences(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text
            separators: Custom sentence separators

        Returns:
            List of sentences
        """
        if separators:
            # Create custom pattern
            escaped_seps = [re.escape(sep) for sep in separators]
            pattern = re.compile(f'([{"".join(escaped_seps)}]+(?:\\s*["\']?\\s*|\\s+))', re.MULTILINE)
        else:
            pattern = self.sentence_basic

        parts = pattern.split(text)
        sentences = []
        current = ""
        
        for part in parts:
            if pattern.match(part):
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        
        if current.strip():
            sentences.append(current.strip())
            
        return [s for s in sentences if s.strip()]


# Global shared instances
text_utils = TextChunkerUtils()
tokenizer_patterns = TokenizerPatterns()
