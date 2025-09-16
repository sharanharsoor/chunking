"""
Sentence-based chunking strategy.

This module implements sentence-aware chunking that respects sentence boundaries
and can group multiple sentences into coherent chunks.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType,
    StreamableChunker
)
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage

logger = logging.getLogger(__name__)


@register_chunker(
    name="sentence_based",
    category="text",
    description="Chunks text by grouping sentences with respect to sentence boundaries",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "html", "json"],
    complexity=ComplexityLevel.LOW,
    dependencies=[],
    optional_dependencies=["nltk", "spacy"],
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    quality=0.7,  # Good quality due to sentence awareness
    parameters_schema={
        "max_sentences": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "default": 5,
            "description": "Maximum number of sentences per chunk"
        },
        "min_sentences": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "default": 1,
            "description": "Minimum number of sentences per chunk"
        },
        "max_chunk_size": {
            "type": "integer",
            "minimum": 100,
            "maximum": 100000,
            "default": 2000,
            "description": "Maximum chunk size in characters"
        },
        "overlap_sentences": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "default": 0,
            "description": "Number of sentences to overlap between chunks"
        },
        "sentence_splitter": {
            "type": "string",
            "enum": ["simple", "nltk", "spacy"],
            "default": "simple",
            "description": "Method to use for sentence splitting"
        }
    },
    default_parameters={
        "max_sentences": 5,
        "min_sentences": 1,
        "max_chunk_size": 2000,
        "overlap_sentences": 0,
        "sentence_splitter": "simple"
    },
    use_cases=["RAG", "document processing", "content analysis", "text summarization"],
    best_for=["natural language text", "articles", "books", "coherent chunks"],
    limitations=["requires sentence boundaries", "language dependent", "not suitable for code"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=False
)
class SentenceBasedChunker(StreamableChunker):
    """
    Sentence-based chunking strategy.

    Chunks text by grouping sentences together while respecting sentence boundaries.
    This approach maintains semantic coherence better than fixed-size chunking
    and is particularly suitable for natural language processing tasks.

    Features:
    - Respects sentence boundaries
    - Configurable sentence grouping
    - Multiple sentence splitting methods
    - Size constraints to prevent overly large chunks
    - Optional overlap between chunks
    - Streaming support for large documents

    Examples:
        Basic usage:
        ```python
        chunker = SentenceBasedChunker(max_sentences=3)
        result = chunker.chunk("First sentence. Second sentence. Third sentence. Fourth sentence.")
        ```

        With overlap:
        ```python
        chunker = SentenceBasedChunker(max_sentences=5, overlap_sentences=1)
        result = chunker.chunk("Long document with many sentences...")
        ```

        Advanced sentence splitting:
        ```python
        chunker = SentenceBasedChunker(sentence_splitter="nltk", max_chunk_size=1500)
        result = chunker.chunk("Complex text with abbreviations etc. and difficult boundaries.")
        ```
    """

    def __init__(
        self,
        max_sentences: int = 5,
        min_sentences: int = 1,
        max_chunk_size: int = 2000,
        overlap_sentences: int = 0,
        sentence_splitter: str = "simple",
        max_text_buffer_size: int = 2 * 1024 * 1024,  # 2MB buffer limit for streaming protection
        **kwargs
    ):
        """
        Initialize the sentence-based chunker.

        Args:
            max_sentences: Maximum sentences per chunk
            min_sentences: Minimum sentences per chunk
            max_chunk_size: Maximum chunk size in characters
            overlap_sentences: Number of sentences to overlap
            sentence_splitter: Method for sentence splitting
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="sentence_based",
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        # Validate parameters
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        if min_sentences <= 0:
            raise ValueError("min_sentences must be positive")
        if min_sentences > max_sentences:
            raise ValueError("min_sentences cannot be greater than max_sentences")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences cannot be negative")
        if overlap_sentences >= max_sentences:
            raise ValueError("overlap_sentences must be less than max_sentences")
        if sentence_splitter not in ["simple", "nltk", "spacy"]:
            raise ValueError("sentence_splitter must be 'simple', 'nltk', or 'spacy'")

        self.max_sentences = max_sentences
        self.min_sentences = min_sentences
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
        self.sentence_splitter = sentence_splitter
        self.max_text_buffer_size = max_text_buffer_size

        # Initialize sentence splitter
        self._init_sentence_splitter()

        self.logger.info(
            f"Initialized SentenceBasedChunker: max_sentences={max_sentences}, "
            f"overlap={overlap_sentences}, splitter={sentence_splitter}"
        )

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk content into sentence-based pieces.

        Args:
            content: Content to chunk
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with generated chunks
        """
        start_time = time.time()

        # Handle different input types
        if isinstance(content, Path):
            file_path = content
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            actual_source = str(file_path)
        elif isinstance(content, str) and len(content) > 0 and len(content) < 260 and '\n' not in content:
            # Check if string might be a file path (short, no newlines)
            try:
                file_path = Path(content)
                if file_path.is_file():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    actual_source = str(file_path)
                else:
                    # Treat as text content
                    text_content = content
                    actual_source = source_info.get("source", "string") if source_info else "string"
            except (OSError, IOError):
                # If any filesystem error, treat as text content
                text_content = content
                actual_source = source_info.get("source", "string") if source_info else "string"
        elif isinstance(content, bytes):
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                raise ValueError("Cannot decode bytes as UTF-8 for sentence-based chunking")
            actual_source = source_info.get("source", "bytes_input") if source_info else "bytes_input"
        else:
            text_content = str(content)
            actual_source = source_info.get("source", "text_input") if source_info else "text_input"

        # Validate input
        self.validate_input(text_content, ModalityType.TEXT)

        if not text_content.strip():
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used=self.name,
                source_info=source_info
            )

        # Split into sentences
        sentences = self._split_sentences(text_content)

        if not sentences:
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used=self.name,
                source_info=source_info
            )

        # Group sentences into chunks
        chunks = self._group_sentences_into_chunks(sentences, actual_source)

        processing_time = time.time() - start_time

        # Create result
        result = ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used=self.name,
            source_info=source_info
        )

        self.logger.info(
            f"Sentence-based chunking completed: {len(sentences)} sentences -> "
            f"{len(chunks)} chunks in {processing_time:.3f}s"
        )

        return result

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream-process content into sentence-based chunks.

        Args:
            content_stream: Iterator yielding content pieces
            source_info: Information about the content source
            **kwargs: Additional parameters

        Yields:
            Individual chunks as they are generated
        """
        sentence_buffer = []
        overlap_buffer = []
        chunk_counter = 0
        text_buffer = ""

        for piece in content_stream:
            if isinstance(piece, bytes):
                try:
                    piece = piece.decode('utf-8')
                except UnicodeDecodeError:
                    self.logger.warning("Skipping non-UTF-8 content in stream")
                    continue

            text_buffer += piece

            # Look for sentence boundaries in the buffer
            new_sentences = self._extract_complete_sentences(text_buffer)
            if new_sentences:
                complete_sentences, text_buffer = new_sentences
                sentence_buffer.extend(complete_sentences)

                # Check if we have enough sentences for a chunk
                while len(sentence_buffer) >= self.max_sentences:
                    chunk_sentences = overlap_buffer + sentence_buffer[:self.max_sentences]

                    # Create chunk
                    chunk_content = ' '.join(chunk_sentences)
                    chunk = self._create_chunk_from_sentences(
                        chunk_sentences,
                        chunk_counter,
                        source_info
                    )

                    yield chunk
                    chunk_counter += 1

                    # Setup overlap for next chunk
                    if self.overlap_sentences > 0:
                        overlap_buffer = sentence_buffer[self.max_sentences - self.overlap_sentences:self.max_sentences]
                    else:
                        overlap_buffer = []

                    # Remove processed sentences
                    sentence_buffer = sentence_buffer[self.max_sentences:]

        # Process any remaining sentences
        if sentence_buffer:
            final_sentences = overlap_buffer + sentence_buffer
            if final_sentences:
                chunk = self._create_chunk_from_sentences(
                    final_sentences,
                    chunk_counter,
                    source_info
                )
                yield chunk

    def _init_sentence_splitter(self) -> None:
        """Initialize the sentence splitter based on configuration."""
        if self.sentence_splitter == "nltk":
            try:
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    self.logger.warning("NLTK punkt tokenizer not found, downloading...")
                    nltk.download('punkt', quiet=True)
                self._nltk_splitter = nltk.sent_tokenize
                self.logger.debug("Initialized NLTK sentence splitter")
            except ImportError:
                self.logger.warning("NLTK not available, falling back to simple splitter")
                self.sentence_splitter = "simple"

        elif self.sentence_splitter == "spacy":
            try:
                import spacy
                # Try to load a small English model
                try:
                    self._spacy_nlp = spacy.load("en_core_web_sm")
                except IOError:
                    self.logger.warning("spaCy model not found, falling back to simple splitter")
                    self.sentence_splitter = "simple"
                self.logger.debug("Initialized spaCy sentence splitter")
            except ImportError:
                self.logger.warning("spaCy not available, falling back to simple splitter")
                self.sentence_splitter = "simple"

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences with robust fallback for dictionary data."""
        # First try the configured sentence splitter
        sentences = []

        if self.sentence_splitter == "nltk" and hasattr(self, '_nltk_splitter'):
            try:
                sentences = self._nltk_splitter(text)
                sentences = [s.strip() for s in sentences if s.strip()]
            except Exception as e:
                self.logger.warning(f"NLTK sentence splitting failed: {e}, falling back to simple")

        elif self.sentence_splitter == "spacy" and hasattr(self, '_spacy_nlp'):
            try:
                doc = self._spacy_nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
                sentences = [s for s in sentences if s]
            except Exception as e:
                self.logger.warning(f"spaCy sentence splitting failed: {e}, falling back to simple")

        # If no sentences found yet, try simple sentence splitting
        if not sentences:
            sentences = self._simple_sentence_split(text)

        # Robust fallback for dictionary data and other edge cases
        if not sentences or (len(sentences) == 1 and len(text) > 1000):
            # Check if this looks like dictionary data (single words per line)
            lines = text.strip().split('\n')
            if len(lines) > 5:
                single_word_lines = sum(1 for line in lines if len(line.strip().split()) <= 2)
                if single_word_lines / len(lines) > 0.7:  # 70% are single/double words
                    self.logger.debug("Dictionary data detected, treating lines as sentences")
                    return [line.strip() for line in lines if line.strip()]

            # Fallback for other problematic data
            if '\n\n' in text:
                # Split by paragraphs
                return [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            elif len(text) > 2000:
                # Split large text into reasonable chunks
                chunk_size = 500  # characters
                chunks = []
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size].strip()
                    if chunk:
                        chunks.append(chunk)
                return chunks

        return sentences

    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple regex-based sentence splitting."""
        # Basic sentence boundary detection
        # This is a simplified approach and may not handle all edge cases
        sentence_endings = r'[.!?]+(?:\s+|$)'

        # Split on sentence endings but keep the punctuation
        parts = re.split(f'({sentence_endings})', text)

        sentences = []
        current_sentence = ""

        for i, part in enumerate(parts):
            if re.match(sentence_endings, part):
                # This is punctuation, add to current sentence
                current_sentence += part.rstrip()
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                # This is text content
                current_sentence += part

        # Add any remaining content
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return [s for s in sentences if s]

    def _group_sentences_into_chunks(self, sentences: List[str], source: str) -> List[Chunk]:
        """Group sentences into chunks according to configuration."""
        chunks = []
        chunk_counter = 0
        i = 0

        while i < len(sentences):
            chunk_sentences = []
            chunk_size = 0
            sentences_added = 0

            # Add sentences until we reach limits
            while (i < len(sentences) and
                   sentences_added < self.max_sentences and
                   chunk_size < self.max_chunk_size):

                sentence = sentences[i]
                sentence_size = len(sentence)

                # Check if adding this sentence would exceed size limit
                if chunk_size + sentence_size > self.max_chunk_size and chunk_sentences:
                    break

                chunk_sentences.append(sentence)
                chunk_size += sentence_size
                sentences_added += 1
                i += 1

            # Ensure minimum sentences requirement
            if len(chunk_sentences) < self.min_sentences and i < len(sentences):
                while len(chunk_sentences) < self.min_sentences and i < len(sentences):
                    chunk_sentences.append(sentences[i])
                    i += 1

            # Create chunk
            if chunk_sentences:
                chunk = self._create_chunk_from_sentences(
                    chunk_sentences,
                    chunk_counter,
                    {"source": source}
                )
                chunks.append(chunk)
                chunk_counter += 1

                # Handle overlap
                if self.overlap_sentences > 0 and i < len(sentences):
                    overlap_start = max(0, len(chunk_sentences) - self.overlap_sentences)
                    i -= len(chunk_sentences) - overlap_start

        return chunks

    def _create_chunk_from_sentences(
        self,
        sentences: List[str],
        chunk_id: int,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """Create a chunk from a list of sentences."""
        content = ' '.join(sentences)

        # Create metadata
        metadata = ChunkMetadata(
            source=extra_metadata.get("source", "unknown") if extra_metadata else "unknown",
            chunker_used=self.name,
            length=len(content)
        )

        # Add sentence-specific metadata
        metadata.extra.update({
            "sentence_count": len(sentences),
            "avg_sentence_length": len(content) / len(sentences) if sentences else 0,
            "first_sentence": sentences[0][:50] + "..." if sentences and len(sentences[0]) > 50 else sentences[0] if sentences else ""
        })

        # Add extra metadata
        if extra_metadata:
            for key, value in extra_metadata.items():
                if key != "source":
                    metadata.extra[key] = value

        return Chunk(
            id=f"sentence_based_{chunk_id}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=metadata
        )

    def _extract_complete_sentences(self, text_buffer: str) -> Optional[tuple[List[str], str]]:
        """Extract complete sentences from a text buffer with robust fallback for dictionary data."""
        # Safety check: prevent buffer from growing too large
        if len(text_buffer) > self.max_text_buffer_size:
            self.logger.warning(f"Buffer size exceeded {self.max_text_buffer_size}, forcing processing")
            # Force processing of current buffer
            sentences = self._split_sentences(text_buffer)
            return sentences, ""

        # First try traditional sentence boundary detection
        last_boundary = -1
        for match in re.finditer(r'[.!?]+\s+', text_buffer):
            last_boundary = match.end()

        if last_boundary > 0:
            complete_text = text_buffer[:last_boundary]
            remaining_text = text_buffer[last_boundary:]
            sentences = self._split_sentences(complete_text)
            return sentences, remaining_text

        # If no sentence boundaries found, check for dictionary-style data
        lines = text_buffer.strip().split('\n')
        if len(lines) >= self.max_sentences:
            # Check if this looks like dictionary data (single words per line)
            single_word_lines = sum(1 for line in lines if len(line.strip().split()) <= 2)
            if single_word_lines / len(lines) > 0.7 and len(lines) >= 3:
                # Process lines as sentences
                complete_lines = lines[:self.max_sentences]
                remaining_lines = lines[self.max_sentences:]

                sentences = [line.strip() for line in complete_lines if line.strip()]
                remaining_text = '\n'.join(remaining_lines)

                return sentences, remaining_text

        # If buffer has multiple paragraphs, process first paragraph
        if '\n\n' in text_buffer and len(text_buffer) > 1000:
            paragraphs = text_buffer.split('\n\n')
            if len(paragraphs) > 1:
                complete_text = paragraphs[0]
                remaining_text = '\n\n'.join(paragraphs[1:])
                sentences = self._split_sentences(complete_text)
                if sentences:
                    return sentences, remaining_text

        return None

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """
        Adapt chunker parameters based on feedback.

        Args:
            feedback_score: Feedback score (0.0 to 1.0)
            feedback_type: Type of feedback
            **kwargs: Additional feedback information
        """
        if feedback_type == "quality" and feedback_score < 0.5:
            # Decrease max_sentences for better coherence
            old_max = self.max_sentences
            self.max_sentences = max(self.max_sentences - 1, self.min_sentences)

            # Ensure overlap_sentences is always less than max_sentences
            if self.overlap_sentences >= self.max_sentences:
                old_overlap = self.overlap_sentences
                self.overlap_sentences = max(0, self.max_sentences - 1)
                self.logger.info(f"Adapted overlap_sentences: {old_overlap} -> {self.overlap_sentences} (to maintain valid configuration)")

            self.logger.info(f"Adapted max_sentences: {old_max} -> {self.max_sentences} (quality feedback)")

        elif feedback_type == "coverage" and feedback_score < 0.5:
            # Increase max_sentences for better coverage
            old_max = self.max_sentences
            self.max_sentences = min(self.max_sentences + 1, 20)
            self.logger.info(f"Adapted max_sentences: {old_max} -> {self.max_sentences} (coverage feedback)")

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Decrease chunk size for better performance
            old_size = self.max_chunk_size
            self.max_chunk_size = max(int(self.max_chunk_size * 0.8), 500)
            self.logger.info(f"Adapted max_chunk_size: {old_size} -> {self.max_chunk_size} (performance feedback)")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        # For now, return empty list - could be enhanced to track changes
        return []
