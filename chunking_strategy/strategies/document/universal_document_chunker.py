"""
Universal document chunker using Apache Tika.

Provides unified chunking for all document types supported by Apache Tika (1,400+ formats),
including DOC, DOCX, PDF, Excel, PowerPoint, and various code files.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.core.streaming import StreamableChunker
from chunking_strategy.core.tika_integration import get_tika_processor


@register_chunker(
    name="universal_document",
    category="document", 
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.MEDIUM,
    memory=MemoryUsage.MEDIUM,
    supported_formats=["*"],  # Supports all formats via Tika
    dependencies=["tika", "python-magic"],
    description="Universal document chunker using Apache Tika for 1,400+ file formats",
    use_cases=["document_processing", "multi_format_support", "enterprise_document_analysis"]
)
class UniversalDocumentChunker(StreamableChunker):
    """
    Universal document chunker that can process any file format supported by Apache Tika.
    
    Features:
    - Supports 1,400+ file formats including PDF, DOC, DOCX, Excel, PowerPoint
    - Automatic file type detection and content extraction
    - Unified metadata extraction across all formats
    - Code file support (.py, .java, .cpp, .c, etc.)
    - Fallback to other chunkers when Tika is not available
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        preserve_structure: bool = True,
        extract_metadata: bool = True,
        fallback_strategy: str = "fixed_size",
        tika_timeout: int = 30,
        **kwargs
    ):
        """
        Initialize universal document chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            preserve_structure: Whether to preserve document structure
            extract_metadata: Whether to extract detailed metadata
            fallback_strategy: Strategy to use when Tika is unavailable
            tika_timeout: Timeout for Tika operations in seconds
            **kwargs: Additional parameters
        """
        super().__init__(
            name="universal_document",
            category="document",
            supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
            **kwargs
        )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_structure = preserve_structure
        self.extract_metadata = extract_metadata
        self.fallback_strategy = fallback_strategy
        self.tika_timeout = tika_timeout
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Tika processor
        self.tika_processor = get_tika_processor(timeout=tika_timeout)

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk document content using Apache Tika extraction.

        Args:
            content: File path or content to chunk
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with extracted and chunked content
        """
        start_time = time.time()
        
        # Handle input types
        if isinstance(content, (str, Path)):
            file_path = Path(content)
            if not file_path.exists():
                # Treat as direct content if not a file
                return self._chunk_text_content(str(content), source_info, start_time)
        elif isinstance(content, bytes):
            # Save bytes to temporary file for Tika processing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                tmp.write(content)
                file_path = Path(tmp.name)
        else:
            raise ValueError(f"Invalid content type for universal chunker: {type(content)}")

        try:
            # Check if Tika is available
            if not self.tika_processor.is_available():
                return self._fallback_chunking(file_path, source_info, start_time)
            
            # Detect file type
            file_info = self.tika_processor.detect_file_type(file_path)
            self.logger.info(f"Detected file type: {file_info.get('mime_type', 'unknown')} for {file_path.name}")
            
            # Extract content and metadata using Tika
            extracted_content, metadata = self.tika_processor.extract_content_and_metadata(file_path)
            
            if not extracted_content or len(extracted_content.strip()) < 10:
                self.logger.warning(f"No content extracted from {file_path.name}, using fallback")
                return self._fallback_chunking(file_path, source_info, start_time)
            
            # Chunk the extracted content
            chunks = self._create_chunks_from_content(
                extracted_content, 
                file_path, 
                metadata, 
                file_info
            )
            
            # Create chunking result
            result = ChunkingResult(
                chunks=chunks,
                processing_time=time.time() - start_time,
                strategy_used="universal_document",
                source_info={
                    "source_file": str(file_path),
                    "file_type": file_info.get('mime_type', 'unknown'),
                    "extraction_method": "apache_tika",
                    "content_length": len(extracted_content),
                    "tika_metadata": metadata if self.extract_metadata else {}
                }
            )
            
            self.logger.info(f"Universal chunking completed: {len(chunks)} chunks from {file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Universal chunking failed for {file_path}: {e}")
            # Try fallback strategy
            return self._fallback_chunking(file_path, source_info, start_time)

    def _chunk_text_content(self, text_content: str, source_info: Optional[Dict], start_time: float) -> ChunkingResult:
        """Chunk direct text content."""
        chunks = self._create_text_chunks(text_content, "direct_input")
        
        return ChunkingResult(
            chunks=chunks,
            processing_time=time.time() - start_time,
            strategy_used="universal_document",
            source_info=source_info or {"content_type": "direct_text"}
        )

    def _create_chunks_from_content(
        self, 
        content: str, 
        file_path: Path, 
        metadata: Dict[str, Any],
        file_info: Dict[str, str]
    ) -> List[Chunk]:
        """Create chunks from extracted content."""
        chunks = []
        
        if self.preserve_structure:
            # Try to preserve document structure
            chunks = self._create_structured_chunks(content, file_path, metadata, file_info)
        else:
            # Create simple fixed-size chunks
            chunks = self._create_text_chunks(content, str(file_path), metadata, file_info)
        
        return chunks

    def _create_structured_chunks(
        self, 
        content: str, 
        file_path: Path, 
        metadata: Dict[str, Any],
        file_info: Dict[str, str]
    ) -> List[Chunk]:
        """Create chunks while preserving document structure."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_chunk(
                    current_chunk.strip(), 
                    chunk_index, 
                    file_path, 
                    metadata, 
                    file_info
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    current_chunk = current_chunk[-self.chunk_overlap:] + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                chunk_index += 1
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(), 
                chunk_index, 
                file_path, 
                metadata, 
                file_info
            )
            chunks.append(chunk)
        
        return chunks

    def _create_text_chunks(
        self, 
        content: str, 
        source_name: str, 
        metadata: Optional[Dict[str, Any]] = None,
        file_info: Optional[Dict[str, str]] = None
    ) -> List[Chunk]:
        """Create simple text chunks."""
        chunks = []
        content_length = len(content)
        
        start = 0
        chunk_index = 0
        
        while start < content_length:
            # Calculate end position
            end = min(start + self.chunk_size, content_length)
            
            # Try to end at a word boundary
            if end < content_length:
                last_space = content.rfind(' ', start, end)
                if last_space > start + self.chunk_size // 2:  # Don't go too far back
                    end = last_space
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunk = self._create_chunk(
                    chunk_content, 
                    chunk_index, 
                    source_name, 
                    metadata or {}, 
                    file_info or {}
                )
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
            chunk_index += 1
        
        return chunks

    def _create_chunk(
        self, 
        content: str, 
        index: int, 
        source: Union[str, Path], 
        metadata: Dict[str, Any],
        file_info: Dict[str, str]
    ) -> Chunk:
        """Create a single chunk object."""
        chunk_metadata = ChunkMetadata(
            source=str(source),
            position={"chunk_index": index, "start_char": index * self.chunk_size},
            extra={
                "chunk_type": "text",
                "extraction_method": "universal_document",
                "file_type": file_info.get('mime_type', 'unknown'),
                "tika_metadata": metadata if self.extract_metadata else {},
                **file_info
            }
        )
        
        # Add specific metadata fields
        if 'language' in metadata:
            chunk_metadata.language = metadata['language']
        if 'author' in metadata:
            chunk_metadata.extra['author'] = metadata['author']
        if 'title' in metadata:
            chunk_metadata.extra['title'] = metadata['title']
        if 'created_date' in metadata:
            chunk_metadata.extra['created_date'] = metadata['created_date']
        
        return Chunk(
            id=f"universal_{Path(source).stem}_{index}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _fallback_chunking(self, file_path: Path, source_info: Optional[Dict], start_time: float) -> ChunkingResult:
        """Fallback to alternative chunking strategy."""
        self.logger.info(f"Using fallback strategy '{self.fallback_strategy}' for {file_path}")
        
        try:
            from chunking_strategy.core.registry import create_chunker
            
            # Create fallback chunker
            fallback_chunker = create_chunker(
                self.fallback_strategy,
                chunk_size=self.chunk_size
            )
            
            # Use fallback chunker
            result = fallback_chunker.chunk(file_path, source_info)
            result.strategy_used = f"universal_document_fallback_{self.fallback_strategy}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fallback chunking also failed: {e}")
            # Return empty result as last resort
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used="universal_document_failed",
                source_info=source_info or {}
            )

    def chunk_stream(self, content_stream, **kwargs):
        """
        Chunk content from a stream.
        Note: Tika requires full content, so we collect the stream first.
        """
        # Collect stream content
        content_bytes = b""
        for chunk in content_stream:
            if isinstance(chunk, str):
                content_bytes += chunk.encode('utf-8')
            elif isinstance(chunk, bytes):
                content_bytes += chunk
        
        # Process as bytes
        result = self.chunk(content_bytes, **kwargs)
        
        # Yield chunks one by one
        for chunk in result.chunks:
            yield chunk

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return self.tika_processor.get_supported_formats()

    def estimate_chunks(self, content: Union[str, Path]) -> int:
        """Estimate number of chunks that will be generated."""
        if isinstance(content, (str, Path)):
            file_path = Path(content)
            if file_path.exists():
                file_size = file_path.stat().st_size
                # Rough estimate: assume 1 byte ≈ 1 character
                estimated_chars = file_size
                return max(1, estimated_chars // self.chunk_size)
        
        return 1  # Default estimate

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """Adapt chunker parameters based on feedback."""
        if feedback_type == "quality" and feedback_score < 0.5:
            # Decrease chunk size for better granularity
            old_size = self.chunk_size
            self.chunk_size = max(200, int(self.chunk_size * 0.8))
            self.logger.info(f"Adapted chunk_size: {old_size} -> {self.chunk_size} (quality feedback)")
            
        elif feedback_type == "performance" and feedback_score < 0.5:
            # Increase chunk size for better performance
            old_size = self.chunk_size
            self.chunk_size = min(5000, int(self.chunk_size * 1.2))
            self.logger.info(f"Adapted chunk_size: {old_size} -> {self.chunk_size} (performance feedback)")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return []
