"""
PDF-based chunking strategy with support for images and tables.

This module provides specialized chunking for PDF documents, extracting and handling:
- Text content with proper formatting
- Images with metadata
- Tables with structure preservation
- Mixed content layouts
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.layout import LAParams
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.core.streaming import StreamableChunker


class PDFChunker(StreamableChunker):
    """
    Specialized chunker for PDF documents supporting:
    - Text extraction with formatting preservation
    - Image extraction with metadata
    - Table detection and extraction
    - Mixed content handling
    - Multiple PDF processing backends
    """

    def __init__(
        self,
        pages_per_chunk: int = 1,
        extract_images: bool = True,
        extract_tables: bool = True,
        preserve_formatting: bool = True,
        backend: str = "auto",  # auto, pymupdf, pypdf2, pdfminer
        image_quality: int = 95,
        table_detection_threshold: float = 0.7,
        min_text_length: int = 50,
        **kwargs
    ):
        """
        Initialize PDF chunker.

        Args:
            pages_per_chunk: Number of pages to include per chunk
            extract_images: Whether to extract and include images
            extract_tables: Whether to detect and extract tables
            preserve_formatting: Whether to preserve text formatting
            backend: PDF processing backend to use
            image_quality: Quality for extracted images (1-100)
            table_detection_threshold: Confidence threshold for table detection
            min_text_length: Minimum text length to consider as valid chunk
            **kwargs: Additional parameters
        """
        super().__init__(
            name="pdf_chunker",
            category="document",
            supported_modalities=[ModalityType.TEXT, ModalityType.IMAGE, ModalityType.TABLE],
            **kwargs
        )

        self.pages_per_chunk = pages_per_chunk
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.preserve_formatting = preserve_formatting
        self.backend = backend
        self.image_quality = image_quality
        self.table_detection_threshold = table_detection_threshold
        self.min_text_length = min_text_length

        self.logger = logging.getLogger(__name__)

        # Validate backend availability
        self._validate_backend()

    def _validate_backend(self) -> None:
        """Validate that the requested backend is available."""
        if self.backend == "auto":
            if HAS_FITZ:
                self.backend = "pymupdf"
            elif HAS_PYPDF2:
                self.backend = "pypdf2"
            elif HAS_PDFMINER:
                self.backend = "pdfminer"
            else:
                raise ImportError(
                    "No PDF processing backend available. Install PyMuPDF, PyPDF2, or pdfminer.six"
                )
        elif self.backend == "pymupdf" and not HAS_FITZ:
            raise ImportError("PyMuPDF not available. Install with: pip install PyMuPDF")
        elif self.backend == "pypdf2" and not HAS_PYPDF2:
            raise ImportError("PyPDF2 not available. Install with: pip install PyPDF2")
        elif self.backend == "pdfminer" and not HAS_PDFMINER:
            raise ImportError("pdfminer.six not available. Install with: pip install pdfminer.six")

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk PDF content with support for images and tables.

        Args:
            content: PDF file path or PDF content bytes
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with PDF chunks including text, images, and tables
        """
        start_time = time.time()

        # Handle input types
        if isinstance(content, Path):
            pdf_path = content
        elif isinstance(content, str) and Path(content).exists():
            pdf_path = Path(content)
        elif isinstance(content, bytes):
            # Save bytes to temporary file for processing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                pdf_path = Path(tmp.name)
        else:
            raise ValueError(f"Invalid content type for PDF chunker: {type(content)}")

        if not pdf_path.exists():
            logger.error(f"📄 PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            # Extract content based on backend
            chunks = self._extract_pdf_content(pdf_path)

            # Create chunking result
            result = ChunkingResult(
                chunks=chunks,
                processing_time=time.time() - start_time,
                strategy_used="pdf_chunker",
                source_info={
                    "source_file": str(pdf_path),
                    "pages_per_chunk": self.pages_per_chunk,
                    "backend": self.backend,
                    "extract_images": self.extract_images,
                    "extract_tables": self.extract_tables
                }
            )

            logger.info(f"📄 PDF chunking completed: {len(chunks)} chunks from {pdf_path.name}")
            return result

        except Exception as e:
            logger.error(f"❌ Error processing PDF {pdf_path.name}: {e}")
            raise

    def _extract_pdf_content(self, pdf_path: Path) -> List[Chunk]:
        """Extract content from PDF using the selected backend."""
        if self.backend == "pymupdf":
            return self._extract_with_pymupdf(pdf_path)
        elif self.backend == "pypdf2":
            return self._extract_with_pypdf2(pdf_path)
        elif self.backend == "pdfminer":
            return self._extract_with_pdfminer(pdf_path)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _extract_with_pymupdf(self, pdf_path: Path) -> List[Chunk]:
        """Extract content using PyMuPDF (fitz) - best for images and tables."""
        if not HAS_FITZ:
            raise ImportError("PyMuPDF not available")

        doc = fitz.open(str(pdf_path))
        chunks = []

        try:
            total_pages = len(doc)

            for page_start in range(0, total_pages, self.pages_per_chunk):
                page_end = min(page_start + self.pages_per_chunk, total_pages)

                # Extract text content
                text_content = []
                images = []
                tables = []

                for page_num in range(page_start, page_end):
                    page = doc[page_num]

                    # Extract text
                    if self.preserve_formatting:
                        page_text = page.get_text("dict")
                        formatted_text = self._format_text_from_dict(page_text)
                    else:
                        formatted_text = page.get_text()

                    if formatted_text.strip():
                        text_content.append(f"=== Page {page_num + 1} ===\n{formatted_text}")

                    # Extract images
                    if self.extract_images:
                        page_images = self._extract_images_from_page(page, page_num)
                        images.extend(page_images)

                    # Extract tables
                    if self.extract_tables:
                        page_tables = self._extract_tables_from_page(page, page_num)
                        tables.extend(page_tables)

                # Create chunks for different content types
                chunk_text = "\n\n".join(text_content) if text_content else ""

                if chunk_text and len(chunk_text.strip()) >= self.min_text_length:
                    # Main text chunk
                    text_chunk = Chunk(
                        id=f"pdf_text_{page_start + 1}-{page_end}",
                        content=chunk_text,
                        modality=ModalityType.TEXT,
                        metadata=ChunkMetadata(
                            source=str(pdf_path),
                            page=page_start + 1,
                            position={"start_page": page_start + 1, "end_page": page_end},
                            extra={
                                "chunk_type": "text",
                                "pages_included": list(range(page_start + 1, page_end + 1)),
                                "backend": "pymupdf"
                            }
                        )
                    )
                    chunks.append(text_chunk)

                # Image chunks
                for img_data in images:
                    img_chunk = Chunk(
                        id=f"pdf_image_{img_data['page']}_{img_data['index']}",
                        content=img_data['description'],
                        modality=ModalityType.IMAGE,
                        metadata=ChunkMetadata(
                            source=str(pdf_path),
                            page=img_data['page'],
                            position=img_data['bbox'],
                            extra={
                                "chunk_type": "image",
                                "image_data": img_data['data'],
                                "image_format": img_data['format'],
                                "backend": "pymupdf"
                            }
                        )
                    )
                    chunks.append(img_chunk)

                # Table chunks
                for table_data in tables:
                    table_chunk = Chunk(
                        id=f"pdf_table_{table_data['page']}_{table_data['index']}",
                        content=table_data['content'],
                        modality=ModalityType.TABLE,
                        metadata=ChunkMetadata(
                            source=str(pdf_path),
                            page=table_data['page'],
                            position=table_data['bbox'],
                            extra={
                                "chunk_type": "table",
                                "table_structure": table_data['structure'],
                                "backend": "pymupdf"
                            }
                        )
                    )
                    chunks.append(table_chunk)

        finally:
            doc.close()

        return chunks

    def _format_text_from_dict(self, text_dict: Dict) -> str:
        """Format text from PyMuPDF dictionary format."""
        formatted_lines = []

        for block in text_dict.get("blocks", []):
            if "lines" not in block:  # Image block
                continue

            block_text = []
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                if line_text.strip():
                    block_text.append(line_text)

            if block_text:
                formatted_lines.append("\n".join(block_text))

        return "\n\n".join(formatted_lines)

    def _extract_images_from_page(self, page, page_num: int) -> List[Dict]:
        """Extract images from a page using PyMuPDF."""
        images = []
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Create description of the image
                bbox = page.get_image_rects(img)[0] if page.get_image_rects(img) else None
                description = f"Image {img_index + 1} on page {page_num + 1}"
                if bbox:
                    description += f" at position {bbox}"

                images.append({
                    "page": page_num + 1,
                    "index": img_index,
                    "data": image_bytes,
                    "format": image_ext,
                    "description": description,
                    "bbox": tuple(bbox) if bbox else None
                })

            except Exception as e:
                self.logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")

        return images

    def _extract_tables_from_page(self, page, page_num: int) -> List[Dict]:
        """Extract tables from a page (simplified implementation)."""
        tables = []

        # Simple table detection based on text layout
        # This is a basic implementation - could be enhanced with ML models
        text_dict = page.get_text("dict")
        potential_tables = self._detect_table_like_structures(text_dict, page_num)

        for table_index, table_data in enumerate(potential_tables):
            tables.append({
                "page": page_num + 1,
                "index": table_index,
                "content": table_data["content"],
                "structure": table_data["structure"],
                "bbox": table_data["bbox"]
            })

        return tables

    def _detect_table_like_structures(self, text_dict: Dict, page_num: int) -> List[Dict]:
        """Detect table-like structures in text layout."""
        tables = []

        # Look for patterns that suggest tabular data
        # This is a simplified heuristic - could be much more sophisticated
        for block_idx, block in enumerate(text_dict.get("blocks", [])):
            if "lines" not in block:
                continue

            lines = []
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                if line_text.strip():
                    lines.append(line_text.strip())

            # Simple heuristic: if multiple lines have similar structure (e.g., multiple spaces/tabs)
            if len(lines) >= 3:
                # Check if lines look tabular (contain multiple spaces/tabs)
                tabular_lines = [line for line in lines if "  " in line or "\t" in line]
                if len(tabular_lines) >= 2:
                    table_content = "\n".join(lines)
                    tables.append({
                        "content": f"Table detected on page {page_num + 1}:\n{table_content}",
                        "structure": {"rows": len(lines), "estimated_columns": self._estimate_columns(lines)},
                        "bbox": block.get("bbox")
                    })

        return tables

    def _estimate_columns(self, lines: List[str]) -> int:
        """Estimate number of columns in tabular data."""
        if not lines:
            return 0

        # Count spaces/tabs as column separators
        max_columns = 0
        for line in lines:
            # Split by multiple spaces or tabs
            import re
            columns = re.split(r'\s{2,}|\t+', line.strip())
            max_columns = max(max_columns, len(columns))

        return max_columns

    def _extract_with_pypdf2(self, pdf_path: Path) -> List[Chunk]:
        """Extract content using PyPDF2 - good for text extraction."""
        if not HAS_PYPDF2:
            raise ImportError("PyPDF2 not available")

        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            chunks = []
            total_pages = len(pdf_reader.pages)

            for page_start in range(0, total_pages, self.pages_per_chunk):
                page_end = min(page_start + self.pages_per_chunk, total_pages)

                text_content = []
                for page_num in range(page_start, page_end):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"=== Page {page_num + 1} ===\n{page_text}")

                chunk_text = "\n\n".join(text_content) if text_content else ""

                if chunk_text and len(chunk_text.strip()) >= self.min_text_length:
                    chunk = Chunk(
                        id=f"pdf_text_{page_start + 1}-{page_end}",
                        content=chunk_text,
                        modality=ModalityType.TEXT,
                        metadata=ChunkMetadata(
                            source=str(pdf_path),
                            page=page_start + 1,
                            position={"start_page": page_start + 1, "end_page": page_end},
                            extra={
                                "chunk_type": "text",
                                "pages_included": list(range(page_start + 1, page_end + 1)),
                                "backend": "pypdf2",
                                "note": "PyPDF2 backend - text only, no images/tables"
                            }
                        )
                    )
                    chunks.append(chunk)

        return chunks

    def _extract_with_pdfminer(self, pdf_path: Path) -> List[Chunk]:
        """Extract content using pdfminer - good for complex layouts."""
        if not HAS_PDFMINER:
            raise ImportError("pdfminer.six not available")

        # Extract full text with pdfminer
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
            all_texts=False
        )

        full_text = pdfminer_extract_text(str(pdf_path), laparams=laparams)

        # Simple chunking by estimated pages
        # This is approximate since pdfminer doesn't give page boundaries directly
        estimated_chars_per_page = 3000  # Rough estimate
        chunks = []

        text_length = len(full_text)
        chunk_size = estimated_chars_per_page * self.pages_per_chunk

        for start in range(0, text_length, chunk_size):
            end = min(start + chunk_size, text_length)
            chunk_text = full_text[start:end]

            if chunk_text.strip() and len(chunk_text.strip()) >= self.min_text_length:
                estimated_page = (start // estimated_chars_per_page) + 1

                chunk = Chunk(
                    id=f"pdf_text_estimated_page_{estimated_page}",
                    content=chunk_text.strip(),
                    modality=ModalityType.TEXT,
                    metadata=ChunkMetadata(
                        source=str(pdf_path),
                        page=estimated_page,
                        position={"start_char": start, "end_char": end},
                        extra={
                            "chunk_type": "text",
                            "backend": "pdfminer",
                            "note": "pdfminer backend - estimated page numbers"
                        }
                    )
                )
                chunks.append(chunk)

        return chunks

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ["pdf"]

    def estimate_chunks(self, content: Union[str, Path]) -> int:
        """Estimate number of chunks that will be generated."""
        if isinstance(content, (str, Path)):
            pdf_path = Path(content)
            if pdf_path.exists():
                try:
                    if HAS_FITZ:
                        doc = fitz.open(str(pdf_path))
                        total_pages = len(doc)
                        doc.close()
                        return (total_pages + self.pages_per_chunk - 1) // self.pages_per_chunk
                    elif HAS_PYPDF2:
                        with open(pdf_path, 'rb') as file:
                            pdf_reader = PdfReader(file)
                            total_pages = len(pdf_reader.pages)
                            return (total_pages + self.pages_per_chunk - 1) // self.pages_per_chunk
                except Exception:
                    pass

        return 1  # Default estimate

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """Adapt chunker parameters based on feedback."""
        if feedback_type == "quality" and feedback_score < 0.5:
            # Decrease pages per chunk for better granularity
            old_pages = self.pages_per_chunk
            self.pages_per_chunk = max(1, int(self.pages_per_chunk * 0.8))
            self.logger.info(f"Adapted pages_per_chunk: {old_pages} -> {self.pages_per_chunk} (quality feedback)")

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Increase pages per chunk for better performance
            old_pages = self.pages_per_chunk
            self.pages_per_chunk = min(10, int(self.pages_per_chunk * 1.2))
            self.logger.info(f"Adapted pages_per_chunk: {old_pages} -> {self.pages_per_chunk} (performance feedback)")

    def chunk_stream(self, content_stream, **kwargs):
        """
        Chunk content from a stream. For PDFs, we need the full content first.

        Args:
            content_stream: Stream containing PDF data
            **kwargs: Additional chunking parameters

        Returns:
            Generator yielding chunks
        """
        # For PDFs, we need to read the entire content first
        # since PDF structure requires full document analysis
        content_bytes = b""
        for chunk in content_stream:
            if isinstance(chunk, str):
                content_bytes += chunk.encode('utf-8')
            elif isinstance(chunk, bytes):
                content_bytes += chunk

        # Process the complete PDF content
        result = self.chunk(content_bytes, **kwargs)

        # Yield chunks one by one
        for chunk in result.chunks:
            yield chunk

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return []


# Only register if at least one PDF library is available
if HAS_FITZ or HAS_PYPDF2 or HAS_PDFMINER:
    # Apply the registration decorator
    PDFChunker = register_chunker(
        name="pdf_chunker",
        category="document",
        complexity=ComplexityLevel.HIGH,
        speed=SpeedLevel.MEDIUM,
        memory=MemoryUsage.MEDIUM,
        supported_formats=["pdf"],
        dependencies=[],  # No required dependencies since any one of the backends is sufficient
        optional_dependencies=["PyMuPDF", "PyPDF2", "pdfminer.six"],
        description="Advanced PDF chunking with image and table support (requires at least one PDF backend)",
        use_cases=["document_processing", "pdf_analysis", "mixed_media_extraction"]
    )(PDFChunker)
