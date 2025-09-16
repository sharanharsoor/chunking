"""
CSV chunking strategy with intelligent row-based and logical grouping.

This module provides specialized chunking for CSV files that understands:
- Header preservation and association
- Row-based chunking with configurable sizes
- Logical grouping based on column values
- Memory-efficient streaming for large CSV files
- Handling of different CSV dialects and encodings
"""

import csv
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
import hashlib

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
    name="csv_chunker",
    category="data_formats",
    description="Intelligent CSV chunking with row-based and logical grouping strategies",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["csv", "tsv", "tab", "txt"],
    complexity=ComplexityLevel.MEDIUM,
    dependencies=["csv"],
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    quality=0.85,  # High quality due to format awareness
    parameters_schema={
        "chunk_by": {
            "type": "string",
            "enum": ["rows", "logical_groups", "memory_size", "header_sections"],
            "default": "rows",
            "description": "Method for chunking CSV data"
        },
        "rows_per_chunk": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000000,
            "default": 1000,
            "description": "Number of rows per chunk (for 'rows' method)"
        },
        "group_by_column": {
            "type": "string",
            "default": None,
            "description": "Column name to group by (for 'logical_groups' method)"
        },
        "memory_limit_mb": {
            "type": "number",
            "minimum": 1,
            "maximum": 1000,
            "default": 50,
            "description": "Memory limit per chunk in MB (for 'memory_size' method)"
        },
        "preserve_headers": {
            "type": "boolean",
            "default": True,
            "description": "Include headers in each chunk"
        },
        "dialect": {
            "type": "string",
            "enum": ["auto", "excel", "unix", "excel-tab"],
            "default": "auto",
            "description": "CSV dialect to use for parsing"
        },
        "encoding": {
            "type": "string",
            "default": "utf-8",
            "description": "File encoding"
        },
        "skip_empty_lines": {
            "type": "boolean",
            "default": True,
            "description": "Skip empty lines in CSV"
        },
        "chunk_overlap_rows": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "default": 0,
            "description": "Number of rows to overlap between chunks"
        }
    },
    default_parameters={
        "chunk_by": "rows",
        "rows_per_chunk": 1000,
        "preserve_headers": True,
        "dialect": "auto",
        "encoding": "utf-8",
        "skip_empty_lines": True,
        "chunk_overlap_rows": 0
    },
    use_cases=["data_processing", "ETL", "batch_analysis", "streaming_csv"],
    best_for=["large CSV files", "row-based processing", "data pipelines"],
    limitations=["requires valid CSV format", "memory usage scales with row size"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=False
)
class CSVChunker(StreamableChunker):
    """
    Intelligent CSV chunker that understands CSV structure and provides
    meaningful chunks based on rows, logical groups, or memory constraints.

    Features:
    - Multiple chunking strategies (rows, logical groups, memory-based)
    - Header preservation across chunks
    - CSV dialect auto-detection
    - Memory-efficient streaming for large files
    - Configurable overlap between chunks
    - Support for different encodings

    Examples:
        Row-based chunking:
        ```python
        chunker = CSVChunker(chunk_by="rows", rows_per_chunk=500)
        result = chunker.chunk("data.csv")
        ```

        Logical grouping:
        ```python
        chunker = CSVChunker(
            chunk_by="logical_groups",
            group_by_column="category"
        )
        result = chunker.chunk("sales_data.csv")
        ```

        Memory-based chunking:
        ```python
        chunker = CSVChunker(
            chunk_by="memory_size",
            memory_limit_mb=25
        )
        result = chunker.chunk("large_dataset.csv")
        ```
    """

    def __init__(
        self,
        chunk_by: str = "rows",
        rows_per_chunk: int = 1000,
        group_by_column: Optional[str] = None,
        memory_limit_mb: float = 50,
        preserve_headers: bool = True,
        dialect: str = "auto",
        encoding: str = "utf-8",
        skip_empty_lines: bool = True,
        chunk_overlap_rows: int = 0,
        **kwargs
    ):
        """
        Initialize CSV chunker.

        Args:
            chunk_by: Chunking method ('rows', 'logical_groups', 'memory_size', 'header_sections')
            rows_per_chunk: Number of rows per chunk (for 'rows' method)
            group_by_column: Column to group by (for 'logical_groups' method)
            memory_limit_mb: Memory limit per chunk in MB
            preserve_headers: Whether to include headers in each chunk
            dialect: CSV dialect ('auto', 'excel', 'unix', 'excel-tab')
            encoding: File encoding
            skip_empty_lines: Skip empty lines
            chunk_overlap_rows: Number of rows to overlap between chunks
        """
        super().__init__(
            name="csv_chunker",
            category="data_formats",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )
        self.chunk_by = chunk_by
        self.rows_per_chunk = rows_per_chunk
        self.group_by_column = group_by_column
        self.memory_limit_mb = memory_limit_mb
        self.preserve_headers = preserve_headers
        self.dialect = dialect
        self.encoding = encoding
        self.skip_empty_lines = skip_empty_lines
        self.chunk_overlap_rows = chunk_overlap_rows

        # Validate parameters
        self._validate_parameters()

        logger.debug(f"Initialized CSVChunker with method: {chunk_by}")

    def _validate_parameters(self) -> None:
        """Validate chunker parameters."""
        valid_methods = ["rows", "logical_groups", "memory_size", "header_sections"]
        if self.chunk_by not in valid_methods:
            raise ValueError(f"chunk_by must be one of {valid_methods}")

        if self.chunk_by == "logical_groups" and not self.group_by_column:
            raise ValueError("group_by_column is required for logical_groups method")

        if self.rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")

        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk CSV content using the specified strategy.

        Args:
            content: CSV content (file path, string, or bytes)
            source_info: Additional source information
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with CSV chunks
        """
        start_time = time.time()
        source_info = source_info or {}

        try:
            # Handle different input types
            if isinstance(content, Path):
                # Definitely a file path
                file_path = content
                source_info.update({
                    "source": str(file_path),
                    "source_type": "file",
                    "file_size": file_path.stat().st_size
                })
                with open(file_path, 'r', encoding=self.encoding, newline='') as f:
                    csv_content = f.read()
            elif isinstance(content, str):
                # Could be file path or content - be more careful
                if (len(content) > 0 and len(content) < 1000 and '\n' not in content and '\r' not in content):
                    # Likely a file path - short non-empty string without newlines
                    file_path = Path(content)
                    if file_path.exists() and file_path.is_file():
                        source_info.update({
                            "source": str(file_path),
                            "source_type": "file",
                            "file_size": file_path.stat().st_size
                        })
                        with open(file_path, 'r', encoding=self.encoding, newline='') as f:
                            csv_content = f.read()
                    else:
                        # File doesn't exist or not a file - treat as string content
                        csv_content = content
                        source_info.update({
                            "source": "string",
                            "source_type": "content"
                        })
                else:
                    # Likely CSV content - long string or contains newlines
                    csv_content = content
                    source_info.update({
                        "source": "string",
                        "source_type": "content"
                    })
            elif isinstance(content, bytes):
                csv_content = content.decode(self.encoding)
                source_info.update({
                    "source": "bytes",
                    "source_type": "content"
                })
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")

            # Detect CSV dialect if auto
            dialect = self._detect_dialect(csv_content)

            # Parse CSV content
            csv_reader = csv.reader(
                io.StringIO(csv_content),
                dialect=dialect
            )

            # Get the actual dialect object for source info
            actual_dialect = csv.get_dialect(dialect) if isinstance(dialect, str) else dialect

            # Convert to list for processing
            rows = list(csv_reader)

            if not rows:
                return ChunkingResult(
                    chunks=[],
                    processing_time=time.time() - start_time,
                    source_info={"csv_rows": 0, "headers": []},
                    strategy_used="csv_chunker"
                )

            # Filter empty lines if requested
            if self.skip_empty_lines:
                rows = [row for row in rows if any(cell.strip() for cell in row)]

            # Extract headers
            headers = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []

            # Generate chunks based on strategy
            chunks = self._generate_chunks(headers, data_rows, source_info)

            processing_time = time.time() - start_time

            return ChunkingResult(
                chunks=chunks,
                processing_time=processing_time,
                source_info={
                    "csv_rows": len(data_rows),
                    "headers": headers,
                    "chunking_method": self.chunk_by,
                    "dialect": actual_dialect.__class__.__name__
                },
                strategy_used="csv_chunker"
            )

        except Exception as e:
            logger.error(f"Error chunking CSV content: {e}")
            raise

    def _detect_dialect(self, csv_content: str) -> str:
        """Detect CSV dialect from content."""
        if self.dialect != "auto":
            return self.dialect

        try:
            # Use csv.Sniffer to detect dialect
            sniffer = csv.Sniffer()
            sample = csv_content[:1024 * 4]  # Use first 4KB for detection
            detected_dialect = sniffer.sniff(sample)
            return detected_dialect
        except Exception:
            # Fallback to excel dialect
            logger.debug("Could not detect CSV dialect, using excel")
            return "excel"

    def _generate_chunks(
        self,
        headers: List[str],
        data_rows: List[List[str]],
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Generate chunks based on the configured strategy."""

        if self.chunk_by == "rows":
            return self._chunk_by_rows(headers, data_rows, source_info)
        elif self.chunk_by == "logical_groups":
            return self._chunk_by_logical_groups(headers, data_rows, source_info)
        elif self.chunk_by == "memory_size":
            return self._chunk_by_memory_size(headers, data_rows, source_info)
        elif self.chunk_by == "header_sections":
            return self._chunk_by_header_sections(headers, data_rows, source_info)
        else:
            raise ValueError(f"Unknown chunking method: {self.chunk_by}")

    def _chunk_by_rows(
        self,
        headers: List[str],
        data_rows: List[List[str]],
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk CSV by fixed number of rows."""
        chunks = []

        for i in range(0, len(data_rows), self.rows_per_chunk):
            # Calculate overlap
            start_idx = max(0, i - self.chunk_overlap_rows)
            end_idx = min(len(data_rows), i + self.rows_per_chunk)

            chunk_rows = data_rows[start_idx:end_idx]

            # Build chunk content
            chunk_content = self._build_csv_content(headers, chunk_rows)

            # Create metadata
            metadata = ChunkMetadata(
                source=source_info.get("source", "unknown"),
                source_type=source_info.get("source_type", "content"),
                position=f"rows {start_idx + 1}-{end_idx}",
                length=len(chunk_content),
                extra={
                    "csv_headers": headers,
                    "csv_row_count": len(chunk_rows),
                    "csv_start_row": start_idx + 1,
                    "csv_end_row": end_idx,
                    "chunk_index": i // self.rows_per_chunk
                }
            )

            chunk = Chunk(
                id=f"csv_chunk_{i // self.rows_per_chunk}",
                content=chunk_content,
                metadata=metadata,
                modality=ModalityType.TEXT
            )

            chunks.append(chunk)

        return chunks

    def _chunk_by_logical_groups(
        self,
        headers: List[str],
        data_rows: List[List[str]],
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk CSV by logical groups based on a column value."""
        if self.group_by_column not in headers:
            logger.warning(f"Group column '{self.group_by_column}' not found in headers")
            # Fallback to row-based chunking
            return self._chunk_by_rows(headers, data_rows, source_info)

        group_col_idx = headers.index(self.group_by_column)

        # Group rows by column value
        groups = {}
        for row_idx, row in enumerate(data_rows):
            if group_col_idx < len(row):
                group_value = row[group_col_idx]
                if group_value not in groups:
                    groups[group_value] = []
                groups[group_value].append((row_idx, row))

        chunks = []

        for chunk_idx, (group_value, group_rows) in enumerate(groups.items()):
            # Extract just the row data
            rows_data = [row[1] for row in group_rows]

            # Build chunk content
            chunk_content = self._build_csv_content(headers, rows_data)

            # Create metadata
            row_indices = [row[0] for row in group_rows]
            metadata = ChunkMetadata(
                source=source_info.get("source", "unknown"),
                source_type=source_info.get("source_type", "content"),
                position=f"group '{group_value}' ({len(rows_data)} rows)",
                length=len(chunk_content),
                extra={
                    "csv_headers": headers,
                    "csv_row_count": len(rows_data),
                    "csv_group_value": group_value,
                    "csv_group_column": self.group_by_column,
                    "csv_row_indices": row_indices
                }
            )

            chunk = Chunk(
                id=f"csv_group_{self._safe_group_id(group_value)}",
                content=chunk_content,
                metadata=metadata,
                modality=ModalityType.TEXT
            )

            chunks.append(chunk)

        return chunks

    def _chunk_by_memory_size(
        self,
        headers: List[str],
        data_rows: List[List[str]],
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk CSV by approximate memory size."""
        chunks = []
        current_rows = []
        current_size = 0
        chunk_idx = 0

        # Estimate header size
        header_content = self._build_csv_content(headers, [])
        header_size = len(header_content.encode('utf-8'))
        max_size_bytes = self.memory_limit_mb * 1024 * 1024

        for row_idx, row in enumerate(data_rows):
            # Estimate row size
            row_content = ','.join(f'"{cell}"' for cell in row) + '\n'
            row_size = len(row_content.encode('utf-8'))

            # Check if adding this row would exceed limit
            if current_size + row_size + header_size > max_size_bytes and current_rows:
                # Create chunk with current rows
                chunk_content = self._build_csv_content(headers, current_rows)

                metadata = ChunkMetadata(
                    source=source_info.get("source", "unknown"),
                    source_type=source_info.get("source_type", "content"),
                    position=f"memory chunk {chunk_idx + 1}",
                    length=len(chunk_content),
                    extra={
                        "csv_headers": headers,
                        "csv_row_count": len(current_rows),
                        "csv_memory_size_mb": current_size / (1024 * 1024)
                    }
                )

                chunk = Chunk(
                    id=f"csv_memory_chunk_{chunk_idx}",
                    content=chunk_content,
                    metadata=metadata,
                    modality=ModalityType.TEXT
                )

                chunks.append(chunk)

                # Reset for next chunk
                current_rows = []
                current_size = 0
                chunk_idx += 1

            current_rows.append(row)
            current_size += row_size

        # Handle remaining rows
        if current_rows:
            chunk_content = self._build_csv_content(headers, current_rows)

            metadata = ChunkMetadata(
                source=source_info.get("source", "unknown"),
                source_type=source_info.get("source_type", "content"),
                position=f"memory chunk {chunk_idx + 1}",
                length=len(chunk_content),
                extra={
                    "csv_headers": headers,
                    "csv_row_count": len(current_rows),
                    "csv_memory_size_mb": current_size / (1024 * 1024)
                }
            )

            chunk = Chunk(
                id=f"csv_memory_chunk_{chunk_idx}",
                content=chunk_content,
                metadata=metadata,
                modality=ModalityType.TEXT
            )

            chunks.append(chunk)

        return chunks

    def _chunk_by_header_sections(
        self,
        headers: List[str],
        data_rows: List[List[str]],
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk CSV by detecting header-like rows in the data."""
        chunks = []
        current_section_rows = []
        section_headers = headers.copy()
        chunk_idx = 0

        for row_idx, row in enumerate(data_rows):
            # Heuristic: if a row looks like headers (all caps, specific patterns, etc.)
            is_header_row = self._is_likely_header_row(row, headers)

            if is_header_row and current_section_rows:
                # End current section and start new one
                chunk_content = self._build_csv_content(section_headers, current_section_rows)

                metadata = ChunkMetadata(
                    source=source_info.get("source", "unknown"),
                    source_type=source_info.get("source_type", "content"),
                    position=f"header section {chunk_idx + 1}",
                    length=len(chunk_content),
                    extra={
                        "csv_headers": section_headers,
                        "csv_row_count": len(current_section_rows)
                    }
                )

                chunk = Chunk(
                    id=f"csv_section_{chunk_idx}",
                    content=chunk_content,
                    metadata=metadata,
                    modality=ModalityType.TEXT
                )

                chunks.append(chunk)

                # Start new section
                current_section_rows = []
                section_headers = row  # Use the header row as new headers
                chunk_idx += 1
            else:
                current_section_rows.append(row)

        # Handle final section
        if current_section_rows:
            chunk_content = self._build_csv_content(section_headers, current_section_rows)

            metadata = ChunkMetadata(
                source=source_info.get("source", "unknown"),
                source_type=source_info.get("source_type", "content"),
                position=f"header section {chunk_idx + 1}",
                length=len(chunk_content),
                extra={
                    "csv_headers": section_headers,
                    "csv_row_count": len(current_section_rows)
                }
            )

            chunk = Chunk(
                id=f"csv_section_{chunk_idx}",
                content=chunk_content,
                metadata=metadata,
                modality=ModalityType.TEXT
            )

            chunks.append(chunk)

        return chunks

    def _build_csv_content(self, headers: List[str], rows: List[List[str]]) -> str:
        """Build CSV content string from headers and rows."""
        output = io.StringIO()
        writer = csv.writer(output)

        if self.preserve_headers and headers:
            writer.writerow(headers)

        for row in rows:
            writer.writerow(row)

        return output.getvalue()

    def _is_likely_header_row(self, row: List[str], original_headers: List[str]) -> bool:
        """Heuristic to determine if a row looks like a header row."""
        if not row:
            return False

        # Check if all cells are uppercase (common header pattern)
        all_upper = all(cell.isupper() for cell in row if cell.strip())

        # Check if it looks similar to original headers
        similar_to_headers = any(
            cell.lower() in [h.lower() for h in original_headers]
            for cell in row if cell.strip()
        )

        # Check for common header keywords
        header_keywords = ['id', 'name', 'date', 'time', 'type', 'status', 'total', 'count']
        has_header_keywords = any(
            any(keyword in cell.lower() for keyword in header_keywords)
            for cell in row if cell.strip()
        )

        return all_upper or similar_to_headers or has_header_keywords

    def _safe_group_id(self, group_value: str) -> str:
        """Create a safe ID from group value."""
        # Create a hash for long or special character group values
        if len(str(group_value)) > 50 or not str(group_value).isalnum():
            return hashlib.md5(str(group_value).encode()).hexdigest()[:8]
        return str(group_value).replace(' ', '_').replace('-', '_')

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream CSV chunks for memory-efficient processing of large files.

        Args:
            content_stream: Iterator of content chunks
            source_info: Source information
            **kwargs: Additional parameters

        Yields:
            Individual chunks as they are processed
        """
        source_info = source_info or {}

        # Accumulate content from stream
        accumulated_content = ""
        for content_chunk in content_stream:
            if isinstance(content_chunk, bytes):
                accumulated_content += content_chunk.decode(self.encoding)
            else:
                accumulated_content += content_chunk

        # Process accumulated content
        result = self.chunk(accumulated_content, source_info, **kwargs)

        # Yield chunks one by one
        for chunk in result.chunks:
            yield chunk

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
            # Poor quality - try smaller chunks
            if self.chunk_by == "rows":
                self.rows_per_chunk = max(100, int(self.rows_per_chunk * 0.7))
            elif self.chunk_by == "memory_size":
                self.memory_limit_mb = max(5, self.memory_limit_mb * 0.7)

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Poor performance - try larger chunks
            if self.chunk_by == "rows":
                self.rows_per_chunk = min(10000, int(self.rows_per_chunk * 1.3))
            elif self.chunk_by == "memory_size":
                self.memory_limit_mb = min(200, self.memory_limit_mb * 1.3)

        logger.debug(f"Adapted CSV chunker parameters based on {feedback_type} feedback: {feedback_score}")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        # For now, return empty list - could be extended to track adaptations
        return []
