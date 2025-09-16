"""
JSON chunking strategy with intelligent object/array-based and semantic grouping.

This module provides specialized chunking for JSON files that understands:
- Object-based chunking with configurable object counts
- Array element chunking with smart batching
- Key-based logical grouping
- Memory-efficient size-based chunking
- Hierarchical depth-aware chunking
- Handling of nested and complex JSON structures
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator
import hashlib
from collections import defaultdict

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
    name="json_chunker",
    category="data_formats",
    description="Intelligent JSON chunking with object/array-based and semantic grouping strategies",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["json", "jsonl", "ndjson", "txt"],
    complexity=ComplexityLevel.MEDIUM,
    dependencies=["json"],
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    quality=0.85,  # High quality due to format awareness
    parameters_schema={
        "chunk_by": {
            "type": "string",
            "enum": ["objects", "array_elements", "key_groups", "size_limit", "depth_level"],
            "default": "objects",
            "description": "Method for chunking JSON data"
        },
        "objects_per_chunk": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10000,
            "default": 100,
            "description": "Number of objects per chunk (for 'objects' method)"
        },
        "elements_per_chunk": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100000,
            "default": 1000,
            "description": "Number of array elements per chunk (for 'array_elements' method)"
        },
        "group_by_key": {
            "type": "string",
            "default": None,
            "description": "Key name to group by (for 'key_groups' method)"
        },
        "size_limit_mb": {
            "type": "number",
            "minimum": 1,
            "maximum": 1000,
            "default": 10,
            "description": "Size limit per chunk in MB (for 'size_limit' method)"
        },
        "max_depth": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "default": 3,
            "description": "Maximum depth level per chunk (for 'depth_level' method)"
        },
        "preserve_structure": {
            "type": "boolean",
            "default": True,
            "description": "Maintain valid JSON structure in chunks"
        },
        "encoding": {
            "type": "string",
            "default": "utf-8",
            "description": "File encoding"
        },
        "handle_jsonl": {
            "type": "boolean",
            "default": True,
            "description": "Handle JSON Lines format (one JSON object per line)"
        },
        "compact_output": {
            "type": "boolean",
            "default": False,
            "description": "Use compact JSON output (no indentation)"
        }
    },
    default_parameters={
        "chunk_by": "objects",
        "objects_per_chunk": 100,
        "elements_per_chunk": 1000,
        "preserve_structure": True,
        "encoding": "utf-8",
        "handle_jsonl": True,
        "compact_output": False
    },
    use_cases=["api_data_processing", "json_etl", "document_stores", "streaming_json"],
    best_for=["large JSON files", "JSON arrays", "structured data", "API responses"],
    limitations=["requires valid JSON format", "memory usage scales with object complexity"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=True
)
class JSONChunker(StreamableChunker):
    """
    Intelligent JSON chunker that understands JSON structure and provides
    meaningful chunks based on objects, arrays, keys, size, or depth.

    Features:
    - Multiple chunking strategies (objects, arrays, key groups, size, depth)
    - JSON Lines (JSONL) support
    - Structure preservation with valid JSON output
    - Memory-efficient processing for large JSON files
    - Nested object and array handling
    - Configurable output formatting

    Examples:
        Object-based chunking:
        ```python
        chunker = JSONChunker(chunk_by="objects", objects_per_chunk=50)
        result = chunker.chunk("data.json")
        ```

        Array element chunking:
        ```python
        chunker = JSONChunker(
            chunk_by="array_elements",
            elements_per_chunk=500
        )
        result = chunker.chunk("large_array.json")
        ```

        Key-based grouping:
        ```python
        chunker = JSONChunker(
            chunk_by="key_groups",
            group_by_key="category"
        )
        result = chunker.chunk("products.json")
        ```

        Size-based chunking:
        ```python
        chunker = JSONChunker(
            chunk_by="size_limit",
            size_limit_mb=5
        )
        result = chunker.chunk("huge_dataset.json")
        ```
    """

    def __init__(
        self,
        chunk_by: str = "objects",
        objects_per_chunk: int = 100,
        elements_per_chunk: int = 1000,
        group_by_key: Optional[str] = None,
        size_limit_mb: float = 10,
        max_depth: int = 3,
        preserve_structure: bool = True,
        encoding: str = "utf-8",
        handle_jsonl: bool = True,
        compact_output: bool = False,
        **kwargs
    ):
        """
        Initialize JSON chunker.

        Args:
            chunk_by: Chunking method ('objects', 'array_elements', 'key_groups', 'size_limit', 'depth_level')
            objects_per_chunk: Number of objects per chunk (for 'objects' method)
            elements_per_chunk: Number of array elements per chunk (for 'array_elements' method)
            group_by_key: Key to group by (for 'key_groups' method)
            size_limit_mb: Size limit per chunk in MB
            max_depth: Maximum depth level per chunk
            preserve_structure: Whether to maintain valid JSON structure
            encoding: File encoding
            handle_jsonl: Handle JSON Lines format
            compact_output: Use compact JSON output
        """
        super().__init__(
            name="json_chunker",
            category="data_formats",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )
        self.chunk_by = chunk_by
        self.objects_per_chunk = objects_per_chunk
        self.elements_per_chunk = elements_per_chunk
        self.group_by_key = group_by_key
        self.size_limit_mb = size_limit_mb
        self.max_depth = max_depth
        self.preserve_structure = preserve_structure
        self.encoding = encoding
        self.handle_jsonl = handle_jsonl
        self.compact_output = compact_output

        # Validate parameters
        self._validate_parameters()

        logger.debug(f"Initialized JSONChunker with method: {chunk_by}")

    def _validate_parameters(self) -> None:
        """Validate chunker parameters."""
        valid_methods = ["objects", "array_elements", "key_groups", "size_limit", "depth_level"]
        if self.chunk_by not in valid_methods:
            raise ValueError(f"chunk_by must be one of {valid_methods}")

        if self.chunk_by == "key_groups" and not self.group_by_key:
            raise ValueError("group_by_key is required for key_groups method")

        if self.objects_per_chunk <= 0:
            raise ValueError("objects_per_chunk must be positive")

        if self.elements_per_chunk <= 0:
            raise ValueError("elements_per_chunk must be positive")

        if self.size_limit_mb <= 0:
            raise ValueError("size_limit_mb must be positive")

        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk JSON content using the specified strategy.

        Args:
            content: JSON content (file path, string, or bytes)
            source_info: Additional source information
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with JSON chunks
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
                with open(file_path, 'r', encoding=self.encoding) as f:
                    json_content = f.read()
            elif isinstance(content, str):
                # Could be file path or content - be more careful
                if len(content) < 1000 and '\n' not in content and '{' not in content and '[' not in content:
                    # Likely a file path - short string without JSON indicators
                    file_path = Path(content)
                    if file_path.exists():
                        source_info.update({
                            "source": str(file_path),
                            "source_type": "file",
                            "file_size": file_path.stat().st_size
                        })
                        with open(file_path, 'r', encoding=self.encoding) as f:
                            json_content = f.read()
                    else:
                        # File doesn't exist - treat as string content
                        json_content = content
                        source_info.update({
                            "source": "string",
                            "source_type": "content"
                        })
                else:
                    # Likely JSON content - long string or contains JSON indicators
                    json_content = content
                    source_info.update({
                        "source": "string",
                        "source_type": "content"
                    })
            elif isinstance(content, bytes):
                json_content = content.decode(self.encoding)
                source_info.update({
                    "source": "bytes",
                    "source_type": "content"
                })
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")

            # Parse JSON content
            json_data, is_jsonl = self._parse_json_content(json_content)

            if json_data is None:
                return ChunkingResult(
                    chunks=[],
                    processing_time=time.time() - start_time,
                    source_info={"json_objects": 0, "format": "invalid"},
                    strategy_used="json_chunker"
                )

            # Generate chunks based on strategy
            chunks = self._generate_chunks(json_data, is_jsonl, source_info)

            processing_time = time.time() - start_time

            return ChunkingResult(
                chunks=chunks,
                processing_time=processing_time,
                source_info={
                    "json_objects": len(json_data) if isinstance(json_data, list) else 1,
                    "format": "jsonl" if is_jsonl else "json",
                    "chunking_method": self.chunk_by,
                    "structure_preserved": self.preserve_structure
                },
                strategy_used="json_chunker"
            )

        except Exception as e:
            logger.error(f"Error chunking JSON content: {e}")
            raise

    def _parse_json_content(self, json_content: str) -> tuple[Optional[Union[Dict, List]], bool]:
        """Parse JSON content and determine if it's JSONL format."""
        json_content = json_content.strip()

        if not json_content:
            return None, False

        # Try standard JSON first
        try:
            data = json.loads(json_content)
            return data, False
        except json.JSONDecodeError:
            pass

        # Try JSON Lines format if enabled
        if self.handle_jsonl:
            try:
                objects = []
                for line in json_content.split('\n'):
                    line = line.strip()
                    if line:  # Skip empty lines
                        obj = json.loads(line)
                        objects.append(obj)
                return objects, True
            except json.JSONDecodeError:
                pass

        # Could not parse as valid JSON
        logger.warning("Could not parse content as valid JSON or JSONL")
        return None, False

    def _generate_chunks(
        self,
        json_data: Union[Dict, List],
        is_jsonl: bool,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Generate chunks based on the configured strategy."""

        if self.chunk_by == "objects":
            return self._chunk_by_objects(json_data, is_jsonl, source_info)
        elif self.chunk_by == "array_elements":
            return self._chunk_by_array_elements(json_data, is_jsonl, source_info)
        elif self.chunk_by == "key_groups":
            return self._chunk_by_key_groups(json_data, is_jsonl, source_info)
        elif self.chunk_by == "size_limit":
            return self._chunk_by_size_limit(json_data, is_jsonl, source_info)
        elif self.chunk_by == "depth_level":
            return self._chunk_by_depth_level(json_data, is_jsonl, source_info)
        else:
            raise ValueError(f"Unknown chunking method: {self.chunk_by}")

    def _chunk_by_objects(
        self,
        json_data: Union[Dict, List],
        is_jsonl: bool,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk JSON by number of objects."""
        chunks = []

        if isinstance(json_data, dict):
            # Single object - create one chunk
            chunk_content = self._format_json_output([json_data] if self.preserve_structure else json_data)

            metadata = ChunkMetadata(
                source=source_info.get("source", "unknown"),
                source_type=source_info.get("source_type", "content"),
                position="single object",
                length=len(chunk_content),
                extra={
                    "json_object_count": 1,
                    "json_format": "jsonl" if is_jsonl else "json",
                    "chunk_index": 0
                }
            )

            chunk = Chunk(
                id="json_object_0",
                content=chunk_content,
                metadata=metadata,
                modality=ModalityType.TEXT
            )

            chunks.append(chunk)

        elif isinstance(json_data, list):
            # Multiple objects - chunk by count
            for i in range(0, len(json_data), self.objects_per_chunk):
                chunk_objects = json_data[i:i + self.objects_per_chunk]

                chunk_content = self._format_json_output(chunk_objects)

                metadata = ChunkMetadata(
                    source=source_info.get("source", "unknown"),
                    source_type=source_info.get("source_type", "content"),
                    position=f"objects {i}-{i + len(chunk_objects) - 1}",
                    length=len(chunk_content),
                    extra={
                        "json_object_count": len(chunk_objects),
                        "json_format": "jsonl" if is_jsonl else "json",
                        "json_start_index": i,
                        "json_end_index": i + len(chunk_objects) - 1,
                        "chunk_index": i // self.objects_per_chunk
                    }
                )

                chunk = Chunk(
                    id=f"json_objects_{i // self.objects_per_chunk}",
                    content=chunk_content,
                    metadata=metadata,
                    modality=ModalityType.TEXT
                )

                chunks.append(chunk)

        return chunks

    def _chunk_by_array_elements(
        self,
        json_data: Union[Dict, List],
        is_jsonl: bool,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk JSON arrays by number of elements."""
        chunks = []

        def extract_arrays(obj, path=""):
            """Recursively extract arrays from JSON structure."""
            arrays = []
            if isinstance(obj, list):
                arrays.append((path or "root", obj))
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    arrays.extend(extract_arrays(value, new_path))
            return arrays

        # Find all arrays in the JSON structure
        arrays = extract_arrays(json_data)

        if not arrays:
            # No arrays found - fall back to object chunking
            return self._chunk_by_objects(json_data, is_jsonl, source_info)

        chunk_index = 0
        for array_path, array_data in arrays:
            for i in range(0, len(array_data), self.elements_per_chunk):
                chunk_elements = array_data[i:i + self.elements_per_chunk]

                if self.preserve_structure:
                    # Wrap in appropriate structure
                    if array_path == "root":
                        chunk_content = self._format_json_output(chunk_elements)
                    else:
                        # Create nested structure
                        nested_obj = {}
                        keys = array_path.split('.')
                        current = nested_obj
                        for key in keys[:-1]:
                            current[key] = {}
                            current = current[key]
                        current[keys[-1]] = chunk_elements
                        chunk_content = self._format_json_output(nested_obj)
                else:
                    chunk_content = self._format_json_output(chunk_elements)

                metadata = ChunkMetadata(
                    source=source_info.get("source", "unknown"),
                    source_type=source_info.get("source_type", "content"),
                    position=f"array '{array_path}' elements {i}-{i + len(chunk_elements) - 1}",
                    length=len(chunk_content),
                    extra={
                        "json_array_path": array_path,
                        "json_element_count": len(chunk_elements),
                        "json_format": "jsonl" if is_jsonl else "json",
                        "json_start_index": i,
                        "json_end_index": i + len(chunk_elements) - 1,
                        "chunk_index": chunk_index
                    }
                )

                chunk = Chunk(
                    id=f"json_array_{self._safe_path_id(array_path)}_{chunk_index}",
                    content=chunk_content,
                    metadata=metadata,
                    modality=ModalityType.TEXT
                )

                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _chunk_by_key_groups(
        self,
        json_data: Union[Dict, List],
        is_jsonl: bool,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk JSON by grouping objects with the same key value."""
        if not isinstance(json_data, list):
            # Convert single object to list for processing
            json_data = [json_data]

        # Group objects by key value
        groups = defaultdict(list)
        ungrouped = []

        for obj in json_data:
            if isinstance(obj, dict) and self.group_by_key in obj:
                key_value = obj[self.group_by_key]
                # Convert to string for consistent grouping
                key_str = str(key_value) if key_value is not None else "null"
                groups[key_str].append(obj)
            else:
                ungrouped.append(obj)

        chunks = []
        chunk_index = 0

        # Create chunks for each group
        for group_value, group_objects in groups.items():
            chunk_content = self._format_json_output(group_objects)

            metadata = ChunkMetadata(
                source=source_info.get("source", "unknown"),
                source_type=source_info.get("source_type", "content"),
                position=f"key group '{group_value}' ({len(group_objects)} objects)",
                length=len(chunk_content),
                extra={
                    "json_group_key": self.group_by_key,
                    "json_group_value": group_value,
                    "json_object_count": len(group_objects),
                    "json_format": "jsonl" if is_jsonl else "json",
                    "chunk_index": chunk_index
                }
            )

            chunk = Chunk(
                id=f"json_group_{self._safe_group_id(group_value)}",
                content=chunk_content,
                metadata=metadata,
                modality=ModalityType.TEXT
            )

            chunks.append(chunk)
            chunk_index += 1

        # Handle ungrouped objects if any
        if ungrouped:
            chunk_content = self._format_json_output(ungrouped)

            metadata = ChunkMetadata(
                source=source_info.get("source", "unknown"),
                source_type=source_info.get("source_type", "content"),
                position=f"ungrouped objects ({len(ungrouped)} objects)",
                length=len(chunk_content),
                extra={
                    "json_group_key": self.group_by_key,
                    "json_group_value": "ungrouped",
                    "json_object_count": len(ungrouped),
                    "json_format": "jsonl" if is_jsonl else "json",
                    "chunk_index": chunk_index
                }
            )

            chunk = Chunk(
                id="json_group_ungrouped",
                content=chunk_content,
                metadata=metadata,
                modality=ModalityType.TEXT
            )

            chunks.append(chunk)

        return chunks

    def _chunk_by_size_limit(
        self,
        json_data: Union[Dict, List],
        is_jsonl: bool,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk JSON by approximate size limits."""
        chunks = []
        max_size_bytes = self.size_limit_mb * 1024 * 1024

        if isinstance(json_data, dict):
            # Single object - check if it fits in one chunk
            content = self._format_json_output(json_data)
            if len(content.encode('utf-8')) <= max_size_bytes:
                metadata = ChunkMetadata(
                    source=source_info.get("source", "unknown"),
                    source_type=source_info.get("source_type", "content"),
                    position="size chunk 1",
                    length=len(content),
                    extra={
                        "json_object_count": 1,
                        "json_format": "jsonl" if is_jsonl else "json",
                        "json_size_mb": len(content.encode('utf-8')) / (1024 * 1024),
                        "chunk_index": 0
                    }
                )

                chunk = Chunk(
                    id="json_size_chunk_0",
                    content=content,
                    metadata=metadata,
                    modality=ModalityType.TEXT
                )

                chunks.append(chunk)
            else:
                # Object too large - need to split (simplified approach)
                logger.warning("Single JSON object exceeds size limit")
                chunks.extend(self._chunk_by_objects(json_data, is_jsonl, source_info))

        elif isinstance(json_data, list):
            current_objects = []
            current_size = 0
            chunk_index = 0

            for obj in json_data:
                obj_content = json.dumps(obj, ensure_ascii=False, separators=(',', ':') if self.compact_output else None)
                obj_size = len(obj_content.encode('utf-8'))

                if current_size + obj_size > max_size_bytes and current_objects:
                    # Create chunk with current objects
                    chunk_content = self._format_json_output(current_objects)

                    metadata = ChunkMetadata(
                        source=source_info.get("source", "unknown"),
                        source_type=source_info.get("source_type", "content"),
                        position=f"size chunk {chunk_index + 1}",
                        length=len(chunk_content),
                        extra={
                            "json_object_count": len(current_objects),
                            "json_format": "jsonl" if is_jsonl else "json",
                            "json_size_mb": current_size / (1024 * 1024),
                            "chunk_index": chunk_index
                        }
                    )

                    chunk = Chunk(
                        id=f"json_size_chunk_{chunk_index}",
                        content=chunk_content,
                        metadata=metadata,
                        modality=ModalityType.TEXT
                    )

                    chunks.append(chunk)

                    # Reset for next chunk
                    current_objects = []
                    current_size = 0
                    chunk_index += 1

                current_objects.append(obj)
                current_size += obj_size

            # Handle remaining objects
            if current_objects:
                chunk_content = self._format_json_output(current_objects)

                metadata = ChunkMetadata(
                    source=source_info.get("source", "unknown"),
                    source_type=source_info.get("source_type", "content"),
                    position=f"size chunk {chunk_index + 1}",
                    length=len(chunk_content),
                    extra={
                        "json_object_count": len(current_objects),
                        "json_format": "jsonl" if is_jsonl else "json",
                        "json_size_mb": current_size / (1024 * 1024),
                        "chunk_index": chunk_index
                    }
                )

                chunk = Chunk(
                    id=f"json_size_chunk_{chunk_index}",
                    content=chunk_content,
                    metadata=metadata,
                    modality=ModalityType.TEXT
                )

                chunks.append(chunk)

        return chunks

    def _chunk_by_depth_level(
        self,
        json_data: Union[Dict, List],
        is_jsonl: bool,
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk JSON by limiting the depth of nested structures."""

        def limit_depth(obj, current_depth=0):
            """Recursively limit the depth of JSON object."""
            if current_depth >= self.max_depth:
                if isinstance(obj, (dict, list)):
                    return f"<truncated_{type(obj).__name__}_depth_{current_depth}>"
                return obj

            if isinstance(obj, dict):
                return {key: limit_depth(value, current_depth + 1) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [limit_depth(item, current_depth + 1) for item in obj]
            else:
                return obj

        # Apply depth limiting
        limited_data = limit_depth(json_data)

        # Create chunks with depth-limited data
        chunks = []
        chunk_content = self._format_json_output(limited_data)

        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"depth-limited (max depth: {self.max_depth})",
            length=len(chunk_content),
            extra={
                "json_max_depth": self.max_depth,
                "json_format": "jsonl" if is_jsonl else "json",
                "json_object_count": len(limited_data) if isinstance(limited_data, list) else 1,
                "chunk_index": 0
            }
        )

        chunk = Chunk(
            id="json_depth_chunk_0",
            content=chunk_content,
            metadata=metadata,
            modality=ModalityType.TEXT
        )

        chunks.append(chunk)
        return chunks

    def _format_json_output(self, data: Union[Dict, List]) -> str:
        """Format JSON data for output."""
        if self.compact_output:
            return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        else:
            return json.dumps(data, ensure_ascii=False, indent=2)

    def _safe_path_id(self, path: str) -> str:
        """Create a safe ID from JSON path."""
        if len(path) > 50:
            return hashlib.md5(path.encode()).hexdigest()[:8]
        return path.replace('.', '_').replace('[', '_').replace(']', '_')

    def _safe_group_id(self, group_value: str) -> str:
        """Create a safe ID from group value."""
        if len(str(group_value)) > 50 or not str(group_value).replace('_', '').isalnum():
            return hashlib.md5(str(group_value).encode()).hexdigest()[:8]
        return str(group_value).replace(' ', '_').replace('-', '_')

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream JSON chunks for memory-efficient processing of large files.

        Args:
            content_stream: Iterator of content chunks
            source_info: Source information
            **kwargs: Additional parameters

        Yields:
            Individual chunks as they are processed
        """
        source_info = source_info or {}

        # For JSON, we need to accumulate content to parse properly
        # TODO: Could be enhanced for true streaming JSON parsing
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
            if self.chunk_by == "objects":
                self.objects_per_chunk = max(10, int(self.objects_per_chunk * 0.7))
            elif self.chunk_by == "array_elements":
                self.elements_per_chunk = max(10, int(self.elements_per_chunk * 0.7))
            elif self.chunk_by == "size_limit":
                self.size_limit_mb = max(1, self.size_limit_mb * 0.7)

        elif feedback_type == "performance" and feedback_score < 0.5:
            # Poor performance - try larger chunks
            if self.chunk_by == "objects":
                self.objects_per_chunk = min(1000, int(self.objects_per_chunk * 1.3))
            elif self.chunk_by == "array_elements":
                self.elements_per_chunk = min(10000, int(self.elements_per_chunk * 1.3))
            elif self.chunk_by == "size_limit":
                self.size_limit_mb = min(100, self.size_limit_mb * 1.3)

        logger.debug(f"Adapted JSON chunker parameters based on {feedback_type} feedback: {feedback_score}")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        # For now, return empty list - could be extended to track adaptations
        return []
