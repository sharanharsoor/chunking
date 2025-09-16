"""
Data format-specific chunking strategies.

This module provides specialized chunking strategies for structured data formats
like CSV, JSON, XML, YAML, and other data interchange formats.

The strategies in this module understand the inherent structure of these formats
and can provide semantically meaningful chunks that preserve data relationships
and boundaries.

Strategies:
- csv_chunker: Row-based and logical grouping for CSV files
- json_chunker: Object/array-based chunking for JSON documents
- xml_chunker: Tag-aware and hierarchical chunking for XML documents
- yaml_chunker: Document-aware chunking for YAML files

Usage:
    from chunking_strategy.strategies.data_formats.csv_chunker import CSVChunker
    from chunking_strategy import create_chunker

    # Direct instantiation
    chunker = CSVChunker(chunk_by="rows", rows_per_chunk=100)

    # Via registry
    chunker = create_chunker("csv_chunker", chunk_by="logical_groups")
"""

from chunking_strategy.strategies.data_formats.csv_chunker import CSVChunker
from chunking_strategy.strategies.data_formats.json_chunker import JSONChunker

__all__ = [
    "CSVChunker",
    "JSONChunker",
]
