"""
Chunking strategies organized by category and use case.

This package contains all available chunking strategies organized into logical
categories based on content type and algorithmic approach.

Categories:
- general: General-purpose algorithms (fixed size, rolling hash, CDC variants)
- text: Text-specific strategies (sentence, paragraph, semantic)
- document: Document-aware strategies (markdown, HTML, PDF)
- data_formats: Structured data formats (CSV, JSON, XML, YAML)
- multimedia: Image, audio, video chunking strategies
- data_streams: Stream processing and time-series strategies
- networking: Network protocol and packet-based strategies
- ml_ai: Machine learning and AI-enhanced strategies
- specialized: Domain-specific strategies (genomics, GIS, blockchain)
- compression_rt: Real-time and compression-aware strategies

Usage:
    # Direct import of specific strategies
    from chunking_strategy.strategies.text.sentence_based import SentenceBasedChunker
    from chunking_strategy.strategies.general.fixed_size import FixedSizeChunker

    # Using the registry system (recommended)
    from chunking_strategy import create_chunker
    chunker = create_chunker("sentence_based", max_sentences=3)
"""

# Import key strategies for easy access
from chunking_strategy.strategies.general.fixed_size import FixedSizeChunker
from chunking_strategy.strategies.general.fastcdc_chunker import FastCDCChunker
from chunking_strategy.strategies.general.adaptive_chunker import AdaptiveChunker
from chunking_strategy.strategies.general.context_enriched_chunker import ContextEnrichedChunker

# Hash-based chunkers
from chunking_strategy.strategies.general.rolling_hash_chunker import RollingHashChunker
from chunking_strategy.strategies.general.rabin_fingerprinting_chunker import RabinFingerprintingChunker
from chunking_strategy.strategies.general.buzhash_chunker import BuzHashChunker
from chunking_strategy.strategies.general.gear_cdc_chunker import GearCDCChunker
from chunking_strategy.strategies.general.ml_cdc_chunker import MLCDCChunker
from chunking_strategy.strategies.general.tttd_chunker import TTTDChunker
from chunking_strategy.strategies.text.sentence_based import SentenceBasedChunker
from chunking_strategy.strategies.text.discourse_aware_chunker import DiscourseAwareChunker
from chunking_strategy.strategies.document.pdf_chunker import PDFChunker
from chunking_strategy.strategies.document.doc_chunker import DocChunker
from chunking_strategy.strategies.document.markdown_chunker import MarkdownChunker
from chunking_strategy.strategies.document.xml_html_chunker import XMLHTMLChunker
from chunking_strategy.strategies.code.python_chunker import PythonCodeChunker
from chunking_strategy.strategies.code.c_cpp_chunker import CCppCodeChunker
from chunking_strategy.strategies.code.javascript_chunker import JavaScriptChunker
from chunking_strategy.strategies.code.go_chunker import GoChunker
from chunking_strategy.strategies.code.css_chunker import CSSChunker
from chunking_strategy.strategies.code.java_chunker import JavaChunker
from chunking_strategy.strategies.code.universal_code_chunker import UniversalCodeChunker
from chunking_strategy.strategies.data_formats.csv_chunker import CSVChunker
from chunking_strategy.strategies.data_formats.json_chunker import JSONChunker

# Re-export commonly used strategies
__all__ = [
    "FixedSizeChunker",
    "FastCDCChunker",
    "AdaptiveChunker",
    "ContextEnrichedChunker",

    # Hash-based chunkers
    "RollingHashChunker",
    "RabinFingerprintingChunker",
    "BuzHashChunker",
    "GearCDCChunker",
    "MLCDCChunker",
    "TTTDChunker",
    "SentenceBasedChunker",
    "DiscourseAwareChunker",
    "PDFChunker",
    "DocChunker",
    "MarkdownChunker",
    "XMLHTMLChunker",
    "PythonCodeChunker",
    "CCppCodeChunker",
    "JavaScriptChunker",
    "GoChunker",
    "CSSChunker",
    "JavaChunker",
    "UniversalCodeChunker",
    "CSVChunker",
    "JSONChunker",
]
