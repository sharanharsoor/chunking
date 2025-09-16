"""
Chunking Strategy Library

A comprehensive chunking library for text, documents, audio, video, and data streams.
Supports multiple chunking strategies, streaming processing, adaptive chunking,
and integration with vector databases and LLM workflows.

Public API Examples:

Direct imports:
    from chunking_strategy import sentence_based, semantic_chunking
    from chunking_strategy.strategies.text import SentenceBasedChunker
    from chunking_strategy.strategies.general import FixedSizeChunker

Orchestrator:
    from chunking_strategy import ChunkerOrchestrator
    orchestrator = ChunkerOrchestrator(config_path="config.yaml")
    chunks = orchestrator.chunk_file("document.pdf")

Streaming:
    from chunking_strategy import StreamingChunker
    chunker = StreamingChunker(strategy="fixed_size", chunk_size=1024)
    for chunk in chunker.stream_file("large_file.txt"):
        process(chunk)

Pipeline:
    from chunking_strategy import ChunkingPipeline
    pipeline = ChunkingPipeline([
        ("text_extraction", TextExtractor()),
        ("sentence_chunking", SentenceBasedChunker()),
        ("quality_filter", QualityFilter())
    ])
    chunks = pipeline.process("document.pdf")

Adaptive:
    from chunking_strategy import AdaptiveChunker
    chunker = AdaptiveChunker(base_strategy="semantic")
    chunks = chunker.chunk_with_feedback(text, feedback_score=0.8)
"""

from chunking_strategy.core.base import (
    BaseChunker,
    StreamableChunker,
    AdaptableChunker,
    HierarchicalChunker,
    ChunkingResult,
    ChunkMetadata,
    Chunk,
    ModalityType,
)
from chunking_strategy.core.registry import (
    ChunkerRegistry,
    register_chunker,
    get_chunker,
    list_chunkers,
    create_chunker,
    get_chunker_metadata,
    get_registry,
)
from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.core.streaming import StreamingChunker
from chunking_strategy.core.pipeline import ChunkingPipeline
from chunking_strategy.core.adaptive import AdaptiveChunker

# Import strategies to register them
from chunking_strategy.strategies.general import *  # noqa: F401,F403
from chunking_strategy.strategies.text import *  # noqa: F401,F403
from chunking_strategy.strategies.multimedia import *  # noqa: F401,F403  # noqa: F401,F403
from chunking_strategy.strategies.document import *  # noqa: F401,F403
from chunking_strategy.strategies.code import *  # noqa: F401,F403
from chunking_strategy.strategies.data_formats import *  # noqa: F401,F403

# Import universal framework
from chunking_strategy.core.universal_framework import (
    apply_universal_strategy,
    get_universal_strategy_registry,
    UniversalStrategy,
)
from chunking_strategy.core.universal_implementations import (
    UniversalFixedSizeStrategy,
    UniversalSentenceStrategy,
    UniversalParagraphStrategy,
    UniversalOverlappingWindowStrategy,
    UniversalRollingHashStrategy,
)

# Auto-register universal strategies
import chunking_strategy.core.universal_auto_registration  # noqa: F401
from chunking_strategy.core.extractors import (
    extract_content,
    get_extractor_registry
)

# Import embeddings functionality
from chunking_strategy.core.embeddings import (
    EmbeddingModel,
    OutputFormat,
    EmbeddingConfig,
    EmbeddedChunk,
    EmbeddingResult,
    create_embedder,
    embed_chunking_result,
    print_embedding_summary,
    export_for_vector_db,
)

# Import custom algorithms framework
from chunking_strategy.core.custom_algorithm_loader import (
    CustomAlgorithmLoader,
    CustomAlgorithmInfo,
    load_custom_algorithm,
    load_custom_algorithms_directory,
    list_custom_algorithms,
    get_custom_algorithm_info,
    get_custom_loader,
)
from chunking_strategy.core.custom_config_integration import (
    CustomConfigProcessor,
    load_config_with_custom_algorithms,
    validate_custom_config_file,
)
from chunking_strategy.core.custom_validation import (
    CustomAlgorithmValidator,
    ValidationReport,
    validate_custom_algorithm_file,
    run_comprehensive_validation,
    batch_validate_algorithms,
)

# Import logging functionality for easy access
from chunking_strategy.logging_config import (
    configure_logging,
    LogConfig,
    LogLevel,
    get_logger,
    enable_debug_mode,
    collect_debug_info,
    create_debug_archive,
    user_info,
    user_success,
    user_warning,
    user_error,
    debug_operation,
    performance_log,
    metrics_log
)

# Configure sensible defaults for Python import usage
# Users can override this by calling configure_logging() explicitly
try:
    configure_logging(
        level=LogLevel.NORMAL,  # Show user-friendly messages by default
        console_output=True,    # Output to console
        file_output=False,      # No file logging by default
        collect_performance=False,  # Don't collect performance by default
        collect_metrics=False   # Don't collect metrics by default
    )
except Exception:
    # If anything goes wrong with logging setup, fail silently
    # Library should still work even if logging setup fails
    pass

# Version info
__version__ = "0.1.0"
__author__ = "Chunking Strategy Team"
__email__ = " "

# Expose main components
__all__ = [
    # Core classes
    "BaseChunker",
    "StreamableChunker",
    "AdaptableChunker",
    "HierarchicalChunker",
    "Chunk",
    "ChunkMetadata",
    "ChunkingResult",
    "ModalityType",

    # Registry
    "ChunkerRegistry",
    "register_chunker",
    "get_chunker",
    "list_chunkers",
    "create_chunker",
    "get_chunker_metadata",
    "get_registry",

    # Main interfaces
    "ChunkerOrchestrator",
    "StreamingChunker",
    "ChunkingPipeline",
    "AdaptiveChunker",

    # Universal framework
    "apply_universal_strategy",
    "get_universal_strategy_registry",
    "extract_content",
    "get_extractor_registry",

    # Embeddings
    "EmbeddingModel",
    "OutputFormat",
    "EmbeddingConfig",
    "EmbeddedChunk",
    "EmbeddingResult",
    "create_embedder",
    "embed_chunking_result",
    "print_embedding_summary",
    "export_for_vector_db",

    # Custom Algorithms Framework
    "CustomAlgorithmLoader",
    "CustomAlgorithmInfo",
    "load_custom_algorithm",
    "load_custom_algorithms_directory",
    "list_custom_algorithms",
    "get_custom_algorithm_info",
    "get_custom_loader",
    "CustomConfigProcessor",
    "load_config_with_custom_algorithms",
    "validate_custom_config_file",
    "CustomAlgorithmValidator",
    "ValidationReport",
    "validate_custom_algorithm_file",
    "run_comprehensive_validation",
    "batch_validate_algorithms",

    # Logging and debugging
    "configure_logging",
    "LogConfig",
    "LogLevel",
    "get_logger",
    "enable_debug_mode",
    "collect_debug_info",
    "create_debug_archive",
    "user_info",
    "user_success",
    "user_warning",
    "user_error",
    "debug_operation",
    "performance_log",
    "metrics_log",

    # Version
    "__version__",
]
