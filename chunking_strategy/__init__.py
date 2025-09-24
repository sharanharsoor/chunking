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

# Suppress annoying warnings from heavy dependencies early
import os
import warnings
import logging
import sys
import contextlib
from io import StringIO

# Set environment variables before any heavy imports
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',           # Suppress TF INFO/WARNING/ERROR
    'TF_ENABLE_ONEDNN_OPTS': '0',          # Disable oneDNN messages
    'PYTHONWARNINGS': 'ignore::UserWarning', # Suppress user warnings globally
    'TOKENIZERS_PARALLELISM': 'false',     # Avoid tokenizer warnings
    'TRANSFORMERS_VERBOSITY': 'error',     # Only show errors from transformers
    'GRPC_VERBOSITY': 'ERROR',            # Suppress gRPC/protobuf messages
    'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',  # Use pure Python protobuf
})

# Comprehensive warning suppression
warnings.filterwarnings('ignore')  # Suppress ALL warnings initially
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*GetPrototype.*')
warnings.filterwarnings('ignore', message='.*cuDNN.*')
warnings.filterwarnings('ignore', message='.*cuFFT.*')
warnings.filterwarnings('ignore', message='.*cuBLAS.*')
warnings.filterwarnings('ignore', message='.*8-bit optimizer.*')
warnings.filterwarnings('ignore', message='.*NumExpr.*')
warnings.filterwarnings('ignore', message='.*MessageFactory.*')

# Also suppress specific loggers that are noisy
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Custom context manager to suppress stderr AttributeError messages
@contextlib.contextmanager
def suppress_protobuf_errors():
    """Suppress protobuf/gRPC AttributeError messages that don't go through warnings."""
    original_stderr = sys.stderr
    captured_stderr = StringIO()

    class FilteredStderr:
        def write(self, text):
            # Filter out specific error patterns
            if not any(pattern in text for pattern in [
                "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'",
                "GetPrototype",
                "MessageFactory"
            ]):
                original_stderr.write(text)

        def flush(self):
            original_stderr.flush()

        def __getattr__(self, name):
            return getattr(original_stderr, name)

    try:
        sys.stderr = FilteredStderr()
        yield
    finally:
        sys.stderr = original_stderr

# Custom exception handler to suppress specific AttributeErrors
_original_excepthook = sys.excepthook

def _custom_excepthook(exc_type, exc_value, exc_traceback):
    """Custom exception handler to suppress protobuf AttributeErrors."""
    if (exc_type == AttributeError and
        exc_value and
        "GetPrototype" in str(exc_value)):
        # Silently ignore these specific protobuf errors
        return
    # Otherwise, use the original exception handler
    _original_excepthook(exc_type, exc_value, exc_traceback)

# Install the custom exception handler
sys.excepthook = _custom_excepthook

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
    get_chunker_metadata,
    get_registry,
)
# Note: list_chunkers and create_chunker are defined with lazy loading below
from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.core.streaming import StreamingChunker
from chunking_strategy.core.pipeline import ChunkingPipeline
from chunking_strategy.core.adaptive import AdaptiveChunker

# Lazy loading of strategies to improve import time
# Strategies are registered when first requested rather than at import time
_STRATEGY_MODULES_LOADED = False
_LIGHT_STRATEGIES_LOADED = False

def _ensure_light_strategies_loaded():
    """Load only lightweight strategies first."""
    global _LIGHT_STRATEGIES_LOADED
    if not _LIGHT_STRATEGIES_LOADED:
        try:
            # Import lightweight strategy modules only
            import chunking_strategy.strategies.general  # noqa: F401
            import chunking_strategy.strategies.text  # noqa: F401
            import chunking_strategy.strategies.code  # noqa: F401
            import chunking_strategy.strategies.data_formats  # noqa: F401
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Light strategies failed to load: {e}")
        finally:
            _LIGHT_STRATEGIES_LOADED = True

def _ensure_strategies_loaded():
    """Lazy load all strategy modules including heavy ones."""
    global _STRATEGY_MODULES_LOADED
    if not _STRATEGY_MODULES_LOADED:
        # First ensure light strategies are loaded
        if not _LIGHT_STRATEGIES_LOADED:
            _ensure_light_strategies_loaded()

        # Then load heavy strategies
        try:
            _load_heavy_strategies()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Heavy strategies failed to load: {e}")
        finally:
            _STRATEGY_MODULES_LOADED = True

def _load_heavy_strategies():
    """Load strategies with heavy dependencies separately."""
    # Load heavy text strategies (semantic, embedding-based) with error suppression
    with suppress_protobuf_errors():
        try:
            import chunking_strategy.strategies.text.semantic_chunker  # noqa: F401
            import chunking_strategy.strategies.text.embedding_based_chunker  # noqa: F401
        except ImportError:
            pass  # Heavy text strategies have optional ML dependencies

        try:
            import chunking_strategy.strategies.document  # noqa: F401
        except ImportError:
            pass  # Document strategies have optional dependencies

        try:
            import chunking_strategy.strategies.multimedia  # noqa: F401
        except ImportError:
            pass  # Multimedia strategies have optional dependencies

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
# Lazy import extractors to avoid heavy PDF dependencies
def _get_extractor_classes():
    """Lazy load extractor functionality to avoid heavy PDF/ML imports."""
    from chunking_strategy.core.extractors import (
        extract_content,
        get_extractor_registry
    )
    return {
        'extract_content': extract_content,
        'get_extractor_registry': get_extractor_registry,
    }

# Lazy import embeddings functionality (heavy dependencies)
def _get_embedding_classes():
    """Lazy load expensive embedding functionality."""
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
    return {
        'EmbeddingModel': EmbeddingModel,
        'OutputFormat': OutputFormat,
        'EmbeddingConfig': EmbeddingConfig,
        'EmbeddedChunk': EmbeddedChunk,
        'EmbeddingResult': EmbeddingResult,
        'create_embedder': create_embedder,
        'embed_chunking_result': embed_chunking_result,
        'print_embedding_summary': print_embedding_summary,
        'export_for_vector_db': export_for_vector_db,
    }

# Global caches for lazy imports
_embedding_globals = None
_extractor_globals = None

def __getattr__(name):
    """Lazy load expensive modules only when accessed."""
    global _embedding_globals, _extractor_globals

    # Check embedding classes first
    if _embedding_globals is None:
        _embedding_globals = _get_embedding_classes()
    if name in _embedding_globals:
        return _embedding_globals[name]

    # Check extractor classes
    if _extractor_globals is None:
        _extractor_globals = _get_extractor_classes()
    if name in _extractor_globals:
        return _extractor_globals[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

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

    # Strategy listing functions
    "list_chunkers",
    "list_strategies",  # Alias for list_chunkers
]

# Add convenient alias for strategy listing with lazy loading
def list_chunkers(include_heavy=False):
    """List all available chunking strategies, with lazy loading.

    Args:
        include_heavy (bool): If True, load heavy strategies with ML dependencies.
                             If False (default), only load lightweight strategies.
    """
    with suppress_protobuf_errors():
        if include_heavy:
            _ensure_strategies_loaded()  # Load everything including heavy strategies
        else:
            # Only load light strategies for fast listing
            if not _LIGHT_STRATEGIES_LOADED:
                _ensure_light_strategies_loaded()

        from chunking_strategy.core.registry import list_chunkers as _list_chunkers
        return _list_chunkers()

def create_chunker(name: str, **kwargs):
    """Create a chunker instance, with lazy loading."""
    _ensure_strategies_loaded()
    with suppress_protobuf_errors():
        from chunking_strategy.core.registry import create_chunker as _create_chunker
        try:
            return _create_chunker(name, **kwargs)
        except Exception as e:
            # Re-raise all exceptions (no more silent None returns)
            raise

list_strategies = list_chunkers  # Alias for backward compatibility

# Add chunker name aliases to fix naming inconsistencies

# Store reference to original function and replace with alias-aware version
_original_create_chunker = create_chunker
def create_chunker(name: str, **kwargs):
    """Create a chunker instance with name aliases and lazy loading."""
    return _create_chunker_with_aliases(name, **kwargs)

# Fix the reference in the alias function
def _create_chunker_with_aliases(name: str, **kwargs):
    """Create chunker with name aliases for backward compatibility."""

    # Define name mappings for better UX
    name_aliases = {
        'semantic_chunker': 'semantic',
        'json_structured': 'json_chunker',
        'adaptive_chunker': 'adaptive',
        'sentence_chunker': 'sentence_based',
        'paragraph_chunker': 'paragraph_based',
        'fixed_word': 'fixed_length_word',
        'fixed_char': 'fixed_size',
    }

    with suppress_protobuf_errors():
        try:
            return _original_create_chunker(name, **kwargs)
        except (KeyError, ImportError):
            # Try with alias
            if name in name_aliases:
                return _original_create_chunker(name_aliases[name], **kwargs)
            # Try reverse lookup
            for alias, real_name in name_aliases.items():
                if real_name == name:
                    return _original_create_chunker(alias, **kwargs)
            # Re-raise original error if no alias found
            raise

# Import exceptions for external use
from chunking_strategy.exceptions import (
    ChunkerError,
    ChunkerNotFoundError,
    ChunkingConfigurationError,
    ChunkingProcessingError,
    InvalidContentError,
    StrategyUnavailableError
)
