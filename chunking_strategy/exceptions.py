"""
Exception classes for the chunking strategy library.
"""


class ChunkerError(Exception):
    """Base exception for all chunker-related errors."""
    pass


class ChunkerNotFoundError(ChunkerError):
    """Raised when a requested chunker strategy is not found."""
    pass


class ChunkingConfigurationError(ChunkerError):
    """Raised when chunker configuration is invalid."""
    pass


class ChunkingProcessingError(ChunkerError):
    """Raised when chunking processing fails."""
    pass


class InvalidContentError(ChunkerError):
    """Raised when input content is invalid for chunking."""
    pass


class StrategyUnavailableError(ChunkerError):
    """Raised when a strategy is temporarily unavailable (e.g., missing dependencies)."""
    pass
