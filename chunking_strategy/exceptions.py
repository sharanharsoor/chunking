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


class MissingExtraError(ChunkerError):
    """Raised when an optional extra is required. Message is the pip install line."""

    def __init__(self, extra: str, hint: str = None):
        self.extra = extra
        super().__init__(hint or f"pip install chunking-strategy[{extra}]")
