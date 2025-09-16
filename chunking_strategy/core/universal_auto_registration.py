"""
Auto-registration for universal strategies.

This module automatically registers all universal strategy implementations
when imported.
"""

from chunking_strategy.core.universal_framework import get_universal_strategy_registry
from chunking_strategy.core.universal_implementations import (
    UniversalFixedSizeStrategy,
    UniversalSentenceStrategy,
    UniversalParagraphStrategy,
    UniversalOverlappingWindowStrategy,
    UniversalRollingHashStrategy,
)

# Get the global registry
_registry = get_universal_strategy_registry()

# Register all universal strategies
_registry.register(UniversalFixedSizeStrategy())
_registry.register(UniversalSentenceStrategy())  
_registry.register(UniversalParagraphStrategy())
_registry.register(UniversalOverlappingWindowStrategy())
_registry.register(UniversalRollingHashStrategy())
