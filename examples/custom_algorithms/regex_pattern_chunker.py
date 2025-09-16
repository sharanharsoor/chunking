#!/usr/bin/env python3
"""
Regex Pattern-Based Chunker Example

This is a simple example of a custom chunking algorithm that splits content
based on regular expression patterns. Used for demonstration purposes in configuration files.
"""

import re
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType


class RegexPatternChunker(BaseChunker):
    """
    Simple regex pattern-based chunker for demonstration.
    Splits content based on configurable regex patterns.
    """

    def __init__(self,
                 name: str = "regex_pattern",
                 patterns: List[str] = None,
                 min_chunk_length: int = 50,
                 max_chunk_length: int = 2000,
                 **kwargs):
        super().__init__(
            name=name,
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        # Default patterns if none provided
        if patterns is None:
            patterns = [
                r'^#{1,6}\s+.*$',  # Markdown headers
                r'\n\n+',         # Double newlines (paragraph breaks)
                r'\n\s*[-*+]\s+', # List items
                r'\d+\.\s+',      # Numbered lists
            ]

        self.patterns = [re.compile(p, re.MULTILINE | re.DOTALL) for p in patterns]
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length

    def chunk(self, content: str) -> ChunkingResult:
        """Split content using regex patterns."""
        chunks = []

        # Apply each pattern in sequence to split content
        current_sections = [content]

        for pattern in self.patterns:
            new_sections = []
            for section in current_sections:
                # Split by pattern
                splits = pattern.split(section)
                # Keep non-empty splits
                for split in splits:
                    if split and split.strip():
                        new_sections.append(split.strip())

            if new_sections:  # Only update if we got meaningful splits
                current_sections = new_sections

        # Create chunks from sections
        for i, section in enumerate(current_sections):
            if len(section) >= self.min_chunk_length:
                # Split large sections if needed
                if len(section) > self.max_chunk_length:
                    subsections = self._split_large_section(section)
                    for j, subsection in enumerate(subsections):
                        chunk = Chunk(
                            id=f"regex_{i}_{j}",
                            content=subsection,
                            modality=ModalityType.TEXT,
                            metadata=ChunkMetadata(
                                source="unknown",
                                extra={
                                    "section_id": i,
                                    "subsection_id": j,
                                    "pattern_split": True,
                                    "original_length": len(section),
                                    "chunker_type": "regex_pattern"
                                }
                            )
                        )
                        chunks.append(chunk)
                else:
                    chunk = Chunk(
                        id=f"regex_{i}",
                        content=section,
                        modality=ModalityType.TEXT,
                        metadata=ChunkMetadata(
                            source="unknown",
                            extra={
                                "section_id": i,
                                "pattern_split": True,
                                "chunker_type": "regex_pattern"
                            }
                        )
                    )
                    chunks.append(chunk)

        # If no chunks were created, create a single chunk
        if not chunks:
            chunks.append(Chunk(
                id="regex_0",
                content=content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source="unknown",
                    extra={
                        "section_id": 0,
                        "pattern_split": False,
                        "chunker_type": "regex_pattern"
                    }
                )
            ))

        return ChunkingResult(
            chunks=chunks,
            strategy_used="regex_pattern",
            source_info={
                "chunker_type": "regex_pattern",
                "patterns_used": len(self.patterns),
                "original_sections": len(current_sections)
            }
        )

    def _split_large_section(self, section: str) -> List[str]:
        """Split large sections into smaller chunks by sentences."""
        sentences = re.split(r'[.!?]+', section)
        subsections = []
        current_subsection = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_length + len(sentence) > self.max_chunk_length and current_subsection:
                # Finish current subsection
                subsections.append('. '.join(current_subsection) + '.')
                current_subsection = [sentence]
                current_length = len(sentence)
            else:
                current_subsection.append(sentence)
                current_length += len(sentence)

        # Add final subsection
        if current_subsection:
            subsections.append('. '.join(current_subsection) + '.')

        return subsections if subsections else [section]


# Register the chunker for framework integration
def get_chunkers():
    """Return available chunkers from this module."""
    return {
        "regex_pattern_based": RegexPatternChunker
    }

# Create a class with the exact name expected by the configuration
# The custom algorithm loader looks for BaseChunker subclasses by their class name
class regex_pattern_based(RegexPatternChunker):
    """Alias class for configuration compatibility."""
    pass


if __name__ == "__main__":
    # Demo usage
    chunker = RegexPatternChunker()

    test_content = """
# Main Title

This is an introduction paragraph with some content.
It has multiple sentences for testing.

## Section 1

This is the first section with detailed information.
- First bullet point
- Second bullet point
- Third bullet point

1. First numbered item
2. Second numbered item
3. Third numbered item

## Section 2

This is the second section with more content.
It demonstrates different text structures.

### Subsection 2.1

More detailed content in a subsection.
This shows hierarchical document processing.
"""

    result = chunker.chunk(test_content)

    print("Regex Pattern-Based Chunking Demo:")
    print(f"Generated {len(result.chunks)} chunks")

    for i, chunk in enumerate(result.chunks):
        section_id = chunk.metadata.extra.get("section_id", "unknown")
        pattern_split = chunk.metadata.extra.get("pattern_split", False)
        print(f"\nChunk {i+1} (section {section_id}, pattern split: {pattern_split}):")
        print(f"  {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}")