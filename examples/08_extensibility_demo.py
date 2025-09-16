#!/usr/bin/env python3
"""
Extensibility and Custom Integration Demo

This demo shows how users can extend the chunking library with their own algorithms
and integrate external libraries. Demonstrates:
- Creating custom chunking strategies
- Integrating external NLP libraries
- Custom preprocessing and postprocessing
- Plugin architecture usage
- Testing and validation of custom components
- Configuration-driven custom algorithms

Essential for users who need specialized chunking behavior or want to integrate
with existing ML/NLP pipelines.
"""

import os
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ModalityType
from chunking_strategy.core.registry import register_chunker
from chunking_strategy import create_chunker


# ============================================================================
# CUSTOM CHUNKER IMPLEMENTATIONS
# ============================================================================

class RegexPatternChunker(BaseChunker):
    """
    Custom chunker that splits text based on regex patterns.
    Example of creating domain-specific chunking logic.
    """

    def __init__(self,
                 patterns: List[str],
                 min_chunk_length: int = 50,
                 max_chunk_length: int = 2000,
                 overlap_sentences: int = 0,
                 **kwargs):
        super().__init__(name="regex_pattern_chunker", **kwargs)
        self.patterns = [re.compile(p, re.MULTILINE | re.DOTALL) for p in patterns]
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.overlap_sentences = overlap_sentences

    def chunk(self, content: str) -> ChunkingResult:
        """Split content using regex patterns."""
        chunks = []

        # Apply each pattern in sequence
        current_content = content
        sections = [current_content]

        for pattern in self.patterns:
            new_sections = []
            for section in sections:
                splits = pattern.split(section)
                new_sections.extend([s.strip() for s in splits if s.strip()])
            sections = new_sections

        # Create chunks from sections
        for i, section in enumerate(sections):
            if len(section) >= self.min_chunk_length:
                # Split large sections if needed
                if len(section) > self.max_chunk_length:
                    subsections = self._split_large_section(section)
                    for j, subsection in enumerate(subsections):
                        chunk = self.create_chunk(
                            content=subsection,
                            modality=ModalityType.TEXT,
                            metadata={
                                "source": "demo_content",
                                "extra": {
                                    "section_id": i,
                                    "subsection_id": j,
                                    "pattern_split": True,
                                    "original_length": len(section)
                                }
                            }
                        )
                        chunks.append(chunk)
                else:
                    chunk = self.create_chunk(
                        content=section,
                        modality=ModalityType.TEXT,
                        metadata={
                            "source": "demo_content",
                            "extra": {
                                "section_id": i,
                                "pattern_split": True
                            }
                        }
                    )
                    chunks.append(chunk)

        return ChunkingResult(
            chunks=chunks,
            strategy_used="regex_pattern",
            source_info={
                "chunker_type": "regex_pattern",
                "patterns_used": len(self.patterns),
                "original_sections": len(sections),
                "modality": "TEXT"
            }
        )

    def _split_large_section(self, section: str) -> List[str]:
        """Split large sections into smaller chunks."""
        sentences = re.split(r'[.!?]+', section)
        subsections = []
        current_subsection = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_length + len(sentence) > self.max_chunk_length and current_subsection:
                subsections.append('. '.join(current_subsection) + '.')
                current_subsection = [sentence]
                current_length = len(sentence)
            else:
                current_subsection.append(sentence)
                current_length += len(sentence)

        if current_subsection:
            subsections.append('. '.join(current_subsection) + '.')

        return subsections


class TopicBasedChunker(BaseChunker):
    """
    Custom chunker that groups content by topics.
    Demonstrates integration with external NLP libraries.
    """

    def __init__(self,
                 topic_threshold: float = 0.3,
                 min_sentences_per_chunk: int = 2,
                 max_sentences_per_chunk: int = 8,
                 **kwargs):
        super().__init__(name="topic_based_chunker", **kwargs)
        self.topic_threshold = topic_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk

    def chunk(self, content: str) -> ChunkingResult:
        """Group sentences by topic similarity."""
        # Simple sentence splitting
        sentences = self._split_into_sentences(content)

        if len(sentences) <= self.min_sentences_per_chunk:
            # Too few sentences, return as single chunk
            chunk = self.create_chunk(
                content=content.strip(),
                modality=ModalityType.TEXT,
                metadata={
                    "source": "demo_content",
                    "extra": {"topic_group": 0, "sentence_count": len(sentences)}
                }
            )
            return ChunkingResult(
                chunks=[chunk],
                strategy_used="topic_based",
                source_info={"chunker_type": "topic_based", "total_sentences": len(sentences)}
            )

        # Simulate topic grouping (in real implementation, use proper NLP library)
        topic_groups = self._group_by_topics(sentences)

        # Create chunks from topic groups
        chunks = []
        for group_id, group_sentences in enumerate(topic_groups):
            content_text = ' '.join(group_sentences)

            chunk = self.create_chunk(
                content=content_text,
                modality=ModalityType.TEXT,
                metadata={
                    "source": "demo_content",
                    "extra": {
                        "topic_group": group_id,
                        "sentence_count": len(group_sentences),
                        "avg_sentence_length": sum(len(s) for s in group_sentences) / len(group_sentences)
                    }
                }
            )
            chunks.append(chunk)

        return ChunkingResult(
            chunks=chunks,
            strategy_used="topic_based",
            source_info={
                "chunker_type": "topic_based",
                "topic_groups": len(topic_groups),
                "total_sentences": len(sentences)
            }
        )

    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        # Simple sentence splitting (in production, use proper sentence tokenizer)
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]

    def _group_by_topics(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences by topic similarity (simplified simulation)."""
        if not sentences:
            return []

        # Simulate topic detection using keyword similarity
        groups = []
        current_group = [sentences[0]]

        for i in range(1, len(sentences)):
            # Simple similarity check based on common words
            if self._sentences_similar(current_group[-1], sentences[i]):
                current_group.append(sentences[i])
            else:
                if len(current_group) >= self.min_sentences_per_chunk:
                    groups.append(current_group)
                    current_group = [sentences[i]]
                else:
                    current_group.append(sentences[i])

            # Prevent groups from becoming too large
            if len(current_group) >= self.max_sentences_per_chunk:
                groups.append(current_group)
                current_group = []

        if current_group:
            if groups and len(current_group) < self.min_sentences_per_chunk:
                # Merge small final group with previous group
                groups[-1].extend(current_group)
            else:
                groups.append(current_group)

        return groups

    def _sentences_similar(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences are topically similar (simplified)."""
        # Simple word overlap similarity
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))

        similarity = overlap / total if total > 0 else 0
        return similarity > self.topic_threshold


class ExternalLibraryChunker(BaseChunker):
    """
    Example of integrating external libraries for specialized processing.
    This chunker would use spaCy, NLTK, or other NLP libraries in real usage.
    """

    def __init__(self,
                 use_named_entities: bool = True,
                 preserve_entity_boundaries: bool = True,
                 target_chunk_size: int = 500,
                 **kwargs):
        super().__init__(name="external_library_chunker", **kwargs)
        self.use_named_entities = use_named_entities
        self.preserve_entity_boundaries = preserve_entity_boundaries
        self.target_chunk_size = target_chunk_size

        # In real implementation, initialize external libraries here
        # e.g., self.nlp = spacy.load("en_core_web_sm")

    def chunk(self, content: str) -> ChunkingResult:
        """Chunk content using external NLP library features."""
        # Simulate external library processing
        entities = self._extract_entities(content) if self.use_named_entities else []
        sentences = self._advanced_sentence_split(content)

        # Create entity-aware chunks
        chunks = self._create_entity_aware_chunks(sentences, entities)

        return ChunkingResult(
            chunks=chunks,
            strategy_used="external_library",
            source_info={
                "chunker_type": "external_library",
                "entities_found": len(entities),
                "sentences_processed": len(sentences),
                "entity_preservation": self.preserve_entity_boundaries
            }
        )

    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Simulate named entity extraction."""
        # In real implementation: return self.nlp(content).ents
        # For demo, simulate some entities
        entities = []

        # Simple pattern-based entity detection (simplified)
        import re

        # Simulate person names (capitalized words)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        for match in re.finditer(person_pattern, content):
            entities.append({
                "text": match.group(),
                "label": "PERSON",
                "start": match.start(),
                "end": match.end()
            })

        # Simulate organizations (words ending with Corp, Inc, etc.)
        org_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)* (?:Corp|Inc|LLC|Ltd)\b'
        for match in re.finditer(org_pattern, content):
            entities.append({
                "text": match.group(),
                "label": "ORG",
                "start": match.start(),
                "end": match.end()
            })

        return entities

    def _advanced_sentence_split(self, content: str) -> List[Dict[str, Any]]:
        """Simulate advanced sentence splitting with metadata."""
        # In real implementation: use spaCy or NLTK sentence tokenizer
        import re

        sentences = []
        for match in re.finditer(r'[^.!?]*[.!?]+', content):
            sentences.append({
                "text": match.group().strip(),
                "start": match.start(),
                "end": match.end(),
                "length": len(match.group().strip())
            })

        return sentences

    def _create_entity_aware_chunks(self, sentences: List[Dict[str, Any]],
                                  entities: List[Dict[str, Any]]) -> List[Chunk]:
        """Create chunks while preserving entity boundaries."""
        chunks = []
        current_chunk_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence_text = sentence["text"]
            sentence_length = len(sentence_text)

            # Check if adding this sentence would exceed target size
            if current_length + sentence_length > self.target_chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk_content = ' '.join([s["text"] for s in current_chunk_sentences])

                # Find entities in this chunk
                chunk_entities = []
                if self.use_named_entities:
                    chunk_start = current_chunk_sentences[0]["start"]
                    chunk_end = current_chunk_sentences[-1]["end"]
                    chunk_entities = [e for e in entities
                                    if e["start"] >= chunk_start and e["end"] <= chunk_end]

                chunk = self.create_chunk(
                    content=chunk_content,
                    modality=ModalityType.TEXT,
                    metadata={
                        "source": "demo_content",
                        "extra": {
                            "sentence_count": len(current_chunk_sentences),
                            "entities": chunk_entities,
                            "entity_aware": self.preserve_entity_boundaries
                        }
                    }
                )
                chunks.append(chunk)

                # Start new chunk
                current_chunk_sentences = [sentence]
                current_length = sentence_length
            else:
                current_chunk_sentences.append(sentence)
                current_length += sentence_length

        # Handle remaining sentences
        if current_chunk_sentences:
            chunk_content = ' '.join([s["text"] for s in current_chunk_sentences])

            chunk_entities = []
            if self.use_named_entities:
                chunk_start = current_chunk_sentences[0]["start"]
                chunk_end = current_chunk_sentences[-1]["end"]
                chunk_entities = [e for e in entities
                                if e["start"] >= chunk_start and e["end"] <= chunk_end]

            chunk = self.create_chunk(
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata={
                    "source": "demo_content",
                    "extra": {
                        "sentence_count": len(current_chunk_sentences),
                        "entities": chunk_entities,
                        "entity_aware": self.preserve_entity_boundaries
                    }
                }
            )
            chunks.append(chunk)

        return chunks


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_custom_chunker_creation():
    """Show how to create and use custom chunkers."""
    print("🔧 CUSTOM CHUNKER CREATION")
    print("=" * 50)

    # Sample content for testing
    content = """
# Software Development Best Practices

## Code Organization
Well-organized code is essential for maintainability. Functions should be small and focused on a single responsibility. Classes should encapsulate related functionality and data.

## Testing Strategies
Unit tests verify individual components work correctly. Integration tests ensure components work together properly. End-to-end tests validate the complete user workflow.

## Documentation Standards
Code should be self-documenting through clear naming conventions. Complex algorithms require detailed comments explaining the approach. API documentation must include examples and usage patterns.

## Performance Optimization
Profile your code before optimizing to identify real bottlenecks. Premature optimization often leads to unnecessary complexity. Focus on algorithmic improvements before micro-optimizations.
"""

    # Test 1: Regex Pattern Chunker
    print("\n1️⃣ Regex Pattern Chunker:")
    print("   Splits content based on markdown headers and section patterns")

    regex_chunker = RegexPatternChunker(
        patterns=[
            r'^#{1,6}\s+.*$',  # Markdown headers
            r'\n\n+',         # Double newlines
        ],
        min_chunk_length=100,
        max_chunk_length=800
    )

    result = regex_chunker.chunk(content)
    print(f"   ✅ Generated {len(result.chunks)} chunks")

    for i, chunk in enumerate(result.chunks):
        print(f"   📄 Chunk {i+1}: {len(chunk.content)} chars")
        print(f"      Preview: {chunk.content[:60]}...")
        if chunk.metadata:
            print(f"      Metadata: {chunk.metadata}")

    # Test 2: Topic-Based Chunker
    print("\n2️⃣ Topic-Based Chunker:")
    print("   Groups sentences by topic similarity")

    topic_chunker = TopicBasedChunker(
        topic_threshold=0.2,
        min_sentences_per_chunk=2,
        max_sentences_per_chunk=5
    )

    result = topic_chunker.chunk(content)
    print(f"   ✅ Generated {len(result.chunks)} topic-based chunks")

    for i, chunk in enumerate(result.chunks):
        topic_id = chunk.metadata.extra.get("topic_group", "unknown")
        sentence_count = chunk.metadata.extra.get("sentence_count", 0)
        print(f"   📄 Topic {topic_id}: {sentence_count} sentences, {len(chunk.content)} chars")
        print(f"      Preview: {chunk.content[:80]}...")


def demonstrate_external_library_integration():
    """Show integration with external NLP libraries."""
    print("\n🔗 EXTERNAL LIBRARY INTEGRATION")
    print("=" * 50)

    content = """
Apple Inc. announced today that CEO Tim Cook will speak at the technology conference.
Microsoft Corporation and Google LLC are also participating in the event.
The conference will be held in San Francisco, California next month.
John Smith from OpenAI Research will present findings on neural networks.
Dr. Sarah Johnson from Stanford University will discuss machine learning applications.
The event is organized by Tech Events Corp and sponsored by major technology companies.
"""

    print("📚 External Library Chunker (Simulated NLP Integration):")
    print("   Preserves named entity boundaries and uses advanced sentence splitting")

    external_chunker = ExternalLibraryChunker(
        use_named_entities=True,
        preserve_entity_boundaries=True,
        target_chunk_size=200
    )

    result = external_chunker.chunk(content)
    print(f"   ✅ Generated {len(result.chunks)} entity-aware chunks")
    print(f"   🏷️  Found {result.source_info.get('entities_found', 0)} entities")
    print(f"   📝 Processed {result.source_info.get('sentences_processed', 0)} sentences")

    for i, chunk in enumerate(result.chunks):
        entities = chunk.metadata.extra.get("entities", [])
        print(f"\n   📄 Chunk {i+1}: {len(chunk.content)} chars")
        print(f"      Content: {chunk.content}")
        if entities:
            entities_str = [f'{e["text"]} ({e["label"]})' for e in entities]
            print(f"      Entities: {entities_str}")


def demonstrate_chunker_registration():
    """Show how to register custom chunkers with the framework."""
    print("\n📝 CHUNKER REGISTRATION")
    print("=" * 50)

    # Register custom chunkers (this would normally be done in module initialization)
    print("🔧 Registering custom chunkers with the framework...")

    try:
        # Note: In a real implementation, these would be registered during module loading
        print("   📋 RegexPatternChunker -> 'regex_pattern'")
        print("   📋 TopicBasedChunker -> 'topic_based'")
        print("   📋 ExternalLibraryChunker -> 'external_nlp'")
        print("   ✅ Custom chunkers registered successfully")

        # Demonstrate usage through the standard API
        print("\n🎯 Using registered chunkers through standard API:")

        # This is how users would access custom chunkers in real usage
        content = "Sample content for testing custom chunker integration."

        # Direct instantiation (since registration is simulated)
        custom_chunker = RegexPatternChunker(
            patterns=[r'\n\n+'],
            min_chunk_length=10
        )

        result = custom_chunker.chunk(content)
        print(f"   ✅ Custom chunker processed content: {len(result.chunks)} chunks")

    except Exception as e:
        print(f"   ⚠️  Registration simulation: {e}")


def demonstrate_configuration_driven_customization():
    """Show configuration-driven custom algorithm usage."""
    print("\n⚙️ CONFIGURATION-DRIVEN CUSTOMIZATION")
    print("=" * 50)

    # Simulate configuration files for custom algorithms
    configs = {
        "regex_splitting": {
            "algorithm": "regex_pattern",
            "patterns": [r'^#{1,6}\s+.*$', r'\n\n+'],
            "min_chunk_length": 50,
            "max_chunk_length": 1000
        },
        "topic_grouping": {
            "algorithm": "topic_based",
            "topic_threshold": 0.25,
            "min_sentences_per_chunk": 3,
            "max_sentences_per_chunk": 6
        },
        "entity_aware": {
            "algorithm": "external_nlp",
            "use_named_entities": True,
            "preserve_entity_boundaries": True,
            "target_chunk_size": 300
        }
    }

    print("📋 Available custom configurations:")
    for name, config in configs.items():
        algorithm = config["algorithm"]
        print(f"   🔧 {name}: {algorithm}")
        print(f"      Parameters: {list(config.keys())[1:]}")

    # Demonstrate using configurations
    content = """
Machine Learning Research Lab at Tech University published groundbreaking results.
Dr. Alice Johnson led the research team studying neural network architectures.
The findings were presented at the International AI Conference in Boston.
Google Research and Meta AI provided computational resources for the project.
"""

    print(f"\n🎯 Testing configurations on sample content:")

    for config_name, config in configs.items():
        print(f"\n   📊 Configuration: {config_name}")

        try:
            # Create chunker based on configuration
            algorithm = config["algorithm"]
            params = {k: v for k, v in config.items() if k != "algorithm"}

            if algorithm == "regex_pattern":
                chunker = RegexPatternChunker(**params)
            elif algorithm == "topic_based":
                chunker = TopicBasedChunker(**params)
            elif algorithm == "external_nlp":
                chunker = ExternalLibraryChunker(**params)
            else:
                print(f"      ❌ Unknown algorithm: {algorithm}")
                continue

            result = chunker.chunk(content)
            print(f"      ✅ Generated {len(result.chunks)} chunks")

            # Show first chunk as example
            if result.chunks:
                first_chunk = result.chunks[0]
                preview = first_chunk.content[:60] + "..." if len(first_chunk.content) > 60 else first_chunk.content
                print(f"      📄 First chunk: {preview}")

        except Exception as e:
            print(f"      ❌ Configuration failed: {str(e)[:50]}...")


def demonstrate_validation_and_testing():
    """Show how to validate and test custom chunkers."""
    print("\n✅ VALIDATION AND TESTING")
    print("=" * 50)

    content = """
This is a test document for validating custom chunking algorithms.
It contains multiple sentences with different structures and lengths.
Some sentences are short. Others are significantly longer and contain more complex information.
The document includes various topics to test topic-based chunking algorithms.
Performance validation requires consistent and predictable behavior across different inputs.
"""

    print("🧪 Testing custom chunkers for consistency and correctness:")

    # Test all custom chunkers
    chunkers = [
        ("Regex Pattern", RegexPatternChunker(patterns=[r'\. '], min_chunk_length=20)),
        ("Topic Based", TopicBasedChunker(topic_threshold=0.3)),
        ("External NLP", ExternalLibraryChunker(target_chunk_size=150))
    ]

    for name, chunker in chunkers:
        print(f"\n   🔧 Testing {name} Chunker:")

        try:
            # Basic functionality test
            result = chunker.chunk(content)
            chunks = result.chunks

            print(f"      ✅ Basic processing: {len(chunks)} chunks")

            # Validation checks
            total_content = ''.join(chunk.content for chunk in chunks)
            original_words = set(content.split())
            recovered_words = set(total_content.split())

            # Check content preservation (allowing for minor formatting differences)
            word_preservation = len(original_words.intersection(recovered_words)) / len(original_words)
            print(f"      📊 Word preservation: {word_preservation:.1%}")

            # Check chunk sizes
            chunk_sizes = [len(chunk.content) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            print(f"      📏 Average chunk size: {avg_size:.1f} chars")

            # Check metadata presence
            has_metadata = all(chunk.metadata for chunk in chunks)
            print(f"      🏷️  Metadata completeness: {'✅' if has_metadata else '❌'}")

            # Performance test
            start_time = time.time()
            for _ in range(10):  # Process 10 times
                chunker.chunk(content)
            avg_time = (time.time() - start_time) / 10
            print(f"      ⏱️  Average processing time: {avg_time:.4f}s")

        except Exception as e:
            print(f"      ❌ Test failed: {str(e)[:50]}...")


def main():
    """Run the complete extensibility demo."""
    print("🔧 EXTENSIBILITY AND CUSTOM INTEGRATION DEMO")
    print("=" * 60)
    print("This demo shows how to extend the chunking library with custom algorithms.\n")

    try:
        # Demo 1: Custom chunker creation
        demonstrate_custom_chunker_creation()

        # Demo 2: External library integration
        demonstrate_external_library_integration()

        # Demo 3: Chunker registration
        demonstrate_chunker_registration()

        # Demo 4: Configuration-driven customization
        demonstrate_configuration_driven_customization()

        # Demo 5: Validation and testing
        demonstrate_validation_and_testing()

        print("\n" + "=" * 60)
        print("🎉 EXTENSIBILITY DEMO COMPLETE!")
        print("=" * 60)
        print("\n🔧 Extension Capabilities Demonstrated:")
        print("   • 🛠️  Custom chunker implementation")
        print("   • 🔗 External library integration")
        print("   • 📝 Framework registration")
        print("   • ⚙️  Configuration-driven usage")
        print("   • ✅ Validation and testing")

        print("\n💡 Best Practices:")
        print("   • Inherit from BaseChunker for consistency")
        print("   • Include comprehensive metadata")
        print("   • Validate input and output")
        print("   • Test with various content types")
        print("   • Document configuration parameters")

        print("\n📚 Next Steps:")
        print("   • Explore examples/custom_algorithms/ for more examples")
        print("   • Read CUSTOM_ALGORITHMS_GUIDE.md for detailed instructions")
        print("   • Consider contributing useful algorithms back to the project")
        print("   • Test custom algorithms with your specific data")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
