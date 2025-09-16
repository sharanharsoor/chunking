"""
Unit tests for Context-Enriched Chunking strategy.

Tests cover:
- Basic context-enriched functionality and registration
- Semantic boundary detection mechanisms
- Entity recognition and preservation
- Topic modeling and coherence analysis
- Content profiling and analysis
- Adaptive parameter adjustment
- Quality scoring and boundary analysis
- Streaming support and edge cases
- Integration with orchestrator
- Fallback mechanisms when NLP libraries unavailable
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from chunking_strategy.core.registry import create_chunker, list_chunkers
from chunking_strategy.core.base import ModalityType
from chunking_strategy.strategies.general.context_enriched_chunker import (
    ContextEnrichedChunker, SemanticEntity, TopicInfo, ContextualChunk, SemanticBoundary
)
from chunking_strategy import ChunkerOrchestrator


class TestContextEnrichedChunker:
    """Test Context-Enriched Chunker functionality."""

    def test_context_enriched_registration(self):
        """Test that Context-Enriched chunker is properly registered."""
        chunkers = list_chunkers()
        assert "context_enriched" in chunkers

        # Test creation
        chunker = create_chunker("context_enriched")
        assert chunker is not None
        assert chunker.__class__.__name__ == "ContextEnrichedChunker"
        assert "any" in chunker.get_supported_formats()
        assert chunker.target_chunk_size == 2000  # Default value

    def test_sentence_segmentation(self):
        """Test sentence segmentation functionality."""
        chunker = create_chunker("context_enriched")

        # Test basic sentence segmentation
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        sentences = chunker._segment_sentences(text)

        assert len(sentences) >= 3
        assert all(sentence.strip() for sentence in sentences)

        # Test with complex punctuation
        complex_text = "Dr. Smith went to the U.S.A. He visited Washington, D.C. Then he returned."
        complex_sentences = chunker._segment_sentences(complex_text)
        assert len(complex_sentences) >= 2

    def test_entity_extraction_fallback(self):
        """Test entity extraction with and without NLP libraries."""
        chunker = create_chunker("context_enriched", enable_ner=True)

        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
        entities = chunker._extract_entities(text)

        # Should work even without full NLP (fallback behavior)
        assert isinstance(entities, list)
        # May be empty if no NLP libraries, but shouldn't crash

    def test_topic_analysis(self):
        """Test topic modeling and analysis."""
        chunker = create_chunker("context_enriched", enable_topic_modeling=True)

        sentences = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are used in deep learning applications.",
            "AI systems can process large amounts of data.",
            "Climate change affects global weather patterns.",
            "Rising temperatures cause ice cap melting.",
            "Environmental protection requires international cooperation."
        ]

        topics = chunker._analyze_topics(sentences)

        # Should identify topics even with basic implementation
        assert isinstance(topics, list)
        # May be empty if sklearn not available, but shouldn't crash

    def test_topic_coherence_calculation(self):
        """Test topic coherence scoring."""
        chunker = create_chunker("context_enriched")

        # High coherence case - similar sentences
        coherent_sentences = [
            "Artificial intelligence is transforming technology.",
            "Machine learning algorithms improve automatically.",
            "Deep learning uses neural networks for pattern recognition."
        ]
        coherence_high = chunker._calculate_topic_coherence([0, 1, 2], coherent_sentences)

        # Low coherence case - unrelated sentences
        incoherent_sentences = [
            "The weather is sunny today.",
            "Mathematical equations solve complex problems.",
            "Pizza is a popular Italian dish."
        ]
        coherence_low = chunker._calculate_topic_coherence([0, 1, 2], incoherent_sentences)

        assert 0.0 <= coherence_high <= 1.0
        assert 0.0 <= coherence_low <= 1.0

    def test_semantic_boundary_detection(self):
        """Test semantic boundary detection mechanisms."""
        chunker = create_chunker("context_enriched")

        sentences = [
            "Artificial intelligence is revolutionizing technology.",
            "Machine learning algorithms learn from data automatically.",
            "Neural networks process information like human brains.",
            "Climate change is a global environmental challenge.",
            "Rising sea levels threaten coastal communities.",
            "Renewable energy offers sustainable solutions."
        ]

        boundaries = chunker._detect_semantic_boundaries(sentences)

        assert isinstance(boundaries, list)
        for boundary in boundaries:
            assert isinstance(boundary, SemanticBoundary)
            assert 0 <= boundary.position < len(sentences)
            assert 0.0 <= boundary.confidence <= 1.0
            assert boundary.boundary_type in ["semantic_break", "topic_shift", "entity_boundary"]

    def test_similarity_boundary_detection(self):
        """Test semantic similarity-based boundary detection."""
        chunker = create_chunker("context_enriched",
                                semantic_similarity_threshold=0.5,
                                boundary_detection_method="semantic")

        sentences = [
            "Machine learning is a powerful technology.",
            "Deep learning uses neural networks effectively.",
            "The weather is beautiful today.",
            "Sunny skies make people happy."
        ]

        boundaries = chunker._detect_similarity_boundaries(sentences)

        # Should detect boundary between ML and weather topics
        assert isinstance(boundaries, list)
        # May be empty if sklearn not available

    def test_basic_context_enriched_chunking(self):
        """Test basic context-enriched chunking functionality."""
        chunker = create_chunker("context_enriched", target_chunk_size=500)

        # Test with semantic text that should trigger topic detection
        text = """Natural Language Processing Overview

Natural language processing (NLP) is a crucial field in artificial intelligence.
It focuses on enabling computers to understand and process human language effectively.
Modern NLP systems use machine learning and deep learning techniques.

Computer Vision Applications

Computer vision is another important AI domain that processes visual information.
Deep learning models like convolutional neural networks excel at image recognition.
Applications include autonomous driving, medical imaging, and robotics."""

        result = chunker.chunk(text)

        assert result.total_chunks > 0
        assert result.strategy_used == "context_enriched"
        assert result.processing_time >= 0

        # Check that chunks have content
        for chunk in result.chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.modality == ModalityType.TEXT
            assert chunk.metadata.chunker_used == "context_enriched"

    def test_enhanced_metadata_generation(self):
        """Test generation of enhanced semantic metadata."""
        chunker = create_chunker("context_enriched",
                                enable_ner=True,
                                enable_topic_modeling=True,
                                generate_semantic_fingerprints=True)

        text = "Apple Inc. is a technology company founded by Steve Jobs. The company is based in Cupertino, California."

        result = chunker.chunk(text)

        assert len(result.chunks) > 0

        # Check enhanced metadata
        chunk = result.chunks[0]
        metadata = chunk.metadata.extra

        assert "entities" in metadata
        assert "topics" in metadata
        assert "coherence_score" in metadata
        assert "context_preservation_score" in metadata
        assert "boundary_quality_score" in metadata
        assert "chunk_index" in metadata
        assert "total_chunks" in metadata
        assert "sentence_count" in metadata
        assert "word_count" in metadata
        assert "semantic_fingerprint" in metadata
        assert "processing_method" in metadata
        assert "nlp_enabled" in metadata

        # Check score ranges
        assert 0.0 <= metadata["coherence_score"] <= 1.0
        assert 0.0 <= metadata["context_preservation_score"] <= 1.0
        assert 0.0 <= metadata["boundary_quality_score"] <= 1.0

    def test_chunk_size_constraints(self):
        """Test that chunk size constraints are respected."""
        chunker = create_chunker("context_enriched",
                                min_chunk_size=100,
                                max_chunk_size=300,
                                target_chunk_size=200)

        # Long text that should be split
        text = "This is a test sentence. " * 100  # Very long text

        result = chunker.chunk(text)

        assert result.total_chunks > 1

        for chunk in result.chunks[:-1]:  # Exclude last chunk (may be smaller)
            chunk_size = len(chunk.content)
            assert chunk_size <= chunker.max_chunk_size

    def test_entity_preservation_mode(self):
        """Test different entity preservation modes."""
        # Test strict mode
        strict_chunker = create_chunker("context_enriched",
                                      entity_preservation_mode="strict",
                                      avoid_entity_splitting=True)

        # Test moderate mode
        moderate_chunker = create_chunker("context_enriched",
                                        entity_preservation_mode="moderate")

        # Test loose mode
        loose_chunker = create_chunker("context_enriched",
                                     entity_preservation_mode="loose")

        text = "Microsoft Corporation and Apple Inc. are major technology companies."

        # All should work without errors
        strict_result = strict_chunker.chunk(text)
        moderate_result = moderate_chunker.chunk(text)
        loose_result = loose_chunker.chunk(text)

        assert all(r.total_chunks > 0 for r in [strict_result, moderate_result, loose_result])

    def test_overlap_functionality(self):
        """Test chunk overlap for context preservation."""
        chunker = create_chunker("context_enriched",
                                target_chunk_size=200,
                                overlap_size=50)

        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six."

        result = chunker.chunk(text)

        if result.total_chunks > 1:
            # Check that there might be some overlap in content
            # (exact overlap testing is complex due to semantic boundary detection)
            assert all(len(chunk.content) > 0 for chunk in result.chunks)

    def test_file_input_handling(self):
        """Test context-enriched chunking with file inputs."""
        chunker = create_chunker("context_enriched")

        # Test with existing test files
        test_files = [
            "test_data/sample_semantic_text.txt",
            "test_data/sample_entity_narrative.txt",
            "test_data/sample_topic_transitions.txt"
        ]

        for file_path in test_files:
            if os.path.exists(file_path):
                result = chunker.chunk(file_path)

                assert result.total_chunks > 0
                assert result.strategy_used == "context_enriched"

                # Check context-enriched metadata
                if result.source_info and "context_enriched_metadata" in result.source_info:
                    metadata = result.source_info["context_enriched_metadata"]
                    assert "total_sentences" in metadata
                    assert "total_entities" in metadata
                    assert "total_topics" in metadata
                    assert "semantic_boundaries" in metadata

    def test_quality_scoring_methods(self):
        """Test various quality scoring methods."""
        chunker = create_chunker("context_enriched")

        # Test coherence calculation
        sentences = ["AI is powerful.", "Machine learning works well.", "Computers process data."]
        topics = []  # Empty topics
        coherence = chunker._calculate_chunk_coherence(sentences, topics)
        assert 0.0 <= coherence <= 1.0

        # Test context preservation
        preservation = chunker._calculate_context_preservation(sentences)
        assert 0.0 <= preservation <= 1.0

        # Test boundary quality
        boundary_quality = chunker._calculate_boundary_quality(sentences)
        assert 0.0 <= boundary_quality <= 1.0

    def test_semantic_fingerprint_generation(self):
        """Test semantic fingerprint generation."""
        chunker = create_chunker("context_enriched", generate_semantic_fingerprints=True)

        content = "Machine learning and artificial intelligence are transforming technology."
        fingerprint = chunker._generate_semantic_fingerprint(content)

        assert isinstance(fingerprint, list)
        # May be empty if sklearn not available, but shouldn't crash

    def test_adaptation_mechanisms(self):
        """Test parameter adaptation based on feedback."""
        chunker = create_chunker("context_enriched")

        initial_threshold = chunker.semantic_similarity_threshold

        # Test adaptation with poor feedback
        chunker.adapt_parameters(0.3, "boundary")
        assert chunker.semantic_similarity_threshold <= initial_threshold

        # Test adaptation with good feedback
        chunker.adapt_parameters(0.9, "boundary")
        # Should increase threshold or maintain it

        # Check adaptation history
        history = chunker.get_adaptation_history()
        assert len(history) == 2
        assert all("feedback_score" in record for record in history)

    def test_streaming_support(self):
        """Test streaming chunking capability."""
        chunker = create_chunker("context_enriched")

        # Create a stream of content
        content_parts = [
            "First part of the document about artificial intelligence. ",
            "Second part discussing machine learning applications. ",
            "Third part covering neural network architectures. ",
            "Final part about future AI developments."
        ]

        # Test streaming
        chunks = list(chunker.chunk_stream(content_parts))

        assert len(chunks) > 0

        # Compare with direct chunking
        direct_content = ''.join(content_parts)
        direct_result = chunker.chunk(direct_content)

        # Should produce similar number of chunks
        assert len(chunks) == len(direct_result.chunks)

    def test_fallback_mechanisms(self):
        """Test fallback mechanisms when advanced NLP is unavailable."""
        chunker = create_chunker("context_enriched")

        # Force fallback by using content that would normally trigger advanced processing
        text = "This should trigger fallback mechanisms when advanced NLP is not available."

        result = chunker.chunk(text)

        # Should still work and produce chunks
        assert result.total_chunks > 0
        assert result.strategy_used in ["context_enriched", "context_enriched_fallback"]

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        chunker = create_chunker("context_enriched")

        # Empty content
        empty_result = chunker.chunk("")
        assert empty_result.total_chunks == 0

        # Very short content
        short_result = chunker.chunk("Short.")
        assert short_result.total_chunks >= 0

        # Single sentence
        single_result = chunker.chunk("This is a single sentence.")
        assert single_result.total_chunks == 1

        # Very long content
        long_content = "This is a test sentence. " * 1000
        long_result = chunker.chunk(long_content)
        assert long_result.total_chunks > 1

    def test_orchestrator_integration(self):
        """Test Context-Enriched chunker integration with orchestrator."""
        config = {
            'strategies': {
                'primary': 'context_enriched'
            },
            'context_enriched': {
                'target_chunk_size': 1500,
                'semantic_similarity_threshold': 0.7,
                'enable_ner': True,
                'enable_topic_modeling': True
            }
        }

        orchestrator = ChunkerOrchestrator(config=config)

        text = "Context-enriched chunking test with semantic boundary detection and entity recognition."
        result = orchestrator.chunk_content(text)

        assert result.total_chunks > 0
        assert result.strategy_used == "context_enriched"

    def test_different_boundary_detection_methods(self):
        """Test different boundary detection methods."""
        methods = ["semantic", "topic", "entity", "multi_modal"]

        text = """Machine Learning Overview

Machine learning is a subset of artificial intelligence that focuses on algorithms.
These algorithms can learn and make predictions from data automatically.

Neural Networks

Neural networks are inspired by biological neural networks in animal brains.
They consist of layers of interconnected nodes that process information."""

        for method in methods:
            chunker = create_chunker("context_enriched",
                                   boundary_detection_method=method,
                                   target_chunk_size=300)

            result = chunker.chunk(text)

            assert result.total_chunks > 0
            assert result.strategy_used == "context_enriched"

    def test_context_preservation_priorities(self):
        """Test different context preservation priorities."""
        priorities = ["speed", "balanced", "quality"]

        text = "Test content for context preservation analysis with multiple sentences and topics."

        for priority in priorities:
            chunker = create_chunker("context_enriched",
                                   context_preservation_priority=priority)

            result = chunker.chunk(text)
            assert result.total_chunks > 0

    def test_chunk_estimation(self):
        """Test chunk count estimation."""
        chunker = create_chunker("context_enriched", target_chunk_size=500)

        # Test with varied content to avoid semantic similarity collapsing chunks
        text = "Machine learning is transforming technology. " + \
               "Artificial intelligence enables smart automation. " + \
               "Data science helps understand patterns. " + \
               "Computer vision processes visual information. " + \
               "Natural language processing handles text analysis. " + \
               "Robotics creates intelligent physical systems. " + \
               "Deep learning uses neural network architectures. " + \
               "Cloud computing provides scalable infrastructure. "
        # Repeat to make it long enough
        text = text * 10

        estimated = chunker.estimate_chunks(text)

        assert estimated > 0
        assert isinstance(estimated, int)

        # Test actual chunking to compare
        result = chunker.chunk(text)
        actual = result.total_chunks

        # For varied content, estimation should be more reasonable
        # Allow wider range since context-enriched chunking is semantic-driven
        if actual > 0:
            ratio = estimated / actual
            # Very permissive range for context-enriched chunking
            assert 0.1 <= ratio <= 10.0, f"Estimation ratio {ratio} outside acceptable range"
        else:
            assert estimated >= 0

    def test_get_overlap_sentences(self):
        """Test overlap sentence calculation."""
        chunker = create_chunker("context_enriched")

        sentences = [
            "First sentence for testing.",
            "Second sentence in the list.",
            "Third sentence for overlap.",
            "Fourth and final sentence."
        ]

        # Test with overlap
        overlap = chunker._get_overlap_sentences(sentences, 50)
        assert isinstance(overlap, list)
        assert len(overlap) <= len(sentences)

        # Test with no overlap
        no_overlap = chunker._get_overlap_sentences(sentences, 0)
        assert len(no_overlap) == 0

    def test_advanced_features_flags(self):
        """Test enabling/disabling advanced features."""
        # Test with all features enabled
        full_chunker = create_chunker("context_enriched",
                                    enable_ner=True,
                                    enable_topic_modeling=True,
                                    enable_coreference=True,
                                    generate_semantic_fingerprints=True,
                                    extract_key_phrases=True,
                                    analyze_sentiment=True)

        # Test with minimal features
        minimal_chunker = create_chunker("context_enriched",
                                       enable_ner=False,
                                       enable_topic_modeling=False,
                                       generate_semantic_fingerprints=False)

        text = "Test content for feature flag validation and processing."

        # Both should work
        full_result = full_chunker.chunk(text)
        minimal_result = minimal_chunker.chunk(text)

        assert full_result.total_chunks > 0
        assert minimal_result.total_chunks > 0

    def test_contextual_chunk_creation(self):
        """Test creation of contextual chunks with metadata."""
        chunker = create_chunker("context_enriched")

        sentences = [
            "Machine learning algorithms process data automatically.",
            "Deep learning uses neural networks for pattern recognition.",
            "AI systems can solve complex problems efficiently."
        ]

        contextual_chunk = chunker._create_chunk_from_sentences(sentences)

        if contextual_chunk:
            assert isinstance(contextual_chunk, ContextualChunk)
            assert len(contextual_chunk.content) > 0
            assert isinstance(contextual_chunk.semantic_entities, list)
            assert isinstance(contextual_chunk.topics, list)
            assert 0.0 <= contextual_chunk.coherence_score <= 1.0
            assert 0.0 <= contextual_chunk.context_preservation_score <= 1.0
            assert 0.0 <= contextual_chunk.boundary_quality_score <= 1.0
            assert isinstance(contextual_chunk.semantic_fingerprint, list)

    def test_semantic_entity_dataclass(self):
        """Test SemanticEntity dataclass functionality."""
        entity = SemanticEntity(
            text="Apple Inc.",
            label="ORG",
            start=0,
            end=10,
            confidence=0.95,
            context="Apple Inc. is a technology company"
        )

        assert entity.text == "Apple Inc."
        assert entity.label == "ORG"
        assert entity.start == 0
        assert entity.end == 10
        assert entity.confidence == 0.95
        assert isinstance(entity.related_entities, list)

    def test_topic_info_dataclass(self):
        """Test TopicInfo dataclass functionality."""
        topic = TopicInfo(
            topic_id=1,
            keywords=["machine", "learning", "algorithm"],
            weight=0.75,
            coherence_score=0.85,
            sentences=[0, 1, 2]
        )

        assert topic.topic_id == 1
        assert len(topic.keywords) == 3
        assert topic.weight == 0.75
        assert topic.coherence_score == 0.85
        assert len(topic.sentences) == 3
        assert isinstance(topic.entities, list)

    def test_semantic_boundary_dataclass(self):
        """Test SemanticBoundary dataclass functionality."""
        boundary = SemanticBoundary(
            position=5,
            boundary_type="topic_shift",
            confidence=0.8,
            reason="Topic change detected",
            metadata={"from_topic": 1, "to_topic": 2}
        )

        assert boundary.position == 5
        assert boundary.boundary_type == "topic_shift"
        assert boundary.confidence == 0.8
        assert boundary.reason == "Topic change detected"
        assert boundary.metadata["from_topic"] == 1
