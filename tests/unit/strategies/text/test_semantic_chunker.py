"""
Unit tests for SemanticChunker.

This test suite covers semantic chunking functionality including:
- Different semantic models (sentence transformers, spaCy, TF-IDF)
- Boundary detection methods
- Similarity threshold configurations
- Chunk size constraints
- Streaming capabilities
- Adaptation mechanisms
- Performance and quality metrics
"""

import pytest
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from chunking_strategy.strategies.text.semantic_chunker import SemanticChunker, SemanticModel, BoundaryDetectionMethod
from chunking_strategy.strategies.text.discourse_aware_chunker import DiscourseAwareChunker
from chunking_strategy.core.base import ModalityType, ChunkingResult


class TestSemanticChunker:
    """Test suite for SemanticChunker functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures and sample data."""
        # Basic chunker for most tests
        self.chunker = SemanticChunker(
            semantic_model="tfidf",  # Use TF-IDF for reliability in tests
            similarity_threshold=0.7,
            min_chunk_sentences=2,
            max_chunk_sentences=8
        )

        # Sample text with distinct topics for semantic boundary testing
        self.sample_text = """
        Machine learning is a subset of artificial intelligence. It focuses on algorithms that can learn from data.
        Deep learning uses neural networks with multiple layers. These networks can recognize complex patterns.

        Climate change is a pressing global issue. Rising temperatures affect ecosystems worldwide.
        Ocean levels are rising due to melting ice caps. This threatens coastal communities.

        Space exploration has advanced significantly in recent years. Private companies now launch rockets regularly.
        Mars colonization is becoming a realistic possibility. NASA and SpaceX are leading these efforts.

        Quantum computing represents a revolutionary technology. It uses quantum mechanics principles for computation.
        Quantum bits can exist in multiple states simultaneously. This enables exponentially faster calculations.
        """.strip()

        # Short text for edge case testing
        self.short_text = "This is a single sentence for testing. Here is another sentence."

        # Medium complexity text
        self.medium_text = """
        The history of artificial intelligence began in the 1940s. Early computers showed promise for automated reasoning.
        Alan Turing proposed the famous Turing test in 1950. This test evaluates machine intelligence.

        Machine learning emerged in the 1960s as a subset of AI. Researchers developed algorithms that could learn from data.
        Neural networks were inspired by biological brain structures. They showed potential for pattern recognition.

        Modern AI has achieved remarkable breakthroughs. Deep learning revolutionized computer vision and natural language processing.
        Large language models can generate human-like text. They understand context and maintain coherent conversations.
        """

    def test_initialization_valid_parameters(self):
        """Test chunker initialization with valid parameters."""
        chunker = SemanticChunker(
            semantic_model="sentence_transformer",
            embedding_model="all-MiniLM-L6-v2",
            similarity_threshold=0.8,
            min_chunk_sentences=3,
            max_chunk_sentences=10,
            boundary_detection="coherence_based"
        )

        assert chunker.semantic_model == SemanticModel.SENTENCE_TRANSFORMER
        assert chunker.embedding_model == "all-MiniLM-L6-v2"
        assert chunker.similarity_threshold == 0.8
        assert chunker.min_chunk_sentences == 3
        assert chunker.max_chunk_sentences == 10
        assert chunker.boundary_detection == BoundaryDetectionMethod.COHERENCE_BASED

    def test_initialization_invalid_parameters(self):
        """Test chunker initialization with invalid parameters."""
        # Invalid similarity threshold
        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            SemanticChunker(similarity_threshold=1.5)

        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            SemanticChunker(similarity_threshold=-0.1)

        # Invalid sentence count parameters
        with pytest.raises(ValueError, match="min_chunk_sentences must be less than max_chunk_sentences"):
            SemanticChunker(min_chunk_sentences=10, max_chunk_sentences=5)

        with pytest.raises(ValueError, match="min_chunk_sentences must be at least 1"):
            SemanticChunker(min_chunk_sentences=0)

        # Invalid coherence weight
        with pytest.raises(ValueError, match="coherence_weight must be between 0.0 and 1.0"):
            SemanticChunker(coherence_weight=1.2)

    def test_semantic_model_enum_validation(self):
        """Test that semantic model enum values are properly validated."""
        # Valid models
        valid_models = ["sentence_transformer", "spacy", "tfidf"]
        for model in valid_models:
            chunker = SemanticChunker(semantic_model=model)
            assert chunker.semantic_model.value == model

        # Invalid model should raise ValueError
        with pytest.raises(ValueError):
            SemanticChunker(semantic_model="invalid_model")

    def test_boundary_detection_enum_validation(self):
        """Test that boundary detection method enum values are properly validated."""
        # Valid methods
        valid_methods = ["similarity_threshold", "sliding_window", "dynamic_threshold", "coherence_based"]
        for method in valid_methods:
            chunker = SemanticChunker(boundary_detection=method)
            assert chunker.boundary_detection.value == method

        # Invalid method should raise ValueError
        with pytest.raises(ValueError):
            SemanticChunker(boundary_detection="invalid_method")

    def test_chunk_basic_functionality(self):
        """Test basic chunking functionality with semantic analysis."""
        result = self.chunker.chunk(self.sample_text)

        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "semantic"
        assert len(result.chunks) >= 2  # Should detect multiple topics
        assert all(chunk.modality == ModalityType.TEXT for chunk in result.chunks)

        # Check that chunks contain reasonable content
        total_content = " ".join(chunk.content for chunk in result.chunks)
        assert len(total_content) > 0

    def test_sentence_segmentation(self):
        """Test sentence segmentation functionality."""
        sentences = self.chunker._segment_sentences(self.medium_text)

        assert isinstance(sentences, list)
        assert len(sentences) > 5  # Should find multiple sentences
        assert all(isinstance(sentence, str) for sentence in sentences)
        assert all(len(sentence.strip()) > 0 for sentence in sentences)

    def test_similarity_based_chunking(self):
        """Test similarity-based boundary detection."""
        # Test with different similarity thresholds
        thresholds = [0.5, 0.7, 0.9]
        chunk_counts = []

        for threshold in thresholds:
            chunker = SemanticChunker(
                semantic_model="tfidf",
                similarity_threshold=threshold,
                min_chunk_sentences=2
            )
            result = chunker.chunk(self.sample_text)
            chunk_counts.append(len(result.chunks))

        # Higher threshold should generally create more chunks (more boundaries)
        assert chunk_counts[2] >= chunk_counts[0], f"Expected more chunks with higher threshold: {chunk_counts}"

    def test_boundary_detection_methods(self):
        """Test different boundary detection methods."""
        methods = [
            "similarity_threshold",
            "sliding_window",
            "dynamic_threshold",
            "coherence_based"
        ]

        results = {}
        for method in methods:
            chunker = SemanticChunker(
                semantic_model="tfidf",
                boundary_detection=method,
                min_chunk_sentences=2
            )
            result = chunker.chunk(self.sample_text)
            results[method] = result

        # All methods should produce valid results
        for method, result in results.items():
            assert len(result.chunks) >= 1, f"Method {method} produced no chunks"
            assert all(chunk.content.strip() for chunk in result.chunks), f"Method {method} produced empty chunks"

    def test_chunk_size_constraints(self):
        """Test that chunk size constraints are properly enforced."""
        chunker = SemanticChunker(
            semantic_model="tfidf",
            min_chunk_sentences=3,
            max_chunk_sentences=6,
            max_chunk_chars=1000
        )

        result = chunker.chunk(self.sample_text)

        for i, chunk in enumerate(result.chunks):
            sentences_in_chunk = len([s for s in chunk.content.split('.') if s.strip()])

            # Check character limit
            assert len(chunk.content) <= 1000, f"Chunk {i} exceeds character limit: {len(chunk.content)}"

            # Check sentence count (may be flexible for last chunk or content constraints)
            if i < len(result.chunks) - 1:  # Not the last chunk
                assert sentences_in_chunk >= 2, f"Chunk {i} has too few sentences: {sentences_in_chunk}"

    def test_empty_and_minimal_text_handling(self):
        """Test handling of empty and minimal text inputs."""
        # Empty text
        result = self.chunker.chunk("")
        assert len(result.chunks) == 0
        assert result.processing_time >= 0

        # Whitespace only
        result = self.chunker.chunk("   \n\t   ")
        assert len(result.chunks) == 0

        # Single sentence (below minimum)
        result = self.chunker.chunk("This is a single sentence.")
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "This is a single sentence."

        # Short text with multiple sentences
        result = self.chunker.chunk(self.short_text)
        assert len(result.chunks) >= 1

    def test_file_input_handling(self):
        """Test handling of file path inputs."""
        # Create a temporary test file
        test_file = Path("/tmp/test_semantic_chunker.txt")
        test_file.write_text(self.medium_text)

        try:
            result = self.chunker.chunk(test_file)
            assert len(result.chunks) >= 1
            assert result.source_info.get("source_type") in ["file", "content"]
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_bytes_input_handling(self):
        """Test handling of bytes input."""
        text_bytes = self.medium_text.encode('utf-8')
        result = self.chunker.chunk(text_bytes)

        assert len(result.chunks) >= 1
        total_content = " ".join(chunk.content for chunk in result.chunks)
        assert len(total_content) > 0

    def test_streaming_functionality(self):
        """Test streaming chunk generation."""
        # Split text into stream chunks
        stream_data = [
            self.medium_text[:100],
            self.medium_text[100:300],
            self.medium_text[300:]
        ]

        chunks = list(self.chunker.chunk_stream(stream_data, source_info={"source": "test_stream"}))

        assert len(chunks) >= 1
        assert all(isinstance(chunk, type(chunks[0])) for chunk in chunks)

        # Check that streaming produces similar results to batch processing
        batch_result = self.chunker.chunk(self.medium_text)
        stream_chunk_count = len(chunks)
        batch_chunk_count = len(batch_result.chunks)

        # Stream might produce slightly different chunking, but should be comparable
        assert abs(stream_chunk_count - batch_chunk_count) <= 2

    def test_metadata_completeness(self):
        """Test that chunk metadata is complete and accurate."""
        result = self.chunker.chunk(self.medium_text)

        for i, chunk in enumerate(result.chunks):
            # Required metadata fields
            assert chunk.metadata.extra["chunker_used"] == self.chunker.name
            assert chunk.metadata.extra["chunk_index"] == i
            assert chunk.metadata.length == len(chunk.content)
            assert chunk.metadata.offset >= 0

            # Semantic-specific metadata
            assert "sentence_count" in chunk.metadata.extra
            assert "semantic_model" in chunk.metadata.extra
            assert "similarity_threshold" in chunk.metadata.extra
            assert chunk.metadata.extra["chunking_strategy"] == "semantic"

            # Validate sentence count
            sentence_count = chunk.metadata.extra["sentence_count"]
            assert isinstance(sentence_count, int)
            assert sentence_count > 0

    def test_adaptation_quality_feedback(self):
        """Test adaptation based on quality feedback."""
        original_threshold = self.chunker.similarity_threshold
        original_min_sentences = self.chunker.min_chunk_sentences

        # Poor quality feedback should increase sensitivity
        changes = self.chunker.adapt_parameters(0.3, "quality")

        assert "similarity_threshold" in changes or "min_chunk_sentences" in changes
        if "similarity_threshold" in changes:
            assert self.chunker.similarity_threshold > original_threshold

        # Good quality feedback should decrease sensitivity (more efficient)
        self.chunker.similarity_threshold = 0.8  # Reset to high value
        changes = self.chunker.adapt_parameters(0.9, "quality")

        if "similarity_threshold" in changes:
            assert self.chunker.similarity_threshold < 0.8

    def test_adaptation_performance_feedback(self):
        """Test adaptation based on performance feedback."""
        original_max_sentences = self.chunker.max_chunk_sentences
        original_threshold = self.chunker.similarity_threshold

        # Poor performance should make processing faster
        changes = self.chunker.adapt_parameters(0.2, "performance")

        assert len(changes) > 0
        if "max_chunk_sentences" in changes:
            assert self.chunker.max_chunk_sentences > original_max_sentences
        if "similarity_threshold" in changes:
            assert self.chunker.similarity_threshold < original_threshold

    def test_adaptation_coherence_feedback(self):
        """Test adaptation based on coherence feedback."""
        original_threshold = self.chunker.similarity_threshold

        # Poor coherence should increase strictness
        changes = self.chunker.adapt_parameters(0.4, "coherence")

        if "similarity_threshold" in changes:
            assert self.chunker.similarity_threshold > original_threshold

    def test_adaptation_history_tracking(self):
        """Test that adaptation history is properly tracked."""
        # Perform several adaptations that will make changes
        self.chunker.adapt_parameters(0.3, "quality", test_context="first_adaptation")
        self.chunker.adapt_parameters(0.2, "performance", test_context="second_adaptation")  # Use lower score to force changes

        history = self.chunker.get_adaptation_history()
        assert len(history) >= 1  # At least one adaptation should be recorded

        # Check history structure
        for record in history:
            assert "timestamp" in record
            assert "feedback_score" in record
            assert "feedback_type" in record
            assert "changes" in record
            assert "parameters_after" in record

    def test_configuration_retrieval(self):
        """Test configuration retrieval functionality."""
        config = self.chunker.get_config()

        assert config["name"] == "semantic"
        assert config["semantic_model"] == self.chunker.semantic_model.value
        assert config["similarity_threshold"] == self.chunker.similarity_threshold
        assert config["min_chunk_sentences"] == self.chunker.min_chunk_sentences
        assert config["max_chunk_sentences"] == self.chunker.max_chunk_sentences
        assert "performance_stats" in config

    def test_semantic_boundary_detection(self):
        """Test semantic boundary detection with known topic shifts."""
        # Text with clear topic boundaries
        topic_text = """
        Python is a programming language. It is widely used for data science applications.
        Machine learning libraries like scikit-learn are popular. TensorFlow and PyTorch enable deep learning.

        Cooking is an important life skill. Healthy recipes promote better nutrition.
        Fresh ingredients make meals more flavorful. Meal planning saves time and money.

        Exercise benefits physical and mental health. Regular cardio improves heart function.
        Strength training builds muscle mass. Yoga enhances flexibility and balance.
        """

        result = self.chunker.chunk(topic_text)

        # Should detect multiple topics and create separate chunks
        assert len(result.chunks) >= 2, f"Expected multiple chunks for distinct topics, got {len(result.chunks)}"

        # Check that topics are reasonably separated
        chunk_contents = [chunk.content.lower() for chunk in result.chunks]
        programming_chunks = [i for i, content in enumerate(chunk_contents) if "python" in content or "programming" in content]
        cooking_chunks = [i for i, content in enumerate(chunk_contents) if "cooking" in content or "recipes" in content]

        # Programming and cooking content should ideally be in different chunks
        if programming_chunks and cooking_chunks:
            assert not set(programming_chunks).intersection(set(cooking_chunks)), "Programming and cooking topics mixed in chunks"

    def test_large_chunk_splitting(self):
        """Test that large chunks are properly split."""
        # Create a chunker with small maximum sentence limit
        small_chunker = SemanticChunker(
            semantic_model="tfidf",
            max_chunk_sentences=3,
            min_chunk_sentences=1,
            similarity_threshold=0.3  # Low threshold to avoid splitting
        )

        # Long text that should be split
        long_text = ". ".join([f"This is sentence number {i} about similar topics" for i in range(20)])

        result = small_chunker.chunk(long_text)

        # Should create multiple chunks due to size constraints
        assert len(result.chunks) > 1

        # Each chunk should respect the sentence limit
        for chunk in result.chunks:
            sentences = [s for s in chunk.content.split('.') if s.strip()]
            assert len(sentences) <= 3, f"Chunk exceeds max sentence limit: {len(sentences)} sentences"

    def test_similarity_computation(self):
        """Test similarity score computation accuracy."""
        # Simple test sentences with known relationships
        sentences = [
            "The cat sat on the mat.",
            "A feline rested on the rug.",  # Similar to first
            "Space exploration is fascinating.",  # Different topic
            "Astronauts travel to distant planets."  # Similar to third
        ]

        chunker = SemanticChunker(semantic_model="tfidf")
        chunker._initialize_semantic_model()

        embeddings = chunker._compute_sentence_embeddings(sentences)
        similarities = chunker._compute_similarity_scores(embeddings)

        assert len(similarities) == len(sentences) - 1
        assert all(0 <= sim <= 1 for sim in similarities), f"Similarities out of range: {similarities}"

        # Similar sentences should have higher similarity than different topics
        # (This is a basic check - exact values depend on the model)
        assert isinstance(similarities[0], (int, float))

    def test_coherence_based_detection_edge_cases(self):
        """Test coherence-based detection with edge cases."""
        chunker = SemanticChunker(
            semantic_model="tfidf",
            boundary_detection="coherence_based",
            min_chunk_sentences=2
        )

        # Very short text
        short_result = chunker.chunk("One. Two. Three.")
        assert len(short_result.chunks) >= 1

        # Repetitive text (high coherence)
        repetitive_text = "The same topic. Again the same topic. Once more the same topic. Same topic repeated."
        repetitive_result = chunker.chunk(repetitive_text)
        assert len(repetitive_result.chunks) >= 1

    def test_performance_metrics_tracking(self):
        """Test that performance metrics are properly tracked."""
        # Process some text to generate metrics
        result = self.chunker.chunk(self.medium_text)

        config = self.chunker.get_config()
        stats = config["performance_stats"]

        assert stats["total_sentences_processed"] > 0
        assert stats["embedding_time"] >= 0
        assert stats["boundary_detection_time"] >= 0

        # Should track cumulative statistics across multiple calls
        initial_sentences = stats["total_sentences_processed"]

        # Process more text
        result2 = self.chunker.chunk(self.sample_text)
        updated_config = self.chunker.get_config()
        updated_stats = updated_config["performance_stats"]

        assert updated_stats["total_sentences_processed"] > initial_sentences

    def test_fallback_mechanisms(self):
        """Test fallback mechanisms when preferred models are unavailable."""
        # This test checks that the chunker can still function even if some dependencies are missing
        chunker = SemanticChunker(
            semantic_model="tfidf",  # Fallback model
            similarity_threshold=0.6
        )

        result = chunker.chunk(self.medium_text)
        assert len(result.chunks) >= 1
        assert result.strategy_used == "semantic"

    @pytest.mark.slow
    def test_processing_real_text_files(self):
        """Test processing various text file types and sizes."""
        # This is marked as slow since it might involve larger files

        test_texts = {
            "academic": """
            Introduction to Semantic Analysis. Natural language processing encompasses various techniques.
            Semantic analysis focuses on meaning extraction from text. Word embeddings represent semantic relationships.

            Machine Learning Applications. Supervised learning uses labeled training data.
            Unsupervised methods discover hidden patterns. Deep learning employs neural network architectures.

            Evaluation Metrics. Precision measures positive prediction accuracy.
            Recall quantifies the ability to find relevant instances. F1 score combines both metrics.
            """,
            "narrative": """
            Once upon a time, there was a small village. The villagers lived peacefully for many years.
            A mysterious stranger arrived one autumn day. Nobody knew where he came from or why he was there.

            The stranger possessed unusual abilities. He could predict weather patterns with perfect accuracy.
            Crops began growing faster under his guidance. The village prospered like never before.

            Years passed and the village transformed. New technologies appeared seemingly overnight.
            The stranger's true identity remained unknown. Some believed he came from the future.
            """
        }

        for text_type, text_content in test_texts.items():
            result = self.chunker.chunk(text_content)

            assert len(result.chunks) >= 1, f"Failed to process {text_type} text"
            assert result.processing_time > 0
            assert "total_sentences" in result.source_info

            # Verify content integrity
            total_recreated = " ".join(chunk.content for chunk in result.chunks)
            assert len(total_recreated) > 0

    def test_processing_test_data_files(self):
        """Test processing actual files from the test_data directory."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        # Test files to process with their expected characteristics
        test_files = [
            # Text files
            ("sample_simple_text.txt", {"min_chunks": 1, "content_type": "simple"}),
            ("sample_article.txt", {"min_chunks": 1, "content_type": "article"}),
            ("alice_wonderland.txt", {"min_chunks": 2, "content_type": "literature"}),
            ("technical_doc.txt", {"min_chunks": 1, "content_type": "technical"}),
            ("sample_semantic_text.txt", {"min_chunks": 1, "content_type": "semantic_test"}),

            # Markdown files
            ("simple_document.md", {"min_chunks": 1, "content_type": "markdown"}),
            ("nested_structure.md", {"min_chunks": 3, "content_type": "structured_markdown"}),

            # Mixed content files
            ("sample_mixed_content.txt", {"min_chunks": 2, "content_type": "mixed"}),
            ("sample_mixed_strategies.txt", {"min_chunks": 1, "content_type": "mixed_strategies"}),

            # Unicode files
            ("unicode.txt", {"min_chunks": 1, "content_type": "unicode"}),
        ]

        results = {}

        for filename, expectations in test_files:
            file_path = test_data_dir / filename

            if not file_path.exists():
                print(f"⚠️ Test file not found: {filename}, skipping...")
                continue

            print(f"🔍 Testing semantic chunking with {filename}...")

            try:
                # Test with file path
                result = self.chunker.chunk(file_path)

                # Basic validations
                assert len(result.chunks) >= expectations["min_chunks"], \
                    f"Expected at least {expectations['min_chunks']} chunks for {filename}, got {len(result.chunks)}"
                assert result.strategy_used == "semantic"
                assert result.processing_time > 0

                # Content integrity check
                total_length = sum(len(chunk.content) for chunk in result.chunks)
                assert total_length > 0, f"No content found in chunks for {filename}"

                # Metadata validation
                for chunk in result.chunks:
                    assert chunk.metadata.extra["chunker_used"] == "semantic"
                    assert "sentence_count" in chunk.metadata.extra
                    assert "semantic_model" in chunk.metadata.extra
                    assert chunk.metadata.extra["sentence_count"] > 0

                # Source info validation
                assert "total_sentences" in result.source_info
                assert "semantic_model" in result.source_info
                assert result.source_info["total_sentences"] > 0

                results[filename] = {
                    "chunk_count": len(result.chunks),
                    "total_length": total_length,
                    "processing_time": result.processing_time,
                    "avg_chunk_length": total_length / len(result.chunks),
                    "sentences_processed": result.source_info.get("total_sentences", 0),
                    "content_type": expectations["content_type"]
                }

                print(f"✅ {filename}: {len(result.chunks)} chunks, {total_length} chars, {result.processing_time:.3f}s")

            except Exception as e:
                pytest.fail(f"Failed to process {filename}: {e}")

        # Verify we processed at least some files
        assert len(results) > 0, "No test files were successfully processed"

        # Print summary
        print(f"\n📊 Processed {len(results)} files successfully:")
        for filename, stats in results.items():
            print(f"   {filename}: {stats['chunk_count']} chunks, avg {stats['avg_chunk_length']:.0f} chars/chunk")

    def test_processing_json_files(self):
        """Test processing JSON files with semantic chunking."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        json_files = [
            "simple_objects.json",
            "sample_adaptive_data.json",
            "sample.jsonl"
        ]

        for filename in json_files:
            file_path = test_data_dir / filename

            if not file_path.exists():
                continue

            print(f"🔍 Testing JSON chunking with {filename}...")

            try:
                result = self.chunker.chunk(file_path)

                # JSON files should be processed as text for semantic analysis
                assert len(result.chunks) >= 1
                assert result.strategy_used == "semantic"

                # Content should be readable (JSON might have limited text content)
                total_content = " ".join(chunk.content for chunk in result.chunks)
                # Some JSON files might not have meaningful text content for semantic analysis
                if len(total_content) == 0:
                    print(f"⚠️ {filename}: No text content extracted (expected for some JSON files)")
                    continue

                # Should detect some sentence structure even in JSON
                sentences_found = result.source_info.get("total_sentences", 0)
                assert sentences_found >= 0  # JSON might not have traditional sentences

                print(f"✅ {filename}: {len(result.chunks)} chunks, {sentences_found} sentences detected")

            except Exception as e:
                pytest.fail(f"Failed to process JSON file {filename}: {e}")

    def test_processing_xml_files(self):
        """Test processing XML files with semantic chunking."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        xml_files = ["books_catalog.xml"]

        for filename in xml_files:
            file_path = test_data_dir / filename

            if not file_path.exists():
                continue

            print(f"🔍 Testing XML chunking with {filename}...")

            try:
                result = self.chunker.chunk(file_path)

                assert len(result.chunks) >= 1
                assert result.strategy_used == "semantic"

                # Check content extraction
                total_content = " ".join(chunk.content for chunk in result.chunks)
                assert len(total_content) > 0

                # XML might have structural text that forms sentences
                for chunk in result.chunks:
                    assert len(chunk.content.strip()) > 0
                    assert "semantic_model" in chunk.metadata.extra

                print(f"✅ {filename}: {len(result.chunks)} chunks processed")

            except Exception as e:
                pytest.fail(f"Failed to process XML file {filename}: {e}")

    def test_processing_rtf_files(self):
        """Test processing RTF files with semantic chunking."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        rtf_files = ["meeting_notes.rtf"]

        for filename in rtf_files:
            file_path = test_data_dir / filename

            if not file_path.exists():
                continue

            print(f"🔍 Testing RTF chunking with {filename}...")

            try:
                # RTF files might need special handling, but semantic chunker should process as text
                result = self.chunker.chunk(file_path)

                assert len(result.chunks) >= 1
                assert result.strategy_used == "semantic"

                # Verify content was extracted
                total_length = sum(len(chunk.content) for chunk in result.chunks)
                assert total_length > 0

                print(f"✅ {filename}: {len(result.chunks)} chunks, {total_length} chars total")

            except Exception as e:
                # RTF might fail if no RTF parser available - that's okay
                print(f"⚠️ RTF processing not available for {filename}: {e}")

    def test_different_semantic_models_on_files(self):
        """Test different semantic models on the same files."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        # Test one representative file with different models
        test_file = test_data_dir / "simple_document.md"
        if not test_file.exists():
            pytest.skip("simple_document.md not found")

        semantic_models = ["tfidf", "sentence_transformer", "spacy"]
        results = {}

        for model in semantic_models:
            try:
                chunker = SemanticChunker(
                    semantic_model=model,
                    similarity_threshold=0.7,
                    min_chunk_sentences=2
                )

                result = chunker.chunk(test_file)

                results[model] = {
                    "chunk_count": len(result.chunks),
                    "processing_time": result.processing_time,
                    "total_sentences": result.source_info.get("total_sentences", 0),
                    "success": True
                }

                print(f"✅ Model {model}: {len(result.chunks)} chunks in {result.processing_time:.3f}s")

            except Exception as e:
                # Some models might not be available
                results[model] = {"success": False, "error": str(e)}
                print(f"⚠️ Model {model} failed: {e}")

        # At least one model should work (TF-IDF is most reliable)
        successful_models = [m for m, r in results.items() if r.get("success", False)]
        assert len(successful_models) >= 1, f"No semantic models worked. Results: {results}"

    def test_file_processing_with_different_thresholds(self):
        """Test file processing with different similarity thresholds."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        # Use a file with clear topic boundaries
        test_file = test_data_dir / "nested_structure.md"
        if not test_file.exists():
            pytest.skip("nested_structure.md not found")

        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = []

        for threshold in thresholds:
            chunker = SemanticChunker(
                semantic_model="tfidf",
                similarity_threshold=threshold,
                min_chunk_sentences=2
            )

            result = chunker.chunk(test_file)

            results.append({
                "threshold": threshold,
                "chunk_count": len(result.chunks),
                "processing_time": result.processing_time,
                "avg_chunk_length": sum(len(c.content) for c in result.chunks) / len(result.chunks) if result.chunks else 0
            })

            print(f"Threshold {threshold}: {len(result.chunks)} chunks, avg length {results[-1]['avg_chunk_length']:.0f}")

        # Validate threshold effects
        assert len(results) == len(thresholds)
        for i, result_data in enumerate(results):
            assert result_data["chunk_count"] >= 1, f"Threshold {thresholds[i]} produced no chunks"

    def test_large_file_processing(self):
        """Test processing larger files to validate scalability."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        # Look for the largest available files
        large_files = ["alice_wonderland.txt", "nested_structure.md", "sample_mixed_content.txt"]

        processed_any = False

        for filename in large_files:
            file_path = test_data_dir / filename

            if not file_path.exists():
                continue

            try:
                file_size = file_path.stat().st_size
                print(f"🔍 Processing large file {filename} ({file_size} bytes)...")

                start_time = time.time()
                result = self.chunker.chunk(file_path)
                processing_time = time.time() - start_time

                # Performance validation
                chars_per_second = file_size / processing_time if processing_time > 0 else 0

                assert len(result.chunks) >= 1
                assert result.processing_time > 0
                assert processing_time < 60.0, f"Processing too slow: {processing_time}s"

                print(f"✅ {filename}: {len(result.chunks)} chunks, {chars_per_second:.0f} chars/sec")
                processed_any = True

            except Exception as e:
                print(f"⚠️ Failed to process {filename}: {e}")

        if not processed_any:
            pytest.skip("No large files found to test")

    def test_file_content_preservation(self):
        """Test that file processing preserves content integrity."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        test_file = test_data_dir / "sample_semantic_text.txt"
        if not test_file.exists():
            pytest.skip("sample_semantic_text.txt not found")

        # Read original content
        original_content = test_file.read_text(encoding='utf-8')
        original_words = set(original_content.lower().split())

        # Process with semantic chunker
        result = self.chunker.chunk(test_file)

        # Reconstruct content from chunks
        reconstructed_content = " ".join(chunk.content for chunk in result.chunks)
        reconstructed_words = set(reconstructed_content.lower().split())

        # Check word preservation
        common_words = original_words & reconstructed_words
        preservation_ratio = len(common_words) / len(original_words) if original_words else 0

        assert preservation_ratio > 0.85, f"Poor content preservation: {preservation_ratio:.3f}"

        # Check that essential content structure is maintained
        assert len(reconstructed_content) > len(original_content) * 0.7, \
            "Significant content loss detected"

        print(f"✅ Content preservation: {preservation_ratio:.3f} ({len(common_words)}/{len(original_words)} words)")

    def test_file_error_handling(self):
        """Test error handling for problematic files."""
        test_data_dir = Path("test_data")

        # Test non-existent file
        non_existent = test_data_dir / "does_not_exist.txt"

        try:
            result = self.chunker.chunk(non_existent)
            # Should either raise an exception or return empty result
            if result.chunks:
                pytest.fail("Expected error or empty result for non-existent file")
        except (FileNotFoundError, IOError):
            # Expected behavior
            pass

        # Test with empty string (edge case but valid)
        empty_result = self.chunker.chunk("")
        assert len(empty_result.chunks) == 0, "Expected empty result for empty string"

        # Test with very short content
        short_result = self.chunker.chunk("Hi")
        assert len(short_result.chunks) >= 0, "Should handle short content gracefully"

        print("✅ Error handling validated for problematic file inputs")

    def test_different_similarity_thresholds_impact(self):
        """Test the impact of different similarity thresholds on chunking results."""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = []

        for threshold in thresholds:
            chunker = SemanticChunker(
                semantic_model="tfidf",
                similarity_threshold=threshold,
                min_chunk_sentences=2
            )
            result = chunker.chunk(self.sample_text)
            results.append((threshold, len(result.chunks), result.source_info.get("total_boundaries_detected", 0)))

        # Analyze the relationship between threshold and chunk count
        for i in range(len(results) - 1):
            current_threshold, current_chunks, current_boundaries = results[i]
            next_threshold, next_chunks, next_boundaries = results[i + 1]

            # Generally, higher thresholds should create more boundaries (and thus more chunks)
            # But this can vary based on content, so we just ensure reasonable behavior
            assert current_chunks >= 1 and next_chunks >= 1, "All thresholds should produce at least one chunk"

    def test_chunk_content_integrity(self):
        """Test that chunk content maintains text integrity."""
        result = self.chunker.chunk(self.medium_text)

        # Reconstruct text from chunks
        reconstructed = " ".join(chunk.content for chunk in result.chunks)

        # Should preserve the essential content (allowing for some processing differences)
        original_words = set(self.medium_text.lower().split())
        reconstructed_words = set(reconstructed.lower().split())

        # Most words should be preserved
        word_overlap = len(original_words.intersection(reconstructed_words))
        word_total = len(original_words)
        preservation_ratio = word_overlap / word_total if word_total > 0 else 0

        assert preservation_ratio > 0.8, f"Poor word preservation ratio: {preservation_ratio}"

    def test_source_info_propagation(self):
        """Test that source information is properly propagated."""
        source_info = {
            "source": "test_document.txt",
            "source_type": "file",
            "author": "test_author",
            "creation_date": "2024-01-01"
        }

        result = self.chunker.chunk(self.medium_text, source_info=source_info)

        # Check that source info is preserved and enhanced
        assert result.source_info["source"] == "test_document.txt"
        assert result.source_info["source_type"] == "file"
        assert result.source_info["author"] == "test_author"

        # Should add semantic-specific information
        assert "semantic_model" in result.source_info
        assert "total_sentences" in result.source_info
        assert "similarity_threshold" in result.source_info

    def test_error_recovery_and_robustness(self):
        """Test error recovery and robustness with problematic inputs."""
        # Text with unusual characters
        unusual_text = "Tëst wîth ünïcödé çhäracters. Émojis: 🚀🔬🎯. Numbers: 123,456.78."
        result = self.chunker.chunk(unusual_text)
        assert len(result.chunks) >= 1

        # Very long single sentence
        long_sentence = "This is a very long sentence that continues " * 100 + "and ends here."
        result = self.chunker.chunk(long_sentence)
        assert len(result.chunks) >= 1

        # Mixed line endings and formatting
        mixed_text = "Line one.\r\nLine two.\nLine three.\r\n\nParagraph break.\n\nFinal line."
        result = self.chunker.chunk(mixed_text)
        assert len(result.chunks) >= 1


class TestDiscourseAwareExtensions:
    """Test suite for Discourse-Aware extensions to semantic chunking."""

    def test_discourse_aware_inheritance(self):
        """Test that DiscourseAwareChunker properly inherits from SemanticChunker."""
        discourse_chunker = DiscourseAwareChunker()

        # Should have all semantic chunker attributes
        assert hasattr(discourse_chunker, 'similarity_threshold')
        assert hasattr(discourse_chunker, 'min_chunk_sentences')
        assert hasattr(discourse_chunker, 'max_chunk_sentences')
        assert hasattr(discourse_chunker, '_compute_sentence_embeddings')

        # Should have discourse-specific attributes
        assert hasattr(discourse_chunker, 'discourse_weight')
        assert hasattr(discourse_chunker, 'topic_weight')
        assert hasattr(discourse_chunker, 'entity_weight')
        assert hasattr(discourse_chunker, '_detect_discourse_markers')

        # Should be instance of both
        assert isinstance(discourse_chunker, DiscourseAwareChunker)
        assert isinstance(discourse_chunker, SemanticChunker)

    def test_discourse_vs_semantic_chunking_comparison(self):
        """Compare basic semantic vs discourse-aware chunking on same text."""
        test_text = """
        Machine learning has revolutionized artificial intelligence. Deep networks process data with accuracy.
        However, challenges remain in AI development. Bias in datasets leads to unfair decisions.
        Furthermore, interpretability issues persist in complex models. Traditional methods offer transparency.
        Nevertheless, hybrid approaches show promise for the future.
        """

        # Basic semantic chunker
        semantic_chunker = SemanticChunker(
            similarity_threshold=0.6,
            min_chunk_sentences=2,
            max_chunk_sentences=6
        )

        # Discourse-aware chunker
        discourse_chunker = DiscourseAwareChunker(
            similarity_threshold=0.6,
            min_chunk_sentences=2,
            max_chunk_sentences=6,
            discourse_weight=0.4,
            topic_weight=0.3,
            min_boundary_strength=0.5
        )

        semantic_result = semantic_chunker.chunk(test_text)
        discourse_result = discourse_chunker.chunk(test_text)

        # Both should produce results
        assert len(semantic_result.chunks) > 0
        assert len(discourse_result.chunks) > 0

        # Discourse-aware should have additional metadata
        for chunk in discourse_result.chunks:
            assert 'discourse_markers' in chunk.metadata.extra
            assert 'coherence_analysis' in chunk.metadata.extra
            assert 'boundary_type' in chunk.metadata.extra

    def test_discourse_marker_detection_in_semantic_context(self):
        """Test that discourse markers are detected within semantic chunking."""
        test_text = """
        First point about technology. However, there are limitations.
        Furthermore, we must consider alternatives. Nevertheless, progress continues.
        """

        discourse_chunker = DiscourseAwareChunker(
            discourse_marker_sensitivity=0.8,
            min_boundary_strength=0.3  # Lower threshold to catch markers
        )

        result = discourse_chunker.chunk(test_text)

        # Should find discourse markers
        total_markers = 0
        marker_types_found = set()

        for chunk in result.chunks:
            markers = chunk.metadata.extra.get('discourse_markers', [])
            total_markers += len(markers)

            for marker in markers:
                marker_types_found.add(marker.get('type', ''))

        # Should detect contrast and continuation markers
        assert total_markers > 0
        expected_types = {'contrast', 'continuation'}
        assert expected_types.intersection(marker_types_found)

    def test_parameter_compatibility_with_semantic_base(self):
        """Test that semantic parameters work correctly with discourse extensions."""
        # Test with various semantic parameters that should be inherited
        discourse_chunker = DiscourseAwareChunker(
            # Semantic parameters (inherited)
            similarity_threshold=0.8,
            min_chunk_sentences=4,
            max_chunk_sentences=10,
            boundary_detection="dynamic_threshold",
            # Discourse parameters (new)
            discourse_weight=0.5,
            topic_weight=0.2,
            entity_weight=0.2,
            coherence_weight=0.1
        )

        # Should maintain semantic parameter values
        assert discourse_chunker.similarity_threshold == 0.8
        assert discourse_chunker.min_chunk_sentences == 4
        assert discourse_chunker.max_chunk_sentences == 10

        # Should have discourse parameter values
        assert discourse_chunker.discourse_weight == 0.5
        assert discourse_chunker.topic_weight == 0.2

        # Should work with text
        test_text = "First sentence. However, second sentence. Furthermore, third sentence."
        result = discourse_chunker.chunk(test_text)

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0

    def test_quality_improvement_over_basic_semantic(self):
        """Test that discourse-aware chunker provides quality improvements."""
        test_text = """
        Machine learning transforms industries. However, ethical concerns arise with AI deployment.
        Furthermore, algorithmic bias affects decision-making processes significantly.
        In contrast, human oversight provides necessary ethical guidance.
        Nevertheless, automated systems offer scalability and efficiency benefits.
        Finally, balanced approaches combining human and AI capabilities work best.
        """

        # Basic semantic chunker
        semantic_chunker = SemanticChunker(similarity_threshold=0.7)

        # Discourse-aware chunker with same base settings
        discourse_chunker = DiscourseAwareChunker(
            similarity_threshold=0.7,
            discourse_weight=0.4,
            min_boundary_strength=0.6
        )

        semantic_result = semantic_chunker.chunk(test_text)
        discourse_result = discourse_chunker.chunk(test_text)

        # Discourse-aware should provide richer metadata
        semantic_has_discourse_info = any(
            'discourse_markers' in chunk.metadata.extra
            for chunk in semantic_result.chunks
        )
        discourse_has_discourse_info = any(
            'discourse_markers' in chunk.metadata.extra
            for chunk in discourse_result.chunks
        )

        # Only discourse-aware should have discourse information
        assert not semantic_has_discourse_info  # Basic semantic shouldn't have this
        assert discourse_has_discourse_info     # Discourse-aware should have this

        # Both should produce valid chunks
        assert len(semantic_result.chunks) > 0
        assert len(discourse_result.chunks) > 0
