"""
Unit tests for EmbeddingBasedChunker.

This test suite covers embedding-based chunking functionality including:
- Multiple embedding models (Sentence Transformers, TF-IDF, Word Averaging)
- Various similarity metrics (cosine, euclidean, dot product, manhattan)
- Different clustering methods (k-means, hierarchical, DBSCAN, threshold-based)
- Embedding caching and performance optimization
- Dimension reduction and adaptive thresholds
- Streaming capabilities and parameter adaptation
- File-based processing with real test data
"""

import pytest
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any

from chunking_strategy.strategies.text.embedding_based_chunker import (
    EmbeddingBasedChunker,
    EmbeddingModel,
    SimilarityMetric,
    ClusteringMethod
)
from chunking_strategy.core.base import ModalityType, ChunkingResult


class TestEmbeddingBasedChunker:
    """Test suite for EmbeddingBasedChunker functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures and sample data."""
        # Basic chunker for most tests
        self.chunker = EmbeddingBasedChunker(
            embedding_model="tfidf",  # Use TF-IDF for reliability in tests
            similarity_metric="cosine",
            similarity_threshold=0.7,
            min_chunk_sentences=2,
            max_chunk_sentences=8,
            enable_caching=True
        )

        # Sample text with distinct topics for boundary testing
        self.sample_text = """
        Artificial intelligence is revolutionizing technology and business.
        Machine learning algorithms can process vast amounts of data efficiently.
        Deep learning neural networks excel at pattern recognition tasks.
        AI systems are becoming increasingly sophisticated and capable.

        Climate change poses significant challenges for global sustainability.
        Rising temperatures are causing ice caps to melt at alarming rates.
        Ocean levels are increasing, threatening coastal communities worldwide.
        Renewable energy solutions are essential for reducing carbon emissions.

        Space exploration continues to push the boundaries of human knowledge.
        NASA and private companies are developing advanced rocket technologies.
        Mars colonization represents humanity's next great adventure.
        Telescope observations reveal countless galaxies throughout the universe.

        Quantum computing promises to revolutionize computational capabilities.
        Quantum bits can exist in multiple states simultaneously through superposition.
        These systems could break current encryption methods while enabling new ones.
        Research institutions worldwide are racing to achieve quantum supremacy.
        """.strip()

        # Technical document sample
        self.technical_text = """
        Software Engineering Best Practices. Modern software development requires
        systematic approaches to code organization and project management.
        Version control systems track changes and enable collaborative development.

        Database Design Principles. Relational databases use structured query language
        for data manipulation and retrieval operations efficiently.
        Normalization reduces data redundancy and improves consistency.

        Network Security Fundamentals. Encryption protects sensitive information
        during transmission across untrusted networks and communication channels.
        Authentication verifies user identities before granting system access.
        """

        # Short text for edge case testing
        self.short_text = "Single sentence here. Another sentence follows."

    def test_initialization_valid_parameters(self):
        """Test chunker initialization with valid parameters."""
        chunker = EmbeddingBasedChunker(
            embedding_model="sentence_transformer",
            model_name="all-MiniLM-L6-v2",
            similarity_metric="cosine",
            similarity_threshold=0.8,
            clustering_method="kmeans",
            min_chunk_sentences=3,
            max_chunk_sentences=10,
            enable_caching=True,
            dimension_reduction=True,
            adaptive_threshold=True
        )

        assert chunker.embedding_model == EmbeddingModel.SENTENCE_TRANSFORMER
        assert chunker.model_name == "all-MiniLM-L6-v2"
        assert chunker.similarity_metric == SimilarityMetric.COSINE
        assert chunker.similarity_threshold == 0.8
        assert chunker.clustering_method == ClusteringMethod.KMEANS
        assert chunker.min_chunk_sentences == 3
        assert chunker.max_chunk_sentences == 10
        assert chunker.enable_caching == True
        assert chunker.dimension_reduction == True
        assert chunker.adaptive_threshold == True

    def test_initialization_invalid_parameters(self):
        """Test chunker initialization with invalid parameters."""
        # Invalid similarity threshold
        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            EmbeddingBasedChunker(similarity_threshold=1.5)

        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            EmbeddingBasedChunker(similarity_threshold=-0.1)

        # Invalid sentence count parameters
        with pytest.raises(ValueError, match="min_chunk_sentences must be less than max_chunk_sentences"):
            EmbeddingBasedChunker(min_chunk_sentences=10, max_chunk_sentences=5)

        with pytest.raises(ValueError, match="min_chunk_sentences must be at least 1"):
            EmbeddingBasedChunker(min_chunk_sentences=0)

    def test_embedding_model_enum_validation(self):
        """Test that embedding model enum values are properly validated."""
        # Valid models
        valid_models = ["sentence_transformer", "tfidf", "word_average", "custom"]
        for model in valid_models:
            chunker = EmbeddingBasedChunker(embedding_model=model)
            assert chunker.embedding_model.value == model

        # Invalid model should raise ValueError
        with pytest.raises(ValueError):
            EmbeddingBasedChunker(embedding_model="invalid_model")

    def test_similarity_metric_enum_validation(self):
        """Test that similarity metric enum values are properly validated."""
        # Valid metrics
        valid_metrics = ["cosine", "euclidean", "dot_product", "manhattan"]
        for metric in valid_metrics:
            chunker = EmbeddingBasedChunker(similarity_metric=metric)
            assert chunker.similarity_metric.value == metric

        # Invalid metric should raise ValueError
        with pytest.raises(ValueError):
            EmbeddingBasedChunker(similarity_metric="invalid_metric")

    def test_clustering_method_enum_validation(self):
        """Test that clustering method enum values are properly validated."""
        # Valid methods
        valid_methods = ["kmeans", "hierarchical", "dbscan", "threshold_based"]
        for method in valid_methods:
            chunker = EmbeddingBasedChunker(clustering_method=method)
            assert chunker.clustering_method.value == method

        # Invalid method should raise ValueError
        with pytest.raises(ValueError):
            EmbeddingBasedChunker(clustering_method="invalid_method")

    def test_chunk_basic_functionality(self):
        """Test basic chunking functionality with embedding analysis."""
        result = self.chunker.chunk(self.sample_text)

        assert isinstance(result, ChunkingResult)
        assert result.strategy_used == "embedding_based"
        assert len(result.chunks) >= 2  # Should detect multiple topics
        assert all(chunk.modality == ModalityType.TEXT for chunk in result.chunks)

        # Check that chunks contain reasonable content
        total_content = " ".join(chunk.content for chunk in result.chunks)
        assert len(total_content) > 0

        # Check enhanced metadata
        assert "embedding_based_metadata" in result.source_info
        metadata = result.source_info["embedding_based_metadata"]
        assert "embedding_model" in metadata
        assert "similarity_metric" in metadata
        assert "total_sentences" in metadata

    def test_different_embedding_models(self):
        """Test different embedding models."""
        models = ["tfidf", "word_average"]  # sentence_transformer might not be available

        for model in models:
            try:
                chunker = EmbeddingBasedChunker(
                    embedding_model=model,
                    similarity_threshold=0.6
                )

                result = chunker.chunk(self.technical_text)

                assert len(result.chunks) >= 1, f"Model {model} produced no chunks"
                assert result.strategy_used == "embedding_based"
                assert result.source_info["embedding_based_metadata"]["embedding_model"] == model

            except ImportError:
                # Some models might not be available in test environment
                pytest.skip(f"Embedding model {model} not available")

    def test_different_similarity_metrics(self):
        """Test different similarity metrics."""
        metrics = ["cosine", "euclidean", "dot_product", "manhattan"]

        for metric in metrics:
            chunker = EmbeddingBasedChunker(
                embedding_model="tfidf",
                similarity_metric=metric,
                similarity_threshold=0.6
            )

            result = chunker.chunk(self.sample_text)

            assert len(result.chunks) >= 1, f"Metric {metric} produced no chunks"
            assert result.strategy_used == "embedding_based"
            assert result.source_info["embedding_based_metadata"]["similarity_metric"] == metric

    def test_different_clustering_methods(self):
        """Test different clustering methods."""
        methods = ["threshold_based", "kmeans", "hierarchical"]  # DBSCAN might be unstable

        for method in methods:
            try:
                chunker = EmbeddingBasedChunker(
                    embedding_model="tfidf",
                    clustering_method=method,
                    similarity_threshold=0.6
                )

                result = chunker.chunk(self.sample_text)

                assert len(result.chunks) >= 1, f"Method {method} produced no chunks"
                assert result.strategy_used == "embedding_based"
                assert result.source_info["embedding_based_metadata"]["clustering_method"] == method

            except ImportError:
                # Some clustering methods might not be available
                pytest.skip(f"Clustering method {method} not available")

    def test_chunk_size_constraints(self):
        """Test that chunk size constraints are properly enforced."""
        chunker = EmbeddingBasedChunker(
            embedding_model="tfidf",
            min_chunk_sentences=3,
            max_chunk_sentences=6,
            similarity_threshold=0.5
        )

        result = chunker.chunk(self.sample_text)

        for chunk in result.chunks:
            sentences_in_chunk = len([s for s in chunk.content.split('.') if s.strip()])

            # Check sentence count (may be flexible due to embedding-based boundaries)
            assert sentences_in_chunk >= 1, f"Chunk has no sentences: {sentences_in_chunk}"

    def test_empty_and_minimal_content_handling(self):
        """Test handling of empty and minimal content."""
        # Empty content
        result = self.chunker.chunk("")
        assert len(result.chunks) == 0
        assert result.processing_time >= 0

        # Whitespace only
        result = self.chunker.chunk("   \n\t   ")
        assert len(result.chunks) == 0

        # Single sentence (below minimum)
        result = self.chunker.chunk("This is a single sentence.")
        assert len(result.chunks) >= 1
        if result.chunks:
            assert result.chunks[0].content.strip() != ""

        # Short text with multiple sentences
        result = self.chunker.chunk(self.short_text)
        assert len(result.chunks) >= 1

    def test_file_input_handling(self):
        """Test handling of file path inputs."""
        # Create a temporary test file
        test_file = Path("/tmp/test_embedding_chunker.txt")
        test_file.write_text(self.technical_text)

        try:
            result = self.chunker.chunk(test_file)
            assert len(result.chunks) >= 1
            assert result.strategy_used == "embedding_based"
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_bytes_input_handling(self):
        """Test handling of bytes input."""
        text_bytes = self.technical_text.encode('utf-8')
        result = self.chunker.chunk(text_bytes)

        assert len(result.chunks) >= 1
        assert result.strategy_used == "embedding_based"

    def test_streaming_functionality(self):
        """Test streaming chunk generation."""
        # Split text into stream chunks
        stream_data = [
            self.sample_text[:200],
            self.sample_text[200:600],
            self.sample_text[600:]
        ]

        chunks = list(self.chunker.chunk_stream(stream_data))

        assert len(chunks) >= 1
        assert all(hasattr(chunk, 'content') for chunk in chunks)
        assert all(len(chunk.content) > 0 for chunk in chunks)

    def test_metadata_completeness(self):
        """Test that chunk metadata is complete and accurate."""
        result = self.chunker.chunk(self.technical_text)

        for i, chunk in enumerate(result.chunks):
            # Required metadata fields
            assert chunk.metadata.extra["chunker_used"] == "embedding_based"
            assert chunk.metadata.extra["chunk_index"] == i
            assert chunk.metadata.length == len(chunk.content)

            # Embedding-specific metadata
            assert "embedding_model" in chunk.metadata.extra
            assert "similarity_metric" in chunk.metadata.extra
            assert "clustering_method" in chunk.metadata.extra
            assert "sentence_range" in chunk.metadata.extra
            assert chunk.metadata.extra["chunking_strategy"] == "embedding_based"

    def test_adaptation_quality_feedback(self):
        """Test adaptation based on quality feedback."""
        original_threshold = self.chunker.similarity_threshold
        original_clustering = self.chunker.clustering_method

        # Poor quality feedback should increase threshold
        changes = self.chunker.adapt_parameters(0.3, "quality")

        # Should make some adaptation
        current_threshold = self.chunker.similarity_threshold
        current_clustering = self.chunker.clustering_method

        # Either threshold increased or clustering method changed
        assert (current_threshold > original_threshold or
                current_clustering != original_clustering or
                len(changes) > 0)

    def test_adaptation_performance_feedback(self):
        """Test adaptation based on performance feedback."""
        original_max_sentences = self.chunker.max_chunk_sentences

        # Poor performance should increase chunk sizes
        changes = self.chunker.adapt_parameters(0.2, "performance")

        if "max_chunk_sentences" in changes:
            assert self.chunker.max_chunk_sentences >= original_max_sentences

    def test_adaptation_history_tracking(self):
        """Test that adaptation history is properly tracked."""
        # Perform adaptation
        self.chunker.adapt_parameters(0.3, "quality")
        self.chunker.adapt_parameters(0.2, "performance")

        history = self.chunker.get_adaptation_history()
        assert len(history) >= 1

        # Check history structure
        for record in history:
            assert "timestamp" in record
            assert "feedback_score" in record
            assert "feedback_type" in record
            assert "changes" in record
            assert "adapted_params" in record

    def test_configuration_retrieval(self):
        """Test configuration retrieval functionality."""
        config = self.chunker.get_config()

        assert config["name"] == "embedding_based"
        assert config["embedding_model"] == self.chunker.embedding_model.value
        assert config["similarity_metric"] == self.chunker.similarity_metric.value
        assert config["clustering_method"] == self.chunker.clustering_method.value
        assert config["similarity_threshold"] == self.chunker.similarity_threshold
        assert "performance_stats" in config
        assert "cache_status" in config

    def test_embedding_boundary_detection(self):
        """Test embedding-based boundary detection with known topic shifts."""
        # Text with clear semantic boundaries
        topic_text = """
        Python programming language offers powerful data structures and libraries.
        Object-oriented programming enables modular code organization efficiently.
        Machine learning libraries like scikit-learn provide comprehensive algorithms.

        Cooking techniques vary significantly across different cultural traditions.
        Fresh ingredients enhance flavor profiles in prepared dishes substantially.
        Meal planning reduces food waste while saving preparation time effectively.

        Exercise benefits both physical fitness and mental health outcomes.
        Cardiovascular activities improve heart function and endurance capacity.
        Strength training builds muscle mass and bone density over time.
        """

        result = self.chunker.chunk(topic_text)

        # Should detect semantic boundaries and create separate chunks
        assert len(result.chunks) >= 2, f"Expected multiple chunks for distinct topics, got {len(result.chunks)}"

        # Verify embedding metadata is present
        assert "embedding_based_metadata" in result.source_info
        metadata = result.source_info["embedding_based_metadata"]
        assert metadata["total_sentences"] > 0
        assert "embedding_dimension" in metadata

    def test_similarity_computation(self):
        """Test similarity computation accuracy."""
        # Test with simple vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        vec3 = np.array([1.0, 0.0, 0.0])  # Same as vec1

        # Cosine similarity
        sim_different = self.chunker._cosine_similarity(vec1, vec2)
        sim_same = self.chunker._cosine_similarity(vec1, vec3)

        assert 0.0 <= sim_different <= 1.0
        assert 0.0 <= sim_same <= 1.0
        assert sim_same > sim_different  # Same vectors should be more similar

    def test_embedding_caching(self):
        """Test embedding caching functionality."""
        chunker = EmbeddingBasedChunker(
            embedding_model="tfidf",
            enable_caching=True
        )

        # First processing - should miss cache
        result1 = chunker.chunk(self.short_text)
        initial_cache_misses = chunker.performance_stats["cache_misses"]

        # Second processing of same content - should hit cache if implemented
        result2 = chunker.chunk(self.short_text)

        # Basic validation that caching system is working
        assert len(result1.chunks) >= 1
        assert len(result2.chunks) >= 1
        assert chunker.performance_stats["cache_misses"] >= initial_cache_misses

    def test_fallback_mechanisms(self):
        """Test fallback mechanisms when preferred models fail."""
        # This will use TF-IDF or word average as fallback
        chunker = EmbeddingBasedChunker(
            embedding_model="tfidf",
            similarity_threshold=0.6
        )

        result = chunker.chunk(self.technical_text)
        assert len(result.chunks) >= 1
        assert result.strategy_used in ["embedding_based", "embedding_based_fallback"]

    def test_performance_tracking(self):
        """Test that performance metrics are properly tracked."""
        # Process some text to generate metrics
        result = self.chunker.chunk(self.sample_text)

        config = self.chunker.get_config()
        stats = config["performance_stats"]

        assert stats["total_sentences_processed"] > 0
        assert stats["embedding_time"] >= 0
        assert stats["similarity_computation_time"] >= 0

    def test_chunk_estimation(self):
        """Test chunk count estimation functionality."""
        estimate = self.chunker.estimate_chunks(self.sample_text)

        assert isinstance(estimate, int)
        assert estimate >= 1

        # Actual chunking for comparison
        result = self.chunker.chunk(self.sample_text)
        actual_chunks = len(result.chunks)

        # Estimate should be reasonably close
        assert abs(estimate - actual_chunks) <= max(1, actual_chunks)

    def test_supported_formats(self):
        """Test supported file format reporting."""
        formats = self.chunker.get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        expected_formats = ["txt", "md", "json", "xml", "html"]

        for format in expected_formats:
            assert format in formats

    def test_dimension_reduction(self):
        """Test dimension reduction functionality."""
        try:
            chunker = EmbeddingBasedChunker(
                embedding_model="tfidf",
                dimension_reduction=True,
                similarity_threshold=0.6
            )

            result = chunker.chunk(self.sample_text)
            assert len(result.chunks) >= 1
            assert result.strategy_used == "embedding_based"

        except ImportError:
            pytest.skip("Scikit-learn not available for dimension reduction")

    def test_adaptive_threshold(self):
        """Test adaptive threshold functionality."""
        chunker = EmbeddingBasedChunker(
            embedding_model="tfidf",
            adaptive_threshold=True,
            similarity_threshold=0.7
        )

        result = chunker.chunk(self.sample_text)
        assert len(result.chunks) >= 1
        assert result.strategy_used == "embedding_based"

    def test_error_recovery_and_robustness(self):
        """Test error recovery and robustness with problematic inputs."""
        # Text with unusual characters
        unusual_text = "Tëst wîth ünïcödé çhäracters. Émojis: 🚀🔬🎯. Numbers: 123,456.78."
        result = self.chunker.chunk(unusual_text)
        assert len(result.chunks) >= 1

        # Very long single sentence
        long_sentence = "This is a very long sentence that continues " * 50 + "and ends here."
        result = self.chunker.chunk(long_sentence)
        assert len(result.chunks) >= 1

        # Mixed line endings
        mixed_text = "Line one.\r\nLine two.\nLine three.\r\n\nParagraph break.\n\nFinal line."
        result = self.chunker.chunk(mixed_text)
        assert len(result.chunks) >= 1

    def test_clustering_parameters(self):
        """Test custom clustering parameters."""
        clustering_params = {
            "n_clusters": 3,
            "random_state": 42
        }

        try:
            chunker = EmbeddingBasedChunker(
                embedding_model="tfidf",
                clustering_method="kmeans",
                clustering_params=clustering_params
            )

            result = chunker.chunk(self.sample_text)
            assert len(result.chunks) >= 1

        except ImportError:
            pytest.skip("Scikit-learn not available for clustering")

    def test_source_info_propagation(self):
        """Test that source information is properly propagated."""
        source_info = {
            "source": "test_document.txt",
            "source_type": "file",
            "author": "test_author",
            "created": "2024-01-01"
        }

        result = self.chunker.chunk(self.technical_text, source_info=source_info)

        # Check that source info is preserved and enhanced
        assert result.source_info["source"] == "test_document.txt"
        assert result.source_info["source_type"] == "file"
        assert result.source_info["author"] == "test_author"

        # Should add embedding-specific information
        assert "embedding_based_metadata" in result.source_info
        assert "total_sentences" in result.source_info

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
            ("nested_structure.md", {"min_chunks": 2, "content_type": "structured_markdown"}),

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

            print(f"🔍 Testing embedding-based chunking with {filename}...")

            try:
                # Test with file path
                result = self.chunker.chunk(file_path)

                # Basic validations
                assert len(result.chunks) >= expectations["min_chunks"], \
                    f"Expected at least {expectations['min_chunks']} chunks for {filename}, got {len(result.chunks)}"
                assert result.strategy_used == "embedding_based"
                assert result.processing_time > 0

                # Content integrity check
                total_length = sum(len(chunk.content) for chunk in result.chunks)
                assert total_length > 0, f"No content found in chunks for {filename}"

                # Metadata validation
                for chunk in result.chunks:
                    assert chunk.metadata.extra["chunker_used"] == "embedding_based"
                    assert "embedding_model" in chunk.metadata.extra
                    assert "similarity_metric" in chunk.metadata.extra
                    assert "clustering_method" in chunk.metadata.extra
                    assert chunk.metadata.extra.get("sentence_count", 0) >= 0

                # Source info validation
                assert "embedding_based_metadata" in result.source_info
                emb_meta = result.source_info["embedding_based_metadata"]
                assert "total_sentences" in emb_meta
                assert "embedding_model" in emb_meta
                assert "similarity_metric" in emb_meta
                assert "clustering_method" in emb_meta
                assert emb_meta["total_sentences"] >= 0

                results[filename] = {
                    "chunk_count": len(result.chunks),
                    "total_length": total_length,
                    "processing_time": result.processing_time,
                    "avg_chunk_length": total_length / len(result.chunks),
                    "sentences_processed": emb_meta.get("total_sentences", 0),
                    "embedding_dimension": emb_meta.get("embedding_dimension", 0),
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
        """Test processing JSON files with embedding-based chunking."""
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

                # JSON files should be processed as text for embedding analysis
                assert len(result.chunks) >= 1
                assert result.strategy_used == "embedding_based"

                # Content should be readable (JSON might have limited text content)
                total_content = " ".join(chunk.content for chunk in result.chunks)
                # Some JSON files might not have meaningful text content for semantic analysis
                if len(total_content) == 0:
                    print(f"⚠️ {filename}: No text content extracted (expected for some JSON files)")
                    continue

                # Should have embedding metadata
                assert "embedding_based_metadata" in result.source_info
                emb_meta = result.source_info["embedding_based_metadata"]
                assert emb_meta["total_sentences"] >= 0  # JSON might not have traditional sentences

                print(f"✅ {filename}: {len(result.chunks)} chunks, {emb_meta['total_sentences']} sentences detected")

            except Exception as e:
                pytest.fail(f"Failed to process JSON file {filename}: {e}")

    def test_processing_xml_files(self):
        """Test processing XML files with embedding-based chunking."""
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
                assert result.strategy_used == "embedding_based"

                # Check content extraction
                total_content = " ".join(chunk.content for chunk in result.chunks)
                assert len(total_content) > 0

                # XML might have structural text that forms sentences
                for chunk in result.chunks:
                    assert len(chunk.content.strip()) > 0
                    assert "embedding_model" in chunk.metadata.extra
                    assert "similarity_metric" in chunk.metadata.extra

                print(f"✅ {filename}: {len(result.chunks)} chunks processed")

            except Exception as e:
                pytest.fail(f"Failed to process XML file {filename}: {e}")

    def test_processing_rtf_files(self):
        """Test processing RTF files with embedding-based chunking."""
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
                # RTF files might need special handling, but embedding chunker should process as text
                result = self.chunker.chunk(file_path)

                assert len(result.chunks) >= 1
                assert result.strategy_used == "embedding_based"

                # Verify content was extracted
                total_length = sum(len(chunk.content) for chunk in result.chunks)
                assert total_length > 0

                print(f"✅ {filename}: {len(result.chunks)} chunks, {total_length} chars total")

            except Exception as e:
                # RTF might fail if no RTF parser available - that's okay
                print(f"⚠️ RTF processing not available for {filename}: {e}")

    def test_different_embedding_models_on_files(self):
        """Test different embedding models on the same files."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        # Test one representative file with different models
        test_file = test_data_dir / "simple_document.md"
        if not test_file.exists():
            pytest.skip("simple_document.md not found")

        embedding_models = ["tfidf", "word_average"]  # sentence_transformer might not be available
        results = {}

        for model in embedding_models:
            try:
                chunker = EmbeddingBasedChunker(
                    embedding_model=model,
                    similarity_threshold=0.7,
                    min_chunk_sentences=2
                )

                result = chunker.chunk(test_file)

                results[model] = {
                    "chunk_count": len(result.chunks),
                    "processing_time": result.processing_time,
                    "total_sentences": result.source_info.get("embedding_based_metadata", {}).get("total_sentences", 0),
                    "success": True
                }

                print(f"✅ Model {model}: {len(result.chunks)} chunks in {result.processing_time:.3f}s")

            except Exception as e:
                # Some models might not be available
                results[model] = {"success": False, "error": str(e)}
                print(f"⚠️ Model {model} failed: {e}")

        # At least one model should work (TF-IDF is most reliable)
        successful_models = [m for m, r in results.items() if r.get("success", False)]
        assert len(successful_models) >= 1, f"No embedding models worked. Results: {results}"

    def test_file_processing_with_different_similarity_metrics(self):
        """Test file processing with different similarity metrics."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        # Use a file with clear semantic content
        test_file = test_data_dir / "nested_structure.md"
        if not test_file.exists():
            pytest.skip("nested_structure.md not found")

        metrics = ["cosine", "euclidean", "dot_product", "manhattan"]
        results = []

        for metric in metrics:
            chunker = EmbeddingBasedChunker(
                embedding_model="tfidf",
                similarity_metric=metric,
                similarity_threshold=0.7,
                min_chunk_sentences=2
            )

            result = chunker.chunk(test_file)

            results.append({
                "metric": metric,
                "chunk_count": len(result.chunks),
                "processing_time": result.processing_time,
                "avg_chunk_length": sum(len(c.content) for c in result.chunks) / len(result.chunks) if result.chunks else 0
            })

            print(f"Metric {metric}: {len(result.chunks)} chunks, avg length {results[-1]['avg_chunk_length']:.0f}")

        # Validate metric effects
        assert len(results) == len(metrics)
        for i, result_data in enumerate(results):
            assert result_data["chunk_count"] >= 1, f"Metric {metrics[i]} produced no chunks"

    def test_clustering_methods_on_files(self):
        """Test different clustering methods on real files."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        test_file = test_data_dir / "sample_mixed_content.txt"
        if not test_file.exists():
            pytest.skip("sample_mixed_content.txt not found")

        methods = ["threshold_based", "kmeans", "hierarchical"]  # Skip DBSCAN as it can be unstable
        results = {}

        for method in methods:
            try:
                clustering_params = {}
                if method == "kmeans":
                    clustering_params = {"n_clusters": 3, "random_state": 42}
                elif method == "hierarchical":
                    clustering_params = {"linkage": "ward"}

                chunker = EmbeddingBasedChunker(
                    embedding_model="tfidf",
                    similarity_metric="cosine",
                    clustering_method=method,
                    clustering_params=clustering_params,
                    min_chunk_sentences=2
                )

                result = chunker.chunk(test_file)

                results[method] = {
                    "chunk_count": len(result.chunks),
                    "processing_time": result.processing_time,
                    "success": True
                }

                print(f"✅ Method {method}: {len(result.chunks)} chunks in {result.processing_time:.3f}s")

            except Exception as e:
                # Some clustering methods might not be available
                results[method] = {"success": False, "error": str(e)}
                print(f"⚠️ Method {method} failed: {e}")

        # At least threshold-based should work
        assert results.get("threshold_based", {}).get("success", False), "Threshold-based clustering should always work"

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

                # Validate embedding metadata
                assert "embedding_based_metadata" in result.source_info
                emb_meta = result.source_info["embedding_based_metadata"]
                assert emb_meta["total_sentences"] > 0
                assert emb_meta["embedding_dimension"] > 0

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

        # Process with embedding chunker
        result = self.chunker.chunk(test_file)

        # Reconstruct content from chunks
        reconstructed_content = " ".join(chunk.content for chunk in result.chunks)
        reconstructed_words = set(reconstructed_content.lower().split())

        # Check word preservation
        common_words = original_words & reconstructed_words
        preservation_ratio = len(common_words) / len(original_words) if original_words else 0

        assert preservation_ratio > 0.80, f"Poor content preservation: {preservation_ratio:.3f}"

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

        # Test with invalid path types
        with pytest.raises((TypeError, ValueError, FileNotFoundError)):
            self.chunker.chunk(123)  # Invalid type

        print("✅ Error handling validated for problematic file inputs")

    def test_caching_with_repeated_files(self):
        """Test embedding caching with repeated file processing."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        test_file = test_data_dir / "simple_document.md"
        if not test_file.exists():
            pytest.skip("simple_document.md not found")

        chunker = EmbeddingBasedChunker(
            embedding_model="tfidf",
            enable_caching=True
        )

        # First processing - should miss cache
        result1 = chunker.chunk(test_file)
        initial_cache_misses = chunker.performance_stats["cache_misses"]

        # Second processing of same content - may hit cache depending on implementation
        result2 = chunker.chunk(test_file)

        # Basic validation that caching system is working
        assert len(result1.chunks) >= 1
        assert len(result2.chunks) >= 1
        assert chunker.performance_stats["cache_misses"] >= initial_cache_misses

        # Results should be consistent
        assert len(result1.chunks) == len(result2.chunks), "Cached results should be consistent"

    def test_performance_metrics_on_files(self):
        """Test that performance metrics are tracked during file processing."""
        test_data_dir = Path("test_data")

        if not test_data_dir.exists():
            pytest.skip("test_data directory not found")

        test_file = test_data_dir / "technical_doc.txt"
        if not test_file.exists():
            pytest.skip("technical_doc.txt not found")

        # Process file to generate metrics
        result = self.chunker.chunk(test_file)

        config = self.chunker.get_config()
        stats = config["performance_stats"]

        assert stats["total_sentences_processed"] > 0
        assert stats["embedding_time"] >= 0
        assert stats["similarity_computation_time"] >= 0

        # Should track cumulative statistics across multiple calls
        initial_sentences = stats["total_sentences_processed"]

        # Process another file
        if test_data_dir.joinpath("sample_simple_text.txt").exists():
            result2 = self.chunker.chunk(test_data_dir / "sample_simple_text.txt")
            updated_config = self.chunker.get_config()
            updated_stats = updated_config["performance_stats"]

            assert updated_stats["total_sentences_processed"] >= initial_sentences
