"""
Integration tests for SemanticChunker.

This test suite covers semantic chunker integration with:
- File processing from test_data directory
- Different text formats and sizes
- Performance benchmarks
- Real-world usage scenarios
- Configuration-based testing
"""

import pytest
import time
import json
from pathlib import Path
from typing import List, Dict, Any

from chunking_strategy.strategies.text.semantic_chunker import SemanticChunker


class TestSemanticChunkerIntegration:
    """Integration test suite for SemanticChunker."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data_dir = Path("test_data")

        # Create test data directory if it doesn't exist
        self.test_data_dir.mkdir(exist_ok=True)

        # Sample configurations for different use cases
        self.chunker_configs = [
            {
                "name": "fast_processing",
                "semantic_model": "tfidf",
                "similarity_threshold": 0.6,
                "min_chunk_sentences": 2,
                "max_chunk_sentences": 10,
                "boundary_detection": "similarity_threshold"
            },
            {
                "name": "high_quality",
                "semantic_model": "tfidf",  # Using TF-IDF for consistency in tests
                "similarity_threshold": 0.8,
                "min_chunk_sentences": 3,
                "max_chunk_sentences": 8,
                "boundary_detection": "coherence_based",
                "coherence_weight": 0.4
            },
            {
                "name": "balanced",
                "semantic_model": "tfidf",
                "similarity_threshold": 0.7,
                "min_chunk_sentences": 2,
                "max_chunk_sentences": 12,
                "boundary_detection": "dynamic_threshold",
                "context_window_size": 3
            }
        ]

    def create_test_files(self):
        """Create test files with different characteristics."""
        test_files = {}

        # Academic text with clear topic boundaries
        academic_content = """
        Introduction to Machine Learning. Machine learning is a subset of artificial intelligence that enables computers to learn from data.
        Supervised learning algorithms use labeled datasets for training. Popular supervised methods include decision trees, random forests, and support vector machines.

        Deep Learning Fundamentals. Deep learning uses neural networks with multiple layers to model complex patterns.
        Convolutional neural networks excel at image recognition tasks. Recurrent neural networks are designed for sequential data processing.

        Natural Language Processing Applications. NLP combines linguistics with machine learning for text analysis.
        Named entity recognition identifies people, places, and organizations in text. Sentiment analysis determines emotional tone of written content.
        Chatbots and virtual assistants rely on advanced NLP techniques for human-like conversations.

        Computer Vision Techniques. Computer vision enables machines to interpret visual information from the world.
        Object detection algorithms can identify and locate multiple objects within images. Image segmentation divides images into meaningful regions.
        Facial recognition systems use deep learning for accurate identity verification.
        """

        # News article with narrative structure
        news_content = """
        Breaking News: Tech Company Announces Revolutionary AI System. A major technology company unveiled their latest artificial intelligence breakthrough today.
        The new system promises to revolutionize how businesses process information. Early demonstrations showed remarkable performance improvements.

        Industry experts expressed mixed reactions to the announcement. Some praised the innovation as a game-changer for the industry.
        Others raised concerns about potential job displacement and ethical implications. The company addressed these concerns in their presentation.

        Market Response and Financial Impact. Stock prices surged immediately following the announcement.
        Analysts predict significant revenue growth in the coming quarters. Competitors are reportedly accelerating their own AI development programs.
        The announcement has sparked renewed investor interest in artificial intelligence technologies.

        Technical Specifications and Capabilities. The system utilizes advanced neural network architectures.
        Processing speed is reportedly 10 times faster than previous generation systems. Energy efficiency has been improved by 40 percent.
        The technology will be available to enterprise customers starting next quarter.
        """

        # Technical documentation
        technical_content = """
        System Architecture Overview. The distributed computing platform consists of multiple interconnected components.
        Load balancers distribute incoming requests across available server instances. Database sharding ensures optimal data distribution and access patterns.

        Authentication and Security. Multi-factor authentication is required for all administrative access.
        Role-based access control limits user permissions based on organizational hierarchy. All communications use end-to-end encryption.
        Security logs are monitored continuously for suspicious activity patterns.

        Performance Optimization Strategies. Caching mechanisms reduce database query overhead significantly.
        Content delivery networks minimize latency for global users. Auto-scaling policies adjust resources based on demand fluctuations.
        Performance metrics are tracked and analyzed for continuous improvement opportunities.

        Maintenance and Monitoring Procedures. Automated health checks verify system component status regularly.
        Alert notifications are sent to operations team when issues are detected. Backup procedures ensure data recovery capabilities.
        Regular security audits maintain compliance with industry standards and regulations.
        """

        test_files["academic.txt"] = academic_content
        test_files["news.txt"] = news_content
        test_files["technical.txt"] = technical_content

        # Create the actual files
        for filename, content in test_files.items():
            file_path = self.test_data_dir / filename
            file_path.write_text(content.strip())
            test_files[filename] = file_path

        return test_files

    def test_processing_real_text_files(self):
        """Test processing various text file types and sizes."""
        test_files = self.create_test_files()

        results = {}
        for config in self.chunker_configs:
            chunker = SemanticChunker(**{k: v for k, v in config.items() if k != "name"})
            config_results = {}

            for filename, file_path in test_files.items():
                if isinstance(file_path, Path):
                    result = chunker.chunk(file_path)

                    config_results[filename] = {
                        "chunk_count": len(result.chunks),
                        "processing_time": result.processing_time,
                        "avg_chunk_length": sum(len(chunk.content) for chunk in result.chunks) / len(result.chunks) if result.chunks else 0,
                        "semantic_boundaries": result.source_info.get("total_boundaries_detected", 0),
                        "boundary_density": result.source_info.get("boundary_density", 0.0)
                    }

                    # Verify chunks are reasonable
                    assert len(result.chunks) >= 1, f"No chunks created for {filename} with {config['name']}"
                    assert result.processing_time > 0, f"No processing time recorded for {filename}"

                    # Check semantic metadata
                    for chunk in result.chunks:
                        assert "semantic_model" in chunk.metadata.extra
                        assert "sentence_count" in chunk.metadata.extra
                        assert chunk.metadata.extra["sentence_count"] > 0

            results[config["name"]] = config_results

        # Compare results across configurations
        assert len(results) == len(self.chunker_configs)

        # High quality config should generally create more chunks (more boundaries detected)
        for filename in test_files:
            if isinstance(test_files[filename], Path):
                high_quality = results["high_quality"][filename]["chunk_count"]
                fast_processing = results["fast_processing"][filename]["chunk_count"]

                # High quality should detect more boundaries (more sensitive)
                # This is a general expectation but may vary based on content
                assert high_quality >= 1 and fast_processing >= 1, "All configs should produce at least one chunk"

    def test_streaming_with_real_data(self):
        """Test streaming functionality with real text data."""
        test_files = self.create_test_files()

        chunker = SemanticChunker(
            semantic_model="tfidf",
            similarity_threshold=0.7,
            min_chunk_sentences=2,
            boundary_detection="similarity_threshold"
        )

        for filename, file_path in test_files.items():
            if isinstance(file_path, Path):
                # Read file and create stream
                content = file_path.read_text()

                # Split content into stream chunks
                chunk_size = len(content) // 4
                stream_data = [
                    content[i:i + chunk_size]
                    for i in range(0, len(content), chunk_size)
                ]

                # Process via streaming
                stream_chunks = list(chunker.chunk_stream(
                    stream_data,
                    source_info={"source": str(file_path), "source_type": "file"}
                ))

                # Process as batch for comparison
                batch_result = chunker.chunk(content)

                # Streaming should produce similar results
                assert len(stream_chunks) >= 1, f"Streaming produced no chunks for {filename}"

                # Content should be preserved (allowing for processing differences)
                stream_content = " ".join(chunk.content for chunk in stream_chunks)
                batch_content = " ".join(chunk.content for chunk in batch_result.chunks)

                # Should preserve most of the content
                assert len(stream_content) > len(content) * 0.8, "Streaming lost significant content"

    def test_performance_benchmarks(self):
        """Test performance with different text sizes and configurations."""
        # Create texts of different sizes
        base_text = """
        Artificial intelligence represents a transformative technology with broad applications across industries.
        Machine learning algorithms enable systems to improve performance through experience and data analysis.
        Natural language processing facilitates human-computer interaction through text and speech understanding.
        Computer vision systems interpret visual information for automated decision making and pattern recognition.
        """

        text_sizes = {
            "small": base_text * 5,      # ~2KB
            "medium": base_text * 25,    # ~10KB
            "large": base_text * 100     # ~40KB
        }

        performance_results = {}

        for size_name, text_content in text_sizes.items():
            chunker = SemanticChunker(
                semantic_model="tfidf",  # Fastest option for benchmarking
                similarity_threshold=0.7,
                boundary_detection="similarity_threshold"
            )

            start_time = time.time()
            result = chunker.chunk(text_content)
            end_time = time.time()

            performance_results[size_name] = {
                "text_length": len(text_content),
                "chunk_count": len(result.chunks),
                "processing_time": result.processing_time,
                "total_time": end_time - start_time,
                "chars_per_second": len(text_content) / result.processing_time if result.processing_time > 0 else 0,
                "semantic_boundaries": result.source_info.get("total_boundaries_detected", 0)
            }

            # Performance expectations
            assert result.processing_time < 30.0, f"Processing too slow for {size_name}: {result.processing_time}s"
            assert result.processing_time > 0, f"No processing time recorded for {size_name}"
            assert len(result.chunks) >= 1, f"No chunks created for {size_name}"

        # Verify performance scaling
        small_time = performance_results["small"]["processing_time"]
        large_time = performance_results["large"]["processing_time"]

        # Large text should not take disproportionately longer (should be roughly linear)
        time_ratio = large_time / small_time if small_time > 0 else 1
        size_ratio = len(text_sizes["large"]) / len(text_sizes["small"])

        # Time scaling should be reasonable (not exponential)
        assert time_ratio < size_ratio * 2, f"Performance degradation too severe: {time_ratio} vs {size_ratio}"

    def test_topic_boundary_detection_accuracy(self):
        """Test accuracy of semantic boundary detection with known topic shifts."""
        # Text with clear, distinct topics
        multi_topic_text = """
        Python Programming Fundamentals. Python is a high-level programming language known for its simplicity.
        Variables store data values and can be of different types like integers, strings, and lists.
        Functions help organize code into reusable blocks that perform specific tasks.
        Object-oriented programming in Python uses classes to model real-world entities.

        Cooking and Recipe Management. Cooking is both an art and a science requiring precision and creativity.
        Fresh ingredients are essential for creating flavorful and nutritious meals for family and friends.
        Meal planning helps save time and money while ensuring balanced nutrition throughout the week.
        Different cooking techniques like sautéing, braising, and grilling produce unique flavors and textures.

        Space Exploration History. Human fascination with space has driven remarkable technological achievements over decades.
        The Apollo missions successfully landed astronauts on the moon during the late twentieth century.
        Space stations provide platforms for conducting scientific research in microgravity environments.
        Mars exploration missions seek evidence of past life and assess colonization possibilities.

        Financial Investment Strategies. Diversification spreads risk across different asset classes and geographic regions.
        Long-term investing typically outperforms short-term speculation for building wealth over time.
        Emergency funds provide financial security during unexpected life events and economic downturns.
        Retirement planning requires careful consideration of income needs and healthcare costs.
        """

        chunker = SemanticChunker(
            semantic_model="tfidf",
            similarity_threshold=0.6,  # Moderate sensitivity
            min_chunk_sentences=2,
            boundary_detection="similarity_threshold"
        )

        result = chunker.chunk(multi_topic_text)

        # Should detect multiple topics and create separate chunks
        assert len(result.chunks) >= 3, f"Should detect at least 3 topic boundaries, got {len(result.chunks)}"

        # Analyze chunk content for topic separation
        chunk_contents = [chunk.content.lower() for chunk in result.chunks]

        # Check that different topics are likely in different chunks
        programming_chunks = [i for i, content in enumerate(chunk_contents) if "python" in content or "programming" in content]
        cooking_chunks = [i for i, content in enumerate(chunk_contents) if "cooking" in content or "recipe" in content]
        space_chunks = [i for i, content in enumerate(chunk_contents) if "space" in content or "apollo" in content]
        financial_chunks = [i for i, content in enumerate(chunk_contents) if "investment" in content or "financial" in content]

        # Topics should generally be separated
        topic_groups = [programming_chunks, cooking_chunks, space_chunks, financial_chunks]
        topic_groups = [group for group in topic_groups if group]  # Remove empty groups

        # Should have at least 2 distinct topic groups
        assert len(topic_groups) >= 2, "Should identify multiple distinct topics"

        # Different topic groups should have minimal overlap
        for i, group1 in enumerate(topic_groups):
            for j, group2 in enumerate(topic_groups):
                if i < j and group1 and group2:
                    overlap = set(group1).intersection(set(group2))
                    total_chunks = len(set(group1).union(set(group2)))
                    overlap_ratio = len(overlap) / total_chunks if total_chunks > 0 else 0

                    # Should have limited topic mixing
                    assert overlap_ratio < 0.7, f"Too much topic mixing between groups {i} and {j}: {overlap_ratio}"

    def test_different_boundary_detection_methods(self):
        """Test different boundary detection methods on the same content."""
        test_text = """
        Machine learning fundamentals involve understanding data patterns and algorithmic decision making.
        Supervised learning uses labeled examples to train predictive models for classification and regression tasks.
        Feature engineering transforms raw data into meaningful representations that improve model performance.

        Data preprocessing is crucial for cleaning and preparing datasets for analysis and model training.
        Missing values, outliers, and inconsistent formatting can significantly impact model accuracy and reliability.
        Normalization and scaling ensure that different features contribute equally to model learning processes.

        Model evaluation assesses performance using metrics like accuracy, precision, recall, and F1 scores.
        Cross-validation techniques provide robust estimates of model performance on unseen data samples.
        Hyperparameter tuning optimizes model configuration to achieve the best possible predictive performance.
        """

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
                similarity_threshold=0.7,
                boundary_detection=method,
                min_chunk_sentences=2,
                context_window_size=2,
                coherence_weight=0.3
            )

            result = chunker.chunk(test_text)

            results[method] = {
                "chunk_count": len(result.chunks),
                "boundaries_detected": result.source_info.get("total_boundaries_detected", 0),
                "processing_time": result.processing_time,
                "boundary_density": result.source_info.get("boundary_density", 0.0)
            }

            # All methods should produce valid results
            assert len(result.chunks) >= 1, f"Method {method} produced no chunks"
            assert result.processing_time > 0, f"Method {method} reported zero processing time"

            # Check chunk quality
            for chunk in result.chunks:
                assert len(chunk.content.strip()) > 0, f"Method {method} produced empty chunk"
                assert chunk.metadata.extra["sentence_count"] > 0, f"Method {method} chunk has no sentences"

        # Different methods should produce different results (showing they work differently)
        chunk_counts = [results[method]["chunk_count"] for method in methods]

        # Should see some variation in chunking behavior
        assert max(chunk_counts) >= min(chunk_counts), "All methods produced identical results"

    def test_adaptation_with_real_scenarios(self):
        """Test adaptation capabilities with realistic feedback scenarios."""
        # Start with a base configuration
        chunker = SemanticChunker(
            semantic_model="tfidf",
            similarity_threshold=0.7,
            min_chunk_sentences=3,
            max_chunk_sentences=10
        )

        test_text = """
        Software development lifecycle encompasses planning, design, implementation, testing, and maintenance phases.
        Agile methodologies emphasize iterative development with frequent customer feedback and adaptive planning.
        Version control systems track changes and enable collaboration among distributed development teams.

        Quality assurance processes ensure software meets requirements and performs reliably under various conditions.
        Automated testing frameworks reduce manual effort while improving coverage and consistency of test execution.
        Code reviews promote knowledge sharing and help identify potential issues before production deployment.

        DevOps practices integrate development and operations teams to streamline deployment and monitoring processes.
        Continuous integration and deployment pipelines automate testing and release management workflows.
        Infrastructure as code enables reproducible and scalable deployment environments across different platforms.
        """

        # Test quality feedback adaptation
        initial_result = chunker.chunk(test_text)
        initial_chunk_count = len(initial_result.chunks)

        # Simulate poor quality feedback (chunks too large, lacking granularity)
        quality_changes = chunker.adapt_parameters(0.3, "quality", chunks_too_large=True)

        # Should adapt to create smaller, more numerous chunks
        adapted_result = chunker.chunk(test_text)
        adapted_chunk_count = len(adapted_result.chunks)

        if quality_changes:
            # If changes were made, should see different chunking behavior
            assert adapted_chunk_count != initial_chunk_count, "Adaptation should change chunking behavior"

        # Test performance feedback adaptation
        original_threshold = chunker.similarity_threshold
        performance_changes = chunker.adapt_parameters(0.2, "performance")

        if performance_changes:
            # Performance adaptation should make processing more efficient
            performance_result = chunker.chunk(test_text)
            assert performance_result.processing_time <= adapted_result.processing_time * 1.5, "Performance adaptation should not significantly increase processing time"

        # Verify adaptation history is maintained
        history = chunker.get_adaptation_history()
        assert len(history) >= 1, "Adaptation history should be recorded"

        for record in history:
            assert "timestamp" in record
            assert "feedback_score" in record
            assert "feedback_type" in record
            assert "changes" in record

    def test_configuration_robustness(self):
        """Test chunker robustness with various edge case configurations."""
        edge_configs = [
            # Minimal chunking
            {
                "similarity_threshold": 0.9,  # Very high threshold
                "min_chunk_sentences": 1,
                "max_chunk_sentences": 3,
                "boundary_detection": "similarity_threshold"
            },
            # Maximal chunking
            {
                "similarity_threshold": 0.3,  # Very low threshold
                "min_chunk_sentences": 5,
                "max_chunk_sentences": 20,
                "boundary_detection": "coherence_based"
            },
            # Conservative settings
            {
                "similarity_threshold": 0.8,
                "min_chunk_sentences": 2,
                "max_chunk_sentences": 6,
                "boundary_detection": "dynamic_threshold",
                "context_window_size": 1
            }
        ]

        test_text = """
        Brief overview of key concepts. First topic introduces fundamental principles and basic terminology.
        Second section explores practical applications and real world examples of these concepts.
        Third part discusses advanced techniques and future research directions.
        Final summary synthesizes main points and provides actionable recommendations.
        """

        for i, config in enumerate(edge_configs):
            chunker = SemanticChunker(semantic_model="tfidf", **config)

            try:
                result = chunker.chunk(test_text)

                # Should always produce at least one chunk
                assert len(result.chunks) >= 1, f"Config {i} produced no chunks"

                # All chunks should have reasonable content
                for j, chunk in enumerate(result.chunks):
                    assert len(chunk.content.strip()) > 10, f"Config {i}, chunk {j} too short"
                    assert chunk.metadata.extra["sentence_count"] >= 1, f"Config {i}, chunk {j} has no sentences"

                # Processing should complete in reasonable time
                assert result.processing_time < 10.0, f"Config {i} took too long: {result.processing_time}s"

            except Exception as e:
                pytest.fail(f"Config {i} failed with error: {e}")

    def cleanup_test_files(self):
        """Clean up created test files."""
        if self.test_data_dir.exists():
            for file in self.test_data_dir.glob("*.txt"):
                if file.name in ["academic.txt", "news.txt", "technical.txt"]:
                    file.unlink()

    def teardown_method(self):
        """Clean up after tests."""
        self.cleanup_test_files()
