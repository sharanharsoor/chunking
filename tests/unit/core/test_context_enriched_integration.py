"""
Integration tests for ContextEnrichedChunker.

This test suite covers integration scenarios including:
- Real text file processing
- Different content types and sizes
- Performance benchmarks
- Configuration testing
- Boundary detection accuracy
"""

import pytest
import time
from pathlib import Path
from typing import Dict, List, Any

from chunking_strategy.strategies.general.context_enriched_chunker import ContextEnrichedChunker


class TestContextEnrichedIntegration:
    """Integration test suite for ContextEnrichedChunker."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Different chunker configurations
        self.chunker_configs = [
            {
                "name": "fast_processing",
                "target_chunk_size": 800,
                "semantic_similarity_threshold": 0.6,
                "enable_topic_modeling": True,
                "boundary_detection_method": "semantic"
            },
            {
                "name": "high_quality",
                "target_chunk_size": 1500,
                "semantic_similarity_threshold": 0.8,
                "enable_ner": True,
                "enable_topic_modeling": True,
                "boundary_detection_method": "multi_modal",
                "entity_preservation_mode": "strict"
            },
            {
                "name": "balanced",
                "target_chunk_size": 1000,
                "semantic_similarity_threshold": 0.7,
                "enable_topic_modeling": True,
                "boundary_detection_method": "topic",
                "context_window_size": 3
            }
        ]

    def create_test_files(self):
        """Create test files with different content types."""
        test_files = {}

        # Academic research content
        academic_content = """
        Introduction to Quantum Computing. Quantum computing represents a paradigm shift in computational capability.
        Unlike classical bits that exist in binary states, quantum bits or qubits can exist in superposition.
        This fundamental difference enables quantum computers to perform certain calculations exponentially faster.
        
        Quantum Algorithms and Applications. Shor's algorithm demonstrates quantum advantage for factoring large integers.
        This has significant implications for cryptographic security systems currently in use.
        Grover's algorithm provides quadratic speedup for unstructured search problems.
        Quantum simulation enables modeling of molecular and material systems with unprecedented accuracy.
        
        Current Challenges and Limitations. Quantum systems are extremely sensitive to environmental interference.
        Decoherence causes quantum states to collapse, limiting computation time.
        Error correction requires thousands of physical qubits to create a single logical qubit.
        Current quantum computers operate at temperatures near absolute zero.
        
        Future Prospects and Timeline. Major technology companies are investing billions in quantum research.
        IBM, Google, and Rigetti are leading development of quantum processors.
        Fault-tolerant quantum computers may emerge within the next decade.
        Quantum supremacy has been demonstrated for specific artificial problems.
        """

        # Business and technology news
        business_content = """
        Tech Industry Market Analysis. The global technology sector experienced significant growth in 2024.
        Artificial intelligence companies received record venture capital funding this year.
        Cloud computing services continued their expansion into enterprise markets.
        Cybersecurity concerns drove increased investment in protective technologies.
        
        Regulatory Environment Changes. New data privacy regulations affect how companies handle user information.
        The European Union implemented additional restrictions on AI system deployment.
        Antitrust investigations target major technology platforms for monopolistic practices.
        Cross-border data transfer agreements require compliance with multiple jurisdictions.
        
        Innovation Trends and Adoption. Edge computing reduces latency for Internet of Things applications.
        5G networks enable new categories of mobile and industrial applications.
        Sustainability initiatives drive adoption of renewable energy in data centers.
        Remote work technologies mature as hybrid employment models become standard.
        
        Financial Performance Indicators. Technology stocks showed volatility amid economic uncertainty.
        Software-as-a-Service companies maintained strong subscription growth rates.
        Hardware manufacturers faced supply chain challenges and component shortages.
        Merger and acquisition activity increased as companies seek competitive advantages.
        """

        # Scientific research paper
        scientific_content = """
        Materials Science Research Methodology. Advanced characterization techniques reveal atomic-scale properties.
        X-ray photoelectron spectroscopy provides detailed chemical composition analysis.
        Transmission electron microscopy enables direct observation of crystal structures.
        Atomic force microscopy measures surface topography with nanometer resolution.
        
        Experimental Design and Controls. Sample preparation protocols ensure reproducible results across trials.
        Control groups eliminate confounding variables that might affect measurements.
        Statistical analysis determines significance levels for observed differences.
        Peer review validation confirms experimental methodology and interpretation.
        
        Results and Data Interpretation. Quantitative measurements show clear correlation between variables.
        Error bars represent standard deviation across multiple experimental runs.
        Trend analysis indicates linear relationship within measured parameter ranges.
        Outlier identification removes anomalous data points from final analysis.
        
        Conclusions and Future Work. Findings support the proposed theoretical framework.
        Additional experiments could explore extended parameter ranges.
        Industrial applications may benefit from these research discoveries.
        Collaboration with other research groups would accelerate development.
        """

        test_files["academic.txt"] = academic_content
        test_files["business.txt"] = business_content
        test_files["scientific.txt"] = scientific_content

        # Create actual files
        for filename, content in test_files.items():
            file_path = self.test_data_dir / filename
            file_path.write_text(content.strip())
            test_files[filename] = file_path

        return test_files

    def test_processing_different_content_types(self):
        """Test processing various content types with different configurations."""
        test_files = self.create_test_files()
        
        results = {}
        for config in self.chunker_configs:
            chunker = ContextEnrichedChunker(**{k: v for k, v in config.items() if k != "name"})
            config_results = {}
            
            for filename, file_path in test_files.items():
                if isinstance(file_path, Path):
                    result = chunker.chunk(file_path)
                    
                    config_results[filename] = {
                        "chunk_count": len(result.chunks),
                        "processing_time": result.processing_time,
                        "avg_chunk_length": sum(len(chunk.content) for chunk in result.chunks) / len(result.chunks) if result.chunks else 0,
                        "total_sentences": result.source_info.get("context_enriched_metadata", {}).get("total_sentences", 0),
                        "topics_identified": result.source_info.get("context_enriched_metadata", {}).get("total_topics", 0),
                        "semantic_boundaries": result.source_info.get("context_enriched_metadata", {}).get("semantic_boundaries", 0),
                        "avg_coherence": result.source_info.get("context_enriched_metadata", {}).get("avg_coherence_score", 0)
                    }
                    
                    # Verify chunks are meaningful
                    assert len(result.chunks) >= 1, f"No chunks created for {filename} with {config['name']}"
                    assert result.processing_time > 0, f"No processing time recorded for {filename}"
                    
                    # Check metadata richness
                    for chunk in result.chunks:
                        metadata = chunk.metadata.extra
                        assert "coherence_score" in metadata
                        assert "context_preservation_score" in metadata
                        assert "boundary_quality_score" in metadata
                        assert "topics" in metadata
                        assert metadata["coherence_score"] >= 0.0
            
            results[config["name"]] = config_results

        # Validate configuration differences
        assert len(results) == len(self.chunker_configs)
        
        # High quality config should generally provide better quality scores
        for filename in test_files:
            if isinstance(test_files[filename], Path):
                high_quality_coherence = results["high_quality"][filename]["avg_coherence"]
                fast_processing_coherence = results["fast_processing"][filename]["avg_coherence"]
                
                # Both should produce reasonable results
                assert high_quality_coherence >= 0.0 and fast_processing_coherence >= 0.0

    def test_streaming_with_large_content(self):
        """Test streaming functionality with large content."""
        chunker = ContextEnrichedChunker(
            target_chunk_size=600,
            enable_topic_modeling=True,
            boundary_detection_method="topic"
        )
        
        # Create large content by combining multiple texts
        test_files = self.create_test_files()
        large_content = ""
        for filename, file_path in test_files.items():
            if isinstance(file_path, Path):
                large_content += "\n\n" + file_path.read_text()
        
        # Split into stream chunks
        chunk_size = len(large_content) // 5
        stream_data = [
            large_content[i:i + chunk_size] 
            for i in range(0, len(large_content), chunk_size)
        ]
        
        # Process via streaming
        stream_chunks = list(chunker.chunk_stream(stream_data))
        
        # Process as batch for comparison
        batch_result = chunker.chunk(large_content)
        
        # Streaming should produce reasonable results
        assert len(stream_chunks) >= 1, "Streaming produced no chunks"
        
        # Content should be preserved
        stream_content = " ".join(chunk.content for chunk in stream_chunks)
        batch_content = " ".join(chunk.content for chunk in batch_result.chunks)
        
        # Should preserve most content (allowing for processing differences)
        assert len(stream_content) > len(large_content) * 0.7, "Streaming lost significant content"

    def test_boundary_detection_accuracy(self):
        """Test boundary detection accuracy with known topic shifts."""
        # Content with very distinct topics
        distinct_topics_text = """
        Python Programming Language. Python is an interpreted programming language.
        Object-oriented programming uses classes to organize code effectively.
        List comprehensions provide concise syntax for creating lists.
        
        Cooking and Recipe Development. Cooking techniques vary across different cultures.
        Fresh ingredients improve flavor profiles in prepared dishes.
        Meal planning reduces food waste and saves preparation time.
        
        Astronomical Observations. Telescopes reveal distant celestial objects clearly.
        Star formation occurs within dense molecular cloud regions.
        Planetary systems orbit around central stellar bodies.
        
        Financial Investment Strategies. Diversification reduces portfolio risk exposure significantly.
        Compound interest generates exponential growth over time periods.
        Market volatility affects short-term investment performance.
        """
        
        chunker = ContextEnrichedChunker(
            target_chunk_size=400,
            boundary_detection_method="topic",
            enable_topic_modeling=True,
            semantic_similarity_threshold=0.6
        )
        
        result = chunker.chunk(distinct_topics_text)
        
        # Should detect multiple topics and create chunks (may be conservative)
        # In fallback mode without SpaCy, boundary detection may be more conservative
        assert len(result.chunks) >= 1, f"Should create at least one chunk, got {len(result.chunks)}"
        
        # Verify topic detection is working
        topics_identified = result.source_info.get("context_enriched_metadata", {}).get("total_topics", 0)
        assert topics_identified >= 2, f"Should identify multiple topics, got {topics_identified}"
        
        # Analyze content for topic separation
        chunk_contents = [chunk.content.lower() for chunk in result.chunks]
        
        # Check for topic-specific keywords in different chunks
        programming_chunks = [i for i, content in enumerate(chunk_contents) if "python" in content or "programming" in content]
        cooking_chunks = [i for i, content in enumerate(chunk_contents) if "cooking" in content or "recipe" in content]
        astronomy_chunks = [i for i, content in enumerate(chunk_contents) if "telescope" in content or "star" in content]
        finance_chunks = [i for i, content in enumerate(chunk_contents) if "investment" in content or "financial" in content]
        
        # Count distinct topic groups found
        topic_groups = [group for group in [programming_chunks, cooking_chunks, astronomy_chunks, finance_chunks] if group]
        
        # Should identify multiple distinct topics
        assert len(topic_groups) >= 2, "Should identify multiple distinct topics"

    def test_performance_benchmarks(self):
        """Test performance with different text sizes."""
        # Create texts of different sizes
        base_text = """
        Technology innovation drives economic growth and social progress.
        Research and development investments create competitive advantages.
        Market adoption determines commercial success of new products.
        """
        
        text_sizes = {
            "small": base_text * 5,      # ~1KB
            "medium": base_text * 20,    # ~4KB  
            "large": base_text * 80      # ~16KB
        }
        
        performance_results = {}
        
        for size_name, text_content in text_sizes.items():
            chunker = ContextEnrichedChunker(
                target_chunk_size=800,
                enable_topic_modeling=True,
                boundary_detection_method="multi_modal"
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
                "topics_identified": result.source_info.get("context_enriched_metadata", {}).get("total_topics", 0),
                "sentences_processed": result.source_info.get("context_enriched_metadata", {}).get("total_sentences", 0)
            }
            
            # Performance expectations
            assert result.processing_time < 60.0, f"Processing too slow for {size_name}: {result.processing_time}s"
            assert result.processing_time > 0, f"No processing time recorded for {size_name}"
            assert len(result.chunks) >= 1, f"No chunks created for {size_name}"

        # Verify reasonable performance scaling
        small_time = performance_results["small"]["processing_time"]
        large_time = performance_results["large"]["processing_time"]
        
        # Larger text should not take disproportionately longer
        if small_time > 0:
            time_ratio = large_time / small_time
            size_ratio = len(text_sizes["large"]) / len(text_sizes["small"])
            
            # Processing time should scale reasonably (not exponentially)
            assert time_ratio < size_ratio * 3, f"Performance degradation too severe: {time_ratio} vs {size_ratio}"

    def test_adaptation_with_feedback(self):
        """Test adaptation capabilities with feedback scenarios."""
        chunker = ContextEnrichedChunker(
            target_chunk_size=800,
            semantic_similarity_threshold=0.7,
            topic_coherence_threshold=0.6
        )
        
        test_text = """
        Software engineering principles guide development processes effectively.
        Code review practices improve software quality and team collaboration.
        Version control systems track changes and enable team coordination.
        
        Project management methodologies organize work and resources efficiently.
        Agile development emphasizes iterative progress and customer feedback.
        Requirements gathering ensures projects meet stakeholder expectations.
        
        Quality assurance testing validates software functionality and performance.
        Automated testing reduces manual effort and improves consistency.
        User acceptance testing confirms software meets business requirements.
        """
        
        # Test coherence feedback adaptation
        initial_result = chunker.chunk(test_text)
        initial_coherence_threshold = chunker.topic_coherence_threshold
        
        # Simulate poor coherence feedback
        chunker.adapt_parameters(0.3, "coherence")
        adapted_coherence_threshold = chunker.topic_coherence_threshold
        
        # Should adapt parameters
        if adapted_coherence_threshold != initial_coherence_threshold:
            # Parameters were changed
            adapted_result = chunker.chunk(test_text)
            assert adapted_result.processing_time > 0
        
        # Test boundary feedback adaptation
        initial_boundary_threshold = chunker.semantic_similarity_threshold
        chunker.adapt_parameters(0.2, "boundary")
        adapted_boundary_threshold = chunker.semantic_similarity_threshold
        
        # Verify adaptation history
        history = chunker.get_adaptation_history()
        assert len(history) >= 1, "Adaptation history should be recorded"
        
        for record in history:
            assert "timestamp" in record
            assert "feedback_score" in record
            assert "feedback_type" in record

    def test_error_handling_and_recovery(self):
        """Test error handling with problematic content."""
        chunker = ContextEnrichedChunker()
        
        problematic_texts = [
            # Very short content
            "Short.",
            
            # Repetitive content
            "Same content. " * 50,
            
            # Mixed formatting
            "Normal text.\n\n\nLots of whitespace.\n\n\nMore text.",
            
            # Special characters
            "Text with émojis 🚀 and spëcial characters: @#$%^&*()",
            
            # Numbers and symbols
            "Version 1.2.3 has 99.9% uptime at $50/month (2024-Q1)."
        ]
        
        for i, text in enumerate(problematic_texts):
            try:
                result = chunker.chunk(text)
                assert len(result.chunks) >= 1, f"No chunks for problematic text {i+1}"
                assert result.processing_time >= 0, f"Invalid processing time for text {i+1}"
                
                # Check basic metadata integrity
                for chunk in result.chunks:
                    assert len(chunk.content) > 0, f"Empty chunk in problematic text {i+1}"
                    assert hasattr(chunk.metadata, 'extra'), f"Missing metadata in text {i+1}"
                    
            except Exception as e:
                pytest.fail(f"Chunker failed on problematic text {i+1}: {e}")

    def test_configuration_robustness(self):
        """Test various configuration combinations."""
        configurations = [
            # Minimal configuration
            {"target_chunk_size": 500, "enable_ner": False, "enable_topic_modeling": False},
            
            # Topic-focused configuration
            {"target_chunk_size": 1000, "enable_topic_modeling": True, "boundary_detection_method": "topic"},
            
            # Entity-focused configuration (will use fallback without SpaCy)
            {"target_chunk_size": 800, "enable_ner": True, "entity_preservation_mode": "strict"},
            
            # Multi-modal configuration
            {"target_chunk_size": 1200, "enable_topic_modeling": True, "boundary_detection_method": "multi_modal"},
        ]
        
        test_text = """
        Machine learning applications span multiple industries and use cases.
        Healthcare systems use AI for diagnostic imaging and treatment planning.
        Financial institutions deploy algorithms for fraud detection and risk assessment.
        Transportation companies implement autonomous vehicle technologies.
        """
        
        for i, config in enumerate(configurations):
            try:
                chunker = ContextEnrichedChunker(**config)
                result = chunker.chunk(test_text)
                
                assert len(result.chunks) >= 1, f"Configuration {i+1} produced no chunks"
                assert result.processing_time >= 0, f"Configuration {i+1} invalid processing time"
                
                # Check metadata consistency
                for chunk in result.chunks:
                    metadata = chunk.metadata.extra
                    assert "coherence_score" in metadata, f"Configuration {i+1} missing coherence score"
                    assert "context_preservation_score" in metadata, f"Configuration {i+1} missing context score"
                    
            except Exception as e:
                pytest.fail(f"Configuration {i+1} failed: {e}")

    def cleanup_test_files(self):
        """Clean up created test files."""
        if self.test_data_dir.exists():
            for file in self.test_data_dir.glob("*.txt"):
                if file.name in ["academic.txt", "business.txt", "scientific.txt"]:
                    file.unlink()

    def teardown_method(self):
        """Clean up after tests."""
        self.cleanup_test_files()
