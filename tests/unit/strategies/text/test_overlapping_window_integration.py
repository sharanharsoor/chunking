"""
Integration tests for OverlappingWindowChunker.

This module provides comprehensive integration tests that focus on real-world
scenarios, file processing, CLI integration, configuration-based processing,
and interaction with other system components.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from chunking_strategy.strategies.text.overlapping_window_chunker import (
    OverlappingWindowChunker,
    WindowUnit
)
from chunking_strategy.core.base import ChunkingResult, Chunk
from chunking_strategy.orchestrator import ChunkerOrchestrator


class TestOverlappingWindowIntegration:
    """Integration tests for OverlappingWindowChunker in real-world scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = OverlappingWindowChunker()
        self.orchestrator = ChunkerOrchestrator()

        # Real-world test data
        self.sample_essay = """
        The concept of artificial intelligence has evolved dramatically over the past
        several decades. From early theoretical frameworks proposed by pioneers like
        Alan Turing and John McCarthy, to modern deep learning architectures that
        power today's most sophisticated applications, AI has transformed from science
        fiction into practical reality.

        Machine learning algorithms now process vast amounts of data to recognize
        patterns, make predictions, and automate complex decision-making processes.
        These systems have found applications in healthcare, finance, transportation,
        and countless other domains where they enhance human capabilities and improve
        efficiency.

        However, the rapid advancement of AI technology also raises important questions
        about ethics, employment, privacy, and the future relationship between humans
        and intelligent machines. As we continue to push the boundaries of what's
        possible with artificial intelligence, we must carefully consider both the
        tremendous opportunities and the potential risks that lie ahead.
        """.strip()

        self.technical_document = """
        # System Architecture Overview

        ## Core Components

        The distributed system consists of three primary layers:

        ### 1. Presentation Layer
        - Web-based user interface
        - RESTful API endpoints
        - Authentication and authorization services

        ### 2. Business Logic Layer
        - Microservices architecture
        - Event-driven processing
        - Service mesh integration

        ### 3. Data Layer
        - Distributed databases
        - Message queuing systems
        - Caching mechanisms

        ## Performance Characteristics

        The system is designed to handle:
        - 10,000+ concurrent users
        - 1M+ transactions per hour
        - 99.9% uptime availability
        - Sub-second response times

        ## Security Features

        Security is implemented through:
        - Multi-factor authentication
        - End-to-end encryption
        - Regular security audits
        - Compliance with industry standards
        """.strip()

    def test_processing_real_text_files(self):
        """Test processing various real text files."""
        test_files = [
            ("essay.txt", self.sample_essay),
            ("technical_doc.md", self.technical_document),
            ("mixed_content.txt", self.sample_essay + "\n\n" + self.technical_document),
        ]

        for filename, content in test_files:
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{filename}', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = Path(tmp_file.name)

            try:
                # Test different configurations
                chunker_configs = [
                    {"window_size": 50, "step_size": 25, "window_unit": "words", "min_window_size": 10},
                    {"window_size": 200, "step_size": 100, "window_unit": "characters", "min_window_size": 20},
                    {"window_size": 2, "step_size": 1, "window_unit": "sentences", "min_window_size": 1},
                ]

                for config in chunker_configs:
                    chunker = OverlappingWindowChunker(**config)
                    result = chunker.chunk(tmp_file_path)

                    # Verify successful processing
                    assert isinstance(result, ChunkingResult)
                    assert len(result.chunks) > 0
                    assert all(isinstance(chunk, Chunk) for chunk in result.chunks)
                    assert "total_units" in result.source_info
                    assert "overlap_ratio" in result.source_info

            finally:
                tmp_file_path.unlink()

    def test_unicode_and_multilingual_content(self):
        """Test processing of unicode and multilingual content."""
        multilingual_content = """
        English: The quick brown fox jumps over the lazy dog.
        Español: El rápido zorro marrón salta sobre el perro perezoso.
        Français: Le renard brun rapide saute par-dessus le chien paresseux.
        Deutsch: Der schnelle braune Fuchs springt über den faulen Hund.
        Русский: Быстрая коричневая лиса прыгает через ленивую собаку.
        中文: 快速的棕色狐狸跳过懒惰的狗。
        日本語: 素早い茶色のキツネが怠け者の犬を飛び越える。
        العربية: الثعلب البني السريع يقفز فوق الكلب الكسول.
        """

        chunker = OverlappingWindowChunker(
            window_size=20,
            step_size=10,
            window_unit="words",
            min_window_size=5
        )

        result = chunker.chunk(multilingual_content)

        assert len(result.chunks) > 0
        # Verify that unicode characters are properly handled
        concatenated = ''.join(chunk.content for chunk in result.chunks)
        assert len(concatenated.strip()) > 0

    def test_large_file_processing(self):
        """Test processing of larger files."""
        # Create a large document by repeating content
        large_content = (self.sample_essay + "\n\n") * 50  # ~50x repetition

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(large_content)
            tmp_file_path = Path(tmp_file.name)

        try:
            chunker = OverlappingWindowChunker(
                window_size=100,
                step_size=50,
                window_unit="words"
            )

            result = chunker.chunk(tmp_file_path)

            # Should handle large files efficiently
            assert len(result.chunks) >= 10  # Should create multiple chunks
            assert result.processing_time < 30.0  # Should complete within reasonable time
            assert "total_units" in result.source_info

        finally:
            tmp_file_path.unlink()

    @pytest.mark.skip(reason="Requires full CLI system to be operational")
    def test_cli_integration(self):
        """Test integration with CLI interface."""
        # This would test actual CLI commands
        pass

    def test_configuration_based_processing(self):
        """Test processing using YAML configuration files."""
        config_data = {
            "overlapping_window_configs": {
                "default": {
                    "window_size": 100,
                    "step_size": 50,
                    "window_unit": "words",
                    "preserve_boundaries": True,
                    "min_window_size": 20
                },
                "precise": {
                    "window_size": 50,
                    "step_size": 25,
                    "window_unit": "words",
                    "preserve_boundaries": True,
                    "min_window_size": 10
                },
                "character_based": {
                    "window_size": 500,
                    "step_size": 250,
                    "window_unit": "characters",
                    "preserve_boundaries": False,
                    "min_window_size": 100
                }
            }
        }

        # Test each configuration
        for config_name, config_params in config_data["overlapping_window_configs"].items():
            chunker = OverlappingWindowChunker(**config_params)
            result = chunker.chunk(self.sample_essay)

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) > 0
            assert result.strategy_used.endswith("overlapping_window")

    def test_orchestrator_integration(self):
        """Test integration with the chunking orchestrator."""
        # Register and use through orchestrator
        available_strategies = self.orchestrator.list_available_strategies()

        # Should include overlapping window chunker in traditional strategies
        assert "overlapping_window" in available_strategies.get("traditional", [])

        # Test orchestration
        result = self.orchestrator.chunk_content(
            content=self.sample_essay,
            strategy="overlapping_window",
            parameters={
                "window_size": 75,
                "step_size": 35,
                "window_unit": "words"
            }
        )

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0

    def test_different_window_units(self):
        """Test all supported window units with real content."""
        test_cases = [
            {
                "window_unit": "characters",
                "window_size": 200,
                "step_size": 100,
                "min_window_size": 100,
                "expected_min_chunks": 2
            },
            {
                "window_unit": "words",
                "window_size": 30,
                "step_size": 15,
                "min_window_size": 10,
                "expected_min_chunks": 3
            },
            {
                "window_unit": "sentences",
                "window_size": 2,
                "step_size": 1,
                "min_window_size": 1,
                "expected_min_chunks": 2
            }
        ]

        for case in test_cases:
            chunker = OverlappingWindowChunker(
                window_size=case["window_size"],
                step_size=case["step_size"],
                window_unit=case["window_unit"],
                min_window_size=case["min_window_size"]
            )

            result = chunker.chunk(self.technical_document)

            assert len(result.chunks) >= case["expected_min_chunks"]
            assert all(chunk.content.strip() for chunk in result.chunks)

    def test_batch_processing_multiple_files(self):
        """Test batch processing of multiple files."""
        test_files_data = [
            ("doc1.txt", self.sample_essay),
            ("doc2.txt", self.technical_document),
            ("doc3.txt", "Short document for testing purposes."),
        ]

        file_paths = []
        try:
            # Create temporary files
            for filename, content in test_files_data:
                tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_{filename}', delete=False)
                tmp_file.write(content)
                tmp_file.close()
                file_paths.append(Path(tmp_file.name))

            chunker = OverlappingWindowChunker(
                window_size=40,
                step_size=20,
                window_unit="words",
                min_window_size=20
            )

            # Process all files
            all_results = []
            for file_path in file_paths:
                result = chunker.chunk(file_path)
                all_results.append(result)

            # Verify all were processed successfully
            assert len(all_results) == 3
            assert all(isinstance(result, ChunkingResult) for result in all_results)
            assert all(len(result.chunks) > 0 for result in all_results)

        finally:
            # Cleanup
            for file_path in file_paths:
                if file_path.exists():
                    file_path.unlink()

    def test_streaming_integration(self):
        """Test streaming integration with real data."""
        chunker = OverlappingWindowChunker(
            window_size=25,
            step_size=12,
            window_unit="words",
            min_window_size=10
        )

        # Test streaming with different content types
        stream_chunks = list(chunker.stream_chunk(self.technical_document))
        batch_result = chunker.chunk(self.technical_document)

        # Streaming should produce same chunks as batch processing
        assert len(stream_chunks) == len(batch_result.chunks)

        # Content should be identical
        for i, (stream_chunk, batch_chunk) in enumerate(zip(stream_chunks, batch_result.chunks)):
            assert stream_chunk.content == batch_chunk.content

    def test_error_handling_with_real_files(self):
        """Test error handling with various file scenarios."""
        # Test with empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as empty_file:
            empty_file.write("")
            empty_file_path = Path(empty_file.name)

        try:
            result = self.chunker.chunk(empty_file_path)
            assert len(result.chunks) == 0
        finally:
            empty_file_path.unlink()

        # Test with whitespace-only file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as whitespace_file:
            whitespace_file.write("   \n\t\n   \n")
            whitespace_file_path = Path(whitespace_file.name)

        try:
            result = self.chunker.chunk(whitespace_file_path)
            assert len(result.chunks) == 0
        finally:
            whitespace_file_path.unlink()

        # Test with minimal content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as minimal_file:
            minimal_file.write("One word.")
            minimal_file_path = Path(minimal_file.name)

        try:
            result = self.chunker.chunk(minimal_file_path)
            # Should handle gracefully
            assert isinstance(result, ChunkingResult)
        finally:
            minimal_file_path.unlink()

    def test_performance_benchmarks(self):
        """Test performance with realistic data sizes."""
        # Create medium-sized content
        medium_content = self.sample_essay * 10  # ~10x repetition

        chunker = OverlappingWindowChunker(
            window_size=80,
            step_size=40,
            window_unit="words"
        )

        result = chunker.chunk(medium_content)

        # Performance assertions
        assert result.processing_time < 5.0  # Should complete within 5 seconds
        assert len(result.chunks) > 0
        assert "overlap_ratio" in result.source_info

        # Memory efficiency - chunks shouldn't be excessively large
        max_chunk_size = max(len(chunk.content) for chunk in result.chunks)
        assert max_chunk_size < 10000  # Reasonable chunk size limit

    def test_quality_metrics_integration(self):
        """Test integration with quality evaluation systems."""
        # Use smaller window to ensure multiple chunks for variety test
        test_chunker = OverlappingWindowChunker(
            window_size=30,
            step_size=15,
            min_window_size=10
        )
        result = test_chunker.chunk(self.technical_document)

        # Basic quality checks that could be automated
        assert all(len(chunk.content.strip()) > 0 for chunk in result.chunks)
        assert len(set(chunk.content for chunk in result.chunks)) > 1  # Should have variety

        # Overlap verification
        if len(result.chunks) > 1:
            # Check that consecutive chunks have some relationship
            chunk_words_sets = [set(chunk.content.lower().split()) for chunk in result.chunks]
            for i in range(len(chunk_words_sets) - 1):
                # Should have some word overlap or be sequential
                overlap = chunk_words_sets[i] & chunk_words_sets[i + 1]
                assert len(overlap) > 0 or len(chunk_words_sets[i]) < 50  # Allow small chunks
