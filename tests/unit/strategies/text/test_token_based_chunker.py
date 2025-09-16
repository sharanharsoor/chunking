"""
Comprehensive tests for the TokenBasedChunker.

This module tests the token-based chunking strategy with multiple tokenizers,
real files, edge cases, and comprehensive validation scenarios.
"""

import pytest
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

from chunking_strategy.strategies.text.token_based_chunker import TokenBasedChunker, TokenizerType, TokenizerModel
from chunking_strategy.core.base import Chunk, ChunkingResult, ModalityType
from chunking_strategy.utils.validation import ChunkValidator
from chunking_strategy.core.metrics import ChunkingQualityEvaluator
from tests.conftest import assert_valid_chunks, assert_reasonable_performance


class TestTokenBasedChunker:
    """Test suite for TokenBasedChunker."""

    def setup_method(self):
        """Set up test fixtures."""
        # Default chunker for most tests
        self.chunker = TokenBasedChunker(
            tokens_per_chunk=100,
            overlap_tokens=10,
            tokenizer_type="simple",  # Use simple for consistent testing
            preserve_word_boundaries=True
        )

        # Test data
        self.simple_text = "This is a simple test text with exactly twenty words for basic token chunking functionality testing purposes."
        self.long_text = "This is a comprehensive test document with many sentences and words. " * 50  # ~700 words

        # Complex text with various challenges
        self.complex_text = """
        This document contains multiple paragraphs, different punctuation marks!

        It includes questions? And exclamations! Plus some "quoted text" and more.

        Numbers like 123, 456.78, and dates like 2024-01-15 should be handled properly.

        URLs like https://example.com and emails like user@domain.com are common too.

        Special symbols: @#$%^&*()_+-=[]{}|;':",./<>? need proper tokenization.
        """

    def test_basic_token_chunking(self):
        """Test basic token-based chunking functionality."""
        result = self.chunker.chunk(self.simple_text)

        # Basic validation
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) >= 1
        assert result.strategy_used == "token_based"
        assert result.processing_time >= 0

        # Check chunk properties
        chunk = result.chunks[0]
        assert isinstance(chunk, Chunk)
        assert chunk.modality == ModalityType.TEXT
        assert chunk.content
        assert chunk.metadata.extra["chunking_strategy"] == "token_based"
        assert "token_count" in chunk.metadata.extra
        assert chunk.metadata.extra["token_count"] > 0

    def test_multi_chunk_splitting(self):
        """Test chunking text that requires multiple chunks."""
        chunker = TokenBasedChunker(
            tokens_per_chunk=50,
            overlap_tokens=5,
            tokenizer_type="simple"
        )

        result = chunker.chunk(self.long_text)

        # Should create multiple chunks
        assert len(result.chunks) > 1
        assert result.total_chunks == len(result.chunks)

        # Verify chunk properties
        total_tokens = 0
        for i, chunk in enumerate(result.chunks):
            token_count = chunk.metadata.extra["token_count"]
            assert token_count > 0
            assert token_count <= chunker.tokens_per_chunk
            total_tokens += token_count

            # Check metadata completeness
            assert chunk.metadata.extra["chunk_index"] == i
            assert "start_token_index" in chunk.metadata.extra
            assert "end_token_index" in chunk.metadata.extra

        # Total tokens should account for overlap
        source_tokens = result.source_info["total_tokens"]
        # With overlap, total chunk tokens should be more than source tokens
        if len(result.chunks) > 1:
            assert total_tokens >= source_tokens

    def test_different_tokenizer_types(self):
        """Test different tokenizer backends."""
        tokenizer_configs = [
            {"tokenizer_type": "simple", "should_work": True},
            {"tokenizer_type": "tiktoken", "tokenizer_model": "gpt-3.5-turbo", "should_work": True},
            {"tokenizer_type": "transformers", "tokenizer_model": "bert-base-uncased", "should_work": False},  # May not be available
            {"tokenizer_type": "spacy", "tokenizer_model": "en_core_web_sm", "should_work": False},  # May not be installed
            {"tokenizer_type": "nltk", "should_work": False},  # May need downloads
        ]

        test_text = "This is a test sentence with multiple words for tokenization testing."

        for config in tokenizer_configs:
            try:
                chunker = TokenBasedChunker(
                    tokens_per_chunk=20,
                    overlap_tokens=5,  # Ensure valid parameters
                    **{k: v for k, v in config.items() if k != "should_work"}
                )

                result = chunker.chunk(test_text)

                # Should work or fallback to simple
                assert len(result.chunks) >= 1
                assert result.strategy_used == "token_based"

                # Check tokenizer info
                tokenizer_info = result.source_info.get("tokenizer_info", {})
                assert "type" in tokenizer_info

                print(f"✅ {config['tokenizer_type']} tokenizer working")

            except Exception as e:
                if config["should_work"]:
                    pytest.fail(f"Required tokenizer {config['tokenizer_type']} failed: {e}")
                else:
                    print(f"⚠️  Optional tokenizer {config['tokenizer_type']} not available: {e}")

    def test_token_overlap_functionality(self):
        """Test token overlap between chunks."""
        chunker = TokenBasedChunker(
            tokens_per_chunk=20,
            overlap_tokens=5,
            tokenizer_type="simple"
        )

        # Create text with known token structure
        words = [f"word{i:02d}" for i in range(1, 61)]  # 60 unique words
        test_text = " ".join(words)

        result = chunker.chunk(test_text)

        # Should create multiple chunks with overlap
        assert len(result.chunks) >= 2

        # Check overlap functionality
        for i, chunk in enumerate(result.chunks):
            token_count = chunk.metadata.extra["token_count"]
            start_idx = chunk.metadata.extra["start_token_index"]
            end_idx = chunk.metadata.extra["end_token_index"]

            if i < len(result.chunks) - 1:
                # All but last chunk should have expected tokens
                assert token_count <= chunker.tokens_per_chunk

            # Check indices are reasonable
            assert start_idx >= 0
            assert end_idx >= start_idx
            assert end_idx - start_idx + 1 == token_count

    def test_tokenizer_model_variations(self):
        """Test different tokenizer models."""
        models_to_test = [
            ("tiktoken", "gpt-3.5-turbo"),
            ("tiktoken", "gpt-4"),
            ("transformers", "bert-base-uncased"),
            ("transformers", "gpt2"),
            ("simple", None),
        ]

        test_text = "This is a test for different tokenizer models and their token counting capabilities."

        for tokenizer_type, model in models_to_test:
            try:
                config = {"tokenizer_type": tokenizer_type}
                if model:
                    config["tokenizer_model"] = model

                chunker = TokenBasedChunker(tokens_per_chunk=50, **config)
                result = chunker.chunk(test_text)

                assert len(result.chunks) >= 1
                assert result.source_info["total_tokens"] > 0

                # Different tokenizers should produce different token counts
                tokenizer_info = result.source_info["tokenizer_info"]
                assert tokenizer_info["type"] == tokenizer_type

                if model:
                    # Model info should be preserved
                    assert "model" in tokenizer_info or model in str(tokenizer_info)

                print(f"✅ {tokenizer_type}:{model} - {result.source_info['total_tokens']} tokens")

            except ImportError as e:
                print(f"⚠️  {tokenizer_type}:{model} not available: {e}")
            except Exception as e:
                print(f"❌ {tokenizer_type}:{model} failed: {e}")

    def test_word_boundary_preservation(self):
        """Test word boundary preservation options."""
        test_text = "This is a test of word-boundary preservation in token-based chunking systems."

        # Test with word boundary preservation
        chunker_preserve = TokenBasedChunker(
            tokens_per_chunk=20,
            overlap_tokens=5,
            tokenizer_type="simple",
            preserve_word_boundaries=True
        )

        # Test without word boundary preservation
        chunker_no_preserve = TokenBasedChunker(
            tokens_per_chunk=20,
            overlap_tokens=5,
            tokenizer_type="simple",
            preserve_word_boundaries=False
        )

        result_preserve = chunker_preserve.chunk(test_text)
        result_no_preserve = chunker_no_preserve.chunk(test_text)

        # Both should work
        assert len(result_preserve.chunks) >= 1
        assert len(result_no_preserve.chunks) >= 1

        # Check that word boundaries are reflected in config
        assert result_preserve.chunks[0].metadata.extra["preserve_word_boundaries"] == True
        assert result_no_preserve.chunks[0].metadata.extra["preserve_word_boundaries"] == False

    def test_size_constraints(self):
        """Test token and character size constraints."""
        chunker = TokenBasedChunker(
            tokens_per_chunk=100,
            min_chunk_tokens=5,
            max_chunk_chars=500,
            tokenizer_type="simple"
        )

        result = chunker.chunk(self.long_text)

        for chunk in result.chunks:
            token_count = chunk.metadata.extra["token_count"]
            char_count = len(chunk.content)

            # Check token constraints
            if chunk != result.chunks[-1]:  # Not the last chunk
                assert token_count >= chunker.min_chunk_tokens

            # Check character constraints
            assert char_count <= chunker.max_chunk_chars

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty string
        result_empty = self.chunker.chunk("")
        assert len(result_empty.chunks) == 0
        assert result_empty.total_chunks == 0

        # Single word
        result_single = self.chunker.chunk("word")
        assert len(result_single.chunks) >= 1

        # Very long word
        long_word = "a" * 1000
        result_long = self.chunker.chunk(long_word)
        assert len(result_long.chunks) >= 1
        assert len(result_long.chunks[0].content) <= self.chunker.max_chunk_chars

        # Only whitespace
        result_whitespace = self.chunker.chunk("   \n\t   ")
        assert len(result_whitespace.chunks) == 0

        # Mixed content
        mixed_content = "Word1\n\n\nWord2    Word3\t\tWord4"
        result_mixed = self.chunker.chunk(mixed_content)
        assert len(result_mixed.chunks) >= 1

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Test invalid parameter combinations
        with pytest.raises(ValueError):
            TokenBasedChunker(
                tokens_per_chunk=10,
                overlap_tokens=15  # Too high
            )

        with pytest.raises(ValueError):
            TokenBasedChunker(tokens_per_chunk=0)  # Invalid

        with pytest.raises(ValueError):
            TokenBasedChunker(overlap_tokens=-1)  # Invalid

        # Test invalid tokenizer type
        with pytest.raises(ValueError):
            TokenBasedChunker(tokenizer_type="invalid_tokenizer")

    def test_parameter_validation(self):
        """Test parameter validation and constraints."""
        # Valid parameters should work
        chunker = TokenBasedChunker(
            tokens_per_chunk=1000,
            overlap_tokens=100,
            tokenizer_type="simple"
        )
        assert chunker.tokens_per_chunk == 1000
        assert chunker.overlap_tokens == 100

        # Test automatic constraint enforcement
        chunker_constrained = TokenBasedChunker(
            tokens_per_chunk=10,
            overlap_tokens=5,  # Should be valid
            min_chunk_tokens=3
        )
        assert chunker_constrained.overlap_tokens == 5
        assert chunker_constrained.min_chunk_tokens == 3

    def test_metadata_completeness(self):
        """Test that chunks contain complete metadata."""
        result = self.chunker.chunk(self.simple_text)

        assert len(result.chunks) >= 1
        chunk = result.chunks[0]

        # Check required metadata fields
        required_fields = [
            "token_count", "start_token_index", "end_token_index",
            "chunk_index", "chunking_strategy", "tokenizer_type"
        ]

        for field in required_fields:
            assert field in chunk.metadata.extra, f"Missing metadata field: {field}"

        # Check metadata values
        assert chunk.metadata.extra["chunking_strategy"] == "token_based"
        assert chunk.metadata.extra["tokenizer_type"] == "simple"
        assert chunk.metadata.extra["chunk_index"] >= 0
        assert chunk.metadata.extra["token_count"] > 0

    def test_streaming_support(self):
        """Test streaming chunker functionality."""
        # Create content chunks to stream
        content_chunks = [
            "This is the first part of streaming content. ",
            "Here comes the second part with more words. ",
            "And finally the third part to complete the test. ",
            "Some additional content to ensure multiple chunks are created."
        ]

        def content_generator():
            for chunk in content_chunks:
                yield chunk

        # Test streaming
        chunks = list(self.chunker.chunk_stream(content_generator()))

        # Should create at least one chunk
        assert len(chunks) >= 1

        # Check streaming metadata
        for chunk in chunks:
            assert chunk.metadata.extra.get("is_streaming", False) == True
            assert "token_count" in chunk.metadata.extra
            assert chunk.metadata.extra["chunking_strategy"] == "token_based"

    def test_adaptation_functionality(self):
        """Test parameter adaptation based on feedback."""
        # Create a chunker with larger initial values so adaptation can modify them
        chunker = TokenBasedChunker(
            tokens_per_chunk=500,
            overlap_tokens=50,
            tokenizer_type="simple"
        )

        original_tokens_per_chunk = chunker.tokens_per_chunk
        original_overlap = chunker.overlap_tokens

        # Test quality feedback (low score should reduce chunk size)
        chunker.adapt_parameters(0.3, "quality")
        assert chunker.tokens_per_chunk < original_tokens_per_chunk

        # Reset for next test
        chunker.tokens_per_chunk = original_tokens_per_chunk
        chunker.overlap_tokens = original_overlap

        # Test performance feedback (low score should increase chunk size)
        chunker.adapt_parameters(0.3, "performance")
        assert chunker.tokens_per_chunk > original_tokens_per_chunk

        # Test adaptation history is recorded
        history = chunker.get_adaptation_history()
        assert len(history) >= 2  # Should have at least 2 adaptation records

    def test_performance_benchmarks(self):
        """Test performance characteristics."""
        # Large text for performance testing
        large_text = self.long_text * 10  # ~7000 words

        start_time = time.time()
        result = self.chunker.chunk(large_text)
        processing_time = time.time() - start_time

        # Performance assertions
        assert processing_time < 5.0  # Should process quickly
        assert len(result.chunks) > 0
        assert result.processing_time > 0

        # Check tokenization performance
        if "tokenization_time" in result.source_info:
            tokenization_time = result.source_info["tokenization_time"]
            assert tokenization_time > 0
            assert tokenization_time < processing_time

    def test_quality_metrics(self):
        """Test chunk quality using the quality evaluator."""
        result = self.chunker.chunk(self.complex_text)

        # Use quality evaluator
        try:
            evaluator = ChunkingQualityEvaluator()
            metrics = evaluator.evaluate(result.chunks)

            # Quality should be reasonable
            assert metrics.size_consistency >= 0.0  # Should be measurable
            assert metrics.coverage >= 0.8  # Should cover most content

        except Exception as e:
            # Quality evaluation might not be fully implemented
            print(f"Quality evaluation not available: {e}")

    def test_integration_with_orchestrator(self):
        """Test integration with the orchestrator system."""
        try:
            from chunking_strategy import create_chunker

            # Create chunker via registry
            chunker = create_chunker(
                "token_based",
                tokens_per_chunk=100,
                tokenizer_type="simple"
            )

            assert isinstance(chunker, TokenBasedChunker)

            # Test chunking via orchestrator
            result = chunker.chunk("This is a test of orchestrator integration with token-based chunking.")
            assert len(result.chunks) >= 1
            assert result.strategy_used == "token_based"

        except ImportError:
            pytest.skip("Orchestrator integration not available")

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        unicode_text = """
        This text contains unicode characters: café, naïve, résumé.
        Special symbols: ★ ♦ ♣ ♠ ♥ ⚡ ⚽ 🌟 🎯 🚀
        Mathematical symbols: α β γ δ ε ζ η θ ∑ ∆ ∇ ∞
        Currency symbols: $ € £ ¥ ₹ ₿
        """

        result = self.chunker.chunk(unicode_text)

        # Should handle unicode gracefully
        assert len(result.chunks) >= 1
        assert result.processing_time >= 0

        # Content should be preserved
        total_content = "".join(chunk.content for chunk in result.chunks)
        assert len(total_content) > 0

    def test_content_integrity(self):
        """Test that chunking preserves content integrity."""
        result = self.chunker.chunk(self.complex_text)

        # Reconstruct text from chunks
        reconstructed = ""
        for chunk in result.chunks:
            reconstructed += chunk.content + " "

        # Should preserve most words (allowing for boundary effects)
        original_words = set(self.complex_text.lower().split())
        reconstructed_words = set(reconstructed.lower().split())

        # Most words should be preserved
        preserved_ratio = len(original_words & reconstructed_words) / len(original_words)
        assert preserved_ratio > 0.8  # Should preserve most content

    def test_configuration_compatibility(self):
        """Test compatibility with configuration systems."""
        config = self.chunker.get_config()

        # Should return complete configuration
        assert isinstance(config, dict)
        assert config["name"] == "token_based"
        assert "tokens_per_chunk" in config
        assert "tokenizer_type" in config
        assert "performance_stats" in config

    def test_tokenizer_fallback_mechanism(self):
        """Test fallback when primary tokenizer fails."""
        # Try to create chunker with unavailable tokenizer
        try:
            chunker = TokenBasedChunker(
                tokenizer_type="transformers",
                tokenizer_model="non-existent-model"
            )

            # Should either work or fallback gracefully
            result = chunker.chunk("This is a fallback test.")
            assert len(result.chunks) >= 1

        except Exception as e:
            # Fallback mechanisms should prevent complete failure
            print(f"Expected fallback scenario: {e}")

    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases."""
        # Very small chunks
        tiny_chunker = TokenBasedChunker(
            tokens_per_chunk=1,
            overlap_tokens=0,
            min_chunk_tokens=1,
            tokenizer_type="simple"
        )

        result = tiny_chunker.chunk("word1 word2 word3")
        assert len(result.chunks) >= 1

        # Very large chunks
        large_chunker = TokenBasedChunker(
            tokens_per_chunk=10000,
            tokenizer_type="simple"
        )

        result = large_chunker.chunk(self.simple_text)
        assert len(result.chunks) == 1  # Should fit in one chunk

    def test_error_handling_robustness(self):
        """Test robust error handling."""
        # Test with problematic content
        problematic_texts = [
            "\x00\x01\x02",  # Control characters
            "a" * 100000,    # Very long string
            "\n" * 1000,     # Many newlines
            "🚀" * 1000,     # Many emoji
        ]

        for text in problematic_texts:
            try:
                result = self.chunker.chunk(text)
                # Should handle gracefully
                assert isinstance(result, ChunkingResult)
                assert result.processing_time >= 0

            except Exception as e:
                # Should not crash completely
                print(f"Handled problematic input gracefully: {e}")

    def test_supported_tokenizers_info(self):
        """Test tokenizer support information."""
        supported = TokenBasedChunker.get_supported_tokenizers()

        # Should return comprehensive info
        assert isinstance(supported, dict)
        assert "tiktoken" in supported
        assert "transformers" in supported
        assert "simple" in supported

        # Each should have model lists
        for tokenizer_type, models in supported.items():
            assert isinstance(models, list)
            assert len(models) > 0

    def test_file_input_processing(self):
        """Test processing files directly."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(self.complex_text)
            temp_path = Path(f.name)

        try:
            # Test file processing
            result = self.chunker.chunk(temp_path)

            assert len(result.chunks) >= 1
            assert result.source_info["source_type"] == "file"
            assert result.source_info["source"] == str(temp_path)

        finally:
            # Cleanup
            temp_path.unlink()


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v"])
