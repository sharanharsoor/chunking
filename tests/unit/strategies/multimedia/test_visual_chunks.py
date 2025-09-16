#!/usr/bin/env python3
"""
Visual test for chunk output - prints actual chunks during testing.
Run with: pytest tests/test_visual_chunks.py -s -v
"""

import pytest
from chunking_strategy.strategies.text.sentence_based import SentenceBasedChunker
from chunking_strategy.strategies.text.paragraph_based import ParagraphBasedChunker
from chunking_strategy.strategies.general.fixed_size import FixedSizeChunker


def print_test_header(test_name: str):
    """Print a nice header for test sections."""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")


def print_chunks_detailed(chunks, chunker_name: str):
    """Print chunks with detailed information."""
    print(f"\n🔧 {chunker_name} produced {len(chunks)} chunks:")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n   📝 Chunk {i}:")
        print(f"      Content: \"{chunk.content}\"")
        print(f"      Length: {len(chunk.content)} chars")
        print(f"      ID: {chunk.id}")


class TestVisualChunks:
    """Test class that shows actual chunk output."""

    def test_sentence_chunker_visual(self):
        """Test sentence chunker and print the actual chunks."""
        print_test_header("SENTENCE CHUNKER VISUAL TEST")

        text = "First sentence here. Second sentence follows. Third sentence too. Fourth and final sentence."
        print(f"\n📄 Input text: \"{text}\"")
        print(f"   Length: {len(text)} characters")

        chunker = SentenceBasedChunker(max_sentences=2)
        result = chunker.chunk(text)

        print_chunks_detailed(result.chunks, "SentenceBasedChunker (max 2 sentences)")

        # Assertions to make it a proper test
        assert len(result.chunks) == 2
        assert result.chunks[0].content == "First sentence here. Second sentence follows."
        assert result.chunks[1].content == "Third sentence too. Fourth and final sentence."

        print(f"\n✅ Test passed! Chunks look correct.")


    def test_paragraph_chunker_visual(self):
        """Test paragraph chunker and print the actual chunks."""
        print_test_header("PARAGRAPH CHUNKER VISUAL TEST")

        text = """First paragraph with some content. It has multiple sentences.

Second paragraph here. Also with sentences.

Third and final paragraph."""

        print(f"\n📄 Input text: \"{text}\"")
        print(f"   Length: {len(text)} characters")

        chunker = ParagraphBasedChunker(max_paragraphs=1)
        result = chunker.chunk(text)

        print_chunks_detailed(result.chunks, "ParagraphBasedChunker (1 paragraph each)")

        # Assertions - paragraph chunker may merge short paragraphs
        assert len(result.chunks) >= 1
        assert "First paragraph" in result.chunks[0].content

        print(f"\n✅ Test passed! Paragraph chunking working correctly.")


    def test_fixed_size_chunker_visual(self):
        """Test fixed size chunker and print the actual chunks."""
        print_test_header("FIXED SIZE CHUNKER VISUAL TEST")

        text = "This is a longer text that will be split into fixed-size chunks of exactly 25 characters each to demonstrate the chunking behavior."
        print(f"\n📄 Input text: \"{text}\"")
        print(f"   Length: {len(text)} characters")

        chunker = FixedSizeChunker(chunk_size=25)
        result = chunker.chunk(text)

        print_chunks_detailed(result.chunks, "FixedSizeChunker (25 chars each)")

        # Assertions
        assert len(result.chunks) >= 5  # Should split into multiple chunks

        # Check that all chunks except the last are exactly 25 characters
        for chunk in result.chunks[:-1]:
            assert len(chunk.content) == 25

        # Last chunk can be shorter
        assert len(result.chunks[-1].content) <= 25

        print(f"\n✅ Test passed! Fixed-size chunking working correctly.")


    def test_chunker_comparison(self):
        """Compare all three chunkers on the same text."""
        print_test_header("CHUNKER COMPARISON TEST")

        text = "Hello world! This is a test sentence. Here's another sentence with more content. And a final sentence to wrap up."
        print(f"\n📄 Comparing chunkers on text: \"{text}\"")
        print(f"   Length: {len(text)} characters")

        # Test all three chunkers
        sentence_chunker = SentenceBasedChunker(max_sentences=2)
        paragraph_chunker = ParagraphBasedChunker(max_paragraphs=1)
        fixed_chunker = FixedSizeChunker(chunk_size=40)

        sentence_result = sentence_chunker.chunk(text)
        paragraph_result = paragraph_chunker.chunk(text)
        fixed_result = fixed_chunker.chunk(text)

        print(f"\n📊 Comparison Results:")
        print(f"   Sentence chunker (2 sent/chunk): {len(sentence_result.chunks)} chunks")
        print(f"   Paragraph chunker (1 para/chunk): {len(paragraph_result.chunks)} chunks")
        print(f"   Fixed size chunker (40 chars): {len(fixed_result.chunks)} chunks")

        print_chunks_detailed(sentence_result.chunks, "Sentence Chunker")
        print_chunks_detailed(paragraph_result.chunks, "Paragraph Chunker")
        print_chunks_detailed(fixed_result.chunks, "Fixed Size Chunker")

        # Basic assertions
        assert len(sentence_result.chunks) >= 1
        assert len(paragraph_result.chunks) >= 1
        assert len(fixed_result.chunks) >= 1

        print(f"\n✅ Comparison test passed! All chunkers working.")


if __name__ == "__main__":
    # Can also be run directly
    import pytest
    pytest.main([__file__, "-s", "-v"])
