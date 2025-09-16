#!/usr/bin/env python3
"""
Sentiment-Based Chunker Example

This is a simple example of a custom chunking algorithm that groups content
based on sentiment polarity. Used for demonstration purposes in configuration files.
"""

import re
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType


class SentimentBasedChunker(BaseChunker):
    """
    Simple sentiment-based chunker for demonstration.
    Groups sentences by estimated sentiment polarity.
    """

    def __init__(self,
                 name: str = "sentiment_based",
                 sentiment_threshold: float = 0.3,
                 min_sentences_per_chunk: int = 2,
                 max_sentences_per_chunk: int = 5,
                 **kwargs):
        super().__init__(
            name=name,
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )
        self.sentiment_threshold = sentiment_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk

        # Simple positive/negative word lists for demo
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'perfect',
            'best', 'awesome', 'brilliant', 'outstanding', 'superb'
        }

        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry',
            'sad', 'disappointed', 'frustrated', 'annoyed', 'worst', 'failed',
            'problem', 'issue', 'error', 'wrong', 'broken', 'difficult'
        }

    def chunk(self, content: str) -> ChunkingResult:
        """Group sentences by sentiment similarity."""
        # Split into sentences
        sentences = self._split_into_sentences(content)

        if len(sentences) <= self.min_sentences_per_chunk:
            # Too few sentences, return as single chunk
            chunk = Chunk(
                id="sentiment_0",
                content=content.strip(),
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source="unknown",
                    extra={
                        "sentiment_group": "mixed",
                        "sentence_count": len(sentences),
                        "chunker_type": "sentiment_based"
                    }
                )
            )
            return ChunkingResult(
                chunks=[chunk],
                strategy_used="sentiment_based",
                source_info={"chunker_type": "sentiment_based", "total_sentences": len(sentences)}
            )

        # Analyze sentiment for each sentence
        sentence_sentiments = []
        for sentence in sentences:
            sentiment = self._analyze_sentiment(sentence)
            sentence_sentiments.append((sentence, sentiment))

        # Group sentences by sentiment
        sentiment_groups = self._group_by_sentiment(sentence_sentiments)

        # Create chunks from sentiment groups
        chunks = []
        for group_id, (sentiment_label, group_sentences) in enumerate(sentiment_groups.items()):
            if not group_sentences:
                continue

            content_text = ' '.join([s[0] for s in group_sentences])
            avg_sentiment = sum(s[1] for s in group_sentences) / len(group_sentences)

            chunk = Chunk(
                id=f"sentiment_{group_id}",
                content=content_text,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source="unknown",
                    extra={
                        "sentiment_group": sentiment_label,
                        "sentiment_score": avg_sentiment,
                        "sentence_count": len(group_sentences),
                        "chunker_type": "sentiment_based"
                    }
                )
            )
            chunks.append(chunk)

        return ChunkingResult(
            chunks=chunks,
            strategy_used="sentiment_based",
            source_info={
                "chunker_type": "sentiment_based",
                "sentiment_groups": len(sentiment_groups),
                "total_sentences": len(sentences)
            }
        )

    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences using simple regex."""
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]

    def _analyze_sentiment(self, sentence: str) -> float:
        """
        Simple sentiment analysis using word lists.
        Returns: -1.0 (negative) to 1.0 (positive)
        """
        words = re.findall(r'\w+', sentence.lower())

        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)

        total_words = len(words)
        if total_words == 0:
            return 0.0

        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment_score))

    def _group_by_sentiment(self, sentence_sentiments: List[tuple]) -> Dict[str, List[tuple]]:
        """Group sentences by sentiment polarity."""
        groups = {
            "positive": [],
            "negative": [],
            "neutral": []
        }

        for sentence, sentiment in sentence_sentiments:
            if sentiment > self.sentiment_threshold:
                groups["positive"].append((sentence, sentiment))
            elif sentiment < -self.sentiment_threshold:
                groups["negative"].append((sentence, sentiment))
            else:
                groups["neutral"].append((sentence, sentiment))

        # Ensure minimum sentences per group, merge small groups
        final_groups = {}
        for label, group in groups.items():
            if len(group) >= self.min_sentences_per_chunk:
                final_groups[label] = group
            elif group:  # Non-empty small group
                # Merge with neutral if it exists, otherwise keep as-is
                if "neutral" in final_groups:
                    final_groups["neutral"].extend(group)
                else:
                    final_groups[label] = group

        return final_groups


# Register the chunker for framework integration
def get_chunkers():
    """Return available chunkers from this module."""
    return {
        "sentiment_based": SentimentBasedChunker
    }


if __name__ == "__main__":
    # Demo usage
    chunker = SentimentBasedChunker()

    test_content = """
    I love this new product! It's absolutely amazing and works perfectly.
    However, the packaging was terrible and arrived damaged.
    The customer service was helpful and resolved the issue quickly.
    Overall, I'm very satisfied with my purchase.
    """

    result = chunker.chunk(test_content)

    print("Sentiment-Based Chunking Demo:")
    print(f"Generated {len(result.chunks)} chunks")

    for i, chunk in enumerate(result.chunks):
        sentiment = chunk.metadata.extra.get("sentiment_group", "unknown")
        score = chunk.metadata.extra.get("sentiment_score", 0.0)
        print(f"\nChunk {i+1} ({sentiment}, score: {score:.2f}):")
        print(f"  {chunk.content}")