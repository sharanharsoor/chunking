#!/usr/bin/env python3
"""
Embeddings Integration Demo

This demo shows how to integrate chunking with embeddings for RAG applications.
Demonstrates:
- Document chunking strategies optimized for embeddings
- Converting chunks to embeddings using different models
- Similarity search and retrieval
- Vector database integration patterns
- Quality evaluation of chunk-embedding pairs

This is essential for building RAG (Retrieval-Augmented Generation) systems.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking_strategy import create_chunker, ChunkerOrchestrator
from chunking_strategy.core.base import Chunk
from chunking_strategy.core.metrics import ChunkingQualityEvaluator


def setup_test_document():
    """Create a comprehensive test document for RAG evaluation."""
    document = """
# Machine Learning Fundamentals

## Introduction to Neural Networks
Neural networks are computational models inspired by biological neural networks.
They consist of interconnected nodes (neurons) that process information through weighted connections.
Deep learning utilizes multiple layers of neurons to learn complex patterns in data.

## Supervised Learning
Supervised learning involves training models on labeled datasets where input-output pairs are known.
Common algorithms include linear regression, decision trees, and support vector machines.
The goal is to generalize from training data to make accurate predictions on new, unseen data.

## Unsupervised Learning
Unsupervised learning discovers hidden patterns in data without labeled examples.
Clustering algorithms group similar data points together.
Dimensionality reduction techniques like PCA compress data while preserving important information.

## Natural Language Processing
NLP enables computers to understand and process human language.
Tokenization breaks text into smaller units like words or subwords.
Language models learn statistical patterns in text to generate coherent responses.

## Computer Vision
Computer vision allows machines to interpret and understand visual information.
Convolutional neural networks excel at recognizing patterns in images.
Object detection combines classification with spatial localization of objects.

## Reinforcement Learning
Reinforcement learning trains agents to make decisions through trial and error.
Agents receive rewards or penalties based on their actions in an environment.
Q-learning and policy gradient methods are popular reinforcement learning approaches.
"""

    # Save document to test_data if it doesn't exist
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)

    doc_path = test_data_dir / "ml_fundamentals.md"
    with open(doc_path, 'w') as f:
        f.write(document)

    return doc_path, document


def demonstrate_embedding_optimized_chunking():
    """Show different chunking strategies optimized for embeddings."""
    print("🔧 EMBEDDING-OPTIMIZED CHUNKING STRATEGIES")
    print("=" * 60)

    doc_path, document = setup_test_document()

    # Different chunking strategies for embeddings
    strategies = {
        "semantic": {
            "name": "semantic",
            "similarity_threshold": 0.8,
            "min_chunk_sentences": 2,
            "max_chunk_sentences": 5,
            "semantic_model": "tfidf"  # Use TF-IDF for demo consistency
        },
        "sentence_based": {
            "name": "sentence_based",
            "max_sentences": 3,
            "overlap": 1
        },
        "paragraph": {
            "name": "paragraph_based",
            "max_paragraphs": 1,
            "overlap_paragraphs": 0
        },
        "fixed_size": {
            "name": "fixed_size",
            "chunk_size": 300,
            "overlap": 50,
            "preserve_boundaries": True
        }
    }

    results = {}

    for strategy_name, config in strategies.items():
        print(f"\n📊 Testing {strategy_name} chunking...")

        try:
            chunker = create_chunker(**config)
            if not chunker:
                print(f"   ❌ Failed to create {strategy_name} chunker")
                continue

            start_time = time.time()
            result = chunker.chunk(document)
            processing_time = time.time() - start_time

            # Analyze chunks for embedding suitability
            chunks = result.chunks
            chunk_sizes = [len(chunk.content) for chunk in chunks]
            avg_size = np.mean(chunk_sizes)
            size_std = np.std(chunk_sizes)

            results[strategy_name] = {
                "chunks": chunks,
                "count": len(chunks),
                "avg_size": avg_size,
                "size_std": size_std,
                "processing_time": processing_time
            }

            print(f"   ✅ Generated {len(chunks)} chunks")
            print(f"   📏 Average size: {avg_size:.1f} chars (std: {size_std:.1f})")
            print(f"   ⏱️  Processing time: {processing_time:.3f}s")

            # Show first chunk as example
            if chunks:
                first_chunk = chunks[0].content.strip()
                preview = first_chunk[:100] + "..." if len(first_chunk) > 100 else first_chunk
                print(f"   📝 First chunk preview: {preview}")

        except Exception as e:
            print(f"   ❌ Error with {strategy_name}: {str(e)[:60]}...")

    return results


def simulate_embedding_generation(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    """Simulate embedding generation for chunks (using TF-IDF as proxy)."""
    print("\n🔮 SIMULATING EMBEDDING GENERATION")
    print("=" * 40)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Extract chunk contents
        chunk_texts = [chunk.content for chunk in chunks]

        # Generate TF-IDF embeddings (simulating real embeddings)
        vectorizer = TfidfVectorizer(
            max_features=384,  # Simulate sentence-transformer dimension
            stop_words='english',
            ngram_range=(1, 2)
        )

        embeddings = vectorizer.fit_transform(chunk_texts).toarray()

        # Create embedding records
        embedding_records = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            embedding_records.append({
                "chunk_id": i,
                "chunk_content": chunk.content,
                "embedding": embedding,
                "embedding_norm": np.linalg.norm(embedding),
                "content_length": len(chunk.content),
                "word_count": len(chunk.content.split())
            })

        print(f"✅ Generated {len(embedding_records)} embeddings")
        print(f"📊 Embedding dimension: {embeddings.shape[1]}")
        print(f"📏 Average embedding norm: {np.mean([r['embedding_norm'] for r in embedding_records]):.3f}")

        return embedding_records

    except ImportError:
        print("❌ scikit-learn not available for embedding simulation")
        return []


def demonstrate_similarity_search(embedding_records: List[Dict[str, Any]]):
    """Demonstrate similarity search using embeddings."""
    if not embedding_records:
        return

    print("\n🔍 SIMILARITY SEARCH DEMONSTRATION")
    print("=" * 40)

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        # Create queries
        queries = [
            "What are neural networks?",
            "How does supervised learning work?",
            "Computer vision applications"
        ]

        # Get embeddings matrix
        embeddings = np.array([record["embedding"] for record in embedding_records])

        for query in queries:
            print(f"\n🔍 Query: '{query}'")

            # Simulate query embedding (use TF-IDF on combined corpus)
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                # Combine query with chunk texts for consistent vocabulary
                all_texts = [query] + [record["chunk_content"] for record in embedding_records]

                vectorizer = TfidfVectorizer(
                    max_features=384,
                    stop_words='english',
                    ngram_range=(1, 2)
                )

                all_embeddings = vectorizer.fit_transform(all_texts).toarray()
                query_embedding = all_embeddings[0:1]  # First row is query
                chunk_embeddings = all_embeddings[1:]  # Rest are chunks

                # Calculate similarities
                similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

                # Get top 3 matches
                top_indices = np.argsort(similarities)[-3:][::-1]

                for rank, idx in enumerate(top_indices, 1):
                    score = similarities[idx]
                    chunk_preview = embedding_records[idx]["chunk_content"][:80] + "..."
                    print(f"   {rank}. Score: {score:.3f} | {chunk_preview}")

            except Exception as e:
                print(f"   ❌ Error in similarity search: {str(e)[:60]}...")

    except ImportError:
        print("❌ scikit-learn not available for similarity search")


def demonstrate_chunk_quality_for_embeddings(embedding_records: List[Dict[str, Any]]):
    """Evaluate chunk quality specifically for embedding applications."""
    if not embedding_records:
        return

    print("\n📊 CHUNK QUALITY ANALYSIS FOR EMBEDDINGS")
    print("=" * 50)

    # Analyze chunk characteristics
    content_lengths = [record["content_length"] for record in embedding_records]
    word_counts = [record["word_count"] for record in embedding_records]
    embedding_norms = [record["embedding_norm"] for record in embedding_records]

    print(f"📏 Content Length Statistics:")
    print(f"   Average: {np.mean(content_lengths):.1f} chars")
    print(f"   Std Dev: {np.std(content_lengths):.1f} chars")
    print(f"   Range: {min(content_lengths)} - {max(content_lengths)} chars")

    print(f"\n📝 Word Count Statistics:")
    print(f"   Average: {np.mean(word_counts):.1f} words")
    print(f"   Std Dev: {np.std(word_counts):.1f} words")
    print(f"   Range: {min(word_counts)} - {max(word_counts)} words")

    print(f"\n🔢 Embedding Quality Indicators:")
    print(f"   Average norm: {np.mean(embedding_norms):.3f}")
    print(f"   Norm std dev: {np.std(embedding_norms):.3f}")

    # Quality recommendations
    print(f"\n💡 QUALITY RECOMMENDATIONS:")

    if np.mean(content_lengths) < 50:
        print("   ⚠️  Chunks may be too short for good embeddings")
    elif np.mean(content_lengths) > 500:
        print("   ⚠️  Chunks may be too long for focused embeddings")
    else:
        print("   ✅ Chunk sizes are good for embeddings")

    if np.std(content_lengths) / np.mean(content_lengths) > 0.5:
        print("   ⚠️  High size variation may affect embedding quality")
    else:
        print("   ✅ Consistent chunk sizes")

    if np.std(embedding_norms) / np.mean(embedding_norms) > 0.3:
        print("   ⚠️  High embedding norm variation detected")
    else:
        print("   ✅ Consistent embedding quality")


def demonstrate_rag_pipeline():
    """Demonstrate a complete RAG pipeline with chunking."""
    print("\n🚀 COMPLETE RAG PIPELINE DEMONSTRATION")
    print("=" * 50)

    doc_path, document = setup_test_document()

    # Step 1: Optimal chunking for RAG
    print("\n1️⃣ Document Chunking (RAG-optimized)")
    chunker = create_chunker(
        name="semantic",
        similarity_threshold=0.75,
        min_chunk_sentences=2,
        max_chunk_sentences=4,
        semantic_model="tfidf"
    )

    if not chunker:
        print("❌ Failed to create semantic chunker")
        return

    result = chunker.chunk(document)
    chunks = result.chunks
    print(f"   ✅ Generated {len(chunks)} semantic chunks")

    # Step 2: Embedding generation
    print("\n2️⃣ Embedding Generation")
    embedding_records = simulate_embedding_generation(chunks)

    # Step 3: Retrieval demonstration
    print("\n3️⃣ Retrieval Demonstration")
    demonstrate_similarity_search(embedding_records)

    # Step 4: Quality analysis
    print("\n4️⃣ Quality Analysis")
    demonstrate_chunk_quality_for_embeddings(embedding_records)

    # Step 5: Performance metrics
    print("\n5️⃣ Performance Summary")
    print(f"   📊 Total chunks: {len(chunks)}")
    print(f"   📏 Average chunk size: {np.mean([len(c.content) for c in chunks]):.1f} chars")
    print(f"   🔢 Embedding dimension: {len(embedding_records[0]['embedding']) if embedding_records else 0}")
    print(f"   💾 Memory footprint: ~{len(chunks) * 384 * 4 / 1024:.1f} KB (estimated)")


def main():
    """Run the complete embeddings integration demo."""
    print("🎯 CHUNKING + EMBEDDINGS INTEGRATION DEMO")
    print("=" * 60)
    print("This demo shows how to integrate chunking with embeddings for RAG applications.\n")

    try:
        # Demo 1: Chunking strategies comparison
        chunking_results = demonstrate_embedding_optimized_chunking()

        # Demo 2: Full RAG pipeline
        demonstrate_rag_pipeline()

        print("\n" + "=" * 60)
        print("🎉 EMBEDDINGS INTEGRATION DEMO COMPLETE!")
        print("=" * 60)
        print("\n💡 Key Takeaways:")
        print("   • Semantic chunking often works best for embeddings")
        print("   • Consistent chunk sizes improve embedding quality")
        print("   • Overlap helps with context preservation")
        print("   • Quality evaluation is crucial for RAG performance")
        print("\n📚 Next Steps:")
        print("   • Experiment with different embedding models")
        print("   • Try vector databases like Pinecone, Weaviate, or Chroma")
        print("   • Implement reranking for better retrieval")
        print("   • Add metadata filtering for complex queries")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
