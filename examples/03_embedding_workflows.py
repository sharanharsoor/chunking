#!/usr/bin/env python3
"""
Embedding Workflows - From Chunking to Vector Database

This script demonstrates the complete workflow from chunking to embedding generation,
covering text and multimodal embeddings for RAG systems and vector databases.

Run with: python examples/03_embedding_workflows.py

Dependencies (optional - script works without them):
    pip install 'chunking-strategy[text]'  # For sentence-transformers
    pip install 'chunking-strategy[ml]'    # For CLIP and other models
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Import chunking and embedding functionality
from chunking_strategy import (
    ChunkerOrchestrator,
    EmbeddingModel,
    OutputFormat,
    EmbeddingConfig,
    embed_chunking_result,
    print_embedding_summary,
    export_for_vector_db,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_ml_dependencies() -> bool:
    """Check if ML dependencies are available."""
    try:
        import sentence_transformers
        import torch
        import transformers
        return True
    except ImportError:
        return False


def example_1_basic_text_embedding():
    """Basic text embedding workflow."""
    print("\n🎯 Example 1: Basic Text Embedding")
    print("=" * 50)

    # Chunk the document
    orchestrator = ChunkerOrchestrator()
    chunking_result = orchestrator.chunk_file("test_data/technical_doc.txt")

    print(f"📝 Generated {len(chunking_result.chunks)} chunks")

    # Configure embeddings
    config = EmbeddingConfig(
        model=EmbeddingModel.ALL_MINILM_L6_V2,
        output_format=OutputFormat.VECTOR_PLUS_TEXT,
        batch_size=8
    )

    print("🔮 Generating embeddings...")

    try:
        # Generate embeddings
        embedding_result = embed_chunking_result(chunking_result, config)

        # Print summary
        print_embedding_summary(embedding_result, max_chunks=3)

        # Show vector database format
        vector_db_data = export_for_vector_db(embedding_result)
        print(f"\n💾 Vector DB Format: {len(vector_db_data)} entries ready for:")
        print("   • Qdrant • Weaviate • Pinecone • ChromaDB")

        if vector_db_data:
            sample = vector_db_data[0]
            print(f"   Sample entry keys: {list(sample.keys())}")
            print(f"   Vector dimension: {len(sample.get('vector', []))}")

    except Exception as e:
        print(f"ℹ️  ML dependencies not available: {e}")
        print("💡 Install with: pip install 'chunking-strategy[text]'")

        # Show mock structure
        print("\n🎭 Mock Embedding Structure:")
        print("   EmbeddingResult:")
        print("     - embedded_chunks: List[EmbeddedChunk]")
        print("     - total_chunks: int")
        print("     - embedding_dim: int (typically 384 or 768)")
        print("     - model_used: str")


def example_2_multimodal_embeddings():
    """Multimodal (text + image) embedding with CLIP."""
    print("\n🎯 Example 2: Multimodal Embeddings (CLIP)")
    print("=" * 50)

    # Process a PDF that might contain both text and images
    orchestrator = ChunkerOrchestrator()

    pdf_path = "test_data/example.pdf"
    if not Path(pdf_path).exists():
        print("❌ PDF file not found - skipping multimodal example")
        return

    chunking_result = orchestrator.chunk_file(pdf_path)
    print(f"📑 Generated {len(chunking_result.chunks)} PDF chunks")

    # Configure for multimodal embeddings
    config = EmbeddingConfig(
        model=EmbeddingModel.CLIP_VIT_B_32,
        output_format=OutputFormat.FULL_METADATA,
        batch_size=4
    )

    try:
        embedding_result = embed_chunking_result(chunking_result, config)

        print(f"🎨 Generated {embedding_result.total_chunks} multimodal embeddings")
        print(f"📐 Vector dimension: {embedding_result.embedding_dim}")

        # Show metadata preservation
        if embedding_result.embedded_chunks:
            sample_chunk = embedding_result.embedded_chunks[0]
            metadata = sample_chunk.metadata
            # Handle different metadata types safely
            if hasattr(metadata, 'dict'):
                metadata_keys = list(metadata.dict().keys())
            elif hasattr(metadata, 'model_dump'):
                metadata_keys = list(metadata.model_dump().keys())
            elif isinstance(metadata, dict):
                metadata_keys = list(metadata.keys())
            else:
                metadata_keys = ['unknown']
            print(f"📊 Preserved metadata: {metadata_keys}")

    except Exception as e:
        print(f"ℹ️  Multimodal dependencies not available: {e}")
        print("💡 Install with: pip install 'chunking-strategy[ml]'")


def example_3_batch_embedding_workflow():
    """Process multiple files with embeddings."""
    print("\n🎯 Example 3: Batch Embedding Workflow")
    print("=" * 50)

    orchestrator = ChunkerOrchestrator()

    # Process multiple files
    test_files = [
        "test_data/technical_doc.txt",
        "test_data/alice_wonderland.txt",
        "test_data/sample_code.py"
    ]

    all_results = []
    total_chunks = 0

    for file_path in test_files:
        if not Path(file_path).exists():
            continue

        # Chunk the file
        chunking_result = orchestrator.chunk_file(file_path)
        print(f"📁 {Path(file_path).name}: {len(chunking_result.chunks)} chunks")

        total_chunks += len(chunking_result.chunks)

        # Configure embeddings (different models for different content)
        if file_path.endswith('.py'):
            model = EmbeddingModel.ALL_DISTILROBERTA_V1  # Good for code
        else:
            model = EmbeddingModel.ALL_MINILM_L6_V2      # Good for text

        config = EmbeddingConfig(
            model=model,
            output_format=OutputFormat.VECTOR_PLUS_TEXT
        )

        try:
            embedding_result = embed_chunking_result(chunking_result, config)
            all_results.append((file_path, embedding_result))
        except Exception as e:
            print(f"   ❌ Embedding failed: {str(e)[:50]}...")

    print(f"\n📊 Batch Results: {len(all_results)} files, {total_chunks} total chunks")

    if all_results and check_ml_dependencies():
        # Demonstrate vector database preparation
        print("\n💾 Preparing for Vector Database:")

        for file_path, embedding_result in all_results:
            vector_data = export_for_vector_db(embedding_result)
            print(f"   {Path(file_path).name}: {len(vector_data)} vectors ready")


def example_4_rag_system_preparation():
    """Prepare embeddings specifically for RAG systems."""
    print("\n🎯 Example 4: RAG System Preparation")
    print("=" * 50)

    # Use RAG-optimized configuration
    config_path = "config_examples/rag_system.yaml"

    if Path(config_path).exists():
        orchestrator = ChunkerOrchestrator(config_path=config_path)
        print("📋 Using RAG-optimized configuration")
    else:
        orchestrator = ChunkerOrchestrator()
        print("📋 Using default configuration")

    # Process documents for RAG
    rag_files = ["test_data/technical_doc.txt", "test_data/alice_wonderland.txt"]

    for file_path in rag_files:
        if not Path(file_path).exists():
            continue

        # Chunk with RAG-appropriate settings
        chunking_result = orchestrator.chunk_file(file_path)

        # Use embedding model optimized for retrieval
        config = EmbeddingConfig(
            model=EmbeddingModel.ALL_MPNET_BASE_V2,  # High quality for RAG
            output_format=OutputFormat.FULL_METADATA,
            normalize_embeddings=True  # Important for similarity search
        )

        try:
            embedding_result = embed_chunking_result(chunking_result, config)

            # Export in format ready for vector database
            vector_data = export_for_vector_db(embedding_result, format='dict')

            print(f"🎯 {Path(file_path).name}:")
            print(f"   📝 Chunks: {embedding_result.total_chunks}")
            print(f"   📐 Dimensions: {embedding_result.embedding_dim}")
            print(f"   💾 Ready for vector DB: {len(vector_data)} entries")

            # Show what the RAG system would get
            if vector_data:
                sample = vector_data[0]
                payload = sample.get('payload', {})
                print(f"   📊 Metadata: {list(payload.keys())}")

        except Exception as e:
            print(f"   ❌ {Path(file_path).name}: {str(e)[:50]}...")


def main():
    """Run all embedding workflow examples."""
    print("🚀 EMBEDDING WORKFLOWS")
    print("=" * 60)

    # Check dependencies
    has_ml = check_ml_dependencies()
    if has_ml:
        print("✅ ML dependencies available - will show real embeddings!")
    else:
        print("ℹ️  ML dependencies not available - will show structure and mock data")
        print("💡 For real embeddings: pip install 'chunking-strategy[text]' 'chunking-strategy[ml]'")

    example_1_basic_text_embedding()
    example_2_multimodal_embeddings()
    example_3_batch_embedding_workflow()
    example_4_rag_system_preparation()

    print("\n✅ All embedding examples completed!")
    if not has_ml:
        print("💡 Install ML dependencies to see actual embeddings and vectors!")


if __name__ == "__main__":
    main()
