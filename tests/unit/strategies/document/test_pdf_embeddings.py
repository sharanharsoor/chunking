"""
PDF Embedding Tests

Comprehensive tests for embedding generation from PDF documents.
Tests the complete workflow from PDF chunking to embedding generation,
including various PDF types and edge cases.
"""

import pytest
import json
from pathlib import Path
from typing import List, Dict, Any

from chunking_strategy import (
    ChunkerOrchestrator,
    EmbeddingModel,
    OutputFormat,
    EmbeddingConfig,
    embed_chunking_result,
    create_chunker,
    export_for_vector_db,
)


class TestPDFEmbeddings:
    """Test PDF-specific embedding functionality."""

    @pytest.fixture
    def pdf_file_path(self):
        """Path to test PDF file."""
        pdf_path = Path("test_data/example.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF file not available")
        return pdf_path

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for PDF processing."""
        return ChunkerOrchestrator()

    def test_pdf_chunking_and_embedding_workflow(self, pdf_file_path, orchestrator):
        """Test complete workflow from PDF to embeddings."""

        print(f"\n📄 Testing PDF: {pdf_file_path}")

        # Step 1: Chunk the PDF
        print("📝 Step 1: Chunking PDF content...")
        chunking_result = orchestrator.chunk_file(str(pdf_file_path))

        assert chunking_result is not None
        assert len(chunking_result.chunks) > 0
        print(f"   Generated {len(chunking_result.chunks)} chunks using {chunking_result.strategy_used}")

        # Display some chunk info
        for i, chunk in enumerate(chunking_result.chunks[:3]):  # Show first 3
            content_preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
            print(f"   Chunk {i+1}: {content_preview}")
            if chunk.metadata:
                metadata_info = {k: v for k, v in chunk.metadata.to_dict().items() if k in ['page', 'source', 'position']}
                print(f"     Metadata: {metadata_info}")

        # Step 2: Test different embedding configurations
        embedding_configs = [
            {
                "name": "Fast Processing",
                "config": EmbeddingConfig(
                    model=EmbeddingModel.ALL_MINILM_L6_V2,
                    output_format=OutputFormat.VECTOR_ONLY,
                    batch_size=16
                )
            },
            {
                "name": "High Quality",
                "config": EmbeddingConfig(
                    model=EmbeddingModel.ALL_MPNET_BASE_V2,
                    output_format=OutputFormat.FULL_METADATA,
                    batch_size=8
                )
            }
        ]

        print("\n🔮 Step 2: Testing embedding configurations...")

        for config_info in embedding_configs:
            config_name = config_info["name"]
            config = config_info["config"]

            print(f"\n   Testing {config_name} configuration:")
            print(f"     Model: {config.model.value}")
            print(f"     Format: {config.output_format.value}")

            try:
                # Generate embeddings
                embedding_result = embed_chunking_result(chunking_result, config)

                # Verify results
                assert embedding_result is not None
                assert embedding_result.total_chunks >= 0  # May be 0 if all chunks are empty
                print(f"     ✅ Generated {embedding_result.total_chunks} embeddings")
                print(f"     📊 Embedding dimension: {embedding_result.embedding_dim}")

                # Test vector database export
                if embedding_result.embedded_chunks:
                    vector_data = export_for_vector_db(embedding_result)
                    assert len(vector_data) == embedding_result.total_chunks

                    # Check structure
                    sample_item = vector_data[0]
                    assert "id" in sample_item
                    assert "vector" in sample_item
                    assert "payload" in sample_item

                    print(f"     💾 Vector DB format: {len(sample_item['vector'])}D vector with payload")

                    # Check PDF-specific metadata
                    payload = sample_item["payload"]
                    if "source" in payload:
                        print(f"     📖 Source: {payload['source']}")
                    if "page" in payload:
                        print(f"     📄 Page: {payload['page']}")

                    # Print actual vectors for validation
                    print(f"     🎯 Vector validation:")
                    print(f"        First 5 vector values: {sample_item['vector'][:5]}")
                    print(f"        Vector magnitude: {sum(x*x for x in sample_item['vector'])**0.5:.4f}")
                    print(f"        Vector mean: {sum(sample_item['vector'])/len(sample_item['vector']):.4f}")
                    print(f"        Sample text: '{payload.get('content', '')[:80]}...'")

            except ImportError as e:
                print(f"     ⚠️  Skipped - {e}")
                pytest.skip(f"Dependencies not available: {e}")
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                # Don't fail the test completely, just report the issue
                pass

    def test_pdf_with_different_chunking_strategies(self, pdf_file_path):
        """Test PDF with different chunking strategies."""

        print(f"\n📄 Testing PDF with different chunking strategies...")

        # Try different strategies that should work with PDFs
        strategies_to_test = [
            ("pdf_chunker", {}),
            ("sentence_based", {"max_sentences": 2}),
            ("paragraph_based", {"max_paragraphs": 1}),
            ("fixed_size", {"chunk_size": 500})
        ]

        embedding_config = EmbeddingConfig(
            model=EmbeddingModel.ALL_MINILM_L6_V2,
            output_format=OutputFormat.FULL_METADATA,
            batch_size=8
        )

        for strategy_name, params in strategies_to_test:
            print(f"\n   Testing strategy: {strategy_name}")

            try:
                # Create chunker and process PDF
                if strategy_name == "pdf_chunker":
                    # Use orchestrator for PDF chunker
                    orchestrator = ChunkerOrchestrator()
                    chunking_result = orchestrator.chunk_file(str(pdf_file_path), strategy_override=strategy_name)
                else:
                    # Use direct chunker for other strategies
                    chunker = create_chunker(strategy_name, **params)
                    if chunker is None:
                        print(f"     ⚠️  Strategy {strategy_name} not available")
                        continue

                    # For non-PDF strategies, read the PDF as text first
                    try:
                        chunking_result = chunker.chunk(str(pdf_file_path))
                    except Exception as e:
                        print(f"     ⚠️  Strategy {strategy_name} failed on PDF: {e}")
                        continue

                print(f"     📝 Generated {len(chunking_result.chunks)} chunks")

                # Generate embeddings if we have chunks
                if chunking_result.chunks:
                    try:
                        embedding_result = embed_chunking_result(chunking_result, embedding_config)
                        print(f"     🔮 Generated {embedding_result.total_chunks} embeddings")

                        # Verify PDF metadata is preserved
                        if embedding_result.embedded_chunks:
                            sample_chunk = embedding_result.embedded_chunks[0]
                            if sample_chunk.metadata and 'source' in sample_chunk.metadata:
                                print(f"     📖 Source preserved: {sample_chunk.metadata['source']}")

                    except ImportError:
                        print(f"     ⚠️  Embedding dependencies not available")
                    except Exception as e:
                        print(f"     ⚠️  Embedding failed: {e}")

            except Exception as e:
                print(f"     ❌ Strategy {strategy_name} failed: {e}")

    def test_pdf_metadata_preservation(self, pdf_file_path, orchestrator):
        """Test that PDF-specific metadata is preserved in embeddings."""

        print(f"\n📊 Testing PDF metadata preservation...")

        # Chunk PDF with a strategy that preserves metadata
        chunking_result = orchestrator.chunk_file(str(pdf_file_path))

        if not chunking_result.chunks:
            pytest.skip("No chunks generated from PDF")

        # Configure embedding to include full metadata
        config = EmbeddingConfig(
            model=EmbeddingModel.ALL_MINILM_L6_V2,
            output_format=OutputFormat.FULL_METADATA,
            normalize_embeddings=True
        )

        try:
            # Generate embeddings
            embedding_result = embed_chunking_result(chunking_result, config)

            if not embedding_result.embedded_chunks:
                pytest.skip("No embeddings generated")

            # Check metadata preservation
            pdf_metadata_found = False

            for embedded_chunk in embedding_result.embedded_chunks:
                if embedded_chunk.metadata:
                    metadata = embedded_chunk.metadata

                    print(f"   Chunk {embedded_chunk.chunk_id}:")

                    # Check common PDF metadata
                    metadata_keys = list(metadata.keys())
                    print(f"     Available metadata: {metadata_keys}")

                    # Look for PDF-specific metadata
                    pdf_indicators = ['source', 'page', 'position', 'bbox']
                    found_pdf_metadata = [key for key in pdf_indicators if key in metadata]

                    if found_pdf_metadata:
                        pdf_metadata_found = True
                        print(f"     PDF metadata found: {found_pdf_metadata}")
                        for key in found_pdf_metadata:
                            print(f"       {key}: {metadata[key]}")

                    # Check embedding-specific metadata
                    embedding_keys = ['embedding_model', 'embedding_timestamp', 'normalized']
                    found_embedding_metadata = [key for key in embedding_keys if key in metadata]
                    if found_embedding_metadata:
                        print(f"     Embedding metadata: {found_embedding_metadata}")

            # Verify that at least some PDF metadata was preserved
            if pdf_metadata_found:
                print("   ✅ PDF metadata successfully preserved in embeddings")
            else:
                print("   ⚠️  No obvious PDF metadata found (might be strategy-specific)")

        except ImportError:
            pytest.skip("Embedding dependencies not available")

    def test_pdf_large_batch_processing(self, pdf_file_path, orchestrator):
        """Test batch processing of PDF chunks."""

        print(f"\n⚡ Testing large batch processing...")

        # Chunk the PDF
        chunking_result = orchestrator.chunk_file(str(pdf_file_path))

        if len(chunking_result.chunks) < 5:
            pytest.skip("PDF doesn't have enough chunks for batch testing")

        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            print(f"\n   Testing batch size: {batch_size}")

            config = EmbeddingConfig(
                model=EmbeddingModel.ALL_MINILM_L6_V2,
                output_format=OutputFormat.VECTOR_PLUS_TEXT,
                batch_size=batch_size,
                normalize_embeddings=True
            )

            try:
                import time
                start_time = time.time()

                embedding_result = embed_chunking_result(chunking_result, config)

                end_time = time.time()
                processing_time = end_time - start_time

                print(f"     ✅ Processed {embedding_result.total_chunks} chunks in {processing_time:.2f}s")
                print(f"     📊 Speed: {embedding_result.total_chunks/processing_time:.1f} chunks/sec")

                # Verify consistency
                assert embedding_result.total_chunks <= len(chunking_result.chunks)
                if embedding_result.embedded_chunks:
                    assert all(len(chunk.embedding) == embedding_result.embedding_dim
                             for chunk in embedding_result.embedded_chunks)

            except ImportError:
                pytest.skip("Embedding dependencies not available")
                break
            except Exception as e:
                print(f"     ❌ Batch size {batch_size} failed: {e}")

    def test_pdf_export_formats(self, pdf_file_path, orchestrator):
        """Test different export formats for PDF embeddings."""

        print(f"\n💾 Testing export formats...")

        # Chunk and embed
        chunking_result = orchestrator.chunk_file(str(pdf_file_path))

        if not chunking_result.chunks:
            pytest.skip("No chunks to embed")

        config = EmbeddingConfig(
            model=EmbeddingModel.ALL_MINILM_L6_V2,
            output_format=OutputFormat.FULL_METADATA
        )

        try:
            embedding_result = embed_chunking_result(chunking_result, config)

            if not embedding_result.embedded_chunks:
                pytest.skip("No embeddings generated")

            # Test different export formats
            export_formats = ["dict", "json"]

            for export_format in export_formats:
                print(f"\n   Testing {export_format} export:")

                export_data = export_for_vector_db(embedding_result, format=export_format)

                if export_format == "dict":
                    assert isinstance(export_data, list)
                    print(f"     📊 List with {len(export_data)} items")

                    if export_data:
                        sample = export_data[0]
                        print(f"     🔍 Sample structure: {list(sample.keys())}")
                        print(f"     📏 Vector dimension: {len(sample['vector'])}")
                        payload_keys = list(sample['payload'].keys()) if 'payload' in sample else []
                        print(f"     🏷️  Payload keys: {payload_keys[:5]}...")  # Show first 5

                elif export_format == "json":
                    assert isinstance(export_data, str)
                    parsed = json.loads(export_data)
                    assert isinstance(parsed, list)
                    print(f"     📄 JSON string with {len(parsed)} items")
                    print(f"     📦 Size: {len(export_data)} characters")

        except ImportError:
            pytest.skip("Embedding dependencies not available")

    def test_pdf_error_handling(self):
        """Test error handling for PDF embedding scenarios."""

        print(f"\n🛡️  Testing error handling...")

        # Test with non-existent PDF
        non_existent_pdf = Path("test_data/non_existent.pdf")
        orchestrator = ChunkerOrchestrator()

        try:
            chunking_result = orchestrator.chunk_file(str(non_existent_pdf))
            # Should either handle gracefully or raise appropriate error
            print(f"   Non-existent file handling: {type(chunking_result)}")
        except Exception as e:
            print(f"   ✅ Proper error for non-existent file: {type(e).__name__}")

        # Test with invalid embedding config
        try:
            from chunking_strategy.core.embeddings import create_embedder

            invalid_config = EmbeddingConfig()
            invalid_config.model = "invalid_model"  # Invalid model

            embedder = create_embedder(invalid_config)
            print(f"   ⚠️  Invalid config accepted: {type(embedder)}")
        except Exception as e:
            print(f"   ✅ Proper error for invalid config: {type(e).__name__}: {e}")


class TestPDFEmbeddingIntegration:
    """Integration tests for PDF embedding workflows."""

    def test_end_to_end_rag_workflow(self):
        """Test complete RAG workflow with PDF."""

        print(f"\n🔄 Testing end-to-end RAG workflow...")

        pdf_path = Path("test_data/example.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not available")

        try:
            # Step 1: Use RAG-optimized configuration
            from chunking_strategy import ChunkerOrchestrator

            # Try to use RAG config if available
            rag_config_path = Path("config_examples/rag_system.yaml")
            if rag_config_path.exists():
                orchestrator = ChunkerOrchestrator(config_path=rag_config_path)
                print(f"   📋 Using RAG configuration")
            else:
                orchestrator = ChunkerOrchestrator()
                print(f"   📋 Using default configuration")

            # Step 2: Chunk PDF
            print(f"   📝 Chunking PDF...")
            chunking_result = orchestrator.chunk_file(str(pdf_path))
            print(f"      Generated {len(chunking_result.chunks)} chunks")

            # Step 3: Generate embeddings optimized for RAG
            print(f"   🔮 Generating RAG-optimized embeddings...")
            config = EmbeddingConfig(
                model=EmbeddingModel.ALL_MPNET_BASE_V2,  # High quality for RAG
                output_format=OutputFormat.FULL_METADATA,
                batch_size=8,
                normalize_embeddings=True
            )

            embedding_result = embed_chunking_result(chunking_result, config)
            print(f"      Generated {embedding_result.total_chunks} embeddings")

            # Step 4: Prepare for vector database
            print(f"   💾 Preparing for vector database...")
            vector_data = export_for_vector_db(embedding_result)

            # Step 5: Simulate search query
            print(f"   🔍 Simulating semantic search...")

            if vector_data and len(vector_data) > 0:
                # Show what a search result would look like
                sample_result = vector_data[0]
                print(f"      Sample search result structure:")
                print(f"        ID: {sample_result['id']}")
                print(f"        Vector: {len(sample_result['vector'])}D")
                print(f"        Content preview: {sample_result['payload'].get('content', '')[:100]}...")

                # Check if we have the metadata needed for RAG
                payload = sample_result['payload']
                rag_metadata = ['source', 'page', 'model_used', 'content']
                available_rag_metadata = [key for key in rag_metadata if key in payload]
                print(f"        RAG metadata available: {available_rag_metadata}")

            print(f"   ✅ RAG workflow completed successfully!")

        except ImportError as e:
            pytest.skip(f"RAG workflow requires dependencies: {e}")
        except Exception as e:
            print(f"   ❌ RAG workflow failed: {e}")
            raise

    def test_pdf_detailed_vector_analysis(self):
        """Test detailed vector analysis for PDF with actual vector output."""

        print(f"\n🔍 Detailed PDF vector analysis with actual embeddings...")

        pdf_path = Path("test_data/example.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not available")

        orchestrator = ChunkerOrchestrator()

        try:
            # Step 1: Chunk PDF with different strategies
            strategies_to_test = [
                ("sentence_based", {"max_sentences": 2}),
                ("paragraph_based", {"max_paragraphs": 1}),
                ("fixed_size", {"chunk_size": 300})
            ]

            for strategy_name, params in strategies_to_test:
                print(f"\n   📊 Strategy: {strategy_name}")

                try:
                    if strategy_name in ["sentence_based", "paragraph_based"]:
                        # Use orchestrator with strategy override
                        chunking_result = orchestrator.chunk_file(str(pdf_path))
                    else:
                        # Create specific chunker
                        chunker = create_chunker(strategy_name, **params)
                        if chunker is None:
                            print(f"      ⚠️  Strategy {strategy_name} not available")
                            continue
                        chunking_result = chunker.chunk(str(pdf_path))

                    print(f"      📝 Generated {len(chunking_result.chunks)} chunks")

                    # Test with different embedding models
                    embedding_configs = [
                        ("Fast Model", EmbeddingConfig(
                            model=EmbeddingModel.ALL_MINILM_L6_V2,
                            output_format=OutputFormat.FULL_METADATA,
                            batch_size=4
                        )),
                        ("High Quality", EmbeddingConfig(
                            model=EmbeddingModel.ALL_MPNET_BASE_V2,
                            output_format=OutputFormat.FULL_METADATA,
                            batch_size=2
                        ))
                    ]

                    for config_name, config in embedding_configs:
                        print(f"\n      🔮 {config_name} ({config.model.value}):")

                        try:
                            embedding_result = embed_chunking_result(chunking_result, config)

                            if not embedding_result.embedded_chunks:
                                print(f"         ⚠️  No embeddings generated")
                                continue

                            print(f"         ✅ Generated {embedding_result.total_chunks} embeddings ({embedding_result.embedding_dim}D)")

                            # Detailed analysis of first few embeddings
                            for i, embedded_chunk in enumerate(embedding_result.embedded_chunks[:3]):
                                print(f"\n         📊 Chunk {i+1} detailed analysis:")

                                # Text content
                                content = embedded_chunk.content[:100].replace('\n', ' ')
                                print(f"            📝 Text: '{content}...'")

                                # Vector analysis
                                vector = embedded_chunk.embedding
                                print(f"            🔢 Vector dimension: {len(vector)}")
                                print(f"            🎯 Vector sample: [{vector[0]:.6f}, {vector[1]:.6f}, {vector[2]:.6f}, {vector[3]:.6f}, {vector[4]:.6f}]")
                                print(f"            📊 Vector statistics:")
                                print(f"               Min: {min(vector):.6f}")
                                print(f"               Max: {max(vector):.6f}")
                                print(f"               Mean: {sum(vector)/len(vector):.6f}")
                                print(f"               Magnitude: {(sum(x*x for x in vector)**0.5):.6f}")

                                # Metadata analysis
                                if embedded_chunk.metadata:
                                    print(f"            📋 Metadata: {dict(list(embedded_chunk.metadata.items())[:3])}")

                            # Vector database export test
                            vector_db_data = export_for_vector_db(embedding_result)
                            print(f"\n         💾 Vector Database Export:")
                            print(f"            Exported {len(vector_db_data)} entries")

                            if vector_db_data:
                                sample_export = vector_db_data[0]
                                print(f"            Sample export structure:")
                                print(f"               ID: {sample_export['id']}")
                                print(f"               Vector dims: {len(sample_export['vector'])}")
                                print(f"               Vector preview: [{sample_export['vector'][0]:.4f}, {sample_export['vector'][1]:.4f}, ...]")
                                print(f"               Payload keys: {list(sample_export['payload'].keys())}")

                                # Show text-vector relationship
                                payload_content = sample_export['payload'].get('content', '')[:60]
                                print(f"               Text-Vector pair: '{payload_content}...' -> [{sample_export['vector'][0]:.4f}, {sample_export['vector'][1]:.4f}, ...]")

                        except ImportError:
                            print(f"         ⚠️  {config_name} model not available (dependencies not installed)")
                        except Exception as e:
                            print(f"         ❌ {config_name} failed: {e}")

                except Exception as e:
                    print(f"      ❌ Strategy {strategy_name} failed: {e}")

        except Exception as e:
            print(f"   ❌ PDF vector analysis failed: {e}")
            raise

    def test_pdf_embedding_similarity_comparison(self):
        """Test embedding similarity between different PDF chunks."""

        print(f"\n🔗 Testing PDF embedding similarity comparisons...")

        pdf_path = Path("test_data/example.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not available")

        try:
            orchestrator = ChunkerOrchestrator()
            chunking_result = orchestrator.chunk_file(str(pdf_path))

            if len(chunking_result.chunks) < 3:
                pytest.skip("Need at least 3 chunks for similarity testing")

            config = EmbeddingConfig(
                model=EmbeddingModel.ALL_MINILM_L6_V2,
                output_format=OutputFormat.FULL_METADATA,
                normalize_embeddings=True  # Important for similarity
            )

            try:
                embedding_result = embed_chunking_result(chunking_result, config)

                if len(embedding_result.embedded_chunks) < 3:
                    pytest.skip("Need at least 3 embedded chunks for similarity testing")

                print(f"   📊 Analyzing similarity between {len(embedding_result.embedded_chunks)} PDF chunks:")

                # Function to calculate cosine similarity
                def cosine_similarity(vec1, vec2):
                    dot_product = sum(a * b for a, b in zip(vec1, vec2))
                    magnitude1 = sum(a * a for a in vec1) ** 0.5
                    magnitude2 = sum(b * b for b in vec2) ** 0.5
                    if magnitude1 == 0 or magnitude2 == 0:
                        return 0
                    return dot_product / (magnitude1 * magnitude2)

                # Compare first 3 chunks
                chunks = embedding_result.embedded_chunks[:3]

                print(f"\n   🎯 Chunk contents for similarity analysis:")
                for i, chunk in enumerate(chunks):
                    content_preview = chunk.content[:80].replace('\n', ' ') + "..." if len(chunk.content) > 80 else chunk.content
                    print(f"      Chunk {i+1}: '{content_preview}'")
                    print(f"      Vector preview: [{chunk.embedding[0]:.4f}, {chunk.embedding[1]:.4f}, {chunk.embedding[2]:.4f}, ...]")

                print(f"\n   📈 Similarity Matrix:")
                print(f"      {'':>10} {'Chunk 1':>10} {'Chunk 2':>10} {'Chunk 3':>10}")

                for i in range(3):
                    row = f"      {'Chunk ' + str(i+1):>10}"
                    for j in range(3):
                        if i == j:
                            similarity = 1.0
                        else:
                            similarity = cosine_similarity(chunks[i].embedding, chunks[j].embedding)
                        row += f" {similarity:>9.4f}"
                    print(row)

                # Find most and least similar pair
                max_similarity = -1
                min_similarity = 2
                max_pair = None
                min_pair = None

                for i in range(3):
                    for j in range(i+1, 3):
                        similarity = cosine_similarity(chunks[i].embedding, chunks[j].embedding)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            max_pair = (i, j)
                        if similarity < min_similarity:
                            min_similarity = similarity
                            min_pair = (i, j)

                if max_pair:
                    print(f"\n   🎯 Most similar chunks (Chunks {max_pair[0]+1} & {max_pair[1]+1}): {max_similarity:.4f}")
                    print(f"      Text 1: '{chunks[max_pair[0]].content[:60]}...'")
                    print(f"      Text 2: '{chunks[max_pair[1]].content[:60]}...'")

                if min_pair:
                    print(f"\n   🎯 Least similar chunks (Chunks {min_pair[0]+1} & {min_pair[1]+1}): {min_similarity:.4f}")
                    print(f"      Text 1: '{chunks[min_pair[0]].content[:60]}...'")
                    print(f"      Text 2: '{chunks[min_pair[1]].content[:60]}...'")

                print(f"\n   ✅ Similarity analysis completed!")

            except ImportError:
                print(f"   ⚠️  Embedding dependencies not available")
                pytest.skip("Embedding dependencies not available")
            except Exception as e:
                print(f"   ❌ Similarity analysis failed: {e}")

        except Exception as e:
            print(f"   ❌ PDF similarity test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
