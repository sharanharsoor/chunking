#!/usr/bin/env python3
"""
🔍 EMBEDDING SETUP VERIFICATION

Run this script to verify that your embedding setup is working correctly.
This will test all critical components needed for embeddings.
"""

import sys
from pathlib import Path

def test_dependencies():
    """Test if all required dependencies are available."""
    print("📦 Testing Dependencies...")
    print("-" * 40)
    
    dependencies = {
        "sentence_transformers": "text embeddings",
        "transformers": "transformer models", 
        "torch": "ML framework",
        "numpy": "numerical computing",
        "huggingface_hub": "model downloads"
    }
    
    all_good = True
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"✅ {package:<20} - {description}")
        except ImportError:
            print(f"❌ {package:<20} - MISSING: {description}")
            all_good = False
    
    return all_good


def test_huggingface_auth():
    """Test HuggingFace authentication."""
    print("\n🔑 Testing HuggingFace Authentication...")
    print("-" * 40)
    
    try:
        from chunking_strategy.core.embeddings import get_huggingface_token
        
        token = get_huggingface_token()
        if token and token != "YOUR_TOKEN_HERE":
            print("✅ HuggingFace token found")
            
            # Try to authenticate
            try:
                from huggingface_hub import whoami
                user_info = whoami(token)
                print(f"✅ Authentication successful - Hello {user_info['name']}!")
                return True
            except Exception as e:
                print(f"⚠️  Token found but authentication failed: {e}")
                return False
        else:
            print("❌ No HuggingFace token found")
            print("   Please set up your token:")
            print("   1. Visit: https://huggingface.co/settings/tokens")
            print("   2. Copy config/huggingface_token.py.template to config/huggingface_token.py")
            print("   3. Add your token to the file")
            return False
            
    except Exception as e:
        print(f"❌ Error testing authentication: {e}")
        return False


def test_basic_embedding():
    """Test basic embedding functionality."""
    print("\n🔮 Testing Basic Embeddings...")
    print("-" * 40)
    
    try:
        from chunking_strategy.orchestrator import ChunkerOrchestrator
        from chunking_strategy.core.embeddings import (
            EmbeddingConfig, EmbeddingModel, embed_chunking_result
        )
        
        # Create test content
        test_file = "temp_test_embedding.txt"
        with open(test_file, "w") as f:
            f.write("This is a test document for embedding verification. "
                   "It contains multiple sentences to test the chunking and embedding process. "
                   "If you can see this working, your setup is perfect!")
        
        print("📝 Creating chunks...")
        orchestrator = ChunkerOrchestrator()
        chunks = orchestrator.chunk_file(test_file)
        print(f"   ✅ Created {chunks.total_chunks} chunks")
        
        print("🔮 Generating embeddings...")
        config = EmbeddingConfig(
            model=EmbeddingModel.ALL_MINILM_L6_V2,  # Fast model for testing
            batch_size=8,
            output_format="vector_plus_text"
        )
        
        embedding_result = embed_chunking_result(chunks, config)
        
        if embedding_result.total_chunks > 0:
            print(f"   ✅ Generated {embedding_result.total_chunks} embeddings")
            print(f"   ✅ Embedding dimensions: {embedding_result.embedding_dim}D")
            print(f"   ✅ Success rate: {embedding_result.success_rate:.1%}")
            print(f"   ✅ Processing time: {embedding_result.processing_time_seconds:.2f}s")
            
            # Show first embedding preview
            first_chunk = embedding_result.embedded_chunks[0]
            preview = first_chunk.embedding[:3]
            print(f"   ✅ Sample vector: [{preview[0]:.4f}, {preview[1]:.4f}, {preview[2]:.4f}, ...]")
            
            success = True
        else:
            print("   ❌ No embeddings were generated")
            success = False
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        return success
        
    except Exception as e:
        print(f"   ❌ Embedding test failed: {e}")
        Path("temp_test_embedding.txt").unlink(missing_ok=True)  # Cleanup on error
        return False


def test_vector_database_export():
    """Test vector database export functionality."""
    print("\n💾 Testing Vector Database Export...")
    print("-" * 40)
    
    try:
        from chunking_strategy.orchestrator import ChunkerOrchestrator
        from chunking_strategy.core.embeddings import (
            EmbeddingConfig, EmbeddingModel, embed_chunking_result, export_for_vector_db
        )
        
        # Quick embedding test
        test_content = "Quick test for vector export. Another sentence here."
        test_file = "temp_vector_test.txt"
        
        with open(test_file, "w") as f:
            f.write(test_content)
        
        orchestrator = ChunkerOrchestrator()
        chunks = orchestrator.chunk_file(test_file)
        
        config = EmbeddingConfig(
            model=EmbeddingModel.ALL_MINILM_L6_V2,
            output_format="full_metadata"
        )
        
        embedding_result = embed_chunking_result(chunks, config)
        
        if embedding_result.total_chunks > 0:
            # Test export
            vector_data = export_for_vector_db(embedding_result)
            
            if vector_data:
                sample = vector_data[0]
                print(f"   ✅ Exported {len(vector_data)} vector entries")
                print(f"   ✅ Sample entry keys: {list(sample.keys())}")
                print(f"   ✅ Vector dimensions: {len(sample['vector'])}D")
                print(f"   ✅ Payload keys: {len(sample['payload'])} metadata fields")
                print("   ✅ Ready for: Qdrant, Weaviate, Pinecone, ChromaDB")
                success = True
            else:
                print("   ❌ Export returned empty data")
                success = False
        else:
            print("   ❌ No embeddings to export")
            success = False
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        return success
        
    except Exception as e:
        print(f"   ❌ Vector export test failed: {e}")
        Path("temp_vector_test.txt").unlink(missing_ok=True)
        return False


def main():
    """Run all verification tests."""
    print("🔍 CHUNKING-STRATEGY EMBEDDING SETUP VERIFICATION")
    print("=" * 60)
    print("This script will test if your embedding setup is working correctly.\n")
    
    # Run all tests
    tests = [
        ("Dependencies", test_dependencies),
        ("HuggingFace Auth", test_huggingface_auth), 
        ("Basic Embeddings", test_basic_embedding),
        ("Vector DB Export", test_vector_database_export),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION RESULTS")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your embedding setup is working perfectly!")
        print("You're ready to generate embeddings for your documents!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("Please check the error messages above and:")
        print("1. Install missing dependencies: pip install chunking-strategy[text,ml]")
        print("2. Set up HuggingFace token (see EMBEDDING_SETUP.md)")
        print("3. Check your internet connection for model downloads")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
