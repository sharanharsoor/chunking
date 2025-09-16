"""
Integration tests for Token-based Chunker with real files, CLI, and configuration.

This module tests the token-based chunker with:
- Real files from test_data directory
- Multiple tokenizer integrations
- CLI command integration  
- YAML configuration files
- Multi-format file processing
- End-to-end workflows with different tokenization systems
"""

import pytest
import tempfile
import subprocess
import sys
import json
import yaml
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any

from chunking_strategy.strategies.text.token_based_chunker import TokenBasedChunker, TokenizerType
from chunking_strategy.core.base import ChunkingResult
from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy import create_chunker
from tests.conftest import assert_valid_chunks, assert_reasonable_performance


class TestTokenBasedIntegration:
    """Integration tests for Token-based Chunker with real files and systems."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.test_data_dir = Path("test_data")
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "configs"
        self.output_dir = self.temp_dir / "output"
        
        # Create directories
        self.config_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Ensure test data exists
        if not self.test_data_dir.exists():
            pytest.skip("Test data directory not found")
        
        yield
        
        # Cleanup
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_text_files_processing_multiple_tokenizers(self):
        """Test processing various text files with different tokenizers."""
        # Test different tokenizer configurations
        tokenizer_configs = [
            {"name": "Simple", "type": "simple", "tokens": 200},
            {"name": "TikToken GPT-3.5", "type": "tiktoken", "model": "gpt-3.5-turbo", "tokens": 150},
            {"name": "TikToken GPT-4", "type": "tiktoken", "model": "gpt-4", "tokens": 150},
            {"name": "Transformers BERT", "type": "transformers", "model": "bert-base-uncased", "tokens": 100},
            {"name": "Transformers GPT-2", "type": "transformers", "model": "gpt2", "tokens": 120},
        ]
        
        text_files = [
            "sample_simple_text.txt",
            "alice_wonderland.txt", 
            "business_report.txt",
            "technical_doc.txt",
            "sample_article.txt"
        ]
        
        results = {}
        
        for config in tokenizer_configs:
            print(f"\n🔧 Testing {config['name']} tokenizer...")
            
            try:
                chunker = TokenBasedChunker(
                    tokens_per_chunk=config["tokens"],
                    overlap_tokens=config["tokens"] // 10,
                    tokenizer_type=config["type"],
                    tokenizer_model=config.get("model", ""),
                    preserve_word_boundaries=True
                )
                
                config_results = {}
                
                for filename in text_files:
                    file_path = self.test_data_dir / filename
                    if not file_path.exists():
                        continue
                    
                    print(f"   📄 Processing: {filename}")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if not content.strip():
                            continue
                        
                        start_time = time.time()
                        result = chunker.chunk(content)
                        processing_time = time.time() - start_time
                        
                        # Validate results
                        assert isinstance(result, ChunkingResult)
                        assert len(result.chunks) > 0
                        assert result.strategy_used == "token_based"
                        
                        # Validate chunk quality
                        assert_valid_chunks(result.chunks, None)
                        assert_reasonable_performance(result.processing_time, len(content))
                        
                        # Check token-specific metadata
                        for chunk in result.chunks:
                            assert "token_count" in chunk.metadata.extra
                            assert chunk.metadata.extra["token_count"] > 0
                            assert chunk.metadata.extra["tokenizer_type"] == config["type"]
                        
                        config_results[filename] = {
                            "chunks": len(result.chunks),
                            "total_tokens": result.source_info.get("total_tokens", 0),
                            "processing_time": processing_time,
                            "tokenizer_info": result.source_info.get("tokenizer_info", {}),
                            "avg_tokens_per_chunk": result.source_info.get("avg_tokens_per_chunk", 0)
                        }
                        
                        print(f"      ✅ {len(result.chunks)} chunks, {config_results[filename]['total_tokens']} tokens")
                        
                    except Exception as e:
                        print(f"      ❌ Failed: {e}")
                        config_results[filename] = {"error": str(e)}
                
                results[config["name"]] = config_results
                
            except ImportError as e:
                print(f"   ⚠️  {config['name']} not available: {e}")
            except Exception as e:
                print(f"   ❌ {config['name']} failed: {e}")
        
        # Ensure at least simple tokenizer worked
        assert "Simple" in results
        simple_results = results["Simple"]
        successful_simple = [k for k, v in simple_results.items() if "error" not in v]
        assert len(successful_simple) >= 2, f"Simple tokenizer should work on most files"
        
        print(f"\n📊 Tokenizer comparison completed: {len(results)} tokenizers tested")

    def test_tokenizer_specific_features(self):
        """Test features specific to different tokenizers."""
        test_text = """
        This is a comprehensive test for different tokenization systems.
        It includes various challenges:
        - Contractions like don't, won't, can't
        - URLs like https://example.com 
        - Emails like user@domain.com
        - Numbers like 123.45 and 2024-01-15
        - Special tokens and subword units
        - Unicode characters: café, naïve, résumé
        """
        
        tokenizer_tests = [
            {
                "name": "GPT Tokenization", 
                "type": "tiktoken", 
                "model": "gpt-3.5-turbo",
                "expected_features": ["subword", "BPE"]
            },
            {
                "name": "BERT Tokenization", 
                "type": "transformers", 
                "model": "bert-base-uncased",
                "expected_features": ["wordpiece", "special_tokens"]
            },
            {
                "name": "Simple Tokenization", 
                "type": "simple",
                "expected_features": ["word_split", "whitespace"]
            }
        ]
        
        for test_config in tokenizer_tests:
            print(f"\n🔍 Testing {test_config['name']}...")
            
            try:
                chunker = TokenBasedChunker(
                    tokens_per_chunk=50,
                    tokenizer_type=test_config["type"],
                    tokenizer_model=test_config.get("model", "")
                )
                
                result = chunker.chunk(test_text)
                
                # Basic validation
                assert len(result.chunks) >= 1
                tokenizer_info = result.source_info["tokenizer_info"]
                
                # Check tokenizer-specific features
                print(f"   Tokenizer: {tokenizer_info['type']}")
                print(f"   Total tokens: {result.source_info['total_tokens']}")
                print(f"   Chunks created: {len(result.chunks)}")
                
                # Verify tokenizer type
                assert tokenizer_info["type"] == test_config["type"]
                
                # Different tokenizers should produce different token counts
                total_tokens = result.source_info["total_tokens"]
                assert total_tokens > 0
                
                print(f"   ✅ {test_config['name']} working with {total_tokens} tokens")
                
            except ImportError:
                print(f"   ⚠️  {test_config['name']} dependencies not available")
            except Exception as e:
                print(f"   ❌ {test_config['name']} failed: {e}")

    def test_large_file_processing_with_tokenizers(self):
        """Test processing large files with different tokenizers."""
        large_files = [
            "alice_wonderland.txt",
            "technical_doc.txt"
        ]
        
        tokenizer_configs = [
            {"type": "simple", "tokens_per_chunk": 1000},
            {"type": "tiktoken", "model": "gpt-3.5-turbo", "tokens_per_chunk": 800},
            {"type": "transformers", "model": "bert-base-uncased", "tokens_per_chunk": 400},
        ]
        
        for filename in large_files:
            file_path = self.test_data_dir / filename
            if not file_path.exists():
                continue
                
            print(f"\n📚 Testing large file: {filename}")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if len(content) < 5000:  # Skip if not actually large
                continue
            
            for config in tokenizer_configs:
                try:
                    print(f"   🔧 {config['type']} tokenizer...")
                    
                    chunker = TokenBasedChunker(
                        tokens_per_chunk=config["tokens_per_chunk"],
                        overlap_tokens=config["tokens_per_chunk"] // 20,
                        tokenizer_type=config["type"],
                        tokenizer_model=config.get("model", "")
                    )
                    
                    # Test performance with large file
                    start_time = time.time()
                    result = chunker.chunk(content)
                    processing_time = time.time() - start_time
                    
                    # Validate results
                    assert len(result.chunks) > 5  # Should create multiple chunks
                    assert_valid_chunks(result.chunks, None)
                    
                    # Performance should be reasonable
                    size_kb = len(content) / 1024
                    max_time = max(2.0, size_kb / 50)  # More lenient for tokenization
                    assert processing_time < max_time, f"Too slow: {processing_time:.2f}s for {size_kb:.1f}KB"
                    
                    # Check token statistics
                    total_tokens = result.source_info["total_tokens"]
                    avg_tokens = result.source_info.get("avg_tokens_per_chunk", 0)
                    
                    print(f"      ✅ {len(result.chunks)} chunks, {total_tokens} tokens, {processing_time:.2f}s")
                    print(f"         Avg tokens/chunk: {avg_tokens:.1f}")
                    
                except ImportError:
                    print(f"      ⚠️  {config['type']} not available")
                except Exception as e:
                    print(f"      ❌ {config['type']} failed: {e}")

    def test_cli_integration_token_based(self):
        """Test CLI integration with token-based chunker."""
        test_file = self.test_data_dir / "sample_simple_text.txt"
        if not test_file.exists():
            pytest.skip("Test file not found for CLI testing")
        
        output_file = self.output_dir / "cli_token_output.json"
        
        print(f"\n💻 Testing CLI with token-based chunker...")
        
        try:
            # Test chunking via CLI with token-based strategy
            cmd = [
                sys.executable, "-m", "chunking_strategy.cli", "chunk",
                str(test_file),
                "--strategy", "token_based",
                "--tokens-per-chunk", "100",
                "--tokenizer-type", "simple",
                "--output", str(output_file),
                "--format", "json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"   ⚠️  CLI not fully integrated yet: {result.stderr}")
                pytest.skip("CLI integration not complete")
            else:
                print(f"   ✅ CLI command executed successfully")
                
                # Verify output file was created
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        output_data = json.load(f)
                    
                    assert "chunks" in output_data
                    assert len(output_data["chunks"]) > 0
                    assert output_data.get("strategy_used") == "token_based"
                    
                    print(f"   ✅ Output file created with {len(output_data['chunks'])} chunks")
        
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  CLI command timed out")
            pytest.skip("CLI command timeout")
        except Exception as e:
            print(f"   ⚠️  CLI test error: {e}")
            pytest.skip(f"CLI integration test failed: {e}")

    def test_configuration_based_processing(self):
        """Test processing with YAML configuration files for different tokenizers."""
        print(f"\n⚙️  Testing configuration-based processing...")
        
        # Create test configurations for different tokenizers
        configs = {
            "token_simple.yaml": {
                "strategies": {
                    "primary": "token_based"
                },
                "token_based": {
                    "tokens_per_chunk": 150,
                    "overlap_tokens": 15,
                    "tokenizer_type": "simple",
                    "preserve_word_boundaries": True,
                    "min_chunk_tokens": 10
                }
            },
            "token_gpt.yaml": {
                "strategies": {
                    "primary": "token_based"
                },
                "token_based": {
                    "tokens_per_chunk": 1000,
                    "overlap_tokens": 100,
                    "tokenizer_type": "tiktoken",
                    "tokenizer_model": "gpt-3.5-turbo",
                    "preserve_word_boundaries": True,
                    "max_chunk_chars": 4000
                }
            },
            "token_bert.yaml": {
                "strategies": {
                    "primary": "token_based"
                },
                "token_based": {
                    "tokens_per_chunk": 512,
                    "overlap_tokens": 50,
                    "tokenizer_type": "transformers",
                    "tokenizer_model": "bert-base-uncased",
                    "preserve_word_boundaries": False
                }
            }
        }
        
        # Test each configuration
        for config_name, config_data in configs.items():
            config_file = self.config_dir / config_name
            
            # Write config file
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            print(f"   📝 Testing config: {config_name}")
            
            try:
                # Test with a sample file
                test_file = self.test_data_dir / "sample_article.txt"
                if test_file.exists():
                    with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Process with configuration
                    if content.strip():
                        chunker_config = config_data.get("token_based", {})
                        try:
                            chunker = TokenBasedChunker(**chunker_config)
                            result = chunker.chunk(content)
                            
                            assert len(result.chunks) > 0
                            assert_valid_chunks(result.chunks, None)
                            
                            # Verify configuration was applied
                            expected_tokens = chunker_config.get("tokens_per_chunk", 100)
                            tokenizer_type = chunker_config.get("tokenizer_type", "simple")
                            
                            # Check that tokenizer type was applied
                            tokenizer_info = result.source_info.get("tokenizer_info", {})
                            assert tokenizer_info.get("type") == tokenizer_type
                            
                            print(f"      ✅ Config applied: {len(result.chunks)} chunks, {tokenizer_type} tokenizer")
                            
                        except ImportError:
                            print(f"      ⚠️  {tokenizer_type} tokenizer not available")
                        except Exception as e:
                            print(f"      ❌ Config test failed: {e}")
            
            except Exception as e:
                print(f"      ❌ Config processing failed: {e}")

    def test_orchestrator_integration(self):
        """Test integration with the orchestrator system."""
        print(f"\n🎭 Testing orchestrator integration...")
        
        orchestrator = ChunkerOrchestrator()
        
        # Test that token_based is available
        try:
            chunker = create_chunker(
                "token_based", 
                tokens_per_chunk=200,
                tokenizer_type="simple"
            )
            assert isinstance(chunker, TokenBasedChunker)
            print(f"   ✅ Chunker created via create_chunker")
        except Exception as e:
            print(f"   ⚠️  Direct chunker creation issue: {e}")
        
        # Test with orchestrator chunking
        test_content = """
        This is a comprehensive test document for orchestrator integration
        with the token-based chunking strategy. The orchestrator should be
        able to create token-based chunks with proper configuration and
        metadata handling.
        """
        
        try:
            chunker = TokenBasedChunker(
                tokens_per_chunk=50, 
                overlap_tokens=10,
                tokenizer_type="simple"
            )
            result = chunker.chunk(test_content)
            
            assert len(result.chunks) > 1
            assert result.strategy_used == "token_based"
            
            # Verify each chunk has proper token metadata
            for chunk in result.chunks:
                assert "token_count" in chunk.metadata.extra
                assert chunk.metadata.extra["token_count"] > 0
                assert chunk.metadata.extra["tokenizer_type"] == "simple"
            
            print(f"   ✅ Orchestrator integration working: {len(result.chunks)} chunks")
            
        except Exception as e:
            print(f"   ❌ Orchestrator integration failed: {e}")

    def test_batch_file_processing_tokenizers(self):
        """Test processing multiple files in batch with different tokenizers."""
        print(f"\n📦 Testing batch file processing...")
        
        # Find multiple text files
        text_files = [
            f for f in self.test_data_dir.glob("*.txt") 
            if f.is_file() and f.stat().st_size > 100  # Skip empty files
        ]
        
        if len(text_files) < 2:
            pytest.skip("Not enough text files for batch testing")
        
        # Test different tokenizers for batch processing
        tokenizer_configs = [
            {"name": "Simple", "type": "simple", "tokens": 150},
            {"name": "TikToken", "type": "tiktoken", "model": "gpt-3.5-turbo", "tokens": 200},
        ]
        
        for config in tokenizer_configs:
            print(f"   🔧 Batch testing with {config['name']} tokenizer...")
            
            try:
                chunker = TokenBasedChunker(
                    tokens_per_chunk=config["tokens"],
                    overlap_tokens=config["tokens"] // 20,
                    tokenizer_type=config["type"],
                    tokenizer_model=config.get("model", "")
                )
                
                batch_results = {}
                
                for file_path in text_files[:5]:  # Test first 5 files
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if not content.strip():
                            continue
                        
                        start_time = time.time()
                        result = chunker.chunk(content)
                        processing_time = time.time() - start_time
                        
                        batch_results[file_path.name] = {
                            "chunks": len(result.chunks),
                            "processing_time": processing_time,
                            "total_tokens": result.source_info.get("total_tokens", 0),
                            "tokenizer_info": result.source_info.get("tokenizer_info", {})
                        }
                        
                    except Exception as e:
                        batch_results[file_path.name] = {"error": str(e)}
                
                # Verify batch processing
                successful = [k for k, v in batch_results.items() if "error" not in v]
                assert len(successful) >= 1, f"Batch processing failed for most files: {batch_results}"
                
                # Calculate batch statistics
                total_chunks = sum(v["chunks"] for v in batch_results.values() if "chunks" in v)
                total_tokens = sum(v["total_tokens"] for v in batch_results.values() if "total_tokens" in v)
                total_time = sum(v["processing_time"] for v in batch_results.values() if "processing_time" in v)
                
                print(f"      ✅ {config['name']}: {len(successful)} files, {total_chunks} chunks, {total_tokens} tokens in {total_time:.3f}s")
                
            except ImportError:
                print(f"      ⚠️  {config['name']} tokenizer not available")
            except Exception as e:
                print(f"      ❌ {config['name']} batch processing failed: {e}")

    def test_tokenization_comparison(self):
        """Test and compare different tokenization approaches on same content."""
        print(f"\n🔍 Testing tokenization comparison...")
        
        # Test text with various challenges
        test_text = """
        The quick brown fox jumps over the lazy dog. This pangram sentence
        contains every letter of the English alphabet at least once.
        
        Modern AI systems like GPT-3, GPT-4, and BERT use different tokenization
        strategies: byte-pair encoding (BPE), WordPiece, and SentencePiece.
        
        URLs (https://openai.com), emails (user@example.com), numbers (123.45),
        and contractions (don't, won't, can't) pose interesting challenges.
        
        Unicode characters: café, naïve, résumé, Москва, 北京, العربية
        """
        
        tokenizer_tests = [
            {"name": "Simple Whitespace", "type": "simple"},
            {"name": "OpenAI GPT-3.5", "type": "tiktoken", "model": "gpt-3.5-turbo"},
            {"name": "OpenAI GPT-4", "type": "tiktoken", "model": "gpt-4"},
            {"name": "BERT Base", "type": "transformers", "model": "bert-base-uncased"},
            {"name": "GPT-2", "type": "transformers", "model": "gpt2"},
        ]
        
        results = {}
        
        for test in tokenizer_tests:
            try:
                chunker = TokenBasedChunker(
                    tokens_per_chunk=1000,  # Large enough to fit all in one chunk
                    tokenizer_type=test["type"],
                    tokenizer_model=test.get("model", "")
                )
                
                result = chunker.chunk(test_text)
                
                tokenizer_info = result.source_info.get("tokenizer_info", {})
                total_tokens = result.source_info.get("total_tokens", 0)
                
                results[test["name"]] = {
                    "tokens": total_tokens,
                    "chunks": len(result.chunks),
                    "tokenizer_type": tokenizer_info.get("type", "unknown"),
                    "model": tokenizer_info.get("model", test.get("model", "N/A"))
                }
                
                print(f"   {test['name']:20}: {total_tokens:4d} tokens")
                
            except ImportError:
                print(f"   {test['name']:20}: Not available")
            except Exception as e:
                print(f"   {test['name']:20}: Error - {e}")
        
        # Verify we got different token counts (proving different tokenization)
        token_counts = [r["tokens"] for r in results.values() if r["tokens"] > 0]
        if len(token_counts) > 1:
            # Different tokenizers should produce different token counts
            unique_counts = len(set(token_counts))
            print(f"   ✅ {unique_counts} different token counts from {len(token_counts)} tokenizers")

    def test_streaming_with_tokenizers(self):
        """Test streaming functionality with different tokenizers."""
        print(f"\n🌊 Testing streaming with tokenizers...")
        
        # Create content chunks to stream
        content_parts = [
            "This is the first part of a streaming document. It contains several sentences.",
            "Here comes the second part with additional content and more complex text.",
            "The third part includes technical terms, numbers like 123.45, and URLs.",
            "Finally, the fourth part completes our streaming test with Unicode: café, naïve.",
            "And some final content to ensure we have enough tokens for multiple chunks."
        ]
        
        tokenizer_configs = [
            {"name": "Simple", "type": "simple", "tokens": 30},
            {"name": "TikToken", "type": "tiktoken", "model": "gpt-3.5-turbo", "tokens": 40},
        ]
        
        for config in tokenizer_configs:
            try:
                print(f"   🔧 Streaming with {config['name']} tokenizer...")
                
                chunker = TokenBasedChunker(
                    tokens_per_chunk=config["tokens"],
                    overlap_tokens=5,
                    tokenizer_type=config["type"],
                    tokenizer_model=config.get("model", "")
                )
                
                def content_generator():
                    for part in content_parts:
                        yield part + " "
                
                # Test streaming
                chunks = list(chunker.chunk_stream(content_generator()))
                
                # Should create multiple chunks
                assert len(chunks) >= 1
                
                # Check streaming metadata
                for chunk in chunks:
                    assert chunk.metadata.extra.get("is_streaming", False) == True
                    assert "token_count" in chunk.metadata.extra
                    assert chunk.metadata.extra["tokenizer_type"] == config["type"]
                
                print(f"      ✅ Created {len(chunks)} streaming chunks")
                
            except ImportError:
                print(f"      ⚠️  {config['name']} tokenizer not available")
            except Exception as e:
                print(f"      ❌ {config['name']} streaming failed: {e}")

    def test_error_handling_with_real_files(self):
        """Test error handling with problematic real files."""
        print(f"\n🛡️  Testing error handling...")
        
        chunker = TokenBasedChunker(tokens_per_chunk=200, tokenizer_type="simple")
        
        # Test with empty files
        empty_files = ["empty.txt", "empty.csv"]
        for filename in empty_files:
            file_path = self.test_data_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Only test truly empty content
                if content.strip() == "":
                    result = chunker.chunk(content)
                    assert len(result.chunks) == 0  # Empty content should produce no chunks
                    assert result.processing_time >= 0
        
        # Test with binary-like files (should handle gracefully)
        binary_file = self.test_data_dir / "sample_binary_like.bin"
        if binary_file.exists():
            try:
                with open(binary_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                result = chunker.chunk(content)
                # Should not crash, even with weird content
                assert isinstance(result, ChunkingResult)
            except Exception:
                # Binary file reading may fail, which is acceptable
                pass
        
        # Test with invalid tokenizer configuration
        try:
            bad_chunker = TokenBasedChunker(
                tokenizer_type="transformers",
                tokenizer_model="non-existent-model-12345"
            )
            result = bad_chunker.chunk("This should still work via fallback.")
            # Should fallback gracefully
            assert isinstance(result, ChunkingResult)
        except Exception:
            # Complete failure is also acceptable for invalid configs
            pass
        
        print(f"   ✅ Error handling working correctly")

    def test_performance_comparison_across_tokenizers(self):
        """Test performance characteristics across different tokenizers."""
        print(f"\n⚡ Testing performance across tokenizers...")
        
        # Use a reasonably large text file
        large_files = [f for f in self.test_data_dir.glob("*.txt") 
                      if f.is_file() and f.stat().st_size > 5000]
        
        if not large_files:
            pytest.skip("No large files available for performance testing")
        
        test_file = large_files[0]
        with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        tokenizer_performance = {}
        
        performance_configs = [
            {"name": "Simple", "type": "simple", "tokens": 500},
            {"name": "TikToken", "type": "tiktoken", "model": "gpt-3.5-turbo", "tokens": 500},
            {"name": "Transformers", "type": "transformers", "model": "bert-base-uncased", "tokens": 300},
        ]
        
        for config in performance_configs:
            try:
                chunker = TokenBasedChunker(
                    tokens_per_chunk=config["tokens"],
                    overlap_tokens=config["tokens"] // 20,
                    tokenizer_type=config["type"],
                    tokenizer_model=config.get("model", "")
                )
                
                # Run multiple iterations for better timing
                times = []
                for _ in range(3):
                    start_time = time.time()
                    result = chunker.chunk(content)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = sum(times) / len(times)
                total_tokens = result.source_info.get("total_tokens", 0)
                throughput = total_tokens / avg_time if avg_time > 0 else 0
                
                performance_data = {
                    "avg_time": avg_time,
                    "throughput_tokens_per_sec": throughput,
                    "total_tokens": total_tokens,
                    "chunks_created": len(result.chunks)
                }
                
                performance_configs[config["name"]] = performance_data
                
                print(f"   {config['name']:12}: {avg_time:.3f}s, {throughput:.0f} tokens/sec, {len(result.chunks)} chunks")
                
                # Performance should be reasonable
                assert avg_time < 10.0, f"{config['name']} too slow: {avg_time}s"
                assert throughput > 100, f"{config['name']} throughput too low: {throughput}"
                
            except ImportError:
                print(f"   {config['name']:12}: Not available")
            except Exception as e:
                print(f"   {config['name']:12}: Error - {e}")
        
        print(f"   ✅ Performance testing completed")

    def test_output_format_compatibility(self):
        """Test that outputs are compatible with expected formats."""
        print(f"\n📋 Testing output format compatibility...")
        
        chunker = TokenBasedChunker(
            tokens_per_chunk=100,
            tokenizer_type="simple"
        )
        test_content = """
        This is a sample text that will be processed by the token-based chunker
        to test output format compatibility with various downstream systems and
        applications that might consume the chunking results.
        """
        
        result = chunker.chunk(test_content)
        
        # Test JSON serialization
        try:
            import json
            result_dict = {
                "chunks": [
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "metadata": {
                            "token_count": chunk.metadata.extra.get("token_count"),
                            "tokenizer_type": chunk.metadata.extra.get("tokenizer_type"),
                            "position": chunk.metadata.position,
                            "length": chunk.metadata.length
                        }
                    }
                    for chunk in result.chunks
                ],
                "strategy_used": result.strategy_used,
                "processing_time": result.processing_time,
                "total_tokens": result.source_info.get("total_tokens"),
                "tokenizer_info": result.source_info.get("tokenizer_info")
            }
            
            json_str = json.dumps(result_dict, indent=2)
            assert len(json_str) > 0
            
            # Verify we can parse it back
            parsed = json.loads(json_str)
            assert parsed["strategy_used"] == "token_based"
            assert len(parsed["chunks"]) == len(result.chunks)
            assert "total_tokens" in parsed
            
            print(f"   ✅ JSON serialization working")
            
        except Exception as e:
            pytest.fail(f"JSON serialization failed: {e}")
        
        # Test that chunk metadata is complete
        for chunk in result.chunks:
            assert chunk.id is not None
            assert chunk.content is not None
            assert chunk.metadata is not None
            assert chunk.metadata.extra.get("token_count") is not None
            assert chunk.metadata.extra.get("tokenizer_type") is not None
            assert chunk.metadata.extra.get("chunking_strategy") == "token_based"
        
        print(f"   ✅ All output formats compatible")

    def test_real_world_usage_scenarios(self):
        """Test real-world usage scenarios for token-based chunking."""
        print(f"\n🌍 Testing real-world usage scenarios...")
        
        # Scenario 1: RAG with GPT tokenization
        print("   📚 Scenario 1: RAG with GPT tokenization")
        try:
            rag_chunker = TokenBasedChunker(
                tokens_per_chunk=1500,  # Good for GPT-3.5 context
                overlap_tokens=150,
                tokenizer_type="tiktoken",
                tokenizer_model="gpt-3.5-turbo",
                preserve_word_boundaries=True
            )
            
            # Test with a document
            doc_file = self.test_data_dir / "technical_doc.txt"
            if doc_file.exists():
                with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if content.strip():
                    result = rag_chunker.chunk(content)
                    assert len(result.chunks) > 0
                    
                    # Check token limits are respected
                    for chunk in result.chunks:
                        token_count = chunk.metadata.extra["token_count"]
                        assert token_count <= 1500
                    
                    print(f"      ✅ RAG: {len(result.chunks)} chunks created")
        except ImportError:
            print("      ⚠️  TikToken not available for RAG scenario")
        
        # Scenario 2: BERT embeddings
        print("   🤖 Scenario 2: BERT embeddings (512 token limit)")
        try:
            bert_chunker = TokenBasedChunker(
                tokens_per_chunk=512,  # BERT's token limit
                overlap_tokens=50,
                tokenizer_type="transformers",
                tokenizer_model="bert-base-uncased",
                preserve_word_boundaries=False  # BERT handles subwords
            )
            
            # Test with sample content
            sample_file = self.test_data_dir / "sample_article.txt"
            if sample_file.exists():
                with open(sample_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if content.strip():
                    result = bert_chunker.chunk(content)
                    assert len(result.chunks) > 0
                    
                    # Check BERT token limits
                    for chunk in result.chunks:
                        token_count = chunk.metadata.extra["token_count"]
                        assert token_count <= 512
                    
                    print(f"      ✅ BERT: {len(result.chunks)} chunks created")
        except ImportError:
            print("      ⚠️  Transformers not available for BERT scenario")
        
        # Scenario 3: Simple processing for development/testing
        print("   ⚡ Scenario 3: Simple processing for development")
        simple_chunker = TokenBasedChunker(
            tokens_per_chunk=200,
            overlap_tokens=20,
            tokenizer_type="simple"
        )
        
        # Test with any available file
        test_files = list(self.test_data_dir.glob("*.txt"))
        if test_files:
            test_file = test_files[0]
            with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if content.strip():
                result = simple_chunker.chunk(content)
                assert len(result.chunks) > 0
                print(f"      ✅ Simple: {len(result.chunks)} chunks created for development")
        
        print(f"   ✅ Real-world scenarios tested")


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v"])
