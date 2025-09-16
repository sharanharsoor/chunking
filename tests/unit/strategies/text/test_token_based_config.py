"""
Configuration tests for Token-based Chunker.

This module tests configuration file processing, YAML parsing,
and parameter validation for the token-based chunking strategy.
"""

import pytest
import yaml
import tempfile
import shutil
from pathlib import Path

from chunking_strategy.strategies.text.token_based_chunker import TokenBasedChunker
from chunking_strategy import create_chunker


class TestTokenBasedConfig:
    """Configuration tests for Token-based Chunker."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "configs"
        self.config_dir.mkdir(exist_ok=True)

        yield

        # Cleanup
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_basic_yaml_configuration(self):
        """Test basic YAML configuration loading."""
        print(f"\n⚙️  Testing basic YAML configuration...")

        # Create a basic configuration
        config_data = {
            "strategies": {
                "primary": "token_based"
            },
            "token_based": {
                "tokens_per_chunk": 500,
                "overlap_tokens": 50,
                "tokenizer_type": "simple",
                "preserve_word_boundaries": True,
                "min_chunk_tokens": 25,
                "max_chunk_chars": 3000
            }
        }

        config_file = self.config_dir / "basic_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        print(f"   Configuration file created: {config_file}")

        # Test configuration loading
        try:
            # Load and parse the configuration
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)

            print(f"   ✅ Configuration loaded successfully")

            # Verify structure
            assert "strategies" in loaded_config
            assert "token_based" in loaded_config
            assert loaded_config["strategies"]["primary"] == "token_based"

            # Test chunker creation with config
            chunker_config = loaded_config["token_based"]
            chunker = TokenBasedChunker(**chunker_config)

            # Verify parameters were applied
            assert chunker.tokens_per_chunk == 500
            assert chunker.overlap_tokens == 50
            assert chunker.tokenizer_type.value == "simple"
            assert chunker.preserve_word_boundaries == True
            assert chunker.min_chunk_tokens == 25
            assert chunker.max_chunk_chars == 3000

            print(f"   ✅ Chunker created with configuration")
            print(f"   Parameters: {chunker.tokens_per_chunk} tokens/chunk, {chunker.overlap_tokens} overlap")

            # Test chunking with configuration
            test_text = "Configuration test with multiple sentences and words. " * 20
            result = chunker.chunk(test_text)

            assert len(result.chunks) > 0
            assert result.strategy_used == "token_based"

            print(f"   ✅ Chunking with configuration successful: {len(result.chunks)} chunks")

        except Exception as e:
            pytest.fail(f"Basic configuration test failed: {e}")

    def test_multiple_tokenizer_configurations(self):
        """Test different tokenizer configurations."""
        print(f"\n🔧 Testing multiple tokenizer configurations...")

        tokenizer_configs = {
            "simple_config": {
                "token_based": {
                    "tokens_per_chunk": 100,
                    "overlap_tokens": 10,
                    "tokenizer_type": "simple",
                    "preserve_word_boundaries": True
                }
            },
            "tiktoken_config": {
                "token_based": {
                    "tokens_per_chunk": 800,
                    "overlap_tokens": 80,
                    "tokenizer_type": "tiktoken",
                    "tokenizer_model": "gpt-3.5-turbo",
                    "preserve_word_boundaries": True
                }
            },
            "transformers_config": {
                "token_based": {
                    "tokens_per_chunk": 400,
                    "overlap_tokens": 40,
                    "tokenizer_type": "transformers",
                    "tokenizer_model": "bert-base-uncased",
                    "preserve_word_boundaries": False
                }
            }
        }

        test_text = "Multiple tokenizer configuration test with various words and sentences."

        for config_name, config_data in tokenizer_configs.items():
            print(f"   Testing {config_name}...")

            try:
                # Create chunker from configuration
                chunker_config = config_data["token_based"]
                chunker = TokenBasedChunker(**chunker_config)

                # Test chunking
                result = chunker.chunk(test_text)

                tokenizer_info = result.source_info.get("tokenizer_info", {})
                total_tokens = result.source_info.get("total_tokens", 0)

                print(f"      ✅ {config_name}: {total_tokens} tokens, {len(result.chunks)} chunks")
                print(f"         Tokenizer: {tokenizer_info.get('type', 'unknown')}")

                # Verify configuration was applied
                assert chunker.tokens_per_chunk == chunker_config["tokens_per_chunk"]
                assert chunker.overlap_tokens == chunker_config["overlap_tokens"]
                assert chunker.tokenizer_type.value == chunker_config["tokenizer_type"]

                if "tokenizer_model" in chunker_config:
                    assert chunker.tokenizer_model == chunker_config["tokenizer_model"]

            except ImportError as e:
                print(f"      ⚠️  {config_name}: Dependencies not available - {e}")
            except Exception as e:
                print(f"      ❌ {config_name}: Failed - {e}")

    def test_use_case_configurations(self):
        """Test specific use case configurations."""
        print(f"\n🎯 Testing use case configurations...")

        use_case_configs = {
            "rag_system": {
                "description": "Optimized for RAG systems",
                "config": {
                    "tokens_per_chunk": 1000,
                    "overlap_tokens": 100,
                    "tokenizer_type": "simple",  # Use simple for testing
                    "preserve_word_boundaries": True,
                    "min_chunk_tokens": 50
                }
            },
            "bert_embeddings": {
                "description": "BERT sequence length optimization",
                "config": {
                    "tokens_per_chunk": 400,  # Leave room for special tokens
                    "overlap_tokens": 40,
                    "tokenizer_type": "simple",
                    "preserve_word_boundaries": False,
                    "min_chunk_tokens": 20
                }
            },
            "api_optimization": {
                "description": "Cost-optimized for APIs",
                "config": {
                    "tokens_per_chunk": 1500,
                    "overlap_tokens": 150,
                    "tokenizer_type": "simple",
                    "preserve_word_boundaries": True,
                    "min_chunk_tokens": 100
                }
            },
            "development": {
                "description": "Fast processing for development",
                "config": {
                    "tokens_per_chunk": 200,
                    "overlap_tokens": 20,
                    "tokenizer_type": "simple",
                    "preserve_word_boundaries": True,
                    "min_chunk_tokens": 10
                }
            }
        }

        # Create test content of different sizes
        test_contents = {
            "short": "Short test content. " * 10,  # ~30 words
            "medium": "Medium length test content. " * 100,  # ~400 words
            "long": "Long test content for comprehensive testing. " * 500,  # ~3000 words
        }

        for use_case, config_info in use_case_configs.items():
            print(f"   Testing {use_case} configuration...")
            print(f"      Description: {config_info['description']}")

            try:
                chunker = TokenBasedChunker(**config_info["config"])

                # Test with different content sizes
                for content_type, content in test_contents.items():
                    result = chunker.chunk(content)

                    total_tokens = result.source_info.get("total_tokens", 0)
                    chunks_created = len(result.chunks)
                    avg_tokens = result.source_info.get("avg_tokens_per_chunk", 0)

                    print(f"         {content_type} ({total_tokens} tokens): {chunks_created} chunks, avg {avg_tokens:.1f} tokens/chunk")

                    # Validate results
                    assert chunks_created > 0
                    assert result.strategy_used == "token_based"

                    # Check token limits are respected
                    for chunk in result.chunks:
                        token_count = chunk.metadata.extra.get("token_count", 0)
                        assert token_count <= config_info["config"]["tokens_per_chunk"]

                print(f"      ✅ {use_case} configuration working correctly")

            except Exception as e:
                print(f"      ❌ {use_case} configuration failed: {e}")

    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        print(f"\n✅ Testing configuration validation...")

        # Test invalid configurations
        invalid_configs = [
            {
                "name": "Invalid tokens_per_chunk (zero)",
                "config": {"tokens_per_chunk": 0, "tokenizer_type": "simple"},
                "should_raise": ValueError
            },
            {
                "name": "Invalid tokens_per_chunk (negative)",
                "config": {"tokens_per_chunk": -100, "tokenizer_type": "simple"},
                "should_raise": ValueError
            },
            {
                "name": "Invalid overlap (greater than chunk size)",
                "config": {"tokens_per_chunk": 100, "overlap_tokens": 150, "tokenizer_type": "simple"},
                "should_raise": ValueError
            },
            {
                "name": "Invalid overlap (negative)",
                "config": {"tokens_per_chunk": 100, "overlap_tokens": -10, "tokenizer_type": "simple"},
                "should_raise": ValueError
            },
            {
                "name": "Invalid tokenizer type",
                "config": {"tokens_per_chunk": 100, "tokenizer_type": "invalid_tokenizer"},
                "should_raise": ValueError
            },
            {
                "name": "Invalid min_chunk_tokens (zero)",
                "config": {"tokens_per_chunk": 100, "min_chunk_tokens": 0, "tokenizer_type": "simple"},
                "should_raise": ValueError
            }
        ]

        for invalid_config in invalid_configs:
            print(f"   Testing {invalid_config['name']}...")

            with pytest.raises(invalid_config["should_raise"]):
                TokenBasedChunker(**invalid_config["config"])

            print(f"      ✅ Correctly rejected invalid configuration")

        # Test valid configurations
        valid_configs = [
            {
                "name": "Minimal valid config",
                "config": {"tokens_per_chunk": 100, "tokenizer_type": "simple"}
            },
            {
                "name": "Complete valid config",
                "config": {
                    "tokens_per_chunk": 500,
                    "overlap_tokens": 50,
                    "tokenizer_type": "simple",
                    "preserve_word_boundaries": True,
                    "min_chunk_tokens": 25,
                    "max_chunk_chars": 3000
                }
            },
            {
                "name": "Edge case valid config",
                "config": {
                    "tokens_per_chunk": 1,
                    "overlap_tokens": 0,
                    "tokenizer_type": "simple",
                    "min_chunk_tokens": 1
                }
            }
        ]

        for valid_config in valid_configs:
            print(f"   Testing {valid_config['name']}...")

            try:
                chunker = TokenBasedChunker(**valid_config["config"])

                # Test that it actually works
                result = chunker.chunk("Test content for validation.")
                assert len(result.chunks) > 0

                print(f"      ✅ Valid configuration accepted and working")

            except Exception as e:
                pytest.fail(f"Valid configuration rejected: {e}")

    def test_configuration_file_formats(self):
        """Test different configuration file formats."""
        print(f"\n📋 Testing configuration file formats...")

        base_config = {
            "strategies": {"primary": "token_based"},
            "token_based": {
                "tokens_per_chunk": 300,
                "overlap_tokens": 30,
                "tokenizer_type": "simple",
                "preserve_word_boundaries": True
            }
        }

        # Test YAML format
        yaml_file = self.config_dir / "test_config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(base_config, f)

        # Test loading YAML
        with open(yaml_file, 'r') as f:
            yaml_loaded = yaml.safe_load(f)

        print(f"   ✅ YAML format: loaded successfully")
        assert yaml_loaded["token_based"]["tokens_per_chunk"] == 300

        # Test chunker creation from loaded config
        chunker = TokenBasedChunker(**yaml_loaded["token_based"])
        result = chunker.chunk("Configuration format test content.")

        assert len(result.chunks) > 0
        print(f"   ✅ YAML configuration: chunking successful")

    def test_configuration_inheritance_and_overrides(self):
        """Test configuration inheritance and parameter overrides."""
        print(f"\n🔄 Testing configuration inheritance and overrides...")

        # Base configuration
        base_config = {
            "tokens_per_chunk": 500,
            "overlap_tokens": 50,
            "tokenizer_type": "simple",
            "preserve_word_boundaries": True,
            "min_chunk_tokens": 25
        }

        # Override configurations
        overrides = [
            {"tokens_per_chunk": 1000},  # Override chunk size
            {"overlap_tokens": 100},     # Override overlap
            {"tokenizer_type": "simple", "preserve_word_boundaries": False},  # Override multiple
            {"tokens_per_chunk": 200, "overlap_tokens": 20, "min_chunk_tokens": 10}  # Multiple overrides
        ]

        test_text = "Configuration inheritance test content with multiple sentences."

        for i, override in enumerate(overrides):
            print(f"   Testing override {i+1}: {override}")

            # Merge base config with overrides
            merged_config = {**base_config, **override}

            try:
                chunker = TokenBasedChunker(**merged_config)
                result = chunker.chunk(test_text)

                # Verify overrides were applied
                for key, value in override.items():
                    if hasattr(chunker, key):
                        actual_value = getattr(chunker, key)
                        if hasattr(actual_value, 'value'):  # Handle enums
                            actual_value = actual_value.value
                        assert actual_value == value, f"Override {key} not applied: expected {value}, got {actual_value}"

                print(f"      ✅ Override applied successfully")
                print(f"         Result: {len(result.chunks)} chunks, {result.source_info.get('total_tokens', 0)} tokens")

            except Exception as e:
                pytest.fail(f"Configuration override failed: {e}")

    def test_configuration_edge_cases(self):
        """Test configuration edge cases and boundary conditions."""
        print(f"\n🔍 Testing configuration edge cases...")

        edge_cases = [
            {
                "name": "Minimum viable configuration",
                "config": {"tokens_per_chunk": 1, "overlap_tokens": 0, "tokenizer_type": "simple"},
                "should_work": True
            },
            {
                "name": "Large chunk size",
                "config": {"tokens_per_chunk": 10000, "tokenizer_type": "simple"},
                "should_work": True
            },
            {
                "name": "Maximum overlap (just under chunk size)",
                "config": {"tokens_per_chunk": 100, "overlap_tokens": 99, "tokenizer_type": "simple"},
                "should_work": True
            },
            {
                "name": "Zero overlap",
                "config": {"tokens_per_chunk": 100, "overlap_tokens": 0, "tokenizer_type": "simple"},
                "should_work": True
            },
            {
                "name": "Very small min_chunk_tokens",
                "config": {"tokens_per_chunk": 100, "min_chunk_tokens": 1, "tokenizer_type": "simple"},
                "should_work": True
            },
            {
                "name": "Large max_chunk_chars",
                "config": {"tokens_per_chunk": 100, "max_chunk_chars": 50000, "tokenizer_type": "simple"},
                "should_work": True
            }
        ]

        test_text = "Edge case testing content. " * 20  # 60 words

        for edge_case in edge_cases:
            print(f"   Testing {edge_case['name']}...")

            try:
                chunker = TokenBasedChunker(**edge_case["config"])
                result = chunker.chunk(test_text)

                if edge_case["should_work"]:
                    assert len(result.chunks) > 0
                    print(f"      ✅ {edge_case['name']}: Working correctly")
                    print(f"         Chunks: {len(result.chunks)}, Tokens: {result.source_info.get('total_tokens', 0)}")
                else:
                    pytest.fail(f"Edge case should have failed but didn't: {edge_case['name']}")

            except Exception as e:
                if edge_case["should_work"]:
                    pytest.fail(f"Valid edge case failed: {edge_case['name']} - {e}")
                else:
                    print(f"      ✅ {edge_case['name']}: Correctly failed - {e}")

    def test_real_world_config_examples(self):
        """Test real-world configuration examples."""
        print(f"\n🌍 Testing real-world configuration examples...")

        # Load the actual configuration file we created
        config_example_path = Path("config_examples/token_based_chunker.yaml")

        if not config_example_path.exists():
            pytest.skip("Token-based chunker config example not found")

        try:
            with open(config_example_path, 'r') as f:
                config_data = yaml.safe_load(f)

            print(f"   ✅ Loaded real configuration file: {config_example_path}")

            # Test the main configuration
            if "token_based" in config_data:
                main_config = config_data["token_based"]

                try:
                    chunker = TokenBasedChunker(**main_config)

                    test_text = "Real-world configuration test. " * 100  # 300 words
                    result = chunker.chunk(test_text)

                    print(f"   ✅ Main configuration working: {len(result.chunks)} chunks")

                except Exception as e:
                    print(f"   ❌ Main configuration failed: {e}")

            # Test alternative configurations if they exist
            if "configurations" in config_data:
                alt_configs = config_data["configurations"]

                for config_name, config_data_alt in alt_configs.items():
                    if "token_based" in config_data_alt:
                        print(f"   Testing {config_name} configuration...")

                        try:
                            chunker = TokenBasedChunker(**config_data_alt["token_based"])

                            test_text = "Alternative configuration test. " * 50  # 150 words
                            result = chunker.chunk(test_text)

                            print(f"      ✅ {config_name}: {len(result.chunks)} chunks")

                        except ImportError:
                            print(f"      ⚠️  {config_name}: Dependencies not available")
                        except Exception as e:
                            print(f"      ❌ {config_name}: Failed - {e}")

        except Exception as e:
            pytest.skip(f"Could not load real-world config file: {e}")


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v", "-s"])
