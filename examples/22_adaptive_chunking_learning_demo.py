#!/usr/bin/env python3
"""
Adaptive Chunking Learning Demo

This example demonstrates the sophisticated learning and adaptation capabilities
of the Adaptive Chunker, which can:

1. Analyze content characteristics automatically
2. Select optimal strategies based on content type
3. Learn from performance feedback over time
4. Adapt parameters based on historical data
5. Persist learned knowledge across sessions
6. Self-optimize through exploration and exploitation

Key Features Demonstrated:
- Content profiling and analysis
- Strategy selection based on content characteristics
- Performance learning and feedback processing
- Parameter adaptation and optimization
- Historical data persistence and loading
- Multi-strategy benchmarking and comparison
- Real-time adaptation during processing

To run this demo:
    python examples/22_adaptive_chunking_learning_demo.py
"""

import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add parent directory to Python path for local development
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from chunking_strategy import create_chunker

# Configure logging to see adaptation process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_contents() -> Dict[str, str]:
    """Create diverse sample contents to test adaptation."""
    return {
        "structured_text": """
# Document Title

## Introduction
This is a well-structured document with clear sections and subsections.
It follows a hierarchical format with markdown syntax.

## Main Content

### Section 1: Overview
This section provides an overview of the topic.
It contains multiple paragraphs with detailed explanations.

### Section 2: Details
Here we dive deeper into the specifics.
The content is organized in a logical manner.

## Conclusion
This concludes our structured document example.
        """.strip(),

        "repetitive_text": """
        ERROR: Connection failed to server 192.168.1.100
        ERROR: Connection failed to server 192.168.1.101
        ERROR: Connection failed to server 192.168.1.102
        INFO: Retrying connection to server 192.168.1.100
        ERROR: Connection failed to server 192.168.1.100
        INFO: Retrying connection to server 192.168.1.101
        ERROR: Connection failed to server 192.168.1.101
        INFO: Retrying connection to server 192.168.1.102
        ERROR: Connection failed to server 192.168.1.102
        WARNING: Max retries exceeded for all servers
        ERROR: System entering maintenance mode
        INFO: Backup servers activated
        ERROR: Backup server 1 unavailable
        ERROR: Backup server 2 unavailable
        INFO: Emergency protocols initiated
        """.strip(),

        "dense_technical": """
        The Quantum Fourier Transform (QFT) algorithm implements a discrete Fourier
        transform on quantum amplitudes. Given a quantum state |x⟩ = Σ(j=0 to N-1) x_j|j⟩,
        the QFT produces |y⟩ = Σ(k=0 to N-1) y_k|k⟩ where y_k = (1/√N) Σ(j=0 to N-1) x_j ω^(jk)
        and ω = e^(2πi/N). The quantum circuit complexity is O(n²) for n qubits using
        controlled rotations R_k where R_k = |0⟩⟨0| + e^(2πi/2^k)|1⟩⟨1|. Implementation
        requires Hadamard gates H = (1/√2)(|0⟩⟨0| + |0⟩⟨1| + |1⟩⟨0| - |1⟩⟨1|) and
        controlled phase gates. The inverse QFT reverses this transformation efficiently.
        """.strip(),

        "conversational": """
        Hey there! How's it going? I hope you're having a great day today.

        So, I wanted to tell you about this awesome thing that happened yesterday.
        I was walking down the street, just minding my own business, when I saw
        this really cute dog. It was a golden retriever, and it was so friendly!

        The owner was super nice too. We ended up chatting for like 20 minutes
        about dogs, the weather, and life in general. It's amazing how a simple
        encounter can brighten your whole day, you know?

        Anyway, I just wanted to share that little story with you. Sometimes
        it's the small things that make the biggest difference. Hope you have
        some nice encounters today too!
        """.strip(),

        "code_like": """
        def calculate_fibonacci(n):
            if n <= 1:
                return n
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

        class DataProcessor:
            def __init__(self, config):
                self.config = config
                self.data = []

            def process_batch(self, batch_size=100):
                for i in range(0, len(self.data), batch_size):
                    batch = self.data[i:i+batch_size]
                    yield self.transform_batch(batch)

            def transform_batch(self, batch):
                return [self.apply_transforms(item) for item in batch]
        """.strip()
    }

def demonstrate_content_profiling(adaptive_chunker):
    """Demonstrate content analysis and profiling."""
    print("🔍 CONTENT PROFILING DEMONSTRATION")
    print("=" * 60)

    contents = create_sample_contents()

    for content_type, content in contents.items():
        print(f"\n📄 Analyzing: {content_type}")
        print("-" * 40)

        # Get the internal content profile (we'll access it via a test chunk)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # This will trigger content profiling internally
            result = adaptive_chunker.chunk(temp_file)

            # Get the profile from the result metadata
            if 'content_profile' in result.source_info:
                profile = result.source_info['content_profile']
                print(f"   📊 Size: {profile['size_bytes']} bytes")
                print(f"   🎯 Text Ratio: {profile['text_ratio']:.2f}")
                print(f"   📐 Structure Score: {profile['structure_score']:.2f}")
                print(f"   🔄 Repetition Score: {profile['repetition_score']:.2f}")
                print(f"   🧠 Complexity Score: {profile['complexity_score']:.2f}")
                print(f"   📈 Entropy: {profile['estimated_entropy']:.2f}")

            if 'adaptive_strategy' in result.source_info:
                strategy = result.source_info['adaptive_strategy']
                params = result.source_info['optimized_parameters']
                print(f"   🎯 Selected Strategy: {strategy}")
                print(f"   ⚙️  Optimized Parameters: {params}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
        finally:
            import os
            os.unlink(temp_file)

def demonstrate_learning_and_adaptation(adaptive_chunker):
    """Demonstrate learning and adaptation over multiple operations."""
    print("\n🧠 LEARNING & ADAPTATION DEMONSTRATION")
    print("=" * 60)

    contents = create_sample_contents()

    print("📈 Processing multiple contents to trigger learning...")

    # Process each content type multiple times with feedback
    for round_num in range(3):
        print(f"\n🔄 Learning Round {round_num + 1}/3")
        print("-" * 40)

        for content_type, content in contents.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_file = f.name

            try:
                start_time = time.time()
                result = adaptive_chunker.chunk(temp_file)
                processing_time = time.time() - start_time

                # Extract performance data
                chunks = list(result.chunks)
                avg_chunk_size = sum(len(chunk.content) for chunk in chunks) / len(chunks)

                print(f"   📄 {content_type}: {len(chunks)} chunks, "
                      f"avg: {avg_chunk_size:.0f} chars, "
                      f"time: {processing_time:.3f}s")

                # Simulate feedback based on performance
                if processing_time < 0.01 and len(chunks) > 2:
                    feedback_score = 0.9  # Good performance
                    print(f"      ✅ Providing positive feedback (0.9)")
                elif processing_time > 0.05 or len(chunks) < 2:
                    feedback_score = 0.3  # Poor performance
                    print(f"      ❌ Providing negative feedback (0.3)")
                else:
                    feedback_score = 0.7  # Average performance
                    print(f"      ⚖️  Providing neutral feedback (0.7)")

                # Provide feedback to trigger adaptation
                adaptive_chunker.adapt_parameters(feedback_score, "performance")

            except Exception as e:
                print(f"   ❌ Error processing {content_type}: {e}")
            finally:
                import os
                os.unlink(temp_file)

        # Show adaptation status after each round
        adaptation_info = adaptive_chunker.get_adaptation_info()
        print(f"\n   📊 After Round {round_num + 1}:")
        print(f"      Operations: {adaptation_info['operation_count']}")
        print(f"      Adaptations: {adaptation_info['total_adaptations']}")
        print(f"      Learning Rate: {adaptation_info['learning_rate']:.3f}")
        print(f"      History Size: {adaptation_info['history_size']}")

def demonstrate_strategy_comparison():
    """Compare adaptive chunker with fixed strategies."""
    print("\n⚔️ ADAPTIVE vs FIXED STRATEGIES COMPARISON")
    print("=" * 60)

    # Test content
    test_content = create_sample_contents()["structured_text"]

    strategies_to_test = [
        ("adaptive", {}),
        ("sentence_based", {"max_sentences": 3}),
        ("paragraph_based", {"max_paragraphs": 2}),
        ("fixed_size", {"chunk_size": 500}),
    ]

    results = {}

    for strategy_name, params in strategies_to_test:
        print(f"\n🧪 Testing: {strategy_name}")
        print("-" * 30)

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_file = f.name

            start_time = time.time()
            chunker = create_chunker(strategy_name, **params)
            result = chunker.chunk(temp_file)
            processing_time = time.time() - start_time

            chunks = list(result.chunks)
            chunk_sizes = [len(chunk.content) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            size_std = (sum((s - avg_size) ** 2 for s in chunk_sizes) / len(chunk_sizes)) ** 0.5 if len(chunk_sizes) > 1 else 0

            results[strategy_name] = {
                'chunks': len(chunks),
                'avg_size': avg_size,
                'size_consistency': 1 - (size_std / avg_size) if avg_size > 0 else 0,
                'processing_time': processing_time,
                'strategy_used': getattr(result, 'strategy_used', strategy_name)
            }

            print(f"   📊 Chunks: {len(chunks)}")
            print(f"   📏 Avg Size: {avg_size:.0f} chars")
            print(f"   📐 Consistency: {results[strategy_name]['size_consistency']:.2f}")
            print(f"   ⏱️  Time: {processing_time:.4f}s")

            if strategy_name == "adaptive" and hasattr(result, 'source_info'):
                adaptive_strategy = result.source_info.get('adaptive_strategy', 'unknown')
                print(f"   🎯 Selected: {adaptive_strategy}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[strategy_name] = None
        finally:
            import os
            os.unlink(temp_file)

    # Summary comparison
    print("\n📊 STRATEGY COMPARISON SUMMARY")
    print("-" * 40)
    successful_results = {k: v for k, v in results.items() if v is not None}

    if successful_results:
        # Find best performer in different categories
        fastest = min(successful_results.items(), key=lambda x: x[1]['processing_time'])
        most_consistent = max(successful_results.items(), key=lambda x: x[1]['size_consistency'])

        print(f"🏃 Fastest: {fastest[0]} ({fastest[1]['processing_time']:.4f}s)")
        print(f"📐 Most Consistent: {most_consistent[0]} (consistency: {most_consistent[1]['size_consistency']:.2f})")

def demonstrate_persistence():
    """Demonstrate persistence of learned data."""
    print("\n💾 PERSISTENCE DEMONSTRATION")
    print("=" * 60)

    # Create adaptive chunker with persistence
    persistence_file = "adaptive_chunker_history.json"

    print("🔧 Creating adaptive chunker with persistence enabled...")
    adaptive_chunker = create_chunker("adaptive",
                                    persistence_file=persistence_file,
                                    auto_save_interval=2)  # Save every 2 operations

    # Process some content to generate history
    test_content = create_sample_contents()["conversational"]

    print("📊 Processing content to build history...")
    for i in range(3):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"Iteration {i+1}: {test_content}")
            temp_file = f.name

        try:
            result = adaptive_chunker.chunk(temp_file)
            adaptive_chunker.adapt_parameters(0.8, "quality")  # Provide good feedback
            print(f"   ✅ Processed iteration {i+1}")
        except Exception as e:
            print(f"   ❌ Error in iteration {i+1}: {e}")
        finally:
            import os
            os.unlink(temp_file)

    # Check if persistence file was created
    if Path(persistence_file).exists():
        print(f"✅ History file created: {persistence_file}")

        # Show some of the saved data
        with open(persistence_file, 'r') as f:
            history = json.load(f)

        print(f"   📊 Operations recorded: {history.get('operation_count', 0)}")
        print(f"   🎯 Current strategy: {history.get('current_strategy', 'unknown')}")
        print(f"   📈 Content mappings: {len(history.get('content_strategy_map', {}))}")

        # Clean up
        Path(persistence_file).unlink()
        print("🧹 Cleaned up persistence file")
    else:
        print("❌ No persistence file created")

def main():
    """Run comprehensive adaptive chunking demonstration."""
    print("🚀 ADAPTIVE CHUNKING LEARNING DEMONSTRATION")
    print("=" * 80)
    print()
    print("This demo showcases the sophisticated learning and adaptation")
    print("capabilities of the Adaptive Chunker system.")
    print()

    try:
        # Create adaptive chunker with learning enabled
        print("🔧 Initializing Adaptive Chunker...")
        adaptive_chunker = create_chunker("adaptive",
                                        # Strategy selection
                                        available_strategies=["sentence_based", "paragraph_based", "fixed_size", "fastcdc"],
                                        strategy_selection_mode="auto",

                                        # Learning parameters
                                        adaptation_threshold=0.1,
                                        learning_rate=0.1,
                                        exploration_rate=0.05,

                                        # Enable all learning features
                                        enable_content_profiling=True,
                                        enable_performance_learning=True,
                                        enable_strategy_comparison=True,

                                        # History tracking
                                        history_size=100,
                                        performance_window=10,
                                        min_samples=2)

        print("✅ Adaptive chunker initialized successfully!")
        print()

        # Run demonstrations
        demonstrate_content_profiling(adaptive_chunker)
        demonstrate_learning_and_adaptation(adaptive_chunker)
        demonstrate_strategy_comparison()
        demonstrate_persistence()

        # Final adaptation status
        print("\n🎯 FINAL ADAPTATION STATUS")
        print("=" * 60)
        adaptation_info = adaptive_chunker.get_adaptation_info()

        print(f"📊 Total Operations: {adaptation_info['operation_count']}")
        print(f"🔄 Total Adaptations: {adaptation_info['total_adaptations']}")
        print(f"🎯 Current Strategy: {adaptation_info['current_strategy']}")
        print(f"📈 Learning Rate: {adaptation_info['learning_rate']:.3f}")
        print(f"🔍 Exploration Rate: {adaptation_info['exploration_rate']:.3f}")

        if adaptation_info['strategy_performance']:
            print(f"\n📈 Strategy Performance Summary:")
            for strategy, stats in adaptation_info['strategy_performance'].items():
                print(f"   {strategy}: {stats['usage_count']} uses, "
                      f"avg score: {stats['avg_score']:.3f}, "
                      f"recent: {stats['recent_score']:.3f}")

        if adaptation_info['content_strategy_mappings']:
            print(f"\n🗺️  Content-Strategy Mappings: {len(adaptation_info['content_strategy_mappings'])} learned")

        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("\nKey Takeaways:")
        print("• Adaptive chunker analyzes content characteristics automatically")
        print("• It learns from performance feedback and adapts strategies/parameters")
        print("• Historical data is used to make better decisions over time")
        print("• The system can persist learned knowledge across sessions")
        print("• Multiple adaptation modes (content-based, performance-based, exploration)")
        print("• Real-time parameter optimization based on feedback")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
