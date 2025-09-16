# Custom Algorithms Directory

This directory contains example custom chunking algorithms that demonstrate how to extend the chunking library with your own strategies. Each algorithm showcases different approaches to content segmentation and can serve as a foundation for your own custom implementations.

## 📚 Available Custom Algorithms

### 1. 🎯 balanced_length_chunker.py
**Purpose**: Maintains consistent chunk sizes across different content types

**Algorithm Details**:
- **Target**: Produces chunks with uniform character length
- **Method**: Dynamic boundary detection with size balancing
- **Use Cases**:
  - Uniform processing requirements
  - Memory-constrained environments
  - Consistent embedding generation

**Key Features**:
- Configurable target length (default: 500 characters)
- Smart boundary detection (sentence/paragraph breaks)
- Overlap support for context preservation
- Quality metrics calculation

**Usage Example**:
```python
from examples.custom_algorithms.balanced_length_chunker import BalancedLengthChunker

chunker = BalancedLengthChunker(
    target_length=500,
    tolerance=0.2,  # ±20% size variation allowed
    overlap_size=50
)

result = chunker.chunk(content)
print(f"Generated {len(result.chunks)} balanced chunks")
```

**Configuration Options**:
- `target_length`: Desired chunk size in characters
- `tolerance`: Acceptable size variation (0.0-1.0)
- `overlap_size`: Number of overlapping characters
- `preserve_sentences`: Keep sentences intact when possible
- `min_chunk_size`: Minimum allowed chunk size

---

### 2. 🔍 regex_pattern_chunker.py
**Purpose**: Pattern-based chunking using configurable regular expressions

**Algorithm Details**:
- **Target**: Split content based on regex patterns
- **Method**: Pattern matching with fallback strategies
- **Use Cases**:
  - Structured content with known patterns
  - Format-specific document processing
  - Custom delimiter-based splitting

**Key Features**:
- Multiple regex pattern support
- Configurable chunk size limits
- Pattern priority system
- Fallback to size-based chunking

**Usage Example**:
```python
from examples.custom_algorithms.regex_pattern_chunker import RegexPatternChunker

# Custom patterns for markdown content
patterns = [
    r'^#{1,6}\s+.*$',      # Headers
    r'\n\n+',             # Paragraph breaks
    r'\n\s*[-*+]\s+',     # List items
    r'```[\s\S]*?```'     # Code blocks
]

chunker = RegexPatternChunker(
    patterns=patterns,
    min_chunk_length=100,
    max_chunk_length=1000
)

result = chunker.chunk(markdown_content)
```

**Configuration Options**:
- `patterns`: List of regex patterns for splitting
- `min_chunk_length`: Minimum chunk size threshold
- `max_chunk_length`: Maximum chunk size limit
- `pattern_priority`: Order of pattern application
- `fallback_strategy`: Action when patterns fail

---

### 3. 🎭 sentiment_based_chunker.py
**Purpose**: Groups content based on sentiment analysis and emotional coherence

**Algorithm Details**:
- **Target**: Create emotionally coherent chunks
- **Method**: Sentiment analysis with clustering
- **Use Cases**:
  - Emotional content analysis
  - Narrative processing
  - Opinion mining applications

**Key Features**:
- Sentiment-aware boundary detection
- Emotional coherence scoring
- Support for multiple sentiment models
- Adaptive chunk sizing based on sentiment distribution

**Usage Example**:
```python
from examples.custom_algorithms.sentiment_based_chunker import SentimentBasedChunker

chunker = SentimentBasedChunker(
    sentiment_model="vader",  # or "textblob", "transformer"
    coherence_threshold=0.7,
    max_chunk_sentences=10
)

result = chunker.chunk(narrative_content)

# Access sentiment metadata
for chunk in result.chunks:
    sentiment = chunk.metadata.extra.get('sentiment_score')
    emotion = chunk.metadata.extra.get('dominant_emotion')
    print(f"Chunk sentiment: {sentiment}, emotion: {emotion}")
```

**Configuration Options**:
- `sentiment_model`: Choice of sentiment analysis model
- `coherence_threshold`: Minimum emotional coherence required
- `max_chunk_sentences`: Maximum sentences per chunk
- `sentiment_window`: Context window for sentiment analysis
- `emotion_categories`: Custom emotion classification

---

### 4. ⚙️ custom_config_example.yaml
**Purpose**: Configuration example for using custom algorithms

**What it demonstrates**:
- Custom algorithm configuration syntax
- Parameter specification
- Integration with existing strategies
- Multi-algorithm workflow setup

**Usage Example**:
```bash
# Use with CLI
chunking-cli process --config custom_config_example.yaml document.txt

# Use with Python
from chunking_strategy import ChunkerOrchestrator
orchestrator = ChunkerOrchestrator(config_path="custom_config_example.yaml")
result = orchestrator.chunk_content(content)
```

---

## 🛠️ Creating Your Own Custom Algorithm

### Step 1: Basic Algorithm Structure

```python
from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult
from typing import List, Optional, Dict, Any

class MyCustomChunker(BaseChunker):
    """
    Your custom chunking algorithm.

    Describe what your algorithm does and when to use it.
    """

    def __init__(self,
                 custom_param1: str = "default_value",
                 custom_param2: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.custom_param1 = custom_param1
        self.custom_param2 = custom_param2

    def chunk(self, content: str, **kwargs) -> ChunkingResult:
        """
        Implement your custom chunking logic here.

        Args:
            content: Input text to chunk
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with generated chunks
        """

        # 1. Implement your chunking logic
        segments = self._custom_segmentation_logic(content)

        # 2. Create Chunk objects
        chunks = []
        for i, segment in enumerate(segments):
            chunk = Chunk(
                content=segment,
                start_index=0,  # Calculate actual start position
                end_index=len(segment),  # Calculate actual end position
                metadata={
                    'chunk_index': i,
                    'algorithm': self.__class__.__name__,
                    'custom_metadata': self._calculate_custom_metadata(segment)
                }
            )
            chunks.append(chunk)

        # 3. Return ChunkingResult
        return ChunkingResult(
            chunks=chunks,
            strategy_used=self.__class__.__name__.lower(),
            total_chunks=len(chunks),
            processing_time=0.0,  # Track actual processing time
            metadata={
                'algorithm_specific_info': 'your_custom_info'
            }
        )

    def _custom_segmentation_logic(self, content: str) -> List[str]:
        """Implement your specific segmentation algorithm."""
        # Your custom logic here
        return [content]  # Placeholder

    def _calculate_custom_metadata(self, segment: str) -> Dict[str, Any]:
        """Calculate custom metadata for each chunk."""
        return {
            'segment_length': len(segment),
            'custom_metric': self._calculate_custom_metric(segment)
        }

    def _calculate_custom_metric(self, segment: str) -> float:
        """Calculate algorithm-specific metrics."""
        return 0.0  # Placeholder
```

### Step 2: Algorithm Registration

```python
# Add to the end of your custom algorithm file
def get_chunkers():
    """Return available chunkers from this module."""
    return {
        "my_custom_algorithm": MyCustomChunker
    }

# For direct usage without configuration
if __name__ == "__main__":
    # Demo usage
    chunker = MyCustomChunker(custom_param1="test", custom_param2=200)

    test_content = "Your test content here..."
    result = chunker.chunk(test_content)

    print(f"Generated {len(result.chunks)} chunks")
    for i, chunk in enumerate(result.chunks):
        print(f"Chunk {i+1}: {chunk.content[:50]}...")
```

### Step 3: Configuration Integration

Create a YAML configuration file:

```yaml
# my_algorithm_config.yaml
profile_name: "my_custom_profile"
description: "Custom algorithm demonstration"

strategies:
  primary: "my_custom_algorithm"
  fallbacks: ["sentence_based", "fixed_size"]

custom_algorithms:
  - path: "path/to/my_custom_chunker.py"
    algorithms: ["my_custom_algorithm"]

parameters:
  my_custom_algorithm:
    custom_param1: "production_value"
    custom_param2: 500
    enable_advanced_features: true

chunking:
  default_strategy: "my_custom_algorithm"
  quality_threshold: 0.8
  enable_metrics: true
```

### Step 4: Testing Your Algorithm

```python
# test_my_custom_algorithm.py
import unittest
from my_custom_chunker import MyCustomChunker

class TestMyCustomChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = MyCustomChunker()
        self.test_content = "Your test content here..."

    def test_basic_chunking(self):
        """Test basic chunking functionality."""
        result = self.chunker.chunk(self.test_content)

        self.assertIsNotNone(result)
        self.assertGreater(len(result.chunks), 0)
        self.assertEqual(result.strategy_used, "mycustomchunker")

    def test_custom_parameters(self):
        """Test custom parameter handling."""
        chunker = MyCustomChunker(custom_param1="test", custom_param2=999)
        result = chunker.chunk(self.test_content)

        self.assertIsNotNone(result)

    def test_empty_content(self):
        """Test handling of empty content."""
        result = self.chunker.chunk("")

        # Define expected behavior for empty content
        self.assertIsNotNone(result)

    def test_large_content(self):
        """Test performance with large content."""
        large_content = self.test_content * 1000
        result = self.chunker.chunk(large_content)

        self.assertIsNotNone(result)
        self.assertGreater(len(result.chunks), 0)

if __name__ == "__main__":
    unittest.main()
```

---

## 🔧 Integration Patterns

### With ChunkerOrchestrator

```python
from chunking_strategy import ChunkerOrchestrator

# Method 1: Configuration-based
orchestrator = ChunkerOrchestrator(config_path="my_algorithm_config.yaml")
result = orchestrator.chunk_content(content)

# Method 2: Programmatic integration
from my_custom_chunker import MyCustomChunker

orchestrator = ChunkerOrchestrator()
orchestrator.register_custom_chunker("my_algorithm", MyCustomChunker)
result = orchestrator.chunk_content(content, strategy="my_algorithm")
```

### With Existing Strategies

```python
# Use as primary with fallbacks
config = {
    "strategies": {
        "primary": "my_custom_algorithm",
        "fallbacks": ["sentence_based", "fixed_size"]
    },
    "custom_algorithms": [
        {
            "path": "my_custom_chunker.py",
            "algorithms": ["my_custom_algorithm"]
        }
    ]
}

orchestrator = ChunkerOrchestrator(config=config)
result = orchestrator.chunk_content(content)
```

### With Multi-Strategy Processing

```python
# Compare your algorithm with others
from chunking_strategy.benchmarking import ChunkingBenchmark

benchmark = ChunkingBenchmark()
strategies = ["my_custom_algorithm", "sentence_based", "semantic"]

results = {}
for strategy in strategies:
    metrics = benchmark.benchmark_strategy(strategy, test_content)
    results[strategy] = metrics

# Analyze results
for strategy, metrics in results.items():
    print(f"{strategy}: {metrics.throughput_mb_per_sec:.2f} MB/s")
```

---

## 📊 Best Practices

### 1. Algorithm Design
- **Clear Purpose**: Define what problem your algorithm solves
- **Configurable**: Make parameters adjustable via configuration
- **Robust**: Handle edge cases (empty content, very large content)
- **Efficient**: Optimize for your specific use case
- **Metadata Rich**: Provide useful metadata for downstream processing

### 2. Code Quality
- **Documentation**: Comprehensive docstrings and comments
- **Type Hints**: Use Python type annotations
- **Error Handling**: Graceful handling of failures
- **Logging**: Appropriate logging for debugging
- **Testing**: Comprehensive test coverage

### 3. Integration
- **Standard Interface**: Follow BaseChunker interface
- **Configuration**: Support YAML configuration
- **Registration**: Implement get_chunkers() function
- **Fallbacks**: Graceful degradation when algorithm fails
- **Metrics**: Provide performance and quality metrics

### 4. Performance
- **Benchmarking**: Test with various content types and sizes
- **Memory Efficiency**: Consider memory usage for large content
- **Speed Optimization**: Profile and optimize critical paths
- **Scalability**: Test with production-scale data
- **Resource Usage**: Monitor CPU and memory consumption

---

## 🔍 Debugging and Troubleshooting

### Common Issues

#### Import Errors
```python
# Ensure proper path setup
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from my_custom_chunker import MyCustomChunker
```

#### Configuration Issues
```python
# Validate configuration before use
def validate_config(config):
    required_keys = ['custom_param1', 'custom_param2']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required parameter: {key}")
```

#### Performance Issues
```python
# Add timing and profiling
import time
import cProfile

def chunk_with_profiling(self, content):
    start_time = time.time()

    # Your chunking logic here
    result = self._actual_chunking_logic(content)

    processing_time = time.time() - start_time
    result.processing_time = processing_time

    return result
```

### Debugging Tools

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug information to chunks
chunk_metadata = {
    'debug_info': {
        'algorithm_version': "1.0.0",
        'processing_steps': ["step1", "step2", "step3"],
        'intermediate_results': debug_data
    }
}
```

---

## 📚 Additional Resources

### Learning Path
1. **Study existing algorithms**: Review the three provided examples
2. **Start simple**: Begin with basic pattern-based chunking
3. **Add complexity gradually**: Introduce advanced features incrementally
4. **Test thoroughly**: Use various content types and edge cases
5. **Optimize**: Profile and improve performance

### Example Use Cases
- **Domain-specific chunking**: Legal documents, medical records, technical manuals
- **Format-specific processing**: Structured data, markup languages, code files
- **Content-aware segmentation**: Narrative flow, dialogue detection, topic boundaries
- **Quality-driven chunking**: Semantic coherence, information density, readability

### Integration Examples
- **RAG systems**: Optimize chunks for embedding quality
- **Content analysis**: Preserve analytical context
- **Data pipelines**: Integrate with streaming workflows
- **Multi-modal processing**: Handle text, metadata, and annotations

---

## 🤝 Contributing New Algorithms

To contribute your custom algorithm back to the project:

1. **Follow the established patterns** in existing examples
2. **Add comprehensive tests** covering edge cases
3. **Include configuration examples** showing usage
4. **Document thoroughly** with clear use cases
5. **Performance benchmark** against existing strategies
6. **Create integration examples** showing real-world usage

---

**Happy Algorithm Development! 🚀**

*This directory provides everything needed to create, test, and deploy custom chunking algorithms that extend the library's capabilities for your specific use cases.*
