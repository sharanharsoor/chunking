# Custom Chunking Algorithms Framework

## Overview

The Chunking Strategy Library now supports a powerful framework for integrating your own custom chunking algorithms. This framework allows you to:

- **Plug in your own algorithms**: Write custom chunking logic in simple Python files
- **Seamless integration**: Use all existing features (metrics, logging, CLI, configuration)
- **Compare algorithms**: Compare your custom algorithms with built-in ones
- **Run multiple algorithms**: Execute multiple custom algorithms in parallel
- **Comprehensive validation**: Ensure your algorithms work correctly and efficiently

## Table of Contents

1. [Quick Start](#quick-start)
2. [Writing Custom Algorithms](#writing-custom-algorithms)
3. [Configuration Integration](#configuration-integration)
4. [CLI Usage](#cli-usage)
5. [Validation and Testing](#validation-and-testing)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

## Quick Start

### 1. Create Your First Custom Algorithm

Create a file called `my_chunker.py`:

```python
from chunking_strategy.core.base import BaseChunker, ChunkingResult, Chunk, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
import time

@register_chunker(
    name="my_word_count_chunker",
    category="text",
    description="Chunks text based on word count with customizable word limits",
    complexity=ComplexityLevel.LOW,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    quality=0.7,
    use_cases=["word-based processing", "even distribution"],
    default_parameters={"words_per_chunk": 50}
)
class MyWordCountChunker(BaseChunker):
    def __init__(self, words_per_chunk=50, **kwargs):
        super().__init__(name="my_word_count_chunker", category="text", **kwargs)
        self.words_per_chunk = words_per_chunk

    def chunk(self, content, source_info=None, **kwargs):
        # Convert content to string
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        content = str(content)

        # Split into words
        words = content.split()
        chunks = []

        # Create chunks based on word count
        for i in range(0, len(words), self.words_per_chunk):
            chunk_words = words[i:i + self.words_per_chunk]
            chunk_content = ' '.join(chunk_words)

            metadata = ChunkMetadata(
                source=source_info.get('source', 'unknown') if source_info else 'unknown',
                chunker_used="my_word_count_chunker",
                extra={"word_count": len(chunk_words)}
            )

            chunk = Chunk(
                id=f"word_chunk_{i // self.words_per_chunk}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=metadata
            )
            chunks.append(chunk)

        return ChunkingResult(
            chunks=chunks,
            strategy_used="my_word_count_chunker"
        )
```

### 2. Load and Use Your Algorithm

```python
from chunking_strategy.core.custom_algorithm_loader import load_custom_algorithm
from chunking_strategy import create_chunker

# Load your custom algorithm
algo_info = load_custom_algorithm("my_chunker.py")
print(f"Loaded: {algo_info.name}")

# Use it like any built-in algorithm
chunker = create_chunker("my_word_count_chunker", words_per_chunk=30)
result = chunker.chunk("Your text content here...")

print(f"Created {len(result.chunks)} chunks")
for chunk in result.chunks:
    print(f"Chunk: {chunk.content[:50]}... ({chunk.metadata.extra['word_count']} words)")
```

### 3. Use with Configuration

Create `config.yaml`:

```yaml
custom_algorithms:
  - path: "my_chunker.py"

strategies:
  primary: "my_word_count_chunker"
  fallbacks: ["fixed_size", "paragraph_based"]

parameters:
  my_word_count_chunker:
    words_per_chunk: 40
```

```python
from chunking_strategy import ChunkerOrchestrator

orchestrator = ChunkerOrchestrator(config_path="config.yaml")
result = orchestrator.chunk_file("document.txt")
```

### 4. Use with CLI

```bash
# Load and test your algorithm
chunking-strategy custom load my_chunker.py

# Use in chunking
chunking-strategy chunk document.txt --config config.yaml

# Validate your algorithm
chunking-strategy custom validate my_chunker.py --comprehensive

# Benchmark against built-in algorithms
chunking-strategy custom benchmark my_chunker.py --compare-with fixed_size,paragraph_based
```

## Writing Custom Algorithms

### Basic Structure

Every custom algorithm must:

1. **Inherit from `BaseChunker`**
2. **Implement the `chunk()` method**
3. **Use the `@register_chunker` decorator** (recommended)

```python
from chunking_strategy.core.base import BaseChunker, ChunkingResult
from chunking_strategy.core.registry import register_chunker

@register_chunker(
    name="my_algorithm",
    category="text",  # or "multimedia", "document", "general", "custom"
    description="Description of what your algorithm does"
)
class MyCustomChunker(BaseChunker):
    def __init__(self, **kwargs):
        super().__init__(name="my_algorithm", **kwargs)

    def chunk(self, content, source_info=None, **kwargs):
        # Your chunking logic here
        chunks = []  # Create your chunks
        return ChunkingResult(chunks=chunks, strategy_used="my_algorithm")
```

### Algorithm Metadata

The `@register_chunker` decorator accepts comprehensive metadata:

```python
@register_chunker(
    # Required
    name="algorithm_name",           # Unique identifier
    category="text",                 # Category: text, multimedia, document, general, custom
    description="Detailed description",

    # Capabilities
    supported_modalities=[ModalityType.TEXT, ModalityType.MIXED],
    supported_formats=["txt", "md", "json"],

    # Characteristics
    complexity=ComplexityLevel.MEDIUM,    # LOW, MEDIUM, HIGH, VERY_HIGH
    speed=SpeedLevel.FAST,               # VERY_FAST, FAST, MEDIUM, SLOW, VERY_SLOW
    memory=MemoryUsage.LOW,              # VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
    quality=0.8,                        # Quality score 0.0-1.0

    # Dependencies
    dependencies=["numpy", "scipy"],      # Required packages
    optional_dependencies=["matplotlib"], # Optional packages

    # Usage information
    use_cases=["specific task", "use case"],
    best_for=["scenario 1", "scenario 2"],
    limitations=["limitation 1", "limitation 2"],

    # Parameters
    parameters_schema={
        "param_name": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000,
            "default": 100,
            "description": "Parameter description"
        }
    },
    default_parameters={"param_name": 100},

    # Advanced features
    streaming_support=False,
    adaptive_support=False,
    hierarchical_support=False
)
```

### Input Handling

Your `chunk()` method receives various input types:

```python
def chunk(self, content, source_info=None, **kwargs):
    # Handle different input types
    if isinstance(content, Path):
        # File path input
        with open(content, 'r', encoding='utf-8') as f:
            text_content = f.read()
        source_name = str(content)
    elif isinstance(content, bytes):
        # Binary content
        text_content = content.decode('utf-8')
        source_name = source_info.get('source', 'unknown') if source_info else 'unknown'
    else:
        # String content
        text_content = str(content)
        source_name = source_info.get('source', 'unknown') if source_info else 'unknown'

    # Your chunking logic here
    chunks = self._create_chunks(text_content, source_name)

    return ChunkingResult(
        chunks=chunks,
        strategy_used=self.name,
        source_info=source_info
    )
```

### Creating Proper Chunks

Always create properly formatted `Chunk` objects:

```python
from chunking_strategy.core.base import Chunk, ChunkMetadata, ModalityType

def _create_chunks(self, content, source_name):
    chunks = []

    # Your chunking logic
    for i, chunk_content in enumerate(self._split_content(content)):
        # Create metadata
        metadata = ChunkMetadata(
            source=source_name,
            position=f"chunk_{i}",
            chunker_used=self.name,
            extra={"custom_field": "custom_value"}  # Add custom metadata
        )

        # Create chunk
        chunk = Chunk(
            id=f"{self.name}_{i:04d}",
            content=chunk_content,
            modality=ModalityType.TEXT,
            metadata=metadata
        )

        chunks.append(chunk)

    return chunks
```

### Advanced Features

#### Streaming Support

For large files, implement streaming:

```python
from chunking_strategy.core.base import StreamableChunker

@register_chunker(
    name="streaming_algorithm",
    streaming_support=True
)
class StreamingCustomChunker(StreamableChunker):
    def chunk_stream(self, content_stream, source_info=None, **kwargs):
        for piece in content_stream:
            # Process piece and yield chunks
            for chunk in self._process_piece(piece):
                yield chunk
```

#### Adaptive Algorithms

For algorithms that learn and adapt:

```python
from chunking_strategy.core.base import AdaptableChunker

@register_chunker(
    name="adaptive_algorithm",
    adaptive_support=True
)
class AdaptiveCustomChunker(AdaptableChunker):
    def adapt_parameters(self, feedback_score, feedback_type="quality", **kwargs):
        # Adjust parameters based on feedback
        if feedback_score < 0.5:
            self.chunk_size *= 0.9  # Make chunks smaller
        elif feedback_score > 0.8:
            self.chunk_size *= 1.1  # Make chunks larger

    def get_adaptation_history(self):
        return self.adaptation_log
```

## Configuration Integration

### Basic Configuration

```yaml
custom_algorithms:
  - path: "path/to/my_algorithm.py"
    algorithms: ["algorithm_name"]  # Optional: specific algorithms to load

strategies:
  primary: "my_algorithm"
  fallbacks: ["built_in_algorithm"]

parameters:
  my_algorithm:
    param1: value1
    param2: value2
```

### Loading from Directories

```yaml
custom_algorithms:
  - path: "algorithms_directory/"
    recursive: true  # Search subdirectories
  - path: "single_algorithm.py"

strategies:
  primary: "algorithm_from_directory"
```

### Multi-Strategy with Custom Algorithms

```yaml
multi_strategy:
  enabled: true
  strategies:
    - name: "my_custom_algorithm"
      weight: 0.5
    - name: "semantic"           # Built-in algorithm
      weight: 0.3
    - name: "paragraph_based"    # Built-in algorithm
      weight: 0.2
```

## CLI Usage

### Loading Algorithms

```bash
# Load single algorithm
chunking-strategy custom load my_algorithm.py

# Load from directory
chunking-strategy custom load-dir algorithms/ --recursive

# List loaded algorithms
chunking-strategy custom list --detailed
```

### Validation

```bash
# Basic validation
chunking-strategy custom validate my_algorithm.py

# Comprehensive validation
chunking-strategy custom validate my_algorithm.py --comprehensive --generate-report

# Batch validation
chunking-strategy custom validate-batch algorithms/ --recursive
```

### Benchmarking

```bash
# Benchmark against default algorithms
chunking-strategy custom benchmark my_algorithm.py

# Compare with specific algorithms
chunking-strategy custom benchmark my_algorithm.py --compare-with fixed_size,semantic,paragraph_based

# Custom test sizes
chunking-strategy custom benchmark my_algorithm.py --test-sizes small,large --iterations 5
```

### Creating Templates

```bash
# Create algorithm template
chunking-strategy custom create-template my_new_algorithm.py --algorithm-name awesome_chunker

# Create configuration template
chunking-strategy custom create-template config.yaml --config-template
```

### Using Custom Algorithms in Chunking

```bash
# Load algorithm and use with config
chunking-strategy chunk document.txt --config custom_config.yaml

# Direct strategy override
chunking-strategy chunk document.txt --strategy my_custom_algorithm
```

## Validation and Testing

### Automatic Validation

The framework automatically validates:

- **Interface compliance**: Proper inheritance and method implementation
- **Functionality**: Basic chunking operations work correctly
- **Performance**: Reasonable speed and memory usage
- **Quality**: Output quality and consistency
- **Security**: Potentially unsafe operations
- **Integration**: Compatibility with framework features

### Running Validation

```python
from chunking_strategy.core.custom_validation import validate_custom_algorithm_file

report = validate_custom_algorithm_file("my_algorithm.py")

if report.is_valid:
    print(f"✅ Algorithm is valid! Score: {report.overall_score:.2f}")
else:
    print("❌ Validation failed:")
    for issue in report.get_all_issues():
        print(f"  {issue.severity.value}: {issue.message}")
```

### Comprehensive Testing

```python
from chunking_strategy.core.custom_validation import run_comprehensive_validation

report = run_comprehensive_validation(
    "my_algorithm.py",
    include_performance=True,
    include_quality_tests=True,
    generate_report_file=True
)

print(f"Overall Score: {report.overall_score:.2f}")
print(f"Interface: {report.interface_score:.2f}")
print(f"Performance: {report.performance_score:.2f}")
print(f"Quality: {report.quality_score:.2f}")
```

## Advanced Features

### Multiple Algorithms in One File

```python
# single_file.py
@register_chunker(name="algorithm_one", category="text")
class AlgorithmOne(BaseChunker):
    def chunk(self, content, **kwargs):
        # Implementation
        pass

@register_chunker(name="algorithm_two", category="text")
class AlgorithmTwo(BaseChunker):
    def chunk(self, content, **kwargs):
        # Implementation
        pass
```

### Parameter Schemas and Validation

```python
@register_chunker(
    name="validated_algorithm",
    parameters_schema={
        "threshold": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.5
        },
        "method": {
            "type": "string",
            "enum": ["fast", "accurate", "balanced"],
            "default": "balanced"
        }
    }
)
class ValidatedAlgorithm(BaseChunker):
    def __init__(self, threshold=0.5, method="balanced", **kwargs):
        super().__init__(name="validated_algorithm", **kwargs)

        # Validate parameters
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if method not in ["fast", "accurate", "balanced"]:
            raise ValueError("method must be 'fast', 'accurate', or 'balanced'")

        self.threshold = threshold
        self.method = method
```

### Integration with Existing Features

Your custom algorithms automatically work with:

- **Metrics tracking**: All metrics are automatically collected
- **Logging**: Framework logging captures your algorithm's behavior
- **Streaming**: Large file processing with memory management
- **Batch processing**: Process multiple files efficiently
- **Quality assessment**: Automatic quality scoring
- **Embeddings**: Generate embeddings for your chunks
- **Export formats**: JSON, YAML, CSV output support

## Best Practices

### 1. Algorithm Design

- **Single responsibility**: Each algorithm should have one clear purpose
- **Configurable**: Make key parameters configurable
- **Boundary aware**: Respect natural text boundaries when possible
- **Error handling**: Handle edge cases gracefully
- **Performance**: Consider memory and CPU efficiency

### 2. Code Quality

```python
# Good: Clear, documented, configurable
@register_chunker(
    name="sentence_window_chunker",
    description="Groups sentences into fixed-size windows with configurable overlap",
    quality=0.8,
    use_cases=["sentence analysis", "overlapping context"]
)
class SentenceWindowChunker(BaseChunker):
    def __init__(self, window_size=5, overlap=1, **kwargs):
        """
        Initialize sentence window chunker.

        Args:
            window_size: Number of sentences per chunk
            overlap: Number of sentences to overlap between chunks
        """
        super().__init__(name="sentence_window_chunker", **kwargs)
        self.window_size = max(1, window_size)
        self.overlap = max(0, min(overlap, window_size - 1))

    def chunk(self, content, source_info=None, **kwargs):
        try:
            sentences = self._extract_sentences(content)
            return self._create_windowed_chunks(sentences, source_info)
        except Exception as e:
            self.logger.error(f"Chunking failed: {e}")
            # Return empty result rather than crashing
            return ChunkingResult(chunks=[], strategy_used=self.name)
```

### 3. Testing

```python
# Include tests in your algorithm file
def test_sentence_window_chunker():
    chunker = SentenceWindowChunker(window_size=2, overlap=1)

    test_content = "First sentence. Second sentence. Third sentence. Fourth sentence."
    result = chunker.chunk(test_content)

    assert len(result.chunks) == 3  # Expected number of chunks
    assert "First sentence" in result.chunks[0].content
    assert "Second sentence" in result.chunks[1].content  # Should overlap

if __name__ == "__main__":
    test_sentence_window_chunker()
    print("✅ Tests passed!")
```

### 4. Documentation

```python
@register_chunker(
    name="well_documented_chunker",
    description="""
    Advanced chunker that uses multiple techniques for optimal results.

    This algorithm combines:
    - Semantic similarity analysis
    - Sentence boundary detection
    - Configurable size constraints

    Best used for: Long-form content analysis, RAG applications
    Limitations: Requires sentence detection, English text preferred
    """,
    use_cases=["RAG", "document analysis", "content summarization"],
    best_for=["academic papers", "articles", "books"],
    limitations=["English text", "requires sentence boundaries", "slower performance"]
)
class WellDocumentedChunker(BaseChunker):
    """
    A well-documented example chunker showing best practices.

    This class demonstrates proper documentation, parameter handling,
    error management, and integration with the framework.
    """

    def __init__(self, target_size=1000, similarity_threshold=0.7, **kwargs):
        """
        Initialize the chunker.

        Args:
            target_size (int): Target size for chunks in characters
            similarity_threshold (float): Similarity threshold for grouping (0.0-1.0)
            **kwargs: Additional parameters passed to BaseChunker
        """
        super().__init__(name="well_documented_chunker", **kwargs)
        # Parameter validation and setup...
```

## Troubleshooting

### Common Issues

#### 1. Algorithm Not Loading

**Problem**: `Failed to load custom algorithm`

**Solutions**:
- Check Python syntax: `python -m py_compile my_algorithm.py`
- Ensure proper imports are available
- Verify file path is correct
- Check for dependency issues

#### 2. Registration Failures

**Problem**: Algorithm loads but not registered

**Solutions**:
- Ensure `@register_chunker` decorator is applied
- Check for duplicate names
- Verify decorator parameters are correct
- Make sure class inherits from `BaseChunker`

#### 3. Runtime Errors

**Problem**: Algorithm fails during chunking

**Solutions**:
- Add proper error handling in `chunk()` method
- Validate input types and handle edge cases
- Test with various input sizes and types
- Use logging to debug issues

#### 4. Performance Issues

**Problem**: Algorithm is too slow

**Solutions**:
- Profile your code to find bottlenecks
- Avoid expensive operations in loops
- Consider streaming for large inputs
- Use efficient data structures

#### 5. Integration Issues

**Problem**: Algorithm doesn't work with other features

**Solutions**:
- Return proper `ChunkingResult` objects
- Create valid `Chunk` objects with metadata
- Handle all expected input types
- Follow the interface contract

### Debugging

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your custom algorithm will now show debug information
```

Use validation to identify issues:

```bash
chunking-strategy custom validate my_algorithm.py --comprehensive --strict
```

Test incrementally:

```python
# Test basic instantiation
chunker = create_chunker("my_algorithm")

# Test with simple content
result = chunker.chunk("Simple test content.")

# Test with various inputs
test_cases = ["", "Short", "Long content " * 100, "Special chars: @#$%"]
for content in test_cases:
    try:
        result = chunker.chunk(content)
        print(f"✅ Handled: {content[:20]}...")
    except Exception as e:
        print(f"❌ Failed: {content[:20]}... - {e}")
```

## API Reference

### Core Classes

#### `BaseChunker`

Base class for all chunking algorithms.

```python
class BaseChunker(ABC):
    def __init__(self, name: str, category: str = "general", **kwargs):
        """Initialize chunker with name and category."""

    @abstractmethod
    def chunk(self, content, source_info=None, **kwargs) -> ChunkingResult:
        """Main chunking method - must be implemented."""

    def validate_input(self, content, expected_modality=None):
        """Validate input content."""

    def create_chunk(self, content, modality, metadata=None, chunk_id=None) -> Chunk:
        """Helper to create properly formatted chunks."""
```

#### `Chunk`

Represents a single chunk of content.

```python
@dataclass
class Chunk:
    id: str                    # Unique identifier
    content: Union[str, bytes, Any]  # Chunk content
    modality: ModalityType     # Content type (TEXT, IMAGE, etc.)
    metadata: ChunkMetadata    # Associated metadata
    size: Optional[int] = None # Size in appropriate units
    hash: Optional[str] = None # Content hash
```

#### `ChunkingResult`

Contains the complete result of a chunking operation.

```python
@dataclass
class ChunkingResult:
    chunks: List[Chunk]        # Generated chunks
    processing_time: Optional[float] = None
    strategy_used: Optional[str] = None
    source_info: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
```

### Registry Functions

#### `@register_chunker`

Decorator to register algorithms with metadata.

```python
@register_chunker(
    name: str,                 # Required: unique algorithm name
    category: str = "general", # Category classification
    description: str = "",     # Algorithm description
    # ... many other optional parameters
)
def decorator(chunker_class):
    # Registers the class with global registry
```

#### Loading Functions

```python
from chunking_strategy.core.custom_algorithm_loader import (
    load_custom_algorithm,     # Load single algorithm
    load_custom_algorithms_directory,  # Load from directory
    list_custom_algorithms,    # List loaded algorithms
    get_custom_algorithm_info  # Get algorithm information
)
```

### Configuration Classes

#### `CustomConfigProcessor`

Processes configurations with custom algorithms.

```python
processor = CustomConfigProcessor()
config = processor.process_config(config_dict, config_path)
loaded_algorithms = processor.get_loaded_algorithms()
```

### Validation Classes

#### `CustomAlgorithmValidator`

Validates custom algorithms comprehensively.

```python
validator = CustomAlgorithmValidator(
    strict_mode=False,
    include_performance_tests=True,
    include_quality_tests=True
)
report = validator.validate_algorithm_file("algorithm.py")
```

---

## Examples

Complete examples are available in the `examples/custom_algorithms/` directory:

1. **`sentiment_based_chunker.py`**: Groups text by sentiment boundaries
2. **`regex_pattern_chunker.py`**: Uses regex patterns for structured text
3. **`balanced_length_chunker.py`**: Creates balanced-length chunks with boundary awareness

## Support

For questions, issues, or contributions:

1. Check existing documentation and examples
2. Use the validation framework to identify issues
3. Run comprehensive tests on your algorithms
4. Review the troubleshooting section

The custom algorithms framework empowers you to extend the chunking library with your own specialized logic while benefiting from all the advanced features and infrastructure already built into the system.
