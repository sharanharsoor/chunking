# 🔧 API Reference

Complete Python API documentation for the chunking-strategy library.

## 📋 Table of Contents

- [Core Functions](#-core-functions)
- [Chunking Results](#-chunking-results)
- [Configuration](#-configuration)
- [Advanced APIs](#-advanced-apis)
- [Utilities](#-utilities)
- [Integration Helpers](#-integration-helpers)

## 🚀 Core Functions

### create_chunker()

Create a chunking strategy instance.

```python
from chunking_strategy import create_chunker

chunker = create_chunker(
    strategy_name: str,
    **parameters
) -> BaseChunker
```

**Parameters:**
- `strategy_name`: Name of the chunking strategy
- `**parameters`: Strategy-specific parameters

**Examples:**
```python
# Basic text chunking
chunker = create_chunker("sentence_based")

# With parameters
chunker = create_chunker(
    "sentence_based",
    max_sentences=3,
    overlap_sentences=1
)

# PDF processing
chunker = create_chunker(
    "pdf_chunker",
    extract_images=True,
    extract_tables=True
)

# Fixed size with custom parameters
chunker = create_chunker(
    "fixed_size",
    chunk_size=1000,
    overlap_size=100
)
```

### list_chunkers()

List all available chunking strategies.

```python
from chunking_strategy import list_chunkers

strategies = list_chunkers(
    category: Optional[str] = None,
    complexity: Optional[str] = None,
    include_heavy: bool = True
) -> List[str]
```

**Parameters:**
- `category`: Filter by category ('text', 'document', 'code', 'multimedia', 'data')
- `complexity`: Filter by complexity ('low', 'medium', 'high')
- `include_heavy`: Include ML-based strategies

**Examples:**
```python
# All strategies
all_strategies = list_chunkers()

# Only text strategies
text_strategies = list_chunkers(category="text")

# Low complexity only
simple_strategies = list_chunkers(complexity="low")

# Exclude heavy ML-based strategies
light_strategies = list_chunkers(include_heavy=False)
```

### ChunkerOrchestrator

High-level orchestration for automatic strategy selection.

```python
from chunking_strategy import ChunkerOrchestrator

orchestrator = ChunkerOrchestrator(
    config: Optional[Dict] = None,
    config_path: Optional[str] = None
)

# Chunk a file with automatic strategy selection
result = orchestrator.chunk_file(
    file_path: Union[str, Path],
    strategy: Optional[str] = None,
    **kwargs
) -> ChunkingResult

# Chunk text content
result = orchestrator.chunk_text(
    content: str,
    strategy: Optional[str] = None,
    **kwargs
) -> ChunkingResult
```

**Examples:**
```python
# Auto strategy selection
orchestrator = ChunkerOrchestrator()
result = orchestrator.chunk_file("document.pdf")

# With configuration
orchestrator = ChunkerOrchestrator(config_path="config.yaml")
result = orchestrator.chunk_file("document.pdf")

# Override strategy
result = orchestrator.chunk_file("doc.pdf", strategy="pdf_chunker")
```

## 📊 Chunking Results

### ChunkingResult

Main result object returned by chunking operations.

```python
class ChunkingResult:
    chunks: List[Chunk]                    # List of generated chunks
    strategy_used: str                     # Strategy name used
    processing_time: float                 # Processing time in seconds
    source_info: Dict[str, Any]           # Source file information
    total_chunks: int                     # Total number of chunks
    quality_score: Optional[float]        # Quality score (if available)

    # Methods
    def to_dict(self) -> Dict              # Convert to dictionary
    def to_json(self) -> str              # Convert to JSON string
    def save(self, path: str)             # Save to file
    def get_chunks_by_type(self, chunk_type: str) -> List[Chunk]  # Filter chunks
```

**Examples:**
```python
result = chunker.chunk("document.pdf")

# Access results
print(f"Created {result.total_chunks} chunks")
print(f"Processing took {result.processing_time:.2f}s")
print(f"Used strategy: {result.strategy_used}")

# Export results
result.save("chunks.json")
result_dict = result.to_dict()
result_json = result.to_json()

# Filter chunks
text_chunks = result.get_chunks_by_type("text")
```

### Chunk

Individual chunk object.

```python
class Chunk:
    id: str                               # Unique chunk identifier
    content: str                          # Chunk text content
    modality: ModalityType               # Content type (TEXT, IMAGE, AUDIO, etc.)
    metadata: ChunkMetadata              # Chunk metadata

    # Properties
    @property
    def word_count(self) -> int          # Number of words
    @property
    def char_count(self) -> int          # Number of characters
    @property
    def is_empty(self) -> bool           # Whether chunk is empty
```

**Examples:**
```python
for chunk in result.chunks:
    print(f"ID: {chunk.id}")
    print(f"Content: {chunk.content[:100]}...")
    print(f"Words: {chunk.word_count}")
    print(f"Type: {chunk.modality}")
    print(f"Source: {chunk.metadata.source}")
```

### ChunkMetadata

Metadata associated with each chunk.

```python
class ChunkMetadata:
    source: Optional[str]                # Source file path
    page: Optional[int]                  # Page number (for documents)
    start_pos: int                       # Start position in source
    end_pos: int                         # End position in source
    word_count: int                      # Number of words
    char_count: int                      # Number of characters
    language: Optional[str]              # Detected language
    extra: Dict[str, Any]               # Additional metadata

    # Methods
    def to_dict(self) -> Dict           # Convert to dictionary
```

## ⚙️ Configuration

### ChunkingConfig

Configuration object for chunking operations.

```python
from chunking_strategy.core.config import ChunkingConfig

config = ChunkingConfig(
    strategy: str = "auto",
    parameters: Dict[str, Any] = None,
    fallback_strategies: List[str] = None,
    quality_validation: bool = False,
    preprocessing: Dict[str, Any] = None,
    postprocessing: Dict[str, Any] = None
)
```

**Examples:**
```python
# Basic configuration
config = ChunkingConfig(
    strategy="sentence_based",
    parameters={"max_sentences": 3}
)

# Advanced configuration with fallbacks
config = ChunkingConfig(
    strategy="semantic",
    fallback_strategies=["sentence_based", "fixed_size"],
    quality_validation=True,
    parameters={
        "similarity_threshold": 0.8,
        "max_chunk_size": 1000
    }
)

# Use configuration
orchestrator = ChunkerOrchestrator(config=config.__dict__)
```

### Configuration from YAML

```python
# Load from YAML file
orchestrator = ChunkerOrchestrator(config_path="config.yaml")

# Or load configuration manually
from chunking_strategy.core.config import load_config
config = load_config("config.yaml")
```

## 🔬 Advanced APIs

### BatchProcessor

High-performance batch processing.

```python
from chunking_strategy.core.batch import BatchProcessor

processor = BatchProcessor(
    default_strategy: str = "auto",
    parallel_mode: str = "auto",
    workers: Optional[int] = None,
    batch_size: Optional[int] = None
)

result = processor.process_files(
    files: List[Union[str, Path]],
    **kwargs
) -> BatchResult
```

**Examples:**
```python
# Basic batch processing
processor = BatchProcessor()
result = processor.process_files([
    "doc1.pdf", "doc2.txt", "doc3.docx"
])

# High-performance processing
processor = BatchProcessor(
    parallel_mode="process",
    workers=8,
    batch_size=50
)
result = processor.process_files(file_list, default_strategy="sentence_based")
```

### StreamingChunker

Memory-efficient processing of large files.

```python
from chunking_strategy.core.streaming import StreamingChunker

streamer = StreamingChunker(
    strategy_name: str,
    block_size: int = 64 * 1024,
    overlap_size: int = 1024,
    **strategy_params
)

# Stream processing
for chunk in streamer.stream_file(file_path: str):
    process_chunk(chunk)

# Get progress information
progress = streamer.get_progress()
```

**Examples:**
```python
# Stream large file
streamer = StreamingChunker(
    "sentence_based",
    block_size=64*1024,
    max_sentences=3
)

for chunk in streamer.stream_file("huge_file.txt"):
    # Process each chunk as it's generated
    print(f"Progress: {streamer.get_progress().progress_percentage:.1f}%")
    process_chunk_immediately(chunk)
```

### Quality Evaluation

```python
from chunking_strategy.core.metrics import ChunkingQualityEvaluator

evaluator = ChunkingQualityEvaluator()

metrics = evaluator.evaluate(
    chunks: List[Chunk]
) -> QualityMetrics

# Quality metrics
print(f"Coherence: {metrics.coherence:.3f}")
print(f"Size consistency: {metrics.size_consistency:.3f}")
print(f"Coverage: {metrics.coverage:.3f}")
print(f"Overall score: {metrics.overall_score:.3f}")
```

### Embedding Generation

```python
from chunking_strategy.core.embeddings import generate_embeddings

embeddings = generate_embeddings(
    chunks: List[Chunk],
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True
) -> List[np.ndarray]
```

**Examples:**
```python
# Generate embeddings for chunks
result = chunker.chunk("document.pdf")
embeddings = generate_embeddings(
    result.chunks,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Use with different models
embeddings = generate_embeddings(
    result.chunks,
    model="sentence-transformers/all-mpnet-base-v2",
    batch_size=16,
    normalize=True
)
```

## 🛠️ Utilities

### Hardware Information

```python
from chunking_strategy.core.hardware import get_hardware_info

hardware = get_hardware_info()

print(f"CPU cores: {hardware.cpu_count}")
print(f"Memory: {hardware.memory_total_gb:.1f} GB")
print(f"GPUs: {hardware.gpu_count}")
print(f"Recommended batch size: {hardware.recommended_batch_size}")
```

### File Type Detection

```python
# File type is typically detected automatically by chunkers
# For manual detection, you can check file extensions:
import pathlib

file_path = pathlib.Path("document.pdf")
file_extension = file_path.suffix  # ".pdf"
file_size = file_path.stat().st_size  # size in bytes
```

### Preprocessing & Postprocessing

```python
from chunking_strategy.utils.preprocessing import PreprocessingPipeline
from chunking_strategy.utils.postprocessing import PostprocessingPipeline

# Preprocessing pipeline
preprocessor = PreprocessingPipeline()
processed_content = preprocessor.process(raw_content)

# Postprocessing pipeline
postprocessor = PostprocessingPipeline()
processed_chunks = postprocessor.process(chunks)
```

## 🔗 Integration Helpers

### LangChain Integration

```python
# Convert chunks to LangChain Document format manually
from langchain.schema import Document

result = chunker.chunk("document.pdf")
langchain_docs = [
    Document(
        page_content=chunk.content,
        metadata={
            "source": chunk.metadata.source,
            "chunk_id": chunk.id
        }
    )
    for chunk in result.chunks
]
```

### Vector Database Export

```python
# Export chunks for vector databases manually
result = chunker.chunk("document.pdf")

# Format for vector database insertion
vector_data = [
    {
        "id": chunk.id,
        "content": chunk.content,
        "metadata": chunk.metadata.to_dict()
    }
    for chunk in result.chunks
]
```

### Universal Apply Strategy

```python
from chunking_strategy import apply_universal_strategy

# Apply any strategy to any file type
result = apply_universal_strategy(
    strategy_name: str,
    file_path: str,
    **parameters
) -> ChunkingResult
```

**Examples:**
```python
# Apply sentence chunking to any file type
result = apply_universal_strategy("sentence", "code.py")
result = apply_universal_strategy("sentence", "document.pdf")
result = apply_universal_strategy("sentence", "data.json")

# Apply paragraph chunking universally
result = apply_universal_strategy(
    "paragraph",
    "any_file.ext",
    max_paragraphs=2
)
```

## 🎯 Common Patterns

### Error Handling with Fallbacks

```python
def robust_chunking(file_path: str, strategies: List[str] = None):
    """Chunk with automatic fallback strategies."""
    if strategies is None:
        strategies = ["sentence_based", "paragraph_based", "fixed_size"]

    for strategy in strategies:
        try:
            chunker = create_chunker(strategy)
            return chunker.chunk(file_path)
        except Exception as e:
            print(f"Strategy {strategy} failed: {e}")
            continue

    raise Exception("All chunking strategies failed")

# Use with automatic fallbacks
result = robust_chunking("document.pdf")
```

### Custom Configuration Pipeline

```python
def create_rag_optimized_chunker(chunk_size: int = 1000):
    """Create chunker optimized for RAG systems."""
    return create_chunker(
        "sentence_based",
        max_sentences=3,
        overlap_sentences=1,
        target_chunk_size=chunk_size,
        preserve_context=True
    )

# Use RAG-optimized chunker
chunker = create_rag_optimized_chunker(chunk_size=800)
result = chunker.chunk("knowledge_base.pdf")
```

### Batch Processing with Quality Control

```python
from chunking_strategy.core.batch import BatchProcessor
from chunking_strategy.core.metrics import ChunkingQualityEvaluator

def quality_controlled_batch(files: List[str], min_quality: float = 0.7):
    """Batch process with quality validation."""
    processor = BatchProcessor(parallel_mode="process")
    evaluator = ChunkingQualityEvaluator()

    results = []
    for file_path in files:
        try:
            result = processor.process_files([file_path])
            metrics = evaluator.evaluate(result.chunks)

            if metrics.overall_score >= min_quality:
                results.append(result)
            else:
                print(f"Quality too low for {file_path}: {metrics.overall_score:.3f}")

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    return results

# Use with quality control
good_results = quality_controlled_batch(file_list, min_quality=0.8)
```

## 🆘 Exception Handling

### Common Exceptions

```python
from chunking_strategy.exceptions import (
    ChunkerError,
    ChunkerNotFoundError,
    ChunkingConfigurationError,
    ChunkingProcessingError
)

try:
    chunker = create_chunker("invalid_strategy")
except ChunkerNotFoundError as e:
    print(f"Strategy not found: {e}")

try:
    result = chunker.chunk("invalid_file.xyz")
except ChunkingProcessingError as e:
    print(f"Processing failed: {e}")

try:
    orchestrator = ChunkerOrchestrator(config_path="invalid_config.yaml")
except ChunkingConfigurationError as e:
    print(f"Configuration error: {e}")
```

---

## 🎯 Quick Reference

**Core Functions:**
```python
create_chunker(strategy, **params)      # Create chunker
list_chunkers(category=None)            # List strategies
ChunkerOrchestrator(config)            # High-level orchestrator
```

**Result Objects:**
```python
result.chunks                          # List of chunks
result.strategy_used                   # Strategy name
result.processing_time                 # Time taken
chunk.content                          # Chunk text
chunk.metadata                         # Chunk metadata
```

**Advanced Features:**
```python
BatchProcessor().process_files(files)   # Batch processing
StreamingChunker().stream_file(path)   # Streaming
generate_embeddings(chunks)            # Embeddings
ChunkingQualityEvaluator().evaluate()  # Quality metrics
```

---

**🚀 Ready to integrate chunking into your application!** Refer to `/examples/` for complete working examples.
