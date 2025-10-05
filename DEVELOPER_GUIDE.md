# 🛠️ Developer Guide

Comprehensive guide for developers contributing to the chunking-strategy library.

## 📋 Table of Contents

- [Getting Started](#-getting-started)
- [Architecture Overview](#-architecture-overview)
- [Development Setup](#-development-setup)
- [Code Organization](#-code-organization)
- [Adding New Strategies](#-adding-new-strategies)
- [Testing Guidelines](#-testing-guidelines)
- [Performance Considerations](#-performance-considerations)
- [Documentation Standards](#-documentation-standards)
- [Release Process](#-release-process)
- [Contributing Guidelines](#-contributing-guidelines)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of text processing and chunking algorithms
- Familiarity with Python packaging and testing

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/sharanharsoor/chunking.git
cd chunking

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev,all]

# Run tests to verify setup
pytest tests/ -v
```

## 🏗️ Architecture Overview

### Core Components

The library follows a modular architecture with clear separation of concerns:

```
chunking_strategy/
├── core/                    # Core framework components
│   ├── base.py             # Base classes and interfaces
│   ├── registry.py         # Strategy registry system
│   ├── pipeline.py         # Processing pipelines
│   ├── streaming.py        # Streaming capabilities
│   ├── adaptive.py         # Adaptive chunking
│   └── ...
├── strategies/             # Chunking strategy implementations
│   ├── text/              # Text-based strategies
│   ├── code/              # Code-aware strategies
│   ├── document/          # Document processing
│   ├── multimedia/        # Audio/video/image processing
│   └── general/           # General-purpose strategies
├── detectors/             # Content analysis and detection
├── utils/                 # Utility functions
├── orchestrator.py        # High-level orchestration
└── cli.py                 # Command-line interface
```

### Design Principles

1. **Plugin Architecture**: Strategies are self-contained and registerable
2. **Universal Schema**: All chunks follow the same data structure
3. **Lazy Loading**: Heavy dependencies are loaded only when needed
4. **Streaming Support**: Memory-efficient processing for large files
5. **Adaptive Behavior**: Strategies can learn and adapt
6. **Extensibility**: Easy to add new strategies and capabilities

### Key Interfaces

#### BaseChunker
```python
class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""
    
    @abstractmethod
    def chunk(self, content: Union[str, bytes, Path], **kwargs) -> ChunkingResult:
        """Chunk content into pieces."""
        pass
```

#### Chunk Schema
```python
@dataclass
class Chunk:
    """Universal chunk representation."""
    id: str
    content: Union[str, bytes, Any]
    modality: ModalityType
    metadata: ChunkMetadata
```

## 🔧 Development Setup

### Environment Configuration

```bash
# Install development dependencies
pip install -e .[dev,all]

# Install pre-commit hooks
pre-commit install

# Configure git hooks
git config core.hooksPath .githooks
```

### IDE Configuration

#### VS Code
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm
1. Open project in PyCharm
2. Configure Python interpreter to use virtual environment
3. Enable pytest as test runner
4. Configure code style to use Black formatter

### Development Tools

```bash
# Code formatting
black chunking_strategy/ tests/ examples/

# Import sorting
isort chunking_strategy/ tests/ examples/

# Linting
ruff check chunking_strategy/ tests/

# Type checking
mypy chunking_strategy/

# Testing
pytest tests/ -v --cov=chunking_strategy

# Documentation
sphinx-build docs/ docs/_build/
```

## 📁 Code Organization

### Directory Structure

```
chunking_strategy/
├── __init__.py              # Main package exports
├── core/                    # Core framework
│   ├── __init__.py
│   ├── base.py             # Base classes and interfaces
│   ├── registry.py         # Strategy registry
│   ├── pipeline.py         # Processing pipelines
│   ├── streaming.py        # Streaming capabilities
│   ├── adaptive.py         # Adaptive chunking
│   ├── metrics.py          # Quality metrics
│   ├── hardware.py         # Hardware optimization
│   ├── embeddings.py       # Embedding generation
│   ├── extractors.py       # Content extraction
│   ├── universal_framework.py  # Universal processing
│   └── ...
├── strategies/             # Strategy implementations
│   ├── __init__.py
│   ├── text/              # Text-based strategies
│   │   ├── __init__.py
│   │   ├── sentence_based.py
│   │   ├── paragraph_based.py
│   │   ├── semantic_chunker.py
│   │   └── ...
│   ├── code/              # Code-aware strategies
│   │   ├── __init__.py
│   │   ├── python_chunker.py
│   │   ├── javascript_chunker.py
│   │   └── ...
│   ├── document/          # Document processing
│   │   ├── __init__.py
│   │   ├── pdf_chunker.py
│   │   ├── doc_chunker.py
│   │   └── ...
│   ├── multimedia/        # Audio/video/image
│   │   ├── __init__.py
│   │   ├── time_based_audio.py
│   │   ├── scene_based_video.py
│   │   └── ...
│   └── general/           # General-purpose
│       ├── __init__.py
│       ├── fixed_size.py
│       ├── adaptive_chunker.py
│       └── ...
├── detectors/             # Content analysis
│   ├── __init__.py
│   ├── file_type_detector.py
│   ├── content_analyzer.py
│   └── ...
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── postprocessing.py
│   └── ...
├── orchestrator.py        # High-level orchestration
├── cli.py                 # Command-line interface
├── exceptions.py          # Custom exceptions
└── logging_config.py      # Logging configuration
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Import Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from chunking_strategy.core.base import BaseChunker, Chunk
from chunking_strategy.core.registry import register_chunker
```

## ➕ Adding New Strategies

### Step 1: Create Strategy File

Create a new file in the appropriate strategy directory:

```python
# chunking_strategy/strategies/text/my_new_strategy.py

"""
My New Chunking Strategy

Description of what this strategy does and when to use it.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType
)
from chunking_strategy.core.registry import (
    register_chunker,
    ComplexityLevel,
    SpeedLevel,
    MemoryUsage
)

logger = logging.getLogger(__name__)


@register_chunker(
    name="my_new_strategy",
    category="text",
    description="Description of the strategy",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md"],
    complexity=ComplexityLevel.LOW,
    dependencies=[],
    optional_dependencies=["nltk"],
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    quality=0.7,
    parameters_schema={
        "param1": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "default": 10,
            "description": "Description of parameter"
        }
    },
    default_parameters={
        "param1": 10
    },
    use_cases=["RAG", "document processing"],
    best_for=["specific use cases"],
    limitations=["known limitations"],
    streaming_support=True,
    adaptive_support=False,
    hierarchical_support=False
)
class MyNewStrategy(BaseChunker):
    """
    My new chunking strategy implementation.
    
    This strategy does X, Y, and Z to achieve optimal chunking
    for specific use cases.
    
    Features:
    - Feature 1
    - Feature 2
    - Feature 3
    
    Examples:
        Basic usage:
        ```python
        chunker = MyNewStrategy(param1=20)
        result = chunker.chunk("content here")
        ```
    """
    
    def __init__(self, param1: int = 10, **kwargs):
        """
        Initialize the strategy.
        
        Args:
            param1: Description of parameter
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="my_new_strategy",
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )
        
        # Validate parameters
        if param1 <= 0:
            raise ValueError("param1 must be positive")
        
        self.param1 = param1
        
        self.logger.info(f"Initialized MyNewStrategy with param1={param1}")
    
    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk content using this strategy.
        
        Args:
            content: Content to chunk
            source_info: Information about the content source
            **kwargs: Additional chunking parameters
            
        Returns:
            ChunkingResult with generated chunks
        """
        import time
        start_time = time.time()
        
        # Handle different input types
        if isinstance(content, Path):
            with open(content, 'r', encoding='utf-8') as f:
                text_content = f.read()
            actual_source = str(content)
        elif isinstance(content, bytes):
            text_content = content.decode('utf-8')
            actual_source = source_info.get("source", "bytes_input") if source_info else "bytes_input"
        else:
            text_content = str(content)
            actual_source = source_info.get("source", "text_input") if source_info else "text_input"
        
        # Validate input
        self.validate_input(text_content, ModalityType.TEXT)
        
        # Implement chunking logic here
        chunks = self._implement_chunking_logic(text_content, actual_source)
        
        processing_time = time.time() - start_time
        
        # Create result
        result = ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used=self.name,
            source_info=source_info
        )
        
        self.logger.info(
            f"MyNewStrategy completed: {len(chunks)} chunks in {processing_time:.3f}s"
        )
        
        return result
    
    def _implement_chunking_logic(self, content: str, source: str) -> List[Chunk]:
        """
        Implement the actual chunking logic.
        
        Args:
            content: Text content to chunk
            source: Source identifier
            
        Returns:
            List of chunks
        """
        chunks = []
        
        # Your chunking implementation here
        # Example: simple fixed-size chunking
        chunk_size = self.param1 * 100  # Convert param to character count
        
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i + chunk_size]
            
            # Create metadata
            metadata = ChunkMetadata(
                source=source,
                offset=i,
                length=len(chunk_content),
                chunker_used=self.name
            )
            
            # Create chunk
            chunk = Chunk(
                id=f"my_new_strategy_{i // chunk_size}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        return chunks
```

### Step 2: Update Strategy Module

Add the new strategy to the appropriate `__init__.py`:

```python
# chunking_strategy/strategies/text/__init__.py

from .my_new_strategy import MyNewStrategy

__all__ = [
    # ... existing strategies
    "MyNewStrategy",
]
```

### Step 3: Add Tests

Create comprehensive tests:

```python
# tests/unit/strategies/test_my_new_strategy.py

import pytest
from chunking_strategy.strategies.text.my_new_strategy import MyNewStrategy
from chunking_strategy.core.base import ModalityType


class TestMyNewStrategy:
    """Test suite for MyNewStrategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        chunker = MyNewStrategy(param1=20)
        assert chunker.param1 == 20
        assert chunker.name == "my_new_strategy"
        assert chunker.category == "text"
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="param1 must be positive"):
            MyNewStrategy(param1=0)
        
        with pytest.raises(ValueError, match="param1 must be positive"):
            MyNewStrategy(param1=-1)
    
    def test_basic_chunking(self):
        """Test basic chunking functionality."""
        chunker = MyNewStrategy(param1=5)  # 500 character chunks
        content = "A" * 1000  # 1000 character content
        
        result = chunker.chunk(content)
        
        assert len(result.chunks) == 2  # 1000 / 500 = 2 chunks
        assert result.strategy_used == "my_new_strategy"
        assert result.processing_time > 0
        
        # Check chunk content
        assert result.chunks[0].content == "A" * 500
        assert result.chunks[1].content == "A" * 500
    
    def test_empty_content(self):
        """Test handling of empty content."""
        chunker = MyNewStrategy()
        result = chunker.chunk("")
        
        assert len(result.chunks) == 0
        assert result.strategy_used == "my_new_strategy"
    
    def test_file_input(self, tmp_path):
        """Test file input handling."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for file input")
        
        chunker = MyNewStrategy(param1=1)  # 100 character chunks
        result = chunker.chunk(test_file)
        
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "Test content for file input"
    
    def test_bytes_input(self):
        """Test bytes input handling."""
        chunker = MyNewStrategy(param1=1)
        content_bytes = "Test bytes content".encode('utf-8')
        
        result = chunker.chunk(content_bytes)
        
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "Test bytes content"
    
    def test_metadata_creation(self):
        """Test chunk metadata creation."""
        chunker = MyNewStrategy(param1=1)
        result = chunker.chunk("Test content")
        
        chunk = result.chunks[0]
        assert chunk.metadata.chunker_used == "my_new_strategy"
        assert chunk.metadata.offset == 0
        assert chunk.metadata.length == len("Test content")
        assert chunk.modality == ModalityType.TEXT
    
    def test_source_info_handling(self):
        """Test source info handling."""
        chunker = MyNewStrategy(param1=1)
        source_info = {"source": "test_source", "custom": "value"}
        
        result = chunker.chunk("Test content", source_info=source_info)
        
        assert result.source_info == source_info
        assert result.chunks[0].metadata.source == "test_source"
```

### Step 4: Add Documentation

Create documentation for the new strategy:

```markdown
# My New Strategy

## Overview

Description of what this strategy does and when to use it.

## Features

- Feature 1
- Feature 2
- Feature 3

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| param1 | int | 10 | Description of parameter |

## Examples

### Basic Usage

```python
from chunking_strategy import create_chunker

chunker = create_chunker("my_new_strategy", param1=20)
result = chunker.chunk("Your content here")
```

### Advanced Configuration

```python
chunker = create_chunker(
    "my_new_strategy",
    param1=15,
    # Additional parameters
)
result = chunker.chunk("content")
```

## Use Cases

- Use case 1
- Use case 2
- Use case 3

## Limitations

- Limitation 1
- Limitation 2
```

### Step 5: Update Examples

Add examples to the examples directory:

```python
# examples/my_new_strategy_demo.py

#!/usr/bin/env python3
"""
My New Strategy Demo

Demonstrates the capabilities of the MyNewStrategy chunker.
"""

from chunking_strategy import create_chunker


def demo_basic_usage():
    """Demonstrate basic usage of MyNewStrategy."""
    print("🎯 MyNewStrategy Basic Demo")
    print("=" * 40)
    
    chunker = create_chunker("my_new_strategy", param1=5)
    
    content = """
    This is a sample document that will be chunked using MyNewStrategy.
    The strategy will process this content and create meaningful chunks.
    Each chunk will contain relevant information for downstream processing.
    """
    
    result = chunker.chunk(content)
    
    print(f"📊 Generated {len(result.chunks)} chunks")
    print(f"⏱️  Processing time: {result.processing_time:.3f}s")
    
    for i, chunk in enumerate(result.chunks, 1):
        print(f"\n📄 Chunk {i}:")
        print(f"   Content: {chunk.content.strip()}")
        print(f"   Length: {len(chunk.content)} characters")


if __name__ == "__main__":
    demo_basic_usage()
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── core/               # Core component tests
│   ├── strategies/         # Strategy tests
│   ├── detectors/          # Detector tests
│   └── utils/              # Utility tests
├── integration/            # Integration tests
├── performance/            # Performance tests
└── fixtures/               # Test data and fixtures
```

### Test Categories

#### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Focus on specific functionality
- Fast execution (< 1 second per test)

#### Integration Tests
- Test component interactions
- Use real dependencies when possible
- Test end-to-end workflows
- Moderate execution time (< 10 seconds per test)

#### Performance Tests
- Benchmark critical paths
- Test with large datasets
- Monitor memory usage
- Longer execution time acceptable

### Test Naming

```python
# Good test names
def test_chunker_initialization_with_valid_parameters():
    """Test that chunker initializes correctly with valid parameters."""

def test_chunking_empty_content_returns_empty_result():
    """Test that chunking empty content returns empty result."""

def test_strategy_handles_unicode_content_correctly():
    """Test that strategy handles Unicode content correctly."""

# Bad test names
def test1():
def test_chunker():
def test_basic():
```

### Test Data

Use the `test_data/` directory for test files:

```
test_data/
├── short.txt              # Small text file
├── alice_wonderland.txt   # Medium text file
├── large_document.txt     # Large text file
├── example.pdf           # PDF file
├── sample.json           # JSON file
└── code_sample.py        # Python code file
```

### Fixtures

Create reusable fixtures in `conftest.py`:

```python
import pytest
from chunking_strategy import create_chunker


@pytest.fixture
def sample_text():
    """Sample text content for testing."""
    return """
    This is a sample text for testing chunking strategies.
    It contains multiple sentences and paragraphs.
    The content is designed to test various chunking scenarios.
    """


@pytest.fixture
def sample_chunker():
    """Sample chunker instance for testing."""
    return create_chunker("sentence_based", max_sentences=3)


@pytest.fixture
def temp_file(tmp_path):
    """Temporary file for testing."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Test content")
    return file_path
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/strategies/test_my_new_strategy.py

# Run with coverage
pytest --cov=chunking_strategy --cov-report=html

# Run performance tests
pytest tests/performance/ -m performance

# Run with verbose output
pytest -v

# Run specific test
pytest tests/unit/strategies/test_my_new_strategy.py::TestMyNewStrategy::test_basic_chunking
```

## ⚡ Performance Considerations

### Memory Management

- Use streaming for large files
- Implement lazy loading for heavy dependencies
- Monitor memory usage in tests
- Clean up resources properly

### Optimization Guidelines

1. **Profile Before Optimizing**
   ```python
   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Your code here
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative').print_stats(10)
   ```

2. **Use Appropriate Data Structures**
   - Lists for ordered collections
   - Sets for membership testing
   - Dicts for key-value mapping
   - Generators for memory efficiency

3. **Minimize String Operations**
   - Use string builders for concatenation
   - Avoid repeated string slicing
   - Use regex efficiently

4. **Lazy Loading**
   ```python
   # Good: Lazy loading
   def get_heavy_dependency():
       if not hasattr(self, '_heavy_dep'):
           self._heavy_dep = load_heavy_dependency()
       return self._heavy_dep
   
   # Bad: Eager loading
   def __init__(self):
       self._heavy_dep = load_heavy_dependency()  # Always loaded
   ```

### Benchmarking

Create performance benchmarks:

```python
# tests/performance/test_my_strategy_performance.py

import time
import pytest
from chunking_strategy import create_chunker


class TestMyStrategyPerformance:
    """Performance tests for MyNewStrategy."""
    
    @pytest.mark.performance
    def test_chunking_speed(self):
        """Test chunking speed with large content."""
        chunker = create_chunker("my_new_strategy")
        
        # Generate large content
        content = "Sample text " * 10000  # ~120KB
        
        start_time = time.time()
        result = chunker.chunk(content)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(content) / processing_time  # bytes/second
        
        # Assertions
        assert processing_time < 1.0  # Should complete in < 1 second
        assert throughput > 100000   # Should process > 100KB/s
        assert len(result.chunks) > 0
    
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage with large files."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        chunker = create_chunker("my_new_strategy")
        content = "Sample text " * 100000  # ~1.2MB
        
        result = chunker.chunk(content)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024  # < 50MB
```

## 📚 Documentation Standards

### Code Documentation

#### Docstrings
Use Google-style docstrings:

```python
def chunk_content(self, content: str, max_size: int = 1000) -> List[str]:
    """
    Chunk content into pieces of specified maximum size.
    
    This function takes text content and divides it into chunks
    that do not exceed the specified maximum size. It attempts
    to preserve word boundaries when possible.
    
    Args:
        content: The text content to chunk
        max_size: Maximum size of each chunk in characters
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If content is empty or max_size is invalid
        
    Example:
        >>> chunker = MyChunker()
        >>> chunks = chunker.chunk_content("Hello world", max_size=5)
        >>> print(chunks)
        ['Hello', 'world']
    """
```

#### Type Hints
Always use type hints:

```python
from typing import List, Dict, Optional, Union
from pathlib import Path

def process_file(
    file_path: Union[str, Path],
    options: Optional[Dict[str, Any]] = None
) -> List[Chunk]:
    """Process a file and return chunks."""
    pass
```

#### Comments
Use comments to explain complex logic:

```python
def complex_algorithm(self, data: List[int]) -> int:
    # Use binary search for O(log n) performance
    # This is critical for large datasets
    left, right = 0, len(data) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        # Check if we found the target
        if data[mid] == target:
            return mid
        
        # Adjust search range
        if data[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Not found
```

### API Documentation

Document all public APIs:

```python
class MyChunker(BaseChunker):
    """
    My custom chunking strategy.
    
    This chunker implements a novel approach to text chunking
    that balances semantic coherence with size constraints.
    
    Features:
    - Semantic boundary detection
    - Configurable chunk sizes
    - Overlap support
    - Streaming capabilities
    
    Example:
        >>> chunker = MyChunker(max_size=1000, overlap=100)
        >>> result = chunker.chunk("Your text here")
        >>> print(f"Created {len(result.chunks)} chunks")
    """
    
    def __init__(self, max_size: int = 1000, overlap: int = 0):
        """
        Initialize the chunker.
        
        Args:
            max_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
        """
        pass
```

### README Updates

When adding new features, update relevant documentation:

1. **README.md**: Add to feature list and examples
2. **API_REFERENCE.md**: Add API documentation
3. **CLI_REFERENCE.md**: Add CLI documentation if applicable
4. **Examples**: Add new example files

## 🚀 Release Process

### Version Management

Follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. **Code Quality**
   - [ ] All tests pass
   - [ ] Code coverage > 90%
   - [ ] No linting errors
   - [ ] Type checking passes

2. **Documentation**
   - [ ] README updated
   - [ ] API documentation updated
   - [ ] Examples updated
   - [ ] Changelog updated

3. **Testing**
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Performance tests pass
   - [ ] Manual testing completed

4. **Release**
   - [ ] Version bumped in `pyproject.toml`
   - [ ] Git tag created
   - [ ] PyPI package built
   - [ ] PyPI package uploaded

### Release Commands

```bash
# Update version
# Edit pyproject.toml to bump version

# Run tests
pytest tests/ -v --cov=chunking_strategy

# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*

# Create git tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

## 🤝 Contributing Guidelines

### Pull Request Process

1. **Fork and Branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Develop**
   - Write code following style guidelines
   - Add comprehensive tests
   - Update documentation
   - Ensure all tests pass

3. **Submit PR**
   - Clear description of changes
   - Reference related issues
   - Include test results
   - Request review from maintainers

### Code Review Process

1. **Automated Checks**
   - CI/CD pipeline runs tests
   - Code quality checks
   - Security scans

2. **Manual Review**
   - Code quality and style
   - Test coverage and quality
   - Documentation completeness
   - Performance implications

3. **Approval**
   - At least one maintainer approval
   - All checks passing
   - Documentation updated

### Issue Reporting

When reporting issues, include:

1. **Environment**
   - Python version
   - Operating system
   - Library version

2. **Reproduction**
   - Minimal code example
   - Expected vs actual behavior
   - Error messages/logs

3. **Context**
   - Use case description
   - Workaround if available
   - Priority/urgency

### Feature Requests

For feature requests, provide:

1. **Problem Statement**
   - What problem does this solve?
   - Why is it important?

2. **Proposed Solution**
   - How should it work?
   - API design suggestions

3. **Alternatives**
   - Other approaches considered
   - Workarounds available

## 🔍 Debugging and Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Test Failures
```bash
# Run with verbose output
pytest -v -s

# Run specific test with debugging
pytest tests/unit/test_specific.py::test_function -v -s --pdb
```

#### Performance Issues
```bash
# Profile with cProfile
python -m cProfile -s cumulative your_script.py

# Memory profiling
python -m memory_profiler your_script.py
```

### Debug Tools

#### Logging
```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Use in code
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

#### Debugging
```python
import pdb

# Set breakpoint
pdb.set_trace()

# Or use breakpoint() in Python 3.7+
breakpoint()
```

## 📞 Getting Help

### Resources

- **Documentation**: Check README.md and API_REFERENCE.md
- **Examples**: Browse examples/ directory
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions

### Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions
- **Code Review**: Help review others' contributions

---

**Happy Contributing! 🚀**

This guide should help you get started with contributing to the chunking-strategy library. If you have questions or suggestions for improving this guide, please open an issue or pull request.
