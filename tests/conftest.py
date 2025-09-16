"""
Pytest configuration and shared fixtures for the test suite.

This module provides common test fixtures, configuration, and utilities
used across all test modules.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

from chunking_strategy.utils.validation import ChunkValidator
from chunking_strategy.core.metrics import ChunkingQualityEvaluator


@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path, None, None]:
    """Provide the real test data directory path."""
    # Use the actual test_data directory from project root
    project_root = Path(__file__).parent.parent
    real_test_data_dir = project_root / "test_data"

    if real_test_data_dir.exists():
        yield real_test_data_dir
        return

    # Fallback: Create a temporary directory with test data files if real one doesn't exist
    temp_dir = Path(tempfile.mkdtemp())

    # Create test files
    test_files = {
        "simple.txt": "This is a simple text file for testing basic functionality.",

        "medium.txt": """This is a medium-sized text file for testing.
It contains multiple paragraphs and sentences.
Each paragraph tests different aspects of chunking.

The second paragraph continues the test content.
It helps verify that chunkers handle paragraph boundaries correctly.
This is important for text-based chunking strategies.

The third paragraph provides additional content.
It ensures we have enough text for meaningful chunking tests.
Testing with realistic content is essential for quality validation.""",

        "long.txt": "This is a test sentence for long content. " * 200,

        "unicode.txt": """Unicode test content with various characters.
Hello world! Héllo wörld! 你好世界! مرحبا بالعالم!
Emojis: 🌍🚀💻📚🎯
Mathematical symbols: α β γ δ ∑ ∏ ∫ ∞
Special characters: ñoñ-ASCII tëst with àccénts.""",

        "structured.md": """# Main Title

## Section 1: Introduction
This is the introduction section with some explanatory text.
It provides context for the document structure.

### Subsection 1.1
More detailed content in a subsection.
This tests hierarchical document processing.

## Section 2: Content
This section contains the main content.
It has multiple paragraphs for testing.

### Subsection 2.1: Details
Detailed information in this subsection.

### Subsection 2.2: Examples
Example content for testing purposes.

## Section 3: Conclusion
Final section with concluding remarks.
This completes the structured document test.""",

        "code.py": '''"""
Sample Python code for testing code-aware chunking.
"""

import os
import sys
from typing import List, Dict, Any

class TestClass:
    """A test class for code chunking."""

    def __init__(self, value: int):
        self.value = value

    def method_one(self) -> str:
        """First method."""
        return f"Value is {self.value}"

    def method_two(self, param: str) -> bool:
        """Second method with parameter."""
        return len(param) > self.value

def standalone_function(data: List[str]) -> Dict[str, Any]:
    """A standalone function for testing."""
    result = {}
    for item in data:
        result[item] = len(item)
    return result

if __name__ == "__main__":
    test = TestClass(5)
    print(test.method_one())
''',

        "empty.txt": "",

        "whitespace.txt": "   \n\t   \n   ",

        "single_line.txt": "This is a single line of text without any line breaks or paragraphs.",

        "html.html": """<!DOCTYPE html>
<html>
<head>
    <title>Test HTML Document</title>
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is a paragraph with <strong>bold text</strong> and <em>italic text</em>.</p>

    <h2>Subheading</h2>
    <p>Another paragraph with <a href="https://example.com">a link</a>.</p>

    <ul>
        <li>List item one</li>
        <li>List item two</li>
        <li>List item three</li>
    </ul>

    <blockquote>
        This is a blockquote for testing structured content.
    </blockquote>
</body>
</html>""",

        "json_data.json": """{
    "title": "Test JSON Document",
    "content": {
        "sections": [
            {
                "name": "Introduction",
                "text": "This is the introduction section."
            },
            {
                "name": "Content",
                "text": "This is the main content section."
            }
        ],
        "metadata": {
            "author": "Test Author",
            "created": "2024-01-01",
            "tags": ["test", "json", "chunking"]
        }
    }
}"""
    }

    # Write test files
    for filename, content in test_files.items():
        file_path = temp_dir / filename
        file_path.write_text(content, encoding='utf-8')

    # Create binary test file
    binary_file = temp_dir / "binary.dat"
    binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05' * 100)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def validator() -> ChunkValidator:
    """Provide a chunk validator instance."""
    return ChunkValidator()


@pytest.fixture
def quality_evaluator() -> ChunkingQualityEvaluator:
    """Provide a quality evaluator instance."""
    return ChunkingQualityEvaluator()


@pytest.fixture
def sample_texts() -> Dict[str, str]:
    """Provide sample text data for testing."""
    return {
        "simple": "This is a simple test text.",
        "medium": "This is a medium test text. It has multiple sentences. Each sentence provides test content.",
        "long": "This is a long test text. " * 50,
        "unicode": "Unicode test: héllo wörld! 你好世界 🌍",
        "empty": "",
        "whitespace": "   \n\t   ",
        "single_char": "A",
        "numbers": "123 456 789 101112 131415",
        "punctuation": "Hello, world! How are you? I'm fine. Thanks!",
        "mixed": "Mixed content: numbers 123, symbols @#$, and unicode 世界."
    }


@pytest.fixture
def chunker_configs() -> Dict[str, Dict[str, Any]]:
    """Provide common chunker configurations for testing."""
    return {
        "tiny": {"chunk_size": 5},
        "small": {"chunk_size": 20},
        "medium": {"chunk_size": 100},
        "large": {"chunk_size": 500},
        "with_overlap": {"chunk_size": 50, "overlap_size": 10},
        "word_based": {"chunk_size": 10, "unit": "word"},
        "byte_based": {"chunk_size": 50, "unit": "byte"},
        "preserve_boundaries": {"chunk_size": 30, "preserve_boundaries": True}
    }


@pytest.fixture(scope="session")
def temp_output_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test outputs."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


# Markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Add unit marker to tests that don't have other markers
        if not any(marker.name in ["integration", "performance", "slow"]
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Helper functions for tests
def assert_valid_chunks(chunks, validator=None):
    """Helper function to assert chunks are valid."""
    if validator is None:
        validator = ChunkValidator()

    assert len(chunks) > 0, "No chunks generated"

    for i, chunk in enumerate(chunks):
        chunk_issues = validator.validate_chunk(chunk)
        assert len(chunk_issues) == 0, f"Chunk {i} validation failed: {chunk_issues}"


def assert_content_integrity(original_content, chunks):
    """Helper function to assert content integrity is maintained."""
    reconstructed = ''.join(str(chunk.content) for chunk in chunks)
    assert reconstructed == original_content, "Content integrity lost during chunking"


def assert_reasonable_performance(processing_time, content_size, max_time_per_kb=0.1):
    """Helper function to assert reasonable performance."""
    size_kb = content_size / 1024
    max_time = max(0.001, size_kb * max_time_per_kb)  # At least 1ms
    assert processing_time <= max_time, f"Processing too slow: {processing_time}s for {size_kb:.1f}KB"
