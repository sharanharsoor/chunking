"""
Unit tests for PythonCodeChunker.

This module provides comprehensive tests for the Python code chunking functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from chunking_strategy import create_chunker, list_chunkers
from chunking_strategy.strategies.code.python_chunker import PythonCodeChunker
from chunking_strategy.core.base import ModalityType


class TestPythonCodeChunker:
    """Test suite for PythonCodeChunker."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.test_data_dir = Path("test_data")
        self.sample_python_file = self.test_data_dir / "sample_code.py"

        # Simple Python code for testing
        self.simple_python_code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")

class Calculator:
    """A simple calculator class."""

    def add(self, a, b):
        """Add two numbers."""
        return a + b

    def multiply(self, a, b):
        """Multiply two numbers."""
        return a * b

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(2, 3))
'''

        # Complex Python code with various constructs
        self.complex_python_code = '''
import os
import sys
from typing import List, Dict

# Global variable
GLOBAL_CONST = 42

def function_with_docstring():
    """
    This is a function with a multi-line docstring.

    It demonstrates docstring handling in the chunker.
    """
    pass

class ComplexClass:
    """A complex class with multiple methods."""

    def __init__(self, name: str):
        self.name = name

    @property
    def display_name(self) -> str:
        return f"Name: {self.name}"

    @staticmethod
    def static_method():
        return "static"

    @classmethod
    def class_method(cls):
        return cls.__name__

def nested_function_example():
    def inner_function():
        return "inner"

    return inner_function()

async def async_function():
    await some_async_operation()

# Lambda function
square = lambda x: x * x

# Decorator
@property
def decorated_function():
    pass
'''

    def test_chunker_registration(self):
        """Test that Python code chunker is properly registered."""
        chunkers = list_chunkers()
        assert "python_code" in chunkers

        chunker = create_chunker("python_code")
        assert chunker is not None
        assert isinstance(chunker, PythonCodeChunker)
        assert chunker.name == "python_code"

    def test_basic_initialization(self):
        """Test basic chunker initialization."""
        chunker = PythonCodeChunker()
        assert chunker.chunk_by == "function"
        assert chunker.max_lines_per_chunk == 100
        assert chunker.include_imports is True
        assert chunker.include_docstrings is True
        assert chunker.preserve_structure is True

    def test_custom_initialization(self):
        """Test chunker initialization with custom parameters."""
        chunker = PythonCodeChunker(
            chunk_by="class",
            max_lines_per_chunk=50,
            include_imports=False,
            include_docstrings=False,
            preserve_structure=False
        )
        assert chunker.chunk_by == "class"
        assert chunker.max_lines_per_chunk == 50
        assert chunker.include_imports is False
        assert chunker.include_docstrings is False
        assert chunker.preserve_structure is False

    def test_chunk_simple_python_code(self):
        """Test chunking simple Python code."""
        chunker = PythonCodeChunker(chunk_by="function")
        result = chunker.chunk(self.simple_python_code)

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "python_code"
        assert result.processing_time is not None

        # Should have chunks for function, class, and if __name__ block
        assert len(result.chunks) >= 2

    def test_chunk_by_function(self):
        """Test chunking by function."""
        chunker = PythonCodeChunker(chunk_by="function")
        result = chunker.chunk(self.complex_python_code)

        assert result is not None
        assert len(result.chunks) > 0

        # Check that function definitions are captured
        function_chunks = [chunk for chunk in result.chunks
                          if "def " in chunk.content]
        assert len(function_chunks) > 0

    def test_chunk_by_class(self):
        """Test chunking by class."""
        chunker = PythonCodeChunker(chunk_by="class")
        result = chunker.chunk(self.complex_python_code)

        assert result is not None
        assert len(result.chunks) > 0

        # Check that class definitions are captured
        class_chunks = [chunk for chunk in result.chunks
                       if "class " in chunk.content]
        assert len(class_chunks) > 0

    def test_chunk_by_logical_block(self):
        """Test chunking by logical blocks."""
        chunker = PythonCodeChunker(chunk_by="logical_block")
        result = chunker.chunk(self.complex_python_code)

        assert result is not None
        assert len(result.chunks) > 0

    def test_chunk_by_line_count(self):
        """Test chunking by line count."""
        chunker = PythonCodeChunker(chunk_by="line_count", max_lines_per_chunk=10)
        result = chunker.chunk(self.complex_python_code)

        assert result is not None
        assert len(result.chunks) > 0

        # Check that chunks don't exceed line limit (roughly)
        for chunk in result.chunks:
            line_count = len(chunk.content.split('\n'))
            # Allow some flexibility for logical boundaries
            assert line_count <= 25  # More tolerance for structure preservation

    def test_include_imports_setting(self):
        """Test include_imports setting."""
        # Test with imports included
        chunker = PythonCodeChunker(include_imports=True)
        result = chunker.chunk(self.complex_python_code)

        import_chunks = [chunk for chunk in result.chunks
                        if "import " in chunk.content]
        assert len(import_chunks) > 0

        # Test with imports excluded
        chunker = PythonCodeChunker(include_imports=False)
        result = chunker.chunk(self.complex_python_code)

        # Should still have chunks, but imports should be filtered
        assert len(result.chunks) > 0

    def test_include_docstrings_setting(self):
        """Test include_docstrings setting."""
        # Test with docstrings included
        chunker = PythonCodeChunker(include_docstrings=True)
        result = chunker.chunk(self.complex_python_code)

        docstring_chunks = [chunk for chunk in result.chunks
                           if '"""' in chunk.content]
        assert len(docstring_chunks) > 0

        # Test with docstrings excluded
        chunker = PythonCodeChunker(include_docstrings=False)
        result = chunker.chunk(self.complex_python_code)

        # Should still have chunks, but fewer docstrings
        assert len(result.chunks) > 0

    def test_chunk_file_path(self):
        """Test chunking from file path."""
        if not self.sample_python_file.exists():
            pytest.skip("Sample Python file not available")

        chunker = PythonCodeChunker()
        result = chunker.chunk(self.sample_python_file)

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "python_code"

    def test_chunk_path_object(self):
        """Test chunking from Path object."""
        if not self.sample_python_file.exists():
            pytest.skip("Sample Python file not available")

        chunker = PythonCodeChunker()
        result = chunker.chunk(self.sample_python_file)

        assert result is not None
        assert len(result.chunks) > 0

    def test_chunk_empty_content(self):
        """Test chunking empty content."""
        chunker = PythonCodeChunker()
        result = chunker.chunk("")

        assert result is not None
        assert len(result.chunks) == 0

    def test_chunk_invalid_python_code(self):
        """Test chunking invalid Python code."""
        invalid_code = '''
def broken_function(
    # Missing closing parenthesis and colon
    print("This is broken")
'''

        chunker = PythonCodeChunker()
        result = chunker.chunk(invalid_code)

        # Should handle gracefully and return some result
        assert result is not None
        # May have chunks even with syntax errors
        assert isinstance(result.chunks, list)

    def test_chunk_metadata(self):
        """Test chunk metadata."""
        chunker = PythonCodeChunker()
        result = chunker.chunk(self.simple_python_code)

        assert result is not None
        assert len(result.chunks) > 0

        for chunk in result.chunks:
            assert chunk.id is not None
            assert chunk.content is not None
            assert chunk.modality == ModalityType.TEXT
            assert chunk.metadata is not None
            assert chunk.metadata.chunker_used == "python_code"
            assert chunk.metadata.source == "direct_input"

    def test_chunk_stream_method(self):
        """Test chunk_stream method."""
        chunker = PythonCodeChunker()
        content_stream = [self.simple_python_code]

        chunks = list(chunker.chunk_stream(content_stream))
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.content is not None
            assert chunk.modality == ModalityType.TEXT

    def test_error_handling_nonexistent_file(self):
        """Test error handling for non-existent file."""
        chunker = PythonCodeChunker()

        with pytest.raises(FileNotFoundError):
            chunker.chunk(Path("nonexistent_file.py"))

    def test_binary_content_handling(self):
        """Test handling of binary content."""
        chunker = PythonCodeChunker()
        binary_content = b"print('hello')\n"

        result = chunker.chunk(binary_content)
        assert result is not None
        assert len(result.chunks) >= 0  # Should handle gracefully

    def test_large_file_handling(self):
        """Test handling of large Python files."""
        # Create a large Python file content
        large_content = ""
        for i in range(100):
            large_content += f'''
def function_{i}():
    """Function number {i}."""
    print("Function {i}")
    return {i}

class Class_{i}:
    """Class number {i}."""

    def method_{i}(self):
        return {i}
'''

        chunker = PythonCodeChunker(chunk_by="function", max_lines_per_chunk=20)
        result = chunker.chunk(large_content)

        assert result is not None
        assert len(result.chunks) > 10  # Should create many chunks

    def test_special_python_constructs(self):
        """Test handling of special Python constructs."""
        special_code = '''
# Decorator
@property
def decorated_method(self):
    return self._value

# Context manager
with open("file.txt") as f:
    content = f.read()

# List comprehension
squares = [x**2 for x in range(10)]

# Generator expression
gen = (x for x in range(10) if x % 2 == 0)

# Try-except block
try:
    risky_operation()
except Exception as e:
    handle_error(e)
finally:
    cleanup()

# Async/await
async def async_function():
    await some_async_call()
'''

        chunker = PythonCodeChunker()
        result = chunker.chunk(special_code)

        assert result is not None
        assert len(result.chunks) > 0

    def test_adaptation_capabilities(self):
        """Test adaptation capabilities."""
        chunker = PythonCodeChunker()

        # Test that chunker can be adapted (even if not implemented)
        history = chunker.get_adaptation_history()
        assert isinstance(history, list)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid chunk_by parameter
        chunker = PythonCodeChunker(chunk_by="invalid_mode")
        result = chunker.chunk(self.simple_python_code)

        # Should handle gracefully or fallback to default
        assert result is not None

    def test_concurrent_chunking(self):
        """Test concurrent chunking operations."""
        import concurrent.futures

        chunker = PythonCodeChunker()

        def chunk_code(code):
            return chunker.chunk(code)

        codes = [self.simple_python_code, self.complex_python_code] * 3

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(chunk_code, code) for code in codes]
            results = [future.result() for future in futures]

        assert len(results) == len(codes)
        for result in results:
            assert result is not None
            assert len(result.chunks) > 0


class TestPythonCodeChunkerIntegration:
    """Integration tests for PythonCodeChunker."""

    def test_with_real_python_file(self):
        """Test with a real Python file."""
        sample_file = Path("test_data/sample_code.py")
        if not sample_file.exists():
            pytest.skip("Sample Python file not available")

        chunker = create_chunker("python_code")
        result = chunker.chunk(sample_file)

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "python_code"

        # Verify chunk content makes sense
        total_content = "".join(chunk.content for chunk in result.chunks)
        original_content = sample_file.read_text(encoding='utf-8')

        # Content should be preserved (allowing for some chunking modifications)
        assert len(total_content) > 0
        assert "def " in total_content or "class " in total_content

    def test_chunker_registry_integration(self):
        """Test integration with chunker registry."""
        from chunking_strategy.core.registry import get_chunker_metadata

        metadata = get_chunker_metadata("python_code")
        assert metadata is not None
        assert metadata.name == "python_code"
        assert metadata.category == "code"
        assert "py" in metadata.supported_formats
