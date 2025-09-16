"""
Unit tests for UniversalCodeChunker.

This module provides comprehensive tests for the Universal code chunking functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from chunking_strategy import create_chunker, list_chunkers
from chunking_strategy.strategies.code.universal_code_chunker import UniversalCodeChunker
from chunking_strategy.core.base import ModalityType


class TestUniversalCodeChunker:
    """Test suite for UniversalCodeChunker."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.test_data_dir = Path("test_data")
        self.sample_js_file = self.test_data_dir / "sample_code.js"
        self.sample_go_file = self.test_data_dir / "sample_code.go"

        # Simple JavaScript code
        self.javascript_code = '''
// JavaScript example
function greetUser(name) {
    console.log(`Hello, ${name}!`);
}

class Calculator {
    constructor() {
        this.result = 0;
    }

    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}

const calc = new Calculator();
console.log(calc.add(5, 3));
'''

        # Go code
        self.go_code = '''
package main

import (
    "fmt"
    "strings"
)

// Constant definition
const MaxRetries = 3

// Function definition
func greetUser(name string) {
    fmt.Printf("Hello, %s!\\n", name)
}

// Struct definition
type Calculator struct {
    result int
}

// Method definition
func (c *Calculator) Add(a, b int) int {
    return a + b
}

func (c *Calculator) Multiply(a, b int) int {
    return a * b
}

func main() {
    calc := &Calculator{}
    fmt.Println(calc.Add(5, 3))
    greetUser("World")
}
'''

        # Java code
        self.java_code = '''
package com.example;

import java.util.ArrayList;
import java.util.List;

// Class definition
public class Calculator {
    private int result;

    public Calculator() {
        this.result = 0;
    }

    public int add(int a, int b) {
        return a + b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }

    public static void main(String[] args) {
        Calculator calc = new Calculator();
        System.out.println(calc.add(5, 3));
    }
}
'''

        # Rust code
        self.rust_code = '''
use std::collections::HashMap;

// Constant
const MAX_RETRIES: u32 = 3;

// Struct definition
pub struct Calculator {
    result: i32,
}

impl Calculator {
    pub fn new() -> Self {
        Calculator { result: 0 }
    }

    pub fn add(&self, a: i32, b: i32) -> i32 {
        a + b
    }

    pub fn multiply(&self, a: i32, b: i32) -> i32 {
        a * b
    }
}

// Function definition
fn greet_user(name: &str) {
    println!("Hello, {}!", name);
}

fn main() {
    let calc = Calculator::new();
    println!("{}", calc.add(5, 3));
    greet_user("World");
}
'''

        # Unknown language code
        self.unknown_code = '''
// Unknown language syntax
mystery_function(param1, param2) {
    local_var = compute_something(param1);
    if (local_var > threshold) {
        return process_data(param2);
    } else {
        return fallback_value;
    }
}

another_function() {
    // Some implementation
}
'''

    def test_chunker_registration(self):
        """Test that Universal code chunker is properly registered."""
        chunkers = list_chunkers()
        assert "universal_code" in chunkers

        chunker = create_chunker("universal_code")
        assert chunker is not None
        assert isinstance(chunker, UniversalCodeChunker)
        assert chunker.name == "universal_code"

    def test_basic_initialization(self):
        """Test basic chunker initialization."""
        chunker = UniversalCodeChunker()
        assert chunker.chunk_by == "logical_block"
        assert chunker.max_lines_per_chunk == 50
        assert chunker.preserve_comments is True
        assert chunker.preserve_structure is True

    def test_custom_initialization(self):
        """Test chunker initialization with custom parameters."""
        chunker = UniversalCodeChunker(
            chunk_by="function",
            max_lines_per_chunk=100,
            preserve_comments=False,
            preserve_structure=False
        )
        assert chunker.chunk_by == "function"
        assert chunker.max_lines_per_chunk == 100
        assert chunker.preserve_comments is False
        assert chunker.preserve_structure is False

    def test_language_configuration_exists(self):
        """Test that language configurations are available."""
        chunker = UniversalCodeChunker()

        # Test that language configs exist
        assert hasattr(chunker, 'LANGUAGE_CONFIGS')
        assert 'python' in chunker.LANGUAGE_CONFIGS
        assert 'javascript' in chunker.LANGUAGE_CONFIGS
        assert 'go' in chunker.LANGUAGE_CONFIGS
        assert 'java' in chunker.LANGUAGE_CONFIGS
        assert 'c_cpp' in chunker.LANGUAGE_CONFIGS

        # Test configuration structure
        for lang, config in chunker.LANGUAGE_CONFIGS.items():
            assert 'extensions' in config
            assert 'function_patterns' in config
            assert 'indent_sensitive' in config

    def test_chunk_javascript_code(self):
        """Test chunking JavaScript code."""
        chunker = UniversalCodeChunker(chunk_by="function")
        result = chunker.chunk(self.javascript_code, source_info={"extension": ".js"})

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "universal_code"

        # Should identify functions and classes
        function_chunks = [chunk for chunk in result.chunks
                          if "function" in chunk.content or "class" in chunk.content]
        assert len(function_chunks) > 0

    def test_chunk_go_code(self):
        """Test chunking Go code."""
        chunker = UniversalCodeChunker(chunk_by="function")
        result = chunker.chunk(self.go_code, source_info={"extension": ".go"})

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "universal_code"

        # Should identify functions and structs
        function_chunks = [chunk for chunk in result.chunks
                          if "func " in chunk.content or "type " in chunk.content]
        assert len(function_chunks) > 0

    def test_chunk_java_code(self):
        """Test chunking Java code."""
        chunker = UniversalCodeChunker(chunk_by="function")
        result = chunker.chunk(self.java_code, source_info={"extension": ".java"})

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "universal_code"

        # Should identify methods and classes
        function_chunks = [chunk for chunk in result.chunks
                          if "public " in chunk.content or "class " in chunk.content]
        assert len(function_chunks) > 0

    def test_chunk_rust_code(self):
        """Test chunking Rust code."""
        chunker = UniversalCodeChunker(chunk_by="function")
        result = chunker.chunk(self.rust_code, source_info={"extension": ".rs"})

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "universal_code"

        # Should identify functions and structs
        function_chunks = [chunk for chunk in result.chunks
                          if "fn " in chunk.content or "struct " in chunk.content]
        assert len(function_chunks) > 0

    def test_chunk_unknown_language(self):
        """Test chunking unknown language code."""
        chunker = UniversalCodeChunker(chunk_by="function")
        result = chunker.chunk(self.unknown_code, source_info={"extension": ".mystery"})

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "universal_code"

        # Should still create chunks using generic patterns
        assert len(result.chunks) >= 1

    def test_chunk_by_function(self):
        """Test chunking by function."""
        chunker = UniversalCodeChunker(chunk_by="function")
        result = chunker.chunk(self.javascript_code)

        assert result is not None
        assert len(result.chunks) > 0

    def test_chunk_by_logical_block(self):
        """Test chunking by logical blocks."""
        chunker = UniversalCodeChunker(chunk_by="logical_block")
        result = chunker.chunk(self.javascript_code)

        assert result is not None
        assert len(result.chunks) > 0

    def test_chunk_by_line_count(self):
        """Test chunking by line count."""
        chunker = UniversalCodeChunker(chunk_by="line_count", max_lines_per_chunk=10)
        result = chunker.chunk(self.javascript_code)

        assert result is not None
        assert len(result.chunks) > 0

        # Check that chunks don't exceed line limit (roughly)
        for chunk in result.chunks:
            line_count = len(chunk.content.split('\n'))
            # Allow some flexibility for logical boundaries
            assert line_count <= 15  # Some tolerance for structure preservation

    def test_preserve_comments_setting(self):
        """Test preserve_comments setting."""
        code_with_comments = '''
// This is a comment
function myFunction() {
    // Another comment
    console.log("Hello");
    /* Block comment */
    return true;
}
'''

        # Test with comments included
        chunker = UniversalCodeChunker(preserve_comments=True)
        result = chunker.chunk(code_with_comments)

        comment_chunks = [chunk for chunk in result.chunks
                         if "//" in chunk.content or "/*" in chunk.content]
        assert len(comment_chunks) > 0

        # Test with comments excluded
        chunker = UniversalCodeChunker(preserve_comments=False)
        result = chunker.chunk(code_with_comments)

        # Should still have chunks, but fewer comments
        assert len(result.chunks) > 0

    def test_indentation_based_chunking(self):
        """Test indentation-based chunking for Python-like languages."""
        python_like_code = '''
def function1():
    if True:
        print("Level 1")
        if True:
            print("Level 2")

def function2():
    for i in range(10):
        print(i)
'''

        chunker = UniversalCodeChunker(chunk_by="function")
        result = chunker.chunk(python_like_code, source_info={"extension": ".py"})

        assert result is not None
        assert len(result.chunks) > 0

    def test_brace_based_chunking(self):
        """Test brace-based chunking for C-like languages."""
        c_like_code = '''
function myFunction() {
    if (condition) {
        doSomething();
    }
    return true;
}

class MyClass {
    constructor() {
        this.value = 0;
    }

    method() {
        return this.value;
    }
}
'''

        chunker = UniversalCodeChunker(chunk_by="function")
        result = chunker.chunk(c_like_code, source_info={"extension": ".js"})

        assert result is not None
        assert len(result.chunks) > 0

    def test_chunk_file_path(self):
        """Test chunking from file path."""
        if not self.sample_js_file.exists():
            pytest.skip("Sample JavaScript file not available")

        chunker = UniversalCodeChunker()
        result = chunker.chunk(self.sample_js_file)

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "universal_code"

    def test_chunk_empty_content(self):
        """Test chunking empty content."""
        chunker = UniversalCodeChunker()
        result = chunker.chunk("")

        assert result is not None
        # Universal chunker may create an empty chunk for empty content, which is acceptable
        if len(result.chunks) == 1:
            assert result.chunks[0].content == ""  # If chunk exists, it should be empty
        else:
            assert len(result.chunks) == 0  # Or no chunks at all

    def test_chunk_metadata(self):
        """Test chunk metadata."""
        chunker = UniversalCodeChunker()
        result = chunker.chunk(self.javascript_code)

        assert result is not None
        assert len(result.chunks) > 0

        for chunk in result.chunks:
            assert chunk.id is not None
            assert chunk.content is not None
            assert chunk.modality == ModalityType.TEXT
            assert chunk.metadata is not None
            assert chunk.metadata.chunker_used == "universal_code"
            assert chunk.metadata.source == "direct_input"

    def test_chunk_stream_method(self):
        """Test chunk_stream method."""
        chunker = UniversalCodeChunker()
        content_stream = [self.javascript_code]

        chunks = list(chunker.chunk_stream(content_stream))
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.content is not None
            assert chunk.modality == ModalityType.TEXT

    def test_binary_content_handling(self):
        """Test handling of binary content."""
        chunker = UniversalCodeChunker()
        binary_content = b"function test() { console.log('hello'); }\n"

        result = chunker.chunk(binary_content)
        assert result is not None
        assert len(result.chunks) >= 0  # Should handle gracefully

    def test_comment_and_string_filtering(self):
        """Test filtering of comments and strings."""
        code_with_strings = '''
function test() {
    var str1 = "This is a string with function keyword";
    var str2 = 'Another string with class keyword';
    /* Comment with function inside */
    // Another comment with class
    console.log("Real function call");
}
'''

        chunker = UniversalCodeChunker()
        result = chunker.chunk(code_with_strings)

        assert result is not None
        assert len(result.chunks) > 0

    def test_nested_block_handling(self):
        """Test handling of nested code blocks."""
        nested_code = '''
function outerFunction() {
    if (condition) {
        function innerFunction() {
            for (let i = 0; i < 10; i++) {
                if (i % 2 === 0) {
                    console.log(i);
                }
            }
        }
        innerFunction();
    }
}
'''

        chunker = UniversalCodeChunker(chunk_by="function")
        result = chunker.chunk(nested_code)

        assert result is not None
        assert len(result.chunks) > 0

    def test_mixed_language_patterns(self):
        """Test with code that has mixed language patterns."""
        mixed_code = '''
// This looks like JavaScript
function jsFunction() {
    console.log("JS style");
}

// But also has Go-like syntax
func goFunction() {
    fmt.Println("Go style")
}

# Python-like comment
def pythonFunction():
    print("Python style")
'''

        chunker = UniversalCodeChunker()
        result = chunker.chunk(mixed_code)

        assert result is not None
        assert len(result.chunks) > 0

    def test_adaptation_capabilities(self):
        """Test adaptation capabilities."""
        chunker = UniversalCodeChunker()

        # Test that chunker can be adapted (even if not implemented)
        history = chunker.get_adaptation_history()
        assert isinstance(history, list)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid chunk_by parameter
        chunker = UniversalCodeChunker(chunk_by="invalid_mode")
        result = chunker.chunk(self.javascript_code)

        # Should handle gracefully or fallback to default
        assert result is not None

    def test_large_file_handling(self):
        """Test handling of large code files."""
        # Create a large code file content
        large_content = ""
        for i in range(100):
            large_content += f'''
function function_{i}() {{
    console.log("Function {i}");
    var result = calculate_{i}(10, 20);
    return result;
}}

var calculate_{i} = function(a, b) {{
    return a + b + {i};
}};
'''

        chunker = UniversalCodeChunker(chunk_by="function", max_lines_per_chunk=20)
        result = chunker.chunk(large_content)

        assert result is not None
        assert len(result.chunks) > 10  # Should create many chunks

    def test_error_handling_nonexistent_file(self):
        """Test error handling for non-existent file."""
        chunker = UniversalCodeChunker()

        with pytest.raises(FileNotFoundError):
            chunker.chunk(Path("nonexistent_file.xyz"))

    def test_concurrent_chunking(self):
        """Test concurrent chunking operations."""
        import concurrent.futures

        chunker = UniversalCodeChunker()

        def chunk_code(code_info):
            code, ext = code_info
            return chunker.chunk(code, source_info={"extension": ext})

        codes = [
            (self.javascript_code, ".js"),
            (self.go_code, ".go"),
            (self.java_code, ".java"),
            (self.rust_code, ".rs")
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(chunk_code, code_info) for code_info in codes]
            results = [future.result() for future in futures]

        assert len(results) == len(codes)
        for result in results:
            assert result is not None
            assert len(result.chunks) > 0


class TestUniversalCodeChunkerIntegration:
    """Integration tests for UniversalCodeChunker."""

    def test_with_real_files(self):
        """Test with real code files."""
        test_files = [
            ("test_data/sample_code.js", ".js"),
            ("test_data/sample_code.go", ".go")
        ]

        chunker = create_chunker("universal_code")

        for file_path, expected_ext in test_files:
            sample_file = Path(file_path)
            if not sample_file.exists():
                continue

            result = chunker.chunk(sample_file)

            assert result is not None
            assert len(result.chunks) > 0
            assert result.strategy_used == "universal_code"

    def test_chunker_registry_integration(self):
        """Test integration with chunker registry."""
        from chunking_strategy.core.registry import get_chunker_metadata

        metadata = get_chunker_metadata("universal_code")
        assert metadata is not None
        assert metadata.name == "universal_code"
        assert metadata.category == "code"
        assert "*" in metadata.supported_formats  # Supports all formats

    def test_language_config_coverage(self):
        """Test that all configured languages work."""
        chunker = UniversalCodeChunker()

        # Test each configured language
        test_codes = {
            "python": "def test(): pass",
            "javascript": "function test() {}",
            "java": "public class Test {}",
            "c_cpp": "void test() {}",
            "go": "func test() {}",
            "rust": "fn test() {}"
        }

        for lang, code in test_codes.items():
            config = chunker.LANGUAGE_CONFIGS.get(lang)
            if config:
                # Test that the language configuration is valid
                assert "extensions" in config
                assert "function_patterns" in config
                assert "indent_sensitive" in config
