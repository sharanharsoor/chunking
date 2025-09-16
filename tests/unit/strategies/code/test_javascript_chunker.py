"""
Unit tests for JavaScriptChunker.

Tests the JavaScript/TypeScript code chunking functionality including:
- Function detection (declarations, expressions, arrow functions)
- Class definitions and methods
- Import/export statements
- JSX components
- TypeScript interfaces and types
- Various chunking strategies
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from chunking_strategy.core.registry import create_chunker
from chunking_strategy.strategies.code.javascript_chunker import JavaScriptChunker
from chunking_strategy.core.base import ChunkingResult, Chunk


class TestJavaScriptChunkerRegistration:
    """Test JavaScript chunker registration and creation."""

    def test_chunker_is_registered(self):
        """Test that JavaScript chunker is properly registered."""
        chunker = create_chunker("javascript_code")
        if chunker is None:
            pytest.skip("JavaScript chunker not available")

        assert chunker is not None
        assert isinstance(chunker, JavaScriptChunker)
        assert chunker.name == "javascript_code"
        assert chunker.category == "code"

    def test_create_chunker_factory(self):
        """Test chunker creation through factory."""
        chunker = create_chunker("javascript_code", chunk_by="function")
        if chunker is None:
            pytest.skip("JavaScript chunker not available")

        assert chunker is not None
        assert chunker.chunk_by == "function"


class TestJavaScriptChunkerInitialization:
    """Test JavaScript chunker initialization and configuration."""

    def test_default_initialization(self):
        """Test chunker with default parameters."""
        chunker = JavaScriptChunker()

        assert chunker.chunk_by == "function"
        assert chunker.max_lines_per_chunk == 100
        assert chunker.include_imports is True
        assert chunker.include_exports is True
        assert chunker.handle_jsx is True
        assert chunker.handle_typescript is True
        assert chunker.preserve_comments is True

    def test_custom_initialization(self):
        """Test chunker with custom parameters."""
        chunker = JavaScriptChunker(
            chunk_by="class",
            max_lines_per_chunk=50,
            include_imports=False,
            handle_jsx=False
        )

        assert chunker.chunk_by == "class"
        assert chunker.max_lines_per_chunk == 50
        assert chunker.include_imports is False
        assert chunker.handle_jsx is False


class TestJavaScriptCodeParsing:
    """Test JavaScript code parsing and element extraction."""

    def test_function_declaration_detection(self):
        """Test detection of function declarations."""
        chunker = JavaScriptChunker(chunk_by="function")
        code = """
function calculateSum(a, b) {
    return a + b;
}

function processData(data) {
    return data.map(item => item * 2);
}
"""
        result = chunker.chunk(code)

        assert result.total_chunks == 2
        assert all(chunk.metadata.extra["element_type"] == "function_declaration" for chunk in result.chunks)

        # Check function names
        function_names = [chunk.metadata.extra["element_name"] for chunk in result.chunks]
        assert "calculateSum" in function_names
        assert "processData" in function_names

    def test_arrow_function_detection(self):
        """Test detection of arrow functions."""
        chunker = JavaScriptChunker(chunk_by="function")
        code = """
const multiply = (a, b) => {
    return a * b;
};

const greet = name => `Hello, ${name}!`;

const complex = async (data) => {
    const result = await processAsync(data);
    return result;
};
"""
        result = chunker.chunk(code)

        assert result.total_chunks >= 2  # At least the multi-line functions

        # Check for arrow functions
        arrow_functions = [chunk for chunk in result.chunks
                          if chunk.metadata.extra["element_type"] == "arrow_function"]
        assert len(arrow_functions) >= 1

    def test_class_detection(self):
        """Test detection of class definitions."""
        chunker = JavaScriptChunker(chunk_by="class")
        code = """
class Calculator {
    constructor() {
        this.result = 0;
    }

    add(number) {
        this.result += number;
        return this;
    }

    getResult() {
        return this.result;
    }
}

export class AdvancedCalculator extends Calculator {
    multiply(number) {
        this.result *= number;
        return this;
    }
}
"""
        result = chunker.chunk(code)

        # Should have at least the class chunks
        class_chunks = [chunk for chunk in result.chunks
                       if chunk.metadata.extra["element_type"] == "class"]
        assert len(class_chunks) >= 1

        # Check class names and methods
        for chunk in class_chunks:
            if chunk.metadata.extra["element_name"] == "Calculator":
                assert "add" in chunk.metadata.extra.get("methods", [])
                assert "getResult" in chunk.metadata.extra.get("methods", [])

    def test_import_export_detection(self):
        """Test detection of import/export statements."""
        chunker = JavaScriptChunker(chunk_by="function", include_imports=True, include_exports=True)
        code = """
import React, { useState, useEffect } from 'react';
import { debounce } from 'lodash';

export const API_BASE_URL = 'https://api.example.com';

function myFunction() {
    return 'test';
}

export default myFunction;
export { useState as useReactState };
"""
        result = chunker.chunk(code)

        # Should have import and export chunks
        import_chunks = [chunk for chunk in result.chunks
                        if chunk.metadata.extra["element_type"] == "import"]
        export_chunks = [chunk for chunk in result.chunks
                        if chunk.metadata.extra["element_type"] == "export"]

        assert len(import_chunks) >= 1
        assert len(export_chunks) >= 1


class TestTypeScriptFeatures:
    """Test TypeScript-specific features."""

    def test_typescript_detection(self):
        """Test TypeScript file detection."""
        chunker = JavaScriptChunker()

        # Test with .ts file
        ts_code = """
interface User {
    id: string;
    name: string;
}

class UserService {
    getUser(id: string): User | null {
        return null;
    }
}
"""
        result = chunker.chunk(ts_code, source_info={"file_extension": ".ts"})
        assert result.source_info["is_typescript"] is True

    def test_interface_detection(self):
        """Test TypeScript interface detection."""
        chunker = JavaScriptChunker(chunk_by="function", handle_typescript=True)
        code = """
interface UserData {
    id: string;
    name: string;
    email: string;
}

interface ApiResponse<T> {
    data: T;
    status: 'success' | 'error';
}

type UserRole = 'admin' | 'user';
"""
        result = chunker.chunk(code)

        # Should detect interfaces and types
        interface_chunks = [chunk for chunk in result.chunks
                           if chunk.metadata.extra["element_type"] in ["interface", "type_alias"]]
        assert len(interface_chunks) >= 2


class TestJSXFeatures:
    """Test JSX/React-specific features."""

    def test_jsx_detection(self):
        """Test JSX file detection."""
        chunker = JavaScriptChunker()

        jsx_code = """
import React from 'react';

function MyComponent() {
    return <div>Hello World</div>;
}

const AnotherComponent = () => {
    return (
        <div>
            <h1>Title</h1>
            <p>Content</p>
        </div>
    );
};
"""
        result = chunker.chunk(jsx_code, source_info={"file_extension": ".jsx"})
        assert result.source_info["is_jsx"] is True

    def test_jsx_component_detection(self):
        """Test JSX component detection."""
        chunker = JavaScriptChunker(chunk_by="component", handle_jsx=True)
        code = """
const UserCard = ({ user }) => {
    return (
        <div className="user-card">
            <h3>{user.name}</h3>
            <p>{user.email}</p>
        </div>
    );
};

function UserProfile() {
    const [user, setUser] = useState(null);

    return (
        <div>
            <UserCard user={user} />
        </div>
    );
}
"""
        result = chunker.chunk(code)

        # Should detect JSX components
        jsx_chunks = [chunk for chunk in result.chunks
                     if chunk.metadata.extra.get("is_jsx_component")]
        assert len(jsx_chunks) >= 1


class TestChunkingStrategies:
    """Test different chunking strategies."""

    def test_function_chunking(self):
        """Test function-based chunking strategy."""
        chunker = JavaScriptChunker(chunk_by="function")
        code = """
function func1() { return 1; }
function func2() { return 2; }
const func3 = () => 3;
"""
        result = chunker.chunk(code)

        assert result.total_chunks >= 2
        assert result.strategy_used == "javascript_code"

    def test_class_chunking(self):
        """Test class-based chunking strategy."""
        chunker = JavaScriptChunker(chunk_by="class")
        code = """
class Class1 {
    method1() {}
}

class Class2 {
    method2() {}
}

function standalone() {}
"""
        result = chunker.chunk(code)

        # Should have chunks for classes
        class_chunks = [chunk for chunk in result.chunks
                       if chunk.metadata.extra["element_type"] == "class"]
        assert len(class_chunks) >= 1

    def test_logical_block_chunking(self):
        """Test logical block chunking strategy."""
        chunker = JavaScriptChunker(chunk_by="logical_block", max_lines_per_chunk=10)
        code = "\n".join([f"function func{i}() {{ return {i}; }}" for i in range(20)])

        result = chunker.chunk(code)

        # Should create multiple chunks due to line limit
        assert result.total_chunks > 1

        # Check that chunks have reasonable sizes
        for chunk in result.chunks:
            lines = len(chunk.content.split('\n'))
            assert lines <= chunker.max_lines_per_chunk + 5  # Some tolerance


class TestInputTypes:
    """Test different input types and formats."""

    def test_string_content(self):
        """Test chunking string content."""
        chunker = JavaScriptChunker()
        code = "function test() { return 'hello'; }"

        result = chunker.chunk(code)
        assert isinstance(result, ChunkingResult)
        assert result.total_chunks >= 1

    def test_file_path_input(self):
        """Test chunking from file path."""
        chunker = JavaScriptChunker()

        # Test with actual test file
        result = chunker.chunk("test_data/sample_code.js")
        assert isinstance(result, ChunkingResult)
        assert result.total_chunks >= 1

    def test_empty_content(self):
        """Test chunking empty content."""
        chunker = JavaScriptChunker()

        result = chunker.chunk("")
        assert result.total_chunks == 0

    def test_whitespace_only(self):
        """Test chunking whitespace-only content."""
        chunker = JavaScriptChunker()

        result = chunker.chunk("   \n\n   \t  ")
        assert result.total_chunks == 0


class TestChunkMetadata:
    """Test chunk metadata extraction."""

    def test_chunk_metadata_structure(self):
        """Test that chunks have proper metadata structure."""
        chunker = JavaScriptChunker()
        code = """
function testFunction(param1, param2) {
    return param1 + param2;
}
"""
        result = chunker.chunk(code)

        assert result.total_chunks >= 1
        chunk = result.chunks[0]

        # Check basic metadata
        assert hasattr(chunk, 'metadata')
        assert chunk.metadata.chunker_used == "javascript_code"
        assert "element_type" in chunk.metadata.extra
        assert "language" in chunk.metadata.extra
        assert "element_name" in chunk.metadata.extra

    def test_function_metadata(self):
        """Test function-specific metadata."""
        chunker = JavaScriptChunker(chunk_by="function")
        code = """
async function processData(data, options = {}) {
    const result = await transform(data);
    return result;
}
"""
        result = chunker.chunk(code)

        function_chunk = result.chunks[0]
        metadata = function_chunk.metadata.extra

        assert metadata["element_type"] == "function_declaration"
        assert metadata["element_name"] == "processData"
        assert metadata["is_async"] is True
        assert "data" in metadata.get("params", [])
        assert "options" in metadata.get("params", [])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_malformed_javascript(self):
        """Test handling of malformed JavaScript."""
        chunker = JavaScriptChunker()
        code = """
function incomplete( {
    return "this is broken
}

// This should still work
function working() {
    return "ok";
}
"""
        result = chunker.chunk(code)

        # Should handle errors gracefully and still process what it can
        assert isinstance(result, ChunkingResult)
        # Might fall back to line-based chunking
        assert result.total_chunks >= 1

    def test_very_large_function(self):
        """Test handling of very large functions."""
        chunker = JavaScriptChunker(max_lines_per_chunk=50)

        # Create a large function
        large_code = "function largeFunction() {\n"
        large_code += "\n".join([f"    const var{i} = {i};" for i in range(100)])
        large_code += "\n    return 'done';\n}"

        result = chunker.chunk(large_code)
        assert result.total_chunks >= 1

    def test_nested_functions(self):
        """Test handling of nested functions."""
        chunker = JavaScriptChunker(chunk_by="function")
        code = """
function outer() {
    function inner() {
        return 'inner';
    }

    const arrowInner = () => {
        return 'arrow inner';
    };

    return inner() + arrowInner();
}
"""
        result = chunker.chunk(code)

        # Should detect the outer function
        assert result.total_chunks >= 1
        assert any(chunk.metadata.extra["element_name"] == "outer" for chunk in result.chunks)


class TestStreamingAndAdaptation:
    """Test streaming and adaptive features."""

    def test_streaming_support(self):
        """Test streaming chunking."""
        chunker = JavaScriptChunker()

        # Simulate stream
        def content_stream():
            yield "function test1() {\n"
            yield "    return 1;\n"
            yield "}\n\n"
            yield "function test2() {\n"
            yield "    return 2;\n"
            yield "}"

        chunks = list(chunker.chunk_stream(content_stream()))
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_adaptation_history(self):
        """Test adaptation history tracking."""
        chunker = JavaScriptChunker()

        # Test parameter adaptation
        chunker.adapt_parameters(0.3, "quality")

        history = chunker.get_adaptation_history()
        assert len(history) >= 1
        assert history[0]["feedback_score"] == 0.3
        assert history[0]["feedback_type"] == "quality"

    def test_chunk_estimation(self):
        """Test chunk count estimation."""
        chunker = JavaScriptChunker(chunk_by="function")

        code = "function test() { return 1; }"
        estimate = chunker.estimate_chunks(code)

        assert isinstance(estimate, int)
        assert estimate >= 1


class TestOrchestrator:
    """Test orchestrator integration."""

    def test_orchestrator_auto_selection_js(self):
        """Test that orchestrator selects JavaScript chunker for .js files."""
        from chunking_strategy import ChunkerOrchestrator
        from chunking_strategy.orchestrator import STRATEGY_NAME_MAPPING

        config = {'strategies': {'primary': 'auto'}}
        orchestrator = ChunkerOrchestrator(config=config)

        file_info = {'file_extension': '.js', 'file_size': 5000}
        primary, fallbacks = orchestrator._auto_select_strategy(file_info)

        assert primary == 'javascript'
        assert STRATEGY_NAME_MAPPING.get('javascript') == 'javascript_code'

    def test_orchestrator_auto_selection_ts(self):
        """Test that orchestrator selects JavaScript chunker for .ts files."""
        from chunking_strategy import ChunkerOrchestrator

        config = {'strategies': {'primary': 'auto'}}
        orchestrator = ChunkerOrchestrator(config=config)

        file_info = {'file_extension': '.ts', 'file_size': 5000}
        primary, fallbacks = orchestrator._auto_select_strategy(file_info)

        assert primary == 'typescript'

    def test_orchestrator_file_chunking(self):
        """Test chunking actual files through orchestrator."""
        from chunking_strategy import ChunkerOrchestrator

        config = {'strategies': {'primary': 'auto'}}
        orchestrator = ChunkerOrchestrator(config=config)

        # Test with sample JavaScript file
        try:
            result = orchestrator.chunk_file("test_data/sample_code.js")
            assert result.strategy_used in ["javascript_code", "javascript_code_fallback"]
            assert result.total_chunks >= 1
        except FileNotFoundError:
            pytest.skip("Sample JavaScript file not available")


class TestConfigurationExamples:
    """Test configuration examples and settings."""

    def test_different_strategies(self):
        """Test different chunking strategies work correctly."""
        strategies = ["function", "class", "logical_block", "line_count"]

        code = """
class TestClass {
    method1() { return 1; }
    method2() { return 2; }
}

function standalone() { return 3; }
"""

        for strategy in strategies:
            chunker = JavaScriptChunker(chunk_by=strategy)
            result = chunker.chunk(code)

            assert result.total_chunks >= 1
            assert result.strategy_used == "javascript_code"

    def test_feature_toggles(self):
        """Test feature toggle configurations."""
        code = """
import React from 'react';
export const API_URL = 'test';

function Component() {
    return <div>Test</div>;
}
"""

        # Test with imports disabled
        chunker_no_imports = JavaScriptChunker(include_imports=False)
        result = chunker_no_imports.chunk(code)

        # Test with JSX disabled
        chunker_no_jsx = JavaScriptChunker(handle_jsx=False)
        result_no_jsx = chunker_no_jsx.chunk(code)

        assert isinstance(result, ChunkingResult)
        assert isinstance(result_no_jsx, ChunkingResult)


if __name__ == "__main__":
    pytest.main([__file__])
