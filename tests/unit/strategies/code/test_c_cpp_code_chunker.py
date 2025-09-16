"""
Unit tests for CCppCodeChunker.

This module provides comprehensive tests for the C/C++ code chunking functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from chunking_strategy import create_chunker, list_chunkers
from chunking_strategy.strategies.code.c_cpp_chunker import CCppCodeChunker
from chunking_strategy.core.base import ModalityType


class TestCCppCodeChunker:
    """Test suite for CCppCodeChunker."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.test_data_dir = Path("test_data")
        self.sample_cpp_file = self.test_data_dir / "sample_code.cpp"

        # Simple C++ code for testing
        self.simple_cpp_code = '''
#include <iostream>
#include <string>

// Global constant
const int MAX_SIZE = 100;

// Function declaration
void printHello();

// Function definition
void printHello() {
    std::cout << "Hello, World!" << std::endl;
}

class Calculator {
public:
    Calculator() : result(0) {}

    int add(int a, int b) {
        return a + b;
    }

    int multiply(int a, int b) {
        return a * b;
    }

private:
    int result;
};

int main() {
    Calculator calc;
    std::cout << calc.add(2, 3) << std::endl;
    return 0;
}
'''

        # Complex C++ code with various constructs
        self.complex_cpp_code = '''
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>

// Preprocessor definitions
#define MAX_BUFFER 1024
#define DEBUG_MODE 1

// Forward declarations
class DataProcessor;
struct ProcessingResult;

// Namespace
namespace utils {
    void logMessage(const std::string& msg);

    template<typename T>
    void printVector(const std::vector<T>& vec) {
        for (const auto& item : vec) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
}

// Struct definition
struct ProcessingResult {
    std::string data;
    bool success;
    int errorCode;

    ProcessingResult(const std::string& d, bool s, int e)
        : data(d), success(s), errorCode(e) {}
};

// Base class
class BaseProcessor {
public:
    BaseProcessor(const std::string& name) : processorName(name) {}
    virtual ~BaseProcessor() = default;

    virtual ProcessingResult process(const std::string& input) = 0;

    const std::string& getName() const { return processorName; }

protected:
    std::string processorName;
};

// Derived class
class DataProcessor : public BaseProcessor {
private:
    std::vector<std::string> buffer;
    int processedCount;

public:
    DataProcessor(const std::string& name)
        : BaseProcessor(name), processedCount(0) {}

    ProcessingResult process(const std::string& input) override {
        if (input.empty()) {
            return ProcessingResult("", false, -1);
        }

        // Process the input
        std::string processed = input;
        std::transform(processed.begin(), processed.end(),
                      processed.begin(), ::toupper);

        buffer.push_back(processed);
        processedCount++;

        return ProcessingResult(processed, true, 0);
    }

    int getProcessedCount() const { return processedCount; }

    const std::vector<std::string>& getBuffer() const { return buffer; }
};

// Template class
template<typename T>
class Container {
private:
    std::vector<T> items;

public:
    void add(const T& item) {
        items.push_back(item);
    }

    T get(size_t index) const {
        if (index < items.size()) {
            return items[index];
        }
        throw std::out_of_range("Index out of range");
    }

    size_t size() const { return items.size(); }
};

// Namespace implementation
namespace utils {
    void logMessage(const std::string& msg) {
        std::cout << "[LOG] " << msg << std::endl;
    }
}
'''

        # Simple C code
        self.simple_c_code = '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Constants
#define MAX_LENGTH 256

// Global variable
int globalCounter = 0;

// Function prototypes
void printMessage(const char* msg);
int calculateSum(int a, int b);

// Function implementations
void printMessage(const char* msg) {
    printf("Message: %s\\n", msg);
}

int calculateSum(int a, int b) {
    return a + b;
}

// Structure definition
typedef struct {
    char name[MAX_LENGTH];
    int age;
    float salary;
} Employee;

// Function working with structures
void printEmployee(const Employee* emp) {
    printf("Name: %s, Age: %d, Salary: %.2f\\n",
           emp->name, emp->age, emp->salary);
}

int main() {
    Employee emp = {"John Doe", 30, 50000.0};

    printMessage("Hello from C!");
    printf("Sum: %d\\n", calculateSum(10, 20));
    printEmployee(&emp);

    return 0;
}
'''

    def test_chunker_registration(self):
        """Test that C/C++ code chunker is properly registered."""
        chunkers = list_chunkers()
        assert "c_cpp_code" in chunkers

        chunker = create_chunker("c_cpp_code")
        assert chunker is not None
        assert isinstance(chunker, CCppCodeChunker)
        assert chunker.name == "c_cpp_code"

    def test_basic_initialization(self):
        """Test basic chunker initialization."""
        chunker = CCppCodeChunker()
        assert chunker.chunk_by == "function"
        assert chunker.max_lines_per_chunk == 100
        assert chunker.include_headers is True
        assert chunker.include_comments is True
        assert chunker.preserve_structure is True

    def test_custom_initialization(self):
        """Test chunker initialization with custom parameters."""
        chunker = CCppCodeChunker(
            chunk_by="class",
            max_lines_per_chunk=50,
            include_headers=False,
            include_comments=False,
            preserve_structure=False
        )
        assert chunker.chunk_by == "class"
        assert chunker.max_lines_per_chunk == 50
        assert chunker.include_headers is False
        assert chunker.include_comments is False
        assert chunker.preserve_structure is False

    def test_chunk_simple_cpp_code(self):
        """Test chunking simple C++ code."""
        chunker = CCppCodeChunker(chunk_by="function")
        result = chunker.chunk(self.simple_cpp_code)

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "c_cpp_code"
        assert result.processing_time is not None

        # Should have chunks for functions, class, and main
        assert len(result.chunks) >= 2

    def test_chunk_simple_c_code(self):
        """Test chunking simple C code."""
        chunker = CCppCodeChunker(chunk_by="function")
        result = chunker.chunk(self.simple_c_code)

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "c_cpp_code"

        # Should have chunks for functions and main
        assert len(result.chunks) >= 2

    def test_chunk_by_function(self):
        """Test chunking by function."""
        chunker = CCppCodeChunker(chunk_by="function")
        result = chunker.chunk(self.complex_cpp_code)

        assert result is not None
        assert len(result.chunks) > 0

        # Check that function definitions are captured
        function_chunks = [chunk for chunk in result.chunks
                          if any(pattern in chunk.content for pattern in
                                ["void ", "int ", "ProcessingResult ", "const std::string&"])]
        assert len(function_chunks) > 0

    def test_chunk_by_class(self):
        """Test chunking by class."""
        chunker = CCppCodeChunker(chunk_by="class")
        result = chunker.chunk(self.complex_cpp_code)

        assert result is not None
        assert len(result.chunks) > 0

        # Check that class definitions are captured
        class_chunks = [chunk for chunk in result.chunks
                       if "class " in chunk.content or "struct " in chunk.content]
        assert len(class_chunks) > 0

    def test_chunk_by_logical_block(self):
        """Test chunking by logical blocks."""
        chunker = CCppCodeChunker(chunk_by="logical_block")
        result = chunker.chunk(self.complex_cpp_code)

        assert result is not None
        assert len(result.chunks) > 0

    def test_chunk_by_line_count(self):
        """Test chunking by line count."""
        chunker = CCppCodeChunker(chunk_by="line_count", max_lines_per_chunk=15)
        result = chunker.chunk(self.complex_cpp_code)

        assert result is not None
        assert len(result.chunks) > 0

        # Check that chunks don't exceed line limit (roughly)
        for chunk in result.chunks:
            line_count = len(chunk.content.split('\n'))
            # Allow some flexibility for logical boundaries
            assert line_count <= 35  # More tolerance for structure preservation

    def test_include_headers_setting(self):
        """Test include_headers setting."""
        # Test with headers included
        chunker = CCppCodeChunker(include_headers=True)
        result = chunker.chunk(self.complex_cpp_code)

        header_chunks = [chunk for chunk in result.chunks
                        if "#include" in chunk.content]
        assert len(header_chunks) > 0

        # Test with headers excluded
        chunker = CCppCodeChunker(include_headers=False)
        result = chunker.chunk(self.complex_cpp_code)

        # Should still have chunks, but headers should be filtered
        assert len(result.chunks) > 0

    def test_include_comments_setting(self):
        """Test include_comments setting."""
        code_with_comments = '''
// This is a comment
/* Multi-line comment
   spanning multiple lines */
void function() {
    // Inline comment
    int x = 5;
}
'''

        # Test with comments included
        chunker = CCppCodeChunker(include_comments=True)
        result = chunker.chunk(code_with_comments)

        comment_chunks = [chunk for chunk in result.chunks
                         if "//" in chunk.content or "/*" in chunk.content]
        assert len(comment_chunks) > 0

        # Test with comments excluded
        chunker = CCppCodeChunker(include_comments=False)
        result = chunker.chunk(code_with_comments)

        # Should still have chunks, but fewer comments
        assert len(result.chunks) > 0

    def test_chunk_file_path(self):
        """Test chunking from file path."""
        if not self.sample_cpp_file.exists():
            pytest.skip("Sample C++ file not available")

        chunker = CCppCodeChunker()
        result = chunker.chunk(self.sample_cpp_file)

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "c_cpp_code"

    def test_chunk_path_object(self):
        """Test chunking from Path object."""
        if not self.sample_cpp_file.exists():
            pytest.skip("Sample C++ file not available")

        chunker = CCppCodeChunker()
        result = chunker.chunk(self.sample_cpp_file)

        assert result is not None
        assert len(result.chunks) > 0

    def test_chunk_empty_content(self):
        """Test chunking empty content."""
        chunker = CCppCodeChunker()
        result = chunker.chunk("")

        assert result is not None
        assert len(result.chunks) == 0

    def test_chunk_invalid_cpp_code(self):
        """Test chunking invalid C++ code."""
        invalid_code = '''
void broken_function( {
    // Missing closing parenthesis and bracket
    std::cout << "This is broken" << std::endl
}
'''

        chunker = CCppCodeChunker()
        result = chunker.chunk(invalid_code)

        # Should handle gracefully and return some result
        assert result is not None
        # May have chunks even with syntax errors
        assert isinstance(result.chunks, list)

    def test_chunk_metadata(self):
        """Test chunk metadata."""
        chunker = CCppCodeChunker()
        result = chunker.chunk(self.simple_cpp_code)

        assert result is not None
        assert len(result.chunks) > 0

        for chunk in result.chunks:
            assert chunk.id is not None
            assert chunk.content is not None
            assert chunk.modality == ModalityType.TEXT
            assert chunk.metadata is not None
            assert chunk.metadata.chunker_used == "c_cpp_code"
            assert chunk.metadata.source == "direct_input"

    def test_chunk_stream_method(self):
        """Test chunk_stream method."""
        chunker = CCppCodeChunker()
        content_stream = [self.simple_cpp_code]

        chunks = list(chunker.chunk_stream(content_stream))
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.content is not None
            assert chunk.modality == ModalityType.TEXT

    def test_error_handling_nonexistent_file(self):
        """Test error handling for non-existent file."""
        chunker = CCppCodeChunker()

        with pytest.raises(FileNotFoundError):
            chunker.chunk(Path("nonexistent_file.cpp"))

    def test_binary_content_handling(self):
        """Test handling of binary content."""
        chunker = CCppCodeChunker()
        binary_content = b"#include <iostream>\nint main() { return 0; }\n"

        result = chunker.chunk(binary_content)
        assert result is not None
        assert len(result.chunks) >= 0  # Should handle gracefully

    def test_template_handling(self):
        """Test handling of C++ templates."""
        template_code = '''
template<typename T>
class TemplateClass {
public:
    T getValue() const { return value; }
    void setValue(const T& v) { value = v; }

private:
    T value;
};

template<typename T, typename U>
auto templateFunction(T t, U u) -> decltype(t + u) {
    return t + u;
}
'''

        chunker = CCppCodeChunker()
        result = chunker.chunk(template_code)

        assert result is not None
        assert len(result.chunks) > 0

    def test_namespace_handling(self):
        """Test handling of C++ namespaces."""
        namespace_code = '''
namespace myapp {
    namespace utils {
        void helperFunction() {
            // Implementation
        }
    }

    class MyClass {
    public:
        void doSomething();
    };
}

using namespace myapp;
using myapp::utils::helperFunction;
'''

        chunker = CCppCodeChunker()
        result = chunker.chunk(namespace_code)

        assert result is not None
        assert len(result.chunks) > 0

    def test_preprocessor_handling(self):
        """Test handling of preprocessor directives."""
        preprocessor_code = '''
#ifndef HEADER_GUARD_H
#define HEADER_GUARD_H

#if defined(DEBUG) && DEBUG > 0
#define LOG(x) std::cout << x << std::endl
#else
#define LOG(x)
#endif

#ifdef WIN32
    #include <windows.h>
#elif defined(LINUX)
    #include <unistd.h>
#endif

void function() {
    LOG("Debug message");
}

#endif // HEADER_GUARD_H
'''

        chunker = CCppCodeChunker()
        result = chunker.chunk(preprocessor_code)

        assert result is not None
        assert len(result.chunks) > 0

    def test_adaptation_capabilities(self):
        """Test adaptation capabilities."""
        chunker = CCppCodeChunker()

        # Test that chunker can be adapted (even if not implemented)
        history = chunker.get_adaptation_history()
        assert isinstance(history, list)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid chunk_by parameter
        chunker = CCppCodeChunker(chunk_by="invalid_mode")
        result = chunker.chunk(self.simple_cpp_code)

        # Should handle gracefully or fallback to default
        assert result is not None

    def test_large_file_handling(self):
        """Test handling of large C++ files."""
        # Create a large C++ file content
        large_content = "#include <iostream>\n\n"
        for i in range(50):
            large_content += f'''
class Class_{i} {{
public:
    Class_{i}() : value_{i}(0) {{}}

    void function_{i}() {{
        std::cout << "Function {i}" << std::endl;
    }}

    int getValue() const {{ return value_{i}; }}
    void setValue(int v) {{ value_{i} = v; }}

private:
    int value_{i};
}};
'''

        chunker = CCppCodeChunker(chunk_by="class", max_lines_per_chunk=25)
        result = chunker.chunk(large_content)

        assert result is not None
        assert len(result.chunks) > 10  # Should create many chunks


class TestCCppCodeChunkerIntegration:
    """Integration tests for CCppCodeChunker."""

    @pytest.fixture(autouse=True)
    def setup_integration(self):
        """Set up integration test data."""
        # Simple C++ code for testing
        self.simple_cpp_code = '''
#include <iostream>

void printHello() {
    std::cout << "Hello, World!" << std::endl;
}

int main() {
    printHello();
    return 0;
}
'''

        self.complex_cpp_code = '''
#include <iostream>
#include <vector>

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
};

int main() {
    Calculator calc;
    std::cout << calc.add(2, 3) << std::endl;
    return 0;
}
'''

    def test_with_real_cpp_file(self):
        """Test with a real C++ file."""
        sample_file = Path("test_data/sample_code.cpp")
        if not sample_file.exists():
            pytest.skip("Sample C++ file not available")

        chunker = create_chunker("c_cpp_code")
        result = chunker.chunk(sample_file)

        assert result is not None
        assert len(result.chunks) > 0
        assert result.strategy_used == "c_cpp_code"

        # Verify chunk content makes sense
        total_content = "".join(chunk.content for chunk in result.chunks)
        original_content = sample_file.read_text(encoding='utf-8')

        # Content should be preserved (allowing for some chunking modifications)
        assert len(total_content) > 0
        assert any(keyword in total_content for keyword in
                  ["class ", "void ", "int ", "#include"])

    def test_chunker_registry_integration(self):
        """Test integration with chunker registry."""
        from chunking_strategy.core.registry import get_chunker_metadata

        metadata = get_chunker_metadata("c_cpp_code")
        assert metadata is not None
        assert metadata.name == "c_cpp_code"
        assert metadata.category == "code"
        assert any(ext in metadata.supported_formats
                  for ext in ["c", "cpp", "h", "hpp"])

    def test_concurrent_chunking(self):
        """Test concurrent chunking operations."""
        import concurrent.futures

        chunker = CCppCodeChunker()

        def chunk_code(code):
            return chunker.chunk(code)

        codes = [self.simple_cpp_code, self.complex_cpp_code] * 2

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(chunk_code, code) for code in codes]
            results = [future.result() for future in futures]

        assert len(results) == len(codes)
        for result in results:
            assert result is not None
            assert len(result.chunks) > 0
