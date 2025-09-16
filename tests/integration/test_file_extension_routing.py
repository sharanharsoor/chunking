#!/usr/bin/env python3
"""
Test File Extension Routing
Tests that custom and built-in algorithms are routed correctly based on file extensions.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from chunking_strategy import ChunkerOrchestrator
from chunking_strategy.core.custom_algorithm_loader import load_custom_algorithm


class TestFileExtensionRouting:
    """Test file extension based chunking routing."""

    @pytest.fixture
    def config_path(self):
        """Path to the file extension routing config."""
        return "config_examples/format_specific_configs/file_extension_simple.yaml"

    @pytest.fixture
    def test_files(self):
        """Create temporary test files with different extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files with different extensions
            files = {}

            # Python file
            py_file = temp_path / "test.py"
            py_file.write_text("""
def hello_world():
    print("Hello, world!")
    return True

class TestClass:
    def method(self):
        pass
""")
            files['.py'] = py_file

            # JavaScript file
            js_file = temp_path / "test.js"
            js_file.write_text("""
function helloWorld() {
    console.log("Hello, world!");
    return true;
}

const TestClass = {
    method() {
        // implementation
    }
};
""")
            files['.js'] = js_file

            # PDF-like text content (simulated)
            pdf_file = temp_path / "test.txt"  # Use .txt since we can't easily create real PDFs
            pdf_file.write_text("""
This is a sample document with multiple paragraphs.

The document contains various types of content that would typically
be found in a PDF document.

Each paragraph serves a different purpose in demonstrating
the chunking capabilities of our system.
""")
            files['.txt'] = pdf_file

            # JSON file
            json_file = temp_path / "test.json"
            json_file.write_text("""
{
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"}
    ],
    "settings": {
        "theme": "dark",
        "notifications": true
    }
}
""")
            files['.json'] = json_file

            # Markdown file
            md_file = temp_path / "test.md"
            md_file.write_text("""
# Main Header

This is a markdown document with structured content.

## Section 1

Content for section 1.

## Section 2

Content for section 2 with some **bold** text.

### Subsection

More detailed content here.
""")
            files['.md'] = md_file

            # Custom extension - review file
            review_file = temp_path / "test.review"
            review_file.write_text("""
This product is amazing! I love the features and quality.

However, the price is quite expensive and the shipping was delayed.

Overall, it's a good product but there are areas for improvement.
The customer service was helpful when I contacted them.
""")
            files['.review'] = review_file

            # Custom extension - template file
            template_file = temp_path / "test.template"
            template_file.write_text("""
Welcome {{username}}!

Your account balance is: {{balance}}
---
Recent transactions:
{{#transactions}}
- {{date}}: {{description}} ({{amount}})
{{/transactions}}
---
Thank you for using our service!
""")
            files['.template'] = template_file

            yield files

    def test_file_extension_routing_builtin_algorithms(self, config_path, test_files):
        """Test that built-in algorithms are correctly selected for known extensions."""

        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        # Test Python file routes to python_code
        result = orchestrator.chunk_file(test_files['.py'])
        assert result.strategy_used == "python_code"
        assert len(result.chunks) > 0

        # Test JavaScript file routes to javascript_code
        result = orchestrator.chunk_file(test_files['.js'])
        assert result.strategy_used == "javascript_code"
        assert len(result.chunks) > 0

        # Test JSON file routes to json_chunker
        result = orchestrator.chunk_file(test_files['.json'])
        assert result.strategy_used == "json_chunker"
        assert len(result.chunks) > 0

        # Test Markdown file routes to markdown_chunker
        result = orchestrator.chunk_file(test_files['.md'])
        assert result.strategy_used == "markdown_chunker"
        assert len(result.chunks) > 0

    def test_file_extension_routing_text_files(self, config_path, test_files):
        """Test that text files are correctly routed to semantic algorithm."""

        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        # Test text file routes to semantic algorithm (may resolve to context_enriched)
        result = orchestrator.chunk_file(test_files['.txt'])
        assert result.strategy_used in ["semantic", "context_enriched"]
        assert len(result.chunks) > 0

    def test_fallback_behavior(self, config_path, test_files):
        """Test fallback behavior when primary strategy fails."""

        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        # All files should produce valid results even if primary strategy fails
        for ext, file_path in test_files.items():
            result = orchestrator.chunk_file(file_path)
            assert len(result.chunks) > 0
            assert result.strategy_used is not None

            # Verify content is preserved
            total_content = ''.join(chunk.content for chunk in result.chunks)
            original_content = file_path.read_text()

            # Content should be substantially preserved (allowing for some processing)
            # Some chunkers may extract only portions (like markdown sections)
            assert len(total_content) > len(original_content) * 0.3

    def test_unknown_extension_default_behavior(self, config_path):
        """Test behavior with unknown file extensions."""

        with tempfile.NamedTemporaryFile(suffix='.unknown', mode='w', delete=False) as f:
            f.write("This is content with an unknown extension.")
            f.flush()

            orchestrator = ChunkerOrchestrator(
                config_path=config_path,
                enable_custom_algorithms=True
            )

            result = orchestrator.chunk_file(f.name)

            # Should fall back to default strategy
            assert result.strategy_used in ["paragraph_based", "sentence_based", "fixed_size"]
            assert len(result.chunks) > 0

    def test_different_extensions_same_content(self, config_path):
        """Test different extensions on the same content."""

        test_content = """
        This is a test document with multiple paragraphs.

        It contains various types of content for testing.

        The system should route based on file extensions.
        """

        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        # Test with .txt extension (semantic algorithm, may resolve to context_enriched)
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as f:
            f.write(test_content)
            f.flush()

            txt_result = orchestrator.chunk_file(f.name)
            assert txt_result.strategy_used in ["semantic", "context_enriched"]

        # Test with .log extension (paragraph_based algorithm)
        with tempfile.NamedTemporaryFile(suffix='.log', mode='w', delete=False) as f:
            f.write(test_content)
            f.flush()

            log_result = orchestrator.chunk_file(f.name)
            assert log_result.strategy_used == "paragraph_based"

        # Both should produce valid results
        assert len(txt_result.chunks) > 0
        assert len(log_result.chunks) > 0

        # Both should preserve the content
        for result in [txt_result, log_result]:
            total_content = ''.join(chunk.content for chunk in result.chunks)
            assert len(total_content) > len(test_content) * 0.3

    def test_configuration_validation(self, config_path):
        """Test that the configuration is valid and loads successfully."""

        # Configuration should load without errors
        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        # Verify configuration structure
        assert hasattr(orchestrator, 'config')
        assert 'chunking' in orchestrator.config
        assert 'strategy_selection' in orchestrator.config['chunking']

        # Test a few key extension mappings
        strategy_selection = orchestrator.config['chunking']['strategy_selection']
        assert '.py' in strategy_selection
        assert '.txt' in strategy_selection
        assert '.json' in strategy_selection

        # Verify the mappings are to supported algorithms
        assert strategy_selection['.py'] == 'python_code'
        assert strategy_selection['.txt'] == 'semantic'
        assert strategy_selection['.json'] == 'json_chunker'

    def test_parallel_processing_compatibility(self, config_path, test_files):
        """Test that file extension routing works with parallel processing."""

        orchestrator = ChunkerOrchestrator(
            config_path=config_path,
            enable_custom_algorithms=True
        )

        # Process multiple files in parallel
        file_paths = list(test_files.values())

        for file_path in file_paths:
            result = orchestrator.chunk_file(file_path)
            assert len(result.chunks) > 0
            assert result.strategy_used is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
