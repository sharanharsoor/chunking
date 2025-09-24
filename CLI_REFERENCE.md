# 🖥️ CLI Reference Guide

Complete command-line interface documentation for the chunking-strategy library.

## 📋 Table of Contents

- [Installation & Setup](#-installation--setup)
- [Basic Commands](#-basic-commands)
- [File Processing](#-file-processing)
- [Batch Operations](#-batch-operations)
- [Strategy Management](#-strategy-management)
- [Configuration & Settings](#-configuration--settings)
- [Advanced Features](#-advanced-features)
- [Output Formats](#-output-formats)
- [Examples & Recipes](#-examples--recipes)

## 🚀 Installation & Setup

```bash
# Install with CLI support
pip install chunking-strategy[all]

# Verify installation
python -m chunking_strategy --version

# Get help
python -m chunking_strategy --help
```

## 📝 Basic Commands

### Main Command Structure
```bash
python -m chunking_strategy [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

### Global Options
```bash
--version                     # Show version and exit
-v, --verbose                 # Enable verbose output
-q, --quiet                   # Suppress output except errors
--debug                       # Enable debug mode
--log-level LEVEL             # Set log level (silent|minimal|normal|verbose|debug|trace)
--log-file PATH               # Write logs to file
--help                        # Show help
```

### Available Commands
```bash
chunk                # Chunk a single file
batch               # Process multiple files
batch-directory     # Process files in directory
process-directory   # Comprehensive directory processing
list-strategies     # List available strategies
test-strategy       # Test a specific strategy
benchmark           # Benchmark strategies
hardware            # Show hardware info
init-config         # Generate config template
custom              # Manage custom algorithms
embed               # Generate embeddings from chunked content
embed-batch         # Generate embeddings for multiple files
list-models         # List available embedding models
debug               # Debug utilities
```

## 📄 File Processing

### chunk - Process Single Files

```bash
python -m chunking_strategy chunk [OPTIONS] INPUT_FILE
```

**Options:**
```bash
-s, --strategy TEXT             # Chunking strategy (default: auto-detect)
-c, --config PATH               # Configuration file
-o, --output PATH               # Output file for chunks
--format [json|text|yaml]       # Output format (default: json)
--no-output                     # Suppress file generation (summary only)
--summary-only                  # Show only processing summary
--skip-large-output INTEGER     # Skip output if >N chunks
--chunk-size INTEGER            # Chunk size (for fixed-size strategy)
--max-sentences INTEGER         # Max sentences per chunk
--validate                      # Validate chunks after creation
--quality-report                # Generate quality report
--help                         # Show help for chunk command
```

**Examples:**
```bash
# Basic file chunking
python -m chunking_strategy chunk document.txt

# Specify strategy and output
python -m chunking_strategy chunk document.pdf --strategy pdf_chunker --output chunks.json

# Text chunking with parameters
python -m chunking_strategy chunk article.txt --strategy sentence_based --max-sentences 3

# PDF with validation and quality report
python -m chunking_strategy chunk report.pdf --validate --quality-report --format yaml

# Code file chunking
python -m chunking_strategy chunk script.py --strategy python_code --output code_chunks.json

# Just see summary, no file output
python -m chunking_strategy chunk large_doc.pdf --summary-only
```

## 📂 Batch Operations

### batch - Process Multiple Files

```bash
python -m chunking_strategy batch [OPTIONS] INPUT_FILES...
```

**Options:**
```bash
-s, --strategy TEXT             # Chunking strategy for all files
-c, --config PATH               # Configuration file
-o, --output-dir PATH           # Output directory
--format [json|text|yaml]       # Output format
--parallel-mode [auto|sequential|thread|process]  # Processing mode
--workers INTEGER               # Number of workers
--show-progress                 # Show progress bar
--validate                      # Validate all chunks
--quality-report                # Generate quality reports
--help                         # Show help
```

**Examples:**
```bash
# Process multiple text files
python -m chunking_strategy batch file1.txt file2.txt file3.txt --strategy paragraph_based

# Batch process with parallel processing
python -m chunking_strategy batch *.pdf --strategy pdf_chunker --parallel-mode process --workers 4

# All files in current directory with specific extension
python -m chunking_strategy batch *.py --strategy python_code --output-dir chunks/

# With configuration file
python -m chunking_strategy batch documents/*.pdf --config my_config.yaml
```

### process-directory - Comprehensive Directory Processing

```bash
python -m chunking_strategy process-directory [OPTIONS] DIRECTORY
```

**Options:**
```bash
-o, --output-dir PATH           # Output directory
-c, --config PATH               # Configuration file
--extensions TEXT               # File extensions (comma-separated)
--recursive / --no-recursive    # Process subdirectories
--parallel-mode [auto|sequential|thread|process]  # Processing mode
--workers INTEGER               # Number of workers
--show-preview                  # Show chunk previews
--max-preview-chunks INTEGER    # Max preview chunks per file
--help                         # Show help
```

**Examples:**
```bash
# Process all files in directory
python -m chunking_strategy process-directory ./documents/

# Specific file types with recursion
python -m chunking_strategy process-directory ./project/ --extensions .py,.js,.md --recursive

# Parallel processing with preview
python -m chunking_strategy process-directory ./docs/ --parallel-mode process --show-preview

# Output to specific directory
python -m chunking_strategy process-directory ./input/ --output-dir ./chunked_output/
```

## 🎯 Strategy Management

### list-strategies - Show Available Strategies

```bash
python -m chunking_strategy list-strategies [OPTIONS]
```

**Options:**
```bash
--category TEXT               # Filter by category
--modality TEXT               # Filter by modality
--format [table|json|simple]  # Output format
--show-details                # Show detailed information
--help                        # Show help
```

**Examples:**
```bash
# List all strategies
python -m chunking_strategy list-strategies

# Filter by category
python -m chunking_strategy list-strategies --category text

# Detailed information in JSON format
python -m chunking_strategy list-strategies --show-details --format json

# Filter by modality
python -m chunking_strategy list-strategies --modality text
```

### test-strategy - Test Strategy Performance

```bash
python -m chunking_strategy test-strategy [OPTIONS] STRATEGY_NAME
```

**Options:**
```bash
--test-file PATH         # File to test with
--test-text TEXT         # Text to test with
--chunk-size INTEGER     # Chunk size parameter
--max-sentences INTEGER  # Max sentences parameter
--validate              # Validate output
--help                  # Show help
```

**Examples:**
```bash
# Test strategy with file
python -m chunking_strategy test-strategy sentence_based --test-file document.txt

# Test with custom text
python -m chunking_strategy test-strategy fixed_size --test-text "Your test content here" --chunk-size 500

# Test with validation
python -m chunking_strategy test-strategy paragraph_based --test-file report.pdf --validate
```

### benchmark - Compare Strategy Performance

```bash
python -m chunking_strategy benchmark [OPTIONS] INPUT_FILE
```

**Options:**
```bash
--strategies TEXT       # Comma-separated strategy names
--runs INTEGER         # Number of runs per strategy (default: 3)
--output PATH          # Save benchmark results
--format [json|csv]    # Output format
--include-quality      # Include quality metrics
--help                 # Show help
```

**Examples:**
```bash
# Benchmark multiple strategies
python -m chunking_strategy benchmark document.pdf --strategies sentence_based,paragraph_based,fixed_size

# Multiple runs with quality metrics
python -m chunking_strategy benchmark large_file.txt --strategies semantic,adaptive --runs 5 --include-quality

# Save results to file
python -m chunking_strategy benchmark test.pdf --strategies pdf_chunker,universal_document --output benchmark_results.json
```

## ⚙️ Configuration & Settings

### init-config - Generate Configuration Template

```bash
python -m chunking_strategy init-config [OPTIONS]
```

**Options:**
```bash
--output PATH           # Output file path (default: config.yaml)
--template TEXT         # Template type (basic|advanced|rag|production)
--strategy TEXT         # Default strategy to include
--help                 # Show help
```

**Examples:**
```bash
# Generate basic config
python -m chunking_strategy init-config

# Advanced configuration template
python -m chunking_strategy init-config --template advanced --output advanced_config.yaml

# RAG-optimized template
python -m chunking_strategy init-config --template rag --strategy sentence_based
```

### hardware - System Information

```bash
python -m chunking_strategy hardware [OPTIONS]
```

**Options:**
```bash
--recommendations      # Show optimization recommendations
--json                # Output as JSON
--help               # Show help
```

**Examples:**
```bash
# Show hardware info
python -m chunking_strategy hardware

# With recommendations
python -m chunking_strategy hardware --recommendations

# JSON format for programmatic use
python -m chunking_strategy hardware --json
```

## 🧠 Advanced Features

### embed - Generate Embeddings

```bash
python -m chunking_strategy embed [OPTIONS] INPUT_FILE
```

**Options:**
```bash
--model TEXT                    # Embedding model name
--output-format TEXT            # Output format (simple|full_metadata)
--batch-size INTEGER            # Processing batch size
--normalize                     # Normalize embeddings
--help                         # Show help
```

### embed-batch - Batch Embedding Generation

```bash
python -m chunking_strategy embed-batch [OPTIONS] INPUT_DIRECTORY
```

### list-models - List Available Models

```bash
python -m chunking_strategy list-models [OPTIONS]
```

**Examples:**
```bash
# Generate embeddings
python -m chunking_strategy embed document.pdf --model all-MiniLM-L6-v2

# Batch embeddings with normalization
python -m chunking_strategy embed-batch documents/ --model all-mpnet-base-v2 --normalize

# List available embedding models
python -m chunking_strategy list-models
```

### custom - Custom Algorithm Management

```bash
python -m chunking_strategy custom [OPTIONS] COMMAND
```

**Subcommands:**
```bash
load PATH              # Load custom algorithms from path
validate NAME          # Validate custom algorithm
list                   # List loaded custom algorithms
help                   # Show custom command help
```

**Examples:**
```bash
# Load custom algorithms
python -m chunking_strategy custom load ./my_algorithms/

# Validate custom algorithm
python -m chunking_strategy custom validate my_custom_chunker

# List all custom algorithms
python -m chunking_strategy custom list
```

### debug - Debug Utilities

```bash
python -m chunking_strategy debug [OPTIONS] COMMAND
```

**Subcommands:**
```bash
collect                # Collect debug information
test-logging          # Test logging configuration
system-info           # Show system information
help                  # Show debug command help
```

**Examples:**
```bash
# Collect debug info
python -m chunking_strategy debug collect --description "Processing issue with PDF files"

# Test logging
python -m chunking_strategy debug test-logging --level verbose

# System information
python -m chunking_strategy debug system-info
```

## 📊 Output Formats

### JSON Format (Default)
```bash
python -m chunking_strategy chunk document.txt --format json
```

**Output Structure:**
```json
{
  "chunks": [
    {
      "id": "chunk_0",
      "content": "Chunk content here...",
      "metadata": {
        "word_count": 45,
        "char_count": 234,
        "start_pos": 0,
        "end_pos": 234
      }
    }
  ],
  "strategy_used": "sentence_based",
  "processing_time": 0.123,
  "total_chunks": 5
}
```

### YAML Format
```bash
python -m chunking_strategy chunk document.txt --format yaml
```

### Text Format
```bash
python -m chunking_strategy chunk document.txt --format text
```

**Output Structure:**
```
=== CHUNKING RESULTS ===
Strategy: sentence_based
Processing Time: 0.123s
Total Chunks: 5

--- Chunk 1 ---
Content: First chunk content here...
Word Count: 45
Character Count: 234

--- Chunk 2 ---
...
```

## 📝 Examples & Recipes

### Quick Start Recipes

```bash
# 🚀 QUICK WINS

# Process any document - auto strategy selection
python -m chunking_strategy chunk document.pdf

# Batch process all PDFs in current directory
python -m chunking_strategy batch *.pdf --parallel-mode process

# See all available strategies
python -m chunking_strategy list-strategies

# Generate config template and customize
python -m chunking_strategy init-config --template rag
```

### Production Workflows

```bash
# 🏭 PRODUCTION READY

# High-performance batch processing
python -m chunking_strategy process-directory ./documents/ \
    --extensions .pdf,.doc,.docx \
    --parallel-mode process \
    --workers 8 \
    --output-dir ./processed/

# Quality-controlled processing with validation
python -m chunking_strategy batch *.pdf \
    --strategy pdf_chunker \
    --validate \
    --quality-report \
    --output-dir ./validated_chunks/

# Benchmark strategies for your data
python -m chunking_strategy benchmark sample_document.pdf \
    --strategies sentence_based,paragraph_based,semantic,adaptive \
    --runs 5 \
    --include-quality \
    --output benchmark_results.json
```

### RAG System Setup

```bash
# 🤖 RAG OPTIMIZATION

# Generate optimal chunks for RAG
python -m chunking_strategy chunk documents.pdf \
    --strategy sentence_based \
    --max-sentences 3 \
    --validate \
    --output rag_chunks.json

# Batch process with embeddings
python -m chunking_strategy batch documents/*.pdf \
    --strategy sentence_based \
    --parallel-mode process

# Then generate embeddings
python -m chunking_strategy embed-batch processed/ \
    --model all-MiniLM-L6-v2 \
    --normalize
```

### Code Analysis Workflows

```bash
# 💻 CODE PROCESSING

# Process Python codebase
python -m chunking_strategy process-directory ./src/ \
    --extensions .py \
    --strategy python_code \
    --recursive \
    --show-preview

# Multi-language code processing
python -m chunking_strategy batch \
    --config code_analysis_config.yaml \
    src/*.py src/*.js src/*.java

# Benchmark code chunking strategies
python -m chunking_strategy benchmark sample_code.py \
    --strategies python_code,universal_code,fixed_size \
    --include-quality
```

### Debugging & Troubleshooting

```bash
# 🔍 DEBUGGING

# Run with maximum debugging info
python -m chunking_strategy --debug --log-level trace \
    chunk problematic_file.pdf

# Collect debug information for support
python -m chunking_strategy debug collect \
    --description "PDF processing fails on large files"

# Test hardware optimization
python -m chunking_strategy hardware --recommendations

# Validate custom algorithms
python -m chunking_strategy custom validate my_custom_chunker
```

## 🆘 Getting Help

### Command-Specific Help
```bash
python -m chunking_strategy COMMAND --help
```

### Examples:
```bash
python -m chunking_strategy chunk --help
python -m chunking_strategy batch --help
python -m chunking_strategy list-strategies --help
```

### Error Troubleshooting
```bash
# Enable verbose logging for issues
python -m chunking_strategy --verbose --debug chunk problematic_file.pdf

# Generate debug report
python -m chunking_strategy debug collect --description "Describe your issue"
```

### Performance Issues
```bash
# Check hardware capabilities
python -m chunking_strategy hardware --recommendations

# Test strategy performance
python -m chunking_strategy test-strategy STRATEGY_NAME --test-file your_file.pdf

# Benchmark multiple strategies
python -m chunking_strategy benchmark your_file.pdf --strategies strategy1,strategy2,strategy3
```

---

## 🎯 Quick Reference Card

**Most Common Commands:**
```bash
# Process single file
python -m chunking_strategy chunk file.pdf

# Batch process
python -m chunking_strategy batch *.txt --strategy sentence_based

# List strategies
python -m chunking_strategy list-strategies

# Process directory
python -m chunking_strategy process-directory ./docs/

# Generate config
python -m chunking_strategy init-config

# Check hardware
python -m chunking_strategy hardware
```

**Most Useful Options:**
- `--strategy STRATEGY_NAME` - Choose chunking method
- `--parallel-mode process` - Use multiple CPU cores
- `--validate` - Check chunk quality
- `--output-dir PATH` - Specify output location
- `--config PATH` - Use configuration file
- `--debug` - Enable detailed logging

---

**🚀 Ready to chunk from the command line!** Use `--help` on any command for detailed options.
