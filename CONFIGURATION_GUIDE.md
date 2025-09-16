# Chunking Strategy Configuration Guide

This guide explains the **77 comprehensive configurations** organized by category for easy navigation and selection. Each configuration is optimized for specific use cases without redundancy.

## 🎯 **Organized Configuration Structure (77 Files)**

We've organized all configurations into logical categories for easy discovery and use:

```
config_examples/
├── basic_configs/              # 5 files - General use and learning
├── strategy_configs/           # 21 files - Algorithm-specific testing
├── format_specific_configs/    # 32 files - File format optimization
├── use_case_configs/          # 16 files - Application scenarios
├── advanced_configs/          # 3 files - Performance and parallelization
└── custom_algorithms/         # Custom extension examples
```

## 🚀 Quick Start Guide

### For Most Users (⭐ **RECOMMENDED**)
```yaml
# Use: basic_configs/enhanced_auto_strategy.yaml
chunking_strategy:
  profile_name: "enhanced_auto_strategy"
```
**Best for**: General use, mixed file types, automatic smart selection

### For Learning (🎓)
```yaml
# Use: basic_configs/basic_example.yaml
chunking_strategy:
  profile_name: "basic_example"
```
**Best for**: Tutorials, understanding chunking concepts

### For Software Development (🔧)
```yaml
# Use: format_specific_configs/code_focused_config.yaml
chunking_strategy:
  profile_name: "code_focused"
```
**Best for**: Code analysis, software development workflows

### For Document Processing (📄)
```yaml
# Use: format_specific_configs/document_first_config.yaml
chunking_strategy:
  profile_name: "document_first"
```
**Best for**: Research, content analysis, document processing

### For Large Files (⚡)
```yaml
# Use: use_case_configs/large_files_streaming.yaml
chunking_strategy:
  profile_name: "large_files_streaming"
```
**Best for**: Memory-efficient processing of large files

## 📁 Category Details

### 🎯 Basic Configs (5 files)
**Purpose**: General use and learning
- `auto_selection.yaml` - Automatic strategy selection
- `basic_example.yaml` - Simple configuration for learning
- `enhanced_auto_strategy.yaml` - ⭐ **RECOMMENDED** for most users
- `essential_text_processing.yaml` - Text-focused processing
- `semantic_analysis.yaml` - Semantic understanding

### 🔧 Strategy Configs (21 files)
**Purpose**: Algorithm-specific testing and comparison
- `adaptive_learning.yaml`, `adaptive_production.yaml` - Adaptive chunking
- `boundary_aware_chunker.yaml` - Content boundary detection
- `context_enriched_*.yaml` - Context-aware processing (3 variants)
- `discourse_aware_default.yaml` - Discourse structure analysis
- `embedding_based_chunker.yaml` - Embedding-driven chunking
- `fastcdc_*.yaml` - FastCDC algorithm variants (2 files)
- `fixed_length_word_chunker.yaml` - Fixed word-length chunks
- Hash-based algorithms: `buzhash_performance.yaml`, `gear_cdc_default.yaml`, `ml_cdc_hierarchical.yaml`, `rabin_fingerprinting_default.yaml`, `rolling_hash_default.yaml`, `tttd_balanced.yaml`
- Text processing: `overlapping_window_chunker.yaml`, `recursive_chunker.yaml`, `semantic_chunker.yaml`, `token_based_chunker.yaml`

### 📄 Format-Specific Configs (32 files)
**Purpose**: Optimized for specific file formats

**Code & Development (8 files)**:
- `code_focused_config.yaml` - General code processing
- `css_responsive.yaml` - CSS stylesheets
- `go_functions.yaml`, `java_enterprise.yaml`, `javascript_functions.yaml`, `react_components.yaml` - Language-specific
- `file_extension_routing.yaml`, `file_extension_simple.yaml` - Extension-based routing

**Documents (10 files)**:
- `document_first_config.yaml` - General document processing
- PDF: `enhanced_pdf_processing.yaml`, `enhanced_pdf_simple.yaml`, `pdf_multimodal.yaml`
- DOC: `doc_content_size.yaml`, `doc_paragraphs.yaml`, `doc_sections.yaml`
- Markdown: `markdown_content_blocks.yaml`, `markdown_granular.yaml`, `markdown_header_focused.yaml`, `markdown_sections.yaml`

**Data Formats (8 files)**:
- CSV: `csv_focused_config.yaml`, `csv_logical_grouping.yaml`, `csv_memory_optimized.yaml`, `csv_with_overlap.yaml`
- JSON: `json_array_processing.yaml`, `json_depth_limited.yaml`, `json_key_grouping.yaml`, `json_object_focused.yaml`, `json_size_optimized.yaml`
- XML/HTML: `html_attribute_based.yaml`, `html_tag_based.yaml`, `xml_hierarchy.yaml`, `xml_semantic.yaml`

### 🎯 Use-Case Configs (16 files)
**Purpose**: Application-specific scenarios

**AI & Embeddings (3 files)**:
- `embedding_optimized.yaml` - Vector databases
- `multimodal_embeddings.yaml` - Text + multimedia
- `rag_system.yaml` - RAG workflows

**Streaming & Large Files (6 files)**:
- `enhanced_streaming.yaml` - Advanced streaming
- `large_files_streaming.yaml` - Memory-efficient processing
- `streaming_large_files.yaml` - Large file handling
- `enhanced_directory_processing.yaml` - Batch processing
- `jsonl_streaming.yaml`, `streaming_jsonl.yaml` - JSONL processing

**Multimedia (7 files)**:
- Audio: `audio_time_based.yaml`, `silence_based_audio.yaml`, `time_based_audio.yaml`
- Video: `scene_based_video.yaml`, `video_time_based.yaml`
- Images: `grid_based_image.yaml`, `patch_based_image.yaml`

### ⚡ Advanced Configs (3 files)
**Purpose**: Performance optimization and testing
- `auto_strategy_test.yaml` - Strategy testing
- `parallel_algorithm_testing.yaml` - Parallelization testing
- `parallel_comparison_simple.yaml` - Performance comparison

## 📊 Configuration Selection Matrix

| **Use Case** | **Recommended Config** | **Category** | **Key Features** |
|--------------|------------------------|--------------|------------------|
| **General Use** | `basic_configs/enhanced_auto_strategy.yaml` | Basic | Auto-selection, robust fallbacks |
| **Learning** | `basic_configs/basic_example.yaml` | Basic | Simple, educational |
| **Software Development** | `format_specific_configs/code_focused_config.yaml` | Format | Code-aware, function extraction |
| **Document Analysis** | `format_specific_configs/document_first_config.yaml` | Format | Structure-aware, reading flow |
| **Large Files** | `use_case_configs/large_files_streaming.yaml` | Use-case | Memory-efficient, streaming |
| **AI/RAG Systems** | `use_case_configs/rag_system.yaml` | Use-case | Embedding-optimized |
| **Multimedia** | `use_case_configs/multimodal_embeddings.yaml` | Use-case | Text + media processing |
| **Performance Testing** | `advanced_configs/parallel_algorithm_testing.yaml` | Advanced | Benchmarking, comparison |

## 🎮 Usage Examples

### Basic Usage
```python
from chunking_strategy import ChunkerOrchestrator

# Load any configuration
orchestrator = ChunkerOrchestrator(
    config_path="config_examples/basic_configs/enhanced_auto_strategy.yaml"
)

# Process different file types
result = orchestrator.chunk_file("document.txt")
print(f"Strategy used: {result.strategy_used}")
print(f"Generated {len(result.chunks)} chunks")
```

### Advanced Usage with Format-Specific Config
```python
# Use JSON-specific configuration
orchestrator = ChunkerOrchestrator(
    config_path="config_examples/format_specific_configs/json_object_focused.yaml"
)

# Process JSON with object-aware chunking
result = orchestrator.chunk_file("data.json")
for chunk in result.chunks[:3]:
    print(f"JSON Object: {chunk.content[:100]}...")
```

### Streaming Large Files
```python
# Use streaming configuration for large files
orchestrator = ChunkerOrchestrator(
    config_path="config_examples/use_case_configs/large_files_streaming.yaml"
)

# Process large file with memory efficiency
result = orchestrator.chunk_file("large_dataset.csv")
print(f"Processed {len(result.chunks)} chunks efficiently")
```

## 🔄 Strategy Selection Logic

### Auto-Selection (Recommended)
The `enhanced_auto_strategy.yaml` config provides smart file-type detection:

```
.py files: python_code → sentence_based → paragraph_based → fixed_size
.js files: javascript → universal_code → paragraph_based → fixed_size
.json files: json_chunker → paragraph_based → sentence_based → fixed_size
.pdf files: pdf_chunker → paragraph_based → sentence_based → fixed_size
.txt files: sentence_based → paragraph_based → fixed_size
Default: sentence_based → paragraph_based → fixed_size
```

### User Override Always Available
```python
# Auto-selection
result = orchestrator.chunk_file("code.py")  # Uses python_code

# Manual override
result = orchestrator.chunk_file("code.py", strategy_override="fixed_size")
```

## 🏆 Configuration Recommendations

### 🥇 **Top Picks by Use Case**

1. **First Time Users**: `basic_configs/basic_example.yaml`
2. **General Production**: `basic_configs/enhanced_auto_strategy.yaml`
3. **Software Development**: `format_specific_configs/code_focused_config.yaml`
4. **Research/Analysis**: `format_specific_configs/document_first_config.yaml`
5. **Large Scale Processing**: `use_case_configs/large_files_streaming.yaml`
6. **AI Applications**: `use_case_configs/rag_system.yaml`

### 🎯 **Selection Guidelines**

**Choose Basic Configs when**:
- ✅ You want simple, reliable processing
- ✅ You're learning about chunking
- ✅ You need general-purpose configuration
- ✅ You want auto-selection with good defaults

**Choose Strategy Configs when**:
- ✅ You want to test specific algorithms
- ✅ You need to compare chunking approaches
- ✅ You're researching optimal strategies
- ✅ You want algorithm-specific features

**Choose Format-Specific Configs when**:
- ✅ You work primarily with specific file types
- ✅ You need format-aware processing
- ✅ You want optimized extraction for your format
- ✅ You need rich metadata from structured files

**Choose Use-Case Configs when**:
- ✅ You have specific application requirements
- ✅ You're building AI/ML applications
- ✅ You process multimedia content
- ✅ You need streaming/large-file support

**Choose Advanced Configs when**:
- ✅ You're benchmarking performance
- ✅ You need parallelization
- ✅ You're doing research or optimization
- ✅ You want to compare multiple strategies

## 🚀 Performance & Features

### ✅ What Works Great
- **Auto-detection**: File extension → appropriate strategy
- **Fallback chains**: Specialized → universal → fixed_size
- **User override**: Always respected for any file
- **Error handling**: Graceful fallbacks on processing errors
- **Memory safety**: Streaming support for files of any size
- **Rich metadata**: Format-specific information extraction
- **Parallelization**: Multi-core processing where beneficial

### ⚠️ Known Considerations
- **Specialized strategies**: May fallback to universal approaches
- **Large files**: Automatically use streaming (memory-efficient)
- **Complex formats**: Rich extraction may take longer
- **Dependencies**: Some formats require additional libraries (e.g., PDF → Tika)

## 🔧 Testing Your Configuration

To test any configuration:

```python
# Test a specific config
from chunking_strategy import ChunkerOrchestrator

config_path = "config_examples/strategy_configs/semantic_chunker.yaml"
orchestrator = ChunkerOrchestrator(config_path=config_path)

# Test with sample content
result = orchestrator.chunk("This is test content for verification.")
print(f"✅ Config works! Generated {len(result.chunks)} chunks")
print(f"Strategy used: {result.strategy_used}")
```

## 📞 Support & Next Steps

### Getting Help
1. **Start with recommended configs** based on your use case
2. **Check strategy used**: `result.strategy_used` shows what was applied
3. **Try user override** if you need a different approach
4. **Check logs** for detailed processing information
5. **Test with small files** before processing large datasets

### Customization
- **Modify existing configs** to suit your needs
- **Combine strategies** using the fallback chain approach
- **Create custom configs** using existing ones as templates
- **Add your own algorithms** using the custom_algorithms/ examples

---

*This guide covers all configurations organized for easy discovery and use. Each config is production-ready and tested for reliability.*
