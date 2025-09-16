# Examples and Demonstrations

This directory contains a comprehensive collection of examples demonstrating the chunking library's capabilities. Each example is carefully ordered to provide a logical learning progression from basic concepts to advanced production scenarios.

## 📚 Learning Path Overview

The examples are numbered 01-16, designed to be followed in sequence for optimal learning:

### 🎯 **Beginner Level (01-04): Foundation**
- **01-02**: Core concepts and basic usage
- **03-04**: Embeddings and CLI fundamentals

### 🔧 **Intermediate Level (05-08): Integration**
- **05-06**: RAG pipelines and streaming
- **07-08**: Metrics and extensibility

### 🚀 **Advanced Level (09-13): Performance**
- **09-11**: Framework and streaming mastery
- **12-13**: Parallelization techniques

### 👨‍💻 **Expert Level (14-16): Production**
- **14-15**: Complex scenarios and batch processing
- **16**: Setup verification and utilities

## 🚀 Quick Start

### Prerequisites

```bash
# Install the chunking library
pip install -e .

# Or from PyPI (when available)
pip install chunking-strategy
```

### Recommended Learning Sequence

```bash
# 1. Start with basics
python 01_basic_usage.py
python 02_advanced_usage.py

# 2. Learn integration patterns
python 03_embedding_workflows.py
python 04_cli_examples.py

# 3. Explore advanced features
python 05_embeddings_integration_demo.py
python 06_streaming_benefits_demo.py

# 4. Master optimization
python 07_metrics_collection_demo.py
python 08_extensibility_demo.py
```

## 📖 Complete Example Guide

---

## 01_basic_usage.py
**🎯 Level**: Beginner | **⏱️ Duration**: 5-10 minutes | **📋 Prerequisites**: None

### Purpose
Introduction to core chunking concepts and basic strategies.

### What You'll Learn
- How to create and use different chunking strategies
- Understanding chunk outputs and metadata
- Basic configuration options
- Performance considerations for different strategies

### Key Demonstrations
- Text chunking with sentence-based strategy
- Fixed-size chunking for consistent output
- Chunk metadata exploration
- Simple error handling

### When to Use This
- First time using the library
- Understanding fundamental concepts
- Quick proof-of-concept development

### Sample Output
```
🎯 BASIC CHUNKING DEMONSTRATION
=====================================
✅ Generated 5 chunks using sentence_based
📊 Average chunk size: 187 characters
⏱️  Processing time: 0.023 seconds
```

---

## 02_advanced_usage.py
**🎯 Level**: Beginner | **⏱️ Duration**: 10-15 minutes | **📋 Prerequisites**: 01_basic_usage.py

### Purpose
Advanced features, configuration patterns, and multi-modal content handling.

### What You'll Learn
- Complex configuration techniques
- Handling different content types (text, code, documents)
- Strategy comparison and selection
- Robust error handling patterns

### Key Demonstrations
- YAML configuration loading
- Multi-format content processing
- Strategy performance comparison
- Advanced error recovery

### When to Use This
- Building production systems
- Need configuration-driven processing
- Handling diverse content types

### Sample Output
```
🔧 ADVANCED CONFIGURATION DEMO
=====================================
📄 Processing: technical_document.pdf
   Strategy: semantic_chunker
   ✅ 12 chunks generated
   📊 Quality score: 0.847
```

---

## 03_embedding_workflows.py
**🎯 Level**: Beginner-Intermediate | **⏱️ Duration**: 15-20 minutes | **📋 Prerequisites**: 01_basic_usage.py

### Purpose
Embedding generation, vector operations, and similarity search fundamentals.

### What You'll Learn
- Converting chunks to embeddings
- Vector similarity calculations
- Quality assessment for embeddings
- Database integration patterns

### Key Demonstrations
- Chunk-to-embedding pipeline
- Similarity search implementation
- Embedding quality metrics
- Vector database preparation

### When to Use This
- Building RAG systems
- Semantic search applications
- Vector database integration

### Sample Output
```
🧠 EMBEDDING WORKFLOWS DEMO
=====================================
📊 Generated 384-dimensional embeddings
🔍 Top similarity: 0.923 (neural networks)
💾 Ready for vector database storage
```

---

## 04_cli_examples.py
**🎯 Level**: Beginner | **⏱️ Duration**: 10-15 minutes | **📋 Prerequisites**: 01_basic_usage.py

### Purpose
Command-line interface usage, batch processing, and automation workflows.

### What You'll Learn
- CLI command structures and options
- Batch processing patterns
- Configuration file management
- Output formatting and piping

### Key Demonstrations
- Single file processing commands
- Batch directory processing
- Configuration-driven CLI usage
- Output format customization

### When to Use This
- Automation and scripting
- DevOps integration
- Batch processing workflows

### Sample Output
```
🖥️  CLI PROCESSING DEMO
=====================================
$ chunking-cli process --strategy semantic document.txt
✅ Processed 1 file, generated 8 chunks
📊 Output saved to document_chunks.json
```

---

## 05_embeddings_integration_demo.py
**🎯 Level**: Intermediate | **⏱️ Duration**: 20-30 minutes | **📋 Prerequisites**: 03_embedding_workflows.py

### Purpose
Complete RAG (Retrieval Augmented Generation) pipeline demonstration.

### What You'll Learn
- End-to-end RAG workflow implementation
- Chunking strategy optimization for embeddings
- Retrieval performance tuning
- Quality evaluation and metrics

### Key Demonstrations
- RAG-optimized chunking strategies
- Embedding generation pipeline
- Similarity-based retrieval
- Quality assessment metrics

### When to Use This
- Building production RAG systems
- Optimizing retrieval quality
- Implementing semantic search

### Sample Output
```
🎯 RAG INTEGRATION DEMO
=====================================
🔧 Optimized chunking: 15 semantic chunks
🧠 Generated embeddings: 384 dimensions
🔍 Query retrieval: 0.834 avg relevance
📊 System quality score: 0.891
```

---

## 06_streaming_benefits_demo.py
**🎯 Level**: Intermediate | **⏱️ Duration**: 15-25 minutes | **📋 Prerequisites**: 02_advanced_usage.py

### Purpose
Memory-efficient streaming processing for large datasets.

### What You'll Learn
- Memory usage optimization techniques
- Streaming vs traditional processing comparison
- Checkpoint and resume functionality
- Real-time processing patterns

### Key Demonstrations
- Memory usage comparison
- Streaming file processing
- Progress monitoring and checkpoints
- Performance benchmarking

### When to Use This
- Processing large files (>100MB)
- Memory-constrained environments
- Real-time data processing

### Sample Output
```
🌊 STREAMING BENEFITS DEMO
=====================================
💾 Traditional: 45.3MB memory usage
🌊 Streaming: 12.1MB memory usage
💡 Memory savings: 73.3%
⚡ Processing: 1.05x faster
```

---

## 07_metrics_collection_demo.py
**🎯 Level**: Intermediate | **⏱️ Duration**: 20-30 minutes | **📋 Prerequisites**: 05_embeddings_integration_demo.py

### Purpose
Comprehensive performance monitoring, analysis, and optimization guidance.

### What You'll Learn
- Metrics collection and analysis
- Performance benchmarking techniques
- Quality evaluation methods
- Data-driven optimization strategies

### Key Demonstrations
- Multi-strategy performance comparison
- Memory and CPU profiling
- Quality metrics calculation
- Optimization recommendations

### When to Use This
- Performance optimization projects
- System monitoring setup
- Strategy selection guidance

### Sample Output
```
📊 PERFORMANCE ANALYSIS
=====================================
⚡ Fastest: fixed_size (0.012s)
💾 Most efficient: paragraph_based (8.4MB)
🎯 Highest quality: semantic (0.923)
🚀 Best throughput: 3.45 MB/s
```

---

## 08_extensibility_demo.py
**🎯 Level**: Intermediate-Advanced | **⏱️ Duration**: 30-45 minutes | **📋 Prerequisites**: 07_metrics_collection_demo.py

### Purpose
Custom algorithms, library integration, and extension development.

### What You'll Learn
- Creating custom chunking strategies
- Integrating external NLP libraries
- Plugin architecture usage
- Testing and validation approaches

### Key Demonstrations
- Custom algorithm development
- External library integration (spaCy, NLTK)
- Configuration-driven customization
- Validation and testing patterns

### When to Use This
- Developing custom algorithms
- Integrating specialized libraries
- Building domain-specific solutions

### Sample Output
```
🔧 EXTENSIBILITY DEMO
=====================================
✅ Custom sentiment chunker created
🔗 spaCy integration successful
📊 Custom algorithm performance: 0.156s
🎯 Validation passed: 100% accuracy
```

---

## 09_universal_framework_demo.py
**🎯 Level**: Advanced | **⏱️ Duration**: 25-35 minutes | **📋 Prerequisites**: 08_extensibility_demo.py

### Purpose
Understanding the universal framework architecture and cross-format processing.

### What You'll Learn
- Universal framework capabilities
- Cross-format content processing
- Strategy abstraction layers
- Content type detection and routing

### Key Demonstrations
- Multi-format content handling
- Universal strategy application
- Content type auto-detection
- Framework extensibility patterns

### When to Use This
- Building content-agnostic systems
- Understanding framework architecture
- Developing universal processors

### Sample Output
```
🌐 UNIVERSAL FRAMEWORK DEMO
=====================================
📄 Auto-detected: PDF document
🔧 Applied strategy: universal_document
✅ Cross-format processing successful
🎯 Unified output format achieved
```

---

## 10_enhanced_streaming_demo.py
**🎯 Level**: Advanced | **⏱️ Duration**: 30-40 minutes | **📋 Prerequisites**: 06_streaming_benefits_demo.py

### Purpose
Production-grade streaming capabilities and real-time processing.

### What You'll Learn
- Advanced streaming configurations
- Real-time processing patterns
- Data pipeline integration
- Monitoring and alerting systems

### Key Demonstrations
- Production streaming setup
- Real-time data processing
- Pipeline integration patterns
- Advanced monitoring capabilities

### When to Use This
- Real-time data processing systems
- Production streaming pipelines
- High-throughput applications

### Sample Output
```
🚀 ENHANCED STREAMING DEMO
=====================================
🌊 Real-time processing: 1.2GB/min
📊 Pipeline throughput: 99.7% uptime
⚠️ Monitoring: 0 alerts triggered
🔄 Auto-recovery: 3 successful restarts
```

---

## 11_streaming_and_tika_demo.py
**🎯 Level**: Advanced | **⏱️ Duration**: 20-30 minutes | **📋 Prerequisites**: 10_enhanced_streaming_demo.py

### Purpose
Document processing with Apache Tika integration and streaming capabilities.

### What You'll Learn
- Apache Tika integration patterns
- Multi-format document processing
- Streaming document analysis
- Metadata extraction workflows

### Key Demonstrations
- Tika-powered content extraction
- Streaming document processing
- Metadata preservation
- Format-agnostic workflows

### When to Use This
- Processing diverse document formats
- Enterprise document workflows
- Content management systems

### Sample Output
```
📄 TIKA STREAMING DEMO
=====================================
🔍 Detected formats: PDF, DOCX, PPTX
📊 Extracted metadata: 127 fields
🌊 Streaming extraction: 45.3MB/s
✅ All formats processed successfully
```

---

## 12_parallelization_demo.py
**🎯 Level**: Advanced | **⏱️ Duration**: 25-35 minutes | **📋 Prerequisites**: 07_metrics_collection_demo.py

### Purpose
Multi-core processing optimization and parallel execution patterns.

### What You'll Learn
- Parallel processing implementation
- Multi-core optimization techniques
- Load balancing strategies
- Performance scaling analysis

### Key Demonstrations
- Multi-core chunking execution
- Load distribution strategies
- Performance scaling measurement
- Resource utilization optimization

### When to Use This
- High-performance computing
- Batch processing optimization
- Multi-core system utilization

### Sample Output
```
⚡ PARALLELIZATION DEMO
=====================================
🖥️ Cores utilized: 8/8 (100%)
📈 Speedup achieved: 6.7x
⚖️ Load balance: 94.3% efficiency
🚀 Throughput: 15.2 MB/s per core
```

---

## 13_smart_parallelization_demo.py
**🎯 Level**: Advanced | **⏱️ Duration**: 30-40 minutes | **📋 Prerequisites**: 12_parallelization_demo.py

### Purpose
Intelligent parallelization with adaptive load balancing and optimization.

### What You'll Learn
- Adaptive parallelization algorithms
- Dynamic load balancing
- Resource-aware processing
- Intelligent scheduling strategies

### Key Demonstrations
- Smart workload distribution
- Adaptive core allocation
- Dynamic optimization
- Resource monitoring integration

### When to Use This
- Variable workload processing
- Resource-constrained environments
- Intelligent system optimization

### Sample Output
```
🧠 SMART PARALLELIZATION DEMO
=====================================
🔄 Adaptive scheduling: enabled
📊 Dynamic load balancing: 97.8%
🎯 Resource optimization: 23% improvement
⚡ Intelligence gain: 1.4x over static
```

---

## 14_python_all_strategies_demo.py
**🎯 Level**: Expert | **⏱️ Duration**: 45-60 minutes | **📋 Prerequisites**: 13_smart_parallelization_demo.py

### Purpose
Comprehensive exploration of all available chunking strategies.

### What You'll Learn
- Complete strategy landscape
- Comparative analysis methods
- Strategy selection criteria
- Performance characterization

### Key Demonstrations
- All strategies tested and compared
- Performance matrix generation
- Quality assessment across strategies
- Decision-making frameworks

### When to Use This
- Strategy selection guidance
- Comprehensive system evaluation
- Research and analysis projects

### Sample Output
```
🔍 ALL STRATEGIES ANALYSIS
=====================================
📊 Strategies tested: 40+
⚡ Performance range: 0.012s - 2.347s
🎯 Quality range: 0.234 - 0.967
🏆 Best overall: semantic (balanced)
📈 Recommendation matrix generated
```

---

## 15_comprehensive_directory_processing_demo.py
**🎯 Level**: Expert | **⏱️ Duration**: 60+ minutes | **📋 Prerequisites**: 14_python_all_strategies_demo.py

### Purpose
Large-scale batch processing and enterprise directory workflows.

### What You'll Learn
- Enterprise-scale processing patterns
- Directory traversal optimization
- Batch operation error handling
- Progress tracking and reporting

### Key Demonstrations
- Large directory batch processing
- File type detection and routing
- Progress monitoring systems
- Comprehensive error handling

### When to Use This
- Enterprise content processing
- Large-scale data migration
- Content management systems

### Sample Output
```
🏢 ENTERPRISE PROCESSING DEMO
=====================================
📁 Directories processed: 1,247
📄 Files analyzed: 15,843
⏱️ Total processing time: 2h 34m
✅ Success rate: 99.2%
📊 Generated report: processing_summary.json
```

---

## 16_verify_embedding_setup.py
**🎯 Level**: Utility | **⏱️ Duration**: 5-10 minutes | **📋 Prerequisites**: None

### Purpose
Setup verification and environment validation utility.

### What You'll Learn
- Environment setup validation
- Dependency checking
- Configuration verification
- Troubleshooting guidance

### Key Demonstrations
- Library installation verification
- Model availability checking
- Configuration validation
- Performance baseline establishment

### When to Use This
- Initial setup verification
- Troubleshooting environment issues
- Pre-deployment validation

### Sample Output
```
✅ SETUP VERIFICATION
=====================================
📦 Library installation: ✅ OK
🧠 Embedding models: ✅ Available
⚙️ Configuration: ✅ Valid
🚀 Performance baseline: 0.123s
🎯 System ready for production
```

---

## 🔧 Custom Algorithms Directory

The `custom_algorithms/` directory contains example custom chunking strategies:

### Available Examples
- **`balanced_length_chunker.py`**: Maintains consistent chunk sizes
- **`regex_pattern_chunker.py`**: Pattern-based chunking with regex
- **`sentiment_based_chunker.py`**: Sentiment-aware content grouping

### Development Workflow
1. Study existing examples in `custom_algorithms/`
2. Follow the `BaseChunker` interface pattern
3. Test with `08_extensibility_demo.py`
4. Validate with `16_verify_embedding_setup.py`

---

## 🏃‍♂️ Running Examples

### Individual Execution
```bash
cd examples
python 01_basic_usage.py
python 05_embeddings_integration_demo.py
```

### Sequential Learning Path
```bash
# Follow the complete learning path
for i in {01..16}; do
    echo "Running example $i..."
    python ${i}_*.py
    echo "Completed example $i"
    echo "---"
done
```

### Category-Based Execution
```bash
# Beginner examples only
python 01_*.py 02_*.py 03_*.py 04_*.py

# Advanced streaming examples
python 06_*.py 10_*.py 11_*.py

# Performance optimization examples
python 07_*.py 12_*.py 13_*.py
```

## 🔍 Troubleshooting

### Quick Diagnostics
```bash
# Verify setup
python 16_verify_embedding_setup.py

# Test basic functionality
python 01_basic_usage.py

# Check embedding capabilities
python 03_embedding_workflows.py
```

### Common Issues
1. **Import Errors**: Ensure `pip install -e .` was run
2. **Memory Issues**: Start with streaming examples (06, 10, 11)
3. **Performance Issues**: Run metrics collection (07) for analysis
4. **Configuration Errors**: Use CLI examples (04) for validation

### Getting Help
- Start with `16_verify_embedding_setup.py` for environment validation
- Follow examples in numerical order for best learning experience
- Check the main README.md for library overview
- Review DEBUGGING_GUIDE.md for detailed troubleshooting

## 📞 Support & Next Steps

After completing the examples:
1. **Review Configuration**: Explore `../config_examples/` for production configs
2. **Read Advanced Guides**: Check CUSTOM_ALGORITHMS_GUIDE.md
3. **Join Development**: See CONTRIBUTING.md for contribution guidelines
4. **Production Deployment**: Review production examples (14-15)

---

**Happy Chunking! 🚀**

*This example collection provides everything needed to master the chunking library, from basic concepts to expert-level production scenarios.*