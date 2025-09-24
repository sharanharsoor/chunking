# ⚡ Quick Start Guide

Get up and running with the chunking-strategy library in under 5 minutes!

## 🚀 Installation (30 seconds)

```bash
# Full installation with all features
pip install chunking-strategy[all]

# Or basic installation
pip install chunking-strategy
```

## 📝 Your First Chunks (2 minutes)

### Option 1: Python API (Recommended)

```python
from chunking_strategy import create_chunker

# Create a chunker
chunker = create_chunker("sentence_based")

# Chunk some text
result = chunker.chunk("""
This is your first document. It has multiple sentences.
Each sentence will become part of a chunk. The chunker
will group sentences intelligently for better processing.
""")

# See your chunks
print(f"Created {len(result.chunks)} chunks:")
for i, chunk in enumerate(result.chunks, 1):
    print(f"Chunk {i}: {chunk.content.strip()}")
```

**Expected Output:**
```
Created 2 chunks:
Chunk 1: This is your first document. It has multiple sentences.
Chunk 2: Each sentence will become part of a chunk. The chunker will group sentences intelligently for better processing.
```

### Option 2: CLI (Super Quick)

```bash
# Create a test file
echo "Hello world. This is a test document. It has multiple sentences for chunking." > test.txt

# Chunk it
python -m chunking_strategy chunk test.txt --strategy sentence_based

# You'll see the chunks printed to console!
```

## 🎯 Real File Processing (1 minute)

### Process Any File Type

```python
from chunking_strategy import ChunkerOrchestrator

# Auto-detects best strategy for any file
orchestrator = ChunkerOrchestrator()

# Works with ANY file type!
result = orchestrator.chunk_file("your_document.pdf")  # PDF
result = orchestrator.chunk_file("your_code.py")       # Python code
result = orchestrator.chunk_file("your_data.json")     # JSON data
result = orchestrator.chunk_file("your_article.txt")   # Text file

print(f"✅ Processed with strategy: {result.strategy_used}")
print(f"📊 Created {len(result.chunks)} chunks")
```

### CLI File Processing

```bash
# Process any file - auto-detects best strategy
python -m chunking_strategy chunk your_document.pdf

# Or specify strategy
python -m chunking_strategy chunk code.py --strategy python_code

# Process multiple files
python -m chunking_strategy batch *.txt --strategy paragraph_based
```

## 🔍 Explore Available Strategies (30 seconds)

```python
from chunking_strategy import list_chunkers

# See all available strategies
strategies = list_chunkers()
print(f"📚 Available strategies: {len(strategies)}")
for strategy in strategies[:10]:  # Show first 10
    print(f"  • {strategy}")
```

Or via CLI:
```bash
python -m chunking_strategy list-strategies
```

## 🎭 Choose Your Approach (1 minute)

### 🤖 Auto (Zero Config) - **Recommended for Beginners**
```python
# Just works - no configuration needed!
orchestrator = ChunkerOrchestrator()
result = orchestrator.chunk_file("any_file.ext")
```

### 🎯 Specific Strategy - **For Targeted Use Cases**
```python
# Text documents
chunker = create_chunker("sentence_based", max_sentences=3)
result = chunker.chunk("your text")

# PDF documents
chunker = create_chunker("pdf_chunker", extract_images=True)
result = chunker.chunk("document.pdf")

# Code files
chunker = create_chunker("python_code")
result = chunker.chunk("script.py")
```

### ⚡ CLI Processing - **For Command Line Users**
```bash
# Quick one-liner
python -m chunking_strategy chunk document.pdf --format json --output chunks.json
```

## 📊 Check Your Results (30 seconds)

```python
# Examine your chunking result
result = chunker.chunk("content here")

print(f"📈 Processing time: {result.processing_time:.2f}s")
print(f"📝 Strategy used: {result.strategy_used}")
print(f"📊 Total chunks: {len(result.chunks)}")

# Look at individual chunks
for chunk in result.chunks[:3]:  # First 3 chunks
    print(f"🔍 Chunk ID: {chunk.id}")
    print(f"📄 Content: {chunk.content[:100]}...")
    print(f"📋 Metadata: {chunk.metadata.word_count} words")
    print("---")
```

## ✅ Success! What's Next?

**🎉 Congratulations!** You've successfully chunked your first content. Here's what to explore next:

### 📚 **Learn More:**
- [Full Documentation](README.md) - Comprehensive guide with all features
- [CLI Reference](CLI_REFERENCE.md) - Complete command-line documentation
- [API Reference](API_REFERENCE.md) - Detailed API documentation

### 🚀 **Try Advanced Features:**
```bash
# Run example demos
python examples/01_basic_usage.py              # Basic usage patterns
python examples/02_advanced_usage.py           # Advanced configurations
python examples/03_embedding_workflows.py      # Generate embeddings

# Interactive web app
pip install streamlit
streamlit run examples/19_streamlit_app_demo.py
```

### 🎯 **Common Next Steps:**

1. **For RAG/Vector Database Users:**
   ```python
   chunker = create_chunker("sentence_based", max_sentences=3, overlap_sentences=1)
   ```

2. **For Document Analysis:**
   ```python
   chunker = create_chunker("paragraph_based", max_paragraphs=2)
   ```

3. **For Consistent Chunk Sizes:**
   ```python
   chunker = create_chunker("fixed_size", chunk_size=1000, overlap_size=100)
   ```

4. **For Code Analysis:**
   ```python
   chunker = create_chunker("python_code")  # or javascript_chunker, java_chunker
   ```

### 🆘 **Need Help?**
- 🐛 **Issues**: [GitHub Issues](https://github.com/sharanharsoor/chunking/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/sharanharsoor/chunking/discussions)
- 📖 **Examples**: Check out `/examples/` directory (23+ demos!)

---

**🚀 You're ready to chunk like a pro!** The library handles all the complexity - you just focus on your application.
