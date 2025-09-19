#!/usr/bin/env python3
"""
Example: How to USE integration_helpers.py in your own projects

This shows how developers would actually use the integration_helpers.py utility library
in their own applications, rather than the standalone demos.
"""

import sys
from pathlib import Path

# Add parent directory to Python path for local development
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# This is how a developer would use integration_helpers.py in their own project
from integration_helpers import (
    LangChainConverter,
    StreamlitHelpers,
    RAGHelpers,
    BatchProcessor,
    quick_rag_setup
)
from chunking_strategy import create_chunker
import tempfile

def example_1_simple_langchain_conversion():
    """Example 1: Quick LangChain conversion using helpers"""
    print("\n📄 EXAMPLE 1: Simple LangChain Conversion")
    print("=" * 50)

    # Step 1: Create some content and chunk it
    content = """
    AI and Machine Learning Guide

    Artificial Intelligence is transforming industries worldwide.

    Machine learning enables computers to learn from data without explicit programming.

    Deep learning uses neural networks to solve complex problems.
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_file = f.name

    # Step 2: Chunk the content
    chunker = create_chunker('paragraph_based', max_paragraphs=1)
    result = chunker.chunk(temp_file)
    print(f"Created {len(result.chunks)} chunks")

    # Step 3: Use integration_helpers to convert to LangChain Documents
    try:
        langchain_docs = LangChainConverter.chunks_to_documents(result.chunks)
        print(f"✅ Converted to {len(langchain_docs)} LangChain Documents using integration_helpers!")

        # Show what we get
        for i, doc in enumerate(langchain_docs):
            print(f"   Doc {i+1}: {len(doc.page_content)} chars, metadata keys: {list(doc.metadata.keys())}")

    except Exception as e:
        print(f"⚠️ LangChain conversion skipped: {e}")

def example_2_rag_pipeline():
    """Example 2: Quick RAG setup using helpers"""
    print("\n🔍 EXAMPLE 2: Quick RAG Pipeline Setup")
    print("=" * 50)

    # Create multiple documents
    documents = [
        ("Python is a programming language known for its simplicity.", "python_intro.txt"),
        ("Machine learning helps computers learn from data automatically.", "ml_basics.txt"),
        ("Data science combines statistics, programming, and domain expertise.", "data_science.txt")
    ]

    temp_files = []
    for content, filename in documents:
        with tempfile.NamedTemporaryFile(mode='w', suffix=filename, delete=False) as f:
            f.write(content)
            temp_files.append(f.name)

    try:
        # Use integration_helpers for quick RAG setup
        print("Using integration_helpers to create document store...")
        all_chunks = RAGHelpers.create_document_store(temp_files, chunker_strategy="paragraph_based")
        print(f"✅ Created document store with {len(all_chunks)} chunks")

        # Search using helpers
        python_chunks = RAGHelpers.search_chunks_by_content(all_chunks, "python", case_sensitive=False)
        print(f"✅ Found {len(python_chunks)} chunks about Python")

        ml_chunks = RAGHelpers.search_chunks_by_content(all_chunks, "machine learning")
        print(f"✅ Found {len(ml_chunks)} chunks about machine learning")

    finally:
        # Cleanup
        import os
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

def example_3_batch_processing():
    """Example 3: Batch processing using helpers"""
    print("\n📦 EXAMPLE 3: Batch Processing with Helpers")
    print("=" * 50)

    # Create a temporary directory with multiple files
    import os
    temp_dir = tempfile.mkdtemp()

    files_to_create = [
        ("file1.txt", "This is the first document about Python programming."),
        ("file2.md", "# Markdown Document\nThis is a markdown file about data science."),
        ("file3.txt", "The third document discusses machine learning algorithms.")
    ]

    for filename, content in files_to_create:
        with open(os.path.join(temp_dir, filename), 'w') as f:
            f.write(content)

    try:
        # Use integration_helpers for batch processing
        print(f"Processing directory: {temp_dir}")
        processor = BatchProcessor(chunker_strategy="auto")  # Auto-detect strategy

        results = processor.process_directory(
            temp_dir,
            file_extensions=['.txt', '.md'],
            recursive=False
        )

        print(f"✅ Batch processed {len(results['processed_files'])} files")
        print(f"✅ Total chunks created: {results['total_chunks']}")
        print(f"✅ Total processing time: {results['total_processing_time']:.3f}s")

        # Show results
        for file_result in results['processed_files']:
            print(f"   📄 {os.path.basename(file_result['file_path'])}: {file_result['chunks_created']} chunks ({file_result['strategy_used']})")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

def example_4_streamlit_helpers():
    """Example 4: How to use Streamlit helpers in your own Streamlit app"""
    print("\n📱 EXAMPLE 4: Streamlit Helpers Usage")
    print("=" * 50)

    print("This is how you'd use StreamlitHelpers in your own Streamlit app:")
    print("""
# In your_streamlit_app.py:
import streamlit as st
from integration_helpers import StreamlitHelpers

# File upload
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file:
    # Use helper to process file
    chunks, processing_info = StreamlitHelpers.process_uploaded_file(
        uploaded_file,
        chunker_strategy="auto"
    )

    if chunks:
        # Use helper to display metrics
        StreamlitHelpers.display_chunk_metrics(chunks, processing_info)

        # Use helper to create chunk selector
        selected_idx = StreamlitHelpers.create_chunk_selector(chunks)

        if selected_idx is not None:
            st.text_area("Selected Chunk", chunks[selected_idx].content)
    """)

    print("✅ Integration helpers make Streamlit development much easier!")

def main():
    """Main demo showing how to use integration_helpers.py"""
    print("🧩 HOW TO USE INTEGRATION_HELPERS.PY")
    print("=" * 60)
    print("This shows how developers use integration_helpers.py in their own projects")

    example_1_simple_langchain_conversion()
    example_2_rag_pipeline()
    example_3_batch_processing()
    example_4_streamlit_helpers()

    print("\n" + "=" * 60)
    print("🎉 SUMMARY:")
    print("- integration_helpers.py provides UTILITY FUNCTIONS")
    print("- Use it to build YOUR OWN applications")
    print("- The demo files (18_, 19_) show what's POSSIBLE")
    print("- This file shows how to ACTUALLY USE the helpers")
    print("=" * 60)

if __name__ == "__main__":
    main()
