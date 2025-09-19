#!/usr/bin/env python3
"""
Integration Helpers for chunking-strategy

This module provides utility functions and classes to make it easier to integrate
chunking-strategy with popular frameworks like LangChain, Streamlit, and others.

Features:
- LangChain Document converters
- Streamlit file upload handlers
- RAG pipeline helpers
- Vector store utilities
- Batch processing utilities
"""

from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import tempfile
import os
import sys

# Add parent directory to Python path for local development
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

class ChunkingIntegrationError(Exception):
    """Custom exception for integration-related errors."""
    pass

class LangChainConverter:
    """Utility class for converting between chunking-strategy and LangChain formats."""

    @staticmethod
    def chunks_to_documents(chunks, include_metadata: bool = True) -> List:
        """
        Convert chunking-strategy chunks to LangChain Documents.

        Args:
            chunks: List of chunks from chunking-strategy
            include_metadata: Whether to include chunk metadata

        Returns:
            List of LangChain Document objects
        """
        try:
            from langchain.schema import Document
        except ImportError:
            try:
                from langchain_core.documents import Document
            except ImportError:
                raise ChunkingIntegrationError(
                    "LangChain not installed. Run: pip install langchain"
                )

        documents = []
        for chunk in chunks:
            metadata = {}

            if include_metadata:
                metadata.update({
                    "source": getattr(chunk.metadata, 'source', 'unknown'),
                    "chunk_id": chunk.id,
                    "chunker_used": getattr(chunk.metadata, 'chunker_used', 'unknown'),
                    "length": len(chunk.content),
                    "page": getattr(chunk.metadata, 'page', None),
                    "source_type": getattr(chunk.metadata, 'source_type', 'unknown')
                })

                # Add any extra metadata
                extra = getattr(chunk.metadata, 'extra', {})
                if isinstance(extra, dict):
                    metadata.update(extra)

            doc = Document(
                page_content=chunk.content,
                metadata=metadata
            )
            documents.append(doc)

        return documents

    @staticmethod
    def create_retrieval_pipeline(chunks, embedding_model: str = "all-MiniLM-L6-v2") -> tuple:
        """
        Create a complete retrieval pipeline from chunks.

        Args:
            chunks: List of chunks from chunking-strategy
            embedding_model: Name of the embedding model to use

        Returns:
            Tuple of (vectorstore, retriever)
        """
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.vectorstores import Chroma
        except ImportError:
            raise ChunkingIntegrationError(
                "Required packages not installed. Run: pip install langchain chromadb sentence-transformers"
            )

        # Convert chunks to documents
        documents = LangChainConverter.chunks_to_documents(chunks)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{embedding_model}",
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
        )

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        return vectorstore, retriever

    @staticmethod
    def create_qa_chain(retriever, llm=None):
        """
        Create a question-answering chain using the retriever.

        Args:
            retriever: LangChain retriever object
            llm: Language model (optional, uses OpenAI if available)

        Returns:
            QA chain object
        """
        try:
            from langchain.chains import RetrievalQA
            from langchain.llms import OpenAI
        except ImportError:
            raise ChunkingIntegrationError(
                "LangChain not fully installed. Run: pip install langchain"
            )

        if llm is None:
            try:
                llm = OpenAI()
            except Exception:
                raise ChunkingIntegrationError(
                    "No LLM provided and OpenAI not configured. Please provide an LLM instance."
                )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        return qa_chain

class StreamlitHelpers:
    """Utility functions for Streamlit integration."""

    @staticmethod
    def process_uploaded_file(uploaded_file, chunker_strategy: str = "auto", **chunker_params):
        """
        Process an uploaded Streamlit file and return chunks.

        Args:
            uploaded_file: Streamlit UploadedFile object
            chunker_strategy: Strategy to use for chunking
            **chunker_params: Additional parameters for the chunker

        Returns:
            Tuple of (chunks, processing_info)
        """
        from chunking_strategy import create_chunker
        import time

        if uploaded_file is None:
            return None, None

        # Read file content
        try:
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
        except Exception as e:
            raise ChunkingIntegrationError(f"Error reading uploaded file: {e}")

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{uploaded_file.name}', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Process with chunker
            start_time = time.time()

            if chunker_strategy == "auto":
                # Auto-detect strategy based on file extension
                file_ext = Path(uploaded_file.name).suffix.lower()
                if file_ext == '.py':
                    chunker_strategy = 'python_code'
                elif file_ext == '.md':
                    chunker_strategy = 'markdown_chunker'
                elif file_ext == '.pdf':
                    chunker_strategy = 'pdf_chunker'
                else:
                    chunker_strategy = 'paragraph_based'

            chunker = create_chunker(chunker_strategy, **chunker_params)
            result = chunker.chunk(temp_file)

            processing_time = time.time() - start_time

            processing_info = {
                'strategy_used': chunker_strategy,
                'processing_time': processing_time,
                'total_chunks': len(result.chunks),
                'total_characters': sum(len(chunk.content) for chunk in result.chunks),
                'file_name': uploaded_file.name,
                'file_size': len(content)
            }

            return result.chunks, processing_info

        finally:
            # Clean up
            try:
                os.unlink(temp_file)
            except:
                pass

    @staticmethod
    def display_chunk_metrics(chunks, processing_info=None):
        """
        Display chunk metrics in Streamlit format.

        Args:
            chunks: List of chunks
            processing_info: Processing information dictionary
        """
        try:
            import streamlit as st
        except ImportError:
            raise ChunkingIntegrationError("Streamlit not installed. Run: pip install streamlit")

        if not chunks:
            st.warning("No chunks to display")
            return

        # Calculate statistics
        lengths = [len(chunk.content) for chunk in chunks]
        stats = {
            'total_chunks': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_chars': sum(lengths)
        }

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Chunks", stats['total_chunks'])

        with col2:
            st.metric("Avg Length", f"{stats['avg_length']:.0f}")

        with col3:
            st.metric("Total Characters", f"{stats['total_chars']:,}")

        with col4:
            if processing_info and 'processing_time' in processing_info:
                st.metric("Processing Time", f"{processing_info['processing_time']:.3f}s")
            else:
                st.metric("Min/Max Length", f"{stats['min_length']}/{stats['max_length']}")

    @staticmethod
    def create_chunk_selector(chunks, key: str = "chunk_selector"):
        """
        Create a Streamlit selectbox for choosing chunks.

        Args:
            chunks: List of chunks
            key: Unique key for the selectbox

        Returns:
            Selected chunk index
        """
        try:
            import streamlit as st
        except ImportError:
            raise ChunkingIntegrationError("Streamlit not installed. Run: pip install streamlit")

        if not chunks:
            return None

        options = [f"Chunk {i+1} ({len(chunk.content)} chars)" for i, chunk in enumerate(chunks)]

        selected = st.selectbox(
            "Select a chunk to view:",
            range(len(chunks)),
            format_func=lambda x: options[x],
            key=key
        )

        return selected

class RAGHelpers:
    """Utility functions for RAG (Retrieval-Augmented Generation) workflows."""

    @staticmethod
    def create_document_store(file_paths: List[Union[str, Path]], chunker_strategy: str = "auto"):
        """
        Create a document store from multiple files.

        Args:
            file_paths: List of file paths to process
            chunker_strategy: Strategy to use for chunking

        Returns:
            List of all chunks from all documents
        """
        from chunking_strategy import create_chunker

        all_chunks = []

        for file_path in file_paths:
            file_path = Path(file_path)

            # Auto-detect strategy if needed
            if chunker_strategy == "auto":
                ext = file_path.suffix.lower()
                if ext == '.py':
                    strategy = 'python_code'
                elif ext == '.md':
                    strategy = 'markdown_chunker'
                elif ext == '.pdf':
                    strategy = 'pdf_chunker'
                else:
                    strategy = 'paragraph_based'
            else:
                strategy = chunker_strategy

            # Process file
            chunker = create_chunker(strategy)
            result = chunker.chunk(str(file_path))
            all_chunks.extend(result.chunks)

        return all_chunks

    @staticmethod
    def filter_chunks_by_metadata(chunks, filter_func: Callable) -> List:
        """
        Filter chunks based on metadata criteria.

        Args:
            chunks: List of chunks to filter
            filter_func: Function that takes a chunk and returns True/False

        Returns:
            Filtered list of chunks
        """
        return [chunk for chunk in chunks if filter_func(chunk)]

    @staticmethod
    def search_chunks_by_content(chunks, query: str, case_sensitive: bool = False) -> List:
        """
        Search chunks by content using simple string matching.

        Args:
            chunks: List of chunks to search
            query: Search query string
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of matching chunks
        """
        if not case_sensitive:
            query = query.lower()

        matching_chunks = []
        for chunk in chunks:
            content = chunk.content if case_sensitive else chunk.content.lower()
            if query in content:
                matching_chunks.append(chunk)

        return matching_chunks

class BatchProcessor:
    """Utility class for batch processing multiple files."""

    def __init__(self, chunker_strategy: str = "auto", **chunker_params):
        """
        Initialize batch processor.

        Args:
            chunker_strategy: Default strategy to use
            **chunker_params: Default parameters for chunkers
        """
        self.chunker_strategy = chunker_strategy
        self.chunker_params = chunker_params
        self.results = []

    def process_directory(self, directory_path: Union[str, Path],
                         file_extensions: Optional[List[str]] = None,
                         recursive: bool = True) -> Dict[str, Any]:
        """
        Process all files in a directory.

        Args:
            directory_path: Path to directory
            file_extensions: List of extensions to process (e.g., ['.txt', '.md'])
            recursive: Whether to process subdirectories

        Returns:
            Dictionary with processing results
        """
        from chunking_strategy import create_chunker
        import time

        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise ChunkingIntegrationError(f"Directory not found: {directory_path}")

        # Find files to process
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        files_to_process = []
        for file_path in directory_path.glob(pattern):
            if file_path.is_file():
                if file_extensions is None or file_path.suffix.lower() in file_extensions:
                    files_to_process.append(file_path)

        # Process files
        results = {
            'processed_files': [],
            'failed_files': [],
            'total_chunks': 0,
            'total_processing_time': 0,
            'all_chunks': []
        }

        for file_path in files_to_process:
            try:
                start_time = time.time()

                # Determine strategy
                strategy = self.chunker_strategy
                if strategy == "auto":
                    ext = file_path.suffix.lower()
                    if ext == '.py':
                        strategy = 'python_code'
                    elif ext == '.md':
                        strategy = 'markdown_chunker'
                    elif ext == '.pdf':
                        strategy = 'pdf_chunker'
                    else:
                        strategy = 'paragraph_based'

                # Process file
                chunker = create_chunker(strategy, **self.chunker_params)
                result = chunker.chunk(str(file_path))

                processing_time = time.time() - start_time

                file_result = {
                    'file_path': str(file_path),
                    'strategy_used': strategy,
                    'chunks_created': len(result.chunks),
                    'processing_time': processing_time,
                    'success': True
                }

                results['processed_files'].append(file_result)
                results['total_chunks'] += len(result.chunks)
                results['total_processing_time'] += processing_time
                results['all_chunks'].extend(result.chunks)

            except Exception as e:
                error_result = {
                    'file_path': str(file_path),
                    'error': str(e),
                    'success': False
                }
                results['failed_files'].append(error_result)

        return results

def quick_rag_setup(file_paths: List[Union[str, Path]],
                   embedding_model: str = "all-MiniLM-L6-v2") -> tuple:
    """
    Quick setup function for a basic RAG pipeline.

    Args:
        file_paths: List of files to process
        embedding_model: Embedding model to use

    Returns:
        Tuple of (vectorstore, retriever, chunks)
    """
    # Process documents
    all_chunks = RAGHelpers.create_document_store(file_paths)

    # Create retrieval pipeline
    vectorstore, retriever = LangChainConverter.create_retrieval_pipeline(
        all_chunks, embedding_model
    )

    return vectorstore, retriever, all_chunks

def demo_integration_helpers():
    """Demonstrate the integration helpers."""
    print("🚀 Integration Helpers Demo")
    print("=" * 50)

    try:
        from chunking_strategy import create_chunker

        # Create sample content
        sample_content = """
        # Python Programming Guide

        Python is a versatile programming language that's great for beginners and experts alike.

        ## Getting Started

        To start with Python, you'll need to install it on your system. Visit python.org for downloads.

        ## Basic Syntax

        Python uses indentation to define code blocks. Here's a simple example:

        ```python
        def greet(name):
            return f"Hello, {name}!"

        print(greet("World"))
        ```
        """

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_content)
            temp_file = f.name

        try:
            # Test chunking
            chunker = create_chunker('markdown_chunker')
            result = chunker.chunk(temp_file)

            print(f"✅ Created {len(result.chunks)} chunks")

            # Test LangChain conversion
            print("\n📄 Testing LangChain conversion...")
            documents = LangChainConverter.chunks_to_documents(result.chunks)
            print(f"✅ Converted to {len(documents)} LangChain documents")

            # Test batch processing
            print("\n📦 Testing batch processing...")
            processor = BatchProcessor()
            # For demo, just process our single file
            batch_results = processor.process_directory(
                Path(temp_file).parent,
                file_extensions=['.md'],
                recursive=False
            )
            print(f"✅ Batch processed {len(batch_results['processed_files'])} files")

            print("\n🎉 All integration helpers working correctly!")

        finally:
            os.unlink(temp_file)

    except Exception as e:
        print(f"❌ Error in demo: {e}")

if __name__ == "__main__":
    demo_integration_helpers()
