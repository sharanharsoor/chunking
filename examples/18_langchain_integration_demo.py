#!/usr/bin/env python3
"""
Comprehensive LangChain Integration Demo

This demo shows how to integrate chunking-strategy with LangChain for various use cases:
1. Basic Document conversion
2. RAG pipeline integration
3. Vector store integration
4. Custom retrieval workflows
5. Question-answering with chunked documents

Prerequisites:
    pip install langchain langchain-community chromadb
"""

import os
import tempfile
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to Python path for local development
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import langchain
    except ImportError:
        missing.append("langchain")

    try:
        import chromadb
    except ImportError:
        missing.append("chromadb")

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False

    print("✅ All dependencies available!")
    return True

def demo_basic_document_conversion():
    """Demo 1: Basic conversion from chunking-strategy to LangChain Documents."""
    print("\n" + "="*60)
    print("📝 DEMO 1: Basic Document Conversion")
    print("="*60)

    from chunking_strategy import create_chunker
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain_core.documents import Document

    # Create sample content
    sample_content = """
    # Machine Learning Fundamentals

    Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.

    ## Types of Machine Learning

    ### Supervised Learning
    Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. Common examples include classification and regression tasks.

    ### Unsupervised Learning
    Unsupervised learning finds hidden patterns in data without labeled examples. Clustering and dimensionality reduction are typical applications.

    ### Reinforcement Learning
    Reinforcement learning trains agents to make sequences of decisions through trial and error, using rewards and penalties.
    """

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(sample_content)
        temp_file = f.name

    try:
        # Chunk with different strategies
        strategies = ['sentence_based', 'paragraph_based', 'markdown_chunker']

        for strategy in strategies:
            print(f"\n🔧 Using {strategy} strategy:")

            chunker = create_chunker(strategy, max_sentences=3 if strategy == 'sentence_based' else None)
            result = chunker.chunk(temp_file)

            # Convert to LangChain Documents
            langchain_docs = [
                Document(
                    page_content=chunk.content,
                    metadata={
                        "source": chunk.metadata.source,
                        "chunk_id": chunk.id,
                        "chunker_used": chunk.metadata.chunker_used,
                        "length": chunk.metadata.length,
                        "strategy": strategy
                    }
                )
                for chunk in result.chunks
            ]

            print(f"   ✅ Created {len(langchain_docs)} LangChain Documents")

            # Show first document
            if langchain_docs:
                doc = langchain_docs[0]
                print(f"   📄 First document: {len(doc.page_content)} chars")
                print(f"   🏷️  Metadata keys: {list(doc.metadata.keys())}")
                print(f"   📝 Preview: \"{doc.page_content[:100]}...\"")

    finally:
        os.unlink(temp_file)

def demo_rag_pipeline():
    """Demo 2: Complete RAG pipeline with vector store."""
    print("\n" + "="*60)
    print("🔍 DEMO 2: RAG Pipeline Integration")
    print("="*60)

    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.text_splitter import CharacterTextSplitter
        from langchain_core.documents import Document
        from chunking_strategy import create_chunker

        # Create knowledge base content
        knowledge_base = [
            "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",

            "Machine learning algorithms can be broadly categorized into supervised, unsupervised, and reinforcement learning. Each category serves different purposes and uses different approaches to learn from data.",

            "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It involves techniques for analyzing, understanding, and generating human language.",

            "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has revolutionized fields like computer vision, speech recognition, and natural language processing.",

            "Data preprocessing is a crucial step in machine learning that involves cleaning, transforming, and preparing raw data for analysis. It often takes 80% of the time in ML projects."
        ]

        # Create temporary files
        temp_files = []
        for i, content in enumerate(knowledge_base):
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_doc_{i}.txt', delete=False) as f:
                f.write(content)
                temp_files.append(f.name)

        try:
            # Chunk all documents using our library
            chunker = create_chunker('sentence_based', max_sentences=2)
            all_chunks = []

            for file_path in temp_files:
                result = chunker.chunk(file_path)
                all_chunks.extend(result.chunks)

            print(f"📊 Processed {len(temp_files)} documents into {len(all_chunks)} chunks")

            # Convert to LangChain Documents with enhanced metadata
            documents = []
            for i, chunk in enumerate(all_chunks):
                doc = Document(
                    page_content=chunk.content,
                    metadata={
                        "source": os.path.basename(chunk.metadata.source),
                        "chunk_id": chunk.id,
                        "doc_index": i // 2,  # Approximate original document
                        "chunker": chunk.metadata.chunker_used,
                        "length": chunk.metadata.length
                    }
                )
                documents.append(doc)

            # Initialize embeddings (using a small, fast model for demo)
            print("🔧 Initializing embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            # Create vector store
            print("🗂️  Creating vector store...")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=None  # In-memory for demo
            )

            print("✅ Vector store created successfully!")

            # Demo retrieval
            queries = [
                "What is Python programming?",
                "Tell me about machine learning types",
                "How does deep learning work?"
            ]

            print("\n🔍 Testing retrieval:")
            for query in queries:
                print(f"\n   Query: '{query}'")
                docs = vectorstore.similarity_search(query, k=2)

                for j, doc in enumerate(docs):
                    print(f"   📄 Result {j+1}: {doc.metadata['source']} (chunk: {doc.metadata['chunk_id']})")
                    print(f"      Preview: \"{doc.page_content[:80]}...\"")

        finally:
            # Cleanup
            for file_path in temp_files:
                try:
                    os.unlink(file_path)
                except:
                    pass

    except ImportError as e:
        print(f"❌ Missing dependencies for RAG demo: {e}")
        print("Install with: pip install sentence-transformers chromadb")

def demo_custom_retrieval_workflow():
    """Demo 3: Custom retrieval workflow with metadata filtering."""
    print("\n" + "="*60)
    print("🎯 DEMO 3: Custom Retrieval Workflow")
    print("="*60)

    try:
        from langchain_core.documents import Document
        from chunking_strategy import create_chunker
        import json

        # Create multi-format documents
        documents_content = {
            "research_paper.txt": """
            Abstract: This paper presents a novel approach to natural language processing using transformer architectures.

            Introduction: Large language models have revolutionized the field of NLP. These models use attention mechanisms to process text.

            Methodology: We employed a transformer-based architecture with 12 attention heads and 768 hidden dimensions.

            Results: Our model achieved state-of-the-art performance on several benchmarks including GLUE and SuperGLUE.

            Conclusion: The proposed approach demonstrates significant improvements in text understanding tasks.
            """,

            "code_documentation.py": '''
            def transformer_attention(query, key, value, mask=None):
                """
                Implements scaled dot-product attention mechanism.

                Args:
                    query: Query tensor of shape (batch_size, seq_len, d_model)
                    key: Key tensor of shape (batch_size, seq_len, d_model)
                    value: Value tensor of shape (batch_size, seq_len, d_model)
                    mask: Optional attention mask

                Returns:
                    Attention output tensor
                """
                # Calculate attention scores
                scores = torch.matmul(query, key.transpose(-2, -1))
                scores = scores / math.sqrt(query.size(-1))

                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)

                attention_weights = F.softmax(scores, dim=-1)
                output = torch.matmul(attention_weights, value)

                return output, attention_weights
            ''',

            "tutorial.md": '''
            # Attention Mechanisms Tutorial

            ## What is Attention?
            Attention allows models to focus on specific parts of the input when making predictions.

            ## Types of Attention
            - Self-attention: Relates different positions within the same sequence
            - Cross-attention: Relates positions between different sequences

            ## Implementation Tips
            1. Use proper scaling to prevent vanishing gradients
            2. Apply dropout for regularization
            3. Consider using relative position encodings
            '''
        }

        # Create temp files and chunk them
        temp_files = []
        all_chunked_docs = []

        for filename, content in documents_content.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{filename}', delete=False) as f:
                f.write(content)
                temp_files.append(f.name)

            # Choose chunker based on file type
            if filename.endswith('.py'):
                chunker = create_chunker('python_code')
            elif filename.endswith('.md'):
                chunker = create_chunker('markdown_chunker')
            else:
                chunker = create_chunker('paragraph_based')

            result = chunker.chunk(f.name)

            # Convert to LangChain docs with rich metadata
            for chunk in result.chunks:
                doc = Document(
                    page_content=chunk.content,
                    metadata={
                        "source": filename,
                        "file_type": filename.split('.')[-1],
                        "chunk_id": chunk.id,
                        "chunker": chunk.metadata.chunker_used,
                        "length": chunk.metadata.length,
                        "domain": "AI/ML" if "transformer" in chunk.content.lower() or "attention" in chunk.content.lower() else "general"
                    }
                )
                all_chunked_docs.append(doc)

        print(f"📊 Created {len(all_chunked_docs)} documents from {len(documents_content)} source files")

        # Custom retrieval functions
        def filter_by_file_type(docs: List[Document], file_type: str) -> List[Document]:
            return [doc for doc in docs if doc.metadata.get('file_type') == file_type]

        def filter_by_domain(docs: List[Document], domain: str) -> List[Document]:
            return [doc for doc in docs if doc.metadata.get('domain') == domain]

        def search_content(docs: List[Document], query: str) -> List[Document]:
            query_lower = query.lower()
            matching_docs = []
            for doc in docs:
                if query_lower in doc.page_content.lower():
                    matching_docs.append(doc)
            return matching_docs

        # Demo custom retrieval
        print("\n🔍 Custom Retrieval Examples:")

        # 1. Get only Python code
        python_docs = filter_by_file_type(all_chunked_docs, 'py')
        print(f"\n   📝 Python code chunks: {len(python_docs)}")
        if python_docs:
            print(f"      Example: \"{python_docs[0].page_content[:60]}...\"")

        # 2. Get AI/ML domain content
        ai_docs = filter_by_domain(all_chunked_docs, 'AI/ML')
        print(f"\n   🤖 AI/ML domain chunks: {len(ai_docs)}")

        # 3. Search for specific terms
        attention_docs = search_content(all_chunked_docs, "attention")
        print(f"\n   🎯 'Attention' mentions: {len(attention_docs)}")

        # 4. Combined filtering
        python_ai_docs = filter_by_domain(filter_by_file_type(all_chunked_docs, 'py'), 'AI/ML')
        print(f"\n   🔧 Python + AI/ML: {len(python_ai_docs)}")

        # Create a simple similarity function
        def simple_similarity_search(docs: List[Document], query: str, top_k: int = 3) -> List[Document]:
            """Simple keyword-based similarity search."""
            query_words = set(query.lower().split())
            doc_scores = []

            for doc in docs:
                doc_words = set(doc.page_content.lower().split())
                similarity = len(query_words.intersection(doc_words)) / len(query_words.union(doc_words))
                doc_scores.append((similarity, doc))

            # Sort by similarity and return top_k
            doc_scores.sort(key=lambda x: x[0], reverse=True)
            return [doc for score, doc in doc_scores[:top_k]]

        # Demo similarity search
        query = "transformer attention mechanism"
        similar_docs = simple_similarity_search(all_chunked_docs, query)

        print(f"\n🔍 Similarity search for '{query}':")
        for i, doc in enumerate(similar_docs):
            print(f"   {i+1}. {doc.metadata['source']} ({doc.metadata['file_type']})")
            print(f"      Preview: \"{doc.page_content[:80]}...\"")

        # Cleanup
        for file_path in temp_files:
            try:
                os.unlink(file_path)
            except:
                pass

    except Exception as e:
        print(f"❌ Error in custom retrieval demo: {e}")

def demo_question_answering():
    """Demo 4: Simple question-answering with chunked documents."""
    print("\n" + "="*60)
    print("❓ DEMO 4: Question-Answering Pipeline")
    print("="*60)

    try:
        from langchain_core.documents import Document
        from chunking_strategy import create_chunker

        # Create a knowledge base
        knowledge_text = """
        Python Programming Language

        Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together.

        Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance. Python supports modules and packages, which encourages program modularity and code reuse. The Python interpreter and the extensive standard library are available in source or binary form without charge for all major platforms, and can be freely distributed.

        Key Features of Python:
        1. Easy to Learn: Python has few keywords, simple structure, and a clearly defined syntax.
        2. Easy to Read: Python code is more clearly defined and visible to the eyes.
        3. Easy to Maintain: Python's source code is fairly easy-to-maintain.
        4. A broad standard library: Python's bulk of the library is very portable and cross-platform compatible.
        5. Interactive Mode: Python has support for an interactive mode which allows interactive testing.
        6. Portable: Python can run on a wide variety of hardware platforms.
        7. Extendable: Python provides interfaces to all major commercial databases.
        8. GUI Programming: Python supports GUI applications and can be ported to many system calls.
        9. Scalable: Python provides a better structure and support for large programs.

        Common Python Applications:
        - Web Development (Django, Flask)
        - Data Science and Analytics (NumPy, Pandas, Matplotlib)
        - Machine Learning (Scikit-learn, TensorFlow, PyTorch)
        - Automation and Scripting
        - Desktop GUI Applications
        - Game Development
        """

        # Chunk the knowledge base
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(knowledge_text)
            temp_file = f.name

        try:
            chunker = create_chunker('paragraph_based', max_paragraphs=2)
            result = chunker.chunk(temp_file)

            # Convert to LangChain documents
            knowledge_docs = [
                Document(
                    page_content=chunk.content,
                    metadata={
                        "source": "python_knowledge_base",
                        "chunk_id": chunk.id,
                        "topic": "Python Programming"
                    }
                )
                for chunk in result.chunks
            ]

            print(f"📚 Knowledge base: {len(knowledge_docs)} chunks")

            # Simple Q&A function
            def answer_question(question: str, docs: List[Document], max_context_length: int = 500) -> str:
                """Simple extractive Q&A using keyword matching."""
                question_words = set(question.lower().split())

                # Find most relevant chunks
                chunk_scores = []
                for doc in docs:
                    doc_words = set(doc.page_content.lower().split())
                    overlap = len(question_words.intersection(doc_words))
                    if overlap > 0:
                        chunk_scores.append((overlap, doc))

                if not chunk_scores:
                    return "I don't have enough information to answer that question."

                # Sort by relevance and combine top chunks
                chunk_scores.sort(key=lambda x: x[0], reverse=True)

                context = ""
                for score, doc in chunk_scores[:2]:  # Use top 2 chunks
                    if len(context) + len(doc.page_content) <= max_context_length:
                        context += doc.page_content + "\\n\\n"
                    else:
                        break

                return f"Based on the knowledge base:\\n\\n{context.strip()}"

            # Demo questions
            questions = [
                "What is Python?",
                "What are the key features of Python?",
                "What applications can be built with Python?",
                "Is Python easy to learn?",
                "Can Python be used for machine learning?"
            ]

            print("\\n❓ Question-Answering Demo:")
            for i, question in enumerate(questions, 1):
                print(f"\\n   Q{i}: {question}")
                answer = answer_question(question, knowledge_docs)
                # Show only first 200 chars of answer for demo
                print(f"   A{i}: {answer[:200]}{'...' if len(answer) > 200 else ''}")

        finally:
            os.unlink(temp_file)

    except Exception as e:
        print(f"❌ Error in Q&A demo: {e}")

def main():
    """Run all LangChain integration demos."""
    print("🚀 LangChain Integration Comprehensive Demo")
    print("=" * 60)

    if not check_dependencies():
        return

    # Run all demos
    demo_basic_document_conversion()
    demo_rag_pipeline()
    demo_custom_retrieval_workflow()
    demo_question_answering()

    print("\n" + "="*60)
    print("🎉 All LangChain integration demos completed!")
    print("="*60)
    print("\n📝 Next Steps:")
    print("1. Explore different chunking strategies for your use case")
    print("2. Experiment with different embedding models")
    print("3. Try real vector databases (Pinecone, Weaviate, etc.)")
    print("4. Integrate with LLMs for better question-answering")
    print("5. Add evaluation metrics for your RAG pipeline")

if __name__ == "__main__":
    main()
