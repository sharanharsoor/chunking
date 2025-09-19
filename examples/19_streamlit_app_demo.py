#!/usr/bin/env python3
"""
Comprehensive Streamlit Integration Demo

A complete Streamlit web application showcasing chunking-strategy integration.

Features:
1. File upload and processing
2. Interactive strategy selection
3. Real-time chunking with preview
4. Chunk visualization and analysis
5. Export functionality
6. Performance metrics

To run this demo:
    pip install streamlit plotly
    streamlit run 19_streamlit_app_demo.py

The app will be available at http://localhost:8501
"""

import streamlit as st
import tempfile
import os
import json
import time
import sys
import statistics
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to Python path for local development (when running from source)
current_dir = Path(__file__).parent
parent_dir = current_dir.parent

# Ensure the chunking directory is in Python path for both direct run and streamlit run
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Also try to add the absolute path to be extra sure
import os
abs_parent_dir = os.path.abspath(parent_dir)
if abs_parent_dir not in sys.path:
    sys.path.insert(0, abs_parent_dir)

# Configure page
st.set_page_config(
    page_title="Chunking Strategy Demo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        missing.append("plotly")

    try:
        from chunking_strategy import create_chunker, list_strategies
    except ImportError:
        missing.append("chunking-strategy")

    if missing:
        st.error(f"❌ Missing dependencies: {', '.join(missing)}")
        st.error("Install with: pip install " + " ".join(missing))
        return False

    return True

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'processed_chunks' not in st.session_state:
        st.session_state.processed_chunks = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = 0
    if 'chunk_stats' not in st.session_state:
        st.session_state.chunk_stats = {}
    if 'detailed_metrics' not in st.session_state:
        st.session_state.detailed_metrics = {}
    if 'temp_files_to_cleanup' not in st.session_state:
        st.session_state.temp_files_to_cleanup = []

@st.cache_data
def get_available_strategies():
    """Get list of available chunking strategies."""
    try:
        from chunking_strategy import list_strategies
        return list_strategies()
    except:
        # Fallback list if function not available
        return [
            'sentence_based', 'paragraph_based', 'fixed_size', 'token_based',
            'semantic', 'boundary_aware', 'recursive', 'overlapping_window',
            'python_code', 'javascript_code', 'markdown_chunker', 'pdf_chunker'
        ]

def create_sample_content(content_type: str) -> str:
    """Create sample content for different types."""
    samples = {
        "Technical Article": """
        # Machine Learning in Practice

        Machine learning has revolutionized how we approach complex problems in various domains. From recommendation systems to autonomous vehicles, ML algorithms are becoming increasingly sophisticated and accessible.

        ## Supervised Learning

        Supervised learning is perhaps the most well-understood category of machine learning. In this paradigm, we train models using labeled examples, where both input features and desired outputs are known.

        Common supervised learning algorithms include:
        - Linear and logistic regression
        - Decision trees and random forests
        - Support vector machines
        - Neural networks

        ## Unsupervised Learning

        Unsupervised learning deals with finding hidden patterns in data without labeled examples. This is particularly useful for exploratory data analysis and feature discovery.

        Key unsupervised techniques include:
        - K-means clustering
        - Hierarchical clustering
        - Principal component analysis (PCA)
        - Autoencoders

        ## Best Practices

        When implementing machine learning solutions, consider these best practices:
        1. Start with simple models before moving to complex ones
        2. Always validate your models on held-out test data
        3. Monitor for data drift in production systems
        4. Implement proper feature engineering pipelines
        5. Document your modeling decisions and assumptions
        """,

        "Python Code": '''
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report

        class MLPipeline:
            """A simple machine learning pipeline for classification tasks."""

            def __init__(self, model=None, test_size=0.2, random_state=42):
                """
                Initialize the ML pipeline.

                Args:
                    model: Scikit-learn model instance
                    test_size: Fraction of data to use for testing
                    random_state: Random seed for reproducibility
                """
                self.model = model or RandomForestClassifier(n_estimators=100, random_state=random_state)
                self.test_size = test_size
                self.random_state = random_state
                self.is_trained = False

            def prepare_data(self, X, y):
                """Split data into training and testing sets."""
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=self.random_state
                )
                return self.X_train, self.X_test, self.y_train, self.y_test

            def train(self, X=None, y=None):
                """Train the model on the provided data."""
                if X is not None and y is not None:
                    self.prepare_data(X, y)

                if not hasattr(self, 'X_train'):
                    raise ValueError("No training data available. Call prepare_data first.")

                print("Training model...")
                self.model.fit(self.X_train, self.y_train)
                self.is_trained = True
                print("Training completed!")

            def evaluate(self):
                """Evaluate the trained model on test data."""
                if not self.is_trained:
                    raise ValueError("Model must be trained before evaluation.")

                predictions = self.model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, predictions)

                print(f"Accuracy: {accuracy:.4f}")
                print("\\nClassification Report:")
                print(classification_report(self.y_test, predictions))

                return accuracy, predictions

        def load_and_preprocess_data(file_path):
            """Load and preprocess data from a CSV file."""
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded data with shape: {df.shape}")

                # Basic preprocessing
                df = df.dropna()  # Remove missing values

                # Encode categorical variables if needed
                categorical_columns = df.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    if col != 'target':  # Assuming 'target' is the label column
                        df[col] = pd.Categorical(df[col]).codes

                return df

            except Exception as e:
                print(f"Error loading data: {e}")
                return None

        # Example usage
        if __name__ == "__main__":
            # Create synthetic data for demonstration
            from sklearn.datasets import make_classification

            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                random_state=42
            )

            # Initialize and train pipeline
            pipeline = MLPipeline()
            pipeline.prepare_data(X, y)
            pipeline.train()

            # Evaluate performance
            accuracy, predictions = pipeline.evaluate()
        ''',

        "Business Document": """
        QUARTERLY BUSINESS REVIEW - Q3 2024

        Executive Summary

        This quarter has shown remarkable growth across all key performance indicators. Revenue increased by 23% compared to Q2, driven primarily by strong performance in our enterprise solutions division and successful expansion into new geographic markets.

        Financial Performance

        Total Revenue: $12.4M (up 23% QoQ, 45% YoY)
        Gross Margin: 78% (up from 75% in Q2)
        Operating Expenses: $6.2M (up 15% QoQ due to planned hiring)
        Net Income: $3.8M (up 35% QoQ)
        Cash Flow: $4.1M positive (compared to $2.8M in Q2)

        Key Achievements

        1. Product Development
           - Launched AI-powered analytics feature with 89% positive user feedback
           - Completed integration with 5 major third-party platforms
           - Reduced average response time by 40% through infrastructure improvements

        2. Sales and Marketing
           - Acquired 150 new enterprise customers (target was 120)
           - Achieved 95% customer retention rate
           - Launched successful digital marketing campaign reaching 2.5M prospects

        3. Operations
           - Hired 25 new team members across engineering and sales
           - Implemented new project management system improving delivery times by 30%
           - Achieved ISO 27001 compliance for information security

        Market Analysis

        The market for our solutions continues to grow rapidly, with total addressable market estimated at $45B globally. Our main competitors include TechCorp Inc. and Innovation Systems, but we maintain competitive advantages in:
        - Superior customer support (NPS score of 78 vs industry average of 52)
        - More flexible pricing models
        - Faster time-to-value for customers

        Challenges and Risks

        1. Talent Acquisition: Competitive market for skilled developers
        2. Supply Chain: Potential disruptions in cloud service providers
        3. Regulatory: New data privacy regulations in European markets
        4. Competition: Increased investment in similar solutions by larger players

        Strategic Initiatives for Q4

        1. Launch mobile application for iOS and Android platforms
        2. Expand sales team by 40% to support growth in North American market
        3. Establish strategic partnerships with two major consulting firms
        4. Begin development of next-generation platform architecture
        5. Conduct Series B funding round to support international expansion

        Conclusion

        Q3 2024 has been a transformative quarter that positions us well for continued growth. The strong financial performance, successful product launches, and expanded team provide a solid foundation for achieving our annual targets and setting ambitious goals for 2025.
        """
    }

    return samples.get(content_type, samples["Technical Article"])

def cleanup_temp_files():
    """Clean up temporary files created during the session."""
    if 'temp_files_to_cleanup' in st.session_state:
        for file_path in st.session_state.temp_files_to_cleanup:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except:
                pass
        st.session_state.temp_files_to_cleanup = []

def generate_performance_report(metrics: Dict[str, Any]) -> str:
    """Generate a comprehensive performance report as text."""

    report_lines = [
        "="*80,
        "CHUNKING STRATEGY - PERFORMANCE REPORT",
        "="*80,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "STRATEGY CONFIGURATION",
        "-" * 40,
        f"Strategy: {metrics.get('strategy', 'N/A')}",
    ]

    if 'strategy_params' in metrics and metrics['strategy_params']:
        report_lines.append("Parameters:")
        for param, value in metrics['strategy_params'].items():
            report_lines.append(f"  • {param}: {value}")

    report_lines.extend([
        "",
        "PERFORMANCE SUMMARY",
        "-" * 40,
        f"Total Processing Time: {metrics.get('total_processing_time', 0):.4f} seconds",
        f"Input Size: {metrics.get('input_size_bytes', 0):,} bytes ({metrics.get('input_size_bytes', 0)/1024/1024:.2f} MB)",
    ])

    if 'performance' in metrics:
        perf = metrics['performance']
        report_lines.extend([
            f"Throughput: {perf.get('chunks_per_second', 0):.2f} chunks/second",
            f"Data Rate: {perf.get('mb_per_second', 0):.2f} MB/second",
        ])

    report_lines.extend([
        "",
        "MEMORY USAGE",
        "-" * 40,
        f"Start Memory: {metrics.get('memory_start', 0):.2f} MB",
        f"Peak Memory: {metrics.get('memory_peak', 0):.2f} MB",
        f"Memory Delta: {metrics.get('memory_delta', 0):.2f} MB",
    ])

    if 'quality' in metrics:
        quality = metrics['quality']
        report_lines.extend([
            "",
            "QUALITY METRICS",
            "-" * 40,
            f"Total Chunks: {quality.get('total_chunks', 0)}",
            f"Average Chunk Size: {quality.get('avg_chunk_size', 0):.1f} characters",
            f"Size Consistency Score: {quality.get('size_consistency', 0):.3f}",
            f"Content Coverage Ratio: {quality.get('coverage_ratio', 0):.3f}",
        ])

    return "\n".join(report_lines)

def process_file_content(file_content: str, strategy: str, strategy_params: Dict[str, Any]) -> tuple:
    """Process file content with the selected strategy and parameters.

    Args:
        file_content: Either text content (str) or file path (str) for uploaded files
        strategy: Chunking strategy name
        strategy_params: Strategy-specific parameters

    Returns:
        tuple: (chunking_result, processing_time, detailed_metrics)
    """
    from chunking_strategy import create_chunker
    import psutil

    # Initialize comprehensive metrics
    metrics = {
        "strategy": strategy,
        "strategy_params": strategy_params,
        "start_time": time.time(),
        "memory_start": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
        "memory_peak": 0,
        "stages": {},
        "chunk_metrics": {
            "sizes": [],
            "processing_times": []
        }
    }

    temp_file_to_cleanup = None

    try:
        # Stage 1: Content preparation
        stage_start = time.time()

        # Determine content size
        if os.path.exists(file_content):
            file_to_process = file_content
            content_size = os.path.getsize(file_content)
        else:
            content_size = len(file_content.encode('utf-8'))
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(file_content)
                file_to_process = f.name
                temp_file_to_cleanup = f.name

        metrics["input_size_bytes"] = content_size
        metrics["stages"]["content_preparation"] = time.time() - stage_start

        # Update peak memory
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        metrics["memory_peak"] = max(metrics["memory_peak"], current_memory)

        # Stage 2: Chunker creation
        stage_start = time.time()
        chunker = create_chunker(strategy, **strategy_params)
        metrics["stages"]["chunker_creation"] = time.time() - stage_start

        # Update peak memory
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        metrics["memory_peak"] = max(metrics["memory_peak"], current_memory)

        # Stage 3: Chunking execution
        stage_start = time.time()
        result = chunker.chunk(file_to_process)
        metrics["stages"]["chunking_execution"] = time.time() - stage_start

        # Stage 4: Results processing
        stage_start = time.time()
        chunks_list = list(result.chunks) if hasattr(result, 'chunks') else result

        # Collect chunk metrics
        for chunk in chunks_list:
            if hasattr(chunk, 'content'):
                chunk_size = len(str(chunk.content))
            elif isinstance(chunk, (str, bytes)):
                chunk_size = len(chunk)
            else:
                chunk_size = len(str(chunk))
            metrics["chunk_metrics"]["sizes"].append(chunk_size)

        metrics["stages"]["results_processing"] = time.time() - stage_start

        # Final metrics calculation
        metrics["end_time"] = time.time()
        metrics["total_processing_time"] = metrics["end_time"] - metrics["start_time"]
        metrics["memory_end"] = psutil.Process().memory_info().rss / 1024 / 1024
        metrics["memory_delta"] = metrics["memory_peak"] - metrics["memory_start"]

        # Performance calculations
        metrics["performance"] = {
            "chunks_per_second": len(chunks_list) / metrics["total_processing_time"] if metrics["total_processing_time"] > 0 else 0,
            "bytes_per_second": content_size / metrics["total_processing_time"] if metrics["total_processing_time"] > 0 else 0,
            "mb_per_second": (content_size / 1024 / 1024) / metrics["total_processing_time"] if metrics["total_processing_time"] > 0 else 0
        }

        # Quality metrics
        if metrics["chunk_metrics"]["sizes"]:
            avg_chunk_size = sum(metrics["chunk_metrics"]["sizes"]) / len(metrics["chunk_metrics"]["sizes"])
            metrics["quality"] = {
                "total_chunks": len(chunks_list),
                "avg_chunk_size": avg_chunk_size,
                "min_chunk_size": min(metrics["chunk_metrics"]["sizes"]),
                "max_chunk_size": max(metrics["chunk_metrics"]["sizes"]),
                "size_std": statistics.stdev(metrics["chunk_metrics"]["sizes"]) if len(metrics["chunk_metrics"]["sizes"]) > 1 else 0,
                "size_consistency": 1 - (statistics.stdev(metrics["chunk_metrics"]["sizes"]) / avg_chunk_size) if len(metrics["chunk_metrics"]["sizes"]) > 1 and avg_chunk_size > 0 else 1,
                "coverage_ratio": sum(metrics["chunk_metrics"]["sizes"]) / content_size if content_size > 0 else 0
            }

        return result, metrics["total_processing_time"], metrics

    except Exception as e:
        metrics["end_time"] = time.time()
        metrics["total_processing_time"] = metrics["end_time"] - metrics["start_time"]
        metrics["error"] = str(e)
        raise Exception(f"Processing failed after {metrics['total_processing_time']:.2f}s: {str(e)}")

    finally:
        # Clean up temporary file if we created one
        if temp_file_to_cleanup:
            try:
                os.unlink(temp_file_to_cleanup)
            except:
                pass

def calculate_chunk_statistics(chunks) -> Dict[str, Any]:
    """Calculate various statistics about the chunks."""
    if not chunks:
        return {}

    lengths = [len(chunk.content) for chunk in chunks]

    stats = {
        'total_chunks': len(chunks),
        'total_characters': sum(lengths),
        'avg_chunk_length': sum(lengths) / len(lengths),
        'min_chunk_length': min(lengths),
        'max_chunk_length': max(lengths),
        'median_chunk_length': sorted(lengths)[len(lengths) // 2],
        'std_chunk_length': (sum((x - sum(lengths) / len(lengths)) ** 2 for x in lengths) / len(lengths)) ** 0.5
    }

    return stats

def create_chunk_length_chart(chunks):
    """Create a histogram of chunk lengths."""
    try:
        import plotly.express as px
        import pandas as pd

        lengths = [len(chunk.content) for chunk in chunks]
        df = pd.DataFrame({'Chunk Length': lengths})

        fig = px.histogram(
            df,
            x='Chunk Length',
            title='Distribution of Chunk Lengths',
            labels={'count': 'Number of Chunks'},
            nbins=20
        )

        fig.update_layout(
            xaxis_title="Characters per Chunk",
            yaxis_title="Number of Chunks",
            showlegend=False
        )

        return fig
    except ImportError:
        return None

def create_chunk_overview_chart(stats):
    """Create a bar chart showing chunk statistics overview."""
    try:
        import plotly.graph_objects as go

        metrics = ['Min Length', 'Avg Length', 'Max Length', 'Median Length']
        values = [
            stats['min_chunk_length'],
            stats['avg_chunk_length'],
            stats['max_chunk_length'],
            stats['median_chunk_length']
        ]

        fig = go.Figure(data=[
            go.Bar(x=metrics, y=values, marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4'])
        ])

        fig.update_layout(
            title='Chunk Length Statistics',
            xaxis_title="Metrics",
            yaxis_title="Characters",
            showlegend=False
        )

        return fig
    except ImportError:
        return None

def main():
    """Main Streamlit app."""
    if not check_dependencies():
        return

    initialize_session_state()

    # App header
    st.title("📄 Chunking Strategy Interactive Demo")
    st.markdown("---")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload File", "Use Sample Content", "Paste Text"]
    )

    # File content handling
    file_content = None

    if input_method == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=[
                # Text files
                'txt', 'md', 'csv', 'json', 'xml', 'html', 'rtf',
                # Code files
                'py', 'js', 'java', 'cpp', 'c', 'h', 'css', 'php', 'rb', 'go', 'rs', 'ts', 'tsx', 'jsx',
                # Document files
                'pdf', 'doc', 'docx', 'odt', 'ppt', 'pptx',
                # Image files
                'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'svg',
                # Audio files
                'mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'wma',
                # Video files
                'mp4', 'avi', 'mov', 'mkv', 'wmv', 'webm', 'flv',
                # Data files
                'xlsx', 'xls', 'tsv'
            ],
            help="Upload any supported file type - text, documents, images, audio, video, code files, etc."
        )

        if uploaded_file is not None:
            try:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_file_path = tmp_file.name

                # Store file path for processing (not the content as text)
                file_content = temp_file_path

                # Track temp file for cleanup
                if temp_file_path not in st.session_state.temp_files_to_cleanup:
                    st.session_state.temp_files_to_cleanup.append(temp_file_path)

                # Show appropriate success message based on file type
                file_ext = uploaded_file.name.split('.')[-1].lower()
                if file_ext in ['txt', 'md', 'json', 'csv', 'py', 'js', 'html', 'xml']:
                    st.sidebar.success(f"✅ Text file loaded: {uploaded_file.name}")
                elif file_ext in ['pdf', 'doc', 'docx']:
                    st.sidebar.success(f"✅ Document loaded: {uploaded_file.name}")
                elif file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
                    st.sidebar.success(f"✅ Image loaded: {uploaded_file.name}")
                elif file_ext in ['mp3', 'wav', 'flac', 'aac', 'ogg']:
                    st.sidebar.success(f"✅ Audio file loaded: {uploaded_file.name}")
                elif file_ext in ['mp4', 'avi', 'mov', 'mkv', 'wmv']:
                    st.sidebar.success(f"✅ Video file loaded: {uploaded_file.name}")
                else:
                    st.sidebar.success(f"✅ File loaded: {uploaded_file.name}")

            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {e}")

    elif input_method == "Use Sample Content":
        sample_type = st.sidebar.selectbox(
            "Select sample content:",
            ["Technical Article", "Python Code", "Business Document"]
        )
        file_content = create_sample_content(sample_type)
        st.sidebar.success(f"✅ Sample loaded: {len(file_content)} characters")

    else:  # Paste Text
        file_content = st.sidebar.text_area(
            "Paste your content here:",
            height=200,
            placeholder="Enter or paste text content to chunk..."
        )

    if not file_content:
        st.info("👆 Please select an input method and provide content in the sidebar to get started.")
        return

    # Strategy selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Chunking Strategy")

    available_strategies = get_available_strategies()
    strategy = st.sidebar.selectbox("Select chunking strategy:", available_strategies)

    # Strategy-specific parameters
    strategy_params = {}

    if strategy == 'sentence_based':
        strategy_params['max_sentences'] = st.sidebar.slider("Max sentences per chunk", 1, 10, 3)
        strategy_params['overlap_sentences'] = st.sidebar.slider("Overlap sentences", 0, 3, 0)

    elif strategy == 'paragraph_based':
        strategy_params['max_paragraphs'] = st.sidebar.slider("Max paragraphs per chunk", 1, 5, 2)

    elif strategy == 'fixed_size':
        strategy_params['chunk_size'] = st.sidebar.slider("Chunk size (characters)", 100, 2000, 500)
        strategy_params['overlap_size'] = st.sidebar.slider("Overlap size", 0, 200, 50)

    elif strategy == 'token_based':
        strategy_params['max_tokens'] = st.sidebar.slider("Max tokens per chunk", 50, 1000, 200)

    # Processing button
    if st.sidebar.button("🚀 Process Content", type="primary"):
        with st.spinner("Processing content..."):
            try:
                result, processing_time, detailed_metrics = process_file_content(file_content, strategy, strategy_params)
                st.session_state.processed_chunks = result.chunks
                st.session_state.processing_time = processing_time
                st.session_state.detailed_metrics = detailed_metrics
                st.session_state.chunk_stats = calculate_chunk_statistics(result.chunks)
                st.sidebar.success(f"✅ Processed in {processing_time:.3f}s")

                # Show quick performance metrics in sidebar
                if detailed_metrics and "performance" in detailed_metrics:
                    perf = detailed_metrics["performance"]
                    st.sidebar.metric("⚡ Throughput", f"{perf['chunks_per_second']:.1f} chunks/s")
                    st.sidebar.metric("💾 Memory Delta", f"{detailed_metrics['memory_delta']:.1f} MB")

            except Exception as e:
                st.sidebar.error(f"❌ Processing failed: {e}")

    # Main content area
    if st.session_state.processed_chunks:
        chunks = st.session_state.processed_chunks
        stats = st.session_state.chunk_stats

        # Results overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Chunks", stats['total_chunks'])

        with col2:
            st.metric("Avg Length", f"{stats['avg_chunk_length']:.0f} chars")

        with col3:
            st.metric("Processing Time", f"{st.session_state.processing_time:.3f}s")

        with col4:
            st.metric("Total Characters", f"{stats['total_characters']:,}")

        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Statistics", "📄 Chunk Preview", "📈 Visualization", "💾 Export", "⚡ Performance Metrics"])

        with tab1:
            st.subheader("📊 Detailed Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Length Statistics:**")
                st.write(f"- Minimum: {stats['min_chunk_length']} characters")
                st.write(f"- Maximum: {stats['max_chunk_length']} characters")
                st.write(f"- Average: {stats['avg_chunk_length']:.1f} characters")
                st.write(f"- Median: {stats['median_chunk_length']} characters")
                st.write(f"- Std Dev: {stats['std_chunk_length']:.1f} characters")

            with col2:
                st.write("**Processing Info:**")
                st.write(f"- Strategy Used: `{strategy}`")
                st.write(f"- Parameters: {strategy_params}")
                st.write(f"- Total Chunks: {stats['total_chunks']}")
                st.write(f"- Processing Time: {st.session_state.processing_time:.3f} seconds")

        with tab2:
            st.subheader("📄 Chunk Preview")

            # Chunk navigation
            chunk_index = st.selectbox(
                "Select chunk to preview:",
                range(len(chunks)),
                format_func=lambda x: f"Chunk {x+1} ({len(chunks[x].content)} chars)"
            )

            if chunk_index < len(chunks):
                chunk = chunks[chunk_index]

                # Chunk metadata
                with st.expander("🏷️ Chunk Metadata", expanded=False):
                    metadata_dict = {
                        "ID": chunk.id,
                        "Length": f"{len(chunk.content)} characters",
                        "Source": getattr(chunk.metadata, 'source', 'N/A'),
                        "Chunker Used": getattr(chunk.metadata, 'chunker_used', 'N/A')
                    }

                    for key, value in metadata_dict.items():
                        st.write(f"**{key}:** {value}")

                # Chunk content
                st.write("**Content:**")
                st.text_area(
                    f"Chunk {chunk_index + 1} Content",
                    chunk.content,
                    height=300,
                    label_visibility="collapsed"
                )

        with tab3:
            st.subheader("📈 Chunk Analysis Visualization")

            # Length distribution histogram
            hist_fig = create_chunk_length_chart(chunks)
            if hist_fig:
                st.plotly_chart(hist_fig, use_container_width=True)

            # Statistics overview
            overview_fig = create_chunk_overview_chart(stats)
            if overview_fig:
                st.plotly_chart(overview_fig, use_container_width=True)

            # Chunk size trend
            try:
                import plotly.graph_objects as go

                chunk_lengths = [len(chunk.content) for chunk in chunks]
                chunk_numbers = list(range(1, len(chunks) + 1))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chunk_numbers,
                    y=chunk_lengths,
                    mode='lines+markers',
                    name='Chunk Length',
                    line=dict(color='#2ca02c', width=2)
                ))

                fig.update_layout(
                    title='Chunk Length Progression',
                    xaxis_title='Chunk Number',
                    yaxis_title='Characters',
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install plotly for advanced visualizations: `pip install plotly`")

        with tab4:
            st.subheader("💾 Export Options")

            # Export format selection
            export_format = st.radio(
                "Choose export format:",
                ["JSON", "CSV", "Plain Text"]
            )

            if st.button("📥 Generate Export"):
                if export_format == "JSON":
                    export_data = []
                    for i, chunk in enumerate(chunks):
                        export_data.append({
                            "chunk_id": chunk.id,
                            "index": i,
                            "content": chunk.content,
                            "length": len(chunk.content),
                            "metadata": {
                                "source": getattr(chunk.metadata, 'source', 'N/A'),
                                "chunker": getattr(chunk.metadata, 'chunker_used', strategy)
                            }
                        })

                    json_string = json.dumps(export_data, indent=2)
                    st.download_button(
                        "⬇️ Download JSON",
                        json_string,
                        file_name=f"chunks_{strategy}.json",
                        mime="application/json"
                    )

                elif export_format == "CSV":
                    csv_content = "chunk_id,index,content,length\\n"
                    for i, chunk in enumerate(chunks):
                        # Escape content for CSV
                        content_escaped = chunk.content.replace('"', '""').replace('\\n', '\\\\n')
                        csv_content += f'"{chunk.id}",{i},"{content_escaped}",{len(chunk.content)}\\n'

                    st.download_button(
                        "⬇️ Download CSV",
                        csv_content,
                        file_name=f"chunks_{strategy}.csv",
                        mime="text/csv"
                    )

                else:  # Plain Text
                    text_content = ""
                    for i, chunk in enumerate(chunks):
                        text_content += f"=== CHUNK {i+1} ({chunk.id}) ===\\n"
                        text_content += chunk.content
                        text_content += "\\n\\n"

                    st.download_button(
                        "⬇️ Download Text",
                        text_content,
                        file_name=f"chunks_{strategy}.txt",
                        mime="text/plain"
                    )

        with tab5:
            st.subheader("⚡ Performance Metrics Dashboard")

            if hasattr(st.session_state, 'detailed_metrics') and st.session_state.detailed_metrics:
                metrics = st.session_state.detailed_metrics

                # Performance Overview
                st.markdown("### 🎯 Performance Overview")
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

                with perf_col1:
                    st.metric(
                        "Total Processing Time",
                        f"{metrics.get('total_processing_time', 0):.3f}s"
                    )

                with perf_col2:
                    if 'performance' in metrics:
                        st.metric(
                            "Throughput",
                            f"{metrics['performance'].get('chunks_per_second', 0):.1f} chunks/s"
                        )

                with perf_col3:
                    st.metric(
                        "Memory Usage",
                        f"{metrics.get('memory_delta', 0):.1f} MB",
                        delta=f"{metrics.get('memory_delta', 0):.1f} MB"
                    )

                with perf_col4:
                    if 'performance' in metrics:
                        mb_per_sec = metrics['performance'].get('mb_per_second', 0)
                        st.metric(
                            "Data Rate",
                            f"{mb_per_sec:.2f} MB/s"
                        )

                # Processing Stages Breakdown
                st.markdown("### ⏱️ Processing Stages Breakdown")
                if 'stages' in metrics and metrics['stages']:
                    stages_col1, stages_col2 = st.columns(2)

                    with stages_col1:
                        st.write("**Stage Timings:**")
                        for stage, duration in metrics['stages'].items():
                            percentage = (duration / metrics.get('total_processing_time', 1)) * 100
                            st.write(f"• {stage.replace('_', ' ').title()}: {duration:.4f}s ({percentage:.1f}%)")

                    with stages_col2:
                        # Create a simple bar chart for stages
                        try:
                            import plotly.graph_objects as go

                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(metrics['stages'].values()),
                                    y=[stage.replace('_', ' ').title() for stage in metrics['stages'].keys()],
                                    orientation='h',
                                    marker_color='lightblue'
                                )
                            ])
                            fig.update_layout(
                                title="Processing Stage Duration",
                                xaxis_title="Time (seconds)",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except ImportError:
                            st.info("Install plotly for stage visualization: pip install plotly")

                # Quality and Efficiency Metrics
                st.markdown("### 📊 Quality & Efficiency Metrics")
                if 'quality' in metrics:
                    quality = metrics['quality']
                    quality_col1, quality_col2, quality_col3 = st.columns(3)

                    with quality_col1:
                        st.metric("Size Consistency", f"{quality.get('size_consistency', 0):.2f}")
                        st.caption("1.0 = perfectly consistent chunk sizes")

                    with quality_col2:
                        st.metric("Content Coverage", f"{quality.get('coverage_ratio', 0):.2f}")
                        st.caption("Ratio of output to input content")

                    with quality_col3:
                        avg_size = quality.get('avg_chunk_size', 0)
                        st.metric("Avg Chunk Size", f"{avg_size:.0f} chars")

                # Memory Usage Timeline
                st.markdown("### 💾 Memory Usage Details")
                mem_col1, mem_col2, mem_col3 = st.columns(3)

                with mem_col1:
                    st.metric("Start Memory", f"{metrics.get('memory_start', 0):.1f} MB")

                with mem_col2:
                    st.metric("Peak Memory", f"{metrics.get('memory_peak', 0):.1f} MB")

                with mem_col3:
                    st.metric("End Memory", f"{metrics.get('memory_end', 0):.1f} MB")

                # Input/Output Analysis
                st.markdown("### 📈 Input/Output Analysis")
                io_col1, io_col2 = st.columns(2)

                with io_col1:
                    input_size_mb = metrics.get('input_size_bytes', 0) / 1024 / 1024
                    st.metric("Input Size", f"{input_size_mb:.2f} MB")
                    st.metric("Input Characters", f"{metrics.get('input_size_bytes', 0):,}")

                with io_col2:
                    if 'quality' in metrics and 'chunk_metrics' in metrics:
                        total_output = sum(metrics['chunk_metrics'].get('sizes', []))
                        st.metric("Output Characters", f"{total_output:,}")
                        efficiency = total_output / metrics.get('input_size_bytes', 1) * 100
                        st.metric("Processing Efficiency", f"{efficiency:.1f}%")

                # Strategy Configuration
                st.markdown("### ⚙️ Strategy Configuration")
                strategy_col1, strategy_col2 = st.columns(2)

                with strategy_col1:
                    st.write(f"**Strategy Used:** `{metrics.get('strategy', 'N/A')}`")

                with strategy_col2:
                    if 'strategy_params' in metrics and metrics['strategy_params']:
                        st.write("**Parameters:**")
                        for param, value in metrics['strategy_params'].items():
                            st.write(f"• {param}: {value}")
                    else:
                        st.write("**Parameters:** Default configuration")

                # Export Metrics
                st.markdown("### 📥 Export Performance Data")

                col_export1, col_export2 = st.columns(2)

                with col_export1:
                    if st.button("📊 Export Metrics to JSON"):
                        metrics_json = json.dumps(metrics, indent=2, default=str)
                        st.download_button(
                            "⬇️ Download Metrics JSON",
                            metrics_json,
                            file_name=f"performance_metrics_{strategy}_{int(time.time())}.json",
                            mime="application/json"
                        )

                with col_export2:
                    if st.button("📈 Generate Performance Report"):
                        report = generate_performance_report(metrics)
                        st.download_button(
                            "⬇️ Download Performance Report",
                            report,
                            file_name=f"performance_report_{strategy}_{int(time.time())}.txt",
                            mime="text/plain"
                        )

            else:
                st.info("📊 No performance metrics available. Process some content first to see detailed performance analytics!")
                st.markdown("""
                **Performance Metrics Include:**
                - Processing time breakdown by stages
                - Memory usage tracking
                - Throughput measurements
                - Quality and consistency scores
                - Input/output analysis
                - Strategy configuration details
                """)

    else:
        # Show preview of content
        st.subheader("📋 Content Preview")

        preview_length = min(1000, len(file_content))
        st.text_area(
            "Current content (first 1000 characters):",
            file_content[:preview_length] + ("..." if len(file_content) > preview_length else ""),
            height=200,
            disabled=True
        )

        st.info("👈 Configure chunking settings in the sidebar and click 'Process Content' to see results.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using <a href='https://streamlit.io/'>Streamlit</a> and
            <a href='https://pypi.org/project/chunking-strategy/'>chunking-strategy</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Cleanup temporary files
    cleanup_temp_files()

if __name__ == "__main__":
    main()
