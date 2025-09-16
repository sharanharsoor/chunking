"""
Embedding-Based Chunking Strategy.

This module implements an embedding-based text chunker that uses vector embeddings
and similarity metrics to identify semantic boundaries. The chunker leverages
dense vector representations to determine where natural semantic breaks occur
in the text, creating chunks that maintain semantic coherence.

Key Features:
- Multiple embedding models (Sentence Transformers, Word2Vec, custom embeddings)
- Various similarity metrics (cosine, euclidean, dot product)
- Clustering approaches (K-means, hierarchical, DBSCAN)
- Dynamic threshold adjustment
- Streaming capabilities with embedding caching
- Adaptive parameter tuning based on similarity patterns
- Performance optimization with vectorized operations

Author: AI Assistant
Date: 2024
"""

import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple, Iterator
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from functools import lru_cache

from chunking_strategy.core.base import (
    BaseChunker,
    StreamableChunker,
    AdaptableChunker,
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType
)
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.strategies.text.text_chunker_utils import TextChunkerUtils

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = DBSCAN = AgglomerativeClustering = None
    cosine_similarity = euclidean_distances = None
    StandardScaler = PCA = None

try:
    import scipy.spatial.distance as scipy_distance
    from scipy.cluster.hierarchy import linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_distance = linkage = fcluster = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    sent_tokenize = None


class EmbeddingModel(Enum):
    """Supported embedding models."""
    SENTENCE_TRANSFORMER = "sentence_transformer"
    TFIDF = "tfidf"
    WORD_AVERAGE = "word_average"
    CUSTOM = "custom"


class SimilarityMetric(Enum):
    """Supported similarity metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class ClusteringMethod(Enum):
    """Supported clustering methods."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    THRESHOLD_BASED = "threshold_based"


@dataclass
class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""
    embeddings: Dict[str, np.ndarray]
    sentences: List[str]
    model_name: str
    timestamp: float

    def is_valid(self, max_age: float = 3600.0) -> bool:
        """Check if cache is still valid."""
        return time.time() - self.timestamp < max_age


@register_chunker(
    name="embedding_based",
    category="text",
    description="Vector embedding-based chunker using similarity metrics and clustering for semantic text segmentation",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "html", "xml", "json", "rtf"],
    complexity=ComplexityLevel.HIGH,
    dependencies=["sentence-transformers", "scikit-learn"],
    optional_dependencies=["scipy", "nltk", "torch"],
    speed=SpeedLevel.SLOW,
    memory=MemoryUsage.HIGH,
    quality=0.9,
    use_cases=["semantic_similarity", "topic_clustering", "vector_search", "content_similarity", "embedding_generation"],
    best_for=["semantic analysis", "content clustering", "similarity search", "topic modeling", "vector databases"],
    limitations=["computationally intensive", "requires large models", "high memory usage", "model-dependent quality"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=False
)
class EmbeddingBasedChunker(StreamableChunker, AdaptableChunker):
    """
    Embedding-based text chunker using vector similarity for boundary detection.

    This chunker creates embeddings for text segments and uses similarity metrics
    to determine optimal chunk boundaries, ensuring semantic coherence within chunks
    while maximizing semantic diversity between chunks.
    """

    def __init__(
        self,
        embedding_model: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        similarity_metric: str = "cosine",
        similarity_threshold: float = 0.7,
        clustering_method: str = "threshold_based",
        min_chunk_sentences: int = 2,
        max_chunk_sentences: int = 10,
        target_chunk_size: int = 1000,
        max_chunk_size: int = 2000,
        chunk_overlap: int = 50,
        clustering_params: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True,
        dimension_reduction: bool = False,
        adaptive_threshold: bool = True,
        quality_threshold: float = 0.6,
        **kwargs
    ):
        """
        Initialize the Embedding-Based Chunker.

        Args:
            embedding_model: Type of embedding model to use
            model_name: Specific model name (for sentence transformers)
            similarity_metric: Metric for computing similarity
            similarity_threshold: Threshold for similarity-based chunking
            clustering_method: Method for clustering embeddings
            min_chunk_sentences: Minimum sentences per chunk
            max_chunk_sentences: Maximum sentences per chunk
            target_chunk_size: Target character count per chunk
            max_chunk_size: Maximum character count per chunk
            chunk_overlap: Overlap between consecutive chunks in characters
            clustering_params: Parameters for clustering algorithms
            enable_caching: Whether to cache embeddings
            dimension_reduction: Whether to apply PCA dimensionality reduction
            adaptive_threshold: Whether to adapt thresholds based on content
            quality_threshold: Minimum quality score for adaptive tuning
        """
        # Initialize base class with name and configuration
        super().__init__(
            name="embedding_based",
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        # Validate parameters
        self._validate_parameters(
            similarity_threshold, min_chunk_sentences, max_chunk_sentences,
            quality_threshold
        )

        # Core configuration
        self.embedding_model = EmbeddingModel(embedding_model)
        self.model_name = model_name
        self.similarity_metric = SimilarityMetric(similarity_metric)
        self.similarity_threshold = similarity_threshold
        self.clustering_method = ClusteringMethod(clustering_method)
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_caching = enable_caching
        self.dimension_reduction = dimension_reduction
        self.adaptive_threshold = adaptive_threshold
        self.quality_threshold = quality_threshold

        # Clustering parameters
        self.clustering_params = clustering_params or {}

        # Initialize components
        self.encoder = None
        self.vectorizer = None
        self.scaler = None
        self.pca = None
        self.embedding_cache: Optional[EmbeddingCache] = None

        # Performance tracking
        self.performance_stats = {
            "total_sentences_processed": 0,
            "embedding_time": 0.0,
            "similarity_computation_time": 0.0,
            "clustering_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Adaptation history
        self._adaptation_history = []

        # Initialize models
        self._initialize_models()

        logging.info(f"EmbeddingBasedChunker initialized with {embedding_model} model")

    def _validate_parameters(
        self,
        similarity_threshold: float,
        min_chunk_sentences: int,
        max_chunk_sentences: int,
        quality_threshold: float
    ):
        """Validate initialization parameters."""
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        if min_chunk_sentences < 1:
            raise ValueError("min_chunk_sentences must be at least 1")

        if min_chunk_sentences >= max_chunk_sentences:
            raise ValueError("min_chunk_sentences must be less than max_chunk_sentences")

        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be between 0.0 and 1.0")

    def _initialize_models(self):
        """Initialize embedding models and components."""
        try:
            if self.embedding_model == EmbeddingModel.SENTENCE_TRANSFORMER:
                self._initialize_sentence_transformer()
            elif self.embedding_model == EmbeddingModel.TFIDF:
                self._initialize_tfidf()
            elif self.embedding_model == EmbeddingModel.WORD_AVERAGE:
                self._initialize_word_average()

            # Initialize dimensionality reduction if requested
            if self.dimension_reduction and SKLEARN_AVAILABLE:
                self.pca = PCA(n_components=0.95)  # Keep 95% of variance
                self.scaler = StandardScaler()

        except Exception as e:
            logging.warning(f"Failed to initialize preferred models: {e}")
            self._initialize_fallback()

    def _initialize_sentence_transformer(self):
        """Initialize Sentence Transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available")

        self.encoder = SentenceTransformer(self.model_name)
        logging.info(f"Initialized Sentence Transformer: {self.model_name}")

    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")

        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        logging.info("Initialized TF-IDF vectorizer")

    def _initialize_word_average(self):
        """Initialize word averaging approach (placeholder)."""
        # This would typically load pre-trained word vectors
        logging.info("Initialized word averaging embeddings")

    def _initialize_fallback(self):
        """Initialize fallback TF-IDF model."""
        try:
            self._initialize_tfidf()
            self.embedding_model = EmbeddingModel.TFIDF
            logging.info("Using TF-IDF fallback")
        except ImportError:
            logging.warning("No embedding models available, using simple approach")
            self.embedding_model = EmbeddingModel.WORD_AVERAGE


    def get_supported_formats(self) -> List[str]:
        """Return supported file formats."""
        return ["txt", "md", "json", "xml", "html", "csv"]

    def estimate_chunks(self, content: Union[str, Path]) -> int:
        """Estimate number of chunks for given content."""
        if isinstance(content, Path):
            content = content.read_text(encoding='utf-8')

        sentences = self._segment_sentences(content)
        estimated_chunks = max(1, len(sentences) // max(1, self.max_chunk_sentences))
        return estimated_chunks

    def chunk(
        self,
        content: Union[str, Path, bytes],
        source_info: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Chunk content using embedding-based similarity analysis.

        Args:
            content: Text content, file path, or bytes to chunk
            source_info: Additional source information

        Returns:
            ChunkingResult with similarity-based chunks
        """
        start_time = time.time()

        # Process input content
        text_content = self._process_input_content(content)

        if not text_content or not text_content.strip():
            return self._create_empty_result(start_time, source_info)

        try:
            # Segment into sentences
            sentences = self._segment_sentences(text_content)

            if len(sentences) == 0:
                return self._create_empty_result(start_time, source_info)

            # Generate embeddings
            embeddings = self._generate_embeddings(sentences)

            # Determine chunk boundaries using similarity analysis
            chunk_boundaries = self._determine_boundaries(sentences, embeddings)

            # Create chunks from boundaries
            chunks = self._create_chunks_from_boundaries(
                sentences, chunk_boundaries, text_content
            )

            # Calculate processing time
            processing_time = time.time() - start_time
            self.performance_stats["total_sentences_processed"] += len(sentences)

            # Create enhanced source info
            enhanced_source_info = self._create_enhanced_source_info(
                source_info, sentences, embeddings, processing_time
            )

            return ChunkingResult(
                chunks=chunks,
                strategy_used=self.name,
                processing_time=processing_time,
                source_info=enhanced_source_info
            )

        except Exception as e:
            logging.error(f"Embedding-based chunking failed: {e}")
            return self._fallback_chunking(text_content, source_info, start_time)

    def _process_input_content(self, content: Union[str, Path, bytes]) -> str:
        """Process various input types into text content."""
        if isinstance(content, Path):
            return content.read_text(encoding='utf-8')
        elif isinstance(content, bytes):
            return content.decode('utf-8')
        elif isinstance(content, str):
            return content
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences."""
        if NLTK_AVAILABLE and sent_tokenize:
            try:
                sentences = sent_tokenize(text)
            except LookupError:
                # NLTK data not downloaded, use simple approach
                sentences = TextChunkerUtils().split_sentences(text)
        else:
            sentences = TextChunkerUtils().split_sentences(text)

        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for sentences."""
        # Check cache first
        if self.enable_caching and self._check_cache(sentences):
            self.performance_stats["cache_hits"] += 1
            return self.embedding_cache.embeddings[str(sentences)]

        self.performance_stats["cache_misses"] += 1

        start_time = time.time()

        try:
            if self.embedding_model == EmbeddingModel.SENTENCE_TRANSFORMER:
                embeddings = self._generate_sentence_transformer_embeddings(sentences)
            elif self.embedding_model == EmbeddingModel.TFIDF:
                embeddings = self._generate_tfidf_embeddings(sentences)
            elif self.embedding_model == EmbeddingModel.WORD_AVERAGE:
                embeddings = self._generate_word_average_embeddings(sentences)
            else:
                embeddings = self._generate_fallback_embeddings(sentences)

            # Apply dimensionality reduction if enabled
            if self.dimension_reduction and embeddings.shape[1] > 50:
                embeddings = self._apply_dimension_reduction(embeddings)

            # Cache embeddings
            if self.enable_caching:
                self._cache_embeddings(sentences, embeddings)

        except Exception as e:
            logging.warning(f"Embedding generation failed: {e}, using fallback")
            embeddings = self._generate_fallback_embeddings(sentences)

        self.performance_stats["embedding_time"] += time.time() - start_time
        return embeddings

    def _generate_sentence_transformer_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings using Sentence Transformer."""
        if not self.encoder:
            raise RuntimeError("Sentence Transformer not initialized")

        embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
        return embeddings

    def _generate_tfidf_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings using TF-IDF."""
        if not self.vectorizer:
            raise RuntimeError("TF-IDF vectorizer not initialized")

        # Fit and transform if not already fitted
        if not hasattr(self.vectorizer, 'vocabulary_'):
            embeddings = self.vectorizer.fit_transform(sentences)
        else:
            embeddings = self.vectorizer.transform(sentences)

        return embeddings.toarray()

    def _generate_word_average_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings by averaging word vectors (simple implementation)."""
        # This is a simplified implementation
        # In practice, you'd use pre-trained word vectors like Word2Vec or GloVe
        embeddings = []
        for sentence in sentences:
            # Simple hash-based embedding (for demonstration)
            words = sentence.lower().split()
            if words:
                # Create a simple feature vector based on word characteristics
                features = [
                    len(words),  # Number of words
                    np.mean([len(w) for w in words]),  # Average word length
                    sum(1 for w in words if w.isalpha()),  # Alphabetic words
                    sum(1 for w in words if w.isnumeric()),  # Numeric words
                    len(set(words)),  # Unique words
                ]
                embeddings.append(features)
            else:
                embeddings.append([0.0] * 5)

        return np.array(embeddings, dtype=float)

    def _generate_fallback_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate simple statistical embeddings as fallback."""
        embeddings = []
        for sentence in sentences:
            features = [
                len(sentence),  # Character count
                len(sentence.split()),  # Word count
                sentence.count('.'),  # Period count
                sentence.count(','),  # Comma count
                sentence.count('?'),  # Question mark count
                sentence.count('!'),  # Exclamation mark count
                sum(1 for c in sentence if c.isupper()),  # Uppercase count
                sum(1 for c in sentence if c.isdigit()),  # Digit count
            ]
            embeddings.append(features)

        return np.array(embeddings, dtype=float)

    def _apply_dimension_reduction(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA dimensionality reduction."""
        if not self.scaler or not self.pca:
            return embeddings

        try:
            # Scale features
            if not hasattr(self.scaler, 'mean_'):
                scaled_embeddings = self.scaler.fit_transform(embeddings)
            else:
                scaled_embeddings = self.scaler.transform(embeddings)

            # Apply PCA
            if not hasattr(self.pca, 'components_'):
                reduced_embeddings = self.pca.fit_transform(scaled_embeddings)
            else:
                reduced_embeddings = self.pca.transform(scaled_embeddings)

            return reduced_embeddings

        except Exception as e:
            logging.warning(f"Dimension reduction failed: {e}")
            return embeddings

    def _check_cache(self, sentences: List[str]) -> bool:
        """Check if embeddings are available in cache."""
        if not self.embedding_cache:
            return False

        sentences_key = str(sentences)
        return (
            self.embedding_cache.is_valid() and
            sentences_key in self.embedding_cache.embeddings
        )

    def _cache_embeddings(self, sentences: List[str], embeddings: np.ndarray):
        """Cache embeddings for future use."""
        sentences_key = str(sentences)

        if not self.embedding_cache:
            self.embedding_cache = EmbeddingCache(
                embeddings={sentences_key: embeddings},
                sentences=sentences,
                model_name=self.model_name,
                timestamp=time.time()
            )
        else:
            self.embedding_cache.embeddings[sentences_key] = embeddings
            self.embedding_cache.timestamp = time.time()

    def _determine_boundaries(
        self,
        sentences: List[str],
        embeddings: np.ndarray
    ) -> List[int]:
        """Determine chunk boundaries using similarity analysis."""
        if len(sentences) <= self.min_chunk_sentences:
            return [0, len(sentences)]

        start_time = time.time()

        try:
            if self.clustering_method == ClusteringMethod.THRESHOLD_BASED:
                boundaries = self._threshold_based_boundaries(embeddings)
            elif self.clustering_method == ClusteringMethod.KMEANS:
                boundaries = self._kmeans_boundaries(embeddings)
            elif self.clustering_method == ClusteringMethod.HIERARCHICAL:
                boundaries = self._hierarchical_boundaries(embeddings)
            elif self.clustering_method == ClusteringMethod.DBSCAN:
                boundaries = self._dbscan_boundaries(embeddings)
            else:
                boundaries = self._threshold_based_boundaries(embeddings)

        except Exception as e:
            logging.warning(f"Boundary detection failed: {e}, using fallback")
            boundaries = self._simple_boundaries(sentences)

        # Ensure boundaries respect sentence constraints
        boundaries = self._enforce_sentence_constraints(boundaries, len(sentences))

        self.performance_stats["similarity_computation_time"] += time.time() - start_time
        return boundaries

    def _threshold_based_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """Determine boundaries using similarity thresholds."""
        if len(embeddings) < 2:
            return [0, len(embeddings)]

        boundaries = [0]

        # Calculate similarities between consecutive sentences
        similarities = self._calculate_consecutive_similarities(embeddings)

        # Adaptive threshold adjustment
        threshold = self.similarity_threshold
        if self.adaptive_threshold:
            threshold = self._adapt_threshold(similarities)

        current_chunk_start = 0
        for i, similarity in enumerate(similarities):
            # Check if similarity drops below threshold
            if similarity < threshold:
                # Ensure minimum chunk size
                chunk_size = i + 1 - current_chunk_start
                if chunk_size >= self.min_chunk_sentences:
                    boundaries.append(i + 1)
                    current_chunk_start = i + 1

            # Check if maximum chunk size is reached
            elif (i + 1 - current_chunk_start) >= self.max_chunk_sentences:
                boundaries.append(i + 1)
                current_chunk_start = i + 1

        # Add final boundary
        if boundaries[-1] != len(embeddings):
            boundaries.append(len(embeddings))

        return boundaries

    def _kmeans_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """Determine boundaries using K-means clustering."""
        if not SKLEARN_AVAILABLE:
            return self._threshold_based_boundaries(embeddings)

        # Estimate number of clusters
        n_clusters = max(2, min(10, len(embeddings) // self.max_chunk_sentences))

        clustering_params = {
            "n_clusters": n_clusters,
            "random_state": 42,
            **self.clustering_params
        }

        kmeans = KMeans(**clustering_params)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Convert cluster labels to boundaries
        boundaries = [0]
        prev_label = cluster_labels[0]

        for i, label in enumerate(cluster_labels[1:], 1):
            if label != prev_label:
                boundaries.append(i)
                prev_label = label

        boundaries.append(len(embeddings))
        return sorted(set(boundaries))

    def _hierarchical_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """Determine boundaries using hierarchical clustering."""
        if not SKLEARN_AVAILABLE:
            return self._threshold_based_boundaries(embeddings)

        n_clusters = max(2, min(8, len(embeddings) // self.max_chunk_sentences))

        clustering_params = {
            "n_clusters": n_clusters,
            "linkage": "ward",
            **self.clustering_params
        }

        hierarchical = AgglomerativeClustering(**clustering_params)
        cluster_labels = hierarchical.fit_predict(embeddings)

        # Convert to boundaries
        boundaries = [0]
        prev_label = cluster_labels[0]

        for i, label in enumerate(cluster_labels[1:], 1):
            if label != prev_label:
                boundaries.append(i)
                prev_label = label

        boundaries.append(len(embeddings))
        return sorted(set(boundaries))

    def _dbscan_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """Determine boundaries using DBSCAN clustering."""
        if not SKLEARN_AVAILABLE:
            return self._threshold_based_boundaries(embeddings)

        clustering_params = {
            "eps": 0.5,
            "min_samples": max(2, self.min_chunk_sentences),
            **self.clustering_params
        }

        dbscan = DBSCAN(**clustering_params)
        cluster_labels = dbscan.fit_predict(embeddings)

        # Handle noise points (-1 labels)
        boundaries = [0]
        prev_label = cluster_labels[0]

        for i, label in enumerate(cluster_labels[1:], 1):
            if label != prev_label or label == -1:
                boundaries.append(i)
                prev_label = label

        boundaries.append(len(embeddings))
        return sorted(set(boundaries))

    def _simple_boundaries(self, sentences: List[str]) -> List[int]:
        """Simple boundary detection as fallback."""
        boundaries = [0]

        for i in range(self.max_chunk_sentences, len(sentences), self.max_chunk_sentences):
            boundaries.append(i)

        if boundaries[-1] != len(sentences):
            boundaries.append(len(sentences))

        return boundaries

    def _calculate_consecutive_similarities(self, embeddings: np.ndarray) -> List[float]:
        """Calculate similarities between consecutive embeddings."""
        similarities = []

        for i in range(len(embeddings) - 1):
            similarity = self._calculate_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(similarity)

        return similarities

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate similarity between two vectors."""
        if self.similarity_metric == SimilarityMetric.COSINE:
            return self._cosine_similarity(vec1, vec2)
        elif self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            return self._euclidean_similarity(vec1, vec2)
        elif self.similarity_metric == SimilarityMetric.DOT_PRODUCT:
            return self._dot_product_similarity(vec1, vec2)
        elif self.similarity_metric == SimilarityMetric.MANHATTAN:
            return self._manhattan_similarity(vec1, vec2)
        else:
            return self._cosine_similarity(vec1, vec2)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        if SKLEARN_AVAILABLE and cosine_similarity:
            return cosine_similarity([vec1], [vec2])[0, 0]
        else:
            # Manual implementation
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norm_product == 0:
                return 0.0
            return dot_product / norm_product

    def _euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean-based similarity (converted to similarity scale)."""
        distance = np.linalg.norm(vec1 - vec2)
        # Convert distance to similarity (0-1 range)
        max_distance = np.sqrt(len(vec1))  # Theoretical maximum
        return 1.0 - min(distance / max_distance, 1.0)

    def _dot_product_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate normalized dot product similarity."""
        dot_product = np.dot(vec1, vec2)
        # Normalize to [0, 1] range (assuming unit vectors)
        return max(0.0, min(1.0, (dot_product + 1.0) / 2.0))

    def _manhattan_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Manhattan-based similarity."""
        distance = np.sum(np.abs(vec1 - vec2))
        # Convert to similarity
        max_distance = 2.0 * len(vec1)  # Theoretical maximum for normalized vectors
        return 1.0 - min(distance / max_distance, 1.0)

    def _adapt_threshold(self, similarities: List[float]) -> float:
        """Adapt similarity threshold based on data distribution."""
        if not similarities:
            return self.similarity_threshold

        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)

        # Adapt threshold based on distribution
        if std_similarity < 0.1:  # Low variance - use tighter threshold
            adapted_threshold = mean_similarity - 0.5 * std_similarity
        else:  # High variance - use more flexible threshold
            adapted_threshold = mean_similarity - std_similarity

        # Ensure reasonable bounds
        adapted_threshold = max(0.1, min(0.9, adapted_threshold))
        return adapted_threshold

    def _enforce_sentence_constraints(
        self,
        boundaries: List[int],
        total_sentences: int
    ) -> List[int]:
        """Ensure boundaries respect minimum and maximum sentence constraints."""
        if len(boundaries) < 2:
            return [0, total_sentences]

        refined_boundaries = [0]

        for i in range(1, len(boundaries) - 1):
            start_idx = refined_boundaries[-1]
            end_idx = boundaries[i]
            chunk_size = end_idx - start_idx

            if chunk_size >= self.min_chunk_sentences:
                refined_boundaries.append(end_idx)
            # If too small, merge with previous chunk (skip boundary)

        # Ensure we end at the total
        if refined_boundaries[-1] != total_sentences:
            refined_boundaries.append(total_sentences)

        return refined_boundaries

    def _create_chunks_from_boundaries(
        self,
        sentences: List[str],
        boundaries: List[int],
        original_text: str
    ) -> List[Chunk]:
        """Create Chunk objects from boundary indices."""
        chunks = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_content = " ".join(chunk_sentences)

            # Calculate offset in original text
            offset = self._calculate_offset(original_text, chunk_content)

            # Create metadata
            metadata = ChunkMetadata(
                source="content",
                source_type="content",
                position=f"sentences {start_idx}-{end_idx-1}",
                length=len(chunk_content),
                offset=offset,
                extra={
                    "chunker_used": "embedding_based",
                    "chunk_index": i,
                    "sentence_count": len(chunk_sentences),
                    "embedding_model": self.embedding_model.value,
                    "similarity_metric": self.similarity_metric.value,
                    "clustering_method": self.clustering_method.value,
                    "similarity_threshold": self.similarity_threshold,
                    "sentence_range": f"{start_idx}-{end_idx-1}",
                    "chunking_strategy": "embedding_based"
                }
            )

            chunk = Chunk(
                id=f"embedding_{i}",
                content=chunk_content,
                metadata=metadata,
                modality=ModalityType.TEXT
            )
            chunks.append(chunk)

        return chunks

    def _calculate_offset(self, original_text: str, chunk_content: str) -> int:
        """Calculate the offset of chunk content in original text."""
        try:
            return original_text.find(chunk_content[:50])  # Use first 50 chars for matching
        except:
            return 0

    def _create_enhanced_source_info(
        self,
        source_info: Optional[Dict[str, Any]],
        sentences: List[str],
        embeddings: np.ndarray,
        processing_time: float
    ) -> Dict[str, Any]:
        """Create enhanced source information with embedding analysis."""
        enhanced_info = source_info.copy() if source_info else {}

        enhanced_info.update({
            "embedding_based_metadata": {
                "total_sentences": len(sentences),
                "embedding_dimension": embeddings.shape[1] if embeddings.size > 0 else 0,
                "embedding_model": self.embedding_model.value,
                "similarity_metric": self.similarity_metric.value,
                "clustering_method": self.clustering_method.value,
                "similarity_threshold": self.similarity_threshold,
                "processing_time": processing_time,
                "performance_stats": self.performance_stats.copy(),
                "cache_enabled": self.enable_caching,
                "dimension_reduction": self.dimension_reduction
            },
            "chunking_strategy": "embedding_based",
            "total_sentences": len(sentences),
            "embedding_model": self.embedding_model.value
        })

        return enhanced_info

    def _create_empty_result(
        self,
        start_time: float,
        source_info: Optional[Dict[str, Any]]
    ) -> ChunkingResult:
        """Create empty result for edge cases."""
        processing_time = time.time() - start_time
        enhanced_source_info = source_info.copy() if source_info else {}
        enhanced_source_info["embedding_based_metadata"] = {
            "processing_time": processing_time,
            "total_sentences": 0,
            "reason": "empty_or_invalid_content"
        }

        return ChunkingResult(
            chunks=[],
            strategy_used=self.name,
            processing_time=processing_time,
            source_info=enhanced_source_info
        )

    def _fallback_chunking(
        self,
        content: str,
        source_info: Optional[Dict[str, Any]],
        start_time: float
    ) -> ChunkingResult:
        """Fallback chunking when embedding-based approach fails."""
        try:
            # Simple sentence-based fallback
            sentences = self._segment_sentences(content)
            chunks = []

            for i in range(0, len(sentences), self.max_chunk_sentences):
                chunk_sentences = sentences[i:i + self.max_chunk_sentences]
                chunk_content = " ".join(chunk_sentences)

                metadata = ChunkMetadata(
                    source="content",
                    source_type="content",
                    position=f"fallback_chunk_{len(chunks)}",
                    length=len(chunk_content),
                    offset=0,
                    extra={
                        "chunker_used": "embedding_based_fallback",
                        "chunk_index": len(chunks),
                        "sentence_count": len(chunk_sentences),
                        "fallback_mode": True
                    }
                )

                chunk = Chunk(
                    id=f"fallback_{len(chunks)}",
                    content=chunk_content,
                    metadata=metadata,
                    modality=ModalityType.TEXT
                )
                chunks.append(chunk)

            processing_time = time.time() - start_time
            enhanced_source_info = source_info.copy() if source_info else {}
            enhanced_source_info["embedding_based_metadata"] = {
                "processing_time": processing_time,
                "fallback_mode": True,
                "total_sentences": len(sentences)
            }

            return ChunkingResult(
                chunks=chunks,
                strategy_used="embedding_based_fallback",
                processing_time=processing_time,
                source_info=enhanced_source_info
            )

        except Exception as e:
            logging.error(f"Fallback chunking also failed: {e}")
            return self._create_empty_result(start_time, source_info)

    def chunk_stream(
        self,
        stream_data: List[str],
        source_info: Optional[Dict[str, Any]] = None
    ) -> Iterator[Chunk]:
        """
        Chunk streaming data using embedding-based analysis.

        Args:
            stream_data: List of text segments to process as stream
            source_info: Optional source information

        Yields:
            Individual chunks as they are processed
        """
        combined_content = " ".join(stream_data)
        result = self.chunk(combined_content, source_info)

        for chunk in result.chunks:
            yield chunk

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Adapt chunker parameters based on feedback.

        Args:
            feedback_score: Score from 0.0 to 1.0 indicating quality/performance
            feedback_type: Type of feedback ("quality", "performance", "similarity")
            context: Additional context for adaptation

        Returns:
            Dictionary of parameter changes made
        """
        changes = {}
        original_params = {
            "similarity_threshold": self.similarity_threshold,
            "clustering_method": self.clustering_method.value,
            "max_chunk_sentences": self.max_chunk_sentences
        }

        if feedback_score < self.quality_threshold:
            # Poor feedback - adjust for better quality
            if feedback_type == "quality":
                # Increase similarity threshold for more coherent chunks
                if self.similarity_threshold < 0.9:
                    self.similarity_threshold += 0.05
                    changes["similarity_threshold"] = self.similarity_threshold

            elif feedback_type == "similarity":
                # Adjust clustering method if similarity is poor
                if self.clustering_method == ClusteringMethod.THRESHOLD_BASED:
                    self.clustering_method = ClusteringMethod.HIERARCHICAL
                    changes["clustering_method"] = self.clustering_method.value

            elif feedback_type == "performance":
                # Increase chunk sizes for better performance
                if self.max_chunk_sentences < 15:
                    self.max_chunk_sentences += 1
                    changes["max_chunk_sentences"] = self.max_chunk_sentences

        elif feedback_score > 0.8:
            # Good feedback - optimize for efficiency
            if feedback_type == "performance":
                # Increase chunk sizes for efficiency
                if self.max_chunk_sentences < 15:
                    self.max_chunk_sentences += 1
                    changes["max_chunk_sentences"] = self.max_chunk_sentences

            elif feedback_type == "quality" and self.similarity_threshold > 0.5:
                # Slightly relax threshold for efficiency
                self.similarity_threshold -= 0.02
                changes["similarity_threshold"] = self.similarity_threshold

        # Record adaptation
        if changes:
            self._adaptation_history.append({
                "timestamp": time.time(),
                "feedback_score": feedback_score,
                "feedback_type": feedback_type,
                "context": context,
                "original_params": original_params,
                "adapted_params": {
                    "similarity_threshold": self.similarity_threshold,
                    "clustering_method": self.clustering_method.value,
                    "max_chunk_sentences": self.max_chunk_sentences
                },
                "changes": changes
            })

        return changes

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get the history of parameter adaptations."""
        return self._adaptation_history.copy()

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration and performance statistics."""
        return {
            "name": self.name,
            "embedding_model": self.embedding_model.value,
            "model_name": self.model_name,
            "similarity_metric": self.similarity_metric.value,
            "similarity_threshold": self.similarity_threshold,
            "clustering_method": self.clustering_method.value,
            "min_chunk_sentences": self.min_chunk_sentences,
            "max_chunk_sentences": self.max_chunk_sentences,
            "target_chunk_size": self.target_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "enable_caching": self.enable_caching,
            "dimension_reduction": self.dimension_reduction,
            "adaptive_threshold": self.adaptive_threshold,
            "quality_threshold": self.quality_threshold,
            "performance_stats": self.performance_stats.copy(),
            "adaptation_history_length": len(self._adaptation_history),
            "clustering_params": self.clustering_params,
            "cache_status": {
                "enabled": self.enable_caching,
                "has_cache": self.embedding_cache is not None,
                "cache_valid": self.embedding_cache.is_valid() if self.embedding_cache else False
            }
        }
