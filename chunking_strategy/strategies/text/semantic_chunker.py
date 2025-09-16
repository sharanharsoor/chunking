"""
Semantic Chunking Strategy.

This module implements meaning-based text chunking using semantic similarity analysis.
It detects topic boundaries by analyzing sentence-level semantic coherence using
pre-trained embedding models and similarity thresholds.

Key features:
- Sentence-level semantic similarity analysis
- Multiple embedding model support (sentence-transformers, spaCy)
- Configurable similarity thresholds for boundary detection
- Topic coherence scoring and optimization
- Semantic boundary detection with context preservation
- Support for different semantic granularities
- Integration with existing embedding infrastructure
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
from enum import Enum
from dataclasses import dataclass

# Core imports
from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType,
    StreamableChunker
)
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.core.adaptive import AdaptableChunker

# Embedding and NLP infrastructure
from chunking_strategy.core.embeddings import (
    EmbeddingModel,
    EmbeddingConfig,
    create_embedder
)

# Import with fallbacks
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticModel(str, Enum):
    """Supported semantic analysis models."""
    # Sentence transformer models (high quality)
    SENTENCE_TRANSFORMER = "sentence_transformer"
    # SpaCy models (balanced speed/quality)
    SPACY = "spacy"
    # Simple TF-IDF with cosine similarity (fast fallback)
    TFIDF = "tfidf"


class BoundaryDetectionMethod(str, Enum):
    """Methods for detecting semantic boundaries."""
    SIMILARITY_THRESHOLD = "similarity_threshold"  # Simple threshold-based
    SLIDING_WINDOW = "sliding_window"             # Moving window analysis
    DYNAMIC_THRESHOLD = "dynamic_threshold"       # Adaptive threshold
    COHERENCE_BASED = "coherence_based"          # Topic coherence scoring


@dataclass
class SemanticBoundary:
    """Represents a detected semantic boundary."""
    sentence_index: int
    similarity_score: float
    coherence_score: float
    boundary_strength: float
    boundary_type: str  # 'strong', 'moderate', 'weak'


@register_chunker(
    name="semantic",
    category="text",
    description="Semantic chunking using sentence-level similarity analysis and topic boundary detection",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "html", "json", "xml", "rtf"],
    complexity=ComplexityLevel.HIGH,
    dependencies=["sentence-transformers"],
    optional_dependencies=["spacy", "nltk", "scikit-learn"],
    speed=SpeedLevel.SLOW,
    memory=MemoryUsage.HIGH,
    quality=0.9,  # High quality due to semantic analysis
    parameters_schema={
        "semantic_model": {
            "type": "string",
            "enum": ["sentence_transformer", "spacy", "tfidf"],
            "default": "sentence_transformer",
            "description": "Semantic model to use for similarity analysis"
        },
        "embedding_model": {
            "type": "string",
            "default": "all-MiniLM-L6-v2",
            "description": "Specific embedding model (for sentence_transformer)"
        },
        "similarity_threshold": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.7,
            "description": "Minimum similarity score to avoid creating boundary"
        },
        "min_chunk_sentences": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "default": 3,
            "description": "Minimum sentences per chunk"
        },
        "max_chunk_sentences": {
            "type": "integer",
            "minimum": 5,
            "maximum": 100,
            "default": 15,
            "description": "Maximum sentences per chunk"
        },
        "boundary_detection": {
            "type": "string",
            "enum": ["similarity_threshold", "sliding_window", "dynamic_threshold", "coherence_based"],
            "default": "similarity_threshold",
            "description": "Method for detecting semantic boundaries"
        },
        "context_window_size": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "default": 3,
            "description": "Number of sentences to consider for context analysis"
        },
        "max_chunk_chars": {
            "type": "integer",
            "minimum": 500,
            "maximum": 50000,
            "default": 4000,
            "description": "Maximum characters per chunk"
        },
        "coherence_weight": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.3,
            "description": "Weight for coherence scoring in boundary detection"
        }
    },
    default_parameters={
        "semantic_model": "sentence_transformer",
        "embedding_model": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.7,
        "min_chunk_sentences": 3,
        "max_chunk_sentences": 15,
        "boundary_detection": "similarity_threshold",
        "context_window_size": 3,
        "max_chunk_chars": 4000,
        "coherence_weight": 0.3
    },
    use_cases=["semantic_analysis", "topic_modeling", "content_understanding", "RAG_quality", "meaning_preservation"],
    best_for=["academic_text", "articles", "documentation", "knowledge_extraction", "topic_analysis"],
    limitations=["requires semantic models", "slower than syntactic methods", "higher memory usage"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=False
)
class SemanticChunker(StreamableChunker, AdaptableChunker):
    """
    Semantic chunker using sentence-level similarity analysis.

    This chunker creates meaningful chunks by analyzing semantic similarity
    between sentences and detecting topic boundaries. It uses pre-trained
    embedding models to understand content meaning and maintains semantic
    coherence within chunks.

    Features:
    - Sentence-level semantic similarity analysis
    - Multiple embedding model backends
    - Configurable similarity thresholds
    - Topic boundary detection with multiple methods
    - Context-aware chunking with coherence optimization
    - Adaptive parameter tuning based on feedback

    Examples:
        # Basic semantic chunking
        chunker = SemanticChunker(
            similarity_threshold=0.75,
            min_chunk_sentences=3,
            max_chunk_sentences=10
        )

        # High-quality model
        chunker = SemanticChunker(
            semantic_model="sentence_transformer",
            embedding_model="all-mpnet-base-v2",
            boundary_detection="coherence_based"
        )

        # Fast processing
        chunker = SemanticChunker(
            semantic_model="spacy",
            boundary_detection="similarity_threshold",
            similarity_threshold=0.6
        )
    """

    def __init__(
        self,
        semantic_model: str = "sentence_transformer",
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        min_chunk_sentences: int = 3,
        max_chunk_sentences: int = 15,
        boundary_detection: str = "similarity_threshold",
        context_window_size: int = 3,
        max_chunk_chars: int = 4000,
        coherence_weight: float = 0.3,
        **kwargs
    ):
        super().__init__(
            name="semantic",
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        # Core parameters
        self.semantic_model = SemanticModel(semantic_model)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.boundary_detection = BoundaryDetectionMethod(boundary_detection)
        self.context_window_size = context_window_size
        self.max_chunk_chars = max_chunk_chars
        self.coherence_weight = coherence_weight

        # Validate parameters
        self._validate_parameters()

        # Model components
        self.embedder = None
        self.nlp = None
        self.vectorizer = None

        # Processing state
        self.sentences = []
        self.embeddings = None
        self.boundaries = []

        # Adaptation tracking
        self._adaptation_history = []

        # Performance tracking
        self._total_sentences_processed = 0
        self._embedding_time = 0.0
        self._boundary_detection_time = 0.0

        logger.info(f"Initialized SemanticChunker with {semantic_model} model, threshold={similarity_threshold}")

    def _validate_parameters(self):
        """Validate chunking parameters."""
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.min_chunk_sentences >= self.max_chunk_sentences:
            raise ValueError("min_chunk_sentences must be less than max_chunk_sentences")
        if self.min_chunk_sentences < 1:
            raise ValueError("min_chunk_sentences must be at least 1")
        if not (0.0 <= self.coherence_weight <= 1.0):
            raise ValueError("coherence_weight must be between 0.0 and 1.0")

    def _initialize_semantic_model(self):
        """Initialize the semantic analysis components."""
        try:
            if self.semantic_model == SemanticModel.SENTENCE_TRANSFORMER:
                self._initialize_sentence_transformer()
            elif self.semantic_model == SemanticModel.SPACY:
                self._initialize_spacy()
            elif self.semantic_model == SemanticModel.TFIDF:
                self._initialize_tfidf()
            else:
                raise ValueError(f"Unsupported semantic model: {self.semantic_model}")

            logger.info(f"Loaded {self.semantic_model.value} model for semantic analysis")

        except Exception as e:
            logger.warning(f"Failed to initialize {self.semantic_model.value}: {e}")
            # Fallback to TF-IDF if available
            if self.semantic_model != SemanticModel.TFIDF and SKLEARN_AVAILABLE:
                logger.info("Falling back to TF-IDF model")
                self.semantic_model = SemanticModel.TFIDF
                self._initialize_tfidf()
            else:
                raise

    def _initialize_sentence_transformer(self):
        """Initialize sentence transformer embedder."""
        try:
            # Map embedding model names
            model_mapping = {
                "all-MiniLM-L6-v2": EmbeddingModel.ALL_MINILM_L6_V2,
                "all-mpnet-base-v2": EmbeddingModel.ALL_MPNET_BASE_V2,
                "all-distilroberta-v1": EmbeddingModel.ALL_DISTILROBERTA_V1
            }

            model_enum = model_mapping.get(self.embedding_model, EmbeddingModel.ALL_MINILM_L6_V2)

            config = EmbeddingConfig(
                model=model_enum,
                batch_size=32,
                show_progress=False,
                normalize_embeddings=True
            )

            self.embedder = create_embedder(config)
            self.embedder.load_model()

        except Exception as e:
            raise ImportError(f"Failed to initialize sentence transformer: {e}")

    def _initialize_spacy(self):
        """Initialize spaCy NLP model."""
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy is required for spacy semantic model")

        try:
            # Try to load the model
            model_name = "en_core_web_md"  # Use medium model for better vectors
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                # Fallback to small model
                model_name = "en_core_web_sm"
                self.nlp = spacy.load(model_name)
                logger.warning(f"Using fallback model {model_name}")

        except OSError:
            raise ImportError(f"SpaCy model not found. Install with: python -m spacy download en_core_web_md")

    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for TF-IDF semantic model")

        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            norm='l2'
        )

    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using the best available method."""
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            return sent_tokenize(text)
        elif SPACY_AVAILABLE and self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Simple fallback
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]

    def _compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute embeddings for sentences using the selected semantic model."""
        embedding_start = time.time()

        try:
            if self.semantic_model == SemanticModel.SENTENCE_TRANSFORMER:
                embeddings = self.embedder.embed_text(sentences)
            elif self.semantic_model == SemanticModel.SPACY:
                embeddings = []
                for sentence in sentences:
                    doc = self.nlp(sentence)
                    # Use document vector (average of token vectors)
                    embeddings.append(doc.vector)
                embeddings = np.array(embeddings)
            elif self.semantic_model == SemanticModel.TFIDF:
                # Fit and transform sentences
                tfidf_matrix = self.vectorizer.fit_transform(sentences)
                embeddings = tfidf_matrix.toarray()
            else:
                raise ValueError(f"Unsupported semantic model: {self.semantic_model}")

            self._embedding_time += time.time() - embedding_start
            self._total_sentences_processed += len(sentences)

            return embeddings

        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            # Return zero embeddings as fallback
            dim = 384 if "MiniLM" in str(self.embedding_model) else 768
            return np.zeros((len(sentences), dim))

    def _compute_similarity_scores(self, embeddings: np.ndarray) -> List[float]:
        """Compute similarity scores between consecutive sentences."""
        if not SKLEARN_AVAILABLE:
            # Simple dot product similarity
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1])
                # Normalize if vectors aren't normalized
                norm_i = np.linalg.norm(embeddings[i])
                norm_j = np.linalg.norm(embeddings[i + 1])
                if norm_i > 0 and norm_j > 0:
                    sim = sim / (norm_i * norm_j)
                similarities.append(max(0, min(1, sim)))  # Clamp to [0, 1]
            return similarities
        else:
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0, 0]
                similarities.append(max(0, sim))  # Ensure non-negative
            return similarities

    def _detect_boundaries(self, sentences: List[str], similarities: List[float]) -> List[SemanticBoundary]:
        """Detect semantic boundaries using the selected method."""
        boundary_start = time.time()
        boundaries = []

        if self.boundary_detection == BoundaryDetectionMethod.SIMILARITY_THRESHOLD:
            boundaries = self._threshold_boundary_detection(similarities)
        elif self.boundary_detection == BoundaryDetectionMethod.SLIDING_WINDOW:
            boundaries = self._sliding_window_boundary_detection(similarities)
        elif self.boundary_detection == BoundaryDetectionMethod.DYNAMIC_THRESHOLD:
            boundaries = self._dynamic_threshold_detection(similarities)
        elif self.boundary_detection == BoundaryDetectionMethod.COHERENCE_BASED:
            boundaries = self._coherence_based_detection(sentences, similarities)

        self._boundary_detection_time += time.time() - boundary_start
        return boundaries

    def _threshold_boundary_detection(self, similarities: List[float]) -> List[SemanticBoundary]:
        """Simple threshold-based boundary detection."""
        boundaries = []

        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                # Found a potential boundary
                boundary = SemanticBoundary(
                    sentence_index=i + 1,  # Boundary is after sentence i
                    similarity_score=sim,
                    coherence_score=1.0 - sim,  # Inverse relationship
                    boundary_strength=1.0 - sim,
                    boundary_type="strong" if sim < 0.5 else "moderate" if sim < 0.7 else "weak"
                )
                boundaries.append(boundary)

        return boundaries

    def _sliding_window_boundary_detection(self, similarities: List[float]) -> List[SemanticBoundary]:
        """Sliding window approach for boundary detection."""
        boundaries = []
        window_size = min(self.context_window_size, len(similarities))

        for i in range(window_size, len(similarities) - window_size):
            # Calculate local average before and after
            before_avg = np.mean(similarities[max(0, i - window_size):i])
            after_avg = np.mean(similarities[i:min(len(similarities), i + window_size)])

            # Check if current similarity is significantly lower
            if similarities[i] < before_avg - 0.1 and similarities[i] < after_avg - 0.1:
                boundary = SemanticBoundary(
                    sentence_index=i + 1,
                    similarity_score=similarities[i],
                    coherence_score=max(before_avg, after_avg) - similarities[i],
                    boundary_strength=max(before_avg, after_avg) - similarities[i],
                    boundary_type="strong" if similarities[i] < 0.4 else "moderate"
                )
                boundaries.append(boundary)

        return boundaries

    def _dynamic_threshold_detection(self, similarities: List[float]) -> List[SemanticBoundary]:
        """Dynamic threshold based on local statistics."""
        boundaries = []

        # Calculate local statistics
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        dynamic_threshold = max(0.3, mean_sim - std_sim)

        for i, sim in enumerate(similarities):
            if sim < dynamic_threshold:
                z_score = (mean_sim - sim) / (std_sim + 1e-8)
                boundary = SemanticBoundary(
                    sentence_index=i + 1,
                    similarity_score=sim,
                    coherence_score=z_score,
                    boundary_strength=z_score,
                    boundary_type="strong" if z_score > 2 else "moderate" if z_score > 1 else "weak"
                )
                boundaries.append(boundary)

        return boundaries

    def _coherence_based_detection(self, sentences: List[str], similarities: List[float]) -> List[SemanticBoundary]:
        """Coherence-based boundary detection with topic analysis."""
        boundaries = []

        # Calculate coherence scores for potential chunks
        for i in range(1, len(sentences) - 1):
            if i >= self.min_chunk_sentences:
                # Calculate coherence for chunk ending at i
                chunk_similarities = similarities[max(0, i - self.context_window_size):i]
                coherence_score = np.mean(chunk_similarities) if chunk_similarities else 0.0

                # Combine similarity and coherence
                boundary_score = (1 - similarities[i]) * (1 - self.coherence_weight) + \
                               (1 - coherence_score) * self.coherence_weight

                if boundary_score > 0.5:  # Threshold for boundary detection
                    boundary = SemanticBoundary(
                        sentence_index=i + 1,
                        similarity_score=similarities[i],
                        coherence_score=coherence_score,
                        boundary_strength=boundary_score,
                        boundary_type="strong" if boundary_score > 0.8 else "moderate"
                    )
                    boundaries.append(boundary)

        return boundaries

    def _create_chunks_from_boundaries(self, sentences: List[str], boundaries: List[SemanticBoundary], source_info: Dict[str, Any]) -> List[Chunk]:
        """Create chunks based on detected boundaries."""
        chunks = []
        chunk_starts = [0]

        # Add boundary positions, ensuring minimum chunk size
        for boundary in boundaries:
            if boundary.sentence_index - chunk_starts[-1] >= self.min_chunk_sentences:
                chunk_starts.append(boundary.sentence_index)

        # Ensure we end at the last sentence
        if chunk_starts[-1] < len(sentences):
            chunk_starts.append(len(sentences))

        # Create chunks
        for i in range(len(chunk_starts) - 1):
            start_idx = chunk_starts[i]
            end_idx = chunk_starts[i + 1]

            # Enforce maximum chunk size
            if end_idx - start_idx > self.max_chunk_sentences:
                # Split large chunks
                sub_chunks = self._split_large_chunk(sentences, start_idx, end_idx)
                chunks.extend(sub_chunks)
            else:
                chunk = self._create_single_chunk(sentences, start_idx, end_idx, i, source_info)
                chunks.append(chunk)

        return chunks

    def _split_large_chunk(self, sentences: List[str], start_idx: int, end_idx: int) -> List[Chunk]:
        """Split a large chunk into smaller ones."""
        sub_chunks = []
        current_start = start_idx
        chunk_index = len(sub_chunks)

        while current_start < end_idx:
            current_end = min(current_start + self.max_chunk_sentences, end_idx)

            chunk = self._create_single_chunk(
                sentences, current_start, current_end, chunk_index,
                {"source": "split_chunk", "source_type": "content"}
            )
            sub_chunks.append(chunk)

            current_start = current_end
            chunk_index += 1

        return sub_chunks

    def _create_single_chunk(self, sentences: List[str], start_idx: int, end_idx: int, chunk_index: int, source_info: Dict[str, Any]) -> Chunk:
        """Create a single chunk from sentence range."""
        chunk_sentences = sentences[start_idx:end_idx]
        chunk_text = " ".join(chunk_sentences)

        # Enforce character limit
        if len(chunk_text) > self.max_chunk_chars:
            # Truncate to character limit at sentence boundary
            truncated_text = ""
            for sentence in chunk_sentences:
                if len(truncated_text + " " + sentence) <= self.max_chunk_chars:
                    truncated_text += " " + sentence if truncated_text else sentence
                else:
                    break
            chunk_text = truncated_text

        # Calculate semantic metadata
        num_sentences = len(chunk_sentences)
        avg_sentence_length = len(chunk_text) / num_sentences if num_sentences > 0 else 0

        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            source_type=source_info.get("source_type", "content"),
            position=f"sentences {start_idx}-{end_idx-1}",
            offset=start_idx,  # Sentence offset
            length=len(chunk_text),
            extra={
                "chunk_index": chunk_index,
                "sentence_count": num_sentences,
                "start_sentence_index": start_idx,
                "end_sentence_index": end_idx - 1,
                "avg_sentence_length": round(avg_sentence_length, 1),
                "semantic_model": self.semantic_model.value,
                "embedding_model": self.embedding_model,
                "similarity_threshold": self.similarity_threshold,
                "chunker_used": self.name,
                "chunking_strategy": "semantic"
            }
        )

        return Chunk(
            id=f"semantic_{chunk_index}",
            content=chunk_text,
            modality=ModalityType.TEXT,
            metadata=metadata
        )

    def chunk(self, content: Union[str, bytes, Path], source_info: Optional[Dict[str, Any]] = None, **kwargs) -> ChunkingResult:
        """
        Chunk content using semantic similarity analysis.

        Args:
            content: Input content to chunk
            source_info: Source information metadata
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with semantically coherent chunks
        """
        start_time = time.time()

        # Handle different input types
        if isinstance(content, (bytes, Path)):
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            elif isinstance(content, Path):
                content = content.read_text(encoding='utf-8', errors='ignore')
        elif not isinstance(content, str):
            content = str(content)

        if not content.strip():
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used=self.name,
                source_info=source_info or {}
            )

        source_info = source_info or {"source": "string", "source_type": "content"}

        # Initialize semantic model if needed
        if not any([self.embedder, self.nlp, self.vectorizer]):
            self._initialize_semantic_model()

        # Segment into sentences
        sentences = self._segment_sentences(content)

        if len(sentences) < self.min_chunk_sentences:
            # Too few sentences, create single chunk
            chunk = self._create_single_chunk(sentences, 0, len(sentences), 0, source_info)
            processing_time = time.time() - start_time

            return ChunkingResult(
                chunks=[chunk],
                processing_time=processing_time,
                strategy_used=self.name,
                source_info={
                    **source_info,
                    "total_sentences": len(sentences),
                    "semantic_model": self.semantic_model.value,
                    "embedding_model": self.embedding_model
                }
            )

        # Compute sentence embeddings
        embeddings = self._compute_sentence_embeddings(sentences)

        # Calculate similarity scores
        similarities = self._compute_similarity_scores(embeddings)

        # Detect semantic boundaries
        boundaries = self._detect_boundaries(sentences, similarities)

        # Create chunks from boundaries
        chunks = self._create_chunks_from_boundaries(sentences, boundaries, source_info)

        processing_time = time.time() - start_time

        # Calculate statistics
        avg_similarity = np.mean(similarities) if similarities else 0.0
        num_boundaries = len(boundaries)
        boundary_density = num_boundaries / len(sentences) if sentences else 0.0

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used=self.name,
            source_info={
                **source_info,
                "total_sentences": len(sentences),
                "total_boundaries_detected": num_boundaries,
                "boundary_density": round(boundary_density, 3),
                "avg_similarity_score": round(avg_similarity, 3),
                "semantic_model": self.semantic_model.value,
                "embedding_model": self.embedding_model,
                "similarity_threshold": self.similarity_threshold,
                "boundary_detection_method": self.boundary_detection.value,
                "embedding_time": round(self._embedding_time, 3),
                "boundary_detection_time": round(self._boundary_detection_time, 3)
            }
        )

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Stream semantic chunks from content stream.

        Args:
            content_stream: Iterator of content pieces
            source_info: Source information
            **kwargs: Additional parameters

        Yields:
            Individual chunks as they're created
        """
        # Collect content from stream (semantic analysis requires full text)
        collected_content = ""
        for content_piece in content_stream:
            if isinstance(content_piece, bytes):
                content_piece = content_piece.decode('utf-8', errors='ignore')
            collected_content += str(content_piece)

        # Process collected content and yield chunks
        result = self.chunk(collected_content, source_info=source_info, **kwargs)
        for chunk in result.chunks:
            yield chunk

    def adapt_parameters(self, feedback_score: float, feedback_type: str = "quality", **kwargs) -> Dict[str, Any]:
        """
        Adapt semantic chunking parameters based on feedback.

        Args:
            feedback_score: Feedback score (0.0 to 1.0, higher is better)
            feedback_type: Type of feedback ("quality", "performance", "coherence")
            **kwargs: Additional feedback context

        Returns:
            Dictionary of parameter changes made
        """
        old_threshold = self.similarity_threshold
        old_max_sentences = self.max_chunk_sentences
        old_min_sentences = self.min_chunk_sentences

        changes = {}

        if feedback_type == "quality":
            if feedback_score < 0.5:
                # Poor quality - increase sensitivity
                self.similarity_threshold = min(0.9, self.similarity_threshold + 0.1)
                self.min_chunk_sentences = max(2, self.min_chunk_sentences - 1)
                changes.update({
                    "similarity_threshold": {"old": old_threshold, "new": self.similarity_threshold, "reason": "increased sensitivity for quality"},
                    "min_chunk_sentences": {"old": old_min_sentences, "new": self.min_chunk_sentences, "reason": "reduced min size for quality"}
                })
            elif feedback_score > 0.8:
                # Good quality - can be less sensitive for efficiency
                self.similarity_threshold = max(0.5, self.similarity_threshold - 0.05)
                changes["similarity_threshold"] = {"old": old_threshold, "new": self.similarity_threshold, "reason": "reduced sensitivity for efficiency"}

        elif feedback_type == "performance":
            if feedback_score < 0.5:
                # Poor performance - make processing faster
                self.max_chunk_sentences = min(20, self.max_chunk_sentences + 3)
                self.similarity_threshold = max(0.4, self.similarity_threshold - 0.1)
                changes.update({
                    "max_chunk_sentences": {"old": old_max_sentences, "new": self.max_chunk_sentences, "reason": "increased max size for performance"},
                    "similarity_threshold": {"old": old_threshold, "new": self.similarity_threshold, "reason": "reduced sensitivity for performance"}
                })

        elif feedback_type == "coherence":
            if feedback_score < 0.6:
                # Poor coherence - increase strictness
                self.similarity_threshold = min(0.9, self.similarity_threshold + 0.15)
                self.coherence_weight = min(1.0, self.coherence_weight + 0.1)
                changes.update({
                    "similarity_threshold": {"old": old_threshold, "new": self.similarity_threshold, "reason": "increased for coherence"},
                    "coherence_weight": {"old": kwargs.get("old_coherence_weight", 0.3), "new": self.coherence_weight, "reason": "increased coherence importance"}
                })

        # Handle specific feedback
        if kwargs.get("chunks_too_small"):
            self.min_chunk_sentences = max(1, self.min_chunk_sentences - 1)
            changes["min_chunk_sentences"] = {"old": old_min_sentences, "new": self.min_chunk_sentences, "reason": "reduced due to small chunks feedback"}
        elif kwargs.get("chunks_too_large"):
            self.max_chunk_sentences = max(5, self.max_chunk_sentences - 2)
            changes["max_chunk_sentences"] = {"old": old_max_sentences, "new": self.max_chunk_sentences, "reason": "reduced due to large chunks feedback"}

        # Record adaptation
        if changes:
            adaptation_record = {
                "timestamp": time.time(),
                "feedback_score": feedback_score,
                "feedback_type": feedback_type,
                "changes": changes,
                "parameters_after": {
                    "similarity_threshold": self.similarity_threshold,
                    "min_chunk_sentences": self.min_chunk_sentences,
                    "max_chunk_sentences": self.max_chunk_sentences,
                    "coherence_weight": self.coherence_weight
                },
                "context": kwargs
            }
            self._adaptation_history.append(adaptation_record)

            logger.info(f"Adapted semantic chunker parameters based on {feedback_type} feedback ({feedback_score:.2f}): {len(changes)} changes made")

        return changes

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self._adaptation_history.copy()

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "name": "semantic",
            "semantic_model": self.semantic_model.value,
            "embedding_model": self.embedding_model,
            "similarity_threshold": self.similarity_threshold,
            "min_chunk_sentences": self.min_chunk_sentences,
            "max_chunk_sentences": self.max_chunk_sentences,
            "boundary_detection": self.boundary_detection.value,
            "context_window_size": self.context_window_size,
            "max_chunk_chars": self.max_chunk_chars,
            "coherence_weight": self.coherence_weight,
            "performance_stats": {
                "total_sentences_processed": self._total_sentences_processed,
                "embedding_time": self._embedding_time,
                "boundary_detection_time": self._boundary_detection_time,
                "avg_embedding_speed": (
                    self._total_sentences_processed / self._embedding_time
                    if self._embedding_time > 0 else 0
                )
            }
        }
