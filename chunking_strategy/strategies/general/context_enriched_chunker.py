"""
Context-Enriched Chunking strategy.

The Context-Enriched Chunker uses advanced NLP techniques to create semantically
coherent chunks that preserve meaning and context. It leverages semantic similarity,
entity recognition, topic modeling, and context analysis to determine optimal
chunk boundaries that maintain conceptual integrity.

Key features:
- Semantic boundary detection using embeddings
- Named Entity Recognition (NER) and preservation
- Topic coherence optimization
- Coreference resolution and context preservation
- Meaning-preserving chunk boundaries
- Rich semantic metadata generation
- Context-aware chunk sizing
- Entity relationship mapping
"""

import logging
import re
import json
import time
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP imports with fallbacks
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import numpy as np

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.core.streaming import StreamableChunker
from chunking_strategy.core.adaptive import AdaptableChunker


@dataclass
class SemanticEntity:
    """Represents a named entity with semantic information."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    context: str = ""
    related_entities: List[str] = None

    def __post_init__(self):
        if self.related_entities is None:
            self.related_entities = []


@dataclass
class TopicInfo:
    """Information about a topic within text."""
    topic_id: int
    keywords: List[str]
    weight: float
    coherence_score: float
    sentences: List[int]  # Sentence indices
    entities: List[str] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = []


@dataclass
class ContextualChunk:
    """Enhanced chunk with contextual information."""
    content: str
    semantic_entities: List[SemanticEntity]
    topics: List[TopicInfo]
    coherence_score: float
    context_preservation_score: float
    boundary_quality_score: float
    semantic_fingerprint: List[float]  # Embedding representation
    cross_references: List[str] = None  # References to other chunks

    def __post_init__(self):
        if self.cross_references is None:
            self.cross_references = []


@dataclass
class SemanticBoundary:
    """Represents a potential semantic boundary."""
    position: int  # Sentence index
    boundary_type: str  # 'topic_shift', 'entity_boundary', 'semantic_break'
    confidence: float
    reason: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@register_chunker(
    name="context_enriched",
    category="general",
    complexity=ComplexityLevel.HIGH,
    speed=SpeedLevel.SLOW,
    memory=MemoryUsage.HIGH,
    supported_formats=["txt", "md", "html", "json", "xml"],
    dependencies=["spacy", "nltk", "scikit-learn"],
    description="Context-Enriched Chunker with semantic boundary detection and meaning preservation",
    use_cases=["semantic_analysis", "document_understanding", "context_preservation", "academic_text", "knowledge_extraction"]
)
class ContextEnrichedChunker(StreamableChunker, AdaptableChunker):
    """
    Context-Enriched Chunker that creates semantically coherent chunks
    by analyzing context, entities, topics, and semantic relationships.

    This chunker goes beyond simple text boundaries to understand meaning
    and preserve context across chunks while maintaining semantic integrity.
    """

    def __init__(
        self,
        # Core chunking parameters
        target_chunk_size: int = 2000,
        min_chunk_size: int = 500,
        max_chunk_size: int = 8000,
        overlap_size: int = 100,

        # Semantic analysis parameters
        semantic_similarity_threshold: float = 0.7,
        topic_coherence_threshold: float = 0.6,
        entity_preservation_mode: str = "strict",  # "strict", "moderate", "loose"
        context_window_size: int = 3,  # Sentences to consider for context

        # NLP model parameters
        spacy_model: str = "en_core_web_sm",
        enable_ner: bool = True,
        enable_topic_modeling: bool = True,
        enable_coreference: bool = False,  # Requires more advanced models

        # Boundary detection parameters
        boundary_detection_method: str = "multi_modal",  # "semantic", "topic", "entity", "multi_modal"
        min_topic_coherence: float = 0.5,
        entity_boundary_weight: float = 0.3,
        topic_boundary_weight: float = 0.4,
        semantic_boundary_weight: float = 0.3,

        # Quality parameters
        preserve_sentence_integrity: bool = True,
        avoid_entity_splitting: bool = True,
        maintain_topic_coherence: bool = True,
        context_preservation_priority: str = "balanced",  # "speed", "balanced", "quality"

        # Advanced features
        enable_cross_references: bool = True,
        generate_semantic_fingerprints: bool = True,
        extract_key_phrases: bool = True,
        analyze_sentiment: bool = False,

        **kwargs
    ):
        """
        Initialize Context-Enriched Chunker.

        Args:
            target_chunk_size: Target size for chunks in characters
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            overlap_size: Overlap between chunks for context preservation
            semantic_similarity_threshold: Threshold for semantic similarity
            topic_coherence_threshold: Threshold for topic coherence
            entity_preservation_mode: How strictly to preserve entities
            context_window_size: Number of sentences for context analysis
            spacy_model: SpaCy model to use for NLP
            enable_ner: Enable Named Entity Recognition
            enable_topic_modeling: Enable topic modeling
            enable_coreference: Enable coreference resolution
            boundary_detection_method: Method for detecting boundaries
            min_topic_coherence: Minimum topic coherence required
            entity_boundary_weight: Weight for entity boundaries
            topic_boundary_weight: Weight for topic boundaries
            semantic_boundary_weight: Weight for semantic boundaries
            preserve_sentence_integrity: Keep sentences intact
            avoid_entity_splitting: Avoid splitting entities
            maintain_topic_coherence: Maintain topic coherence
            context_preservation_priority: Priority for context preservation
            enable_cross_references: Enable cross-chunk references
            generate_semantic_fingerprints: Generate semantic embeddings
            extract_key_phrases: Extract key phrases from chunks
            analyze_sentiment: Analyze sentiment (optional)
            **kwargs: Additional parameters
        """
        super().__init__(
            name="context_enriched",
            category="general",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        # Core parameters
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

        # Semantic parameters
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.topic_coherence_threshold = topic_coherence_threshold
        self.entity_preservation_mode = entity_preservation_mode
        self.context_window_size = context_window_size

        # NLP parameters
        self.spacy_model = spacy_model
        self.enable_ner = enable_ner
        self.enable_topic_modeling = enable_topic_modeling
        self.enable_coreference = enable_coreference

        # Boundary detection
        self.boundary_detection_method = boundary_detection_method
        self.min_topic_coherence = min_topic_coherence
        self.entity_boundary_weight = entity_boundary_weight
        self.topic_boundary_weight = topic_boundary_weight
        self.semantic_boundary_weight = semantic_boundary_weight

        # Quality parameters
        self.preserve_sentence_integrity = preserve_sentence_integrity
        self.avoid_entity_splitting = avoid_entity_splitting
        self.maintain_topic_coherence = maintain_topic_coherence
        self.context_preservation_priority = context_preservation_priority

        # Advanced features
        self.enable_cross_references = enable_cross_references
        self.generate_semantic_fingerprints = generate_semantic_fingerprints
        self.extract_key_phrases = extract_key_phrases
        self.analyze_sentiment = analyze_sentiment

        # Initialize NLP components
        self.nlp = None
        self.vectorizer = None
        self.lemmatizer = None
        self.stopwords = set()

        # Processing state
        self.sentences = []
        self.entities = []
        self.topics = []
        self.semantic_boundaries = []

        # Adaptation history
        self.adaptation_history = []

        self.logger = logging.getLogger(__name__)
        self._initialize_nlp_components()

    def _initialize_nlp_components(self):
        """Initialize NLP components with fallbacks."""
        try:
            # Initialize SpaCy
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load(self.spacy_model)
                    self.logger.info(f"Loaded SpaCy model: {self.spacy_model}")
                except OSError:
                    try:
                        # Fallback to smaller model
                        self.nlp = spacy.load("en_core_web_sm")
                        self.logger.warning("Fallback to en_core_web_sm model")
                    except OSError:
                        self.logger.warning("No SpaCy model available, using basic NLP")
                        self.nlp = None

            # Initialize NLTK
            if NLTK_AVAILABLE:
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    nltk.download('wordnet', quiet=True)
                    self.stopwords = set(stopwords.words('english'))
                    self.lemmatizer = WordNetLemmatizer()
                    self.logger.info("NLTK components initialized")
                except Exception as e:
                    self.logger.warning(f"NLTK initialization issues: {e}")

            # Initialize scikit-learn
            if SKLEARN_AVAILABLE:
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.logger.info("Scikit-learn vectorizer initialized")

        except Exception as e:
            self.logger.error(f"NLP initialization error: {e}")

    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using available NLP tools."""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        elif NLTK_AVAILABLE:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        else:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]

    def _extract_entities(self, text: str) -> List[SemanticEntity]:
        """Extract named entities from text."""
        if not self.enable_ner or not self.nlp:
            return []

        entities = []
        doc = self.nlp(text)

        for ent in doc.ents:
            entity = SemanticEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,  # SpaCy doesn't provide confidence by default
                context=text[max(0, ent.start_char-50):ent.end_char+50]
            )
            entities.append(entity)

        return entities

    def _analyze_topics(self, sentences: List[str]) -> List[TopicInfo]:
        """Analyze topics in the text using TF-IDF and clustering."""
        if not self.enable_topic_modeling or not SKLEARN_AVAILABLE or len(sentences) < 3:
            return []

        try:
            # Vectorize sentences
            tfidf_matrix = self.vectorizer.fit_transform(sentences)

            # Determine number of topics (heuristic)
            n_topics = min(max(2, len(sentences) // 3), 5)

            # Cluster sentences into topics
            kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
            topic_labels = kmeans.fit_predict(tfidf_matrix)

            topics = []
            feature_names = self.vectorizer.get_feature_names_out()

            for topic_id in range(n_topics):
                # Get sentences for this topic
                topic_sentences = [i for i, label in enumerate(topic_labels) if label == topic_id]

                if not topic_sentences:
                    continue

                # Get top keywords for this topic
                topic_center = kmeans.cluster_centers_[topic_id]
                top_indices = topic_center.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_indices if topic_center[i] > 0.1]

                # Calculate coherence score (simplified)
                coherence_score = self._calculate_topic_coherence(topic_sentences, sentences)

                topic = TopicInfo(
                    topic_id=topic_id,
                    keywords=keywords[:5],  # Top 5 keywords
                    weight=len(topic_sentences) / len(sentences),
                    coherence_score=coherence_score,
                    sentences=topic_sentences
                )
                topics.append(topic)

            return topics

        except Exception as e:
            self.logger.warning(f"Topic analysis failed: {e}")
            return []

    def _calculate_topic_coherence(self, topic_sentences: List[int], sentences: List[str]) -> float:
        """Calculate coherence score for a topic."""
        if len(topic_sentences) < 2:
            return 1.0

        try:
            topic_texts = [sentences[i] for i in topic_sentences]

            if SKLEARN_AVAILABLE:
                # Use TF-IDF similarity
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(topic_texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)

                # Average pairwise similarity
                n = len(topic_sentences)
                total_similarity = 0
                count = 0

                for i in range(n):
                    for j in range(i + 1, n):
                        total_similarity += similarity_matrix[i][j]
                        count += 1

                return total_similarity / count if count > 0 else 0.0
            else:
                # Fallback: keyword overlap
                all_words = []
                for text in topic_texts:
                    words = set(text.lower().split())
                    all_words.append(words)

                if len(all_words) < 2:
                    return 1.0

                # Calculate average overlap
                total_overlap = 0
                count = 0

                for i in range(len(all_words)):
                    for j in range(i + 1, len(all_words)):
                        overlap = len(all_words[i] & all_words[j])
                        union = len(all_words[i] | all_words[j])
                        if union > 0:
                            total_overlap += overlap / union
                            count += 1

                return total_overlap / count if count > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Coherence calculation failed: {e}")
            return 0.5

    def _detect_semantic_boundaries(self, sentences: List[str]) -> List[SemanticBoundary]:
        """Detect semantic boundaries between sentences."""
        boundaries = []

        if len(sentences) < 2:
            return boundaries

        # Method 1: Semantic similarity boundaries
        if self.boundary_detection_method in ["semantic", "multi_modal"]:
            boundaries.extend(self._detect_similarity_boundaries(sentences))

        # Method 2: Topic shift boundaries
        if self.boundary_detection_method in ["topic", "multi_modal"] and self.topics:
            boundaries.extend(self._detect_topic_boundaries(sentences))

        # Method 3: Entity boundaries
        if self.boundary_detection_method in ["entity", "multi_modal"] and self.entities:
            boundaries.extend(self._detect_entity_boundaries(sentences))

        # Sort boundaries by position and filter by confidence
        boundaries.sort(key=lambda x: x.position)
        return [b for b in boundaries if b.confidence >= 0.5]

    def _detect_similarity_boundaries(self, sentences: List[str]) -> List[SemanticBoundary]:
        """Detect boundaries based on semantic similarity drops."""
        boundaries = []

        if not SKLEARN_AVAILABLE or len(sentences) < 3:
            return boundaries

        try:
            # Create sentence embeddings using TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english')
            sentence_vectors = vectorizer.fit_transform(sentences)

            # Calculate similarity between adjacent sentences
            for i in range(len(sentences) - 1):
                similarity = cosine_similarity(
                    sentence_vectors[i:i+1],
                    sentence_vectors[i+1:i+2]
                )[0][0]

                # If similarity drops below threshold, it's a potential boundary
                if similarity < self.semantic_similarity_threshold:
                    confidence = 1.0 - similarity  # Lower similarity = higher boundary confidence

                    boundary = SemanticBoundary(
                        position=i + 1,
                        boundary_type="semantic_break",
                        confidence=confidence * self.semantic_boundary_weight,
                        reason=f"Low semantic similarity: {similarity:.3f}",
                        metadata={"similarity_score": similarity}
                    )
                    boundaries.append(boundary)

        except Exception as e:
            self.logger.warning(f"Similarity boundary detection failed: {e}")

        return boundaries

    def _detect_topic_boundaries(self, sentences: List[str]) -> List[SemanticBoundary]:
        """Detect boundaries based on topic shifts."""
        boundaries = []

        if not self.topics:
            return boundaries

        # Create sentence-to-topic mapping
        sentence_topics = {}
        for topic in self.topics:
            for sent_idx in topic.sentences:
                sentence_topics[sent_idx] = topic.topic_id

        # Find topic transitions
        for i in range(len(sentences) - 1):
            current_topic = sentence_topics.get(i)
            next_topic = sentence_topics.get(i + 1)

            if current_topic is not None and next_topic is not None and current_topic != next_topic:
                # Calculate boundary confidence based on topic coherence
                current_topic_obj = next((t for t in self.topics if t.topic_id == current_topic), None)
                next_topic_obj = next((t for t in self.topics if t.topic_id == next_topic), None)

                if current_topic_obj and next_topic_obj:
                    confidence = (current_topic_obj.coherence_score + next_topic_obj.coherence_score) / 2

                    boundary = SemanticBoundary(
                        position=i + 1,
                        boundary_type="topic_shift",
                        confidence=confidence * self.topic_boundary_weight,
                        reason=f"Topic shift: {current_topic} -> {next_topic}",
                        metadata={
                            "from_topic": current_topic,
                            "to_topic": next_topic,
                            "topic_coherence": confidence
                        }
                    )
                    boundaries.append(boundary)

        return boundaries

    def _detect_entity_boundaries(self, sentences: List[str]) -> List[SemanticBoundary]:
        """Detect boundaries based on entity transitions."""
        boundaries = []

        if not self.entities:
            return boundaries

        # Map entities to sentences (simplified)
        sentence_entities = defaultdict(set)

        # Calculate character positions for sentences
        char_pos = 0
        sentence_positions = []
        for sentence in sentences:
            sentence_positions.append((char_pos, char_pos + len(sentence)))
            char_pos += len(sentence) + 1  # +1 for space/separator

        # Map entities to sentences
        for entity in self.entities:
            for i, (start, end) in enumerate(sentence_positions):
                if entity.start >= start and entity.end <= end:
                    sentence_entities[i].add(entity.label)
                    break

        # Find entity transitions
        for i in range(len(sentences) - 1):
            current_entities = sentence_entities[i]
            next_entities = sentence_entities[i + 1]

            if current_entities or next_entities:
                # Calculate entity overlap
                overlap = len(current_entities & next_entities)
                total = len(current_entities | next_entities)

                if total > 0:
                    entity_similarity = overlap / total

                    # Low entity similarity indicates boundary
                    if entity_similarity < 0.5:
                        confidence = 1.0 - entity_similarity

                        boundary = SemanticBoundary(
                            position=i + 1,
                            boundary_type="entity_boundary",
                            confidence=confidence * self.entity_boundary_weight,
                            reason=f"Entity transition: {current_entities} -> {next_entities}",
                            metadata={
                                "entity_similarity": entity_similarity,
                                "current_entities": list(current_entities),
                                "next_entities": list(next_entities)
                            }
                        )
                        boundaries.append(boundary)

        return boundaries

    def _create_contextual_chunks(self, sentences: List[str]) -> List[ContextualChunk]:
        """Create contextual chunks based on semantic boundaries."""
        if not sentences:
            return []

        # If no boundaries detected, use size-based chunking with semantic awareness
        if not self.semantic_boundaries:
            return self._create_size_based_semantic_chunks(sentences)

        chunks = []
        current_sentences = []
        current_size = 0

        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)

            # Check if we should create a chunk at this position
            should_chunk = False

            # Check for semantic boundary
            boundary_at_position = any(b.position == i for b in self.semantic_boundaries)

            # Size-based constraints
            if current_size + sentence_size > self.max_chunk_size:
                should_chunk = True
            elif boundary_at_position and current_size >= self.min_chunk_size:
                should_chunk = True
            elif current_size >= self.target_chunk_size:
                should_chunk = True

            if should_chunk and current_sentences:
                # Create chunk from current sentences
                chunk = self._create_chunk_from_sentences(current_sentences)
                if chunk:
                    chunks.append(chunk)

                # Start new chunk with overlap if configured
                if self.overlap_size > 0:
                    overlap_sentences = self._get_overlap_sentences(current_sentences, self.overlap_size)
                    current_sentences = overlap_sentences + [sentence]
                    current_size = sum(len(s) for s in current_sentences)
                else:
                    current_sentences = [sentence]
                    current_size = sentence_size
            else:
                current_sentences.append(sentence)
                current_size += sentence_size

        # Handle remaining sentences
        if current_sentences:
            chunk = self._create_chunk_from_sentences(current_sentences)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_size_based_semantic_chunks(self, sentences: List[str]) -> List[ContextualChunk]:
        """Create size-based chunks with semantic awareness."""
        chunks = []
        current_sentences = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.max_chunk_size and current_sentences:
                # Create chunk
                chunk = self._create_chunk_from_sentences(current_sentences)
                if chunk:
                    chunks.append(chunk)

                current_sentences = [sentence]
                current_size = sentence_size
            else:
                current_sentences.append(sentence)
                current_size += sentence_size

        # Handle remaining sentences
        if current_sentences:
            chunk = self._create_chunk_from_sentences(current_sentences)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_chunk_from_sentences(self, sentences: List[str]) -> Optional[ContextualChunk]:
        """Create a contextual chunk from a list of sentences."""
        if not sentences:
            return None

        content = " ".join(sentences)

        # Extract entities for this chunk
        chunk_entities = self._extract_entities(content)

        # Find relevant topics
        chunk_topics = self._get_chunk_topics(sentences)

        # Calculate quality scores
        coherence_score = self._calculate_chunk_coherence(sentences, chunk_topics)
        context_preservation_score = self._calculate_context_preservation(sentences)
        boundary_quality_score = self._calculate_boundary_quality(sentences)

        # Generate semantic fingerprint
        semantic_fingerprint = self._generate_semantic_fingerprint(content)

        return ContextualChunk(
            content=content,
            semantic_entities=chunk_entities,
            topics=chunk_topics,
            coherence_score=coherence_score,
            context_preservation_score=context_preservation_score,
            boundary_quality_score=boundary_quality_score,
            semantic_fingerprint=semantic_fingerprint
        )

    def _get_chunk_topics(self, sentences: List[str]) -> List[TopicInfo]:
        """Get topics relevant to the given sentences."""
        if not self.topics:
            return []

        # Map sentence indices to find relevant topics
        sentence_indices = set(range(len(sentences)))
        relevant_topics = []

        for topic in self.topics:
            topic_sentence_indices = set(topic.sentences)
            overlap = len(sentence_indices & topic_sentence_indices)

            if overlap > 0:
                # Calculate topic relevance
                relevance = overlap / len(topic.sentences)
                if relevance >= 0.3:  # At least 30% of topic sentences
                    relevant_topics.append(topic)

        return relevant_topics

    def _calculate_chunk_coherence(self, sentences: List[str], topics: List[TopicInfo]) -> float:
        """Calculate coherence score for a chunk."""
        if not sentences:
            return 0.0

        if len(sentences) == 1:
            return 1.0

        # Base coherence on topic coherence
        if topics:
            avg_topic_coherence = sum(t.coherence_score for t in topics) / len(topics)
            return avg_topic_coherence

        # Fallback: calculate based on sentence similarity
        return self._calculate_topic_coherence(list(range(len(sentences))), sentences)

    def _calculate_context_preservation(self, sentences: List[str]) -> float:
        """Calculate how well context is preserved in the chunk."""
        if len(sentences) <= 1:
            return 1.0

        # Simple heuristic: longer chunks preserve more context
        total_chars = sum(len(s) for s in sentences)

        if total_chars >= self.target_chunk_size:
            return 1.0
        elif total_chars >= self.min_chunk_size:
            return 0.8
        else:
            return 0.6

    def _calculate_boundary_quality(self, sentences: List[str]) -> float:
        """Calculate the quality of chunk boundaries."""
        if not sentences:
            return 0.0

        # Check if chunk ends with complete sentences
        last_sentence = sentences[-1].strip()
        if last_sentence.endswith(('.', '!', '?', ':', ';')):
            sentence_quality = 1.0
        else:
            sentence_quality = 0.5

        # Check entity preservation
        entity_quality = 1.0
        if self.avoid_entity_splitting and self.entities:
            content = " ".join(sentences)
            # Simple check: if entities are mentioned completely
            for entity in self.entities:
                if entity.text in content:
                    # Entity is preserved
                    continue
                elif any(word in content for word in entity.text.split()):
                    # Partial entity - penalize
                    entity_quality *= 0.8

        return (sentence_quality + entity_quality) / 2

    def _generate_semantic_fingerprint(self, content: str) -> List[float]:
        """Generate semantic fingerprint for content."""
        if not self.generate_semantic_fingerprints:
            return []

        try:
            if SKLEARN_AVAILABLE:
                # Use TF-IDF as semantic fingerprint
                vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform([content])
                return tfidf_matrix.toarray()[0].tolist()
            else:
                # Fallback: word frequency fingerprint
                words = content.lower().split()
                word_count = Counter(words)
                # Get top 20 words
                top_words = word_count.most_common(20)
                return [count for word, count in top_words]

        except Exception as e:
            self.logger.warning(f"Fingerprint generation failed: {e}")
            return []

    def _get_overlap_sentences(self, sentences: List[str], overlap_size: int) -> List[str]:
        """Get sentences for overlap based on character count."""
        if not sentences or overlap_size <= 0:
            return []

        overlap_sentences = []
        current_size = 0

        # Take sentences from the end
        for sentence in reversed(sentences):
            if current_size + len(sentence) <= overlap_size:
                overlap_sentences.insert(0, sentence)  # Insert at beginning
                current_size += len(sentence)
            else:
                break

        return overlap_sentences

    def chunk(
        self,
        content: Union[str, bytes, 'Path'],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Perform context-enriched chunking with semantic boundary detection.

        Args:
            content: Content to chunk (string, bytes, or file path)
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with semantically coherent chunks and rich metadata
        """
        start_time = time.time()

        # Convert content to string
        text_content = self._prepare_content(content)
        if not text_content.strip():
            return ChunkingResult(chunks=[], processing_time=0.0, strategy_used="context_enriched")

        try:
            # Step 1: Segment into sentences
            self.sentences = self._segment_sentences(text_content)
            self.logger.debug(f"Segmented into {len(self.sentences)} sentences")

            # Step 2: Extract entities
            if self.enable_ner:
                self.entities = self._extract_entities(text_content)
                self.logger.debug(f"Extracted {len(self.entities)} entities")

            # Step 3: Analyze topics
            if self.enable_topic_modeling:
                self.topics = self._analyze_topics(self.sentences)
                self.logger.debug(f"Identified {len(self.topics)} topics")

            # Step 4: Detect semantic boundaries
            self.semantic_boundaries = self._detect_semantic_boundaries(self.sentences)
            self.logger.debug(f"Detected {len(self.semantic_boundaries)} semantic boundaries")

            # Step 5: Create contextual chunks
            contextual_chunks = self._create_contextual_chunks(self.sentences)

            # Step 6: Convert to standard chunks
            chunks = []
            for i, ctx_chunk in enumerate(contextual_chunks):
                # Create enhanced metadata
                metadata = ChunkMetadata(
                    source=source_info.get('source', 'unknown') if source_info else 'unknown',
                    chunker_used="context_enriched",
                    extra={
                        # Semantic information
                        "entities": [asdict(entity) for entity in ctx_chunk.semantic_entities],
                        "topics": [asdict(topic) for topic in ctx_chunk.topics],
                        "coherence_score": ctx_chunk.coherence_score,
                        "context_preservation_score": ctx_chunk.context_preservation_score,
                        "boundary_quality_score": ctx_chunk.boundary_quality_score,

                        # Chunk information
                        "chunk_index": i,
                        "total_chunks": len(contextual_chunks),
                        "sentence_count": len(ctx_chunk.content.split('. ')),
                        "word_count": len(ctx_chunk.content.split()),

                        # Semantic fingerprint
                        "semantic_fingerprint": ctx_chunk.semantic_fingerprint if self.generate_semantic_fingerprints else [],

                        # Processing information
                        "processing_method": self.boundary_detection_method,
                        "nlp_enabled": {
                            "ner": self.enable_ner and bool(self.nlp),
                            "topic_modeling": self.enable_topic_modeling,
                            "semantic_boundaries": len(self.semantic_boundaries) > 0
                        }
                    }
                )

                chunk = Chunk(
                    id=f"context_enriched_{i}",
                    content=ctx_chunk.content,
                    modality=ModalityType.TEXT,
                    metadata=metadata
                )
                chunks.append(chunk)

            processing_time = time.time() - start_time

            # Create result with enhanced metadata
            result = ChunkingResult(
                chunks=chunks,
                processing_time=processing_time,
                strategy_used="context_enriched",
                source_info={
                    **(source_info or {}),
                    "context_enriched_metadata": {
                        "total_sentences": len(self.sentences),
                        "total_entities": len(self.entities),
                        "total_topics": len(self.topics),
                        "semantic_boundaries": len(self.semantic_boundaries),
                        "boundary_types": list(set(b.boundary_type for b in self.semantic_boundaries)),
                        "avg_coherence_score": sum(c.coherence_score for c in contextual_chunks) / len(contextual_chunks) if contextual_chunks else 0,
                        "processing_time": processing_time,
                        "nlp_components_used": {
                            "spacy": bool(self.nlp),
                            "nltk": NLTK_AVAILABLE,
                            "sklearn": SKLEARN_AVAILABLE
                        }
                    }
                }
            )

            self.logger.info(f"Context-enriched chunking completed: {len(chunks)} chunks in {processing_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Context-enriched chunking failed: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(text_content, source_info)

    def _prepare_content(self, content: Union[str, bytes, 'Path']) -> str:
        """Prepare content for processing."""
        if hasattr(content, 'read_text'):  # Path object
            return content.read_text(encoding='utf-8')
        elif isinstance(content, bytes):
            return content.decode('utf-8', errors='ignore')
        elif isinstance(content, str):
            return content
        else:
            return str(content)

    def _fallback_chunking(self, content: str, source_info: Optional[Dict[str, Any]]) -> ChunkingResult:
        """Fallback to simple sentence-based chunking."""
        sentences = self._segment_sentences(content)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.target_chunk_size and current_chunk:
                # Create chunk
                chunk_content = " ".join(current_chunk)
                metadata = ChunkMetadata(
                    source=source_info.get('source', 'unknown') if source_info else 'unknown',
                    chunker_used="context_enriched_fallback"
                )

                chunk = Chunk(
                    id=f"fallback_{len(chunks)}",
                    content=chunk_content,
                    modality=ModalityType.TEXT,
                    metadata=metadata
                )
                chunks.append(chunk)

                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Handle remaining content
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            metadata = ChunkMetadata(
                source=source_info.get('source', 'unknown') if source_info else 'unknown',
                chunker_used="context_enriched_fallback"
            )

            chunk = Chunk(
                id=f"fallback_{len(chunks)}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=metadata
            )
            chunks.append(chunk)

        return ChunkingResult(
            chunks=chunks,
            processing_time=0.0,
            strategy_used="context_enriched_fallback"
        )

    def chunk_stream(self, content_stream, **kwargs):
        """Chunk data from a stream (simplified implementation)."""
        # Collect stream data
        data_parts = []
        for chunk in content_stream:
            if isinstance(chunk, str):
                data_parts.append(chunk)
            else:
                data_parts.append(chunk.decode('utf-8', errors='ignore'))

        # Process as single content
        full_content = ''.join(data_parts)
        result = self.chunk(full_content, **kwargs)

        # Yield chunks
        for chunk in result.chunks:
            yield chunk

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ["txt", "md", "html", "json", "xml", "any"]

    def estimate_chunks(self, content: Union[str, bytes, 'Path']) -> int:
        """Estimate number of chunks that will be generated."""
        text_content = self._prepare_content(content)
        content_size = len(text_content)

        # Rough estimation based on target chunk size
        estimated = max(1, content_size // self.target_chunk_size)
        return estimated

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """Adapt parameters based on feedback."""
        adaptation = {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "current_params": {
                "semantic_similarity_threshold": self.semantic_similarity_threshold,
                "topic_coherence_threshold": self.topic_coherence_threshold,
                "target_chunk_size": self.target_chunk_size
            }
        }

        # Adapt based on feedback
        if feedback_score < 0.5:
            # Poor performance - adjust thresholds
            if feedback_type == "coherence":
                self.topic_coherence_threshold = max(0.3, self.topic_coherence_threshold - 0.1)
            elif feedback_type == "boundary":
                self.semantic_similarity_threshold = max(0.5, self.semantic_similarity_threshold - 0.1)
        elif feedback_score > 0.8:
            # Good performance - can be more strict
            if feedback_type == "coherence":
                self.topic_coherence_threshold = min(0.8, self.topic_coherence_threshold + 0.05)
            elif feedback_type == "boundary":
                self.semantic_similarity_threshold = min(0.9, self.semantic_similarity_threshold + 0.05)

        adaptation["adapted_params"] = {
            "semantic_similarity_threshold": self.semantic_similarity_threshold,
            "topic_coherence_threshold": self.topic_coherence_threshold,
            "target_chunk_size": self.target_chunk_size
        }

        self.adaptation_history.append(adaptation)
        self.logger.info(f"Parameters adapted based on {feedback_type} feedback: {feedback_score}")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self.adaptation_history.copy()
