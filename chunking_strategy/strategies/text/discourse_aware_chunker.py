"""
Discourse-Aware Semantic Chunker.

This module implements advanced semantic chunking with explicit discourse marker detection,
topic modeling, entity boundary preservation, and coherence analysis. It builds upon
the existing semantic chunking infrastructure to provide even better semantic coherence.

Key enhancements:
- Comprehensive discourse marker detection
- Enhanced topic boundary detection
- Improved entity boundary preservation
- Multi-layered coherence analysis
- Integration with existing semantic infrastructure
"""

import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from enum import Enum
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

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
from chunking_strategy.strategies.text.semantic_chunker import SemanticChunker, SemanticBoundary

logger = logging.getLogger(__name__)


class DiscourseMarkerType(str, Enum):
    """Types of discourse markers."""
    CONTRAST = "contrast"           # however, but, nevertheless, on the other hand
    CONTINUATION = "continuation"   # furthermore, moreover, additionally, also
    CAUSATION = "causation"        # therefore, consequently, as a result, because
    SEQUENCE = "sequence"          # first, then, next, finally, in conclusion
    EXEMPLIFICATION = "exemplification"  # for example, for instance, such as
    EMPHASIS = "emphasis"          # indeed, in fact, clearly, obviously
    SUMMARY = "summary"            # in summary, to summarize, overall, in brief
    COMPARISON = "comparison"      # similarly, likewise, in comparison, compared to
    ELABORATION = "elaboration"    # specifically, in particular, namely, that is


@dataclass
class DiscourseMarker:
    """Represents a discourse marker found in text."""
    text: str
    position: int
    marker_type: DiscourseMarkerType
    sentence_index: int
    confidence: float
    boundary_strength: float  # How strong this marker is as a chunk boundary


@dataclass
class EnhancedSemanticBoundary(SemanticBoundary):
    """Enhanced semantic boundary with discourse information."""
    discourse_markers: List[DiscourseMarker]
    topic_shift_score: float
    entity_preservation_score: float
    overall_coherence_score: float


@register_chunker(
    name="discourse_aware",
    category="text",
    description="Advanced semantic chunking with discourse marker detection and multi-layered coherence analysis",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "html", "json", "xml", "rtf"],
    complexity=ComplexityLevel.HIGH,
    dependencies=["sentence-transformers"],
    optional_dependencies=["spacy", "nltk", "scikit-learn", "transformers"],
    speed=SpeedLevel.SLOW,
    memory=MemoryUsage.HIGH,
    quality=0.95,  # Very high quality due to comprehensive analysis
    parameters_schema={
        "discourse_weight": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.3,
            "description": "Weight for discourse markers in boundary detection"
        },
        "topic_weight": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.4,
            "description": "Weight for topic modeling in boundary detection"
        },
        "entity_weight": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.2,
            "description": "Weight for entity boundary preservation"
        },
        "coherence_weight": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.1,
            "description": "Weight for overall coherence analysis"
        },
        "min_boundary_strength": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.6,
            "description": "Minimum boundary strength to create a chunk split"
        },
        "preserve_entity_boundaries": {
            "type": "boolean",
            "default": True,
            "description": "Avoid breaking chunks in the middle of named entities"
        },
        "detect_topic_shifts": {
            "type": "boolean",
            "default": True,
            "description": "Use topic modeling to detect semantic shifts"
        },
        "discourse_marker_sensitivity": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.7,
            "description": "Sensitivity to discourse markers (higher = more splits)"
        }
    },
    default_parameters={
        "semantic_model": "sentence_transformer",
        "embedding_model": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.7,
        "min_chunk_sentences": 3,
        "max_chunk_sentences": 15,
        "discourse_weight": 0.3,
        "topic_weight": 0.4,
        "entity_weight": 0.2,
        "coherence_weight": 0.1,
        "min_boundary_strength": 0.6,
        "preserve_entity_boundaries": True,
        "detect_topic_shifts": True,
        "discourse_marker_sensitivity": 0.7
    },
    use_cases=["advanced_RAG", "semantic_analysis", "topic_modeling", "content_understanding", "discourse_analysis"],
    best_for=["academic_papers", "technical_documentation", "long_articles", "complex_narratives", "analytical_text"],
    limitations=["computationally intensive", "requires multiple NLP models", "slower processing"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=False
)
class DiscourseAwareChunker(SemanticChunker):
    """
    Discourse-aware semantic chunker that combines multiple layers of analysis:
    1. Discourse marker detection for natural transition points
    2. Topic modeling for semantic coherence
    3. Entity boundary preservation
    4. Multi-factor coherence scoring
    """

    def __init__(
        self,
        discourse_weight: float = 0.3,
        topic_weight: float = 0.4,
        entity_weight: float = 0.2,
        coherence_weight: float = 0.1,
        min_boundary_strength: float = 0.6,
        preserve_entity_boundaries: bool = True,
        detect_topic_shifts: bool = True,
        discourse_marker_sensitivity: float = 0.7,
        **kwargs
    ):
        # Initialize parent SemanticChunker
        super().__init__(**kwargs)

        # Override the name to match our registration
        self.name = "discourse_aware"

        # Discourse-aware parameters
        self.discourse_weight = discourse_weight
        self.topic_weight = topic_weight
        self.entity_weight = entity_weight
        self.coherence_weight = coherence_weight
        self.min_boundary_strength = min_boundary_strength
        self.preserve_entity_boundaries = preserve_entity_boundaries
        self.detect_topic_shifts = detect_topic_shifts
        self.discourse_marker_sensitivity = discourse_marker_sensitivity

        # Initialize discourse marker patterns
        self._init_discourse_patterns()

        # Initialize topic modeling (optional)
        self._topic_model = None
        if self.detect_topic_shifts:
            self._init_topic_modeling()

        # Initialize entity recognition (optional)
        self._entity_recognizer = None
        if self.preserve_entity_boundaries:
            self._init_entity_recognition()

    def _init_discourse_patterns(self):
        """Initialize comprehensive discourse marker patterns."""
        self.discourse_patterns = {
            DiscourseMarkerType.CONTRAST: [
                # Strong contrast markers
                (r'\bhowever\b', 0.9), (r'\bnevertheless\b', 0.9), (r'\bnonetheless\b', 0.9),
                (r'\bon the other hand\b', 0.9), (r'\bin contrast\b', 0.9), (r'\bconversely\b', 0.8),
                (r'\byet\b', 0.7), (r'\bbut\b', 0.6), (r'\balthough\b', 0.6), (r'\bwhile\b', 0.5),
                (r'\bdespite\b', 0.7), (r'\bin spite of\b', 0.7), (r'\bunlike\b', 0.6)
            ],
            DiscourseMarkerType.CONTINUATION: [
                (r'\bfurthermore\b', 0.8), (r'\bmoreover\b', 0.8), (r'\badditionally\b', 0.8),
                (r'\balso\b', 0.5), (r'\bbesides\b', 0.7), (r'\bin addition\b', 0.8),
                (r'\blikewise\b', 0.7), (r'\bsimilarly\b', 0.7), (r'\bfurther\b', 0.6)
            ],
            DiscourseMarkerType.CAUSATION: [
                (r'\btherefore\b', 0.9), (r'\bconsequently\b', 0.9), (r'\bas a result\b', 0.9),
                (r'\bthus\b', 0.8), (r'\bhence\b', 0.8), (r'\baccordingly\b', 0.8),
                (r'\bbecause of this\b', 0.8), (r'\bfor this reason\b', 0.8), (r'\bso\b', 0.6)
            ],
            DiscourseMarkerType.SEQUENCE: [
                (r'\bfirst\b', 0.8), (r'\bsecond\b', 0.8), (r'\bthird\b', 0.8), (r'\bnext\b', 0.7),
                (r'\bthen\b', 0.7), (r'\bafter\b', 0.6), (r'\bfinally\b', 0.9), (r'\blast\b', 0.8),
                (r'\bin conclusion\b', 0.9), (r'\bto conclude\b', 0.9), (r'\bto sum up\b', 0.9),
                (r'\binitially\b', 0.7), (r'\bsubsequently\b', 0.7)
            ],
            DiscourseMarkerType.EXEMPLIFICATION: [
                (r'\bfor example\b', 0.7), (r'\bfor instance\b', 0.7), (r'\bsuch as\b', 0.6),
                (r'\bnamely\b', 0.7), (r'\bspecifically\b', 0.7), (r'\bin particular\b', 0.7),
                (r'\be\.g\.\b', 0.6), (r'\bi\.e\.\b', 0.6)
            ],
            DiscourseMarkerType.EMPHASIS: [
                (r'\bindeed\b', 0.6), (r'\bin fact\b', 0.7), (r'\bclearly\b', 0.6),
                (r'\bobviously\b', 0.6), (r'\bundoubtedly\b', 0.6), (r'\bcertainly\b', 0.6),
                (r'\bimportantly\b', 0.7), (r'\bnotably\b', 0.7)
            ],
            DiscourseMarkerType.SUMMARY: [
                (r'\bin summary\b', 0.9), (r'\bto summarize\b', 0.9), (r'\boverall\b', 0.8),
                (r'\bin brief\b', 0.8), (r'\bin short\b', 0.8), (r'\bon the whole\b', 0.8),
                (r'\ball in all\b', 0.8), (r'\bgenerally\b', 0.6)
            ],
            DiscourseMarkerType.COMPARISON: [
                (r'\bsimilarly\b', 0.7), (r'\blikewise\b', 0.7), (r'\bin comparison\b', 0.8),
                (r'\bcompared to\b', 0.7), (r'\bin the same way\b', 0.7), (r'\bby comparison\b', 0.8)
            ],
            DiscourseMarkerType.ELABORATION: [
                (r'\bspecifically\b', 0.6), (r'\bin particular\b', 0.6), (r'\bnamely\b', 0.7),
                (r'\bthat is\b', 0.6), (r'\bthat is to say\b', 0.7), (r'\bin other words\b', 0.7)
            ]
        }

    def _init_topic_modeling(self):
        """Initialize topic modeling for detecting semantic shifts."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import LatentDirichletAllocation

            self._topic_model = {
                'vectorizer': TfidfVectorizer(max_features=100, stop_words='english'),
                'lda': LatentDirichletAllocation(n_components=5, random_state=42)
            }
            logger.info("Initialized topic modeling for discourse-aware chunking")

        except ImportError:
            logger.warning("scikit-learn not available, topic modeling disabled")
            self.detect_topic_shifts = False

    def _init_entity_recognition(self):
        """Initialize entity recognition for boundary preservation."""
        try:
            import spacy
            # Try to load a spacy model
            models_to_try = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']

            for model_name in models_to_try:
                try:
                    self._entity_recognizer = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model {model_name} for entity recognition")
                    break
                except OSError:
                    continue

            if not self._entity_recognizer:
                logger.warning("No spaCy model available, entity boundary preservation disabled")
                self.preserve_entity_boundaries = False

        except ImportError:
            logger.warning("spaCy not available, entity boundary preservation disabled")
            self.preserve_entity_boundaries = False

    def _detect_discourse_markers(self, sentences: List[str]) -> List[DiscourseMarker]:
        """Detect discourse markers in sentences."""
        markers = []

        for sent_idx, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()

            for marker_type, patterns in self.discourse_patterns.items():
                for pattern, base_strength in patterns:
                    matches = list(re.finditer(pattern, sentence_lower, re.IGNORECASE))

                    for match in matches:
                        # Adjust strength based on position in sentence
                        position_in_sentence = match.start() / len(sentence)
                        position_weight = 1.2 if position_in_sentence < 0.2 else 1.0  # Stronger if at beginning

                        # Adjust strength based on sentence position
                        sentence_weight = 1.1 if sent_idx == 0 else 1.0  # Slightly stronger if first sentence

                        final_strength = base_strength * position_weight * sentence_weight * self.discourse_marker_sensitivity
                        final_strength = min(1.0, final_strength)  # Cap at 1.0

                        marker = DiscourseMarker(
                            text=match.group(),
                            position=match.start(),
                            marker_type=marker_type,
                            sentence_index=sent_idx,
                            confidence=base_strength,
                            boundary_strength=final_strength
                        )
                        markers.append(marker)

        return markers

    def _detect_enhanced_boundaries(self, sentences: List[str]) -> List[EnhancedSemanticBoundary]:
        """Detect enhanced semantic boundaries with discourse analysis."""
        # Compute embeddings and similarities first (required by parent class)
        embeddings = self._compute_sentence_embeddings(sentences)
        similarities = self._compute_similarity_scores(embeddings)

        # Get basic semantic boundaries from parent class
        basic_boundaries = self._detect_boundaries(sentences, similarities)

        # Detect discourse markers
        discourse_markers = self._detect_discourse_markers(sentences)

        # Group discourse markers by sentence
        markers_by_sentence = {}
        for marker in discourse_markers:
            sent_idx = marker.sentence_index
            if sent_idx not in markers_by_sentence:
                markers_by_sentence[sent_idx] = []
            markers_by_sentence[sent_idx].append(marker)

        enhanced_boundaries = []

        # Create enhanced boundaries that combine multiple factors
        for sent_idx in range(1, len(sentences)):
            # Get basic semantic score
            semantic_score = 0.0
            for boundary in basic_boundaries:
                if boundary.sentence_index == sent_idx:
                    semantic_score = 1.0 - boundary.similarity_score
                    break

            # Calculate discourse score
            discourse_score = 0.0
            sentence_markers = markers_by_sentence.get(sent_idx, [])
            if sentence_markers:
                discourse_score = max(marker.boundary_strength for marker in sentence_markers)

            # Calculate topic shift score (if enabled)
            topic_score = 0.0
            if self.detect_topic_shifts:
                topic_score = self._calculate_topic_shift_score(sentences, sent_idx)

            # Calculate entity preservation score
            entity_score = 0.0
            if self.preserve_entity_boundaries:
                entity_score = self._calculate_entity_preservation_score(sentences, sent_idx)

            # Combine scores with weights
            combined_score = (
                semantic_score * (1.0 - self.discourse_weight - self.topic_weight - self.entity_weight) +
                discourse_score * self.discourse_weight +
                topic_score * self.topic_weight +
                entity_score * self.entity_weight
            )

            # Create boundary if score exceeds threshold
            if combined_score >= self.min_boundary_strength:
                boundary = EnhancedSemanticBoundary(
                    sentence_index=sent_idx,
                    similarity_score=1.0 - semantic_score,
                    coherence_score=combined_score,
                    boundary_strength=combined_score,
                    boundary_type="strong" if combined_score > 0.8 else "moderate",
                    discourse_markers=sentence_markers,
                    topic_shift_score=topic_score,
                    entity_preservation_score=entity_score,
                    overall_coherence_score=combined_score
                )
                enhanced_boundaries.append(boundary)

        return enhanced_boundaries

    def _calculate_topic_shift_score(self, sentences: List[str], sent_idx: int) -> float:
        """Calculate topic shift score between sentences."""
        if not self._topic_model or sent_idx == 0:
            return 0.0

        try:
            # Compare topic distributions of surrounding context
            context_before = ' '.join(sentences[max(0, sent_idx-2):sent_idx])
            context_after = ' '.join(sentences[sent_idx:min(len(sentences), sent_idx+2)])

            if not context_before or not context_after:
                return 0.0

            # Vectorize contexts
            vectorizer = self._topic_model['vectorizer']
            lda = self._topic_model['lda']

            # Simple implementation: compare TF-IDF similarity
            contexts = [context_before, context_after]
            tfidf_matrix = vectorizer.fit_transform(contexts)

            if tfidf_matrix.shape[0] < 2:
                return 0.0

            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            # Convert to topic shift score (lower similarity = higher topic shift)
            topic_shift = 1.0 - similarity
            return max(0.0, min(1.0, topic_shift))

        except Exception as e:
            logger.debug(f"Topic shift calculation failed: {e}")
            return 0.0

    def _calculate_entity_preservation_score(self, sentences: List[str], sent_idx: int) -> float:
        """Calculate entity preservation score to avoid breaking entity boundaries."""
        if not self._entity_recognizer or sent_idx == 0:
            return 0.0

        try:
            # Check if splitting here would break an entity
            current_sentence = sentences[sent_idx-1]
            next_sentence = sentences[sent_idx]

            # Process both sentences to find entities
            current_doc = self._entity_recognizer(current_sentence)
            next_doc = self._entity_recognizer(next_sentence)

            # Check for entities spanning the boundary (simple heuristic)
            current_entities = set(ent.text.lower() for ent in current_doc.ents)
            next_entities = set(ent.text.lower() for ent in next_doc.ents)

            # If there are shared entities, splitting here might be bad
            shared_entities = current_entities.intersection(next_entities)

            if shared_entities:
                # Penalize splits that break entity continuity
                return -0.3  # Negative score discourages boundary
            else:
                # Encourage splits that preserve entity boundaries
                return 0.1

        except Exception as e:
            logger.debug(f"Entity preservation calculation failed: {e}")
            return 0.0

    def chunk(self, content: Union[str, Path], source_info: Optional[Dict[str, Any]] = None, **kwargs) -> ChunkingResult:
        """Enhanced chunking with discourse analysis."""
        start_time = time.time()

        # Get text content
        if isinstance(content, Path):
            text_content = content.read_text(encoding='utf-8')
            source_path = str(content)
            source_type = "file"
        else:
            text_content = content
            source_path = source_info.get("source", "string") if source_info else "string"
            source_type = "string"

        if not text_content.strip():
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used="discourse_aware",
                source_info={
                    "source": source_path,
                    "source_type": source_type,
                    "discourse_markers_detected": 0,
                    "enhanced_boundaries": 0
                }
            )

        # Initialize semantic model if needed (inherited from SemanticChunker)
        if not any([self.embedder, self.nlp, self.vectorizer]):
            self._initialize_semantic_model()

        # Split into sentences
        sentences = self._segment_sentences(text_content)

        if len(sentences) <= 1:
            # Single sentence, return as single chunk
            chunk = Chunk(
                id="chunk_0",
                content=text_content.strip(),
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=source_path,
                    source_type=source_type,
                    position="sentence_0_1",
                    offset=0,
                    extra={
                        "chunk_index": 0,
                        "start_char": 0,
                        "end_char": len(text_content),
                        "sentence_count": 1,
                        "discourse_markers": [],
                        "semantic_score": 1.0,
                        "boundary_type": "none"
                    }
                )
            )

            return ChunkingResult(
                chunks=[chunk],
                processing_time=time.time() - start_time,
                strategy_used="discourse_aware",
                source_info={
                    "source": source_path,
                    "source_type": source_type,
                    "discourse_markers_detected": 0,
                    "enhanced_boundaries": 0
                }
            )

        # Detect enhanced boundaries
        enhanced_boundaries = self._detect_enhanced_boundaries(sentences)

        # Create chunks from enhanced boundaries
        chunks = self._create_chunks_from_enhanced_boundaries(sentences, enhanced_boundaries, source_info or {})

        processing_time = time.time() - start_time

        # Count discourse markers
        total_markers = sum(len(boundary.discourse_markers) for boundary in enhanced_boundaries)

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used="discourse_aware",
            source_info={
                "source": source_path,
                "source_type": source_type,
                "total_sentences": len(sentences),
                "enhanced_boundaries": len(enhanced_boundaries),
                "discourse_markers_detected": total_markers,
                "avg_processing_time": processing_time / len(chunks) if chunks else 0
            }
        )

    def _create_chunks_from_enhanced_boundaries(
        self,
        sentences: List[str],
        boundaries: List[EnhancedSemanticBoundary],
        source_info: Dict[str, Any]
    ) -> List[Chunk]:
        """Create chunks from enhanced semantic boundaries."""
        if not boundaries:
            # No boundaries, return entire text as one chunk
            content = ' '.join(sentences)
            chunk = Chunk(
                id="chunk_0",
                content=content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=source_info.get("source", "unknown"),
                    source_type=source_info.get("source_type", "content"),
                    position=f"sentences_0_{len(sentences)}",
                    offset=0,
                    extra={
                        "chunk_index": 0,
                        "start_char": 0,
                        "end_char": len(content),
                        "sentence_count": len(sentences),
                        "discourse_markers": [],
                        "semantic_score": 1.0,
                        "boundary_type": "none",
                        "coherence_analysis": {
                            "topic_shift_score": 0.0,
                            "entity_preservation_score": 0.0,
                            "overall_coherence": 1.0
                        }
                    }
                )
            )
            return [chunk]

        chunks = []
        start_idx = 0

        # Sort boundaries by sentence index
        boundaries.sort(key=lambda x: x.sentence_index)

        for i, boundary in enumerate(boundaries):
            end_idx = boundary.sentence_index

            # Create chunk from start_idx to end_idx
            if start_idx < end_idx:
                chunk_sentences = sentences[start_idx:end_idx]
                content = ' '.join(chunk_sentences)

                # Calculate chunk statistics
                start_char = sum(len(s) + 1 for s in sentences[:start_idx])
                end_char = start_char + len(content)

                # Collect discourse markers for this chunk
                chunk_markers = []
                for sent_idx in range(start_idx, end_idx):
                    for boundary_check in boundaries:
                        if boundary_check.sentence_index == sent_idx:
                            chunk_markers.extend(boundary_check.discourse_markers)

                chunk = Chunk(
                    id=f"chunk_{len(chunks)}",
                    content=content,
                    modality=ModalityType.TEXT,
                    metadata=ChunkMetadata(
                        source=source_info.get("source", "unknown"),
                        source_type=source_info.get("source_type", "content"),
                        position=f"sentences_{start_idx}_{end_idx}",
                        offset=start_char,
                        extra={
                            "chunk_index": len(chunks),
                            "start_char": start_char,
                            "end_char": end_char,
                            "sentence_count": len(chunk_sentences),
                            "discourse_markers": [
                                {
                                    "text": m.text,
                                    "type": m.marker_type.value,
                                    "confidence": m.confidence,
                                    "boundary_strength": m.boundary_strength
                                }
                                for m in chunk_markers
                            ],
                            "semantic_score": 1.0 - boundary.similarity_score,
                            "boundary_type": boundary.boundary_type,
                            "coherence_analysis": {
                                "topic_shift_score": boundary.topic_shift_score,
                                "entity_preservation_score": boundary.entity_preservation_score,
                                "overall_coherence": boundary.overall_coherence_score
                            }
                        }
                    )
                )
                chunks.append(chunk)

            start_idx = end_idx

        # Handle remaining sentences
        if start_idx < len(sentences):
            chunk_sentences = sentences[start_idx:]
            content = ' '.join(chunk_sentences)

            start_char = sum(len(s) + 1 for s in sentences[:start_idx])

            chunk = Chunk(
                id=f"chunk_{len(chunks)}",
                content=content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=source_info.get("source", "unknown"),
                    source_type=source_info.get("source_type", "content"),
                    position=f"sentences_{start_idx}_{len(sentences)}",
                    offset=start_char,
                    extra={
                        "chunk_index": len(chunks),
                        "start_char": start_char,
                        "end_char": start_char + len(content),
                        "sentence_count": len(chunk_sentences),
                        "discourse_markers": [],
                        "semantic_score": 1.0,
                        "boundary_type": "final",
                        "coherence_analysis": {
                            "topic_shift_score": 0.0,
                            "entity_preservation_score": 0.0,
                            "overall_coherence": 1.0
                        }
                    }
                )
            )
            chunks.append(chunk)

        return chunks
