"""
Balanced Length Chunking Algorithm

This custom algorithm creates chunks that are balanced in length while respecting
word and sentence boundaries. Unlike simple fixed-size chunking, this algorithm
tries to optimize chunk sizes to be as close to a target length as possible
while maintaining readability and coherence.

This algorithm demonstrates:
- Advanced boundary detection and preservation
- Length optimization with multiple constraints
- Adaptive chunk sizing based on content structure
- Quality metrics for chunk balance
- Support for different boundary types (word, sentence, paragraph)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import time
import statistics

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    ModalityType
)
from chunking_strategy.core.registry import (
    register_chunker,
    ComplexityLevel,
    SpeedLevel,
    MemoryUsage
)

# Optional imports
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


@register_chunker(
    name="balanced_length",
    category="text",
    description="Creates balanced-length chunks while preserving word and sentence boundaries",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "json", "html"],
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.MEDIUM,
    memory=MemoryUsage.LOW,
    quality=0.8,  # High quality due to boundary preservation and length optimization
    use_cases=["balanced processing", "parallel processing", "uniform chunks", "readability"],
    best_for=["LLM input", "embedding generation", "search indexing", "display formatting"],
    limitations=["more complex than fixed-size", "requires boundary detection"],
    parameters_schema={
        "target_length": {
            "type": "integer",
            "minimum": 100,
            "maximum": 10000,
            "default": 1000,
            "description": "Target length for each chunk in characters"
        },
        "length_tolerance": {
            "type": "number",
            "minimum": 0.1,
            "maximum": 0.9,
            "default": 0.3,
            "description": "Acceptable deviation from target length (0.0-1.0)"
        },
        "boundary_preference": {
            "type": "string",
            "enum": ["sentence", "word", "paragraph", "line"],
            "default": "sentence",
            "description": "Preferred boundary type for chunk breaks"
        },
        "balance_algorithm": {
            "type": "string",
            "enum": ["greedy", "optimal", "adaptive"],
            "default": "adaptive",
            "description": "Algorithm for length balancing"
        },
        "min_chunk_length": {
            "type": "integer",
            "minimum": 50,
            "default": 200,
            "description": "Minimum acceptable chunk length"
        },
        "max_chunk_length": {
            "type": "integer",
            "minimum": 500,
            "default": 2000,
            "description": "Maximum acceptable chunk length"
        }
    },
    default_parameters={
        "target_length": 1000,
        "length_tolerance": 0.3,
        "boundary_preference": "sentence",
        "balance_algorithm": "adaptive",
        "min_chunk_length": 200,
        "max_chunk_length": 2000
    }
)
class BalancedLengthChunker(BaseChunker):
    """
    Balanced length text chunker.

    This chunker creates chunks that are as close to a target length as possible
    while respecting natural text boundaries. It's particularly useful when you need:

    - Consistent chunk sizes for parallel processing
    - Balanced input for LLMs or embedding models
    - Uniform chunks for search indexing
    - Readable chunks that don't cut off mid-sentence

    The algorithm works by:
    1. Identifying natural boundaries (sentences, words, paragraphs)
    2. Using optimization algorithms to group boundaries into balanced chunks
    3. Applying length constraints and tolerance settings
    4. Computing balance metrics for quality assessment

    Balance Algorithms:
    - greedy: Fast, locally optimal decisions
    - optimal: Slower, globally optimal using dynamic programming
    - adaptive: Switches between strategies based on content
    """

    def __init__(
        self,
        name: str = "balanced_length",
        target_length: int = 1000,
        length_tolerance: float = 0.3,
        boundary_preference: str = "sentence",
        balance_algorithm: str = "adaptive",
        min_chunk_length: int = 200,
        max_chunk_length: int = 2000,
        **kwargs
    ):
        """
        Initialize the balanced length chunker.

        Args:
            target_length: Target length for each chunk
            length_tolerance: Acceptable deviation from target (0.0-1.0)
            boundary_preference: Preferred boundary type
            balance_algorithm: Balancing algorithm to use
            min_chunk_length: Minimum chunk length
            max_chunk_length: Maximum chunk length
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        self.target_length = target_length
        self.length_tolerance = length_tolerance
        self.boundary_preference = boundary_preference
        self.balance_algorithm = balance_algorithm
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length

        # Computed tolerance bounds
        self.min_target = int(target_length * (1 - length_tolerance))
        self.max_target = int(target_length * (1 + length_tolerance))

        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, using basic boundary detection")

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Create balanced-length chunks.

        Args:
            content: Input content to chunk
            source_info: Optional source information
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with balanced chunks
        """
        start_time = time.time()

        # Handle different input types
        if isinstance(content, Path):
            with open(content, 'r', encoding='utf-8') as f:
                text_content = f.read()
            source_name = str(content)
        elif isinstance(content, bytes):
            text_content = content.decode('utf-8')
            source_name = source_info.get('source', 'unknown') if source_info else 'unknown'
        else:
            text_content = str(content)
            source_name = source_info.get('source', 'unknown') if source_info else 'unknown'

        # Validate input
        self.validate_input(text_content, ModalityType.TEXT)

        # Create balanced chunks
        chunks = self._create_balanced_chunks(text_content, source_name)

        processing_time = time.time() - start_time

        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(chunks)

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used="balanced_length",
            source_info=source_info,
            quality_score=quality_metrics.get("balance_score", 0.5)
        )

    def _create_balanced_chunks(self, content: str, source_name: str) -> List[Chunk]:
        """Create balanced chunks using the specified algorithm."""

        # Step 1: Detect boundaries
        boundaries = self._detect_boundaries(content)
        if not boundaries:
            # Fallback to single chunk
            return [self._create_chunk(content, source_name, 0, 0, len(content))]

        # Step 2: Apply balancing algorithm
        if self.balance_algorithm == "greedy":
            chunk_groups = self._greedy_balance(boundaries)
        elif self.balance_algorithm == "optimal":
            chunk_groups = self._optimal_balance(boundaries)
        else:  # adaptive
            chunk_groups = self._adaptive_balance(boundaries)

        # Step 3: Create chunk objects
        chunks = []
        for i, group in enumerate(chunk_groups):
            if group:
                start_pos = group[0]["start"]
                end_pos = group[-1]["end"]
                chunk_content = content[start_pos:end_pos]

                chunk = self._create_chunk(
                    chunk_content, source_name, i, start_pos, end_pos, group
                )
                chunks.append(chunk)

        logger.info(f"Generated {len(chunks)} balanced chunks from {len(boundaries)} boundaries")
        return chunks

    def _detect_boundaries(self, content: str) -> List[Dict[str, Any]]:
        """Detect natural boundaries in the text."""
        boundaries = []

        if self.boundary_preference == "sentence":
            boundaries = self._detect_sentence_boundaries(content)
        elif self.boundary_preference == "paragraph":
            boundaries = self._detect_paragraph_boundaries(content)
        elif self.boundary_preference == "line":
            boundaries = self._detect_line_boundaries(content)
        else:  # word
            boundaries = self._detect_word_boundaries(content)

        return boundaries

    def _detect_sentence_boundaries(self, content: str) -> List[Dict[str, Any]]:
        """Detect sentence boundaries."""
        boundaries = []

        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(content)
                current_pos = 0

                for sentence in sentences:
                    # Find sentence in content (handle potential spacing issues)
                    sentence_start = content.find(sentence, current_pos)
                    if sentence_start == -1:
                        # Fallback: estimate position
                        sentence_start = current_pos

                    sentence_end = sentence_start + len(sentence)

                    boundaries.append({
                        "type": "sentence",
                        "start": sentence_start,
                        "end": sentence_end,
                        "content": sentence,
                        "length": len(sentence)
                    })

                    current_pos = sentence_end

            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}")
                return self._fallback_sentence_detection(content)
        else:
            return self._fallback_sentence_detection(content)

        return boundaries

    def _fallback_sentence_detection(self, content: str) -> List[Dict[str, Any]]:
        """Fallback sentence detection using regex."""
        import re

        boundaries = []
        sentences = re.split(r'[.!?]+', content)
        current_pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find sentence position in content
            sentence_start = content.find(sentence, current_pos)
            if sentence_start == -1:
                sentence_start = current_pos

            sentence_end = sentence_start + len(sentence)

            boundaries.append({
                "type": "sentence",
                "start": sentence_start,
                "end": sentence_end,
                "content": sentence,
                "length": len(sentence)
            })

            current_pos = sentence_end

        return boundaries

    def _detect_paragraph_boundaries(self, content: str) -> List[Dict[str, Any]]:
        """Detect paragraph boundaries."""
        boundaries = []
        paragraphs = content.split('\n\n')
        current_pos = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            paragraph_start = content.find(paragraph, current_pos)
            if paragraph_start == -1:
                paragraph_start = current_pos

            paragraph_end = paragraph_start + len(paragraph)

            boundaries.append({
                "type": "paragraph",
                "start": paragraph_start,
                "end": paragraph_end,
                "content": paragraph,
                "length": len(paragraph)
            })

            current_pos = paragraph_end

        return boundaries

    def _detect_line_boundaries(self, content: str) -> List[Dict[str, Any]]:
        """Detect line boundaries."""
        boundaries = []
        lines = content.split('\n')
        current_pos = 0

        for line in lines:
            if line.strip():  # Skip empty lines
                line_start = current_pos
                line_end = current_pos + len(line)

                boundaries.append({
                    "type": "line",
                    "start": line_start,
                    "end": line_end,
                    "content": line,
                    "length": len(line)
                })

            current_pos += len(line) + 1  # +1 for newline

        return boundaries

    def _detect_word_boundaries(self, content: str) -> List[Dict[str, Any]]:
        """Detect word boundaries."""
        boundaries = []

        if NLTK_AVAILABLE:
            try:
                words = word_tokenize(content)
                current_pos = 0

                for word in words:
                    word_start = content.find(word, current_pos)
                    if word_start == -1:
                        word_start = current_pos

                    word_end = word_start + len(word)

                    boundaries.append({
                        "type": "word",
                        "start": word_start,
                        "end": word_end,
                        "content": word,
                        "length": len(word)
                    })

                    current_pos = word_end

            except Exception:
                return self._fallback_word_detection(content)
        else:
            return self._fallback_word_detection(content)

        return boundaries

    def _fallback_word_detection(self, content: str) -> List[Dict[str, Any]]:
        """Fallback word detection using simple splitting."""
        boundaries = []
        words = content.split()
        current_pos = 0

        for word in words:
            word_start = content.find(word, current_pos)
            if word_start == -1:
                word_start = current_pos

            word_end = word_start + len(word)

            boundaries.append({
                "type": "word",
                "start": word_start,
                "end": word_end,
                "content": word,
                "length": len(word)
            })

            current_pos = word_end

        return boundaries

    def _greedy_balance(self, boundaries: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Greedy algorithm for balancing chunk lengths."""
        chunks = []
        current_chunk = []
        current_length = 0

        for boundary in boundaries:
            boundary_length = boundary["length"]

            # Check if adding this boundary would exceed max length
            if current_length + boundary_length > self.max_chunk_length and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [boundary]
                current_length = boundary_length
            else:
                current_chunk.append(boundary)
                current_length += boundary_length

                # Check if we've reached a good target length
                if current_length >= self.min_target:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_length = 0

        # Handle remaining boundaries
        if current_chunk:
            if chunks and len(current_chunk) == 1 and current_length < self.min_chunk_length:
                # Merge small final chunk with previous
                chunks[-1].extend(current_chunk)
            else:
                chunks.append(current_chunk)

        return chunks

    def _optimal_balance(self, boundaries: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Optimal dynamic programming algorithm for chunk balancing."""
        if not boundaries:
            return []

        n = len(boundaries)
        # For simplicity, use greedy as fallback for large inputs
        if n > 1000:
            logger.info("Using greedy fallback for large input in optimal balance")
            return self._greedy_balance(boundaries)

        # Precompute cumulative lengths
        cum_lengths = [0]
        for boundary in boundaries:
            cum_lengths.append(cum_lengths[-1] + boundary["length"])

        # Dynamic programming to find optimal split points
        dp = [float('inf')] * (n + 1)
        splits = [-1] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i):
                chunk_length = cum_lengths[i] - cum_lengths[j]

                # Check constraints
                if (chunk_length >= self.min_chunk_length and
                    chunk_length <= self.max_chunk_length):

                    # Cost function: deviation from target length
                    cost = abs(chunk_length - self.target_length)

                    if dp[j] + cost < dp[i]:
                        dp[i] = dp[j] + cost
                        splits[i] = j

        # Reconstruct solution
        chunks = []
        i = n
        while i > 0:
            j = splits[i]
            if j >= 0:
                chunk_boundaries = boundaries[j:i]
                chunks.append(chunk_boundaries)
                i = j
            else:
                break

        return list(reversed(chunks))

    def _adaptive_balance(self, boundaries: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Adaptive algorithm that chooses strategy based on input characteristics."""

        # Analyze input characteristics
        total_length = sum(b["length"] for b in boundaries)
        avg_boundary_length = total_length / len(boundaries) if boundaries else 0

        # Use optimal for smaller inputs or when boundaries are very uneven
        if len(boundaries) < 500:
            return self._optimal_balance(boundaries)

        # Use greedy for large inputs or when boundaries are relatively uniform
        else:
            return self._greedy_balance(boundaries)

    def _create_chunk(
        self,
        content: str,
        source_name: str,
        chunk_index: int,
        start_pos: int,
        end_pos: int,
        boundaries: Optional[List[Dict[str, Any]]] = None
    ) -> Chunk:
        """Create a chunk with balance-specific metadata."""

        # Calculate balance metrics
        length = len(content)
        target_deviation = abs(length - self.target_length)
        balance_score = 1.0 - (target_deviation / max(self.target_length, length))

        # Determine if chunk meets constraints
        meets_constraints = (self.min_chunk_length <= length <= self.max_chunk_length)
        in_tolerance = (self.min_target <= length <= self.max_target)

        extra_metadata = {
            "chunk_length": length,
            "target_length": self.target_length,
            "target_deviation": target_deviation,
            "balance_score": balance_score,
            "meets_constraints": meets_constraints,
            "in_tolerance": in_tolerance,
            "boundary_count": len(boundaries) if boundaries else 0,
            "boundary_preference": self.boundary_preference,
            "balance_algorithm": self.balance_algorithm
        }

        # Add boundary information
        if boundaries:
            boundary_info = {
                "boundary_types": [b["type"] for b in boundaries],
                "boundary_lengths": [b["length"] for b in boundaries],
                "first_boundary": boundaries[0]["content"][:50] if boundaries[0]["content"] else "",
                "last_boundary": boundaries[-1]["content"][:50] if boundaries[-1]["content"] else ""
            }
            extra_metadata["boundary_info"] = boundary_info

        metadata = ChunkMetadata(
            source=source_name,
            position=f"chars_{start_pos}_{end_pos}",
            offset=start_pos,
            length=length,
            chunker_used="balanced_length",
            quality_score=balance_score,
            extra=extra_metadata
        )

        return Chunk(
            id=f"balanced_{chunk_index:04d}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=metadata
        )

    def _compute_quality_metrics(self, chunks: List[Chunk]) -> Dict[str, float]:
        """Compute overall quality metrics for the chunking result."""
        if not chunks:
            return {"balance_score": 0.0}

        lengths = [len(chunk.content) for chunk in chunks]

        # Length statistics
        mean_length = statistics.mean(lengths)
        median_length = statistics.median(lengths)

        if len(lengths) > 1:
            stdev_length = statistics.stdev(lengths)
            coefficient_variation = stdev_length / mean_length if mean_length > 0 else 0
        else:
            stdev_length = 0
            coefficient_variation = 0

        # Balance score (lower variation = higher score)
        balance_score = max(0.0, 1.0 - coefficient_variation)

        # Target adherence
        target_deviations = [abs(length - self.target_length) for length in lengths]
        avg_target_deviation = statistics.mean(target_deviations)
        target_adherence = max(0.0, 1.0 - (avg_target_deviation / self.target_length))

        # Constraint compliance
        constraint_violations = sum(1 for length in lengths
                                  if not (self.min_chunk_length <= length <= self.max_chunk_length))
        constraint_compliance = 1.0 - (constraint_violations / len(lengths))

        return {
            "balance_score": balance_score,
            "target_adherence": target_adherence,
            "constraint_compliance": constraint_compliance,
            "mean_length": mean_length,
            "median_length": median_length,
            "stdev_length": stdev_length,
            "coefficient_variation": coefficient_variation,
            "avg_target_deviation": avg_target_deviation
        }


if __name__ == "__main__":
    # Demo usage
    chunker = BalancedLengthChunker(
        target_length=800,          # Target 800 characters per chunk
        length_tolerance=0.25,      # Allow 25% deviation
        boundary_preference="sentence",  # Respect sentence boundaries
        balance_algorithm="optimal" # Use optimal balancing
    )

    # Example text
    text = '''Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario. It involves algorithms that can identify patterns in data and use those patterns to make predictions or decisions about new, unseen data.

There are three main types of machine learning: supervised learning, where the algorithm learns from labeled examples; unsupervised learning, where it finds hidden patterns in data without labels; and reinforcement learning, where it learns through interaction with an environment and receives rewards or penalties.

The applications of machine learning are vast and growing. From recommendation systems on streaming platforms to autonomous vehicles, from medical diagnosis to financial fraud detection, machine learning is transforming industries and creating new possibilities for innovation and efficiency.

Deep learning, a subset of machine learning, uses artificial neural networks with multiple layers to model and understand complex patterns in data. This approach has revolutionized fields such as computer vision, natural language processing, and speech recognition, enabling breakthroughs like image classification, language translation, and voice assistants.'''

    result = chunker.chunk(text)

    print("Balanced Length Chunking Demo:")
    print(f"Generated {len(result.chunks)} chunks")
    print(f"Quality score: {result.quality_score:.3f}")

    # Print chunk information
    for i, chunk in enumerate(result.chunks):
        meta = chunk.metadata.extra
        print(f"\nChunk {i+1}:")
        print(f"  Length: {meta['chunk_length']} chars (target: {meta['target_length']})")
        print(f"  Balance score: {meta['balance_score']:.3f}")
        print(f"  Boundaries: {meta['boundary_count']} {meta['boundary_preference']}s")
        print(f"  Content: {chunk.content[:100]}...")
