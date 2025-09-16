"""
Token-based chunking strategy.

This module implements advanced token-based chunking that integrates with multiple
tokenization systems including tiktoken (OpenAI), transformers (HuggingFace),
sentence-transformers, spacy, and nltk. It provides precise token-level control
for chunking, essential for LLM applications, embedding models, and token-aware processing.

Key features:
- Multiple tokenizer support (tiktoken, transformers, spacy, nltk)
- Configurable token limits per chunk with overlap
- Model-specific tokenizers (GPT, BERT, T5, etc.)
- Subword and BPE tokenization handling
- Token-preserving chunk boundaries
- Integration with embedding models
- Streaming support for large documents
- Hardware optimization integration
"""

import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
from enum import Enum

# Suppress tokenizer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

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

logger = logging.getLogger(__name__)


class TokenizerType(str, Enum):
    """Supported tokenizer types."""
    TIKTOKEN = "tiktoken"          # OpenAI tokenizers (GPT models)
    TRANSFORMERS = "transformers"  # HuggingFace tokenizers
    SPACY = "spacy"               # SpaCy tokenizers
    NLTK = "nltk"                 # NLTK tokenizers
    ANTHROPIC = "anthropic"       # Anthropic/Claude tokenizers
    SIMPLE = "simple"             # Simple whitespace tokenization


class TokenizerModel(str, Enum):
    """Pre-configured tokenizer models."""
    # OpenAI models (tiktoken)
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO = "gpt-4-turbo"
    TEXT_DAVINCI_003 = "text-davinci-003"

    # HuggingFace models (transformers)
    BERT_BASE_UNCASED = "bert-base-uncased"
    BERT_LARGE_UNCASED = "bert-large-uncased"
    GPT2 = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    ROBERTA_BASE = "roberta-base"
    DISTILBERT_BASE = "distilbert-base-uncased"
    T5_BASE = "t5-base"
    T5_LARGE = "t5-large"

    # Sentence transformers
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"

    # Anthropic models
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"

    # Language models
    SPACY_EN_CORE_WEB_SM = "en_core_web_sm"
    SPACY_EN_CORE_WEB_MD = "en_core_web_md"
    SPACY_EN_CORE_WEB_LG = "en_core_web_lg"


@register_chunker(
    name="token_based",
    category="text",
    description="Advanced token-based chunking with multiple tokenizer support for LLM and embedding applications",
    supported_modalities=[ModalityType.TEXT],
    supported_formats=["txt", "md", "html", "json", "xml", "csv"],
    complexity=ComplexityLevel.MEDIUM,
    dependencies=["tiktoken"],
    optional_dependencies=["transformers", "sentence-transformers", "spacy", "nltk", "torch", "anthropic"],
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    quality=0.9,  # High quality due to precise token control
    parameters_schema={
        "tokens_per_chunk": {
            "type": "integer",
            "minimum": 1,
            "maximum": 32000,
            "default": 1000,
            "description": "Maximum number of tokens per chunk"
        },
        "overlap_tokens": {
            "type": "integer",
            "minimum": 0,
            "maximum": 1000,
            "default": 50,
            "description": "Number of tokens to overlap between chunks"
        },
        "tokenizer_type": {
            "type": "string",
            "enum": ["tiktoken", "transformers", "spacy", "nltk", "anthropic", "simple"],
            "default": "tiktoken",
            "description": "Tokenization system to use"
        },
        "tokenizer_model": {
            "type": "string",
            "default": "gpt-3.5-turbo",
            "description": "Specific tokenizer model to use"
        },
        "preserve_word_boundaries": {
            "type": "boolean",
            "default": True,
            "description": "Attempt to preserve word boundaries when splitting"
        },
        "min_chunk_tokens": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000,
            "default": 10,
            "description": "Minimum tokens per chunk (for last chunk)"
        },
        "max_chunk_chars": {
            "type": "integer",
            "minimum": 100,
            "maximum": 100000,
            "default": 8000,
            "description": "Maximum chunk size in characters (safety limit)"
        }
    },
    default_parameters={
        "tokens_per_chunk": 1000,
        "overlap_tokens": 50,
        "tokenizer_type": "tiktoken",
        "tokenizer_model": "gpt-3.5-turbo",
        "preserve_word_boundaries": True,
        "min_chunk_tokens": 10,
        "max_chunk_chars": 8000
    },
    use_cases=["LLM_input", "embedding_generation", "RAG", "token_aware_processing", "API_optimization"],
    best_for=["GPT models", "BERT models", "token-limited APIs", "precise chunking", "embedding models"],
    limitations=["requires tokenizer dependencies", "model-specific", "may be slower than word-based"],
    streaming_support=True,
    adaptive_support=True,
    hierarchical_support=False
)
class TokenBasedChunker(StreamableChunker, AdaptableChunker):
    """
    Advanced token-based chunker with multiple tokenizer support.

    This chunker provides precise token-level control for chunking text, essential
    for LLM applications, embedding models, and token-aware processing. It supports
    multiple tokenization systems and handles different tokenization schemes.

    Key features:
    - Multiple tokenizer backends (tiktoken, transformers, spacy, nltk)
    - Model-specific tokenizers for accurate token counting
    - Configurable token limits and overlap
    - Word boundary preservation options
    - Integration with LLM and embedding workflows
    - Streaming support for large documents
    - Adaptive parameter tuning

    Examples:
        # GPT model tokenization
        chunker = TokenBasedChunker(
            tokens_per_chunk=1000,
            overlap_tokens=50,
            tokenizer_type="tiktoken",
            tokenizer_model="gpt-3.5-turbo"
        )

        # BERT model tokenization
        chunker = TokenBasedChunker(
            tokens_per_chunk=512,
            overlap_tokens=25,
            tokenizer_type="transformers",
            tokenizer_model="bert-base-uncased"
        )

        # Sentence transformer tokenization
        chunker = TokenBasedChunker(
            tokens_per_chunk=256,
            tokenizer_type="transformers",
            tokenizer_model="all-MiniLM-L6-v2"
        )
    """

    def __init__(
        self,
        tokens_per_chunk: int = 1000,
        overlap_tokens: int = 50,
        tokenizer_type: str = "tiktoken",
        tokenizer_model: str = "gpt-3.5-turbo",
        preserve_word_boundaries: bool = True,
        min_chunk_tokens: int = 10,
        max_chunk_chars: int = 8000,
        **kwargs
    ):
        super().__init__(
            name="token_based",
            category="text",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )

        # Core parameters
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.tokenizer_type = TokenizerType(tokenizer_type)
        self.tokenizer_model = tokenizer_model
        self.preserve_word_boundaries = preserve_word_boundaries
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_chars = max_chunk_chars

        # Validate parameters
        if self.overlap_tokens >= self.tokens_per_chunk:
            raise ValueError("overlap_tokens must be less than tokens_per_chunk")
        if self.tokens_per_chunk <= 0:
            raise ValueError("tokens_per_chunk must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if self.min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be positive")

        # Initialize tokenizer
        self.tokenizer = None
        self.encode_func = None
        self.decode_func = None
        self._tokenizer_info = {}

        # Performance tracking
        self._total_tokens_processed = 0
        self._tokenization_time = 0.0

        # Adaptation tracking
        self._adaptation_history = []

        logger.info(
            f"Initialized TokenBasedChunker: {tokens_per_chunk} tokens/chunk, "
            f"{overlap_tokens} overlap, {tokenizer_type}:{tokenizer_model}"
        )

    def _initialize_tokenizer(self):
        """Initialize the appropriate tokenizer based on configuration."""
        if self.tokenizer is not None:
            return  # Already initialized

        try:
            if self.tokenizer_type == TokenizerType.TIKTOKEN:
                self._initialize_tiktoken()
            elif self.tokenizer_type == TokenizerType.TRANSFORMERS:
                self._initialize_transformers()
            elif self.tokenizer_type == TokenizerType.SPACY:
                self._initialize_spacy()
            elif self.tokenizer_type == TokenizerType.NLTK:
                self._initialize_nltk()
            elif self.tokenizer_type == TokenizerType.ANTHROPIC:
                self._initialize_anthropic()
            elif self.tokenizer_type == TokenizerType.SIMPLE:
                self._initialize_simple()
            else:
                raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

            logger.info(f"Using {self.tokenizer_type} tokenizer for token-based chunking")

        except Exception as e:
            logger.warning(f"Failed to initialize {self.tokenizer_type} tokenizer: {e}")
            # Fallback to simple tokenization
            logger.info("Falling back to simple tokenization")
            self.tokenizer_type = TokenizerType.SIMPLE
            self._initialize_simple()

    def _initialize_tiktoken(self):
        """Initialize tiktoken for OpenAI models."""
        try:
            import tiktoken

            # Map model names to tiktoken encodings
            model_encodings = {
                "gpt-4": "cl100k_base",
                "gpt-4-turbo": "cl100k_base",
                "gpt-3.5-turbo": "cl100k_base",
                "text-davinci-003": "p50k_base",
                "text-davinci-002": "p50k_base",
                "code-davinci-002": "p50k_base"
            }

            # Get encoding for the model
            if self.tokenizer_model in model_encodings:
                encoding_name = model_encodings[self.tokenizer_model]
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            else:
                # Try to get encoding for the model directly
                try:
                    self.tokenizer = tiktoken.encoding_for_model(self.tokenizer_model)
                except KeyError:
                    logger.warning(f"Unknown model {self.tokenizer_model}, using cl100k_base")
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")

            self.encode_func = self.tokenizer.encode
            self.decode_func = self.tokenizer.decode

            self._tokenizer_info = {
                "type": "tiktoken",
                "model": self.tokenizer_model,
                "encoding": getattr(self.tokenizer, "name", "unknown")
            }

        except ImportError:
            raise ImportError(
                "tiktoken is required for OpenAI tokenization. "
                "Install with: pip install tiktoken or pip install 'chunking-strategy[text]'"
            )

    def _initialize_transformers(self):
        """Initialize HuggingFace transformers tokenizer."""
        try:
            from transformers import AutoTokenizer
            import warnings

            # Suppress tokenizer warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Load tokenizer with error handling
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.tokenizer_model,
                        trust_remote_code=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to load {self.tokenizer_model}: {e}")
                    # Fallback to a reliable tokenizer
                    logger.info("Falling back to bert-base-uncased tokenizer")
                    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            # Define encode/decode functions
            self.encode_func = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
            self.decode_func = lambda tokens: self.tokenizer.decode(tokens, skip_special_tokens=True)

            self._tokenizer_info = {
                "type": "transformers",
                "model": self.tokenizer_model,
                "vocab_size": getattr(self.tokenizer, "vocab_size", "unknown"),
                "model_max_length": getattr(self.tokenizer, "model_max_length", "unknown")
            }

        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFace tokenization. "
                "Install with: pip install transformers or pip install 'chunking-strategy[ml]'"
            )

    def _initialize_spacy(self):
        """Initialize spaCy tokenizer."""
        try:
            import spacy

            # Load spaCy model
            try:
                if self.tokenizer_model.startswith("en_core_web"):
                    self.tokenizer = spacy.load(self.tokenizer_model)
                else:
                    # Default to small English model
                    self.tokenizer = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(f"spaCy model {self.tokenizer_model} not found, trying en_core_web_sm")
                try:
                    self.tokenizer = spacy.load("en_core_web_sm")
                except OSError:
                    raise ImportError(
                        "spaCy English model not found. Install with: python -m spacy download en_core_web_sm"
                    )

            # Define encode/decode functions
            def spacy_encode(text: str) -> List[int]:
                """Encode text using spaCy tokenizer (returns word indices)."""
                doc = self.tokenizer(text)
                # Use token text hash as pseudo-token-id for consistency
                return [hash(token.text) % 50000 for token in doc]

            def spacy_decode(tokens: List[int]) -> str:
                """Decode is not directly supported for spaCy, return placeholder."""
                return f"[{len(tokens)} spacy tokens]"

            self.encode_func = spacy_encode
            self.decode_func = spacy_decode

            self._tokenizer_info = {
                "type": "spacy",
                "model": self.tokenizer.meta.get("name", self.tokenizer_model),
                "version": self.tokenizer.meta.get("version", "unknown"),
                "lang": self.tokenizer.meta.get("lang", "en")
            }

        except ImportError:
            raise ImportError(
                "spacy is required for spaCy tokenization. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )

    def _initialize_nltk(self):
        """Initialize NLTK tokenizer."""
        try:
            import nltk
            from nltk.tokenize import word_tokenize

            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)

            # Define encode/decode functions
            def nltk_encode(text: str) -> List[int]:
                """Encode text using NLTK tokenizer (returns word indices)."""
                tokens = word_tokenize(text)
                # Use token text hash as pseudo-token-id for consistency
                return [hash(token) % 50000 for token in tokens]

            def nltk_decode(tokens: List[int]) -> str:
                """Decode is not directly supported for NLTK, return placeholder."""
                return f"[{len(tokens)} nltk tokens]"

            self.encode_func = nltk_encode
            self.decode_func = nltk_decode

            self._tokenizer_info = {
                "type": "nltk",
                "tokenizer": "word_tokenize",
                "version": nltk.__version__
            }

        except ImportError:
            raise ImportError(
                "nltk is required for NLTK tokenization. "
                "Install with: pip install nltk"
            )

    def _initialize_anthropic(self):
        """Initialize Anthropic/Claude tokenizer."""
        try:
            import anthropic

            # Claude models use a specific tokenizer that's accessible via the SDK
            # For now, we'll create a wrapper that uses the Anthropic client's count_tokens method

            # Map of supported Claude models
            claude_models = {
                "claude-3-haiku-20240307": "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229": "claude-3-sonnet-20240229",
                "claude-3-opus-20240229": "claude-3-opus-20240229",
                "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022"
            }

            # Validate model
            if self.tokenizer_model not in claude_models:
                logger.warning(f"Unknown Claude model {self.tokenizer_model}, using claude-3-sonnet-20240229")
                self.tokenizer_model = "claude-3-sonnet-20240229"

            # Create a pseudo-client for token counting (doesn't need real API key for counting)
            # We'll use tiktoken as a reasonable approximation since Claude's tokenizer isn't publicly available
            try:
                import tiktoken
                # Use cl100k_base as approximation for Claude tokenization
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.warning("Using tiktoken cl100k_base as approximation for Claude tokenization")
            except ImportError:
                # Fallback to simple tokenization
                def simple_encode(text: str) -> List[int]:
                    """Simple word-based tokenization as Claude fallback."""
                    words = text.split()
                    return [hash(word) % 50000 for word in words]

                def simple_decode(tokens: List[int]) -> str:
                    """Simple decode for Claude fallback."""
                    return f"[{len(tokens)} claude-approx tokens]"

                self.encode_func = simple_encode
                self.decode_func = simple_decode
                logger.warning("Using simple tokenization as Claude approximation (install tiktoken for better approximation)")

                self._tokenizer_info = {
                    "type": "anthropic",
                    "model": self.tokenizer_model,
                    "approximation": "simple_tokenization"
                }
                return

            self.encode_func = self.tokenizer.encode
            self.decode_func = self.tokenizer.decode

            self._tokenizer_info = {
                "type": "anthropic",
                "model": self.tokenizer_model,
                "approximation": "tiktoken_cl100k_base",
                "note": "Claude tokenizer not publicly available, using tiktoken approximation"
            }

        except ImportError:
            raise ImportError(
                "anthropic is required for Claude tokenization. "
                "Install with: pip install anthropic"
            )

    def _initialize_simple(self):
        """Initialize simple whitespace tokenizer."""
        def simple_encode(text: str) -> List[int]:
            """Simple tokenization by splitting on whitespace."""
            tokens = text.split()
            # Use token text hash as pseudo-token-id for consistency
            return [hash(token) % 50000 for token in tokens]

        def simple_decode(tokens: List[int]) -> str:
            """Decode is not directly supported for simple tokenizer."""
            return f"[{len(tokens)} simple tokens]"

        self.encode_func = simple_encode
        self.decode_func = simple_decode

        self._tokenizer_info = {
            "type": "simple",
            "method": "whitespace_split"
        }

    def chunk(self, content: Union[str, Path], source_info: Optional[Dict[str, Any]] = None, **kwargs) -> ChunkingResult:
        """
        Chunk content using token-based approach.

        Args:
            content: Text content to chunk or path to file
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with token-based chunks
        """
        start_time = time.time()

        # Initialize tokenizer if needed
        self._initialize_tokenizer()

        # Handle file input and source_info
        if isinstance(content, Path):
            with open(content, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
            source_type = "file"
            source_path = str(content)
        else:
            text_content = str(content)
            source_type = "content"
            source_path = None

        # Merge provided source_info
        if source_info:
            source_type = source_info.get("source_type", source_type)
            source_path = source_info.get("source", source_path)

        if not text_content.strip():
            return ChunkingResult(
                chunks=[],
                processing_time=time.time() - start_time,
                strategy_used="token_based",
                source_info={
                    "source": source_path or "string",
                    "source_type": source_type,
                    "total_tokens": 0,
                    "tokenizer_info": self._tokenizer_info
                }
            )

        # Tokenize the entire text
        tokenize_start = time.time()
        try:
            tokens = self.encode_func(text_content)
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            # Fallback to simple tokenization
            tokens = text_content.split()
            tokens = [hash(token) % 50000 for token in tokens]

        tokenize_time = time.time() - tokenize_start
        self._tokenization_time += tokenize_time
        self._total_tokens_processed += len(tokens)

        logger.debug(f"Tokenized {len(tokens)} tokens in {tokenize_time:.3f}s")

        # Create chunks based on token boundaries
        chunks = self._create_token_chunks(text_content, tokens)

        processing_time = time.time() - start_time

        # Calculate statistics
        total_chunk_tokens = sum(chunk.metadata.extra.get("token_count", 0) for chunk in chunks)
        avg_tokens_per_chunk = total_chunk_tokens / len(chunks) if chunks else 0

        return ChunkingResult(
            chunks=chunks,
            processing_time=processing_time,
            strategy_used="token_based",
            source_info={
                "source": source_path or "string",
                "source_type": source_type,
                "total_tokens": len(tokens),
                "tokenizer_info": self._tokenizer_info,
                "tokenization_time": tokenize_time,
                "avg_tokens_per_chunk": avg_tokens_per_chunk,
                "tokens_per_chunk_config": self.tokens_per_chunk,
                "overlap_tokens_config": self.overlap_tokens
            },
            avg_chunk_size=sum(len(chunk.content) for chunk in chunks) / len(chunks) if chunks else 0,
        )

    def _create_token_chunks(self, text: str, tokens: List[int]) -> List[Chunk]:
        """Create chunks based on token boundaries."""
        if not tokens:
            return []

        chunks = []
        total_tokens = len(tokens)

        # Calculate step size (tokens between chunk starts)
        step_size = max(1, self.tokens_per_chunk - self.overlap_tokens)

        # Split text into words for boundary preservation
        words = text.split() if self.preserve_word_boundaries else None

        for start_idx in range(0, total_tokens, step_size):
            end_idx = min(start_idx + self.tokens_per_chunk, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]

            # Skip chunks that are too small (except the last chunk)
            if len(chunk_tokens) < self.min_chunk_tokens and end_idx < total_tokens:
                continue

            # Get chunk text content
            if self.preserve_word_boundaries and words:
                chunk_text = self._get_text_from_token_range(
                    text, words, start_idx, end_idx, total_tokens, len(words)
                )
            else:
                # For non-boundary-preserving or when word split doesn't work
                try:
                    if hasattr(self.tokenizer, 'decode') or self.decode_func:
                        chunk_text = self.decode_func(chunk_tokens)
                        if chunk_text.startswith("[") and "tokens]" in chunk_text:
                            # Fallback for tokenizers without proper decode
                            chunk_text = self._approximate_text_from_tokens(text, start_idx, end_idx, total_tokens)
                    else:
                        chunk_text = self._approximate_text_from_tokens(text, start_idx, end_idx, total_tokens)
                except:
                    chunk_text = self._approximate_text_from_tokens(text, start_idx, end_idx, total_tokens)

            # Ensure chunk doesn't exceed character limit
            if len(chunk_text) > self.max_chunk_chars:
                chunk_text = chunk_text[:self.max_chunk_chars].rsplit(' ', 1)[0]

            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                source="string",
                source_type="content",
                position=f"tokens {start_idx}-{end_idx-1}",
                length=len(chunk_text),
                extra={
                    "token_count": len(chunk_tokens),
                    "start_token_index": start_idx,
                    "end_token_index": end_idx - 1,
                    "chunk_index": len(chunks),
                    "chunking_strategy": "token_based",
                    "tokenizer_type": self.tokenizer_type.value,
                    "tokenizer_model": self.tokenizer_model,
                    "overlap_tokens": min(self.overlap_tokens, start_idx) if start_idx > 0 else 0,
                    "preserve_word_boundaries": self.preserve_word_boundaries
                }
            )

            chunk_id = f"token_{len(chunks)}"
            chunk = Chunk(
                id=chunk_id,
                content=chunk_text,
                modality=ModalityType.TEXT,
                metadata=chunk_metadata,
                size=len(chunk_text)
            )

            chunks.append(chunk)

            # Break if we've reached the end
            if end_idx >= total_tokens:
                break

        logger.debug(f"Created {len(chunks)} token-based chunks")
        return chunks

    def _get_text_from_token_range(
        self,
        text: str,
        words: List[str],
        start_token_idx: int,
        end_token_idx: int,
        total_tokens: int,
        total_words: int
    ) -> str:
        """Get text content for a token range while preserving word boundaries."""
        try:
            # Map token indices to approximate word indices
            # This is an approximation since tokens and words don't map 1:1
            start_word_idx = max(0, int((start_token_idx / total_tokens) * total_words))
            end_word_idx = min(total_words, int((end_token_idx / total_tokens) * total_words))

            # Ensure we have at least one word
            if end_word_idx <= start_word_idx:
                end_word_idx = min(start_word_idx + 1, total_words)

            chunk_words = words[start_word_idx:end_word_idx]
            return " ".join(chunk_words)

        except:
            # Fallback to character-based approximation
            return self._approximate_text_from_tokens(text, start_token_idx, end_token_idx, total_tokens)

    def _approximate_text_from_tokens(self, text: str, start_token_idx: int, end_token_idx: int, total_tokens: int) -> str:
        """Approximate text content from token indices."""
        # Map token indices to character indices (rough approximation)
        text_len = len(text)
        start_char = max(0, int((start_token_idx / total_tokens) * text_len))
        end_char = min(text_len, int((end_token_idx / total_tokens) * text_len))

        chunk_text = text[start_char:end_char]

        # Try to end on word boundary if preserve_word_boundaries is True
        if self.preserve_word_boundaries and chunk_text and not chunk_text.endswith(' '):
            # Find last space to end on word boundary
            last_space = chunk_text.rfind(' ')
            if last_space > len(chunk_text) * 0.8:  # Only if we don't lose too much content
                chunk_text = chunk_text[:last_space]

        return chunk_text

    def chunk_stream(
        self,
        content_stream: Iterator[Union[str, bytes]],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Iterator[Chunk]:
        """
        Chunk content from a stream using token-based approach.

        Args:
            content_stream: Iterator of content chunks
            source_info: Additional source information
            **kwargs: Additional parameters

        Yields:
            Chunk objects
        """
        self._initialize_tokenizer()

        buffer = ""
        buffer_tokens = []
        chunk_index = 0

        for content_chunk in content_stream:
            if isinstance(content_chunk, bytes):
                content_chunk = content_chunk.decode('utf-8', errors='ignore')

            buffer += content_chunk

            # Tokenize new content
            try:
                current_tokens = self.encode_func(buffer)
            except:
                # Fallback tokenization
                words = buffer.split()
                current_tokens = [hash(word) % 50000 for word in words]

            buffer_tokens = current_tokens

            # Create chunks when we have enough tokens
            while len(buffer_tokens) >= self.tokens_per_chunk:
                # Extract chunk tokens
                chunk_tokens = buffer_tokens[:self.tokens_per_chunk]

                # Get corresponding text
                try:
                    if hasattr(self.tokenizer, 'decode') or self.decode_func:
                        chunk_text = self.decode_func(chunk_tokens)
                        if chunk_text.startswith("[") and "tokens]" in chunk_text:
                            # Approximate if decode doesn't work properly
                            approx_chars = len(chunk_tokens) * 4  # rough estimate
                            chunk_text = buffer[:min(approx_chars, len(buffer))]
                    else:
                        # Approximate text length
                        approx_chars = len(chunk_tokens) * 4
                        chunk_text = buffer[:min(approx_chars, len(buffer))]
                except:
                    approx_chars = len(chunk_tokens) * 4
                    chunk_text = buffer[:min(approx_chars, len(buffer))]

                # Ensure we don't exceed character limit
                if len(chunk_text) > self.max_chunk_chars:
                    chunk_text = chunk_text[:self.max_chunk_chars]

                # Create chunk
                chunk_metadata = ChunkMetadata(
                    source=source_info.get("source", "stream") if source_info else "stream",
                    source_type="stream",
                    position=f"tokens {chunk_index * self.tokens_per_chunk}-{chunk_index * self.tokens_per_chunk + len(chunk_tokens) - 1}",
                    length=len(chunk_text),
                    extra={
                        "token_count": len(chunk_tokens),
                        "chunk_index": chunk_index,
                        "chunking_strategy": "token_based",
                        "tokenizer_type": self.tokenizer_type.value,
                        "tokenizer_model": self.tokenizer_model,
                        "is_streaming": True
                    }
                )

                chunk = Chunk(
                    id=f"stream_token_{chunk_index}",
                    content=chunk_text,
                    modality=ModalityType.TEXT,
                    metadata=chunk_metadata,
                    size=len(chunk_text)
                )

                yield chunk
                chunk_index += 1

                # Remove processed tokens (accounting for overlap)
                tokens_to_remove = max(1, self.tokens_per_chunk - self.overlap_tokens)
                buffer_tokens = buffer_tokens[tokens_to_remove:]

                # Update buffer text to match remaining tokens
                if buffer_tokens:
                    try:
                        buffer = self.decode_func(buffer_tokens)
                        if buffer.startswith("[") and "tokens]" in buffer:
                            # Keep proportion of original buffer
                            keep_ratio = len(buffer_tokens) / len(current_tokens)
                            buffer = buffer[int(len(buffer) * (1 - keep_ratio)):]
                    except:
                        # Keep approximate portion of buffer
                        keep_ratio = len(buffer_tokens) / len(current_tokens) if current_tokens else 0.5
                        buffer = buffer[int(len(buffer) * (1 - keep_ratio)):]
                else:
                    buffer = ""

        # Handle remaining content in buffer
        if buffer.strip() and len(buffer_tokens) >= self.min_chunk_tokens:
            chunk_metadata = ChunkMetadata(
                source=source_info.get("source", "stream") if source_info else "stream",
                source_type="stream",
                position=f"tokens {chunk_index * self.tokens_per_chunk}-{chunk_index * self.tokens_per_chunk + len(buffer_tokens) - 1}",
                length=len(buffer),
                extra={
                    "token_count": len(buffer_tokens),
                    "chunk_index": chunk_index,
                    "chunking_strategy": "token_based",
                    "tokenizer_type": self.tokenizer_type.value,
                    "tokenizer_model": self.tokenizer_model,
                    "is_streaming": True,
                    "is_final_chunk": True
                }
            )

            chunk = Chunk(
                id=f"stream_token_{chunk_index}",
                content=buffer,
                modality=ModalityType.TEXT,
                metadata=chunk_metadata,
                size=len(buffer)
            )

            yield chunk

    def adapt_parameters(self, feedback_score: float, feedback_type: str = "quality", **kwargs) -> Dict[str, Any]:
        """
        Adapt chunker parameters based on feedback.

        Args:
            feedback_score: Score from 0-1 indicating performance
            feedback_type: Type of feedback ("quality", "performance", "size")
            **kwargs: Additional feedback parameters
        """
        import time

        old_tokens_per_chunk = self.tokens_per_chunk
        old_overlap_tokens = self.overlap_tokens

        # Apply adaptations based on feedback score and type
        if feedback_score < 0.5:  # Poor performance
            if feedback_type == "quality":
                # Smaller chunks for better granularity
                self.tokens_per_chunk = max(100, int(self.tokens_per_chunk * 0.8))
                self.overlap_tokens = min(self.tokens_per_chunk // 10, int(self.overlap_tokens * 1.2))
            elif feedback_type == "performance":
                # Larger chunks for better performance
                self.tokens_per_chunk = min(4000, int(self.tokens_per_chunk * 1.2))
                self.overlap_tokens = max(0, int(self.overlap_tokens * 0.8))

        elif feedback_score > 0.8:  # Good performance
            if feedback_type == "quality":
                # Slightly larger chunks to improve efficiency
                self.tokens_per_chunk = min(2000, int(self.tokens_per_chunk * 1.1))
            elif feedback_type == "performance":
                # Maintain current settings or slightly optimize
                self.tokens_per_chunk = min(3000, int(self.tokens_per_chunk * 1.05))

        # Ensure constraints are maintained
        self.overlap_tokens = min(self.overlap_tokens, self.tokens_per_chunk - 1)

        # Record adaptation if changes were made
        if old_tokens_per_chunk != self.tokens_per_chunk or old_overlap_tokens != self.overlap_tokens:
            adaptation_record = {
                "timestamp": time.time(),
                "feedback_score": feedback_score,
                "feedback_type": feedback_type,
                "changes": {
                    "tokens_per_chunk": {
                        "old": old_tokens_per_chunk,
                        "new": self.tokens_per_chunk
                    },
                    "overlap_tokens": {
                        "old": old_overlap_tokens,
                        "new": self.overlap_tokens
                    }
                },
                "reason": f"feedback_score={feedback_score:.2f}, type={feedback_type}",
                "kwargs": kwargs
            }
            self._adaptation_history.append(adaptation_record)

            logger.info(
                f"Adapted parameters: tokens_per_chunk {old_tokens_per_chunk} → {self.tokens_per_chunk}, "
                f"overlap_tokens {old_overlap_tokens} → {self.overlap_tokens} "
                f"(feedback: {feedback_score:.2f}, type: {feedback_type})"
            )

            return adaptation_record["changes"]

        return {}  # No changes made

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of adaptations made.

        Returns:
            List of adaptation records with timestamps and changes
        """
        return self._adaptation_history.copy()

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "name": "token_based",
            "tokens_per_chunk": self.tokens_per_chunk,
            "overlap_tokens": self.overlap_tokens,
            "tokenizer_type": self.tokenizer_type.value,
            "tokenizer_model": self.tokenizer_model,
            "preserve_word_boundaries": self.preserve_word_boundaries,
            "min_chunk_tokens": self.min_chunk_tokens,
            "max_chunk_chars": self.max_chunk_chars,
            "tokenizer_info": self._tokenizer_info,
            "performance_stats": {
                "total_tokens_processed": self._total_tokens_processed,
                "tokenization_time": self._tokenization_time,
                "avg_tokenization_speed": (
                    self._total_tokens_processed / self._tokenization_time
                    if self._tokenization_time > 0 else 0
                )
            }
        }

    @classmethod
    def get_supported_tokenizers(cls) -> Dict[str, List[str]]:
        """Get list of supported tokenizers and models."""
        return {
            "tiktoken": [
                "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
                "text-davinci-003", "text-davinci-002"
            ],
            "transformers": [
                "bert-base-uncased", "bert-large-uncased", "gpt2", "gpt2-medium",
                "gpt2-large", "roberta-base", "distilbert-base-uncased",
                "t5-base", "t5-large", "all-MiniLM-L6-v2", "all-mpnet-base-v2"
            ],
            "spacy": [
                "en_core_web_sm", "en_core_web_md", "en_core_web_lg"
            ],
            "nltk": [
                "word_tokenize", "punkt"
            ],
            "simple": [
                "whitespace"
            ]
        }
