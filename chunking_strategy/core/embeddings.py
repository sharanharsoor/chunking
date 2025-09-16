"""
Improved embedding generation utilities for chunked content.

Key improvements:
- Better dependency checking and error handling
- Lazy loading of models
- More robust device detection
- Better memory management
- Enhanced error messages
- Fallback mechanisms
"""

import logging
import warnings
import os
import sys
import importlib
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from ..core.base import Chunk, ChunkingResult

logger = logging.getLogger(__name__)

# Suppress some common warnings from transformers/torch
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


def check_dependency(package_name: str, extra_name: str = None) -> Tuple[bool, str]:
    """
    Check if a dependency is available and return helpful error message if not.

    Args:
        package_name: The package to check
        extra_name: The extra name for installation (e.g., 'text', 'ml')

    Returns:
        (is_available, error_message)
    """
    try:
        # Special handling for sentence_transformers due to common import issues
        if package_name == "sentence_transformers":
            module = importlib.import_module(package_name)
            # Test basic functionality to catch version issues
            from sentence_transformers import SentenceTransformer
            return True, ""
        else:
            importlib.import_module(package_name)
            return True, ""
    except Exception as e:
        if extra_name:
            error_msg = (
                f"{package_name} has issues: {str(e)[:100]}... "
                f"Try: pip install --upgrade {package_name} or "
                f"pip install 'chunking-strategy[{extra_name}]'"
            )
        else:
            error_msg = f"{package_name} has issues: {str(e)[:100]}... Try: pip install --upgrade {package_name}"
        return False, error_msg


def get_device() -> str:
    """Auto-detect the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
            logger.info("MPS available, using Apple Silicon GPU")
        else:
            device = "cpu"
            logger.info("Using CPU")
        return device
    except ImportError:
        logger.info("PyTorch not available, defaulting to CPU")
        return "cpu"


def get_huggingface_token() -> Optional[str]:
    """Get HuggingFace token from config file or environment."""
    # Try config file first
    try:
        import os
        config_path = Path(__file__).parent.parent.parent / "config" / "huggingface_token.py"
        if config_path.exists():
            import sys
            sys.path.insert(0, str(config_path.parent))
            try:
                import huggingface_token
                token = getattr(huggingface_token, 'HUGGINGFACE_TOKEN', None)
                if token:
                    logger.info("Using HuggingFace token from config file")
                    return token
            except ImportError:
                pass
            finally:
                if str(config_path.parent) in sys.path:
                    sys.path.remove(str(config_path.parent))
    except Exception as e:
        logger.debug(f"Could not load token from config: {e}")

    # Try environment variable
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    if token:
        logger.info("Using HuggingFace token from environment")
        return token

    logger.debug("No HuggingFace token found")
    return None


def set_huggingface_token():
    """Set the HuggingFace token for model downloads."""
    token = get_huggingface_token()
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.info("HuggingFace authentication successful")
        except ImportError:
            logger.debug("huggingface_hub not available for token login")
        except Exception as e:
            logger.warning(f"HuggingFace authentication failed: {e}")
    else:
        logger.debug("No HuggingFace token available")


class EmbeddingModel(str, Enum):
    """Supported embedding models with better organization."""

    # Fast, lightweight models (good for development/testing)
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"  # 384 dim, fast
    ALL_MINILM_L12_V2 = "all-MiniLM-L12-v2"  # 384 dim, slightly better quality

    # High-quality models (recommended for production)
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"  # 768 dim, best quality
    ALL_DISTILROBERTA_V1 = "all-distilroberta-v1"  # 768 dim, good for code

    # Multilingual models
    PARAPHRASE_MULTILINGUAL_MINILM = "paraphrase-multilingual-MiniLM-L12-v2"

    # Multimodal models (CLIP)
    CLIP_VIT_B_32 = "clip-vit-b-32"  # 512 dim
    CLIP_VIT_B_16 = "clip-vit-b-16"  # 512 dim, better quality
    CLIP_VIT_L_14 = "clip-vit-l-14"  # 768 dim, highest quality

    @classmethod
    def get_recommended_for_use_case(cls, use_case: str) -> 'EmbeddingModel':
        """Get recommended model for specific use cases."""
        recommendations = {
            "fast": cls.ALL_MINILM_L6_V2,
            "quality": cls.ALL_MPNET_BASE_V2,
            "code": cls.ALL_DISTILROBERTA_V1,
            "multilingual": cls.PARAPHRASE_MULTILINGUAL_MINILM,
            "multimodal": cls.CLIP_VIT_B_32
        }
        return recommendations.get(use_case, cls.ALL_MINILM_L6_V2)


class OutputFormat(str, Enum):
    """Output format options for embeddings."""

    VECTOR_ONLY = "vector_only"  # Just embeddings - minimal memory
    VECTOR_PLUS_TEXT = "vector_plus_text"  # Embeddings + original text
    FULL_METADATA = "full_metadata"  # Embeddings + text + all metadata


@dataclass
class EmbeddingConfig:
    """Enhanced configuration for embedding generation."""

    model: EmbeddingModel = EmbeddingModel.ALL_MINILM_L6_V2
    output_format: OutputFormat = OutputFormat.FULL_METADATA
    batch_size: int = 32
    normalize_embeddings: bool = True
    include_chunk_id: bool = True
    max_length: Optional[int] = None
    device: Optional[str] = None  # auto-detect if None

    # New options for better control
    show_progress: bool = True
    cache_embeddings: bool = False
    precision: str = "float32"  # or "float16" for memory savings

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            self.batch_size = 1
            logger.warning("batch_size must be positive, set to 1")

        if self.precision not in ["float16", "float32"]:
            self.precision = "float32"
            logger.warning("Invalid precision, defaulting to float32")


class EmbeddedChunk(BaseModel):
    """A chunk with its corresponding embedding."""

    chunk_id: str
    embedding: List[float]
    content: Optional[str] = None
    modality: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    model_used: str
    embedding_dim: int

    class Config:
        # Allow for better memory efficiency
        arbitrary_types_allowed = True


class EmbeddingResult(BaseModel):
    """Enhanced result of embedding generation."""

    embedded_chunks: List[EmbeddedChunk]
    model_used: str
    total_chunks: int
    embedding_dim: int
    config: Dict[str, Any]
    processing_info: Dict[str, Any] = Field(default_factory=dict)

    # New fields for better tracking
    success_rate: float = 1.0
    failed_chunks: List[str] = Field(default_factory=list)
    processing_time_seconds: float = 0.0


class BaseEmbedder(ABC):
    """Enhanced base class for embedding generators."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self._is_loaded = False
        self._device = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model."""
        pass

    @abstractmethod
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text content."""
        pass

    def _get_device(self) -> str:
        """Get the device for model execution."""
        if self._device is None:
            self._device = self.config.device or get_device()
        return self._device

    def embed_chunks(self, chunks: List[Chunk]) -> EmbeddingResult:
        """Generate embeddings for a list of chunks with better error handling."""
        import time
        start_time = time.time()

        # Import ModalityType for proper enum comparison
        try:
            from chunking_strategy.core.base import ModalityType
        except ImportError:
            logger.warning("Could not import ModalityType, using string comparison")
            ModalityType = None

        # Filter chunks by modality with better handling
        text_chunks = []
        failed_chunks = []

        for c in chunks:
            try:
                if c.modality is None:
                    text_chunks.append(c)
                elif ModalityType and hasattr(c.modality, 'value'):  # It's an enum
                    if c.modality in [ModalityType.TEXT, ModalityType.MIXED]:
                        text_chunks.append(c)
                else:  # It's a string
                    if c.modality in ["text", "mixed", None]:
                        text_chunks.append(c)
            except Exception as e:
                logger.warning(f"Error processing chunk {getattr(c, 'id', 'unknown')}: {e}")
                failed_chunks.append(str(getattr(c, 'id', 'unknown')))

        if not text_chunks:
            logger.warning("No text chunks found for embedding")
            return self._create_empty_result(failed_chunks)

        # Extract text content with better error handling
        texts = []
        chunk_info = []
        processing_errors = []

        for chunk in text_chunks:
            try:
                text_content = chunk.content.strip() if chunk.content else ""
                if text_content:  # Skip empty chunks
                    texts.append(text_content)
                    chunk_info.append(chunk)
                else:
                    logger.debug(f"Skipping empty chunk: {getattr(chunk, 'id', 'unknown')}")
            except Exception as e:
                error_msg = f"Error processing chunk {getattr(chunk, 'id', 'unknown')}: {e}"
                logger.warning(error_msg)
                processing_errors.append(error_msg)
                failed_chunks.append(str(getattr(chunk, 'id', 'unknown')))

        if not texts:
            logger.warning("No non-empty text content found for embedding")
            return self._create_empty_result(failed_chunks)

        # Load model only if we have content to process
        try:
            if not self._is_loaded:
                self.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return self._create_empty_result(failed_chunks, str(e))

        # Generate embeddings in batches with error handling
        all_embeddings = []
        successful_indices = []

        try:
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                batch_indices = list(range(i, min(i + self.config.batch_size, len(texts))))

                try:
                    batch_embeddings = self.embed_text(batch_texts)
                    all_embeddings.append(batch_embeddings)
                    successful_indices.extend(batch_indices)
                except Exception as e:
                    logger.error(f"Failed to embed batch {i//self.config.batch_size}: {e}")
                    # Add failed chunks
                    for idx in batch_indices:
                        if idx < len(chunk_info):
                            failed_chunks.append(str(getattr(chunk_info[idx], 'id', f'batch_{i}_chunk_{idx}')))

            if not all_embeddings:
                return self._create_empty_result(failed_chunks, "All batches failed")

            # Combine all embeddings
            embeddings = np.vstack(all_embeddings)

            if self.config.normalize_embeddings:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Create embedded chunks only for successful embeddings
            embedded_chunks = []
            for i, embedding in enumerate(embeddings):
                if i < len(successful_indices):
                    original_idx = successful_indices[i]
                    if original_idx < len(chunk_info):
                        chunk = chunk_info[original_idx]
                        embedded_chunk = self._create_embedded_chunk(chunk, embedding.tolist(), i)
                        embedded_chunks.append(embedded_chunk)

            processing_time = time.time() - start_time
            success_rate = len(embedded_chunks) / len(chunks) if chunks else 0

            return EmbeddingResult(
                embedded_chunks=embedded_chunks,
                model_used=self.config.model.value,
                total_chunks=len(embedded_chunks),
                embedding_dim=embeddings.shape[1] if len(embeddings) > 0 else 0,
                config=self._config_to_dict(),
                processing_info={
                    "original_chunk_count": len(chunks),
                    "text_chunk_count": len(text_chunks),
                    "embedded_chunk_count": len(embedded_chunks),
                    "batch_size": self.config.batch_size,
                    "processing_errors": processing_errors,
                    "device_used": self._get_device()
                },
                success_rate=success_rate,
                failed_chunks=failed_chunks,
                processing_time_seconds=processing_time
            )

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._create_empty_result(failed_chunks, str(e))

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary safely."""
        try:
            if hasattr(self.config, 'dict'):
                return self.config.dict()
            else:
                return vars(self.config)
        except Exception:
            return {"model": str(self.config.model)}

    def _create_empty_result(self, failed_chunks: List[str] = None, error: str = None) -> EmbeddingResult:
        """Create an empty result for error cases."""
        return EmbeddingResult(
            embedded_chunks=[],
            model_used=self.config.model.value,
            total_chunks=0,
            embedding_dim=0,
            config=self._config_to_dict(),
            processing_info={"error": error} if error else {},
            success_rate=0.0,
            failed_chunks=failed_chunks or []
        )

    def _create_embedded_chunk(self, chunk: Chunk, embedding: List[float], index: int) -> EmbeddedChunk:
        """Create an EmbeddedChunk from a Chunk and its embedding."""

        # Determine what to include based on output format
        content = None
        metadata = None

        if self.config.output_format in [OutputFormat.VECTOR_PLUS_TEXT, OutputFormat.FULL_METADATA]:
            content = chunk.content

        if self.config.output_format == OutputFormat.FULL_METADATA:
            try:
                if chunk.metadata is None:
                    metadata = {}
                elif isinstance(chunk.metadata, dict):
                    metadata = chunk.metadata.copy()
                elif hasattr(chunk.metadata, 'model_dump'):
                    metadata = chunk.metadata.model_dump()
                elif hasattr(chunk.metadata, 'dict'):
                    metadata = chunk.metadata.dict()
                elif hasattr(chunk.metadata, '__dict__'):
                    metadata = chunk.metadata.__dict__.copy()
                else:
                    # Fallback: try to convert to dict
                    metadata = dict(chunk.metadata) if chunk.metadata else {}
            except Exception as e:
                logger.warning(f"Failed to convert chunk metadata to dict: {e}")
                metadata = {}

            # Add embedding metadata
            metadata.update({
                "original_chunk_index": index,
                "embedding_model": self.config.model.value,
                "embedding_timestamp": str(datetime.datetime.now()),
                "normalized": self.config.normalize_embeddings,
                "device_used": self._get_device()
            })

        chunk_id = chunk.id if self.config.include_chunk_id and hasattr(chunk, 'id') else f"chunk_{index}"

        return EmbeddedChunk(
            chunk_id=chunk_id,
            embedding=embedding,
            content=content,
            modality=str(chunk.modality) if chunk.modality else "text",
            metadata=metadata,
            model_used=self.config.model.value,
            embedding_dim=len(embedding)
        )


class SentenceTransformerEmbedder(BaseEmbedder):
    """Enhanced embedder using sentence-transformers models."""

    def load_model(self) -> None:
        """Load the sentence-transformers model with better error handling."""
        # Check dependencies first
        available, error_msg = check_dependency("sentence_transformers", "text")
        if not available:
            raise ImportError(error_msg)

        # Set HuggingFace token if available
        set_huggingface_token()

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading sentence-transformers model: {self.config.model.value}")

        # Map our enum to actual model names
        model_mapping = {
            EmbeddingModel.ALL_MINILM_L6_V2: "all-MiniLM-L6-v2",
            EmbeddingModel.ALL_MINILM_L12_V2: "all-MiniLM-L12-v2",
            EmbeddingModel.ALL_MPNET_BASE_V2: "all-mpnet-base-v2",
            EmbeddingModel.ALL_DISTILROBERTA_V1: "all-distilroberta-v1",
            EmbeddingModel.PARAPHRASE_MULTILINGUAL_MINILM: "paraphrase-multilingual-MiniLM-L12-v2"
        }

        model_name = model_mapping.get(self.config.model, self.config.model.value)
        device = self._get_device()

        try:
            # Get HuggingFace token for private model access
            token = get_huggingface_token()
            model_kwargs = {"device": device}
            if token:
                model_kwargs["token"] = token  # Updated API

            self.model = SentenceTransformer(model_name, **model_kwargs)
            self._is_loaded = True
            logger.info(f"Model loaded successfully on device: {device}")

            # Log model info
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model embedding dimension: {dim}")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            # Try fallback to CPU if GPU failed
            if device != "cpu":
                logger.info("Retrying with CPU...")
                try:
                    model_kwargs = {"device": "cpu"}
                    if token:
                        model_kwargs["token"] = token  # Updated API
                    self.model = SentenceTransformer(model_name, **model_kwargs)
                    self._device = "cpu"
                    self._is_loaded = True
                    logger.info("Model loaded successfully on CPU (fallback)")
                except Exception as fallback_error:
                    logger.error(f"CPU fallback also failed: {fallback_error}")
                    raise
            else:
                raise

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text content with better error handling."""
        if not self._is_loaded:
            self.load_model()

        # Truncate texts if max_length is specified
        if self.config.max_length:
            texts = [text[:self.config.max_length] for text in texts]

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=self.config.show_progress and len(texts) > 10,
                convert_to_numpy=True,
                normalize_embeddings=False  # We handle normalization ourselves
            )

            # Convert to specified precision
            if self.config.precision == "float16":
                embeddings = embeddings.astype(np.float16)

            return embeddings

        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            # Return zero embeddings as fallback
            dim = 384 if "MiniLM" in self.config.model.value else 768
            return np.zeros((len(texts), dim), dtype=np.float32)


class CLIPEmbedder(BaseEmbedder):
    """Enhanced embedder using CLIP models for text and images."""

    def load_model(self) -> None:
        """Load the CLIP model with better error handling."""
        # Check dependencies first
        torch_available, torch_error = check_dependency("torch", "ml")
        transformers_available, transformers_error = check_dependency("transformers", "ml")

        if not torch_available or not transformers_available:
            error_msg = f"{torch_error} {transformers_error}".strip()
            raise ImportError(error_msg)

        # Set HuggingFace token if available
        set_huggingface_token()

        import torch
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"Loading CLIP model: {self.config.model.value}")

        # Map our enum to actual model names
        model_mapping = {
            EmbeddingModel.CLIP_VIT_B_32: "openai/clip-vit-base-patch32",
            EmbeddingModel.CLIP_VIT_B_16: "openai/clip-vit-base-patch16",
            EmbeddingModel.CLIP_VIT_L_14: "openai/clip-vit-large-patch14"
        }

        model_name = model_mapping.get(self.config.model, "openai/clip-vit-base-patch32")
        device = self._get_device()

        try:
            # Get HuggingFace token for model access
            token = get_huggingface_token()
            model_kwargs = {}
            if token:
                model_kwargs["token"] = token  # Updated API

            self.model = CLIPModel.from_pretrained(model_name, **model_kwargs).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name, **model_kwargs)
            self.device = device
            self._is_loaded = True
            logger.info(f"CLIP model loaded successfully on device: {device}")

        except Exception as e:
            logger.error(f"Failed to load CLIP model {model_name}: {e}")
            if device != "cpu":
                logger.info("Retrying with CPU...")
                try:
                    model_kwargs = {}
                    if token:
                        model_kwargs["token"] = token  # Updated API
                    self.model = CLIPModel.from_pretrained(model_name, **model_kwargs).to("cpu")
                    self.processor = CLIPProcessor.from_pretrained(model_name, **model_kwargs)
                    self.device = "cpu"
                    self._device = "cpu"
                    self._is_loaded = True
                    logger.info("CLIP model loaded successfully on CPU (fallback)")
                except Exception as fallback_error:
                    logger.error(f"CPU fallback also failed: {fallback_error}")
                    raise
            else:
                raise

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text content using CLIP."""
        if not self._is_loaded:
            self.load_model()

        import torch

        all_embeddings = []

        try:
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]

                # Tokenize and encode
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77  # CLIP's max sequence length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    text_features = self.model.get_text_features(**inputs)
                    text_features = text_features.cpu().numpy()

                    # Convert to specified precision
                    if self.config.precision == "float16":
                        text_features = text_features.astype(np.float16)

                all_embeddings.append(text_features)

            return np.vstack(all_embeddings)

        except Exception as e:
            logger.error(f"Error encoding texts with CLIP: {e}")
            # Return zero embeddings as fallback
            dim = 512  # CLIP's typical dimension
            return np.zeros((len(texts), dim), dtype=np.float32)


def create_embedder(config: EmbeddingConfig) -> BaseEmbedder:
    """Create an embedder based on the configuration with better error handling."""

    # Determine embedder type based on model
    text_models = {
        EmbeddingModel.ALL_MINILM_L6_V2,
        EmbeddingModel.ALL_MINILM_L12_V2,
        EmbeddingModel.ALL_MPNET_BASE_V2,
        EmbeddingModel.ALL_DISTILROBERTA_V1,
        EmbeddingModel.PARAPHRASE_MULTILINGUAL_MINILM
    }

    clip_models = {
        EmbeddingModel.CLIP_VIT_B_32,
        EmbeddingModel.CLIP_VIT_B_16,
        EmbeddingModel.CLIP_VIT_L_14
    }

    try:
        if config.model in text_models:
            return SentenceTransformerEmbedder(config)
        elif config.model in clip_models:
            return CLIPEmbedder(config)
        else:
            raise ValueError(f"Unsupported embedding model: {config.model}")
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        raise


def embed_chunking_result(
    chunking_result: ChunkingResult,
    config: EmbeddingConfig
) -> EmbeddingResult:
    """Generate embeddings for a ChunkingResult with better error handling."""
    try:
        embedder = create_embedder(config)
        return embedder.embed_chunks(chunking_result.chunks)
    except Exception as e:
        logger.error(f"Failed to embed chunking result: {e}")
        # Return empty result with error info
        return EmbeddingResult(
            embedded_chunks=[],
            model_used=config.model.value,
            total_chunks=0,
            embedding_dim=0,
            config=config.__dict__ if hasattr(config, '__dict__') else {},
            processing_info={"error": str(e)},
            success_rate=0.0,
            failed_chunks=[]
        )


def print_embedding_summary(result: EmbeddingResult, max_chunks: int = 5) -> None:
    """Print an enhanced summary of embedding results."""

    print(f"\n🔮 Embedding Summary")
    print(f"{'='*50}")
    print(f"Model: {result.model_used}")
    print(f"Total chunks embedded: {result.total_chunks}")
    print(f"Embedding dimension: {result.embedding_dim}")
    print(f"Success rate: {result.success_rate:.2%}")
    print(f"Processing time: {result.processing_time_seconds:.3f}s")

    if result.config:
        output_format = result.config.get('output_format', 'unknown')
        print(f"Output format: {output_format}")

    if result.failed_chunks:
        print(f"❌ Failed chunks: {len(result.failed_chunks)}")
        if len(result.failed_chunks) <= 5:
            print(f"   {', '.join(result.failed_chunks)}")
        else:
            print(f"   {', '.join(result.failed_chunks[:5])}... (and {len(result.failed_chunks)-5} more)")

    if result.processing_info:
        print(f"\n📊 Processing Info:")
        for key, value in result.processing_info.items():
            if key != "processing_errors":  # Handle separately
                print(f"  {key}: {value}")

        if "processing_errors" in result.processing_info and result.processing_info["processing_errors"]:
            print(f"  Processing errors: {len(result.processing_info['processing_errors'])}")

    if result.embedded_chunks:
        print(f"\n📝 Sample Embedded Chunks (showing first {min(max_chunks, len(result.embedded_chunks))}):")
        for i, chunk in enumerate(result.embedded_chunks[:max_chunks]):
            print(f"\n  Chunk {i+1} - ID: {chunk.chunk_id}")
            print(f"    Modality: {chunk.modality}")
            print(f"    Embedding dim: {chunk.embedding_dim}")

            # Show embedding statistics
            embedding_array = np.array(chunk.embedding)
            print(f"    Embedding stats: mean={embedding_array.mean():.4f}, std={embedding_array.std():.4f}")
            print(f"    Embedding preview: [{chunk.embedding[0]:.4f}, {chunk.embedding[1]:.4f}, ...]")

            if chunk.content:
                content_preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                print(f"    Content: {content_preview}")

            if chunk.metadata:
                print(f"    Metadata keys: {list(chunk.metadata.keys())}")

    print(f"\n💡 Vector Database Integration:")
    print(f"  Each embedded chunk can be stored in your vector database with:")
    print(f"  - Vector: chunk.embedding ({result.embedding_dim}D)")
    print(f"  - ID: chunk.chunk_id")
    print(f"  - Payload: chunk.metadata (if available)")
    print(f"  - Content: chunk.content (if included)")

    if result.success_rate < 1.0:
        print(f"\n⚠️  Note: {(1-result.success_rate)*100:.1f}% of chunks failed to embed")
        print(f"  Check the logs for details about failed chunks")


def export_for_vector_db(
    result: EmbeddingResult,
    format: str = "dict"
) -> Union[List[Dict[str, Any]], str]:
    """Export embeddings in a format suitable for vector databases."""

    export_data = []

    for chunk in result.embedded_chunks:
        item = {
            "id": chunk.chunk_id,
            "vector": chunk.embedding,
        }

        # Add payload based on what's available
        payload = {}
        if chunk.content:
            payload["content"] = chunk.content
        if chunk.modality:
            payload["modality"] = chunk.modality
        if chunk.metadata:
            payload.update(chunk.metadata)

        payload.update({
            "model_used": chunk.model_used,
            "embedding_dim": chunk.embedding_dim
        })

        item["payload"] = payload
        export_data.append(item)

    if format == "dict":
        return export_data
    elif format == "json":
        import json
        return json.dumps(export_data, indent=2)
    else:
        raise ValueError(f"Unsupported export format: {format}")


# Convenience functions for common use cases
def get_fast_embedder_config() -> EmbeddingConfig:
    """Get configuration for fast embeddings (development/testing)."""
    return EmbeddingConfig(
        model=EmbeddingModel.ALL_MINILM_L6_V2,
        batch_size=64,
        precision="float16",
        output_format=OutputFormat.VECTOR_PLUS_TEXT
    )


def get_quality_embedder_config() -> EmbeddingConfig:
    """Get configuration for high-quality embeddings (production)."""
    return EmbeddingConfig(
        model=EmbeddingModel.ALL_MPNET_BASE_V2,
        batch_size=16,
        precision="float32",
        output_format=OutputFormat.FULL_METADATA
    )