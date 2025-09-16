"""
Configuration-driven orchestrator for chunking strategies.

This module provides the main orchestration interface that can detect file types,
select appropriate chunking strategies, apply fallbacks, and handle complex
chunking workflows based on configuration profiles.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import yaml

# Import centralized logging system
from chunking_strategy.logging_config import (
    get_logger, user_info, user_success, user_warning, user_error,
    debug_operation, performance_log
)

from chunking_strategy.core.base import (
    BaseChunker,
    Chunk,
    ChunkingResult,
    ModalityType
)
from chunking_strategy.core.registry import (
    create_chunker,
    list_chunkers,
    get_chunker_metadata,
    get_registry
)
from chunking_strategy.core.custom_config_integration import (
    CustomConfigProcessor,
    load_config_with_custom_algorithms,
    CustomConfigError
)
from chunking_strategy.core.hardware import get_smart_parallelization_config

# Strategy name mapping for backward compatibility
STRATEGY_NAME_MAPPING = {
    'sentence': 'sentence_based',
    'paragraph': 'paragraph_based',
    'fastcdc': 'fastcdc',
    'content_defined': 'fastcdc',
    'adaptive': 'adaptive',
    'adaptive_dynamic': 'adaptive',
    'intelligent': 'adaptive',
    'context_enriched': 'context_enriched',
    'semantic': 'context_enriched',
    'contextual': 'context_enriched',

    # Hash-based chunkers
    'rolling_hash': 'rolling_hash',
    'rabin': 'rabin_fingerprinting',
    'rabin_fingerprinting': 'rabin_fingerprinting',
    'rfc': 'rabin_fingerprinting',
    'buzhash': 'buzhash',
    'buzz_hash': 'buzhash',
    'gear': 'gear_cdc',
    'gear_cdc': 'gear_cdc',
    'gear_based': 'gear_cdc',
    'ml_cdc': 'ml_cdc',
    'multilevel': 'ml_cdc',
    'multi_level': 'ml_cdc',
    'tttd': 'tttd',
    'two_threshold': 'tttd',
    'two_threshold_two_divisor': 'tttd',
    'python': 'python_code',
    'c_cpp': 'c_cpp_code',
    'java': 'java_code',
    'javascript': 'javascript_code',
    'typescript': 'javascript_code',
    'go': 'go_code',
    'css': 'css_code',
    'scss': 'css_code',
    'sass': 'css_code',
    'rust': 'universal_code',
    'php': 'universal_code',
    'ruby': 'universal_code',
    'csharp': 'universal_code',
    'swift': 'universal_code',
    'kotlin': 'universal_code',
    'scala': 'universal_code',
    'code': 'universal_code',
    'pdf': 'pdf_chunker',
    'doc': 'doc_chunker',
    'csv': 'csv_chunker',
    'json': 'json_chunker',
    'markdown': 'markdown_chunker',
    'xml': 'xml_html_chunker',
    'html': 'xml_html_chunker'
}
from chunking_strategy.core.universal_framework import (
    apply_universal_strategy, get_universal_strategy_registry
)
from chunking_strategy.core.extractors import get_extractor_registry
from chunking_strategy.core.pipeline import ChunkingPipeline, ChunkerStep, FilterStep, MergeStep
from chunking_strategy.core.streaming import StreamingChunker
from chunking_strategy.core.adaptive import AdaptiveChunker
from chunking_strategy.core.tika_integration import get_tika_processor
from chunking_strategy.detectors.file_type_detector import FileTypeDetector
from chunking_strategy.detectors.content_analyzer import ContentAnalyzer
from chunking_strategy.utils.preprocessing import PreprocessingPipeline
from chunking_strategy.utils.postprocessing import PostprocessingPipeline
from chunking_strategy.core.hardware import HardwareDetector, HardwareInfo

logger = get_logger(__name__)


class ChunkerOrchestrator:
    """
    Main orchestrator for configuration-driven chunking.

    The orchestrator can:
    - Detect file types and content characteristics
    - Select appropriate chunking strategies based on configuration
    - Apply fallback strategies when primary strategies fail
    - Handle preprocessing and postprocessing steps
    - Manage streaming and memory-efficient processing
    - Apply adaptive chunking with feedback

    Examples:
        Basic usage:
        ```python
        orchestrator = ChunkerOrchestrator()
        result = orchestrator.chunk_file("document.pdf")
        ```

        With configuration:
        ```python
        orchestrator = ChunkerOrchestrator(config_path="my_config.yaml")
        result = orchestrator.chunk_file("document.pdf")
        ```

        With custom configuration:
        ```python
        config = {
            "strategies": {
                "primary": "semantic_chunking",
                "fallbacks": ["sentence_based", "paragraph_based"]
            }
        }
        orchestrator = ChunkerOrchestrator(config=config)
        ```
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        default_profile: str = "balanced",
        enable_hardware_optimization: Optional[bool] = None,
        enable_smart_parallelization: bool = True,
        enable_custom_algorithms: bool = True
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Configuration dictionary
            config_path: Path to configuration file
            default_profile: Default profile to use if none specified
            enable_hardware_optimization: Whether to enable hardware optimization
            enable_smart_parallelization: Whether to use smart parallelization (recommended)
            enable_custom_algorithms: Whether to process custom algorithms in configs
        """
        self.logger = get_logger(f"{__name__}.ChunkerOrchestrator")
        self.enable_custom_algorithms = enable_custom_algorithms
        self.custom_config_processor = CustomConfigProcessor() if enable_custom_algorithms else None
        self.loaded_custom_algorithms: Dict[str, Any] = {}

        # Load configuration
        if config_path:
            self.config = self._load_config_file(config_path)
        elif config:
            self.config = self._process_config(config)
        else:
            self.config = self._get_default_config(default_profile)

        # Initialize components
        self.file_detector = FileTypeDetector()
        self.content_analyzer = ContentAnalyzer()
        self.preprocessor = PreprocessingPipeline()
        self.postprocessor = PostprocessingPipeline()

        # Initialize hardware detection and smart parallelization
        # Check environment variables for hardware optimization (useful for testing)
        if enable_hardware_optimization is None:
            env_hw_opt = os.environ.get("CHUNKING_ENABLE_HARDWARE_OPT", "True").lower()
            self.enable_hardware_optimization = env_hw_opt in ("true", "1", "yes")
        else:
            self.enable_hardware_optimization = enable_hardware_optimization

        self.enable_smart_parallelization = enable_smart_parallelization
        self.smart_config = get_smart_parallelization_config() if enable_smart_parallelization else None

        if self.enable_hardware_optimization:
            self.hardware_detector = HardwareDetector()
            # Use cached hardware info if smart parallelization is enabled
            if self.enable_smart_parallelization and self.smart_config:
                self.hardware_info = self.smart_config.get_cached_hardware_info()
            else:
                self.hardware_info = self.hardware_detector.detect_hardware()

            user_info(f"Hardware detected: {self.hardware_info.cpu_count} cores, "
                     f"{self.hardware_info.memory_total_gb:.1f}GB RAM, "
                     f"{self.hardware_info.gpu_count} GPUs")
        else:
            self.hardware_detector = None
            self.hardware_info = None

        # Cache for strategy instances
        self._chunker_cache: Dict[str, BaseChunker] = {}

        user_info(f"Using '{default_profile}' chunking profile")

    def get_loaded_custom_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about loaded custom algorithms.

        Returns:
            Dictionary mapping algorithm names to their information
        """
        result = {}
        for name, info in self.loaded_custom_algorithms.items():
            result[name] = info.to_dict() if hasattr(info, 'to_dict') else str(info)
        return result

    def list_all_available_strategies(self) -> List[str]:
        """
        List all available strategies including custom algorithms.

        Returns:
            List of all available strategy names
        """
        built_in = list_chunkers()
        custom = list(self.loaded_custom_algorithms.keys())
        return sorted(built_in + custom)

    def is_custom_algorithm(self, strategy_name: str) -> bool:
        """
        Check if a strategy is a custom algorithm.

        Args:
            strategy_name: Name of the strategy

        Returns:
            True if it's a custom algorithm
        """
        return strategy_name in self.loaded_custom_algorithms

    def _normalize_strategy_name(self, strategy_name: str) -> str:
        """Normalize strategy name using backward compatibility mapping."""
        return STRATEGY_NAME_MAPPING.get(strategy_name, strategy_name)

    def _normalize_strategy_list(self, strategies: List[str]) -> List[str]:
        """Normalize a list of strategy names."""
        return [self._normalize_strategy_name(strategy) for strategy in strategies]

    def chunk_files_batch(
        self,
        file_paths: List[Union[str, Path]],
        strategies: Optional[List[str]] = None,
        parallel_mode: str = "auto",
        max_workers: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[ChunkingResult]:
        """
        Process multiple files with hardware-optimized batch processing.

        Args:
            file_paths: List of file paths to process
            strategies: Optional list of strategies (uses auto-selection if None)
            parallel_mode: "auto", "thread", "process", or "sequential"
            max_workers: Maximum number of workers (auto-detected if None)
            progress_callback: Optional progress callback function

        Returns:
            List of ChunkingResult objects
        """
        if not file_paths:
            return []

        # Import here to avoid circular imports
        from chunking_strategy.core.batch import BatchProcessor

        # Convert to Path objects and calculate total/average file sizes
        paths = [Path(p) for p in file_paths]
        total_size = sum(p.stat().st_size for p in paths if p.exists())
        avg_file_size_mb = total_size / len(paths) / (1024 * 1024) if paths else 0

        # Smart decision: use hardware optimization only when beneficial
        use_hw_optimization = self.enable_hardware_optimization
        if self.enable_smart_parallelization and self.smart_config:
            use_hw_optimization = (
                use_hw_optimization and
                self.smart_config.should_use_threading(total_size, len(paths))
            )
            if not use_hw_optimization:
                self.logger.debug(f"Smart parallelization: Using sequential processing "
                                f"(total_size={total_size}, num_files={len(paths)})")

        # Get optimal configuration
        if use_hw_optimization and self.hardware_detector:
            config = self.hardware_detector.get_optimal_batch_config(
                total_files=len(paths),
                avg_file_size_mb=avg_file_size_mb,
                user_workers=max_workers,
                force_cpu=(parallel_mode == "sequential")
            )

            # Override with user preferences
            if parallel_mode != "auto":
                if parallel_mode == "sequential":
                    actual_mode = "sequential"
                elif parallel_mode == "thread":
                    actual_mode = "thread"
                else:
                    actual_mode = "process"
            else:
                # Use hardware recommendation
                actual_mode = "thread" if config['use_gpu'] else "process"

        else:
            # Fallback configuration without hardware optimization
            config = {
                'batch_size': 4,
                'workers': max_workers or 2,
                'use_gpu': False
            }
            actual_mode = parallel_mode if parallel_mode != "auto" else "sequential"

        # Create batch processor
        processor = BatchProcessor(
            progress_callback=progress_callback
        )

        # Determine default strategy
        default_strategy = strategies[0] if strategies else "sentence_based"

        # Process files
        result = processor.process_files(
            files=paths,
            default_strategy=default_strategy,
            batch_size=config.get('batch_size'),
            workers=config.get('workers'),
            use_gpu=config.get('use_gpu'),
            parallel_mode=actual_mode
        )

        # Extract individual results
        chunk_results = []
        for file_path in paths:
            file_str = str(file_path)
            if file_str in result.chunk_results:
                chunk_results.append(result.chunk_results[file_str])
            else:
                # Create empty result for failed files
                from chunking_strategy.core.base import ChunkingResult
                chunk_results.append(ChunkingResult(chunks=[], source_info={"source": file_str}))

        return chunk_results

    def chunk_file(
        self,
        file_path: Union[str, Path],
        strategy_override: Optional[str] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk a file using automatic strategy selection.

        Args:
            file_path: Path to the file to chunk
            strategy_override: Override automatic strategy selection
            **kwargs: Additional parameters for chunking

        Returns:
            ChunkingResult with chunks and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            user_error(f"File not found: {file_path}")
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        user_info(f"Chunking file: {file_path}")
        start_time = time.time()

        # Detect file characteristics
        file_info = self._analyze_file(file_path)

        # Ensure file_extension is always present for strategy selection
        if "file_extension" not in file_info:
            file_info["file_extension"] = file_path.suffix.lower()

        # Handle backward compatibility for 'strategy' parameter in kwargs
        if 'strategy' in kwargs and not strategy_override:
            strategy_override = kwargs.pop('strategy')

        # Select chunking strategy
        if strategy_override:
            primary_strategy = self._normalize_strategy_name(strategy_override)
            fallback_strategies = []
        else:
            primary_strategy, fallback_strategies = self._select_strategy(file_info)
            primary_strategy = self._normalize_strategy_name(primary_strategy)
            fallback_strategies = self._normalize_strategy_list(fallback_strategies)

        # Check if we should use streaming for large files
        file_size = file_path.stat().st_size
        use_streaming = False
        if self.enable_smart_parallelization and self.smart_config:
            use_streaming = self.smart_config.should_use_streaming(file_size)

        # Allow user override via kwargs
        if 'force_streaming' in kwargs:
            use_streaming = kwargs.pop('force_streaming')
        elif 'disable_streaming' in kwargs:
            use_streaming = not kwargs.pop('disable_streaming')

        if use_streaming:
            user_info(f"Using streaming for large file ({file_size:,} bytes)")
            return self._chunk_file_streaming(
                file_path, primary_strategy, fallback_strategies, file_info, **kwargs
            )

        # Determine if we should pass file path or loaded content
        pdf_chunkers = {'enhanced_pdf_chunker', 'pdf_chunker', 'pdf'}
        should_pass_file_path = (primary_strategy in pdf_chunkers or
                                any(fb in pdf_chunkers for fb in fallback_strategies))

        if should_pass_file_path:
            # Pass file path directly to PDF chunkers (they handle binary files properly)
            content = file_path
            self.logger.debug(f"Passing file path directly to PDF chunker: {primary_strategy}")
        else:
            # Load and preprocess content (regular path for text files)
            content = self._load_content(file_path, file_info)
            if self.config.get("preprocessing", {}).get("enabled", False):
                content = self.preprocessor.process(content, file_info)

        # Perform chunking with fallbacks
        result = self._chunk_with_fallbacks(
            content=content,
            primary_strategy=primary_strategy,
            fallback_strategies=fallback_strategies,
            source_info={
                "source": str(file_path),
                "file_type": file_info.get("file_type"),
                "detected_modality": file_info.get("modality"),
                "file_size": file_path.stat().st_size,
                **file_info
            },
            **kwargs
        )

        # Apply postprocessing
        if self.config.get("postprocessing", {}).get("enabled", False):
            result = self.postprocessor.process(result, self.config.get("postprocessing", {}))

        # Update result metadata
        result.processing_time = time.time() - start_time
        result.source_info = result.source_info or {}
        result.source_info.update({
            "orchestrator_used": True,
            "primary_strategy": primary_strategy,
            "fallback_strategies": fallback_strategies,
            "config_profile": self.config.get("profile_name", "unknown")
        })

        user_success(
            f"Chunking completed in {result.processing_time:.3f}s, "
            f"produced {len(result.chunks)} chunks using {result.strategy_used}"
        )

        return result

    def chunk_content(
        self,
        content: Union[str, bytes],
        content_type: Optional[str] = None,
        strategy_override: Optional[str] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk raw content using automatic strategy selection.

        Args:
            content: Content to chunk
            content_type: MIME type or content type hint
            strategy_override: Override automatic strategy selection
            **kwargs: Additional parameters for chunking

        Returns:
            ChunkingResult with chunks and metadata
        """
        user_info(f"Chunking content ({len(content)} chars/bytes)")
        start_time = time.time()

        # Analyze content characteristics
        content_info = self._analyze_content(content, content_type)

        # Select chunking strategy
        if strategy_override:
            primary_strategy = strategy_override
            fallback_strategies = []
        else:
            primary_strategy, fallback_strategies = self._select_strategy(content_info)

        # Preprocess content
        if self.config.get("preprocessing", {}).get("enabled", False):
            content = self.preprocessor.process(content, content_info)

        # Perform chunking with fallbacks
        result = self._chunk_with_fallbacks(
            content=content,
            primary_strategy=primary_strategy,
            fallback_strategies=fallback_strategies,
            source_info={
                "source": "direct_content",
                "content_type": content_type,
                "detected_modality": content_info.get("modality"),
                "content_length": len(content),
                **content_info
            },
            **kwargs
        )

        # Apply postprocessing
        if self.config.get("postprocessing", {}).get("enabled", False):
            result = self.postprocessor.process(result, self.config.get("postprocessing", {}))

        # Update result metadata
        result.processing_time = time.time() - start_time

        return result

    def chunk_streaming(
        self,
        file_path: Union[str, Path],
        strategy_override: Optional[str] = None,
        **kwargs
    ) -> StreamingChunker:
        """
        Create a streaming chunker for a file.

        Args:
            file_path: Path to the file to stream
            strategy_override: Override automatic strategy selection
            **kwargs: Additional parameters for streaming

        Returns:
            Configured StreamingChunker instance
        """
        file_path = Path(file_path)
        file_info = self._analyze_file(file_path)

        if strategy_override:
            strategy = strategy_override
        else:
            strategy, _ = self._select_strategy(file_info)

        # Get strategy configuration
        strategy_config = self.config.get("strategies", {}).get("configs", {}).get(strategy, {})

        return StreamingChunker(
            strategy=strategy,
            block_size=kwargs.get("block_size", 64 * 1024),
            overlap_size=kwargs.get("overlap_size", 0),
            **strategy_config
        )

    def create_adaptive_chunker(
        self,
        base_strategy: Optional[str] = None,
        **kwargs
    ) -> AdaptiveChunker:
        """
        Create an adaptive chunker with orchestrator configuration.

        Args:
            base_strategy: Base strategy to use (auto-detected if None)
            **kwargs: Additional parameters for adaptive chunker

        Returns:
            Configured AdaptiveChunker instance
        """
        if not base_strategy:
            # Use default strategy from config
            base_strategy = self.config.get("strategies", {}).get("primary", "fixed_size")

        fallback_strategies = self.config.get("strategies", {}).get("fallbacks", [])
        strategy_config = self.config.get("strategies", {}).get("configs", {}).get(base_strategy, {})

        return AdaptiveChunker(
            base_strategy=base_strategy,
            fallback_strategies=fallback_strategies,
            **strategy_config,
            **kwargs
        )

    def create_pipeline(
        self,
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> ChunkingPipeline:
        """
        Create a chunking pipeline from configuration.

        Args:
            pipeline_config: Pipeline configuration (uses default if None)

        Returns:
            Configured ChunkingPipeline instance
        """
        if not pipeline_config:
            pipeline_config = self.config.get("pipeline", {})

        pipeline = ChunkingPipeline(name=pipeline_config.get("name", "orchestrated_pipeline"))

        # Add steps from configuration
        for step_config in pipeline_config.get("steps", []):
            step_name = step_config["name"]
            step_type = step_config["type"]

            if step_type == "chunker":
                strategy = step_config.get("strategy", "fixed_size")
                config = step_config.get("config", {})
                step = ChunkerStep(step_name, strategy, **config)

            elif step_type == "filter":
                config = step_config.get("config", {})
                step = FilterStep(step_name, **config)

            elif step_type == "merge":
                config = step_config.get("config", {})
                step = MergeStep(step_name, **config)

            else:
                self.logger.warning(f"Unknown pipeline step type: {step_type}")
                continue

            pipeline.add_step(step_name, step)

        return pipeline

    def get_strategy_recommendations(
        self,
        file_path: Optional[Union[str, Path]] = None,
        content: Optional[Union[str, bytes]] = None,
        use_case: str = "general",
        performance_priority: str = "balanced"
    ) -> List[str]:
        """
        Get recommended strategies for given input.

        Args:
            file_path: Path to file (for file-based recommendations)
            content: Content to analyze (for content-based recommendations)
            use_case: Target use case
            performance_priority: Performance optimization preference

        Returns:
            List of recommended strategy names in priority order
        """
        if file_path:
            file_info = self._analyze_file(Path(file_path))
            modality = file_info.get("modality", ModalityType.TEXT)
        elif content:
            content_info = self._analyze_content(content)
            modality = content_info.get("modality", ModalityType.TEXT)
        else:
            modality = ModalityType.TEXT

        registry = get_registry()
        return registry.get_recommendations(
            modality=modality,
            use_case=use_case,
            performance_priority=performance_priority
        )

    def validate_config(self) -> List[str]:
        """
        Validate orchestrator configuration.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check required sections
        required_sections = ["strategies"]
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required configuration section: {section}")

        # Validate strategies section
        strategies = self.config.get("strategies", {})
        primary = strategies.get("primary")
        if primary and not create_chunker(primary):
            issues.append(f"Primary strategy not available: {primary}")

        fallbacks = strategies.get("fallbacks", [])
        for fallback in fallbacks:
            if not create_chunker(fallback):
                issues.append(f"Fallback strategy not available: {fallback}")

        return issues

    def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze file characteristics using both standard detectors and Tika when beneficial."""
        try:
            # Standard file analysis
            file_info = self.file_detector.detect(file_path)
            content_info = self.content_analyzer.analyze_file(file_path)

            base_info = {
                **file_info,
                **content_info,
                "file_size": file_path.stat().st_size,
                "file_name": file_path.name,
                "file_extension": file_path.suffix.lower()  # Ensure lowercase for consistent strategy matching
            }

            # Use Tika for enhanced document analysis if available and beneficial
            file_extension = file_path.suffix.lower()
            tika_beneficial_formats = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
                                     '.odt', '.rtf', '.epub', '.html', '.xml'}

            if file_extension in tika_beneficial_formats:
                try:
                    tika_processor = get_tika_processor()
                    if tika_processor and tika_processor.is_available():
                        self.logger.debug(f"Using Tika for enhanced analysis of {file_path}")

                        # Get Tika file type detection
                        tika_file_info = tika_processor.detect_file_type(file_path)

                        # Enhance base info with Tika results
                        base_info.update({
                            "tika_available": True,
                            "tika_mime_type": tika_file_info.get("mime_type"),
                            "tika_detected_type": tika_file_info.get("file_type"),
                            "enhanced_analysis": True
                        })

                        # Use Tika's file type if it's more specific
                        if tika_file_info.get("file_type") and base_info.get("file_type") == "unknown":
                            base_info["file_type"] = tika_file_info["file_type"]

                        self.logger.debug(f"Tika enhanced analysis: {tika_file_info}")
                    else:
                        base_info["tika_available"] = False
                except Exception as tika_error:
                    self.logger.debug(f"Tika analysis failed for {file_path}: {tika_error}")
                    base_info["tika_available"] = False
                    base_info["tika_error"] = str(tika_error)
            else:
                base_info["tika_available"] = False
                base_info["tika_reason"] = "Format not beneficial for Tika"

            return base_info

        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {
                "file_type": "unknown",
                "modality": ModalityType.TEXT,
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
                "file_name": file_path.name,
                "file_extension": file_path.suffix,
                "tika_available": False
            }

    def _analyze_content(
        self,
        content: Union[str, bytes],
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze content characteristics."""
        try:
            return self.content_analyzer.analyze_content(content, content_type)
        except Exception as e:
            self.logger.error(f"Error analyzing content: {e}")
            return {"modality": ModalityType.TEXT}

    def _select_strategy(
        self,
        content_info: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Select primary and fallback strategies based on content info."""
        modality = content_info.get("modality", ModalityType.TEXT)
        file_type = content_info.get("file_type", "text")

        # Check for multi-strategy configuration first
        multi_strategy_config = self.config.get("multi_strategy", {})
        if multi_strategy_config.get("enabled", False):
            strategies = multi_strategy_config.get("strategies", [])
            if strategies:
                # Return multi_strategy as primary with strategy names as fallbacks
                strategy_names = [s.get("name") for s in strategies if s.get("name")]
                if strategy_names:
                    return "multi_strategy", strategy_names

        # Get strategy selection rules from config (support multiple config formats)

        # Format 1: Standard strategies.primary format (most common)
        strategies_config = self.config.get("strategies", {})
        if strategies_config and "primary" in strategies_config:
            default_strategy = strategies_config.get("primary", "fixed_size")
            fallback_strategies = strategies_config.get("fallbacks", strategies_config.get("fallback", ["sentence_based", "paragraph_based", "fixed_size"]))
            # Also check for root-level strategy_selection rules
            selection_rules = self.config.get("strategy_selection", {})

        # Format 2: Audio chunking config format (chunking.strategy_selection)
        elif "chunking" in self.config:
            chunking_config = self.config.get("chunking", {})
            selection_rules = chunking_config.get("strategy_selection", {})
            default_strategy = chunking_config.get("default_strategy", "fixed_size")
            fallback_strategies = chunking_config.get("fallbacks", ["sentence_based", "paragraph_based", "fixed_size"])

        # Format 3: Direct config format (strategy_selection at root)
        else:
            selection_rules = self.config.get("strategy_selection", {})
            default_strategy = self.config.get("default_strategy", "fixed_size")
            fallback_strategies = self.config.get("fallbacks", ["sentence_based", "paragraph_based", "fixed_size"])

        file_extension = content_info.get("file_extension", "").lower()

        # Check for file extension rules first (most specific)
        if file_extension in selection_rules:
            strategy = selection_rules[file_extension]
            if isinstance(strategy, str):
                # Simple string mapping: ".wav": "time_based_audio"
                return strategy, ["fixed_size", "sentence_based"]
            elif isinstance(strategy, dict):
                # Complex rule mapping: ".wav": {"primary": "...", "fallbacks": [...]}
                return strategy.get("primary", default_strategy), strategy.get("fallbacks", ["fixed_size"])

        # Check for specific file type rules
        if file_type in selection_rules:
            strategy = selection_rules[file_type]
            if isinstance(strategy, str):
                return strategy, ["fixed_size", "sentence_based"]
            elif isinstance(strategy, dict):
                return strategy.get("primary", default_strategy), strategy.get("fallbacks", ["fixed_size"])

        # Check for modality-based rules
        modality_str = modality.value if hasattr(modality, 'value') else str(modality)
        if modality_str in selection_rules:
            strategy = selection_rules[modality_str]
            if isinstance(strategy, str):
                return strategy, ["fixed_size", "sentence_based"]
            elif isinstance(strategy, dict):
                return strategy.get("primary", default_strategy), strategy.get("fallbacks", ["fixed_size"])

        # If default_strategy is "auto", use auto-selection
        if default_strategy == "auto":
            return self._auto_select_strategy(content_info)

        # Use configured default strategy
        return default_strategy, fallback_strategies

    def _auto_select_strategy(self, content_info: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Auto-select the best strategy based on simple file extension mapping."""
        file_extension = content_info.get("file_extension", "").lower()
        file_size = content_info.get("file_size", 0)

        # Simple but effective file extension to strategy mapping
        auto_strategy_map = {
            # Text files - sentence-based works well for readable content
            ".txt": ("sentence_based", ["paragraph_based", "fixed_size"]),
            ".md": ("markdown", ["paragraph_based", "sentence_based"]),  # Markdown has clear structure
            ".markdown": ("markdown", ["paragraph_based", "sentence_based"]),
            ".mdown": ("markdown", ["paragraph_based", "sentence_based"]),
            ".mkd": ("markdown", ["paragraph_based", "sentence_based"]),
            ".rst": ("paragraph_based", ["sentence_based", "fixed_size"]),

            # Code files - use specialized code chunkers for optimal structure preservation
            ".py": ("python", ["paragraph_based", "sentence_based"]),
            ".pyx": ("python", ["paragraph_based", "sentence_based"]),
            ".pyi": ("python", ["paragraph_based", "sentence_based"]),

            # C/C++ files - use dedicated C/C++ chunker
            ".c": ("c_cpp", ["paragraph_based", "sentence_based"]),
            ".cpp": ("c_cpp", ["paragraph_based", "sentence_based"]),
            ".cc": ("c_cpp", ["paragraph_based", "sentence_based"]),
            ".cxx": ("c_cpp", ["paragraph_based", "sentence_based"]),
            ".h": ("c_cpp", ["paragraph_based", "sentence_based"]),
            ".hpp": ("c_cpp", ["paragraph_based", "sentence_based"]),
            ".hxx": ("c_cpp", ["paragraph_based", "sentence_based"]),

            # JavaScript/TypeScript - use dedicated JavaScript chunker
            ".js": ("javascript", ["paragraph_based", "sentence_based"]),
            ".jsx": ("javascript", ["paragraph_based", "sentence_based"]),
            ".ts": ("typescript", ["paragraph_based", "sentence_based"]),
            ".tsx": ("typescript", ["paragraph_based", "sentence_based"]),
            ".mjs": ("javascript", ["paragraph_based", "sentence_based"]),  # ES modules
            ".cjs": ("javascript", ["paragraph_based", "sentence_based"]),  # CommonJS

            # Other languages - use universal code chunker
            ".java": ("java", ["paragraph_based", "sentence_based"]),
            ".go": ("go", ["paragraph_based", "sentence_based"]),
            ".rs": ("rust", ["paragraph_based", "sentence_based"]),
            ".rb": ("ruby", ["paragraph_based", "sentence_based"]),
            ".php": ("php", ["paragraph_based", "sentence_based"]),
            ".cs": ("csharp", ["paragraph_based", "sentence_based"]),
            ".swift": ("swift", ["paragraph_based", "sentence_based"]),
            ".kt": ("kotlin", ["paragraph_based", "sentence_based"]),
            ".scala": ("scala", ["paragraph_based", "sentence_based"]),

            # Additional languages supported by universal chunker
            ".r": ("code", ["paragraph_based", "sentence_based"]),
            ".R": ("code", ["paragraph_based", "sentence_based"]),
            ".m": ("code", ["paragraph_based", "sentence_based"]),  # MATLAB/Objective-C
            ".pl": ("code", ["paragraph_based", "sentence_based"]),  # Perl
            ".pm": ("code", ["paragraph_based", "sentence_based"]),  # Perl modules
            ".sh": ("code", ["paragraph_based", "sentence_based"]),  # Shell
            ".bash": ("code", ["paragraph_based", "sentence_based"]),
            ".zsh": ("code", ["paragraph_based", "sentence_based"]),
            ".fish": ("code", ["paragraph_based", "sentence_based"]),
            ".lua": ("code", ["paragraph_based", "sentence_based"]),
            ".vim": ("code", ["paragraph_based", "sentence_based"]),
            ".sql": ("code", ["paragraph_based", "sentence_based"]),
            ".dockerfile": ("code", ["paragraph_based", "sentence_based"]),
            ".makefile": ("code", ["paragraph_based", "sentence_based"]),
            ".cmake": ("code", ["paragraph_based", "sentence_based"]),

            # Documents - format-specific chunkers for optimal processing
            ".pdf": ("enhanced_pdf_chunker", ["pdf_chunker", "sentence_based"]),
            ".doc": ("doc", ["paragraph_based", "sentence_based"]),
            ".docx": ("doc", ["paragraph_based", "sentence_based"]),
            ".odt": ("doc", ["paragraph_based", "sentence_based"]),
            ".rtf": ("doc", ["paragraph_based", "sentence_based"]),

            # Data files - format-specific chunkers for structured data
            ".csv": ("csv", ["fixed_size", "paragraph_based"]),
            ".json": ("json", ["fixed_size", "paragraph_based"]),
            ".jsonl": ("json", ["fixed_size", "paragraph_based"]),
            ".ndjson": ("json", ["fixed_size", "paragraph_based"]),
            ".xml": ("xml", ["fixed_size", "paragraph_based"]),
            ".html": ("html", ["paragraph_based", "fixed_size"]),
            ".htm": ("html", ["paragraph_based", "fixed_size"]),
            ".xhtml": ("html", ["paragraph_based", "fixed_size"]),
            ".svg": ("xml", ["fixed_size", "paragraph_based"]),
            ".yaml": ("fixed_size", ["sentence_based", "paragraph_based"]),
            ".yml": ("fixed_size", ["sentence_based", "paragraph_based"]),

            # Config files - sentence-based for readability
            ".conf": ("sentence_based", ["fixed_size", "paragraph_based"]),
            ".ini": ("sentence_based", ["fixed_size", "paragraph_based"]),
            ".toml": ("sentence_based", ["fixed_size", "paragraph_based"]),

            # Log files - fixed-size for consistent chunks
            ".log": ("fixed_size", ["sentence_based", "paragraph_based"]),

            # Web files - paragraph-based for structure
            ".css": ("css", ["paragraph_based", "sentence_based"]),
            ".scss": ("scss", ["paragraph_based", "sentence_based"]),
            ".sass": ("sass", ["paragraph_based", "sentence_based"]),
            ".less": ("css", ["paragraph_based", "sentence_based"]),

        # Audio files - silence-based segmentation with time-based fallback
        ".mp3": ("silence_based_audio", ["time_based_audio", "fixed_size", "sentence_based"]),
        ".wav": ("silence_based_audio", ["time_based_audio", "fixed_size", "sentence_based"]),
        ".ogg": ("silence_based_audio", ["time_based_audio", "fixed_size", "sentence_based"]),
        ".flac": ("silence_based_audio", ["time_based_audio", "fixed_size", "sentence_based"]),
        ".m4a": ("silence_based_audio", ["time_based_audio", "fixed_size", "sentence_based"]),
        ".aac": ("silence_based_audio", ["time_based_audio", "fixed_size", "sentence_based"]),
        ".wma": ("silence_based_audio", ["time_based_audio", "fixed_size", "sentence_based"]),
        ".opus": ("silence_based_audio", ["time_based_audio", "fixed_size", "sentence_based"]),

        # Video files - scene-based segmentation with time-based fallback
        ".mp4": ("scene_based_video", ["time_based_video", "fixed_size", "sentence_based"]),
        ".avi": ("scene_based_video", ["time_based_video", "fixed_size", "sentence_based"]),
        ".mov": ("scene_based_video", ["time_based_video", "fixed_size", "sentence_based"]),
        ".mkv": ("scene_based_video", ["time_based_video", "fixed_size", "sentence_based"]),
        ".webm": ("scene_based_video", ["time_based_video", "fixed_size", "sentence_based"]),
        ".flv": ("scene_based_video", ["time_based_video", "fixed_size", "sentence_based"]),
        ".wmv": ("scene_based_video", ["time_based_video", "fixed_size", "sentence_based"]),
        ".m4v": ("scene_based_video", ["time_based_video", "fixed_size", "sentence_based"]),

        # Image files - grid-based tiling for processing
        ".jpg": ("grid_based_image", ["fixed_size", "sentence_based"]),
        ".jpeg": ("grid_based_image", ["fixed_size", "sentence_based"]),
        ".png": ("grid_based_image", ["fixed_size", "sentence_based"]),
        ".gif": ("grid_based_image", ["fixed_size", "sentence_based"]),
        ".bmp": ("grid_based_image", ["fixed_size", "sentence_based"]),
        ".tiff": ("grid_based_image", ["fixed_size", "sentence_based"]),
        ".tif": ("grid_based_image", ["fixed_size", "sentence_based"]),
        ".webp": ("grid_based_image", ["fixed_size", "sentence_based"]),
        ".ico": ("grid_based_image", ["fixed_size", "sentence_based"]),
        ".svg": ("grid_based_image", ["fixed_size", "sentence_based"])
        }

        # Get strategy for file extension
        if file_extension in auto_strategy_map:
            primary, fallbacks = auto_strategy_map[file_extension]
        else:
            # Default fallback
            primary, fallbacks = "fixed_size", ["sentence_based", "paragraph_based"]

        # Define multimedia and format-specific strategies that should be preserved regardless of file size
        multimedia_strategies = [
            "time_based_audio", "silence_based_audio", "time_based_video", "scene_based_video", "grid_based_image", "patch_based_image"
        ]

        format_specific_strategies = [
            "csv", "json", "pdf", "doc", "markdown", "xml", "html",
            "python", "c_cpp", "javascript", "typescript", "go", "css", "scss", "sass", "java",
            "rust", "ruby", "php", "csharp", "swift", "kotlin", "scala", "code"
        ] + multimedia_strategies

        # Adjust strategy based on file size
        if file_size > 10 * 1024 * 1024:  # > 10MB
            # For very large files, optimize strategy but preserve multimedia and format-specific chunkers
            if (primary not in format_specific_strategies and
                primary not in ["overlapping_window", "rolling_hash"]):
                # Only apply large file optimization to generic text/document files
                # Multimedia and format-specific files should keep their specialized chunkers regardless of size
                fallbacks = [primary] + fallbacks
                primary = "rolling_hash"  # Better for large generic text files
        elif file_size < 1024:  # < 1KB
            # For very small files, prefer sentence-based, but respect format-specific chunkers
            if primary not in format_specific_strategies and primary != "sentence_based":
                fallbacks = [primary] + fallbacks
                primary = "sentence_based"

        self.logger.info(f"Auto-selected strategy '{primary}' for {file_extension} file")
        return primary, fallbacks

    def _select_strategy_for_file(self, filename: str) -> str:
        """Select strategy for a given filename based on extension."""
        file_path = Path(filename)
        file_extension = file_path.suffix.lower()

        # Create basic content info for strategy selection
        content_info = {
            "file_extension": file_extension,
            "file_size": 0,  # Default size
            "file_type": "unknown",  # Default type
            "modality": None  # Will be determined by auto-selection if needed
        }

        # Use full strategy selection logic to respect config rules
        primary, _ = self._select_strategy(content_info)
        return primary

    def _create_chunker(self, strategy_name: str, file_extension: str):
        """Create a chunker instance with configuration parameters."""
        from chunking_strategy.core.registry import create_chunker

        # Get strategy parameters from config
        strategy_params = {}
        if self.config and hasattr(self.config, "chunking") and hasattr(self.config.chunking, "strategy_params"):
            strategy_params = getattr(self.config.chunking.strategy_params, strategy_name, {})
        elif self.config and isinstance(self.config, dict):
            # Check both nested chunking.strategy_params and top-level parameters
            chunking_config = self.config.get("chunking", {})
            all_params = chunking_config.get("strategy_params", {})
            strategy_params = all_params.get(strategy_name, {})

            # Also check for top-level parameters key (more intuitive for custom algorithms)
            if not strategy_params and "parameters" in self.config:
                top_level_params = self.config.get("parameters", {})
                strategy_params = top_level_params.get(strategy_name, {})

        # Convert string parameters to appropriate types for multimedia chunkers
        if strategy_name in ["time_based_audio", "silence_based_audio", "time_based_video", "scene_based_video", "grid_based_image", "patch_based_image"] and strategy_params:
            converted_params = {}
            for key, value in strategy_params.items():
                if key in ["segment_duration", "overlap_duration", "min_segment_duration",
                          "silence_threshold_db", "min_silence_duration", "max_segment_duration", "padding_duration",
                          "scene_threshold", "min_scene_duration", "max_scene_duration", "sample_rate"]:
                    # Convert to float
                    converted_params[key] = float(value) if isinstance(value, str) else value
                elif key in ["sample_rate", "channels", "target_fps", "tile_width", "tile_height", "overlap_pixels",
                           "patch_width", "patch_height", "stride_x", "stride_y", "max_patches", "random_seed"]:
                    # Convert to int or None
                    if isinstance(value, str):
                        converted_params[key] = int(value) if value.isdigit() else None
                    else:
                        converted_params[key] = value
                elif key == "target_resolution":
                    # Convert to tuple or None
                    if isinstance(value, str):
                        try:
                            # Handle formats like "1920,720" or "1920x720"
                            if ',' in value:
                                width, height = value.split(',')
                            elif 'x' in value.lower():
                                width, height = value.lower().split('x')
                            else:
                                converted_params[key] = None
                                continue
                            converted_params[key] = (int(width.strip()), int(height.strip()))
                        except (ValueError, IndexError):
                            converted_params[key] = None
                    elif isinstance(value, (list, tuple)) and len(value) == 2:
                        converted_params[key] = tuple(value)
                    else:
                        converted_params[key] = value
                elif key in ["preserve_format", "pad_incomplete_tiles", "preserve_aspect_ratio"]:
                    # Convert to boolean
                    if isinstance(value, str):
                        converted_params[key] = value.lower() in ["true", "1", "yes", "on"]
                    else:
                        converted_params[key] = bool(value)
                else:
                    converted_params[key] = value
            strategy_params = converted_params

        # Create chunker with parameters - let validation errors bubble up
        chunker = create_chunker(strategy_name, **strategy_params)

        if chunker is None:
            raise ValueError(f"Failed to create chunker for strategy: {strategy_name}")

        return chunker

    def _load_content(self, file_path: Path, file_info: Dict[str, Any]) -> Union[str, bytes]:
        """Load content from file, using Tika for complex document extraction when beneficial."""
        try:
            # Check if we should use Tika for content extraction
            file_extension = file_path.suffix.lower()
            use_tika = (file_info.get("tika_available", False) and
                       file_extension in {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
                                         '.odt', '.rtf', '.epub', '.html', '.xml'})

            if use_tika:
                try:
                    tika_processor = get_tika_processor()
                    if tika_processor and tika_processor.is_available():
                        self.logger.debug(f"Using Tika for content extraction from {file_path}")

                        # Extract content and metadata using Tika
                        extraction_result = tika_processor.extract_content_and_metadata(file_path)

                        if extraction_result.get("content"):
                            self.logger.info(f"Tika successfully extracted {len(extraction_result['content']):,} characters from {file_path}")
                            return extraction_result["content"]
                        else:
                            self.logger.warning(f"Tika extraction returned empty content for {file_path}")

                except Exception as tika_error:
                    self.logger.warning(f"Tika content extraction failed for {file_path}: {tika_error}")
                    self.logger.info("Falling back to standard content loading")

            # Standard content loading (fallback or when Tika not needed)
            if file_info.get("modality") == ModalityType.TEXT:
                encoding = file_info.get("encoding", "utf-8")
                content = file_path.read_text(encoding=encoding)
                self.logger.debug(f"Standard text loading: {len(content):,} characters from {file_path}")
                return content
            else:
                content = file_path.read_bytes()
                self.logger.debug(f"Standard binary loading: {len(content):,} bytes from {file_path}")
                return content

        except Exception as e:
            user_warning(f"Content loading issue for {file_path.name}: {e}")
            self.logger.error(f"Error loading content from {file_path}: {e}")
            # Ultimate fallback to binary mode
            try:
                user_info(f"Attempting binary fallback for {file_path.name}")
                return file_path.read_bytes()
            except Exception as fallback_error:
                user_error(f"Failed to read file {file_path.name}: {fallback_error}")
                self.logger.error(f"Even binary fallback failed for {file_path}: {fallback_error}")
                raise

    def _chunk_with_fallbacks(
        self,
        content: Union[str, bytes, Path],
        primary_strategy: str,
        fallback_strategies: List[str],
        source_info: Dict[str, Any],
        **kwargs
    ) -> ChunkingResult:
        """Chunk content with fallback strategies (traditional + universal)."""
        strategies_tried = []

        # Handle multi-strategy case
        if primary_strategy == "multi_strategy":
            return self._execute_multi_strategy(content, fallback_strategies, source_info, **kwargs)

        # Try primary strategy
        try:
            result = self._apply_strategy(primary_strategy, content, source_info, **kwargs)
            if result and result.chunks:
                result.strategy_used = primary_strategy
                result.fallback_strategies = strategies_tried
                return result
        except Exception as e:
            self.logger.warning(f"Primary strategy {primary_strategy} failed: {e}")
            strategies_tried.append(primary_strategy)

        # Try fallback strategies
        for fallback in fallback_strategies:
            try:
                self.logger.info(f"Trying fallback strategy: {fallback}")
                result = self._apply_strategy(fallback, content, source_info, **kwargs)
                if result and result.chunks:
                    result.strategy_used = fallback
                    result.fallback_strategies = strategies_tried
                    return result
            except Exception as e:
                self.logger.warning(f"Fallback strategy {fallback} failed: {e}")
                strategies_tried.append(fallback)

        # If all strategies fail, create minimal result
        self.logger.error("All chunking strategies failed")
        from chunking_strategy.core.base import ChunkMetadata

        # Create a single chunk with the content
        metadata = ChunkMetadata(
            source=source_info.get("source", "unknown"),
            chunker_used="fallback_single_chunk"
        )

        chunk = Chunk(
            id="fallback_chunk_0",
            content=content,
            modality=source_info.get("detected_modality", ModalityType.TEXT),
            metadata=metadata
        )

        return ChunkingResult(
            chunks=[chunk],
            strategy_used="fallback_single_chunk",
            fallback_strategies=strategies_tried,
            source_info=source_info,
            errors=[f"All strategies failed: {strategies_tried}"]
        )

    def _apply_strategy(
        self,
        strategy_name: str,
        content: Union[str, bytes, Path],
        source_info: Dict[str, Any],
        **kwargs
    ) -> Optional[ChunkingResult]:
        """Apply a strategy (traditional or universal) to content."""
        # Get strategy configuration
        strategy_config = self.config.get("strategies", {}).get("configs", {}).get(strategy_name, {})
        combined_kwargs = {**strategy_config, **kwargs}

        # Check if this is a universal strategy
        universal_registry = get_universal_strategy_registry()
        if universal_registry.get_strategy(strategy_name):
            self.logger.debug(f"Using universal strategy: {strategy_name}")

            # Determine extractor based on file type/extension
            extractor_name = self._determine_extractor(content, source_info)

            return apply_universal_strategy(
                strategy_name=strategy_name,
                content=content,
                extractor_name=extractor_name,
                source_info=source_info,
                **combined_kwargs
            )

        # Otherwise, try traditional chunker
        else:
            self.logger.debug(f"Using traditional chunker: {strategy_name}")
            chunker = self._get_chunker(strategy_name)
            if chunker:
                # Configuration parameters are already passed during chunker initialization
                # Check if the chunk method accepts source_info parameter
                import inspect
                chunk_method = getattr(chunker, 'chunk', None)
                if chunk_method:
                    sig = inspect.signature(chunk_method)
                    params = sig.parameters

                    # Check if source_info is in the method signature
                    if 'source_info' in params:
                        return chunker.chunk(content, source_info=source_info)
                    else:
                        return chunker.chunk(content)

        return None

    def _execute_multi_strategy(
        self,
        content: Union[str, bytes, Path],
        strategy_names: List[str],
        source_info: Dict[str, Any],
        **kwargs
    ) -> ChunkingResult:
        """Execute multiple strategies and return combined result."""
        if not strategy_names:
            raise ValueError("No strategies provided for multi-strategy execution")

        results = {}
        successful_strategies = []
        failed_strategies = []

        self.logger.info(f"Executing multi-strategy with {len(strategy_names)} strategies: {strategy_names}")

        # Execute each strategy
        for strategy_name in strategy_names:
            try:
                result = self._apply_strategy(strategy_name, content, source_info, **kwargs)
                if result and result.chunks:
                    results[strategy_name] = result
                    successful_strategies.append(strategy_name)
                    self.logger.info(f"Strategy {strategy_name} succeeded with {len(result.chunks)} chunks")
                else:
                    failed_strategies.append(strategy_name)
                    self.logger.warning(f"Strategy {strategy_name} produced no chunks")
            except Exception as e:
                failed_strategies.append(strategy_name)
                self.logger.warning(f"Strategy {strategy_name} failed: {e}")

        if not successful_strategies:
            raise RuntimeError(f"All multi-strategy strategies failed: {failed_strategies}")

        # For now, return the first successful result but mark it as multi-strategy
        # In the future, this could be enhanced to combine results intelligently
        primary_result = results[successful_strategies[0]]
        primary_result.strategy_used = "multi_strategy"
        primary_result.fallback_strategies = failed_strategies

        # Add multi-strategy info to source_info
        if not primary_result.source_info:
            primary_result.source_info = {}
        primary_result.source_info.update({
            "multi_strategy_results": {name: len(results[name].chunks) for name in successful_strategies},
            "successful_strategies": successful_strategies,
            "failed_strategies": failed_strategies
        })

        return primary_result

    def _determine_extractor(
        self,
        content: Union[str, bytes, Path],
        source_info: Dict[str, Any]
    ) -> Optional[str]:
        """Determine the appropriate extractor for content."""
        if isinstance(content, Path):
            extension = content.suffix.lower()
        else:
            extension = source_info.get("file_extension", "").lower()

        # Get extractor registry and find suitable extractor
        extractor_registry = get_extractor_registry()
        extractor = extractor_registry.get_extractor(extension)

        return extractor.name if extractor else None

    def _get_chunker(self, strategy_name: str) -> Optional[BaseChunker]:
        """Get or create chunker instance."""
        # Normalize strategy name for backward compatibility
        normalized_name = self._normalize_strategy_name(strategy_name)

        if normalized_name not in self._chunker_cache:
            # Check multiple config locations for parameters
            config = {}

            # First try strategies.configs (traditional location)
            strategy_configs = self.config.get("strategies", {}).get("configs", {})
            if strategy_name in strategy_configs:
                config.update(strategy_configs[strategy_name])

            # Also check top-level parameters section (more intuitive for custom algorithms)
            if "parameters" in self.config:
                parameters_section = self.config.get("parameters", {})
                if strategy_name in parameters_section:
                    config.update(parameters_section[strategy_name])
                # Also try with normalized name
                if normalized_name in parameters_section:
                    config.update(parameters_section[normalized_name])

            chunker = create_chunker(normalized_name, **config)
            if chunker:
                self._chunker_cache[normalized_name] = chunker

        return self._chunker_cache.get(normalized_name)

    def list_available_strategies(self) -> Dict[str, List[str]]:
        """List all available chunking strategies."""
        traditional_strategies = list_chunkers()
        universal_strategies = get_universal_strategy_registry().list_strategies()

        return {
            "traditional": traditional_strategies,
            "universal": universal_strategies,
            "all": traditional_strategies + universal_strategies
        }

    def list_supported_file_types(self) -> Dict[str, List[str]]:
        """List all supported file types and their extractors."""
        registry = get_extractor_registry()

        extractors_info = {}
        for extractor in registry.extractors:
            extractors_info[extractor.name] = extractor.supported_extensions

        return {
            "extractors": extractors_info,
            "all_extensions": registry.list_supported_extensions()
        }

    def validate_strategy_config(self, strategy_name: str, file_extension: str) -> Dict[str, Any]:
        """Validate if a strategy can work with a file type."""
        # Import extractor registry at the top to avoid scoping issues
        from chunking_strategy.core.extractors import get_extractor_registry

        # Normalize strategy name for backward compatibility
        normalized_strategy = self._normalize_strategy_name(strategy_name)

        result = {
            "strategy": strategy_name,  # Keep original name in result
            "file_extension": file_extension,
            "is_valid": False,
            "method": None,
            "extractor": None,
            "reason": None
        }

        # Check if it's a traditional chunker
        traditional_chunker = create_chunker(normalized_strategy)
        if traditional_chunker:
            # Get supported formats from registration metadata
            metadata = get_chunker_metadata(normalized_strategy)
            if metadata:
                supported_formats = getattr(metadata, 'supported_formats', ['txt'])
            else:
                # Fallback to instance attribute
                supported_formats = getattr(traditional_chunker, 'supported_formats', ['txt'])

            if file_extension.lstrip('.') in supported_formats or '*' in supported_formats:
                result.update({
                    "is_valid": True,
                    "method": "traditional",
                    "reason": "Strategy directly supports this file type"
                })
                return result
            else:
                # Check if traditional chunker can work with this file type via extractor
                extractor_registry = get_extractor_registry()
                extractor = extractor_registry.get_extractor(file_extension)
                if extractor:
                    result.update({
                        "is_valid": True,
                        "method": "traditional_with_extractor",
                        "extractor": extractor.name,
                        "reason": f"Traditional strategy with {extractor.name} extractor"
                    })
                    return result

        # Check if it's a universal strategy
        universal_strategy = get_universal_strategy_registry().get_strategy(normalized_strategy)
        if universal_strategy:
            # Check if there's an extractor for this file type
            extractor_registry = get_extractor_registry()
            extractor = extractor_registry.get_extractor(file_extension)
            if extractor:
                result.update({
                    "is_valid": True,
                    "method": "universal",
                    "extractor": extractor.name,
                    "reason": f"Universal strategy with {extractor.name} extractor"
                })
                return result
            else:
                result.update({
                    "reason": f"No extractor available for {file_extension}"
                })
                return result

        result.update({
            "reason": f"Strategy '{strategy_name}' not found"
        })
        return result

    def _load_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file with custom algorithm support."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            # Load raw config first
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    # Handle both single and multi-document YAML files
                    try:
                        raw_config = yaml.safe_load(f)
                    except yaml.composer.ComposerError:
                        # Multi-document YAML, load the first document
                        f.seek(0)
                        raw_config = next(yaml.safe_load_all(f))
                elif config_path.suffix.lower() == '.json':
                    import json
                    raw_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

            # Process config for custom algorithms
            return self._process_config(raw_config, config_path)

        except Exception as e:
            self.logger.error(f"Error loading configuration file: {e}")
            raise

    def _process_config(
        self,
        config: Dict[str, Any],
        config_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Process configuration with custom algorithm support."""
        if not self.enable_custom_algorithms or not self.custom_config_processor:
            return config

        try:
            # Process custom algorithms in config
            processed_config = self.custom_config_processor.process_config(config, config_path)

            # Track loaded custom algorithms
            self.loaded_custom_algorithms.update(
                self.custom_config_processor.get_loaded_algorithms()
            )

            if self.loaded_custom_algorithms:
                self.logger.info(f"Loaded {len(self.loaded_custom_algorithms)} custom algorithms: "
                                f"{list(self.loaded_custom_algorithms.keys())}")

            return processed_config

        except CustomConfigError as e:
            self.logger.error(f"Custom algorithm configuration error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing custom algorithms: {e}")
            # If not in strict mode, continue without custom algorithms
            if self.custom_config_processor.loader.strict_validation:
                raise
            else:
                self.logger.warning("Continuing without custom algorithms due to processing error")
                return config

    def _chunk_file_streaming(
        self,
        file_path: Path,
        primary_strategy: str,
        fallback_strategies: List[str],
        file_info: Dict[str, Any],
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk a large file using streaming to avoid memory issues.

        Args:
            file_path: Path to the large file
            primary_strategy: Primary chunking strategy to use
            fallback_strategies: Fallback strategies if primary fails
            file_info: File analysis information
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with chunks and metadata
        """
        try:
            # Get streaming configuration
            if self.smart_config:
                streaming_config = self.smart_config.get_streaming_config()
            else:
                # Fallback streaming configuration
                streaming_config = {
                    'block_size': 64 * 1024 * 1024,  # 64MB
                    'overlap_size': 1024 * 1024,     # 1MB
                    'min_file_size': 100 * 1024 * 1024  # 100MB
                }

            # Allow user override of streaming parameters
            if 'streaming_block_size' in kwargs:
                streaming_config['block_size'] = kwargs.pop('streaming_block_size')
            if 'streaming_overlap_size' in kwargs:
                streaming_config['overlap_size'] = kwargs.pop('streaming_overlap_size')

            # Create streaming chunker with primary strategy
            streaming_chunker = StreamingChunker(
                primary_strategy,
                block_size=streaming_config['block_size'],
                overlap_size=streaming_config['overlap_size'],
                **kwargs
            )

            # Stream the file and collect chunks
            chunks = []
            total_chunks = 0
            start_time = time.time()

            self.logger.info(f"Streaming file with block_size={streaming_config['block_size']:,}, "
                           f"overlap_size={streaming_config['overlap_size']:,}")

            try:
                for chunk in streaming_chunker.stream_file(file_path):
                    chunks.append(chunk)
                    total_chunks += 1

                    # Log progress every 1000 chunks
                    if total_chunks % 1000 == 0:
                        elapsed = time.time() - start_time
                        self.logger.info(f"Streamed {total_chunks:,} chunks in {elapsed:.2f}s "
                                       f"({total_chunks/elapsed:.1f} chunks/sec)")

                # Create final result
                processing_time = time.time() - start_time
                self.logger.info(f"Streaming completed: {total_chunks:,} chunks in {processing_time:.2f}s")

                source_info = {
                    "source": str(file_path),
                    "file_type": file_info.get("file_type"),
                    "detected_modality": file_info.get("modality"),
                    "file_size": file_path.stat().st_size,
                    "streaming_used": True,
                    "streaming_config": streaming_config,
                    "orchestrator_used": True,
                    "primary_strategy": primary_strategy,
                    "processing_time": processing_time,
                    **file_info
                }

                from chunking_strategy.core.base import ChunkingResult
                return ChunkingResult(
                    chunks=chunks,
                    strategy_used=primary_strategy,
                    source_info=source_info
                )

            except Exception as streaming_error:
                self.logger.warning(f"Streaming with {primary_strategy} failed: {streaming_error}")

                # Try fallback strategies with streaming
                for fallback_strategy in fallback_strategies:
                    try:
                        self.logger.info(f"Trying fallback streaming with {fallback_strategy}")
                        fallback_streaming_chunker = StreamingChunker(
                            fallback_strategy,
                            block_size=streaming_config['block_size'],
                            overlap_size=streaming_config['overlap_size'],
                            **kwargs
                        )

                        chunks = []
                        for chunk in fallback_streaming_chunker.stream_file(file_path):
                            chunks.append(chunk)

                        processing_time = time.time() - start_time
                        self.logger.info(f"Fallback streaming succeeded with {fallback_strategy}")

                        source_info = {
                            "source": str(file_path),
                            "file_type": file_info.get("file_type"),
                            "detected_modality": file_info.get("modality"),
                            "file_size": file_path.stat().st_size,
                            "streaming_used": True,
                            "streaming_config": streaming_config,
                            "orchestrator_used": True,
                            "primary_strategy": fallback_strategy,
                            "processing_time": processing_time,
                            "fallback_used": True,
                            "primary_strategy_failed": primary_strategy,
                            **file_info
                        }

                        from chunking_strategy.core.base import ChunkingResult
                        return ChunkingResult(
                            chunks=chunks,
                            strategy_used=fallback_strategy,
                            source_info=source_info
                        )

                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback streaming with {fallback_strategy} failed: {fallback_error}")
                        continue

                # If all streaming attempts failed, raise the original error
                raise streaming_error

        except Exception as e:
            self.logger.error(f"Streaming chunking failed for {file_path}: {e}")
            # As a last resort, try non-streaming approach with size warning
            self.logger.warning("Falling back to non-streaming approach - may cause memory issues!")

            # Load content and use regular chunking (risky for large files)
            content = self._load_content(file_path, file_info)
            return self._chunk_with_fallbacks(
                content=content,
                primary_strategy=primary_strategy,
                fallback_strategies=fallback_strategies,
                source_info={
                    "source": str(file_path),
                    "file_type": file_info.get("file_type"),
                    "detected_modality": file_info.get("modality"),
                    "file_size": file_path.stat().st_size,
                    "streaming_failed": True,
                    "orchestrator_used": True,
                    **file_info
                },
                **kwargs
            )

    def _get_default_config(self, profile: str) -> Dict[str, Any]:
        """Get default configuration for profile."""
        if profile == "rag_optimized":
            return {
                "profile_name": "rag_optimized",
                "strategies": {
                    "primary": "semantic_chunking",
                    "fallbacks": ["sentence_based", "paragraph_based"],
                    "configs": {
                        "semantic_chunking": {"similarity_threshold": 0.7},
                        "sentence_based": {"max_sentences": 3},
                        "paragraph_based": {"max_paragraphs": 2}
                    }
                },
                "preprocessing": {"enabled": True, "normalize_whitespace": True},
                "postprocessing": {"enabled": True, "merge_short_chunks": True, "min_chunk_size": 100}
            }
        elif profile == "summarization":
            return {
                "profile_name": "summarization",
                "strategies": {
                    "primary": "paragraph_based",
                    "fallbacks": ["sentence_based", "fixed_size"],
                    "configs": {
                        "paragraph_based": {"max_paragraphs": 5},
                        "sentence_based": {"max_sentences": 10},
                        "fixed_size": {"chunk_size": 2048}
                    }
                },
                "preprocessing": {"enabled": True},
                "postprocessing": {"enabled": True, "remove_headers_footers": True}
            }
        else:  # balanced
            return {
                "profile_name": "balanced",
                "strategies": {
                    "primary": "auto",
                    "fallbacks": ["sentence_based", "paragraph_based", "fixed_size"],
                    "configs": {
                        "sentence_based": {"max_sentences": 5},
                        "paragraph_based": {"max_paragraphs": 3},
                        "fixed_size": {"chunk_size": 1024}
                    }
                },
                "preprocessing": {"enabled": False},
                "postprocessing": {"enabled": True, "merge_short_chunks": True}
            }
