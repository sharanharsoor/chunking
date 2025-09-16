"""
Enhanced Orchestrator with Advanced Strategy Selection.

This module extends the base orchestrator with intelligent strategy selection
that leverages advanced chunkers (FastCDC, Adaptive, Context-Enriched) based
on content analysis, file characteristics, and performance requirements.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import yaml

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
from chunking_strategy.orchestrator import ChunkerOrchestrator, STRATEGY_NAME_MAPPING


logger = logging.getLogger(__name__)


class EnhancedOrchestrator(ChunkerOrchestrator):
    """
    Enhanced orchestrator with intelligent strategy selection.

    This orchestrator extends the base orchestrator with:
    - Intelligent auto-selection of advanced chunkers
    - Content-aware strategy optimization
    - Performance-based strategy adaptation
    - Multi-modal content handling
    - Smart fallback chains
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        default_profile: str = "enhanced",
        enable_advanced_selection: bool = True,
        enable_performance_optimization: bool = True,
        enable_content_analysis: bool = True,
        **kwargs
    ):
        """
        Initialize enhanced orchestrator.

        Args:
            config: Configuration dictionary
            config_path: Path to configuration file
            default_profile: Default configuration profile
            enable_advanced_selection: Enable advanced strategy selection
            enable_performance_optimization: Enable performance-based optimization
            enable_content_analysis: Enable deep content analysis
            **kwargs: Additional arguments passed to base orchestrator
        """
        super().__init__(
            config=config,
            config_path=config_path,
            default_profile=default_profile,
            **kwargs
        )

        self.enable_advanced_selection = enable_advanced_selection
        self.enable_performance_optimization = enable_performance_optimization
        self.enable_content_analysis = enable_content_analysis

        # Strategy performance tracking
        self.strategy_performance = {}
        self.content_type_preferences = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Enhanced orchestrator initialized with advanced features")

    def _enhanced_auto_select_strategy(self, content_info: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Enhanced auto-selection with intelligent strategy prioritization.

        This method analyzes content characteristics and selects the most
        appropriate advanced chunker based on:
        - Content type and structure
        - File size and complexity
        - Performance requirements
        - Semantic content analysis
        """
        file_extension = content_info.get("file_extension", "").lower()
        file_size = content_info.get("file_size", 0)
        modality = content_info.get("modality", ModalityType.TEXT)

        # Get additional content characteristics
        content_complexity = content_info.get("complexity", "medium")
        has_structure = content_info.get("has_structure", False)
        text_ratio = content_info.get("text_ratio", 1.0)
        estimated_entropy = content_info.get("estimated_entropy", 4.0)

        self.logger.debug(f"Enhanced auto-selection for {file_extension}, size: {file_size}, "
                         f"complexity: {content_complexity}, text_ratio: {text_ratio}")

        # Priority 1: Advanced General-Purpose Chunkers
        advanced_strategy = self._select_advanced_strategy(
            file_size, content_complexity, text_ratio, estimated_entropy, has_structure
        )

        if advanced_strategy:
            primary, fallbacks = advanced_strategy
            self.logger.info(f"Selected advanced strategy: {primary}")
            return primary, fallbacks

        # Priority 2: Format-Specific Chunkers
        format_strategy = self._select_format_specific_strategy(file_extension, file_size)

        if format_strategy:
            primary, fallbacks = format_strategy
            self.logger.info(f"Selected format-specific strategy: {primary}")
            return primary, fallbacks

        # Priority 3: Content-Type Specific Strategies
        content_strategy = self._select_content_type_strategy(
            modality, text_ratio, has_structure, file_size
        )

        if content_strategy:
            primary, fallbacks = content_strategy
            self.logger.info(f"Selected content-type strategy: {primary}")
            return primary, fallbacks

        # Priority 4: Fallback to Traditional Auto-Selection
        return super()._auto_select_strategy(content_info)

    def _select_advanced_strategy(
        self,
        file_size: int,
        content_complexity: str,
        text_ratio: float,
        estimated_entropy: float,
        has_structure: bool
    ) -> Optional[Tuple[str, List[str]]]:
        """Select advanced general-purpose chunkers based on content characteristics."""

        # Adaptive Dynamic Chunker - Best for learning and optimization scenarios
        if self._should_use_adaptive(file_size, content_complexity, text_ratio):
            return "adaptive", ["context_enriched", "fastcdc", "paragraph"]

        # Context-Enriched Chunker - Best for semantic content analysis
        if self._should_use_context_enriched(text_ratio, has_structure, file_size):
            return "context_enriched", ["adaptive", "paragraph", "sentence"]

        # FastCDC Chunker - Best for large files and deduplication
        if self._should_use_fastcdc(file_size, estimated_entropy, text_ratio):
            return "fastcdc", ["adaptive", "fixed_size", "paragraph"]

        return None

    def _should_use_adaptive(self, file_size: int, content_complexity: str, text_ratio: float) -> bool:
        """Determine if Adaptive Dynamic Chunker should be used."""
        # Use adaptive for complex content that would benefit from learning
        if content_complexity in ["high", "very_high"]:
            return True

        # Use adaptive for medium-sized files where optimization is beneficial
        if 10_000 <= file_size <= 1_000_000 and text_ratio > 0.7:
            return True

        # Use adaptive for mixed content types
        if 0.3 <= text_ratio <= 0.9:
            return True

        return False

    def _should_use_context_enriched(self, text_ratio: float, has_structure: bool, file_size: int) -> bool:
        """Determine if Context-Enriched Chunker should be used."""
        # Use context-enriched for highly textual content
        if text_ratio > 0.8 and has_structure:
            return True

        # Use context-enriched for academic/technical documents
        if text_ratio > 0.9 and 5_000 <= file_size <= 500_000:
            return True

        # Use context-enriched for narrative content
        if text_ratio > 0.95 and file_size > 2_000:
            return True

        return False

    def _should_use_fastcdc(self, file_size: int, estimated_entropy: float, text_ratio: float) -> bool:
        """Determine if FastCDC Chunker should be used."""
        # Use FastCDC for large files
        if file_size > 1_000_000:
            return True

        # Use FastCDC for high-entropy content (good for deduplication)
        if estimated_entropy > 6.0:
            return True

        # Use FastCDC for binary or mixed content
        if text_ratio < 0.5:
            return True

        # Use FastCDC for medium-sized structured files
        if 100_000 <= file_size <= 1_000_000 and text_ratio < 0.8:
            return True

        return False

    def _select_format_specific_strategy(self, file_extension: str, file_size: int) -> Optional[Tuple[str, List[str]]]:
        """Select format-specific chunkers with enhanced fallback chains."""

        # Enhanced format-specific mapping with advanced fallbacks
        enhanced_format_map = {
            # Document formats - prioritize semantic understanding
            ".txt": ("context_enriched", ["adaptive", "sentence", "paragraph"]),
            ".md": ("markdown", ["context_enriched", "adaptive", "paragraph"]),
            ".markdown": ("markdown", ["context_enriched", "adaptive", "paragraph"]),

            # Code formats - prioritize AST-based chunking
            ".py": ("python", ["adaptive", "context_enriched", "paragraph"]),
            ".js": ("javascript", ["adaptive", "context_enriched", "paragraph"]),
            ".jsx": ("javascript", ["adaptive", "context_enriched", "paragraph"]),
            ".ts": ("typescript", ["adaptive", "context_enriched", "paragraph"]),
            ".tsx": ("typescript", ["adaptive", "context_enriched", "paragraph"]),
            ".java": ("java", ["adaptive", "context_enriched", "paragraph"]),
            ".go": ("go", ["adaptive", "context_enriched", "paragraph"]),
            ".css": ("css", ["adaptive", "context_enriched", "paragraph"]),
            ".scss": ("scss", ["adaptive", "context_enriched", "paragraph"]),
            ".c": ("c_cpp", ["adaptive", "context_enriched", "paragraph"]),
            ".cpp": ("c_cpp", ["adaptive", "context_enriched", "paragraph"]),

            # Data formats - prioritize structure-aware chunking
            ".json": ("json", ["fastcdc", "adaptive", "fixed_size"]),
            ".xml": ("xml", ["fastcdc", "adaptive", "fixed_size"]),
            ".html": ("html", ["context_enriched", "adaptive", "paragraph"]),
            ".csv": ("csv", ["fastcdc", "adaptive", "fixed_size"]),

            # Document processing formats
            ".pdf": ("pdf", ["context_enriched", "adaptive", "paragraph"]),
            ".doc": ("doc", ["context_enriched", "adaptive", "paragraph"]),
            ".docx": ("doc", ["context_enriched", "adaptive", "paragraph"]),

            # Large file optimization
            ".log": ("fastcdc", ["adaptive", "fixed_size", "sentence"]),
        }

        if file_extension in enhanced_format_map:
            primary, fallbacks = enhanced_format_map[file_extension]

            # Adjust for file size
            if file_size > 10_000_000:  # > 10MB
                # Prioritize FastCDC for very large files
                if primary != "fastcdc":
                    fallbacks = [primary] + [f for f in fallbacks if f != "fastcdc"]
                    primary = "fastcdc"

            return primary, fallbacks

        return None

    def _select_content_type_strategy(
        self,
        modality: ModalityType,
        text_ratio: float,
        has_structure: bool,
        file_size: int
    ) -> Optional[Tuple[str, List[str]]]:
        """Select strategy based on content type characteristics."""

        # Highly textual content with structure
        if modality == ModalityType.TEXT and text_ratio > 0.9 and has_structure:
            return "context_enriched", ["adaptive", "paragraph", "sentence"]

        # Mixed content
        if modality == ModalityType.MIXED:
            if file_size > 1_000_000:
                return "fastcdc", ["adaptive", "context_enriched", "fixed_size"]
            else:
                return "adaptive", ["context_enriched", "fastcdc", "paragraph"]

        # Large textual content
        if modality == ModalityType.TEXT and file_size > 500_000:
            return "fastcdc", ["context_enriched", "adaptive", "paragraph"]

        # Structured textual content
        if modality == ModalityType.TEXT and has_structure and text_ratio > 0.8:
            return "context_enriched", ["adaptive", "paragraph", "sentence"]

        # Default textual content
        if modality == ModalityType.TEXT and text_ratio > 0.7:
            return "adaptive", ["context_enriched", "paragraph", "sentence"]

        return None

    def _auto_select_strategy(self, content_info: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Override base auto-selection with enhanced logic."""
        if self.enable_advanced_selection:
            return self._enhanced_auto_select_strategy(content_info)
        else:
            return super()._auto_select_strategy(content_info)

    def _analyze_content_deeply(self, content: Union[str, bytes], file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Perform deep content analysis for enhanced strategy selection."""
        if not self.enable_content_analysis:
            return {}

        analysis = {}

        try:
            # Basic content analysis
            if isinstance(content, bytes):
                try:
                    text_content = content.decode('utf-8', errors='ignore')
                except:
                    text_content = str(content)
            else:
                text_content = content

            # Calculate text characteristics
            content_length = len(text_content)
            if content_length > 0:
                # Text ratio (printable characters)
                printable_chars = sum(1 for c in text_content if c.isprintable() or c.isspace())
                analysis["text_ratio"] = printable_chars / content_length

                # Estimate entropy (simplified)
                char_freq = {}
                for char in text_content:
                    char_freq[char] = char_freq.get(char, 0) + 1

                import math
                entropy = 0
                for freq in char_freq.values():
                    p = freq / content_length
                    entropy -= p * math.log2(p)
                analysis["estimated_entropy"] = entropy

                # Structure detection (simplified)
                structure_indicators = [
                    '\n\n',  # Paragraphs
                    '# ',    # Headers
                    '## ',   # Subheaders
                    '{',     # JSON/Code blocks
                    '<',     # XML/HTML tags
                    'def ',  # Function definitions
                    'class ', # Class definitions
                ]
                structure_count = sum(text_content.count(indicator) for indicator in structure_indicators)
                analysis["has_structure"] = structure_count > (content_length / 1000)  # Heuristic

                # Complexity assessment
                unique_chars = len(set(text_content))
                complexity_ratio = unique_chars / min(content_length, 1000)

                if complexity_ratio > 0.1:
                    analysis["complexity"] = "high"
                elif complexity_ratio > 0.05:
                    analysis["complexity"] = "medium"
                else:
                    analysis["complexity"] = "low"

        except Exception as e:
            self.logger.warning(f"Deep content analysis failed: {e}")

        return analysis

    def chunk_content(
        self,
        content: Union[str, bytes],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """Enhanced content chunking with deep analysis."""

        # Perform deep content analysis if enabled
        if self.enable_content_analysis:
            deep_analysis = self._analyze_content_deeply(content)
            if source_info is None:
                source_info = {}
            source_info.update(deep_analysis)

        # Use enhanced selection logic
        return super().chunk_content(content, source_info, **kwargs)

    def _analyze_file_characteristics(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze file characteristics for enhanced strategy selection.

        Returns:
            Dictionary with file characteristics including size, content analysis, etc.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Basic file info
        file_stat = file_path.stat()
        characteristics = {
            "file_path": str(file_path),
            "file_size": file_stat.st_size,
            "file_extension": file_path.suffix.lower(),
            "modality": ModalityType.TEXT  # Default assumption
        }

        # Content-based analysis if enabled
        if self.enable_content_analysis and file_stat.st_size > 0:
            try:
                # Read sample for analysis
                sample_size = min(10000, file_stat.st_size)
                with open(file_path, 'rb') as f:
                    sample_content = f.read(sample_size)

                # Perform deep content analysis
                content_analysis = self._analyze_content_deeply(sample_content, file_path)
                characteristics.update(content_analysis)

            except Exception as e:
                self.logger.warning(f"Content analysis failed for {file_path}: {e}")
                # Set defaults for failed analysis
                characteristics.update({
                    "text_ratio": 0.8,
                    "complexity": "medium",
                    "has_structure": False,
                    "estimated_entropy": 4.0
                })
        else:
            # Set defaults when content analysis is disabled
            characteristics.update({
                "text_ratio": 0.8,
                "complexity": "medium",
                "has_structure": False,
                "estimated_entropy": 4.0
            })

        return characteristics

    def chunk_file(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> ChunkingResult:
        """Enhanced file chunking with deep analysis."""

        file_path = Path(file_path)

        # Enhanced file analysis
        enhanced_info = {}

        if self.enable_content_analysis and file_path.exists():
            try:
                # Read sample of file for analysis
                sample_size = min(10000, file_path.stat().st_size)
                with open(file_path, 'rb') as f:
                    sample_content = f.read(sample_size)

                deep_analysis = self._analyze_content_deeply(sample_content, file_path)
                enhanced_info.update(deep_analysis)

            except Exception as e:
                self.logger.warning(f"Enhanced file analysis failed for {file_path}: {e}")

        # Add enhanced info to analysis
        original_analyze = self._analyze_file

        def enhanced_analyze(path):
            info = original_analyze(path)
            info.update(enhanced_info)
            return info

        self._analyze_file = enhanced_analyze

        try:
            result = super().chunk_file(file_path, **kwargs)
            return result
        finally:
            self._analyze_file = original_analyze

    def get_strategy_recommendations(
        self,
        content_info: Dict[str, Any]
    ) -> List[Tuple[str, float, str]]:
        """
        Get ranked strategy recommendations with confidence scores.

        Returns:
            List of (strategy_name, confidence_score, reason) tuples
        """
        recommendations = []

        file_size = content_info.get("file_size", 0)
        text_ratio = content_info.get("text_ratio", 1.0)
        complexity = content_info.get("complexity", "medium")
        has_structure = content_info.get("has_structure", False)
        estimated_entropy = content_info.get("estimated_entropy", 4.0)

        # Evaluate each advanced strategy

        # Adaptive Dynamic Chunker
        adaptive_score = 0.5  # Base score
        if complexity in ["high", "very_high"]:
            adaptive_score += 0.3
        if 10_000 <= file_size <= 1_000_000:
            adaptive_score += 0.2
        if 0.3 <= text_ratio <= 0.9:
            adaptive_score += 0.2
        recommendations.append(("adaptive", adaptive_score, "Intelligent self-tuning for complex content"))

        # Context-Enriched Chunker
        context_score = 0.4  # Base score
        if text_ratio > 0.8 and has_structure:
            context_score += 0.4
        if text_ratio > 0.9 and 5_000 <= file_size <= 500_000:
            context_score += 0.3
        if text_ratio > 0.95:
            context_score += 0.2
        recommendations.append(("context_enriched", context_score, "Semantic boundary detection for textual content"))

        # FastCDC Chunker
        fastcdc_score = 0.3  # Base score
        if file_size > 1_000_000:
            fastcdc_score += 0.4
        if estimated_entropy > 6.0:
            fastcdc_score += 0.3
        if text_ratio < 0.5:
            fastcdc_score += 0.3
        if 100_000 <= file_size <= 1_000_000 and text_ratio < 0.8:
            fastcdc_score += 0.2
        recommendations.append(("fastcdc", fastcdc_score, "Content-defined chunking for large/binary files"))

        # Sort by confidence score
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations

    def explain_strategy_selection(self, content_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide detailed explanation of strategy selection reasoning.

        Returns:
            Dictionary with selection reasoning and alternatives
        """
        primary, fallbacks = self._enhanced_auto_select_strategy(content_info)
        recommendations = self.get_strategy_recommendations(content_info)

        explanation = {
            "selected_strategy": primary,
            "fallback_strategies": fallbacks,
            "selection_reasoning": self._get_selection_reasoning(primary, content_info),
            "all_recommendations": recommendations,
            "content_characteristics": {
                "file_size": content_info.get("file_size", 0),
                "text_ratio": content_info.get("text_ratio", 1.0),
                "complexity": content_info.get("complexity", "medium"),
                "has_structure": content_info.get("has_structure", False),
                "estimated_entropy": content_info.get("estimated_entropy", 4.0),
                "modality": str(content_info.get("modality", ModalityType.TEXT))
            }
        }

        return explanation

    def _get_selection_reasoning(self, strategy: str, content_info: Dict[str, Any]) -> str:
        """Generate human-readable explanation for strategy selection."""
        file_size = content_info.get("file_size", 0)
        text_ratio = content_info.get("text_ratio", 1.0)
        complexity = content_info.get("complexity", "medium")

        if strategy == "adaptive":
            return f"Selected Adaptive chunker due to {complexity} complexity content and optimal size ({file_size:,} bytes) for learning optimization"
        elif strategy == "context_enriched":
            return f"Selected Context-Enriched chunker due to high text ratio ({text_ratio:.2f}) and structured content suitable for semantic analysis"
        elif strategy == "fastcdc":
            return f"Selected FastCDC chunker due to large file size ({file_size:,} bytes) or binary content optimal for content-defined boundaries"
        else:
            return f"Selected {strategy} chunker based on file format and content characteristics"
