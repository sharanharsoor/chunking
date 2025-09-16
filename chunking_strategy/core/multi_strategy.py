"""
Multi-strategy parallel processing module.

This module enables processing the same file with multiple chunking strategies
simultaneously, utilizing available hardware resources for optimal performance.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from chunking_strategy.core.base import ChunkingResult
from chunking_strategy.core.hardware import HardwareDetector, get_optimal_batch_config, get_smart_parallelization_config
from chunking_strategy.orchestrator import ChunkerOrchestrator

logger = logging.getLogger(__name__)

# Strategy name mapping for backward compatibility
STRATEGY_NAME_MAPPING = {
    'sentence': 'sentence_based',
    'paragraph': 'paragraph_based',
    'python': 'python_code',
    'c_cpp': 'c_cpp_code',
    'pdf': 'pdf_chunker'
}

@dataclass
class MultiStrategyResult:
    """Result from multi-strategy processing."""

    file_path: Path
    strategy_results: Dict[str, ChunkingResult]  # strategy_name -> result
    successful_strategies: List[str]
    failed_strategies: List[Tuple[str, str]]  # (strategy_name, error_message)
    total_processing_time: float
    hardware_config: Dict[str, Any]

    @property
    def total_chunks(self) -> int:
        """Total chunks across all successful strategies."""
        return sum(len(result.chunks) for result in self.strategy_results.values())

    @property
    def success_rate(self) -> float:
        """Percentage of strategies that succeeded."""
        total_strategies = len(self.successful_strategies) + len(self.failed_strategies)
        return len(self.successful_strategies) / total_strategies if total_strategies > 0 else 0.0


class MultiStrategyProcessor:
    """Processes files with multiple chunking strategies in parallel."""

    def __init__(
        self,
        orchestrator_config: Optional[Dict[str, Any]] = None,
        enable_hardware_optimization: bool = True,
        enable_smart_parallelization: bool = True
    ):
        """
        Initialize multi-strategy processor.

        Args:
            orchestrator_config: Configuration for the orchestrator
            enable_hardware_optimization: Enable hardware-based optimizations
            enable_smart_parallelization: Enable smart parallelization decisions (recommended)
        """
        self.logger = logging.getLogger(__name__)
        self.orchestrator_config = orchestrator_config or {}
        self.enable_hardware_optimization = enable_hardware_optimization
        self.enable_smart_parallelization = enable_smart_parallelization
        self.smart_config = get_smart_parallelization_config() if enable_smart_parallelization else None

        if enable_hardware_optimization:
            self.hardware_detector = HardwareDetector()
            # Use cached hardware info if smart parallelization is enabled
            if enable_smart_parallelization and self.smart_config:
                self.hardware_info = self.smart_config.get_cached_hardware_info()
            else:
                self.hardware_info = self.hardware_detector.detect_hardware()
        else:
            self.hardware_detector = None
            self.hardware_info = None

    def _normalize_strategy_name(self, strategy_name: str) -> str:
        """Normalize strategy name using backward compatibility mapping."""
        return STRATEGY_NAME_MAPPING.get(strategy_name, strategy_name)

    def _normalize_strategy_list(self, strategies: List[str]) -> List[str]:
        """Normalize a list of strategy names."""
        return [self._normalize_strategy_name(strategy) for strategy in strategies]

    def process_file_with_strategies(
        self,
        file_path: Union[str, Path],
        strategies: List[str],
        strategy_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        parallel_mode: str = "auto",  # "auto", "thread", "process", "sequential"
        max_workers: Optional[int] = None
    ) -> MultiStrategyResult:
        """
        Process a single file with multiple strategies in parallel.

        Args:
            file_path: Path to the file to process
            strategies: List of strategy names to use
            strategy_configs: Strategy-specific configurations
            parallel_mode: Parallelization mode
            max_workers: Maximum number of workers (auto-detected if None)

        Returns:
            MultiStrategyResult with results from all strategies
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        start_time = time.time()
        strategy_configs = strategy_configs or {}

        # Normalize strategy names for backward compatibility
        strategies = self._normalize_strategy_list(strategies)

        # Determine optimal processing configuration
        processing_config = self._get_processing_config(
            strategies=strategies,
            file_size_mb=file_path.stat().st_size / (1024 * 1024),
            parallel_mode=parallel_mode,
            max_workers=max_workers
        )

        self.logger.info(f"Processing {file_path} with {len(strategies)} strategies: "
                        f"mode={processing_config['mode']}, workers={processing_config['workers']}")

        # Process strategies
        if processing_config['mode'] == 'sequential':
            result = self._process_sequential(file_path, strategies, strategy_configs)
        elif processing_config['mode'] == 'thread':
            result = self._process_threaded(file_path, strategies, strategy_configs, processing_config)
        else:  # process
            result = self._process_multiprocess(file_path, strategies, strategy_configs, processing_config)

        # Finalize result
        result.total_processing_time = time.time() - start_time
        result.hardware_config = processing_config

        self.logger.info(f"Multi-strategy processing completed: {len(result.successful_strategies)}/{len(strategies)} "
                        f"strategies succeeded in {result.total_processing_time:.2f}s")

        return result

    def process_multiple_files_with_strategies(
        self,
        files: List[Union[str, Path]],
        strategies: List[str],
        strategy_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        parallel_mode: str = "auto",
        max_workers: Optional[int] = None
    ) -> List[MultiStrategyResult]:
        """
        Process multiple files with multiple strategies.

        This creates a matrix of processing: each file is processed with each strategy.
        Parallelization occurs at both the file level and strategy level.

        Args:
            files: List of file paths to process
            strategies: List of strategy names to use
            strategy_configs: Strategy-specific configurations
            parallel_mode: Parallelization mode
            max_workers: Maximum number of workers

        Returns:
            List of MultiStrategyResult objects
        """
        results = []

        # Process each file with all strategies
        for file_path in files:
            try:
                result = self.process_file_with_strategies(
                    file_path=file_path,
                    strategies=strategies,
                    strategy_configs=strategy_configs,
                    parallel_mode=parallel_mode,
                    max_workers=max_workers
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                # Create an error result
                error_result = MultiStrategyResult(
                    file_path=Path(file_path),
                    strategy_results={},
                    successful_strategies=[],
                    failed_strategies=[(str(file_path), str(e))],
                    total_processing_time=0.0,
                    hardware_config={}
                )
                results.append(error_result)

        return results

    def _get_processing_config(
        self,
        strategies: List[str],
        file_size_mb: float,
        parallel_mode: str = "auto",
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Determine optimal processing configuration."""

        if parallel_mode != "auto":
            mode = parallel_mode
        else:
            # Smart auto-selection based on workload characteristics
            file_size_bytes = int(file_size_mb * 1024 * 1024)

            if not self.enable_hardware_optimization or not self.hardware_info:
                mode = "sequential"
            elif self.enable_smart_parallelization and self.smart_config:
                # Use smart parallelization decisions
                should_parallelize = self.smart_config.should_use_parallel_strategies(
                    len(strategies), file_size_bytes
                )
                if not should_parallelize:
                    mode = "sequential"
                    self.logger.debug(f"Smart parallelization: Using sequential processing "
                                    f"(strategies={len(strategies)}, file_size={file_size_bytes})")
                elif file_size_mb > 50:  # Large files benefit from process-based parallelism
                    mode = "process"
                else:
                    mode = "thread"
            else:
                # Fallback to original logic
                if len(strategies) <= 2:
                    mode = "sequential"  # Not worth parallelizing few strategies
                elif file_size_mb > 50:  # Large files benefit from process-based parallelism
                    mode = "process"
                else:
                    mode = "thread"

        # Determine worker count
        if max_workers:
            workers = max_workers
        elif self.hardware_info:
            # Use hardware recommendations but limit by number of strategies
            recommended_workers = min(self.hardware_info.recommended_workers, len(strategies))
            workers = max(1, recommended_workers)
        else:
            workers = min(4, len(strategies))  # Conservative default

        return {
            'mode': mode,
            'workers': workers,
            'strategies_count': len(strategies),
            'file_size_mb': file_size_mb,
            'hardware_optimized': self.enable_hardware_optimization,
            'smart_parallelization': self.enable_smart_parallelization
        }

    def _process_sequential(
        self,
        file_path: Path,
        strategies: List[str],
        strategy_configs: Dict[str, Dict[str, Any]]
    ) -> MultiStrategyResult:
        """Process strategies sequentially."""
        results = {}
        successful = []
        failed = []

        for strategy in strategies:
            try:
                result = self._process_single_strategy(file_path, strategy, strategy_configs.get(strategy, {}))
                results[strategy] = result
                successful.append(strategy)
            except Exception as e:
                self.logger.error(f"Strategy {strategy} failed for {file_path}: {e}")
                failed.append((strategy, str(e)))

        return MultiStrategyResult(
            file_path=file_path,
            strategy_results=results,
            successful_strategies=successful,
            failed_strategies=failed,
            total_processing_time=0.0,  # Will be set by caller
            hardware_config={}
        )

    def _process_threaded(
        self,
        file_path: Path,
        strategies: List[str],
        strategy_configs: Dict[str, Dict[str, Any]],
        processing_config: Dict[str, Any]
    ) -> MultiStrategyResult:
        """Process strategies using thread pool."""
        results = {}
        successful = []
        failed = []
        workers = processing_config['workers']

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all strategy tasks
            future_to_strategy = {
                executor.submit(
                    self._process_single_strategy,
                    file_path,
                    strategy,
                    strategy_configs.get(strategy, {})
                ): strategy
                for strategy in strategies
            }

            # Collect results
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    results[strategy] = result
                    successful.append(strategy)
                except Exception as e:
                    self.logger.error(f"Strategy {strategy} failed for {file_path}: {e}")
                    failed.append((strategy, str(e)))

        return MultiStrategyResult(
            file_path=file_path,
            strategy_results=results,
            successful_strategies=successful,
            failed_strategies=failed,
            total_processing_time=0.0,  # Will be set by caller
            hardware_config=processing_config
        )

    def _process_multiprocess(
        self,
        file_path: Path,
        strategies: List[str],
        strategy_configs: Dict[str, Dict[str, Any]],
        processing_config: Dict[str, Any]
    ) -> MultiStrategyResult:
        """Process strategies using process pool."""
        results = {}
        successful = []
        failed = []
        workers = processing_config['workers']

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all strategy tasks
            future_to_strategy = {
                executor.submit(
                    _process_strategy_worker,  # Separate function for pickling
                    str(file_path),
                    strategy,
                    strategy_configs.get(strategy, {}),
                    self.orchestrator_config
                ): strategy
                for strategy in strategies
            }

            # Collect results
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    results[strategy] = result
                    successful.append(strategy)
                except Exception as e:
                    self.logger.error(f"Strategy {strategy} failed for {file_path}: {e}")
                    failed.append((strategy, str(e)))

        return MultiStrategyResult(
            file_path=file_path,
            strategy_results=results,
            successful_strategies=successful,
            failed_strategies=failed,
            total_processing_time=0.0,  # Will be set by caller
            hardware_config=processing_config
        )

    def _process_single_strategy(
        self,
        file_path: Path,
        strategy: str,
        strategy_config: Dict[str, Any]
    ) -> ChunkingResult:
        """Process a single strategy for a file."""
        from chunking_strategy.core.registry import list_chunkers

        # Check if strategy exists
        available_strategies = list_chunkers()
        if strategy not in available_strategies:
            raise ValueError(f"Strategy '{strategy}' not found. Available strategies: {available_strategies}")

        orchestrator = ChunkerOrchestrator(
            config=self.orchestrator_config,
            enable_hardware_optimization=False  # Already optimized at this level
        )

        return orchestrator.chunk_file(
            file_path=file_path,
            strategy_override=strategy,
            **strategy_config
        )


def _process_strategy_worker(
    file_path_str: str,
    strategy: str,
    strategy_config: Dict[str, Any],
    orchestrator_config: Dict[str, Any]
) -> ChunkingResult:
    """
    Worker function for multiprocessing.

    This needs to be a module-level function for pickle serialization.
    """
    orchestrator = ChunkerOrchestrator(
        config=orchestrator_config,
        enable_hardware_optimization=False
    )

    return orchestrator.chunk_file(
        file_path=Path(file_path_str),
        strategy_override=strategy,
        **strategy_config
    )
