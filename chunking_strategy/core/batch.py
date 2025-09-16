"""
Batch processing module for efficient multi-file chunking.

Provides high-performance batch processing capabilities with automatic
hardware optimization, progress tracking, and error handling.
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple

from chunking_strategy.core.base import ChunkingResult
from chunking_strategy.core.hardware import get_optimal_batch_config
from chunking_strategy.core.registry import create_chunker


@dataclass
class BatchFile:
    """Represents a file to be processed in batch."""
    path: Path
    size_mb: float
    chunker_strategy: str
    chunker_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result from batch processing operation."""

    # Results
    successful_files: List[Path] = field(default_factory=list)
    failed_files: List[Tuple[Path, str]] = field(default_factory=list)  # (path, error)
    chunk_results: Dict[str, ChunkingResult] = field(default_factory=dict)  # path -> result

    # Statistics
    total_files: int = 0
    total_chunks: int = 0
    total_processing_time: float = 0.0
    total_file_size_mb: float = 0.0

    # Performance metrics
    files_per_second: float = 0.0
    chunks_per_second: float = 0.0
    mb_per_second: float = 0.0

    # Hardware utilization
    hardware_config: Optional[Dict[str, Any]] = None
    peak_memory_usage_mb: Optional[float] = None
    cpu_utilization: Optional[float] = None

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.total_processing_time > 0:
            self.files_per_second = self.total_files / self.total_processing_time
            self.chunks_per_second = self.total_chunks / self.total_processing_time
            self.mb_per_second = self.total_file_size_mb / self.total_processing_time


class BatchProcessor:
    """High-performance batch processor for chunking multiple files."""

    def __init__(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        error_callback: Optional[Callable[[Path, Exception], None]] = None
    ):
        """
        Initialize batch processor.

        Args:
            progress_callback: Optional callback for progress updates (current, total, status)
            error_callback: Optional callback for error handling (file_path, error)
        """
        self.logger = logging.getLogger(__name__)
        self.progress_callback = progress_callback
        self.error_callback = error_callback

    def process_files(
        self,
        files: List[Union[Path, str, BatchFile]],
        default_strategy: str = "fixed_size",
        default_params: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        workers: Optional[int] = None,
        use_gpu: Optional[bool] = None,
        parallel_mode: str = "process"  # "process", "thread", or "sequential"
    ) -> BatchResult:
        """
        Process multiple files in batches with optimal hardware utilization.

        Args:
            files: List of files to process (paths or BatchFile objects)
            default_strategy: Default chunking strategy for files
            default_params: Default parameters for chunking strategy
            batch_size: Batch size override (auto-detected if None)
            workers: Number of workers override (auto-detected if None)
            use_gpu: Use GPU override (auto-detected if None)
            parallel_mode: Parallelization mode

        Returns:
            BatchResult with comprehensive processing results
        """
        start_time = time.time()

        # Prepare batch files
        batch_files = self._prepare_batch_files(files, default_strategy, default_params or {})

        if not batch_files:
            self.logger.warning("No valid files to process")
            return BatchResult()

        # Get optimal configuration
        avg_file_size = sum(bf.size_mb for bf in batch_files) / len(batch_files)
        config = get_optimal_batch_config(
            total_files=len(batch_files),
            avg_file_size_mb=avg_file_size,
            user_batch_size=batch_size,
            user_workers=workers,
            force_cpu=use_gpu is False
        )

        self.logger.info(f"Starting batch processing: {len(batch_files)} files, "
                        f"batch_size={config['batch_size']}, workers={config['workers']}, "
                        f"mode={parallel_mode}")

        # Process in batches
        if parallel_mode == "sequential":
            result = self._process_sequential(batch_files, config)
        elif parallel_mode == "thread":
            result = self._process_threaded(batch_files, config)
        else:  # process
            result = self._process_multiprocess(batch_files, config)

        # Calculate final metrics
        result.total_processing_time = time.time() - start_time
        result.hardware_config = config

        self.logger.info(f"Batch processing completed: {result.total_files} files, "
                        f"{result.total_chunks} chunks, {result.total_processing_time:.2f}s")

        return result

    def _prepare_batch_files(
        self,
        files: List[Union[Path, str, BatchFile]],
        default_strategy: str,
        default_params: Dict[str, Any]
    ) -> List[BatchFile]:
        """Prepare list of BatchFile objects from input."""
        batch_files = []

        for file_input in files:
            if isinstance(file_input, BatchFile):
                batch_files.append(file_input)
            else:
                path = Path(file_input)
                if path.exists() and path.is_file():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    batch_file = BatchFile(
                        path=path,
                        size_mb=size_mb,
                        chunker_strategy=default_strategy,
                        chunker_params=default_params.copy()
                    )
                    batch_files.append(batch_file)
                else:
                    self.logger.warning(f"File not found or not a file: {path}")

        return batch_files

    def _process_sequential(self, batch_files: List[BatchFile], config: Dict) -> BatchResult:
        """Process files sequentially (single-threaded)."""
        result = BatchResult(total_files=len(batch_files))

        for i, batch_file in enumerate(batch_files):
            if self.progress_callback:
                self.progress_callback(i + 1, len(batch_files), f"Processing {batch_file.path}")

            try:
                # Print file path before processing for better visibility
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"📄 Processing file: {batch_file.path}")
                logger.info(f"   Size: {batch_file.size_mb:.2f} MB | Strategy: {batch_file.chunker_strategy}")

                chunk_result = self._process_single_file(batch_file)
                result.successful_files.append(batch_file.path)
                result.chunk_results[str(batch_file.path)] = chunk_result
                result.total_chunks += len(chunk_result.chunks)
                result.total_file_size_mb += batch_file.size_mb

            except Exception as e:
                self.logger.error(f"Error processing {batch_file.path}: {e}")
                result.failed_files.append((batch_file.path, str(e)))
                if self.error_callback:
                    self.error_callback(batch_file.path, e)

        return result

    def _process_threaded(self, batch_files: List[BatchFile], config: Dict) -> BatchResult:
        """Process files using thread pool."""
        result = BatchResult(total_files=len(batch_files))
        workers = config['workers']

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, batch_file): batch_file
                for batch_file in batch_files
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_file):
                batch_file = future_to_file[future]
                completed += 1

                if self.progress_callback:
                    self.progress_callback(completed, len(batch_files), f"Completed {batch_file.path}")

                try:
                    # Log completion with full path
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"✅ Completed file: {batch_file.path}")

                    chunk_result = future.result()
                    result.successful_files.append(batch_file.path)
                    result.chunk_results[str(batch_file.path)] = chunk_result
                    result.total_chunks += len(chunk_result.chunks)
                    result.total_file_size_mb += batch_file.size_mb

                except Exception as e:
                    self.logger.error(f"Error processing {batch_file.path}: {e}")
                    result.failed_files.append((batch_file.path, str(e)))
                    if self.error_callback:
                        self.error_callback(batch_file.path, e)

        return result

    def _process_multiprocess(self, batch_files: List[BatchFile], config: Dict) -> BatchResult:
        """Process files using process pool for CPU-intensive tasks."""
        result = BatchResult(total_files=len(batch_files))
        workers = config['workers']

        # Create batches
        batch_size = config['batch_size']
        batches = [
            batch_files[i:i + batch_size]
            for i in range(0, len(batch_files), batch_size)
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit batch tasks
            future_to_batch = {
                executor.submit(_process_file_batch, batch): batch
                for batch in batches
            }

            # Process completed batches
            completed_files = 0
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]

                try:
                    batch_results = future.result()

                    for file_path, chunk_result, error in batch_results:
                        completed_files += 1

                        if self.progress_callback:
                            self.progress_callback(completed_files, len(batch_files), f"Completed {file_path}")

                        # Log completion with full path for multiprocess mode
                        import logging
                        logger = logging.getLogger(__name__)
                        if not error:
                            logger.info(f"✅ Completed file: {file_path}")
                        else:
                            logger.error(f"❌ Failed file: {file_path} - {error}")

                        if error:
                            result.failed_files.append((Path(file_path), error))
                            if self.error_callback:
                                self.error_callback(Path(file_path), Exception(error))
                        else:
                            result.successful_files.append(Path(file_path))
                            result.chunk_results[file_path] = chunk_result
                            result.total_chunks += len(chunk_result.chunks)

                            # Find file size
                            for bf in batch:
                                if str(bf.path) == file_path:
                                    result.total_file_size_mb += bf.size_mb
                                    break

                except Exception as e:
                    self.logger.error(f"Error processing batch: {e}")
                    for batch_file in batch:
                        result.failed_files.append((batch_file.path, str(e)))
                        if self.error_callback:
                            self.error_callback(batch_file.path, e)

        return result

    def _process_single_file(self, batch_file: BatchFile) -> ChunkingResult:
        """Process a single file."""
        chunker = create_chunker(batch_file.chunker_strategy, **batch_file.chunker_params)
        return chunker.chunk(batch_file.path)


def _process_file_batch(batch_files: List[BatchFile]) -> List[Tuple[str, Optional[ChunkingResult], Optional[str]]]:
    """
    Process a batch of files (used for multiprocessing).

    Returns:
        List of tuples: (file_path, chunk_result, error_message)
    """
    results = []

    for batch_file in batch_files:
        try:
            chunker = create_chunker(batch_file.chunker_strategy, **batch_file.chunker_params)
            chunk_result = chunker.chunk(batch_file.path)
            results.append((str(batch_file.path), chunk_result, None))
        except Exception as e:
            results.append((str(batch_file.path), None, str(e)))

    return results


def process_files_batch(
    files: List[Union[Path, str]],
    strategy: str = "fixed_size",
    strategy_params: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    workers: Optional[int] = None,
    parallel_mode: str = "process",
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> BatchResult:
    """
    Convenience function for batch processing files.

    Args:
        files: List of file paths to process
        strategy: Chunking strategy to use
        strategy_params: Parameters for the chunking strategy
        batch_size: Batch size (auto-detected if None)
        workers: Number of workers (auto-detected if None)
        parallel_mode: "process", "thread", or "sequential"
        progress_callback: Optional progress callback

    Returns:
        BatchResult with processing results
    """
    processor = BatchProcessor(progress_callback=progress_callback)
    return processor.process_files(
        files=files,
        default_strategy=strategy,
        default_params=strategy_params or {},
        batch_size=batch_size,
        workers=workers,
        parallel_mode=parallel_mode
    )
