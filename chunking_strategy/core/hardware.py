"""
Hardware detection and optimization module.

Provides automatic detection of CPU cores, memory, and GPU availability
to optimize chunking performance and batch processing.
"""

import logging
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Optional hardware detection dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # For safe access in recommendations


@dataclass
class HardwareInfo:
    """Comprehensive hardware information."""

    # CPU Information
    cpu_count: int
    cpu_count_physical: Optional[int]
    cpu_freq: Optional[float]
    cpu_usage: Optional[float]

    # Memory Information
    memory_total_gb: Optional[float]
    memory_available_gb: Optional[float]
    memory_usage_percent: Optional[float]

    # GPU Information
    gpu_count: int
    gpu_names: List[str]
    gpu_memory_total: List[float]  # GB
    gpu_memory_free: List[float]   # GB
    gpu_utilization: List[float]   # Percentage

    # Platform Information
    platform: str
    architecture: str
    python_version: str

    # Recommendations
    recommended_batch_size: int
    recommended_workers: int
    use_gpu: bool


class HardwareDetector:
    """Detects and analyzes available hardware for optimal chunking performance."""

    def __init__(self):
        """Initialize hardware detector."""
        self.logger = logging.getLogger(__name__)

    def detect_hardware(self) -> HardwareInfo:
        """
        Detect all available hardware and provide optimization recommendations.

        Returns:
            HardwareInfo object with comprehensive hardware details
        """
        self.logger.info("Detecting hardware configuration...")

        # CPU Detection
        cpu_info = self._detect_cpu()

        # Memory Detection
        memory_info = self._detect_memory()

        # GPU Detection
        gpu_info = self._detect_gpu()

        # Platform Detection
        platform_info = self._detect_platform()

        # Generate recommendations
        recommendations = self._generate_recommendations(cpu_info, memory_info, gpu_info)

        hardware_info = HardwareInfo(
            # CPU
            cpu_count=cpu_info['logical_cores'],
            cpu_count_physical=cpu_info.get('physical_cores'),
            cpu_freq=cpu_info.get('frequency'),
            cpu_usage=cpu_info.get('usage'),

            # Memory
            memory_total_gb=memory_info.get('total_gb'),
            memory_available_gb=memory_info.get('available_gb'),
            memory_usage_percent=memory_info.get('usage_percent'),

            # GPU
            gpu_count=gpu_info['count'],
            gpu_names=gpu_info['names'],
            gpu_memory_total=gpu_info['memory_total'],
            gpu_memory_free=gpu_info['memory_free'],
            gpu_utilization=gpu_info['utilization'],

            # Platform
            platform=platform_info['system'],
            architecture=platform_info['architecture'],
            python_version=platform_info['python_version'],

            # Recommendations
            recommended_batch_size=recommendations['batch_size'],
            recommended_workers=recommendations['workers'],
            use_gpu=recommendations['use_gpu']
        )

        self.logger.info(f"Hardware detection completed: {cpu_info['logical_cores']} CPU cores, "
                        f"{gpu_info['count']} GPUs, {memory_info.get('total_gb', 0):.1f}GB RAM")

        return hardware_info

    def _detect_cpu(self) -> Dict:
        """Detect CPU information."""
        cpu_info = {
            'logical_cores': os.cpu_count() or 1,
            'physical_cores': None,
            'frequency': None,
            'usage': None
        }

        if HAS_PSUTIL:
            try:
                cpu_info['physical_cores'] = psutil.cpu_count(logical=False)
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    cpu_info['frequency'] = cpu_freq.current
                cpu_info['usage'] = psutil.cpu_percent(interval=0.1)
            except Exception as e:
                self.logger.warning(f"Error getting detailed CPU info: {e}")

        return cpu_info

    def _detect_memory(self) -> Dict:
        """Detect memory information."""
        memory_info = {}

        if HAS_PSUTIL:
            try:
                virtual_memory = psutil.virtual_memory()
                memory_info = {
                    'total_gb': virtual_memory.total / (1024**3),
                    'available_gb': virtual_memory.available / (1024**3),
                    'usage_percent': virtual_memory.percent
                }
            except Exception as e:
                self.logger.warning(f"Error getting memory info: {e}")

        return memory_info

    def _detect_gpu(self) -> Dict:
        """Detect GPU information."""
        gpu_info = {
            'count': 0,
            'names': [],
            'memory_total': [],
            'memory_free': [],
            'utilization': []
        }

        # Try NVIDIA GPUs first with GPUtil
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info['count'] += 1
                    gpu_info['names'].append(gpu.name)
                    gpu_info['memory_total'].append(gpu.memoryTotal / 1024)  # Convert MB to GB
                    gpu_info['memory_free'].append(gpu.memoryFree / 1024)
                    gpu_info['utilization'].append(gpu.load * 100)
            except Exception as e:
                self.logger.warning(f"Error detecting NVIDIA GPUs with GPUtil: {e}")

        # Try PyTorch GPU detection as fallback/supplement
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch_gpu_count = torch.cuda.device_count()
                if torch_gpu_count > gpu_info['count']:
                    # PyTorch detected more GPUs, update count
                    gpu_info['count'] = torch_gpu_count

                    # If GPUtil didn't work, use PyTorch info
                    if not gpu_info['names']:
                        for i in range(torch_gpu_count):
                            gpu_info['names'].append(torch.cuda.get_device_name(i))
                            # Get memory info
                            memory_info = torch.cuda.get_device_properties(i)
                            total_memory_gb = memory_info.total_memory / (1024**3)
                            gpu_info['memory_total'].append(total_memory_gb)

                            # Get current memory usage
                            torch.cuda.set_device(i)
                            allocated = torch.cuda.memory_allocated() / (1024**3)
                            free_memory = total_memory_gb - allocated
                            gpu_info['memory_free'].append(free_memory)
                            gpu_info['utilization'].append(0.0)  # PyTorch doesn't provide utilization
            except Exception as e:
                self.logger.warning(f"Error detecting GPUs with PyTorch: {e}")

        return gpu_info

    def _detect_platform(self) -> Dict:
        """Detect platform information."""
        return {
            'system': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version()
        }

    def _generate_recommendations(self, cpu_info: Dict, memory_info: Dict, gpu_info: Dict) -> Dict:
        """Generate optimization recommendations based on hardware."""
        cpu_cores = cpu_info['logical_cores']
        memory_gb = memory_info.get('total_gb', 8)  # Default to 8GB if unknown
        gpu_count = gpu_info['count']

        # Batch size recommendations
        if memory_gb >= 32:
            base_batch_size = 64
        elif memory_gb >= 16:
            base_batch_size = 32
        elif memory_gb >= 8:
            base_batch_size = 16
        else:
            base_batch_size = 8

        # Adjust for CPU cores
        batch_size = min(base_batch_size, max(4, cpu_cores * 2))

        # Worker recommendations (for multiprocessing)
        if cpu_cores >= 16:
            workers = min(cpu_cores // 2, 8)  # Don't use too many workers
        elif cpu_cores >= 8:
            workers = cpu_cores // 2
        elif cpu_cores >= 4:
            workers = max(2, cpu_cores - 1)  # Leave one core for main process
        else:
            workers = 1

        # GPU usage recommendation - only if CUDA is actually available
        cuda_available = (HAS_TORCH and torch is not None and
                         hasattr(torch, 'cuda') and torch.cuda.is_available())
        use_gpu = gpu_count > 0 and cuda_available

        return {
            'batch_size': batch_size,
            'workers': workers,
            'use_gpu': use_gpu
        }

    def get_optimal_batch_config(
        self,
        total_files: int,
        avg_file_size_mb: float = 1.0,
        user_batch_size: Optional[int] = None,
        user_workers: Optional[int] = None,
        force_cpu: bool = False
    ) -> Dict:
        """
        Get optimal batch processing configuration.

        Args:
            total_files: Total number of files to process
            avg_file_size_mb: Average file size in MB
            user_batch_size: User-specified batch size override
            user_workers: User-specified worker count override
            force_cpu: Force CPU-only processing

        Returns:
            Dictionary with optimal batch configuration
        """
        hardware = self.detect_hardware()

        # Use user overrides if provided
        batch_size = user_batch_size or hardware.recommended_batch_size
        workers = user_workers or hardware.recommended_workers
        use_gpu = hardware.use_gpu and not force_cpu

        # Adjust batch size based on file size
        if avg_file_size_mb > 10:  # Large files
            batch_size = max(1, batch_size // 4)
        elif avg_file_size_mb > 5:  # Medium files
            batch_size = max(1, batch_size // 2)

        # Ensure we don't create more batches than necessary
        if total_files < batch_size:
            batch_size = total_files
            workers = min(workers, total_files)

        # Calculate number of batches
        num_batches = (total_files + batch_size - 1) // batch_size

        return {
            'batch_size': batch_size,
            'num_batches': num_batches,
            'workers': workers,
            'use_gpu': use_gpu,
            'hardware_info': hardware,
            'estimated_memory_per_worker_mb': avg_file_size_mb * batch_size,
            'total_estimated_memory_mb': avg_file_size_mb * batch_size * workers
        }


# Global hardware detector instance
_hardware_detector = None

def get_hardware_info() -> HardwareInfo:
    """Get cached hardware information."""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector.detect_hardware()

def get_optimal_batch_config(**kwargs) -> Dict:
    """Get optimal batch configuration with given parameters."""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector.get_optimal_batch_config(**kwargs)


@dataclass
class SmartParallelizationConfig:
    """
    Configuration for smart parallelization decisions.

    This class determines when parallelization should be enabled based on
    workload characteristics to avoid performance regressions from overhead.
    """

    # File size thresholds (in bytes)
    min_file_size_for_hw_optimization: int = 50000  # 50KB+ files benefit from HW optimization
    min_total_size_for_threading: int = 100000       # 100KB+ total work benefits from threading
    min_files_for_batch: int = 10                    # 10+ files benefit from batch processing
    min_strategies_for_parallel: int = 3             # 3+ strategies benefit from parallel processing

    # Streaming thresholds (in bytes)
    min_file_size_for_streaming: int = 100 * 1024 * 1024  # 100MB+ files should use streaming
    streaming_block_size: int = 64 * 1024 * 1024           # 64MB default block size for streaming
    streaming_overlap_size: int = 1024 * 1024              # 1MB overlap for streaming chunks

    # Performance thresholds
    max_hw_detection_overhead: float = 0.2           # Don't spend >200ms on HW detection
    max_thread_setup_overhead: float = 0.3          # Don't spend >300ms on thread setup

    # Cache hardware info to avoid repeated detection
    _cached_hardware_info: Optional[Dict] = field(default=None, init=False)
    _hardware_detection_time: Optional[float] = field(default=None, init=False)

    def should_use_hardware_optimization(self, file_size: int) -> bool:
        """Decide if hardware optimization should be used based on file size."""
        return file_size >= self.min_file_size_for_hw_optimization

    def should_use_threading(self, total_size: int, num_files: int) -> bool:
        """Decide if threading should be used based on workload size."""
        return (total_size >= self.min_total_size_for_threading or
                num_files >= self.min_files_for_batch)

    def should_use_parallel_strategies(self, num_strategies: int, file_size: int) -> bool:
        """Decide if parallel strategy processing should be used."""
        return (num_strategies >= self.min_strategies_for_parallel and
                file_size >= self.min_file_size_for_hw_optimization)

    def should_use_streaming(self, file_size: int) -> bool:
        """Decide if streaming should be used based on file size."""
        return file_size >= self.min_file_size_for_streaming

    def get_streaming_config(self) -> Dict[str, int]:
        """Get streaming configuration parameters."""
        return {
            'block_size': self.streaming_block_size,
            'overlap_size': self.streaming_overlap_size,
            'min_file_size': self.min_file_size_for_streaming
        }

    def get_cached_hardware_info(self) -> Dict:
        """Get cached hardware info or detect if not cached."""
        if self._cached_hardware_info is None:
            start_time = time.time()
            detector = HardwareDetector()
            self._cached_hardware_info = detector.detect_hardware()
            self._hardware_detection_time = time.time() - start_time
        return self._cached_hardware_info


# Global smart parallelization configuration instance
_smart_config = SmartParallelizationConfig()

def get_smart_parallelization_config() -> SmartParallelizationConfig:
    """Get the global smart parallelization configuration instance."""
    return _smart_config

def configure_smart_parallelization(**kwargs) -> None:
    """Configure smart parallelization thresholds."""
    global _smart_config
    for key, value in kwargs.items():
        if hasattr(_smart_config, key):
            setattr(_smart_config, key, value)
