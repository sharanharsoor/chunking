"""
Comprehensive logging configuration system for the chunking strategy library.

This module provides centralized logging configuration with support for:
- Regular user logs (minimal, essential information)
- Developer debug logs (detailed for troubleshooting)
- Log collection for bug reporting
- File and console output management
- Performance monitoring logs
"""

import logging
import logging.handlers
import sys
import os
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """Logging levels with user-friendly names."""
    SILENT = "silent"      # Only critical errors
    MINIMAL = "minimal"    # Basic status updates for users
    NORMAL = "normal"      # Standard logging for users
    VERBOSE = "verbose"    # Detailed logs for power users
    DEBUG = "debug"        # Full debugging information
    TRACE = "trace"        # Maximum verbosity for development


@dataclass
class LogConfig:
    """Configuration for logging behavior."""
    level: LogLevel = LogLevel.NORMAL
    console_output: bool = True
    file_output: bool = False
    log_file: Optional[Path] = None
    collect_performance: bool = True
    collect_metrics: bool = True
    format_json: bool = False
    include_module_names: bool = True
    max_file_size: str = "10MB"
    backup_count: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['level'] = self.level.value
        if self.log_file:
            result['log_file'] = str(self.log_file)
        return result


class ChunkingLogger:
    """Centralized logger for the chunking strategy library."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.config = LogConfig()
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_records: List[Dict[str, Any]] = []
        self.performance_logs: List[Dict[str, Any]] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._temp_dir: Optional[Path] = None

        # Set up root configuration
        self._configure_logging()

    def configure(self, config: Optional[LogConfig] = None, **kwargs) -> None:
        """
        Configure logging behavior.

        Args:
            config: LogConfig object with settings
            **kwargs: Individual config parameters
        """
        if config:
            self.config = config

        # Create new config if needed
        if not hasattr(self, 'config') or self.config is None:
            self.config = LogConfig()

        # Override with kwargs - create new config to handle dataclass immutability
        config_dict = {
            'level': self.config.level,
            'console_output': self.config.console_output,
            'file_output': self.config.file_output,
            'log_file': self.config.log_file,
            'collect_performance': self.config.collect_performance,
            'collect_metrics': self.config.collect_metrics,
            'format_json': self.config.format_json,
            'include_module_names': self.config.include_module_names,
            'max_file_size': self.config.max_file_size,
            'backup_count': self.config.backup_count
        }

        for key, value in kwargs.items():
            if key in config_dict:
                if key == 'level' and isinstance(value, str):
                    try:
                        value = LogLevel(value.lower())
                    except ValueError:
                        # Just skip invalid values
                        continue
                elif key == 'log_file' and value:
                    value = Path(value)
                config_dict[key] = value

        # Create new config object
        self.config = LogConfig(**config_dict)

        self._configure_logging()

    def _configure_logging(self) -> None:
        """Set up logging based on current configuration."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set root level based on our config
        root_level = self._get_python_log_level(self.config.level)
        root_logger.setLevel(root_level)

        # Create formatters
        if self.config.format_json:
            formatter = self._create_json_formatter()
        else:
            formatter = self._create_text_formatter()

        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(root_level)
            root_logger.addHandler(console_handler)

        # File handler
        if self.config.file_output and self.config.log_file:
            try:
                self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.handlers.RotatingFileHandler(
                    self.config.log_file,
                    maxBytes=self._parse_size(self.config.max_file_size),
                    backupCount=self.config.backup_count,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(root_level)
                root_logger.addHandler(file_handler)
            except Exception as e:
                # Fallback to console logging
                logging.warning(f"Failed to set up file logging: {e}")

        # Custom handler for log collection
        if self.config.level in [LogLevel.DEBUG, LogLevel.TRACE]:
            collection_handler = LogCollectionHandler(self)
            collection_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(collection_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for the specified module.

        Args:
            name: Module name (usually __name__)

        Returns:
            Configured logger instance
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger

        return self.loggers[name]

    def user_info(self, message: str, **kwargs) -> None:
        """Log user-facing informational message."""
        if self.config.level.value not in ['silent']:
            logger = self.get_logger('chunking_strategy.user')
            logger.info(f"📝 {message}", extra={'user_message': True, **kwargs})

    def user_success(self, message: str, **kwargs) -> None:
        """Log user-facing success message."""
        if self.config.level.value not in ['silent']:
            logger = self.get_logger('chunking_strategy.user')
            logger.info(f"✅ {message}", extra={'user_message': True, **kwargs})

    def user_warning(self, message: str, **kwargs) -> None:
        """Log user-facing warning message."""
        logger = self.get_logger('chunking_strategy.user')
        logger.warning(f"⚠️  {message}", extra={'user_message': True, **kwargs})

    def user_error(self, message: str, **kwargs) -> None:
        """Log user-facing error message."""
        logger = self.get_logger('chunking_strategy.user')
        logger.error(f"❌ {message}", extra={'user_message': True, **kwargs})

    def debug_operation(self, operation: str, details: Dict[str, Any], **kwargs) -> None:
        """Log detailed operation information for debugging."""
        if self.config.level in [LogLevel.DEBUG, LogLevel.TRACE]:
            logger = self.get_logger('chunking_strategy.debug')
            logger.debug(f"🔧 {operation}", extra={'operation': operation, 'details': details, **kwargs})

    def performance_log(self, operation: str, duration: float, **kwargs) -> None:
        """Log performance metrics."""
        if self.config.collect_performance:
            perf_data = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'duration_seconds': duration,
                'session_id': self.session_id,
                **kwargs
            }
            self.performance_logs.append(perf_data)

            if self.config.level in [LogLevel.VERBOSE, LogLevel.DEBUG, LogLevel.TRACE]:
                logger = self.get_logger('chunking_strategy.performance')
                logger.info(f"⏱️  {operation}: {duration:.3f}s", extra=perf_data)

    def metrics_log(self, metrics: Dict[str, Any], **kwargs) -> None:
        """Log quality and processing metrics."""
        if self.config.collect_metrics:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                'metrics': metrics,
                **kwargs
            }

            if self.config.level in [LogLevel.VERBOSE, LogLevel.DEBUG, LogLevel.TRACE]:
                logger = self.get_logger('chunking_strategy.metrics')
                logger.info(f"📊 Metrics collected", extra=metrics_data)

    def collect_debug_info(self) -> Path:
        """
        Collect all debug information into a zip file for bug reporting.

        Returns:
            Path to the created debug zip file
        """
        if not self._temp_dir:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="chunking_debug_"))

        debug_dir = self._temp_dir / f"debug_{self.session_id}"
        debug_dir.mkdir(exist_ok=True)

        # Collect system information
        system_info = self._collect_system_info()
        with open(debug_dir / "system_info.json", "w") as f:
            json.dump(system_info, f, indent=2)

        # Collect configuration
        with open(debug_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Collect recent log records
        if self.log_records:
            with open(debug_dir / "recent_logs.jsonl", "w") as f:
                for record in self.log_records[-1000:]:  # Last 1000 records
                    f.write(json.dumps(record) + "\n")

        # Collect performance logs
        if self.performance_logs:
            with open(debug_dir / "performance_logs.json", "w") as f:
                json.dump(self.performance_logs, f, indent=2)

        # Create zip file
        zip_path = self._temp_dir / f"chunking_debug_{self.session_id}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in debug_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(debug_dir)
                    zipf.write(file_path, arcname)

        return zip_path

    def enable_debug_mode(self, log_file: Optional[Path] = None) -> Path:
        """
        Enable comprehensive debug logging and return path to debug info.

        Args:
            log_file: Optional path for log file

        Returns:
            Path where debug information will be collected
        """
        # Configure for debug mode
        debug_config = LogConfig(
            level=LogLevel.DEBUG,
            console_output=True,
            file_output=bool(log_file),
            log_file=log_file,
            collect_performance=True,
            collect_metrics=True,
            include_module_names=True
        )

        self.configure(debug_config)

        self.user_info("Debug mode enabled - detailed logging active")
        self.user_info(f"Debug session ID: {self.session_id}")

        if log_file:
            self.user_info(f"Debug logs will be written to: {log_file}")

        return self._temp_dir or Path(tempfile.gettempdir())

    def _get_python_log_level(self, level: LogLevel) -> int:
        """Convert our log level to Python logging level."""
        mapping = {
            LogLevel.SILENT: logging.CRITICAL,
            LogLevel.MINIMAL: logging.INFO,  # Allow INFO messages for basic user updates
            LogLevel.NORMAL: logging.INFO,
            LogLevel.VERBOSE: logging.INFO,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.TRACE: logging.DEBUG
        }
        return mapping.get(level, logging.INFO)

    def _create_text_formatter(self) -> logging.Formatter:
        """Create human-readable text formatter."""
        if self.config.include_module_names:
            if self.config.level in [LogLevel.DEBUG, LogLevel.TRACE]:
                fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            else:
                fmt = '%(asctime)s - %(levelname)s - %(message)s'
        else:
            if self.config.level == LogLevel.MINIMAL:
                fmt = '%(message)s'
            else:
                fmt = '%(asctime)s - %(message)s'

        return logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')

    def _create_json_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging."""
        return JsonFormatter()

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes."""
        if not isinstance(size_str, str):
            return int(size_str)

        size_str = size_str.upper()
        multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}

        # Handle common variations like '10M'
        if size_str.endswith('M') and not size_str.endswith('MB'):
            try:
                return int(size_str[:-1]) * 1024**2
            except ValueError:
                return 10 * 1024**2

        # Handle standard suffixes
        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                try:
                    number_part = size_str[:-len(suffix)]
                    return int(float(number_part) * multiplier)
                except ValueError:
                    return 10 * 1024**2

        # Try parsing as plain number
        try:
            return int(size_str)
        except ValueError:
            # Default to 10MB if parsing fails
            return 10 * 1024**2

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for debugging."""
        import platform
        import sys

        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
            memory_info_dict = memory_info._asdict()
        except ImportError:
            memory_info_dict = {}
            cpu_info = {}

        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'chunking_strategy_version': getattr(sys.modules.get('chunking_strategy', {}), '__version__', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'memory': memory_info_dict,
            'cpu': cpu_info,
            'environment': {
                'python_path': sys.path[:3],  # First 3 entries
                'working_directory': str(Path.cwd()),
                'temp_directory': str(Path(tempfile.gettempdir()))
            }
        }


class LogCollectionHandler(logging.Handler):
    """Custom handler to collect log records for debug purposes."""

    def __init__(self, chunking_logger: ChunkingLogger):
        super().__init__()
        self.chunking_logger = chunking_logger

    def emit(self, record: logging.LogRecord) -> None:
        """Collect log record for debugging."""
        try:
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'module': record.name,
                'message': record.getMessage(),
                'filename': record.filename,
                'line_number': record.lineno,
                'function': record.funcName
            }

            # Add extra fields if present
            if hasattr(record, '__dict__'):
                extra_fields = {k: v for k, v in record.__dict__.items()
                              if k not in ['name', 'msg', 'args', 'levelname', 'levelno',
                                         'pathname', 'filename', 'module', 'lineno', 'funcName',
                                         'created', 'msecs', 'relativeCreated', 'thread',
                                         'threadName', 'processName', 'process', 'stack_info',
                                         'exc_info', 'exc_text']}
                if extra_fields:
                    log_data['extra'] = extra_fields

            self.chunking_logger.log_records.append(log_data)

            # Keep only recent records to manage memory
            if len(self.chunking_logger.log_records) > 2000:
                self.chunking_logger.log_records = self.chunking_logger.log_records[-1000:]

        except Exception:
            # Never fail on logging
            pass


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
            'filename': record.filename,
            'line_number': record.lineno,
            'function': record.funcName
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        extra_fields = {k: v for k, v in record.__dict__.items()
                       if k not in ['name', 'msg', 'args', 'levelname', 'levelno',
                                  'pathname', 'filename', 'module', 'lineno', 'funcName',
                                  'created', 'msecs', 'relativeCreated', 'thread',
                                  'threadName', 'processName', 'process', 'stack_info',
                                  'exc_info', 'exc_text']}
        if extra_fields:
            log_data.update(extra_fields)

        return json.dumps(log_data, default=str)


# Global logger instance
_logger = ChunkingLogger()


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger for the chunking strategy library.

    Args:
        name: Module name (usually __name__)

    Returns:
        Configured logger instance
    """
    if name is None:
        # Get caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'chunking_strategy')

    return _logger.get_logger(name)


def configure_logging(level: Union[str, LogLevel] = LogLevel.NORMAL, **kwargs) -> None:
    """
    Configure library-wide logging.

    Args:
        level: Logging level
        **kwargs: Additional configuration parameters
    """
    if isinstance(level, str):
        level = LogLevel(level.lower())

    _logger.configure(level=level, **kwargs)


def user_info(message: str, **kwargs) -> None:
    """Log user-facing informational message."""
    _logger.user_info(message, **kwargs)


def user_success(message: str, **kwargs) -> None:
    """Log user-facing success message."""
    _logger.user_success(message, **kwargs)


def user_warning(message: str, **kwargs) -> None:
    """Log user-facing warning message."""
    _logger.user_warning(message, **kwargs)


def user_error(message: str, **kwargs) -> None:
    """Log user-facing error message."""
    _logger.user_error(message, **kwargs)


def debug_operation(operation: str, details: Dict[str, Any], **kwargs) -> None:
    """Log detailed operation information for debugging."""
    _logger.debug_operation(operation, details, **kwargs)


def performance_log(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics."""
    _logger.performance_log(operation, duration, **kwargs)


def metrics_log(metrics: Dict[str, Any], **kwargs) -> None:
    """Log quality and processing metrics."""
    _logger.metrics_log(metrics, **kwargs)


def enable_debug_mode(log_file: Optional[Union[str, Path]] = None) -> Path:
    """
    Enable comprehensive debug logging.

    Args:
        log_file: Optional path for debug log file

    Returns:
        Path where debug information is being collected
    """
    if isinstance(log_file, str):
        log_file = Path(log_file)

    return _logger.enable_debug_mode(log_file)


def collect_debug_info() -> Path:
    """
    Collect all debug information for bug reporting.

    Returns:
        Path to debug zip file
    """
    return _logger.collect_debug_info()


def create_debug_archive(description: str = "") -> Dict[str, Any]:
    """
    Create a comprehensive debug archive for sharing.

    Args:
        description: Optional description of the issue

    Returns:
        Dictionary with debug archive info and instructions
    """
    # Enable debug mode temporarily if not already enabled
    original_level = _logger.config.level
    if original_level not in [LogLevel.DEBUG, LogLevel.TRACE]:
        _logger.configure(level=LogLevel.DEBUG)

    # Collect debug info
    debug_path = collect_debug_info()

    # Restore original level
    if original_level not in [LogLevel.DEBUG, LogLevel.TRACE]:
        _logger.configure(level=original_level)

    return {
        'debug_archive': str(debug_path),
        'session_id': _logger.session_id,
        'description': description,
        'instructions': [
            "Debug information has been collected in a zip file.",
            "This archive contains system info, logs, and performance data.",
            "You can safely share this file when reporting bugs.",
            "No sensitive data is included - only technical debugging information."
        ],
        'next_steps': [
            f"1. Locate the debug archive at: {debug_path}",
            "2. Attach this file to your bug report or GitHub issue",
            "3. Include the description of what you were trying to do",
            "4. Include any error messages you saw"
        ]
    }
