"""
Unit tests for the comprehensive logging configuration system.

This module tests the logging infrastructure including:
- Log level configuration
- User-friendly vs developer logging
- Debug collection and archiving
- Performance and metrics logging
- Integration with CLI
"""

import json
import logging
import pytest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from chunking_strategy.logging_config import (
    ChunkingLogger,
    LogLevel,
    LogConfig,
    configure_logging,
    get_logger,
    user_info, user_success, user_warning, user_error,
    debug_operation, performance_log, metrics_log,
    enable_debug_mode, collect_debug_info, create_debug_archive
)


class TestLogConfig:
    """Test LogConfig dataclass functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LogConfig()
        assert config.level == LogLevel.NORMAL
        assert config.console_output is True
        assert config.file_output is False
        assert config.log_file is None
        assert config.collect_performance is True
        assert config.collect_metrics is True

    def test_config_serialization(self):
        """Test configuration serialization to dictionary."""
        config = LogConfig(
            level=LogLevel.DEBUG,
            log_file=Path("test.log"),
            collect_performance=False
        )
        config_dict = config.to_dict()

        assert config_dict['level'] == 'debug'
        assert config_dict['log_file'] == 'test.log'
        assert config_dict['collect_performance'] is False


class TestChunkingLogger:
    """Test the main ChunkingLogger singleton class."""

    def setup_method(self):
        """Set up test environment."""
        # Reset singleton state completely
        if ChunkingLogger._instance is not None:
            # Clear any existing configuration and state
            ChunkingLogger._instance.config = None
            ChunkingLogger._instance.loggers = {}
            ChunkingLogger._instance.log_records = []
            ChunkingLogger._instance.performance_logs = []
        ChunkingLogger._instance = None
        ChunkingLogger._initialized = False

        # Also reset the global _logger reference
        import chunking_strategy.logging_config as logging_module
        logging_module._logger = ChunkingLogger()

        # Clear any existing handlers to avoid interference
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_singleton_behavior(self):
        """Test that ChunkingLogger is a proper singleton."""
        logger1 = ChunkingLogger()
        logger2 = ChunkingLogger()
        assert logger1 is logger2

    def test_default_initialization(self):
        """Test default logger initialization."""
        logger = ChunkingLogger()
        assert logger.config.level == LogLevel.NORMAL
        assert logger.session_id is not None
        assert len(logger.session_id) > 0

    def test_configure_with_config_object(self):
        """Test configuration with LogConfig object."""
        logger = ChunkingLogger()
        config = LogConfig(level=LogLevel.DEBUG, console_output=False)

        logger.configure(config)
        assert logger.config.level == LogLevel.DEBUG
        assert logger.config.console_output is False

    def test_configure_with_kwargs(self):
        """Test configuration with keyword arguments."""
        logger = ChunkingLogger()
        logger.configure(level='verbose', collect_performance=False)

        assert logger.config.level == LogLevel.VERBOSE
        assert logger.config.collect_performance is False

    def test_get_logger(self):
        """Test getting logger instances."""
        chunking_logger = ChunkingLogger()
        logger = chunking_logger.get_logger('test_module')

        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_module'

        # Test caching
        logger2 = chunking_logger.get_logger('test_module')
        assert logger is logger2


class TestUserLogging:
    """Test user-friendly logging functions."""

    def setup_method(self):
        """Set up test environment."""
        # Reset singleton state completely
        if ChunkingLogger._instance is not None:
            # Clear any existing configuration and state
            ChunkingLogger._instance.config = None
            ChunkingLogger._instance.loggers = {}
            ChunkingLogger._instance.log_records = []
            ChunkingLogger._instance.performance_logs = []
        ChunkingLogger._instance = None
        ChunkingLogger._initialized = False

        # Also reset the global _logger reference
        import chunking_strategy.logging_config as logging_module
        logging_module._logger = ChunkingLogger()

        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configure for testing
        configure_logging(level=LogLevel.NORMAL, console_output=False)

    def test_user_info_logging(self, caplog):
        """Test user info message logging."""
        with caplog.at_level(logging.INFO):
            user_info("Test info message")

        assert "📝 Test info message" in caplog.text

    def test_user_success_logging(self, caplog):
        """Test user success message logging."""
        with caplog.at_level(logging.INFO):
            user_success("Test success message")

        assert "✅ Test success message" in caplog.text

    def test_user_warning_logging(self, caplog):
        """Test user warning message logging."""
        with caplog.at_level(logging.WARNING):
            user_warning("Test warning message")

        assert "⚠️  Test warning message" in caplog.text

    def test_user_error_logging(self, caplog):
        """Test user error message logging."""
        with caplog.at_level(logging.ERROR):
            user_error("Test error message")

        assert "❌ Test error message" in caplog.text

    def test_user_logging_with_extra_data(self, caplog):
        """Test user logging with extra metadata."""
        with caplog.at_level(logging.INFO):
            user_info("Test message", extra_field="extra_value")

        assert "📝 Test message" in caplog.text


class TestDeveloperLogging:
    """Test developer/debug logging functions."""

    def setup_method(self):
        """Set up test environment."""
        # Reset singleton state completely
        if ChunkingLogger._instance is not None:
            # Clear any existing configuration and state
            ChunkingLogger._instance.config = None
            ChunkingLogger._instance.loggers = {}
            ChunkingLogger._instance.log_records = []
            ChunkingLogger._instance.performance_logs = []
        ChunkingLogger._instance = None
        ChunkingLogger._initialized = False

        # Also reset the global _logger reference
        import chunking_strategy.logging_config as logging_module
        logging_module._logger = ChunkingLogger()

    def test_debug_operation_logging(self):
        """Test debug operation logging by checking collection."""
        chunking_logger = ChunkingLogger()
        chunking_logger.configure(level=LogLevel.DEBUG, console_output=False)

        # Clear records before collecting
        chunking_logger.log_records.clear()

        # Enable debug mode to activate log collection
        chunking_logger.enable_debug_mode()

        # Clear records again after debug mode (it adds its own logs)
        chunking_logger.log_records.clear()

        debug_operation("test_operation", {"param": "value", "count": 42})

        # Check that the operation was logged in collection
        assert len(chunking_logger.log_records) > 0, f"No log records collected. Records: {chunking_logger.log_records}"
        found_debug_log = any("🔧 test_operation" in record.get('message', '')
                            for record in chunking_logger.log_records)
        assert found_debug_log, f"Debug log not found in records: {[r.get('message', '') for r in chunking_logger.log_records]}"

    def test_debug_operation_not_logged_at_normal_level(self, caplog):
        """Test that debug operations are not logged at normal level."""
        configure_logging(level=LogLevel.NORMAL, console_output=False)

        with caplog.at_level(logging.DEBUG):
            debug_operation("test_operation", {"param": "value"})

        # Should not appear in logs at normal level
        assert "🔧 test_operation" not in caplog.text

    def test_performance_logging(self):
        """Test performance metrics logging."""
        configure_logging(level=LogLevel.DEBUG, collect_performance=True, console_output=False)
        logger = ChunkingLogger()

        performance_log("test_operation", 1.23, chunks_processed=10)

        # Check that performance data is collected
        assert len(logger.performance_logs) > 0
        perf_log = logger.performance_logs[-1]
        assert perf_log['operation'] == 'test_operation'
        assert perf_log['duration_seconds'] == 1.23
        assert perf_log['chunks_processed'] == 10

    def test_metrics_logging(self, caplog, capsys):
        """Test metrics logging functionality."""
        configure_logging(level=LogLevel.VERBOSE, collect_metrics=True, console_output=True)

        test_metrics = {
            "quality_score": 0.85,
            "chunk_count": 15,
            "avg_chunk_size": 512
        }

        # The metrics log goes to console output
        metrics_log(test_metrics, operation="test_chunking")

        # Check captured stdout instead of caplog
        captured = capsys.readouterr()
        assert "📊 Metrics collected" in captured.out


class TestLogLevels:
    """Test different logging levels and their behavior."""

    def setup_method(self):
        """Set up test environment."""
        # Reset singleton state completely
        if ChunkingLogger._instance is not None:
            # Clear any existing configuration and state
            ChunkingLogger._instance.config = None
            ChunkingLogger._instance.loggers = {}
            ChunkingLogger._instance.log_records = []
            ChunkingLogger._instance.performance_logs = []
        ChunkingLogger._instance = None
        ChunkingLogger._initialized = False

        # Also reset the global _logger reference
        import chunking_strategy.logging_config as logging_module
        logging_module._logger = ChunkingLogger()

    @pytest.mark.parametrize("level,should_show_user,should_show_debug", [
        (LogLevel.SILENT, False, False),
        (LogLevel.MINIMAL, True, False),
        (LogLevel.NORMAL, True, False),
        (LogLevel.VERBOSE, True, False),
        (LogLevel.DEBUG, True, True),
        (LogLevel.TRACE, True, True),
    ])
    def test_log_level_filtering(self, level, should_show_user, should_show_debug, capsys):
        """Test that different log levels filter messages appropriately."""
        # Use console_output=True so we can capture with capsys
        configure_logging(level=level, console_output=True)

        user_info("User message")
        debug_operation("debug_op", {"test": "data"})

        # Capture stdout output instead of logs
        captured = capsys.readouterr()
        user_present = "📝 User message" in captured.out
        debug_present = "🔧 debug_op" in captured.out

        if should_show_user:
            assert user_present, f"User message should be shown at {level}"

        if should_show_debug:
            assert debug_present, f"Debug message should be shown at {level}"
        elif level != LogLevel.SILENT:  # Silent level might not show anything
            assert not debug_present, f"Debug message should NOT be shown at {level}"


class TestDebugCollection:
    """Test debug information collection and archiving."""

    def setup_method(self):
        """Set up test environment."""
        # Reset singleton state completely
        if ChunkingLogger._instance is not None:
            # Clear any existing configuration and state
            ChunkingLogger._instance.config = None
            ChunkingLogger._instance.loggers = {}
            ChunkingLogger._instance.log_records = []
            ChunkingLogger._instance.performance_logs = []
        ChunkingLogger._instance = None
        ChunkingLogger._initialized = False

        # Also reset the global _logger reference
        import chunking_strategy.logging_config as logging_module
        logging_module._logger = ChunkingLogger()

    def test_enable_debug_mode(self):
        """Test enabling debug mode."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as tmp_file:
            log_file = Path(tmp_file.name)

        try:
            # Since we're using a singleton pattern, get the instance first
            logger = ChunkingLogger()
            debug_dir = enable_debug_mode(log_file)

            # The same logger instance should now have debug configuration
            assert logger.config.level == LogLevel.DEBUG
            assert logger.config.file_output is True
            assert logger.config.log_file == log_file
            assert isinstance(debug_dir, Path)

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_collect_debug_info(self):
        """Test debug information collection."""
        configure_logging(level=LogLevel.DEBUG)

        # Generate some activity to collect
        user_info("Test message")
        debug_operation("test_op", {"data": "value"})
        performance_log("test_perf", 0.123)

        debug_archive_path = collect_debug_info()

        assert debug_archive_path.exists()
        assert debug_archive_path.suffix == '.zip'

        # Verify archive contents
        with zipfile.ZipFile(debug_archive_path, 'r') as archive:
            files = archive.namelist()
            assert 'system_info.json' in files
            assert 'config.json' in files

        # Cleanup
        debug_archive_path.unlink()

    def test_create_debug_archive(self):
        """Test creating debug archive with description."""
        configure_logging(level=LogLevel.DEBUG)

        # Generate some activity
        user_info("Test activity")
        debug_operation("test_operation", {"param": "value"})

        archive_info = create_debug_archive("Test issue description")

        assert 'debug_archive' in archive_info
        assert 'session_id' in archive_info
        assert 'description' in archive_info
        assert 'instructions' in archive_info
        assert 'next_steps' in archive_info

        assert archive_info['description'] == "Test issue description"

        # Verify archive exists
        archive_path = Path(archive_info['debug_archive'])
        assert archive_path.exists()

        # Cleanup
        archive_path.unlink()


class TestLogCollection:
    """Test log record collection for debugging."""

    def setup_method(self):
        """Set up test environment."""
        # Reset singleton state completely
        if ChunkingLogger._instance is not None:
            # Clear any existing configuration and state
            ChunkingLogger._instance.config = None
            ChunkingLogger._instance.loggers = {}
            ChunkingLogger._instance.log_records = []
            ChunkingLogger._instance.performance_logs = []
        ChunkingLogger._instance = None
        ChunkingLogger._initialized = False

        # Also reset the global _logger reference
        import chunking_strategy.logging_config as logging_module
        logging_module._logger = ChunkingLogger()

        # Configure for debug collection
        configure_logging(level=LogLevel.DEBUG)

    def test_log_record_collection(self):
        """Test that log records are collected for debugging."""
        logger = ChunkingLogger()
        initial_count = len(logger.log_records)

        user_info("Test message 1")
        user_warning("Test message 2")
        debug_operation("test_op", {"data": "value"})

        # Should have collected log records
        assert len(logger.log_records) > initial_count

        # Check structure of collected records
        recent_records = logger.log_records[-3:]
        for record in recent_records:
            assert 'timestamp' in record
            assert 'level' in record
            assert 'module' in record
            assert 'message' in record

    def test_log_record_memory_management(self):
        """Test that log record collection manages memory properly."""
        # Enable DEBUG level to activate LogCollectionHandler
        configure_logging(level=LogLevel.DEBUG, console_output=False)
        logger = ChunkingLogger()

        # Generate many log records
        for i in range(2500):  # Over the 2000 limit
            user_info(f"Test message {i}")

        # Memory management: trims to 1000 at 2001, then grows to 1499 at 2500
        # Should be between 1000-1500 records (trimmed once when exceeded 2000)
        assert 1000 <= len(logger.log_records) <= 1500, f"Expected 1000-1500 records, got {len(logger.log_records)}"


class TestFileLogging:
    """Test file-based logging functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Reset singleton state completely
        if ChunkingLogger._instance is not None:
            # Clear any existing configuration and state
            ChunkingLogger._instance.config = None
            ChunkingLogger._instance.loggers = {}
            ChunkingLogger._instance.log_records = []
            ChunkingLogger._instance.performance_logs = []
        ChunkingLogger._instance = None
        ChunkingLogger._initialized = False

        # Also reset the global _logger reference
        import chunking_strategy.logging_config as logging_module
        logging_module._logger = ChunkingLogger()

    def test_file_logging_configuration(self):
        """Test that file logging can be configured."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as tmp_file:
            log_file = Path(tmp_file.name)

        try:
            configure_logging(
                level=LogLevel.NORMAL,
                file_output=True,
                log_file=log_file,
                console_output=False
            )

            user_info("Test file logging")

            # Give logger time to write
            import time
            time.sleep(0.1)

            # Check that log file was created and contains content
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test file logging" in content

        finally:
            if log_file.exists():
                log_file.unlink()


class TestSystemIntegration:
    """Test integration with external systems and CLI."""

    def setup_method(self):
        """Set up test environment."""
        # Reset singleton state completely
        if ChunkingLogger._instance is not None:
            # Clear any existing configuration and state
            ChunkingLogger._instance.config = None
            ChunkingLogger._instance.loggers = {}
            ChunkingLogger._instance.log_records = []
            ChunkingLogger._instance.performance_logs = []
        ChunkingLogger._instance = None
        ChunkingLogger._initialized = False

        # Also reset the global _logger reference
        import chunking_strategy.logging_config as logging_module
        logging_module._logger = ChunkingLogger()

    def test_get_logger_function(self):
        """Test the public get_logger function."""
        logger = get_logger('test_module')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_module'

    def test_get_logger_with_caller_detection(self):
        """Test automatic caller detection in get_logger."""
        logger = get_logger()  # Should auto-detect module name
        assert isinstance(logger, logging.Logger)
        # Name should contain this test module
        assert 'test_logging_config' in logger.name

    def test_system_info_collection(self):
        """Test system information collection for debug archives."""
        chunking_logger = ChunkingLogger()
        system_info = chunking_logger._collect_system_info()

        # Basic system info should always be present
        assert 'platform' in system_info
        assert 'python_version' in system_info
        assert 'session_id' in system_info
        assert 'timestamp' in system_info
        assert 'environment' in system_info

        # Memory and CPU info depend on psutil availability
        # Both empty dict (when psutil not available) and populated dict (when available) are valid
        assert 'memory' in system_info
        assert 'cpu' in system_info
        assert isinstance(system_info['memory'], dict)
        assert isinstance(system_info['cpu'], dict)


class TestConfiguration:
    """Test various configuration scenarios."""

    def setup_method(self):
        """Set up test environment."""
        # Reset singleton state completely
        if ChunkingLogger._instance is not None:
            # Clear any existing configuration and state
            ChunkingLogger._instance.config = None
            ChunkingLogger._instance.loggers = {}
            ChunkingLogger._instance.log_records = []
            ChunkingLogger._instance.performance_logs = []
        ChunkingLogger._instance = None
        ChunkingLogger._initialized = False

        # Also reset the global _logger reference
        import chunking_strategy.logging_config as logging_module
        logging_module._logger = ChunkingLogger()

    def test_configure_logging_function(self):
        """Test the public configure_logging function."""
        configure_logging(
            level=LogLevel.VERBOSE,
            collect_performance=False,
            console_output=True
        )

        logger = ChunkingLogger()
        assert logger.config.level == LogLevel.VERBOSE
        assert logger.config.collect_performance is False
        assert logger.config.console_output is True

    def test_invalid_log_level_handling(self):
        """Test handling of invalid log level strings."""
        chunking_logger = ChunkingLogger()

        # Should handle invalid level gracefully
        chunking_logger.configure(level='invalid_level')

        # Should not crash and should maintain some valid level
        assert isinstance(chunking_logger.config.level, LogLevel)

    def test_multiple_configuration_changes(self):
        """Test multiple configuration changes."""
        chunking_logger = ChunkingLogger()

        # Initial config
        chunking_logger.configure(level=LogLevel.DEBUG)
        assert chunking_logger.config.level == LogLevel.DEBUG

        # Change config
        chunking_logger.configure(level=LogLevel.NORMAL, collect_performance=False)
        assert chunking_logger.config.level == LogLevel.NORMAL
        assert chunking_logger.config.collect_performance is False


if __name__ == '__main__':
    pytest.main([__file__])
