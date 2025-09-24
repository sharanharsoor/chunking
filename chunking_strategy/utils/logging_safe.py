"""
Safe logging utilities to prevent string formatting errors.

This module provides utilities to safely log user content that might
contain format strings or curly braces that could cause formatting errors.
"""

import logging
import re
from typing import Any, Optional


def safe_format_message(message: str, *args, **kwargs) -> str:
    """
    Safely format a log message, escaping any potential format strings in content.

    Args:
        message: Message template with format placeholders
        *args: Positional arguments for formatting
        **kwargs: Keyword arguments for formatting

    Returns:
        Safely formatted message string
    """
    try:
        # If we have format args, try normal formatting first
        if args or kwargs:
            return message.format(*args, **kwargs)
        else:
            # If no format args, just return the message as-is
            return message
    except (IndexError, KeyError, ValueError) as e:
        # If formatting fails, escape curly braces and return a safe message
        safe_message = message.replace('{', '{{').replace('}', '}}')
        return f"[SAFE_LOG] {safe_message} (Original format error: {e})"


def safe_log_content(logger: logging.Logger, level: int, message: str, content: Any = None, **kwargs):
    """
    Safely log a message that includes user content.

    Args:
        logger: Logger instance
        level: Logging level (e.g., logging.INFO)
        message: Base message
        content: User content to include (will be escaped)
        **kwargs: Additional context to log
    """
    try:
        # Escape any format strings in user content
        if content is not None:
            # Convert content to string and escape format chars
            content_str = str(content)
            # Limit content length to prevent excessive logs
            if len(content_str) > 200:
                content_str = content_str[:200] + "..."
            safe_content = content_str.replace('{', '{{').replace('}', '}}')
        else:
            safe_content = "None"

        # Create safe context dict
        safe_context = {}
        for key, value in kwargs.items():
            safe_value = str(value).replace('{', '{{').replace('}', '}}') if value is not None else "None"
            safe_context[key] = safe_value

        # Log with safe formatting
        if safe_context:
            context_str = ", ".join(f"{k}={v}" for k, v in safe_context.items())
            logger.log(level, f"{message} Content: {safe_content} Context: {context_str}")
        else:
            logger.log(level, f"{message} Content: {safe_content}")

    except Exception as e:
        # Last resort: log a basic error message
        logger.error(f"Logging error prevented: {e}")


class SafeLogger:
    """
    Wrapper around standard logger that safely handles user content.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def info_safe(self, message: str, content: Any = None, **kwargs):
        """Safely log info message with content."""
        safe_log_content(self.logger, logging.INFO, message, content, **kwargs)

    def error_safe(self, message: str, content: Any = None, **kwargs):
        """Safely log error message with content."""
        safe_log_content(self.logger, logging.ERROR, message, content, **kwargs)

    def warning_safe(self, message: str, content: Any = None, **kwargs):
        """Safely log warning message with content."""
        safe_log_content(self.logger, logging.WARNING, message, content, **kwargs)

    def debug_safe(self, message: str, content: Any = None, **kwargs):
        """Safely log debug message with content."""
        safe_log_content(self.logger, logging.DEBUG, message, content, **kwargs)


def create_safe_logger(name: str) -> SafeLogger:
    """Create a safe logger for the given name."""
    return SafeLogger(logging.getLogger(name))
