# Debugging and Log Collection Guide

This guide explains how to collect logs and debug issues with the chunking strategy library, whether you're a regular user experiencing problems or a developer contributing to the project.

## Table of Contents

- [Quick Start: Report a Bug](#quick-start-report-a-bug)
- [Understanding Log Levels](#understanding-log-levels)
- [Collecting Debug Information](#collecting-debug-information)
- [Using Logging in Your Code](#using-logging-in-your-code)
- [Advanced Debugging](#advanced-debugging)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Quick Start: Report a Bug

If you're experiencing an issue and want to report a bug, follow these steps:

### 1. Collect Debug Information (Easiest Method)

```bash
# One command to collect everything needed for a bug report
chunking-strategy debug archive "Brief description of the issue you're experiencing"
```

This creates a debug archive with all the information developers need to help you.

### 2. Alternative: Step-by-step Collection

```bash
# Enable debug mode
chunking-strategy --debug chunk your-problem-file.txt

# Or for more control:
chunking-strategy --log-level debug --log-file debug.log chunk your-file.txt

# Collect debug information
chunking-strategy debug collect --description "What you were trying to do"
```

### 3. Share the Debug Archive

The commands above create a `.zip` file containing:
- System information
- Log files
- Performance data
- Configuration used
- Error details

**This file contains NO sensitive data** - only technical information needed for debugging.

## Understanding Log Levels

The library uses different log levels for different audiences:

### For Regular Users

- **`silent`**: Only critical errors
- **`minimal`**: Basic status updates
- **`normal`** (default): Standard user information with progress and results

```bash
# Examples
chunking-strategy --log-level minimal chunk document.pdf
chunking-strategy --quiet chunk document.pdf  # Same as minimal
```

### For Power Users and Developers

- **`verbose`**: Detailed progress and performance information
- **`debug`**: Comprehensive debugging information
- **`trace`**: Maximum verbosity for development

```bash
# Examples
chunking-strategy --verbose chunk document.pdf
chunking-strategy --debug chunk document.pdf
chunking-strategy --log-level trace --log-file trace.log chunk document.pdf
```

## Collecting Debug Information

### Using CLI Commands

#### Enable Debug Mode
```bash
# Enable debug mode with default log file
chunking-strategy debug enable

# Enable with custom log file
chunking-strategy debug enable --log-file my-debug.log
```

#### Collect Debug Archive
```bash
# Basic collection
chunking-strategy debug collect

# With description and custom output location
chunking-strategy debug collect \
    --description "Chunking fails on large PDF files" \
    --output ./debug-archives/
```

#### Test Logging Functionality
```bash
# Test all log levels and see what each produces
chunking-strategy debug test-logging
```

### Using Python API

```python
import chunking_strategy as cs

# Enable debug mode programmatically
debug_dir = cs.enable_debug_mode()
print(f"Debug info will be collected in: {debug_dir}")

# Your chunking operations here
chunker = cs.create_chunker("sentence_based")
result = chunker.chunk("your content")

# Collect debug archive
debug_info = cs.create_debug_archive("Description of issue")
print(f"Debug archive created: {debug_info['debug_archive']}")
```

### Configure Logging in Your Application

```python
import chunking_strategy as cs

# Configure logging for your use case
cs.configure_logging(
    level=cs.LogLevel.VERBOSE,
    file_output=True,
    log_file="my_app_chunking.log",
    collect_performance=True
)

# Use user-friendly logging in your application
cs.user_info("Starting document processing...")
chunker = cs.create_chunker("semantic")
result = chunker.chunk("content")
cs.user_success(f"Processing complete: {len(result.chunks)} chunks created")
```

## Using Logging in Your Code

### For Application Developers

Use the user-friendly logging functions for clean, consistent output:

```python
import chunking_strategy as cs

# User-facing messages (shown at normal log level)
cs.user_info("Processing started...")
cs.user_success("Processing completed successfully!")
cs.user_warning("Large file detected, this may take longer")
cs.user_error("Failed to process file")

# Developer debugging (shown at debug log level)
cs.debug_operation("file_processing", {
    "file_size": 1024000,
    "strategy": "semantic",
    "parameters": {"threshold": 0.7}
})

# Performance tracking
cs.performance_log("chunking_operation", duration_seconds=1.23,
                  chunks_created=15, file_size_mb=2.1)

# Metrics logging
cs.metrics_log({
    "quality_score": 0.85,
    "processing_speed": "1.2MB/s",
    "memory_usage": "150MB"
})
```

### For Library Developers

Use standard Python logging in your modules:

```python
# In your module
from chunking_strategy.logging_config import get_logger

logger = get_logger(__name__)

def your_function():
    logger.info("Starting processing")
    logger.debug("Detailed processing info")
    logger.warning("Something might be wrong")
    logger.error("Something went wrong")
```

## Advanced Debugging

### Custom Log Configuration

```python
import chunking_strategy as cs
from pathlib import Path

# Advanced configuration
config = cs.LogConfig(
    level=cs.LogLevel.DEBUG,
    console_output=True,
    file_output=True,
    log_file=Path("detailed_debug.log"),
    format_json=False,  # Set to True for structured logging
    collect_performance=True,
    collect_metrics=True,
    max_file_size="50MB",
    backup_count=5
)

cs.configure_logging(config)
```

### Performance Monitoring

```python
import time
import chunking_strategy as cs

def monitored_chunking():
    start_time = time.time()

    # Your chunking operation
    chunker = cs.create_chunker("semantic")
    result = chunker.chunk(content)

    # Log performance
    cs.performance_log(
        operation="semantic_chunking",
        duration=time.time() - start_time,
        input_size=len(content),
        chunks_created=len(result.chunks),
        strategy_used=result.strategy_used
    )

    return result
```

### Integration with External Monitoring

```python
import json
import chunking_strategy as cs

# Configure JSON logging for external log aggregation
cs.configure_logging(
    level=cs.LogLevel.VERBOSE,
    format_json=True,  # Structured JSON output
    log_file="chunking_metrics.jsonl"
)

# Your operations will now produce structured JSON logs
# suitable for Elasticsearch, Splunk, etc.
```

## Troubleshooting Common Issues

### Issue: "No logs are showing"

**Solution:**
```bash
# Check your log level
chunking-strategy --log-level debug chunk yourfile.txt

# Or enable verbose mode
chunking-strategy --verbose chunk yourfile.txt
```

### Issue: "Too much log output"

**Solution:**
```bash
# Use quiet mode for minimal output
chunking-strategy --quiet chunk yourfile.txt

# Or set minimal log level
chunking-strategy --log-level minimal chunk yourfile.txt
```

### Issue: "Need logs for bug report"

**Solution:**
```bash
# One command to collect everything
chunking-strategy debug archive "Describe your issue here"
```

### Issue: "Logs are missing performance data"

**Solution:**
```python
# Enable performance collection
cs.configure_logging(
    collect_performance=True,
    collect_metrics=True
)
```

### Issue: "Can't find log files"

**Solution:**
```bash
# Specify explicit log file location
chunking-strategy --log-file /path/to/my/logs.log chunk yourfile.txt

# Or check the debug command output
chunking-strategy debug enable --log-file debug.log
```

## Best Practices

### For Regular Users

1. **Use normal log level** for day-to-day operations
2. **Use debug mode** when experiencing issues
3. **Collect debug archives** when reporting bugs
4. **Include descriptions** when creating debug archives

### For Developers

1. **Use appropriate log levels** - debug info shouldn't spam users
2. **Use structured logging** for performance metrics
3. **Include context** in debug operations
4. **Test your logging** with different log levels

### For Production Use

1. **Use minimal or normal** log levels in production
2. **Enable file logging** for troubleshooting
3. **Set up log rotation** for long-running applications
4. **Monitor performance logs** for optimization opportunities

## Examples by Use Case

### Regular User: Processing Documents

```bash
# Normal usage - clean, minimal output
chunking-strategy chunk document.pdf

# With progress information
chunking-strategy --verbose chunk large-document.pdf

# Having issues? Create debug archive
chunking-strategy debug archive "PDF processing fails on page 10"
```

### Power User: Performance Analysis

```bash
# Detailed performance logging
chunking-strategy --log-level verbose --log-file performance.log \
    batch-directory ./documents/ --parallel

# Analyze the logs
chunking-strategy debug test-logging
```

### Developer: Debugging New Feature

```bash
# Maximum debugging information
chunking-strategy --log-level trace --log-file debug.log \
    chunk test-file.txt --strategy my_new_strategy

# Or in code:
```

```python
import chunking_strategy as cs

cs.configure_logging(level=cs.LogLevel.DEBUG)
cs.debug_operation("new_feature_test", {
    "feature": "my_new_strategy",
    "test_file": "test-file.txt"
})

# Your development code here
```

### Application Integration

```python
import logging
import chunking_strategy as cs

# Set up logging for your application
cs.configure_logging(
    level=cs.LogLevel.NORMAL,
    file_output=True,
    log_file="myapp_chunking.log"
)

# Use in your application
def process_documents(file_paths):
    cs.user_info(f"Processing {len(file_paths)} documents...")

    for i, file_path in enumerate(file_paths):
        cs.user_info(f"Processing {i+1}/{len(file_paths)}: {file_path.name}")

        try:
            chunker = cs.create_chunker("auto")
            result = chunker.chunk(file_path)
            cs.user_success(f"Created {len(result.chunks)} chunks")

        except Exception as e:
            cs.user_error(f"Failed to process {file_path}: {e}")
            # Debug info automatically collected in debug mode

    cs.user_success("All documents processed!")
```

## Getting Help

If this guide doesn't solve your issue:

1. **Create a debug archive** with a detailed description
2. **Check existing issues** on GitHub
3. **Open a new issue** with the debug archive attached
4. **Include the session ID** from the debug archive

The debug archive contains everything developers need to reproduce and fix your issue!
