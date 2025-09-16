"""
Global pytest configuration for chunking strategy tests.

This file optimizes test performance by configuring default settings
to avoid hardware detection overhead during testing.
"""

import pytest
import os

# Set environment variables to optimize for testing
@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure optimal settings for test performance."""

    # Disable hardware optimization by default in tests (can be overridden)
    if "CHUNKING_ENABLE_HARDWARE_OPT" not in os.environ:
        os.environ["CHUNKING_ENABLE_HARDWARE_OPT"] = "False"

    # Disable GPU detection overhead in tests
    if "CHUNKING_FORCE_CPU" not in os.environ:
        os.environ["CHUNKING_FORCE_CPU"] = "True"

    # Use conservative parallelization settings
    if "CHUNKING_DEFAULT_WORKERS" not in os.environ:
        os.environ["CHUNKING_DEFAULT_WORKERS"] = "2"

    yield

    # Cleanup after tests
    for key in ["CHUNKING_ENABLE_HARDWARE_OPT", "CHUNKING_FORCE_CPU", "CHUNKING_DEFAULT_WORKERS"]:
        os.environ.pop(key, None)

@pytest.fixture
def fast_orchestrator():
    """Create an orchestrator optimized for fast testing."""
    from chunking_strategy.orchestrator import ChunkerOrchestrator
    return ChunkerOrchestrator(
        enable_hardware_optimization=False,
        enable_smart_parallelization=True
    )

@pytest.fixture
def hw_orchestrator():
    """Create an orchestrator with hardware optimization for performance tests."""
    from chunking_strategy.orchestrator import ChunkerOrchestrator
    return ChunkerOrchestrator(
        enable_hardware_optimization=True,
        enable_smart_parallelization=True
    )
