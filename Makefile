# Makefile for Chunking Strategy Library

.PHONY: help install install-dev test test-unit test-integration test-performance lint format type-check clean build docs benchmark

# Default target
help:
	@echo "Available targets:"
	@echo "  install        Install package for production"
	@echo "  install-dev    Install package for development"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-performance  Run performance tests only"
	@echo "  lint           Run all linting checks"
	@echo "  format         Format code with black and isort"
	@echo "  type-check     Run type checking with mypy"
	@echo "  clean          Clean build artifacts"
	@echo "  build          Build package for distribution"
	@echo "  docs           Build documentation"
	@echo "  benchmark      Run benchmarks"

# Installation targets
install:
	pip install .

install-dev:
	pip install -e ".[dev,all]"

# Testing targets
test:
	pytest tests/ -v --cov=chunking_strategy --cov-report=html --cov-report=term

test-unit:
	pytest tests/ -v -m "unit" --cov=chunking_strategy

test-integration:
	pytest tests/ -v -m "integration"

test-performance:
	pytest tests/ -v -m "performance"

test-quick:
	pytest tests/ -v -x --ff

# Code quality targets
lint: ruff-check black-check isort-check mypy

ruff-check:
	ruff check chunking_strategy/ tests/

black-check:
	black --check chunking_strategy/ tests/

isort-check:
	isort --check-only chunking_strategy/ tests/

mypy:
	mypy chunking_strategy/

format:
	black chunking_strategy/ tests/
	isort chunking_strategy/ tests/
	ruff check --fix chunking_strategy/ tests/

# Build and distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish-test: build
	python -m twine upload --repository testpypi dist/*

publish: build
	python -m twine upload dist/*

# Documentation
docs:
	@echo "Documentation build not yet implemented"
	@echo "Placeholder for sphinx documentation build"

# Benchmarking
benchmark:
	python -c "from chunking_strategy.utils.benchmarking import BenchmarkRunner; \
	           import tempfile; \
	           runner = BenchmarkRunner(); \
	           text = 'This is a test sentence. ' * 100; \
	           results = runner.quick_benchmark(text); \
	           print('Quick benchmark results:'); \
	           [print(f'{r.strategy_name}: {r.processing_time:.3f}s') for r in results if r.success]"

# Development workflow helpers
check: format lint test

pre-commit: format lint test-unit

ci: lint test

# Algorithm development workflow
new-algorithm:
	@echo "Algorithm Development Workflow:"
	@echo "1. Create strategy file in appropriate category"
	@echo "2. Implement with @register_chunker decorator"
	@echo "3. Add comprehensive tests (see tests/test_fixed_size_chunker.py)"
	@echo "4. Run: make test-unit"
	@echo "5. Add documentation and examples"
	@echo "6. Run: make benchmark"
	@echo "7. Validate quality: make test"
	@echo "8. Only then proceed to next algorithm"

# Install pre-commit hooks
setup-hooks:
	pre-commit install

# Quick development setup
setup: install-dev setup-hooks
	@echo "Development environment ready!"
	@echo "Run 'make check' to verify everything works"

# Release workflow
release-check: clean format lint test build
	@echo "Release checks passed!"
	@echo "Ready for release. Run 'make publish' or 'make publish-test'"
