# Changelog

All notable changes to the Chunking Strategy Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffold
- Core architecture and base classes
- Universal chunk schema for all modalities
- Registry system for strategy management
- Orchestrator for intelligent strategy selection
- Comprehensive benchmarking suite for performance testing
- Example documentation and usage guides

### Added
- **Output control for large files**: New CLI options to prevent huge output files
  - `--summary-only`: Process files but show only summary (no chunk content)
  - `--no-output`: Suppress all output file generation
  - `--skip-large-output N`: Skip output if result has more than N chunks
  - `--no-output-files`: Batch processing without creating output files
  - `--skip-large-files N`: Skip output for files with more than N chunks
- Large files optimized configuration profile
- Enhanced batch processing with selective output control

### Fixed
- Fixed hanging issue in sentence chunker benchmark when processing large files
- Added timeout protection for quality evaluation in benchmarks
- Created fast benchmark alternative without quality evaluation
- Resolved parameter injection conflicts in chunker registry
- Streaming capabilities for large files
- Pipeline system for chained operations
- Adaptive chunking with feedback
- Quality metrics and validation
- Comprehensive CLI interface
- Configuration profiles (RAG, summarization, search)
- Benchmarking utilities
- Plugin discovery system

### Strategies Implemented
- Fixed-size chunking (characters, bytes, words)
- Sentence-based chunking (with overlap support)

### Strategies Planned
- Paragraph-based chunking
- Overlapping window chunking
- Rolling hash chunking
- Semantic chunking with embeddings
- Structure-aware chunking
- And 90+ more strategies across all categories

## [0.1.0] - 2024-01-XX

### Added
- Initial release with core functionality
- Basic text chunking strategies
- CLI interface
- Configuration system
- Quality evaluation framework
- Comprehensive test suite
- Documentation and examples

### Note
This is the initial scaffold release. The first working algorithm (fixed-size chunking)
has been implemented with full testing, documentation, and quality validation.
Additional algorithms will be added incrementally following the same rigorous process.

### Development Workflow
1. Implement one algorithm at a time
2. Add comprehensive unit tests
3. Document with examples
4. Benchmark performance
5. Validate quality metrics
6. Only then proceed to next algorithm

This ensures each algorithm is production-ready before moving forward.
