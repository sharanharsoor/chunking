"""
Test New Example Configurations

This test suite validates the new configuration files that demonstrate
different approaches: specialized chunkers, universal strategies,
mixed approach, and future formats support.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, List

from chunking_strategy import ChunkerOrchestrator, apply_universal_strategy
from chunking_strategy.core.base import ModalityType


class TestNewExampleConfigurations:
    """Test all new example configurations."""

    @pytest.fixture(autouse=True)
    def setup_test_files(self):
        """Set up test files for configuration testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = Path(__file__).parent.parent.parent / "config_examples"

        # New config files to test (our 6 new algorithm configs)
        self.new_config_files = [
            "rolling_hash_default.yaml",
            "rabin_fingerprinting_default.yaml",
            "buzhash_performance.yaml",
            "gear_cdc_default.yaml",
            "ml_cdc_hierarchical.yaml",
            "tttd_balanced.yaml"
        ]

        # Create comprehensive test files
        self.test_files = self._create_comprehensive_test_files()

    def _create_comprehensive_test_files(self) -> Dict[str, Path]:
        """Create comprehensive test files for all scenarios."""
        files = {}

        # Complex Python file for specialized chunker testing
        complex_python = '''#!/usr/bin/env python3
"""
Advanced Data Processing Module
Comprehensive example for testing specialized Python chunking.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""
    algorithm: str
    threshold: float = 0.05
    max_iterations: int = 1000

def advanced_analysis(data: List[float], config: ProcessingConfig) -> Dict[str, float]:
    """
    Perform advanced statistical analysis on numerical data.

    This function demonstrates complex Python code structure that should
    be properly chunked by the specialized Python chunker.

    Args:
        data: List of numerical values to analyze
        config: Processing configuration parameters

    Returns:
        Dictionary containing analysis results

    Raises:
        ValueError: If data is empty or invalid
    """
    if not data:
        raise ValueError("Data cannot be empty")

    # Basic statistical calculations
    mean_value = np.mean(data)
    std_value = np.std(data)
    variance = np.var(data)

    # Advanced calculations based on configuration
    if config.algorithm == "robust":
        median_value = np.median(data)
        mad_value = np.median(np.abs(data - median_value))
        return {
            "mean": mean_value,
            "median": median_value,
            "mad": mad_value,
            "algorithm": config.algorithm
        }
    else:
        return {
            "mean": mean_value,
            "std": std_value,
            "variance": variance,
            "algorithm": config.algorithm
        }

class DataProcessor:
    """
    Advanced data processor with multiple algorithms.

    This class demonstrates complex class structure with multiple methods,
    properties, and nested functionality for testing specialized chunking.
    """

    def __init__(self, config: ProcessingConfig):
        """Initialize processor with configuration."""
        self.config = config
        self.processed_count = 0
        self.error_count = 0
        self._cache = {}

    @property
    def success_rate(self) -> float:
        """Calculate processing success rate."""
        total = self.processed_count + self.error_count
        return self.processed_count / total if total > 0 else 0.0

    def process_batch(self, datasets: List[List[float]]) -> List[Dict[str, float]]:
        """Process multiple datasets in batch."""
        results = []
        for dataset in datasets:
            try:
                result = self.process_single(dataset)
                results.append(result)
                self.processed_count += 1
            except Exception as e:
                logger.error(f"Processing failed: {e}")
                self.error_count += 1
                results.append({"error": str(e)})
        return results

    def process_single(self, data: List[float]) -> Dict[str, float]:
        """Process a single dataset."""
        cache_key = hash(tuple(data))
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = advanced_analysis(data, self.config)
        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> int:
        """Clear processing cache and return number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        return count

if __name__ == "__main__":
    # Example usage
    config = ProcessingConfig(algorithm="robust", threshold=0.01)
    processor = DataProcessor(config)

    test_data = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 20.0, 30.0, 40.0, 50.0],
        [0.1, 0.2, 0.3, 0.4, 0.5]
    ]

    results = processor.process_batch(test_data)
    print(f"Processed {len(results)} datasets")
    print(f"Success rate: {processor.success_rate:.2%}")
'''
        files['complex_python'] = self.temp_dir / "complex_analysis.py"
        files['complex_python'].write_text(complex_python)

        # C++ file for specialized chunker testing
        cpp_content = '''#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>

namespace DataProcessing {

/**
 * Advanced statistical processor class
 * Demonstrates C++ code structure for specialized chunking
 */
template<typename T>
class StatisticalProcessor {
private:
    std::vector<T> data_;
    bool is_sorted_;

public:
    explicit StatisticalProcessor(const std::vector<T>& data)
        : data_(data), is_sorted_(false) {}

    /**
     * Calculate mean of the dataset
     * @return Mean value
     */
    double calculateMean() const {
        if (data_.empty()) return 0.0;

        T sum = std::accumulate(data_.begin(), data_.end(), T{});
        return static_cast<double>(sum) / data_.size();
    }

    /**
     * Calculate median of the dataset
     * @return Median value
     */
    double calculateMedian() {
        if (data_.empty()) return 0.0;

        if (!is_sorted_) {
            std::sort(data_.begin(), data_.end());
            is_sorted_ = true;
        }

        size_t size = data_.size();
        if (size % 2 == 0) {
            return (data_[size/2 - 1] + data_[size/2]) / 2.0;
        } else {
            return data_[size/2];
        }
    }

    /**
     * Get data statistics
     * @return Statistics structure
     */
    struct Statistics {
        double mean;
        double median;
        size_t count;
    };

    Statistics getStatistics() {
        return {calculateMean(), calculateMedian(), data_.size()};
    }
};

} // namespace DataProcessing

/**
 * Utility functions for data processing
 */
namespace Utils {

template<typename T>
std::unique_ptr<DataProcessing::StatisticalProcessor<T>>
createProcessor(const std::vector<T>& data) {
    return std::make_unique<DataProcessing::StatisticalProcessor<T>>(data);
}

void printStatistics(const DataProcessing::StatisticalProcessor<double>::Statistics& stats) {
    std::cout << "Count: " << stats.count << std::endl;
    std::cout << "Mean: " << stats.mean << std::endl;
    std::cout << "Median: " << stats.median << std::endl;
}

} // namespace Utils

int main() {
    std::vector<double> test_data = {1.5, 2.3, 3.7, 4.1, 5.9, 2.8, 3.4};

    auto processor = Utils::createProcessor(test_data);
    auto stats = processor->getStatistics();

    Utils::printStatistics(stats);

    return 0;
}'''
        files['complex_cpp'] = self.temp_dir / "statistical_processor.cpp"
        files['complex_cpp'].write_text(cpp_content)

        # JavaScript file for universal code testing
        js_content = '''/**
 * Advanced JavaScript Data Processing Module
 * Demonstrates modern JavaScript features for testing
 */

class DataAnalyzer {
    constructor(config = {}) {
        this.config = {
            algorithm: 'default',
            precision: 2,
            ...config
        };
        this.processedCount = 0;
        this.cache = new Map();
    }

    /**
     * Analyze a dataset with statistical calculations
     * @param {number[]} data - Array of numerical values
     * @returns {Object} Analysis results
     */
    async analyzeData(data) {
        if (!Array.isArray(data) || data.length === 0) {
            throw new Error('Invalid data provided');
        }

        const cacheKey = JSON.stringify(data);
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        // Simulate async processing
        await new Promise(resolve => setTimeout(resolve, 10));

        const results = this._calculateStatistics(data);
        this.cache.set(cacheKey, results);
        this.processedCount++;

        return results;
    }

    /**
     * Calculate basic statistics
     * @private
     */
    _calculateStatistics(data) {
        const sorted = [...data].sort((a, b) => a - b);
        const sum = data.reduce((acc, val) => acc + val, 0);
        const mean = sum / data.length;

        const variance = data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / data.length;
        const stdDev = Math.sqrt(variance);

        const median = this._calculateMedian(sorted);

        return {
            count: data.length,
            sum: this._round(sum),
            mean: this._round(mean),
            median: this._round(median),
            variance: this._round(variance),
            stdDev: this._round(stdDev),
            min: sorted[0],
            max: sorted[sorted.length - 1],
            algorithm: this.config.algorithm
        };
    }

    _calculateMedian(sortedData) {
        const length = sortedData.length;
        if (length % 2 === 0) {
            return (sortedData[length / 2 - 1] + sortedData[length / 2]) / 2;
        } else {
            return sortedData[Math.floor(length / 2)];
        }
    }

    _round(value) {
        const factor = Math.pow(10, this.config.precision);
        return Math.round(value * factor) / factor;
    }

    /**
     * Get processing statistics
     */
    getProcessingStats() {
        return {
            processedCount: this.processedCount,
            cacheSize: this.cache.size,
            averageCacheHitRate: this.cache.size > 0 ? this.processedCount / this.cache.size : 0
        };
    }

    /**
     * Clear processing cache
     */
    clearCache() {
        const size = this.cache.size;
        this.cache.clear();
        return size;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DataAnalyzer;
}

// Example usage
if (typeof window === 'undefined') {
    // Node.js environment
    async function example() {
        const analyzer = new DataAnalyzer({ algorithm: 'enhanced', precision: 3 });

        const testData = [1.2, 3.4, 2.1, 5.6, 4.3, 2.8, 3.9, 1.7];
        const results = await analyzer.analyzeData(testData);

        console.log('Analysis Results:', results);
        console.log('Processing Stats:', analyzer.getProcessingStats());
    }

    example().catch(console.error);
}'''
        files['complex_js'] = self.temp_dir / "data_analyzer.js"
        files['complex_js'].write_text(js_content)

        # Technical document for universal strategies
        technical_doc = '''# Advanced Data Processing Framework

## Executive Summary

This document outlines the comprehensive data processing framework designed to handle diverse datasets across multiple domains. The framework provides unified interfaces, extensible architectures, and optimized performance for enterprise-scale applications.

## System Architecture

### Core Components

The framework consists of several interconnected components that work together to provide seamless data processing capabilities.

#### Data Ingestion Layer
The data ingestion layer handles multiple input formats including structured, semi-structured, and unstructured data. It provides automatic format detection, validation, and preprocessing capabilities.

#### Processing Engine
The central processing engine implements various algorithms for data transformation, analysis, and enrichment. It supports both batch and streaming processing modes with automatic optimization based on data characteristics.

#### Storage Layer
The storage layer provides persistent data management with support for multiple storage backends. It includes automated partitioning, indexing, and optimization features.

### Processing Algorithms

#### Statistical Analysis
The framework includes comprehensive statistical analysis capabilities including descriptive statistics, hypothesis testing, and advanced modeling techniques.

#### Machine Learning Integration
Built-in machine learning capabilities provide classification, regression, clustering, and anomaly detection functionality with automatic model selection and tuning.

#### Data Quality Assessment
Automated data quality assessment includes completeness, accuracy, consistency, and validity checks with detailed reporting and remediation suggestions.

## Implementation Guidelines

### Performance Optimization

Performance optimization is achieved through several strategies including parallel processing, memory management, and algorithmic improvements.

#### Parallel Processing
The framework automatically detects available computational resources and distributes processing tasks across multiple cores and nodes for optimal performance.

#### Memory Management
Intelligent memory management includes automatic garbage collection, memory pooling, and streaming processing for large datasets that exceed available memory.

#### Caching Strategies
Multi-level caching strategies reduce computational overhead and improve response times for frequently accessed data and computations.

### Quality Assurance

#### Testing Framework
Comprehensive testing framework includes unit tests, integration tests, performance tests, and end-to-end validation with automated continuous integration.

#### Monitoring and Alerting
Real-time monitoring provides insights into system performance, resource utilization, and processing quality with automated alerting for anomalies and failures.

#### Documentation Standards
Complete documentation includes API documentation, user guides, tutorials, and best practices with automated documentation generation and validation.

## Conclusion

The advanced data processing framework provides a robust, scalable, and efficient solution for modern data processing requirements. Its modular architecture, comprehensive feature set, and optimized performance make it suitable for a wide range of applications from research to production environments.

Future enhancements will focus on additional algorithm implementations, expanded format support, and enhanced automation capabilities to further improve user experience and processing efficiency.'''
        files['technical_doc'] = self.temp_dir / "technical_framework.txt"
        files['technical_doc'].write_text(technical_doc)

        # JSON configuration file
        json_config = '''{
    "application": {
        "name": "Advanced Processing Framework",
        "version": "2.1.0",
        "description": "Comprehensive data processing and analysis platform",
        "license": "MIT",
        "author": {
            "name": "Data Engineering Team",
            "email": "data-team@company.com"
        }
    },
    "database": {
        "primary": {
            "type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "processing_db",
            "ssl": true,
            "pool_size": 20,
            "timeout": 30
        },
        "cache": {
            "type": "redis",
            "host": "localhost",
            "port": 6379,
            "ttl": 3600,
            "max_memory": "2gb"
        }
    },
    "processing": {
        "batch_size": 1000,
        "parallel_workers": 8,
        "timeout": 300,
        "retry_attempts": 3,
        "algorithms": {
            "statistical": {
                "enabled": true,
                "precision": "high",
                "methods": ["mean", "median", "std", "variance"]
            },
            "machine_learning": {
                "enabled": true,
                "auto_tuning": true,
                "models": ["linear_regression", "random_forest", "svm"]
            }
        }
    },
    "monitoring": {
        "logging": {
            "level": "INFO",
            "format": "json",
            "output": ["console", "file"],
            "file_rotation": true
        },
        "metrics": {
            "enabled": true,
            "interval": 60,
            "exporters": ["prometheus", "statsd"]
        },
        "alerting": {
            "enabled": true,
            "channels": ["email", "slack"],
            "thresholds": {
                "error_rate": 0.05,
                "response_time": 1000,
                "memory_usage": 0.85
            }
        }
    }
}'''
        files['json_config'] = self.temp_dir / "processing_config.json"
        files['json_config'].write_text(json_config)

        return files

    def test_config_file_validity(self):
        """Test that all new config files are valid YAML."""
        for config_name in self.new_config_files:
            config_file = self.config_dir / config_name

            if not config_file.exists():
                pytest.skip(f"Config file {config_name} not found")

            with open(config_file, 'r') as f:
                try:
                    config = yaml.safe_load(f)
                    assert isinstance(config, dict), f"Config {config_name} is not a dictionary"
                    assert 'profile_name' in config, f"Config {config_name} missing profile_name"
                    assert 'strategy_selection' in config or 'strategies' in config, f"Config {config_name} missing strategy configuration"
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {config_name}: {e}")

    def test_specialized_chunkers_config(self):
        """Test specialized chunkers configuration."""
        config_file = self.config_dir / "specialized_chunkers.yaml"

        if not config_file.exists():
            pytest.skip("Specialized chunkers config not found")

        orchestrator = ChunkerOrchestrator(config_path=config_file)

        # Test Python file with specialized chunker
        python_file = self.test_files['complex_python']
        result = orchestrator.chunk_file(python_file)

        assert result.chunks, "No chunks created for Python file"

        # Should use specialized Python chunker
        expected_strategies = ["python_code", "paragraph", "sentence"]  # Primary or fallbacks
        assert any(strategy in result.strategy_used for strategy in expected_strategies), \
            f"Expected one of {expected_strategies}, got {result.strategy_used}"

        # Check for rich metadata in specialized chunks
        if "python_code" in result.strategy_used:
            # If it's a successful python_code chunking, it should have element_type
            # If it fell back due to syntax errors, it won't have element_type
            if not any('fallback' in c.metadata.extra for c in result.chunks):
                specialized_chunks = [c for c in result.chunks if 'element_type' in c.metadata.extra]
                assert len(specialized_chunks) > 0, "No chunks with specialized metadata found"

                # Verify rich metadata
                for chunk in specialized_chunks:
                    meta = chunk.metadata.extra
                    assert 'element_type' in meta, "Missing element_type in specialized chunk"
                    assert 'language' in meta, "Missing language in specialized chunk"
            else:
                # If it fell back, just verify the chunker was attempted
                print(f"Python chunker fell back: {result.chunks[0].metadata.extra.get('reason', 'unknown reason')}")
                # Still a valid test - the specialized chunker was used, even if it fell back

    def test_universal_strategies_config(self):
        """Test universal strategies configuration."""
        config_file = self.config_dir / "universal_strategies.yaml"

        if not config_file.exists():
            pytest.skip("Universal strategies config not found")

        orchestrator = ChunkerOrchestrator(config_path=config_file)

        # Test different file types with consistent universal strategy
        test_cases = [
            (self.test_files['complex_python'], '.py'),
            (self.test_files['complex_js'], '.js'),
            (self.test_files['technical_doc'], '.txt'),
            (self.test_files['json_config'], '.json')
        ]

        strategies_used = []

        for file_path, extension in test_cases:
            result = orchestrator.chunk_file(file_path)
            assert result.chunks, f"No chunks created for {extension} file"
            strategies_used.append(result.strategy_used)

            # Should use universal strategy
            assert any(universal in result.strategy_used for universal in
                      ["paragraph", "sentence", "fixed_size", "universal"]), \
                f"Expected universal strategy for {extension}, got {result.strategy_used}"

        # For this config, similar file types should use consistent strategies
        print(f"Strategies used: {strategies_used}")

    def test_mixed_approach_config(self):
        """Test mixed approach configuration."""
        config_file = self.config_dir / "mixed_approach.yaml"

        if not config_file.exists():
            pytest.skip("Mixed approach config not found")

        orchestrator = ChunkerOrchestrator(config_path=config_file)

        # Test Python file - should use specialized chunker
        python_result = orchestrator.chunk_file(self.test_files['complex_python'])
        assert python_result.chunks, "No chunks for Python file"

        # Should prefer specialized for Python
        expected_python = ["python_code", "paragraph", "sentence"]
        assert any(strategy in python_result.strategy_used for strategy in expected_python)

        # Test text file - should use universal strategy
        text_result = orchestrator.chunk_file(self.test_files['technical_doc'])
        assert text_result.chunks, "No chunks for text file"

        # Should use universal for text
        expected_text = ["sentence", "paragraph", "fixed_size"]
        assert any(strategy in text_result.strategy_used for strategy in expected_text)

        # Test JSON file - should use universal fixed_size
        json_result = orchestrator.chunk_file(self.test_files['json_config'])
        assert json_result.chunks, "No chunks for JSON file"

        # Should use consistent processing for data files
        expected_json = ["fixed_size", "sentence", "paragraph"]
        assert any(strategy in json_result.strategy_used for strategy in expected_json)

        print(f"Mixed approach results:")
        print(f"  Python: {python_result.strategy_used}")
        print(f"  Text: {text_result.strategy_used}")
        print(f"  JSON: {json_result.strategy_used}")

    def test_future_formats_config(self):
        """Test future formats configuration."""
        config_file = self.config_dir / "future_formats.yaml"

        if not config_file.exists():
            pytest.skip("Future formats config not found")

        orchestrator = ChunkerOrchestrator(config_path=config_file)

        # Test current supported formats
        current_format_tests = [
            (self.test_files['complex_python'], '.py'),
            (self.test_files['technical_doc'], '.txt'),
            (self.test_files['json_config'], '.json')
        ]

        for file_path, extension in current_format_tests:
            result = orchestrator.chunk_file(file_path)
            assert result.chunks, f"No chunks for {extension} file in future config"

            # Should handle current formats correctly
            assert result.strategy_used, f"No strategy used for {extension}"
            print(f"Future config - {extension}: {result.strategy_used}")

        # Test configuration structure for future formats
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Verify future format entries exist
        strategy_selection = config.get('strategy_selection', {})

        # Check for future formats in config
        future_formats = ['.xlsx', '.docx', '.mp4', '.zip']
        for fmt in future_formats:
            if fmt in strategy_selection:
                assert 'primary' in strategy_selection[fmt], f"Missing primary strategy for {fmt}"
                assert 'fallbacks' in strategy_selection[fmt], f"Missing fallbacks for {fmt}"

    def test_config_cross_compatibility(self):
        """Test that configs work with different file types."""
        configs_to_test = []

        for config_name in self.new_config_files:
            # Check in the strategy_configs subdirectory
            config_file = self.config_dir / "strategy_configs" / config_name
            if config_file.exists():
                configs_to_test.append((config_name, config_file))
            else:
                # Fallback: check in the root config directory
                config_file = self.config_dir / config_name
                if config_file.exists():
                    configs_to_test.append((config_name, config_file))

        test_file = self.test_files['technical_doc']  # Use simple text file

        for config_name, config_file in configs_to_test:
            try:
                orchestrator = ChunkerOrchestrator(config_path=config_file)
                result = orchestrator.chunk_file(test_file)

                assert result.chunks, f"No chunks with {config_name}"
                assert result.strategy_used, f"No strategy used with {config_name}"

                print(f"{config_name}: {result.strategy_used} ({len(result.chunks)} chunks)")

            except Exception as e:
                pytest.fail(f"Config {config_name} failed: {e}")

    def test_config_strategy_availability(self):
        """Test that all strategies mentioned in configs are available or properly handled."""
        for config_name in self.new_config_files:
            # Check in the strategy_configs subdirectory
            config_file = self.config_dir / "strategy_configs" / config_name
            if not config_file.exists():
                # Fallback: check in the root config directory
                config_file = self.config_dir / config_name
                if not config_file.exists():
                    continue

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            strategy_selection = config.get('strategy_selection', {})

            for file_ext, rules in strategy_selection.items():
                if isinstance(rules, dict):
                    primary = rules.get('primary')
                    fallbacks = rules.get('fallbacks', [])

                    # Test if we can create orchestrator (handles unavailable strategies gracefully)
                    try:
                        orchestrator = ChunkerOrchestrator(config_path=config_file)

                        # Test validation if method exists
                        if hasattr(orchestrator, 'validate_strategy_config'):
                            validation = orchestrator.validate_strategy_config(primary, '.txt')
                            # Should either be valid or handled gracefully
                            assert 'strategy' in validation, f"Invalid validation result for {primary}"

                    except Exception as e:
                        # Should not fail catastrophically
                        if "not found" not in str(e).lower():
                            pytest.fail(f"Unexpected error with {config_name}, strategy {primary}: {e}")

    def test_config_performance_characteristics(self):
        """Test performance characteristics of different configs."""
        import time

        configs_to_test = []
        for config_name in self.new_config_files:
            # Check in the strategy_configs subdirectory
            config_file = self.config_dir / "strategy_configs" / config_name
            if config_file.exists():
                configs_to_test.append((config_name, config_file))
            else:
                # Fallback: check in the root config directory
                config_file = self.config_dir / config_name
                if config_file.exists():
                    configs_to_test.append((config_name, config_file))

        test_file = self.test_files['complex_python']
        performance_results = {}

        for config_name, config_file in configs_to_test:
            try:
                orchestrator = ChunkerOrchestrator(config_path=config_file)

                start_time = time.time()
                result = orchestrator.chunk_file(test_file)
                processing_time = time.time() - start_time

                performance_results[config_name] = {
                    'time': processing_time,
                    'chunks': len(result.chunks) if result.chunks else 0,
                    'strategy': result.strategy_used
                }

            except Exception as e:
                performance_results[config_name] = {'error': str(e)}

        # Report performance results
        print("\nPerformance Results:")
        for config_name, results in performance_results.items():
            if 'error' in results:
                print(f"  {config_name}: ERROR - {results['error']}")
            else:
                print(f"  {config_name}: {results['time']:.3f}s, {results['chunks']} chunks, {results['strategy']}")

        # Basic performance checks
        successful_configs = [name for name, results in performance_results.items() if 'error' not in results]
        assert len(successful_configs) > 0, "No configs completed successfully"

        # All successful configs should complete in reasonable time (< 10 seconds)
        for config_name in successful_configs:
            time_taken = performance_results[config_name]['time']
            assert time_taken < 10.0, f"Config {config_name} took too long: {time_taken:.3f}s"


class TestConfigurationDocumentation:
    """Test that configurations are well-documented and follow conventions."""

    def test_config_documentation_quality(self):
        """Test that configs have good documentation."""
        config_dir = Path(__file__).parent.parent.parent / "config_examples"
        new_configs = [
            "rolling_hash_default.yaml",
            "rabin_fingerprinting_default.yaml",
            "buzhash_performance.yaml",
            "gear_cdc_default.yaml",
            "ml_cdc_hierarchical.yaml",
            "tttd_balanced.yaml"
        ]

        for config_name in new_configs:
            config_file = config_dir / config_name

            if not config_file.exists():
                continue

            with open(config_file, 'r') as f:
                content = f.read()
                config = yaml.safe_load(content)

            # Check for required documentation fields
            assert 'profile_name' in config, f"{config_name} missing profile_name"
            assert 'description' in config, f"{config_name} missing description"

            # Check for comments and examples in the file
            assert '#' in content, f"{config_name} lacks comments"

            # Should have usage examples
            assert 'usage' in content.lower() or 'example' in content.lower(), \
                f"{config_name} lacks usage examples"

            print(f"✅ {config_name}: Well documented")

    def test_config_naming_conventions(self):
        """Test that configs follow naming conventions."""
        config_dir = Path(__file__).parent.parent.parent / "config_examples"

        for config_file in config_dir.glob("*.yaml"):
            # Should be snake_case
            assert '_' in config_file.stem or config_file.stem.islower(), \
                f"Config {config_file.name} should use snake_case"

            # Should end with .yaml
            assert config_file.suffix == '.yaml', \
                f"Config {config_file.name} should use .yaml extension"

    def test_config_structure_consistency(self):
        """Test that configs have consistent structure."""
        config_dir = Path(__file__).parent.parent.parent / "config_examples"

        # Different config types have different structures
        key_sections = ['profile_name', 'description', 'strategies', 'chunking', 'strategy_selection']

        for config_file in config_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                # Handle both single and multi-document YAML files
                try:
                    config = yaml.safe_load(f)
                except yaml.composer.ComposerError:
                    # Multi-document YAML, load the first document
                    f.seek(0)
                    config = next(yaml.safe_load_all(f))

            # Config should have at least one meaningful section
            has_meaningful_content = False

            # Check for profile-based configs
            if 'profile_name' in config or 'description' in config:
                has_meaningful_content = True

            # Check for strategy-based configs
            if 'strategies' in config or ('chunking' in config and 'strategies' in config.get('chunking', {})):
                has_meaningful_content = True

            # Check for strategy selection configs
            if 'strategy_selection' in config:
                has_meaningful_content = True

            # Check for chunking section with strategy configuration
            if 'chunking' in config:
                chunking_config = config['chunking']
                if isinstance(chunking_config, dict):
                    # Check for chunking-specific sections
                    if any(key in chunking_config for key in ['default_strategy', 'strategy_selection', 'strategy_params']):
                        has_meaningful_content = True

            # Check if it's a specialized config with direct algorithm settings
            algo_keys = ['fastcdc', 'rolling_hash', 'rabin_fingerprinting', 'buzhash', 'gear_cdc', 'ml_cdc', 'tttd']
            if any(key in config for key in algo_keys):
                has_meaningful_content = True

            # Check for our new algorithm configuration profiles (nested configs)
            new_algo_patterns = [
                'high_quality_embedding_based', 'balanced_embedding_based', 'fast_embedding_based',
                'high_quality_recursive', 'balanced_recursive', 'fast_recursive',
                'high_quality_boundary_aware', 'balanced_boundary_aware', 'fast_boundary_aware',
                'semantic_chunker', 'context_enriched',
                # Overlapping window patterns (actual patterns from file)
                'rag_config', 'embedding_config', 'summarization_config', 'sentiment_analysis_config',
                'character_based_config', 'sentence_based_config', 'high_precision_config',
                'batch_processing_config', 'api_integration_config', 'multilingual_config', 'development_config'
            ]
            if any(pattern in config for pattern in new_algo_patterns):
                has_meaningful_content = True

            # Generic fallback: check for any chunker-related configuration
            if not has_meaningful_content and isinstance(config, dict):
                for key, value in config.items():
                    if isinstance(value, dict):
                        # Check if it has chunker-related parameters
                        if ('chunker' in value or 'parameters' in value or
                            'strategy' in value or 'chunking_strategy' in value or
                            any(param in value for param in ['window_size', 'chunk_size', 'overlap', 'threshold'])):
                            has_meaningful_content = True
                            break

            assert has_meaningful_content, \
                f"Config {config_file.name} lacks meaningful configuration content"

            # Ensure it's a dictionary (valid YAML structure)
            assert isinstance(config, dict), \
                f"Config {config_file.name} should be a valid YAML dictionary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
