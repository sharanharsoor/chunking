# Chunking Library Benchmarking System

The chunking library includes a production-ready benchmarking system designed for pip-installed environments. This system provides robust performance measurement capabilities with seamless custom algorithm integration and comprehensive error handling.

## 🚀 Key Features

- **Pip Install Ready**: No hardcoded paths, works in any environment
- **Custom Algorithm Support**: Seamless integration with built-in strategies
- **Multiple Output Formats**: Console, JSON, CSV, and text reports
- **Flexible Configuration**: CLI options, config files, and environment variables
- **Robust Error Handling**: Graceful fallbacks and isolated failures

## 📋 Quick Start

### Basic Usage

```bash
# Benchmark built-in strategies
python -m chunking_strategy benchmark my_document.txt

# Quick benchmark (faster, fewer strategies)
python -m chunking_strategy benchmark --quick my_document.txt

# Specify strategies to test
python -m chunking_strategy benchmark --strategies "fixed_size,sentence_based" my_file.txt

# Include custom algorithms
python -m chunking_strategy benchmark --custom-algorithms ./my_chunker.py my_document.txt
```

### Custom Algorithm Benchmarking

```bash
# Benchmark your custom algorithm
python -m chunking_strategy custom benchmark my_custom_chunker.py

# Compare with specific built-in strategies
python -m chunking_strategy custom benchmark my_custom_chunker.py \
    --compare-with fixed_size --compare-with sentence_based

# Specify output directory
python -m chunking_strategy custom benchmark my_custom_chunker.py --output-dir ./my_results
```

### Python API

```python
from chunking_strategy.core.production_benchmark import run_quick_benchmark
from pathlib import Path

# Quick benchmark
suite = run_quick_benchmark(
    content="Your test content here",
    strategies=["fixed_size", "sentence_based"],
    output_dir=Path("./results")
)

print(f"Success rate: {suite.summary_stats['success_rate']:.1%}")
```

## ⚙️ Configuration

### Output Directory

By default, results are saved to `chunking_benchmarks/` in your current directory:

```bash
# CLI option
python -m chunking_strategy benchmark --output-dir ./my_results

# Environment variable
export CHUNKING_BENCHMARK_OUTPUT_DIR=/path/to/results

# Configuration file (config_examples/benchmarking_configs/benchmark_config.yaml)
output_dir: "./my_benchmark_results"
```

### Output Formats Control

```bash
# Disable specific outputs
python -m chunking_strategy benchmark --no-console  # Skip console summary
python -m chunking_strategy benchmark --no-json     # Skip JSON output
python -m chunking_strategy benchmark --no-csv      # Skip CSV output
```

### Environment Variables

```bash
export CHUNKING_BENCHMARK_OUTPUT_DIR=/path/to/results
export CHUNKING_BENCHMARK_RUNS=5
export CHUNKING_BENCHMARK_CONSOLE=false
```

### Configuration File

Create a YAML configuration file for complex setups:

```yaml
# config_examples/benchmark_config.yaml
output_dir: "./my_benchmark_results"
console_summary: true
save_json: true
save_csv: true
runs_per_strategy: 3
# custom_algorithm_paths:  # Uncomment and modify as needed
#   - "./my_custom_chunker.py"
```

## 🔧 Custom Algorithms

### Requirements

Your custom algorithm must:

1. Inherit from `BaseChunker`
2. Use proper `@register_chunker` decoration
3. Implement required methods

Example structure:

```python
from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult
from chunking_strategy.core.registry import register_chunker

@register_chunker(name="my_custom_chunker", category="text")
class MyCustomChunker(BaseChunker):
    def __init__(self, **kwargs):
        super().__init__(name="my_custom_chunker", category="text", **kwargs)

    def chunk(self, content, **kwargs):
        # Your chunking logic here
        pass
```

### Including Multiple Custom Algorithms

```bash
python -m chunking_strategy benchmark \
    --custom-algorithms ./chunker1.py \
    --custom-algorithms ./chunker2.py \
    --custom-algorithms ./examples/custom_algorithms/balanced_length_chunker.py
```

## 📊 Output Examples

### Console Summary
```
🏁 BENCHMARK RESULTS: comprehensive_benchmark
================================================================================
⏰ Completed: 2024-09-16 14:30:22
📁 Results saved to: /home/user/chunking_benchmarks
📊 Total results: 6 | ✅ Success rate: 100.0%

🚀 TOP PERFORMERS
⚡ Fastest: my_custom_chunker (0.023s) 🔧
🏆 Best quality: balanced_length (score: 0.847) 🔧

📈 STRATEGY COMPARISON
Strategy              Avg Time (s)  Quality    Custom
-----------------------------------------------------
my_custom_chunker     0.023         0.820      🔧
fixed_size            0.031         0.750
sentence_based        0.028         0.780
balanced_length       0.035         0.847      🔧
```

### File Outputs

Results are saved with timestamped filenames:

```
chunking_benchmarks/
├── benchmark_20240916_143022.json        # Detailed structured data
├── benchmark_20240916_143022.csv         # Summary spreadsheet
└── benchmark_20240916_143022_report.txt  # Human-readable report
```

## 📈 Advanced Usage

### Comprehensive Benchmarking

```python
from chunking_strategy.core.production_benchmark import (
    ProductionBenchmarkRunner, ProductionBenchmarkConfig
)
from pathlib import Path

# Create custom configuration
config = ProductionBenchmarkConfig(
    output_dir=Path("./detailed_results"),
    runs_per_strategy=5,
    custom_algorithm_paths=[Path("./my_chunker.py")]
)

runner = ProductionBenchmarkRunner(config)

# Define test scenarios
strategies = ["fixed_size", "sentence_based", "my_chunker"]
datasets = {
    "short_text": "Brief content for testing.",
    "long_text": Path("./long_document.txt"),
    "technical_doc": Path("./technical.md")
}

# Run comprehensive benchmark
suite = runner.run_comprehensive_benchmark(
    strategies=strategies,
    datasets=datasets,
    suite_name="production_analysis"
)

print(f"Results saved to: {config.output_dir}")
```

### Custom Algorithm Comparison

```python
from chunking_strategy.core.production_benchmark import run_custom_algorithm_benchmark

suite = run_custom_algorithm_benchmark(
    custom_algorithm_path=Path("./my_chunker.py"),
    compare_with=["fixed_size", "sentence_based"],
    test_content="Your test content",
    output_dir=Path("./results")
)

# Find performance metrics
successful = [r for r in suite.results if r.success]
fastest = min(successful, key=lambda r: r.processing_time)
print(f"Fastest: {fastest.strategy_name} ({fastest.processing_time:.3f}s)")
```

### Batch Testing

```python
from pathlib import Path

algorithms_dir = Path("./my_algorithms")
results = {}

for algo_file in algorithms_dir.glob("*.py"):
    try:
        suite = run_custom_algorithm_benchmark(
            custom_algorithm_path=algo_file,
            compare_with=["fixed_size"],
            output_dir=Path(f"./results/{algo_file.stem}")
        )
        results[algo_file.name] = suite.summary_stats
    except Exception as e:
        print(f"Failed to benchmark {algo_file}: {e}")

# Compare all algorithms
for name, stats in results.items():
    print(f"{name}: {stats['success_rate']:.1%} success, "
          f"{stats['avg_processing_time']:.3f}s avg time")
```

## 🛠️ Error Handling

The system handles errors gracefully without stopping benchmarks:

### Directory Permissions
```
WARNING: Cannot write to /restricted/path, using fallback: /tmp/chunking_benchmarks
```

### Missing Strategies
```
WARNING: Strategy 'nonexistent_strategy' not found, skipping
```

### Custom Algorithm Failures
```
WARNING: Failed to load custom algorithm from ./broken_chunker.py: ImportError
```

Individual failures are recorded in results with `success: false`.

## 🚀 CI/CD Integration

Perfect for automated testing:

```bash
# Silent benchmark with structured output
export CHUNKING_BENCHMARK_CONSOLE=false
export CHUNKING_BENCHMARK_OUTPUT_DIR=./ci_results

python -m chunking_strategy benchmark --strategies "fixed_size,sentence_based" test_data.txt

# Process results programmatically
python analyze_benchmark_results.py ./ci_results/*.json
```

GitHub Actions example:

```yaml
- name: Benchmark Performance
  run: |
    export CHUNKING_BENCHMARK_CONSOLE=false
    python -m chunking_strategy benchmark --quick test_data.txt
    python scripts/analyze_results.py ./chunking_benchmarks/*.json
```

## 🎯 Performance Optimization

### For Large Files
```bash
# Fast benchmark for large files
python -m chunking_strategy benchmark --quick --runs 1 large_file.txt

# Limit strategies
python -m chunking_strategy benchmark --strategies "fixed_size,sentence_based" large_file.txt
```

### Memory Tracking

Install `psutil` for memory usage tracking:

```bash
pip install psutil  # Enables memory tracking
```

## 🧪 Demo and Testing

### Run the Demo
```bash
python examples/benchmark_demo.py
```

### Test Configuration Files
```bash
# Test with basic benchmark (no separate config tester needed)
echo "Test content" > test.txt && python -m chunking_strategy benchmark --quick test.txt && rm test.txt

# Validate custom algorithms work
python -m chunking_strategy benchmark --custom-algorithms examples/custom_algorithms/balanced_length_chunker.py --quick
```

## 📚 JSON Output Structure

```json
{
  "name": "comprehensive_benchmark",
  "timestamp": 1694871022.0,
  "system_info": {
    "platform": "Linux",
    "python_version": "3.11.0",
    "output_directory": "/path/to/chunking_benchmarks"
  },
  "results": [
    {
      "strategy_name": "my_custom_chunker",
      "dataset_name": "test_content",
      "processing_time": 0.023,
      "chunk_count": 12,
      "quality_metrics": {"overall_score": 0.82},
      "success": true,
      "is_custom_algorithm": true,
      "custom_algorithm_path": "./my_chunker.py"
    }
  ],
  "summary_stats": {
    "success_rate": 1.0,
    "custom_algorithm_count": 2,
    "avg_processing_time": 0.029,
    "strategies_tested": 4
  }
}
```

## 🎯 Best Practices

1. **Start with Quick Benchmarks** - Use `--quick` for initial testing
2. **Test Custom Algorithms Individually** - Validate before batch testing
3. **Use Representative Data** - Match test content to your use case
4. **Save Important Results** - Always specify output directory for important benchmarks
5. **Version Control Configurations** - Include benchmark configs in your repository
6. **Monitor Over Time** - Track performance changes in CI/CD pipelines

## 🔍 Troubleshooting

### Common Issues

1. **Custom Algorithm Not Found**: Ensure proper `@register_chunker` decoration
2. **Permission Errors**: System automatically uses fallback directories
3. **Import Errors**: Check Python path and dependencies in custom algorithms
4. **Memory Issues**: Reduce dataset size or number of runs

### Debug Output

Enable verbose logging:

```bash
python -m chunking_strategy --verbose benchmark my_file.txt
```

### Validate Setup

Test with built-in examples:

```bash
python -m chunking_strategy benchmark \
    --custom-algorithms examples/custom_algorithms/balanced_length_chunker.py \
    --strategies "fixed_size,balanced_length" \
    --quick
```

The production benchmarking system ensures consistent, reliable performance measurement for all users, whether they're using built-in algorithms, developing custom solutions, or integrating the library into larger systems.
