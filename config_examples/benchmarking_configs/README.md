# Benchmarking Configuration Examples

This directory contains configuration files for the **benchmarking system** (not chunking strategies).

## 📁 Files

- **`benchmark_config.yaml`** - Example benchmarking configuration showing all options

## 🚀 How to Test Benchmark Configurations

### Method 1: CLI Testing (Recommended)

```bash
# Test with a simple benchmark using the config settings
python -m chunking_strategy benchmark --strategies "fixed_size" --runs 1 --output-dir ./test_output test_file.txt

# The config file can be referenced for setting up these parameters
```

### Method 2: Python API Testing

```python
import yaml
from pathlib import Path
from chunking_strategy.core.production_benchmark import ProductionBenchmarkConfig

# Load and validate the config
config_path = Path("config_examples/benchmarking_configs/benchmark_config.yaml")
with open(config_path) as f:
    config_data = yaml.safe_load(f)

# Test that it creates a valid benchmark config
benchmark_config = ProductionBenchmarkConfig(**config_data)
print(f"Config loaded: {benchmark_config.output_dir}")
```

## 🔍 Key Differences from Chunking Strategy Configs

| Aspect | Benchmarking Configs | Chunking Strategy Configs |
|--------|---------------------|--------------------------|
| **Purpose** | Configure how benchmarks run | Configure how text is chunked |
| **Used by** | `ProductionBenchmarkRunner` | `ChunkerOrchestrator` |
| **Fields** | `output_dir`, `runs_per_strategy`, `save_json` | `strategy`, `chunk_size`, `file_types` |
| **Testing** | CLI benchmark commands | `test_single_config.py` |

## 📊 Example Usage

```bash
# Create a test file
echo "Sample text for benchmarking" > test.txt

# Run benchmark with config-inspired settings
python -m chunking_strategy benchmark \
    --strategies "fixed_size,sentence_based" \
    --runs 3 \
    --output-dir ./my_benchmark_results \
    test.txt

# Results will be saved in JSON, CSV, and text formats
ls ./my_benchmark_results/
```

## 🎯 Configuration Fields Explained

```yaml
# Output Configuration
output_dir: "./my_benchmark_results"  # Where to save results
console_summary: true                 # Show summary in terminal
save_json: true                      # Save detailed JSON results
save_csv: true                       # Save CSV summary
save_report: true                    # Save text report

# Performance Settings
runs_per_strategy: 3                 # Number of benchmark runs
include_system_info: true           # Include system specs

# Custom Algorithms (optional)
custom_algorithm_paths:              # Paths to custom algorithms
  - "./my_custom_chunker.py"
```

For more details, see the main **[BENCHMARKING.md](../../BENCHMARKING.md)** documentation.
