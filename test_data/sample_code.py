#!/usr/bin/env python3
"""
Sample Python code for testing code chunking strategies.

This file demonstrates various Python constructs that should be properly
chunked by the Python code chunker.
"""

import os
import sys
from typing import List, Dict, Optional
from pathlib import Path


def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.

    Args:
        n: Position in the Fibonacci sequence

    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)


class DataProcessor:
    """A class for processing data with various methods."""

    def __init__(self, data_source: str):
        """Initialize the data processor."""
        self.data_source = data_source
        self.processed_data = []

    def load_data(self) -> List[Dict]:
        """Load data from the source."""
        # Simulate loading data
        data = [
            {"id": 1, "name": "Alice", "score": 95},
            {"id": 2, "name": "Bob", "score": 87},
            {"id": 3, "name": "Charlie", "score": 92}
        ]
        return data

    def process_data(self, data: List[Dict]) -> List[Dict]:
        """Process the loaded data."""
        processed = []
        for item in data:
            if item["score"] > 90:
                item["grade"] = "A"
            elif item["score"] > 80:
                item["grade"] = "B"
            else:
                item["grade"] = "C"
            processed.append(item)
        return processed

    def save_results(self, results: List[Dict], output_path: str) -> None:
        """Save processed results to file."""
        with open(output_path, 'w') as f:
            for result in results:
                f.write(f"{result}\n")


def main():
    """Main execution function."""
    print("Starting data processing...")

    processor = DataProcessor("sample_data.json")
    raw_data = processor.load_data()
    processed_data = processor.process_data(raw_data)
    processor.save_results(processed_data, "output.txt")

    print("Processing complete!")

    # Test Fibonacci
    fib_result = calculate_fibonacci(10)
    print(f"10th Fibonacci number: {fib_result}")


if __name__ == "__main__":
    main()