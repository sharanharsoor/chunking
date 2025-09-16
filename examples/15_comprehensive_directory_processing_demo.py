#!/usr/bin/env python3
"""
Comprehensive Directory Processing Demo

This demo showcases the complete directory processing capabilities:
- Directory-level processing with file path printing
- Multiple chunking strategies for different file types
- Parallel processing with different modes
- Config-driven directory processing
- CLI integration examples
- Chunk content preview
"""

import logging
import tempfile
import time
import subprocess
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from chunking_strategy import ChunkerOrchestrator
from chunking_strategy.core.batch import BatchProcessor


def create_demo_directory() -> Path:
    """Create a comprehensive demo directory with various file types."""
    demo_dir = Path(tempfile.mkdtemp(prefix="chunking_demo_dir_"))

    # Create directory structure
    (demo_dir / "documents").mkdir()
    (demo_dir / "code").mkdir()
    (demo_dir / "data").mkdir()
    (demo_dir / "configs").mkdir()
    (demo_dir / "web").mkdir()

    # Create diverse files with different content types
    files_to_create = [
        # Text documents - will use sentence_based strategy
        ("documents/business_report.txt", """
Executive Summary

Our quarterly revenue increased by 25% compared to last quarter. The growth was driven primarily by our new product launches and expanded market presence. Customer satisfaction scores have improved significantly, reaching 92% positive feedback.

Market Performance Analysis

The technology sector showed strong performance this quarter. Our main competitors saw average growth of 15%, while we achieved 25% growth. This indicates our strategic initiatives are working effectively.

Key Metrics

Revenue: $2.5M (up 25%)
Customer Acquisition: 1,200 new customers
Retention Rate: 94%
Employee Satisfaction: 88%

Future Outlook

Based on current trends and market analysis, we project continued growth in the next quarter. The launch of our AI-powered features is expected to drive additional revenue streams.
        """),

        # Markdown document - will use markdown_chunker strategy
        ("documents/project_documentation.md", """
# Project Documentation

## Overview
This project implements advanced text chunking algorithms for document processing.

## Features
- **Multi-Strategy Support**: Over 35 different chunking strategies
- **Parallel Processing**: Thread and process-based parallelization
- **Format Support**: Text, code, documents, multimedia files
- **Hardware Optimization**: Automatic CPU and memory optimization

## Installation
```bash
pip install chunking-strategy
```

## Quick Start
```python
from chunking_strategy import ChunkerOrchestrator

orchestrator = ChunkerOrchestrator()
result = orchestrator.chunk_file("document.txt")
```

## Advanced Usage
### Configuration Files
You can use YAML configuration files to customize chunking behavior:

```yaml
strategies:
  primary: sentence_based
  fallbacks: [paragraph_based, fixed_size]
```

### CLI Usage
Process directories with the command line:
```bash
chunking-strategy process-directory /path/to/docs
```

## API Reference
### ChunkerOrchestrator
Main class for orchestrating chunking operations.

#### Methods
- `chunk_file(path)` - Chunk a single file
- `chunk_files_batch(files)` - Chunk multiple files
        """),

        # Python code - will use python_code strategy
        ("code/data_processor.py", '''
"""
Advanced data processing module for chunking operations.

This module provides utilities for processing various data formats
and optimizing chunking performance.
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path


class DataProcessor:
    """Main data processing class with chunking utilities."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize processor with optional configuration."""
        self.config = config or {}
        self.processed_files = []
        self.performance_stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'processing_time': 0.0
        }

    def process_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a text file and return chunks with metadata."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = file_path.read_text(encoding='utf-8')
        chunks = self._create_text_chunks(content)

        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                'chunk_id': i,
                'content': chunk,
                'length': len(chunk),
                'file_source': str(file_path)
            })

        self.performance_stats['files_processed'] += 1
        self.performance_stats['chunks_created'] += len(result)

        return result

    def _create_text_chunks(self, content: str, max_size: int = 1000) -> List[str]:
        """Create text chunks with specified maximum size."""
        words = content.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word) + 1  # +1 for space

            if current_size + word_size > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process_json_data(self, json_path: Path) -> Dict[str, Any]:
        """Process JSON file and extract structured information."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        return self._analyze_json_structure(data)

    def _analyze_json_structure(self, data: Any, level: int = 0) -> Dict[str, Any]:
        """Analyze JSON structure recursively."""
        analysis = {
            'type': type(data).__name__,
            'level': level
        }

        if isinstance(data, dict):
            analysis['keys'] = list(data.keys())
            analysis['size'] = len(data)
        elif isinstance(data, list):
            analysis['length'] = len(data)
            analysis['item_types'] = list(set(type(item).__name__ for item in data))

        return analysis

    def generate_report(self) -> str:
        """Generate processing performance report."""
        stats = self.performance_stats
        report = f"""
Processing Report
================
Files Processed: {stats['files_processed']}
Chunks Created: {stats['chunks_created']}
Average Chunks per File: {stats['chunks_created'] / max(stats['files_processed'], 1):.1f}
Processing Time: {stats['processing_time']:.2f}s
        """
        return report.strip()


def main():
    """Main function demonstrating data processing capabilities."""
    processor = DataProcessor()

    # Example usage
    sample_text = "This is a sample text for demonstration purposes. " * 50
    temp_file = Path("/tmp/sample.txt")
    temp_file.write_text(sample_text)

    try:
        result = processor.process_text_file(temp_file)
        print(f"Processed {len(result)} chunks from sample file")
        print(processor.generate_report())
    finally:
        if temp_file.exists():
            temp_file.unlink()


if __name__ == "__main__":
    main()
        '''),

        # JavaScript code - will use javascript_code strategy
        ("code/frontend_utils.js", '''
/**
 * Frontend utilities for chunking visualization and interaction
 * Provides interactive components for displaying chunked content
 */

class ChunkVisualizer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            maxChunksVisible: 50,
            chunkPreviewLength: 200,
            highlightSimilarity: true,
            ...options
        };
        this.chunks = [];
        this.currentFilter = null;
    }

    /**
     * Load and display chunks from JSON data
     * @param {Array} chunksData - Array of chunk objects
     */
    loadChunks(chunksData) {
        this.chunks = chunksData;
        this.renderChunks();
        this.setupEventListeners();
    }

    /**
     * Render chunks in the container
     */
    renderChunks() {
        if (!this.container) return;

        const chunksToShow = this.currentFilter
            ? this.chunks.filter(this.currentFilter)
            : this.chunks.slice(0, this.options.maxChunksVisible);

        this.container.innerHTML = chunksToShow.map((chunk, index) => `
            <div class="chunk-item" data-chunk-id="${chunk.id || index}">
                <div class="chunk-header">
                    <span class="chunk-id">Chunk ${index + 1}</span>
                    <span class="chunk-length">${chunk.content?.length || 0} chars</span>
                    <span class="chunk-strategy">${chunk.strategy || 'unknown'}</span>
                </div>
                <div class="chunk-content">
                    ${this.formatContent(chunk.content)}
                </div>
                <div class="chunk-metadata">
                    <small>Source: ${chunk.source || 'unknown'}</small>
                </div>
            </div>
        `).join('');
    }

    /**
     * Format chunk content with preview truncation
     * @param {string} content - Raw chunk content
     * @returns {string} Formatted HTML content
     */
    formatContent(content) {
        if (!content) return '<em>No content</em>';

        const previewLength = this.options.chunkPreviewLength;
        const truncated = content.length > previewLength
            ? content.substring(0, previewLength) + '...'
            : content;

        return `<pre>${this.escapeHtml(truncated)}</pre>`;
    }

    /**
     * Escape HTML characters
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Set up event listeners for interactive features
     */
    setupEventListeners() {
        // Add click handlers for chunk expansion
        this.container.addEventListener('click', (e) => {
            const chunkItem = e.target.closest('.chunk-item');
            if (chunkItem) {
                this.toggleChunkExpansion(chunkItem);
            }
        });

        // Add search functionality
        const searchInput = document.getElementById('chunk-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.filterChunks(e.target.value);
            });
        }
    }

    /**
     * Toggle expansion of a chunk item
     * @param {Element} chunkItem - Chunk DOM element
     */
    toggleChunkExpansion(chunkItem) {
        const content = chunkItem.querySelector('.chunk-content pre');
        const chunkId = chunkItem.dataset.chunkId;
        const chunk = this.chunks[chunkId];

        if (chunkItem.classList.contains('expanded')) {
            content.textContent = this.formatContent(chunk.content);
            chunkItem.classList.remove('expanded');
        } else {
            content.textContent = chunk.content || '';
            chunkItem.classList.add('expanded');
        }
    }

    /**
     * Filter chunks based on search term
     * @param {string} searchTerm - Search term to filter by
     */
    filterChunks(searchTerm) {
        if (!searchTerm.trim()) {
            this.currentFilter = null;
        } else {
            this.currentFilter = chunk =>
                chunk.content?.toLowerCase().includes(searchTerm.toLowerCase());
        }
        this.renderChunks();
    }

    /**
     * Export chunks as JSON
     * @returns {string} JSON string of chunks
     */
    exportChunks() {
        return JSON.stringify(this.chunks, null, 2);
    }

    /**
     * Get processing statistics
     * @returns {Object} Statistics object
     */
    getStats() {
        const totalChunks = this.chunks.length;
        const totalLength = this.chunks.reduce((sum, chunk) => sum + (chunk.content?.length || 0), 0);
        const avgLength = totalLength / totalChunks || 0;

        return {
            totalChunks,
            totalLength,
            averageChunkLength: Math.round(avgLength),
            strategies: [...new Set(this.chunks.map(chunk => chunk.strategy))].filter(Boolean)
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChunkVisualizer;
}
        '''),

        # JSON data - will use json_chunker strategy
        ("data/sample_dataset.json", '''{
    "metadata": {
        "dataset_name": "Customer Analysis Dataset",
        "version": "2.1.0",
        "created": "2024-01-15T10:30:00Z",
        "records_count": 1000,
        "schema_version": "1.2"
    },
    "customers": [
        {
            "customer_id": "CUST_001",
            "profile": {
                "name": "Alice Johnson",
                "age": 32,
                "location": {
                    "city": "San Francisco",
                    "state": "CA",
                    "country": "USA",
                    "coordinates": {
                        "lat": 37.7749,
                        "lng": -122.4194
                    }
                },
                "contact": {
                    "email": "alice.johnson@email.com",
                    "phone": "+1-555-123-4567",
                    "preferred_contact": "email"
                }
            },
            "purchase_history": [
                {
                    "order_id": "ORD_20240101_001",
                    "date": "2024-01-01T14:30:00Z",
                    "items": [
                        {"product_id": "PROD_A1", "name": "Wireless Headphones", "quantity": 1, "price": 199.99},
                        {"product_id": "PROD_B2", "name": "Phone Case", "quantity": 2, "price": 29.99}
                    ],
                    "total": 259.97,
                    "payment_method": "credit_card",
                    "shipping_address": "123 Main St, San Francisco, CA 94102"
                },
                {
                    "order_id": "ORD_20240115_002",
                    "date": "2024-01-15T09:15:00Z",
                    "items": [
                        {"product_id": "PROD_C3", "name": "Bluetooth Speaker", "quantity": 1, "price": 89.99}
                    ],
                    "total": 89.99,
                    "payment_method": "paypal",
                    "shipping_address": "123 Main St, San Francisco, CA 94102"
                }
            ],
            "preferences": {
                "categories": ["electronics", "accessories"],
                "brands": ["TechPro", "SoundMax", "PhoneGuard"],
                "price_range": {"min": 25, "max": 300},
                "notification_settings": {
                    "email_promotions": true,
                    "sms_alerts": false,
                    "push_notifications": true
                }
            },
            "analytics": {
                "customer_lifetime_value": 349.96,
                "average_order_value": 174.98,
                "purchase_frequency": "monthly",
                "satisfaction_score": 4.8,
                "churn_risk": "low"
            }
        },
        {
            "customer_id": "CUST_002",
            "profile": {
                "name": "Bob Smith",
                "age": 28,
                "location": {
                    "city": "New York",
                    "state": "NY",
                    "country": "USA",
                    "coordinates": {
                        "lat": 40.7128,
                        "lng": -74.0060
                    }
                },
                "contact": {
                    "email": "bob.smith@email.com",
                    "phone": "+1-555-987-6543",
                    "preferred_contact": "phone"
                }
            },
            "purchase_history": [
                {
                    "order_id": "ORD_20240105_003",
                    "date": "2024-01-05T16:45:00Z",
                    "items": [
                        {"product_id": "PROD_D4", "name": "Laptop Stand", "quantity": 1, "price": 79.99},
                        {"product_id": "PROD_E5", "name": "USB-C Hub", "quantity": 1, "price": 49.99},
                        {"product_id": "PROD_F6", "name": "Wireless Mouse", "quantity": 1, "price": 39.99}
                    ],
                    "total": 169.97,
                    "payment_method": "debit_card",
                    "shipping_address": "456 Broadway, New York, NY 10013"
                }
            ],
            "preferences": {
                "categories": ["office", "electronics", "productivity"],
                "brands": ["DeskPro", "HubMax", "MouseTech"],
                "price_range": {"min": 20, "max": 150},
                "notification_settings": {
                    "email_promotions": false,
                    "sms_alerts": true,
                    "push_notifications": false
                }
            },
            "analytics": {
                "customer_lifetime_value": 169.97,
                "average_order_value": 169.97,
                "purchase_frequency": "quarterly",
                "satisfaction_score": 4.2,
                "churn_risk": "medium"
            }
        }
    ],
    "summary_statistics": {
        "total_revenue": 519.93,
        "average_customer_value": 259.97,
        "most_popular_category": "electronics",
        "geographic_distribution": {
            "CA": 1,
            "NY": 1
        },
        "payment_methods": {
            "credit_card": 1,
            "paypal": 1,
            "debit_card": 1
        }
    }
}'''),

        # CSV data - will use csv_chunker strategy
        ("data/sales_metrics.csv", '''date,region,product_category,sales_amount,units_sold,customer_count,avg_order_value
2024-01-01,North,Electronics,15000.50,125,87,172.42
2024-01-01,South,Electronics,12500.75,98,65,192.32
2024-01-01,East,Electronics,18750.25,156,102,183.83
2024-01-01,West,Electronics,22100.00,189,134,164.93
2024-01-02,North,Clothing,8900.25,234,156,57.05
2024-01-02,South,Clothing,7650.50,198,124,61.69
2024-01-02,East,Clothing,9870.75,267,178,55.45
2024-01-02,West,Clothing,11250.00,298,201,55.97
2024-01-03,North,Books,3450.75,145,98,35.21
2024-01-03,South,Books,2980.50,128,89,33.49
2024-01-03,East,Books,4125.25,172,115,35.87
2024-01-03,West,Books,4890.00,201,138,35.43
2024-01-04,North,Home,12700.50,89,67,189.71
2024-01-04,South,Home,10950.75,76,58,188.81
2024-01-04,East,Home,14280.25,102,78,183.08
2024-01-04,West,Home,16450.00,118,92,178.80
2024-01-05,North,Sports,6780.25,156,112,60.54
2024-01-05,South,Sports,5890.50,134,97,60.73
2024-01-05,East,Sports,7650.75,176,128,59.77
2024-01-05,West,Sports,8920.00,203,148,60.27
2024-01-06,North,Electronics,16800.75,142,95,177.01
2024-01-06,South,Electronics,13750.50,108,72,190.98
2024-01-06,East,Electronics,19500.25,167,108,180.56
2024-01-06,West,Electronics,23100.00,198,142,162.68
2024-01-07,North,Clothing,9250.25,243,162,57.10
2024-01-07,South,Clothing,8100.50,211,139,58.28
2024-01-07,East,Clothing,10450.75,284,189,55.29
2024-01-07,West,Clothing,12100.00,321,218,55.50'''),

        # CSS file - will use fixed_size strategy
        ("web/styles.css", '''
/* Main application styles for chunking visualization */
:root {
    --primary-color: #2563eb;
    --secondary-color: #64748b;
    --success-color: #059669;
    --warning-color: #d97706;
    --error-color: #dc2626;
    --background-color: #f8fafc;
    --surface-color: #ffffff;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: var(--text-primary);
    background-color: var(--background-color);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

.header {
    background: var(--surface-color);
    border-bottom: 1px solid var(--border-color);
    padding: 1rem 0;
    box-shadow: var(--shadow-sm);
}

.header h1 {
    color: var(--primary-color);
    font-size: 1.875rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.header p {
    color: var(--text-secondary);
    font-size: 1rem;
}

.main-content {
    padding: 2rem 0;
    display: grid;
    gap: 2rem;
}

.chunk-visualizer {
    background: var(--surface-color);
    border-radius: 0.5rem;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}

.chunk-controls {
    padding: 1.5rem;
    background: #f1f5f9;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
}

.search-input {
    flex: 1;
    min-width: 200px;
    padding: 0.5rem 1rem;
    border: 1px solid var(--border-color);
    border-radius: 0.375rem;
    font-size: 0.875rem;
    transition: border-color 0.2s;
}

.search-input:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgb(37 99 235 / 0.1);
}

.btn {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.btn-primary {
    background: var(--primary-color);
    color: white;
}

.btn-primary:hover {
    background: #1d4ed8;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.btn-secondary {
    background: var(--secondary-color);
    color: white;
}

.btn-secondary:hover {
    background: #475569;
}

.chunk-grid {
    padding: 1.5rem;
    display: grid;
    gap: 1rem;
}

.chunk-item {
    border: 1px solid var(--border-color);
    border-radius: 0.375rem;
    padding: 1rem;
    background: var(--surface-color);
    transition: all 0.2s;
    cursor: pointer;
}

.chunk-item:hover {
    border-color: var(--primary-color);
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.chunk-item.expanded {
    border-color: var(--success-color);
    background: #f0fdf4;
}

.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.chunk-id {
    font-weight: 600;
    color: var(--primary-color);
    font-size: 0.875rem;
}

.chunk-length, .chunk-strategy {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-weight: 500;
}

.chunk-length {
    background: #ddd6fe;
    color: #5b21b6;
}

.chunk-strategy {
    background: #fef3c7;
    color: #92400e;
}

.chunk-content {
    margin-bottom: 0.75rem;
}

.chunk-content pre {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 0.25rem;
    padding: 0.75rem;
    font-size: 0.875rem;
    line-height: 1.4;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

.chunk-metadata {
    font-size: 0.75rem;
    color: var(--text-secondary);
    padding-top: 0.5rem;
    border-top: 1px solid var(--border-color);
}

.stats-panel {
    background: var(--surface-color);
    border-radius: 0.5rem;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-sm);
    padding: 1.5rem;
}

.stats-panel h3 {
    color: var(--text-primary);
    font-size: 1.125rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
}

.stat-item {
    text-align: center;
    padding: 1rem;
    background: #f8fafc;
    border-radius: 0.375rem;
}

.stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary-color);
    display: block;
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

.loading {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 3rem;
    color: var(--text-secondary);
}

.loading::after {
    content: '';
    width: 1rem;
    height: 1rem;
    margin-left: 0.5rem;
    border: 2px solid var(--border-color);
    border-top-color: var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

@media (max-width: 768px) {
    .container {
        padding: 0 1rem;
    }

    .chunk-controls {
        flex-direction: column;
        align-items: stretch;
    }

    .search-input {
        min-width: auto;
    }

    .chunk-header {
        flex-direction: column;
        align-items: flex-start;
    }

    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
        '''),

        # Valid config file with correct strategy names
        ("configs/directory_config.yaml", yaml.dump({
            'profile_name': 'comprehensive_directory_demo',
            'strategies': {
                'primary': 'auto',
                'fallbacks': ['sentence_based', 'paragraph_based', 'fixed_size'],
                'configs': {
                    'sentence_based': {'max_sentences': 3, 'overlap': 0},
                    'paragraph_based': {'max_paragraphs': 2, 'preserve_structure': True},
                    'fixed_size': {'chunk_size': 800, 'overlap_size': 100},
                    'python_code': {'preserve_functions': True, 'preserve_classes': True},
                    'javascript_code': {'preserve_functions': True, 'preserve_classes': True},
                    'markdown_chunker': {'preserve_headers': True, 'preserve_structure': True},
                    'json_chunker': {'preserve_structure': True, 'max_depth': 3},
                    'css_code': {'preserve_selectors': True, 'preserve_rules': True},
                    'csv_chunker': {'header_chunks': True, 'max_rows_per_chunk': 10}
                }
            },
            'strategy_selection': {
                '.txt': 'sentence_based',
                '.md': 'markdown_chunker',
                '.py': 'python_code',
                '.js': 'javascript_code',
                '.json': 'json_chunker',
                '.csv': 'csv_chunker',
                '.css': 'css_code'
            },
            'preprocessing': {'enabled': True, 'normalize_whitespace': True},
            'postprocessing': {'enabled': True, 'merge_short_chunks': True, 'min_chunk_size': 50}
        }))
    ]

    # Create all files
    for file_path, content in files_to_create:
        full_path = demo_dir / file_path
        full_path.write_text(content.strip())

    logger.info(f"Created demo directory: {demo_dir}")
    logger.info(f"Directory structure:")
    for path in sorted(demo_dir.rglob("*")):
        if path.is_file():
            relative_path = path.relative_to(demo_dir)
            size = path.stat().st_size
            logger.info(f"  {relative_path} ({size:,} bytes)")

    return demo_dir


def demonstrate_basic_directory_processing(demo_dir: Path):
    """Demonstrate basic directory processing with different strategies."""

    logger.info("\n" + "="*60)
    logger.info("DEMO 1: BASIC DIRECTORY PROCESSING WITH DIFFERENT STRATEGIES")
    logger.info("="*60)

    # Create orchestrator with default configuration
    orchestrator = ChunkerOrchestrator()

    # Get files organized by type
    file_types = {
        'Text Documents (.txt)': list(demo_dir.glob("**/*.txt")),
        'Markdown (.md)': list(demo_dir.glob("**/*.md")),
        'Python Code (.py)': list(demo_dir.glob("**/*.py")),
        'JavaScript (.js)': list(demo_dir.glob("**/*.js")),
        'JSON Data (.json)': list(demo_dir.glob("**/*.json")),
        'CSV Data (.csv)': list(demo_dir.glob("**/*.csv")),
        'CSS Styles (.css)': list(demo_dir.glob("**/*.css")),
    }

    logger.info(f"Processing files by type to demonstrate different strategies:")

    total_chunks = 0

    for file_type, files in file_types.items():
        if not files:
            continue

        logger.info(f"\n--- {file_type} ---")

        for file_path in files:
            logger.info(f"\n📄 Processing: {file_path.name}")

            try:
                result = orchestrator.chunk_file(file_path)

                if result and result.chunks:
                    chunk_count = len(result.chunks)
                    total_chunks += chunk_count

                    logger.info(f"   ✅ Strategy: {result.strategy_used}")
                    logger.info(f"   ✅ Chunks: {chunk_count}")
                    logger.info(f"   ✅ Processing time: {result.processing_time:.3f}s")

                    # Show first chunk content preview
                    if result.chunks:
                        first_chunk = result.chunks[0]
                        preview = first_chunk.content[:200].replace('\n', ' ').replace('\r', '')
                        logger.info(f"   📖 First chunk preview: {preview}...")
                        logger.info(f"   📏 First chunk length: {len(first_chunk.content)} characters")

                        # Show chunk metadata
                        if hasattr(first_chunk, 'metadata') and first_chunk.metadata:
                            logger.info(f"   🏷️  Metadata: {first_chunk.metadata}")
                else:
                    logger.info(f"   ❌ No chunks generated")

            except Exception as e:
                logger.error(f"   ❌ Error processing {file_path.name}: {e}")

    logger.info(f"\n📊 Total chunks generated: {total_chunks}")


def demonstrate_config_driven_processing(demo_dir: Path):
    """Demonstrate config-driven processing with valid strategy names."""

    logger.info("\n" + "="*60)
    logger.info("DEMO 2: CONFIG-DRIVEN PROCESSING WITH CUSTOM SETTINGS")
    logger.info("="*60)

    # Use the custom config file with valid strategy names
    config_path = demo_dir / "configs" / "directory_config.yaml"

    logger.info(f"Using configuration: {config_path}")

    try:
        # Create orchestrator with custom config
        orchestrator = ChunkerOrchestrator(config_path=config_path)

        # Process files with explicit strategy selection based on extension
        extensions_to_test = ['.txt', '.md', '.py', '.js', '.json', '.csv', '.css']

        for ext in extensions_to_test:
            files = list(demo_dir.rglob(f"*{ext}"))
            if files:
                logger.info(f"\n--- Processing {ext.upper()} files with config-driven strategies ---")

                for file_path in files:
                    logger.info(f"\n📄 File: {file_path.name}")

                    try:
                        result = orchestrator.chunk_file(file_path)

                        if result and result.chunks:
                            logger.info(f"   ✅ Strategy used: {result.strategy_used}")
                            logger.info(f"   ✅ Chunks generated: {len(result.chunks)}")
                            logger.info(f"   ⏱️  Processing time: {result.processing_time:.3f}s")

                            # Show detailed chunk information
                            for i, chunk in enumerate(result.chunks[:2]):  # Show first 2 chunks
                                chunk_preview = chunk.content[:150].replace('\n', ' ').replace('\r', '')
                                logger.info(f"   📖 Chunk {i+1}: {chunk_preview}...")
                                logger.info(f"   📏 Chunk {i+1} length: {len(chunk.content)} chars")

                                if hasattr(chunk, 'metadata') and chunk.metadata:
                                    logger.info(f"   🏷️  Chunk {i+1} metadata: {chunk.metadata}")

                            if len(result.chunks) > 2:
                                logger.info(f"   ... and {len(result.chunks) - 2} more chunks")
                        else:
                            logger.info(f"   ❌ No chunks generated")

                    except Exception as e:
                        logger.error(f"   ❌ Error processing {file_path.name}: {e}")

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")


def demonstrate_parallel_processing(demo_dir: Path):
    """Demonstrate parallel processing modes with performance comparison."""

    logger.info("\n" + "="*60)
    logger.info("DEMO 3: PARALLEL PROCESSING PERFORMANCE COMPARISON")
    logger.info("="*60)

    all_files = [f for f in demo_dir.rglob("*") if f.is_file() and not f.name.endswith('.yaml')]
    orchestrator = ChunkerOrchestrator()

    modes = ["sequential", "thread"]
    performance_results = {}

    for mode in modes:
        logger.info(f"\n🚀 Testing {mode.upper()} mode:")

        start_time = time.time()

        try:
            results = orchestrator.chunk_files_batch(
                file_paths=all_files,
                parallel_mode=mode,
                max_workers=2 if mode != "sequential" else None
            )

            end_time = time.time()
            processing_time = end_time - start_time

            successful = sum(1 for r in results if r and r.chunks)
            total_chunks = sum(len(r.chunks) for r in results if r and r.chunks)

            performance_results[mode] = {
                'time': processing_time,
                'successful': successful,
                'total_chunks': total_chunks,
                'files_per_sec': successful / processing_time if processing_time > 0 else 0
            }

            logger.info(f"   ⏱️  Processing time: {processing_time:.2f}s")
            logger.info(f"   ✅ Successful files: {successful}/{len(all_files)}")
            logger.info(f"   📊 Total chunks: {total_chunks}")
            logger.info(f"   🏃 Performance: {successful/processing_time:.1f} files/sec")

            # Show sample results with chunk previews
            logger.info(f"   📖 Sample results:")
            for i, result in enumerate(results[:3]):  # Show first 3 results
                if result and result.chunks:
                    file_path = all_files[i]
                    first_chunk_preview = result.chunks[0].content[:100].replace('\n', ' ')
                    logger.info(f"      {file_path.name}: {len(result.chunks)} chunks, strategy: {result.strategy_used}")
                    logger.info(f"         Preview: {first_chunk_preview}...")

        except Exception as e:
            logger.error(f"   ❌ Error in {mode} mode: {e}")

    # Performance comparison
    if len(performance_results) > 1:
        logger.info(f"\n📊 PERFORMANCE COMPARISON:")
        for mode, stats in performance_results.items():
            logger.info(f"   {mode.capitalize():>12}: {stats['files_per_sec']:>6.1f} files/sec, {stats['total_chunks']:>4} chunks")


def demonstrate_cli_examples(demo_dir: Path):
    """Demonstrate CLI command examples."""

    logger.info("\n" + "="*60)
    logger.info("DEMO 4: CLI COMMAND EXAMPLES")
    logger.info("="*60)

    logger.info("Here are the CLI commands you can use for directory processing:")
    logger.info("")

    # Show comprehensive examples
    examples = [
        ("Basic directory processing", f"python -m chunking_strategy process-directory {demo_dir}"),
        ("Process specific file types", f"python -m chunking_strategy process-directory {demo_dir} --extensions .txt,.py,.json"),
        ("With config file", f"python -m chunking_strategy process-directory {demo_dir} --config {demo_dir}/configs/directory_config.yaml"),
        ("With chunk preview", f"python -m chunking_strategy process-directory {demo_dir} --show-preview --max-preview-chunks 2"),
        ("Parallel processing", f"python -m chunking_strategy process-directory {demo_dir} --parallel-mode thread --workers 2"),
        ("Save output files", f"python -m chunking_strategy process-directory {demo_dir} --output-dir /tmp/chunks"),
        ("Non-recursive", f"python -m chunking_strategy process-directory {demo_dir} --no-recursive"),
        ("Enhanced batch processing", f"python -m chunking_strategy batch-directory {demo_dir} --pattern '*.py' --recursive"),
    ]

    for description, command in examples:
        logger.info(f"🔧 {description}:")
        logger.info(f"   {command}")
        logger.info("")


def main():
    """Main demo function - no user input required."""

    logger.info("🚀 COMPREHENSIVE DIRECTORY PROCESSING DEMO")
    logger.info("=" * 60)
    logger.info("This demo showcases:")
    logger.info("  ✅ Multiple chunking strategies for different file types")
    logger.info("  ✅ File path printing and detailed chunk previews")
    logger.info("  ✅ Config-driven processing with valid strategy names")
    logger.info("  ✅ Parallel processing performance comparison")
    logger.info("  ✅ CLI integration examples")
    logger.info("  ✅ No user interaction required - fully automated")

    # Create demo directory
    demo_dir = create_demo_directory()

    try:
        # Run all demonstrations
        demonstrate_basic_directory_processing(demo_dir)
        demonstrate_config_driven_processing(demo_dir)
        demonstrate_parallel_processing(demo_dir)
        demonstrate_cli_examples(demo_dir)

        logger.info("\n" + "="*60)
        logger.info("🎉 DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"📁 Demo directory location: {demo_dir}")
        logger.info("🔧 Try these commands:")
        logger.info(f"   python -m chunking_strategy process-directory {demo_dir} --show-preview")
        logger.info(f"   python -m chunking_strategy process-directory {demo_dir} --extensions .py,.js --parallel-mode thread")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Automatic cleanup - no user input required
        try:
            shutil.rmtree(demo_dir)
            logger.info("🗑️  Demo directory automatically cleaned up.")
        except Exception as e:
            logger.warning(f"⚠️  Could not clean up demo directory {demo_dir}: {e}")


if __name__ == "__main__":
    main()
