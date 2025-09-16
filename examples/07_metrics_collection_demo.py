#!/usr/bin/env python3
"""
Metrics Collection and Analysis Demo

This demo shows how to collect, analyze, and visualize performance metrics 
at different levels (file-level and chunk-level). Demonstrates:
- Performance metrics collection
- Quality assessment
- Comparative analysis across strategies
- File-level vs chunk-level metrics
- Visualization and reporting
- Metrics-driven optimization

Essential for optimizing chunking strategies and monitoring system performance.
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking_strategy.benchmarking import ChunkingBenchmark, PerformanceMetrics
from chunking_strategy.core.metrics import ChunkingQualityEvaluator
from chunking_strategy import create_chunker


def create_test_documents() -> Dict[str, Path]:
    """Create test documents with different characteristics."""
    print("📝 Creating test documents...")
    
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    documents = {}
    
    # Technical document
    technical_content = """
# Advanced Machine Learning Systems

## Neural Network Architectures
Deep neural networks consist of multiple layers of interconnected nodes. Each layer performs transformations on the input data, learning increasingly complex representations. Modern architectures like transformers use attention mechanisms to process sequences efficiently.

## Training Optimization
Gradient descent optimization algorithms update model parameters iteratively. Adam optimizer adapts learning rates for each parameter individually. Batch normalization stabilizes training by normalizing layer inputs.

## Regularization Techniques
Dropout randomly sets neuron outputs to zero during training to prevent overfitting. L1 and L2 regularization add penalty terms to the loss function. Early stopping prevents overfitting by monitoring validation performance.

## Performance Evaluation
Cross-validation provides robust performance estimates by training on multiple data splits. Precision and recall measure classification performance for imbalanced datasets. ROC curves visualize the trade-off between sensitivity and specificity.
"""
    
    # Narrative document
    narrative_content = """
The old lighthouse stood sentinel on the rocky cliff, its beacon cutting through the thick fog that rolled in from the Atlantic. Sarah had grown up seeing its light every night from her bedroom window, a constant companion through childhood fears and teenage anxieties.

Tonight was different. Tonight, the light flickered erratically, casting strange shadows across the churning waters below. She grabbed her jacket and headed out into the storm, driven by an inexplicable need to understand what was wrong.

The path to the lighthouse was treacherous in good weather, but with the wind howling and rain lashing against her face, each step became an act of determination. Lightning illuminated the rocky coastline in stark relief, revealing the true danger of her journey.

As she reached the lighthouse door, Sarah noticed it stood ajar, creaking ominously in the wind. The light above continued its erratic dance, casting her shadow long and dark across the threshold.
"""
    
    # Mixed content document
    mixed_content = """
# Research Report: Climate Change Impacts

## Executive Summary
This report analyzes the latest climate data and projects future impacts on global ecosystems. Temperature records show an accelerating warming trend over the past decade.

Data shows:
- Average global temperature increase: 1.2°C since 1880
- Sea level rise: 8-9 inches over the past century  
- Arctic ice extent: declining 13% per decade

## Methodology
We analyzed temperature records from 15,000 weather stations worldwide. Satellite data provided additional coverage for remote regions. Statistical models projected future trends based on current emission scenarios.

The research team collected data using standardized protocols. Quality control procedures ensured data accuracy and consistency across all measurement sites.

## Results and Discussion
The findings reveal unprecedented rates of change in Earth's climate system. Polar regions show the most dramatic warming, with temperature increases of 2-3°C in some areas.

"The data speaks for itself," said Dr. Elena Rodriguez, lead climatologist. "We're seeing changes that would normally take centuries happening in just decades."

## Implications
These changes will have profound impacts on agriculture, water resources, and human settlements. Coastal communities face particular risks from rising sea levels and increased storm intensity.

Adaptation strategies must be implemented immediately to minimize negative impacts on vulnerable populations.
"""
    
    # Save documents
    docs = {
        "technical": (technical_content, "technical_doc.md"),
        "narrative": (narrative_content, "narrative_doc.txt"),  
        "mixed": (mixed_content, "mixed_content_doc.md")
    }
    
    for doc_type, (content, filename) in docs.items():
        file_path = test_data_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        documents[doc_type] = file_path
        print(f"   ✅ Created {filename} ({len(content)} chars)")
    
    return documents


def collect_performance_metrics():
    """Demonstrate comprehensive performance metrics collection."""
    print("\n📊 PERFORMANCE METRICS COLLECTION")
    print("=" * 50)
    
    documents = create_test_documents()
    benchmark = ChunkingBenchmark()
    
    # Test different strategies
    strategies = [
        "sentence_based",
        "paragraph_based", 
        "fixed_size",
        "semantic"
    ]
    
    all_results = {}
    
    for doc_type, doc_path in documents.items():
        print(f"\n📄 Testing document: {doc_type}")
        doc_results = {}
        
        for strategy in strategies:
            print(f"   🔧 Strategy: {strategy}...")
            
            try:
                # Collect performance metrics
                metrics = benchmark.benchmark_strategy(strategy, doc_path)
                doc_results[strategy] = metrics
                
                # Display key metrics
                print(f"      ⏱️  Time: {metrics.processing_time:.3f}s")
                print(f"      💾 Memory: {metrics.memory_usage_mb:.1f}MB")
                print(f"      📊 Chunks: {metrics.chunks_generated}")
                print(f"      📏 Avg size: {metrics.avg_chunk_size:.1f} chars")
                print(f"      🚀 Throughput: {metrics.throughput_mb_per_sec:.2f} MB/s")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)[:50]}...")
                
        all_results[doc_type] = doc_results
    
    return all_results


def analyze_chunk_level_metrics():
    """Demonstrate chunk-level quality metrics."""
    print("\n🔍 CHUNK-LEVEL QUALITY ANALYSIS")
    print("=" * 50)
    
    documents = create_test_documents()
    evaluator = ChunkingQualityEvaluator()
    
    # Test with semantic chunking for detailed analysis
    for doc_type, doc_path in documents.items():
        print(f"\n📄 Analyzing {doc_type} document:")
        
        try:
            # Read document content
            with open(doc_path, 'r') as f:
                content = f.read()
            
            # Create chunker and process
            chunker = create_chunker(
                name="semantic",
                similarity_threshold=0.7,
                min_chunk_sentences=2,
                max_chunk_sentences=5,
                semantic_model="tfidf"
            )
            
            if not chunker:
                print("   ❌ Failed to create semantic chunker")
                continue
                
            result = chunker.chunk(content)
            chunks = result.chunks
            
            # Analyze chunk characteristics
            chunk_sizes = [len(chunk.content) for chunk in chunks]
            chunk_words = [len(chunk.content.split()) for chunk in chunks]
            
            print(f"   📊 Generated {len(chunks)} chunks")
            print(f"   📏 Size range: {min(chunk_sizes)}-{max(chunk_sizes)} chars")
            print(f"   📝 Word range: {min(chunk_words)}-{max(chunk_words)} words")
            print(f"   🎯 Average size: {sum(chunk_sizes)/len(chunk_sizes):.1f} chars")
            print(f"   📈 Size std dev: {(sum((s - sum(chunk_sizes)/len(chunk_sizes))**2 for s in chunk_sizes) / len(chunk_sizes))**0.5:.1f}")
            
            # Show chunk previews
            print(f"   📋 Sample chunks:")
            for i, chunk in enumerate(chunks[:3]):
                preview = chunk.content.strip()[:80] + "..." if len(chunk.content) > 80 else chunk.content.strip()
                print(f"      {i+1}. {preview}")
                
        except Exception as e:
            print(f"   ❌ Analysis failed: {str(e)[:60]}...")


def create_comparative_analysis(all_results: Dict[str, Dict[str, PerformanceMetrics]]):
    """Create comprehensive comparative analysis."""
    print("\n📈 COMPARATIVE ANALYSIS")
    print("=" * 50)
    
    if not all_results:
        print("❌ No results to analyze")
        return
    
    # Performance summary across all documents
    print("\n1️⃣ Overall Performance Summary:")
    
    strategy_totals = {}
    for doc_type, doc_results in all_results.items():
        for strategy, metrics in doc_results.items():
            if strategy not in strategy_totals:
                strategy_totals[strategy] = {
                    'total_time': 0,
                    'total_memory': 0,
                    'total_chunks': 0,
                    'total_throughput': 0,
                    'count': 0
                }
            
            strategy_totals[strategy]['total_time'] += metrics.processing_time
            strategy_totals[strategy]['total_memory'] += metrics.memory_usage_mb
            strategy_totals[strategy]['total_chunks'] += metrics.chunks_generated
            strategy_totals[strategy]['total_throughput'] += metrics.throughput_mb_per_sec
            strategy_totals[strategy]['count'] += 1
    
    # Calculate averages and display
    for strategy, totals in strategy_totals.items():
        count = totals['count']
        if count > 0:
            avg_time = totals['total_time'] / count
            avg_memory = totals['total_memory'] / count
            avg_chunks = totals['total_chunks'] / count
            avg_throughput = totals['total_throughput'] / count
            
            print(f"\n   🔧 {strategy}:")
            print(f"      ⏱️  Avg time: {avg_time:.3f}s")
            print(f"      💾 Avg memory: {avg_memory:.1f}MB") 
            print(f"      📊 Avg chunks: {avg_chunks:.1f}")
            print(f"      🚀 Avg throughput: {avg_throughput:.2f} MB/s")
    
    # Best performers
    print("\n2️⃣ Best Performers:")
    
    if strategy_totals:
        # Fastest strategy
        fastest = min(strategy_totals.keys(), 
                     key=lambda s: strategy_totals[s]['total_time'] / strategy_totals[s]['count'])
        print(f"   ⚡ Fastest: {fastest}")
        
        # Most memory efficient
        most_efficient = min(strategy_totals.keys(),
                           key=lambda s: strategy_totals[s]['total_memory'] / strategy_totals[s]['count'])
        print(f"   💾 Most memory efficient: {most_efficient}")
        
        # Highest throughput
        highest_throughput = max(strategy_totals.keys(),
                               key=lambda s: strategy_totals[s]['total_throughput'] / strategy_totals[s]['count'])
        print(f"   🚀 Highest throughput: {highest_throughput}")
    
    # Document-specific analysis
    print("\n3️⃣ Document-Specific Performance:")
    
    for doc_type, doc_results in all_results.items():
        print(f"\n   📄 {doc_type.upper()} document:")
        
        if doc_results:
            # Find best strategy for this document type
            best_speed = min(doc_results.keys(), key=lambda s: doc_results[s].processing_time)
            best_memory = min(doc_results.keys(), key=lambda s: doc_results[s].memory_usage_mb)
            
            print(f"      ⚡ Fastest: {best_speed} ({doc_results[best_speed].processing_time:.3f}s)")
            print(f"      💾 Most efficient: {best_memory} ({doc_results[best_memory].memory_usage_mb:.1f}MB)")


def demonstrate_metrics_visualization():
    """Simulate metrics visualization (text-based)."""
    print("\n📊 METRICS VISUALIZATION")
    print("=" * 50)
    
    # Simulate some performance data
    strategies = ["sentence_based", "paragraph_based", "fixed_size", "semantic"]
    metrics_data = {
        "processing_time": [0.125, 0.089, 0.045, 0.234],
        "memory_usage": [12.3, 8.7, 6.2, 18.9],
        "throughput": [2.45, 3.12, 4.67, 1.89],
        "chunk_count": [45, 23, 78, 34]
    }
    
    print("\n📈 Performance Comparison (Text Visualization):")
    
    # Processing Time Chart
    print("\n⏱️  Processing Time (seconds):")
    max_time = max(metrics_data["processing_time"])
    for i, strategy in enumerate(strategies):
        time_val = metrics_data["processing_time"][i]
        bar_length = int((time_val / max_time) * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {strategy:15} |{bar}| {time_val:.3f}s")
    
    # Memory Usage Chart  
    print("\n💾 Memory Usage (MB):")
    max_memory = max(metrics_data["memory_usage"])
    for i, strategy in enumerate(strategies):
        memory_val = metrics_data["memory_usage"][i]
        bar_length = int((memory_val / max_memory) * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {strategy:15} |{bar}| {memory_val:.1f}MB")
    
    # Throughput Chart
    print("\n🚀 Throughput (MB/s):")
    max_throughput = max(metrics_data["throughput"])
    for i, strategy in enumerate(strategies):
        throughput_val = metrics_data["throughput"][i]
        bar_length = int((throughput_val / max_throughput) * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {strategy:15} |{bar}| {throughput_val:.2f} MB/s")


def save_metrics_report(all_results: Dict[str, Dict[str, PerformanceMetrics]]):
    """Save comprehensive metrics report."""
    print("\n💾 SAVING METRICS REPORT")
    print("=" * 50)
    
    if not all_results:
        print("❌ No results to save")
        return
    
    try:
        # Convert results to serializable format
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_documents": len(all_results),
            "strategies_tested": list(next(iter(all_results.values())).keys()) if all_results else [],
            "detailed_results": {}
        }
        
        for doc_type, doc_results in all_results.items():
            report_data["detailed_results"][doc_type] = {}
            for strategy, metrics in doc_results.items():
                # Convert PerformanceMetrics to dict
                report_data["detailed_results"][doc_type][strategy] = asdict(metrics)
        
        # Save to file
        reports_dir = Path("benchmarks/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = reports_dir / f"metrics_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"   ✅ Report saved: {report_file}")
        print(f"   📊 Contains data for {len(all_results)} documents")
        print(f"   🔧 Tested {len(report_data['strategies_tested'])} strategies")
        
        # Save summary CSV for spreadsheet analysis
        csv_file = reports_dir / f"metrics_summary_{int(time.time())}.csv"
        with open(csv_file, 'w') as f:
            f.write("Document,Strategy,ProcessingTime,MemoryMB,ChunkCount,ThroughputMBps\n")
            
            for doc_type, doc_results in all_results.items():
                for strategy, metrics in doc_results.items():
                    f.write(f"{doc_type},{strategy},{metrics.processing_time:.3f},"
                           f"{metrics.memory_usage_mb:.1f},{metrics.chunks_generated},"
                           f"{metrics.throughput_mb_per_sec:.2f}\n")
        
        print(f"   📈 CSV summary: {csv_file}")
        
    except Exception as e:
        print(f"   ❌ Failed to save report: {e}")


def demonstrate_optimization_guidance():
    """Provide optimization guidance based on metrics."""
    print("\n💡 OPTIMIZATION GUIDANCE")
    print("=" * 50)
    
    print("\n🎯 Strategy Selection Guidelines:")
    print("   ⚡ For speed-critical applications:")
    print("      • Use fixed_size chunking (fastest)")
    print("      • Avoid semantic chunking for large volumes")
    print("      • Consider paragraph_based for balanced performance")
    
    print("\n   💾 For memory-constrained environments:")
    print("      • Use streaming processing")
    print("      • Prefer paragraph_based over sentence_based")
    print("      • Set smaller chunk sizes")
    
    print("\n   🎯 For quality-focused applications:")
    print("      • Use semantic chunking despite performance cost")
    print("      • Enable overlap for better context preservation")
    print("      • Monitor chunk size consistency")
    
    print("\n📊 Monitoring Recommendations:")
    print("   • Track throughput trends over time")
    print("   • Monitor memory usage patterns")
    print("   • Measure chunk quality metrics")
    print("   • Compare strategy performance on your specific data")
    
    print("\n🔧 Tuning Parameters:")
    print("   • Chunk size: Balance between granularity and context")
    print("   • Overlap: Trade-off between redundancy and continuity")
    print("   • Similarity thresholds: Adjust for content coherence")


def main():
    """Run the complete metrics collection demo."""
    print("📊 METRICS COLLECTION AND ANALYSIS DEMO")
    print("=" * 60)
    print("This demo shows comprehensive metrics collection and analysis capabilities.\n")
    
    try:
        # Demo 1: Performance metrics collection
        all_results = collect_performance_metrics()
        
        # Demo 2: Chunk-level analysis
        analyze_chunk_level_metrics()
        
        # Demo 3: Comparative analysis
        create_comparative_analysis(all_results)
        
        # Demo 4: Visualization
        demonstrate_metrics_visualization()
        
        # Demo 5: Save detailed report
        save_metrics_report(all_results)
        
        # Demo 6: Optimization guidance
        demonstrate_optimization_guidance()
        
        print("\n" + "=" * 60)
        print("🎉 METRICS COLLECTION DEMO COMPLETE!")
        print("=" * 60)
        print("\n📊 Capabilities Demonstrated:")
        print("   • 📈 Performance metrics collection")
        print("   • 🔍 Chunk-level quality analysis")
        print("   • 📊 Comparative strategy analysis")
        print("   • 📋 Comprehensive reporting")
        print("   • 💡 Optimization guidance")
        
        print("\n🔧 Applications:")
        print("   • Strategy selection and tuning")
        print("   • Performance monitoring and alerting")
        print("   • Quality assurance and validation")
        print("   • Capacity planning and scaling")
        print("   • A/B testing of different approaches")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
