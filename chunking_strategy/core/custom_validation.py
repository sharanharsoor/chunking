"""
Custom Algorithm Validation Framework

This module provides comprehensive validation tools to ensure custom chunking
algorithms conform to interfaces, perform correctly, and integrate seamlessly
with the existing library infrastructure.

Key Features:
- Interface compliance validation
- Performance benchmarking and validation
- Quality assessment and scoring
- Integration testing with existing systems
- Security and safety checks
- Comprehensive reporting and recommendations

Example Usage:
    ```python
    from chunking_strategy.core.custom_validation import (
        CustomAlgorithmValidator,
        validate_custom_algorithm_file,
        run_comprehensive_validation
    )
    
    # Validate a custom algorithm file
    validator = CustomAlgorithmValidator()
    report = validator.validate_algorithm_file("my_custom_chunker.py")
    
    if report.is_valid:
        print("Algorithm is valid and ready for use!")
    else:
        print(f"Validation failed: {report.errors}")
        
    # Run comprehensive validation including performance tests
    comprehensive_report = run_comprehensive_validation(
        "my_custom_chunker.py", 
        include_performance=True,
        include_quality_tests=True
    )
    ```
"""

import inspect
import logging
import time
import statistics
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import importlib.util
import sys

from chunking_strategy.core.base import (
    BaseChunker,
    StreamableChunker,
    AdaptableChunker,
    HierarchicalChunker,
    Chunk,
    ChunkingResult,
    ModalityType
)
from chunking_strategy.core.registry import ChunkerMetadata
from chunking_strategy.core.custom_algorithm_loader import (
    CustomAlgorithmLoader,
    CustomAlgorithmInfo
)

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"   # Algorithm cannot be used
    ERROR = "error"         # Major issues that should be fixed
    WARNING = "warning"     # Minor issues or recommendations
    INFO = "info"           # Informational notes


class ValidationType(Enum):
    """Types of validation checks."""
    INTERFACE = "interface"           # Interface compliance
    FUNCTIONALITY = "functionality"   # Basic functionality
    PERFORMANCE = "performance"       # Performance characteristics
    QUALITY = "quality"              # Output quality
    SECURITY = "security"            # Security and safety
    INTEGRATION = "integration"      # Integration with framework
    METADATA = "metadata"            # Metadata completeness


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    type: ValidationType
    severity: ValidationSeverity
    message: str
    details: Optional[str] = None
    suggestion: Optional[str] = None
    code_location: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
            "code_location": self.code_location
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report for a custom algorithm."""
    algorithm_name: str
    source_file: Path
    is_valid: bool = True
    overall_score: float = 0.0
    
    # Issues by severity
    critical_issues: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    
    # Validation results by type
    interface_score: float = 0.0
    functionality_score: float = 0.0
    performance_score: float = 0.0
    quality_score: float = 0.0
    security_score: float = 0.0
    integration_score: float = 0.0
    metadata_score: float = 0.0
    
    # Performance metrics
    avg_processing_time: Optional[float] = None
    memory_usage: Optional[float] = None
    chunks_per_second: Optional[float] = None
    
    # Quality metrics
    avg_chunk_size: Optional[float] = None
    chunk_size_variance: Optional[float] = None
    boundary_quality: Optional[float] = None
    
    # Additional metadata
    validation_time: float = 0.0
    test_cases_run: int = 0
    test_cases_passed: int = 0
    
    def __post_init__(self):
        """Compute derived fields."""
        all_issues = self.critical_issues + self.errors + self.warnings + self.info
        
        # Determine if valid (no critical issues or errors)
        self.is_valid = len(self.critical_issues) == 0 and len(self.errors) == 0
        
        # Compute overall score
        scores = [
            self.interface_score,
            self.functionality_score,
            self.performance_score,
            self.quality_score,
            self.security_score,
            self.integration_score,
            self.metadata_score
        ]
        valid_scores = [s for s in scores if s > 0]
        self.overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue to the report."""
        if issue.severity == ValidationSeverity.CRITICAL:
            self.critical_issues.append(issue)
        elif issue.severity == ValidationSeverity.ERROR:
            self.errors.append(issue)
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
        else:
            self.info.append(issue)
            
    def get_all_issues(self) -> List[ValidationIssue]:
        """Get all issues sorted by severity."""
        return self.critical_issues + self.errors + self.warnings + self.info
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        return {
            "algorithm_name": self.algorithm_name,
            "is_valid": self.is_valid,
            "overall_score": self.overall_score,
            "issue_counts": {
                "critical": len(self.critical_issues),
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.info)
            },
            "scores": {
                "interface": self.interface_score,
                "functionality": self.functionality_score,
                "performance": self.performance_score,
                "quality": self.quality_score,
                "security": self.security_score,
                "integration": self.integration_score,
                "metadata": self.metadata_score
            },
            "test_results": {
                "total_tests": self.test_cases_run,
                "passed_tests": self.test_cases_passed,
                "success_rate": self.test_cases_passed / max(self.test_cases_run, 1)
            }
        }


class CustomAlgorithmValidator:
    """
    Comprehensive validator for custom chunking algorithms.
    
    Performs multiple types of validation to ensure custom algorithms
    are safe, functional, and well-integrated.
    """
    
    def __init__(
        self,
        strict_mode: bool = False,
        include_performance_tests: bool = True,
        include_quality_tests: bool = True,
        max_test_content_size: int = 10000
    ):
        """
        Initialize the validator.
        
        Args:
            strict_mode: Enable strict validation (fail on warnings)
            include_performance_tests: Run performance benchmarks
            include_quality_tests: Run quality assessment tests
            max_test_content_size: Maximum size of test content
        """
        self.strict_mode = strict_mode
        self.include_performance_tests = include_performance_tests
        self.include_quality_tests = include_quality_tests
        self.max_test_content_size = max_test_content_size
        
        # Test content for validation
        self.test_contents = self._generate_test_content()
        
    def validate_algorithm_file(
        self, 
        file_path: Union[str, Path]
    ) -> ValidationReport:
        """
        Validate a custom algorithm file comprehensively.
        
        Args:
            file_path: Path to the custom algorithm file
            
        Returns:
            ValidationReport with detailed results
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # Initialize report
        report = ValidationReport(
            algorithm_name="unknown",
            source_file=file_path
        )
        
        try:
            # Load the algorithm
            loader = CustomAlgorithmLoader(strict_validation=False)
            algo_info = loader.load_algorithm(file_path)
            
            if not algo_info:
                report.add_issue(ValidationIssue(
                    type=ValidationType.INTERFACE,
                    severity=ValidationSeverity.CRITICAL,
                    message="Failed to load custom algorithm",
                    suggestion="Check file syntax and ensure it contains a valid BaseChunker subclass"
                ))
                return report
                
            report.algorithm_name = algo_info.name
            
            # Run all validation checks
            self._validate_interface_compliance(algo_info, report)
            self._validate_functionality(algo_info, report)
            self._validate_metadata(algo_info, report)
            self._validate_security(algo_info, report)
            self._validate_integration(algo_info, report)
            
            if self.include_performance_tests:
                self._validate_performance(algo_info, report)
                
            if self.include_quality_tests:
                self._validate_quality(algo_info, report)
                
        except Exception as e:
            report.add_issue(ValidationIssue(
                type=ValidationType.INTERFACE,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation failed with exception: {str(e)}",
                details=traceback.format_exc(),
                suggestion="Check the algorithm implementation for errors"
            ))
            
        finally:
            report.validation_time = time.time() - start_time
            
        return report
        
    def _validate_interface_compliance(
        self, 
        algo_info: CustomAlgorithmInfo, 
        report: ValidationReport
    ) -> None:
        """Validate interface compliance."""
        score = 1.0
        chunker_class = algo_info.chunker_class
        
        # Check inheritance
        if not issubclass(chunker_class, BaseChunker):
            report.add_issue(ValidationIssue(
                type=ValidationType.INTERFACE,
                severity=ValidationSeverity.CRITICAL,
                message="Class does not inherit from BaseChunker",
                suggestion="Ensure your class inherits from BaseChunker"
            ))
            score = 0.0
            
        # Check required methods
        required_methods = ['chunk']
        for method_name in required_methods:
            if not hasattr(chunker_class, method_name):
                report.add_issue(ValidationIssue(
                    type=ValidationType.INTERFACE,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Missing required method: {method_name}",
                    suggestion=f"Implement the {method_name} method in your class"
                ))
                score = 0.0
            elif not callable(getattr(chunker_class, method_name)):
                report.add_issue(ValidationIssue(
                    type=ValidationType.INTERFACE,
                    severity=ValidationSeverity.ERROR,
                    message=f"Method {method_name} is not callable",
                    suggestion=f"Ensure {method_name} is defined as a method"
                ))
                score *= 0.8
                
        # Check method signatures
        if hasattr(chunker_class, 'chunk'):
            chunk_method = getattr(chunker_class, 'chunk')
            sig = inspect.signature(chunk_method)
            
            # Should accept content parameter
            if 'content' not in sig.parameters and len(sig.parameters) < 2:
                report.add_issue(ValidationIssue(
                    type=ValidationType.INTERFACE,
                    severity=ValidationSeverity.WARNING,
                    message="chunk method should accept 'content' parameter",
                    suggestion="Add 'content' parameter to chunk method signature"
                ))
                score *= 0.9
                
        # Check constructor
        try:
            sig = inspect.signature(chunker_class.__init__)
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_kwargs:
                report.add_issue(ValidationIssue(
                    type=ValidationType.INTERFACE,
                    severity=ValidationSeverity.WARNING,
                    message="Constructor should accept **kwargs for flexibility",
                    suggestion="Add **kwargs parameter to constructor"
                ))
                score *= 0.95
        except Exception:
            report.add_issue(ValidationIssue(
                type=ValidationType.INTERFACE,
                severity=ValidationSeverity.WARNING,
                message="Could not inspect constructor signature"
            ))
            
        # Check special interface implementations
        special_interfaces = [
            (StreamableChunker, "chunk_stream"),
            (AdaptableChunker, "adapt_parameters"),
            (HierarchicalChunker, "chunk_hierarchical")
        ]
        
        for interface_class, method_name in special_interfaces:
            if issubclass(chunker_class, interface_class):
                if not hasattr(chunker_class, method_name):
                    report.add_issue(ValidationIssue(
                        type=ValidationType.INTERFACE,
                        severity=ValidationSeverity.ERROR,
                        message=f"Class inherits from {interface_class.__name__} but missing {method_name} method",
                        suggestion=f"Implement {method_name} method or remove {interface_class.__name__} inheritance"
                    ))
                    score *= 0.7
                    
        report.interface_score = max(0.0, score)
        
    def _validate_functionality(
        self, 
        algo_info: CustomAlgorithmInfo, 
        report: ValidationReport
    ) -> None:
        """Validate basic functionality."""
        score = 1.0
        tests_run = 0
        tests_passed = 0
        
        try:
            # Try to instantiate the chunker
            chunker = algo_info.chunker_class()
            tests_run += 1
            tests_passed += 1
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                type=ValidationType.FUNCTIONALITY,
                severity=ValidationSeverity.CRITICAL,
                message=f"Cannot instantiate chunker: {str(e)}",
                details=traceback.format_exc(),
                suggestion="Check constructor implementation and dependencies"
            ))
            score = 0.0
            tests_run += 1
            report.functionality_score = score
            report.test_cases_run += tests_run
            report.test_cases_passed += tests_passed
            return
            
        # Test basic chunking functionality
        for test_name, test_content in self.test_contents.items():
            tests_run += 1
            
            try:
                result = chunker.chunk(test_content)
                
                # Validate result type
                if not isinstance(result, ChunkingResult):
                    report.add_issue(ValidationIssue(
                        type=ValidationType.FUNCTIONALITY,
                        severity=ValidationSeverity.ERROR,
                        message=f"chunk() method returned {type(result)} instead of ChunkingResult",
                        suggestion="Ensure chunk() method returns ChunkingResult object"
                    ))
                    score *= 0.8
                    continue
                    
                # Validate chunks
                if not hasattr(result, 'chunks') or not isinstance(result.chunks, list):
                    report.add_issue(ValidationIssue(
                        type=ValidationType.FUNCTIONALITY,
                        severity=ValidationSeverity.ERROR,
                        message="ChunkingResult.chunks is not a list",
                        suggestion="Ensure chunks attribute is a list of Chunk objects"
                    ))
                    score *= 0.8
                    continue
                    
                # Validate individual chunks
                for i, chunk in enumerate(result.chunks):
                    if not isinstance(chunk, Chunk):
                        report.add_issue(ValidationIssue(
                            type=ValidationType.FUNCTIONALITY,
                            severity=ValidationSeverity.ERROR,
                            message=f"Chunk {i} is not a Chunk object",
                            suggestion="Ensure all items in chunks list are Chunk objects"
                        ))
                        score *= 0.9
                        break
                        
                    if not chunk.content:
                        report.add_issue(ValidationIssue(
                            type=ValidationType.FUNCTIONALITY,
                            severity=ValidationSeverity.WARNING,
                            message=f"Chunk {i} has empty content",
                            suggestion="Consider filtering out empty chunks"
                        ))
                        score *= 0.95
                        
                tests_passed += 1
                
            except Exception as e:
                report.add_issue(ValidationIssue(
                    type=ValidationType.FUNCTIONALITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"chunk() method failed on {test_name} test: {str(e)}",
                    details=traceback.format_exc(),
                    suggestion="Debug the chunk() method implementation"
                ))
                score *= 0.8
                
        report.functionality_score = max(0.0, score)
        report.test_cases_run += tests_run
        report.test_cases_passed += tests_passed
        
    def _validate_performance(
        self, 
        algo_info: CustomAlgorithmInfo, 
        report: ValidationReport
    ) -> None:
        """Validate performance characteristics."""
        score = 1.0
        
        try:
            chunker = algo_info.chunker_class()
            
            # Performance test with different content sizes
            performance_results = []
            
            for size_name, test_content in self.test_contents.items():
                start_time = time.time()
                result = chunker.chunk(test_content)
                end_time = time.time()
                
                processing_time = end_time - start_time
                content_size = len(test_content)
                chunks_created = len(result.chunks) if result.chunks else 0
                
                performance_results.append({
                    "test_name": size_name,
                    "content_size": content_size,
                    "processing_time": processing_time,
                    "chunks_created": chunks_created,
                    "chars_per_second": content_size / processing_time if processing_time > 0 else 0
                })
                
            # Analyze performance results
            if performance_results:
                avg_time = statistics.mean([r["processing_time"] for r in performance_results])
                avg_throughput = statistics.mean([r["chars_per_second"] for r in performance_results])
                
                report.avg_processing_time = avg_time
                report.chunks_per_second = statistics.mean([
                    r["chunks_created"] / r["processing_time"] 
                    for r in performance_results if r["processing_time"] > 0
                ])
                
                # Performance scoring based on throughput
                if avg_throughput < 1000:  # Very slow
                    report.add_issue(ValidationIssue(
                        type=ValidationType.PERFORMANCE,
                        severity=ValidationSeverity.WARNING,
                        message=f"Low throughput: {avg_throughput:.0f} chars/sec",
                        suggestion="Consider optimizing the chunking algorithm for better performance"
                    ))
                    score *= 0.8
                elif avg_throughput < 100:  # Extremely slow
                    report.add_issue(ValidationIssue(
                        type=ValidationType.PERFORMANCE,
                        severity=ValidationSeverity.ERROR,
                        message=f"Very low throughput: {avg_throughput:.0f} chars/sec",
                        suggestion="Algorithm may be too slow for practical use"
                    ))
                    score *= 0.6
                    
        except Exception as e:
            report.add_issue(ValidationIssue(
                type=ValidationType.PERFORMANCE,
                severity=ValidationSeverity.WARNING,
                message=f"Performance testing failed: {str(e)}",
                suggestion="Check algorithm implementation for performance issues"
            ))
            score *= 0.7
            
        report.performance_score = max(0.0, score)
        
    def _validate_quality(
        self, 
        algo_info: CustomAlgorithmInfo, 
        report: ValidationReport
    ) -> None:
        """Validate output quality."""
        score = 1.0
        
        try:
            chunker = algo_info.chunker_class()
            
            quality_metrics = []
            
            for test_name, test_content in self.test_contents.items():
                result = chunker.chunk(test_content)
                
                if not result.chunks:
                    report.add_issue(ValidationIssue(
                        type=ValidationType.QUALITY,
                        severity=ValidationSeverity.WARNING,
                        message=f"No chunks produced for {test_name} test",
                        suggestion="Ensure algorithm produces chunks for all input types"
                    ))
                    score *= 0.9
                    continue
                    
                # Analyze chunk quality
                chunk_sizes = [len(chunk.content) for chunk in result.chunks]
                
                if chunk_sizes:
                    avg_size = statistics.mean(chunk_sizes)
                    size_variance = statistics.variance(chunk_sizes) if len(chunk_sizes) > 1 else 0
                    
                    quality_metrics.append({
                        "test_name": test_name,
                        "num_chunks": len(result.chunks),
                        "avg_chunk_size": avg_size,
                        "size_variance": size_variance,
                        "min_size": min(chunk_sizes),
                        "max_size": max(chunk_sizes)
                    })
                    
                    # Check for extremely uneven chunks
                    if len(chunk_sizes) > 1:
                        size_ratio = max(chunk_sizes) / min(chunk_sizes)
                        if size_ratio > 100:  # Very uneven
                            report.add_issue(ValidationIssue(
                                type=ValidationType.QUALITY,
                                severity=ValidationSeverity.WARNING,
                                message=f"Very uneven chunk sizes in {test_name}: {size_ratio:.1f}x difference",
                                suggestion="Consider improving chunk size consistency"
                            ))
                            score *= 0.95
                            
                    # Check for very small chunks
                    tiny_chunks = [s for s in chunk_sizes if s < 10]
                    if len(tiny_chunks) > len(chunk_sizes) * 0.2:  # More than 20% tiny chunks
                        report.add_issue(ValidationIssue(
                            type=ValidationType.QUALITY,
                            severity=ValidationSeverity.WARNING,
                            message=f"Many very small chunks in {test_name}: {len(tiny_chunks)} chunks < 10 chars",
                            suggestion="Consider merging very small chunks or adjusting algorithm parameters"
                        ))
                        score *= 0.9
                        
            # Compute overall quality metrics
            if quality_metrics:
                report.avg_chunk_size = statistics.mean([m["avg_chunk_size"] for m in quality_metrics])
                report.chunk_size_variance = statistics.mean([m["size_variance"] for m in quality_metrics])
                
        except Exception as e:
            report.add_issue(ValidationIssue(
                type=ValidationType.QUALITY,
                severity=ValidationSeverity.WARNING,
                message=f"Quality assessment failed: {str(e)}",
                suggestion="Check algorithm implementation"
            ))
            score *= 0.8
            
        report.quality_score = max(0.0, score)
        
    def _validate_security(
        self, 
        algo_info: CustomAlgorithmInfo, 
        report: ValidationReport
    ) -> None:
        """Validate security and safety aspects."""
        score = 1.0
        chunker_class = algo_info.chunker_class
        
        # Check for potentially dangerous operations
        source_code = inspect.getsource(chunker_class)
        
        dangerous_patterns = [
            ("exec", "Potentially dangerous exec() call"),
            ("eval", "Potentially dangerous eval() call"),
            ("__import__", "Dynamic import may be unsafe"),
            ("subprocess", "Subprocess execution may be unsafe"),
            ("os.system", "System command execution is unsafe"),
            ("open(", "File operations should be handled carefully")
        ]
        
        for pattern, warning in dangerous_patterns:
            if pattern in source_code:
                severity = ValidationSeverity.ERROR if pattern in ["exec", "eval", "os.system"] else ValidationSeverity.WARNING
                report.add_issue(ValidationIssue(
                    type=ValidationType.SECURITY,
                    severity=severity,
                    message=f"Potentially unsafe operation detected: {warning}",
                    suggestion="Review security implications of this operation"
                ))
                score *= 0.8 if severity == ValidationSeverity.WARNING else 0.6
                
        # Check for infinite loop potential
        if "while True" in source_code and "break" not in source_code:
            report.add_issue(ValidationIssue(
                type=ValidationType.SECURITY,
                severity=ValidationSeverity.WARNING,
                message="Potential infinite loop detected",
                suggestion="Ensure while loops have proper break conditions"
            ))
            score *= 0.9
            
        report.security_score = max(0.0, score)
        
    def _validate_integration(
        self, 
        algo_info: CustomAlgorithmInfo, 
        report: ValidationReport
    ) -> None:
        """Validate integration with framework."""
        score = 1.0
        
        # Check if algorithm is properly registered
        if not algo_info.is_registered:
            report.add_issue(ValidationIssue(
                type=ValidationType.INTEGRATION,
                severity=ValidationSeverity.WARNING,
                message="Algorithm is not registered with the framework",
                suggestion="Ensure the @register_chunker decorator is properly applied"
            ))
            score *= 0.8
            
        # Check metadata completeness
        if not algo_info.metadata:
            report.add_issue(ValidationIssue(
                type=ValidationType.INTEGRATION,
                severity=ValidationSeverity.WARNING,
                message="No metadata available for algorithm",
                suggestion="Add metadata using @register_chunker decorator"
            ))
            score *= 0.9
            
        report.integration_score = max(0.0, score)
        
    def _validate_metadata(
        self, 
        algo_info: CustomAlgorithmInfo, 
        report: ValidationReport
    ) -> None:
        """Validate metadata completeness and accuracy."""
        score = 1.0
        
        if not algo_info.metadata:
            report.add_issue(ValidationIssue(
                type=ValidationType.METADATA,
                severity=ValidationSeverity.WARNING,
                message="No metadata provided",
                suggestion="Add @register_chunker decorator with comprehensive metadata"
            ))
            score = 0.5
        else:
            metadata = algo_info.metadata
            
            # Check required fields
            if not metadata.description or len(metadata.description.strip()) < 10:
                report.add_issue(ValidationIssue(
                    type=ValidationType.METADATA,
                    severity=ValidationSeverity.WARNING,
                    message="Missing or insufficient description",
                    suggestion="Provide a detailed description of the algorithm"
                ))
                score *= 0.9
                
            if not metadata.use_cases:
                report.add_issue(ValidationIssue(
                    type=ValidationType.METADATA,
                    severity=ValidationSeverity.INFO,
                    message="No use cases specified",
                    suggestion="Specify use cases where this algorithm is most effective"
                ))
                score *= 0.95
                
            # Check quality score reasonableness
            if metadata.quality < 0.0 or metadata.quality > 1.0:
                report.add_issue(ValidationIssue(
                    type=ValidationType.METADATA,
                    severity=ValidationSeverity.WARNING,
                    message=f"Quality score {metadata.quality} is out of range [0.0, 1.0]",
                    suggestion="Set quality score between 0.0 and 1.0"
                ))
                score *= 0.9
                
        report.metadata_score = max(0.0, score)
        
    def _generate_test_content(self) -> Dict[str, str]:
        """Generate test content for validation."""
        return {
            "short": "This is a short test content.",
            "medium": "This is a medium-length test content. " * 20,
            "long": "This is a longer test content for performance testing. " * 100,
            "mixed": """This is mixed content.

It has multiple paragraphs with different structures.

- Some bullet points
- More bullet points

And some other text at the end.""",
            "edge_cases": "\n\n\n   \n\t\t\n   \nContent with weird whitespace.\n\n\n"
        }


def validate_custom_algorithm_file(
    file_path: Union[str, Path],
    **validator_kwargs
) -> ValidationReport:
    """
    Convenience function to validate a custom algorithm file.
    
    Args:
        file_path: Path to the custom algorithm file
        **validator_kwargs: Additional arguments for CustomAlgorithmValidator
        
    Returns:
        ValidationReport with results
    """
    validator = CustomAlgorithmValidator(**validator_kwargs)
    return validator.validate_algorithm_file(file_path)


def run_comprehensive_validation(
    file_path: Union[str, Path],
    include_performance: bool = True,
    include_quality_tests: bool = True,
    generate_report_file: bool = False
) -> ValidationReport:
    """
    Run comprehensive validation with all checks enabled.
    
    Args:
        file_path: Path to custom algorithm file
        include_performance: Include performance tests
        include_quality_tests: Include quality assessment
        generate_report_file: Generate a detailed report file
        
    Returns:
        ValidationReport with comprehensive results
    """
    validator = CustomAlgorithmValidator(
        include_performance_tests=include_performance,
        include_quality_tests=include_quality_tests
    )
    
    report = validator.validate_algorithm_file(file_path)
    
    if generate_report_file:
        report_path = Path(file_path).parent / f"{report.algorithm_name}_validation_report.json"
        
        import json
        with open(report_path, 'w') as f:
            json.dump({
                "summary": report.get_summary(),
                "issues": [issue.to_dict() for issue in report.get_all_issues()],
                "detailed_scores": {
                    "interface": report.interface_score,
                    "functionality": report.functionality_score, 
                    "performance": report.performance_score,
                    "quality": report.quality_score,
                    "security": report.security_score,
                    "integration": report.integration_score,
                    "metadata": report.metadata_score
                }
            }, f, indent=2)
            
        logger.info(f"Detailed validation report saved to: {report_path}")
        
    return report


def batch_validate_algorithms(
    directory_path: Union[str, Path],
    recursive: bool = True,
    **validator_kwargs
) -> Dict[str, ValidationReport]:
    """
    Validate all custom algorithms in a directory.
    
    Args:
        directory_path: Directory containing custom algorithm files
        recursive: Search subdirectories recursively
        **validator_kwargs: Additional validator arguments
        
    Returns:
        Dictionary mapping file names to validation reports
    """
    directory_path = Path(directory_path)
    
    # Find all Python files
    if recursive:
        python_files = list(directory_path.rglob("*.py"))
    else:
        python_files = list(directory_path.glob("*.py"))
        
    results = {}
    validator = CustomAlgorithmValidator(**validator_kwargs)
    
    for py_file in python_files:
        try:
            report = validator.validate_algorithm_file(py_file)
            results[str(py_file)] = report
        except Exception as e:
            logger.error(f"Failed to validate {py_file}: {e}")
            # Create minimal error report
            error_report = ValidationReport(
                algorithm_name=py_file.stem,
                source_file=py_file
            )
            error_report.add_issue(ValidationIssue(
                type=ValidationType.INTERFACE,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation failed with exception: {str(e)}"
            ))
            results[str(py_file)] = error_report
            
    return results


"""
Example Usage:

# Basic validation
from chunking_strategy.core.custom_validation import validate_custom_algorithm_file

report = validate_custom_algorithm_file("my_custom_chunker.py")

if report.is_valid:
    print(f"✅ Algorithm '{report.algorithm_name}' is valid!")
    print(f"Overall score: {report.overall_score:.2f}")
else:
    print(f"❌ Algorithm '{report.algorithm_name}' has issues:")
    for issue in report.get_all_issues():
        print(f"  {issue.severity.value.upper()}: {issue.message}")

# Comprehensive validation with report generation
comprehensive_report = run_comprehensive_validation(
    "my_custom_chunker.py",
    include_performance=True,
    include_quality_tests=True,
    generate_report_file=True
)

print(f"Comprehensive validation completed:")
print(f"Score breakdown:")
print(f"  Interface: {comprehensive_report.interface_score:.2f}")
print(f"  Functionality: {comprehensive_report.functionality_score:.2f}")
print(f"  Performance: {comprehensive_report.performance_score:.2f}")
print(f"  Quality: {comprehensive_report.quality_score:.2f}")

# Batch validation
from chunking_strategy.core.custom_validation import batch_validate_algorithms

results = batch_validate_algorithms("custom_algorithms/")
for file_path, report in results.items():
    status = "✅" if report.is_valid else "❌"
    print(f"{status} {file_path}: {report.overall_score:.2f}")
"""
