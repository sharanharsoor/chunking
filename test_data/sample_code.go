package main

import (
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
)

// Constants
const (
	MaxRetries     = 3
	DefaultTimeout = 30 * time.Second
)

// ProcessingResult represents the result of a processing operation
type ProcessingResult struct {
	Original  string    `json:"original"`
	Processed string    `json:"processed"`
	Length    int       `json:"length"`
	Success   bool      `json:"success"`
	Timestamp time.Time `json:"timestamp"`
}

// DataProcessor interface defines the contract for data processors
type DataProcessor interface {
	Process(data string) ProcessingResult
	ProcessBatch(data []string) []ProcessingResult
	GetStatistics() ProcessorStats
}

// ProcessorStats holds statistics about processing operations
type ProcessorStats struct {
	Name           string `json:"name"`
	ProcessedCount int    `json:"processed_count"`
	ErrorCount     int    `json:"error_count"`
}

// DefaultProcessor implements the DataProcessor interface
type DefaultProcessor struct {
	name           string
	processedCount int
	errorCount     int
	uppercase      bool
	mu             sync.RWMutex
}

// NewDefaultProcessor creates a new DefaultProcessor instance
func NewDefaultProcessor(name string, uppercase bool) *DefaultProcessor {
	return &DefaultProcessor{
		name:      name,
		uppercase: uppercase,
	}
}

// Process processes a single data item
func (p *DefaultProcessor) Process(data string) ProcessingResult {
	p.mu.Lock()
	defer p.mu.Unlock()

	if data == "" {
		p.errorCount++
		return ProcessingResult{
			Original:  data,
			Processed: "",
			Length:    0,
			Success:   false,
			Timestamp: time.Now(),
		}
	}

	// Process the data
	processed := strings.TrimSpace(data)
	if p.uppercase {
		processed = strings.ToUpper(processed)
	}

	p.processedCount++
	return ProcessingResult{
		Original:  data,
		Processed: processed,
		Length:    len(processed),
		Success:   true,
		Timestamp: time.Now(),
	}
}

// ProcessBatch processes multiple data items
func (p *DefaultProcessor) ProcessBatch(data []string) []ProcessingResult {
	results := make([]ProcessingResult, 0, len(data))
	
	for _, item := range data {
		result := p.Process(item)
		results = append(results, result)
	}
	
	return results
}

// GetStatistics returns processor statistics
func (p *DefaultProcessor) GetStatistics() ProcessorStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	return ProcessorStats{
		Name:           p.name,
		ProcessedCount: p.processedCount,
		ErrorCount:     p.errorCount,
	}
}

// MetricsCalculator provides utility functions for calculating metrics
type MetricsCalculator struct{}

// CalculateMetrics calculates metrics from processing results
func (mc *MetricsCalculator) CalculateMetrics(results []ProcessingResult) map[string]interface{} {
	if len(results) == 0 {
		return map[string]interface{}{
			"count":        0,
			"avg_length":   0.0,
			"total_length": 0,
			"success_rate": 0.0,
		}
	}

	totalLength := 0
	successCount := 0

	for _, result := range results {
		if result.Success {
			totalLength += result.Length
			successCount++
		}
	}

	avgLength := 0.0
	if successCount > 0 {
		avgLength = float64(totalLength) / float64(successCount)
	}

	successRate := float64(successCount) / float64(len(results)) * 100

	return map[string]interface{}{
		"count":        len(results),
		"avg_length":   avgLength,
		"total_length": totalLength,
		"success_count": successCount,
		"success_rate": successRate,
	}
}

// ProcessingService provides high-level processing functionality
type ProcessingService struct {
	processor DataProcessor
	calculator *MetricsCalculator
}

// NewProcessingService creates a new ProcessingService
func NewProcessingService(processor DataProcessor) *ProcessingService {
	return &ProcessingService{
		processor:  processor,
		calculator: &MetricsCalculator{},
	}
}

// ProcessAndAnalyze processes data and returns analysis
func (s *ProcessingService) ProcessAndAnalyze(data []string) ([]ProcessingResult, map[string]interface{}) {
	results := s.processor.ProcessBatch(data)
	metrics := s.calculator.CalculateMetrics(results)
	return results, metrics
}

// Helper function for logging
func logError(message string, err error) {
	if err != nil {
		log.Printf("ERROR: %s: %v", message, err)
	} else {
		log.Printf("ERROR: %s", message)
	}
}

// Helper function for validation
func validateData(data []string) []string {
	valid := make([]string, 0, len(data))
	for _, item := range data {
		if strings.TrimSpace(item) != "" {
			valid = append(valid, item)
		}
	}
	return valid
}

func main() {
	// Create processor
	processor := NewDefaultProcessor("go_test_processor", true)
	service := NewProcessingService(processor)

	// Sample data
	sampleData := []string{
		"hello world",
		"go programming",
		"concurrent processing",
		"  whitespace test  ",
		"", // This will be filtered out
		"goroutines and channels",
	}

	// Validate data
	validData := validateData(sampleData)

	// Process and analyze
	results, metrics := service.ProcessAndAnalyze(validData)

	// Output results
	fmt.Println("Processing Results:")
	fmt.Printf("Processed %v items\n", metrics["count"])
	fmt.Printf("Successful: %v\n", metrics["success_count"])
	fmt.Printf("Success rate: %.2f%%\n", metrics["success_rate"])
	fmt.Printf("Average length: %.2f\n", metrics["avg_length"])
	fmt.Printf("Total length: %v\n", metrics["total_length"])

	// Print processor statistics
	stats := processor.GetStatistics()
	fmt.Printf("\nProcessor Statistics:\n")
	fmt.Printf("Name: %s\n", stats.Name)
	fmt.Printf("Total processed: %d\n", stats.ProcessedCount)
	fmt.Printf("Errors: %d\n", stats.ErrorCount)

	// Print sample results
	fmt.Printf("\nSample Results:\n")
	for i, result := range results {
		if i >= 3 { // Show only first 3
			break
		}
		fmt.Printf("  %s -> %s (success: %v)\n", 
			result.Original, result.Processed, result.Success)
	}
}
