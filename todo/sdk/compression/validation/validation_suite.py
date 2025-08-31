"""
CIP-SC Validation Framework

Comprehensive testing and validation suite for semantic compression implementations.
Validates compression ratios, reconstruction fidelity, and protocol compliance.
"""

import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Test Data Generators
# ============================================================================

class TestDataGenerator:
    """Generate various types of test data for compression validation."""
    
    @staticmethod
    def generate_geometric_image(size: int = 128, complexity: str = 'simple') -> bytes:
        """Generate geometric test images of varying complexity."""
        img = np.zeros((size, size))
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        if complexity == 'simple':
            # Basic shapes - should achieve 1000:1+ compression
            img[(X**2 + Y**2) < 0.3**2] = 1.0  # Circle
            img[(np.abs(X) < 0.2) & (np.abs(Y) < 0.2)] = 0.5  # Square
            img[(Y > -0.5) & (Y < X) & (Y < -X)] = 0.8  # Triangle
            
        elif complexity == 'medium':
            # Multiple shapes with variations
            # Circle variations
            for i, r in enumerate([0.15, 0.25, 0.35]):
                offset_x = 0.4 * np.cos(i * 2 * np.pi / 3)
                offset_y = 0.4 * np.sin(i * 2 * np.pi / 3)
                mask = ((X - offset_x)**2 + (Y - offset_y)**2) < r**2
                img[mask] = 0.3 + i * 0.2
            
            # Rectangles
            for i in range(3):
                angle = i * np.pi / 4
                x_rot = X * np.cos(angle) - Y * np.sin(angle)
                y_rot = X * np.sin(angle) + Y * np.cos(angle)
                mask = (np.abs(x_rot) < 0.1) & (np.abs(y_rot) < 0.3)
                img[mask] = 0.6 + i * 0.1
                
        elif complexity == 'complex':
            # Complex geometric patterns
            # Concentric circles
            for i, r in enumerate(np.linspace(0.1, 0.8, 8)):
                mask = ((X**2 + Y**2) < r**2) & ((X**2 + Y**2) >= (r-0.05)**2)
                img[mask] = 0.2 + (i % 4) * 0.2
            
            # Grid pattern
            grid_mask = ((X % 0.2 < 0.05) | (Y % 0.2 < 0.05))
            img[grid_mask] = np.minimum(img[grid_mask] + 0.3, 1.0)
            
        return img.astype(np.float64).tobytes()
    
    @staticmethod
    def generate_structured_data(data_type: str, size: int = 1000) -> bytes:
        """Generate structured data for testing different compression scenarios."""
        if data_type == 'repetitive':
            # Highly repetitive data - should compress very well
            pattern = np.array([1, 2, 3, 4, 5], dtype=np.float64)
            data = np.tile(pattern, size // len(pattern))
            
        elif data_type == 'random':
            # Random data - should compress poorly
            data = np.random.random(size).astype(np.float64)
            
        elif data_type == 'fractal':
            # Fractal-like data - should have intermediate compression
            data = np.zeros(size, dtype=np.float64)
            for i in range(size):
                data[i] = np.sin(i * 0.1) + 0.5 * np.sin(i * 0.3) + 0.25 * np.sin(i * 0.7)
                
        return data.tobytes()
    
    @staticmethod
    def generate_synthetic_cad(complexity: int = 5) -> bytes:
        """Generate synthetic CAD-like data."""
        size = 256
        img = np.zeros((size, size))
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        
        # Generate mechanical-like shapes
        for i in range(complexity):
            # Random circle
            cx = np.random.uniform(-1, 1)
            cy = np.random.uniform(-1, 1)
            r = np.random.uniform(0.1, 0.5)
            intensity = np.random.uniform(0.3, 1.0)
            
            mask = ((X - cx)**2 + (Y - cy)**2) < r**2
            img[mask] = intensity
            
            # Random rectangle
            x0 = np.random.uniform(-1.5, 1.5)
            y0 = np.random.uniform(-1.5, 1.5)
            w = np.random.uniform(0.1, 0.4)
            h = np.random.uniform(0.1, 0.4)
            intensity = np.random.uniform(0.3, 1.0)
            
            mask = ((X > x0) & (X < x0 + w) & (Y > y0) & (Y < y0 + h))
            img[mask] = intensity
        
        return img.astype(np.float64).tobytes()

# ============================================================================
# Validation Metrics
# ============================================================================

@dataclass
class CompressionMetrics:
    """Metrics for evaluating compression performance."""
    compression_ratio: float
    compression_time: float
    decompression_time: float
    reconstruction_error: float
    memory_usage: float
    symbols_count: int
    
    def get_efficiency_score(self) -> float:
        """Calculate overall efficiency score (0-1)."""
        # Normalize factors
        ratio_score = min(self.compression_ratio / 1000.0, 1.0)  # Target 1000:1
        speed_score = max(0, 1.0 - self.compression_time / 10.0)  # Target <10s
        fidelity_score = max(0, 1.0 - self.reconstruction_error * 1000)  # Target <0.001
        
        return (ratio_score + speed_score + fidelity_score) / 3.0

@dataclass
class ValidationResult:
    """Results from validation testing."""
    test_name: str
    data_type: str
    data_size: int
    success: bool
    metrics: Optional[CompressionMetrics]
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.metrics:
            result['metrics'] = asdict(self.metrics)
        return result

class ValidationSuite:
    """Comprehensive validation suite for CIP-SC implementations."""
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[ValidationResult] = []
    
    def run_geometric_validation(self, compressor, sizes: List[int] = None, 
                                complexities: List[str] = None) -> List[ValidationResult]:
        """Run validation tests on geometric data."""
        if sizes is None:
            sizes = [64, 128, 256]
        if complexities is None:
            complexities = ['simple', 'medium', 'complex']
        
        results = []
        
        for size in sizes:
            for complexity in complexities:
                test_name = f"geometric_{complexity}_{size}x{size}"
                print(f"🧪 Running test: {test_name}")
                
                try:
                    # Generate test data
                    test_data = TestDataGenerator.generate_geometric_image(size, complexity)
                    
                    # Measure compression
                    start_time = time.time()
                    compression_result = compressor.compress(test_data)
                    compression_time = time.time() - start_time
                    
                    if not compression_result.success:
                        results.append(ValidationResult(
                            test_name=test_name,
                            data_type=f"geometric_{complexity}",
                            data_size=len(test_data),
                            success=False,
                            metrics=None,
                            error_message=compression_result.error_message
                        ))
                        continue
                    
                    # Measure decompression
                    start_time = time.time()
                    decompression_result = compressor.decompress(compression_result.compressed_data)
                    decompression_time = time.time() - start_time
                    
                    if not decompression_result.success:
                        results.append(ValidationResult(
                            test_name=test_name,
                            data_type=f"geometric_{complexity}",
                            data_size=len(test_data),
                            success=False,
                            metrics=None,
                            error_message=decompression_result.error_message
                        ))
                        continue
                    
                    # Calculate metrics
                    metrics = CompressionMetrics(
                        compression_ratio=compression_result.compression_ratio,
                        compression_time=compression_time,
                        decompression_time=decompression_time,
                        reconstruction_error=decompression_result.reconstruction_error,
                        memory_usage=0.0,  # Would measure in practice
                        symbols_count=len(compression_result.compressed_data.payload.symbols)
                    )
                    
                    results.append(ValidationResult(
                        test_name=test_name,
                        data_type=f"geometric_{complexity}",
                        data_size=len(test_data),
                        success=True,
                        metrics=metrics
                    ))
                    
                    print(f"   ✅ Ratio: {metrics.compression_ratio:.1f}:1, "
                          f"Error: {metrics.reconstruction_error:.8f}, "
                          f"Symbols: {metrics.symbols_count}")
                
                except Exception as e:
                    results.append(ValidationResult(
                        test_name=test_name,
                        data_type=f"geometric_{complexity}",
                        data_size=len(test_data) if 'test_data' in locals() else 0,
                        success=False,
                        metrics=None,
                        error_message=str(e)
                    ))
                    print(f"   ❌ Failed: {e}")
        
        self.results.extend(results)
        return results
    
    def run_scalability_test(self, compressor, max_size: int = 512) -> List[ValidationResult]:
        """Test compression scalability with increasing data sizes."""
        sizes = [32, 64, 128, 256, 512]
        sizes = [s for s in sizes if s <= max_size]
        
        results = []
        
        for size in sizes:
            test_name = f"scalability_{size}x{size}"
            print(f"📈 Running scalability test: {test_name}")
            
            try:
                test_data = TestDataGenerator.generate_geometric_image(size, 'simple')
                
                start_time = time.time()
                compression_result = compressor.compress(test_data)
                compression_time = time.time() - start_time
                
                if compression_result.success:
                    start_time = time.time()
                    decompression_result = compressor.decompress(compression_result.compressed_data)
                    decompression_time = time.time() - start_time
                    
                    if decompression_result.success:
                        metrics = CompressionMetrics(
                            compression_ratio=compression_result.compression_ratio,
                            compression_time=compression_time,
                            decompression_time=decompression_time,
                            reconstruction_error=decompression_result.reconstruction_error,
                            memory_usage=0.0,
                            symbols_count=len(compression_result.compressed_data.payload.symbols)
                        )
                        
                        results.append(ValidationResult(
                            test_name=test_name,
                            data_type="scalability",
                            data_size=len(test_data),
                            success=True,
                            metrics=metrics
                        ))
                        
                        print(f"   ✅ Size: {len(test_data):,} bytes, "
                              f"Ratio: {metrics.compression_ratio:.1f}:1, "
                              f"Time: {compression_time:.2f}s + {decompression_time:.2f}s")
                    else:
                        results.append(ValidationResult(
                            test_name=test_name,
                            data_type="scalability",
                            data_size=len(test_data),
                            success=False,
                            metrics=None,
                            error_message=decompression_result.error_message
                        ))
                else:
                    results.append(ValidationResult(
                        test_name=test_name,
                        data_type="scalability",
                        data_size=len(test_data),
                        success=False,
                        metrics=None,
                        error_message=compression_result.error_message
                    ))
            
            except Exception as e:
                results.append(ValidationResult(
                    test_name=test_name,
                    data_type="scalability",
                    data_size=len(test_data) if 'test_data' in locals() else 0,
                    success=False,
                    metrics=None,
                    error_message=str(e)
                ))
                print(f"   ❌ Failed: {e}")
        
        self.results.extend(results)
        return results
    
    def run_performance_benchmarks(self, compressor, iterations: int = 5) -> Dict[str, Any]:
        """Run performance benchmarks with multiple iterations."""
        print(f"⚡ Running performance benchmarks ({iterations} iterations)")
        
        benchmark_data = TestDataGenerator.generate_geometric_image(128, 'simple')
        
        compression_times = []
        decompression_times = []
        compression_ratios = []
        reconstruction_errors = []
        
        for i in range(iterations):
            print(f"   Iteration {i+1}/{iterations}")
            
            # Compression benchmark
            start_time = time.time()
            compression_result = compressor.compress(benchmark_data)
            compression_time = time.time() - start_time
            compression_times.append(compression_time)
            
            if compression_result.success:
                compression_ratios.append(compression_result.compression_ratio)
                
                # Decompression benchmark
                start_time = time.time()
                decompression_result = compressor.decompress(compression_result.compressed_data)
                decompression_time = time.time() - start_time
                decompression_times.append(decompression_time)
                
                if decompression_result.success:
                    reconstruction_errors.append(decompression_result.reconstruction_error)
        
        benchmark_results = {
            'iterations': iterations,
            'data_size': len(benchmark_data),
            'compression_time': {
                'mean': np.mean(compression_times),
                'std': np.std(compression_times),
                'min': np.min(compression_times),
                'max': np.max(compression_times)
            },
            'decompression_time': {
                'mean': np.mean(decompression_times),
                'std': np.std(decompression_times),
                'min': np.min(decompression_times),
                'max': np.max(decompression_times)
            },
            'compression_ratio': {
                'mean': np.mean(compression_ratios),
                'std': np.std(compression_ratios),
                'min': np.min(compression_ratios),
                'max': np.max(compression_ratios)
            },
            'reconstruction_error': {
                'mean': np.mean(reconstruction_errors),
                'std': np.std(reconstruction_errors),
                'min': np.min(reconstruction_errors),
                'max': np.max(reconstruction_errors)
            }
        }
        
        print(f"📊 Benchmark Results:")
        print(f"   Compression: {benchmark_results['compression_time']['mean']:.3f}±{benchmark_results['compression_time']['std']:.3f}s")
        print(f"   Decompression: {benchmark_results['decompression_time']['mean']:.3f}±{benchmark_results['decompression_time']['std']:.3f}s")
        print(f"   Ratio: {benchmark_results['compression_ratio']['mean']:.1f}±{benchmark_results['compression_ratio']['std']:.1f}:1")
        print(f"   Error: {benchmark_results['reconstruction_error']['mean']:.8f}")
        
        return benchmark_results
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        if not self.results:
            return "No validation results available."
        
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        report = []
        report.append("# CIP-SC Validation Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- Total tests: {len(self.results)}")
        report.append(f"- Successful: {len(successful_tests)} ({len(successful_tests)/len(self.results)*100:.1f}%)")
        report.append(f"- Failed: {len(failed_tests)} ({len(failed_tests)/len(self.results)*100:.1f}%)")
        report.append("")
        
        if successful_tests:
            # Performance metrics
            ratios = [r.metrics.compression_ratio for r in successful_tests]
            errors = [r.metrics.reconstruction_error for r in successful_tests]
            times = [r.metrics.compression_time + r.metrics.decompression_time for r in successful_tests]
            
            report.append("## Performance Metrics")
            report.append(f"- Compression ratios: {np.min(ratios):.1f}:1 to {np.max(ratios):.1f}:1 (avg: {np.mean(ratios):.1f}:1)")
            report.append(f"- Reconstruction errors: {np.min(errors):.2e} to {np.max(errors):.2e} (avg: {np.mean(errors):.2e})")
            report.append(f"- Processing times: {np.min(times):.3f}s to {np.max(times):.3f}s (avg: {np.mean(times):.3f}s)")
            report.append("")
            
            # Test results by category
            categories = {}
            for result in successful_tests:
                category = result.data_type
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)
            
            report.append("## Results by Category")
            for category, results in categories.items():
                report.append(f"### {category.title()}")
                avg_ratio = np.mean([r.metrics.compression_ratio for r in results])
                avg_error = np.mean([r.metrics.reconstruction_error for r in results])
                report.append(f"- Tests: {len(results)}")
                report.append(f"- Average ratio: {avg_ratio:.1f}:1")
                report.append(f"- Average error: {avg_error:.2e}")
                report.append("")
        
        if failed_tests:
            report.append("## Failed Tests")
            for result in failed_tests:
                report.append(f"- {result.test_name}: {result.error_message}")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None) -> str:
        """Save validation results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"validation_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        results_data = {
            'timestamp': time.time(),
            'total_tests': len(self.results),
            'successful_tests': len([r for r in self.results if r.success]),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return str(filepath)
    
    def visualize_results(self, save_plots: bool = True) -> None:
        """Create visualization plots of validation results."""
        successful_results = [r for r in self.results if r.success and r.metrics]
        
        if not successful_results:
            print("No successful results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Compression ratios
        ratios = [r.metrics.compression_ratio for r in successful_results]
        labels = [r.test_name for r in successful_results]
        
        axes[0, 0].bar(range(len(ratios)), ratios)
        axes[0, 0].set_title('Compression Ratios')
        axes[0, 0].set_ylabel('Ratio (X:1)')
        axes[0, 0].set_xticks(range(len(labels)))
        axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
        
        # Reconstruction errors
        errors = [r.metrics.reconstruction_error for r in successful_results]
        axes[0, 1].semilogy(range(len(errors)), errors, 'ro-')
        axes[0, 1].set_title('Reconstruction Errors')
        axes[0, 1].set_ylabel('Error (log scale)')
        axes[0, 1].set_xticks(range(len(labels)))
        axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
        
        # Processing times
        comp_times = [r.metrics.compression_time for r in successful_results]
        decomp_times = [r.metrics.decompression_time for r in successful_results]
        
        x = np.arange(len(labels))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, comp_times, width, label='Compression')
        axes[1, 0].bar(x + width/2, decomp_times, width, label='Decompression')
        axes[1, 0].set_title('Processing Times')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 0].legend()
        
        # Efficiency scores
        efficiency_scores = [r.metrics.get_efficiency_score() for r in successful_results]
        axes[1, 1].bar(range(len(efficiency_scores)), efficiency_scores)
        axes[1, 1].set_title('Efficiency Scores')
        axes[1, 1].set_ylabel('Score (0-1)')
        axes[1, 1].set_xticks(range(len(labels)))
        axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_file = self.output_dir / f"validation_plots_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to {plot_file}")
        
        plt.show()

# ============================================================================
# Example Usage
# ============================================================================

def run_comprehensive_validation():
    """Run comprehensive validation suite."""
    print("🧪 CIP-SC Comprehensive Validation Suite")
    print("=" * 50)
    
    # This would import from the actual compression engine
    # For now, we'll create a mock for demonstration
    class MockGeometricCompressor:
        def __init__(self):
            self.compressor_id = "cipsc.geometric.v1"
        
        def compress(self, data):
            # Mock compression with realistic results
            from dataclasses import dataclass
            
            @dataclass
            class MockResult:
                success: bool = True
                compression_ratio: float = 1024.0
                error_message: str = ""
                compressed_data: Any = None
            
            @dataclass 
            class MockCompressed:
                payload: Any = None
            
            @dataclass
            class MockPayload:
                symbols: List = None
            
            result = MockResult()
            result.compressed_data = MockCompressed()
            result.compressed_data.payload = MockPayload()
            result.compressed_data.payload.symbols = ['symbol1', 'symbol2', 'symbol3']
            
            return result
        
        def decompress(self, compressed):
            @dataclass
            class MockDecompResult:
                success: bool = True
                reconstruction_error: float = 0.0
                error_message: str = ""
            
            return MockDecompResult()
    
    # Initialize validation suite
    validator = ValidationSuite("validation_results")
    
    # Initialize mock compressor
    compressor = MockGeometricCompressor()
    
    # Run validation tests
    print("\n🔬 Running geometric validation...")
    geometric_results = validator.run_geometric_validation(
        compressor, 
        sizes=[64, 128], 
        complexities=['simple', 'medium']
    )
    
    print(f"\n📈 Running scalability test...")
    scalability_results = validator.run_scalability_test(compressor, max_size=256)
    
    print(f"\n⚡ Running performance benchmarks...")
    benchmark_results = validator.run_performance_benchmarks(compressor, iterations=3)
    
    # Generate and save report
    print(f"\n📋 Generating validation report...")
    report = validator.generate_validation_report()
    
    report_file = validator.output_dir / "validation_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to {report_file}")
    
    # Save results
    results_file = validator.save_results()
    print(f"💾 Results saved to {results_file}")
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    validator.visualize_results(save_plots=True)
    
    print(f"\n🎉 Validation complete!")
    print(f"📁 Results available in: {validator.output_dir}")

if __name__ == "__main__":
    run_comprehensive_validation()
