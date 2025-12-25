"""
PAC Physics Engine - Benchmarking Suite
=======================================

Comprehensive performance benchmarking and scaling analysis
for the PAC Physics Engine across different hardware configurations.

Author: GitHub Copilot
Date: September 2025
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import platform
import subprocess
import os
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    test_name: str
    lattice_size: int
    simulation_steps: int
    total_time: float
    time_per_step: float
    memory_peak_mb: float
    memory_average_mb: float
    cpu_utilization: float
    gpu_utilization: float
    conservation_accuracy: float
    signature_detection_rate: float
    throughput_points_per_second: float
    
@dataclass 
class SystemInfo:
    """System hardware and software information"""
    platform: str
    processor: str
    total_memory_gb: float
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str
    gpu_name: str
    gpu_memory_gb: float

class PACBenchmarkSuite:
    """
    Comprehensive benchmarking suite for PAC Physics Engine.
    
    Tests performance across:
    - Different lattice sizes
    - Various simulation lengths  
    - Multiple hardware configurations
    - Conservation accuracy vs speed trade-offs
    """
    
    def __init__(self, output_dir: str = "benchmarks"):
        """Initialize benchmarking suite"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.system_info = self._collect_system_info()
        self.benchmark_results = []
        
        # Benchmark configurations
        self.test_configs = {
            'scaling_test': [
                {'size': 8, 'steps': 100},
                {'size': 12, 'steps': 100}, 
                {'size': 16, 'steps': 100},
                {'size': 20, 'steps': 100},
                {'size': 24, 'steps': 100}
            ],
            'endurance_test': [
                {'size': 16, 'steps': 1000},
                {'size': 16, 'steps': 2000},
                {'size': 16, 'steps': 5000}
            ],
            'accuracy_test': [
                {'size': 16, 'steps': 200, 'tolerance': 1e-6},
                {'size': 16, 'steps': 200, 'tolerance': 1e-9},
                {'size': 16, 'steps': 200, 'tolerance': 1e-12}
            ]
        }
    
    def run_full_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """
        Run complete benchmark suite.
        
        Returns:
            Dictionary of benchmark results by test type
        """
        print("="*70)
        print("PAC PHYSICS ENGINE - COMPREHENSIVE BENCHMARK SUITE")
        print("="*70)
        print(f"System: {self.system_info.platform}")
        print(f"Processor: {self.system_info.processor}")
        print(f"Memory: {self.system_info.total_memory_gb:.1f} GB")
        print(f"GPU: {self.system_info.gpu_name} ({self.system_info.gpu_memory_gb:.1f} GB)")
        print("-"*70)
        
        all_results = {}
        
        # Run scaling tests
        print("\n🔬 SCALING PERFORMANCE TESTS")
        print("Testing performance across different lattice sizes...")
        scaling_results = self._run_scaling_tests()
        all_results['scaling'] = scaling_results
        
        # Run endurance tests
        print("\n⏱️  ENDURANCE TESTS")
        print("Testing long-running simulation stability...")
        endurance_results = self._run_endurance_tests()
        all_results['endurance'] = endurance_results
        
        # Run accuracy tests
        print("\n🎯 ACCURACY vs PERFORMANCE TESTS")
        print("Testing conservation accuracy at different tolerances...")
        accuracy_results = self._run_accuracy_tests()
        all_results['accuracy'] = accuracy_results
        
        # Generate comprehensive report
        self._generate_benchmark_report(all_results)
        
        print(f"\n📊 Benchmark results saved to: {self.output_dir}")
        print("="*70)
        
        return all_results
    
    def _run_scaling_tests(self) -> List[BenchmarkResult]:
        """Test performance scaling with lattice size"""
        results = []
        
        for config in self.test_configs['scaling_test']:
            print(f"  Testing {config['size']}³ lattice, {config['steps']} steps...")
            
            result = self._run_single_benchmark(
                test_name=f"scaling_{config['size']}x{config['size']}x{config['size']}",
                lattice_size=config['size'],
                simulation_steps=config['steps']
            )
            
            results.append(result)
            print(f"    Time: {result.total_time:.2f}s, "
                  f"Memory: {result.memory_peak_mb:.1f}MB, "
                  f"Throughput: {result.throughput_points_per_second:.0f} pts/s")
        
        return results
    
    def _run_endurance_tests(self) -> List[BenchmarkResult]:
        """Test long-running simulation performance"""
        results = []
        
        for config in self.test_configs['endurance_test']:
            print(f"  Testing {config['steps']} steps endurance...")
            
            result = self._run_single_benchmark(
                test_name=f"endurance_{config['steps']}_steps",
                lattice_size=config['size'],
                simulation_steps=config['steps']
            )
            
            results.append(result)
            print(f"    Time: {result.total_time:.2f}s, "
                  f"Avg time/step: {result.time_per_step:.4f}s, "
                  f"Conservation: {result.conservation_accuracy:.6f}")
        
        return results
    
    def _run_accuracy_tests(self) -> List[BenchmarkResult]:
        """Test accuracy vs performance trade-offs"""
        results = []
        
        for config in self.test_configs['accuracy_test']:
            tolerance = config['tolerance']
            print(f"  Testing tolerance {tolerance:.0e}...")
            
            result = self._run_single_benchmark(
                test_name=f"accuracy_tol_{tolerance:.0e}",
                lattice_size=config['size'],
                simulation_steps=config['steps'],
                tolerance=tolerance
            )
            
            results.append(result)
            print(f"    Time: {result.total_time:.2f}s, "
                  f"Conservation: {result.conservation_accuracy:.9f}")
        
        return results
    
    def _run_single_benchmark(self, 
                            test_name: str,
                            lattice_size: int,
                            simulation_steps: int,
                            tolerance: float = 1e-12) -> BenchmarkResult:
        """
        Run a single benchmark test.
        
        Args:
            test_name: Name of the test
            lattice_size: Size of cubic lattice
            simulation_steps: Number of evolution steps
            tolerance: Conservation tolerance
        
        Returns:
            BenchmarkResult object
        """
        # Import here to avoid circular dependencies
        from .lattice_substrate import MultiScaleLatticeSubstrate, ScaleType
        from .pac_kernel import ConservationType
        
        # Initialize system monitoring
        process = psutil.Process()
        memory_samples = []
        cpu_samples = []
        
        # Record start state
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.time()
        
        # GPU monitoring setup
        gpu_utilization = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        try:
            # Initialize lattice
            lattice = MultiScaleLatticeSubstrate(
                dimensions=(lattice_size, lattice_size, lattice_size),
                active_scales=[ScaleType.QUANTUM, ScaleType.GEOMETRIC, 
                              ScaleType.INFORMATION],
                device="auto"
            )
            
            # Set conservation tolerance
            lattice.pac_kernel.tolerance = tolerance
            
            # Run simulation with monitoring
            conservation_accuracies = []
            signature_detections = 0
            
            for step in range(simulation_steps):
                # Evolution step
                step_start = time.time()
                evolution_metrics = lattice.evolve_one_step()
                
                # Monitor system resources
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent()
                memory_samples.append(memory_mb)
                cpu_samples.append(cpu_percent)
                
                # Track conservation accuracy
                if 'pac_conservation' in evolution_metrics:
                    pac_metrics = evolution_metrics['pac_conservation']
                    conservation_accuracies.append(pac_metrics.get('post_residual_norm', 1.0))
                
                # Track signature detection
                system_state = lattice.get_system_state()
                if 'signatures' in system_state['pac_state'] and system_state['pac_state']['signatures']:
                    signature_detections += 1
                
                # Progress reporting for long tests
                if simulation_steps > 500 and step % 100 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / (step + 1)) * (simulation_steps - step - 1)
                    print(f"    Progress: {step+1}/{simulation_steps} "
                          f"(ETA: {eta:.1f}s)")
        
        except Exception as e:
            print(f"    ⚠️  Benchmark failed: {e}")
            # Return dummy result for failed test
            return BenchmarkResult(
                test_name=test_name,
                lattice_size=lattice_size,
                simulation_steps=simulation_steps,
                total_time=0.0,
                time_per_step=0.0,
                memory_peak_mb=0.0,
                memory_average_mb=0.0,
                cpu_utilization=0.0,
                gpu_utilization=0.0,
                conservation_accuracy=0.0,
                signature_detection_rate=0.0,
                throughput_points_per_second=0.0
            )
        
        # Calculate final metrics
        end_time = time.time()
        total_time = end_time - start_time
        time_per_step = total_time / simulation_steps
        
        # Memory statistics
        memory_peak_mb = max(memory_samples) if memory_samples else start_memory
        memory_average_mb = np.mean(memory_samples) if memory_samples else start_memory
        
        # CPU utilization
        cpu_utilization = np.mean(cpu_samples) if cpu_samples else 0.0
        
        # GPU utilization (simplified)
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            gpu_utilization = min(100.0, (gpu_memory_allocated / self.system_info.gpu_memory_gb) * 100)
        
        # Conservation accuracy
        conservation_accuracy = 1.0 - np.mean(conservation_accuracies) if conservation_accuracies else 0.0
        
        # Signature detection rate
        signature_detection_rate = signature_detections / simulation_steps
        
        # Throughput
        total_points = lattice_size ** 3
        throughput_points_per_second = (total_points * simulation_steps) / total_time
        
        result = BenchmarkResult(
            test_name=test_name,
            lattice_size=lattice_size,
            simulation_steps=simulation_steps,
            total_time=total_time,
            time_per_step=time_per_step,
            memory_peak_mb=memory_peak_mb,
            memory_average_mb=memory_average_mb,
            cpu_utilization=cpu_utilization,
            gpu_utilization=gpu_utilization,
            conservation_accuracy=conservation_accuracy,
            signature_detection_rate=signature_detection_rate,
            throughput_points_per_second=throughput_points_per_second
        )
        
        self.benchmark_results.append(result)
        return result
    
    def _collect_system_info(self) -> SystemInfo:
        """Collect system hardware and software information"""
        
        # Basic system info
        platform_info = platform.platform()
        processor_info = platform.processor() or "Unknown"
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        python_version = platform.python_version()
        
        # PyTorch info
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        # CUDA info
        cuda_version = "N/A"
        gpu_name = "N/A"
        gpu_memory_gb = 0.0
        
        if cuda_available:
            cuda_version = torch.version.cuda or "Unknown"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return SystemInfo(
            platform=platform_info,
            processor=processor_info,
            total_memory_gb=total_memory,
            python_version=python_version,
            torch_version=torch_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb
        )
    
    def _generate_benchmark_report(self, results: Dict[str, List[BenchmarkResult]]):
        """Generate comprehensive benchmark report"""
        
        # Save raw results as JSON
        json_data = {
            'system_info': asdict(self.system_info),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': {
                category: [asdict(result) for result in result_list]
                for category, result_list in results.items()
            }
        }
        
        json_path = self.output_dir / 'benchmark_results.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Generate summary report
        report_path = self.output_dir / 'benchmark_summary.txt'
        with open(report_path, 'w') as f:
            f.write("PAC PHYSICS ENGINE BENCHMARK REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"System: {self.system_info.platform}\n")
            f.write(f"Processor: {self.system_info.processor}\n")
            f.write(f"Memory: {self.system_info.total_memory_gb:.1f} GB\n")
            f.write(f"GPU: {self.system_info.gpu_name}\n\n")
            
            # Scaling test summary
            if 'scaling' in results:
                f.write("SCALING TEST RESULTS\n")
                f.write("-" * 30 + "\n")
                for result in results['scaling']:
                    f.write(f"  {result.lattice_size}³ lattice: "
                           f"{result.total_time:.2f}s, "
                           f"{result.throughput_points_per_second:.0f} pts/s\n")
                f.write("\n")
            
            # Endurance test summary
            if 'endurance' in results:
                f.write("ENDURANCE TEST RESULTS\n")
                f.write("-" * 30 + "\n")
                for result in results['endurance']:
                    f.write(f"  {result.simulation_steps} steps: "
                           f"{result.total_time:.2f}s, "
                           f"{result.time_per_step:.4f}s/step\n")
                f.write("\n")
            
            # Accuracy test summary
            if 'accuracy' in results:
                f.write("ACCURACY TEST RESULTS\n")
                f.write("-" * 30 + "\n")
                for result in results['accuracy']:
                    f.write(f"  {result.test_name}: "
                           f"{result.conservation_accuracy:.9f} accuracy, "
                           f"{result.total_time:.2f}s\n")
                f.write("\n")
        
        print(f"📊 Detailed results saved to {json_path}")
        print(f"📄 Summary report saved to {report_path}")

def run_quick_benchmark() -> Dict:
    """Run a quick performance benchmark"""
    benchmark = PACBenchmarkSuite("quick_benchmark")
    
    # Quick test configuration
    benchmark.test_configs = {
        'scaling_test': [
            {'size': 8, 'steps': 50},
            {'size': 12, 'steps': 50},
            {'size': 16, 'steps': 50}
        ],
        'endurance_test': [
            {'size': 12, 'steps': 200}
        ],
        'accuracy_test': [
            {'size': 12, 'steps': 100, 'tolerance': 1e-9}
        ]
    }
    
    return benchmark.run_full_benchmark()

if __name__ == "__main__":
    # Run quick benchmark for testing
    results = run_quick_benchmark()
    print("\nQuick benchmark completed!")
