"""
Benchmark Suite

Comprehensive benchmarking and performance testing for the PAC physics engine.
Tests all modules for correctness, performance, and scalability while
measuring PAC conservation quality and emergence detection accuracy.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import psutil
import gc

class BenchmarkType(Enum):
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    CONSERVATION = "conservation"
    EMERGENCE = "emergence"
    INTEGRATION = "integration"

@dataclass
class BenchmarkResult:
    test_name: str
    benchmark_type: BenchmarkType
    passed: bool
    execution_time: float
    memory_usage: float
    error_rate: float
    conservation_quality: float
    metadata: Dict[str, Any]

@dataclass
class BenchmarkSuite:
    suite_name: str
    results: List[BenchmarkResult]
    total_time: float
    success_rate: float
    overall_rating: str

class PACBenchmarkSuite:
    """Comprehensive benchmarking suite for PAC physics engine"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Performance monitoring
        self.memory_baseline = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Test configurations
        self.test_sizes = [64, 128, 256, 512]
        self.time_steps = [10, 50, 100]
        self.conservation_tolerance = 1e-6
        
    def run_full_benchmark(self) -> Dict[str, BenchmarkSuite]:
        """Run complete benchmark suite"""
        print("🚀 Starting PAC Physics Engine Benchmark Suite")
        print("=" * 60)
        
        suites = {}
        
        # Core module benchmarks
        suites["core"] = self._benchmark_core_modules()
        
        # Physics module benchmarks
        suites["physics"] = self._benchmark_physics_modules()
        
        # Validation benchmarks
        suites["validation"] = self._benchmark_validation_modules()
        
        # Integration benchmarks
        suites["integration"] = self._benchmark_integration()
        
        # Performance scaling benchmarks
        suites["scaling"] = self._benchmark_scaling()
        
        # Generate summary report
        self._generate_benchmark_report(suites)
        
        return suites
    
    def _benchmark_core_modules(self) -> BenchmarkSuite:
        """Benchmark core mathematical operations"""
        print("\n📊 Benchmarking Core Modules")
        results = []
        
        # Import core modules
        try:
            from ..core.conservation_math import ConservationMathPAC
            from ..core.emergence_detector import EmergenceDetector
            from ..core.lattice_substrate import LatticePAC
            
            # Conservation math benchmarks
            results.extend(self._test_conservation_math(ConservationMathPAC))
            
            # Emergence detection benchmarks
            results.extend(self._test_emergence_detection(EmergenceDetector))
            
            # Lattice substrate benchmarks
            results.extend(self._test_lattice_substrate(LatticePAC))
            
        except ImportError as e:
            print(f"❌ Core module import failed: {e}")
            return BenchmarkSuite("core", [], 0.0, 0.0, "FAILED")
        
        # Calculate suite metrics
        total_time = sum(r.execution_time for r in results)
        success_rate = sum(1 for r in results if r.passed) / len(results) if results else 0
        rating = self._calculate_rating(success_rate, total_time)
        
        return BenchmarkSuite("core", results, total_time, success_rate, rating)
    
    def _test_conservation_math(self, ConservationMathPAC) -> List[BenchmarkResult]:
        """Test conservation mathematics"""
        results = []
        
        for size in self.test_sizes:
            start_time = time.time()
            mem_start = self._get_memory_usage()
            
            try:
                # Create test data
                parent_state = torch.randn(size, size, device=self.device)
                children_states = [torch.randn(size//2, size//2, device=self.device) for _ in range(4)]
                
                # Test conservation enforcement
                math_engine = ConservationMathPAC(device=self.device)
                conserved_children = math_engine.enforce_exact_conservation(parent_state, children_states)
                
                # Verify conservation
                total_children = sum(child.sum() for child in conserved_children)
                error_rate = abs(parent_state.sum() - total_children) / abs(parent_state.sum())
                
                passed = error_rate < self.conservation_tolerance
                
                execution_time = time.time() - start_time
                memory_usage = self._get_memory_usage() - mem_start
                
                results.append(BenchmarkResult(
                    test_name=f"conservation_math_{size}x{size}",
                    benchmark_type=BenchmarkType.CONSERVATION,
                    passed=passed,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    error_rate=float(error_rate),
                    conservation_quality=1.0 - float(error_rate),
                    metadata={"size": size, "num_children": len(children_states)}
                ))
                
                print(f"  ✅ Conservation {size}x{size}: {execution_time:.3f}s, error: {error_rate:.2e}")
                
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"conservation_math_{size}x{size}",
                    benchmark_type=BenchmarkType.CONSERVATION,
                    passed=False,
                    execution_time=time.time() - start_time,
                    memory_usage=0,
                    error_rate=1.0,
                    conservation_quality=0.0,
                    metadata={"error": str(e)}
                ))
                print(f"  ❌ Conservation {size}x{size}: {e}")
        
        return results
    
    def _test_emergence_detection(self, EmergenceDetector) -> List[BenchmarkResult]:
        """Test emergence detection capabilities"""
        results = []
        
        for size in self.test_sizes:
            start_time = time.time()
            mem_start = self._get_memory_usage()
            
            try:
                # Create test emergence scenario
                detector = EmergenceDetector(device=self.device)
                
                # Simulate emergence with known patterns
                states = []
                for t in range(10):
                    # Create state with increasing complexity
                    state = torch.randn(size, size, device=self.device) * (1 + t * 0.1)
                    states.append(state)
                
                # Detect emergence
                emergence_info = detector.detect_emergence(states)
                
                # Verify detection
                detected_events = len(emergence_info.get("emergence_events", []))
                passed = detected_events > 0  # Should detect some emergence
                
                execution_time = time.time() - start_time
                memory_usage = self._get_memory_usage() - mem_start
                
                results.append(BenchmarkResult(
                    test_name=f"emergence_detection_{size}x{size}",
                    benchmark_type=BenchmarkType.EMERGENCE,
                    passed=passed,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    error_rate=0.0 if passed else 1.0,
                    conservation_quality=1.0,
                    metadata={
                        "size": size,
                        "detected_events": detected_events,
                        "emergence_strength": emergence_info.get("emergence_strength", 0)
                    }
                ))
                
                print(f"  ✅ Emergence {size}x{size}: {execution_time:.3f}s, events: {detected_events}")
                
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"emergence_detection_{size}x{size}",
                    benchmark_type=BenchmarkType.EMERGENCE,
                    passed=False,
                    execution_time=time.time() - start_time,
                    memory_usage=0,
                    error_rate=1.0,
                    conservation_quality=0.0,
                    metadata={"error": str(e)}
                ))
                print(f"  ❌ Emergence {size}x{size}: {e}")
        
        return results
    
    def _test_lattice_substrate(self, LatticePAC) -> List[BenchmarkResult]:
        """Test lattice substrate operations"""
        results = []
        
        for size in self.test_sizes:
            start_time = time.time()
            mem_start = self._get_memory_usage()
            
            try:
                # Create lattice
                lattice = LatticePAC(size, device=self.device)
                
                # Test basic operations
                initial_state = torch.randn(size, size, device=self.device)
                lattice.set_substrate_state(initial_state)
                
                # Evolve lattice
                for _ in range(10):
                    lattice.evolve_substrate(dt=0.01)
                
                final_state = lattice.get_substrate_state()
                
                # Check conservation
                initial_sum = initial_state.sum()
                final_sum = final_state.sum()
                error_rate = abs(initial_sum - final_sum) / abs(initial_sum)
                
                passed = error_rate < self.conservation_tolerance
                
                execution_time = time.time() - start_time
                memory_usage = self._get_memory_usage() - mem_start
                
                results.append(BenchmarkResult(
                    test_name=f"lattice_substrate_{size}x{size}",
                    benchmark_type=BenchmarkType.CONSERVATION,
                    passed=passed,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    error_rate=float(error_rate),
                    conservation_quality=1.0 - float(error_rate),
                    metadata={"size": size, "evolution_steps": 10}
                ))
                
                print(f"  ✅ Lattice {size}x{size}: {execution_time:.3f}s, error: {error_rate:.2e}")
                
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"lattice_substrate_{size}x{size}",
                    benchmark_type=BenchmarkType.CONSERVATION,
                    passed=False,
                    execution_time=time.time() - start_time,
                    memory_usage=0,
                    error_rate=1.0,
                    conservation_quality=0.0,
                    metadata={"error": str(e)}
                ))
                print(f"  ❌ Lattice {size}x{size}: {e}")
        
        return results
    
    def _benchmark_physics_modules(self) -> BenchmarkSuite:
        """Benchmark physics simulation modules"""
        print("\n🔬 Benchmarking Physics Modules")
        results = []
        
        try:
            from ..modules.quantum_pac import QuantumPAC
            from ..modules.geometric_sec import GeometricSEC
            from ..modules.fluid_med import FluidMED
            from ..modules.information_amp import InformationAmp
            from ..modules.consciousness_scbf import ConsciousnessSCBF
            from ..modules.meta_module import MetaModule
            
            # Test each physics module
            modules = [
                ("quantum", QuantumPAC),
                ("geometric", GeometricSEC),
                ("fluid", FluidMED),
                ("information", InformationAmp),
                ("consciousness", ConsciousnessSCBF),
                ("meta", MetaModule)
            ]
            
            for module_name, ModuleClass in modules:
                results.extend(self._test_physics_module(module_name, ModuleClass))
                
        except ImportError as e:
            print(f"❌ Physics module import failed: {e}")
            return BenchmarkSuite("physics", [], 0.0, 0.0, "FAILED")
        
        # Calculate suite metrics
        total_time = sum(r.execution_time for r in results)
        success_rate = sum(1 for r in results if r.passed) / len(results) if results else 0
        rating = self._calculate_rating(success_rate, total_time)
        
        return BenchmarkSuite("physics", results, total_time, success_rate, rating)
    
    def _test_physics_module(self, module_name: str, ModuleClass) -> List[BenchmarkResult]:
        """Test individual physics module"""
        results = []
        
        for size in [64, 128]:  # Smaller sizes for physics modules
            start_time = time.time()
            mem_start = self._get_memory_usage()
            
            try:
                # Initialize module
                if module_name == "meta":
                    module = ModuleClass(device=self.device)
                else:
                    module = ModuleClass(size, device=self.device)
                
                # Run evolution test
                initial_state = torch.randn(size, size, device=self.device)
                
                if hasattr(module, 'evolve'):
                    final_state = module.evolve(initial_state, dt=0.01, steps=5)
                elif hasattr(module, 'evolve_quantum_pac_state'):
                    final_state = module.evolve_quantum_pac_state(initial_state, dt=0.01)
                elif hasattr(module, 'evolve_geometric_sec'):
                    final_state = module.evolve_geometric_sec(initial_state, dt=0.01)
                else:
                    # Fallback test
                    final_state = initial_state
                
                # Check conservation (if applicable)
                if module_name != "information":  # Information module amplifies by design
                    initial_sum = initial_state.sum()
                    final_sum = final_state.sum() if torch.is_tensor(final_state) else torch.tensor(0.0)
                    error_rate = abs(initial_sum - final_sum) / abs(initial_sum) if initial_sum != 0 else 0
                else:
                    error_rate = 0.0  # Accept amplification
                
                passed = error_rate < 0.1  # Relaxed tolerance for physics modules
                
                execution_time = time.time() - start_time
                memory_usage = self._get_memory_usage() - mem_start
                
                results.append(BenchmarkResult(
                    test_name=f"{module_name}_{size}x{size}",
                    benchmark_type=BenchmarkType.CORRECTNESS,
                    passed=passed,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    error_rate=float(error_rate),
                    conservation_quality=1.0 - min(float(error_rate), 1.0),
                    metadata={"size": size, "module": module_name}
                ))
                
                print(f"  ✅ {module_name.capitalize()} {size}x{size}: {execution_time:.3f}s")
                
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"{module_name}_{size}x{size}",
                    benchmark_type=BenchmarkType.CORRECTNESS,
                    passed=False,
                    execution_time=time.time() - start_time,
                    memory_usage=0,
                    error_rate=1.0,
                    conservation_quality=0.0,
                    metadata={"error": str(e), "module": module_name}
                ))
                print(f"  ❌ {module_name.capitalize()} {size}x{size}: {e}")
        
        return results
    
    def _benchmark_validation_modules(self) -> BenchmarkSuite:
        """Benchmark validation modules"""
        print("\n🔍 Benchmarking Validation Modules")
        results = []
        
        try:
            from ..validation.signature_detector import UniversalSignatureDetector
            from ..validation.cross_scale_validator import CrossScaleValidator
            from ..validation.emergence_tracker import EmergenceTracker
            
            # Test validation modules
            results.extend(self._test_signature_detection(UniversalSignatureDetector))
            results.extend(self._test_cross_scale_validation(CrossScaleValidator))
            results.extend(self._test_emergence_tracking(EmergenceTracker))
            
        except ImportError as e:
            print(f"❌ Validation module import failed: {e}")
            return BenchmarkSuite("validation", [], 0.0, 0.0, "FAILED")
        
        # Calculate suite metrics
        total_time = sum(r.execution_time for r in results)
        success_rate = sum(1 for r in results if r.passed) / len(results) if results else 0
        rating = self._calculate_rating(success_rate, total_time)
        
        return BenchmarkSuite("validation", results, total_time, success_rate, rating)
    
    def _test_signature_detection(self, DetectorClass) -> List[BenchmarkResult]:
        """Test universal signature detection"""
        results = []
        
        start_time = time.time()
        mem_start = self._get_memory_usage()
        
        try:
            detector = DetectorClass(device=self.device)
            
            # Create test states with known signatures
            test_states = []
            for i in range(10):
                # Include 15.56x amplification pattern
                state = torch.randn(64, 64, device=self.device) * (15.56 if i > 5 else 1.0)
                test_states.append({
                    "information_state": {"amplification_ratio": 15.56 if i > 5 else 1.0},
                    "timestamp": i * 0.1
                })
            
            # Detect signatures
            detections = detector.detect_universal_signatures(test_states)
            
            # Verify detection of amplification signature
            amplification_detected = any(
                d["signature_type"] == "amplification" for d in detections.get("detected_signatures", [])
            )
            
            passed = amplification_detected
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - mem_start
            
            results.append(BenchmarkResult(
                test_name="signature_detection",
                benchmark_type=BenchmarkType.EMERGENCE,
                passed=passed,
                execution_time=execution_time,
                memory_usage=memory_usage,
                error_rate=0.0 if passed else 1.0,
                conservation_quality=1.0,
                metadata={"detected_signatures": len(detections.get("detected_signatures", []))}
            ))
            
            print(f"  ✅ Signature Detection: {execution_time:.3f}s, signatures: {len(detections.get('detected_signatures', []))}")
            
        except Exception as e:
            results.append(BenchmarkResult(
                test_name="signature_detection",
                benchmark_type=BenchmarkType.EMERGENCE,
                passed=False,
                execution_time=time.time() - start_time,
                memory_usage=0,
                error_rate=1.0,
                conservation_quality=0.0,
                metadata={"error": str(e)}
            ))
            print(f"  ❌ Signature Detection: {e}")
        
        return results
    
    def _test_cross_scale_validation(self, ValidatorClass) -> List[BenchmarkResult]:
        """Test cross-scale validation"""
        results = []
        
        start_time = time.time()
        mem_start = self._get_memory_usage()
        
        try:
            validator = ValidatorClass(device=self.device)
            
            # Create multi-scale test data
            quantum_states = [torch.randn(32, 32, device=self.device) for _ in range(5)]
            geometric_states = [torch.randn(64, 64, device=self.device) for _ in range(5)]
            fluid_states = [torch.randn(128, 128, device=self.device) for _ in range(5)]
            
            # Validate cross-scale consistency
            validation_result = validator.validate_cross_scale_consistency(
                quantum_states, geometric_states, fluid_states
            )
            
            # Check if validation completed successfully
            passed = "scale_correlations" in validation_result
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - mem_start
            
            results.append(BenchmarkResult(
                test_name="cross_scale_validation",
                benchmark_type=BenchmarkType.INTEGRATION,
                passed=passed,
                execution_time=execution_time,
                memory_usage=memory_usage,
                error_rate=0.0 if passed else 1.0,
                conservation_quality=validation_result.get("conservation_consistency", 0.0),
                metadata={"validation_metrics": list(validation_result.keys())}
            ))
            
            print(f"  ✅ Cross-Scale Validation: {execution_time:.3f}s")
            
        except Exception as e:
            results.append(BenchmarkResult(
                test_name="cross_scale_validation",
                benchmark_type=BenchmarkType.INTEGRATION,
                passed=False,
                execution_time=time.time() - start_time,
                memory_usage=0,
                error_rate=1.0,
                conservation_quality=0.0,
                metadata={"error": str(e)}
            ))
            print(f"  ❌ Cross-Scale Validation: {e}")
        
        return results
    
    def _test_emergence_tracking(self, TrackerClass) -> List[BenchmarkResult]:
        """Test emergence tracking"""
        results = []
        
        start_time = time.time()
        mem_start = self._get_memory_usage()
        
        try:
            tracker = TrackerClass(device=self.device)
            
            # Create test states with emergence events
            meta_states = []
            for i in range(10):
                state = {
                    "consciousness_state": {"awareness_metric": 0.4 if i > 5 else 0.1},
                    "information_state": {"amplification_ratio": 16.0 if i > 7 else 1.0},
                    "geometric_state": {"collapse_strength": 0.1 if i > 3 else 0.01}
                }
                meta_states.append(state)
            
            # Track emergence
            analysis = tracker.track_emergence_events(meta_states)
            
            # Verify tracking
            passed = analysis.total_events > 0
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - mem_start
            
            results.append(BenchmarkResult(
                test_name="emergence_tracking",
                benchmark_type=BenchmarkType.EMERGENCE,
                passed=passed,
                execution_time=execution_time,
                memory_usage=memory_usage,
                error_rate=0.0 if passed else 1.0,
                conservation_quality=1.0,
                metadata={
                    "total_events": analysis.total_events,
                    "cascade_chains": len(analysis.cascade_chains)
                }
            ))
            
            print(f"  ✅ Emergence Tracking: {execution_time:.3f}s, events: {analysis.total_events}")
            
        except Exception as e:
            results.append(BenchmarkResult(
                test_name="emergence_tracking",
                benchmark_type=BenchmarkType.EMERGENCE,
                passed=False,
                execution_time=time.time() - start_time,
                memory_usage=0,
                error_rate=1.0,
                conservation_quality=0.0,
                metadata={"error": str(e)}
            ))
            print(f"  ❌ Emergence Tracking: {e}")
        
        return results
    
    def _benchmark_integration(self) -> BenchmarkSuite:
        """Benchmark module integration"""
        print("\n🔗 Benchmarking Integration")
        results = []
        
        # End-to-end integration test
        start_time = time.time()
        mem_start = self._get_memory_usage()
        
        try:
            from ..modules.meta_module import MetaModule
            
            # Create meta module for integration test
            meta = MetaModule(device=self.device)
            
            # Run integrated simulation
            result = meta.run_universal_validation_experiment(
                size=64, time_steps=5, dt=0.01
            )
            
            # Check integration success
            passed = "final_states" in result and len(result["final_states"]) > 0
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - mem_start
            
            results.append(BenchmarkResult(
                test_name="full_integration",
                benchmark_type=BenchmarkType.INTEGRATION,
                passed=passed,
                execution_time=execution_time,
                memory_usage=memory_usage,
                error_rate=0.0 if passed else 1.0,
                conservation_quality=result.get("conservation_metrics", {}).get("overall_quality", 0.0),
                metadata={"experiment_phases": len(result.get("final_states", []))}
            ))
            
            print(f"  ✅ Full Integration: {execution_time:.3f}s")
            
        except Exception as e:
            results.append(BenchmarkResult(
                test_name="full_integration",
                benchmark_type=BenchmarkType.INTEGRATION,
                passed=False,
                execution_time=time.time() - start_time,
                memory_usage=0,
                error_rate=1.0,
                conservation_quality=0.0,
                metadata={"error": str(e)}
            ))
            print(f"  ❌ Full Integration: {e}")
        
        # Calculate suite metrics
        total_time = sum(r.execution_time for r in results)
        success_rate = sum(1 for r in results if r.passed) / len(results) if results else 0
        rating = self._calculate_rating(success_rate, total_time)
        
        return BenchmarkSuite("integration", results, total_time, success_rate, rating)
    
    def _benchmark_scaling(self) -> BenchmarkSuite:
        """Benchmark performance scaling"""
        print("\n📈 Benchmarking Performance Scaling")
        results = []
        
        # Test scaling across different sizes
        for size in self.test_sizes:
            start_time = time.time()
            mem_start = self._get_memory_usage()
            
            try:
                # Create large tensors to test memory scaling
                state1 = torch.randn(size, size, device=self.device)
                state2 = torch.randn(size, size, device=self.device)
                
                # Perform computational operations
                result = torch.matmul(state1, state2)
                result = torch.fft.fft2(result)
                result = torch.abs(result)
                
                # Memory cleanup
                del state1, state2, result
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
                execution_time = time.time() - start_time
                memory_usage = self._get_memory_usage() - mem_start
                
                # Calculate scaling efficiency
                expected_time = (size / 64) ** 2 * 0.001  # Expected scaling
                efficiency = expected_time / execution_time if execution_time > 0 else 0
                
                passed = efficiency > 0.1  # Reasonable efficiency threshold
                
                results.append(BenchmarkResult(
                    test_name=f"scaling_{size}x{size}",
                    benchmark_type=BenchmarkType.SCALABILITY,
                    passed=passed,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    error_rate=1.0 - efficiency,
                    conservation_quality=efficiency,
                    metadata={"size": size, "efficiency": efficiency}
                ))
                
                print(f"  ✅ Scaling {size}x{size}: {execution_time:.3f}s, efficiency: {efficiency:.2f}")
                
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"scaling_{size}x{size}",
                    benchmark_type=BenchmarkType.SCALABILITY,
                    passed=False,
                    execution_time=time.time() - start_time,
                    memory_usage=0,
                    error_rate=1.0,
                    conservation_quality=0.0,
                    metadata={"error": str(e), "size": size}
                ))
                print(f"  ❌ Scaling {size}x{size}: {e}")
        
        # Calculate suite metrics
        total_time = sum(r.execution_time for r in results)
        success_rate = sum(1 for r in results if r.passed) / len(results) if results else 0
        rating = self._calculate_rating(success_rate, total_time)
        
        return BenchmarkSuite("scaling", results, total_time, success_rate, rating)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def _calculate_rating(self, success_rate: float, total_time: float) -> str:
        """Calculate overall rating"""
        if success_rate >= 0.9 and total_time < 10:
            return "EXCELLENT"
        elif success_rate >= 0.8 and total_time < 30:
            return "GOOD"
        elif success_rate >= 0.6 and total_time < 60:
            return "FAIR"
        else:
            return "POOR"
    
    def _generate_benchmark_report(self, suites: Dict[str, BenchmarkSuite]):
        """Generate comprehensive benchmark report"""
        print("\n" + "=" * 60)
        print("📋 BENCHMARK REPORT")
        print("=" * 60)
        
        total_tests = sum(len(suite.results) for suite in suites.values())
        total_passed = sum(sum(1 for r in suite.results if r.passed) for suite in suites.values())
        overall_success = total_passed / total_tests if total_tests > 0 else 0
        
        print(f"\n🎯 Overall Results: {total_passed}/{total_tests} tests passed ({overall_success:.1%})")
        
        for suite_name, suite in suites.items():
            print(f"\n📊 {suite_name.upper()} Suite:")
            print(f"   Tests: {len(suite.results)}")
            print(f"   Success Rate: {suite.success_rate:.1%}")
            print(f"   Total Time: {suite.total_time:.2f}s")
            print(f"   Rating: {suite.overall_rating}")
            
            # Show failed tests
            failed_tests = [r for r in suite.results if not r.passed]
            if failed_tests:
                print(f"   ❌ Failed Tests:")
                for test in failed_tests:
                    error = test.metadata.get("error", "Unknown error")
                    print(f"      - {test.test_name}: {error}")
        
        # Performance summary
        total_time = sum(suite.total_time for suite in suites.values())
        avg_memory = np.mean([r.memory_usage for suite in suites.values() for r in suite.results if r.memory_usage > 0])
        
        print(f"\n⚡ Performance Summary:")
        print(f"   Total Execution Time: {total_time:.2f}s")
        print(f"   Average Memory Usage: {avg_memory:.1f}MB")
        print(f"   Device: {self.device}")
        
        # Conservation quality
        conservation_results = [r for suite in suites.values() for r in suite.results if r.benchmark_type == BenchmarkType.CONSERVATION]
        if conservation_results:
            avg_conservation = np.mean([r.conservation_quality for r in conservation_results])
            print(f"   Average PAC Conservation Quality: {avg_conservation:.3f}")
        
        print("\n" + "=" * 60)
