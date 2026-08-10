"""
Perturbation Suite

Comprehensive perturbation testing framework for the PAC physics engine.
Tests system response to various types of perturbations across different scales,
measuring stability, conservation quality, and emergence characteristics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import json

# Import core modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.meta_module import MetaModule
from validation.cross_scale_validator import CrossScaleValidator
from core.conservation_math import PACMathematicalOperations

class PerturbationType(Enum):
    """Types of perturbations to test"""
    NOISE_INJECTION = "noise_injection"
    LOCALIZED_SPIKE = "localized_spike"
    FREQUENCY_SWEEP = "frequency_sweep"
    CONSERVATION_VIOLATION = "conservation_violation"
    RESONANCE_DRIVE = "resonance_drive"
    COHERENCE_DISRUPTION = "coherence_disruption"
    SCALE_COUPLING = "scale_coupling"
    CHAOS_INJECTION = "chaos_injection"
    PHASE_RESET = "phase_reset"
    BOUNDARY_STRESS = "boundary_stress"

@dataclass
class PerturbationConfig:
    """Configuration for a perturbation test"""
    name: str
    perturbation_type: PerturbationType
    target_scale: str
    magnitude: float
    duration: int
    spatial_pattern: str  # "random", "localized", "wave", "gradient"
    temporal_pattern: str  # "impulse", "step", "ramp", "oscillating"
    expected_recovery_time: float
    conservation_tolerance: float

@dataclass
class PerturbationResult:
    """Results from a perturbation test"""
    config: PerturbationConfig
    initial_state_norm: float
    peak_deviation: float
    recovery_time: float
    final_conservation_error: float
    stability_metric: float
    emergence_events: List[Dict[str, Any]]
    response_characteristics: Dict[str, float]
    success: bool

class PerturbationSuite:
    """Comprehensive perturbation testing framework"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.meta_module = MetaModule(device=self.device)
        self.validator = CrossScaleValidator(device=self.device)
        self.conservation_math = PACMathematicalOperations(device=self.device)
        
        # Test configurations
        self.perturbation_configs = self._create_perturbation_configs()
        
        # System parameters
        self.system_size = 64
        self.base_dt = 0.01
        self.measurement_interval = 5
        
    def _create_perturbation_configs(self) -> List[PerturbationConfig]:
        """Create comprehensive set of perturbation test configurations"""
        
        configs = []
        scales = ["quantum", "geometric", "fluid", "information", "consciousness"]
        
        # 1. Noise injection tests
        for scale in scales:
            configs.append(PerturbationConfig(
                name=f"noise_injection_{scale}",
                perturbation_type=PerturbationType.NOISE_INJECTION,
                target_scale=scale,
                magnitude=0.1,
                duration=100,
                spatial_pattern="random",
                temporal_pattern="impulse",
                expected_recovery_time=50.0,
                conservation_tolerance=1e-3
            ))
        
        # 2. Localized spike tests
        for scale in scales:
            configs.append(PerturbationConfig(
                name=f"localized_spike_{scale}",
                perturbation_type=PerturbationType.LOCALIZED_SPIKE,
                target_scale=scale,
                magnitude=1.0,
                duration=150,
                spatial_pattern="localized",
                temporal_pattern="impulse",
                expected_recovery_time=75.0,
                conservation_tolerance=1e-3
            ))
        
        # 3. Frequency sweep tests
        for scale in scales:
            configs.append(PerturbationConfig(
                name=f"frequency_sweep_{scale}",
                perturbation_type=PerturbationType.FREQUENCY_SWEEP,
                target_scale=scale,
                magnitude=0.5,
                duration=200,
                spatial_pattern="wave",
                temporal_pattern="oscillating",
                expected_recovery_time=100.0,
                conservation_tolerance=1e-3
            ))
        
        # 4. Conservation violation tests
        for scale in scales:
            configs.append(PerturbationConfig(
                name=f"conservation_violation_{scale}",
                perturbation_type=PerturbationType.CONSERVATION_VIOLATION,
                target_scale=scale,
                magnitude=0.2,
                duration=100,
                spatial_pattern="gradient",
                temporal_pattern="step",
                expected_recovery_time=80.0,
                conservation_tolerance=1e-2  # More tolerant for this test
            ))
        
        # 5. Resonance drive tests (specific to information scale)
        configs.append(PerturbationConfig(
            name="resonance_drive_15_56x",
            perturbation_type=PerturbationType.RESONANCE_DRIVE,
            target_scale="information",
            magnitude=15.56,
            duration=150,
            spatial_pattern="random",
            temporal_pattern="step",
            expected_recovery_time=75.0,
            conservation_tolerance=0.1  # Allow amplification
        ))
        
        # 6. Cross-scale coupling tests
        configs.append(PerturbationConfig(
            name="cross_scale_coupling",
            perturbation_type=PerturbationType.SCALE_COUPLING,
            target_scale="all",
            magnitude=0.3,
            duration=200,
            spatial_pattern="wave",
            temporal_pattern="oscillating",
            expected_recovery_time=120.0,
            conservation_tolerance=1e-3
        ))
        
        # 7. Chaos injection test
        configs.append(PerturbationConfig(
            name="chaos_injection",
            perturbation_type=PerturbationType.CHAOS_INJECTION,
            target_scale="fluid",
            magnitude=2.0,
            duration=250,
            spatial_pattern="random",
            temporal_pattern="step",
            expected_recovery_time=150.0,
            conservation_tolerance=1e-2
        ))
        
        # 8. Coherence disruption test
        configs.append(PerturbationConfig(
            name="coherence_disruption",
            perturbation_type=PerturbationType.COHERENCE_DISRUPTION,
            target_scale="quantum",
            magnitude=0.8,
            duration=120,
            spatial_pattern="random",
            temporal_pattern="ramp",
            expected_recovery_time=90.0,
            conservation_tolerance=1e-3
        ))
        
        return configs
    
    def run_single_perturbation_test(self, config: PerturbationConfig) -> PerturbationResult:
        """Run a single perturbation test"""
        
        print(f"\n🧪 Running perturbation test: {config.name}")
        print(f"📋 Type: {config.perturbation_type.value}")
        print(f"🎯 Target: {config.target_scale}")
        
        # Initialize system
        initial_states = self._create_stable_initial_states()
        initial_norm = self._calculate_system_norm(initial_states)
        
        # Apply perturbation
        perturbed_states = self._apply_perturbation(initial_states, config)
        
        # Track system evolution
        states_history = []
        conservation_history = []
        norm_history = []
        emergence_events = []
        
        current_states = perturbed_states
        peak_deviation = 0.0
        
        for step in range(config.duration):
            # Evolve system
            evolved_states = self.meta_module.evolve_meta_system(
                current_states, dt=self.base_dt
            )
            
            # Measure system properties every few steps
            if step % self.measurement_interval == 0:
                states_history.append(evolved_states.copy())
                
                # Calculate conservation quality
                conservation_error = self._calculate_conservation_error(initial_states, evolved_states)
                conservation_history.append(1.0 - conservation_error)
                
                # Calculate norm deviation
                current_norm = self._calculate_system_norm(evolved_states)
                norm_deviation = abs(current_norm - initial_norm) / initial_norm
                norm_history.append(norm_deviation)
                peak_deviation = max(peak_deviation, norm_deviation)
                
                # Check for emergence events
                emergence_event = self._detect_emergence_events(evolved_states, step)
                if emergence_event:
                    emergence_events.append(emergence_event)
            
            current_states = evolved_states
        
        # Analyze results
        recovery_time = self._calculate_recovery_time(norm_history, config.expected_recovery_time)
        final_conservation_error = 1.0 - conservation_history[-1] if conservation_history else 1.0
        stability_metric = self._calculate_stability_metric(norm_history)
        response_characteristics = self._analyze_response_characteristics(norm_history, conservation_history)
        
        # Determine success
        success = (
            final_conservation_error <= config.conservation_tolerance and
            recovery_time <= config.expected_recovery_time * 1.5 and  # 50% tolerance
            stability_metric > 0.5
        )
        
        result = PerturbationResult(
            config=config,
            initial_state_norm=initial_norm,
            peak_deviation=peak_deviation,
            recovery_time=recovery_time,
            final_conservation_error=final_conservation_error,
            stability_metric=stability_metric,
            emergence_events=emergence_events,
            response_characteristics=response_characteristics,
            success=success
        )
        
        print(f"✅ Test completed - Success: {success}")
        print(f"📊 Peak deviation: {peak_deviation:.3f}, Recovery: {recovery_time:.1f}s")
        
        return result
    
    def _create_stable_initial_states(self) -> Dict[str, torch.Tensor]:
        """Create stable initial system states"""
        
        states = {}
        size = self.system_size
        
        # Quantum state - coherent ground state
        states["quantum_state"] = torch.zeros(size, size, device=self.device)
        states["quantum_state"][size//2-2:size//2+2, size//2-2:size//2+2] = 1.0
        
        # Geometric state - ordered configuration
        x, y = torch.meshgrid(torch.linspace(0, 4*np.pi, size, device=self.device),
                             torch.linspace(0, 4*np.pi, size, device=self.device))
        states["geometric_state"] = 0.5 * (torch.sin(x) + torch.cos(y))
        
        # Fluid state - equilibrium
        states["fluid_state"] = torch.zeros(size, size, device=self.device)
        
        # Information state - low entropy
        states["information_state"] = torch.ones(size, size, device=self.device) * 0.1
        
        # Consciousness state - subcritical
        states["consciousness_state"] = torch.randn(size, size, device=self.device) * 0.05
        
        return states
    
    def _apply_perturbation(self, states: Dict[str, torch.Tensor], 
                          config: PerturbationConfig) -> Dict[str, torch.Tensor]:
        """Apply perturbation according to configuration"""
        
        perturbed_states = {k: v.clone() for k, v in states.items()}
        size = self.system_size
        
        # Generate spatial pattern
        if config.spatial_pattern == "random":
            spatial_mask = torch.randn(size, size, device=self.device)
        elif config.spatial_pattern == "localized":
            spatial_mask = torch.zeros(size, size, device=self.device)
            center = size // 2
            radius = size // 8
            y, x = torch.meshgrid(torch.arange(size, device=self.device), 
                                 torch.arange(size, device=self.device))
            dist = torch.sqrt((x - center)**2 + (y - center)**2)
            spatial_mask[dist <= radius] = 1.0
        elif config.spatial_pattern == "wave":
            x, y = torch.meshgrid(torch.linspace(0, 4*np.pi, size, device=self.device),
                                 torch.linspace(0, 4*np.pi, size, device=self.device))
            spatial_mask = torch.sin(x) * torch.cos(y)
        elif config.spatial_pattern == "gradient":
            spatial_mask = torch.linspace(-1, 1, size, device=self.device).unsqueeze(0).repeat(size, 1)
        else:
            spatial_mask = torch.ones(size, size, device=self.device)
        
        # Apply perturbation based on type
        if config.target_scale == "all":
            target_keys = list(perturbed_states.keys())
        else:
            target_keys = [f"{config.target_scale}_state"]
        
        for key in target_keys:
            if key in perturbed_states:
                
                if config.perturbation_type == PerturbationType.NOISE_INJECTION:
                    noise = torch.randn_like(perturbed_states[key]) * config.magnitude
                    perturbed_states[key] += noise * spatial_mask
                
                elif config.perturbation_type == PerturbationType.LOCALIZED_SPIKE:
                    perturbed_states[key] += config.magnitude * spatial_mask
                
                elif config.perturbation_type == PerturbationType.FREQUENCY_SWEEP:
                    # Apply sinusoidal perturbation
                    perturbed_states[key] += config.magnitude * spatial_mask * 0.5
                
                elif config.perturbation_type == PerturbationType.CONSERVATION_VIOLATION:
                    # Add energy without compensation
                    perturbed_states[key] += config.magnitude * spatial_mask
                
                elif config.perturbation_type == PerturbationType.RESONANCE_DRIVE:
                    # Multiply by resonance factor
                    perturbed_states[key] *= config.magnitude
                
                elif config.perturbation_type == PerturbationType.COHERENCE_DISRUPTION:
                    # Add phase noise
                    phase_noise = torch.randn_like(perturbed_states[key]) * config.magnitude
                    perturbed_states[key] = perturbed_states[key] * torch.exp(1j * phase_noise).real
                
                elif config.perturbation_type == PerturbationType.SCALE_COUPLING:
                    # Cross-scale coupling perturbation
                    coupling_strength = config.magnitude
                    for other_key in perturbed_states:
                        if other_key != key:
                            coupling = coupling_strength * torch.mean(perturbed_states[other_key]) * spatial_mask
                            perturbed_states[key] += coupling
                
                elif config.perturbation_type == PerturbationType.CHAOS_INJECTION:
                    # Chaotic perturbation
                    chaos = self._generate_chaotic_field(size) * config.magnitude
                    perturbed_states[key] += chaos * spatial_mask
                
                elif config.perturbation_type == PerturbationType.PHASE_RESET:
                    # Reset phase information
                    perturbed_states[key] = torch.abs(perturbed_states[key]) * torch.sign(spatial_mask)
                
                elif config.perturbation_type == PerturbationType.BOUNDARY_STRESS:
                    # Apply stress at boundaries
                    boundary_mask = torch.zeros_like(spatial_mask)
                    boundary_mask[0, :] = 1.0
                    boundary_mask[-1, :] = 1.0
                    boundary_mask[:, 0] = 1.0
                    boundary_mask[:, -1] = 1.0
                    perturbed_states[key] += config.magnitude * boundary_mask
        
        return perturbed_states
    
    def _generate_chaotic_field(self, size: int) -> torch.Tensor:
        """Generate chaotic field using logistic map"""
        
        field = torch.zeros(size, size, device=self.device)
        
        # Use logistic map for chaos
        x = 0.5
        r = 3.8  # Chaotic regime
        
        for i in range(size):
            for j in range(size):
                x = r * x * (1 - x)
                field[i, j] = x
        
        return (field - 0.5) * 2  # Center around 0
    
    def _calculate_system_norm(self, states: Dict[str, torch.Tensor]) -> float:
        """Calculate total system norm"""
        total_norm = 0.0
        for state in states.values():
            total_norm += torch.norm(state).item()**2
        return np.sqrt(total_norm)
    
    def _calculate_conservation_error(self, initial_states: Dict[str, torch.Tensor], 
                                    current_states: Dict[str, torch.Tensor]) -> float:
        """Calculate conservation error"""
        
        initial_total = sum(torch.sum(state).item() for state in initial_states.values())
        current_total = sum(torch.sum(state).item() for state in current_states.values())
        
        if abs(initial_total) > 1e-6:
            error = abs(current_total - initial_total) / abs(initial_total)
        else:
            error = abs(current_total)
        
        return error
    
    def _detect_emergence_events(self, states: Dict[str, torch.Tensor], step: int) -> Optional[Dict[str, Any]]:
        """Detect emergence events in current state"""
        
        # Simple emergence detection
        for scale_name, state in states.items():
            if torch.is_tensor(state):
                magnitude = torch.norm(state).item()
                
                # Check for unusual activity
                if magnitude > 10.0:  # Threshold for significant activity
                    return {
                        "step": step,
                        "scale": scale_name,
                        "type": "high_activity",
                        "magnitude": magnitude
                    }
        
        return None
    
    def _calculate_recovery_time(self, norm_history: List[float], 
                               expected_recovery: float) -> float:
        """Calculate time to recover to baseline"""
        
        if len(norm_history) < 2:
            return float('inf')
        
        baseline = norm_history[0]
        threshold = 0.1  # 10% of baseline
        
        # Find when system returns to within threshold of baseline
        for i, norm_dev in enumerate(norm_history):
            if norm_dev <= threshold:
                return i * self.measurement_interval * self.base_dt
        
        return len(norm_history) * self.measurement_interval * self.base_dt
    
    def _calculate_stability_metric(self, norm_history: List[float]) -> float:
        """Calculate stability metric based on norm history"""
        
        if len(norm_history) < 2:
            return 0.0
        
        # Measure how quickly fluctuations decay
        variance = np.var(norm_history)
        mean_deviation = np.mean(norm_history)
        
        # Stability is inverse of normalized variance
        stability = 1.0 / (1.0 + variance / (mean_deviation + 1e-6))
        
        return stability
    
    def _analyze_response_characteristics(self, norm_history: List[float], 
                                        conservation_history: List[float]) -> Dict[str, float]:
        """Analyze response characteristics"""
        
        characteristics = {}
        
        if norm_history:
            characteristics["max_deviation"] = max(norm_history)
            characteristics["final_deviation"] = norm_history[-1]
            characteristics["oscillation_amplitude"] = np.std(norm_history)
            
            # Calculate damping ratio
            if len(norm_history) > 10:
                early_std = np.std(norm_history[:len(norm_history)//3])
                late_std = np.std(norm_history[-len(norm_history)//3:])
                characteristics["damping_ratio"] = late_std / (early_std + 1e-6)
            else:
                characteristics["damping_ratio"] = 1.0
        
        if conservation_history:
            characteristics["min_conservation"] = min(conservation_history)
            characteristics["final_conservation"] = conservation_history[-1]
            characteristics["conservation_stability"] = 1.0 - np.std(conservation_history)
        
        return characteristics
    
    def run_perturbation_suite(self) -> Dict[str, PerturbationResult]:
        """Run complete perturbation test suite"""
        
        print("🚀 Starting Perturbation Test Suite")
        print("=" * 60)
        
        suite_results = {}
        
        for config in self.perturbation_configs:
            try:
                result = self.run_single_perturbation_test(config)
                suite_results[config.name] = result
                
            except Exception as e:
                print(f"❌ Test {config.name} failed: {e}")
                continue
        
        # Generate suite summary
        self._generate_suite_summary(suite_results)
        
        return suite_results
    
    def _generate_suite_summary(self, results: Dict[str, PerturbationResult]):
        """Generate summary of perturbation test suite"""
        
        print("\n" + "=" * 60)
        print("📊 PERTURBATION TEST SUITE SUMMARY")
        print("=" * 60)
        
        if not results:
            print("❌ No successful tests")
            return
        
        # Overall statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r.success)
        success_rate = successful_tests / total_tests
        
        print(f"\n🎯 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful Tests: {successful_tests}")
        print(f"   Success Rate: {success_rate:.1%}")
        
        # Results by perturbation type
        type_results = {}
        for result in results.values():
            ptype = result.config.perturbation_type.value
            if ptype not in type_results:
                type_results[ptype] = {"total": 0, "success": 0}
            type_results[ptype]["total"] += 1
            if result.success:
                type_results[ptype]["success"] += 1
        
        print(f"\n📈 Results by Perturbation Type:")
        for ptype, stats in type_results.items():
            success_rate = stats["success"] / stats["total"]
            status = "✅" if success_rate > 0.8 else "🟡" if success_rate > 0.5 else "❌"
            print(f"   {status} {ptype}: {stats['success']}/{stats['total']} ({success_rate:.1%})")
        
        # Performance metrics
        avg_recovery_time = np.mean([r.recovery_time for r in results.values()])
        avg_stability = np.mean([r.stability_metric for r in results.values()])
        avg_conservation_error = np.mean([r.final_conservation_error for r in results.values()])
        
        print(f"\n📊 Average Performance Metrics:")
        print(f"   Recovery Time: {avg_recovery_time:.1f}s")
        print(f"   Stability Metric: {avg_stability:.3f}")
        print(f"   Conservation Error: {avg_conservation_error:.2e}")
        
        # Most challenging tests
        challenging_tests = sorted(results.items(), key=lambda x: x[1].peak_deviation, reverse=True)[:3]
        print(f"\n🏔️  Most Challenging Tests:")
        for name, result in challenging_tests:
            print(f"   {name}: {result.peak_deviation:.3f} peak deviation")
    
    def export_suite_results(self, results: Dict[str, PerturbationResult], 
                           filename: str = "perturbation_suite_results.json"):
        """Export suite results to file"""
        
        export_data = {}
        
        for name, result in results.items():
            export_data[name] = {
                "config": {
                    "perturbation_type": result.config.perturbation_type.value,
                    "target_scale": result.config.target_scale,
                    "magnitude": result.config.magnitude,
                    "duration": result.config.duration
                },
                "results": {
                    "success": result.success,
                    "peak_deviation": result.peak_deviation,
                    "recovery_time": result.recovery_time,
                    "final_conservation_error": result.final_conservation_error,
                    "stability_metric": result.stability_metric,
                    "num_emergence_events": len(result.emergence_events),
                    "response_characteristics": result.response_characteristics
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Suite results exported to {filename}")
    
    def create_custom_perturbation_test(self, name: str, perturbation_type: PerturbationType,
                                      target_scale: str, magnitude: float,
                                      duration: int = 100) -> PerturbationConfig:
        """Create custom perturbation test configuration"""
        
        config = PerturbationConfig(
            name=name,
            perturbation_type=perturbation_type,
            target_scale=target_scale,
            magnitude=magnitude,
            duration=duration,
            spatial_pattern="random",
            temporal_pattern="impulse",
            expected_recovery_time=duration * 0.5,
            conservation_tolerance=1e-3
        )
        
        return config

# Convenience function for running perturbation tests
def run_perturbation_tests(device: str = "auto") -> Dict[str, PerturbationResult]:
    """Run the complete perturbation test suite"""
    
    suite = PerturbationSuite(device=device)
    results = suite.run_perturbation_suite()
    suite.export_suite_results(results)
    
    return results

if __name__ == "__main__":
    # Run perturbation tests if script is executed directly
    run_perturbation_tests()
