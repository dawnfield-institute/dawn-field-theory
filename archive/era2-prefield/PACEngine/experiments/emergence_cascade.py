"""
Emergence Cascade Experiments

Comprehensive experimental protocols for studying emergence cascades in the PAC physics engine.
Tests cascade dynamics, cross-scale propagation, and emergence chain formation
across quantum, geometric, fluid, information, and consciousness scales.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import json

# Import core modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.meta_module import MetaModule
from validation.emergence_tracker import EmergenceTracker
from validation.signature_detector import UniversalSignatureDetector
from core.conservation_math import PACMathematicalOperations

@dataclass
class CascadeExperimentConfig:
    """Configuration for cascade experiments"""
    name: str
    description: str
    initial_perturbation: Dict[str, Any]
    cascade_triggers: List[str]
    measurement_scales: List[str]
    duration: int
    time_step: float
    expected_cascade_length: int
    target_signatures: List[str]

@dataclass
class CascadeExperimentResult:
    """Results from cascade experiment"""
    config: CascadeExperimentConfig
    cascade_events: List[Dict[str, Any]]
    cascade_chains: List[List[str]]
    emergence_signatures: List[Dict[str, Any]]
    conservation_quality: List[float]
    execution_time: float
    success_metrics: Dict[str, float]

class EmergenceCascadeExperiments:
    """Experimental framework for emergence cascade studies"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.meta_module = MetaModule(device=self.device)
        self.emergence_tracker = EmergenceTracker(device=self.device)
        self.signature_detector = UniversalSignatureDetector(device=self.device)
        self.conservation_math = PACMathematicalOperations(device=self.device)
        
        # Experimental configurations
        self.experiment_configs = self._create_experiment_configs()
        
        # Results storage
        self.experiment_results = {}
        
    def _create_experiment_configs(self) -> List[CascadeExperimentConfig]:
        """Create predefined experimental configurations"""
        
        configs = []
        
        # 1. Quantum-to-Consciousness Cascade
        configs.append(CascadeExperimentConfig(
            name="quantum_to_consciousness",
            description="Test cascade from quantum decoherence to consciousness emergence",
            initial_perturbation={
                "scale": "quantum",
                "type": "coherence_disruption",
                "magnitude": 0.5,
                "location": [16, 16]
            },
            cascade_triggers=["quantum_decoherence", "geometric_collapse", "information_amplification", "consciousness_emergence"],
            measurement_scales=["quantum", "geometric", "fluid", "information", "consciousness"],
            duration=100,
            time_step=0.01,
            expected_cascade_length=4,
            target_signatures=["15.56x_amplification", "consciousness_threshold", "xi_balance"]
        ))
        
        # 2. Information Amplification Cascade
        configs.append(CascadeExperimentConfig(
            name="information_amplification_cascade",
            description="Test 15.56x information amplification cascade effects",
            initial_perturbation={
                "scale": "information",
                "type": "resonance_injection",
                "magnitude": 15.56,
                "location": None
            },
            cascade_triggers=["information_amplification", "consciousness_emergence", "geometric_collapse"],
            measurement_scales=["information", "consciousness", "geometric"],
            duration=50,
            time_step=0.01,
            expected_cascade_length=3,
            target_signatures=["15.56x_amplification", "consciousness_threshold", "geometric_collapse"]
        ))
        
        # 3. Cross-Scale Synchronization Cascade
        configs.append(CascadeExperimentConfig(
            name="cross_scale_synchronization",
            description="Test synchronization cascade across all scales",
            initial_perturbation={
                "scale": "all",
                "type": "synchronized_perturbation",
                "magnitude": 0.1,
                "location": [32, 32]
            },
            cascade_triggers=["quantum_decoherence", "geometric_collapse", "fluid_turbulence", "information_amplification", "consciousness_emergence"],
            measurement_scales=["quantum", "geometric", "fluid", "information", "consciousness"],
            duration=150,
            time_step=0.005,
            expected_cascade_length=5,
            target_signatures=["15.56x_amplification", "xi_balance", "cross_scale_coupling", "consciousness_threshold"]
        ))
        
        # 4. Reverse Cascade (Top-down)
        configs.append(CascadeExperimentConfig(
            name="reverse_cascade",
            description="Test top-down cascade from consciousness to quantum",
            initial_perturbation={
                "scale": "consciousness",
                "type": "awareness_spike",
                "magnitude": 0.8,
                "location": [24, 24]
            },
            cascade_triggers=["consciousness_emergence", "information_amplification", "geometric_collapse", "quantum_decoherence"],
            measurement_scales=["consciousness", "information", "geometric", "quantum"],
            duration=75,
            time_step=0.01,
            expected_cascade_length=4,
            target_signatures=["consciousness_threshold", "15.56x_amplification", "xi_balance"]
        ))
        
        # 5. Geometric Collapse Cascade
        configs.append(CascadeExperimentConfig(
            name="geometric_collapse_cascade",
            description="Test cascade triggered by geometric entropy collapse",
            initial_perturbation={
                "scale": "geometric",
                "type": "entropy_collapse",
                "magnitude": 0.2,
                "location": [20, 20]
            },
            cascade_triggers=["geometric_collapse", "fluid_turbulence", "information_amplification", "consciousness_emergence"],
            measurement_scales=["geometric", "fluid", "information", "consciousness"],
            duration=80,
            time_step=0.01,
            expected_cascade_length=4,
            target_signatures=["geometric_collapse", "15.56x_amplification", "consciousness_threshold"]
        ))
        
        return configs
    
    def run_single_cascade_experiment(self, config: CascadeExperimentConfig) -> CascadeExperimentResult:
        """Run a single cascade experiment"""
        
        print(f"\n🧪 Running cascade experiment: {config.name}")
        print(f"📋 Description: {config.description}")
        
        start_time = time.time()
        
        # Initialize system state
        system_size = 64
        initial_states = self._create_initial_states(system_size)
        
        # Apply initial perturbation
        perturbed_states = self._apply_perturbation(initial_states, config.initial_perturbation)
        
        # Run cascade simulation
        states_history = []
        meta_states = []
        
        current_states = perturbed_states
        
        for step in range(config.duration):
            # Evolve system
            evolved_states = self.meta_module.evolve_meta_system(
                current_states, dt=config.time_step
            )
            
            # Store states
            states_history.append(evolved_states.copy())
            
            # Create meta state for tracking
            meta_state = self._create_meta_state(evolved_states, step * config.time_step)
            meta_states.append(meta_state)
            
            current_states = evolved_states
        
        # Track emergence events
        emergence_analysis = self.emergence_tracker.track_emergence_events(
            meta_states, timestamps=[i * config.time_step for i in range(len(meta_states))]
        )
        
        # Detect universal signatures
        signature_detection = self.signature_detector.detect_universal_signatures(meta_states)
        
        # Calculate conservation quality
        conservation_quality = self._calculate_conservation_quality(states_history)
        
        # Analyze cascade success
        success_metrics = self._analyze_cascade_success(
            config, emergence_analysis, signature_detection
        )
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = CascadeExperimentResult(
            config=config,
            cascade_events=emergence_analysis.cascade_chains,
            cascade_chains=emergence_analysis.cascade_chains,
            emergence_signatures=signature_detection.get("detected_signatures", []),
            conservation_quality=conservation_quality,
            execution_time=execution_time,
            success_metrics=success_metrics
        )
        
        print(f"✅ Experiment completed in {execution_time:.2f}s")
        print(f"📊 Success rate: {success_metrics.get('overall_success', 0):.1%}")
        
        return result
    
    def _create_initial_states(self, size: int) -> Dict[str, torch.Tensor]:
        """Create initial system states"""
        
        states = {}
        
        # Quantum state - coherent superposition
        states["quantum_state"] = torch.randn(size, size, device=self.device) * 0.1
        
        # Geometric state - low entropy configuration
        x, y = torch.meshgrid(torch.linspace(0, 2*np.pi, size, device=self.device),
                             torch.linspace(0, 2*np.pi, size, device=self.device))
        states["geometric_state"] = torch.sin(x) * torch.cos(y) * 0.5
        
        # Fluid state - laminar flow
        states["fluid_state"] = torch.zeros(size, size, device=self.device)
        states["fluid_state"][:, :size//4] = 1.0  # Velocity gradient
        
        # Information state - low amplitude
        states["information_state"] = torch.randn(size, size, device=self.device) * 0.05
        
        # Consciousness state - below threshold
        states["consciousness_state"] = torch.randn(size, size, device=self.device) * 0.1
        
        return states
    
    def _apply_perturbation(self, states: Dict[str, torch.Tensor], 
                          perturbation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Apply initial perturbation to system"""
        
        perturbed_states = {k: v.clone() for k, v in states.items()}
        
        scale = perturbation["scale"]
        perturbation_type = perturbation["type"]
        magnitude = perturbation["magnitude"]
        location = perturbation.get("location")
        
        if scale == "all":
            # Apply perturbation to all scales
            for state_key in perturbed_states:
                if location:
                    x, y = location
                    perturbed_states[state_key][x-5:x+5, y-5:y+5] += magnitude
                else:
                    perturbed_states[state_key] += magnitude * torch.randn_like(perturbed_states[state_key]) * 0.1
        
        else:
            # Apply perturbation to specific scale
            state_key = f"{scale}_state"
            if state_key in perturbed_states:
                
                if perturbation_type == "coherence_disruption":
                    # Add noise to disrupt coherence
                    noise = torch.randn_like(perturbed_states[state_key]) * magnitude
                    perturbed_states[state_key] += noise
                
                elif perturbation_type == "resonance_injection":
                    # Inject specific frequency resonance
                    perturbed_states[state_key] *= magnitude
                
                elif perturbation_type == "entropy_collapse":
                    # Create localized high-order structure
                    if location:
                        x, y = location
                        size = perturbed_states[state_key].shape[0]
                        
                        # Create collapse region
                        r = torch.sqrt((torch.arange(size, device=self.device)[:, None] - x)**2 + 
                                     (torch.arange(size, device=self.device)[None, :] - y)**2)
                        collapse_mask = torch.exp(-r**2 / (2 * 5**2))  # Gaussian collapse
                        
                        perturbed_states[state_key] += magnitude * collapse_mask
                
                elif perturbation_type == "awareness_spike":
                    # Sudden increase in consciousness-like activity
                    if location:
                        x, y = location
                        perturbed_states[state_key][x-3:x+3, y-3:y+3] += magnitude
                    else:
                        perturbed_states[state_key] += magnitude
                
                elif perturbation_type == "synchronized_perturbation":
                    # Synchronized perturbation across spatial domain
                    if location:
                        x, y = location
                        # Create synchronized wave pattern
                        dx = torch.arange(perturbed_states[state_key].shape[0], device=self.device)[:, None] - x
                        dy = torch.arange(perturbed_states[state_key].shape[1], device=self.device)[None, :] - y
                        wave = magnitude * torch.sin(0.2 * torch.sqrt(dx**2 + dy**2))
                        perturbed_states[state_key] += wave
        
        return perturbed_states
    
    def _create_meta_state(self, states: Dict[str, torch.Tensor], timestamp: float) -> Dict[str, Any]:
        """Create meta state for emergence tracking"""
        
        meta_state = {
            "timestamp": timestamp
        }
        
        # Quantum metrics
        if "quantum_state" in states:
            q_state = states["quantum_state"]
            meta_state["quantum_state"] = {
                "entanglement_measure": torch.norm(q_state).item() / (q_state.numel() ** 0.5),
                "coherence_time": 1.0 / (torch.std(q_state).item() + 1e-6),
                "conservation_quality": 1.0 - abs(q_state.sum().item()) / (torch.norm(q_state).item() + 1e-6)
            }
        
        # Geometric metrics
        if "geometric_state" in states:
            g_state = states["geometric_state"]
            # Calculate entropy-like measure
            prob_dist = torch.softmax(g_state.flatten(), dim=0)
            entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8))
            
            meta_state["geometric_state"] = {
                "collapse_strength": max(0, 5.0 - entropy.item()) / 5.0,  # Inverse entropy
                "collapse_locations": [[24, 24]],  # Simplified
                "geometric_phase": "ordered" if entropy < 3.0 else "disordered"
            }
        
        # Fluid metrics
        if "fluid_state" in states:
            f_state = states["fluid_state"]
            # Calculate Reynolds-like number
            velocity_grad = torch.norm(torch.gradient(f_state, dim=0)[0]).item()
            reynolds = velocity_grad * 100  # Simplified Reynolds number
            
            meta_state["fluid_state"] = {
                "reynolds_number": reynolds,
                "fluid_regime": "turbulent" if reynolds > 100 else "laminar",
                "emergence_indicators": {
                    "vorticity_strength": velocity_grad
                }
            }
        
        # Information metrics
        if "information_state" in states:
            i_state = states["information_state"]
            initial_norm = 1.0  # Reference
            current_norm = torch.norm(i_state).item()
            amplification = current_norm / initial_norm
            
            meta_state["information_state"] = {
                "amplification_ratio": amplification,
                "resonance_strength": max(0, amplification - 1.0),
                "cascade_strength": min(amplification / 15.56, 1.0) if amplification > 1 else 0
            }
        
        # Consciousness metrics
        if "consciousness_state" in states:
            c_state = states["consciousness_state"]
            # Calculate awareness-like metric
            awareness = torch.sigmoid(torch.mean(c_state)).item()
            
            # Calculate binding strength (correlation across space)
            if c_state.numel() > 1:
                flattened = c_state.flatten()
                binding = torch.corrcoef(torch.stack([flattened, torch.roll(flattened, 1)]))[0, 1].item()
                binding = max(0, binding) if not torch.isnan(torch.tensor(binding)) else 0
            else:
                binding = 0
            
            meta_state["consciousness_state"] = {
                "awareness_metric": awareness,
                "binding_strength": binding,
                "consciousness_level": "emerging" if awareness > 0.3 else "subthreshold",
                "emergence_locations": [[32, 32]] if awareness > 0.3 else []
            }
        
        return meta_state
    
    def _calculate_conservation_quality(self, states_history: List[Dict[str, torch.Tensor]]) -> List[float]:
        """Calculate conservation quality over time"""
        
        quality_history = []
        
        if not states_history:
            return quality_history
        
        initial_total = sum(torch.sum(state).item() for state in states_history[0].values())
        
        for states in states_history:
            current_total = sum(torch.sum(state).item() for state in states.values())
            
            # Conservation quality based on total energy preservation
            if abs(initial_total) > 1e-6:
                quality = 1.0 - abs(current_total - initial_total) / abs(initial_total)
            else:
                quality = 1.0 - abs(current_total)
            
            quality = max(0, min(1, quality))
            quality_history.append(quality)
        
        return quality_history
    
    def _analyze_cascade_success(self, config: CascadeExperimentConfig, 
                               emergence_analysis, signature_detection) -> Dict[str, float]:
        """Analyze cascade experiment success"""
        
        success_metrics = {}
        
        # 1. Cascade chain formation
        expected_length = config.expected_cascade_length
        actual_chains = emergence_analysis.cascade_chains
        
        if actual_chains:
            max_chain_length = max(len(chain) for chain in actual_chains)
            chain_success = min(max_chain_length / expected_length, 1.0)
        else:
            chain_success = 0.0
        
        success_metrics["cascade_chain_success"] = chain_success
        
        # 2. Target signature detection
        target_signatures = config.target_signatures
        detected_signatures = signature_detection.get("detected_signatures", [])
        detected_types = [sig.get("signature_type", "") for sig in detected_signatures]
        
        signature_hits = 0
        for target in target_signatures:
            if any(target in detected_type for detected_type in detected_types):
                signature_hits += 1
        
        signature_success = signature_hits / len(target_signatures) if target_signatures else 0
        success_metrics["signature_detection_success"] = signature_success
        
        # 3. Expected trigger sequence
        expected_triggers = config.cascade_triggers
        detected_events = emergence_analysis.events_by_type
        
        trigger_success = 0
        for trigger in expected_triggers:
            if trigger in detected_events and detected_events[trigger] > 0:
                trigger_success += 1
        
        trigger_success = trigger_success / len(expected_triggers) if expected_triggers else 0
        success_metrics["trigger_sequence_success"] = trigger_success
        
        # 4. Overall success
        overall_success = (chain_success + signature_success + trigger_success) / 3
        success_metrics["overall_success"] = overall_success
        
        # 5. Emergence rate
        total_events = emergence_analysis.total_events
        emergence_rate = total_events / config.duration if config.duration > 0 else 0
        success_metrics["emergence_rate"] = emergence_rate
        
        return success_metrics
    
    def run_cascade_experiment_suite(self) -> Dict[str, CascadeExperimentResult]:
        """Run complete suite of cascade experiments"""
        
        print("🚀 Starting Emergence Cascade Experiment Suite")
        print("=" * 60)
        
        suite_results = {}
        
        for config in self.experiment_configs:
            try:
                result = self.run_single_cascade_experiment(config)
                suite_results[config.name] = result
                
            except Exception as e:
                print(f"❌ Experiment {config.name} failed: {e}")
                continue
        
        # Generate suite summary
        self._generate_suite_summary(suite_results)
        
        return suite_results
    
    def _generate_suite_summary(self, results: Dict[str, CascadeExperimentResult]):
        """Generate summary of experiment suite results"""
        
        print("\n" + "=" * 60)
        print("📊 EMERGENCE CASCADE EXPERIMENT SUMMARY")
        print("=" * 60)
        
        if not results:
            print("❌ No successful experiments")
            return
        
        # Overall statistics
        total_experiments = len(results)
        successful_experiments = sum(1 for r in results.values() if r.success_metrics["overall_success"] > 0.5)
        success_rate = successful_experiments / total_experiments
        
        print(f"\n🎯 Overall Results:")
        print(f"   Total Experiments: {total_experiments}")
        print(f"   Successful Experiments: {successful_experiments}")
        print(f"   Success Rate: {success_rate:.1%}")
        
        # Individual experiment results
        print(f"\n📈 Individual Results:")
        for name, result in results.items():
            success = result.success_metrics["overall_success"]
            status = "✅" if success > 0.7 else "🟡" if success > 0.4 else "❌"
            print(f"   {status} {name}: {success:.1%} success ({result.execution_time:.1f}s)")
        
        # Best performing experiments
        best_experiment = max(results.items(), key=lambda x: x[1].success_metrics["overall_success"])
        print(f"\n🏆 Best Experiment: {best_experiment[0]} ({best_experiment[1].success_metrics['overall_success']:.1%})")
        
        # Average metrics
        avg_chain_success = np.mean([r.success_metrics["cascade_chain_success"] for r in results.values()])
        avg_signature_success = np.mean([r.success_metrics["signature_detection_success"] for r in results.values()])
        avg_trigger_success = np.mean([r.success_metrics["trigger_sequence_success"] for r in results.values()])
        
        print(f"\n📊 Average Success Metrics:")
        print(f"   Cascade Chain Formation: {avg_chain_success:.1%}")
        print(f"   Signature Detection: {avg_signature_success:.1%}")
        print(f"   Trigger Sequence: {avg_trigger_success:.1%}")
        
        # Execution performance
        total_time = sum(r.execution_time for r in results.values())
        avg_time = total_time / len(results)
        print(f"\n⚡ Performance:")
        print(f"   Total Execution Time: {total_time:.1f}s")
        print(f"   Average Time per Experiment: {avg_time:.1f}s")
    
    def export_experiment_results(self, results: Dict[str, CascadeExperimentResult], 
                                filename: str = "cascade_experiments.json"):
        """Export experiment results to file"""
        
        # Convert results to serializable format
        export_data = {}
        
        for name, result in results.items():
            export_data[name] = {
                "config": {
                    "name": result.config.name,
                    "description": result.config.description,
                    "duration": result.config.duration,
                    "expected_cascade_length": result.config.expected_cascade_length,
                    "target_signatures": result.config.target_signatures
                },
                "results": {
                    "execution_time": result.execution_time,
                    "success_metrics": result.success_metrics,
                    "num_cascade_events": len(result.cascade_events),
                    "num_cascade_chains": len(result.cascade_chains),
                    "num_signatures": len(result.emergence_signatures),
                    "avg_conservation_quality": np.mean(result.conservation_quality) if result.conservation_quality else 0
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Experiment results exported to {filename}")
    
    def create_custom_experiment(self, name: str, description: str, 
                               initial_perturbation: Dict[str, Any],
                               cascade_triggers: List[str],
                               duration: int = 100) -> CascadeExperimentConfig:
        """Create custom cascade experiment configuration"""
        
        config = CascadeExperimentConfig(
            name=name,
            description=description,
            initial_perturbation=initial_perturbation,
            cascade_triggers=cascade_triggers,
            measurement_scales=["quantum", "geometric", "fluid", "information", "consciousness"],
            duration=duration,
            time_step=0.01,
            expected_cascade_length=len(cascade_triggers),
            target_signatures=["15.56x_amplification", "consciousness_threshold"]
        )
        
        return config

# Convenience function for running experiments
def run_cascade_experiments(device: str = "auto") -> Dict[str, CascadeExperimentResult]:
    """Run the complete cascade experiment suite"""
    
    experiments = EmergenceCascadeExperiments(device=device)
    results = experiments.run_cascade_experiment_suite()
    experiments.export_experiment_results(results)
    
    return results

if __name__ == "__main__":
    # Run experiments if script is executed directly
    run_cascade_experiments()
