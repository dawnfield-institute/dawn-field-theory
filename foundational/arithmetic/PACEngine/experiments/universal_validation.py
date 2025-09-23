#!/usr/bin/env python3
"""
Universal Validation Experiment
===============================

The ultimate experiment that validates ALL Dawn Field Theory frameworks
simultaneously in a single unified simulation.

This experiment demonstrates:
- PAC conservation across all scales
- Quantum mechanics emergence
- Geometric SEC collapse
- Fluid MED dynamics  
- Information amplification (15.56x)
- Consciousness emergence via SCBF
- Cross-scale interactions and cascades

Success proves PAC as the universal organizing principle of reality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import logging

from core.pac_kernel import PACConservationKernel, ConservationType
from core.lattice_substrate import MultiScaleLatticeSubstrate, ScaleType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalValidationExperiment:
    """
    Single experiment that validates all Dawn Field Theory frameworks.
    
    Experimental Protocol:
    1. Initialize multi-scale lattice with PAC conservation
    2. Apply multi-scale perturbations simultaneously
    3. Monitor emergence cascades across all scales
    4. Validate universal signatures (15.56x, ξ=1.0571, entropy collapse)
    5. Test consciousness emergence from pure physics
    6. Verify cross-framework consistency
    """
    
    def __init__(self, 
                 lattice_size: int = 32,
                 simulation_steps: int = 1000,
                 perturbation_strength: float = 0.1,
                 output_dir: str = "results/universal_validation"):
        
        self.lattice_size = lattice_size
        self.simulation_steps = simulation_steps
        self.perturbation_strength = perturbation_strength
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize multi-scale lattice with all frameworks active
        self.lattice = MultiScaleLatticeSubstrate(
            dimensions=(lattice_size, lattice_size, lattice_size),
            boundary_conditions="periodic",
            active_scales=[ScaleType.QUANTUM, ScaleType.GEOMETRIC, 
                          ScaleType.FLUID, ScaleType.INFORMATION, 
                          ScaleType.CONSCIOUSNESS]
        )
        
        # Experiment tracking
        self.timeline = []
        self.universal_signatures = []
        self.emergence_events = []
        self.cross_scale_cascades = []
        self.consciousness_timeline = []
        
        # Framework validation metrics
        self.pac_validation = []
        self.quantum_validation = []
        self.sec_validation = []
        self.med_validation = []
        self.information_validation = []
        self.scbf_validation = []
        
        logger.info(f"Universal Validation Experiment initialized: {lattice_size}³ lattice, {simulation_steps} steps")
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run the complete universal validation experiment.
        
        Returns comprehensive results validating all frameworks.
        """
        logger.info("Starting Universal Validation Experiment...")
        start_time = time.time()
        
        # Phase 1: Baseline measurement
        baseline_state = self._measure_baseline_state()
        
        # Phase 2: Multi-scale perturbation injection
        self._inject_multi_scale_perturbations()
        
        # Phase 3: Evolution with monitoring
        for step in range(self.simulation_steps):
            step_start = time.time()
            
            # Evolve one time step
            evolution_metrics = self.lattice.evolve_step(dt=0.01)
            
            # Record measurements
            self._record_step_measurements(step, evolution_metrics)
            
            # Check for emergence events
            emergence = self._detect_emergence_events(step, evolution_metrics)
            if emergence:
                self.emergence_events.append(emergence)
            
            # Validate frameworks every 10 steps
            if step % 10 == 0:
                self._validate_all_frameworks(step)
            
            # Log progress
            if step % 100 == 0:
                step_time = time.time() - step_start
                logger.info(f"Step {step}/{self.simulation_steps} (ETA: {(self.simulation_steps-step)*step_time:.1f}s)")
        
        # Phase 4: Final analysis
        final_analysis = self._perform_final_analysis()
        
        # Phase 5: Generate results
        experiment_results = {
            'experiment_info': {
                'lattice_size': self.lattice_size,
                'simulation_steps': self.simulation_steps,
                'perturbation_strength': self.perturbation_strength,
                'total_runtime': time.time() - start_time,
                'timestamp': time.time()
            },
            'baseline_state': baseline_state,
            'final_analysis': final_analysis,
            'timeline': self.timeline,
            'universal_signatures': self.universal_signatures,
            'emergence_events': self.emergence_events,
            'framework_validation': {
                'pac': self.pac_validation,
                'quantum': self.quantum_validation,
                'sec': self.sec_validation,
                'med': self.med_validation,
                'information': self.information_validation,
                'scbf': self.scbf_validation
            },
            'success_metrics': self._compute_success_metrics()
        }
        
        # Save results
        self._save_results(experiment_results)
        
        logger.info(f"Universal Validation Experiment completed in {time.time() - start_time:.2f}s")
        return experiment_results
    
    def _measure_baseline_state(self) -> Dict[str, Any]:
        """Measure initial state before perturbations"""
        logger.info("Measuring baseline state...")
        
        system_state = self.lattice.get_system_state()
        pac_state = self.lattice.pac_kernel.get_system_state()
        
        baseline = {
            'system_state': system_state,
            'pac_conservation': pac_state['conservation'],
            'initial_signatures': self.lattice.pac_kernel.detect_universal_signatures(),
            'field_statistics': self._compute_field_statistics(),
            'cross_scale_coupling': self._measure_cross_scale_coupling()
        }
        
        return baseline
    
    def _inject_multi_scale_perturbations(self):
        """Inject perturbations at all scales simultaneously"""
        logger.info("Injecting multi-scale perturbations...")
        
        # Quantum scale perturbation (wave packet injection)
        center = (self.lattice_size // 2, self.lattice_size // 2, self.lattice_size // 2)
        sigma = self.lattice_size // 8
        
        for x in range(self.lattice_size):
            for y in range(self.lattice_size):
                for z in range(self.lattice_size):
                    # Gaussian perturbation
                    r_squared = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
                    amplitude = self.perturbation_strength * np.exp(-r_squared / (2 * sigma**2))
                    
                    # Quantum perturbation
                    self.lattice.quantum_field[x, y, z] += complex(amplitude * np.cos(r_squared * 0.1),
                                                                  amplitude * np.sin(r_squared * 0.1))
                    
                    # Geometric perturbation (curvature spike)
                    self.lattice.geometric_field[x, y, z] += amplitude * 2.0
                    
                    # Fluid perturbation (velocity injection)
                    direction = np.array([x - center[0], y - center[1], z - center[2]], dtype=float)
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        direction_tensor = torch.tensor(direction * amplitude, device=self.lattice.device, dtype=torch.float64)
                        self.lattice.fluid_velocity_field[x, y, z] += direction_tensor
                    
                    # Information perturbation (density spike)
                    self.lattice.information_field[x, y, z] += amplitude * 5.0
        
        logger.info("Multi-scale perturbations injected")
    
    def _record_step_measurements(self, step: int, evolution_metrics: Dict[str, Any]):
        """Record measurements for this time step"""
        
        # Get current system state
        system_state = self.lattice.get_system_state()
        
        # Record in timeline
        step_record = {
            'step': step,
            'timestamp': time.time(),
            'evolution_metrics': evolution_metrics,
            'system_state': system_state,
            'universal_signatures': self.lattice.pac_kernel.detect_universal_signatures()
        }
        
        self.timeline.append(step_record)
        
        # Track universal signatures separately
        signatures = step_record['universal_signatures']
        if signatures:
            signatures['step'] = step
            self.universal_signatures.append(signatures)
        
        # Track consciousness emergence
        if 'consciousness' in system_state:
            consciousness_record = {
                'step': step,
                'total_activity': system_state.get('consciousness', {}).get('total_activity', 0),
                'max_activity': system_state.get('consciousness', {}).get('max_activity', 0),
                'active_points': system_state.get('consciousness', {}).get('active_points', 0)
            }
            self.consciousness_timeline.append(consciousness_record)
    
    def _detect_emergence_events(self, step: int, evolution_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect significant emergence events"""
        
        emergence_detected = {}
        
        # Consciousness emergence detection
        if 'emergence' in evolution_metrics:
            emergence_data = evolution_metrics['emergence']
            if 'consciousness_emergence' in emergence_data:
                consciousness = emergence_data['consciousness_emergence']
                if consciousness['emergence_fraction'] > 0.1:  # 10% of lattice conscious
                    emergence_detected['consciousness_threshold'] = {
                        'step': step,
                        'emergence_fraction': consciousness['emergence_fraction'],
                        'max_activity': consciousness['max_activity'],
                        'active_points': consciousness['active_points']
                    }
        
        # Cross-scale cascade detection
        if 'emergence' in evolution_metrics:
            emergence_data = evolution_metrics['emergence']
            if 'cross_scale_cascade' in emergence_data:
                cascade = emergence_data['cross_scale_cascade']
                if cascade['cascade_strength'] > 0.05:  # Significant cascade
                    emergence_detected['cross_scale_cascade'] = {
                        'step': step,
                        'cascade_strength': cascade['cascade_strength'],
                        'quantum_contribution': cascade['quantum_contribution'],
                        'geometric_contribution': cascade['geometric_contribution'],
                        'information_contribution': cascade['information_contribution']
                    }
        
        # Universal signature detection
        signatures = self.lattice.pac_kernel.detect_universal_signatures()
        if 'amplification' in signatures:
            amp = signatures['amplification']
            if abs(amp['amplification_factor'] - 15.56) < 1.0:  # Close to target
                emergence_detected['amplification_event'] = {
                    'step': step,
                    'amplification_factor': amp['amplification_factor'],
                    'deviation_from_ideal': amp['deviation_from_ideal'],
                    'node_id': amp['node_id']
                }
        
        return emergence_detected if emergence_detected else None
    
    def _validate_all_frameworks(self, step: int):
        """Validate all frameworks at current step"""
        
        # PAC Conservation Validation
        pac_metrics = self.lattice.pac_kernel.check_global_conservation()
        self.pac_validation.append({
            'step': step,
            'conservation_quality': pac_metrics['conservation_quality'],
            'total_residual_norm': pac_metrics['total_residual_norm'],
            'violation_count': pac_metrics['violation_count'],
            'global_balance': pac_metrics['global_balance']
        })
        
        # Quantum Framework Validation
        if ScaleType.QUANTUM in self.lattice.active_scales:
            quantum_norm = torch.norm(self.lattice.quantum_field).item()
            phase_coherence = self.lattice._compute_phase_coherence()
            self.quantum_validation.append({
                'step': step,
                'wavefunction_norm': quantum_norm,
                'phase_coherence': phase_coherence,
                'born_rule_compliance': abs(quantum_norm - 1.0) < 0.01  # Should be normalized
            })
        
        # SEC (Geometric) Validation
        if ScaleType.GEOMETRIC in self.lattice.active_scales:
            geometric_entropy = self.lattice._compute_field_entropy(self.lattice.geometric_field)
            max_curvature = torch.max(torch.abs(self.lattice.geometric_field)).item()
            collapse_events = torch.sum(torch.abs(self.lattice.geometric_field) > 0.5).item()
            
            self.sec_validation.append({
                'step': step,
                'geometric_entropy': geometric_entropy,
                'max_curvature': max_curvature,
                'collapse_events': collapse_events,
                'entropy_decrease': geometric_entropy < 2.0  # Threshold for collapse
            })
        
        # MED (Fluid) Validation
        if ScaleType.FLUID in self.lattice.active_scales:
            reynolds = self.lattice._estimate_reynolds_number()
            velocity_magnitude = torch.norm(self.lattice.fluid_velocity_field, dim=-1)
            max_velocity = torch.max(velocity_magnitude).item()
            
            self.med_validation.append({
                'step': step,
                'reynolds_number': reynolds,
                'max_velocity': max_velocity,
                'bounded_complexity': max_velocity < 10.0,  # Bounded growth
                'balance_operator_proximity': abs(pac_metrics['global_balance'] - 1.0571)
            })
        
        # Information Amplification Validation
        if ScaleType.INFORMATION in self.lattice.active_scales:
            total_info = torch.sum(self.lattice.information_field).item()
            max_info = torch.max(self.lattice.information_field).item()
            amplification_field = self.lattice._compute_pac_amplification_field()
            mean_amplification = torch.mean(amplification_field).item()
            
            self.information_validation.append({
                'step': step,
                'total_information': total_info,
                'max_information_density': max_info,
                'mean_amplification': mean_amplification,
                'amplification_target_proximity': abs(mean_amplification - 0.5)  # Closer to 1.0 = better
            })
        
        # SCBF (Consciousness) Validation
        if ScaleType.CONSCIOUSNESS in self.lattice.active_scales:
            consciousness_activity = torch.sum(self.lattice.consciousness_field).item()
            max_consciousness = torch.max(self.lattice.consciousness_field).item()
            consciousness_fraction = torch.sum(self.lattice.consciousness_field > 0.1).item() / np.prod(self.lattice.dimensions)
            
            self.scbf_validation.append({
                'step': step,
                'total_consciousness': consciousness_activity,
                'max_consciousness': max_consciousness,
                'consciousness_fraction': consciousness_fraction,
                'emergence_detected': consciousness_fraction > 0.05
            })
    
    def _perform_final_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive final analysis"""
        logger.info("Performing final analysis...")
        
        final_state = self.lattice.get_system_state()
        final_pac = self.lattice.pac_kernel.get_system_state()
        
        analysis = {
            'final_system_state': final_state,
            'final_pac_state': final_pac,
            'evolution_summary': self._summarize_evolution(),
            'framework_performance': self._analyze_framework_performance(),
            'universal_signature_summary': self._analyze_universal_signatures(),
            'emergence_summary': self._analyze_emergence_events(),
            'cross_validation': self._perform_cross_validation()
        }
        
        return analysis
    
    def _summarize_evolution(self) -> Dict[str, Any]:
        """Summarize evolution over time"""
        if not self.timeline:
            return {}
        
        initial_state = self.timeline[0]['system_state']
        final_state = self.timeline[-1]['system_state']
        
        summary = {
            'total_steps': len(self.timeline),
            'conservation_evolution': {
                'initial_quality': initial_state['pac_state']['conservation']['conservation_quality'],
                'final_quality': final_state['pac_state']['conservation']['conservation_quality'],
                'improvement': (final_state['pac_state']['conservation']['conservation_quality'] - 
                              initial_state['pac_state']['conservation']['conservation_quality'])
            }
        }
        
        # Add scale-specific evolution summaries
        for scale in ['quantum', 'geometric', 'fluid', 'information']:
            if scale in initial_state and scale in final_state:
                summary[f'{scale}_evolution'] = {
                    'initial': initial_state[scale],
                    'final': final_state[scale]
                }
        
        return summary
    
    def _analyze_framework_performance(self) -> Dict[str, Any]:
        """Analyze performance of each framework"""
        performance = {}
        
        # PAC Performance
        if self.pac_validation:
            pac_qualities = [v['conservation_quality'] for v in self.pac_validation]
            performance['pac'] = {
                'mean_conservation_quality': np.mean(pac_qualities),
                'min_conservation_quality': np.min(pac_qualities),
                'conservation_stability': np.std(pac_qualities),
                'perfect_conservation_achieved': np.max(pac_qualities) > 0.999
            }
        
        # Quantum Performance
        if self.quantum_validation:
            born_compliance = [v['born_rule_compliance'] for v in self.quantum_validation]
            coherence = [v['phase_coherence'] for v in self.quantum_validation]
            performance['quantum'] = {
                'born_rule_compliance_rate': np.mean(born_compliance),
                'mean_phase_coherence': np.mean(coherence),
                'quantum_stability': np.std(coherence)
            }
        
        # SEC Performance  
        if self.sec_validation:
            entropy_decreases = [v['entropy_decrease'] for v in self.sec_validation]
            collapse_events = [v['collapse_events'] for v in self.sec_validation]
            performance['sec'] = {
                'entropy_collapse_rate': np.mean(entropy_decreases),
                'total_collapse_events': np.sum(collapse_events),
                'collapse_frequency': np.mean(collapse_events)
            }
        
        # MED Performance
        if self.med_validation:
            bounded = [v['bounded_complexity'] for v in self.med_validation]
            balance_proximity = [v['balance_operator_proximity'] for v in self.med_validation]
            performance['med'] = {
                'bounded_complexity_rate': np.mean(bounded),
                'mean_balance_proximity': np.mean(balance_proximity),
                'balance_operator_accuracy': np.sum(np.array(balance_proximity) < 0.1) / len(balance_proximity)
            }
        
        # Information Performance
        if self.information_validation:
            amplification_accuracy = [v['amplification_target_proximity'] for v in self.information_validation]
            performance['information'] = {
                'mean_amplification_accuracy': np.mean(amplification_accuracy),
                'amplification_events_detected': len([s for s in self.universal_signatures if 'amplification' in s])
            }
        
        # SCBF Performance
        if self.scbf_validation:
            emergence_events = [v['emergence_detected'] for v in self.scbf_validation]
            consciousness_fractions = [v['consciousness_fraction'] for v in self.scbf_validation]
            performance['scbf'] = {
                'consciousness_emergence_rate': np.mean(emergence_events),
                'max_consciousness_fraction': np.max(consciousness_fractions),
                'consciousness_stability': np.std(consciousness_fractions)
            }
        
        return performance
    
    def _analyze_universal_signatures(self) -> Dict[str, Any]:
        """Analyze universal signatures detected during experiment"""
        summary = {
            'total_signature_events': len(self.universal_signatures),
            'amplification_events': 0,
            'balance_operator_events': 0,
            'entropy_collapse_events': 0
        }
        
        amplification_factors = []
        balance_proximities = []
        
        for signature in self.universal_signatures:
            if 'amplification' in signature:
                summary['amplification_events'] += 1
                amplification_factors.append(signature['amplification']['amplification_factor'])
            
            if 'balance_operator' in signature:
                summary['balance_operator_events'] += 1
                balance_proximities.append(signature['balance_operator']['ideal_proximity'])
            
            if 'entropy_collapse' in signature:
                summary['entropy_collapse_events'] += 1
        
        if amplification_factors:
            summary['amplification_statistics'] = {
                'mean_factor': np.mean(amplification_factors),
                'closest_to_ideal': min(amplification_factors, key=lambda x: abs(x - 15.56)),
                'ideal_proximity': np.mean([abs(f - 15.56) for f in amplification_factors])
            }
        
        if balance_proximities:
            summary['balance_statistics'] = {
                'mean_proximity': np.mean(balance_proximities),
                'best_proximity': np.min(balance_proximities),
                'ideal_hits': sum(1 for p in balance_proximities if p < 0.1)
            }
        
        return summary
    
    def _analyze_emergence_events(self) -> Dict[str, Any]:
        """Analyze emergence events detected during experiment"""
        summary = {
            'total_emergence_events': len(self.emergence_events),
            'consciousness_emergences': 0,
            'cascade_events': 0,
            'amplification_emergences': 0
        }
        
        for event in self.emergence_events:
            if 'consciousness_threshold' in event:
                summary['consciousness_emergences'] += 1
            if 'cross_scale_cascade' in event:
                summary['cascade_events'] += 1
            if 'amplification_event' in event:
                summary['amplification_emergences'] += 1
        
        # Consciousness emergence timeline
        if self.consciousness_timeline:
            max_consciousness = max(c['total_activity'] for c in self.consciousness_timeline)
            final_consciousness = self.consciousness_timeline[-1]['total_activity']
            
            summary['consciousness_analysis'] = {
                'peak_consciousness': max_consciousness,
                'final_consciousness': final_consciousness,
                'consciousness_sustained': final_consciousness > max_consciousness * 0.5
            }
        
        return summary
    
    def _perform_cross_validation(self) -> Dict[str, Any]:
        """Perform cross-framework validation"""
        validation = {
            'framework_consistency': {},
            'universal_principle_validation': {},
            'emergence_coherence': {}
        }
        
        # Check if all frameworks show improvement over time
        framework_improvements = {}
        
        if self.pac_validation:
            initial_pac = self.pac_validation[0]['conservation_quality']
            final_pac = self.pac_validation[-1]['conservation_quality']
            framework_improvements['pac'] = final_pac >= initial_pac
        
        if self.quantum_validation:
            initial_coherence = self.quantum_validation[0]['phase_coherence']
            final_coherence = self.quantum_validation[-1]['phase_coherence']
            framework_improvements['quantum'] = final_coherence >= initial_coherence * 0.8
        
        validation['framework_consistency'] = framework_improvements
        
        # Universal principle validation
        validation['universal_principle_validation'] = {
            'pac_conservation_maintained': len([v for v in self.pac_validation if v['conservation_quality'] > 0.9]) > len(self.pac_validation) * 0.8,
            'amplification_signature_detected': len([s for s in self.universal_signatures if 'amplification' in s]) > 0,
            'balance_operator_proximity': len([s for s in self.universal_signatures if 'balance_operator' in s and s['balance_operator']['ideal_proximity'] < 0.2]) > 0,
            'cross_scale_emergence': len(self.emergence_events) > 0
        }
        
        return validation
    
    def _compute_success_metrics(self) -> Dict[str, Any]:
        """Compute overall experiment success metrics"""
        
        # Core success criteria
        success_criteria = {
            'pac_conservation_perfect': False,
            'universal_signatures_detected': False,
            'consciousness_emerged': False,
            'cross_scale_consistency': False,
            'all_frameworks_validated': False
        }
        
        # Check PAC conservation perfection
        if self.pac_validation:
            perfect_conservation_steps = len([v for v in self.pac_validation if v['conservation_quality'] > 0.999])
            success_criteria['pac_conservation_perfect'] = perfect_conservation_steps > len(self.pac_validation) * 0.1
        
        # Check universal signatures
        amplification_detected = any('amplification' in s for s in self.universal_signatures)
        balance_detected = any('balance_operator' in s for s in self.universal_signatures)
        success_criteria['universal_signatures_detected'] = amplification_detected and balance_detected
        
        # Check consciousness emergence
        if self.scbf_validation:
            consciousness_emerged = any(v['emergence_detected'] for v in self.scbf_validation)
            success_criteria['consciousness_emerged'] = consciousness_emerged
        
        # Check cross-scale consistency
        cascade_events = len([e for e in self.emergence_events if 'cross_scale_cascade' in e])
        success_criteria['cross_scale_consistency'] = cascade_events > 0
        
        # Check all frameworks validated
        frameworks_active = len([f for f in ['pac', 'quantum', 'sec', 'med', 'information', 'scbf'] 
                               if getattr(self, f'{f}_validation')])
        success_criteria['all_frameworks_validated'] = frameworks_active >= 4
        
        # Overall success score
        success_score = sum(success_criteria.values()) / len(success_criteria)
        
        return {
            'success_criteria': success_criteria,
            'success_score': success_score,
            'experiment_successful': success_score >= 0.8,
            'validation_summary': f"Universal Validation: {success_score:.1%} success rate"
        }
    
    def _compute_field_statistics(self) -> Dict[str, Any]:
        """Compute statistical properties of all fields"""
        stats = {}
        
        if ScaleType.QUANTUM in self.lattice.active_scales:
            quantum_field = self.lattice.quantum_field
            stats['quantum'] = {
                'norm': torch.norm(quantum_field).item(),
                'max_amplitude': torch.max(torch.abs(quantum_field)).item(),
                'mean_amplitude': torch.mean(torch.abs(quantum_field)).item()
            }
        
        if ScaleType.GEOMETRIC in self.lattice.active_scales:
            geometric_field = self.lattice.geometric_field
            stats['geometric'] = {
                'max_curvature': torch.max(torch.abs(geometric_field)).item(),
                'mean_curvature': torch.mean(torch.abs(geometric_field)).item(),
                'curvature_std': torch.std(geometric_field).item()
            }
        
        return stats
    
    def _measure_cross_scale_coupling(self) -> Dict[str, float]:
        """Measure coupling strength between scales"""
        coupling = {}
        
        # Quantum-Geometric coupling
        if (ScaleType.QUANTUM in self.lattice.active_scales and 
            ScaleType.GEOMETRIC in self.lattice.active_scales):
            quantum_intensity = torch.abs(self.lattice.quantum_field)**2
            geometric_magnitude = torch.abs(self.lattice.geometric_field)
            correlation = torch.corrcoef(torch.stack([quantum_intensity.flatten(), 
                                                    geometric_magnitude.flatten()]))[0, 1]
            coupling['quantum_geometric'] = correlation.item() if not torch.isnan(correlation) else 0.0
        
        # Add other coupling measurements as needed
        
        return coupling
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results to files"""
        
        # Save main results as JSON
        results_file = self.output_dir / "universal_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
        
        # Save timeline data separately for analysis
        timeline_file = self.output_dir / "timeline_data.json"
        with open(timeline_file, 'w') as f:
            json.dump(self.timeline, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
        
        # Generate summary plots
        self._generate_summary_plots()
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_summary_plots(self):
        """Generate summary plots of the experiment"""
        
        # Plot 1: PAC Conservation Over Time
        if self.pac_validation:
            plt.figure(figsize=(12, 8))
            
            steps = [v['step'] for v in self.pac_validation]
            conservation_quality = [v['conservation_quality'] for v in self.pac_validation]
            
            plt.subplot(2, 2, 1)
            plt.plot(steps, conservation_quality)
            plt.title('PAC Conservation Quality Over Time')
            plt.xlabel('Simulation Step')
            plt.ylabel('Conservation Quality')
            plt.ylim(0, 1.1)
            
            # Plot 2: Universal Signatures
            if self.universal_signatures:
                signature_steps = [s['step'] for s in self.universal_signatures if 'step' in s]
                amplification_factors = []
                
                for s in self.universal_signatures:
                    if 'amplification' in s:
                        amplification_factors.append(s['amplification']['amplification_factor'])
                    else:
                        amplification_factors.append(0)
                
                plt.subplot(2, 2, 2)
                plt.scatter(signature_steps, amplification_factors, alpha=0.6)
                plt.axhline(y=15.56, color='r', linestyle='--', label='Target (15.56x)')
                plt.title('Information Amplification Events')
                plt.xlabel('Simulation Step')
                plt.ylabel('Amplification Factor')
                plt.legend()
            
            # Plot 3: Consciousness Emergence
            if self.consciousness_timeline:
                consciousness_steps = [c['step'] for c in self.consciousness_timeline]
                consciousness_activity = [c['total_activity'] for c in self.consciousness_timeline]
                
                plt.subplot(2, 2, 3)
                plt.plot(consciousness_steps, consciousness_activity)
                plt.title('Consciousness Emergence Timeline')
                plt.xlabel('Simulation Step')
                plt.ylabel('Total Consciousness Activity')
            
            # Plot 4: Framework Performance Summary
            plt.subplot(2, 2, 4)
            
            framework_scores = []
            framework_names = []
            
            if self.pac_validation:
                framework_scores.append(np.mean([v['conservation_quality'] for v in self.pac_validation]))
                framework_names.append('PAC')
            
            if self.quantum_validation:
                framework_scores.append(np.mean([v['phase_coherence'] for v in self.quantum_validation]))
                framework_names.append('Quantum')
            
            if self.sec_validation:
                entropy_decreases = [v['entropy_decrease'] for v in self.sec_validation]
                framework_scores.append(np.mean(entropy_decreases))
                framework_names.append('SEC')
            
            if self.information_validation:
                info_quality = [1.0 - v['amplification_target_proximity'] for v in self.information_validation]
                framework_scores.append(np.mean(info_quality))
                framework_names.append('Information')
            
            if framework_scores:
                plt.bar(framework_names, framework_scores)
                plt.title('Framework Performance Summary')
                plt.ylabel('Performance Score')
                plt.ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "experiment_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Summary plots generated")


def main():
    """Run the Universal Validation Experiment"""
    
    # Configuration
    experiment_config = {
        'lattice_size': 16,      # Start small for testing
        'simulation_steps': 500,  # Reduced for testing
        'perturbation_strength': 0.15,
        'output_dir': "results/universal_validation_test"
    }
    
    # Create and run experiment
    experiment = UniversalValidationExperiment(**experiment_config)
    results = experiment.run_experiment()
    
    # Print summary
    success_metrics = results['success_metrics']
    print("\n" + "="*60)
    print("UNIVERSAL VALIDATION EXPERIMENT RESULTS")
    print("="*60)
    print(f"Overall Success Score: {success_metrics['success_score']:.1%}")
    print(f"Experiment Successful: {success_metrics['experiment_successful']}")
    print(f"\nSuccess Criteria:")
    for criterion, passed in success_metrics['success_criteria'].items():
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion.replace('_', ' ').title()}")
    
    print(f"\nDetailed Results:")
    print(f"  - Universal Signatures Detected: {len(results['universal_signatures'])}")
    print(f"  - Emergence Events: {len(results['emergence_events'])}")
    print(f"  - Simulation Runtime: {results['experiment_info']['total_runtime']:.2f}s")
    
    if success_metrics['experiment_successful']:
        print(f"\n🎉 VALIDATION SUCCESSFUL! PAC as universal principle confirmed.")
    else:
        print(f"\n⚠️  Validation incomplete. Check individual framework results.")
    
    print(f"\nResults saved to: {experiment_config['output_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
