"""
TinyCIMM-Navier Live CIMM Experiment Runner

Progressive validation of fluid dynamics learning using True CIMM Architecture.
Validates live pattern crystallization, entropy-driven insights, and real-time adaptation.

Experimental progression:
1. Live pattern discovery (no training loops)
2. Entropy collapse detection (flow insights)  
3. Reynolds regime adaptation (structural dynamics)
4. Turbulent pattern crystallization (breakthrough challenge)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

from tinycimm_navier import TinyCIMMNavier, create_flow_boundary_conditions
from tinycimm_navier_dashboard import generate_tinycimm_navier_dashboards

class LiveCIMMFlowBenchmark:
    """
    Live CIMM benchmark suite for TinyCIMM-Navier validation.
    Tests real-time pattern crystallization and entropy-driven adaptation.
    No training loops - pure CIMM architecture validation.
    """
    
    def __init__(self, save_results=True):
        self.save_results = save_results
        self.results = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        if save_results:
            self.results_dir = f"results/live_cimm_experiment_{self.experiment_id}"
            os.makedirs(self.results_dir, exist_ok=True)
    
    def run_live_pattern_discovery(self):
        """
        Phase 1: Live pattern discovery validation
        Test real-time pattern crystallization without training
        """
        print("=== Phase 1: Live Pattern Discovery ===")
        
        model = TinyCIMMNavier(device='cpu')
        
        # Test scenarios for pattern discovery
        scenarios = [
            {"name": "poiseuille_flow", "reynolds": 800, "complexity": 0.1, "steps": 50},
            {"name": "couette_flow", "reynolds": 1200, "complexity": 0.15, "steps": 40},
            {"name": "stagnation_flow", "reynolds": 600, "complexity": 0.08, "steps": 30},
        ]
        
        phase_results = {}
        
        for scenario in scenarios:
            print(f"\nTesting {scenario['name']}...")
            
            scenario_results = {
                'patterns_discovered': [],
                'collapse_events': [],
                'entropy_budget_history': [],
                'prediction_times': [],
                'flow_regime_transitions': [],
                'scbf_neural_dynamics': []  # SCBF tracking
            }            # Live pattern discovery session
            for step in range(scenario['steps']):
                # Generate flow input
                flow_input = torch.randn(1, 8) * scenario['complexity']
                
                # Live prediction
                start_time = time.time()
                prediction, diagnostics = model.live_predict(flow_input, scenario['reynolds'])
                prediction_time = (time.time() - start_time) * 1000
                
                # Track results
                scenario_results['prediction_times'].append(prediction_time)
                scenario_results['entropy_budget_history'].append(diagnostics['entropy_budget'])
                
                if diagnostics['collapse_event']['flow_insight_detected']:
                    scenario_results['collapse_events'].append({
                        'step': step,
                        'magnitude': diagnostics['collapse_event']['collapse_magnitude'],
                        'type': diagnostics['collapse_event']['insight_type']
                    })
                
                if diagnostics['resonant_patterns']:
                    scenario_results['patterns_discovered'].append({
                        'step': step,
                        'pattern': diagnostics['resonant_patterns'][0]
                    })
                
                # SCBF neural dynamics tracking
                if diagnostics.get('scbf_metrics'):
                    scenario_results['scbf_neural_dynamics'].append({
                        'step': step,
                        'neural_score': diagnostics['scbf_metrics'].get('neural_dynamics_score', 0),
                        'entropy_collapse': diagnostics['scbf_metrics'].get('entropy_collapse', {}),
                        'structural_evolution': diagnostics['scbf_metrics'].get('structural_evolution', {}),
                        'neural_ancestry': diagnostics['scbf_metrics'].get('neural_ancestry', {}),
                        'pattern_attractors': diagnostics['scbf_metrics'].get('pattern_attractors', {})
                    })
                
                # Progress report with SCBF metrics
                if step % 10 == 0:
                    scbf_info = ""
                    if diagnostics.get('scbf_metrics'):
                        scbf = diagnostics['scbf_metrics']
                        neural_score = scbf.get('neural_dynamics_score', 0)
                        entropy_collapse = scbf.get('entropy_collapse', {}).get('collapse_magnitude', 0)
                        mutation_rate = scbf.get('structural_evolution', {}).get('mutation_rate', 0)
                        scbf_info = f" | 🧠 Neural: {neural_score:.2f} | 🔬 Collapse: {entropy_collapse:.3f} | 🧬 Mutation: {mutation_rate:.3f}"
                    
                    print(f"  Step {step:2d}: {prediction_time:.1f}ms | "
                          f"Regime: {diagnostics['flow_regime']:>10s} | "
                          f"Patterns: {diagnostics['crystals_discovered']} | "
                          f"Budget: {diagnostics['entropy_budget']:.3f}{scbf_info}")
            
            phase_results[scenario['name']] = scenario_results
            
            # Scenario summary
            avg_time = np.mean(scenario_results['prediction_times'])
            total_patterns = len(set([p['pattern'] for p in scenario_results['patterns_discovered']]))
            total_collapses = len(scenario_results['collapse_events'])
            
            print(f"  Summary: {avg_time:.1f}ms avg | {total_patterns} unique patterns | {total_collapses} collapses")
        
        return phase_results
    
    def run_entropy_collapse_validation(self):
        """
        Phase 2: Entropy collapse detection validation
        Test sensitivity to symbolic collapse events
        """
        print("\n=== Phase 2: Entropy Collapse Detection ===")
        
        model = TinyCIMMNavier(device='cpu')
        
        # Scenarios designed to trigger entropy collapses
        collapse_scenarios = [
            {"name": "regime_transition", "reynolds_sequence": [500, 1000, 2000, 4000, 8000], "complexity": 0.2},
            {"name": "complexity_ramp", "reynolds": 3000, "complexity_sequence": [0.1, 0.3, 0.6, 1.0, 1.5], "base_complexity": 0.2},
            {"name": "pattern_repetition", "reynolds": 1500, "complexity": 0.15, "repeat_pattern": True}
        ]
        
        phase_results = {}
        
        for scenario in collapse_scenarios:
            print(f"\nTesting {scenario['name']}...")
            
            scenario_results = {
                'major_collapses': [],
                'pattern_crystallizations': [],
                'entropy_dynamics': [],
                'insights_timeline': []
            }
            
            if scenario['name'] == 'regime_transition':
                # Test Reynolds regime transitions
                for i, reynolds in enumerate(scenario['reynolds_sequence']):
                    flow_input = torch.randn(1, 8) * scenario['complexity']
                    
                    prediction, diagnostics = model.live_predict(flow_input, reynolds)
                    
                    scenario_results['entropy_dynamics'].append({
                        'step': i,
                        'reynolds': reynolds,
                        'entropy_budget': diagnostics['entropy_budget'],
                        'flow_regime': diagnostics['flow_regime']
                    })
                    
                    if diagnostics['collapse_event']['flow_insight_detected']:
                        scenario_results['major_collapses'].append({
                            'reynolds': reynolds,
                            'magnitude': diagnostics['collapse_event']['collapse_magnitude'],
                            'type': diagnostics['collapse_event']['insight_type']
                        })
                        
                        print(f"  🔮 Collapse at Re={reynolds}: {diagnostics['collapse_event']['insight_type']}")
            
            elif scenario['name'] == 'complexity_ramp':
                # Test complexity-driven collapses
                reynolds = scenario['reynolds']
                for i, complexity in enumerate(scenario['complexity_sequence']):
                    flow_input = torch.randn(1, 8) * complexity
                    
                    prediction, diagnostics = model.live_predict(flow_input, reynolds)
                    
                    if diagnostics['collapse_event']['flow_insight_detected']:
                        scenario_results['major_collapses'].append({
                            'complexity': complexity,
                            'magnitude': diagnostics['collapse_event']['collapse_magnitude']
                        })
                        
                        print(f"  🔮 Collapse at complexity={complexity:.1f}: {diagnostics['collapse_event']['insight_type']}")
            
            elif scenario['name'] == 'pattern_repetition':
                # Test repeated pattern recognition
                reynolds = scenario['reynolds']
                base_pattern = torch.randn(1, 8) * scenario['complexity']
                
                for i in range(20):
                    # Add small variations to base pattern
                    flow_input = base_pattern + torch.randn(1, 8) * 0.01
                    
                    prediction, diagnostics = model.live_predict(flow_input, reynolds)
                    
                    if diagnostics['resonant_patterns']:
                        scenario_results['pattern_crystallizations'].append({
                            'step': i,
                            'pattern': diagnostics['resonant_patterns'][0]
                        })
            
            phase_results[scenario['name']] = scenario_results
        
        return phase_results
    
    def run_reynolds_adaptation_test(self):
        """
        Phase 3: Reynolds regime adaptation test
        Test structural dynamics and regime recognition
        """
        print("\n=== Phase 3: Reynolds Regime Adaptation ===")
        
        model = TinyCIMMNavier(device='cpu')
        
        # Reynolds sweep test
        reynolds_sweep = [100, 500, 1000, 2000, 3000, 5000, 8000, 15000, 30000, 50000]
        
        adaptation_results = {
            'regime_recognition': [],
            'entropy_budget_evolution': [],
            'pattern_evolution': [],
            'structural_changes': []
        }
        
        print(f"Testing Reynolds sweep: {reynolds_sweep}")
        
        for reynolds in reynolds_sweep:
            # Generate appropriate complexity for Reynolds number
            complexity = min(1.5, 0.1 + (reynolds / 10000) * 0.5)
            flow_input = torch.randn(1, 8) * complexity
            
            # Live prediction
            prediction, diagnostics = model.live_predict(flow_input, reynolds)
            
            adaptation_results['regime_recognition'].append({
                'reynolds': reynolds,
                'detected_regime': diagnostics['flow_regime'],
                'entropy_budget': diagnostics['entropy_budget']
            })
            
            adaptation_results['entropy_budget_evolution'].append(diagnostics['entropy_budget'])
            
            if diagnostics['crystals_discovered'] > 0:
                adaptation_results['pattern_evolution'].append({
                    'reynolds': reynolds,
                    'patterns': diagnostics['crystals_discovered']
                })
            
            print(f"  Re={reynolds:5d}: Regime={diagnostics['flow_regime']:>10s} | "
                  f"Budget={diagnostics['entropy_budget']:.3f} | "
                  f"Patterns={diagnostics['crystals_discovered']}")
        
        return adaptation_results
    
    def run_turbulent_crystallization_challenge(self):
        """
        Phase 4: Turbulent pattern crystallization challenge
        Ultimate test of live pattern discovery in chaotic regimes
        """
        print("\n=== Phase 4: Turbulent Crystallization Challenge ===")
        
        model = TinyCIMMNavier(device='cpu')
        
        # High Reynolds turbulent scenarios with enhanced complexity
        turbulent_challenges = [
            {"name": "pipe_turbulence", "reynolds": 10000, "complexity": 1.3, "steps": 120},  # Increased complexity and steps
            {"name": "mixing_layer", "reynolds": 25000, "complexity": 1.5, "steps": 100},     # Enhanced complexity
            {"name": "high_re_chaos", "reynolds": 100000, "complexity": 2.5, "steps": 80},    # Maximum complexity challenge
            {"name": "extreme_turbulence", "reynolds": 200000, "complexity": 3.0, "steps": 60} # New extreme challenge
        ]
        
        challenge_results = {}
        
        for challenge in turbulent_challenges:
            print(f"\nTurbulent Challenge: {challenge['name']} (Re={challenge['reynolds']})")
            
            challenge_data = {
                'breakthrough_detected': False,
                'breakthrough_step': None,
                'patterns_discovered': [],
                'major_insights': [],
                'entropy_evolution': []
            }
            
            for step in range(challenge['steps']):
                # High complexity turbulent input
                flow_input = torch.randn(1, 8) * challenge['complexity']
                
                prediction, diagnostics = model.live_predict(flow_input, challenge['reynolds'])
                
                challenge_data['entropy_evolution'].append(diagnostics['entropy_budget'])
                
                # Enhanced breakthrough detection with multiple criteria
                breakthrough_conditions = [
                    (diagnostics['collapse_event']['flow_insight_detected'] and 
                     diagnostics['collapse_event']['insight_type'] == 'major_flow_insight'),
                    (diagnostics['collapse_event']['flow_insight_detected'] and
                     diagnostics['collapse_event']['collapse_magnitude'] > 0.08),  # High magnitude threshold
                    (diagnostics['entropy_budget'] > 2.5),  # High entropy accumulation
                    (diagnostics['crystals_discovered'] > challenge_data['patterns_discovered'][-1] + 2 
                     if challenge_data['patterns_discovered'] else False)  # Rapid pattern discovery
                ]
                
                if any(breakthrough_conditions):
                    if not challenge_data['breakthrough_detected']:
                        challenge_data['breakthrough_detected'] = True
                        challenge_data['breakthrough_step'] = step
                        print(f"  *** TURBULENT BREAKTHROUGH DETECTED at step {step}! ***")
                        print(f"      Trigger: Budget={diagnostics['entropy_budget']:.3f}, "
                              f"Magnitude={diagnostics['collapse_event'].get('collapse_magnitude', 0):.3f}")
                    
                    challenge_data['major_insights'].append({
                        'step': step,
                        'magnitude': diagnostics['collapse_event'].get('collapse_magnitude', 0),
                        'entropy_budget': diagnostics['entropy_budget']
                    })
                
                if diagnostics['crystals_discovered'] > len(challenge_data['patterns_discovered']):
                    new_patterns = diagnostics['crystals_discovered'] - len(challenge_data['patterns_discovered'])
                    challenge_data['patterns_discovered'].extend([step] * new_patterns)
                    print(f"  🔮 New turbulent pattern crystallized at step {step}")
                
                # Progress report
                if step % 20 == 0:
                    print(f"  Step {step:2d}: Budget={diagnostics['entropy_budget']:.3f} | "
                          f"Patterns={diagnostics['crystals_discovered']} | "
                          f"Insights={diagnostics['insights_discovered']}")
            
            challenge_results[challenge['name']] = challenge_data
            
            # Challenge summary
            if challenge_data['breakthrough_detected']:
                print(f"  ✅ BREAKTHROUGH: Step {challenge_data['breakthrough_step']} | "
                      f"Patterns: {len(challenge_data['patterns_discovered'])} | "
                      f"Insights: {len(challenge_data['major_insights'])}")
            else:
                print(f"  ⚠️  No breakthrough detected | "
                      f"Patterns: {len(challenge_data['patterns_discovered'])}")
        
        return challenge_results
    
    def run_comprehensive_validation(self):
        """Run complete live CIMM validation suite"""
        print("🚀 TinyCIMM-Navier Live CIMM Comprehensive Validation")
        print("True CIMM Architecture: Live Prediction + Pattern Crystallization + Entropy Insights")
        print("=" * 80)
        
        start_time = time.time()
        
        # Phase 1: Live pattern discovery
        pattern_results = self.run_live_pattern_discovery()
        
        # Phase 2: Entropy collapse validation
        collapse_results = self.run_entropy_collapse_validation()
        
        # Phase 3: Reynolds adaptation
        adaptation_results = self.run_reynolds_adaptation_test()
        
        # Phase 4: Turbulent crystallization challenge
        turbulent_results = self.run_turbulent_crystallization_challenge()
        
        total_time = time.time() - start_time
        
        # Compile comprehensive results
        comprehensive_results = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'total_validation_time': total_time,
            'cimm_architecture': 'live_prediction',
            'training_loops_used': False,
            'phase_1_pattern_discovery': pattern_results,
            'phase_2_entropy_collapse': collapse_results,
            'phase_3_reynolds_adaptation': adaptation_results,
            'phase_4_turbulent_challenge': turbulent_results
        }
        
        # Save results
        if self.save_results:
            results_file = f"{self.results_dir}/comprehensive_live_cimm_results.json"
            with open(results_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            
            # Generate comprehensive dashboards
            print(f"\n🎨 Generating analytical dashboards...")
            try:
                dashboard_paths = generate_tinycimm_navier_dashboards(results_file, self.results_dir)
                print(f"✅ Generated {len(dashboard_paths)} dashboard visualizations:")
                for path in dashboard_paths:
                    print(f"   📊 {os.path.basename(path)}")
            except Exception as e:
                print(f"⚠️ Dashboard generation failed: {e}")
        
        # Final summary
        print(f"\n🎯 Live CIMM Validation Complete!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Architecture: True CIMM (no training loops)")
        print(f"   Results saved to: {self.results_dir}")
        
        self._print_validation_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _print_validation_summary(self, results):
        """Print comprehensive validation summary"""
        print(f"\n✨ CIMM Validation Summary:")
        
        # Pattern discovery summary
        phase1 = results['phase_1_pattern_discovery']
        total_patterns = sum(len(set([p['pattern'] for p in scenario['patterns_discovered']])) 
                           for scenario in phase1.values())
        total_collapses = sum(len(scenario['collapse_events']) for scenario in phase1.values())
        
        print(f"   Phase 1 - Pattern Discovery:")
        print(f"     ✅ Unique patterns discovered: {total_patterns}")
        print(f"     ✅ Entropy collapses detected: {total_collapses}")
        
        # Turbulent challenge summary
        phase4 = results['phase_4_turbulent_challenge']
        breakthroughs = sum(1 for challenge in phase4.values() if challenge['breakthrough_detected'])
        total_turbulent_patterns = sum(len(challenge['patterns_discovered']) for challenge in phase4.values())
        
        print(f"   Phase 4 - Turbulent Challenge:")
        print(f"     🚀 Turbulent breakthroughs: {breakthroughs}/{len(phase4)}")
        print(f"     🔮 Turbulent patterns: {total_turbulent_patterns}")
        

def main():
    """Run the live CIMM validation experiment"""
    benchmark = LiveCIMMFlowBenchmark(save_results=True)
    results = benchmark.run_comprehensive_validation()
    return results

if __name__ == "__main__":
    results = main()
