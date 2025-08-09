#!/usr/bin/env python3
"""
SCBF Experiment Runner
=====================

Modular entry point for registering and running SCBF experiments.
Provides a clean interface for TinyCIMM-Euler integration.

Usage:
    python scbf_runner.py --experiment my_experiment
    python scbf_runner.py --list
    python scbf_runner.py --analyze experiment_id
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import importlib.util

# Add SCBF to path
SCBF_DIR = Path(__file__).parent
sys.path.insert(0, str(SCBF_DIR))

from loggers import create_experiment_logger, finalize_experiment, list_all_experiments, load_experiment_results
from visualization import plot_complete_scbf_dashboard, save_all_plots

class SCBFExperimentRunner:
    """Main experiment runner for SCBF analysis."""
    
    def __init__(self):
        self.registered_experiments: Dict[str, Callable] = {}
        self.experiment_configs: Dict[str, Dict] = {}
        
    def register_experiment(self, name: str, experiment_func: Callable, config: Optional[Dict] = None):
        """Register an experiment function."""
        self.registered_experiments[name] = experiment_func
        self.experiment_configs[name] = config or {}
        print(f"📝 Registered experiment: {name}")
    
    def run_experiment(self, name: str, **kwargs) -> str:
        """Run a registered experiment."""
        if name not in self.registered_experiments:
            raise ValueError(f"Experiment '{name}' not found. Available: {list(self.registered_experiments.keys())}")
        
        print(f"🚀 Starting SCBF experiment: {name}")
        
        # Create logger
        logger = create_experiment_logger(name)
        
        # Get experiment function and config
        experiment_func = self.registered_experiments[name]
        config = self.experiment_configs[name].copy()
        config.update(kwargs)
        
        try:
            # Run the experiment
            results = experiment_func(logger=logger, **config)
            
            # Finalize and save results
            results_file = finalize_experiment(logger)
            
            # Generate visualizations
            logs = logger.get_logs()
            if logs:
                print("📊 Generating visualizations...")
                plot_complete_scbf_dashboard(logs)
                save_all_plots(logs, f"scbf_plots_{logger.experiment_id}")
            
            print(f"✅ Experiment completed successfully!")
            print(f"📁 Results saved to: {results_file}")
            
            return results_file
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            logger.experiment_metadata['status'] = 'failed'
            logger.experiment_metadata['error'] = str(e)
            finalize_experiment(logger)
            raise
    
    def list_experiments(self) -> List[str]:
        """List all registered experiments."""
        return list(self.registered_experiments.keys())
    
    def list_completed_experiments(self) -> List[Dict]:
        """List all completed experiments."""
        return list_all_experiments()
    
    def analyze_experiment(self, experiment_id: str) -> None:
        """Analyze a completed experiment."""
        results = load_experiment_results(experiment_id)
        if not results:
            print(f"❌ Experiment {experiment_id} not found")
            return
        
        print(f"🔍 Analyzing experiment: {results['metadata']['name']}")
        print(f"📊 Generating analysis dashboard...")
        
        # Plot analysis dashboard
        logs = results['logs']
        plot_complete_scbf_dashboard(logs)
        save_all_plots(logs, f"analysis_{experiment_id}")
        
        # Print summary
        summary = results['summary']
        print(f"\n📋 Experiment Summary:")
        print(f"   Total Steps: {summary['total_steps']}")
        print(f"   SCBF Steps: {summary['scbf_analyzed_steps']}")
        print(f"   Analysis Coverage: {summary['analysis_coverage']:.1f}%")
        print(f"   SCBF Success Rate: {summary['scbf_success_rate']:.1f}%")
        
        print(f"\n🔬 Metrics Breakdown:")
        for metric, count in summary['metrics_breakdown'].items():
            print(f"   {metric}: {count} occurrences")

# Global runner instance
_global_runner = SCBFExperimentRunner()

def register_experiment(name: str, experiment_func: Callable, config: Optional[Dict] = None):
    """Register an experiment with the global runner."""
    _global_runner.register_experiment(name, experiment_func, config)

def run_experiment(name: str, **kwargs) -> str:
    """Run a registered experiment."""
    return _global_runner.run_experiment(name, **kwargs)

def list_experiments() -> List[str]:
    """List all registered experiments."""
    return _global_runner.list_experiments()

def analyze_experiment(experiment_id: str) -> None:
    """Analyze a completed experiment."""
    return _global_runner.analyze_experiment(experiment_id)

# Built-in experiment functions
def run_scbf_analysis_step(model, step_idx, x_batch=None, prev_weights=None):
    """Run SCBF analysis for a single adaptation step."""
    try:
        # Import SCBF metrics
        from metrics.entropy_collapse import compute_symbolic_entropy_collapse
        from metrics.activation_ancestry import compute_activation_ancestry  
        from metrics.semantic_attractors import compute_semantic_attractor_density
        from metrics.bifractal_lineage import compute_bifractal_lineage
        
        # Extract model data
        activations = extract_model_activations(model, x_batch)
        
        scbf_results = {}
        
        # Run analysis if we have sufficient activation data
        if activations is not None and activations.shape[0] >= 2:
            
            # Entropy collapse analysis
            try:
                entropy_metrics = compute_symbolic_entropy_collapse(activations)
                scbf_results['entropy_collapse'] = {
                    'magnitude': entropy_metrics['collapse_magnitude'],
                    'rate': entropy_metrics['collapse_rate'], 
                    'symbolic_states': entropy_metrics['symbolic_states']
                }
            except Exception as e:
                print(f"SCBF entropy analysis failed: {e}")
            
            # Activation ancestry (if enough temporal data)
            if activations.shape[0] >= 3:
                try:
                    ancestry_metrics = compute_activation_ancestry(activations)
                    scbf_results['ancestry'] = {
                        'strength': ancestry_metrics['ancestry_strength'],
                        'stability': ancestry_metrics['lineage_stability'],
                        'dimension': ancestry_metrics['bifractal_dimension']
                    }
                except Exception as e:
                    print(f"SCBF ancestry analysis failed: {e}")
            
            # Semantic attractors (if enough samples)
            if activations.shape[0] >= 5:
                try:
                    attractor_metrics = compute_semantic_attractor_density(activations)
                    scbf_results['attractors'] = {
                        'count': attractor_metrics['attractor_count'],
                        'density': attractor_metrics['attractor_density'],
                        'stability': attractor_metrics['attractor_stability']
                    }
                except Exception as e:
                    print(f"SCBF attractor analysis failed: {e}")
        
        # Weight analysis
        try:
            current_weights = []
            for param in model.parameters():
                if hasattr(param, 'detach'):
                    current_weights.append(param.detach().cpu().numpy().flatten())
                else:
                    current_weights.append(param.flatten())
            
            if current_weights:
                import numpy as np
                current_weights = np.concatenate(current_weights)
                lineage_metrics = compute_bifractal_lineage(current_weights, prev_weights)
                scbf_results['lineage'] = {
                    'fractal_dimension': lineage_metrics['fractal_dimension'],
                    'entropy': lineage_metrics['lineage_entropy'],
                    'similarity': lineage_metrics['structural_similarity']
                }
        except Exception as e:
            print(f"SCBF lineage analysis failed: {e}")
        
        return scbf_results
        
    except ImportError:
        print("SCBF modules not available")
        return {}
    except Exception as e:
        print(f"SCBF analysis error: {e}")
        return {}

def extract_model_activations(model, x_batch=None):
    """Extract activations from model for SCBF analysis."""
    try:
        import numpy as np
        
        # Method 1: If model stores activations internally
        if hasattr(model, 'activations_history') and len(model.activations_history) > 0:
            return np.array(model.activations_history)
        
        # Method 2: Extract from hidden state
        if hasattr(model, 'hidden') and model.hidden is not None:
            hidden_np = model.hidden
            if hasattr(hidden_np, 'detach'):
                hidden_np = hidden_np.detach().cpu().numpy()
            if hidden_np.ndim == 1:
                hidden_np = hidden_np.reshape(1, -1)
            return hidden_np
        
        # Method 3: Generate activations from forward pass
        if x_batch is not None and hasattr(model, 'forward'):
            _ = model.forward(x_batch)
            if hasattr(model, 'hidden') and model.hidden is not None:
                activations = model.hidden
                if hasattr(activations, 'detach'):
                    activations = activations.detach().cpu().numpy()
                if activations.ndim == 1:
                    activations = activations.reshape(1, -1)
                return activations
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not extract activations: {e}")
        return None

# Example experiment registration
def example_tinycimm_experiment(logger, model_cls=None, steps=100, **kwargs):
    """Example experiment showing TinyCIMM-Euler integration."""
    import numpy as np
    
    # Mock model for demonstration
    class MockModel:
        def __init__(self):
            self.hidden = np.random.randn(1, 32)
            self.activations_history = []
            
        def forward(self, x):
            self.hidden = np.random.randn(1, 32) * 0.8 + self.hidden * 0.2
            self.activations_history.append(self.hidden[0].copy())
            if len(self.activations_history) > 20:
                self.activations_history = self.activations_history[-20:]
            return self.hidden
            
        def parameters(self):
            return [np.random.randn(32, 32), np.random.randn(32)]
    
    model = MockModel()
    
    print(f"Running example experiment with {steps} steps...")
    
    for step in range(steps):
        # Simulate adaptation step
        x_batch = np.random.randn(1, 32)
        
        # Run SCBF analysis
        scbf_results = run_scbf_analysis_step(model, step, x_batch)
        
        # Log results
        log_entry = {
            'step': step,
            'loss': np.random.exponential(1.0 / (step + 1)),
            'accuracy': min(0.95, 0.5 + step * 0.01),
        }
        
        if scbf_results:
            log_entry['scbf'] = scbf_results
        
        logger.log_step(log_entry)
        
        # Progress update
        if step % 10 == 0:
            print(f"Step {step}: {len(scbf_results)} SCBF metrics computed")
    
    print(f"Completed {steps} steps")
    return {'status': 'completed', 'steps': steps}

# Register the example experiment
register_experiment('example', example_tinycimm_experiment, {'steps': 100})

def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(description='SCBF Experiment Runner')
    parser.add_argument('--experiment', '-e', type=str, help='Name of experiment to run')
    parser.add_argument('--list', action='store_true', help='List available experiments')
    parser.add_argument('--completed', action='store_true', help='List completed experiments')
    parser.add_argument('--analyze', '-a', type=str, help='Analyze completed experiment by ID')
    parser.add_argument('--steps', type=int, default=100, help='Number of adaptation steps')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available experiments:")
        for exp in list_experiments():
            print(f"  - {exp}")
    
    elif args.completed:
        print("Completed experiments:")
        for exp in _global_runner.list_completed_experiments():
            print(f"  - {exp['name']} ({exp['id']}) - {exp['status']}")
    
    elif args.analyze:
        analyze_experiment(args.analyze)
    
    elif args.experiment:
        try:
            run_experiment(args.experiment, steps=args.steps)
        except Exception as e:
            print(f"Error running experiment: {e}")
            return 1
    
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
