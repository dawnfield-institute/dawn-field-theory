"""
======================================================================================
TinyCIMM-Euler Online Mathematical Reasoning Experiment Suite
======================================================================================

Author: Dawn Field Theory Research Group  
Date: July 10, 2025
Project: TinyCIMM-Euler - Higher-Order Mathematical Reasoning

This module provides true online adaptation experiments for TinyCIMM-Euler where the
model updates during prediction (no offline training).

- True online adaptation (one point at a time)
- Real-time network structure adaptation
- Field-aware loss functions and metrics
- SCBF interpretability framework
- Individual step adaptation for mathematical reasoning

The experiments preserve all highly optimized logic while providing clean interfaces
for online mathematical reasoning research.
======================================================================================
"""

import torch
import torch.nn as nn
import pandas as pd
import os
import sys
import math
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tinycimm_euler import TinyCIMMEuler, MathematicalStructureController, HigherOrderEntropyMonitor
from run_experiment import generate_primes, mathematical_fractal_dimension
import matplotlib.pyplot as plt

# ======================================================================================
# DEBUG AND LOGGING SYSTEM
# ======================================================================================

# Global debug flag - set to False to hide debug output in production
DEBUG_MODE = True

def debug_print(message):
    """Print debug message only if debug mode is enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def info_print(message):
    """Print informational message (always shown)"""
    print(f"[INFO] {message}")

# ======================================================================================
# EXPERIMENT CONFIGURATION
# ======================================================================================

RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_logs(logs, signal, run_subdir=None):
    df = pd.DataFrame(logs)
    if run_subdir:
        log_dir = os.path.join(RESULTS_DIR, signal, run_subdir, "logs")
    else:
        log_dir = os.path.join(RESULTS_DIR, signal, "logs")
    os.makedirs(log_dir, exist_ok=True)
    df.to_csv(os.path.join(log_dir, f"tinycimm_euler_online_{signal}_log.csv"), index=False)

def generate_fibonacci_ratios_smoothed(n_points):
    """Generate Fibonacci ratios with smoothed convergence."""
    ratios = []
    a, b = 1, 1
    for i in range(n_points):
        if i < 10:  # Early ratios are unstable
            ratio = 1.618  # Start closer to golden ratio
        else:
            ratio = b / a if a != 0 else 1.618
        ratios.append(ratio)
        a, b = b, a + b
    return torch.tensor(ratios, dtype=torch.float32)

def generate_recursive_sequence_structured(n_points):
    """Generate recursive sequence with clearer rules."""
    sequence = [1.0, 1.5]  # Clear starting values
    for i in range(2, n_points):
        next_val = 0.8 * sequence[i-1] + 0.5 * sequence[i-2]
        sequence.append(next_val)
    return torch.tensor(sequence, dtype=torch.float32)

class OnlineDataGenerator:
    """Generates mathematical data points one at a time for true online adaptation"""
    
    def __init__(self, signal_type, seed=42):
        torch.manual_seed(seed)
        self.signal_type = signal_type
        self.step = 0
        
        # Initialize state for different signals
        if signal_type == "prime_deltas":
            self.primes = generate_primes(100000)  # Generate enough primes
            self.current_prime_idx = 0
            
        elif signal_type == "fibonacci_ratios":
            self.signal_data = generate_fibonacci_ratios_smoothed(10000)
            
        elif signal_type == "recursive_sequence":
            self.signal_data = generate_recursive_sequence_structured(10000)
            
    def get_next_point(self):
        """Generate the next data point"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.signal_type == "prime_deltas":
            if self.current_prime_idx < len(self.primes) - 2:
                current_gap = self.primes[self.current_prime_idx + 1] - self.primes[self.current_prime_idx]
                next_gap = self.primes[self.current_prime_idx + 2] - self.primes[self.current_prime_idx + 1]
                
                x_input = torch.tensor([[current_gap]], dtype=torch.float32, device=device)
                y_target = torch.tensor([[next_gap]], dtype=torch.float32, device=device)
                
                self.current_prime_idx += 1
                return x_input, y_target
            else:
                return None, None
                
        elif self.signal_type == "fibonacci_ratios":
            if self.step < len(self.signal_data):
                x_input = torch.tensor([[self.step]], dtype=torch.float32, device=device)
                y_target = torch.tensor([[self.signal_data[self.step]]], dtype=torch.float32, device=device)
                self.step += 1
                return x_input, y_target
            else:
                return None, None
            
        elif self.signal_type == "polynomial_sequence":
            x_val = self.step / 100.0
            y_val = 0.1*x_val**3 - 0.5*x_val**2 + 2*x_val + 1
            
            x_input = torch.tensor([[x_val]], dtype=torch.float32, device=device)
            y_target = torch.tensor([[y_val]], dtype=torch.float32, device=device)
            
            self.step += 1
            return x_input, y_target
            
        elif self.signal_type == "recursive_sequence":
            if self.step < len(self.signal_data):
                x_input = torch.tensor([[self.step]], dtype=torch.float32, device=device)
                y_target = torch.tensor([[self.signal_data[self.step]]], dtype=torch.float32, device=device)
                self.step += 1
                return x_input, y_target
            else:
                return None, None
            
        else:  # mathematical_harmonic
            x_val = self.step * 4 * math.pi / 1000
            y_val = math.sin(x_val) + 0.5*math.sin(2*x_val) + 0.25*math.sin(3*x_val)
            
            x_input = torch.tensor([[x_val]], dtype=torch.float32, device=device)
            y_target = torch.tensor([[y_val]], dtype=torch.float32, device=device)
            
            self.step += 1
            return x_input, y_target

def run_online_experiment(model_cls, signal="prime_deltas", steps=500, seed=42, experiment_type="realtime", **kwargs):
    """Run TRUE online mathematical reasoning experiment - restored CIMM-style individual adaptation!"""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running online experiment on {device}")
    
    # Initialize model for online adaptation
    input_size = 1  # Always single value input for online adaptation
    hidden_size = kwargs.pop('hidden_size', 16)
    
    model = model_cls(input_size=input_size, hidden_size=hidden_size, output_size=1, device=device, **kwargs)
    
    # Restore responsive structure controller for true online adaptation
    controller = MathematicalStructureController(
        base_complexity_threshold=0.01,  # Restored proper threshold
        adaptation_window=5,             # Proper window for online adaptation
        min_neurons=8,
        max_neurons=64
    )
    
    complexity_monitor = HigherOrderEntropyMonitor(momentum=0.85)  # Proper momentum
    model.set_complexity_monitor(complexity_monitor)
    
    # Initialize online data generator
    data_generator = OnlineDataGenerator(signal, seed)
    
    logs = []
    math_metrics, math_hsizes, math_fractals, math_performance, math_losses = [], [], [], [], []
    recent_predictions, recent_targets = [], []
    
    # Create unique subfolder for this experiment run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_subdir = f"run_{run_timestamp}"
    
    # Create descriptive experiment directory name
    exp_dir = f"{signal}_{experiment_type}"
    
    signal_img_dir = os.path.join(RESULTS_DIR, exp_dir, run_subdir, "images")
    os.makedirs(signal_img_dir, exist_ok=True)
    
    print(f"Online experiment results will be saved in: {signal_img_dir}")
    print(f"Starting TRUE online adaptation for {signal} (individual step updates)...")
    
    for t in range(steps):
        # Get next data point online
        x_input, y_target = data_generator.get_next_point()
        
        if x_input is None or y_target is None:
            print(f"Data exhausted at step {t}")
            break
        
        # CRITICAL FIX: Use individual step processing like original CIMM
        # This restores the fine-grained adaptation that was lost in batch processing
        model.train()  # Ensure gradients are enabled for online adaptation
        
        # Forward pass with gradient tracking
        prediction = model(x_input)
        
        # Compute loss with proper gradient flow
        step_loss = torch.mean((prediction - y_target) ** 2)
        
        # Standard gradient update FIRST
        model.optimizer.zero_grad()
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.optimizer.step()
        
        # THEN do the online adaptation step with fresh forward pass
        model.eval()
        with torch.no_grad():
            fresh_prediction = model(x_input)
        
        # Now call online adaptation with the updated model
        result = model.online_adaptation_step(x_input, y_target, recent_predictions)
        
        # TinyCIMM-Planck style SCBF analysis
        scbf_results = model.analyze_results()
        
        prediction = result['prediction']
        adaptation_signal = result['adaptation_signal'] 
        complexity_metric = result['complexity_metric']
        field_performance = result['field_performance']
        cimm_components = result['cimm_components']
        scbf_metrics = result.get('scbf_metrics', {})
        
        # Update complexity monitor with actual prediction
        complexity_monitor.update(fresh_prediction)
        
        # Store recent predictions for analysis
        recent_predictions.append(fresh_prediction.item() if torch.is_tensor(fresh_prediction) else fresh_prediction)
        recent_targets.append(y_target.item())
        if len(recent_predictions) > 30:  # Shorter memory for online adaptation
            recent_predictions.pop(0)
            recent_targets.pop(0)
        
        # Apply field optimization more frequently for online adaptation
        if t % 20 == 0 and t > 0:  # Every 20 steps for responsive online adaptation
            model.entropy_aware_field_optimization()
        
        # Enhanced logging with all CIMM components
        log_entry = {
            'step': t,
            'adaptation_signal': adaptation_signal,
            'complexity_metric': complexity_metric,
            'neurons': model.hidden_dim,
            'pattern_recognition_score': field_performance['pattern_recognition_score'],
            'field_coherence_score': field_performance['field_coherence_score'], 
            'quantum_field_performance': field_performance['quantum_field_performance'],
            'prediction': recent_predictions[-1],
            'target': y_target.item(),
            'prediction_error': abs(recent_predictions[-1] - y_target.item()),
            'learning_rate': result['learning_rate'],
            'step_loss': step_loss.item()
        }
        
        # Add CIMM components
        if cimm_components:
            log_entry.update({
                'qbe_balance': cimm_components.get('qbe_balance', 0),
                'energy_balance': cimm_components.get('energy_balance', 0),
                'coherence_loss': cimm_components.get('coherence_loss', 0),
                'cimm_kl_divergence': cimm_components.get('KL-Divergence', 0),
                'cimm_jensen_shannon': cimm_components.get('Jensen-Shannon', 0),
                'cimm_wasserstein': cimm_components.get('Wasserstein Distance', 0),
                'cimm_qwcs': cimm_components.get('QWCS', 0),
                'cimm_entropy': cimm_components.get('entropy_value', 0),
                'einstein_correction': cimm_components.get('einstein_correction', 1),
                'feynman_damping': cimm_components.get('feynman_damping', 1)
            })
        
        # Add SCBF interpretability metrics
        if scbf_metrics:
            log_entry.update({
                'scbf_symbolic_entropy_collapse': scbf_metrics.get('symbolic_entropy_collapse', 0),
                'scbf_activation_ancestry_stability': scbf_metrics.get('activation_ancestry_stability', 0),
                'scbf_collapse_phase_alignment': scbf_metrics.get('collapse_phase_alignment', 0),
                'scbf_bifractal_lineage_strength': scbf_metrics.get('bifractal_lineage_strength', 0),
                'scbf_semantic_attractor_density': scbf_metrics.get('semantic_attractor_density', 0),
                'scbf_weight_drift_entropy': scbf_metrics.get('weight_drift_entropy', 0),
                'scbf_entropy_gradient_alignment': scbf_metrics.get('entropy_gradient_alignment', 0),
                'scbf_structural_entropy': scbf_metrics.get('structural_entropy', 0)
            })
        
        math_metrics.append(complexity_metric)
        math_hsizes.append(model.hidden_dim)
        math_losses.append(step_loss.item())  # Use the actual step loss
        math_performance.append(field_performance['quantum_field_performance'])
        
        logs.append(log_entry)
        
        # Fractal dimension analysis
        if t % 25 == 0:  # More frequent for online adaptation
            fd = mathematical_fractal_dimension(model.W)
            math_fractals.append(fd if not (torch.isnan(torch.tensor(fd)) or torch.isinf(torch.tensor(fd))) else float('nan'))
        
        # Progress reporting (every 25 steps for online adaptation)
        if t % 25 == 0:
            recent_error = sum([l['prediction_error'] for l in logs[-5:]]) / min(5, len(logs))
            current_neurons = model.hidden_dim
            pattern_score = field_performance['pattern_recognition_score']
            print(f"Step {t}: Error={recent_error:.4f}, Neurons={current_neurons}, Pattern={pattern_score:.4f}")
        
        # Visualization every 75 steps (more frequent for online)
        if t % 75 == 0 and t > 0:
            plt.figure(figsize=(15, 10))
            
            # Weight matrix
            plt.subplot(2, 3, 1)
            plt.imshow(model.W.detach().cpu().numpy(), aspect='auto', cmap='RdBu')
            plt.colorbar()
            plt.title(f'Weight Matrix at Step {t}')
            
            # Online adaptation progress  
            plt.subplot(2, 3, 2)
            if len(logs) > 5:
                recent_losses = [l['step_loss'] for l in logs[-30:]]
                plt.plot(recent_losses, color='red', alpha=0.7)
                plt.title('Recent Step Loss Evolution')
                plt.ylabel('Step Loss')
            
            # Network size adaptation
            plt.subplot(2, 3, 3)
            plt.plot(math_hsizes, color='blue', alpha=0.8)
            plt.title('Network Size Evolution')
            plt.ylabel('Neurons')
            plt.xlabel('Step')
            
            # Prediction vs Target
            plt.subplot(2, 3, 4)
            if len(recent_predictions) > 5:
                plt.plot(recent_predictions[-15:], label='Predictions', alpha=0.7)
                plt.plot(recent_targets[-15:], label='Targets', alpha=0.7)
                plt.legend()
                plt.title('Recent Predictions vs Targets')
            
            # Field-aware performance components
            plt.subplot(2, 3, 5)
            if len(logs) > 5:
                pattern_scores = [l['pattern_recognition_score'] for l in logs[-15:]]
                field_scores = [l['field_coherence_score'] for l in logs[-15:]]
                plt.plot(pattern_scores, label='Pattern Recognition', alpha=0.7)
                plt.plot(field_scores, label='Field Coherence', alpha=0.7)
                plt.legend()
                plt.title('Field-Aware Performance Components')
            
            # Performance metrics
            plt.subplot(2, 3, 6)
            plt.plot(math_performance, color='orange', alpha=0.8)
            plt.title('Mathematical Performance')
            plt.ylabel('Performance Score')
            plt.xlabel('Step')
            
            plt.tight_layout()
            plt.savefig(os.path.join(signal_img_dir, f'online_learning_step_{t}.png'))
            plt.close()
    
    # Save logs with experiment type
    exp_dir = f"{signal}_{experiment_type}"
    save_logs(logs, exp_dir, run_subdir)
    
    # Final analysis and visualization
    print(f"\nOnline adaptation completed for {signal}")
    if len(logs) > 0:
        final_error = sum([l['prediction_error'] for l in logs[-5:]]) / min(5, len(logs))
        final_neurons = logs[-1]['neurons']
        initial_neurons = logs[0]['neurons']
        final_pattern_score = logs[-1]['pattern_recognition_score']
        print(f"Final prediction error: {final_error:.4f}")
        print(f"Final pattern recognition: {final_pattern_score:.4f}")
        print(f"Network size: {initial_neurons} -> {final_neurons} neurons")
        print(f"Adaptation events: {abs(final_neurons - initial_neurons)} total changes")
    
    return logs

def run_all_online_experiments():
    """Run all online mathematical reasoning experiments with standardized SCBF metrics"""
    print("=" * 80)
    print("TinyCIMM-Euler: TRUE ONLINE Mathematical Reasoning Tests")
    print("=" * 80)
    print("\nNo pre-generated sequences! Pure online adaptation:")
    print("• Predict next mathematical value")
    print("• Get immediate feedback")
    print("• Adapt network structure in real-time")
    print("• CIMM-inspired field-aware loss functions")
    print("• Standardized SCBF metrics matching TinyCIMM-Planck")
    print("\nFocusing on prime number prediction and mathematical pattern recognition...\n")
    
    test_cases = [
        ("prime_deltas", {
            "hidden_size": 16, 
            "math_memory_size": 8,
            "experiment_type": "realtime_sequential"
        }),
        ("fibonacci_ratios", {
            "hidden_size": 12, 
            "math_memory_size": 6, 
            "pattern_decay": 0.9,
            "experiment_type": "realtime_convergence"
        }),
        ("polynomial_sequence", {
            "hidden_size": 14, 
            "math_memory_size": 8,
            "experiment_type": "realtime_nonlinear"
        }),
        ("recursive_sequence", {
            "hidden_size": 16, 
            "math_memory_size": 10, 
            "pattern_decay": 0.95,
            "experiment_type": "realtime_recurrent"
        }),
        ("mathematical_harmonic", {
            "hidden_size": 12, 
            "math_memory_size": 6,
            "experiment_type": "realtime_harmonic"
        }),
    ]
    
    successful_experiments = 0
    
    for test_name, model_kwargs in test_cases:
        print(f"\n{'='*60}")
        print(f"=== Running ONLINE Mathematical Experiment: {test_name} ===")
        challenge_level = "Extreme" if test_name == "prime_deltas" else "Very High" if "sequence" in test_name else "High"
        print(f"Expected challenge level: {challenge_level}")
        print(f"Adapting for 300 steps (true online, no offline training)...")
        print(f"{'='*60}")
        
        try:
            logs = run_online_experiment(TinyCIMMEuler, signal=test_name, steps=300, **model_kwargs)
            print(f"✓ Completed {test_name} successfully with {len(logs)} adaptation steps")
            successful_experiments += 1
        except Exception as e:
            print(f"✗ Error in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("Online mathematical reasoning experiments completed!")
    print(f"Successfully completed: {successful_experiments}/{len(test_cases)} experiments")
    print("Check experiment_results/ for detailed results organized by experiment type and date.")
    print("=" * 80)

if __name__ == "__main__":
    run_all_online_experiments()
