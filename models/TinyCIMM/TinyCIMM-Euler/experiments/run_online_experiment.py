import torch
import torch.nn as nn
import pandas as pd
import os
import sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tinycimm_euler import TinyCIMMEuler, MathematicalStructureController, HigherOrderEntropyMonitor, CIMMInspiredLoss
import matplotlib.pyplot as plt

IMG_DIR = "experiment_images"
os.makedirs(IMG_DIR, exist_ok=True)

def generate_primes(limit):
    """Generate prime numbers up to limit using Sieve of Eratosthenes"""
    if limit < 2:
        return []
    
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    
    return [i for i in range(2, limit + 1) if sieve[i]]

def mathematical_fractal_dimension(weights):
    """Compute fractal dimension for mathematical weight patterns"""
    if isinstance(weights, torch.Tensor):
        W = weights.detach().abs() > 1e-6
    else:
        W = torch.tensor(weights).abs() > 1e-6
    
    if W.ndim != 2 or min(W.shape) < 4:
        return float('nan')
    if not torch.any(W):
        return float('nan')
    
    min_size = min(W.shape) // 2 + 1
    sizes = torch.arange(2, min_size)
    counts = []
    
    for size in sizes:
        count = 0
        for i in range(0, W.shape[0], int(size)):
            for j in range(0, W.shape[1], int(size)):
                if torch.any(W[i:i+int(size), j:j+int(size)]):
                    count += 1
        if count > 0:
            counts.append(count)
    
    if len(counts) > 1:
        sizes_log = torch.log(sizes[:len(counts)].float())
        counts_log = torch.log(torch.tensor(counts, dtype=torch.float))
        coeffs = torch.linalg.lstsq(sizes_log.unsqueeze(1), counts_log).solution
        return -coeffs[0].item()
    else:
        return float('nan')

def save_logs(logs, signal):
    df = pd.DataFrame(logs)
    os.makedirs("experiment_logs", exist_ok=True)
    df.to_csv(f"experiment_logs/tinycimm_euler_online_{signal}_log.csv", index=False)

class OnlineDataGenerator:
    """Generates mathematical data points one at a time for true online learning"""
    
    def __init__(self, signal_type, seed=42):
        torch.manual_seed(seed)
        self.signal_type = signal_type
        self.step = 0
        
        # Initialize state for different signals
        if signal_type == "prime_deltas":
            self.primes = generate_primes(100000)  # Generate enough primes
            self.current_prime_idx = 0
            
        elif signal_type == "fibonacci_ratios":
            self.fib_prev, self.fib_curr = 1, 1
            
        elif signal_type == "recursive_sequence":
            self.seq_values = [1.0, 1.0]
            
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
            fib_next = self.fib_prev + self.fib_curr
            current_ratio = fib_next / self.fib_curr if self.fib_curr != 0 else 1.0
            
            x_input = torch.tensor([[self.step]], dtype=torch.float32, device=device)
            y_target = torch.tensor([[current_ratio]], dtype=torch.float32, device=device)
            
            self.fib_prev, self.fib_curr = self.fib_curr, fib_next
            self.step += 1
            return x_input, y_target
            
        elif self.signal_type == "polynomial_sequence":
            x_val = self.step / 100.0
            y_val = 0.1*x_val**3 - 0.5*x_val**2 + 2*x_val + 1
            
            x_input = torch.tensor([[x_val]], dtype=torch.float32, device=device)
            y_target = torch.tensor([[y_val]], dtype=torch.float32, device=device)
            
            self.step += 1
            return x_input, y_target
            
        elif self.signal_type == "recursive_sequence":
            if len(self.seq_values) >= 2:
                next_val = 0.7*self.seq_values[-1] + 0.3*self.seq_values[-2] + 0.1*math.sin(self.step/10.0)
                self.seq_values.append(next_val)
            else:
                next_val = 1.0
                
            x_input = torch.tensor([[self.step]], dtype=torch.float32, device=device)
            y_target = torch.tensor([[next_val]], dtype=torch.float32, device=device)
            
            self.step += 1
            return x_input, y_target
            
        else:  # mathematical_harmonic
            x_val = self.step * 4 * math.pi / 1000
            y_val = math.sin(x_val) + 0.5*math.sin(2*x_val) + 0.25*math.sin(3*x_val)
            
            x_input = torch.tensor([[x_val]], dtype=torch.float32, device=device)
            y_target = torch.tensor([[y_val]], dtype=torch.float32, device=device)
            
            self.step += 1
            return x_input, y_target

def run_online_experiment(model_cls, signal="prime_deltas", steps=500, seed=42, **kwargs):
    """Run TRUE online mathematical reasoning experiment - no pre-training sequences!"""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running online experiment on {device}")
    
    # Initialize model for online learning
    input_size = 1  # Always single value input for online learning
    hidden_size = kwargs.pop('hidden_size', 16)
    
    model = model_cls(input_size=input_size, hidden_size=hidden_size, output_size=1, device=device, **kwargs)
    
    # More responsive structure controller for online learning
    controller = MathematicalStructureController(
        complexity_threshold=0.005,  # Lower threshold for more responsive adaptation
        adaptation_window=3,         # Shorter window for faster response
        min_neurons=8,
        max_neurons=64
    )
    controller.cooldown_period = 5   # Shorter cooldown for online learning
    
    complexity_monitor = HigherOrderEntropyMonitor(momentum=0.8)  # More responsive
    model.set_complexity_monitor(complexity_monitor)
    
    # Initialize online data generator
    data_generator = OnlineDataGenerator(signal, seed)
    
    logs = []
    math_metrics, math_hsizes, math_fractals, math_performance, math_losses = [], [], [], [], []
    recent_predictions, recent_targets = [], []
    
    # Create signal-specific subfolder
    signal_img_dir = os.path.join(IMG_DIR, f"online_{signal}")
    os.makedirs(signal_img_dir, exist_ok=True)
    
    print(f"Starting online learning for {signal}...")
    
    for t in range(steps):
        # Get next data point online
        x_input, y_target = data_generator.get_next_point()
        
        if x_input is None or y_target is None:
            print(f"Data exhausted at step {t}")
            break
        
        # Make prediction
        prediction = model(x_input)
        
        # Learn from feedback (this is the key - immediate learning from each prediction)
        loss, complexity_metric, performance, loss_components = model.mathematical_train_step(x_input, y_target)
        
        # Update complexity monitor
        complexity_monitor.update(prediction)
        
        # Store recent predictions for analysis
        recent_predictions.append(prediction.item())
        recent_targets.append(y_target.item())
        if len(recent_predictions) > 50:
            recent_predictions.pop(0)
            recent_targets.pop(0)
        
        # Apply entropy-aware pruning periodically
        if t % 30 == 0 and t > 0:  # More frequent for online learning
            model.entropy_aware_pruning()
        
        # Logging with field-aware loss components
        log_entry = {
            'step': t,
            'loss': loss,
            'complexity_metric': complexity_metric,
            'neurons': model.hidden_dim,
            'performance': performance,
            'prediction': prediction.item(),
            'target': y_target.item(),
            'prediction_error': abs(prediction.item() - y_target.item()),
            'qbe_loss': loss_components['qbe_loss'],
            'energy_balance': loss_components['energy_balance'],
            'superfluid_coherence': loss_components['superfluid_coherence'],
            'entropy_cost': loss_components['entropy_cost']
        }
        
        math_metrics.append(complexity_metric)
        math_hsizes.append(model.hidden_dim)
        math_losses.append(loss)
        math_performance.append(performance)
        
        # Mathematical analysis every 10 steps
        if t % 10 == 0:
            analysis_results = model.analyze_mathematical_results()
            log_entry.update(analysis_results)
        
        logs.append(log_entry)
        
        # Fractal dimension analysis
        if t % 15 == 0:
            fd = mathematical_fractal_dimension(model.W)
            math_fractals.append(fd if not (torch.isnan(torch.tensor(fd)) or torch.isinf(torch.tensor(fd))) else float('nan'))
        
        # Progress reporting
        if t % 50 == 0:
            recent_error = sum([l['prediction_error'] for l in logs[-10:]]) / min(10, len(logs))
            current_neurons = model.hidden_dim
            print(f"Step {t}: Error={recent_error:.4f}, Neurons={current_neurons}, Loss={loss:.4f}")
        
        # Visualization every 100 steps
        if t % 100 == 0 and t > 0:
            # Weight evolution
            plt.figure(figsize=(15, 10))
            
            # Subplot 1: Weight matrix
            plt.subplot(2, 3, 1)
            plt.imshow(model.W.detach().cpu().numpy(), aspect='auto', cmap='RdBu')
            plt.colorbar()
            plt.title(f'Weight Matrix at Step {t}')
            
            # Subplot 2: Online learning progress
            plt.subplot(2, 3, 2)
            if len(logs) > 10:
                recent_losses = [l['loss'] for l in logs[-50:]]
                plt.plot(recent_losses, color='red', alpha=0.7)
                plt.title('Recent Loss Evolution')
                plt.ylabel('Loss')
            
            # Subplot 3: Network size adaptation
            plt.subplot(2, 3, 3)
            plt.plot(math_hsizes, color='blue', alpha=0.8)
            plt.title('Network Size Evolution')
            plt.ylabel('Neurons')
            plt.xlabel('Step')
            
            # Subplot 4: Prediction vs Target
            plt.subplot(2, 3, 4)
            if len(recent_predictions) > 10:
                plt.plot(recent_predictions[-20:], label='Predictions', alpha=0.7)
                plt.plot(recent_targets[-20:], label='Targets', alpha=0.7)
                plt.legend()
                plt.title('Recent Predictions vs Targets')
            
            # Subplot 5: Field-aware loss components
            plt.subplot(2, 3, 5)
            if len(logs) > 10:
                qbe_losses = [l['qbe_loss'] for l in logs[-20:]]
                energy_balance = [l['energy_balance'] for l in logs[-20:]]
                plt.plot(qbe_losses, label='QBE Loss', alpha=0.7)
                plt.plot(energy_balance, label='Energy Balance', alpha=0.7)
                plt.legend()
                plt.title('Field-Aware Loss Components')
            
            # Subplot 6: Performance metrics
            plt.subplot(2, 3, 6)
            plt.plot(math_performance, color='orange', alpha=0.8)
            plt.title('Mathematical Performance')
            plt.ylabel('Performance Score')
            plt.xlabel('Step')
            
            plt.tight_layout()
            plt.savefig(os.path.join(signal_img_dir, f'online_learning_step_{t}.png'))
            plt.close()
    
    # Save logs
    save_logs(logs, signal)
    
    # Final analysis and visualization
    print(f"\nOnline learning completed for {signal}")
    if len(logs) > 0:
        final_error = sum([l['prediction_error'] for l in logs[-10:]]) / min(10, len(logs))
        final_neurons = logs[-1]['neurons']
        initial_neurons = logs[0]['neurons']
        print(f"Final prediction error: {final_error:.4f}")
        print(f"Network size: {initial_neurons} -> {final_neurons} neurons")
        print(f"Adaptation events: {abs(final_neurons - initial_neurons)} total changes")
    
    return logs

def run_all_online_experiments():
    """Run all online mathematical reasoning experiments"""
    test_cases = [
        ("prime_deltas", {
            "hidden_size": 16, 
            "math_memory_size": 8, 
            "adaptation_steps": 15
        }),
        ("fibonacci_ratios", {
            "hidden_size": 12, 
            "math_memory_size": 6, 
            "pattern_decay": 0.9
        }),
        ("polynomial_sequence", {
            "hidden_size": 14, 
            "math_memory_size": 8, 
            "adaptation_steps": 12
        }),
        ("recursive_sequence", {
            "hidden_size": 16, 
            "math_memory_size": 10, 
            "pattern_decay": 0.95
        }),
        ("mathematical_harmonic", {
            "hidden_size": 12, 
            "math_memory_size": 6
        }),
    ]
    
    for test_name, model_kwargs in test_cases:
        print(f"\n{'='*60}")
        print(f"Running ONLINE Mathematical Experiment: {test_name}")
        print(f"Expected challenge level: {'Very High' if test_name == 'prime_deltas' else 'High' if 'sequence' in test_name else 'Medium'}")
        print(f"{'='*60}")
        
        try:
            logs = run_online_experiment(TinyCIMMEuler, signal=test_name, steps=300, **model_kwargs)
            print(f"✓ Completed {test_name} successfully with {len(logs)} learning steps")
        except Exception as e:
            print(f"✗ Error in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    print("=" * 80)
    print("TinyCIMM-Euler: TRUE ONLINE Mathematical Reasoning Tests")
    print("=" * 80)
    print("\nNo pre-training sequences! Pure online learning:")
    print("• Predict next mathematical value")
    print("• Get immediate feedback")
    print("• Adapt network structure in real-time")
    print("• CIMM-inspired field-aware loss functions")
    print("\nFocusing on prime number prediction and mathematical pattern recognition...\n")
    
    run_all_online_experiments()
    
    print("\n" + "=" * 80)
    print("Online mathematical reasoning experiments completed!")
    print("Check experiment_images/online_* and experiment_logs/ for detailed results.")
    print("=" * 80)
