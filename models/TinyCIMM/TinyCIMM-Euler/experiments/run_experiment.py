import torch
import torch.nn as nn
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tinycimm_euler import TinyCIMMEuler, MathematicalStructureController, HigherOrderEntropyMonitor, compute_mathematical_coherence, CIMMInspiredLoss
import matplotlib.pyplot as plt
import math

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

def get_prime_deltas(num_primes=5000):
    """Get prime number deltas for sequence prediction - increased for longer sequences"""
    primes = generate_primes(num_primes * 20)  # Generate much more primes
    if len(primes) < num_primes:
        primes = generate_primes(num_primes * 30)
    
    primes = primes[:num_primes]
    deltas = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
    return deltas

def create_sequences(deltas, sequence_length=4, prediction_horizon=1):
    """Create sequences for prime delta prediction"""
    X, y = [], []
    for i in range(len(deltas) - sequence_length - prediction_horizon + 1):
        X.append(deltas[i:i+sequence_length])
        y.append(deltas[i+sequence_length:i+sequence_length+prediction_horizon])
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_signal(signal_type, steps, seed=42):
    """Generate test signals for mathematical reasoning"""
    torch.manual_seed(seed)
    
    if signal_type == "prime_deltas":
        # Generate prime delta sequences - much longer for deep mathematical learning
        deltas = get_prime_deltas(steps + 500)
        x_seq, y_seq = create_sequences(deltas[:steps], sequence_length=4)
        return x_seq, y_seq.squeeze()
    
    elif signal_type == "fibonacci_ratios":
        # Fibonacci sequence and golden ratio convergence - properly limited to steps
        fib = [1, 1]
        for i in range(steps + 10):  # Generate extra to ensure we have enough
            fib.append(fib[-1] + fib[-2])
        
        # Calculate ratios (should converge to golden ratio ~1.618)
        ratios = []
        for i in range(1, len(fib)-1):
            if fib[i] != 0:  # Avoid division by zero
                ratios.append(fib[i+1]/fib[i])
            if len(ratios) >= steps:
                break
        
        # Ensure we have exactly 'steps' number of ratios
        if len(ratios) < steps:
            # Fill with golden ratio if we don't have enough
            golden_ratio = 1.618033988749895
            while len(ratios) < steps:
                ratios.append(golden_ratio)
        else:
            ratios = ratios[:steps]
        
        print(f"Fibonacci ratios range: {min(ratios):.6f} to {max(ratios):.6f}")
        print(f"Final few ratios: {ratios[-5:]}")
        
        x = torch.arange(len(ratios)).float().unsqueeze(1)
        y = torch.tensor(ratios, dtype=torch.float32)
        return x, y
    
    elif signal_type == "polynomial_sequence":
        # Higher-order polynomial sequence
        x = torch.linspace(0, 4, steps).unsqueeze(1)
        y = 0.1*x.squeeze()**3 - 0.5*x.squeeze()**2 + 2*x.squeeze() + 1
        return x, y
    
    elif signal_type == "recursive_sequence":
        # Mathematical recursive sequence
        seq = [1, 1]
        for i in range(steps-2):
            next_val = 0.7*seq[-1] + 0.3*seq[-2] + 0.1*torch.sin(torch.tensor(i/10.0)).item()
            seq.append(next_val)
        x = torch.arange(len(seq)).float().unsqueeze(1)
        y = torch.tensor(seq, dtype=torch.float32)
        return x, y
    
    else:  # mathematical_harmonic
        x = torch.linspace(0, 4*math.pi, steps).unsqueeze(1)
        # Complex harmonic with mathematical progression
        y = torch.sin(x.squeeze()) + 0.5*torch.sin(2*x.squeeze()) + 0.25*torch.sin(3*x.squeeze())
        return x, y

def compute_field_adaptation_signal(yhat, y_true):
    """Compute field-aware adaptation signal instead of loss"""
    prediction_error = torch.mean((yhat - y_true) ** 2)
    return prediction_error

def compute_mathematical_entropy(model):
    return model.log_complexity_metric()

def save_logs(logs, signal):
    df = pd.DataFrame(logs)
    os.makedirs("experiment_logs", exist_ok=True)
    df.to_csv(f"experiment_logs/tinycimm_euler_{signal}_log.csv", index=False)

def mathematical_fractal_dimension(weights):
    """Compute fractal dimension for mathematical weight patterns"""
    if isinstance(weights, torch.Tensor):
        W = weights.detach().abs() > 1e-6  # More sensitive threshold for math
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

def run_experiment(model_cls, signal="prime_deltas", steps=10000, seed=42, **kwargs):
    """Run long-term online mathematical reasoning experiment (CIMM-style prediction and feedback)"""
    x, y = get_signal(signal, steps, seed)
    device = x.device
    
    # Adjust input size based on signal type
    if signal == "prime_deltas":
        input_size = x.shape[1] if len(x.shape) > 1 else 1
        hidden_size = kwargs.pop('hidden_size', 16)  # Larger for mathematical reasoning
    else:
        input_size = x.shape[1] if len(x.shape) > 1 else 1
        hidden_size = kwargs.pop('hidden_size', 12)
    
    model = model_cls(input_size=input_size, hidden_size=hidden_size, output_size=1, device=device, **kwargs)
    controller = MathematicalStructureController()
    complexity_monitor = HigherOrderEntropyMonitor(momentum=0.9)
    model.set_complexity_monitor(complexity_monitor)
    
    logs = []
    math_metrics, math_hsizes, math_fractals, math_performance, math_losses = [], [], [], [], []
    math_raw_preds, math_smoothed_preds = [], []

    # Create signal-specific subfolder
    signal_img_dir = os.path.join(IMG_DIR, signal)
    os.makedirs(signal_img_dir, exist_ok=True)

    for t in range(min(steps, len(x))):
        # Progress tracking for long experiments
        if t % 1000 == 0 and t > 0:
            progress_pct = (t / steps) * 100
            print(f"Progress: {t}/{steps} steps ({progress_pct:.1f}%) - Current neurons: {model.hidden_dim}")
        
        # Handle different input shapes
        if signal == "prime_deltas":
            x_input = x[t:t+1] if t < len(x) else x[-1:] 
            y_target = y[t:t+1] if t < len(y) else y[-1:]
        else:
            x_input = x[t:t+1] if t < len(x) else x[-1:]
            y_target = y[t:t+1] if t < len(y) else y[-1:]
        
        # Make prediction first (like CIMM's run method)
        with torch.no_grad():
            prediction = model(x_input)
        
        # Store prediction and actual for analysis
        math_raw_preds.append(prediction.detach().cpu().numpy().flatten())
        
        # Give feedback and adapt online (like CIMM's give_feedback method)
        result = model.online_adaptation_step(x_input, y_target)
        
        # Update complexity monitor
        complexity_monitor.update(prediction)
        
        # Apply field optimization periodically (more frequently for longer experiments)
        if t % 100 == 0 and t > 0:
            model.entropy_aware_field_optimization()
        
        # Debug output every 100 steps
        if t % 100 == 0:
            field_performance = result['field_performance']
            print(f"Step {t}: Pattern Recognition = {field_performance['pattern_recognition_score']:.4f}, "
                  f"Adaptation Signal = {result['adaptation_signal']:.4f}, "
                  f"Neurons = {model.hidden_dim}")
        
        # Extract field-aware performance metrics
        field_performance = result['field_performance']
        
        # Logging with field-aware metrics instead of loss-based metrics
        log_entry = {
            'step': t,
            'adaptation_signal': result['adaptation_signal'],
            'complexity_metric': result['complexity_metric'],
            'neurons': model.hidden_dim,
            'pattern_recognition_score': field_performance['pattern_recognition_score'],
            'field_coherence_score': field_performance['field_coherence_score'],
            'structural_complexity_score': field_performance['structural_complexity_score'],
            'higher_order_dynamics': field_performance['higher_order_dynamics'],
            'mathematical_reasoning_consistency': field_performance['mathematical_reasoning_consistency'],
            'quantum_field_performance': field_performance['quantum_field_performance'],
            'learning_rate': result['learning_rate']
        }
        
        # Add CIMM components if available
        if 'cimm_components' in result and result['cimm_components']:
            cimm_components = result['cimm_components']
            log_entry.update({
                'qbe_balance': cimm_components.get('qbe_balance', 0),
                'energy_balance': cimm_components.get('energy_balance', 0),
                'coherence_loss': cimm_components.get('coherence_loss', 0),
                'einstein_correction': cimm_components.get('einstein_correction', 1),
                'feynman_damping': cimm_components.get('feynman_damping', 1),
                'cimm_kl_divergence': cimm_components.get('cimm_kl_divergence', 0),
                'cimm_jensen_shannon': cimm_components.get('cimm_jensen_shannon', 0),
                'cimm_wasserstein': cimm_components.get('cimm_wasserstein', 0),
                'cimm_qwcs': cimm_components.get('cimm_qwcs', 0),
                'cimm_entropy': cimm_components.get('cimm_entropy', 0),
                'prediction': prediction.detach().cpu().numpy().flatten()[0],
                'actual': y_target.detach().cpu().numpy().flatten()[0]
            })
        
        logs.append(log_entry)
        
        # Store field-aware metrics instead of traditional loss-based metrics
        math_metrics.append(result['complexity_metric'])
        math_hsizes.append(model.hidden_dim)
        
        # Store adaptation signal as "prediction error" (but this is field-aware, not MSE-based)
        adaptation_signal = result['adaptation_signal']
        math_losses.append(adaptation_signal.item() if hasattr(adaptation_signal, 'item') else adaptation_signal)
        
        # Store field performance score
        overall_field_performance = sum(field_performance.values()) / len(field_performance)
        math_performance.append(overall_field_performance)
        
        # Mathematical analysis
        if t % 10 == 0:
            analysis_results = model.analyze_mathematical_results()
            logs[-1].update(analysis_results)
        
        # Fractal dimension analysis
        if t % 15 == 0:
            fd = mathematical_fractal_dimension(model.W)
            math_fractals.append(fd if not (torch.isnan(torch.tensor(fd)) or torch.isinf(torch.tensor(fd))) else float('nan'))
        
        # Visualization (less frequent for longer experiments)
        if t % 2000 == 0 and t > 0:
            # Weight visualization
            plt.figure(figsize=(12, 8))
            plt.imshow(model.W.detach().cpu().numpy(), aspect='auto', cmap='RdBu')
            plt.colorbar()
            plt.title(f'TinyCIMM-Euler Mathematical Weights at step {t}')
            plt.tight_layout()
            plt.savefig(os.path.join(signal_img_dir, f'math_weights_step_{t}.png'))
            plt.close()
        
        # Mathematical memory analysis (less frequent)
        if t % 1000 == 0 and len(model.math_memory) >= 2:
            # Handle different tensor sizes due to network growth
            try:
                memory_norms = [torch.norm(mem).item() for mem in model.math_memory]
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(model.math_memory)), memory_norms)
                plt.title(f'Mathematical Memory Pattern Evolution (step {t})')
                plt.xlabel('Memory Index')
                plt.ylabel('Pattern Magnitude')
                plt.tight_layout()
                plt.savefig(os.path.join(signal_img_dir, f'math_memory_evolution_{t}.png'))
                plt.close()
            except Exception as e:
                print(f"Memory analysis skipped at step {t}: {e}")
                pass
        
        # Store predictions (already done above)
        # math_raw_preds.append(prediction.detach().cpu().numpy().flatten())
    
    # Save logs
    save_logs(logs, signal)
    
    # Final visualizations
    plt.figure(figsize=(15, 10))
    
    if signal == "prime_deltas":
        # Special visualization for prime deltas
        plt.subplot(2, 2, 1)
        if len(math_raw_preds) > 0:
            actual_deltas = y[:len(math_raw_preds)].cpu().numpy()
            predicted_deltas = [pred[0] if len(pred) > 0 else 0 for pred in math_raw_preds]
            plt.plot(actual_deltas, label='Actual Prime Deltas', alpha=0.7)
            plt.plot(predicted_deltas, label='Predicted Prime Deltas', alpha=0.7)
            plt.title('Prime Delta Prediction')
            plt.legend()
        
        # Field adaptation analysis
        plt.subplot(2, 2, 2)
        if len(math_raw_preds) > 0:
            errors = [abs(actual_deltas[i] - predicted_deltas[i]) for i in range(min(len(actual_deltas), len(predicted_deltas)))]
            plt.plot(errors, label='Pattern Mismatch', color='red', alpha=0.7)
            plt.title('Prime Delta Pattern Recognition Error')
            plt.ylabel('Pattern Mismatch')
            plt.legend()
    else:
        # Standard visualization
        plt.subplot(2, 2, 1)
        x_plot = x.cpu().numpy()
        y_plot = y.cpu().numpy()
        
        # Debug: Print ground truth range
        print(f"Ground truth range: {y_plot.min():.6f} to {y_plot.max():.6f}")
        if len(math_raw_preds) > 0:
            pred_plot = [pred[0] if len(pred) > 0 else 0 for pred in math_raw_preds]
            print(f"Prediction range: {min(pred_plot):.6f} to {max(pred_plot):.6f}")
        
        plt.plot(x_plot, y_plot, label='Ground Truth', linewidth=2)
        if len(math_raw_preds) > 0:
            pred_plot = [pred[0] if len(pred) > 0 else 0 for pred in math_raw_preds]
            plt.plot(x_plot[:len(pred_plot)], pred_plot, label='TinyCIMM-Euler Prediction', alpha=0.7)
        plt.legend()
        plt.title(f'Mathematical Predictions ({signal})')
        
        # Add y-axis limits to better see ground truth if predictions are extreme
        if len(math_raw_preds) > 0:
            pred_plot = [pred[0] if len(pred) > 0 else 0 for pred in math_raw_preds]
            if max(pred_plot) > 100 * max(y_plot):  # If predictions are much larger than ground truth
                plt.ylim(min(y_plot) - 0.1, max(y_plot) + 0.1)
                plt.text(0.02, 0.98, f'Note: Predictions range {min(pred_plot):.1f} to {max(pred_plot):.1f}', 
                        transform=plt.gca().transAxes, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Complexity evolution
    plt.subplot(2, 2, 3)
    plt.plot(math_metrics, label='Mathematical Complexity', color='purple', alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('Complexity Metric')
    plt.title('Mathematical Complexity Evolution')
    plt.legend()
    
    # Network size evolution
    plt.subplot(2, 2, 4)
    plt.plot(math_hsizes, label='Network Size', color='green', alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('Number of Neurons')
    plt.title('Mathematical Network Size Evolution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(signal_img_dir, f'mathematical_analysis_{signal}.png'))
    plt.close()
    
    # Performance and fractal analysis
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(math_performance, label='Mathematical Performance', color='orange', alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('Performance Score')
    plt.title('Mathematical Reasoning Performance')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(math_losses, label='Field Adaptation Signal', color='red', alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('Adaptation Signal')
    plt.title('Field-Aware Adaptation Signal Evolution')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    if math_fractals:
        fractal_x = torch.arange(0, len(math_fractals)*15, 15)
        fractal_tensor = torch.tensor(math_fractals)
        mask = ~torch.isnan(fractal_tensor)
        plt.plot(fractal_x[mask].cpu().numpy(), fractal_tensor[mask].cpu().numpy(), 
                label='Mathematical Fractal Dimension', color='blue', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Fractal Dimension')
        plt.title('Mathematical Structure Complexity')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(signal_img_dir, f'mathematical_performance_{signal}.png'))
    plt.close()

    # Field-aware loss component analysis
    plt.figure(figsize=(15, 10))
    
    # Extract field-aware balance components
    qbe_balances = [log.get('qbe_balance', 0) for log in logs if 'qbe_balance' in log]
    energy_balances = [log.get('energy_balance', 0) for log in logs if 'energy_balance' in log]
    coherence_losses = [log.get('coherence_loss', 0) for log in logs if 'coherence_loss' in log]
    einstein_corrections = [log.get('einstein_correction', 1) for log in logs if 'einstein_correction' in log]
    feynman_dampings = [log.get('feynman_damping', 1) for log in logs if 'feynman_damping' in log]
    
    plt.subplot(2, 3, 1)
    if qbe_balances:
        plt.plot(qbe_balances, label='QBE Balance', color='purple', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('QBE Balance')
        plt.title('Quantum Balance Evolution')
        plt.legend()
    
    plt.subplot(2, 3, 2)
    if energy_balances:
        plt.plot(energy_balances, label='Energy Balance', color='orange', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Energy Balance')
        plt.title('Energy-Information Balance')
        plt.legend()
    
    plt.subplot(2, 3, 3)
    if coherence_losses:
        plt.plot(coherence_losses, label='Coherence Loss', color='cyan', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Coherence Loss')
        plt.title('Superfluid Coherence Loss')
        plt.legend()
    
    plt.subplot(2, 3, 4)
    if einstein_corrections:
        plt.plot(einstein_corrections, label='Einstein Correction', color='red', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Correction Factor')
        plt.title('Einstein Energy Correction')
        plt.legend()
    
    plt.subplot(2, 3, 5)
    if feynman_dampings:
        plt.plot(feynman_dampings, label='Feynman Damping', color='green', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Damping Factor')
        plt.title('Feynman Entropy Damping')
        plt.legend()
    
    plt.subplot(2, 3, 6)
    # Combined field dynamics
    if qbe_balances and energy_balances and coherence_losses:
        combined_field = [qbe_balances[i] + energy_balances[i] + coherence_losses[i] 
                         for i in range(min(len(qbe_balances), len(energy_balances), len(coherence_losses)))]
        plt.plot(combined_field, label='Combined Field Signal', color='black', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Combined Field Signal')
        plt.title('Total Field-Aware Adaptation Signal')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(signal_img_dir, f'field_aware_loss_analysis_{signal}.png'))
    plt.close()

def run_all_mathematical_experiments():
    """Run all long-term online mathematical reasoning experiments (like CIMM)"""
    test_cases = [
        ("prime_deltas", {
            "hidden_size": 32,  # Larger initial capacity for complex patterns
            "math_memory_size": 25,  # More memory for longer sequences
            "adaptation_steps": 50   # More adaptation steps
        }),
        ("fibonacci_ratios", {
            "hidden_size": 24, 
            "math_memory_size": 20, 
            "pattern_decay": 0.95
        }),
        ("polynomial_sequence", {
            "hidden_size": 28, 
            "math_memory_size": 18, 
            "adaptation_steps": 30
        }),
        ("recursive_sequence", {
            "hidden_size": 26, 
            "math_memory_size": 20, 
            "pattern_decay": 0.97
        }),
        ("mathematical_harmonic", {
            "hidden_size": 20, 
            "math_memory_size": 15
        }),
    ]
    
    for test_name, model_kwargs in test_cases:
        print(f"\n=== Running Long-term Mathematical Experiment: {test_name} ===")
        print(f"Expected challenge level: {'Extreme' if test_name == 'prime_deltas' else 'Very High' if 'sequence' in test_name else 'High'}")
        print(f"Training for 10,000 steps (CIMM-style long-term learning)...")
        
        try:
            # Run for 10,000 steps instead of 400
            run_experiment(TinyCIMMEuler, signal=test_name, steps=10000, **model_kwargs)
            print(f"✓ Completed {test_name} successfully")
        except Exception as e:
            print(f"✗ Error in {test_name}: {str(e)}")
            continue

def run_quick_test():
    """Run a quick test with shorter sequences for development"""
    print("=== Quick Test: Prime Deltas (1000 steps) ===")
    try:
        run_experiment(TinyCIMMEuler, signal="prime_deltas", steps=1000, 
                      hidden_size=24, math_memory_size=15, adaptation_steps=25)
        print("✓ Quick test completed successfully")
    except Exception as e:
        print(f"✗ Quick test failed: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("TinyCIMM-Euler: Long-term Mathematical Reasoning (CIMM-Style)")
    print("=" * 60)
    print("\nFocusing on long-term online learning for mathematical patterns...")
    print("No pre-training - pure online learning like CIMM over 10,000 steps.\n")
    
    # Ask user for test type
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_test()
    else:
        run_all_mathematical_experiments()
    
    print("\n" + "=" * 60)
    print("Online mathematical reasoning experiments completed!")
    print("Check experiment_images/ and experiment_logs/ for detailed results.")
    print("=" * 60)
