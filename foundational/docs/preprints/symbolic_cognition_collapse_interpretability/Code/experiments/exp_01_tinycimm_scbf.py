#!/usr/bin/env python3
"""
TinyCIMM-SCBF Integration Experiment
===================================

Single-file implementation showing how to integrate SCBF analysis with
actual TinyCIMM-Euler models. This demonstrates the SCBF framework
working with real neural networks on mathematical reasoning tasks.

Usage:
    python tinycimm_scbf_experiment.py --signal prime_deltas --steps 1000
    python tinycimm_scbf_experiment.py --signal fibonacci_ratios --steps 500
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add paths for imports - do this before any imports to avoid linting errors
SCBF_DIR = Path(__file__).parent
TINYCIMM_DIR = Path(__file__).parent.parent / "TinyCIMM" / "TinyCIMM-Euler"
sys.path.insert(0, str(SCBF_DIR))
sys.path.insert(0, str(TINYCIMM_DIR))
sys.path.insert(0, str(TINYCIMM_DIR / "experiments"))

# Import TinyCIMM components with type ignore for linter
try:
    import tinycimm_euler  # type: ignore # noqa
    from tinycimm_euler import TinyCIMMEuler  # type: ignore # noqa
    import run_experiment  # type: ignore # noqa
    from run_experiment import get_signal, generate_primes, get_prime_deltas  # type: ignore # noqa
    print("✓ Successfully imported TinyCIMM-Euler components")
except ImportError as e:
    print(f"❌ Failed to import TinyCIMM-Euler: {e}")
    print("Please ensure TinyCIMM-Euler is properly installed")
    sys.exit(1)

# Import SCBF components with fallback handling
try:
    # Try relative imports first (when imported as module)
    from . import loggers  # type: ignore # noqa
    from .loggers import create_experiment_logger, finalize_experiment  # type: ignore # noqa
    from . import visualization  # type: ignore # noqa
    from .visualization import plot_complete_scbf_dashboard, save_all_plots  # type: ignore # noqa
    from .metrics import entropy_collapse, activation_ancestry, semantic_attractors, bifractal_lineage  # type: ignore # noqa
    from .metrics.entropy_collapse import compute_symbolic_entropy_collapse  # type: ignore # noqa
    from .metrics.activation_ancestry import compute_activation_ancestry  # type: ignore # noqa
    from .metrics.semantic_attractors import compute_semantic_attractor_density  # type: ignore # noqa
    from .metrics.bifractal_lineage import compute_bifractal_lineage  # type: ignore # noqa
    print("✓ Successfully imported SCBF components")
except ImportError:
    # Fallback to absolute imports (when run as script)
    try:
        import loggers  # type: ignore # noqa
        from loggers import create_experiment_logger, finalize_experiment  # type: ignore # noqa
        import visualization  # type: ignore # noqa
        from visualization import plot_complete_scbf_dashboard, save_all_plots  # type: ignore # noqa
        from metrics.entropy_collapse import compute_symbolic_entropy_collapse  # type: ignore # noqa
        from metrics.activation_ancestry import compute_activation_ancestry  # type: ignore # noqa
        from metrics.semantic_attractors import compute_semantic_attractor_density  # type: ignore # noqa
        from metrics.bifractal_lineage import compute_bifractal_lineage  # type: ignore # noqa
        print("✓ Successfully imported SCBF components (fallback)")
    except ImportError as e:
        print(f"❌ Failed to import SCBF: {e}")
        print("Please ensure SCBF modules are available")
        sys.exit(1)

def compute_quantum_error_metrics(y_true, y_pred, device='cuda', prediction_history=None, target_history=None):
    """
    Computes entropy-aware and quantum-based error metrics using PyTorch tensors.
    Enhanced for single-value predictions by using historical context.
    """
    # Ensure inputs are PyTorch tensors on the correct device
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float64, device=device)
    else:
        y_true = y_true.to(device, dtype=torch.float64)
        
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float64, device=device) 
    else:
        y_pred = y_pred.to(device, dtype=torch.float64)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # If we have historical context, use it for better metrics
    if prediction_history is not None and target_history is not None and len(prediction_history) > 1:
        # Use recent history to build sequences
        recent_preds = torch.tensor(prediction_history[-10:], dtype=torch.float64, device=device)
        recent_targets = torch.tensor(target_history[-10:], dtype=torch.float64, device=device)
        
        # Combine current with recent history
        y_true_seq = torch.cat([recent_targets, y_true])
        y_pred_seq = torch.cat([recent_preds, y_pred])
    else:
        # Fallback: use current values with some artificial variation for metrics
        y_true_seq = y_true
        y_pred_seq = y_pred
    
    # Mean center but keep original scale for better sensitivity
    y_true_centered = y_true_seq - y_true_seq.mean()
    y_pred_centered = y_pred_seq - y_pred_seq.mean()
    
    # Scale by std deviation to normalize but keep differences
    y_true_std = torch.std(y_true_centered) + 1e-6
    y_pred_std = torch.std(y_pred_centered) + 1e-6
    
    y_true_norm = y_true_centered / y_true_std
    y_pred_norm = y_pred_centered / y_pred_std

    # For probability distributions: shift to positive range and normalize
    y_true_pos = y_true_norm - y_true_norm.min() + 1e-6
    y_pred_pos = y_pred_norm - y_pred_norm.min() + 1e-6
    
    y_true_prob = y_true_pos / torch.sum(y_true_pos)
    y_pred_prob = y_pred_pos / torch.sum(y_pred_pos)

    epsilon = 1e-9
    
    # KL-Divergence (now sensitive to actual differences)
    kl_div = torch.sum(y_true_prob * torch.log((y_true_prob + epsilon) / (y_pred_prob + epsilon)))

    # Jensen-Shannon Divergence
    js_div = torch.sqrt(0.5 * (kl_div + torch.sum(y_pred_prob * torch.log((y_pred_prob + epsilon) / (y_true_prob + epsilon)))))
    js_div = torch.nan_to_num(js_div, nan=0.0)

    # Wasserstein Distance (using normalized sequences)
    wd = torch.sum(torch.abs(torch.cumsum(y_true_norm, dim=0) - torch.cumsum(y_pred_norm, dim=0)))

    # Quantum Wave Coherence Score (QWCS) - measure correlation
    if len(y_true_seq) > 1:
        # Add small noise for numerical stability
        noise = torch.normal(0, 1e-4, size=y_pred_norm.shape, device=device)
        y_pred_noisy = y_pred_norm + noise

        if torch.var(y_true_norm) < 1e-6 or torch.var(y_pred_noisy) < 1e-6:
            qwcs = torch.tensor(0.5, device=device)
        else:
            try:
                corr_matrix = torch.corrcoef(torch.stack([y_true_norm, y_pred_noisy]))
                qwcs = 1 - torch.abs(corr_matrix[0, 1])
            except:
                qwcs = torch.tensor(0.5, device=device)
    else:
        # Single value: use direct error as coherence measure
        direct_error = torch.abs(y_true - y_pred)
        qwcs = torch.exp(-direct_error)  # Higher coherence for smaller errors

    qwcs = torch.nan_to_num(qwcs, nan=0.5)

    # Entropy scaling adjustments
    entropy_value = torch.sum(-y_true_prob * torch.log(y_true_prob + epsilon))
    qwcs = qwcs * (1 + 0.02 * entropy_value)
    entropy_scaling = torch.clamp(1.0 + 0.1 * entropy_value, min=0.8, max=1.2)
    kl_div = kl_div * entropy_scaling

    return {
        "KL-Divergence": kl_div.item(),
        "Jensen-Shannon": js_div.item(),
        "Wasserstein Distance": wd.item(),
        "QWCS": qwcs.item()
    }

def compute_composite_quantum_loss(y_true, y_pred, weights=None, device='cuda', prediction_history=None, target_history=None):
    """
    Computes a composite loss using quantum-aware metrics.
    Returns both individual metrics and combined loss.
    """
    if weights is None:
        # Default weights - emphasize KL-Divergence and QWCS for neural adaptation
        weights = {
            "KL-Divergence": 0.4,
            "Jensen-Shannon": 0.2, 
            "Wasserstein Distance": 0.2,
            "QWCS": 0.2
        }
    
    metrics = compute_quantum_error_metrics(y_true, y_pred, device, prediction_history, target_history)
    
    # Compute weighted composite loss
    composite_loss = (
        weights["KL-Divergence"] * metrics["KL-Divergence"] +
        weights["Jensen-Shannon"] * metrics["Jensen-Shannon"] +
        weights["Wasserstein Distance"] * metrics["Wasserstein Distance"] +
        weights["QWCS"] * metrics["QWCS"]
    )
    
    return torch.tensor(composite_loss, requires_grad=True, device=device), metrics

def extract_tinycimm_activations(model, device='cuda'):
    """Extract activations from TinyCIMM-Euler model using PyTorch tensors."""
    try:
        # Debug: Print model attributes to understand structure
        if not hasattr(model, '_debug_printed'):
            model._debug_printed = True
            print(f"Debug: Model type: {type(model)}")
            print(f"Debug: Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        
        # Try multiple strategies to get activations
        activations = []
        
        # Strategy 1: Check for hidden state
        if hasattr(model, 'hidden') and model.hidden is not None:
            if torch.is_tensor(model.hidden):
                hidden = model.hidden.detach().to(device)
                if hidden.ndim == 1:
                    hidden = hidden.unsqueeze(0)  # Add batch dimension
                activations.append(hidden)
        
        # Strategy 2: Check for memory states  
        if hasattr(model, 'math_memory') and model.math_memory is not None:
            if torch.is_tensor(model.math_memory):
                memory = model.math_memory.detach().to(device)
                if memory.ndim == 1:
                    memory = memory.unsqueeze(0)  # Add batch dimension
                activations.append(memory)
        
        # Strategy 3: Extract from named parameters/buffers
        for name, param in model.named_parameters():
            if 'hidden' in name.lower() or 'state' in name.lower():
                if torch.is_tensor(param):
                    param_data = param.detach().to(device)
                    if param_data.ndim == 1:
                        param_data = param_data.unsqueeze(0)  # Add batch dimension
                    activations.append(param_data)
                    break  # Only take first match
        
        # Strategy 4: Extract from any buffers
        for name, buffer in model.named_buffers():
            if buffer is not None and torch.is_tensor(buffer):
                buffer_data = buffer.detach().to(device)
                if buffer_data.ndim == 1:
                    buffer_data = buffer_data.unsqueeze(0)  # Add batch dimension
                activations.append(buffer_data)
                break  # Only take first match
        
        # Strategy 5: Fallback - use layer weights as pseudo-activations
        if not activations:
            for name, param in model.named_parameters():
                if torch.is_tensor(param) and param.ndim == 2:  # Weight matrix
                    # Use first few rows as activation proxy
                    weight_data = param.detach().to(device)[:min(10, param.shape[0]), :]
                    activations.append(weight_data)
                    break
        
        if activations:
            # Combine all activation sources using PyTorch
            combined = torch.cat(activations, dim=0)
            print(f"Debug: Extracted activations shape: {combined.shape} on {combined.device}")
            return combined
        
        print("Warning: No activations found in model")
        return None
        
    except Exception as e:
        print(f"Warning: Could not extract activations: {e}")
        return None

def run_scbf_analysis(model, step_idx, x_batch=None, prev_weights=None, device='cuda'):
    """Run SCBF analysis on TinyCIMM model using PyTorch tensors."""
    scbf_results = {}
    
    try:
        # Extract activations
        activations = extract_tinycimm_activations(model, device)
        
        if activations is not None and activations.shape[0] >= 2:
            # Convert to numpy only when necessary for SCBF functions
            # (until we fully convert SCBF metrics to PyTorch)
            activations_np = activations.detach().cpu().numpy()
            
            # Entropy collapse analysis
            try:
                entropy_metrics = compute_symbolic_entropy_collapse(activations_np)
                scbf_results['entropy_collapse'] = {
                    'magnitude': entropy_metrics['collapse_magnitude'],
                    'rate': entropy_metrics['collapse_rate'],
                    'symbolic_states': entropy_metrics['symbolic_states']
                }
            except Exception as e:
                print(f"Entropy analysis failed: {e}")
            
            # Activation ancestry (if enough data)
            if activations.shape[0] >= 3:
                try:
                    ancestry_metrics = compute_activation_ancestry(activations_np)
                    scbf_results['ancestry'] = {
                        'strength': ancestry_metrics['ancestry_strength'],
                        'stability': ancestry_metrics['lineage_stability'],
                        'dimension': ancestry_metrics['bifractal_dimension']
                    }
                except Exception as e:
                    print(f"Ancestry analysis failed: {e}")
            
            # Semantic attractors (if enough samples)
            if activations.shape[0] >= 5:
                try:
                    attractor_metrics = compute_semantic_attractor_density(activations_np)
                    scbf_results['attractors'] = {
                        'count': attractor_metrics['attractor_count'],
                        'density': attractor_metrics['attractor_density'],
                        'stability': attractor_metrics['attractor_stability']
                    }
                except Exception as e:
                    print(f"Attractor analysis failed: {e}")
        
        # Weight analysis using PyTorch tensors with dynamic network support
        try:
            # Collect all current weights as PyTorch tensors
            current_weights_tensors = []
            for param in model.parameters():
                current_weights_tensors.append(param.detach().to(device).flatten())
            
            if current_weights_tensors:
                # Concatenate using PyTorch
                current_weights_tensor = torch.cat(current_weights_tensors)
                current_weights_np = current_weights_tensor.cpu().numpy()  # Convert for legacy SCBF function
                
                # Handle dynamic network growth - only compare if shapes match or can be aligned
                if prev_weights is not None:
                    if len(current_weights_np) == len(prev_weights):
                        # Same size - normal comparison
                        lineage_metrics = compute_bifractal_lineage(current_weights_np, prev_weights)
                        scbf_results['lineage'] = {
                            'fractal_dimension': lineage_metrics['fractal_dimension'],
                            'entropy': lineage_metrics['lineage_entropy'],
                            'similarity': lineage_metrics['structural_similarity']
                        }
                    elif len(current_weights_np) > len(prev_weights):
                        # Network grew - compare only the overlapping portion
                        overlap_size = len(prev_weights)
                        current_overlap = current_weights_np[:overlap_size]
                        lineage_metrics = compute_bifractal_lineage(current_overlap, prev_weights)
                        scbf_results['lineage'] = {
                            'fractal_dimension': lineage_metrics['fractal_dimension'],
                            'entropy': lineage_metrics['lineage_entropy'],
                            'similarity': lineage_metrics['structural_similarity'],
                            'network_growth': len(current_weights_np) - len(prev_weights)
                        }
                    else:
                        # Network shrank (unusual) - skip comparison this step
                        print(f"Network size decreased from {len(prev_weights)} to {len(current_weights_np)} - skipping lineage analysis")
                else:
                    # First analysis - just compute basic metrics
                    lineage_metrics = compute_bifractal_lineage(current_weights_np, None)
                    scbf_results['lineage'] = {
                        'fractal_dimension': lineage_metrics['fractal_dimension'],
                        'entropy': lineage_metrics['lineage_entropy'],
                        'similarity': 1.0  # Perfect similarity with self
                    }
        except Exception as e:
            print(f"Lineage analysis failed: {e}")
    
    except Exception as e:
        print(f"SCBF analysis error: {e}")
    
    return scbf_results

def run_tinycimm_scbf_experiment(signal="prime_deltas", steps=1000, scbf_interval=50, **kwargs):
    """
    Run TinyCIMM-Euler experiment with integrated SCBF analysis.
    
    Args:
        signal: Mathematical signal type ("prime_deltas", "fibonacci_ratios", etc.)
        steps: Number of adaptation steps
        scbf_interval: Run SCBF analysis every N steps
        **kwargs: Additional TinyCIMM model parameters
    """
    
    print(f"🚀 Starting TinyCIMM-SCBF experiment: {signal}")
    print(f"📊 Steps: {steps}, SCBF interval: {scbf_interval}")
    
    # Create SCBF logger
    logger = create_experiment_logger(f"tinycimm_{signal}_{steps}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Generate signal data (using TinyCIMM's get_signal function)
    print(f"📈 Generating {signal} data...")
    x, y = get_signal(signal, steps, seed=42)
    x = x.to(device)
    y = y.to(device)
    
    print(f"✓ Data generated: x.shape={x.shape}, y.shape={y.shape}")
    
    # Model configuration based on signal (from TinyCIMM run_experiment)
    input_size = x.shape[1] if len(x.shape) > 1 else 1
    
    if signal == "prime_deltas":
        hidden_size = kwargs.get('hidden_size', 32)
        model_kwargs = {
            'math_memory_size': 40,
            'adaptation_steps': 50,
            'pattern_decay': 0.995,
            'learning_rate': 0.005
        }
    else:
        hidden_size = kwargs.get('hidden_size', 20)
        model_kwargs = {
            'math_memory_size': 25,
            'adaptation_steps': 35,
            'pattern_decay': 0.96
        }
    
    # Override with any provided kwargs
    model_kwargs.update(kwargs)
    
    # Initialize TinyCIMM-Euler model
    print(f"🧠 Initializing TinyCIMM-Euler (hidden_size={hidden_size})...")
    model = TinyCIMMEuler(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=1,
        device=device,  # Add required device parameter
        **model_kwargs
    ).to(device)
    
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Live adaptation loop with SCBF integration and proper TinyCIMM-style adaptation
    print("🎯 Starting live adaptation with SCBF analysis...")
    prev_weights = None
    performance_history = torch.tensor([], device=device)  # Use PyTorch tensor for performance tracking
    prediction_history = []  # For quantum metrics
    target_history = []  # For quantum metrics
    adaptive_lr = model_kwargs.get('learning_rate', 0.005)
    
    for step in range(steps):
        # Get current batch (individual processing like original)
        x_batch = x[step:step+1] if step < len(x) else x[-1:] 
        y_batch = y[step:step+1] if step < len(y) else y[-1:]
        
        # Forward pass - live adaptation inference
        model.train()  # enable gradients for online adaptation (no offline training)
        output = model(x_batch)
        
        # Compute MSE loss for adaptation feedback (like original TinyCIMM-Euler)
        if output.shape != y_batch.shape:
            y_batch = y_batch.view_as(output)  # Reshape target to match output
        
        # Use MSE for gradient-based adaptation (matches original run_experiment.py)
        mse_loss = torch.nn.functional.mse_loss(output, y_batch)
        loss = mse_loss  # Keep MSE for actual adaptation
        
        # Store current predictions and targets for quantum metrics
        prediction_history.append(output.detach().cpu().numpy().flatten())
        target_history.append(y_batch.detach().cpu().numpy().flatten())
        
        # Keep only recent history (last 10 steps for quantum metrics)
        if len(prediction_history) > 10:
            prediction_history.pop(0)
            target_history.pop(0)
        
        # Compute quantum metrics for analysis (but don't use for gradients)
        with torch.no_grad():
            # Prepare history for quantum metrics
            quantum_history_pred = np.concatenate(prediction_history) if len(prediction_history) > 1 else prediction_history[0] if prediction_history else None
            quantum_history_target = np.concatenate(target_history) if len(target_history) > 1 else target_history[0] if target_history else None
            
            _, quantum_metrics = compute_composite_quantum_loss(
                y_batch, output, device=device,
                prediction_history=quantum_history_pred,
                target_history=quantum_history_target
            )
        
        # Backward pass with proper gradient updates for live adaptation (like original)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping like original
        
        # Update model using TinyCIMM's individual adaptation approach
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            model.optimizer.step()
        elif hasattr(model, 'adapt'):
            model.adapt()
        else:
            # Fallback to simple gradient update with original-style learning rate
            lr = model_kwargs.get('learning_rate', 0.005)
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad.data
        
        # Individual adaptation step (like original TinyCIMM-Euler)
        if hasattr(model, 'online_adaptation_step'):
            model.online_adaptation_step(x_batch, y_batch)
        
        # Track performance for adaptive learning rate using PyTorch tensors
        performance_history = torch.cat([performance_history, torch.tensor([loss.item()], device=device)])
        
        # Dynamic learning rate adjustment based on performance trends (using PyTorch)
        if len(performance_history) > 20 and step % 10 == 0:
            recent_10 = performance_history[-10:]
            previous_10 = performance_history[-20:-10]
            recent_trend = torch.mean(recent_10) - torch.mean(previous_10)
            recent_trend = recent_trend.item()  # Convert to Python float for comparison
            
            # Adaptive LR based on signal type (like original)
            if recent_trend < -0.005:  # Performance dropping
                adaptive_lr = min(adaptive_lr * 1.05, 0.02)
                print(f"Step {step}: LR increased to {adaptive_lr:.6f} due to performance drop")
            elif recent_trend > 0.005:  # Performance improving
                adaptive_lr = max(adaptive_lr * 0.98, 5e-6)
                print(f"Step {step}: LR decreased to {adaptive_lr:.6f} due to performance gain")
            
            # Update optimizer learning rate if available
            if hasattr(model, 'optimizer') and model.optimizer is not None:
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = adaptive_lr
        
        # Log basic metrics including quantum-aware analysis
        log_entry = {
            'step': step,
            'loss': loss.item(),  # MSE loss used for adaptation feedback
            'output': output.item(),
            'target': y_batch.item(),
            'quantum_metrics': quantum_metrics  # Include all four quantum metrics for analysis
        }
        
        # Run SCBF analysis periodically
        if step % scbf_interval == 0:
            scbf_results = run_scbf_analysis(model, step, x_batch, prev_weights, device)
            if scbf_results:
                log_entry['scbf'] = scbf_results
                
                # Store current weights for next comparison using PyTorch
                current_weights_tensors = []
                for param in model.parameters():
                    current_weights_tensors.append(param.detach().to(device).flatten())
                prev_weights = torch.cat(current_weights_tensors).cpu().numpy()  # Convert to numpy for legacy SCBF
            
            # Progress update
            metrics_count = len(scbf_results) if scbf_results else 0
            print(f"Step {step:4d}: Loss={loss.item():.6f}, "
                  f"Output={output.item():.3f}, Target={y_batch.item():.3f}, "
                  f"SCBF metrics={metrics_count}")
        
        # Log the step
        logger.log_step(log_entry)
    
    print(f"✅ Live adaptation completed after {steps} steps")
    
    # Finalize experiment and generate analysis
    print("📊 Finalizing experiment and generating analysis...")
    results_file = finalize_experiment(logger)
    
    # Generate visualizations
    logs = logger.get_logs()
    scbf_logs = [log for log in logs if 'scbf' in log]
    
    print(f"📈 Generating visualizations ({len(scbf_logs)} SCBF data points)...")
    if scbf_logs:
        plot_complete_scbf_dashboard(logs)
        # Save plots to the same experiment directory as results
        plots_dir = logger.experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        save_all_plots(logs, str(plots_dir))
    else:
        print("⚠️ No SCBF data found for visualization")
    
    # Print summary
    print(f"\n🎉 Experiment Summary:")
    print(f"   Signal: {signal}")
    print(f"   Total steps: {steps}")
    print(f"   SCBF analyses: {len(scbf_logs)}")
    print(f"   Final loss: {logs[-1].get('loss', 'N/A')}")
    print(f"   Results saved to: {results_file}")
    
    return {
        'signal': signal,
        'steps': steps,
        'final_loss': logs[-1].get('loss'),
        'scbf_analyses': len(scbf_logs),
        'results_file': results_file
    }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='TinyCIMM-SCBF Integration Experiment')
    parser.add_argument('--signal', type=str, default='prime_deltas',
                       help='Mathematical signal type (prime_deltas, fibonacci_ratios, etc.)')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of adaptation steps')
    parser.add_argument('--scbf-interval', type=int, default=50,
                       help='Run SCBF analysis every N steps')
    parser.add_argument('--hidden-size', type=int, default=None,
                       help='Model hidden size (auto-configured if not specified)')
    
    args = parser.parse_args()
    
    kwargs = {}
    if args.hidden_size:
        kwargs['hidden_size'] = args.hidden_size
    
    try:
        results = run_tinycimm_scbf_experiment(
            signal=args.signal,
            steps=args.steps, 
            scbf_interval=args.scbf_interval,
            **kwargs
        )
        print(f"\n✅ Experiment completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
