"""
exp_01_threshold_detector.py - Core SEC Threshold Detection Algorithm

Detects phase transition thresholds from dynamical system trajectories
using SEC gradient analysis.

Key Method:
1. Compute information I(t) via histogram entropy
2. Compute entropy rate H(t) via finite differences
3. Compute gradient ratio R(t) = |∇I|/|∇H|
4. Find parameter where dR/dλ → 0 (critical point)
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy import stats

XI = 1 + np.pi / 55  # Balance operator ≈ 1.0571
PHI = (1 + np.sqrt(5)) / 2


def compute_information(trajectory, n_bins=50):
    """
    Compute information content via histogram entropy.
    I = -Σ p_i log(p_i)
    """
    # Flatten trajectory if multi-dimensional
    if trajectory.ndim > 1:
        trajectory = trajectory.flatten()
    
    # Create histogram
    hist, _ = np.histogram(trajectory, bins=n_bins, density=True)
    hist = hist[hist > 0]  # Remove zeros
    
    # Compute entropy (information)
    return -np.sum(hist * np.log(hist + 1e-10))


def compute_entropy_rate(trajectory, window=100):
    """
    Compute local entropy rate via sliding window.
    H(t) = d(entropy)/dt
    """
    n = len(trajectory)
    rates = []
    
    for i in range(0, n - window, window // 2):
        segment1 = trajectory[i:i + window // 2]
        segment2 = trajectory[i + window // 2:i + window]
        
        h1 = compute_information(segment1)
        h2 = compute_information(segment2)
        
        rate = (h2 - h1) / (window // 2)
        rates.append(rate)
    
    return np.array(rates)


def gradient_ratio(info_values, entropy_rates, epsilon=1e-10):
    """
    Compute R(t) = |∇I| / |∇H|
    """
    grad_i = np.gradient(info_values)
    grad_h = np.abs(entropy_rates) + epsilon
    
    # Align lengths
    min_len = min(len(grad_i), len(grad_h))
    return np.abs(grad_i[:min_len]) / grad_h[:min_len]


def detect_threshold(simulate_fn, param_range, n_samples=50):
    """
    Detect SEC threshold by scanning parameter space.
    
    Args:
        simulate_fn: Function that takes parameter and returns trajectory
        param_range: (min, max) parameter range to scan
        n_samples: Number of parameter values to test
    
    Returns:
        Detected threshold parameter value
    """
    params = np.linspace(param_range[0], param_range[1], n_samples)
    ratios = []
    info_vals = []
    
    for param in params:
        traj = simulate_fn(param)
        
        # Compute information
        info = compute_information(traj)
        info_vals.append(info)
        
        # Compute entropy rate
        h_rate = compute_entropy_rate(traj)
        
        # Mean gradient ratio
        if len(h_rate) > 1:
            r = gradient_ratio(np.array([info] * len(h_rate)), h_rate)
            ratios.append(np.mean(r))
        else:
            ratios.append(0)
    
    ratios = np.array(ratios)
    
    # Find where gradient of ratio approaches zero (critical point)
    grad_r = np.gradient(ratios)
    
    # Find index of minimum |gradient| (critical point)
    critical_idx = np.argmin(np.abs(grad_r[1:-1])) + 1  # Avoid edges
    
    return params[critical_idx], ratios, params


def simulate_logistic(r, n_steps=1000, x0=0.5):
    """Logistic map: x_{n+1} = r * x_n * (1 - x_n)"""
    x = np.zeros(n_steps)
    x[0] = x0
    for i in range(1, n_steps):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x


def simulate_lorenz(rho, n_steps=5000, dt=0.01, sigma=10, beta=8/3):
    """Lorenz system trajectory."""
    x, y, z = 1.0, 1.0, 1.0
    trajectory = []
    
    for _ in range(n_steps):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x, y, z = x + dx, y + dy, z + dz
        trajectory.append(x)
    
    return np.array(trajectory)


def run_experiment():
    """Test threshold detection on logistic map and Lorenz system."""
    
    print("=" * 60)
    print("SEC Threshold Detection Algorithm")
    print("=" * 60)
    
    np.random.seed(42)
    results = {}
    
    # Test 1: Logistic Map
    print("\n--- Logistic Map ---")
    
    threshold_logistic, ratios_log, params_log = detect_threshold(
        simulate_fn=simulate_logistic,
        param_range=(2.5, 4.0),
        n_samples=60
    )
    
    feigenbaum_point = 3.5699456  # Known value
    
    print(f"Detected threshold: r* = {threshold_logistic:.4f}")
    print(f"Feigenbaum point:   r_inf = {feigenbaum_point:.4f}")
    print(f"Error: {abs(threshold_logistic - feigenbaum_point) / feigenbaum_point * 100:.2f}%")
    
    # Check xi relationship
    xi_ratio_log = threshold_logistic / 3.37
    print(f"r*/3.37 = {xi_ratio_log:.4f} (expected ξ = {XI:.4f})")
    
    results['logistic'] = {
        'detected_threshold': threshold_logistic,
        'known_threshold': feigenbaum_point,
        'error_percent': abs(threshold_logistic - feigenbaum_point) / feigenbaum_point * 100,
        'xi_ratio': xi_ratio_log,
        'xi_expected': XI
    }
    
    # Test 2: Lorenz System
    print("\n--- Lorenz System ---")
    
    threshold_lorenz, ratios_lor, params_lor = detect_threshold(
        simulate_fn=simulate_lorenz,
        param_range=(20, 30),
        n_samples=40
    )
    
    lorenz_critical = 24.74  # Known onset of chaos
    
    print(f"Detected threshold: ρ* = {threshold_lorenz:.4f}")
    print(f"Known critical:     ρ_c = {lorenz_critical:.4f}")
    print(f"Error: {abs(threshold_lorenz - lorenz_critical) / lorenz_critical * 100:.2f}%")
    
    # Lorenz dimension prediction
    xi_minus_1 = XI - 1
    predicted_dim = 2 + xi_minus_1
    observed_dim = 2.06  # From literature
    
    print(f"\nDimension Analysis:")
    print(f"Predicted D = 2 + (ξ-1) = {predicted_dim:.4f}")
    print(f"Observed D = {observed_dim:.3f}")
    print(f"Error: {abs(predicted_dim - observed_dim) / observed_dim * 100:.2f}%")
    
    results['lorenz'] = {
        'detected_threshold': threshold_lorenz,
        'known_threshold': lorenz_critical,
        'error_percent': abs(threshold_lorenz - lorenz_critical) / lorenz_critical * 100,
        'predicted_dimension': predicted_dim,
        'observed_dimension': observed_dim,
        'dimension_error_percent': abs(predicted_dim - observed_dim) / observed_dim * 100
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary: ξ Relationships")
    print("=" * 60)
    print(f"ξ = 1 + π/55 = {XI:.6f}")
    print(f"ξ - 1 = {XI - 1:.6f}")
    print(f"\nLogistic: r*/3.37 = {xi_ratio_log:.4f} ≈ ξ")
    print(f"Lorenz dimension: 2 + (ξ-1) = {predicted_dim:.4f} ≈ D_observed = {observed_dim}")
    
    # Save results
    output = {
        'experiment': 'sec_threshold_detector',
        'timestamp': datetime.now().isoformat(),
        'constants': {
            'xi': XI,
            'xi_minus_1': XI - 1,
            'phi': PHI
        },
        'results': results
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_01_threshold_detector_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return output


if __name__ == '__main__':
    run_experiment()
