"""
exp_03_phi_threshold.py
=======================

Purpose: Detailed analysis of the φ-threshold phenomenon.

Key Questions:
1. At what exact threshold does E(n)>0 maximally separate primes?
2. Is 1/φ truly optimal or just coincidentally close?
3. How does the threshold behave under continuous parameter variation?

Traces:
- threshold_sweep: frac(E>threshold) vs threshold curve
- optimal_threshold: value that maximizes prime/composite separation
- phi_comparison: distance from 1/φ under various configurations
"""

import sys
import os
import argparse
import numpy as np
from typing import Dict, List, Any

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sec_core import compute_sec, FIRST_50_PRIMES as PRIMES


def sweep_thresholds(E: np.ndarray, is_prime_mask: np.ndarray, 
                     thresholds: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute prime fraction above each threshold."""
    prime_fracs = []
    composite_fracs = []
    separations = []
    
    for thresh in thresholds:
        above = E > thresh
        
        prime_rate = np.mean(is_prime_mask[above]) if np.any(above) else 0
        composite_rate = np.mean(~is_prime_mask[above]) if np.any(above) else 0
        
        # Fraction of primes that are above threshold
        prime_recall = np.sum(is_prime_mask & above) / np.sum(is_prime_mask) if np.sum(is_prime_mask) > 0 else 0
        
        # Fraction of composites that are above threshold
        composite_recall = np.sum(~is_prime_mask & above) / np.sum(~is_prime_mask) if np.sum(~is_prime_mask) > 0 else 0
        
        separation = prime_recall - composite_recall
        
        prime_fracs.append(prime_recall)
        composite_fracs.append(composite_recall)
        separations.append(separation)
    
    return {
        'thresholds': thresholds,
        'prime_recall': np.array(prime_fracs),
        'composite_recall': np.array(composite_fracs),
        'separation': np.array(separations)
    }


def find_optimal_threshold(E: np.ndarray, is_prime_mask: np.ndarray) -> Dict[str, float]:
    """Find threshold that maximizes prime/composite separation."""
    thresholds = np.linspace(np.min(E), np.max(E), 1000)
    result = sweep_thresholds(E, is_prime_mask, thresholds)
    
    idx = np.argmax(result['separation'])
    
    return {
        'optimal_threshold': float(thresholds[idx]),
        'max_separation': float(result['separation'][idx]),
        'prime_recall_at_optimal': float(result['prime_recall'][idx]),
        'composite_recall_at_optimal': float(result['composite_recall'][idx])
    }


def analyze_E_distribution(E: np.ndarray, is_prime_mask: np.ndarray) -> Dict[str, Any]:
    """Analyze the distribution of E for primes vs composites."""
    E_primes = E[is_prime_mask]
    E_composites = E[~is_prime_mask]
    
    return {
        'prime_E_mean': float(np.mean(E_primes)),
        'prime_E_std': float(np.std(E_primes)),
        'prime_E_median': float(np.median(E_primes)),
        'prime_E_q25': float(np.percentile(E_primes, 25)),
        'prime_E_q75': float(np.percentile(E_primes, 75)),
        'composite_E_mean': float(np.mean(E_composites)),
        'composite_E_std': float(np.std(E_composites)),
        'composite_E_median': float(np.median(E_composites)),
        'composite_E_q25': float(np.percentile(E_composites, 25)),
        'composite_E_q75': float(np.percentile(E_composites, 75)),
        'mean_difference': float(np.mean(E_primes) - np.mean(E_composites)),
        'median_difference': float(np.median(E_primes) - np.median(E_composites))
    }


def continuous_phi_test(n_max: int, lmbda: float = 0.99) -> Dict[str, Any]:
    """Test φ-threshold under continuous factor base size and window variation."""
    PHI_INV = 0.618033988749895
    
    # Test factor base sizes 1-25
    sizes = list(range(1, 26))
    size_results = []
    
    for size in sizes:
        fb = PRIMES[:size]
        sec = compute_sec(n_max=n_max, factor_base=fb, window=101, lam=lmbda)
        
        odds = np.arange(3, n_max + 1, 2)
        E_odds = sec.E[odds]
        
        frac_positive = float(np.mean(E_odds > 0))
        error = frac_positive - PHI_INV
        
        size_results.append({
            'size': size,
            'frac_E_positive': frac_positive,
            'error_vs_phi': error,
            'abs_error': abs(error)
        })
    
    # Find optimal size
    best_size_idx = np.argmin([r['abs_error'] for r in size_results])
    best_size = size_results[best_size_idx]
    
    # Test window sizes (odd values from 11 to 201)
    windows = list(range(11, 202, 10))
    window_results = []
    
    for window in windows:
        sec = compute_sec(n_max=n_max, factor_base=PRIMES[:10], window=window, lam=lmbda)
        
        odds = np.arange(3, n_max + 1, 2)
        E_odds = sec.E[odds]
        frac_positive = float(np.mean(E_odds > 0))
        error = frac_positive - PHI_INV
        
        window_results.append({
            'window': window,
            'frac_E_positive': frac_positive,
            'error_vs_phi': error,
            'abs_error': abs(error)
        })
    
    # Find optimal window
    best_window_idx = np.argmin([r['abs_error'] for r in window_results])
    best_window = window_results[best_window_idx]
    
    return {
        'size_sweep': size_results,
        'best_size': best_size,
        'window_sweep': window_results,
        'best_window': best_window
    }


def run_experiment(n_max: int, output_dir: str) -> Dict[str, Any]:
    """Run complete φ-threshold analysis."""
    PHI_INV = 0.618033988749895
    
    # Standard configuration
    sec = compute_sec(
        n_max=n_max,
        factor_base=PRIMES[:10],
        window=101,
        lam=0.99
    )
    
    # Build masks for odd numbers
    odds = np.arange(3, n_max + 1, 2)
    is_prime_mask = sec.prime_mask[odds]
    E_odds = sec.E[odds]
    
    # Threshold sweep
    thresholds = np.linspace(-5, 5, 500)
    sweep = sweep_thresholds(E_odds, is_prime_mask, thresholds)
    
    # Optimal threshold
    optimal = find_optimal_threshold(E_odds, is_prime_mask)
    
    # E distribution analysis
    distribution = analyze_E_distribution(E_odds, is_prime_mask)
    
    # Continuous φ test
    continuous = continuous_phi_test(n_max)
    
    # φ-threshold analysis
    frac_E_positive = float(np.mean(E_odds > 0))
    
    results = {
        'parameters': {
            'n_max': n_max,
            'factor_base': list(PRIMES[:10]),
            'window': 101,
            'lambda': 0.99
        },
        'phi_threshold': {
            'phi_inverse': PHI_INV,
            'frac_E_positive': frac_E_positive,
            'error': frac_E_positive - PHI_INV,
            'percent_error': 100 * (frac_E_positive - PHI_INV) / PHI_INV
        },
        'optimal_threshold': optimal,
        'E_distribution': distribution,
        'continuous_test': {
            'best_size': continuous['best_size'],
            'best_window': continuous['best_window'],
            'phi_is_achieved': (continuous['best_size']['abs_error'] < 0.001 or 
                               continuous['best_window']['abs_error'] < 0.001)
        },
        'validation': {
            'zero_is_near_optimal': abs(optimal['optimal_threshold']) < 1.0,
            'primes_have_higher_E': distribution['mean_difference'] > 0,
            'phi_within_1pct': abs(frac_E_positive - PHI_INV) / PHI_INV < 0.01,
            'continuous_phi_achieved': continuous['best_size']['abs_error'] < 0.001
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Experiment 03: φ-Threshold Analysis')
    parser.add_argument('--n_max', type=int, default=10000, help='Maximum n to test')
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("EXPERIMENT 03: φ-Threshold Analysis")
    print("=" * 70)
    print()
    
    results = run_experiment(args.n_max, output_dir)
    
    # Display results
    print("Parameters:")
    print(f"  n_max: {results['parameters']['n_max']}")
    print(f"  factor_base: {results['parameters']['factor_base']}")
    print(f"  window: {results['parameters']['window']}")
    print()
    
    print("-" * 70)
    print("φ-THRESHOLD ANALYSIS")
    print("-" * 70)
    print()
    
    phi = results['phi_threshold']
    print(f"  Target (1/φ):    {phi['phi_inverse']:.6f}")
    print(f"  Observed:        {phi['frac_E_positive']:.6f}")
    print(f"  Error:           {phi['error']:+.6f}")
    print(f"  Percent error:   {phi['percent_error']:+.4f}%")
    print()
    
    print("-" * 70)
    print("OPTIMAL THRESHOLD")
    print("-" * 70)
    print()
    
    opt = results['optimal_threshold']
    print(f"  Optimal threshold:        {opt['optimal_threshold']:.4f}")
    print(f"  Max separation:           {opt['max_separation']:.4f}")
    print(f"  Prime recall at optimal:  {opt['prime_recall_at_optimal']:.4f}")
    print(f"  Composite recall:         {opt['composite_recall_at_optimal']:.4f}")
    print()
    
    print("-" * 70)
    print("E DISTRIBUTION")  
    print("-" * 70)
    print()
    
    dist = results['E_distribution']
    print(f"  Prime E:     mean={dist['prime_E_mean']:+.4f}, std={dist['prime_E_std']:.4f}")
    print(f"  Composite E: mean={dist['composite_E_mean']:+.4f}, std={dist['composite_E_std']:.4f}")
    print(f"  Mean difference: {dist['mean_difference']:+.4f}")
    print()
    
    print("-" * 70)
    print("CONTINUOUS PARAMETER SEARCH")
    print("-" * 70)
    print()
    
    cont = results['continuous_test']
    print(f"  Best size:   {cont['best_size']['size']}")
    print(f"    frac(E>0): {cont['best_size']['frac_E_positive']:.6f}")
    print(f"    error:     {cont['best_size']['error_vs_phi']:+.6f}")
    print()
    print(f"  Best window: {cont['best_window']['window']}")
    print(f"    frac(E>0): {cont['best_window']['frac_E_positive']:.6f}")
    print(f"    error:     {cont['best_window']['error_vs_phi']:+.6f}")
    print()
    
    print("-" * 70)
    print("VALIDATION")
    print("-" * 70)
    print()
    
    for key, passed in results['validation'].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {key}: {status}")
    print()
    
    # Save trace
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = os.path.join(output_dir, f'exp_03_phi_threshold_{timestamp}.json')
    
    # Prepare serializable results
    save_results = {
        'parameters': results['parameters'],
        'phi_threshold': results['phi_threshold'],
        'optimal_threshold': results['optimal_threshold'],
        'E_distribution': results['E_distribution'],
        'continuous_test': {
            'best_size': results['continuous_test']['best_size'],
            'best_window': results['continuous_test']['best_window'],
            'phi_is_achieved': results['continuous_test']['phi_is_achieved']
        },
        'validation': results['validation']
    }
    
    with open(trace_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"Trace saved: {os.path.basename(trace_file)}")


if __name__ == '__main__':
    main()