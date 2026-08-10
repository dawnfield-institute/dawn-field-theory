"""
Experiment 10: Large-Scale Asymptotic Test

Tests λ₁ behavior at 10⁸+ primes to determine asymptotic convergence.
Key question: Does λ₁ stabilize at 1/φ, oscillate, or continue drifting?
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import PHI_INV
import numpy as np
import sympy as sp
from collections import Counter


def compute_lambda1_for_range(prime_limit, topK=25):
    """Compute leading eigenvalue for primes up to prime_limit."""
    primes_list = list(sp.primerange(2, prime_limit))
    if len(primes_list) < 100:
        return None, len(primes_list), None
    
    primes = np.array(primes_list, dtype=float)
    gaps = np.diff(primes)
    
    # Build chords
    g1 = gaps[:-1]
    g2 = gaps[1:]
    chords = [tuple([g1[i], g2[i]]) for i in range(len(g1))]
    
    # Count and get top chords
    counts = Counter(chords)
    top_chords = [c for c, _ in counts.most_common(topK)]
    chord_to_idx = {c: i for i, c in enumerate(top_chords)}
    
    # Build sequence
    seq_idx = [chord_to_idx.get(c, topK) for c in chords]
    
    # Transition matrix
    T = np.zeros((topK+1, topK+1), dtype=int)
    for a, b in zip(seq_idx[:-1], seq_idx[1:]):
        T[a, b] += 1
    
    P = T.astype(float)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P /= row_sums
    
    # Eigenvalues
    eigenvals = np.abs(np.linalg.eigvals(P[:topK, :topK]))
    eigenvals_sorted = np.sort(eigenvals)[::-1]
    
    return eigenvals_sorted[:5], len(primes_list), len(counts)


def run_experiment():
    """Run large-scale asymptotic test."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: Large-Scale Asymptotic Test")
    print("=" * 70)
    print(f"Testing λ₁ convergence up to 10⁸ primes")
    print(f"Reference: 1/φ = {PHI_INV:.6f}")
    
    # Extended test limits - go to 100 million
    test_limits = [
        10_000, 50_000, 100_000, 200_000, 500_000,
        1_000_000, 2_000_000, 5_000_000, 10_000_000,
        20_000_000, 50_000_000, 100_000_000
    ]
    
    print(f"\n{'Limit':<14} {'# Primes':<12} {'Unique':<8} {'λ₁':<12} {'λ₂':<10} {'λ₁ - 1/φ':<12} {'Time':<8}")
    print("-" * 80)
    
    results_list = []
    
    for lim in test_limits:
        start = time.time()
        try:
            eigenvals, n_primes, n_unique = compute_lambda1_for_range(lim)
            elapsed = time.time() - start
            
            if eigenvals is not None:
                l1 = eigenvals[0]
                l2 = eigenvals[1] if len(eigenvals) > 1 else 0
                diff = l1 - PHI_INV
                
                print(f"{lim:<14,} {n_primes:<12,} {n_unique:<8} {l1:<12.6f} {l2:<10.6f} {diff:+.6f}   {elapsed:.1f}s")
                
                results_list.append({
                    'limit': lim,
                    'n_primes': n_primes,
                    'n_unique_chords': n_unique,
                    'lambda1': float(l1),
                    'lambda2': float(l2),
                    'diff_phi_inv': float(diff),
                    'time_seconds': elapsed,
                })
        except MemoryError:
            print(f"{lim:<14,} MEMORY ERROR - skipping")
            break
        except Exception as e:
            print(f"{lim:<14,} ERROR: {e}")
            break
    
    if len(results_list) < 2:
        print("\nInsufficient data for analysis")
        return
    
    # Analysis
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)
    
    lambda1_vals = [r['lambda1'] for r in results_list]
    limits = [r['limit'] for r in results_list]
    log_limits = np.log10(limits)
    
    # Statistics
    print(f"\nλ₁ statistics across all scales:")
    print(f"  Mean: {np.mean(lambda1_vals):.6f}")
    print(f"  Std:  {np.std(lambda1_vals):.6f}")
    print(f"  Min:  {min(lambda1_vals):.6f} at N={limits[lambda1_vals.index(min(lambda1_vals))]:,}")
    print(f"  Max:  {max(lambda1_vals):.6f} at N={limits[lambda1_vals.index(max(lambda1_vals))]:,}")
    
    # Linear fit in log-space
    from scipy.optimize import curve_fit
    
    def linear_log(log_n, a, b):
        return a * log_n + b
    
    try:
        popt, _ = curve_fit(linear_log, log_limits, lambda1_vals)
        a, b = popt
        
        print(f"\n  Linear fit: λ₁ = {a:.4f} * log₁₀(N) + {b:.4f}")
        
        # Find crossing points
        crossing_phi = (PHI_INV - b) / a if a != 0 else float('inf')
        crossing_half = (0.5 - b) / a if a != 0 else float('inf')
        
        print(f"\n  Predicted crossings:")
        print(f"    λ₁ = 1/φ at N ≈ 10^{crossing_phi:.2f}")
        print(f"    λ₁ = 0.5 at N ≈ 10^{crossing_half:.2f}")
        
        # Extrapolations
        print(f"\n  Extrapolations:")
        for exp in [9, 10, 12, 15, 20]:
            pred = linear_log(exp, a, b)
            print(f"    N = 10^{exp}: λ₁ → {pred:.4f}")
        
        # Asymptotic limit
        if a < 0:
            asymp_limit = "0 (λ₁ → 0 as N → ∞)"
        elif a > 0:
            asymp_limit = "1 (λ₁ → 1 as N → ∞)"
        else:
            asymp_limit = f"{b:.4f} (constant)"
        print(f"\n  Asymptotic behavior: {asymp_limit}")
        
    except Exception as e:
        print(f"\n  Fit failed: {e}")
        a, b = 0, np.mean(lambda1_vals)
    
    # Check if mean equals 1/φ
    mean_l1 = np.mean(lambda1_vals)
    print(f"\n" + "-" * 70)
    print(f"KEY RESULT: Mean λ₁ = {mean_l1:.6f}")
    print(f"            1/φ     = {PHI_INV:.6f}")
    print(f"            Error   = {abs(mean_l1 - PHI_INV):.6f} ({abs(mean_l1 - PHI_INV)/PHI_INV*100:.3f}%)")
    
    # Save results
    results = {
        'experiment': 'exp_10_asymptotic_test',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'topK': 25, 'max_limit': max(limits)},
        'results': {
            'scale_tests': results_list,
            'mean_lambda1': float(mean_l1),
            'std_lambda1': float(np.std(lambda1_vals)),
            'fit_slope': float(a),
            'fit_intercept': float(b),
            'phi_inv': PHI_INV,
            'mean_vs_phi_inv_error': float(abs(mean_l1 - PHI_INV)),
        },
        'conclusion': 'MEAN_EQUALS_PHI_INV' if abs(mean_l1 - PHI_INV) < 0.01 else 'DRIFTING'
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_10_asymptotic_test_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()
