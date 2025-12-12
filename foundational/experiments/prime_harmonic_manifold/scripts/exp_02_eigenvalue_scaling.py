"""
Experiment 02: Eigenvalue Scaling Test

Tests whether λ₁ ≈ 1/φ holds across different prime ranges,
from 10k to 10M primes.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import (
    get_primes, compute_gaps, extract_chords,
    build_transition_matrix, compute_eigenvalues, PHI_INV
)
import numpy as np


def run_experiment(top_k: int = 25):
    """Run scale invariance test."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: Scale Invariance Test")
    print("=" * 70)
    
    test_limits = [10000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000]
    
    print(f"\n{'Limit':<12} {'# Primes':<12} {'λ₁':<12} {'λ₂':<12} {'λ₁ vs 1/φ':<12}")
    print("-" * 60)
    
    results_list = []
    
    for lim in test_limits:
        primes = get_primes(lim)
        gaps = compute_gaps(primes)
        chords = extract_chords(gaps, n_gaps=2)
        P, _ = build_transition_matrix(chords, top_k=top_k)
        eigenvals = compute_eigenvalues(P[:top_k, :top_k])
        
        l1 = eigenvals[0] if len(eigenvals) > 0 else 0
        l2 = eigenvals[1] if len(eigenvals) > 1 else 0
        diff = l1 - PHI_INV
        
        print(f"{lim:<12} {len(primes):<12} {l1:<12.6f} {l2:<12.6f} {diff:+.6f}")
        
        results_list.append({
            'limit': lim,
            'n_primes': len(primes),
            'lambda1': l1,
            'lambda2': l2,
            'diff_phi_inv': diff,
        })
    
    # Compute statistics
    lambda1_values = [r['lambda1'] for r in results_list]
    mean_l1 = np.mean(lambda1_values)
    std_l1 = np.std(lambda1_values)
    
    print(f"\nReference: 1/φ = {PHI_INV:.6f}")
    print(f"\nλ₁ statistics across scales:")
    print(f"  Mean: {mean_l1:.6f}")
    print(f"  Std:  {std_l1:.6f}")
    print(f"  Mean vs 1/φ: {abs(mean_l1 - PHI_INV):.6f}")
    
    # Save results
    results = {
        'experiment': 'exp_02_eigenvalue_scaling',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'top_k': top_k},
        'results': {
            'scale_tests': results_list,
            'mean_lambda1': mean_l1,
            'std_lambda1': std_l1,
            'mean_vs_phi_inv': abs(mean_l1 - PHI_INV),
        },
        'conclusion': 'SCALE_INVARIANT' if abs(mean_l1 - PHI_INV) < 0.01 else 'VARIABLE'
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_02_eigenvalue_scaling_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()
