"""
Experiment 07: Fibonacci Connection Test

Tests for Fibonacci structure in prime gaps:
- What fraction of gaps are Fibonacci numbers?
- How many consecutive gap ratios are near φ or 1/φ?
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import get_primes, compute_gaps, PHI, PHI_INV
from analysis import check_fibonacci_gaps, check_consecutive_ratios_near_phi
import numpy as np


def run_experiment(prime_limit: int = 500000):
    """Run Fibonacci connection test."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: Fibonacci Connection Test")
    print("=" * 70)
    
    # Generate data
    print(f"\nGenerating primes up to {prime_limit:,}...")
    primes = get_primes(prime_limit)
    gaps = compute_gaps(primes)
    print(f"  Primes: {len(primes):,}")
    print(f"  Gaps: {len(gaps):,}")
    
    # Fibonacci gap analysis
    print("\n" + "-" * 60)
    print("FIBONACCI GAP ANALYSIS")
    print("-" * 60)
    
    fib_results = check_fibonacci_gaps(gaps)
    print(f"  Fibonacci numbers: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...")
    print(f"  Fraction of gaps that are Fibonacci: {fib_results['fib_fraction']:.4f}")
    print(f"  Expected from gap distribution: {fib_results['expected_fraction']:.4f}")
    print(f"  Enrichment: {fib_results['enrichment']:.2f}x")
    
    # Consecutive ratio analysis
    print("\n" + "-" * 60)
    print("CONSECUTIVE GAP RATIOS NEAR φ")
    print("-" * 60)
    
    tolerances = [0.05, 0.1, 0.15, 0.2]
    ratio_results = []
    
    for tol in tolerances:
        result = check_consecutive_ratios_near_phi(gaps, tolerance=tol)
        print(f"  Within ±{tol:.2f} of φ or 1/φ: {result['near_phi_fraction']*100:.2f}%")
        ratio_results.append(result)
    
    # What would random give?
    print(f"\n  Expected if random (uniform on [0,4]): ~{2*0.1/4*100:.1f}% for ±0.1")
    
    # Gap ratio distribution
    print("\n" + "-" * 60)
    print("GAP RATIO STATISTICS")
    print("-" * 60)
    
    ratios = []
    for i in range(len(gaps) - 1):
        if gaps[i] > 0:
            ratios.append(gaps[i+1] / gaps[i])
    
    ratios = np.array(ratios)
    print(f"  Mean ratio: {np.mean(ratios):.4f}")
    print(f"  Median ratio: {np.median(ratios):.4f}")
    print(f"  Std ratio: {np.std(ratios):.4f}")
    print(f"  φ = {PHI:.4f}, 1/φ = {PHI_INV:.4f}")
    
    # Save results
    results = {
        'experiment': 'exp_07_fibonacci_connection',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'prime_limit': prime_limit},
        'results': {
            'fibonacci': fib_results,
            'ratio_analysis': ratio_results,
            'ratio_stats': {
                'mean': float(np.mean(ratios)),
                'median': float(np.median(ratios)),
                'std': float(np.std(ratios)),
            },
        },
        'conclusion': 'FIBONACCI_ENRICHED' if fib_results['enrichment'] > 1.1 else 'NO_ENRICHMENT'
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_07_fibonacci_connection_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()
