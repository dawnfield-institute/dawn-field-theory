"""
Experiment 08: PAC Depth and Autocorrelation Test

Tests PAC hierarchy predictions:
- Gap autocorrelation decay
- Decorrelation length vs PAC depth
- φ-decay signature
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import get_primes, compute_gaps, PHI, PHI_INV
from analysis import gap_autocorrelation, find_decorrelation_length, pac_depth_estimate
import numpy as np


def run_experiment(prime_limit: int = 500000):
    """Run PAC depth and autocorrelation test."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: PAC Depth & Autocorrelation Test")
    print("=" * 70)
    
    # Generate data
    print(f"\nGenerating primes up to {prime_limit:,}...")
    primes = get_primes(prime_limit)
    gaps = compute_gaps(primes)
    print(f"  Primes: {len(primes):,}")
    print(f"  Gaps: {len(gaps):,}")
    
    # Autocorrelation
    print("\n" + "-" * 60)
    print("GAP AUTOCORRELATION")
    print("-" * 60)
    
    acf = gap_autocorrelation(gaps, max_lag=30)
    decorr_len = find_decorrelation_length(acf)
    
    print(f"  ACF at lag 1: {acf[1]:.4f}")
    print(f"  ACF at lag 2: {acf[2]:.4f}")
    print(f"  ACF at lag 5: {acf[5]:.4f}")
    print(f"  ACF at lag 10: {acf[10]:.4f}")
    print(f"  Decorrelation length (ACF < 1/e): {decorr_len} steps")
    
    # PAC depth prediction
    print("\n" + "-" * 60)
    print("PAC DEPTH ANALYSIS")
    print("-" * 60)
    
    pac_depth = pac_depth_estimate(primes)
    print(f"  Median prime: {np.median(primes):.0f}")
    print(f"  PAC depth (log_φ): {pac_depth:.2f}")
    print(f"  Ratio decorr_length/depth: {decorr_len/pac_depth:.4f}")
    
    # Compare ACF decay to φ^(-k)
    print("\n" + "-" * 60)
    print("φ-DECAY COMPARISON")
    print("-" * 60)
    
    print(f"  {'Lag':<6} {'ACF':<12} {'φ^(-k)':<12} {'Ratio':<12}")
    print("  " + "-" * 42)
    
    decay_ratios = []
    for k in range(1, 11):
        theoretical = 1 / PHI**k
        if acf[k] > 0.001:
            ratio = acf[k] / theoretical
            decay_ratios.append(ratio)
            print(f"  {k:<6} {acf[k]:<12.4f} {theoretical:<12.4f} {ratio:<12.4f}")
    
    mean_decay_ratio = np.mean(decay_ratios) if decay_ratios else 0
    print(f"\n  Mean ACF/φ^(-k) ratio: {mean_decay_ratio:.4f}")
    
    # Save results
    results = {
        'experiment': 'exp_08_pac_depth',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'prime_limit': prime_limit},
        'results': {
            'acf': [float(x) for x in acf],
            'decorrelation_length': int(decorr_len),
            'pac_depth': float(pac_depth),
            'depth_ratio': float(decorr_len / pac_depth),
            'phi_decay_match': float(mean_decay_ratio),
        },
        'conclusion': 'PHI_DECAY_CONFIRMED' if 0.5 < mean_decay_ratio < 2.0 else 'DECAY_DIFFERS'
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_08_pac_depth_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()
