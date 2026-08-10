"""
Experiment 09: Local Entropy and SEC Phase Analysis

Tests SEC phase threshold connection:
- Local chord entropy evolution with scale
- Entropy peaks and valleys
- Threshold behavior
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import get_primes, compute_gaps, PHI_INV
from analysis import compute_local_entropy, rolling_std
import numpy as np


def run_experiment(prime_limit: int = 500000):
    """Run local entropy and SEC phase analysis."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: Local Entropy & SEC Phase Analysis")
    print("=" * 70)
    
    # Generate data
    print(f"\nGenerating primes up to {prime_limit:,}...")
    primes = get_primes(prime_limit)
    gaps = compute_gaps(primes)
    print(f"  Primes: {len(primes):,}")
    print(f"  Gaps: {len(gaps):,}")
    
    # Local entropy
    print("\n" + "-" * 60)
    print("LOCAL CHORD ENTROPY")
    print("-" * 60)
    
    entropies, positions = compute_local_entropy(gaps, window=50)
    
    print(f"  Windows computed: {len(entropies)}")
    print(f"  Mean entropy: {np.mean(entropies):.4f} bits")
    print(f"  Std entropy: {np.std(entropies):.4f}")
    print(f"  Min entropy: {np.min(entropies):.4f}")
    print(f"  Max entropy: {np.max(entropies):.4f}")
    
    # Rolling curvature (gap volatility)
    print("\n" + "-" * 60)
    print("LOCAL CURVATURE (Gap Volatility)")
    print("-" * 60)
    
    curvature = rolling_std(gaps, window=50)
    
    print(f"  Mean curvature: {np.mean(curvature):.4f}")
    print(f"  Std curvature: {np.std(curvature):.4f}")
    print(f"  Min curvature: {np.min(curvature):.4f}")
    print(f"  Max curvature: {np.max(curvature):.4f}")
    
    # Normalized gap curvature
    logs = np.log(primes[:-1])
    norm_gaps = gaps / logs
    norm_curvature = rolling_std(norm_gaps, window=50)
    
    print(f"\n  Normalized gap curvature:")
    print(f"    Mean: {np.mean(norm_curvature):.4f}")
    print(f"    Std: {np.std(norm_curvature):.4f}")
    
    # Entropy-curvature correlation
    min_len = min(len(entropies), len(curvature))
    correlation = np.corrcoef(entropies[:min_len], curvature[:min_len])[0, 1]
    print(f"\n  Entropy-curvature correlation: {correlation:.4f}")
    
    # SEC phase threshold check
    print("\n" + "-" * 60)
    print("SEC PHASE THRESHOLD CHECK")
    print("-" * 60)
    
    # Normalize position by log scale
    mid_positions = np.arange(len(entropies)) / len(entropies)
    
    # Check entropy at φ-related positions
    phi_positions = [PHI_INV, 1 - PHI_INV, 0.5]
    for pos in phi_positions:
        idx = int(pos * len(entropies))
        if idx < len(entropies):
            print(f"  Entropy at position {pos:.3f}: {entropies[idx]:.4f} bits")
    
    # Save results
    results = {
        'experiment': 'exp_09_local_entropy',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'prime_limit': prime_limit, 'window': 50},
        'results': {
            'entropy_stats': {
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
                'min': float(np.min(entropies)),
                'max': float(np.max(entropies)),
            },
            'curvature_stats': {
                'mean': float(np.mean(curvature)),
                'std': float(np.std(curvature)),
            },
            'norm_curvature_stats': {
                'mean': float(np.mean(norm_curvature)),
                'std': float(np.std(norm_curvature)),
            },
            'entropy_curvature_correlation': float(correlation),
        },
        'conclusion': 'STRUCTURED' if np.std(entropies) > 0.1 else 'UNIFORM'
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_09_local_entropy_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()
