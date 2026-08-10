"""
Experiment 15: Special Prime Subsequences

Tests λ₁ behavior for special classes of primes:
- Twin primes (p, p+2)
- Sophie Germain primes (p where 2p+1 is also prime)
- Cousin primes (p, p+4)
- Sexy primes (p, p+6)
- Safe primes (p where (p-1)/2 is also prime)

Key question: Do these special primes show different φ-structure?
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
import sympy as sp
from collections import Counter


def is_prime(n):
    """Check if n is prime."""
    return sp.isprime(n)


def get_twin_primes(limit):
    """Get smaller primes from twin prime pairs (p, p+2)."""
    primes = []
    for p in sp.primerange(2, limit):
        if is_prime(p + 2):
            primes.append(p)
    return np.array(primes, dtype=float)


def get_sophie_germain_primes(limit):
    """Get Sophie Germain primes p where 2p+1 is also prime."""
    primes = []
    for p in sp.primerange(2, limit):
        if is_prime(2 * p + 1):
            primes.append(p)
    return np.array(primes, dtype=float)


def get_cousin_primes(limit):
    """Get smaller primes from cousin prime pairs (p, p+4)."""
    primes = []
    for p in sp.primerange(2, limit):
        if is_prime(p + 4):
            primes.append(p)
    return np.array(primes, dtype=float)


def get_sexy_primes(limit):
    """Get smaller primes from sexy prime pairs (p, p+6)."""
    primes = []
    for p in sp.primerange(2, limit):
        if is_prime(p + 6):
            primes.append(p)
    return np.array(primes, dtype=float)


def get_safe_primes(limit):
    """Get safe primes p where (p-1)/2 is also prime."""
    primes = []
    for p in sp.primerange(5, limit):
        if (p - 1) % 2 == 0 and is_prime((p - 1) // 2):
            primes.append(p)
    return np.array(primes, dtype=float)


def analyze_prime_sequence(primes, name, topK=25):
    """Analyze eigenvalues for a prime sequence."""
    if len(primes) < 100:
        return None
    
    gaps = np.diff(primes)
    
    if len(gaps) < 10:
        return None
    
    # Extract chords
    g1, g2 = gaps[:-1], gaps[1:]
    chords = [(g1[i], g2[i]) for i in range(len(g1))]
    
    if len(chords) < 10:
        return None
    
    # Build transition matrix
    counts = Counter(chords)
    top_chords = [c for c, _ in counts.most_common(topK)]
    chord_to_idx = {c: i for i, c in enumerate(top_chords)}
    
    seq_idx = [chord_to_idx.get(c, topK) for c in chords]
    
    T = np.zeros((topK+1, topK+1), dtype=int)
    for a, b in zip(seq_idx[:-1], seq_idx[1:]):
        T[a, b] += 1
    
    P = T.astype(float)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P /= row_sums
    
    # Eigenvalues
    eigenvals = np.abs(np.linalg.eigvals(P[:topK, :topK]))
    eigenvals = np.sort(eigenvals)[::-1]
    
    return {
        'name': name,
        'n_primes': len(primes),
        'n_gaps': len(gaps),
        'n_unique_chords': len(counts),
        'lambda1': float(eigenvals[0]),
        'lambda2': float(eigenvals[1]) if len(eigenvals) > 1 else 0,
        'eigenvalues_top5': eigenvals[:5].tolist(),
        'gap_mean': float(np.mean(gaps)),
        'gap_std': float(np.std(gaps)),
        'gap_min': int(np.min(gaps)),
        'gap_max': int(np.max(gaps)),
        'top_chords': [(list(c), int(ct)) for c, ct in counts.most_common(5)],
    }


def run_experiment(prime_limit=2_000_000):
    """Run special prime subsequence analysis."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: Special Prime Subsequences")
    print("=" * 70)
    print(f"Searching primes up to {prime_limit:,}")
    
    # Regular primes for comparison
    print("\nGenerating prime sequences...")
    
    sequences = [
        ("All Primes", get_primes(prime_limit)),
        ("Twin Primes", get_twin_primes(prime_limit)),
        ("Sophie Germain", get_sophie_germain_primes(prime_limit)),
        ("Cousin Primes", get_cousin_primes(prime_limit)),
        ("Sexy Primes", get_sexy_primes(prime_limit)),
        ("Safe Primes", get_safe_primes(prime_limit)),
    ]
    
    for name, primes in sequences:
        print(f"  {name}: {len(primes):,} primes")
    
    # Analyze each
    print("\n" + "-" * 70)
    print("EIGENVALUE ANALYSIS")
    print("-" * 70)
    
    results = []
    
    print(f"\n  {'Sequence':<18} {'N':<10} {'Unique':<8} {'λ₁':<10} {'λ₂':<10} {'Δ from 1/φ':<12}")
    print("  " + "-" * 70)
    
    for name, primes in sequences:
        data = analyze_prime_sequence(primes, name)
        if data:
            results.append(data)
            diff = data['lambda1'] - PHI_INV
            print(f"  {name:<18} {data['n_primes']:<10,} {data['n_unique_chords']:<8} "
                  f"{data['lambda1']:<10.6f} {data['lambda2']:<10.6f} {diff:+.6f}")
    
    # Gap statistics
    print("\n" + "-" * 70)
    print("GAP STATISTICS")
    print("-" * 70)
    
    print(f"\n  {'Sequence':<18} {'Gap Mean':<12} {'Gap Std':<12} {'Gap Min':<10} {'Gap Max':<10}")
    print("  " + "-" * 55)
    
    for r in results:
        print(f"  {r['name']:<18} {r['gap_mean']:<12.2f} {r['gap_std']:<12.2f} "
              f"{r['gap_min']:<10} {r['gap_max']:<10}")
    
    # Most common chords
    print("\n" + "-" * 70)
    print("TOP CHORDS BY SEQUENCE")
    print("-" * 70)
    
    for r in results:
        print(f"\n  {r['name']}:")
        for chord, count in r['top_chords']:
            print(f"    {chord}: {count:,}")
    
    # Key comparisons
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    all_primes = results[0]
    special = results[1:]  # Twin, Sophie Germain, etc.
    
    print(f"\n  Reference: All Primes λ₁ = {all_primes['lambda1']:.6f}")
    print(f"  Reference: 1/φ = {PHI_INV:.6f}")
    
    # Sort by closeness to 1/φ
    sorted_by_phi = sorted(results, key=lambda r: abs(r['lambda1'] - PHI_INV))
    
    print(f"\n  Ranked by closeness to 1/φ:")
    for i, r in enumerate(sorted_by_phi):
        dist = abs(r['lambda1'] - PHI_INV)
        print(f"    {i+1}. {r['name']}: {r['lambda1']:.6f} (dist = {dist:.6f})")
    
    # Check if any special sequence is closer to φ
    closest = sorted_by_phi[0]
    if closest['name'] != 'All Primes':
        print(f"\n  💡 {closest['name']} are CLOSER to 1/φ than all primes!")
    
    # Check for unique structure
    print("\n  Structural uniqueness (unique chords / count):")
    for r in sorted(results, key=lambda x: x['n_unique_chords'] / x['n_primes']):
        ratio = r['n_unique_chords'] / r['n_primes'] * 100
        print(f"    {r['name']}: {ratio:.2f}% unique")
    
    # Save results
    output = {
        'experiment': 'exp_15_special_subsequences',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'prime_limit': prime_limit},
        'results': results,
        'rankings': {
            'by_phi_distance': [(r['name'], float(abs(r['lambda1'] - PHI_INV))) for r in sorted_by_phi],
        },
        'phi_inv': PHI_INV,
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_15_special_subsequences_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return output


if __name__ == '__main__':
    run_experiment()
