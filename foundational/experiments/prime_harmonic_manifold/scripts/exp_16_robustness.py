"""
Experiment 16: Robustness Across Parameters

Tests if key findings hold across different construction choices:
- Varying topK (vocabulary size): 10, 15, 20, 25, 30, 40, 50, 75, 100
- Varying chord length (n_gaps): 2, 3, 4

If Real >> Cramér holds across ALL parameter choices, result is robust.
If it only holds for specific parameters, we have an artifact.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import get_primes, PHI_INV
import numpy as np
import sympy as sp
from collections import Counter


def generate_cramer_primes(limit: int, seed: int = None) -> np.ndarray:
    """Generate Cramér random primes."""
    rng = np.random.default_rng(seed)
    primes = [2]
    for n in range(3, limit):
        if rng.random() < 1 / np.log(n):
            primes.append(n)
    return np.array(primes, dtype=float)


def compute_lambda1(primes, topK, n_gaps):
    """Compute λ₁ with given parameters."""
    if len(primes) < 100:
        return None
    
    gaps = np.diff(primes)
    
    # Build n-gap chords
    if len(gaps) < n_gaps + 1:
        return None
    
    chords = []
    for i in range(len(gaps) - n_gaps + 1):
        chord = tuple(gaps[i:i+n_gaps])
        chords.append(chord)
    
    if len(chords) < 10:
        return None
    
    # Count and build transition matrix
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
    
    eigenvals = np.abs(np.linalg.eigvals(P[:topK, :topK]))
    return float(np.max(eigenvals))


def run_experiment(prime_limit=500_000):
    """Test robustness across parameter space."""
    
    print("=" * 70)
    print("VALIDATION: Robustness Across Parameters")
    print("=" * 70)
    
    # Generate data
    print(f"\nGenerating primes up to {prime_limit:,}...")
    real_primes = get_primes(prime_limit)
    print(f"  Real primes: {len(real_primes):,}")
    
    n_cramer = 10
    cramer_primes_list = [generate_cramer_primes(prime_limit, seed=i) for i in range(n_cramer)]
    print(f"  Cramér trials: {n_cramer}")
    
    # Parameter grid
    topK_values = [10, 15, 20, 25, 30, 40, 50, 75, 100]
    n_gaps_values = [2, 3, 4]
    
    results = []
    
    print("\n" + "=" * 70)
    print("TESTING PARAMETER GRID")
    print("=" * 70)
    
    for n_gaps in n_gaps_values:
        print(f"\n--- n_gaps = {n_gaps} ---")
        print(f"  {'topK':<8} {'Real λ₁':<12} {'Cramér λ₁':<15} {'Z-score':<10} {'Significant'}")
        print("  " + "-" * 60)
        
        for topK in topK_values:
            # Real
            real_l1 = compute_lambda1(real_primes, topK, n_gaps)
            
            # Cramér
            cramer_l1s = []
            for cp in cramer_primes_list:
                l1 = compute_lambda1(cp, topK, n_gaps)
                if l1:
                    cramer_l1s.append(l1)
            
            if real_l1 and len(cramer_l1s) > 1:
                cramer_mean = np.mean(cramer_l1s)
                cramer_std = np.std(cramer_l1s)
                z_score = (real_l1 - cramer_mean) / cramer_std if cramer_std > 0 else 0
                significant = abs(z_score) > 2
                
                print(f"  {topK:<8} {real_l1:<12.4f} {cramer_mean:.4f} ± {cramer_std:.4f}  {z_score:<10.1f} {'YES' if significant else 'NO'}")
                
                results.append({
                    'n_gaps': n_gaps,
                    'topK': topK,
                    'real_lambda1': float(real_l1),
                    'cramer_mean': float(cramer_mean),
                    'cramer_std': float(cramer_std),
                    'z_score': float(z_score),
                    'significant': bool(significant),
                })
    
    # Summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)
    
    n_tests = len(results)
    n_significant = sum(1 for r in results if r['significant'])
    
    print(f"\n  Total parameter combinations tested: {n_tests}")
    print(f"  Significant (|z| > 2): {n_significant} ({n_significant/n_tests*100:.0f}%)")
    
    if n_significant == n_tests:
        conclusion = "FULLY_ROBUST"
        print("\n  ✅ RESULT: Real >> Cramér holds across ALL parameter choices")
        print("     The finding is ROBUST — not an artifact of construction.")
    elif n_significant > n_tests * 0.8:
        conclusion = "MOSTLY_ROBUST"
        print(f"\n  ⚠️ RESULT: Real >> Cramér holds for {n_significant}/{n_tests} combinations")
        print("     Finding is mostly robust with some parameter sensitivity.")
    else:
        conclusion = "PARAMETER_DEPENDENT"
        print(f"\n  ❌ RESULT: Only {n_significant}/{n_tests} combinations significant")
        print("     Finding may be an artifact of specific parameter choices.")
    
    # Check if real is always closer to 1/φ
    print("\n  Distance to 1/φ comparison:")
    real_closer_count = 0
    for r in results:
        real_dist = abs(r['real_lambda1'] - PHI_INV)
        cram_dist = abs(r['cramer_mean'] - PHI_INV)
        if real_dist < cram_dist:
            real_closer_count += 1
    
    print(f"    Real closer to 1/φ: {real_closer_count}/{n_tests} ({real_closer_count/n_tests*100:.0f}%)")
    
    # Save
    output = {
        'experiment': 'exp_16_robustness',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'prime_limit': prime_limit, 'n_cramer': n_cramer},
        'results': results,
        'summary': {
            'n_tests': n_tests,
            'n_significant': n_significant,
            'pct_significant': n_significant / n_tests * 100,
            'real_closer_to_phi': real_closer_count,
            'conclusion': conclusion,
        }
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_16_robustness_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return output


if __name__ == '__main__':
    run_experiment()
