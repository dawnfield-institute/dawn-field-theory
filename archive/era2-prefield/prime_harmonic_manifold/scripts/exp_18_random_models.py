"""
Experiment 18: Alternative Random Models

Tests the Real vs Random gap against multiple null models:
1. Cramér model (1/log(n) probability)
2. Poisson gaps with matched mean
3. Gaussian gaps with matched mean/std
4. Shuffled real gaps (permutation test)
5. Geometric gaps with matched mean

If ALL random models give lower λ₁, the structure is truly intrinsic.
If only Cramér differs, we may have a model-specific artifact.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import get_primes, PHI_INV
import numpy as np
from collections import Counter


def compute_lambda1_from_gaps(gaps, topK=25):
    """Compute λ₁ from gap sequence."""
    if len(gaps) < 10:
        return None, 0
    
    gaps = np.array(gaps)
    g1, g2 = gaps[:-1], gaps[1:]
    chords = [tuple([g1[i], g2[i]]) for i in range(len(g1))]
    
    counts = Counter(chords)
    n_unique = len(counts)
    
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
    return float(np.max(eigenvals)), n_unique


def generate_cramer_gaps(n_gaps, mean_gap, seed=None):
    """Generate gaps from Cramér model."""
    rng = np.random.default_rng(seed)
    
    # Cramér: gaps are approximately exponential with mean ~ log(n)
    # For simplicity, use exponential with matched mean
    gaps = rng.exponential(scale=mean_gap, size=n_gaps)
    
    # Round to even integers (prime gaps > 2 are even)
    gaps = np.round(gaps / 2) * 2
    gaps = np.maximum(gaps, 2)
    
    return gaps.astype(int)


def generate_poisson_gaps(n_gaps, mean_gap, seed=None):
    """Generate gaps from Poisson distribution."""
    rng = np.random.default_rng(seed)
    gaps = rng.poisson(lam=mean_gap, size=n_gaps)
    gaps = np.round(gaps / 2) * 2
    gaps = np.maximum(gaps, 2)
    return gaps.astype(int)


def generate_gaussian_gaps(n_gaps, mean_gap, std_gap, seed=None):
    """Generate gaps from Gaussian distribution."""
    rng = np.random.default_rng(seed)
    gaps = rng.normal(loc=mean_gap, scale=std_gap, size=n_gaps)
    gaps = np.round(gaps / 2) * 2
    gaps = np.maximum(gaps, 2)
    return gaps.astype(int)


def generate_geometric_gaps(n_gaps, mean_gap, seed=None):
    """Generate gaps from geometric distribution."""
    rng = np.random.default_rng(seed)
    # Geometric with mean = mean_gap
    p = 1 / mean_gap
    gaps = rng.geometric(p=p, size=n_gaps)
    gaps = np.round(gaps / 2) * 2
    gaps = np.maximum(gaps, 2)
    return gaps.astype(int)


def shuffle_gaps(gaps, seed=None):
    """Randomly permute gap sequence."""
    rng = np.random.default_rng(seed)
    shuffled = gaps.copy()
    rng.shuffle(shuffled)
    return shuffled


def run_experiment(prime_limit=500_000, n_trials=20):
    """Test against multiple random models."""
    
    print("=" * 70)
    print("VALIDATION: Alternative Random Models")
    print("=" * 70)
    
    # Real primes
    print(f"\nGenerating real primes up to {prime_limit:,}...")
    real_primes = get_primes(prime_limit)
    real_gaps = np.diff(real_primes)
    
    n_gaps = len(real_gaps)
    mean_gap = np.mean(real_gaps)
    std_gap = np.std(real_gaps)
    
    print(f"  Primes: {len(real_primes):,}")
    print(f"  Gaps: {n_gaps:,}")
    print(f"  Mean gap: {mean_gap:.2f}")
    print(f"  Std gap: {std_gap:.2f}")
    
    # Real λ₁
    real_l1, real_unique = compute_lambda1_from_gaps(real_gaps)
    print(f"\n  Real λ₁ = {real_l1:.6f}")
    print(f"  Real unique chords = {real_unique}")
    
    # Test models
    models = [
        ("Cramér (exponential)", lambda seed: generate_cramer_gaps(n_gaps, mean_gap, seed)),
        ("Poisson", lambda seed: generate_poisson_gaps(n_gaps, mean_gap, seed)),
        ("Gaussian", lambda seed: generate_gaussian_gaps(n_gaps, mean_gap, std_gap, seed)),
        ("Geometric", lambda seed: generate_geometric_gaps(n_gaps, mean_gap, seed)),
        ("Shuffled Real", lambda seed: shuffle_gaps(real_gaps, seed)),
    ]
    
    results = []
    
    print("\n" + "-" * 70)
    print("MODEL COMPARISON")
    print("-" * 70)
    
    print(f"\n  {'Model':<22} {'λ₁ (mean±std)':<20} {'Unique':<12} {'Z-score':<10} {'Sig?'}")
    print("  " + "-" * 70)
    
    for model_name, generator in models:
        l1_samples = []
        unique_samples = []
        
        for seed in range(n_trials):
            gaps = generator(seed)
            l1, n_unique = compute_lambda1_from_gaps(gaps)
            if l1:
                l1_samples.append(l1)
                unique_samples.append(n_unique)
        
        if len(l1_samples) > 1:
            mean_l1 = np.mean(l1_samples)
            std_l1 = np.std(l1_samples)
            mean_unique = np.mean(unique_samples)
            
            z_score = (real_l1 - mean_l1) / std_l1 if std_l1 > 0 else 0
            significant = abs(z_score) > 2
            
            print(f"  {model_name:<22} {mean_l1:.4f} ± {std_l1:.4f}      {mean_unique:<12.0f} {z_score:<10.1f} {'YES' if significant else 'NO'}")
            
            results.append({
                'model': model_name,
                'mean_lambda1': float(mean_l1),
                'std_lambda1': float(std_l1),
                'mean_unique': float(mean_unique),
                'z_score': float(z_score),
                'significant': bool(significant),
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    n_models = len(results)
    n_significant = sum(1 for r in results if r['significant'])
    
    print(f"\n  Real λ₁ = {real_l1:.6f}")
    print(f"  1/φ = {PHI_INV:.6f}")
    print(f"  Real distance to 1/φ: {abs(real_l1 - PHI_INV):.6f}")
    
    print(f"\n  Models tested: {n_models}")
    print(f"  Real significantly different from: {n_significant}/{n_models}")
    
    # Check shuffled specifically
    shuffled_result = next(r for r in results if 'Shuffled' in r['model'])
    
    if shuffled_result['significant']:
        print("\n  ✅ CRITICAL: Real differs from SHUFFLED gaps!")
        print("     This proves the λ₁ structure depends on GAP ORDERING,")
        print("     not just the gap distribution itself.")
        order_matters = True
    else:
        print("\n  ⚠️ Real does NOT significantly differ from shuffled gaps.")
        print("     The structure may come from gap distribution, not ordering.")
        order_matters = False
    
    if n_significant == n_models:
        conclusion = "ALL_MODELS_DIFFER"
        print("\n  ✅ CONCLUSION: Real primes differ from ALL random models!")
        print("     The structure is INTRINSIC and cannot be explained by:")
        for r in results:
            print(f"       - {r['model']}")
    elif n_significant > n_models * 0.5:
        conclusion = "MOST_MODELS_DIFFER"
        print(f"\n  ⚠️ CONCLUSION: Real differs from {n_significant}/{n_models} models.")
    else:
        conclusion = "FEW_MODELS_DIFFER"
        print(f"\n  ❌ CONCLUSION: Only {n_significant}/{n_models} models show difference.")
    
    # Save
    output = {
        'experiment': 'exp_18_random_models',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'prime_limit': prime_limit,
            'n_trials': n_trials,
            'n_gaps': n_gaps,
            'mean_gap': float(mean_gap),
            'std_gap': float(std_gap),
        },
        'results': {
            'real_lambda1': real_l1,
            'real_unique': real_unique,
            'models': results,
        },
        'summary': {
            'n_significant': n_significant,
            'n_models': n_models,
            'order_matters': order_matters,
            'conclusion': conclusion,
        },
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_18_random_models_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return output


if __name__ == '__main__':
    run_experiment()
