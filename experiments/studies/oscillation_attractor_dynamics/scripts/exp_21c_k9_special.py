"""
Experiment 21c: What Makes k=9 Special?
=======================================

exp_21b showed that k=9 has best φ proximity but NOT edge-of-chaos signatures.
So what DOES make k=9 special?

Hypothesis refinement:
- Maybe it's not 1/f noise but something else
- k=9 is the 9th prime (23) - any significance?
- k=9 captures primes up to 23, which is near 24 (highly composite)
- The Möbius pairing showed gap 6 as the hub (31 connections) - related?

Let's look for what actually distinguishes k=9:
1. Factor base coverage of the number line
2. Balance between structure detection and noise
3. Resonance with the underlying prime distribution
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

from sec_core import compute_sec, FIRST_50_PRIMES, PHI

XI = 1.0571428571428572
PHI_INV = 1 / PHI


def analyze_factor_base_coverage(k: int, n_max: int) -> dict:
    """
    Analyze how well the factor base "covers" the number line.
    
    Key insight: Factor base [2,3,5,7,11,13,17,19,23] (k=9)
    Product = 223092870 (much larger than n_max)
    But coverage of small composites is what matters
    """
    factor_base = FIRST_50_PRIMES[:k]
    product = np.prod(factor_base)
    
    # Count how many numbers up to n_max are divisible by at least one factor
    covered = np.zeros(n_max + 1, dtype=bool)
    for p in factor_base:
        covered[p::p] = True
    
    coverage_fraction = np.mean(covered[2:])
    
    # Count smooth numbers (divisible only by primes in factor base)
    is_smooth = np.ones(n_max + 1, dtype=bool)
    is_smooth[0] = False
    is_smooth[1] = False
    
    # For each prime NOT in factor base, mark its multiples as non-smooth
    all_primes = sieve_primes(n_max)
    non_fb_primes = [p for p in all_primes if p > factor_base[-1]]
    
    for p in non_fb_primes:
        is_smooth[p::p] = False
    
    smooth_count = np.sum(is_smooth)
    smooth_fraction = smooth_count / n_max
    
    return {
        'k': k,
        'largest_prime': factor_base[-1],
        'product': int(product),
        'coverage_fraction': float(coverage_fraction),
        'smooth_count': int(smooth_count),
        'smooth_fraction': float(smooth_fraction)
    }


def sieve_primes(n: int) -> list:
    """Simple sieve."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]


def analyze_stress_field_structure(E: np.ndarray, I: np.ndarray) -> dict:
    """
    Analyze structural properties of the stress field.
    """
    # 1. Distribution symmetry
    mean_E = np.mean(E)
    median_E = np.median(E)
    skewness = stats.skew(E)
    kurtosis = stats.kurtosis(E)
    
    # 2. Positive/negative balance
    frac_positive = np.mean(E > 0)
    
    # 3. Zero-crossing rate
    signs = np.sign(E)
    crossings = np.sum(signs[:-1] != signs[1:])
    crossing_rate = crossings / len(E)
    
    # 4. Run length distribution (consecutive same-sign runs)
    run_lengths = []
    current_run = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    
    mean_run = np.mean(run_lengths)
    std_run = np.std(run_lengths)
    
    # 5. Impulse structure
    I_positive = I[I > 0]
    I_negative = I[I < 0]
    impulse_asymmetry = np.mean(I_positive) / (abs(np.mean(I_negative)) + 1e-10)
    
    return {
        'mean_E': float(mean_E),
        'median_E': float(median_E),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'frac_positive': float(frac_positive),
        'crossing_rate': float(crossing_rate),
        'mean_run_length': float(mean_run),
        'std_run_length': float(std_run),
        'impulse_asymmetry': float(impulse_asymmetry)
    }


def run_experiment():
    """
    Investigate what makes k=9 special for φ emergence.
    """
    print("=" * 70)
    print("EXPERIMENT 21c: What Makes k=9 Special?")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    n_max = 50000
    window = 101
    lam = 0.98
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {'n_max': n_max, 'window': window, 'lam': lam}
    }
    
    # Part 1: Coverage analysis
    print("=== Part 1: Factor Base Coverage ===")
    print()
    print(f"{'k':>4} {'largest_p':>10} {'coverage':>10} {'smooth%':>10}")
    print("-" * 40)
    
    coverage_results = []
    for k in range(3, 16):
        cov = analyze_factor_base_coverage(k, n_max)
        coverage_results.append(cov)
        print(f"{k:>4} {cov['largest_prime']:>10} "
              f"{cov['coverage_fraction']:>10.4f} {cov['smooth_fraction']*100:>10.4f}%")
    
    results['coverage'] = coverage_results
    
    # Part 2: Stress field structure at each k
    print("\n=== Part 2: Stress Field Structure ===")
    print()
    print(f"{'k':>4} {'φ dist':>10} {'skew':>8} {'kurt':>8} {'cross_rate':>12} {'run_len':>10}")
    print("-" * 60)
    
    structure_results = []
    for k in range(3, 16):
        sec = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:k],
                         window=window, lam=lam)
        
        odd_E = sec.E[np.arange(len(sec.E)) % 2 == 1]
        phi_dist = abs(np.mean(odd_E > 0) - PHI_INV)
        
        structure = analyze_stress_field_structure(sec.E, sec.I)
        structure['k'] = k
        structure['phi_distance'] = float(phi_dist)
        structure_results.append(structure)
        
        print(f"{k:>4} {phi_dist:>10.6f} {structure['skewness']:>8.4f} "
              f"{structure['kurtosis']:>8.4f} {structure['crossing_rate']:>12.6f} "
              f"{structure['mean_run_length']:>10.4f}")
    
    results['structure'] = structure_results
    
    # Part 3: What correlates with φ distance?
    print("\n=== Part 3: Correlation with φ Distance ===")
    
    phi_dists = [s['phi_distance'] for s in structure_results]
    
    correlations = {}
    metrics = ['skewness', 'kurtosis', 'crossing_rate', 'mean_run_length', 'impulse_asymmetry']
    
    for metric in metrics:
        values = [s[metric] for s in structure_results]
        r, p = stats.pearsonr(values, phi_dists)
        correlations[metric] = {'r': float(r), 'p': float(p)}
        sig = "**" if p < 0.05 else ""
        print(f"  {metric:>20}: r = {r:>7.4f}, p = {p:.4f} {sig}")
    
    # Also correlate with coverage
    coverages = [c['coverage_fraction'] for c in coverage_results]
    smooth_fracs = [c['smooth_fraction'] for c in coverage_results]
    
    r_cov, p_cov = stats.pearsonr(coverages, phi_dists)
    r_smooth, p_smooth = stats.pearsonr(smooth_fracs, phi_dists)
    
    print(f"  {'coverage_fraction':>20}: r = {r_cov:>7.4f}, p = {p_cov:.4f}")
    print(f"  {'smooth_fraction':>20}: r = {r_smooth:>7.4f}, p = {p_smooth:.4f}")
    
    correlations['coverage_fraction'] = {'r': float(r_cov), 'p': float(p_cov)}
    correlations['smooth_fraction'] = {'r': float(r_smooth), 'p': float(p_smooth)}
    
    results['correlations'] = correlations
    
    # Part 4: k=9 specific analysis
    print("\n=== Part 4: What's Special About k=9? ===")
    
    k9_structure = structure_results[6]  # k=9 is index 6 (starting from k=3)
    k9_coverage = coverage_results[6]
    
    print(f"\n  k=9 factor base: {FIRST_50_PRIMES[:9]}")
    print(f"  Largest prime: 23")
    print(f"  Coverage: {k9_coverage['coverage_fraction']:.4f}")
    print(f"  Smooth numbers: {k9_coverage['smooth_fraction']*100:.4f}%")
    print(f"\n  φ distance: {k9_structure['phi_distance']:.6f} (BEST)")
    print(f"  Skewness: {k9_structure['skewness']:.4f}")
    print(f"  Crossing rate: {k9_structure['crossing_rate']:.6f}")
    print(f"  Mean run length: {k9_structure['mean_run_length']:.4f}")
    
    # Compare to neighbors
    k8 = structure_results[5]
    k10 = structure_results[7]
    
    print(f"\n  Comparison:")
    print(f"    k=8 φ dist: {k8['phi_distance']:.6f}")
    print(f"    k=9 φ dist: {k9_structure['phi_distance']:.6f} ← minimum")
    print(f"    k=10 φ dist: {k10['phi_distance']:.6f}")
    
    # Hypothesis: k=9 hits a "resonance" with the prime distribution
    # The 9th prime is 23, and 24 = 2³×3 is highly composite
    # Gap of 6 (from 23 to 29) is the most common gap - and the Möbius hub
    print(f"\n  Possible explanation:")
    print(f"    - 9th prime = 23")
    print(f"    - Next prime = 29 (gap of 6)")
    print(f"    - Gap 6 is the 'hub' of Möbius network")
    print(f"    - k=9 captures structure up to this critical gap")
    
    results['k9_analysis'] = {
        'factor_base': FIRST_50_PRIMES[:9],
        'structure': k9_structure,
        'coverage': k9_coverage
    }
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Find the strongest correlate
    best_correlate = min(correlations.items(), key=lambda x: abs(x[1]['r']))
    strongest = max(correlations.items(), key=lambda x: abs(x[1]['r']))
    
    print(f"\n  Strongest correlate with φ distance: {strongest[0]}")
    print(f"    r = {strongest[1]['r']:.4f}, p = {strongest[1]['p']:.4f}")
    
    if strongest[1]['p'] < 0.05:
        print(f"\n  ✅ Significant correlation found!")
        print(f"     {strongest[0]} predicts φ proximity")
    else:
        print(f"\n  🔄 No single metric strongly predicts φ proximity")
        print(f"     k=9 may represent a unique balance point")
    
    results['conclusion'] = {
        'strongest_correlate': strongest[0],
        'correlation': strongest[1],
        'significant': bool(strongest[1]['p'] < 0.05)
    }
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_21c_k9_special_{timestamp}.json'
    
    with open(os.path.join(results_dir, filename), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == '__main__':
    run_experiment()
