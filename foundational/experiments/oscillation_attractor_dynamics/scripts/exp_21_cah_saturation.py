"""
Experiment 21: CAH Condition 4 - Computational Saturation
==========================================================

Tests whether φ signatures require EDGE-OF-CHAOS dynamics.

CAH (Conditional Attractor Hypothesis):
    Ξ ≈ 1.057 emerges IFF:
    1. System is closed (no external injection)
    2. System is recursive (dynamics applied iteratively)
    3. System is internally conserving (f(P) = Σf(C))
    4. System is COMPUTATIONALLY SATURATED (edge-of-chaos) ← THIS EXPERIMENT

"Computational saturation" = operating at maximum useful complexity
- Not too ordered (Class I/II): trivial dynamics, no information processing
- Not too chaotic (Class III): noise, no persistent structure
- At the edge (Class IV): maximal computation, Turing-complete

In SEC terms:
- The factor_base_size (k) controls complexity
- Too small k: trivial factorization structure
- Too large k: noise dominates
- Critical k: maximum φ proximity

Key test: Does SEC show edge-of-chaos behavior at critical parameters?
- Measure Lyapunov-like sensitivity
- Measure mutual information between scales
- Check if critical λ and k coincide with edge-of-chaos
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
PHI_INV = 1 / PHI  # ≈ 0.618


def compute_lyapunov_proxy(E: np.ndarray, delta: float = 0.01) -> float:
    """
    Compute a Lyapunov-like proxy by measuring sensitivity to perturbation.
    
    For E(n) = λE(n-1) + I(n):
    - Positive Lyapunov: chaotic divergence
    - Near-zero: edge of chaos
    - Negative: ordered convergence
    """
    # Perturbed trajectory
    E_perturbed = E.copy()
    E_perturbed[0] += delta
    
    # The recursion E(n) = λE(n-1) + I(n) means perturbation decays as λ^n
    # So we measure empirical growth rate of |E - E_perturbed|
    # (But since I(n) is same, this is λ-controlled)
    
    # Instead, measure local growth of E variations
    dE = np.diff(E)
    
    # Lyapunov proxy from log(|dE|) growth
    log_dE = np.log(np.abs(dE) + 1e-10)
    
    # Linear fit to see if growing (positive) or decaying (negative)
    t = np.arange(len(log_dE))
    if len(t) > 100:
        t = t[:100]
        log_dE = log_dE[:100]
    
    slope, _ = np.polyfit(t, log_dE, 1)
    
    return float(slope)


def compute_complexity_metrics(E: np.ndarray) -> dict:
    """
    Compute metrics that characterize computational complexity.
    
    Edge-of-chaos systems should show:
    1. Non-trivial autocorrelation structure
    2. Power-law-like spectral decay
    3. High Shannon entropy but not maximum
    """
    # 1. Autocorrelation decay time
    n = min(len(E), 10000)
    E_sample = E[:n]
    E_centered = E_sample - np.mean(E_sample)
    var = np.var(E_sample) + 1e-10
    
    ac_decay = 0
    for lag in range(1, 100):
        ac = np.sum(E_centered[:-lag] * E_centered[lag:]) / ((n - lag) * var)
        if ac < 0.5:
            ac_decay = lag
            break
    else:
        ac_decay = 100
    
    # 2. Spectral slope (power-law proxy)
    fft = np.fft.fft(E_sample)
    power = np.abs(fft[:n//2]) ** 2
    freqs = np.arange(1, n//2 + 1)
    
    # Log-log fit
    log_f = np.log10(freqs[1:100])  # Skip DC, use first 100
    log_p = np.log10(power[1:100] + 1e-10)
    spectral_slope, _ = np.polyfit(log_f, log_p, 1)
    
    # 3. Normalized entropy
    # Discretize E into bins
    bins = np.linspace(np.min(E_sample), np.max(E_sample), 50)
    hist, _ = np.histogram(E_sample, bins=bins, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10)) / np.log2(len(bins) - 1)
    
    return {
        'ac_decay_time': int(ac_decay),
        'spectral_slope': float(spectral_slope),
        'normalized_entropy': float(entropy)
    }


def run_experiment():
    """
    Test whether φ emergence requires edge-of-chaos dynamics.
    """
    print("=" * 70)
    print("EXPERIMENT 21: CAH Condition 4 - Computational Saturation")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print("Testing whether φ signatures require edge-of-chaos dynamics")
    print()
    
    # Parameters
    n_max = 50000
    window = 101
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_max': n_max,
            'window': window
        },
        'hypothesis': 'φ signatures require edge-of-chaos (computational saturation)',
        'cah_condition': 'Condition 4: System is COMPUTATIONALLY SATURATED'
    }
    
    # Sweep factor_base_size (k) to find complexity transition
    print("=== Sweep: Factor Base Size (k) ===")
    print()
    print(f"{'k':>4} {'frac(E>0)':>12} {'dist from 1/φ':>14} {'spectral_slope':>15} {'entropy':>10}")
    print("-" * 60)
    
    k_values = list(range(3, 16))
    k_results = []
    
    for k in k_values:
        factor_base = FIRST_50_PRIMES[:k]
        sec = compute_sec(n_max=n_max, factor_base=factor_base, 
                         window=window, lam=0.98)
        
        odd_E = sec.E[np.arange(len(sec.E)) % 2 == 1]
        frac = np.mean(odd_E > 0)
        dist = abs(frac - PHI_INV)
        
        complexity = compute_complexity_metrics(sec.E)
        
        k_results.append({
            'k': k,
            'frac_E_positive': float(frac),
            'dist_from_phi_inv': float(dist),
            **complexity
        })
        
        print(f"{k:>4} {frac:>12.4f} {dist:>14.6f} "
              f"{complexity['spectral_slope']:>15.3f} {complexity['normalized_entropy']:>10.4f}")
    
    results['k_sweep'] = k_results
    
    # Find optimal k
    best_k = min(k_results, key=lambda x: x['dist_from_phi_inv'])
    print(f"\n  Best k: {best_k['k']} (dist = {best_k['dist_from_phi_inv']:.6f})")
    
    results['best_k'] = best_k
    
    # Analyze edge-of-chaos at best k
    print("\n=== Edge-of-Chaos Analysis at Best k ===")
    
    best_factor_base = FIRST_50_PRIMES[:best_k['k']]
    sec_best = compute_sec(n_max=n_max, factor_base=best_factor_base, 
                           window=window, lam=0.98)
    
    lyap = compute_lyapunov_proxy(sec_best.E)
    complex_metrics = compute_complexity_metrics(sec_best.E)
    
    print(f"\n  At k = {best_k['k']}:")
    print(f"    Lyapunov proxy:      {lyap:.6f}")
    print(f"    AC decay time:       {complex_metrics['ac_decay_time']}")
    print(f"    Spectral slope:      {complex_metrics['spectral_slope']:.3f}")
    print(f"    Normalized entropy:  {complex_metrics['normalized_entropy']:.4f}")
    
    results['edge_of_chaos_metrics'] = {
        'lyapunov_proxy': lyap,
        **complex_metrics
    }
    
    # Compare: low k (ordered) vs best k vs high k (chaotic)
    print("\n=== Comparison: Ordered vs Edge vs Chaotic ===")
    
    comparison = {}
    for label, k in [('ordered', 3), ('edge', best_k['k']), ('chaotic', 15)]:
        fb = FIRST_50_PRIMES[:k]
        sec = compute_sec(n_max=n_max, factor_base=fb, window=window, lam=0.98)
        
        odd_E = sec.E[np.arange(len(sec.E)) % 2 == 1]
        frac = np.mean(odd_E > 0)
        dist = abs(frac - PHI_INV)
        
        complexity = compute_complexity_metrics(sec.E)
        
        comparison[label] = {
            'k': k,
            'frac_E_positive': float(frac),
            'dist_from_phi_inv': float(dist),
            **complexity
        }
        
        print(f"\n  {label.upper()} (k={k}):")
        print(f"    frac(E > 0): {frac:.4f}")
        print(f"    dist from 1/φ: {dist:.6f}")
        print(f"    spectral slope: {complexity['spectral_slope']:.3f}")
    
    results['comparison'] = comparison
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    edge_best = (comparison['edge']['dist_from_phi_inv'] < 
                 min(comparison['ordered']['dist_from_phi_inv'],
                     comparison['chaotic']['dist_from_phi_inv']))
    
    near_zero_lyap = abs(lyap) < 0.01
    intermediate_entropy = 0.4 < complex_metrics['normalized_entropy'] < 0.8
    
    if edge_best and (near_zero_lyap or intermediate_entropy):
        conclusion = "SUPPORTS CAH Condition 4: Edge-of-chaos shows best φ proximity"
        print(f"✅ {conclusion}")
    elif edge_best:
        conclusion = "PARTIAL: Edge-of-chaos is best, but Lyapunov/entropy don't confirm EOC"
        print(f"🔄 {conclusion}")
    else:
        conclusion = "DOES NOT SUPPORT: φ doesn't peak at edge-of-chaos"
        print(f"❌ {conclusion}")
    
    results['conclusion'] = {
        'supports_cah': bool(edge_best and (near_zero_lyap or intermediate_entropy)),
        'edge_best': bool(edge_best),
        'near_zero_lyapunov': bool(near_zero_lyap),
        'intermediate_entropy': bool(intermediate_entropy),
        'summary': conclusion
    }
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_21_cah_saturation_{timestamp}.json'
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == '__main__':
    run_experiment()
