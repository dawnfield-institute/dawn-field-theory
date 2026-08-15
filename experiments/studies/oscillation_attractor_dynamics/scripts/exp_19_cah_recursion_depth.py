"""
Experiment 19: CAH Condition 2 - Recursive Dynamics
====================================================

Tests whether φ signatures require RECURSIVE application of dynamics.

CAH (Conditional Attractor Hypothesis):
    Ξ ≈ 1.057 emerges IFF:
    1. System is closed (no external injection)
    2. System is RECURSIVE (dynamics applied iteratively) ← THIS EXPERIMENT
    3. System is internally conserving (f(P) = Σf(C))
    4. System is computationally saturated (edge-of-chaos)

In the oscillation_attractor_dynamics context:
- SEC stress field E(n) = λE(n-1) + I(n) is RECURSIVE by definition
- The λ parameter controls memory depth (recursion strength)
- λ → 0: No recursion (each E(n) independent)
- λ → 1: Deep recursion (long memory)

Key test: How does recursion depth (λ) affect φ emergence?
- At λ ≈ 0.98 (critical), φ signatures are strongest
- This suggests recursion is necessary for Ξ emergence
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

from sec_core import compute_sec, FIRST_50_PRIMES, PHI

XI = 1.0571428571428572
PHI_INV = 1 / PHI  # ≈ 0.618


def compute_recursion_metrics(E: np.ndarray, odd_only: bool = True) -> dict:
    """
    Compute metrics related to recursion depth in E(n).
    
    Key insight: E(n) = λE(n-1) + I(n)
    - Autocorrelation measures effective recursion depth
    - Spectral analysis reveals memory structure
    """
    if odd_only:
        E = E[np.arange(len(E)) % 2 == 1]
    
    # 1. Autocorrelation at various lags
    n = len(E)
    E_centered = E - np.mean(E)
    var = np.var(E)
    
    autocorr = []
    for lag in range(1, 21):
        if lag < n:
            corr = np.sum(E_centered[:-lag] * E_centered[lag:]) / ((n - lag) * var + 1e-10)
            autocorr.append(float(corr))
    
    # 2. Effective memory depth (lag where autocorr drops below 1/e)
    memory_depth = 1
    for i, ac in enumerate(autocorr):
        if ac < 1/np.e:
            memory_depth = i + 1
            break
    else:
        memory_depth = len(autocorr)
    
    # 3. Frac E > 0
    frac_positive = np.mean(E > 0)
    
    return {
        'autocorrelation': autocorr,
        'memory_depth': memory_depth,
        'frac_E_positive': float(frac_positive),
        'dist_from_phi_inv': float(abs(frac_positive - PHI_INV))
    }


def run_experiment():
    """
    Test how recursion depth (λ) affects φ emergence.
    """
    print("=" * 70)
    print("EXPERIMENT 19: CAH Condition 2 - Recursive Dynamics")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print("Testing whether φ signatures require recursive dynamics (high λ)")
    print()
    
    # Parameters
    n_max = 50000
    factor_base = FIRST_50_PRIMES[:9]
    window = 101
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_max': n_max,
            'factor_base_size': len(factor_base),
            'window': window
        },
        'hypothesis': 'φ signatures require recursive dynamics (high λ)',
        'cah_condition': 'Condition 2: System is RECURSIVE'
    }
    
    # Sweep λ from low (no recursion) to high (deep recursion)
    print("=== Lambda Sweep: Recursion Depth Analysis ===")
    print()
    print(f"{'λ':>8} {'frac(E>0)':>12} {'dist from 1/φ':>14} {'memory_depth':>14}")
    print("-" * 50)
    
    lambdas = np.linspace(0.0, 0.999, 20)
    lambda_results = []
    
    for lam in lambdas:
        sec = compute_sec(n_max=n_max, factor_base=factor_base, 
                         window=window, lam=lam)
        metrics = compute_recursion_metrics(sec.E, odd_only=True)
        
        lambda_results.append({
            'lambda': float(lam),
            **metrics
        })
        
        print(f"{lam:>8.4f} {metrics['frac_E_positive']:>12.4f} "
              f"{metrics['dist_from_phi_inv']:>14.6f} {metrics['memory_depth']:>14}")
    
    results['lambda_sweep'] = lambda_results
    
    # Find optimal λ
    print("\n=== Finding Critical λ (minimum distance to 1/φ) ===")
    
    def objective(lam):
        sec = compute_sec(n_max=n_max, factor_base=factor_base, 
                         window=window, lam=float(lam))
        odd_E = sec.E[np.arange(len(sec.E)) % 2 == 1]
        frac = np.mean(odd_E > 0)
        return abs(frac - PHI_INV)
    
    # Fine search near critical region
    lambdas_fine = np.linspace(0.95, 0.999, 50)
    best_lam = None
    best_dist = float('inf')
    
    for lam in lambdas_fine:
        dist = objective(lam)
        if dist < best_dist:
            best_dist = dist
            best_lam = lam
    
    print(f"\n  Critical λ*: {best_lam:.6f}")
    print(f"  Distance from 1/φ at λ*: {best_dist:.6f}")
    
    results['critical_lambda'] = {
        'lambda_star': float(best_lam),
        'min_distance': float(best_dist)
    }
    
    # Compare low vs high recursion
    print("\n=== Comparison: Low vs High Recursion ===")
    
    # Low recursion (λ = 0.1)
    sec_low = compute_sec(n_max=n_max, factor_base=factor_base, 
                          window=window, lam=0.1)
    metrics_low = compute_recursion_metrics(sec_low.E, odd_only=True)
    
    # High recursion (λ = 0.98)
    sec_high = compute_sec(n_max=n_max, factor_base=factor_base, 
                           window=window, lam=0.98)
    metrics_high = compute_recursion_metrics(sec_high.E, odd_only=True)
    
    print(f"\n  Low recursion (λ=0.1):")
    print(f"    frac(E>0): {metrics_low['frac_E_positive']:.4f}")
    print(f"    dist from 1/φ: {metrics_low['dist_from_phi_inv']:.4f}")
    print(f"    memory depth: {metrics_low['memory_depth']}")
    
    print(f"\n  High recursion (λ=0.98):")
    print(f"    frac(E>0): {metrics_high['frac_E_positive']:.4f}")
    print(f"    dist from 1/φ: {metrics_high['dist_from_phi_inv']:.4f}")
    print(f"    memory depth: {metrics_high['memory_depth']}")
    
    results['comparison'] = {
        'low_recursion': {'lambda': 0.1, **metrics_low},
        'high_recursion': {'lambda': 0.98, **metrics_high}
    }
    
    # Statistical test: correlation between λ and φ proximity
    print("\n=== Statistical Analysis ===")
    
    lambdas_for_corr = [r['lambda'] for r in lambda_results]
    dists_for_corr = [r['dist_from_phi_inv'] for r in lambda_results]
    
    # Note: we expect NEGATIVE correlation in middle λ range,
    # since distance should decrease as λ approaches critical value
    
    # Pearson correlation
    r, p = stats.pearsonr(lambdas_for_corr, dists_for_corr)
    
    print(f"\n  Correlation (λ vs dist): r = {r:.4f}, p = {p:.6f}")
    
    results['correlation'] = {
        'pearson_r': float(r),
        'p_value': float(p)
    }
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    high_recursion_better = metrics_high['dist_from_phi_inv'] < metrics_low['dist_from_phi_inv']
    critical_exists = best_dist < 0.05
    
    if high_recursion_better and critical_exists:
        conclusion = "SUPPORTS CAH Condition 2: Recursion (high λ) is necessary for φ emergence"
        print(f"✅ {conclusion}")
    elif critical_exists:
        conclusion = "PARTIAL: Critical λ exists but improvement over low λ is unclear"
        print(f"🔄 {conclusion}")
    else:
        conclusion = "DOES NOT SUPPORT: No clear φ emergence at any λ"
        print(f"❌ {conclusion}")
    
    results['conclusion'] = {
        'supports_cah': bool(high_recursion_better and critical_exists),
        'high_recursion_better': bool(high_recursion_better),
        'critical_exists': bool(critical_exists),
        'summary': conclusion
    }
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_19_cah_recursion_depth_{timestamp}.json'
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == '__main__':
    run_experiment()
