"""
Experiment 22b: Refined CAH Unified Validation
==============================================

Updates based on exp_21b/21c findings:
- Condition 4 is NOT "edge-of-chaos (1/f noise)"
- Condition 4 IS "optimal kurtosis" (distribution shape)

Revised CAH:
    Ξ emerges IFF:
    1. CLOSED: No external injection
    2. RECURSIVE: High λ (memory depth)
    3. CONSERVING: Bounded cumulative impulse
    4. BALANCED: Optimal kurtosis (near -0.15)

The kurtosis finding (r=-0.935) is the key:
- k=9 has kurtosis ≈ -0.15, which is φ-optimal
- This represents a balance between too-peaked and too-flat distributions
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
OPTIMAL_KURTOSIS = -0.15  # From exp_21c


def measure_cah_conditions_v2(E: np.ndarray, I: np.ndarray, lam: float, 
                               noise_level: float = 0.0) -> dict:
    """
    Measure CAH conditions with refined Condition 4.
    
    Conditions:
    1. CLOSED: 1/(1 + 10*noise)
    2. RECURSIVE: λ
    3. CONSERVING: 1/(1 + drift_ratio)
    4. BALANCED: exp(-(kurtosis - optimal)^2 / 0.1)
    """
    # Condition 1: CLOSED
    closed_score = 1.0 / (1.0 + 10 * noise_level)
    
    # Condition 2: RECURSIVE
    recursive_score = lam
    
    # Condition 3: CONSERVING
    cumsum = np.abs(np.cumsum(I)[-1])
    max_possible = len(I) * np.std(I) + 1e-10
    drift_ratio = cumsum / max_possible
    conserving_score = 1.0 / (1.0 + drift_ratio)
    
    # Condition 4: BALANCED (optimal kurtosis)
    kurtosis = stats.kurtosis(E)
    kurtosis_dist = abs(kurtosis - OPTIMAL_KURTOSIS)
    balanced_score = np.exp(-(kurtosis_dist ** 2) / 0.1)
    
    # Product score
    product = closed_score * recursive_score * conserving_score * balanced_score
    
    return {
        'closed': float(closed_score),
        'recursive': float(recursive_score),
        'conserving': float(conserving_score),
        'balanced': float(balanced_score),
        'kurtosis': float(kurtosis),
        'product': float(product)
    }


def compute_phi_distance(E: np.ndarray) -> float:
    """Compute distance from 1/φ."""
    odd_E = E[np.arange(len(E)) % 2 == 1]
    frac = np.mean(odd_E > 0)
    return abs(frac - PHI_INV)


def run_experiment():
    """
    Refined CAH validation with corrected Condition 4.
    """
    print("=" * 70)
    print("EXPERIMENT 22b: Refined CAH Unified Validation")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print("Revised CAH: Ξ ↔ (Closed ∧ Recursive ∧ Conserving ∧ Balanced)")
    print(f"Where 'Balanced' = optimal kurtosis ≈ {OPTIMAL_KURTOSIS}")
    print()
    
    n_max = 50000
    window = 101
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {'n_max': n_max, 'window': window},
        'optimal_kurtosis': OPTIMAL_KURTOSIS
    }
    
    # Test 1: Optimal configuration
    print("=== Test 1: Optimal Configuration (k=9, λ=0.98) ===")
    
    sec_opt = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9],
                          window=window, lam=0.98)
    phi_dist_opt = compute_phi_distance(sec_opt.E)
    cond_opt = measure_cah_conditions_v2(sec_opt.E, sec_opt.I, lam=0.98)
    
    print(f"\n  CAH Scores:")
    print(f"    Closed:     {cond_opt['closed']:.4f}")
    print(f"    Recursive:  {cond_opt['recursive']:.4f}")
    print(f"    Conserving: {cond_opt['conserving']:.4f}")
    print(f"    Balanced:   {cond_opt['balanced']:.4f} (kurtosis={cond_opt['kurtosis']:.4f})")
    print(f"    Product:    {cond_opt['product']:.4f}")
    print(f"\n  φ distance: {phi_dist_opt:.6f}")
    
    results['optimal'] = {'conditions': cond_opt, 'phi_distance': float(phi_dist_opt)}
    
    # Test 2: Violate each condition
    print("\n=== Test 2: Single Condition Violations ===")
    
    violations = {}
    
    # 2a: Violate CLOSED
    print("\n  2a. Violate CLOSED (noise=0.5):")
    np.random.seed(42)
    E_noisy = sec_opt.E + np.random.normal(0, 0.5 * np.std(sec_opt.E), len(sec_opt.E))
    phi_noisy = abs(np.mean(E_noisy[np.arange(len(E_noisy)) % 2 == 1] > 0) - PHI_INV)
    cond_noisy = measure_cah_conditions_v2(E_noisy, sec_opt.I, lam=0.98, noise_level=0.5)
    print(f"      Product: {cond_noisy['product']:.4f}, φ dist: {phi_noisy:.6f}")
    violations['closed_violated'] = {'conditions': cond_noisy, 'phi_distance': float(phi_noisy)}
    
    # 2b: Violate RECURSIVE
    print("\n  2b. Violate RECURSIVE (λ=0.1):")
    sec_lowlam = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9],
                             window=window, lam=0.1)
    phi_lowlam = compute_phi_distance(sec_lowlam.E)
    cond_lowlam = measure_cah_conditions_v2(sec_lowlam.E, sec_lowlam.I, lam=0.1)
    print(f"      Product: {cond_lowlam['product']:.4f}, φ dist: {phi_lowlam:.6f}")
    violations['recursive_violated'] = {'conditions': cond_lowlam, 'phi_distance': float(phi_lowlam)}
    
    # 2c: Violate CONSERVING
    print("\n  2c. Violate CONSERVING (add drift):")
    drift = 0.01 * np.arange(len(sec_opt.E))
    E_drift = sec_opt.E + drift
    phi_drift = abs(np.mean(E_drift[np.arange(len(E_drift)) % 2 == 1] > 0) - PHI_INV)
    I_drift = sec_opt.I + 0.1  # Biased impulse
    cond_drift = measure_cah_conditions_v2(E_drift, I_drift, lam=0.98)
    print(f"      Product: {cond_drift['product']:.4f}, φ dist: {phi_drift:.6f}")
    violations['conserving_violated'] = {'conditions': cond_drift, 'phi_distance': float(phi_drift)}
    
    # 2d: Violate BALANCED (wrong kurtosis via k=3)
    print("\n  2d. Violate BALANCED (k=3, wrong kurtosis):")
    sec_k3 = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:3],
                         window=window, lam=0.98)
    phi_k3 = compute_phi_distance(sec_k3.E)
    cond_k3 = measure_cah_conditions_v2(sec_k3.E, sec_k3.I, lam=0.98)
    print(f"      Product: {cond_k3['product']:.4f}, φ dist: {phi_k3:.6f}")
    print(f"      (kurtosis = {cond_k3['kurtosis']:.4f}, far from optimal {OPTIMAL_KURTOSIS})")
    violations['balanced_violated'] = {'conditions': cond_k3, 'phi_distance': float(phi_k3)}
    
    results['violations'] = violations
    
    # Test 3: Sweep configurations and test correlation
    print("\n=== Test 3: Configuration Sweep ===")
    
    sweep_data = []
    
    # Systematic sweep
    for k in [3, 5, 7, 9, 11, 13, 15]:
        for lam in [0.1, 0.5, 0.8, 0.98]:
            for noise in [0.0, 0.2, 0.5]:
                sec = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:k],
                                 window=window, lam=lam)
                
                if noise > 0:
                    np.random.seed(k * 100 + int(lam * 100) + int(noise * 100))
                    E_test = sec.E + np.random.normal(0, noise * np.std(sec.E), len(sec.E))
                else:
                    E_test = sec.E
                
                phi_dist = abs(np.mean(E_test[np.arange(len(E_test)) % 2 == 1] > 0) - PHI_INV)
                cond = measure_cah_conditions_v2(E_test, sec.I, lam=lam, noise_level=noise)
                
                sweep_data.append({
                    'k': k, 'lam': lam, 'noise': noise,
                    'phi_distance': float(phi_dist),
                    **cond
                })
    
    products = [d['product'] for d in sweep_data]
    phi_dists = [d['phi_distance'] for d in sweep_data]
    
    r, p = stats.pearsonr(products, phi_dists)
    
    print(f"\n  Configurations tested: {len(sweep_data)}")
    print(f"  Correlation (product vs φ dist): r = {r:.4f}, p = {p:.6f}")
    
    results['sweep'] = {
        'n_configs': len(sweep_data),
        'correlation': {'r': float(r), 'p': float(p)}
    }
    
    # Test 4: Component correlations
    print("\n=== Test 4: Component Correlations ===")
    
    for component in ['closed', 'recursive', 'conserving', 'balanced']:
        values = [d[component] for d in sweep_data]
        r_comp, p_comp = stats.pearsonr(values, phi_dists)
        sig = "**" if p_comp < 0.05 else ""
        print(f"  {component:>12} vs φ dist: r = {r_comp:>7.4f}, p = {p_comp:.4f} {sig}")
    
    # Kurtosis directly
    kurtoses = [d['kurtosis'] for d in sweep_data]
    r_kurt, p_kurt = stats.pearsonr(kurtoses, phi_dists)
    print(f"  {'kurtosis':>12} vs φ dist: r = {r_kurt:>7.4f}, p = {p_kurt:.4f}")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    opt_best = (phi_dist_opt < violations['closed_violated']['phi_distance'] and
                phi_dist_opt < violations['recursive_violated']['phi_distance'] and
                phi_dist_opt < violations['conserving_violated']['phi_distance'] and
                phi_dist_opt < violations['balanced_violated']['phi_distance'])
    
    negative_corr = r < -0.3 and p < 0.05
    
    if opt_best and negative_corr:
        conclusion = "✅ VALIDATED: Refined CAH holds. Product score predicts φ proximity."
        status = "VALIDATED"
    elif opt_best:
        conclusion = "🔄 PARTIAL: Optimal is best, but correlation could be stronger"
        status = "PARTIAL"
    else:
        conclusion = "❌ NOT VALIDATED: Optimal configuration not consistently best"
        status = "NOT_VALIDATED"
    
    print(f"\n{conclusion}")
    print(f"\n  Optimal φ distance: {phi_dist_opt:.6f}")
    print(f"  All violations worse: {opt_best}")
    print(f"  Product correlation: r = {r:.4f}, p = {p:.6f}")
    print(f"  Significant negative: {negative_corr}")
    
    results['conclusion'] = {
        'status': status,
        'optimal_best': bool(opt_best),
        'negative_correlation': bool(negative_corr),
        'summary': conclusion
    }
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_22b_refined_cah_{timestamp}.json'
    
    with open(os.path.join(results_dir, filename), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == '__main__':
    run_experiment()
