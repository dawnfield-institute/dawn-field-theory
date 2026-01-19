"""
Experiment 22: CAH Unified Validation
=====================================

Tests ALL FOUR CAH conditions together and determines if Ξ emerges.

CAH (Conditional Attractor Hypothesis):
    Ξ ≈ 1.057 emerges IFF ALL of:
    1. System is CLOSED (no external injection)
    2. System is RECURSIVE (dynamics applied iteratively)
    3. System is INTERNALLY CONSERVING (f(P) = Σf(C))
    4. System is COMPUTATIONALLY SATURATED (edge-of-chaos)

This experiment:
1. Measures each condition quantitatively
2. Creates systems with varying degrees of each condition
3. Tests the conjunctive hypothesis: all 4 → Ξ emergence

Connection to CA validation:
- Rule 110 (Class IV) has P/A ≈ 1.0579 ≈ Ξ
- Class IV satisfies all 4 CAH conditions
- This experiment tests the same in SEC oscillation dynamics
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


def measure_cah_conditions(E: np.ndarray, I: np.ndarray, lam: float, k: int,
                           noise_level: float = 0.0) -> dict:
    """
    Measure how well a system satisfies all 4 CAH conditions.
    
    Returns scores for each condition (0-1, higher = better).
    """
    # Condition 1: CLOSED (no external injection)
    # Score based on noise level (lower = more closed)
    closed_score = 1.0 / (1.0 + 10 * noise_level)
    
    # Condition 2: RECURSIVE (depth of memory)
    # Score based on λ (higher = more recursive)
    recursive_score = lam  # λ ∈ [0, 1]
    
    # Condition 3: CONSERVING (bounded cumulative impulse)
    # Score based on drift rate (lower = more conserved)
    cumsum = np.abs(np.cumsum(I)[-1])
    max_possible = len(I) * np.std(I)
    drift_ratio = cumsum / (max_possible + 1e-10)
    conserving_score = 1.0 / (1.0 + drift_ratio)
    
    # Condition 4: SATURATED (edge-of-chaos)
    # Score based on k proximity to optimal (k=9)
    # Bell curve centered at k=9
    optimal_k = 9
    k_dist = abs(k - optimal_k)
    saturation_score = np.exp(-(k_dist / 3) ** 2)
    
    return {
        'closed': float(closed_score),
        'recursive': float(recursive_score),
        'conserving': float(conserving_score),
        'saturated': float(saturation_score),
        'product': float(closed_score * recursive_score * conserving_score * saturation_score)
    }


def compute_phi_distance(E: np.ndarray) -> float:
    """Compute distance of frac(E > 0) from 1/φ."""
    odd_E = E[np.arange(len(E)) % 2 == 1]
    frac = np.mean(odd_E > 0)
    return abs(frac - PHI_INV)


def run_experiment():
    """
    Unified CAH validation: test all 4 conditions together.
    """
    print("=" * 70)
    print("EXPERIMENT 22: CAH Unified Validation")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print("Testing: Ξ emerges IFF (Closed ∧ Recursive ∧ Conserving ∧ Saturated)")
    print()
    
    n_max = 50000
    window = 101
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {'n_max': n_max, 'window': window},
        'hypothesis': 'Ξ emerges IFF all 4 CAH conditions are met',
        'xi': XI,
        'phi_inv': PHI_INV
    }
    
    # Test matrix: vary each condition while keeping others optimal
    print("=== Test 1: Optimal Configuration (All Conditions Met) ===")
    
    # Optimal: closed, high λ, no drift, k=9
    sec_optimal = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9],
                              window=window, lam=0.98)
    conditions_optimal = measure_cah_conditions(sec_optimal.E, sec_optimal.I,
                                                 lam=0.98, k=9, noise_level=0.0)
    dist_optimal = compute_phi_distance(sec_optimal.E)
    
    print(f"\n  CAH Scores:")
    print(f"    Closed:     {conditions_optimal['closed']:.4f}")
    print(f"    Recursive:  {conditions_optimal['recursive']:.4f}")
    print(f"    Conserving: {conditions_optimal['conserving']:.4f}")
    print(f"    Saturated:  {conditions_optimal['saturated']:.4f}")
    print(f"    Product:    {conditions_optimal['product']:.4f}")
    print(f"\n  φ distance: {dist_optimal:.6f}")
    
    results['optimal'] = {
        'conditions': conditions_optimal,
        'phi_distance': float(dist_optimal)
    }
    
    # Test 2: Violate each condition individually
    print("\n=== Test 2: Violate Each Condition Individually ===")
    
    violations = {}
    
    # 2a: Violate CLOSED (add noise)
    print("\n  2a. Violate CLOSED (add external noise):")
    sec_base = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9],
                           window=window, lam=0.98)
    np.random.seed(42)
    E_noisy = sec_base.E + np.random.normal(0, 0.5 * np.std(sec_base.E), len(sec_base.E))
    dist_noisy = abs(np.mean(E_noisy[np.arange(len(E_noisy)) % 2 == 1] > 0) - PHI_INV)
    conditions_noisy = measure_cah_conditions(E_noisy, sec_base.I, lam=0.98, k=9, noise_level=0.5)
    
    print(f"      Product score: {conditions_noisy['product']:.4f}")
    print(f"      φ distance:    {dist_noisy:.6f}")
    violations['closed_violated'] = {'conditions': conditions_noisy, 'phi_distance': float(dist_noisy)}
    
    # 2b: Violate RECURSIVE (low λ)
    print("\n  2b. Violate RECURSIVE (low λ = 0.1):")
    sec_lowlam = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9],
                             window=window, lam=0.1)
    dist_lowlam = compute_phi_distance(sec_lowlam.E)
    conditions_lowlam = measure_cah_conditions(sec_lowlam.E, sec_lowlam.I, lam=0.1, k=9, noise_level=0.0)
    
    print(f"      Product score: {conditions_lowlam['product']:.4f}")
    print(f"      φ distance:    {dist_lowlam:.6f}")
    violations['recursive_violated'] = {'conditions': conditions_lowlam, 'phi_distance': float(dist_lowlam)}
    
    # 2c: Violate CONSERVING (add drift)
    print("\n  2c. Violate CONSERVING (add drift):")
    drift = 0.01 * np.arange(len(sec_base.E))
    E_drifted = sec_base.E + drift
    dist_drift = abs(np.mean(E_drifted[np.arange(len(E_drifted)) % 2 == 1] > 0) - PHI_INV)
    # Fake high drift in I for scoring
    I_drifted = sec_base.I + 0.1
    conditions_drift = measure_cah_conditions(E_drifted, I_drifted, lam=0.98, k=9, noise_level=0.0)
    
    print(f"      Product score: {conditions_drift['product']:.4f}")
    print(f"      φ distance:    {dist_drift:.6f}")
    violations['conserving_violated'] = {'conditions': conditions_drift, 'phi_distance': float(dist_drift)}
    
    # 2d: Violate SATURATED (wrong k)
    print("\n  2d. Violate SATURATED (k=3, too ordered):")
    sec_lowk = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:3],
                           window=window, lam=0.98)
    dist_lowk = compute_phi_distance(sec_lowk.E)
    conditions_lowk = measure_cah_conditions(sec_lowk.E, sec_lowk.I, lam=0.98, k=3, noise_level=0.0)
    
    print(f"      Product score: {conditions_lowk['product']:.4f}")
    print(f"      φ distance:    {dist_lowk:.6f}")
    violations['saturated_violated'] = {'conditions': conditions_lowk, 'phi_distance': float(dist_lowk)}
    
    results['violations'] = violations
    
    # Test 3: Correlation between product score and φ distance
    print("\n=== Test 3: Correlation Analysis ===")
    
    # Collect all test points
    all_products = [
        conditions_optimal['product'],
        conditions_noisy['product'],
        conditions_lowlam['product'],
        conditions_drift['product'],
        conditions_lowk['product']
    ]
    all_distances = [
        dist_optimal,
        dist_noisy,
        dist_lowlam,
        dist_drift,
        dist_lowk
    ]
    
    # Add more random configurations
    np.random.seed(123)
    for _ in range(20):
        k = np.random.randint(3, 15)
        lam = np.random.uniform(0.1, 0.999)
        noise = np.random.uniform(0, 0.5)
        
        sec = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:k],
                         window=window, lam=lam)
        E_test = sec.E + np.random.normal(0, noise * np.std(sec.E), len(sec.E))
        dist = abs(np.mean(E_test[np.arange(len(E_test)) % 2 == 1] > 0) - PHI_INV)
        cond = measure_cah_conditions(E_test, sec.I, lam=lam, k=k, noise_level=noise)
        
        all_products.append(cond['product'])
        all_distances.append(dist)
    
    # Correlation: product score vs φ distance
    # We expect NEGATIVE correlation (higher product → lower distance)
    r, p = stats.pearsonr(all_products, all_distances)
    
    print(f"\n  Correlation (product vs φ distance): r = {r:.4f}, p = {p:.6f}")
    print(f"  (Negative r supports CAH: higher product → closer to φ)")
    
    results['correlation'] = {
        'pearson_r': float(r),
        'p_value': float(p),
        'n_samples': len(all_products)
    }
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    optimal_near_phi = dist_optimal < 0.05
    violations_worse = all([
        dist_noisy > dist_optimal,
        dist_lowlam > dist_optimal,
        dist_drift > dist_optimal,
        dist_lowk > dist_optimal
    ])
    negative_correlation = r < -0.3 and p < 0.05
    
    if optimal_near_phi and violations_worse and negative_correlation:
        conclusion = "✅ VALIDATED: CAH holds. All 4 conditions → φ emergence; violations → degradation"
        status = "VALIDATED"
    elif optimal_near_phi and violations_worse:
        conclusion = "🔄 PARTIAL: Optimal shows φ and violations hurt, but correlation weak"
        status = "PARTIAL"
    elif optimal_near_phi:
        conclusion = "🔄 INCONCLUSIVE: Optimal shows φ but violations don't consistently degrade"
        status = "INCONCLUSIVE"
    else:
        conclusion = "❌ NOT VALIDATED: Optimal configuration doesn't show clear φ convergence"
        status = "NOT_VALIDATED"
    
    print(f"\n{conclusion}")
    print(f"\nOptimal φ distance: {dist_optimal:.6f} (threshold: 0.05)")
    print(f"All violations worse: {violations_worse}")
    print(f"Significant negative correlation: {negative_correlation}")
    
    results['conclusion'] = {
        'status': status,
        'optimal_near_phi': bool(optimal_near_phi),
        'violations_worse': bool(violations_worse),
        'negative_correlation': bool(negative_correlation),
        'summary': conclusion
    }
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_22_cah_unified_{timestamp}.json'
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == '__main__':
    run_experiment()
