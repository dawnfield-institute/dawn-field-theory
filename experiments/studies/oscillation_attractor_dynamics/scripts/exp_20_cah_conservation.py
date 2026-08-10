"""
Experiment 20: CAH Condition 3 - Internal Conservation
======================================================

Tests whether φ signatures require CONSERVATION dynamics.

CAH (Conditional Attractor Hypothesis):
    Ξ ≈ 1.057 emerges IFF:
    1. System is closed (no external injection)
    2. System is recursive (dynamics applied iteratively)
    3. System is INTERNALLY CONSERVING (f(P) = Σf(C)) ← THIS EXPERIMENT
    4. System is computationally saturated (edge-of-chaos)

PAC Conservation: f(Parent) = Σf(Children)
- When potential collapses to actualization, the total is conserved
- This is the "no free lunch" constraint

In SEC terms:
- I(n) = Ŝ(n) - S(n) is the collapse impulse
- If conservation holds: Σ I(n) should be bounded (not diverge)
- The stress field E(n) should show bounded variance

Key test: Does violating conservation destroy φ signatures?
- Conserved: Standard SEC
- Non-conserved: Add drift/bias to E(n) that violates Σ I ≈ 0
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


def measure_conservation(I: np.ndarray, E: np.ndarray) -> dict:
    """
    Measure how well the system conserves "energy" (PAC-sense).
    
    Key metrics:
    1. Cumulative I: Should be bounded if conserved
    2. E variance: Should be stationary if conserved
    3. Drift rate: Should be ~0 if conserved
    """
    # 1. Cumulative impulse
    cumsum_I = np.cumsum(I)
    final_cumsum = cumsum_I[-1]
    max_cumsum = np.max(np.abs(cumsum_I))
    
    # Normalized by length (drift per step)
    drift_rate = final_cumsum / len(I)
    
    # 2. E variance over windows
    window_size = 1000
    n_windows = len(E) // window_size
    variances = [np.var(E[i*window_size:(i+1)*window_size]) for i in range(n_windows)]
    variance_stability = np.std(variances) / (np.mean(variances) + 1e-10)
    
    # 3. Mean reversion (autocorrelation of first differences)
    dE = np.diff(E)
    E_centered = E[:-1] - np.mean(E[:-1])
    mean_reversion = -np.corrcoef(E_centered, dE)[0, 1]  # Negative = mean reverting
    
    return {
        'final_cumsum_I': float(final_cumsum),
        'max_cumsum_I': float(max_cumsum),
        'drift_rate': float(drift_rate),
        'variance_stability': float(variance_stability),
        'mean_reversion': float(mean_reversion)
    }


def compute_nonconserved_sec(n_max: int, factor_base: list, window: int, lam: float,
                              drift: float = 0.001) -> dict:
    """
    Compute SEC with added drift (violates conservation).
    
    The drift injects persistent bias, breaking f(P) = Σf(C).
    """
    sec = compute_sec(n_max=n_max, factor_base=factor_base, 
                      window=window, lam=lam)
    
    # Add cumulative drift to E
    drift_term = drift * np.arange(len(sec.E))
    E_drifted = sec.E + drift_term
    
    return {
        'E': E_drifted,
        'I': sec.I,
        'primes': sec.primes,
        'prime_mask': sec.prime_mask
    }


def run_experiment():
    """
    Test whether conservation is necessary for φ emergence.
    """
    print("=" * 70)
    print("EXPERIMENT 20: CAH Condition 3 - Internal Conservation")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print("Testing whether φ signatures require conservation: f(P) = Σf(C)")
    print()
    
    # Parameters
    n_max = 50000
    factor_base = FIRST_50_PRIMES[:9]
    window = 101
    lam = 0.98
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_max': n_max,
            'factor_base_size': len(factor_base),
            'window': window,
            'lam': lam
        },
        'hypothesis': 'φ signatures require conservation (bounded cumulative impulse)',
        'cah_condition': 'Condition 3: System is INTERNALLY CONSERVING'
    }
    
    # Test 1: Verify standard SEC is approximately conserved
    print("=== Test 1: Conservation in Standard SEC ===")
    
    sec = compute_sec(n_max=n_max, factor_base=factor_base, 
                      window=window, lam=lam)
    conservation = measure_conservation(sec.I, sec.E)
    
    print(f"\n  Conservation metrics (standard SEC):")
    print(f"    Final Σ I(n):       {conservation['final_cumsum_I']:.4f}")
    print(f"    Max |Σ I(n)|:       {conservation['max_cumsum_I']:.4f}")
    print(f"    Drift rate:         {conservation['drift_rate']:.6f}")
    print(f"    Variance stability: {conservation['variance_stability']:.4f}")
    print(f"    Mean reversion:     {conservation['mean_reversion']:.4f}")
    
    # φ signature
    odd_E = sec.E[np.arange(len(sec.E)) % 2 == 1]
    frac_std = np.mean(odd_E > 0)
    dist_std = abs(frac_std - PHI_INV)
    
    print(f"\n  φ signature:")
    print(f"    frac(E > 0): {frac_std:.4f}")
    print(f"    dist from 1/φ: {dist_std:.6f}")
    
    results['standard_sec'] = {
        **conservation,
        'frac_E_positive': float(frac_std),
        'dist_from_phi_inv': float(dist_std)
    }
    
    # Test 2: Non-conserved SEC with various drift rates
    print("\n=== Test 2: Non-Conserved SEC (With Drift) ===")
    
    drifts = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    drift_results = []
    
    print(f"\n{'drift':>10} {'frac(E>0)':>12} {'dist from 1/φ':>14} {'conserved?':>12}")
    print("-" * 50)
    
    for drift in drifts:
        noncons = compute_nonconserved_sec(n_max, factor_base, window, lam, drift)
        
        odd_E = noncons['E'][np.arange(len(noncons['E'])) % 2 == 1]
        frac = np.mean(odd_E > 0)
        dist = abs(frac - PHI_INV)
        
        # With positive drift, frac should approach 1.0
        drift_results.append({
            'drift': float(drift),
            'frac_E_positive': float(frac),
            'dist_from_phi_inv': float(dist)
        })
        
        is_conserved = "No" if drift > 0.0001 else "~Yes"
        print(f"{drift:>10.4f} {frac:>12.4f} {dist:>14.6f} {is_conserved:>12}")
    
    results['drift_analysis'] = drift_results
    
    # Test 3: Detrended non-conserved (restore conservation)
    print("\n=== Test 3: Detrending to Restore Conservation ===")
    
    # Take highly drifted E and detrend it
    high_drift = compute_nonconserved_sec(n_max, factor_base, window, lam, drift=0.01)
    E_drifted = high_drift['E']
    
    # Linear detrend
    t = np.arange(len(E_drifted))
    slope, intercept = np.polyfit(t, E_drifted, 1)
    E_detrended = E_drifted - (slope * t + intercept)
    
    odd_E_detrended = E_detrended[np.arange(len(E_detrended)) % 2 == 1]
    frac_detrended = np.mean(odd_E_detrended > 0)
    dist_detrended = abs(frac_detrended - PHI_INV)
    
    print(f"\n  Before detrending (drift=0.01):")
    print(f"    frac(E > 0): {drift_results[-1]['frac_E_positive']:.4f}")
    print(f"    dist from 1/φ: {drift_results[-1]['dist_from_phi_inv']:.6f}")
    
    print(f"\n  After detrending:")
    print(f"    frac(E > 0): {frac_detrended:.4f}")
    print(f"    dist from 1/φ: {dist_detrended:.6f}")
    
    detrending_helps = dist_detrended < drift_results[-1]['dist_from_phi_inv']
    
    results['detrending'] = {
        'before_dist': float(drift_results[-1]['dist_from_phi_inv']),
        'after_dist': float(dist_detrended),
        'detrending_helps': bool(detrending_helps)
    }
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    standard_near_phi = dist_std < 0.05
    drift_destroys = drift_results[-1]['dist_from_phi_inv'] > 0.2  # High drift → frac → 1.0
    
    if standard_near_phi and drift_destroys and detrending_helps:
        conclusion = "SUPPORTS CAH Condition 3: Conservation is necessary; drift destroys φ, detrending restores it"
        print(f"✅ {conclusion}")
    elif standard_near_phi and drift_destroys:
        conclusion = "PARTIAL: Standard is conserved and shows φ; drift destroys it; detrending unclear"
        print(f"🔄 {conclusion}")
    else:
        conclusion = "DOES NOT SUPPORT: Conservation doesn't clearly affect φ emergence"
        print(f"❌ {conclusion}")
    
    results['conclusion'] = {
        'supports_cah': bool(standard_near_phi and drift_destroys and detrending_helps),
        'standard_near_phi': bool(standard_near_phi),
        'drift_destroys_phi': bool(drift_destroys),
        'detrending_helps': bool(detrending_helps),
        'summary': conclusion
    }
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_20_cah_conservation_{timestamp}.json'
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == '__main__':
    run_experiment()
