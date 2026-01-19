"""
Experiment 18: CAH Condition 1 - Closed System Analysis
=======================================================

Tests whether the oscillation dynamics operate as a CLOSED system.

CAH (Conditional Attractor Hypothesis):
    Ξ ≈ 1.057 emerges IFF:
    1. System is CLOSED (no external injection) ← THIS EXPERIMENT
    2. System is recursive (dynamics applied iteratively)
    3. System is internally conserving (f(P) = Σf(C))
    4. System is computationally saturated (edge-of-chaos)

In the oscillation_attractor_dynamics context:
- SEC stress field E(n) = λE(n-1) + I(n) is computed recursively
- The question: is this a CLOSED system?

Key test: Compare E(n) dynamics WITH and WITHOUT external injection:
- Closed: E(n) from pure recursive SEC computation
- Open: E(n) with injected noise at each step

If CAH holds:
- Closed system should show convergence toward φ-related signatures
- Open system should NOT show these signatures
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from scipy import stats

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

from sec_core import compute_sec, FIRST_50_PRIMES, PHI

XI = 1.0571428571428572
PHI_INV = 1 / PHI  # ≈ 0.618


def compute_open_system_sec(n_max: int, factor_base: list, window: int, lam: float,
                            noise_level: float = 0.1) -> dict:
    """
    Compute SEC with external noise injection (OPEN system).
    
    This violates CAH Condition 1: the system receives external input.
    """
    # First compute closed SEC
    closed_sec = compute_sec(n_max=n_max, factor_base=factor_base, 
                             window=window, lam=lam)
    
    # Inject noise into the stress field (simulating external input)
    np.random.seed(42)  # Reproducibility
    noise = np.random.normal(0, noise_level * np.std(closed_sec.E), len(closed_sec.E))
    E_open = closed_sec.E + noise
    
    return {
        'E': E_open,
        'I': closed_sec.I,
        'primes': closed_sec.primes,
        'prime_mask': closed_sec.prime_mask
    }


def analyze_phi_signatures(E: np.ndarray, I: np.ndarray, primes: np.ndarray,
                           prime_mask: np.ndarray) -> dict:
    """
    Analyze φ-related signatures in the stress field.
    
    Key metrics:
    1. Fraction of E > 0 (should → 1/φ at critical λ)
    2. Gap alternation rate (should → 1/φ)
    3. I(prime) / I(composite) ratio
    """
    # 1. Fraction E > 0 (on odd indices for proper comparison)
    odd_mask = np.arange(len(E)) % 2 == 1
    frac_E_positive = np.mean(E[odd_mask] > 0)
    dist_from_phi_inv = abs(frac_E_positive - PHI_INV)
    
    # 2. Prime gap alternation  
    gaps = np.diff(primes)
    alternations = np.diff(gaps)
    alternation_rate = np.mean(alternations[:-1] * alternations[1:] < 0)
    
    # 3. Injection ratio
    I_primes = I[prime_mask[:len(I)]]
    I_composites = I[~prime_mask[:len(I)]]
    mean_I_prime = np.mean(I_primes[I_primes > 0]) if np.any(I_primes > 0) else 0
    mean_I_composite = np.mean(np.abs(I_composites))
    injection_ratio = mean_I_prime / (mean_I_composite + 1e-10)
    
    return {
        'frac_E_positive': float(frac_E_positive),
        'dist_from_phi_inv': float(dist_from_phi_inv),
        'gap_alternation_rate': float(alternation_rate),
        'injection_ratio': float(injection_ratio)
    }


def run_experiment():
    """
    Compare closed vs open system dynamics.
    
    Hypothesis: φ signatures should be stronger in closed systems.
    """
    print("=" * 70)
    print("EXPERIMENT 18: CAH Condition 1 - Closed System Analysis")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print("Testing whether φ signatures require CLOSED dynamics (no external injection)")
    print()
    
    # Parameters
    n_max = 50000
    factor_base = FIRST_50_PRIMES[:9]  # k=9 as per original research
    window = 101
    lam = 0.9816  # Critical λ from exp_17c
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_max': n_max,
            'factor_base_size': len(factor_base),
            'window': window,
            'lam': lam
        },
        'hypothesis': 'φ signatures require closed system (no external injection)',
        'cah_condition': 'Condition 1: System is CLOSED'
    }
    
    # Test 1: CLOSED system (standard SEC)
    print("=== Test 1: CLOSED System (Standard SEC) ===")
    closed_sec = compute_sec(n_max=n_max, factor_base=factor_base, 
                             window=window, lam=lam)
    closed_signatures = analyze_phi_signatures(
        closed_sec.E, closed_sec.I, closed_sec.primes, closed_sec.prime_mask
    )
    
    print(f"  frac(E > 0):       {closed_signatures['frac_E_positive']:.4f}")
    print(f"  dist from 1/φ:     {closed_signatures['dist_from_phi_inv']:.4f}")
    print(f"  gap alternation:   {closed_signatures['gap_alternation_rate']:.4f}")
    print(f"  injection ratio:   {closed_signatures['injection_ratio']:.4f}")
    
    results['closed_system'] = closed_signatures
    
    # Test 2: OPEN systems with varying noise levels
    print("\n=== Test 2: OPEN Systems (With External Noise) ===")
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    open_results = []
    
    for noise in noise_levels:
        open_sec = compute_open_system_sec(n_max, factor_base, window, lam, noise)
        open_signatures = analyze_phi_signatures(
            open_sec['E'], open_sec['I'], open_sec['primes'], open_sec['prime_mask']
        )
        open_results.append({
            'noise_level': noise,
            **open_signatures
        })
        print(f"\n  Noise level: {noise}")
        print(f"    frac(E > 0):     {open_signatures['frac_E_positive']:.4f}")
        print(f"    dist from 1/φ:   {open_signatures['dist_from_phi_inv']:.4f}")
    
    results['open_systems'] = open_results
    
    # Test 3: Statistical comparison
    print("\n=== Test 3: Statistical Comparison ===")
    
    # Run multiple trials for closed system (different seeds don't affect closed)
    # vs open system (noise varies)
    n_trials = 100
    closed_fracs = []
    open_fracs_high_noise = []
    
    for trial in range(n_trials):
        np.random.seed(trial)
        
        # Closed is deterministic, but we can sample at different windows
        start = trial * 100
        end = start + 5000
        if end < len(closed_sec.E):
            window_E = closed_sec.E[start:end]
            odd_mask = np.arange(len(window_E)) % 2 == 1
            closed_fracs.append(np.mean(window_E[odd_mask] > 0))
        
        # Open with high noise
        np.random.seed(trial * 1000 + 42)
        noise = np.random.normal(0, 0.5 * np.std(closed_sec.E), len(closed_sec.E))
        E_open = closed_sec.E + noise
        if end < len(E_open):
            window_E = E_open[start:end]
            odd_mask = np.arange(len(window_E)) % 2 == 1
            open_fracs_high_noise.append(np.mean(window_E[odd_mask] > 0))
    
    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(closed_fracs, open_fracs_high_noise, equal_var=False)
    
    closed_mean = np.mean(closed_fracs)
    open_mean = np.mean(open_fracs_high_noise)
    
    print(f"\n  Closed system mean frac(E>0): {closed_mean:.4f}")
    print(f"  Open system mean frac(E>0):   {open_mean:.4f}")
    print(f"  Welch's t-test p-value:       {p_value:.6f}")
    
    results['statistical_test'] = {
        'closed_mean': float(closed_mean),
        'open_mean': float(open_mean),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'n_trials': n_trials,
        'significant_difference': bool(p_value < 0.05)
    }
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    closed_near_phi = closed_signatures['dist_from_phi_inv'] < 0.05
    noise_degrades = open_results[-1]['dist_from_phi_inv'] > closed_signatures['dist_from_phi_inv']
    
    if closed_near_phi and noise_degrades:
        conclusion = "SUPPORTS CAH Condition 1: Closed system shows φ convergence, noise degrades it"
        print(f"✅ {conclusion}")
    elif closed_near_phi:
        conclusion = "PARTIAL: Closed shows φ, but noise doesn't clearly degrade it"
        print(f"🔄 {conclusion}")
    else:
        conclusion = "DOES NOT SUPPORT: Closed system doesn't show clear φ convergence"
        print(f"❌ {conclusion}")
    
    results['conclusion'] = {
        'supports_cah': bool(closed_near_phi and noise_degrades),
        'closed_near_phi': bool(closed_near_phi),
        'noise_degrades': bool(noise_degrades),
        'summary': conclusion
    }
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_18_cah_closed_system_{timestamp}.json'
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == '__main__':
    run_experiment()
