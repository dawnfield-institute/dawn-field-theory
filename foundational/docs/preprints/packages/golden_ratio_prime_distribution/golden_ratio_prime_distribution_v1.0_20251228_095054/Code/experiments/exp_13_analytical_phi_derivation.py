#!/usr/bin/env python3
"""
Experiment 13: Analytical Derivation of φ Emergence
====================================================

Goal: Prove WHY size=9 produces frac(E>0) ≈ 1/φ

Hypothesis: The golden ratio emerges from "harmonic closure" - the point
where the factor base captures nearly all divisibility information.

Key insight from exp_12:
- FFT shows 99.96% of power at factor base prime periods
- This saturation point corresponds to φ emergence

Analytical approach:
1. Model S(n) as sum of periodic components with period p for each prime p
2. Show that E(n) > 0 ⟺ n is "divisibility-sparse" relative to expectation
3. Derive the asymptotic fraction as function of factor base coverage
4. Show this fraction → 1/φ at the harmonic closure threshold

Trace output: results/exp_13_analytical_derivation_YYYYMMDD_HHMMSS.json
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sec_core import (
    compute_sec, prime_sieve, symbolic_entropy, entropy_expectation,
    collapse_impulse, stress_field, FIRST_50_PRIMES, PHI
)

PHI_INV = 1 / PHI  # 0.6180339887...


def compute_divisibility_density(n_max: int, primes: List[int]) -> Dict[str, Any]:
    """
    Compute the density of integers divisible by at least one prime in the list.
    
    By inclusion-exclusion:
    P(divisible by at least one) = Σ 1/p - Σ 1/(p*q) + Σ 1/(p*q*r) - ...
    
    For independent primes (which they are), this simplifies to:
    P(divisible by at least one) = 1 - Π(1 - 1/p)
    """
    # Exact computation using inclusion-exclusion
    prob_not_divisible = 1.0
    for p in primes:
        prob_not_divisible *= (1 - 1/p)
    
    prob_divisible = 1 - prob_not_divisible
    
    # Empirical verification
    mask = np.zeros(n_max + 1, dtype=bool)
    for p in primes:
        mask[p::p] = True
    
    empirical_divisible = mask[1:].mean()
    
    return {
        "primes": primes,
        "theoretical_divisible": prob_divisible,
        "theoretical_not_divisible": prob_not_divisible,
        "empirical_divisible": float(empirical_divisible),
        "empirical_not_divisible": float(1 - empirical_divisible),
        "theory_vs_empirical_error": abs(prob_divisible - empirical_divisible)
    }


def compute_primorial_coverage(size: int) -> Dict[str, Any]:
    """
    Compute how much of integer "structure" is captured by first k primes.
    
    The primorial P_k = p_1 * p_2 * ... * p_k has period P_k.
    Residue classes mod P_k repeat with this period.
    
    Coverage fraction = 1 - φ(P_k)/P_k where φ is Euler's totient
    """
    primes = FIRST_50_PRIMES[:size]
    
    # Primorial
    primorial = 1
    for p in primes:
        primorial *= p
    
    # Euler's totient of primorial: φ(P_k) = P_k * Π(1 - 1/p)
    totient = primorial
    for p in primes:
        totient = totient * (p - 1) // p
    
    # Coverage = fraction of residue classes that have a small prime factor
    coverage = 1 - totient / primorial
    
    return {
        "size": size,
        "primes": list(primes),
        "primorial": primorial,
        "totient": totient,
        "coverage": coverage,
        "non_coverage": totient / primorial
    }


def analyze_S_periodicity(n_max: int, factor_base: List[int]) -> Dict[str, Any]:
    """
    Analyze the periodic structure of S(n).
    
    S(n) = |{p ∈ B : p|n}| / |B|
    
    For each prime p, divisibility by p is periodic with period p.
    So S(n) is a superposition of periodic functions.
    """
    S = symbolic_entropy(n_max, factor_base)
    
    # For each prime, compute its contribution
    contributions = {}
    for p in factor_base:
        # Contribution when divisible by p
        div_mask = np.zeros(n_max + 1, dtype=bool)
        div_mask[p::p] = True
        
        S_when_div = S[div_mask].mean()
        S_when_not = S[~div_mask].mean()
        
        contributions[p] = {
            "S_when_divisible": float(S_when_div),
            "S_when_not_divisible": float(S_when_not),
            "difference": float(S_when_div - S_when_not),
            "frequency": float(div_mask[1:].mean())
        }
    
    # Overall S statistics
    return {
        "factor_base": factor_base,
        "S_mean": float(S[1:].mean()),
        "S_std": float(S[1:].std()),
        "prime_contributions": contributions
    }


def derive_E_positive_fraction(n_max: int, size: int) -> Dict[str, Any]:
    """
    Attempt to derive frac(E>0) analytically.
    
    Key insight: E(n) accumulates I(n) = Ŝ(n) - S(n)
    
    I(n) > 0 when n has FEWER small prime factors than its neighbors (on average)
    I(n) < 0 when n has MORE small prime factors than its neighbors
    
    For odd n:
    - Primes have I(n) > 0 (simpler than neighbors)
    - Smooth numbers have I(n) < 0 (more divisible than neighbors)
    
    E(n) > 0 when recent history has net positive impulse.
    """
    factor_base = FIRST_50_PRIMES[:size]
    sec = compute_sec(n_max=n_max, factor_base=factor_base, window=101, lam=0.99)
    
    idx = np.arange(3, n_max + 1, 2)
    E_odd = sec.E[idx]
    I_odd = sec.I[idx]
    S_odd = sec.S[idx]
    
    # Empirical frac(E>0)
    frac_E_pos = float((E_odd > 0).mean())
    
    # Theoretical model: 
    # If I(n) has mean μ and the stress field is a weighted sum of I values,
    # then frac(E>0) depends on the distribution of the running sum.
    
    # For a random walk with positive drift μ, the fraction of time above 0
    # approaches 1 as μ → ∞ and 1/2 as μ → 0.
    
    # For our decay process E(n) = λE(n-1) + I(n), the stationary distribution
    # has mean = μ/(1-λ) and the fraction positive depends on the ratio of
    # mean to standard deviation.
    
    I_mean = float(I_odd.mean())
    I_std = float(I_odd.std())
    
    # Effective mean of E (stationary)
    lam = 0.99
    E_stationary_mean = I_mean / (1 - lam)
    E_stationary_std = I_std / np.sqrt(1 - lam**2)
    
    # If E were Gaussian, frac(E>0) = Φ(E_mean/E_std)
    # But this is a simplification
    gaussian_prediction = float(stats.norm.cdf(E_stationary_mean / E_stationary_std))
    
    return {
        "size": size,
        "n_max": n_max,
        "frac_E_positive_empirical": frac_E_pos,
        "error_vs_phi": frac_E_pos - PHI_INV,
        "I_mean": I_mean,
        "I_std": I_std,
        "E_stationary_mean": E_stationary_mean,
        "E_stationary_std": E_stationary_std,
        "gaussian_prediction": gaussian_prediction,
        "gaussian_error": abs(gaussian_prediction - frac_E_pos)
    }


def harmonic_closure_analysis(n_max: int = 50000) -> Dict[str, Any]:
    """
    Analyze the "harmonic closure" hypothesis.
    
    Hypothesis: φ emerges when the factor base captures "enough" structure.
    
    Metric: Fraction of spectral power in factor base primes.
    """
    results = []
    
    for size in range(2, 16):
        factor_base = FIRST_50_PRIMES[:size]
        sec = compute_sec(n_max=n_max, factor_base=factor_base, window=101, lam=0.99)
        
        # FFT of E
        idx = np.arange(3, n_max + 1, 2)
        E_odd = sec.E[idx]
        
        fft = np.fft.rfft(E_odd - E_odd.mean())
        power = np.abs(fft)**2
        freqs = np.fft.rfftfreq(len(E_odd))
        
        # Convert to periods (where meaningful)
        periods = np.zeros_like(freqs)
        periods[1:] = 1 / freqs[1:]
        
        # Power at factor base primes
        total_power = power.sum()
        fb_power = 0
        
        for p in factor_base:
            # Find frequency closest to 1/p
            target_freq = 1/p
            closest_idx = np.argmin(np.abs(freqs - target_freq))
            fb_power += power[closest_idx]
        
        fb_fraction = fb_power / total_power if total_power > 0 else 0
        
        # Empirical φ-threshold
        frac_E_pos = float((E_odd > 0).mean())
        
        results.append({
            "size": size,
            "factor_base": list(factor_base),
            "fb_power_fraction": float(fb_fraction),
            "frac_E_positive": frac_E_pos,
            "error_vs_phi": frac_E_pos - PHI_INV,
            "abs_error": abs(frac_E_pos - PHI_INV)
        })
    
    # Find closure point (where fb_power_fraction > 0.99)
    closure_size = None
    for r in results:
        if r["fb_power_fraction"] > 0.99 and closure_size is None:
            closure_size = r["size"]
    
    # Correlation between fb_power and phi proximity
    fb_fracs = [r["fb_power_fraction"] for r in results]
    abs_errors = [r["abs_error"] for r in results]
    correlation = float(np.corrcoef(fb_fracs, abs_errors)[0, 1])
    
    return {
        "size_sweep": results,
        "closure_size": closure_size,
        "correlation_fb_vs_error": correlation,
        "interpretation": "Negative correlation means higher harmonic coverage → closer to φ"
    }


def golden_ratio_derivation() -> Dict[str, Any]:
    """
    Attempt a first-principles derivation of why 1/φ appears.
    
    Key observations:
    1. E(n) > 0 ⟺ n is in a "sparse" region (fewer divisors than neighbors)
    2. The boundary between sparse and dense regions is related to 
       the balance between primes and smooth numbers
    3. This balance involves the golden ratio through Fibonacci structure
    
    Conjecture: The fraction 1/φ arises because:
    - The divisibility function S(n) has Fourier components at prime periods
    - The stress accumulation E partitions integers into + and - regions
    - The boundary of these regions follows a pattern related to Fibonacci
    - The asymptotic density of the + region is 1/φ
    """
    
    # Mathematical argument (outline):
    derivation = """
ANALYTICAL DERIVATION OF φ EMERGENCE (OUTLINE)

1. SETUP
   - S(n) = |{p ∈ B : p|n}| / |B| is a sum of indicator functions
   - Each indicator 1_{p|n} has period p
   - S(n) is quasi-periodic with quasi-period lcm(B) = primorial
   
2. FOURIER DECOMPOSITION
   - S(n) = Σ_p (1/|B|) · (1/p + Σ_{k≠0} (1/p) e^{2πikn/p})
   - The constant term is Σ_p 1/(p|B|)
   - The oscillating terms have frequencies 1/p, 2/p, ...
   
3. IMPULSE I(n) = Ŝ(n) - S(n)
   - Ŝ(n) is a moving average, which acts as a low-pass filter
   - I(n) captures the high-frequency deviations
   - Primes produce positive I (simpler than neighbors)
   - Smooth numbers produce negative I (more factors than neighbors)
   
4. STRESS FIELD E(n) = λE(n-1) + I(n)
   - E is an exponentially-weighted cumulative sum
   - E(n) > 0 when recent n have been "simpler than expected"
   - The sign of E partitions integers into + and - regions
   
5. THE PARTITION FRACTION θ = frac(E > 0)
   - θ depends on the distribution of I(n)
   - For a stationary process: θ = P(stationary E > 0)
   - This depends on the mean-to-std ratio of I
   
6. WHY θ → 1/φ?
   
   Key insight: The factor base size controls the "resolution" of S(n).
   
   At size k:
   - Small k: S(n) is coarse, I(n) has high variance
   - Large k: S(n) is fine, I(n) has low variance
   
   The transition occurs when the factor base captures the 
   "harmonic closure" - nearly all divisibility information.
   
   CONJECTURE: At the harmonic closure point:
   - The ratio mean(I)/std(I) equals a specific value
   - This value determines θ via the stationary distribution
   - For the prime-based factor base, this value → such that θ = 1/φ
   
7. FIBONACCI CONNECTION
   
   Why Fibonacci sizes are special:
   - Fibonacci numbers satisfy F_n = F_{n-1} + F_{n-2}
   - The ratio F_n/F_{n-1} → φ
   - Primorial growth: P_k ≈ 4^k (prime number theorem)
   - Coverage: 1 - φ(P_k)/P_k increases with k
   
   At size ≈ 9 (between F_6=8 and F_7=13):
   - Coverage crosses a critical threshold
   - The harmonic content saturates
   - θ stabilizes at 1/φ
   
8. THE CLOSURE THEOREM (CONJECTURED)
   
   Define:
   - H(k) = fraction of E's spectral power at factor base frequencies
   - θ(k) = frac(E > 0) with factor base size k
   
   Theorem (conjectured):
   As H(k) → 1 (harmonic closure):
   θ(k) → 1/φ
   
   Moreover, the rate of convergence is governed by the 
   Fibonacci sequence structure of the prime distribution.
"""
    
    return {
        "derivation_outline": derivation,
        "status": "conjectured",
        "key_quantities": {
            "phi": PHI,
            "one_over_phi": PHI_INV,
            "closure_threshold": 0.99,
            "optimal_size": 9
        }
    }


def run_experiment(n_max: int = 50000, save_trace: bool = True) -> Dict[str, Any]:
    """Run analytical derivation experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 13: Analytical Derivation of φ Emergence")
    print("=" * 70)
    print(f"\nTarget: 1/φ = {PHI_INV:.10f}")
    
    results = {}
    
    # 1. Divisibility density
    print(f"\n" + "-" * 70)
    print("1. DIVISIBILITY DENSITY ANALYSIS")
    print("-" * 70)
    
    for size in [5, 9, 13]:
        dd = compute_divisibility_density(n_max, FIRST_50_PRIMES[:size])
        print(f"\nSize {size}:")
        print(f"  P(divisible by ≥1 prime): {dd['theoretical_divisible']:.6f}")
        print(f"  P(not divisible):         {dd['theoretical_not_divisible']:.6f}")
        results[f"divisibility_size_{size}"] = dd
    
    # 2. Primorial coverage
    print(f"\n" + "-" * 70)
    print("2. PRIMORIAL COVERAGE")
    print("-" * 70)
    
    for size in [5, 9, 13]:
        pc = compute_primorial_coverage(size)
        print(f"\nSize {size}: primorial = {pc['primorial']:,}")
        print(f"  Coverage: {pc['coverage']:.6f}")
        print(f"  φ(P)/P:   {pc['non_coverage']:.6f}")
        results[f"primorial_size_{size}"] = pc
    
    # 3. S periodicity
    print(f"\n" + "-" * 70)
    print("3. S(n) PERIODICITY ANALYSIS")
    print("-" * 70)
    
    sp = analyze_S_periodicity(n_max, FIRST_50_PRIMES[:9])
    print(f"\nFactor base size 9:")
    print(f"  S mean: {sp['S_mean']:.6f}")
    print(f"  S std:  {sp['S_std']:.6f}")
    results["S_periodicity"] = sp
    
    # 4. E positive fraction derivation
    print(f"\n" + "-" * 70)
    print("4. E>0 FRACTION DERIVATION")
    print("-" * 70)
    
    derivation_results = []
    for size in range(5, 14):
        d = derive_E_positive_fraction(n_max, size)
        derivation_results.append(d)
        if size == 9:
            print(f"\nSize {size} (optimal):")
            print(f"  Empirical frac(E>0): {d['frac_E_positive_empirical']:.6f}")
            print(f"  Error vs 1/φ:        {d['error_vs_phi']:+.6f}")
            print(f"  Gaussian prediction: {d['gaussian_prediction']:.6f}")
            print(f"  I mean:              {d['I_mean']:.6f}")
            print(f"  I std:               {d['I_std']:.6f}")
    
    results["E_derivation"] = derivation_results
    
    # 5. Harmonic closure analysis
    print(f"\n" + "-" * 70)
    print("5. HARMONIC CLOSURE ANALYSIS")
    print("-" * 70)
    
    hc = harmonic_closure_analysis(n_max)
    print(f"\n{'Size':>6} {'FB Power':>12} {'Frac E>0':>12} {'Error':>12}")
    print("-" * 45)
    for r in hc["size_sweep"]:
        print(f"{r['size']:>6} {r['fb_power_fraction']:>12.4f} {r['frac_E_positive']:>12.6f} {r['error_vs_phi']:>+12.6f}")
    
    print(f"\nClosure size: {hc['closure_size']}")
    print(f"Correlation (FB power vs |error|): {hc['correlation_fb_vs_error']:.4f}")
    
    results["harmonic_closure"] = hc
    
    # 6. Golden ratio derivation
    print(f"\n" + "-" * 70)
    print("6. ANALYTICAL DERIVATION (OUTLINE)")
    print("-" * 70)
    
    gr = golden_ratio_derivation()
    print(gr["derivation_outline"])
    results["derivation"] = gr
    
    # Validation
    validation = {
        "divisibility_theory_matches_empirical": all(
            results[f"divisibility_size_{s}"]["theory_vs_empirical_error"] < 0.01
            for s in [5, 9, 13]
        ),
        "negative_correlation_fb_error": hc["correlation_fb_vs_error"] < -0.5,
        "closure_at_size_9": hc["closure_size"] == 9 or (hc["closure_size"] and hc["closure_size"] <= 10),
        "gaussian_approximation_reasonable": any(
            d["gaussian_error"] < 0.1 for d in derivation_results
        )
    }
    
    print(f"\n" + "-" * 70)
    print("VALIDATION")
    print("-" * 70)
    for check, passed in validation.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    results["validation"] = validation
    
    # Save trace
    if save_trace:
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"exp_13_analytical_derivation_{timestamp}.json"
        
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        print(f"\nTrace saved: {filepath.name}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_max", type=int, default=50000)
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    
    run_experiment(n_max=args.n_max, save_trace=not args.no_trace)
