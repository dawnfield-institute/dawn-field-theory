"""
Experiment 01: Zero-Crossing Detection
=======================================

Core hypothesis test: Do zero-crossings in the SEC stress field E(n)
correlate with prime positions more than expected by chance?

If primes are "attractors" in the oscillation framework, the system
should pass through zero (convergence point) near prime positions.
"""

import numpy as np
import sys
import os
import json
from datetime import datetime

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

from oscillation_engine import (
    detect_zero_crossings, 
    crossing_prime_correlation,
    full_oscillation_analysis,
    save_results
)
from sec_core import compute_sec, FIRST_50_PRIMES, PHI


def run_experiment():
    """
    Test zero-crossing / prime correlation across multiple scales and parameters.
    """
    print("=" * 70)
    print("EXPERIMENT 01: Zero-Crossing Detection")
    print("Hypothesis: Primes correlate with zero-crossings in SEC stress field")
    print("=" * 70)
    
    results = {
        "experiment_id": "exp_01_zero_crossing_detection",
        "hypothesis": "Primes are zero-crossings in oscillatory attractor dynamics",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    # Test 1: Baseline correlation
    print("\n[Test 1] Baseline: n_max=100K, k=9, λ=0.99")
    print("-" * 50)
    
    analysis = full_oscillation_analysis(
        n_max=100000,
        factor_base_size=9,
        window=13,
        lam=0.99,
        proximity=2
    )
    
    zc = analysis["zero_crossings"]
    print(f"  Total crossings: {zc['total_crossings']}")
    print(f"  Crossings near primes: {zc['prime_crossings']}")
    print(f"  Observed fraction: {zc['crossing_fraction']:.4f}")
    print(f"  Expected fraction: {zc['expected_fraction']:.4f}")
    print(f"  Enrichment: {zc['enrichment']:.2f}x")
    print(f"  P-value: {zc['p_value']:.2e}")
    
    if zc['enrichment'] > 1.0 and zc['p_value'] < 0.05:
        print("  ✓ SIGNIFICANT: Primes enriched at zero-crossings")
    else:
        print("  ✗ Not significant")
    
    results["tests"].append({
        "name": "baseline",
        "parameters": analysis["parameters"],
        "zero_crossings": zc,
        "oscillation": analysis["oscillation"],
        "intervals": analysis["intervals"]
    })
    
    # Test 2: Scale dependence
    print("\n[Test 2] Scale Dependence")
    print("-" * 50)
    
    scales = [10000, 50000, 100000, 200000]
    scale_results = []
    
    for n_max in scales:
        analysis = full_oscillation_analysis(
            n_max=n_max,
            factor_base_size=9,
            window=13,
            lam=0.99,
            proximity=2
        )
        zc = analysis["zero_crossings"]
        print(f"  n={n_max:,}: enrichment={zc['enrichment']:.2f}x, p={zc['p_value']:.2e}")
        scale_results.append({
            "n_max": n_max,
            "enrichment": zc["enrichment"],
            "p_value": zc["p_value"],
            "crossing_fraction": zc["crossing_fraction"]
        })
    
    results["tests"].append({
        "name": "scale_dependence",
        "scales": scale_results
    })
    
    # Test 3: λ (lambda) sweep - testing different decay rates
    print("\n[Test 3] Lambda (Decay Rate) Sweep")
    print("-" * 50)
    
    lambdas = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    lambda_results = []
    
    for lam in lambdas:
        analysis = full_oscillation_analysis(
            n_max=100000,
            factor_base_size=9,
            window=13,
            lam=lam,
            proximity=2
        )
        zc = analysis["zero_crossings"]
        print(f"  λ={lam}: enrichment={zc['enrichment']:.2f}x, crossings={zc['total_crossings']}")
        lambda_results.append({
            "lambda": lam,
            "total_crossings": zc["total_crossings"],
            "enrichment": zc["enrichment"],
            "p_value": zc["p_value"]
        })
    
    results["tests"].append({
        "name": "lambda_sweep",
        "lambdas": lambda_results
    })
    
    # Test 4: Proximity sensitivity
    print("\n[Test 4] Proximity Sensitivity")
    print("-" * 50)
    
    proximities = [0, 1, 2, 3, 5, 10]
    prox_results = []
    
    for prox in proximities:
        analysis = full_oscillation_analysis(
            n_max=100000,
            factor_base_size=9,
            window=13,
            lam=0.99,
            proximity=prox
        )
        zc = analysis["zero_crossings"]
        print(f"  proximity={prox}: fraction={zc['crossing_fraction']:.3f}, enrichment={zc['enrichment']:.2f}x")
        prox_results.append({
            "proximity": prox,
            "crossing_fraction": zc["crossing_fraction"],
            "expected_fraction": zc["expected_fraction"],
            "enrichment": zc["enrichment"],
            "p_value": zc["p_value"]
        })
    
    results["tests"].append({
        "name": "proximity_sensitivity",
        "proximities": prox_results
    })
    
    # Test 5: Odd vs All manifold comparison
    print("\n[Test 5] Odd Manifold vs All Numbers")
    print("-" * 50)
    
    from sec_core import compute_sec, prime_sieve
    
    n_max = 100000
    factor_base = FIRST_50_PRIMES[:9]
    sec = compute_sec(n_max=n_max, factor_base=factor_base, window=13, lam=0.99)
    
    # All numbers
    crossings_all, _ = detect_zero_crossings(sec.E, start_idx=100)
    result_all = crossing_prime_correlation(crossings_all, sec.prime_mask, proximity=2)
    
    # Odd numbers only: create a new stress field just for odds
    odds = np.arange(3, n_max + 1, 2)
    E_odd = sec.E[odds]
    prime_mask_odd = sec.prime_mask[odds]
    
    # Detect crossings in odd-indexed E values
    crossings_odd_local, _ = detect_zero_crossings(E_odd, start_idx=50)
    result_odd = crossing_prime_correlation(crossings_odd_local, prime_mask_odd, proximity=1)
    
    print(f"  All numbers: enrichment={result_all.enrichment:.2f}x, p={result_all.p_value:.2e}")
    print(f"  Odd manifold: enrichment={result_odd.enrichment:.2f}x, p={result_odd.p_value:.2e}")
    
    results["tests"].append({
        "name": "manifold_comparison",
        "all_numbers": {
            "total_crossings": result_all.total_crossings,
            "enrichment": result_all.enrichment,
            "p_value": result_all.p_value
        },
        "odd_manifold": {
            "total_crossings": result_odd.total_crossings,
            "enrichment": result_odd.enrichment,
            "p_value": result_odd.p_value
        }
    })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    baseline = results["tests"][0]
    enrichment = baseline["zero_crossings"]["enrichment"]
    p_value = baseline["zero_crossings"]["p_value"]
    
    results["summary"] = {
        "baseline_enrichment": enrichment,
        "baseline_p_value": p_value,
        "hypothesis_supported": enrichment > 1.0 and p_value < 0.05,
        "key_findings": []
    }
    
    if enrichment > 1.0 and p_value < 0.05:
        results["summary"]["key_findings"].append(
            f"Primes are {enrichment:.1f}x enriched at zero-crossings (p={p_value:.2e})"
        )
        print(f"✓ HYPOTHESIS SUPPORTED: Primes {enrichment:.1f}x enriched at crossings")
    else:
        results["summary"]["key_findings"].append(
            "No significant enrichment of primes at zero-crossings"
        )
        print("✗ Hypothesis not supported at this parameter setting")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_results(results, "exp_01_zero_crossing", results_dir)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
