#!/usr/bin/env python3
"""
Experiment 02: Factor Base Independence
=======================================

Critical test: Does SEC detect primes OUTSIDE the factor base?

If SEC is genuinely predictive, it should work for primes it can't
directly measure (primes > max(factor_base)).

If correlation only exists for primes IN the factor base, SEC is 
measuring factor-base membership (tautological).

Trace output: results/exp_02_independence_YYYYMMDD_HHMMSS.json
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sec_core import (
    prime_sieve, symbolic_entropy, entropy_expectation,
    collapse_impulse, stress_field, enrichment_analysis,
    create_trace, FIRST_50_PRIMES, PHI
)


def test_external_primes(factor_base: list, n_max: int = 50000) -> dict:
    """
    Test enrichment for primes OUTSIDE the factor base.
    
    This is the critical independence test.
    """
    prime_mask_full, primes = prime_sieve(n_max)
    S = symbolic_entropy(n_max, factor_base)
    S_hat = entropy_expectation(S)
    I = collapse_impulse(S, S_hat)
    
    # Odd numbers >= 3
    idx = np.arange(3, n_max + 1, 2)
    
    # Create mask for primes OUTSIDE the factor base
    max_factor = max(factor_base)
    external_prime_mask = np.zeros(n_max + 1, dtype=bool)
    for p in primes:
        if p > max_factor:
            external_prime_mask[p] = True
    
    pm_external = external_prime_mask[idx]
    pm_full = prime_mask_full[idx]
    
    baseline_all = float(pm_full.mean())
    baseline_external = float(pm_external.mean())
    
    # Test enrichment on external primes only
    pos_I = np.clip(I[idx], 0, None)
    
    enrichment_all = enrichment_analysis(pos_I, pm_full, [0.01, 0.05, 0.10])
    enrichment_ext = enrichment_analysis(pos_I, pm_external, [0.01, 0.05, 0.10])
    
    return {
        "factor_base": factor_base,
        "max_factor": max_factor,
        "baseline_all_primes": baseline_all,
        "baseline_external_primes": baseline_external,
        "enrichment_all_primes": enrichment_all,
        "enrichment_external_primes": enrichment_ext,
        "external_to_all_ratio": {
            q: enrichment_ext[q] / enrichment_all[q] if enrichment_all[q] > 0 else 0
            for q in enrichment_all.keys()
        }
    }


def test_control_bases(n_max: int = 50000) -> dict:
    """Test with various control factor bases."""
    
    results = {}
    
    # Prime factor bases
    results["small_primes_2357"] = test_factor_base(
        [2, 3, 5, 7], n_max, "Small primes (2,3,5,7)"
    )
    results["first_10_primes"] = test_factor_base(
        FIRST_50_PRIMES[:10], n_max, "First 10 primes"
    )
    results["larger_primes_11_29"] = test_factor_base(
        [11, 13, 17, 19, 23, 29], n_max, "Larger primes (11-29)"
    )
    
    # Control bases (should fail)
    results["composites_control"] = test_factor_base(
        [4, 6, 9, 10, 12, 14, 15, 16, 18, 20], n_max, "Composites (control)"
    )
    
    np.random.seed(42)
    random_odds = sorted(np.random.choice(range(3, 100, 2), 10, replace=False))
    results["random_odds_control"] = test_factor_base(
        list(random_odds), n_max, "Random odds (control)"
    )
    
    return results


def test_factor_base(factor_base: list, n_max: int, name: str) -> dict:
    """Test a single factor base and return enrichment stats."""
    
    prime_mask, _ = prime_sieve(n_max)
    S = symbolic_entropy(n_max, factor_base)
    S_hat = entropy_expectation(S)
    I = collapse_impulse(S, S_hat)
    
    idx = np.arange(3, n_max + 1, 2)
    pm = prime_mask[idx]
    baseline = float(pm.mean())
    
    pos_I = np.clip(I[idx], 0, None)
    enrichment = enrichment_analysis(pos_I, pm, [0.01, 0.05, 0.10])
    
    return {
        "name": name,
        "factor_base": factor_base,
        "baseline": baseline,
        "enrichment": enrichment,
        "top_1pct_ratio": enrichment[0.01] / baseline if baseline > 0 else 0
    }


def run_experiment(n_max: int = 50000, save_trace: bool = True) -> dict:
    """Run factor base independence experiment."""
    
    print("=" * 60)
    print("EXPERIMENT 02: Factor Base Independence")
    print("=" * 60)
    
    parameters = {"n_max": n_max}
    
    # Run control tests
    print("\nTesting various factor bases...")
    control_results = test_control_bases(n_max)
    
    print(f"\n" + "-" * 60)
    print("FACTOR BASE COMPARISON")
    print("-" * 60)
    print(f"{'Name':<25} {'Top 1% rate':>12} {'Ratio':>10}")
    print("-" * 50)
    
    for key, res in control_results.items():
        ratio = res['top_1pct_ratio']
        rate = res['enrichment'][0.01]
        print(f"{res['name']:<25} {rate:>12.4f} {ratio:>10.2f}x")
    
    # Run critical external primes test
    print(f"\n" + "-" * 60)
    print("CRITICAL TEST: External Primes")
    print("-" * 60)
    
    external_tests = {}
    for fb, name in [
        ([2, 3, 5, 7], "FB={2,3,5,7}"),
        (FIRST_50_PRIMES[:6], "FB=first 6 primes"),
    ]:
        result = test_external_primes(fb, n_max)
        external_tests[name] = result
        
        print(f"\n{name} (testing primes > {result['max_factor']}):")
        print(f"  All primes enrichment (top 1%): {result['enrichment_all_primes'][0.01]:.4f}")
        print(f"  External primes enrichment:     {result['enrichment_external_primes'][0.01]:.4f}")
        print(f"  External/All ratio:             {result['external_to_all_ratio'][0.01]:.4f}")
    
    # Compile results
    results = {
        "control_tests": control_results,
        "external_primes_tests": external_tests
    }
    
    # Validation
    validation = {
        "prime_bases_enrich": all(
            control_results[k]['top_1pct_ratio'] > 1.5 
            for k in ["small_primes_2357", "first_10_primes"]
        ),
        "composite_control_fails": control_results["composites_control"]['top_1pct_ratio'] < 1.2,
        "external_primes_detected": all(
            ext['enrichment_external_primes'][0.01] > ext['baseline_external_primes'] * 1.5
            for ext in external_tests.values()
        ),
        "independence_confirmed": all(
            ext['external_to_all_ratio'][0.01] > 0.8
            for ext in external_tests.values()
        )
    }
    
    print(f"\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    for check, passed in validation.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    # Save trace
    if save_trace:
        trace = create_trace(
            experiment_id="exp_02_factor_base_independence",
            parameters=parameters,
            results=results,
            validation=validation
        )
        
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / f"exp_02_independence_{trace.timestamp}.json"
        trace.save(str(filepath))
        print(f"\nTrace saved: {filepath.name}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_max", type=int, default=50000)
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    
    run_experiment(n_max=args.n_max, save_trace=not args.no_trace)
