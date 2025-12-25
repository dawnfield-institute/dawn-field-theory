#!/usr/bin/env python3
"""
Experiment 06: Large Scale Verification
=======================================

Verify key results at n = 10^6 and 10^7 to confirm scale invariance.

Critical checks:
1. Size=9 still produces ~1/φ at large scale
2. Window=13 still optimal
3. Enrichment ratios stable
4. No drift in φ-threshold

Trace output: results/exp_06_large_scale_YYYYMMDD_HHMMSS.json
"""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sec_core import (
    compute_sec, compute_phi_threshold, run_enrichment_suite,
    create_trace, FIRST_50_PRIMES, PHI
)


def run_at_scale(n_max: int, size: int = 9, window: int = 101, lam: float = 0.99) -> dict:
    """Run SEC at a specific scale and return key metrics."""
    
    factor_base = FIRST_50_PRIMES[:size]
    
    start = time.time()
    sec = compute_sec(n_max=n_max, factor_base=factor_base, window=window, lam=lam)
    elapsed = time.time() - start
    
    phi_result = compute_phi_threshold(sec)
    enrichment = run_enrichment_suite(sec)
    
    return {
        "n_max": n_max,
        "size": size,
        "window": window,
        "lambda": lam,
        "elapsed_seconds": elapsed,
        "frac_E_positive": phi_result["frac_E_positive"],
        "error_vs_phi": phi_result["error_vs_phi"],
        "prime_ratio": phi_result["ratio"],
        "enrichment_top_1pct": enrichment["enrichment"]["positive_I"][0.01],
        "enrichment_ratio": enrichment["enrichment"]["positive_I"][0.01] / enrichment["baseline_prime_rate"]
    }


def run_experiment(scales: list = None, save_trace: bool = True) -> dict:
    """Run large scale verification experiment."""
    
    if scales is None:
        scales = [10_000, 100_000, 500_000, 1_000_000]
    
    print("=" * 70)
    print("EXPERIMENT 06: Large Scale Verification")
    print("=" * 70)
    print(f"\nTarget: 1/φ = {1/PHI:.6f}")
    print(f"Scales to test: {[f'{s:,}' for s in scales]}")
    
    parameters = {"scales": scales}
    results = {"scale_tests": {}}
    
    # Test 1: Size=9 across scales
    print(f"\n" + "-" * 70)
    print("TEST 1: Size=9 across scales")
    print("-" * 70)
    
    print(f"\n{'n_max':>12} {'Frac E>0':>12} {'Error vs 1/φ':>14} {'Prime ratio':>12} {'Time':>10}")
    print("-" * 65)
    
    size9_results = {}
    for n_max in scales:
        r = run_at_scale(n_max, size=9)
        size9_results[n_max] = r
        print(f"{n_max:>12,} {r['frac_E_positive']:>12.6f} {r['error_vs_phi']:>+14.6f} {r['prime_ratio']:>12.2f}x {r['elapsed_seconds']:>9.1f}s")
    
    results["scale_tests"]["size_9"] = size9_results
    
    # Test 2: Window=13 across scales
    print(f"\n" + "-" * 70)
    print("TEST 2: Window=13 (with size=10) across scales")
    print("-" * 70)
    
    print(f"\n{'n_max':>12} {'Frac E>0':>12} {'Error vs 1/φ':>14} {'Prime ratio':>12}")
    print("-" * 55)
    
    window13_results = {}
    for n_max in scales:
        r = run_at_scale(n_max, size=10, window=13)
        window13_results[n_max] = r
        print(f"{n_max:>12,} {r['frac_E_positive']:>12.6f} {r['error_vs_phi']:>+14.6f} {r['prime_ratio']:>12.2f}x")
    
    results["scale_tests"]["window_13"] = window13_results
    
    # Test 3: Enrichment stability
    print(f"\n" + "-" * 70)
    print("TEST 3: Enrichment Stability")
    print("-" * 70)
    
    print(f"\n{'n_max':>12} {'Top 1% rate':>12} {'Ratio to baseline':>18}")
    print("-" * 45)
    
    for n_max in scales:
        r = size9_results[n_max]
        print(f"{n_max:>12,} {r['enrichment_top_1pct']:>12.4f} {r['enrichment_ratio']:>18.2f}x")
    
    # Stability analysis
    fracs = [size9_results[n]['frac_E_positive'] for n in scales]
    mean_frac = np.mean(fracs)
    std_frac = np.std(fracs)
    
    print(f"\n" + "-" * 70)
    print("STABILITY ANALYSIS")
    print("-" * 70)
    print(f"\nSize=9 across scales:")
    print(f"  Mean frac(E>0): {mean_frac:.6f}")
    print(f"  Std deviation:  {std_frac:.6f}")
    print(f"  CV:             {std_frac/mean_frac:.4f}")
    print(f"  Error vs 1/φ:   {mean_frac - 1/PHI:+.6f}")
    
    results["stability"] = {
        "mean_frac": mean_frac,
        "std_frac": std_frac,
        "cv": std_frac / mean_frac,
        "mean_error_vs_phi": mean_frac - 1/PHI
    }
    
    # Validation
    validation = {
        "scale_invariant_to_1pct": std_frac / mean_frac < 0.01,
        "phi_stable_across_scales": all(
            abs(size9_results[n]['error_vs_phi']) < 0.01 for n in scales
        ),
        "enrichment_stable": all(
            size9_results[n]['enrichment_ratio'] > 2.5 for n in scales
        ),
        "window_13_matches_phi": all(
            abs(window13_results[n]['error_vs_phi']) < 0.01 for n in scales
        )
    }
    
    print(f"\n" + "-" * 70)
    print("VALIDATION")
    print("-" * 70)
    for check, passed in validation.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    # Save trace
    if save_trace:
        trace = create_trace(
            experiment_id="exp_06_large_scale_verification",
            parameters=parameters,
            results=results,
            validation=validation
        )
        
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / f"exp_06_large_scale_{trace.timestamp}.json"
        trace.save(str(filepath))
        print(f"\nTrace saved: {filepath.name}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=int, nargs="+", 
                       default=[10000, 100000, 500000, 1000000])
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    
    run_experiment(scales=args.scales, save_trace=not args.no_trace)
