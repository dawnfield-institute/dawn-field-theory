#!/usr/bin/env python3
"""
Experiment 01: Baseline Validation
==================================

Reproduce the original SEC claims and verify enrichment results.

Key claims to verify:
- Top 1% positive I(n) → ~67% primes (3.3x baseline)
- Top 5% positive I(n) → ~65% primes
- Top 10% positive I(n) → ~64% primes

Trace output: results/exp_01_baseline_YYYYMMDD_HHMMSS.json
"""

import sys
import os
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sec_core import (
    compute_sec, run_enrichment_suite, compute_phi_threshold,
    create_trace, get_timestamp, FIRST_50_PRIMES, PHI
)


def run_experiment(n_max: int = 50000, save_trace: bool = True) -> dict:
    """Run baseline validation experiment."""
    
    print("=" * 60)
    print("EXPERIMENT 01: Baseline Validation")
    print("=" * 60)
    
    # Parameters
    factor_base = FIRST_50_PRIMES[:10]
    window = 101
    lam = 0.99
    
    parameters = {
        "n_max": n_max,
        "factor_base": factor_base,
        "window": window,
        "lambda": lam
    }
    
    print(f"\nParameters:")
    for k, v in parameters.items():
        print(f"  {k}: {v}")
    
    # Compute SEC
    print(f"\nComputing SEC for n ≤ {n_max:,}...")
    sec = compute_sec(n_max=n_max, factor_base=factor_base, window=window, lam=lam)
    
    # Run enrichment analysis
    enrichment = run_enrichment_suite(sec)
    
    # Phi threshold
    phi_result = compute_phi_threshold(sec)
    
    # Extract key results
    results = {
        "baseline_prime_rate": enrichment["baseline_prime_rate"],
        "n_analyzed": enrichment["n_analyzed"],
        "enrichment": {
            "positive_I": enrichment["enrichment"]["positive_I"],
            "abs_I": enrichment["enrichment"]["abs_I"],
            "abs_E": enrichment["enrichment"]["abs_E"],
        },
        "phi_threshold": phi_result
    }
    
    # Print results
    print(f"\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    
    baseline = results["baseline_prime_rate"]
    print(f"\nBaseline prime rate (odd): {baseline:.4f}")
    print(f"Numbers analyzed: {results['n_analyzed']:,}")
    
    print(f"\nPositive I(n) enrichment:")
    for q, rate in sorted(results["enrichment"]["positive_I"].items()):
        ratio = rate / baseline
        print(f"  Top {float(q)*100:5.1f}%: {rate:.4f} ({ratio:.2f}x baseline)")
    
    print(f"\nφ-threshold:")
    print(f"  frac(E>0): {phi_result['frac_E_positive']:.6f}")
    print(f"  1/φ target: {1/PHI:.6f}")
    print(f"  error: {phi_result['error_vs_phi']:+.6f}")
    
    # Validation
    validation = {
        "top_1pct_above_60pct": results["enrichment"]["positive_I"][0.01] > 0.60,
        "top_10pct_above_60pct": results["enrichment"]["positive_I"][0.10] > 0.60,
        "enrichment_above_3x": results["enrichment"]["positive_I"][0.01] / baseline > 3.0,
        "phi_within_1pct": abs(phi_result['error_vs_phi']) < 0.01
    }
    
    print(f"\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    for check, passed in validation.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    # Create and save trace
    if save_trace:
        trace = create_trace(
            experiment_id="exp_01_baseline_validation",
            parameters=parameters,
            results=results,
            validation=validation
        )
        
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / f"exp_01_baseline_{trace.timestamp}.json"
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
