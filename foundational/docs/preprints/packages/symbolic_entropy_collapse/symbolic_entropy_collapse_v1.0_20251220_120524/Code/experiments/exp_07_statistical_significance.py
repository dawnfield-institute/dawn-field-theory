#!/usr/bin/env python3
"""
Experiment 07: Statistical Significance of φ-Threshold
=======================================================

Make the golden ratio discovery DEFINITIVE by:
1. Bootstrap confidence intervals for size=9 threshold
2. Null hypothesis test: random factor bases shouldn't produce φ
3. Permutation test: is Fibonacci cascade real or spurious?
4. Large-scale convergence: does error → 0 as n → ∞?

If φ emerges from random noise, it's not significant.
If φ ONLY emerges from prime factor bases with specific structure, it's real.

Trace output: results/exp_07_statistical_YYYYMMDD_HHMMSS.json
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sec_core import (
    prime_sieve, symbolic_entropy, entropy_expectation,
    collapse_impulse, stress_field, create_trace,
    FIRST_50_PRIMES, PHI
)

PHI_INV = 1 / PHI  # 0.6180339887498948


def compute_threshold(n_max: int, factor_base: list, window: int = 101, lam: float = 0.99) -> float:
    """Compute frac(E>0) for odd integers."""
    S = symbolic_entropy(n_max, factor_base)
    S_hat = entropy_expectation(S, window)
    I = collapse_impulse(S, S_hat)
    E = stress_field(I, lam)
    
    odds = np.arange(3, n_max + 1, 2)
    return float(np.mean(E[odds] > 0))


def bootstrap_confidence_interval(n_max: int, factor_base: list, 
                                   n_bootstrap: int = 1000, 
                                   confidence: float = 0.95) -> dict:
    """
    Bootstrap CI for the threshold estimate.
    
    Resamples the odd integers with replacement and computes threshold distribution.
    """
    # Compute full E field once
    S = symbolic_entropy(n_max, factor_base)
    S_hat = entropy_expectation(S)
    I = collapse_impulse(S, S_hat)
    E = stress_field(I)
    
    odds = np.arange(3, n_max + 1, 2)
    E_odds = E[odds]
    
    # Point estimate
    point_estimate = float(np.mean(E_odds > 0))
    
    # Bootstrap
    np.random.seed(42)
    boot_thresholds = []
    n = len(E_odds)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        E_sample = E_odds[idx]
        boot_thresholds.append(float(np.mean(E_sample > 0)))
    
    boot_thresholds = np.array(boot_thresholds)
    
    # CI
    alpha = 1 - confidence
    ci_low = np.percentile(boot_thresholds, 100 * alpha / 2)
    ci_high = np.percentile(boot_thresholds, 100 * (1 - alpha / 2))
    
    # Is 1/φ within the CI?
    phi_in_ci = ci_low <= PHI_INV <= ci_high
    
    # How many SEs away is 1/φ?
    se = np.std(boot_thresholds)
    z_score = (point_estimate - PHI_INV) / se if se > 0 else 0
    
    return {
        'point_estimate': point_estimate,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'se': se,
        'phi_in_ci': phi_in_ci,
        'z_score_from_phi': z_score,
        'p_value_two_sided': 2 * (1 - stats.norm.cdf(abs(z_score))),
        'n_bootstrap': n_bootstrap,
        'confidence': confidence
    }


def null_hypothesis_test(n_max: int, n_random: int = 100) -> dict:
    """
    Null hypothesis: Random factor bases produce similar thresholds.
    
    If true, φ emergence is spurious.
    If false, φ requires specific prime structure.
    """
    # True threshold with size=9 primes
    true_threshold = compute_threshold(n_max, FIRST_50_PRIMES[:9])
    
    # Random factor bases (same size=9 but random integers)
    np.random.seed(42)
    random_thresholds = []
    
    for i in range(n_random):
        # Random odd numbers in similar range to first 9 primes [2,23]
        # Use odd composites and random numbers
        random_base = sorted(np.random.choice(range(2, 50), size=9, replace=False))
        thresh = compute_threshold(n_max, list(random_base))
        random_thresholds.append(thresh)
        
        if (i + 1) % 20 == 0:
            print(f"    Null test {i+1}/{n_random}...")
    
    random_thresholds = np.array(random_thresholds)
    
    # How many random bases hit within 1% of φ?
    close_to_phi = np.sum(np.abs(random_thresholds - PHI_INV) < 0.01)
    
    # p-value: what fraction of random bases are closer to φ than true threshold?
    true_error = abs(true_threshold - PHI_INV)
    random_errors = np.abs(random_thresholds - PHI_INV)
    p_value = np.mean(random_errors <= true_error)
    
    return {
        'true_threshold': true_threshold,
        'true_error': true_error,
        'random_mean': float(np.mean(random_thresholds)),
        'random_std': float(np.std(random_thresholds)),
        'random_min_error': float(np.min(random_errors)),
        'random_max_error': float(np.max(random_errors)),
        'n_close_to_phi': int(close_to_phi),
        'p_value': p_value,
        'null_rejected': p_value < 0.05,
        'n_random': n_random
    }


def permutation_test_fibonacci_cascade(n_max: int, n_permutations: int = 1000) -> dict:
    """
    Test: Is the Fibonacci size cascade real?
    
    Null: Shuffling size-threshold correspondence should produce same pattern.
    """
    # True cascade pattern
    sizes = list(range(1, 26))
    true_thresholds = []
    
    for size in sizes:
        thresh = compute_threshold(n_max, FIRST_50_PRIMES[:size])
        true_thresholds.append(thresh)
    
    true_thresholds = np.array(true_thresholds)
    
    # Measure: correlation between size and |threshold - 1/φ|
    # The cascade shows a minimum at size=9
    errors = np.abs(true_thresholds - PHI_INV)
    
    # Find where minimum occurs
    true_min_idx = np.argmin(errors)
    true_min_size = sizes[true_min_idx]
    true_min_error = errors[true_min_idx]
    
    # Does error decrease then increase? (V-shape around size=9)
    # Compute "V-score": sum of errors at extremes minus error at minimum
    v_score_true = (errors[0] + errors[-1]) / 2 - errors[true_min_idx]
    
    # Permutation test: shuffle thresholds
    np.random.seed(42)
    perm_min_sizes = []
    perm_v_scores = []
    
    for _ in range(n_permutations):
        perm_thresh = np.random.permutation(true_thresholds)
        perm_errors = np.abs(perm_thresh - PHI_INV)
        perm_min_idx = np.argmin(perm_errors)
        perm_min_sizes.append(sizes[perm_min_idx])
        perm_v_scores.append((perm_errors[0] + perm_errors[-1]) / 2 - perm_errors[perm_min_idx])
    
    perm_min_sizes = np.array(perm_min_sizes)
    perm_v_scores = np.array(perm_v_scores)
    
    # p-values
    p_min_at_9 = np.mean(perm_min_sizes == 9)  # How often does random min land at 9?
    p_v_score = np.mean(perm_v_scores >= v_score_true)  # How often is V-shape as strong?
    
    return {
        'true_min_size': true_min_size,
        'true_min_error': true_min_error,
        'true_v_score': v_score_true,
        'perm_min_size_mean': float(np.mean(perm_min_sizes)),
        'perm_min_size_mode': int(stats.mode(perm_min_sizes, keepdims=False).mode),
        'p_min_at_9': p_min_at_9,
        'p_v_score': p_v_score,
        'cascade_significant': p_v_score < 0.05,
        'n_permutations': n_permutations
    }


def convergence_test(scales: list) -> dict:
    """
    Test: Does threshold converge to 1/φ as n → ∞?
    
    If error shrinks systematically, there's genuine convergence.
    """
    results = []
    
    for n_max in scales:
        print(f"  Scale n={n_max:,}...")
        start = time.time()
        thresh = compute_threshold(n_max, FIRST_50_PRIMES[:9])
        elapsed = time.time() - start
        
        results.append({
            'n_max': n_max,
            'threshold': thresh,
            'error': thresh - PHI_INV,
            'abs_error': abs(thresh - PHI_INV),
            'elapsed': elapsed
        })
    
    # Fit error vs 1/sqrt(n) - if converging, error ~ 1/sqrt(n)
    n_vals = np.array([r['n_max'] for r in results])
    errors = np.array([r['abs_error'] for r in results])
    
    # Log-log regression: log(error) = a + b*log(n)
    # If b < 0, error decreases with n
    log_n = np.log(n_vals)
    log_err = np.log(errors + 1e-10)  # Avoid log(0)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_err)
    
    return {
        'scale_results': results,
        'convergence_slope': slope,  # Should be negative
        'convergence_r2': r_value**2,
        'convergence_p_value': p_value,
        'converging': slope < 0 and p_value < 0.05,
        'extrapolated_limit': float(np.exp(intercept + slope * np.log(1e9)))  # At n=1B
    }


def run_experiment(n_max: int = 50000, save_trace: bool = True) -> dict:
    """Run complete statistical significance experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 07: Statistical Significance of φ-Threshold")
    print("=" * 70)
    print(f"\nTarget: 1/φ = {PHI_INV:.10f}")
    print(f"Base n_max: {n_max:,}")
    
    parameters = {"n_max": n_max}
    results = {}
    
    # Test 1: Bootstrap CI
    print(f"\n" + "-" * 70)
    print("TEST 1: Bootstrap Confidence Interval (size=9)")
    print("-" * 70)
    
    boot_result = bootstrap_confidence_interval(n_max, FIRST_50_PRIMES[:9], n_bootstrap=2000)
    results['bootstrap_ci'] = boot_result
    
    print(f"\n  Point estimate: {boot_result['point_estimate']:.6f}")
    print(f"  95% CI: [{boot_result['ci_low']:.6f}, {boot_result['ci_high']:.6f}]")
    print(f"  Standard error: {boot_result['se']:.6f}")
    print(f"  1/φ in CI: {'YES ✅' if boot_result['phi_in_ci'] else 'NO ❌'}")
    print(f"  Z-score from φ: {boot_result['z_score_from_phi']:.2f}")
    print(f"  p-value (H₀: θ = 1/φ): {boot_result['p_value_two_sided']:.4f}")
    
    # Test 2: Null hypothesis - random bases
    print(f"\n" + "-" * 70)
    print("TEST 2: Null Hypothesis - Random Factor Bases")
    print("-" * 70)
    
    null_result = null_hypothesis_test(n_max, n_random=100)
    results['null_hypothesis'] = null_result
    
    print(f"\n  True threshold (primes): {null_result['true_threshold']:.6f}")
    print(f"  True error from φ: {null_result['true_error']:.6f}")
    print(f"  Random bases mean: {null_result['random_mean']:.6f} ± {null_result['random_std']:.6f}")
    print(f"  Random bases min error: {null_result['random_min_error']:.6f}")
    print(f"  # random close to φ (<1%): {null_result['n_close_to_phi']}/{null_result['n_random']}")
    print(f"  p-value (H₀: random = prime): {null_result['p_value']:.4f}")
    print(f"  Null rejected: {'YES ✅' if null_result['null_rejected'] else 'NO ❌'}")
    
    # Test 3: Fibonacci cascade permutation test
    print(f"\n" + "-" * 70)
    print("TEST 3: Fibonacci Cascade Permutation Test")
    print("-" * 70)
    
    perm_result = permutation_test_fibonacci_cascade(n_max, n_permutations=1000)
    results['permutation_test'] = perm_result
    
    print(f"\n  True minimum at size: {perm_result['true_min_size']}")
    print(f"  True minimum error: {perm_result['true_min_error']:.6f}")
    print(f"  True V-score: {perm_result['true_v_score']:.4f}")
    print(f"  Random min size mode: {perm_result['perm_min_size_mode']}")
    print(f"  p(min at 9 by chance): {perm_result['p_min_at_9']:.4f}")
    print(f"  p(V-score by chance): {perm_result['p_v_score']:.4f}")
    print(f"  Cascade significant: {'YES ✅' if perm_result['cascade_significant'] else 'NO ❌'}")
    
    # Test 4: Convergence
    print(f"\n" + "-" * 70)
    print("TEST 4: Large-Scale Convergence")
    print("-" * 70)
    
    scales = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    conv_result = convergence_test(scales)
    results['convergence'] = conv_result
    
    print(f"\n  {'Scale':>12} {'Threshold':>12} {'Error':>12}")
    print("  " + "-" * 40)
    for r in conv_result['scale_results']:
        print(f"  {r['n_max']:>12,} {r['threshold']:>12.6f} {r['error']:>+12.6f}")
    
    print(f"\n  Convergence slope (log-log): {conv_result['convergence_slope']:.4f}")
    print(f"  R²: {conv_result['convergence_r2']:.4f}")
    print(f"  p-value: {conv_result['convergence_p_value']:.4f}")
    print(f"  Converging to φ: {'YES ✅' if conv_result['converging'] else 'NO ❌'}")
    
    # Validation summary
    validation = {
        'phi_in_95pct_ci': boot_result['phi_in_ci'],
        'null_rejected_p05': null_result['null_rejected'],
        'cascade_significant_p05': perm_result['cascade_significant'],
        'converging_to_phi': conv_result['converging'],
        'overall_definitive': (
            boot_result['phi_in_ci'] and
            null_result['null_rejected'] and
            perm_result['cascade_significant']
        )
    }
    
    print(f"\n" + "=" * 70)
    print("OVERALL VALIDATION")
    print("=" * 70)
    for check, passed in validation.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    if validation['overall_definitive']:
        print(f"\n  🎯 DEFINITIVE: φ emergence is STATISTICALLY SIGNIFICANT")
    else:
        print(f"\n  ⚠️  NOT DEFINITIVE: Further investigation needed")
    
    # Save trace
    if save_trace:
        trace = create_trace(
            experiment_id="exp_07_statistical_significance",
            parameters=parameters,
            results=results,
            validation=validation
        )
        
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / f"exp_07_statistical_{trace.timestamp}.json"
        trace.save(str(filepath))
        print(f"\nTrace saved: {filepath.name}")
    
    return {'parameters': parameters, 'results': results, 'validation': validation}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_max", type=int, default=50000)
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    
    run_experiment(n_max=args.n_max, save_trace=not args.no_trace)
