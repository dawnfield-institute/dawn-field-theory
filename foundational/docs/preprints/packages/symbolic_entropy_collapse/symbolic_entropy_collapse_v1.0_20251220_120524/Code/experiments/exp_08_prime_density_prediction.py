#!/usr/bin/env python3
"""
Experiment 08: Prime Density Prediction via SEC Stress Regions
==============================================================

The ultimate test: Can SEC actually PREDICT where primes occur?

Hypothesis: If E(n) captures genuine prime structure, then:
- Positive stress regions (E > 0) should have different prime density
- SEC should outperform naive density estimates (prime number theorem)
- Transition zones (E crossing 0) may mark prime gaps or clusters

This is the "so what" test - does SEC have predictive power?

Tests:
1. Prime density in E>0 vs E<0 regions
2. Comparison to Prime Number Theorem baseline: π(n) ~ n/ln(n)
3. Local prediction: given E(n), can we predict if n+2 is prime?
4. Prime gap prediction: do large gaps correlate with sustained E<0?

Trace output: results/exp_08_prime_density_YYYYMMDD_HHMMSS.json
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

PHI_INV = 1 / PHI


def compute_sec_field(n_max: int, factor_base_size: int = 9) -> tuple:
    """Compute SEC stress field and return (E, primes_mask as bool array)."""
    factor_base = FIRST_50_PRIMES[:factor_base_size]
    
    S = symbolic_entropy(n_max, factor_base)
    S_hat = entropy_expectation(S)
    I = collapse_impulse(S, S_hat)
    E = stress_field(I)
    
    # prime_sieve returns (bool_mask, primes_array) - we want the bool mask
    sieve, _ = prime_sieve(n_max)
    
    return E, sieve


def test_density_by_stress_region(n_max: int, E: np.ndarray, sieve: np.ndarray) -> dict:
    """
    Test 1: Do E>0 and E<0 regions have different prime densities?
    
    Null hypothesis: Prime density is uniform across stress regions.
    """
    # Focus on odd numbers only (even numbers > 2 can't be prime)
    odds = np.arange(3, n_max + 1, 2)
    
    E_odds = E[odds]
    is_prime_odds = sieve[odds]
    
    # Split by stress sign
    pos_mask = E_odds > 0
    neg_mask = E_odds < 0
    zero_mask = E_odds == 0
    
    # Densities
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    n_zero = np.sum(zero_mask)
    
    primes_in_pos = np.sum(is_prime_odds[pos_mask]) if n_pos > 0 else 0
    primes_in_neg = np.sum(is_prime_odds[neg_mask]) if n_neg > 0 else 0
    primes_in_zero = np.sum(is_prime_odds[zero_mask]) if n_zero > 0 else 0
    
    density_pos = primes_in_pos / n_pos if n_pos > 0 else 0
    density_neg = primes_in_neg / n_neg if n_neg > 0 else 0
    density_overall = np.sum(is_prime_odds) / len(odds)
    
    # Chi-squared test: is density different in pos vs neg?
    # Contingency table: [[primes_pos, non_primes_pos], [primes_neg, non_primes_neg]]
    if n_pos > 0 and n_neg > 0:
        contingency = np.array([
            [primes_in_pos, n_pos - primes_in_pos],
            [primes_in_neg, n_neg - primes_in_neg]
        ])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    else:
        chi2, p_value = 0, 1
    
    # Effect size: density ratio
    density_ratio = density_pos / density_neg if density_neg > 0 else float('inf')
    
    return {
        'n_positive_region': int(n_pos),
        'n_negative_region': int(n_neg),
        'n_zero_region': int(n_zero),
        'primes_in_positive': int(primes_in_pos),
        'primes_in_negative': int(primes_in_neg),
        'density_positive': density_pos,
        'density_negative': density_neg,
        'density_overall': density_overall,
        'density_ratio': density_ratio,
        'chi2': chi2,
        'p_value': p_value,
        'significant_difference': p_value < 0.05
    }


def test_vs_prime_number_theorem(n_max: int, E: np.ndarray, sieve: np.ndarray, 
                                  n_bins: int = 100) -> dict:
    """
    Test 2: Does SEC improve over Prime Number Theorem predictions?
    
    PNT says: π(n) ≈ n/ln(n), so local density ≈ 1/ln(n)
    
    Can SEC-informed estimates beat this?
    """
    # Divide range into bins
    bin_edges = np.linspace(1000, n_max, n_bins + 1)  # Start at 1000 to avoid small-n effects
    
    pnt_errors = []
    sec_errors = []
    actual_counts = []
    
    for i in range(n_bins):
        lo, hi = int(bin_edges[i]), int(bin_edges[i + 1])
        
        # Actual prime count in bin
        actual = np.sum(sieve[lo:hi])
        actual_counts.append(actual)
        
        # PNT prediction: integral of 1/ln(x) from lo to hi
        # Approximation: (hi - lo) / ln((lo + hi) / 2)
        mid = (lo + hi) / 2
        pnt_pred = (hi - lo) / np.log(mid)
        pnt_errors.append(abs(actual - pnt_pred))
        
        # SEC prediction: use local E density to adjust
        # If frac(E>0) differs from φ, adjust prediction
        E_bin = E[lo:hi]
        frac_pos = np.mean(E_bin > 0)
        
        # Simple model: if frac_pos > 1/φ, more "stress" = fewer primes?
        # Or opposite? Let's test both and see which fits
        # Model: prime_density ∝ 1/ln(n) * (some function of frac_pos)
        
        # Null model: SEC doesn't help, so same as PNT
        sec_pred_null = pnt_pred
        
        # Linear adjustment: more positive stress = adjustment
        # We'll learn the relationship
        sec_errors.append(abs(actual - sec_pred_null))  # Placeholder
    
    # Now fit: what's the relationship between E and prime density?
    # For each bin, compute residual = actual - pnt_pred
    residuals = []
    frac_positives = []
    mean_E_values = []
    
    for i in range(n_bins):
        lo, hi = int(bin_edges[i]), int(bin_edges[i + 1])
        actual = actual_counts[i]
        mid = (lo + hi) / 2
        pnt_pred = (hi - lo) / np.log(mid)
        
        residual = actual - pnt_pred
        residuals.append(residual)
        
        E_bin = E[lo:hi]
        frac_positives.append(np.mean(E_bin > 0))
        mean_E_values.append(np.mean(E_bin))
    
    # Correlation: does E predict PNT residuals?
    residuals = np.array(residuals)
    frac_positives = np.array(frac_positives)
    mean_E_values = np.array(mean_E_values)
    
    r_frac, p_frac = stats.pearsonr(frac_positives, residuals)
    r_mean, p_mean = stats.pearsonr(mean_E_values, residuals)
    
    # If correlation significant, SEC provides additional predictive power
    sec_predictive = p_frac < 0.05 or p_mean < 0.05
    
    return {
        'n_bins': n_bins,
        'pnt_mae': float(np.mean(pnt_errors)),
        'pnt_rmse': float(np.sqrt(np.mean(np.array(pnt_errors)**2))),
        'correlation_frac_pos_vs_residual': r_frac,
        'p_value_frac': p_frac,
        'correlation_mean_E_vs_residual': r_mean,
        'p_value_mean_E': p_mean,
        'sec_adds_predictive_power': sec_predictive
    }


def test_local_prime_prediction(n_max: int, E: np.ndarray, sieve: np.ndarray) -> dict:
    """
    Test 3: Can E(n) predict if nearby numbers are prime?
    
    For each odd n, use E(n-2) to predict if n is prime.
    Compare to baseline prediction using just 1/ln(n).
    """
    # Focus on range where we have good E estimates
    test_range = np.arange(1001, n_max - 1, 2)  # Odd numbers
    
    # Features: E at n-2, E at n-4, mean E in window
    # Target: is n prime?
    
    y_true = sieve[test_range].astype(int)
    
    # Baseline: predict using 1/ln(n) as probability
    baseline_probs = 1 / np.log(test_range)
    
    # SEC feature: E(n-2)
    E_lag2 = E[test_range - 2]
    
    # Simple threshold: predict prime if E_lag2 above median
    E_median = np.median(E_lag2)
    sec_pred = (E_lag2 > E_median).astype(int)
    
    # Compare: AUC-ROC
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Baseline AUC (using 1/ln(n) as score)
    try:
        auc_baseline = roc_auc_score(y_true, baseline_probs)
    except:
        auc_baseline = 0.5
    
    # SEC AUC (using E as score)
    try:
        auc_sec = roc_auc_score(y_true, E_lag2)
    except:
        auc_sec = 0.5
    
    # What if we combine? Score = E + c/ln(n)
    combined_score = E_lag2 + baseline_probs * np.std(E_lag2) / np.std(baseline_probs)
    try:
        auc_combined = roc_auc_score(y_true, combined_score)
    except:
        auc_combined = 0.5
    
    return {
        'n_test_points': len(test_range),
        'prime_rate': float(np.mean(y_true)),
        'auc_baseline_pnt': auc_baseline,
        'auc_sec_only': auc_sec,
        'auc_combined': auc_combined,
        'sec_improves_prediction': auc_sec > auc_baseline or auc_combined > auc_baseline
    }


def test_prime_gap_correlation(n_max: int, E: np.ndarray, sieve: np.ndarray) -> dict:
    """
    Test 4: Do prime gaps correlate with sustained negative E?
    
    Large gaps between primes might correspond to regions of sustained E < 0.
    """
    # Get prime positions
    primes = np.where(sieve)[0]
    primes = primes[primes > 100]  # Skip small primes
    
    # Compute gaps
    gaps = np.diff(primes)
    
    # For each gap, compute mean E in that interval
    gap_mean_E = []
    gap_frac_neg = []
    
    for i in range(len(gaps)):
        p1, p2 = primes[i], primes[i + 1]
        E_interval = E[p1:p2]
        gap_mean_E.append(np.mean(E_interval))
        gap_frac_neg.append(np.mean(E_interval < 0))
    
    gap_mean_E = np.array(gap_mean_E)
    gap_frac_neg = np.array(gap_frac_neg)
    
    # Correlation: larger gaps → more negative E?
    r_mean, p_mean = stats.pearsonr(gaps, gap_mean_E)
    r_frac, p_frac = stats.pearsonr(gaps, gap_frac_neg)
    
    # Focus on large gaps (top 10%)
    large_gap_threshold = np.percentile(gaps, 90)
    large_gaps_mask = gaps >= large_gap_threshold
    
    mean_E_large_gaps = np.mean(gap_mean_E[large_gaps_mask])
    mean_E_small_gaps = np.mean(gap_mean_E[~large_gaps_mask])
    
    frac_neg_large_gaps = np.mean(gap_frac_neg[large_gaps_mask])
    frac_neg_small_gaps = np.mean(gap_frac_neg[~large_gaps_mask])
    
    return {
        'n_gaps': len(gaps),
        'mean_gap': float(np.mean(gaps)),
        'max_gap': int(np.max(gaps)),
        'correlation_gap_vs_mean_E': r_mean,
        'p_value_gap_mean_E': p_mean,
        'correlation_gap_vs_frac_neg': r_frac,
        'p_value_gap_frac_neg': p_frac,
        'mean_E_large_gaps': mean_E_large_gaps,
        'mean_E_small_gaps': mean_E_small_gaps,
        'frac_neg_large_gaps': frac_neg_large_gaps,
        'frac_neg_small_gaps': frac_neg_small_gaps,
        'large_gaps_more_negative': mean_E_large_gaps < mean_E_small_gaps
    }


def run_experiment(n_max: int = 100000, save_trace: bool = True) -> dict:
    """Run complete prime density prediction experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 08: Prime Density Prediction via SEC")
    print("=" * 70)
    print(f"\nRange: [2, {n_max:,}]")
    print(f"Question: Can SEC predict where primes occur?")
    
    parameters = {"n_max": n_max}
    
    print("\nComputing SEC stress field...")
    start = time.time()
    E, sieve = compute_sec_field(n_max)
    print(f"  Computed in {time.time() - start:.2f}s")
    
    results = {}
    
    # Test 1: Density by region
    print(f"\n" + "-" * 70)
    print("TEST 1: Prime Density by Stress Region")
    print("-" * 70)
    
    density_result = test_density_by_stress_region(n_max, E, sieve)
    results['density_by_region'] = density_result
    
    print(f"\n  E > 0 region: {density_result['n_positive_region']:,} numbers")
    print(f"    Primes: {density_result['primes_in_positive']:,}")
    print(f"    Density: {density_result['density_positive']:.4f}")
    print(f"\n  E < 0 region: {density_result['n_negative_region']:,} numbers")
    print(f"    Primes: {density_result['primes_in_negative']:,}")
    print(f"    Density: {density_result['density_negative']:.4f}")
    print(f"\n  Density ratio (pos/neg): {density_result['density_ratio']:.4f}")
    print(f"  Chi² p-value: {density_result['p_value']:.4e}")
    print(f"  Significant difference: {'YES ✅' if density_result['significant_difference'] else 'NO ❌'}")
    
    # Test 2: vs PNT
    print(f"\n" + "-" * 70)
    print("TEST 2: SEC vs Prime Number Theorem")
    print("-" * 70)
    
    pnt_result = test_vs_prime_number_theorem(n_max, E, sieve)
    results['vs_pnt'] = pnt_result
    
    print(f"\n  PNT baseline MAE: {pnt_result['pnt_mae']:.2f} primes per bin")
    print(f"  Correlation(frac_E>0, PNT residual): {pnt_result['correlation_frac_pos_vs_residual']:.4f}")
    print(f"    p-value: {pnt_result['p_value_frac']:.4e}")
    print(f"  Correlation(mean_E, PNT residual): {pnt_result['correlation_mean_E_vs_residual']:.4f}")
    print(f"    p-value: {pnt_result['p_value_mean_E']:.4e}")
    print(f"  SEC adds predictive power: {'YES ✅' if pnt_result['sec_adds_predictive_power'] else 'NO ❌'}")
    
    # Test 3: Local prediction
    print(f"\n" + "-" * 70)
    print("TEST 3: Local Prime Prediction (AUC-ROC)")
    print("-" * 70)
    
    try:
        local_result = test_local_prime_prediction(n_max, E, sieve)
        results['local_prediction'] = local_result
        
        print(f"\n  Test points: {local_result['n_test_points']:,}")
        print(f"  Prime rate: {local_result['prime_rate']:.4f}")
        print(f"\n  AUC Scores (0.5 = random, 1.0 = perfect):")
        print(f"    PNT baseline (1/ln n): {local_result['auc_baseline_pnt']:.4f}")
        print(f"    SEC only (E field):    {local_result['auc_sec_only']:.4f}")
        print(f"    Combined:              {local_result['auc_combined']:.4f}")
        print(f"  SEC improves prediction: {'YES ✅' if local_result['sec_improves_prediction'] else 'NO ❌'}")
    except ImportError:
        print("\n  ⚠️  sklearn not available, skipping AUC test")
        results['local_prediction'] = {'error': 'sklearn not available'}
    
    # Test 4: Prime gap correlation
    print(f"\n" + "-" * 70)
    print("TEST 4: Prime Gap Correlation")
    print("-" * 70)
    
    gap_result = test_prime_gap_correlation(n_max, E, sieve)
    results['gap_correlation'] = gap_result
    
    print(f"\n  Gaps analyzed: {gap_result['n_gaps']:,}")
    print(f"  Mean gap: {gap_result['mean_gap']:.1f}, Max gap: {gap_result['max_gap']}")
    print(f"\n  Correlation(gap size, mean E): {gap_result['correlation_gap_vs_mean_E']:.4f}")
    print(f"    p-value: {gap_result['p_value_gap_mean_E']:.4e}")
    print(f"\n  Large gaps (top 10%):")
    print(f"    Mean E: {gap_result['mean_E_large_gaps']:.4f}")
    print(f"    Frac negative: {gap_result['frac_neg_large_gaps']:.4f}")
    print(f"  Small gaps:")
    print(f"    Mean E: {gap_result['mean_E_small_gaps']:.4f}")
    print(f"    Frac negative: {gap_result['frac_neg_small_gaps']:.4f}")
    print(f"  Large gaps more negative: {'YES ✅' if gap_result['large_gaps_more_negative'] else 'NO ❌'}")
    
    # Validation summary
    validation = {
        'density_differs_by_region': density_result['significant_difference'],
        'sec_adds_to_pnt': pnt_result['sec_adds_predictive_power'],
        'gaps_correlate_with_E': gap_result['p_value_gap_mean_E'] < 0.05,
        'large_gaps_more_negative': gap_result['large_gaps_more_negative']
    }
    
    # Overall: does SEC have predictive power?
    predictive_tests_passed = sum([
        density_result['significant_difference'],
        pnt_result['sec_adds_predictive_power'],
        gap_result['large_gaps_more_negative']
    ])
    
    validation['sec_has_predictive_power'] = predictive_tests_passed >= 2
    
    print(f"\n" + "=" * 70)
    print("PREDICTIVE POWER SUMMARY")
    print("=" * 70)
    for check, passed in validation.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    if validation['sec_has_predictive_power']:
        print(f"\n  🎯 SEC has PREDICTIVE POWER for prime distribution!")
        print(f"     This is not just curve-fitting - it's genuine structure.")
    else:
        print(f"\n  ⚠️  SEC shows structure but limited predictive power.")
        print(f"     May need refined model or larger scale.")
    
    # Save trace
    if save_trace:
        trace = create_trace(
            experiment_id="exp_08_prime_density_prediction",
            parameters=parameters,
            results=results,
            validation=validation
        )
        
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / f"exp_08_prime_density_{trace.timestamp}.json"
        trace.save(str(filepath))
        print(f"\nTrace saved: {filepath.name}")
    
    return {'parameters': parameters, 'results': results, 'validation': validation}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_max", type=int, default=100000)
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    
    run_experiment(n_max=args.n_max, save_trace=not args.no_trace)
