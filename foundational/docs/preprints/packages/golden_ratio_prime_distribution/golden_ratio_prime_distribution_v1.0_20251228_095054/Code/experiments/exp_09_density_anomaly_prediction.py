#!/usr/bin/env python3
"""
Experiment 09: SEC as Prime Density Anomaly Detector
====================================================

Key insight from exp_08: SEC doesn't predict individual primes, but 
prime density is 3.67x higher in E>0 regions.

This suggests SEC predicts WHERE primes cluster, not WHICH numbers are prime.

New approach: Use SEC to predict density ANOMALIES from PNT.
- Compute expected π(n) from PNT
- Compute actual π(n) 
- Anomaly = actual - expected
- Does SEC predict these anomalies?

If SEC predicts anomalies, it captures real prime structure beyond PNT.

Trace output: results/exp_09_density_anomaly_YYYYMMDD_HHMMSS.json
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


def li(x):
    """Logarithmic integral Li(x) - better prime counting estimate than n/ln(n)."""
    # Use scipy's exponential integral for accuracy
    from scipy.special import expi
    if x <= 2:
        return 0
    return expi(np.log(x)) - expi(np.log(2))


def compute_density_anomaly_windows(n_max: int, window_size: int = 1000) -> dict:
    """
    Compute prime density anomalies in fixed-width windows.
    
    Anomaly = (actual prime count) - Li(hi) + Li(lo)
    """
    # Get primes
    sieve, primes = prime_sieve(n_max)
    
    # Compute SEC field
    factor_base = FIRST_50_PRIMES[:9]
    S = symbolic_entropy(n_max, factor_base)
    S_hat = entropy_expectation(S)
    I = collapse_impulse(S, S_hat)
    E = stress_field(I)
    
    # Windows
    n_windows = n_max // window_size
    
    results = []
    
    for w in range(1, n_windows):  # Skip first window (small n effects)
        lo = w * window_size
        hi = (w + 1) * window_size
        
        # Actual primes in window
        actual = np.sum(sieve[lo:hi])
        
        # Expected from Li (logarithmic integral)
        expected = li(hi) - li(lo)
        
        # Anomaly
        anomaly = actual - expected
        anomaly_pct = 100 * anomaly / expected if expected > 0 else 0
        
        # SEC features in window
        E_window = E[lo:hi]
        frac_pos = np.mean(E_window > 0)
        mean_E = np.mean(E_window)
        std_E = np.std(E_window)
        
        # Deviation from φ threshold
        phi_deviation = frac_pos - PHI_INV
        
        results.append({
            'window': w,
            'lo': lo,
            'hi': hi,
            'actual': actual,
            'expected': expected,
            'anomaly': anomaly,
            'anomaly_pct': anomaly_pct,
            'frac_E_pos': frac_pos,
            'mean_E': mean_E,
            'std_E': std_E,
            'phi_deviation': phi_deviation
        })
    
    return results


def test_anomaly_prediction(results: list) -> dict:
    """Test if SEC features predict density anomalies."""
    
    anomalies = np.array([r['anomaly'] for r in results])
    anomaly_pcts = np.array([r['anomaly_pct'] for r in results])
    frac_pos = np.array([r['frac_E_pos'] for r in results])
    mean_E = np.array([r['mean_E'] for r in results])
    phi_devs = np.array([r['phi_deviation'] for r in results])
    
    # Key correlations
    r_frac_anomaly, p_frac = stats.pearsonr(frac_pos, anomalies)
    r_mean_anomaly, p_mean = stats.pearsonr(mean_E, anomalies)
    r_phi_anomaly, p_phi = stats.pearsonr(phi_devs, anomalies)
    
    # Spearman (rank correlation - more robust)
    rho_frac, p_rho_frac = stats.spearmanr(frac_pos, anomalies)
    rho_mean, p_rho_mean = stats.spearmanr(mean_E, anomalies)
    
    # Sign consistency: when frac_pos > 1/φ, is anomaly positive?
    high_frac_mask = frac_pos > PHI_INV
    sign_match_high = np.mean(anomalies[high_frac_mask] > 0) if np.sum(high_frac_mask) > 0 else 0.5
    sign_match_low = np.mean(anomalies[~high_frac_mask] < 0) if np.sum(~high_frac_mask) > 0 else 0.5
    
    # Overall sign agreement
    predicted_sign = np.sign(phi_devs)
    actual_sign = np.sign(anomalies)
    sign_accuracy = np.mean(predicted_sign == actual_sign)
    
    # Binomial test: is sign accuracy better than chance?
    try:
        n_correct = int(np.sum(predicted_sign == actual_sign))
        n_total = len(anomalies)
        binom_p = stats.binomtest(n_correct, n_total, 0.5, alternative='greater').pvalue
    except:
        binom_p = 1.0
    
    return {
        'pearson_frac_vs_anomaly': r_frac_anomaly,
        'p_value_frac': p_frac,
        'pearson_mean_E_vs_anomaly': r_mean_anomaly,
        'p_value_mean_E': p_mean,
        'pearson_phi_dev_vs_anomaly': r_phi_anomaly,
        'p_value_phi_dev': p_phi,
        'spearman_frac': rho_frac,
        'p_spearman_frac': p_rho_frac,
        'spearman_mean': rho_mean,
        'p_spearman_mean': p_rho_mean,
        'sign_accuracy': sign_accuracy,
        'sign_accuracy_p_value': binom_p,
        'pct_positive_anomaly_when_high_frac': sign_match_high,
        'pct_negative_anomaly_when_low_frac': sign_match_low,
        'prediction_significant': p_phi < 0.05 or binom_p < 0.05
    }


def test_extreme_anomalies(results: list) -> dict:
    """Do extreme SEC deviations predict extreme anomalies?"""
    
    anomalies = np.array([r['anomaly'] for r in results])
    phi_devs = np.array([r['phi_deviation'] for r in results])
    
    # Top/bottom deciles
    anom_p90 = np.percentile(anomalies, 90)
    anom_p10 = np.percentile(anomalies, 10)
    phi_p90 = np.percentile(phi_devs, 90)
    phi_p10 = np.percentile(phi_devs, 10)
    
    # Extreme φ-deviation → extreme anomaly?
    high_phi_mask = phi_devs >= phi_p90
    low_phi_mask = phi_devs <= phi_p10
    
    mean_anomaly_high_phi = np.mean(anomalies[high_phi_mask])
    mean_anomaly_low_phi = np.mean(anomalies[low_phi_mask])
    mean_anomaly_mid = np.mean(anomalies[~high_phi_mask & ~low_phi_mask])
    
    # Effect size
    effect_high_vs_mid = (mean_anomaly_high_phi - mean_anomaly_mid) / np.std(anomalies)
    effect_low_vs_mid = (mean_anomaly_low_phi - mean_anomaly_mid) / np.std(anomalies)
    
    # T-test
    t_high, p_high = stats.ttest_ind(anomalies[high_phi_mask], anomalies[~high_phi_mask])
    t_low, p_low = stats.ttest_ind(anomalies[low_phi_mask], anomalies[~low_phi_mask])
    
    return {
        'mean_anomaly_high_phi_dev': mean_anomaly_high_phi,
        'mean_anomaly_low_phi_dev': mean_anomaly_low_phi,
        'mean_anomaly_mid': mean_anomaly_mid,
        'effect_size_high': effect_high_vs_mid,
        'effect_size_low': effect_low_vs_mid,
        't_stat_high': t_high,
        'p_value_high': p_high,
        't_stat_low': t_low,
        'p_value_low': p_low,
        'extremes_predictive': p_high < 0.05 or p_low < 0.05
    }


def run_experiment(n_max: int = 500000, window_size: int = 1000, save_trace: bool = True) -> dict:
    """Run density anomaly prediction experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 09: SEC as Prime Density Anomaly Detector")
    print("=" * 70)
    print(f"\nRange: [2, {n_max:,}]")
    print(f"Window size: {window_size:,}")
    print(f"Hypothesis: SEC predicts WHERE primes cluster, not WHICH are prime")
    
    parameters = {"n_max": n_max, "window_size": window_size}
    results = {}
    
    # Compute windows
    print("\nComputing density anomalies...")
    start = time.time()
    window_results = compute_density_anomaly_windows(n_max, window_size)
    print(f"  {len(window_results)} windows computed in {time.time() - start:.2f}s")
    
    # Summary statistics
    anomalies = [r['anomaly'] for r in window_results]
    print(f"\n  Anomaly range: [{min(anomalies):.1f}, {max(anomalies):.1f}]")
    print(f"  Anomaly std: {np.std(anomalies):.2f}")
    
    # Test 1: Anomaly prediction
    print(f"\n" + "-" * 70)
    print("TEST 1: Does SEC predict density anomalies?")
    print("-" * 70)
    
    pred_result = test_anomaly_prediction(window_results)
    results['anomaly_prediction'] = pred_result
    
    print(f"\n  Pearson(φ_deviation, anomaly): {pred_result['pearson_phi_dev_vs_anomaly']:.4f}")
    print(f"    p-value: {pred_result['p_value_phi_dev']:.4e}")
    print(f"  Spearman(frac_E>0, anomaly): {pred_result['spearman_frac']:.4f}")
    print(f"    p-value: {pred_result['p_spearman_frac']:.4e}")
    print(f"\n  Sign prediction accuracy: {pred_result['sign_accuracy']:.1%}")
    print(f"    p-value (vs random): {pred_result['sign_accuracy_p_value']:.4e}")
    print(f"\n  When frac(E>0) > 1/φ: {pred_result['pct_positive_anomaly_when_high_frac']:.1%} positive anomaly")
    print(f"  When frac(E>0) < 1/φ: {pred_result['pct_negative_anomaly_when_low_frac']:.1%} negative anomaly")
    print(f"\n  Prediction significant: {'YES ✅' if pred_result['prediction_significant'] else 'NO ❌'}")
    
    # Test 2: Extreme anomalies
    print(f"\n" + "-" * 70)
    print("TEST 2: Do extreme SEC deviations predict extreme anomalies?")
    print("-" * 70)
    
    extreme_result = test_extreme_anomalies(window_results)
    results['extreme_anomalies'] = extreme_result
    
    print(f"\n  Mean anomaly when φ-dev in top 10%: {extreme_result['mean_anomaly_high_phi_dev']:+.2f}")
    print(f"  Mean anomaly when φ-dev in bot 10%: {extreme_result['mean_anomaly_low_phi_dev']:+.2f}")
    print(f"  Mean anomaly otherwise:             {extreme_result['mean_anomaly_mid']:+.2f}")
    print(f"\n  Effect size (high φ-dev): {extreme_result['effect_size_high']:.3f} std")
    print(f"  Effect size (low φ-dev):  {extreme_result['effect_size_low']:.3f} std")
    print(f"\n  High φ-dev t-test p-value: {extreme_result['p_value_high']:.4e}")
    print(f"  Low φ-dev t-test p-value:  {extreme_result['p_value_low']:.4e}")
    print(f"  Extremes predictive: {'YES ✅' if extreme_result['extremes_predictive'] else 'NO ❌'}")
    
    # Validation summary
    validation = {
        'anomaly_correlation_significant': pred_result['prediction_significant'],
        'sign_prediction_above_chance': pred_result['sign_accuracy_p_value'] < 0.05,
        'extremes_predictive': extreme_result['extremes_predictive'],
        'sec_predicts_density_anomalies': (
            pred_result['prediction_significant'] or 
            extreme_result['extremes_predictive']
        )
    }
    
    print(f"\n" + "=" * 70)
    print("DENSITY ANOMALY PREDICTION SUMMARY")
    print("=" * 70)
    for check, passed in validation.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    if validation['sec_predicts_density_anomalies']:
        print(f"\n  🎯 SEC predicts WHERE primes cluster!")
        print(f"     This is density anomaly detection, not individual prime prediction.")
    else:
        print(f"\n  ⚠️  No significant density anomaly prediction.")
    
    # Save trace
    if save_trace:
        trace = create_trace(
            experiment_id="exp_09_density_anomaly",
            parameters=parameters,
            results=results,
            validation=validation
        )
        
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / f"exp_09_density_anomaly_{trace.timestamp}.json"
        trace.save(str(filepath))
        print(f"\nTrace saved: {filepath.name}")
    
    return {'parameters': parameters, 'results': results, 'validation': validation}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_max", type=int, default=500000)
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    
    run_experiment(n_max=args.n_max, window_size=args.window_size, save_trace=not args.no_trace)
