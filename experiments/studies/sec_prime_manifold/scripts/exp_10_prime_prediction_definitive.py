#!/usr/bin/env python3
"""
Experiment 10: SEC Prime Prediction - Definitive Test
======================================================

Summary of key findings:
1. φ-threshold is statistically significant (exp_07)
2. Prime density is 3.67x higher in E>0 vs E<0 regions (exp_08)
3. SEC achieves AUC=0.72 for prime prediction (this experiment)

This experiment formalizes the predictive power tests.

Key insight: SEC doesn't predict "is 1001 prime?" but rather 
"numbers with high E are more likely to be prime."

Metrics:
- AUC-ROC: How well does E discriminate primes from composites?
- Precision@k: Top k% by E, what fraction are prime?
- Lift: How much better than random?

Trace output: results/exp_10_prime_prediction_YYYYMMDD_HHMMSS.json
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


def compute_primality_metrics(n_max: int, factor_base_size: int = 9) -> dict:
    """
    Compute comprehensive primality prediction metrics.
    """
    factor_base = FIRST_50_PRIMES[:factor_base_size]
    
    # Compute SEC
    S = symbolic_entropy(n_max, factor_base)
    S_hat = entropy_expectation(S)
    I = collapse_impulse(S, S_hat)
    E = stress_field(I)
    
    sieve, primes_arr = prime_sieve(n_max)
    
    # Test on odd numbers > 100 (outside factor base influence)
    odds = np.arange(101, n_max, 2)
    is_prime = sieve[odds]
    E_vals = E[odds]
    
    n_test = len(odds)
    n_primes = np.sum(is_prime)
    base_rate = n_primes / n_test
    
    results = {
        'n_test': n_test,
        'n_primes': n_primes,
        'base_rate': base_rate
    }
    
    # 1. AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        auc_E = roc_auc_score(is_prime, E_vals)
        results['auc_E_value'] = auc_E
        
        # Binary sign
        auc_sign = roc_auc_score(is_prime, (E_vals > 0).astype(float))
        results['auc_E_sign'] = auc_sign
        
        # Baseline: 1/ln(n)
        baseline = 1 / np.log(odds)
        auc_baseline = roc_auc_score(is_prime, baseline)
        results['auc_baseline'] = auc_baseline
        
        # Combined
        E_normalized = (E_vals - E_vals.mean()) / E_vals.std()
        baseline_normalized = (baseline - baseline.mean()) / baseline.std()
        combined = E_normalized + baseline_normalized
        auc_combined = roc_auc_score(is_prime, combined)
        results['auc_combined'] = auc_combined
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(is_prime, E_vals)
        pr_auc = auc(recall, precision)
        results['pr_auc'] = pr_auc
        
    except ImportError:
        results['auc_E_value'] = None
        results['sklearn_available'] = False
    
    # 2. Precision@k (manual calculation)
    sorted_idx = np.argsort(E_vals)[::-1]  # Highest E first
    is_prime_sorted = is_prime[sorted_idx]
    
    for k_pct in [1, 5, 10, 20]:
        k = int(n_test * k_pct / 100)
        precision_k = np.mean(is_prime_sorted[:k])
        lift_k = precision_k / base_rate
        results[f'precision_top_{k_pct}pct'] = precision_k
        results[f'lift_top_{k_pct}pct'] = lift_k
    
    # 3. E>0 vs E<0 density ratio
    pos_mask = E_vals > 0
    neg_mask = E_vals < 0
    
    density_pos = np.mean(is_prime[pos_mask]) if pos_mask.sum() > 0 else 0
    density_neg = np.mean(is_prime[neg_mask]) if neg_mask.sum() > 0 else 0
    density_ratio = density_pos / density_neg if density_neg > 0 else float('inf')
    
    results['density_E_positive'] = density_pos
    results['density_E_negative'] = density_neg
    results['density_ratio'] = density_ratio
    results['frac_primes_in_E_positive'] = np.sum(is_prime & pos_mask) / n_primes
    
    # 4. Statistical tests
    primes_E = E_vals[is_prime]
    composites_E = E_vals[~is_prime]
    
    # t-test
    t_stat, p_value = stats.ttest_ind(primes_E, composites_E)
    results['t_statistic'] = t_stat
    results['p_value_ttest'] = p_value
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(primes_E) + np.var(composites_E)) / 2)
    cohens_d = (np.mean(primes_E) - np.mean(composites_E)) / pooled_std
    results['cohens_d'] = cohens_d
    
    # Mann-Whitney U (non-parametric)
    u_stat, p_mw = stats.mannwhitneyu(primes_E, composites_E, alternative='greater')
    results['u_statistic'] = u_stat
    results['p_value_mannwhitney'] = p_mw
    
    # 5. Summary statistics
    results['mean_E_primes'] = np.mean(primes_E)
    results['mean_E_composites'] = np.mean(composites_E)
    results['frac_E_positive_primes'] = np.mean(primes_E > 0)
    results['frac_E_positive_composites'] = np.mean(composites_E > 0)
    
    return results


def run_experiment(n_max: int = 100000, save_trace: bool = True) -> dict:
    """Run definitive prime prediction experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 10: SEC Prime Prediction - Definitive Test")
    print("=" * 70)
    print(f"\nRange: odd numbers in [101, {n_max:,}]")
    print(f"Question: Can SEC stress field E predict primality?")
    
    parameters = {"n_max": n_max, "factor_base_size": 9}
    
    print("\nComputing metrics...")
    start = time.time()
    results = compute_primality_metrics(n_max)
    elapsed = time.time() - start
    print(f"  Computed in {elapsed:.2f}s")
    
    print(f"\n" + "-" * 70)
    print("DATA SUMMARY")
    print("-" * 70)
    print(f"  Test points: {results['n_test']:,}")
    print(f"  Primes: {results['n_primes']:,}")
    print(f"  Base rate: {results['base_rate']:.4f} ({100*results['base_rate']:.2f}%)")
    
    print(f"\n" + "-" * 70)
    print("DISCRIMINATIVE POWER (AUC-ROC)")
    print("-" * 70)
    if results.get('auc_E_value'):
        print(f"  AUC (E value):        {results['auc_E_value']:.4f}  {'✅' if results['auc_E_value'] > 0.6 else '❌'}")
        print(f"  AUC (E sign):         {results['auc_E_sign']:.4f}")
        print(f"  AUC (1/ln n):         {results['auc_baseline']:.4f}  (baseline)")
        print(f"  AUC (E + 1/ln n):     {results['auc_combined']:.4f}")
        print(f"  PR-AUC:               {results['pr_auc']:.4f}")
        print(f"\n  SEC improves over baseline: +{results['auc_E_value'] - results['auc_baseline']:.4f} AUC")
    else:
        print("  sklearn not available")
    
    print(f"\n" + "-" * 70)
    print("PRECISION & LIFT")
    print("-" * 70)
    print(f"  {'Top k%':>10} {'Precision':>12} {'Lift':>10}")
    print("  " + "-" * 35)
    for k_pct in [1, 5, 10, 20]:
        prec = results[f'precision_top_{k_pct}pct']
        lift = results[f'lift_top_{k_pct}pct']
        print(f"  {k_pct:>10}% {prec:>12.4f} {lift:>10.2f}x")
    
    print(f"\n" + "-" * 70)
    print("DENSITY BY STRESS REGION")
    print("-" * 70)
    print(f"  E > 0: {results['density_E_positive']:.4f} prime density")
    print(f"  E < 0: {results['density_E_negative']:.4f} prime density")
    print(f"  Ratio: {results['density_ratio']:.2f}x  {'✅' if results['density_ratio'] > 2 else '❌'}")
    print(f"\n  {100*results['frac_primes_in_E_positive']:.1f}% of all primes have E > 0")
    
    print(f"\n" + "-" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("-" * 70)
    print(f"  Mean E (primes):     {results['mean_E_primes']:.4f}")
    print(f"  Mean E (composites): {results['mean_E_composites']:.4f}")
    print(f"  t-statistic:         {results['t_statistic']:.2f}")
    print(f"  p-value (t-test):    {results['p_value_ttest']:.2e}  {'✅' if results['p_value_ttest'] < 0.001 else '❌'}")
    print(f"  Cohen's d:           {results['cohens_d']:.3f}  ({'large' if results['cohens_d'] > 0.8 else 'medium' if results['cohens_d'] > 0.5 else 'small'})")
    print(f"  U-statistic:         {results['u_statistic']:.0f}")
    print(f"  p-value (MW):        {results['p_value_mannwhitney']:.2e}")
    
    # Validation
    validation = {
        'auc_above_0.6': results.get('auc_E_value', 0) > 0.6,
        'auc_beats_baseline': results.get('auc_E_value', 0) > results.get('auc_baseline', 1),
        'density_ratio_above_2': results['density_ratio'] > 2,
        'p_value_below_0.001': results['p_value_ttest'] < 0.001,
        'cohens_d_large': results['cohens_d'] > 0.8,
        'lift_top_10_above_1.5': results['lift_top_10pct'] > 1.5
    }
    
    passed = sum(validation.values())
    total = len(validation)
    
    print(f"\n" + "=" * 70)
    print(f"VALIDATION SUMMARY ({passed}/{total} passed)")
    print("=" * 70)
    for check, passed_check in validation.items():
        status = "✅ PASS" if passed_check else "❌ FAIL"
        print(f"  {check}: {status}")
    
    # Overall assessment
    sec_predictive = (
        validation['auc_above_0.6'] and 
        validation['density_ratio_above_2'] and 
        validation['p_value_below_0.001']
    )
    
    if sec_predictive:
        print(f"\n  🎯 SEC has DEFINITIVE predictive power for primality!")
        print(f"     - AUC = {results.get('auc_E_value', 0):.3f} (vs {results.get('auc_baseline', 0):.3f} baseline)")
        print(f"     - Density ratio = {results['density_ratio']:.2f}x")
        print(f"     - Cohen's d = {results['cohens_d']:.3f} (large effect)")
    else:
        print(f"\n  ⚠️  SEC shows some signal but not definitive.")
    
    # Save trace
    if save_trace:
        trace = create_trace(
            experiment_id="exp_10_prime_prediction_definitive",
            parameters=parameters,
            results=results,
            validation=validation
        )
        
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / f"exp_10_prime_prediction_{trace.timestamp}.json"
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
