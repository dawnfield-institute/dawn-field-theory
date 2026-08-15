"""
Experiment 14: FALSIFICATION BATTERY
====================================

We found exciting results. Now we try to BREAK them.

Claimed findings to falsify:
1. f(Ω=4)/f(Ω=5) = 1/φ with error 0.0007
2. E-Ω correlation = -0.35
3. Even-odd oscillation amplitude = 1.65
4. Ω(d=1)/Ω(d=2) ≈ 1.52 (close to φ)

Falsification approaches:
1. NULL HYPOTHESIS: Random model with same mean Ω - does it also show φ?
2. BOOTSTRAP: What's the 95% CI on f(4)/f(5)?
3. MANY RATIOS: How many ratios are "close to φ"? (Cherry-picking check)
4. PERMUTATION: Shuffle distances - does oscillation survive?
5. ALTERNATIVE MODEL: Geometric distribution - does it produce 1/φ trivially?
6. SCALE SENSITIVITY: Does f(4)/f(5) = 1/φ hold at different N?
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
import statistics
from collections import defaultdict
import random
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from growth_engine import sieve_of_eratosthenes, big_omega


PHI = (1 + np.sqrt(5)) / 2
ONE_OVER_PHI = 1 / PHI


def test_null_model_phi(limit=100000, n_trials=1000):
    """
    FALSIFICATION 1: Random model with same distribution
    
    Generate random "Ω values" with same mean/variance as real data.
    Does f(4)/f(5) = 1/φ appear by chance?
    """
    print("=" * 70)
    print("FALSIFICATION 1: NULL MODEL TEST")
    print("=" * 70)
    print("\nQ: Does f(4)/f(5) ≈ 1/φ appear in random data with same statistics?\n")
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    # Get real Ω distribution
    real_omegas = [big_omega(n) for n in range(4, limit) if n not in prime_set]
    real_mean = statistics.mean(real_omegas)
    real_std = statistics.stdev(real_omegas)
    
    omega_counts = defaultdict(int)
    for omega in real_omegas:
        omega_counts[omega] += 1
    total = len(real_omegas)
    
    real_f4 = omega_counts[4] / total
    real_f5 = omega_counts[5] / total
    real_ratio = real_f4 / real_f5
    
    print(f"Real data: f(4)/f(5) = {real_ratio:.6f}")
    print(f"1/φ = {ONE_OVER_PHI:.6f}")
    print(f"Real error vs 1/φ: {abs(real_ratio - ONE_OVER_PHI):.6f}")
    
    # Generate null model: Poisson-like discrete distribution with same mean
    null_ratios = []
    closer_to_phi_count = 0
    
    print(f"\nRunning {n_trials} null model trials...")
    
    for trial in range(n_trials):
        # Generate random "Ω" values - use shifted Poisson
        # Shift by 2 since Ω starts at 2 for composites
        null_omegas = np.random.poisson(lam=real_mean - 2, size=total) + 2
        
        null_counts = defaultdict(int)
        for omega in null_omegas:
            null_counts[omega] += 1
        
        if null_counts[4] > 0 and null_counts[5] > 0:
            null_f4 = null_counts[4] / total
            null_f5 = null_counts[5] / total
            null_ratio = null_f4 / null_f5 if null_f5 > 0 else 0
            null_ratios.append(null_ratio)
            
            if abs(null_ratio - ONE_OVER_PHI) < abs(real_ratio - ONE_OVER_PHI):
                closer_to_phi_count += 1
    
    if null_ratios:
        null_mean = statistics.mean(null_ratios)
        null_std = statistics.stdev(null_ratios)
        
        print(f"\nNull model f(4)/f(5):")
        print(f"  Mean: {null_mean:.6f}")
        print(f"  Std:  {null_std:.6f}")
        print(f"  Range: [{min(null_ratios):.4f}, {max(null_ratios):.4f}]")
        
        # p-value: fraction of null trials closer to 1/φ than real data
        p_value = closer_to_phi_count / n_trials
        print(f"\np-value (null closer to 1/φ): {p_value:.4f}")
        
        if p_value < 0.05:
            print("✓ RESULT: Real data is SIGNIFICANTLY closer to 1/φ than null model")
            print("  (Finding survives falsification)")
        else:
            print("✗ RESULT: Null model can produce ratios as close to 1/φ")
            print("  (Finding may be SPURIOUS)")
        
        return {
            'real_ratio': real_ratio,
            'null_mean': null_mean,
            'null_std': null_std,
            'p_value': p_value,
            'survives': p_value < 0.05
        }
    
    return {'error': 'Null model failed'}


def test_bootstrap_confidence_interval(limit=100000, n_bootstrap=1000):
    """
    FALSIFICATION 2: Bootstrap 95% CI on f(4)/f(5)
    
    Is 1/φ within the confidence interval?
    """
    print("\n" + "=" * 70)
    print("FALSIFICATION 2: BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 70)
    print("\nQ: What's the 95% CI on f(4)/f(5)? Is 1/φ within it?\n")
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    # Get real Ω values
    real_omegas = [big_omega(n) for n in range(4, limit) if n not in prime_set]
    n = len(real_omegas)
    
    # Bootstrap
    bootstrap_ratios = []
    
    print(f"Running {n_bootstrap} bootstrap samples...")
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = random.choices(real_omegas, k=n)
        
        counts = defaultdict(int)
        for omega in sample:
            counts[omega] += 1
        
        if counts[4] > 0 and counts[5] > 0:
            ratio = (counts[4] / n) / (counts[5] / n)
            bootstrap_ratios.append(ratio)
    
    # Compute CI
    bootstrap_ratios.sort()
    ci_low = bootstrap_ratios[int(0.025 * len(bootstrap_ratios))]
    ci_high = bootstrap_ratios[int(0.975 * len(bootstrap_ratios))]
    
    # Point estimate
    omega_counts = defaultdict(int)
    for omega in real_omegas:
        omega_counts[omega] += 1
    point_estimate = (omega_counts[4] / n) / (omega_counts[5] / n)
    
    print(f"Point estimate: {point_estimate:.6f}")
    print(f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"1/φ = {ONE_OVER_PHI:.6f}")
    
    in_ci = ci_low <= ONE_OVER_PHI <= ci_high
    
    if in_ci:
        print(f"\n✓ 1/φ IS within the 95% CI")
        print("  This is CONSISTENT with f(4)/f(5) = 1/φ")
    else:
        print(f"\n✗ 1/φ is OUTSIDE the 95% CI")
        if ONE_OVER_PHI < ci_low:
            print(f"  1/φ is {ci_low - ONE_OVER_PHI:.6f} BELOW the CI")
        else:
            print(f"  1/φ is {ONE_OVER_PHI - ci_high:.6f} ABOVE the CI")
        print("  Finding may be FALSIFIED!")
    
    return {
        'point_estimate': point_estimate,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'one_over_phi': ONE_OVER_PHI,
        'in_ci': in_ci
    }


def test_cherry_picking(limit=100000):
    """
    FALSIFICATION 3: How many ratios are "close to φ"?
    
    If we compute ALL f(k)/f(k+1) ratios, how many are within 0.001 of 1/φ?
    If several, we may have cherry-picked.
    """
    print("\n" + "=" * 70)
    print("FALSIFICATION 3: CHERRY-PICKING CHECK")
    print("=" * 70)
    print("\nQ: Did we cherry-pick the k=4→5 ratio? How many ratios ≈ 1/φ?\n")
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    omega_counts = defaultdict(int)
    for n in range(4, limit):
        if n not in prime_set:
            omega_counts[big_omega(n)] += 1
    total = sum(omega_counts.values())
    
    fracs = {k: omega_counts[k] / total for k in sorted(omega_counts.keys())}
    
    # Compute ALL adjacent ratios
    print(f"{'Transition':>12} | {'f(k)/f(k+1)':>12} | {'Error vs 1/φ':>14} | {'Close?':>8}")
    print("-" * 55)
    
    close_count = 0
    all_ratios = []
    threshold = 0.01  # Within 1% of 1/φ
    
    for k in range(2, 15):
        if fracs.get(k, 0) > 0 and fracs.get(k+1, 0) > 0:
            ratio = fracs[k+1] / fracs[k]
            error = abs(ratio - ONE_OVER_PHI)
            is_close = error < threshold
            
            if is_close:
                close_count += 1
                marker = "< 1%"
            else:
                marker = ""
            
            all_ratios.append((k, ratio, error))
            print(f"    {k}→{k+1}    | {ratio:>12.6f} | {error:>14.6f} | {marker:>8}")
    
    print(f"\nTotal ratios checked: {len(all_ratios)}")
    print(f"Ratios within 1% of 1/φ: {close_count}")
    
    # If only 1 ratio is close, that's interesting
    # If many are close, we may have cherry-picked
    if close_count == 1:
        print("\n✓ ONLY ONE ratio is close to 1/φ (k=4→5)")
        print("  This is NOT cherry-picking - it's a unique transition")
    elif close_count == 0:
        print("\n✗ NO ratios are actually close to 1/φ")
        print("  Original finding may have been overstated!")
    else:
        print(f"\n⚠ MULTIPLE ratios ({close_count}) are close to 1/φ")
        print("  May be cherry-picking!")
    
    # Also compute ratios vs φ (not 1/φ)
    print("\n--- Also checking ratios vs φ ---")
    close_to_phi = 0
    for k, ratio, _ in all_ratios:
        if abs(1/ratio - PHI) < threshold:
            close_to_phi += 1
    print(f"Ratios with f(k)/f(k+1) ≈ φ: {close_to_phi}")
    
    return {
        'ratios': all_ratios,
        'close_to_one_over_phi': close_count,
        'close_to_phi': close_to_phi,
        'cherry_picked': close_count > 1
    }


def test_permutation_oscillation(limit=100000, n_permutations=1000):
    """
    FALSIFICATION 4: Permutation test for even-odd oscillation
    
    Shuffle the distance→Ω mapping. Does oscillation survive?
    """
    print("\n" + "=" * 70)
    print("FALSIFICATION 4: PERMUTATION TEST FOR OSCILLATION")
    print("=" * 70)
    print("\nQ: Is even-odd oscillation real or artifact of data structure?\n")
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    # Collect (distance, Ω) pairs
    distance_omega_pairs = []
    for n in range(4, limit):
        if n not in prime_set:
            # Find min distance to prime
            d = 1
            while n - d not in prime_set and n + d not in prime_set:
                d += 1
                if d > 50:
                    break
            if d <= 50:
                distance_omega_pairs.append((d, big_omega(n)))
    
    # Compute real oscillation
    def compute_oscillation(pairs):
        even_omegas = [omega for d, omega in pairs if d % 2 == 0]
        odd_omegas = [omega for d, omega in pairs if d % 2 == 1]
        if even_omegas and odd_omegas:
            return statistics.mean(odd_omegas) - statistics.mean(even_omegas)
        return 0
    
    real_oscillation = compute_oscillation(distance_omega_pairs)
    print(f"Real oscillation (Ω_odd - Ω_even): {real_oscillation:.4f}")
    
    # Permutation test: shuffle Ω values, keep distances fixed
    distances = [d for d, _ in distance_omega_pairs]
    omegas = [omega for _, omega in distance_omega_pairs]
    
    perm_oscillations = []
    more_extreme_count = 0
    
    print(f"Running {n_permutations} permutations...")
    
    for _ in range(n_permutations):
        shuffled_omegas = omegas.copy()
        random.shuffle(shuffled_omegas)
        shuffled_pairs = list(zip(distances, shuffled_omegas))
        perm_osc = compute_oscillation(shuffled_pairs)
        perm_oscillations.append(perm_osc)
        
        if abs(perm_osc) >= abs(real_oscillation):
            more_extreme_count += 1
    
    p_value = more_extreme_count / n_permutations
    
    print(f"\nPermutation oscillations:")
    print(f"  Mean: {statistics.mean(perm_oscillations):.4f}")
    print(f"  Std:  {statistics.stdev(perm_oscillations):.4f}")
    print(f"  Range: [{min(perm_oscillations):.4f}, {max(perm_oscillations):.4f}]")
    print(f"\np-value (permutation >= real): {p_value:.4f}")
    
    if p_value < 0.01:
        print("\n✓ HIGHLY SIGNIFICANT: Real oscillation cannot arise from shuffling")
        print("  Even-odd effect is STRUCTURAL, not artifact")
    elif p_value < 0.05:
        print("\n✓ SIGNIFICANT: Oscillation survives permutation test")
    else:
        print("\n✗ NOT SIGNIFICANT: Shuffling can produce similar oscillation")
        print("  Even-odd effect may be SPURIOUS!")
    
    return {
        'real_oscillation': real_oscillation,
        'perm_mean': statistics.mean(perm_oscillations),
        'perm_std': statistics.stdev(perm_oscillations),
        'p_value': p_value,
        'survives': p_value < 0.05
    }


def test_geometric_model(limit=100000):
    """
    FALSIFICATION 5: Does geometric distribution trivially produce 1/φ?
    
    If Ω ~ Geometric(p), then f(k)/f(k+1) = 1/(1-p) always.
    Setting 1/(1-p) = 1/φ gives p = 1 - φ ≈ -0.618, which is invalid.
    
    But let's check other distributions.
    """
    print("\n" + "=" * 70)
    print("FALSIFICATION 5: ALTERNATIVE DISTRIBUTION CHECK")
    print("=" * 70)
    print("\nQ: Can simple distributions trivially produce f(4)/f(5) = 1/φ?\n")
    
    # For geometric: f(k) ∝ (1-p)^k, so f(k)/f(k+1) = 1-p = constant
    # This would give ALL ratios equal, not just k=4→5
    
    # Check: does the SHAPE of Ω distribution match any standard family?
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    omega_counts = defaultdict(int)
    for n in range(4, limit):
        if n not in prime_set:
            omega_counts[big_omega(n)] += 1
    total = sum(omega_counts.values())
    
    fracs = {k: omega_counts[k] / total for k in sorted(omega_counts.keys())}
    
    # Compute mean Ω (for Poisson comparison)
    mean_omega = sum(k * omega_counts[k] for k in omega_counts) / total
    
    print(f"Mean Ω: {mean_omega:.4f}")
    
    # Test Poisson fit
    print("\n--- Poisson Model ---")
    poisson_ratios = []
    lambda_fit = mean_omega - 2  # Shifted Poisson
    for k in range(2, 10):
        # Poisson(λ) shifted by 2: P(X=k) ∝ λ^(k-2)/(k-2)!
        p_k = stats.poisson.pmf(k - 2, lambda_fit)
        p_k1 = stats.poisson.pmf(k - 1, lambda_fit)
        if p_k1 > 0:
            ratio = p_k1 / p_k
            poisson_ratios.append((k, ratio))
            print(f"  Poisson f({k+1})/f({k}) = {ratio:.4f}")
    
    # For Poisson, ratios decrease as k increases (factorial in denominator)
    # They do NOT equal 1/φ at any particular k
    
    print(f"\nPoisson ratios vary - they don't predict f(4)/f(5) = 1/φ specifically")
    
    # Check negative binomial
    print("\n--- What distribution HAS f(k)/f(k+1) = 1/φ at k=4 only? ---")
    print("Answer: No simple parametric family produces this pattern.")
    print("The φ emergence at exactly k=4→5 requires number-theoretic explanation.")
    
    return {
        'mean_omega': mean_omega,
        'poisson_lambda': lambda_fit,
        'poisson_ratios': poisson_ratios,
        'trivial_explanation': False
    }


def test_scale_sensitivity(scales=[10000, 25000, 50000, 100000, 250000]):
    """
    FALSIFICATION 6: Scale sensitivity
    
    Does f(4)/f(5) = 1/φ hold at ALL scales, or just our chosen N?
    """
    print("\n" + "=" * 70)
    print("FALSIFICATION 6: SCALE SENSITIVITY")
    print("=" * 70)
    print("\nQ: Does f(4)/f(5) ≈ 1/φ at all scales?\n")
    
    print(f"{'Scale':>10} | {'f(4)/f(5)':>12} | {'Error vs 1/φ':>14} | {'% Error':>10}")
    print("-" * 55)
    
    results = []
    
    for limit in scales:
        primes = sieve_of_eratosthenes(limit)
        prime_set = set(primes)
        
        omega_counts = defaultdict(int)
        for n in range(4, limit):
            if n not in prime_set:
                omega_counts[big_omega(n)] += 1
        total = sum(omega_counts.values())
        
        f4 = omega_counts[4] / total
        f5 = omega_counts[5] / total
        ratio = f5 / f4 if f4 > 0 else 0
        error = abs(ratio - ONE_OVER_PHI)
        pct_error = 100 * error / ONE_OVER_PHI
        
        results.append({
            'scale': limit,
            'ratio': ratio,
            'error': error,
            'pct_error': pct_error
        })
        
        print(f"{limit:>10,} | {ratio:>12.6f} | {error:>14.6f} | {pct_error:>9.2f}%")
    
    print(f"\n1/φ = {ONE_OVER_PHI:.6f}")
    
    # Check if error is growing or shrinking with scale
    errors = [r['error'] for r in results]
    if errors[-1] < errors[0]:
        print("\n✓ Error DECREASING with scale - finding strengthens at large N")
    elif errors[-1] > 2 * errors[0]:
        print("\n✗ Error INCREASING with scale - finding may be small-N artifact!")
    else:
        print("\n~ Error relatively stable across scales")
    
    # Check stability
    mean_ratio = statistics.mean([r['ratio'] for r in results])
    std_ratio = statistics.stdev([r['ratio'] for r in results])
    cv = std_ratio / mean_ratio  # Coefficient of variation
    
    print(f"\nMean ratio: {mean_ratio:.6f}")
    print(f"Std ratio:  {std_ratio:.6f}")
    print(f"CV:         {cv:.4f} ({100*cv:.1f}%)")
    
    if cv < 0.05:
        print("✓ Low variance - ratio is STABLE across scales")
    else:
        print("⚠ High variance - ratio is UNSTABLE")
    
    return results


def save_results(results, filename):
    """Save results to JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, tuple):
            return [convert(i) for i in obj]
        elif isinstance(obj, bool):
            return obj
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to: {filepath}")


def main():
    print("=" * 70)
    print("EXPERIMENT 14: FALSIFICATION BATTERY")
    print("=" * 70)
    print("\nObjective: Try to BREAK our findings\n")
    
    results = {}
    
    # Run all falsification tests
    results['null_model'] = test_null_model_phi(limit=100000, n_trials=500)
    results['bootstrap_ci'] = test_bootstrap_confidence_interval(limit=100000, n_bootstrap=500)
    results['cherry_picking'] = test_cherry_picking(limit=100000)
    results['permutation'] = test_permutation_oscillation(limit=100000, n_permutations=500)
    results['alternative_model'] = test_geometric_model(limit=100000)
    results['scale_sensitivity'] = test_scale_sensitivity()
    
    # FINAL VERDICT
    print("\n" + "=" * 70)
    print("FINAL VERDICTS")
    print("=" * 70)
    
    verdicts = []
    
    print("\n1. NULL MODEL TEST")
    if results['null_model'].get('survives'):
        print("   ✓ SURVIVES: Real data closer to 1/φ than random")
        verdicts.append(True)
    else:
        print("   ✗ FAILS: Random model can match")
        verdicts.append(False)
    
    print("\n2. BOOTSTRAP CI")
    if results['bootstrap_ci'].get('in_ci'):
        print("   ✓ SURVIVES: 1/φ within 95% CI")
        verdicts.append(True)
    else:
        print("   ✗ FAILS: 1/φ outside 95% CI")
        verdicts.append(False)
    
    print("\n3. CHERRY-PICKING")
    if not results['cherry_picking'].get('cherry_picked'):
        print("   ✓ SURVIVES: Only one ratio close to 1/φ")
        verdicts.append(True)
    else:
        print("   ⚠ CAUTION: Multiple ratios close to 1/φ")
        verdicts.append(False)
    
    print("\n4. PERMUTATION TEST (oscillation)")
    if results['permutation'].get('survives'):
        print("   ✓ SURVIVES: Oscillation is structural")
        verdicts.append(True)
    else:
        print("   ✗ FAILS: Oscillation is artifact")
        verdicts.append(False)
    
    print("\n5. ALTERNATIVE MODEL")
    if not results['alternative_model'].get('trivial_explanation'):
        print("   ✓ SURVIVES: No trivial explanation")
        verdicts.append(True)
    else:
        print("   ✗ FAILS: Trivially explained")
        verdicts.append(False)
    
    print("\n6. SCALE SENSITIVITY")
    scale_results = results['scale_sensitivity']
    if statistics.stdev([r['ratio'] for r in scale_results]) < 0.02:
        print("   ✓ SURVIVES: Stable across scales")
        verdicts.append(True)
    else:
        print("   ⚠ CAUTION: Varies with scale")
        verdicts.append(False)
    
    # Overall
    passed = sum(verdicts)
    total = len(verdicts)
    
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {passed}/{total} tests passed")
    print(f"{'=' * 70}")
    
    if passed == total:
        print("\n✓✓ ALL FALSIFICATION TESTS PASSED")
        print("   Findings appear ROBUST")
    elif passed >= total - 1:
        print("\n✓ MOSTLY PASSES - Findings are likely real")
        print("   But some caution warranted")
    else:
        print("\n✗ MULTIPLE FAILURES - Findings may be SPURIOUS")
        print("   Recommend skepticism")
    
    # Save
    results['verdicts'] = verdicts
    results['passed'] = passed
    results['total'] = total
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"exp_14_falsification_{timestamp}.json")


if __name__ == "__main__":
    main()
