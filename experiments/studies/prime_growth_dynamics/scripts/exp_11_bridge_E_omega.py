"""
Experiment 11: Bridge between SEC Stress Field E(n) and Crystallization Depth Ω(n)
==================================================================================

CRITICAL HYPOTHESIS:
- SEC computes E(n) = cumulative stress field from entropy dynamics
- We compute Ω(n) = total prime factors (crystallization depth)
- oscillation_attractor_dynamics found: E(composite) < 0 (discharged/crystallized)
- We found: High potential → LOW Ω (the "inversion")

BRIDGE CONJECTURE:
The inversion is not a problem - it's the SAME phenomenon from different angles:
- E < 0 means "crystallization has occurred" (entropy discharged)
- High Ω means "deep crystallization" (many factors accumulated)
- Therefore: E(n) < 0 should correlate with HIGH Ω(n)

This experiment tests whether E and Ω are measuring the same underlying structure.

Also tests the 55% = F₁₀/100 hypothesis:
- 2 seeds 55.7% of composites
- F₁₀ = 55
- Is 55% related to Fibonacci?
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
import statistics
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'oscillation_attractor_dynamics', 'core'))

from growth_engine import sieve_of_eratosthenes, big_omega


def get_primes(limit):
    """Helper to get list of primes up to limit."""
    return sieve_of_eratosthenes(limit)


def compute_sec_simple(n_max, factor_base=None, window=101, lam=0.99):
    """
    Simplified SEC computation (standalone, no imports needed).
    
    S(n) = fraction of factor_base primes dividing n
    S_hat(n) = local moving average
    I(n) = S_hat(n) - S(n)
    E(n) = λ*E(n-1) + I(n)
    """
    if factor_base is None:
        factor_base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    # Symbolic entropy
    S = np.zeros(n_max + 1, dtype=float)
    k = len(factor_base)
    for n in range(2, n_max + 1):
        count = sum(1 for p in factor_base if n % p == 0)
        S[n] = count / k
    
    # Moving average expectation
    S_hat = np.zeros_like(S)
    half = window // 2
    for n in range(2, n_max + 1):
        lo = max(2, n - half)
        hi = min(n_max, n + half)
        S_hat[n] = S[lo:hi+1].mean()
    
    # Impulse and stress
    I = S_hat - S
    E = np.zeros_like(I)
    for n in range(2, len(I)):
        E[n] = lam * E[n-1] + I[n]
    
    return S, I, E


def test_E_omega_correlation(limit=50000):
    """
    Test 1: Does E(n) correlate with Ω(n)?
    
    Hypothesis: E < 0 (crystallized) should correlate with HIGH Ω (deep crystallization)
    """
    print("=== TEST 1: E(n) vs Ω(n) CORRELATION ===\n")
    
    primes = get_primes(limit)
    prime_set = set(primes)
    
    # Compute SEC
    print("Computing SEC stress field E(n)...")
    S, I, E = compute_sec_simple(limit)
    
    # Compute Ω for composites only
    print("Computing Ω(n) for composites...")
    E_values = []
    omega_values = []
    I_values = []
    
    for n in range(4, limit):
        if n not in prime_set:
            E_values.append(E[n])
            omega_values.append(big_omega(n))
            I_values.append(I[n])
    
    E_arr = np.array(E_values)
    omega_arr = np.array(omega_values)
    I_arr = np.array(I_values)
    
    # Correlations
    r_pearson, p_pearson = pearsonr(E_arr, omega_arr)
    r_spearman, p_spearman = spearmanr(E_arr, omega_arr)
    
    print(f"Composites analyzed: {len(E_arr):,}")
    print(f"\nCorrelation(E, Ω):")
    print(f"  Pearson:  r = {r_pearson:.4f}  (p = {p_pearson:.2e})")
    print(f"  Spearman: r = {r_spearman:.4f} (p = {p_spearman:.2e})")
    
    # Check direction
    if r_pearson < 0:
        print("\n  ✓ NEGATIVE correlation: E < 0 → higher Ω")
        print("    Interpretation: Discharged stress → deep crystallization")
    else:
        print("\n  ⚠ POSITIVE correlation: E > 0 → higher Ω")
        print("    This contradicts the bridge hypothesis!")
    
    # Also check I vs Ω
    r_I, p_I = pearsonr(I_arr, omega_arr)
    print(f"\nCorrelation(I, Ω):")
    print(f"  Pearson:  r = {r_I:.4f}  (p = {p_I:.2e})")
    
    # Binned analysis
    print("\n--- Binned Analysis ---")
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    E_bins = np.percentile(E_arr, percentiles)
    
    print(f"\n{'E percentile':>15} | {'Mean Ω':>8} | {'Count':>8}")
    print("-" * 40)
    
    for i in range(len(E_bins) - 1):
        mask = (E_arr >= E_bins[i]) & (E_arr < E_bins[i+1])
        if mask.sum() > 0:
            mean_omega = omega_arr[mask].mean()
            print(f"{percentiles[i]:>3}-{percentiles[i+1]:<3}%    | {mean_omega:>8.3f} | {mask.sum():>8}")
    
    return {
        'pearson_E_omega': float(r_pearson),
        'p_value_E_omega': float(p_pearson),
        'spearman_E_omega': float(r_spearman),
        'pearson_I_omega': float(r_I),
        'n_composites': len(E_arr)
    }


def test_E_polarity_partition(limit=50000):
    """
    Test 2: Do E > 0 vs E < 0 regions have different Ω distributions?
    
    From oscillation_attractor_dynamics:
    - E(prime) > 0 at 87% of primes
    - E(composite) < 0 on average
    """
    print("\n\n=== TEST 2: E POLARITY PARTITION ===\n")
    
    primes = get_primes(limit)
    prime_set = set(primes)
    
    S, I, E = compute_sec_simple(limit)
    
    # Partition by E sign
    E_pos_omega = []  # E > 0 composites
    E_neg_omega = []  # E < 0 composites
    
    E_pos_primes = 0
    E_neg_primes = 0
    
    for n in range(4, limit):
        if n in prime_set:
            if E[n] > 0:
                E_pos_primes += 1
            else:
                E_neg_primes += 1
        else:
            omega = big_omega(n)
            if E[n] > 0:
                E_pos_omega.append(omega)
            else:
                E_neg_omega.append(omega)
    
    print("Prime E-polarity:")
    total_primes = E_pos_primes + E_neg_primes
    print(f"  E > 0: {E_pos_primes} ({100*E_pos_primes/total_primes:.1f}%)")
    print(f"  E < 0: {E_neg_primes} ({100*E_neg_primes/total_primes:.1f}%)")
    
    print("\nComposite E-polarity vs Ω:")
    print(f"  E > 0 composites: {len(E_pos_omega):,}, mean Ω = {statistics.mean(E_pos_omega):.4f}")
    print(f"  E < 0 composites: {len(E_neg_omega):,}, mean Ω = {statistics.mean(E_neg_omega):.4f}")
    
    omega_diff = statistics.mean(E_neg_omega) - statistics.mean(E_pos_omega)
    print(f"\n  ΔΩ (E<0 - E>0) = {omega_diff:+.4f}")
    
    if omega_diff > 0:
        print("  ✓ E < 0 regions have HIGHER Ω (deeper crystallization)")
    else:
        print("  ⚠ E < 0 regions have LOWER Ω (contradicts hypothesis)")
    
    return {
        'primes_E_positive_frac': E_pos_primes / total_primes,
        'E_pos_count': len(E_pos_omega),
        'E_neg_count': len(E_neg_omega),
        'E_pos_mean_omega': statistics.mean(E_pos_omega),
        'E_neg_mean_omega': statistics.mean(E_neg_omega),
        'omega_diff': omega_diff
    }


def test_55_fibonacci_connection(limit=100000):
    """
    Test 3: Is the 55% (2-seeding rate) related to F₁₀ = 55?
    
    From exp_09: 2 seeds 55.7% of all composites
    F₁₀ = 55
    
    Hypothesis: The seeding rate converges to F₁₀/100 = 0.55
    """
    print("\n\n=== TEST 3: 55% = F₁₀/100 HYPOTHESIS ===\n")
    
    primes = get_primes(limit)
    prime_set = set(primes)
    
    FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    
    # Check seeding rates at different scales
    print("2-seeding rate at different scales:")
    print(f"{'Limit':>10} | {'2-seeds':>10} | {'Total':>10} | {'Rate':>8} | {'|Rate - 0.55|':>12}")
    print("-" * 60)
    
    scales = [1000, 5000, 10000, 25000, 50000, 75000, 100000]
    rates = []
    
    for scale in scales:
        if scale > limit:
            break
        seeded_by_2 = 0
        total_comp = 0
        for n in range(4, scale):
            if n not in prime_set:
                total_comp += 1
                if n % 2 == 0:
                    seeded_by_2 += 1
        rate = seeded_by_2 / total_comp
        rates.append((scale, rate))
        diff = abs(rate - 0.55)
        print(f"{scale:>10,} | {seeded_by_2:>10,} | {total_comp:>10,} | {rate:>8.5f} | {diff:>12.6f}")
    
    # Check if rate converges to F₁₀/100
    print(f"\nF₁₀ = {FIBONACCI[9]}")
    print(f"F₁₀/100 = 0.55")
    
    # Check rate as function of log(n) - is it asymptotically 0.55?
    final_rate = rates[-1][1]
    deviation = final_rate - 0.55
    print(f"\nAt N={limit}: rate = {final_rate:.5f}")
    print(f"Deviation from 0.55: {deviation:+.5f}")
    
    # Alternative: maybe it's 55% of EVEN numbers that are composite?
    # No wait - all evens > 2 are composite. So it's about what fraction of composites are even.
    # For large N: evens = N/2, primes ≈ N/ln(N), composites = N - primes
    # Even composites = evens - 1 = N/2 - 1
    # Rate = (N/2 - 1) / (N - N/ln(N)) → 1/2 * ln(N)/(ln(N) - 1) → 1/2 for large N
    
    print("\n--- Asymptotic Analysis ---")
    print("For large N:")
    print("  Even composites ≈ N/2 - 1")
    print(f"  Total composites ≈ N - N/ln(N) = N(1 - 1/ln(N))")
    print(f"  Rate ≈ (1/2) / (1 - 1/ln(N)) = (1/2) * ln(N)/(ln(N)-1)")
    
    import math
    for scale, rate in rates:
        ln_n = math.log(scale)
        theoretical = 0.5 * ln_n / (ln_n - 1)
        print(f"  N={scale}: ln(N)={ln_n:.2f}, theory={theoretical:.5f}, actual={rate:.5f}")
    
    # Actually the 55% might be about something else - let's check Ω distributions
    print("\n--- Checking if 55 appears elsewhere ---")
    
    # Count composites with Ω ≥ 2 divided by 2
    omega_2_plus = 0
    omega_total = 0
    for n in range(4, limit):
        if n not in prime_set:
            omega_total += 1
            if big_omega(n) >= 2:
                omega_2_plus += 1
    
    ratio_omega_2 = omega_2_plus / omega_total
    print(f"Ω ≥ 2 composites: {omega_2_plus:,} / {omega_total:,} = {ratio_omega_2:.5f}")
    print(f"  (This is composites with 'deep' crystallization)")
    
    return {
        '2_seeding_rates': dict(rates),
        'final_rate': final_rate,
        'omega_2_plus_rate': ratio_omega_2,
        'F10': FIBONACCI[9]
    }


def test_mobius_twist_at_55(limit=50000):
    """
    Test 4: Does depth 55 show special behavior?
    
    From oscillation_attractor: 55 levels = 1 Möbius half-twist
    Ξ - 1 = π/55
    """
    print("\n\n=== TEST 4: DEPTH 55 SPECIAL STRUCTURE ===\n")
    
    primes = get_primes(limit)
    prime_set = set(primes)
    
    # Group composites by Ω
    omega_groups = defaultdict(list)
    for n in range(4, limit):
        if n not in prime_set:
            omega = big_omega(n)
            omega_groups[omega].append(n)
    
    print("Distribution by Ω (crystallization depth):")
    print(f"{'Ω':>4} | {'Count':>8} | {'Fraction':>10} | {'Cumulative':>10}")
    print("-" * 50)
    
    total = sum(len(v) for v in omega_groups.values())
    cumulative = 0
    for omega in sorted(omega_groups.keys()):
        count = len(omega_groups[omega])
        frac = count / total
        cumulative += frac
        print(f"{omega:>4} | {count:>8} | {frac:>10.5f} | {cumulative:>10.5f}")
        if omega > 20:
            break
    
    # Check Ξ = 1 + π/55
    XI = 1 + np.pi / 55
    print(f"\nΞ = 1 + π/55 = {XI:.6f}")
    
    # Does any Ω distribution match Ξ?
    frac_omega_2 = len(omega_groups[2]) / total
    frac_omega_3 = len(omega_groups[3]) / total
    frac_omega_4 = len(omega_groups[4]) / total
    
    # Check ratios
    ratio_2_3 = frac_omega_2 / frac_omega_3 if frac_omega_3 > 0 else 0
    ratio_3_4 = frac_omega_3 / frac_omega_4 if frac_omega_4 > 0 else 0
    
    PHI = (1 + np.sqrt(5)) / 2
    
    print(f"\nRatios:")
    print(f"  frac(Ω=2)/frac(Ω=3) = {ratio_2_3:.4f} (φ = {PHI:.4f})")
    print(f"  frac(Ω=3)/frac(Ω=4) = {ratio_3_4:.4f} (φ = {PHI:.4f})")
    print(f"  frac(Ω=2) = {frac_omega_2:.4f} (1/Ξ = {1/XI:.4f})")
    
    return {
        'xi': XI,
        'frac_omega_2': frac_omega_2,
        'frac_omega_3': frac_omega_3,
        'ratio_2_3': ratio_2_3,
        'ratio_3_4': ratio_3_4,
        'phi': float(PHI)
    }


def test_injection_crystallization_balance(limit=50000):
    """
    Test 5: Does the I > 0 vs I < 0 split match φ?
    
    From SEC: I(prime) > 0 at 100% of primes
    From oscillation_attractor: φ is the injection/crystallization balance signature
    """
    print("\n\n=== TEST 5: INJECTION/CRYSTALLIZATION BALANCE ===\n")
    
    primes = get_primes(limit)
    prime_set = set(primes)
    
    S, I, E = compute_sec_simple(limit)
    
    # Primes: injection rate
    I_prime_pos = sum(1 for p in primes if p < limit and I[p] > 0)
    I_prime_total = len([p for p in primes if p < limit])
    
    # Composites: crystallization rate
    I_comp_neg = 0
    I_comp_total = 0
    
    for n in range(4, limit):
        if n not in prime_set:
            I_comp_total += 1
            if I[n] < 0:
                I_comp_neg += 1
    
    print("Impulse I(n) analysis:")
    print(f"  Primes with I > 0:    {I_prime_pos}/{I_prime_total} = {100*I_prime_pos/I_prime_total:.1f}%")
    print(f"  Composites with I < 0: {I_comp_neg}/{I_comp_total} = {100*I_comp_neg/I_comp_total:.1f}%")
    
    PHI = (1 + np.sqrt(5)) / 2
    ONE_OVER_PHI = 1 / PHI
    
    comp_neg_rate = I_comp_neg / I_comp_total
    prime_pos_rate = I_prime_pos / I_prime_total
    
    print(f"\n  Composite I<0 rate: {comp_neg_rate:.4f}")
    print(f"  1/φ = {ONE_OVER_PHI:.4f}")
    print(f"  Difference: {abs(comp_neg_rate - ONE_OVER_PHI):.4f}")
    
    # Balance ratio
    balance_ratio = comp_neg_rate / prime_pos_rate if prime_pos_rate > 0 else 0
    print(f"\n  Balance ratio (crystallization/injection): {balance_ratio:.4f}")
    print(f"  1/φ = {ONE_OVER_PHI:.4f}, φ-1 = {PHI-1:.4f}")
    
    return {
        'prime_injection_rate': prime_pos_rate,
        'comp_crystallization_rate': comp_neg_rate,
        'balance_ratio': balance_ratio,
        'phi': float(PHI),
        'one_over_phi': float(ONE_OVER_PHI)
    }


def save_results(results, filename):
    """Save results to JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")


def main():
    print("=" * 70)
    print("EXPERIMENT 11: BRIDGE E(n) ↔ Ω(n)")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Direct correlation
    results['E_omega_correlation'] = test_E_omega_correlation(limit=50000)
    
    # Test 2: Polarity partition
    results['E_polarity'] = test_E_polarity_partition(limit=50000)
    
    # Test 3: 55% = F₁₀?
    results['fibonacci_55'] = test_55_fibonacci_connection(limit=100000)
    
    # Test 4: Depth 55 structure
    results['depth_55'] = test_mobius_twist_at_55(limit=50000)
    
    # Test 5: Injection/crystallization balance
    results['balance'] = test_injection_crystallization_balance(limit=50000)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n1. E-Ω CORRELATION:")
    r = results['E_omega_correlation']['pearson_E_omega']
    if r < 0:
        print(f"   ✓ Negative correlation (r = {r:.4f})")
        print("   → E < 0 (discharged) correlates with HIGH Ω (deep crystallization)")
        print("   → The 'inversion' is CONSISTENT: both measure crystallization!")
    else:
        print(f"   ⚠ Positive correlation (r = {r:.4f}) - unexpected")
    
    print("\n2. E POLARITY:")
    diff = results['E_polarity']['omega_diff']
    if diff > 0:
        print(f"   ✓ E < 0 regions have +{diff:.4f} higher Ω")
    else:
        print(f"   ⚠ E < 0 regions have {diff:.4f} Ω difference")
    
    print("\n3. 55% FIBONACCI:")
    rate = results['fibonacci_55']['final_rate']
    print(f"   2-seeding rate: {rate:.5f}")
    print(f"   F₁₀/100 = 0.55, difference = {abs(rate - 0.55):.5f}")
    print("   → Rate is dominated by ln(N) asymptotic, not Fibonacci")
    
    print("\n4. Ω DISTRIBUTION:")
    r23 = results['depth_55']['ratio_2_3']
    print(f"   frac(Ω=2)/frac(Ω=3) = {r23:.4f} (φ = {results['depth_55']['phi']:.4f})")
    
    print("\n5. INJECTION/CRYSTALLIZATION:")
    bal = results['balance']['balance_ratio']
    print(f"   Balance ratio = {bal:.4f}")
    print(f"   1/φ = {results['balance']['one_over_phi']:.4f}")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"exp_11_bridge_E_omega_{timestamp}.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50000)
    args = parser.parse_args()
    main()
