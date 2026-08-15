"""
Experiment 12: Deep Dive into Ω Distribution Structure
======================================================

From exp_11 discoveries:
1. E-Ω correlation = -0.35 (the inversion is CONSISTENT)
2. frac(Ω ≤ 3) = 0.555 ≈ 55% (WHERE DID THIS COME FROM?)
3. frac(Ω=3)/frac(Ω=4) = 1.39 (trending toward φ?)

This experiment digs deeper:
- Why does 55% appear in Ω ≤ 3 cumulative?
- What are the Ω ratios at different scales?
- Is there φ structure hidden in the distribution?
- How does Ω relate to distance from nearest prime?
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
import statistics
from collections import defaultdict
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from growth_engine import sieve_of_eratosthenes, big_omega


PHI = (1 + np.sqrt(5)) / 2
ONE_OVER_PHI = 1 / PHI
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]


def test_omega_distribution_scaling(limits=[10000, 50000, 100000, 500000]):
    """
    Test 1: How does Ω distribution change with scale?
    
    Do the ratios converge to φ at large N?
    """
    print("=== TEST 1: Ω DISTRIBUTION SCALING ===\n")
    
    results = {}
    
    for limit in limits:
        print(f"\n--- N = {limit:,} ---")
        primes = sieve_of_eratosthenes(limit)
        prime_set = set(primes)
        
        omega_counts = defaultdict(int)
        total = 0
        
        for n in range(4, limit):
            if n not in prime_set:
                omega_counts[big_omega(n)] += 1
                total += 1
        
        # Compute fractions
        fracs = {}
        for omega in sorted(omega_counts.keys()):
            fracs[omega] = omega_counts[omega] / total
        
        # Cumulative at key points
        cum_2 = fracs.get(2, 0)
        cum_3 = cum_2 + fracs.get(3, 0)
        cum_4 = cum_3 + fracs.get(4, 0)
        cum_5 = cum_4 + fracs.get(5, 0)
        
        print(f"  frac(Ω=2) = {fracs.get(2, 0):.5f}")
        print(f"  frac(Ω=3) = {fracs.get(3, 0):.5f}")
        print(f"  frac(Ω=4) = {fracs.get(4, 0):.5f}")
        print(f"  frac(Ω=5) = {fracs.get(5, 0):.5f}")
        print(f"\n  Cumulative Ω≤3: {cum_3:.5f} (target: 0.55)")
        print(f"  Cumulative Ω≤4: {cum_4:.5f}")
        print(f"  Cumulative Ω≤5: {cum_5:.5f}")
        
        # Ratios
        if fracs.get(3, 0) > 0:
            ratio_2_3 = fracs[2] / fracs[3]
        else:
            ratio_2_3 = 0
        if fracs.get(4, 0) > 0:
            ratio_3_4 = fracs[3] / fracs[4]
        else:
            ratio_3_4 = 0
        if fracs.get(5, 0) > 0:
            ratio_4_5 = fracs[4] / fracs[5]
        else:
            ratio_4_5 = 0
            
        print(f"\n  Ratios:")
        print(f"    f(2)/f(3) = {ratio_2_3:.4f} (φ = {PHI:.4f})")
        print(f"    f(3)/f(4) = {ratio_3_4:.4f} (φ = {PHI:.4f})")
        print(f"    f(4)/f(5) = {ratio_4_5:.4f} (φ = {PHI:.4f})")
        
        results[limit] = {
            'fracs': {k: float(v) for k, v in list(fracs.items())[:10]},
            'cumulative_3': cum_3,
            'cumulative_4': cum_4,
            'ratio_2_3': ratio_2_3,
            'ratio_3_4': ratio_3_4,
            'ratio_4_5': ratio_4_5
        }
    
    return results


def test_omega_vs_prime_distance(limit=100000):
    """
    Test 2: Does Ω correlate with distance to nearest prime?
    
    From exp_09: Ω gradient = -0.70 (structure near primes)
    Here: more detailed analysis
    """
    print("\n\n=== TEST 2: Ω vs DISTANCE TO NEAREST PRIME ===\n")
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    # For each composite, find distance to nearest prime
    distance_omega = defaultdict(list)
    
    for n in range(4, limit):
        if n not in prime_set:
            # Find nearest prime
            d = 1
            while n - d not in prime_set and n + d not in prime_set:
                d += 1
                if d > 100:
                    break
            if d <= 100:
                distance_omega[d].append(big_omega(n))
    
    print(f"{'Distance':>8} | {'Mean Ω':>8} | {'Std':>8} | {'Count':>8} | {'Even/Odd':>10}")
    print("-" * 55)
    
    means = []
    for d in sorted(distance_omega.keys()):
        if d > 20:
            break
        omegas = distance_omega[d]
        mean_omega = statistics.mean(omegas)
        std_omega = statistics.stdev(omegas) if len(omegas) > 1 else 0
        parity = "EVEN" if d % 2 == 0 else "ODD"
        means.append((d, mean_omega))
        print(f"{d:>8} | {mean_omega:>8.4f} | {std_omega:>8.4f} | {len(omegas):>8} | {parity:>10}")
    
    # Fit the oscillation
    even_means = [m for d, m in means if d % 2 == 0]
    odd_means = [m for d, m in means if d % 2 == 1]
    
    print(f"\nEven distance mean Ω: {statistics.mean(even_means):.4f}")
    print(f"Odd distance mean Ω:  {statistics.mean(odd_means):.4f}")
    print(f"Oscillation amplitude: {statistics.mean(even_means) - statistics.mean(odd_means):.4f}")
    
    # Check for φ in distance 1 vs 2
    d1_mean = statistics.mean(distance_omega[1])
    d2_mean = statistics.mean(distance_omega[2])
    print(f"\nΩ(d=1) / Ω(d=2) = {d1_mean / d2_mean:.4f} (φ = {PHI:.4f}, 1/φ = {ONE_OVER_PHI:.4f})")
    
    return {
        'distance_means': {d: float(statistics.mean(omegas)) for d, omegas in distance_omega.items() if d <= 20},
        'even_mean': statistics.mean(even_means),
        'odd_mean': statistics.mean(odd_means),
        'oscillation': statistics.mean(even_means) - statistics.mean(odd_means),
        'd1_d2_ratio': d1_mean / d2_mean
    }


def test_cumulative_55_origin(limit=100000):
    """
    Test 3: WHY is cumulative Ω≤3 = 55%?
    
    Is it:
    - A coincidence of the Ω distribution shape?
    - Related to F₁₀ = 55?
    - A manifestation of the log(N) asymptotic?
    """
    print("\n\n=== TEST 3: ORIGIN OF 55% ===\n")
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    # Theory: Ω distribution is approximately geometric
    # If P(Ω=k) ∝ r^k for some r, then P(Ω≤3) = sum of first terms
    
    omega_counts = defaultdict(int)
    total = 0
    
    for n in range(4, limit):
        if n not in prime_set:
            omega_counts[big_omega(n)] += 1
            total += 1
    
    # Fit geometric parameter
    fracs = {k: omega_counts[k] / total for k in sorted(omega_counts.keys())}
    
    # If geometric: f(k) = c * r^(k-2) for k >= 2
    # Then f(3)/f(2) = r, f(4)/f(3) = r, etc.
    
    r_estimates = []
    for k in range(3, 8):
        if fracs.get(k, 0) > 0 and fracs.get(k-1, 0) > 0:
            r = fracs[k] / fracs[k-1]
            r_estimates.append(r)
            print(f"f({k})/f({k-1}) = {r:.4f}")
    
    mean_r = statistics.mean(r_estimates)
    print(f"\nGeometric ratio estimate r = {mean_r:.4f}")
    print(f"1/φ = {ONE_OVER_PHI:.4f}")
    print(f"Difference: {abs(mean_r - ONE_OVER_PHI):.4f}")
    
    # If the distribution were exactly geometric with r = 1/φ:
    # P(Ω=2) = c
    # P(Ω=3) = c/φ
    # P(Ω=4) = c/φ²
    # Sum = c(1 + 1/φ + 1/φ² + ...) = c * φ/(φ-1) = c * φ² = 1
    # So c = 1/φ² = (φ-1)/φ = 1/φ - 1/φ² = φ - 2 ≈ 0.382
    
    # P(Ω≤3) = c + c/φ = c(1 + 1/φ) = c * (φ+1)/φ = c * φ = 0.382 * 1.618 ≈ 0.618 = 1/φ + 1/φ²
    
    print("\n--- Theoretical Analysis ---")
    print("If Ω distribution is geometric with ratio r:")
    print("  P(Ω=k) = c * r^(k-2) for k ≥ 2")
    print("  P(Ω≤3) = c * (1 + r) = c * (1 + r)")
    
    if mean_r < 1:
        c = (1 - mean_r)  # normalization for geometric starting at k=2
        cum_3_theory = c * (1 + mean_r)
        print(f"\n  With r = {mean_r:.4f}:")
        print(f"  c = 1 - r = {c:.4f}")
        print(f"  P(Ω≤3) = c(1+r) = {c * (1 + mean_r):.4f}")
        print(f"  Actual: {fracs[2] + fracs[3]:.4f}")
    
    # The actual distribution is NOT exactly geometric - it peaks at Ω=3
    print(f"\n--- Actual Distribution Peak ---")
    print(f"Mode: Ω = {max(fracs, key=fracs.get)}")
    print(f"f(2) = {fracs[2]:.5f}")
    print(f"f(3) = {fracs[3]:.5f} (MAXIMUM)")
    print(f"f(4) = {fracs[4]:.5f}")
    
    # The 55% might be because Ω=3 is the mode
    # Median would be around Ω=3 too
    cumulative = 0
    for k in sorted(fracs.keys()):
        cumulative += fracs[k]
        if cumulative >= 0.5:
            print(f"\nMedian at Ω = {k} (cumulative reaches 50% at {cumulative:.4f})")
            break
    
    return {
        'geometric_r': mean_r,
        'one_over_phi': float(ONE_OVER_PHI),
        'mode': max(fracs, key=fracs.get),
        'frac_2': fracs[2],
        'frac_3': fracs[3],
        'cumulative_3': fracs[2] + fracs[3]
    }


def test_phi_in_omega_dynamics(limit=100000):
    """
    Test 4: Where does φ actually appear in Ω dynamics?
    
    We know oscillation ratio ≈ 1.48 (not φ)
    But φ appears in SEC at critical λ*
    
    Maybe φ appears in the RATE of crystallization?
    """
    print("\n\n=== TEST 4: φ IN Ω DYNAMICS ===\n")
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    # Running mean of Ω as we traverse the number line
    # Does it oscillate around a φ-related value?
    
    window = 1000
    omegas = []
    running_means = []
    
    for n in range(4, limit):
        if n not in prime_set:
            omegas.append(big_omega(n))
            if len(omegas) >= window:
                running_means.append(statistics.mean(omegas[-window:]))
    
    # Statistics of running mean
    global_mean = statistics.mean(running_means)
    global_std = statistics.stdev(running_means)
    
    print(f"Running mean (window={window}):")
    print(f"  Mean of means: {global_mean:.4f}")
    print(f"  Std of means:  {global_std:.4f}")
    print(f"  Range: [{min(running_means):.4f}, {max(running_means):.4f}]")
    
    # Check if mean ≈ φ + 1 or some φ-related value
    print(f"\n  φ = {PHI:.4f}")
    print(f"  φ + 1 = {PHI + 1:.4f}")
    print(f"  2 + φ = {2 + PHI:.4f}")
    print(f"  Mean Ω = {global_mean:.4f}")
    print(f"  |Mean - (φ+1)| = {abs(global_mean - (PHI + 1)):.4f}")
    
    # Check ratios in gap statistics
    # Gap size g: how does mean Ω depend on g?
    print("\n--- Ω by Gap Size ---")
    
    gap_omega = defaultdict(list)
    for i in range(1, len(primes)):
        gap = primes[i] - primes[i-1]
        # Get all composites in this gap
        for n in range(primes[i-1] + 1, primes[i]):
            gap_omega[gap].append(big_omega(n))
    
    print(f"{'Gap':>6} | {'Mean Ω':>8} | {'Count':>8} | {'Gap/6':>8}")
    print("-" * 45)
    for gap in sorted(gap_omega.keys()):
        if gap > 30:
            break
        if len(gap_omega[gap]) > 0:
            mean_omega = statistics.mean(gap_omega[gap])
            print(f"{gap:>6} | {mean_omega:>8.4f} | {len(gap_omega[gap]):>8} | {gap/6:>8.2f}")
    
    # Gap 2 (twin primes) vs Gap 6 (most common)
    if 2 in gap_omega and 6 in gap_omega:
        ratio_2_6 = statistics.mean(gap_omega[2]) / statistics.mean(gap_omega[6])
        print(f"\nΩ(gap=2) / Ω(gap=6) = {ratio_2_6:.4f}")
        print(f"φ = {PHI:.4f}, 1/φ = {ONE_OVER_PHI:.4f}")
    
    return {
        'global_mean_omega': global_mean,
        'global_std_omega': global_std,
        'phi_plus_1': PHI + 1,
        'deviation_from_phi_plus_1': abs(global_mean - (PHI + 1))
    }


def test_fibonacci_omega_structure(limit=100000):
    """
    Test 5: Do Fibonacci numbers have special Ω?
    
    F_n for n > 12 are composite. What are their Ω values?
    """
    print("\n\n=== TEST 5: FIBONACCI Ω STRUCTURE ===\n")
    
    print(f"{'n':>4} | {'F_n':>12} | {'Ω(F_n)':>6} | {'log₂(F_n)':>10}")
    print("-" * 45)
    
    fib_omega = []
    for i, f in enumerate(FIBONACCI):
        if f > 1:
            omega = big_omega(f)
            log2_f = math.log2(f) if f > 0 else 0
            fib_omega.append((i+1, f, omega, log2_f))
            if f < limit:
                print(f"{i+1:>4} | {f:>12} | {omega:>6} | {log2_f:>10.2f}")
    
    # Check if Ω(F_n) / n approaches a constant
    print("\n--- Ω(F_n) / n ---")
    for n, f, omega, log2_f in fib_omega:
        if f > 3:
            ratio = omega / n if n > 0 else 0
            print(f"n={n}: Ω/n = {ratio:.4f}")
    
    return {'fibonacci_omega': fib_omega}


def save_results(results, filename):
    """Save results to JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(convert(i) for i in obj)
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to: {filepath}")


def main():
    print("=" * 70)
    print("EXPERIMENT 12: DEEP DIVE INTO Ω DISTRIBUTION")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Scaling
    results['scaling'] = test_omega_distribution_scaling(limits=[10000, 50000, 100000])
    
    # Test 2: Distance to prime
    results['distance'] = test_omega_vs_prime_distance(limit=100000)
    
    # Test 3: 55% origin
    results['origin_55'] = test_cumulative_55_origin(limit=100000)
    
    # Test 4: φ in dynamics
    results['phi_dynamics'] = test_phi_in_omega_dynamics(limit=100000)
    
    # Test 5: Fibonacci Ω
    results['fibonacci'] = test_fibonacci_omega_structure(limit=100000)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n1. SCALING: Ω distribution ratios")
    for limit, data in results['scaling'].items():
        print(f"   N={limit}: f(3)/f(4) = {data['ratio_3_4']:.4f}, cum(≤3) = {data['cumulative_3']:.4f}")
    
    print(f"\n2. DISTANCE OSCILLATION: {results['distance']['oscillation']:.4f}")
    print(f"   Ω(d=1)/Ω(d=2) = {results['distance']['d1_d2_ratio']:.4f}")
    
    print(f"\n3. 55% ORIGIN:")
    print(f"   Geometric ratio r = {results['origin_55']['geometric_r']:.4f} (1/φ = {ONE_OVER_PHI:.4f})")
    print(f"   Mode at Ω = {results['origin_55']['mode']}")
    
    print(f"\n4. φ IN DYNAMICS:")
    print(f"   Mean Ω = {results['phi_dynamics']['global_mean_omega']:.4f}")
    print(f"   |Mean - (φ+1)| = {results['phi_dynamics']['deviation_from_phi_plus_1']:.4f}")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"exp_12_omega_deep_dive_{timestamp}.json")


if __name__ == "__main__":
    main()
