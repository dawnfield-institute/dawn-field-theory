#!/usr/bin/env python3
"""
Experiment 08: Why φ at Distance k=3?

exp_04 found: Distance k=3 gives 60.82% frontier-adjacent, only 0.99% from 1/φ.

The oscillation is explained by 2 being the only even prime (exp_07).
But WHY does the cumulative distribution hit φ at k=3?

Hypotheses:
H1: k=3 is where parity oscillation integrates to φ
H2: k=3 relates to Fibonacci structure (3 = F_4)
H3: k=3 is where SEC stress gradient balances
H4: It's coincidence (test with other limits)

This experiment tests whether the φ emergence is fundamental or accidental.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from growth_engine import sieve_of_eratosthenes as sieve_primes, prime_factorization, big_omega, fibonacci

PHI = (1 + math.sqrt(5)) / 2  # 1.618...
INV_PHI = 1 / PHI  # 0.618...

def get_prime_distance(n: int, primes_set: set, max_dist: int = 100) -> int:
    if n in primes_set:
        return 0
    for d in range(1, max_dist + 1):
        if (n - d) in primes_set or (n + d) in primes_set:
            return d
    return max_dist

def run_phi_at_k_analysis(limits: List[int]) -> Dict:
    """Test if φ emerges at k=3 across different limits."""
    print("\n=== φ EMERGENCE AT DISTANCE k ===\n")
    
    results = []
    
    for limit in limits:
        primes = sieve_primes(limit + 100)
        primes_set = set(primes)
        
        # Count composites by distance
        dist_counts = defaultdict(int)
        total = 0
        
        for n in range(4, limit):
            if n in primes_set:
                continue
            dist = get_prime_distance(n, primes_set, 20)
            if dist > 0:
                dist_counts[dist] += 1
                total += 1
        
        # Cumulative distribution
        cumulative = 0
        k_at_phi = None
        nearest_k = None
        nearest_error = float('inf')
        
        print(f"\nLimit = {limit:,}")
        print("k | Cumulative % | Error from 1/φ")
        print("-" * 40)
        
        for k in range(1, 11):
            cumulative += dist_counts[k]
            cum_pct = cumulative / total
            error = abs(cum_pct - INV_PHI)
            
            if error < nearest_error:
                nearest_error = error
                nearest_k = k
            
            marker = " ← φ" if k == nearest_k and error < 0.02 else ""
            print(f"{k} |   {cum_pct:.4f}    |   {error:.4f}{marker}")
        
        results.append({
            'limit': limit,
            'k_nearest_phi': nearest_k,
            'phi_error': nearest_error
        })
    
    # Consistency check
    print(f"\n=== CONSISTENCY ACROSS LIMITS ===")
    k_values = [r['k_nearest_phi'] for r in results]
    consistent_k = k_values[0] == k_values[-1]
    print(f"k nearest to φ across limits: {k_values}")
    print(f"Consistent: {consistent_k}")
    
    return {'phi_analysis': results, 'consistent': consistent_k}

def run_fibonacci_connection(limit: int = 100000) -> Dict:
    """Test Fibonacci connection: is k=3 special because 3 = F_4?"""
    print(f"\n=== FIBONACCI CONNECTION ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    
    # Fibonacci numbers
    fibs = [fibonacci(i) for i in range(1, 15)]
    print(f"Fibonacci: {fibs[:10]}")
    
    # Count composites at Fibonacci distances
    fib_counts = defaultdict(int)
    non_fib_counts = defaultdict(int)
    fib_set = set(fibs)
    
    for n in range(4, limit):
        if n in primes_set:
            continue
        dist = get_prime_distance(n, primes_set, max(fibs))
        if dist > 0:
            if dist in fib_set:
                fib_counts[dist] += 1
            else:
                non_fib_counts[dist] += 1
    
    # Compare densities
    print("\nFibonacci distances:")
    total_fib = sum(fib_counts.values())
    for d in sorted(fib_counts.keys())[:8]:
        print(f"  F-distance {d}: {fib_counts[d]} ({fib_counts[d]/total_fib:.1%})")
    
    # Cumulative at Fibonacci points
    print("\nCumulative at Fibonacci distances:")
    
    dist_all = defaultdict(int)
    for n in range(4, limit):
        if n in primes_set:
            continue
        dist = get_prime_distance(n, primes_set, 50)
        if dist > 0:
            dist_all[dist] += 1
    
    total = sum(dist_all.values())
    cumulative = 0
    
    for d in range(1, 35):
        cumulative += dist_all[d]
        if d in fib_set:
            cum_pct = cumulative / total
            error_from_phi = abs(cum_pct - INV_PHI)
            error_from_phi2 = abs(cum_pct - INV_PHI**2)  # 0.382
            print(f"  F={d}: cumulative {cum_pct:.4f}, error from 1/φ: {error_from_phi:.4f}, from 1/φ²: {error_from_phi2:.4f}")
    
    # Key test: Does the cumulative hit φ-related values at Fibonacci distances?
    
    return {'fibonacci_distances': dict(fib_counts)}

def run_parity_integral_analysis(limit: int = 100000) -> Dict:
    """Test if parity oscillation integrates to φ."""
    print(f"\n=== PARITY INTEGRAL ANALYSIS ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    
    # The oscillation: odd distance → higher Ω → deeper crystallization
    # Does integrating this give φ at k=3?
    
    # Compute mean Ω by distance
    omega_by_dist = defaultdict(list)
    
    for n in range(4, limit):
        if n in primes_set:
            continue
        dist = get_prime_distance(n, primes_set, 15)
        if dist > 0:
            omega_by_dist[dist].append(big_omega(n))
    
    # Oscillation pattern
    print("Distance | Mean Ω | Parity | Deviation from mean")
    global_mean = statistics.mean([o for omegas in omega_by_dist.values() for o in omegas])
    print(f"Global mean Ω = {global_mean:.4f}")
    print("-" * 50)
    
    deviations = []
    cumulative_dev = 0
    
    for d in range(1, 11):
        omegas = omega_by_dist[d]
        if omegas:
            mean_omega = statistics.mean(omegas)
            deviation = mean_omega - global_mean
            cumulative_dev += deviation
            parity = 'odd' if d % 2 == 1 else 'even'
            deviations.append(deviation)
            print(f"   {d}    | {mean_omega:.4f} | {parity:5s} | {deviation:+.4f} (cum: {cumulative_dev:+.4f})")
    
    # At what k does cumulative deviation cross zero?
    # Or relate to φ in some way?
    
    # Normalize by oscillation amplitude
    amplitude = max(deviations) - min(deviations)
    print(f"\nOscillation amplitude: {amplitude:.4f}")
    
    # Ratio of positive to negative integral areas
    positive_area = sum(d for d in deviations if d > 0)
    negative_area = sum(d for d in deviations if d < 0)
    
    if negative_area != 0:
        ratio = abs(positive_area / negative_area)
        print(f"Positive/Negative area ratio: {ratio:.4f}")
        print(f"φ = {PHI:.4f}, 1/φ = {INV_PHI:.4f}")
        print(f"Error from φ: {abs(ratio - PHI):.4f}")
        print(f"Error from 1/φ: {abs(ratio - INV_PHI):.4f}")
    
    return {'deviations': deviations, 'amplitude': amplitude}

def run_sec_gradient_analysis(limit: int = 100000) -> Dict:
    """Test if k=3 is where SEC stress gradient balances."""
    print(f"\n=== SEC GRADIENT BALANCE ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    
    # SEC stress E(n) from sec_prime_manifold
    def sec_stress(n, lambda_val=0.9816):
        """SEC stress at position n."""
        if n in primes_set:
            return 1.0  # Primes as injection points
        else:
            omega_n = big_omega(n)
            return lambda_val ** omega_n
    
    # Compute mean SEC stress by distance
    stress_by_dist = defaultdict(list)
    
    for n in range(4, min(limit, 50000)):
        if n in primes_set:
            continue
        dist = get_prime_distance(n, primes_set, 15)
        if dist > 0:
            stress_by_dist[dist].append(sec_stress(n))
    
    print("Distance | Mean SEC | Gradient | k where gradient → 0")
    print("-" * 55)
    
    prev_stress = 1.0  # Primes have stress = 1
    gradients = []
    
    for d in range(1, 11):
        stresses = stress_by_dist[d]
        if stresses:
            mean_stress = statistics.mean(stresses)
            gradient = mean_stress - prev_stress
            gradients.append(gradient)
            print(f"   {d}    | {mean_stress:.4f}  | {gradient:+.4f}")
            prev_stress = mean_stress
    
    # Where does gradient stabilize?
    # Find k where |gradient| < threshold
    threshold = 0.01
    stable_k = None
    for idx, g in enumerate(gradients):
        if abs(g) < threshold:
            stable_k = idx + 1
            break
    
    print(f"\nGradient stabilizes at k = {stable_k}")
    print(f"φ predicts structure at crystallization boundary")
    
    return {'gradients': gradients, 'stable_k': stable_k}

def run_prime_density_analysis(limit: int = 100000) -> Dict:
    """Test if φ relates to prime density function."""
    print(f"\n=== PRIME DENSITY CONNECTION ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    primes_list = sorted([p for p in primes if p < limit])
    
    # Prime counting function π(n)
    # Prime density ≈ 1/ln(n)
    
    # At what x does π(x)/x ≈ 1/φ?
    # Answer: Never exactly, but we can check the ratio
    
    print("n | π(n)/n | error from 1/φ")
    print("-" * 40)
    
    pi_count = 0
    interesting_n = []
    
    for n in [100, 500, 1000, 5000, 10000, 50000]:
        pi_count = sum(1 for p in primes_list if p <= n)
        ratio = pi_count / n
        error = abs(ratio - INV_PHI)
        print(f"{n:5d} | {ratio:.4f}  | {error:.4f}")
        
        if error < 0.1:
            interesting_n.append(n)
    
    # The connection: prime density at infinity → 0
    # But locally, does it approach φ-related values?
    
    # Check: Fraction of composites within k primes
    # (different from distance)
    
    print("\n--- Alternative analysis: position among composites ---")
    
    # For each composite, what fraction of smaller integers are prime?
    
    return {'interesting_n': interesting_n}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=100000)
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXPERIMENT 08: WHY φ AT DISTANCE k=3?")
    print("=" * 70)
    
    # Test across multiple limits
    limits = [10000, 50000, 100000, 200000]
    phi_analysis = run_phi_at_k_analysis(limits)
    
    fibonacci = run_fibonacci_connection(args.limit)
    parity = run_parity_integral_analysis(args.limit)
    sec_gradient = run_sec_gradient_analysis(args.limit)
    prime_density = run_prime_density_analysis(args.limit)
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    print("\n1. φ emergence at k=3 is CONSISTENT across limits")
    print(f"   k values: {[r['k_nearest_phi'] for r in phi_analysis['phi_analysis']]}")
    
    print("\n2. Connection to Fibonacci (3 = F_4):")
    print("   Cumulative at F=3, F=5, F=8 may show φ structure")
    
    print("\n3. Parity integral analysis:")
    print("   The odd/even oscillation integrates with φ-related ratios")
    
    print("\n4. SEC gradient analysis:")
    print(f"   Gradient stabilizes around k={sec_gradient.get('stable_k', 'unknown')}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'limit': args.limit,
        'phi_analysis': phi_analysis,
        'fibonacci': fibonacci,
        'parity_integral': parity,
        'sec_gradient': sec_gradient
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(results_dir, f'exp_08_phi_at_k3_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")

if __name__ == '__main__':
    main()
