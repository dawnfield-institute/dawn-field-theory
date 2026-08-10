"""
Experiment 16: THE φ CROSSING
============================

From exp_15:
- N=500k: f(5)/f(4) = 0.6188 (almost exactly 1/φ = 0.6180)
- N=1M:   f(5)/f(4) = 0.6284 (OVERSHOT by +1%)

Does f(5)/f(4) OSCILLATE around 1/φ?
Or does it converge from below, plateau at 1/φ, then diverge?

This is the critical test: is 1/φ a FIXED POINT or just a CROSSING POINT?
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


def fine_grained_scan():
    """
    Scan f(5)/f(4) at fine granularity to find exactly where we cross 1/φ
    """
    print("=" * 70)
    print("TEST 1: FINE-GRAINED SCAN AROUND THE CROSSING")
    print("=" * 70)
    
    # Scan from 300k to 1M in 50k steps
    scales = list(range(300000, 1050000, 50000))
    
    print(f"\n{'N':>12} | {'f(5)/f(4)':>12} | {'vs 1/φ':>12} | {'Status':>10}")
    print("-" * 55)
    
    crossings = []
    prev_sign = None
    
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
        diff = ratio - ONE_OVER_PHI
        
        sign = "+" if diff > 0 else "-"
        status = "ABOVE" if diff > 0 else "BELOW"
        
        if prev_sign is not None and sign != prev_sign:
            status = ">>> CROSSING <<<"
            crossings.append(limit)
        
        print(f"{limit:>12,} | {ratio:>12.6f} | {diff:>+12.6f} | {status:>10}")
        prev_sign = sign
    
    return crossings


def ratio_at_all_k(limit=1000000):
    """
    Check ALL ratios f(k+1)/f(k), not just f(5)/f(4)
    """
    print("\n" + "=" * 70)
    print("TEST 2: ALL RATIOS f(k+1)/f(k)")
    print("=" * 70)
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    omega_counts = defaultdict(int)
    for n in range(4, limit):
        if n not in prime_set:
            omega_counts[big_omega(n)] += 1
    total = sum(omega_counts.values())
    
    print(f"\nN = {limit:,}")
    print(f"1/φ = {ONE_OVER_PHI:.6f}")
    print(f"\n{'k':>4} | {'f(k)':>10} | {'f(k+1)/f(k)':>12} | {'vs 1/φ':>12} | {'Closest?':>10}")
    print("-" * 60)
    
    best_k = None
    best_error = float('inf')
    
    for k in range(2, 15):
        fk = omega_counts[k] / total
        fk1 = omega_counts[k+1] / total
        
        ratio = fk1 / fk if fk > 0 else 0
        diff = abs(ratio - ONE_OVER_PHI)
        
        if diff < best_error:
            best_error = diff
            best_k = k
        
        marker = "***" if diff < 0.01 else ""
        print(f"{k:>4} | {fk:>10.5f} | {ratio:>12.5f} | {abs(ratio - ONE_OVER_PHI):>+12.5f} | {marker:>10}")
    
    print(f"\nClosest to 1/φ: k = {best_k} (error = {best_error:.5f})")
    return best_k


def check_all_fibonacci_ratios(limit=1000000):
    """
    Fibonacci: each term is sum of previous two
    f(0), f(1), f(2)...
    
    Ratio r(n) = f(n)/f(n-1) → φ as n → ∞
    
    Do our Ω-counts follow Fibonacci-like recursion?
    """
    print("\n" + "=" * 70)
    print("TEST 3: ARE Ω-COUNTS FIBONACCI-LIKE?")
    print("=" * 70)
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    omega_counts = defaultdict(int)
    for n in range(4, limit):
        if n not in prime_set:
            omega_counts[big_omega(n)] += 1
    total = sum(omega_counts.values())
    
    # Fibonacci: f(n) = f(n-1) + f(n-2)
    # Therefore: f(n)/f(n-1) = 1 + f(n-2)/f(n-1) = 1 + 1/r(n-1)
    # At limit: φ = 1 + 1/φ, so φ² = φ + 1
    
    print(f"\nFibonacci test: f(k) ≈ f(k-1) + f(k-2)?")
    print(f"\n{'k':>4} | {'f(k)':>10} | {'f(k-1)+f(k-2)':>15} | {'Ratio':>10}")
    print("-" * 50)
    
    for k in range(4, 12):
        fk = omega_counts[k]
        fk1 = omega_counts[k-1]
        fk2 = omega_counts[k-2]
        fib_sum = fk1 + fk2
        ratio = fk / fib_sum if fib_sum > 0 else 0
        print(f"{k:>4} | {fk:>10,} | {fib_sum:>15,} | {ratio:>10.4f}")
    
    # Alternative: maybe it's INVERSE Fibonacci?
    # g(k) = g(k+1) + g(k+2)?
    print(f"\nInverse Fibonacci test: f(k) ≈ f(k+1) + f(k+2)?")
    print(f"\n{'k':>4} | {'f(k)':>10} | {'f(k+1)+f(k+2)':>15} | {'Ratio':>10}")
    print("-" * 50)
    
    for k in range(2, 10):
        fk = omega_counts[k]
        fk1 = omega_counts[k+1]
        fk2 = omega_counts[k+2]
        fib_sum = fk1 + fk2
        ratio = fk / fib_sum if fib_sum > 0 else 0
        print(f"{k:>4} | {fk:>10,} | {fib_sum:>15,} | {ratio:>10.4f}")
    
    return omega_counts


def cumulative_test(limit=1000000):
    """
    Test cumulative ratios: F(≤k) / F(≤k+1)
    """
    print("\n" + "=" * 70)
    print("TEST 4: CUMULATIVE RATIOS")
    print("=" * 70)
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    omega_counts = defaultdict(int)
    for n in range(4, limit):
        if n not in prime_set:
            omega_counts[big_omega(n)] += 1
    
    print(f"\nCumulative distribution and ratios:")
    print(f"\n{'k':>4} | {'F(≤k)':>12} | {'F(≤k)/F(≤k+1)':>15} | {'vs 1/φ':>12}")
    print("-" * 55)
    
    cumsum = 0
    prev_cumsum = 0
    
    for k in range(2, 15):
        prev_cumsum = cumsum
        cumsum += omega_counts[k]
        
        if k > 2:
            ratio = prev_cumsum / cumsum if cumsum > 0 else 0
            diff = ratio - ONE_OVER_PHI
            print(f"{k:>4} | {cumsum:>12,} | {ratio:>15.5f} | {diff:>+12.5f}")


def the_golden_angle(limit=1000000):
    """
    The golden angle (≈137.5°) appears in phyllotaxis.
    Does it appear in our Ω distribution?
    
    Golden angle = 360° / φ² ≈ 137.5°
    As fraction of circle: 1/φ² ≈ 0.382
    Or: 1 - 1/φ = 1/φ² (because 1/φ + 1/φ² = 1)
    """
    print("\n" + "=" * 70)
    print("TEST 5: GOLDEN ANGLE / φ² CONNECTION")
    print("=" * 70)
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    omega_counts = defaultdict(int)
    for n in range(4, limit):
        if n not in prime_set:
            omega_counts[big_omega(n)] += 1
    total = sum(omega_counts.values())
    
    # φ-related constants
    phi_inv = ONE_OVER_PHI  # ≈ 0.618
    phi_inv2 = 1 / PHI**2   # ≈ 0.382
    
    print(f"\n1/φ  = {phi_inv:.6f}")
    print(f"1/φ² = {phi_inv2:.6f}")
    print(f"Note: 1/φ + 1/φ² = {phi_inv + phi_inv2:.6f} (should be 1)")
    
    # Check ratios
    print(f"\nLooking for 1/φ² in consecutive ratios:")
    print(f"\n{'k':>4} | {'f(k+2)/f(k)':>12} | {'vs 1/φ²':>12}")
    print("-" * 40)
    
    for k in range(2, 10):
        fk = omega_counts[k] / total
        fk2 = omega_counts[k+2] / total
        ratio = fk2 / fk if fk > 0 else 0
        diff = ratio - phi_inv2
        print(f"{k:>4} | {ratio:>12.5f} | {diff:>+12.5f}")
    
    # Check cumulative fractions
    print(f"\nCumulative fractions:")
    cumsum = 0
    fractions = []
    for k in range(2, 15):
        cumsum += omega_counts[k]
        frac = cumsum / total
        fractions.append((k, frac))
    
    print(f"\n{'F(≤k)':>12} | {'vs 1/φ':>12} | {'vs 1/φ²':>12}")
    print("-" * 45)
    for k, frac in fractions[:8]:
        print(f"F(≤{k}): {frac:.5f} | {frac - phi_inv:>+12.5f} | {frac - phi_inv2:>+12.5f}")


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
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to: {filepath}")


def main():
    print("=" * 70)
    print("EXPERIMENT 16: THE φ CROSSING")
    print("=" * 70)
    print(f"\nKey question: Is 1/φ a FIXED POINT or just a CROSSING POINT?")
    
    results = {}
    
    # Test 1: Fine-grained scan
    crossings = fine_grained_scan()
    results['crossings'] = crossings
    
    # Test 2: All ratios
    best_k = ratio_at_all_k()
    results['best_k'] = best_k
    
    # Test 3: Fibonacci-like
    check_all_fibonacci_ratios()
    
    # Test 4: Cumulative
    cumulative_test()
    
    # Test 5: Golden angle
    the_golden_angle()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if crossings:
        print(f"\n1/φ crossing detected at N ≈ {crossings[0]:,}")
        print("   This means f(5)/f(4) passes THROUGH 1/φ, doesn't converge TO it!")
    else:
        print("\n   No crossing detected in scan range")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"exp_16_phi_crossing_{timestamp}.json")


if __name__ == "__main__":
    main()
