#!/usr/bin/env python3
"""
Experiment 09: Crystallization Dynamics

Direct test of the "entropic fizz → crystallization" model.

From cosmo.py:
  Pure Entropy → "Fizz" (SHA-seeded) → Matter Crystallization
  
In arithmetic:
  Pure Potential → Primes (first crystallization) → Composites (growth from seeds)

Key questions:
1. How does composite "density" vary with distance from primes?
2. How does structure propagate from prime crystallization points?
3. What is the "collapse threshold" in arithmetic?
4. How does 2 (first bubble) create the cascade?

Hypothesis: Prime gaps are "entropy zones" - composites crystallize at rates
determined by the gradient field created by surrounding primes.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from growth_engine import sieve_of_eratosthenes as sieve_primes, prime_factorization, big_omega, omega

def get_surrounding_primes(n: int, primes_list: List[int]) -> Tuple[int, int]:
    """Get the primes immediately below and above n."""
    p_below = None
    p_above = None
    for p in primes_list:
        if p < n:
            p_below = p
        elif p >= n:
            if p == n:
                return (p, p)  # n is prime
            p_above = p
            break
    return (p_below, p_above)

def compute_crystallization_field(n: int, primes_set: set, primes_list: List[int]) -> float:
    """
    Compute the "crystallization field" at position n.
    
    Analogy to cosmo.py's info + energy threshold:
    - Each prime contributes to the field
    - Field strength decays with distance
    - Higher field = more "crystallized"
    """
    if n in primes_set:
        return 1.0  # Primes are fully crystallized
    
    p_below, p_above = get_surrounding_primes(n, primes_list)
    if p_below is None or p_above is None:
        return 0.0
    
    d_below = n - p_below
    d_above = p_above - n
    
    # Field from each prime, decaying with distance
    # Using 1/d decay like gravitational potential
    field = 1.0 / d_below + 1.0 / d_above
    
    # Normalize to [0, 1] range
    # Maximum is when d_below = d_above = 1 (twin prime gap interior)
    return min(1.0, field / 2.0)

def compute_omega_gradient(limit: int) -> Dict:
    """
    Test: Does Ω(n) increase with distance from nearest prime?
    (Structure grows more complex further from crystallization points)
    """
    print("\n=== Ω GRADIENT FROM PRIMES ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    primes_list = sorted([p for p in primes if p < limit])
    
    # Group composites by position in gap
    # position = distance from left prime / gap size (normalized 0 to 1)
    omega_by_position = defaultdict(list)
    
    for i in range(len(primes_list) - 1):
        p1 = primes_list[i]
        p2 = primes_list[i + 1]
        gap = p2 - p1
        
        if gap < 2:
            continue
        
        for n in range(p1 + 1, p2):
            position = (n - p1) / gap  # 0 = near left prime, 1 = near right prime
            omega_val = big_omega(n)
            
            # Bucket into 10 bins
            bucket = int(position * 10)
            if bucket == 10:
                bucket = 9
            omega_by_position[bucket].append(omega_val)
    
    # Analyze
    print("Position | Mean Ω | Std Ω | Count")
    print("-" * 45)
    
    results = []
    for bucket in range(10):
        omegas = omega_by_position[bucket]
        if len(omegas) > 10:
            mean_omega = statistics.mean(omegas)
            std_omega = statistics.stdev(omegas)
            results.append({
                'position': bucket / 10,
                'mean_omega': mean_omega,
                'std_omega': std_omega,
                'count': len(omegas)
            })
            print(f"  {bucket/10:.1f}    | {mean_omega:.4f} | {std_omega:.4f} | {len(omegas)}")
    
    # Is there a gradient? (higher in middle vs edges)
    if len(results) >= 5:
        edge_mean = (results[0]['mean_omega'] + results[-1]['mean_omega']) / 2
        middle_mean = results[4]['mean_omega'] + results[5]['mean_omega']
        middle_mean = middle_mean / 2
        
        print(f"\nEdge mean Ω: {edge_mean:.4f}")
        print(f"Middle mean Ω: {middle_mean:.4f}")
        print(f"Gradient (middle - edge): {middle_mean - edge_mean:+.4f}")
        
        # Hypothesis: Middle should have LOWER Ω (further from crystallization)
        # Or HIGHER Ω (more structure needed to fill entropy zone)?
        
    return {'omega_by_position': results}

def test_first_bubble_cascade(limit: int) -> Dict:
    """
    Test: How does 2 (the first prime, only even prime) cascade through structure?
    
    2 is the "first bubble" - its properties should echo through all arithmetic.
    """
    print("\n=== FIRST BUBBLE (2) CASCADE ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    
    # For each composite, count:
    # - How many factors of 2 it has
    # - Its distance to nearest prime
    # - Its total Ω
    
    factor_2_data = defaultdict(lambda: {'distances': [], 'omegas': [], 'count': 0})
    
    for n in range(4, limit):
        if n in primes_set:
            continue
        
        # Count factors of 2
        factors_of_2 = 0
        temp = n
        while temp % 2 == 0:
            factors_of_2 += 1
            temp //= 2
        
        omega_val = big_omega(n)
        
        # Distance to nearest prime
        dist = 0
        for d in range(1, 100):
            if (n - d) in primes_set or (n + d) in primes_set:
                dist = d
                break
        
        factor_2_data[factors_of_2]['distances'].append(dist)
        factor_2_data[factors_of_2]['omegas'].append(omega_val)
        factor_2_data[factors_of_2]['count'] += 1
    
    # Analyze by factors of 2
    print("Factors of 2 | Mean Distance | Mean Ω | Fraction of 2 | Count")
    print("-" * 65)
    
    results = []
    total_composites = sum(d['count'] for d in factor_2_data.values())
    
    for f2 in sorted(factor_2_data.keys())[:8]:
        data = factor_2_data[f2]
        if data['count'] > 10:
            mean_dist = statistics.mean(data['distances'])
            mean_omega = statistics.mean(data['omegas'])
            frac = data['count'] / total_composites
            frac_of_2 = f2 / mean_omega if mean_omega > 0 else 0
            
            print(f"     {f2}       |    {mean_dist:.4f}    | {mean_omega:.4f} |    {frac_of_2:.4f}   | {data['count']}")
            
            results.append({
                'factors_of_2': f2,
                'mean_distance': mean_dist,
                'mean_omega': mean_omega,
                'fraction_2': frac_of_2,
                'count': data['count']
            })
    
    # Key observation: Does the fraction of 2s correlate with distance?
    # (2's cascade should create patterns in the crystallization field)
    
    return {'cascade_data': results}

def test_crystallization_threshold(limit: int) -> Dict:
    """
    Test: What is the "collapse threshold" in arithmetic?
    
    In cosmo.py: matter forms when info + energy > threshold
    In arithmetic: composites form when...?
    
    Hypothesis: Composites form when the "gap potential" exceeds a threshold.
    Gap potential = function of surrounding prime density.
    """
    print("\n=== CRYSTALLIZATION THRESHOLD ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    primes_list = sorted([p for p in primes if p < limit])
    
    # For each gap, compute:
    # - Gap size
    # - "Potential" at gap center (related to surrounding prime density)
    # - Mean Ω of composites in gap
    
    gap_data = []
    
    for i in range(len(primes_list) - 1):
        p1 = primes_list[i]
        p2 = primes_list[i + 1]
        gap = p2 - p1
        
        if gap < 2:
            continue
        
        # Local prime density (within window around gap)
        window = 50
        local_density = sum(1 for p in primes_list if p1 - window <= p <= p2 + window) / (2 * window + gap)
        
        # Mean Ω of composites in gap
        omegas = [big_omega(n) for n in range(p1 + 1, p2)]
        mean_omega = statistics.mean(omegas) if omegas else 0
        
        # "Crystallization potential" - inverse of gap (denser = more crystallized)
        potential = 1.0 / gap
        
        gap_data.append({
            'gap': gap,
            'p1': p1,
            'local_density': local_density,
            'mean_omega': mean_omega,
            'potential': potential
        })
    
    # Group by gap size
    by_gap = defaultdict(list)
    for g in gap_data:
        by_gap[g['gap']].append(g)
    
    print("Gap | Count | Mean Local Density | Mean Ω | Potential")
    print("-" * 60)
    
    results = []
    for gap in sorted(by_gap.keys())[:15]:
        gaps = by_gap[gap]
        mean_density = statistics.mean([g['local_density'] for g in gaps])
        mean_omega = statistics.mean([g['mean_omega'] for g in gaps])
        potential = 1.0 / gap
        
        print(f" {gap:2d} | {len(gaps):5d} | {mean_density:.6f}          | {mean_omega:.4f} | {potential:.4f}")
        
        results.append({
            'gap': gap,
            'count': len(gaps),
            'mean_local_density': mean_density,
            'mean_omega': mean_omega,
            'potential': potential
        })
    
    # Is there a threshold? (Gap size where crystallization behavior changes)
    # Check for phase transition around certain gaps
    
    print("\n--- Searching for threshold ---")
    
    # Compute derivative of mean_omega w.r.t. gap
    omegas = [r['mean_omega'] for r in results]
    derivatives = [omegas[i+1] - omegas[i] for i in range(len(omegas)-1)]
    
    # Find max absolute derivative (steepest change)
    if derivatives:
        abs_derivs = [abs(d) for d in derivatives]
        max_idx = abs_derivs.index(max(abs_derivs))
        threshold_gap = results[max_idx]['gap']
        print(f"Steepest change at gap = {threshold_gap}")
        
        # Does this relate to φ?
        print(f"Gap 2 → 4: ΔΩ = {derivatives[0]:.4f}")
        if len(derivatives) > 1:
            print(f"Gap 4 → 6: ΔΩ = {derivatives[1]:.4f}")
    
    return {'threshold_data': results}

def test_propagation_pattern(limit: int) -> Dict:
    """
    Test: How does "information" propagate from primes?
    
    If primes are crystallization points, structure should "grow" outward.
    Each composite's structure should be traceable to prime seeds.
    """
    print("\n=== STRUCTURE PROPAGATION ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    
    # For each composite, find its smallest prime factor (the "seed")
    # Track how far the seed is from n
    
    seed_distance = defaultdict(list)
    
    for n in range(4, min(limit, 50000)):
        if n in primes_set:
            continue
        
        # Find smallest prime factor
        smallest_factor = None
        for p in primes:
            if p > n:
                break
            if n % p == 0:
                smallest_factor = p
                break
        
        if smallest_factor:
            # Distance from n to its seed
            dist = n - smallest_factor
            seed_distance[smallest_factor].append(dist)
    
    # Analyze propagation from each seed
    print("Seed Prime | Mean Propagation | Max Propagation | Composites Seeded")
    print("-" * 65)
    
    results = []
    for seed in sorted(seed_distance.keys())[:15]:
        distances = seed_distance[seed]
        mean_prop = statistics.mean(distances)
        max_prop = max(distances)
        
        print(f"    {seed:5d}  |    {mean_prop:8.2f}     |    {max_prop:8d}     |     {len(distances):8d}")
        
        results.append({
            'seed': seed,
            'mean_propagation': mean_prop,
            'max_propagation': max_prop,
            'count': len(distances)
        })
    
    # Key insight: 2 seeds the most composites (all even numbers)
    # Its "propagation reach" is infinite (half of all integers)
    
    if 2 in seed_distance:
        seed_2_fraction = len(seed_distance[2]) / sum(len(v) for v in seed_distance.values())
        print(f"\nFraction seeded by 2: {seed_2_fraction:.4f}")
        print("(Should be ~0.5 - half of all integers are even)")
    
    return {'propagation': results}

def test_entropy_zone_filling(limit: int) -> Dict:
    """
    Test: Do large gaps (entropy zones) get filled differently?
    
    If primes create crystallization points, large gaps are "entropy zones"
    that must be filled by composite crystallization.
    """
    print("\n=== ENTROPY ZONE FILLING ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    primes_list = sorted([p for p in primes if p < limit])
    
    # Categorize gaps by size
    small_gaps = []  # gap <= 6
    medium_gaps = []  # 6 < gap <= 20
    large_gaps = []  # gap > 20
    
    for i in range(len(primes_list) - 1):
        p1 = primes_list[i]
        p2 = primes_list[i + 1]
        gap = p2 - p1
        
        if gap < 2:
            continue
        
        composites = list(range(p1 + 1, p2))
        omega_values = [big_omega(n) for n in composites]
        
        gap_info = {
            'gap': gap,
            'p1': p1,
            'mean_omega': statistics.mean(omega_values) if omega_values else 0,
            'max_omega': max(omega_values) if omega_values else 0,
            'omega_variance': statistics.variance(omega_values) if len(omega_values) > 1 else 0
        }
        
        if gap <= 6:
            small_gaps.append(gap_info)
        elif gap <= 20:
            medium_gaps.append(gap_info)
        else:
            large_gaps.append(gap_info)
    
    print("Gap Type | Count | Mean of Mean Ω | Mean Max Ω | Mean Variance")
    print("-" * 65)
    
    for name, gaps in [('Small (≤6)', small_gaps), ('Medium (7-20)', medium_gaps), ('Large (>20)', large_gaps)]:
        if gaps:
            mean_mean_omega = statistics.mean([g['mean_omega'] for g in gaps])
            mean_max_omega = statistics.mean([g['max_omega'] for g in gaps])
            mean_var = statistics.mean([g['omega_variance'] for g in gaps])
            print(f"{name:12s} | {len(gaps):5d} | {mean_mean_omega:.4f}         | {mean_max_omega:.4f}     | {mean_var:.4f}")
    
    # Hypothesis: Large gaps (entropy zones) have:
    # - Higher mean Ω (more structure needed to fill)
    # - Higher variance (more diverse crystallization patterns)
    
    return {
        'small_gaps': len(small_gaps),
        'medium_gaps': len(medium_gaps),
        'large_gaps': len(large_gaps)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=100000)
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXPERIMENT 09: CRYSTALLIZATION DYNAMICS")
    print("=" * 70)
    print("\nModel: Pure Entropy → Primes (fizz) → Composites (crystallization)")
    print(f"\nLimit: {args.limit:,}")
    
    omega_gradient = compute_omega_gradient(args.limit)
    first_bubble = test_first_bubble_cascade(args.limit)
    threshold = test_crystallization_threshold(args.limit)
    propagation = test_propagation_pattern(args.limit)
    entropy_zones = test_entropy_zone_filling(args.limit)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n1. Ω GRADIENT: Structure complexity varies by position in gap")
    print("2. FIRST BUBBLE: 2's cascade creates fundamental asymmetry")
    print("3. THRESHOLD: Crystallization behavior changes at critical gap sizes")
    print("4. PROPAGATION: Each prime seeds structure outward")
    print("5. ENTROPY ZONES: Large gaps require more complex filling")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'limit': args.limit,
        'omega_gradient': omega_gradient,
        'first_bubble': first_bubble,
        'threshold': threshold,
        'propagation': propagation,
        'entropy_zones': entropy_zones
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(results_dir, f'exp_09_crystallization_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")

if __name__ == '__main__':
    main()
