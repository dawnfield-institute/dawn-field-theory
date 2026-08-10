#!/usr/bin/env python3
"""
Experiment 10: Crystallization Field Decay Function

exp_09 found:
- Structure DENSEST near primes (Ω gradient = -0.70)
- Gap 2 (twins) = maximum crystallization (Ω = 5.27)
- Threshold at gap 2→4: ΔΩ = -1.41

Questions:
1. What is the decay function? (1/r, 1/r², exponential?)
2. Why is gap 2 special? (Twin prime effect)
3. Does the decay rate relate to φ or other constants?
4. Connection to cosmo.py parameters?

Hypothesis: The crystallization field decays as 1/d where d is distance,
with interference patterns from multiple primes creating the oscillation.
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
from growth_engine import sieve_of_eratosthenes as sieve_primes, prime_factorization, big_omega

PHI = (1 + math.sqrt(5)) / 2
INV_PHI = 1 / PHI

def test_decay_functions(limit: int = 100000) -> Dict:
    """Test which decay function best fits the Ω gradient."""
    print("\n=== TESTING DECAY FUNCTIONS ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    primes_list = sorted([p for p in primes if p < limit])
    
    # Collect Ω by distance to NEAREST prime
    omega_by_dist = defaultdict(list)
    
    for n in range(4, limit):
        if n in primes_set:
            continue
        
        # Find distance to nearest prime
        dist = 0
        for d in range(1, 50):
            if (n - d) in primes_set or (n + d) in primes_set:
                dist = d
                break
        
        if dist > 0:
            omega_by_dist[dist].append(big_omega(n))
    
    # Compute mean Ω by distance
    print("Distance | Mean Ω | 1/d model | 1/d² model | exp(-d) model")
    print("-" * 65)
    
    results = []
    base_omega = statistics.mean(omega_by_dist[1])  # Ω at distance 1
    
    for dist in range(1, 16):
        omegas = omega_by_dist[dist]
        if len(omegas) > 10:
            mean_omega = statistics.mean(omegas)
            
            # Predicted values from different decay models
            # Model: Ω(d) = α + β * decay(d)
            pred_1_d = base_omega * (1/dist)
            pred_1_d2 = base_omega * (1/dist**2)
            pred_exp = base_omega * math.exp(-(dist-1)/3)
            
            results.append({
                'distance': dist,
                'mean_omega': mean_omega,
                'pred_1_d': pred_1_d,
                'pred_1_d2': pred_1_d2,
                'pred_exp': pred_exp
            })
            
            print(f"   {dist:2d}    | {mean_omega:.4f} |   {pred_1_d:.4f}   |   {pred_1_d2:.4f}   |    {pred_exp:.4f}")
    
    # Fit decay rate
    if len(results) >= 5:
        # Linear regression on log(Ω) vs log(d) to find power law
        log_d = [math.log(r['distance']) for r in results if r['distance'] > 0]
        log_omega = [math.log(r['mean_omega']) for r in results]
        
        mean_log_d = statistics.mean(log_d)
        mean_log_omega = statistics.mean(log_omega)
        
        numerator = sum((x - mean_log_d) * (y - mean_log_omega) for x, y in zip(log_d, log_omega))
        denominator = sum((x - mean_log_d)**2 for x in log_d)
        
        if denominator > 0:
            slope = numerator / denominator
            intercept = mean_log_omega - slope * mean_log_d
            
            print(f"\nPower law fit: Ω ∝ d^{slope:.4f}")
            print(f"  (1/d decay would be slope = -1.0)")
            print(f"  (1/d² decay would be slope = -2.0)")
    
    return {'decay_data': results}

def test_twin_prime_enhancement(limit: int = 100000) -> Dict:
    """Why is gap 2 (twin primes) the maximum crystallization?"""
    print("\n=== TWIN PRIME ENHANCEMENT ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    primes_list = sorted([p for p in primes if p < limit])
    
    # Identify twin primes
    twins = set()
    for i in range(len(primes_list) - 1):
        if primes_list[i + 1] - primes_list[i] == 2:
            twins.add(primes_list[i])
            twins.add(primes_list[i + 1])
    
    # Compare Ω near twin primes vs non-twin primes
    omega_near_twin = []
    omega_near_nontwin = []
    
    for n in range(4, limit):
        if n in primes_set:
            continue
        
        # Find nearest prime
        nearest_p = None
        for d in range(1, 20):
            if (n - d) in primes_set:
                nearest_p = n - d
                break
            if (n + d) in primes_set:
                nearest_p = n + d
                break
        
        if nearest_p:
            omega_val = big_omega(n)
            if nearest_p in twins:
                omega_near_twin.append(omega_val)
            else:
                omega_near_nontwin.append(omega_val)
    
    twin_mean = statistics.mean(omega_near_twin)
    nontwin_mean = statistics.mean(omega_near_nontwin)
    
    print(f"Mean Ω near twin primes:     {twin_mean:.4f} (n={len(omega_near_twin)})")
    print(f"Mean Ω near non-twin primes: {nontwin_mean:.4f} (n={len(omega_near_nontwin)})")
    print(f"Enhancement: {twin_mean - nontwin_mean:+.4f}")
    
    # Is the enhancement related to φ?
    print(f"\nEnhancement / (1/φ) = {(twin_mean - nontwin_mean) / INV_PHI:.4f}")
    print(f"Enhancement × φ = {(twin_mean - nontwin_mean) * PHI:.4f}")
    
    # Twin prime density creates "double crystallization field"
    # Each prime contributes, so twins have 2x local intensity
    
    return {
        'twin_mean': twin_mean,
        'nontwin_mean': nontwin_mean,
        'enhancement': twin_mean - nontwin_mean
    }

def test_interference_pattern(limit: int = 50000) -> Dict:
    """Test if the oscillation arises from prime field interference."""
    print("\n=== FIELD INTERFERENCE PATTERN ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    primes_list = sorted([p for p in primes if p < limit])
    
    # For each composite, compute "field strength" from all nearby primes
    # and compare to actual Ω
    
    samples = []
    
    for n in range(4, min(limit, 20000)):
        if n in primes_set:
            continue
        
        # Compute total field from primes within distance 10
        field = 0.0
        for d in range(-10, 11):
            if d == 0:
                continue
            if (n + d) in primes_set:
                field += 1.0 / abs(d)  # 1/d decay
        
        omega_val = big_omega(n)
        samples.append({
            'n': n,
            'field': field,
            'omega': omega_val
        })
    
    # Bin by field strength
    field_bins = defaultdict(list)
    for s in samples:
        bin_idx = int(s['field'] * 5)  # Bin width 0.2
        field_bins[bin_idx].append(s['omega'])
    
    print("Field Bin | Mean Ω | Count")
    print("-" * 35)
    
    results = []
    for bin_idx in sorted(field_bins.keys())[:15]:
        omegas = field_bins[bin_idx]
        if len(omegas) > 10:
            mean_omega = statistics.mean(omegas)
            field_val = bin_idx / 5
            results.append({
                'field': field_val,
                'mean_omega': mean_omega,
                'count': len(omegas)
            })
            print(f"  {field_val:.1f}-{field_val+0.2:.1f}   | {mean_omega:.4f} | {len(omegas)}")
    
    # Correlation between field strength and Ω
    fields = [s['field'] for s in samples]
    omegas = [s['omega'] for s in samples]
    
    mean_field = statistics.mean(fields)
    mean_omega = statistics.mean(omegas)
    
    cov = sum((f - mean_field) * (o - mean_omega) for f, o in zip(fields, omegas)) / len(samples)
    std_field = statistics.stdev(fields)
    std_omega = statistics.stdev(omegas)
    
    correlation = cov / (std_field * std_omega) if std_field > 0 and std_omega > 0 else 0
    
    print(f"\nCorrelation(field, Ω): {correlation:.4f}")
    
    # If correlation is positive, higher field → higher Ω
    # This confirms that primes CREATE structure nearby
    
    return {'correlation': correlation, 'bins': results}

def test_cosmo_parameters(limit: int = 50000) -> Dict:
    """Find arithmetic equivalents of cosmo.py parameters."""
    print("\n=== COSMO.PY PARAMETER MAPPING ===\n")
    
    primes = sieve_primes(limit + 100)
    primes_set = set(primes)
    
    # cosmo.py parameters:
    # collapse_threshold = 0.4
    # energy_threshold = 0.05
    # info_growth_rate = 0.05
    # matter_generation_rate = 0.2
    
    # In arithmetic:
    # "Collapse" happens when a position is NOT prime (becomes composite)
    # What's the "threshold"?
    
    # Compute "info" and "energy" analogs
    # Info = density of primes nearby
    # Energy = "potential" from factorization
    
    samples = []
    
    for n in range(4, limit):
        if n in primes_set:
            continue
        
        # "Info" = local prime density (within window)
        window = 10
        local_primes = sum(1 for p in primes_set if n - window <= p <= n + window)
        info = local_primes / (2 * window)
        
        # "Energy" = 1/Ω (inverse complexity, available "energy")
        omega_val = big_omega(n)
        energy = 1.0 / omega_val if omega_val > 0 else 0
        
        # "Collapse potential" = info + energy
        potential = info + energy
        
        samples.append({
            'n': n,
            'info': info,
            'energy': energy,
            'potential': potential,
            'omega': omega_val
        })
    
    # Find threshold where behavior changes
    # Sort by potential and look for phase transition
    samples.sort(key=lambda x: x['potential'])
    
    # Look at top 10% vs bottom 10%
    n_samples = len(samples)
    top_10 = samples[int(0.9 * n_samples):]
    bottom_10 = samples[:int(0.1 * n_samples)]
    
    top_mean_omega = statistics.mean([s['omega'] for s in top_10])
    bottom_mean_omega = statistics.mean([s['omega'] for s in bottom_10])
    
    print(f"High potential (top 10%): mean Ω = {top_mean_omega:.4f}")
    print(f"Low potential (bottom 10%): mean Ω = {bottom_mean_omega:.4f}")
    
    # Find threshold - sample at 10 points instead of every position
    threshold_candidates = []
    step = max(1, len(samples) // 10)
    for i in range(step, len(samples) - step, step):
        below_omegas = [s['omega'] for s in samples[:i]]
        above_omegas = [s['omega'] for s in samples[i:]]
        if below_omegas and above_omegas:
            below = statistics.mean(below_omegas)
            above = statistics.mean(above_omegas)
            diff = abs(above - below)
            threshold_candidates.append((samples[i]['potential'], diff))
    
    if threshold_candidates:
        best_threshold = max(threshold_candidates, key=lambda x: x[1])
        print(f"\nOptimal threshold: {best_threshold[0]:.4f}")
        print(f"  (cosmo.py collapse_threshold = 0.4)")
        print(f"  Ratio: {best_threshold[0] / 0.4:.4f}")
        optimal = best_threshold[0]
    else:
        print("\nCould not determine threshold")
        optimal = 0.0
    
    return {
        'optimal_threshold': optimal,
        'high_potential_omega': top_mean_omega,
        'low_potential_omega': bottom_mean_omega
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50000)
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXPERIMENT 10: CRYSTALLIZATION FIELD DECAY")
    print("=" * 70)
    
    decay = test_decay_functions(args.limit)
    twin = test_twin_prime_enhancement(args.limit)
    interference = test_interference_pattern(args.limit)
    cosmo = test_cosmo_parameters(args.limit)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n1. DECAY: Field decays from primes (see power law fit)")
    print(f"2. TWINS: +{twin['enhancement']:.4f} Ω enhancement near twin primes")
    print(f"3. INTERFERENCE: Correlation(field, Ω) = {interference['correlation']:.4f}")
    print(f"4. THRESHOLD: Optimal = {cosmo['optimal_threshold']:.4f}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'limit': args.limit,
        'decay': decay,
        'twin': twin,
        'interference': interference,
        'cosmo': cosmo
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(results_dir, f'exp_10_field_decay_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")

if __name__ == '__main__':
    main()
