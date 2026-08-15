"""
Experiment 13: Test if Ω(d=1)/Ω(d=2) → φ at Large Scale
========================================================

From exp_12 discovery:
- Ω(d=1) / Ω(d=2) = 1.518 at N=100k
- φ = 1.618

Hypothesis: This ratio converges to φ as N → ∞

Also test:
- Does mean Ω → 2 + φ?
- Does f(k)/f(k+1) → 1/φ for large k?
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


def test_ratio_convergence(scales=[10000, 50000, 100000, 250000, 500000]):
    """
    Test if Ω(d=1)/Ω(d=2) converges to φ.
    """
    print("=== Ω(d=1)/Ω(d=2) CONVERGENCE TEST ===\n")
    
    print(f"{'Scale':>10} | {'Ω(d=1)':>8} | {'Ω(d=2)':>8} | {'Ratio':>8} | {'Error vs φ':>12}")
    print("-" * 60)
    
    results = []
    
    for limit in scales:
        primes = sieve_of_eratosthenes(limit)
        prime_set = set(primes)
        
        d1_omegas = []
        d2_omegas = []
        
        for n in range(4, limit):
            if n not in prime_set:
                # Check if d=1 or d=2 from nearest prime
                d1_near = (n-1 in prime_set) or (n+1 in prime_set)
                d2_near = (n-2 in prime_set) or (n+2 in prime_set)
                
                if d1_near and not d2_near:
                    # Exactly d=1 from a prime, not d=2 from another
                    d1_omegas.append(big_omega(n))
                elif d2_near and not d1_near:
                    d2_omegas.append(big_omega(n))
        
        if d1_omegas and d2_omegas:
            mean_d1 = statistics.mean(d1_omegas)
            mean_d2 = statistics.mean(d2_omegas)
            ratio = mean_d1 / mean_d2
            error = abs(ratio - PHI)
            
            print(f"{limit:>10,} | {mean_d1:>8.4f} | {mean_d2:>8.4f} | {ratio:>8.4f} | {error:>12.6f}")
            results.append({
                'scale': limit,
                'omega_d1': mean_d1,
                'omega_d2': mean_d2,
                'ratio': ratio,
                'error': error
            })
    
    print(f"\nφ = {PHI:.6f}")
    
    # Check if error is decreasing
    if len(results) > 1:
        errors = [r['error'] for r in results]
        if errors[-1] < errors[0]:
            print("\n✓ Error is DECREASING - suggests convergence to φ")
        else:
            print("\n⚠ Error is NOT monotonically decreasing")
    
    return results


def test_mean_omega_convergence(scales=[10000, 50000, 100000, 250000, 500000]):
    """
    Test if mean Ω → 2 + φ = 3.618
    """
    print("\n\n=== MEAN Ω CONVERGENCE TEST ===\n")
    
    target = 2 + PHI
    print(f"Target: 2 + φ = {target:.6f}")
    print(f"\n{'Scale':>10} | {'Mean Ω':>10} | {'Error':>12}")
    print("-" * 40)
    
    results = []
    
    for limit in scales:
        primes = sieve_of_eratosthenes(limit)
        prime_set = set(primes)
        
        omegas = [big_omega(n) for n in range(4, limit) if n not in prime_set]
        mean_omega = statistics.mean(omegas)
        error = abs(mean_omega - target)
        
        print(f"{limit:>10,} | {mean_omega:>10.6f} | {error:>12.6f}")
        results.append({
            'scale': limit,
            'mean_omega': mean_omega,
            'error': error
        })
    
    return results


def test_geometric_ratio_convergence(limit=500000):
    """
    Test if f(k)/f(k+1) → 1/φ for large k.
    """
    print("\n\n=== GEOMETRIC RATIO CONVERGENCE ===\n")
    
    print(f"Target: 1/φ = {ONE_OVER_PHI:.6f}")
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    omega_counts = defaultdict(int)
    total = 0
    
    for n in range(4, limit):
        if n not in prime_set:
            omega_counts[big_omega(n)] += 1
            total += 1
    
    fracs = {k: omega_counts[k] / total for k in sorted(omega_counts.keys())}
    
    print(f"\n{'k':>4} | {'f(k)':>10} | {'f(k)/f(k+1)':>12} | {'Error vs 1/φ':>14}")
    print("-" * 50)
    
    ratios = []
    for k in range(2, 15):
        if fracs.get(k, 0) > 0 and fracs.get(k+1, 0) > 0:
            ratio = fracs[k+1] / fracs[k]
            error = abs(ratio - ONE_OVER_PHI)
            print(f"{k:>4} | {fracs[k]:>10.6f} | {ratio:>12.6f} | {error:>14.6f}")
            ratios.append({'k': k, 'ratio': ratio, 'error': error})
    
    # Check convergence trend
    if len(ratios) > 3:
        early_error = statistics.mean([r['error'] for r in ratios[:3]])
        late_error = statistics.mean([r['error'] for r in ratios[-3:]])
        print(f"\nEarly errors (k=2-4): {early_error:.6f}")
        print(f"Late errors (k>10): {late_error:.6f}")
        
        if late_error < early_error:
            print("✓ Converging toward 1/φ at large k")
        else:
            print("⚠ Not clearly converging")
    
    return ratios


def test_twin_vs_nontwin_phi(limit=500000):
    """
    Test: Is Ω(near twin) / Ω(near non-twin) related to φ?
    """
    print("\n\n=== TWIN vs NON-TWIN PRIME Ω RATIO ===\n")
    
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    # Identify twin primes
    twin_primes = set()
    for i in range(len(primes) - 1):
        if primes[i+1] - primes[i] == 2:
            twin_primes.add(primes[i])
            twin_primes.add(primes[i+1])
    
    near_twin_omega = []
    near_nontwin_omega = []
    
    for n in range(4, limit):
        if n not in prime_set:
            # Check if nearest prime is twin
            for d in range(1, 20):
                left = n - d in prime_set
                right = n + d in prime_set
                if left or right:
                    if left and (n - d) in twin_primes:
                        near_twin_omega.append(big_omega(n))
                    elif right and (n + d) in twin_primes:
                        near_twin_omega.append(big_omega(n))
                    else:
                        near_nontwin_omega.append(big_omega(n))
                    break
    
    if near_twin_omega and near_nontwin_omega:
        mean_twin = statistics.mean(near_twin_omega)
        mean_nontwin = statistics.mean(near_nontwin_omega)
        ratio = mean_twin / mean_nontwin
        
        print(f"Near twin primes:     mean Ω = {mean_twin:.4f} (n = {len(near_twin_omega):,})")
        print(f"Near non-twin primes: mean Ω = {mean_nontwin:.4f} (n = {len(near_nontwin_omega):,})")
        print(f"\nRatio: {ratio:.6f}")
        print(f"φ = {PHI:.6f}")
        print(f"1/φ = {ONE_OVER_PHI:.6f}")
        print(f"Error vs φ: {abs(ratio - PHI):.6f}")
        
        return {
            'mean_twin': mean_twin,
            'mean_nontwin': mean_nontwin,
            'ratio': ratio
        }
    
    return {}


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
    print("EXPERIMENT 13: φ CONVERGENCE AT LARGE SCALE")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Ω(d=1)/Ω(d=2) → φ?
    results['ratio_convergence'] = test_ratio_convergence()
    
    # Test 2: Mean Ω → 2 + φ?
    results['mean_convergence'] = test_mean_omega_convergence()
    
    # Test 3: f(k)/f(k+1) → 1/φ?
    results['geometric_convergence'] = test_geometric_ratio_convergence(limit=500000)
    
    # Test 4: Twin vs non-twin
    results['twin_ratio'] = test_twin_vs_nontwin_phi(limit=500000)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results['ratio_convergence']:
        final = results['ratio_convergence'][-1]
        print(f"\n1. Ω(d=1)/Ω(d=2) at N={final['scale']:,}: {final['ratio']:.4f} (φ={PHI:.4f})")
    
    if results['mean_convergence']:
        final = results['mean_convergence'][-1]
        print(f"2. Mean Ω at N={final['scale']:,}: {final['mean_omega']:.4f} (2+φ={2+PHI:.4f})")
    
    if results['twin_ratio']:
        print(f"3. Twin/Non-twin Ω ratio: {results['twin_ratio']['ratio']:.4f}")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"exp_13_phi_convergence_{timestamp}.json")


if __name__ == "__main__":
    main()
