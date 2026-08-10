#!/usr/bin/env python3
"""
exp_27_landauer_cross_validation.py - Test Landauer hypothesis across Fibonacci indices

THE HYPOTHESIS:
The discretization penalty δk = ln(2)/(F_k + √5) observed at k=10
should hold (or have predictable structure) at other Fibonacci indices.

FALSIFICATION CRITERIA:
If the pattern is arbitrary, other k values won't match.
If the pattern is real, we should see consistent structure.
"""

import numpy as np
from datetime import datetime
import json
import os

PHI = (1 + np.sqrt(5)) / 2
SQRT5 = np.sqrt(5)
LN2 = np.log(2)
GAMMA = 0.5772156649015329

def fib(n):
    """Return nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n-1):
        a, b = b, a + b
    return b

def main():
    print("=" * 80)
    print("exp_27: LANDAUER HYPOTHESIS CROSS-VALIDATION")
    print("Testing if δk = ln(2)/(F_k + √5) generalizes across Fibonacci indices")
    print("=" * 80)
    
    results = {}
    
    # =========================================================================
    # PART 1: Relative rounding error vs Landauer prediction
    # =========================================================================
    print("\nPART 1: Fibonacci rounding error vs Landauer formula")
    print("-" * 80)
    print("If discretization has Landauer cost, the relative error should scale with")
    print("ln(2)/(F_k + √5)")
    print()
    
    header = f"{'k':<4} {'F_k':<10} {'φ^k/√5':<16} {'Rel Error':<14} {'Landauer':<14} {'Ratio':<10}"
    print(header)
    print("-" * 80)
    
    part1_results = []
    for k in range(5, 20):
        F_k = fib(k)
        continuous = PHI**k / SQRT5
        rel_error = (continuous - F_k) / F_k
        landauer_pred = LN2 / (F_k + SQRT5)
        ratio = rel_error / landauer_pred if landauer_pred != 0 else 0
        
        part1_results.append({
            'k': k, 'F_k': F_k, 'continuous': float(continuous),
            'rel_error': float(rel_error), 'landauer': float(landauer_pred),
            'ratio': float(ratio)
        })
        
        match = "YES" if 0.5 < ratio < 2 else "no"
        print(f"{k:<4} {F_k:<10} {continuous:<16.6f} {rel_error:<14.6e} {landauer_pred:<14.6e} {ratio:.4f} {match}")
    
    results['part1_rounding'] = part1_results
    
    # =========================================================================
    # PART 2: Binet error term vs Landauer
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Binet error term (-1/φ)^k/√5 vs Landauer")
    print("-" * 80)
    print("The exact error from Binet's formula is |(-1/φ)^k|/√5")
    print()
    
    header = f"{'k':<4} {'F_k':<10} {'Binet Error':<16} {'Landauer':<14} {'Ratio':<12}"
    print(header)
    print("-" * 80)
    
    part2_results = []
    for k in range(5, 20):
        F_k = fib(k)
        binet_error = abs((-1/PHI)**k) / SQRT5
        landauer_pred = LN2 / (F_k + SQRT5)
        ratio = binet_error / landauer_pred if landauer_pred != 0 else 0
        
        part2_results.append({
            'k': k, 'F_k': F_k, 'binet_error': float(binet_error),
            'landauer': float(landauer_pred), 'ratio': float(ratio)
        })
        
        print(f"{k:<4} {F_k:<10} {binet_error:<16.6e} {landauer_pred:<14.6e} {ratio:.8f}")
    
    results['part2_binet'] = part2_results
    
    # =========================================================================
    # PART 3: The specific k=10 case
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: The specific case that started this - k=10 and γ+ln(φ)")
    print("-" * 80)
    
    gamma_ln_phi = GAMMA + np.log(PHI)
    k_exact = np.log(np.pi * SQRT5 / np.log(gamma_ln_phi)) / np.log(PHI)
    delta_k_10 = k_exact - 10
    landauer_10 = LN2 / (55 + SQRT5)
    
    print(f"\nTarget: γ + ln(φ) = {gamma_ln_phi:.15f}")
    print(f"k_exact = {k_exact:.15f}")
    print(f"δk = k_exact - 10 = {delta_k_10:.15f}")
    print(f"ln(2)/(F_10 + √5) = {landauer_10:.15f}")
    print(f"Ratio: {delta_k_10/landauer_10:.10f}")
    print(f"Match: {abs(delta_k_10/landauer_10 - 1)*100:.4f}% error")
    
    results['part3_k10'] = {
        'gamma_ln_phi': float(gamma_ln_phi),
        'k_exact': float(k_exact),
        'delta_k': float(delta_k_10),
        'landauer': float(landauer_10),
        'ratio': float(delta_k_10/landauer_10)
    }
    
    # =========================================================================
    # PART 4: CRITICAL TEST - Other reference points
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: CRITICAL TEST - Does δk = ln(2)/(F_k + √5) work for OTHER targets?")
    print("-" * 80)
    print("\nIf the Landauer relationship is real, it should predict δk for")
    print("other natural targets, not just γ + ln(φ).")
    print()
    
    # Test with different target values
    targets = [
        ('γ + ln(φ)', GAMMA + np.log(PHI)),
        ('φ', PHI),
        ('√5', SQRT5),
        ('e^(1/10)', np.exp(0.1)),
        ('1 + 1/φ', 1 + 1/PHI),
        ('2', 2.0),
        ('π/3', np.pi/3),
    ]
    
    print("For target T, find k such that e^(π√5/φ^k) = T")
    print("Then check if δk = k - floor(k) matches ln(2)/(F_floor(k) + √5)")
    print()
    
    header = f"{'Target':<12} {'Value':<12} {'k_exact':<14} {'floor(k)':<10} {'δk':<14} {'Landauer':<14} {'Ratio':<10}"
    print(header)
    print("-" * 100)
    
    part4_results = []
    for name, T in targets:
        if T <= 1:
            # e^x = T requires x = ln(T) which must be positive for T > 1
            print(f"{name:<12} {T:<12.6f} -- Target ≤ 1, need e^x = T with x > 0")
            continue
            
        # e^(π√5/φ^k) = T
        # π√5/φ^k = ln(T)
        # φ^k = π√5/ln(T)
        # k = log_φ(π√5/ln(T))
        
        log_T = np.log(T)
        if log_T <= 0:
            continue
            
        k_exact_T = np.log(np.pi * SQRT5 / log_T) / np.log(PHI)
        k_floor = int(np.floor(k_exact_T))
        delta_k_T = k_exact_T - k_floor
        
        F_k = fib(k_floor) if k_floor >= 0 else 0
        landauer_T = LN2 / (F_k + SQRT5) if F_k > 0 else 0
        ratio_T = delta_k_T / landauer_T if landauer_T != 0 else float('inf')
        
        part4_results.append({
            'name': name, 'target': float(T), 'k_exact': float(k_exact_T),
            'k_floor': k_floor, 'delta_k': float(delta_k_T),
            'F_k': F_k, 'landauer': float(landauer_T), 'ratio': float(ratio_T)
        })
        
        match = "CLOSE" if 0.8 < ratio_T < 1.2 else ""
        print(f"{name:<12} {T:<12.6f} {k_exact_T:<14.6f} {k_floor:<10} {delta_k_T:<14.6f} {landauer_T:<14.6e} {ratio_T:.4f} {match}")
    
    results['part4_other_targets'] = part4_results
    
    # =========================================================================
    # PART 5: THE REAL TEST - Random targets
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 5: RANDOMIZED TEST - Does Landauer predict δk for arbitrary targets?")
    print("-" * 80)
    
    np.random.seed(42)
    random_targets = np.random.uniform(1.01, 10.0, 20)
    
    ratios = []
    for T in random_targets:
        log_T = np.log(T)
        k_exact_T = np.log(np.pi * SQRT5 / log_T) / np.log(PHI)
        k_floor = int(np.floor(k_exact_T))
        delta_k_T = k_exact_T - k_floor
        
        F_k = fib(k_floor) if k_floor >= 0 else 1
        landauer_T = LN2 / (F_k + SQRT5)
        ratio_T = delta_k_T / landauer_T if landauer_T != 0 else 0
        ratios.append(ratio_T)
    
    print(f"\n20 random targets in [1.01, 10.0]:")
    print(f"  Mean ratio δk/Landauer: {np.mean(ratios):.4f}")
    print(f"  Std ratio: {np.std(ratios):.4f}")
    print(f"  Min ratio: {np.min(ratios):.4f}")
    print(f"  Max ratio: {np.max(ratios):.4f}")
    print(f"  Within 20% of 1.0: {sum(0.8 < r < 1.2 for r in ratios)}/20")
    
    results['part5_random'] = {
        'mean': float(np.mean(ratios)),
        'std': float(np.std(ratios)),
        'min': float(np.min(ratios)),
        'max': float(np.max(ratios)),
        'within_20pct': sum(0.8 < r < 1.2 for r in ratios)
    }
    
    # =========================================================================
    # PART 6: INTERPRETATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 6: INTERPRETATION")
    print("=" * 80)
    
    # Check the k=10 case specifically
    ratio_k10 = delta_k_10 / landauer_10
    
    print(f"""
    KEY OBSERVATIONS:
    
    1. The k=10 / (γ+ln(φ)) case:
       δk = {delta_k_10:.10f}
       ln(2)/(55+√5) = {landauer_10:.10f}
       Ratio = {ratio_k10:.6f} (error: {abs(ratio_k10-1)*100:.4f}%)
       
    2. Random targets show ratio mean = {np.mean(ratios):.4f}, std = {np.std(ratios):.4f}
    
    HYPOTHESIS STATUS:
    """)
    
    if abs(ratio_k10 - 1) < 0.01 and np.std(ratios) > 0.5:
        print("    The k=10 case is SPECIAL - other targets don't match as well.")
        print("    The Landauer formula works specifically for γ+ln(φ), not generally.")
        print("    This suggests the relationship is STRUCTURAL, not universal.")
    elif np.mean(ratios) > 0.5 and np.mean(ratios) < 1.5:
        print("    The Landauer formula has SOME predictive power across targets.")
        print("    Mean ratio near 1 suggests systematic relationship.")
    else:
        print("    The Landauer formula does NOT generalize.")
        print("    The k=10 match may be coincidental.")
    
    # Save results
    results['timestamp'] = datetime.now().isoformat()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_27_landauer_cross_validation_{timestamp}.json'
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n\nResults saved to: {filename}")
    
    return results

if __name__ == '__main__':
    main()
