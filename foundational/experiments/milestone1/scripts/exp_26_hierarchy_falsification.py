#!/usr/bin/env python3
"""
Experiment 26: Hierarchy Falsification

FALSIFICATION TEST: Is F₁₈₃ uniquely matched to the hierarchy?

Test whether other Fibonacci numbers could explain 10³⁸.
"""

import numpy as np
from constants import PHI, F7, fib, print_header, print_result

TARGET_LOG = 38.0  # Looking for 10³⁸

def fibonacci_magnitudes():
    """Calculate log₁₀ of various Fibonacci numbers."""
    results = []
    
    for k in range(170, 200):
        log_fk = k * np.log10(PHI) - 0.5 * np.log10(5)
        diff = abs(log_fk - TARGET_LOG)
        results.append({
            'k': k,
            'log10_Fk': log_fk,
            'diff_from_target': diff
        })
    
    return results

def find_best_match():
    """Find which Fibonacci index best matches 10³⁸."""
    results = fibonacci_magnitudes()
    best = min(results, key=lambda x: x['diff_from_target'])
    return best

def check_special_indices():
    """Check indices with special structure."""
    special = {
        '183 = F₇² + F₇ + 1': 183,
        '144 = F₁₂': 144,
        '169 = F₇²': 169,
        '196 = 14²': 196,
    }
    
    results = {}
    for name, k in special.items():
        log_fk = k * np.log10(PHI) - 0.5 * np.log10(5)
        results[name] = {
            'k': k,
            'log10': log_fk,
            'diff': abs(log_fk - TARGET_LOG)
        }
    
    return results

def main():
    print_header("Experiment 26: Hierarchy Falsification")
    
    best = find_best_match()
    special = check_special_indices()
    
    print(f"\n=== Target: 10^{TARGET_LOG} ===")
    
    print(f"\n=== Best Fibonacci Match ===")
    print(f"k = {best['k']}")
    print(f"log₁₀(F_k) = {best['log10_Fk']:.2f}")
    print(f"Difference: {best['diff_from_target']:.2f}")
    
    print("\n=== Special Indices ===")
    for name, data in special.items():
        marker = " ← BEST" if data['k'] == best['k'] else ""
        print(f"{name}: log₁₀(F_{data['k']}) = {data['log10']:.2f}, diff = {data['diff']:.2f}{marker}")
    
    # Check if 183 is special
    is_183_best = best['k'] == 183
    is_183_structural = True  # 183 = F₇² + F₇ + 1
    
    print("\n" + "="*50)
    if is_183_best:
        print(f"RESULT: F₁₈₃ IS the best match!")
        print(f"AND 183 = F₇² + F₇ + 1 has structural meaning.")
    else:
        print(f"RESULT: Best match is k = {best['k']}, not 183")
        print(f"But 183 is close and has structural meaning.")
    
    print_result("F₁₈₃ hierarchy", best['diff_from_target'] < 0.5)

if __name__ == "__main__":
    main()
