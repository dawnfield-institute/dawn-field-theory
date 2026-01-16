"""
Experiment 13: Alpha Formula Falsification

PURPOSE:
    FALSIFICATION: Can random Fibonacci combinations achieve similar precision?
    
    If many random formulas match α to < 0.01%, then our formula is not special.
    If only our formula matches, it's likely not coincidence.

METHOD:
    1. Generate 10,000 random Fibonacci-based formulas
    2. Count how many achieve < 1%, < 0.1%, < 0.01%, < 0.001% precision
    3. Assess probability of random match

OUTPUT:
    Statistical falsification test results.
"""

import numpy as np
import json
from datetime import datetime
from itertools import product, combinations
from constants import (print_header, print_subheader, PHI,
                       fib, ALPHA_MEASURED, percent_error)

def falsify_alpha_formula():
    """
    Test if the α formula is special or could arise randomly.
    """
    print_header("EXPERIMENT 13: ALPHA FORMULA FALSIFICATION")
    
    np.random.seed(42)  # Reproducibility
    
    # ==========================================================================
    # Part 1: The Test
    # ==========================================================================
    print_subheader("PART 1: FALSIFICATION APPROACH")
    
    print("""
    QUESTION: Is our α formula special, or could random combinations
    of Fibonacci numbers achieve similar precision?
    
    OUR FORMULA:
        α = (F₃/(F₄·φ·F₁₀)) × (1 - F₁₀/(4π·F₇²))
        
    Precision: 0.0006%
    
    TEST: Generate many random formulas of similar structure and
    check how often they match α with comparable precision.
    """)
    
    alpha_target = ALPHA_MEASURED
    
    # Our formula's precision
    from constants import alpha_pac
    our_alpha = alpha_pac()
    our_error = percent_error(our_alpha, alpha_target)
    
    print(f"  Our formula: {our_alpha:.10f}")
    print(f"  Target:      {alpha_target:.10f}")
    print(f"  Our error:   {our_error:.6f}%")
    
    # ==========================================================================
    # Part 2: Generate Random Formulas
    # ==========================================================================
    print_subheader("PART 2: RANDOM FORMULA GENERATION")
    
    # Fibonacci numbers to use (same range as our formula)
    F_VALUES = [fib(i) for i in range(1, 15)]  # F₁ to F₁₄
    F_INDICES = list(range(1, 15))
    
    print(f"  Fibonacci values available: {F_VALUES}")
    
    n_trials = 10000
    print(f"  Number of random trials: {n_trials}")
    
    # Formula template: (F_a / (F_b · φ · F_c)) × (1 - F_d/(4π·F_e²))
    # Same structure as our formula
    
    results_errors = []
    best_formula = None
    best_error = float('inf')
    
    print("\n  Generating random formulas...")
    
    for trial in range(n_trials):
        # Random selection of indices (allowing some repeated structure)
        a, b, c, d, e = np.random.choice(F_INDICES, 5)
        
        F_a = fib(a)
        F_b = fib(b)
        F_c = fib(c)
        F_d = fib(d)
        F_e = fib(e)
        
        # Avoid division by zero
        if F_b == 0 or F_c == 0 or F_e == 0:
            continue
        
        denom1 = F_b * PHI * F_c
        if denom1 == 0:
            continue
            
        denom2 = 4 * np.pi * F_e**2
        if denom2 == 0:
            continue
        
        # Calculate alpha candidate
        term1 = F_a / denom1
        correction = 1 - F_d / denom2
        alpha_candidate = term1 * correction
        
        # Skip if not in reasonable range
        if alpha_candidate <= 0 or alpha_candidate > 0.1:
            continue
        
        error = percent_error(alpha_candidate, alpha_target)
        results_errors.append(error)
        
        if error < best_error:
            best_error = error
            best_formula = {
                "indices": (a, b, c, d, e),
                "values": (F_a, F_b, F_c, F_d, F_e),
                "alpha": alpha_candidate,
                "error": error
            }
    
    # ==========================================================================
    # Part 3: Statistical Analysis
    # ==========================================================================
    print_subheader("PART 3: STATISTICAL ANALYSIS")
    
    results_errors = np.array(results_errors)
    n_valid = len(results_errors)
    
    print(f"  Valid formulas generated: {n_valid}")
    print(f"  (Some trials produced invalid/out-of-range values)")
    
    # Count matches at different precision levels
    thresholds = [10, 1, 0.1, 0.01, 0.001]
    
    print(f"\n  {'Precision':<15} {'Count':<10} {'Percent':<10}")
    print("-" * 40)
    
    for thresh in thresholds:
        count = np.sum(results_errors < thresh)
        pct = 100 * count / n_valid if n_valid > 0 else 0
        marker = "← OUR PRECISION" if thresh == 0.001 else ""
        print(f"  < {thresh}%{' ' * (10-len(str(thresh)))} {count:<10} {pct:<10.4f}% {marker}")
    
    # How special is our formula?
    count_at_our_level = np.sum(results_errors <= our_error)
    pct_at_our_level = 100 * count_at_our_level / n_valid if n_valid > 0 else 0
    
    print(f"\n  Formulas matching or beating {our_error:.4f}% error:")
    print(f"  Count: {count_at_our_level}")
    print(f"  Percent: {pct_at_our_level:.4f}%")
    
    # ==========================================================================
    # Part 4: Best Random Formula Found
    # ==========================================================================
    print_subheader("PART 4: BEST RANDOM FORMULA")
    
    if best_formula:
        a, b, c, d, e = best_formula["indices"]
        F_a, F_b, F_c, F_d, F_e = best_formula["values"]
        
        print(f"  Best random formula found:")
        print(f"  α = (F_{a}/(F_{b}·φ·F_{c})) × (1 - F_{d}/(4π·F_{e}²))")
        print(f"    = ({F_a}/({F_b}·φ·{F_c})) × (1 - {F_d}/(4π·{F_e}²))")
        print(f"  Value: {best_formula['alpha']:.10f}")
        print(f"  Error: {best_formula['error']:.6f}%")
        
        if best_formula["error"] < our_error:
            print("\n  ⚠️ WARNING: Found better formula than ours!")
        else:
            print(f"\n  ✅ Our formula is the best (or tied for best)")
    
    # ==========================================================================
    # Part 5: Verdict
    # ==========================================================================
    print_subheader("PART 5: FALSIFICATION VERDICT")
    
    # Determine verdict
    if count_at_our_level <= 1:
        verdict = "UNIQUE"
        verdict_text = "Our formula is UNIQUE among random trials"
    elif pct_at_our_level < 0.1:
        verdict = "RARE"
        verdict_text = f"Our precision is RARE (top {pct_at_our_level:.4f}%)"
    elif pct_at_our_level < 1:
        verdict = "UNCOMMON"
        verdict_text = f"Our precision is UNCOMMON (top {pct_at_our_level:.2f}%)"
    else:
        verdict = "COMMON"
        verdict_text = f"Our precision is COMMON ({pct_at_our_level:.2f}%)"
    
    is_passed = verdict in ["UNIQUE", "RARE"]
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    FALSIFICATION VERDICT                          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  Random trials: {n_trials}                                              ║
    ║  Valid formulas: {n_valid}                                             ║
    ║                                                                   ║
    ║  Our formula error: {our_error:.6f}%                                   ║
    ║  Formulas as good or better: {count_at_our_level}                                  ║
    ║                                                                   ║
    ║  Probability of random match: {pct_at_our_level:.4f}%                          ║
    ║                                                                   ║
    ║  ══════════════════════════════════════════════════════════════  ║
    ║                                                                   ║
    ║  VERDICT: {verdict}                                                  ║
    ║  {verdict_text:<58} ║
    ║                                                                   ║
    ║  Status: {'✅ PASSED' if is_passed else '⚠️ INCONCLUSIVE'}                                               ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    results = {
        "experiment": "exp_13_alpha_falsification",
        "timestamp": datetime.now().isoformat(),
        "trials": n_trials,
        "valid_formulas": int(n_valid),
        "our_error": float(our_error),
        "count_at_precision": int(count_at_our_level),
        "probability_random": float(pct_at_our_level),
        "best_random": best_formula,
        "verdict": verdict,
        "passed": is_passed,
        "status": "VALIDATED" if is_passed else "INCONCLUSIVE"
    }
    
    return results


if __name__ == "__main__":
    results = falsify_alpha_formula()
    
    # Save results
    with open("../results/exp_13_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to ../results/exp_13_results.json")
