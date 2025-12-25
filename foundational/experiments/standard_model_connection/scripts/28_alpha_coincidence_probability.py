#!/usr/bin/env python3
"""
28_alpha_coincidence_probability.py - Statistical Analysis of α Formula

QUESTION: How likely is it that α = 1/137.036 can be expressed as a 
Fibonacci-based formula to 5.7 ppm by CHANCE?

If p < 10⁻⁶, that's significant evidence against cherry-picking.

The formula: α = (2/(3φF₁₀)) × (1 - F₁₀/(4πF₇²))
           = (2/(3 × 1.618 × 55)) × (1 - 55/(4π × 13²))
           = 0.007498... × 0.97395...
           = 0.0073017... ≈ 1/137.04 (5.7 ppm error)

The question: Given the "alphabet" of {φ, π, F_n}, how many formulas of 
similar complexity exist? And what's the probability one matches α?

Author: Dawn Field Institute
Date: December 24, 2025
Status: Probability analysis for publication
"""

import numpy as np
from itertools import combinations, permutations, product
from typing import List, Dict, Tuple
import json
from datetime import datetime

# =============================================================================
# CONSTANTS
# =============================================================================

PI = np.pi
PHI = (1 + np.sqrt(5)) / 2
E = np.e

# Measured value
ALPHA_CODATA = 1 / 137.035999084  # CODATA 2018

# Fibonacci numbers F_1 through F_15
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

# The formula that works
def alpha_formula(F_upper, F_lower):
    """α = (2/(3φF_upper)) × (1 - F_upper/(4πF_lower²))"""
    if F_upper <= 0 or F_lower <= 0:
        return None
    correction = 1 - F_upper / (4 * PI * F_lower**2)
    if correction <= 0:
        return None
    return (2 / (3 * PHI * F_upper)) * correction


# =============================================================================
# ANALYSIS 1: COUNT ALL POSSIBLE FORMULAS OF SIMILAR STRUCTURE
# =============================================================================

def count_formulas_same_structure():
    """
    Count formulas with EXACTLY the structure:
    
    (k/(m×φ×F_i)) × (1 - F_i/(n×π×F_j²))
    
    Where k, m, n ∈ {1, 2, 3, 4} (small integers)
    And i, j ∈ {1, ..., 15} (Fibonacci indices)
    """
    print("=" * 70)
    print("ANALYSIS 1: Same structure, different parameters")
    print("=" * 70)
    
    small_integers = [1, 2, 3, 4]
    fib_indices = list(range(15))
    
    total_formulas = 0
    matches_5ppm = []
    matches_100ppm = []
    matches_1000ppm = []
    
    target = ALPHA_CODATA
    
    for k in small_integers:
        for m in small_integers:
            for n in small_integers:
                for i in fib_indices:
                    for j in fib_indices:
                        F_i = FIB[i]
                        F_j = FIB[j]
                        
                        # Skip degenerate cases
                        if F_i == 0 or F_j == 0:
                            continue
                        
                        # Correction must be positive
                        correction = 1 - F_i / (n * PI * F_j**2)
                        if correction <= 0:
                            continue
                        
                        value = (k / (m * PHI * F_i)) * correction
                        
                        # Skip if value is clearly wrong (out of range)
                        if value <= 0 or value > 1:
                            continue
                        
                        total_formulas += 1
                        
                        error_ppm = abs(value - target) / target * 1e6
                        
                        if error_ppm <= 6:  # 5.7 ppm or better
                            matches_5ppm.append((k, m, n, i+1, j+1, value, error_ppm))
                        if error_ppm <= 100:
                            matches_100ppm.append((k, m, n, i+1, j+1, value, error_ppm))
                        if error_ppm <= 1000:
                            matches_1000ppm.append((k, m, n, i+1, j+1, value, error_ppm))
    
    print(f"\nTotal valid formulas in this family: {total_formulas}")
    print(f"Matches to 6 ppm:   {len(matches_5ppm)}")
    print(f"Matches to 100 ppm: {len(matches_100ppm)}")
    print(f"Matches to 1000 ppm: {len(matches_1000ppm)}")
    
    if matches_5ppm:
        print("\n5.7 ppm matches:")
        for m in sorted(matches_5ppm, key=lambda x: x[6]):
            print(f"  k={m[0]}, m={m[1]}, n={m[2]}, F_{m[3]}={FIB[m[3]-1]}, "
                  f"F_{m[4]}={FIB[m[4]-1]} → {m[5]:.10f} ({m[6]:.1f} ppm)")
    
    # Probability calculation
    p_6ppm = len(matches_5ppm) / total_formulas if total_formulas > 0 else 0
    
    print(f"\nP(match to 6 ppm | same structure) = {len(matches_5ppm)}/{total_formulas} = {p_6ppm:.2e}")
    
    return {
        'total_formulas': total_formulas,
        'matches_6ppm': len(matches_5ppm),
        'matches_100ppm': len(matches_100ppm),
        'matches_1000ppm': len(matches_1000ppm),
        'p_6ppm': p_6ppm
    }


# =============================================================================
# ANALYSIS 2: COUNT MORE GENERAL FORMULAS
# =============================================================================

def count_general_formulas():
    """
    Count a broader space of "Fibonacci-based" formulas.
    
    Consider formulas of form: F_a / (c × F_b × X)
    Where X ∈ {1, φ, π, e, φ², π², etc.}
    And correction factor (1 - F_c / (d × Y × F_e^p))
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: General Fibonacci formulas")
    print("=" * 70)
    
    # Expanded alphabet
    constants = {
        '1': 1,
        'φ': PHI,
        'π': PI,
        'e': E,
        'φ²': PHI**2,
        'π²': PI**2,
        '2π': 2*PI,
        '4π': 4*PI,
        '√2': np.sqrt(2),
        '√5': np.sqrt(5),
    }
    
    small_int = [1, 2, 3, 4, 5, 6]
    powers = [1, 2]
    fib_indices = list(range(1, 13))  # F_1 to F_12
    
    target = ALPHA_CODATA
    
    total = 0
    hits_6ppm = []
    hits_100ppm = []
    hits_1000ppm = []
    
    # Simple form: F_a / (k × const × F_b^p)
    for a in fib_indices:
        F_a = FIB[a-1]
        for k in small_int:
            for const_name, const_val in constants.items():
                for b in fib_indices:
                    F_b = FIB[b-1]
                    for p in powers:
                        denom = k * const_val * F_b**p
                        if denom == 0:
                            continue
                        
                        value = F_a / denom
                        
                        if value <= 0 or value > 1:
                            continue
                        
                        total += 1
                        error_ppm = abs(value - target) / target * 1e6
                        
                        formula_str = f"F_{a}/({k}×{const_name}×F_{b}^{p})"
                        
                        if error_ppm <= 6:
                            hits_6ppm.append((formula_str, value, error_ppm))
                        if error_ppm <= 100:
                            hits_100ppm.append((formula_str, value, error_ppm))
                        if error_ppm <= 1000:
                            hits_1000ppm.append((formula_str, value, error_ppm))
    
    print(f"\nSimple formulas (F_a / (k×const×F_b^p)): {total}")
    print(f"  Matches 6 ppm:    {len(hits_6ppm)}")
    print(f"  Matches 100 ppm:  {len(hits_100ppm)}")
    print(f"  Matches 1000 ppm: {len(hits_1000ppm)}")
    
    if hits_6ppm:
        print("\n  Best 6 ppm matches:")
        for h in sorted(hits_6ppm, key=lambda x: x[2])[:5]:
            print(f"    {h[0]} = {h[1]:.10f} ({h[2]:.1f} ppm)")
    
    # Compound form: base × (1 - correction)
    total_compound = 0
    hits_compound_6ppm = []
    
    for a in fib_indices[:8]:  # Limit range for tractability
        F_a = FIB[a-1]
        for k1 in small_int[:4]:
            for c1_name, c1_val in list(constants.items())[:5]:
                for b in fib_indices[:8]:
                    F_b = FIB[b-1]
                    
                    base = F_a / (k1 * c1_val * F_b) if (k1 * c1_val * F_b) != 0 else 0
                    if base == 0 or base > 1:
                        continue
                    
                    for c in fib_indices[:8]:
                        F_c = FIB[c-1]
                        for k2 in small_int[:4]:
                            for c2_name, c2_val in list(constants.items())[:5]:
                                for d in fib_indices[:8]:
                                    F_d = FIB[d-1]
                                    for p in powers:
                                        denom = k2 * c2_val * F_d**p
                                        if denom == 0:
                                            continue
                                        
                                        correction = F_c / denom
                                        if correction >= 1:
                                            continue
                                        
                                        value = base * (1 - correction)
                                        
                                        if value <= 0 or value > 1:
                                            continue
                                        
                                        total_compound += 1
                                        error_ppm = abs(value - target) / target * 1e6
                                        
                                        if error_ppm <= 6:
                                            formula_str = f"F_{a}/({k1}×{c1_name}×F_{b}) × (1 - F_{c}/({k2}×{c2_name}×F_{d}^{p}))"
                                            hits_compound_6ppm.append((formula_str, value, error_ppm))
    
    print(f"\nCompound formulas (base × (1 - correction)): {total_compound}")
    print(f"  Matches 6 ppm: {len(hits_compound_6ppm)}")
    
    if hits_compound_6ppm:
        print("\n  Best 6 ppm compound matches:")
        for h in sorted(hits_compound_6ppm, key=lambda x: x[2])[:10]:
            print(f"    {h[0]}")
            print(f"      = {h[1]:.10f} ({h[2]:.1f} ppm)")
    
    p_simple = len(hits_6ppm) / total if total > 0 else 0
    p_compound = len(hits_compound_6ppm) / total_compound if total_compound > 0 else 0
    
    return {
        'simple_total': total,
        'simple_6ppm': len(hits_6ppm),
        'compound_total': total_compound,
        'compound_6ppm': len(hits_compound_6ppm),
        'p_simple': p_simple,
        'p_compound': p_compound
    }


# =============================================================================
# ANALYSIS 3: INFORMATION-THEORETIC BOUND
# =============================================================================

def information_theoretic_analysis():
    """
    How many bits of information are in the formula vs the target?
    
    α has about 11 significant digits: log2(10^11) ≈ 37 bits
    Our formula specifies:
      - F_10 (one of 15 choices): log2(15) ≈ 4 bits
      - F_7  (one of 15 choices): log2(15) ≈ 4 bits
      - Structure (assumed): hard to quantify
    
    Key insight: The 5.7 ppm precision corresponds to:
      log2(1e6 / 5.7) ≈ 17.4 bits of "matching"
    
    If our formula has fewer than 17 bits of freedom, it's over-constrained.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Information-theoretic constraint")
    print("=" * 70)
    
    # Target precision
    ppm_error = 5.7
    bits_matched = np.log2(1e6 / ppm_error)
    
    print(f"\nTarget: α = 1/137.036 to 5.7 ppm")
    print(f"Bits of precision matched: log2(10^6 / 5.7) = {bits_matched:.1f} bits")
    
    # Formula degrees of freedom
    print("\nFormula: α = (2/(3φF₁₀)) × (1 - F₁₀/(4πF₇²))")
    print("\nDegrees of freedom:")
    print("  - Numerator constant (2): from {1,2,3,4} → 2 bits")
    print("  - Denominator integer (3): from {1,2,3,4} → 2 bits")  
    print("  - F_upper index (10): from {1,...,15} → 4 bits")
    print("  - Correction numerator (1): from {1,2,3,4} → 2 bits")
    print("  - Correction divisor (4): from {1,2,3,4} → 2 bits")
    print("  - F_lower index (7): from {1,...,15} → 4 bits")
    print("  - Power (2): from {1,2} → 1 bit")
    
    total_bits = 2 + 2 + 4 + 2 + 2 + 4 + 1
    print(f"\nTotal free bits: {total_bits}")
    print(f"Total combinations: 2^{total_bits} = {2**total_bits}")
    
    # But we're NOT free to choose structure (φ must be in denominator, etc.)
    # The structure itself encodes ~10 bits of information
    
    # Expected matches
    expected_at_6ppm = 2**total_bits * (6 / 1e6)  # Random chance
    
    print(f"\nExpected random matches at 6 ppm: {2**total_bits} × 6×10⁻⁶ = {expected_at_6ppm:.4f}")
    print(f"If expected < 1, finding a match is already surprising")
    
    return {
        'bits_matched': bits_matched,
        'bits_freedom': total_bits,
        'total_combinations': 2**total_bits,
        'expected_random_matches': expected_at_6ppm
    }


# =============================================================================
# ANALYSIS 4: CONSTRAINED SEARCH (Most Rigorous)
# =============================================================================

def constrained_search():
    """
    Most rigorous analysis: Define EXACTLY what formulas count as "Fibonacci-based"
    and search exhaustively.
    
    Formula space:
    1. Must use exactly 2 Fibonacci numbers (not arbitrary integers)
    2. Must use φ (the Fibonacci limit)
    3. Must use π (for physics relevance)
    4. Integer coefficients limited to {1, 2, 3, 4, 5, 6}
    5. Structure: A/B × (1 - C/D) where A,B,C,D use the ingredients
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Rigorous constrained search")
    print("=" * 70)
    
    target = ALPHA_CODATA
    integers = [1, 2, 3, 4, 5, 6]
    fib_nums = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    transcendentals = [('φ', PHI), ('π', PI)]
    
    # All formulas of form: k/(m × T × F_i) × (1 - F_j/(n × U × F_p^q))
    # where T, U ∈ {φ, π}
    
    matches = []
    total = 0
    
    for k in integers:
        for m in integers:
            for T_name, T in transcendentals:
                for i, F_i in enumerate(fib_nums):
                    if F_i == 0:
                        continue
                    
                    base = k / (m * T * F_i)
                    if base > 1 or base < 1e-4:
                        continue
                    
                    for j, F_j in enumerate(fib_nums):
                        if F_j == 0:
                            continue
                        for n in integers:
                            for U_name, U in transcendentals:
                                for p, F_p in enumerate(fib_nums):
                                    if F_p == 0:
                                        continue
                                    for q in [1, 2]:
                                        denom = n * U * F_p**q
                                        if denom == 0:
                                            continue
                                        
                                        correction = F_j / denom
                                        if correction >= 1:
                                            continue
                                        
                                        value = base * (1 - correction)
                                        if value <= 0 or value > 0.1:  # α-sized
                                            continue
                                        
                                        total += 1
                                        error_ppm = abs(value - target) / target * 1e6
                                        
                                        if error_ppm <= 6:
                                            matches.append({
                                                'k': k, 'm': m, 'T': T_name,
                                                'F_i': F_i, 'i_idx': i+1,
                                                'F_j': F_j, 'j_idx': j+1,
                                                'n': n, 'U': U_name,
                                                'F_p': F_p, 'p_idx': p+1,
                                                'q': q,
                                                'value': value,
                                                'error_ppm': error_ppm
                                            })
    
    print(f"\nTotal formulas in constrained space: {total}")
    print(f"Matches to 6 ppm: {len(matches)}")
    
    if matches:
        print("\nAll 6 ppm matches:")
        for m in sorted(matches, key=lambda x: x['error_ppm']):
            formula = f"{m['k']}/({m['m']}×{m['T']}×F_{m['i_idx']}) × (1 - F_{m['j_idx']}/({m['n']}×{m['U']}×F_{m['p_idx']}^{m['q']}))"
            print(f"  {formula}")
            print(f"    = {m['value']:.10f} ({m['error_ppm']:.2f} ppm)")
    
    # Probability
    p = len(matches) / total if total > 0 else 0
    
    # How significant?
    # Under null hypothesis (random), P(match at 6 ppm) = 6e-6 per trial
    # Observed: len(matches) / total
    
    # But each formula is NOT independent - they share structure
    # Conservative estimate: treat them as independent
    
    expected_random = total * 6e-6
    
    print(f"\nProbability analysis:")
    print(f"  P(≤6 ppm | this formula space) = {p:.2e}")
    print(f"  Expected random matches: {expected_random:.2f}")
    print(f"  Observed matches: {len(matches)}")
    
    # Binomial test
    from scipy import stats
    if total > 0:
        p_value = stats.binom.sf(len(matches) - 1, total, 6e-6)
        print(f"  Binomial p-value (one-tailed): {p_value:.2e}")
    else:
        p_value = 1.0
    
    return {
        'total_formulas': total,
        'matches_6ppm': len(matches),
        'probability': p,
        'expected_random': expected_random,
        'p_value': p_value,
        'matches': matches
    }


# =============================================================================
# ANALYSIS 5: COMPARISON TO OTHER CONSTANTS
# =============================================================================

def compare_to_other_constants():
    """
    If the formula structure is arbitrary, it should match OTHER constants
    equally well. Let's test this.
    
    Other constants: e, π, ln(2), γ (Euler), √2, √3, √5
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Cross-validation with other constants")
    print("=" * 70)
    
    constants_to_test = {
        '1/α = 137.036': 137.035999084,
        'α = 1/137': ALPHA_CODATA,
        'e = 2.718': np.e,
        'π = 3.14159': np.pi,
        'ln(2) = 0.693': np.log(2),
        'γ = 0.5772': 0.5772156649,  # Euler-Mascheroni
        '√2 = 1.414': np.sqrt(2),
        '√3 = 1.732': np.sqrt(3),
        '√5 = 2.236': np.sqrt(5),
        '1/e = 0.368': 1/np.e,
        '1/π = 0.318': 1/np.pi,
        'sin²θ_W = 0.231': 0.23122,
        'α_s = 0.118': 0.1179,
    }
    
    # For each constant, search for best Fibonacci-based match
    # Using our formula structure
    
    results = {}
    
    for name, target in constants_to_test.items():
        best_error = float('inf')
        best_formula = None
        best_value = None
        
        for i in range(1, 13):
            F_i = FIB[i-1]
            for j in range(1, 13):
                F_j = FIB[j-1]
                
                # Try both transcendentals
                for T, T_val in [('φ', PHI), ('π', PI)]:
                    for k in [1, 2, 3, 4]:
                        for m in [1, 2, 3, 4]:
                            base = k / (m * T_val * F_i) if F_i != 0 else 0
                            if base == 0:
                                continue
                            
                            for n in [1, 2, 3, 4]:
                                for U, U_val in [('φ', PHI), ('π', PI)]:
                                    for p in [1, 2]:
                                        for q in range(1, 10):
                                            F_q = FIB[q-1]
                                            denom = n * U_val * F_q**p
                                            if denom == 0:
                                                continue
                                            
                                            corr = F_j / denom
                                            if corr >= 1:
                                                continue
                                            
                                            value = base * (1 - corr)
                                            error = abs(value - target) / target
                                            
                                            if error < best_error:
                                                best_error = error
                                                best_value = value
                                                best_formula = f"{k}/({m}×{T}×F_{i}) × (1 - F_{j}/({n}×{U}×F_{q}^{p}))"
        
        results[name] = {
            'target': target,
            'best_value': best_value,
            'best_error_ppm': best_error * 1e6 if best_error < float('inf') else float('inf'),
            'best_formula': best_formula
        }
        
        print(f"\n{name}:")
        print(f"  Best match: {best_value:.10f} ({best_error*1e6:.1f} ppm)")
        if best_formula:
            print(f"  Formula: {best_formula}")
    
    # Compare how special α is
    print("\n" + "-" * 70)
    print("SUMMARY: How special is α?")
    print("-" * 70)
    
    errors = [(name, r['best_error_ppm']) for name, r in results.items()]
    errors.sort(key=lambda x: x[1])
    
    for name, err in errors:
        print(f"  {name:30s}: {err:10.1f} ppm")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " FINE STRUCTURE CONSTANT COINCIDENCE PROBABILITY ANALYSIS ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("\n" + "The formula: α = (2/(3φF₁₀)) × (1 - F₁₀/(4πF₇²))")
    print(f"Value: {alpha_formula(55, 13):.10f}")
    print(f"Target: {ALPHA_CODATA:.10f}")
    print(f"Error: {abs(alpha_formula(55, 13) - ALPHA_CODATA)/ALPHA_CODATA * 1e6:.2f} ppm")
    
    results = {}
    
    # Analysis 1
    results['same_structure'] = count_formulas_same_structure()
    
    # Analysis 2
    results['general'] = count_general_formulas()
    
    # Analysis 3
    results['information'] = information_theoretic_analysis()
    
    # Analysis 4
    results['constrained'] = constrained_search()
    
    # Analysis 5
    results['comparison'] = compare_to_other_constants()
    
    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    print("\n" + "═" * 70)
    print(" FINAL PROBABILITY ASSESSMENT ")
    print("═" * 70)
    
    constrained = results['constrained']
    
    print(f"\n1. FORMULA SPACE SIZE: {constrained['total_formulas']:,} formulas")
    print(f"   (Using φ, π, F_1...F_12, integers 1-6, powers 1-2)")
    
    print(f"\n2. MATCHES TO 6 PPM: {constrained['matches_6ppm']}")
    
    print(f"\n3. PROBABILITY: P = {constrained['probability']:.2e}")
    
    print(f"\n4. SIGNIFICANCE (binomial p-value): {constrained['p_value']:.2e}")
    
    if constrained['p_value'] < 1e-6:
        verdict = "HIGHLY SIGNIFICANT (p < 10⁻⁶)"
    elif constrained['p_value'] < 0.01:
        verdict = "SIGNIFICANT (p < 0.01)"
    elif constrained['p_value'] < 0.05:
        verdict = "MARGINALLY SIGNIFICANT (p < 0.05)"
    else:
        verdict = "NOT SIGNIFICANT (p ≥ 0.05)"
    
    print(f"\n5. VERDICT: {verdict}")
    
    # Context
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    
    if constrained['matches_6ppm'] <= 3:
        print("""
The formula α = (2/(3φF₁₀)) × (1 - F₁₀/(4πF₇²)) is one of very few
in this formula space that matches α to 6 ppm precision.

This is NOT easily explained by cherry-picking, because:
- The formula space is well-defined (φ, π, Fibonacci numbers)
- The structure is constrained (product of ratio and correction)
- Very few matches exist

The probability that such a match occurs by chance is {:.2e}.
""".format(constrained['p_value']))
    else:
        print("""
Multiple formulas match α to 6 ppm. This suggests either:
- The formula space is over-constrained (many formulas give small values)
- α is not uniquely determined by Fibonacci structure
- Or the formula structure itself needs justification
""")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': timestamp,
        'alpha_formula_value': alpha_formula(55, 13),
        'alpha_codata': ALPHA_CODATA,
        'error_ppm': abs(alpha_formula(55, 13) - ALPHA_CODATA)/ALPHA_CODATA * 1e6,
        'same_structure': results['same_structure'],
        'information': results['information'],
        'constrained': {
            'total_formulas': constrained['total_formulas'],
            'matches_6ppm': constrained['matches_6ppm'],
            'probability': constrained['probability'],
            'p_value': constrained['p_value']
        },
        'verdict': verdict
    }
    
    with open(f'../results/28_alpha_coincidence_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: ../results/28_alpha_coincidence_{timestamp}.json")
    
    return results


if __name__ == '__main__':
    main()
