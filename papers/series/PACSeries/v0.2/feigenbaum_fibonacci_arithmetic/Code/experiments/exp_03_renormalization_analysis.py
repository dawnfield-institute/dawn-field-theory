#!/usr/bin/env python3
"""
exp_08_renormalization_analysis.py
===================================

THEORETICAL ANALYSIS: RENORMALIZATION GROUP CONNECTIONS

This script explores the theoretical connections between the Feigenbaum
closed-form formulas and renormalization group (RG) theory.

Key discoveries:
1. The δ formula is a Möbius transformation with structured coefficients
2. The matrix determinant = -26π = -2 × F₇ × π (Fibonacci!)
3. All constants have form: (rational base) + O(π) correction
4. The universal coefficient a ≈ 55/36 (another F₁₀ connection)

Date: 2026-01-06
Status: THEORETICAL EXPLORATION - Patterns found, derivation open
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

# ===========================================================================
# CONSTANTS
# ===========================================================================

# Feigenbaum constants (high precision)
R_INF_KNOWN = 3.56994567187094490184200515138649893676383691151483237810797550
DELTA_KNOWN = 4.66920160910299067185320382047240927606510947219218
ALPHA_KNOWN = 2.50290787509589282228390287321821578636462643780702

# Universal coefficient in fixed-point expansion
A_UNIVERSAL = 1.5276329556642  # g*(x) = 1 - a*x^2 - ...

# Fibonacci numbers
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

# Key structural constant
XI = 1 + np.pi / 55


# ===========================================================================
# MÖBIUS TRANSFORMATION ANALYSIS
# ===========================================================================

def analyze_delta_mobius():
    """
    Analyze the Möbius transformation structure of the δ formula.
    
    δ = (14x + 32π)/(3x + 5π) where x = 3575 = 55 × 65
    
    Returns:
        dict: Analysis results
    """
    print()
    print("=" * 75)
    print("MÖBIUS TRANSFORMATION ANALYSIS OF δ")
    print("=" * 75)
    print()
    
    # Coefficients
    a, b = 14, 32 * np.pi
    c, d = 3, 5 * np.pi
    x = 3575  # = 55 * 65
    
    print("The δ formula has Möbius transformation structure:")
    print()
    print("  δ = (ax + b)/(cx + d)")
    print()
    print("  where:")
    print(f"    a = 14 = 2 × 7")
    print(f"    b = 32π = 2⁵ × π = {b:.10f}")
    print(f"    c = 3 = F₄")
    print(f"    d = 5π = F₅ × π = {d:.10f}")
    print(f"    x = 3575 = 55 × 65 = F₁₀ × (F₁₀ + 10)")
    print()
    
    # Transformation matrix
    print("Transformation matrix:")
    print(f"  | {a:4}    {b:.4f} |")
    print(f"  | {c:4}    {d:.4f}  |")
    print()
    
    # Determinant
    det = a * d - b * c
    det_simplified = -26 * np.pi
    
    print(f"Determinant = ad - bc")
    print(f"           = 14 × 5π - 32π × 3")
    print(f"           = 70π - 96π")
    print(f"           = -26π")
    print(f"           = {det:.10f}")
    print()
    print(f"Key insight: det = -26π = -2 × 13 × π = -2 × F₇ × π")
    print()
    
    # Compute δ
    delta_computed = (a * x + b) / (c * x + d)
    
    print(f"Computing δ:")
    print(f"  δ = (14 × 3575 + 32π)/(3 × 3575 + 5π)")
    print(f"    = {a * x + b:.6f} / {c * x + d:.6f}")
    print(f"    = {delta_computed:.15f}")
    print()
    print(f"Known δ = {DELTA_KNOWN:.15f}")
    print(f"Error: {100 * abs(delta_computed - DELTA_KNOWN) / DELTA_KNOWN:.10f}%")
    print()
    
    # Self-consistency: solve for x from known δ
    # δ(cx + d) = ax + b
    # δcx + δd = ax + b
    # x(δc - a) = b - δd
    # x = (b - δd)/(δc - a) = (32π - 5πδ)/(3δ - 14) = π(32 - 5δ)/(3δ - 14)
    
    x_from_delta = np.pi * (32 - 5 * DELTA_KNOWN) / (3 * DELTA_KNOWN - 14)
    
    print("Self-consistency check:")
    print(f"  Solving for x from δ_known:")
    print(f"  x = π(32 - 5δ)/(3δ - 14)")
    print(f"    = {x_from_delta:.10f}")
    print(f"  Actual x = 3575")
    print(f"  Ratio: {x_from_delta / 3575:.15f}")
    print(f"  Off by only {100 * abs(x_from_delta / 3575 - 1):.8f}%")
    print()
    
    return {
        'matrix': {'a': 14, 'b': '32π', 'c': 3, 'd': '5π'},
        'determinant': det,
        'determinant_symbolic': '-26π = -2 × F₇ × π',
        'x': 3575,
        'x_factored': '55 × 65 = F₁₀ × (F₁₀ + 10)',
        'delta_computed': delta_computed,
        'self_consistency_ratio': x_from_delta / 3575
    }


def analyze_coefficient_structure():
    """
    Analyze the Fibonacci and power-of-2 structure in coefficients.
    
    Returns:
        dict: Coefficient analysis
    """
    print()
    print("=" * 75)
    print("COEFFICIENT STRUCTURE ANALYSIS")
    print("=" * 75)
    print()
    
    print("Coefficients in Möbius matrix for δ:")
    print()
    print("  Position       Value    Factorization        Pattern")
    print("  ─────────────────────────────────────────────────────────")
    print("  a (x-coef num)   14     2 × 7               Mixed")
    print("  b (π-coef num)   32     2⁵                  Power of 2")
    print("  c (x-coef den)    3     F₄                  Fibonacci")
    print("  d (π-coef den)    5     F₅                  Fibonacci")
    print()
    
    print("Key observations:")
    print("  • π coefficients (32, 5) = (2⁵, F₅)")
    print("  • x coefficients (14, 3) = (2×7, F₄)")
    print("  • Determinant factor 26 = 2 × 13 = 2 × F₇")
    print("  • Base x = 3575 = 55 × 65 = F₁₀ × (F₁₀ + 10)")
    print()
    
    print("Fibonacci numbers appearing:")
    for i in [3, 4, 5, 6, 7, 9, 10]:
        print(f"  F_{i:2} = {FIBONACCI[i]}")
    print()
    
    print("Powers of 2 appearing:")
    print(f"  2⁴ + 1 = 17 (in r∞ formula)")
    print(f"  2⁵     = 32 (π coefficient in δ)")
    print(f"  2⁶ + 1 = 65 (factor in x = 55×65)")
    print()
    
    return {
        'fibonacci_in_coefficients': [3, 5, 13, 55],
        'powers_of_2': [17, 32, 65],
        'pattern': 'π coefficients are (2^5, F_5), x coefficients are (2×7, F_4)'
    }


def analyze_universal_coefficient():
    """
    Analyze the connection between universal coefficient a and 55/36.
    
    Returns:
        dict: Analysis results
    """
    print()
    print("=" * 75)
    print("UNIVERSAL COEFFICIENT ANALYSIS")
    print("=" * 75)
    print()
    
    print("The fixed-point function g*(x) has expansion:")
    print("  g*(x) = 1 - a·x² - b·x⁴ - c·x⁶ - ...")
    print()
    print("The universal coefficient 'a' is:")
    print(f"  a = {A_UNIVERSAL:.13f}")
    print()
    
    # Test 55/36
    approx = 55 / 36
    error = 100 * abs(A_UNIVERSAL - approx) / A_UNIVERSAL
    
    print(f"Comparison with 55/36:")
    print(f"  55/36 = {approx:.13f}")
    print(f"  Error = {error:.6f}%")
    print()
    
    print("This is remarkable: 55 = F₁₀ appears not just in r∞")
    print("but in the fixed-point structure itself!")
    print()
    
    print("Structural analysis of 55/36:")
    print(f"  55 = F₁₀ (10th Fibonacci)")
    print(f"  36 = 6² = (F₇ - 1)² = (13 - 1)² = 12² ... no")
    print(f"  36 = 6² = 2² × 3² = 4 × 9")
    print()
    
    # Try pi corrections
    print("Testing π corrections to 55/36:")
    for k in [5000, 10000, 20000, 21693]:
        test = 55/36 - np.pi/k
        err = 100 * abs(A_UNIVERSAL - test) / A_UNIVERSAL
        marker = " <--" if k == 21693 else ""
        print(f"  55/36 - π/{k:5} = {test:.10f}, error = {err:.6f}%{marker}")
    print()
    
    optimal_k = np.pi / (55/36 - A_UNIVERSAL)
    print(f"Optimal divisor: k = {optimal_k:.4f}")
    print()
    
    return {
        'a_universal': A_UNIVERSAL,
        '55_over_36': approx,
        'error_percent': error,
        'optimal_correction': optimal_k,
        'insight': '55 = F₁₀ appears in fixed-point coefficient, not just r∞'
    }


def analyze_perturbation_structure():
    """
    Analyze how all constants have (rational base) + π correction.
    
    Returns:
        dict: Perturbation analysis
    """
    print()
    print("=" * 75)
    print("PERTURBATION STRUCTURE: (RATIONAL BASE) + π CORRECTION")
    print("=" * 75)
    print()
    
    xi_m1 = np.pi / 55  # = ξ - 1
    
    results = {}
    
    # r∞ analysis
    r_base = np.pi * (55 + np.sqrt(17)) * (55 + np.pi) / 55**2
    r_error = 100 * abs(r_base - R_INF_KNOWN) / R_INF_KNOWN
    
    print("1. ACCUMULATION POINT r∞")
    print("-" * 50)
    print(f"   Rational base: π(55+√17)(55+π)/55²")
    print(f"   Base value: {r_base:.15f}")
    print(f"   Known:      {R_INF_KNOWN:.15f}")
    print(f"   Base error: {r_error:.6f}%")
    print(f"   Correction needed: {R_INF_KNOWN - r_base:.6e}")
    print()
    
    results['r_inf'] = {
        'base': 'π(55+√17)(55+π)/55²',
        'base_value': r_base,
        'base_error_percent': r_error,
        'correction': R_INF_KNOWN - r_base
    }
    
    # δ analysis
    delta_base = 14/3
    delta_error = 100 * abs(delta_base - DELTA_KNOWN) / DELTA_KNOWN
    delta_correction = DELTA_KNOWN - delta_base
    
    print("2. BIFURCATION RATIO δ")
    print("-" * 50)
    print(f"   Rational base: 14/3")
    print(f"   Base value: {delta_base:.15f}")
    print(f"   Known:      {DELTA_KNOWN:.15f}")
    print(f"   Base error: {delta_error:.6f}%")
    print(f"   Correction: {delta_correction:.15f}")
    print(f"   Correction / (ξ-1): {delta_correction / xi_m1:.10f}")
    print()
    
    results['delta'] = {
        'base': '14/3',
        'base_value': delta_base,
        'base_error_percent': delta_error,
        'correction': delta_correction,
        'correction_over_xi_m1': delta_correction / xi_m1
    }
    
    # α analysis
    alpha_base = 5/2
    alpha_error = 100 * abs(alpha_base - ALPHA_KNOWN) / ALPHA_KNOWN
    alpha_correction = ALPHA_KNOWN - alpha_base
    
    print("3. SCALING CONSTANT α")
    print("-" * 50)
    print(f"   Rational base: 5/2")
    print(f"   Base value: {alpha_base:.15f}")
    print(f"   Known:      {ALPHA_KNOWN:.15f}")
    print(f"   Base error: {alpha_error:.6f}%")
    print(f"   Correction: {alpha_correction:.15f}")
    print(f"   Correction / (ξ-1): {alpha_correction / xi_m1:.10f}")
    print()
    
    results['alpha'] = {
        'base': '5/2',
        'base_value': alpha_base,
        'base_error_percent': alpha_error,
        'correction': alpha_correction,
        'correction_over_xi_m1': alpha_correction / xi_m1
    }
    
    print("SUMMARY:")
    print("-" * 50)
    print("All three constants follow the pattern:")
    print("  constant = (simple rational) + O(π) correction")
    print()
    print("The rational bases are remarkably simple:")
    print("  r∞: involves √17, π, and 55")
    print("  δ: 14/3 ≈ 4.667")
    print("  α: 5/2 = 2.5")
    print()
    print("The π corrections bring precision from ~3 digits to 6-13 digits.")
    print()
    
    return results


def analyze_circle_map_connection():
    """
    Explore connection to circle doubling map.
    
    Returns:
        dict: Analysis results
    """
    print()
    print("=" * 75)
    print("CIRCLE MAP AND DOUBLING CONNECTIONS")
    print("=" * 75)
    print()
    
    print("The period-doubling universality class includes:")
    print("  • Logistic map: f(x) = rx(1-x)")
    print("  • Circle doubling: θ → 2θ (mod 2π)")
    print("  • Any map with quadratic maximum")
    print()
    
    print("Interesting numerical connections:")
    print()
    
    # 2^10 mod 55
    pow_2_10 = 2**10
    mod_55 = pow_2_10 % 55
    print(f"  2^10 = {pow_2_10}")
    print(f"  2^10 mod 55 = {mod_55} = F₉")
    print()
    
    # Period of 2^n mod 55
    print("  Period of 2^n mod 55:")
    powers = []
    for n in range(1, 25):
        powers.append(2**n % 55)
    
    # Find period
    for period in range(1, 25):
        if powers[period:period+period] == powers[:period]:
            print(f"    Period = {period}")
            break
    print(f"    Sequence: {powers[:12]}...")
    print()
    
    # Golden angle
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi / phi**2
    
    print(f"Golden angle connection:")
    print(f"  φ = {phi:.15f}")
    print(f"  Golden angle = 2π/φ² = {golden_angle:.10f} rad = {np.degrees(golden_angle):.6f}°")
    print()
    
    print(f"Fibonacci ratio convergence to φ:")
    for i in range(5, 11):
        ratio = FIBONACCI[i] / FIBONACCI[i-1]
        print(f"    F_{i}/F_{i-1} = {FIBONACCI[i]}/{FIBONACCI[i-1]} = {ratio:.10f}")
    print()
    
    print(f"At F₁₀ = 55:")
    print(f"  F₁₀/F₉ = 55/34 = {55/34:.10f}")
    print(f"  φ     = {phi:.10f}")
    print(f"  Error = {100 * abs(55/34 - phi) / phi:.6f}%")
    print()
    
    return {
        '2^10_mod_55': mod_55,
        'golden_angle_deg': np.degrees(golden_angle),
        'F10_over_F9': 55/34,
        'phi': phi,
        'ratio_error_percent': 100 * abs(55/34 - phi) / phi
    }


def print_theoretical_summary():
    """Print summary of theoretical insights."""
    print()
    print("=" * 75)
    print("THEORETICAL SUMMARY")
    print("=" * 75)
    print()
    
    print("KEY DISCOVERIES:")
    print()
    print("1. δ is a MÖBIUS TRANSFORMATION")
    print("   • Matrix has Fibonacci coefficients (3 = F₄, 5 = F₅)")
    print("   • Determinant = -26π = -2 × F₇ × π")
    print("   • Möbius transforms preserve cross-ratios")
    print("   • This suggests projective geometry underlies RG")
    print()
    
    print("2. 55 = F₁₀ IS FUNDAMENTAL")
    print("   • Appears in r∞ formula (base and correction)")
    print("   • Appears in δ formula (3575 = 55 × 65)")
    print("   • Appears in universal coefficient (a ≈ 55/36)")
    print("   • 2^10 mod 55 = 34 = F₉")
    print()
    
    print("3. ALL CONSTANTS ARE (RATIONAL) + O(π)")
    print("   • r∞: base has √17, correction has π⁴")
    print("   • δ:  base = 14/3, correction via (32π, 5π)")
    print("   • α:  base = 5/2, correction = π/1080")
    print()
    
    print("4. FERMAT NUMBERS ENCODE DOUBLING")
    print("   • 17 = 2⁴ + 1 (in r∞ under √)")
    print("   • 65 = 2⁶ + 1 (factor in δ)")
    print("   • 32 = 2⁵ (π coefficient in δ)")
    print()
    
    print("CONJECTURED UNIFYING PRINCIPLE:")
    print()
    print("  The Feigenbaum constants emerge from a projective")
    print("  (Möbius) structure on the renormalization group,")
    print("  with Fibonacci numbers encoding the golden-ratio")
    print("  attractor and Fermat numbers encoding the doubling")
    print("  cascade. The constant ξ = 1 + π/55 captures how")
    print("  circle geometry (π) interacts with Fibonacci (55)")
    print("  to produce universal chaos constants.")
    print()
    
    print("WHAT REMAINS OPEN:")
    print()
    print("  • Derivation of 55 = F₁₀ from RG fixed point equation")
    print("  • Proof that δ must be a Möbius transformation")
    print("  • Explanation of why 17 and not other Fermat numbers")
    print("  • Connection between correction term and RG perturbation")
    print()


def save_results(results):
    """Save analysis results to JSON."""
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_08_renormalization_analysis_{timestamp}.json'
    filepath = results_dir / filename
    
    with open(filepath, 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


def main():
    """Main analysis routine."""
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                           ║")
    print("║     RENORMALIZATION GROUP THEORY - STRUCTURAL ANALYSIS                    ║")
    print("║                                                                           ║")
    print("║     Exploring theoretical connections between Feigenbaum closed forms     ║")
    print("║     and renormalization group theory                                      ║")
    print("║                                                                           ║")
    print("║     Date: 2026-01-06                                                      ║")
    print("║     Status: THEORETICAL EXPLORATION                                       ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    
    results = {}
    
    # Run analyses
    results['mobius'] = analyze_delta_mobius()
    results['coefficients'] = analyze_coefficient_structure()
    results['universal_coefficient'] = analyze_universal_coefficient()
    results['perturbation'] = analyze_perturbation_structure()
    results['circle_map'] = analyze_circle_map_connection()
    
    # Summary
    print_theoretical_summary()
    
    # Save results
    print("SAVING RESULTS:")
    print("-" * 75)
    results['timestamp'] = datetime.now().isoformat()
    results['status'] = 'Theoretical exploration - patterns found, derivation open'
    save_results(results)
    print()
    
    return results


if __name__ == "__main__":
    main()
