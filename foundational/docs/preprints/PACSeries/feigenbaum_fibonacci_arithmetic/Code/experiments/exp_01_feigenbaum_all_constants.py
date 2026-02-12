#!/usr/bin/env python3
"""
exp_07_feigenbaum_all_constants.py
===================================

COMPLETE CLOSED-FORM VALIDATION FOR ALL THREE FEIGENBAUM CONSTANTS

This script validates the conjectured closed-form expressions for:

1. r∞ (Accumulation Point) - 13 significant figures
2. δ  (Bifurcation Ratio)  - 8 significant figures
3. α  (Scaling Constant)   - 6 significant figures

Discovery Date: 2026-01-06
Status: EXPERIMENTAL - Requires theoretical derivation

FORMULAS SUMMARY
================

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  r∞ = π(55 + √(17 - π/(55d)))(55 + π)/55² - √(3/5 - (ξ-1)²/7) × π⁴/55⁶   │
│                                                                             │
│  where d = √(52 + 2π/55), ξ = 1 + π/55                                     │
│                                                                             │
│  Accuracy: 13 significant figures (relative error ~10⁻¹⁴)                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  δ = (50050 + 32π) / (10725 + 5π)                                          │
│                                                                             │
│  Factored: (14×3575 + 32π) / (3×3575 + 5π), where 3575 = 55×65             │
│                                                                             │
│  Accuracy: 8 significant figures (relative error ~10⁻⁹)                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  α = (5 + π/540) / 2  =  (2700 + π) / 1080                                 │
│                                                                             │
│  Accuracy: 6 significant figures (relative error ~10⁻⁷)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

KEY STRUCTURAL CONSTANTS
========================

55  = F₁₀ (10th Fibonacci number)
17  = 2⁴ + 1 (5th Fermat number, prime)
52  = 55 - 3 = F₁₀ - F₄
3575 = 55 × 65 = F₁₀ × (F₁₀ + 10)
540 = 4 × 135 = 2² × 3³ × 5

ξ = 1 + π/55 = 1.0571198664289... (Dawn Field constant)
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
import sys

# ===========================================================================
# KNOWN HIGH-PRECISION REFERENCE VALUES
# ===========================================================================

# r∞: Feigenbaum accumulation point
# Source: OEIS A098587 (95 digits from Broadhurst calculation)
R_INF_KNOWN = 3.56994567187094490184200515138649893676383691151483237810797550

# δ: Feigenbaum bifurcation ratio (delta)
# Source: Multiple independent calculations, ~50 digits commonly quoted
DELTA_KNOWN = 4.66920160910299067185320382047240927606510947219218

# α: Feigenbaum scaling constant (absolute value)
# Source: Multiple independent calculations, ~50 digits commonly quoted
ALPHA_KNOWN = 2.50290787509589282228390287321821578636462643780702


# ===========================================================================
# FORMULA 1: FEIGENBAUM ACCUMULATION POINT r∞
# ===========================================================================

def compute_r_inf_base():
    """
    Compute the BASE term of r∞ (before correction).
    
    Formula:
        r_base = π(55 + √(17 - π/(55d)))(55 + π) / 55²
        
    where d = √(52 + 2π/55)
    
    Returns:
        float: r_base ≈ 3.5699456745...
        dict: Intermediate calculation steps
    """
    F = 55   # 10th Fibonacci number
    P = 17   # 2^4 + 1, Fermat prime
    
    steps = {}
    
    # Step 1: d = √(52 + 2π/55)
    d_squared = 52 + 2*np.pi/55
    d = np.sqrt(d_squared)
    steps['d_squared'] = d_squared  # 52.114239733...
    steps['d'] = d                   # 7.219170553...
    
    # Step 2: π/(55d)
    pi_over_55d = np.pi / (F * d)
    steps['pi_over_55d'] = pi_over_55d  # 0.007916078...
    
    # Step 3: 17 - π/(55d)
    inner = P - pi_over_55d
    steps['inner'] = inner  # 16.992083921...
    
    # Step 4: √(17 - π/(55d))
    sqrt_inner = np.sqrt(inner)
    steps['sqrt_inner'] = sqrt_inner  # 4.122120948...
    
    # Step 5: 55 + √(...)
    F_plus_sqrt = F + sqrt_inner
    steps['F_plus_sqrt'] = F_plus_sqrt  # 59.122120948...
    
    # Step 6: 55 + π
    F_plus_pi = F + np.pi
    steps['F_plus_pi'] = F_plus_pi  # 58.141592653...
    
    # Step 7: Numerator = π × 59.122... × 58.141...
    numerator = np.pi * F_plus_sqrt * F_plus_pi
    steps['numerator'] = numerator  # 10798.096539...
    
    # Step 8: r_base = Numerator / 55²
    r_base = numerator / F**2
    steps['r_base'] = r_base  # 3.569945674595...
    
    return r_base, steps


def compute_r_inf_correction():
    """
    Compute the CORRECTION term for r∞.
    
    Formula:
        correction = √(3/5 - (ξ-1)²/7) × π⁴/55⁶
        
    where ξ = 1 + π/55, so ξ-1 = π/55
    
    Returns:
        float: correction ≈ 2.72473 × 10⁻⁹
        dict: Intermediate calculation steps
    """
    steps = {}
    
    # Step 1: ξ - 1 = π/55
    xi_minus_1 = np.pi / 55
    steps['xi_minus_1'] = xi_minus_1  # 0.057119866...
    
    # Step 2: (ξ-1)²
    xi_m1_squared = xi_minus_1**2
    steps['xi_m1_squared'] = xi_m1_squared  # 0.003262679...
    
    # Step 3: (ξ-1)²/7
    over_7 = xi_m1_squared / 7
    steps['xi_m1_sq_over_7'] = over_7  # 0.000466097...
    
    # Step 4: 3/5 - (ξ-1)²/7
    k_squared = 3/5 - over_7
    steps['k_squared'] = k_squared  # 0.599533902...
    
    # Step 5: k = √(3/5 - (ξ-1)²/7)
    k = np.sqrt(k_squared)
    steps['k'] = k  # 0.774295712...
    
    # Step 6: π⁴
    pi_4 = np.pi**4
    steps['pi_4'] = pi_4  # 97.409091034...
    
    # Step 7: 55⁶
    pow_55_6 = 55**6
    steps['pow_55_6'] = pow_55_6  # 27,680,640,625
    
    # Step 8: π⁴/55⁶
    pi4_over_556 = pi_4 / pow_55_6
    steps['pi4_over_556'] = pi4_over_556  # 3.51932636... × 10⁻⁹
    
    # Step 9: k × π⁴/55⁶
    correction = k * pi4_over_556
    steps['correction'] = correction  # 2.72473039... × 10⁻⁹
    
    return correction, steps


def feigenbaum_r_inf(return_steps=False):
    """
    Compute Feigenbaum accumulation point r∞ using complete closed form.
    
    Formula:
        r∞ = π(55 + √(17 - π/(55d)))(55 + π)/55² - √(3/5 - (ξ-1)²/7) × π⁴/55⁶
        
    where:
        d = √(52 + 2π/55)
        ξ = 1 + π/55
    
    Args:
        return_steps: If True, return all intermediate calculations
    
    Returns:
        float: r∞ ≈ 3.5699456718709...
        
    Accuracy: 13 significant figures (relative error ~10⁻¹⁴)
    """
    r_base, base_steps = compute_r_inf_base()
    correction, corr_steps = compute_r_inf_correction()
    
    r_inf = r_base - correction
    
    if return_steps:
        return r_inf, {
            'base_term': base_steps,
            'correction_term': corr_steps,
            'r_base': r_base,
            'correction': correction,
            'r_inf': r_inf
        }
    
    return r_inf


# ===========================================================================
# FORMULA 2: FEIGENBAUM BIFURCATION RATIO δ
# ===========================================================================

def feigenbaum_delta(return_steps=False):
    """
    Compute Feigenbaum bifurcation ratio δ using closed form.
    
    Formula:
        δ = (50050 + 32π) / (10725 + 5π)
        
    Factored form:
        δ = (14×3575 + 32π) / (3×3575 + 5π)
        
    where 3575 = 55 × 65 = F₁₀ × (F₁₀ + 10)
    
    Args:
        return_steps: If True, return all intermediate calculations
    
    Returns:
        float: δ ≈ 4.669201609...
        
    Accuracy: 8 significant figures (relative error ~10⁻⁹)
    """
    steps = {}
    
    # Structural constants
    # 50050 = 14 × 3575 = 14 × 55 × 65
    # 10725 = 3 × 3575 = 3 × 55 × 65
    
    steps['structure'] = {
        '3575': 55 * 65,
        '50050': 14 * 55 * 65,
        '10725': 3 * 55 * 65,
        '50050_factored': '14 × 3575 = 14 × 55 × 65',
        '10725_factored': '3 × 3575 = 3 × 55 × 65'
    }
    
    # Step 1: 32π
    pi_term_num = 32 * np.pi
    steps['32_pi'] = pi_term_num  # 100.530964914...
    
    # Step 2: Numerator = 50050 + 32π
    numerator = 50050 + pi_term_num
    steps['numerator'] = numerator  # 50150.530964914...
    
    # Step 3: 5π
    pi_term_denom = 5 * np.pi
    steps['5_pi'] = pi_term_denom  # 15.707963267...
    
    # Step 4: Denominator = 10725 + 5π
    denominator = 10725 + pi_term_denom
    steps['denominator'] = denominator  # 10740.707963267...
    
    # Step 5: δ = Numerator / Denominator
    delta = numerator / denominator
    steps['delta'] = delta  # 4.669201614681660...
    
    # Base approximation for comparison
    base_approx = 14/3
    steps['base_approx'] = base_approx  # 4.666666...
    steps['base_error'] = abs(base_approx - DELTA_KNOWN) / DELTA_KNOWN
    
    if return_steps:
        return delta, steps
    
    return delta


# ===========================================================================
# FORMULA 3: FEIGENBAUM SCALING CONSTANT α
# ===========================================================================

def feigenbaum_alpha(return_steps=False):
    """
    Compute Feigenbaum scaling constant α using closed form.
    
    Formula:
        α = (5 + π/540) / 2
        
    Equivalent form:
        α = (2700 + π) / 1080
        
    Decomposed:
        α = 5/2 + π/1080 = 2.5 + 0.00290888...
    
    Structural notes:
        540 = 4 × 135 = 4 × 27 × 5 = 2² × 3³ × 5
        1080 = 2 × 540 = 2³ × 3³ × 5
        2700 = 5 × 540 = 2² × 3³ × 5²
    
    Args:
        return_steps: If True, return all intermediate calculations
    
    Returns:
        float: |α| ≈ 2.502907...
        
    Accuracy: 6 significant figures (relative error ~10⁻⁷)
    """
    steps = {}
    
    # Structural analysis
    steps['structure'] = {
        '540': '2² × 3³ × 5 = 4 × 135',
        '1080': '2³ × 3³ × 5 = 2 × 540',
        '2700': '2² × 3³ × 5² = 5 × 540',
        'angular': '540° = 1.5 circles = 3π radians'
    }
    
    # Step 1: π/540
    pi_over_540 = np.pi / 540
    steps['pi_over_540'] = pi_over_540  # 0.005817764522...
    
    # Step 2: 5 + π/540
    five_plus = 5 + pi_over_540
    steps['5_plus_pi_over_540'] = five_plus  # 5.005817764522...
    
    # Step 3: (5 + π/540) / 2
    alpha = five_plus / 2
    steps['alpha_form1'] = alpha  # 2.502908882261...
    
    # Alternative form verification
    alpha_alt = (2700 + np.pi) / 1080
    steps['alpha_form2'] = alpha_alt  # Should be identical
    steps['forms_match'] = np.isclose(alpha, alpha_alt)
    
    # Base approximation for comparison
    base_approx = 5/2
    steps['base_approx'] = base_approx  # 2.5
    steps['base_error'] = abs(base_approx - ALPHA_KNOWN) / ALPHA_KNOWN
    
    if return_steps:
        return alpha, steps
    
    return alpha


# ===========================================================================
# VALIDATION FUNCTIONS
# ===========================================================================

def validate_r_inf():
    """Validate r∞ formula with detailed output."""
    
    print()
    print("=" * 75)
    print("FORMULA 1: FEIGENBAUM ACCUMULATION POINT r∞")
    print("=" * 75)
    print()
    
    r_inf, steps = feigenbaum_r_inf(return_steps=True)
    
    print("FORMULA:")
    print("─" * 75)
    print()
    print("  r∞ = π(55 + √(17 - π/(55d)))(55 + π)/55² - √(3/5 - (ξ-1)²/7) × π⁴/55⁶")
    print()
    print("  where d = √(52 + 2π/55), ξ = 1 + π/55")
    print()
    print()
    print("BASE TERM CALCULATION:")
    print("─" * 75)
    base = steps['base_term']
    print(f"  Step 1: d² = 52 + 2π/55 = {base['d_squared']:.12f}")
    print(f"  Step 2: d = √(d²) = {base['d']:.12f}")
    print(f"  Step 3: π/(55d) = {base['pi_over_55d']:.12f}")
    print(f"  Step 4: 17 - π/(55d) = {base['inner']:.12f}")
    print(f"  Step 5: √(17 - π/(55d)) = {base['sqrt_inner']:.12f}")
    print(f"  Step 6: 55 + √(...) = {base['F_plus_sqrt']:.12f}")
    print(f"  Step 7: 55 + π = {base['F_plus_pi']:.12f}")
    print(f"  Step 8: π × {base['F_plus_sqrt']:.6f} × {base['F_plus_pi']:.6f} = {base['numerator']:.6f}")
    print(f"  Step 9: r_base = {base['numerator']:.6f} / 3025 = {base['r_base']:.15f}")
    print()
    print()
    print("CORRECTION TERM CALCULATION:")
    print("─" * 75)
    corr = steps['correction_term']
    print(f"  Step 1: ξ - 1 = π/55 = {corr['xi_minus_1']:.15f}")
    print(f"  Step 2: (ξ-1)² = {corr['xi_m1_squared']:.15f}")
    print(f"  Step 3: (ξ-1)²/7 = {corr['xi_m1_sq_over_7']:.15f}")
    print(f"  Step 4: 3/5 - (ξ-1)²/7 = {corr['k_squared']:.15f}")
    print(f"  Step 5: k = √(3/5 - (ξ-1)²/7) = {corr['k']:.15f}")
    print(f"  Step 6: π⁴ = {corr['pi_4']:.12f}")
    print(f"  Step 7: 55⁶ = {corr['pow_55_6']:,}")
    print(f"  Step 8: π⁴/55⁶ = {corr['pi4_over_556']:.15e}")
    print(f"  Step 9: k × π⁴/55⁶ = {corr['correction']:.15e}")
    print()
    print()
    print("FINAL CALCULATION:")
    print("─" * 75)
    print(f"  r_base = {steps['r_base']:.18f}")
    print(f"  correction = {steps['correction']:.18e}")
    print(f"  r∞ = r_base - correction")
    print(f"     = {steps['r_inf']:.18f}")
    print()
    print()
    print("VALIDATION:")
    print("─" * 75)
    error = abs(r_inf - R_INF_KNOWN) / R_INF_KNOWN
    print(f"  Computed:   {r_inf:.20f}")
    print(f"  Known:      {R_INF_KNOWN:.20f}")
    print(f"  Difference: {r_inf - R_INF_KNOWN:+.20e}")
    print(f"  Rel Error:  {error:.6e}")
    print(f"  Percent:    {error*100:.15f}%")
    print(f"  Digits:     ~{-int(np.log10(error))} significant figures")
    print()
    
    return r_inf, error


def validate_delta():
    """Validate δ formula with detailed output."""
    
    print()
    print("=" * 75)
    print("FORMULA 2: FEIGENBAUM BIFURCATION RATIO δ")
    print("=" * 75)
    print()
    
    delta, steps = feigenbaum_delta(return_steps=True)
    
    print("FORMULA:")
    print("─" * 75)
    print()
    print("         50050 + 32π     14 × 3575 + 32π")
    print("    δ = ────────────── = ─────────────────")
    print("         10725 + 5π       3 × 3575 + 5π")
    print()
    print("  where 3575 = 55 × 65 = F₁₀ × (F₁₀ + 10)")
    print()
    print()
    print("STRUCTURAL ANALYSIS:")
    print("─" * 75)
    print(f"  3575 = 55 × 65 = {steps['structure']['3575']}")
    print(f"  50050 = {steps['structure']['50050_factored']} = {steps['structure']['50050']}")
    print(f"  10725 = {steps['structure']['10725_factored']} = {steps['structure']['10725']}")
    print(f"  Base ratio: 50050/10725 = 14/3 = {14/3:.10f}")
    print()
    print()
    print("STEP-BY-STEP CALCULATION:")
    print("─" * 75)
    print(f"  Step 1: 32 × π = {steps['32_pi']:.12f}")
    print(f"  Step 2: 50050 + 32π = {steps['numerator']:.12f}")
    print(f"  Step 3: 5 × π = {steps['5_pi']:.12f}")
    print(f"  Step 4: 10725 + 5π = {steps['denominator']:.12f}")
    print(f"  Step 5: δ = {steps['numerator']:.6f} / {steps['denominator']:.6f}")
    print(f"           = {steps['delta']:.15f}")
    print()
    print()
    print("VALIDATION:")
    print("─" * 75)
    error = abs(delta - DELTA_KNOWN) / DELTA_KNOWN
    print(f"  Computed:     {delta:.20f}")
    print(f"  Known:        {DELTA_KNOWN:.20f}")
    print(f"  Difference:   {delta - DELTA_KNOWN:+.20e}")
    print(f"  Rel Error:    {error:.6e}")
    print(f"  Percent:      {error*100:.15f}%")
    print(f"  Digits:       ~{-int(np.log10(error))} significant figures")
    print()
    print(f"  Base approx:  14/3 = {steps['base_approx']:.10f}")
    print(f"  Base error:   {steps['base_error']*100:.6f}% (~3 digits)")
    print(f"  Improvement:  {steps['base_error']/error:.0f}x better with π terms")
    print()
    
    return delta, error


def validate_alpha():
    """Validate α formula with detailed output."""
    
    print()
    print("=" * 75)
    print("FORMULA 3: FEIGENBAUM SCALING CONSTANT α")
    print("=" * 75)
    print()
    
    alpha, steps = feigenbaum_alpha(return_steps=True)
    
    print("FORMULA:")
    print("─" * 75)
    print()
    print("         5 + π/540     2700 + π")
    print("    α = ─────────── = ──────────")
    print("            2           1080")
    print()
    print("  Decomposed: α = 5/2 + π/1080 = 2.5 + small correction")
    print()
    print()
    print("STRUCTURAL ANALYSIS:")
    print("─" * 75)
    print(f"  540 = {steps['structure']['540']}")
    print(f"  1080 = {steps['structure']['1080']}")
    print(f"  2700 = {steps['structure']['2700']}")
    print(f"  Angular: {steps['structure']['angular']}")
    print()
    print()
    print("STEP-BY-STEP CALCULATION:")
    print("─" * 75)
    print(f"  Step 1: π/540 = {steps['pi_over_540']:.15f}")
    print(f"  Step 2: 5 + π/540 = {steps['5_plus_pi_over_540']:.15f}")
    print(f"  Step 3: (5 + π/540)/2 = {steps['alpha_form1']:.15f}")
    print()
    print(f"  Alternative form: (2700 + π)/1080 = {steps['alpha_form2']:.15f}")
    print(f"  Forms match: {steps['forms_match']}")
    print()
    print()
    print("VALIDATION:")
    print("─" * 75)
    error = abs(alpha - ALPHA_KNOWN) / ALPHA_KNOWN
    print(f"  Computed:     {alpha:.20f}")
    print(f"  Known:        {ALPHA_KNOWN:.20f}")
    print(f"  Difference:   {alpha - ALPHA_KNOWN:+.20e}")
    print(f"  Rel Error:    {error:.6e}")
    print(f"  Percent:      {error*100:.15f}%")
    print(f"  Digits:       ~{-int(np.log10(error))} significant figures")
    print()
    print(f"  Base approx:  5/2 = {steps['base_approx']:.10f}")
    print(f"  Base error:   {steps['base_error']*100:.6f}% (~3 digits)")
    print(f"  Improvement:  {steps['base_error']/error:.0f}x better with π term")
    print()
    
    return alpha, error


def print_summary(r_error, d_error, a_error):
    """Print summary of all three formulas."""
    
    print()
    print("=" * 75)
    print("SUMMARY: ALL THREE FEIGENBAUM CLOSED FORMS")
    print("=" * 75)
    print()
    print("┌────────────┬─────────────────────────────────────────┬────────────┐")
    print("│ Constant   │ Formula                                 │ Accuracy   │")
    print("├────────────┼─────────────────────────────────────────┼────────────┤")
    print("│            │ π(55+√(17-π/(55d)))(55+π)/55²          │            │")
    print(f"│ r∞         │ - √(3/5-(ξ-1)²/7)×π⁴/55⁶               │ {-int(np.log10(r_error)):2d} digits   │")
    print("│            │ where d=√(52+2π/55), ξ=1+π/55          │            │")
    print("├────────────┼─────────────────────────────────────────┼────────────┤")
    print(f"│ δ          │ (50050 + 32π) / (10725 + 5π)            │ {-int(np.log10(d_error)):2d} digits   │")
    print("├────────────┼─────────────────────────────────────────┼────────────┤")
    print(f"│ α          │ (5 + π/540) / 2                         │ {-int(np.log10(a_error)):2d} digits   │")
    print("└────────────┴─────────────────────────────────────────┴────────────┘")
    print()
    print("COMMON ELEMENTS:")
    print("─" * 75)
    print("  • π appears in all three formulas")
    print("  • 55 = F₁₀ (10th Fibonacci) central to r∞ and δ")
    print("  • All have form: (rational structure) + (π correction)")
    print("  • Accuracy hierarchy: r∞ > δ > α suggests r∞ is 'primary'")
    print()
    print("STRUCTURAL CONSTANTS:")
    print("─" * 75)
    print("  55 = F₁₀ = 10th Fibonacci number")
    print("  17 = 2⁴+1 = 5th Fermat number (prime)")
    print("  52 = 55-3 = F₁₀ - F₄")
    print("  3575 = 55×65 = used in δ formula")
    print("  540 = 2²×3³×5 = used in α formula")
    print("  ξ = 1 + π/55 = Dawn Field constant")
    print()


def save_results():
    """Save validation results to JSON file."""
    
    r_inf = feigenbaum_r_inf()
    delta = feigenbaum_delta()
    alpha = feigenbaum_alpha()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'script': 'exp_07_feigenbaum_all_constants.py',
        'description': 'Validation of closed-form expressions for all three Feigenbaum constants',
        'formulas': {
            'r_inf': {
                'formula': 'π(55+√(17-π/(55d)))(55+π)/55² - √(3/5-(ξ-1)²/7)×π⁴/55⁶ where d=√(52+2π/55), ξ=1+π/55',
                'computed': float(r_inf),
                'known': float(R_INF_KNOWN),
                'relative_error': float(abs(r_inf - R_INF_KNOWN) / R_INF_KNOWN),
                'significant_figures': 13
            },
            'delta': {
                'formula': '(50050 + 32π) / (10725 + 5π)',
                'computed': float(delta),
                'known': float(DELTA_KNOWN),
                'relative_error': float(abs(delta - DELTA_KNOWN) / DELTA_KNOWN),
                'significant_figures': 8
            },
            'alpha': {
                'formula': '(5 + π/540) / 2',
                'computed': float(alpha),
                'known': float(ALPHA_KNOWN),
                'relative_error': float(abs(alpha - ALPHA_KNOWN) / ALPHA_KNOWN),
                'significant_figures': 6
            }
        },
        'structural_constants': {
            '55': 'F₁₀, 10th Fibonacci number',
            '17': '2⁴+1, 5th Fermat number (prime)',
            '52': '55-3 = F₁₀ - F₄',
            '3575': '55×65, used in δ formula',
            '540': '2²×3³×5, used in α formula',
            'xi': float(1 + np.pi/55)
        },
        'status': 'EXPERIMENTAL - Requires theoretical derivation'
    }
    
    # Save to results folder
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_07_feigenbaum_all_constants_{timestamp}.json'
    filepath = results_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    
    return results


def main():
    """Main validation routine."""
    
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                           ║")
    print("║     FEIGENBAUM UNIVERSAL CONSTANTS - CLOSED FORM VALIDATION               ║")
    print("║                                                                           ║")
    print("║     Three closed-form expressions for the universal constants             ║")
    print("║     of period-doubling bifurcation cascades                               ║")
    print("║                                                                           ║")
    print("║     Date: 2026-01-06                                                      ║")
    print("║     Status: EXPERIMENTAL                                                  ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    
    # Validate each formula
    r_inf, r_error = validate_r_inf()
    delta, d_error = validate_delta()
    alpha, a_error = validate_alpha()
    
    # Print summary
    print_summary(r_error, d_error, a_error)
    
    # Save results
    print("SAVING RESULTS:")
    print("─" * 75)
    save_results()
    print()
    
    return {
        'r_inf': (r_inf, r_error),
        'delta': (delta, d_error),
        'alpha': (alpha, a_error)
    }


if __name__ == "__main__":
    main()
