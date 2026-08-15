#!/usr/bin/env python3
"""
exp_23_xi_derivation_contest.py - Which formula for Ξ is fundamental?

THREE COMPETING HYPOTHESES FOR Ξ:

1. Ξ = 1 + π/55 (oscillation_attractor_dynamics)
   - Derived from PAC collapse dynamics
   - Each recursion level adds π/55 Möbius twist
   - At depth 55 (F₁₀): 55 × π/55 = π (half-twist)
   - Source: within = -0.0283, cross = +0.0854, net = 0.0571

2. Ξ = γ + ln(φ) (prime smoothing / algebra-geometry interface)
   - γ = Euler-Mascheroni = algebra-geometry translation
   - ln(φ) = logarithmic growth rate of PAC recursion
   - Would unify number theory with DFT

3. BOTH are approximations to something deeper

This experiment:
- Re-derives Ξ from the original within/cross level mechanism
- Tests γ + ln(φ) in multiple independent contexts
- Builds falsification tests
- Determines which (if either) is fundamental
"""

import numpy as np
from datetime import datetime
import json
import os

# Full-precision constants
GAMMA = 0.5772156649015329  # Euler-Mascheroni
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
INV_PHI = 1 / PHI

# The two candidate formulas
XI_FROM_PI = 1 + np.pi / 55  # = 1.05711986898...
XI_FROM_GAMMA_PHI = GAMMA + np.log(PHI)  # = 1.05842749...

# ============================================================================
# PART 1: Re-derive Ξ from PAC collapse dynamics
# ============================================================================

def derive_xi_from_pac_collapse():
    """
    Re-derive Ξ from the original mechanism found in oscillation_attractor_dynamics.
    
    The derivation (from exp_01_define_metrics.py):
    - Within-level contribution: 2√(r(1-r)) - 1 where r = 1/φ
    - Cross-level interference: +0.0854 (amplification)
    - Net = -0.0283 + 0.0854 = 0.0571 ≈ π/55
    """
    print("=" * 70)
    print("PART 1: RE-DERIVE Ξ FROM PAC COLLAPSE DYNAMICS")
    print("=" * 70)
    
    r = INV_PHI  # The PAC ratio
    
    # Within-level contribution (coherence reduction)
    within_raw = 2 * np.sqrt(r * (1 - r)) - 1
    
    print(f"\n  r = 1/φ = {r:.10f}")
    print(f"\n  Within-level formula: 2√(r(1-r)) - 1")
    print(f"    = 2√({r:.6f} × {1-r:.6f}) - 1")
    print(f"    = 2√{r*(1-r):.6f} - 1")
    print(f"    = 2 × {np.sqrt(r*(1-r)):.6f} - 1")
    print(f"    = {within_raw:.6f}")
    
    # What should the cross-level contribution be?
    # If net = π/55, then cross = π/55 - within
    target_net = np.pi / 55
    implied_cross = target_net - within_raw
    
    print(f"\n  If net = π/55 = {target_net:.6f}:")
    print(f"    Required cross-level = {implied_cross:.6f}")
    
    # Check what the documented values were
    doc_within = -0.0283
    doc_cross = 0.0854
    doc_net = doc_within + doc_cross
    
    print(f"\n  Documented values (from oscillation_attractor_dynamics):")
    print(f"    within = {doc_within}")
    print(f"    cross = {doc_cross}")
    print(f"    net = {doc_net:.6f}")
    print(f"    π/55 = {np.pi/55:.6f}")
    print(f"    Difference: {abs(doc_net - np.pi/55):.6f}")
    
    # Now test: does γ + ln(φ) - 1 match the net?
    gamma_phi_net = GAMMA + np.log(PHI) - 1
    
    print(f"\n  Alternative: (γ + ln(φ)) - 1 = {gamma_phi_net:.6f}")
    print(f"  Difference from π/55: {abs(gamma_phi_net - np.pi/55):.6f}")
    print(f"  Difference from doc_net: {abs(gamma_phi_net - doc_net):.6f}")
    
    return {
        'within_level': float(within_raw),
        'implied_cross': float(implied_cross),
        'doc_within': doc_within,
        'doc_cross': doc_cross,
        'doc_net': float(doc_net),
        'pi_over_55': float(np.pi/55),
        'gamma_phi_minus_1': float(gamma_phi_net),
        'doc_net_vs_pi_55_error': float(abs(doc_net - np.pi/55)),
        'gamma_phi_vs_pi_55_error': float(abs(gamma_phi_net - np.pi/55))
    }

# ============================================================================
# PART 2: Test γ + ln(φ) in independent contexts
# ============================================================================

def test_gamma_lnphi_independently():
    """
    Look for γ + ln(φ) appearing independently from Ξ.
    If it appears elsewhere, it's fundamental. If not, coincidence.
    """
    print("\n" + "=" * 70)
    print("PART 2: TEST γ + ln(φ) IN INDEPENDENT CONTEXTS")
    print("=" * 70)
    
    target = GAMMA + np.log(PHI)
    
    print(f"\n  Target: γ + ln(φ) = {target:.10f}")
    
    tests = {}
    
    # Test 1: In prime counting
    # li(x) - π(x) ~ √x ln(x). Does γ + ln(φ) appear?
    print(f"\n  TEST 1: Prime counting error ratios")
    
    # Chebyshev function ratio
    # ψ(x)/x → 1 with error ~ 1/ln(x)
    # Check if γ + ln(φ) appears in corrections
    
    # At x=10000, ln(10000) ≈ 9.21
    # 1/(γ + ln(φ)) = 0.945
    # 1/ln(10000) = 0.109
    # Ratio: 0.109/0.109 = 1 (not useful)
    
    # Try: ln(γ + ln(φ)) = 0.0568
    # This is surprisingly close to π/55 - 1 ≈ 0.0571 !
    
    ln_target = np.log(target)
    print(f"    ln(γ + ln(φ)) = {ln_target:.6f}")
    print(f"    π/55 = {np.pi/55:.6f}")
    print(f"    Difference: {abs(ln_target - np.pi/55):.6f}")
    
    # Wait - this suggests:
    # ln(Ξ) ≈ π/55  which means Ξ ≈ e^(π/55) ≈ 1.0588
    exp_pi_55 = np.exp(np.pi/55)
    print(f"\n    If ln(Ξ) = π/55:")
    print(f"    Then Ξ = e^(π/55) = {exp_pi_55:.6f}")
    print(f"    Compare to 1 + π/55 = {XI_FROM_PI:.6f}")
    print(f"    Compare to γ + ln(φ) = {XI_FROM_GAMMA_PHI:.6f}")
    
    tests['ln_of_target_vs_pi_55'] = {
        'ln_target': float(ln_target),
        'pi_55': float(np.pi/55),
        'error': float(abs(ln_target - np.pi/55)),
        'relative_error': float(abs(ln_target - np.pi/55) / (np.pi/55))
    }
    
    # Test 2: In Riemann zeta special values
    print(f"\n  TEST 2: Riemann zeta special values")
    
    # ζ(2) = π²/6 ≈ 1.6449
    # ζ(3) ≈ 1.2021
    # ζ(4) = π⁴/90 ≈ 1.0823
    
    zeta_2 = np.pi**2 / 6
    zeta_4 = np.pi**4 / 90
    
    # Check ratios
    print(f"    ζ(2) = {zeta_2:.6f}")
    print(f"    ζ(4) = {zeta_4:.6f}")
    print(f"    ζ(2)/φ = {zeta_2/PHI:.6f}")
    print(f"    Target = {target:.6f}")
    print(f"    ζ(4)/γ+ln(φ) = {zeta_4/target:.6f}")
    
    tests['zeta_ratios'] = {
        'zeta_2_over_phi': float(zeta_2/PHI),
        'zeta_4_over_target': float(zeta_4/target)
    }
    
    # Test 3: In Fibonacci growth
    print(f"\n  TEST 3: Fibonacci asymptotic")
    
    # F_n ≈ φ^n / √5
    # ln(F_n) ≈ n ln(φ) - ln(√5)/2
    # So ln(φ) is the growth rate of Fibonacci
    
    # What about γ + ln(φ)?
    # This would be: Mertens interference + Fibonacci growth
    
    # Test: at what n does F_n / e^(γn) reach steady state?
    fib = [1, 1]
    for i in range(50):
        fib.append(fib[-1] + fib[-2])
    
    ratios = []
    for n in range(5, 50):
        ratio = np.log(fib[n]) / n
        ratios.append(ratio)
    
    asymptotic_growth = ratios[-1]
    print(f"    Asymptotic: ln(F_n)/n → {asymptotic_growth:.6f}")
    print(f"    ln(φ) = {np.log(PHI):.6f}")
    print(f"    γ + ln(φ) = {target:.6f}")
    
    # Hmm, the growth rate is just ln(φ), not γ + ln(φ)
    # But what if we look at something involving both?
    
    tests['fibonacci_growth'] = {
        'ln_phi': float(np.log(PHI)),
        'asymptotic_ln_Fn_over_n': float(asymptotic_growth)
    }
    
    # Test 4: Harmonic-Fibonacci hybrid
    print(f"\n  TEST 4: Harmonic-Fibonacci hybrid")
    
    # Σ 1/F_n converges (slowly)
    # The limit is approximately 3.36...
    fib_harmonic = sum(1/f for f in fib[1:40])  # avoid F_0=0
    print(f"    Σ 1/F_n (n=1 to 40) ≈ {fib_harmonic:.6f}")
    
    # Product formula?
    # ∏ F_n/F_{n+1} → 0 (trivially)
    # But ∏ (1 + 1/F_n) might be interesting
    fib_prod = np.prod([1 + 1/f for f in fib[1:20]])
    print(f"    ∏ (1 + 1/F_n) (n=1 to 20) ≈ {fib_prod:.6f}")
    
    tests['fibonacci_series'] = {
        'fib_harmonic': float(fib_harmonic),
        'fib_product': float(fib_prod)
    }
    
    # Test 5: The 55 connection
    print(f"\n  TEST 5: Why 55? (F₁₀)")
    
    # 55 is Fibonacci. The interpretation was:
    # Each level adds π/55 of a twist
    # At depth 55, you get exactly π (one Möbius half-twist)
    
    # But from γ + ln(φ): why would F₁₀ be special?
    # F₁₀ = 55, and 55 = 5 × 11
    # Also: ln(55) = 4.007...
    
    print(f"    F₁₀ = 55")
    print(f"    55 = 5 × 11")
    print(f"    ln(55) = {np.log(55):.6f}")
    print(f"    55 × (γ + ln(φ) - 1) = {55 * (target - 1):.6f}")
    print(f"    55 × π/55 = π = {np.pi:.6f}")
    print(f"    Compare: {55 * (target - 1):.6f} vs π = {np.pi:.6f}")
    print(f"    Ratio: {55 * (target - 1) / np.pi:.6f}")
    
    # Interesting: 55(γ + ln(φ) - 1) ≈ 3.21, not π
    # So the π/55 formula doesn't come from γ + ln(φ) directly
    
    tests['55_connection'] = {
        '55_times_gamma_phi_minus_1': float(55 * (target - 1)),
        '55_times_pi_55': float(np.pi),
        'ratio': float(55 * (target - 1) / np.pi)
    }
    
    return tests

# ============================================================================
# PART 3: Derive Ξ from first principles
# ============================================================================

def derive_xi_first_principles():
    """
    Try to derive Ξ without assuming either formula.
    
    What we know:
    - Ξ appears where SEC dynamics meet PAC structure
    - Ξ - 1 encodes "twist per recursion level"
    - At some critical depth d, d × (Ξ-1) = π
    
    Question: What determines d?
    """
    print("\n" + "=" * 70)
    print("PART 3: FIRST PRINCIPLES DERIVATION")
    print("=" * 70)
    
    results = {}
    
    # Approach 1: From PAC balance ratio
    print("\n  Approach 1: From PAC balance ratio r = 1/φ")
    
    r = INV_PHI
    
    # The within-level formula 2√(r(1-r)) - 1 gives the "coherence reduction"
    within = 2 * np.sqrt(r * (1 - r)) - 1
    
    # What's the geometric meaning?
    # 2√(r(1-r)) is the geometric mean formula
    # For r = 1/φ: r(1-r) = (1/φ)(1 - 1/φ) = (1/φ)(1/φ²) = 1/φ³
    # So √(r(1-r)) = 1/φ^(3/2) = φ^(-3/2)
    # 2φ^(-3/2) = 2/φ^(3/2) ≈ 0.9442
    # 2/φ^(3/2) - 1 = 1/φ^(3/2) - (1 - 1/φ^(3/2)) ≈ -0.0558
    
    r_times_1_minus_r = r * (1 - r)
    sqrt_term = np.sqrt(r_times_1_minus_r)
    
    print(f"    r(1-r) = 1/φ³ = {1/PHI**3:.6f} (exact: {r_times_1_minus_r:.6f})")
    print(f"    √(r(1-r)) = φ^(-3/2) = {PHI**(-1.5):.6f}")
    print(f"    2√(r(1-r)) - 1 = {within:.6f}")
    
    # The cross-level must compensate and add π/55
    # What's the geometric source of cross-level?
    
    # Hypothesis: Cross-level comes from interference between levels
    # Interference ~ (1/φ)^depth × (1/φ)^(depth+1) summed
    # = sum of (1/φ)^(2d+1) for d = 0, 1, 2...
    # = (1/φ) / (1 - 1/φ²) = (1/φ) × φ² / (φ² - 1) = φ / (φ²-1) = φ/φ = 1
    
    # That's too simple. Let me think differently.
    
    # Cross-level as resonance:
    # Each level n interferes with level n+1
    # Contribution: sin²(2πn/d) where d is critical depth
    
    print(f"\n  If cross-level = 0.0854 is required:")
    net_required = np.pi / 55
    cross_required = net_required - within  # Should be positive
    print(f"    within = {within:.6f}")
    print(f"    net = π/55 = {net_required:.6f}")
    print(f"    cross required = {cross_required:.6f}")
    
    # Check: is cross_required related to any simple formula?
    print(f"\n    Cross required / (1/φ²) = {cross_required / (1/PHI**2):.6f}")
    print(f"    Cross required × φ = {cross_required * PHI:.6f}")
    print(f"    Cross required / γ = {cross_required / GAMMA:.6f}")
    print(f"    Cross required / ln(φ) = {cross_required / np.log(PHI):.6f}")
    
    # Hmm, cross / ln(φ) ≈ 0.178
    # cross / (1/φ²) ≈ 0.221
    
    results['within_level'] = float(within)
    results['cross_required'] = float(cross_required)
    
    # Approach 2: From Möbius topology
    print("\n  Approach 2: From Möbius topology")
    
    # On a Möbius strip, circumnavigating twice returns to start
    # One circumnavigation = 2π
    # But Möbius gives 1/2 twist per 2π → π per circumnavigation
    
    # If the "twist per level" is Ξ - 1, and d levels gives π:
    # d × (Ξ - 1) = π
    # d = π / (Ξ - 1)
    
    # For Ξ = 1 + π/55: d = π / (π/55) = 55 ✓
    # For Ξ = γ + ln(φ): d = π / (γ + ln(φ) - 1) = π / 0.0584 ≈ 53.8
    
    d_from_pi_55 = np.pi / (np.pi/55)
    d_from_gamma_phi = np.pi / (XI_FROM_GAMMA_PHI - 1)
    
    print(f"    Critical depth for Ξ = 1 + π/55: {d_from_pi_55:.1f}")
    print(f"    Critical depth for Ξ = γ + ln(φ): {d_from_gamma_phi:.1f}")
    print(f"    F₁₀ = 55")
    print(f"    F₉ = 34")
    
    # 55 is EXACTLY right for π/55
    # 53.8 is close to 55 but not exact
    
    results['critical_depth_pi_55'] = float(d_from_pi_55)
    results['critical_depth_gamma_phi'] = float(d_from_gamma_phi)
    
    # Approach 3: Why must depth be Fibonacci?
    print("\n  Approach 3: Why Fibonacci depth?")
    
    # PAC creates Fibonacci naturally
    # The Möbius half-twist (π) must occur at Fibonacci depth
    # Closest Fibonacci to 53.8 is 55
    
    # This suggests: Ξ = 1 + π/F₁₀ is the TRUE formula
    # And γ + ln(φ) ≈ 1 + π/55 is the approximation
    
    fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    print(f"\n    Fibonacci sequence: {fibs}")
    print(f"\n    Testing: Ξ = 1 + π/F_n")
    
    for i, f in enumerate(fibs[5:], start=5):
        xi_test = 1 + np.pi / f
        depth_check = np.pi / (xi_test - 1)
        error_vs_gamma_phi = abs(xi_test - XI_FROM_GAMMA_PHI)
        print(f"      F_{i} = {f}: Ξ = {xi_test:.6f}, depth = {depth_check:.1f}, "
              f"error vs γ+ln(φ) = {error_vs_gamma_phi:.6f}")
    
    # F₁₀ = 55 gives the closest match!
    
    results['fibonacci_depths'] = {
        f'F_{i}': {'fib': f, 'xi': float(1 + np.pi/f), 'error_vs_gamma_phi': float(abs(1 + np.pi/f - XI_FROM_GAMMA_PHI))}
        for i, f in enumerate(fibs[5:], start=5)
    }
    
    return results

# ============================================================================
# PART 4: The verdict
# ============================================================================

def verdict():
    """Synthesize findings."""
    print("\n" + "=" * 70)
    print("PART 4: THE VERDICT")
    print("=" * 70)
    
    # The numbers
    pi_55 = np.pi / 55
    gamma_phi_minus_1 = GAMMA + np.log(PHI) - 1
    
    print(f"""
    CANDIDATE FORMULAS FOR Ξ - 1:
    
    1. π/55 = {pi_55:.10f}
    2. γ + ln(φ) - 1 = {gamma_phi_minus_1:.10f}
    
    Difference: {abs(pi_55 - gamma_phi_minus_1):.10f}
    Relative: {abs(pi_55 - gamma_phi_minus_1) / pi_55 * 100:.4f}%
    
    EVIDENCE FOR π/55:
    - Exact Fibonacci depth (55 = F₁₀)
    - Clean geometric meaning (55 twists = π)
    - Derived from documented PAC collapse dynamics
    
    EVIDENCE FOR γ + ln(φ):
    - Unifies algebra (γ) and geometry (ln φ)
    - Connects prime smoothing to PAC growth
    - ln(γ + ln(φ)) ≈ π/55 (nested relationship!)
    
    RESOLUTION:
    The nested relationship suggests BOTH might be correct at different levels:
    
    Level 1: Ξ - 1 = π/55 (topological - one twist per F₁₀ steps)
    Level 2: ln(Ξ) ≈ Ξ - 1 (because Ξ ≈ 1, so ln(Ξ) ≈ Ξ - 1)
    Level 3: ln(γ + ln(φ)) ≈ π/55
    
    This would mean:
    γ + ln(φ) = e^(π/55) ≈ 1.0588
    
    TEST:
    e^(π/55) = {np.exp(np.pi/55):.10f}
    γ + ln(φ) = {GAMMA + np.log(PHI):.10f}
    Difference: {abs(np.exp(np.pi/55) - (GAMMA + np.log(PHI))):.10f}
    
    VERDICT: Neither formula is exact! They're both approximations.
    The true relationship might be more subtle.
    """)
    
    # Check the "e^(π/55) = γ + ln(φ)" hypothesis
    exp_pi_55 = np.exp(np.pi/55)
    gamma_ln_phi = GAMMA + np.log(PHI)
    
    return {
        'pi_55': float(pi_55),
        'gamma_phi_minus_1': float(gamma_phi_minus_1),
        'difference': float(abs(pi_55 - gamma_phi_minus_1)),
        'relative_error': float(abs(pi_55 - gamma_phi_minus_1) / pi_55),
        'exp_pi_55': float(exp_pi_55),
        'gamma_ln_phi': float(gamma_ln_phi),
        'exp_pi_55_vs_gamma_ln_phi': float(abs(exp_pi_55 - gamma_ln_phi)),
        'verdict': 'neither_exact',
        'recommendation': 'use_pi_55_for_topology'
    }

# ============================================================================
# PART 5: Build falsification tests
# ============================================================================

def build_falsification_tests():
    """
    Create tests that could FALSIFY each formula.
    """
    print("\n" + "=" * 70)
    print("PART 5: FALSIFICATION TESTS")
    print("=" * 70)
    
    tests = []
    
    # Test 1: Precision measurement in turbulence
    print("\n  TEST F1: Precision turbulence measurement")
    print("    If Ξ can be measured to 0.01% precision from turbulence data,")
    print("    the measured value will distinguish π/55 from γ+ln(φ)")
    print(f"    1 + π/55   = {1 + np.pi/55:.8f}")
    print(f"    γ + ln(φ)  = {GAMMA + np.log(PHI):.8f}")
    print(f"    Difference = {abs((1 + np.pi/55) - (GAMMA + np.log(PHI))):.8f} ({abs((1 + np.pi/55) - (GAMMA + np.log(PHI)))/(1+np.pi/55)*100:.4f}%)")
    print(f"    Required precision: 0.13%")
    
    tests.append({
        'id': 'F1',
        'name': 'Precision turbulence measurement',
        'prediction_pi_55': float(1 + np.pi/55),
        'prediction_gamma_phi': float(GAMMA + np.log(PHI)),
        'distinguishing_precision': 0.13
    })
    
    # Test 2: Depth for π twist
    print("\n  TEST F2: Critical depth for π twist")
    print("    If π/55 is exact: depth should be EXACTLY 55")
    print("    If γ+ln(φ) is exact: depth should be 53.77")
    print("    Measure recursion depth where coherence inverts")
    print(f"    π / (π/55) = 55 (exact)")
    print(f"    π / (γ+ln(φ)-1) = {np.pi / (GAMMA + np.log(PHI) - 1):.2f}")
    
    tests.append({
        'id': 'F2',
        'name': 'Critical depth for π twist',
        'prediction_pi_55': 55.0,
        'prediction_gamma_phi': float(np.pi / (GAMMA + np.log(PHI) - 1))
    })
    
    # Test 3: Temperature dependence
    print("\n  TEST F3: Does Ξ depend on temperature/energy?")
    print("    If Ξ = 1 + π/55: purely topological, temperature-independent")
    print("    If Ξ = γ + ln(φ): might have thermodynamic corrections")
    print("    Test: Measure Ξ at different temperatures in turbulence")
    
    tests.append({
        'id': 'F3',
        'name': 'Temperature independence',
        'prediction_pi_55': 'temperature_independent',
        'prediction_gamma_phi': 'may_have_thermal_corrections'
    })
    
    # Test 4: Higher Fibonacci behavior
    print("\n  TEST F4: Higher Fibonacci behavior")
    print("    If π/55 is fundamental, we should see similar behavior at other F_n")
    print("    Test: Look for 2π twist at depth 110 (2 × F₁₀)")
    print("          Look for π/2 twist at depth ~27.5 (F₉ ≈ 34)")
    
    tests.append({
        'id': 'F4',
        'name': 'Higher Fibonacci behavior',
        'test': 'Look for quantized twist at Fibonacci depths',
        'prediction': '2π at 110, π/2 at ~27-28'
    })
    
    # Test 5: Prime smoothing match
    print("\n  TEST F5: Does γ+ln(φ) appear in prime statistics?")
    print("    If γ+ln(φ) is fundamental for primes, it should appear in:")
    print("    - Prime gap distribution")
    print("    - Mertens product corrections")
    print("    - Riemann zero spacing")
    print("    Current status: γ appears, but (γ+ln(φ)) as a unit is UNTESTED")
    
    tests.append({
        'id': 'F5',
        'name': 'Prime statistics',
        'test': 'Look for γ+ln(φ) as unit in prime distributions',
        'status': 'untested'
    })
    
    return tests

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("exp_23: Ξ DERIVATION CONTEST")
    print("Which formula is fundamental: π/55 or γ + ln(φ)?")
    print("=" * 70)
    
    results = {}
    
    # Part 1: Re-derive from PAC
    results['part1_pac_derive'] = derive_xi_from_pac_collapse()
    
    # Part 2: Test γ + ln(φ) independently
    results['part2_gamma_phi_tests'] = test_gamma_lnphi_independently()
    
    # Part 3: First principles
    results['part3_first_principles'] = derive_xi_first_principles()
    
    # Part 4: Verdict
    results['part4_verdict'] = verdict()
    
    # Part 5: Falsification tests
    results['part5_falsification'] = build_falsification_tests()
    
    # Save
    results['timestamp'] = datetime.now().isoformat()
    results['constants'] = {
        'gamma': GAMMA,
        'phi': PHI,
        'inv_phi': INV_PHI,
        'xi_from_pi': float(XI_FROM_PI),
        'xi_from_gamma_phi': float(XI_FROM_GAMMA_PHI)
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_23_xi_derivation_contest_{timestamp}.json'
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {filename}")
    
    return results

if __name__ == '__main__':
    main()
