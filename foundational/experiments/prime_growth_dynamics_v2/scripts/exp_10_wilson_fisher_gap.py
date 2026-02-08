"""
Experiment 10: Wilson-Fisher Gap
==================================

Tests whether the discrepancy between 1/φ = 0.6180 and the Wilson-
Fisher critical exponent ν = 0.630 is a γ-correction:

    ν = 1/φ + f(γ)

where f(γ) is some function of the Euler-Mascheroni constant.

The gap is: ν - 1/φ = 0.630 - 0.6180 = 0.0120

Source: oscillation_attractor_dynamics found 1/φ = 0.618 appearing
as an attractor but couldn't derive the full ν = 0.630.

Success criterion: Find a closed-form expression for ν - 1/φ using
phase constants (γ, ln(φ), φ, π) with < 1% error.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from phase_engine import *

def run():
    print("=" * 70)
    print("EXP 10: Wilson-Fisher Gap — ν vs 1/φ")
    print("=" * 70)

    # Known values
    inv_phi = 1 / PHI  # 0.6180339887...
    nu_WF = 0.6301     # Wilson-Fisher ν for 3D Ising (best estimate)
    gap = nu_WF - inv_phi  # ≈ 0.01207

    print(f"  1/φ        = {inv_phi:.10f}")
    print(f"  ν (W-F)    = {nu_WF:.10f}")
    print(f"  Gap         = {gap:.10f}")
    print(f"  Relative gap = {gap/nu_WF*100:.4f}%")

    # ================================================================
    # Test 1: Systematic formula search for the gap
    # ================================================================
    print("\n--- Test 1: Systematic formula search for gap ≈ 0.01207 ---")

    gap_candidates = phase_formula_search(gap, max_depth=3)

    print(f"  Found {len(gap_candidates)} candidates within 2%:")
    for m in gap_candidates[:20]:
        print(f"    {m['expression']} = {m['value']:.10f}  (error = {m['error_pct']:.4f}%)")

    # ================================================================
    # Test 2: Physics-motivated candidates for ν
    # ================================================================
    print("\n--- Test 2: Direct ν candidates ---")

    nu_candidates = [
        ("1/φ + γ/(4π²)", inv_phi + GAMMA / (4 * math.pi**2)),
        ("1/φ + γ/φ⁵", inv_phi + GAMMA / PHI**5),
        ("1/φ + γ²/φ³", inv_phi + GAMMA**2 / PHI**3),
        ("1/φ + ln(φ)/φ⁵", inv_phi + LN_PHI / PHI**5),
        ("1/φ + 1/(4π²)", inv_phi + 1 / (4 * math.pi**2)),
        ("1/φ + γ·ln(φ)/φ³", inv_phi + GAMMA * LN_PHI / PHI**3),
        ("1/φ + (Ξ-1)/φ³", inv_phi + (XI_ANALYTIC - 1) / PHI**3),
        ("1/φ + γ/(8π)", inv_phi + GAMMA / (8 * math.pi)),
        ("1/φ + γ²/4", inv_phi + GAMMA**2 / 4),
        ("φ/(1 + φ + γ)", PHI / (1 + PHI + GAMMA)),
        ("2/(3 + γ/φ)", 2 / (3 + GAMMA / PHI)),
        ("(1 + γ)/(2 + φ)", (1 + GAMMA) / (2 + PHI)),
        ("φ²/(φ² + 2 + γ)", PHI**2 / (PHI**2 + 2 + GAMMA)),
        ("(3 - φ)/(3 + γ)", (3 - PHI) / (3 + GAMMA)),
        ("1/(φ + γ/φ)", 1 / (PHI + GAMMA/PHI)),
        ("(1 + γ·ln(φ))/(2 + φ)", (1 + GAMMA * LN_PHI) / (2 + PHI)),
        ("1/φ + γ/(F₇²)", inv_phi + GAMMA / F7**2),
        ("1/φ + 1/(φ⁴·π)", inv_phi + 1 / (PHI**4 * math.pi)),
        ("(2·γ + 1)/(2·φ + 1)", (2*GAMMA + 1) / (2*PHI + 1)),
        ("F₃/(F₄ + γ/φ)", F[3] / (F[4] + GAMMA/PHI)),
        # Higher-order corrections
        ("1/φ + γ³", inv_phi + GAMMA**3),
        ("1/φ + γ/55", inv_phi + GAMMA / 55),
        ("1/φ + π/(φ⁴·F₇)", inv_phi + math.pi / (PHI**4 * F7)),
        ("1/φ + (γ·ln(φ))/F₈", inv_phi + (GAMMA * LN_PHI) / F[8]),
    ]

    print(f"  {'Expression':<30} {'Value':>12} {'Error %':>10}")
    print(f"  {'-'*55}")
    best_nu = None
    best_nu_err = float('inf')
    for name, val in sorted(nu_candidates, key=lambda x: abs(x[1] - nu_WF)):
        err = abs(val - nu_WF) / nu_WF * 100
        marker = " ← BEST" if err < best_nu_err and err < 1 else ""
        if err < best_nu_err:
            best_nu = (name, val, err)
            best_nu_err = err
        if err < 5:
            print(f"  {name:<30} {val:12.10f} {err:10.4f}%{marker}")

    # ================================================================
    # Test 3: Broader formula search for ν directly
    # ================================================================
    print("\n--- Test 3: Systematic formula search for ν ---")

    nu_systematic = phase_formula_search(nu_WF, max_depth=3)
    print(f"  Found {len(nu_systematic)} candidates within 2%:")
    for m in nu_systematic[:15]:
        print(f"    {m['expression']} = {m['value']:.10f}  (error = {m['error_pct']:.4f}%)")

    # ================================================================
    # Test 4: Gap interpretation
    # ================================================================
    print("\n--- Test 4: Gap physical interpretation ---")

    print(f"\n  Gap = ν - 1/φ = {gap:.10f}")
    print(f"  Gap/γ = {gap/GAMMA:.10f}")
    print(f"  Gap/ln(φ) = {gap/LN_PHI:.10f}")
    print(f"  Gap·φ³ = {gap*PHI**3:.10f}")
    print(f"  Gap·φ⁴ = {gap*PHI**4:.10f}")
    print(f"  Gap·φ⁵ = {gap*PHI**5:.10f}")
    print(f"  Gap·4π² = {gap*4*math.pi**2:.10f}")
    print(f"  Gap/γ³ = {gap/GAMMA**3:.10f}")
    print(f"  Gap·55 = {gap*55:.10f}")
    print(f"  Gap·F₇² = {gap*F7**2:.10f}")
    print(f"  Gap·F₇  = {gap*F7:.10f}")

    # Check if any of these round to known values
    checks = {
        'Gap·φ⁵': gap * PHI**5,
        'Gap·4π²': gap * 4 * math.pi**2,
        'Gap·55': gap * 55,
        'Gap/γ³': gap / GAMMA**3,
        'Gap·F₇': gap * F7,
    }

    for name, val in checks.items():
        nearest_const = min(
            [('γ', GAMMA), ('ln(φ)', LN_PHI), ('1/φ', inv_phi),
             ('φ-1', PHI-1), ('1', 1.0), ('π/55', math.pi/55),
             ('Ξ-1', XI_ANALYTIC-1)],
            key=lambda x: abs(x[1] - val)
        )
        err = abs(nearest_const[1] - val) / val * 100 if val != 0 else float('inf')
        if err < 10:
            print(f"  {name} ≈ {nearest_const[0]} "
                  f"({val:.6f} vs {nearest_const[1]:.6f}, err={err:.2f}%)")

    # ================================================================
    # Test 5: Other critical exponents
    # ================================================================
    print("\n--- Test 5: Other 3D Ising critical exponents ---")

    # Standard 3D Ising critical exponents (best estimates)
    critical_exponents = {
        'α': 0.1096,   # Specific heat
        'β_ising': 0.3265,   # Order parameter
        'γ_ising': 1.2372,   # Susceptibility
        'δ': 4.7893,   # Critical isotherm
        'η': 0.0364,   # Anomalous dimension
        'ν': 0.6301,   # Correlation length
    }

    print(f"  {'Exponent':<12} {'Value':>10} {'φ-formula':>20} {'Error %':>10}")
    print(f"  {'-'*55}")
    for name, val in critical_exponents.items():
        # Quick search in phase constants
        results = phase_formula_search(val, max_depth=2)
        if results:
            best = results[0]
            print(f"  {name:<12} {val:10.4f} {best['expression']:>20} {best['error_pct']:10.4f}%")
        else:
            print(f"  {name:<12} {val:10.4f} {'(none found)':>20}")

    # ================================================================
    # Results
    # ================================================================
    success = best_nu is not None and best_nu[2] < 1.0

    data = {
        'experiment': 'exp_10_wilson_fisher_gap',
        'hypothesis': 'ν - 1/φ is a γ-correction',
        'known_values': {
            'inv_phi': float(inv_phi),
            'nu_WF': nu_WF,
            'gap': float(gap),
        },
        'best_nu_formula': best_nu[0] if best_nu else None,
        'best_nu_value': float(best_nu[1]) if best_nu else None,
        'best_nu_error_pct': float(best_nu[2]) if best_nu else None,
        'gap_candidates_top5': [{'expr': m['expression'], 'val': m['value'], 'err': m['error']} for m in gap_candidates[:5]],
        'nu_candidates_top5': [{'expr': m['expression'], 'val': m['value'], 'err': m['error']} for m in nu_systematic[:5]],
        'success': success,
        'success_criterion': 'Closed-form ν with < 1% error from phase constants',
    }

    print(f"\n{'='*70}")
    if best_nu:
        print(f"BEST ν FORMULA: {best_nu[0]} = {best_nu[1]:.10f} (error={best_nu[2]:.4f}%)")
    print(f"SUCCESS: {'YES' if success else 'INCONCLUSIVE'}")
    print(f"{'='*70}")

    save_results(data, 'exp_10_wilson_fisher_gap')
    return data


if __name__ == '__main__':
    run()
