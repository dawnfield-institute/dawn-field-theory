"""
Experiment 02: Critical Exponent Beta
======================================

Tests H3: The sec_prime_manifold critical exponent β ≈ 0.79 has a
phase-ratio origin in {γ, ln(φ), Ξ, φ, F_n}.

Success criterion: A formula matching β to < 2% without free parameters.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from phase_engine import *

def run():
    print("=" * 70)
    print("EXP 02: Critical Exponent Beta from Phase Constants")
    print("=" * 70)

    target_beta = 0.79

    # ================================================================
    # Test 1: Systematic search
    # ================================================================
    print("\n--- Test 1: Formula search for β ≈ 0.79 ---")
    matches = phase_formula_search(target_beta, max_depth=3, tolerance=0.03)
    print(f"Found {len(matches)} candidates within 3%:")
    for m in matches[:20]:
        print(f"  {m['expression']:40s} = {m['value']:.6f}  error: {m['error_pct']:.4f}%")

    # ================================================================
    # Test 2: Physics-motivated candidates
    # ================================================================
    print("\n--- Test 2: Physics-motivated candidates ---")

    candidates = {
        'ln(φ)/γ': LN_PHI / GAMMA,
        'γ/Ξ': GAMMA / XI_ANALYTIC,
        '1 - ln(φ)': 1 - LN_PHI,
        '2*ln(φ)/Ξ': 2 * LN_PHI / XI_ANALYTIC,
        'γ^(1/φ)': GAMMA ** (1/PHI),
        'ln(φ) + γ/π': LN_PHI + GAMMA / math.pi,
        'φ - ln(φ)/γ': PHI - LN_PHI / GAMMA,
        '1/ln(Ξ+1)': 1 / math.log(XI_ANALYTIC + 1),
        'γ*ln(φ)*π': GAMMA * LN_PHI * math.pi,
        'F4/φ^2': F4 / PHI**2,
        'ln(2)/ln(φ+1)': math.log(2) / math.log(PHI + 1),
        'Ξ - ln(φ)/γ': XI_ANALYTIC - LN_PHI / GAMMA,
        'γ + ln(φ)/π': GAMMA + LN_PHI / math.pi,
        '2γ - ln(φ)': 2 * GAMMA - LN_PHI,
        '(1+γ)/Ξ/φ': (1 + GAMMA) / XI_ANALYTIC / PHI,
        'πγln(φ)': math.pi * GAMMA * LN_PHI,
        'φ^(-ln(2))': PHI ** (-math.log(2)),
        'ln(φ)*φ': LN_PHI * PHI,
        'γ/(1-ln(φ))': GAMMA / (1 - LN_PHI),
        'Ξ*ln(φ)': XI_ANALYTIC * LN_PHI,
        'sqrt(γ*ln(φ))': math.sqrt(GAMMA * LN_PHI),
        '1 - ln(φ)/Ξ': 1 - LN_PHI / XI_ANALYTIC,
    }

    results = []
    for name, val in sorted(candidates.items(), key=lambda x: abs(x[1] - target_beta)):
        error = abs(val - target_beta) / target_beta * 100
        results.append({'formula': name, 'value': val, 'error_pct': error})
        marker = " *** " if error < 2 else ""
        print(f"  {name:35s} = {val:.6f}  error: {error:.3f}%{marker}")

    # ================================================================
    # Test 3: Known critical exponents comparison
    # ================================================================
    print("\n--- Test 3: Known critical exponents ---")
    known = {
        'Ising 2D β': 1/8,           # 0.125
        'Ising 3D β': 0.3265,
        'Mean-field β': 0.5,
        'XY 3D β': 0.3485,
        'Heisenberg 3D β': 0.3689,
        'Percolation 2D β': 5/36,     # 0.1389
        'Percolation 3D β': 0.4181,
        'Our β': 0.79,
    }
    print("  Note: Our β = 0.79 is unusually high for standard universality classes.")
    print("  This might indicate it's not a standard critical exponent but a phase ratio.\n")
    for name, val in sorted(known.items(), key=lambda x: x[1]):
        print(f"  {name:25s} = {val:.4f}")

    # ================================================================
    # Results
    # ================================================================
    best = min(results, key=lambda x: x['error_pct'])

    data = {
        'experiment': 'exp_02_critical_exponent_beta',
        'target': target_beta,
        'formula_search_top10': matches[:10] if matches else [],
        'physics_candidates': results,
        'best_match': best,
        'known_exponents': known,
        'success': best['error_pct'] < 2.0,
        'success_criterion': '< 2% error without free parameters',
    }

    print(f"\n{'='*70}")
    print(f"BEST MATCH: {best['formula']} = {best['value']:.6f} ({best['error_pct']:.3f}%)")
    print(f"SUCCESS: {'YES' if data['success'] else 'NO'} (criterion: < 2%)")
    print(f"{'='*70}")

    save_results(data, 'exp_02_beta')
    return data


if __name__ == '__main__':
    run()
