"""
Experiment 01: Lambda Star Derivation
======================================

Tests H1: The sec_prime_manifold critical point λ* = 0.9816 should be
expressible in terms of phase constants {γ, ln(φ), Ξ, φ, F_n}.

Success criterion: A formula matching λ* to < 0.5% without free parameters.

Known data:
- λ* = 0.9816 (k=9, optimal)
- λ* varies with k: k=8→0.9967, k=9→0.9809, k=10→0.9302, k=11→0.9005
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from phase_engine import *

def run():
    print("=" * 70)
    print("EXP 01: Lambda Star Derivation from Phase Constants")
    print("=" * 70)

    # Target values from sec_prime_manifold
    lambda_star = {
        3: 0.9998, 4: 0.9996, 6: 0.9992, 7: 0.9987,
        8: 0.9967, 9: 0.9809, 10: 0.9302, 11: 0.9005
    }
    target_k9 = 0.9816

    # ================================================================
    # Test 1: Direct search for λ*(k=9)
    # ================================================================
    print("\n--- Test 1: Formula search for λ* = 0.9816 ---")
    matches = phase_formula_search(target_k9, max_depth=3, tolerance=0.01)
    print(f"Found {len(matches)} candidates within 1%:")
    for m in matches[:15]:
        print(f"  {m['expression']:40s} = {m['value']:.6f}  error: {m['error_pct']:.4f}%")

    # ================================================================
    # Test 2: Specific physics-motivated candidates
    # ================================================================
    print("\n--- Test 2: Physics-motivated candidates ---")

    candidates = {
        '1 - (1-ln(φ))/F10': 1 - (1 - LN_PHI) / F10,
        '1 - γ/F10': 1 - GAMMA / F10,
        '1 - 1/(F10*φ)': 1 - 1 / (F10 * PHI),
        '1 - ln(φ)/F7': 1 - LN_PHI / F7,
        'ln(φ)/γ * (something needs F)': LN_PHI / GAMMA,
        '1 - Ξ/F10': 1 - XI_ANALYTIC / F10,
        '1 - π/(F10*F4)': 1 - math.pi / (F10 * F4),
        'φ^(-ln(φ))': PHI ** (-LN_PHI),
        '1 - 1/F10': 1 - 1/F10,
        '1 - F3/(F10*φ)': 1 - F3 / (F10 * PHI),
        'exp(-γ/F10)': math.exp(-GAMMA / F10),
        'exp(-ln(φ)/F7)': math.exp(-LN_PHI / F7),
        '1 - γ*ln(φ)/F7': 1 - GAMMA * LN_PHI / F7,
        'φ^(-1/F7)': PHI ** (-1/F7),
        '1 - Ξ/(F10+F3)': 1 - XI_ANALYTIC / (F10 + F3),
        'cos(π/F10)': math.cos(math.pi / F10),
        '1 - 1/(φ*F7)': 1 - 1 / (PHI * F7),
        '(F10-1)/F10': (F10 - 1) / F10,
        'exp(-1/F10)': math.exp(-1 / F10),
        '1 - (Ξ-1)': 1 - (XI_ANALYTIC - 1),
        '1 - π/F10^2': 1 - math.pi / F10**2,
    }

    results = []
    for name, val in sorted(candidates.items(), key=lambda x: abs(x[1] - target_k9)):
        error = abs(val - target_k9) / target_k9 * 100
        results.append({'formula': name, 'value': val, 'error_pct': error})
        marker = " *** " if error < 0.5 else ""
        print(f"  {name:35s} = {val:.6f}  error: {error:.3f}%{marker}")

    # ================================================================
    # Test 3: Does λ* scale with k in a predictable way?
    # ================================================================
    print("\n--- Test 3: λ*(k) scaling pattern ---")

    # Check if (1 - λ*) has Fibonacci structure
    print(f"\n  k → 1-λ* and potential patterns:")
    for k, lam in sorted(lambda_star.items()):
        delta = 1 - lam
        # Check against various Fibonacci ratios
        print(f"  k={k:2d}: λ*={lam:.4f}, 1-λ*={delta:.4f}, "
              f"1/F(k)={1/fibonacci(k):.4f}, "
              f"ln(φ)/F(k)={LN_PHI/fibonacci(k):.4f}, "
              f"γ/F(k+1)={GAMMA/fibonacci(k+1):.4f}")

    # Test: does 1-λ* ≈ C * φ^(-k) for some C?
    print(f"\n  Testing 1-λ* = C × φ^(-k):")
    for k, lam in sorted(lambda_star.items()):
        delta = 1 - lam
        if delta > 0:
            c_implied = delta * PHI**k
            print(f"  k={k:2d}: C = {c_implied:.4f}")

    # Test: does 1-λ* ≈ γ^k / F(something)?
    print(f"\n  Testing 1-λ* ≈ γ^(k-7):")
    for k, lam in sorted(lambda_star.items()):
        delta = 1 - lam
        if delta > 0 and delta < 1:
            ratio = delta / GAMMA**(k-7) if k != 7 else delta
            print(f"  k={k:2d}: 1-λ* / γ^(k-7) = {ratio:.4f}")

    # ================================================================
    # Results
    # ================================================================
    best = min(results, key=lambda x: x['error_pct'])

    data = {
        'experiment': 'exp_01_lambda_star_derivation',
        'target': target_k9,
        'lambda_star_by_k': lambda_star,
        'formula_search_top10': matches[:10] if matches else [],
        'physics_candidates': results,
        'best_match': best,
        'success': best['error_pct'] < 0.5,
        'success_criterion': '< 0.5% error without free parameters',
    }

    print(f"\n{'='*70}")
    print(f"BEST MATCH: {best['formula']} = {best['value']:.6f} ({best['error_pct']:.3f}%)")
    print(f"SUCCESS: {'YES' if data['success'] else 'NO'} (criterion: < 0.5%)")
    print(f"{'='*70}")

    save_results(data, 'exp_01_lambda_star')
    return data


if __name__ == '__main__':
    run()
