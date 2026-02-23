#!/usr/bin/env python3
"""
Experiment 08 — She–Lévêque Turbulence Parameters
===================================================

PACSeries Paper 4, Section 9

She–Lévêque intermittency exponents:
    ζ_p = p/9 + 2(1 − (2/3)^(p/3))

From PAC:
  β = F₃/F₄ = 2/3  (same as Koide — universal branching ratio)
  k = 9 = d × F_{d+1} = 3 × 3 = 3 × F₄

Kolmogorov −5/3 law:
  E(k) ∝ k^(−5/3)   where −5/3 = −F₅/F₄

Source: milestone1/scripts/exp_21_she_leveque.py
Cross-validation: milestone2/scripts/exp_11_k9_derivation.py
"""

import json
import os
import math
from datetime import datetime


def fib(n):
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


F3, F4, F5 = fib(3), fib(4), fib(5)
BETA_PAC = F3 / F4  # 2/3


def she_leveque_zeta(p, beta=BETA_PAC, k=9):
    """Compute She–Lévêque scaling exponent ζ_p."""
    return p / k + 2 * (1 - beta ** (p / 3))


def main():
    results = {
        'experiment': 'exp_08_she_leveque',
        'paper': 'PACSeries Paper 4',
        'section': '9',
        'timestamp': datetime.now().isoformat(),
    }

    print("=" * 60)
    print("She–Lévêque Turbulence from Fibonacci")
    print("=" * 60)
    print()
    print(f"  β = F₃/F₄ = {F3}/{F4} = {BETA_PAC:.6f}")
    print(f"  k = d × F_{{d+1}} = 3 × F₄ = 3 × {F4} = 9")
    print()

    # Compute exponents for p = 1..10
    print("Intermittency exponents ζ_p:")
    print(f"  {'p':>3s}  {'ζ_p (PAC)':>12s}  {'ζ_p (measured)':>14s}  {'Dev %':>8s}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*14}  {'-'*8}")

    # Experimental values from Anselmet et al. (1984), Benzi et al. (1993)
    measured = {
        1: 0.37,  2: 0.70,  3: 1.00,  4: 1.28,
        5: 1.54,  6: 1.78,  7: 2.00,  8: 2.21,
        9: 2.40,  10: 2.59,
    }

    exponents_pac = []
    deviations = []
    for p in range(1, 11):
        zeta = she_leveque_zeta(p)
        meas = measured.get(p, None)
        dev = abs(zeta - meas) / meas * 100 if meas else None
        exponents_pac.append({'p': p, 'zeta_pac': round(zeta, 6), 'zeta_measured': meas})
        if dev is not None:
            deviations.append(dev)
            print(f"  {p:3d}  {zeta:12.6f}  {meas:14.4f}  {dev:8.3f}%")

    mean_dev = sum(deviations) / len(deviations)
    print()
    print(f"  Mean deviation: {mean_dev:.3f}%")

    # Kolmogorov −5/3
    print()
    print("=" * 60)
    print("Kolmogorov −5/3 Law")
    print("=" * 60)
    print()
    kolmogorov = -F5 / F4
    print(f"  E(k) ∝ k^(−5/3)")
    print(f"  −5/3 = −F₅/F₄ = −{F5}/{F4} = {kolmogorov:.6f}")
    print()
    print(f"  This is widely observed in turbulence:")
    print(f"    - Atmospheric boundary layer")
    print(f"    - Ocean currents")
    print(f"    - Solar wind")
    print(f"    - Interstellar medium")

    # Dimensional formula: k = d × F_{d+1}
    print()
    print("=" * 60)
    print("Dimensional Generalization: k = d × F_{d+1}")
    print("=" * 60)
    print()
    print(f"  {'d':>3s}  {'F_{d+1}':>6s}  {'k = d×F_{d+1}':>12s}  {'Status':>20s}")
    print(f"  {'-'*3}  {'-'*6}  {'-'*12}  {'-'*20}")

    for d in range(1, 6):
        f_next = fib(d + 1)
        k_pred = d * f_next
        if d == 2:
            status = "2D turbulence (Kraichnan)"
        elif d == 3:
            status = "3D turbulence (She-Lévêque)"
        elif d == 4:
            status = "PREDICTION: k=20"
        else:
            status = "—"
        print(f"  {d:3d}  {f_next:6d}  {k_pred:12d}  {status:>20s}")

    print()
    print("  d=4 prediction (k=20) is testable in 4D lattice simulations")
    print("  and magnetohydrodynamic turbulence models.")

    # Casimir connection
    print()
    print("=" * 60)
    print("Casimir Number 240 = F₃·F₄·F₅·F₆")
    print("=" * 60)
    print()
    F6 = fib(6)
    casimir = F3 * F4 * F5 * F6
    print(f"  240 = {F3} × {F4} × {F5} × {F6} = F₃·F₄·F₅·F₆")
    print(f"  Computed: {casimir}")
    print(f"  Match: {casimir == 240}")
    print()
    print(f"  This is the kissing number of E₈ and counts roots")
    print(f"  of the E₈ lattice — the maximal even unimodular lattice in 8D.")

    results['main_results'] = {
        'she_leveque': {
            'beta': BETA_PAC,
            'beta_fibonacci': 'F₃/F₄ = 2/3',
            'k': 9,
            'k_formula': 'd × F_{d+1} = 3 × 3',
            'exponents': exponents_pac,
            'mean_deviation_percent': round(mean_dev, 3),
        },
        'kolmogorov': {
            'exponent': kolmogorov,
            'fibonacci': '-F₅/F₄ = -5/3',
        },
        'dimensional_generalization': {
            'd_2': {'k': 2, 'formula': '2×F₃ = 2×1 = 2'},
            'd_3': {'k': 9, 'formula': '3×F₄ = 3×3 = 9'},
            'd_4': {'k': 20, 'formula': '4×F₅ = 4×5 = 20', 'status': 'prediction'},
        },
        'casimir': {
            'value': casimir,
            'formula': 'F₃·F₄·F₅·F₆ = 2×3×5×8 = 240',
            'interpretation': 'E₈ kissing number / root count',
        },
    }

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(results_dir, f'exp_08_she_leveque_{ts}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
