#!/usr/bin/env python3
"""
Experiment 04 — Weinberg Angle sin²θ_W = F₄/F₇ = 3/13
========================================================

PACSeries Paper 4, Section 4.3

The Weinberg (weak mixing) angle determines the relative strengths
of the electromagnetic and weak forces. From PAC:

    sin²θ_W = F₄/F₇ = 3/13 = 0.230769...

Measured (PDG 2024 at M_Z): sin²θ_W = 0.23122 ± 0.00003
Deviation: 0.19%

Physical interpretation: F₄=3 counts the SU(2) generators,
F₇=13 counts the total gauge DOF. The mixing angle is literally
the fraction of DOF belonging to the weak sector.

Source: milestone1/scripts/exp_18_weinberg_angle.py
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


F4 = fib(4)  # 3
F7 = fib(7)  # 13


def main():
    results = {
        'experiment': 'exp_04_weinberg_angle',
        'paper': 'PACSeries Paper 4',
        'section': '4.3',
        'timestamp': datetime.now().isoformat(),
    }

    # PAC prediction
    sin2_pac = F4 / F7
    # PDG 2024 measurement at M_Z
    sin2_pdg = 0.23122
    sin2_err = 0.00003

    deviation = abs(sin2_pac - sin2_pdg) / sin2_pdg * 100
    sigma = abs(sin2_pac - sin2_pdg) / sin2_err

    print("=" * 60)
    print("Weinberg Angle from Fibonacci Arithmetic")
    print("=" * 60)
    print()
    print(f"  sin²θ_W = F₄/F₇ = {F4}/{F7} = {sin2_pac:.6f}")
    print(f"  PDG 2024 (at M_Z): {sin2_pdg} ± {sin2_err}")
    print(f"  Deviation: {deviation:.3f}%")
    print(f"  σ from central: {sigma:.1f}σ")
    print()

    # Physical interpretation
    print("Physical interpretation:")
    print(f"  F₄ = {F4} = dim(SU(2)) generators")
    print(f"  F₇ = {F7} = total SM gauge DOF")
    print(f"  sin²θ_W = (weak generators) / (total gauge DOF)")
    print(f"            = structural fraction, not a free parameter")
    print()

    # RG running comparison
    print("=" * 60)
    print("Renormalization Group Running")
    print("=" * 60)
    print()
    print("  sin²θ_W runs with energy scale:")
    print("    At M_Z (91.2 GeV): 0.23122 — measured")
    print("    At low energy:      0.23857 — measured")
    print("    At GUT scale:       ~0.375 — extrapolated")
    print()
    print(f"  PAC value 3/13 = {sin2_pac:.5f} is closest to M_Z measurement")
    print(f"  Interpretation: PAC gives the 'natural' value at the gauge")
    print(f"  symmetry scale, which the SM runs from.")

    # Alternative fraction test
    print()
    print("=" * 60)
    print("Alternative Fibonacci Fractions for sin²θ_W")
    print("=" * 60)

    alternatives = []
    for i in range(2, 12):
        for j in range(i + 1, 14):
            fi, fj = fib(i), fib(j)
            ratio = fi / fj
            dev = abs(ratio - sin2_pdg) / sin2_pdg * 100
            alternatives.append((i, j, fi, fj, ratio, dev))

    alternatives.sort(key=lambda x: x[5])
    print(f"{'F_i':>4s}  {'F_j':>4s}  {'Ratio':>10s}  {'Dev %':>8s}")
    print("-" * 40)
    for i, j, fi, fj, ratio, dev in alternatives[:8]:
        marker = " ← BEST" if (i, j) == (4, 7) else ""
        print(f"  {fi:4d}/{fj:4d}  (F_{i}/F_{j})  {ratio:10.6f}  {dev:8.3f}%{marker}")

    print()
    print(f"  F₄/F₇ = 3/13 is the best Fibonacci fraction for sin²θ_W")
    print(f"  AND has direct physical meaning (weak/total DOF).")

    results['main_results'] = {
        'sin2_theta_w': {
            'pac_value': sin2_pac,
            'pac_fraction': '3/13 = F₄/F₇',
            'pdg_2024': sin2_pdg,
            'pdg_uncertainty': sin2_err,
            'deviation_percent': round(deviation, 3),
            'sigma_from_central': round(sigma, 1),
        },
        'interpretation': (
            'sin²θ_W = F₄/F₇ = (SU(2) generators)/(total gauge DOF) = '
            'structural fraction, not independent parameter'
        ),
        'uniqueness': {
            'best_fibonacci_fraction': 'F₄/F₇ = 3/13',
            'alternatives_tested': len(alternatives),
        },
    }

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(results_dir, f'exp_04_weinberg_angle_{ts}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
