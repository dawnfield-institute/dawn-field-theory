#!/usr/bin/env python3
"""
Experiment 06 — Mass Ratios from Fibonacci Arithmetic
======================================================

PACSeries Paper 4, Section 6

Derives fermion mass ratios from Fibonacci-structured expressions:

    m_μ/m_e  → F₄ × F₆² × (1 + 1/F₇) = 3 × 64 × 14/13 ≈ 206.769  (5 ppm)
    m_τ/m_e  → F₄ × F₇ × F₁₁ + F₅    = 3 × 13 × 89 + 5  = 3476    (350 ppm)
    m_p/m_e  → F₄ × F₉ × F₁₂ / F₆    = 3 × 34 × 144/8   = 1836    (83 ppm)

Source: mass_derivation/exp_05_tighten_mass.py, exp_06_validate_tight.py
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


PHI = (1 + math.sqrt(5)) / 2
F2, F3, F4, F5 = fib(2), fib(3), fib(4), fib(5)
F6, F7, F9, F10, F11, F12 = fib(6), fib(7), fib(9), fib(10), fib(11), fib(12)

# PDG 2024 masses (MeV/c²)
M_E = 0.51099895
M_MU = 105.6583755
M_TAU = 1776.86
M_P = 938.27208816

# Measured ratios
RATIO_MU_E = M_MU / M_E      # 206.7682830
RATIO_P_E = M_P / M_E         # 1836.15267
RATIO_TAU_E = M_TAU / M_E     # 3477.23


def main():
    results = {
        'experiment': 'exp_06_mass_ratios',
        'paper': 'PACSeries Paper 4',
        'section': '6',
        'timestamp': datetime.now().isoformat(),
    }

    print("=" * 60)
    print("Fermion Mass Ratios from Fibonacci Arithmetic")
    print("=" * 60)
    print()

    # μ/e ratio: F₄ × F₆² × (1 + 1/F₇) = 3 × 64 × (14/13)
    mu_e_pac = F4 * F6**2 * (1 + 1/F7)
    mu_e_dev = abs(mu_e_pac - RATIO_MU_E) / RATIO_MU_E * 1e6

    print("1. Muon/Electron Mass Ratio")
    print(f"   Formula: m_μ/m_e = F₄ × F₆² × (1 + 1/F₇) = {F4} × {F6**2} × {14}/{13}")
    print(f"   PAC:      {mu_e_pac:.6f}")
    print(f"   Measured: {RATIO_MU_E:.6f}")
    print(f"   Deviation: {mu_e_dev:.1f} ppm")
    print()

    # τ/e ratio: F₄ × F₇ × F₁₁ + F₅ = 3 × 13 × 89 + 5
    tau_e_pac = F4 * F7 * F11 + F5
    tau_e_dev = abs(tau_e_pac - RATIO_TAU_E) / RATIO_TAU_E * 1e6

    print("2. Tau/Electron Mass Ratio")
    print(f"   Formula: m_τ/m_e = F₄ × F₇ × F₁₁ + F₅ = {F4}×{F7}×{F11}+{F5} = {tau_e_pac}")
    print(f"   PAC:      {tau_e_pac:.2f}")
    print(f"   Measured: {RATIO_TAU_E:.2f}")
    print(f"   Deviation: {tau_e_dev:.0f} ppm ({tau_e_dev/1e4:.4f}%)")
    print()

    # p/e ratio: F₄ × F₉ × F₁₂ / F₆ = 3 × 34 × 144 / 8
    p_e_pac = F4 * F9 * F12 / F6
    p_e_dev = abs(p_e_pac - RATIO_P_E) / RATIO_P_E * 1e6

    print("3. Proton/Electron Mass Ratio")
    print(f"   Formula: m_p/m_e = F₄ × F₉ × F₁₂ / F₆ = {F4}×{F9}×{F12}/{F6} = {p_e_pac:.0f}")
    print(f"   PAC:      {p_e_pac:.4f}")
    print(f"   Measured: {RATIO_P_E:.4f}")
    print(f"   Deviation: {p_e_dev:.0f} ppm ({p_e_dev/1e4:.4f}%)")
    print()

    # Derived ratios
    tau_mu_pac = tau_e_pac / mu_e_pac
    tau_mu_meas = RATIO_TAU_E / RATIO_MU_E
    tau_mu_dev = abs(tau_mu_pac - tau_mu_meas) / tau_mu_meas * 1e6

    p_mu_pac = p_e_pac / mu_e_pac
    p_mu_meas = RATIO_P_E / RATIO_MU_E
    p_mu_dev = abs(p_mu_pac - p_mu_meas) / p_mu_meas * 1e6

    # Summary table
    print("=" * 60)
    print("Summary: Mass Ratio Precision")
    print("=" * 60)
    print()
    print(f"  {'Ratio':>10s}  {'PAC':>12s}  {'Measured':>12s}  {'Dev (ppm)':>10s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}")
    print(f"  {'μ/e':>10s}  {mu_e_pac:12.4f}  {RATIO_MU_E:12.4f}  {mu_e_dev:10.1f}")
    print(f"  {'τ/e':>10s}  {tau_e_pac:12.2f}  {RATIO_TAU_E:12.2f}  {tau_e_dev:10.0f}")
    print(f"  {'p/e':>10s}  {p_e_pac:12.4f}  {RATIO_P_E:12.4f}  {p_e_dev:10.0f}")
    print(f"  {'τ/μ':>10s}  {tau_mu_pac:12.4f}  {tau_mu_meas:12.4f}  {tau_mu_dev:10.0f}")
    print(f"  {'p/μ':>10s}  {p_mu_pac:12.4f}  {p_mu_meas:12.4f}  {p_mu_dev:10.0f}")
    print()

    # F₄ = 3 universality
    print("=" * 60)
    print("F₄ = 3 Universality")
    print("=" * 60)
    print()
    print(f"  F₄ = {F4} appears as leading factor in ALL three mass formulas.")
    print(f"  If F₄ encodes the number of fermion generations, its universal")
    print(f"  presence follows from PAC branching structure.")
    print()

    results['main_results'] = {
        'mu_over_e': {
            'formula': 'F₄ × F₆² × (1 + 1/F₇)',
            'decomposition': f'{F4} × {F6}² × (1 + 1/{F7}) = {F4} × {F6**2} × {14}/{13}',
            'pac_value': round(mu_e_pac, 6),
            'measured': round(RATIO_MU_E, 6),
            'deviation_ppm': round(mu_e_dev, 1),
        },
        'tau_over_e': {
            'formula': 'F₄ × F₇ × F₁₁ + F₅',
            'decomposition': f'{F4} × {F7} × {F11} + {F5} = {tau_e_pac}',
            'pac_value': round(tau_e_pac, 2),
            'measured': round(RATIO_TAU_E, 2),
            'deviation_ppm': round(tau_e_dev, 0),
        },
        'proton_over_e': {
            'formula': 'F₄ × F₉ × F₁₂ / F₆',
            'decomposition': f'{F4} × {F9} × {F12} / {F6} = {p_e_pac:.0f}',
            'pac_value': round(p_e_pac, 4),
            'measured': round(RATIO_P_E, 4),
            'deviation_ppm': round(p_e_dev, 0),
        },
        'derived_ratios': {
            'tau_over_mu': {
                'pac_value': round(tau_mu_pac, 4),
                'measured': round(tau_mu_meas, 4),
                'deviation_ppm': round(tau_mu_dev, 0),
            },
            'proton_over_mu': {
                'pac_value': round(p_mu_pac, 4),
                'measured': round(p_mu_meas, 4),
                'deviation_ppm': round(p_mu_dev, 0),
            },
        },
        'f4_universality': 'F₄ = 3 appears as leading factor in all three mass formulas',
    }

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(results_dir, f'exp_06_mass_ratios_{ts}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
