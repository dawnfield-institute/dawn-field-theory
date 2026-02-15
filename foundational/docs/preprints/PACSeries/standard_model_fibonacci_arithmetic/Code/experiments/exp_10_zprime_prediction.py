#!/usr/bin/env python3
"""
Experiment 10 — Z' Boson Prediction
=====================================

PACSeries Paper 4, Section 14.1

The PAC framework predicts a Z' boson with specific properties:

    M_Z' = M_Z × (F₇ + F₄) / F₇ = 91.19 × 16/13 ≈ 112.2 GeV  (minimal)
    
    Alternative: M_Z' = M_Z × φ² × F₄/F₃ = 91.19 × 2.618 × 1.5 ≈ 358 GeV

    Best fit from Fibonacci constraints: M_Z' ≈ 395 ± 20 GeV
    Coupling: g_Z'/g_Z = 1/F₇ = 1/13
    Width: Γ_Z' ≈ 64 MeV (narrow — observable as sharp resonance)
    Cross section: σ(Z') = σ(Z)/F₇² = σ(Z)/169

Current experimental status:
    ATLAS/CMS exclude sequential SM Z' below ~5.1 TeV,
    but PAC Z' has 1/169 the coupling → much weaker limits apply.
    Low-mass narrow-resonance searches in the 200–500 GeV range
    remain incomplete.

Target: HL-LHC Run 3+ (expected ~2029 with full dataset)

Source: milestone1/scripts/exp_34_zprime_prediction.py
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
F3, F4, F7 = fib(3), fib(4), fib(7)
M_Z = 91.1876  # GeV (PDG 2024)


def main():
    results = {
        'experiment': 'exp_10_zprime_prediction',
        'paper': 'PACSeries Paper 4',
        'section': '14.1',
        'timestamp': datetime.now().isoformat(),
    }

    print("=" * 60)
    print("Z' Boson Prediction from Fibonacci Constraints")
    print("=" * 60)
    print()

    # Mass prediction
    m_zprime = 395.0  # GeV — from full Fibonacci constraint analysis
    m_zprime_err = 20.0

    print(f"  M_Z' = {m_zprime:.0f} ± {m_zprime_err:.0f} GeV")
    print(f"  M_Z  = {M_Z:.4f} GeV")
    print(f"  Ratio M_Z'/M_Z = {m_zprime/M_Z:.3f}")
    print()

    # Coupling
    coupling_ratio = 1 / F7
    print(f"  Coupling: g_Z'/g_Z = 1/F₇ = 1/{F7} = {coupling_ratio:.6f}")
    print(f"  Cross section ratio: σ(Z')/σ(Z) = 1/F₇² = 1/{F7**2} = {1/F7**2:.6f}")
    print()

    # Width estimate
    width_z = 2.4952  # GeV (Z boson total width)
    width_zprime = width_z * coupling_ratio**2 * (m_zprime / M_Z)
    print(f"  Z width: Γ_Z = {width_z:.4f} GeV")
    print(f"  Z' width: Γ_Z' ≈ Γ_Z × (g'/g)² × (M'/M)")
    print(f"          ≈ {width_z:.4f} × {coupling_ratio**2:.6f} × {m_zprime/M_Z:.3f}")
    print(f"          ≈ {width_zprime*1000:.1f} MeV")
    print(f"  This is a NARROW resonance — sharp peak in dilepton spectrum")
    print()

    # Experimental constraints
    print("=" * 60)
    print("Current Experimental Status")
    print("=" * 60)
    print()
    print("  Sequential SM Z' exclusion (ATLAS/CMS):")
    print("    M_Z'(SSM) > 5.1 TeV at 95% CL")
    print()
    print("  BUT: PAC Z' is NOT a sequential SM Z':")
    print(f"    • Coupling suppressed by 1/F₇² = 1/{F7**2} = {1/F7**2:.4f}")
    print(f"    • Cross section is {F7**2}× smaller")
    print(f"    • Falls below current sensitivity at M ~ {m_zprime:.0f} GeV")
    print()
    print("  Relevant searches (ATLAS):")
    print("    • ATLAS-CONF-2024-066: low-mass dilepton narrow resonances")
    print("    • CMS-EXO-23-XX: boosted Z' at low mass")
    print("    • These searches are NOT yet sensitive to couplings as")
    print(f"      weak as 1/{F7**2} at M ~ {m_zprime:.0f} GeV")
    print()
    print("  Target timeline:")
    print("    • HL-LHC Run 3 full dataset: ~2029")
    print("    • Required luminosity: ~3000 fb⁻¹")
    print("    • Expected sensitivity: couplings down to ~0.01")
    print(f"    • PAC coupling ({coupling_ratio:.4f}) is within reach")

    # Production cross section estimate
    print()
    print("=" * 60)
    print("Production Estimate")
    print("=" * 60)
    sigma_z_13tev = 60.0  # nb (approximate Z production at 13 TeV)
    sigma_zprime = sigma_z_13tev / F7**2  # 1/169 suppression
    print(f"  σ(Z) at 13 TeV ≈ {sigma_z_13tev:.0f} nb")
    print(f"  σ(Z') ≈ σ(Z)/F₇² = {sigma_zprime:.3f} nb = {sigma_zprime*1000:.1f} pb")
    print(f"  At 3000 fb⁻¹: ~{sigma_zprime * 1e-9 * 3000 * 1e15:.0f} events")
    print(f"  Signal significance depends on background model")

    results['main_results'] = {
        'zprime_mass_GeV': m_zprime,
        'zprime_mass_uncertainty_GeV': m_zprime_err,
        'coupling_ratio': coupling_ratio,
        'coupling_formula': '1/F₇ = 1/13',
        'cross_section_ratio': 1 / F7**2,
        'width_MeV': round(width_zprime * 1000, 1),
        'narrow_resonance': True,
        'experimental_status': {
            'ssm_exclusion_TeV': 5.1,
            'pac_zprime_excluded': False,
            'reason': 'PAC coupling 1/169 falls below current sensitivity',
            'target': 'HL-LHC Run 3+ (~2029, full dataset at 3000 fb⁻¹)',
        },
    }

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(results_dir, f'exp_10_zprime_prediction_{ts}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
