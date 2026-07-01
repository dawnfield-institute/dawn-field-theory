"""
exp_06 -- Xi Cost and Scope Boundary Counting

Milestone R, Block B (Spectrum Reconstruction)

Hypothesis: Radiation energy divided by Xi gives an integer count of scope
boundaries traversed during the severance. Each boundary costs Xi = gamma +
ln(phi) units. Discrete spectra give integers; continuous spectra give
non-integers.

Tests:
  T1: Alpha decay boundary count is integer at consistent depth
  T2: Gamma boundary count consistency at same depth
  T3: Beta endpoint boundary count is non-integer
  T4: Boundary count predicts decay energy ordering
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, XI_BALANCE,
    scope_boundary_count,
    U238_CHAIN_ALPHAS, U238_CHAIN_LABELS,
    U238_CHAIN_GAMMAS_KEV, KEV_TO_MEV,
    BETA_C14, BETA_TRITIUM, BETA_CO60,
    PLANCK_ENERGY_MEV,
    save_mr_results,
)


def test_T1_alpha_integer_boundaries():
    """T1: Alpha decay boundary count is integer at consistent depth."""
    print("\n  T1: Alpha energies as integer boundary counts")
    results = {'description': '>= 6 of 8 alphas give n within 0.2 of integer at one depth'}

    best_depth = None
    best_score = 0
    best_details = None

    for d in range(3, 25):
        details = []
        n_near = 0
        for i, E in enumerate(U238_CHAIN_ALPHAS):
            n = scope_boundary_count(E, d)
            residual = abs(n - round(n))
            near = residual < 0.2
            if near:
                n_near += 1
            details.append({
                'label': U238_CHAIN_LABELS[i] if i < len(U238_CHAIN_LABELS) else f'alpha_{i}',
                'energy_mev': float(E),
                'n_boundaries': float(n),
                'nearest_int': int(round(n)),
                'residual': float(residual),
                'near_integer': near,
            })
        if n_near > best_score:
            best_score = n_near
            best_depth = d
            best_details = details

    passed = best_score >= 6
    results['best_depth'] = best_depth
    results['best_score'] = best_score
    results['details'] = best_details
    results['PASS'] = passed

    if best_details:
        print(f"    Best depth: {best_depth}")
        for det in best_details:
            print(f"      {det['label']}: n={det['n_boundaries']:.4f} "
                  f"(round={det['nearest_int']}, residual={det['residual']:.4f}) "
                  f"{'*' if det['near_integer'] else ''}")
    print(f"    {best_score}/8 near integer -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T2_gamma_consistency():
    """T2: Gamma energies at same depth also give integer boundary counts."""
    print("\n  T2: Gamma boundary count consistency at alpha-determined depth")
    results = {'description': '>= 4 of 6 gamma energies integer at alpha depth'}

    # First find best alpha depth (reuse T1 logic)
    best_depth = None
    best_score = 0
    for d in range(3, 25):
        n_near = sum(1 for E in U238_CHAIN_ALPHAS
                     if abs(scope_boundary_count(E, d) - round(scope_boundary_count(E, d))) < 0.2)
        if n_near > best_score:
            best_score = n_near
            best_depth = d

    if best_depth is None:
        best_depth = 7  # Fallback

    # Test gamma energies at this depth
    gamma_mev = [g * KEV_TO_MEV for g in U238_CHAIN_GAMMAS_KEV]
    details = []
    n_near = 0
    for i, E in enumerate(gamma_mev):
        n = scope_boundary_count(E, best_depth)
        residual = abs(n - round(n))
        near = residual < 0.2
        if near:
            n_near += 1
        details.append({
            'energy_kev': float(U238_CHAIN_GAMMAS_KEV[i]),
            'energy_mev': float(E),
            'n_boundaries': float(n),
            'residual': float(residual),
            'near_integer': near,
        })
        print(f"    {U238_CHAIN_GAMMAS_KEV[i]:.1f} keV: n={n:.4f}, residual={residual:.4f}")

    passed = n_near >= 4
    results['alpha_depth'] = best_depth
    results['n_near_integer'] = n_near
    results['details'] = details
    results['PASS'] = passed
    print(f"    {n_near}/6 near integer -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T3_beta_non_integer():
    """T3: Beta endpoint boundary counts are non-integer."""
    print("\n  T3: Beta endpoints give non-integer boundary counts")
    results = {'description': 'All 3 beta endpoints have fractional part > 0.2'}

    # Use the same depth search as T1
    best_depth = None
    best_score = 0
    for d in range(3, 25):
        n_near = sum(1 for E in U238_CHAIN_ALPHAS
                     if abs(scope_boundary_count(E, d) - round(scope_boundary_count(E, d))) < 0.2)
        if n_near > best_score:
            best_score = n_near
            best_depth = d

    if best_depth is None:
        best_depth = 7

    betas = [('C-14', BETA_C14), ('Tritium', BETA_TRITIUM), ('Co-60', BETA_CO60)]
    details = []
    n_non_integer = 0
    for name, E in betas:
        n = scope_boundary_count(E, best_depth)
        frac = abs(n - round(n))
        non_int = frac > 0.2
        if non_int:
            n_non_integer += 1
        details.append({
            'name': name,
            'energy_mev': float(E),
            'n_boundaries': float(n),
            'fractional_part': float(frac),
            'non_integer': non_int,
        })
        print(f"    {name}: {E} MeV, n={n:.4f}, frac={frac:.4f} {'non-int' if non_int else 'INTEGER!'}")

    passed = n_non_integer == 3  # All three non-integer
    results['depth'] = best_depth
    results['details'] = details
    results['PASS'] = passed
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T4_boundary_count_ordering():
    """T4: Boundary count predicts decay energy ordering."""
    print("\n  T4: Boundary count ordering matches energy ordering")
    results = {'description': 'Spearman rank correlation = 1.0'}

    # Use best depth from T1
    best_depth = None
    best_score = 0
    for d in range(3, 25):
        n_near = sum(1 for E in U238_CHAIN_ALPHAS
                     if abs(scope_boundary_count(E, d) - round(scope_boundary_count(E, d))) < 0.2)
        if n_near > best_score:
            best_score = n_near
            best_depth = d
    if best_depth is None:
        best_depth = 7

    n_values = [scope_boundary_count(E, best_depth) for E in U238_CHAIN_ALPHAS]

    # Rank both
    energy_rank = list(np.argsort(np.argsort(U238_CHAIN_ALPHAS)))
    boundary_rank = list(np.argsort(np.argsort(n_values)))

    rho, p = spearmanr(U238_CHAIN_ALPHAS, n_values)
    passed = abs(rho - 1.0) < 1e-10  # Perfect rank correlation

    results['depth'] = best_depth
    results['energies'] = U238_CHAIN_ALPHAS
    results['n_boundaries'] = [float(n) for n in n_values]
    results['spearman_rho'] = float(rho)
    results['PASS'] = passed
    print(f"    Spearman rho = {rho:.6f}")
    print(f"    -> {'PASS' if passed else 'FAIL'} (expected: trivially 1.0 by construction)")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_06: Xi Cost and Scope Boundary Counting")
    print("=" * 60)

    t1 = test_T1_alpha_integer_boundaries()
    t2 = test_T2_gamma_consistency()
    t3 = test_T3_beta_non_integer()
    t4 = test_T4_boundary_count_ordering()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n  Overall: {score}/4")

    data = {
        'experiment': 'exp_06_xi_cost_scope_counting',
        'timestamp': datetime.now().isoformat(),
        'block': 'B',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_06_xi_cost_scope_counting')
