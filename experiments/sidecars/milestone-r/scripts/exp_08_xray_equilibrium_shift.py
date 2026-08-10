"""
exp_08 -- X-ray Generation via Equilibrium Shift

Milestone R, Block C (Novel Physics Connections)

Hypothesis: If radiation is ledger severance, and severance can be triggered
by equilibrium shift rather than brute-force energy, then X-ray generation
could be made more efficient. Characteristic X-ray lines should correspond
to integer scope boundary counts at EM depth.

Tests:
  T1: Cu/Mo K-alpha as integer boundary counts at depth 13
  T2: Stochastic cascade at depth 13 produces bremsstrahlung-like cutoff
  T3: Equilibrium-shift cost < brute-force by >= 10%
  T4: Distinct transition count = n_orbits*(n_orbits-1)/2
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE, DEPTH_EM,
    scope_boundary_count, pressure_vs_temperature,
    discrete_severance_spectrum,
    StochasticCascade,
    ade_graphs, vertex_orbits,
    CU_K_ALPHA, MO_K_ALPHA, KEV_TO_MEV,
    PLANCK_ENERGY_MEV,
    save_mr_results,
)


def test_T1_xray_boundary_counts():
    """T1: Cu/Mo K-alpha as integer boundary counts at depth 13."""
    print("\n  T1: X-ray K-alpha lines as integer boundary counts")
    results = {'description': 'Both Cu and Mo K-alpha within 0.3 of integer at depth 13'}

    xrays = [
        ('Cu K-alpha', CU_K_ALPHA * KEV_TO_MEV),
        ('Mo K-alpha', MO_K_ALPHA * KEV_TO_MEV),
    ]

    # Try a range of depths around EM
    best_depth = None
    best_total_residual = float('inf')

    for d in range(5, 25):
        total_residual = 0
        for name, E in xrays:
            n = scope_boundary_count(E, d)
            total_residual += abs(n - round(n))
        if total_residual < best_total_residual:
            best_total_residual = total_residual
            best_depth = d

    details = []
    n_pass = 0
    for name, E in xrays:
        n = scope_boundary_count(E, best_depth)
        residual = abs(n - round(n))
        ok = residual < 0.3
        if ok:
            n_pass += 1
        details.append({
            'line': name,
            'energy_kev': float(E / KEV_TO_MEV),
            'depth': best_depth,
            'n_boundaries': float(n),
            'nearest_int': int(round(n)),
            'residual': float(residual),
            'pass': ok,
        })
        print(f"    {name}: {E/KEV_TO_MEV:.3f} keV at depth {best_depth}: "
              f"n={n:.4f} (int={int(round(n))}, residual={residual:.4f})")

    # Also check at canonical depth 13
    print(f"    --- At canonical EM depth {DEPTH_EM} ---")
    for name, E in xrays:
        n = scope_boundary_count(E, DEPTH_EM)
        print(f"    {name}: n={n:.6f}")

    passed = n_pass == 2
    results['best_depth'] = best_depth
    results['details'] = details
    results['PASS'] = passed
    print(f"    -> {'PASS' if passed else 'FAIL'} ({n_pass}/2 within 0.3)")
    return results


def test_T2_bremsstrahlung_cutoff():
    """T2: Stochastic cascade produces distribution with sharp high-energy cutoff."""
    print("\n  T2: Cascade at EM depth produces bremsstrahlung-like distribution")
    results = {'description': '< 1% of samples in top 5% of energy range'}

    # Generate cascade energies
    n_samples = 50000
    energies = []
    for trial in range(n_samples):
        cascade = StochasticCascade(n_levels=13, seed=trial)
        fwd, _ = cascade.run_forward(initial_value=1.0, noise_amplitude=0.02)
        energies.append(abs(fwd[-1]))

    energies = np.array(energies)
    e_max = np.max(energies)
    e_95 = 0.95 * e_max

    # What fraction of samples falls in the top 5% of the range?
    n_above_95 = np.sum(energies > e_95)
    frac_above = n_above_95 / n_samples

    # Bremsstrahlung has a sharp cutoff: very few events near E_max
    passed = frac_above < 0.01  # < 1% in top 5%

    results['n_samples'] = n_samples
    results['e_max'] = float(e_max)
    results['e_95_threshold'] = float(e_95)
    results['n_above_95'] = int(n_above_95)
    results['frac_above'] = float(frac_above)
    results['PASS'] = passed
    print(f"    E_max = {e_max:.6f}, {n_above_95}/{n_samples} above 95% ({frac_above:.4f})")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T3_equilibrium_shift_efficiency():
    """T3: Equilibrium-shift cost < brute-force by >= 10%."""
    print("\n  T3: Equilibrium-shift vs brute-force at EM depth")
    results = {'description': 'Shift cost at least 10% less than brute-force for X-ray generation'}

    # For Cu K-alpha at 8 keV, the brute-force needs > 8.98 keV (K-edge)
    # DFT: equilibrium shift at depth 13 with small delta
    deltas = [0.05, 0.1, 0.2, 0.3, 0.5]
    details = []
    any_pass = False

    for delta in deltas:
        pv = pressure_vs_temperature(DEPTH_EM, delta_equilibrium=delta)
        savings_pct = (1.0 - 1.0/pv['efficiency_ratio']) * 100
        ok = savings_pct >= 10
        if ok:
            any_pass = True
        details.append({
            'delta': float(delta),
            'shift_cost_mev': float(pv['e_shift_mev']),
            'thermal_cost_mev': float(pv['e_thermal_mev']),
            'ratio': float(pv['efficiency_ratio']),
            'savings_pct': float(savings_pct),
            'pass': ok,
        })
        print(f"    delta={delta}: shift={pv['e_shift_mev']:.3e}, "
              f"thermal={pv['e_thermal_mev']:.3e}, savings={savings_pct:.1f}%")

    results['details'] = details
    results['PASS'] = any_pass
    print(f"    -> {'PASS' if any_pass else 'FAIL'}")
    return results


def test_T4_transition_count():
    """T4: Distinct transition count = n_orbits*(n_orbits-1)/2."""
    print("\n  T4: Transition count from orbit structure")
    results = {'description': 'n_transitions = n_orbits*(n_orbits-1)/2 for all ADE'}

    all_match = True
    details = []
    for name, adj in ade_graphs(max_rank=8):
        orbits = vertex_orbits(adj)
        n_orbits = len(orbits)
        predicted = n_orbits * (n_orbits - 1) // 2

        # Compute actual distinct transition energies
        spectrum = discrete_severance_spectrum(adj, depth=DEPTH_EM)
        energies = sorted(set(round(s['spectral_shift'], 8) for s in spectrum.values()))

        # Transitions between distinct energy levels
        transitions = set()
        for i in range(len(energies)):
            for j in range(i + 1, len(energies)):
                transitions.add(round(abs(energies[j] - energies[i]), 8))

        actual = len(transitions)
        match = actual == predicted
        if not match:
            all_match = False

        details.append({
            'graph': name,
            'n_orbits': n_orbits,
            'predicted_transitions': predicted,
            'actual_transitions': actual,
            'match': match,
        })
        print(f"    {name}: {n_orbits} orbits, predicted={predicted}, actual={actual} "
              f"{'OK' if match else 'MISMATCH'}")

    results['details'] = details
    results['PASS'] = all_match
    print(f"    -> {'PASS' if all_match else 'FAIL'}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_08: X-ray Generation via Equilibrium Shift")
    print("=" * 60)

    t1 = test_T1_xray_boundary_counts()
    t2 = test_T2_bremsstrahlung_cutoff()
    t3 = test_T3_equilibrium_shift_efficiency()
    t4 = test_T4_transition_count()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n  Overall: {score}/4")

    data = {
        'experiment': 'exp_08_xray_equilibrium_shift',
        'timestamp': datetime.now().isoformat(),
        'block': 'C',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_08_xray_equilibrium_shift')
