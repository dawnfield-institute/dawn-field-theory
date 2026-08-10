"""
exp_07 -- Bubble Ejection Analogy (0g CO2)

Milestone R, Block C (Novel Physics Connections)

Hypothesis: Nucleon ejection shares mathematical structure with CO2 bubble
ejection from carbonated water in 0g: both isotropic, stochastic, driven
by internal energy differential (not gravity), governed by Fibonacci depth.
The "pressure not temperature" mechanism maps to PAC rebalancing.

Tests:
  T1: Isotropy from graph automorphism (orbit vertices = identical P_eject)
  T2: Stochastic ejection timing follows exponential distribution
  T3: Equilibrium-shift mechanism >= 2x more efficient than brute-force
  T4: Gravity is a spectator at nuclear depths: phi^(-178) < 1e-37
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE, LN_PHI,
    ledger_severance, ejection_probability,
    pressure_vs_temperature,
    StochasticCascade,
    build_pac_tree,
    ade_graphs, vertex_orbits,
    DEPTH_EM, DEPTH_GRAVITY,
    save_mr_results,
)


def test_T1_isotropy():
    """T1: Isotropy from graph automorphism."""
    print("\n  T1: Isotropy -- orbit vertices have identical ejection probability")
    results = {'description': 'Same-orbit vertices give identical severance energy (machine precision)'}

    all_isotropic = True
    details = []
    for name, adj in ade_graphs(max_rank=8):
        orbits = vertex_orbits(adj)
        for oi, orbit in enumerate(orbits):
            if len(orbit) < 2:
                continue
            energies = []
            for v in orbit:
                sev = ledger_severance(adj, v)
                energies.append(sev['spectral_shift'])

            spread = max(energies) - min(energies)
            isotropic = spread < 1e-10
            if not isotropic:
                all_isotropic = False

            details.append({
                'graph': name,
                'orbit': oi,
                'orbit_size': len(orbit),
                'energy_spread': float(spread),
                'isotropic': isotropic,
            })

        # Report D_4 specifically (the key case)
        if name == 'D_4':
            hub_orbit = [o for o in orbits if len(o) == 1]
            leaf_orbit = [o for o in orbits if len(o) > 1]
            if hub_orbit and leaf_orbit:
                hub_e = ledger_severance(adj, hub_orbit[0][0])['spectral_shift']
                leaf_e = ledger_severance(adj, leaf_orbit[0][0])['spectral_shift']
                diff_pct = abs(hub_e - leaf_e) / max(abs(hub_e), abs(leaf_e)) * 100
                print(f"    D_4: hub energy={hub_e:.4f}, leaf energy={leaf_e:.4f} ({diff_pct:.1f}% different)")

    results['details'] = details
    results['PASS'] = all_isotropic
    print(f"    All orbits isotropic: {all_isotropic}")
    print(f"    -> {'PASS' if all_isotropic else 'FAIL'}")
    return results


def test_T2_exponential_timing():
    """T2: Stochastic ejection timing follows exponential distribution."""
    print("\n  T2: Ejection timing follows exponential (Poisson process)")
    results = {'description': 'Anderson-Darling test for exponential, p > 0.05'}

    ejection_times = []
    threshold = 0.5  # Ejection when any node exceeds this fraction of initial

    for trial in range(5000):
        cascade = StochasticCascade(n_levels=20, seed=trial * 7 + 13)
        fwd, _ = cascade.run_forward(initial_value=1.0, noise_amplitude=0.05)

        # "Ejection time" = first level where value drops below threshold
        # (energy transferred to radiation exceeds threshold)
        ejection_step = len(fwd)
        for step, val in enumerate(fwd):
            if step > 0 and abs(val) < threshold * abs(fwd[0]):
                ejection_step = step
                break
        ejection_times.append(ejection_step)

    ejection_times = np.array(ejection_times, dtype=float)
    # Remove any that never ejected (hit max steps)
    ejected = ejection_times[ejection_times < 20]

    if len(ejected) > 100:
        # Anderson-Darling test for exponential
        ad_stat, critical_values, significance_levels = stats.anderson(ejected, dist='expon')
        # p > 0.05 means we can't reject exponential
        # Anderson-Darling: pass if stat < critical value at 5%
        passed = ad_stat < critical_values[2]  # Index 2 = 5% significance

        results['n_ejected'] = int(len(ejected))
        results['n_total'] = 5000
        results['mean_time'] = float(np.mean(ejected))
        results['ad_statistic'] = float(ad_stat)
        results['critical_5pct'] = float(critical_values[2])
        results['PASS'] = passed
        print(f"    {len(ejected)}/5000 ejected, mean time = {np.mean(ejected):.1f}")
        print(f"    Anderson-Darling stat = {ad_stat:.4f} (critical 5% = {critical_values[2]:.4f})")
    else:
        results['n_ejected'] = int(len(ejected))
        results['PASS'] = False
        passed = False
        print(f"    Only {len(ejected)} ejected -- insufficient data")

    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T3_pressure_efficiency():
    """T3: Equilibrium-shift >= 2x more efficient than brute-force."""
    print("\n  T3: Pressure (equilibrium shift) vs temperature (brute force)")
    results = {'description': 'Efficiency ratio >= 2.0 for nuclear depths'}

    ratios = []
    for d in [5, 6, 7, 8, 10, 13]:
        # Equilibrium shift of 0.1 (10% shift in PAC split ratio)
        pv = pressure_vs_temperature(d, delta_equilibrium=0.1)
        ratios.append(pv['efficiency_ratio'])
        print(f"    depth={d}: shift={pv['e_shift_mev']:.3e}, "
              f"thermal={pv['e_thermal_mev']:.3e}, ratio={pv['efficiency_ratio']:.2f}")

    mean_ratio = np.mean(ratios)
    all_above_2 = all(r >= 2.0 for r in ratios)
    passed = all_above_2

    results['ratios'] = [float(r) for r in ratios]
    results['mean_ratio'] = float(mean_ratio)
    results['all_above_2'] = all_above_2
    results['PASS'] = passed
    print(f"    Mean ratio: {mean_ratio:.2f}, All >= 2: {all_above_2}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T4_gravity_spectator():
    """T4: Gravity is a spectator at nuclear depths."""
    print("\n  T4: phi^(-178) confirms gravity irrelevant at nuclear scales")
    results = {'description': 'phi^(-(DEPTH_GRAVITY - nuclear_depth)) < 1e-37'}

    nuclear_depth = 5  # Strong force approximate depth
    gravity_depth = DEPTH_GRAVITY  # Should be 183

    depth_difference = gravity_depth - nuclear_depth
    ratio = PHI ** (-depth_difference)

    # Also compute the coupling ratio
    gravity_coupling = PHI ** (-gravity_depth)
    nuclear_coupling = PHI ** (-nuclear_depth)
    coupling_ratio = gravity_coupling / nuclear_coupling

    passed = ratio < 1e-37
    results['nuclear_depth'] = nuclear_depth
    results['gravity_depth'] = int(gravity_depth)
    results['depth_difference'] = int(depth_difference)
    results['ratio'] = float(ratio)
    results['log10_ratio'] = float(np.log10(ratio)) if ratio > 0 else float('-inf')
    results['coupling_ratio'] = float(coupling_ratio)
    results['PASS'] = passed
    print(f"    Gravity depth: {gravity_depth}, Nuclear depth: {nuclear_depth}")
    print(f"    phi^(-{depth_difference}) = {ratio:.3e}")
    print(f"    log10(ratio) = {np.log10(ratio):.1f}")
    print(f"    -> {'PASS' if passed else 'FAIL'} (gravity is {abs(np.log10(ratio)):.0f} orders weaker)")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_07: Bubble Ejection Analogy (0g CO2)")
    print("=" * 60)

    t1 = test_T1_isotropy()
    t2 = test_T2_exponential_timing()
    t3 = test_T3_pressure_efficiency()
    t4 = test_T4_gravity_spectator()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n  Overall: {score}/4")

    data = {
        'experiment': 'exp_07_bubble_ejection_analogy',
        'timestamp': datetime.now().isoformat(),
        'block': 'C',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_07_bubble_ejection_analogy')
