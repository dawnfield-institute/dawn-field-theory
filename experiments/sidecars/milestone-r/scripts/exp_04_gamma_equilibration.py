"""
exp_04 -- Gamma Emission as Post-Severance Equilibration

Milestone R, Block B (Spectrum Reconstruction)

Hypothesis: Gamma rays are emitted when a daughter nucleus (post-severance)
relaxes from its excited to ground PAC state. The gamma energy equals the
spectral energy difference between excited and equilibrium configurations.

Tests:
  T1: Equilibration energy is always non-negative (SEC)
  T2: Co-60 gamma ratio 1.332/1.173 in ADE equilibration spectrum
  T3: Multi-step equilibration produces >= 2 distinct energy releases
  T4: Line width monotonically correlates with perturbation magnitude
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE,
    ledger_severance, equilibration_energy,
    line_width_from_disequilibrium,
    ade_graphs, vertex_orbits,
    GAMMA_CO60,
    save_mr_results,
)


def test_T1_equilibration_nonnegative():
    """T1: Equilibration energy >= 0 for all (graph, vertex) pairs."""
    print("\n  T1: Equilibration energy >= 0 (SEC: entropy increases)")
    results = {'description': 'Equilibration energy non-negative for all ADE graphs'}

    all_nonneg = True
    details = []
    for name, adj in ade_graphs(max_rank=8):
        n = adj.shape[0]
        for v in range(n):
            e_eq = equilibration_energy(adj, v)
            if e_eq < -1e-10:  # Allow tiny numerical noise
                all_nonneg = False
                details.append({'graph': name, 'vertex': v, 'energy': float(e_eq), 'negative': True})

    n_tested = sum(adj.shape[0] for _, adj in ade_graphs(max_rank=8))
    results['n_tested'] = n_tested
    results['negative_cases'] = details
    results['PASS'] = all_nonneg
    print(f"    Tested {n_tested} (graph, vertex) pairs")
    print(f"    Negative cases: {len(details)}")
    print(f"    -> {'PASS' if all_nonneg else 'FAIL'}")
    return results


def test_T2_co60_gamma_ratio():
    """T2: Co-60 gamma ratio in ADE equilibration spectrum."""
    print("\n  T2: Co-60 gamma ratio 1.332/1.173 = 1.1355 in ADE spectra")
    results = {'description': 'ADE graph produces equilibration ratio within 5% of Co-60'}

    target_ratio = GAMMA_CO60[1] / GAMMA_CO60[0]  # 1.1355
    best_error = float('inf')
    best_graph = None

    details = []
    for name, adj in ade_graphs(max_rank=8):
        n = adj.shape[0]
        eq_energies = []
        for v in range(n):
            e = equilibration_energy(adj, v)
            if abs(e) > 1e-10:
                eq_energies.append(abs(e))

        if len(eq_energies) < 2:
            continue

        eq_energies = sorted(set(round(e, 8) for e in eq_energies))
        if len(eq_energies) < 2:
            continue

        # Check all pairwise ratios
        for i in range(len(eq_energies)):
            for j in range(i + 1, len(eq_energies)):
                r = eq_energies[j] / eq_energies[i] if eq_energies[i] > 0 else 0
                error = abs(r - target_ratio) / target_ratio
                if error < best_error:
                    best_error = error
                    best_graph = name
                details.append({
                    'graph': name,
                    'ratio': float(r),
                    'error': float(error),
                })

    passed = best_error < 0.05  # Within 5%
    results['target_ratio'] = float(target_ratio)
    results['best_error'] = float(best_error)
    results['best_graph'] = best_graph
    results['n_ratios_checked'] = len(details)
    results['PASS'] = passed
    print(f"    Target: {target_ratio:.4f}, Best match: {best_graph} (error={best_error:.1%})")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T3_multistep_equilibration():
    """T3: Multi-step equilibration produces >= 2 distinct energy releases."""
    print("\n  T3: Multi-step equilibration produces multiple energy releases")
    results = {'description': 'Graphs with >= 3 orbits produce >= 2 distinct releases'}

    all_pass = True
    details = []
    for name, adj in ade_graphs(max_rank=8):
        orbits = vertex_orbits(adj)
        if len(orbits) < 3:
            continue

        n = adj.shape[0]
        # Sever first vertex in each orbit, track multi-step relaxation
        v = orbits[0][0]
        sev = ledger_severance(adj, v)
        daughter = sev['daughter_adj']
        nd = daughter.shape[0]
        if nd < 2:
            continue

        # Multi-step: partial equilibration then full
        state = np.ones(nd) / nd
        energy_releases = []

        for phase in range(3):
            # Run 20 steps
            entropy_before = -np.sum(state * np.log(state + 1e-30))
            for _ in range(20):
                new_state = np.zeros(nd)
                for u in range(nd):
                    neighbors = np.where(daughter[u] > 0)[0]
                    if len(neighbors) > 0:
                        share = state[u] * INV_PHI / len(neighbors)
                        new_state[u] += state[u] * (1 - INV_PHI)
                        for w in neighbors:
                            new_state[w] += share
                    else:
                        new_state[u] += state[u]
                state = new_state

            entropy_after = -np.sum(state * np.log(state + 1e-30))
            release = entropy_after - entropy_before
            if abs(release) > 1e-10:
                energy_releases.append(float(release))

        n_distinct = len(set(round(e, 6) for e in energy_releases))
        ok = n_distinct >= 2
        if not ok:
            all_pass = False

        details.append({
            'graph': name,
            'n_orbits': len(orbits),
            'energy_releases': energy_releases,
            'n_distinct': n_distinct,
            'pass': ok,
        })
        print(f"    {name}: {len(orbits)} orbits, {n_distinct} distinct releases")

    results['details'] = details
    results['PASS'] = all_pass
    print(f"    -> {'PASS' if all_pass else 'FAIL'}")
    return results


def test_T4_line_width_disequilibrium():
    """T4: Line width correlates monotonically with perturbation magnitude."""
    print("\n  T4: Line width vs disequilibrium (Spearman rho > 0.9)")
    results = {'description': 'Monotonic correlation between perturbation and line variance'}

    # Use D_4 as test graph (non-trivial automorphisms)
    for name, adj in ade_graphs(max_rank=8):
        if name == 'D_4':
            break

    fracs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    variances = []
    for f in fracs:
        lw = line_width_from_disequilibrium(adj, vertex=0, disequilibrium_frac=f,
                                              n_trials=500, seed=42)
        variances.append(lw['variance'])
        print(f"    frac={f:.3f}: variance={lw['variance']:.6f}")

    rho, p = spearmanr(fracs, variances)
    passed = rho > 0.9
    results['fracs'] = fracs
    results['variances'] = variances
    results['spearman_rho'] = float(rho)
    results['p_value'] = float(p)
    results['PASS'] = passed
    print(f"    Spearman rho = {rho:.4f} (p={p:.2e})")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_04: Gamma Emission as Post-Severance Equilibration")
    print("=" * 60)

    t1 = test_T1_equilibration_nonnegative()
    t2 = test_T2_co60_gamma_ratio()
    t3 = test_T3_multistep_equilibration()
    t4 = test_T4_line_width_disequilibrium()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n  Overall: {score}/4")

    data = {
        'experiment': 'exp_04_gamma_equilibration',
        'timestamp': datetime.now().isoformat(),
        'block': 'B',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_04_gamma_equilibration')
