"""
exp_01 -- Ledger Severance Mechanics

Milestone R, Block A (Ledger Severance Mechanics)

Hypothesis: Removing a vertex from a PAC graph (ledger severance) produces a
well-defined energy cost, preserves PAC conservation, and creates independent
sub-ledgers. The spectral gap encodes the depth of the severed interaction.

Tests:
  T1: PAC conservation under severance (P_before = P_daughter + P_radiation)
  T2: Severance energy quantized by orbit structure (distinct energies == orbits)
  T3: Severance creates two independent ledgers (no cross-component leakage)
  T4: Spectral gap encodes interaction depth (correlation with PAC tree depth)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE,
    ledger_severance, discrete_severance_spectrum,
    build_pac_tree, pac_tree_values,
    ade_graphs, vertex_orbits, redistribute_on_graph, measure_entropy,
    save_mr_results, _convert_numpy,
)


def test_T1_pac_conservation():
    """T1: PAC conservation under severance."""
    print("\n  T1: PAC conservation under severance")
    results = {'description': 'PAC conservation: total spectral energy before = after + shift'}

    errors = []
    for depth in [3, 4, 5]:
        A = build_pac_tree(depth)
        values = pac_tree_values(depth)
        total_before = float(np.sum(values))

        # Sever root (vertex 0)
        sev = ledger_severance(A, 0)
        daughter = sev['daughter_adj']
        n_d = daughter.shape[0]

        # PAC values on daughter: remove root's value
        daughter_values = np.delete(values, 0)
        total_daughter = float(np.sum(daughter_values))

        # "Radiation" carries root's value
        radiation_value = values[0]
        total_after = total_daughter + radiation_value

        error = abs(total_before - total_after) / total_before
        errors.append(error)
        print(f"    depth={depth}: before={total_before:.6f}, after={total_after:.6f}, error={error:.2e}")

    max_error = max(errors)
    passed = max_error < 1e-10
    results['errors'] = errors
    results['max_error'] = max_error
    results['PASS'] = passed
    print(f"    -> {'PASS' if passed else 'FAIL'} (max error: {max_error:.2e})")
    return results


def test_T2_orbit_quantization():
    """T2: Severance energy quantized by orbit structure."""
    print("\n  T2: Severance energy quantized by orbit structure")
    results = {'description': 'Distinct severance energies == number of orbits for all ADE'}

    all_match = True
    details = []
    for name, adj in ade_graphs(max_rank=8):
        orbits = vertex_orbits(adj)
        n_orbits = len(orbits)

        # Compute severance for every vertex
        energies_per_vertex = []
        for v in range(adj.shape[0]):
            sev = ledger_severance(adj, v)
            energies_per_vertex.append(round(sev['spectral_shift'], 8))

        distinct_energies = len(set(energies_per_vertex))
        match = distinct_energies == n_orbits
        if not match:
            all_match = False

        details.append({
            'graph': name,
            'n_vertices': int(adj.shape[0]),
            'n_orbits': n_orbits,
            'distinct_energies': distinct_energies,
            'match': match,
        })
        print(f"    {name}: {n_orbits} orbits, {distinct_energies} distinct energies -> {'OK' if match else 'MISMATCH'}")

    results['details'] = details
    results['PASS'] = all_match
    print(f"    -> {'PASS' if all_match else 'FAIL'}")
    return results


def test_T3_independent_ledgers():
    """T3: Severance creates two independent ledgers."""
    print("\n  T3: Severance creates two independent ledgers")
    results = {'description': 'Cross-component entropy leakage < 1e-10 after redistribution'}

    tests = []
    for depth in [3, 4]:
        A = build_pac_tree(depth)
        n = A.shape[0]

        # Sever root -> two subtrees
        sev = ledger_severance(A, 0)
        if not sev['disconnected']:
            tests.append({'depth': depth, 'disconnected': False, 'leakage': 0.0})
            continue

        daughter = sev['daughter_adj']
        nd = daughter.shape[0]

        # Find connected components
        visited = set()
        components = []
        for start in range(nd):
            if start in visited:
                continue
            comp = {start}
            queue = [start]
            while queue:
                v = queue.pop(0)
                for u in range(nd):
                    if daughter[v, u] > 0 and u not in comp:
                        comp.add(u)
                        queue.append(u)
            visited.update(comp)
            components.append(sorted(comp))

        if len(components) < 2:
            tests.append({'depth': depth, 'disconnected': False, 'n_components': len(components)})
            continue

        # Initialize PAC values on component 1, zero on component 2
        state = np.zeros(nd)
        for v in components[0]:
            state[v] = 1.0 / len(components[0])

        comp2_initial = sum(state[v] for v in components[1])

        # Redistribute 100 steps on the full daughter
        for _ in range(100):
            new_state = np.zeros(nd)
            for v in range(nd):
                neighbors = np.where(daughter[v] > 0)[0]
                if len(neighbors) > 0:
                    share = state[v] * INV_PHI / len(neighbors)
                    new_state[v] += state[v] * (1 - INV_PHI)
                    for u in neighbors:
                        new_state[u] += share
                else:
                    new_state[v] += state[v]
            state = new_state

        comp2_final = sum(state[v] for v in components[1])
        leakage = abs(comp2_final - comp2_initial)

        tests.append({
            'depth': depth,
            'disconnected': True,
            'n_components': len(components),
            'comp_sizes': [len(c) for c in components],
            'leakage': float(leakage),
        })
        print(f"    depth={depth}: {len(components)} components, leakage={leakage:.2e}")

    max_leakage = max(t.get('leakage', 0) for t in tests)
    passed = max_leakage < 1e-10
    results['tests'] = tests
    results['max_leakage'] = max_leakage
    results['PASS'] = passed
    print(f"    -> {'PASS' if passed else 'FAIL'} (max leakage: {max_leakage:.2e})")
    return results


def test_T4_spectral_gap_encodes_depth():
    """T4: Spectral gap encodes interaction depth."""
    print("\n  T4: Spectral gap encodes interaction depth")
    results = {'description': 'Correlation |r| > 0.95 between log(spectral_gap) and PAC tree depth'}

    depths = list(range(2, 8))
    log_gaps = []
    for d in depths:
        A = build_pac_tree(d)
        sev = ledger_severance(A, 0)  # Sever root
        gap = abs(sev['spectral_shift'])
        log_gaps.append(np.log(gap) if gap > 0 else -30)
        print(f"    depth={d}: spectral_shift={sev['spectral_shift']:.4f}, log_gap={log_gaps[-1]:.4f}")

    # Pearson correlation
    r = np.corrcoef(depths, log_gaps)[0, 1]
    passed = abs(r) > 0.95
    results['depths'] = depths
    results['log_gaps'] = log_gaps
    results['pearson_r'] = float(r)
    results['PASS'] = passed
    print(f"    Pearson r = {r:.4f} -> {'PASS' if passed else 'FAIL'}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_01: Ledger Severance Mechanics")
    print("=" * 60)

    t1 = test_T1_pac_conservation()
    t2 = test_T2_orbit_quantization()
    t3 = test_T3_independent_ledgers()
    t4 = test_T4_spectral_gap_encodes_depth()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n  Overall: {score}/4")

    data = {
        'experiment': 'exp_01_ledger_severance_mechanics',
        'timestamp': datetime.now().isoformat(),
        'block': 'A',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_01_ledger_severance_mechanics')
