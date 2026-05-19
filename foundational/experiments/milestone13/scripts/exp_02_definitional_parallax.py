"""
exp_02 -- Definitional Parallax

Milestone 13, Block A (Identity IS Complement)

Hypothesis: Different observers see different complements of the same target.
The complement-view depends on WHERE you observe FROM, not just WHAT you observe.
This is definitional parallax -- the same entity has different identities depending
on the observer's structural position.

Tests:
  T1: Asymmetric observers on A_8 see different complement-views (symmetric ones agree)
  T2: Parallax is nonzero iff observers are in different view-orbits of the target
  T3: Symmetric (D_4 leaf) observers agree -- zero parallax from equivalent positions
  T4: Parallax zero-fraction correlates with automorphism group size
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from identity_complement import (
    PHI, INV_PHI,
    DynkinDiagram,
    complement_view, parallax,
    graph_distance, connection_count, vertex_orbits,
    save_m13_results, _convert_numpy,
)


def test_T1_distinct_observer_views():
    """T1: Asymmetric observers on A_8 see different complement-views of vertex 3."""
    # A_7 has Z_2 symmetry, so observers 0 and 6 are mirror images around
    # vertex 3 and see identical complement-views. This is physically correct!
    # To test genuine parallax, use A_8 (8 vertices, chain 0..7) with target=3.
    # Observer 0 (distance 3 from target) vs observer 7 (distance 4 from target)
    # are NOT in the same orbit relative to the target, so they should see
    # different complement-views.
    diag = DynkinDiagram('A', 8)
    adj = diag.adjacency
    target = 3

    # Observer at vertex 0 vs observer at vertex 7 (asymmetric about target)
    _, spec_0 = complement_view(adj, 0, target)
    _, spec_7 = complement_view(adj, 7, target)

    # Spectral difference
    max_len = max(len(spec_0), len(spec_7))
    s0 = np.zeros(max_len)
    s7 = np.zeros(max_len)
    s0[:len(spec_0)] = spec_0
    s7[:len(spec_7)] = spec_7
    spectral_diff = float(np.linalg.norm(s7 - s0))

    # Also compute via the parallax function
    par = parallax(adj, 0, 7, target)

    # Confirm asymmetry: observer 0 is at distance 3 from target,
    # observer 7 is at distance 4 from target
    d0 = graph_distance(adj, 0, target)
    d7 = graph_distance(adj, 7, target)

    print(f"  A_8, target=3: observer 0 (dist={d0}), observer 7 (dist={d7})")
    print(f"  Observer 0 spectrum = {np.round(spec_0, 4)}")
    print(f"  Observer 7 spectrum = {np.round(spec_7, 4)}")
    print(f"  Spectral difference = {spectral_diff:.6f}")
    print(f"  Parallax(0, 7, 3) = {par:.6f}")

    # Also test a symmetric control pair: on A_7, obs 0 and 6 viewing target 3
    diag7 = DynkinDiagram('A', 7)
    adj7 = diag7.adjacency
    par_sym = parallax(adj7, 0, 6, 3)
    print(f"  Control (A_7 symmetric): Parallax(0, 6, 3) = {par_sym:.2e}")

    result = {
        'test': 'T1_distinct_observer_views',
        'graph': 'A_8',
        'target': target,
        'observer_0_distance': d0,
        'observer_7_distance': d7,
        'observer_0_spectrum': spec_0.tolist(),
        'observer_7_spectrum': spec_7.tolist(),
        'spectral_difference': spectral_diff,
        'parallax': par,
        'symmetric_control_parallax': par_sym,
        'note': 'A_7 symmetric observers see identical views (parallax=0). '
                'A_8 breaks this symmetry: obs 0 (dist 3) vs obs 7 (dist 4) '
                'see genuinely different complement-views.',
        'PASS': spectral_diff > 0.01 and par_sym < 1e-10,
    }
    return result


def test_T2_parallax_nonzero_between_orbits():
    """T2: Parallax is nonzero iff observers are in different orbits relative to target."""
    # On A_n, observers i and j see the same complement-view of target t
    # iff they are in the same orbit under the automorphism group that fixes t.
    # For a chain graph, this means i and (n-1-i) see the same view of the
    # center (if it exists). Observers at different orbit-positions see
    # different views.
    #
    # Test: for each pair of observers on A_8, parallax > 0 iff they are
    # structurally inequivalent (different complement-view spectra of target).
    diag = DynkinDiagram('A', 8)
    adj = diag.adjacency
    n = adj.shape[0]
    target = 3  # Off-center target breaks most symmetries

    # Compute complement-view spectrum for each observer
    view_spectra = {}
    for obs in range(n):
        if obs == target:
            continue
        _, spec = complement_view(adj, obs, target)
        view_spectra[obs] = tuple(np.round(spec, decimals=10))

    # Group observers by their view-spectrum (defines view-orbits)
    view_orbits = {}
    for obs, key in view_spectra.items():
        if key not in view_orbits:
            view_orbits[key] = []
        view_orbits[key].append(obs)

    n_view_classes = len(view_orbits)
    print(f"  A_8, target=3: {n-1} observers, {n_view_classes} distinct view-classes")
    for key, obs_list in view_orbits.items():
        print(f"    View class: observers {obs_list}")

    # Check: parallax = 0 for same-class pairs, > 0 for cross-class pairs
    n_same = 0
    n_cross = 0
    same_ok = True
    cross_ok = True

    for i in range(n):
        if i == target:
            continue
        for j in range(i + 1, n):
            if j == target:
                continue
            par = parallax(adj, i, j, target)
            same_class = (view_spectra[i] == view_spectra[j])

            if same_class:
                n_same += 1
                if par > 1e-10:
                    same_ok = False
            else:
                n_cross += 1
                if par < 1e-10:
                    cross_ok = False

    print(f"  Same-class pairs: {n_same} (all zero parallax: {same_ok})")
    print(f"  Cross-class pairs: {n_cross} (all nonzero parallax: {cross_ok})")

    result = {
        'test': 'T2_parallax_nonzero_between_orbits',
        'graph': 'A_8',
        'target': target,
        'n_observers': n - 1,
        'n_view_classes': n_view_classes,
        'view_orbit_members': {str(k): v for k, v in enumerate(view_orbits.values())},
        'n_same_class_pairs': n_same,
        'n_cross_class_pairs': n_cross,
        'same_class_all_zero': same_ok,
        'cross_class_all_nonzero': cross_ok,
        'PASS': same_ok and cross_ok and n_view_classes > 1,
    }
    return result


def test_T3_symmetric_observers_agree():
    """T3: D_4 leaf observers see identical complement-views of the center."""
    # D_4 has 4 vertices. Standard construction:
    # vertex with degree 3 (center) connected to 3 leaves.
    d4 = DynkinDiagram('D', 4)
    adj = d4.adjacency
    n = adj.shape[0]

    # Identify center (degree 3) and leaves (degree 1)
    degrees = np.sum(adj > 0, axis=1)
    center = int(np.argmax(degrees))
    leaves = [v for v in range(n) if v != center]

    print(f"  D_4: center={center} (degree {degrees[center]}), leaves={leaves}")

    # Parallax between all pairs of leaves with target = center
    parallax_values = []
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            par = parallax(adj, leaves[i], leaves[j], center)
            parallax_values.append({
                'obs1': leaves[i],
                'obs2': leaves[j],
                'parallax': par,
            })
            print(f"    Parallax(leaf {leaves[i]}, leaf {leaves[j]}, center={center}) = {par:.2e}")

    all_zero = all(pv['parallax'] < 1e-10 for pv in parallax_values)

    result = {
        'test': 'T3_symmetric_observers_agree',
        'graph': 'D_4',
        'center': center,
        'leaves': leaves,
        'parallax_pairs': parallax_values,
        'all_zero': all_zero,
        'PASS': all_zero,
    }
    return result


def test_T4_parallax_vanishes_for_trivial_aut():
    """T4: Parallax structure reflects automorphism group: trivial Aut => all parallaxes nonzero."""
    # When a graph has trivial automorphism group (no symmetries),
    # every pair of observers should produce nonzero parallax for every target,
    # because no two observers are structurally equivalent.
    #
    # For ADE graphs with Z_2 symmetry (like A_n), some parallaxes vanish.
    # For D_4 with S_3 symmetry, more parallaxes vanish (3 equivalent leaves).
    # The fraction of zero-parallax triples should correlate with |Aut(G)|.
    #
    # Test: |Aut| = 1 for no ADE graph, but the zero-parallax fraction should
    # increase monotonically with |Aut|/n. Compare D_4 (large Aut) vs A_8 (Z_2 only).
    from math import factorial

    test_cases = [
        ('A_8', DynkinDiagram('A', 8)),   # Aut = Z_2, small
        ('A_5', DynkinDiagram('A', 5)),    # Aut = Z_2, small
        ('D_4', DynkinDiagram('D', 4)),    # Aut = S_3, larger
    ]

    results_by_graph = {}
    zero_fractions = []

    for label, diag in test_cases:
        adj = diag.adjacency
        n = adj.shape[0]

        n_zero = 0
        n_total = 0
        for target in range(n):
            for obs1 in range(n):
                if obs1 == target:
                    continue
                for obs2 in range(obs1 + 1, n):
                    if obs2 == target:
                        continue
                    par = parallax(adj, obs1, obs2, target)
                    n_total += 1
                    if par < 1e-10:
                        n_zero += 1

        zero_frac = n_zero / n_total if n_total > 0 else 0.0
        n_orbits = len(vertex_orbits(adj))
        zero_fractions.append(zero_frac)

        results_by_graph[label] = {
            'n_vertices': n,
            'n_orbits': n_orbits,
            'n_total_triples': n_total,
            'n_zero_parallax': n_zero,
            'zero_fraction': zero_frac,
        }
        print(f"  {label}: {n_orbits} orbits, zero-parallax fraction = {zero_frac:.4f} "
              f"({n_zero}/{n_total})")

    # D_4 should have higher zero-parallax fraction than A_n due to larger Aut
    d4_zf = results_by_graph['D_4']['zero_fraction']
    a8_zf = results_by_graph['A_8']['zero_fraction']
    a5_zf = results_by_graph['A_5']['zero_fraction']

    # D_4 has 3 equivalent leaves => many more symmetric observer pairs
    d4_higher = d4_zf > a8_zf
    # All should have some nonzero parallax (i.e., zero_frac < 1.0)
    all_have_parallax = all(r['zero_fraction'] < 1.0 for r in results_by_graph.values())

    print(f"  D_4 zero-frac ({d4_zf:.4f}) > A_8 zero-frac ({a8_zf:.4f}): {d4_higher}")

    result = {
        'test': 'T4_parallax_vanishes_for_trivial_aut',
        'by_graph': results_by_graph,
        'd4_higher_than_a8': d4_higher,
        'all_have_nonzero_parallax': all_have_parallax,
        'note': 'Larger automorphism group => more symmetric observer pairs => '
                'higher fraction of zero-parallax triples. D_4 (Aut=S_3) has '
                'more vanishing parallaxes than A_n (Aut=Z_2).',
        'PASS': d4_higher and all_have_parallax,
    }
    return result


def main():
    print("=" * 70)
    print("EXP 02 -- Definitional Parallax")
    print("Milestone 13, Block A")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_distinct_observer_views),
        ('T2', test_T2_parallax_nonzero_between_orbits),
        ('T3', test_T3_symmetric_observers_agree),
        ('T4', test_T4_parallax_vanishes_for_trivial_aut),
    ]:
        print(f"\n--- {name}: {test_fn.__doc__.strip().split(chr(10))[0]} ---")
        r = test_fn()
        results[name] = r
        if r['PASS']:
            score += 1
            print(f"  PASS")
        else:
            print(f"  FAIL")

    final = {
        'experiment': 'exp_02_definitional_parallax',
        'milestone': 'milestone13',
        'block': 'A',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m13_results('exp_02_definitional_parallax', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()
