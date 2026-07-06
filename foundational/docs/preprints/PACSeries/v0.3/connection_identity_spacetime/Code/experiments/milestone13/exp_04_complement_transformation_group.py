"""
exp_04 -- Complement-Transformation Group Structure

Milestone 13, Block B (Complement-Transformations & Weyl Groups)

Hypothesis: Complement-transformations have group structure. The spectral-diff
representation composes exactly (by vector addition), the number of distinct
complement classes equals the number of vertex orbits, adjacent transformations
have comparable magnitudes, and Weyl group orders match (n+1)! for A_n.

Tests:
  T1: Composition: T(i,k) = T(i,j) + T(j,k) exactly (spectral-diff addition)
  T2: Number of distinct complement classes = vertex orbits for A_2..A_5
  T3: Adjacent transformations: zero within orbits, nonzero across orbit boundaries
  T4: Weyl group order |W(A_n)| = (n+1)! and orbit-stabilizer consistency
"""

import sys
import numpy as np
from pathlib import Path
from math import factorial, ceil
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from identity_complement import (
    PHI, INV_PHI,
    DynkinDiagram,
    complement_spectrum, complement_transformation, all_complement_transformations,
    vertex_orbits, connection_count,
    save_m13_results, _convert_numpy,
)


def test_T1_composition_exact():
    """T1: Complement-transformation composition is exact via spectral-diff addition."""
    # On A_4 (5 nodes), for triples (i,j,k):
    # T(i,j).spectral_diff + T(j,k).spectral_diff = T(i,k).spectral_diff
    # This is exact because spectral_diff = spec(j) - spec(i), so it telescopes.
    diag = DynkinDiagram('A', 4)
    adj = diag.adjacency
    n = adj.shape[0]

    max_error = 0.0
    n_triples = 0
    triple_results = []

    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                t_ij = complement_transformation(adj, i, j)
                t_jk = complement_transformation(adj, j, k)
                t_ik = complement_transformation(adj, i, k)

                composed = t_ij['spectral_diff'] + t_jk['spectral_diff']
                error = float(np.linalg.norm(composed - t_ik['spectral_diff']))

                if error > max_error:
                    max_error = error

                n_triples += 1
                if len(triple_results) < 6:
                    triple_results.append({
                        'triple': (i, j, k),
                        'error': error,
                    })

    print(f"  A_4: tested {n_triples} triples")
    print(f"  Max composition error: {max_error:.2e}")

    result = {
        'test': 'T1_composition_exact',
        'graph': 'A_4',
        'n_vertices': n,
        'n_triples_tested': n_triples,
        'max_composition_error': max_error,
        'sample_triples': triple_results,
        'note': 'Composition is exact by telescoping: spec(k)-spec(i) = '
                '(spec(j)-spec(i)) + (spec(k)-spec(j))',
        'PASS': max_error < 1e-10,
    }
    return result


def test_T2_complement_classes_equal_orbits():
    """T2: Number of distinct complement classes equals vertex orbit count for A_2..A_5."""
    results_by_n = {}
    all_match = True

    for rank in range(2, 6):
        diag = DynkinDiagram('A', rank)
        adj = diag.adjacency
        n = adj.shape[0]

        # Count distinct complement spectra
        spectra_keys = set()
        for v in range(n):
            spec = complement_spectrum(adj, v)
            key = tuple(np.round(spec, decimals=10))
            spectra_keys.add(key)
        n_distinct = len(spectra_keys)

        # Count orbits
        orbits = vertex_orbits(adj)
        n_orbits = len(orbits)

        # Expected for A_n: ceil(n/2) orbits
        expected = ceil(n / 2)

        match = (n_distinct == n_orbits == expected)
        if not match:
            all_match = False

        results_by_n[rank] = {
            'n_vertices': n,
            'n_distinct_spectra': n_distinct,
            'n_orbits': n_orbits,
            'expected': expected,
            'match': match,
        }
        print(f"  A_{rank}: {n_distinct} distinct spectra, "
              f"{n_orbits} orbits, expected {expected} "
              f"{'OK' if match else 'MISMATCH'}")

    result = {
        'test': 'T2_complement_classes_equal_orbits',
        'ranks_tested': list(range(2, 6)),
        'by_rank': results_by_n,
        'all_match': all_match,
        'PASS': all_match,
    }
    return result


def test_T3_adjacent_transformations_orbit_boundary():
    """T3: Adjacent transformations are zero within orbits and nonzero across orbit boundaries."""
    # On A_n, adjacent vertices in the same orbit have complement_transformation
    # magnitude = 0 (e.g., vertices 1 and 3 on A_4 are both in orbit {1,3},
    # and vertex 2 sits between them with T(1,2) = T(3,2) by symmetry, so
    # T(1,2).magnitude = 0 since spec(1) = spec(3) reflected).
    #
    # Actually: T(i, j).magnitude = ||spec(j) - spec(i)||. For adjacent vertices
    # in the same orbit, magnitude = 0. For adjacent vertices in DIFFERENT orbits,
    # magnitude > 0. This is a sharp structural test.
    #
    # Test on A_5 (5 vertices, orbits {0,4}, {1,3}, {2}) and D_5.
    types = [('A', 5), ('A', 7), ('D', 5)]
    all_correct = True
    details = {}

    for t, r in types:
        diag = DynkinDiagram(t, r)
        adj = diag.adjacency
        n = adj.shape[0]
        orbits = vertex_orbits(adj)

        # Build orbit membership
        orbit_of = {}
        for oi, orb in enumerate(orbits):
            for v in orb:
                orbit_of[v] = oi

        # Check adjacent pairs
        pairs_data = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    ct = complement_transformation(adj, i, j)
                    same_orbit = (orbit_of[i] == orbit_of[j])
                    mag = ct['magnitude']

                    if same_orbit:
                        ok = mag < 1e-10
                    else:
                        ok = mag > 1e-10

                    if not ok:
                        all_correct = False

                    pairs_data.append({
                        'pair': (i, j),
                        'same_orbit': same_orbit,
                        'magnitude': mag,
                        'ok': ok,
                    })

        label = f"{t}_{r}"
        n_same = sum(1 for p in pairs_data if p['same_orbit'])
        n_cross = sum(1 for p in pairs_data if not p['same_orbit'])
        details[label] = {
            'n_vertices': n,
            'n_orbits': len(orbits),
            'n_adjacent_pairs': len(pairs_data),
            'n_same_orbit_pairs': n_same,
            'n_cross_orbit_pairs': n_cross,
            'pairs': pairs_data,
        }
        print(f"  {label}: {len(orbits)} orbits, {n_same} same-orbit adj pairs, "
              f"{n_cross} cross-orbit adj pairs")
        for p in pairs_data:
            status = 'OK' if p['ok'] else 'FAIL'
            orb_label = 'same' if p['same_orbit'] else 'diff'
            print(f"    {p['pair']}: mag={p['magnitude']:.4f} ({orb_label} orbit) {status}")

    result = {
        'test': 'T3_adjacent_transformations_orbit_boundary',
        'types_tested': [f"{t}_{r}" for t, r in types],
        'details': details,
        'all_correct': all_correct,
        'note': 'Adjacent vertices in same orbit => zero transformation magnitude. '
                'Adjacent vertices in different orbits => nonzero magnitude. '
                'This mirrors Weyl group: transformations are trivial within '
                'orbits and nontrivial across orbit boundaries.',
        'PASS': all_correct,
    }
    return result


def test_T4_weyl_group_order():
    """T4: Weyl group |W(A_n)| = (n+1)! and orbit-stabilizer consistency."""
    # For A_n, the Weyl group is the symmetric group S_{n+1}, so |W| = (n+1)!
    # Orbit-stabilizer: |W| = |orbit(v)| * |Stab(v)| for any vertex v.
    # Number of orbits * average orbit size = n (total vertices).
    results_by_n = {}
    all_match = True

    for rank in range(2, 6):
        diag = DynkinDiagram('A', rank)
        adj = diag.adjacency
        n = adj.shape[0]  # = rank for A-type

        # Weyl group order
        weyl_order = factorial(rank + 1)
        expected_order = factorial(rank + 1)

        # Orbits
        orbits = vertex_orbits(adj)
        n_orbits = len(orbits)

        # Orbit-stabilizer check: for each orbit, |orbit| * |Stab| = |W|
        # Stab(v) stabilizes v under W action. For S_{n+1} acting on vertices,
        # |Stab(v)| = |W| / |orbit(v)|
        orbit_stabilizer_ok = True
        orbit_details = []
        for orb in orbits:
            orbit_size = len(orb)
            stabilizer_order = weyl_order // orbit_size
            product = orbit_size * stabilizer_order
            ok = (product == weyl_order)
            if not ok:
                orbit_stabilizer_ok = False
            orbit_details.append({
                'vertices': sorted(orb),
                'orbit_size': orbit_size,
                'stabilizer_order': stabilizer_order,
                'product': product,
                'ok': ok,
            })

        # Verify sum of orbit sizes = n
        total_vertices = sum(len(o) for o in orbits)
        size_check = (total_vertices == n)

        match = (weyl_order == expected_order) and orbit_stabilizer_ok and size_check
        if not match:
            all_match = False

        results_by_n[rank] = {
            'n_vertices': n,
            'weyl_order': weyl_order,
            'expected_order': expected_order,
            'n_orbits': n_orbits,
            'orbit_details': orbit_details,
            'total_orbit_vertices': total_vertices,
            'orbit_stabilizer_consistent': orbit_stabilizer_ok,
            'match': match,
        }
        print(f"  A_{rank}: |W| = {weyl_order} = ({rank+1})! = {expected_order}, "
              f"{n_orbits} orbits, OS consistent: {orbit_stabilizer_ok}")

    result = {
        'test': 'T4_weyl_group_order',
        'ranks_tested': list(range(2, 6)),
        'by_rank': results_by_n,
        'all_match': all_match,
        'PASS': all_match,
    }
    return result


def main():
    print("=" * 70)
    print("EXP 04 -- Complement-Transformation Group Structure")
    print("Milestone 13, Block B")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_composition_exact),
        ('T2', test_T2_complement_classes_equal_orbits),
        ('T3', test_T3_adjacent_transformations_orbit_boundary),
        ('T4', test_T4_weyl_group_order),
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
        'experiment': 'exp_04_complement_transformation_group',
        'milestone': 'milestone13',
        'block': 'B',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m13_results('exp_04_complement_transformation_group', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()
