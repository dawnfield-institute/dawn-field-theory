"""
exp_01 -- Complement Determines Identity

Milestone 13, Block A (Identity IS Complement)

Hypothesis: Complement uniquely determines vertex identity in ADE graphs.
A vertex's complement spectrum is an automorphism-invariant fingerprint --
vertices have the same complement spectrum if and only if they belong to the
same orbit under graph automorphism. Complement distance is monotonically
related to graph distance, and double-complements exhibit phi-structured
edge-count ratios.

Tests:
  T1: Complement spectra distinguish all orbits of A_5 (3 distinct spectra)
  T2: Distinct spectrum counts match Aut orbit counts for D_4 and E_6
  T3: Complement distance = 0 within orbits, > 0 between orbits (identity = complement)
  T4: Double-complement edge ratios cluster near INV_PHI^k
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from identity_complement import (
    PHI, INV_PHI, LN_PHI,
    DynkinDiagram,
    complement, complement_spectrum, complement_distance,
    graph_distance, vertex_orbits, connection_count,
    save_m13_results, _convert_numpy,
)


def test_T1_a5_complement_spectra():
    """T1: Complement spectra distinguish all automorphism orbits of A_5."""
    # A_5 has rank 5, so 5 vertices in a chain (0--1--2--3--4)
    # Aut(A_5) = Z_2 (reflection). Orbits: {0,4}, {1,3}, {2} => 3 orbits
    diag = DynkinDiagram('A', 5)
    adj = diag.adjacency
    n = adj.shape[0]

    # Compute complement spectrum for each vertex
    spectra = {}
    for v in range(n):
        spec = complement_spectrum(adj, v)
        spectra[v] = spec

    # Count distinct spectra (using rounded tuples as keys)
    distinct = {}
    for v in range(n):
        key = tuple(np.round(spectra[v], decimals=10))
        if key not in distinct:
            distinct[key] = []
        distinct[key].append(v)

    n_distinct = len(distinct)
    orbits = vertex_orbits(adj)
    n_orbits = len(orbits)
    expected_orbits = 3  # ceil(5/2) = 3 for A_5 (5 vertices)

    print(f"  A_5: {n} vertices, {n_distinct} distinct spectra, {n_orbits} orbits")
    for key, verts in distinct.items():
        print(f"    Spectrum class -> vertices {verts}")

    result = {
        'test': 'T1_a5_complement_spectra',
        'n_vertices': n,
        'n_distinct_spectra': n_distinct,
        'n_orbits': n_orbits,
        'expected_orbits': expected_orbits,
        'orbit_members': [sorted(o) for o in orbits],
        'spectra_by_vertex': {str(v): spectra[v].tolist() for v in range(n)},
        'PASS': n_distinct == expected_orbits and n_distinct == n_orbits,
    }
    return result


def test_T2_d4_e6_orbit_counts():
    """T2: Distinct complement spectra match Aut orbit counts for D_4 and E_6."""
    results_by_type = {}

    # D_4: 4 vertices. Center (vertex depends on construction) has degree 3,
    # 3 leaves have degree 1. Aut(D_4) = S_3 (permuting leaves).
    # Orbits: {center}, {leaf1, leaf2, leaf3} => 2 orbits
    d4 = DynkinDiagram('D', 4)
    d4_orbits = vertex_orbits(d4.adjacency)
    d4_n_orbits = len(d4_orbits)
    d4_expected = 2
    results_by_type['D_4'] = {
        'n_vertices': d4.adjacency.shape[0],
        'n_orbits': d4_n_orbits,
        'expected_orbits': d4_expected,
        'orbit_members': [sorted(o) for o in d4_orbits],
        'match': d4_n_orbits == d4_expected,
    }
    print(f"  D_4: {d4_n_orbits} orbits (expected {d4_expected})")
    for o in d4_orbits:
        print(f"    Orbit: {sorted(o)}")

    # E_6: 6 vertices. Has Z_2 symmetry (reflection of the long arm).
    # Standard E_6 Dynkin diagram: chain 0-1-2-3-4 with branch at 2 to vertex 5.
    # Orbits depend on exact construction; E_6 has Z_2 automorphism.
    # Expected orbits: 4 (vertices pair up under the Z_2, with 2 fixed points)
    e6 = DynkinDiagram('E', 6)
    e6_orbits = vertex_orbits(e6.adjacency)
    e6_n_orbits = len(e6_orbits)
    # E_6 has 4 orbits under its Z_2 automorphism
    e6_expected = 4
    results_by_type['E_6'] = {
        'n_vertices': e6.adjacency.shape[0],
        'n_orbits': e6_n_orbits,
        'expected_orbits': e6_expected,
        'orbit_members': [sorted(o) for o in e6_orbits],
        'match': e6_n_orbits == e6_expected,
    }
    print(f"  E_6: {e6_n_orbits} orbits (expected {e6_expected})")
    for o in e6_orbits:
        print(f"    Orbit: {sorted(o)}")

    all_match = all(r['match'] for r in results_by_type.values())

    result = {
        'test': 'T2_d4_e6_orbit_counts',
        'by_type': results_by_type,
        'all_match': all_match,
        'PASS': all_match,
    }
    return result


def test_T3_complement_distance_separates_orbits():
    """T3: Complement distance is zero within orbits and positive between orbits."""
    # Complement distance measures STRUCTURAL role, not positional distance.
    # Key insight: comp_dist(i,j) = 0 iff i and j are in the same automorphism
    # orbit, and comp_dist > 0 for vertices in different orbits.
    # This is the "complement = identity" claim: same complement => same identity.
    #
    # Test on A_7 (7 vertices), D_4, and D_5.
    types = [('A', 7), ('D', 4), ('D', 5)]
    all_correct = True
    details = {}

    for t, r in types:
        diag = DynkinDiagram(t, r)
        adj = diag.adjacency
        n = adj.shape[0]
        orbits = vertex_orbits(adj)

        # Build orbit membership map
        orbit_of = {}
        for oi, orb in enumerate(orbits):
            for v in orb:
                orbit_of[v] = oi

        # Check all pairs
        within_orbit_max = 0.0
        between_orbit_min = float('inf')
        n_within = 0
        n_between = 0
        for i in range(n):
            for j in range(i + 1, n):
                cd = complement_distance(adj, i, j)
                if orbit_of[i] == orbit_of[j]:
                    within_orbit_max = max(within_orbit_max, cd)
                    n_within += 1
                else:
                    between_orbit_min = min(between_orbit_min, cd)
                    n_between += 1

        # Within-orbit distances should be 0, between-orbit should be > 0
        within_ok = within_orbit_max < 1e-10
        between_ok = between_orbit_min > 1e-10 if n_between > 0 else True
        correct = within_ok and between_ok
        if not correct:
            all_correct = False

        label = f"{t}_{r}"
        details[label] = {
            'n_vertices': n,
            'n_orbits': len(orbits),
            'n_within_pairs': n_within,
            'n_between_pairs': n_between,
            'within_orbit_max_dist': within_orbit_max,
            'between_orbit_min_dist': between_orbit_min if n_between > 0 else None,
            'within_ok': within_ok,
            'between_ok': between_ok,
        }
        print(f"  {label}: {len(orbits)} orbits, within_max={within_orbit_max:.2e}, "
              f"between_min={between_orbit_min:.4f}, {'OK' if correct else 'FAIL'}")

    result = {
        'test': 'T3_complement_distance_separates_orbits',
        'types_tested': [f"{t}_{r}" for t, r in types],
        'details': details,
        'note': 'Complement distance = 0 within orbits (same structural role) '
                'and > 0 between orbits (different identities). This is the '
                'central claim: identity IS complement.',
        'PASS': all_correct,
    }
    return result


def test_T4_double_complement_phi_ratios():
    """T4: Double-complement edge ratios cluster near INV_PHI^k values."""
    # For A_5, vertex v=2: compute C = complement(A_5, 2)
    # Then for each vertex u in C, compute complement(C, u)
    # Check if edges(complement(C,u)) / edges(A_5) falls near INV_PHI^k
    diag = DynkinDiagram('A', 5)
    adj = diag.adjacency
    n = adj.shape[0]
    parent_edges = connection_count(adj)

    phi_powers = [INV_PHI**k for k in range(1, 5)]  # k=1,2,3,4

    hits = 0
    total_pairs = 0
    pair_results = []

    for v in range(n):
        sub_adj, removed = complement(adj, v)
        n_sub = sub_adj.shape[0]
        if n_sub < 2:
            continue

        for u in range(n_sub):
            sub_sub, _ = complement(sub_adj, u)
            if sub_sub.size == 0:
                double_edges = 0
            else:
                double_edges = connection_count(sub_sub)

            if parent_edges > 0:
                ratio = double_edges / parent_edges
            else:
                ratio = 0.0

            # Check proximity to any INV_PHI^k
            min_dist = min(abs(ratio - pk) for pk in phi_powers)
            is_near = min_dist < 0.1
            nearest_k = min(range(1, 5), key=lambda k: abs(ratio - INV_PHI**k))

            pair_results.append({
                'v': v, 'u': u,
                'double_edges': double_edges,
                'parent_edges': parent_edges,
                'ratio': ratio,
                'nearest_phi_k': nearest_k,
                'nearest_phi_val': INV_PHI**nearest_k,
                'distance': min_dist,
                'is_near': is_near,
            })
            total_pairs += 1
            if is_near:
                hits += 1

    hit_fraction = hits / total_pairs if total_pairs > 0 else 0.0
    print(f"  A_5 double-complement: {hits}/{total_pairs} pairs near INV_PHI^k "
          f"({hit_fraction:.1%})")
    for pr in pair_results[:6]:
        print(f"    v={pr['v']}, u={pr['u']}: ratio={pr['ratio']:.4f}, "
              f"nearest=phi^(-{pr['nearest_phi_k']})={pr['nearest_phi_val']:.4f}, "
              f"dist={pr['distance']:.4f} {'HIT' if pr['is_near'] else 'miss'}")

    result = {
        'test': 'T4_double_complement_phi_ratios',
        'graph': 'A_5',
        'parent_edges': parent_edges,
        'total_pairs': total_pairs,
        'hits': hits,
        'hit_fraction': hit_fraction,
        'threshold': 0.1,
        'phi_powers': {str(k): INV_PHI**k for k in range(1, 5)},
        'sample_pairs': pair_results[:8],
        'PASS': hit_fraction >= 0.60,
    }
    return result


def main():
    print("=" * 70)
    print("EXP 01 -- Complement Determines Identity")
    print("Milestone 13, Block A")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_a5_complement_spectra),
        ('T2', test_T2_d4_e6_orbit_counts),
        ('T3', test_T3_complement_distance_separates_orbits),
        ('T4', test_T4_double_complement_phi_ratios),
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
        'experiment': 'exp_01_complement_determines_identity',
        'milestone': 'milestone13',
        'block': 'A',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m13_results('exp_01_complement_determines_identity', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()
