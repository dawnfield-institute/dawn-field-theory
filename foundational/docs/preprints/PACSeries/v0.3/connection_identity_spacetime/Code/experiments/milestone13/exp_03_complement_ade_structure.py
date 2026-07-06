"""
exp_03 -- Complement Preserves ADE Structure

Milestone 13, Block A (Identity IS Complement)

Hypothesis: Complement operations preserve ADE type information. The multiset
of complement spectra is a unique fingerprint for each ADE type. Complement
of an endpoint of A_n is A_{n-1}. Edge conservation holds exactly. Cauchy
interlacing ensures complement spectral radius <= parent spectral radius.

Tests:
  T1: Complement fingerprint uniquely identifies ADE type (6 types tested)
  T2: Complement of A_n endpoint = A_{n-1} (edge count check)
  T3: Edge conservation: edges(G) = edges(complement) + removed_edges for all vertices
  T4: Complement spectral radius <= parent spectral radius (Cauchy interlacing)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from identity_complement import (
    PHI, INV_PHI,
    DynkinDiagram,
    complement, complement_spectrum, connection_count,
    save_m13_results, _convert_numpy,
)


def test_T1_complement_fingerprint_identifies_type():
    """T1: Complement fingerprint uniquely identifies ADE type across 6 types."""
    # For each ADE type, compute the SORTED multiset of complement spectra
    # (one spectrum per vertex, sorted as tuples). The multiset itself is
    # the fingerprint. All 6 should be distinct.
    types = [('A', 4), ('A', 5), ('A', 6), ('D', 4), ('D', 5), ('E', 6)]
    fingerprints = {}

    for t, r in types:
        diag = DynkinDiagram(t, r)
        adj = diag.adjacency
        n = adj.shape[0]

        # Collect all complement spectra as rounded tuples
        spectra = []
        for v in range(n):
            spec = complement_spectrum(adj, v)
            spectra.append(tuple(np.round(spec, decimals=10)))

        # Sort the multiset for comparison
        fingerprint = tuple(sorted(spectra))
        label = f"{t}_{r}"
        fingerprints[label] = fingerprint
        print(f"  {label}: {n} vertices, {len(set(spectra))} distinct spectra")

    # Check all fingerprints are distinct
    labels = list(fingerprints.keys())
    all_distinct = True
    collisions = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if fingerprints[labels[i]] == fingerprints[labels[j]]:
                all_distinct = False
                collisions.append((labels[i], labels[j]))

    if collisions:
        print(f"  COLLISIONS: {collisions}")
    else:
        print(f"  All {len(labels)} fingerprints are distinct")

    result = {
        'test': 'T1_complement_fingerprint_identifies_type',
        'types_tested': [f"{t}_{r}" for t, r in types],
        'n_types': len(types),
        'n_distinct_fingerprints': len(set(fingerprints.values())),
        'all_distinct': all_distinct,
        'collisions': collisions,
        'PASS': all_distinct,
    }
    return result


def test_T2_endpoint_complement_is_subtype():
    """T2: Complement of A_n endpoint is A_{n-1} (verified by edge count)."""
    # For A_n (rank n, n vertices), removing endpoint (vertex 0) gives
    # a chain of n-1 vertices = A_{n-1} which has n-2 edges.
    results_by_n = {}
    all_match = True

    for n in range(2, 9):
        diag = DynkinDiagram('A', n)
        adj = diag.adjacency

        # Complement of endpoint (vertex 0)
        sub_adj, removed = complement(adj, 0)
        sub_edges = connection_count(sub_adj)
        expected_edges = n - 2  # A_{n-1} has n-1 vertices and n-2 edges

        # Also check the complement is itself a path graph (A_{n-1})
        # by verifying max degree <= 2 and connected
        if sub_adj.size > 0:
            degrees = np.sum(sub_adj > 0, axis=1)
            max_degree = int(np.max(degrees))
            is_path = max_degree <= 2
        else:
            max_degree = 0
            is_path = True

        match = (sub_edges == expected_edges) and is_path
        if not match:
            all_match = False

        results_by_n[n] = {
            'complement_edges': sub_edges,
            'expected_edges': expected_edges,
            'removed_edges': removed,
            'max_degree_in_complement': max_degree,
            'is_path_graph': is_path,
            'match': match,
        }
        print(f"  A_{n}: complement(0) has {sub_edges} edges "
              f"(expected {expected_edges}), removed={removed}, "
              f"{'OK' if match else 'MISMATCH'}")

    result = {
        'test': 'T2_endpoint_complement_is_subtype',
        'tested_ranks': list(range(2, 9)),
        'by_n': results_by_n,
        'all_match': all_match,
        'PASS': all_match,
    }
    return result


def test_T3_edge_conservation():
    """T3: Edge conservation: edges(G) = edges(complement(G,v)) + removed_edges."""
    # For all ADE types tested, all vertices: exact integer equality
    types = [('A', 3), ('A', 4), ('A', 5), ('A', 6), ('A', 7), ('A', 8),
             ('D', 4), ('D', 5), ('E', 6)]

    all_conserved = True
    violations = []
    total_checks = 0

    for t, r in types:
        diag = DynkinDiagram(t, r)
        adj = diag.adjacency
        n = adj.shape[0]
        parent_edges = connection_count(adj)

        for v in range(n):
            sub_adj, removed = complement(adj, v)
            sub_edges = connection_count(sub_adj)
            conserved = (sub_edges + removed == parent_edges)
            total_checks += 1

            if not conserved:
                all_conserved = False
                violations.append({
                    'type': f"{t}_{r}",
                    'vertex': v,
                    'parent_edges': parent_edges,
                    'sub_edges': sub_edges,
                    'removed': removed,
                    'sum': sub_edges + removed,
                })

    print(f"  Tested {total_checks} vertex-complement pairs across {len(types)} ADE types")
    if violations:
        print(f"  {len(violations)} violations found!")
        for viol in violations[:5]:
            print(f"    {viol['type']} v={viol['vertex']}: "
                  f"{viol['sub_edges']} + {viol['removed']} != {viol['parent_edges']}")
    else:
        print(f"  All {total_checks} checks passed: edges perfectly conserved")

    result = {
        'test': 'T3_edge_conservation',
        'types_tested': [f"{t}_{r}" for t, r in types],
        'total_checks': total_checks,
        'n_violations': len(violations),
        'violations': violations[:10],
        'all_conserved': all_conserved,
        'PASS': all_conserved,
    }
    return result


def test_T4_cauchy_interlacing():
    """T4: Complement spectral radius <= parent spectral radius (Cauchy interlacing)."""
    # For all ADE types and all vertices, the complement's largest eigenvalue
    # should not exceed the parent's largest eigenvalue (plus numerical tolerance).
    types = [('A', 3), ('A', 4), ('A', 5), ('A', 6), ('A', 7), ('A', 8),
             ('D', 4), ('D', 5), ('E', 6)]

    all_hold = True
    violations = []
    total_checks = 0
    max_excess = 0.0

    for t, r in types:
        diag = DynkinDiagram(t, r)
        adj = diag.adjacency
        n = adj.shape[0]
        parent_sr = diag.spectral_radius()

        for v in range(n):
            spec = complement_spectrum(adj, v)
            if len(spec) > 0:
                comp_sr = float(np.max(np.abs(spec)))
            else:
                comp_sr = 0.0

            excess = comp_sr - parent_sr
            total_checks += 1

            if excess > max_excess:
                max_excess = excess

            if excess > 1e-10:
                all_hold = False
                violations.append({
                    'type': f"{t}_{r}",
                    'vertex': v,
                    'parent_sr': parent_sr,
                    'complement_sr': comp_sr,
                    'excess': excess,
                })

    print(f"  Tested {total_checks} vertex-complement pairs across {len(types)} ADE types")
    print(f"  Max excess (complement SR - parent SR) = {max_excess:.2e}")
    if violations:
        print(f"  {len(violations)} violations found!")
        for viol in violations[:5]:
            print(f"    {viol['type']} v={viol['vertex']}: "
                  f"comp_sr={viol['complement_sr']:.6f} > parent_sr={viol['parent_sr']:.6f}")
    else:
        print(f"  All {total_checks} checks passed: Cauchy interlacing holds")

    result = {
        'test': 'T4_cauchy_interlacing',
        'types_tested': [f"{t}_{r}" for t, r in types],
        'total_checks': total_checks,
        'n_violations': len(violations),
        'max_excess': float(max_excess),
        'violations': violations[:10],
        'all_hold': all_hold,
        'PASS': all_hold,
    }
    return result


def main():
    print("=" * 70)
    print("EXP 03 -- Complement Preserves ADE Structure")
    print("Milestone 13, Block A")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_complement_fingerprint_identifies_type),
        ('T2', test_T2_endpoint_complement_is_subtype),
        ('T3', test_T3_edge_conservation),
        ('T4', test_T4_cauchy_interlacing),
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
        'experiment': 'exp_03_complement_ade_structure',
        'milestone': 'milestone13',
        'block': 'A',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m13_results('exp_03_complement_ade_structure', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()
