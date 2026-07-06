"""
exp_02 -- Permutation Representation Decomposition

Milestone 14, Block A (Orbit Hilbert Space)

Hypothesis: The permutation representation of Aut(G) on C^n decomposes into
irreducible representations that correspond to quantum modes. For Z_2 (A_n):
symmetric + antisymmetric. For S_3 (D_4): trivial + sign + standard 2D irrep.
The number of times the trivial irrep appears equals the number of orbits
(Burnside's lemma). D_4 is the ONLY ADE type with higher-dimensional irreps.

Tests:
  T1: Z_2 decomposition on A_5 (symmetric + antisymmetric modes)
  T2: S_3 decomposition on D_4 (trivial + standard 2D irrep)
  T3: Trivial irrep multiplicity = number of orbits (Burnside) for all ADE <= rank 8
  T4: D_4 is ONLY ADE type with higher-dim irreps
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_complement import (
    PHI, INV_PHI, LN_PHI,
    DynkinDiagram, all_ade_diagrams,
    graph_automorphisms, conjugacy_classes,
    orbit_hilbert_basis, permutation_rep_decompose,
    vertex_orbits,
    save_m14_results, _convert_numpy,
)


def test_T1_z2_decomposition_a5():
    """T1: Z_2 decomposition on A_5 gives symmetric + antisymmetric modes."""
    diag = DynkinDiagram('A', 5)
    adj = diag.adjacency
    n = adj.shape[0]  # 5

    decomp = permutation_rep_decompose(adj)

    # A_5 has Aut = Z_2 (reflection symmetry)
    is_z2 = decomp['type'] == 'Z_2'
    has_two_irreps = len(decomp.get('irreps', [])) == 2

    # Check irrep names and dimensions
    irrep_dict = {ir['name']: ir for ir in decomp.get('irreps', [])}
    has_trivial = 'trivial' in irrep_dict
    has_sign = 'sign' in irrep_dict

    # Multiplicities should sum to n=5
    total_mult = sum(ir['dim'] * ir['multiplicity'] for ir in decomp.get('irreps', []))
    dim_check = total_mult == n

    # For A_5 with Z_2: symmetric modes = ceil(5/2) = 3, antisymmetric = floor(5/2) = 2
    m_trivial = irrep_dict.get('trivial', {}).get('multiplicity', 0)
    m_sign = irrep_dict.get('sign', {}).get('multiplicity', 0)

    # 3 orbits means trivial multiplicity = 3
    expected_m_trivial = 3  # number of orbits of A_5
    expected_m_sign = 2     # n - n_orbits = 5 - 3 = 2
    mults_correct = (m_trivial == expected_m_trivial and m_sign == expected_m_sign)

    # Eigenspace check should confirm
    eigenspace_matches = decomp.get('eigenspace_check', {}).get('matches_character', False)

    passed = is_z2 and has_two_irreps and dim_check and mults_correct and eigenspace_matches

    print(f"  A_5: type={decomp['type']}, irreps={decomp.get('irreps', [])}")
    print(f"  dim_check={dim_check}, mults_correct={mults_correct}, eigenspace={eigenspace_matches}")

    result = {
        'test': 'T1_z2_decomposition_a5',
        'n': n,
        'group_type': decomp['type'],
        'group_order': decomp['order'],
        'irreps': decomp.get('irreps', []),
        'is_z2': is_z2,
        'dim_check': dim_check,
        'm_trivial': m_trivial,
        'm_sign': m_sign,
        'expected_m_trivial': expected_m_trivial,
        'expected_m_sign': expected_m_sign,
        'mults_correct': mults_correct,
        'eigenspace_matches': eigenspace_matches,
        'PASS': passed,
    }
    return result


def test_T2_s3_decomposition_d4():
    """T2: S_3 decomposition on D_4 (trivial + sign + standard 2D irrep)."""
    diag = DynkinDiagram('D', 4)
    adj = diag.adjacency
    n = adj.shape[0]  # 4

    decomp = permutation_rep_decompose(adj)

    # D_4 has Aut = S_3 (triality symmetry)
    is_s3 = decomp['type'] == 'S_3'
    is_order_6 = decomp['order'] == 6
    has_3_classes = decomp['n_conjugacy_classes'] == 3

    # S_3 has three irreps: trivial(1D), sign(1D), standard(2D)
    irrep_dict = {ir['name']: ir for ir in decomp.get('irreps', [])}

    # Check we have all three types
    has_all_irreps = all(name in irrep_dict for name in ['trivial', 'sign', 'standard'])

    # Dimensions should sum to n=4
    total_dim = sum(ir['dim'] * ir['multiplicity'] for ir in decomp.get('irreps', []))
    dim_check = total_dim == n

    # D_4 has 2 orbits: {hub} and {3 leaves}
    # Trivial multiplicity should be 2 (Burnside)
    m_trivial = irrep_dict.get('trivial', {}).get('multiplicity', 0)
    m_sign = irrep_dict.get('sign', {}).get('multiplicity', 0)
    m_standard = irrep_dict.get('standard', {}).get('multiplicity', 0)

    # Expected: trivial x 2 (2 orbits), sign x 0, standard x 1
    # 1*2 + 1*0 + 2*1 = 4 = n  checks out
    expected_decomp = (m_trivial == 2 and m_standard == 1)

    # Standard irrep has dim 2 — this is the key quantum mode
    has_2d_irrep = irrep_dict.get('standard', {}).get('dim', 0) == 2

    passed = is_s3 and dim_check and expected_decomp and has_2d_irrep

    print(f"  D_4: type={decomp['type']}, order={decomp['order']}, "
          f"classes={decomp['n_conjugacy_classes']}")
    print(f"  irreps: {decomp.get('irreps', [])}")
    print(f"  dim_check={dim_check}, expected_decomp={expected_decomp}, has_2d={has_2d_irrep}")

    result = {
        'test': 'T2_s3_decomposition_d4',
        'n': n,
        'group_type': decomp['type'],
        'group_order': decomp['order'],
        'n_conjugacy_classes': decomp['n_conjugacy_classes'],
        'irreps': decomp.get('irreps', []),
        'is_s3': is_s3,
        'dim_check': dim_check,
        'm_trivial': m_trivial,
        'm_sign': m_sign,
        'm_standard': m_standard,
        'expected_decomp': expected_decomp,
        'has_2d_irrep': has_2d_irrep,
        'PASS': passed,
    }
    return result


def test_T3_burnside_trivial_multiplicity():
    """T3: Trivial irrep multiplicity = number of orbits (Burnside) for all ADE <= rank 8."""
    diagrams = all_ade_diagrams(max_rank=8)
    all_pass = True
    results_by_type = {}

    for diag in diagrams:
        label = diag.name
        adj = diag.adjacency

        # Number of orbits
        orbits = vertex_orbits(adj)
        n_orbits = len(orbits)

        # Decompose
        decomp = permutation_rep_decompose(adj)

        # Find trivial irrep multiplicity
        m_trivial = 0
        for ir in decomp.get('irreps', []):
            if ir['name'] == 'trivial':
                m_trivial = ir['multiplicity']
                break

        # Burnside: m_trivial should equal n_orbits
        burnside_match = (m_trivial == n_orbits)
        all_pass = all_pass and burnside_match

        print(f"  {label}: n_orbits={n_orbits}, m_trivial={m_trivial}, "
              f"match={burnside_match}, group={decomp['type']}")

        results_by_type[label] = {
            'n_orbits': n_orbits,
            'm_trivial': m_trivial,
            'burnside_match': burnside_match,
            'group_type': decomp['type'],
            'group_order': decomp['order'],
            'PASS': burnside_match,
        }

    result = {
        'test': 'T3_burnside_trivial_multiplicity',
        'n_diagrams': len(diagrams),
        'results_by_type': results_by_type,
        'PASS': all_pass,
    }
    return result


def test_T4_d4_only_higher_dim_irreps():
    """T4: D_4 is ONLY ADE type with higher-dim irreps (dim > 1)."""
    diagrams = all_ade_diagrams(max_rank=8)
    types_with_higher_dim = []
    results_by_type = {}

    for diag in diagrams:
        label = diag.name
        adj = diag.adjacency

        decomp = permutation_rep_decompose(adj)
        max_dim = max((ir['dim'] for ir in decomp.get('irreps', [])
                       if ir['multiplicity'] > 0), default=1)
        has_higher = max_dim > 1

        if has_higher:
            types_with_higher_dim.append(label)

        print(f"  {label}: group={decomp['type']}, max_irrep_dim={max_dim}, "
              f"higher_dim={has_higher}")

        results_by_type[label] = {
            'group_type': decomp['type'],
            'group_order': decomp['order'],
            'max_irrep_dim': max_dim,
            'has_higher_dim_irreps': has_higher,
            'irreps': decomp.get('irreps', []),
        }

    # D_4 should be the ONLY one
    only_d4 = (types_with_higher_dim == ['D_4'])
    passed = only_d4

    print(f"\n  Types with higher-dim irreps: {types_with_higher_dim}")
    print(f"  Only D_4: {only_d4}")

    result = {
        'test': 'T4_d4_only_higher_dim_irreps',
        'types_with_higher_dim': types_with_higher_dim,
        'only_d4': only_d4,
        'results_by_type': results_by_type,
        'PASS': passed,
    }
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Experiment 02: Permutation Representation Decomposition")
    print("Milestone 14, Block A")
    print("=" * 70)

    results = {}
    scorecard = []

    tests = [
        ("T1", test_T1_z2_decomposition_a5),
        ("T2", test_T2_s3_decomposition_d4),
        ("T3", test_T3_burnside_trivial_multiplicity),
        ("T4", test_T4_d4_only_higher_dim_irreps),
    ]

    for name, fn in tests:
        print(f"\n--- {name}: {fn.__doc__.strip()} ---")
        r = fn()
        results[name] = r
        scorecard.append(r['PASS'])
        status = "PASS" if r['PASS'] else "FAIL"
        print(f"  => {status}")

    n_pass = sum(scorecard)
    n_total = len(scorecard)
    print(f"\n{'=' * 70}")
    print(f"Score: {n_pass}/{n_total}")
    print(f"{'=' * 70}")

    save_data = {
        'experiment': 'exp_02_permutation_rep_decomposition',
        'milestone': 14,
        'block': 'A',
        'results': results,
        'scorecard': {f"T{i+1}": s for i, s in enumerate(scorecard)},
        'score': f"{n_pass}/{n_total}",
        'n_pass': n_pass,
        'n_total': n_total,
    }

    save_m14_results('exp_02_permutation_rep_decomposition', _convert_numpy(save_data))
    return n_pass, n_total


if __name__ == "__main__":
    main()
