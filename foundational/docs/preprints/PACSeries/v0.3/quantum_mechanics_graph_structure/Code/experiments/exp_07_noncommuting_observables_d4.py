"""
exp_07 -- Non-Commuting Observables on D_4

Milestone 14, Block D (Uncertainty)

Hypothesis: D_4 is the ONLY ADE type with non-abelian automorphism group (S_3),
and therefore the ONLY type with genuinely non-commuting observables. This is
the graph-theoretic origin of quantum uncertainty. Abelian groups (Z_2 for A_n,
trivial for E_7, E_8) give commuting operators — the classical limit.

Tests:
  T1: Aut(D_4) = S_3 verified (6 elements, 3 conjugacy classes)
  T2: Non-commuting permutation operators [P_1, P_2] != 0
  T3: Abelian groups (Z_2) give commuting operators (classical limit)
  T4: D_4 is ONLY non-abelian ADE type <= rank 8
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_complement import (
    PHI, INV_PHI, LN_PHI,
    DynkinDiagram, all_ade_diagrams,
    graph_automorphisms, conjugacy_classes, noncommutativity_measure,
    save_m14_results, _convert_numpy,
)


def test_T1_d4_aut_is_s3():
    """T1: Aut(D_4) = S_3 verified (6 elements, 3 conjugacy classes)."""
    diag = DynkinDiagram('D', 4)
    adj = diag.adjacency

    auts = graph_automorphisms(adj)
    n_auts = len(auts)
    classes = conjugacy_classes(auts)
    n_classes = len(classes)
    class_sizes = sorted([len(c) for c in classes])

    # S_3 has: order 6, 3 conjugacy classes with sizes {1, 2, 3}
    is_order_6 = n_auts == 6
    has_3_classes = n_classes == 3
    correct_sizes = class_sizes == [1, 2, 3]

    # Verify group closure: product of any two elements is in the group
    auts_set = set()
    for P in auts:
        auts_set.add(tuple(P.flatten()))

    closure_check = True
    for i, P1 in enumerate(auts):
        for j, P2 in enumerate(auts):
            product = P1 @ P2
            key = tuple(product.flatten())
            if key not in auts_set:
                closure_check = False
                break

    # Verify identity element exists
    n = adj.shape[0]
    has_identity = any(np.allclose(P, np.eye(n)) for P in auts)

    passed = is_order_6 and has_3_classes and correct_sizes and closure_check and has_identity

    print(f"  |Aut(D_4)| = {n_auts} (expected 6)")
    print(f"  Conjugacy classes: {n_classes} with sizes {class_sizes}")
    print(f"  Group closure: {closure_check}, has identity: {has_identity}")

    result = {
        'test': 'T1_d4_aut_is_s3',
        'n_automorphisms': n_auts,
        'n_conjugacy_classes': n_classes,
        'class_sizes': class_sizes,
        'is_order_6': is_order_6,
        'has_3_classes': has_3_classes,
        'correct_sizes': correct_sizes,
        'closure_check': closure_check,
        'has_identity': has_identity,
        'PASS': passed,
    }
    return result


def test_T2_noncommuting_operators():
    """T2: Non-commuting permutation operators [P_1, P_2] != 0 on D_4."""
    diag = DynkinDiagram('D', 4)
    adj = diag.adjacency
    n = adj.shape[0]

    auts = graph_automorphisms(adj)

    # Find two non-commuting elements
    found_noncommuting = False
    max_commutator_norm = 0.0
    example_pair = None

    for i in range(len(auts)):
        for j in range(i + 1, len(auts)):
            comm = auts[i] @ auts[j] - auts[j] @ auts[i]
            norm = np.linalg.norm(comm, 'fro')
            if norm > max_commutator_norm:
                max_commutator_norm = norm
                example_pair = (i, j)
            if norm > 1e-10:
                found_noncommuting = True

    # Count non-commuting pairs
    n_pairs = 0
    n_noncommuting = 0
    for i in range(len(auts)):
        for j in range(i + 1, len(auts)):
            n_pairs += 1
            comm = auts[i] @ auts[j] - auts[j] @ auts[i]
            if np.linalg.norm(comm, 'fro') > 1e-10:
                n_noncommuting += 1

    # Non-commutativity measure
    nc_measure = noncommutativity_measure(auts)

    passed = found_noncommuting and nc_measure > 1e-10

    print(f"  Found non-commuting pair: {found_noncommuting}")
    print(f"  Max commutator norm: {max_commutator_norm:.4f}")
    print(f"  Non-commuting pairs: {n_noncommuting}/{n_pairs}")
    print(f"  NC measure: {nc_measure:.4f}")

    result = {
        'test': 'T2_noncommuting_operators',
        'found_noncommuting': found_noncommuting,
        'max_commutator_norm': float(max_commutator_norm),
        'n_pairs': n_pairs,
        'n_noncommuting': n_noncommuting,
        'nc_measure': float(nc_measure),
        'PASS': passed,
    }
    return result


def test_T3_abelian_commuting():
    """T3: Abelian groups (Z_2) give commuting operators (classical limit)."""
    # All A_n have Aut = Z_2 (except A_1 which is trivial)
    # Z_2 is abelian: all elements commute
    test_cases = [('A', 3), ('A', 5), ('A', 7), ('E', 6)]  # All Z_2 or trivial
    all_pass = True
    results_by_type = {}

    for family, rank in test_cases:
        label = f"{family}_{rank}"
        diag = DynkinDiagram(family, rank)
        adj = diag.adjacency

        auts = graph_automorphisms(adj)
        nc_measure = noncommutativity_measure(auts)

        # Check all pairs commute
        all_commute = True
        for i in range(len(auts)):
            for j in range(i + 1, len(auts)):
                comm = auts[i] @ auts[j] - auts[j] @ auts[i]
                if np.linalg.norm(comm, 'fro') > 1e-10:
                    all_commute = False
                    break

        passed = all_commute and nc_measure < 1e-10
        all_pass = all_pass and passed

        print(f"  {label}: |Aut|={len(auts)}, all_commute={all_commute}, NC={nc_measure:.6f}")

        results_by_type[label] = {
            'n_automorphisms': len(auts),
            'all_commute': all_commute,
            'nc_measure': float(nc_measure),
            'PASS': passed,
        }

    result = {
        'test': 'T3_abelian_commuting',
        'results_by_type': results_by_type,
        'PASS': all_pass,
    }
    return result


def test_T4_d4_only_nonabelian():
    """T4: D_4 is ONLY non-abelian ADE type <= rank 8."""
    diagrams = all_ade_diagrams(max_rank=8)
    nonabelian_types = []
    results_by_type = {}

    for diag in diagrams:
        label = diag.name
        adj = diag.adjacency

        auts = graph_automorphisms(adj)
        nc_measure = noncommutativity_measure(auts)
        is_nonabelian = nc_measure > 1e-10

        if is_nonabelian:
            nonabelian_types.append(label)

        results_by_type[label] = {
            'n_automorphisms': len(auts),
            'nc_measure': float(nc_measure),
            'is_nonabelian': is_nonabelian,
        }

        print(f"  {label}: |Aut|={len(auts)}, NC={nc_measure:.4f}, "
              f"nonabelian={is_nonabelian}")

    only_d4 = nonabelian_types == ['D_4']
    passed = only_d4

    print(f"\n  Non-abelian types: {nonabelian_types}")
    print(f"  Only D_4: {only_d4}")

    result = {
        'test': 'T4_d4_only_nonabelian',
        'nonabelian_types': nonabelian_types,
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
    print("Experiment 07: Non-Commuting Observables on D_4")
    print("Milestone 14, Block D")
    print("=" * 70)

    results = {}
    scorecard = []

    tests = [
        ("T1", test_T1_d4_aut_is_s3),
        ("T2", test_T2_noncommuting_operators),
        ("T3", test_T3_abelian_commuting),
        ("T4", test_T4_d4_only_nonabelian),
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
        'experiment': 'exp_07_noncommuting_observables_d4',
        'milestone': 14,
        'block': 'D',
        'results': results,
        'scorecard': {f"T{i+1}": s for i, s in enumerate(scorecard)},
        'score': f"{n_pass}/{n_total}",
        'n_pass': n_pass,
        'n_total': n_total,
    }

    save_m14_results('exp_07_noncommuting_observables_d4', _convert_numpy(save_data))
    return n_pass, n_total


if __name__ == "__main__":
    main()
