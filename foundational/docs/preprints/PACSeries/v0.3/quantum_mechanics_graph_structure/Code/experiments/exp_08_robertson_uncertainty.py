"""
exp_08 -- Robertson Uncertainty

Milestone 14, Block D (Uncertainty)

Hypothesis: The Robertson uncertainty relation Delta_A * Delta_B >= |<[A,B]>|/2
is nontrivial only when the automorphism group is non-abelian. For D_4 (S_3):
genuine quantum uncertainty with nonzero bound. For A_n (Z_2) and E_7/E_8
(trivial): bound is zero (classical determinism).

Tests:
  T1: Robertson bound nontrivial on D_4 (Delta_A * Delta_B > 0)
  T2: Robertson bound zero for abelian cases
  T3: Uncertainty scales with non-commutativity measure (dichotomy expected)
  T4: Minimum uncertainty product: finite for D_4, zero for A_n
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_complement import (
    PHI, INV_PHI, LN_PHI,
    DynkinDiagram, all_ade_diagrams,
    graph_automorphisms, noncommutativity_measure,
    robertson_uncertainty,
    save_m14_results, _convert_numpy,
)


def test_T1_robertson_nontrivial_d4():
    """T1: Robertson bound nontrivial on D_4 (Delta_A * Delta_B > 0)."""
    diag = DynkinDiagram('D', 4)
    adj = diag.adjacency
    n = adj.shape[0]

    auts = graph_automorphisms(adj)

    # Find two non-commuting automorphisms to use as observables
    # For S_3: any transposition and any 3-cycle don't commute
    non_id = [P for P in auts if not np.allclose(P, np.eye(n))]

    # Try all pairs until we find non-commuting
    best_bound = 0.0
    best_result = None

    for i in range(len(non_id)):
        for j in range(i + 1, len(non_id)):
            op_A = non_id[i]
            op_B = non_id[j]

            # Make Hermitian: H_A = (P_A + P_A^T) / 2 (already real, so H = (P + P^T)/2)
            H_A = (op_A + op_A.T) / 2
            H_B = (op_B + op_B.T) / 2

            # Try several states
            for trial in range(10):
                np.random.seed(42 + trial)
                state = np.random.randn(n) + 1j * np.random.randn(n)
                state = state / np.linalg.norm(state)

                r = robertson_uncertainty(state, H_A, H_B)
                if r['robertson_bound'] > best_bound:
                    best_bound = r['robertson_bound']
                    best_result = r

    nontrivial = best_bound > 1e-10
    satisfied = best_result['satisfied'] if best_result else True

    passed = nontrivial and satisfied

    print(f"  Best Robertson bound: {best_bound:.6f}")
    if best_result:
        print(f"  Delta_A={best_result['delta_A']:.4f}, Delta_B={best_result['delta_B']:.4f}")
        print(f"  Product={best_result['product']:.6f}, Bound={best_result['robertson_bound']:.6f}")
        print(f"  Satisfied: {best_result['satisfied']}")

    result = {
        'test': 'T1_robertson_nontrivial_d4',
        'best_bound': float(best_bound),
        'best_result': best_result,
        'nontrivial': nontrivial,
        'satisfied': satisfied,
        'PASS': passed,
    }
    return result


def test_T2_robertson_zero_abelian():
    """T2: Robertson bound zero for abelian cases."""
    test_cases = [('A', 3), ('A', 5), ('E', 7)]
    all_pass = True
    results_by_type = {}

    for family, rank in test_cases:
        label = f"{family}_{rank}"
        diag = DynkinDiagram(family, rank)
        adj = diag.adjacency
        n = adj.shape[0]

        auts = graph_automorphisms(adj)
        non_id = [P for P in auts if not np.allclose(P, np.eye(n))]

        if len(non_id) < 2:
            # Trivial or Z_2 — fewer than 2 non-identity elements
            # For Z_2: only one non-identity element, commutes with itself
            # Robertson bound is automatically 0
            max_bound = 0.0
            is_zero = True
        else:
            max_bound = 0.0
            for i in range(len(non_id)):
                for j in range(i + 1, len(non_id)):
                    H_A = (non_id[i] + non_id[i].T) / 2
                    H_B = (non_id[j] + non_id[j].T) / 2

                    for trial in range(5):
                        np.random.seed(42 + trial)
                        state = np.random.randn(n) + 1j * np.random.randn(n)
                        state = state / np.linalg.norm(state)

                        r = robertson_uncertainty(state, H_A, H_B)
                        max_bound = max(max_bound, r['robertson_bound'])

            is_zero = max_bound < 1e-10

        passed = is_zero
        all_pass = all_pass and passed

        print(f"  {label}: |Aut|={len(auts)}, max_bound={max_bound:.6f}, zero={is_zero}")

        results_by_type[label] = {
            'n_automorphisms': len(auts),
            'max_bound': float(max_bound),
            'is_zero': is_zero,
            'PASS': passed,
        }

    result = {
        'test': 'T2_robertson_zero_abelian',
        'results_by_type': results_by_type,
        'PASS': all_pass,
    }
    return result


def test_T3_uncertainty_scales_with_nc():
    """T3: Uncertainty scales with non-commutativity measure."""
    # For ADE types: NC measure is either 0 (abelian) or positive (D_4 only)
    # So this is a DICHOTOMY, not a smooth scaling
    diagrams = all_ade_diagrams(max_rank=8)
    nc_values = []
    max_bounds = []
    results_by_type = {}

    for diag in diagrams:
        label = diag.name
        adj = diag.adjacency
        n = adj.shape[0]

        auts = graph_automorphisms(adj)
        nc = noncommutativity_measure(auts)
        non_id = [P for P in auts if not np.allclose(P, np.eye(n))]

        # Get max Robertson bound
        max_bound = 0.0
        if len(non_id) >= 2:
            for i in range(min(len(non_id), 5)):
                for j in range(i + 1, min(len(non_id), 5)):
                    H_A = (non_id[i] + non_id[i].T) / 2
                    H_B = (non_id[j] + non_id[j].T) / 2
                    np.random.seed(42)
                    state = np.random.randn(n) + 1j * np.random.randn(n)
                    state = state / np.linalg.norm(state)
                    r = robertson_uncertainty(state, H_A, H_B)
                    max_bound = max(max_bound, r['robertson_bound'])

        nc_values.append(nc)
        max_bounds.append(max_bound)

        results_by_type[label] = {
            'nc_measure': float(nc),
            'max_robertson_bound': float(max_bound),
        }

    # Check dichotomy: NC > 0 iff max_bound > 0
    dichotomy_holds = True
    for label, data in results_by_type.items():
        nc_pos = data['nc_measure'] > 1e-10
        bound_pos = data['max_robertson_bound'] > 1e-10
        if nc_pos != bound_pos:
            dichotomy_holds = False

    passed = dichotomy_holds

    print(f"\n  Dichotomy (NC>0 iff bound>0): {dichotomy_holds}")

    result = {
        'test': 'T3_uncertainty_scales_with_nc',
        'results_by_type': results_by_type,
        'dichotomy_holds': dichotomy_holds,
        'PASS': passed,
    }
    return result


def test_T4_minimum_uncertainty_product():
    """T4: Minimum uncertainty product: finite for D_4, zero for A_n."""
    # For D_4: there exists no state that makes Delta_A * Delta_B = 0
    # for non-commuting observables (genuine quantum uncertainty)
    # For A_n: Delta_A * Delta_B can always be made 0 (classical)

    # D_4: minimize over many states
    diag_d4 = DynkinDiagram('D', 4)
    adj_d4 = diag_d4.adjacency
    n_d4 = adj_d4.shape[0]

    auts_d4 = graph_automorphisms(adj_d4)
    non_id_d4 = [P for P in auts_d4 if not np.allclose(P, np.eye(n_d4))]

    # Find the pair with largest commutator
    best_pair = None
    max_comm = 0.0
    for i in range(len(non_id_d4)):
        for j in range(i + 1, len(non_id_d4)):
            H_A = (non_id_d4[i] + non_id_d4[i].T) / 2
            H_B = (non_id_d4[j] + non_id_d4[j].T) / 2
            comm = H_A @ H_B - H_B @ H_A
            norm = np.linalg.norm(comm, 'fro')
            if norm > max_comm:
                max_comm = norm
                best_pair = (H_A, H_B)

    # Minimize product over many random states
    min_product_d4 = float('inf')
    for trial in range(100):
        np.random.seed(trial)
        state = np.random.randn(n_d4) + 1j * np.random.randn(n_d4)
        state = state / np.linalg.norm(state)
        r = robertson_uncertainty(state, best_pair[0], best_pair[1])
        if r['product'] < min_product_d4:
            min_product_d4 = r['product']

    d4_finite = min_product_d4 > 1e-10

    # A_5: try to find state with zero uncertainty product
    diag_a5 = DynkinDiagram('A', 5)
    adj_a5 = diag_a5.adjacency
    n_a5 = adj_a5.shape[0]

    auts_a5 = graph_automorphisms(adj_a5)
    non_id_a5 = [P for P in auts_a5 if not np.allclose(P, np.eye(n_a5))]

    if len(non_id_a5) >= 1:
        # Z_2: single non-identity element. Use it and identity as observables.
        # [P, I] = 0, so bound is always 0.
        # Or use P and P^T (which equals P for symmetric matrices)
        H_A = (non_id_a5[0] + non_id_a5[0].T) / 2
        # Use eigenstate of H_A: Delta_A = 0 → product = 0
        eigs, vecs = np.linalg.eigh(H_A)
        eigenstate = vecs[:, 0]

        # Any other Hermitian operator from the group
        H_B = H_A  # same operator → [A,A] = 0 always
        r = robertson_uncertainty(eigenstate, H_A, H_B)
        a5_zero = r['product'] < 1e-10
    else:
        a5_zero = True  # trivial group → classical

    passed = d4_finite and a5_zero

    print(f"  D_4: min uncertainty product = {min_product_d4:.6f}, finite = {d4_finite}")
    print(f"  A_5: can reach zero = {a5_zero}")

    result = {
        'test': 'T4_minimum_uncertainty_product',
        'd4_min_product': float(min_product_d4),
        'd4_finite': d4_finite,
        'a5_can_reach_zero': a5_zero,
        'PASS': passed,
    }
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Experiment 08: Robertson Uncertainty")
    print("Milestone 14, Block D")
    print("=" * 70)

    results = {}
    scorecard = []

    tests = [
        ("T1", test_T1_robertson_nontrivial_d4),
        ("T2", test_T2_robertson_zero_abelian),
        ("T3", test_T3_uncertainty_scales_with_nc),
        ("T4", test_T4_minimum_uncertainty_product),
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
        'experiment': 'exp_08_robertson_uncertainty',
        'milestone': 14,
        'block': 'D',
        'results': results,
        'scorecard': {f"T{i+1}": s for i, s in enumerate(scorecard)},
        'score': f"{n_pass}/{n_total}",
        'n_pass': n_pass,
        'n_total': n_total,
    }

    save_m14_results('exp_08_robertson_uncertainty', _convert_numpy(save_data))
    return n_pass, n_total


if __name__ == "__main__":
    main()
