"""
exp_05 -- Complement-Transformation Algebra Embeds in Lie Algebra (HARDENED v0.3)

Milestone 13, Block B (Complement-Transformations & Weyl Groups)

Hypothesis: The complement-transformation algebra embeds in the ADE Lie algebra.
Complement-difference vectors align with root directions (HIGH RISK), path
independence generalises to weighted/cyclic graphs (non-trivial), the rank of the
complement-difference matrix grows monotonically with graph size, and the
inner product of adjacent complement-differences is positive-definite (EXPECTED FAIL).

Tests:
  T1: Root alignment cos>0.8 at 75% for A_4, A_5, D_4; adversarial random graph fails
  T2: Path independence on WEIGHTED A_5 (epsilon-perturbed) + cycle closure on C_5
  T3: Rank monotonicity across A_3..A_8; D_4 rank differs from A_4 rank
  T4: Gram matrix strictly positive-definite (min eig > 1e-8) — EXPECTED FAIL
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from identity_complement import (
    PHI, INV_PHI,
    DynkinDiagram,
    complement_spectrum, complement_transformation,
    save_m13_results, _convert_numpy,
)


def _cosine_similarity(a, b):
    """Cosine similarity between two vectors, handling zero vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _build_random_symmetric_graph(n, density=0.4, seed=42):
    """Build a random symmetric graph that is NOT an ADE Dynkin diagram."""
    rng = np.random.RandomState(seed)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                A[i, j] = 1.0
                A[j, i] = 1.0
    # Ensure connected: add chain edges if disconnected
    for i in range(n - 1):
        if A[i, i + 1] == 0:
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
    return A


def _build_cycle_graph(n):
    """Build a cycle graph C_n."""
    A = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def _root_alignment_for_graph(diag, cos_threshold=0.8, frac_threshold=0.75):
    """
    Compute root alignment for one ADE Dynkin diagram.

    Returns (fraction_aligned, n_pairs, alignments_detail, pass_flag).
    """
    adj = diag.adjacency
    n = adj.shape[0]
    cartan = diag.cartan_matrix()

    # Collect adjacent complement-difference vectors
    adj_diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                t = complement_transformation(adj, i, j)
                adj_diffs.append({
                    'pair': (i, j),
                    'spectral_diff': t['spectral_diff'],
                })

    # Simple root directions from Cartan matrix
    root_vectors = [cartan[i, :] for i in range(n)]

    # For each complement-diff, find best cosine similarity with any root vector
    alignments = []
    for ad in adj_diffs:
        sd = ad['spectral_diff']
        best_cos = 0.0
        best_root_idx = -1

        for ri, root in enumerate(root_vectors):
            dim = min(len(sd), len(root))
            cos = abs(_cosine_similarity(sd[:dim], root[:dim]))
            if cos > best_cos:
                best_cos = cos
                best_root_idx = ri

        alignments.append({
            'pair': ad['pair'],
            'best_cosine': best_cos,
            'best_root_index': best_root_idx,
        })

    n_aligned = sum(1 for a in alignments if a['best_cosine'] > cos_threshold)
    fraction = n_aligned / len(alignments) if alignments else 0.0
    passed = fraction >= frac_threshold

    return fraction, len(alignments), alignments, passed


def test_T1_complement_diff_root_alignment():
    """T1: Root alignment cos>0.8 at 75% for A_4, A_5, D_4 + adversarial random fail."""
    cos_threshold = 0.8
    frac_threshold = 0.75

    # --- ADE graphs: A_4, A_5, D_4 ---
    ade_graphs = [
        DynkinDiagram('A', 4),
        DynkinDiagram('A', 5),
        DynkinDiagram('D', 4),
    ]

    ade_results = {}
    ade_pass_count = 0
    for diag in ade_graphs:
        frac, n_pairs, alignments, passed = _root_alignment_for_graph(
            diag, cos_threshold, frac_threshold
        )
        ade_results[diag.name] = {
            'n_pairs': n_pairs,
            'fraction_aligned': frac,
            'passed': passed,
            'alignments': alignments,
        }
        if passed:
            ade_pass_count += 1
        print(f"    {diag.name}: {frac:.2%} aligned (cos>{cos_threshold}), "
              f"{n_pairs} pairs, {'PASS' if passed else 'FAIL'}")

    # --- Adversarial: random symmetric graph (NOT ADE) ---
    # Use n=6 to have enough pairs for statistics
    random_adj = _build_random_symmetric_graph(6, density=0.4, seed=42)
    n_rand = random_adj.shape[0]

    # Build a fake "Cartan" from this random graph (2I - A) to use as root directions
    rand_cartan = 2 * np.eye(n_rand) - random_adj
    rand_roots = [rand_cartan[i, :] for i in range(n_rand)]

    # Collect adjacent complement-diffs for the random graph
    rand_diffs = []
    for i in range(n_rand):
        for j in range(i + 1, n_rand):
            if random_adj[i, j] > 0:
                t = complement_transformation(random_adj, i, j)
                rand_diffs.append(t['spectral_diff'])

    rand_cosines = []
    for sd in rand_diffs:
        best_cos = 0.0
        for root in rand_roots:
            dim = min(len(sd), len(root))
            cos = abs(_cosine_similarity(sd[:dim], root[:dim]))
            if cos > best_cos:
                best_cos = cos
        rand_cosines.append(best_cos)

    # For adversarial: MAJORITY of pairs should have cos < 0.5
    n_low = sum(1 for c in rand_cosines if c < 0.5)
    adversarial_fails = n_low > len(rand_cosines) / 2
    print(f"    Random graph: {n_low}/{len(rand_cosines)} pairs have cos<0.5, "
          f"adversarial {'FAIL (good)' if adversarial_fails else 'PASS (bad)'}")

    # PASS: >=2 of 3 ADE graphs pass AND adversarial fails
    overall = ade_pass_count >= 2 and adversarial_fails

    result = {
        'test': 'T1_complement_diff_root_alignment',
        'cos_threshold': cos_threshold,
        'frac_threshold': frac_threshold,
        'ade_graphs': ade_results,
        'ade_pass_count': ade_pass_count,
        'adversarial': {
            'n_pairs': len(rand_cosines),
            'cosines': rand_cosines,
            'n_below_0.5': n_low,
            'majority_fail': adversarial_fails,
        },
        'risk': 'HIGH — complement eigenvalue diffs live in different space than roots',
        'PASS': overall,
    }
    return result


def test_T2_path_independence_weighted_and_cyclic():
    """T2: Path independence on weighted A_5 + cycle closure on C_5."""
    # --- Part A: Weighted A_5 ---
    # Take A_5 adjacency, perturb edge weights by epsilon*random
    epsilon = 0.01
    rng = np.random.RandomState(99)

    diag = DynkinDiagram('A', 5)
    adj_base = diag.adjacency.copy()
    n = adj_base.shape[0]

    # Build weighted adjacency: add epsilon*uniform noise to existing edges
    adj_w = adj_base.copy()
    for i in range(n):
        for j in range(i + 1, n):
            if adj_base[i, j] > 0:
                noise = epsilon * rng.uniform(-1, 1)
                adj_w[i, j] += noise
                adj_w[j, i] += noise

    # Path: 0 -> 1 -> 2 -> 3 -> 4
    path = list(range(n))
    cumulative_diff = np.zeros(n - 1)  # complement spectra have n-1 components
    step_diffs_w = []
    for i in range(len(path) - 1):
        t = complement_transformation(adj_w, path[i], path[i + 1])
        cumulative_diff += t['spectral_diff']
        step_diffs_w.append(t['spectral_diff'].tolist())

    # Direct transformation on weighted graph
    t_direct = complement_transformation(adj_w, path[0], path[-1])
    direct_diff = t_direct['spectral_diff']

    weighted_error = float(np.linalg.norm(cumulative_diff - direct_diff))
    print(f"    Weighted A_5 (eps={epsilon}): path vs direct error = {weighted_error:.2e}")

    # --- Part B: Cycle closure on C_5 ---
    # Sum of complement-diffs around full cycle 0->1->2->3->4->0 should be ~zero
    cycle_adj = _build_cycle_graph(5)
    n_c = cycle_adj.shape[0]

    cycle_sum = np.zeros(n_c - 1)
    cycle_steps = []
    for i in range(n_c):
        j = (i + 1) % n_c
        t = complement_transformation(cycle_adj, i, j)
        cycle_sum += t['spectral_diff']
        cycle_steps.append({
            'from': i, 'to': j,
            'diff': t['spectral_diff'].tolist(),
        })

    cycle_error = float(np.linalg.norm(cycle_sum))
    print(f"    C_5 cycle closure: error = {cycle_error:.2e}")

    # PASS: weighted path error < 0.05 AND cycle closure error < 1e-6
    weighted_ok = weighted_error < 0.05
    cycle_ok = cycle_error < 1e-6

    result = {
        'test': 'T2_path_independence_weighted_and_cyclic',
        'weighted_graph': {
            'base': 'A_5',
            'epsilon': epsilon,
            'path': path,
            'step_diffs': step_diffs_w,
            'cumulative_vs_direct_error': weighted_error,
            'pass': weighted_ok,
        },
        'cycle_closure': {
            'graph': 'C_5',
            'cycle': [0, 1, 2, 3, 4, 0],
            'steps': cycle_steps,
            'closure_error': cycle_error,
            'pass': cycle_ok,
        },
        'PASS': weighted_ok and cycle_ok,
    }
    return result


def test_T3_rank_monotonicity():
    """T3: Rank monotonicity across A_3..A_8 + D_4 rank differs from A_4 rank."""
    # --- Part A: Monotonicity across A_3..A_8 ---
    ranks_to_test = list(range(3, 9))  # A_3, A_4, A_5, A_6, A_7, A_8
    a_ranks = {}

    for r in ranks_to_test:
        diag = DynkinDiagram('A', r)
        adj = diag.adjacency
        n = adj.shape[0]

        # Collect all pairwise spectral diffs
        diff_vectors = []
        for i in range(n):
            for j in range(i + 1, n):
                t = complement_transformation(adj, i, j)
                diff_vectors.append(t['spectral_diff'])

        M = np.array(diff_vectors)
        if M.size > 0:
            sv = np.linalg.svd(M, compute_uv=False)
            numerical_rank = int(np.sum(sv > 1e-10))
        else:
            numerical_rank = 0

        a_ranks[r] = numerical_rank
        print(f"    A_{r}: diff matrix rank = {numerical_rank} "
              f"(spec_dim = {n - 1})")

    # Check monotonicity: rank should be non-decreasing
    rank_values = [a_ranks[r] for r in ranks_to_test]
    monotonic = all(rank_values[i] <= rank_values[i + 1]
                    for i in range(len(rank_values) - 1))
    print(f"    Monotonicity: {rank_values} -> {'YES' if monotonic else 'NO'}")

    # --- Part B: D_4 rank != A_4 rank ---
    diag_d4 = DynkinDiagram('D', 4)
    adj_d4 = diag_d4.adjacency
    n_d4 = adj_d4.shape[0]

    diff_vectors_d4 = []
    for i in range(n_d4):
        for j in range(i + 1, n_d4):
            t = complement_transformation(adj_d4, i, j)
            diff_vectors_d4.append(t['spectral_diff'])

    M_d4 = np.array(diff_vectors_d4)
    if M_d4.size > 0:
        sv_d4 = np.linalg.svd(M_d4, compute_uv=False)
        d4_rank = int(np.sum(sv_d4 > 1e-10))
    else:
        d4_rank = 0

    a4_rank = a_ranks[4]
    rank_differs = d4_rank != a4_rank
    print(f"    D_4 rank = {d4_rank}, A_4 rank = {a4_rank} -> "
          f"{'DIFFER (good)' if rank_differs else 'SAME (bad)'}")

    # PASS: monotonicity holds AND D_4 rank != A_4 rank
    overall = monotonic and rank_differs

    result = {
        'test': 'T3_rank_monotonicity',
        'a_series': {str(r): {'rank': a_ranks[r]} for r in ranks_to_test},
        'rank_sequence': rank_values,
        'monotonic': monotonic,
        'd4_rank': d4_rank,
        'a4_rank': a4_rank,
        'rank_differs': rank_differs,
        'PASS': overall,
    }
    return result


def test_T4_gram_matrix_strictly_positive_definite():
    """T4: Gram matrix strictly positive-definite (min eig > 1e-8) — EXPECTED FAIL."""
    strict_threshold = 1e-8

    graphs = [
        DynkinDiagram('A', 4),
        DynkinDiagram('A', 5),
        DynkinDiagram('D', 4),
    ]

    all_strict = True
    graph_results = {}

    for diag in graphs:
        adj = diag.adjacency
        n = adj.shape[0]

        # Collect adjacent pair diff vectors
        adj_pairs = []
        diff_vectors = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    t = complement_transformation(adj, i, j)
                    adj_pairs.append((i, j))
                    diff_vectors.append(t['spectral_diff'])

        # Build Gram matrix
        k = len(diff_vectors)
        gram = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                gram[i, j] = np.dot(diff_vectors[i], diff_vectors[j])

        eigenvalues = np.linalg.eigvalsh(gram)
        min_eig = float(np.min(eigenvalues))
        is_strict = bool(np.all(eigenvalues > strict_threshold))

        if not is_strict:
            all_strict = False

        graph_results[diag.name] = {
            'n_adjacent_pairs': k,
            'gram_eigenvalues': sorted(eigenvalues.tolist()),
            'min_eigenvalue': min_eig,
            'strictly_positive_definite': is_strict,
        }

        print(f"    {diag.name}: {k} pairs, min eig = {min_eig:.6e}, "
              f"strict PD = {is_strict}")

    # EXPECTED FAIL: A_4 likely has exact-zero eigenvalues from symmetric pairs.
    # We document this honestly.
    if not all_strict:
        print("    NOTE: Expected failure — symmetric vertex pairs produce "
              "linearly dependent complement-diffs, yielding zero Gram eigenvalues.")
        print("    This is an HONEST FAILURE: strict positive-definiteness "
              "requires all complement-diffs to be linearly independent,")
        print("    but ADE graph symmetries force some pairs to have identical "
              "complement spectra (and thus zero difference vectors).")

    result = {
        'test': 'T4_gram_matrix_strictly_positive_definite',
        'strict_threshold': strict_threshold,
        'graphs': graph_results,
        'all_strictly_positive_definite': all_strict,
        'expected_fail_reason': (
            'ADE graph symmetries produce vertex pairs with identical '
            'complement spectra. Their spectral-diff vectors are zero or '
            'linearly dependent, making the Gram matrix singular. '
            'A_4 specifically has mirror symmetry (node 0<->3, 1<->2), '
            'so complement-diffs for symmetric edges are identical, '
            'yielding at least one zero Gram eigenvalue.'
        ),
        'risk': 'EXPECTED FAIL — symmetric pairs guarantee zero eigenvalues',
        'PASS': all_strict,
    }
    return result


def main():
    print("=" * 70)
    print("EXP 05 -- Complement-Transformation Algebra Embeds in Lie Algebra")
    print("Milestone 13, Block B  (HARDENED v0.3)")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_complement_diff_root_alignment),
        ('T2', test_T2_path_independence_weighted_and_cyclic),
        ('T3', test_T3_rank_monotonicity),
        ('T4', test_T4_gram_matrix_strictly_positive_definite),
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
        'experiment': 'exp_05_complement_algebra_embeds_lie',
        'milestone': 'milestone13',
        'block': 'B',
        'hardening': 'v0.3',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m13_results('exp_05_complement_algebra_embeds_lie', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()
