"""
exp_16 -- Alternative Complement Metrics

Milestone 13.5, Investigation Experiment

Central question: The Gram matrix of complement-differences is PSD not PD
(exp_05 T4, exp_14 T3). This is because vertices in the same orbit produce
identical complement spectra, giving zero-distance pairs. Can an ALTERNATIVE
complement metric break this degeneracy?

Three alternative metrics tested:
  1. Heat kernel trace: Tr(exp(-t * L_complement)) at multiple diffusion times
  2. Characteristic polynomial coefficients: det(xI - A_complement)
  3. Spectral zeta function: sum(|lambda_i|^(-s)) at multiple s-values

The key test: does the Gram matrix of the alternative metric become positive
DEFINITE (not just PSD)? If yes, the PSD issue is an artifact of the eigenvalue-
norm metric. If no, the degeneracy is fundamental to the complement structure.

Tests:
  T1: Heat kernel Gram matrix -- PD or PSD on A_5, D_4?
  T2: Characteristic polynomial Gram matrix -- PD or PSD?
  T3: Spectral zeta Gram matrix -- PD or PSD?
  T4: Combined metric (all three concatenated) -- PD or PSD?
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from identity_complement import (
    PHI, LN_PHI,
    DynkinDiagram,
    complement_spectrum,
    vertex_orbits,
    save_m13_results, _convert_numpy,
)


def complement_laplacian_spectrum(adjacency, vertex):
    """Eigenvalues of the graph Laplacian of the complement subgraph."""
    n = adjacency.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[vertex] = False
    sub = adjacency[np.ix_(mask, mask)]
    # Laplacian = D - A
    degree = np.diag(np.sum(sub, axis=1))
    L = degree - sub
    return np.sort(np.linalg.eigvalsh(L))


def heat_kernel_signature(adjacency, vertex, times=None):
    """
    Heat kernel trace Tr(exp(-t*L)) of the complement subgraph at multiple times.

    The heat kernel encodes multi-scale geometric information. Different
    diffusion times t probe different scales of the complement structure.
    """
    if times is None:
        times = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    laplacian_eigs = complement_laplacian_spectrum(adjacency, vertex)
    signature = []
    for t in times:
        trace = float(np.sum(np.exp(-t * laplacian_eigs)))
        signature.append(trace)
    return np.array(signature)


def char_poly_coefficients(adjacency, vertex):
    """
    Coefficients of the characteristic polynomial of the complement adjacency.

    det(xI - A) = x^n + c_{n-1}*x^{n-1} + ... + c_0
    These are symmetric functions of the eigenvalues and encode the full
    spectral information without lossy sorting.
    """
    n = adjacency.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[vertex] = False
    sub = adjacency[np.ix_(mask, mask)]
    coeffs = np.polynomial.polynomial.polyfromroots(np.linalg.eigvalsh(sub))
    return coeffs


def spectral_zeta_signature(adjacency, vertex, s_values=None):
    """
    Spectral zeta function sum(|lambda_i|^{-s}) at multiple s-values.

    Uses the adjacency spectrum (not Laplacian) with regularization for
    zero eigenvalues.
    """
    if s_values is None:
        s_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    spec = complement_spectrum(adjacency, vertex)
    # Regularize: replace zeros with small epsilon
    eps = 1e-10
    abs_spec = np.abs(spec) + eps
    signature = []
    for s in s_values:
        zeta_val = float(np.sum(abs_spec ** (-s)))
        signature.append(zeta_val)
    return np.array(signature)


def compute_gram_matrix(adjacency, metric_fn):
    """
    Compute the Gram matrix of pairwise distances using a given metric.

    metric_fn(adjacency, vertex) -> feature vector

    Returns (Gram matrix, min eigenvalue, is_PD, feature_vectors).
    """
    n = adjacency.shape[0]
    features = []
    for v in range(n):
        feat = metric_fn(adjacency, v)
        features.append(feat)

    features = np.array(features)

    # Gram matrix: G_{ij} = <f_i, f_j>
    G = features @ features.T

    # Also compute distance-based Gram: G_{ij} = -0.5 * (d^2_{ij} - d^2_{i0} - d^2_{0j} + d^2_{00})
    # But the feature inner product is more natural for PD testing

    eigs = np.linalg.eigvalsh(G)
    min_eig = float(np.min(eigs))
    is_pd = min_eig > 1e-10

    return G, min_eig, is_pd, features


def compute_distance_gram(adjacency, metric_fn):
    """
    Compute the centered distance Gram matrix.

    This tests whether the DISTANCE metric (not inner product) embeds
    in a Euclidean space with positive definite Gram matrix.
    """
    n = adjacency.shape[0]
    features = []
    for v in range(n):
        feat = metric_fn(adjacency, v)
        features.append(feat)

    features = np.array(features)

    # Pairwise distance matrix
    D2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D2[i, j] = np.sum((features[i] - features[j]) ** 2)

    # Double-centering to get Gram matrix: G = -0.5 * J * D2 * J
    # where J = I - (1/n)*11^T
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ D2 @ J

    eigs = np.linalg.eigvalsh(G)
    min_eig = float(np.min(eigs))
    # PD requires all eigenvalues > 0, but centered matrix always has one zero eigenvalue
    # So check if the second-smallest is > 0
    sorted_eigs = np.sort(eigs)
    second_min = float(sorted_eigs[1]) if len(sorted_eigs) > 1 else 0.0
    # The centered Gram has rank at most n-1, so we check n-1 positive eigenvalues
    n_positive = int(np.sum(eigs > 1e-10))
    is_full_rank = n_positive >= n - 1  # max possible rank for centered matrix

    return G, min_eig, second_min, is_full_rank, n_positive, features


def test_metric_on_diagrams(metric_name, metric_fn, diagrams):
    """Test a metric across multiple ADE diagrams for PD/PSD status."""
    results = {}
    n_pd = 0
    n_full_rank = 0

    for dname, dtype, drank in diagrams:
        d = DynkinDiagram(dtype, drank)
        adj = d.adjacency
        n = adj.shape[0]
        orbits = vertex_orbits(adj)
        n_orbits = len(orbits)

        # Feature inner product Gram
        G_feat, min_eig_feat, is_pd_feat, features = compute_gram_matrix(adj, metric_fn)

        # Distance-based centered Gram
        G_dist, min_eig_dist, second_min_dist, is_full_rank, n_positive, _ = \
            compute_distance_gram(adj, metric_fn)

        # Check if metric distinguishes all orbits
        feature_by_orbit = {}
        for orbit in orbits:
            rep = orbit[0]
            feat = features[rep]
            key = tuple(np.round(feat, decimals=8))
            feature_by_orbit[key] = feature_by_orbit.get(key, 0) + 1

        n_distinct_features = len(feature_by_orbit)
        distinguishes_orbits = n_distinct_features == n_orbits

        result = {
            'diagram': dname,
            'n_vertices': n,
            'n_orbits': n_orbits,
            'feat_gram_min_eig': min_eig_feat,
            'feat_gram_is_PD': is_pd_feat,
            'dist_gram_second_min_eig': second_min_dist,
            'dist_gram_is_full_rank': is_full_rank,
            'dist_gram_n_positive': n_positive,
            'n_distinct_features': n_distinct_features,
            'distinguishes_orbits': distinguishes_orbits,
        }
        results[dname] = result

        if is_pd_feat:
            n_pd += 1
        if is_full_rank:
            n_full_rank += 1

        print(f"    {dname} (n={n}, orbits={n_orbits}): "
              f"feat PD={is_pd_feat} (min_eig={min_eig_feat:.2e}), "
              f"dist full_rank={is_full_rank} ({n_positive}/{n-1}), "
              f"orbit distinguish={distinguishes_orbits}")

    return results, n_pd, n_full_rank


def test_T1_heat_kernel():
    """T1: Heat kernel Gram matrix -- PD or PSD on ADE diagrams?"""
    diagrams = [
        ('A_3', 'A', 3), ('A_5', 'A', 5), ('A_7', 'A', 7),
        ('D_4', 'D', 4), ('D_6', 'D', 6),
        ('E_6', 'E', 6),
    ]

    results, n_pd, n_full_rank = test_metric_on_diagrams(
        'heat_kernel', heat_kernel_signature, diagrams
    )

    # Heat kernel should distinguish orbits (it encodes more info than eigenvalues)
    n_distinguishes = sum(1 for r in results.values() if r['distinguishes_orbits'])

    # PD means the degeneracy is broken
    any_pd = n_pd > 0
    all_distinguish = n_distinguishes == len(diagrams)

    print(f"\n  Heat kernel: {n_pd}/{len(diagrams)} PD, "
          f"{n_full_rank}/{len(diagrams)} full-rank distance, "
          f"{n_distinguishes}/{len(diagrams)} orbit-distinguishing")

    result = {
        'test': 'T1_heat_kernel',
        'metric': 'heat_kernel_trace',
        'diagrams': results,
        'n_pd': n_pd,
        'n_full_rank': n_full_rank,
        'n_distinguishes': n_distinguishes,
        'any_pd': any_pd,
        'note': 'Heat kernel Tr(exp(-tL)) at t=[0.1,0.5,1,2,5,10] encodes multi-scale '
                'geometry. PD would mean the PSD issue is metric-specific, not fundamental.',
        'PASS': any_pd,
    }
    return result


def test_T2_char_poly():
    """T2: Characteristic polynomial Gram matrix -- PD or PSD?"""
    diagrams = [
        ('A_3', 'A', 3), ('A_5', 'A', 5), ('A_7', 'A', 7),
        ('D_4', 'D', 4), ('D_6', 'D', 6),
        ('E_6', 'E', 6),
    ]

    results, n_pd, n_full_rank = test_metric_on_diagrams(
        'char_poly', char_poly_coefficients, diagrams
    )

    n_distinguishes = sum(1 for r in results.values() if r['distinguishes_orbits'])
    any_pd = n_pd > 0

    print(f"\n  Char poly: {n_pd}/{len(diagrams)} PD, "
          f"{n_full_rank}/{len(diagrams)} full-rank distance, "
          f"{n_distinguishes}/{len(diagrams)} orbit-distinguishing")

    result = {
        'test': 'T2_char_poly',
        'metric': 'characteristic_polynomial_coefficients',
        'diagrams': results,
        'n_pd': n_pd,
        'n_full_rank': n_full_rank,
        'n_distinguishes': n_distinguishes,
        'any_pd': any_pd,
        'note': 'Characteristic polynomial coefficients are symmetric functions of eigenvalues. '
                'They encode the same spectral data as eigenvalues but in a different basis.',
        'PASS': any_pd,
    }
    return result


def test_T3_spectral_zeta():
    """T3: Spectral zeta Gram matrix -- PD or PSD?"""
    diagrams = [
        ('A_3', 'A', 3), ('A_5', 'A', 5), ('A_7', 'A', 7),
        ('D_4', 'D', 4), ('D_6', 'D', 6),
        ('E_6', 'E', 6),
    ]

    results, n_pd, n_full_rank = test_metric_on_diagrams(
        'spectral_zeta', spectral_zeta_signature, diagrams
    )

    n_distinguishes = sum(1 for r in results.values() if r['distinguishes_orbits'])
    any_pd = n_pd > 0

    print(f"\n  Spectral zeta: {n_pd}/{len(diagrams)} PD, "
          f"{n_full_rank}/{len(diagrams)} full-rank distance, "
          f"{n_distinguishes}/{len(diagrams)} orbit-distinguishing")

    result = {
        'test': 'T3_spectral_zeta',
        'metric': 'spectral_zeta_function',
        'diagrams': results,
        'n_pd': n_pd,
        'n_full_rank': n_full_rank,
        'n_distinguishes': n_distinguishes,
        'any_pd': any_pd,
        'note': 'Spectral zeta sum(|lambda|^{-s}) at s=[0.5,1,1.5,2,3]. Non-linear '
                'transform of eigenvalues might break degeneracy.',
        'PASS': any_pd,
    }
    return result


def test_T4_combined_metric():
    """T4: Combined metric (all three) -- PD or PSD?"""
    diagrams = [
        ('A_3', 'A', 3), ('A_5', 'A', 5), ('A_7', 'A', 7),
        ('D_4', 'D', 4), ('D_6', 'D', 6),
        ('E_6', 'E', 6),
    ]

    def combined_metric(adjacency, vertex):
        hk = heat_kernel_signature(adjacency, vertex)
        cp = char_poly_coefficients(adjacency, vertex)
        sz = spectral_zeta_signature(adjacency, vertex)
        return np.concatenate([hk, cp, sz])

    results, n_pd, n_full_rank = test_metric_on_diagrams(
        'combined', combined_metric, diagrams
    )

    n_distinguishes = sum(1 for r in results.values() if r['distinguishes_orbits'])
    any_pd = n_pd > 0

    print(f"\n  Combined: {n_pd}/{len(diagrams)} PD, "
          f"{n_full_rank}/{len(diagrams)} full-rank distance, "
          f"{n_distinguishes}/{len(diagrams)} orbit-distinguishing")

    # The key question: does combining all metrics break the degeneracy?
    # If NONE of the individual or combined metrics are PD, the degeneracy is
    # truly fundamental to the complement structure (same-orbit vertices have
    # identical complements -> identical features -> zero distance -> PSD).
    all_psd = n_pd == 0

    result = {
        'test': 'T4_combined_metric',
        'metric': 'heat_kernel + char_poly + spectral_zeta',
        'diagrams': results,
        'n_pd': n_pd,
        'n_full_rank': n_full_rank,
        'n_distinguishes': n_distinguishes,
        'any_pd': any_pd,
        'all_psd': all_psd,
        'note': 'Concatenation of all three metrics. If still PSD, the degeneracy is '
                'fundamental: same-orbit vertices have isomorphic complements, so ANY '
                'isomorphism-invariant metric assigns them the same feature vector.',
        'PASS': any_pd,
    }
    return result


def main():
    print("=" * 70)
    print("EXP 16 -- Alternative Complement Metrics")
    print("Milestone 13.5, Investigation Experiment")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_heat_kernel),
        ('T2', test_T2_char_poly),
        ('T3', test_T3_spectral_zeta),
        ('T4', test_T4_combined_metric),
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
        'experiment': 'exp_16_alternative_complement_metrics',
        'milestone': 'milestone13.5',
        'block': 'investigation',
        'version': 'v0.1',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m13_results('exp_16_alternative_complement_metrics', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()
