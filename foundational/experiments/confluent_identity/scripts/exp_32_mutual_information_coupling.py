"""
exp_32_mutual_information_coupling.py -- Confluent Identity Phase 23

PURPOSE:
    Test whether the coupling ceiling at partial_rho ~0.40 is a Spearman
    limitation. All 8 weighting schemes used Spearman (monotonic only).
    Mutual information captures ANY dependence (U-shaped, threshold, etc.).
    Distance correlation provides a second nonlinear measure.

METHODS:
    1. Collect (coupling_w, natural_w, size_frac) using gradient scheme
    2. Baseline Spearman partial_rho (should replicate ~0.41)
    3. Bin-based MI: 10x10 histogram
    4. KNN-based MI: k=5, Kraskov estimator via KDTree
    5. MI on size-residualized ranks
    6. Permutation test: 5000 shuffles
    7. NMI = MI / sqrt(H(X)*H(Y))
    8. Distance correlation via pairwise distances

VERIFICATION:
    - MI(coupling, natural) permutation p < 0.01
    - Residualized MI > 0.05 nats (nonlinear signal survives deconfound)
    - NMI > Spearman_rho^2 by at least 0.05
    - Distance correlation of residuals > Spearman partial_rho by at least 0.03

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr, rankdata
from scipy.spatial import KDTree

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    compute_spectral_identity, get_parent_children_data,
)
from exp_08_gradient_coupling import (
    compute_coupling_weights_weighted, compute_natural_weights,
    compute_gradient_field,
)
from exp_14_partial_correlation import partial_spearman


def mutual_information_binned(x, y, n_bins=10):
    """
    Mutual information via 2D histogram.
    I(X;Y) = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    Returns MI in nats.
    """
    hist, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist / hist.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if pxy[i, j] > 1e-15 and px[i] > 1e-15 and py[j] > 1e-15:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    return float(mi)


def mutual_information_knn(x, y, k=5):
    """
    KNN-based mutual information (Kraskov-Stogbauer-Grassberger estimator).
    Uses digamma function for bias correction.
    Returns MI in nats.
    """
    from scipy.special import digamma

    n = len(x)
    if n < k + 1:
        return 0.0

    # Joint space
    xy = np.column_stack([x, y])
    tree_xy = KDTree(xy)

    # Find k-th neighbor distance in joint space (Chebyshev/max norm)
    mi_sum = 0.0
    for i in range(n):
        # k+1 because query point is in the tree
        dists, _ = tree_xy.query(xy[i], k=k + 1, p=np.inf)
        eps = dists[-1]  # distance to k-th neighbor

        if eps < 1e-15:
            eps = 1e-15

        # Count neighbors within eps in marginal spaces
        n_x = np.sum(np.abs(x - x[i]) < eps) - 1
        n_y = np.sum(np.abs(y - y[i]) < eps) - 1

        mi_sum += digamma(max(n_x, 1)) + digamma(max(n_y, 1))

    mi = digamma(k) - mi_sum / n + digamma(n)
    return float(max(mi, 0.0))


def entropy_1d(x, n_bins=10):
    """Shannon entropy of x in nats."""
    hist, _ = np.histogram(x, bins=n_bins)
    p = hist / hist.sum()
    return float(-np.sum(p[p > 1e-15] * np.log(p[p > 1e-15])))


def distance_correlation(x, y):
    """
    Distance correlation (dCor) between x and y.
    Detects any dependence, including nonlinear and non-monotonic.
    Returns dCor in [0, 1].
    """
    n = len(x)
    if n < 4:
        return 0.0

    # Distance matrices
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])

    # Double-center
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    dcov2 = float(np.mean(A * B))
    dvar_x = float(np.mean(A * A))
    dvar_y = float(np.mean(B * B))

    if dvar_x < 1e-15 or dvar_y < 1e-15:
        return 0.0

    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y)))


def residualize_on_size(x, size):
    """Rank-residualize x on size (same as partial_spearman does internally)."""
    rx = rankdata(x)
    rs = rankdata(size)
    X = np.column_stack([np.ones(len(x)), rs])
    beta = np.linalg.lstsq(X, rx, rcond=None)[0]
    return rx - X @ beta


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 23, Experiment 32")
    print("Mutual Information Coupling: Breaking the Spearman Ceiling?")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency and gradient field...")
    adjacency = build_lattice_adjacency(C)
    grad_mag = compute_gradient_field(C)
    grad_flat = grad_mag.ravel()

    # =====================================================================
    # Collect coupling data (gradient scheme)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Collecting Coupling Data (Gradient Scheme)")
    print(f"{'=' * 70}")

    coupling_all = []
    natural_all = []
    size_all = []

    n_parents = 0
    for (level, pid), parent_indices, children_list, L_parent, state_parent in \
            get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):

        n_parents += 1
        identity_parent = compute_spectral_identity(L_parent, state_parent)
        eigvecs_parent = identity_parent.get('eigenvectors')
        if eigvecs_parent is None:
            continue

        natural_weights, size_fractions = compute_natural_weights(
            state_flat, parent_indices, children_list, eigvecs_parent
        )

        w_gradient = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, grad_flat
        )

        for child_id, _ in children_list:
            cid = child_id
            if cid in w_gradient:
                coupling_all.append(w_gradient[cid])
                natural_all.append(natural_weights.get(cid, 0))
                size_all.append(size_fractions.get(cid, 0))

    coupling = np.array(coupling_all)
    natural = np.array(natural_all)
    size = np.array(size_all)
    n = len(coupling)
    print(f"  Collected {n} measurements from {n_parents} parents")

    # =====================================================================
    # Baseline: Spearman
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Baseline: Spearman Correlation")
    print(f"{'=' * 70}")

    rho_raw, p_raw = spearmanr(coupling, natural)
    partial_rho, partial_p = partial_spearman(coupling, natural, size)
    print(f"  Raw rho(coupling, natural) = {rho_raw:.4f}, p={p_raw:.2e}")
    print(f"  Partial rho(| size) = {partial_rho:.4f}, p={partial_p:.2e}")

    # =====================================================================
    # Mutual Information
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Mutual Information Analysis")
    print(f"{'=' * 70}")

    # Raw MI
    mi_binned = mutual_information_binned(coupling, natural, n_bins=10)
    mi_knn = mutual_information_knn(coupling, natural, k=5)
    print(f"  MI (binned, 10x10): {mi_binned:.4f} nats")
    print(f"  MI (KNN, k=5):      {mi_knn:.4f} nats")

    # Entropies for NMI
    h_coupling = entropy_1d(coupling, n_bins=10)
    h_natural = entropy_1d(natural, n_bins=10)
    nmi = mi_binned / (np.sqrt(h_coupling * h_natural) + 1e-15)
    print(f"  H(coupling) = {h_coupling:.4f}, H(natural) = {h_natural:.4f}")
    print(f"  NMI = {nmi:.4f}")

    # Residualized MI (after removing size)
    coupling_resid = residualize_on_size(coupling, size)
    natural_resid = residualize_on_size(natural, size)
    mi_resid_binned = mutual_information_binned(coupling_resid, natural_resid, n_bins=10)
    mi_resid_knn = mutual_information_knn(coupling_resid, natural_resid, k=5)
    print(f"\n  Residualized MI (binned): {mi_resid_binned:.4f} nats")
    print(f"  Residualized MI (KNN):    {mi_resid_knn:.4f} nats")

    # =====================================================================
    # Distance Correlation
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Distance Correlation")
    print(f"{'=' * 70}")

    dcor_raw = distance_correlation(coupling, natural)
    dcor_resid = distance_correlation(coupling_resid, natural_resid)
    print(f"  dCor(coupling, natural) = {dcor_raw:.4f}")
    print(f"  dCor(residualized)      = {dcor_resid:.4f}")

    # =====================================================================
    # Permutation test for MI
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Permutation Test (5000 shuffles)")
    print(f"{'=' * 70}")

    rng = np.random.RandomState(42)
    n_perm = 5000
    mi_perm = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        mi_perm[i] = mutual_information_binned(coupling[perm], natural, n_bins=10)

    mi_p_value = float(np.mean(mi_perm >= mi_binned))
    mi_resid_perm = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        mi_resid_perm[i] = mutual_information_binned(coupling_resid[perm], natural_resid, n_bins=10)

    mi_resid_p = float(np.mean(mi_resid_perm >= mi_resid_binned))

    print(f"  MI permutation p-value:    {mi_p_value:.4f}")
    print(f"  MI(resid) permutation p:   {mi_resid_p:.4f}")
    print(f"  MI 95th percentile (null):  {np.percentile(mi_perm, 95):.4f}")
    print(f"  MI(resid) 95th pct (null):  {np.percentile(mi_resid_perm, 95):.4f}")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Test 1: MI significant
    test1 = mi_p_value < 0.01
    print(f"\n  Test 1: MI permutation p < 0.01?")
    print(f"    p = {mi_p_value:.4f}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Residualized MI > 0.05 nats
    test2 = mi_resid_binned > 0.05
    print(f"\n  Test 2: Residualized MI > 0.05 nats?")
    print(f"    MI(resid) = {mi_resid_binned:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: NMI > Spearman_rho^2 + 0.05
    spearman_r2 = rho_raw ** 2
    nmi_excess = nmi - spearman_r2
    test3 = nmi_excess > 0.05
    print(f"\n  Test 3: NMI > Spearman rho^2 + 0.05?")
    print(f"    NMI = {nmi:.4f}, rho^2 = {spearman_r2:.4f}, excess = {nmi_excess:.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: dCor(resid) > partial_rho + 0.03
    dcor_excess = dcor_resid - abs(partial_rho)
    test4 = dcor_excess > 0.03
    print(f"\n  Test 4: dCor(resid) > |partial_rho| + 0.03?")
    print(f"    dCor(resid) = {dcor_resid:.4f}, |partial_rho| = {abs(partial_rho):.4f}, "
          f"excess = {dcor_excess:.4f}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 mutual information tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_32_mutual_information_coupling',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Nonlinear coupling: MI and distance correlation vs Spearman ceiling',
        'n_measurements': n,
        'n_parents': n_parents,
        'spearman': {
            'rho_raw': float(rho_raw),
            'partial_rho': float(partial_rho),
        },
        'mutual_information': {
            'mi_binned': mi_binned,
            'mi_knn': mi_knn,
            'nmi': float(nmi),
            'mi_resid_binned': mi_resid_binned,
            'mi_resid_knn': mi_resid_knn,
            'mi_p_value': mi_p_value,
            'mi_resid_p_value': mi_resid_p,
        },
        'distance_correlation': {
            'dcor_raw': dcor_raw,
            'dcor_resid': dcor_resid,
        },
        'verification': {
            'test1_mi_significant': bool(test1),
            'test2_resid_mi_above_005': bool(test2),
            'test3_nmi_exceeds_spearman': bool(test3),
            'test4_dcor_exceeds_partial': bool(test4),
            'n_verified': n_verified,
        },
    }

    output_file = RESULTS_DIR / f'exp_32_mutual_info_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
