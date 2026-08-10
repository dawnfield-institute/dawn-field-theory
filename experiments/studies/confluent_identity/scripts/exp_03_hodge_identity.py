"""
exp_03_hodge_identity.py -- Confluent Identity Phase 1

PURPOSE:
    Compute the spectral identity I(R) for each region in the hierarchical
    partition, and the coupling weights w(S,R) measuring how strongly each
    child shapes the parent's identity.

DESIGN:
    1. Build weighted adjacency matrix for the 128x128 lattice
       Edge weight = exp(-|C_i - C_j| / C_mean)
    2. For each region, extract subgraph Laplacian L = D - W
    3. Compute spectral identity:
       - Eigenvalues/vectors of L (sparse eigsh for large regions, dense eigh for small)
       - Identity fingerprint = state coefficient vector [<state, v_i> for i in 0..k]
       - Fiedler value lambda_2 (coherence measure)
       - Spectral entropy
    4. Compute coupling weights via perturbation sensitivity:
       - Perturb child region by epsilon
       - Measure change in parent identity
       - Normalize across children

VERIFICATION:
    - L @ ones = 0 (kernel contains constant vector)
    - All eigenvalues >= 0
    - sum(weights) = 1.0 per parent

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import eigsh

RESULTS_DIR = Path(__file__).parent.parent / 'results'
K_MODES = 10  # number of spectral modes for identity fingerprint


def load_data():
    """Load exp_01 and exp_02 results."""
    P = np.load(RESULTS_DIR / 'exp_01_P_steady.npy')
    A = np.load(RESULTS_DIR / 'exp_01_A_steady.npy')
    stone_mask = np.load(RESULTS_DIR / 'exp_01_stone_mask.npy')

    # Load all level labels
    labels_by_level = []
    level = 0
    while True:
        path = RESULTS_DIR / f'exp_02_labels_level{level}.npy'
        if path.exists():
            labels_by_level.append(np.load(path))
            level += 1
        else:
            break

    # Load hierarchy from most recent exp_02 result
    import glob
    exp02_files = sorted(RESULTS_DIR.glob('exp_02_partition_*.json'))
    with open(exp02_files[-1]) as f:
        partition_data = json.load(f)

    hierarchy = {}
    for key, children in partition_data['hierarchy'].items():
        level_str, rid_str = key.split(',')
        hierarchy[(int(level_str), int(rid_str))] = [
            (int(c[0]), int(c[1])) for c in children
        ]

    return P, A, stone_mask, labels_by_level, hierarchy


def build_lattice_adjacency(C):
    """
    Build sparse weighted adjacency matrix for the periodic lattice.

    Edge weight: w(i,j) = exp(-|C_i - C_j| / C_mean)
    Strong within coherent regions, weak across boundaries.

    Returns: scipy.sparse.csr_matrix of shape (N*N, N*N)
    """
    N = C.shape[0]
    C_mean = C.mean()
    n_cells = N * N

    rows, cols, weights = [], [], []

    for i in range(N):
        for j in range(N):
            idx = i * N + j
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = (i + di) % N, (j + dj) % N
                nidx = ni * N + nj
                w = np.exp(-abs(C[i, j] - C[ni, nj]) / C_mean)
                rows.append(idx)
                cols.append(nidx)
                weights.append(w)

    adj = sparse.csr_matrix(
        (weights, (rows, cols)), shape=(n_cells, n_cells)
    )
    return adj


def graph_laplacian_subgraph(adjacency, indices):
    """
    Extract subgraph and compute its Laplacian L = D - W.

    Args:
        adjacency: full lattice adjacency (sparse)
        indices: 1D array of cell indices in this region

    Returns: L (sparse), W_sub (sparse)
    """
    W_sub = adjacency[np.ix_(indices, indices)]
    degrees = np.array(W_sub.sum(axis=1)).ravel()
    D_sub = sparse.diags(degrees)
    L_sub = D_sub - W_sub
    return L_sub, W_sub


def compute_spectral_identity(L, state_vector, k=K_MODES):
    """
    Compute the spectral identity fingerprint for a region.

    The confluent identity operator: CI(T) = projection onto harmonic space
    + spectral coefficient vector capturing internal structure.

    Returns dict with:
        harmonic_projection: weighted mean (ker(L) component)
        eigenvalues: first k+1 eigenvalues
        fiedler_value: lambda_2 (spectral gap = coherence)
        spectral_entropy: entropy of eigenvalue distribution
        state_coefficients: [<state, v_i> for i in 0..k]
        n_cells: region size
    """
    n = L.shape[0]
    k_actual = min(k + 1, n - 1)

    if k_actual < 2:
        # Tiny region: trivial identity
        return {
            'harmonic_projection': float(np.mean(state_vector)),
            'eigenvalues': [0.0],
            'fiedler_value': 0.0,
            'spectral_entropy': 0.0,
            'state_coefficients': [float(np.mean(state_vector))],
            'n_cells': n,
        }

    # Use dense for small regions, sparse for large
    if n < 50:
        L_dense = L.toarray() if sparse.issparse(L) else L
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
    else:
        try:
            eigenvalues, eigenvectors = eigsh(
                L.astype(float), k=k_actual, which='SM',
                tol=1e-8, maxiter=5000
            )
        except Exception:
            # Fallback to dense if sparse fails
            L_dense = L.toarray() if sparse.issparse(L) else L
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            eigenvalues = eigenvalues[:k_actual]
            eigenvectors = eigenvectors[:, :k_actual]

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Harmonic projection (onto kernel = constant vector for connected graph)
    harmonic_proj = float(np.mean(state_vector))

    # Fiedler value (first nonzero eigenvalue)
    nonzero_mask = eigenvalues > 1e-10
    fiedler_value = float(eigenvalues[nonzero_mask][0]) if nonzero_mask.any() else 0.0

    # State coefficients: project centered state onto each eigenvector
    state_centered = state_vector - harmonic_proj
    coefficients = []
    for i in range(min(k_actual, eigenvectors.shape[1])):
        coeff = float(np.dot(state_centered, eigenvectors[:, i]))
        coefficients.append(coeff)

    # Spectral entropy
    nonzero_eigs = eigenvalues[nonzero_mask]
    if len(nonzero_eigs) > 0:
        p = nonzero_eigs / nonzero_eigs.sum()
        spectral_entropy = float(-np.sum(p * np.log(p + 1e-15)))
    else:
        spectral_entropy = 0.0

    return {
        'harmonic_projection': harmonic_proj,
        'eigenvalues': eigenvalues.tolist(),
        'fiedler_value': fiedler_value,
        'spectral_entropy': spectral_entropy,
        'state_coefficients': coefficients,
        'n_cells': n,
    }


def compute_coupling_weights(adjacency, state_flat, parent_indices,
                             children_indices_list, epsilon=None):
    """
    Coupling weight w(S, R): sensitivity of parent identity to child perturbation.

    For each child S:
      1. Perturb state in S by epsilon
      2. Recompute parent identity
      3. w(S) = ||delta_coefficients|| / epsilon

    Normalize: sum(w) = 1.

    Args:
        adjacency: full lattice adjacency
        state_flat: 1D state array (C field flattened)
        parent_indices: indices of parent region cells
        children_indices_list: list of (child_id, child_indices) pairs
        epsilon: perturbation size (default: 0.01 * mean(state))
    """
    if epsilon is None:
        epsilon = 0.01 * np.mean(state_flat[parent_indices])

    # Baseline parent identity
    L_parent, _ = graph_laplacian_subgraph(adjacency, parent_indices)
    state_parent = state_flat[parent_indices]
    I_baseline = compute_spectral_identity(L_parent, state_parent)
    baseline_coeffs = np.array(I_baseline['state_coefficients'])

    # Index mapping: global index -> position in parent_indices
    parent_idx_set = set(parent_indices.tolist())
    parent_pos_map = {int(idx): pos for pos, idx in enumerate(parent_indices)}

    sensitivities = {}
    for child_id, child_indices in children_indices_list:
        # Perturb child cells within parent subgraph
        state_perturbed = state_parent.copy()
        for cidx in child_indices:
            cidx_int = int(cidx)
            if cidx_int in parent_pos_map:
                state_perturbed[parent_pos_map[cidx_int]] += epsilon

        I_perturbed = compute_spectral_identity(L_parent, state_perturbed)
        perturbed_coeffs = np.array(I_perturbed['state_coefficients'])

        # Align lengths
        min_len = min(len(baseline_coeffs), len(perturbed_coeffs))
        delta = np.linalg.norm(
            perturbed_coeffs[:min_len] - baseline_coeffs[:min_len]
        ) / epsilon
        sensitivities[child_id] = float(delta)

    # Normalize
    total = sum(sensitivities.values())
    if total > 1e-15:
        weights = {cid: s / total for cid, s in sensitivities.items()}
    else:
        n = len(sensitivities)
        weights = {cid: 1.0 / n for cid in sensitivities} if n > 0 else {}

    return weights


def run_experiment():
    """Run Hodge identity and coupling weight computation."""

    print("=" * 70)
    print("Confluent Identity -- Phase 1, Experiment 03")
    print("Spectral Identity and Coupling Weights")
    print("=" * 70)

    # Load data
    P, A, stone_mask, labels_by_level, hierarchy = load_data()
    C = P + A
    N = C.shape[0]
    state_flat = C.ravel()
    n_levels = len(labels_by_level)
    print(f"\nLoaded: {N}x{N} field, {n_levels} hierarchy levels")

    # Build adjacency
    print(f"\nBuilding weighted adjacency matrix...")
    adjacency = build_lattice_adjacency(C)
    print(f"  Shape: {adjacency.shape}, nnz: {adjacency.nnz}")

    # Compute identities for all regions at all levels
    all_identities = {}  # (level, region_id) -> identity dict

    for level in range(n_levels):
        labels = labels_by_level[level]
        region_ids = sorted(np.unique(labels).tolist())
        print(f"\nLevel {level}: {len(region_ids)} regions")

        for rid in region_ids:
            mask = labels == rid
            indices = np.where(mask.ravel())[0]

            if len(indices) < 3:
                print(f"  Region {rid}: too small ({len(indices)} cells), skipping")
                continue

            L, W = graph_laplacian_subgraph(adjacency, indices)
            state_region = state_flat[indices]
            identity = compute_spectral_identity(L, state_region)

            # Verification: L @ ones should be ~0
            ones = np.ones(len(indices))
            L_ones_norm = np.linalg.norm(L.dot(ones))
            identity['laplacian_kernel_check'] = float(L_ones_norm)

            all_identities[(level, rid)] = identity
            print(f"  Region {rid}: {identity['n_cells']} cells, "
                  f"fiedler={identity['fiedler_value']:.6f}, "
                  f"spec_entropy={identity['spectral_entropy']:.4f}, "
                  f"L@1={L_ones_norm:.2e}")

    # Compute coupling weights for each parent-child relationship
    all_weights = {}  # (level, parent_id) -> {child_id: weight}

    print(f"\n{'=' * 70}")
    print("Coupling Weights")
    print(f"{'=' * 70}")

    for (level, pid), children in hierarchy.items():
        if len(children) < 2:
            continue

        parent_labels = labels_by_level[level]
        parent_mask = parent_labels == pid
        parent_indices = np.where(parent_mask.ravel())[0]

        children_indices_list = []
        for child_level, child_id in children:
            child_labels = labels_by_level[child_level]
            child_mask = child_labels == child_id
            child_indices = np.where(child_mask.ravel())[0]
            children_indices_list.append((child_id, child_indices))

        weights = compute_coupling_weights(
            adjacency, state_flat, parent_indices, children_indices_list
        )
        all_weights[(level, pid)] = weights

        # Display
        weight_str = ", ".join(
            f"child {cid}: {w:.3f}" for cid, w in sorted(weights.items())
        )
        print(f"  Level {level}, Parent {pid} ({len(children)} children): {weight_str}")

        # Weight entropy
        w_vals = list(weights.values())
        if len(w_vals) > 1:
            w_arr = np.array(w_vals)
            w_entropy = float(-np.sum(w_arr * np.log(w_arr + 1e-15)))
            max_entropy = np.log(len(w_vals))
            print(f"    Weight entropy: {w_entropy:.4f} / {max_entropy:.4f} "
                  f"(ratio: {w_entropy/max_entropy:.3f})")

    # Verification summary
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Check Laplacian kernel
    max_kernel_error = max(
        (id_dict.get('laplacian_kernel_check', 0)
         for id_dict in all_identities.values()),
        default=0
    )
    print(f"  Max |L @ 1|: {max_kernel_error:.2e} "
          f"({'[PASS]' if max_kernel_error < 1e-8 else '[FAIL]'})")

    # Check eigenvalues non-negative
    all_nonneg = all(
        all(ev >= -1e-10 for ev in id_dict.get('eigenvalues', []))
        for id_dict in all_identities.values()
    )
    print(f"  All eigenvalues >= 0: {'[PASS]' if all_nonneg else '[FAIL]'}")

    # Check weight normalization
    weight_ok = all(
        abs(sum(w.values()) - 1.0) < 1e-10
        for w in all_weights.values()
    )
    print(f"  Weight normalization (sum=1): {'[PASS]' if weight_ok else '[FAIL]'}")

    # Save results
    # Convert tuple keys to strings for JSON
    identities_json = {
        f"{level},{rid}": identity
        for (level, rid), identity in all_identities.items()
    }
    weights_json = {
        f"{level},{pid}": {str(cid): w for cid, w in weights.items()}
        for (level, pid), weights in all_weights.items()
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'experiment': 'exp_03_hodge_identity',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'k_modes': K_MODES,
            'adjacency_nnz': adjacency.nnz,
        },
        'n_identities': len(all_identities),
        'n_weight_sets': len(all_weights),
        'identities': identities_json,
        'coupling_weights': weights_json,
        'verification': {
            'max_kernel_error': float(max_kernel_error),
            'all_eigenvalues_nonneg': bool(all_nonneg),
            'weights_normalized': bool(weight_ok),
        },
    }

    output_file = RESULTS_DIR / f'exp_03_identity_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_file.name}")

    return results


if __name__ == '__main__':
    run_experiment()
