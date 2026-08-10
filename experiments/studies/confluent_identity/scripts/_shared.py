"""
_shared.py -- Shared utilities for Confluent Identity experiments.

Extracted from exp_03_hodge_identity.py. New experiments (exp_08+) import
from here rather than duplicating code. Original scripts (exp_01-07) are
left untouched.
"""

import numpy as np
import json
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import eigsh

RESULTS_DIR = Path(__file__).parent.parent / 'results'
K_MODES = 10


def load_baseline():
    """Load exp_01 steady-state fields + exp_02 hierarchy."""
    P = np.load(RESULTS_DIR / 'exp_01_P_steady.npy')
    A = np.load(RESULTS_DIR / 'exp_01_A_steady.npy')
    stone_mask = np.load(RESULTS_DIR / 'exp_01_stone_mask.npy')
    C = P + A

    # Level labels
    labels_by_level = []
    level = 0
    while True:
        path = RESULTS_DIR / f'exp_02_labels_level{level}.npy'
        if path.exists():
            labels_by_level.append(np.load(path))
            level += 1
        else:
            break

    # Hierarchy from most recent exp_02 result
    exp02_files = sorted(RESULTS_DIR.glob('exp_02_partition_*.json'))
    with open(exp02_files[-1]) as f:
        partition_data = json.load(f)

    hierarchy = {}
    for key, children in partition_data['hierarchy'].items():
        level_str, rid_str = key.split(',')
        hierarchy[(int(level_str), int(rid_str))] = [
            (int(c[0]), int(c[1])) for c in children
        ]

    return P, A, C, stone_mask, labels_by_level, hierarchy


def build_lattice_adjacency(C):
    """
    Sparse weighted adjacency for periodic lattice.
    Edge weight: w(i,j) = exp(-|C_i - C_j| / C_mean).
    """
    N = C.shape[0]
    C_mean = C.mean()
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

    return sparse.csr_matrix(
        (weights, (rows, cols)), shape=(N * N, N * N)
    )


def graph_laplacian_subgraph(adjacency, indices):
    """Extract subgraph Laplacian L = D - W."""
    W_sub = adjacency[np.ix_(indices, indices)]
    degrees = np.array(W_sub.sum(axis=1)).ravel()
    D_sub = sparse.diags(degrees)
    return D_sub - W_sub, W_sub


def compute_spectral_identity(L, state_vector, k=K_MODES):
    """Compute spectral identity fingerprint for a region."""
    n = L.shape[0]
    k_actual = min(k + 1, n - 1)

    if k_actual < 2:
        return {
            'harmonic_projection': float(np.mean(state_vector)),
            'eigenvalues': [0.0],
            'fiedler_value': 0.0,
            'spectral_entropy': 0.0,
            'state_coefficients': [float(np.mean(state_vector))],
            'n_cells': n,
        }

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
            L_dense = L.toarray() if sparse.issparse(L) else L
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            eigenvalues = eigenvalues[:k_actual]
            eigenvectors = eigenvectors[:, :k_actual]

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    harmonic_proj = float(np.mean(state_vector))
    nonzero_mask = eigenvalues > 1e-10
    fiedler_value = float(eigenvalues[nonzero_mask][0]) if nonzero_mask.any() else 0.0

    state_centered = state_vector - harmonic_proj
    coefficients = [
        float(np.dot(state_centered, eigenvectors[:, i]))
        for i in range(min(k_actual, eigenvectors.shape[1]))
    ]

    nonzero_eigs = eigenvalues[nonzero_mask]
    if len(nonzero_eigs) > 0:
        p = nonzero_eigs / nonzero_eigs.sum()
        spectral_entropy = float(-np.sum(p * np.log(p + 1e-15)))
    else:
        spectral_entropy = 0.0

    return {
        'harmonic_projection': harmonic_proj,
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors,
        'fiedler_value': fiedler_value,
        'spectral_entropy': spectral_entropy,
        'state_coefficients': coefficients,
        'n_cells': n,
    }


def get_region_indices(labels_by_level, level, rid):
    """Get flat indices for a region."""
    labels = labels_by_level[level]
    return np.where((labels == rid).ravel())[0]


def get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):
    """
    Yield (parent_key, parent_indices, children_list, L_parent, state_parent)
    for each parent with >= 2 children.

    children_list: [(child_id, child_indices), ...]
    """
    for (level, pid), children in hierarchy.items():
        if len(children) < 2:
            continue

        parent_indices = get_region_indices(labels_by_level, level, pid)
        if len(parent_indices) < 3:
            continue

        children_list = []
        for child_level, child_id in children:
            child_indices = get_region_indices(labels_by_level, child_level, child_id)
            if len(child_indices) > 0:
                children_list.append((child_id, child_indices))

        if len(children_list) < 2:
            continue

        L_parent, _ = graph_laplacian_subgraph(adjacency, parent_indices)
        state_parent = state_flat[parent_indices]

        yield (level, pid), parent_indices, children_list, L_parent, state_parent


def compute_subgraph_laplacian_from_field(C_flat, indices, N):
    """
    Build subgraph Laplacian directly from C field for given cell indices.
    Avoids building the full NxN adjacency matrix — O(|indices|) instead of O(N^2).

    C_flat: 1D array of length N*N
    indices: array of flat indices into C_flat
    N: grid size

    Returns: L (sparse CSR), W (sparse CSR), both shape (len(indices), len(indices))
    """
    C_mean = np.mean(C_flat)
    index_set = set(int(i) for i in indices)
    local_map = {int(g): pos for pos, g in enumerate(indices)}
    n = len(indices)

    rows, cols, weights = [], [], []
    for g in indices:
        g = int(g)
        i, j = divmod(g, N)
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = (i + di) % N, (j + dj) % N
            neighbor = ni * N + nj
            if neighbor in index_set:
                w = np.exp(-abs(C_flat[g] - C_flat[neighbor]) / C_mean)
                rows.append(local_map[g])
                cols.append(local_map[neighbor])
                weights.append(w)

    W = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
    degrees = np.array(W.sum(axis=1)).ravel()
    D = sparse.diags(degrees)
    return D - W, W
