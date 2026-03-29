"""
scope.py -- Transfer matrix infrastructure for Milestone 6: Scoped Mediation.

Built by exp_01, imported by exp_02-10. Provides:
- build_transfer_matrix: construct T mapping parent spectral to child spectral
- decompose_harmonic_transient: split T = T_harm + T_trans
- harmonic_fixed_point: iterate T_harm to rank-1 projector
- scope_attenuation: compute ||T_harm^n|| for n hops
- pac_budget: compute P, A, xi, Theta from spectral decomposition

All functions operate on the confluent identity hierarchy built by
exp_01/exp_02 of the confluent_identity series.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


# ============================================================
# Constants
# ============================================================
PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI
LN_PHI = np.log(PHI)
GAMMA_EM = 0.5772156649015329
XI_BALANCE = GAMMA_EM + LN_PHI  # 1.0584


# ============================================================
# Transfer matrix construction
# ============================================================

def _get_eigenbasis(L, state_vector, k=10):
    """
    Compute k eigenvectors of graph Laplacian L, sorted by eigenvalue.
    Returns (eigenvalues, eigenvectors) with zero modes included.
    """
    n = L.shape[0]
    k_actual = min(k + 1, n - 1)

    if k_actual < 2:
        return np.array([0.0]), np.ones((n, 1)) / np.sqrt(n)

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
    return eigenvalues[idx], eigenvectors[:, idx]


def build_transfer_matrix(parent_eigvecs, child_indices_in_parent, k=10):
    """
    Construct the transfer matrix T that maps a parent's spectral identity
    to a child's contribution in that basis.

    T[i,j] = sum_{cell in child} v_i[cell] * v_j[cell]

    where v_i are the parent's eigenvectors. This is the child's "spectral
    footprint" in the parent's eigenbasis -- a k x k matrix whose (i,j) entry
    measures how much the child's cells correlate mode i with mode j.

    Parameters:
        parent_eigvecs: (n_parent, k) eigenvectors of parent's Laplacian
        child_indices_in_parent: local indices of child cells within parent
        k: number of modes to use

    Returns:
        T: (k, k) transfer matrix
    """
    k_actual = min(k, parent_eigvecs.shape[1])
    child_vecs = parent_eigvecs[child_indices_in_parent, :k_actual]
    T = child_vecs.T @ child_vecs
    # Normalize by child size so T measures density not total
    T /= max(len(child_indices_in_parent), 1)
    return T


def decompose_harmonic_transient(T, n_harmonic=1):
    """
    Decompose transfer matrix T = T_harm + T_trans.

    T_harm captures the harmonic (zero-mode) component -- the part that
    survives arbitrarily many scope boundaries. T_trans captures the
    transient modes that decay with each hop.

    The harmonic subspace corresponds to the first n_harmonic eigenmodes
    (typically just the zero mode, n_harmonic=1).

    Parameters:
        T: (k, k) transfer matrix
        n_harmonic: number of modes in the harmonic subspace

    Returns:
        T_harm: (k, k) harmonic component
        T_trans: (k, k) transient component
        eigenvalues: eigenvalues of T (sorted descending by magnitude)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    # Sort by magnitude (descending)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Harmonic projection: first n_harmonic modes
    V_harm = eigenvectors[:, :n_harmonic]
    T_harm = V_harm @ np.diag(eigenvalues[:n_harmonic]) @ V_harm.T

    T_trans = T - T_harm
    return T_harm, T_trans, eigenvalues


def harmonic_fixed_point(T_harm, n_iter=20, tol=1e-10):
    """
    Iterate T_harm^n and test convergence to rank-1 projector.

    Returns:
        converged: bool -- did T_harm^n stabilize?
        rank1_error: float -- ||T^n - T^(n-1)|| at final iteration
        powers: list of (n, T^n, frobenius_norm) at each step
    """
    T_n = T_harm.copy()
    powers = [(1, T_n.copy(), np.linalg.norm(T_n, 'fro'))]

    for n in range(2, n_iter + 1):
        T_prev = T_n.copy()
        T_n = T_n @ T_harm
        norm = np.linalg.norm(T_n, 'fro')
        diff = np.linalg.norm(T_n - T_prev, 'fro')
        powers.append((n, T_n.copy(), norm))

        if diff < tol:
            return True, diff, powers

    final_diff = np.linalg.norm(powers[-1][1] - powers[-2][1], 'fro')
    return final_diff < tol, final_diff, powers


def scope_attenuation(T_harm, n_hops):
    """
    Compute the Frobenius norm of T_harm^n for n = 1..n_hops.

    Returns:
        norms: list of float, ||T_harm^n||_F for n=1..n_hops
        ratios: list of float, ||T^(n+1)||/||T^n|| for n=1..n_hops-1
    """
    T_n = T_harm.copy()
    norms = [np.linalg.norm(T_n, 'fro')]

    for _ in range(n_hops - 1):
        T_n = T_n @ T_harm
        norms.append(np.linalg.norm(T_n, 'fro'))

    ratios = []
    for i in range(len(norms) - 1):
        if norms[i] > 1e-15:
            ratios.append(norms[i + 1] / norms[i])
        else:
            ratios.append(np.nan)

    return norms, ratios


def pac_budget(state_vector, L, eigenvectors, eigenvalues):
    """
    Compute the PAC information budget at a scope boundary.

    P (Potential)  = total spectral energy entering
    A (Actualized) = harmonic component (zero-mode projection)
    xi (Structure) = energy in the first few non-zero modes (organized)
    Theta (Thermal) = energy in remaining modes (dissipated)

    Parameters:
        state_vector: field values at cells in this region
        L: graph Laplacian
        eigenvectors: eigenvectors of L
        eigenvalues: eigenvalues of L

    Returns:
        dict with P, A, xi, Theta, conservation_error
    """
    state_centered = state_vector - np.mean(state_vector)
    coefficients = eigenvectors.T @ state_centered
    energies = coefficients ** 2

    # Total energy
    P = float(np.sum(energies))

    # Harmonic = zero-mode energy
    zero_mask = eigenvalues < 1e-10
    A = float(np.sum(energies[zero_mask]))

    # Structure = energy in first few non-zero modes (modes 1-3)
    nonzero_eigs = np.where(~zero_mask)[0]
    structure_modes = nonzero_eigs[:3] if len(nonzero_eigs) >= 3 else nonzero_eigs
    xi = float(np.sum(energies[structure_modes]))

    # Thermal = everything else
    all_budget = set(range(len(eigenvalues)))
    used = set(np.where(zero_mask)[0]) | set(structure_modes)
    thermal_modes = list(all_budget - used)
    Theta = float(np.sum(energies[thermal_modes]))

    conservation_error = abs(P - (A + xi + Theta))

    return {
        'P': P,
        'A': A,
        'xi': xi,
        'Theta': Theta,
        'conservation_error': conservation_error,
        'A_fraction': A / P if P > 0 else 0,
        'xi_fraction': xi / P if P > 0 else 0,
        'Theta_fraction': Theta / P if P > 0 else 0,
    }


def matrix_rank_at_tolerance(M, tol=1e-6):
    """Effective rank of matrix M at given tolerance."""
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * s[0]))
