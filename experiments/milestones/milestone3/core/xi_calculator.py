"""
Milestone 3: Shared ξ (structure information) calculator.

Computes structure information ξ from the mutual information of multi-mode
energy dispersals, following the Landauer erasure model.

This extracts the duplicated ξ computation from cascade_deep_dive.py,
landauer_generative.py, eis_resonance.py, and cascade_void_prime.py into
a single reusable module.
"""

import numpy as np
from .constants import LANDAUER_MIN, KT_DEFAULT


def compute_xi(mode_energies):
    """
    Compute structure information ξ from mode energy samples.

    ξ = (1/2) * [Σ ln(diag(C)) - Σ ln(eig(C))]

    This is the total mutual information of the covariance matrix,
    measuring how much structure (inter-mode correlation) exists
    in the energy distribution.

    Parameters
    ----------
    mode_energies : np.ndarray, shape (n_samples, n_modes)
        Energy samples across modes.

    Returns
    -------
    float
        ξ value (non-negative). Returns 0.0 if computation fails
        (e.g., singular covariance, insufficient samples).
    """
    if mode_energies.shape[0] < 3 or mode_energies.shape[1] < 2:
        return 0.0

    try:
        cov = np.cov(mode_energies.T)
        diag = np.diag(cov)

        # Guard against zero or negative diagonal entries
        if np.any(diag <= 0):
            return 0.0

        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]  # Drop non-positive

        if len(eigenvalues) == 0:
            return 0.0

        xi = 0.5 * (np.sum(np.log(diag)) - np.sum(np.log(eigenvalues)))
        return max(0.0, xi)  # ξ is non-negative by definition
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def coupling_weights(n_modes, decay_rate=0.5):
    """
    Compute normalized exponential coupling weights for n modes.

    c_i = exp(-decay_rate * i), normalized to sum to 1.

    Parameters
    ----------
    n_modes : int
        Number of energy modes.
    decay_rate : float
        Exponential decay rate (default: 0.5).

    Returns
    -------
    np.ndarray
        Normalized coupling weights, shape (n_modes,).
    """
    raw = np.exp(-decay_rate * np.arange(n_modes))
    return raw / raw.sum()


def single_landauer_event(T, n_modes=8, n_samples=100000, rng=None):
    """
    Model a single Landauer erasure event.

    Distributes kT·ln(2) of erasure energy across n_modes with exponential
    coupling, computes ξ from the resulting correlations.

    Parameters
    ----------
    T : float
        Temperature (in natural units where k=1).
    n_modes : int
        Number of environmental modes.
    n_samples : int
        Monte Carlo samples.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    dict
        Keys: 'xi' (float), 'theta' (float), 'mode_energies' (ndarray),
        'coupling' (ndarray), 'total_energy' (float).
    """
    if rng is None:
        rng = np.random.default_rng()

    erasure_cost = T * np.log(2)
    coupling = coupling_weights(n_modes)

    # Each mode receives energy proportional to coupling + thermal noise
    mode_means = erasure_cost * coupling
    mode_energies = np.column_stack([
        rng.exponential(scale=max(m, 1e-10), size=n_samples)
        for m in mode_means
    ])

    xi = compute_xi(mode_energies)
    total = mode_energies.mean(axis=0).sum()
    theta = max(0.0, erasure_cost - xi)  # Thermal remainder

    return {
        'xi': xi,
        'theta': theta,
        'mode_energies': mode_energies,
        'coupling': coupling,
        'total_energy': total,
    }


def participation_ratio(eigenvalues):
    """
    Compute the participation ratio of eigenvalues.

    PR = (Σ λ_i)² / Σ λ_i²

    Measures how many modes effectively participate.
    PR = 1 means one mode dominates, PR = n means all modes equal.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Non-negative eigenvalues.

    Returns
    -------
    float
        Participation ratio.
    """
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0.0
    return (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
