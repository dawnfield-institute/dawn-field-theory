"""
phase_rate.py -- Shared infrastructure for Midnight: Observational Frontiers of DFT.

The phase-rate primitive w(x) = (mc^2/hbar) * (dtau/dt) is the rate at which an
internal quantum clock advances at a given embedding in the connection structure.

Relativity = how w is shifted (mean sector, reversible, no environment).
Decoherence = how w is spread (variance sector, irreversible, requires environment).
Entanglement = harmonic phase-locking of w between mesh-adjacent nodes.

Prediction: gravity gates entanglement at commensurate phase-rate ratios (phi^n).
Standard QM predicts flat formation probability; DFT predicts structured peaks.

Provides:
- phase_rate_at_depth, phase_rate_ratio: phase-rate field definition
- commensurate_ratios, is_commensurate: phi-harmonic ratio structure
- phase_coupling_kernel: harmonic locking kernel K(r)
- entanglement_vs_ratio: entanglement entropy as f(phase-rate ratio)
- save_midnight_results, _convert_numpy: output utilities
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent

# Milestones live under experiments/milestones/, this sidecar under experiments/sidecars/,
# so the hop is up TWO levels and across -- not `MIDNIGHT_ROOT.parent`, which was correct
# only before the August 2026 layer reorganization (see MIGRATION.md) moved sidecars out of
# experiments/. Every Midnight script imports through this module, so the stale path made
# the whole sidecar unrunnable rather than failing in one visible place.
MILESTONES_ROOT = MIDNIGHT_ROOT.parent.parent / "milestones"
SIDECARS_ROOT = MIDNIGHT_ROOT.parent
M14_ROOT = MILESTONES_ROOT / "milestone14"
sys.path.insert(0, str(M14_ROOT / "core"))


def find_data_root():
    """The shared `data/` directory, resolved by SEARCH rather than by counting parents.

    Scripts here used `MIDNIGHT_ROOT.parent.parent.parent.parent / "data"`, which assumes the
    repository sits directly inside the workspace. That holds for a primary checkout and is
    wrong in every git worktree, because a worktree adds a directory level -- and workspace
    convention routes all repo work through worktrees, so the fragile case is the normal one.

    Walks up looking for a `data/` directory, preferring one that actually carries the
    catalogues Midnight reads. `DFT_DATA_ROOT` overrides.
    """
    override = os.environ.get("DFT_DATA_ROOT")
    if override:
        return Path(override)
    fallback = None
    here = MIDNIGHT_ROOT
    for _ in range(8):
        here = here.parent
        cand = here / "data"
        if cand.is_dir():
            if (cand / "sdss_mgii").is_dir():
                return cand
            fallback = fallback or cand
    return fallback or (MIDNIGHT_ROOT.parent.parent.parent.parent / "data")


DATA_ROOT = find_data_root()

from quantum_complement import (
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE, PI,
    HBAR, C_LIGHT,
    DynkinDiagram,
    orbit_hilbert_basis, graph_automorphisms, vertex_orbits,
    cartesian_product_graph, partial_trace, von_neumann_entropy, purity,
    _convert_numpy,
)

# ============================================================
# Constants
# ============================================================

M_ELECTRON_KG = 9.1093837015e-31
COMPTON_FREQ = M_ELECTRON_KG * C_LIGHT**2 / HBAR


# ============================================================
# Phase-Rate Field
# ============================================================

def phase_rate_at_depth(depth, epsilon=0.01):
    """
    Phase-rate w(d) at PAC depth d.

    w(d) = COMPTON_FREQ * sqrt(1 - 2 * epsilon * phi^(-d))

    epsilon is a dimensionless potential scale. Small epsilon = weak field.
    The gravitational potential at depth d goes as phi^(-d) from the PAC tree.
    """
    potential = epsilon * PHI**(-depth)
    if 2 * potential >= 1.0:
        return 0.0
    return COMPTON_FREQ * np.sqrt(1 - 2 * potential)


def phase_rate_ratio(d1, d2, epsilon=0.01):
    """Ratio w(d1) / w(d2) of phase-rates at two depths."""
    w1 = phase_rate_at_depth(d1, epsilon)
    w2 = phase_rate_at_depth(d2, epsilon)
    if w2 == 0:
        return float('inf')
    return w1 / w2


def commensurate_ratios(max_n=8):
    """
    Commensurate phase-rate ratios predicted by DFT.

    From g_out = g_in^2 and PAC tree structure, the natural ratios are
    phi^n for integer n. These are the ratios at which entanglement
    formation should peak.

    Returns list of (n, phi^n) tuples for n = -max_n to max_n (nonzero).
    """
    ratios = []
    for n in range(-max_n, max_n + 1):
        if n == 0:
            continue
        ratios.append((n, PHI**n))
    return ratios


def is_commensurate(ratio, tolerance=0.05, max_n=8):
    """
    Check if a phase-rate ratio is near a commensurate value phi^n.

    Returns (is_near, nearest_n, deviation).
    """
    if ratio <= 0:
        return False, 0, float('inf')
    log_r = np.log(ratio)
    best_n = round(log_r / LN_PHI)
    best_n = max(-max_n, min(max_n, best_n))
    if best_n == 0:
        best_n = 1 if log_r > 0 else -1
    deviation = abs(log_r - best_n * LN_PHI) / abs(best_n * LN_PHI)
    return deviation < tolerance, int(best_n), float(deviation)


# ============================================================
# Phase-Coupling Kernel
# ============================================================

def phase_coupling_kernel(ratio, sigma_log=0.15, max_n=6):
    """
    Phase-rate coupling kernel K(r).

    DFT prediction: peaks at commensurate ratios r = phi^n.
    Working in log-space where phi^n are equally spaced at n*ln(phi).

    K(r) = sum_{n=-max_n}^{max_n} exp(-(ln(r) - n*ln(phi))^2 / (2*sigma_log^2))

    Standard QM prediction: K(r) = 1 (flat).
    """
    ratio = np.asarray(ratio, dtype=float)
    log_r = np.log(np.clip(ratio, 1e-15, None))

    kernel = np.zeros_like(log_r)
    for n in range(-max_n, max_n + 1):
        center = n * LN_PHI
        kernel += np.exp(-(log_r - center)**2 / (2 * sigma_log**2))

    return kernel


# ============================================================
# Entanglement Formation Model
# ============================================================

def graph_laplacian_eigenvalues(adj):
    """Normalized Laplacian eigenvalues of a graph."""
    degree = np.sum(adj, axis=1)
    D = np.diag(degree)
    L = D - adj
    eigs = np.linalg.eigvalsh(L)
    eigs = np.sort(eigs)
    max_eig = eigs[-1]
    if max_eig > 0:
        eigs = eigs / max_eig
    return eigs


def entanglement_vs_ratio(adj, ratio_array, sigma_log=0.15, max_n=6):
    """
    Entanglement entropy of a phase-modulated product-graph state
    as a function of phase-rate ratio between two subsystems.

    Mechanism: for two copies of the same ADE graph at different
    gravitational embeddings (ratio r of phase-rates):
    1. Compute graph Laplacian eigenvalues (spectral clock rates)
    2. For each ratio r, construct amplitudes c_{ij}(r) = K(lambda_i/lambda_j * r)
    3. Normalize to get |psi(r)>, compute rho, partial trace, entropy

    At commensurate ratios, the kernel concentrates amplitude on
    specific orbit pairs -> higher entanglement.
    At incommensurate ratios, amplitude spreads uniformly -> lower.
    """
    n = adj.shape[0]

    eigs = graph_laplacian_eigenvalues(adj)
    eigs_pos = eigs[eigs > 1e-10]
    if len(eigs_pos) < 2:
        eigs_pos = np.array([INV_PHI, 1.0])

    m = len(eigs_pos)
    ratio_array = np.asarray(ratio_array, dtype=float)
    entropy_array = np.zeros(len(ratio_array))

    for idx, r in enumerate(ratio_array):
        amplitudes = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                eig_ratio = eigs_pos[i] / eigs_pos[j] * r
                amplitudes[i, j] = phase_coupling_kernel(eig_ratio, sigma_log, max_n)

        norm = np.sqrt(np.sum(amplitudes**2))
        if norm < 1e-15:
            entropy_array[idx] = 0.0
            continue
        amplitudes /= norm

        psi = amplitudes.flatten()
        rho = np.outer(psi, psi)
        rho_A = partial_trace(rho, m, m, trace_out='second')
        entropy_array[idx] = von_neumann_entropy(rho_A)

    return {
        'ratio_array': ratio_array,
        'entropy_array': entropy_array,
        'n_spectral': m,
        'eigenvalues': eigs_pos.tolist(),
    }


# ============================================================
# PAC Tree Construction
# ============================================================

def build_weighted_pac_tree(depth, split_ratio=None):
    """
    Build a binary PAC tree with conservation-weighted edges.

    At each node at level k, the potential V(k) = split_ratio^k is split
    between two children: child1 gets fraction split_ratio, child2 gets
    fraction (1 - split_ratio). For the PAC tree, split_ratio = 1/phi
    and 1 - 1/phi = 1/phi^2 = split_ratio^2.

    Returns:
        potentials: array of node potentials (length 2^(depth+1) - 1)
        adjacency: adjacency matrix
        levels: array of depth level per node
    """
    if split_ratio is None:
        split_ratio = INV_PHI

    n = 2**(depth + 1) - 1
    potentials = np.zeros(n)
    levels = np.zeros(n, dtype=int)
    adjacency = np.zeros((n, n))

    potentials[0] = 1.0
    levels[0] = 0

    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n:
            adjacency[i, left] = adjacency[left, i] = 1
            potentials[left] = potentials[i] * split_ratio
            levels[left] = levels[i] + 1
        if right < n:
            adjacency[i, right] = adjacency[right, i] = 1
            potentials[right] = potentials[i] * (1 - split_ratio)
            levels[right] = levels[i] + 1

    return potentials, adjacency, levels


def _lowest_common_ancestor(i, j):
    """LCA in a binary tree with 0-indexed heap layout."""
    while i != j:
        if i > j:
            i = (i - 1) // 2
        else:
            j = (j - 1) // 2
    return i


def stochastic_pac_tree(depth, split_ratio=None, noise_scale=0.05, rng=None):
    """
    Generate a stochastic PAC tree with noisy branching.

    At each node, conservation holds exactly (V_parent = V_left + V_right)
    but the split fraction is perturbed: child1 gets (split_ratio + noise).
    """
    if split_ratio is None:
        split_ratio = INV_PHI
    if rng is None:
        rng = np.random.RandomState()

    n = 2**(depth + 1) - 1
    potentials = np.zeros(n)
    levels = np.zeros(n, dtype=int)
    potentials[0] = 1.0

    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and right < n:
            s = split_ratio + rng.normal(0, noise_scale)
            s = np.clip(s, 0.01, 0.99)
            potentials[left] = potentials[i] * s
            potentials[right] = potentials[i] * (1 - s)
            levels[left] = levels[i] + 1
            levels[right] = levels[i] + 1

    return potentials, levels


def pac_tree_correlation_profile(depth, split_ratio=None, noise_scale=0.05,
                                  n_trials=2000, seed=42):
    """
    Measure correlation between node potentials at different depth separations
    via Monte Carlo over stochastic PAC trees.

    For each trial, generate a noisy PAC tree. Collect potentials at each
    depth level. Compute Pearson correlation between depth-level means
    across trials. The correlation decay with depth separation reveals
    the tree's conservation structure.
    """
    if split_ratio is None:
        split_ratio = INV_PHI

    rng = np.random.RandomState(seed)

    level_means = {d: [] for d in range(depth + 1)}

    for _ in range(n_trials):
        potentials, levels = stochastic_pac_tree(depth, split_ratio, noise_scale, rng)
        for d in range(depth + 1):
            mask = levels == d
            if np.any(mask):
                level_means[d].append(float(np.mean(potentials[mask])))

    for d in level_means:
        level_means[d] = np.array(level_means[d])

    profile = {}
    for delta in range(0, depth + 1):
        correlations = []
        for d1 in range(depth + 1):
            d2 = d1 + delta
            if d2 > depth:
                continue
            if len(level_means[d1]) > 1 and len(level_means[d2]) > 1:
                r = np.corrcoef(level_means[d1], level_means[d2])[0, 1]
                if not np.isnan(r):
                    correlations.append(abs(r))

        if correlations:
            profile[delta] = {
                'mean_correlation': float(np.mean(correlations)),
                'std_correlation': float(np.std(correlations)),
                'n_pairs': len(correlations),
            }

    return profile


# ============================================================
# Model Comparison
# ============================================================

def chi_squared(data, model):
    """Chi-squared statistic (unit weights)."""
    residuals = data - model
    return float(np.sum(residuals**2))


def fit_flat_model(entropy_array):
    """Fit constant model S(r) = C. Returns (C, chi2)."""
    C = np.mean(entropy_array)
    model = np.full_like(entropy_array, C)
    chi2 = chi_squared(entropy_array, model)
    return float(C), chi2


def fit_phi_harmonic_model(ratio_array, entropy_array, max_n=6):
    """
    Fit phi-harmonic model: S(r) = B + A * K(r, sigma_log).

    Peak positions fixed at phi^n. Free params: A, B, sigma_log.
    Grid search over sigma_log, analytic solution for A, B at each sigma.
    """
    best_chi2 = float('inf')
    best_params = None

    for sigma_log in np.linspace(0.05, 0.5, 50):
        K = phase_coupling_kernel(ratio_array, sigma_log, max_n)
        K_mean = np.mean(K)
        S_mean = np.mean(entropy_array)
        K_centered = K - K_mean
        S_centered = entropy_array - S_mean

        denom = np.sum(K_centered**2)
        if denom < 1e-15:
            continue
        A = np.sum(S_centered * K_centered) / denom
        B = S_mean - A * K_mean

        model = B + A * K
        chi2 = chi_squared(entropy_array, model)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_params = {'A': float(A), 'B': float(B), 'sigma_log': float(sigma_log)}

    if best_params is None:
        return {'A': 0, 'B': float(np.mean(entropy_array)), 'sigma_log': 0.15}, float('inf')

    return best_params, best_chi2


def bic(chi2, n_params, n_data):
    """Bayesian Information Criterion: BIC = chi2 + k*ln(N)."""
    return chi2 + n_params * np.log(n_data)


# ============================================================
# Results Saving
# ============================================================

RESULTS_DIR = MIDNIGHT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_midnight_results(experiment_name, data):
    """Save experiment results as timestamped JSON to midnight/results/."""
    data = _convert_numpy(data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  Results saved: {filepath}")
    return str(filepath)
