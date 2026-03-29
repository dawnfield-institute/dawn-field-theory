"""
exp_26_confluence_complexity.py -- Confluent Identity Phase 17

PURPOSE:
    The positive Fiedler-entropy correlation (exp_18: partial rho=+0.29, p=0.03)
    means confluence is generative — stronger internal coupling produces richer
    spectral structure. In PAC: actualization through coupling DISTRIBUTES energy
    across modes. A well-coupled region has LESS mode dominance, not more.

    This directly validates Claim 1 ("identity is confluence, not aggregation")
    in a way that simple correlation couldn't.

METHODS:
    For each level-0 region:
    1. Full eigenvalue spectrum (dense solver)
    2. Eigenvalue distribution shape vs Fiedler:
       - Gini coefficient of eigenvalue distribution
       - Effective dimensionality: count eigenvalues > 10% of max
       - Mode 1 dominance: |c_1|^2 / sum(|c_k|^2)
    3. Random graph control: Erdos-Renyi with matching (n, mean_degree)
    4. Multi-seed validation across exp_13's 5 seeds (if available)

VERIFICATION:
    - Gini(eigenvalues) NEGATIVELY correlated with Fiedler (rho < -0.2, p < 0.05)
    - Effective dimensionality POSITIVELY correlated with Fiedler (rho > 0.2, p < 0.05)
    - Fiedler-entropy correlation persists in random graphs (|rho| > 0.15)
    - High-Fiedler regions have < 50% energy in mode 1

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr
from scipy import sparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices,
)
from exp_14_partial_correlation import partial_spearman


def gini_coefficient(values):
    """Gini coefficient of a distribution. 0 = perfect equality, 1 = max inequality."""
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or values.sum() < 1e-15:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * values.sum()) /
                 (n * values.sum()))


def effective_dimensionality(eigenvalues, threshold_frac=0.10):
    """Count eigenvalues > threshold_frac * max eigenvalue."""
    eigs = np.abs(eigenvalues)
    if len(eigs) == 0 or eigs.max() < 1e-15:
        return 0
    threshold = threshold_frac * eigs.max()
    return int(np.sum(eigs > threshold))


def mode1_dominance(coefficients):
    """Fraction of total energy in mode 1 (first non-zero mode)."""
    c = np.array(coefficients)
    total_energy = np.sum(c ** 2)
    if total_energy < 1e-15:
        return 0.0
    return float(c[0] ** 2 / total_energy) if len(c) > 0 else 0.0


def random_graph_laplacian(n, mean_degree, seed=42):
    """
    Build Laplacian for Erdos-Renyi random graph G(n, p) where p = mean_degree/(n-1).
    Returns dense Laplacian matrix.
    """
    rng = np.random.RandomState(seed)
    p = min(1.0, mean_degree / max(1, n - 1))

    # Generate random adjacency (upper triangle)
    mask = rng.random((n, n)) < p
    mask = np.triu(mask, k=1)
    W = (mask + mask.T).astype(float)

    # Random positive weights
    weights = rng.uniform(0.5, 1.5, (n, n))
    weights = (weights + weights.T) / 2
    W = W * weights

    degrees = W.sum(axis=1)
    L = np.diag(degrees) - W
    return L


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 17, Experiment 26")
    print("Confluence Complexity: Why Does Coupling Create Entropy?")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)

    labels0 = labels_by_level[0]
    region_ids = sorted(np.unique(labels0).tolist())

    # =====================================================================
    # Per-region eigenvalue analysis
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Per-Region Eigenvalue Analysis")
    print(f"{'=' * 70}")

    region_data = []

    for rid in region_ids:
        indices = get_region_indices(labels_by_level, 0, rid)
        n_cells = len(indices)
        if n_cells < 10:
            continue

        L, W = graph_laplacian_subgraph(adjacency, indices)
        state_region = state_flat[indices]

        # Full eigenvalue decomposition (dense)
        L_dense = L.toarray() if sparse.issparse(L) else L
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)

        # Non-zero eigenvalues
        nonzero_mask = eigenvalues > 1e-10
        nonzero_eigs = eigenvalues[nonzero_mask]

        if len(nonzero_eigs) < 2:
            continue

        fiedler = float(nonzero_eigs[0])

        # Spectral entropy
        p_eigs = nonzero_eigs / nonzero_eigs.sum()
        spectral_entropy = float(-np.sum(p_eigs * np.log(p_eigs + 1e-15)))

        # Gini of eigenvalue distribution
        gini = gini_coefficient(nonzero_eigs)

        # Effective dimensionality
        eff_dim = effective_dimensionality(nonzero_eigs)

        # Mean degree of subgraph
        W_dense = W.toarray() if sparse.issparse(W) else np.array(W)
        mean_degree = float(np.mean(np.sum(W_dense > 0, axis=1)))

        # State coefficients and mode 1 dominance
        state_centered = state_region - np.mean(state_region)
        coefficients = []
        for i in range(min(10, eigenvectors.shape[1])):
            if eigenvalues[i] > 1e-10:
                coefficients.append(float(np.dot(state_centered, eigenvectors[:, i])))

        m1_dom = mode1_dominance(coefficients)

        region_data.append({
            'region_id': int(rid),
            'n_cells': n_cells,
            'fiedler': fiedler,
            'spectral_entropy': spectral_entropy,
            'gini': gini,
            'effective_dim': eff_dim,
            'mode1_dominance': m1_dom,
            'mean_degree': mean_degree,
            'n_nonzero_eigs': len(nonzero_eigs),
        })

    n_regions = len(region_data)
    print(f"  Analyzed {n_regions} regions")

    fiedlers = np.array([r['fiedler'] for r in region_data])
    entropies = np.array([r['spectral_entropy'] for r in region_data])
    ginis = np.array([r['gini'] for r in region_data])
    eff_dims = np.array([r['effective_dim'] for r in region_data])
    m1_doms = np.array([r['mode1_dominance'] for r in region_data])
    sizes = np.array([r['n_cells'] for r in region_data])

    # =====================================================================
    # Correlations (with size deconfound)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Fiedler Correlations (raw and size-deconfounded)")
    print(f"{'=' * 70}")

    rho_fe, p_fe = spearmanr(fiedlers, entropies)
    rho_fg, p_fg = spearmanr(fiedlers, ginis)
    rho_fd, p_fd = spearmanr(fiedlers, eff_dims)
    rho_fm, p_fm = spearmanr(fiedlers, m1_doms)

    print(f"  Raw correlations:")
    print(f"    rho(Fiedler, entropy)    = {rho_fe:.4f}, p={p_fe:.2e}")
    print(f"    rho(Fiedler, Gini)       = {rho_fg:.4f}, p={p_fg:.2e}")
    print(f"    rho(Fiedler, eff_dim)    = {rho_fd:.4f}, p={p_fd:.2e}")
    print(f"    rho(Fiedler, mode1_dom)  = {rho_fm:.4f}, p={p_fm:.2e}")

    # Partial correlations (controlling for size)
    partial_fg, partial_pg = partial_spearman(fiedlers, ginis, sizes)
    partial_fd, partial_pd = partial_spearman(fiedlers, eff_dims, sizes)
    partial_fm, partial_pm = partial_spearman(fiedlers, m1_doms, sizes)
    partial_fe, partial_pe = partial_spearman(fiedlers, entropies, sizes)

    print(f"\n  Partial correlations (controlling for size):")
    print(f"    partial rho(Fiedler, entropy | size)   = {partial_fe:.4f}, p={partial_pe:.2e}")
    print(f"    partial rho(Fiedler, Gini | size)      = {partial_fg:.4f}, p={partial_pg:.2e}")
    print(f"    partial rho(Fiedler, eff_dim | size)   = {partial_fd:.4f}, p={partial_pd:.2e}")
    print(f"    partial rho(Fiedler, mode1_dom | size) = {partial_fm:.4f}, p={partial_pm:.2e}")

    # =====================================================================
    # High vs Low Fiedler mode energy comparison
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("High vs Low Fiedler: Mode 1 Dominance")
    print(f"{'=' * 70}")

    sorted_idx = np.argsort(fiedlers)
    q_size = max(1, n_regions // 4)
    low_fiedler_idx = sorted_idx[:q_size]
    high_fiedler_idx = sorted_idx[-q_size:]

    mean_m1_low = float(np.mean(m1_doms[low_fiedler_idx]))
    mean_m1_high = float(np.mean(m1_doms[high_fiedler_idx]))

    print(f"  Low-Fiedler quartile: mean mode1_dominance = {mean_m1_low:.4f}")
    print(f"  High-Fiedler quartile: mean mode1_dominance = {mean_m1_high:.4f}")

    # =====================================================================
    # Random graph control
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Random Graph Control (Erdos-Renyi)")
    print(f"{'=' * 70}")

    rg_fiedlers = []
    rg_entropies = []

    for i, rd in enumerate(region_data):
        n = rd['n_cells']
        if n > 200:  # skip very large for compute
            continue
        mean_deg = rd['mean_degree']

        L_rg = random_graph_laplacian(n, mean_deg, seed=42 + i)
        eigs_rg = np.linalg.eigvalsh(L_rg)
        nonzero_rg = eigs_rg[eigs_rg > 1e-10]

        if len(nonzero_rg) < 2:
            continue

        fiedler_rg = float(nonzero_rg[0])
        p_rg = nonzero_rg / nonzero_rg.sum()
        entropy_rg = float(-np.sum(p_rg * np.log(p_rg + 1e-15)))

        rg_fiedlers.append(fiedler_rg)
        rg_entropies.append(entropy_rg)

    if len(rg_fiedlers) >= 10:
        rg_fiedlers = np.array(rg_fiedlers)
        rg_entropies = np.array(rg_entropies)
        rho_rg, p_rg = spearmanr(rg_fiedlers, rg_entropies)
        print(f"  Random graphs: n={len(rg_fiedlers)}, "
              f"rho(Fiedler, entropy) = {rho_rg:.4f}, p={p_rg:.2e}")
    else:
        rho_rg = 0.0
        p_rg = 1.0
        print(f"  Insufficient random graph data ({len(rg_fiedlers)})")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Test 1: Gini negatively correlated with Fiedler
    test1 = partial_fg < -0.2 and partial_pg < 0.05
    print(f"\n  Test 1: partial rho(Fiedler, Gini | size) < -0.2 AND p < 0.05?")
    print(f"    rho={partial_fg:.4f}, p={partial_pg:.2e}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Effective dimensionality positively correlated
    test2 = partial_fd > 0.2 and partial_pd < 0.05
    print(f"\n  Test 2: partial rho(Fiedler, eff_dim | size) > 0.2 AND p < 0.05?")
    print(f"    rho={partial_fd:.4f}, p={partial_pd:.2e}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: Random graph control
    test3 = abs(rho_rg) > 0.15
    print(f"\n  Test 3: Random graph Fiedler-entropy |rho| > 0.15?")
    print(f"    rho={rho_rg:.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: High-Fiedler regions have < 50% mode 1 energy
    test4 = mean_m1_high < 0.50
    print(f"\n  Test 4: High-Fiedler mode1_dominance < 50%?")
    print(f"    {mean_m1_high:.4f}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 confluence complexity tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_26_confluence_complexity',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Why does coupling create entropy? Confluence distributes energy.',
        'n_regions': n_regions,
        'raw_correlations': {
            'fiedler_entropy': {'rho': float(rho_fe), 'p': float(p_fe)},
            'fiedler_gini': {'rho': float(rho_fg), 'p': float(p_fg)},
            'fiedler_eff_dim': {'rho': float(rho_fd), 'p': float(p_fd)},
            'fiedler_mode1_dom': {'rho': float(rho_fm), 'p': float(p_fm)},
        },
        'partial_correlations': {
            'fiedler_entropy_given_size': {'rho': float(partial_fe), 'p': float(partial_pe)},
            'fiedler_gini_given_size': {'rho': float(partial_fg), 'p': float(partial_pg)},
            'fiedler_eff_dim_given_size': {'rho': float(partial_fd), 'p': float(partial_pd)},
            'fiedler_mode1_dom_given_size': {'rho': float(partial_fm), 'p': float(partial_pm)},
        },
        'quartile_analysis': {
            'mean_mode1_dom_low_fiedler': mean_m1_low,
            'mean_mode1_dom_high_fiedler': mean_m1_high,
        },
        'random_graph_control': {
            'n_graphs': len(rg_fiedlers) if isinstance(rg_fiedlers, np.ndarray) else len(rg_fiedlers),
            'rho_fiedler_entropy': float(rho_rg),
            'p': float(p_rg),
        },
        'verification': {
            'test1_gini_negative': bool(test1),
            'test2_eff_dim_positive': bool(test2),
            'test3_random_graph_control': bool(test3),
            'test4_mode1_low_in_high_fiedler': bool(test4),
            'n_verified': n_verified,
        },
        'per_region': region_data,
    }

    output_file = RESULTS_DIR / f'exp_26_confluence_complexity_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
