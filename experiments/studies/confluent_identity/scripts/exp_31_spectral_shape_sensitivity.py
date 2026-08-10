"""
exp_31_spectral_shape_sensitivity.py -- Confluent Identity Phase 22

PURPOSE:
    Test whether eigenmode energy DISTRIBUTION SHAPE predicts sensitivity.
    All prior spectral predictors (Fiedler, entropy, Gini, eff_dim) are
    scalar summaries that collapsed to size after deconfound. Spectral
    flatness is scale-invariant by construction (ratio). Mid-mode ratio
    directly tests the exp_24 clue that modes 4-7 dominate all perturbation
    types equally at ~24%.

METHODS:
    For each level-0 region (>=20 cells):
    1. Compute state_coefficients c_k, eigenmode energy E_k = c_k^2
    2. 5 spectral shape descriptors:
       - spectral_flatness (SF): exp(mean(log(E_k))) / mean(E_k)
       - spectral_centroid (SC): sum(k*E_k) / sum(E_k)
       - spectral_bandwidth (SB): sqrt(sum((k-SC)^2 * E_k) / sum(E_k))
       - mode_concentration (MC): max(E_k) / sum(E_k)
       - mid_mode_ratio (MMR): sum(E_k, k=4..7) / sum(E_k)
    3. Gaussian sensitivity (exp_16 pattern)
    4. Raw + partial Spearman (controlling for size) per descriptor
    5. Multiple regression: SF + SC + MMR -> sensitivity | size

VERIFICATION:
    - At least one descriptor |partial_rho(desc, sens | size)| > 0.25, p < 0.05
    - Spectral flatness NOT a size proxy: |rho(SF, size)| < 0.40
    - Mid-mode ratio partial_rho with sensitivity > 0.15
    - Multiple regression R^2 (SF+SC+MMR | size) > 0.15

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr, rankdata

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices,
)
from exp_14_partial_correlation import partial_spearman


def compute_spectral_shape(coefficients):
    """
    Compute 5 spectral shape descriptors from eigenmode coefficients.
    coefficients: 1D array of state_coefficients (skip index 0 = harmonic/DC).

    Returns dict of shape descriptors, or None if insufficient data.
    """
    # Skip harmonic (k=0), use modes 1..K
    c = np.array(coefficients)
    if len(c) < 3:
        return None

    # Mode energies (skip k=0)
    E = c[1:] ** 2
    k_indices = np.arange(1, len(c))

    total_energy = np.sum(E)
    if total_energy < 1e-30:
        return None

    # Normalize to probability distribution
    p = E / total_energy

    # 1. Spectral flatness: exp(mean(log(E))) / mean(E)
    # Use log of normalized energies to avoid underflow
    log_E = np.log(E + 1e-30)
    sf = float(np.exp(np.mean(log_E)) / (np.mean(E) + 1e-30))

    # 2. Spectral centroid: weighted average mode index
    sc = float(np.sum(k_indices * p))

    # 3. Spectral bandwidth: weighted std of mode index
    sb = float(np.sqrt(np.sum((k_indices - sc) ** 2 * p)))

    # 4. Mode concentration: fraction in dominant mode
    mc = float(np.max(p))

    # 5. Mid-mode ratio: energy in modes 4-7 / total
    mid_mask = (k_indices >= 4) & (k_indices <= 7)
    mmr = float(np.sum(E[mid_mask]) / total_energy) if mid_mask.any() else 0.0

    return {
        'spectral_flatness': sf,
        'spectral_centroid': sc,
        'spectral_bandwidth': sb,
        'mode_concentration': mc,
        'mid_mode_ratio': mmr,
    }


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 22, Experiment 31")
    print("Spectral Shape Descriptors as Sensitivity Predictors")
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
    # Per-region spectral shape + sensitivity
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Per-Region Spectral Shape + Sensitivity")
    print(f"{'=' * 70}")

    region_data = []

    for rid in region_ids:
        indices = get_region_indices(labels_by_level, 0, rid)
        n_cells = len(indices)
        if n_cells < 20:
            continue

        L, _ = graph_laplacian_subgraph(adjacency, indices)
        state_region = state_flat[indices]
        I_base = compute_spectral_identity(L, state_region)
        coeffs_base = np.array(I_base['state_coefficients'])
        coeff_norm = float(np.linalg.norm(coeffs_base))

        if coeff_norm < 1e-15:
            continue

        # Spectral shape
        shape = compute_spectral_shape(coeffs_base)
        if shape is None:
            continue

        # Gaussian sensitivity
        rng = np.random.RandomState(42 + rid)
        noise = rng.randn(n_cells) * 0.1 * np.mean(state_region)
        state_noisy = state_region + noise
        I_noisy = compute_spectral_identity(L, state_noisy)
        coeffs_noisy = np.array(I_noisy['state_coefficients'])
        min_len = min(len(coeffs_base), len(coeffs_noisy))
        delta = float(np.linalg.norm(coeffs_noisy[:min_len] - coeffs_base[:min_len]))
        sensitivity = delta / (coeff_norm + 1e-15)

        region_data.append({
            'region_id': int(rid),
            'n_cells': n_cells,
            'sensitivity': sensitivity,
            'fiedler': float(I_base['fiedler_value']),
            **shape,
        })

    n_regions = len(region_data)
    print(f"  Analyzed {n_regions} regions")

    # Extract arrays
    sizes = np.array([r['n_cells'] for r in region_data], dtype=float)
    sensitivities = np.array([r['sensitivity'] for r in region_data])
    sf = np.array([r['spectral_flatness'] for r in region_data])
    sc = np.array([r['spectral_centroid'] for r in region_data])
    sb = np.array([r['spectral_bandwidth'] for r in region_data])
    mc = np.array([r['mode_concentration'] for r in region_data])
    mmr = np.array([r['mid_mode_ratio'] for r in region_data])

    descriptor_names = ['spectral_flatness', 'spectral_centroid', 'spectral_bandwidth',
                        'mode_concentration', 'mid_mode_ratio']
    descriptor_arrays = [sf, sc, sb, mc, mmr]

    # =====================================================================
    # Raw correlations with sensitivity
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Raw Correlations: spectral shape vs sensitivity")
    print(f"{'=' * 70}")

    raw_results = {}
    for name, arr in zip(descriptor_names, descriptor_arrays):
        rho, p = spearmanr(arr, sensitivities)
        raw_results[name] = {'rho': float(rho), 'p': float(p)}
        print(f"  rho({name}, sensitivity) = {rho:.4f}, p={p:.2e}")

    # Size correlations
    print(f"\n  Size correlations:")
    size_corrs = {}
    for name, arr in zip(descriptor_names, descriptor_arrays):
        rho_s, _ = spearmanr(arr, sizes)
        size_corrs[name] = float(rho_s)
        print(f"    rho({name}, size) = {rho_s:.4f}")

    # =====================================================================
    # Partial correlations (controlling for size)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Partial Correlations: spectral shape vs sensitivity | size")
    print(f"{'=' * 70}")

    partial_results = {}
    for name, arr in zip(descriptor_names, descriptor_arrays):
        pr, pp = partial_spearman(arr, sensitivities, sizes)
        partial_results[name] = {'rho': float(pr), 'p': float(pp)}
        sig = '*' if pp < 0.05 else ''
        print(f"  partial_rho({name}, sens | size) = {pr:.4f}, p={pp:.2e} {sig}")

    # =====================================================================
    # Multiple regression (rank-based)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Multiple Regression: SF + SC + MMR -> sensitivity | size")
    print(f"{'=' * 70}")

    # Rank all variables, then regress out size ranks
    y_rank = rankdata(sensitivities)
    size_rank = rankdata(sizes)

    # Residualize y on size
    X_size = np.column_stack([np.ones(n_regions), size_rank])
    beta_y = np.linalg.lstsq(X_size, y_rank, rcond=None)[0]
    y_resid = y_rank - X_size @ beta_y

    # Residualize predictors on size
    pred_resids = []
    for arr in [sf, sc, mmr]:
        r = rankdata(arr)
        beta_p = np.linalg.lstsq(X_size, r, rcond=None)[0]
        pred_resids.append(r - X_size @ beta_p)

    X_pred = np.column_stack([np.ones(n_regions)] + pred_resids)

    try:
        beta = np.linalg.lstsq(X_pred, y_resid, rcond=None)[0]
        y_pred = X_pred @ beta
        ss_res = float(np.sum((y_resid - y_pred) ** 2))
        ss_tot = float(np.sum((y_resid - y_resid.mean()) ** 2))
        r_squared = 1 - ss_res / (ss_tot + 1e-15)
    except np.linalg.LinAlgError:
        r_squared = 0.0
        beta = np.zeros(4)

    pred_names = ['spectral_flatness', 'spectral_centroid', 'mid_mode_ratio']
    print(f"  R² (after size residualization) = {r_squared:.4f}")
    for i, name in enumerate(pred_names):
        print(f"    {name}: beta = {beta[i+1]:.4f}")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Test 1: At least one descriptor |partial_rho| > 0.25, p < 0.05
    sig_descriptors = [(name, pr) for name, pr in partial_results.items()
                       if abs(pr['rho']) > 0.25 and pr['p'] < 0.05]
    test1 = len(sig_descriptors) > 0
    print(f"\n  Test 1: Any descriptor |partial_rho| > 0.25, p < 0.05?")
    if sig_descriptors:
        for name, pr in sig_descriptors:
            print(f"    {name}: rho={pr['rho']:.4f}, p={pr['p']:.2e}")
    else:
        best = max(partial_results.items(), key=lambda x: abs(x[1]['rho']))
        print(f"    Best: {best[0]} rho={best[1]['rho']:.4f}, p={best[1]['p']:.2e}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Spectral flatness NOT a size proxy
    sf_size_rho = abs(size_corrs['spectral_flatness'])
    test2 = sf_size_rho < 0.40
    print(f"\n  Test 2: |rho(spectral_flatness, size)| < 0.40?")
    print(f"    |rho| = {sf_size_rho:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: Mid-mode ratio partial_rho > 0.15
    mmr_partial = partial_results['mid_mode_ratio']['rho']
    test3 = mmr_partial > 0.15
    print(f"\n  Test 3: partial_rho(mid_mode_ratio, sens | size) > 0.15?")
    print(f"    partial_rho = {mmr_partial:.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: Multiple regression R^2 > 0.15
    test4 = r_squared > 0.15
    print(f"\n  Test 4: Multiple regression R² > 0.15?")
    print(f"    R² = {r_squared:.4f}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 spectral shape tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_31_spectral_shape_sensitivity',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Spectral shape descriptors as sensitivity predictors',
        'n_regions': n_regions,
        'raw_correlations': raw_results,
        'size_correlations': size_corrs,
        'partial_correlations': partial_results,
        'multiple_regression': {
            'predictors': pred_names,
            'r_squared': float(r_squared),
            'betas': {name: float(beta[i+1]) for i, name in enumerate(pred_names)},
        },
        'verification': {
            'test1_descriptor_signal': bool(test1),
            'test2_flatness_not_size': bool(test2),
            'test3_mid_mode_ratio': bool(test3),
            'test4_regression_r_squared': bool(test4),
            'n_verified': n_verified,
        },
        'per_region': region_data,
    }

    output_file = RESULTS_DIR / f'exp_31_spectral_shape_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
