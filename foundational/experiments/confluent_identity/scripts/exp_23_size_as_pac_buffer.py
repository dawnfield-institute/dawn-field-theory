"""
exp_23_size_as_pac_buffer.py -- Confluent Identity Phase 14

PURPOSE:
    Fiedler was FALSIFIED as a coherence predictor (exp_18: partial rho=+0.07
    after size deconfound). But the raw correlation was real (rho=-0.32). This
    means SIZE is the true predictor of identity robustness, not connectivity.

    PAC hypothesis: Size predicts perturbation sensitivity because PAC
    conservation means perturbation energy is spread across N cells. Expected
    scaling: sensitivity ~ 1/sqrt(N) (central limit theorem on N conserved cells).

METHODS:
    1. partial_rho(size, noise_sensitivity | Fiedler) -- is size the TRUE predictor?
    2. Power-law fit: sensitivity ~ N^(-alpha), PAC predicts alpha ~ 0.5
    3. Sub-partition test: halve large regions, compare sensitivity ratio to sqrt(2)
    4. Multi-perturbation consistency: does 1/sqrt(N) hold for Gaussian, uniform, and
       structured perturbations?

VERIFICATION:
    - partial_rho(size, noise_sensitivity | Fiedler) > 0.3 AND p < 0.05
    - Power-law exponent alpha in [0.3, 0.7]
    - Sub-partition sensitivity ratio in [1.2, 1.8] for >= 3/5 large regions
    - Exponent alpha consistent across perturbation types (std < 0.15)

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr, linregress

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices, compute_subgraph_laplacian_from_field,
)
from exp_14_partial_correlation import partial_spearman
from exp_21_h1_revision_powered import kmeans_subpartition


def compute_sensitivity(L, state_region, perturbation, coeffs_base, coeff_norm):
    """Compute identity shift from a perturbation vector."""
    state_perturbed = state_region + perturbation
    I_pert = compute_spectral_identity(L, state_perturbed)
    coeffs_pert = np.array(I_pert['state_coefficients'])
    min_len = min(len(coeffs_base), len(coeffs_pert))
    shift = float(np.linalg.norm(
        coeffs_pert[:min_len] - coeffs_base[:min_len]))
    return shift / (coeff_norm + 1e-15)


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 14, Experiment 23")
    print("Size as PAC Buffer: Conservation Predicts Robustness")
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
    # Collect per-region: size, Fiedler, sensitivity under 3 perturbation types
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Per-Region Analysis: 3 Perturbation Types")
    print(f"{'=' * 70}")

    region_data = []

    for rid in region_ids:
        indices = get_region_indices(labels_by_level, 0, rid)
        n_cells = len(indices)
        if n_cells < 10:
            continue

        L, _ = graph_laplacian_subgraph(adjacency, indices)
        state_region = state_flat[indices]
        I_baseline = compute_spectral_identity(L, state_region)

        fiedler = I_baseline['fiedler_value']
        coeffs_base = np.array(I_baseline['state_coefficients'])
        coeff_norm = float(np.linalg.norm(coeffs_base))
        if coeff_norm < 1e-15:
            continue

        mean_state = float(np.mean(state_region))

        # Type 1: Gaussian noise (random per cell)
        rng = np.random.RandomState(42 + rid)
        noise_gauss = rng.randn(n_cells) * 0.1 * mean_state
        sens_gauss = compute_sensitivity(L, state_region, noise_gauss,
                                         coeffs_base, coeff_norm)

        # Type 2: Uniform perturbation (same epsilon to all cells)
        eps_uniform = 0.1 * mean_state
        noise_uniform = np.full(n_cells, eps_uniform)
        sens_uniform = compute_sensitivity(L, state_region, noise_uniform,
                                           coeffs_base, coeff_norm)

        # Type 3: Structured (remove 10% of cells, use subgraph Laplacian)
        n_remove = max(1, int(n_cells * 0.10))
        rng2 = np.random.RandomState(42 + rid)
        remove_mask = rng2.choice(n_cells, n_remove, replace=False)
        remaining_mask = np.ones(n_cells, dtype=bool)
        remaining_mask[remove_mask] = False
        remaining_indices = indices[remaining_mask]

        if len(remaining_indices) >= 10:
            L_rem, _ = compute_subgraph_laplacian_from_field(
                state_flat, remaining_indices, N)
            state_remaining = state_flat[remaining_indices]
            I_rem = compute_spectral_identity(L_rem, state_remaining)
            coeffs_rem = np.array(I_rem['state_coefficients'])
            min_len = min(len(coeffs_base), len(coeffs_rem))
            sens_struct = float(np.linalg.norm(
                coeffs_rem[:min_len] - coeffs_base[:min_len])) / (coeff_norm + 1e-15)
        else:
            sens_struct = None

        region_data.append({
            'region_id': int(rid),
            'n_cells': n_cells,
            'fiedler': float(fiedler),
            'sens_gauss': float(sens_gauss),
            'sens_uniform': float(sens_uniform),
            'sens_structured': float(sens_struct) if sens_struct is not None else None,
        })

    n_regions = len(region_data)
    print(f"  Analyzed {n_regions} regions")

    sizes = np.array([r['n_cells'] for r in region_data])
    fiedlers = np.array([r['fiedler'] for r in region_data])
    sens_g = np.array([r['sens_gauss'] for r in region_data])
    sens_u = np.array([r['sens_uniform'] for r in region_data])
    sens_s_valid = [(r['n_cells'], r['sens_structured']) for r in region_data
                    if r['sens_structured'] is not None]

    # =====================================================================
    # Test 1: partial_rho(size, noise_sensitivity | Fiedler)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("1. Size as True Predictor (Fiedler Deconfound)")
    print(f"{'=' * 70}")

    # Use negative size — larger should be LESS sensitive
    neg_sizes = -sizes.astype(float)
    partial_rho_sf, partial_p_sf = partial_spearman(neg_sizes, sens_g, fiedlers)
    print(f"  partial_rho(-size, gauss_sens | Fiedler) = {partial_rho_sf:.4f}, "
          f"p = {partial_p_sf:.2e}")

    # Also raw correlations for context
    rho_size_sens, p_size_sens = spearmanr(sizes, sens_g)
    rho_fiedler_sens, p_fiedler_sens = spearmanr(fiedlers, sens_g)
    print(f"  Raw rho(size, gauss_sens) = {rho_size_sens:.4f}")
    print(f"  Raw rho(Fiedler, gauss_sens) = {rho_fiedler_sens:.4f}")

    # =====================================================================
    # Test 2: Power-law fit: sensitivity ~ N^(-alpha)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("2. Power-Law Scaling: sensitivity ~ N^(-alpha)")
    print(f"{'=' * 70}")

    log_sizes = np.log(sizes.astype(float))
    log_sens_g = np.log(sens_g + 1e-15)
    log_sens_u = np.log(sens_u + 1e-15)

    # Gaussian
    slope_g, intercept_g, r_g, p_slope_g, se_g = linregress(log_sizes, log_sens_g)
    alpha_gauss = -slope_g  # negative slope = positive alpha
    print(f"  Gaussian: alpha = {alpha_gauss:.4f} (slope = {slope_g:.4f}), "
          f"R² = {r_g**2:.4f}, p = {p_slope_g:.2e}")

    # Uniform
    slope_u, intercept_u, r_u, p_slope_u, se_u = linregress(log_sizes, log_sens_u)
    alpha_uniform = -slope_u
    print(f"  Uniform:  alpha = {alpha_uniform:.4f} (slope = {slope_u:.4f}), "
          f"R² = {r_u**2:.4f}, p = {p_slope_u:.2e}")

    # Structured
    if len(sens_s_valid) >= 10:
        sizes_s = np.array([x[0] for x in sens_s_valid], dtype=float)
        sens_s_arr = np.array([x[1] for x in sens_s_valid])
        log_sizes_s = np.log(sizes_s)
        log_sens_s = np.log(sens_s_arr + 1e-15)
        slope_s, intercept_s, r_s, p_slope_s, se_s = linregress(log_sizes_s, log_sens_s)
        alpha_struct = -slope_s
        print(f"  Struct:   alpha = {alpha_struct:.4f} (slope = {slope_s:.4f}), "
              f"R² = {r_s**2:.4f}, p = {p_slope_s:.2e}")
    else:
        alpha_struct = None
        print(f"  Struct:   insufficient data ({len(sens_s_valid)})")

    alphas = [alpha_gauss, alpha_uniform]
    if alpha_struct is not None:
        alphas.append(alpha_struct)
    alpha_mean = float(np.mean(alphas))
    alpha_std = float(np.std(alphas))
    print(f"\n  Mean alpha = {alpha_mean:.4f}, std = {alpha_std:.4f}")
    print(f"  PAC prediction: alpha ~ 0.5 (CLT on N conserved cells)")

    # =====================================================================
    # Test 3: Sub-partition sensitivity ratio ~ sqrt(2)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("3. Sub-Partition Test: Halve Region, Check sqrt(2) Ratio")
    print(f"{'=' * 70}")

    # Find large regions (>400 cells) for sub-partitioning
    large_regions = [(r['region_id'], r['n_cells'], r['sens_gauss'])
                     for r in region_data if r['n_cells'] > 400]
    large_regions.sort(key=lambda x: -x[1])  # largest first
    selected_large = large_regions[:5]

    subpart_results = []
    for rid, n_cells_full, sens_full in selected_large:
        indices = get_region_indices(labels_by_level, 0, rid)

        # Split into 2 halves via k-means
        children = kmeans_subpartition(indices, N, k=2, seed=42 + rid)
        if children is None or len(children) < 2:
            continue

        # Compute sensitivity for each half
        half_sensitivities = []
        for child_indices in children:
            if len(child_indices) < 10:
                continue
            L_half, _ = graph_laplacian_subgraph(adjacency, child_indices)
            state_half = state_flat[child_indices]
            I_half = compute_spectral_identity(L_half, state_half)
            coeffs_half = np.array(I_half['state_coefficients'])
            norm_half = float(np.linalg.norm(coeffs_half))
            if norm_half < 1e-15:
                continue

            rng_half = np.random.RandomState(42 + rid)
            noise_half = rng_half.randn(len(child_indices)) * 0.1 * np.mean(state_half)
            sens_half = compute_sensitivity(L_half, state_half, noise_half,
                                            coeffs_half, norm_half)
            half_sensitivities.append(sens_half)

        if len(half_sensitivities) >= 2:
            mean_half_sens = float(np.mean(half_sensitivities))
            ratio = mean_half_sens / (sens_full + 1e-15)
            subpart_results.append({
                'region_id': int(rid),
                'n_cells_full': n_cells_full,
                'n_halves': len(half_sensitivities),
                'sens_full': float(sens_full),
                'mean_sens_half': mean_half_sens,
                'ratio': ratio,
            })
            print(f"  R{rid}: {n_cells_full} cells, "
                  f"full={sens_full:.4f}, half={mean_half_sens:.4f}, "
                  f"ratio={ratio:.3f} (target: ~1.41)")

    n_in_range = sum(1 for r in subpart_results if 1.2 <= r['ratio'] <= 1.8)
    print(f"\n  {n_in_range}/{len(subpart_results)} regions with ratio in [1.2, 1.8]")

    # =====================================================================
    # Test 4: Exponent consistency across perturbation types
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("4. Cross-Perturbation Consistency")
    print(f"{'=' * 70}")

    print(f"  Gaussian alpha:    {alpha_gauss:.4f}")
    print(f"  Uniform alpha:     {alpha_uniform:.4f}")
    if alpha_struct is not None:
        print(f"  Structured alpha:  {alpha_struct:.4f}")
    print(f"  Std across types:  {alpha_std:.4f}")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Test 1: Size is true predictor after Fiedler deconfound
    test1 = partial_rho_sf > 0.3 and partial_p_sf < 0.05
    print(f"\n  Test 1: partial_rho(-size, sens | Fiedler) > 0.3 AND p < 0.05?")
    print(f"    rho={partial_rho_sf:.4f}, p={partial_p_sf:.2e}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Power-law exponent in [0.3, 0.7]
    test2 = 0.3 <= alpha_gauss <= 0.7
    print(f"\n  Test 2: Power-law exponent alpha in [0.3, 0.7]?")
    print(f"    alpha_gauss = {alpha_gauss:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: Sub-partition ratio in [1.2, 1.8] for >= 3/5 regions
    test3 = n_in_range >= 3 and len(subpart_results) >= 3
    print(f"\n  Test 3: Sub-partition ratio in [1.2, 1.8] for >= 3/5 regions?")
    print(f"    {n_in_range}/{len(subpart_results)} in range")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: Exponent consistent (std < 0.15)
    test4 = alpha_std < 0.15
    print(f"\n  Test 4: Exponent alpha consistent (std < 0.15)?")
    print(f"    std = {alpha_std:.4f}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 PAC buffer tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_23_size_as_pac_buffer',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Size as PAC conservation buffer: sensitivity ~ 1/sqrt(N)',
        'n_regions': n_regions,
        'deconfound': {
            'partial_rho_size_sens_given_fiedler': float(partial_rho_sf),
            'partial_p': float(partial_p_sf),
            'raw_rho_size_sens': float(rho_size_sens),
            'raw_rho_fiedler_sens': float(rho_fiedler_sens),
        },
        'power_law': {
            'alpha_gauss': float(alpha_gauss),
            'alpha_uniform': float(alpha_uniform),
            'alpha_structured': float(alpha_struct) if alpha_struct is not None else None,
            'alpha_mean': alpha_mean,
            'alpha_std': alpha_std,
            'r2_gauss': float(r_g**2),
            'r2_uniform': float(r_u**2),
        },
        'subpartition': {
            'n_tested': len(subpart_results),
            'n_in_range': n_in_range,
            'results': subpart_results,
        },
        'verification': {
            'test1_size_true_predictor': bool(test1),
            'test2_power_law_exponent': bool(test2),
            'test3_subpartition_ratio': bool(test3),
            'test4_cross_type_consistency': bool(test4),
            'n_verified': n_verified,
        },
        'per_region': region_data,
    }

    output_file = RESULTS_DIR / f'exp_23_pac_buffer_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
