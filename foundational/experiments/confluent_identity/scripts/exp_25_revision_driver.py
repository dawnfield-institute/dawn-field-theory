"""
exp_25_revision_driver.py -- Confluent Identity Phase 16

PURPOSE:
    H^1 doesn't drive backward revision (exp_21: rho=0.016). What DOES?
    SEC-based smoothing operates on LOCAL gradients (diffusion), not global
    consistency. Test 8 candidate predictors to find the true driver.

METHODS:
    Using n=96 sub-partitioned groups from exp_21 infrastructure:
    For each group, compute 8 candidate predictors of spectral_revision:
    1. h1_proxy (control — expected to fail again)
    2. perturbation_l2: L2 norm of perturbation vector
    3. boundary_coupling: sum of edge weights crossing parent-child boundary
    4. n_children: number of child sub-regions
    5. size_ratio: max(child_size) / min(child_size) (asymmetry)
    6. spectral_gap_parent: Fiedler value of parent region
    7. perturbation_spread: entropy of perturbation distribution across children
    8. spectral_energy_shift: energy change in first 3 eigenmode coefficients

    Spearman rho for each predictor, then multiple regression.

VERIFICATION:
    - At least one non-H^1 predictor has |rho| > 0.3 AND p < 0.05
    - h1_proxy has |rho| < 0.15 (replication of falsification)
    - Multiple regression R^2 > 0.15
    - Top predictor is boundary_coupling or spectral_energy_shift

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
from exp_11_sheaf_cohomology import (
    build_coboundary_operator, build_section,
    perturb_children_only, smooth_section,
)
from exp_21_h1_revision_powered import kmeans_subpartition


MAX_PARENT_CELLS = 2000
MAX_MATRIX_SIZE = 8_000_000


def compute_boundary_coupling(adjacency, parent_indices, children_list):
    """
    Sum of edge weights crossing parent-child boundaries.
    Higher = more coupling between children = more revision potential.
    """
    # Build child membership map
    cell_to_child = {}
    for child_id, child_indices in children_list:
        for idx in child_indices:
            cell_to_child[int(idx)] = child_id

    boundary_weight = 0.0
    parent_set = set(int(i) for i in parent_indices)

    # Extract submatrix rows for parent indices
    W_sub = adjacency[parent_indices][:, parent_indices]

    # For each edge in the subgraph, check if endpoints are in different children
    if hasattr(W_sub, 'toarray'):
        W_dense = W_sub.toarray()
    else:
        W_dense = np.array(W_sub)

    n = len(parent_indices)
    for i in range(n):
        gi = int(parent_indices[i])
        ci = cell_to_child.get(gi, -1)
        for j in range(i + 1, n):
            gj = int(parent_indices[j])
            cj = cell_to_child.get(gj, -1)
            if ci != cj and ci >= 0 and cj >= 0:
                boundary_weight += W_dense[i, j]

    return float(boundary_weight)


def compute_perturbation_spread(state_flat, children_list, perturbation_scale, seed):
    """
    Entropy of how perturbation energy distributes across children.
    High entropy = perturbation evenly spread = more revision.
    """
    rng = np.random.RandomState(seed)
    child_energies = []
    for child_id, child_indices in children_list:
        n_child = len(child_indices)
        noise = rng.randn(n_child) * perturbation_scale * np.mean(state_flat[child_indices])
        energy = float(np.sum(noise ** 2))
        child_energies.append(energy)

    child_energies = np.array(child_energies)
    total = child_energies.sum()
    if total < 1e-15:
        return 0.0
    p = child_energies / total
    return float(-np.sum(p * np.log(p + 1e-15)))


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 16, Experiment 25")
    print("Revision Driver: What Actually Drives Backward Revision?")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)

    # =====================================================================
    # Build sub-partitioned groups (same as exp_21 Strategy A)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Building Sub-Partitioned Groups (exp_21 pattern)")
    print(f"{'=' * 70}")

    labels0 = labels_by_level[0]
    region_ids = sorted(np.unique(labels0).tolist())

    group_data = []
    perturbation_scale = 0.1
    alpha = 0.3

    for rid in region_ids:
        indices = get_region_indices(labels_by_level, 0, rid)
        n_cells = len(indices)
        if n_cells < 200:
            continue

        for k in [2, 3, 4]:
            children_indices = kmeans_subpartition(indices, N, k=k, seed=42 + rid + k)
            if children_indices is None:
                continue

            children_list = [(i, ci) for i, ci in enumerate(children_indices)]
            parent_indices = indices

            total_child_cells = sum(len(ci) for _, ci in children_list)
            dim_c0 = len(parent_indices) + total_child_cells
            dim_c1 = total_child_cells
            if dim_c0 * dim_c1 > MAX_MATRIX_SIZE:
                continue
            if len(parent_indices) > MAX_PARENT_CELLS:
                continue

            # --- Compute H^1 proxy ---
            delta = build_coboundary_operator(parent_indices, children_list)
            sigma_actual = build_section(state_flat, parent_indices, children_list)
            sigma_norm = float(np.linalg.norm(sigma_actual))
            if sigma_norm < 1e-15:
                continue

            state_perturbed = perturb_children_only(
                state_flat, children_list,
                perturbation_scale=perturbation_scale, seed=42
            )

            sigma_broken = sigma_actual.copy()
            offset = len(parent_indices)
            for child_id, child_idx in children_list:
                n_child = len(child_idx)
                sigma_broken[offset:offset + n_child] = state_perturbed[child_idx]
                offset += n_child

            residual_broken = float(np.linalg.norm(delta @ sigma_broken))
            h1_proxy = residual_broken / (sigma_norm + 1e-15)

            # --- Spectral revision ---
            L_parent, _ = graph_laplacian_subgraph(adjacency, parent_indices)
            state_parent_orig = state_flat[parent_indices]
            I_orig = compute_spectral_identity(L_parent, state_parent_orig)
            coeffs_orig = np.array(I_orig['state_coefficients'])

            state_parent_perturbed = state_perturbed[parent_indices]
            I_perturbed = compute_spectral_identity(L_parent, state_parent_perturbed)
            coeffs_perturbed = np.array(I_perturbed['state_coefficients'])

            state_smoothed = smooth_section(
                state_perturbed, state_flat, parent_indices, children_list,
                alpha=alpha
            )
            state_parent_smoothed = state_smoothed[parent_indices]
            I_smoothed = compute_spectral_identity(L_parent, state_parent_smoothed)
            coeffs_smoothed = np.array(I_smoothed['state_coefficients'])

            min_len = min(len(coeffs_orig), len(coeffs_perturbed), len(coeffs_smoothed))
            if min_len < 3:
                continue

            spectral_revision = float(np.linalg.norm(
                coeffs_smoothed[:min_len] - coeffs_perturbed[:min_len]))
            coeff_norm = float(np.linalg.norm(coeffs_orig[:min_len]))
            if coeff_norm < 1e-15:
                continue

            # --- 8 candidate predictors ---

            # 1. h1_proxy (already computed)

            # 2. perturbation_l2
            pert_vector = state_perturbed[parent_indices] - state_flat[parent_indices]
            perturbation_l2 = float(np.linalg.norm(pert_vector))

            # 3. boundary_coupling
            boundary_coupling = compute_boundary_coupling(
                adjacency, parent_indices, children_list)

            # 4. n_children
            n_children = len(children_list)

            # 5. size_ratio (asymmetry)
            child_sizes = [len(ci) for _, ci in children_list]
            size_ratio = max(child_sizes) / (min(child_sizes) + 1e-15)

            # 6. spectral_gap_parent (Fiedler)
            spectral_gap_parent = I_orig['fiedler_value']

            # 7. perturbation_spread
            perturbation_spread = compute_perturbation_spread(
                state_flat, children_list, perturbation_scale, seed=42)

            # 8. spectral_energy_shift (energy change in first 3 modes)
            energy_orig = float(np.sum(coeffs_orig[:3] ** 2))
            energy_pert = float(np.sum(coeffs_perturbed[:3] ** 2))
            spectral_energy_shift = abs(energy_pert - energy_orig) / (energy_orig + 1e-15)

            group_data.append({
                'source_region': int(rid),
                'k': k,
                'n_parent': len(parent_indices),
                'spectral_revision': spectral_revision,
                'predictors': {
                    'h1_proxy': h1_proxy,
                    'perturbation_l2': perturbation_l2,
                    'boundary_coupling': boundary_coupling,
                    'n_children': float(n_children),
                    'size_ratio': size_ratio,
                    'spectral_gap_parent': spectral_gap_parent,
                    'perturbation_spread': perturbation_spread,
                    'spectral_energy_shift': spectral_energy_shift,
                },
            })

    n_groups = len(group_data)
    print(f"  Generated {n_groups} groups")

    if n_groups < 10:
        print("  INSUFFICIENT DATA — aborting")
        return None

    # =====================================================================
    # Univariate correlations
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Univariate Correlations: predictor vs spectral_revision")
    print(f"{'=' * 70}")

    revision = np.array([g['spectral_revision'] for g in group_data])
    predictor_names = list(group_data[0]['predictors'].keys())
    predictor_results = {}

    for pname in predictor_names:
        values = np.array([g['predictors'][pname] for g in group_data])
        rho, p = spearmanr(values, revision)
        predictor_results[pname] = {'rho': float(rho), 'p': float(p)}
        sig = '*' if p < 0.05 else ''
        print(f"  {pname:<25} rho={rho:>7.4f}  p={p:.2e} {sig}")

    # Find best non-H^1 predictor
    best_non_h1 = max(
        ((name, res) for name, res in predictor_results.items() if name != 'h1_proxy'),
        key=lambda x: abs(x[1]['rho'])
    )

    print(f"\n  Best non-H^1 predictor: {best_non_h1[0]} "
          f"(rho={best_non_h1[1]['rho']:.4f}, p={best_non_h1[1]['p']:.2e})")

    # =====================================================================
    # Multiple regression (rank-based OLS)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Multiple Regression (rank-based)")
    print(f"{'=' * 70}")

    # Use top 3 non-H^1 predictors by |rho|
    sorted_predictors = sorted(
        [(name, res) for name, res in predictor_results.items() if name != 'h1_proxy'],
        key=lambda x: -abs(x[1]['rho'])
    )[:3]

    X_cols = []
    col_names = []
    for pname, _ in sorted_predictors:
        values = np.array([g['predictors'][pname] for g in group_data])
        X_cols.append(rankdata(values))
        col_names.append(pname)

    X = np.column_stack(X_cols)
    X = np.column_stack([np.ones(n_groups), X])  # add intercept
    y = rankdata(revision)

    # OLS: beta = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1 - ss_res / (ss_tot + 1e-15)
    except np.linalg.LinAlgError:
        r_squared = 0.0
        beta = np.zeros(len(col_names) + 1)

    print(f"  Predictors: {col_names}")
    print(f"  R² = {r_squared:.4f}")
    for i, name in enumerate(col_names):
        print(f"    {name}: beta = {beta[i+1]:.4f}")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    test1 = abs(best_non_h1[1]['rho']) > 0.3 and best_non_h1[1]['p'] < 0.05
    print(f"\n  Test 1: Non-H^1 predictor |rho| > 0.3 AND p < 0.05?")
    print(f"    {best_non_h1[0]}: rho={best_non_h1[1]['rho']:.4f}, "
          f"p={best_non_h1[1]['p']:.2e}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    h1_rho = abs(predictor_results['h1_proxy']['rho'])
    test2 = h1_rho < 0.15
    print(f"\n  Test 2: h1_proxy |rho| < 0.15 (falsification replication)?")
    print(f"    |rho| = {h1_rho:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    test3 = r_squared > 0.15
    print(f"\n  Test 3: Multiple regression R² > 0.15?")
    print(f"    R² = {r_squared:.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    meaningful_predictors = {'boundary_coupling', 'spectral_energy_shift'}
    test4 = best_non_h1[0] in meaningful_predictors
    print(f"\n  Test 4: Top predictor is boundary_coupling or spectral_energy_shift?")
    print(f"    Top: {best_non_h1[0]}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 revision driver tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_25_revision_driver',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'What drives backward revision? 8 candidate predictors',
        'n_groups': n_groups,
        'predictor_correlations': predictor_results,
        'best_non_h1': {
            'name': best_non_h1[0],
            'rho': best_non_h1[1]['rho'],
            'p': best_non_h1[1]['p'],
        },
        'multiple_regression': {
            'predictors': col_names,
            'r_squared': float(r_squared),
            'betas': {name: float(beta[i+1]) for i, name in enumerate(col_names)},
        },
        'verification': {
            'test1_non_h1_predictor': bool(test1),
            'test2_h1_falsification_replication': bool(test2),
            'test3_regression_r_squared': bool(test3),
            'test4_meaningful_top_predictor': bool(test4),
            'n_verified': n_verified,
        },
        'group_data': group_data,
    }

    output_file = RESULTS_DIR / f'exp_25_revision_driver_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
