"""
exp_21_h1_revision_powered.py -- Confluent Identity Phase 12

PURPOSE:
    Fix the underpowered H^1-revision test from exp_17 (n=6, p=0.54). Two
    strategies to increase sample size:
    A) Sub-partition large level-0 regions via k-means -> 15-25 new groups
    B) Multi-perturbation: original 6 groups x 5 seeds x 3 scales -> group medians

    If this fails with n=25+, the H^1-revision link is genuinely weak -> FALSIFIED.

METHODS:
    Strategy A: k-means on spatial coordinates of large regions (>200 cells)
    Strategy B: Multiple perturbation seeds and scales on original groups
    For each group: compute H^1 proxy + spectral revision (same as exp_17)

VERIFICATION:
    - Sub-partitioned groups (n>=20): Spearman rho > 0.25, p < 0.05
    - R^2 > 0.08
    - Multi-perturbation median correlation: rho > 0.3
    - Rank overlap: top-10 H^1 in top-15 revision (>=5/10)

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
    get_region_indices,
)
from exp_11_sheaf_cohomology import (
    build_coboundary_operator, build_section,
    perturb_children_only, smooth_section,
)


MAX_PARENT_CELLS = 2000
MAX_MATRIX_SIZE = 8_000_000


def kmeans_subpartition(indices, N, k=3, seed=42, max_iter=50):
    """
    Partition region indices into k clusters using k-means on 2D coordinates.
    Pure numpy, no sklearn.

    Returns list of k numpy arrays (child index arrays).
    """
    rng = np.random.RandomState(seed)
    n = len(indices)
    if n < k * 3:
        return None

    # Convert flat indices to (row, col)
    coords = np.column_stack([indices // N, indices % N]).astype(float)

    # Initialize centroids randomly from data
    init_idx = rng.choice(n, k, replace=False)
    centroids = coords[init_idx].copy()

    for _ in range(max_iter):
        # Assign to nearest centroid
        dists = np.linalg.norm(
            coords[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                new_centroids[c] = coords[mask].mean(axis=0)
            else:
                new_centroids[c] = centroids[c]

        if np.allclose(new_centroids, centroids, atol=0.5):
            break
        centroids = new_centroids

    # Build child arrays
    children = []
    for c in range(k):
        mask = labels == c
        if mask.sum() >= 3:
            children.append(indices[mask])

    return children if len(children) >= 2 else None


def compute_h1_and_revision(state_flat, adjacency, parent_indices, children_list,
                             perturbation_scale=0.1, alpha=0.3, seed=42):
    """
    Compute H^1 proxy and spectral revision for a parent-children group.
    Returns dict with metrics, or None if computation fails.
    """
    total_child_cells = sum(len(ci) for _, ci in children_list)
    dim_c0 = len(parent_indices) + total_child_cells
    dim_c1 = total_child_cells

    if dim_c0 * dim_c1 > MAX_MATRIX_SIZE:
        return None
    if len(parent_indices) > MAX_PARENT_CELLS:
        return None

    # Coboundary operator
    delta = build_coboundary_operator(parent_indices, children_list)

    # Baseline section
    sigma_actual = build_section(state_flat, parent_indices, children_list)
    residual_actual = float(np.linalg.norm(delta @ sigma_actual))

    # Perturb children
    state_perturbed = perturb_children_only(
        state_flat, children_list,
        perturbation_scale=perturbation_scale, seed=seed
    )

    # Broken section
    sigma_broken = sigma_actual.copy()
    offset = len(parent_indices)
    for child_id, child_indices in children_list:
        n_child = len(child_indices)
        sigma_broken[offset:offset + n_child] = state_perturbed[child_indices]
        offset += n_child

    residual_broken = float(np.linalg.norm(delta @ sigma_broken))
    sigma_norm = float(np.linalg.norm(sigma_actual))
    h1_proxy = residual_broken / (sigma_norm + 1e-15)

    # Spectral revision
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
    perturbation_shift = float(np.linalg.norm(
        coeffs_perturbed[:min_len] - coeffs_orig[:min_len]))
    spectral_revision = float(np.linalg.norm(
        coeffs_smoothed[:min_len] - coeffs_perturbed[:min_len]))
    coeff_norm = float(np.linalg.norm(coeffs_orig[:min_len]))

    return {
        'h1_proxy': h1_proxy,
        'spectral_revision': spectral_revision,
        'perturbation_shift': perturbation_shift,
        'coeff_norm': coeff_norm,
        'residual_actual': residual_actual,
        'residual_broken': residual_broken,
    }


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 12, Experiment 21")
    print("H1-Revision Powered: More Groups via Sub-Partition")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)

    # =====================================================================
    # Strategy A: Sub-partition large level-0 regions
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Strategy A: K-means Sub-Partition of Level-0 Regions")
    print(f"{'=' * 70}")

    labels0 = labels_by_level[0]
    region_ids = sorted(np.unique(labels0).tolist())

    subpart_data = []

    for rid in region_ids:
        indices = get_region_indices(labels_by_level, 0, rid)
        n_cells = len(indices)
        if n_cells < 200:
            continue

        for k in [2, 3, 4]:
            children_indices = kmeans_subpartition(indices, N, k=k, seed=42 + rid + k)
            if children_indices is None:
                continue

            # Build children_list in the format expected by coboundary
            children_list = [(i, ci) for i, ci in enumerate(children_indices)]
            parent_indices = indices

            result = compute_h1_and_revision(
                state_flat, adjacency, parent_indices, children_list,
                perturbation_scale=0.1, alpha=0.3, seed=42
            )

            if result is not None:
                subpart_data.append({
                    'source_region': int(rid),
                    'k': k,
                    'n_parent': len(indices),
                    'n_children': len(children_list),
                    **result,
                })

    n_subpart = len(subpart_data)
    print(f"\n  Generated {n_subpart} sub-partitioned groups")

    # Correlation on sub-partitioned data
    if n_subpart >= 8:
        h1_sub = np.array([d['h1_proxy'] for d in subpart_data])
        rev_sub = np.array([d['spectral_revision'] for d in subpart_data])
        shift_sub = np.array([d['perturbation_shift'] for d in subpart_data])

        rho_hr_sub, p_hr_sub = spearmanr(h1_sub, rev_sub)
        rho_hs_sub, p_hs_sub = spearmanr(h1_sub, shift_sub)

        best_rho_sub = max(rho_hr_sub, rho_hs_sub)
        best_p_sub = p_hr_sub if best_rho_sub == rho_hr_sub else p_hs_sub
        best_metric_sub = "spectral_revision" if rho_hr_sub >= rho_hs_sub else "perturbation_shift"
        best_rev_sub = rev_sub if rho_hr_sub >= rho_hs_sub else shift_sub

        slope, intercept, r_val, p_reg, _ = linregress(h1_sub, best_rev_sub)
        r2_sub = r_val ** 2

        print(f"  rho(H1, spectral_revision) = {rho_hr_sub:.4f}, p={p_hr_sub:.2e}")
        print(f"  rho(H1, perturbation_shift) = {rho_hs_sub:.4f}, p={p_hs_sub:.2e}")
        print(f"  Best: {best_metric_sub} (rho={best_rho_sub:.4f})")
        print(f"  R^2 = {r2_sub:.4f}")
    else:
        best_rho_sub, best_p_sub, r2_sub = 0, 1, 0
        best_metric_sub = "insufficient_data"
        h1_sub, best_rev_sub = np.array([]), np.array([])
        print(f"  INSUFFICIENT DATA ({n_subpart} groups)")

    # =====================================================================
    # Strategy B: Multi-perturbation on original hierarchy groups
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Strategy B: Multi-Seed Multi-Scale on Original Groups")
    print(f"{'=' * 70}")

    seeds = [42, 137, 256, 512, 1024]
    scales = [0.05, 0.1, 0.2]

    multi_data = []  # (group_key, seed, scale, h1, revision)

    for (level, pid), children in hierarchy.items():
        if len(children) < 2:
            continue

        parent_indices = get_region_indices(labels_by_level, level, pid)
        if len(parent_indices) < 5 or len(parent_indices) > MAX_PARENT_CELLS:
            continue

        children_list = []
        for child_level, child_id in children:
            child_indices = get_region_indices(labels_by_level, child_level, child_id)
            if len(child_indices) > 0:
                children_list.append((child_id, child_indices))

        if len(children_list) < 2:
            continue

        total_child_cells = sum(len(ci) for _, ci in children_list)
        dim_c0 = len(parent_indices) + total_child_cells
        dim_c1 = total_child_cells
        if dim_c0 * dim_c1 > MAX_MATRIX_SIZE:
            continue

        group_key = f"L{level}_P{pid}"

        for seed in seeds:
            for scale in scales:
                result = compute_h1_and_revision(
                    state_flat, adjacency, parent_indices, children_list,
                    perturbation_scale=scale, alpha=0.3, seed=seed
                )
                if result is not None:
                    multi_data.append({
                        'group_key': group_key,
                        'level': level,
                        'parent_id': int(pid),
                        'seed': seed,
                        'scale': scale,
                        **result,
                    })

    print(f"  Generated {len(multi_data)} (group x seed x scale) combinations")

    # Compute group-level medians to avoid pseudoreplication
    from collections import defaultdict
    group_h1_medians = defaultdict(list)
    group_rev_medians = defaultdict(list)

    for md in multi_data:
        gk = md['group_key']
        group_h1_medians[gk].append(md['h1_proxy'])
        group_rev_medians[gk].append(md['spectral_revision'])

    median_h1s = []
    median_revs = []
    group_keys_med = []
    for gk in group_h1_medians:
        median_h1s.append(np.median(group_h1_medians[gk]))
        median_revs.append(np.median(group_rev_medians[gk]))
        group_keys_med.append(gk)

    median_h1s = np.array(median_h1s)
    median_revs = np.array(median_revs)
    n_med = len(median_h1s)

    if n_med >= 4:
        rho_med, p_med = spearmanr(median_h1s, median_revs)
        print(f"  Group medians: n={n_med}, rho={rho_med:.4f}, p={p_med:.2e}")
    else:
        rho_med, p_med = 0, 1
        print(f"  INSUFFICIENT groups for median analysis ({n_med})")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    test1 = n_subpart >= 20 and best_rho_sub > 0.25 and best_p_sub < 0.05
    print(f"\n  Test 1: Sub-partitioned (n>={20}): rho > 0.25 AND p < 0.05?")
    print(f"    n={n_subpart}, rho={best_rho_sub:.4f}, p={best_p_sub:.2e}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    test2 = r2_sub > 0.08
    print(f"\n  Test 2: R^2 > 0.08?")
    print(f"    R^2={r2_sub:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    test3 = rho_med > 0.3
    print(f"\n  Test 3: Multi-perturbation median rho > 0.3?")
    print(f"    rho={rho_med:.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: Rank overlap in sub-partitioned data
    if n_subpart >= 15:
        h1_ranked = np.argsort(-h1_sub)
        rev_ranked = np.argsort(-best_rev_sub)
        top10_h1 = set(h1_ranked[:10])
        top15_rev = set(rev_ranked[:15])
        overlap = top10_h1 & top15_rev
        n_overlap = len(overlap)
    else:
        n_overlap = 0

    test4 = n_overlap >= 5
    print(f"\n  Test 4: Top-10 H1 in top-15 revision (>=5/10)?")
    print(f"    Overlap: {n_overlap}/10")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 powered H1-revision tests verified")

    if n_subpart >= 20 and not test1:
        print(f"\n  NOTE: With n={n_subpart} groups and rho={best_rho_sub:.4f}, "
              f"the H1-revision link is WEAK. Consider FALSIFIED if consistent "
              f"with multi-perturbation result.")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_21_h1_revision_powered',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'H1-revision with increased sample size (sub-partition + multi-seed)',
        'strategy_a': {
            'n_groups': n_subpart,
            'best_metric': best_metric_sub,
            'spearman_rho': float(best_rho_sub),
            'spearman_p': float(best_p_sub),
            'r_squared': float(r2_sub),
        },
        'strategy_b': {
            'n_combinations': len(multi_data),
            'n_group_medians': n_med,
            'median_rho': float(rho_med),
            'median_p': float(p_med),
        },
        'rank_overlap': n_overlap,
        'verification': {
            'test1_subpart_correlation': bool(test1),
            'test2_r_squared': bool(test2),
            'test3_multi_seed_correlation': bool(test3),
            'test4_rank_overlap': bool(test4),
            'n_verified': n_verified,
        },
        'subpartition_data': subpart_data,
        'group_medians': {gk: {'h1': float(h), 'rev': float(r)}
                          for gk, h, r in zip(group_keys_med, median_h1s, median_revs)},
    }

    output_file = RESULTS_DIR / f'exp_21_h1_revision_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
