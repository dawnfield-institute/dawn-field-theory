"""
exp_35_frame_relative_identity.py -- Confluent Identity Phase 26

PURPOSE:
    Measure the Delta buffer explicitly. Every parent is also a child.
    The mismatch between a node's identity-as-parent (looking down at its
    children) and its weight-as-child (looking up at its parent) is the
    frame shift -- analogous to the Delta buffer in asymmetric conservation
    (P + A + Delta = C).

    The coupling ceiling at 0.42 may be the frame-local portion of coupling.
    The remaining variance lives in the cross-frame Delta.

    Connects three corpus FDOs:
    - asymmetric-conservation: P + A + Delta = C (bounded reconciliation buffer)
    - observation-dependency-pac: O(S) -> D(S,O) -> A(S|O) (circular dependency)
    - confluence-operator: non-commutative, path-dependent actualization

METHODS:
    1. Find all dual-role nodes (both parent AND child in hierarchy)
    2. For each: compute identity-as-parent (spectral identity of sub-region)
       and weight-as-child (coupling + natural weight in parent)
    3. Define 4 Delta measures: spectral, fiedler, entropy, weight mismatch
    4. Test Delta as predictor of coupling residual
    5. Test frame covariance: Delta scales with hierarchy depth?
    6. Within-parent Delta structure (parent (2,1) with 20 children)

VERIFICATION:
    - Delta is bounded: max(|Delta_spectral|) < 1.0, std < mean
    - Delta adds predictive power: R^2 gain > 0.05
    - Delta scales with depth: |rho(Delta, depth)| > 0.20
    - Frame-augmented coupling > 0.42 baseline by at least 0.03

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import cosine as cosine_distance

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    compute_spectral_identity, get_parent_children_data,
)
from exp_08_gradient_coupling import (
    compute_coupling_weights_weighted, compute_natural_weights,
    compute_gradient_field,
)
from exp_14_partial_correlation import partial_spearman


def find_dual_role_nodes(labels_by_level, hierarchy):
    """
    Find nodes that appear as both parent (have children) and child
    (appear in some other parent's children list).

    Returns list of dicts:
      {level, rid, parent_level, parent_id, children: [(child_level, child_id), ...]}
    """
    # Build child->parent lookup
    child_to_parent = {}
    for (plevel, pid), children in hierarchy.items():
        for (clevel, cid) in children:
            child_to_parent[(clevel, cid)] = (plevel, pid)

    # Find nodes that are BOTH a parent key AND a child
    parent_keys = set(hierarchy.keys())
    dual_nodes = []

    for (level, rid) in parent_keys:
        children = hierarchy[(level, rid)]
        if len(children) < 2:
            continue  # need at least 2 children to be a meaningful parent

        # Is this node also a child of something?
        if (level, rid) in child_to_parent:
            plevel, pid = child_to_parent[(level, rid)]
            dual_nodes.append({
                'level': level,
                'rid': rid,
                'parent_level': plevel,
                'parent_id': pid,
                'n_children': len(children),
            })

    return dual_nodes


def compute_child_projection(state_flat, child_indices, parent_indices, eigvecs_parent):
    """
    Compute the projection of a child's state into the parent's eigenbasis.

    Returns the projection coefficients (how the child's state maps onto
    parent eigenvectors) -- this is the child's identity as seen from
    the parent's frame.
    """
    # Map child indices to positions within parent
    parent_set = set(parent_indices.tolist())
    idx_map = {idx: pos for pos, idx in enumerate(parent_indices)}

    child_positions = []
    for ci in child_indices:
        if ci in idx_map:
            child_positions.append(idx_map[ci])

    if len(child_positions) == 0:
        return np.zeros(eigvecs_parent.shape[1])

    child_positions = np.array(child_positions)

    # Child's state within parent frame
    state_parent = state_flat[parent_indices]
    state_centered = state_parent - np.mean(state_parent)

    # Child's contribution to each parent eigenvector
    child_mask = np.zeros(len(parent_indices))
    child_mask[child_positions] = 1.0

    # Project: for each mode k, child's contribution = sum of
    # (centered state * eigenvector) over child cells only
    n_modes = eigvecs_parent.shape[1]
    projection = np.zeros(n_modes)
    for k in range(n_modes):
        projection[k] = np.sum(state_centered[child_positions] * eigvecs_parent[child_positions, k])

    return projection


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 26, Experiment 35")
    print("Frame-Relative Identity: The Delta Buffer")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency and gradient field...")
    adjacency = build_lattice_adjacency(C)
    grad_mag = compute_gradient_field(C)
    grad_flat = grad_mag.ravel()

    # =====================================================================
    # Step 1: Find dual-role nodes
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Step 1: Finding Dual-Role Nodes (Both Parent AND Child)")
    print(f"{'=' * 70}")

    dual_nodes = find_dual_role_nodes(labels_by_level, hierarchy)
    print(f"  Found {len(dual_nodes)} dual-role nodes:")
    for dn in dual_nodes:
        print(f"    Level {dn['level']}, Region {dn['rid']}: "
              f"parent of {dn['n_children']} children, "
              f"child of ({dn['parent_level']},{dn['parent_id']})")

    # =====================================================================
    # Step 2: Compute dual identity for each node
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Step 2: Computing Dual Identities")
    print(f"{'=' * 70}")

    # First, collect all parent-children coupling data (we need this for
    # the "weight-as-child" computation)
    # Build lookup: (level, rid) -> coupling data when this node is a CHILD
    child_coupling_data = {}  # (level, rid) -> {coupling, natural, size}

    for (level, pid), parent_indices, children_list, L_parent, state_parent in \
            get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):

        identity_parent = compute_spectral_identity(L_parent, state_parent)
        eigvecs_parent = identity_parent.get('eigenvectors')
        if eigvecs_parent is None:
            continue

        natural_weights, size_fractions = compute_natural_weights(
            state_flat, parent_indices, children_list, eigvecs_parent
        )

        w_gradient = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, grad_flat
        )

        # Store coupling data for each child
        for child_id, child_indices in children_list:
            child_level = level - 1  # children are one level below parent
            child_coupling_data[(child_level, child_id)] = {
                'coupling': w_gradient.get(child_id, 0),
                'natural': natural_weights.get(child_id, 0),
                'size_frac': size_fractions.get(child_id, 0),
                'parent_level': level,
                'parent_id': pid,
                'parent_indices': parent_indices,
                'parent_eigvecs': eigvecs_parent,
                'parent_identity': identity_parent,
                'child_indices': child_indices,
            }

    # Now compute dual identity for each dual-role node
    dual_data = []

    for dn in dual_nodes:
        level, rid = dn['level'], dn['rid']

        # --- Identity as parent (looking down) ---
        labels = labels_by_level[level]
        node_indices = np.where(labels.ravel() == rid)[0]

        if len(node_indices) < 3:
            continue

        # Build subgraph Laplacian for this node's region
        from _shared import graph_laplacian_subgraph
        L_node, _ = graph_laplacian_subgraph(adjacency, node_indices)
        state_node = state_flat[node_indices]
        identity_down = compute_spectral_identity(L_node, state_node)

        coeffs_down = np.array(identity_down['state_coefficients'])
        fiedler_down = identity_down['fiedler_value']
        entropy_down = identity_down['spectral_entropy']

        # --- Weight as child (looking up) ---
        child_key = (level, rid)
        if child_key not in child_coupling_data:
            continue

        cd = child_coupling_data[child_key]
        coupling_up = cd['coupling']
        natural_up = cd['natural']
        size_frac = cd['size_frac']

        # Compute child's projection in parent's eigenbasis
        proj_in_parent = compute_child_projection(
            state_flat, node_indices, cd['parent_indices'], cd['parent_eigvecs']
        )

        # Parent's fiedler and entropy for reference
        parent_fiedler = cd['parent_identity']['fiedler_value']
        parent_entropy = cd['parent_identity']['spectral_entropy']

        # --- Compute Delta measures ---

        # Delta_spectral: cosine distance between own coefficients and
        # projection in parent eigenbasis
        if np.linalg.norm(coeffs_down) > 1e-15 and np.linalg.norm(proj_in_parent) > 1e-15:
            # Align lengths (may differ if different k_modes)
            min_len = min(len(coeffs_down), len(proj_in_parent))
            cd_trunc = coeffs_down[:min_len]
            pp_trunc = proj_in_parent[:min_len]
            if np.linalg.norm(cd_trunc) > 1e-15 and np.linalg.norm(pp_trunc) > 1e-15:
                delta_spectral = float(cosine_distance(cd_trunc, pp_trunc))
            else:
                delta_spectral = 1.0
        else:
            delta_spectral = 1.0

        # Delta_fiedler: gap between own coherence and parent's coherence
        delta_fiedler = abs(fiedler_down - parent_fiedler)

        # Delta_entropy: complexity mismatch
        delta_entropy = abs(entropy_down - parent_entropy)

        # Delta_weight: deviation from size-predicted contribution
        delta_weight = abs(natural_up - size_frac)

        dual_data.append({
            'level': level,
            'rid': rid,
            'n_cells': len(node_indices),
            'n_children': dn['n_children'],
            'parent_level': dn['parent_level'],
            'parent_id': dn['parent_id'],
            # Identity as parent
            'fiedler_down': float(fiedler_down),
            'entropy_down': float(entropy_down),
            'n_coeffs_down': len(coeffs_down),
            # Weight as child
            'coupling_up': float(coupling_up),
            'natural_up': float(natural_up),
            'size_frac': float(size_frac),
            # Delta measures
            'delta_spectral': float(delta_spectral),
            'delta_fiedler': float(delta_fiedler),
            'delta_entropy': float(delta_entropy),
            'delta_weight': float(delta_weight),
            # Parent reference
            'parent_fiedler': float(parent_fiedler),
            'parent_entropy': float(parent_entropy),
        })

        print(f"  L{level} R{rid} ({len(node_indices)} cells, {dn['n_children']} children):")
        print(f"    Down: fiedler={fiedler_down:.4f}, entropy={entropy_down:.4f}")
        print(f"    Up:   coupling={coupling_up:.4f}, natural={natural_up:.4f}, size={size_frac:.4f}")
        print(f"    Delta: spectral={delta_spectral:.4f}, fiedler={delta_fiedler:.4f}, "
              f"entropy={delta_entropy:.4f}, weight={delta_weight:.4f}")

    n_dual = len(dual_data)
    print(f"\n  Total dual-role nodes analyzed: {n_dual}")

    if n_dual < 3:
        print("\n  INSUFFICIENT DATA for correlation analysis. Saving raw data only.")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = {
            'experiment': 'exp_35_frame_relative_identity',
            'timestamp': datetime.now().isoformat(),
            'purpose': 'Frame-relative identity: Delta buffer between parent and child views',
            'n_dual_nodes': n_dual,
            'dual_data': dual_data,
            'verification': {'n_verified': 0, 'error': 'insufficient dual-role nodes'},
        }
        output_file = RESULTS_DIR / f'exp_35_frame_identity_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2,
                      default=lambda o: int(o) if hasattr(o, 'item') else o)
        print(f"  Results saved to: {output_file.name}")
        return output

    # =====================================================================
    # Step 3: Delta statistics and boundedness
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Step 3: Delta Buffer Statistics")
    print(f"{'=' * 70}")

    ds = np.array([d['delta_spectral'] for d in dual_data])
    df = np.array([d['delta_fiedler'] for d in dual_data])
    de = np.array([d['delta_entropy'] for d in dual_data])
    dw = np.array([d['delta_weight'] for d in dual_data])

    for name, arr in [('spectral', ds), ('fiedler', df), ('entropy', de), ('weight', dw)]:
        print(f"\n  Delta_{name}:")
        print(f"    mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, "
              f"min={np.min(arr):.4f}, max={np.max(arr):.4f}")

    # =====================================================================
    # Step 4: Delta as predictor of coupling
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Step 4: Delta as Predictor of Coupling")
    print(f"{'=' * 70}")

    coupling = np.array([d['coupling_up'] for d in dual_data])
    natural = np.array([d['natural_up'] for d in dual_data])
    sizes = np.array([d['size_frac'] for d in dual_data])

    # Baseline: coupling ~ natural
    if n_dual >= 5:
        rho_base, p_base = spearmanr(coupling, natural)
        print(f"\n  Baseline rho(coupling, natural) = {rho_base:.4f}, p={p_base:.2e}")

        if n_dual >= 8:
            pr_base, pp_base = partial_spearman(coupling, natural, sizes)
            print(f"  Baseline partial rho(| size) = {pr_base:.4f}, p={pp_base:.2e}")
        else:
            pr_base = rho_base
            pp_base = p_base

        # Correlate each Delta with coupling
        print(f"\n  Delta correlations with coupling:")
        delta_corrs = {}
        for name, arr in [('spectral', ds), ('fiedler', df), ('entropy', de), ('weight', dw)]:
            rho, p = spearmanr(arr, coupling)
            delta_corrs[name] = {'rho': float(rho), 'p': float(p)}
            sig = '*' if p < 0.05 else ''
            print(f"    rho(Delta_{name}, coupling) = {rho:.4f}, p={p:.2e} {sig}")

        # Multiple regression: coupling ~ natural + Delta_spectral + Delta_fiedler
        print(f"\n  Multiple Regression: coupling ~ natural + Delta_spectral + Delta_fiedler")
        X_base = np.column_stack([np.ones(n_dual), rankdata(natural)])
        X_augmented = np.column_stack([np.ones(n_dual), rankdata(natural),
                                        rankdata(ds), rankdata(df)])
        y = rankdata(coupling)

        # R^2 baseline
        beta_base = np.linalg.lstsq(X_base, y, rcond=None)[0]
        resid_base = y - X_base @ beta_base
        ss_res_base = np.sum(resid_base**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_base = 1 - ss_res_base / (ss_tot + 1e-15)

        # R^2 augmented
        beta_aug = np.linalg.lstsq(X_augmented, y, rcond=None)[0]
        resid_aug = y - X_augmented @ beta_aug
        ss_res_aug = np.sum(resid_aug**2)
        r2_aug = 1 - ss_res_aug / (ss_tot + 1e-15)

        r2_gain = r2_aug - r2_base
        print(f"    R^2 (natural only): {r2_base:.4f}")
        print(f"    R^2 (natural + Delta_spectral + Delta_fiedler): {r2_aug:.4f}")
        print(f"    R^2 gain: {r2_gain:.4f}")

        # Frame-augmented partial correlation
        # Residualize coupling and natural on [size, delta_spectral]
        if n_dual >= 8:
            confounders = np.column_stack([sizes, ds])
            X_conf = np.column_stack([np.ones(n_dual), rankdata(confounders[:, 0]),
                                       rankdata(confounders[:, 1])])
            # Residualize coupling on confounders
            rc = rankdata(coupling)
            beta_c = np.linalg.lstsq(X_conf, rc, rcond=None)[0]
            coupling_resid = rc - X_conf @ beta_c

            rn = rankdata(natural)
            beta_n = np.linalg.lstsq(X_conf, rn, rcond=None)[0]
            natural_resid = rn - X_conf @ beta_n

            if np.std(coupling_resid) > 1e-15 and np.std(natural_resid) > 1e-15:
                frame_aug_rho, frame_aug_p = spearmanr(coupling_resid, natural_resid)
            else:
                frame_aug_rho, frame_aug_p = 0.0, 1.0

            print(f"\n  Frame-augmented partial rho(coupling, natural | size, Delta_spectral):")
            print(f"    rho = {frame_aug_rho:.4f}, p={frame_aug_p:.2e}")
            print(f"    Baseline partial rho(| size only) = {pr_base:.4f}")
            print(f"    Improvement: {frame_aug_rho - pr_base:.4f}")
        else:
            frame_aug_rho = pr_base
            frame_aug_p = pp_base
    else:
        rho_base = 0
        pr_base = 0
        r2_base = 0
        r2_aug = 0
        r2_gain = 0
        frame_aug_rho = 0
        frame_aug_p = 1.0
        delta_corrs = {}

    # =====================================================================
    # Step 5: Frame covariance -- Delta vs hierarchy depth
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Step 5: Frame Covariance (Delta vs Hierarchy Depth)")
    print(f"{'=' * 70}")

    depths = np.array([d['level'] for d in dual_data], dtype=float)

    if len(np.unique(depths)) >= 2:
        for name, arr in [('spectral', ds), ('fiedler', df), ('entropy', de), ('weight', dw)]:
            rho, p = spearmanr(arr, depths)
            print(f"  rho(Delta_{name}, depth) = {rho:.4f}, p={p:.2e}")
    else:
        print("  Only one hierarchy depth represented -- cannot test covariance")

    # Per-level Delta summary
    print(f"\n  Per-level Delta_spectral:")
    for level in sorted(set(d['level'] for d in dual_data)):
        level_ds = [d['delta_spectral'] for d in dual_data if d['level'] == level]
        print(f"    Level {level} (n={len(level_ds)}): mean={np.mean(level_ds):.4f}, "
              f"std={np.std(level_ds):.4f}")

    # =====================================================================
    # Step 6: Within-parent Delta structure
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Step 6: Within-Parent Delta Structure")
    print(f"{'=' * 70}")

    # Find largest parent group
    parent_groups = {}
    for d in dual_data:
        key = (d['parent_level'], d['parent_id'])
        if key not in parent_groups:
            parent_groups[key] = []
        parent_groups[key].append(d)

    for (plevel, pid), group in sorted(parent_groups.items()):
        if len(group) < 3:
            continue

        print(f"\n  Parent ({plevel},{pid}): {len(group)} dual-role children")

        g_coupling = np.array([d['coupling_up'] for d in group])
        g_ds = np.array([d['delta_spectral'] for d in group])
        g_df = np.array([d['delta_fiedler'] for d in group])

        if len(group) >= 4:
            rho_ds_c, p_ds_c = spearmanr(g_ds, g_coupling)
            rho_df_c, p_df_c = spearmanr(g_df, g_coupling)
            print(f"    rho(Delta_spectral, coupling) = {rho_ds_c:.4f}, p={p_ds_c:.2e}")
            print(f"    rho(Delta_fiedler, coupling) = {rho_df_c:.4f}, p={p_df_c:.2e}")
        else:
            print(f"    (n={len(group)}, too few for within-parent correlation)")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Test 1: Delta is bounded
    test1 = (np.max(ds) < 1.0) and (np.std(ds) < np.mean(ds) + 1e-15)
    print(f"\n  Test 1: Delta_spectral bounded? (max < 1.0, std < mean)")
    print(f"    max={np.max(ds):.4f}, mean={np.mean(ds):.4f}, std={np.std(ds):.4f}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Delta adds predictive power (R^2 gain > 0.05)
    test2 = r2_gain > 0.05
    print(f"\n  Test 2: R^2 gain from Delta > 0.05?")
    print(f"    R^2 gain = {r2_gain:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: Delta scales with depth
    if len(np.unique(depths)) >= 2:
        rho_depth, p_depth = spearmanr(ds, depths)
        test3 = abs(rho_depth) > 0.20
        print(f"\n  Test 3: |rho(Delta_spectral, depth)| > 0.20?")
        print(f"    rho = {rho_depth:.4f}")
        print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")
    else:
        test3 = False
        rho_depth = 0.0
        print(f"\n  Test 3: Cannot test (single depth level)")
        print(f"    [FAILED]")

    # Test 4: Frame-augmented > baseline by 0.03
    improvement = frame_aug_rho - pr_base
    test4 = improvement > 0.03
    print(f"\n  Test 4: Frame-augmented partial rho > baseline + 0.03?")
    print(f"    Frame-augmented: {frame_aug_rho:.4f}, baseline: {pr_base:.4f}, "
          f"improvement: {improvement:.4f}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 frame-relative identity tests verified")

    # =====================================================================
    # Save
    # =====================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_35_frame_relative_identity',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Frame-relative identity: Delta buffer between parent and child views',
        'n_dual_nodes': n_dual,
        'delta_statistics': {
            'spectral': {'mean': float(np.mean(ds)), 'std': float(np.std(ds)),
                         'min': float(np.min(ds)), 'max': float(np.max(ds))},
            'fiedler': {'mean': float(np.mean(df)), 'std': float(np.std(df)),
                        'min': float(np.min(df)), 'max': float(np.max(df))},
            'entropy': {'mean': float(np.mean(de)), 'std': float(np.std(de)),
                        'min': float(np.min(de)), 'max': float(np.max(de))},
            'weight': {'mean': float(np.mean(dw)), 'std': float(np.std(dw)),
                       'min': float(np.min(dw)), 'max': float(np.max(dw))},
        },
        'coupling_analysis': {
            'baseline_rho': float(rho_base),
            'baseline_partial_rho': float(pr_base),
            'r2_base': float(r2_base),
            'r2_augmented': float(r2_aug),
            'r2_gain': float(r2_gain),
            'frame_augmented_rho': float(frame_aug_rho),
            'delta_correlations': delta_corrs,
        },
        'depth_scaling': {
            'rho_delta_depth': float(rho_depth) if len(np.unique(depths)) >= 2 else None,
        },
        'verification': {
            'test1_bounded': bool(test1),
            'test2_r2_gain': bool(test2),
            'test3_depth_scaling': bool(test3),
            'test4_frame_augmented': bool(test4),
            'n_verified': n_verified,
        },
        'dual_data': dual_data,
    }

    output_file = RESULTS_DIR / f'exp_35_frame_identity_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
