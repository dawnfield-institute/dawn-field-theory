"""
exp_36_eigenstructure_ceiling.py -- Confluent Identity Phase 27

PURPOSE:
    Derive the rho=0.42 coupling ceiling from the geometric relationship between
    gradient and state fields projected through the parent's eigenbasis. The ceiling
    should follow from how well |grad(C)| approximates C within each child's cells,
    filtered through the parent's spectral modes.

    Also extends the Delta buffer (exp_35) to ALL children (n~57 vs n=8),
    properly powering the frame-mismatch test.

MATHEMATICAL FRAMEWORK:
    coupling_weight(c) ~ ||[sum_{cell in c} v_i[cell] * |grad_C[cell]|]_{i=0..K}||
    natural_weight(c)  ~ ||[sum_{cell in c} v_i[cell] * state[cell]]_{i=0..K}||

    The Spearman correlation between these is bounded by the geometric alignment
    of gradient and state fields within the K_MODES eigenbasis.

VERIFICATION (4 tests, predict 2/4):
    1. Eigenbasis-projected alignment within [0.29, 0.55] of ceiling  (PREDICT PASS)
    2. Delta_self predicts coupling residuals at |rho|>0.25, p<0.05   (PREDICT PASS)
    3. Raw field alignment matches ceiling within +/-0.10              (PREDICT FAIL)
    4. Conservation: >=10% variance reduction with Delta               (PREDICT FAIL)

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_distance

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, K_MODES, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_parent_children_data, compute_subgraph_laplacian_from_field,
)
from exp_08_gradient_coupling import (
    compute_coupling_weights_weighted, compute_natural_weights,
    compute_gradient_field,
)
from exp_14_partial_correlation import partial_spearman


# ── Helpers ──────────────────────────────────────────────────────────

def compute_raw_field_alignment(grad_flat, state_flat, child_indices, parent_indices):
    """
    Cosine similarity between |grad_C| and C restricted to a child's cells
    within the parent's coordinate system.
    """
    parent_pos_map = {int(idx): pos for pos, idx in enumerate(parent_indices)}
    local_positions = [parent_pos_map[int(g)] for g in child_indices
                       if int(g) in parent_pos_map]
    if len(local_positions) < 2:
        return np.nan

    grad_child = grad_flat[child_indices]
    state_child = state_flat[child_indices]

    norm_g = np.linalg.norm(grad_child)
    norm_s = np.linalg.norm(state_child)
    if norm_g < 1e-15 or norm_s < 1e-15:
        return np.nan

    return float(np.dot(grad_child, state_child) / (norm_g * norm_s))


def compute_eigenbasis_alignment(grad_flat, state_flat, child_indices,
                                  parent_indices, eigvecs_parent):
    """
    Project both |grad_C| and C through parent's eigenvectors for a child,
    then compute cosine similarity of the resulting K-vectors.

    grad_proj(c) = [sum_{cell in c} v_i[cell] * |grad[cell]|]_{i=0..K}
    state_proj(c) = [sum_{cell in c} v_i[cell] * state[cell]]_{i=0..K}
    """
    parent_pos_map = {int(idx): pos for pos, idx in enumerate(parent_indices)}
    local_positions = np.array([parent_pos_map[int(g)] for g in child_indices
                                 if int(g) in parent_pos_map])
    if len(local_positions) < 2:
        return np.nan

    state_parent = state_flat[parent_indices]
    state_centered = state_parent - np.mean(state_parent)

    grad_parent = grad_flat[parent_indices]
    grad_centered = grad_parent - np.mean(grad_parent)

    child_state = state_centered[local_positions]
    child_grad = grad_centered[local_positions]
    child_eigvec = eigvecs_parent[local_positions, :]

    # Project through eigenbasis
    state_proj = child_state @ child_eigvec   # shape: (K,)
    grad_proj = child_grad @ child_eigvec     # shape: (K,)

    norm_s = np.linalg.norm(state_proj)
    norm_g = np.linalg.norm(grad_proj)
    if norm_s < 1e-15 or norm_g < 1e-15:
        return np.nan

    return float(np.dot(state_proj, grad_proj) / (norm_s * norm_g))


def compute_delta_self(C_flat, child_indices, parent_indices, eigvecs_parent,
                        state_flat, N):
    """
    Frame mismatch for ANY child (not just dual-role nodes).

    Internal view: child's own spectral entropy from its subgraph Laplacian.
    External view: child's projection norm in parent's eigenbasis.

    Delta_self = |internal_entropy - log(external_norm + 1)|
    """
    # Internal: child's own spectral entropy
    if len(child_indices) < 4:
        return np.nan, np.nan, np.nan

    L_child, _ = compute_subgraph_laplacian_from_field(C_flat, child_indices, N)
    state_child = state_flat[child_indices]
    identity_child = compute_spectral_identity(L_child, state_child)
    internal_entropy = identity_child['spectral_entropy']

    # External: projection norm in parent's eigenbasis
    parent_pos_map = {int(idx): pos for pos, idx in enumerate(parent_indices)}
    local_positions = np.array([parent_pos_map[int(g)] for g in child_indices
                                 if int(g) in parent_pos_map])
    if len(local_positions) < 2:
        return np.nan, np.nan, np.nan

    state_parent = state_flat[parent_indices]
    state_centered = state_parent - np.mean(state_parent)
    child_state = state_centered[local_positions]
    child_eigvec = eigvecs_parent[local_positions, :]
    contrib = child_state @ child_eigvec
    external_norm = float(np.linalg.norm(contrib))

    # Delta: mismatch between internal complexity and external appearance
    delta_self = abs(internal_entropy - np.log(external_norm + 1.0))

    return delta_self, internal_entropy, external_norm


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXP 36: EIGENSTRUCTURE CEILING DERIVATION")
    print("Phase 27 — Confluent Identity")
    print("=" * 70)

    # Load baseline
    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    C_flat = C.ravel()
    state_flat = C_flat.copy()

    # Build adjacency and gradient field
    print("\nBuilding adjacency and gradient field...")
    adjacency = build_lattice_adjacency(C)
    grad_C = compute_gradient_field(C)
    grad_flat = grad_C.ravel()

    # ── Collect per-child data ───────────────────────────────────────
    print("\nIterating over parent-children groups...")

    all_raw_alignments = []
    all_eigenbasis_alignments = []
    all_delta_self = []
    all_coupling_weights = []
    all_natural_weights = []
    all_sizes = []
    all_internal_entropy = []
    all_external_norm = []

    parent_groups = []  # for within-parent conservation test

    for (level, pid), parent_indices, children_list, L_parent, state_parent in \
            get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):

        # Compute spectral identity and eigenvectors for parent
        identity = compute_spectral_identity(L_parent, state_parent)
        if 'eigenvectors' not in identity:
            continue
        eigvecs = identity['eigenvectors']

        # Compute coupling weights (gradient-weighted)
        coupling = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, grad_flat
        )

        # Compute natural weights
        natural, size_fracs = compute_natural_weights(
            state_flat, parent_indices, children_list, eigvecs
        )

        group_data = {
            'coupling': [], 'natural': [], 'delta': [], 'size': [],
        }

        for child_id, child_indices in children_list:
            if child_id not in coupling or child_id not in natural:
                continue

            # Step 1: Raw field alignment
            raw_align = compute_raw_field_alignment(
                grad_flat, state_flat, child_indices, parent_indices
            )
            if not np.isnan(raw_align):
                all_raw_alignments.append(raw_align)

            # Step 2: Eigenbasis-projected alignment
            eig_align = compute_eigenbasis_alignment(
                grad_flat, state_flat, child_indices,
                parent_indices, eigvecs
            )
            if not np.isnan(eig_align):
                all_eigenbasis_alignments.append(eig_align)

            # Step 3: Delta_self
            delta, int_ent, ext_norm = compute_delta_self(
                C_flat, child_indices, parent_indices, eigvecs, state_flat, N
            )

            all_coupling_weights.append(coupling[child_id])
            all_natural_weights.append(natural[child_id])
            all_sizes.append(len(child_indices))

            if not np.isnan(delta):
                all_delta_self.append(delta)
                all_internal_entropy.append(int_ent)
                all_external_norm.append(ext_norm)
                group_data['coupling'].append(coupling[child_id])
                group_data['natural'].append(natural[child_id])
                group_data['delta'].append(delta)
                group_data['size'].append(len(child_indices))

        if len(group_data['coupling']) >= 3:
            parent_groups.append(group_data)

        print(f"  Parent ({level},{pid}): {len(children_list)} children processed")

    # Convert to arrays
    coupling_arr = np.array(all_coupling_weights)
    natural_arr = np.array(all_natural_weights)
    size_arr = np.array(all_sizes)
    delta_arr = np.array(all_delta_self)

    n_total = len(coupling_arr)
    n_delta = len(delta_arr)
    print(f"\nTotal children: {n_total}")
    print(f"Children with Delta_self: {n_delta}")

    # ── Step 1: Raw Field Alignment ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: RAW FIELD ALIGNMENT (|grad_C| vs C in cell space)")
    print("=" * 60)

    raw_alignments = np.array(all_raw_alignments)
    mean_raw = float(np.mean(raw_alignments))
    std_raw = float(np.std(raw_alignments))
    median_raw = float(np.median(raw_alignments))

    print(f"  Mean cosine(|grad_C|, C):  {mean_raw:.4f} +/- {std_raw:.4f}")
    print(f"  Median:                     {median_raw:.4f}")
    print(f"  Range:                      [{np.min(raw_alignments):.4f}, {np.max(raw_alignments):.4f}]")
    print(f"  Distance from ceiling 0.42: {abs(mean_raw - 0.42):.4f}")

    # ── Step 2: Eigenbasis-Projected Alignment ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: EIGENBASIS-PROJECTED ALIGNMENT (the key test)")
    print("=" * 60)

    eig_alignments = np.array(all_eigenbasis_alignments)
    mean_eig = float(np.mean(eig_alignments))
    std_eig = float(np.std(eig_alignments))
    median_eig = float(np.median(eig_alignments))

    print(f"  Mean cos(grad_proj, state_proj): {mean_eig:.4f} +/- {std_eig:.4f}")
    print(f"  Median:                           {median_eig:.4f}")
    print(f"  Range:                            [{np.min(eig_alignments):.4f}, {np.max(eig_alignments):.4f}]")
    print(f"  Distance from ceiling 0.42:       {abs(mean_eig - 0.42):.4f}")

    # ── Step 3: Delta_self for All Children ──────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: DELTA_SELF FOR ALL CHILDREN")
    print("=" * 60)

    print(f"  n = {n_delta} children with Delta_self")
    print(f"  Delta_self: mean={np.mean(delta_arr):.4f}, std={np.std(delta_arr):.4f}")
    print(f"  Range: [{np.min(delta_arr):.4f}, {np.max(delta_arr):.4f}]")

    # Coupling residuals: residualize coupling on natural+size
    # Use the subset that has delta values
    # We need aligned arrays — delta_arr aligns with the children that had valid delta
    # Reconstruct aligned coupling/natural/size arrays from group_data
    aligned_coupling = []
    aligned_natural = []
    aligned_size = []
    aligned_delta = []
    for group in parent_groups:
        aligned_coupling.extend(group['coupling'])
        aligned_natural.extend(group['natural'])
        aligned_size.extend(group['size'])
        aligned_delta.extend(group['delta'])

    aligned_coupling = np.array(aligned_coupling)
    aligned_natural = np.array(aligned_natural)
    aligned_size = np.array(aligned_size, dtype=float)
    aligned_delta = np.array(aligned_delta)

    n_aligned = len(aligned_coupling)
    print(f"  Aligned data points: {n_aligned}")

    # Compute coupling residuals (coupling with natural+size removed)
    if n_aligned >= 10:
        from numpy.polynomial.polynomial import polyfit, polyval
        # Rank-based residuals
        rank_coupling = np.argsort(np.argsort(aligned_coupling)).astype(float)
        rank_natural = np.argsort(np.argsort(aligned_natural)).astype(float)
        rank_size = np.argsort(np.argsort(aligned_size)).astype(float)

        # Residualize coupling on natural + size
        X = np.column_stack([rank_natural, rank_size])
        coeffs_fit = np.linalg.lstsq(
            np.column_stack([X, np.ones(n_aligned)]),
            rank_coupling, rcond=None
        )[0]
        coupling_predicted = X @ coeffs_fit[:2] + coeffs_fit[2]
        coupling_residual = rank_coupling - coupling_predicted

        rho_delta_resid, p_delta_resid = spearmanr(aligned_delta, coupling_residual)
        print(f"\n  Spearman rho(Delta_self, coupling_residual): {rho_delta_resid:.4f}")
        print(f"  p-value: {p_delta_resid:.6f}")

        # Also direct correlation
        rho_delta_coupling, p_delta_coupling = spearmanr(aligned_delta, aligned_coupling)
        print(f"  Spearman rho(Delta_self, coupling): {rho_delta_coupling:.4f} (p={p_delta_coupling:.6f})")

        # Partial correlation controlling for size
        if n_aligned >= 10:
            partial_rho, partial_p = partial_spearman(aligned_delta, aligned_coupling, aligned_size)
            print(f"  Partial rho(Delta_self, coupling | size): {partial_rho:.4f} (p={partial_p:.6f})")
    else:
        rho_delta_resid = np.nan
        p_delta_resid = 1.0
        print("  Insufficient data for residual analysis")

    # ── Step 4: Conservation Test ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: CONSERVATION TEST (P + A + Delta = C)")
    print("=" * 60)

    if n_aligned >= 10:
        # Grid search for optimal alpha
        base_residual = aligned_coupling - aligned_natural
        base_var = np.var(base_residual)

        alphas = np.linspace(-2.0, 2.0, 41)
        best_alpha = 0.0
        best_var = base_var
        best_reduction = 0.0

        for alpha in alphas:
            augmented_residual = aligned_coupling - (aligned_natural + alpha * aligned_delta)
            aug_var = np.var(augmented_residual)
            reduction = 1.0 - aug_var / base_var if base_var > 0 else 0.0
            if aug_var < best_var:
                best_var = aug_var
                best_alpha = alpha
                best_reduction = reduction

        print(f"  Baseline var(coupling - natural): {base_var:.6f}")
        print(f"  Best alpha: {best_alpha:.3f}")
        print(f"  Augmented var(coupling - natural - alpha*Delta): {best_var:.6f}")
        print(f"  Variance reduction: {best_reduction:.4f} ({best_reduction*100:.1f}%)")

        # Within-parent conservation (more precise)
        print("\n  Within-parent conservation:")
        within_reductions = []
        for i, group in enumerate(parent_groups):
            if len(group['coupling']) < 3:
                continue
            g_coup = np.array(group['coupling'])
            g_nat = np.array(group['natural'])
            g_delta = np.array(group['delta'])
            g_base_var = np.var(g_coup - g_nat)
            if g_base_var < 1e-15:
                continue
            # Find best alpha for this parent
            best_g_red = 0.0
            for alpha in alphas:
                g_aug_var = np.var(g_coup - (g_nat + alpha * g_delta))
                g_red = 1.0 - g_aug_var / g_base_var
                if g_red > best_g_red:
                    best_g_red = g_red
            within_reductions.append(best_g_red)
            print(f"    Parent {i}: {len(group['coupling'])} children, "
                  f"best reduction = {best_g_red:.4f} ({best_g_red*100:.1f}%)")

        if within_reductions:
            mean_within_red = np.mean(within_reductions)
            print(f"  Mean within-parent reduction: {mean_within_red:.4f} ({mean_within_red*100:.1f}%)")
    else:
        best_reduction = 0.0
        best_alpha = 0.0
        print("  Insufficient data for conservation test")

    # ── Reference: Actual coupling ceiling ───────────────────────────
    print("\n" + "=" * 60)
    print("REFERENCE: ACTUAL COUPLING CORRELATION")
    print("=" * 60)

    if n_total >= 10:
        rho_actual, p_actual = spearmanr(coupling_arr, natural_arr)
        print(f"  Spearman rho(coupling, natural): {rho_actual:.4f} (p={p_actual:.6f})")
        if n_total >= 10:
            partial_rho_ref, partial_p_ref = partial_spearman(
                coupling_arr, natural_arr, size_arr.astype(float)
            )
            print(f"  Partial rho(coupling, natural | size): {partial_rho_ref:.4f} (p={partial_p_ref:.6f})")
    else:
        rho_actual = np.nan
        partial_rho_ref = np.nan

    # ── Verification ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    results = {}

    # Test 1: Eigenbasis alignment ~ ceiling
    test1_pass = abs(mean_eig - 0.42) < 0.13
    results['test1_eigenbasis_alignment'] = {
        'mean_cos_eigenbasis': mean_eig,
        'distance_from_ceiling': abs(mean_eig - 0.42),
        'threshold': 0.13,
        'verified': test1_pass,
    }
    status1 = "VERIFIED" if test1_pass else "NOT VERIFIED"
    print(f"\n  Test 1: Eigenbasis alignment ~ ceiling (|{mean_eig:.4f} - 0.42| < 0.13)")
    print(f"    Distance: {abs(mean_eig - 0.42):.4f}")
    print(f"    -> {status1}")

    # Test 2: Delta_self predicts coupling residuals
    test2_pass = (abs(rho_delta_resid) > 0.25 and p_delta_resid < 0.05) if not np.isnan(rho_delta_resid) else False
    results['test2_delta_predicts_residuals'] = {
        'rho_delta_residual': float(rho_delta_resid) if not np.isnan(rho_delta_resid) else None,
        'p_value': float(p_delta_resid) if not np.isnan(p_delta_resid) else None,
        'n': n_aligned,
        'verified': test2_pass,
    }
    status2 = "VERIFIED" if test2_pass else "NOT VERIFIED"
    print(f"\n  Test 2: Delta_self predicts coupling residuals (|rho|>0.25, p<0.05)")
    print(f"    rho = {rho_delta_resid:.4f}, p = {p_delta_resid:.6f}, n = {n_aligned}")
    print(f"    -> {status2}")

    # Test 3: Raw alignment matches ceiling
    test3_pass = abs(mean_raw - 0.42) < 0.10
    results['test3_raw_alignment'] = {
        'mean_cos_raw': mean_raw,
        'distance_from_ceiling': abs(mean_raw - 0.42),
        'threshold': 0.10,
        'verified': test3_pass,
    }
    status3 = "VERIFIED" if test3_pass else "NOT VERIFIED"
    print(f"\n  Test 3: Raw alignment matches ceiling (|{mean_raw:.4f} - 0.42| < 0.10)")
    print(f"    Distance: {abs(mean_raw - 0.42):.4f}")
    print(f"    -> {status3}")

    # Test 4: Conservation variance reduction >= 10%
    test4_pass = best_reduction >= 0.10
    results['test4_conservation'] = {
        'best_alpha': float(best_alpha),
        'variance_reduction': float(best_reduction),
        'threshold': 0.10,
        'verified': test4_pass,
    }
    status4 = "VERIFIED" if test4_pass else "NOT VERIFIED"
    print(f"\n  Test 4: Conservation >=10% variance reduction ({best_reduction:.4f})")
    print(f"    Best alpha: {best_alpha:.3f}")
    print(f"    -> {status4}")

    n_verified = sum([test1_pass, test2_pass, test3_pass, test4_pass])
    print(f"\n  TOTAL: {n_verified}/4 verified")

    # ── Save results ─────────────────────────────────────────────────
    results.update({
        'experiment': 'exp_36_eigenstructure_ceiling',
        'phase': 27,
        'n_total_children': n_total,
        'n_delta_children': n_delta,
        'n_aligned': n_aligned,
        'raw_alignment': {
            'mean': mean_raw, 'std': std_raw, 'median': median_raw,
        },
        'eigenbasis_alignment': {
            'mean': mean_eig, 'std': std_eig, 'median': median_eig,
        },
        'delta_self': {
            'mean': float(np.mean(delta_arr)),
            'std': float(np.std(delta_arr)),
            'rho_with_coupling_residual': float(rho_delta_resid) if not np.isnan(rho_delta_resid) else None,
            'p_value': float(p_delta_resid) if not np.isnan(p_delta_resid) else None,
        },
        'conservation': {
            'best_alpha': float(best_alpha),
            'variance_reduction': float(best_reduction),
        },
        'reference_coupling': {
            'rho': float(rho_actual) if not np.isnan(rho_actual) else None,
            'partial_rho': float(partial_rho_ref) if not np.isnan(partial_rho_ref) else None,
        },
        'verified_count': n_verified,
        'timestamp': datetime.now().isoformat(),
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'exp_36_eigenstructure_ceiling_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
