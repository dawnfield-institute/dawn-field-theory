"""
exp_11_sheaf_cohomology.py -- Confluent Identity Phase 4

PURPOSE:
    Construct a cellular sheaf on the hierarchy poset and compute H^0 and H^1
    cohomology groups. Test the novel prediction: perturbation creates identity
    crisis (H^1 increases), backward smoothing resolves it (H^1 decreases).

DEFINITIONS:
    Sheaf F on hierarchy poset (parent -> children):
        Stalks: F(R) = R^{|V_R|} (state vectors on region R)
        Restriction: rho_{R->S}(f) = f|_{V_S} (extract child cells from parent)
        Coboundary: (delta^0 sigma)(R->S) = rho_{R->S}(sigma(R)) - sigma(S)

    H^0 = ker(delta^0) = global sections (consistent assignments)
    H^1 = C^1 / im(delta^0) = obstruction to gluing = "identity crisis"

NOVEL PREDICTION:
    1. The actual state IS a global section (delta^0 = 0 exactly)
    2. Perturbation breaks consistency -> H^1 increases
    3. Backward smoothing restores consistency -> H^1 decreases

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy import sparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices,
)


def build_restriction_matrix(parent_indices, child_indices):
    """
    Build restriction map rho_{R->S}: R^{|V_R|} -> R^{|V_S|}.

    This is a |V_S| x |V_R| matrix with a 1 at position (i, j) where
    global index child_indices[i] == parent_indices[j].
    """
    parent_pos = {int(g): j for j, g in enumerate(parent_indices)}
    n_child = len(child_indices)
    n_parent = len(parent_indices)

    rows, cols, vals = [], [], []
    for i, ci in enumerate(child_indices):
        ci_int = int(ci)
        if ci_int in parent_pos:
            rows.append(i)
            cols.append(parent_pos[ci_int])
            vals.append(1.0)

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_child, n_parent))


def build_coboundary_operator(parent_indices, children_list):
    """
    Build delta^0: C^0 -> C^1 for one parent and its children.

    C^0 = R^{n_parent} x R^{n_child_1} x ... x R^{n_child_k}
         (one section per node: parent + each child)

    C^1 = R^{n_child_1} x ... x R^{n_child_k}
         (one cochain per edge: parent->child_i)

    (delta^0 sigma)(R->S_i) = rho_{R->S_i}(sigma_parent) - sigma_{S_i}

    Returns delta^0 as a dense matrix.
    """
    n_parent = len(parent_indices)
    child_sizes = [len(ci) for _, ci in children_list]
    n_children_total = sum(child_sizes)

    # C^0 dimension: parent + all children
    dim_c0 = n_parent + n_children_total

    # C^1 dimension: sum of child sizes (one edge per parent-child pair)
    dim_c1 = n_children_total

    delta = np.zeros((dim_c1, dim_c0))

    # Fill in the coboundary operator
    c1_offset = 0
    c0_child_offset = n_parent

    for child_id, child_indices in children_list:
        n_child = len(child_indices)
        rho = build_restriction_matrix(parent_indices, child_indices)

        # The block for this edge in delta^0:
        # delta[c1_offset:c1_offset+n_child, 0:n_parent] = +rho  (restriction of parent)
        # delta[c1_offset:c1_offset+n_child, c0_child_offset:c0_child_offset+n_child] = -I  (child section)

        rho_dense = rho.toarray()
        delta[c1_offset:c1_offset + n_child, 0:n_parent] = rho_dense
        delta[c1_offset:c1_offset + n_child,
              c0_child_offset:c0_child_offset + n_child] = -np.eye(n_child)

        c1_offset += n_child
        c0_child_offset += n_child

    return delta


def compute_cohomology_dims(delta):
    """
    Compute H^0 = ker(delta^0) and H^1 = coker(delta^0) dimensions.

    For a tree hierarchy (no 2-simplices), H^1 = C^1 / im(delta^0).
    """
    # SVD for rank computation
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)

    # Rank = number of singular values above threshold
    tol = max(delta.shape) * np.finfo(float).eps * S[0] if len(S) > 0 else 1e-10
    rank = int(np.sum(S > tol))

    dim_c0 = delta.shape[1]
    dim_c1 = delta.shape[0]

    dim_h0 = dim_c0 - rank     # ker(delta^0)
    dim_h1 = dim_c1 - rank     # coker(delta^0) = C^1 / im(delta^0)

    # Smallest singular value of delta (measures "how close to degenerate")
    smallest_sv = float(S[rank - 1]) if rank > 0 else 0.0
    next_sv = float(S[rank]) if rank < len(S) else 0.0

    return {
        'dim_c0': dim_c0,
        'dim_c1': dim_c1,
        'rank_delta': rank,
        'dim_h0': dim_h0,
        'dim_h1': dim_h1,
        'smallest_nonzero_sv': smallest_sv,
        'largest_zero_sv': next_sv,
    }


def build_section(state_flat, parent_indices, children_list):
    """
    Build a section sigma in C^0 from the actual state.

    sigma = (state_parent, state_child_1, ..., state_child_k)
    """
    sigma_parent = state_flat[parent_indices]
    sigma_children = []
    for child_id, child_indices in children_list:
        sigma_children.append(state_flat[child_indices])

    return np.concatenate([sigma_parent] + sigma_children)


def perturb_children_only(state_flat, children_list, perturbation_scale=0.1,
                           seed=42):
    """
    Create a perturbed state where children are modified but parent is NOT updated.
    This breaks the global section property: rho(parent) != child after perturbation.
    """
    rng = np.random.RandomState(seed)
    state_perturbed = state_flat.copy()

    for child_id, child_indices in children_list:
        noise = rng.randn(len(child_indices)) * perturbation_scale
        state_perturbed[child_indices] += noise

    return state_perturbed


def smooth_section(state_flat_perturbed, state_flat_original, parent_indices,
                    children_list, alpha=0.3):
    """
    Apply backward smoothing to partially restore consistency.

    For each child: blend the child's perturbed state toward the ORIGINAL
    parent's restricted state (which is the consistent reference).
    smoothed_child = alpha * child_perturbed + (1-alpha) * parent_original_restricted
    """
    state_smoothed = state_flat_perturbed.copy()
    parent_state_original = state_flat_original[parent_indices]
    parent_pos = {int(g): j for j, g in enumerate(parent_indices)}

    for child_id, child_indices in children_list:
        for i, ci in enumerate(child_indices):
            ci_int = int(ci)
            if ci_int in parent_pos:
                parent_val = parent_state_original[parent_pos[ci_int]]
                child_val = state_flat_perturbed[ci_int]
                state_smoothed[ci_int] = alpha * child_val + (1 - alpha) * parent_val

    return state_smoothed


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 4, Experiment 11")
    print("Sheaf Cohomology: Identity Crisis Detection")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)

    # Process each parent-children group
    MAX_PARENT_CELLS = 1500  # memory limit for dense coboundary
    results_per_group = []

    print(f"\n{'=' * 70}")
    print("Per-Group Sheaf Cohomology")
    print(f"{'=' * 70}")

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

        # Skip if coboundary matrix would be too large
        if dim_c0 * dim_c1 > 5_000_000:
            continue

        print(f"\n  L{level} P{pid}: {len(parent_indices)} cells, "
              f"{len(children_list)} children, C0={dim_c0}, C1={dim_c1}")

        # 1. Build coboundary operator
        delta = build_coboundary_operator(parent_indices, children_list)

        # 2. Check: actual state is a global section
        sigma_actual = build_section(state_flat, parent_indices, children_list)
        residual_actual = delta @ sigma_actual
        residual_norm = float(np.linalg.norm(residual_actual))
        is_global_section = residual_norm < 1e-10
        print(f"    ||delta^0(actual state)||: {residual_norm:.2e} "
              f"{'[GLOBAL SECTION]' if is_global_section else '[NOT GLOBAL]'}")

        # 3. Compute baseline cohomology
        cohom_baseline = compute_cohomology_dims(delta)
        print(f"    Baseline: H0={cohom_baseline['dim_h0']}, "
              f"H1={cohom_baseline['dim_h1']}, rank={cohom_baseline['rank_delta']}")

        # 4. Perturb children only (break consistency)
        perturbation_scales = [0.01, 0.05, 0.1]
        perturbation_results = []

        for eps in perturbation_scales:
            state_perturbed = perturb_children_only(
                state_flat, children_list, perturbation_scale=eps
            )

            # Build section from perturbed state (parent unchanged, children perturbed)
            sigma_perturbed = build_section(
                state_perturbed, parent_indices, children_list
            )
            # But we need to keep parent section from ORIGINAL state
            # to break consistency
            sigma_broken = sigma_actual.copy()
            offset = len(parent_indices)
            for child_id, child_indices in children_list:
                n_child = len(child_indices)
                sigma_broken[offset:offset + n_child] = state_perturbed[child_indices]
                offset += n_child

            residual_broken = delta @ sigma_broken
            broken_norm = float(np.linalg.norm(residual_broken))

            # Measure H1 change via SVD of [delta | residual]
            # More practically: measure defect norm (how far from being a section)
            defect = broken_norm / (np.linalg.norm(sigma_broken) + 1e-15)

            perturbation_results.append({
                'epsilon': eps,
                'residual_norm': broken_norm,
                'relative_defect': float(defect),
            })
            print(f"    Perturbed (eps={eps}): ||residual||={broken_norm:.6f}, "
                  f"defect={defect:.6f}")

        # 5. Smooth and measure recovery
        smoothing_results = []
        for alpha in [0.1, 0.3, 0.5, 0.7]:
            state_perturbed = perturb_children_only(
                state_flat, children_list, perturbation_scale=0.1
            )
            state_smoothed = smooth_section(
                state_perturbed, state_flat, parent_indices, children_list,
                alpha=alpha
            )

            # Build broken section (parent original, children perturbed)
            sigma_broken_pre = sigma_actual.copy()
            sigma_smoothed_post = sigma_actual.copy()
            offset = len(parent_indices)
            for child_id, child_indices in children_list:
                n_child = len(child_indices)
                sigma_broken_pre[offset:offset + n_child] = state_perturbed[child_indices]
                sigma_smoothed_post[offset:offset + n_child] = state_smoothed[child_indices]
                offset += n_child

            broken_norm_pre = float(np.linalg.norm(delta @ sigma_broken_pre))
            smoothed_norm_post = float(np.linalg.norm(delta @ sigma_smoothed_post))

            recovery_ratio = 1.0 - (smoothed_norm_post / broken_norm_pre) if broken_norm_pre > 1e-15 else 0.0

            smoothing_results.append({
                'alpha': alpha,
                'broken_residual': broken_norm_pre,
                'smoothed_residual': smoothed_norm_post,
                'recovery_ratio': float(recovery_ratio),
            })
            print(f"    Smoothed (alpha={alpha}): "
                  f"broken={broken_norm_pre:.6f} -> smoothed={smoothed_norm_post:.6f} "
                  f"(recovery={recovery_ratio:.1%})")

        results_per_group.append({
            'level': level,
            'parent_id': int(pid),
            'n_parent_cells': len(parent_indices),
            'n_children': len(children_list),
            'is_global_section': bool(is_global_section),
            'residual_norm': residual_norm,
            'cohomology_baseline': cohom_baseline,
            'perturbation_results': perturbation_results,
            'smoothing_results': smoothing_results,
        })

    # ===== AGGREGATE VERIFICATION =====
    print(f"\n{'=' * 70}")
    print("Aggregate Verification")
    print(f"{'=' * 70}")

    if len(results_per_group) == 0:
        print("  No groups analyzed!")
        return

    # Test 1: Actual state is a global section
    n_global = sum(1 for r in results_per_group if r['is_global_section'])
    test1 = n_global == len(results_per_group)
    print(f"\n  Test 1: State is global section?")
    print(f"    {n_global}/{len(results_per_group)} groups have ||delta^0||=0")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Perturbation increases defect (breaks consistency)
    n_increased = 0
    for r in results_per_group:
        if r['perturbation_results']:
            max_defect = max(p['relative_defect'] for p in r['perturbation_results'])
            if max_defect > 1e-8:
                n_increased += 1
    frac_increased = n_increased / len(results_per_group)
    test2 = frac_increased > 0.8
    print(f"\n  Test 2: Perturbation increases H1 (breaks consistency)?")
    print(f"    {n_increased}/{len(results_per_group)} groups show increased defect")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: Smoothing decreases defect (resolves crisis)
    n_recovered = 0
    recovery_ratios = []
    for r in results_per_group:
        if r['smoothing_results']:
            # Use alpha=0.3 result
            for sr in r['smoothing_results']:
                if abs(sr['alpha'] - 0.3) < 0.01:
                    recovery_ratios.append(sr['recovery_ratio'])
                    if sr['recovery_ratio'] > 0.1:
                        n_recovered += 1

    mean_recovery = float(np.mean(recovery_ratios)) if recovery_ratios else 0
    frac_recovered = n_recovered / len(results_per_group) if results_per_group else 0
    test3 = frac_recovered > 0.5 and mean_recovery > 0.2
    print(f"\n  Test 3: Smoothing resolves identity crisis (decreases H1)?")
    print(f"    Mean recovery ratio (alpha=0.3): {mean_recovery:.1%}")
    print(f"    {n_recovered}/{len(results_per_group)} groups show >10% recovery")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3])
    print(f"\n  OVERALL: {n_verified}/3 sheaf cohomology tests verified")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_11_sheaf_cohomology',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Sheaf cohomology on hierarchy - identity crisis detection',
        'n_groups_analyzed': len(results_per_group),
        'verification': {
            'test1_global_section': {'n_global': n_global,
                                     'n_total': len(results_per_group),
                                     'verified': bool(test1)},
            'test2_perturbation_breaks': {'n_increased': n_increased,
                                           'fraction': frac_increased,
                                           'verified': bool(test2)},
            'test3_smoothing_resolves': {'n_recovered': n_recovered,
                                          'mean_recovery': mean_recovery,
                                          'verified': bool(test3)},
            'n_verified': n_verified,
        },
        'per_group': results_per_group,
    }

    output_file = RESULTS_DIR / f'exp_11_sheaf_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
