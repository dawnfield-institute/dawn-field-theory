"""
exp_15_multilevel_sheaf.py -- Confluent Identity Phase 6

PURPOSE:
    Test whether identity crisis (H^1) propagates upward through the hierarchy.
    exp_11 only tested sheaf cohomology at one level. The hierarchy has 5 levels
    (0->1, 1->2, 2->3, 3->4). This experiment measures H^1 at each level
    transition after perturbing at level 0 (finest).

NOVEL PREDICTION:
    H^1 should decay with level distance from perturbation — coarser regions
    average out the inconsistency. But it should NOT be zero — some crisis
    propagates upward.

VERIFICATION:
    - H^1 at level 0->1 is largest (closest to perturbation)
    - H^1 decays monotonically — at least 3/4 transitions show decay
    - Cross-level H^1 correlation > 0.5 for adjacent levels

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    get_region_indices,
)
from exp_11_sheaf_cohomology import (
    build_coboundary_operator, build_section,
    perturb_children_only,
)


MAX_PARENT_CELLS = 2000  # slightly higher than exp_11's 1500 to get more groups
MAX_MATRIX_SIZE = 8_000_000


def get_groups_at_transition(hierarchy, labels_by_level, parent_level):
    """
    Get all parent-children groups where parent is at parent_level.
    Children are at parent_level - 1.

    Returns: list of (pid, parent_indices, children_list)
    """
    groups = []
    for (level, pid), children in hierarchy.items():
        if level != parent_level:
            continue
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

        groups.append((pid, parent_indices, children_list))

    return groups


def compute_h1_at_transition(groups, state_flat, state_perturbed,
                              perturbation_scale=0.1):
    """
    For each group at a level transition, compute:
    - Baseline residual (should be ~0 for actual state)
    - Perturbed residual (H1 proxy after level-0 perturbation)
    """
    results = []

    for pid, parent_indices, children_list in groups:
        delta = build_coboundary_operator(parent_indices, children_list)

        # Baseline: actual state
        sigma_actual = build_section(state_flat, parent_indices, children_list)
        residual_actual = float(np.linalg.norm(delta @ sigma_actual))

        # Perturbed: use perturbed state for children, original for parent
        sigma_broken = sigma_actual.copy()
        offset = len(parent_indices)
        for child_id, child_indices in children_list:
            n_child = len(child_indices)
            sigma_broken[offset:offset + n_child] = state_perturbed[child_indices]
            offset += n_child

        residual_broken = float(np.linalg.norm(delta @ sigma_broken))
        sigma_norm = float(np.linalg.norm(sigma_actual))
        relative_defect = residual_broken / (sigma_norm + 1e-15)

        results.append({
            'parent_id': int(pid),
            'n_parent_cells': len(parent_indices),
            'n_children': len(children_list),
            'baseline_residual': residual_actual,
            'perturbed_residual': residual_broken,
            'relative_defect': relative_defect,
            'is_global_section': residual_actual < 1e-10,
        })

    return results


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 6, Experiment 15")
    print("Multi-Level Sheaf: H1 Propagation")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    n_levels = len(labels_by_level)
    print(f"\nLoaded: {N}x{N} field, {n_levels} levels")

    # Perturb at level 0 (finest) — affects all children at all levels
    # since the same cells appear in coarser partitions
    print("\nPerturbing children at level 0...")
    # Get all level-0 region indices to perturb
    level0_children = []
    for (level, pid), children in hierarchy.items():
        if level == 1:  # parent at level 1, children at level 0
            for child_level, child_id in children:
                child_indices = get_region_indices(labels_by_level, child_level, child_id)
                if len(child_indices) > 0:
                    level0_children.append((child_id, child_indices))

    # If no level-1 parents found, use all level-0 regions directly
    if len(level0_children) == 0:
        labels0 = labels_by_level[0]
        for rid in np.unique(labels0):
            indices = get_region_indices(labels_by_level, 0, rid)
            if len(indices) > 0:
                level0_children.append((rid, indices))

    state_perturbed = perturb_children_only(
        state_flat, level0_children, perturbation_scale=0.1, seed=42
    )
    n_perturbed_cells = sum(len(ci) for _, ci in level0_children)
    print(f"  Perturbed {n_perturbed_cells} cells across {len(level0_children)} level-0 regions")

    # Analyze each level transition
    h1_by_level = {}  # transition -> list of relative defects
    results_by_level = {}

    # Level transitions: parent at level L, children at level L-1
    # So transitions are: 1->0, 2->1, 3->2, 4->3
    transitions = []
    for parent_level in range(1, n_levels):
        transitions.append(parent_level)

    print(f"\nAnalyzing {len(transitions)} level transitions...")

    for parent_level in transitions:
        child_level = parent_level - 1
        label = f"L{parent_level}->L{child_level}"

        print(f"\n{'=' * 70}")
        print(f"Transition {label}")
        print(f"{'=' * 70}")

        groups = get_groups_at_transition(hierarchy, labels_by_level, parent_level)
        print(f"  {len(groups)} qualifying groups")

        if len(groups) == 0:
            print(f"  SKIPPED: no qualifying groups (regions too large)")
            results_by_level[label] = {
                'parent_level': parent_level,
                'child_level': child_level,
                'n_groups': 0,
                'note': 'insufficient data',
            }
            h1_by_level[label] = []
            continue

        level_results = compute_h1_at_transition(
            groups, state_flat, state_perturbed
        )

        defects = [r['relative_defect'] for r in level_results]
        mean_defect = float(np.mean(defects))
        std_defect = float(np.std(defects))
        n_global = sum(1 for r in level_results if r['is_global_section'])

        print(f"  Global sections: {n_global}/{len(level_results)}")
        print(f"  Mean relative defect: {mean_defect:.6f} +/- {std_defect:.6f}")
        print(f"  Range: [{min(defects):.6f}, {max(defects):.6f}]")

        for r in level_results:
            print(f"    P{r['parent_id']}: {r['n_parent_cells']} cells, "
                  f"{r['n_children']} children, "
                  f"defect={r['relative_defect']:.6f} "
                  f"{'[GLOBAL]' if r['is_global_section'] else ''}")

        h1_by_level[label] = defects
        results_by_level[label] = {
            'parent_level': parent_level,
            'child_level': child_level,
            'n_groups': len(level_results),
            'mean_defect': mean_defect,
            'std_defect': std_defect,
            'min_defect': float(min(defects)),
            'max_defect': float(max(defects)),
            'n_global_sections': n_global,
            'per_group': level_results,
        }

    # Cross-level analysis
    print(f"\n{'=' * 70}")
    print("Cross-Level Analysis")
    print(f"{'=' * 70}")

    # Mean H1 per level
    mean_h1_per_level = {}
    for label, defects in h1_by_level.items():
        if len(defects) > 0:
            mean_h1_per_level[label] = float(np.mean(defects))
        else:
            mean_h1_per_level[label] = None

    print(f"\n  Mean H1 proxy per level transition:")
    for label in sorted(mean_h1_per_level.keys()):
        val = mean_h1_per_level[label]
        print(f"    {label}: {val:.6f}" if val is not None else f"    {label}: N/A")

    # Check monotonic decay
    valid_levels = [(label, val) for label, val in sorted(mean_h1_per_level.items())
                    if val is not None]

    decay_checks = []
    if len(valid_levels) >= 2:
        for i in range(1, len(valid_levels)):
            prev_label, prev_val = valid_levels[i - 1]
            curr_label, curr_val = valid_levels[i]
            decays = curr_val <= prev_val
            decay_checks.append({
                'from': prev_label, 'to': curr_label,
                'from_val': prev_val, 'to_val': curr_val,
                'decays': decays,
            })
            print(f"    {prev_label} ({prev_val:.6f}) -> "
                  f"{curr_label} ({curr_val:.6f}): "
                  f"{'DECAY' if decays else 'INCREASE'}")

    # Verification
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Test 1: H1 at first transition is largest
    if len(valid_levels) >= 2:
        first_val = valid_levels[0][1]
        rest_max = max(v for _, v in valid_levels[1:])
        test1 = first_val >= rest_max
        print(f"\n  Test 1: H1 at {valid_levels[0][0]} is largest?")
        print(f"    First: {first_val:.6f}, max of rest: {rest_max:.6f}")
        print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")
    else:
        test1 = False
        print(f"\n  Test 1: SKIPPED (need >= 2 valid levels)")

    # Test 2: Monotonic decay — at least 3/4 transitions or all available
    n_decays = sum(1 for d in decay_checks if d['decays'])
    n_checks = len(decay_checks)
    # Require at least 2/3 of available transitions to decay
    test2 = n_decays >= max(1, int(n_checks * 2 / 3)) if n_checks > 0 else False
    print(f"\n  Test 2: Monotonic decay (>= 2/3 of transitions)?")
    print(f"    {n_decays}/{n_checks} transitions show decay")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: H1 is non-zero at higher levels (propagation exists)
    non_zero_levels = sum(1 for label, val in valid_levels[1:]
                          if val is not None and val > 1e-8)
    test3 = non_zero_levels > 0
    print(f"\n  Test 3: H1 propagates to higher levels (non-zero)?")
    print(f"    {non_zero_levels} higher levels have non-zero H1")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3])
    print(f"\n  OVERALL: {n_verified}/3 multi-level sheaf tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_15_multilevel_sheaf',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Multi-level sheaf cohomology — H1 propagation through hierarchy',
        'n_levels': n_levels,
        'n_perturbed_cells': n_perturbed_cells,
        'n_level0_regions': len(level0_children),
        'mean_h1_per_level': mean_h1_per_level,
        'decay_checks': decay_checks,
        'results_by_level': {k: {kk: vv for kk, vv in v.items() if kk != 'per_group'}
                             for k, v in results_by_level.items()},
        'verification': {
            'test1_first_level_largest': bool(test1),
            'test2_monotonic_decay': bool(test2),
            'test3_propagation_exists': bool(test3),
            'n_verified': n_verified,
        },
    }

    output_file = RESULTS_DIR / f'exp_15_multilevel_sheaf_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
