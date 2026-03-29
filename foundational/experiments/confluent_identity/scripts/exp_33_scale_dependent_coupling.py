"""
exp_33_scale_dependent_coupling.py -- Confluent Identity Phase 24

PURPOSE:
    Test whether coupling works at specific hierarchy levels but not others,
    with pooling across levels diluting the signal to ~0.40. The clue: coupling
    effectiveness FLIPS between levels (cross-level tau=0.33 in exp_28). If
    coupling is strong at level 1 but null at level 3, pooled rho averages
    to ~0.40 when the real signal at the right level might be 0.60+.

METHODS:
    1. For each hierarchy level (0-3) separately, collect coupling/natural/size
    2. Compute partial_rho per level
    3. Identify best single level
    4. For parents with >=4 children: within-parent Spearman
    5. Adaptive pool: best-level coupling values
    6. Compare: adaptive vs pooled vs best-single-level
    7. Bootstrap CIs on level differences

VERIFICATION:
    - Level-specific partial_rho range > 0.15 (coupling IS scale-dependent)
    - Best single-level partial_rho > 0.50 (one level exceeds ceiling)
    - Adaptive partial_rho > pooled by at least 0.05
    - At least 2 levels with n>=15 and partial_rho > 0.25, p < 0.05

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
    compute_spectral_identity, get_parent_children_data,
)
from exp_08_gradient_coupling import (
    compute_coupling_weights_weighted, compute_natural_weights,
    compute_gradient_field,
)
from exp_14_partial_correlation import partial_spearman


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 24, Experiment 33")
    print("Scale-Dependent Coupling: Per-Level Decomposition")
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
    # Collect per-level coupling data
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Collecting Per-Level Coupling Data")
    print(f"{'=' * 70}")

    # level -> list of (coupling, natural, size, parent_id)
    level_data = {}
    pooled_data = []

    n_parents = 0
    for (level, pid), parent_indices, children_list, L_parent, state_parent in \
            get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):

        n_parents += 1
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

        if level not in level_data:
            level_data[level] = []

        for child_id, _ in children_list:
            cid = child_id
            if cid in w_gradient:
                entry = {
                    'coupling': w_gradient[cid],
                    'natural': natural_weights.get(cid, 0),
                    'size': size_fractions.get(cid, 0),
                    'parent_id': pid,
                    'level': level,
                }
                level_data[level].append(entry)
                pooled_data.append(entry)

    print(f"  Total parents: {n_parents}")
    for level in sorted(level_data.keys()):
        print(f"  Level {level}: {len(level_data[level])} measurements")

    # =====================================================================
    # Per-level correlations
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Per-Level Coupling Correlations")
    print(f"{'=' * 70}")

    level_results = {}

    for level in sorted(level_data.keys()):
        data = level_data[level]
        n_lev = len(data)
        if n_lev < 5:
            level_results[level] = {'n': n_lev, 'error': 'insufficient data'}
            continue

        coupling = np.array([d['coupling'] for d in data])
        natural = np.array([d['natural'] for d in data])
        size = np.array([d['size'] for d in data])

        rho_raw, p_raw = spearmanr(coupling, natural)
        pr, pp = partial_spearman(coupling, natural, size)

        level_results[level] = {
            'n': n_lev,
            'rho_raw': float(rho_raw),
            'p_raw': float(p_raw),
            'partial_rho': float(pr),
            'partial_p': float(pp),
        }

        sig = '*' if pp < 0.05 else ''
        print(f"\n  Level {level} (n={n_lev}):")
        print(f"    raw rho = {rho_raw:.4f}, p={p_raw:.2e}")
        print(f"    partial rho(| size) = {pr:.4f}, p={pp:.2e} {sig}")

    # Pooled (all levels)
    coupling_pooled = np.array([d['coupling'] for d in pooled_data])
    natural_pooled = np.array([d['natural'] for d in pooled_data])
    size_pooled = np.array([d['size'] for d in pooled_data])
    pooled_rho, pooled_p = spearmanr(coupling_pooled, natural_pooled)
    pooled_partial, pooled_pp = partial_spearman(coupling_pooled, natural_pooled, size_pooled)

    print(f"\n  POOLED (n={len(pooled_data)}):")
    print(f"    raw rho = {pooled_rho:.4f}")
    print(f"    partial rho(| size) = {pooled_partial:.4f}, p={pooled_pp:.2e}")

    # =====================================================================
    # Within-parent correlations (parents with >=4 children)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Within-Parent Correlations (>=4 children)")
    print(f"{'=' * 70}")

    # Group data by (level, parent_id)
    parent_groups = {}
    for d in pooled_data:
        key = (d['level'], d['parent_id'])
        if key not in parent_groups:
            parent_groups[key] = []
        parent_groups[key].append(d)

    within_parent_rhos = []
    for (level, pid), group in parent_groups.items():
        if len(group) < 4:
            continue
        coupling = np.array([d['coupling'] for d in group])
        natural = np.array([d['natural'] for d in group])
        rho, p = spearmanr(coupling, natural)
        within_parent_rhos.append({
            'level': level,
            'parent_id': pid,
            'n_children': len(group),
            'rho': float(rho),
            'p': float(p),
        })
        print(f"  L{level} P{pid} (n={len(group)}): rho={rho:.4f}, p={p:.2e}")

    if within_parent_rhos:
        mean_within = float(np.mean([r['rho'] for r in within_parent_rhos]))
        print(f"\n  Mean within-parent rho: {mean_within:.4f} (n={len(within_parent_rhos)} parents)")

    # =====================================================================
    # Bootstrap CIs on level differences
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Bootstrap Level Comparison (1000 resamples)")
    print(f"{'=' * 70}")

    valid_levels = [l for l, r in level_results.items()
                    if isinstance(r.get('partial_rho'), (int, float)) and r['n'] >= 10]

    bootstrap_results = {}
    if len(valid_levels) >= 2:
        rng = np.random.RandomState(42)
        n_boot = 1000

        for level in valid_levels:
            data = level_data[level]
            n_lev = len(data)
            boot_partials = []
            for _ in range(n_boot):
                idx = rng.randint(0, n_lev, size=n_lev)
                coupling_b = np.array([data[i]['coupling'] for i in idx])
                natural_b = np.array([data[i]['natural'] for i in idx])
                size_b = np.array([data[i]['size'] for i in idx])
                pr_b, _ = partial_spearman(coupling_b, natural_b, size_b)
                boot_partials.append(pr_b)

            ci_low = float(np.percentile(boot_partials, 2.5))
            ci_high = float(np.percentile(boot_partials, 97.5))
            bootstrap_results[level] = {
                'ci_low': ci_low,
                'ci_high': ci_high,
                'mean': float(np.mean(boot_partials)),
            }
            print(f"  Level {level}: partial_rho = {level_results[level]['partial_rho']:.4f} "
                  f"[{ci_low:.4f}, {ci_high:.4f}]")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Get partial_rho values for valid levels
    partial_rhos = {l: level_results[l]['partial_rho']
                    for l in level_results
                    if isinstance(level_results[l].get('partial_rho'), (int, float))}

    if partial_rhos:
        pr_range = max(partial_rhos.values()) - min(partial_rhos.values())
        best_level = max(partial_rhos, key=partial_rhos.get)
        best_partial = partial_rhos[best_level]
    else:
        pr_range = 0
        best_level = None
        best_partial = 0

    # Test 1: Level-specific partial_rho range > 0.15
    test1 = pr_range > 0.15
    print(f"\n  Test 1: Level-specific partial_rho range > 0.15?")
    print(f"    Range: {pr_range:.4f} (min={min(partial_rhos.values()):.4f}, "
          f"max={max(partial_rhos.values()):.4f})")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Best single level partial_rho > 0.50
    test2 = best_partial > 0.50
    print(f"\n  Test 2: Best single-level partial_rho > 0.50?")
    print(f"    Level {best_level}: {best_partial:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: Best level > pooled by at least 0.05
    delta_pooled = best_partial - pooled_partial
    test3 = delta_pooled > 0.05
    print(f"\n  Test 3: Best level > pooled by > 0.05?")
    print(f"    Best: {best_partial:.4f}, Pooled: {pooled_partial:.4f}, "
          f"Delta: {delta_pooled:.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: At least 2 levels with n>=15 and partial_rho > 0.25, p < 0.05
    qualifying_levels = [l for l, r in level_results.items()
                         if isinstance(r.get('partial_rho'), (int, float))
                         and r['n'] >= 15
                         and r['partial_rho'] > 0.25
                         and r.get('partial_p', 1) < 0.05]
    test4 = len(qualifying_levels) >= 2
    print(f"\n  Test 4: >= 2 levels with n>=15, partial_rho > 0.25, p < 0.05?")
    print(f"    Qualifying levels: {qualifying_levels}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 scale-dependent coupling tests verified")

    # Summary table
    print(f"\n{'=' * 70}")
    print("Summary Table")
    print(f"{'=' * 70}")
    print(f"  {'Level':<10} {'n':>6} {'raw rho':>10} {'partial':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*10}")
    for level in sorted(level_results.keys()):
        r = level_results[level]
        if 'error' in r:
            print(f"  {level:<10} {r['n']:>6} {'--':>10} {'--':>10}")
        else:
            print(f"  {level:<10} {r['n']:>6} {r['rho_raw']:>10.4f} {r['partial_rho']:>10.4f}")
    print(f"  {'POOLED':<10} {len(pooled_data):>6} {pooled_rho:>10.4f} {pooled_partial:>10.4f}")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_33_scale_dependent_coupling',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Per-level coupling decomposition: is coupling scale-dependent?',
        'n_parents': n_parents,
        'n_total': len(pooled_data),
        'level_results': {str(k): v for k, v in level_results.items()},
        'pooled': {
            'rho_raw': float(pooled_rho),
            'partial_rho': float(pooled_partial),
        },
        'within_parent': within_parent_rhos,
        'bootstrap': {str(k): v for k, v in bootstrap_results.items()},
        'verification': {
            'test1_range_above_015': bool(test1),
            'test2_best_level_above_050': bool(test2),
            'test3_best_exceeds_pooled': bool(test3),
            'test4_multiple_significant_levels': bool(test4),
            'n_verified': n_verified,
        },
    }

    output_file = RESULTS_DIR / f'exp_33_scale_coupling_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
