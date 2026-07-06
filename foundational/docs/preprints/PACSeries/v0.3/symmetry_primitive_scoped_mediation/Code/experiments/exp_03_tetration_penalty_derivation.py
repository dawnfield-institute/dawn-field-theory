"""
Milestone 6 -- Exp 03: Tetration Penalty Derivation

Block A: The Scope Boundary Mechanism

PURPOSE: Show that scope boundaries attenuate spectral information
geometrically per hierarchy level. The cumulative attenuation across
4 levels constrains hierarchies to at most ~4 usable levels — the
tetration termination penalty.

The previous version used a toy lattice model (diffusion + watershed)
that produced near-zero confounding. This version uses the real
confluent identity hierarchy and measures actual transfer matrix
attenuation at each level boundary.

Method:
1. Load confluent identity hierarchy (same as exp_01)
2. For each parent-child boundary, compute transfer matrix T and
   dominant eigenvalue (spectral retention per boundary)
3. Group by hierarchy level: measure mean retention per level
4. Test whether retention decreases geometrically with level depth

Tests:
  1. Spectral retention decreases with level (Spearman rho < -0.5)
  2. Per-level attenuation fits approximate geometric decay (R^2 > 0.75, HARDENED)
  3. Geometric base in phi range: 1/phi^2 < base < 1/phi
  4. Within-level consistency: CV < 1.0 at all levels with n >= 5

Predicted: 3/4
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
M6_ROOT = SCRIPT_DIR.parent
CI_SCRIPTS = SCRIPT_DIR.parents[1] / "confluent_identity" / "scripts"
sys.path.insert(0, str(M6_ROOT))
sys.path.insert(0, str(CI_SCRIPTS))

from core.scope import (
    PHI, INV_PHI,
    build_transfer_matrix, decompose_harmonic_transient,
    _get_eigenbasis,
)
from _shared import (
    load_baseline, build_lattice_adjacency,
    get_parent_children_data, K_MODES,
)

RESULTS_DIR = M6_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("MILESTONE 6 - EXP 03: TETRATION PENALTY DERIVATION")
    print("Block A: The Scope Boundary Mechanism")
    print("=" * 70)

    P_field, A_field, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    adjacency = build_lattice_adjacency(C)
    state_flat = C.ravel()
    n_levels = len(labels_by_level)

    print(f"\n  Hierarchy: {n_levels} levels")
    for i, labels in enumerate(labels_by_level):
        n_regions = len(set(labels.ravel()) - {-1})
        print(f"    Level {i}: {n_regions} regions")

    # ============================================================
    # STEP 1: Compute transfer matrix properties at every boundary
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: TRANSFER MATRIX AT EVERY BOUNDARY")
    print("=" * 60)

    boundary_data = []  # list of dicts per parent-child pair

    for (level, pid), pidx, children, L_parent, state_parent in \
            get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):

        eigenvalues_p, eigenvectors_p = _get_eigenbasis(L_parent, state_parent, k=K_MODES)

        # Map global indices to parent-local indices
        global_to_local = {int(g): pos for pos, g in enumerate(pidx)}

        for child_id, child_indices in children:
            # Map child global indices to parent local space
            child_local = np.array([global_to_local[int(c)] for c in child_indices
                                    if int(c) in global_to_local])
            if len(child_local) < 4:
                continue

            k = min(K_MODES, eigenvectors_p.shape[1])
            T = build_transfer_matrix(eigenvectors_p, child_local, k=k)
            T_harm, T_trans, T_eigs = decompose_harmonic_transient(T)

            dom_eig = float(abs(T_eigs[0]))
            size_ratio = len(child_indices) / len(pidx)
            efficiency = dom_eig / (size_ratio + 1e-15)

            boundary_data.append({
                'level': level,
                'parent_id': pid,
                'child_id': child_id,
                'parent_size': len(pidx),
                'child_size': len(child_indices),
                'size_ratio': size_ratio,
                'dominant_eigenvalue': dom_eig,
                'spectral_efficiency': efficiency,
                'norm_T': float(np.linalg.norm(T, 'fro')),
                'norm_T_harm': float(np.linalg.norm(T_harm, 'fro')),
            })

    print(f"\n  Total boundaries: {len(boundary_data)}")

    # ============================================================
    # STEP 2: Group by level, compute per-level statistics
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: PER-LEVEL ATTENUATION ANALYSIS")
    print("=" * 60)

    by_level = {}
    for bd in boundary_data:
        lv = bd['level']
        if lv not in by_level:
            by_level[lv] = []
        by_level[lv].append(bd)

    level_stats = {}
    for lv in sorted(by_level.keys()):
        entries = by_level[lv]
        eigs = [e['dominant_eigenvalue'] for e in entries]
        sizes = [e['size_ratio'] for e in entries]
        effs = [e['spectral_efficiency'] for e in entries]
        norms = [e['norm_T'] for e in entries]

        level_stats[lv] = {
            'n': len(entries),
            'mean_eigenvalue': float(np.mean(eigs)),
            'std_eigenvalue': float(np.std(eigs)),
            'cv_eigenvalue': float(np.std(eigs) / (np.mean(eigs) + 1e-15)),
            'mean_size_ratio': float(np.mean(sizes)),
            'mean_efficiency': float(np.mean(effs)),
            'mean_norm': float(np.mean(norms)),
        }

        print(f"\n    Level {lv} (n={len(entries)}):")
        print(f"      Mean dominant eigenvalue: {np.mean(eigs):.6f}")
        print(f"      CV eigenvalue: {np.std(eigs) / (np.mean(eigs) + 1e-15):.4f}")
        print(f"      Mean size ratio: {np.mean(sizes):.4f}")
        print(f"      Mean spectral efficiency: {np.mean(effs):.4f}")
        print(f"      Mean ||T||_F: {np.mean(norms):.6f}")

    # ============================================================
    # STEP 3: Geometric decay analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: GEOMETRIC DECAY ANALYSIS")
    print("=" * 60)

    sorted_levels = sorted(level_stats.keys())
    level_means = [(lv, level_stats[lv]['mean_eigenvalue']) for lv in sorted_levels]

    print(f"\n  Eigenvalue trajectory across levels:")
    for lv, mean_eig in level_means:
        print(f"    Level {lv}: {mean_eig:.6f}")

    # Spearman trend
    if len(level_means) >= 3:
        levels_arr = [x[0] for x in level_means]
        eigs_arr = [x[1] for x in level_means]
        trend_rho, trend_p = spearmanr(levels_arr, eigs_arr)
    elif len(level_means) >= 2:
        trend_rho = -1.0 if level_means[-1][1] < level_means[0][1] else 1.0
        trend_p = 0.5
    else:
        trend_rho = 0.0
        trend_p = 1.0

    print(f"\n  Spearman rho (level vs eigenvalue): {trend_rho:.4f} (p={trend_p:.4f})")

    # Geometric fit: log(eigenvalue) = a * level + b
    if len(level_means) >= 3:
        levels_fit = np.array([x[0] for x in level_means], dtype=float)
        eigs_fit = np.array([x[1] for x in level_means])
        # Filter out zero/tiny eigenvalues
        valid = eigs_fit > 1e-15
        if np.sum(valid) >= 3:
            log_eigs = np.log(eigs_fit[valid])
            coeffs = np.polyfit(levels_fit[valid], log_eigs, 1)
            pred = np.polyval(coeffs, levels_fit[valid])
            ss_res = np.sum((log_eigs - pred) ** 2)
            ss_tot = np.sum((log_eigs - np.mean(log_eigs)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            decay_rate = coeffs[0]  # log-space slope
            geometric_base = np.exp(decay_rate)  # per-level attenuation factor

            print(f"\n  Geometric fit (log-linear):")
            print(f"    Decay rate (log slope): {decay_rate:.4f}")
            print(f"    Geometric base: {geometric_base:.4f}")
            print(f"    1/phi = {INV_PHI:.4f}")
            print(f"    Base vs 1/phi: {abs(geometric_base - INV_PHI) / INV_PHI * 100:.1f}%")
            print(f"    R^2: {r_squared:.4f}")
        else:
            r_squared = 0
            geometric_base = 0
            decay_rate = 0
    else:
        r_squared = 0
        geometric_base = 0
        decay_rate = 0

    # Per-level ratios
    per_level_ratios = []
    for i in range(len(level_means) - 1):
        if level_means[i][1] > 1e-15:
            ratio = level_means[i + 1][1] / level_means[i][1]
            per_level_ratios.append(ratio)
            print(f"\n    Level {level_means[i][0]}->{level_means[i+1][0]}: "
                  f"ratio = {ratio:.4f}")

    # Cumulative product
    if level_means:
        print(f"\n  Cumulative eigenvalue product:")
        cum_prod = 1.0
        for lv, mean_eig in level_means:
            cum_prod *= mean_eig
            n_lv = lv - level_means[0][0] + 1
            inv_phi_n = INV_PHI ** n_lv
            print(f"    After level {lv}: {cum_prod:.6f} "
                  f"(1/phi^{n_lv} = {inv_phi_n:.6f})")

    # ============================================================
    # STEP 4: Hierarchy termination analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: HIERARCHY TERMINATION")
    print("=" * 60)

    # Count usable parents at each level
    usable_by_level = {}
    for lv in sorted(by_level.keys()):
        entries = by_level[lv]
        usable = sum(1 for e in entries if e['child_size'] >= 4)
        usable_by_level[lv] = usable
        print(f"    Level {lv}: {len(entries)} boundaries, "
              f"{usable} with child_size >= 4")

    # The hierarchy terminates when regions are too small to bisect
    deepest_level = max(by_level.keys())
    deepest_entries = by_level[deepest_level]
    deepest_child_sizes = [e['child_size'] for e in deepest_entries]
    mean_deepest_size = np.mean(deepest_child_sizes)

    print(f"\n  Deepest level: {deepest_level}")
    print(f"  Mean child size at deepest: {mean_deepest_size:.1f}")
    print(f"  Min child size at deepest: {min(deepest_child_sizes)}")
    print(f"  Hierarchy naturally bounded at ~{deepest_level} levels")

    # Check: does the hierarchy reach level 4?
    max_level = max(sorted_levels)
    n_usable_levels = len([lv for lv in sorted_levels if level_stats[lv]['n'] >= 2])

    print(f"\n  Max level: {max_level}")
    print(f"  Usable levels (n >= 2 boundaries): {n_usable_levels}")

    # ============================================================
    # VERIFICATION
    # ============================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Test 1: Spectral retention decreases with level
    test1 = trend_rho < -0.5
    print(f"\n  Test 1: Spectral retention decreases with level (Spearman rho < -0.5)")
    print(f"    Spearman rho: {trend_rho:.4f}")
    print(f"    -> {'VERIFIED' if test1 else 'NOT VERIFIED'}")

    # Test 2: Approximate geometric decay
    # HARDENED: round 1 — tighten from R^2 > 0.5 to R^2 > 0.75
    # R^2 > 0.5 is permissive (explains only half the variance). A genuine
    # geometric decay should clear 0.75 comfortably.
    test2 = r_squared > 0.75
    print(f"\n  Test 2: Approximate geometric decay (R^2 > 0.75) [HARDENED]")
    print(f"    R^2: {r_squared:.4f}")
    print(f"    Threshold: 0.75 (hardened from 0.5)")
    if geometric_base > 0:
        print(f"    Geometric base: {geometric_base:.4f} (1/phi = {INV_PHI:.4f})")
    print(f"    -> {'VERIFIED' if test2 else 'NOT VERIFIED'}")

    # Test 3: Geometric base in phi range (1/phi^2 < base < 1/phi)
    inv_phi2 = INV_PHI ** 2
    test3 = geometric_base > inv_phi2 and geometric_base < INV_PHI
    print(f"\n  Test 3: Geometric base in phi range ({inv_phi2:.4f} < base < {INV_PHI:.4f})")
    print(f"    Geometric base: {geometric_base:.4f}")
    print(f"    1/phi^2 = {inv_phi2:.4f}, 1/phi = {INV_PHI:.4f}")
    print(f"    -> {'VERIFIED' if test3 else 'NOT VERIFIED'}")

    # Test 4: Within-level consistency (CV < 1.0 at levels with n >= 5)
    level_cvs_strict = [(lv, level_stats[lv]['cv_eigenvalue'])
                        for lv in sorted_levels if level_stats[lv]['n'] >= 5]
    if level_cvs_strict:
        all_consistent = all(cv < 1.0 for _, cv in level_cvs_strict)
    else:
        all_consistent = False
    test4 = all_consistent
    print(f"\n  Test 4: Within-level consistency (CV < 1.0 at levels with n >= 5)")
    for lv, cv in level_cvs_strict:
        status = 'ok' if cv < 1.0 else 'HIGH'
        print(f"    Level {lv}: CV = {cv:.4f} [{status}]")
    if not level_cvs_strict:
        print(f"    No levels with n >= 5")
    print(f"    -> {'VERIFIED' if test4 else 'NOT VERIFIED'}")

    verified = sum([test1, test2, test3, test4])
    print(f"\n  TOTAL: {verified}/4 verified")

    # -- Save results --
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_03_tetration_penalty_derivation',
        'milestone': 6,
        'block': 'A',
        'n_boundaries': len(boundary_data),
        'n_levels': n_levels,
        'level_stats': {
            str(lv): level_stats[lv] for lv in sorted_levels
        },
        'geometric_fit': {
            'decay_rate': float(decay_rate),
            'geometric_base': float(geometric_base),
            'r_squared': float(r_squared),
            'inv_phi': float(INV_PHI),
        },
        'trend': {
            'spearman_rho': float(trend_rho),
            'spearman_p': float(trend_p),
        },
        'termination': {
            'max_level': int(max_level),
            'mean_deepest_child_size': float(mean_deepest_size),
            'n_usable_levels': n_usable_levels,
        },
        'within_level_cv': {
            str(lv): float(cv) for lv, cv in level_cvs_strict
        } if level_cvs_strict else {},
        'verification': {
            'test1': test1,
            'test2': test2,
            'test3': test3,
            'test4': test4,
            'verified_count': verified,
        },
        'timestamp': datetime.now().isoformat(),
    }

    outpath = RESULTS_DIR / f"exp_03_tetration_penalty_derivation_{ts}.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
