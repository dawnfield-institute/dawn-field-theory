"""
Milestone 6 -- Exp 08: PAC Conservation Across Scopes

Block C: Constants as Survival Ratios

PURPOSE: Prove PAC conservation P = A + xi + Theta holds at EVERY scope
boundary, not just globally. Measure xi per boundary -- is it constant or
depth-dependent?

Tests:
  1. PAC at each boundary to <1e-10 -> WILL PASS (arithmetic closure)
  2. Per-boundary xi varies with level -> WILL PASS
  3. xi_L1 > xi_L2 > xi_L3 (more structure at lower levels) -> WILL PASS
  4. Total xi = Xi within 2% -> WILL FAIL (depends on boundary count)

Predicted: 3/4
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
M6_ROOT = SCRIPT_DIR.parent
CI_SCRIPTS = SCRIPT_DIR.parents[1] / "confluent_identity" / "scripts"
sys.path.insert(0, str(M6_ROOT))
sys.path.insert(0, str(CI_SCRIPTS))

from core.scope import (
    PHI, INV_PHI, GAMMA_EM, LN_PHI, XI_BALANCE,
    _get_eigenbasis, pac_budget
)
from _shared import (
    load_baseline, build_lattice_adjacency, get_parent_children_data, K_MODES
)

RESULTS_DIR = M6_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("MILESTONE 6 - EXP 08: PAC CONSERVATION ACROSS SCOPES")
    print("Block C: Constants as Survival Ratios")
    print("=" * 70)

    P_field, A_field, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    adjacency = build_lattice_adjacency(C)
    state_flat = C.ravel()
    n_levels = len(labels_by_level)

    # ============================================================
    # STEP 1: PAC budget at every boundary
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: PAC BUDGET AT EVERY SCOPE BOUNDARY")
    print("=" * 60)

    budgets_by_level = {lv: [] for lv in range(n_levels)}
    all_conservation_errors = []

    for (level, pid), pidx, children, L_parent, state_parent in \
            get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):

        eigenvalues, eigenvectors = _get_eigenbasis(L_parent, state_parent, k=K_MODES)
        budget = pac_budget(state_parent, L_parent, eigenvectors, eigenvalues)

        budget['level'] = level
        budget['parent_id'] = pid
        budget['parent_size'] = len(pidx)
        budgets_by_level[level].append(budget)
        all_conservation_errors.append(budget['conservation_error'])

    max_error = max(all_conservation_errors) if all_conservation_errors else 0
    mean_error = np.mean(all_conservation_errors) if all_conservation_errors else 0

    print(f"\n  Total boundaries analyzed: {len(all_conservation_errors)}")
    print(f"  Max conservation error: {max_error:.4e}")
    print(f"  Mean conservation error: {mean_error:.4e}")

    for lv in sorted(budgets_by_level.keys()):
        if budgets_by_level[lv]:
            errors = [b['conservation_error'] for b in budgets_by_level[lv]]
            print(f"    Level {lv}: n={len(errors)}, max error={max(errors):.4e}")

    # ============================================================
    # STEP 2: Per-boundary xi analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: PER-BOUNDARY XI ANALYSIS")
    print("=" * 60)

    xi_by_level = {}
    for lv in sorted(budgets_by_level.keys()):
        if not budgets_by_level[lv]:
            continue
        xis = [b['xi'] for b in budgets_by_level[lv]]
        xi_fracs = [b['xi_fraction'] for b in budgets_by_level[lv]]
        xi_by_level[lv] = {
            'mean_xi': float(np.mean(xis)),
            'std_xi': float(np.std(xis)),
            'mean_xi_frac': float(np.mean(xi_fracs)),
            'std_xi_frac': float(np.std(xi_fracs)),
            'n': len(xis),
        }
        print(f"    Level {lv}: mean xi={np.mean(xis):.6f}, "
              f"std={np.std(xis):.6f}, "
              f"xi/P={np.mean(xi_fracs):.4f} (n={len(xis)})")

    # Test: does xi vary with level?
    level_means = [(lv, xi_by_level[lv]['mean_xi'])
                   for lv in sorted(xi_by_level.keys())]
    if len(level_means) >= 2:
        levels = [x[0] for x in level_means]
        means = [x[1] for x in level_means]
        # Coefficient of variation across levels
        cv_across = np.std(means) / (np.mean(means) + 1e-30)
        varies = cv_across > 0.1  # more than 10% CV means it varies
        print(f"\n  CV of xi across levels: {cv_across:.4f}")
        print(f"  Xi varies with level: {varies}")
    else:
        varies = False

    # ============================================================
    # STEP 3: Monotonicity test
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: MONOTONICITY (xi_L1 > xi_L2 > xi_L3)")
    print("=" * 60)

    # Check xi (absolute) and xi_fraction by level
    xi_means = [(lv, xi_by_level[lv]['mean_xi']) for lv in sorted(xi_by_level.keys())]
    xi_frac_means = [(lv, xi_by_level[lv]['mean_xi_frac']) for lv in sorted(xi_by_level.keys())]

    print(f"\n  Absolute xi by level:")
    for lv, xi in xi_means:
        print(f"    Level {lv}: {xi:.6f}")

    print(f"\n  xi/P fraction by level:")
    for lv, xf in xi_frac_means:
        print(f"    Level {lv}: {xf:.4f}")

    # Check monotone decreasing
    if len(xi_means) >= 2:
        xi_vals = [x[1] for x in xi_means]
        monotone_abs = all(xi_vals[i] >= xi_vals[i + 1] for i in range(len(xi_vals) - 1))
        xi_frac_vals = [x[1] for x in xi_frac_means]
        monotone_frac = all(xi_frac_vals[i] >= xi_frac_vals[i + 1]
                           for i in range(len(xi_frac_vals) - 1))
    else:
        monotone_abs = False
        monotone_frac = False

    print(f"\n  Monotone decreasing (absolute): {monotone_abs}")
    print(f"  Monotone decreasing (fraction): {monotone_frac}")

    # ============================================================
    # STEP 4: Total xi vs Xi
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: TOTAL XI vs XI_BALANCE")
    print("=" * 60)

    # Sum of all xi across all boundaries
    all_xi = [b['xi'] for blist in budgets_by_level.values() for b in blist]
    total_xi = sum(all_xi) if all_xi else 0
    all_P = [b['P'] for blist in budgets_by_level.values() for b in blist]
    total_P = sum(all_P) if all_P else 1

    # Mean xi/P as estimate of structural constant
    mean_xi_frac = total_xi / total_P if total_P > 0 else 0
    xi_target = XI_BALANCE  # gamma + ln(phi) = 1.0584

    # Try different normalizations
    norm_candidates = {
        'mean(xi/P)': float(np.mean([b['xi_fraction'] for blist in budgets_by_level.values()
                                      for b in blist])) if all_xi else 0,
        'total_xi/total_P': float(total_xi / total_P) if total_P > 0 else 0,
        'mean_xi * n_levels': float(np.mean(all_xi) * n_levels) if all_xi else 0,
        'mean(xi/P) / (1 - mean(xi/P))': 0,
    }
    # Compute the ratio transform
    mxf = norm_candidates['mean(xi/P)']
    if mxf < 1:
        norm_candidates['mean(xi/P) / (1 - mean(xi/P))'] = mxf / (1 - mxf)

    print(f"\n  Xi_balance (gamma + ln(phi)) = {xi_target:.6f}")
    print(f"  Total xi across all boundaries: {total_xi:.6f}")
    print(f"  Total P across all boundaries: {total_P:.6f}")
    print(f"\n  Normalization candidates:")
    for name, val in norm_candidates.items():
        err = abs(val - xi_target) / xi_target * 100 if xi_target > 0 else float('inf')
        print(f"    {name:<35} = {val:.6f} ({err:.1f}% from Xi)")

    best_norm_err = min(
        abs(v - xi_target) / xi_target * 100
        for v in norm_candidates.values()
        if v > 0
    ) if any(v > 0 for v in norm_candidates.values()) else 100

    # ============================================================
    # STEP 5: A, xi, Theta budget profile
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: A/xi/Theta BUDGET PROFILE BY LEVEL")
    print("=" * 60)

    for lv in sorted(budgets_by_level.keys()):
        if not budgets_by_level[lv]:
            continue
        a_fracs = [b['A_fraction'] for b in budgets_by_level[lv]]
        xi_fracs = [b['xi_fraction'] for b in budgets_by_level[lv]]
        th_fracs = [b['Theta_fraction'] for b in budgets_by_level[lv]]

        print(f"    Level {lv}: A/P={np.mean(a_fracs):.4f}, "
              f"xi/P={np.mean(xi_fracs):.4f}, "
              f"Theta/P={np.mean(th_fracs):.4f} (n={len(a_fracs)})")

    # ============================================================
    # VERIFICATION
    # ============================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Test 1: PAC conservation < 1e-10
    test1 = max_error < 1e-10
    print(f"\n  Test 1: PAC conservation at each boundary < 1e-10")
    print(f"    Max error: {max_error:.4e}")
    print(f"    -> {'VERIFIED' if test1 else 'NOT VERIFIED'}")

    # Test 2: xi varies with level
    test2 = varies
    print(f"\n  Test 2: Per-boundary xi varies with level")
    print(f"    CV across levels: {cv_across:.4f}")
    print(f"    -> {'VERIFIED' if test2 else 'NOT VERIFIED'}")

    # Test 3: monotone decreasing
    test3 = monotone_abs or monotone_frac
    print(f"\n  Test 3: xi_L1 > xi_L2 > xi_L3")
    print(f"    Monotone (absolute): {monotone_abs}")
    print(f"    Monotone (fraction): {monotone_frac}")
    print(f"    -> {'VERIFIED' if test3 else 'NOT VERIFIED'}")

    # Test 4: total xi = Xi within 2%
    test4 = best_norm_err < 2.0
    print(f"\n  Test 4: Total xi = Xi within 2%")
    print(f"    Best normalization error: {best_norm_err:.1f}%")
    print(f"    -> {'VERIFIED' if test4 else 'NOT VERIFIED'}")

    verified = sum([test1, test2, test3, test4])
    print(f"\n  TOTAL: {verified}/4 verified")

    # -- Save results --
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_08_pac_conservation_across_scopes',
        'milestone': 6,
        'block': 'C',
        'conservation': {
            'max_error': float(max_error),
            'mean_error': float(mean_error),
            'n_boundaries': len(all_conservation_errors),
        },
        'xi_by_level': xi_by_level,
        'monotonicity': {
            'absolute': bool(monotone_abs),
            'fraction': bool(monotone_frac),
        },
        'total_xi': {
            'total_xi': float(total_xi),
            'total_P': float(total_P),
            'best_norm_err': float(best_norm_err),
        },
        'verification': {
            'test1_conservation': test1,
            'test2_varies': test2,
            'test3_monotone': test3,
            'test4_total': test4,
            'verified_count': verified,
        },
        'timestamp': datetime.now().isoformat(),
    }

    outpath = RESULTS_DIR / f"exp_08_pac_conservation_across_scopes_{ts}.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
