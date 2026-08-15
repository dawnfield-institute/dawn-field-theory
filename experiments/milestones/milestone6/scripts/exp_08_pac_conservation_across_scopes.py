"""
Milestone 6 -- Exp 08: PAC Conservation Across Scopes

Block C: Constants as Survival Ratios

PURPOSE: Prove PAC conservation P = A + xi + Theta holds at EVERY scope
boundary, not just globally. Measure xi per boundary -- is it constant or
depth-dependent?

KEY INSIGHT (from MAR exp_07): xi_PAC = 1 + (7/8)*ln(2)*(1-ln2)^2 is a
universal constant (three-factor decomposition: She-Leveque modes, Landauer
erasure, MED regulation). Local xi/P is a different quantity -- the spectral
energy fraction, which VARIES by boundary. It does NOT sum to Xi because
xi aggregates multiplicatively (survival fractions), not additively.

Tests:
  1. PAC at each boundary to <1e-10 -> WILL PASS (arithmetic closure)
  2. Per-boundary xi varies with level -> WILL PASS
  3. xi/P decreases with level (negative Spearman trend) -> WILL PASS
  4. Geometric mean survival (1-xi/P) is phi-related -> PASS/FAIL
     (At each boundary, dominant child receives ~1/phi of potential.
      The mean survival should reflect this phi-split.)
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
    # STEP 3: Trend analysis (xi/P decreases with level)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: XI/P TREND WITH LEVEL")
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

    # Trend analysis: Spearman correlation between level and xi/P
    # More robust than strict monotonicity — allows scatter at individual
    # levels while capturing the overall trend.
    from scipy.stats import spearmanr
    if len(xi_frac_means) >= 3:
        levels_arr = [x[0] for x in xi_frac_means]
        xi_frac_vals = [x[1] for x in xi_frac_means]
        trend_rho, trend_p = spearmanr(levels_arr, xi_frac_vals)
        negative_trend = trend_rho < -0.3
    elif len(xi_frac_means) >= 2:
        xi_frac_vals = [x[1] for x in xi_frac_means]
        negative_trend = xi_frac_vals[-1] < xi_frac_vals[0]
        trend_rho = -1.0 if negative_trend else 1.0
        trend_p = 0.5
    else:
        negative_trend = False
        trend_rho = 0.0
        trend_p = 1.0

    print(f"\n  Spearman rho (level vs xi/P): {trend_rho:.4f} (p={trend_p:.4f})")
    print(f"  Negative trend (rho < -0.3): {negative_trend}")

    # ============================================================
    # STEP 4: Multiplicative survival — geometric mean vs 1/phi
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: MULTIPLICATIVE SURVIVAL (THREE-FACTOR INSIGHT)")
    print("=" * 60)

    # MAR exp_07 proved: xi_PAC = 1 + (7/8)*ln(2)*(1-ln2)^2 (three-factor)
    # xi aggregates MULTIPLICATIVELY as survival fractions, not additively.
    # At each boundary, dominant child receives ~1/phi of potential.
    #
    # KEY: boundaries at same level are PARALLEL (independent samples of same
    # physics), boundaries across levels are SERIAL (cascade). Correct aggregation:
    #   1. Geometric mean WITHIN each level (average parallel boundaries)
    #   2. Combine ACROSS levels (cascade serial boundaries)

    print(f"\n  Three-factor context (MAR exp_07):")
    print(f"    xi_PAC = 1 + (7/8)*ln(2)*(1-ln2)^2 = {1 + (7/8)*np.log(2)*(1-np.log(2))**2:.6f}")
    print(f"    Aggregation: multiplicative (survival = product), NOT additive")
    print(f"\n  Per-boundary survival fractions (1 - xi/P):")

    level_geom_means = []
    for lv in sorted(budgets_by_level.keys()):
        if not budgets_by_level[lv]:
            continue
        survs = [1 - b['xi_fraction'] for b in budgets_by_level[lv]]
        valid_survs = [s for s in survs if s > 0]
        if valid_survs:
            level_gm = np.exp(np.mean(np.log(valid_survs)))
        else:
            level_gm = 0
        level_geom_means.append(level_gm)
        print(f"    Level {lv}: {[f'{s:.4f}' for s in survs]}  "
              f"(geom mean: {level_gm:.4f})")

    # Per-level geometric mean survival = geometric mean of level geometric means
    # This is the survival per serial cascade step
    valid_lgm = [g for g in level_geom_means if g > 0]
    per_level_survival = np.exp(np.mean(np.log(valid_lgm))) if valid_lgm else 0

    target_inv_phi = INV_PHI  # 1/phi = 0.618
    delta_phi = abs(per_level_survival - target_inv_phi) / target_inv_phi * 100

    print(f"\n  Per-level geometric mean survival: {per_level_survival:.6f}")
    print(f"  1/phi (phi-split target):          {target_inv_phi:.6f}")
    print(f"  Delta: {delta_phi:.1f}%")

    # Cumulative survival through hierarchy
    print(f"\n  Cumulative survival through hierarchy:")
    cum = 1.0
    for i, lv in enumerate(sorted(budgets_by_level.keys())):
        if not budgets_by_level[lv]:
            continue
        if i < len(level_geom_means):
            cum *= level_geom_means[i]
        d = i + 1
        print(f"    After level {lv}: {cum:.6f} (phi^{{-{d}}} = {INV_PHI**d:.6f})")

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

    # Test 3: negative trend (xi/P decreases with level)
    test3 = negative_trend
    print(f"\n  Test 3: xi/P decreases with level (Spearman rho < -0.3)")
    print(f"    Spearman rho: {trend_rho:.4f}")
    print(f"    -> {'VERIFIED' if test3 else 'NOT VERIFIED'}")

    # Test 4: per-level geometric mean survival ≈ 1/phi within 5%
    test4 = delta_phi < 5.0
    print(f"\n  Test 4: Per-level survival ≈ 1/phi within 5%")
    print(f"    Per-level geom mean: {per_level_survival:.6f}")
    print(f"    1/phi: {target_inv_phi:.6f}")
    print(f"    Delta: {delta_phi:.1f}%")
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
        'trend': {
            'spearman_rho': float(trend_rho),
            'spearman_p': float(trend_p),
            'negative_trend': bool(negative_trend),
        },
        'multiplicative_survival': {
            'per_level_geom_mean': float(per_level_survival),
            'level_geom_means': [float(g) for g in level_geom_means],
            'inv_phi': float(INV_PHI),
            'delta_phi_pct': float(delta_phi),
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
