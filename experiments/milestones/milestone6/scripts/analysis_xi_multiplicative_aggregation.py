"""
Milestone 6 -- Analysis: Xi Multiplicative Aggregation

PURPOSE: Test whether the xi non-additivity "failure" resolves when using
the correct aggregation rule from the corpus.

M7 exp_03 established:
  - Xi = gamma + ln(phi) = 1.058 per boundary crossing
  - Survival per boundary = e^{-Xi} = e^{-gamma} * (1/phi)
  - Across N serial boundaries: survival = e^{-N*Xi} (MULTIPLICATIVE)

Hypothesis: The PAC budget's xi/P captures ONLY the splitting cost ln(phi),
not the counting cost gamma. Therefore:
  -ln(1 - xi/P) ≈ ln(phi) per boundary
  Adding gamma from hierarchy structure gives Xi.

If true:
  1. Per-boundary -ln(1-xi/P) ≈ ln(phi) within 5%
  2. Per-boundary -ln(1-xi/P) + gamma ≈ Xi within 5%
  3. The 35% "failure" in exp_08 disappears under multiplicative aggregation
  4. Cross-level survival product matches e^{-N*Xi}
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
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
    _get_eigenbasis, pac_budget,
)
from _shared import (
    load_baseline, build_lattice_adjacency,
    get_parent_children_data, K_MODES,
)

RESULTS_DIR = M6_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("M6 ANALYSIS: XI MULTIPLICATIVE AGGREGATION")
    print("Does -ln(1-xi/P) = ln(phi) per boundary?")
    print("=" * 70)

    P_field, A_field, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    adjacency = build_lattice_adjacency(C)
    state_flat = C.ravel()

    # ============================================================
    # COLLECT PAC BUDGETS AT EVERY BOUNDARY
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: PAC BUDGET AT EVERY BOUNDARY")
    print("=" * 60)

    budgets = []

    for (level, pid), pidx, children, L_parent, state_parent in \
            get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):

        eigenvalues, eigenvectors = _get_eigenbasis(L_parent, state_parent, k=K_MODES)
        budget = pac_budget(state_parent, L_parent, eigenvectors, eigenvalues)
        budget['level'] = level
        budget['parent_id'] = pid
        budget['parent_size'] = len(pidx)
        budget['n_children'] = len(children)
        budgets.append(budget)

    print(f"  Total scope boundaries: {len(budgets)}")

    # ============================================================
    # STEP 2: COMPUTE INFO CONTENT PER BOUNDARY
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: INFO CONTENT = -ln(1 - xi/P)")
    print("=" * 60)

    for b in budgets:
        xi_frac = b['xi_fraction']
        if xi_frac < 1:
            b['info_content'] = -np.log(1 - xi_frac)
        else:
            b['info_content'] = float('inf')

        # Also compute odds ratio
        if xi_frac < 1:
            b['odds_ratio'] = xi_frac / (1 - xi_frac)
        else:
            b['odds_ratio'] = float('inf')

    # Group by level
    by_level = {}
    for b in budgets:
        lv = b['level']
        if lv not in by_level:
            by_level[lv] = []
        by_level[lv].append(b)

    sorted_levels = sorted(by_level.keys())

    print(f"\n  {'Level':<8} {'n':>4} {'mean xi/P':>12} {'-ln(1-xi/P)':>14} "
          f"{'ln(phi)':>10} {'delta%':>8}")
    print(f"  {'-'*60}")

    all_info = []
    for lv in sorted_levels:
        entries = by_level[lv]
        xi_fracs = [b['xi_fraction'] for b in entries]
        infos = [b['info_content'] for b in entries if b['info_content'] != float('inf')]
        mean_xi = np.mean(xi_fracs)
        mean_info = np.mean(infos) if infos else 0
        delta = abs(mean_info - LN_PHI) / LN_PHI * 100

        all_info.extend(infos)

        print(f"  Level {lv:<4} {len(entries):>4} {mean_xi:>12.6f} {mean_info:>14.6f} "
              f"{LN_PHI:>10.6f} {delta:>7.1f}%")

    overall_info = np.mean(all_info)
    overall_delta = abs(overall_info - LN_PHI) / LN_PHI * 100
    print(f"\n  Overall mean -ln(1-xi/P): {overall_info:.6f}")
    print(f"  ln(phi):                   {LN_PHI:.6f}")
    print(f"  Delta:                     {overall_delta:.1f}%")

    # ============================================================
    # STEP 3: ADD COUNTING COST (GAMMA)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: SPLITTING + COUNTING = Xi?")
    print("-ln(1-xi/P) + gamma ≈ Xi?")
    print("=" * 60)

    # Per-level: info_content + gamma should give Xi
    print(f"\n  {'Level':<8} {'-ln(1-xi/P)':>14} {'+ gamma':>10} {'= total':>10} "
          f"{'Xi':>8} {'delta%':>8}")
    print(f"  {'-'*60}")

    for lv in sorted_levels:
        entries = by_level[lv]
        infos = [b['info_content'] for b in entries if b['info_content'] != float('inf')]
        mean_info = np.mean(infos) if infos else 0
        total = mean_info + GAMMA_EM
        delta = abs(total - XI_BALANCE) / XI_BALANCE * 100
        print(f"  Level {lv:<4} {mean_info:>14.6f} {GAMMA_EM:>10.6f} {total:>10.6f} "
              f"{XI_BALANCE:>8.6f} {delta:>7.1f}%")

    overall_total = overall_info + GAMMA_EM
    overall_total_delta = abs(overall_total - XI_BALANCE) / XI_BALANCE * 100
    print(f"\n  Overall: -ln(1-xi/P) + gamma = {overall_total:.6f}")
    print(f"  Xi_balance:                     {XI_BALANCE:.6f}")
    print(f"  Delta:                          {overall_total_delta:.1f}%")

    # ============================================================
    # STEP 4: MULTIPLICATIVE SURVIVAL THROUGH LEVELS
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: MULTIPLICATIVE SURVIVAL PRODUCT")
    print("Product of (1-xi/P) across serial levels vs e^{-N*Xi}")
    print("=" * 60)

    # Each level's mean survival = mean(1 - xi/P) at that level
    # Serial product across levels gives total survival
    level_survivals = []
    cum_survival = 1.0
    for lv in sorted_levels:
        entries = by_level[lv]
        xi_fracs = [b['xi_fraction'] for b in entries]
        mean_survival = np.mean([1 - xf for xf in xi_fracs])
        cum_survival *= mean_survival
        n_levels_so_far = lv - sorted_levels[0] + 1

        # Predicted from Xi
        predicted_survival = np.exp(-n_levels_so_far * XI_BALANCE)

        # Predicted from ln(phi) only (splitting, no counting)
        predicted_split_only = np.exp(-n_levels_so_far * LN_PHI)

        print(f"  After level {lv}:")
        print(f"    Mean survival at this level: {mean_survival:.6f}")
        print(f"    Cumulative survival: {cum_survival:.6f}")
        print(f"    e^(-N*Xi):           {predicted_survival:.6f} "
              f"({abs(cum_survival - predicted_survival)/predicted_survival*100:.1f}% off)")
        print(f"    e^(-N*ln(phi)):      {predicted_split_only:.6f}")

        level_survivals.append({
            'level': lv,
            'mean_survival': float(mean_survival),
            'cumulative': float(cum_survival),
            'predicted_Xi': float(predicted_survival),
            'predicted_lnphi': float(predicted_split_only),
        })

    # ============================================================
    # STEP 5: THE COUNTING COST FROM HIERARCHY STRUCTURE
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: WHERE DOES GAMMA COME FROM?")
    print("Gamma = counting cost = discrete enumeration overhead")
    print("=" * 60)

    # How many children does each parent have?
    for lv in sorted_levels:
        entries = by_level[lv]
        n_children = [b['n_children'] for b in entries]
        parent_sizes = [b['parent_size'] for b in entries]

        # Harmonic number approximation for counting cost
        # H_n - ln(n) -> gamma as n -> inf
        # For finite n: gamma_n = H_n - ln(n)
        gamma_ns = []
        for b in entries:
            n = b['parent_size']
            if n > 1:
                H_n = sum(1.0 / k for k in range(1, min(n, 1000) + 1))
                gamma_n = H_n - np.log(n)
                gamma_ns.append(gamma_n)

        if gamma_ns:
            mean_gamma_n = np.mean(gamma_ns)
            delta_g = abs(mean_gamma_n - GAMMA_EM) / GAMMA_EM * 100
            print(f"  Level {lv}: mean H_n - ln(n) = {mean_gamma_n:.6f} "
                  f"(gamma = {GAMMA_EM:.6f}, delta = {delta_g:.1f}%)")
            print(f"    Mean parent size: {np.mean(parent_sizes):.0f}, "
                  f"mean n_children: {np.mean(n_children):.1f}")

    # ============================================================
    # STEP 6: DISTRIBUTION OF INFO CONTENT
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 6: DISTRIBUTION OF -ln(1-xi/P)")
    print("=" * 60)

    valid_infos = [i for i in all_info if np.isfinite(i)]
    if valid_infos:
        print(f"  n = {len(valid_infos)}")
        print(f"  Mean:   {np.mean(valid_infos):.6f} (target: {LN_PHI:.6f})")
        print(f"  Median: {np.median(valid_infos):.6f}")
        print(f"  Std:    {np.std(valid_infos):.6f}")
        print(f"  CV:     {np.std(valid_infos) / np.mean(valid_infos):.4f}")
        print(f"  Min:    {min(valid_infos):.6f}")
        print(f"  Max:    {max(valid_infos):.6f}")

        # What fraction are within 20% of ln(phi)?
        within_20 = sum(1 for i in valid_infos
                        if abs(i - LN_PHI) / LN_PHI < 0.20)
        print(f"  Within 20% of ln(phi): {within_20}/{len(valid_infos)} "
              f"({100*within_20/len(valid_infos):.0f}%)")

    # ============================================================
    # SYNTHESIS
    # ============================================================
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    # Test 1: -ln(1-xi/P) ≈ ln(phi) within 5%?
    test1 = overall_delta < 5.0
    print(f"\n  1. Per-boundary -ln(1-xi/P) ≈ ln(phi) within 5%?")
    print(f"     Mean: {overall_info:.6f}, target: {LN_PHI:.6f}, delta: {overall_delta:.1f}%")
    print(f"     -> {'YES' if test1 else 'NO'}")

    # Test 2: -ln(1-xi/P) + gamma ≈ Xi within 5%?
    test2 = overall_total_delta < 5.0
    print(f"\n  2. -ln(1-xi/P) + gamma ≈ Xi within 5%?")
    print(f"     Total: {overall_total:.6f}, Xi: {XI_BALANCE:.6f}, delta: {overall_total_delta:.1f}%")
    print(f"     -> {'YES' if test2 else 'NO'}")

    # Test 3: Cumulative survival matches e^{-N*Xi} better than additive sum?
    # Compare additive error (from exp_08: 35%) with multiplicative error
    if level_survivals:
        final = level_survivals[-1]
        mult_error = abs(final['cumulative'] - final['predicted_Xi']) / final['predicted_Xi'] * 100
        # Additive error was 35% (from exp_08 test 4)
        test3 = mult_error < 35.0  # multiplicative should be closer than additive
        print(f"\n  3. Multiplicative aggregation beats additive (35%)?")
        print(f"     Multiplicative error: {mult_error:.1f}%")
        print(f"     Additive error: ~35% (from exp_08)")
        print(f"     -> {'YES' if test3 else 'NO'}")
    else:
        test3 = False

    # Test 4: H_n - ln(n) ≈ gamma (counting cost derivable from hierarchy)?
    all_gamma_ns = []
    for b in budgets:
        n = b['parent_size']
        if n > 1:
            H_n = sum(1.0 / k for k in range(1, min(n, 1000) + 1))
            all_gamma_ns.append(H_n - np.log(n))
    mean_gamma_from_hierarchy = np.mean(all_gamma_ns) if all_gamma_ns else 0
    gamma_delta = abs(mean_gamma_from_hierarchy - GAMMA_EM) / GAMMA_EM * 100
    test4 = gamma_delta < 5.0
    print(f"\n  4. H_n - ln(n) from hierarchy ≈ gamma within 5%?")
    print(f"     Mean H_n - ln(n): {mean_gamma_from_hierarchy:.6f}")
    print(f"     gamma: {GAMMA_EM:.6f}, delta: {gamma_delta:.1f}%")
    print(f"     -> {'YES' if test4 else 'NO'}")

    n_yes = sum([test1, test2, test3, test4])
    print(f"\n  RESULT: {n_yes}/4 tests pass")

    if n_yes >= 3:
        print("\n  CONCLUSION: Xi decomposes cleanly into:")
        print(f"    - SPLITTING cost: -ln(1-xi/P) ≈ ln(phi) = {LN_PHI:.4f}")
        print(f"    - COUNTING cost:  H_n - ln(n) ≈ gamma = {GAMMA_EM:.4f}")
        print(f"    - Total per boundary: Xi = {XI_BALANCE:.4f}")
        print(f"    - Aggregation: MULTIPLICATIVE (survival = e^{{-N*Xi}})")
        print(f"\n  The 'failure' in exp_08 was using additive aggregation")
        print(f"  on a multiplicative process. The correct rule was already")
        print(f"  in the corpus (M7 exp_03).")

    # -- Save results --
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'analysis': 'xi_multiplicative_aggregation',
        'milestone': 6,
        'n_boundaries': len(budgets),
        'per_level_info_content': {
            str(lv): {
                'n': len(by_level[lv]),
                'mean_xi_frac': float(np.mean([b['xi_fraction'] for b in by_level[lv]])),
                'mean_info_content': float(np.mean([b['info_content'] for b in by_level[lv]
                                                     if b['info_content'] != float('inf')])),
            }
            for lv in sorted_levels
        },
        'overall': {
            'mean_info_content': float(overall_info),
            'ln_phi': float(LN_PHI),
            'delta_pct': float(overall_delta),
            'info_plus_gamma': float(overall_total),
            'Xi_balance': float(XI_BALANCE),
            'total_delta_pct': float(overall_total_delta),
        },
        'survival_cascade': level_survivals,
        'counting_cost': {
            'mean_gamma_n': float(mean_gamma_from_hierarchy),
            'gamma_EM': float(GAMMA_EM),
            'delta_pct': float(gamma_delta),
        },
        'tests': {
            'info_content_eq_lnphi': test1,
            'info_plus_gamma_eq_Xi': test2,
            'multiplicative_beats_additive': test3,
            'hierarchy_gives_gamma': test4,
            'total_yes': n_yes,
        },
        'timestamp': datetime.now().isoformat(),
    }

    outpath = RESULTS_DIR / f"analysis_xi_multiplicative_{ts}.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
