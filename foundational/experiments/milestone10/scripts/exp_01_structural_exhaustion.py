"""
Milestone 10 -- Exp 01: Structural Exhaustion -- Self-Application x Symmetry

Block A: Uniqueness & Foundations

PURPOSE: Computationally verify the uniqueness argument from the M10 thesis (section 2).
Build a laboratory of candidate primitives parameterized by (self_applies, symmetric).
Show that only self-applying + symmetric systems produce stable hierarchical structure.

The model: a recursive coupling network where N nodes interact via matrix W.
  x' = tanh(W @ x)           -- state evolves under coupling
  W' = tanh(W @ W / n)       -- self-application: rule transforms itself (if enabled)
  W = W^T                    -- symmetry constraint (if enabled)

Mathematical prediction:
  - Fixed W: x converges to attractor, low-rank covariance (no hierarchy)
  - Self-applying + asymmetric W: complex eigenvalues drift -> chaos or collapse
  - Self-applying + symmetric W: real eigenvalues, bounded -> stable multi-scale

Tests:
  1. Case A elimination: non-self-applying systems produce no sustained hierarchy
  2. Case B elimination: self-applying but asymmetric systems diverge or collapse
  3. Case C survival: self-applying AND symmetric systems produce stable structure
  4. Exhaustion: 2x2 grid fully tested, only (yes,yes) cell survives

Builds on: M7 exp_01 (phi from self-reference), iddea.md section 2
Predicted: 3/4 (Case B weakest -- threshold matters)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    SelfApplicator, measure_hierarchical_structure,
    save_results, setup_experiment,
)

_, RESULTS_DIR = setup_experiment(__file__)

N_SYSTEMS = 500  # per quadrant
N_STEPS = 300
NODE_SIZE = 32


def run_quadrant(self_applies, symmetric, n_systems):
    """Run n_systems in one quadrant, return hierarchy statistics."""
    hierarchy_count = 0
    complexities = []
    active_scales_list = []

    for seed in range(n_systems):
        sa = SelfApplicator(seed, self_applies=self_applies,
                           symmetric=symmetric, size=NODE_SIZE)
        traj = sa.run(N_STEPS)
        result = measure_hierarchical_structure(traj)

        complexities.append(result['mean_complexity'])
        active_scales_list.append(result['n_active_scales'])

        if result['has_hierarchy']:
            hierarchy_count += 1

    frac = hierarchy_count / n_systems
    return {
        'hierarchy_count': hierarchy_count,
        'fraction': float(frac),
        'mean_complexity': float(np.mean(complexities)),
        'mean_active_scales': float(np.mean(active_scales_list)),
    }


def test1_case_a_elimination():
    """Case A: non-self-applying systems produce no sustained hierarchy."""
    print("\n" + "=" * 70)
    print("TEST 1: CASE A ELIMINATION -- No Self-Application")
    print("=" * 70)

    res_asym = run_quadrant(False, False, N_SYSTEMS)
    res_sym = run_quadrant(False, True, N_SYSTEMS)

    combined_hierarchy = res_asym['hierarchy_count'] + res_sym['hierarchy_count']
    frac_combined = combined_hierarchy / (2 * N_SYSTEMS)

    print(f"\n  Non-self-applying, asymmetric: {res_asym['hierarchy_count']}/{N_SYSTEMS} ({res_asym['fraction']:.1%})")
    print(f"    Mean complexity:  {res_asym['mean_complexity']:.3f}")
    print(f"    Mean scales:      {res_asym['mean_active_scales']:.1f}")
    print(f"  Non-self-applying, symmetric:  {res_sym['hierarchy_count']}/{N_SYSTEMS} ({res_sym['fraction']:.1%})")
    print(f"    Mean complexity:  {res_sym['mean_complexity']:.3f}")
    print(f"    Mean scales:      {res_sym['mean_active_scales']:.1f}")
    print(f"  Combined fraction: {frac_combined:.1%}")

    passed = frac_combined < 0.05
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {frac_combined:.1%} < 5%")

    return {
        'test': 'case_a_elimination',
        'n_systems': N_SYSTEMS,
        'asym': res_asym,
        'sym': res_sym,
        'fraction_combined': float(frac_combined),
        'passed': bool(passed),
    }


def test2_case_b_elimination():
    """Case B: self-applying but asymmetric systems diverge or collapse."""
    print("\n" + "=" * 70)
    print("TEST 2: CASE B ELIMINATION -- Self-Applying, No Symmetry")
    print("=" * 70)

    hierarchy_count = 0
    collapsed = 0
    chaotic = 0

    for seed in range(N_SYSTEMS):
        sa = SelfApplicator(seed, self_applies=True, symmetric=False, size=NODE_SIZE)
        traj = sa.run(N_STEPS)
        result = measure_hierarchical_structure(traj)

        # Check for saturation (tanh clips to +/-1)
        saturation = np.mean(np.abs(traj[-20:]) > 0.99)

        if saturation > 0.9 or result['mean_complexity'] < 1.5:
            collapsed += 1
        elif result['has_hierarchy']:
            hierarchy_count += 1
        else:
            chaotic += 1

    frac_hierarchy = hierarchy_count / N_SYSTEMS

    print(f"\n  Collapsed/saturated: {collapsed}/{N_SYSTEMS}")
    print(f"  Chaotic (no hierarchy): {chaotic}/{N_SYSTEMS}")
    print(f"  Stable hierarchical:    {hierarchy_count}/{N_SYSTEMS} ({frac_hierarchy:.1%})")

    passed = frac_hierarchy < 0.05
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {frac_hierarchy:.1%} < 5% hierarchical")

    return {
        'test': 'case_b_elimination',
        'n_systems': N_SYSTEMS,
        'collapsed': collapsed,
        'chaotic': chaotic,
        'hierarchy_count': hierarchy_count,
        'fraction_hierarchy': float(frac_hierarchy),
        'passed': bool(passed),
    }


def test3_case_c_survival():
    """Case C: self-applying AND symmetric systems produce stable structure."""
    print("\n" + "=" * 70)
    print("TEST 3: CASE C SURVIVAL -- Self-Applying + Symmetric")
    print("=" * 70)

    res = run_quadrant(True, True, N_SYSTEMS)

    print(f"\n  Stable hierarchical: {res['hierarchy_count']}/{N_SYSTEMS} ({res['fraction']:.1%})")
    print(f"  Mean complexity:     {res['mean_complexity']:.3f}")
    print(f"  Mean active scales:  {res['mean_active_scales']:.1f}")

    passed = res['fraction'] > 0.20
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {res['fraction']:.1%} > 20% threshold")

    return {
        'test': 'case_c_survival',
        'n_systems': N_SYSTEMS,
        'hierarchy_count': res['hierarchy_count'],
        'fraction': res['fraction'],
        'mean_complexity': res['mean_complexity'],
        'mean_active_scales': res['mean_active_scales'],
        'passed': bool(passed),
    }


def test4_exhaustion_grid():
    """Exhaustion: only (self_applies=True, symmetric=True) produces structure."""
    print("\n" + "=" * 70)
    print("TEST 4: EXHAUSTION -- 2x2 Grid Complete")
    print("=" * 70)

    grid = {}
    n_per_cell = 300

    for sa_flag in [False, True]:
        for sym_flag in [False, True]:
            label = f"sa={sa_flag},sym={sym_flag}"
            res = run_quadrant(sa_flag, sym_flag, n_per_cell)
            grid[label] = res
            print(f"  {label}: {res['hierarchy_count']}/{n_per_cell} ({res['fraction']:.1%})")
            print(f"    complexity={res['mean_complexity']:.2f}, scales={res['mean_active_scales']:.1f}")

    cc_frac = grid['sa=True,sym=True']['fraction']
    other_fracs = [grid[k]['fraction'] for k in grid if k != 'sa=True,sym=True']
    max_other = max(other_fracs)

    dominates = cc_frac > max_other * 3 and cc_frac > 0.15

    print(f"\n  Case C fraction:    {cc_frac:.1%}")
    print(f"  Best other:         {max_other:.1%}")
    print(f"  Ratio:              {cc_frac / max(max_other, 0.001):.1f}x")
    print(f"\n  -> {'PASS' if dominates else 'FAIL'}: Case C dominates {cc_frac:.1%} vs {max_other:.1%}")

    return {
        'test': 'exhaustion_grid',
        'grid': grid,
        'cc_fraction': float(cc_frac),
        'max_other_fraction': float(max_other),
        'dominance_ratio': float(cc_frac / max(max_other, 0.001)),
        'passed': bool(dominates),
    }


def main():
    print("=" * 70)
    print("MILESTONE 10 - EXP 01: STRUCTURAL EXHAUSTION")
    print("Block A: Uniqueness & Foundations")
    print("=" * 70)

    r1 = test1_case_a_elimination()
    r2 = test2_case_b_elimination()
    r3 = test3_case_c_survival()
    r4 = test4_exhaustion_grid()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, r in enumerate(tests, 1):
        print(f"  Test {i} ({r['test']}): {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/{len(tests)}")

    results = {
        'experiment': 'exp_01_structural_exhaustion',
        'milestone': 10,
        'block': 'A',
        'tests': {r['test']: r for r in tests},
        'score': f"{n_passed}/{len(tests)}",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_01_structural_exhaustion', RESULTS_DIR)


if __name__ == '__main__':
    main()
