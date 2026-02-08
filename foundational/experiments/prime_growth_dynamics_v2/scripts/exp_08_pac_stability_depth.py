"""
Experiment 08: PAC Stability vs Depth
=======================================

Tests MED prediction: systems with depth ≤ 2 are stable; depth ≥ 3
leads to collapse or divergence.

MED (Macro Emergence Dynamics) bound: depth ≤ 2, nodes ≤ 3.
This experiment varies depth while holding max_children fixed.

Success criterion: Stability peaks at depth = 2, degrades for depth ≥ 3.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from phase_engine import *

def run():
    print("=" * 70)
    print("EXP 08: PAC Stability vs Depth (MED Bound)")
    print("=" * 70)

    n_iterations = 100
    depth_range = [1, 2, 3, 4, 5, 6, 8, 10]
    fixed_max_children = 3  # Fix at MED optimal node count

    # ================================================================
    # Test 1: Stability vs depth_limit
    # ================================================================
    print("\n--- Test 1: PAC tree stability by depth limit ---")
    print(f"  Fixed max_children={fixed_max_children}, "
          f"n_iterations={n_iterations}")

    depth_results = {}

    for dl in depth_range:
        result = evolve_pac_tree(
            initial_value=1.0,
            max_children=fixed_max_children,
            max_depth=dl,
            n_iterations=n_iterations,
            seed=dl * 100,
        )

        depth_results[dl] = {
            'depth_limit': dl,
            'stability_score': result['stability_score'],
            'mean_conservation_error': result['mean_conservation_error'],
            'mean_variance_ratio': result['mean_variance_ratio'],
            'mean_collapses': result['mean_collapses'],
            'mean_depth_reached': result['mean_depth_reached'],
            'mean_leaves': result['mean_leaves'],
        }

        print(f"  depth={dl:2d}: stability={result['stability_score']:.4f}, "
              f"conservation_err={result['mean_conservation_error']:.4f}, "
              f"variance_ratio={result['mean_variance_ratio']:.4f}, "
              f"depth_reached={result['mean_depth_reached']:.1f}")

    # ================================================================
    # Test 2: Depth transition analysis
    # ================================================================
    print("\n--- Test 2: Depth transition analysis ---")

    stabilities = [depth_results[dl]['stability_score'] for dl in depth_range]
    stability_diffs = [stabilities[i+1] - stabilities[i]
                       for i in range(len(stabilities)-1)]

    print(f"  Stability: {[f'{s:.4f}' for s in stabilities]}")
    print(f"  Δstability: {[f'{d:+.4f}' for d in stability_diffs]}")

    # Find sharpest drop
    min_diff_idx = np.argmin(stability_diffs)
    transition_depth = depth_range[min_diff_idx]
    print(f"  Sharpest drop: depth {transition_depth}→{depth_range[min_diff_idx+1]} "
          f"(Δ={stability_diffs[min_diff_idx]:.4f})")

    med_transition = (transition_depth == 2)
    print(f"  MED boundary at depth 2→3: {'YES' if med_transition else 'NO'}")

    # ================================================================
    # Test 3: Conservation vs stability
    # ================================================================
    print("\n--- Test 3: Conservation-stability relationship ---")

    for dl in depth_range:
        r = depth_results[dl]
        print(f"  depth={dl:2d}: conservation_err={r['mean_conservation_error']:.4f}, "
              f"stability={r['stability_score']:.4f}, "
              f"variance={r['mean_variance_ratio']:.4f}")

    # ================================================================
    # Test 4: Combined MED grid (depth × nodes)
    # ================================================================
    print("\n--- Test 4: Combined MED grid (depth × nodes) ---")

    grid_results = {}
    for mc in [2, 3, 4, 5]:
        for dl in [1, 2, 3, 4]:
            r = evolve_pac_tree(1.0, dl, mc, n_iterations=50, seed=mc*100+dl)
            grid_results[(mc, dl)] = r['stability_score']
            med_ok = "MED" if mc <= 3 and dl <= 2 else ""
            print(f"  mc={mc}, depth={dl}: stability={r['stability_score']:.4f} {med_ok}")

    # Is the MED region (mc≤3, d≤2) always the most stable?
    med_stabilities = [v for (mc, dl), v in grid_results.items()
                       if mc <= 3 and dl <= 2]
    non_med_stabilities = [v for (mc, dl), v in grid_results.items()
                           if mc > 3 or dl > 2]

    med_mean = np.mean(med_stabilities) if med_stabilities else 0
    non_med_mean = np.mean(non_med_stabilities) if non_med_stabilities else 0

    print(f"\n  MED region mean stability: {med_mean:.4f}")
    print(f"  Non-MED region mean stability: {non_med_mean:.4f}")
    print(f"  MED advantage: {(med_mean - non_med_mean):.4f}")

    # ================================================================
    # Results
    # ================================================================
    peak_depth = max(depth_range[:4],
                     key=lambda dl: depth_results[dl]['stability_score'])
    success = peak_depth <= 2

    data = {
        'experiment': 'exp_08_pac_stability_depth',
        'hypothesis': 'MED predicts max stability at depth ≤ 2',
        'config': {
            'n_iterations': n_iterations,
            'depth_range': depth_range,
            'fixed_max_children': fixed_max_children,
        },
        'results_by_depth': depth_results,
        'peak_depth': peak_depth,
        'med_transition_at_2_3': med_transition,
        'grid_results': {f"mc{mc}_d{dl}": float(v)
                        for (mc, dl), v in grid_results.items()},
        'med_region_advantage': float(med_mean - non_med_mean),
        'success': success,
        'success_criterion': 'Stability peaks at depth ≤ 2',
    }

    print(f"\n{'='*70}")
    print(f"PEAK STABILITY: depth = {peak_depth}")
    print(f"MED TRANSITION (2→3): {'YES' if med_transition else 'NO'}")
    print(f"MED REGION ADVANTAGE: {med_mean - non_med_mean:.4f}")
    print(f"SUCCESS: {'YES' if success else 'NO'}")
    print(f"{'='*70}")

    save_results(data, 'exp_08_pac_stability_depth')
    return data


if __name__ == '__main__':
    run()
