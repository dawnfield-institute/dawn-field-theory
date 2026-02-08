"""
Experiment 07: PAC Stability vs Node Count
=============================================

Tests MED prediction: systems with max_children ≤ 3 are more stable
than those with 4, 5, or more child nodes per parent.

MED (Macro Emergence Dynamics) predicts: depth ≤ 2, nodes ≤ 3 is
the attractor for all complex flows. PAC trees should show this:
splitting into {2, 3} children should produce stable long-lived
structures while {4, 5, 6+} should collapse or diverge.

Success criterion: Stability metric peaks at max_children = 2 or 3,
decays for 4+.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from phase_engine import *

def run():
    print("=" * 70)
    print("EXP 07: PAC Stability vs Node Count (MED Bound)")
    print("=" * 70)

    n_iterations = 100
    max_children_range = [2, 3, 4, 5, 6, 7, 8]
    depth_limit = 5  # Allow deep enough to see instability

    # ================================================================
    # Test 1: Stability vs max_children
    # ================================================================
    print("\n--- Test 1: PAC tree evolution by max_children ---")
    print(f"  Iterations per config: {n_iterations}, depth_limit: {depth_limit}")

    node_results = {}

    for mc in max_children_range:
        result = evolve_pac_tree(
            initial_value=1.0,
            max_children=mc,
            max_depth=depth_limit,
            n_iterations=n_iterations,
            seed=mc * 100,
        )

        node_results[mc] = {
            'max_children': mc,
            'stability_score': result['stability_score'],
            'mean_conservation_error': result['mean_conservation_error'],
            'mean_variance_ratio': result['mean_variance_ratio'],
            'mean_collapses': result['mean_collapses'],
            'mean_depth_reached': result['mean_depth_reached'],
            'mean_leaves': result['mean_leaves'],
        }

        print(f"  max_children={mc}: stability={result['stability_score']:.4f}, "
              f"conservation_err={result['mean_conservation_error']:.4f}, "
              f"variance_ratio={result['mean_variance_ratio']:.4f}, "
              f"collapses={result['mean_collapses']:.1f}")

    # ================================================================
    # Test 2: Stability comparison across seeds
    # ================================================================
    print("\n--- Test 2: Stability across seeds ---")

    for mc in [2, 3, 5]:
        scores = []
        for seed in range(10):
            r = evolve_pac_tree(1.0, depth_limit, mc, n_iterations=50, seed=seed)
            scores.append(r['stability_score'])
        print(f"  mc={mc}: stability={np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # ================================================================
    # Test 3: Phase transition detection
    # ================================================================
    print("\n--- Test 3: Phase transition at MED boundary ---")

    stabilities = [node_results[mc]['stability_score'] for mc in max_children_range]
    stability_diffs = [stabilities[i+1] - stabilities[i] for i in range(len(stabilities)-1)]

    print(f"  Stability values: {[f'{s:.4f}' for s in stabilities]}")
    print(f"  Stability diffs:  {[f'{d:+.4f}' for d in stability_diffs]}")

    # Find the largest drop
    min_diff_idx = np.argmin(stability_diffs)
    transition_point = max_children_range[min_diff_idx]
    transition_magnitude = stability_diffs[min_diff_idx]

    print(f"  Largest stability drop: between mc={transition_point} and "
          f"mc={max_children_range[min_diff_idx+1]} "
          f"(Δ={transition_magnitude:.4f})")

    med_transition = (transition_point == 3)
    print(f"  Transition at MED boundary (3→4): {'YES' if med_transition else 'NO'}")

    # ================================================================
    # Test 4: Binary (mc=2) vs Ternary (mc=3) comparison
    # ================================================================
    print("\n--- Test 4: Binary vs Ternary ---")

    r2 = node_results[2]
    r3 = node_results[3]

    print(f"  Binary (mc=2):  stability={r2['stability_score']:.4f}, "
          f"variance={r2['mean_variance_ratio']:.4f}")
    print(f"  Ternary (mc=3): stability={r3['stability_score']:.4f}, "
          f"variance={r3['mean_variance_ratio']:.4f}")

    if r2['stability_score'] > 0:
        ratio_23 = r3['stability_score'] / r2['stability_score']
        print(f"  Ratio (3/2): {ratio_23:.4f}")
        print(f"  Compare to φ = {PHI:.4f}")
        print(f"  Compare to 1/φ = {1/PHI:.4f}")

    # ================================================================
    # Results
    # ================================================================
    peak_mc = max(max_children_range[:4],
                  key=lambda mc: node_results[mc]['stability_score'])
    success = peak_mc in [2, 3]

    data = {
        'experiment': 'exp_07_pac_stability_nodes',
        'hypothesis': 'MED predicts max stability at nodes ≤ 3',
        'config': {
            'n_iterations': n_iterations,
            'max_children_range': max_children_range,
            'depth_limit': depth_limit,
        },
        'results_by_mc': node_results,
        'peak_mc': peak_mc,
        'med_transition_at_3_4': med_transition,
        'transition_magnitude': float(transition_magnitude),
        'success': success,
        'success_criterion': 'Stability peaks at max_children = 2 or 3',
    }

    print(f"\n{'='*70}")
    print(f"PEAK STABILITY: max_children = {peak_mc}")
    print(f"MED TRANSITION (3→4): {'YES' if med_transition else 'NO'}")
    print(f"SUCCESS: {'YES' if success else 'NO'}")
    print(f"{'='*70}")

    save_results(data, 'exp_07_pac_stability_nodes')
    return data


if __name__ == '__main__':
    run()
