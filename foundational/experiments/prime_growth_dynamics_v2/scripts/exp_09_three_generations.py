"""
Experiment 09: Three Generations Emergence
============================================

Tests whether exactly 3 stable resonance modes emerge from MED-
constrained PAC evolution — mirroring the 3 generations of fermions
in the Standard Model.

MED bound: depth ≤ 2, nodes ≤ 3. Under this constraint, PAC trees
can evolve in limited ways. The question: does iterating
PAC+SEC+MED produce exactly 3 qualitatively distinct stable
configurations?

Success criterion: PAC evolution under MED constraints produces
exactly 3 distinct stable attractors (or a clear reason why not).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from phase_engine import *

def enumerate_med_topologies(max_children=3, max_depth=2, current_depth=0):
    """
    Enumerate all possible tree topologies under MED constraints.
    Returns list of canonical topology strings.
    """
    if current_depth >= max_depth:
        return [f"L{current_depth}"]

    topologies = [f"L{current_depth}"]  # Can always be a leaf

    for nc in range(2, max_children + 1):
        child_topos = enumerate_med_topologies(max_children, max_depth, current_depth + 1)
        from itertools import combinations_with_replacement
        for combo in combinations_with_replacement(child_topos, nc):
            topo = f"N{nc}({','.join(sorted(combo))})"
            if topo not in topologies:
                topologies.append(topo)

    return topologies


def run():
    print("=" * 70)
    print("EXP 09: Three Generations from MED-Constrained PAC")
    print("=" * 70)

    # ================================================================
    # Test 1: Enumerate all MED-legal topologies
    # ================================================================
    print("\n--- Test 1: All MED-legal tree topologies ---")

    for mc in [2, 3]:
        for md in [1, 2, 3]:
            topos = enumerate_med_topologies(max_children=mc, max_depth=md)
            print(f"  mc={mc}, depth={md}: {len(topos)} topologies")
            if len(topos) <= 20:
                for t in topos:
                    print(f"    {t}")

    med_topos = enumerate_med_topologies(max_children=3, max_depth=2)
    print(f"\n  MED-strict (mc=3, d=2): {len(med_topos)} topologies")
    for t in med_topos:
        print(f"    {t}")

    # ================================================================
    # Test 2: Stability across different configurations
    # ================================================================
    print("\n--- Test 2: PAC stability landscape ---")

    n_iterations = 100
    configs = []
    results_list = []

    # Sweep mc and depth within MED bounds and slightly beyond
    for mc in [2, 3, 4]:
        for depth in [1, 2, 3]:
            for noise in [0.001, 0.01, 0.05]:
                result = evolve_pac_tree(
                    initial_value=1.0,
                    max_depth=depth,
                    max_children=mc,
                    n_iterations=n_iterations,
                    noise=noise,
                    seed=mc*1000 + depth*100 + int(noise*1000),
                )
                configs.append((mc, depth, noise))
                results_list.append(result)

    # Print summary
    print(f"  {'mc':>3} {'depth':>5} {'noise':>6} {'stability':>10} "
          f"{'conservation':>13} {'variance':>10} {'collapses':>10}")
    for (mc, depth, noise), result in zip(configs, results_list):
        med = "MED" if mc <= 3 and depth <= 2 else ""
        print(f"  {mc:3d} {depth:5d} {noise:6.3f} "
              f"{result['stability_score']:10.4f} "
              f"{result['mean_conservation_error']:13.6f} "
              f"{result['mean_variance_ratio']:10.4f} "
              f"{result['mean_collapses']:10.1f} {med}")

    # ================================================================
    # Test 3: Cluster analysis on stability landscape
    # ================================================================
    print("\n--- Test 3: Cluster analysis ---")

    # Feature vectors: [stability, conservation_error, variance_ratio]
    features = np.array([
        [r['stability_score'], r['mean_conservation_error'], r['mean_variance_ratio']]
        for r in results_list
    ])

    # Normalize
    feat_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

    # Simple k-means for k=2,3,4,5
    best_k_score = {}
    for k in [2, 3, 4, 5]:
        np.random.seed(42)
        indices = np.random.choice(len(feat_norm), k, replace=False)
        centers = feat_norm[indices].copy()

        for iteration in range(50):
            distances = np.array([[np.sum((f - c)**2) for c in centers]
                                 for f in feat_norm])
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([
                feat_norm[labels == j].mean(axis=0) if np.sum(labels == j) > 0
                else centers[j] for j in range(k)
            ])
            if np.allclose(new_centers, centers, atol=1e-6):
                break
            centers = new_centers

        inertia = sum(np.sum((feat_norm[labels == j] - centers[j])**2)
                      for j in range(k))
        cluster_sizes = [int(np.sum(labels == j)) for j in range(k)]

        best_k_score[k] = {
            'inertia': float(inertia),
            'cluster_sizes': cluster_sizes,
        }
        print(f"  k={k}: inertia={inertia:.2f}, sizes={cluster_sizes}")

    # Elbow detection
    inertias = [best_k_score[k]['inertia'] for k in [2, 3, 4, 5]]
    inertia_drops = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    elbow_ratios = [inertia_drops[i] / inertia_drops[i+1]
                    if inertia_drops[i+1] > 0 else 0
                    for i in range(len(inertia_drops)-1)]

    print(f"\n  Inertia drops: {[f'{d:.2f}' for d in inertia_drops]}")
    print(f"  Elbow ratios:  {[f'{r:.2f}' for r in elbow_ratios]}")

    if elbow_ratios:
        best_k_idx = np.argmax(elbow_ratios)
        best_k = [2, 3, 4, 5][best_k_idx + 1]
    else:
        best_k = 3

    print(f"  Best k (elbow): {best_k}")

    # ================================================================
    # Test 4: MED region characterization
    # ================================================================
    print("\n--- Test 4: MED region vs beyond ---")

    med_configs = [(mc, d, n) for (mc, d, n) in configs if mc <= 3 and d <= 2]
    non_med = [(mc, d, n) for (mc, d, n) in configs if mc > 3 or d > 2]

    med_stab = [results_list[i]['stability_score']
                for i, (mc, d, n) in enumerate(configs) if mc <= 3 and d <= 2]
    non_stab = [results_list[i]['stability_score']
                for i, (mc, d, n) in enumerate(configs) if mc > 3 or d > 2]

    print(f"  MED region:     stability = {np.mean(med_stab):.4f} ± {np.std(med_stab):.4f}")
    print(f"  Non-MED region: stability = {np.mean(non_stab):.4f} ± {np.std(non_stab):.4f}")

    # ================================================================
    # Results
    # ================================================================
    success = best_k == 3

    data = {
        'experiment': 'exp_09_three_generations',
        'hypothesis': 'MED-PAC produces exactly 3 stable modes',
        'n_topologies_enumerated': len(med_topos),
        'cluster_analysis': best_k_score,
        'best_k': best_k,
        'n_configs': len(configs),
        'med_stability_mean': float(np.mean(med_stab)),
        'non_med_stability_mean': float(np.mean(non_stab)),
        'success': success,
        'success_criterion': 'Exactly 3 stable attractor modes emerge',
    }

    print(f"\n{'='*70}")
    print(f"BEST CLUSTER COUNT: {best_k}")
    print(f"THREE GENERATIONS: {'YES' if success else 'NO'}")
    print(f"{'='*70}")

    save_results(data, 'exp_09_three_generations')
    return data


if __name__ == '__main__':
    run()
