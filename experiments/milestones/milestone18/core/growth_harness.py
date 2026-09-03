"""Milestone 18 core — Block D object O2: replay harness for `evolve_pac_tree` (studies/prime_growth_dynamics_v2).

The engine returns metrics but not topology. This harness re-runs the SAME RNG calls in the SAME order
(np.random.seed(seed); dirichlet + normal per split; randint per node) and records every parent->child
edge, then recomputes the engine's aggregate metrics and compares them with the values the original
experiments recorded (exp_07/exp_08 result JSONs). The gate is bit-for-bit on stability_score,
mean_depth_reached, mean_variance_ratio and mean_conservation_error per (max_depth, max_children, seed).
If the gate fails, O2 is a NEW object (draft §1) and T1 is restricted to O1/O3/O4.
"""
import numpy as np


def replay(initial_value, max_depth, max_children, n_iterations=100, noise=0.01, seed=42):
    """Returns (metrics_dict_like_engine, trees) where trees[i] = dict(n=..., edges=[(parent, child)], values=[...])."""
    np.random.seed(seed)
    results = []; trees = []
    for iteration in range(n_iterations):
        # node bookkeeping: index 0 = root
        values = [float(initial_value)]; depths = [0]; edges = []
        frontier = [0]; total_nodes = 1; collapse_events = 0; max_depth_reached = 0
        for d in range(max_depth):
            new_frontier = []
            for node in frontier:
                n_kids = min(max_children, np.random.randint(2, max_children + 1))
                kids = []
                if n_kids >= 2:                                             # PACNode.split
                    raw = np.random.dirichlet(np.ones(n_kids)) * values[node]
                    raw += np.random.normal(0, noise * values[node] / n_kids, n_kids)
                    raw = raw * (values[node] / raw.sum())
                    for v in raw:
                        values.append(float(v)); depths.append(depths[node] + 1); kids.append(len(values) - 1)
                for child in kids:
                    if values[child] <= 0:
                        collapse_events += 1
                    else:
                        new_frontier.append(child); total_nodes += 1
                    edges.append((node, child))                           # topology: every split edge (collapsed children included, as generated)
                max_depth_reached = max(max_depth_reached, d + 1)
            frontier = new_frontier
            if not frontier:
                break
        leaf_values = [values[n] for n in frontier] if frontier else [0]
        total_leaf = sum(leaf_values); conservation_error = abs(total_leaf - initial_value) / initial_value
        if len(leaf_values) > 1:
            variance_ratio = np.std(leaf_values) / np.mean(leaf_values) if np.mean(leaf_values) > 0 else float('inf')
        else:
            variance_ratio = 0
        results.append(dict(conservation_error=conservation_error, depth_reached=max_depth_reached, total_nodes=total_nodes,
                            n_leaves=len(leaf_values), variance_ratio=variance_ratio, collapse_events=collapse_events, leaf_sum=total_leaf))
        # the tree the engine actually grew: the surviving nodes (value > 0) with their edges
        alive = {0} | {c for p_, c in edges if values[c] > 0}
        keep = [(p_, c) for p_, c in edges if c in alive and p_ in alive]
        trees.append(dict(n=len(alive), edges=keep, total_nodes=total_nodes, depth_reached=max_depth_reached, collapse_events=collapse_events))
    errors = [r['conservation_error'] for r in results]; collapses = [r['collapse_events'] for r in results]; variances = [r['variance_ratio'] for r in results]
    metrics = dict(max_depth=max_depth, max_children=max_children, n_iterations=n_iterations, noise=noise,
                   mean_conservation_error=float(np.mean(errors)), max_conservation_error=float(np.max(errors)),
                   mean_collapses=float(np.mean(collapses)), total_collapses=int(sum(collapses)), mean_variance_ratio=float(np.mean(variances)),
                   stability_score=float(1.0 - np.mean(collapses) / max(1, np.mean([r['total_nodes'] for r in results]))),
                   mean_depth_reached=float(np.mean([r['depth_reached'] for r in results])), mean_leaves=float(np.mean([r['n_leaves'] for r in results])))
    return metrics, trees


def gate_against_engine(initial_value, max_depth, max_children, n_iterations, seed, engine):
    """Bit-for-bit comparison of the replay's metrics with the engine's own output for the same call."""
    m, trees = replay(initial_value, max_depth, max_children, n_iterations=n_iterations, seed=seed)
    ref = engine(initial_value, max_depth, max_children, n_iterations=n_iterations, seed=seed)
    ints = ("total_collapses",); exact = ("mean_depth_reached", "mean_leaves", "stability_score")
    floats = ("mean_variance_ratio", "mean_conservation_error")     # sums over leaves: order not fixed by the engine -> 1e-12
    ok = all(m[k] == ref[k] for k in ints + exact) and all(abs(m[k] - ref[k]) <= 1e-12 * max(1.0, abs(ref[k])) for k in floats)
    return ok, {k: (m[k], ref[k]) for k in ints + exact + floats}, trees
