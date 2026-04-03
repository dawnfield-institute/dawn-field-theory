"""
Milestone 7 -- Exp 06: Symmetry Breaking as Symmetry-Seeking

Block C: Consequences

HYPOTHESIS: Every local symmetry break improves global phi-balance, regardless
of the break mechanism. The phi-ratio split is the OPTIMAL break.

The previous version was tautological: it explicitly split at phi ratio then
checked if phi-balance improved. This version tests non-tautologically:

1. Various break mechanisms (random, equal, Fibonacci, phi) all improve
   phi-balance vs uniform — because uniform is MAXIMALLY unbalanced
   (exp_05 showed this)
2. The phi-ratio split produces the best phi-balance among alternatives
3. The improvement holds at every cascade level, not just the first
4. Results hold across graph topologies

The DFT interpretation: symmetry breaking isn't moving AWAY from symmetry —
ANY break moves TOWARD phi-balance because the uniform state is the
furthest from phi-structured. Nature's mechanism (multi-scale drive)
produces the optimal break (phi ratio), but even "wrong" breaks improve.

Tests:
  1. ALL break types improve phi-balance vs uniform (>= 4/5 types)
  2. Phi-ratio break gives highest phi-balance (or within 2% of best)
  3. Multi-level cascade: phi-balance improves at each level (>= 75%)
  4. Results hold across topologies (>= 2/3 graphs)
"""

import sys
import numpy as np
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
M7_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M7_ROOT))

from core.symmetry import (PHI, INV_PHI, save_results,
                           build_ring, build_torus, build_random_regular)

RESULTS_DIR = M7_ROOT / "results"


def phi_balance_score(state, L, n_levels=4):
    """
    Phi-balance score: how close the hierarchical D/S ratios are to phi.
    Uses spectral bisection to define hierarchy.
    Score = 1 - mean(|R - phi| / phi) across levels.
    """
    n = len(state)
    all_nodes = list(range(n))
    current_groups = [all_nodes]
    deviations = []

    for level in range(1, n_levels + 1):
        new_groups = []
        for group in current_groups:
            if len(group) < 4:
                new_groups.append(group)
                continue

            sub_L = L[np.ix_(group, group)]
            eigs, vecs = np.linalg.eigh(sub_L)
            fiedler = vecs[:, 1]

            half1 = [group[i] for i in range(len(group)) if fiedler[i] >= 0]
            half2 = [group[i] for i in range(len(group)) if fiedler[i] < 0]

            if not half1 or not half2:
                new_groups.append(group)
                continue

            s1 = np.sum(state[half1])
            s2 = np.sum(state[half2])
            D = max(s1, s2)
            S = min(s1, s2)
            total = D + S

            if D > 1e-15 and S > 1e-15:
                R = total / D  # P/D — should be phi
                dev = abs(R - PHI) / PHI
                deviations.append(dev)

            new_groups.extend([half1, half2])
        current_groups = new_groups

    if deviations:
        return 1.0 - np.mean(deviations)
    return 0.5


def apply_break(state, group, half1, half2, method, rng=None):
    """
    Apply a symmetry break to a group using the specified method.
    All methods conserve the total within the group.

    Methods:
    - 'phi': redistribute at D/S = phi (the DFT prediction)
    - 'equal': redistribute at D/S = 1 (50/50)
    - 'two_to_one': redistribute at D/S = 2
    - 'random': random Dirichlet split
    - 'noise': add random noise (±10%) to each node
    """
    group_total = np.sum(state[group])

    s1 = np.sum(state[half1])
    s2 = np.sum(state[half2])

    # Determine dominant half (larger sum)
    if s1 >= s2:
        dom_nodes, sub_nodes = half1, half2
    else:
        dom_nodes, sub_nodes = half2, half1

    dom_current = np.sum(state[dom_nodes])
    sub_current = np.sum(state[sub_nodes])

    if method == 'phi':
        dom_target = group_total / PHI
        sub_target = group_total - dom_target
    elif method == 'equal':
        dom_target = group_total / 2
        sub_target = group_total / 2
    elif method == 'two_to_one':
        dom_target = group_total * 2 / 3
        sub_target = group_total / 3
    elif method == 'random':
        if rng is None:
            rng = np.random.RandomState(42)
        frac = rng.beta(2, 2)  # Random fraction, symmetric around 0.5
        frac = max(frac, 1 - frac)  # Dominant always gets more
        dom_target = group_total * frac
        sub_target = group_total * (1 - frac)
    elif method == 'noise':
        # Just add noise, don't redistribute
        if rng is None:
            rng = np.random.RandomState(42)
        noise = rng.randn(len(group)) * 0.1 * np.mean(state[group])
        state[group] += noise
        state[group] = np.maximum(state[group], 1e-10)
        state[group] *= group_total / np.sum(state[group])
        return state
    else:
        return state

    # Apply redistribution
    if dom_current > 1e-15:
        state[dom_nodes] *= dom_target / dom_current
    if sub_current > 1e-15:
        state[sub_nodes] *= sub_target / sub_current

    return state


def run_break_cascade(state, L, method, max_levels=4, seed=42):
    """
    Run a full cascade of symmetry breaks using the given method.
    Returns phi-balance at each level and fraction of breaks that improved.
    """
    rng = np.random.RandomState(seed)
    n = len(state)
    all_nodes = list(range(n))
    current_groups = [all_nodes]

    pb_history = [phi_balance_score(state, L, n_levels=4)]
    break_improved = []

    for level in range(1, max_levels + 1):
        new_groups = []
        for group in current_groups:
            if len(group) < 4:
                new_groups.append(group)
                continue

            sub_L = L[np.ix_(group, group)]
            eigs, vecs = np.linalg.eigh(sub_L)
            fiedler = vecs[:, 1]

            half1 = [group[i] for i in range(len(group)) if fiedler[i] >= 0]
            half2 = [group[i] for i in range(len(group)) if fiedler[i] < 0]

            if not half1 or not half2:
                new_groups.append(group)
                continue

            # Measure before
            pb_before = phi_balance_score(state, L, n_levels=4)

            # Apply break
            state = apply_break(state, group, half1, half2, method, rng)

            # Measure after
            pb_after = phi_balance_score(state, L, n_levels=4)
            break_improved.append(pb_after >= pb_before - 0.001)

            new_groups.extend([half1, half2])

        pb_history.append(phi_balance_score(state, L, n_levels=4))
        current_groups = new_groups

    frac_improved = (sum(1 for x in break_improved if x) / len(break_improved)
                     if break_improved else 0)

    return pb_history, frac_improved, state


def build_graph(gtype, n):
    """Build adjacency and Laplacian for graph type."""
    if gtype == 'ring':
        A = build_ring(n).toarray()
    elif gtype == 'torus':
        side = int(np.sqrt(n))
        A = build_torus(side, side).toarray()
        n = side * side
    elif gtype == 'random_regular':
        A = build_random_regular(n, k=4).toarray()
    else:
        A = build_ring(n).toarray()
    D_mat = np.diag(A.sum(axis=1))
    L = D_mat - A
    return A, L, n


def main():
    print("=" * 70)
    print("MILESTONE 7 - EXP 06: SYMMETRY BREAKING AS SYMMETRY-SEEKING")
    print("Block C: Consequences")
    print("=" * 70)

    print(f"\n  NOTE: Tests whether ANY break mechanism improves phi-balance,")
    print(f"  not just phi-ratio splits. The drive determines the optimal")
    print(f"  ratio; the test shows ALL breaks are symmetry-seeking.\n")

    configs = [
        ("Ring N=64", 'ring', 64),
        ("Torus 8x8", 'torus', 64),
        ("Random Regular k=4", 'random_regular', 64),
    ]

    methods = ['phi', 'equal', 'two_to_one', 'random', 'noise']

    all_graph_results = []

    for name, gtype, n in configs:
        print(f"\n{'=' * 60}")
        print(f"GRAPH: {name}")
        print("=" * 60)

        A, L, n = build_graph(gtype, n)
        graph_results = {}

        for method in methods:
            # Start from near-uniform state each time
            rng = np.random.RandomState(42)
            state = np.ones(n) + rng.randn(n) * 1e-4
            state = np.maximum(state, 1e-10)
            total = np.sum(state)
            state *= n / total  # normalize to mean=1

            pb_history, frac_improved, final = run_break_cascade(
                state.copy(), L, method, max_levels=4, seed=42)

            graph_results[method] = {
                'pb_initial': pb_history[0],
                'pb_final': pb_history[-1],
                'gain': pb_history[-1] - pb_history[0],
                'frac_improved': frac_improved,
                'pb_history': pb_history,
            }

            print(f"\n  {method:14s}: PB {pb_history[0]:.4f} -> {pb_history[-1]:.4f} "
                  f"(gain={pb_history[-1] - pb_history[0]:+.4f}, "
                  f"{frac_improved:.0%} breaks improved)")

        all_graph_results.append((name, graph_results))

    # ============================================================
    # Analysis across all graphs
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Test 1: How many methods improve PB vs uniform?
    print("\n  Test 1: Break types that improve phi-balance vs uniform")
    method_gains = {m: [] for m in methods}
    for name, gr in all_graph_results:
        for method in methods:
            method_gains[method].append(gr[method]['gain'])

    improved_methods = 0
    for method in methods:
        mean_gain = np.mean(method_gains[method])
        all_positive = all(g > -0.01 for g in method_gains[method])
        if mean_gain > 0:
            improved_methods += 1
        print(f"    {method:14s}: mean gain = {mean_gain:+.4f} "
              f"({'IMPROVES' if mean_gain > 0 else 'worsens'})")

    # Test 2: Is phi-ratio the best?
    print(f"\n  Test 2: Is phi-ratio break optimal?")
    for name, gr in all_graph_results:
        gains = {m: gr[m]['gain'] for m in methods}
        best = max(gains, key=gains.get)
        phi_gain = gains['phi']
        best_gain = gains[best]
        is_best = best == 'phi' or abs(phi_gain - best_gain) < 0.02
        print(f"    {name}: phi={phi_gain:+.4f}, best={best}({best_gain:+.4f})"
              f" {'PHI OPTIMAL' if is_best else f'phi not best'}")

    # Test 3: Multi-level cascade improvement
    print(f"\n  Test 3: Per-level phi-balance improvement")
    all_frac_improved = []
    for name, gr in all_graph_results:
        # Use phi cascade for multi-level analysis
        phi_data = gr['phi']
        hist = phi_data['pb_history']
        gains = [hist[i+1] - hist[i] for i in range(len(hist)-1)]
        n_positive = sum(1 for g in gains if g >= -0.001)
        frac = n_positive / len(gains) if gains else 0
        all_frac_improved.append(frac)
        print(f"    {name}: {n_positive}/{len(gains)} levels improved "
              f"({frac:.0%})")
        print(f"      PB trajectory: " +
              " -> ".join(f"{h:.3f}" for h in hist))

    # Test 4: Cross-topology consistency
    print(f"\n  Test 4: Cross-topology consistency")
    topo_improved = []
    for name, gr in all_graph_results:
        # All methods improve on this graph?
        n_methods_up = sum(1 for m in methods if gr[m]['gain'] > -0.01)
        consistent = n_methods_up >= 4
        topo_improved.append(consistent)
        print(f"    {name}: {n_methods_up}/{len(methods)} methods improve "
              f"({'consistent' if consistent else 'NOT consistent'})")

    # ============================================================
    # VERIFICATION
    # ============================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    test1 = improved_methods >= 4
    print(f"\n  Test 1: >= 4/5 break types improve phi-balance")
    print(f"    {improved_methods}/5 methods improve")
    print(f"    -> {'VERIFIED' if test1 else 'NOT VERIFIED'}")

    # Check if phi is best or near-best across graphs
    phi_is_optimal = []
    for name, gr in all_graph_results:
        gains = {m: gr[m]['gain'] for m in methods}
        best_gain = max(gains.values())
        phi_gain = gains['phi']
        phi_is_optimal.append(
            abs(phi_gain - best_gain) < 0.02 or phi_gain == best_gain)
    test2 = sum(phi_is_optimal) >= 2
    print(f"\n  Test 2: Phi-ratio break optimal (>= 2/3 graphs)")
    print(f"    Phi optimal: {sum(phi_is_optimal)}/3")
    print(f"    -> {'VERIFIED' if test2 else 'NOT VERIFIED'}")

    mean_frac_improved = np.mean(all_frac_improved)
    test3 = mean_frac_improved >= 0.75
    print(f"\n  Test 3: >= 75% of cascade levels improve phi-balance")
    print(f"    Mean: {mean_frac_improved:.0%}")
    print(f"    -> {'VERIFIED' if test3 else 'NOT VERIFIED'}")

    n_consistent = sum(1 for x in topo_improved if x)
    test4 = n_consistent >= 2
    print(f"\n  Test 4: Cross-topology consistency (>= 2/3)")
    print(f"    Consistent: {n_consistent}/3")
    print(f"    -> {'VERIFIED' if test4 else 'NOT VERIFIED'}")

    verified = sum([test1, test2, test3, test4])
    print(f"\n  TOTAL: {verified}/4 verified")

    results = {
        'experiment': 'exp_06_symmetry_breaking_as_seeking',
        'milestone': 7,
        'block': 'C',
        'methods': methods,
        'per_graph': {
            name: {
                method: {
                    'pb_initial': float(gr[method]['pb_initial']),
                    'pb_final': float(gr[method]['pb_final']),
                    'gain': float(gr[method]['gain']),
                    'frac_improved': float(gr[method]['frac_improved']),
                }
                for method in methods
            }
            for name, gr in all_graph_results
        },
        'method_means': {
            method: float(np.mean(method_gains[method]))
            for method in methods
        },
        'verification': {
            'test1_all_improve': test1,
            'test2_phi_optimal': test2,
            'test3_cascade_improve': test3,
            'test4_cross_topology': test4,
            'verified_count': verified,
        },
    }
    save_results(results, 'exp_06_symmetry_breaking_as_seeking', RESULTS_DIR)


if __name__ == '__main__':
    main()
