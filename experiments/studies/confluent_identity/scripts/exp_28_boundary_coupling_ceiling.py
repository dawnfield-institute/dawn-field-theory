"""
exp_28_boundary_coupling_ceiling.py -- Confluent Identity Phase 19

PURPOSE:
    Test whether boundary-weighted perturbation breaks through the coupling
    ceiling at partial_rho = 0.41 (exp_22). All 5 prior schemes weight by
    interior properties — but coupling is a boundary phenomenon.

METHODS:
    3 new boundary-aware weighting schemes + gradient control:
    1. boundary_gradient: weight by |grad C| on boundary cells only, 0 for interior
    2. boundary_proximity: weight by 1 / (distance_to_boundary + 1) via BFS
    3. boundary_fiedler: weight by |v_fiedler| on boundary cells only, 0 for interior

    For each: raw rho, partial_rho(coupling, natural | size), rho(coupling, size).
    Cross-level Kendall tau for consistency.

VERIFICATION:
    - At least one boundary scheme partial_rho > 0.45 (beats gradient's 0.41)
    - Best boundary scheme outperforms gradient by delta > 0.04
    - Best boundary scheme rho(coupling, size) < 0.30
    - Cross-level Kendall tau > 0 for best boundary scheme

Planck units throughout.
"""

import numpy as np
import json
from collections import deque
from datetime import datetime
from scipy.stats import spearmanr, kendalltau

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_parent_children_data,
)
from exp_08_gradient_coupling import (
    compute_coupling_weights_weighted, compute_natural_weights,
    compute_gradient_field, compute_fiedler_field,
)
from exp_14_partial_correlation import partial_spearman
from exp_27_boundary_geometry import compute_boundary_metrics


def identify_boundary_cells(indices, N):
    """
    Return set of flat indices that are boundary cells (have >= 1 neighbor
    outside the region) on a periodic NxN lattice.
    """
    index_set = set(int(i) for i in indices)
    boundary = set()
    for g in indices:
        g = int(g)
        i, j = divmod(g, N)
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = (i + di) % N, (j + dj) % N
            neighbor = ni * N + nj
            if neighbor not in index_set:
                boundary.add(g)
                break
    return boundary


def compute_boundary_distance(indices, N):
    """
    BFS distance from each cell to the nearest boundary cell.
    Returns dict: flat_index -> distance (0 for boundary cells).
    """
    index_set = set(int(i) for i in indices)
    boundary = identify_boundary_cells(indices, N)

    dist = {}
    queue = deque()
    for b in boundary:
        dist[b] = 0
        queue.append(b)

    while queue:
        g = queue.popleft()
        i, j = divmod(g, N)
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = (i + di) % N, (j + dj) % N
            neighbor = ni * N + nj
            if neighbor in index_set and neighbor not in dist:
                dist[neighbor] = dist[g] + 1
                queue.append(neighbor)

    return dist


def build_boundary_gradient_field(indices, N, grad_flat):
    """
    Weight field: |grad C| on boundary cells, 0 on interior.
    Returns array aligned with global flat indices.
    """
    boundary = identify_boundary_cells(indices, N)
    field = np.zeros(len(grad_flat))
    for g in boundary:
        field[g] = grad_flat[g]
    return field


def build_boundary_proximity_field(indices, N):
    """
    Weight field: 1 / (distance_to_boundary + 1).
    Boundary cells get weight 1.0, interior cells decay.
    Returns array aligned with global flat indices.
    """
    dist = compute_boundary_distance(indices, N)
    n_total = N * N
    field = np.zeros(n_total)
    for g, d in dist.items():
        field[g] = 1.0 / (d + 1)
    return field


def build_boundary_fiedler_field(indices, N, adjacency):
    """
    Weight field: |v_fiedler| on boundary cells, 0 on interior.
    Returns array aligned with global flat indices.
    """
    boundary = identify_boundary_cells(indices, N)
    fiedler_local = compute_fiedler_field(adjacency, indices)

    n_total = N * N
    field = np.zeros(n_total)
    for i, gi in enumerate(indices):
        gi = int(gi)
        if gi in boundary:
            field[gi] = abs(fiedler_local[i])
    return field


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 19, Experiment 28")
    print("Boundary Coupling Ceiling: 3 Boundary-Aware Schemes")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency and gradient field...")
    adjacency = build_lattice_adjacency(C)
    grad_mag = compute_gradient_field(C)
    grad_flat = grad_mag.ravel()

    # Scheme names: control + 3 boundary schemes
    schemes = ['gradient', 'boundary_gradient', 'boundary_proximity', 'boundary_fiedler']
    scheme_data = {s: {'coupling': [], 'natural': [], 'size': [], 'level': []}
                   for s in schemes}

    n_parents = 0
    for (level, pid), parent_indices, children_list, L_parent, state_parent in \
            get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):

        n_parents += 1
        identity_parent = compute_spectral_identity(L_parent, state_parent)
        eigvecs_parent = identity_parent.get('eigenvectors')
        if eigvecs_parent is None:
            continue

        natural_weights, size_fractions = compute_natural_weights(
            state_flat, parent_indices, children_list, eigvecs_parent
        )

        # Control: gradient (interior-based, best from exp_22)
        w_gradient = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, grad_flat
        )

        # Scheme 1: boundary_gradient
        bg_field = build_boundary_gradient_field(parent_indices, N, grad_flat)
        w_bg = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, bg_field
        )

        # Scheme 2: boundary_proximity
        bp_field = build_boundary_proximity_field(parent_indices, N)
        w_bp = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, bp_field
        )

        # Scheme 3: boundary_fiedler
        bf_field = build_boundary_fiedler_field(parent_indices, N, adjacency)
        w_bf = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, bf_field
        )

        weight_maps = {
            'gradient': w_gradient,
            'boundary_gradient': w_bg,
            'boundary_proximity': w_bp,
            'boundary_fiedler': w_bf,
        }

        for child_id, _ in children_list:
            cid = child_id
            nat_w = natural_weights.get(cid, 0)
            size_f = size_fractions.get(cid, 0)

            for scheme_name, w_map in weight_maps.items():
                if cid in w_map:
                    scheme_data[scheme_name]['coupling'].append(w_map[cid])
                    scheme_data[scheme_name]['natural'].append(nat_w)
                    scheme_data[scheme_name]['size'].append(size_f)
                    scheme_data[scheme_name]['level'].append(level)

    print(f"\nCollected data from {n_parents} parents")

    # =====================================================================
    # Correlations per scheme
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Correlation Comparison: Gradient vs 3 Boundary Schemes")
    print(f"{'=' * 70}")

    scheme_results = {}
    for scheme_name in schemes:
        sd = scheme_data[scheme_name]
        n = len(sd['coupling'])
        if n < 10:
            scheme_results[scheme_name] = {'n': n, 'error': 'insufficient data'}
            continue

        coupling = np.array(sd['coupling'])
        natural = np.array(sd['natural'])
        size = np.array(sd['size'])

        rho_raw, p_raw = spearmanr(coupling, natural)
        partial_rho, partial_p = partial_spearman(coupling, natural, size)
        rho_cs, _ = spearmanr(coupling, size)

        scheme_results[scheme_name] = {
            'n': n,
            'rho_raw': float(rho_raw),
            'p_raw': float(p_raw),
            'partial_rho': float(partial_rho),
            'partial_p': float(partial_p),
            'rho_coupling_size': float(rho_cs),
        }

        sig = '*' if partial_p < 0.05 else ''
        print(f"\n  {scheme_name.upper()} (n={n}):")
        print(f"    raw rho(coupling, natural) = {rho_raw:.4f}")
        print(f"    partial rho(| size)        = {partial_rho:.4f}  p={partial_p:.2e} {sig}")
        print(f"    rho(coupling, size)        = {rho_cs:.4f}")

    # =====================================================================
    # Cross-level consistency
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Cross-Level Consistency")
    print(f"{'=' * 70}")

    levels_present = sorted(set(scheme_data['gradient']['level']))
    level_ranks = {level: {} for level in levels_present}

    for level in levels_present:
        for scheme_name in schemes:
            sd = scheme_data[scheme_name]
            mask = [i for i, l in enumerate(sd['level']) if l == level]
            if len(mask) < 5:
                continue

            coupling = np.array([sd['coupling'][i] for i in mask])
            natural = np.array([sd['natural'][i] for i in mask])
            size = np.array([sd['size'][i] for i in mask])

            pr, _ = partial_spearman(coupling, natural, size)
            level_ranks[level][scheme_name] = float(pr)

    level_keys = [l for l in levels_present if len(level_ranks[l]) >= 3]
    tau_cross = None
    if len(level_keys) >= 2:
        l0, l1 = level_keys[0], level_keys[1]
        common_schemes = sorted(set(level_ranks[l0].keys()) & set(level_ranks[l1].keys()))
        if len(common_schemes) >= 3:
            ranks_l0 = [level_ranks[l0][s] for s in common_schemes]
            ranks_l1 = [level_ranks[l1][s] for s in common_schemes]
            tau_cross, p_tau = kendalltau(ranks_l0, ranks_l1)
            print(f"  Levels {l0} vs {l1}: Kendall tau = {tau_cross:.4f}")
            for s, r0, r1 in zip(common_schemes, ranks_l0, ranks_l1):
                print(f"    {s}: L{l0}={r0:.4f}, L{l1}={r1:.4f}")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    boundary_schemes = ['boundary_gradient', 'boundary_proximity', 'boundary_fiedler']
    gradient_partial = scheme_results.get('gradient', {}).get('partial_rho', 0)

    # Best boundary scheme by partial_rho
    best_boundary = max(
        ((name, scheme_results.get(name, {}).get('partial_rho', 0))
         for name in boundary_schemes),
        key=lambda x: x[1]
    )

    # Test 1: At least one boundary scheme partial_rho > 0.45
    test1 = best_boundary[1] > 0.45
    print(f"\n  Test 1: Any boundary scheme partial_rho > 0.45?")
    print(f"    Best: {best_boundary[0]} at {best_boundary[1]:.4f}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Best boundary outperforms gradient by delta > 0.04
    delta = best_boundary[1] - gradient_partial
    test2 = delta > 0.04
    print(f"\n  Test 2: Best boundary outperforms gradient by > 0.04?")
    print(f"    Gradient: {gradient_partial:.4f}, Best boundary: {best_boundary[1]:.4f}")
    print(f"    Delta: {delta:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: Best boundary scheme rho(coupling, size) < 0.30
    best_cs = scheme_results.get(best_boundary[0], {}).get('rho_coupling_size', 1.0)
    test3 = abs(best_cs) < 0.30
    print(f"\n  Test 3: Best boundary rho(coupling, size) < 0.30?")
    print(f"    {best_boundary[0]}: rho_cs = {best_cs:.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: Cross-level tau > 0
    test4 = tau_cross is not None and tau_cross > 0
    print(f"\n  Test 4: Cross-level Kendall tau > 0?")
    if tau_cross is not None:
        print(f"    tau = {tau_cross:.4f}")
    else:
        print(f"    Insufficient cross-level data")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 boundary coupling tests verified")

    # Summary table
    print(f"\n{'=' * 70}")
    print("Summary Table")
    print(f"{'=' * 70}")
    print(f"  {'Scheme':<22} {'raw rho':>10} {'partial':>10} {'rho(c,s)':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for name in schemes:
        sr = scheme_results.get(name, {})
        rr = sr.get('rho_raw', 0)
        pr = sr.get('partial_rho', 0)
        cs = sr.get('rho_coupling_size', 0)
        print(f"  {name:<22} {rr:>10.4f} {pr:>10.4f} {cs:>10.4f}")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_28_boundary_coupling_ceiling',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Break coupling ceiling with boundary-aware weighting schemes',
        'n_parents': n_parents,
        'scheme_results': scheme_results,
        'cross_level': {
            'tau': float(tau_cross) if tau_cross is not None else None,
            'level_ranks': {str(k): v for k, v in level_ranks.items()},
        },
        'verification': {
            'test1_boundary_above_045': bool(test1),
            'test2_boundary_beats_gradient': bool(test2),
            'test3_low_size_correlation': bool(test3),
            'test4_cross_level_consistency': bool(test4),
            'n_verified': n_verified,
        },
    }

    output_file = RESULTS_DIR / f'exp_28_boundary_coupling_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
