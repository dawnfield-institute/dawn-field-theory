"""
exp_22_coupling_ceiling.py -- Confluent Identity Phase 13

PURPOSE:
    Find the coupling-contribution correlation ceiling. The rho plateaued at
    0.42 (uniform) and 0.50 (eigenvector). Is 0.50 the ceiling or can better
    weighting schemes push higher? Also establishes cross-experiment consistency
    by testing 5 schemes on identical data.

METHODS:
    For all parent-children groups, compute coupling weights under 5 schemes:
    1. Uniform: equal perturbation per cell
    2. Eigenvector: weight by |Fiedler vector|
    3. Gradient: weight by |grad(C)|
    4. Entropy-weighted: weight by local PAC entropy S(cell)
    5. Laplacian-response: weight by |(L @ state)[cell]|

    For each scheme: raw rho, partial_rho(| size), rho(coupling, size)

VERIFICATION:
    - At least one scheme achieves partial_rho > 0.45
    - Best scheme has rho(coupling, size) < 0.3
    - Scheme ranking consistent across hierarchy levels (Kendall tau > 0)
    - Entropy or Laplacian scheme outperforms uniform on partial_rho

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr, kendalltau

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, K_MODES, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices, get_parent_children_data,
)
from exp_08_gradient_coupling import (
    compute_coupling_weights_uniform, compute_coupling_weights_weighted,
    compute_natural_weights, compute_gradient_field, compute_fiedler_field,
)
from exp_14_partial_correlation import partial_spearman


def compute_entropy_field(P, A):
    """
    Local PAC entropy: S(cell) = -p*log(p) - (1-p)*log(1-p)
    where p = P / (P + A). Cells near equilibrium (S ~ max) are at transitions.
    """
    C = P + A
    # Avoid division by zero
    safe_C = np.where(C > 1e-15, C, 1e-15)
    p = P / safe_C
    # Clip to avoid log(0)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return entropy


def compute_laplacian_response_field(adjacency, state_flat):
    """
    Laplacian response: |(L @ state)[cell]|.
    Cells where Laplacian is large are flow-active.
    """
    N_total = adjacency.shape[0]
    degrees = np.array(adjacency.sum(axis=1)).ravel()
    L_global = np.diag(degrees) - adjacency.toarray()  # Too expensive for large grids
    # Instead, compute locally: L@f = degree*f - sum(w*f_neighbor)
    response = degrees * state_flat
    response -= adjacency.dot(state_flat)
    return np.abs(response)


def compute_laplacian_response_sparse(adjacency, state_flat):
    """
    Sparse Laplacian response: |(D - W) @ state|.
    """
    from scipy import sparse
    degrees = np.array(adjacency.sum(axis=1)).ravel()
    D = sparse.diags(degrees)
    L = D - adjacency
    response = L.dot(state_flat)
    return np.abs(response)


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 13, Experiment 22")
    print("Coupling Ceiling: 5 Weighting Schemes Compared")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency and weight fields...")
    adjacency = build_lattice_adjacency(C)

    # Pre-compute weight fields
    grad_mag = compute_gradient_field(C)
    grad_flat = grad_mag.ravel()

    entropy_field = compute_entropy_field(P, A)
    entropy_flat = entropy_field.ravel()

    laplacian_response = compute_laplacian_response_sparse(adjacency, state_flat)

    print(f"  Gradient: mean={grad_flat.mean():.6f}")
    print(f"  Entropy: mean={entropy_flat.mean():.6f}")
    print(f"  Laplacian response: mean={laplacian_response.mean():.6f}")

    # Collect per-scheme data
    schemes = ['uniform', 'eigenvector', 'gradient', 'entropy', 'laplacian']
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

        # Scheme 1: Uniform
        w_uniform = compute_coupling_weights_uniform(
            adjacency, state_flat, parent_indices, children_list
        )

        # Scheme 2: Eigenvector (Fiedler)
        fiedler_local = compute_fiedler_field(adjacency, parent_indices)
        fiedler_global = np.zeros(len(state_flat))
        for i, gi in enumerate(parent_indices):
            fiedler_global[gi] = fiedler_local[i]
        w_eigenvec = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, fiedler_global
        )

        # Scheme 3: Gradient
        w_gradient = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, grad_flat
        )

        # Scheme 4: Entropy
        w_entropy = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, entropy_flat
        )

        # Scheme 5: Laplacian response
        w_laplacian = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, laplacian_response
        )

        weight_maps = {
            'uniform': w_uniform,
            'eigenvector': w_eigenvec,
            'gradient': w_gradient,
            'entropy': w_entropy,
            'laplacian': w_laplacian,
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

    # --- Compute correlations for each scheme ---
    print(f"\n{'=' * 70}")
    print("Correlation Comparison: 5 Weighting Schemes")
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

        # Also partial with log(size)
        log_size = np.log(size + 1e-15)
        partial_rho_log, partial_p_log = partial_spearman(coupling, natural, log_size)

        scheme_results[scheme_name] = {
            'n': n,
            'rho_raw': float(rho_raw),
            'p_raw': float(p_raw),
            'partial_rho': float(partial_rho),
            'partial_p': float(partial_p),
            'partial_rho_logsize': float(partial_rho_log),
            'partial_p_logsize': float(partial_p_log),
            'rho_coupling_size': float(rho_cs),
        }

        print(f"\n  {scheme_name.upper()} (n={n}):")
        print(f"    raw rho(coupling, natural) = {rho_raw:.4f}")
        print(f"    partial rho(| size)        = {partial_rho:.4f}  p={partial_p:.2e}")
        print(f"    partial rho(| log(size))   = {partial_rho_log:.4f}")
        print(f"    rho(coupling, size)        = {rho_cs:.4f}")

    # --- Cross-level consistency ---
    print(f"\n{'=' * 70}")
    print("Cross-Level Consistency")
    print(f"{'=' * 70}")

    # Compute partial_rho per level for each scheme, then rank schemes within levels
    levels_present = sorted(set(scheme_data['uniform']['level']))
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

            if len(coupling) < 5:
                continue

            pr, _ = partial_spearman(coupling, natural, size)
            level_ranks[level][scheme_name] = float(pr)

    # Compute Kendall tau between level rankings
    level_keys = [l for l in levels_present if len(level_ranks[l]) >= 3]
    tau_cross = None
    if len(level_keys) >= 2:
        # Compare first two levels with enough data
        l0, l1 = level_keys[0], level_keys[1]
        common_schemes = sorted(set(level_ranks[l0].keys()) & set(level_ranks[l1].keys()))
        if len(common_schemes) >= 3:
            ranks_l0 = [level_ranks[l0][s] for s in common_schemes]
            ranks_l1 = [level_ranks[l1][s] for s in common_schemes]
            tau_cross, p_tau = kendalltau(ranks_l0, ranks_l1)
            print(f"  Levels {l0} vs {l1}: Kendall tau = {tau_cross:.4f} "
                  f"(n={len(common_schemes)} schemes)")
            for s, r0, r1 in zip(common_schemes, ranks_l0, ranks_l1):
                print(f"    {s}: L{l0}={r0:.4f}, L{l1}={r1:.4f}")

    # --- Verification ---
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Test 1: At least one scheme achieves partial_rho > 0.45
    best_partial = max(
        (sr.get('partial_rho', 0) for sr in scheme_results.values()
         if isinstance(sr.get('partial_rho'), (int, float))),
        default=0
    )
    best_scheme = max(
        ((name, sr.get('partial_rho', 0)) for name, sr in scheme_results.items()
         if isinstance(sr.get('partial_rho'), (int, float))),
        key=lambda x: x[1], default=('none', 0)
    )
    test1 = best_partial > 0.45
    print(f"\n  Test 1: Any scheme with partial_rho > 0.45?")
    print(f"    Best: {best_scheme[0]} at {best_scheme[1]:.4f}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Best scheme has rho(coupling, size) < 0.3
    best_scheme_name = best_scheme[0]
    best_cs = scheme_results.get(best_scheme_name, {}).get('rho_coupling_size', 1.0)
    test2 = abs(best_cs) < 0.3
    print(f"\n  Test 2: Best scheme rho(coupling, size) < 0.3?")
    print(f"    {best_scheme_name}: rho_cs = {best_cs:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: Cross-level consistency (Kendall tau > 0)
    test3 = tau_cross is not None and tau_cross > 0
    print(f"\n  Test 3: Scheme ranking consistent across levels (tau > 0)?")
    if tau_cross is not None:
        print(f"    Kendall tau = {tau_cross:.4f}")
    else:
        print(f"    Insufficient cross-level data")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: Entropy or Laplacian outperforms uniform
    uniform_partial = scheme_results.get('uniform', {}).get('partial_rho', 0)
    entropy_partial = scheme_results.get('entropy', {}).get('partial_rho', 0)
    laplacian_partial = scheme_results.get('laplacian', {}).get('partial_rho', 0)
    test4 = max(entropy_partial, laplacian_partial) > uniform_partial
    print(f"\n  Test 4: Entropy or Laplacian outperforms uniform on partial_rho?")
    print(f"    Uniform: {uniform_partial:.4f}, Entropy: {entropy_partial:.4f}, "
          f"Laplacian: {laplacian_partial:.4f}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 coupling ceiling tests verified")

    # Summary table
    print(f"\n{'=' * 70}")
    print("Summary Table")
    print(f"{'=' * 70}")
    print(f"  {'Scheme':<15} {'raw rho':>10} {'partial':>10} {'rho(c,s)':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for name in schemes:
        sr = scheme_results.get(name, {})
        rr = sr.get('rho_raw', 0)
        pr = sr.get('partial_rho', 0)
        cs = sr.get('rho_coupling_size', 0)
        print(f"  {name:<15} {rr:>10.4f} {pr:>10.4f} {cs:>10.4f}")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_22_coupling_ceiling',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Coupling ceiling via 5 weighting schemes',
        'n_parents': n_parents,
        'scheme_results': scheme_results,
        'cross_level': {
            'tau': float(tau_cross) if tau_cross is not None else None,
            'level_ranks': {str(k): v for k, v in level_ranks.items()},
        },
        'verification': {
            'test1_partial_rho_above_045': bool(test1),
            'test2_best_scheme_low_size_corr': bool(test2),
            'test3_cross_level_consistency': bool(test3),
            'test4_entropy_or_laplacian_beats_uniform': bool(test4),
            'n_verified': n_verified,
        },
    }

    output_file = RESULTS_DIR / f'exp_22_coupling_ceiling_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
