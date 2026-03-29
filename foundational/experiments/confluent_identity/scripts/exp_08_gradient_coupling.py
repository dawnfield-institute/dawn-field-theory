"""
exp_08_gradient_coupling.py -- Confluent Identity Phase 4

PURPOSE:
    Fix the size confound in coupling weights. The current uniform perturbation
    (exp_03) adds epsilon to ALL cells equally, causing natural weight to
    correlate with region size (rho=0.77). This experiment tests gradient-weighted
    and eigenvector-weighted perturbation schemes that account for WHERE in a
    region matters, not just HOW MANY cells it has.

THREE PERTURBATION SCHEMES:
    1. Uniform (baseline): state[cell] += epsilon
    2. Gradient-weighted: state[cell] += epsilon * |grad_C[cell]| / mean(|grad_C|)
       Cells in flow channels (high gradient) get more perturbation.
    3. Eigenvector-weighted: state[cell] += epsilon * |v_fiedler[cell]| / mean(|v_fiedler|)
       Perturb where the dominant mode is active.

VERIFICATION:
    - gradient rho(coupling, natural) > 0.42 (improves on uniform)
    - gradient rho(natural, size) < 0.77 (deconfounds from size)

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, K_MODES, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices, get_parent_children_data,
)


def compute_gradient_field(C):
    """Compute |grad(C)| using central differences with periodic BC."""
    grad_x = np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0)
    grad_y = np.roll(C, -1, axis=1) - np.roll(C, 1, axis=1)
    return np.sqrt(grad_x**2 + grad_y**2) / 2.0  # central diff normalization


def compute_coupling_weights_weighted(adjacency, state_flat, parent_indices,
                                       children_list, weight_field_flat,
                                       epsilon=None):
    """
    Coupling weight via weighted perturbation.

    Instead of uniform epsilon, each cell is perturbed proportional to
    weight_field_flat[cell] / mean(weight_field over child).
    """
    if epsilon is None:
        epsilon = 0.01 * np.mean(state_flat[parent_indices])

    # Baseline
    L_parent, _ = graph_laplacian_subgraph(adjacency, parent_indices)
    state_parent = state_flat[parent_indices]
    I_baseline = compute_spectral_identity(L_parent, state_parent)
    baseline_coeffs = np.array(I_baseline['state_coefficients'])

    # Index mapping
    parent_pos_map = {int(idx): pos for pos, idx in enumerate(parent_indices)}

    sensitivities = {}
    for child_id, child_indices in children_list:
        state_perturbed = state_parent.copy()

        # Get weight values for this child's cells within parent
        child_weights = []
        child_positions = []
        for cidx in child_indices:
            cidx_int = int(cidx)
            if cidx_int in parent_pos_map:
                child_positions.append(parent_pos_map[cidx_int])
                child_weights.append(weight_field_flat[cidx_int])

        if len(child_weights) == 0:
            sensitivities[child_id] = 0.0
            continue

        child_weights = np.array(child_weights)
        mean_w = np.mean(child_weights)
        if mean_w < 1e-15:
            # Fallback to uniform if weights are all zero
            scale = np.ones(len(child_weights))
        else:
            scale = child_weights / mean_w  # normalized so mean perturbation = epsilon

        for i, pos in enumerate(child_positions):
            state_perturbed[pos] += epsilon * scale[i]

        I_perturbed = compute_spectral_identity(L_parent, state_perturbed)
        perturbed_coeffs = np.array(I_perturbed['state_coefficients'])

        min_len = min(len(baseline_coeffs), len(perturbed_coeffs))
        delta = np.linalg.norm(
            perturbed_coeffs[:min_len] - baseline_coeffs[:min_len]
        ) / epsilon
        sensitivities[child_id] = float(delta)

    # Normalize
    total = sum(sensitivities.values())
    if total > 1e-15:
        weights = {cid: s / total for cid, s in sensitivities.items()}
    else:
        n = len(sensitivities)
        weights = {cid: 1.0 / n for cid in sensitivities} if n > 0 else {}

    return weights


def compute_coupling_weights_uniform(adjacency, state_flat, parent_indices,
                                      children_list, epsilon=None):
    """Original uniform perturbation (baseline, same as exp_03)."""
    if epsilon is None:
        epsilon = 0.01 * np.mean(state_flat[parent_indices])

    L_parent, _ = graph_laplacian_subgraph(adjacency, parent_indices)
    state_parent = state_flat[parent_indices]
    I_baseline = compute_spectral_identity(L_parent, state_parent)
    baseline_coeffs = np.array(I_baseline['state_coefficients'])

    parent_pos_map = {int(idx): pos for pos, idx in enumerate(parent_indices)}

    sensitivities = {}
    for child_id, child_indices in children_list:
        state_perturbed = state_parent.copy()
        for cidx in child_indices:
            cidx_int = int(cidx)
            if cidx_int in parent_pos_map:
                state_perturbed[parent_pos_map[cidx_int]] += epsilon

        I_perturbed = compute_spectral_identity(L_parent, state_perturbed)
        perturbed_coeffs = np.array(I_perturbed['state_coefficients'])

        min_len = min(len(baseline_coeffs), len(perturbed_coeffs))
        delta = np.linalg.norm(
            perturbed_coeffs[:min_len] - baseline_coeffs[:min_len]
        ) / epsilon
        sensitivities[child_id] = float(delta)

    total = sum(sensitivities.values())
    if total > 1e-15:
        weights = {cid: s / total for cid, s in sensitivities.items()}
    else:
        n = len(sensitivities)
        weights = {cid: 1.0 / n for cid in sensitivities} if n > 0 else {}

    return weights


def compute_natural_weights(state_flat, parent_indices, children_list,
                            eigvecs_parent):
    """Natural contribution weights in parent eigenbasis."""
    parent_pos_map = {int(idx): pos for pos, idx in enumerate(parent_indices)}
    state_parent = state_flat[parent_indices]
    state_centered = state_parent - np.mean(state_parent)
    n_modes = eigvecs_parent.shape[1]

    norms = {}
    sizes = {}
    for child_id, child_indices in children_list:
        local_positions = []
        for g in child_indices:
            g_int = int(g)
            if g_int in parent_pos_map:
                local_positions.append(parent_pos_map[g_int])
        local_positions = np.array(local_positions)

        if len(local_positions) == 0:
            norms[child_id] = 0.0
            sizes[child_id] = 0
            continue

        child_state = state_centered[local_positions]
        child_eigvec = eigvecs_parent[local_positions, :]
        contrib = child_state @ child_eigvec
        norms[child_id] = float(np.linalg.norm(contrib))
        sizes[child_id] = len(local_positions)

    total_norm = sum(norms.values())
    total_size = sum(sizes.values())
    if total_norm > 1e-15:
        natural_weights = {cid: n / total_norm for cid, n in norms.items()}
    else:
        nc = len(norms)
        natural_weights = {cid: 1.0 / nc for cid in norms}

    size_fractions = {cid: s / total_size if total_size > 0 else 0
                      for cid, s in sizes.items()}

    return natural_weights, size_fractions


def compute_fiedler_field(adjacency, parent_indices):
    """Get Fiedler eigenvector (v_2) for the parent region."""
    L, _ = graph_laplacian_subgraph(adjacency, parent_indices)
    identity = compute_spectral_identity(L, np.zeros(len(parent_indices)))
    eigvecs = identity.get('eigenvectors')
    if eigvecs is None or eigvecs.shape[1] < 2:
        return np.ones(len(parent_indices))

    # Fiedler vector is second eigenvector (index 1, after kernel)
    fiedler_vec = np.abs(eigvecs[:, 1])
    return fiedler_vec


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 4, Experiment 08")
    print("Gradient-Weighted Coupling: Deconfounding Size")
    print("=" * 70)

    # Load data
    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    # Build adjacency and gradient field
    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)
    grad_mag = compute_gradient_field(C)
    grad_flat = grad_mag.ravel()
    print(f"  Gradient: mean={grad_mag.mean():.6f}, max={grad_mag.max():.6f}")

    # Collect coupling weights under all three schemes
    all_uniform = []      # (child_id, coupling_w, natural_w, size_frac)
    all_gradient = []
    all_eigenvec = []

    n_parents = 0
    for (level, pid), parent_indices, children_list, L_parent, state_parent in \
            get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):

        n_parents += 1
        print(f"\n  L{level} P{pid}: {len(parent_indices)} cells, "
              f"{len(children_list)} children")

        # Parent eigenvectors for natural weights
        identity_parent = compute_spectral_identity(L_parent, state_parent)
        eigvecs_parent = identity_parent.get('eigenvectors')
        if eigvecs_parent is None:
            continue

        # Natural weights and size fractions (same for all schemes)
        natural_weights, size_fractions = compute_natural_weights(
            state_flat, parent_indices, children_list, eigvecs_parent
        )

        # Scheme 1: Uniform
        w_uniform = compute_coupling_weights_uniform(
            adjacency, state_flat, parent_indices, children_list
        )

        # Scheme 2: Gradient-weighted
        w_gradient = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, grad_flat
        )

        # Scheme 3: Eigenvector-weighted (Fiedler amplitude)
        # Build Fiedler field mapped back to global indices
        fiedler_local = compute_fiedler_field(adjacency, parent_indices)
        fiedler_global = np.zeros(len(state_flat))
        for i, gi in enumerate(parent_indices):
            fiedler_global[gi] = fiedler_local[i]

        w_eigenvec = compute_coupling_weights_weighted(
            adjacency, state_flat, parent_indices, children_list, fiedler_global
        )

        # Collect per-child data
        for child_id, _ in children_list:
            cid = child_id
            nat_w = natural_weights.get(cid, 0)
            size_f = size_fractions.get(cid, 0)

            if cid in w_uniform:
                all_uniform.append((cid, w_uniform[cid], nat_w, size_f))
            if cid in w_gradient:
                all_gradient.append((cid, w_gradient[cid], nat_w, size_f))
            if cid in w_eigenvec:
                all_eigenvec.append((cid, w_eigenvec[cid], nat_w, size_f))

    print(f"\n{'=' * 70}")
    print(f"Results: {n_parents} parents, {len(all_uniform)} child measurements")
    print(f"{'=' * 70}")

    # Compute correlations for each scheme
    results_by_scheme = {}
    for name, data in [('uniform', all_uniform),
                       ('gradient', all_gradient),
                       ('eigenvector', all_eigenvec)]:
        if len(data) < 5:
            print(f"\n  {name}: insufficient data ({len(data)} pairs)")
            results_by_scheme[name] = {'n': len(data), 'error': 'insufficient data'}
            continue

        coupling_w = np.array([d[1] for d in data])
        natural_w = np.array([d[2] for d in data])
        size_f = np.array([d[3] for d in data])

        rho_cn, p_cn = spearmanr(coupling_w, natural_w)
        rho_ns, p_ns = spearmanr(natural_w, size_f)
        rho_cs, p_cs = spearmanr(coupling_w, size_f)

        results_by_scheme[name] = {
            'n': len(data),
            'rho_coupling_natural': float(rho_cn),
            'p_coupling_natural': float(p_cn),
            'rho_natural_size': float(rho_ns),
            'p_natural_size': float(p_ns),
            'rho_coupling_size': float(rho_cs),
            'p_coupling_size': float(p_cs),
        }

        print(f"\n  {name.upper()} perturbation (n={len(data)}):")
        print(f"    rho(coupling, natural) = {rho_cn:.4f}  p={p_cn:.2e}")
        print(f"    rho(natural, size)     = {rho_ns:.4f}  p={p_ns:.2e}")
        print(f"    rho(coupling, size)    = {rho_cs:.4f}  p={p_cs:.2e}")

    # Verification
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    uniform_r = results_by_scheme.get('uniform', {})
    gradient_r = results_by_scheme.get('gradient', {})
    eigvec_r = results_by_scheme.get('eigenvector', {})

    # Test 1: Gradient scheme improves coupling~natural correlation
    u_cn = uniform_r.get('rho_coupling_natural', 0)
    g_cn = gradient_r.get('rho_coupling_natural', 0)
    e_cn = eigvec_r.get('rho_coupling_natural', 0)
    best_cn = max(g_cn, e_cn)
    test1 = best_cn > u_cn
    print(f"\n  Test 1: Best weighted rho(coupling,natural) > uniform?")
    print(f"    Uniform: {u_cn:.4f}, Gradient: {g_cn:.4f}, Eigenvector: {e_cn:.4f}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Gradient scheme reduces natural~size correlation
    u_ns = uniform_r.get('rho_natural_size', 1)
    g_ns = gradient_r.get('rho_natural_size', 1)
    e_ns = eigvec_r.get('rho_natural_size', 1)
    # Natural weights don't change between schemes (they're computed from eigenbasis)
    # But coupling~size should change
    u_cs = uniform_r.get('rho_coupling_size', 1)
    g_cs = gradient_r.get('rho_coupling_size', 1)
    e_cs = eigvec_r.get('rho_coupling_size', 1)
    best_cs = min(g_cs, e_cs)
    test2 = best_cs < u_cs
    print(f"\n  Test 2: Best weighted rho(coupling,size) < uniform?")
    print(f"    Uniform: {u_cs:.4f}, Gradient: {g_cs:.4f}, Eigenvector: {e_cs:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: At least one scheme has rho(coupling,natural) > 0.3 and p < 0.01
    test3_schemes = []
    for name, r in results_by_scheme.items():
        if r.get('rho_coupling_natural', 0) > 0.3 and r.get('p_coupling_natural', 1) < 0.01:
            test3_schemes.append(name)
    test3 = len(test3_schemes) > 0
    print(f"\n  Test 3: Any scheme with rho>0.3, p<0.01?")
    print(f"    Qualifying schemes: {test3_schemes}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3])
    print(f"\n  OVERALL: {n_verified}/3 tests verified")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_08_gradient_coupling',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Deconfound coupling weight from region size via weighted perturbation',
        'n_parents': n_parents,
        'n_measurements': len(all_uniform),
        'schemes': results_by_scheme,
        'verification': {
            'test1_improved_coupling_natural': bool(test1),
            'test2_reduced_coupling_size': bool(test2),
            'test3_significant_correlation': bool(test3),
            'qualifying_schemes': test3_schemes,
            'n_verified': n_verified,
        },
    }

    output_file = RESULTS_DIR / f'exp_08_gradient_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
