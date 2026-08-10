"""
exp_19_structured_perturbation.py -- Confluent Identity Phase 10

PURPOSE:
    Test identity response to structural perturbations (node removal, edge
    rewiring) vs additive Gaussian noise. All prior experiments use Gaussian
    only. Physical identity disruptions are topological — this validates whether
    the CI framework captures structural robustness.

METHODS:
    For each level-0 region (>=20 cells), apply 3 perturbation types:
    - Type A (Gaussian): N(0, 0.1*mean) noise — baseline comparison
    - Type B (Node removal): set 5/10/20% of cells to zero, recompute Laplacian
    - Type C (Edge rewiring): redistribute removed cell weight to neighbors

    Measure identity shift, Fiedler change per type. Cross-type rank correlation.

VERIFICATION:
    - Node removal produces larger identity shift than Gaussian (ratio > 1.2)
    - Kendall tau(Gaussian, node-removal rankings) > 0.3
    - Kendall tau(Gaussian, rewiring rankings) > 0.4
    - At 20% removal, >=50% of regions show Fiedler change > 10%

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import kendalltau

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices, compute_subgraph_laplacian_from_field,
)


def remove_nodes(indices, fraction, seed):
    """Remove a random fraction of nodes. Returns (remaining, removed)."""
    rng = np.random.RandomState(seed)
    n = len(indices)
    n_remove = max(1, int(n * fraction))
    remove_mask = rng.choice(n, n_remove, replace=False)
    removed = indices[remove_mask]
    remaining = np.delete(indices, remove_mask)
    return remaining, removed


def rewire_edges(state_flat, indices, removed_indices, N):
    """
    Redistribute removed cell values to their lattice neighbors within the region.
    Returns modified state_flat copy.
    """
    state_mod = state_flat.copy()
    index_set = set(int(i) for i in indices)

    for g in removed_indices:
        g = int(g)
        i, j = divmod(g, N)
        value = state_mod[g]

        # Find neighbors within the region
        neighbors_in_region = []
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = (i + di) % N, (j + dj) % N
            neighbor = ni * N + nj
            if neighbor in index_set and neighbor not in set(int(r) for r in removed_indices):
                neighbors_in_region.append(neighbor)

        if neighbors_in_region:
            share = value / len(neighbors_in_region)
            for nb in neighbors_in_region:
                state_mod[nb] += share

        state_mod[g] = 0.0

    return state_mod


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 10, Experiment 19")
    print("Structured Perturbation: Node Removal + Edge Rewiring")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)

    labels0 = labels_by_level[0]
    region_ids = sorted(np.unique(labels0).tolist())
    print(f"Level 0: {len(region_ids)} regions")

    removal_fractions = [0.05, 0.10, 0.20]
    region_data = []

    for rid in region_ids:
        indices = get_region_indices(labels_by_level, 0, rid)
        n_cells = len(indices)
        if n_cells < 20:
            continue

        # Baseline identity
        L_base, _ = graph_laplacian_subgraph(adjacency, indices)
        state_region = state_flat[indices]
        I_baseline = compute_spectral_identity(L_base, state_region)
        coeffs_base = np.array(I_baseline['state_coefficients'])
        fiedler_base = I_baseline['fiedler_value']
        coeff_norm = float(np.linalg.norm(coeffs_base))

        if coeff_norm < 1e-15:
            continue

        # --- Type A: Gaussian noise ---
        rng = np.random.RandomState(42 + rid)
        noise = rng.randn(n_cells) * 0.1 * np.mean(state_region)
        state_gauss = state_region + noise
        I_gauss = compute_spectral_identity(L_base, state_gauss)
        coeffs_gauss = np.array(I_gauss['state_coefficients'])
        min_len = min(len(coeffs_base), len(coeffs_gauss))
        gauss_shift = float(np.linalg.norm(
            coeffs_gauss[:min_len] - coeffs_base[:min_len])) / (coeff_norm + 1e-15)

        entry = {
            'region_id': int(rid),
            'n_cells': n_cells,
            'fiedler_base': float(fiedler_base),
            'gauss_shift': gauss_shift,
            'removal_results': {},
        }

        for frac in removal_fractions:
            remaining, removed = remove_nodes(indices, frac, seed=42 + rid)

            if len(remaining) < 10:
                continue

            # --- Type B: Node removal (recompute Laplacian on remaining) ---
            L_rem, _ = compute_subgraph_laplacian_from_field(
                state_flat, remaining, N)
            state_remaining = state_flat[remaining]
            I_removal = compute_spectral_identity(L_rem, state_remaining)
            coeffs_removal = np.array(I_removal['state_coefficients'])
            fiedler_removal = I_removal['fiedler_value']

            # Compare coefficients (may have different lengths due to fewer nodes)
            min_len_r = min(len(coeffs_base), len(coeffs_removal))
            removal_shift = float(np.linalg.norm(
                coeffs_removal[:min_len_r] - coeffs_base[:min_len_r])) / (coeff_norm + 1e-15)
            fiedler_change_removal = abs(fiedler_removal - fiedler_base) / (fiedler_base + 1e-15)

            # --- Type C: Edge rewiring ---
            state_rewired = rewire_edges(state_flat, indices, removed, N)
            # Use full region (all indices) but with redistributed values
            state_rewired_region = state_rewired[indices]
            I_rewire = compute_spectral_identity(L_base, state_rewired_region)
            coeffs_rewire = np.array(I_rewire['state_coefficients'])
            fiedler_rewire = I_rewire['fiedler_value']

            min_len_w = min(len(coeffs_base), len(coeffs_rewire))
            rewire_shift = float(np.linalg.norm(
                coeffs_rewire[:min_len_w] - coeffs_base[:min_len_w])) / (coeff_norm + 1e-15)
            fiedler_change_rewire = abs(fiedler_rewire - fiedler_base) / (fiedler_base + 1e-15)

            entry['removal_results'][str(frac)] = {
                'n_removed': len(removed),
                'n_remaining': len(remaining),
                'removal_shift': removal_shift,
                'rewire_shift': rewire_shift,
                'gauss_shift': gauss_shift,
                'removal_vs_gauss_ratio': removal_shift / (gauss_shift + 1e-15),
                'fiedler_change_removal': fiedler_change_removal,
                'fiedler_change_rewire': fiedler_change_rewire,
            }

        region_data.append(entry)

    n_regions = len(region_data)
    print(f"\nAnalyzed {n_regions} regions (>= 20 cells)")

    # --- Aggregate analysis at 10% removal ---
    print(f"\n{'=' * 70}")
    print("Aggregate Analysis (10% removal)")
    print(f"{'=' * 70}")

    gauss_shifts = []
    removal_shifts = []
    rewire_shifts = []

    for rd in region_data:
        res = rd['removal_results'].get('0.1')
        if res:
            gauss_shifts.append(res['gauss_shift'])
            removal_shifts.append(res['removal_shift'])
            rewire_shifts.append(res['rewire_shift'])

    gauss_shifts = np.array(gauss_shifts)
    removal_shifts = np.array(removal_shifts)
    rewire_shifts = np.array(rewire_shifts)
    n_compare = len(gauss_shifts)

    print(f"  Regions with 10% removal data: {n_compare}")
    print(f"  Mean shifts: Gaussian={gauss_shifts.mean():.4f}, "
          f"Removal={removal_shifts.mean():.4f}, "
          f"Rewiring={rewire_shifts.mean():.4f}")

    mean_ratio = float(removal_shifts.mean() / (gauss_shifts.mean() + 1e-15))
    print(f"  Mean removal/Gaussian ratio: {mean_ratio:.2f}x")

    # Cross-type ranking correlations
    tau_gr, p_gr = kendalltau(gauss_shifts, removal_shifts)
    tau_gw, p_gw = kendalltau(gauss_shifts, rewire_shifts)
    print(f"\n  Kendall tau(Gaussian, node-removal): {tau_gr:.4f}, p={p_gr:.2e}")
    print(f"  Kendall tau(Gaussian, edge-rewiring): {tau_gw:.4f}, p={p_gw:.2e}")

    # --- 20% removal: Fiedler disruption ---
    print(f"\n{'=' * 70}")
    print("Fiedler Disruption at 20% Removal")
    print(f"{'=' * 70}")

    fiedler_changes_20 = []
    for rd in region_data:
        res = rd['removal_results'].get('0.2')
        if res:
            fiedler_changes_20.append(res['fiedler_change_removal'])

    fiedler_changes_20 = np.array(fiedler_changes_20)
    n_large_change = int(np.sum(fiedler_changes_20 > 0.10))
    frac_large = n_large_change / (len(fiedler_changes_20) + 1e-15)
    print(f"  Regions with Fiedler change > 10%: {n_large_change}/{len(fiedler_changes_20)} ({frac_large:.1%})")
    if len(fiedler_changes_20) > 0:
        print(f"  Mean Fiedler change: {fiedler_changes_20.mean():.4f}")
        print(f"  Median Fiedler change: {np.median(fiedler_changes_20):.4f}")

    # --- Verification ---
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    test1 = mean_ratio > 1.2
    print(f"\n  Test 1: Node removal shift > 1.2x Gaussian?")
    print(f"    Ratio: {mean_ratio:.2f}x")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    test2 = tau_gr > 0.3
    print(f"\n  Test 2: Kendall tau(Gaussian, node-removal) > 0.3?")
    print(f"    tau={tau_gr:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    test3 = tau_gw > 0.4
    print(f"\n  Test 3: Kendall tau(Gaussian, edge-rewiring) > 0.4?")
    print(f"    tau={tau_gw:.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    test4 = frac_large >= 0.50
    print(f"\n  Test 4: >= 50% of regions show > 10% Fiedler change at 20% removal?")
    print(f"    Fraction: {frac_large:.1%}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 structured perturbation tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_19_structured_perturbation',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Structured perturbation: node removal vs edge rewiring vs Gaussian',
        'n_regions': n_regions,
        'removal_fractions': removal_fractions,
        'aggregate_10pct': {
            'n_compare': n_compare,
            'mean_gauss_shift': float(gauss_shifts.mean()) if n_compare > 0 else None,
            'mean_removal_shift': float(removal_shifts.mean()) if n_compare > 0 else None,
            'mean_rewire_shift': float(rewire_shifts.mean()) if n_compare > 0 else None,
            'removal_vs_gauss_ratio': mean_ratio,
            'kendall_gauss_removal': {'tau': float(tau_gr), 'p': float(p_gr)},
            'kendall_gauss_rewiring': {'tau': float(tau_gw), 'p': float(p_gw)},
        },
        'fiedler_disruption_20pct': {
            'n_regions': len(fiedler_changes_20),
            'n_large_change': n_large_change,
            'fraction_large_change': float(frac_large),
        },
        'verification': {
            'test1_removal_larger_than_gaussian': bool(test1),
            'test2_gauss_removal_correlation': bool(test2),
            'test3_gauss_rewiring_correlation': bool(test3),
            'test4_fiedler_disruption': bool(test4),
            'n_verified': n_verified,
        },
        'per_region': region_data,
    }

    output_file = RESULTS_DIR / f'exp_19_structured_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
