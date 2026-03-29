"""
exp_24_perturbation_mode_decomposition.py -- Confluent Identity Phase 15

PURPOSE:
    Perturbation types are independent (exp_19: tau~0). PAC hypothesis: Gaussian
    and node-removal perturbations activate orthogonal eigenspaces — the Hodge
    decomposition manifesting. Amplitude perturbations (exact component) and
    topological perturbations (co-exact component) live in orthogonal subspaces.

METHODS:
    For each level-0 region (>=20 cells):
    1. Compute baseline spectral coefficients c_base
    2. Apply Gaussian perturbation -> c_gauss. Shift: delta_gauss = c_gauss - c_base
    3. Apply node removal (10%) -> c_removal. Shift: delta_removal = c_removal - c_base
    4. Apply edge rewiring (10%) -> c_rewire. Shift: delta_rewire = c_rewire - c_base
    5. Cosine similarity between shift vectors
    6. Project shifts onto eigenmode bands: low (1-3), mid (4-7), high (8-10)
    7. Identify which perturbation type activates which band

VERIFICATION:
    - Mean |cos(delta_gauss, delta_removal)| < 0.3 (near-orthogonal)
    - Mean |cos(delta_gauss, delta_rewire)| < 0.4
    - Perturbation types activate different eigenmode bands (>= 60% of regions)
    - Mean |cos(delta_removal, delta_rewire)| > 0.3 (topological share structure)

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices, compute_subgraph_laplacian_from_field,
)
from exp_19_structured_perturbation import remove_nodes, rewire_edges


def cosine_similarity(a, b):
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def band_energy(coeffs, band_start, band_end):
    """Sum of squared coefficients in a band [start, end)."""
    valid = coeffs[band_start:band_end]
    return float(np.sum(valid ** 2))


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 15, Experiment 24")
    print("Perturbation Mode Decomposition: Hodge Orthogonality")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)

    labels0 = labels_by_level[0]
    region_ids = sorted(np.unique(labels0).tolist())

    region_results = []
    band_names = ['low (1-3)', 'mid (4-7)', 'high (8-10)']
    band_ranges = [(0, 3), (3, 7), (7, 10)]

    for rid in region_ids:
        indices = get_region_indices(labels_by_level, 0, rid)
        n_cells = len(indices)
        if n_cells < 20:
            continue

        # Baseline
        L_base, _ = graph_laplacian_subgraph(adjacency, indices)
        state_region = state_flat[indices]
        I_base = compute_spectral_identity(L_base, state_region)
        coeffs_base = np.array(I_base['state_coefficients'])
        if np.linalg.norm(coeffs_base) < 1e-15:
            continue

        # --- Gaussian perturbation ---
        rng = np.random.RandomState(42 + rid)
        noise = rng.randn(n_cells) * 0.1 * np.mean(state_region)
        state_gauss = state_region + noise
        I_gauss = compute_spectral_identity(L_base, state_gauss)
        coeffs_gauss = np.array(I_gauss['state_coefficients'])

        # --- Node removal (10%) ---
        remaining, removed = remove_nodes(indices, 0.10, seed=42 + rid)
        if len(remaining) < 10:
            continue

        L_rem, _ = compute_subgraph_laplacian_from_field(state_flat, remaining, N)
        state_remaining = state_flat[remaining]
        I_removal = compute_spectral_identity(L_rem, state_remaining)
        coeffs_removal = np.array(I_removal['state_coefficients'])

        # --- Edge rewiring (10%) ---
        state_rewired = rewire_edges(state_flat, indices, removed, N)
        state_rewired_region = state_rewired[indices]
        I_rewire = compute_spectral_identity(L_base, state_rewired_region)
        coeffs_rewire = np.array(I_rewire['state_coefficients'])

        # Compute shift vectors (truncate to common length)
        min_len = min(len(coeffs_base), len(coeffs_gauss),
                      len(coeffs_removal), len(coeffs_rewire))
        if min_len < 3:
            continue

        delta_gauss = coeffs_gauss[:min_len] - coeffs_base[:min_len]
        delta_removal = coeffs_removal[:min_len] - coeffs_base[:min_len]
        delta_rewire = coeffs_rewire[:min_len] - coeffs_base[:min_len]

        # Cosine similarities
        cos_gr = cosine_similarity(delta_gauss, delta_removal)
        cos_gw = cosine_similarity(delta_gauss, delta_rewire)
        cos_rw = cosine_similarity(delta_removal, delta_rewire)

        # Band energy fractions for each perturbation type
        band_fracs = {}
        for name, delta in [('gauss', delta_gauss),
                            ('removal', delta_removal),
                            ('rewire', delta_rewire)]:
            total_energy = float(np.sum(delta ** 2))
            if total_energy < 1e-30:
                band_fracs[name] = [0.0, 0.0, 0.0]
                continue
            fracs = []
            for b_start, b_end in band_ranges:
                be = band_energy(delta, b_start, min(b_end, min_len))
                fracs.append(be / total_energy)
            band_fracs[name] = fracs

        # Peak band for each perturbation type
        peak_bands = {}
        for name in ['gauss', 'removal', 'rewire']:
            fracs = band_fracs[name]
            if sum(fracs) > 0:
                peak_bands[name] = int(np.argmax(fracs))
            else:
                peak_bands[name] = -1

        # Do perturbation types activate different bands?
        different_bands = (peak_bands['gauss'] != peak_bands['removal'])

        region_results.append({
            'region_id': int(rid),
            'n_cells': n_cells,
            'cos_gauss_removal': cos_gr,
            'cos_gauss_rewire': cos_gw,
            'cos_removal_rewire': cos_rw,
            'band_fracs': band_fracs,
            'peak_bands': peak_bands,
            'different_bands': different_bands,
        })

    n_regions = len(region_results)
    print(f"\nAnalyzed {n_regions} regions")

    # =====================================================================
    # Aggregate
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Aggregate Cosine Similarities")
    print(f"{'=' * 70}")

    cos_gr_all = np.array([r['cos_gauss_removal'] for r in region_results])
    cos_gw_all = np.array([r['cos_gauss_rewire'] for r in region_results])
    cos_rw_all = np.array([r['cos_removal_rewire'] for r in region_results])

    mean_abs_cos_gr = float(np.mean(np.abs(cos_gr_all)))
    mean_abs_cos_gw = float(np.mean(np.abs(cos_gw_all)))
    mean_abs_cos_rw = float(np.mean(np.abs(cos_rw_all)))

    print(f"  Mean |cos(gauss, removal)| = {mean_abs_cos_gr:.4f}")
    print(f"  Mean |cos(gauss, rewire)|  = {mean_abs_cos_gw:.4f}")
    print(f"  Mean |cos(removal, rewire)| = {mean_abs_cos_rw:.4f}")

    # Band activation summary
    print(f"\n{'=' * 70}")
    print("Eigenmode Band Activation")
    print(f"{'=' * 70}")

    n_different = sum(1 for r in region_results if r['different_bands'])
    frac_different = n_different / (n_regions + 1e-15)
    print(f"  Regions where gauss vs removal activate different bands: "
          f"{n_different}/{n_regions} ({frac_different:.1%})")

    # Mean band fractions per perturbation type
    for ptype in ['gauss', 'removal', 'rewire']:
        mean_fracs = np.mean(
            [r['band_fracs'][ptype] for r in region_results], axis=0)
        print(f"  {ptype:>10}: low={mean_fracs[0]:.3f}, "
              f"mid={mean_fracs[1]:.3f}, high={mean_fracs[2]:.3f}")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    test1 = mean_abs_cos_gr < 0.3
    print(f"\n  Test 1: Mean |cos(gauss, removal)| < 0.3?")
    print(f"    {mean_abs_cos_gr:.4f}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    test2 = mean_abs_cos_gw < 0.4
    print(f"\n  Test 2: Mean |cos(gauss, rewire)| < 0.4?")
    print(f"    {mean_abs_cos_gw:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    test3 = frac_different >= 0.60
    print(f"\n  Test 3: >= 60% of regions activate different bands?")
    print(f"    {frac_different:.1%}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    test4 = mean_abs_cos_rw > 0.3
    print(f"\n  Test 4: Mean |cos(removal, rewire)| > 0.3?")
    print(f"    {mean_abs_cos_rw:.4f}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 mode decomposition tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_24_perturbation_mode_decomposition',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Hodge decomposition of vulnerability: orthogonal perturbation modes',
        'n_regions': n_regions,
        'cosine_similarities': {
            'mean_abs_cos_gauss_removal': mean_abs_cos_gr,
            'mean_abs_cos_gauss_rewire': mean_abs_cos_gw,
            'mean_abs_cos_removal_rewire': mean_abs_cos_rw,
        },
        'band_activation': {
            'n_different_bands': n_different,
            'frac_different': float(frac_different),
        },
        'verification': {
            'test1_gauss_removal_orthogonal': bool(test1),
            'test2_gauss_rewire_orthogonal': bool(test2),
            'test3_different_band_activation': bool(test3),
            'test4_topological_share_structure': bool(test4),
            'n_verified': n_verified,
        },
        'per_region': region_results,
    }

    output_file = RESULTS_DIR / f'exp_24_mode_decomp_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
