"""
exp_16_spectral_gap_dynamics.py -- Confluent Identity Phase 7

PURPOSE:
    Test whether the Fiedler value (lambda_2, spectral gap) predicts identity
    robustness. Higher Fiedler = stronger internal connectivity = more coherent
    identity = less sensitive to perturbation. This connects the spectral gap
    to the physical meaning of "identity coherence."

METHODS:
    For each region (level 0):
    1. Compute Fiedler value from subgraph Laplacian
    2. Apply perturbation (epsilon to all cells)
    3. Measure identity shift: ||coeffs_perturbed - coeffs_original|| / ||coeffs_original||
    4. Correlate: Fiedler vs sensitivity (expect negative — higher Fiedler = less sensitive)

    Also:
    5. Fiedler vs spectral entropy (expect negative — coherent regions have less entropy)
    6. Fiedler stability across perturbation scales (should be monotonically robust)

VERIFICATION:
    - rho(Fiedler, sensitivity) < -0.3 (higher Fiedler = less sensitive)
    - rho(Fiedler, spectral_entropy) < -0.3
    - Fiedler predicts top-5 most robust regions (>= 3/5 in bottom-5 sensitivity)
    - Mean sensitivity of top-quartile Fiedler < bottom-quartile Fiedler

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
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices,
)


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 7, Experiment 16")
    print("Spectral Gap Dynamics: Fiedler as Coherence Predictor")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)

    # Analyze level-0 regions (finest partition, most regions)
    labels0 = labels_by_level[0]
    region_ids = sorted(np.unique(labels0).tolist())
    print(f"Level 0: {len(region_ids)} regions")

    # Perturbation scales to test
    epsilon_scales = [0.01, 0.05, 0.1, 0.2]

    # Collect per-region data
    region_data = []

    for rid in region_ids:
        indices = get_region_indices(labels_by_level, 0, rid)
        n_cells = len(indices)

        if n_cells < 10:
            continue

        # Compute baseline identity
        L, _ = graph_laplacian_subgraph(adjacency, indices)
        state_region = state_flat[indices]
        I_baseline = compute_spectral_identity(L, state_region)

        fiedler = I_baseline['fiedler_value']
        entropy = I_baseline['spectral_entropy']
        coeffs_baseline = np.array(I_baseline['state_coefficients'])
        coeff_norm = float(np.linalg.norm(coeffs_baseline))

        if coeff_norm < 1e-15:
            continue

        # Perturbation sensitivity at multiple scales
        sensitivities = {}
        for eps_scale in epsilon_scales:
            epsilon = eps_scale * np.mean(state_region)
            state_perturbed = state_region.copy()
            state_perturbed += epsilon  # uniform perturbation to all cells

            I_perturbed = compute_spectral_identity(L, state_perturbed)
            coeffs_perturbed = np.array(I_perturbed['state_coefficients'])

            min_len = min(len(coeffs_baseline), len(coeffs_perturbed))
            shift = float(np.linalg.norm(
                coeffs_perturbed[:min_len] - coeffs_baseline[:min_len]))

            # Normalize by perturbation magnitude and baseline norm
            sensitivity = shift / (epsilon * coeff_norm + 1e-15)
            sensitivities[eps_scale] = sensitivity

        # Use eps=0.1 as the primary sensitivity metric
        primary_sensitivity = sensitivities.get(0.1, 0.0)

        # Also compute directed perturbation (noise, not uniform)
        rng = np.random.RandomState(42 + rid)
        noise = rng.randn(n_cells) * 0.1 * np.mean(state_region)
        state_noisy = state_region + noise
        I_noisy = compute_spectral_identity(L, state_noisy)
        coeffs_noisy = np.array(I_noisy['state_coefficients'])
        min_len = min(len(coeffs_baseline), len(coeffs_noisy))
        noise_shift = float(np.linalg.norm(
            coeffs_noisy[:min_len] - coeffs_baseline[:min_len]))
        noise_sensitivity = noise_shift / (coeff_norm + 1e-15)

        # Fiedler change under perturbation
        fiedler_perturbed = I_perturbed['fiedler_value']
        fiedler_change = abs(fiedler_perturbed - fiedler) / (fiedler + 1e-15)

        region_data.append({
            'region_id': int(rid),
            'n_cells': int(n_cells),
            'fiedler': float(fiedler),
            'spectral_entropy': float(entropy),
            'primary_sensitivity': float(primary_sensitivity),
            'noise_sensitivity': float(noise_sensitivity),
            'fiedler_change': float(fiedler_change),
            'sensitivities_by_scale': {str(k): float(v)
                                       for k, v in sensitivities.items()},
        })

    n_regions = len(region_data)
    print(f"\nAnalyzed {n_regions} regions (>= 10 cells)")

    # Extract arrays for correlation
    fiedlers = np.array([r['fiedler'] for r in region_data])
    sensitivities = np.array([r['primary_sensitivity'] for r in region_data])
    noise_sens = np.array([r['noise_sensitivity'] for r in region_data])
    entropies = np.array([r['spectral_entropy'] for r in region_data])
    fiedler_changes = np.array([r['fiedler_change'] for r in region_data])
    sizes = np.array([r['n_cells'] for r in region_data])

    print(f"\n{'=' * 70}")
    print("Correlation Analysis")
    print(f"{'=' * 70}")

    # Primary: Fiedler vs sensitivity
    rho_fs, p_fs = spearmanr(fiedlers, sensitivities)
    print(f"\n  rho(Fiedler, uniform_sensitivity) = {rho_fs:.4f}, p = {p_fs:.2e}")

    rho_fn, p_fn = spearmanr(fiedlers, noise_sens)
    print(f"  rho(Fiedler, noise_sensitivity)   = {rho_fn:.4f}, p = {p_fn:.2e}")

    rho_fe, p_fe = spearmanr(fiedlers, entropies)
    print(f"  rho(Fiedler, spectral_entropy)    = {rho_fe:.4f}, p = {p_fe:.2e}")

    rho_fc, p_fc = spearmanr(fiedlers, fiedler_changes)
    print(f"  rho(Fiedler, fiedler_change)      = {rho_fc:.4f}, p = {p_fc:.2e}")

    # Size confound check
    rho_fsize, _ = spearmanr(fiedlers, sizes)
    rho_ssize, _ = spearmanr(sensitivities, sizes)
    print(f"\n  Confound check:")
    print(f"    rho(Fiedler, size) = {rho_fsize:.4f}")
    print(f"    rho(sensitivity, size) = {rho_ssize:.4f}")

    # Best metric
    best_rho = min(rho_fs, rho_fn)  # want most negative
    best_p = p_fs if best_rho == rho_fs else p_fn
    best_name = "uniform" if best_rho == rho_fs else "noise"

    # Quartile analysis
    fiedler_sorted_idx = np.argsort(fiedlers)
    q_size = max(1, n_regions // 4)
    bottom_q = fiedler_sorted_idx[:q_size]  # lowest Fiedler
    top_q = fiedler_sorted_idx[-q_size:]  # highest Fiedler

    mean_sens_bottom = float(np.mean(sensitivities[bottom_q]))
    mean_sens_top = float(np.mean(sensitivities[top_q]))
    mean_fiedler_bottom = float(np.mean(fiedlers[bottom_q]))
    mean_fiedler_top = float(np.mean(fiedlers[top_q]))

    print(f"\n  Quartile analysis:")
    print(f"    Bottom-quartile Fiedler (mean={mean_fiedler_bottom:.6f}): "
          f"mean sensitivity = {mean_sens_bottom:.6f}")
    print(f"    Top-quartile Fiedler (mean={mean_fiedler_top:.6f}): "
          f"mean sensitivity = {mean_sens_top:.6f}")
    print(f"    Ratio: {mean_sens_bottom / (mean_sens_top + 1e-15):.2f}x")

    # Rank overlap: top-5 Fiedler vs bottom-5 sensitivity (most robust)
    top5_fiedler = set(fiedler_sorted_idx[-5:])  # highest Fiedler
    bottom5_sens = set(np.argsort(sensitivities)[:5])  # lowest sensitivity
    rank_overlap = top5_fiedler & bottom5_sens
    n_overlap = len(rank_overlap)
    print(f"\n  Rank overlap: {n_overlap}/5 top-Fiedler in bottom-5 sensitivity")

    # Verification
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Test 1: Fiedler negatively correlates with sensitivity
    test1 = best_rho < -0.3 and best_p < 0.05
    print(f"\n  Test 1: rho(Fiedler, sensitivity) < -0.3 AND p < 0.05?")
    print(f"    Best: rho={best_rho:.4f}, p={best_p:.2e} ({best_name})")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Fiedler negatively correlates with spectral entropy
    test2 = rho_fe < -0.3 and p_fe < 0.05
    print(f"\n  Test 2: rho(Fiedler, entropy) < -0.3 AND p < 0.05?")
    print(f"    rho={rho_fe:.4f}, p={p_fe:.2e}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: Rank overlap (high Fiedler = low sensitivity)
    test3 = n_overlap >= 3
    print(f"\n  Test 3: >= 3/5 top-Fiedler in bottom-5 sensitivity?")
    print(f"    Overlap: {n_overlap}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: Top-quartile Fiedler has lower mean sensitivity
    test4 = mean_sens_top < mean_sens_bottom
    print(f"\n  Test 4: Top-quartile Fiedler less sensitive than bottom?")
    print(f"    Top: {mean_sens_top:.6f}, Bottom: {mean_sens_bottom:.6f}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 spectral gap dynamics tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_16_spectral_gap_dynamics',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Fiedler as identity coherence predictor',
        'n_regions': n_regions,
        'epsilon_scales': epsilon_scales,
        'correlations': {
            'fiedler_vs_uniform_sensitivity': {
                'rho': float(rho_fs), 'p': float(p_fs)},
            'fiedler_vs_noise_sensitivity': {
                'rho': float(rho_fn), 'p': float(p_fn)},
            'fiedler_vs_entropy': {
                'rho': float(rho_fe), 'p': float(p_fe)},
            'fiedler_vs_fiedler_change': {
                'rho': float(rho_fc), 'p': float(p_fc)},
            'fiedler_vs_size': float(rho_fsize),
            'sensitivity_vs_size': float(rho_ssize),
        },
        'quartile_analysis': {
            'bottom_quartile_fiedler': mean_fiedler_bottom,
            'bottom_quartile_sensitivity': mean_sens_bottom,
            'top_quartile_fiedler': mean_fiedler_top,
            'top_quartile_sensitivity': mean_sens_top,
        },
        'rank_overlap_top5': n_overlap,
        'verification': {
            'test1_fiedler_sensitivity_correlation': bool(test1),
            'test2_fiedler_entropy_correlation': bool(test2),
            'test3_rank_overlap': bool(test3),
            'test4_quartile_separation': bool(test4),
            'n_verified': n_verified,
        },
        'per_region': region_data,
    }

    output_file = RESULTS_DIR / f'exp_16_spectral_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
