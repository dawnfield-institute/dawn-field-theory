"""
exp_09_eigenvalue_decay.py -- Confluent Identity Phase 4

PURPOSE:
    Data-driven determination of K_MODES. The current K_MODES=10 is arbitrary.
    This experiment computes full eigenspectra for all regions, applies knee
    detection, and reports the natural cutoff k* for identity fingerprinting.

METHOD:
    For each region with >20 cells:
    1. Compute full eigenspectrum of subgraph Laplacian (dense eigh)
    2. Normalized decay: lambda_i / lambda_1 for nonzero eigenvalues
    3. Knee detection via second derivative of decay curve
    4. Variance-explained criterion: k* where cumsum(lambda)/total > 0.95

VERIFICATION:
    Report whether K_MODES=10 is justified (within 1 std of median k*)
    or should be adjusted.

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
    graph_laplacian_subgraph, get_region_indices,
)


def full_eigendecomposition(L):
    """Compute ALL eigenvalues and eigenvectors of Laplacian (dense)."""
    L_dense = L.toarray() if hasattr(L, 'toarray') else L
    eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def knee_detection(values):
    """
    Find knee point via maximum second derivative.
    Returns index of knee in the values array.
    """
    if len(values) < 4:
        return len(values) - 1

    # Second derivative (discrete)
    d2 = np.diff(values, n=2)
    if len(d2) == 0:
        return len(values) - 1

    # Knee = point of maximum curvature (largest second derivative)
    knee_idx = int(np.argmax(np.abs(d2))) + 1  # +1 for diff offset
    return knee_idx


def variance_explained_cutoff(eigenvalues, threshold=0.95):
    """Find k* where cumulative sum of eigenvalues exceeds threshold fraction."""
    if len(eigenvalues) == 0:
        return 0

    cumsum = np.cumsum(eigenvalues)
    total = cumsum[-1]
    if total < 1e-15:
        return len(eigenvalues)

    ratios = cumsum / total
    above = np.where(ratios >= threshold)[0]
    if len(above) == 0:
        return len(eigenvalues)
    return int(above[0]) + 1  # +1 for 1-indexed count


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 4, Experiment 09")
    print("Eigenvalue Decay Analysis: Data-Driven K_MODES")
    print("=" * 70)

    # Load data
    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)

    # Analyze eigenvalue decay for all qualifying regions
    MIN_CELLS = 20
    MAX_CELLS_DENSE = 2000  # dense eigh limit

    results_per_region = []

    for level in range(len(labels_by_level)):
        labels = labels_by_level[level]
        region_ids = sorted(np.unique(labels).tolist())

        for rid in region_ids:
            indices = get_region_indices(labels_by_level, level, rid)
            n_cells = len(indices)

            if n_cells < MIN_CELLS or n_cells > MAX_CELLS_DENSE:
                continue

            L, _ = graph_laplacian_subgraph(adjacency, indices)
            eigenvalues, eigenvectors = full_eigendecomposition(L)

            # Nonzero eigenvalues only
            nonzero_mask = eigenvalues > 1e-10
            nonzero_eigs = eigenvalues[nonzero_mask]
            nonzero_vecs = eigenvectors[:, nonzero_mask]

            if len(nonzero_eigs) < 3:
                continue

            # Normalized eigenvalue decay
            normalized = nonzero_eigs / nonzero_eigs[0]

            # Knee detection on eigenvalue decay
            k_knee = knee_detection(normalized)

            # STATE VARIANCE EXPLAINED (the correct metric):
            # How many modes capture 95%/99% of the state's energy?
            state = state_flat[indices]
            state_centered = state - np.mean(state)
            state_energy = np.dot(state_centered, state_centered)

            if state_energy > 1e-15:
                # Compute coefficients |<state, v_i>|^2 for each mode
                coeffs_sq = np.array([
                    np.dot(state_centered, nonzero_vecs[:, i])**2
                    for i in range(nonzero_vecs.shape[1])
                ])
                k_state_95 = variance_explained_cutoff(coeffs_sq, 0.95)
                k_state_99 = variance_explained_cutoff(coeffs_sq, 0.99)
                frac_10 = float(np.sum(coeffs_sq[:min(10, len(coeffs_sq))]) / state_energy)
            else:
                k_state_95 = 1
                k_state_99 = 1
                frac_10 = 1.0

            # Spectral gap ratio
            gap_ratio = float(nonzero_eigs[0] / nonzero_eigs[-1]) if nonzero_eigs[-1] > 1e-15 else 0

            region_result = {
                'level': level,
                'region_id': int(rid),
                'n_cells': n_cells,
                'n_nonzero_eigenvalues': len(nonzero_eigs),
                'k_knee': int(k_knee),
                'k_state_95': int(k_state_95),
                'k_state_99': int(k_state_99),
                'frac_energy_k10': frac_10,
                'gap_ratio': gap_ratio,
                'top_10_normalized': normalized[:10].tolist() if len(normalized) >= 10 else normalized.tolist(),
                'eigenvalue_range': [float(nonzero_eigs[0]), float(nonzero_eigs[-1])],
            }
            results_per_region.append(region_result)

            print(f"  L{level} R{rid}: {n_cells} cells, "
                  f"k_state_95={k_state_95}, k_state_99={k_state_99}, "
                  f"energy_k10={frac_10:.1%}")

    # Aggregate statistics
    print(f"\n{'=' * 70}")
    print(f"Aggregate Analysis ({len(results_per_region)} regions)")
    print(f"{'=' * 70}")

    if len(results_per_region) == 0:
        print("  No qualifying regions found!")
        return

    k_knees = np.array([r['k_knee'] for r in results_per_region])
    k_s95 = np.array([r['k_state_95'] for r in results_per_region])
    k_s99 = np.array([r['k_state_99'] for r in results_per_region])
    frac_k10 = np.array([r['frac_energy_k10'] for r in results_per_region])

    print(f"\n  Eigenvalue knee detection (k_knee):")
    print(f"    mean={k_knees.mean():.1f}, median={np.median(k_knees):.1f}, "
          f"std={k_knees.std():.1f}")

    print(f"\n  STATE energy: 95% explained by k modes (k_state_95):")
    print(f"    mean={k_s95.mean():.1f}, median={np.median(k_s95):.1f}, "
          f"std={k_s95.std():.1f}")
    print(f"    min={k_s95.min()}, max={k_s95.max()}")

    print(f"\n  STATE energy: 99% explained by k modes (k_state_99):")
    print(f"    mean={k_s99.mean():.1f}, median={np.median(k_s99):.1f}, "
          f"std={k_s99.std():.1f}")

    print(f"\n  Energy captured by first 10 modes:")
    print(f"    mean={frac_k10.mean():.1%}, median={np.median(frac_k10):.1%}, "
          f"min={frac_k10.min():.1%}, max={frac_k10.max():.1%}")

    # Is K_MODES=10 justified?
    print(f"\n{'=' * 70}")
    print("Verification: Is K_MODES=10 appropriate?")
    print(f"{'=' * 70}")

    median_s95 = float(np.median(k_s95))
    k10_covers_95 = 10 >= median_s95

    # What fraction of regions have k_state_95 <= 10?
    frac_covered = float(np.mean(k_s95 <= 10))

    # What fraction of state energy do 10 modes capture on average?
    mean_frac = float(frac_k10.mean())
    k10_captures_majority = mean_frac > 0.5

    print(f"\n  K_MODES=10 vs 95% state energy: median_k_state_95={median_s95:.1f}")
    print(f"    10 >= median? {'[VERIFIED]' if k10_covers_95 else '[FAILED]'}")
    print(f"    Fraction of regions where k_state_95 <= 10: {frac_covered:.1%}")

    print(f"\n  Energy captured by K_MODES=10:")
    print(f"    Mean fraction: {mean_frac:.1%}")
    print(f"    Captures >50% of state energy? "
          f"{'[VERIFIED]' if k10_captures_majority else '[FAILED]'}")

    # Recommendation
    recommended_k = int(np.ceil(np.percentile(k_s95, 75)))
    print(f"\n  Recommended K_MODES (75th pct of k_state_95): {recommended_k}")

    if k10_covers_95:
        verdict = "K_MODES=10 is JUSTIFIED (captures 95% state energy for majority)"
    elif k10_captures_majority:
        verdict = f"K_MODES=10 is ADEQUATE (captures >{mean_frac:.0%} energy) but k={recommended_k} would be better"
    else:
        verdict = f"K_MODES=10 is INSUFFICIENT -- recommend k={recommended_k}"

    print(f"\n  VERDICT: {verdict}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_09_eigenvalue_decay',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Data-driven K_MODES determination',
        'n_regions_analyzed': len(results_per_region),
        'aggregate': {
            'k_knee': {'mean': float(k_knees.mean()), 'median': float(np.median(k_knees)),
                       'std': float(k_knees.std()), 'min': int(k_knees.min()),
                       'max': int(k_knees.max())},
            'k_state_95': {'mean': float(k_s95.mean()), 'median': median_s95,
                           'std': float(k_s95.std()), 'min': int(k_s95.min()),
                           'max': int(k_s95.max())},
            'k_state_99': {'mean': float(k_s99.mean()), 'median': float(np.median(k_s99)),
                           'std': float(k_s99.std())},
            'energy_k10': {'mean': float(frac_k10.mean()), 'median': float(np.median(frac_k10)),
                           'min': float(frac_k10.min()), 'max': float(frac_k10.max())},
        },
        'verification': {
            'k10_covers_95pct_state_energy': bool(k10_covers_95),
            'k10_captures_majority': bool(k10_captures_majority),
            'mean_energy_fraction_k10': float(mean_frac),
            'fraction_covered_by_k10': frac_covered,
            'recommended_k': recommended_k,
            'verdict': verdict,
        },
        'per_region': results_per_region,
    }

    output_file = RESULTS_DIR / f'exp_09_eigenvalue_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
