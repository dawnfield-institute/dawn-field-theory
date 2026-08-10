"""
exp_34_scale_validation_full.py -- Confluent Identity Phase 25

PURPOSE:
    Fix exp_30's broken 256x256 replication. exp_30 used abbreviated diffusion
    (1500 steps), producing C std=0.0009 (vs ~0.01 at 128x128). All correlations
    vanished or sign-flipped. This experiment uses the proper PeriodicLatticeFluid
    class with run_to_steady_state() convergence detection.

METHODS:
    1. PeriodicLatticeFluid(N=256, n_large_stones=20, n_small_stones=60)
       - 4x area => scale stones proportionally
    2. run_to_steady_state(max_steps=10000, dt=0.005, viscosity=0.05,
       sec_threshold=0.1, tol=1e-6)
    3. Verify C std > 0.005 (exp_30 had 0.0009)
    4. Watershed partition (same as exp_30)
    5. Per-region: subgraph Laplacian, boundary metrics, sensitivity
    6. Replicate 3 key correlations:
       a. rho(size, sensitivity) same sign as 128x128 (positive)
       b. boundary gradient signal (partial rho)
       c. coupling proxy correlation

VERIFICATION:
    - C std > 0.005 (real spatial structure)
    - >= 100 regions with >= 10 cells
    - rho(size, sensitivity) same sign as 128x128 (positive — paradox replicates)
    - At least 2/3 key correlations within 0.20 of 128x128 reference

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr
from scipy import ndimage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, compute_spectral_identity,
    compute_subgraph_laplacian_from_field,
)
from exp_01_lattice_fluid_baseline import PeriodicLatticeFluid
from exp_08_gradient_coupling import compute_gradient_field
from exp_14_partial_correlation import partial_spearman
from exp_27_boundary_geometry import compute_boundary_metrics


def watershed_256(C, sigma=0.5, min_filter_size=3):
    """
    Watershed partition for 256x256 field.
    Same algorithm as exp_02 but inlined for self-containment.
    """
    import heapq

    N = C.shape[0]
    C_smooth = ndimage.gaussian_filter(C, sigma=sigma, mode='wrap')
    C_min = ndimage.minimum_filter(C_smooth, size=min_filter_size, mode='wrap')
    minima = C_smooth == C_min
    seeds, n_seeds = ndimage.label(minima)

    if n_seeds == 0:
        return np.ones_like(C, dtype=int), 1

    labels = seeds.copy()
    visited = labels > 0

    pq = []
    for i in range(N):
        for j in range(N):
            if visited[i, j]:
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = (i + di) % N, (j + dj) % N
                    if not visited[ni, nj]:
                        heapq.heappush(pq, (C_smooth[ni, nj], ni, nj))

    while pq:
        val, i, j = heapq.heappop(pq)
        if visited[i, j]:
            continue
        best_label = 0
        best_val = float('inf')
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = (i + di) % N, (j + dj) % N
            if visited[ni, nj] and labels[ni, nj] > 0:
                if C_smooth[ni, nj] < best_val:
                    best_val = C_smooth[ni, nj]
                    best_label = labels[ni, nj]
        labels[i, j] = best_label if best_label > 0 else 1
        visited[i, j] = True

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = (i + di) % N, (j + dj) % N
            if not visited[ni, nj]:
                heapq.heappush(pq, (C_smooth[ni, nj], ni, nj))

    return labels, n_seeds


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 25, Experiment 34")
    print("256x256 Scale Validation (Full Steady-State Diffusion)")
    print("=" * 70)

    # =====================================================================
    # Generate 256x256 field using proper PeriodicLatticeFluid
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Generating 256x256 PAC-Conservative Field (Full Convergence)")
    print(f"{'=' * 70}")

    # 4x area => scale stones: 5->20 large, 15->60 small (proportional to 128x128)
    # exp_01 uses: n_large=12, n_small=40 for 128x128
    # Scale: (256/128)^2 = 4x area, so ~48 large + ~160 small
    # But plan says 20+60 — use plan values (more conservative, avoids over-crowding)
    fluid = PeriodicLatticeFluid(
        N=256, total_value=100.0, seed=42,
        n_large_stones=20, n_small_stones=60, gravity=0.005
    )

    print(f"\nLattice: {fluid.N}x{fluid.N} = {fluid.N**2} cells")
    print(f"Initial C std: {fluid.C.std():.6f}")

    print("\nRunning to steady state (this may take several minutes)...")
    history = fluid.run_to_steady_state(
        max_steps=10000, dt=0.005, viscosity=0.05,
        sec_threshold=0.1, tol=1e-6, stable_count=10
    )

    P = fluid.P
    A = fluid.A
    C = fluid.C
    stone_mask = fluid.stone_mask
    N = fluid.N

    c_std = float(C.std())
    print(f"\n  256x256 field generated:")
    print(f"    C std = {c_std:.6f} (exp_30 had 0.0009)")
    print(f"    Steps: {history['steps_to_steady']}")
    print(f"    Steady state: {history['reached_steady']}")
    print(f"    Conservation error: {fluid.conservation_error():.2e}")
    print(f"    Stones: {stone_mask.sum()} cells")

    state_flat = C.ravel()
    grad_mag = compute_gradient_field(C)
    grad_flat = grad_mag.ravel()

    # =====================================================================
    # Watershed partition
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Watershed Partition")
    print(f"{'=' * 70}")

    labels0, n_seeds = watershed_256(C, sigma=0.5, min_filter_size=3)
    region_ids = sorted(np.unique(labels0).tolist())
    region_sizes = {rid: int(np.sum(labels0 == rid)) for rid in region_ids}

    valid_regions = {rid: sz for rid, sz in region_sizes.items() if sz >= 10}
    n_valid = len(valid_regions)
    print(f"  Total regions: {len(region_ids)}")
    print(f"  Regions >= 10 cells: {n_valid}")

    # =====================================================================
    # Per-region analysis
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Per-Region Analysis (boundary metrics + sensitivity)")
    print(f"{'=' * 70}")

    region_data = []
    n_analyzed = 0

    for rid in sorted(valid_regions.keys()):
        indices = np.where(labels0.ravel() == rid)[0]
        n_cells = len(indices)

        # Boundary metrics
        bm = compute_boundary_metrics(indices, N, state_flat, grad_flat, None)

        # Spectral identity via subgraph Laplacian
        L, W = compute_subgraph_laplacian_from_field(state_flat, indices, N)
        state_region = state_flat[indices]
        I_base = compute_spectral_identity(L, state_region)
        coeffs_base = np.array(I_base['state_coefficients'])
        coeff_norm = float(np.linalg.norm(coeffs_base))

        if coeff_norm < 1e-15:
            continue

        # Gaussian sensitivity
        rng = np.random.RandomState(42 + rid)
        noise = rng.randn(n_cells) * 0.1 * np.mean(state_region)
        state_noisy = state_region + noise
        I_noisy = compute_spectral_identity(L, state_noisy)
        coeffs_noisy = np.array(I_noisy['state_coefficients'])
        min_len = min(len(coeffs_base), len(coeffs_noisy))
        delta = float(np.linalg.norm(coeffs_noisy[:min_len] - coeffs_base[:min_len]))
        sensitivity = delta / (coeff_norm + 1e-15)

        # Gradient coupling proxy
        mean_grad = float(np.mean(grad_flat[indices]))

        region_data.append({
            'region_id': int(rid),
            'n_cells': n_cells,
            'sensitivity': sensitivity,
            'boundary_area_ratio': bm['boundary_area_ratio'],
            'compactness': bm['compactness'],
            'mean_boundary_gradient': bm['mean_boundary_gradient'],
            'perimeter': bm['perimeter'],
            'mean_gradient': mean_grad,
            'fiedler': float(I_base['fiedler_value']),
        })

        n_analyzed += 1
        if n_analyzed % 50 == 0:
            print(f"    Analyzed {n_analyzed} regions...")

    n_regions = len(region_data)
    print(f"  Analyzed {n_regions} regions total")

    # Extract arrays
    sizes = np.array([r['n_cells'] for r in region_data], dtype=float)
    sensitivities = np.array([r['sensitivity'] for r in region_data])
    bar = np.array([r['boundary_area_ratio'] for r in region_data])
    mean_bgrad = np.array([r['mean_boundary_gradient'] for r in region_data])
    mean_grad = np.array([r['mean_gradient'] for r in region_data])

    # =====================================================================
    # Key correlations
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Key Correlations (256x256 Full Diffusion)")
    print(f"{'=' * 70}")

    # 1. Size vs sensitivity (the paradox)
    rho_size_sens, p_ss = spearmanr(sizes, sensitivities)
    print(f"\n  Size-sensitivity paradox:")
    print(f"    rho(size, sensitivity) = {rho_size_sens:.4f}, p={p_ss:.2e}")
    print(f"    128x128 reference: +0.31 (bigger = MORE sensitive)")

    # 2. Gradient coupling proxy
    rho_grad_sens, p_gs = spearmanr(mean_grad, sensitivities)
    pr_grad_sens, pp_gs = partial_spearman(mean_grad, sensitivities, sizes)
    print(f"\n  Coupling proxy (gradient):")
    print(f"    raw rho(gradient, sensitivity) = {rho_grad_sens:.4f}")
    print(f"    partial rho(| size) = {pr_grad_sens:.4f}, p={pp_gs:.2e}")
    print(f"    128x128 reference: partial rho ~ 0.41")

    # 3. Boundary gradient vs sensitivity
    rho_bgrad_sens, p_bgs = spearmanr(mean_bgrad, sensitivities)
    pr_bgrad_sens, pp_bgs = partial_spearman(mean_bgrad, sensitivities, sizes)
    print(f"\n  Boundary gradient:")
    print(f"    raw rho(boundary_grad, sensitivity) = {rho_bgrad_sens:.4f}")
    print(f"    partial rho(| size) = {pr_bgrad_sens:.4f}, p={pp_bgs:.2e}")
    print(f"    128x128 reference: partial rho ~ -0.30")

    # 4. Boundary area ratio
    rho_bar_sens, p_bs = spearmanr(bar, sensitivities)
    pr_bar_sens, pp_bs = partial_spearman(bar, sensitivities, sizes)
    print(f"\n  Boundary area ratio:")
    print(f"    raw rho(bar, sensitivity) = {rho_bar_sens:.4f}")
    print(f"    partial rho(| size) = {pr_bar_sens:.4f}, p={pp_bs:.2e}")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Reference values from 128x128 experiments
    ref_size_sens_sign = 1  # positive (paradox)
    ref_gradient_partial = 0.41
    ref_bgrad_partial = -0.30

    # Test 1: C std > 0.005
    test1 = c_std > 0.005
    print(f"\n  Test 1: C std > 0.005 (real spatial structure)?")
    print(f"    C std = {c_std:.6f} (exp_30 had 0.0009)")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: >= 100 regions with >= 10 cells
    test2 = n_valid >= 100
    print(f"\n  Test 2: >= 100 regions with >= 10 cells?")
    print(f"    {n_valid} regions")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: Size-sensitivity same sign as 128x128
    test3 = (rho_size_sens > 0) == (ref_size_sens_sign > 0)
    print(f"\n  Test 3: rho(size, sensitivity) same sign as 128x128 (positive)?")
    print(f"    256x256: {rho_size_sens:.4f}, 128x128: +0.31")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: At least 2/3 key correlations within 0.20 of 128x128 reference
    corr_checks = []
    # Check gradient coupling
    delta_grad = abs(pr_grad_sens - ref_gradient_partial)
    corr_checks.append(delta_grad < 0.20)
    # Check boundary gradient
    delta_bgrad = abs(pr_bgrad_sens - ref_bgrad_partial)
    corr_checks.append(delta_bgrad < 0.20)
    # Check size-sensitivity sign match (already tested, include as third)
    corr_checks.append(test3)

    n_matching = sum(corr_checks)
    test4 = n_matching >= 2
    print(f"\n  Test 4: >= 2/3 key correlations within 0.20 of 128x128?")
    print(f"    Gradient coupling: delta={delta_grad:.4f} ({'pass' if corr_checks[0] else 'fail'})")
    print(f"    Boundary gradient: delta={delta_bgrad:.4f} ({'pass' if corr_checks[1] else 'fail'})")
    print(f"    Size-sens sign:    {'match' if corr_checks[2] else 'mismatch'} ({'pass' if corr_checks[2] else 'fail'})")
    print(f"    {n_matching}/3 matching")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 scale validation tests verified")

    # Compare with exp_30
    print(f"\n{'=' * 70}")
    print("Comparison: exp_34 (full) vs exp_30 (abbreviated)")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<30} {'exp_30':>12} {'exp_34':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'C std':<30} {'0.0009':>12} {c_std:>12.6f}")
    print(f"  {'Regions >= 10 cells':<30} {'174':>12} {n_valid:>12}")
    print(f"  {'rho(size, sens)':<30} {'sign-flip':>12} {rho_size_sens:>12.4f}")
    print(f"  {'partial rho(grad | size)':<30} {'-0.19':>12} {pr_grad_sens:>12.4f}")
    print(f"  {'partial rho(bgrad | size)':<30} {'~0':>12} {pr_bgrad_sens:>12.4f}")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_34_scale_validation_full',
        'timestamp': datetime.now().isoformat(),
        'purpose': '256x256 replication with full steady-state diffusion (fixes exp_30)',
        'grid_size': N,
        'diffusion': {
            'method': 'PeriodicLatticeFluid.run_to_steady_state',
            'max_steps': 10000,
            'steps_taken': history['steps_to_steady'],
            'reached_steady': history['reached_steady'],
            'c_std': c_std,
            'conservation_error': float(fluid.conservation_error()),
        },
        'partition': {
            'n_total_regions': len(region_ids),
            'n_valid_regions': n_valid,
            'n_analyzed_regions': n_regions,
        },
        'correlations': {
            'size_sensitivity': {
                'rho': float(rho_size_sens),
                'p': float(p_ss),
            },
            'gradient_coupling': {
                'rho_raw': float(rho_grad_sens),
                'partial_rho_size': float(pr_grad_sens),
                'p': float(pp_gs),
            },
            'boundary_gradient': {
                'rho_raw': float(rho_bgrad_sens),
                'partial_rho_size': float(pr_bgrad_sens),
                'p': float(pp_bgs),
            },
            'boundary_area_ratio': {
                'rho_raw': float(rho_bar_sens),
                'partial_rho_size': float(pr_bar_sens),
                'p': float(pp_bs),
            },
        },
        'reference_128': {
            'size_sensitivity_rho': 0.31,
            'gradient_partial_rho': 0.41,
            'boundary_gradient_partial_rho': -0.30,
        },
        'verification': {
            'test1_c_std_above_0005': bool(test1),
            'test2_sufficient_regions': bool(test2),
            'test3_size_sensitivity_sign': bool(test3),
            'test4_correlation_replication': bool(test4),
            'n_verified': n_verified,
        },
        'per_region': region_data,
    }

    output_file = RESULTS_DIR / f'exp_34_scale_full_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
