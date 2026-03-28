"""
exp_30_scale_validation.py -- Confluent Identity Phase 21

PURPOSE:
    Replicate core findings on a 256x256 lattice (~4x statistical power).
    If boundary geometry findings are real physics, they must survive scale-up.
    If they're artifacts of the 128x128 lattice, they'll vanish.

METHODS:
    1. Generate 256x256 field using PeriodicLatticeFluid(N=256)
    2. Watershed partition (exp_02 pattern)
    3. Replicate 4 key measurements:
       a. Coupling-contribution (gradient-weighted, partial_rho | size)
       b. Boundary_area_ratio vs sensitivity
       c. Compactness independence from size
       d. Boundary gradient vs sensitivity
    4. Compare 128 vs 256 effect sizes

    CRITICAL: Cannot use build_lattice_adjacency for 256x256 (65536^2 dense matrix).
    Uses compute_subgraph_laplacian_from_field per region (O(|region|) each).

VERIFICATION:
    - >= 150 level-0 regions with >= 10 cells
    - Coupling partial_rho(gradient | size) > 0.30 (replicates exp_22)
    - boundary_area_ratio is stronger sensitivity predictor than size
    - Effect size stability: key correlations within 0.15 of 128x128 values

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
from exp_08_gradient_coupling import compute_gradient_field
from exp_14_partial_correlation import partial_spearman
from exp_27_boundary_geometry import compute_boundary_metrics


# =====================================================================
# Inline 256x256 field generation + watershed (self-contained)
# =====================================================================

def generate_256_field(seed=42):
    """
    Generate a 256x256 PAC-conservative field.
    Same physics as exp_01 PeriodicLatticeFluid but parameterized for N=256.
    Runs abbreviated diffusion (fewer steps — structure matters, not exact steady state).
    """
    N = 256
    rng = np.random.default_rng(seed)
    total_value = 100.0
    gravity = 0.005

    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)

    C_raw = np.ones((N, N)) * 0.5
    for freq in [2, 3, 5, 8, 13]:  # Fibonacci frequencies
        phase_x = rng.random() * 2 * np.pi
        phase_y = rng.random() * 2 * np.pi
        amplitude = 0.3 / freq
        C_raw += amplitude * np.sin(2 * np.pi * freq * X / N + phase_x)
        C_raw += amplitude * np.cos(2 * np.pi * freq * Y / N + phase_y)

    C_raw = np.maximum(C_raw, 0.1)
    C_raw *= total_value / C_raw.sum()

    # Stones
    stone_mask = np.zeros((N, N), dtype=bool)
    for _ in range(20):  # large stones
        cx, cy = rng.integers(10, N - 10, size=2)
        r = rng.integers(5, 9)
        mask = (X - cx)**2 + (Y - cy)**2 < r**2
        stone_mask |= mask
        C_raw[mask] *= 3.0
    for _ in range(60):  # small stones
        cx, cy = rng.integers(5, N - 5, size=2)
        r = rng.integers(1, 4)
        mask = (X - cx)**2 + (Y - cy)**2 < r**2
        stone_mask |= mask
        C_raw[mask] *= 2.0

    C_raw *= total_value / C_raw.sum()

    alpha = np.where(stone_mask, 0.1, 0.7)
    alpha += 0.05 * rng.random((N, N))
    P = alpha * C_raw
    A = (1 - alpha) * C_raw

    stone_P = P.copy()
    stone_A = A.copy()
    fluid_mask = ~stone_mask
    n_fluid = fluid_mask.sum()

    # Abbreviated diffusion: 1500 steps (enough for structure, not full steady state)
    dt, viscosity, sec_threshold = 0.005, 0.05, 0.1

    for step in range(1500):
        # Diffusion
        for field in [P, A]:
            lap = (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
                   np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)
            field += dt * viscosity * lap

        # Gravity
        flow_down = gravity * P * fluid_mask.astype(float)
        P -= flow_down
        P += np.roll(flow_down, 1, axis=0)

        # Stone restoration
        P_delta = P[stone_mask].sum() - stone_P[stone_mask].sum()
        A_delta = A[stone_mask].sum() - stone_A[stone_mask].sum()
        P[stone_mask] = stone_P[stone_mask]
        A[stone_mask] = stone_A[stone_mask]
        if abs(P_delta) > 1e-15 and n_fluid > 0:
            P[fluid_mask] += P_delta / n_fluid
        if abs(A_delta) > 1e-15 and n_fluid > 0:
            A[fluid_mask] += A_delta / n_fluid

        P = np.maximum(P, 0)
        A = np.maximum(A, 0)

        if step % 500 == 0:
            print(f"    Step {step}: C std={float((P+A).std()):.6f}")

    C = P + A
    print(f"  256x256 field generated: C std={C.std():.6f}, "
          f"stones={stone_mask.sum()} cells")
    return P, A, C, stone_mask


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
    print("Confluent Identity -- Phase 21, Experiment 30")
    print("256x256 Scale Validation")
    print("=" * 70)

    # =====================================================================
    # Generate 256x256 field
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Generating 256x256 PAC-Conservative Field")
    print(f"{'=' * 70}")

    P, A, C, stone_mask = generate_256_field(seed=42)
    N = C.shape[0]
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

    # Filter to regions >= 10 cells
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

        # Spectral identity via subgraph Laplacian (no full adjacency needed)
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

        # Gradient-weighted coupling proxy: mean |grad C| normalized
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
    compactness = np.array([r['compactness'] for r in region_data])
    mean_bgrad = np.array([r['mean_boundary_gradient'] for r in region_data])
    mean_grad = np.array([r['mean_gradient'] for r in region_data])

    # =====================================================================
    # Key correlations
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Key Correlations (256x256)")
    print(f"{'=' * 70}")

    # 1. Gradient coupling proxy
    rho_grad_sens, p_gs = spearmanr(mean_grad, sensitivities)
    pr_grad_sens_size, pp_gss = partial_spearman(mean_grad, sensitivities, sizes)
    print(f"\n  Coupling proxy (gradient):")
    print(f"    raw rho(gradient, sensitivity) = {rho_grad_sens:.4f}")
    print(f"    partial rho(| size) = {pr_grad_sens_size:.4f}, p={pp_gss:.2e}")

    # 2. Boundary_area_ratio vs sensitivity
    rho_bar_sens, p_bs = spearmanr(bar, sensitivities)
    rho_size_sens, p_ss = spearmanr(sizes, sensitivities)
    pr_bar_sens_size, pp_bss = partial_spearman(bar, sensitivities, sizes)
    print(f"\n  Sensitivity predictors:")
    print(f"    rho(size, sensitivity) = {rho_size_sens:.4f}")
    print(f"    rho(bar, sensitivity) = {rho_bar_sens:.4f}")
    print(f"    partial rho(bar, sens | size) = {pr_bar_sens_size:.4f}, p={pp_bss:.2e}")

    # 3. Compactness vs size
    rho_comp_size, _ = spearmanr(compactness, sizes)
    print(f"\n  Compactness:")
    print(f"    rho(compactness, size) = {rho_comp_size:.4f}")

    # 4. Boundary gradient vs sensitivity
    rho_bgrad_sens, p_bgs = spearmanr(mean_bgrad, sensitivities)
    pr_bgrad_sens_size, pp_bgss = partial_spearman(mean_bgrad, sensitivities, sizes)
    print(f"\n  Boundary gradient:")
    print(f"    rho(mean_boundary_gradient, sensitivity) = {rho_bgrad_sens:.4f}")
    print(f"    partial rho(| size) = {pr_bgrad_sens_size:.4f}, p={pp_bgss:.2e}")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Test 1: >= 150 level-0 regions with >= 10 cells
    test1 = n_valid >= 150
    print(f"\n  Test 1: >= 150 regions with >= 10 cells?")
    print(f"    {n_valid} regions")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Coupling partial_rho > 0.30
    test2 = pr_grad_sens_size > 0.30
    print(f"\n  Test 2: Coupling partial_rho(gradient | size) > 0.30?")
    print(f"    partial_rho = {pr_grad_sens_size:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: bar is stronger sensitivity predictor than size
    test3 = abs(rho_bar_sens) > abs(rho_size_sens)
    print(f"\n  Test 3: |rho(bar, sens)| > |rho(size, sens)|?")
    print(f"    |bar| = {abs(rho_bar_sens):.4f}, |size| = {abs(rho_size_sens):.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: Effect size stability — partial_rho within 0.15 of expected 128x128 values
    # Reference: exp_22 gradient partial_rho ~ 0.41
    ref_gradient_partial = 0.41
    delta_effect = abs(pr_grad_sens_size - ref_gradient_partial)
    test4 = delta_effect < 0.15
    print(f"\n  Test 4: Effect size stability (gradient partial rho within 0.15 of 0.41)?")
    print(f"    256x256: {pr_grad_sens_size:.4f}, 128x128 ref: {ref_gradient_partial}")
    print(f"    delta = {delta_effect:.4f}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 scale validation tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_30_scale_validation',
        'timestamp': datetime.now().isoformat(),
        'purpose': '256x256 replication of boundary geometry findings',
        'grid_size': N,
        'n_total_regions': len(region_ids),
        'n_valid_regions': n_valid,
        'n_analyzed_regions': n_regions,
        'correlations': {
            'gradient_coupling': {
                'rho_raw': float(rho_grad_sens),
                'partial_rho_size': float(pr_grad_sens_size),
                'p': float(pp_gss),
            },
            'bar_sensitivity': {
                'rho_raw': float(rho_bar_sens),
                'partial_rho_size': float(pr_bar_sens_size),
                'p': float(pp_bss),
            },
            'size_sensitivity': {
                'rho_raw': float(rho_size_sens),
            },
            'compactness_size': {
                'rho': float(rho_comp_size),
            },
            'boundary_gradient_sensitivity': {
                'rho_raw': float(rho_bgrad_sens),
                'partial_rho_size': float(pr_bgrad_sens_size),
                'p': float(pp_bgss),
            },
        },
        'verification': {
            'test1_sufficient_regions': bool(test1),
            'test2_coupling_replicates': bool(test2),
            'test3_bar_stronger_than_size': bool(test3),
            'test4_effect_size_stability': bool(test4),
            'n_verified': n_verified,
        },
        'per_region': region_data,
    }

    output_file = RESULTS_DIR / f'exp_30_scale_validation_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
