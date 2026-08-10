"""
exp_29_sensitivity_mediation.py -- Confluent Identity Phase 20

PURPOSE:
    Resolve the sensitivity puzzle: bigger regions are MORE sensitive (exp_23),
    opposite of CLT. Hypothesis: large irregular watershed regions have
    disproportionately high boundary_area_ratio. Sensitivity is driven by
    exposed surface, not volume. After controlling for boundary_area_ratio,
    the size-sensitivity relationship should vanish (mediation).

METHODS:
    For each level-0 region (>=10 cells):
    1. Compute: size, boundary_area_ratio, compactness, Gaussian sensitivity
    2. Mediation cascade:
       - rho(size, sensitivity) — raw positive (from exp_23)
       - rho(boundary_area_ratio, sensitivity) — proposed mediator
       - partial_rho(size, sensitivity | boundary_area_ratio) — should drop to ~0
       - partial_rho(boundary_area_ratio, sensitivity | size) — direct effect
    3. Bootstrap mediation test (10,000 resamples)
    4. Sub-partition check: halve 5 large regions, verify halves have higher
       boundary_area_ratio AND higher sensitivity

VERIFICATION:
    - |rho(boundary_area_ratio, sensitivity)| > |rho(size, sensitivity)|
    - partial_rho(size, sensitivity | boundary_area_ratio) < 0.10 (full mediation)
    - partial_rho(boundary_area_ratio, sensitivity | size) > 0.25, p < 0.05
    - Sub-partition: >= 3/5 halved regions show both higher bar AND higher sensitivity

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
from exp_08_gradient_coupling import compute_gradient_field, compute_fiedler_field
from exp_14_partial_correlation import partial_spearman
from exp_21_h1_revision_powered import kmeans_subpartition
from exp_27_boundary_geometry import compute_boundary_metrics


def compute_gaussian_sensitivity(L, state_region, seed):
    """
    Sensitivity to Gaussian noise (exp_16 pattern).
    Returns sensitivity = |delta_coeffs| / |coeffs_base|.
    """
    I_base = compute_spectral_identity(L, state_region)
    coeffs_base = np.array(I_base['state_coefficients'])
    coeff_norm = float(np.linalg.norm(coeffs_base))
    if coeff_norm < 1e-15:
        return None

    rng = np.random.RandomState(seed)
    noise = rng.randn(len(state_region)) * 0.1 * np.mean(state_region)
    state_noisy = state_region + noise
    I_noisy = compute_spectral_identity(L, state_noisy)
    coeffs_noisy = np.array(I_noisy['state_coefficients'])

    min_len = min(len(coeffs_base), len(coeffs_noisy))
    delta = float(np.linalg.norm(coeffs_noisy[:min_len] - coeffs_base[:min_len]))
    return delta / (coeff_norm + 1e-15)


def bootstrap_mediation(x, m, y, n_boot=10000, seed=42):
    """
    Bootstrap test of mediation: does m mediate x -> y?
    Returns: indirect_effect (mean), ci_low, ci_high, p_value.

    Uses rank-based approach consistent with Spearman correlation.
    Indirect effect = rho(x,m) * rho(m,y|x) via rank regression.
    """
    rng = np.random.RandomState(seed)
    n = len(x)

    indirect_effects = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        x_b, m_b, y_b = x[idx], m[idx], y[idx]

        rho_xm, _ = spearmanr(x_b, m_b)
        pr_my_x, _ = partial_spearman(m_b, y_b, x_b)
        indirect_effects.append(rho_xm * pr_my_x)

    indirect_effects = np.array(indirect_effects)
    mean_indirect = float(np.mean(indirect_effects))
    ci_low = float(np.percentile(indirect_effects, 2.5))
    ci_high = float(np.percentile(indirect_effects, 97.5))

    # p-value: proportion of bootstrap samples where indirect effect <= 0
    p_value = float(np.mean(indirect_effects <= 0))

    return mean_indirect, ci_low, ci_high, p_value


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 20, Experiment 29")
    print("Sensitivity Mediation: Does Boundary Surface Explain Size Effect?")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency and gradient field...")
    adjacency = build_lattice_adjacency(C)
    grad_mag = compute_gradient_field(C)
    grad_flat = grad_mag.ravel()

    labels0 = labels_by_level[0]
    region_ids = sorted(np.unique(labels0).tolist())

    # =====================================================================
    # Collect per-region data
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Per-Region: Size, Boundary Metrics, Sensitivity")
    print(f"{'=' * 70}")

    region_data = []

    for rid in region_ids:
        indices = get_region_indices(labels_by_level, 0, rid)
        n_cells = len(indices)
        if n_cells < 10:
            continue

        # Boundary metrics (no Fiedler needed for mediation)
        bm = compute_boundary_metrics(indices, N, state_flat, grad_flat, None)

        # Sensitivity
        L, _ = graph_laplacian_subgraph(adjacency, indices)
        state_region = state_flat[indices]
        sens = compute_gaussian_sensitivity(L, state_region, seed=42 + rid)
        if sens is None:
            continue

        region_data.append({
            'region_id': int(rid),
            'n_cells': n_cells,
            'boundary_area_ratio': bm['boundary_area_ratio'],
            'compactness': bm['compactness'],
            'perimeter': bm['perimeter'],
            'sensitivity': sens,
        })

    n_regions = len(region_data)
    print(f"  Analyzed {n_regions} regions")

    sizes = np.array([r['n_cells'] for r in region_data], dtype=float)
    bar = np.array([r['boundary_area_ratio'] for r in region_data])
    compactness = np.array([r['compactness'] for r in region_data])
    sensitivities = np.array([r['sensitivity'] for r in region_data])

    # =====================================================================
    # Mediation Cascade
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Mediation Cascade: size -> boundary_area_ratio -> sensitivity")
    print(f"{'=' * 70}")

    rho_size_sens, p_ss = spearmanr(sizes, sensitivities)
    print(f"\n  rho(size, sensitivity) = {rho_size_sens:.4f}  p={p_ss:.2e}")

    rho_bar_sens, p_bs = spearmanr(bar, sensitivities)
    print(f"  rho(boundary_area_ratio, sensitivity) = {rho_bar_sens:.4f}  p={p_bs:.2e}")

    rho_size_bar, p_sb = spearmanr(sizes, bar)
    print(f"  rho(size, boundary_area_ratio) = {rho_size_bar:.4f}  p={p_sb:.2e}")

    pr_size_sens_bar, pp_ssb = partial_spearman(sizes, sensitivities, bar)
    print(f"\n  partial_rho(size, sens | bar) = {pr_size_sens_bar:.4f}  p={pp_ssb:.2e}")

    pr_bar_sens_size, pp_bss = partial_spearman(bar, sensitivities, sizes)
    print(f"  partial_rho(bar, sens | size) = {pr_bar_sens_size:.4f}  p={pp_bss:.2e}")

    # Also check compactness
    rho_comp_sens, p_cs = spearmanr(compactness, sensitivities)
    pr_comp_sens_size, pp_css = partial_spearman(compactness, sensitivities, sizes)
    print(f"\n  rho(compactness, sensitivity) = {rho_comp_sens:.4f}  p={p_cs:.2e}")
    print(f"  partial_rho(compactness, sens | size) = {pr_comp_sens_size:.4f}  p={pp_css:.2e}")

    # =====================================================================
    # Bootstrap mediation test
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Bootstrap Mediation Test (10,000 resamples)")
    print(f"{'=' * 70}")

    indirect, ci_low, ci_high, p_med = bootstrap_mediation(
        sizes, bar, sensitivities, n_boot=10000, seed=42
    )
    print(f"  Indirect effect (size -> bar -> sens): {indirect:.4f}")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  p-value (indirect <= 0): {p_med:.4f}")

    # =====================================================================
    # Sub-partition check
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Sub-Partition Validation")
    print(f"{'=' * 70}")

    # Select 5 largest regions
    sorted_regions = sorted(region_data, key=lambda r: r['n_cells'], reverse=True)
    test_regions = sorted_regions[:5]
    subpart_results = []

    for rd in test_regions:
        rid = rd['region_id']
        indices = get_region_indices(labels_by_level, 0, rid)
        n_cells = len(indices)

        # Split into 2 sub-regions
        halves = kmeans_subpartition(indices, N, k=2, seed=42 + rid)
        if halves is None or len(halves) < 2:
            continue

        half_bars = []
        half_sens = []
        for half_indices in halves:
            if len(half_indices) < 10:
                continue
            bm_h = compute_boundary_metrics(half_indices, N, state_flat, grad_flat, None)
            L_h, _ = graph_laplacian_subgraph(adjacency, half_indices)
            state_h = state_flat[half_indices]
            sens_h = compute_gaussian_sensitivity(L_h, state_h, seed=42 + rid)
            if sens_h is not None:
                half_bars.append(bm_h['boundary_area_ratio'])
                half_sens.append(sens_h)

        if len(half_bars) < 2:
            continue

        mean_half_bar = float(np.mean(half_bars))
        mean_half_sens = float(np.mean(half_sens))
        parent_bar = rd['boundary_area_ratio']
        parent_sens = rd['sensitivity']

        higher_bar = mean_half_bar > parent_bar
        higher_sens = mean_half_sens > parent_sens
        both = higher_bar and higher_sens

        subpart_results.append({
            'region_id': rid,
            'n_cells': n_cells,
            'parent_bar': parent_bar,
            'half_mean_bar': mean_half_bar,
            'parent_sens': parent_sens,
            'half_mean_sens': mean_half_sens,
            'higher_bar': higher_bar,
            'higher_sens': higher_sens,
            'both_higher': both,
        })

        status = 'BOTH HIGHER' if both else 'partial'
        print(f"  Region {rid} ({n_cells} cells):")
        print(f"    bar: {parent_bar:.4f} -> {mean_half_bar:.4f} "
              f"({'UP' if higher_bar else 'down'})")
        print(f"    sens: {parent_sens:.4f} -> {mean_half_sens:.4f} "
              f"({'UP' if higher_sens else 'down'})")
        print(f"    [{status}]")

    n_both = sum(1 for r in subpart_results if r['both_higher'])
    n_tested = len(subpart_results)
    print(f"\n  {n_both}/{n_tested} regions show both higher bar AND higher sensitivity")

    # =====================================================================
    # Verification
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Test 1: bar is stronger sensitivity predictor than size
    test1 = abs(rho_bar_sens) > abs(rho_size_sens)
    print(f"\n  Test 1: |rho(bar, sens)| > |rho(size, sens)|?")
    print(f"    |rho(bar,sens)| = {abs(rho_bar_sens):.4f}, "
          f"|rho(size,sens)| = {abs(rho_size_sens):.4f}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: Full mediation — size effect vanishes after controlling for bar
    test2 = abs(pr_size_sens_bar) < 0.10
    print(f"\n  Test 2: partial_rho(size, sens | bar) < 0.10 (full mediation)?")
    print(f"    partial_rho = {pr_size_sens_bar:.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    # Test 3: bar has direct effect after size control
    test3 = pr_bar_sens_size > 0.25 and pp_bss < 0.05
    print(f"\n  Test 3: partial_rho(bar, sens | size) > 0.25, p < 0.05?")
    print(f"    partial_rho = {pr_bar_sens_size:.4f}, p = {pp_bss:.2e}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: Sub-partition validation
    test4 = n_both >= 3 and n_tested >= 5
    print(f"\n  Test 4: >= 3/5 halved regions show both higher bar AND sens?")
    print(f"    {n_both}/{n_tested}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 sensitivity mediation tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_29_sensitivity_mediation',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Does boundary_area_ratio mediate size -> sensitivity?',
        'n_regions': n_regions,
        'mediation_cascade': {
            'rho_size_sens': float(rho_size_sens),
            'rho_bar_sens': float(rho_bar_sens),
            'rho_size_bar': float(rho_size_bar),
            'partial_size_sens_bar': float(pr_size_sens_bar),
            'partial_bar_sens_size': float(pr_bar_sens_size),
            'p_partial_bar_sens_size': float(pp_bss),
        },
        'bootstrap': {
            'indirect_effect': indirect,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p_med,
        },
        'compactness': {
            'rho_comp_sens': float(rho_comp_sens),
            'partial_comp_sens_size': float(pr_comp_sens_size),
        },
        'subpartition': subpart_results,
        'verification': {
            'test1_bar_stronger_than_size': bool(test1),
            'test2_full_mediation': bool(test2),
            'test3_bar_direct_effect': bool(test3),
            'test4_subpartition_validation': bool(test4),
            'n_verified': n_verified,
        },
        'per_region': region_data,
    }

    output_file = RESULTS_DIR / f'exp_29_sensitivity_mediation_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
