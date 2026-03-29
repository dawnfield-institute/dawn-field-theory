"""
exp_37_multiseed_phi_validation.py -- Confluent Identity Phase 28

PURPOSE:
    Validate whether the coupling ceiling decomposes into phi-constants
    across multiple random seeds. exp_36 found on a SINGLE realization:

        partial_rho  ≈ γ          = 0.5772  (0.33%)
        raw_rho      ≈ γ - 1/φ⁴  = 0.4313  (0.17%)
        size_confound ≈ 1/φ⁴     = 0.1459  (0.83%)
        raw_align    ≈ ln(φ)·φ   = 0.7786  (0.11%)
        eig_align    ≈ 1/φ² + γ/φ² = 0.6024  (0.19%)

    If these hold across seeds, the coupling ceiling IS a phi-constant
    (specifically γ, the additive divergence cost from ADE Level 1),
    and the "0.42" we measured was γ minus the tetration termination
    penalty 1/φ⁴.

    If they DON'T hold, the exp_36 matches were overfitting to seed=42.

MATHEMATICAL FRAMEWORK:
    From exp_30o: ξ = γ + ln(φ) is the PAC reconciliation constant.
    γ = additive divergence cost (L1 in ADE)
    ln(φ) = multiplicative convergence rate (L2 in ADE)
    1/φ⁴ = Level 4 (tetration) termination correction

    The coupling ceiling should decompose as:
        coupling_full = γ                  (size-independent, the physics)
        coupling_raw  = γ - 1/φ⁴          (size-confounded measurement)
        confounding   = 1/φ⁴              (the tetration penalty)

VERIFICATION (4 tests, predict 2/4):
    1. Mean partial_rho across seeds within 2σ of γ              (PREDICT PASS)
    2. Mean raw_rho across seeds within 2σ of γ - 1/φ⁴           (PREDICT PASS)
    3. Mean size_confound across seeds within 2σ of 1/φ⁴          (PREDICT FAIL)
       — confounding is seed-dependent, may not be stable
    4. Cross-seed CV < 0.15 for all measures (low variance)       (PREDICT FAIL)
       — hierarchy topology varies by seed, expect higher variance

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_01_lattice_fluid_baseline import PeriodicLatticeFluid
from _shared import (
    RESULTS_DIR, K_MODES, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    compute_subgraph_laplacian_from_field,
)
from exp_02_hierarchical_partition import (
    watershed_from_minima, compute_boundary_gradients, merge_regions,
    build_hierarchy,
)
from exp_08_gradient_coupling import (
    compute_coupling_weights_weighted, compute_natural_weights,
    compute_gradient_field,
)
from exp_14_partial_correlation import partial_spearman


# ── Constants ────────────────────────────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
GAMMA = 0.5772156649015329
XI_PAC = GAMMA + LN_PHI
INV_PHI4 = 1 / PHI**4
INV_PHI2 = 1 / PHI**2

# Theoretical predictions
PRED_PARTIAL_RHO = GAMMA                    # 0.5772
PRED_RAW_RHO = GAMMA - INV_PHI4             # 0.4313
PRED_CONFOUND = INV_PHI4                    # 0.1459
PRED_RAW_ALIGN = LN_PHI * PHI              # 0.7786
PRED_EIG_ALIGN = INV_PHI2 + GAMMA * INV_PHI2  # 0.6024

N_SEEDS = 20


# ── Per-seed measurement ─────────────────────────────────────────────

def measure_seed(seed, verbose=True):
    """
    Run full pipeline for one seed: fluid → hierarchy → coupling → measures.
    Returns dict of all key values, or None if seed fails.
    """
    if verbose:
        print(f"\n  === Seed {seed} ===")

    try:
        # 1. Generate steady-state fluid
        fluid = PeriodicLatticeFluid(N=128, total_value=100.0, seed=seed)
        history = fluid.run_to_steady_state(
            max_steps=5000, dt=0.005, viscosity=0.05,
            sec_threshold=0.1, tol=1e-6, stable_count=10
        )

        P = fluid.P.copy()
        A = fluid.A.copy()
        C = fluid.C.copy()
        stone_mask = fluid.stone_mask.copy()
        N = C.shape[0]

        # Check conservation
        cons_err = fluid.conservation_error()
        if verbose:
            print(f"    Conservation error: {cons_err:.2e}")
            n_steps = len(history.get('max_change', []))
            print(f"    C std: {C.std():.6f}, steps: {n_steps}")

        if C.std() < 1e-6:
            if verbose:
                print("    SKIP: C field is flat (no structure)")
            return None

        # 2. Build hierarchy (replicate exp_02 pipeline)
        C_flat = C.ravel()
        state_flat = C_flat.copy()
        adjacency = build_lattice_adjacency(C)
        grad_C = compute_gradient_field(C)
        grad_flat = grad_C.ravel()

        # Watershed at level 0
        labels_0, n_seeds_ws = watershed_from_minima(C, sigma=0.5, min_filter_size=3)
        if n_seeds_ws < 3:
            if verbose:
                print(f"    SKIP: only {n_seeds_ws} watershed seeds")
            return None

        # Adaptive merge thresholds
        boundaries_0 = compute_boundary_gradients(labels_0, C)
        if boundaries_0:
            grad_values = sorted(boundaries_0.values())
            thresholds = [np.percentile(grad_values, p) for p in [25, 50, 75, 90]]
        else:
            thresholds = [0.001, 0.003, 0.01, 0.03]

        labels_by_level = [labels_0]
        current_labels = labels_0
        for thresh in thresholds:
            new_labels, _ = merge_regions(current_labels, C, thresh)
            n_regions = len(np.unique(new_labels))
            labels_by_level.append(new_labels)
            current_labels = new_labels
            if n_regions <= 3:
                break

        hierarchy = build_hierarchy(labels_by_level)

        # 3. Iterate parents, collect coupling/natural/size/alignments
        all_coupling = []
        all_natural = []
        all_sizes = []
        all_raw_align = []
        all_eig_align = []

        for (level, pid), children_tuples in hierarchy.items():
            if len(children_tuples) < 2:
                continue

            labels = labels_by_level[level]
            parent_indices = np.where(labels.ravel() == pid)[0]
            if len(parent_indices) < 10:
                continue

            # Build children list from hierarchy
            # children_tuples is [(child_level, child_id), ...]
            child_level = children_tuples[0][0]
            child_labels = labels_by_level[child_level]

            children_list = []
            for _, cid in children_tuples:
                child_idx = np.where(child_labels.ravel() == cid)[0]
                child_in_parent = np.intersect1d(child_idx, parent_indices)
                if len(child_in_parent) >= 4:
                    children_list.append((int(cid), child_in_parent))

            if len(children_list) < 3:
                continue

            # Compute parent spectral identity
            L_parent, _ = graph_laplacian_subgraph(adjacency, parent_indices)
            state_parent = state_flat[parent_indices]
            identity = compute_spectral_identity(L_parent, state_parent)
            if 'eigenvectors' not in identity:
                continue
            eigvecs = identity['eigenvectors']

            # Coupling and natural weights
            coupling = compute_coupling_weights_weighted(
                adjacency, state_flat, parent_indices, children_list, grad_flat
            )
            natural, _ = compute_natural_weights(
                state_flat, parent_indices, children_list, eigvecs
            )

            for child_id, child_indices in children_list:
                if child_id not in coupling or child_id not in natural:
                    continue

                all_coupling.append(coupling[child_id])
                all_natural.append(natural[child_id])
                all_sizes.append(len(child_indices))

                # Raw field alignment
                grad_child = grad_flat[child_indices]
                state_child = state_flat[child_indices]
                ng = np.linalg.norm(grad_child)
                ns = np.linalg.norm(state_child)
                if ng > 1e-15 and ns > 1e-15:
                    all_raw_align.append(float(np.dot(grad_child, state_child) / (ng * ns)))

                # Eigenbasis-projected alignment
                parent_pos_map = {int(idx): pos for pos, idx in enumerate(parent_indices)}
                local_pos = np.array([parent_pos_map[int(g)] for g in child_indices
                                      if int(g) in parent_pos_map])
                if len(local_pos) >= 2:
                    sp = state_flat[parent_indices]
                    sp_c = sp - np.mean(sp)
                    gp = grad_flat[parent_indices]
                    gp_c = gp - np.mean(gp)
                    cs = sp_c[local_pos] @ eigvecs[local_pos, :]
                    cg = gp_c[local_pos] @ eigvecs[local_pos, :]
                    ncs = np.linalg.norm(cs)
                    ncg = np.linalg.norm(cg)
                    if ncs > 1e-15 and ncg > 1e-15:
                        all_eig_align.append(float(np.dot(cs, cg) / (ncs * ncg)))

        if len(all_coupling) < 10:
            if verbose:
                print(f"    SKIP: only {len(all_coupling)} children")
            return None

        coupling_arr = np.array(all_coupling)
        natural_arr = np.array(all_natural)
        size_arr = np.array(all_sizes, dtype=float)

        # 4. Compute measures
        rho_raw, _ = spearmanr(coupling_arr, natural_arr)
        rho_partial, p_partial = partial_spearman(coupling_arr, natural_arr, size_arr)
        confound = rho_partial - rho_raw

        raw_align = float(np.mean(all_raw_align)) if all_raw_align else np.nan
        eig_align = float(np.mean(all_eig_align)) if all_eig_align else np.nan

        result = {
            'seed': seed,
            'n_children': len(all_coupling),
            'n_steps': len(history.get('max_change', [])),
            'C_std': float(C.std()),
            'raw_rho': float(rho_raw),
            'partial_rho': float(rho_partial),
            'p_partial': float(p_partial),
            'size_confound': float(confound),
            'raw_align': float(raw_align),
            'eig_align': float(eig_align),
        }

        if verbose:
            print(f"    n_children={result['n_children']}, "
                  f"raw_rho={rho_raw:.4f}, partial_rho={rho_partial:.4f}, "
                  f"confound={confound:.4f}")

        return result

    except Exception as e:
        if verbose:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
        return None


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXP 37: MULTI-SEED PHI-CONSTANT VALIDATION")
    print("Phase 28 — Confluent Identity")
    print(f"Testing {N_SEEDS} seeds against ADE predictions")
    print("=" * 70)

    print(f"\n  THEORETICAL PREDICTIONS (from exp_30 ADE + exp_36):")
    print(f"    partial_rho  = gamma       = {PRED_PARTIAL_RHO:.6f}")
    print(f"    raw_rho      = gamma-1/phi^4 = {PRED_RAW_RHO:.6f}")
    print(f"    confounding  = 1/phi^4     = {PRED_CONFOUND:.6f}")
    print(f"    raw_align    = ln(phi)*phi = {PRED_RAW_ALIGN:.6f}")
    print(f"    eig_align    = (1+gamma)/phi^2 = {PRED_EIG_ALIGN:.6f}")

    # Run seeds
    seeds = list(range(N_SEEDS))
    results_list = []
    for seed in seeds:
        r = measure_seed(seed, verbose=True)
        if r is not None:
            results_list.append(r)

    n_valid = len(results_list)
    print(f"\n\n{'='*70}")
    print(f"RESULTS: {n_valid}/{N_SEEDS} seeds produced valid data")
    print("=" * 70)

    if n_valid < 5:
        print("  INSUFFICIENT DATA — cannot validate")
        return

    # Extract arrays
    raw_rhos = np.array([r['raw_rho'] for r in results_list])
    partial_rhos = np.array([r['partial_rho'] for r in results_list])
    confounds = np.array([r['size_confound'] for r in results_list])
    raw_aligns = np.array([r['raw_align'] for r in results_list if not np.isnan(r['raw_align'])])
    eig_aligns = np.array([r['eig_align'] for r in results_list if not np.isnan(r['eig_align'])])

    def report(name, data, prediction):
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(len(data))
        ci_lo = mean - 2 * se
        ci_hi = mean + 2 * se
        within = ci_lo <= prediction <= ci_hi
        delta = mean - prediction
        rel = abs(delta / prediction) * 100 if prediction != 0 else float('inf')
        cv = abs(std / mean) if mean != 0 else float('inf')

        print(f"\n  {name}:")
        print(f"    Mean:        {mean:.6f} ± {std:.6f}")
        print(f"    95% CI:      [{ci_lo:.6f}, {ci_hi:.6f}]")
        print(f"    Prediction:  {prediction:.6f}")
        print(f"    Delta:       {delta:+.6f} ({rel:.2f}%)")
        print(f"    CV:          {cv:.4f}")
        print(f"    In 95% CI:   {'YES' if within else 'NO'}")
        print(f"    Range:       [{np.min(data):.4f}, {np.max(data):.4f}]")

        return {
            'mean': float(mean),
            'std': float(std),
            'se': float(se),
            'ci_lo': float(ci_lo),
            'ci_hi': float(ci_hi),
            'prediction': float(prediction),
            'delta': float(delta),
            'relative_error_pct': float(rel),
            'cv': float(cv),
            'within_ci': bool(within),
            'n': len(data),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'values': [float(x) for x in data],
        }

    print("\n" + "-" * 60)
    print("PER-MEASURE ANALYSIS")
    print("-" * 60)

    stats = {}
    stats['partial_rho'] = report("partial_rho vs gamma", partial_rhos, PRED_PARTIAL_RHO)
    stats['raw_rho'] = report("raw_rho vs gamma-1/phi^4", raw_rhos, PRED_RAW_RHO)
    stats['confound'] = report("size_confound vs 1/phi^4", confounds, PRED_CONFOUND)
    if len(raw_aligns) >= 3:
        stats['raw_align'] = report("raw_align vs ln(phi)*phi", raw_aligns, PRED_RAW_ALIGN)
    if len(eig_aligns) >= 3:
        stats['eig_align'] = report("eig_align vs (1+gamma)/phi^2", eig_aligns, PRED_EIG_ALIGN)

    # ── Per-seed decomposition table ──────────────────────────────
    print("\n" + "-" * 60)
    print("PER-SEED VALUES")
    print("-" * 60)
    print(f"  {'Seed':>4s} {'n':>4s} {'raw_rho':>8s} {'p_rho':>8s} {'conf':>8s} "
          f"{'raw_al':>8s} {'eig_al':>8s}")
    print(f"  {'-'*48}")
    for r in results_list:
        print(f"  {r['seed']:4d} {r['n_children']:4d} {r['raw_rho']:8.4f} "
              f"{r['partial_rho']:8.4f} {r['size_confound']:8.4f} "
              f"{r['raw_align']:8.4f} {r['eig_align']:8.4f}")

    print(f"\n  PREDICTIONS:")
    print(f"  {'':4s} {'':4s} {PRED_RAW_RHO:8.4f} {PRED_PARTIAL_RHO:8.4f} "
          f"{PRED_CONFOUND:8.4f} {PRED_RAW_ALIGN:8.4f} {PRED_EIG_ALIGN:8.4f}")

    # ── Verification ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Test 1: partial_rho mean within 2-sigma of gamma
    test1_pass = stats['partial_rho']['within_ci']
    status1 = "VERIFIED" if test1_pass else "NOT VERIFIED"
    print(f"\n  Test 1: Mean partial_rho within 2-sigma of gamma = {PRED_PARTIAL_RHO:.4f}")
    print(f"    Mean = {stats['partial_rho']['mean']:.4f}, "
          f"CI = [{stats['partial_rho']['ci_lo']:.4f}, {stats['partial_rho']['ci_hi']:.4f}]")
    print(f"    -> {status1}")

    # Test 2: raw_rho mean within 2-sigma of gamma - 1/phi^4
    test2_pass = stats['raw_rho']['within_ci']
    status2 = "VERIFIED" if test2_pass else "NOT VERIFIED"
    print(f"\n  Test 2: Mean raw_rho within 2-sigma of gamma-1/phi^4 = {PRED_RAW_RHO:.4f}")
    print(f"    Mean = {stats['raw_rho']['mean']:.4f}, "
          f"CI = [{stats['raw_rho']['ci_lo']:.4f}, {stats['raw_rho']['ci_hi']:.4f}]")
    print(f"    -> {status2}")

    # Test 3: confound mean within 2-sigma of 1/phi^4
    test3_pass = stats['confound']['within_ci']
    status3 = "VERIFIED" if test3_pass else "NOT VERIFIED"
    print(f"\n  Test 3: Mean confound within 2-sigma of 1/phi^4 = {PRED_CONFOUND:.4f}")
    print(f"    Mean = {stats['confound']['mean']:.4f}, "
          f"CI = [{stats['confound']['ci_lo']:.4f}, {stats['confound']['ci_hi']:.4f}]")
    print(f"    -> {status3}")

    # Test 4: CV < 0.15 for partial_rho and raw_rho
    cv_partial = stats['partial_rho']['cv']
    cv_raw = stats['raw_rho']['cv']
    test4_pass = cv_partial < 0.15 and cv_raw < 0.15
    status4 = "VERIFIED" if test4_pass else "NOT VERIFIED"
    print(f"\n  Test 4: CV < 0.15 for stability")
    print(f"    CV(partial_rho) = {cv_partial:.4f}, CV(raw_rho) = {cv_raw:.4f}")
    print(f"    -> {status4}")

    n_verified = sum([test1_pass, test2_pass, test3_pass, test4_pass])
    print(f"\n  TOTAL: {n_verified}/4 verified")

    # ── Honest assessment ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    if test1_pass and test2_pass:
        print("  The gamma and gamma-1/phi^4 relationships HOLD across seeds.")
        print("  The coupling ceiling IS the Euler-Mascheroni constant.")
        print("  This is not an artifact of seed 42.")
    elif test1_pass:
        print("  partial_rho ~ gamma holds, but the decomposition is less clean.")
        print("  The ceiling is gamma but the size correction isn't exactly 1/phi^4.")
    else:
        print("  The phi-constant matches from exp_36 do NOT generalize.")
        print("  They were specific to seed 42. The ceiling is real but")
        print("  does not have a clean analytical form from ADE constants.")

    # ── Save ──────────────────────────────────────────────────────
    output = {
        'experiment': 'exp_37_multiseed_phi_validation',
        'phase': 28,
        'n_seeds': N_SEEDS,
        'n_valid': n_valid,
        'predictions': {
            'partial_rho': float(PRED_PARTIAL_RHO),
            'raw_rho': float(PRED_RAW_RHO),
            'confound': float(PRED_CONFOUND),
            'raw_align': float(PRED_RAW_ALIGN),
            'eig_align': float(PRED_EIG_ALIGN),
        },
        'stats': stats,
        'per_seed': results_list,
        'verification': {
            'test1_partial_rho_is_gamma': test1_pass,
            'test2_raw_rho_is_gamma_minus_phi4': test2_pass,
            'test3_confound_is_phi4': test3_pass,
            'test4_low_variance': test4_pass,
            'verified_count': n_verified,
        },
        'timestamp': datetime.now().isoformat(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'exp_37_multiseed_phi_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
