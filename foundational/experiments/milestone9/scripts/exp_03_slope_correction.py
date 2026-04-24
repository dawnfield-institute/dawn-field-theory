"""
Milestone 9 -- Exp 03: Slope Correction

PURPOSE: Analyze the 8.9% discrepancy between the free-fit slope
(B_FREE = 2.264) and the DFT-constrained slope (B_DFT = 1/ln(phi) = 2.078).
Tests four correction mechanisms: ghost heart (boundary-determined PAC trees),
ADE dimensional factors, leave-one-out instability, and Monte Carlo
sensitivity.

Block A: Cascade Dynamics

Tests:
  1. Ghost heart correction: boundary-determined trees reduce effective
     independent levels, inflating the slope
  2. ADE dimensional correction: spacetime dimensionality factors
  3. Leave-one-out stability: 3-point fit instability analysis
  4. Monte Carlo sensitivity: B_DFT falls within 95% CI of perturbed fits
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
M9_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M9_ROOT))
from core.infodynamics import *

_, RESULTS_DIR = setup_experiment(__file__)


def test1_ghost_heart_correction():
    """
    In exp_33d, PAC trees are boundary-determined: interior nodes carry
    zero independent information. For a cascade of depth d, the effective
    independent levels = number of leaf nodes, not total nodes.

    For a phi-cascade (not binary), the branching is unequal. Build a
    depth-6 PAC tree where each node splits into D (weight 1/phi) and
    S (weight 1/phi^2). Count "effective independent" nodes = nodes
    with weight > 1% of root. Compute correction = n_total / n_effective.
    Apply to slope: B_corrected = B_DFT * n_total / n_effective.

    PASS if |B_corrected - B_FREE| / B_FREE < 0.03 (within 3%).
    """
    print("\n" + "-" * 70)
    print("TEST 1: GHOST HEART CORRECTION")
    print("-" * 70)

    max_depth = 6
    threshold_frac = 0.01  # 1% of root weight

    # Build PAC tree: each node splits into D (1/phi) and S (1/phi^2)
    # Level 0: root with weight 1.0
    # Level k: each node from level k-1 spawns D and S children
    all_nodes = []  # (depth, weight)
    current_level = [(0, 1.0)]
    all_nodes.extend(current_level)

    for d in range(1, max_depth + 1):
        next_level = []
        for _, parent_weight in current_level:
            # Dominant child
            d_weight = parent_weight * INV_PHI
            next_level.append((d, d_weight))
            # Subordinate child
            s_weight = parent_weight * INV_PHI**2
            next_level.append((d, s_weight))
        all_nodes.extend(next_level)
        current_level = next_level

    n_total = len(all_nodes)
    weights = np.array([w for _, w in all_nodes])
    n_effective = int(np.sum(weights > threshold_frac))

    # Leaf nodes (deepest level only)
    leaf_weights = np.array([w for dep, w in all_nodes if dep == max_depth])
    n_leaves = len(leaf_weights)
    n_leaves_above = int(np.sum(leaf_weights > threshold_frac))

    # Correction factor
    if n_effective > 0:
        correction = n_total / n_effective
    else:
        correction = float('inf')
    B_corrected = B_DFT * correction

    print(f"\n  PAC tree (depth {max_depth}):")
    print(f"    Total nodes:     {n_total}")
    print(f"    Leaf nodes:      {n_leaves}")
    print(f"    Threshold:       {threshold_frac*100:.0f}% of root")
    print(f"    Effective nodes: {n_effective} (weight > {threshold_frac})")
    print(f"    Leaves above:    {n_leaves_above}")

    print(f"\n  Weight distribution by depth:")
    for d in range(max_depth + 1):
        level_weights = [w for dep, w in all_nodes if dep == d]
        n_above = sum(1 for w in level_weights if w > threshold_frac)
        max_w = max(level_weights)
        min_w = min(level_weights)
        print(f"    Depth {d}: {len(level_weights):3d} nodes, "
              f"range [{min_w:.4f}, {max_w:.4f}], "
              f"{n_above} above threshold")

    print(f"\n  Correction:")
    print(f"    n_total / n_effective = {n_total} / {n_effective} = {correction:.4f}")
    print(f"    B_DFT = {B_DFT:.4f}")
    print(f"    B_corrected = B_DFT * {correction:.4f} = {B_corrected:.4f}")
    print(f"    B_FREE = {B_FREE:.4f}")

    dev = abs(B_corrected - B_FREE) / B_FREE
    print(f"    |B_corrected - B_FREE| / B_FREE = {dev*100:.2f}%")

    passed = dev < 0.03
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: ghost heart correction "
          f"{'matches' if passed else 'does not match'} B_FREE "
          f"(dev = {dev*100:.2f}%)")

    return {
        'test': 'ghost_heart_correction',
        'max_depth': max_depth,
        'n_total': n_total,
        'n_effective': n_effective,
        'n_leaves': n_leaves,
        'n_leaves_above_threshold': n_leaves_above,
        'threshold_frac': threshold_frac,
        'correction_factor': float(correction),
        'B_DFT': float(B_DFT),
        'B_corrected': float(B_corrected),
        'B_FREE': float(B_FREE),
        'deviation_pct': float(dev * 100),
        'passed': bool(passed),
    }


def test2_ade_dimensional_correction():
    """
    The cascade operates in D=3+1 spacetime. Test correction factors
    from dimensional analysis:
      - 3/2     (spatial dims / (dims-1))
      - ln(4)/ln(3) (D+1 over D in log)
      - sqrt(3) (spatial dimension norm)
      - 4/3     (spacetime/spatial)
      - D/(D-1) = 3/2

    For each, compute B_DFT * factor and compare to B_FREE.

    PASS if at least one gives |result - B_FREE| / B_FREE < 0.03.
    """
    print("\n" + "-" * 70)
    print("TEST 2: ADE DIMENSIONAL CORRECTION")
    print("-" * 70)

    print(f"\n  Reference:")
    print(f"    B_DFT  = {B_DFT:.6f}")
    print(f"    B_FREE = {B_FREE:.6f}")
    print(f"    Ratio B_FREE/B_DFT = {B_FREE/B_DFT:.6f}")
    print(f"    Gap = {(B_FREE/B_DFT - 1)*100:.2f}%")

    factors = {
        '3/2 (spatial/spatial-1)':     3.0 / 2.0,
        'ln(4)/ln(3) (log dim)':       np.log(4) / np.log(3),
        'sqrt(3) (spatial norm)':      np.sqrt(3),
        '4/3 (spacetime/spatial)':     4.0 / 3.0,
        'D/(D-1) = 3/2':              3.0 / 2.0,
    }

    best_dev = float('inf')
    best_name = None
    any_within_3pct = False
    results_detail = {}

    print(f"\n  Dimensional correction factors:")
    for name, factor in factors.items():
        B_result = B_DFT * factor
        dev = abs(B_result - B_FREE) / B_FREE
        match = dev < 0.03
        marker = " <-- MATCH" if match else ""

        if match:
            any_within_3pct = True
        if dev < best_dev:
            best_dev = dev
            best_name = name

        print(f"    {name:30s}: factor = {factor:.6f}, "
              f"B = {B_result:.4f}, dev = {dev*100:.2f}%{marker}")

        results_detail[name] = {
            'factor': float(factor),
            'B_result': float(B_result),
            'deviation_pct': float(dev * 100),
            'within_3pct': match,
        }

    # What factor is needed?
    needed = B_FREE / B_DFT
    print(f"\n  Needed factor for exact match: {needed:.6f}")
    print(f"  Best factor: {best_name} (dev = {best_dev*100:.2f}%)")

    # Check if needed factor has a clean form
    print(f"\n  Needed factor analysis:")
    candidates = {
        'Xi':           XI_BALANCE,
        'sqrt(phi)':    np.sqrt(PHI),
        '1 + 1/PI':     1.0 + 1.0/PI,
        'phi^(1/5)':    PHI**(1.0/5),
        'ln(3)':        np.log(3),
    }
    for cname, cval in candidates.items():
        cdev = abs(needed - cval) / needed * 100
        print(f"    {needed:.6f} vs {cname} = {cval:.6f} (dev = {cdev:.2f}%)")

    passed = any_within_3pct
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: ADE dimensional correction "
          f"{'found' if passed else 'not found'} within 3%")

    return {
        'test': 'ade_dimensional_correction',
        'B_DFT': float(B_DFT),
        'B_FREE': float(B_FREE),
        'needed_factor': float(needed),
        'factors': results_detail,
        'best_factor': best_name,
        'best_deviation_pct': float(best_dev * 100),
        'any_within_3pct': any_within_3pct,
        'passed': bool(passed),
    }


def test3_leave_one_out_stability():
    """
    With only 3 data points, the free fit is underdetermined. Use
    leave-one-out cross-validation: fit slope with 2 of 3 points,
    predict the 3rd. Compute the prediction error for each left-out
    point.

    If errors are large relative to the residuals, the 3-point fit
    is unstable and the 8.9% discrepancy may be noise.

    PASS if at least one leave-one-out prediction error > 0.5.
    """
    print("\n" + "-" * 70)
    print("TEST 3: LEAVE-ONE-OUT STABILITY")
    print("-" * 70)

    # Extract data points
    names = list(N_DATA.keys())
    t_all = np.array([N_DATA[n]['t_lookback_gyr'] for n in names])
    n_all = np.array([N_DATA[n]['N'] for n in names])

    print(f"\n  Data points:")
    for i, name in enumerate(names):
        print(f"    {name:8s}: t_look = {t_all[i]:5.1f} Gyr, N = {n_all[i]:.2f}")

    # Full 3-point fit (unconstrained)
    from scipy.optimize import curve_fit as cf

    def log_model(t, a, b):
        return a + b * np.log(t)

    popt_full, _ = cf(log_model, t_all, n_all)
    a_full, b_full = popt_full
    resid_full = n_all - log_model(t_all, a_full, b_full)
    rms_full = float(np.sqrt(np.mean(resid_full**2)))

    print(f"\n  Full 3-point free fit:")
    print(f"    a = {a_full:.4f}, slope = {b_full:.4f}")
    print(f"    RMS residual = {rms_full:.4f}")

    # Leave-one-out
    loo_results = {}
    max_pred_error = 0.0

    print(f"\n  Leave-one-out cross-validation:")
    for i, left_out in enumerate(names):
        # Fit with remaining 2 points
        mask = np.array([j != i for j in range(len(names))])
        t_train = t_all[mask]
        n_train = n_all[mask]

        # With 2 points, solve exactly: N = a + b*ln(t)
        ln_t = np.log(t_train)
        A_mat = np.column_stack([np.ones(2), ln_t])
        params = np.linalg.solve(A_mat, n_train)
        a_loo, b_loo = params

        # Predict left-out point
        n_pred = a_loo + b_loo * np.log(t_all[i])
        pred_error = abs(n_pred - n_all[i])
        max_pred_error = max(max_pred_error, pred_error)

        # Slope deviation from full fit
        slope_dev = abs(b_loo - b_full) / b_full * 100

        print(f"\n    Left out: {left_out}")
        print(f"      Fit: a = {a_loo:.4f}, slope = {b_loo:.4f} "
              f"(slope dev from full: {slope_dev:.1f}%)")
        print(f"      Prediction: N_pred = {n_pred:.2f}, "
              f"N_obs = {n_all[i]:.2f}, error = {pred_error:.3f}")

        loo_results[left_out] = {
            'a': float(a_loo),
            'slope': float(b_loo),
            'slope_deviation_pct': float(slope_dev),
            'N_pred': float(n_pred),
            'N_obs': float(n_all[i]),
            'prediction_error': float(pred_error),
        }

    print(f"\n  Maximum prediction error: {max_pred_error:.4f}")
    print(f"  Threshold: 0.5")

    # Context: B_DFT residual for comparison
    a_dft, slope_dft, rms_dft = cascade_clock_fit(constrained=True)
    print(f"\n  Context:")
    print(f"    RMS (free fit):      {rms_full:.4f}")
    print(f"    RMS (B_DFT fit):     {rms_dft:.4f}")
    print(f"    Max LOO error:       {max_pred_error:.4f}")
    print(f"    If LOO error >> RMS, the fit is sensitive to point selection.")

    passed = max_pred_error > 0.5
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: leave-one-out "
          f"{'shows instability' if passed else 'shows stability'} "
          f"(max error = {max_pred_error:.4f} {'>' if passed else '<='} 0.5)")

    return {
        'test': 'leave_one_out_stability',
        'full_fit_a': float(a_full),
        'full_fit_slope': float(b_full),
        'full_fit_rms': rms_full,
        'dft_fit_rms': float(rms_dft),
        'loo_results': loo_results,
        'max_prediction_error': float(max_pred_error),
        'passed': bool(passed),
    }


def test4_monte_carlo_sensitivity():
    """
    Perturb N_data values by drawing from N(N_obs, sigma=0.5) for
    10,000 trials. Refit the free slope each time. Compute the 95% CI
    of the slope distribution.

    PASS if B_DFT = 2.0781 falls within the 95% CI.
    """
    print("\n" + "-" * 70)
    print("TEST 4: MONTE CARLO SENSITIVITY")
    print("-" * 70)

    n_trials = 10000
    sigma = 0.5
    rng = np.random.default_rng(42)

    names = list(N_DATA.keys())
    t_all = np.array([N_DATA[n]['t_lookback_gyr'] for n in names])
    n_obs = np.array([N_DATA[n]['N'] for n in names])
    ln_t = np.log(t_all)

    print(f"\n  Monte Carlo: {n_trials} trials, sigma = {sigma}")
    print(f"  Data points: {', '.join(names)}")
    print(f"  N_obs: [{', '.join(f'{n:.2f}' for n in n_obs)}]")

    slopes = []
    intercepts = []

    # Precompute design matrix for least-squares
    A_mat = np.column_stack([np.ones(len(t_all)), ln_t])

    for trial in range(n_trials):
        # Perturb N values
        n_perturbed = n_obs + rng.normal(0, sigma, size=len(n_obs))

        # Least-squares fit: N = a + b*ln(t)
        params, _, _, _ = np.linalg.lstsq(A_mat, n_perturbed, rcond=None)
        intercepts.append(params[0])
        slopes.append(params[1])

    slopes = np.array(slopes)
    intercepts = np.array(intercepts)

    # Statistics
    slope_mean = float(np.mean(slopes))
    slope_std = float(np.std(slopes))
    slope_median = float(np.median(slopes))

    # 95% CI (percentile method)
    ci_lo = float(np.percentile(slopes, 2.5))
    ci_hi = float(np.percentile(slopes, 97.5))

    # Where do B_DFT and B_FREE fall?
    b_dft_percentile = float(np.mean(slopes <= B_DFT) * 100)
    b_free_percentile = float(np.mean(slopes <= B_FREE) * 100)

    print(f"\n  Slope distribution:")
    print(f"    Mean   = {slope_mean:.4f}")
    print(f"    Median = {slope_median:.4f}")
    print(f"    Std    = {slope_std:.4f}")
    print(f"    95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"\n  Reference slopes:")
    print(f"    B_DFT  = {B_DFT:.4f} (percentile: {b_dft_percentile:.1f}%)")
    print(f"    B_FREE = {B_FREE:.4f} (percentile: {b_free_percentile:.1f}%)")

    # Histogram summary (10 bins)
    hist_counts, hist_edges = np.histogram(slopes, bins=10)
    print(f"\n  Slope histogram:")
    for i in range(len(hist_counts)):
        lo = hist_edges[i]
        hi = hist_edges[i + 1]
        bar_len = int(hist_counts[i] / max(hist_counts) * 30)
        bar = "#" * bar_len
        dft_marker = " <-- B_DFT" if lo <= B_DFT <= hi else ""
        free_marker = " <-- B_FREE" if lo <= B_FREE <= hi else ""
        print(f"    [{lo:.3f}, {hi:.3f}): {hist_counts[i]:4d} "
              f"{bar}{dft_marker}{free_marker}")

    b_dft_in_ci = ci_lo <= B_DFT <= ci_hi

    print(f"\n  B_DFT in 95% CI: {b_dft_in_ci}")
    if b_dft_in_ci:
        print(f"  --> The 8.9% discrepancy is consistent with noise in 3-point fit.")
    else:
        print(f"  --> The 8.9% discrepancy is statistically significant.")

    passed = b_dft_in_ci
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: B_DFT "
          f"{'inside' if passed else 'outside'} "
          f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        'test': 'monte_carlo_sensitivity',
        'n_trials': n_trials,
        'sigma': sigma,
        'slope_mean': slope_mean,
        'slope_std': slope_std,
        'slope_median': slope_median,
        'ci_95_lo': ci_lo,
        'ci_95_hi': ci_hi,
        'B_DFT': float(B_DFT),
        'B_FREE': float(B_FREE),
        'b_dft_percentile': b_dft_percentile,
        'b_free_percentile': b_free_percentile,
        'b_dft_in_95ci': bool(b_dft_in_ci),
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("EXP_03: SLOPE CORRECTION")
    print("Milestone 9 | Block A: Cascade Dynamics")
    print("=" * 70)

    r1 = test1_ghost_heart_correction()
    r2 = test2_ade_dimensional_correction()
    r3 = test3_leave_one_out_stability()
    r4 = test4_monte_carlo_sensitivity()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nScore: {n_passed}/4")
    for t in tests:
        status = "PASS" if t['passed'] else "FAIL"
        print(f"  [{status}] {t['test']}")

    if r4['passed']:
        print(f"\n  KEY FINDING: B_DFT falls within the 95% CI of the free-fit")
        print(f"  slope distribution. The 8.9% discrepancy may be a statistical")
        print(f"  artifact of fitting with only 3 data points.")

    results = {
        'experiment': 'exp_03_slope_correction',
        'milestone': 9,
        'block': 'A',
        'block_name': 'Cascade Dynamics',
        'tests': {t['test']: t for t in tests},
        'score': f'{n_passed}/4',
        'timestamp': datetime.now().isoformat(),
    }
    save_results(results, 'exp_03_slope_correction', RESULTS_DIR)


if __name__ == '__main__':
    main()
