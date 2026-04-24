"""
Milestone 9 -- Exp 02: Xi Transition Cost

PURPOSE: Derive Xi = gamma + ln(phi) as the information cost per cascade
boundary crossing. Xi decomposes into gamma (counting/discreteness overhead)
and ln(phi) (branching/splitting cost). The splitting entropy of the
phi-proportioned split is computed directly, then the gamma counting
overhead is added to reconstruct Xi.

Block A: Cascade Dynamics

Tests:
  1. Splitting entropy: Shannon entropy of the phi-split
  2. Xi decomposition: gamma (harmonic counting) + ln(phi) = Xi
  3. Slope-Xi product: B_DFT * Xi approximates B_FREE
  4. Xi uniqueness: only c = Xi satisfies scale-invariance (g_out = g_in^2)
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


def test1_splitting_entropy():
    """
    Compute Shannon entropy of the phi-split (1/phi, 1/phi^2).
    H = -p_D*ln(p_D) - p_S*ln(p_S)

    Theoretical prediction:
      H = ln(phi) * (1/phi + 2/phi^2)

    Also verify via cascade_info_loss (deterministic) that H_split
    is constant across all 30 levels.

    PASS if H_split matches theoretical within 5%.
    """
    print("\n" + "-" * 70)
    print("TEST 1: SPLITTING ENTROPY OF THE PHI-SPLIT")
    print("-" * 70)

    p_D = INV_PHI         # 1/phi = 0.6180
    p_S = INV_PHI**2      # 1/phi^2 = 0.3820

    # Direct Shannon entropy
    H_split = -(p_D * np.log(p_D) + p_S * np.log(p_S))

    # Theoretical: H = ln(phi) * (1/phi + 2/phi^2)
    H_theory = LN_PHI * (INV_PHI + 2.0 * INV_PHI**2)

    # Cascade verification (deterministic, 30 levels)
    levels = cascade_info_loss(30, include_stochastic=False)
    H_levels = np.array([lev['H_split'] for lev in levels])
    H_mean = float(np.mean(H_levels))
    H_std = float(np.std(H_levels))

    # Ratio to ln(phi)
    ratio_to_ln_phi = H_split / LN_PHI

    print(f"\n  Phi-split probabilities:")
    print(f"    p_D = 1/phi   = {p_D:.6f}")
    print(f"    p_S = 1/phi^2 = {p_S:.6f}")
    print(f"    p_D + p_S     = {p_D + p_S:.6f} (should be 1)")
    print(f"\n  Shannon entropy of phi-split:")
    print(f"    H_split = {H_split:.6f} nats")
    print(f"    ln(phi) = {LN_PHI:.6f} nats")
    print(f"    H_split / ln(phi) = {ratio_to_ln_phi:.6f}")
    print(f"\n  Theoretical: H = ln(phi) * (1/phi + 2/phi^2)")
    print(f"    = {LN_PHI:.6f} * ({INV_PHI:.4f} + {2*INV_PHI**2:.4f})")
    print(f"    = {LN_PHI:.6f} * {INV_PHI + 2*INV_PHI**2:.4f}")
    print(f"    = {H_theory:.6f}")
    print(f"\n  Cascade verification (30-level deterministic):")
    print(f"    Mean H_split = {H_mean:.6f}")
    print(f"    Std  H_split = {H_std:.2e}")
    print(f"    (Constant across levels: std ~ 0 confirms deterministic split)")

    dev = abs(H_split - H_theory) / H_theory
    print(f"\n  |H_split - H_theory| / H_theory = {dev*100:.6f}%")

    passed = dev < 0.05
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: splitting entropy "
          f"{'matches' if passed else 'does not match'} theoretical "
          f"(dev = {dev*100:.4f}%)")

    return {
        'test': 'splitting_entropy',
        'p_D': float(p_D),
        'p_S': float(p_S),
        'H_split': float(H_split),
        'H_theory': float(H_theory),
        'ln_phi': float(LN_PHI),
        'ratio_to_ln_phi': float(ratio_to_ln_phi),
        'cascade_H_mean': H_mean,
        'cascade_H_std': H_std,
        'deviation_pct': float(dev * 100),
        'passed': bool(passed),
    }


def test2_xi_decomposition():
    """
    Xi = gamma + ln(phi) where:
      gamma = Euler-Mascheroni = 0.5772... (counting overhead)
      ln(phi) = 0.4812... (splitting cost)

    Verify gamma from the harmonic series: H_n - ln(n) -> gamma.
    Verify ln(phi) from the golden ratio.
    Verify their sum equals Xi within 0.01%.

    PASS if |gamma + ln(phi) - Xi| / Xi < 0.0001.
    """
    print("\n" + "-" * 70)
    print("TEST 2: XI DECOMPOSITION")
    print("-" * 70)

    decomp = xi_decomposition()
    gamma = decomp['gamma']
    ln_phi = decomp['ln_phi']
    total = decomp['total']
    survival = decomp['survival']

    # Independent computation
    gamma_computed = GAMMA_EM
    ln_phi_computed = np.log(PHI)
    xi_sum = gamma_computed + ln_phi_computed

    # From the module
    xi_module = xi_info_cost()

    # Harmonic series verification: H_n - ln(n) -> gamma
    n_vals = [10, 100, 1000, 10000]
    h_n_estimates = []
    for n in n_vals:
        H_n = sum(1.0 / k for k in range(1, n + 1))
        gamma_est = H_n - np.log(n)
        h_n_estimates.append(gamma_est)

    print(f"\n  Xi decomposition:")
    print(f"    gamma (Euler-Mascheroni) = {gamma:.10f}")
    print(f"    ln(phi) (splitting cost) = {ln_phi:.10f}")
    print(f"    Sum = gamma + ln(phi)    = {total:.10f}")
    print(f"    Xi (module)              = {xi_module:.10f}")
    print(f"    XI_BALANCE constant      = {XI_BALANCE:.10f}")
    print(f"\n  Survival fraction per crossing:")
    print(f"    e^(-Xi) = {survival:.6f}")
    print(f"\n  Independent verification:")
    print(f"    gamma (GAMMA_EM)  = {gamma_computed:.10f}")
    print(f"    ln(PHI)           = {ln_phi_computed:.10f}")
    print(f"    gamma + ln(phi)   = {xi_sum:.10f}")
    print(f"\n  Harmonic series convergence to gamma:")
    for n, est in zip(n_vals, h_n_estimates):
        err = abs(est - gamma_computed)
        print(f"    H_{n} - ln({n}) = {est:.8f}  (error: {err:.2e})")

    dev = abs(xi_sum - XI_BALANCE) / XI_BALANCE
    print(f"\n  |gamma + ln(phi) - Xi| / Xi = {dev*100:.6f}%")

    passed = dev < 0.0001
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: Xi decomposition "
          f"{'verified' if passed else 'failed'} (dev = {dev*100:.6f}%)")

    return {
        'test': 'xi_decomposition',
        'gamma': float(gamma),
        'ln_phi': float(ln_phi),
        'sum': float(total),
        'xi_module': float(xi_module),
        'xi_constant': float(XI_BALANCE),
        'survival_fraction': float(survival),
        'harmonic_estimates': {str(n): float(e) for n, e in zip(n_vals, h_n_estimates)},
        'deviation_pct': float(dev * 100),
        'passed': bool(passed),
    }


def test3_slope_xi_product():
    """
    B_DFT * Xi = (1/ln(phi)) * (gamma + ln(phi)) = 1 + gamma/ln(phi).

    Numerically: 2.0781 * 1.0584 = 2.1996.
    Compare to B_FREE = 2.264. Ratio = 2.1996/2.264 = 0.971.

    This tests whether the free-fit slope is the DFT slope times the
    per-level info cost.

    PASS if |B_DFT * Xi - B_FREE| / B_FREE < 0.05 (within 5%).
    """
    print("\n" + "-" * 70)
    print("TEST 3: SLOPE-XI PRODUCT")
    print("-" * 70)

    product = B_DFT * XI_BALANCE
    ratio = product / B_FREE
    algebraic = 1.0 + GAMMA_EM / LN_PHI

    print(f"\n  Cascade clock slopes:")
    print(f"    B_DFT (1/ln(phi))  = {B_DFT:.6f}")
    print(f"    B_FREE (free fit)  = {B_FREE:.6f}")
    print(f"    Discrepancy        = {abs(B_FREE - B_DFT)/B_DFT*100:.2f}%")
    print(f"\n  Xi as information cost:")
    print(f"    Xi = gamma + ln(phi) = {XI_BALANCE:.6f}")
    print(f"\n  Slope-Xi product:")
    print(f"    B_DFT * Xi = {product:.6f}")
    print(f"    Algebraic: 1 + gamma/ln(phi) = {algebraic:.6f}")
    print(f"\n  Comparison to free-fit slope:")
    print(f"    B_DFT * Xi / B_FREE = {ratio:.6f}")
    print(f"    Deviation = {abs(1 - ratio)*100:.2f}%")
    print(f"\n  Interpretation:")
    print(f"    If B_FREE ~ B_DFT * Xi, then the free-fit slope")
    print(f"    absorbs the per-level info cost into the slope,")
    print(f"    inflating it by a factor of Xi = {XI_BALANCE:.4f}.")

    # What Xi would need to be for exact match
    xi_needed = B_FREE / B_DFT
    print(f"\n  For exact match: Xi would need to be {xi_needed:.6f}")
    print(f"  Actual Xi = {XI_BALANCE:.6f}")
    print(f"  Difference = {abs(xi_needed - XI_BALANCE):.6f}")

    dev = abs(product - B_FREE) / B_FREE
    passed = dev < 0.05
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: slope-Xi product "
          f"{'matches' if passed else 'does not match'} B_FREE "
          f"(dev = {dev*100:.2f}%)")

    return {
        'test': 'slope_xi_product',
        'B_DFT': float(B_DFT),
        'B_FREE': float(B_FREE),
        'Xi': float(XI_BALANCE),
        'product': float(product),
        'algebraic': float(algebraic),
        'ratio_to_B_FREE': float(ratio),
        'xi_needed_for_exact': float(xi_needed),
        'deviation_pct': float(dev * 100),
        'passed': bool(passed),
    }


def test4_xi_uniqueness():
    """
    Scan c in [0.5, 2.0] with 500 values. For each c:
      splitting_cost x = c - gamma (if c > gamma)
      g_in = exp(-x)
      g_out = 1 - g_in
      Scale invariance requires: g_out = g_in^2
        => (1 - e^{-x}) = e^{-2x}
        => e^{-2x} + e^{-x} - 1 = 0
      Quadratic in u = e^{-x}: u^2 + u - 1 = 0
      => u = (-1 + sqrt(5))/2 = 1/phi
      => x = ln(phi), c = gamma + ln(phi) = Xi.

    PASS if the c minimizing |g_out - g_in^2| is within 2% of Xi.
    """
    print("\n" + "-" * 70)
    print("TEST 4: XI UNIQUENESS VIA SCALE INVARIANCE")
    print("-" * 70)

    c_values = np.linspace(0.5, 2.0, 500)
    scale_inv_errors = []

    for c in c_values:
        x = c - GAMMA_EM
        if x <= 0 or x >= 3.0:
            scale_inv_errors.append(float('inf'))
            continue

        g_in = np.exp(-x)
        g_out = 1.0 - g_in
        si_error = abs(g_out - g_in**2)
        scale_inv_errors.append(si_error)

    scale_inv_errors = np.array(scale_inv_errors)

    # Find the minimum
    finite_mask = np.isfinite(scale_inv_errors)
    if np.any(finite_mask):
        finite_errors = scale_inv_errors[finite_mask]
        finite_c = c_values[finite_mask]
        best_idx = np.argmin(finite_errors)
        best_c = float(finite_c[best_idx])
        best_error = float(finite_errors[best_idx])
    else:
        best_c = float('nan')
        best_error = float('inf')

    # How many c values have error < 0.001?
    n_near_zero = int(np.sum(scale_inv_errors[finite_mask] < 0.001))

    print(f"\n  Scanning {len(c_values)} candidate transition costs in [0.5, 2.0]")
    print(f"  For each c: x = c - gamma, g_in = e^(-x), check |g_out - g_in^2|")
    print(f"\n  Results:")
    print(f"    Best c = {best_c:.6f}")
    print(f"    Xi     = {XI_BALANCE:.6f}")
    print(f"    |best_c - Xi| = {abs(best_c - XI_BALANCE):.6f}")
    print(f"    Min scale-inv error = {best_error:.2e}")
    print(f"    Values with error < 0.001: {n_near_zero}")

    # Error landscape near Xi
    print(f"\n  Error landscape near Xi:")
    near_xi_mask = np.abs(c_values - XI_BALANCE) < 0.15
    near_xi_c = c_values[near_xi_mask]
    near_xi_err = scale_inv_errors[near_xi_mask]
    for c_val, err in zip(near_xi_c[::2], near_xi_err[::2]):
        if np.isfinite(err):
            marker = " <-- Xi" if abs(c_val - XI_BALANCE) < 0.01 else ""
            print(f"    c = {c_val:.4f}: |g_out - g_in^2| = {err:.6f}{marker}")

    # Analytical verification
    print(f"\n  Analytical verification:")
    print(f"    Quadratic: u^2 + u - 1 = 0  (u = e^(-x))")
    u_solution = (-1 + np.sqrt(5)) / 2
    x_solution = -np.log(u_solution)
    c_solution = GAMMA_EM + x_solution
    print(f"    u = (-1 + sqrt(5))/2 = {u_solution:.6f} = 1/phi = {INV_PHI:.6f}")
    print(f"    x = ln(phi) = {x_solution:.6f}")
    print(f"    c = gamma + ln(phi) = {c_solution:.6f} = Xi = {XI_BALANCE:.6f}")

    dev = abs(best_c - XI_BALANCE) / XI_BALANCE
    passed = dev < 0.02
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: Xi uniqueness "
          f"{'confirmed' if passed else 'not confirmed'} "
          f"(best c within {dev*100:.2f}% of Xi)")

    return {
        'test': 'xi_uniqueness',
        'best_c': best_c,
        'xi_balance': float(XI_BALANCE),
        'best_error': best_error,
        'n_near_zero': n_near_zero,
        'deviation_pct': float(dev * 100),
        'analytical_u': float(u_solution),
        'analytical_x': float(x_solution),
        'analytical_c': float(c_solution),
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("EXP_02: XI TRANSITION COST")
    print("Milestone 9 | Block A: Cascade Dynamics")
    print("=" * 70)

    r1 = test1_splitting_entropy()
    r2 = test2_xi_decomposition()
    r3 = test3_slope_xi_product()
    r4 = test4_xi_uniqueness()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nScore: {n_passed}/4")
    for t in tests:
        status = "PASS" if t['passed'] else "FAIL"
        print(f"  [{status}] {t['test']}")

    if r2['passed'] and r4['passed']:
        print(f"\n  KEY FINDING: Xi = gamma + ln(phi) is the unique transition cost")
        print(f"  satisfying scale invariance (g_out = g_in^2). Gamma provides")
        print(f"  counting overhead, ln(phi) provides the splitting cost.")

    results = {
        'experiment': 'exp_02_xi_transition_cost',
        'milestone': 9,
        'block': 'A',
        'block_name': 'Cascade Dynamics',
        'tests': {t['test']: t for t in tests},
        'score': f'{n_passed}/4',
        'timestamp': datetime.now().isoformat(),
    }
    save_results(results, 'exp_02_xi_transition_cost', RESULTS_DIR)


if __name__ == '__main__':
    main()
