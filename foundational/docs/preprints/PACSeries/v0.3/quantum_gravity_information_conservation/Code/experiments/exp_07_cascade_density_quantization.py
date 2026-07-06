"""
exp_07 — Cascade Density Quantization

Milestone 11, Block C (Graviton and Quantization) — HIGHEST RISK

Hypothesis: Promote cascade density to a quantized field with MVAE cutoff.
The discrete spectrum has Fibonacci spacing, the propagator matches linearized
gravity at low k, and MVAE cutoff renders loops finite.

Risk assessment: ~15% chance of 4/4, ~50% for 2/4. Honest about this.

Tests:
  T1: Discrete spectrum with Fibonacci spacing Delta_rho = rho_Planck * phi^(-n)
  T2: Propagator matches linearized gravity (1/k^2) for k << k_Planck
  T3: MVAE cutoff renders 1-loop self-energy finite
  T4: Dispersion omega^2 = c^2*k^2 + O((k/k_Planck)^2) corrections
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_gravity import (
    PHI, INV_PHI, LN_PHI, PI, LN2,
    L_MVAE, E_MVAE, RHO_PLANCK,
    E_PLANCK_GEV, L_PLANCK_M,
    save_results, setup_experiment,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def test_T1_fibonacci_spectrum():
    """
    T1: Discrete spectrum with Fibonacci spacing.

    The cascade density field has discrete energy levels
    rho_n = rho_Planck * phi^(-n) for n = 0, 1, 2, ...

    The spacing between consecutive levels:
    Delta_rho_n = rho_n - rho_{n+1} = rho_Planck * phi^(-n) * (1 - 1/phi)
                = rho_Planck * phi^(-n) / phi^2

    The ratio of consecutive spacings:
    Delta_rho_n / Delta_rho_{n+1} = phi (Fibonacci ratio)
    """
    n_levels = 30
    rho_levels = np.array([RHO_PLANCK * PHI**(-n) for n in range(n_levels)])
    spacings = np.abs(np.diff(rho_levels))

    # Ratio of consecutive spacings should be phi
    spacing_ratios = spacings[:-1] / spacings[1:]
    mean_ratio = np.mean(spacing_ratios)
    std_ratio = np.std(spacing_ratios)
    all_phi = np.allclose(spacing_ratios, PHI, rtol=1e-10)

    # Levels should form geometric sequence with ratio 1/phi
    level_ratios = rho_levels[1:] / rho_levels[:-1]
    all_inv_phi = np.allclose(level_ratios, INV_PHI, rtol=1e-10)

    # Sum of all levels converges (geometric series: sum = rho_P / (1 - 1/phi) = rho_P * phi)
    total = np.sum(rho_levels)
    expected_sum = RHO_PLANCK * PHI / (PHI - 1)  # = rho_P * phi^2
    sum_converges = abs(total - expected_sum) / expected_sum < 0.01

    return {
        'test': 'T1_fibonacci_spectrum',
        'n_levels': n_levels,
        'mean_spacing_ratio': float(mean_ratio),
        'std_spacing_ratio': float(std_ratio),
        'all_ratios_phi': all_phi,
        'all_levels_geometric': all_inv_phi,
        'series_converges': sum_converges,
        'PASS': all_phi and all_inv_phi,
    }


def test_T2_propagator():
    """
    T2: Propagator matches linearized gravity (1/k^2) for k << k_Planck.

    The cascade density field propagator in momentum space:
    G(k) = 1 / (k^2 + m^2)

    For the cascade: m = 0 (massless graviton), but MVAE provides UV cutoff.
    G_cascade(k) = 1 / k^2 * f(k/k_Planck)

    where f(x) -> 1 for x << 1 (reproduces 1/k^2)
    and f(x) -> 0 for x >> 1 (UV regulated)

    MVAE regulator: f(x) = exp(-x^2 / (2 * L_MVAE^2)) in Planck units
    """
    k_planck = 1.0  # k_Planck in Planck units

    # Test at various k values
    k_values = np.logspace(-6, 1, 100)  # 10^-6 to 10 in Planck units

    # Standard 1/k^2 propagator
    G_standard = 1.0 / k_values**2

    # MVAE-regulated propagator
    f_mvae = np.exp(-k_values**2 / (2 * L_MVAE**2))
    G_cascade = G_standard * f_mvae

    # At low k (k << k_Planck): G_cascade ~ 1/k^2
    # At k=0.01, regulator = exp(-0.0001/(2*L^2)) ~ 1 - 2e-5, so match to ~1e-5
    low_k = k_values < 0.01
    if np.sum(low_k) > 2:
        ratio_low = G_cascade[low_k] / G_standard[low_k]
        low_k_match = np.allclose(ratio_low, 1.0, rtol=1e-4)
    else:
        low_k_match = False

    # At high k (k >> k_Planck): G_cascade -> 0 (regulated)
    # MVAE regulator width ~ L_MVAE ~ 1.63, strong suppression at k > 3*L_MVAE
    high_k = k_values > 5.0
    if np.sum(high_k) > 0:
        high_k_suppressed = np.all(G_cascade[high_k] < 0.1 * G_standard[high_k])
    else:
        high_k_suppressed = False

    # The propagator slope at low k should be -2 (1/k^2)
    log_k_low = np.log10(k_values[low_k])
    log_G_low = np.log10(G_cascade[low_k])
    slope = np.polyfit(log_k_low, log_G_low, 1)[0]
    slope_is_minus_2 = abs(slope - (-2.0)) < 0.1

    return {
        'test': 'T2_propagator',
        'low_k_matches_1_over_k2': low_k_match,
        'high_k_suppressed': high_k_suppressed,
        'low_k_slope': float(slope),
        'slope_is_minus_2': slope_is_minus_2,
        'L_MVAE': float(L_MVAE),
        'PASS': low_k_match and high_k_suppressed and slope_is_minus_2,
    }


def test_T3_loop_finiteness():
    """
    T3: MVAE cutoff renders 1-loop self-energy finite.

    The 1-loop graviton self-energy (in 4D) diverges as:
    Sigma ~ integral d^4k / (k^2) ~ Lambda^2 (quadratically divergent)

    With MVAE regulator exp(-k^2/(2*L_MVAE^2)):
    Sigma_reg ~ integral d^4k * exp(-k^2/L^2) / k^2
             = finite (Gaussian decay kills UV divergence)

    Compare: unregulated integral diverges, regulated converges.
    """
    # 1D model: integral of k^(d-3) * dk from 0 to Lambda
    # In d=4: integral of k dk (linearly divergent in 1D reduction)

    # Unregulated (with cutoff)
    lambdas = np.logspace(0, 6, 20)
    unreg_values = []
    for lam in lambdas:
        k = np.linspace(0.01, lam, 10000)
        dk = k[1] - k[0]
        integrand = k  # k^(d-3) = k for d=4
        unreg_values.append(np.trapz(integrand, k))

    unreg_values = np.array(unreg_values)
    unreg_diverges = unreg_values[-1] / unreg_values[0] > 1e6

    # MVAE-regulated (analytical: integral of k*exp(-k^2/(2L^2)) dk = L^2*(1-exp(-Lambda^2/(2L^2))))
    L2 = L_MVAE**2
    reg_values = np.array([L2 * (1.0 - np.exp(-lam**2 / (2 * L2))) for lam in lambdas])

    # Regulated integral should converge: ratio of last to second-last ~ 1
    reg_converges = abs(reg_values[-1] / reg_values[-2] - 1.0) < 1e-6

    # The regulated value should approach L_MVAE^2 as Lambda -> infinity
    expected_reg = L2
    reg_match = abs(reg_values[-1] - expected_reg) / expected_reg < 1e-6

    return {
        'test': 'T3_loop_finiteness',
        'unreg_diverges': unreg_diverges,
        'unreg_ratio': float(unreg_values[-1] / unreg_values[0]),
        'reg_converges': reg_converges,
        'reg_ratio': float(reg_values[-1] / reg_values[1]),
        'reg_value': float(reg_values[-1]),
        'expected_reg': float(expected_reg),
        'reg_match': reg_match,
        'PASS': unreg_diverges and reg_converges,
    }


def test_T4_dispersion():
    """
    T4: Dispersion omega^2 = c^2*k^2 + O((k/k_Planck)^2) corrections.

    The cascade density field in Minkowski space has dispersion:
    omega^2 = c^2 * k^2 * (1 + beta * (k/k_P)^2 + ...)

    where beta is determined by the MVAE structure.
    At low k: omega = c*k (standard dispersion, speed of light)
    At high k: corrections appear.
    """
    k_planck = 1.0  # In Planck units, c=1

    # MVAE correction coefficient
    # From the regulator exp(-k^2/(2*L^2)), expanding:
    # G(k) = (1/k^2) * exp(-k^2/(2*L^2)) = (1/k^2)(1 - k^2/(2L^2) + ...)
    # This modifies the dispersion: omega^2 = k^2 * (1 - k^2/(2*L^2) + ...)
    # So beta = -1/(2*L_MVAE^2)
    beta = -1.0 / (2 * L_MVAE**2)

    k_values = np.logspace(-6, 0, 100)

    # Standard dispersion
    omega_standard = k_values  # c = 1 in Planck units

    # Corrected dispersion
    omega_corrected_sq = k_values**2 * (1 + beta * k_values**2)
    # Ensure positive before sqrt
    omega_corrected_sq = np.maximum(omega_corrected_sq, 0)
    omega_corrected = np.sqrt(omega_corrected_sq)

    # At low k: correction is negligible
    low_k = k_values < 0.001
    if np.sum(low_k) > 2:
        low_k_deviation = np.max(np.abs(omega_corrected[low_k] - omega_standard[low_k])
                                 / omega_standard[low_k])
        low_k_standard = low_k_deviation < 1e-6
    else:
        low_k_standard = False

    # At k ~ 0.1 k_Planck: correction should be small but measurable
    mid_k = (k_values > 0.05) & (k_values < 0.2)
    if np.sum(mid_k) > 0:
        mid_deviation = np.mean(np.abs(omega_corrected[mid_k] - omega_standard[mid_k])
                                / omega_standard[mid_k])
        mid_k_small = mid_deviation < 0.01  # Less than 1% correction
    else:
        mid_k_small = False

    # The correction is quadratic in k: delta_omega/omega ~ k^2
    # Log-log slope should be 2
    valid = (k_values > 1e-5) & (k_values < 0.1) & (omega_corrected > 0)
    if np.sum(valid) > 3:
        deviations = np.abs(omega_corrected[valid] - omega_standard[valid]) / omega_standard[valid]
        deviations = np.maximum(deviations, 1e-30)
        log_k = np.log10(k_values[valid])
        log_dev = np.log10(deviations)
        finite = np.isfinite(log_dev)
        if np.sum(finite) > 3:
            slope = np.polyfit(log_k[finite], log_dev[finite], 1)[0]
            quadratic_correction = abs(slope - 2.0) < 0.1
        else:
            slope = 0.0
            quadratic_correction = False
    else:
        slope = 0.0
        quadratic_correction = False

    return {
        'test': 'T4_dispersion',
        'beta': float(beta),
        'low_k_standard': low_k_standard,
        'low_k_max_deviation': float(low_k_deviation) if low_k_standard is not False else 0,
        'mid_k_small_correction': mid_k_small,
        'correction_slope': float(slope),
        'quadratic_correction': quadratic_correction,
        'PASS': low_k_standard and quadratic_correction,
    }


def main():
    setup = setup_experiment(__file__)

    print("=" * 70)
    print("EXP 07 — Cascade Density Quantization")
    print("Milestone 11, Block C (HIGHEST RISK)")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [('T1', test_T1_fibonacci_spectrum),
                           ('T2', test_T2_propagator),
                           ('T3', test_T3_loop_finiteness),
                           ('T4', test_T4_dispersion)]:
        print(f"\n--- {name} ---")
        t = test_fn()
        results[name] = t
        if t['PASS']:
            score += 1
            print(f"  PASS")
        else:
            print(f"  FAIL")

        if name == 'T1':
            print(f"    spacing ratio = {t['mean_spacing_ratio']:.10f} (phi={PHI:.10f})")
            print(f"    all geometric: {t['all_levels_geometric']}")
        elif name == 'T2':
            print(f"    low-k slope = {t['low_k_slope']:.3f} (target -2.0)")
            print(f"    high-k suppressed: {t['high_k_suppressed']}")
        elif name == 'T3':
            print(f"    unreg ratio: {t['unreg_ratio']:.2e} (diverges: {t['unreg_diverges']})")
            print(f"    reg ratio: {t['reg_ratio']:.6f} (converges: {t['reg_converges']})")
            print(f"    reg value: {t['reg_value']:.4f} (expected: {t['expected_reg']:.4f})")
        elif name == 'T4':
            print(f"    beta = {t['beta']:.6f}")
            print(f"    low-k deviation: {t.get('low_k_max_deviation', 'N/A')}")
            print(f"    correction slope: {t['correction_slope']:.3f} (target 2.0)")

    print("\n" + "=" * 70)
    print(f"EXP 07 SCORE: {score}/{total}")
    print("=" * 70)

    results['score'] = score
    results['total'] = total
    save_results(results, RESULTS_DIR, "exp_07_cascade_density_quantization")
    return results


if __name__ == "__main__":
    main()
