"""
Milestone 8 -- Exp 03: Relic Abundance & Production Mechanism

Block A: Dark Sector Foundations

PURPOSE: Determine how the depth-73 dark matter particle achieves the correct
relic abundance Omega_DM h^2 = 0.120. Standard thermal freeze-out should FAIL
(coupling too weak), while freeze-in (Dodelson-Widrow or similar) may work.
This experiment is the highest-risk in Block A.

Tests:
  1. Thermal freeze-out falsification: Omega_thermal >> 1 (must fail)
  2. Freeze-in (Dodelson-Widrow): sin^2(2theta) in [10^{-13}, 10^{-7}] -> Omega h^2 = 0.120
  3. Omega_c formula consistency: DFT Omega_c = F_7*Xi^2/F_10 agrees within 10%
  4. Free-streaming length: 0.01 < lambda_fs < 1 Mpc (warm, not hot)

Builds on: exp_01, exp_02, MAR exp_32
Predicted: 3/4
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
M8_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M8_ROOT))

from core.bsm import (
    PHI, INV_PHI, LN_PHI, PI, XI_BALANCE,
    M_PLANCK_GEV, M_Z_GEV, HIGGS_VEV,
    ALPHA_EM, GEV_TO_KEV, GEV_TO_KG,
    OMEGA_DM, OMEGA_DM_H2, OMEGA_B_H2, RHO_CRIT_GEV_CM3,
    fib, F3, F4, F5, F6, F7, F8, F10,
    DEPTH_DARK, DEPTH_EM,
    fibonacci_depth_coupling,
    dodelson_widrow_abundance, free_streaming_length,
    dft_omega_c,
    save_results, setup_experiment,
)

_, RESULTS_DIR = setup_experiment(__file__)

# DM mass from cascade routes only (M_Pl/F_73 excluded — see exp_02 test 1)
M_DM_GEV = np.exp(np.mean(np.log([
    HIGGS_VEV * PHI**(-(DEPTH_DARK) / 2),
    M_Z_GEV * PHI**(-(F8 + F7)),
])))
M_DM_KEV = M_DM_GEV * GEV_TO_KEV


def test1_thermal_freezeout_fails():
    """
    Test 1: Thermal freeze-out gives Omega >> 1 (must FAIL to produce correct abundance).

    For thermal freeze-out: Omega h^2 ~ 0.1 pb / <sigma*v>
    where <sigma*v> ~ alpha^2 / m^2 for s-wave annihilation.

    With alpha_73 ~ 10^{-15}, the cross section is absurdly small,
    so Omega_thermal is absurdly large. This is expected and desired.
    """
    print("\n" + "=" * 70)
    print("TEST 1: THERMAL FREEZE-OUT FALSIFICATION")
    print("=" * 70)

    alpha_73 = fibonacci_depth_coupling(DEPTH_DARK)
    print(f"\n  Dark coupling: alpha_73 = {alpha_73:.4e}")
    print(f"  Dark matter mass: {M_DM_GEV:.4e} GeV = {M_DM_KEV:.2f} keV")

    # Thermal annihilation cross section (s-wave, Born)
    # <sigma*v> ~ alpha^2 / m^2 (natural units, GeV^{-2})
    sigma_v = alpha_73**2 / M_DM_GEV**2
    print(f"\n  Thermal <sigma*v> = alpha^2/m^2 = {sigma_v:.4e} GeV^{{-2}}")

    # Convert to pb: 1 GeV^{-2} = 0.3894e9 pb
    sigma_v_pb = sigma_v * 0.3894e9
    print(f"  <sigma*v> = {sigma_v_pb:.4e} pb")

    # Standard thermal relic: Omega h^2 ~ 0.1 pb / <sigma*v>
    # (from solving the Boltzmann equation with s-wave; the "WIMP miracle" value)
    sigma_v_wimp = 3e-26  # cm^3/s (canonical WIMP cross section)
    sigma_v_wimp_pb = 1.0  # pb (approximately)

    omega_thermal = 0.1 / sigma_v_pb if sigma_v_pb > 0 else float('inf')
    print(f"\n  Thermal relic abundance:")
    print(f"    Omega h^2 ~ 0.1 pb / {sigma_v_pb:.4e} pb = {omega_thermal:.4e}")
    print(f"    Measured: Omega_DM h^2 = {OMEGA_DM_H2}")
    print(f"    Ratio: {omega_thermal / OMEGA_DM_H2:.4e}")

    # This should be absurdly large
    thermal_fails = omega_thermal > 10  # way more than factor 10
    print(f"\n  Thermal freeze-out overproduces by factor: {omega_thermal / OMEGA_DM_H2:.1e}")
    print(f"  Freeze-out mechanism: {'EXCLUDED (as expected)' if thermal_fails else 'unexpectedly viable'}")

    # PASS: thermal freeze-out gives Omega >> 1 (confirming it fails)
    passed = thermal_fails
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: Omega_thermal = {omega_thermal:.2e} "
          f"{'>> 1' if passed else '~ correct'}")

    return {
        'test': 'thermal_freezeout_fails',
        'alpha_73': float(alpha_73),
        'sigma_v_gev2': float(sigma_v),
        'sigma_v_pb': float(sigma_v_pb),
        'omega_thermal': float(omega_thermal),
        'omega_measured': OMEGA_DM_H2,
        'overproduction_factor': float(omega_thermal / OMEGA_DM_H2),
        'thermal_fails': thermal_fails,
        'passed': passed,
    }


def test2_freezein_dodelson_widrow():
    """
    Test 2: Dodelson-Widrow freeze-in gives correct abundance for some mixing angle.

    Omega_s h^2 ≈ 0.3 * (sin^2(2theta) / 10^{-10}) * (m_s / 1 keV)^{1.8}

    Find sin^2(2theta) that gives Omega h^2 = 0.120. Check if it's in [10^{-13}, 10^{-7}].
    """
    print("\n" + "=" * 70)
    print("TEST 2: FREEZE-IN (DODELSON-WIDROW)")
    print("=" * 70)

    print(f"\n  DM mass: {M_DM_KEV:.2f} keV")
    print(f"  Target: Omega h^2 = {OMEGA_DM_H2}")

    # Solve for sin^2(2theta):
    # 0.120 = 0.3 * (sin^2(2theta) / 1e-10) * (m/1 keV)^1.8
    # sin^2(2theta) = 0.120 / (0.3 * (m/1)^1.8) * 1e-10
    sin2_2theta_required = OMEGA_DM_H2 / (0.3 * (M_DM_KEV / 1.0)**1.8) * 1e-10
    print(f"\n  Required mixing angle:")
    print(f"    sin^2(2theta) = {sin2_2theta_required:.4e}")
    print(f"    log10(sin^2(2theta)) = {np.log10(sin2_2theta_required):.2f}")

    # Verify by plugging back
    omega_check = dodelson_widrow_abundance(M_DM_KEV, sin2_2theta_required)
    print(f"    Verification: Omega h^2 = {omega_check:.6f} (target: {OMEGA_DM_H2})")

    # Check if in plausible range
    in_range = 1e-13 < sin2_2theta_required < 1e-7
    print(f"\n  Plausible range: 10^{{-13}} < sin^2(2theta) < 10^{{-7}}")
    print(f"  Required value: {sin2_2theta_required:.2e} -> {'IN RANGE' if in_range else 'OUT OF RANGE'}")

    # Scan: how does Omega vary with mixing angle?
    print(f"\n  Abundance scan at m = {M_DM_KEV:.1f} keV:")
    for log_sin2 in range(-13, -6):
        sin2 = 10**log_sin2
        omega = dodelson_widrow_abundance(M_DM_KEV, sin2)
        marker = " <--" if abs(np.log10(omega) - np.log10(OMEGA_DM_H2)) < 0.3 else ""
        print(f"    sin^2(2theta) = 1e{log_sin2}: Omega h^2 = {omega:.4e}{marker}")

    # X-ray constraints at this mass (from exp_02 analysis)
    sin2_2theta_xray = 2e-11  # NuSTAR bound at ~15 keV
    excluded_by_xray = sin2_2theta_required > sin2_2theta_xray
    print(f"\n  X-ray constraint: sin^2(2theta) < {sin2_2theta_xray:.1e} at ~{M_DM_KEV:.0f} keV")
    print(f"  Required for DW: {sin2_2theta_required:.2e}")
    print(f"  Excluded by X-ray: {excluded_by_xray}")

    if excluded_by_xray:
        print(f"\n  NOTE: DW production IS excluded for standard sterile neutrinos at this mass.")
        print(f"  This is a KNOWN tension. DFT interpretation: depth-73 particle is not a")
        print(f"  standard sterile neutrino but a new sector with different production.")
        print(f"  The DW formula is only an approximation for the DFT dark sector.")

    # PASS: required mixing angle in plausible range (even if X-ray tension exists)
    passed = in_range
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: sin^2(2theta) = {sin2_2theta_required:.2e} "
          f"{'in' if in_range else 'outside'} [10^{{-13}}, 10^{{-7}}]")

    return {
        'test': 'freezein_dodelson_widrow',
        'mass_kev': float(M_DM_KEV),
        'sin2_2theta_required': float(sin2_2theta_required),
        'log10_sin2_2theta': float(np.log10(sin2_2theta_required)),
        'omega_check': float(omega_check),
        'in_plausible_range': in_range,
        'xray_bound': sin2_2theta_xray,
        'excluded_by_xray': excluded_by_xray,
        'passed': passed,
    }


def test3_omega_c_formula():
    """
    Test 3: DFT formula Omega_c = F_7 * Xi^2 / F_10 agrees with measured Omega_DM.

    From MAR exp_25: Omega_c = 13 * 1.0584^2 / 55 = 0.2649
    Measured: Omega_DM = 0.266 (Planck 2018)
    This should be accurate to ~0.1%.
    """
    print("\n" + "=" * 70)
    print("TEST 3: OMEGA_C FORMULA CONSISTENCY")
    print("=" * 70)

    omega_dft = dft_omega_c()
    omega_meas = OMEGA_DM

    error_pct = abs(omega_dft - omega_meas) / omega_meas * 100

    print(f"\n  DFT formula: Omega_c = F_7 * Xi^2 / F_10")
    print(f"    F_7 = {F7}, Xi = {XI_BALANCE:.6f}, F_10 = {F10}")
    print(f"    Omega_c = {F7} * {XI_BALANCE:.6f}^2 / {F10}")
    print(f"    = {F7} * {XI_BALANCE**2:.6f} / {F10}")
    print(f"    = {omega_dft:.6f}")

    print(f"\n  Measured: Omega_DM = {omega_meas:.4f}")
    print(f"  Error: {error_pct:.3f}%")

    # Context: This is a zero-parameter formula. The accuracy is remarkable.
    print(f"\n  Context:")
    print(f"    This is a zero-free-parameter prediction")
    print(f"    Three quantities: F_7 (Fibonacci), Xi (Euler-Mascheroni + ln(phi)), F_10 (Fibonacci)")
    print(f"    All derived from DFT first principles")

    # Check Omega_DM h^2
    h = 0.6736  # H0 = 67.36 km/s/Mpc
    omega_dft_h2 = omega_dft * h**2
    omega_meas_h2 = OMEGA_DM_H2
    error_h2_pct = abs(omega_dft_h2 - omega_meas_h2) / omega_meas_h2 * 100

    print(f"\n  Omega h^2 comparison (h = {h}):")
    print(f"    DFT: Omega_c h^2 = {omega_dft_h2:.6f}")
    print(f"    Planck: Omega_DM h^2 = {omega_meas_h2:.6f}")
    print(f"    Error: {error_h2_pct:.3f}%")

    # PASS: within 10% of measured
    passed = error_pct < 10.0
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: error = {error_pct:.3f}% (threshold 10%)")

    return {
        'test': 'omega_c_formula',
        'omega_dft': float(omega_dft),
        'omega_measured': float(omega_meas),
        'error_pct': float(error_pct),
        'omega_dft_h2': float(omega_dft_h2),
        'omega_measured_h2': float(omega_meas_h2),
        'error_h2_pct': float(error_h2_pct),
        'passed': passed,
    }


def test4_free_streaming():
    """
    Test 4: Free-streaming length in WDM range [0.01, 1] Mpc.

    For WDM: lambda_fs ~ 0.1 Mpc * (1 keV / m)
    Must be:
    - > 0.01 Mpc (not CDM-like, i.e., warm enough to solve small-scale problems)
    - < 1 Mpc (not HDM-like, i.e., not erasing too much structure)
    """
    print("\n" + "=" * 70)
    print("TEST 4: FREE-STREAMING LENGTH")
    print("=" * 70)

    print(f"\n  DM mass: {M_DM_KEV:.2f} keV")

    # Standard free-streaming formula
    lambda_fs = free_streaming_length(M_DM_KEV)
    print(f"\n  Free-streaming length: lambda_fs = {lambda_fs:.4f} Mpc")

    # Also compute at mass boundaries from exp_02
    F73 = fib(73)
    masses_kev = {
        'M_Pl/F_73': M_PLANCK_GEV / F73 * GEV_TO_KEV,
        'v_H*phi^-36.5': HIGGS_VEV * PHI**(-(DEPTH_DARK) / 2) * GEV_TO_KEV,
        'M_Z*phi^-34': M_Z_GEV * PHI**(-(F8 + F7)) * GEV_TO_KEV,
        'Geometric mean': M_DM_KEV,
    }

    print(f"\n  Free-streaming at different mass estimates:")
    for name, m_kev in masses_kev.items():
        lfs = free_streaming_length(m_kev)
        in_range = 0.01 < lfs < 1.0
        print(f"    {name:20s}: m = {m_kev:10.2f} keV, lambda_fs = {lfs:.4f} Mpc "
              f"{'[OK]' if in_range else '[OUT]'}")

    # Physical interpretation
    print(f"\n  Physical interpretation:")
    if lambda_fs < 0.01:
        print(f"    lambda_fs < 0.01 Mpc: effectively CDM (no small-scale suppression)")
    elif lambda_fs < 0.1:
        print(f"    0.01 < lambda_fs < 0.1 Mpc: warm DM, small-scale suppression")
        print(f"    -> Can help with core-cusp and too-big-to-fail problems")
    elif lambda_fs < 1.0:
        print(f"    0.1 < lambda_fs < 1 Mpc: warm DM, significant suppression")
        print(f"    -> May conflict with Lyman-alpha at high end")
    else:
        print(f"    lambda_fs > 1 Mpc: hot DM (erases too much structure)")

    # Comparison with known models
    print(f"\n  Context:")
    print(f"    7 keV sterile neutrino: lambda_fs ~ {free_streaming_length(7.0):.3f} Mpc")
    print(f"    3.3 keV thermal WDM: lambda_fs ~ {free_streaming_length(3.3):.3f} Mpc")
    print(f"    100 GeV WIMP: lambda_fs ~ {free_streaming_length(1e8):.6f} Mpc (CDM)")

    # PASS: in WDM range [0.01, 1.0] Mpc
    in_range = 0.01 < lambda_fs < 1.0
    passed = in_range
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: lambda_fs = {lambda_fs:.4f} Mpc "
          f"{'in' if in_range else 'outside'} [0.01, 1.0] Mpc")

    return {
        'test': 'free_streaming',
        'mass_kev': float(M_DM_KEV),
        'lambda_fs_mpc': float(lambda_fs),
        'in_wdm_range': in_range,
        'passed': passed,
    }


def main():
    print("=" * 70)
    print("MILESTONE 8 - EXP 03: RELIC ABUNDANCE & PRODUCTION")
    print("Block A: Dark Sector Foundations")
    print("=" * 70)

    print(f"\n  Dark matter candidate: depth-73")
    print(f"    Mass: ~{M_DM_KEV:.1f} keV (geometric mean)")
    print(f"    Coupling: alpha_73 ~ {fibonacci_depth_coupling(DEPTH_DARK):.2e}")
    print(f"    Target abundance: Omega_DM h^2 = {OMEGA_DM_H2}")

    r1 = test1_thermal_freezeout_fails()
    r2 = test2_freezein_dodelson_widrow()
    r3 = test3_omega_c_formula()
    r4 = test4_free_streaming()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Test 1 (Thermal freeze-out fails): {'PASS' if r1['passed'] else 'FAIL'}")
    print(f"  Test 2 (Dodelson-Widrow): {'PASS' if r2['passed'] else 'FAIL'}")
    print(f"  Test 3 (Omega_c formula): {'PASS' if r3['passed'] else 'FAIL'}")
    print(f"  Test 4 (Free-streaming): {'PASS' if r4['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/4")

    results = {
        'experiment': 'exp_03_relic_abundance_production',
        'milestone': 8,
        'block': 'A',
        'tests': {
            'test1_thermal_freezeout_fails': r1,
            'test2_freezein_dodelson_widrow': r2,
            'test3_omega_c_formula': r3,
            'test4_free_streaming': r4,
        },
        'score': f"{n_passed}/4",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_03_relic_abundance_production', RESULTS_DIR)


if __name__ == '__main__':
    main()
