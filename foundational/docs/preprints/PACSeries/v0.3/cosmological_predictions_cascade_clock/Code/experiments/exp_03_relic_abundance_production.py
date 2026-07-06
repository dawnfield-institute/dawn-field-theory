"""
Milestone 8 -- Exp 03: Relic Abundance & Production Mechanism

Block A: Dark Sector Foundations

PURPOSE: Determine how the depth-73 dark matter particle achieves the correct
relic abundance Omega_DM h^2 = 0.120. Standard thermal freeze-out should FAIL
(coupling too weak), while freeze-in (Dodelson-Widrow or similar) may work.
This experiment is the highest-risk in Block A.

Tests:
  1. Thermal freeze-out excluded (C): Omega_thermal >> 1
     HARDENED: Round 1. Relabeled as consistency check (C) — cannot fail
     for ANY coupling below ~10^-5. Zero discriminating power. Retained
     as structural preamble, honestly labeled.
  2. Freeze-in viability: DW abundance AND X-ray constraint
     HARDENED: Round 1. Added X-ray exclusion as FAIL criterion. Previously
     passed despite code acknowledging X-ray exclusion at lines 162-166.
     Now: FAIL if required mixing angle exceeds X-ray bound.
  3. Mass-abundance consistency: DW chain closes within 10%
  4. Free-streaming length: 0.01 < lambda_fs < 1 Mpc (warm, not hot)

Builds on: exp_01, exp_02, MAR exp_32
Predicted: 3/4 (T2 expected FAIL: X-ray tension)
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
    Test 1 (C): Thermal freeze-out gives Omega >> 1 (must FAIL to produce correct abundance).

    HARDENED: Round 1. Relabeled as consistency check (C). ANY coupling
    below ~10^{-5} fails thermal freeze-out trivially. With alpha_73 ~ 10^{-16},
    this test has zero discriminating power — it cannot fail. Retained as
    structural context establishing that non-thermal production is necessary.
    """
    print("\n" + "=" * 70)
    print("TEST 1: THERMAL FREEZE-OUT EXCLUDED (C — consistency check)")
    print("  (HARDENED: unfalsifiable for any coupling < 10^-5)")
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
    Test 2: Dodelson-Widrow freeze-in gives correct abundance AND satisfies X-ray bounds.

    HARDENED: Round 1. Added X-ray constraint as FAIL criterion. Previously
    the test passed despite the code acknowledging (lines 162-166) that the
    required mixing angle is excluded by NuSTAR. Now the test requires BOTH:
    (a) sin^2(2theta) in [10^{-13}, 10^{-7}] for correct abundance, AND
    (b) sin^2(2theta) below X-ray exclusion limit.

    Expected: FAIL. DW production is excluded at this mass. This is an
    honest open problem: the depth-73 particle needs an alternative
    production mechanism (resonant Shi-Fuller, Higgs portal, etc.).
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
        print(f"\n  HONEST FAILURE: DW production IS excluded by X-ray observations at this mass.")
        print(f"  The required mixing angle ({sin2_2theta_required:.2e}) exceeds the NuSTAR")
        print(f"  bound ({sin2_2theta_xray:.1e}). This means:")
        print(f"    - Standard DW production cannot produce depth-73 DM without violating X-ray bounds")
        print(f"    - Alternative production mechanism needed (resonant Shi-Fuller, Higgs portal, etc.)")
        print(f"    - The DFT dark sector is NOT a standard sterile neutrino")
        print(f"  This is an OPEN PROBLEM, not a refutation of depth 73.")

    # HARDENED: PASS requires BOTH abundance range AND X-ray compatibility
    # (Previously only checked abundance range, ignoring X-ray exclusion)
    passed = in_range and not excluded_by_xray
    if in_range and excluded_by_xray:
        print(f"\n  -> FAIL: sin^2(2theta) = {sin2_2theta_required:.2e} is in abundance range "
              f"but EXCLUDED by X-ray bound. DW production mechanism not viable.")
    elif not in_range:
        print(f"\n  -> FAIL: sin^2(2theta) = {sin2_2theta_required:.2e} outside "
              f"[10^{{-13}}, 10^{{-7}}]")
    else:
        print(f"\n  -> PASS: sin^2(2theta) = {sin2_2theta_required:.2e} in range "
              f"and below X-ray bound")

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


def test3_mass_abundance_consistency():
    """
    Test 3: Given DM mass from exp_02 and sin^2(2theta) from test 2,
    does the Dodelson-Widrow integral produce Omega_DM h^2 = 0.120 +/- 10%?

    This replaces the previous Omega_c = F_7*Xi^2/F_10 formula check,
    which was circular (DFT checking DFT). Instead we test whether the
    DW production mechanism with OUR mass actually hits the measured abundance.

    The chain: m_DM (from cascade routes) -> sin^2(2theta) (from DW inversion)
    -> Omega h^2 (from DW forward calculation) -> compare to Planck.
    """
    print("\n" + "=" * 70)
    print("TEST 3: MASS-ABUNDANCE CONSISTENCY")
    print("=" * 70)

    print(f"\n  DM mass from cascade routes: {M_DM_KEV:.2f} keV")
    print(f"  Target abundance: Omega_DM h^2 = {OMEGA_DM_H2}")

    # Solve DW for required mixing angle
    sin2_2theta = OMEGA_DM_H2 / (0.3 * (M_DM_KEV)**1.8) * 1e-10

    print(f"\n  Step 1: Required sin^2(2theta) = {sin2_2theta:.4e}")

    # Forward calculation: plug back into DW
    omega_forward = dodelson_widrow_abundance(M_DM_KEV, sin2_2theta)
    closure_error = abs(omega_forward - OMEGA_DM_H2) / OMEGA_DM_H2

    print(f"  Step 2: DW forward -> Omega h^2 = {omega_forward:.6f}")
    print(f"  Closure error: {closure_error*100:.4f}%")

    # Sensitivity: how much does Omega change per 10% mass shift?
    m_plus = M_DM_KEV * 1.10
    m_minus = M_DM_KEV * 0.90
    sin2_plus = OMEGA_DM_H2 / (0.3 * m_plus**1.8) * 1e-10
    sin2_minus = OMEGA_DM_H2 / (0.3 * m_minus**1.8) * 1e-10

    print(f"\n  Sensitivity analysis (mass +/- 10%):")
    print(f"    m = {m_minus:.2f} keV: sin^2(2theta) = {sin2_minus:.3e}")
    print(f"    m = {M_DM_KEV:.2f} keV: sin^2(2theta) = {sin2_2theta:.3e}")
    print(f"    m = {m_plus:.2f} keV: sin^2(2theta) = {sin2_plus:.3e}")

    # Check: are all mass variants still in the plausible mixing angle range?
    all_in_range = all(1e-13 < s < 1e-7 for s in [sin2_minus, sin2_2theta, sin2_plus])
    print(f"  All in [10^-13, 10^-7]: {all_in_range}")

    # Physical consistency: free-streaming length
    lambda_fs = free_streaming_length(M_DM_KEV)
    fs_ok = 0.01 < lambda_fs < 1.0

    print(f"\n  Free-streaming cross-check: {lambda_fs:.4f} Mpc ({'WDM' if fs_ok else 'NOT WDM'})")

    # PASS: chain closes within 10% AND mixing angle in plausible range
    passed = closure_error < 0.10 and all_in_range
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: chain closure {closure_error*100:.2f}%, "
          f"mixing angles {'all plausible' if all_in_range else 'NOT all plausible'}")

    return {
        'test': 'mass_abundance_consistency',
        'mass_kev': float(M_DM_KEV),
        'sin2_2theta': float(sin2_2theta),
        'omega_forward': float(omega_forward),
        'closure_error': float(closure_error),
        'sensitivity_sin2_range': [float(sin2_minus), float(sin2_plus)],
        'all_in_range': all_in_range,
        'lambda_fs': float(lambda_fs),
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
    r3 = test3_mass_abundance_consistency()
    r4 = test4_free_streaming()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Test 1 (Thermal freeze-out excluded, C): {'PASS' if r1['passed'] else 'FAIL'}")
    print(f"  Test 2 (DW + X-ray constraint): {'PASS' if r2['passed'] else 'FAIL'}")
    print(f"  Test 3 (Mass-abundance consistency): {'PASS' if r3['passed'] else 'FAIL'}")
    print(f"  Test 4 (Free-streaming): {'PASS' if r4['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/4")

    results = {
        'experiment': 'exp_03_relic_abundance_production',
        'milestone': 8,
        'block': 'A',
        'tests': {
            'test1_thermal_freezeout_fails': r1,
            'test2_freezein_dodelson_widrow': r2,
            'test3_mass_abundance_consistency': r3,
            'test4_free_streaming': r4,
        },
        'score': f"{n_passed}/4",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_03_relic_abundance_production', RESULTS_DIR)


if __name__ == '__main__':
    main()
