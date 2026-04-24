"""
Milestone 9 -- Exp 07: S8 Redshift Evolution

PURPOSE: S8 varies with redshift because the cascade level N varies with
lookback time. The cascade clock predicts scale-dependent structure growth
suppression, which should resolve the S8 tension between Planck (CMB) and
weak lensing surveys (KiDS, DES) without introducing new physics by hand.

Block C: Scale-Dependent Predictions

Tests:
  1. S8(z) curve: variation exceeds 3% of S8_PLANCK across z=[0.1, 3.0]
  2. S8 at lensing z: matches lensing mean at z_eff=0.35 within 2-sigma
  3. Tension resolution: DFT reduces S8 tension vs LCDM
  4. Euclid falsification: DFT distinguishable from LCDM at >95% CL
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


def test1_s8_curve():
    """
    Test 1: Create a constrained CascadeClock. Compute S8 at a range of
    redshifts. Report the variation (max - min). The cascade clock should
    produce measurable S8 variation across cosmic time.

    PASS if the range spans > 3% of S8_PLANCK (variation > 0.025).
    """
    print("\n" + "-" * 70)
    print("TEST 1: S8(z) CURVE")
    print("-" * 70)

    clock = CascadeClock(constrained=True)
    z_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    print(f"\n  Cascade clock: a = {clock.a:.4f}, slope = {clock.slope:.4f}")
    print(f"  S8_PLANCK = {S8_PLANCK:.4f}")
    print(f"  Threshold: variation > {0.03 * S8_PLANCK:.4f} (3% of S8_PLANCK)")

    print(f"\n  {'z':>6s}  {'t_look (Gyr)':>12s}  {'N(t)':>8s}  {'S8(z)':>8s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*8}")

    s8_values = []
    for z in z_vals:
        s8_z = clock.s8(z)
        t_look = z_to_lookback(z)
        n_z = clock.N_at_z(z)
        s8_values.append(s8_z)
        print(f"  {z:6.2f}  {t_look:12.4f}  {n_z:8.4f}  {s8_z:8.4f}")

    s8_arr = np.array(s8_values)
    s8_range = np.max(s8_arr) - np.min(s8_arr)
    threshold = 0.03 * S8_PLANCK

    print(f"\n  S8 range: {np.max(s8_arr):.4f} - {np.min(s8_arr):.4f} = {s8_range:.4f}")
    print(f"  Threshold (3% of {S8_PLANCK:.3f}): {threshold:.4f}")

    passed = s8_range > threshold
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: S8 variation "
          f"{'exceeds' if passed else 'below'} 3% threshold")

    return {
        'test': 's8_curve',
        'z_values': z_vals,
        's8_values': [float(x) for x in s8_values],
        's8_range': float(s8_range),
        'threshold': float(threshold),
        'passed': bool(passed),
    }


def test2_s8_matches_lensing():
    """
    Test 2: At the effective lensing redshift z_eff = 0.35 (KiDS/DES),
    compute S8 from the cascade clock. Compare to the lensing mean
    S8_LENSING = (S8_KIDS + S8_DES) / 2 = 0.7675.

    PASS if within 2-sigma of lensing mean (sigma ~ 0.02, so within 0.04).
    """
    print("\n" + "-" * 70)
    print("TEST 2: S8 AT LENSING REDSHIFT")
    print("-" * 70)

    clock = CascadeClock(constrained=True)
    z_eff = 0.35
    sigma_lensing = 0.02

    s8_dft = clock.s8(z_eff)
    t_look = z_to_lookback(z_eff)
    n_z = clock.N_at_z(z_eff)

    diff = abs(s8_dft - S8_LENSING)
    n_sigma = diff / sigma_lensing

    print(f"\n  Effective lensing redshift: z = {z_eff}")
    print(f"  Lookback time: {t_look:.4f} Gyr")
    print(f"  Cascade level: N = {n_z:.4f}")

    print(f"\n  Lensing surveys:")
    print(f"    S8_KiDS   = {S8_KIDS:.4f}")
    print(f"    S8_DES    = {S8_DES:.4f}")
    print(f"    S8_LENSING (mean) = {S8_LENSING:.4f}")
    print(f"    sigma     = {sigma_lensing}")

    print(f"\n  DFT prediction:")
    print(f"    S8_DFT(z={z_eff}) = {s8_dft:.4f}")
    print(f"    |S8_DFT - S8_LENSING| = {diff:.4f}")
    print(f"    Deviation = {n_sigma:.2f} sigma")
    print(f"    Threshold: < 2 sigma ({2 * sigma_lensing:.4f})")

    passed = diff < 2 * sigma_lensing
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: S8 at lensing z "
          f"{'matches' if passed else 'does not match'} lensing mean "
          f"({n_sigma:.2f} sigma)")

    return {
        'test': 's8_matches_lensing',
        'z_eff': z_eff,
        's8_dft': float(s8_dft),
        's8_lensing': float(S8_LENSING),
        'diff': float(diff),
        'n_sigma': float(n_sigma),
        'sigma': sigma_lensing,
        'passed': bool(passed),
    }


def test3_tension_resolution():
    """
    Test 3: The S8 tension is |S8_Planck - S8_lensing| / sigma_lensing.
    LCDM predicts constant S8 = 0.832 at all z, giving ~3.2 sigma tension.
    DFT predicts scale-dependent S8(z), which should reduce the tension.

    PASS if DFT tension < LCDM tension (DFT reduces the discrepancy).
    """
    print("\n" + "-" * 70)
    print("TEST 3: TENSION RESOLUTION")
    print("-" * 70)

    clock = CascadeClock(constrained=True)
    z_eff = 0.35
    sigma_lensing = 0.02
    s8_lensing_obs = S8_LENSING  # 0.7675

    # LCDM prediction: constant S8 = S8_PLANCK at all z
    lcdm_tension = abs(S8_PLANCK - s8_lensing_obs) / sigma_lensing

    # DFT prediction: scale-dependent S8(z)
    s8_dft = clock.s8(z_eff)
    dft_tension = abs(s8_dft - s8_lensing_obs) / sigma_lensing

    print(f"\n  S8 tension (at z_eff = {z_eff}):")
    print(f"    Lensing measurement: {s8_lensing_obs:.4f} +/- {sigma_lensing}")

    print(f"\n  LCDM (constant S8):")
    print(f"    S8_pred = {S8_PLANCK:.4f}")
    print(f"    |S8_pred - S8_obs| / sigma = {lcdm_tension:.2f} sigma")

    print(f"\n  DFT (scale-dependent S8):")
    print(f"    S8_DFT(z={z_eff}) = {s8_dft:.4f}")
    print(f"    |S8_DFT - S8_obs| / sigma = {dft_tension:.2f} sigma")

    reduction = lcdm_tension - dft_tension
    reduction_pct = (reduction / lcdm_tension) * 100 if lcdm_tension > 0 else 0

    print(f"\n  Tension reduction:")
    print(f"    LCDM tension:  {lcdm_tension:.2f} sigma")
    print(f"    DFT tension:   {dft_tension:.2f} sigma")
    print(f"    Reduction:     {reduction:.2f} sigma ({reduction_pct:.1f}%)")

    passed = dft_tension < lcdm_tension
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: DFT "
          f"{'reduces' if passed else 'does not reduce'} S8 tension")

    return {
        'test': 'tension_resolution',
        'z_eff': z_eff,
        's8_lensing': float(s8_lensing_obs),
        's8_planck': float(S8_PLANCK),
        's8_dft': float(s8_dft),
        'lcdm_tension_sigma': float(lcdm_tension),
        'dft_tension_sigma': float(dft_tension),
        'reduction_sigma': float(reduction),
        'reduction_pct': float(reduction_pct),
        'passed': bool(passed),
    }


def test4_euclid_falsification():
    """
    Test 4: Euclid will measure S8 in ~10 redshift bins from z=0.2 to z=2.0
    with ~1% precision per bin (sigma ~ 0.008). Compute chi^2 of DFT S8(z)
    predictions vs constant LCDM S8 = 0.832.

    If chi^2/dof > 2, DFT is distinguishable from LCDM at >95% CL.

    PASS if chi^2/dof > 2.
    """
    print("\n" + "-" * 70)
    print("TEST 4: EUCLID FALSIFICATION WINDOW")
    print("-" * 70)

    clock = CascadeClock(constrained=True)
    sigma_euclid = 0.008  # ~1% per bin

    # 10 bins from z=0.2 to z=2.0
    z_bins = np.linspace(0.2, 2.0, 10)

    print(f"\n  Euclid forecast: 10 bins, z = [{z_bins[0]:.1f}, {z_bins[-1]:.1f}]")
    print(f"  Per-bin precision: sigma = {sigma_euclid}")
    print(f"  LCDM null hypothesis: S8 = {S8_PLANCK:.4f} (constant)")

    print(f"\n  {'z':>6s}  {'S8_DFT':>8s}  {'S8_LCDM':>8s}  "
          f"{'Delta':>8s}  {'(Delta/sigma)^2':>15s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*15}")

    chi2_terms = []
    s8_dft_vals = []
    for z in z_bins:
        s8_dft = clock.s8(z)
        s8_dft_vals.append(s8_dft)
        delta = s8_dft - S8_PLANCK
        chi2_i = (delta / sigma_euclid) ** 2
        chi2_terms.append(chi2_i)
        print(f"  {z:6.2f}  {s8_dft:8.4f}  {S8_PLANCK:8.4f}  "
              f"{delta:+8.4f}  {chi2_i:15.2f}")

    chi2_total = sum(chi2_terms)
    dof = len(z_bins)
    chi2_per_dof = chi2_total / dof

    print(f"\n  chi^2 = {chi2_total:.2f}")
    print(f"  dof   = {dof}")
    print(f"  chi^2/dof = {chi2_per_dof:.2f}")
    print(f"  Threshold: chi^2/dof > 2 (95% CL distinguishable)")

    if chi2_per_dof > 2:
        print(f"\n  Interpretation: DFT S8(z) curve is distinguishable from")
        print(f"  constant LCDM at >95% CL with Euclid-level precision.")
        print(f"  This is a FALSIFIABLE prediction.")
    else:
        print(f"\n  Interpretation: DFT S8(z) variation too small to distinguish")
        print(f"  from LCDM at Euclid precision. Need higher precision surveys.")

    passed = chi2_per_dof > 2
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: Euclid "
          f"{'can' if passed else 'cannot'} distinguish DFT from LCDM "
          f"(chi^2/dof = {chi2_per_dof:.2f})")

    return {
        'test': 'euclid_falsification',
        'z_bins': [float(z) for z in z_bins],
        's8_dft': [float(x) for x in s8_dft_vals],
        's8_lcdm': float(S8_PLANCK),
        'sigma_euclid': sigma_euclid,
        'chi2': float(chi2_total),
        'dof': dof,
        'chi2_per_dof': float(chi2_per_dof),
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("MILESTONE 9 - EXP 07: S8 REDSHIFT EVOLUTION")
    print("Block C: Scale-Dependent Predictions")
    print("S8 varies with z because N varies with lookback time")
    print("=" * 70)

    r1 = test1_s8_curve()
    r2 = test2_s8_matches_lensing()
    r3 = test3_tension_resolution()
    r4 = test4_euclid_falsification()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Test 1 (S8(z) curve):            {'PASS' if r1['passed'] else 'FAIL'}")
    print(f"  Test 2 (S8 at lensing z):        {'PASS' if r2['passed'] else 'FAIL'}")
    print(f"  Test 3 (Tension resolution):     {'PASS' if r3['passed'] else 'FAIL'}")
    print(f"  Test 4 (Euclid falsification):   {'PASS' if r4['passed'] else 'FAIL'}")
    print(f"\n  Score: {n_passed}/4")

    for t in tests:
        status = "PASS" if t['passed'] else "FAIL"
        print(f"  [{status}] {t['test']}")

    if r1['passed'] and r3['passed']:
        print(f"\n  KEY FINDING: Cascade clock produces scale-dependent S8(z)")
        print(f"  that reduces the Planck-lensing tension. The S8 'discrepancy'")
        print(f"  is a natural consequence of PAC cascade dissipation varying")
        print(f"  with lookback time.")

    results = {
        'experiment': 'exp_07_s8_redshift_evolution',
        'milestone': 9,
        'block': 'C',
        'block_name': 'Scale-Dependent Predictions',
        'tests': {t['test']: t for t in tests},
        'score': f'{n_passed}/4',
        'timestamp': datetime.now().isoformat(),
    }
    save_results(results, 'exp_07_s8_redshift_evolution', RESULTS_DIR)


if __name__ == '__main__':
    main()
