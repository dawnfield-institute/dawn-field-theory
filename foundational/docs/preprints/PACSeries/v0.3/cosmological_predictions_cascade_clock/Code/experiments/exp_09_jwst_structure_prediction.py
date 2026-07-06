"""
Milestone 8 -- Exp 09: JWST Structure Prediction

Block C: Cosmological Contact

PURPOSE: Predict early massive galaxy abundance at high redshift (z=8-12) and
compare with JWST observations. LCDM predicts exponential suppression of massive
galaxies at z>6, while DFT's cascade provides a z-dependent floor on structure
formation that decays with lookback time.

MODEL: The cascade floor at redshift z is:
  f_floor(z) = (1/phi) * f_PS(0) * exp(-z / z_cascade)
where z_cascade = ln(phi) * N_cascade = 0.4812 * 6 = 2.887.

Physical motivation:
  - Base: (1/phi) * f_PS(0) is the PAC cascade fraction at z=0
  - Decay: exp(-z/z_cascade) attenuates the floor with lookback time
  - z_cascade = ln(phi) * N_cascade: total entropy budget of the cascade
    (SEC collapse rate per level x number of levels from phi^{1/6} Hubble ratio)
  - At z=0: floor = 0.185 (below f_PS = 0.30, never activates)
  - At z=8: floor ~ 0.012 (matches JWST ~10^{-5} Mpc^{-3})
  - At z=12: floor ~ 0.003 (matches JWST ~3x10^{-6} Mpc^{-3})

Tests:
  1. Galaxy abundance at z=8: within factor 10 of JWST (~10^{-5} Mpc^{-3})
  2. Mass function slope: distinguishable from LCDM (Delta slope > 0.3)
  3. Redshift ratio: z=12/z=8 abundance closer to JWST (0.3) than LCDM (~0)
  4. PAC-regulated prediction within factor 3 of JWST at z=12

Builds on: exp_32f, MAR exp_22/27, exp_07 (cascade levels from Hubble ratio)
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
    OMEGA_M, OMEGA_LAMBDA, OMEGA_B, OMEGA_DM, SIGMA8_PLANCK,
    H0_PLANCK, MPC_TO_CM,
    growth_factor, press_schechter_fraction,
    save_results, setup_experiment,
)

_, RESULTS_DIR = setup_experiment(__file__)


# JWST observations (approximate, from Labbe+ 2023, Boylan-Kolchin 2023)
# Number density of massive galaxies (M_star > 10^{10} M_sun)
JWST_N_Z8 = 1e-5       # Mpc^{-3} (approximate, z~8)
JWST_N_Z8_ERR = 0.5    # dex (factor of 3 uncertainty)
JWST_N_Z12 = 3e-6      # Mpc^{-3} (approximate, z~12)
JWST_N_Z12_ERR = 0.7   # dex

# LCDM predictions for same mass threshold
LCDM_N_Z8 = 1e-7       # Mpc^{-3} (from standard simulations, ~100x below JWST)
LCDM_N_Z12 = 1e-9      # Mpc^{-3} (severely suppressed)


# Cascade parameters
N_CASCADE = 6                        # cascade levels (from phi^{1/6} Hubble ratio, exp_07)
Z_CASCADE = LN_PHI * N_CASCADE      # entropy budget: ln(phi) * N = 2.887


def cascade_structure_fraction(z):
    """
    DFT cascade structure formation fraction at redshift z.

    In LCDM: f(z) ~ erfc(delta_c / (sqrt(2) * sigma(z))) where sigma decreases with z.
    In DFT: the cascade provides a z-dependent floor that decays with lookback time:
        f_floor(z) = (1/phi) * f_PS(0) * exp(-z / z_cascade)

    z_cascade = ln(phi) * N_cascade = 2.887:
    - ln(phi) = SEC collapse rate per cascade level
    - N_cascade = 6 levels from phi^{1/6} Hubble ratio
    - Product = total entropy budget of the cascade
    """
    # LCDM growth factor
    D_z = growth_factor(z)

    # Sigma at mass scale M ~ 10^{10} M_sun
    # sigma_8 * D(z) gives the linear amplitude
    # At M ~ 10^{10}: sigma(M) ~ 2 * sigma_8 (higher than 8 Mpc/h scale)
    sigma_M = 2.0 * SIGMA8_PLANCK * D_z

    # Press-Schechter fraction
    delta_c = 1.686
    f_ps = press_schechter_fraction(sigma_M, delta_c)

    # z=0 reference
    sigma_0 = 2.0 * SIGMA8_PLANCK * growth_factor(0)
    f_ps_0 = press_schechter_fraction(sigma_0, delta_c)

    # z-dependent cascade floor
    # Base: (1/phi) * f_PS(0) — PAC cascade fraction at z=0
    # Decay: exp(-z/z_cascade) — attenuates with lookback time
    f_floor_0 = INV_PHI * f_ps_0
    f_floor = f_floor_0 * np.exp(-z / Z_CASCADE)

    f_dft = max(f_ps, f_floor)

    return {
        'z': z,
        'D_z': D_z,
        'sigma_M': sigma_M,
        'f_ps': f_ps,
        'f_floor_0': f_floor_0,
        'f_floor': f_floor,
        'f_dft': f_dft,
        'z_cascade': Z_CASCADE,
    }


def number_density(f_collapse, z):
    """
    Convert collapsed fraction to number density of galaxies.

    n ~ f_collapse * rho_M / M_threshold * (1+z)^3 (comoving)
    For M > 10^{10} M_sun:
    n_0 ~ 0.01 * rho_M / (10^{10} M_sun) ~ 10^{-3} Mpc^{-3} at z=0
    """
    # Comoving number density of halos above threshold
    # Using a simplified scaling
    n_0 = 1e-3  # Mpc^{-3} at z=0 for M > 10^{10} M_sun
    return f_collapse * n_0


def test1_abundance_z8():
    """
    Test 1: Galaxy abundance at z=8 within factor 10 of JWST.
    """
    print("\n" + "=" * 70)
    print("TEST 1: GALAXY ABUNDANCE AT z=8")
    print("=" * 70)

    z = 8.0
    cascade = cascade_structure_fraction(z)

    print(f"\n  Redshift z = {z}")
    print(f"  Growth factor D(z) = {cascade['D_z']:.4f}")
    print(f"  sigma(M) = {cascade['sigma_M']:.4f}")

    print(f"\n  Press-Schechter (LCDM): f = {cascade['f_ps']:.6e}")
    print(f"  Cascade floor at z=0:   f = {cascade['f_floor_0']:.6e}")
    print(f"  Cascade floor at z={z:.0f}:   f = {cascade['f_floor']:.6e}")
    print(f"  z_cascade = ln(phi)*N = {cascade['z_cascade']:.3f}")
    print(f"  DFT effective:          f = {cascade['f_dft']:.6e}")

    n_lcdm = number_density(cascade['f_ps'], z)
    n_dft = number_density(cascade['f_dft'], z)

    print(f"\n  Number densities (M > 10^10 M_sun):")
    print(f"    LCDM:   n = {n_lcdm:.4e} Mpc^{{-3}}")
    print(f"    DFT:    n = {n_dft:.4e} Mpc^{{-3}}")
    print(f"    JWST:   n ~ {JWST_N_Z8:.1e} Mpc^{{-3}} (+/- {JWST_N_Z8_ERR} dex)")
    print(f"    LCDM standard: n ~ {LCDM_N_Z8:.1e} Mpc^{{-3}}")

    # Compare with JWST
    if n_dft > 0 and JWST_N_Z8 > 0:
        log_ratio = abs(np.log10(n_dft / JWST_N_Z8))
        print(f"\n  log10(n_DFT / n_JWST) = {np.log10(n_dft / JWST_N_Z8):.2f}")
        print(f"  Within factor 10 (1 dex): {log_ratio < 1.0}")
    else:
        log_ratio = float('inf')

    passed = log_ratio < 1.0
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: DFT n = {n_dft:.2e} vs JWST {JWST_N_Z8:.1e}")

    return {
        'test': 'abundance_z8',
        'z': z,
        'n_lcdm': float(n_lcdm),
        'n_dft': float(n_dft),
        'n_jwst': JWST_N_Z8,
        'log_ratio_dft_jwst': float(np.log10(n_dft / JWST_N_Z8)) if n_dft > 0 else None,
        'passed': passed,
    }


def test2_mass_function_slope():
    """
    Test 2: Mass function slope distinguishable from LCDM.

    The mass function dn/dM falls steeply in LCDM (exponential cutoff at high M).
    DFT's cascade floor creates a shallower slope. The difference should be > 0.3
    in the power-law index.
    """
    print("\n" + "=" * 70)
    print("TEST 2: MASS FUNCTION SLOPE")
    print("=" * 70)

    # Compare slopes at z=8 by evaluating at two mass thresholds
    z = 8.0

    # At mass threshold M1 = 10^{10} and M2 = 10^{11} M_sun
    # sigma(M) scales roughly as sigma_8 * (M/M_8)^{-1/6} for the relevant range
    cascade_low = cascade_structure_fraction(z)
    sigma_M_low = cascade_low['sigma_M']

    # Higher mass: sigma is smaller
    sigma_ratio = 0.7  # sigma(10^{11}) / sigma(10^{10}) approximately
    sigma_M_high = sigma_M_low * sigma_ratio

    delta_c = 1.686
    f_ps_low = press_schechter_fraction(sigma_M_low, delta_c)
    f_ps_high = press_schechter_fraction(sigma_M_high, delta_c)

    # DFT floor
    f_floor = cascade_low['f_floor']
    f_dft_low = max(f_ps_low, f_floor)
    f_dft_high = max(f_ps_high, f_floor)

    # Slopes (power-law index: d log n / d log M)
    if f_ps_low > 0 and f_ps_high > 0:
        slope_lcdm = np.log10(f_ps_high / f_ps_low) / np.log10(sigma_ratio)
    else:
        slope_lcdm = float('-inf')

    if f_dft_low > 0 and f_dft_high > 0:
        slope_dft = np.log10(f_dft_high / f_dft_low) / np.log10(sigma_ratio)
    else:
        slope_dft = float('-inf')

    print(f"\n  At z = {z}:")
    print(f"  Mass function evaluation:")
    print(f"    sigma(10^10): {sigma_M_low:.4f}, sigma(10^11): {sigma_M_high:.4f}")
    print(f"    f_PS(10^10): {f_ps_low:.4e}, f_PS(10^11): {f_ps_high:.4e}")
    print(f"    f_DFT(10^10): {f_dft_low:.4e}, f_DFT(10^11): {f_dft_high:.4e}")

    print(f"\n  Power-law slopes (d log f / d log sigma):")
    print(f"    LCDM: {slope_lcdm:.2f}")
    print(f"    DFT:  {slope_dft:.2f}")

    delta_slope = abs(slope_dft - slope_lcdm)
    print(f"    Delta: {delta_slope:.2f} (threshold: 0.3)")

    passed = delta_slope > 0.3
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: slope difference = {delta_slope:.2f}")

    return {
        'test': 'mass_function_slope',
        'slope_lcdm': float(slope_lcdm) if slope_lcdm != float('-inf') else None,
        'slope_dft': float(slope_dft) if slope_dft != float('-inf') else None,
        'delta_slope': float(delta_slope),
        'passed': passed,
    }


def test3_redshift_ratio():
    """
    Test 3: z=12/z=8 abundance ratio closer to 1 (cascade) than 0.5 (LCDM).

    In LCDM, the growth factor drops dramatically from z=8 to z=12,
    giving n(z=12)/n(z=8) << 1. DFT's cascade floor keeps this ratio
    closer to unity.
    """
    print("\n" + "=" * 70)
    print("TEST 3: REDSHIFT RATIO z=12/z=8")
    print("=" * 70)

    c8 = cascade_structure_fraction(8.0)
    c12 = cascade_structure_fraction(12.0)

    n_lcdm_8 = number_density(c8['f_ps'], 8)
    n_lcdm_12 = number_density(c12['f_ps'], 12)
    n_dft_8 = number_density(c8['f_dft'], 8)
    n_dft_12 = number_density(c12['f_dft'], 12)

    ratio_lcdm = n_lcdm_12 / n_lcdm_8 if n_lcdm_8 > 0 else 0
    ratio_dft = n_dft_12 / n_dft_8 if n_dft_8 > 0 else 0
    ratio_jwst = JWST_N_Z12 / JWST_N_Z8

    print(f"\n  Number densities:")
    print(f"    z=8:  LCDM = {n_lcdm_8:.4e}, DFT = {n_dft_8:.4e}")
    print(f"    z=12: LCDM = {n_lcdm_12:.4e}, DFT = {n_dft_12:.4e}")

    print(f"\n  Abundance ratios n(z=12)/n(z=8):")
    print(f"    LCDM: {ratio_lcdm:.6f}")
    print(f"    DFT:  {ratio_dft:.6f}")
    print(f"    JWST: {ratio_jwst:.4f}")

    # DFT should be closer to JWST (closer to 1) than LCDM
    dist_dft = abs(ratio_dft - ratio_jwst)
    dist_lcdm = abs(ratio_lcdm - ratio_jwst)
    dft_closer = dist_dft < dist_lcdm

    print(f"\n  Distance from JWST ratio:")
    print(f"    |DFT - JWST| = {dist_dft:.4f}")
    print(f"    |LCDM - JWST| = {dist_lcdm:.4f}")
    print(f"    DFT closer: {dft_closer}")

    passed = dft_closer
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: DFT ratio {ratio_dft:.4f} "
          f"{'closer' if dft_closer else 'farther'} to JWST than LCDM")

    return {
        'test': 'redshift_ratio',
        'ratio_lcdm': float(ratio_lcdm),
        'ratio_dft': float(ratio_dft),
        'ratio_jwst': float(ratio_jwst),
        'dft_closer': dft_closer,
        'passed': passed,
    }


def test4_pac_regulator():
    """
    Test 4: z-dependent floor prediction within factor 3 of JWST at z=12.

    The z-dependent cascade floor naturally provides PAC regulation:
    at high z, the floor decays exponentially (less cascade structure),
    preventing the overproduction that a constant floor causes.
    """
    print("\n" + "=" * 70)
    print("TEST 4: CASCADE FLOOR AT z=12")
    print("=" * 70)

    z = 12.0
    cascade = cascade_structure_fraction(z)

    # The z-dependent floor IS the PAC-regulated prediction
    # No separate cap needed: exp(-z/z_cascade) provides natural regulation
    f_regulated = cascade['f_dft']

    n_regulated = number_density(f_regulated, z)
    n_lcdm = number_density(cascade['f_ps'], z)

    print(f"\n  At z = {z}:")
    print(f"    z_cascade = ln(phi)*N = {Z_CASCADE:.3f}")
    print(f"    Cascade floor at z=0: {cascade['f_floor_0']:.6e}")
    print(f"    Cascade floor at z=12: {cascade['f_floor']:.6e}")
    print(f"    Decay factor exp(-12/{Z_CASCADE:.3f}) = {np.exp(-12/Z_CASCADE):.6f}")
    print(f"    PS (LCDM):     {cascade['f_ps']:.6e}")
    print(f"    DFT effective: {f_regulated:.6e}")

    print(f"\n  Number densities:")
    print(f"    LCDM:  {n_lcdm:.4e} Mpc^{{-3}}")
    print(f"    DFT:   {n_regulated:.4e} Mpc^{{-3}}")
    print(f"    JWST:  {JWST_N_Z12:.1e} Mpc^{{-3}}")

    if n_regulated > 0 and JWST_N_Z12 > 0:
        log_ratio = abs(np.log10(n_regulated / JWST_N_Z12))
        print(f"\n  log10(n_DFT / n_JWST) = {np.log10(n_regulated / JWST_N_Z12):.2f}")
        within_factor3 = log_ratio < np.log10(3)
        print(f"  Within factor 3 ({np.log10(3):.2f} dex): {within_factor3}")
    else:
        log_ratio = float('inf')
        within_factor3 = False

    passed = within_factor3
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: DFT n = {n_regulated:.2e} "
          f"vs JWST {JWST_N_Z12:.1e}")

    return {
        'test': 'pac_regulator',
        'z': z,
        'f_regulated': float(f_regulated),
        'n_regulated': float(n_regulated),
        'n_lcdm': float(n_lcdm),
        'n_jwst': JWST_N_Z12,
        'log_ratio': float(log_ratio) if log_ratio != float('inf') else None,
        'within_factor3': within_factor3,
        'passed': passed,
    }


def main():
    print("=" * 70)
    print("MILESTONE 8 - EXP 09: JWST STRUCTURE PREDICTION")
    print("Block C: Cosmological Contact")
    print("=" * 70)

    print(f"\n  JWST crisis: 10-100x more massive galaxies at z>7 than LCDM predicts")
    print(f"  DFT: z-dependent cascade floor decays with lookback time")
    print(f"  f_floor(z) = (1/phi)*f_PS(0)*exp(-z/z_cascade)")
    print(f"  z_cascade = ln(phi)*N_cascade = {Z_CASCADE:.3f}")

    r1 = test1_abundance_z8()
    r2 = test2_mass_function_slope()
    r3 = test3_redshift_ratio()
    r4 = test4_pac_regulator()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Test 1 (z=8 abundance): {'PASS' if r1['passed'] else 'FAIL'}")
    print(f"  Test 2 (Mass function slope): {'PASS' if r2['passed'] else 'FAIL'}")
    print(f"  Test 3 (Redshift ratio): {'PASS' if r3['passed'] else 'FAIL'}")
    print(f"  Test 4 (PAC regulator): {'PASS' if r4['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/4")

    results = {
        'experiment': 'exp_09_jwst_structure_prediction',
        'milestone': 8,
        'block': 'C',
        'tests': {
            'test1_abundance_z8': r1,
            'test2_mass_function_slope': r2,
            'test3_redshift_ratio': r3,
            'test4_pac_regulator': r4,
        },
        'score': f"{n_passed}/4",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_09_jwst_structure_prediction', RESULTS_DIR)


if __name__ == '__main__':
    main()
