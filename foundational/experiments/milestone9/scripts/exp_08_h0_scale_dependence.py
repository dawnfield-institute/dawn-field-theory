"""
Milestone 9 -- Exp 08: H0 Scale Dependence

PURPOSE: The Hubble tension (H0 = 67.4 vs 73.0 km/s/Mpc) arises from two
distinct DFT mechanisms:
  1. DISCRETE cascade correction: H0_local = H0_Planck * phi^{1/N_floor},
     where N_floor = floor(N_max). This is the M8 result — the tension between
     Planck and SH0ES comes from the discrete step, not the continuous clock.
  2. CONTINUOUS cascade clock: H0(z) = H0_Planck * phi^{1/N_physical(z)},
     which gives scale-dependent variation between BAO probes at different z.

With N_physical, both SH0ES (z=0.01) and Planck (z=1100) return N_max
(current epoch value), so the continuous formula predicts NO tension between
them. The tension is purely from the discrete floor correction.

Block C: Scale-Dependent Predictions

Tests:
  1. H0 vs probe lookback: predicted spread >= 2 km/s/Mpc (using clock.h0(z))
  2. M8 compatibility: discrete phi^{1/N_floor} matches SH0ES, continuous
     clock gives correct z-ordering between DESI bins
  3. BAO scale dependence: H0_DFT decreases monotonically with z
  4. TDSL prediction: H0 at z~0.5 matches HOLiCOW within 2 sigma
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


def test1_h0_vs_probe_lookback():
    """
    Test 1: For each probe in H0_PROBES, compute the DFT-predicted H0 using
    clock.h0(z) which calls N_physical internally.

    PASS if predicted spread (max - min across all probes) >= 2 km/s/Mpc.
    """
    print("\n" + "-" * 70)
    print("TEST 1: H0 VS PROBE LOOKBACK TIME")
    print("-" * 70)

    clock = CascadeClock(constrained=True)

    print(f"\n  Cascade clock: a = {clock.a:.4f}, slope = {clock.slope:.4f}")
    print(f"  N_max = {clock.n_max:.4f}, N_floor = {clock.n_floor}")
    print(f"  H0_PLANCK = {H0_PLANCK:.2f} km/s/Mpc")
    print(f"  Formula: H0(z) = H0_Planck * phi^{{1/N_physical(z)}}")

    print(f"\n  {'Probe':>8s}  {'z_eff':>8s}  {'N_phys':>8s}  "
          f"{'H0_pred':>8s}  {'H0_obs':>7s}  {'err':>5s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*5}")

    h0_predicted = []
    probe_data = []
    for name, data in H0_PROBES.items():
        z_eff = data['z_eff']
        h0_pred = clock.h0(z_eff)
        n_phys = N_physical(z_eff, clock.a)

        h0_predicted.append(h0_pred)
        probe_data.append({
            'probe': name,
            'z_eff': z_eff,
            'N_physical': float(n_phys),
            'h0_pred': float(h0_pred),
            'h0_obs': data['H0'],
            'h0_err': data['err'],
        })
        print(f"  {name:>8s}  {z_eff:8.3f}  {n_phys:8.4f}  "
              f"{h0_pred:8.2f}  {data['H0']:7.2f}  {data['err']:5.2f}")

    h0_arr = np.array(h0_predicted)
    spread = np.max(h0_arr) - np.min(h0_arr)

    print(f"\n  Predicted H0 range: {np.min(h0_arr):.2f} -- {np.max(h0_arr):.2f}")
    print(f"  Spread: {spread:.2f} km/s/Mpc")
    print(f"  Threshold: >= 2 km/s/Mpc")

    passed = spread >= 2.0
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: H0 spread "
          f"{'exceeds' if passed else 'below'} 2 km/s/Mpc threshold")

    return {
        'test': 'h0_vs_probe_lookback',
        'probes': probe_data,
        'h0_range_min': float(np.min(h0_arr)),
        'h0_range_max': float(np.max(h0_arr)),
        'spread': float(spread),
        'passed': bool(passed),
    }


def test2_early_vs_late():
    """
    Test 2: M8 Compatibility — LEAVE-ONE-OUT for H0.

    HARDENED: Round 1. Previously the clock was fitted to all 3 data points
    (including Hubble), then used to predict H0 — circular. Now:
      (a) Leave-one-out: fit clock to S8+JWST only, predict H0 blind
      (b) Continuous clock z-ordering (uses full clock, structural test)

    PASS if:
      (a) Blind H0 prediction within 2 sigma of SH0ES
      (b) Continuous H0(z=0.15) > H0(z=1.48)
    """
    print("\n" + "-" * 70)
    print("TEST 2: M8 COMPATIBILITY — LEAVE-ONE-OUT H0")
    print("  HARDENED: fit to S8+JWST only, predict H0 blind")
    print("-" * 70)

    clock_full = CascadeClock(constrained=True)

    # --- Part (a): Leave-one-out H0 prediction ---
    # Fit clock to S8 + JWST only (drop Hubble)
    t_look_train = np.array([
        N_DATA['s8']['t_lookback_gyr'],
        N_DATA['jwst']['t_lookback_gyr'],
    ])
    n_obs_train = np.array([
        N_DATA['s8']['N'],
        N_DATA['jwst']['N'],
    ])
    a_loo = np.mean(n_obs_train - B_DFT * np.log(t_look_train))

    # Predict N at Hubble lookback time
    t_look_hubble = N_DATA['hubble']['t_lookback_gyr']
    n_pred_hubble = cascade_clock(t_look_hubble, a_loo, B_DFT)
    n_obs_hubble = N_DATA['hubble']['N']

    # Predict H0 using blind clock
    n_max_loo = cascade_clock(T_UNIVERSE, a_loo, B_DFT)
    n_floor_loo = int(np.floor(n_max_loo))
    h0_blind = H0_PLANCK * PHI ** (1.0 / n_floor_loo)
    h0_shoes_obs = H0_SHOES
    sigma_shoes = H0_PROBES['shoes']['err']

    # Full clock prediction (for comparison)
    h0_full = H0_PLANCK * PHI ** (1.0 / clock_full.n_floor)

    diff_blind = abs(h0_blind - h0_shoes_obs)
    n_sigma_blind = diff_blind / sigma_shoes
    diff_full = abs(h0_full - h0_shoes_obs)
    n_sigma_full = diff_full / sigma_shoes

    print(f"\n  Part (a): Leave-one-out H0 prediction")
    print(f"    Full clock: a = {clock_full.a:.4f}, N_max = {clock_full.n_max:.4f}")
    print(f"    LOO clock:  a = {a_loo:.4f}, N_max = {n_max_loo:.4f}")
    print(f"    LOO predicted N at Hubble lookback: {n_pred_hubble:.2f} (obs: {n_obs_hubble:.2f})")
    print(f"    N_floor: LOO = {n_floor_loo}, full = {clock_full.n_floor}")
    print(f"\n    H0 predictions:")
    print(f"      Full fit:  {h0_full:.2f} km/s/Mpc ({n_sigma_full:.2f} sigma from SH0ES)")
    print(f"      Blind LOO: {h0_blind:.2f} km/s/Mpc ({n_sigma_blind:.2f} sigma from SH0ES)")
    print(f"      SH0ES obs: {h0_shoes_obs:.2f} +/- {sigma_shoes:.2f}")
    print(f"    Threshold: < 2 sigma ({2*sigma_shoes:.2f} km/s/Mpc)")

    passed_a = diff_blind < 2 * sigma_shoes
    if not passed_a:
        print(f"\n    HONEST FAILURE: blind LOO prediction {n_sigma_blind:.2f} sigma from SH0ES.")
        print(f"    The Hubble data point was needed to constrain the clock intercept.")
    print(f"    -> {'PASS' if passed_a else 'FAIL'}")

    # --- Part (b): Continuous clock z-ordering (structural, uses full clock) ---
    z_low = DESI_Z_EFF[0]   # 0.15
    z_high = DESI_Z_EFF[-1]  # 1.48
    h0_low_z = clock_full.h0(z_low)
    h0_high_z = clock_full.h0(z_high)

    print(f"\n  Part (b): Continuous clock z-ordering (structural)")
    print(f"    H0(z={z_low}) = {h0_low_z:.4f} km/s/Mpc")
    print(f"    H0(z={z_high}) = {h0_high_z:.4f} km/s/Mpc")
    print(f"    H0(z={z_low}) > H0(z={z_high}): {h0_low_z > h0_high_z}")

    passed_b = h0_low_z > h0_high_z
    print(f"    -> {'PASS' if passed_b else 'FAIL'}")

    # --- Overall ---
    passed = passed_a and passed_b

    print(f"\n  Overall: {'PASS' if passed else 'FAIL'} "
          f"(a={'PASS' if passed_a else 'FAIL'}, "
          f"b={'PASS' if passed_b else 'FAIL'})")

    return {
        'test': 'early_vs_late',
        'hardened': 'Round 1: leave-one-out (S8+JWST only, H0 blind)',
        'a_full': float(clock_full.a),
        'a_loo': float(a_loo),
        'N_max_full': float(clock_full.n_max),
        'N_max_loo': float(n_max_loo),
        'N_floor_full': clock_full.n_floor,
        'N_floor_loo': n_floor_loo,
        'n_pred_hubble': float(n_pred_hubble),
        'n_obs_hubble': float(n_obs_hubble),
        'h0_full': float(h0_full),
        'h0_blind': float(h0_blind),
        'h0_shoes_obs': float(h0_shoes_obs),
        'sigma_shoes': float(sigma_shoes),
        'diff_blind': float(diff_blind),
        'n_sigma_blind': float(n_sigma_blind),
        'passed_a': bool(passed_a),
        'z_low': z_low,
        'z_high': z_high,
        'h0_low_z': float(h0_low_z),
        'h0_high_z': float(h0_high_z),
        'passed_b': bool(passed_b),
        'passed': bool(passed),
    }


def test3_bao_scale_dependence():
    """
    Test 3: At each DESI effective redshift, compute H0_DFT(z) using
    clock.h0(z). LCDM predicts constant H0 at all z. The DFT cascade clock
    predicts H0 decreasing with increasing z (higher z = more lookback =
    higher N = smaller phi^{1/N} correction).

    PASS if H0_DFT is monotonically decreasing with z across all 5 DESI bins.
    """
    print("\n" + "-" * 70)
    print("TEST 3: BAO SCALE DEPENDENCE")
    print("-" * 70)

    clock = CascadeClock(constrained=True)

    print(f"\n  DESI effective redshifts: {DESI_Z_EFF}")
    print(f"  LCDM: H0 = {H0_PLANCK:.2f} (constant at all z)")
    print(f"  DFT:  H0(z) = H0_Planck * phi^{{1/N_physical(z)}}")

    print(f"\n  {'z':>6s}  {'N_phys':>8s}  {'H0_DFT':>8s}  {'Delta':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")

    h0_dft_vals = []
    z_data = []
    for z in DESI_Z_EFF:
        h0_dft = clock.h0(z)
        n_phys = N_physical(z, clock.a)

        h0_dft_vals.append(h0_dft)
        delta = h0_dft - H0_PLANCK
        z_data.append({
            'z': z,
            'N_physical': float(n_phys),
            'h0_dft': float(h0_dft),
            'delta': float(delta),
        })
        print(f"  {z:6.2f}  {n_phys:8.4f}  {h0_dft:8.2f}  {delta:+8.2f}")

    # Check monotonic decrease with z
    is_monotonic = True
    for i in range(len(h0_dft_vals) - 1):
        if h0_dft_vals[i + 1] >= h0_dft_vals[i]:
            is_monotonic = False
            break

    print(f"\n  H0_DFT values: {', '.join(f'{x:.2f}' for x in h0_dft_vals)}")
    print(f"  Monotonically decreasing with z: {is_monotonic}")

    if is_monotonic:
        print(f"\n  Interpretation: Higher z probes see through more cascade levels.")
        print(f"  With larger N, the phi^{{1/N}} correction shrinks, bringing H0")
        print(f"  closer to the Planck baseline. This is the cascade clock signature.")
    else:
        print(f"\n  Interpretation: Non-monotonic H0(z) -- unexpected. Check N_physical")
        print(f"  behavior at these redshifts.")

    passed = is_monotonic
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: H0_DFT "
          f"{'is' if passed else 'is not'} monotonically decreasing with z")

    return {
        'test': 'bao_scale_dependence',
        'desi_z': DESI_Z_EFF,
        'h0_dft': [float(x) for x in h0_dft_vals],
        'z_data': z_data,
        'monotonic': bool(is_monotonic),
        'passed': bool(passed),
    }


def test4_tdsl_prediction():
    """
    Test 4: Time-delay strong lensing (HOLiCOW/TDCOSMO) probes z ~ 0.5.
    Predict H0 at this redshift using clock.h0(z_eff) and compare to the
    observed H0 = 73.3 +/- 1.7 km/s/Mpc.

    PASS if |H0_pred - 73.3| < 2 * 1.7 = 3.4.
    """
    print("\n" + "-" * 70)
    print("TEST 4: TDSL (HOLiCOW/TDCOSMO) PREDICTION")
    print("-" * 70)

    clock = CascadeClock(constrained=True)

    # TDSL probe parameters
    z_eff_tdsl = H0_PROBES['tdsl']['z_eff']   # 0.5
    h0_obs_tdsl = H0_PROBES['tdsl']['H0']     # 73.3
    h0_err_tdsl = H0_PROBES['tdsl']['err']     # 1.7

    h0_pred = clock.h0(z_eff_tdsl)
    n_phys = N_physical(z_eff_tdsl, clock.a)

    diff = abs(h0_pred - h0_obs_tdsl)
    n_sigma = diff / h0_err_tdsl
    threshold = 2 * h0_err_tdsl  # 3.4

    print(f"\n  TDSL (HOLiCOW/TDCOSMO) probe:")
    print(f"    Effective redshift: z = {z_eff_tdsl}")
    print(f"    H0_observed: {h0_obs_tdsl:.1f} +/- {h0_err_tdsl:.1f} km/s/Mpc")

    print(f"\n  DFT prediction (via clock.h0(z)):")
    print(f"    N_physical(z={z_eff_tdsl}) = {n_phys:.4f}")
    print(f"    phi^(1/N) = {PHI**(1.0/n_phys):.6f}")
    print(f"    H0_pred = {H0_PLANCK:.2f} * {PHI**(1.0/n_phys):.6f} = {h0_pred:.2f} km/s/Mpc")

    print(f"\n  Comparison:")
    print(f"    |H0_pred - H0_obs| = {diff:.2f} km/s/Mpc")
    print(f"    Deviation = {n_sigma:.2f} sigma")
    print(f"    Threshold: < 2 sigma ({threshold:.1f} km/s/Mpc)")

    passed = diff < threshold
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: TDSL prediction "
          f"{'matches' if passed else 'does not match'} observation "
          f"({n_sigma:.2f} sigma)")

    return {
        'test': 'tdsl_prediction',
        'z_eff': z_eff_tdsl,
        'N_physical': float(n_phys),
        'h0_pred': float(h0_pred),
        'h0_obs': float(h0_obs_tdsl),
        'h0_err': float(h0_err_tdsl),
        'diff': float(diff),
        'n_sigma': float(n_sigma),
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("MILESTONE 9 - EXP 08: H0 SCALE DEPENDENCE")
    print("Block C: Scale-Dependent Predictions")
    print("Hubble tension as cascade clock artifact")
    print("=" * 70)

    r1 = test1_h0_vs_probe_lookback()
    r2 = test2_early_vs_late()
    r3 = test3_bao_scale_dependence()
    r4 = test4_tdsl_prediction()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Test 1 (H0 vs probe lookback):   {'PASS' if r1['passed'] else 'FAIL'}")
    print(f"  Test 2 (M8 discrete + ordering): {'PASS' if r2['passed'] else 'FAIL'}")
    print(f"  Test 3 (BAO scale dependence):    {'PASS' if r3['passed'] else 'FAIL'}")
    print(f"  Test 4 (TDSL prediction):         {'PASS' if r4['passed'] else 'FAIL'}")
    print(f"\n  Score: {n_passed}/4")

    for t in tests:
        status = "PASS" if t['passed'] else "FAIL"
        print(f"  [{status}] {t['test']}")

    if r1['passed'] and r2['passed']:
        print(f"\n  KEY FINDING: The Hubble tension has two DFT mechanisms:")
        print(f"    1. DISCRETE: phi^{{1/N_floor}} gives H0_local matching SH0ES")
        print(f"    2. CONTINUOUS: N_physical(z) gives scale-dependent H0(z) for BAO")
        print(f"  Both SH0ES and Planck see the same N_max (current epoch),")
        print(f"  so the continuous clock correctly predicts no continuous tension.")

    results = {
        'experiment': 'exp_08_h0_scale_dependence',
        'milestone': 9,
        'block': 'C',
        'block_name': 'Scale-Dependent Predictions',
        'tests': {t['test']: t for t in tests},
        'score': f'{n_passed}/4',
        'timestamp': datetime.now().isoformat(),
    }
    save_results(results, 'exp_08_h0_scale_dependence', RESULTS_DIR)


if __name__ == '__main__':
    main()
