"""
exp_08 -- Cascade Signal Panel: Z-Trend-Immune Tests

Midnight Initiative, Thread 1 (Photon Archaeology)

Exp_07 showed that smooth z-trends confound most cascade signals. This panel
tests channels that a smooth z-trend CANNOT mimic:

  A: Periodicity in N-space (power spectrum at cascade frequency)
  B: Sharp excess at transition redshifts (narrow windows, not smooth correlation)
  C: Multi-absorber sightlines (inter-absorber correlations straddling transitions)
  D: Doublet ratio distribution shape at transitions vs troughs (narrow windows)

Every test is inherently immune to smooth z-trends by construction.

Data: SDSS DR16 MgII (159K), CIV DR12 (446K)
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, ks_2samp, mannwhitneyu
from scipy.integrate import quad
from collections import defaultdict

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = MIDNIGHT_ROOT.parent.parent.parent.parent / "data"

sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
from phase_rate import (
    PHI, INV_PHI, LN_PHI,
    save_midnight_results, _convert_numpy,
)

B_DFT = 1.0 / LN_PHI
A_CLOCK = 1.360
H0, OMEGA_M, OMEGA_LAMBDA = 67.36, 0.3153, 0.6847


def z_to_lookback(z):
    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(OMEGA_M * (1 + zp)**3 + OMEGA_LAMBDA))
    r, _ = quad(integrand, 0, z)
    return r / (H0 * 1.022e-3)


def n_at_z(z):
    t = z_to_lookback(z)
    if t <= 0.001: t = 0.001
    return max(A_CLOCK + B_DFT * np.log(t), 1.0)


def diseq_at_z(z):
    N = n_at_z(z)
    return max(0.0, 1.0 - 2.0 * abs(N - round(N)))


def load_mgii():
    from astropy.io import fits
    path = DATA_ROOT / "sdss_mgii" / "SDSS_DR16_MgII_Catalog.fits"
    hdul = fits.open(str(path))
    d = hdul[1].data
    good = ((d['FWHM_VDISP_MGII_2796'] > 10) & (d['FWHM_VDISP_MGII_2796'] < 500) &
            (d['REST_EW_MGII_2796'] > 0.2) & (d['SNR_2796'] > 5) &
            (d['FWHM_VDISP_MGII_2803'] > 10) & (d['REST_EW_MGII_2803'] > 0.1) &
            np.isfinite(d['FWHM_VDISP_MGII_2796']) & np.isfinite(d['FWHM_VDISP_MGII_2803']))
    result = {
        'z': d['Z_ABS'][good], 'ew1': d['REST_EW_MGII_2796'][good],
        'ew2': d['REST_EW_MGII_2803'][good], 'fw1': d['FWHM_VDISP_MGII_2796'][good],
        'fw2': d['FWHM_VDISP_MGII_2803'][good],
        'plate': d['PLATE'][good], 'mjd': d['MJD'][good], 'fiber': d['FIBER_ID'][good],
    }
    hdul.close()
    return result


# ============================================================
# Panel A: Periodicity in N-space
# ============================================================

def test_panel_A_periodicity(mgii):
    """A: Power spectrum of EW variance in N-space — peak at period=1?"""
    print("\n  Panel A: Periodicity in N-space")

    # Map each absorber to N-space
    z_arr = mgii['z']
    ew_arr = mgii['ew1']

    # Bin in N-space (uniform bins in N, NOT in z)
    N_values = np.array([n_at_z(z) for z in z_arr])
    N_min, N_max = np.min(N_values), np.max(N_values)
    n_bins = 80
    N_bins = np.linspace(N_min, N_max, n_bins + 1)
    N_centers = (N_bins[:-1] + N_bins[1:]) / 2

    ew_stds = []
    valid_N = []
    for i in range(len(N_centers)):
        mask = (N_values >= N_bins[i]) & (N_values < N_bins[i + 1])
        if np.sum(mask) >= 30:
            ew_stds.append(np.std(ew_arr[mask]))
            valid_N.append(N_centers[i])

    ew_stds = np.array(ew_stds)
    valid_N = np.array(valid_N)

    if len(valid_N) < 10:
        print("    Insufficient bins in N-space")
        return {'test': 'panel_A_periodicity', 'PASS': False}

    # Detrend in N-space (remove smooth N-dependence)
    coeffs = np.polyfit(valid_N, ew_stds, 2)
    detrended = ew_stds - np.polyval(coeffs, valid_N)

    # Lomb-Scargle periodogram
    from scipy.signal import lombscargle
    frequencies = np.linspace(0.3, 5.0, 500)  # periods from 0.2 to 3.3 in N-space
    angular_freqs = 2 * np.pi * frequencies
    power = lombscargle(valid_N, detrended, angular_freqs, normalize=True)

    # Find peak
    peak_idx = np.argmax(power)
    peak_freq = frequencies[peak_idx]
    peak_period = 1.0 / peak_freq
    peak_power = power[peak_idx]

    # Is the peak near period=1 (the cascade period)?
    near_cascade = abs(peak_period - 1.0) < 0.2

    # Significance: compare to shuffled
    rng = np.random.RandomState(42)
    n_shuffle = 500
    shuffle_peaks = []
    for _ in range(n_shuffle):
        shuffled = rng.permutation(detrended)
        shuf_power = lombscargle(valid_N, shuffled, angular_freqs, normalize=True)
        shuffle_peaks.append(np.max(shuf_power))
    percentile = np.mean(np.array(shuffle_peaks) < peak_power) * 100

    above_95 = percentile > 95.0

    print(f"    N-space bins: {len(valid_N)} (N range: {valid_N[0]:.2f} to {valid_N[-1]:.2f})")
    print(f"    Peak period: {peak_period:.3f} (cascade period = 1.0)")
    print(f"    Near cascade period: {near_cascade}")
    print(f"    Peak power: {peak_power:.4f}")
    print(f"    Percentile vs shuffled: {percentile:.1f}% (>95%: {above_95})")

    passed = near_cascade and above_95
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'panel_A_periodicity',
        'peak_period': float(peak_period),
        'peak_power': float(peak_power),
        'near_cascade': near_cascade,
        'shuffle_percentile': float(percentile),
        'PASS': passed,
    }


# ============================================================
# Panel B: Sharp excess at transition redshifts
# ============================================================

def test_panel_B_sharp_excess(mgii):
    """B: Do narrow z-windows at transitions show excess EW variance vs adjacent?"""
    print("\n  Panel B: Sharp excess at transition redshifts")

    z_arr = mgii['z']
    ew_arr = mgii['ew1']
    fw1_arr = mgii['fw1']
    fw2_arr = mgii['fw2']

    transition_z = [0.302, 0.579, 1.416]
    trough_z = [0.412, 0.857]
    dz = 0.04  # narrow window half-width

    trans_vars = []
    trough_vars = []
    control_vars = []

    print(f"    Window half-width: dz={dz}")

    for zt in transition_z:
        mask = (z_arr >= zt - dz) & (z_arr < zt + dz)
        n = np.sum(mask)
        if n >= 30:
            v = np.std(ew_arr[mask])
            disc = np.median(np.abs(fw1_arr[mask] - fw2_arr[mask]) / (fw1_arr[mask] + fw2_arr[mask]))
            trans_vars.append(v)
            print(f"    Transition z={zt:.3f}: EW std={v:.4f}, disc={disc:.4f} (n={n})")

    for zt in trough_z:
        mask = (z_arr >= zt - dz) & (z_arr < zt + dz)
        n = np.sum(mask)
        if n >= 30:
            v = np.std(ew_arr[mask])
            disc = np.median(np.abs(fw1_arr[mask] - fw2_arr[mask]) / (fw1_arr[mask] + fw2_arr[mask]))
            trough_vars.append(v)
            print(f"    Trough z={zt:.3f}:     EW std={v:.4f}, disc={disc:.4f} (n={n})")

    # Control: random z-values in the same range
    rng = np.random.RandomState(42)
    for _ in range(10):
        zc = rng.uniform(0.36, 2.0)
        mask = (z_arr >= zc - dz) & (z_arr < zc + dz)
        n = np.sum(mask)
        if n >= 30:
            control_vars.append(np.std(ew_arr[mask]))

    if not trans_vars or not trough_vars:
        print("    Insufficient data in windows")
        return {'test': 'panel_B_sharp_excess', 'PASS': False}

    mean_trans = np.mean(trans_vars)
    mean_trough = np.mean(trough_vars)
    mean_control = np.mean(control_vars) if control_vars else 0

    trans_above_trough = mean_trans > mean_trough
    trans_above_control = mean_trans > mean_control

    print(f"    Mean EW std — transitions: {mean_trans:.4f}")
    print(f"    Mean EW std — troughs: {mean_trough:.4f}")
    print(f"    Mean EW std — random controls: {mean_control:.4f}")
    print(f"    Transitions > troughs: {trans_above_trough}")
    print(f"    Transitions > controls: {trans_above_control}")

    passed = trans_above_trough and trans_above_control
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'panel_B_sharp_excess',
        'mean_trans': float(mean_trans),
        'mean_trough': float(mean_trough),
        'mean_control': float(mean_control),
        'trans_above_trough': trans_above_trough,
        'trans_above_control': trans_above_control,
        'PASS': passed,
    }


# ============================================================
# Panel C: Multi-absorber sightlines
# ============================================================

def test_panel_C_sightlines(mgii):
    """C: Do absorber pairs straddling a cascade transition differ from non-straddling?"""
    print("\n  Panel C: Multi-absorber sightlines")

    z_arr = mgii['z']
    ew_arr = mgii['ew1']
    plate = mgii['plate']
    mjd = mgii['mjd']
    fiber = mgii['fiber']

    # Group absorbers by sightline (plate-mjd-fiber)
    sightlines = defaultdict(list)
    for i in range(len(z_arr)):
        key = (int(plate[i]), int(mjd[i]), int(fiber[i]))
        sightlines[key].append(i)

    # Find sightlines with 2+ absorbers
    multi = {k: v for k, v in sightlines.items() if len(v) >= 2}
    print(f"    Sightlines with 2+ absorbers: {len(multi)}")

    # For each pair, compute EW difference and whether they straddle a transition
    transition_N = [2, 3, 4, 5, 6, 7]
    straddle_diffs = []
    non_straddle_diffs = []

    for key, indices in multi.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                N_i = n_at_z(z_arr[idx_i])
                N_j = n_at_z(z_arr[idx_j])

                # Do they straddle an integer N?
                N_lo, N_hi = min(N_i, N_j), max(N_i, N_j)
                straddles = any(N_lo < n < N_hi for n in transition_N)

                ew_diff = abs(ew_arr[idx_i] - ew_arr[idx_j])

                if straddles:
                    straddle_diffs.append(ew_diff)
                else:
                    non_straddle_diffs.append(ew_diff)

    straddle_diffs = np.array(straddle_diffs)
    non_straddle_diffs = np.array(non_straddle_diffs)

    print(f"    Straddling pairs: {len(straddle_diffs)}")
    print(f"    Non-straddling pairs: {len(non_straddle_diffs)}")

    if len(straddle_diffs) < 50 or len(non_straddle_diffs) < 50:
        print("    Insufficient pairs")
        return {'test': 'panel_C_sightlines', 'PASS': False}

    # Mann-Whitney U test: are straddling pairs more different?
    U, p_mw = mannwhitneyu(straddle_diffs, non_straddle_diffs, alternative='greater')
    ks, p_ks = ks_2samp(straddle_diffs, non_straddle_diffs)

    med_straddle = np.median(straddle_diffs)
    med_non = np.median(non_straddle_diffs)

    print(f"    Median EW diff — straddling: {med_straddle:.4f}")
    print(f"    Median EW diff — non-straddling: {med_non:.4f}")
    print(f"    Mann-Whitney (straddle > non): p={p_mw:.4f}")
    print(f"    KS test: KS={ks:.4f}, p={p_ks:.4f}")

    significant = p_mw < 0.05 or p_ks < 0.05

    passed = significant
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'panel_C_sightlines',
        'n_straddling': len(straddle_diffs),
        'n_non_straddling': len(non_straddle_diffs),
        'median_straddle': float(med_straddle),
        'median_non_straddle': float(med_non),
        'mannwhitney_p': float(p_mw),
        'ks_p': float(p_ks),
        'PASS': passed,
    }


# ============================================================
# Panel D: Doublet ratio shape at transitions vs troughs (narrow window)
# ============================================================

def test_panel_D_doublet_shape(mgii):
    """D: Doublet ratio distribution SHAPE differs at transitions vs troughs (narrow z-windows)."""
    print("\n  Panel D: Doublet ratio distribution shape at transitions vs troughs")

    z_arr = mgii['z']
    ew1 = mgii['ew1']
    ew2 = mgii['ew2']
    fw1 = mgii['fw1']
    fw2 = mgii['fw2']

    DR = ew1 / ew2
    disc = np.abs(fw1 - fw2) / (fw1 + fw2)

    # Collect absorbers in narrow windows around transitions and troughs
    dz = 0.06
    trans_z_list = [0.302, 0.579, 1.416]
    trough_z_list = [0.412, 0.857]

    trans_DR = []
    trans_disc = []
    for zt in trans_z_list:
        mask = (z_arr >= zt - dz) & (z_arr < zt + dz)
        trans_DR.extend(DR[mask])
        trans_disc.extend(disc[mask])

    trough_DR = []
    trough_disc = []
    for zt in trough_z_list:
        mask = (z_arr >= zt - dz) & (z_arr < zt + dz)
        trough_DR.extend(DR[mask])
        trough_disc.extend(disc[mask])

    trans_DR = np.array(trans_DR)
    trough_DR = np.array(trough_DR)
    trans_disc = np.array(trans_disc)
    trough_disc = np.array(trough_disc)

    print(f"    Transition absorbers (dz={dz} windows): {len(trans_DR)}")
    print(f"    Trough absorbers: {len(trough_DR)}")

    if len(trans_DR) < 50 or len(trough_DR) < 50:
        print("    Insufficient data")
        return {'test': 'panel_D_doublet_shape', 'PASS': False}

    # KS test on doublet ratio
    ks_dr, p_dr = ks_2samp(trans_DR, trough_DR)

    # KS test on FWHM discrepancy
    ks_disc, p_disc = ks_2samp(trans_disc, trough_disc)

    # Distribution summaries
    from scipy.stats import kurtosis, skew
    print(f"    Doublet ratio — trans: med={np.median(trans_DR):.3f}, kurt={kurtosis(trans_DR):.2f}")
    print(f"    Doublet ratio — trough: med={np.median(trough_DR):.3f}, kurt={kurtosis(trough_DR):.2f}")
    print(f"    FWHM disc — trans: med={np.median(trans_disc):.4f}")
    print(f"    FWHM disc — trough: med={np.median(trough_disc):.4f}")
    print(f"    KS doublet ratio: KS={ks_dr:.4f}, p={p_dr:.4f}")
    print(f"    KS FWHM disc: KS={ks_disc:.4f}, p={p_disc:.4f}")

    significant = p_dr < 0.05 or p_disc < 0.05

    passed = significant
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'panel_D_doublet_shape',
        'n_transition': len(trans_DR),
        'n_trough': len(trough_DR),
        'ks_doublet_ratio': float(ks_dr), 'p_doublet_ratio': float(p_dr),
        'ks_fwhm_disc': float(ks_disc), 'p_fwhm_disc': float(p_disc),
        'dr_median_trans': float(np.median(trans_DR)),
        'dr_median_trough': float(np.median(trough_DR)),
        'disc_median_trans': float(np.median(trans_disc)),
        'disc_median_trough': float(np.median(trough_disc)),
        'PASS': passed,
    }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("exp_08: Cascade Signal Panel — Z-Trend-Immune Tests")
    print("Midnight Initiative, Thread 1 (Photon Archaeology)")
    print("=" * 70)

    print("\n  Loading SDSS DR16 MgII catalog...")
    mgii = load_mgii()
    print(f"  Loaded {len(mgii['z'])} absorbers")

    a = test_panel_A_periodicity(mgii)
    b = test_panel_B_sharp_excess(mgii)
    c = test_panel_C_sightlines(mgii)
    d = test_panel_D_doublet_shape(mgii)

    score = sum(1 for t in [a, b, c, d] if t['PASS'])
    print(f"\n{'=' * 70}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 70}")

    data = {
        'experiment': 'exp_08_cascade_panel',
        'initiative': 'midnight',
        'thread': 'photon_archaeology',
        'test_results': {'A': a, 'B': b, 'C': c, 'D': d},
        'score': f"{score}/4",
        'n_pass': score,
        'n_total': 4,
    }

    save_midnight_results('exp_08_cascade_panel', _convert_numpy(data))
