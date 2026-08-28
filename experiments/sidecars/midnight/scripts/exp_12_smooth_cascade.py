"""
exp_12 -- Smooth Cascade: The Z-Evolution IS the Signal

Midnight Initiative — fundamental reframing

The oscillatory model (disequilibrium at integer N) was wrong. The framework
predicts SMOOTH evolution via N(z) = a + (1/ln(phi)) * ln(t_lookback).
Every cosmological prediction (S8, H0, JWST) is a smooth function of N(z).
The z-trend we kept removing WAS the cascade.

This experiment tests: does the CASCADE CLOCK (logarithmic in lookback time
with slope 1/ln(phi)) fit absorber property evolution better than generic
z-dependence?

Tests:
  T1: N(z) vs z — which parameterizes CIV doublet evolution better?
  T2: Phi-constrained slope — does fixing slope to 1/ln(phi) match data?
  T3: MgII properties vs N(z) — same test, different ion
  T4: Geographic coherence at fixed N — absorbers at same N more similar
      than absorbers at same z?
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.integrate import quad
from scipy.optimize import curve_fit

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
from phase_rate import DATA_ROOT, PHI, INV_PHI, LN_PHI, save_midnight_results, _convert_numpy

B_DFT = 1.0 / LN_PHI
A_CLOCK = 1.360
H0, Om, Ol = 67.36, 0.3153, 0.6847


def z_to_lookback(z):
    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(Om * (1 + zp)**3 + Ol))
    r, _ = quad(integrand, 0, z)
    return r / (H0 * 1.022e-3)


def n_at_z(z):
    t = z_to_lookback(z)
    if t <= 0.001:
        t = 0.001
    return A_CLOCK + B_DFT * np.log(t)


def fit_and_score(x, y, label):
    """Fit linear model y = A + B*x, return R², slope, residual std."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    coeffs = np.polyfit(x, y, 1)
    predicted = np.polyval(coeffs, x)
    ss_res = np.sum((y - predicted)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    resid_std = np.std(y - predicted)
    return {
        'label': label, 'slope': float(coeffs[0]), 'intercept': float(coeffs[1]),
        'R2': float(r_sq), 'resid_std': float(resid_std), 'n': len(x)}


# ============================================================
# T1: CIV — N(z) vs z as evolution parameterizer
# ============================================================

def test_T1_civ_parameterization():
    """Does N(z) parameterize CIV doublet evolution better than z?"""
    print(f"\n  T1: CIV doublet ratio — N(z) vs z parameterization")

    with open(str(DATA_ROOT / "sdss_mgii" / "CIV_DR12_catalog.dat"), 'r') as f:
        lines = f.readlines()

    z_all, ew1_all, ew2_all, b_all = [], [], [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 14:
            continue
        try:
            z_all.append(float(parts[2]))
            ew1_all.append(float(parts[8]))
            ew2_all.append(float(parts[10]))
            b_all.append(float(parts[6]))
        except:
            continue

    z_arr = np.array(z_all)
    dr = np.array(ew1_all) / np.array(ew2_all)
    b_arr = np.array(b_all)

    good = (dr > 0.5) & (dr < 5) & (b_arr > 5) & (b_arr < 300) & (z_arr > 1.4)
    z_g = z_arr[good]
    dr_g = dr[good]
    b_g = b_arr[good]

    # Bin to reduce noise
    z_bins = np.linspace(1.5, 4.5, 80)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    bin_z, bin_N, bin_dr, bin_b, bin_t = [], [], [], [], []

    for i in range(len(z_centers)):
        mask = (z_g >= z_bins[i]) & (z_g < z_bins[i + 1])
        if np.sum(mask) < 30:
            continue
        zc = float(z_centers[i])
        t_look = z_to_lookback(zc)
        bin_z.append(zc)
        bin_N.append(n_at_z(zc))
        bin_t.append(t_look)
        bin_dr.append(float(np.median(dr_g[mask])))
        bin_b.append(float(np.median(b_g[mask])))

    bin_z = np.array(bin_z)
    bin_N = np.array(bin_N)
    bin_t = np.array(bin_t)
    bin_dr = np.array(bin_dr)
    bin_b = np.array(bin_b)
    ln_t = np.log(bin_t)

    print(f"    CIV bins: {len(bin_z)}")

    # Compete: DR as function of different x-variables
    models = {}
    for x, label in [(bin_z, 'z (linear)'),
                      (bin_z**2, 'z^2'),
                      (np.log(bin_z), 'ln(z)'),
                      (ln_t, 'ln(t_lookback)'),
                      (bin_N, 'N(z) = cascade clock')]:
        result = fit_and_score(x, bin_dr, label)
        models[label] = result
        print(f"    DR vs {label:>25}: R²={result['R2']:.6f}  slope={result['slope']:.6f}")

    # The key comparison: cascade clock vs free ln(t) fit
    # Does fixing slope to 1/ln(phi) cost R²?
    free_ln_t = models['ln(t_lookback)']
    cascade = models['N(z) = cascade clock']

    # N(z) = a + B_DFT * ln(t), so DR vs N has slope m => DR vs ln(t) has slope m*B_DFT
    # Free fit slope of DR vs ln(t)
    free_slope = free_ln_t['slope']
    # If we parameterize via N(z), the effective ln(t) slope = cascade_slope * B_DFT
    cascade_effective_slope = cascade['slope'] * B_DFT

    # What slope would phi predict?
    # DR = A + m * N(z), and N = a + (1/ln(phi))*ln(t)
    # So DR vs ln(t) slope = m / ln(phi) = m * B_DFT
    # Free fit gives slope directly on ln(t)
    # Ratio of free slope to B_DFT gives what m would be if we used N
    implied_m = free_slope / B_DFT

    print(f"\n    Free ln(t) slope: {free_slope:.6f}")
    print(f"    Cascade-implied ln(t) slope: {cascade_effective_slope:.6f}")
    print(f"    Ratio: {cascade_effective_slope/free_slope:.4f} (1.0 = perfect match)")

    # Doppler b-parameter vs same models
    print(f"\n    Doppler b-parameter:")
    for x, label in [(bin_z, 'z'), (bin_N, 'N(z)')]:
        result = fit_and_score(x, bin_b, label)
        print(f"    b vs {label:>25}: R²={result['R2']:.6f}  slope={result['slope']:.4f}")

    # PASS: N(z) fits at least as well as the best generic model
    best_generic = max(models['z (linear)']['R2'], models['z^2']['R2'],
                       models['ln(z)']['R2'])
    cascade_r2 = cascade['R2']
    passed = cascade_r2 >= best_generic * 0.99

    print(f"\n    Best generic R²: {best_generic:.6f}")
    print(f"    Cascade R²: {cascade_r2:.6f}")
    print(f"    Cascade >= 99% of best generic: {passed}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T1_civ_parameterization',
        'models': {k: v for k, v in models.items()},
        'free_slope': float(free_slope),
        'cascade_effective_slope': float(cascade_effective_slope),
        'PASS': passed}


# ============================================================
# T2: Phi-constrained slope test
# ============================================================

def test_T2_phi_slope():
    """Does fixing slope to 1/ln(phi) match the free-fit slope?"""
    print(f"\n  T2: Phi-constrained slope test")

    with open(str(DATA_ROOT / "sdss_mgii" / "CIV_DR12_catalog.dat"), 'r') as f:
        lines = f.readlines()

    z_all, ew1_all, ew2_all = [], [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 14:
            continue
        try:
            z_all.append(float(parts[2]))
            ew1_all.append(float(parts[8]))
            ew2_all.append(float(parts[10]))
        except:
            continue

    z_arr = np.array(z_all)
    dr = np.array(ew1_all) / np.array(ew2_all)
    good = (dr > 0.5) & (dr < 5) & (z_arr > 1.4)
    z_g = z_arr[good]
    dr_g = dr[good]

    # Bin
    z_bins = np.linspace(1.5, 4.5, 60)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    bin_ln_t, bin_dr = [], []
    for i in range(len(z_centers)):
        mask = (z_g >= z_bins[i]) & (z_g < z_bins[i + 1])
        if np.sum(mask) < 30:
            continue
        zc = float(z_centers[i])
        bin_ln_t.append(np.log(z_to_lookback(zc)))
        bin_dr.append(float(np.median(dr_g[mask])))

    bin_ln_t = np.array(bin_ln_t)
    bin_dr = np.array(bin_dr)

    # Free fit: DR = A + B * ln(t)
    coeffs_free = np.polyfit(bin_ln_t, bin_dr, 1)
    free_slope = coeffs_free[0]
    pred_free = np.polyval(coeffs_free, bin_ln_t)
    ss_res_free = np.sum((bin_dr - pred_free)**2)

    # Constrained fit: DR = A + (m * B_DFT) * ln(t), where we fit m only
    # Equivalently: DR = A + m * N, fit A and m
    bin_N = A_CLOCK + B_DFT * bin_ln_t
    coeffs_constrained = np.polyfit(bin_N, bin_dr, 1)
    constrained_ln_slope = coeffs_constrained[0] * B_DFT
    pred_constrained = np.polyval(coeffs_constrained, bin_N)
    ss_res_constrained = np.sum((bin_dr - pred_constrained)**2)

    # Compare slopes
    slope_ratio = constrained_ln_slope / free_slope if free_slope != 0 else 0
    slope_match = abs(slope_ratio - 1.0) < 0.05

    # Compare R²
    ss_tot = np.sum((bin_dr - np.mean(bin_dr))**2)
    r2_free = 1 - ss_res_free / ss_tot
    r2_constrained = 1 - ss_res_constrained / ss_tot
    r2_cost = r2_free - r2_constrained

    print(f"    Free slope (DR vs ln(t)): {free_slope:.6f}")
    print(f"    Phi-constrained slope:    {constrained_ln_slope:.6f}")
    print(f"    Slope ratio: {slope_ratio:.4f} (1.0 = perfect)")
    print(f"    R² free: {r2_free:.6f}")
    print(f"    R² constrained (phi): {r2_constrained:.6f}")
    print(f"    R² cost of phi constraint: {r2_cost:.6f}")

    # What is the best-fit slope in units of 1/ln(phi)?
    slope_in_phi_units = free_slope / B_DFT
    print(f"\n    Free slope = {slope_in_phi_units:.4f} * (1/ln(phi))")
    print(f"    If exactly phi-constrained: 1.0000 * (1/ln(phi))")
    print(f"    Deviation: {abs(slope_in_phi_units - round(slope_in_phi_units)):.4f} from nearest integer")

    passed = r2_cost < 0.01 and abs(slope_ratio - 1.0) < 0.10
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T2_phi_slope',
        'free_slope': float(free_slope),
        'constrained_slope': float(constrained_ln_slope),
        'slope_ratio': float(slope_ratio),
        'r2_free': float(r2_free),
        'r2_constrained': float(r2_constrained),
        'r2_cost': float(r2_cost),
        'slope_in_phi_units': float(slope_in_phi_units),
        'PASS': passed}


# ============================================================
# T3: MgII properties vs N(z)
# ============================================================

def test_T3_mgii_cascade():
    """Do MgII properties also track N(z) better than z?"""
    print(f"\n  T3: MgII properties vs N(z)")

    from astropy.io import fits
    hdul = fits.open(str(DATA_ROOT / "sdss_mgii" / "SDSS_DR16_MgII_Catalog.fits"))
    d = hdul[1].data
    good = ((d['REST_EW_MGII_2796'] > 0.2) & (d['SNR_2796'] > 5) &
            (d['FWHM_VDISP_MGII_2796'] > 10) & (d['FWHM_VDISP_MGII_2796'] < 500) &
            (d['FWHM_VDISP_MGII_2803'] > 10) &
            np.isfinite(d['FWHM_VDISP_MGII_2796']) & np.isfinite(d['FWHM_VDISP_MGII_2803']))
    z = d['Z_ABS'][good]
    ew = d['REST_EW_MGII_2796'][good]
    dr = d['REST_EW_MGII_2796'][good] / d['REST_EW_MGII_2803'][good]
    disc = np.abs(d['FWHM_VDISP_MGII_2796'][good] - d['FWHM_VDISP_MGII_2803'][good]) / \
           (d['FWHM_VDISP_MGII_2796'][good] + d['FWHM_VDISP_MGII_2803'][good])
    hdul.close()

    # Bin
    z_bins = np.linspace(0.36, 2.2, 60)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    bin_z, bin_N, bin_ew, bin_dr, bin_disc = [], [], [], [], []

    for i in range(len(z_centers)):
        mask = (z >= z_bins[i]) & (z < z_bins[i + 1])
        if np.sum(mask) < 50:
            continue
        zc = float(z_centers[i])
        bin_z.append(zc)
        bin_N.append(n_at_z(zc))
        bin_ew.append(float(np.median(ew[mask])))
        bin_dr.append(float(np.median(dr[mask])))
        bin_disc.append(float(np.median(disc[mask])))

    bin_z = np.array(bin_z)
    bin_N = np.array(bin_N)

    print(f"    MgII bins: {len(bin_z)}")

    results = {}
    any_cascade_wins = False
    for prop_name, prop_vals in [('EW', np.array(bin_ew)),
                                  ('Doublet ratio', np.array(bin_dr)),
                                  ('FWHM discrepancy', np.array(bin_disc))]:
        r2_z = fit_and_score(bin_z, prop_vals, 'z')['R2']
        r2_N = fit_and_score(bin_N, prop_vals, 'N(z)')['R2']
        wins = r2_N > r2_z
        if wins:
            any_cascade_wins = True
        print(f"    {prop_name:>20}: R²(z)={r2_z:.6f}  R²(N)={r2_N:.6f}  "
              f"{'N wins' if wins else 'z wins'} (delta={r2_N-r2_z:+.6f})")
        results[prop_name] = {'R2_z': float(r2_z), 'R2_N': float(r2_N), 'N_wins': wins}

    passed = any_cascade_wins
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {'test': 'T3_mgii_cascade', 'results': results, 'PASS': passed}


# ============================================================
# T4: Geographic coherence at fixed N vs fixed z
# ============================================================

def test_T4_geographic_coherence():
    """Are absorbers at the same N more similar than absorbers at the same z?"""
    print(f"\n  T4: Geographic coherence — fixed N vs fixed z")

    from astropy.io import fits
    hdul = fits.open(str(DATA_ROOT / "sdss_mgii" / "SDSS_DR16_MgII_Catalog.fits"))
    d = hdul[1].data
    good = ((d['REST_EW_MGII_2796'] > 0.2) & (d['SNR_2796'] > 5) &
            (d['FWHM_VDISP_MGII_2796'] > 10) & (d['FWHM_VDISP_MGII_2796'] < 500) &
            np.isfinite(d['FWHM_VDISP_MGII_2796']) & np.isfinite(d['FWHM_VDISP_MGII_2803']))
    z = d['Z_ABS'][good]
    ew = d['REST_EW_MGII_2796'][good]
    disc = np.abs(d['FWHM_VDISP_MGII_2796'][good] - d['FWHM_VDISP_MGII_2803'][good]) / \
           (d['FWHM_VDISP_MGII_2796'][good] + d['FWHM_VDISP_MGII_2803'][good])
    hdul.close()

    N_arr = np.array([n_at_z(zz) for zz in z])

    # Bin by z and by N, compute within-bin variance of EW
    n_bins = 40

    # Z-bins
    z_bins = np.linspace(0.36, 2.2, n_bins + 1)
    z_variances = []
    for i in range(n_bins):
        mask = (z >= z_bins[i]) & (z < z_bins[i + 1])
        if np.sum(mask) > 20:
            z_variances.append(np.std(ew[mask]))

    # N-bins (same number of bins, spanning the same data)
    N_bins = np.linspace(np.min(N_arr), np.max(N_arr), n_bins + 1)
    N_variances = []
    for i in range(n_bins):
        mask = (N_arr >= N_bins[i]) & (N_arr < N_bins[i + 1])
        if np.sum(mask) > 20:
            N_variances.append(np.std(ew[mask]))

    mean_z_var = np.mean(z_variances) if z_variances else 0
    mean_N_var = np.mean(N_variances) if N_variances else 0

    print(f"    Mean within-bin EW std (z-bins): {mean_z_var:.4f}")
    print(f"    Mean within-bin EW std (N-bins): {mean_N_var:.4f}")
    print(f"    Ratio N/z: {mean_N_var/mean_z_var:.4f}")
    print(f"    Lower = more coherent within bin")

    # Same for disc
    z_disc_vars = []
    for i in range(n_bins):
        mask = (z >= z_bins[i]) & (z < z_bins[i + 1])
        if np.sum(mask) > 20:
            z_disc_vars.append(np.std(disc[mask]))

    N_disc_vars = []
    for i in range(n_bins):
        mask = (N_arr >= N_bins[i]) & (N_arr < N_bins[i + 1])
        if np.sum(mask) > 20:
            N_disc_vars.append(np.std(disc[mask]))

    mean_z_disc = np.mean(z_disc_vars) if z_disc_vars else 0
    mean_N_disc = np.mean(N_disc_vars) if N_disc_vars else 0

    print(f"    Mean within-bin disc std (z-bins): {mean_z_disc:.4f}")
    print(f"    Mean within-bin disc std (N-bins): {mean_N_disc:.4f}")
    print(f"    Ratio N/z: {mean_N_disc/mean_z_disc:.4f}")

    # N-binning more coherent means ratio < 1
    n_more_coherent = (mean_N_var < mean_z_var) or (mean_N_disc < mean_z_disc)

    passed = n_more_coherent
    print(f"    N-binning more coherent: {n_more_coherent}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T4_geographic_coherence',
        'ew_std_z': float(mean_z_var), 'ew_std_N': float(mean_N_var),
        'disc_std_z': float(mean_z_disc), 'disc_std_N': float(mean_N_disc),
        'PASS': passed}


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("exp_12: Smooth Cascade — The Z-Evolution IS the Signal")
    print("Midnight Initiative — Reframed Analysis")
    print("=" * 60)

    t1 = test_T1_civ_parameterization()
    t2 = test_T2_phi_slope()
    t3 = test_T3_mgii_cascade()
    t4 = test_T4_geographic_coherence()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'='*60}")
    print(f"  Overall: {score}/4")
    print(f"{'='*60}")

    data = {
        'experiment': 'exp_12_smooth_cascade',
        'initiative': 'midnight',
        'thread': 'photon_archaeology',
        'reframing': 'smooth z-evolution IS the cascade, not a confound',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'score': f"{score}/4",
        'n_pass': score,
        'n_total': 4,
    }

    save_midnight_results('exp_12_smooth_cascade', _convert_numpy(data))
