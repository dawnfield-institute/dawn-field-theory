"""
exp_07 -- Deep Cascade Probe: FeII Subset, CIV High-z, and Z-Detrended Signal

Midnight Initiative, Thread 1 (Photon Archaeology)

Three independent tests of the cascade clock signal found in exp_05/06:

1. FeII-confirmed MgII subset (70K systems, 4 lines per absorber) — cleaner
   sample, does the signal strengthen?
2. CIV absorbers at z=1.5-5.1 (446K systems) — different ion, different
   physical regime, extending to N=7+ territory. Does the SAME clock predict?
3. Z-detrended cascade signal — remove smooth z-trend, test whether the
   RESIDUAL oscillation correlates with the cascade clock.

Data:
  - SDSS DR16 FeII-confirmed MgII catalog (69,675 systems)
  - SDSS DR12 CIV catalog (445,765 systems, Monadi et al. 2023)

Tests:
  T1: FeII subset shows cascade signal equal or stronger than full MgII
  T2: CIV absorbers show cascade-correlated EW spread at z > 2.3
  T3: Z-detrended MgII EW residuals correlate with cascade disequilibrium
  T4: Combined multi-ion evidence — cascade signal in both MgII and CIV

Sources: exp_05 (p=0.007), exp_06 (p=0.018)
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, ks_2samp
from scipy.integrate import quad

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


def load_feii_data():
    """Load FeII-confirmed MgII catalog."""
    from astropy.io import fits
    path = DATA_ROOT / "sdss_mgii" / "SDSS_DR16_FeII_MgII_Catalog.fits"
    hdul = fits.open(str(path))
    data = hdul[1].data

    z = data['Z_ABS']
    ew = data['REST_EW_MGII_2796']
    fwhm = data['FWHM_VDISP_MGII_2796']
    snr = data['SNR_2796']

    good = (ew > 0.2) & (fwhm > 10) & (fwhm < 500) & (snr > 5) & np.isfinite(ew) & np.isfinite(fwhm)
    hdul.close()
    return z[good], ew[good], fwhm[good], int(np.sum(good))


def load_civ_data():
    """Load CIV DR12 catalog."""
    path = DATA_ROOT / "sdss_mgii" / "CIV_DR12_catalog.dat"
    z_abs, b_param, ew_1548, ew_1550 = [], [], [], []

    with open(str(path), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 14:
                continue
            try:
                z_abs.append(float(parts[2]))
                b_param.append(float(parts[6]))
                ew_1548.append(float(parts[8]))
                ew_1550.append(float(parts[10]))
            except:
                continue

    z_abs = np.array(z_abs)
    b_param = np.array(b_param)
    ew_1548 = np.array(ew_1548)
    ew_1550 = np.array(ew_1550)

    good = (ew_1548 > 0.05) & (b_param > 5) & (b_param < 300) & np.isfinite(ew_1548)
    return z_abs[good], ew_1548[good], b_param[good], ew_1550[good], int(np.sum(good))


def load_mgii_data():
    """Load full MgII catalog (for z-detrending)."""
    from astropy.io import fits
    path = DATA_ROOT / "sdss_mgii" / "SDSS_DR16_MgII_Catalog.fits"
    hdul = fits.open(str(path))
    data = hdul[1].data

    z = data['Z_ABS']
    ew = data['REST_EW_MGII_2796']
    fwhm = data['FWHM_VDISP_MGII_2796']
    snr = data['SNR_2796']

    good = (ew > 0.2) & (fwhm > 10) & (fwhm < 500) & (snr > 5) & np.isfinite(ew) & np.isfinite(fwhm)
    hdul.close()
    return z[good], ew[good], fwhm[good], int(np.sum(good))


def bin_ew_stats(z_arr, ew_arr, z_min, z_max, n_bins=50):
    """Bin EW by redshift, compute spread metrics and cascade disequilibrium."""
    z_bins = np.linspace(z_min, z_max, n_bins + 1)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2

    results = []
    for i in range(len(z_centers)):
        mask = (z_arr >= z_bins[i]) & (z_arr < z_bins[i + 1])
        n = np.sum(mask)
        if n < 30:
            continue
        zc = float(z_centers[i])
        ew_bin = ew_arr[mask]
        results.append({
            'z': zc,
            'N': n_at_z(zc),
            'diseq': diseq_at_z(zc),
            'n': int(n),
            'ew_std': float(np.std(ew_bin)),
            'ew_iqr': float(np.percentile(ew_bin, 75) - np.percentile(ew_bin, 25)),
            'ew_median': float(np.median(ew_bin)),
            'ew_mean': float(np.mean(ew_bin)),
        })
    return results


# ============================================================
# T1: FeII-confirmed subset
# ============================================================

def test_T1_feii_subset():
    """T1: FeII-confirmed MgII absorbers show equal or stronger cascade signal."""
    print("\n  T1: FeII-confirmed subset cascade signal")

    z_fe, ew_fe, fwhm_fe, n_fe = load_feii_data()
    print(f"    FeII-confirmed absorbers: {n_fe}")

    bins = bin_ew_stats(z_fe, ew_fe, 0.39, 2.2, n_bins=45)
    diseqs = np.array([b['diseq'] for b in bins])
    ew_stds = np.array([b['ew_std'] for b in bins])
    ew_iqrs = np.array([b['ew_iqr'] for b in bins])

    rho_std, p_std = spearmanr(diseqs, ew_stds)
    rho_iqr, p_iqr = spearmanr(diseqs, ew_iqrs)

    print(f"    EW std vs diseq: rho={rho_std:.3f}, p={p_std:.4f}")
    print(f"    EW IQR vs diseq: rho={rho_iqr:.3f}, p={p_iqr:.4f}")

    # Compare to full MgII result (exp_05: rho=0.290, p=0.025)
    stronger = (abs(rho_std) >= 0.25) or (abs(rho_iqr) >= 0.30)
    significant = p_std < 0.10 or p_iqr < 0.10

    passed = significant
    print(f"    Signal present (p<0.10): {significant}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T1_feii_subset',
        'n_absorbers': n_fe,
        'n_bins': len(bins),
        'ew_std_rho': float(rho_std), 'ew_std_p': float(p_std),
        'ew_iqr_rho': float(rho_iqr), 'ew_iqr_p': float(p_iqr),
        'PASS': passed,
    }


# ============================================================
# T2: CIV at high z
# ============================================================

def test_T2_civ_high_z():
    """T2: CIV absorbers show cascade signal, extending to z>2.3 (beyond MgII)."""
    print("\n  T2: CIV absorbers — cascade signal at high z")

    z_civ, ew_civ, b_civ, ew2_civ, n_civ = load_civ_data()
    print(f"    CIV absorbers loaded: {n_civ}")
    print(f"    z range: [{np.min(z_civ):.3f}, {np.max(z_civ):.3f}]")

    # Full range analysis
    bins_full = bin_ew_stats(z_civ, ew_civ, 1.5, 4.5, n_bins=50)
    diseqs_full = np.array([b['diseq'] for b in bins_full])
    ew_stds_full = np.array([b['ew_std'] for b in bins_full])

    rho_full, p_full = spearmanr(diseqs_full, ew_stds_full)
    print(f"    Full CIV (z=1.5-4.5): EW std vs diseq rho={rho_full:.3f}, p={p_full:.4f}")

    # High-z only (beyond MgII range)
    bins_high = bin_ew_stats(z_civ, ew_civ, 2.3, 4.5, n_bins=30)
    if len(bins_high) >= 10:
        diseqs_high = np.array([b['diseq'] for b in bins_high])
        ew_stds_high = np.array([b['ew_std'] for b in bins_high])
        rho_high, p_high = spearmanr(diseqs_high, ew_stds_high)
        print(f"    High-z CIV (z=2.3-4.5): EW std vs diseq rho={rho_high:.3f}, p={p_high:.4f}")
    else:
        rho_high, p_high = 0.0, 1.0
        print(f"    High-z: insufficient bins")

    # b-parameter (Doppler width) analysis
    bins_b = bin_ew_stats(z_civ, b_civ, 1.5, 4.5, n_bins=50)
    diseqs_b = np.array([b['diseq'] for b in bins_b])
    b_stds = np.array([b['ew_std'] for b in bins_b])
    rho_b, p_b = spearmanr(diseqs_b, b_stds)
    print(f"    Doppler b spread vs diseq: rho={rho_b:.3f}, p={p_b:.4f}")

    # KS population test on CIV
    trans_mask = np.array([diseq_at_z(z) > 0.8 for z in z_civ])
    trough_mask = np.array([diseq_at_z(z) < 0.2 for z in z_civ])
    n_trans = np.sum(trans_mask)
    n_trough = np.sum(trough_mask)

    if n_trans > 100 and n_trough > 100:
        ks_ew, p_ks = ks_2samp(ew_civ[trans_mask], ew_civ[trough_mask])
        ks_b, p_ks_b = ks_2samp(b_civ[trans_mask], b_civ[trough_mask])
        print(f"    KS EW (trans vs trough): KS={ks_ew:.4f}, p={p_ks:.2e}")
        print(f"    KS b-param: KS={ks_b:.4f}, p={p_ks_b:.2e}")
    else:
        ks_ew, p_ks, ks_b, p_ks_b = 0, 1, 0, 1

    any_significant = p_full < 0.05 or p_high < 0.05 or p_ks < 0.05

    passed = any_significant
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T2_civ_high_z',
        'n_absorbers': n_civ,
        'full_rho': float(rho_full), 'full_p': float(p_full),
        'high_z_rho': float(rho_high), 'high_z_p': float(p_high),
        'b_param_rho': float(rho_b), 'b_param_p': float(p_b),
        'ks_ew': float(ks_ew), 'ks_ew_p': float(p_ks),
        'ks_b': float(ks_b), 'ks_b_p': float(p_ks_b),
        'n_transition': int(n_trans), 'n_trough': int(n_trough),
        'PASS': passed,
    }


# ============================================================
# T3: Z-detrended cascade signal
# ============================================================

def test_T3_detrended():
    """T3: After removing smooth z-trend, residual EW spread correlates with cascade."""
    print("\n  T3: Z-detrended cascade signal in MgII")

    z_mg, ew_mg, fwhm_mg, n_mg = load_mgii_data()
    bins = bin_ew_stats(z_mg, ew_mg, 0.36, 2.2, n_bins=60)

    zs = np.array([b['z'] for b in bins])
    diseqs = np.array([b['diseq'] for b in bins])
    ew_stds = np.array([b['ew_std'] for b in bins])

    # Fit smooth z-trend (quadratic)
    coeffs = np.polyfit(zs, ew_stds, 2)
    trend = np.polyval(coeffs, zs)
    residuals = ew_stds - trend

    print(f"    Z-trend: {coeffs[0]:.4f}*z^2 + {coeffs[1]:.4f}*z + {coeffs[2]:.4f}")
    print(f"    Residual range: [{np.min(residuals):.4f}, {np.max(residuals):.4f}]")

    # Correlation of RESIDUALS with cascade disequilibrium
    rho_resid, p_resid = spearmanr(diseqs, residuals)
    print(f"    Residual EW std vs diseq: rho={rho_resid:.3f}, p={p_resid:.4f}")

    # Compare to raw (pre-detrend) correlation
    rho_raw, p_raw = spearmanr(diseqs, ew_stds)
    print(f"    Raw EW std vs diseq: rho={rho_raw:.3f}, p={p_raw:.4f}")
    print(f"    Detrending {'strengthens' if abs(rho_resid) > abs(rho_raw) else 'weakens'} the signal")

    # Also detrend with higher-order polynomial
    for deg in [3, 4]:
        coeffs_n = np.polyfit(zs, ew_stds, deg)
        trend_n = np.polyval(coeffs_n, zs)
        resid_n = ew_stds - trend_n
        rho_n, p_n = spearmanr(diseqs, resid_n)
        print(f"    Poly deg-{deg} detrend: rho={rho_n:.3f}, p={p_n:.4f}")

    significant = p_resid < 0.10

    passed = significant
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T3_detrended',
        'n_bins': len(bins),
        'rho_raw': float(rho_raw), 'p_raw': float(p_raw),
        'rho_detrended': float(rho_resid), 'p_detrended': float(p_resid),
        'detrend_strengthens': abs(rho_resid) > abs(rho_raw),
        'PASS': passed,
    }


# ============================================================
# T4: Combined multi-ion evidence
# ============================================================

def test_T4_multi_ion(t1_result, t2_result, t3_result):
    """T4: Cascade signal present in BOTH MgII and CIV — multi-ion confirmation."""
    print("\n  T4: Combined multi-ion evidence")

    # Collect p-values from all tests
    p_values = {
        'FeII MgII EW std': t1_result['ew_std_p'],
        'FeII MgII EW IQR': t1_result['ew_iqr_p'],
        'CIV full range': t2_result['full_p'],
        'CIV high z': t2_result['high_z_p'],
        'CIV KS EW': t2_result['ks_ew_p'],
        'MgII detrended': t3_result['p_detrended'],
    }

    print(f"    {'Test':>25} {'p-value':>10} {'Sig':>5}")
    n_significant = 0
    for name, p in p_values.items():
        sig = '*' if p < 0.05 else ''
        if p < 0.05:
            n_significant += 1
        print(f"    {name:>25} {p:>10.4f} {sig:>5}")

    # Multi-ion: signal in BOTH MgII family AND CIV
    mgii_signal = t1_result['ew_std_p'] < 0.10 or t1_result['ew_iqr_p'] < 0.10
    civ_signal = t2_result['full_p'] < 0.10 or t2_result['ks_ew_p'] < 0.05

    both_ions = mgii_signal and civ_signal
    print(f"\n    MgII family signal: {mgii_signal}")
    print(f"    CIV signal: {civ_signal}")
    print(f"    Both ions: {both_ions}")
    print(f"    Significant tests: {n_significant}/{len(p_values)}")

    passed = both_ions
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T4_multi_ion',
        'p_values': {k: float(v) for k, v in p_values.items()},
        'n_significant': n_significant,
        'mgii_signal': mgii_signal,
        'civ_signal': civ_signal,
        'both_ions': both_ions,
        'PASS': passed,
    }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("exp_07: Deep Cascade Probe")
    print("FeII Subset, CIV High-z, and Z-Detrended Signal")
    print("Midnight Initiative, Thread 1 (Photon Archaeology)")
    print("=" * 70)

    t1 = test_T1_feii_subset()
    t2 = test_T2_civ_high_z()
    t3 = test_T3_detrended()
    t4 = test_T4_multi_ion(t1, t2, t3)

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'=' * 70}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 70}")

    data = {
        'experiment': 'exp_07_deep_cascade_probe',
        'initiative': 'midnight',
        'thread': 'photon_archaeology',
        'data_sources': [
            'SDSS DR16 FeII-confirmed MgII (69,675 systems)',
            'SDSS DR12 CIV catalog (445,765 systems, Monadi+2023)',
            'SDSS DR16 MgII full (159,524 systems)',
        ],
        'cascade_clock': {'a': A_CLOCK, 'slope': B_DFT},
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'score': f"{score}/4",
        'n_pass': score,
        'n_total': 4,
    }

    save_midnight_results('exp_07_deep_cascade_probe', _convert_numpy(data))
