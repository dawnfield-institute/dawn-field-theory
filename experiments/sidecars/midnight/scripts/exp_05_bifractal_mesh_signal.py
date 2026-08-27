"""
exp_05 -- Bifractal Mesh Signal in SDSS MgII Absorber Statistics

Midnight Initiative, Thread 1 (Photon Archaeology)

Hypothesis: The cascade clock signal is encoded in the COLLECTIVE statistics
of absorber populations, not just individual line widths. At cascade transitions
(integer N), the PAC mesh is restructuring → more diverse severance events →
wider distributions of equivalent width and FWHM. Between transitions, the
mesh is settled → uniform → narrow distributions.

This tests the bifractal mesh prediction: photons are fragments of a larger
PAC structure, and the full cascade signal lives in how their properties
distribute, not in any single measurement.

Data: SDSS DR16 MgII Absorber Catalog (159,524 systems)
      Downloaded from https://wwwmpa.mpa-garching.mpg.de/SDSS/MgII/

Tests:
  T1: EW spread correlates with cascade disequilibrium (p < 0.05)
  T2: FWHM distribution shape changes at cascade transitions
  T3: The correlation is specific to the cascade clock, not generic z-dependence
  T4: Transition vs trough populations are statistically distinguishable

Predictions registered: commit 193d1c8e (2026-06-06), pushed to GitHub
BEFORE any observational data was examined.

Sources: exp_03/04, journals/2026-06-06_bifractal-mesh-photon-signal.md
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, ks_2samp, kurtosis, skew
from scipy.integrate import quad

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = MIDNIGHT_ROOT.parent

sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
from phase_rate import (
    DATA_ROOT,
    PHI, INV_PHI, LN_PHI, PI,
    save_midnight_results, _convert_numpy,
)

# Cascade clock parameters (from M9, calibrated on S8/Hubble/JWST)
B_DFT = 1.0 / LN_PHI  # 2.0781
A_CLOCK = 1.360  # constrained fit intercept

# Cosmological parameters
H0 = 67.36
OMEGA_M = 0.3153
OMEGA_LAMBDA = 0.6847


def z_to_lookback(z):
    """Convert redshift to lookback time in Gyr via Friedmann integration."""
    def integrand(zp):
        Ez = np.sqrt(OMEGA_M * (1 + zp)**3 + OMEGA_LAMBDA)
        return 1.0 / ((1 + zp) * Ez)
    result, _ = quad(integrand, 0, z)
    H0_gyr = H0 * 1.022e-3
    return result / H0_gyr


def n_at_z(z):
    """Cascade level at redshift z."""
    t = z_to_lookback(z)
    if t <= 0.001:
        t = 0.001
    return max(A_CLOCK + B_DFT * np.log(t), 1.0)


def disequilibrium_at_z(z):
    """Cascade disequilibrium: 1.0 at integer N, 0.0 at half-integer."""
    N = n_at_z(z)
    return max(0.0, 1.0 - 2.0 * abs(N - round(N)))


def load_sdss_data():
    """Load SDSS DR16 MgII catalog with quality cuts."""
    from astropy.io import fits
    fits_path = DATA_ROOT / "sdss_mgii" / "SDSS_DR16_MgII_Catalog.fits"
    hdul = fits.open(str(fits_path))
    data = hdul[1].data

    z = data['Z_ABS']
    fwhm = data['FWHM_VDISP_MGII_2796']
    ew = data['REST_EW_MGII_2796']
    snr = data['SNR_2796']

    good = (fwhm > 10) & (fwhm < 500) & (snr > 5) & (ew > 0.3) & np.isfinite(fwhm)
    hdul.close()
    return z[good], fwhm[good], ew[good], int(np.sum(good))


def bin_statistics(z_arr, fwhm_arr, ew_arr, n_bins=60):
    """Compute per-bin distribution statistics."""
    z_bins = np.linspace(0.36, 2.2, n_bins + 1)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2

    results = []
    for i in range(len(z_centers)):
        mask = (z_arr >= z_bins[i]) & (z_arr < z_bins[i+1])
        n = np.sum(mask)
        if n < 50:
            continue

        fw = fwhm_arr[mask]
        ew = ew_arr[mask]
        zc = float(z_centers[i])

        results.append({
            'z': zc,
            'N': n_at_z(zc),
            'diseq': disequilibrium_at_z(zc),
            'n': int(n),
            'fwhm_std': float(np.std(fw)),
            'fwhm_cv': float(np.std(fw) / np.mean(fw)),
            'fwhm_iqr': float(np.percentile(fw, 75) - np.percentile(fw, 25)),
            'fwhm_skew': float(skew(fw)),
            'fwhm_kurt': float(kurtosis(fw)),
            'fwhm_median': float(np.median(fw)),
            'ew_std': float(np.std(ew)),
            'ew_cv': float(np.std(ew) / np.mean(ew)),
            'ew_iqr': float(np.percentile(ew, 75) - np.percentile(ew, 25)),
            'ew_median': float(np.median(ew)),
        })

    return results


# ============================================================
# T1: EW spread correlates with cascade disequilibrium
# ============================================================

def test_T1_ew_spread_correlation(bin_stats):
    """T1: Equivalent width distribution width correlates with disequilibrium."""
    print("\n  T1: EW spread correlates with cascade disequilibrium")

    diseqs = np.array([b['diseq'] for b in bin_stats])
    ew_stds = np.array([b['ew_std'] for b in bin_stats])
    ew_iqrs = np.array([b['ew_iqr'] for b in bin_stats])

    rho_std, p_std = spearmanr(diseqs, ew_stds)
    rho_iqr, p_iqr = spearmanr(diseqs, ew_iqrs)

    print(f"    EW std vs diseq: rho={rho_std:.3f}, p={p_std:.4f}")
    print(f"    EW IQR vs diseq: rho={rho_iqr:.3f}, p={p_iqr:.4f}")

    significant = p_std < 0.05 or p_iqr < 0.05
    positive = rho_std > 0 or rho_iqr > 0

    passed = significant and positive
    print(f"    Significant (p<0.05): {significant}")
    print(f"    Positive correlation: {positive}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T1_ew_spread_correlation',
        'ew_std_rho': float(rho_std), 'ew_std_p': float(p_std),
        'ew_iqr_rho': float(rho_iqr), 'ew_iqr_p': float(p_iqr),
        'PASS': passed,
    }


# ============================================================
# T2: FWHM distribution shape changes at transitions
# ============================================================

def test_T2_fwhm_shape_change(bin_stats):
    """T2: FWHM distribution kurtosis or shape metric correlates with disequilibrium."""
    print("\n  T2: FWHM distribution shape changes at cascade transitions")

    diseqs = np.array([b['diseq'] for b in bin_stats])
    kurts = np.array([b['fwhm_kurt'] for b in bin_stats])
    skews = np.array([b['fwhm_skew'] for b in bin_stats])
    stds = np.array([b['fwhm_std'] for b in bin_stats])
    iqrs = np.array([b['fwhm_iqr'] for b in bin_stats])

    metrics = {
        'FWHM kurtosis': kurts,
        'FWHM skewness': skews,
        'FWHM std': stds,
        'FWHM IQR': iqrs,
    }

    any_significant = False
    results_detail = {}
    print(f"    {'Metric':>20} {'rho':>8} {'p-value':>10}")
    for name, vals in metrics.items():
        rho, p = spearmanr(diseqs, vals)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"    {name:>20} {rho:>8.3f} {p:>10.4f} {sig}")
        results_detail[name] = {'rho': float(rho), 'p': float(p)}
        if p < 0.05:
            any_significant = True

    passed = any_significant
    print(f"    Any metric significant: {any_significant}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T2_fwhm_shape_change',
        'metrics': results_detail,
        'any_significant': any_significant,
        'PASS': passed,
    }


# ============================================================
# T3: Correlation is specific to cascade clock, not generic z
# ============================================================

def test_T3_cascade_specificity(bin_stats):
    """T3: The EW spread correlation is specific to cascade disequilibrium,
    not just a generic function of redshift."""
    print("\n  T3: Correlation is cascade-specific, not generic z-dependence")

    diseqs = np.array([b['diseq'] for b in bin_stats])
    zs = np.array([b['z'] for b in bin_stats])
    ew_stds = np.array([b['ew_std'] for b in bin_stats])

    # Correlation with disequilibrium (cascade-specific)
    rho_diseq, p_diseq = spearmanr(diseqs, ew_stds)

    # Correlation with plain redshift (generic)
    rho_z, p_z = spearmanr(zs, ew_stds)

    # Correlation with random shuffled disequilibrium (control)
    rng = np.random.RandomState(42)
    n_shuffle = 1000
    shuffle_rhos = []
    for _ in range(n_shuffle):
        shuffled = rng.permutation(diseqs)
        r, _ = spearmanr(shuffled, ew_stds)
        shuffle_rhos.append(r)
    shuffle_rhos = np.array(shuffle_rhos)
    percentile = np.mean(shuffle_rhos < rho_diseq) * 100

    print(f"    EW std vs cascade diseq: rho={rho_diseq:.3f}, p={p_diseq:.4f}")
    print(f"    EW std vs plain z:       rho={rho_z:.3f}, p={p_z:.4f}")
    print(f"    Cascade rho percentile vs shuffled: {percentile:.1f}%")

    # The cascade correlation should be stronger than the z correlation
    # (if it were just z-dependence, rho_z would be stronger)
    cascade_stronger = abs(rho_diseq) > abs(rho_z)

    # The cascade rho should be above the 95th percentile of shuffled
    above_shuffle = percentile > 95.0

    print(f"    Cascade stronger than z: {cascade_stronger}")
    print(f"    Above 95th percentile of shuffled: {above_shuffle}")

    passed = cascade_stronger or above_shuffle
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T3_cascade_specificity',
        'rho_diseq': float(rho_diseq), 'p_diseq': float(p_diseq),
        'rho_z': float(rho_z), 'p_z': float(p_z),
        'shuffle_percentile': float(percentile),
        'cascade_stronger': cascade_stronger,
        'above_shuffle': above_shuffle,
        'PASS': passed,
    }


# ============================================================
# T4: Transition vs trough populations are distinguishable
# ============================================================

def test_T4_population_separation(z_arr, fwhm_arr, ew_arr):
    """T4: Absorbers at cascade transitions have statistically different
    distributions than absorbers at troughs (KS test)."""
    print("\n  T4: Transition vs trough populations are distinguishable")

    # Classify each absorber by cascade state
    transition_mask = np.zeros(len(z_arr), dtype=bool)
    trough_mask = np.zeros(len(z_arr), dtype=bool)

    for i in range(len(z_arr)):
        d = disequilibrium_at_z(z_arr[i])
        if d > 0.8:
            transition_mask[i] = True
        elif d < 0.2:
            trough_mask[i] = True

    n_trans = np.sum(transition_mask)
    n_trough = np.sum(trough_mask)
    print(f"    Transition absorbers (diseq > 0.8): {n_trans}")
    print(f"    Trough absorbers (diseq < 0.2): {n_trough}")

    if n_trans < 100 or n_trough < 100:
        print("    Insufficient data")
        return {'test': 'T4_population_separation', 'PASS': False}

    # KS test on EW distributions
    ew_trans = ew_arr[transition_mask]
    ew_trough = ew_arr[trough_mask]
    ks_ew, p_ew = ks_2samp(ew_trans, ew_trough)

    # KS test on FWHM distributions
    fwhm_trans = fwhm_arr[transition_mask]
    fwhm_trough = fwhm_arr[trough_mask]
    ks_fwhm, p_fwhm = ks_2samp(fwhm_trans, fwhm_trough)

    print(f"    EW distributions: KS={ks_ew:.4f}, p={p_ew:.2e}")
    print(f"    FWHM distributions: KS={ks_fwhm:.4f}, p={p_fwhm:.2e}")

    # Distribution summaries
    print(f"    EW median — transition: {np.median(ew_trans):.3f}, trough: {np.median(ew_trough):.3f}")
    print(f"    EW std    — transition: {np.std(ew_trans):.3f}, trough: {np.std(ew_trough):.3f}")
    print(f"    FWHM median — transition: {np.median(fwhm_trans):.1f}, trough: {np.median(fwhm_trough):.1f}")
    print(f"    FWHM std    — transition: {np.std(fwhm_trans):.1f}, trough: {np.std(fwhm_trough):.1f}")

    either_significant = p_ew < 0.05 or p_fwhm < 0.05

    passed = either_significant
    print(f"    Either KS test significant: {either_significant}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T4_population_separation',
        'n_transition': int(n_trans),
        'n_trough': int(n_trough),
        'ks_ew': float(ks_ew), 'p_ew': float(p_ew),
        'ks_fwhm': float(ks_fwhm), 'p_fwhm': float(p_fwhm),
        'ew_median_trans': float(np.median(ew_trans)),
        'ew_median_trough': float(np.median(ew_trough)),
        'ew_std_trans': float(np.std(ew_trans)),
        'ew_std_trough': float(np.std(ew_trough)),
        'fwhm_median_trans': float(np.median(fwhm_trans)),
        'fwhm_median_trough': float(np.median(fwhm_trough)),
        'fwhm_std_trans': float(np.std(fwhm_trans)),
        'fwhm_std_trough': float(np.std(fwhm_trough)),
        'PASS': passed,
    }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("exp_05: Bifractal Mesh Signal in SDSS MgII Absorbers")
    print("Midnight Initiative, Thread 1 (Photon Archaeology)")
    print("=" * 70)

    print("\n  Loading SDSS DR16 MgII catalog...")
    z_arr, fwhm_arr, ew_arr, n_total = load_sdss_data()
    print(f"  Loaded {n_total} high-quality absorbers")

    print("\n  Computing per-bin statistics...")
    bin_stats = bin_statistics(z_arr, fwhm_arr, ew_arr, n_bins=60)
    print(f"  {len(bin_stats)} bins with >= 50 absorbers")

    t1 = test_T1_ew_spread_correlation(bin_stats)
    t2 = test_T2_fwhm_shape_change(bin_stats)
    t3 = test_T3_cascade_specificity(bin_stats)
    t4 = test_T4_population_separation(z_arr, fwhm_arr, ew_arr)

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'=' * 70}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 70}")

    data = {
        'experiment': 'exp_05_bifractal_mesh_signal',
        'initiative': 'midnight',
        'thread': 'photon_archaeology',
        'data_source': 'SDSS DR16 MgII Absorber Catalog',
        'data_url': 'https://wwwmpa.mpa-garching.mpg.de/SDSS/MgII/',
        'n_absorbers': n_total,
        'n_bins': len(bin_stats),
        'cascade_clock': {'a': A_CLOCK, 'slope': B_DFT},
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'score': f"{score}/4",
        'n_pass': score,
        'n_total': 4,
    }

    save_midnight_results('exp_05_bifractal_mesh_signal', _convert_numpy(data))
