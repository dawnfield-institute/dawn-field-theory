"""
exp_06 -- Doublet Coherence: PAC Coupling in the MgII 2796/2803 Pair

Midnight Initiative, Thread 1 (Photon Archaeology)

Hypothesis: The MgII doublet (2796/2803 Angstrom) encodes PAC coherence — the
degree to which the two transitions are coupled through the conservation law.
At cascade transitions (integer N), the ledger is actively balancing → the
two lines lock together (small FWHM discrepancy, high inter-line correlation).
Between transitions (settled), the constraint relaxes → lines diverge.

This is the first test of the "coherence as overlooked channel" idea from the
phase-rate primitive journal. Not quantum coherence, but PAC coherence — the
conservation law coupling two severance channels of the same ion.

Data: SDSS DR16 MgII Absorber Catalog (159,524 systems, both doublet components)

Tests:
  T1: FWHM discrepancy between doublet lines correlates with disequilibrium
  T2: Doublet coupling is cascade-specific, not generic z-dependence
  T3: Inter-line FWHM correlation tightens at transitions
  T4: Doublet coherence metric separates transition/trough populations

Sources: exp_05, journals/2026-06-06_bifractal-mesh-photon-signal.md,
         journals/2026-06-03_phase-rate-primitive.md (Part I.2, coherence channel)
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, ks_2samp
from scipy.integrate import quad

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
from phase_rate import (
    DATA_ROOT,
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


def load_doublet_data():
    """Load SDSS DR16 MgII catalog with both doublet components."""
    from astropy.io import fits
    fits_path = DATA_ROOT / "sdss_mgii" / "SDSS_DR16_MgII_Catalog.fits"
    hdul = fits.open(str(fits_path))
    data = hdul[1].data

    z = data['Z_ABS']
    ew1 = data['REST_EW_MGII_2796']
    ew2 = data['REST_EW_MGII_2803']
    fw1 = data['FWHM_VDISP_MGII_2796']
    fw2 = data['FWHM_VDISP_MGII_2803']
    dv1 = data['DELTA_V_MGII_2796']
    dv2 = data['DELTA_V_MGII_2803']
    snr = data['SNR_2796']

    good = ((ew1 > 0.2) & (ew2 > 0.1) & (fw1 > 10) & (fw2 > 10) &
            (fw1 < 500) & (fw2 < 500) & (snr > 5) &
            np.isfinite(ew1) & np.isfinite(ew2) &
            np.isfinite(fw1) & np.isfinite(fw2) &
            np.isfinite(dv1) & np.isfinite(dv2))

    hdul.close()
    return {
        'z': z[good], 'ew1': ew1[good], 'ew2': ew2[good],
        'fw1': fw1[good], 'fw2': fw2[good],
        'dv1': dv1[good], 'dv2': dv2[good],
        'n': int(np.sum(good)),
    }


def compute_doublet_metrics(d):
    """Compute per-absorber doublet coherence metrics."""
    return {
        'doublet_ratio': d['ew1'] / d['ew2'],
        'fwhm_discrepancy': np.abs(d['fw1'] - d['fw2']) / (d['fw1'] + d['fw2']),
        'fwhm_ratio': d['fw1'] / d['fw2'],
        'velocity_offset': np.abs(d['dv1'] - d['dv2']),
    }


def bin_doublet_stats(d, metrics, n_bins=50):
    """Bin doublet metrics by redshift, compute per-bin statistics."""
    z_bins = np.linspace(0.36, 2.2, n_bins + 1)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2

    results = []
    for i in range(len(z_centers)):
        mask = (d['z'] >= z_bins[i]) & (d['z'] < z_bins[i + 1])
        n = np.sum(mask)
        if n < 50:
            continue

        zc = float(z_centers[i])
        fw1_bin = d['fw1'][mask]
        fw2_bin = d['fw2'][mask]

        # Inter-line FWHM Pearson correlation within this bin
        if len(fw1_bin) > 10:
            fw_r, fw_p = pearsonr(fw1_bin, fw2_bin)
        else:
            fw_r, fw_p = 0.0, 1.0

        results.append({
            'z': zc,
            'N': n_at_z(zc),
            'diseq': diseq_at_z(zc),
            'n': int(n),
            'fwhm_disc_median': float(np.median(metrics['fwhm_discrepancy'][mask])),
            'fwhm_disc_mean': float(np.mean(metrics['fwhm_discrepancy'][mask])),
            'dr_median': float(np.median(metrics['doublet_ratio'][mask])),
            'dr_std': float(np.std(metrics['doublet_ratio'][mask])),
            'dv_median': float(np.median(metrics['velocity_offset'][mask])),
            'fw_interline_r': float(fw_r),
            'fw_interline_p': float(fw_p),
        })

    return results


# ============================================================
# T1: FWHM discrepancy correlates with disequilibrium
# ============================================================

def test_T1_fwhm_discrepancy(bin_stats):
    """T1: Doublet FWHM discrepancy anticorrelates with cascade disequilibrium."""
    print("\n  T1: FWHM discrepancy correlates with cascade disequilibrium")

    diseqs = np.array([b['diseq'] for b in bin_stats])
    disc_med = np.array([b['fwhm_disc_median'] for b in bin_stats])
    disc_mean = np.array([b['fwhm_disc_mean'] for b in bin_stats])

    rho_med, p_med = spearmanr(diseqs, disc_med)
    rho_mean, p_mean = spearmanr(diseqs, disc_mean)

    print(f"    FWHM disc (median) vs diseq: rho={rho_med:.3f}, p={p_med:.4f}")
    print(f"    FWHM disc (mean) vs diseq:   rho={rho_mean:.3f}, p={p_mean:.4f}")
    print(f"    Direction: {'lines LOCK at transitions' if rho_med < 0 else 'lines DIVERGE at transitions'}")

    significant = p_med < 0.05 or p_mean < 0.05
    correct_direction = rho_med < 0  # negative = less discrepancy at transitions

    passed = significant and correct_direction
    print(f"    Significant: {significant}, Correct direction: {correct_direction}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T1_fwhm_discrepancy',
        'rho_median': float(rho_med), 'p_median': float(p_med),
        'rho_mean': float(rho_mean), 'p_mean': float(p_mean),
        'direction': 'lock_at_transitions' if rho_med < 0 else 'diverge_at_transitions',
        'PASS': passed,
    }


# ============================================================
# T2: Cascade-specific, not generic z
# ============================================================

def test_T2_cascade_specificity(bin_stats):
    """T2: Doublet coupling signal is specific to cascade clock, not plain z."""
    print("\n  T2: Doublet coupling is cascade-specific")

    diseqs = np.array([b['diseq'] for b in bin_stats])
    zs = np.array([b['z'] for b in bin_stats])
    disc = np.array([b['fwhm_disc_median'] for b in bin_stats])

    rho_diseq, p_diseq = spearmanr(diseqs, disc)
    rho_z, p_z = spearmanr(zs, disc)

    # Permutation test
    rng = np.random.RandomState(42)
    n_shuffle = 1000
    shuffle_rhos = []
    for _ in range(n_shuffle):
        shuffled = rng.permutation(diseqs)
        r, _ = spearmanr(shuffled, disc)
        shuffle_rhos.append(r)
    percentile = np.mean(np.array(shuffle_rhos) < rho_diseq) * 100
    # For negative rho, we want LOW percentile (below shuffled)
    if rho_diseq < 0:
        percentile = 100 - percentile

    print(f"    FWHM disc vs cascade diseq: rho={rho_diseq:.3f}, p={p_diseq:.4f}")
    print(f"    FWHM disc vs plain z:       rho={rho_z:.3f}, p={p_z:.4f}")
    print(f"    Cascade |rho| percentile vs shuffled: {percentile:.1f}%")

    cascade_significant = p_diseq < 0.05
    above_shuffle = percentile > 95.0

    passed = cascade_significant and above_shuffle
    print(f"    Cascade significant: {cascade_significant}")
    print(f"    Above 95th percentile of shuffled: {above_shuffle}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T2_cascade_specificity',
        'rho_diseq': float(rho_diseq), 'p_diseq': float(p_diseq),
        'rho_z': float(rho_z), 'p_z': float(p_z),
        'shuffle_percentile': float(percentile),
        'PASS': passed,
    }


# ============================================================
# T3: Inter-line correlation tightens at transitions
# ============================================================

def test_T3_interline_correlation(bin_stats):
    """T3: FWHM Pearson r between the two doublet lines correlates with diseq."""
    print("\n  T3: Inter-line FWHM correlation tightens at transitions")

    diseqs = np.array([b['diseq'] for b in bin_stats])
    fw_rs = np.array([b['fw_interline_r'] for b in bin_stats])

    rho, p = spearmanr(diseqs, fw_rs)

    # Show transition vs trough inter-line r
    trans_rs = [b['fw_interline_r'] for b in bin_stats if b['diseq'] > 0.8]
    trough_rs = [b['fw_interline_r'] for b in bin_stats if b['diseq'] < 0.2]

    trans_mean = np.mean(trans_rs) if trans_rs else 0
    trough_mean = np.mean(trough_rs) if trough_rs else 0

    print(f"    Inter-line r vs diseq: rho={rho:.3f}, p={p:.4f}")
    print(f"    Mean inter-line r at transitions: {trans_mean:.4f}")
    print(f"    Mean inter-line r at troughs:     {trough_mean:.4f}")
    print(f"    Direction: {'tighter at transitions' if rho > 0 else 'looser at transitions'}")

    significant = p < 0.10  # relaxed to 10% for this exploratory test
    correct_direction = rho > 0

    passed = significant and correct_direction
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T3_interline_correlation',
        'rho': float(rho), 'p': float(p),
        'mean_r_transition': float(trans_mean),
        'mean_r_trough': float(trough_mean),
        'PASS': passed,
    }


# ============================================================
# T4: Coherence metric separates populations
# ============================================================

def test_T4_coherence_population(d, metrics):
    """T4: KS test on doublet coherence metrics between transition and trough absorbers."""
    print("\n  T4: Doublet coherence separates transition/trough populations")

    trans_mask = np.zeros(len(d['z']), dtype=bool)
    trough_mask = np.zeros(len(d['z']), dtype=bool)

    for i in range(len(d['z'])):
        dq = diseq_at_z(d['z'][i])
        if dq > 0.8:
            trans_mask[i] = True
        elif dq < 0.2:
            trough_mask[i] = True

    n_trans = np.sum(trans_mask)
    n_trough = np.sum(trough_mask)
    print(f"    Transition absorbers: {n_trans}")
    print(f"    Trough absorbers: {n_trough}")

    results = {}
    any_sig = False

    for name, vals in [
        ('FWHM discrepancy', metrics['fwhm_discrepancy']),
        ('Doublet ratio', metrics['doublet_ratio']),
        ('Velocity offset', metrics['velocity_offset']),
        ('FWHM ratio', metrics['fwhm_ratio']),
    ]:
        v_trans = vals[trans_mask]
        v_trough = vals[trough_mask]
        ks, p = ks_2samp(v_trans, v_trough)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"    {name:>22}: KS={ks:.4f}, p={p:.2e} {sig}")
        print(f"      transition median={np.median(v_trans):.4f}, trough median={np.median(v_trough):.4f}")
        results[name] = {'ks': float(ks), 'p': float(p),
                         'median_trans': float(np.median(v_trans)),
                         'median_trough': float(np.median(v_trough))}
        if p < 0.05:
            any_sig = True

    passed = any_sig
    print(f"    Any metric significant: {any_sig}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T4_coherence_population',
        'n_transition': int(n_trans),
        'n_trough': int(n_trough),
        'metrics': results,
        'PASS': passed,
    }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("exp_06: Doublet Coherence — PAC Coupling in MgII 2796/2803")
    print("Midnight Initiative, Thread 1 (Photon Archaeology)")
    print("=" * 70)

    print("\n  Loading SDSS DR16 MgII catalog (both doublet components)...")
    d = load_doublet_data()
    print(f"  Loaded {d['n']} absorbers with valid doublet measurements")

    metrics = compute_doublet_metrics(d)
    bin_stats = bin_doublet_stats(d, metrics, n_bins=50)
    print(f"  {len(bin_stats)} bins with >= 50 absorbers")

    t1 = test_T1_fwhm_discrepancy(bin_stats)
    t2 = test_T2_cascade_specificity(bin_stats)
    t3 = test_T3_interline_correlation(bin_stats)
    t4 = test_T4_coherence_population(d, metrics)

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'=' * 70}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 70}")

    data = {
        'experiment': 'exp_06_doublet_coherence',
        'initiative': 'midnight',
        'thread': 'photon_archaeology',
        'data_source': 'SDSS DR16 MgII Absorber Catalog',
        'n_absorbers': d['n'],
        'n_bins': len(bin_stats),
        'cascade_clock': {'a': A_CLOCK, 'slope': B_DFT},
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'score': f"{score}/4",
        'n_pass': score,
        'n_total': 4,
    }

    save_midnight_results('exp_06_doublet_coherence', _convert_numpy(data))
