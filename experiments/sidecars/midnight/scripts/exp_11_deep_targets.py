"""
exp_11 -- Deep Targets: CIV Detrend, Spatial Dipole, XQR-30 Tapestry,
          Entropy Gradient Follow-up

Midnight Initiative — five high-value targets in one run.
"""

import sys
import csv
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, ks_2samp
from scipy.integrate import quad
from scipy.optimize import minimize
from collections import defaultdict

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
from phase_rate import DATA_ROOT, save_midnight_results, _convert_numpy

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
B_DFT = 1.0 / LN_PHI
A_CLOCK = 1.360
H0, Om, Ol = 67.36, 0.3153, 0.6847


def diseq_at_z(zz):
    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(Om * (1 + zp)**3 + Ol))
    r, _ = quad(integrand, 0, zz)
    t = r / (H0 * 1.022e-3)
    if t <= 0.001:
        t = 0.001
    N = max(A_CLOCK + B_DFT * np.log(t), 1.0)
    return N, max(0.0, 1.0 - 2.0 * abs(N - round(N)))


# ============================================================
# TARGET 1: CIV z-detrend
# ============================================================

def target_1_civ_detrend():
    """Does the CIV KS=0.21 intra-doublet signal survive z-detrending?"""
    print(f"\n{'='*60}")
    print("TARGET 1: CIV doublet coupling — z-detrend test")
    print(f"{'='*60}")

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

    z_all = np.array(z_all)
    ew1_all = np.array(ew1_all)
    ew2_all = np.array(ew2_all)

    good = (ew1_all > 0.05) & (ew2_all > 0.02)
    z_c = z_all[good]
    dr = ew1_all[good] / ew2_all[good]
    disc = np.abs(ew1_all[good] - ew2_all[good]) / (ew1_all[good] + ew2_all[good])
    print(f"  CIV absorbers: {len(z_c)}")

    # Bin and compute metrics
    z_bins = np.linspace(1.5, 4.5, 60)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    bin_data = []
    for i in range(len(z_centers)):
        mask = (z_c >= z_bins[i]) & (z_c < z_bins[i + 1])
        n = np.sum(mask)
        if n < 50:
            continue
        zc = float(z_centers[i])
        _, dq = diseq_at_z(zc)
        bin_data.append({
            'z': zc, 'diseq': dq,
            'dr_median': float(np.median(dr[mask])),
            'disc_median': float(np.median(disc[mask])),
            'n': int(n),
        })

    zs = np.array([b['z'] for b in bin_data])
    diseqs = np.array([b['diseq'] for b in bin_data])

    for metric_name, key in [('Doublet ratio', 'dr_median'), ('EW discrepancy', 'disc_median')]:
        vals = np.array([b[key] for b in bin_data])
        rho_raw, p_raw = spearmanr(diseqs, vals)

        for deg in [2, 3]:
            coeffs = np.polyfit(zs, vals, deg)
            resid = vals - np.polyval(coeffs, zs)
            rho_det, p_det = spearmanr(diseqs, resid)
            survives = p_det < 0.05
            marker = ' *** SURVIVES ***' if survives else ''
            print(f"  {metric_name}: raw rho={rho_raw:.3f} p={p_raw:.4f} | "
                  f"detrend(deg-{deg}) rho={rho_det:.3f} p={p_det:.4f}{marker}")

    return bin_data


# ============================================================
# TARGET 2: Spatial dipole
# ============================================================

def target_2_spatial_dipole():
    """Fit a dipole to the coupling differential across the sky."""
    print(f"\n{'='*60}")
    print("TARGET 2: Spatial dipole in coupling differential")
    print(f"{'='*60}")

    from astropy.io import fits
    hdul = fits.open(str(DATA_ROOT / "sdss_mgii" / "SDSS_DR16_MgII_Catalog.fits"))
    d = hdul[1].data
    good = ((d['FWHM_VDISP_MGII_2796'] > 10) & (d['FWHM_VDISP_MGII_2796'] < 500) &
            (d['FWHM_VDISP_MGII_2803'] > 10) & (d['REST_EW_MGII_2796'] > 0.2) &
            (d['SNR_2796'] > 5) & np.isfinite(d['FWHM_VDISP_MGII_2796']) &
            np.isfinite(d['FWHM_VDISP_MGII_2803']))
    z = d['Z_ABS'][good]
    fw1 = d['FWHM_VDISP_MGII_2796'][good]
    fw2 = d['FWHM_VDISP_MGII_2803'][good]
    ra = d['RA_QSO'][good]
    dec_arr = d['DEC_QSO'][good]
    hdul.close()

    disc = np.abs(fw1 - fw2) / (fw1 + fw2)
    trans = np.zeros(len(z), dtype=bool)
    trough = np.zeros(len(z), dtype=bool)
    for i in range(len(z)):
        _, dq = diseq_at_z(z[i])
        if dq > 0.8:
            trans[i] = True
        elif dq < 0.2:
            trough[i] = True

    # Compute per-absorber "cascade response": disc relative to local median
    # Then fit dipole to transition absorbers
    # Dipole model: delta(RA, DEC) = A * cos(angle from pole) + B
    n_sectors = 12
    ra_edges = np.linspace(0, 360, n_sectors + 1)
    dec_edges = np.array([-90, 0, 90])

    sector_data = []
    for i in range(n_sectors):
        for j in range(len(dec_edges) - 1):
            mask = ((ra >= ra_edges[i]) & (ra < ra_edges[i + 1]) &
                    (dec_arr >= dec_edges[j]) & (dec_arr < dec_edges[j + 1]))
            t_m = trans & mask
            tr_m = trough & mask
            if np.sum(t_m) > 30 and np.sum(tr_m) > 30:
                delta = float(np.median(disc[t_m]) - np.median(disc[tr_m]))
                ra_c = (ra_edges[i] + ra_edges[i + 1]) / 2
                dec_c = (dec_edges[j] + dec_edges[j + 1]) / 2
                sector_data.append({
                    'ra': ra_c, 'dec': dec_c, 'delta': delta,
                    'n_t': int(np.sum(t_m)), 'n_tr': int(np.sum(tr_m))})

    if len(sector_data) < 6:
        print("  Insufficient sectors for dipole fit")
        return sector_data

    # Convert to Cartesian for dipole fit
    ras = np.array([s['ra'] for s in sector_data]) * np.pi / 180
    decs = np.array([s['dec'] for s in sector_data]) * np.pi / 180
    deltas = np.array([s['delta'] for s in sector_data])

    x = np.cos(decs) * np.cos(ras)
    y = np.cos(decs) * np.sin(ras)
    z_cart = np.sin(decs)

    # Fit: delta = a*x + b*y + c*z + d (dipole + monopole)
    A = np.vstack([x, y, z_cart, np.ones_like(x)]).T
    coeffs, residuals, _, _ = np.linalg.lstsq(A, deltas, rcond=None)

    dipole_amp = np.sqrt(coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2)
    monopole = coeffs[3]

    # Dipole direction
    if dipole_amp > 0:
        pole_x, pole_y, pole_z = coeffs[0] / dipole_amp, coeffs[1] / dipole_amp, coeffs[2] / dipole_amp
        pole_ra = np.arctan2(pole_y, pole_x) * 180 / np.pi
        if pole_ra < 0:
            pole_ra += 360
        pole_dec = np.arcsin(pole_z) * 180 / np.pi
    else:
        pole_ra, pole_dec = 0, 0

    # Significance: compare dipole amplitude to shuffled
    rng = np.random.RandomState(42)
    shuffle_amps = []
    for _ in range(1000):
        shuffled = rng.permutation(deltas)
        c_shuf, _, _, _ = np.linalg.lstsq(A, shuffled, rcond=None)
        shuffle_amps.append(np.sqrt(c_shuf[0]**2 + c_shuf[1]**2 + c_shuf[2]**2))
    percentile = np.mean(np.array(shuffle_amps) < dipole_amp) * 100

    # Webb et al. alpha dipole: RA ~ 17.3h = 260 deg, DEC ~ -61 deg
    webb_ra, webb_dec = 260, -61
    angular_sep = np.arccos(
        np.sin(pole_dec * np.pi / 180) * np.sin(webb_dec * np.pi / 180) +
        np.cos(pole_dec * np.pi / 180) * np.cos(webb_dec * np.pi / 180) *
        np.cos((pole_ra - webb_ra) * np.pi / 180)) * 180 / np.pi

    print(f"  Sectors used: {len(sector_data)}")
    print(f"  Monopole: {monopole:.4f}")
    print(f"  Dipole amplitude: {dipole_amp:.4f}")
    print(f"  Dipole direction: RA={pole_ra:.1f} deg, DEC={pole_dec:.1f} deg")
    print(f"  Percentile vs shuffled: {percentile:.1f}%")
    print(f"  Webb alpha dipole: RA=260 deg, DEC=-61 deg")
    print(f"  Angular separation from Webb: {angular_sep:.1f} deg")
    print(f"  {'ALIGNED' if angular_sep < 30 else 'MISALIGNED' if angular_sep > 60 else 'INTERMEDIATE'}")

    return {
        'monopole': float(monopole), 'dipole_amp': float(dipole_amp),
        'pole_ra': float(pole_ra), 'pole_dec': float(pole_dec),
        'percentile': float(percentile),
        'webb_separation': float(angular_sep),
        'sectors': sector_data}


# ============================================================
# TARGET 3: XQR-30 deep tapestry
# ============================================================

def target_3_xqr30_tapestry():
    """Full pairwise analysis of XQR-30 multi-ion systems."""
    print(f"\n{'='*60}")
    print("TARGET 3: XQR-30 deep tapestry (multi-ion systems)")
    print(f"{'='*60}")

    rows = []
    catalog_path = DATA_ROOT / "xqr30" / "xqr30_merged_catalog.csv"
    with open(str(catalog_path), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row and not row[0].startswith('#'):
                rows.append(row)

    # Group by SystemID
    systems = defaultdict(list)
    for r in rows:
        sid = r[0].strip()
        species = r[3].strip()
        try:
            z_abs = float(r[2])
            ew = float(r[4]) if r[4].strip() else 0
            b = float(r[11]) if r[11].strip() else 0
        except:
            continue
        systems[sid].append({
            'species': species, 'z': z_abs, 'ew': ew, 'b': b})

    # Find systems with 4+ distinct ion species
    rich_systems = {}
    for sid, components in systems.items():
        species_set = set(c['species'] for c in components if c['ew'] > 0)
        # Group by base ion (strip wavelength)
        base_ions = set()
        for sp in species_set:
            base = sp.split('_')[0]
            base_ions.add(base)
        if len(base_ions) >= 3:
            mean_z = np.mean([c['z'] for c in components])
            rich_systems[sid] = {
                'components': components,
                'ions': sorted(base_ions),
                'n_ions': len(base_ions),
                'z': float(mean_z),
            }

    print(f"  Systems with 3+ base ions: {len(rich_systems)}")

    # For each rich system, compute intra-ion vs cross-ion EW coherence
    tapestry_results = []
    for sid, sys_info in rich_systems.items():
        N, dq = diseq_at_z(sys_info['z'])

        # Group components by base ion
        ion_ews = defaultdict(list)
        for c in sys_info['components']:
            base = c['species'].split('_')[0]
            if c['ew'] > 0:
                ion_ews[base].append(c['ew'])

        # Intra-ion: std of EW within each ion (normalized)
        intra_spreads = []
        for ion, ews in ion_ews.items():
            if len(ews) >= 2:
                cv = np.std(ews) / np.mean(ews)
                intra_spreads.append(cv)

        # Cross-ion: std of mean EW across ions (normalized)
        ion_means = [np.mean(ews) for ews in ion_ews.values() if len(ews) >= 1]
        cross_spread = np.std(ion_means) / np.mean(ion_means) if len(ion_means) >= 2 else 0

        intra_mean = np.mean(intra_spreads) if intra_spreads else 0

        tapestry_results.append({
            'sid': sid, 'z': sys_info['z'], 'N': float(N), 'diseq': float(dq),
            'n_ions': sys_info['n_ions'], 'ions': sys_info['ions'],
            'intra_spread': float(intra_mean),
            'cross_spread': float(cross_spread),
            'ratio': float(intra_mean / max(cross_spread, 0.001)),
        })

    if tapestry_results:
        diseqs = [t['diseq'] for t in tapestry_results]
        intras = [t['intra_spread'] for t in tapestry_results]
        crosses = [t['cross_spread'] for t in tapestry_results]
        ratios = [t['ratio'] for t in tapestry_results]

        rho_intra, p_intra = spearmanr(diseqs, intras) if len(diseqs) > 5 else (0, 1)
        rho_cross, p_cross = spearmanr(diseqs, crosses) if len(diseqs) > 5 else (0, 1)
        rho_ratio, p_ratio = spearmanr(diseqs, ratios) if len(diseqs) > 5 else (0, 1)

        print(f"  Intra-ion spread vs diseq: rho={rho_intra:.3f}, p={p_intra:.4f}")
        print(f"  Cross-ion spread vs diseq: rho={rho_cross:.3f}, p={p_cross:.4f}")
        print(f"  Intra/cross ratio vs diseq: rho={rho_ratio:.3f}, p={p_ratio:.4f}")

        # Show richest systems
        print(f"\n  Richest systems:")
        for t in sorted(tapestry_results, key=lambda x: -x['n_ions'])[:5]:
            print(f"    z={t['z']:.3f} N={t['N']:.2f} diseq={t['diseq']:.2f} "
                  f"ions={t['n_ions']} intra={t['intra_spread']:.3f} "
                  f"cross={t['cross_spread']:.3f}")

    return tapestry_results


# ============================================================
# TARGET 5: Entropy gradient follow-up
# ============================================================

def target_5_entropy_gradients():
    """What makes the 34% monotonic sightlines special?"""
    print(f"\n{'='*60}")
    print("TARGET 5: Entropy gradient sightline characterization")
    print(f"{'='*60}")

    from astropy.io import fits
    hdul = fits.open(str(DATA_ROOT / "sdss_mgii" / "SDSS_DR16_MgII_Catalog.fits"))
    d = hdul[1].data
    good = ((d['REST_EW_MGII_2796'] > 0.2) & (d['SNR_2796'] > 5) &
            np.isfinite(d['REST_EW_MGII_2796']) & np.isfinite(d['FWHM_VDISP_MGII_2796']))
    z = d['Z_ABS'][good]
    ew = d['REST_EW_MGII_2796'][good]
    fw = d['FWHM_VDISP_MGII_2796'][good]
    plates = d['PLATE'][good]
    mjds = d['MJD'][good]
    fibers = d['FIBER_ID'][good]
    ra = d['RA_QSO'][good]
    dec_arr = d['DEC_QSO'][good]
    hdul.close()

    sightlines = defaultdict(list)
    for i in range(len(z)):
        key = (int(plates[i]), int(mjds[i]), int(fibers[i]))
        sightlines[key].append({
            'z': float(z[i]), 'ew': float(ew[i]), 'fw': float(fw[i]),
            'ra': float(ra[i]), 'dec': float(dec_arr[i])})

    multi = {k: sorted(v, key=lambda x: x['z'])
             for k, v in sightlines.items() if len(v) >= 3}

    # Classify sightlines
    gradient_info = []
    non_gradient_info = []
    for key, absorbers in multi.items():
        zs = [a['z'] for a in absorbers]
        ews = [a['ew'] for a in absorbers]
        fws = [a['fw'] for a in absorbers]
        rho_ew, _ = spearmanr(zs, ews)
        rho_fw, _ = spearmanr(zs, fws)

        info = {
            'n': len(absorbers),
            'z_span': max(zs) - min(zs),
            'mean_ew': np.mean(ews),
            'mean_fw': np.mean(fws),
            'ew_gradient': float(rho_ew),
            'fw_gradient': float(rho_fw),
            'ra': absorbers[0]['ra'],
            'dec': absorbers[0]['dec'],
        }

        if abs(rho_ew) > 0.8:
            gradient_info.append(info)
        else:
            non_gradient_info.append(info)

    print(f"  Gradient sightlines: {len(gradient_info)}")
    print(f"  Non-gradient sightlines: {len(non_gradient_info)}")

    # Compare properties
    for prop in ['n', 'z_span', 'mean_ew', 'mean_fw']:
        g_vals = [g[prop] for g in gradient_info]
        ng_vals = [g[prop] for g in non_gradient_info]
        g_med = np.median(g_vals)
        ng_med = np.median(ng_vals)
        ks, p = ks_2samp(g_vals, ng_vals)
        print(f"  {prop}: gradient={g_med:.3f} non-gradient={ng_med:.3f} KS p={p:.2e}")

    # Are gradient sightlines spatially clustered?
    g_ras = [g['ra'] for g in gradient_info]
    ng_ras = [g['ra'] for g in non_gradient_info[:len(gradient_info)]]
    ks_ra, p_ra = ks_2samp(g_ras, ng_ras)
    print(f"  RA distribution: KS={ks_ra:.4f} p={p_ra:.4f}")

    # Do gradient sightlines have a preferred EW direction?
    increasing = sum(1 for g in gradient_info if g['ew_gradient'] > 0.8)
    decreasing = sum(1 for g in gradient_info if g['ew_gradient'] < -0.8)
    print(f"  EW increases with z: {increasing} ({increasing/len(gradient_info):.0%})")
    print(f"  EW decreases with z: {decreasing} ({decreasing/len(gradient_info):.0%})")

    # Do FWHM gradients correlate with EW gradients?
    ew_grads = [g['ew_gradient'] for g in gradient_info]
    fw_grads = [g['fw_gradient'] for g in gradient_info]
    rho_ef, p_ef = spearmanr(ew_grads, fw_grads)
    print(f"  EW-FWHM gradient correlation: rho={rho_ef:.3f}, p={p_ef:.4f}")

    return {'n_gradient': len(gradient_info), 'n_non_gradient': len(non_gradient_info),
            'increasing': increasing, 'decreasing': decreasing}


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("exp_11: Deep Targets")
    print("=" * 60)

    results = {}
    results['civ_detrend'] = target_1_civ_detrend()
    results['spatial_dipole'] = target_2_spatial_dipole()
    results['xqr30_tapestry'] = target_3_xqr30_tapestry()
    results['entropy_gradients'] = target_5_entropy_gradients()

    save_midnight_results('exp_11_deep_targets', _convert_numpy(results))
