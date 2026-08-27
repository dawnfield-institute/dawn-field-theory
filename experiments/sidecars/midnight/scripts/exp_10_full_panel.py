"""
exp_10 -- Full Panel: Spatial, Kinematic, Thread 2/3 Seeds

Midnight Initiative — comprehensive sweep

Panel A: Spatial deep dive — is there a dipole in coupling differential?
Panel B: Kinematic deep dive — how does cascade sensitivity scale with velocity?
Panel C: Thread 2 seed — entropy gradient detection (substrate-independent life)
Panel D: Thread 3 seed — non-SEC channel detection (cross-sightline coherence)
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, ks_2samp
from scipy.integrate import quad
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


def load_mgii():
    from astropy.io import fits
    path = DATA_ROOT / "sdss_mgii" / "SDSS_DR16_MgII_Catalog.fits"
    hdul = fits.open(str(path))
    d = hdul[1].data
    good = ((d['FWHM_VDISP_MGII_2796'] > 10) & (d['FWHM_VDISP_MGII_2796'] < 500) &
            (d['FWHM_VDISP_MGII_2803'] > 10) & (d['FWHM_VDISP_MGII_2803'] < 500) &
            (d['REST_EW_MGII_2796'] > 0.2) & (d['REST_EW_MGII_2803'] > 0.1) &
            (d['SNR_2796'] > 5) & np.isfinite(d['FWHM_VDISP_MGII_2796']))
    result = {k: d[k][good] for k in
              ['Z_ABS', 'REST_EW_MGII_2796', 'REST_EW_MGII_2803',
               'FWHM_VDISP_MGII_2796', 'FWHM_VDISP_MGII_2803',
               'RA_QSO', 'DEC_QSO', 'SNR_2796', 'PLATE', 'MJD', 'FIBER_ID']}
    hdul.close()
    return result


if __name__ == '__main__':
    print("=" * 60)
    print("exp_10: Full Panel")
    print("=" * 60)

    d = load_mgii()
    z = d['Z_ABS']
    ew1 = d['REST_EW_MGII_2796']
    fw1 = d['FWHM_VDISP_MGII_2796']
    fw2 = d['FWHM_VDISP_MGII_2803']
    ra = d['RA_QSO']
    dec = d['DEC_QSO']
    disc = np.abs(fw1 - fw2) / (fw1 + fw2)

    trans = np.zeros(len(z), dtype=bool)
    trough = np.zeros(len(z), dtype=bool)
    for i in range(len(z)):
        _, dq = diseq_at_z(z[i])
        if dq > 0.8:
            trans[i] = True
        elif dq < 0.2:
            trough[i] = True

    print(f"Loaded {len(z)} absorbers ({np.sum(trans)} trans, {np.sum(trough)} trough)")
    results = {}

    # ========== PANEL A: SPATIAL ==========
    print(f"\n{'='*60}\nPANEL A: SPATIAL — Sky sector coupling differential\n{'='*60}")
    n_sectors = 8
    ra_edges = np.linspace(0, 360, n_sectors + 1)
    spatial = []
    for i in range(n_sectors):
        mask = (ra >= ra_edges[i]) & (ra < ra_edges[i + 1])
        t_m = trans & mask
        tr_m = trough & mask
        if np.sum(t_m) > 50 and np.sum(tr_m) > 50:
            delta = float(np.median(disc[t_m]) - np.median(disc[tr_m]))
            ks, p = ks_2samp(disc[t_m], disc[tr_m])
            ra_c = (ra_edges[i] + ra_edges[i + 1]) / 2
            spatial.append({'ra': float(ra_c), 'delta': delta, 'ks': float(ks),
                           'p': float(p), 'n_t': int(np.sum(t_m)), 'n_tr': int(np.sum(tr_m))})
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  RA {ra_edges[i]:>3.0f}-{ra_edges[i+1]:>3.0f}: delta={delta:+.4f} p={p:.2e} {sig}")

    if len(spatial) >= 4:
        deltas = [s['delta'] for s in spatial]
        max_d = max(deltas, key=abs)
        min_d = min(deltas, key=abs)
        dipole_ratio = abs(max_d) / max(abs(min_d), 0.0001)
        print(f"  Max |delta|: {max_d:+.4f}, Min |delta|: {min_d:+.4f}")
        print(f"  Dipole ratio: {dipole_ratio:.1f}x")
        print(f"  Range of deltas: {max(deltas) - min(deltas):.4f}")
    results['spatial'] = spatial

    # ========== PANEL B: KINEMATIC ==========
    print(f"\n{'='*60}\nPANEL B: KINEMATIC — Cascade sensitivity vs velocity\n{'='*60}")
    fw_quartiles = np.percentile(fw1, [0, 25, 50, 75, 100])
    kinematic = []
    for q in range(4):
        mask = (fw1 >= fw_quartiles[q]) & (fw1 < fw_quartiles[q + 1])
        t_m = trans & mask
        tr_m = trough & mask
        if np.sum(t_m) > 50 and np.sum(tr_m) > 50:
            d_t = float(np.median(disc[t_m]))
            d_tr = float(np.median(disc[tr_m]))
            delta = d_t - d_tr
            ks, p = ks_2samp(disc[t_m], disc[tr_m])
            direction = 'TIGHTER' if delta < 0 else 'LOOSER'
            kinematic.append({'quartile': q + 1, 'fw_lo': float(fw_quartiles[q]),
                             'fw_hi': float(fw_quartiles[q + 1]),
                             'delta': float(delta), 'p': float(p), 'dir': direction})
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  Q{q+1} (FWHM {fw_quartiles[q]:>5.0f}-{fw_quartiles[q+1]:>5.0f}): "
                  f"delta={delta:+.4f} {direction} p={p:.2e} {sig}")

    if len(kinematic) >= 3:
        fw_centers = [(k['fw_lo'] + k['fw_hi']) / 2 for k in kinematic]
        abs_deltas = [abs(k['delta']) for k in kinematic]
        rho_kin, p_kin = spearmanr(fw_centers, abs_deltas)
        print(f"  Sensitivity scales with FWHM: rho={rho_kin:.3f}, p={p_kin:.4f}")
    results['kinematic'] = kinematic

    # ========== PANEL C: THREAD 2 — ENTROPY GRADIENTS ==========
    print(f"\n{'='*60}\nPANEL C: THREAD 2 — Entropy gradient detection\n{'='*60}")
    plates = d['PLATE']
    mjds = d['MJD']
    fibers = d['FIBER_ID']
    sightlines = defaultdict(list)
    for i in range(len(z)):
        key = (int(plates[i]), int(mjds[i]), int(fibers[i]))
        sightlines[key].append((float(z[i]), float(ew1[i]), float(disc[i])))

    multi = {k: sorted(v, key=lambda x: x[0]) for k, v in sightlines.items() if len(v) >= 3}
    print(f"  Sightlines with 3+ absorbers: {len(multi)}")

    monotonic_count = 0
    gradient_strengths = []
    gradient_keys = []
    for key, absorbers in multi.items():
        zs_sl = [a[0] for a in absorbers]
        ews_sl = [a[1] for a in absorbers]
        if len(zs_sl) >= 3:
            rho, _ = spearmanr(zs_sl, ews_sl)
            gradient_strengths.append(abs(rho))
            if abs(rho) > 0.8:
                monotonic_count += 1
                gradient_keys.append(key)

    print(f"  Monotonic EW gradients (|rho|>0.8): {monotonic_count} ({monotonic_count/len(multi):.1%})")
    print(f"  Mean gradient strength: {np.mean(gradient_strengths):.3f}")
    print(f"  Expected random: ~0.33")
    excess = np.mean(gradient_strengths) - 0.33
    print(f"  Excess: {excess:+.3f} ({'above random' if excess > 0 else 'at random'})")

    # Do gradient sightlines cluster at transitions?
    if gradient_keys:
        grad_z = [np.mean([a[0] for a in multi[k]]) for k in gradient_keys]
        grad_dq = [diseq_at_z(zm)[1] for zm in grad_z]
        nongrad_keys = [k for k in multi if k not in gradient_keys][:500]
        nongrad_dq = [diseq_at_z(np.mean([a[0] for a in multi[k]]))[1] for k in nongrad_keys]

        mean_g = np.mean(grad_dq)
        mean_ng = np.mean(nongrad_dq)
        print(f"  Gradient sightlines mean diseq: {mean_g:.3f}")
        print(f"  Non-gradient mean diseq: {mean_ng:.3f}")
        print(f"  Gradients at transitions: {mean_g > mean_ng}")
    results['entropy_gradients'] = {
        'n_multi': len(multi), 'n_monotonic': monotonic_count,
        'mean_strength': float(np.mean(gradient_strengths)), 'excess': float(excess)}

    # ========== PANEL D: THREAD 3 — CROSS-SIGHTLINE COHERENCE ==========
    print(f"\n{'='*60}\nPANEL D: THREAD 3 — Cross-sightline coherence\n{'='*60}")
    z_bins = np.linspace(0.5, 2.0, 25)
    coherence_data = []
    for i in range(len(z_bins) - 1):
        mask = (z >= z_bins[i]) & (z < z_bins[i + 1])
        if np.sum(mask) < 20:
            continue

        sight_ews = defaultdict(list)
        for j in np.where(mask)[0]:
            sid = (int(plates[j]), int(mjds[j]), int(fibers[j]))
            sight_ews[sid].append(float(ew1[j]))

        if len(sight_ews) >= 10:
            mean_ews = [np.mean(v) for v in sight_ews.values()]
            within_stds = [np.std(v) for v in sight_ews.values() if len(v) >= 2]

            cross_std = np.std(mean_ews)
            within_std = np.mean(within_stds) if within_stds else 0
            ratio = cross_std / max(within_std, 0.001)

            zc = (z_bins[i] + z_bins[i + 1]) / 2
            _, dq = diseq_at_z(zc)
            coherence_data.append({
                'z': float(zc), 'diseq': float(dq),
                'cross_std': float(cross_std), 'within_std': float(within_std),
                'ratio': float(ratio), 'n_sightlines': len(sight_ews)})

    if coherence_data:
        diseqs_c = [c['diseq'] for c in coherence_data]
        ratios_c = [c['ratio'] for c in coherence_data]
        rho_cross, p_cross = spearmanr(diseqs_c, ratios_c)
        print(f"  Cross-sightline coherence vs cascade: rho={rho_cross:.3f}, p={p_cross:.4f}")
        print(f"  Mean cross/within ratio: {np.mean(ratios_c):.3f}")

        trans_r = [r for r, dq in zip(ratios_c, diseqs_c) if dq > 0.7]
        trough_r = [r for r, dq in zip(ratios_c, diseqs_c) if dq < 0.3]
        if trans_r and trough_r:
            print(f"  At transitions: cross/within = {np.mean(trans_r):.3f}")
            print(f"  At troughs: cross/within = {np.mean(trough_r):.3f}")
            more_coherent = np.mean(trans_r) < np.mean(trough_r)
            print(f"  More coherent at transitions: {more_coherent}")
    results['cross_sightline'] = coherence_data

    # ========== SUMMARY ==========
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    data = {
        'experiment': 'exp_10_full_panel',
        'initiative': 'midnight',
        'panels': results,
    }
    save_midnight_results('exp_10_full_panel', _convert_numpy(data))
