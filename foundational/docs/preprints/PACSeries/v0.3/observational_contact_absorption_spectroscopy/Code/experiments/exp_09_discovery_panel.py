"""
exp_09 -- Discovery Panel: What Can the Tapestry Measurement Teach Us?

Midnight Initiative, Thread 1 (Photon Archaeology)

The doublet coupling differential is a new observable. This panel asks:
what can we LEARN from it? Seven analyses — spatial structure, environment,
gas state, velocity structure, systematics, optical depth, absorber type.

Not testing DFT. Using the measurement to discover.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp, spearmanr
from scipy.integrate import quad

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = MIDNIGHT_ROOT.parent.parent.parent.parent / "data"
sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
from phase_rate import save_midnight_results, _convert_numpy

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
            (d['SNR_2796'] > 5) & np.isfinite(d['FWHM_VDISP_MGII_2796']) &
            np.isfinite(d['FWHM_VDISP_MGII_2803']))
    result = {k: d[k][good] for k in ['Z_ABS', 'REST_EW_MGII_2796', 'REST_EW_MGII_2803',
              'FWHM_VDISP_MGII_2796', 'FWHM_VDISP_MGII_2803', 'RA_QSO', 'DEC_QSO', 'SNR_2796']}
    hdul.close()
    return result


def classify(z_arr):
    trans = np.zeros(len(z_arr), dtype=bool)
    trough = np.zeros(len(z_arr), dtype=bool)
    for i in range(len(z_arr)):
        _, dq = diseq_at_z(z_arr[i])
        if dq > 0.8: trans[i] = True
        elif dq < 0.2: trough[i] = True
    return trans, trough


if __name__ == '__main__':
    print("=" * 60)
    print("exp_09: Discovery Panel")
    print("=" * 60)

    d = load_mgii()
    z = d['Z_ABS']
    ew1 = d['REST_EW_MGII_2796']; ew2 = d['REST_EW_MGII_2803']
    fw1 = d['FWHM_VDISP_MGII_2796']; fw2 = d['FWHM_VDISP_MGII_2803']
    ra = d['RA_QSO']; dec = d['DEC_QSO']; snr = d['SNR_2796']

    dr = ew1 / ew2
    disc = np.abs(fw1 - fw2) / (fw1 + fw2)
    trans, trough = classify(z)
    print(f"Loaded {len(z)} absorbers ({np.sum(trans)} trans, {np.sum(trough)} trough)")

    results = {}

    # --- PANEL 1: SPATIAL ---
    print(f"\n{'='*60}\nPANEL 1: SPATIAL STRUCTURE\n{'='*60}")
    north = dec >= 0; south = dec < 0
    spatial = {}
    for label, mask in [('North', north), ('South', south)]:
        t_m = trans & mask; tr_m = trough & mask
        if np.sum(t_m) > 100 and np.sum(tr_m) > 100:
            ks, p = ks_2samp(disc[t_m], disc[tr_m])
            delta = float(np.median(disc[t_m]) - np.median(disc[tr_m]))
            spatial[label] = {'ks': float(ks), 'p': float(p), 'delta': delta,
                              'n_t': int(np.sum(t_m)), 'n_tr': int(np.sum(tr_m))}
            print(f"  {label}: delta={delta:+.4f} KS={ks:.4f} p={p:.2e}")
    results['spatial'] = spatial

    # --- PANEL 2: ABSORBER STRENGTH ---
    print(f"\n{'='*60}\nPANEL 2: ABSORBER STRENGTH\n{'='*60}")
    ew_med = float(np.median(ew1))
    strength = {}
    for label, mask in [('Strong (EW>med)', ew1 > ew_med), ('Weak (EW<=med)', ew1 <= ew_med)]:
        t_m = trans & mask; tr_m = trough & mask
        if np.sum(t_m) > 100 and np.sum(tr_m) > 100:
            ks, p = ks_2samp(disc[t_m], disc[tr_m])
            d_t = float(np.median(disc[t_m])); d_tr = float(np.median(disc[tr_m]))
            direction = 'TIGHTER' if d_t < d_tr else 'LOOSER'
            strength[label] = {'trans': d_t, 'trough': d_tr, 'dir': direction,
                               'ks': float(ks), 'p': float(p)}
            print(f"  {label}: trans={d_t:.4f} trough={d_tr:.4f} {direction} p={p:.2e}")
    results['strength'] = strength

    # --- PANEL 3: GAS STATE FROM DOUBLET RATIO ---
    print(f"\n{'='*60}\nPANEL 3: GAS STATE (DOUBLET RATIO CLASSES)\n{'='*60}")
    gas_state = {}
    for label, lo, hi in [('Saturated (DR<1.2)', 0, 1.2),
                           ('Intermediate (1.2-1.5)', 1.2, 1.5),
                           ('Thin (DR>1.5)', 1.5, 10)]:
        mask = (dr >= lo) & (dr < hi)
        frac_t = float(np.sum(trans & mask) / max(np.sum(trans), 1))
        frac_tr = float(np.sum(trough & mask) / max(np.sum(trough), 1))
        gas_state[label] = {'frac_trans': frac_t, 'frac_trough': frac_tr}
        print(f"  {label}: {frac_t:.1%} of trans, {frac_tr:.1%} of troughs")
    results['gas_state'] = gas_state

    # --- PANEL 4: VELOCITY STRUCTURE ---
    print(f"\n{'='*60}\nPANEL 4: VELOCITY STRUCTURE (FWHM CLASSES)\n{'='*60}")
    fw_med = float(np.median(fw1))
    velocity = {}
    for label, mask in [('Narrow (FWHM<med)', fw1 < fw_med), ('Broad (FWHM>=med)', fw1 >= fw_med)]:
        t_m = trans & mask; tr_m = trough & mask
        if np.sum(t_m) > 100 and np.sum(tr_m) > 100:
            ks, p = ks_2samp(disc[t_m], disc[tr_m])
            d_t = float(np.median(disc[t_m])); d_tr = float(np.median(disc[tr_m]))
            direction = 'TIGHTER' if d_t < d_tr else 'LOOSER'
            velocity[label] = {'trans': d_t, 'trough': d_tr, 'dir': direction,
                                'ks': float(ks), 'p': float(p)}
            print(f"  {label}: trans={d_t:.4f} trough={d_tr:.4f} {direction} p={p:.2e}")
    results['velocity'] = velocity

    # --- PANEL 5: SNR SYSTEMATIC CHECK ---
    print(f"\n{'='*60}\nPANEL 5: SNR SYSTEMATIC CHECK\n{'='*60}")
    snr_med = float(np.median(snr))
    snr_check = {}
    for label, mask in [('High SNR', snr > snr_med), ('Low SNR', snr <= snr_med)]:
        t_m = trans & mask; tr_m = trough & mask
        if np.sum(t_m) > 100 and np.sum(tr_m) > 100:
            ks, p = ks_2samp(disc[t_m], disc[tr_m])
            d_t = float(np.median(disc[t_m])); d_tr = float(np.median(disc[tr_m]))
            direction = 'TIGHTER' if d_t < d_tr else 'LOOSER'
            snr_check[label] = {'trans': d_t, 'trough': d_tr, 'dir': direction,
                                 'ks': float(ks), 'p': float(p)}
            print(f"  {label}: trans={d_t:.4f} trough={d_tr:.4f} {direction} p={p:.2e}")
    results['snr_check'] = snr_check

    # --- PANEL 6: OPTICAL DEPTH FROM DOUBLET ---
    print(f"\n{'='*60}\nPANEL 6: OPTICAL DEPTH (TAU FROM DOUBLET RATIO)\n{'='*60}")
    valid_dr = (dr > 1.01) & (dr < 1.99)
    tau = -np.log(2.0 - dr[valid_dr])
    t_tau = trans[valid_dr]; tr_tau = trough[valid_dr]
    if np.sum(t_tau) > 100 and np.sum(tr_tau) > 100:
        tau_t = tau[t_tau]; tau_tr = tau[tr_tau]
        ks, p = ks_2samp(tau_t, tau_tr)
        direction = 'denser' if np.median(tau_t) > np.median(tau_tr) else 'thinner'
        results['optical_depth'] = {
            'tau_trans': float(np.median(tau_t)), 'tau_trough': float(np.median(tau_tr)),
            'direction': direction, 'ks': float(ks), 'p': float(p)}
        print(f"  Transition: median tau = {np.median(tau_t):.3f}")
        print(f"  Trough:     median tau = {np.median(tau_tr):.3f}")
        print(f"  Gas is {direction} at transitions, KS={ks:.4f} p={p:.2e}")

    # --- PANEL 7: ABSORBER TYPE ---
    print(f"\n{'='*60}\nPANEL 7: ABSORBER TYPE CLASSIFICATION\n{'='*60}")
    types = {}
    for label, lo, hi in [('Very weak (0.2-0.5A)', 0.2, 0.5), ('Weak (0.5-1.0A)', 0.5, 1.0),
                           ('Strong (1.0-2.0A)', 1.0, 2.0), ('Very strong (>2.0A)', 2.0, 10)]:
        mask = (ew1 >= lo) & (ew1 < hi)
        t_m = trans & mask; tr_m = trough & mask
        n_t = int(np.sum(t_m)); n_tr = int(np.sum(tr_m))
        if n_t > 30 and n_tr > 30:
            ks, p = ks_2samp(disc[t_m], disc[tr_m])
            d_t = float(np.median(disc[t_m])); d_tr = float(np.median(disc[tr_m]))
            direction = 'TIGHTER' if d_t < d_tr else 'LOOSER'
            types[label] = {'trans': d_t, 'trough': d_tr, 'dir': direction,
                            'ks': float(ks), 'p': float(p), 'n_t': n_t, 'n_tr': n_tr}
            print(f"  {label}: {direction} p={p:.2e} (n={n_t}+{n_tr})")
    results['absorber_types'] = types

    # --- SUMMARY ---
    print(f"\n{'='*60}")
    print("SUMMARY OF DISCOVERIES")
    print(f"{'='*60}")

    data = {
        'experiment': 'exp_09_discovery_panel',
        'initiative': 'midnight',
        'thread': 'photon_archaeology',
        'n_absorbers': len(z),
        'panels': results,
    }

    save_midnight_results('exp_09_discovery_panel', _convert_numpy(data))
