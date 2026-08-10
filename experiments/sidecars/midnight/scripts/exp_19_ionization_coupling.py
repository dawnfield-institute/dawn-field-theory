"""
exp_19 -- The Ionization Coupling Law

Midnight Initiative, Thread 1 (Photon Archaeology)

PRE-REGISTERED: see journals/2026-06-10_exp19-exp20-preregistration.md and the
git commit containing this file. The registered quantities (per-ion coupling
beta, the beta-vs-IP curve, and its zero crossing) are NOT computed before
that commit; only loader smoke tests (--selftest) are run.

Hypothesis: coupling of absorber observables to the cascade clock
N(z) = 1.360 + (1/ln phi) * ln(t_lookback) is a monotonic function of the
ion's creation ionization energy. Settled phases (low IP) decouple;
actively-processing phases (high IP) carry the clock.

REGISTERED PREDICTION: the coupling zero crossing lies at
    E_cross = alpha^2 * m_e * c^2 = 1 Hartree = 27.20 eV
(the full Coulomb severance cost; Milestone R exp_24 energy-scale machinery,
factor-of-2 argument committed in the registration journal).

Locked metric: per ion, bin systems in z (quantile bins, >=15 systems/bin,
6-20 bins), median rest EW per bin, fit median = A + beta*N(z); coupling =
beta / (ion's overall median EW). Bootstrap (1000) for CI95. Monotone
(PAVA isotonic) fit of beta vs ln(IP); zero crossing E0 with bootstrap CI95.

Decision rule (registered):
  SUPPORTED:    Spearman(beta, IP) > 0, p < 0.05 (one-sided), AND
                27.2 eV in CI95(E0), AND CI95 width < 14.7 eV
  KILLED:       monotone AND 27.2 eV not in CI95(E0)
  INCONCLUSIVE: non-monotone, or CI95 spans the full known bracket

Outputs: results/exp_19_ionization_coupling_YYYYMMDD_HHMMSS.json
"""

import sys
import csv
import numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.stats import spearmanr

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = MIDNIGHT_ROOT.parent.parent.parent.parent / "data"
sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
from phase_rate import PHI, LN_PHI, save_midnight_results, _convert_numpy

B_DFT = 1.0 / LN_PHI
A_CLOCK = 1.360
N_FLOOR = 1.0
H0, Om, Ol = 67.36, 0.3153, 0.6847

E_CROSS_PRED = 27.2  # eV, registered: alpha^2 m_e c^2 (Hartree)
BRACKET_WIDTH = 33.49 - 18.83  # 14.66 eV, known A-E bracket

# Creation ionization energies (eV) -- locked convention (matches A-E plane)
ION_IP = {
    'AlII': 5.99, 'CaII': 6.11, 'MgII': 7.65, 'FeII': 7.87, 'SiII': 8.15,
    'CII': 11.26, 'AlIII': 18.83, 'SiIV': 33.49, 'CIV': 47.89, 'NV': 77.47,
}

N_BOOT = 1000


def z_to_lookback(z):
    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(Om * (1 + zp)**3 + Ol))
    r, _ = quad(integrand, 0, z)
    return r / (H0 * 1.022e-3)


_n_cache = {}

def n_at_z(z):
    key = round(float(z), 4)
    if key not in _n_cache:
        t = max(z_to_lookback(key), 0.001)
        _n_cache[key] = max(A_CLOCK + B_DFT * np.log(t), N_FLOOR)
    return _n_cache[key]


# ============================================================
# Loaders (return dict: ion -> (z_array, ew_array), with source tag)
# ============================================================

def load_xqr30():
    """XQR-30 merged catalog: per-species z and W."""
    out = {}
    with open(str(DATA_ROOT / "xqr30" / "xqr30_merged_catalog.csv"), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            try:
                z = float(row[2]); w = float(row[4])
                # per-transition rows carry W; strip the wavelength suffix
                sp = row[3].strip().split('_')[0]
            except (ValueError, IndexError):
                continue
            if w <= 0 or w > 20:
                continue
            out.setdefault(sp, [[], []])
            out[sp][0].append(z); out[sp][1].append(w)
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in out.items()
            if len(v[0]) >= 60}


def load_caii():
    raw = (DATA_ROOT / "sdss_mgii" / "CaII_Sardane_2014.tsv").read_text(
        encoding='utf-8', errors='replace')
    lines = [l for l in raw.splitlines() if l and not l.startswith('#')]
    hdr = [h.strip() for h in lines[0].split('\t')]
    iz, iw = hdr.index('zabs'), hdr.index('W0a')
    zs, ws = [], []
    for line in lines[1:]:
        if set(line.strip()) <= set('- \t'):
            continue
        p = line.split('\t')
        try:
            zs.append(float(p[iz])); ws.append(float(p[iw]))
        except (ValueError, IndexError):
            continue
    return np.array(zs), np.array(ws)


def load_dr16_mgii():
    from astropy.io import fits
    h = fits.open(str(DATA_ROOT / "sdss_mgii" / "SDSS_DR16_MgII_Catalog.fits"))
    d = h[1].data
    good = (d['REST_EW_MGII_2796'] > 0.2) & (d['SNR_2796'] > 5)
    z, ew = d['Z_ABS'][good], d['REST_EW_MGII_2796'][good]
    h.close()
    return np.asarray(z, float), np.asarray(ew, float)


def load_dr16_feii():
    from astropy.io import fits
    h = fits.open(str(DATA_ROOT / "sdss_mgii" / "SDSS_DR16_FeII_MgII_Catalog.fits"))
    d = h[1].data
    good = ((d['REST_EW_FEII_2600'] > 0.01) & (d['REST_EW_MGII_2796'] > 0.1) &
            (d['SNR_2796'] > 5) & np.isfinite(d['REST_EW_FEII_2600']))
    z, ew = d['Z_ABS'][good], d['REST_EW_FEII_2600'][good]
    h.close()
    return np.asarray(z, float), np.asarray(ew, float)


def load_dr12_civ():
    zs, ews = [], []
    with open(str(DATA_ROOT / "sdss_mgii" / "CIV_DR12_catalog.dat"), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 14:
                continue
            try:
                z, ew = float(parts[2]), float(parts[8])
            except (ValueError, IndexError):
                continue
            if 0 < ew < 10 and z > 1.4:
                zs.append(z); ews.append(ew)
    return np.array(zs), np.array(ews)


def gather_datasets():
    """Registered source-priority: XQR-30 primary above z=2; SDSS below."""
    datasets = []  # (ion, source, z, ew)
    xqr = load_xqr30()
    for sp, (z, w) in xqr.items():
        if sp in ION_IP:
            datasets.append((sp, 'XQR30', z, w))
    z, w = load_caii();      datasets.append(('CaII', 'Sardane14', z, w))
    z, w = load_dr16_mgii(); datasets.append(('MgII', 'DR16', z, w))
    z, w = load_dr16_feii(); datasets.append(('FeII', 'DR16', z, w))
    z, w = load_dr12_civ();  datasets.append(('CIV', 'DR12', z, w))
    return datasets


# ============================================================
# Coupling metric (locked)
# ============================================================

def coupling_beta(z, ew, rng=None):
    """beta = slope of binned-median EW vs N(z), / overall median EW."""
    if rng is not None:
        idx = rng.randint(0, len(z), len(z))
        z, ew = z[idx], ew[idx]
    n = len(z)
    n_bins = int(np.clip(n // 100, 6, 20))
    if n < 15 * 4:
        return None, 0
    edges = np.quantile(z, np.linspace(0, 1, n_bins + 1))
    bN, bw = [], []
    for i in range(n_bins):
        m = (z >= edges[i]) & (z < edges[i + 1]) if i < n_bins - 1 else (z >= edges[i])
        if m.sum() < 15:
            continue
        bN.append(n_at_z(np.median(z[m])))
        bw.append(np.median(ew[m]))
    if len(bN) < 4:
        return None, len(bN)
    bN, bw = np.array(bN), np.array(bw)
    slope = np.polyfit(bN, bw, 1)[0]
    return float(slope / np.median(ew)), len(bN)


def pava_increasing(x_order_vals, weights):
    """Pool-adjacent-violators: weighted isotonic increasing fit."""
    vals = list(map(float, x_order_vals))
    w = list(map(float, weights))
    # blocks: [value, weight, count]
    blocks = [[v, wt, 1] for v, wt in zip(vals, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-15:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / \
                (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1], blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = []
    for v, _, c in blocks:
        fit.extend([v] * c)
    return np.array(fit)


def zero_crossing(ips, betas, weights):
    """Zero of the isotonic fit of beta vs ln(IP); linear interp in ln space."""
    order = np.argsort(ips)
    ips_s = np.array(ips)[order]
    fit = pava_increasing(np.array(betas)[order], np.array(weights)[order])
    ln_ip = np.log(ips_s)
    for i in range(len(fit) - 1):
        if fit[i] <= 0 <= fit[i + 1]:
            if fit[i + 1] == fit[i]:
                return float(np.exp((ln_ip[i] + ln_ip[i + 1]) / 2))
            f = -fit[i] / (fit[i + 1] - fit[i])
            return float(np.exp(ln_ip[i] + f * (ln_ip[i + 1] - ln_ip[i])))
    if fit[0] > 0:
        return float(ips_s[0])   # crossing below the lowest ion
    return None                  # no crossing (all negative)


# ============================================================
# Main analysis
# ============================================================

def run():
    print("\nLoading catalogs...")
    datasets = gather_datasets()
    for ion, src, z, ew in datasets:
        print(f"  {ion:<6} [{src:<9}] n={len(z):6d}  z={z.min():.2f}-{z.max():.2f}")

    # Per-ion coupling with bootstrap
    print("\nPer-ion coupling beta (locked metric):")
    ion_results = []
    rng = np.random.RandomState(20260610)
    for ion, src, z, ew in datasets:
        beta, nbins = coupling_beta(z, ew)
        if beta is None:
            print(f"  {ion:<6} [{src}] EXCLUDED (<4 usable bins, n={len(z)})")
            ion_results.append({'ion': ion, 'source': src, 'excluded': True,
                                'n': int(len(z))})
            continue
        boots = []
        for _ in range(N_BOOT):
            b, _ = coupling_beta(z, ew, rng=rng)
            if b is not None:
                boots.append(b)
        boots = np.array(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ion_results.append({
            'ion': ion, 'source': src, 'excluded': False, 'n': int(len(z)),
            'n_bins': nbins, 'IP_eV': ION_IP[ion], 'beta': beta,
            'beta_CI95': [float(lo), float(hi)],
            'beta_boot_std': float(np.std(boots)),
            'beta_boots': boots.tolist(),
        })
        print(f"  {ion:<6} [{src:<9}] IP={ION_IP[ion]:6.2f} eV  "
              f"beta={beta:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  bins={nbins}")

    # Primary value per ion (registered: XQR30 above z=2, SDSS otherwise)
    primary = {}
    for r in ion_results:
        if r.get('excluded'):
            continue
        ion = r['ion']
        prefer_xqr = ion in ('SiII', 'CII', 'AlIII', 'SiIV', 'NV', 'FeII_highz')
        if ion not in primary:
            primary[ion] = r
        else:
            # XQR30 primary for high-z ions; SDSS primary for MgII/FeII/CIV/CaII
            if ion in ('MgII', 'FeII', 'CIV', 'CaII'):
                if r['source'] in ('DR16', 'DR12', 'Sardane14'):
                    primary[ion] = r
            elif r['source'] == 'XQR30':
                primary[ion] = r

    ions = sorted(primary.values(), key=lambda r: r['IP_eV'])
    ips = [r['IP_eV'] for r in ions]
    betas = [r['beta'] for r in ions]
    weights = [1.0 / max(r['beta_boot_std'], 1e-6)**2 for r in ions]

    print(f"\nPrimary curve ({len(ions)} ions):")
    for r in ions:
        print(f"  {r['ion']:<6} IP={r['IP_eV']:6.2f}  beta={r['beta']:+.4f} [{r['source']}]")

    # T1: monotonicity
    rho, p_two = spearmanr(ips, betas)
    p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
    t1 = bool(rho > 0 and p_one < 0.05)
    print(f"\n  T1 monotonicity: Spearman rho={rho:+.3f}, one-sided p={p_one:.4f} "
          f"-> {'PASS' if t1 else 'FAIL'}")

    # T2/T3: zero crossing with bootstrap
    e0_point = zero_crossing(ips, betas, weights)
    boot_mat = [np.array(r['beta_boots']) for r in ions]
    e0_boots = []
    rng2 = np.random.RandomState(7)
    n_nocross = 0
    for _ in range(N_BOOT):
        bs = [bm[rng2.randint(0, len(bm))] for bm in boot_mat]
        e0 = zero_crossing(ips, bs, weights)
        if e0 is None:
            n_nocross += 1
        else:
            e0_boots.append(e0)
    e0_boots = np.array(e0_boots)
    if len(e0_boots) >= N_BOOT * 0.5:
        e0_lo, e0_hi = np.percentile(e0_boots, [2.5, 97.5])
        ci_width = e0_hi - e0_lo
        t2 = bool(e0_lo <= E_CROSS_PRED <= e0_hi)
        t3 = bool(ci_width < BRACKET_WIDTH)
    else:
        e0_lo = e0_hi = ci_width = None
        t2 = t3 = False
    print(f"  T2 zero crossing: E0={e0_point if e0_point else float('nan'):.2f} eV, "
          f"CI95=[{e0_lo if e0_lo else float('nan'):.2f},{e0_hi if e0_hi else float('nan'):.2f}]  "
          f"predicted {E_CROSS_PRED} -> {'PASS' if t2 else 'FAIL'}")
    print(f"  T3 CI width {ci_width if ci_width else float('nan'):.2f} eV vs bracket "
          f"{BRACKET_WIDTH:.2f} -> {'PASS' if t3 else 'FAIL'}  "
          f"(no-crossing draws: {n_nocross}/{N_BOOT})")

    # T4: CaII weakest |beta|
    caii_b = next((abs(r['beta']) for r in ions if r['ion'] == 'CaII'), None)
    others = [abs(r['beta']) for r in ions if r['ion'] != 'CaII']
    t4 = bool(caii_b is not None and others and caii_b <= min(others))
    print(f"  T4 CaII weakest |beta|: {'PASS' if t4 else 'FAIL'}")

    # Registered verdict
    if t1 and t2 and t3:
        verdict = 'SUPPORTED'
    elif t1 and not t2 and ci_width is not None:
        verdict = 'KILLED'
    else:
        verdict = 'INCONCLUSIVE'
    print(f"\n  VERDICT (registered rule): {verdict}")

    score = sum([t1, t2, t3, t4])
    # strip bulky boot arrays from saved per-ion results
    for r in ion_results:
        r.pop('beta_boots', None)
    return {
        'experiment': 'exp_19_ionization_coupling',
        'initiative': 'midnight',
        'registered_prediction_eV': E_CROSS_PRED,
        'ion_results': ion_results,
        'primary_curve': [{'ion': r['ion'], 'IP_eV': r['IP_eV'], 'beta': r['beta'],
                           'source': r['source']} for r in ions],
        'T1': {'rho': float(rho), 'p_one_sided': float(p_one), 'PASS': t1},
        'T2': {'E0_point': e0_point, 'E0_CI95': [e0_lo, e0_hi], 'PASS': t2},
        'T3': {'CI_width': ci_width, 'bracket_width': BRACKET_WIDTH, 'PASS': t3,
               'n_nocross': int(n_nocross)},
        'T4': {'PASS': t4},
        'verdict': verdict,
        'score': f"{score}/4",
    }


def selftest():
    print("SELFTEST: loaders only (no registered quantities computed)")
    datasets = gather_datasets()
    for ion, src, z, ew in datasets:
        print(f"  {ion:<6} [{src:<9}] n={len(z):6d}  z={z.min():.2f}-{z.max():.2f}  "
              f"EW median={np.median(ew):.3f}")
    print("  OK")


if __name__ == '__main__':
    print("=" * 60)
    print("exp_19: The Ionization Coupling Law")
    print("Midnight Initiative -- pre-registered (E_cross = 27.2 eV)")
    print("=" * 60)
    if '--selftest' in sys.argv:
        selftest()
    else:
        data = run()
        save_midnight_results('exp_19_ionization_coupling', _convert_numpy(data))
