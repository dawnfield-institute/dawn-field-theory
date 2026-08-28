"""
exp_18 -- CaII Low-z Kill Test: Discriminating the Cascade Clock from Smooth Mimics

Midnight Initiative, Thread 1 (Photon Archaeology)

PRE-REGISTRATION (Phase B is registered BEFORE any CaII data is examined;
see journals/2026-06-10_caii-preregistration.md and the git commit containing
this file. The previously fetched data/sdss_mgii/CaII_Sardane_2014.tsv is a
747-byte VizieR ERROR stub -- "Table 'table3' does not exist" -- so no CaII
data has been seen at registration time.)

Background (Paper 12, exp_12, exp_13): within z = 1.5-4.5, the cascade clock
    N(z) = a + (1/ln(phi)) * ln(t_lookback),  a = 1.360 (LOCKED, from M9)
fits CIV Doppler-b evolution as well as any smooth mimic. Discriminating power
lives OUTSIDE the fitted range: at z < 0.5 the log form steepens
(dN/dz grows as t_lookback -> 0) while polynomial mimics trained on the CIV
range stay gentle. The clock also predicts a FLOOR: N is floored at 1 for
t_lookback < t1 (M9 boundary handling), i.e. flattening below z* = 0.061.

Phase A (CIV calibration -- already-analyzed data, NOT part of the registration):
  Reproduces, with full provenance, the model-comparison table quoted in the
  2026-06-08 cascade-vs-halo-virial journal: R^2/AIC/BIC for linear z,
  quadratic z, cubic z, halo virial A + B*(1+z)^alpha, free ln(t), and the
  cascade clock, on binned CIV median Doppler b (98 z-bins, 1.5-4.5).
  Also reproduces the phi-constrained-slope zero-cost result on b.

Phase B (CaII test -- PRE-REGISTERED, runs only when a valid catalog exists):
  Data: Sardane, Rao & Turnshek 2014 (MNRAS 444, 1747) SDSS CaII absorber
  catalog (~435 systems, z ~< 0.7), via VizieR. Observable: rest-frame
  equivalent width of CaII K 3934 (W3934), binned medians in z
  (>= 15 systems per bin).

  Registered tests:
    B1 (shape extrapolation): the z^2 and z^3 shapes fitted to CIV b in
        Phase A are extrapolated to the CaII range and affinely rescaled
        (y = A + B*f(z), 2 free params per family -- same freedom as the
        clock's A + m*N(z)). Equal parameter count; compare R^2/BIC.
    B2 (direct family fit): each family fit directly to the CaII bins with
        its own parameters (z:2, z^2:3, z^3:4, clock:2). Compare BIC.
    B3 (floor signature): qualitative -- does the lowest-z end flatten,
        as the N-floor predicts and no polynomial does?

  Registered decision rule (BIC, lower = better):
    - Clock DISCRIMINATED FOR if it beats every polynomial family by
      delta-BIC >= 6 in B1 or B2.
    - Clock KILLED in this channel if some polynomial family beats it by
      delta-BIC >= 6 in BOTH B1 and B2.
    - Otherwise INCONCLUSIVE (expected risk: only ~435 systems; CaII
      selection is biased toward dusty sightlines -- a registered threat
      to validity).

Outputs: results/exp_18_caii_low_z_YYYYMMDD_HHMMSS.json
"""

import sys
import numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import curve_fit

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
from phase_rate import DATA_ROOT, PHI, LN_PHI, save_midnight_results, _convert_numpy

B_DFT = 1.0 / LN_PHI          # 2.0781 -- slope locked by phi
A_CLOCK = 1.360               # intercept locked from M9 (S8/H0/JWST fit)
N_FLOOR = 1.0                 # M9 boundary handling: N >= 1
H0, Om, Ol = 67.36, 0.3153, 0.6847

CAII_PATH = DATA_ROOT / "sdss_mgii" / "CaII_Sardane_2014.tsv"


def z_to_lookback(z):
    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(Om * (1 + zp)**3 + Ol))
    r, _ = quad(integrand, 0, z)
    return r / (H0 * 1.022e-3)


def n_at_z(z):
    t = z_to_lookback(z)
    if t <= 0.001:
        t = 0.001
    return max(A_CLOCK + B_DFT * np.log(t), N_FLOOR)


def fit_metrics(y, pred, k):
    """R^2, AIC, BIC for a fit with k free parameters."""
    n = len(y)
    ss_res = float(np.sum((y - pred)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    aic = n * np.log(ss_res / n) + 2 * k
    bic = n * np.log(ss_res / n) + k * np.log(n)
    return {'R2': float(r2), 'AIC': float(aic), 'BIC': float(bic),
            'k': k, 'n': n, 'ss_res': ss_res}


def load_civ():
    z_all, b_all = [], []
    with open(str(DATA_ROOT / "sdss_mgii" / "CIV_DR12_catalog.dat"), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 14:
                continue
            try:
                z_all.append(float(parts[2]))
                b_all.append(float(parts[6]))
            except (ValueError, IndexError):
                continue
    z = np.array(z_all)
    b = np.array(b_all)
    good = (b > 5) & (b < 300) & (z > 1.4)
    return z[good], b[good]


def bin_medians(z, y, lo, hi, n_edges, min_per_bin):
    edges = np.linspace(lo, hi, n_edges)
    centers = (edges[:-1] + edges[1:]) / 2
    bz, by = [], []
    for i in range(len(centers)):
        m = (z >= edges[i]) & (z < edges[i + 1])
        if np.sum(m) < min_per_bin:
            continue
        bz.append(float(centers[i]))
        by.append(float(np.median(y[m])))
    return np.array(bz), np.array(by)


# ============================================================
# Phase A: CIV model comparison (calibration; provenance for Paper 12 Sec 4.2)
# ============================================================

def phase_A():
    print("\nPhase A: CIV Doppler-b model comparison (z = 1.5-4.5)")
    z, b = load_civ()
    print(f"  systems after cuts: {len(z)}")

    bz, bb = bin_medians(z, b, 1.5, 4.5, 99, 30)   # 98 bins max
    print(f"  bins: {len(bz)}")

    bN = np.array([n_at_z(zz) for zz in bz])
    blnt = np.array([np.log(z_to_lookback(zz)) for zz in bz])

    models = {}

    def poly_model(x, deg, k, label):
        c = np.polyfit(x, bb, deg)
        pred = np.polyval(c, x)
        m = fit_metrics(bb, pred, k)
        m['coeffs'] = [float(v) for v in c]
        models[label] = m
        return c

    poly_model(bz, 1, 2, 'z (linear)')
    coeffs_z2 = poly_model(bz, 2, 3, 'z^2 (quadratic)')
    coeffs_z3 = poly_model(bz, 3, 4, 'z^3 (cubic)')

    # halo virial: b = A + B*(1+z)^alpha
    def virial(zv, A, B, alpha):
        return A + B * (1 + zv)**alpha
    try:
        p, _ = curve_fit(virial, bz, bb, p0=[20.0, 5.0, 1.0], maxfev=20000)
        m = fit_metrics(bb, virial(bz, *p), 3)
        m['alpha'] = float(p[2])
        models['halo virial A+B(1+z)^alpha'] = m
    except RuntimeError:
        models['halo virial A+B(1+z)^alpha'] = {'error': 'fit failed'}

    # free ln(t): b = A + B*ln(t)
    c_free = np.polyfit(blnt, bb, 1)
    m = fit_metrics(bb, np.polyval(c_free, blnt), 2)
    m['slope_ln_t'] = float(c_free[0])
    models['ln(t) free slope'] = m

    # cascade clock: b = A + m*N(z); slope of N vs ln(t) fixed by phi
    c_clock = np.polyfit(bN, bb, 1)
    m = fit_metrics(bb, np.polyval(c_clock, bN), 2)
    m['slope_vs_N'] = float(c_clock[0])
    models['N(z) cascade clock'] = m

    # phi-constraint cost on b: clock-implied ln(t) slope vs free slope
    implied = c_clock[0] * B_DFT
    slope_ratio = implied / c_free[0] if c_free[0] != 0 else 0
    r2_cost = models['ln(t) free slope']['R2'] - models['N(z) cascade clock']['R2']

    print(f"\n  {'model':<32}{'k':>3}{'R2':>10}{'AIC':>10}{'BIC':>10}")
    for label, m in models.items():
        if 'error' in m:
            print(f"  {label:<32}  FIT FAILED")
            continue
        print(f"  {label:<32}{m['k']:>3}{m['R2']:>10.4f}{m['AIC']:>10.1f}{m['BIC']:>10.1f}")
    va = models.get('halo virial A+B(1+z)^alpha', {})
    if 'alpha' in va:
        print(f"  halo virial alpha = {va['alpha']:.4f}")
    print(f"  phi-constrained vs free ln(t) slope ratio: {slope_ratio:.6f}")
    print(f"  R2 cost of phi constraint: {r2_cost:.6f}")

    return {
        'phase': 'A_civ_calibration',
        'n_systems': int(len(z)),
        'n_bins': int(len(bz)),
        'models': models,
        'phi_slope_ratio': float(slope_ratio),
        'phi_r2_cost': float(r2_cost),
        'z2_coeffs': [float(v) for v in coeffs_z2],
        'z3_coeffs': [float(v) for v in coeffs_z3],
    }


# ============================================================
# Phase B: CaII low-z test (PRE-REGISTERED)
# ============================================================

def load_caii():
    """Sardane+ 2014 CaII catalog from VizieR TSV. Returns (z_abs, W3934) or None."""
    if not CAII_PATH.exists():
        return None
    raw = CAII_PATH.read_text(encoding='utf-8', errors='replace')
    if 'Error' in raw[:2000] and 'does not exist' in raw[:2000]:
        return None   # the known VizieR error stub
    # VizieR TSV: comment lines start with '#'; header row then '---' separator
    lines = [l for l in raw.splitlines() if l and not l.startswith('#')]
    if len(lines) < 3:
        return None
    header = [h.strip() for h in lines[0].split('\t')]

    def find_col(cands):
        for c in cands:
            for i, h in enumerate(header):
                if h.lower() == c.lower():
                    return i
        for c in cands:
            for i, h in enumerate(header):
                if c.lower() in h.lower():
                    return i
        return None

    # W0a is the lambda-3934 rest EW column name in the VizieR table1 export
    iz = find_col(['zabs', 'z_abs', 'zAbs', 'z'])
    iw = find_col(['W0a', 'W3934', 'EW3934', 'Wr3934', 'W(3934)', 'WK'])
    if iz is None or iw is None:
        print(f"  CaII columns not identified in header: {header}")
        return None
    z_list, w_list = [], []
    for line in lines[1:]:
        if set(line.strip()) <= set('- \t'):
            continue
        parts = line.split('\t')
        if len(parts) <= max(iz, iw):
            continue
        try:
            z_list.append(float(parts[iz]))
            w_list.append(float(parts[iw]))
        except ValueError:
            continue
    if len(z_list) < 50:
        return None
    return np.array(z_list), np.array(w_list)


def phase_B(cal):
    print("\nPhase B: CaII low-z discrimination test (PRE-REGISTERED)")
    data = load_caii()
    if data is None:
        print("  CaII catalog not available or invalid -- Phase B SKIPPED.")
        print("  (Registration remains valid: no CaII data examined.)")
        return {'phase': 'B_caii_test', 'status': 'SKIPPED_no_data'}

    z, w = data
    print(f"  CaII systems loaded: {len(z)}  (z range {z.min():.3f}-{z.max():.3f})")

    n_target_bins = max(6, min(15, len(z) // 25))
    bz, bw = bin_medians(z, w, float(z.min()), float(z.max()),
                         n_target_bins + 1, 15)
    if len(bz) < 5:
        print(f"  only {len(bz)} usable bins -- INCONCLUSIVE by registered rule")
        return {'phase': 'B_caii_test', 'status': 'INCONCLUSIVE_too_few_bins',
                'n_bins': int(len(bz))}
    print(f"  bins: {len(bz)}")

    bN = np.array([n_at_z(zz) for zz in bz])
    results = {}

    # B1: shape extrapolation -- CIV-trained z^2/z^3 shapes, affine rescale (k=2)
    f_z2 = np.polyval(cal['z2_coeffs'], bz)
    f_z3 = np.polyval(cal['z3_coeffs'], bz)
    b1 = {}
    for label, shape in [('z^2 shape (CIV-trained)', f_z2),
                         ('z^3 shape (CIV-trained)', f_z3),
                         ('clock shape N(z)', bN)]:
        c = np.polyfit(shape, bw, 1)
        b1[label] = fit_metrics(bw, np.polyval(c, shape), 2)
    results['B1_shape_extrapolation'] = b1

    # B2: direct family fits with own parameters
    b2 = {}
    for label, x, deg, k in [('z (linear)', bz, 1, 2),
                             ('z^2 (quadratic)', bz, 2, 3),
                             ('z^3 (cubic)', bz, 3, 4)]:
        c = np.polyfit(x, bw, deg)
        b2[label] = fit_metrics(bw, np.polyval(c, x), k)
    c = np.polyfit(bN, bw, 1)
    b2['N(z) cascade clock'] = fit_metrics(bw, np.polyval(c, bN), 2)
    results['B2_direct_fits'] = b2

    # B3: floor signature (qualitative)
    lowz = bz < 0.1
    results['B3_floor_note'] = (
        f"{int(np.sum(lowz))} bins below z=0.1 (floor regime z* = 0.061)")

    # Registered decision rule
    def dbic(d, clock_key):
        cb = d[clock_key]['BIC']
        polys = {k: v['BIC'] for k, v in d.items() if k != clock_key}
        best_poly = min(polys.values())
        return cb - best_poly   # negative = clock better

    d1 = dbic(b1, 'clock shape N(z)')
    d2 = dbic(b2, 'N(z) cascade clock')
    if d1 <= -6 or d2 <= -6:
        verdict = 'DISCRIMINATED_FOR_CLOCK'
    elif d1 >= 6 and d2 >= 6:
        verdict = 'CLOCK_KILLED_IN_CHANNEL'
    else:
        verdict = 'INCONCLUSIVE'

    print(f"\n  B1 delta-BIC (clock - best poly): {d1:+.1f}")
    print(f"  B2 delta-BIC (clock - best poly): {d2:+.1f}")
    print(f"  VERDICT (registered rule): {verdict}")

    results.update({'phase': 'B_caii_test', 'status': 'RUN',
                    'n_systems': int(len(z)), 'n_bins': int(len(bz)),
                    'd_bic_B1': float(d1), 'd_bic_B2': float(d2),
                    'verdict': verdict})
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_18: CaII Low-z Kill Test")
    print("Midnight Initiative -- pre-registered discrimination test")
    print("=" * 60)

    cal = phase_A()
    caii = phase_B(cal)

    data = {
        'experiment': 'exp_18_caii_low_z',
        'initiative': 'midnight',
        'thread': 'photon_archaeology',
        'locked_clock': {'a': A_CLOCK, 'slope': B_DFT, 'floor': N_FLOOR},
        'phase_A': cal,
        'phase_B': caii,
    }
    save_midnight_results('exp_18_caii_low_z', _convert_numpy(data))
