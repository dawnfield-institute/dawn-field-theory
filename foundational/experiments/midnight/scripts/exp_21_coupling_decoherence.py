"""
exp_21 -- The Coupling Law, Decoherence-Framed (P17.1-P17.4)

Midnight Initiative, Thread 1 (Photon Archaeology)

PRE-REGISTERED: journals/2026-06-11_exp21-exp22-preregistration.md, derivation
in journals/2026-06-11_p17-coupling-law-derivation.md (same commit). Registered
quantities not computed before that commit (--selftest = loaders only).

Model (P17, phi-split ladder, parameter-free shape):
  monitoring capacity P(IP) = 1 - (IP/E_H)^2 / phi^2   for IP <= E_H, else 0
  |beta|(IP) = A_panel / (1 + r * P(IP))
  free: A_S, A_X, r;  E_H = 27.2 eV fixed (free only for the T1 knee test)

Tests (registered):
  T1: knee at the Hartree -- CI95(E_H free) contains 27.2 AND dBIC(fixed-free) <= 2
  T2: quadratic envelope beats/ties logistic capacity at same param count (dBIC <= 2)
  T3: per-panel E_H CI95s overlap (epoch invariance); report distance from HeI 24.6
  T4: turnover structure -- T4a FeII flip within dz<=0.5 of MgII flip;
      T4b CIV no flip either survey; T4c SiIV no flip (XQR-30).
      MgII flip anchors N_H (reported, not scored).

Verdict: SUPPORTED = T1&T2&T4a&T4b. KILLED = free-knee CI95 excludes 27.2 with
adequate fit, OR CIV flips. Else INCONCLUSIVE.

Outputs: results/exp_21_coupling_decoherence_YYYYMMDD_HHMMSS.json
"""

import sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import spearmanr

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_rate import PHI, LN_PHI, save_midnight_results, _convert_numpy
from exp_19_ionization_coupling import (
    ION_IP, n_at_z, coupling_beta, load_xqr30, load_caii,
    load_dr16_mgii, load_dr16_feii, load_dr12_civ)

E_H = 27.2          # eV, fixed by M-R exp_24 (alpha^2 m_e c^2)
HEI_EDGE = 24.6     # eV, the UV-background alternative's landmark
N_BOOT = 500


# ============================================================
# Panels
# ============================================================

def build_panels():
    """Panel S (SDSS, low/mid z) and Panel X (XQR-30, high z)."""
    panels = {'S': [], 'X': []}
    z, w = load_caii();      panels['S'].append(('CaII', z, w))
    z, w = load_dr16_mgii(); panels['S'].append(('MgII', z, w))
    z, w = load_dr16_feii(); panels['S'].append(('FeII', z, w))
    z, w = load_dr12_civ();  panels['S'].append(('CIV', z, w))
    xqr = load_xqr30()
    for sp in ('MgII', 'FeII', 'SiII', 'SiIV', 'CIV'):
        if sp in xqr:
            zx, wx = xqr[sp]
            panels['X'].append((sp, zx, wx))
    return panels


def measure_betas(panels):
    """Per (ion, panel) beta with bootstrap draws (exp_19 locked metric)."""
    rng = np.random.RandomState(20260611)
    rows = []
    for pname, items in panels.items():
        for ion, z, w in items:
            beta, nbins = coupling_beta(z, w)
            if beta is None:
                rows.append({'panel': pname, 'ion': ion, 'excluded': True,
                             'n': int(len(z))})
                continue
            boots = []
            for _ in range(N_BOOT):
                b, _ = coupling_beta(z, w, rng=rng)
                if b is not None:
                    boots.append(b)
            rows.append({'panel': pname, 'ion': ion, 'excluded': False,
                         'n': int(len(z)), 'n_bins': nbins,
                         'IP_eV': ION_IP[ion], 'beta': float(beta),
                         'beta_boots': np.array(boots)})
    return rows


# ============================================================
# Capacity models and fits
# ============================================================

def cap_envelope(ip, eh):
    p = 1.0 - (ip / eh)**2 / PHI**2
    return np.where(ip <= eh, np.maximum(p, 0.0), 0.0)


def cap_logistic(ip, eh):
    """Logistic alternative, same parameter count (center fixed at eh,
    width fixed at ln(phi) in log-energy so counts match the envelope)."""
    x = (np.log(ip) - np.log(eh)) / LN_PHI
    return 1.0 / (1.0 + np.exp(x * 4.0))   # slope-matched logistic, no free width


def fit_absbeta(rows, capacity, eh_free=False):
    """Fit |beta| = A_panel / (1 + r*P(IP)). Returns params, BIC, ss_res."""
    pts = [(r['panel'], r['IP_eV'], abs(r['beta'])) for r in rows
           if not r.get('excluded')]
    panels = sorted({p for p, _, _ in pts})
    y = np.array([b for _, _, b in pts])
    # log-space residuals (betas span decades across panels)
    ly = np.log(np.maximum(y, 1e-6))

    def model(theta):
        if eh_free:
            amps = theta[:len(panels)]; r_par = theta[len(panels)]
            eh = theta[len(panels) + 1]
        else:
            amps = theta[:len(panels)]; r_par = theta[len(panels)]; eh = E_H
        pred = []
        for p, ip, _ in pts:
            a = amps[panels.index(p)]
            pred.append(a / (1.0 + abs(r_par) * float(capacity(np.array([ip]), eh)[0])))
        return np.log(np.maximum(np.array(pred), 1e-9))

    def loss(theta):
        return float(np.sum((ly - model(theta))**2))

    k = len(panels) + 1 + (1 if eh_free else 0)
    best = None
    for r0 in (1.0, 10.0, 100.0):
        for eh0 in ((20.0, 27.2, 40.0) if eh_free else (E_H,)):
            th0 = [np.exp(np.mean(ly))] * len(panels) + [r0] + ([eh0] if eh_free else [])
            res = minimize(loss, th0, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-7, 'fatol': 1e-10})
            if best is None or res.fun < best.fun:
                best = res
    n = len(pts)
    ss = best.fun
    bic = n * np.log(max(ss, 1e-12) / n) + k * np.log(n)
    out = {'k': k, 'n': n, 'ss_res': float(ss), 'BIC': float(bic),
           'amps': [float(a) for a in best.x[:len(panels)]],
           'r': float(abs(best.x[len(panels)]))}
    if eh_free:
        out['E_H_fit'] = float(best.x[len(panels) + 1])
    return out


def bootstrap_eh(rows, capacity, n_rounds=200):
    """CI95 on free-knee E_H by resampling per-ion betas from bootstrap draws."""
    rng = np.random.RandomState(11)
    ehs = []
    usable = [r for r in rows if not r.get('excluded')]
    for _ in range(n_rounds):
        sampled = []
        for r in usable:
            r2 = dict(r)
            r2['beta'] = float(r['beta_boots'][rng.randint(0, len(r['beta_boots']))])
            sampled.append(r2)
        try:
            f = fit_absbeta(sampled, capacity, eh_free=True)
            ehs.append(np.clip(f['E_H_fit'], 1.0, 500.0))
        except Exception:
            continue
    ehs = np.array(ehs)
    return float(np.percentile(ehs, 2.5)), float(np.percentile(ehs, 97.5)), ehs


# ============================================================
# T4: rolling-window sign analysis
# ============================================================

def sign_flip_z(z, w, n_windows=6):
    """Rolling z-window betas; return (window_centers, signs, flip_z or None)."""
    edges = np.quantile(z, np.linspace(0, 1, n_windows + 1))
    centers, signs = [], []
    for i in range(n_windows):
        m = (z >= edges[i]) & (z < edges[i + 1]) if i < n_windows - 1 else (z >= edges[i])
        if m.sum() < 60:
            continue
        b, nb = coupling_beta(z[m], w[m])
        if b is None:
            # small windows: fall back to direct binned-median slope sign
            zz, ww = z[m], w[m]
            sub = np.quantile(zz, np.linspace(0, 1, 5))
            bN, bw = [], []
            for j in range(4):
                mm = (zz >= sub[j]) & (zz < sub[j + 1]) if j < 3 else (zz >= sub[j])
                if mm.sum() < 10:
                    continue
                bN.append(n_at_z(np.median(zz[mm]))); bw.append(np.median(ww[mm]))
            if len(bN) < 3:
                continue
            b = float(np.polyfit(bN, bw, 1)[0])
        centers.append(float(np.median(z[m])))
        signs.append(float(np.sign(b)))
    flip = None
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1] and signs[i] != 0:
            flip = (centers[i] + centers[i + 1]) / 2
            break
    return centers, signs, flip


def test_T4(panels):
    print("\n  T4: turnover structure (rolling-window sign analysis)")
    flips = {}
    for pname, items in panels.items():
        for ion, z, w in items:
            c, s, flip = sign_flip_z(z, w)
            flips[(ion, pname)] = {'centers': c, 'signs': s, 'flip_z': flip}
            print(f"    {ion:<5}[{pname}] signs={['+' if x > 0 else '-' for x in s]} "
                  f"flip_z={flip if flip else 'none'}")

    # Cross-survey flip bracket for MgII/FeII: SDSS sign at top window vs XQR-30 at bottom
    def cross_flip(ion):
        s_panel = flips.get((ion, 'S')); x_panel = flips.get((ion, 'X'))
        within = [p['flip_z'] for p in (s_panel, x_panel) if p and p['flip_z']]
        if within:
            return within[0]
        if s_panel and x_panel and s_panel['signs'] and x_panel['signs']:
            if s_panel['signs'][-1] != x_panel['signs'][0]:
                return (s_panel['centers'][-1] + x_panel['centers'][0]) / 2
        return None

    mg_flip = cross_flip('MgII')
    fe_flip = cross_flip('FeII')
    t4a = bool(mg_flip and fe_flip and abs(mg_flip - fe_flip) <= 0.5)
    civ_flips = [flips[k]['flip_z'] for k in flips if k[0] == 'CIV' and flips[k]['flip_z']]
    t4b = len(civ_flips) == 0
    siiv = flips.get(('SiIV', 'X'))
    t4c = bool(siiv is None or siiv['flip_z'] is None)

    # Anchor N_H from MgII flip (reported, not scored)
    n_h = None
    if mg_flip:
        n_h = n_at_z(mg_flip) + np.log(E_H / ION_IP['MgII']) / LN_PHI
    print(f"    MgII flip z = {mg_flip}, FeII flip z = {fe_flip}")
    print(f"    T4a FeII within dz<=0.5 of MgII: {'PASS' if t4a else 'FAIL'}")
    print(f"    T4b CIV monotone (no flip): {'PASS' if t4b else 'FAIL'}")
    print(f"    T4c SiIV monotone: {'PASS' if t4c else 'FAIL'}")
    if n_h:
        print(f"    Anchor N_H = {n_h:.2f} (reported, not scored)")
    return {'flips': {f"{k[0]}_{k[1]}": v for k, v in flips.items()},
            'mgii_flip_z': mg_flip, 'feii_flip_z': fe_flip,
            'N_H_anchor': float(n_h) if n_h else None,
            'T4a': t4a, 'T4b': t4b, 'T4c': t4c}


# ============================================================
# Main
# ============================================================

def run():
    panels = build_panels()
    rows = measure_betas(panels)
    print("\n  Per-(ion,panel) beta:")
    for r in rows:
        if r.get('excluded'):
            print(f"    {r['ion']:<5}[{r['panel']}] EXCLUDED n={r['n']}")
        else:
            print(f"    {r['ion']:<5}[{r['panel']}] IP={r['IP_eV']:6.2f} "
                  f"beta={r['beta']:+.4f}")

    # T1: knee at the Hartree
    fit_fixed = fit_absbeta(rows, cap_envelope, eh_free=False)
    fit_free = fit_absbeta(rows, cap_envelope, eh_free=True)
    lo, hi, eh_dist = bootstrap_eh(rows, cap_envelope)
    dbic = fit_fixed['BIC'] - fit_free['BIC']
    t1 = bool(lo <= E_H <= hi and dbic <= 2.0)
    print(f"\n  T1: E_H free = {fit_free.get('E_H_fit', float('nan')):.1f} eV, "
          f"CI95=[{lo:.1f},{hi:.1f}]; dBIC(fixed-free)={dbic:+.2f} "
          f"-> {'PASS' if t1 else 'FAIL'}")

    # T2: envelope vs logistic (both with E_H fixed)
    fit_logi = fit_absbeta(rows, cap_logistic, eh_free=False)
    dbic2 = fit_fixed['BIC'] - fit_logi['BIC']
    t2 = bool(dbic2 <= 2.0)
    print(f"  T2: BIC envelope={fit_fixed['BIC']:.2f} vs logistic={fit_logi['BIC']:.2f} "
          f"(d={dbic2:+.2f}) -> {'PASS' if t2 else 'FAIL'}")

    # T3: per-panel free knee
    panel_eh = {}
    for pname in ('S', 'X'):
        sub = [r for r in rows if r.get('panel') == pname]
        try:
            plo, phi_, _ = bootstrap_eh(sub, cap_envelope, n_rounds=120)
            panel_eh[pname] = [plo, phi_]
        except Exception:
            panel_eh[pname] = None
    if panel_eh.get('S') and panel_eh.get('X'):
        s, x = panel_eh['S'], panel_eh['X']
        t3 = bool(max(s[0], x[0]) <= min(s[1], x[1]))
    else:
        t3 = False
    print(f"  T3: per-panel E_H CI95 S={panel_eh.get('S')}, X={panel_eh.get('X')} "
          f"-> {'PASS' if t3 else 'FAIL'}  (HeI alternative at {HEI_EDGE} eV)")

    # T4
    t4 = test_T4(panels)

    # Registered verdict
    killed = (not (lo <= E_H <= hi) and fit_free['ss_res'] < fit_fixed['ss_res'] * 2) \
             or (not t4['T4b'])
    if t1 and t2 and t4['T4a'] and t4['T4b']:
        verdict = 'SUPPORTED'
    elif killed:
        verdict = 'KILLED'
    else:
        verdict = 'INCONCLUSIVE'
    score = sum([t1, t2, t3, t4['T4a'] and t4['T4b'] and t4['T4c']])
    print(f"\n  VERDICT (registered rule): {verdict}   score {score}/4")

    for r in rows:
        r.pop('beta_boots', None)
    return {
        'experiment': 'exp_21_coupling_decoherence',
        'initiative': 'midnight',
        'registered': {'E_H': E_H, 'model': 'A/(1+r*P), P=1-(IP/E_H)^2/phi^2'},
        'betas': rows,
        'T1': {'fit_fixed': fit_fixed, 'fit_free': fit_free,
               'E_H_CI95': [lo, hi], 'dBIC': float(dbic), 'PASS': t1},
        'T2': {'fit_logistic': fit_logi, 'dBIC': float(dbic2), 'PASS': t2},
        'T3': {'panel_E_H_CI95': panel_eh, 'PASS': t3, 'HeI_eV': HEI_EDGE},
        'T4': t4,
        'verdict': verdict,
        'score': f"{score}/4",
    }


def selftest():
    print("SELFTEST: panel construction only")
    panels = build_panels()
    for pname, items in panels.items():
        for ion, z, w in items:
            print(f"  {ion:<5}[{pname}] n={len(z):6d} z={z.min():.2f}-{z.max():.2f}")
    print("  OK")


if __name__ == '__main__':
    print("=" * 60)
    print("exp_21: Coupling Law, Decoherence-Framed (P17.1-P17.4)")
    print("Midnight Initiative -- pre-registered")
    print("=" * 60)
    if '--selftest' in sys.argv:
        selftest()
    else:
        data = run()
        save_midnight_results('exp_21_coupling_decoherence', _convert_numpy(data))
