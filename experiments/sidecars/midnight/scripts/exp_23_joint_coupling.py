"""
exp_23 -- The Coupling Law, Within-Scope (Local/Relational) Form (P17 v2)

Midnight Initiative, Thread 1 (Photon Archaeology)

PRE-REGISTERED: journals/2026-06-13_exp23-preregistration.md (same commit).
Registered quantities are NOT computed before that commit (--selftest = the
multi-ion system census only).

------------------------------------------------------------------------------
THE CORRECTION (global -> local)
------------------------------------------------------------------------------
exp_19/21 (and the first exp_23 draft) measured the coupling beta per-ion as a
slope of EW vs the cascade clock across a GLOBAL redshift axis, then compared
betas ACROSS surveys. That is the globalist failure path: raw beta is a
COORDINATE, not an invariant. The exp_19 "portability disaster" (XQR-30 betas
1-2 orders larger than SDSS for the same ion; MgII +0.22 in SDSS vs -0.43 in
XQR-30) is not weak signal -- it is the SAME quantity read in two FRAMES with no
transformation law between their magnitudes. Relativity forbids comparing them
directly; PAC is per-ledger; SEC collapse is relative to the LOCAL gradient.

The relativistically-correct experiment measures the coupling law as a
WITHIN-SCOPE invariant. An XQR-30 SystemID is one absorber = one local ledger =
one frame, and within it many ions are measured simultaneously at one redshift.
The frame-clean observable is the multi-ion abundance PATTERN inside a scope
(differences/within-scope-centered values cancel total column, metallicity,
dust). The coupling is then how an ion's RELATIVE (within-scope) abundance
responds to the LOCAL cascade phase -- aggregated across scopes by local phase
(the tapestry logic, p=1e-12), never by a global coordinate.

This dissolves the "data wall": every multi-ion system is one frame-clean test,
and a single system spans the full IP range (5.99-77.47 eV) at once -- the IP
leverage exp_19 lacked because it measured one ion at a time across global z.

------------------------------------------------------------------------------
THE MODEL (within-scope coupling to LOCAL cascade phase)
------------------------------------------------------------------------------
XQR-30 is a near-single-epoch snapshot (all scopes N ~ 6.3-6.7), so the leverage
is NOT epoch (N) but the LOCAL cascade phase diseq(N) = 1 - 2|N - round(N)|
(1 at a transition, 0 at a trough) -- the SEC-local gradient (M13), which varies
0->~0.44 across the scopes. For system s (scope) with ions {i} present:
  x_i^s  = logN_i^s - mean_j(logN_j^s)       # within-scope-centered abundance
                                             # (frame-invariant: total column cancels)
  c_i    = slope of x_i^s vs diseq_s across scopes containing i   # cascade coupling

The coupling law (P17), local form: an ion's WITHIN-SCOPE relative abundance
responds to the local cascade phase, and the response c_i is monotone in
ionization energy IP -- actively-processing high-IP ions couple to the cascade,
settled low-IP ions do not. This is DFT-discriminating: standard photoionization
predicts an ionization pattern but NO dependence on cascade phase. The ORDERING
is the invariant; absolute c_i are not registered. This is the multi-ion,
IP-resolved generalization of the tapestry result (p=1e-12).

------------------------------------------------------------------------------
REGISTERED PREDICTIONS & DECISION RULE (invariant-registration rule)
------------------------------------------------------------------------------
PRIMARY (relational; decides the verdict):
  R1 cascade-coupling ordering: Spearman(IP, c_i) > 0, one-sided p < 0.05, over
     ions with >= MIN_SYS scopes and >= MIN_DSPAN distinct cascade phases. The
     within-scope cascade coupling orders by ionization energy. Fully
     frame-invariant. Mirrors exp_19 T1 (Spearman(IP, beta) > 0) but measured
     LOCALLY. KILLED if Spearman(IP, c_i) < 0 at p < 0.05 (coupling anti-orders).

SUPPORTING (reported, do not gate; NOT independent of R1):
  R2 aggregate gradient response: the within-scope ionization gradient
     g_s = slope(x_i vs ln IP_i) tracks local cascade phase -- Spearman(diseq, g_s).
     With mean-centering + near-linear-in-ln(IP) patterns, R1 (per-ion ordering) is
     INDUCED by R2 (per-scope gradient response): they are ONE signal viewed two
     ways, counted as one piece of evidence. R1 carries the direction and the kill.
  R3 pattern amplitude: Spearman(diseq, A_s), A_s = within-scope pattern amplitude
     (SEC divergence direction) -- reported, not gated.

ROUTE B -- M6 scoped mediation (independent ordering):
  |c_i| ordered by ionization STAGE (boundary count from the cosmic flow),
  D(ion) = S_max - stage. Stage is atomic structure, NOT phi-rungs -> independent
  of Route A. Agreement = both Spearman(IP, |c|) and Spearman(stage, |c|) > 0.

VERDICT:
  SUPPORTED   = R1 holds.
  KILLED      = R1 reversed (Spearman(IP, c_i) < 0 at p < 0.05).
  INCONCLUSIVE= too few multi-ion scopes / cascade-phase span to resolve R1.

Outputs: results/exp_23_joint_coupling_YYYYMMDD_HHMMSS.json
"""

import sys
import csv
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = MIDNIGHT_ROOT.parent.parent.parent.parent / "data"
sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_rate import PHI, LN_PHI, XI_BALANCE, save_midnight_results, _convert_numpy
from exp_19_ionization_coupling import ION_IP, n_at_z

E_H = 27.2          # eV, Hartree (M-R exp_24); used only in the reported ladder
MIN_IONS = 3        # ions per scope to define a within-scope pattern with IP leverage
MIN_SYS = 15        # scopes containing an ion for its coupling to be estimated
MIN_DSPAN = 4       # distinct cascade-phase values required to fit a slope
N_BOOT = 1000

# Ionization stage (roman numeral value) -- atomic structure, Route B only
ION_STAGE = {
    'AlII': 2, 'CaII': 2, 'MgII': 2, 'FeII': 2, 'SiII': 2, 'CII': 2,
    'AlIII': 3, 'SiIV': 4, 'CIV': 4, 'NV': 5,
}


def diseq_at_N(N):
    """Local cascade phase: 1 at a transition (integer N), 0 at a trough."""
    return max(0.0, 1.0 - 2.0 * abs(N - round(N)))


# ============================================================
# Load XQR-30 as multi-ion SCOPES (SystemID = one absorber = one frame)
# ============================================================

def load_xqr30_scopes():
    """Group XQR-30 by SystemID -> {sysid: {'z':, 'ions': {ion: logN}}}.

    System-level transition rows (SystemID is a digit, Species like 'MgII_2796')
    carry the ion's logN on the primary transition row. logN is the abundance;
    within-scope differences of logN are frame-invariant (total column cancels).
    """
    scopes = {}
    path = DATA_ROOT / "xqr30" / "xqr30_merged_catalog.csv"
    with open(str(path), 'r') as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#'):
                continue
            sysid = row[0].strip()
            if not sysid.isdigit():
                continue  # component rows (ComponentID set, SystemID '-') skipped
            try:
                z = float(row[2])
                ion = row[3].strip().split('_')[0]
                logN = float(row[7])
                logN_flag = row[9].strip().upper()
            except (ValueError, IndexError):
                continue
            if ion not in ION_IP:
                continue
            # drop upper-limit (U) and saturated (S) logN -- not clean measurements
            if 'U' in logN_flag or 'S' in logN_flag:
                continue
            sc = scopes.setdefault(sysid, {'z': z, 'ions': {}})
            # keep the first (primary-transition) logN per ion in the scope
            sc['ions'].setdefault(ion, logN)
            sc['z'] = min(sc['z'], z) if sc['z'] else z
    # keep multi-ion scopes only
    return {s: d for s, d in scopes.items() if len(d['ions']) >= MIN_IONS}


# ============================================================
# Within-scope coupling (the local invariant)
# ============================================================

def within_scope_table(scopes):
    """Per-ion lists of (diseq_s, x_i^s); x is within-scope-centered logN.

    Also returns per-scope (diseq, gradient g_s, pattern amplitude A_s) for the
    supporting R2/R3 aggregate tests.
    """
    table = {}
    scope_rows = []
    for sysid, d in scopes.items():
        ions = d['ions']
        N_s = n_at_z(d['z'])
        dq = diseq_at_N(N_s)
        vals = np.array(list(ions.values()), float)
        mean = float(np.mean(vals))
        ip_arr, x_arr = [], []
        for ion, logN in ions.items():
            x = logN - mean
            table.setdefault(ion, {'diseq': [], 'x': []})
            table[ion]['diseq'].append(dq)
            table[ion]['x'].append(x)
            ip_arr.append(ION_IP[ion]); x_arr.append(x)
        # within-scope ionization gradient (slope of centered logN vs ln IP)
        g_s = float(np.polyfit(np.log(ip_arr), x_arr, 1)[0]) if len(ip_arr) >= 3 else np.nan
        A_s = float(np.std(x_arr))   # within-scope pattern amplitude
        scope_rows.append({'sysid': sysid, 'z': d['z'], 'N': N_s, 'diseq': dq,
                           'n_ions': len(ions), 'g_s': g_s, 'A_s': A_s})
    for ion in table:
        table[ion]['diseq'] = np.array(table[ion]['diseq'])
        table[ion]['x'] = np.array(table[ion]['x'])
    return table, scope_rows


def couplings(table, rng):
    """Cascade coupling c_i = slope(x_i vs diseq) per ion, with bootstrap CI."""
    out = {}
    for ion, d in table.items():
        dq, x = d['diseq'], d['x']
        n_span = len(np.unique(np.round(dq, 3)))
        if len(dq) < MIN_SYS or n_span < MIN_DSPAN:
            out[ion] = {'IP': ION_IP[ion], 'stage': ION_STAGE[ion], 'n': int(len(dq)),
                        'n_span': int(n_span), 'excluded': True}
            continue
        c = float(np.polyfit(dq, x, 1)[0])
        boots = []
        for _ in range(N_BOOT):
            idx = rng.randint(0, len(dq), len(dq))
            if len(np.unique(np.round(dq[idx], 3))) < 2:
                continue
            boots.append(np.polyfit(dq[idx], x[idx], 1)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        out[ion] = {'IP': ION_IP[ion], 'stage': ION_STAGE[ion], 'n': int(len(dq)),
                    'n_span': int(n_span), 'c': c, 'c_CI95': [float(lo), float(hi)],
                    'excluded': False}
    return out


# ============================================================
# R2/R3: aggregate gradient response + PAC stability (supporting, reported)
# ============================================================

def aggregate_tests(scope_rows):
    """R2: within-scope ionization gradient g_s vs local cascade phase diseq.
    R3: within-scope pattern amplitude A_s stability across cascade phase (PAC).
    """
    dq = np.array([r['diseq'] for r in scope_rows])
    g = np.array([r['g_s'] for r in scope_rows])
    A = np.array([r['A_s'] for r in scope_rows])
    ok = np.isfinite(g)
    res = {'n_scopes': int(np.sum(ok))}
    if np.sum(ok) >= 10 and len(np.unique(np.round(dq[ok], 3))) >= 4:
        rho_g, p_g = spearmanr(dq[ok], g[ok])
        res['R2_gradient'] = {'rho': float(rho_g), 'p': float(p_g)}
        rho_A, p_A = spearmanr(dq[ok], A[ok])
        # pattern amplitude vs cascade phase (SEC divergence direction) -- reported
        res['R3_amplitude'] = {'rho': float(rho_A), 'p': float(p_A)}
    return res


# ============================================================
# Route B: M6 scoped mediation (independent, stage-based ordering)
# ============================================================

def route_b(coup):
    a = float(np.exp(-XI_BALANCE))
    ions = [i for i, d in coup.items() if not d['excluded']]
    if len(ions) < 3:
        return {'PASS': None, 'note': 'need >=3 ions with estimated coupling',
                'a_attenuation': a, 'ions': ions}
    s_max = max(ION_STAGE[i] for i in ions)
    ip_order = [ION_IP[i] for i in ions]                  # Route A regressor
    stage_pred = [a ** (s_max - ION_STAGE[i]) for i in ions]  # Route B magnitude
    c_obs = [abs(coup[i]['c']) for i in ions]             # observed |coupling|
    rho_A, pA = spearmanr(ip_order, c_obs)
    rho_B, pB = spearmanr(stage_pred, c_obs)
    rho_AB, _ = spearmanr(ip_order, stage_pred)
    return {'a_attenuation': a, 's_max': int(s_max), 'ions': ions,
            'rho_A_obs': float(rho_A), 'p_A_obs_one': float(pA / 2 if rho_A > 0 else 1),
            'rho_B_obs': float(rho_B), 'p_B_obs_one': float(pB / 2 if rho_B > 0 else 1),
            'rho_AB': float(rho_AB),
            'PASS': bool(rho_A > 0 and (pA / 2) < 0.05 and rho_B > 0 and (pB / 2) < 0.05)}


# ============================================================
# Main
# ============================================================

def run():
    scopes = load_xqr30_scopes()
    print(f"\n  Multi-ion scopes (>= {MIN_IONS} ions): {len(scopes)}")
    table, scope_rows = within_scope_table(scopes)
    rng = np.random.RandomState(20260613)
    coup = couplings(table, rng)

    print("\n  Within-scope cascade coupling c_i = slope(centered logN vs diseq):")
    usable = []
    for ion in sorted(coup, key=lambda i: ION_IP[i]):
        d = coup[ion]
        if d['excluded']:
            print(f"    {ion:<5} IP={d['IP']:6.2f}  EXCLUDED (n={d['n']}, span={d['n_span']})")
        else:
            print(f"    {ion:<5} IP={d['IP']:6.2f}  c={d['c']:+.4f} "
                  f"CI95=[{d['c_CI95'][0]:+.4f},{d['c_CI95'][1]:+.4f}]  "
                  f"(n={d['n']}, span={d['n_span']})")
            usable.append(ion)

    # R1: coupling orders by ionization energy (PRIMARY invariant)
    if len(usable) >= 3:
        ips = [coup[i]['IP'] for i in usable]
        cs = [coup[i]['c'] for i in usable]
        rho, p2 = spearmanr(ips, cs)
        p1 = p2 / 2 if rho > 0 else 1 - p2 / 2
        r1 = bool(rho > 0 and p1 < 0.05)
        r1_killed = bool(rho < 0 and (p2 / 2) < 0.05)
        r1d = {'rho': float(rho), 'p_one_sided': float(p1), 'n_ions': len(usable),
               'ions': usable}
    else:
        r1, r1_killed = None, False
        r1d = {'n_ions': len(usable), 'note': 'need >=3 ions with estimated coupling'}
    print(f"\n  R1 coupling ordering Spearman(IP, c): {r1d.get('rho')}, "
          f"p_one={r1d.get('p_one_sided')} -> {r1}")

    # R2/R3: aggregate gradient response + PAC stability (supporting)
    agg = aggregate_tests(scope_rows)
    print(f"  R2 gradient g_s vs diseq (= R1 per-scope): {agg.get('R2_gradient')}")
    print(f"  R3 pattern amplitude vs diseq: {agg.get('R3_amplitude')}")

    # Route B
    rb = route_b(coup)
    print(f"  Route B (M6): rho_A_obs={rb.get('rho_A_obs')}, rho_B_obs={rb.get('rho_B_obs')} "
          f"-> PASS={rb['PASS']}")

    # Verdict
    if r1 is True:
        verdict = 'SUPPORTED'
    elif r1_killed:
        verdict = 'KILLED'
    else:
        verdict = 'INCONCLUSIVE'
    print(f"\n  VERDICT (registered rule): {verdict}")

    return {
        'experiment': 'exp_23_joint_coupling',
        'initiative': 'midnight',
        'form': 'within-scope (local/relational) coupling',
        'registered': {
            'observable': 'c_i = slope(within-scope-centered logN vs N), per ion',
            'scope': 'XQR-30 SystemID (one absorber = one frame)',
            'MIN_IONS': MIN_IONS, 'MIN_SYS': MIN_SYS, 'MIN_DSPAN': MIN_DSPAN,
        },
        'n_scopes': len(scopes),
        'couplings': coup,
        'R1_coupling_ordering': {'PASS': r1, 'killed': r1_killed, **r1d},
        'R2_R3_aggregate': agg,
        'route_B': rb,
        'verdict': verdict,
    }


def selftest():
    """Multi-ion scope census ONLY -- no registered quantities."""
    print("SELFTEST: multi-ion scope census (no registered quantities)")
    scopes = load_xqr30_scopes()
    print(f"\n  Multi-ion scopes (>= {MIN_IONS} ions): {len(scopes)}")
    # ion occurrence + cascade-phase span
    occ = {}
    for d in scopes.values():
        dq = diseq_at_N(n_at_z(d['z']))
        for ion in d['ions']:
            occ.setdefault(ion, {'n': 0, 'dq': []})
            occ[ion]['n'] += 1
            occ[ion]['dq'].append(dq)
    print(f"\n  {'ion':<6}{'IP':>7}{'stage':>6}{'#scopes':>9}{'#phases':>9}{'diseq range':>16}")
    n_usable = 0
    for ion in sorted(occ, key=lambda i: ION_IP[i]):
        dqs = occ[ion]['dq']
        nspan = len(np.unique(np.round(dqs, 3)))
        usable = occ[ion]['n'] >= MIN_SYS and nspan >= MIN_DSPAN
        n_usable += usable
        print(f"  {ion:<6}{ION_IP[ion]:>7.2f}{ION_STAGE[ion]:>6}{occ[ion]['n']:>9}"
              f"{nspan:>9}{f'{min(dqs):.2f}-{max(dqs):.2f}':>16}"
              f"{'  <-usable' if usable else ''}")
    # scope size distribution
    sizes = [len(d['ions']) for d in scopes.values()]
    print(f"\n  Scope sizes: min={min(sizes)}, median={int(np.median(sizes))}, max={max(sizes)}")
    print(f"  Ions with >= {MIN_SYS} scopes and >= {MIN_DSPAN} phases: {n_usable} "
          f"(R1 needs >= 3)")
    print("  OK")


if __name__ == '__main__':
    print("=" * 64)
    print("exp_23: Coupling Law, Within-Scope (Local) Form (P17 v2)")
    print("Midnight Initiative -- pre-registered")
    print("=" * 64)
    if '--selftest' in sys.argv:
        selftest()
    else:
        data = run()
        save_midnight_results('exp_23_joint_coupling', _convert_numpy(data))
