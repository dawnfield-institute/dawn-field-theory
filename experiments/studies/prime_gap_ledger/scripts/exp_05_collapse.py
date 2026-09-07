#!/usr/bin/env python3
"""exp_05 — the collapse delta_q(y) = F(phi(q)/gbar), scored to the seal.

Implements journals/2026-09-07_exp05_registration.md EXACTLY. Every choice below is fixed by that file:
the depths, the seed, the modulus pool, the mechanical split, 18 bins with training-only edges, linear
interpolation between bin centres, the tolerances, the kill scopes and the four vacuity guards of its §4.

R1  the collapse generalises   -- heldout rms <= 1.5 * train rms          (KILL if > 2.0 *)
R2  the variable is phi/gbar   -- rms(phi/gbar) < rms(q/gbar) on heldout, >=3 sigma paired bootstrap
                                                                          (KILL if q/gbar is <= at >=3 sigma)
R3  residual structured by omega(q) -- positive correlation, >=3 sigma, heldout only  (KILL if negative >=3 sigma)

Precedence: KILL -> CONFIRM -> CONVERGED -> INCONCLUSIVE. Nothing here is tuned after the seal.
Writes results/exp_05_collapse_<ts>.json (append-only, timestamped).
"""
import sys, json, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

SEAL = "journals/2026-09-07_exp05_registration.md"
EULER = R.EULER_GAMMA
SEED, W, L, NBINS = 20260909, 40, 2_000_000, 18
DEPTHS = (90, 360, 1440, 5760, 23040, 92160)
POOL = tuple(q for q in range(3, 61) if not (q % 2 == 0 and (q // 2) % 2 == 1))
HELDOUT = tuple(q for i, q in enumerate(POOL) if i % 3 == 2)
TRAIN = tuple(q for q in POOL if q not in HELDOUT)
BOOT, BOOT_SEED = 10_000, 20260909

primes = R.odd_sieve(200_000)


def phi(q):
    n = 1
    for p, a in R.factor_int(q).items():
        n *= (p - 1) * p ** (a - 1)
    return n


def omega(q):
    return len(R.factor_int(q))


def gbar(y):
    return math.exp(EULER) * math.log(y)


def fit_F(x, d, nbins=NBINS):
    """Binned mean on the TRAINING x, linear interpolation between bin centres, clipped at the ends (seal §1)."""
    x, d = np.asarray(x), np.asarray(d)
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    cx, cy = [], []
    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1]) if i < nbins - 1 else (x >= edges[i])
        if m.sum():
            cx.append(0.5 * (edges[i] + edges[i + 1])); cy.append(float(d[m].mean()))
    cx, cy = np.array(cx), np.array(cy)
    return lambda xx: np.interp(np.asarray(xx), cx, cy), cx, cy


# ---- measure every cell -------------------------------------------------------------------------------------
cells, mean_gap_check = [], {}
for y in DEPTHS:
    rng = np.random.default_rng(SEED)
    ps = R.primes_upto(primes, y); per = {q: [] for q in POOL}; gaps_seen = []
    t0 = time.time()
    for o, dr in R.loop_sample(ps, L, W, rng):
        gaps_seen.append(float(R.gaps_of(o).mean()))
        for q in POOL:
            r = R.residues_mod(o, R.n_mod_q_from_draws(q, dr, rng), q)
            per[q].append(R.diagonal_deficit(R.transition_matrix(r, q)))
    gb = gbar(y); obs_gap = float(np.mean(gaps_seen))
    mean_gap_check[y] = dict(measured=obs_gap, e_gamma_log_y=gb, rel_err=abs(obs_gap - gb) / gb)
    for q in POOL:
        v = np.array([z for z in per[q] if z == z])
        if len(v) < 8:
            raise RuntimeError(f"instrument: q={q} y={y} gave only {len(v)} usable windows")
        cells.append(dict(q=q, phi=phi(q), omega=omega(q), y=y, gbar=gb,
                          delta=float(v.mean()), se=float(v.std(ddof=1) / math.sqrt(len(v))),
                          heldout=q in HELDOUT))
    print(f"y={y} done [{time.time()-t0:.0f}s]  mean gap {obs_gap:.4f} vs e^g log y {gb:.4f} "
          f"({100*mean_gap_check[y]['rel_err']:.2f}%)", flush=True)

# seal §7.2 — a depth whose gbar is off by >2% is recorded, not scored
bad_depths = {y for y, c in mean_gap_check.items() if c["rel_err"] > 0.02}
scored = [c for c in cells if c["y"] not in bad_depths]
print(f"\ndepths recorded-not-scored (gbar off >2%): {sorted(bad_depths) or 'none'}")

tr = [c for c in scored if not c["heldout"]]
ho = [c for c in scored if c["heldout"]]
res = dict(script="exp_05_collapse.py", seal=SEAL, generated=ts, seed=SEED, depths=list(DEPTHS),
           nbins=NBINS, windows=W, L=L, train=list(TRAIN), heldout=list(HELDOUT),
           mean_gap_check={str(k): v for k, v in mean_gap_check.items()},
           depths_not_scored=sorted(bad_depths), n_train=len(tr), n_heldout=len(ho), verdicts={})


def run(xname):
    xf = (lambda c: c["phi"] / c["gbar"]) if xname == "phi/gbar" else (lambda c: c["q"] / c["gbar"])
    F, cx, cy = fit_F([xf(c) for c in tr], [c["delta"] for c in tr])
    rtr = np.array([c["delta"] for c in tr]) - F([xf(c) for c in tr])
    rho = np.array([c["delta"] for c in ho]) - F([xf(c) for c in ho])
    return F, cx, cy, rtr, rho


F_phi, cx_p, cy_p, rtr_p, rho_p = run("phi/gbar")
F_q, cx_q, cy_q, rtr_q, rho_q = run("q/gbar")
rms = lambda a: float(np.sqrt((np.asarray(a) ** 2).mean()))

# ---- R1 -----------------------------------------------------------------------------------------------------
train_rms, ho_rms = rms(rtr_p), rms(rho_p)
ratio = ho_rms / train_rms
F_range = float(cy_p.max() - cy_p.min())
guard_flat = F_range > 10 * train_rms                                    # seal §4
r1 = ("KILL" if ratio > 2.0 else "CONFIRM" if ratio <= 1.5 else "INCONCLUSIVE")
if not guard_flat:
    r1 = "INCONCLUSIVE"
res["verdicts"]["R1"] = dict(train_rms=train_rms, heldout_rms=ho_rms, ratio=ratio,
                             F_range=F_range, guard_F_not_flat=bool(guard_flat), verdict=r1)

# ---- R2 -----------------------------------------------------------------------------------------------------
xs_p = np.array([c["phi"] / c["gbar"] for c in scored]); xs_q = np.array([c["q"] / c["gbar"] for c in scored])


def spearman(a, b):
    """Rank correlation, numpy only — the replication kit is numpy-only and must stay that way."""
    def rank(v):
        o = np.argsort(v, kind="mergesort"); r = np.empty(len(v), float); r[o] = np.arange(len(v), dtype=float)
        # average ties so the coefficient is the standard one
        v = np.asarray(v)[o]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[j + 1] == v[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = np.arange(i, j + 1).mean()
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


sp = spearman(xs_p, xs_q)
guard_order = sp <= 0.99                                                 # seal §4
brng = np.random.default_rng(BOOT_SEED); n = len(rho_p); diffs = []
for _ in range(BOOT):
    idx = brng.integers(0, n, n)
    diffs.append(rms(rho_q[idx]) - rms(rho_p[idx]))
diffs = np.array(diffs); sig = float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else 0.0
better = rms(rho_p) < rms(rho_q)
r2 = ("KILL" if (not better and abs(sig) >= 3) else
      "CONFIRM" if (better and sig >= 3) else "INCONCLUSIVE")
if not guard_order:
    r2 = "INCONCLUSIVE"
res["verdicts"]["R2"] = dict(rms_phi=rms(rho_p), rms_q=rms(rho_q), sigma=sig,
                             spearman_orderings=sp, guard_orderings_distinct=bool(guard_order), verdict=r2)

# ---- R3 -----------------------------------------------------------------------------------------------------
om = np.array([c["omega"] for c in ho]); qq = np.array([c["q"] for c in ho])
med_se = float(np.median([c["se"] for c in ho]))
guard_noise = rms(rho_p) > med_se                                        # seal §4
c_om = float(np.corrcoef(om, rho_p)[0, 1]); c_q = float(np.corrcoef(qq, rho_p)[0, 1])
t_om = c_om * math.sqrt((len(om) - 2) / max(1e-12, 1 - c_om ** 2))
r3 = ("KILL" if t_om <= -3 else "CONFIRM" if t_om >= 3 else "INCONCLUSIVE")
if not guard_noise:
    r3 = "INCONCLUSIVE"
res["verdicts"]["R3"] = dict(corr_omega=c_om, t_omega=t_om, corr_q=c_q, median_cell_se=med_se,
                             residual_rms=rms(rho_p), guard_above_noise=bool(guard_noise), verdict=r3)

# ---- sensitivity (recorded, never scored: seal §7.3) --------------------------------------------------------
sens = {}
for nb in (12, 24):
    F2, _, _ = fit_F([c["phi"] / c["gbar"] for c in tr], [c["delta"] for c in tr], nbins=nb)
    sens[nb] = rms(np.array([c["delta"] for c in ho]) - F2([c["phi"] / c["gbar"] for c in ho]))
res["sensitivity_bins_recorded_not_scored"] = sens
res["cells"] = cells
res["F_phi_bins"] = dict(centres=cx_p.tolist(), means=cy_p.tolist())

(RES / f"exp_05_collapse_{ts}.json").write_text(json.dumps(res, indent=1))
print("\n" + "=" * 78)
for k in ("R1", "R2", "R3"):
    print(f"  {k}  {res['verdicts'][k]['verdict']}")
    for kk, vv in res["verdicts"][k].items():
        if kk != "verdict":
            print(f"        {kk} = {vv}")
print("=" * 78)
print(f"score: {sum(1 for k in ('R1','R2','R3') if res['verdicts'][k]['verdict']=='CONFIRM')}/3")
print(f"wrote results/exp_05_collapse_{ts}.json")
