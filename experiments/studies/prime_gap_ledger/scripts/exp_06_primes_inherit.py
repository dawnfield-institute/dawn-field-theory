#!/usr/bin/env python3
"""exp_06 — do the primes inherit the loop's collapse? Scored to the seal.

Implements journals/2026-09-07_exp06_registration.md EXACTLY.

    prediction:  delta_q(primes) = F( phi(q) / gbar(y_eff) )        ZERO free parameters

F is the sealed exp_05 curve, fitted on LOOP cells only and not refitted, rescaled or shifted here.
y_eff is solved from the arc's measured density (round 3's method), not fitted. Nothing in this round is fitted.

R1  the primes inherit F   -- rms(measured - predicted) <= 2.0 * 0.03459   (KILL if > 3.0 *)
R2  the depth shift does the work -- gbar(y) in place of gbar(y_eff) must be WORSE, >=3 sigma paired bootstrap
R3  the omega(q) structure persists, sign registered in advance as POSITIVE, >=3 sigma

Precedence: KILL -> CONFIRM -> CONVERGED -> INCONCLUSIVE. Writes results/exp_06_primes_inherit_<ts>.json.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

SEAL = "journals/2026-09-07_exp06_registration.md"
SEED, BOOT = 20260910, 10_000
DECADES = (7, 8, 9)
POOL = tuple(q for q in range(3, 61) if not (q % 2 == 0 and (q // 2) % 2 == 1))
EXP05_RMS = 0.03459265462394738                   # the sealed reference; tolerances are multiples of it
TOL_CONFIRM, TOL_KILL = 2.0 * EXP05_RMS, 3.0 * EXP05_RMS

d05 = json.loads(sorted(RES.glob("exp_05_collapse_*.json"))[-1].read_text())
CX = np.array(d05["F_phi_bins"]["centres"]); CY = np.array(d05["F_phi_bins"]["means"])
primes = R.odd_sieve(200_000)


def phi(q):
    n = 1
    for p, a in R.factor_int(q).items():
        n *= (p - 1) * p ** (a - 1)
    return n


def omega(q):
    return len(R.factor_int(q))


def gbar(y):
    return math.exp(R.EULER_GAMMA) * math.log(y)


def F(x):
    """The sealed exp_05 curve: linear interpolation between bin centres, clipped at the ends."""
    return float(np.interp(x, CX, CY))


cells = []
for m in DECADES:
    N0 = 10 ** m; y = math.isqrt(2 * N0) + 1; ps = R.primes_upto(primes, y)
    chunk = max(N0 // 10, 1); t0 = time.time()
    rd = R.chunked_read(N0, N0, chunk, ps, POOL)
    dens = rd["density"]
    ye = R.y_eff_from_density(dens, primes); y_eff = ye["y_eff"]
    gb_eff, gb_y = gbar(y_eff), gbar(y)
    if rd["transitions"][POOL[0]] != rd["n"] - 1:                     # loud, never silent (G3's invariant)
        raise RuntimeError(f"carry broken at m={m}: transitions != n-1")
    for q in POOL:
        d = R.diagonal_deficit(rd["T"][q])
        se = float(R.detrended_se_log(rd["parts"][q], rd["logpos"]))
        x_eff, x_y = phi(q) / gb_eff, phi(q) / gb_y
        cells.append(dict(m=m, q=q, phi=phi(q), omega=omega(q), y=y, y_eff=y_eff,
                          density=dens, delta=float(d), se=se,
                          x_eff=x_eff, x_y=x_y, pred_eff=F(x_eff), pred_y=F(x_y),
                          in_domain=bool(CX.min() <= x_eff <= CX.max()),
                          bracket=dict(p_lo=ye["p_lo"], p_hi=ye["p_hi"], mismatch_log=ye["mismatch_log"])))
    print(f"m={m} done [{time.time()-t0:.0f}s]  n={rd['n']}  y={y} y_eff={y_eff}", flush=True)

# scoring basis (seal §1/§4): inside F's domain, and not saturated at 1 across every decade
by_q = {}
for c in cells:
    by_q.setdefault(c["q"], []).append(c)
saturated = {q for q, cs in by_q.items() if all(abs(c["delta"] - 1.0) <= c["se"] for c in cs)}
scored = [c for c in cells if c["in_domain"] and c["q"] not in saturated]
out_of_domain = sorted({c["q"] for c in cells if not c["in_domain"]})

res_eff = np.array([c["delta"] - c["pred_eff"] for c in scored])
res_y = np.array([c["delta"] - c["pred_y"] for c in scored])
rms = lambda a: float(np.sqrt((np.asarray(a) ** 2).mean()))

res = dict(script="exp_06_primes_inherit.py", seal=SEAL, generated=ts, seed=SEED, decades=list(DECADES),
           exp05_reference_rms=EXP05_RMS, tol_confirm=TOL_CONFIRM, tol_kill=TOL_KILL,
           n_cells=len(cells), n_scored=len(scored),
           recorded_not_scored=dict(out_of_domain=out_of_domain, saturated=sorted(saturated)),
           verdicts={})

# ---- R1 ------------------------------------------------------------------------------------------------------
r1_rms = rms(res_eff)
xs = [c["x_eff"] for c in scored]
F_span = float(max(F(x) for x in xs) - min(F(x) for x in xs))
guard_flat = F_span > 10 * EXP05_RMS
r1 = "KILL" if r1_rms > TOL_KILL else "CONFIRM" if r1_rms <= TOL_CONFIRM else "INCONCLUSIVE"
if not guard_flat:
    r1 = "INCONCLUSIVE"
res["verdicts"]["R1"] = dict(rms=r1_rms, tol_confirm=TOL_CONFIRM, tol_kill=TOL_KILL,
                             F_span_over_scored=F_span, guard_F_not_flat=bool(guard_flat),
                             mean_signed_residual=float(res_eff.mean()), verdict=r1)

# ---- R2 ------------------------------------------------------------------------------------------------------
shifts = {}
for m in DECADES:                                   # match on the decade, never on list position
    c = next(c for c in cells if c["m"] == m)
    shifts[m] = abs(math.log(c["y_eff"]) / math.log(c["y"]) - 1)
guard_shift = all(v > 0.02 for v in shifts.values())
rng = np.random.default_rng(SEED); n = len(res_eff); diffs = []
for _ in range(BOOT):
    idx = rng.integers(0, n, n)
    diffs.append(rms(res_y[idx]) - rms(res_eff[idx]))
diffs = np.array(diffs); sig = float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else 0.0
better = rms(res_eff) < rms(res_y)
r2 = "KILL" if (not better and abs(sig) >= 3) else "CONFIRM" if (better and sig >= 3) else "INCONCLUSIVE"
if not guard_shift:
    r2 = "INCONCLUSIVE"
res["verdicts"]["R2"] = dict(rms_y_eff=rms(res_eff), rms_y=rms(res_y), sigma=sig,
                             log_shift_per_decade={str(k): v for k, v in shifts.items()},
                             guard_shift_resolvable=bool(guard_shift), verdict=r2)

# ---- R3 ------------------------------------------------------------------------------------------------------
om = np.array([c["omega"] for c in scored]); qq = np.array([c["q"] for c in scored])
med_se = float(np.median([c["se"] for c in scored]))
guard_noise = rms(res_eff) > med_se
c_om = float(np.corrcoef(om, res_eff)[0, 1]); c_q = float(np.corrcoef(qq, res_eff)[0, 1])
t_om = c_om * math.sqrt((len(om) - 2) / max(1e-12, 1 - c_om ** 2))
r3 = "KILL" if t_om <= -3 else "CONFIRM" if t_om >= 3 else "INCONCLUSIVE"
if not guard_noise:
    r3 = "INCONCLUSIVE"
res["verdicts"]["R3"] = dict(corr_omega=c_om, t_omega=t_om, corr_q=c_q, median_cell_se=med_se,
                             residual_rms=rms(res_eff), guard_above_noise=bool(guard_noise), verdict=r3)

# seal §7.1 — bin-edge proximity, recorded not scored
dist = [min(abs(c["x_eff"] - float(cc)) for cc in CX) for c in scored]
res["bin_edge_check_recorded_not_scored"] = dict(corr_residual_vs_distance=float(np.corrcoef(dist, np.abs(res_eff))[0, 1]))
res["cells"] = cells

(RES / f"exp_06_primes_inherit_{ts}.json").write_text(json.dumps(res, indent=1, default=str))
print("\n" + "=" * 78)
print(f"  scored {len(scored)} of {len(cells)} cells "
      f"(out of F's domain: {out_of_domain}; saturated: {sorted(saturated)})")
for k in ("R1", "R2", "R3"):
    print(f"  {k}  {res['verdicts'][k]['verdict']}")
    for kk, vv in res["verdicts"][k].items():
        if kk != "verdict":
            print(f"        {kk} = {vv}")
print("=" * 78)
print(f"score: {sum(1 for k in ('R1','R2','R3') if res['verdicts'][k]['verdict']=='CONFIRM')}/3")
print(f"wrote results/exp_06_primes_inherit_{ts}.json")
