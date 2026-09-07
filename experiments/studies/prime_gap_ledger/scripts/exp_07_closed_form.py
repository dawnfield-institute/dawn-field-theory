#!/usr/bin/env python3
"""exp_07 — the closed form at a fresh decade, and where the primes read. Scored to the seal.

Implements journals/2026-09-07_exp07_registration.md EXACTLY.

    F_c(x) = tanh(a * x),  x = phi(q)/gbar      a fitted on the LOOP's training moduli, frozen at the seal

R1  the closed form predicts the m=10 primes: rms <= 1.5 * the loop's held-out rms   (KILL if > 2.5 *)
R2  it beats the 18-bin F on those same cells, >= 3 sigma paired bootstrap
R3  the depth position lambda(m=10) is within 0.15 of the forecast 0.652             (KILL outside [0.35,0.95])

Nothing is fitted at m=10. Writes results/exp_07_closed_form_<ts>.json.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

SEAL = "journals/2026-09-07_exp07_registration.md"
SEED, BOOT, M = 20260911, 10_000, 10
LAMBDA_FORECAST, LAMBDA_TOL = 0.652, 0.15
LAMBDA_KILL = (0.35, 0.95)
POOL = tuple(q for q in range(3, 61) if not (q % 2 == 0 and (q // 2) % 2 == 1))
GBAR = math.exp(R.EULER_GAMMA)
rms = lambda a: float(np.sqrt((np.asarray(a) ** 2).mean()))

gates = json.loads(sorted(RES.glob("exp_07_gates_*.json"))[-1].read_text())
A = gates["a"]; LOOP_HO_RMS = gates["loop_heldout_rms"]
d5 = json.loads(sorted(RES.glob("exp_05_collapse_*.json"))[-1].read_text())
CX = np.array(d5["F_phi_bins"]["centres"]); CY = np.array(d5["F_phi_bins"]["means"])

primes = R.odd_sieve(200_000)
primes_eff = R.odd_sieve(2_000_000)          # y_eff at m=10 lies beyond the sieving list (gate lesson)


def phi(q):
    n = 1
    for p, a in R.factor_int(q).items():
        n *= (p - 1) * p ** (a - 1)
    return n


N0 = 10 ** M; y = math.isqrt(2 * N0) + 1; ps = R.primes_upto(primes, y)
t0 = time.time()
rd = R.chunked_read(N0, N0, 10 ** 8, ps, POOL)
if rd["transitions"][POOL[0]] != rd["n"] - 1:
    raise RuntimeError("carry broken at m=10")
dens = rd["density"]
ye = R.y_eff_from_density(dens, primes_eff); y_eff = ye["y_eff"]
gb_eff, gb_y = GBAR * math.log(y_eff), GBAR * math.log(y)
print(f"m=10 read in {time.time()-t0:.0f}s  n={rd['n']}  y={y}  y_eff={y_eff}", flush=True)

cells = []
for q in POOL:
    d = float(R.diagonal_deficit(rd["T"][q]))
    se = float(R.detrended_se_log(rd["parts"][q], rd["logpos"]))
    x = phi(q) / gb_eff
    cells.append(dict(q=q, phi=phi(q), omega=len(R.factor_int(q)), delta=d, se=se, x=x,
                      pred_tanh=float(np.tanh(A * x)), pred_bins=float(np.interp(x, CX, CY))))
saturated = {c["q"] for c in cells if abs(c["delta"] - 1.0) <= c["se"]}
scored = [c for c in cells if c["q"] not in saturated]

dd = np.array([c["delta"] for c in scored]); ph = np.array([c["phi"] for c in scored])
r_tanh = dd - np.array([c["pred_tanh"] for c in scored])
r_bins = dd - np.array([c["pred_bins"] for c in scored])

res = dict(script="exp_07_closed_form.py", seal=SEAL, generated=ts, a=A, m=M, y=y, y_eff=y_eff,
           n_primes=rd["n"], density=dens, loop_heldout_rms=LOOP_HO_RMS,
           n_cells=len(cells), n_scored=len(scored), saturated=sorted(saturated), verdicts={})

# ---- R1 -------------------------------------------------------------------------------------------------------
r1_rms = rms(r_tanh)
span = float(max(c["pred_tanh"] for c in scored) - min(c["pred_tanh"] for c in scored))
guard = span > 10 * LOOP_HO_RMS
r1 = "KILL" if r1_rms > 2.5 * LOOP_HO_RMS else "CONFIRM" if r1_rms <= 1.5 * LOOP_HO_RMS else "INCONCLUSIVE"
if not guard:
    r1 = "INCONCLUSIVE"
res["verdicts"]["R1"] = dict(rms=r1_rms, confirm_at=1.5 * LOOP_HO_RMS, kill_at=2.5 * LOOP_HO_RMS,
                             F_span=span, guard_not_flat=bool(guard),
                             mean_signed_residual=float(r_tanh.mean()), verdict=r1)

# ---- R2 -------------------------------------------------------------------------------------------------------
rng = np.random.default_rng(SEED); n = len(scored); diffs = []
for _ in range(BOOT):
    i = rng.integers(0, n, n)
    diffs.append(rms(r_bins[i]) - rms(r_tanh[i]))
diffs = np.array(diffs); sig = float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else 0.0
sep = float(np.mean(np.abs(np.array([c["pred_tanh"] - c["pred_bins"] for c in scored]))))
guard2 = sep > diffs.std(ddof=1)
r2 = "CONFIRM" if (rms(r_tanh) < rms(r_bins) and sig >= 3) else \
     "KILL" if (rms(r_bins) <= rms(r_tanh) and abs(sig) >= 3) else "INCONCLUSIVE"
if not guard2:
    r2 = "INCONCLUSIVE"
# recorded, not scored: the same comparison restricted to the bins' own fitted domain
inb = [i for i, c in enumerate(scored) if CX.min() <= c["x"] <= CX.max()]
res["verdicts"]["R2"] = dict(rms_tanh=rms(r_tanh), rms_bins=rms(r_bins), sigma=sig,
                             mean_separation=sep, guard_predictors_differ=bool(guard2),
                             within_bin_domain_recorded_not_scored=dict(
                                 n=len(inb), rms_tanh=rms(r_tanh[inb]), rms_bins=rms(r_bins[inb])),
                             verdict=r2)

# ---- R3 -------------------------------------------------------------------------------------------------------
grid = np.unique(np.round(np.exp(np.linspace(math.log(y * 0.5), math.log(y_eff * 2.0), 900))).astype(int))
curve = np.array([rms(dd - np.tanh(A * ph / (GBAR * math.log(g)))) for g in grid])
best = int(grid[curve.argmin()]); lam = (math.log(best) - math.log(y)) / (math.log(y_eff) - math.log(y))
rms_y = rms(dd - np.tanh(A * ph / gb_y)); rms_e = rms(dd - np.tanh(A * ph / gb_eff))
# §4 guard: the minimum must be resolved below BOTH endpoints by more than the bootstrap spread
bs = []
for _ in range(2000):
    i = rng.integers(0, n, n)
    c2 = np.array([rms(dd[i] - np.tanh(A * ph[i] / (GBAR * math.log(g)))) for g in grid[::12]])
    bs.append(c2.min())
spread = float(np.std(bs, ddof=1))
resolved = (rms_y - curve.min() > spread) and (rms_e - curve.min() > spread)
r3 = "KILL" if not (LAMBDA_KILL[0] <= lam <= LAMBDA_KILL[1]) else \
     "CONFIRM" if abs(lam - LAMBDA_FORECAST) <= LAMBDA_TOL else "INCONCLUSIVE"
if not resolved:
    r3 = "INCONCLUSIVE"
res["verdicts"]["R3"] = dict(lambda_measured=lam, forecast=LAMBDA_FORECAST, tol=LAMBDA_TOL,
                             argmin_depth=best, rms_at_min=float(curve.min()), rms_at_y=rms_y,
                             rms_at_y_eff=rms_e, bootstrap_spread=spread,
                             guard_minimum_resolved=bool(resolved), verdict=r3)

# recorded, never scored (§7.5): the best-fit a at the primes
ag = np.linspace(0.5, 3.0, 5001)
res["best_fit_a_at_primes_recorded_not_scored"] = float(ag[int(np.argmin([rms(dd - np.tanh(g * ph / gb_eff)) for g in ag]))])
res["cells"] = cells

(RES / f"exp_07_closed_form_{ts}.json").write_text(json.dumps(res, indent=1, default=str))
print("\n" + "=" * 78)
print(f"  scored {len(scored)}/{len(cells)} cells; saturated {sorted(saturated)}")
for k in ("R1", "R2", "R3"):
    print(f"  {k}  {res['verdicts'][k]['verdict']}")
    for kk, vv in res["verdicts"][k].items():
        if kk != "verdict":
            print(f"        {kk} = {vv}")
print("=" * 78)
print(f"  a from the loop = {A:.5f}; best-fit a at the primes (recorded) = "
      f"{res['best_fit_a_at_primes_recorded_not_scored']:.5f}")
print(f"score: {sum(1 for k in ('R1','R2','R3') if res['verdicts'][k]['verdict']=='CONFIRM')}/3")
print(f"wrote results/exp_07_closed_form_{ts}.json")
