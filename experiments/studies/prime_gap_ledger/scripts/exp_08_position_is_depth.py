#!/usr/bin/env python3
"""exp_08 — is the coherence transition just the depth moving? Scored to the seal.

Implements journals/2026-09-08_exp08_registration.md EXACTLY.

    delta_q(u) = F_c( phi(q) / gbar(y_eff(u)) )     F_c = tanh(1.2998 x), sealed in exp_07
                                                    y_eff(u) solved from the measured density
    ZERO free parameters.

R1  position is depth      -- rms <= 1.75 * 0.03043 = 0.05325   (KILL if > 3.0 * = 0.09129)
R2  the transition is universal across moduli -- sd of C_q(u) across moduli <= 0.15  (KILL > 0.35)
R3  positive control       -- the fixed-depth model must be WORSE, >= 3 sigma paired bootstrap

Writes results/exp_08_position_is_depth_<ts>.json.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

SEAL = "journals/2026-09-08_exp08_registration.md"
SEED, BOOT = 20260912, 10_000
Y, L = 4473, 2_000_000
U_GRID = (2.0, 2.1, 2.2, 2.35, 2.5, 2.65, 2.8, 3.0, 3.5, 4.5, 6.0)
POOL = tuple(q for q in range(3, 61) if not (q % 2 == 0 and (q // 2) % 2 == 1))
GBAR = math.exp(R.EULER_GAMMA)
rms = lambda a: float(np.sqrt((np.asarray(a) ** 2).mean()))

g = json.loads(sorted(RES.glob("exp_08_gates_*.json"))[-1].read_text())
A = g["a"]; REF = g["loop_heldout_rms"]
CONFIRM, KILL = 1.75 * REF, 3.0 * REF

primes = R.odd_sieve(200_000)
primes_eff = R.odd_sieve(2_000_000)
ps = R.primes_upto(primes, Y)


def phi(q):
    n = 1
    for p, a in R.factor_int(q).items():
        n *= (p - 1) * p ** (a - 1)
    return n


cells, meta = [], {}
for u in U_GRID:
    N = int(round(Y ** u)); t0 = time.time()
    off = R.segmented_rough(R.window_residues(N, ps), ps, L)
    n = int(len(off)); dens = n / L
    ye = R.y_eff_from_density(dens, primes_eff); y_eff = ye["y_eff"]
    gb = GBAR * math.log(y_eff)
    h = n // 2
    for q in POOL:
        r = R.residues_mod(off, N % q, q)
        d = float(R.diagonal_deficit(R.transition_matrix(r, q)))
        # SE from a half-split of the window (no chunk ensemble at a single position)
        da = R.diagonal_deficit(R.transition_matrix(r[:h], q))
        db = R.diagonal_deficit(R.transition_matrix(r[h:], q))
        se = float(abs(da - db) / 2.0)
        del r
        x = phi(q) / gb
        cells.append(dict(u=u, q=q, phi=phi(q), N=str(N), n=n, density=dens, y_eff=y_eff,
                          delta=d, se=se, x=x, pred=float(np.tanh(A * x)),
                          pred_fixed=float(np.tanh(A * phi(q) / (GBAR * math.log(Y)))),
                          log_shift=math.log(y_eff) / math.log(Y) - 1))
    meta[u] = dict(N=str(N), n=n, density=dens, y_eff=y_eff, log_shift=cells[-1]["log_shift"])
    del off
    print(f"u={u:<5} N={N:<22} n={n:<8} y_eff={y_eff:<7} shift={meta[u]['log_shift']:+.4f} [{time.time()-t0:.0f}s]",
          flush=True)

by_q = {}
for c in cells:
    by_q.setdefault(c["q"], []).append(c)
saturated = {q for q, cs in by_q.items() if all(abs(c["delta"] - 1.0) <= c["se"] for c in cs)}
scored = [c for c in cells if c["q"] not in saturated]

res_track = np.array([c["delta"] - c["pred"] for c in scored])
res_fixed = np.array([c["delta"] - c["pred_fixed"] for c in scored])
out = dict(script="exp_08_position_is_depth.py", seal=SEAL, generated=ts, a=A, y=Y, L=L,
           u_grid=list(U_GRID), reference_rms=REF, confirm_at=CONFIRM, kill_at=KILL,
           positions=meta, n_cells=len(cells), n_scored=len(scored), saturated=sorted(saturated), verdicts={})

# ---- R1 --------------------------------------------------------------------------------------------------
r1_rms = rms(res_track)
span = float(max(c["pred"] for c in scored) - min(c["pred"] for c in scored))
guard = span > 10 * REF
no3 = [i for i, c in enumerate(scored) if c["q"] != 3]
r1 = "KILL" if r1_rms > KILL else "CONFIRM" if r1_rms <= CONFIRM else "INCONCLUSIVE"
if not guard:
    r1 = "INCONCLUSIVE"
out["verdicts"]["R1"] = dict(rms=r1_rms, confirm_at=CONFIRM, kill_at=KILL, F_span=span,
                             guard_not_flat=bool(guard),
                             rms_excluding_q3_recorded=rms(res_track[no3]),
                             mean_signed=float(res_track.mean()), verdict=r1)

# ---- R2 --------------------------------------------------------------------------------------------------
u0, u1 = U_GRID[0], U_GRID[-1]
elig, curves = [], {}
for q, cs in by_q.items():
    if q in saturated:
        continue
    a0 = next(c for c in cs if c["u"] == u0); a1 = next(c for c in cs if c["u"] == u1)
    denom = a0["delta"] - a1["delta"]
    if abs(denom) > 3 * max(a0["se"], a1["se"], 1e-12):
        elig.append(q)
        curves[q] = {c["u"]: (c["delta"] - a1["delta"]) / denom for c in cs}
if len(elig) >= 10:
    sds = [float(np.std([curves[q][u] for q in elig], ddof=1)) for u in U_GRID]
    mean_sd = float(np.mean(sds))
    r2 = "KILL" if mean_sd > 0.35 else "CONFIRM" if mean_sd <= 0.15 else "INCONCLUSIVE"
else:
    sds, mean_sd, r2 = [], float("nan"), "INCONCLUSIVE"
out["verdicts"]["R2"] = dict(n_eligible=len(elig), eligible=sorted(elig), mean_sd_across_moduli=mean_sd,
                             sd_per_u={str(u): s for u, s in zip(U_GRID, sds)},
                             mean_curve={str(u): float(np.mean([curves[q][u] for q in elig])) for u in U_GRID}
                             if elig else {}, verdict=r2)

# ---- R3 --------------------------------------------------------------------------------------------------
shifts = [abs(v["log_shift"]) for v in meta.values()]
guard3 = sum(1 for s in shifts if s > 0.01) >= 3
rng = np.random.default_rng(SEED); n = len(scored); diffs = []
for _ in range(BOOT):
    i = rng.integers(0, n, n)
    diffs.append(rms(res_fixed[i]) - rms(res_track[i]))
diffs = np.array(diffs); sig = float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else 0.0
better = rms(res_track) < rms(res_fixed)
r3 = "CONFIRM" if (better and sig >= 3) else "KILL" if (not better and abs(sig) >= 3) else "INCONCLUSIVE"
if not guard3:
    r3 = "INCONCLUSIVE"
out["verdicts"]["R3"] = dict(rms_tracking=rms(res_track), rms_fixed=rms(res_fixed), sigma=sig,
                             n_positions_with_shift=sum(1 for s in shifts if s > 0.01),
                             guard_shift_present=bool(guard3), verdict=r3)
out["cells"] = cells

(RES / f"exp_08_position_is_depth_{ts}.json").write_text(json.dumps(out, indent=1, default=str))
print("\n" + "=" * 78)
print(f"  scored {len(scored)}/{len(cells)} cells; saturated {sorted(saturated)}")
for k in ("R1", "R2", "R3"):
    print(f"  {k}  {out['verdicts'][k]['verdict']}")
    for kk, vv in out["verdicts"][k].items():
        if kk not in ("verdict", "eligible", "sd_per_u", "mean_curve"):
            print(f"        {kk} = {vv}")
print("=" * 78)
print(f"score: {sum(1 for k in ('R1','R2','R3') if out['verdicts'][k]['verdict']=='CONFIRM')}/3")
print(f"wrote results/exp_08_position_is_depth_{ts}.json")
