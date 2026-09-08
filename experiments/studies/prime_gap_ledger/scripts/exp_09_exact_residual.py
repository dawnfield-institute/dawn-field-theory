#!/usr/bin/env python3
"""exp_09 — is F's residual real structure, or was it our sampling? Scored to the seal.

Implements journals/2026-09-08_exp09_registration.md EXACTLY.

R1  the residual is real structure   -- rms(exact delta - F_c) >= 0.60*0.03043 = 0.01826
                                        KILL if <= 0.30* = 0.00913 (it was noise; F is near-exact)
R2  the omega(q) structure is real   -- positive correlation on EXACT values, >= 3 sigma
R3  truncation is not driving it     -- residual at G=36 vs G=40 differs by < 20%

delta_q is DERIVED (inclusion-exclusion, core/exact_gaps.py) plus a measured tail above G.
F_c = tanh(1.2998 x) is frozen from exp_07. Nothing is fitted here.
Writes results/exp_09_exact_residual_<ts>.json.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "core")); import rough as R; import exact_gaps as X
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

SEAL = "journals/2026-09-08_exp09_registration.md"
SEED = 20260913
DEPTHS = (23, 47, 97, 199, 401)
QS = tuple(q for q in range(3, 31) if not (q % 2 == 0 and (q // 2) % 2 == 1))
G_MAIN, G_CTRL = 40, 36
W_TAIL, L_TAIL = 24, 2_000_000
rms = lambda a: float(np.sqrt((np.asarray(a) ** 2).mean()))

gates = json.loads(sorted(RES.glob("exp_09_gates_*.json"))[-1].read_text())
A = gates["a"]; REF = gates["loop_heldout_rms"]
CONFIRM, KILL = 0.60 * REF, 0.30 * REF
primes = R.odd_sieve(200_000)


def phi(q):
    n = 1
    for p, a in R.factor_int(q).items():
        n *= (p - 1) * p ** (a - 1)
    return n


def build(G, rng):
    cells = []
    for y in DEPTHS:
        ps = R.primes_upto(primes, y)
        ex = X.gap_dist_exact(ps, G)
        hist = np.zeros(4001)
        for o, _ in R.loop_sample(ps, L_TAIL, W_TAIL, rng):
            hist += np.bincount(R.gaps_of(o), minlength=4001)[:4001]
        hist /= hist.sum()
        tail = {int(g): float(hist[g]) for g in range(G + 2, 4001, 2) if hist[g] > 0}
        gb = 1.0 / R.mertens_product(ps)                 # EXACT mean gap, not e^gamma log y
        for q in QS:
            pq, exp_, tl = X.p_divides_gap(ex, q, tail)
            d = X.delta_from_gaps(pq, phi(q))
            x = phi(q) / gb
            cells.append(dict(y=y, q=q, phi=phi(q), omega=len(R.factor_int(q)), gbar=gb, x=x,
                              delta=d, p_exact=exp_, p_tail=tl,
                              tail_share=(tl / pq if pq > 0 else 0.0),
                              pred=float(np.tanh(A * x))))
    return cells


t0 = time.time()
cells = build(G_MAIN, np.random.default_rng(SEED))
print(f"main grid G={G_MAIN}: {len(cells)} cells [{time.time()-t0:.0f}s]", flush=True)

# §7.1 — a cell whose tail share exceeds 10% is recorded, not scored
scored = [c for c in cells if c["tail_share"] <= 0.10]
excluded = sorted({(c["y"], c["q"]) for c in cells if c["tail_share"] > 0.10})
res = np.array([c["delta"] - c["pred"] for c in scored])

out = dict(script="exp_09_exact_residual.py", seal=SEAL, generated=ts, a=A, reference_rms=REF,
           confirm_at=CONFIRM, kill_at=KILL, G_main=G_MAIN, G_ctrl=G_CTRL,
           n_cells=len(cells), n_scored=len(scored),
           recorded_not_scored_high_tail=[list(e) for e in excluded], verdicts={})

# ---- R1 ---------------------------------------------------------------------------------------------------
r1_rms = rms(res)
span = float(max(c["pred"] for c in scored) - min(c["pred"] for c in scored))
guard1 = span > 10 * REF
r1 = "KILL" if r1_rms <= KILL else "CONFIRM" if r1_rms >= CONFIRM else "INCONCLUSIVE"
if not guard1:
    r1 = "INCONCLUSIVE"
out["verdicts"]["R1"] = dict(rms=r1_rms, confirm_at=CONFIRM, kill_at=KILL, sampled_reference=REF,
                             ratio_to_sampled=r1_rms / REF, F_span=span, guard_not_flat=bool(guard1),
                             mean_signed=float(res.mean()), verdict=r1)

# ---- R2 ---------------------------------------------------------------------------------------------------
om = np.array([c["omega"] for c in scored], float); qq = np.array([c["q"] for c in scored], float)
vals, counts = np.unique(om, return_counts=True)
guard2 = (len(vals) >= 3) and (counts >= 10).sum() >= 3
c_om = float(np.corrcoef(om, res)[0, 1]); c_q = float(np.corrcoef(qq, res)[0, 1])
n = len(res)
t_om = c_om * math.sqrt((n - 2) / max(1e-12, 1 - c_om ** 2))
r2 = "KILL" if t_om <= -3 else "CONFIRM" if t_om >= 3 else "INCONCLUSIVE"
if not guard2:
    r2 = "INCONCLUSIVE"
out["verdicts"]["R2"] = dict(corr_omega=c_om, t_omega=t_om, corr_q=c_q,
                             omega_values={int(v): int(c) for v, c in zip(vals, counts)},
                             guard_omega_varies=bool(guard2), verdict=r2)

# ---- R3 ---------------------------------------------------------------------------------------------------
cells_c = build(G_CTRL, np.random.default_rng(SEED))
sc_c = [c for c in cells_c if c["tail_share"] <= 0.10]
res_c = np.array([c["delta"] - c["pred"] for c in sc_c])
r3_rms = rms(res_c)
rel = abs(r3_rms - r1_rms) / max(r1_rms, 1e-12)
r3 = "CONFIRM" if rel < 0.20 else "KILL"
out["verdicts"]["R3"] = dict(rms_G40=r1_rms, rms_G36=r3_rms, relative_change=rel, verdict=r3)
if r3 == "KILL":
    out["verdicts"]["R1"]["verdict"] = "INCONCLUSIVE"
    out["verdicts"]["R2"]["verdict"] = "INCONCLUSIVE"

# §7.3 recorded, never scored: the best-fit a on exact values
grid = np.linspace(0.8, 2.0, 4801)
xs = np.array([c["x"] for c in scored]); ds = np.array([c["delta"] for c in scored])
out["best_fit_a_on_exact_recorded_not_scored"] = float(
    grid[int(np.argmin([rms(ds - np.tanh(g * xs)) for g in grid]))])
out["cells"] = cells

(RES / f"exp_09_exact_residual_{ts}.json").write_text(json.dumps(out, indent=1, default=str))
print("\n" + "=" * 78)
print(f"  scored {len(scored)}/{len(cells)} cells; high-tail excluded: {excluded or 'none'}")
for k in ("R1", "R2", "R3"):
    print(f"  {k}  {out['verdicts'][k]['verdict']}")
    for kk, vv in out["verdicts"][k].items():
        if kk != "verdict":
            print(f"        {kk} = {vv}")
print("=" * 78)
print(f"  a frozen from exp_07 = {A:.5f}; best fit on exact values (recorded) = "
      f"{out['best_fit_a_on_exact_recorded_not_scored']:.5f}")
print(f"score: {sum(1 for k in ('R1','R2','R3') if out['verdicts'][k]['verdict']=='CONFIRM')}/3")
print(f"wrote results/exp_09_exact_residual_{ts}.json")
