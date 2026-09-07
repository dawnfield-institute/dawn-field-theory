#!/usr/bin/env python3
"""exp_01: the loop, the gap and the delta — SEALED by the commit carrying journals/2026-09-07_exp01_registration.md;
scored to that text. Three registered relations, CONFIRM / KILL / INCONCLUSIVE, scored on decades m = 6..9
(m = 4, 5 recorded; m = 4 was seen in the pre-seal smoke and is never scored):
  R1 the shape delta Δ(u; m) = TV(mean-rescaled local gap shape, loop gap shape) obeys a drift law across decades
     (successive-decade differences shrink monotonically at every evaluable live u)
  R2 the residue bias is a truncation phenomenon: δ(u; m) (consecutive-survivor diagonal deficit mod 3 against the
     product-of-marginals null) obeys the same drift law; at u = 2 it decreases across decades m = 5..9; at the largest
     live u it meets the loop's own value within the floor
  R3 each gap is a cascade that truncates itself: ρ(m) = median(log d/log p) / median(log d_null/log p) is scale-free
     (drift law), d = max least-prime-factor over the gap's interior, d_null the same for matched random composites
Live cells: P(y) > L (the loop's period exceeds the window). Floors: half-splits of the window; for a SAMPLED loop the
half-split of its windows; an ENUMERATED loop is exact and carries floor 0. Checkpoint after every decade.
Usage:
  python exp_01_cascade_truncation.py --decades 4 5 6 7 8 9 --windows 25 --L 20000000 --seed 20260907 --tag main
  python exp_01_cascade_truncation.py --decades 3 4 --windows 4 --L 1000000 --X 200000 --tag smoke --out-dir <scratch>"""
import sys, json, time, math, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "core"))
import rough as R

ap = argparse.ArgumentParser()
ap.add_argument("--decades", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9]); ap.add_argument("--windows", type=int, default=25)
ap.add_argument("--L", type=int, default=20_000_000); ap.add_argument("--seed", type=int, default=20260907)
ap.add_argument("--tag", default=""); ap.add_argument("--out-dir", default=None); ap.add_argument("--X", type=int, default=None)
ap.add_argument("--no-depth", action="store_true", help="skip R3 (the lpf table is 2 bytes per integer of the window)")
ap.add_argument("--scored-min", type=int, default=6, help="lowest scored decade (registration: 6)")
args = ap.parse_args()
RES = Path(args.out_dir) if args.out_dir else ROOT / "results"; RES.mkdir(parents=True, exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S"); name = f"exp_01_cascade_truncation{('_' + args.tag) if args.tag else ''}_{ts}"
log = open(RES / f"{name}_log.txt", "w")
def L(s): print(s, flush=True); log.write(s + "\n"); log.flush()

out = dict(registration="exp_01 (journals/2026-09-07_exp01_registration.md)", args=vars(args), cells={}, decades={}, exact_loop_bias={})
X = args.X or 2 * 10 ** max(args.decades); T0 = time.time()
primes = R.odd_sieve(X); L(f"sieve to {X}: {len(primes)} primes [{time.time()-T0:.1f}s]")
rng = np.random.default_rng(args.seed); LOOPS = {}


def loop_for(ps):
    """Enumerated loop (k ≤ 9, cached) or a CRT-uniform sample: gap arrays per window and residues mod q per window."""
    k = len(ps)
    if k <= R.LOOP_ENUM_KMAX:
        if k not in LOOPS:
            off, P = R.loop_enumerate(ps); LOOPS[k] = (off, P)
        off, P = LOOPS[k]
        return dict(kind="enumerated", k=k, P=P, gaps=[R.gaps_of(off, P)], res={q: [R.residues_mod(off, 0, q)] for q in R.Q_LIST})
    sample = R.loop_sample(ps, args.L, args.windows, rng)
    gaps = [R.gaps_of(o) for o, _ in sample]
    res = {q: [R.residues_mod(o, R.n_mod_q_from_draws(q, d, rng), q) for o, d in sample] for q in R.Q_LIST}
    return dict(kind="sampled", k=k, P=None, gaps=gaps, res=res)


def halves(lst):
    h = max(1, len(lst) // 2)
    return lst[:h], (lst[h:] if len(lst) > 1 else lst[:h])


def deficit_of(res_list, q):
    Tm = sum(R.transition_matrix(r, q) for r in res_list)
    return R.diagonal_deficit(Tm), Tm


def checkpoint():
    (RES / f"{name}.json").write_text(json.dumps(out, indent=1, default=str))


for m in args.decades:
    N = 10 ** m; Lw = N; t_dec = time.time(); dec = {"N": N, "L": Lw, "u": {}}
    for u in R.U_GRID:
        y = R.depth_y(N, u); ps = R.primes_upto(primes, y); k = len(ps); P = R.primorial(ps)
        periodic = P <= Lw; t0 = time.time()
        # local window ---------------------------------------------------------------------------------------------
        off = R.segmented_rough(R.window_residues(N, ps), ps, Lw); g = R.gaps_of(off)
        h = len(g) // 2
        sh = R.scaled_hist(g); floor_local = R.tv(R.scaled_hist(g[:h]), R.scaled_hist(g[h:]))
        del g
        dl, floor_dl = {}, {}
        for q in R.Q_LIST:
            r = R.residues_mod(off, N % q, q); hh = len(r) // 2
            dq, _ = deficit_of([r], q); da, _ = deficit_of([r[:hh]], q); db, _ = deficit_of([r[hh:]], q)
            dl[q] = dq; floor_dl[q] = abs(da - db); del r
        n_surv = int(len(off)); del off
        # the loop --------------------------------------------------------------------------------------------------
        lp = loop_for(ps); gl = np.concatenate(lp["gaps"]); shl = R.scaled_hist(gl)
        if lp["kind"] == "sampled":
            ga, gb = halves(lp["gaps"]); floor_loop = R.tv(R.scaled_hist(np.concatenate(ga)), R.scaled_hist(np.concatenate(gb)))
        else:
            floor_loop = 0.0                                   # an enumerated loop is exact
        dloop, floor_dloop, exact = {}, {}, {}
        for q in R.Q_LIST:
            dq, Tm = deficit_of(lp["res"][q], q); dloop[q] = dq
            if lp["kind"] == "sampled":
                ra, rb = halves(lp["res"][q]); floor_dloop[q] = abs(deficit_of(ra, q)[0] - deficit_of(rb, q)[0])
            else:
                floor_dloop[q] = 0.0
                if lp["P"] % q == 0:
                    e = R.diagonal_deficit_exact(Tm); exact[q] = str(e); out["exact_loop_bias"][f"k={k},q={q}"] = str(e)
        n_gl = int(len(gl)); del gl
        cell = dict(m=m, u=u, y=y, k=k, P=str(P) if P < 10**30 else f"~1e{len(str(P))-1}", periodic=periodic, live=not periodic,
                    loop=lp["kind"], survivors=n_surv, gaps_loop=n_gl, density_ratio=(n_surv / Lw) / R.mertens_product(ps),
                    delta_shape=R.tv(sh, shl), floor_shape=max(floor_local, floor_loop),
                    deficit_local=dl, deficit_loop=dloop, floor_deficit={q: max(floor_dl[q], floor_dloop[q]) for q in R.Q_LIST},
                    exact_loop_deficit=exact, seconds=round(time.time() - t0, 1))
        dec["u"][str(u)] = cell; out["cells"][f"m={m},u={u}"] = cell; del lp
        L(f"m={m} u={u:<4} y={y:<6} k={k:<5} {'periodic' if periodic else 'live    '} loop={cell['loop']:<10} Δshape={cell['delta_shape']:.4f} (floor {cell['floor_shape']:.4f}) "
          f"δ3 local={dl[3]:.4f} loop={dloop[3]:.4f} (floor {cell['floor_deficit'][3]:.4f}) ratio={cell['density_ratio']:.3f} [{cell['seconds']}s]")
    # R3: termination depth ------------------------------------------------------------------------------------------
    if not args.no_depth:
        t0 = time.time(); psq = R.primes_upto(primes, math.isqrt(2 * N) + 1); tbl = R.lpf_table(N, Lw, psq)
        poff = np.flatnonzero(tbl == 0); d = R.termination_depths(poff, tbl).astype(float)
        interior = (np.diff(poff) - 1).astype(np.int64); dn = R.null_depths(interior, tbl, rng).astype(float); del tbl
        lp_ = np.log((N + poff[:-1]).astype(float)); del poff
        x_obs = np.log(np.maximum(d, 2.0)) / lp_; x_null = np.log(np.maximum(dn, 2.0)) / lp_
        rho = float(np.median(x_obs) / np.median(x_null)); hh = len(x_obs) // 2
        rho_a = float(np.median(x_obs[:hh]) / np.median(x_null[:hh])); rho_b = float(np.median(x_obs[hh:]) / np.median(x_null[hh:]))
        dec["depth"] = dict(gaps=int(len(d)), median_obs=float(np.median(x_obs)), median_null=float(np.median(x_null)), rho=rho,
                            floor=abs(rho_a - rho_b), mean_d=float(d.mean()), mean_d_null=float(dn.mean()), seconds=round(time.time() - t0, 1))
        del d, dn, x_obs, x_null, lp_, interior
        L(f"m={m} depth: median log d/log p obs {dec['depth']['median_obs']:.4f} null {dec['depth']['median_null']:.4f} ρ={rho:.4f} (floor {dec['depth']['floor']:.4f}) [{dec['depth']['seconds']}s]")
    dec["seconds"] = round(time.time() - t_dec, 1); out["decades"][str(m)] = dec; checkpoint()
    L(f"m={m} done [{dec['seconds']}s]; checkpoint written")


def drift_verdict(values, floors, label):
    """values[m], floors[m] over consecutive decades: successive differences D_m must shrink monotonically, each above its
    floor (max of the two cells' floors). CONFIRM = all above floor and strictly shrinking; KILL = all above floor and
    strictly growing; INCONCLUSIVE otherwise (a difference below its floor, or non-monotone)."""
    ms = sorted(values); D = {}; F = {}
    for a, b in zip(ms, ms[1:]):
        va, vb = values[a], values[b]
        D[a] = abs(va - vb) if (va == va and vb == vb) else float("nan"); F[a] = max(floors[a], floors[b])
    if len(D) < 2: return "INCONCLUSIVE", dict(label=label, D=D, floors=F, note="fewer than two differences")
    above = all((D[a] == D[a]) and D[a] > F[a] for a in D)
    seq = [D[a] for a in sorted(D)]
    shrink = all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)); grow = all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    if not above: return "INCONCLUSIVE", dict(label=label, D=D, floors=F, note="a difference is below its floor or undefined")
    return ("CONFIRM" if shrink else "KILL" if grow else "INCONCLUSIVE"), dict(label=label, D=D, floors=F, shrink=shrink, grow=grow)


def score(out):
    tests = {}; decs = sorted(int(m) for m in out["decades"]); scored = [m for m in decs if m >= args.scored_min]
    live_u = [u for u in R.U_GRID if all(out["cells"].get(f"m={m},u={u}", {}).get("live") for m in scored) and abs(u - 2.0) > 1e-12]

    def drift_over_u(key, sub, label):
        per_u = {}; verdicts = []
        for u in live_u:
            v = {m: (out["cells"][f"m={m},u={u}"][key][sub] if sub is not None else out["cells"][f"m={m},u={u}"][key]) for m in scored}
            fkey = "floor_deficit" if key == "deficit_local" else "floor_shape"
            f = {m: (out["cells"][f"m={m},u={u}"][fkey][sub] if sub is not None else out["cells"][f"m={m},u={u}"][fkey]) for m in scored}
            ver, det = drift_verdict(v, f, f"{label} u={u}"); per_u[str(u)] = dict(verdict=ver, values=v, **det); verdicts.append(ver)
        n_eval = sum(1 for v in verdicts if v != "INCONCLUSIVE")
        overall = ("CONFIRM" if n_eval >= 3 and all(v == "CONFIRM" for v in verdicts if v != "INCONCLUSIVE") else
                   "KILL" if verdicts.count("KILL") >= 2 else "INCONCLUSIVE")
        return overall, n_eval, per_u

    # R1 ----------------------------------------------------------------------------------------------------------------
    r1, n1, pu1 = drift_over_u("delta_shape", None, "R1"); tests["R1"] = dict(verdict=r1, live_u=live_u, scored=scored, evaluable=n1, per_u=pu1)
    # R2 ----------------------------------------------------------------------------------------------------------------
    r2a, n2, pu2 = drift_over_u("deficit_local", 3, "R2a")
    d2 = {m: out["cells"][f"m={m},u=2.0"]["deficit_local"][3] for m in decs if m >= 5 and f"m={m},u=2.0" in out["cells"]}
    seq = [d2[m] for m in sorted(d2)]
    r2b = ("CONFIRM" if len(seq) >= 2 and all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)) else "KILL" if len(seq) >= 2 else "INCONCLUSIVE")
    meet = {}
    for m in scored:
        if not live_u: continue
        u = max(live_u); c = out["cells"][f"m={m},u={u}"]
        meet[m] = dict(u=u, local=c["deficit_local"][3], loop=c["deficit_loop"][3], floor=c["floor_deficit"][3],
                       ok=abs(c["deficit_local"][3] - c["deficit_loop"][3]) <= c["floor_deficit"][3])
    r2c = "CONFIRM" if meet and all(v["ok"] for v in meet.values()) else ("KILL" if meet and not any(v["ok"] for v in meet.values()) else "INCONCLUSIVE")
    r2 = "CONFIRM" if (r2a, r2b, r2c) == ("CONFIRM",) * 3 else ("KILL" if r2b == "KILL" or r2c == "KILL" else "INCONCLUSIVE")
    tests["R2"] = dict(verdict=r2, a=dict(verdict=r2a, evaluable=n2, per_u=pu2), b=dict(verdict=r2b, deficit_u2=d2), c=dict(verdict=r2c, meet=meet),
                       q_ratio_recorded={m: (out["cells"][f"m={m},u=2.0"]["deficit_local"][10] / out["cells"][f"m={m},u=2.0"]["deficit_local"][3])
                                         for m in scored if f"m={m},u=2.0" in out["cells"]},
                       deficit_u2_recorded_m4={m: out["cells"][f"m={m},u=2.0"]["deficit_local"][3] for m in decs if m < 5 and f"m={m},u=2.0" in out["cells"]})
    # R3 ----------------------------------------------------------------------------------------------------------------
    if all("depth" in out["decades"][str(m)] for m in scored) and len(scored) >= 3:
        v = {m: out["decades"][str(m)]["depth"]["rho"] for m in scored}; f = {m: out["decades"][str(m)]["depth"]["floor"] for m in scored}
        ver, det = drift_verdict(v, f, "R3"); tests["R3"] = dict(verdict=ver, rho=v, **det)
    else:
        tests["R3"] = dict(verdict="INCONCLUSIVE", note="depth not computed on enough scored decades")
    return tests


out["tests"] = score(out); out["seconds"] = round(time.time() - T0, 1)
checkpoint()
L("TESTS: " + json.dumps({k: v["verdict"] for k, v in out["tests"].items()}))
L(f"wrote {RES.name}/{name}.json [{out['seconds']}s]")
L("SCORE DONE")
