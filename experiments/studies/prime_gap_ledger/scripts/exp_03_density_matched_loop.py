#!/usr/bin/env python3
"""exp_03: the density-matched loop — SEALED by the commit carrying journals/2026-09-07_exp03_registration.md; scored to
that text. The primes' arc (and every position) is the uniform loop read at the effective depth y_eff whose exact Mertens
product equals the arc's measured density. Per cell and modulus: δ_q (the read) with SE from the de-trended chunk scatter;
δ_q(y) and δ_q(y_eff) on the uniform loops (W_loop windows each); r_q = 1 − δ_q/δ_q(y); r_q^eff = 1 − δ_q(y_eff)/δ_q(y);
ρ_q = r_q − r_q^eff (the departure of the primes' p-adic gap profile from the density-matched loop — Lemke Oliver–
Soundararajan evaluated exactly), with SE propagated from the three measurements.
  R1 ρ_q = 0 at the primes' arc (u = 2), m ∈ {9, 10}, every unsaturated q on the ladders {3, 9}, {4, 8, 16}, {5}, {7}
  R2 ρ_q = 0 along the curve and through the flip (u ∈ {2.1, 2.25, 3}, m = 10), including the predicted sign at u = 3
Precedence: KILL (resolved shell-ordered ρ difference, or wrong sign at u = 3) → CONVERGED → INCONCLUSIVE.
Saturated moduli (δ_q(y) > 0.9) are recorded, never scored. Checkpoint after every loop and cell.
Usage: python exp_03_density_matched_loop.py --depths 8 9 10 --W-loop 200 --seed 20260909 --tag main"""
import sys, json, time, math, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "core"))
import rough as R

ap = argparse.ArgumentParser()
ap.add_argument("--depths", type=int, nargs="+", default=[8, 9, 10]); ap.add_argument("--W-loop", type=int, default=200)
ap.add_argument("--L-loop", type=int, default=20_000_000); ap.add_argument("--W", type=int, default=256); ap.add_argument("--L", type=int, default=10_000_000)
ap.add_argument("--chunk", type=int, default=100_000_000); ap.add_argument("--seed", type=int, default=20260909)
ap.add_argument("--tag", default=""); ap.add_argument("--out-dir", default=None)
args = ap.parse_args()
RES = Path(args.out_dir) if args.out_dir else ROOT / "results"; RES.mkdir(parents=True, exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S"); name = f"exp_03_density_matched_loop{('_' + args.tag) if args.tag else ''}_{ts}"
log = open(RES / f"{name}_log.txt", "w")
def L(s): print(s, flush=True); log.write(s + "\n"); log.flush()

Q = (3, 9, 27, 4, 8, 16, 5, 25, 7, 49); LADDERS = ((3, 9), (4, 8), (8, 16)); SCORED_U = (2.0, 2.1, 2.25, 3.0); U_ALL = (2.0, 2.1, 2.25, 3.0, 5.0)
out = dict(registration="exp_03 (journals/2026-09-07_exp03_registration.md)", args=vars(args), loops={}, cells={}, decades={})
T0 = time.time(); primes = R.odd_sieve(4_000_000); rng = np.random.default_rng(args.seed); LOOPS = {}
L(f"primes to 4e6: {len(primes)} [{time.time()-T0:.1f}s]")


def checkpoint():
    (RES / f"{name}.json").write_text(json.dumps(out, indent=1, default=str))


def loop_at(yq, W):
    """The uniform loop at depth yq (a prime), cached by (yq, W)."""
    key = f"{yq},{W}"
    if key not in LOOPS:
        t0 = time.time(); ps = R.primes_upto(primes, yq); lp = R.loop_read(ps, W, args.L_loop, Q, rng)
        lp.update(y=int(yq), k=int(len(ps)), seconds=round(time.time() - t0, 1)); LOOPS[key] = lp; out["loops"][key] = lp
        L(f"  loop y={yq} k={len(ps)} W={W}: δ3={lp['delta'][3]:.5f}±{lp['se'][3]:.5f} δ9={lp['delta'][9]:.5f} δ4={lp['delta'][4]:.5f} δ8={lp['delta'][8]:.5f} δ16={lp['delta'][16]:.4f} δ5={lp['delta'][5]:.5f} δ7={lp['delta'][7]:.5f} [{lp['seconds']}s]")
        checkpoint()
    return LOOPS[key]


for m in args.depths:
    t_dec = time.time(); N0 = 10 ** m; y = math.isqrt(2 * N0) + 1; ps = R.primes_upto(primes, y); W_loop = args.W_loop if m >= 9 else max(50, args.W_loop // 2)
    W_pos = args.W if m == 10 else (args.W // 2 if m == 9 else args.W // 4)
    L(f"m={m} y={y} k={len(ps)}: W_loop={W_loop} W_pos={W_pos}")
    lo_y = loop_at(y, W_loop)
    for u in U_ALL:
        t0 = time.time()
        if abs(u - 2.0) < 1e-12:
            N, Lr, chunk, kind = N0, N0, (args.chunk if m == 10 else max(N0 // 10, 1)), "decade"
        else:
            N = int(round(y ** u / 2)); Wc = W_pos if u <= 3.0 else 32; Lr, chunk, kind = Wc * args.L, args.L, "windows"
        rd = R.chunked_read(N, Lr, chunk, ps, Q)
        u_top = math.log(N + Lr) / math.log(y); u_bot = math.log(N) / math.log(y)
        dens = rd["density"]; ratio = dens / R.mertens_product(ps); se_dens = R.detrended_se_log(rd["dens"], rd["logpos"])
        ye = R.y_eff_from_density(dens, primes); yq = ye["y_eff"]
        lo_e = lo_y if yq == y else loop_at(yq, W_loop if u in SCORED_U else max(50, W_loop // 4))
        cell = dict(m=m, u=u, kind=kind, N=str(N), L=Lr, chunks=len(rd["dens"]), n=rd["n"], transitions=rd["transitions"][3], density=dens, density_ratio=ratio,
                    se_density=se_dens, u_top=u_top, u_bottom=u_bot, y=y, y_eff=yq, y_eff_bracket=dict(p_lo=ye["p_lo"], p_hi=ye["p_hi"], mismatch_log=ye["mismatch_log"]),
                    log_ratio=math.log(yq) / math.log(y), q={})
        for q in Q:
            dq = R.diagonal_deficit(rd["T"][q]); se_q = R.detrended_se_log(rd["parts"][q], rd["logpos"])
            dy, sy = lo_y["delta"][q], lo_y["se"][q]; de, se_ = lo_e["delta"][q], lo_e["se"][q]
            r = 1.0 - dq / dy; r_eff = 1.0 - de / dy; rho = r - r_eff
            se_rho = math.sqrt((se_q / dy) ** 2 + (se_ / dy) ** 2 + ((de - dq) * sy / dy ** 2) ** 2)
            se_r = math.sqrt((se_q / dy) ** 2 + (dq * sy / dy ** 2) ** 2)
            beta = (-math.log(de / dy) / math.log(math.log(yq) / math.log(y))) if (yq != y and de > 0 and dy > 0) else float("nan")
            cell["q"][str(q)] = dict(delta=dq, se=se_q, delta_y=dy, se_y=sy, delta_eff=de, se_eff=se_, r=r, se_r=se_r, r_eff=r_eff, rho=rho, se_rho=se_rho,
                                     saturated=(dy > 0.9), tol=max(3 * se_rho, 0.02 * abs(r)), beta=beta)
        cell["seconds"] = round(time.time() - t0, 1); out["cells"][f"m={m},u={u}"] = cell; checkpoint()
        qq = cell["q"]
        L(f"  u={u:<4} {kind:<7} n={rd['n']:<10} ratio={ratio:.4f} y_eff={yq} (log-ratio {cell['log_ratio']:.4f}) | " +
          " ".join(f"ρ{q}={qq[str(q)]['rho']:+.4f}±{qq[str(q)]['se_rho']:.4f}" for q in (3, 9, 4, 8, 16, 5, 7)) +
          f" | r3={qq['3']['r']:+.4f} r3_eff={qq['3']['r_eff']:+.4f} [{cell['seconds']}s]")
    out["decades"][str(m)] = dict(seconds=round(time.time() - t_dec, 1)); checkpoint(); L(f"m={m} done [{time.time()-t_dec:.0f}s]")


def score(out):
    tests = {}
    def cell(m, u): return out["cells"].get(f"m={m},u={u}")

    def evaluate(cells_spec, label, sign_check_u3=False):
        """cells_spec: list of (m, u). KILL (resolved shell-ordered ρ difference, same sign across the spec's cells on ≥ 2
        ladders; or wrong sign at u = 3) → CONVERGED (every resolved unsaturated cell within tolerance) → INCONCLUSIVE."""
        # ladder keys are strings: the first main run computed every cell and then died serialising tuple keys (kept)
        rows = []; ladder_signs = {str(lad): [] for lad in LADDERS}; wrong_sign = 0; unresolved = 0; within_all = True; n_eval = 0
        for (m, u) in cells_spec:
            c = cell(m, u)
            if c is None: continue
            for q in (3, 9, 4, 8, 16, 5, 7):
                x = c["q"][str(q)]
                if x["saturated"]: continue
                # a cell counts only when its scatter SE rests on ≥ 8 parts (chunks / windows): a 2-part scatter is not an SE
                if c["chunks"] < 8: unresolved += 1; continue
                within = abs(x["rho"]) <= x["tol"]
                rows.append(dict(m=m, u=u, q=q, rho=x["rho"], se=x["se_rho"], tol=x["tol"], within=within, r=x["r"], r_eff=x["r_eff"]))
                n_eval += 1; within_all &= within
                if sign_check_u3 and abs(u - 3.0) < 1e-12 and abs(x["r_eff"]) > 3 * x["se_rho"]:
                    if np.sign(x["r"]) != np.sign(x["r_eff"]) and abs(x["r"]) > 3 * x["se_r"]: wrong_sign += 1
            for lad in LADDERS:
                a, b = c["q"][str(lad[0])], c["q"][str(lad[1])]
                if a["saturated"] or b["saturated"]: continue
                D = b["rho"] - a["rho"]; seD = math.sqrt(a["se_rho"] ** 2 + b["se_rho"] ** 2)
                ladder_signs[str(lad)].append(dict(m=m, u=u, D=D, se=seD, resolved=abs(D) >= 3 * seD, sign=int(np.sign(D))))
        # KILL: on ≥ 2 ladders, every cell of the spec has a resolved difference of the same sign
        kill_ladders = [lad for lad, ls in ladder_signs.items() if len(ls) >= 2 and all(x["resolved"] for x in ls) and len({x["sign"] for x in ls}) == 1]
        kill = len(kill_ladders) >= 2 or wrong_sign >= 2
        if kill: v = "KILL"
        elif n_eval > 0 and within_all: v = "CONVERGED"
        else: v = "INCONCLUSIVE"
        return dict(verdict=v, label=label, rows=rows, ladders=ladder_signs, kill_ladders=[str(l) for l in kill_ladders], wrong_sign_u3=wrong_sign, evaluated=n_eval)

    scored_ms = [m for m in (9, 10) if cell(m, 2.0) is not None]
    tests["R1"] = evaluate([(m, 2.0) for m in scored_ms], "R1 the origin is a depth shift") if len(scored_ms) == 2 else dict(verdict="INCONCLUSIVE", note=f"scored decades present: {scored_ms}")
    if cell(10, 2.1) is not None:
        tests["R2"] = evaluate([(10, u) for u in (2.1, 2.25, 3.0)], "R2 along the curve and through the flip", sign_check_u3=True)
    else:
        tests["R2"] = dict(verdict="INCONCLUSIVE", note="m = 10 position cells absent")
    tests["recorded"] = dict(
        beta={f"m={m},u={u}": {q: cell(m, u)["q"][str(q)]["beta"] for q in (3, 9, 4, 8, 5, 7)} for m in args.depths for u in U_ALL if cell(m, u)},
        y_eff={f"m={m},u={u}": dict(y=cell(m, u)["y"], y_eff=cell(m, u)["y_eff"], log_ratio=cell(m, u)["log_ratio"], density_ratio=cell(m, u)["density_ratio"]) for m in args.depths for u in U_ALL if cell(m, u)},
        saturated={f"m={m},u={u}": [q for q in Q if cell(m, u)["q"][str(q)]["saturated"]] for m in args.depths for u in U_ALL if cell(m, u)},
        m8_rows=evaluate([(8, 2.0)], "m = 8 (recorded)")["rows"] if cell(8, 2.0) else None,
        m9_curve=evaluate([(9, u) for u in (2.1, 2.25, 3.0)], "m = 9 curve (recorded)", sign_check_u3=True) if cell(9, 2.1) else None)
    return tests


out["tests"] = score(out); out["seconds"] = round(time.time() - T0, 1); checkpoint()
L("TESTS: " + json.dumps({k: v["verdict"] for k, v in out["tests"].items() if k != "recorded"}))
L(f"wrote {RES.name}/{name}.json [{out['seconds']}s]")
L("SCORE DONE")
