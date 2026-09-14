#!/usr/bin/env python3
"""exp_02: the position of the read — SEALED by the commit carrying journals/2026-09-07_exp02_registration.md; scored to
that text. One object (the y-rough numbers at depth y) read at positions u_top = log 2N / log y from the origin outward;
the uniform loop (CRT-uniform residues) is the read at infinity. Per cell: survivors n, the measured density deficit d
against Mertens, the consecutive-survivor diagonal deficit δ_q (q = 3, 4, 5, 8, 9, 10) with SE from window scatter,
ε_q = δ_loop,q − δ_q (sealed primary form), r_q = ε_q/δ_loop,q (alternative, recorded), c = ε₃/d.
  R1 ε₃ at the primes' arc is scale-free across m = 7..9 (CONVERGED), and decreases over u ∈ {2, 2.1, 2.25} for y ≥ 4473
  R2 the sign flip: at u ∈ {2.5, 2.75, 3} for m ∈ {7, 8} (W = 200) ε₃ < 0 and sign(ε₃) = sign(d)
  R3 c = ε₃/d constant across the resolved cells (CONVERGED within 25 %; KILL beyond a factor 2)
  R4 shell independence: r_q equal across q ∈ {3, 9, 4, 8, 5} at the primes' arc (Peter) vs ordered by shell depth (Andy)
Verdicts CONFIRM / CONVERGED / KILL / INCONCLUSIVE. Checkpoint per depth. Results append-only, timestamped.
Usage: python exp_02_position_of_the_read.py --depths 6 7 8 9 --W 32 --W-flip 200 --L 10000000 --seed 20260908 --tag main"""
import sys, json, time, math, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "core"))
import rough as R

ap = argparse.ArgumentParser()
ap.add_argument("--depths", type=int, nargs="+", default=[6, 7, 8, 9]); ap.add_argument("--W", type=int, default=32)
ap.add_argument("--W-flip", type=int, default=200); ap.add_argument("--L", type=int, default=10_000_000)
ap.add_argument("--W-loop", type=int, default=25); ap.add_argument("--L-loop", type=int, default=20_000_000)
ap.add_argument("--seed", type=int, default=20260908); ap.add_argument("--tag", default=""); ap.add_argument("--out-dir", default=None)
ap.add_argument("--X", type=int, default=None)
args = ap.parse_args()
RES = Path(args.out_dir) if args.out_dir else ROOT / "results"; RES.mkdir(parents=True, exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S"); name = f"exp_02_position_of_the_read{('_' + args.tag) if args.tag else ''}_{ts}"
log = open(RES / f"{name}_log.txt", "w")
def L(s): print(s, flush=True); log.write(s + "\n"); log.flush()

U_ALL = (1.75, 2.0, 2.1, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 7.0); FLIP_U = (2.5, 2.75, 3.0); FLIP_M = (7, 8); Q = (3, 4, 5, 8, 9, 10)
out = dict(registration="exp_02 (journals/2026-09-07_exp02_registration.md)", args=vars(args), cells={}, loops={}, decades={})
X = args.X or 2 * 10 ** max(args.depths); T0 = time.time()
primes = R.odd_sieve(X); L(f"sieve to {X}: {len(primes)} primes [{time.time()-T0:.1f}s]")
rng = np.random.default_rng(args.seed); table = R.buchstab_omega()


def position(m, y, u):
    if abs(u - 2.0) < 1e-12:
        return 10 ** m, "decade", 10 ** m, 1
    N = int(round(y ** u / 2))
    if 2 * N <= 2 * 10 ** 8:
        return N, "arc", N, 1
    return N, "windows", args.L, (args.W_flip if (u in FLIP_U and m in FLIP_M) else args.W)


def read_cell(N, kind, Lw, W, ps):
    Tq = {q: 0 for q in Q}; parts = {q: [] for q in Q}; counts = []; dens_parts = []
    if kind in ("decade", "arc"):
        off = R.segmented_rough(R.window_residues(N, ps), ps, Lw); counts.append(len(off)); total_L = Lw
        for sub in np.array_split(off, 8): dens_parts.append(len(sub) / (Lw / 8))
        for q in Q:
            r = R.residues_mod(off, N % q, q); Tq[q] = R.transition_matrix(r, q)
            for part in np.array_split(r, 8): parts[q].append(R.diagonal_deficit(R.transition_matrix(part, q)))
        del off
    else:
        total_L = 0
        for j in range(W):
            Nj = N + j * Lw; off = R.segmented_rough(R.window_residues(Nj, ps), ps, Lw); counts.append(len(off)); total_L += Lw; dens_parts.append(len(off) / Lw)
            for q in Q:
                r = R.residues_mod(off, Nj % q, q); T = R.transition_matrix(r, q); Tq[q] = Tq[q] + T; parts[q].append(R.diagonal_deficit(T))
    n = int(sum(counts)); dens = n / total_L
    return dict(n=n, density=dens, se_density=R.scatter_se(dens_parts), windows=len(counts),
                delta={q: R.diagonal_deficit(Tq[q]) for q in Q}, se={q: R.scatter_se(parts[q]) for q in Q})


def loop_read(ps):
    Tq = {q: 0 for q in Q}; parts = {q: [] for q in Q}; n = 0
    for _ in range(args.W_loop):
        res = R.loop_residues(ps, rng); off = R.segmented_rough(res, ps, args.L_loop); d = dict(zip((int(p) for p in ps), res)); n += len(off)
        for q in Q:
            r = R.residues_mod(off, R.n_mod_q_from_draws(q, d, rng), q); T = R.transition_matrix(r, q); Tq[q] = Tq[q] + T; parts[q].append(R.diagonal_deficit(T))
    return dict(delta={q: R.diagonal_deficit(Tq[q]) for q in Q}, se={q: R.scatter_se(parts[q]) for q in Q}, n=n, density=n / (args.W_loop * args.L_loop))


def checkpoint():
    (RES / f"{name}.json").write_text(json.dumps(out, indent=1, default=str))


for m in args.depths:
    t_dec = time.time(); y = math.isqrt(2 * 10 ** m) + 1; ps = R.primes_upto(primes, y); mert = R.mertens_product(ps)
    loop = loop_read(ps); out["loops"][str(m)] = dict(y=y, k=int(len(ps)), **loop)
    L(f"m={m} y={y} k={len(ps)}: uniform loop δ3={loop['delta'][3]:.4f}±{loop['se'][3]:.4f} δ4={loop['delta'][4]:.4f} δ8={loop['delta'][8]:.4f} δ9={loop['delta'][9]:.4f} δ10={loop['delta'][10]:.4f} [{time.time()-t_dec:.0f}s]")
    for u in U_ALL:
        t0 = time.time(); N, kind, Lw, W = position(m, y, u); cell = read_cell(N, kind, Lw, W, ps)
        u_top = math.log(2 * N) / math.log(y) if kind != "windows" else math.log(N + W * Lw) / math.log(y); u_bot = math.log(N) / math.log(y)
        ratio = cell["density"] / mert; d = 1.0 - ratio; se_d = (cell["se_density"] or 0.0) / mert
        eps = {q: loop["delta"][q] - cell["delta"][q] for q in Q}
        se_eps = {q: math.sqrt((cell["se"][q] or 0) ** 2 + (loop["se"][q] or 0) ** 2) for q in Q}
        r = {q: eps[q] / loop["delta"][q] for q in Q}; se_r = {q: se_eps[q] / abs(loop["delta"][q]) for q in Q}
        c = eps[3] / d if d != 0 else float("nan"); se_c = abs(c) * math.sqrt((se_eps[3] / eps[3]) ** 2 + (se_d / d) ** 2) if (d != 0 and eps[3] != 0) else float("nan")
        rec = dict(m=m, y=y, u=u, u_top=u_top, u_bottom=u_bot, N=str(N), kind=kind, windows=cell["windows"], n=cell["n"], density_ratio=ratio, d=d, se_d=se_d,
                   delta=cell["delta"], se=cell["se"], eps=eps, se_eps=se_eps, r=r, se_r=se_r, c=c, se_c=se_c,
                   buchstab_arc=R.arc_integral_omega(u_top, u_bot, table), seconds=round(time.time() - t0, 1))
        out["cells"][f"m={m},u={u}"] = rec
        L(f"  u={u:<4} {kind:<7} n={cell['n']:<9} ratio={ratio:.4f} d={d:+.4f}  δ3={cell['delta'][3]:.4f}±{cell['se'][3]:.4f}  ε3={eps[3]:+.4f}±{se_eps[3]:.4f}  r3={r[3]:+.4f}  c={c:+.3f}  "
          f"ε4={eps[4]:+.4f} ε8={eps[8]:+.4f} ε9={eps[9]:+.4f} ε10={eps[10]:+.4f} [{rec['seconds']}s]")
    out["decades"][str(m)] = dict(seconds=round(time.time() - t_dec, 1)); checkpoint(); L(f"m={m} done [{time.time()-t_dec:.0f}s]; checkpoint written")


def cell(m, u): return out["cells"].get(f"m={m},u={u}")


def score(out):
    tests = {}; ms = sorted(int(m) for m in out["decades"]); scored = [m for m in ms if m >= 7]
    if len(scored) < 2:
        # a run without the scored decades (smoke, partial) decides nothing — an empty set must not read as converged
        note = f"scored decades present: {scored} (need m = 7..9)"
        return {k: dict(verdict="INCONCLUSIVE", note=note) for k in ("R1", "R2", "R3", "R4")} | {"recorded": dict(note=note)}
    # R1 ------------------------------------------------------------------------------------------------------------------
    def converged(key, sekey, label):
        vals = {m: cell(m, 2.0)[key][3] for m in scored}; ses = {m: cell(m, 2.0)[sekey][3] for m in scored}
        mean = float(np.mean(list(vals.values()))); tol = {m: max(3 * ses[m], 0.10 * abs(mean)) for m in scored}
        within = {m: abs(vals[m] - mean) <= tol[m] for m in scored}
        kill = any(abs(vals[m] - mean) > max(3 * ses[m], 0.30 * abs(mean)) for m in scored)
        return dict(label=label, values=vals, se=ses, mean=mean, tol=tol, within=within, converged=all(within.values()), kill=kill)
    e_form = converged("eps", "se_eps", "ε (sealed)"); r_form = converged("r", "se_r", "r (alternative, recorded)")
    steps = {}
    for m in [m for m in scored if m >= 7]:
        seq = [(u, cell(m, u)["eps"][3], cell(m, u)["se_eps"][3]) for u in (2.0, 2.1, 2.25)]
        st = []
        for (ua, ea, sa), (ub, eb, sb) in zip(seq, seq[1:]):
            resolved = abs(ea - eb) >= 2 * math.sqrt(sa ** 2 + sb ** 2); st.append(dict(step=f"{ua}->{ub}", d_eps=eb - ea, resolved=resolved, decreasing=(eb < ea)))
        steps[m] = st
    resolved = [s for m in steps for s in steps[m] if s["resolved"]]; inc = sum(1 for s in resolved if not s["decreasing"]); dec = sum(1 for s in resolved if s["decreasing"])
    if e_form["kill"] or inc >= 2: r1 = "KILL"
    elif e_form["converged"] and inc == 0: r1 = "CONVERGED" if dec == 0 else "CONFIRM"
    else: r1 = "INCONCLUSIVE"
    tests["R1"] = dict(verdict=r1, eps_form=e_form, r_form=r_form, steps=steps, resolved_steps=len(resolved), decreasing=dec, increasing=inc)
    # R2 ------------------------------------------------------------------------------------------------------------------
    cells = []
    for m in FLIP_M:
        for u in FLIP_U:
            c = cell(m, u)
            if c is None: continue
            res = abs(c["eps"][3]) >= 3 * c["se_eps"][3]; dres = abs(c["d"]) >= 3 * c["se_d"] if c["se_d"] else False
            cells.append(dict(m=m, u=u, eps=c["eps"][3], se=c["se_eps"][3], d=c["d"], se_d=c["se_d"], resolved=res, d_resolved=dres,
                              negative=c["eps"][3] < 0, sign_match=(np.sign(c["eps"][3]) == np.sign(c["d"])) if (res and dres) else None))
    neg_res = sum(1 for c in cells if c["resolved"] and c["negative"]); pos_res = sum(1 for c in cells if c["resolved"] and not c["negative"])
    mism = sum(1 for c in cells if c["sign_match"] is False); matched = all(c["sign_match"] is not False for c in cells)
    if neg_res >= 4 and matched: r2 = "CONFIRM"
    elif pos_res >= 4 or mism >= 2: r2 = "KILL"
    else: r2 = "INCONCLUSIVE"
    tests["R2"] = dict(verdict=r2, cells=cells, negative_resolved=neg_res, positive_resolved=pos_res, sign_mismatches=mism)
    # R3 ------------------------------------------------------------------------------------------------------------------
    cc = []
    for m in scored:
        for u in (2.0, 2.1, 2.25):
            c = cell(m, u)
            if c and abs(c["eps"][3]) >= 3 * c["se_eps"][3] and c["se_d"] and abs(c["d"]) >= 3 * c["se_d"]: cc.append(dict(m=m, u=u, c=c["c"], se_c=c["se_c"]))
    for x in cells:
        if x["resolved"] and x["d_resolved"]:
            c = cell(x["m"], x["u"]); cc.append(dict(m=x["m"], u=x["u"], c=c["c"], se_c=c["se_c"]))
    if len(cc) >= 2:
        vals = [x["c"] for x in cc]; mean = float(np.mean(vals)); spread = max(vals) - min(vals); tol = max(3 * max(x["se_c"] for x in cc if x["se_c"] == x["se_c"]), 0.25 * abs(mean))
        ratio = (max(vals) / min(vals)) if (min(vals) > 0 or max(vals) < 0) else float("inf")
        # The seal states a CONVERGED clause and a KILL clause without precedence. The first scored run gave CONVERGED
        # precedence when both fired (spread 0.073 ≤ tol 0.108 and max/min 2.11 > 2 on two noisy flip cells); corrected
        # toward the seal: both firing is INCONCLUSIVE, said so, with the cells on the record.
        conv, kill = spread <= tol, ratio > 2
        r3 = "INCONCLUSIVE" if (conv and kill) else ("CONVERGED" if conv else ("KILL" if kill else "INCONCLUSIVE"))
        tests["R3"] = dict(verdict=r3, cells=cc, mean=mean, spread=spread, tol=tol, max_over_min=ratio, converged_clause=conv, kill_clause=kill,
                           note=("both the CONVERGED and the KILL clause fire — the seal gives no precedence — INCONCLUSIVE" if (conv and kill) else ""))
    else:
        tests["R3"] = dict(verdict="INCONCLUSIVE", cells=cc, note="fewer than two resolved cells")
    # R4 ------------------------------------------------------------------------------------------------------------------
    per = {}; confirm_all = True; kill_votes = 0
    for m in scored:
        c = cell(m, 2.0); rq = {q: c["r"][q] for q in (3, 9, 4, 8, 5)}; sq = {q: c["se_r"][q] for q in (3, 9, 4, 8, 5)}
        mean = float(np.mean(list(rq.values()))); tol = {q: max(3 * sq[q], 0.15 * abs(mean)) for q in rq}
        within = all(abs(rq[q] - mean) <= tol[q] for q in rq); confirm_all &= within
        d93 = rq[9] - rq[3]; d84 = rq[8] - rq[4]; t93 = max(3 * math.sqrt(sq[9] ** 2 + sq[3] ** 2), 0.15 * abs(mean)); t84 = max(3 * math.sqrt(sq[8] ** 2 + sq[4] ** 2), 0.15 * abs(mean))
        shell = (np.sign(d93) == np.sign(d84)) and abs(d93) > t93 and abs(d84) > t84; kill_votes += int(shell)
        per[m] = dict(r=rq, se=sq, mean=mean, tol=tol, within=within, r9_minus_r3=d93, r8_minus_r4=d84, shell_ordered=bool(shell))
    r4 = "CONFIRM" if confirm_all else ("KILL" if kill_votes >= 2 else "INCONCLUSIVE")
    tests["R4"] = dict(verdict=r4, per_decade=per, shell_votes=kill_votes)
    # recorded --------------------------------------------------------------------------------------------------------------
    tests["recorded"] = dict(
        control_u175={m: dict(eps=cell(m, 1.75)["eps"][3], se=cell(m, 1.75)["se_eps"][3]) for m in ms if cell(m, 1.75)},
        q10_sign_agreement=[dict(m=m, u=u, eps3=cell(m, u)["eps"][3], eps10=cell(m, u)["eps"][10]) for m in ms for u in U_ALL if cell(m, u)],
        m6_eps=cell(6, 2.0)["eps"][3] if cell(6, 2.0) else None)
    return tests


out["tests"] = score(out); out["seconds"] = round(time.time() - T0, 1); checkpoint()
L("TESTS: " + json.dumps({k: v["verdict"] for k, v in out["tests"].items() if k != "recorded"}))
L(f"wrote {RES.name}/{name}.json [{out['seconds']}s]")
L("SCORE DONE")
