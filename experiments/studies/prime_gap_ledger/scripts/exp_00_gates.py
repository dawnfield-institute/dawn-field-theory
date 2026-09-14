#!/usr/bin/env python3
"""exp_00: the gates (known answers), run BEFORE the seal of journals/2026-09-07_exp01_registration.md. Nothing here is
scored; every value is a theorem or a published numeric. A gate that fails stops the round.
  G1 loop count = φ(P_k) and the cyclic mean gap = P_k/φ(P_k) exactly, k ≤ 9
  G2 the enumerated loop's pair count at distance g equals the CRT product ∏_{p|g}(p−1)·∏_{p∤g}(p−2), even g ≤ 30, k ≤ 8
  G3 the champion gap is 6 in every decade m = 4..8, beating both 2 and 4 (known numerics; 2 and 4 share the HL weight)
  G4 local/loop density ratio = e^{γ}·ω(u) on the u-grid, m = 5..8 (Buchstab), within 3/log y
  G5 at u = 2 the ratio → e^{γ}/2 (Mertens + PNT), within 3/log y
  G6 the CRT-uniform sampler reproduces the enumerated loop's gap histogram at k = 9 (instrument)
  G7 Lemke Oliver–Soundararajan sign: the diagonal is the least frequent transition at q = 3 and q = 4, m = 4..8
     (their tables to 10^11). SIGN ONLY is printed — the deficit magnitudes are registered quantities (R2)
  G8 periodicity: where P(y) ≤ L/100 the window's gap histogram equals the loop's (TV < 1e-3)
Usage: python exp_00_gates.py [--X 200000000]. Writes results/exp_00_gates_<ts>.json."""
import sys, json, time, math, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "core"))
from rough import *  # noqa: F401,F403
import rough as R

ap = argparse.ArgumentParser(); ap.add_argument("--X", type=int, default=200_000_000); ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
RES = ROOT / "results"; RES.mkdir(exist_ok=True); ts = time.strftime("%Y%m%d_%H%M%S")
out = {"gates": {}, "args": vars(args)}; T0 = time.time()
def L(s): print(s, flush=True)

t0 = time.time(); primes = R.odd_sieve(args.X); L(f"sieve to {args.X}: {len(primes)} primes [{time.time()-t0:.1f}s]")
rng = np.random.default_rng(args.seed)

# G1 / G2 --------------------------------------------------------------------------------------------------------
g1 = {"k": [], "fail": []}; g2 = {"cells": 0, "fail": []}; loops = {}
for k in range(1, R.LOOP_ENUM_KMAX + 1):
    pk = primes[:k]; off, P = R.loop_enumerate(pk); loops[k] = (off, P, pk)
    phi = R.phi_of_primorial(pk); gsum = int(R.gaps_of(off, P).sum())
    ok = (len(off) == phi) and (gsum == P)
    g1["k"].append({"k": k, "P": P, "units": int(len(off)), "phi": phi, "gap_sum": gsum, "ok": ok})
    if not ok: g1["fail"].append(k)
    if k <= 8:
        units = np.zeros(P, dtype=bool); units[off] = True
        for g in range(2, 31, 2):
            cnt = int((units & np.roll(units, -g)).sum()); pred = R.pair_count_formula(g, pk); g2["cells"] += 1
            if cnt != pred: g2["fail"].append({"k": k, "g": g, "count": cnt, "formula": pred})
g1["pass"] = not g1["fail"]; g2["pass"] = not g2["fail"]; out["gates"]["G1"] = g1; out["gates"]["G2"] = g2
L(f"G1 loop count = φ(P_k), cyclic gaps sum to P_k, k ≤ 9: fails {g1['fail']} -> {g1['pass']}")
L(f"G2 pair counts = CRT product, {g2['cells']} cells (k ≤ 8, even g ≤ 30): fails {len(g2['fail'])} -> {g2['pass']}")

# G3 --------------------------------------------------------------------------------------------------------------
g3 = {"decades": {}, "pass": True}
for m in range(4, 9):
    N = 10 ** m; pr = primes[(primes >= N) & (primes < 2 * N)]; g = np.diff(pr); h = np.bincount(g)
    # Known answer: 6 is the champion and beats BOTH 2 and 4. Gaps 2 and 4 carry the SAME Hardy–Littlewood weight
    # (no odd prime divides 4), so no ordering between them is a known answer — the first draft of this gate wrongly
    # demanded 4 > 2 and failed at every decade; corrected before the seal, both counts recorded.
    mode = int(np.argmax(h)); ok = (mode == 6) and (h[6] > max(h[2], h[4]))
    g3["decades"][m] = {"primes": int(len(pr)), "mode": mode, "h2": int(h[2]), "h4": int(h[4]), "h6": int(h[6]), "ok": ok}
    g3["pass"] &= ok
out["gates"]["G3"] = g3; L(f"G3 champion 6 in every decade: {[(m, d['mode']) for m, d in g3['decades'].items()]} -> {g3['pass']}")

# G4 / G5 (densities only — no shape, no transitions, no depths) ------------------------------------------------------
table = R.buchstab_omega(); w8 = R.omega_at(8.0, table); w2 = R.omega_at(2.0, table)
g4 = {"omega_2": w2, "omega_8": w8, "e_minus_gamma": math.exp(-R.EULER_GAMMA), "cells": [], "fail": []}; g5 = {"cells": [], "fail": []}
for m in range(5, 9):
    N = 10 ** m; Lw = N
    for u in R.U_GRID:
        y = R.depth_y(N, u); ps = R.primes_upto(primes, y); k = len(ps); P = R.primorial(ps)
        off = R.segmented_rough(R.window_residues(N, ps), ps, Lw)
        dens_local = len(off) / Lw; dens_loop = R.mertens_product(ps); ratio = dens_local / dens_loop
        pred = math.exp(R.EULER_GAMMA) * R.omega_at(u, table); tol = 3.0 / math.log(y)
        cell = {"m": m, "u": u, "y": y, "k": k, "periodic": P <= Lw, "ratio": ratio, "predicted": pred, "tol": tol, "ok": abs(ratio - pred) <= tol}
        if abs(u - 2.0) < 1e-12:
            cell["predicted_u2"] = math.exp(R.EULER_GAMMA) / 2; cell["ok"] = abs(ratio - cell["predicted_u2"]) <= tol
            g5["cells"].append(cell); g5["fail"] += [] if cell["ok"] else [cell]
        else:
            g4["cells"].append(cell); g4["fail"] += [] if cell["ok"] else [cell]
g4["pass"] = not g4["fail"] and abs(w2 - 0.5) < 1e-9 and abs(w8 - math.exp(-R.EULER_GAMMA)) < 2e-3
g5["pass"] = not g5["fail"]; out["gates"]["G4"] = g4; out["gates"]["G5"] = g5
L(f"G4 Buchstab ratio e^γ ω(u), {len(g4['cells'])} cells (ω(2)={w2:.4f}, ω(8)={w8:.4f} vs e^-γ={g4['e_minus_gamma']:.4f}): fails {len(g4['fail'])} -> {g4['pass']}")
L(f"G5 u = 2 ratio vs e^γ/2 = {math.exp(R.EULER_GAMMA)/2:.4f}: " + ", ".join(f"m={c['m']}: {c['ratio']:.4f}" for c in g5["cells"]) + f" -> {g5['pass']}")

# G6 --------------------------------------------------------------------------------------------------------------
off9, P9, p9 = loops[9]; h_enum = R.gap_hist(R.gaps_of(off9, P9))
sample = R.loop_sample(p9, 10_000_000, 20, rng); gs = [R.gaps_of(o) for o, _ in sample]
h_all = R.gap_hist(np.concatenate(gs)); h_a = R.gap_hist(np.concatenate(gs[:10])); h_b = R.gap_hist(np.concatenate(gs[10:]))
g6 = {"k": 9, "tv_sample_vs_enum": R.tv(h_all, h_enum), "tv_half_split": R.tv(h_a, h_b), "tol": 2e-3}
g6["pass"] = g6["tv_sample_vs_enum"] <= max(g6["tol"], 2 * g6["tv_half_split"]); out["gates"]["G6"] = g6
L(f"G6 CRT-uniform sampler vs enumerated loop (k = 9): TV {g6['tv_sample_vs_enum']:.2e} (half-split {g6['tv_half_split']:.2e}) -> {g6['pass']}")

# G7 (sign only) -----------------------------------------------------------------------------------------------------
g7 = {"decades": {}, "pass": True}
for m in range(4, 9):
    N = 10 ** m; pr = primes[(primes >= N) & (primes < 2 * N)]; rec = {}
    for q in (3, 4):
        Tm = R.transition_matrix(pr % q, q); rows = [a for a in range(q) if Tm[a].sum() > 0]
        sign_ok = all(Tm[a, a] == Tm[a][[b for b in range(q) if Tm[a].sum() > 0 and Tm[:, b].sum() > 0]].min() and
                      all(Tm[a, a] < Tm[a, b] for b in range(q) if b != a and Tm[:, b].sum() > 0) for a in rows)
        rec[f"q={q}"] = {"diagonal_least_in_every_row": bool(sign_ok)}; g7["pass"] &= bool(sign_ok)
    g7["decades"][m] = rec
out["gates"]["G7"] = g7; L(f"G7 LO–S sign (diagonal least, q = 3, 4, every decade): -> {g7['pass']}")

# G8 --------------------------------------------------------------------------------------------------------------
g8 = {"cells": [], "fail": []}
for m in range(5, 9):
    N = 10 ** m; Lw = N
    for u in R.U_GRID:
        y = R.depth_y(N, u); ps = R.primes_upto(primes, y); k = len(ps); P = R.primorial(ps)
        if P > Lw // 100 or k > R.LOOP_ENUM_KMAX: continue
        off = R.segmented_rough(R.window_residues(N, ps), ps, Lw); loff, LP, _ = loops[k]
        d = R.tv(R.gap_hist(R.gaps_of(off)), R.gap_hist(R.gaps_of(loff, LP)))
        cell = {"m": m, "u": u, "y": y, "k": k, "tv": d, "ok": d < 1e-3}; g8["cells"].append(cell)
        if not cell["ok"]: g8["fail"].append(cell)
g8["pass"] = not g8["fail"]; out["gates"]["G8"] = g8
L(f"G8 periodic cells (P(y) ≤ L/100) window = loop: {len(g8['cells'])} cells, max TV {max([c['tv'] for c in g8['cells']] or [0]):.1e} -> {g8['pass']}")

out["all_pass"] = all(g["pass"] for g in out["gates"].values()); out["seconds"] = round(time.time() - T0, 1)
(RES / f"exp_00_gates_{ts}.json").write_text(json.dumps(out, indent=1, default=str))
L(f"wrote results/exp_00_gates_{ts}.json [{out['seconds']}s]")
assert out["all_pass"], "a gate failed — the round does not seal"
L("gates passed")
