#!/usr/bin/env python3
"""explore_f0: Block F gates KA-1..8, run BEFORE the seal (registration journals/2026-09-07_blockF_registration.md).
Every value here is a known answer or a theorem check on the pure signed cycle; nothing here is scored, and a gate
that fails stops the round (a census whose enumerator miscounts is not a census; an instrument that disagrees with
M15's recorded rows is not M15's instrument).
  KA-1 balanced C_m reproduces M15 exp_05 K3 rows (m, k: det H, angles, reflection PARITY) from the recorded JSON —
       the raw reflection count is eigenvector-sign-gauge dependent (Theorem 2a) and is recorded, not compared
  KA-2 switching invariance of the exact charpoly and of the holonomy invariants (20 random switchings)
  KA-3 charpoly(C_2n^bal) = charpoly(C_n^bal)·charpoly(C_n^tw) exactly, and det C_tw = 4, n ≤ 15 / 30
  KA-4 ones ∈ ker C_bal; λ_min(C_tw) = 4 sin²(π/2n) to 1e-12, n ≤ 30
  KA-5 the Möbius holonomy relation: angles → π − θ, det → (−1)^k det, deficit² sum 4k; H_tw = I at (6, 2)
  KA-6 the conductor and strictness rules on signed cycles n ∈ [3, 30], seven fields (392 cells)
  KA-7 the unicyclic enumerator vs OEIS A001429, n ≤ 11 (12–14 asserted inside exp_20)
  KA-8 per-factor grading = exp_12 part-1 grading on every tree n ≤ 10, seven fields; DomainMatrix = Matrix.charpoly
Usage: python explore_f0_gates.py. Writes results/explore_f0_gates_<ts>.json."""
import sys, json, time, math
from pathlib import Path
import numpy as np, sympy as sp, networkx as nx
HERE = Path(__file__).parent; ROOT = HERE.parent; EXP = ROOT.parent.parent
sys.path.insert(0, str(ROOT / "core"))
from signed import (t, FIELDS, A001429, cond, signed_cycle, switch, class_sign, twist, cartan, charpoly_exact,
                    grade_by_factor, unicyclic_graphs, adjacency, holonomy_matrix, holonomy_invariants,
                    predicted_cycle_grade, build_cycle, cycle_basis_single)
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")
K3 = EXP / "milestones" / "milestone15" / "results" / "exp_05_general_k_limit_20260717_150325.json"
out = {"gates": {}, "m15_anchor": str(K3.relative_to(EXP.parent))}
T0 = time.time()


def L(s):
    print(s, flush=True)


# ---- exp_12 part-1's grade, ported VERBATIM (the script has no main guard, so it is copied, not imported) ----
def grade12(p, dd):
    sd = sp.sqrt(dd); f = sp.factor(p, extension=sd); facs = [g for g in sp.Mul.make_args(f) if g.has(t)]
    gold = [g for g in facs if g.has(sd)]; rat = [g for g in facs if not g.has(sd)]
    if not gold: return "-"
    if not rat: return "strict"
    return "core" if all(g.as_base_exp()[1] % 2 == 0 for g in rat) else "partial"


# KA-1 ------------------------------------------------------------------------------------------------------------
k3 = json.loads(K3.read_text())["k3_z2_scan"]
ka1 = {"rows": 0, "mismatch": [], "raw_count_differs": []}
for kk, rows in k3.items():
    k = int(kk.split("=")[1])
    for r in rows:
        m = r["m"]; A = build_cycle(m)
        assert np.array_equal(A, signed_cycle(m, 1)), "build_cycle ≠ signed_cycle(+1)"
        H, dets, gap = holonomy_matrix(A, list(range(m)), k); inv = holonomy_invariants(H)
        # The per-row reflection COUNT is not an invariant: eigh's column signs act on each transport's det and
        # telescope only round the loop (Theorem 2a). The anchor compares the invariants — det H, the angles and
        # the PARITY of the count (K3's "even count" statement) — and records the raw counts as declared.
        ok = ((dets.count(-1) % 2 == 0) == r["reflections_even"] and abs(inv["det"] - r["det_H"]) < 1e-9
              and np.allclose(inv["angles"], r["holonomy_angles"], atol=1e-8))
        ka1["rows"] += 1
        if dets.count(-1) != r["n_reflections"]:
            ka1["raw_count_differs"].append({"k": k, "m": m, "got": dets.count(-1), "rec": r["n_reflections"]})
        if not ok: ka1["mismatch"].append({"k": k, "m": m, "got": (dets.count(-1), inv), "rec": r})
ka1["pass"] = not ka1["mismatch"]; out["gates"]["KA-1"] = ka1
L(f"KA-1 M15 K3 anchor: {ka1['rows']} rows; det H, angles, reflection parity reproduced, mismatches {len(ka1['mismatch'])}; "
  f"raw reflection counts differ in {len(ka1['raw_count_differs'])} rows (gauge-dependent, declared) -> {ka1['pass']}")

# KA-2 ------------------------------------------------------------------------------------------------------------
rng = np.random.RandomState(20)
ka2 = {"cases": [], "max_dev_charpoly": 0, "max_dev_holonomy": 0.0}
G9 = unicyclic_graphs(9)[17]; A9 = adjacency(G9, 9); cyc9 = cycle_basis_single(A9)
for label, A, cyc, k in (("C12 twisted", signed_cycle(12, -1), list(range(12)), 3),
                         ("C9 twisted", signed_cycle(9, -1), list(range(9)), 2),
                         ("unicyclic n=9 twisted", twist(A9, cyc9), cyc9, 2)):
    p0 = charpoly_exact(A); inv0 = holonomy_invariants(holonomy_matrix(A, cyc, k)[0]); dev = 0.0
    for _ in range(20):
        s = rng.choice([-1.0, 1.0], size=A.shape[0]); B = switch(A, s)
        assert class_sign(B, cyc) == class_sign(A, cyc) == -1
        if sp.expand(charpoly_exact(B) - p0) != 0: ka2["max_dev_charpoly"] += 1
        inv = holonomy_invariants(holonomy_matrix(B, cyc, k)[0])
        dev = max(dev, abs(inv["det"] - inv0["det"]), abs(inv["deficit"] - inv0["deficit"]),
                  float(np.max(np.abs(np.array(inv["angles"]) - np.array(inv0["angles"])))))
    ka2["cases"].append({"case": label, "k": k, "max_dev": dev}); ka2["max_dev_holonomy"] = max(ka2["max_dev_holonomy"], dev)
ka2["pass"] = ka2["max_dev_charpoly"] == 0 and ka2["max_dev_holonomy"] < 1e-10; out["gates"]["KA-2"] = ka2
L(f"KA-2 switching invariance: charpoly deviations {ka2['max_dev_charpoly']}, holonomy max dev {ka2['max_dev_holonomy']:.1e} -> {ka2['pass']}")

# KA-3 ------------------------------------------------------------------------------------------------------------
ka3 = {"cover_fail": [], "det_fail": []}
P = {(n, e): charpoly_exact(signed_cycle(n, e)) for n in range(3, 31) for e in (1, -1)}
for n in range(3, 16):
    if sp.expand(P[(2 * n, 1)] - sp.expand(P[(n, 1)] * P[(n, -1)])) != 0: ka3["cover_fail"].append(n)
for n in range(3, 31):
    detC = (-1) ** n * sp.Poly(P[(n, -1)], t).eval(0)          # p(0) = det(−C)
    if detC != 4: ka3["det_fail"].append((n, int(detC)))
ka3["pass"] = not ka3["cover_fail"] and not ka3["det_fail"]; out["gates"]["KA-3"] = ka3
L(f"KA-3 double cover (n ≤ 15) and det C_tw = 4 (n ≤ 30): fails {ka3['cover_fail']} {ka3['det_fail']} -> {ka3['pass']}")

# KA-4 ------------------------------------------------------------------------------------------------------------
ka4 = {"max_zero_mode": 0.0, "max_lambda_min_dev": 0.0}
for n in range(3, 31):
    Cb = cartan(signed_cycle(n, 1)); Ct = cartan(signed_cycle(n, -1))
    ka4["max_zero_mode"] = max(ka4["max_zero_mode"], float(np.abs(Cb @ np.ones(n)).max()))
    ka4["max_lambda_min_dev"] = max(ka4["max_lambda_min_dev"],
                                    abs(float(np.linalg.eigvalsh(Ct).min()) - 4 * math.sin(math.pi / (2 * n)) ** 2))
ka4["pass"] = ka4["max_zero_mode"] < 1e-12 and ka4["max_lambda_min_dev"] < 1e-12; out["gates"]["KA-4"] = ka4
L(f"KA-4 zero mode {ka4['max_zero_mode']:.1e}, λ_min dev {ka4['max_lambda_min_dev']:.1e} -> {ka4['pass']}")

# KA-5 ------------------------------------------------------------------------------------------------------------
ka5 = {"cells": 0, "fail": [], "degenerate_declared": [], "H_tw_at_6_2_deficit": None, "H_bal_at_6_2_deficit": None}
for k in (2, 3, 4):
    for m in range(6, 31):
        if (m, k) == (6, 4): continue
        Hb, _, gb = holonomy_matrix(signed_cycle(m, 1), list(range(m)), k)
        Ht, _, gt = holonomy_matrix(signed_cycle(m, -1), list(range(m)), k)
        if min(gb, gt) < 1e-6: ka5["degenerate_declared"].append((m, k)); continue
        ib, it = holonomy_invariants(Hb), holonomy_invariants(Ht); ka5["cells"] += 1
        ok = (np.allclose(sorted(math.pi - a for a in ib["angles"]), it["angles"], atol=1e-8)
              and abs(it["det"] - (-1) ** k * ib["det"]) < 1e-9
              and abs(it["deficit"] ** 2 + ib["deficit"] ** 2 - 4 * k) < 1e-8)
        if not ok: ka5["fail"].append({"m": m, "k": k, "bal": ib, "tw": it})
        if (m, k) == (6, 2): ka5["H_tw_at_6_2_deficit"] = it["deficit"]; ka5["H_bal_at_6_2_deficit"] = ib["deficit"]
ka5["pass"] = not ka5["fail"] and ka5["H_tw_at_6_2_deficit"] is not None and ka5["H_tw_at_6_2_deficit"] < 1e-9
out["gates"]["KA-5"] = ka5
L(f"KA-5 Möbius holonomy relation: {ka5['cells']} cells, fails {len(ka5['fail'])}, degenerate {ka5['degenerate_declared']}, "
  f"(6,2): deficit bal {ka5['H_bal_at_6_2_deficit']:.6f} tw {ka5['H_tw_at_6_2_deficit']:.1e} -> {ka5['pass']}")

# KA-6 ------------------------------------------------------------------------------------------------------------
ka6 = {"cells": 0, "positive": [], "fail": []}
for n in range(3, 31):
    for e in (1, -1):
        for d in FIELDS:
            g = grade_by_factor(P[(n, e)], d)[0]; pr = predicted_cycle_grade(n, e, d); ka6["cells"] += 1
            if g != "-": ka6["positive"].append({"n": n, "class": "bal" if e == 1 else "tw", "d": d, "grade": g})
            if g != pr: ka6["fail"].append({"n": n, "class": e, "d": d, "got": g, "predicted": pr})
ka6["n_positive"] = len(ka6["positive"]); ka6["pass"] = not ka6["fail"]; out["gates"]["KA-6"] = ka6
L(f"KA-6 cycle grade table: {ka6['cells']} cells, {ka6['n_positive']} positive, fails {len(ka6['fail'])} -> {ka6['pass']}")
for c in ka6["positive"]:
    if c["grade"] in ("strict", "core"): L(f"     n={c['n']:>2} {c['class']:<3} √{c['d']:<2} {c['grade']}")

# KA-7 ------------------------------------------------------------------------------------------------------------
ka7 = {"counts": {}}
for n in range(3, 12):
    ka7["counts"][n] = len(unicyclic_graphs(n))       # asserts A001429 inside
ka7["pass"] = all(ka7["counts"][n] == A001429[n] for n in ka7["counts"]); out["gates"]["KA-7"] = ka7
L(f"KA-7 unicyclic enumerator vs A001429, n ≤ 11: {ka7['counts']} -> {ka7['pass']}")

# KA-8 ------------------------------------------------------------------------------------------------------------
ka8 = {"trees": 0, "cells": 0, "grade_mismatch": [], "charpoly_mismatch": 0}
for n in range(2, 11):
    for T in nx.nonisomorphic_trees(n):
        A = adjacency(T, n); p = charpoly_exact(A); ka8["trees"] += 1
        Cm = sp.Matrix(n, n, lambda i, j: int(2 * (i == j) - A[i, j]))
        if sp.expand(p - Cm.charpoly(t).as_expr()) != 0: ka8["charpoly_mismatch"] += 1
        for d in FIELDS:
            ka8["cells"] += 1
            if grade_by_factor(p, d)[0] != grade12(p, d): ka8["grade_mismatch"].append((n, sorted(T.edges()), d))
for G in unicyclic_graphs(8):
    A = adjacency(G, 8); B = twist(A, cycle_basis_single(A))
    for M in (A, B):
        Cm = sp.Matrix(8, 8, lambda i, j: int(2 * (i == j) - M[i, j]))
        if sp.expand(charpoly_exact(M) - Cm.charpoly(t).as_expr()) != 0: ka8["charpoly_mismatch"] += 1
ka8["pass"] = not ka8["grade_mismatch"] and ka8["charpoly_mismatch"] == 0; out["gates"]["KA-8"] = ka8
L(f"KA-8 per-factor grade = exp_12 grade on {ka8['trees']} trees × 7 fields ({ka8['cells']} cells): mismatches "
  f"{len(ka8['grade_mismatch'])}; charpoly mismatches {ka8['charpoly_mismatch']} -> {ka8['pass']}")

out["all_pass"] = all(g["pass"] for g in out["gates"].values()); out["seconds"] = round(time.time() - T0, 1)
RES.mkdir(exist_ok=True)
(RES / f"explore_f0_gates_{ts}.json").write_text(json.dumps(out, indent=1, default=str))
L(f"wrote results/explore_f0_gates_{ts}.json [{out['seconds']}s]")
assert out["all_pass"], "a gate failed — the round does not seal"
L("gates passed")
