#!/usr/bin/env python3
"""exp_31 — R1c: score the paired, matched-occupancy, fresh-seed grid against the SEALED registration.

    python exp_31_paired_ledger.py results/exp_31_paired_ledger_grid_full_<ts>.json
    python exp_31_paired_ledger.py --selftest

Reads the grid JSON from reality-engine POC-12 exp_04_aggregate.py on results/full_r1c/ (copied into
results/). REFUSES any seed outside the registered fresh set, any size but full, any grid short of the
thirty registered runs, and any run missing the connectivity observable. Every test is a WITHIN-SEED
difference (registration §3): the seed fixes the initial condition for every arm. Thresholds below are
the registration's, verbatim (journals/2026-09-14_exp31_registration.md §4). No fallback numbers.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
JOURNAL = HERE.parent / "journals" / "2026-09-14_exp31_registration.md"

# ---- the sealed thresholds (registration §1/§4/§7), verbatim ------------------------------------
SIZE = "full"
BOX_OVER_RANGE_MIN = 3.0               # box / 2r0 >= 3, carried from exp_30
SEEDS = (7, 8, 9, 10, 11, 12)          # six fresh seeds; anything else VOIDS the grid
KAPPAS = (0.0, 0.5, 1.0, 1.25, None)   # the five arms; None = inf; thirty runs
KAPPA_CLAIM = 0.5                      # T1/T2: the claim arm
KAPPA_CONTRAST = 1.0                   # T3: the fattening regime
KAPPA_EDGE = 1.25                      # T4: the edge arm
Q_REG, Q_SPINE, Q_BODY = "conn_q10", "conn_q05", "conn_q20"
T1_SIGN = 6                            # T1: sign 6/6
T1_MEAN_MIN = 0.05                     # T1: mean Δ1 >= 0.05
T1_T_MIN = 3.0                         # T1: paired t >= 3.0 (dof 5)
T2_SIGN_MIN = 5                        # T2: sign >= 5/6 at q = 0.05
T2_T_MIN = 2.0                         # T2: paired t >= 2.0
CV_RATIO_BAND = (0.85, 1.15)           # T2: cv(k=0.5)/cv(k=0) in [0.85, 1.15], 6/6
T3_SIGN_MIN = 5                        # T3: q20 positive in >= 5/6 AND q05 negative in >= 5/6
T3_CV_RATIO_MAX = 0.70                 # T3: cv(k=1)/cv(k=0) < 0.70, 6/6
T4_SIGN = 6                            # T4: 6/6 — work(k=1.25) > 0, spine(k=1.25) below gravity, work(k=1) < 0
TRANSFER_RESIDUAL_MAX = 1e-6           # gates (invalidate the run, not the claim)
CLOSURE_PAC_MAX = 0.05
AT_CAP_MAX = 0.02
RESOLUTION = 0.02                      # §7: |Δ1| <= 0.02 in 6/6 -> recorded as T1 FAIL, not vacuity
BUDGET_BINDS_MIN = 0.01                # §7: vacuous if the budget never binds at k = 0.5
SELFTEST_STRINGS = ["Sign 6/6", "mean Δ₁ ≥ 0.05", "paired t ≥ 3.0", "sign ≥ 5/6", "paired t ≥ 2.0",
                    "cv(κ = 0.5) / cv(κ = 0) ∈ [0.85, 1.15] in 6/6", "at q = 0.20 positive in ≥ 5/6",
                    "at q = 0.05 negative in ≥ 5/6", "cv(κ = 1) / cv(κ = 0) < 0.70\nin 6/6", "6/6 seeds",
                    "`work_pressure_cum / P(0) > 0`", "Seeds:** {7, 8, 9, 10, 11, 12}", "|Δ₁| ≤ 0.02 in 6/6",
                    "`budget_bound_frac_max < 0.01`", "box / 2r₀ = 3.0 ≥ 3"]


def selftest():
    text = JOURNAL.read_text(encoding="utf-8")
    missing = [s for s in SELFTEST_STRINGS if s not in text]
    if missing:
        raise SystemExit(f"selftest FAILED — registration text does not carry: {missing}")
    print(f"selftest OK — {len(SELFTEST_STRINGS)} threshold strings present in {JOURNAL.name}")


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(diffs=[float(x) for x in d], mean=float(d.mean()), se=float(se),
                t=(float(d.mean() / se) if se > 0 else float("inf")), pos=int((d > 0).sum()), neg=int((d < 0).sum()))


def load(paths):
    if len(paths) != 1:
        raise SystemExit("exactly one full-size grid is scored in this round (no fallback)")
    return json.loads(Path(paths[0]).read_text(encoding="utf-8"))


def check_grid(grid):
    runs = grid["runs"]
    bad = sorted({r["seed"] for r in runs if r["seed"] not in SEEDS})
    if bad:
        raise SystemExit(f"GRID VOID — seeds outside the registered fresh set: {bad} (§7)")
    if {r["size"] for r in runs} != {SIZE}:
        raise SystemExit(f"GRID VOID — sizes {set(r['size'] for r in runs)}; this round is {SIZE} only")
    have = {(r["kappa"], r["seed"]) for r in runs}; want = {(k, s) for k in KAPPAS for s in SEEDS}
    if have != want:
        raise SystemExit(f"GRID INCOMPLETE — missing {sorted(want - have, key=str)}; extra {sorted(have - want, key=str)}")
    for r in runs:
        c = r["config"]
        if c["n"] != 4000 or c["box"] / (2 * c["r0"]) < BOX_OVER_RANGE_MIN:
            raise SystemExit(f"GRID VOID — {r['kappa_label']} s{r['seed']}: n={c['n']} box/2r0={c['box']/(2*c['r0']):.2f}")
        for q in (Q_REG, Q_SPINE, Q_BODY):
            v = r["_summary"].get(q)
            if v is None or not np.isfinite(v):
                raise SystemExit(f"GRID VOID — {r['kappa_label']} s{r['seed']}: {q} not recorded (instrument predates the run?)")
    problems = []
    for r in runs:
        s = r["_summary"]
        if not s["finite"]: problems.append(f"{r['kappa_label']} s{r['seed']}: non-finite")
        if s["at_cap_max"] > AT_CAP_MAX: problems.append(f"{r['kappa_label']} s{r['seed']}: at_cap {s['at_cap_max']:.3f}")
        if r["kappa"] is not None and r["kappa"] > 0:
            if s["transfer_residual_max"] > TRANSFER_RESIDUAL_MAX: problems.append(f"{r['kappa_label']} s{r['seed']}: transfer residual {s['transfer_residual_max']:.2e}")
            if s["closure_pac_max"] > CLOSURE_PAC_MAX: problems.append(f"{r['kappa_label']} s{r['seed']}: closure {s['closure_pac_max']:.3f}")
    return problems


def score(grid):
    runs = grid["runs"]; floor = grid["floors"][SIZE]
    def arm(k): return {r["seed"]: r for r in runs if r["kappa"] == k}
    def col(k, key): return [arm(k)[s]["_summary"][key] for s in SEEDS]
    def work(k): return [arm(k)[s]["_summary"]["work_pressure_over_p0"] for s in SEEDS]
    res = {}
    # T1 the claim: conn_q10, k=0.5 minus k=0, paired
    p1 = paired(col(KAPPA_CLAIM, Q_REG), col(0.0, Q_REG))
    res["T1"] = dict(**p1, ok=bool(p1["pos"] == T1_SIGN and p1["mean"] >= T1_MEAN_MIN and p1["t"] >= T1_T_MIN),
                     resolution_fail=bool(all(abs(x) <= RESOLUTION for x in p1["diffs"])))
    # T2 gravity's web, better connected: spine lift and contrast preserved
    p2 = paired(col(KAPPA_CLAIM, Q_SPINE), col(0.0, Q_SPINE))
    cvr = [a / b for a, b in zip(col(KAPPA_CLAIM, "cv"), col(0.0, "cv"))]
    res["T2"] = dict(**p2, cv_ratio=cvr, ok=bool(p2["pos"] >= T2_SIGN_MIN and p2["t"] >= T2_T_MIN and all(CV_RATIO_BAND[0] <= x <= CV_RATIO_BAND[1] for x in cvr)))
    # T3 k=1's signature: body up, spine down, contrast halved
    p3b = paired(col(KAPPA_CONTRAST, Q_BODY), col(KAPPA_CLAIM, Q_BODY)); p3s = paired(col(KAPPA_CONTRAST, Q_SPINE), col(KAPPA_CLAIM, Q_SPINE))
    cvr1 = [a / b for a, b in zip(col(KAPPA_CONTRAST, "cv"), col(0.0, "cv"))]
    res["T3"] = dict(body=p3b, spine=p3s, cv_ratio=cvr1, ok=bool(p3b["pos"] >= T3_SIGN_MIN and p3s["neg"] >= T3_SIGN_MIN and all(x < T3_CV_RATIO_MAX for x in cvr1)))
    # T4 the edge: work positive at 1.25, spine below gravity at 1.25, work negative at 1
    w125, w1 = work(KAPPA_EDGE), work(KAPPA_CONTRAST); p4 = paired(col(KAPPA_EDGE, Q_SPINE), col(0.0, Q_SPINE))
    res["T4"] = dict(work_k125=w125, work_k1=w1, spine=p4, ok=bool(sum(x > 0 for x in w125) == T4_SIGN and p4["neg"] == T4_SIGN and sum(x < 0 for x in w1) == T4_SIGN))
    binds = [b if b is not None else 0.0 for b in col(KAPPA_CLAIM, "budget_bound_frac_max")]
    g0 = col(0.0, Q_REG); fl_m, fl_s = floor[f"{Q_REG}_mean"], max(floor[f"{Q_REG}_std"], 1e-9)
    vac = dict(no_web_to_add_to=bool(sum(x <= fl_m + 2 * fl_s for x in g0) >= 3), budget_never_binds=bool(max(binds) < BUDGET_BINDS_MIN),
               resolution_recorded_as_T1_fail=res["T1"]["resolution_fail"])
    side = dict(SP1_legacy_perc_lift=paired(col(KAPPA_CLAIM, "perc"), col(0.0, "perc")),
                SP2_occ_k05_minus_g0=[a - b for a, b in zip(col(KAPPA_CLAIM, "occ"), col(0.0, "occ"))],
                SP2_occ_k1_minus_g0=[a - b for a, b in zip(col(KAPPA_CONTRAST, "occ"), col(0.0, "occ"))],
                SP4_ke_order=[bool(arm(1.0)[s]["_summary"]["ke_over_u"] < arm(0.5)[s]["_summary"]["ke_over_u"] < arm(0.0)[s]["_summary"]["ke_over_u"] and arm(1.25)[s]["_summary"]["ke_over_u"] > arm(1.0)[s]["_summary"]["ke_over_u"]) for s in SEEDS],
                SP5_floor_below_g0=[bool(x > fl_m) for x in g0])
    tests = {t: ("PASS" if res[t]["ok"] else "FAIL") for t in ("T1", "T2", "T3", "T4")}
    return dict(tests=tests, score=sum(v == "PASS" for v in tests.values()), detail=res, floor=floor, vacuous=vac,
                side_predictions=side, kill=dict(adder_question_closed=tests["T1"] == "FAIL"),
                arms={str(k): {q: col(k, q) for q in (Q_REG, Q_SPINE, Q_BODY, "perc", "occ", "cv", "void", "ke_over_u")} for k in KAPPAS})


def main(argv):
    if "--selftest" in argv:
        selftest(); return 0
    selftest()
    grid = load([a for a in argv if not a.startswith("--")])
    problems = check_grid(grid)
    if problems:
        print("INSTRUMENT GATE FAILURES (the run is invalid, not the claim):"); [print("  -", p) for p in problems]; return 2
    out = score(grid); out["reality_engine_commit"] = grid.get("commit"); out["registration"] = JOURNAL.name; out["grid_n_runs"] = grid.get("n_runs")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"); dest = RESULTS / f"exp_31_paired_ledger_{ts}.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    r = out["detail"]; rd = lambda xs, n=3: [round(x, n) for x in xs]
    print(f"exp_31 R1c — {SIZE}, seeds {SEEDS}, reality-engine commit {out['reality_engine_commit']}, {out['grid_n_runs']} runs")
    t = r["T1"]; print(f"  T1 Δ1 = conn_q10(0.5) − conn_q10(0): {rd(t['diffs'])} mean {t['mean']:+.3f} ± {t['se']:.3f} t {t['t']:.2f} sign {t['pos']}/6 -> {out['tests']['T1']}")
    t = r["T2"]; print(f"  T2 Δ2 spine: {rd(t['diffs'])} mean {t['mean']:+.3f} t {t['t']:.2f} sign {t['pos']}/6; cv ratio {rd(t['cv_ratio'],2)} -> {out['tests']['T2']}")
    t = r["T3"]; print(f"  T3 k=1−k=0.5: body {rd(t['body']['diffs'])} ({t['body']['pos']}/6 +); spine {rd(t['spine']['diffs'])} ({t['spine']['neg']}/6 −); cv ratio {rd(t['cv_ratio'],2)} -> {out['tests']['T3']}")
    t = r["T4"]; print(f"  T4 work/P0 k=1.25 {rd(t['work_k125'],2)}, k=1 {rd(t['work_k1'],2)}; spine k=1.25−G0 {rd(t['spine']['diffs'])} ({t['spine']['neg']}/6 −) -> {out['tests']['T4']}")
    print(f"  floor conn_q10 {out['floor']['conn_q10_mean']:.3f}±{out['floor']['conn_q10_std']:.3f}  vacuous {out['vacuous']}")
    print(f"\nSCORE {out['score']}/4   kill fires (adder question closed): {out['kill']['adder_question_closed']}\nwrote results/{dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
