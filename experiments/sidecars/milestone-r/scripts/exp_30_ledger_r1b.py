#!/usr/bin/env python3
"""exp_30 — R1b: score the fresh-seed full-size ledger grid against the SEALED registration.

    python exp_30_ledger_r1b.py results/exp_30_ledger_r1b_grid_full_<ts>.json
    python exp_30_ledger_r1b.py --selftest      # thresholds byte-equal to the journal

Loads the grid JSON produced by reality-engine POC-12 exp_04_aggregate.py on results/full_r1b/
(copied into results/), re-verifies the instrument gates recorded in it, REFUSES any run whose
seed is not in the registered fresh set, and scores T1-T4 with the thresholds below — the
registration's, verbatim (journals/2026-09-14_exp30_registration.md §4). Exits nonzero if no grid
is given: no fallback, no transcribed numbers. Grid schema is exp_29's: per-run window metrics
under `_summary` (window = t in [10, 15], the mean over marks, never the t = 15 point).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
JOURNAL = HERE.parent / "journals" / "2026-09-14_exp30_registration.md"

# ---- the sealed thresholds (registration §1/§4/§7), verbatim ------------------------------------
SIZE = "full"                          # full size only; the proxy is retired for this question
BOX_OVER_RANGE_MIN = 3.0               # declared condition: box / 2r0 >= 3  (60 / 20 = 3.0)
SEEDS = (4, 5, 6)                      # fresh seeds; any other seed present VOIDS the grid
KAPPAS = (0.0, 0.5, 1.0, 1.5, None)    # the five arms; None = inf (the unbounded engine); 15 runs
KAPPA_CLAIM = 1.0                      # T1: the arm the claim is made at
KAPPA_NEW = 1.5                        # T2/T3: the predicted intermediate, never run before
SIGMA_MULT = 2.0                       # T1: mean margin > 2x the pooled sigma of the two arms, vs kappa = 0
KE_OVER_U_BOUND = 1.0                  # T2: KE/|U_grav| < 1 at every mark of the window, at kappa = 1 and 1.5
T3_KE_ORDER = (1.0, 1.5, None)         # T3a: KE/|U_grav| window mean ordered kappa = 1 < kappa = 1.5 < kappa = inf, per seed
T3_PERC_ORDER = (1.0, 1.5)             # T3b: percolation window mean ordered kappa = 1 > kappa = 1.5, per seed
BUDGET_BINDS_FULLY = 1.0               # T4: budget_bound_frac_max = 1 at kappa = 1
WORK_OVER_P0_ALLOWANCE = 1.10          # T4: pressure work <= P(0) within the integrator's 10% truncation allowance
TRANSFER_RESIDUAL_MAX = 1e-6           # instrument: the exact part of the ledger
CLOSURE_PAC_MAX = 0.05                 # instrument: per-tick drift of the total, coarse re-check
AT_CAP_MAX = 0.02                      # instrument: no run may lean on a numerical cap (exp_29 gate, carried)
BUDGET_BINDS_MIN = 0.01                # §7 vacuous if the budget never binds at kappa = 1
INDISTINGUISHABLE_SIGMA = 1.0          # §7: kappa = 1 within 1 pooled sigma of kappa = 0 in every seed -> recorded as T1 FAIL
SELFTEST_STRINGS = ["κ = 1", "3/3 seeds", "2× the pooled σ of the two arms", "κ = 0** control",
                    "KE/|U_grav| < 1 at\nevery mark", "κ = 1 < κ = 1.5 < κ = ∞", "κ = 1 > κ = 1.5",
                    "10% truncation\nallowance", "budget_bound_frac_max = 1", "box / 2r₀ ≥ 3",
                    "Seeds:** {4, 5, 6}", "seed ∈ {1, 2, 3}", "budget_bound_frac_max < 0.01",
                    "within 1 pooled σ of κ = 0"]


def selftest():
    text = JOURNAL.read_text(encoding="utf-8")
    missing = [s for s in SELFTEST_STRINGS if s not in text]
    if missing:
        raise SystemExit(f"selftest FAILED — registration text does not carry: {missing}")
    print(f"selftest OK — {len(SELFTEST_STRINGS)} threshold strings present in {JOURNAL.name}")


def pooled_std(a, b):
    """exp_29's definition, unchanged: sqrt((s_a^2 + s_b^2) / 2) across the two arms' seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))


def load(paths):
    if not paths:
        raise SystemExit("no grid given — nothing to score (no fallback)")
    if len(paths) != 1:
        raise SystemExit("exactly one full-size grid is scored in this round")
    return json.loads(Path(paths[0]).read_text(encoding="utf-8"))


def check_grid(grid):
    """Refusals (the grid is VOID) and instrument gates (the RUN is invalid, not the claim)."""
    runs = grid["runs"]
    bad_seed = sorted({r["seed"] for r in runs if r["seed"] not in SEEDS})
    if bad_seed:
        raise SystemExit(f"GRID VOID — seeds outside the registered fresh set present: {bad_seed} (§7)")
    sizes = {r["size"] for r in runs}
    if sizes != {SIZE}:
        raise SystemExit(f"GRID VOID — sizes {sizes}; this round is {SIZE} only (§1)")
    have = {(r["kappa"], r["seed"]) for r in runs}
    want = {(k, s) for k in KAPPAS for s in SEEDS}
    if have != want:
        raise SystemExit(f"GRID INCOMPLETE — missing {sorted(want - have, key=str)}; extra {sorted(have - want, key=str)}")
    for r in runs:
        c = r["config"]
        if c["n"] != 4000 or c["box"] / (2 * c["r0"]) < BOX_OVER_RANGE_MIN:
            raise SystemExit(f"GRID VOID — {r['kappa_label']} s{r['seed']}: n={c['n']} box/2r0={c['box']/(2*c['r0']):.2f} (§1 condition)")
    problems = []
    for r in runs:
        s = r["_summary"]
        if not s["finite"]:
            problems.append(f"{r['kappa_label']} s{r['seed']}: non-finite")
        if s["at_cap_max"] > AT_CAP_MAX:
            problems.append(f"{r['kappa_label']} s{r['seed']}: at_cap {s['at_cap_max']:.3f}")
        if r["kappa"] is not None and r["kappa"] > 0:
            if s["transfer_residual_max"] > TRANSFER_RESIDUAL_MAX:
                problems.append(f"{r['kappa_label']} s{r['seed']}: transfer residual {s['transfer_residual_max']:.2e}")
            if s["closure_pac_max"] > CLOSURE_PAC_MAX:
                problems.append(f"{r['kappa_label']} s{r['seed']}: closure {s['closure_pac_max']:.3f}")
    return problems


def score(grid):
    runs = grid["runs"]
    floor = grid["floors"][SIZE]

    def arm(k):
        return {r["seed"]: r for r in runs if r["kappa"] == k}

    def col(k, key):
        return [arm(k)[s]["_summary"][key] for s in SEEDS]

    res = {}
    # T1 the claim: kappa = 1 above kappa = 0 seed by seed, mean margin > 2x pooled sigma of the two arms
    p1, p0 = col(KAPPA_CLAIM, "perc"), col(0.0, "perc")
    margin, sd = float(np.mean(p1) - np.mean(p0)), pooled_std(p1, p0)
    seedwise = [a > b for a, b in zip(p1, p0)]
    indist = all(abs(a - b) <= INDISTINGUISHABLE_SIGMA * sd for a, b in zip(p1, p0))
    res["T1"] = dict(perc_k1=p1, perc_G0=p0, perc_Binf=col(None, "perc"), perc_k05=col(0.5, "perc"),
                     seedwise_vs_G0=seedwise, margin_G0=margin, pooled_std=sd,
                     sigma=(margin / sd if sd > 0 else float("inf")),
                     indistinguishable_recorded_as_fail=indist,
                     ok=bool(all(seedwise) and margin > SIGMA_MULT * sd and not indist))
    # T2 bound at kappa = 1 and 1.5, every mark of the window
    k1, k15 = col(KAPPA_CLAIM, "ke_over_u_max"), col(KAPPA_NEW, "ke_over_u_max")
    res["T2"] = dict(ke_over_u_max_k1=k1, ke_over_u_max_k15=k15,
                     ok=bool(all(v < KE_OVER_U_BOUND for v in k1 + k15)))
    # T3 the predicted position of kappa = 1.5: KE ordering 1 < 1.5 < inf AND percolation 1 > 1.5, per seed
    ke_ord, pc_ord = [], []
    for s in SEEDS:
        a, b, c = (arm(k)[s]["_summary"]["ke_over_u"] for k in T3_KE_ORDER)
        ke_ord.append(bool(a < b < c))
        pa, pb = (arm(k)[s]["_summary"]["perc"] for k in T3_PERC_ORDER)
        pc_ord.append(bool(pa > pb))
    res["T3"] = dict(ke_order_per_seed=ke_ord, perc_order_per_seed=pc_ord,
                     ke_over_u_k1=col(1.0, "ke_over_u"), ke_over_u_k15=col(1.5, "ke_over_u"),
                     ke_over_u_inf=col(None, "ke_over_u"), perc_k15=col(1.5, "perc"),
                     ok=bool(all(ke_ord) and all(pc_ord)))
    # T4 the ledger did the work, on every ledgered run
    led = [r for r in runs if r["kappa"] is not None and r["kappa"] > 0]
    exact = all(r["_summary"]["sec_transfer_cum"] <= r["budget0"] * (1 + 1e-6) for r in led)
    within = all((r["_summary"]["work_pressure_over_p0"] or 0.0) <= WORK_OVER_P0_ALLOWANCE for r in led)
    binds = [b if b is not None else 0.0 for b in col(KAPPA_CLAIM, "budget_bound_frac_max")]
    res["T4"] = dict(net_creation_le_p0=bool(exact), work_within_allowance=bool(within), bound_frac_max_k1=binds,
                     work_over_p0_k1=col(1.0, "work_pressure_over_p0"),
                     ok=bool(exact and within and all(b >= BUDGET_BINDS_FULLY for b in binds)))
    vac = dict(budget_never_binds_k1=bool(max(binds) < BUDGET_BINDS_MIN),
               gravity_above_floor=bool((np.mean(p0) - floor["percolation_mean"]) > 2 * max(floor["percolation_std"], 1e-9)),
               k1_indistinguishable_from_G0=indist)
    # side predictions (registered, unscored)
    side = dict(SP1_work_negative_k1=[w < 0 for w in col(1.0, "work_pressure_over_p0")],
                SP2_spent_frac_falls={str(k): col(k, "budget_frac_end") for k in (0.5, 1.0, 1.5)},
                SP3_no_floor_ticks_le_15={str(k): col(k, "floor_ticks") for k in (0.5, 1.0, 1.5)},
                SP4_k05_between=[b < m < a for a, m, b in zip(p1, col(0.5, "perc"), p0)],
                SP5_floor_below_all=bool(floor["percolation_mean"] < min(min(col(k, "perc")) for k in KAPPAS)))
    tests = {t: ("PASS" if res[t]["ok"] else "FAIL") for t in ("T1", "T2", "T3", "T4")}
    return dict(tests=tests, score=sum(v == "PASS" for v in tests.values()), detail=res, floor=floor,
                vacuous=vac, side_predictions=side, kill=dict(mapping_retired_as_adder=tests["T1"] == "FAIL"))


def main(argv):
    if "--selftest" in argv:
        selftest()
        return 0
    selftest()
    grid = load([a for a in argv if not a.startswith("--")])
    problems = check_grid(grid)
    if problems:
        print("INSTRUMENT GATE FAILURES (the run is invalid, not the claim):")
        for p in problems:
            print("  -", p)
        return 2
    out = score(grid)
    out["reality_engine_commit"] = grid.get("commit")
    out["registration"] = JOURNAL.name
    out["grid_n_runs"] = grid.get("n_runs")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = RESULTS / f"exp_30_ledger_r1b_{ts}.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    r = out["detail"]
    rd = lambda xs, n=3: [round(x, n) for x in xs]
    print(f"exp_30 R1b — {SIZE}, seeds {SEEDS}, reality-engine commit {out['reality_engine_commit']}, {out['grid_n_runs']} runs")
    t = r["T1"]; print(f"  T1 perc k=1 {rd(t['perc_k1'])} vs G0 {rd(t['perc_G0'])}: seedwise {t['seedwise_vs_G0']}, margin {t['margin_G0']:.3f},"
                       f" pooled std {t['pooled_std']:.3f} ({t['sigma']:.1f} sigma)  [k=0.5 {rd(t['perc_k05'])}; inf {rd(t['perc_Binf'])}] -> {out['tests']['T1']}")
    t = r["T2"]; print(f"  T2 max KE/|U| in window k=1 {rd(t['ke_over_u_max_k1'],2)} k=1.5 {rd(t['ke_over_u_max_k15'],2)} -> {out['tests']['T2']}")
    t = r["T3"]; print(f"  T3 KE/|U| 1<1.5<inf {t['ke_order_per_seed']} (k=1 {rd(t['ke_over_u_k1'],2)} k=1.5 {rd(t['ke_over_u_k15'],2)} inf {rd(t['ke_over_u_inf'],1)});"
                       f" perc 1>1.5 {t['perc_order_per_seed']} (k=1.5 {rd(t['perc_k15'])}) -> {out['tests']['T3']}")
    t = r["T4"]; print(f"  T4 exact {t['net_creation_le_p0']} within {t['work_within_allowance']} bound_frac_max@k1 {t['bound_frac_max_k1']}"
                       f" work/P0@k1 {rd(t['work_over_p0_k1'],2)} -> {out['tests']['T4']}")
    print(f"  floor {out['floor']['percolation_mean']:.3f}±{out['floor']['percolation_std']:.3f}  vacuous {out['vacuous']}")
    print(f"  side {out['side_predictions']}")
    print(f"\nSCORE {out['score']}/4   kill fires (mapping retired as structure-adder): {out['kill']['mapping_retired_as_adder']}")
    print(f"wrote results/{dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
