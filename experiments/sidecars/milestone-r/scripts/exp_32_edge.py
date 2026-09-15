#!/usr/bin/env python3
"""exp_32 — the edge as a number: score the fine sweep, the gravity arms and the size arm against the
SEALED registration.

    python exp_32_edge.py results/exp_32_edge_grid_fine_<ts>.json results/exp_32_edge_grid_g_<ts>.json results/exp_32_edge_grid_double_<ts>.json
    python exp_32_edge.py --selftest

Three grids from reality-engine POC-12 exp_04_aggregate.py (fine: results/full_r2a_fine, g arms:
results/full_r2a_g, size arm: results/full_r2a_n8000), copied into results/. REFUSES seeds outside the
registered fresh set and any arm short of its registered runs. Every statistic is read from the grid
(`_summary`), never recomputed. Thresholds below are the registration's verbatim
(journals/2026-09-15_exp32_registration.md §1/§4/§7).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
JOURNAL = HERE.parent / "journals" / "2026-09-15_exp32_registration.md"

# ---- the sealed objects and thresholds, verbatim ------------------------------------------------
SEEDS = (13, 14, 15)                                   # fresh; anything else VOIDS the grid
FINE = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25)               # the fine sweep (base geometry, g = 1.5)
G_ARMS = (0.75, 3.0)                                   # the gravity arms, at kappa in {1.00, 1.25}
EDGE_KAPPAS = (1.0, 1.25)
BASE_G = 1.5
SIZE_ARM = "double"                                    # n = 8000, box 75.6, at kappa in {1.00, 1.25}
KC_SPREAD_MAX = 0.05                                   # T1: the three seeds' brackets overlap or are adjacent
T2_FRAC_BAND = (0.45, 0.55)                            # T2: positive-work fraction at the crossing step
T4_FRAC_BAND = (0.45, 0.60)                            # T4: positive-work fraction at kappa = 1.25, n = 8000
TRANSFER_RESIDUAL_MAX = 1e-6                           # gates
CLOSURE_PAC_MAX = 0.05
AT_CAP_MAX = 0.02
IDENTITY_TOL = 0.02                                    # |dE_SEC − (T − W_p)| <= 0.02 |U0|  (re-checked from the marks' summary)
UNRESOLVED = 0.02                                      # §7: |W_p| <= 0.02 |U0| at every fine kappa -> UNSCORED
BUDGET_BINDS_MIN = 0.01                                # §7: an arm whose budget never binds is UNSCORED
SELFTEST_STRINGS = ["exactly once between\nκ = 1.00 and 1.25", "κ_c spread ≤ 0.05", "same bracketing step as the total (or the adjacent one)",
                    "within [0.45, 0.55]", "g = 0.75, g = 3.0", "W_p < 0 at κ = 1.00 and W_p > 0 at κ = 1.25",
                    "n = 8000, 3/3 seeds", "within [0.45, 0.60]", "|ΔE_SEC − (T − W_p)| ≤ 0.02 |U₀|",
                    "W_p within ±0.02 |U₀| of zero at every κ", "`budget_bound_frac_max` < 0.01",
                    "Seeds:** {13, 14, 15}", "resolution 0.025"]


def selftest():
    text = JOURNAL.read_text(encoding="utf-8")
    missing = [s for s in SELFTEST_STRINGS if s not in text]
    if missing:
        raise SystemExit(f"selftest FAILED — registration text does not carry: {missing}")
    print(f"selftest OK — {len(SELFTEST_STRINGS)} threshold strings present in {JOURNAL.name}")


def load(paths):
    grids = {}
    for p in paths:
        g = json.loads(Path(p).read_text(encoding="utf-8"))
        runs = g["runs"]
        sizes = {r["size"] for r in runs}; gs = {round(r["config"]["g"], 3) for r in runs}
        if sizes == {SIZE_ARM}: key = "double"
        elif gs == {BASE_G}: key = "fine"
        elif gs <= set(G_ARMS): key = "g"
        else: raise SystemExit(f"{p}: cannot classify grid (sizes {sizes}, g {gs})")
        grids[key] = g
    for k in ("fine", "g", "double"):
        if k not in grids:
            raise SystemExit(f"missing the {k} grid (no fallback)")
    return grids


def wp_u0(r):
    return r["_summary"]["work_pressure_cum"] / r["_summary"]["u0"]


def check(grids):
    problems = []
    for key, g in grids.items():
        runs = g["runs"]
        bad = sorted({r["seed"] for r in runs if r["seed"] not in SEEDS})
        if bad:
            raise SystemExit(f"GRID VOID ({key}) — seeds outside the registered fresh set: {bad}")
        want = ({(k, BASE_G, "full") for k in FINE} if key == "fine" else
                {(k, gg, "full") for k in EDGE_KAPPAS for gg in G_ARMS} if key == "g" else
                {(k, BASE_G, SIZE_ARM) for k in EDGE_KAPPAS})
        have = {(r["kappa"], round(r["config"]["g"], 3), r["size"]) for r in runs}
        if have != want:
            raise SystemExit(f"GRID INCOMPLETE ({key}) — missing {sorted(want - have)}; extra {sorted(have - want)}")
        for r in runs:
            s = r["_summary"]
            if not s["finite"]: problems.append(f"{key} k={r['kappa']} s{r['seed']}: non-finite")
            if s["at_cap_max"] > AT_CAP_MAX: problems.append(f"{key} k={r['kappa']} s{r['seed']}: at_cap {s['at_cap_max']:.3f}")
            if s["transfer_residual_max"] > TRANSFER_RESIDUAL_MAX: problems.append(f"{key} k={r['kappa']} s{r['seed']}: transfer residual {s['transfer_residual_max']:.2e}")
            if s["closure_pac_max"] > CLOSURE_PAC_MAX: problems.append(f"{key} k={r['kappa']} s{r['seed']}: closure {s['closure_pac_max']:.3f}")
            # the derivation's identity, re-checked from the recorded end values: dE_SEC = T − W_p
            ident = abs(s["e_sec_end"] - (s["sec_transfer_cum"] - s["work_pressure_cum"])) / s["u0"]
            if ident > IDENTITY_TOL: problems.append(f"{key} k={r['kappa']} s{r['seed']}: ledger identity off by {ident:.3f} |U0|")
    return problems


def score(grids):
    fine, garm, dbl = grids["fine"], grids["g"], grids["double"]
    def arm(g, k, gg=BASE_G, size="full"):
        return {r["seed"]: r for r in g["runs"] if r["kappa"] == k and round(r["config"]["g"], 3) == gg and r["size"] == size}
    res = {}
    # T1: a single sign change on the fine grid, seeds agree within a step
    brackets, crossings, medbr = {}, {}, {}
    unresolved = True
    for s in SEEDS:
        w = [wp_u0(arm(fine, k)[s]) for k in FINE]
        med = [arm(fine, k)[s]["_summary"]["work_pressure_median_end"] for k in FINE]
        if any(abs(x) > UNRESOLVED for x in w): unresolved = False
        signs = [x > 0 for x in w]
        changes = [i for i in range(len(FINE) - 1) if signs[i] != signs[i + 1]]
        crossings[s] = len(changes)
        brackets[s] = ((FINE[changes[0]] + FINE[changes[0] + 1]) / 2 if len(changes) == 1 and not signs[0] and signs[-1] else None)
        msigns = [x > 0 for x in med]
        mch = [i for i in range(len(FINE) - 1) if msigns[i] != msigns[i + 1]]
        medbr[s] = ((FINE[mch[0]] + FINE[mch[0] + 1]) / 2 if len(mch) == 1 else None)
        res.setdefault("fine", {})[s] = dict(wp=w, median=med, frac=[arm(fine, k)[s]["_summary"]["work_pressure_pos_frac"] for k in FINE],
                                             esec_over_T=[arm(fine, k)[s]["_summary"]["e_sec_end"] / arm(fine, k)[s]["_summary"]["sec_transfer_cum"] for k in FINE])
    kcs = [brackets[s] for s in SEEDS]
    t1_ok = all(crossings[s] == 1 and brackets[s] is not None for s in SEEDS) and (max(kcs) - min(kcs) <= KC_SPREAD_MAX if all(k is not None for k in kcs) else False)
    res["T1"] = dict(crossings=crossings, kappa_c_per_seed=brackets, kappa_c=(float(np.mean(kcs)) if all(k is not None for k in kcs) else None), ok=bool(t1_ok))
    # T2: the median crosses in the same or adjacent step; the fraction at the crossing step within band
    t2 = []
    for s in SEEDS:
        same = brackets[s] is not None and medbr[s] is not None and abs(brackets[s] - medbr[s]) <= 0.05 + 1e-9
        if brackets[s] is None: t2.append(False); continue
        i = min(range(len(FINE) - 1), key=lambda j: abs(FINE[j] - (brackets[s] - 0.025)))   # the bracket's lower grid point (float-safe; a 1.2000000000000002 must not crash the seal)
        frac_step = 0.5 * (res["fine"][s]["frac"][i] + res["fine"][s]["frac"][i + 1])
        t2.append(bool(same and T2_FRAC_BAND[0] <= frac_step <= T2_FRAC_BAND[1]))
    res["T2"] = dict(median_bracket=medbr, per_seed=t2, ok=bool(all(t2)))
    # T3: the gravity arms — sign at 1.00 negative and at 1.25 positive, 3/3 for each g
    t3 = {}
    for gg in G_ARMS:
        neg = [wp_u0(arm(garm, 1.0, gg)[s]) < 0 for s in SEEDS]; pos = [wp_u0(arm(garm, 1.25, gg)[s]) > 0 for s in SEEDS]
        t3[str(gg)] = dict(wp_k1=[wp_u0(arm(garm, 1.0, gg)[s]) for s in SEEDS], wp_k125=[wp_u0(arm(garm, 1.25, gg)[s]) for s in SEEDS],
                           reservoir_k1=[arm(garm, 1.0, gg)[s]["_summary"]["e_sec_end"] / arm(garm, 1.0, gg)[s]["_summary"]["u0"] for s in SEEDS],
                           ok=bool(all(neg) and all(pos)))
    res["T3"] = dict(arms=t3, ok=bool(all(v["ok"] for v in t3.values())))
    # T4: the size arm
    neg = [wp_u0(arm(dbl, 1.0, BASE_G, SIZE_ARM)[s]) < 0 for s in SEEDS]; pos = [wp_u0(arm(dbl, 1.25, BASE_G, SIZE_ARM)[s]) > 0 for s in SEEDS]
    fr = [arm(dbl, 1.25, BASE_G, SIZE_ARM)[s]["_summary"]["work_pressure_pos_frac"] for s in SEEDS]
    res["T4"] = dict(wp_k1=[wp_u0(arm(dbl, 1.0, BASE_G, SIZE_ARM)[s]) for s in SEEDS], wp_k125=[wp_u0(arm(dbl, 1.25, BASE_G, SIZE_ARM)[s]) for s in SEEDS], frac_k125=fr,
                     ok=bool(all(neg) and all(pos) and all(T4_FRAC_BAND[0] <= f <= T4_FRAC_BAND[1] for f in fr)))
    binds = {key: min((r["_summary"]["budget_bound_frac_max"] or 0.0) for r in g["runs"]) for key, g in grids.items()}
    vac = dict(edge_unresolved_on_fine_grid=unresolved, budget_never_binds={k: v < BUDGET_BINDS_MIN for k, v in binds.items()})
    tests = {t: ("PASS" if res[t]["ok"] else "FAIL") for t in ("T1", "T2", "T3", "T4")}
    if unresolved: tests["T1"] = tests["T2"] = "UNSCORED"
    side = dict(SP1_compression_crosses_with_wp={s: [x < 1 for x in res["fine"][s]["esec_over_T"]] for s in SEEDS},
                SP4_spine_k125_below_k1={f"{k}": [arm(g, 1.25, gg, sz)[s]["_summary"]["conn_q05"] < arm(g, 1.0, gg, sz)[s]["_summary"]["conn_q05"] for s in SEEDS]
                                         for k, (g, gg, sz) in {"base": (fine, BASE_G, "full"), "g0.75": (garm, 0.75, "full"), "g3": (garm, 3.0, "full"), "n8000": (dbl, BASE_G, SIZE_ARM)}.items()})
    return dict(tests=tests, score=sum(v == "PASS" for v in tests.values()), detail=res, vacuous=vac, side_predictions=side,
                kill=dict(edge_not_a_number=tests["T1"] == "FAIL", edge_moves_with_g=tests["T3"] == "FAIL"))


def main(argv):
    if "--selftest" in argv:
        selftest(); return 0
    selftest()
    grids = load([a for a in argv if not a.startswith("--")])
    problems = check(grids)
    if problems:
        print("INSTRUMENT GATE FAILURES (the run is invalid, not the claim):"); [print("  -", p) for p in problems]; return 2
    out = score(grids); out["reality_engine_commits"] = {k: g.get("commit") for k, g in grids.items()}; out["registration"] = JOURNAL.name
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"); dest = RESULTS / f"exp_32_edge_{ts}.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    r = out["detail"]
    print(f"exp_32 — the edge as a number; seeds {SEEDS}; commits {out['reality_engine_commits']}")
    for s in SEEDS:
        f = r["fine"][s]; print(f"  seed {s}: W_p/|U0| on {FINE}: {[round(x,3) for x in f['wp']]}  median: {[round(x,3) for x in f['median']]}  frac: {[round(x,2) for x in f['frac']]}  Esec/T: {[round(x,2) for x in f['esec_over_T']]}")
    t = r["T1"]; print(f"  T1 crossings {t['crossings']} kappa_c per seed {t['kappa_c_per_seed']} -> kappa_c = {t['kappa_c']} -> {out['tests']['T1']}")
    t = r["T2"]; print(f"  T2 median brackets {t['median_bracket']} per seed {t['per_seed']} -> {out['tests']['T2']}")
    t = r["T3"]; print(f"  T3 " + "; ".join(f"g={k}: k1 {[round(x,2) for x in v['wp_k1']]} k1.25 {[round(x,2) for x in v['wp_k125']]} reservoir@1 {[round(x,2) for x in v['reservoir_k1']]} {'ok' if v['ok'] else 'FAIL'}" for k, v in t['arms'].items()) + f" -> {out['tests']['T3']}")
    t = r["T4"]; print(f"  T4 n=8000: k1 {[round(x,2) for x in t['wp_k1']]} k1.25 {[round(x,2) for x in t['wp_k125']]} frac@1.25 {[round(x,2) for x in t['frac_k125']]} -> {out['tests']['T4']}")
    print(f"  vacuous {out['vacuous']}\n  side {out['side_predictions']}")
    print(f"\nSCORE {out['score']}/4   kills: {out['kill']}\nwrote results/{dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
