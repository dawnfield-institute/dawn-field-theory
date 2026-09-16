#!/usr/bin/env python3
"""exp_32c — is kappa_c predicted, seed by seed, from the initial large-scale power?

    python exp_32c_predict.py results/exp_32c_predict_grid_fine_<ts>.json
    python exp_32c_predict.py --selftest

One grid from reality-engine POC-12 exp_04_aggregate.py (results/full_r2c_fine), copied into
results/. REFUSES seeds outside the registered fresh set and any grid short of its registered runs.
Every statistic is read from the grid (`_summary`), never recomputed. The line and the six
predictions below are the registration's verbatim (journals/2026-09-16_exp32c_registration.md
§1/§4), and nothing here refits anything.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
JOURNAL = HERE.parent / "journals" / "2026-09-16_exp32c_registration.md"

# ---- the sealed objects, verbatim -----------------------------------------------------------
SEEDS = (19, 20, 21, 22, 23, 24)                       # fresh; anything else VOIDS the grid
FINE = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3)
BASE_G = 1.5
B0, B1 = 1.011924, 0.874965                            # THE FROZEN LINE, fitted on seeds 13-18
RESID_SD = 0.01555                                     # training residual sd
TRAIN_MEAN = 1.169100                                  # the null T1 must beat
F = {19: 0.213757, 20: 0.142599, 21: 0.126312,         # computed from each seed BEFORE any run
     22: 0.220468, 23: 0.137870, 24: 0.190808}
T3_MIN_IN_BAND = 4                                     # T3: at least 4 of 6 inside +/- 2 resid sd
TRANSFER_RESIDUAL_MAX = 1e-6
CLOSURE_PAC_MAX = 0.05
AT_CAP_MAX = 0.05
IDENTITY_TOL = 0.08
UNRESOLVED = 0.02
BUDGET_BINDS_MIN = 0.01
SELFTEST_STRINGS = [
    "kappa_c = 1.011924 + 0.874965 * f",
    "Training residual sd **0.01555**",
    "training mean κ_c **1.169100**",
    "| 19 | 0.213757 | **1.1990** |",
    "| 21 | 0.126312 | **1.1224** |",
    "| 24 | 0.190808 | **1.1789** |",
    "**Seeds:** {19, 20, 21, 22, 23, 24}",
    "κ ∈ {1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30} — 42 runs",
    "RMS error of the frozen\nline is **strictly smaller**",
    "Spearman ρ(f, κ_c) over the six fresh seeds is **> 0**",
    "At least **4 of 6** measured κ_c fall inside their ±2",
    "exactly\nonce across κ ∈ [1.00, 1.30], negative to positive",
    "|ΔE_SEC − (T − W_p)| ≤ 0.08 |U₀|",
    "W_p within ±0.02 |U₀| of zero at every κ",
    "`budget_bound_frac_max` < 0.01",
]


def selftest():
    text = JOURNAL.read_text(encoding="utf-8")
    missing = [s for s in SELFTEST_STRINGS if s not in text]
    if missing:
        raise SystemExit(f"selftest FAILED — registration text does not carry: {missing}")
    # the table in the seal must equal the line applied to the seal's own f, to the printed 4 dp
    for s, f in F.items():
        shown = f"| {s} | {f:.6f} | **{B0 + B1 * f:.4f}** |"
        if shown not in text:
            raise SystemExit(f"selftest FAILED — seed {s}: the seal's table does not carry {shown!r}")
    print(f"selftest OK — {len(SELFTEST_STRINGS)} threshold strings present in {JOURNAL.name}")


def load(paths):
    if len(paths) != 1:
        raise SystemExit("exactly one grid is registered (the fine sweep); got "
                         f"{len(paths)}")
    g = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    runs = g["runs"]
    bad = sorted({r["seed"] for r in runs if r["seed"] not in SEEDS})
    if bad:
        raise SystemExit(f"GRID VOID — seeds outside the registered fresh set: {bad}")
    want = {(s, k) for s in SEEDS for k in FINE}
    have = {(r["seed"], r["kappa"]) for r in runs}
    if have != want:
        raise SystemExit(f"GRID INCOMPLETE — missing {sorted(want - have)}; extra {sorted(have - want)}")
    return g


def check(g):
    problems = []
    for r in g["runs"]:
        s = r["_summary"]
        tag = f"k={r['kappa']} s{r['seed']}"
        if not s["finite"]:
            problems.append(f"{tag}: non-finite")
        if s["at_cap_max"] > AT_CAP_MAX:
            problems.append(f"{tag}: at_cap {s['at_cap_max']:.3f}")
        if s["transfer_residual_max"] > TRANSFER_RESIDUAL_MAX:
            problems.append(f"{tag}: transfer residual {s['transfer_residual_max']:.2e}")
        if s["closure_pac_max"] > CLOSURE_PAC_MAX:
            problems.append(f"{tag}: closure {s['closure_pac_max']:.3f}")
        ident = abs(s["e_sec_end"] - (s["sec_transfer_cum"] - s["work_pressure_cum"])) / s["u0"]
        if ident > IDENTITY_TOL:
            problems.append(f"{tag}: ledger identity off by {ident:.3f} |U0|")
    return problems


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    return float(np.corrcoef(ra, rb)[0, 1])


def score(g):
    by = {}
    for r in g["runs"]:
        by.setdefault(r["seed"], {})[r["kappa"]] = r["_summary"]
    res, kc, crossings, unresolved = {}, {}, {}, True
    for s in SEEDS:
        w = [by[s][k]["work_pressure_cum"] / by[s][k]["u0"] for k in FINE]
        if any(abs(x) > UNRESOLVED for x in w):
            unresolved = False
        signs = [x > 0 for x in w]
        ch = [i for i in range(len(FINE) - 1) if signs[i] != signs[i + 1]]
        crossings[s] = len(ch)
        if len(ch) == 1 and not signs[0] and signs[-1]:
            i = ch[0]
            kc[s] = FINE[i] + (FINE[i + 1] - FINE[i]) * (-w[i]) / (w[i + 1] - w[i])
        else:
            kc[s] = None
        res.setdefault("fine", {})[s] = dict(
            wp=w, kappa_c=kc[s], f=F[s], predicted=B0 + B1 * F[s],
            esec_over_T=[by[s][k]["e_sec_end"] / by[s][k]["sec_transfer_cum"] for k in FINE])

    have_all = all(kc[s] is not None for s in SEEDS)
    # T4 first: every other test needs kappa_c
    res["T4"] = dict(crossings=crossings, ok=bool(all(crossings[s] == 1 and kc[s] is not None for s in SEEDS)))

    if have_all:
        y = np.array([kc[s] for s in SEEDS])
        pred = np.array([B0 + B1 * F[s] for s in SEEDS])
        rms_line = float(np.sqrt(np.mean((y - pred) ** 2)))
        rms_null = float(np.sqrt(np.mean((y - TRAIN_MEAN) ** 2)))
        res["T1"] = dict(rms_line=rms_line, rms_null=rms_null,
                         skill=float(1 - rms_line ** 2 / rms_null ** 2), ok=bool(rms_line < rms_null))
        rho = spearman([F[s] for s in SEEDS], list(y))
        res["T2"] = dict(spearman=rho, ok=bool(rho > 0))
        inband = [bool(abs(kc[s] - (B0 + B1 * F[s])) <= 2 * RESID_SD) for s in SEEDS]
        res["T3"] = dict(in_band=dict(zip(map(str, SEEDS), inband)), n_in_band=int(sum(inband)),
                         ok=bool(sum(inband) >= T3_MIN_IN_BAND))
    else:
        for t in ("T1", "T2", "T3"):
            res[t] = dict(ok=False, note="kappa_c undefined on at least one seed")

    binds = min((r["_summary"]["budget_bound_frac_max"] or 0.0) for r in g["runs"])
    vac = dict(edge_unresolved_on_fine_grid=unresolved, budget_never_binds=binds < BUDGET_BINDS_MIN)
    tests = {t: ("PASS" if res[t]["ok"] else "FAIL") for t in ("T1", "T2", "T3", "T4")}
    if unresolved:
        for t in ("T1", "T2", "T3"):
            tests[t] = "UNSCORED"
    allkc = [1.2003, 1.1288, 1.1851, 1.1967, 1.1546, 1.1491] + [kc[s] for s in SEEDS if kc[s]]
    side = dict(SP1_twelve_seed_sd=float(np.std(allkc, ddof=1)) if len(allkc) > 6 else None,
                SP3_compression_crosses_with_wp={str(s): [x < 1 for x in res["fine"][s]["esec_over_T"]] for s in SEEDS})
    return dict(tests=tests, score=sum(v == "PASS" for v in tests.values()), detail=res,
                vacuous=vac, side_predictions=side,
                kill=dict(kappa_c_not_predictable=tests["T1"] == "FAIL"))


def main(argv):
    if "--selftest" in argv:
        selftest()
        return 0
    selftest()
    g = load([a for a in argv if not a.startswith("--")])
    problems = check(g)
    if problems:
        print("INSTRUMENT GATE FAILURES (the run is invalid, not the claim):")
        [print("  -", p) for p in problems]
        return 2
    out = score(g)
    out["reality_engine_commit"] = g.get("commit")
    out["registration"] = JOURNAL.name
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (RESULTS / f"exp_32c_predict_{ts}.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    r = out["detail"]
    print(f"exp_32c — kappa_c predicted from the initial large-scale power; seeds {SEEDS}; "
          f"commit {g.get('commit')}")
    print(f"  {'seed':>5} {'f':>9} {'predicted':>10} {'measured':>10} {'error':>9}  in +/-2sd")
    for s in SEEDS:
        d = r["fine"][s]
        m = d["kappa_c"]
        e = (m - d["predicted"]) if m is not None else float("nan")
        inb = abs(e) <= 2 * RESID_SD if m is not None else False
        print(f"  {s:>5} {d['f']:>9.6f} {d['predicted']:>10.4f} "
              f"{(f'{m:.4f}' if m is not None else 'none'):>10} {e:>+9.4f}  {'yes' if inb else 'NO'}")
    if "rms_line" in r["T1"]:
        print(f"  T1 rms(line) {r['T1']['rms_line']:.5f} vs rms(null={TRAIN_MEAN}) "
              f"{r['T1']['rms_null']:.5f}  skill {r['T1']['skill']:+.3f} -> {out['tests']['T1']}")
        print(f"  T2 spearman(f, kappa_c) = {r['T2']['spearman']:+.3f} -> {out['tests']['T2']}")
        print(f"  T3 {r['T3']['n_in_band']}/6 inside +/-2 resid sd -> {out['tests']['T3']}")
    print(f"  T4 crossings {r['T4']['crossings']} -> {out['tests']['T4']}")
    print(f"  vacuous {out['vacuous']}")
    print(f"  side {out['side_predictions']['SP1_twelve_seed_sd']}")
    print(f"\nSCORE {out['score']}/4   kill: {out['kill']}")
    print(f"wrote results/exp_32c_predict_{ts}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
