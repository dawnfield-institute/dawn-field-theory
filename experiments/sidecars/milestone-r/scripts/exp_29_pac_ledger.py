#!/usr/bin/env python3
"""exp_29 — score the ledger-on-particles grid against the SEALED registration.

    python exp_29_pac_ledger.py results/exp_29_pac_ledger_grid_proxy_<ts>.json [results/..._grid_full_<ts>.json]
    python exp_29_pac_ledger.py --selftest      # thresholds byte-equal to the journal

Loads one or more grid JSONs produced by reality-engine POC-12 exp_04_aggregate.py (copied into
results/), re-verifies the instrument gates recorded in them, and scores T1-T4 with the thresholds
below — the registration's, verbatim (journals/2026-09-06_exp29_registration.md §4). A test passes
only if it passes at EVERY size present (the proxy decides, n = 4000 confirms). Exits nonzero if
no grid is given or found: no fallback, no transcribed numbers.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
JOURNAL = HERE.parent / "journals" / "2026-09-06_exp29_registration.md"

# ---- the sealed thresholds (registration §4/§7), verbatim ---------------------------------------
KAPPA_TEST = 0.5                       # the ledgered arm the claims are made at
KAPPA_SWEEP = (0.5, 1.0, 2.0)          # the ordering T3 is scored on
SEEDS = (1, 2, 3)
SIGMA_MULT = 2.0                       # mean margin > 2x pooled within-seed std
KE_OVER_U_BOUND = 1.0                  # T2: KE/|U_grav| < 1 at every mark of the window, at kappa = 0.5 and 1
T3_ORDER = (1.0, 2.0, None)            # T3: KE/|U_grav| ordered kappa = 1 < kappa = 2 < kappa = inf (None), per seed
BUDGET_BINDS_FULLY = 1.0               # T4: budget_bound_frac_max = 1 at kappa = 0.5 (every would-grow particle clipped)
WORK_OVER_P0_ALLOWANCE = 1.10          # T4: pressure work <= P(0) within the integrator's 10% truncation allowance
TRANSFER_RESIDUAL_MAX = 1e-6           # instrument: the exact part of the ledger
CLOSURE_PAC_MAX = 0.05                 # instrument: per-tick drift of the total, coarse re-check
BUDGET_BINDS_MIN = 0.01                # vacuous if the budget never binds at any kappa <= 2
SELFTEST_STRINGS = ["κ = 0.5", "3/3 seeds", "2× the pooled within-seed σ", "KE/|U_grav| < 1 at every mark",
                    "κ = 1 < κ = 2 < κ = ∞", "10% truncation allowance", "budget_bound_frac_max = 1"]


def selftest() -> int:
    norm = lambda s: " ".join(s.split())
    text = norm(JOURNAL.read_text(encoding="utf-8"))
    missing = [s for s in SELFTEST_STRINGS if norm(s) not in text]
    print("thresholds byte-equal to the registration:", "OK" if not missing else f"MISSING {missing}")
    return 0 if not missing else 1


def load(argv):
    paths = [Path(a) for a in argv[1:] if a != "--selftest"]
    if not paths:
        cands = sorted(RESULTS.glob("exp_29_pac_ledger_grid_*.json"))
        if not cands:
            raise SystemExit("no results/exp_29_pac_ledger_grid_*.json — nothing to score (no fallback)")
        paths = cands
    runs, floors, commits = [], {}, []
    for p in paths:
        g = json.loads(p.read_text(encoding="utf-8")); runs += g["runs"]; floors.update(g.get("floors", {})); commits.append(g.get("commit"))
    return [p.name for p in paths], runs, floors, commits


def pooled_std(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)) if len(a) > 1 and len(b) > 1 else float("nan")


def score(runs, floors):
    out = {"gates": {}, "per_size": {}, "tests": {}, "kill": {}, "vacuous": {}}
    ledgered = [r for r in runs if r["kappa"] is not None and r["kappa"] > 0]
    out["gates"]["transfer_residual_le_1e-6"] = all(r["_summary"]["transfer_residual_max"] <= TRANSFER_RESIDUAL_MAX for r in ledgered)
    out["gates"]["closure_pac_le_0.05"] = all(r["_summary"]["closure_pac_max"] <= CLOSURE_PAC_MAX for r in ledgered)
    out["gates"]["all_finite"] = all(r["_summary"]["finite"] for r in runs)
    out["gates"]["at_cap_le_0.02"] = all(r["_summary"]["at_cap_max"] <= 0.02 for r in runs)
    sizes = sorted({r["size"] for r in runs}, key=lambda s: 0 if s == "proxy" else 1)
    for size in sizes:
        def arm(kappa):
            return {r["seed"]: r["_summary"] for r in runs if r["size"] == size and
                    ((kappa is None and r["kappa"] is None) or (kappa is not None and r["kappa"] == kappa))}
        A, B0, G0 = arm(KAPPA_TEST), arm(None), arm(0.0)
        res = {}
        # vacuity
        binds = [r["_summary"]["budget_bound_frac_max"] or 0.0 for r in runs if r["size"] == size and r["kappa"] is not None and 0 < r["kappa"] <= 2]
        fl = floors.get(size, {})
        g0p = [G0[s]["perc"] for s in SEEDS if s in G0]
        res["vacuous"] = dict(budget_never_binds=(bool(binds) and max(binds) < BUDGET_BINDS_MIN),
                              gravity_structures=(bool(g0p) and fl and (np.mean(g0p) - fl["percolation_mean"]) > SIGMA_MULT * max(fl["percolation_std"], 1e-9)),
                              floor=fl)
        # T1 holding: kappa=0.5 above BOTH B0 and G0 seed by seed, margins > 2 sigma
        pa = [A[s]["perc"] for s in SEEDS if s in A]; pb = [B0[s]["perc"] for s in SEEDS if s in B0]; pg = [G0[s]["perc"] for s in SEEDS if s in G0]
        sw_b = [A[s]["perc"] > B0[s]["perc"] for s in SEEDS if s in A and s in B0]
        sw_g = [A[s]["perc"] > G0[s]["perc"] for s in SEEDS if s in A and s in G0]
        mb, mg = (np.mean(pa) - np.mean(pb)) if pa and pb else float("nan"), (np.mean(pa) - np.mean(pg)) if pa and pg else float("nan")
        sb, sg = pooled_std(pa, pb), pooled_std(pa, pg)
        res["T1"] = dict(perc_k05=pa, perc_B0=pb, perc_G0=pg, seedwise_vs_B0=sw_b, seedwise_vs_G0=sw_g, margin_B0=mb, std_B0=sb, margin_G0=mg, std_G0=sg,
                         ok=(len(sw_b) == 3 and all(sw_b) and len(sw_g) == 3 and all(sw_g) and mb > SIGMA_MULT * sb and mg > SIGMA_MULT * sg))
        # T2 bound: KE/|U_grav| < 1 at EVERY mark of the window, at kappa = 0.5 and at kappa = 1, 3/3 seeds.
        # (The conserved total is (kappa - 1)|U0| by construction, so its sign is not a test.)
        A1 = arm(1.0)
        kmax = [A[s]["ke_over_u_max"] for s in SEEDS if s in A]; kmax1 = [A1[s]["ke_over_u_max"] for s in SEEDS if s in A1]
        res["T2"] = dict(ke_over_u_max_k05=kmax, ke_over_u_max_k1=kmax1,
                         ok=(len(kmax) == 3 and len(kmax1) == 3 and all(x < KE_OVER_U_BOUND for x in kmax + kmax1)))
        # T3 the engine's ordering above the virial threshold: KE/|U| at kappa 1 < 2 < inf, per seed
        ords = []
        for s in SEEDS:
            arms = [arm(k).get(s) for k in T3_ORDER]
            if any(x is None for x in arms): ords.append(None); continue
            u = [x["ke_over_u"] for x in arms]; ords.append(bool(u[0] < u[1] < u[2]))
        res["T3"] = dict(per_seed=ords, ok=(len(ords) == 3 and all(ords)))
        # T4 the ledger did the work: net creation <= P(0) exactly, pressure work within the truncation
        # allowance, and at kappa = 0.5 the budget binds on every would-grow particle (bound frac max = 1)
        led = [r for r in runs if r["size"] == size and r["kappa"] is not None and r["kappa"] > 0]
        exact = all(r["_summary"]["sec_transfer_cum"] <= r["budget0"] * (1 + 1e-6) for r in led)
        within = all((r["_summary"]["work_pressure_over_p0"] or 0.0) <= WORK_OVER_P0_ALLOWANCE for r in led)
        binds = [A[s]["budget_bound_frac_max"] for s in SEEDS if s in A]
        res["T4"] = dict(net_creation_le_p0=exact, work_within_allowance=within, bound_frac_max_k05=binds,
                         ok=(exact and within and len(binds) == 3 and all(b is not None and b >= BUDGET_BINDS_FULLY for b in binds)))
        out["per_size"][size] = res
    for t in ("T1", "T2", "T3", "T4"):
        out["tests"][t] = "PASS" if sizes and all(out["per_size"][s][t]["ok"] for s in sizes) else "FAIL"
    out["kill"]["mapping_dies"] = out["tests"]["T1"] == "FAIL"
    out["vacuous"] = {s: out["per_size"][s]["vacuous"] for s in sizes}
    out["score"] = f"{sum(1 for v in out['tests'].values() if v == 'PASS')}/4"
    return out


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if selftest() != 0:
        return 2
    files, runs, floors, commits = load(argv)
    out = score(runs, floors); out["grid_files"] = files; out["reality_engine_commits"] = commits
    dest = RESULTS / f"exp_29_pac_ledger_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    dest.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("gates", "tests", "kill", "score")}, indent=1))
    for size, r in out["per_size"].items():
        print(f"\n[{size}] vacuous={r['vacuous']['budget_never_binds']}/gravity_structures={r['vacuous']['gravity_structures']} floor={r['vacuous']['floor'].get('percolation_mean', float('nan')):.3f}±{r['vacuous']['floor'].get('percolation_std', float('nan')):.3f}")
        t = r["T1"]; print(f"  T1 k=0.5 {[round(x,3) for x in t['perc_k05']]} vs B0 {[round(x,3) for x in t['perc_B0']]} (margin {t['margin_B0']:.3f}, std {t['std_B0']:.3f}) vs G0 {[round(x,3) for x in t['perc_G0']]} (margin {t['margin_G0']:.3f}, std {t['std_G0']:.3f}) -> {t['ok']}")
        t = r["T2"]; print(f"  T2 max KE/|U| in window k=0.5 {[round(x,2) for x in t['ke_over_u_max_k05']]} k=1 {[round(x,2) for x in t['ke_over_u_max_k1']]} -> {t['ok']}")
        t = r["T3"]; print(f"  T3 KE/|U| ordered 1 < 2 < inf per seed {t['per_seed']} -> {t['ok']}")
        t = r["T4"]; print(f"  T4 exact {t['net_creation_le_p0']} within {t['work_within_allowance']} bound_frac_max@k0.5 {t['bound_frac_max_k05']} -> {t['ok']}")
    print(f"\nwrote results/{dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
