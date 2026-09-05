#!/usr/bin/env python3
"""exp_28 — score the derived-sink grid against the SEALED registration.

    python exp_28_dynamical_severance.py [results/exp_28_dynamical_severance_grid_<ts>.json]
    python exp_28_dynamical_severance.py --selftest      # thresholds byte-equal to the journal

Loads the grid JSON produced by reality-engine POC-11 exp_04_aggregate.py (copied into results/),
re-verifies the instrument gates recorded in it, and scores T1-T4 with the thresholds below —
which are the registration's, verbatim (journals/2026-09-05_exp28_registration.md §4). Exits
nonzero if the grid is absent: no fallback, no transcribed numbers (the milestone1 trap).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
JOURNAL = HERE.parent / "journals" / "2026-09-05_exp28_registration.md"

# ---- the sealed thresholds (registration §4/§7), verbatim ---------------------------------------
TAUS = (5.0, 10.0, 20.0)
SEEDS = (1, 2, 3)
T1_WINDOW = (5.0, 15.0)              # e_int^S(t) < e_int^B0(t) at every mark in [5, 15]
T2_SIGMA_MULT = 2.0                  # mean difference > 2x pooled within-seed std
MIN_TAUS_PASSING = 2                 # >= 2 of 3 tau
INFORMATIVE_SEV_FRAC = (0.01, 0.9)   # cumulative severed fraction in [0.01, 0.9]
VACUOUS_N_RET = {"proxy": 512, "full": 500}   # n_ret at t = 10 below this -> vacuous
CLOSURE_MAX = 1e-5                   # the instrument's own falsification
SELFTEST_STRINGS = ["≥ 2 of 3 τ", "2× the pooled within-seed standard deviation", "[0.01, 0.9]",
                    "below **512**", "τ ∈\n{5, 10, 20}"]


def selftest() -> int:
    norm = lambda s: " ".join(s.split())          # line wraps in the journal are not thresholds
    text = norm(JOURNAL.read_text(encoding="utf-8"))
    missing = [s for s in SELFTEST_STRINGS if norm(s) not in text]
    print("thresholds byte-equal to the registration:", "OK" if not missing else f"MISSING {missing}")
    return 0 if not missing else 1


def load_grid(argv):
    if len(argv) > 1 and argv[1] != "--selftest":
        path = Path(argv[1])
    else:
        cands = sorted(RESULTS.glob("exp_28_dynamical_severance_grid_*.json"))
        if not cands:
            raise SystemExit("no results/exp_28_dynamical_severance_grid_*.json — nothing to score (no fallback)")
        path = cands[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))


def pooled_std(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)) if len(a) > 1 and len(b) > 1 else float("nan")


def score(grid: dict) -> dict:
    runs = grid["runs"]; comps = grid["comparisons"]
    def sel(size, arm, tau=None, seed=None, md=0.95):
        return [r for r in runs if r["size"] == size and r["arm"] == arm and (tau is None or r["tau"] == tau)
                and (seed is None or r["seed"] == seed) and abs(r["md"] - md) < 1e-9]
    out = {"gates": {}, "tests": {}, "kills": {}, "notes": []}

    # instrument gates re-verified from the recorded summaries
    out["gates"]["closure_max_le_1e-5"] = all(r["_summary"]["closure_max"] <= CLOSURE_MAX for r in runs)
    out["gates"]["all_finite"] = all(r["_summary"]["finite"] for r in runs)
    out["gates"]["at_cap_le_0.02"] = all(r["_summary"]["at_cap_max"] <= 0.02 for r in runs)

    per_size = {}
    for size in ("proxy", "full"):
        if not sel(size, "S"): continue
        res = {"T1": {}, "T2": {}, "vacuous": {}}
        for tau in TAUS:
            S = {r["seed"]: r for r in sel(size, "S", tau)}
            B = {r["seed"]: r for r in sel(size, "B0")}
            if not S: continue
            # vacuity
            sev = [S[s]["_summary"]["sev_frac"] for s in S]
            nret = [S[s]["_summary"]["n_alive_t10"] for s in S]
            vac = (not all(INFORMATIVE_SEV_FRAC[0] <= f <= INFORMATIVE_SEV_FRAC[1] for f in sev)
                   or any(n < VACUOUS_N_RET[size] for n in nret))
            res["vacuous"][tau] = dict(vacuous=vac, sev_frac=sev, n_ret_t10=nret)
            # T1 per seed: e_int^S(t) < e_int^B0(t) at every mark in [5,15]; sum loss_severance_energy > 0
            t1 = []
            for s in S:
                if s not in B: t1.append(None); continue
                eS, eB = S[s]["_summary"]["e_int_series"], B[s]["_summary"]["e_int_series"]
                ok = all(float(eS[str(t)] if str(t) in eS else eS[t]) < float(eB[str(t)] if str(t) in eB else eB[t])
                         for t in range(int(T1_WINDOW[0]), int(T1_WINDOW[1]) + 1)
                         if (str(t) in eS or t in eS) and (str(t) in eB or t in eB))
                t1.append(bool(ok))
            res["T1"][tau] = dict(per_seed=t1, ok=(len(t1) == len(SEEDS) and all(t1)) and not vac)
            # T2: S perc vs B0 matched perc per seed, 3/3 and mean diff > 2x pooled std
            pairs = [c for c in comps if c["size"] == size and c["other_arm"] == "B0" and c["tau"] == tau and abs(c["md"] - 0.95) < 1e-9]
            byseed = {c["seed"]: c for c in pairs}
            sp = [byseed[s]["S_perc"] for s in SEEDS if s in byseed]; bp = [byseed[s]["other_perc_matched"] for s in SEEDS if s in byseed]
            seedwise = [a > b for a, b in zip(sp, bp)]
            ps = pooled_std(sp, bp); margin = (float(np.mean(sp)) - float(np.mean(bp))) if sp else float("nan")
            res["T2"][tau] = dict(S_perc=sp, B0_perc_matched=bp, seedwise=seedwise, pooled_std=ps, margin=margin,
                                  ok=(len(seedwise) == len(SEEDS) and all(seedwise) and ps == ps and margin > T2_SIGMA_MULT * ps) and not vac)
        res["T1_pass"] = sum(1 for t in TAUS if res["T1"].get(t, {}).get("ok")) >= MIN_TAUS_PASSING
        res["T2_pass"] = sum(1 for t in TAUS if res["T2"].get(t, {}).get("ok")) >= MIN_TAUS_PASSING
        # tau*: largest T2 margin among informative taus; 10 if null everywhere
        margins = {t: res["T2"][t]["margin"] for t in TAUS if t in res["T2"] and not res["vacuous"][t]["vacuous"]}
        tau_star = max(margins, key=margins.get) if margins and any(res["T2"][t]["ok"] for t in margins) else 10.0
        res["tau_star"] = tau_star
        # T3 / T4 at tau*: conditional on T2 at tau*
        for arm, name in (("D", "T3"), ("R", "T4")):
            pairs = [c for c in comps if c["size"] == size and c["other_arm"] == arm and c["tau"] == tau_star and abs(c["md"] - 0.95) < 1e-9]
            byseed = {c["seed"]: c for c in pairs}
            sp = [byseed[s]["S_perc"] for s in SEEDS if s in byseed]; op = [byseed[s]["other_perc_matched"] for s in SEEDS if s in byseed]
            seedwise = [a > b for a, b in zip(sp, op)]; ps = pooled_std(sp, op)
            margin = (float(np.mean(sp)) - float(np.mean(op))) if sp else float("nan")
            t2_ok = res["T2"].get(tau_star, {}).get("ok", False)
            d_runs = sel(size, "D") if arm == "D" else []
            uninformative = (not t2_ok) or (arm == "D" and d_runs and all(abs(r["damping"] - 1.0) < 1e-12 for r in d_runs))
            res[name] = dict(tau_star=tau_star, S_perc=sp, other_perc=op, seedwise=seedwise, pooled_std=ps, margin=margin,
                             verdict=("UNINFORMATIVE" if uninformative else
                                      ("PASS" if (len(seedwise) == len(SEEDS) and all(seedwise) and ps == ps and margin > T2_SIGMA_MULT * ps) else "FAIL")))
        res["K1_fires"] = res["T2"].get(tau_star, {}).get("ok", False) and res["T3"]["verdict"] == "FAIL"
        res["K2_fires"] = res["T2"].get(tau_star, {}).get("ok", False) and res["T4"]["verdict"] == "FAIL"
        per_size[size] = res
    out["per_size"] = per_size
    # the scorecard: M = 4, scored on the FULL size when present, else the proxy (declared)
    size = "full" if "full" in per_size else ("proxy" if "proxy" in per_size else None)
    if size:
        r = per_size[size]
        out["scored_size"] = size
        out["tests"] = {"T1": "PASS" if r["T1_pass"] else "FAIL", "T2": "PASS" if r["T2_pass"] else "FAIL",
                        "T3": r["T3"]["verdict"], "T4": r["T4"]["verdict"]}
        out["kills"] = {"K1": r["K1_fires"], "K2": r["K2_fires"]}
        out["score"] = f"{sum(1 for v in out['tests'].values() if v == 'PASS')}/4"
    return out


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if selftest() != 0:
        return 2
    path, grid = load_grid(argv)
    out = score(grid)
    out["grid_file"] = path.name; out["reality_engine_commit"] = grid.get("commit")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = RESULTS / f"exp_28_dynamical_severance_{stamp}.json"
    dest.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("gates", "scored_size", "tests", "kills", "score") if k in out}, indent=1))
    for size, r in out["per_size"].items():
        print(f"\n[{size}] tau*={r['tau_star']}")
        for tau in TAUS:
            if tau in r["T2"]:
                v = r["vacuous"][tau]; t2 = r["T2"][tau]
                print(f"  tau={tau:>4}: sev_frac={[round(x,3) for x in v['sev_frac']]} n_ret@10={v['n_ret_t10']} vacuous={v['vacuous']} "
                      f"T1={r['T1'][tau]['per_seed']} T2 S={[round(x,3) for x in t2['S_perc']]} B0m={[round(x,3) for x in t2['B0_perc_matched']]} "
                      f"margin={t2['margin']:.3f} pooled_std={t2['pooled_std']:.3f} ok={t2['ok']}")
        for name in ("T3", "T4"):
            t = r[name]; print(f"  {name} @tau*: S={[round(x,3) for x in t['S_perc']]} other={[round(x,3) for x in t['other_perc']]} margin={t['margin']:.3f} std={t['pooled_std']:.3f} -> {t['verdict']}")
    print(f"\nwrote results/{dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
