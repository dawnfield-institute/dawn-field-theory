#!/usr/bin/env python3
"""What does the instrument resolve? Run this over whatever grids exist BEFORE sealing a registration,
and put the numbers in §0 beside the thresholds.

    python resolution_probe.py results/exp_32_edge_grid_*.json

This is a TOOL, not a scorer: it has no thresholds, no verdicts and no kills, and it never writes.
It exists because three Milestone R registrations in a row carried a bar set without first measuring
what the instrument can resolve — exp_32's ledger-identity gate at 0.02 |U0| against a truncation
floor of ~0.027 (29 of 36 runs gated, the round UNSCORED), a kappa_c spread bar of 0.05 against a
measured spread of 0.07, and a positive-work-fraction floor of 0.45 with 0.001 of headroom.

For each statistic it prints min / median / max / sd over the runs given. A bar belongs ABOVE the
measured max by a stated factor, or it belongs nowhere. It reproduces the numbers cited in
journals/2026-09-16_exp32b_registration.md §0.4 and §0.7.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

GATES = (
    ("at_cap_max", "at_cap_max"),
    ("closure_pac_max", "closure_pac_max"),
    ("transfer_residual_max", "transfer_residual_max"),
)
STATS = (
    ("work_pressure_pos_frac", "positive-work fraction"),
    ("ke_over_u", "KE/|U|"),
    ("conn_q05", "conn_q05"),
    ("conn_q10", "conn_q10"),
)


def summarise(xs):
    xs = sorted(xs)
    return dict(min=xs[0], p50=st.median(xs), max=xs[-1],
                sd=(st.stdev(xs) if len(xs) > 1 else 0.0), n=len(xs))


def line(name, xs, fmt="{:.4f}"):
    s = summarise(xs)
    print(f"  {name:32s} min {fmt.format(s['min'])}  p50 {fmt.format(s['p50'])}  "
          f"max {fmt.format(s['max'])}  sd {fmt.format(s['sd'])}  (n={s['n']})")
    return s


def main(argv):
    paths = [Path(a) for a in argv if not a.startswith("--")]
    if not paths:
        raise SystemExit(__doc__.strip().split("\n\n")[1].strip())
    runs, kappas = [], {}
    for p in paths:
        g = json.loads(p.read_text(encoding="utf-8"))
        for r in g["runs"]:
            runs.append(r)
            kappas.setdefault(p.name, set()).add(r["kappa"])
    print(f"{len(runs)} runs over {len(paths)} grid(s)\n")

    print("== the ledger identity |dE_SEC - (T - W_p)| / |U0| — the gate that unscored exp_32 ==")
    ident = [abs(r["_summary"]["e_sec_end"]
                 - (r["_summary"]["sec_transfer_cum"] - r["_summary"]["work_pressure_cum"]))
             / r["_summary"]["u0"] for r in runs]
    s = line("identity residual", ident)
    for cand in (0.02, 0.06, 0.08, 0.10):
        fires = sum(1 for x in ident if x > cand)
        print(f"      a gate at {cand:<5}: fires on {fires:2d}/{len(ident)} runs, "
              f"{cand / s['max']:.2f}x the measured max")

    print("\n== the other instrument gates ==")
    for key, name in GATES:
        vals = [r["_summary"][key] for r in runs if r["_summary"].get(key) is not None]
        if vals:
            line(name, vals, "{:.2e}" if "residual" in key else "{:.4f}")

    print("\n== statistics a test might put a band on ==")
    for key, name in STATS:
        vals = [r["_summary"][key] for r in runs if r["_summary"].get(key) is not None]
        if vals:
            line(name, vals)

    print("\n== W_p/|U0| and its sign change, per arm and seed ==")
    # An ARM is (g, size). Runs from different arms must NEVER be merged: the gravity arms carry the
    # same size "full" at the same kappas as the fine sweep, and pooling them silently overwrites
    # the sweep's endpoints and shifts kappa_c. (That bug was in this file's first draft.)
    arms = {}
    for r in runs:
        arms.setdefault((round(r["config"]["g"], 3), r["size"]), {}) \
            .setdefault(r["seed"], {})[r["kappa"]] = r["_summary"]
    for (g, size), by_seed in sorted(arms.items()):
        xs_all = sorted({k for ks in by_seed.values() for k in ks})
        if len(xs_all) < 3:
            print(f"  arm g={g} size={size}: {len(xs_all)} kappa point(s) — too few to locate a crossing")
            continue
        print(f"  arm g={g} size={size}, kappas {xs_all}:")
        kcs = []
        for seed, ks in sorted(by_seed.items()):
            xs = sorted(ks)
            w = [ks[k]["work_pressure_cum"] / ks[k]["u0"] for k in xs]
            kc = None
            for (k0, k1, w0, w1) in zip(xs, xs[1:], w, w[1:]):
                if w0 < 0 <= w1:
                    kc = k0 + (k1 - k0) * (-w0) / (w1 - w0)
                    break
            changes = sum(1 for a, b in zip(w, w[1:]) if (a > 0) != (b > 0))
            print(f"    seed {seed}: {changes} sign change(s), kappa_c = "
                  f"{f'{kc:.4f}' if kc is not None else 'none'}")
            if kc is not None:
                kcs.append(kc)
        if len(kcs) > 1:
            sp = max(kcs) - min(kcs)
            print(f"    kappa_c over {len(kcs)} seeds: mean {st.fmean(kcs):.4f}  "
                  f"spread {sp:.4f}  sd {st.stdev(kcs):.4f}")
            print(f"        any spread bar below {sp:.4f} would fire on this data")

    floors = {}
    for p in paths:
        floors.update(json.loads(p.read_text(encoding="utf-8")).get("floors", {}).get("full", {}))
    if floors:
        print("\n== the occupancy-matched null draws recorded in the grids ==")
        for k in sorted(floors):
            if k.endswith("_mean"):
                base = k[:-5]
                sd = floors.get(base + "_std")
                extra = f" +/- {sd:.5f}  ({floors[k] / sd:.1f} sd to clear it)" if sd else ""
                print(f"  {base:24s} {floors[k]:.5f}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
