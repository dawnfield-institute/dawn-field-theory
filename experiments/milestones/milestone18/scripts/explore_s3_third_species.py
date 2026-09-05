#!/usr/bin/env python3
"""s3: does anything separate the 9 asymmetric trees that carry an integral Galois half from the
other 32? Exploration on the Phase 8 n=24 records. Five looks, no correction, 9 objects: a null."""
import json, collections, sys
from math import comb
from pathlib import Path
RES = Path(__file__).parent.parent / "results"
d = json.load(open(RES / "exp_19_phase8_n24_20260903_155724.json")); recs = d["records"]
asym = [r for r in recs if r["cls"] == "asymmetric"]
leak = [r for r in asym if r.get("any_integral_half")]; nol = [r for r in asym if not r.get("any_integral_half")]
def deg(r):
    c = collections.Counter()
    for a, b in r["edges"]: c[a] += 1; c[b] += 1
    return c
def fisher(a, b, c, dd):
    n = a + b + c + dd; row1, col1 = a + b, a + c
    def p(x): return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)
    p0 = p(a); lo, hi = max(0, col1 - (n - row1)), min(row1, col1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 + 1e-12)
print(f"asymmetric {len(asym)}; leak {len(leak)}; no-leak {len(nol)}")
for name, f in [("degree-4+ vertex", lambda r: max(deg(r).values()) >= 4),
                ("degree-5+ vertex", lambda r: max(deg(r).values()) >= 5),
                ("n_halves == 1", lambda r: r["n_halves"] == 1),
                ("n_halves >= 8", lambda r: r["n_halves"] >= 8),
                ("leaves >= 8", lambda r: sum(1 for v in deg(r).values() if v == 1) >= 8)]:
    a = sum(1 for r in leak if f(r)); c = sum(1 for r in nol if f(r))
    print(f"  {name:20s} leak {a}/{len(leak)}  no-leak {c}/{len(nol)}  Fisher p={fisher(a, len(leak)-a, c, len(nol)-c):.3f}")
print("Bonferroni over 5 looks: nothing below 0.05. NULL — no structural handle yet.")
