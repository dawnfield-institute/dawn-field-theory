#!/usr/bin/env python3
"""P-B1 GOLDEN RUN — first computation of D6-tree and E8-tree under Block B.
Registration: 74bcd0df. Grid, functional, and criteria exactly as sealed."""
import json, numpy as np
from pathlib import Path

def M_of(n, edges, s):
    A = np.zeros((n, n))
    for i, j in edges: A[i, j] = A[j, i] = 1
    D = np.diag(A.sum(1))
    return (D - A) + s * (2*np.eye(n) - D)

def mfpt_framefree(n, edges, s):
    M = M_of(n, edges, s); tot = 0.0
    k = 0
    for tgt in range(n):
        sub = np.delete(np.delete(M, tgt, 0), tgt, 1)
        if abs(np.linalg.det(sub)) < 1e-10: continue   # singular at this s: skip target
        tot += float(np.linalg.solve(sub, np.ones(n-1)).mean()); k += 1
    return tot / k if k else np.nan

S = np.round(np.arange(0.50, 1.501, 0.01), 3)
def s_star(n, e):
    v = [mfpt_framefree(n, e, s) for s in S]
    return float(S[int(np.nanargmin(v))])

GOLDEN = {"D6-tree": (6, [(0,1),(1,2),(2,3),(3,4),(3,5)]),
          "E8-tree": (8, [(i,i+1) for i in range(6)]+[(2,7)])}
PANEL = {  # symmetric controls, non-golden, per registration
  "star7":  (7, [(0,k) for k in range(1,7)]),
  "star8":  (8, [(0,k) for k in range(1,8)]),
  "spider3x2": (7, [(0,1),(1,2),(0,3),(3,4),(0,5),(5,6)]),
  "caterpillar8": (8, [(0,1),(1,2),(2,3),(1,4),(2,5),(0,6),(3,7)]),
  "broom8": (8, [(0,1),(1,2),(2,3),(3,4),(4,5),(4,6),(4,7)]),
  "double-star8": (8, [(0,1),(0,2),(0,3),(1,4),(1,5),(1,6),(1,7)]),
}
out = {"registration": "74bcd0df", "golden": {}, "panel": {}}
print("GOLDEN DIAGRAMS (first look):")
for nm, (n, e) in GOLDEN.items():
    st = s_star(n, e); out["golden"][nm] = st
    print(f"  {nm:<10} s* = {st:.2f}   |s*-1| = {abs(st-1):.2f}   pass |s*-1|<=0.01: {abs(st-1)<=0.0101}")
print("SYMMETRIC PANEL:")
for nm, (n, e) in PANEL.items():
    st = s_star(n, e); out["panel"][nm] = st
    print(f"  {nm:<13} s* = {st:.2f}   |s*-1| = {abs(st-1):.2f}")
gd = [abs(v-1) for v in out["golden"].values()]
pd = [abs(v-1) for v in out["panel"].values()]
t1 = all(d <= 0.0101 for d in gd)
t2 = max(gd) < min(pd)
confound = any(d <= 0.0101 for d in pd)
verdict = "PASS" if (t1 and t2 and not confound) else ("INCONCLUSIVE-BY-CONFOUND" if (t1 and confound) else "FAIL")
out["verdict"] = verdict
print(f"\nP-B1: both golden at self-dual: {t1} | beat panel best: {t2} | panel confound: {confound}")
print(f"VERDICT: {verdict}")
Path(__file__).parent.parent.joinpath("results","exp_04c_golden_20260831.json").write_text(json.dumps(out, indent=1))
