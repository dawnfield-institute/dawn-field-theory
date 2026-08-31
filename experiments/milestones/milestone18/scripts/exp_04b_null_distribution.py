#!/usr/bin/env python3
"""Calibration part 2 — FRAME-FREE functional + the null distribution. CONTROLS ONLY.
Fix from part 1: MFPT to vertex 0 is frame-dependent (an index is not a place).
Frame-free version: MFPT averaged over ALL absorbing targets. Null: |s* - 1| for
random 8-vertex trees, fine grid. Golden diagrams remain untouched."""
import json, numpy as np
from pathlib import Path
rng = np.random.default_rng(20260831)

def M_of(n, edges, s):
    A = np.zeros((n, n))
    for i, j in edges: A[i, j] = A[j, i] = 1
    D = np.diag(A.sum(1))
    return (D - A) + s * (2*np.eye(n) - D)

def mfpt_framefree(n, edges, s):
    M = M_of(n, edges, s); tot = 0.0
    for tgt in range(n):
        sub = np.delete(np.delete(M, tgt, 0), tgt, 1)
        try: tot += float(np.linalg.solve(sub, np.ones(n-1)).mean())
        except np.linalg.LinAlgError: return np.nan
    return tot / n

def s_star(n, edges, S):
    v = [mfpt_framefree(n, edges, s) for s in S]
    return float(S[int(np.nanargmin(v))])

def prufer_tree(n, rng):
    seq = rng.integers(0, n, size=n-2)
    deg = np.ones(n, int)
    for x in seq: deg[x] += 1
    edges = []; ptr = list(seq)
    leaves = sorted(i for i in range(n) if deg[i] == 1)
    import heapq; heapq.heapify(leaves)
    for x in ptr:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, int(x))); deg[x] -= 1
        if deg[x] == 1: heapq.heappush(leaves, int(x))
    u = heapq.heappop(leaves); v = heapq.heappop(leaves)
    edges.append((u, v))
    return edges

S = np.round(np.arange(0.50, 1.501, 0.01), 3)
NULL_N = 40
null8 = []
seen = set()
while len(null8) < NULL_N:
    e = prufer_tree(8, rng)
    k = frozenset(map(frozenset, e))
    if k in seen: continue
    seen.add(k)
    null8.append(s_star(8, e, S))
null8 = np.array(null8)
dev = np.abs(null8 - 1.0)
print(f"NULL (40 distinct random 8-trees, frame-free MFPT, grid 0.5..1.5 step .01):")
print(f"  s* range [{null8.min():.2f}, {null8.max():.2f}]  median {np.median(null8):.3f}")
print(f"  |s*-1|: median {np.median(dev):.3f}   5th pct {np.percentile(dev,5):.3f}   min {dev.min():.3f}")
print(f"  fraction of nulls with |s*-1| <= 0.01 (i.e., exactly at self-dual): {np.mean(dev<=0.0101):.3f}")
named = {"D5": (5,[(0,1),(1,2),(2,3),(2,4)]), "E6": (6,[(0,1),(1,2),(2,3),(3,4),(2,5)]),
         "E7": (7,[(i,i+1) for i in range(5)]+[(2,6)]), "path6": (6,[(i,i+1) for i in range(5)]),
         "path7": (7,[(i,i+1) for i in range(6)]), "star6": (6,[(0,k) for k in range(1,6)])}
print("\nnamed non-golden controls (frame-free):")
res = {"null8_sstar": null8.tolist()}
for nm, (n, e) in named.items():
    st = s_star(n, e, S); res[nm] = st
    print(f"  {nm:<6} s* = {st:.2f}   |s*-1| = {abs(st-1):.2f}")
Path(__file__).parent.parent.joinpath("results","exp_04b_null_20260831.json").write_text(json.dumps(res, indent=1))
print("\nSEALABLE PREDICTION TEMPLATE (golden diagrams still untouched):")
print("  'E8-tree and D6-tree frame-free MFPT optima satisfy |s*-1| <= 0.01, below the")
print("   null 5th percentile; size-matched random trees scatter per the null above.'")
