#!/usr/bin/env python3
"""exp_01: The three foldings exist and are exact. Registered 2026-08-31 (cf886c00)."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from folding import (e8_roots, d6_roots, E8_SIMPLE, D6_SIMPLE, coxeter_element, PHI)
import sympy as sp

res = {"experiment_id": "exp_01_foldings_exist", "registration": "cf886c00", "tests": {}}

# A4 roots: 20 vectors e_i - e_j (i != j) in R^5
a4 = np.array([[1 if k == i else (-1 if k == j else 0) for k in range(5)]
               for i in range(5) for j in range(5) if i != j], float)

# T1: root counts = rank * Coxeter number
counts = {"A4": len(a4), "D6": len(d6_roots()), "E8": len(e8_roots())}
res["tests"]["T1"] = {"counts": counts, "pass": counts == {"A4": 20, "D6": 60, "E8": 240}}

# T2: H-side root counts (explicit constructions) and 2:1
h2 = [(np.cos(2*np.pi*k/10), np.sin(2*np.pi*k/10)) for k in range(10)]           # decagon
# H3 roots = icosidodecahedron vertices (30): cyclic perms of (0,0,+-phi)
# and (1/2)*cyclic perms of (+-1, +-phi, +-phi^2); all norm phi (|(1,phi,phi^2)| = 2phi).
h3 = set()
for c in ([0,1,2],[1,2,0],[2,0,1]):
    for s in (1,-1):
        v=[0.0,0.0,0.0]; v[c[2]]=s*PHI; h3.add(tuple(np.round(v,12)))
    for s1 in (1,-1):
        for s2 in (1,-1):
            for s3 in (1,-1):
                v=[0.0,0.0,0.0]
                v[c[0]]=s1*0.5; v[c[1]]=s2*PHI/2; v[c[2]]=s3*PHI**2/2
                h3.add(tuple(np.round(v,12)))
h3 = list(h3)
# H4: 120 unit icosians (600-cell vertices)
h4 = set()
for i in range(4):
    for s in (1,-1):
        v=[0,0,0,0]; v[i]=s; h4.add(tuple(v))
for signs in range(16):
    h4.add(tuple(((1 if (signs>>k)&1 else -1)*0.5) for k in range(4)))
from itertools import permutations
def parity(p):
    p=list(p); n=0
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            if p[i]>p[j]: n+=1
    return n%2
base=[0.5,PHI/2,1/(2*PHI),0.0]
for p in permutations(range(4)):
    if parity(p)==0:
        for s1 in (1,-1):
            for s2 in (1,-1):
                for s3 in (1,-1):
                    v=[0,0,0,0]
                    vals=[s1*base[0], s2*base[1], s3*base[2], 0.0]
                    for k in range(4): v[p[k]]=vals[k]
                    h4.add(tuple(np.round(v,12)))
res["tests"]["T2"] = {"H2": len(h2), "H3": len(h3), "H4": len(h4),
                      "ratios": [counts["A4"]/len(h2), counts["D6"]/len(h3), counts["E8"]/len(h4)],
                      "pass": (len(h2), len(h3), len(h4)) == (10, 30, 120)}

# T3: Coxeter numbers — order of the ADE Coxeter element AND roots/rank on both sides
def cox_order(simples, hmax=40):
    W = coxeter_element(simples); M = np.eye(simples.shape[1])
    for k in range(1, hmax+1):
        M = M @ W
        if np.allclose(M, np.eye(simples.shape[1]), atol=1e-9): return k
    return -1
a4_simple = np.array([[1,-1,0,0,0],[0,1,-1,0,0],[0,0,1,-1,0],[0,0,0,1,-1]], float)
hA4, hD6, hE8 = cox_order(a4_simple), cox_order(D6_SIMPLE), cox_order(E8_SIMPLE)
pairs = {"A4->H2": (hA4, len(h2)//2), "D6->H3": (hD6, len(h3)//3), "E8->H4": (hE8, len(h4)//4)}
res["tests"]["T3"] = {"pairs": {k: list(v) for k, v in pairs.items()},
                      "pass": all(a == b for a, b in pairs.values()) and (hA4, hD6, hE8) == (5, 10, 30)}

# T4: A4 unique golden A_n, spectrum exact (sympy)
t = sp.Symbol('t'); phi_s = (1+sp.sqrt(5))/2
golden = []
for n in range(2, 13):
    deg = sp.degree(sp.minimal_polynomial(2*sp.cos(sp.pi/(n+1)), t), t)
    if deg == 2 and sp.simplify(sp.minimal_polynomial(2*sp.cos(sp.pi/(n+1)), t) - (t**2 - t - 1)) == 0:
        golden.append(n)
C4 = sp.Matrix(4, 4, lambda i, j: 2 if i == j else (-1 if abs(i-j) == 1 else 0))
spec = sorted(C4.eigenvals(), key=lambda z: float(sp.re(sp.N(z))))
target = [2-phi_s, 3-phi_s, 1+phi_s, 2+phi_s]
exact = all(sp.simplify(a-b) == 0 for a, b in zip(spec, sorted(target, key=float)))
res["tests"]["T4"] = {"golden_An": golden, "spectrum_exact": bool(exact), "pass": golden == [4] and exact}

res["score"] = sum(res["tests"][k]["pass"] for k in res["tests"])
print(json.dumps({k: v["pass"] for k, v in res["tests"].items()}, indent=1), "\nSCORE", res["score"], "/4")
out = Path(__file__).parent.parent / "results" / "exp_01_foldings_20260831.json"
out.write_text(json.dumps(res, indent=1, default=str))
