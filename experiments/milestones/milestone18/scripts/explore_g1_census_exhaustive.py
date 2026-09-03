#!/usr/bin/env python3
"""Panel G, part 1 (EXPLORING): exhaustive census of golden trees, n <= 13, over several
quadratic fields. Exact. Three grades: strict complete pairing / ledger-with-rational-core /
partial content. Regime by adjacency spectral radius. Norm signature of conjugate pairs."""
import json, time, sys, numpy as np, sympy as sp, networkx as nx
from pathlib import Path
t = sp.Symbol('t')
FIELDS = {"sqrt5": 5, "sqrt2": 2, "sqrt3": 3, "sqrt13": 13}

def cartan_of(G):
    n = G.number_of_nodes(); A = sp.zeros(n, n)
    for i, j in G.edges(): A[i, j] = A[j, i] = 1
    return 2*sp.eye(n) - A, A

def grade(p, d):
    """Return (grade, golden_factor_list, rational_factor_list, norm_signature) over Q(sqrt d)."""
    sd = sp.sqrt(d)
    fQ = sp.factor(p)
    facsQ = [g for g in sp.Mul.make_args(fQ) if g.has(t)]
    # prefilter: any odd-degree Q-irreducible factor cannot split into a conjugate pair
    for g in facsQ:
        b, e = g.as_base_exp()
        if sp.degree(b, t) % 2 == 1 and sp.degree(b, t) > 1: return "none", [], [], []
    f = sp.factor(p, extension=sd)
    facs = [g for g in sp.Mul.make_args(f) if g.has(t)]
    gold = [g for g in facs if g.has(sd)]
    rat  = [g for g in facs if not g.has(sd)]
    if not gold: return "none", [], [str(r) for r in rat], []
    # norm signature: constant term of each golden factor times its conjugate
    sig = []
    seen = set()
    for g in gold:
        b, e = g.as_base_exp()
        c0 = sp.Poly(b, t).all_coeffs()[-1]
        nrm = sp.simplify(sp.expand(c0*c0.subs(sd, -sd)))
        key = str(sp.expand(b))
        conj = str(sp.expand(b.subs(sd, -sd)))
        if conj not in seen:
            seen.add(key); sig.append(str(nrm))
    if not rat: return "strict", [str(g) for g in gold], [], sig
    even = all((g.as_base_exp()[1] % 2 == 0) for g in rat)
    return ("core" if even else "partial"), [str(g) for g in gold], [str(r) for r in rat], sig

def analyze(G):
    C, A = cartan_of(G)
    p = sp.expand(C.charpoly(t).as_expr())
    rho = float(max(abs(np.linalg.eigvalsh(np.array(A.tolist(), dtype=float)))))
    regime = "finite" if rho < 2 - 1e-9 else ("affine" if abs(rho - 2) < 1e-9 else "hyperbolic")
    out = {"n": G.number_of_nodes(), "edges": sorted(map(list, G.edges())), "rho": round(rho, 6),
           "regime": regime, "det": int(sp.Poly(p, t).all_coeffs()[-1] * (-1)**G.number_of_nodes()),
           "fields": {}}
    for name, d in FIELDS.items():
        gr, gold, rat, sig = grade(p, d)
        if gr != "none": out["fields"][name] = {"grade": gr, "golden": gold, "rational": rat, "norm_signature": sig}
    return out

# ---- self-tests ----
def tree(n, e): G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(e); return G
E8 = tree(8, [(i,i+1) for i in range(6)]+[(2,7)]); D6 = tree(6, [(0,1),(1,2),(2,3),(3,4),(3,5)])
A5 = tree(5, [(i,i+1) for i in range(4)]); CAT8 = tree(8, [(0,6),(2,1),(5,3),(6,1),(1,3),(3,4),(4,7)])
assert analyze(E8)["fields"]["sqrt5"]["grade"] == "strict", "E8 strict"
assert analyze(D6)["fields"]["sqrt5"]["grade"] == "core",   "D6 ledger-with-rational-core"
assert "sqrt5" not in analyze(A5)["fields"],                  "A5 none over sqrt5"
assert analyze(A5)["fields"].get("sqrt3", {}).get("grade") == "partial", "A5 is sqrt3-partial (h=6; rational factors t-1,t-2,t-3 odd multiplicity)"
assert analyze(CAT8)["fields"]["sqrt5"]["grade"] == "strict", "cat8 strict"
print("self-tests passed", flush=True)

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 13
results = []; t0 = time.time()
for n in range(4, NMAX+1):
    cnt = 0; hits = 0
    for G in nx.nonisomorphic_trees(n):
        cnt += 1; r = analyze(G)
        if r["fields"]:
            hits += 1; results.append(r)
    print(f"n={n}: {cnt} trees, {hits} with any-field content  [{time.time()-t0:.0f}s]", flush=True)
    Path(__file__).parent.parent.joinpath("results", "explore_g1_census_20260901.json").write_text(json.dumps(results, indent=1))
print("G1 DONE", flush=True)
