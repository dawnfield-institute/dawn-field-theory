#!/usr/bin/env python3
"""exp_07: the knife-edge under Bipartite Duality. Registration 74bcd0df (sealed).

ENUMERATION FIXED BEFORE THE SWEEP (from field theory, not from running):
  paths, s=0 and s=2 (dual pair): golden content iff 5 | n  -> n in {5, 10}
  paths, s=1 (self-dual):         golden content iff 5 | n+1 -> n in {4, 9}
  outer columns s=-1 and s=3:     SCORED prediction = the two columns are identical
     for every n (duality); their content is recorded as exploratory (no closed-form
     prediction was derived; interpretation documented in the outcomes journal).
  branched folding trees (D6, E8): golden ONLY at s=1 across the whole grid.

T1: operator identity Pi M(s) Pi = 4I - M(2-s), SYMBOLIC in s, every tree.
T2: the golden s-set is closed under s -> 2-s; s=1 the unique self-dual point.
T3: the enumeration above.
T4: E8 at s=3/4 ([4,4] rational): explained or flagged."""
import json, sympy as sp
from pathlib import Path
t, sv = sp.symbols('t s')
res = {"experiment_id": "exp_07_knife_edge", "registration": "74bcd0df", "tests": {}}

def tree_M(n, edges, s):
    A = sp.zeros(n, n)
    for i, j in edges: A[i, j] = A[j, i] = 1
    D = sp.diag(*[sum(A[i, k] for k in range(n)) for i in range(n)])
    return (D - A) + s*(2*sp.eye(n) - D), A

def bipartition(n, edges):
    color = {0: 1}; stack = [0]
    adj = {i: [] for i in range(n)}
    for i, j in edges: adj[i].append(j); adj[j].append(i)
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in color: color[w] = -color[u]; stack.append(w)
    return sp.diag(*[color[i] for i in range(n)])

def golden(n, edges, s):
    M, _ = tree_M(n, edges, sp.Rational(s))
    f = sp.factor(sp.expand(M.charpoly(t).as_expr()), extension=sp.sqrt(5))
    return any(g.has(sp.sqrt(5)) for g in sp.Mul.make_args(f))

PATHS = {n: [(i, i+1) for i in range(n-1)] for n in range(3, 13)}
BRANCHED = {"D6": (6, [(0,1),(1,2),(2,3),(3,4),(3,5)]),
            "E8": (8, [(i,i+1) for i in range(6)]+[(2,7)])}
GRID = [-1, 0, 1, 2, 3]

# T1 symbolic operator identity
ok1 = True
for n, e in list(PATHS.items()) + [(v[0], v[1]) for v in BRANCHED.values()]:
    M1, _ = tree_M(n, e, sv); M2, _ = tree_M(n, e, 2 - sv)
    Pi = bipartition(n, e)
    if sp.simplify(Pi*M1*Pi + M2 - 4*sp.eye(n)) != sp.zeros(n, n): ok1 = False; break
res["tests"]["T1"] = {"symbolic_in_s_all_trees": ok1, "pass": ok1}

# sweep
table = {}
for n, e in PATHS.items():
    table[f"path{n}"] = {s: golden(n, e, s) for s in GRID}
for nm, (n, e) in BRANCHED.items():
    table[nm] = {s: golden(n, e, s) for s in GRID}
res["sweep"] = {k: {str(s): v for s, v in row.items()} for k, row in table.items()}

# T2 closure under s -> 2-s on the grid (pairs: (-1,3), (0,2), (1,1))
ok2 = all(row[-1] == row[3] and row[0] == row[2] for row in table.values())
res["tests"]["T2"] = {"closed_under_duality": ok2, "pass": ok2}

# T3 the fixed enumeration
pred = {}
okA = all(table[f"path{n}"][0] == (n % 5 == 0) and table[f"path{n}"][2] == (n % 5 == 0)
          for n in PATHS)
okB = all(table[f"path{n}"][1] == ((n+1) % 5 == 0) for n in PATHS)
okC = all(table[f"path{n}"][-1] == table[f"path{n}"][3] for n in PATHS)   # scored outer claim
okD = all(all(v == (s == 1) for s, v in table[nm].items()) for nm in BRANCHED)
res["tests"]["T3"] = {"paths_s0_s2_iff_5_divides_n": okA, "paths_s1_iff_5_divides_n_plus_1": okB,
                      "outer_columns_mirror": okC, "branched_only_self_dual": okD,
                      "outer_content_exploratory": {f"path{n}": table[f"path{n}"][-1] for n in PATHS},
                      "pass": okA and okB and okC and okD}

# T4: E8 at s=3/4 — explain or flag
M34, _ = tree_M(8, BRANCHED["E8"][1], sp.Rational(3, 4))
fQ = sp.factor(sp.expand(M34.charpoly(t).as_expr()))
quartics = [g for g in sp.Mul.make_args(fQ) if sp.degree(g, t) == 4]
expl = None
if len(quartics) == 2:
    q1, q2 = [sp.Poly(q, t) for q in quartics]
    # test: are the two rational quartics exchanged by the DUALITY t -> 4 - t?
    swap = sp.expand(q1.as_expr().subs(t, 4 - t) * (1 if q1.degree() % 2 == 0 else -1))
    if sp.simplify(swap - q2.as_expr()) == 0 or sp.simplify(swap + q2.as_expr()) == 0:
        expl = "the two rational quartics are exchanged by the duality t -> 4-t: at s=3/4 the charpoly factors into a DUALITY-conjugate pair over Q (the additive analogue of the golden sigma-pair; dual point s=5/4 carries the mirrored factorization)"
res["tests"]["T4"] = {"factorization": str(fQ), "explained": expl is not None,
                      "explanation": expl or "FLAGGED: two rational quartics, mechanism not identified",
                      "pass": True}   # sealed: "explained or flagged" — either satisfies

res["score"] = sum(res["tests"][k]["pass"] for k in res["tests"])
print(json.dumps(res["tests"], indent=1, default=str)); print("SCORE", res["score"], "/4")
Path(__file__).parent.parent.joinpath("results", "exp_07_knife_edge_20260831.json").write_text(
    json.dumps(res, indent=1, default=str))
