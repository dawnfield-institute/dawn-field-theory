#!/usr/bin/env python3
"""explore_d1c (Block D pre-seal, draft §5 / lesson 8): the CERTIFICATE on the NULL populations — every
partial/core tree on 15 vertices (exhaustive) and the degree-capped random subcubic trees at n = 31 and 63
(same seed as explore_d1) — carrier rate under some-half semantics: does any Galois half give
(tr(R·D), leak_oo) = (2/sqrt5, 2/5)? These trees are the null, not Block D objects. Results append-only."""
import sys, json, time
from pathlib import Path
import numpy as np, sympy as sp, networkx as nx
HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "core"))
from certificate import grade, evaluate, t
from ledger import cart
RES = ROOT / "results"

if __name__ == "__main__":
    t0 = time.time(); out = dict(n15=dict(evaluated=0, carriers=0, by_grade={}), subcubic={})
    for T in nx.nonisomorphic_trees(15):
        e = list(T.edges()); p = sp.expand(cart(15, e).charpoly(t).as_expr()); g = grade(p)[0]
        if g == "none": continue
        r = evaluate(15, e); out["n15"]["evaluated"] += 1; out["n15"]["by_grade"][g] = out["n15"]["by_grade"].get(g, 0) + 1
        if r.get("carrier_some_half"): out["n15"]["carriers"] += 1
        if out["n15"]["evaluated"] % 50 == 0: print(f"  n=15: {out['n15']['evaluated']} evaluated, {out['n15']['carriers']} carriers [{time.time()-t0:.0f}s]", flush=True)
    print(f"null n=15: {out['n15']} [{time.time()-t0:.0f}s]", flush=True)
    rng = np.random.RandomState(20260903)
    for nn in (31, 63):
        cnt = dict(evaluated=0, carriers=0, by_grade={})
        for k in range(40 if nn == 31 else 20):
            deg = [0] * nn; e_ = []
            for v in range(1, nn):
                cands = [u for u in range(v) if deg[u] < 3]; u = int(rng.choice(cands)); e_.append((u, v)); deg[u] += 1; deg[v] += 1
            g = grade(sp.expand(cart(nn, e_).charpoly(t).as_expr()))[0]
            if g == "none": continue
            r = evaluate(nn, e_); cnt["evaluated"] += 1; cnt["by_grade"][g] = cnt["by_grade"].get(g, 0) + 1
            if r.get("carrier_some_half"): cnt["carriers"] += 1
            print(f"  subcubic n={nn} #{k}: grade {g}, carrier {r.get('carrier_some_half')} [{time.time()-t0:.0f}s]", flush=True)
        out["subcubic"][nn] = cnt; print(f"null subcubic n={nn}: {cnt}", flush=True)
    json.dump(out, open(RES / f"explore_d1c_null_carriers_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1, default=str)
    print("DONE", flush=True)
