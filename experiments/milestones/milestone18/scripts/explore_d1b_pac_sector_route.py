#!/usr/bin/env python3
"""explore_d1b (Block D pre-seal, draft §2.7 + §5): the exact SECTOR ROUTE for complete PAC binary trees.
The Cartan of the depth-d complete binary tree (n = 2^(d+1) - 1) decomposes under its automorphism
group into sqrt2-weighted radial paths: charpoly(d) = P_{d+1} * prod_{l=0}^{d-1} P_{d-l}^(2^l), where P_k
is the charpoly of the k-node path Cartan with off-diagonal -sqrt2 (rational coefficients). GATE: equals
the direct charpoly at d = 3, 4, 5 exactly. Then grades and sector classes at d = 6, 7, 8 from the
per-sector factorizations (never a certificate on the tree itself)."""
import sys, json, time
from pathlib import Path
import sympy as sp
HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "core"))
from certificate import grade, golden_pairs, rational_cores, class_sectors, t, s5
from ledger import cart
RES = ROOT / "results"
r2 = sp.sqrt(2)

def path_charpoly(k):
    C = 2 * sp.eye(k)
    for i in range(k - 1): C[i, i + 1] = C[i + 1, i] = -r2
    return sp.expand(C.charpoly(t).as_expr())

def sector_charpoly(d):
    P = {k: path_charpoly(k) for k in range(1, d + 2)}
    mult = {d + 1: 1}
    for l in range(d): mult[d - l] = mult.get(d - l, 0) + 2 ** l
    p = sp.Integer(1)
    for k, m in mult.items(): p *= P[k] ** m
    return sp.expand(p), P, mult

def pac_edges(d):
    nn = 2 ** (d + 1) - 1
    return nn, [(i, 2 * i + 1) for i in range(nn) if 2 * i + 1 < nn] + [(i, 2 * i + 2) for i in range(nn) if 2 * i + 2 < nn]

if __name__ == "__main__":
    t0 = time.time(); out = dict(gate={}, depths={})
    for d in (3, 4, 5):
        nn, ed = pac_edges(d); direct = sp.expand(cart(nn, ed).charpoly(t).as_expr()); sect, P, mult = sector_charpoly(d)
        ok = sp.expand(direct - sect) == 0; out["gate"][d] = ok
        print(f"sector-route gate d={d} (n={nn}): equals direct charpoly = {ok}; multiplicities {mult} [{time.time()-t0:.0f}s]", flush=True)
    assert all(out["gate"].values()), "sector route does not reproduce the direct charpoly — stop"
    for d in (3, 4, 5, 6, 7, 8):
        nn = 2 ** (d + 1) - 1; sect, P, mult = sector_charpoly(d)
        # per-sector factorization over Q(sqrt5): rational factors with total multiplicity; golden pairs with class
        rat_mult = {}; pairs = {}
        for k, m in mult.items():
            for r, e in rational_cores(P[k]): rat_mult[str(r)] = rat_mult.get(str(r), 0) + e * m
            for g, sg, e in golden_pairs(P[k]):
                key = str(sp.expand(g)); pairs.setdefault(key, dict(mult=0, sectors=[], cls=None))
                pairs[key]["mult"] += e * m; pairs[key]["sectors"].append((k, m))
                cl = class_sectors(P[k]); pairs[key]["cls"] = [(c, str(s)) for g2, c, s in cl if str(sp.expand(g2)) == key or str(sp.expand(g2.subs(s5, -s5))) == key] or pairs[key]["cls"]
        if not pairs: gr = "none"
        elif not rat_mult: gr = "strict"
        else: gr = "core" if all(v % 2 == 0 for v in rat_mult.values()) else "partial"
        out["depths"][d] = dict(n=nn, grade=gr, rational_factors=rat_mult, golden_pairs=pairs, n_pairs=len(pairs), n_halves=2 ** max(0, len(pairs) - 1))
        print(f"PAC depth {d} (n={nn}): grade {gr}; golden pairs {len(pairs)}: { {k: (v['mult'], v['cls'], v['sectors']) for k, v in pairs.items()} }; rational {rat_mult} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(RES / f"explore_d1b_pac_sector_route_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1, default=str)
    print("DONE", flush=True)
