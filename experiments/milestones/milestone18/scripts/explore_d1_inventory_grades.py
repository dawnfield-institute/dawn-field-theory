#!/usr/bin/env python3
"""explore_d1 (Block D pre-seal numbers; draft §5): GRADES ONLY of the sealed objects O1–O4 and the nulls,
the informative count |E| and the halves count per object. NEVER a certificate value on O1–O4 — the
instrument's `certificate`/`evaluate` are not called on any object here; only `grade`, `halves`,
`class_sectors`, `is_regular`. Results append-only, timestamped. Usage: python explore_d1_inventory_grades.py [max_pac_depth]."""
import sys, json, time, glob
from pathlib import Path
import numpy as np, sympy as sp, networkx as nx
HERE = Path(__file__).parent; ROOT = HERE.parent; EXP = ROOT.parent.parent
sys.path.insert(0, str(ROOT / "core")); sys.path.insert(0, str(EXP / "studies" / "prime_growth_dynamics_v2" / "core"))
sys.path.insert(0, str(EXP / "milestones" / "milestone15" / "core")); sys.path.insert(0, str(EXP / "milestones" / "milestone13" / "core"))
from certificate import grade, halves, class_sectors, is_regular, golden_pairs, t
from ledger import cart
from growth_harness import replay
RES = ROOT / "results"

def cartan_from_adj(A):
    n = A.shape[0]; e = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] != 0]; return n, e

def describe(n, e, label):
    C = cart(n, [tuple(x) for x in e]); p = sp.expand(C.charpoly(t).as_expr())
    G = nx.Graph([tuple(x) for x in e]); G.add_nodes_from(range(n)); D = sp.diag(*[G.degree(v) for v in range(n)])
    gr = grade(p)[0]; hs = halves(p) if gr != "none" else []
    return dict(label=label, n=n, grade=gr, regular=is_regular(D), n_pairs=len([1 for g, sg, m in golden_pairs(p) if sg is not None]),
                n_halves=len(hs), classes=[(c, str(s)) for g, c, s in class_sectors(p)] if gr != "none" else [],
                evaluable=(gr != "none" and not is_regular(D)))

if __name__ == "__main__":
    max_pac = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time(); out = dict(O1=[], O2=[], O3=[], O4=[], nulls={}); L = lambda s: print(s, flush=True)
    # O1: complete PAC binary trees (adjacency only)
    for d in range(3, max_pac + 1):
        nn = 2 ** (d + 1) - 1; ed = [(i, 2 * i + 1) for i in range(nn) if 2 * i + 1 < nn] + [(i, 2 * i + 2) for i in range(nn) if 2 * i + 2 < nn]
        r = describe(nn, ed, f"PAC depth {d}"); out["O1"].append(r); L(f"O1 d={d} n={nn}: {r} [{time.time()-t0:.0f}s]")
    out["O1_deferred"] = [f"depth {d} (n={2**(d+1)-1}): sector route, after n=24 frees the cores" for d in range(max_pac + 1, 9)]
    # O2: growth trees (replayed), evaluable subset n <= 100, grades only
    triples = [(5, mc, 100, mc * 100, "exp_07") for mc in (2, 3, 4, 5, 6, 7, 8)] + [(dl, 3, 100, dl * 100, "exp_08") for dl in (1, 2, 3, 4, 5, 6, 8, 10)] + \
              [(dl, mc, 50, mc * 100 + dl, "exp_08_grid") for mc in (2, 3, 4, 5) for dl in (1, 2, 3, 4)]
    seen = set(); o2 = []
    for dl, mc, it, seed, src in triples:
        _, trees = replay(1.0, dl, mc, n_iterations=it, seed=seed)
        for i, tr in enumerate(trees):
            key = (tr["n"], tuple(sorted(tr["edges"])))
            if key in seen: continue
            seen.add(key)
            if tr["n"] > 100 or tr["n"] < 4: o2.append(dict(src=src, depth=dl, mc=mc, seed=seed, i=i, n=tr["n"], grade="not-evaluated(size)")); continue
            # relabel to 0..n-1 contiguous
            nodes = sorted({v for e_ in tr["edges"] for v in e_}); idx = {v: k for k, v in enumerate(nodes)}
            r = describe(len(nodes), [(idx[a], idx[b]) for a, b in tr["edges"]], f"{src} d{dl} mc{mc} s{seed} #{i}"); r.update(src=src, depth=dl, mc=mc, seed=seed, i=i); o2.append(r)
    out["O2"] = o2; ev2 = [r for r in o2 if r.get("evaluable")]
    L(f"O2: {len(o2)} distinct trees; evaluable (grade != none, n<=100): {len(ev2)}; grades: { {g: sum(1 for r in o2 if r.get('grade') == g) for g in ('strict','core','partial','none','not-evaluated(size)')} } [{time.time()-t0:.0f}s]")
    # O3: M15 unicyclic controls (RandomState(152), m in 7, 9, 11, 20 each, in the exp_01 order)
    from representative import random_unicyclic, cycle_basis_single
    rng = np.random.RandomState(152); o3 = []
    for m in (7, 9, 11):
        for k in range(20):
            g = random_unicyclic(m, rng)
            try:
                cyc = cycle_basis_single(g)
                if len(cyc) < 3: o3.append(dict(label=f"unicyclic m={m} #{k}", skipped="cycle<3")); continue
            except Exception as ex:
                o3.append(dict(label=f"unicyclic m={m} #{k}", skipped=repr(ex))); continue
            n_, e_ = cartan_from_adj(g); o3.append(describe(n_, e_, f"unicyclic m={m} #{k}"))
    out["O3"] = o3; L(f"O3: {len(o3)} graphs; grades: { {g: sum(1 for r in o3 if r.get('grade') == g) for g in ('strict','core','partial','none')} }; evaluable {sum(1 for r in o3 if r.get('evaluable'))} [{time.time()-t0:.0f}s]")
    # O4: M13 density sweep (n in 10,12,14 x extra in 1..6) + the two n=12 graphs (extra 4 lump, extra 0 control)
    from identity_complement import build_density_graph
    o4 = []
    for n_ in (10, 12, 14):
        for extra in (1, 2, 3, 4, 5, 6):
            A = build_density_graph(n=n_, lump_center=n_ // 2, lump_radius=2, lump_extra_edges=extra); nn, ee = cartan_from_adj(A); o4.append(describe(nn, ee, f"density n={n_} extra={extra}"))
    for extra, lab in ((4, "lump"), (0, "control")):
        A = build_density_graph(n=12, lump_center=6, lump_radius=2, lump_extra_edges=extra); nn, ee = cartan_from_adj(A); o4.append(describe(nn, ee, f"density n=12 {lab}"))
    out["O4"] = o4; L(f"O4: {len(o4)} graphs; grades: { {g: sum(1 for r in o4 if r.get('grade') == g) for g in ('strict','core','partial','none')} }; evaluable {sum(1 for r in o4 if r.get('evaluable'))} [{time.time()-t0:.0f}s]")
    # nulls: exhaustive n=15 grade distribution; degree-matched random subcubic trees at 31 (and 63 if time)
    dist = {}
    for T in nx.nonisomorphic_trees(15):
        e_ = list(T.edges()); p = sp.expand(cart(15, e_).charpoly(t).as_expr()); g = grade(p)[0]; dist[g] = dist.get(g, 0) + 1
    out["nulls"]["n15_exhaustive_grades"] = dist; L(f"null n=15 exhaustive: {dist} [{time.time()-t0:.0f}s]")
    rng = np.random.RandomState(20260903); sub = {}
    for nn in (31, 63):
        cnt = {}
        for k in range(40 if nn == 31 else 20):
            # random subcubic tree by random attachment with degree cap 3
            deg = [0] * nn; e_ = []
            for v in range(1, nn):
                cands = [u for u in range(v) if deg[u] < 3]; u = int(rng.choice(cands)); e_.append((u, v)); deg[u] += 1; deg[v] += 1
            g = grade(sp.expand(cart(nn, e_).charpoly(t).as_expr()))[0]; cnt[g] = cnt.get(g, 0) + 1
        sub[nn] = cnt; L(f"null subcubic random trees n={nn}: {cnt} [{time.time()-t0:.0f}s]")
    out["nulls"]["subcubic_random"] = sub
    E = sum(1 for r in out["O1"] if r["evaluable"]) + len(ev2) + sum(1 for r in o3 if r.get("evaluable")) + sum(1 for r in o4 if r.get("evaluable"))
    out["informative_count_E"] = E; out["halves_total"] = sum(r.get("n_halves", 0) for grp in ("O1", "O2", "O3", "O4") for r in out[grp] if r.get("evaluable"))
    L(f"|E| (evaluable, non-blind, grade != none) = {E}; halves total = {out['halves_total']}")
    json.dump(out, open(RES / f"explore_d1_inventory_grades_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1, default=str)
    L("DONE")
