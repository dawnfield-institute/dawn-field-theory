#!/usr/bin/env python3
"""exp_08 (Block C extension): the census. Registration: Phase 2 seal 06073227.
Prufer-sampled random trees n=6..10, >=500 distinct per n. Criterion: COMPLETE sigma-pairing
(charpoly = q*sigma(q) over Q(sqrt5), q not in Q[t]) in (a) Cartan and (b) A^2 channels.
Predictions fixed: (a) ~0 outside ADE; (b) >0 with hits at spectral radius > 2."""
import json, sys, numpy as np, sympy as sp, heapq, time
from pathlib import Path
t=sp.Symbol('t'); s5=sp.sqrt(5); rng=np.random.default_rng(20260901)

def prufer_tree(n):
    seq=rng.integers(0,n,size=n-2); deg=np.ones(n,int)
    for x in seq: deg[x]+=1
    leaves=[i for i in range(n) if deg[i]==1]; heapq.heapify(leaves); edges=[]
    for x in seq:
        leaf=heapq.heappop(leaves); edges.append((leaf,int(x))); deg[x]-=1
        if deg[x]==1: heapq.heappush(leaves,int(x))
    u=heapq.heappop(leaves); v=heapq.heappop(leaves); edges.append((u,v)); return edges
def adj(n,e):
    A=sp.zeros(n,n)
    for i,j in e: A[i,j]=A[j,i]=1
    return A
def complete_pairing(M):
    """COMPLETE sigma-pairing: every irreducible factor over Q(sqrt5) carries sqrt5 (no
    rational-coefficient factor). For a rational polynomial the golden factors are then
    automatically sigma-closed, so charpoly = q*sigma(q) with q not in Q[t]. EXACT.
    (First version mis-paired factors and missed E8's Cartan; caught by the self-test.)"""
    p=sp.expand(M.charpoly(t).as_expr())
    f=sp.factor(p, extension=s5)
    facs=[g for g in sp.Mul.make_args(f) if g.has(t)]
    gold=[g for g in facs if g.has(s5)]
    return bool(gold) and len(gold)==len(facs)

import networkx as nx
def iso_class(n,e):
    G=nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(e)
    return nx.weisfeiler_lehman_graph_hash(G, iterations=n)

# ---- instrument self-test (aborts the census if the criterion is wrong) ----
_E8=[(i,i+1) for i in range(6)]+[(2,7)]; _A5=[(i,i+1) for i in range(4)]
_cat8=[(0,1),(1,2),(2,3),(1,4),(2,5),(0,6),(3,7)]
assert complete_pairing(2*sp.eye(8)-adj(8,_E8)), "self-test: E8 Cartan must be complete-paired"
assert not complete_pairing(2*sp.eye(5)-adj(5,_A5)), "self-test: A5 Cartan must NOT be"
assert complete_pairing(adj(8,_cat8)*adj(8,_cat8)), "self-test: cat8 A^2 must be complete-paired"
print("instrument self-test passed", flush=True)
out={"registration":"06073227","per_n":{}}
for n in range(6,11):
    seen=set(); classes={}; t0=time.time()
    while len(seen)<500:
        e=prufer_tree(n); key=frozenset(frozenset(p) for p in e)
        if key in seen: continue
        seen.add(key); h=iso_class(n,e)
        if h in classes: classes[h]["labeled_draws"]+=1; continue
        A=adj(n,e); C=2*sp.eye(n)-A
        lam=max(abs(x) for x in np.linalg.eigvalsh(np.array(A.tolist(),dtype=float)))
        classes[h]={"edges":e,"labeled_draws":1,"cartan":complete_pairing(C),
                    "a2":complete_pairing(A*A),"spectral_radius":round(float(lam),6)}
    cl=list(classes.values())
    out["per_n"][n]={"labeled_trees":len(seen),"iso_classes_seen":len(cl),
        "cartan_hit_classes":sum(c["cartan"] for c in cl),"a2_hit_classes":sum(c["a2"] for c in cl),
        "a2_hits_radius_gt2":sum(c["a2"] and c["spectral_radius"]>2+1e-9 for c in cl),
        "a2_hits_radius_le2":sum(c["a2"] and c["spectral_radius"]<=2+1e-9 for c in cl),
        "cartan_hit_examples":[c["edges"] for c in cl if c["cartan"]][:3],
        "a2_hit_examples":[{"edges":c["edges"],"radius":c["spectral_radius"]} for c in cl if c["a2"]][:6],
        "seconds":round(time.time()-t0,1)}
    o=out["per_n"][n]
    print(f"n={n}: {o['labeled_trees']} labeled / {o['iso_classes_seen']} iso classes; Cartan hits={o['cartan_hit_classes']}, "
          f"A^2 hits={o['a2_hit_classes']} (radius>2: {o['a2_hits_radius_gt2']}, <=2: {o['a2_hits_radius_le2']})  [{o['seconds']}s]", flush=True)
    Path(__file__).parent.parent.joinpath("results","exp_08_census_20260901.json").write_text(json.dumps(out,indent=1,default=str))
print("CENSUS DONE")
