#!/usr/bin/env python3
"""explore_r7 (exploration, not registered): vertex structure of the strict Galois folds at n<=13.
Uses the exp_12 instrument: P = Bezout projector for the H-diagram's own polynomial q.
Checks |R_vv|=1/sqrt5 at every vertex, the copy/conjugate vertex split, the cut, and compares the
conjugate-side component sizes with the two sides of the H-diagram when its 5-bond is removed."""
import json, sympy as sp, networkx as nx, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import cart, simp, sigma, bezout_proj, t
s5=sp.sqrt(5); phi=(1+s5)/2; rs=lambda x: sp.radsimp(sp.expand(x)); RES=Path(__file__).parent.parent/"results"
diag={}
for k in range(2,7):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr()); diag.setdefault(sp.expand(q*q.subs(s5,-s5)),(q,M,E,pos))
census=json.load(open(RES/'explore_g1_census_20260901.json'))
strict=[r for r in census if r.get("fields",{}).get("sqrt5",{}).get("grade")=="strict"]
out=[]
for r in strict:
    n=r["n"]; E=[tuple(e) for e in r["edges"]]; C=cart(n,E)
    q,M,DE,pos=diag[sp.expand(C.charpoly(t).as_expr())]
    P=bezout_proj(C,q); R=simp(P-sigma(P)); d=[rs(R[v,v]) for v in range(n)]
    ok=all(sp.simplify(x**2-sp.Rational(1,5))==0 for x in d)
    copy=[v for v in range(n) if sp.simplify(d[v]-1/s5)==0]; conj=[v for v in range(n) if v not in copy]
    G=nx.Graph(E); cut=[e for e in E if (e[0] in copy)!=(e[1] in copy)]
    cc,cj=nx.number_connected_components(G.subgraph(copy)),nx.number_connected_components(G.subgraph(conj))
    dc,dj=sum(G.degree(v) for v in copy),sum(G.degree(v) for v in conj)
    sizes=sorted(len(c) for c in nx.connected_components(G.subgraph(conj)))
    DG=nx.Graph(DE); DG.remove_edge(*DE[pos]); dsides=sorted(len(c) for c in nx.connected_components(DG))
    print(f"n={n:>2} det={r['det']:>4} |R_vv|=1/sqrt5:{ok} copy {len(copy)}v {cc}comp deg-sum {dc} | conj {len(conj)}v {cj}comp sizes {sizes} deg-sum {dj} | cut {len(cut)} | diagram sides at 5-bond {dsides} match:{sizes==dsides}")
    out.append({"n":n,"det":r["det"],"edges":E,"copy":copy,"conj":conj,"cut":cut,"conj_comp_sizes":sizes,"deg_sums":[dc,dj],"diagram_edges":DE,"five_bond":list(DE[pos]),"diagram_sides":dsides,"sides_match":sizes==dsides})
json.dump(out,open(RES/'explore_r7_two_component_corollary.json','w'),indent=1)
