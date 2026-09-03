#!/usr/bin/env python3
"""explore_r7b (exploration): the two-component structure on the REGISTERED-DOMAIN core-grade folds
(conic-resolved core, gauge-independent traces, Ledger projector) at n<=14 — the 12 folds on which
tr(RD)=2/sqrt5 held in exp_12 T5. Uses exp_12's projector. Reports |R_vv| values, the copy/conjugate
vertex split by sign of R_vv, components, cut, and the diagram's halves at its 5-bond."""
import sys, json, sympy as sp, networkx as nx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from foldlaws import *   # promoted from exp_12_part2 on 2026-09-02
census, golden, diag = load_context(14)
rs=lambda x: sp.radsimp(sp.expand(x))
# rebuild diag with edge/pos info for the 5-bond halves
diag2={}
for k in range(2,8):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr()); diag2.setdefault(sp.expand(q*q.subs(s5,-s5)),(E,pos))
out=[]
for r in golden:
    n=r["n"]; e=[tuple(x) for x in r["edges"]]; p=charpoly(n,e)
    if p not in diag or grade_of(r)!="core": continue
    q,M=diag[p]; Ps,C,standin=projector_for(n,e,q)
    if standin: continue
    inv=[invariants(P_,n,e) for P_ in Ps]
    if not all(sp.simplify(inv[0][i]-inv[-1][i])==0 for i in range(3)): continue   # gauge-dependent: skip
    P=Ps[0]; R=simp(P-sigma(P)); d=[rs(R[v,v]) for v in range(n)]
    vals=sorted(set(str(x) for x in d))
    copy=[v for v in range(n) if d[v].evalf()>0]; conj=[v for v in range(n) if d[v].evalf()<0]; zero=[v for v in range(n) if d[v]==0]
    G=nx.Graph(e); cut=[x for x in e if (x[0] in copy)!=(x[1] in copy)]
    sizes=sorted(len(c) for c in nx.connected_components(G.subgraph(conj))) if conj else []
    ccopy=nx.number_connected_components(G.subgraph(copy)) if copy else 0
    E,pos=diag2[p]; DG=nx.Graph(E); DG.remove_edge(*E[pos]); halves=sorted(len(c) for c in nx.connected_components(DG))
    print(f"n={n} det={r['det']:>5} trRD={inv[0][0]} |R_vv| values={vals} copy {len(copy)}v ({ccopy} comp) conj {len(conj)}v comps {sizes} zero {len(zero)} cut {len(cut)} diagram halves {halves}")
    out.append({"n":n,"det":r["det"],"edges":e,"R_diag":[str(x) for x in d],"copy":copy,"conj":conj,"zero":zero,"cut":cut,"conj_comp_sizes":sizes,"copy_components":ccopy,"diagram_halves":halves})
json.dump(out,open(RES/'explore_r7b_core_fold_structure.json','w'),indent=1,default=str)
print("DONE")
