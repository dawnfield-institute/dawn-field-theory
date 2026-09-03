#!/usr/bin/env python3
"""explore_r15c: is the defect edge always COPY-internal (both endpoints sign +1)? If yes, then
clauses 3-5 of the matching conjecture imply the trace law outright: every non-defect parent edge
pairs with its Pi-image to a net-zero sign contribution (S*Pi = -Pi*S), so
sum_v d_v s_v = s_a + s_b over the defect edge alone = 2, i.e. tr(RD) = 2/sqrt5."""
import json, sys, sympy as sp, networkx as nx
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import cart, simp, sigma, bezout_proj, t
s5=sp.sqrt(5); phi=(1+s5)/2; RES=Path(__file__).parent.parent/"results"
partners={}
for k in range(2,9):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr())
            partners.setdefault(sp.expand(q*q.subs(s5,-s5)),q)
allok=True; tot=0
for f in ('explore_r15_matching_n12.json','explore_r15_matching_n16.json'):
    for x in json.load(open(RES/f)):
        n=x["n"]; e=[tuple(v) for v in x["edges"]]; C=cart(n,e)
        q=partners[sp.expand(C.charpoly(t).as_expr())]
        R=simp(bezout_proj(C,q)); R=simp(R-sigma(R))
        S={v:1 if sp.re((s5*R[v,v]).evalf())>0 else -1 for v in range(n)}
        m={int(k2):v for k2,v in x["matching"].items()}
        E={tuple(sorted(v)) for v in x["edges"]}
        defect=[Ed for Ed in E if tuple(sorted((m[Ed[0]],m[Ed[1]]))) not in E][0]
        side="copy" if (S[defect[0]]==1 and S[defect[1]]==1) else ("conj" if (S[defect[0]]==-1 and S[defect[1]]==-1) else "cut")
        checks=sum(sp.re((s5*R[v,v]).evalf())*d for v,d in dict(nx.Graph(e).degree()).items())
        print(f"n={n} defect {defect} side: {side}  (sum d_v s_v = {round(float(checks))})",flush=True)
        tot+=1; allok&=(side=="copy")
print(f"defect edge copy-internal: {tot} folds, all: {allok}",flush=True)
print("DONE",flush=True)
