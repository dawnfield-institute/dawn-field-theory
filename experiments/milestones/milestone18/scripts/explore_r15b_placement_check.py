#!/usr/bin/env python3
"""explore_r15b: on the cospectral-placement fold (n=16, det -239), does the Pi-quotient's
multiplicity-3 edge map onto the placement the fold realized ([3,5]) rather than the other ([1,7])?
Result 2026-09-01: yes — (0,5)/[3,5] carries the mult-3 edge under an isomorphism; (1,4)/[1,7] does not."""
import json, sympy as sp, networkx as nx, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import cart, t
s5=sp.sqrt(5); phi=(1+s5)/2; RES=Path(__file__).parent.parent/"results"
r=json.load(open(RES/'explore_r15_matching_n16.json'))
for x in r:
    e=[tuple(w) for w in x["edges"]]; C=cart(16,e)
    if int(C.det())!=-239: continue
    p=sp.expand(C.charpoly(t).as_expr()); plist=[]
    for T in nx.nonisomorphic_trees(8):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(8)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr())
            if sp.expand(q*q.subs(s5,-s5)-p)==0: plist.append((E,pos))
    match={int(k):v for k,v in x["matching"].items()}
    pairs=sorted({frozenset((v,match[v])) for v in match},key=sorted); pid={fs:i for i,fs in enumerate(pairs)}
    mult={}
    for a,b in e:
        pa=pid[[fs for fs in pairs if a in fs][0]]; pb=pid[[fs for fs in pairs if b in fs][0]]
        if pa!=pb: mult[tuple(sorted((pa,pb)))]=mult.get(tuple(sorted((pa,pb))),0)+1
    m3=[k for k,v in mult.items() if v==3]; QG=nx.Graph(list(mult.keys()))
    for E,pos in plist:
        gm=nx.algorithms.isomorphism.GraphMatcher(QG,nx.Graph(E))
        hit=any(any(tuple(sorted((iso[a],iso[b])))==tuple(sorted(E[pos])) for a,b in m3) for iso in gm.isomorphisms_iter())
        DGc=nx.Graph(E); DGc.remove_edge(*E[pos]); halves=sorted(len(c) for c in nx.connected_components(DGc))
        print(f"partner 5-bond {E[pos]} halves {halves}: mult-3 edge maps onto it: {hit}")
