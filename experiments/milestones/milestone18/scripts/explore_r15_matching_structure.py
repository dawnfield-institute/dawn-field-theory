#!/usr/bin/env python3
"""explore_r15 (exploration): the matching form of the reflection. Conjecture: on every strict fold,
sqrt5*R = S + 2*Pi with S = diag(+-1) and Pi a symmetric permutation (perfect matching) satisfying
S*Pi = -Pi*S (matched vertices carry opposite signs). Then |R_vv|=1/sqrt5, R^2=I, and the fold
structure are corollaries. Also: quotient the parent by Pi and compare pair-adjacency multiplicities
with the one-5 diagram — is the 5-bond the multiplicity-3 edge?"""
import json, sys, sympy as sp, networkx as nx
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import cart, simp, sigma, bezout_proj, t
s5=sp.sqrt(5); phi=(1+s5)/2; RES=Path(__file__).parent.parent/"results"
NMAX=int(sys.argv[1]) if len(sys.argv)>1 else 12
partners={}
for k in range(2,NMAX//2+1):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr())
            partners.setdefault(sp.expand(q*q.subs(s5,-s5)),(q,E,pos))
folds=[]
if NMAX<=13:
    cen=json.load(open(RES/'explore_g1_census_20260901.json'))
    folds=[(r["n"],r["edges"],r["det"]) for r in cen if r["fields"].get("sqrt5",{}).get("grade")=="strict"]
else:
    r13=json.load(open(RES/'exp_13_n16.json'))
    folds=[(16,x["edges"],None) for x in r13["T3"]]
out=[]
for n,e,det in folds:
    e=[tuple(v) for v in e]; C=cart(n,e); p=sp.expand(C.charpoly(t).as_expr())
    q,dE,pos=partners[p]
    R=simp(bezout_proj(C,q)); R=simp(R-sigma(R))
    S5R=simp(s5*R).applyfunc(sp.nsimplify)
    ok_form=True; match={}
    for v in range(n):
        row=[(w,S5R[v,w]) for w in range(n) if w!=v and S5R[v,w]!=0]
        if not (S5R[v,v]**2==1 and len(row)==1 and row[0][1]**2==4): ok_form=False; break
        match[v]=row[0][0]
    anti=ok_form and all(S5R[v,v]==-S5R[match[v],match[v]] for v in match)
    quot_ok=fivebond_mult3=None
    if ok_form:
        pairs={frozenset((v,match[v])) for v in match}
        pid={fs:i for i,fs in enumerate(sorted(pairs,key=sorted))}
        Q=nx.MultiGraph()
        for a,b in e:
            pa=pid[[fs for fs in pairs if a in fs][0]]; pb=pid[[fs for fs in pairs if b in fs][0]]
            if pa!=pb: Q.add_edge(pa,pb)
        mult={tuple(sorted((a,b))):0 for a,b in Q.edges()}
        for a,b in Q.edges(): mult[tuple(sorted((a,b)))]+=1
        DG=nx.Graph(dE)
        gm=nx.algorithms.isomorphism.GraphMatcher(nx.Graph(list(mult.keys())),DG)
        quot_ok=gm.is_isomorphic()
        if quot_ok:
            iso=gm.mapping  # quotient node -> diagram node
            fb=tuple(sorted(dE[pos]))
            m3={k2 for k2,v2 in mult.items() if v2==3}
            fivebond_mult3=any(tuple(sorted((iso[a],iso[b])))==fb for a,b in m3)
    print(f"n={n} det={det if det is not None else int(C.det()):>5} form S+2Pi:{ok_form} signs-anticommute:{anti} quotient=diagram:{quot_ok} 5-bond=mult-3 edge:{fivebond_mult3} mults:{sorted(mult.values()) if ok_form else '-'}",flush=True)
    out.append({"n":n,"edges":e,"form":ok_form,"anti":anti,"quotient_iso":quot_ok,"fivebond_mult3":fivebond_mult3,
                "matching":match if ok_form else None,"multiplicities":sorted(mult.values()) if ok_form else None})
json.dump(out,open(RES/f'explore_r15_matching_n{NMAX}.json','w'),indent=1,default=str)
print("DONE",flush=True)
