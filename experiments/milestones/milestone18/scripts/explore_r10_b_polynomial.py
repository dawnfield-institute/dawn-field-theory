#!/usr/bin/env python3
"""explore_r10 (exploration): the R-generator polynomial. On a strict fold R = sqrt5*b(C) with
b in Q[t] the unique rational polynomial (deg < n) taking value 1/sqrt5 on the roots of q and
-1/sqrt5 on the roots of sigma(q). Computes b for the seven strict folds. Finding: 5*b is an
INTEGER polynomial on every fold (leading coefficient 2 or 4); the vertex law |R_vv| = 1/sqrt5
is the statement diag(b(C)) = +-1/5, i.e. an integer-polynomial-in-C congruence. Also records
(sigma q - q)/sqrt5, a monic integer polynomial (the 'Galois direction')."""
import json, sympy as sp, sys, networkx as nx
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import cart, t
s5=sp.sqrt(5); phi=(1+s5)/2; RES=Path(__file__).parent.parent/"results"
census=json.load(open(RES/'explore_g1_census_20260901.json'))
strict=[r for r in census if r.get("fields",{}).get("sqrt5",{}).get("grade")=="strict"]
diag={}
for k in range(2,7):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr()); diag.setdefault(sp.expand(q*q.subs(s5,-s5)),q)
out=[]
for r in strict:
    n=r["n"]; E=[tuple(e) for e in r["edges"]]; C=cart(n,E)
    p=sp.expand(C.charpoly(t).as_expr()); q=diag[p]; sq=sp.expand(q.subs(s5,-s5))
    K=sp.QQ.algebraic_field(s5)
    Q,SQ=sp.Poly(q,t,domain=K),sp.Poly(sq,t,domain=K)
    u,v,g=sp.gcdex(Q,SQ)
    a=sp.Poly(sp.expand(((v*SQ)/g.all_coeffs()[0]).as_expr()),t)
    b=sp.Poly(sp.rem(sp.expand((a.as_expr()-a.as_expr().subs(s5,-s5))/s5),p,t),t)
    fb=sp.factor(b.as_expr()); five_b=sp.Poly(sp.expand(5*b.as_expr()),t)
    assert all(c.is_integer for c in five_b.all_coeffs()), "5b must be integral"
    w=sp.factor(sp.expand((sq-q)/s5))
    print(f"n={n} det={r['det']:>4} deg b={b.degree()}  5b integral: yes  lead(5b)={five_b.LC()}")
    print("   b =",fb); print("   (sigma q - q)/sqrt5 =",w)
    out.append({"n":n,"det":r["det"],"edges":E,"b":str(fb),"five_b_coeffs":[str(c) for c in five_b.all_coeffs()],"galois_direction":str(w)})
json.dump(out,open(RES/'explore_r10_b_polynomial.json','w'),indent=1)
print("DONE")
