#!/usr/bin/env python3
"""Panel L1 (EXPLORING): the non-folding ('accidental') golden trees — where does their sqrt5 come from?"""
import json, sympy as sp, numpy as np, networkx as nx
t=sp.Symbol('t'); s5=sp.sqrt(5); phi=(1+s5)/2
def cart(n,e):
    C=2*sp.eye(n)
    for i,j in e: C[i,j]=C[j,i]=-1
    return C
d=json.load(open('/Users/petergroom/repos/core_workspace/worktrees/dft-m18-founding/experiments/milestones/milestone18/results/explore_g1_census_20260901.json'))
gold={}
for r in d:
    if r["fields"].get("sqrt5",{}).get("grade") in ("strict","core"):
        gold[sp.expand(cart(r["n"],[tuple(x) for x in r["edges"]]).charpoly(t).as_expr())]=r
parents=set()
for k in range(2,7):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr()); parents.add(sp.expand(q*q.subs(s5,-s5)))
acc=[(p,r) for p,r in gold.items() if p not in parents]
print(f"accidental golden trees: {len(acc)}")
for p,r in sorted(acc,key=lambda x:x[1]["n"]):
    G=nx.Graph(r["edges"]); n=r["n"]; A=nx.to_numpy_array(G)
    adj=sorted(np.linalg.eigvalsh(A)); degs=sorted(dict(G.degree()).values(),reverse=True)
    f=sp.factor(p,extension=s5); gold_f=[g for g in sp.Mul.make_args(f) if g.has(s5)]
    has_s5=any(abs(abs(x)-5**0.5)<1e-9 for x in adj)
    print(f"\n n={n} det={r['det']} degrees={degs}")
    print(f"   adjacency eigenvalues: {[round(x,4) for x in adj]}   contains ±sqrt5: {has_s5}")
    print(f"   golden factors: {gold_f}")
    print(f"   rational core: {r['fields']['sqrt5']['rational']}")
