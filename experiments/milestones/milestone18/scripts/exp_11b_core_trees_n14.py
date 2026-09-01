#!/usr/bin/env python3
"""Supplement to exp_11: record every core-grade sqrt5-golden tree on 14 vertices (edges,
det) so exp_12 T5/T6 can evaluate the fold laws at n=14. Same grade function as exp_11."""
import json, time, sympy as sp, networkx as nx
from pathlib import Path
t=sp.Symbol('t'); s5=sp.sqrt(5)
def cart(n,e):
    C=2*sp.eye(n)
    for i,j in e: C[i,j]=C[j,i]=-1
    return C
def grade(p):
    fQ=sp.factor(p); facsQ=[g for g in sp.Mul.make_args(fQ) if g.has(t)]
    for g in facsQ:
        b,e=g.as_base_exp()
        if sp.degree(b,t)%2==1 and sp.degree(b,t)>1: return "none"
    f=sp.factor(p,extension=s5); facs=[g for g in sp.Mul.make_args(f) if g.has(t)]
    gold=[g for g in facs if g.has(s5)]; rat=[g for g in facs if not g.has(s5)]
    if not gold: return "none"
    if not rat: return "strict"
    return "core" if all(g.as_base_exp()[1]%2==0 for g in rat) else "partial"
out=[]; t0=time.time(); n=14; c=0
for T in nx.nonisomorphic_trees(n):
    e=sorted(map(list,T.edges())); p=sp.expand(cart(n,[tuple(x) for x in e]).charpoly(t).as_expr()); c+=1
    g=grade(p)
    if g in ("strict","core"):
        out.append({"n":n,"edges":e,"grade":g,"det":int(sp.Poly(p,t).all_coeffs()[-1]),"rational":[str(x) for x in sp.Mul.make_args(sp.factor(p,extension=s5)) if x.has(t) and not x.has(s5)]})
    if c%500==0: print(f"  {c} [{time.time()-t0:.0f}s]",flush=True)
Path(__file__).parent.parent.joinpath("results","exp_11b_core_trees_n14.json").write_text(json.dumps(out,indent=1))
print(f"DONE: {len(out)} golden (strict+core) trees at n=14 [{time.time()-t0:.0f}s]",flush=True)
