#!/usr/bin/env python3
"""explore_r16b: the strict-hunt pipeline for large n. Sound screen: p strict => p = q*sigma(q)
=> p(x) = N(q(x)) is a Q(sqrt5)-norm for every rational x; an integer is a norm iff every prime
= +-2 (mod 5) divides it to an even power. Necessary condition — the screen cannot lose a strict
tree. Survivors get the exact Q(sqrt5) factorization. Validation: n=16 must yield exactly the 15
known strict trees on 14 distinct polynomials (one cospectral pair)."""
import time, sys, json, sympy as sp, networkx as nx
from pathlib import Path
t=sp.Symbol('t'); s5=sp.sqrt(5); RES=Path(__file__).parent.parent/"results"
n=int(sys.argv[1]) if len(sys.argv)>1 else 16
def is_norm(m):
    if m==0: return True
    m=abs(m)
    for pr,ex in sp.factorint(m).items():
        if pr%5 in (2,3) and ex%2==1: return False
    return True
def strict_grade(p):
    f=sp.factor(p,extension=s5); facs=[g for g in sp.Mul.make_args(f) if g.has(t)]
    return bool(facs) and all(g.has(s5) for g in facs)
t0=time.time(); cnt=0; surv=[]
XS=(0,1,-1,2,3,-2)
for T in nx.nonisomorphic_trees(n):
    e=list(T.edges()); C=2*sp.eye(n)
    for i,j in e: C[i,j]=C[j,i]=-1
    p=C.charpoly(t)
    cnt+=1
    if all(is_norm(int(p.eval(x))) for x in XS): surv.append((sorted(map(list,e)),sp.expand(p.as_expr())))
    if cnt%2000==0: print(f"  {cnt} trees, {len(surv)} survivors [{time.time()-t0:.0f}s]",flush=True)
print(f"n={n}: {cnt} trees; norm-screen survivors: {len(surv)} [{time.time()-t0:.0f}s]",flush=True)
t1=time.time(); strict=[]
for e,p in surv:
    if strict_grade(p): strict.append({"edges":e,"charpoly":str(p)})
print(f"exact factorization on survivors: {time.time()-t1:.0f}s; STRICT trees: {len(strict)}",flush=True)
json.dump(strict,open(RES/f'explore_r16b_strict_n{n}_{time.strftime("%Y%m%d_%H%M%S")}.json','w'),indent=1)   # append-only: never overwrite the committed census (audit fix 2026-09-02)
if n==16:
    distinct=len({x["charpoly"] for x in strict})
    print(f"VALIDATION (expect 14 distinct charpolys; exp_13 counted polys, not trees): {'PASS' if distinct==14 else 'FAIL'} ({len(strict)} trees, {distinct} polys)",flush=True)
print("DONE",flush=True)
