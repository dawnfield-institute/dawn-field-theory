#!/usr/bin/env python3
"""Panel L4 (EXPLORING): exp_04's loose ends — the E/D-series s* march limit, and whether
accidental golden trees (star6 sat at s*=1.01) see the self-dual point."""
import json, numpy as np, networkx as nx
def M_of(n, edges, s):
    A=np.zeros((n,n))
    for i,j in edges: A[i,j]=A[j,i]=1
    D=np.diag(A.sum(1)); return (D-A)+s*(2*np.eye(n)-D)
def mfpt_ff(n, edges, s):
    M=M_of(n,edges,s); tot=0.0; k=0
    for tgt in range(n):
        sub=np.delete(np.delete(M,tgt,0),tgt,1)
        if abs(np.linalg.det(sub))<1e-10: continue
        tot+=float(np.linalg.solve(sub,np.ones(n-1)).mean()); k+=1
    return tot/k if k else np.nan
S=np.round(np.arange(0.50,1.501,0.01),3)
def s_star(n,e):
    v=[mfpt_ff(n,e,s) for s in S]; return float(S[int(np.nanargmin(v))])
def E_tree(n): return [(i,i+1) for i in range(n-2)]+[(2,n-1)]
def D_tree(n): return [(i,i+1) for i in range(n-2)]+[(n-3,n-1)]
print("E-series march (frame-free MFPT s*):")
for n in range(6,15): print(f"  E{n}-tree: s* = {s_star(n,E_tree(n)):.2f}")
print("D-series:")
for n in range(4,13): print(f"  D{n}-tree: s* = {s_star(n,D_tree(n)):.2f}")
d=json.load(open('/Users/petergroom/repos/core_workspace/worktrees/dft-m18-founding/experiments/milestones/milestone18/results/explore_g1_census_20260901.json'))
print("\ns* for ALL sqrt5-golden trees n<=12 (strict+core), vs null median |s*-1| = 0.25:")
for r in sorted([r for r in d if r["fields"].get("sqrt5",{}).get("grade") in ("strict","core")], key=lambda r:(r["n"],r["det"])):
    G=nx.Graph(r["edges"]); st=s_star(r["n"],[tuple(x) for x in r["edges"]])
    print(f"  n={r['n']:<3} det={r['det']:>6} maxdeg={max(dict(G.degree()).values()):>2} grade={r['fields']['sqrt5']['grade']:<6} s*={st:.2f}  |s*-1|={abs(st-1):.2f}")
