#!/usr/bin/env python3
"""explore_r20 (exploration, not registered): the MAGNITUDE of the branch asymmetry. For each of the
47 n=20 folds (and the 21 at n<=16): the peak Pi-asymmetry a_max at the branch, the degrees of the
defect endpoints (d_a, d_b) and of their Pi-partners (d_a-1, d_b-1 by the degree ledger), and the
asymmetry of each of the two cut-edge pairs. Looks for a closed form: does a_max depend only on the
local degree data at the branch?"""
import json, networkx as nx, mpmath as mp, sys, sympy as sp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import cart, simp, sigma, bezout_proj, t
mp.mp.dps=30; RES=Path(__file__).parent.parent/"results"; s5=sp.sqrt(5); phi=(1+s5)/2
def heat_u(n,e,beta=mp.mpf(1)):
    C=mp.matrix(n,n)
    for i in range(n): C[i,i]=2
    for a,b in e: C[a,b]=C[b,a]=-1
    w,V=mp.eigsy(C); K=V*mp.diag([mp.e**(-beta*w[i]) for i in range(n)])*V.T; Z=sum(K[i,i] for i in range(n))
    return {tuple(sorted((a,b))):K[a,b]/Z for a,b in e}
objs=[]
for x in json.load(open(RES/'explore_r19_n20_defects.json')):
    objs.append((20,[tuple(v) for v in x["edges"]],{int(k):v for k,v in x["matching"].items()},tuple(x["defect"])))
# n<=16 folds: recompute matchings quickly via partners k<=8
partners={}
for k in range(2,9):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr()); partners.setdefault(str(sp.expand(q*q.subs(s5,-s5))),q)
small=[]
for f in ('explore_r15_matching_n12.json','explore_r15_matching_n16.json'):
    for x in json.load(open(RES/f)): small.append((x["n"],[tuple(v) for v in x["edges"]],{int(k):v for k,v in x["matching"].items()}))
for n,e,m in small:
    E={tuple(sorted(v)) for v in e}; d=[Ed for Ed in E if tuple(sorted((m[Ed[0]],m[Ed[1]]))) not in E][0]
    objs.append((n,e,m,d))
rows=[]
for n,e,m,d0 in objs:
    G=nx.Graph(e); deg=dict(G.degree()); u=heat_u(n,e)
    E={tuple(sorted(v)) for v in e}
    def asym(ed):
        im=tuple(sorted((m[ed[0]],m[ed[1]]))); return abs(u[ed]-u[im])/(u[ed]+u[im])
    a,b=d0; Pa,Pb=m[a],m[b]
    cut=[x for x in E if x!=d0 and ((x[0]==a and x[1]==Pb) or (x[0]==Pb and x[1]==a) or (x[0]==Pa and x[1]==b) or (x[0]==b and x[1]==Pa))]
    others=[x for x in E if x!=d0 and x not in cut]
    amax=max(asym(x) for x in others+cut)
    rows.append({"n":n,"deg_defect":(deg[a],deg[b]),"deg_partners":(deg[Pa],deg[Pb]),"cut_asym":[float(asym(x)) for x in cut],"a_max":float(amax),
                 "a_max_is_cut":any(abs(asym(x)-amax)<1e-20 for x in cut)})
import collections
print(f"{'n':>3} {'deg(a,b)':>9} {'deg(Pa,Pb)':>11} {'cut asym':>22} {'a_max':>9} cut-is-peak")
for r in sorted(rows,key=lambda r:(r["n"],r["deg_defect"])):
    print(f"{r['n']:>3} {str(r['deg_defect']):>9} {str(r['deg_partners']):>11} {str([round(v,5) for v in r['cut_asym']]):>22} {r['a_max']:.5f} {r['a_max_is_cut']}")
by=collections.defaultdict(list)
for r in rows: by[(r["n"],r["deg_defect"])].append(round(r["a_max"],6))
print("\na_max grouped by (n, defect degree pair):")
for k,v in sorted(by.items()): print("  ",k,":",sorted(set(v)),f"(count {len(v)})")
json.dump(rows,open(RES/'explore_r20_branch_magnitude.json','w'),indent=1)
print("DONE",flush=True)
