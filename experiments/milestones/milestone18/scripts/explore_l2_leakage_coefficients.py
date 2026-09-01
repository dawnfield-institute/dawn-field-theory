#!/usr/bin/env python3
"""Panel L2 (EXPLORING): exp_05's coefficient ||(I-P)BP||^2 across ALL strict folds n<=12.
A4 and E8 both gave 2/5. Is it universal, or does it track something?"""
import sys, json, sympy as sp, networkx as nx
sys.path.insert(0,'/Users/petergroom/repos/core_workspace/worktrees/dft-m18-founding/experiments/milestones/milestone18/core')
from ledger import bezout_proj, simp, cart
t=sp.Symbol('t'); s5=sp.sqrt(5); phi=(1+s5)/2
d=json.load(open('/Users/petergroom/repos/core_workspace/worktrees/dft-m18-founding/experiments/milestones/milestone18/results/explore_g1_census_20260901.json'))
strict={sp.expand(cart(r["n"],[tuple(x) for x in r["edges"]]).charpoly(t).as_expr()):r for r in d if r["fields"].get("sqrt5",{}).get("grade")=="strict"}
diag={}
for k in (2,4,6):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr()); diag.setdefault(sp.expand(q*q.subs(s5,-s5)),(q,sorted(map(list,E)),E[pos]))
print(f"{'n':>3} {'det':>5} {'diagram (5-bond)':<34} {'||(I-P)BP||^2':>14} {'||(I-P)B||^2':>13} {'tr(PB)':>8}")
for p,r in sorted(strict.items(), key=lambda x:(x[1]["n"],x[1]["det"])):
    n=r["n"]; e=[tuple(x) for x in r["edges"]]; C=cart(n,e)
    q,E,bond=diag[p]; P=bezout_proj(C,q)
    A=sp.zeros(n,n)
    for i,j in e: A[i,j]=A[j,i]=1
    D=sp.diag(*[sum(A[i,k] for k in range(n)) for i in range(n)]); B=2*sp.eye(n)-D
    X=simp((sp.eye(n)-P)*B*P); nrm=sp.nsimplify(sp.expand(sum(x**2 for x in X)))
    Y=simp((sp.eye(n)-P)*B); nrmB=sp.nsimplify(sp.expand(sum(x**2 for x in Y)))
    trPB=sp.nsimplify(sp.expand((P*B).trace()))
    print(f"{n:>3} {r['det']:>5} {str(E)+' '+str(bond):<34} {str(nrm):>14} {str(nrmB):>13} {str(trPB):>8}")
