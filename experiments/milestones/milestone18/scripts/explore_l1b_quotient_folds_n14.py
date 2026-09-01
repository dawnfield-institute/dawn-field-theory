#!/usr/bin/env python3
"""Panel L1b at n=14 (EXPLORING; feeds exp_12 T6 scope): are the accidental golden trees SYMMETRY-QUOTIENT folds?
Equitable partition by automorphism orbits -> quotient matrix -> does its charpoly carry
exactly the golden factors of the tree?"""
import json, sympy as sp, numpy as np, networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
t=sp.Symbol('t'); s5=sp.sqrt(5); phi=(1+s5)/2
def cart(n,e):
    C=2*sp.eye(n)
    for i,j in e: C[i,j]=C[j,i]=-1
    return C
d=json.load(open('/Users/petergroom/repos/core_workspace/worktrees/dft-m18-founding/experiments/milestones/milestone18/results/explore_g1_census_20260901.json'))
d14=json.load(open('/Users/petergroom/repos/core_workspace/worktrees/dft-m18-founding/experiments/milestones/milestone18/results/exp_11b_core_trees_n14.json'))
d=[{"n":14,"edges":r["edges"],"det":r["det"],"fields":{"sqrt5":{"grade":r["grade"],"rational":r["rational"]}}} for r in d14]
gold={sp.expand(cart(r["n"],[tuple(x) for x in r["edges"]]).charpoly(t).as_expr()):r for r in d if r["fields"].get("sqrt5",{}).get("grade") in ("strict","core")}
parents=set()
for k in range(2,8):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr()); parents.add(sp.expand(q*q.subs(s5,-s5)))
def orbits(G):
    """Automorphism orbits of a TREE via the AHU rooted canonical form: two vertices of a tree lie in
    the same orbit iff the tree rooted at each has the same canonical string (exact for trees; no
    automorphism enumeration)."""
    def canon(root):
        def rec(v,parent):
            return "("+"".join(sorted(rec(w,v) for w in G[v] if w!=parent))+")"
        return rec(root,None)
    key={v:canon(v) for v in G}; seen={}; out=[]
    for v in sorted(G):
        seen.setdefault(key[v],[]).append(v)
    return [sorted(c) for c in seen.values()]
results=[]
for p,r in sorted(gold.items(), key=lambda x:(x[1]["n"],x[1]["det"])):
    if p in parents: continue
    n=r["n"]; e=[tuple(x) for x in r["edges"]]; G=nx.Graph(e); orb=orbits(G)
    # quotient of the Cartan matrix by the orbit partition (equitable: sum over target cell)
    A=nx.to_numpy_array(G,nodelist=sorted(G)); C=2*np.eye(n)-A; k=len(orb)
    Q=np.zeros((k,k))
    for a,ca in enumerate(orb):
        for b,cb in enumerate(orb):
            Q[a,b]=C[ca[0],cb].sum()
    Qs=sp.Matrix(Q.round(9).astype(int).tolist()) if np.allclose(Q,Q.round()) else sp.Matrix(Q)
    qp=sp.expand(Qs.charpoly(t).as_expr()); fq=sp.factor(qp,extension=s5)
    gold_tree=sorted(str(sp.expand(g.as_base_exp()[0])) for g in sp.Mul.make_args(sp.factor(p,extension=s5)) if g.has(s5))
    gold_quot=sorted(str(sp.expand(g.as_base_exp()[0])) for g in sp.Mul.make_args(fq) if g.has(s5))
    print(f"n={n} det={r['det']} orbits={k} (sizes {[len(c) for c in orb]})")
    print(f"   quotient charpoly over Q(sqrt5): {fq}")
    print(f"   golden factors: tree {set(gold_tree)==set(gold_quot) and 'EXACTLY those of the quotient' or ('tree='+str(gold_tree)+' quot='+str(gold_quot))}")
    results.append({"n":n,"det":r["det"],"edges":e,"orbits":len(orb),"orbit_sizes":[len(c) for c in orb],"quotient_charpoly":str(fq),"golden_exact":set(gold_tree)==set(gold_quot),"tree_golden":gold_tree,"quotient_golden":gold_quot})
json.dump(results,open('/Users/petergroom/repos/core_workspace/worktrees/dft-m18-founding/experiments/milestones/milestone18/results/explore_l1b_quotient_folds_n14.json','w'),indent=1)
print("DONE", sum(x["golden_exact"] for x in results), "of", len(results), "non-Galois golden trees at n=14 are automorphism-quotient folds")
