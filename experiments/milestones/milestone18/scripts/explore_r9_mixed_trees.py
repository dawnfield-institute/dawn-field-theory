#!/usr/bin/env python3
"""explore_r9 (exploration): the two n=14 golden trees that are neither Galois folds nor
automorphism-quotient folds (dets -620, -80). Split R^14 into the symmetric sector S (span of orbit
indicators; the quotient lives here) and its complement; restrict the Cartan matrix to each; factor the
charpolys over Q(sqrt5). Question: do the golden factors missing from the quotient form a Galois pair
q*sigma(q) in the complement, and on what sub-tree do those eigenvectors live?"""
import json, sympy as sp, networkx as nx
from pathlib import Path
t=sp.Symbol('t'); s5=sp.sqrt(5); RES=Path(__file__).parent.parent/"results"
def cart(n,e):
    C=2*sp.eye(n)
    for i,j in e: C[i,j]=C[j,i]=-1
    return C
def orbits(G):
    def canon(root):
        def rec(v,parent): return "("+"".join(sorted(rec(w,v) for w in G[v] if w!=parent))+")"
        return rec(root,None)
    key={v:canon(v) for v in G}; seen={}
    for v in sorted(G): seen.setdefault(key[v],[]).append(v)
    return list(seen.values())
def restricted_charpoly(C,B):
    """charpoly of C on the subspace spanned by the columns of B (generalized: det(B^T C B - t B^T B))."""
    G=B.T*B; M=B.T*C*B
    return sp.expand(sp.cancel((M-t*G).det()/G.det()))
l1b=json.load(open(RES/"explore_l1b_quotient_folds_n14.json"))
out=[]
for r in l1b:
    if r["golden_exact"]: continue
    n=r["n"]; e=[tuple(x) for x in r["edges"]]; G=nx.Graph(e); C=cart(n,e); orb=orbits(G)
    S=sp.Matrix.hstack(*[sp.Matrix([1 if v in c else 0 for v in range(n)]) for c in orb])
    Sperp=sp.Matrix.hstack(*S.T.nullspace())
    pS=restricted_charpoly(C,S); pP=restricted_charpoly(C,Sperp)
    fS=sp.factor(pS,extension=s5); fP=sp.factor(pP,extension=s5)
    goldP=[g for g in sp.Mul.make_args(fP) if g.has(s5)]
    # is the complement's golden part a Galois pair q*sigma(q)?
    bases=[sp.expand(g.as_base_exp()[0]) for g in goldP]
    paired=all(any(sp.expand(b.subs(s5,-s5)-b2)==0 for b2 in bases) for b in bases)
    # support of the complement's golden eigenvectors
    supp=set()
    for g in goldP:
        b=g.as_base_exp()[0]
        for lam in (sp.solve(b,t) if sp.Poly(b,t).degree()<=2 else []):
            for v in (C-lam*sp.eye(n)).nullspace():
                supp|={i for i in range(n) if sp.simplify(v[i])!=0}
    sub=G.subgraph(supp)
    print(f"det={r['det']} orbits={len(orb)} sizes={[len(c) for c in orb]}")
    print(f"   symmetric sector ({S.shape[1]}-dim): {fS}")
    print(f"   complement ({Sperp.shape[1]}-dim): {fP}")
    print(f"   complement golden part is a Galois pair: {paired}")
    print(f"   support of complement golden eigenvectors (linear/quadratic factors only): {sorted(supp)} -> induced subgraph edges {list(sub.edges())}, components {nx.number_connected_components(sub) if supp else 0}")
    out.append({"det":r["det"],"edges":e,"orbit_sizes":[len(c) for c in orb],"symmetric_charpoly":str(fS),"complement_charpoly":str(fP),"complement_golden_pair":paired,"support":sorted(supp),"support_edges":list(sub.edges())})
json.dump(out,open(RES/"explore_r9_mixed_trees_n14.json","w"),indent=1)
print("DONE")
