#!/usr/bin/env python3
"""explore_r17: the branched-double-cover construction. parent(D, bond): two sheets, trivial over
ordinary edges, cross-wired + one direct (defect) edge over the 5-bond. Checks:
(1) charpoly(parent) = q*sigma(q) for EVERY one-5 placement, k=2..7 (117 objects);
(2) on sample parents, the Bezout reflection equals the sheet form exactly:
    sqrt5*R diagonal +1/sqrt5 on sheet 0, -1/sqrt5 on sheet 1, off-diagonal 2/sqrt5 on sheet
    pairs, zero elsewhere — the matching form as a computation, not an observation."""
import sympy as sp, networkx as nx, time, sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import bezout_proj, simp, cart as _cart, t
s5=sp.sqrt(5); phi=(1+s5)/2; RES=Path(__file__).parent.parent/"results"
def sigma(M): return M.applyfunc(lambda x: sp.expand(x.subs(s5,-s5)))
def build_parent(E,pos,k):
    idx=lambda v,s: v+k*s; pe=[]
    for m,(a,b) in enumerate(E):
        if m==pos: pe+=[(idx(a,0),idx(b,0)),(idx(a,0),idx(b,1)),(idx(a,1),idx(b,0))]
        else: pe+=[(idx(a,0),idx(b,0)),(idx(a,1),idx(b,1))]
    return pe
t0=time.time(); tot=ok=0
samples=[]; PER_K=2; SAMPLE_K=(2,4,6,7)   # stratified: up to PER_K strict samples per k (audit fix 2026-09-02, HIFI-002)
K5=sp.QQ.algebraic_field(s5)
def is_strict(q): return sp.gcd(sp.Poly(q,t,domain=K5),sp.Poly(sp.expand(q.subs(s5,-s5)),t,domain=K5)).degree()==0
skipped_degenerate=0
for k in range(2,8):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr())
            pe=build_parent(E,pos,k); C=_cart(2*k,pe)
            p=sp.expand(C.charpoly(t).as_expr())
            tot+=1; hit=sp.expand(p-q*q.subs(s5,-s5))==0; ok+=hit
            if hit and k in SAMPLE_K and sum(1 for s in samples if s[0]==k)<PER_K:
                if is_strict(q): samples.append((k,pe,q))
                else: skipped_degenerate+=1
print(f"(1) charpoly identity: {ok}/{tot} placements, k=2..7 [{time.time()-t0:.0f}s]",flush=True)
form_ok=0
for k,pe,q in samples:
    n=2*k; C=_cart(n,pe)
    P=bezout_proj(C,q); R=simp(P-sigma(P)); S5R=simp(s5*R)   # exact: radsimp+expand, no nsimplify (HIFI-003)
    good=all(S5R[v,v]==(1 if v<k else -1) for v in range(n)) and \
         all(S5R[v,(v+k)%n if v<k else v-k]==2 for v in range(n)) and \
         all(S5R[v,w]==0 for v in range(n) for w in range(n) if w not in (v,(v+k) if v<k else v-k))
    print(f"   k={k}: sqrt5*R == S + 2*Pi (sheet form) exactly: {good}",flush=True)
    form_ok+=good
print(f"(2) sheet form verified on {form_ok}/{len(samples)} strict samples, per k: {sorted(set(s[0] for s in samples))}; degenerate placements passed over while sampling: {skipped_degenerate}",flush=True)
print("DONE",flush=True)
