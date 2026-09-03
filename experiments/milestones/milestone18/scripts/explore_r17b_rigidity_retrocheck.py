#!/usr/bin/env python3
"""explore_r17b: rigidity retro-check. For every known strict Galois fold at n<=16 (the 20 of r15
plus the cospectral twin), is the tree ISOMORPHIC to construction(D, bond) for some one-5
placement of its partner diagram? Also: which placement does each of the two n=16 twins match —
the same or different?"""
import json, sympy as sp, networkx as nx, sys
from pathlib import Path
t=sp.Symbol('t'); s5=sp.sqrt(5); phi=(1+s5)/2
RES=Path(__file__).parent.parent/"results"
def cart(n,e):
    C=2*sp.eye(n)
    for i,j in e: C[i,j]=C[j,i]=-1
    return C
def build_parent(E,pos,k):
    idx=lambda v,s: v+k*s; pe=[]
    for m,(a,b) in enumerate(E):
        if m==pos: pe+=[(idx(a,0),idx(b,0)),(idx(a,0),idx(b,1)),(idx(a,1),idx(b,0))]
        else: pe+=[(idx(a,0),idx(b,0)),(idx(a,1),idx(b,1))]
    return pe
partners={}
for k in range(2,9):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr())
            partners.setdefault(sp.expand(q*q.subs(s5,-s5)),[]).append((E,pos,k))
folds=[]; seen=set()
def add(n,e):
    key=(n,tuple(sorted(tuple(sorted(v)) for v in e)))
    if key not in seen: seen.add(key); folds.append((n,[tuple(v) for v in e]))
for x in json.load(open(RES/'explore_r15_matching_n12.json')): add(x["n"],x["edges"])
for x in json.load(open(RES/'explore_r16b_strict_n16.json')): add(16,x["edges"])   # exhaustive n=16 list incl. the cospectral twin (audit fix HIFI-001: no inlined edge list)
print(f"loaded {len(folds)} known strict folds at n<=16",flush=True)
tot=0; ok=0
for n,e in folds:
    p=sp.expand(cart(n,e).charpoly(t).as_expr()); G=nx.Graph(e)
    hits=[]
    for E,pos,k in partners.get(p,[]):
        H=nx.Graph(build_parent(E,pos,k))
        if nx.is_isomorphic(G,H): hits.append((sorted(map(list,E)),list(E[pos])))
    tot+=1; ok+=bool(hits)
    tag=f"n={n} isomorphic to construction: {bool(hits)}"
    if n==16 and len(partners.get(p,[]))>2: tag+=f"  placements matched: {[h[1] for h in hits]}"
    print(tag,flush=True)
print(f"\nRIGIDITY RETRO-CHECK: {ok}/{tot} known strict folds are construction parents",flush=True)
# twin pair placement comparison
# the twin pair = the cospectral pair among the n=16 folds (found, not assumed to be last in the list)
from collections import defaultdict
bypoly=defaultdict(list)
for n,e in folds:
    if n==16: bypoly[sp.expand(cart(16,e).charpoly(t).as_expr())].append((n,e))
pairs=[v for v in bypoly.values() if len(v)>1]
print(f"cospectral pairs at n=16: {len(pairs)}")
pair=pairs[0] if pairs else []; p_twin=sp.expand(cart(16,pair[0][1]).charpoly(t).as_expr()) if pair else None
print(f"twin-pair members found: {len(pair)}")
for n,e in pair:
    G=nx.Graph(e)
    for E,pos,k in partners.get(p_twin,[]):
        if nx.is_isomorphic(G,nx.Graph(build_parent(E,pos,k))):
            DG=nx.Graph(E); DG.remove_edge(*E[pos]); halves=sorted(len(c) for c in nx.connected_components(DG))
            print(f"  twin member matches placement {E[pos]} halves {halves}")
print("DONE",flush=True)
