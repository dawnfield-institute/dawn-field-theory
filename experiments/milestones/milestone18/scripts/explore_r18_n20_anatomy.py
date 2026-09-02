#!/usr/bin/env python3
"""explore_r18: anatomy of the n=20 strict census (exp_15). (1) The four ASYMMETRIC strict trees
(trivial Aut, no one-5 partner): do their polynomials match 10-node tree diagrams with TWO golden
bonds (weights in {-phi, -1/phi})? (2) The six degenerate-partner trees: automorphism orbit counts
and sector status. (3) Full rigidity at 20: is every partnered strict tree graph-isomorphic to a
construction parent for some placement, and do cospectral twins realize distinct placements?"""
import json, sympy as sp, networkx as nx, itertools, time
from pathlib import Path
t=sp.Symbol('t'); s5=sp.sqrt(5); phi=(1+s5)/2; RES=Path(__file__).parent.parent/"results"
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
def orbits(G):
    def canon(root,parent): return "("+"".join(sorted(canon(w,root) for w in G[root] if w!=parent))+")"
    return len(set(canon(v,None) for v in G))
r=json.load(open(RES/'exp_15_n20.json'))
t0=time.time()
# partner map k=10 with placements
partners={}
for T in nx.nonisomorphic_trees(10):
    E=list(T.edges())
    for pos in range(len(E)):
        M=2*sp.eye(10)
        for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
        q=sp.expand(M.charpoly(t).as_expr()); partners.setdefault(str(sp.expand(q*q.subs(s5,-s5))),[]).append((E,pos))
print(f"one-5 map: {len(partners)} targets [{time.time()-t0:.0f}s]",flush=True)
# (1) two-bond map: weights (-phi,-phi) and (-phi,-1/phi) on two distinct edges
asym=[x for x in r["T2"] if not x["ok"]]
asym_p={str(sp.expand(cart(20,[tuple(v) for v in x["edges"]]).charpoly(t).as_expr())):x for x in asym}
hits={}
for T in nx.nonisomorphic_trees(10):
    E=list(T.edges())
    for i,j in itertools.combinations(range(len(E)),2):
        for w2 in (-phi, -1/phi):
            M=2*sp.eye(10)
            for m,(a,b) in enumerate(E): M[a,b]=M[b,a]=(-phi if m==i else (w2 if m==j else -1))
            q=sp.expand(M.charpoly(t).as_expr()); key=str(sp.expand(q*q.subs(s5,-s5)))
            if key in asym_p: hits.setdefault(key,[]).append((sorted(map(list,E)),E[i],E[j],str(w2)))
print(f"(1) two-bond search [{time.time()-t0:.0f}s]: asymmetric trees matched: {len(hits)}/{len(asym_p)}",flush=True)
for key,x in asym_p.items():
    G=nx.Graph([tuple(v) for v in x["edges"]]); print(f"    det={int(cart(20,[tuple(v) for v in x['edges']]).det())}: two-bond partners {len(hits.get(key,[]))}" + (f" e.g. bonds {hits[key][0][1]},{hits[key][0][2]} weights (-phi,{hits[key][0][3]})" if key in hits else ""))
# (2) degenerate partners
print("(2) degenerate-partner trees:")
for x in [y for y in r["T3"] if y.get("degenerate_partner")]:
    e=[tuple(v) for v in x["edges"]]; G=nx.Graph(e)
    print(f"    det={int(cart(20,e).det())} orbits={orbits(G)}/20 sector_strict={x['sector_strict']}",flush=True)
# (3) rigidity at 20 + twin placements
part=[x for x in r["T3"] if x.get("ok") is not None]
iso_ok=0; byp={}
for x in part:
    e=[tuple(v) for v in x["edges"]]; G=nx.Graph(e); p=str(sp.expand(cart(20,e).charpoly(t).as_expr()))
    matched=[]
    for E,pos in partners[p]:
        if nx.is_isomorphic(G,nx.Graph(build_parent(E,pos,10))):
            DG=nx.Graph(E); DG.remove_edge(*E[pos]); matched.append(sorted(len(c) for c in nx.connected_components(DG)))
    iso_ok+=bool(matched); byp.setdefault(p,[]).append(matched[0] if matched else None)
print(f"(3) rigidity at 20: {iso_ok}/{len(part)} evaluable partnered strict trees are construction parents [{time.time()-t0:.0f}s]",flush=True)
twins={p:v for p,v in byp.items() if len(v)>1}
print(f"    cospectral twin pairs among evaluable: {len(twins)}; placements (halves): {list(twins.values())}",flush=True)
json.dump({"two_bond_hits":{k:v for k,v in hits.items()},"rigidity":[iso_ok,len(part)],"twins":list(twins.values())},open(RES/'explore_r18_n20_anatomy.json','w'),indent=1,default=str)
print("DONE",flush=True)
