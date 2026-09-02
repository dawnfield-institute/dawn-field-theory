#!/usr/bin/env python3
"""exp_16: Phase 7 (SEALED) — does heat-kernel dynamics see the reinjection port? Objects: the 21
strict folds at n<=16; defect edge found from R (S+2Pi), not from the construction. Observable:
bond occupation u_e(beta)=K_ab/tr K with K=exp(-beta C), mpmath 40 digits. Ranks within
degree-matched classes of exact edge orbits. T1: distinguished count vs 99% Poisson-binomial
quantile. T2: null calibration on seeded degree-matched random trees with designated edges.
T3/T4 recorded."""
import json, sys, random, sympy as sp, networkx as nx, mpmath as mp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import cart, simp, sigma, bezout_proj, t
mp.mp.dps=40; TOL=mp.mpf('1e-25')
s5=sp.sqrt(5); phi=(1+s5)/2; RES=Path(__file__).parent.parent/"results"
BETAS=[mp.mpf(1),mp.mpf('0.5'),mp.mpf(2)]
partners={}
for k in range(2,9):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr()); partners.setdefault(str(sp.expand(q*q.subs(s5,-s5))),q)
def defect_from_R(n,e):
    C=cart(n,e); q=partners[str(sp.expand(C.charpoly(t).as_expr()))]
    R=simp(bezout_proj(C,q)); R=simp(R-sigma(R)); S5R=simp(s5*R).applyfunc(sp.nsimplify)
    match={v:[w for w in range(n) if w!=v and S5R[v,w]!=0][0] for v in range(n)}
    E={tuple(sorted(x)) for x in e}
    d=[Ed for Ed in E if tuple(sorted((match[Ed[0]],match[Ed[1]]))) not in E]
    assert len(d)==1; return d[0], match
def heat_u(n,e,beta):
    C=mp.matrix(n,n)
    for i in range(n): C[i,i]=2
    for a,b in e: C[a,b]=C[b,a]=-1
    w,V=mp.eigsy(C); ex=[mp.e**(-beta*w[i]) for i in range(n)]
    K=V*mp.diag(ex)*V.T; Z=sum(K[i,i] for i in range(n))
    return {tuple(sorted((a,b))):K[a,b]/Z for a,b in e}
def canon(G,root,parent):
    return "("+"".join(sorted(canon(G,w,root) for w in G[root] if w!=parent))+")"
def edge_orbits(G):
    key={}
    for a,b in G.edges():
        H=G.copy(); H.remove_edge(a,b)
        ca=canon(H,a,None); cb=canon(H,b,None)
        key[tuple(sorted((a,b)))]=tuple(sorted((ca,cb)))
    orb={}
    for ed,kk in key.items(): orb.setdefault(kk,[]).append(ed)
    return list(orb.values())
def analyse(G,u,target):
    """returns (m, rank info, distinguished) for the target edge's matched class; None if uninformative."""
    deg=dict(G.degree()); orbs=edge_orbits(G)
    # orbit-invariance gate
    for o in orbs:
        vals=[u[ed] for ed in o]
        if max(vals)-min(vals)>TOL: raise RuntimeError("orbit invariance violated")
    reps={}
    for o in orbs:
        ed=o[0]; k=tuple(sorted((deg[ed[0]],deg[ed[1]]))); reps.setdefault(k,[]).append((o,u[ed]))
    tk=tuple(sorted((deg[target[0]],deg[target[1]])))
    cls=reps[tk]; m=len(cls)
    if m<3: return {"m":m,"informative":False}
    tv=[v for o,v in cls if target in o][0]
    others=[v for o,v in cls if target not in o]
    ismax=all(tv-v>TOL for v in others); ismin=all(v-tv>TOL for v in others)
    tie=any(abs(tv-v)<=TOL for v in others)
    return {"m":m,"informative":True,"distinguished":ismax or ismin,"max":ismax,"min":ismin,"tie":tie,
            "rank":1+sum(1 for v in others if v>tv+TOL)}
def pb_quantile(ps,q=0.99):
    dist=[mp.mpf(1)]
    for p in ps:
        new=[mp.mpf(0)]*(len(dist)+1)
        for i,d in enumerate(dist): new[i]+=d*(1-p); new[i+1]+=d*p
        dist=new
    cdf=mp.mpf(0)
    for k,d in enumerate(dist):
        cdf+=d
        if cdf>=q: return k
    return len(dist)-1
def prufer_tree(degseq,seed):
    rng=random.Random(seed); seq=[]
    for v,d in enumerate(degseq): seq+=[v]*(d-1)
    rng.shuffle(seq); n=len(degseq)
    deg=[1]*n
    for v in seq: deg[v]+=1
    edges=[]
    for v in seq:
        leaf=min(i for i in range(n) if deg[i]==1); edges.append((leaf,v)); deg[leaf]-=1; deg[v]-=1
    rest=[i for i in range(n) if deg[i]==1]; edges.append((rest[0],rest[1]))
    return edges
folds=[]
for f in ('explore_r15_matching_n12.json','explore_r15_matching_n16.json'):
    for x in json.load(open(RES/f)): folds.append((x["n"],[tuple(v) for v in x["edges"]]))
folds.append((16,[(0,9),(1,0),(1,2),(1,6),(1,8),(2,3),(3,4),(4,5),(6,7),(9,10),(9,14),(10,11),(11,12),(12,13),(14,15)]))
# instrument gates
e8=[(i,i+1) for i in range(6)]+[(2,7)]; G8=nx.Graph(e8); u8=heat_u(8,e8,mp.mpf(1))
for o in edge_orbits(G8): assert max(u8[x] for x in o)-min(u8[x] for x in o)<=TOL
a4=[(0,1),(1,2),(2,3)]; C4=mp.matrix([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]]); w4,_=mp.eigsy(C4)
Z4=sum(mp.e**(-w4[i]) for i in range(4)); u4=heat_u(4,a4,mp.mpf(1))
# real gate: sum over edges of K_ab*(-C_ab) equals tr(A K) = tr((2I-C)K) = 2 tr K - tr(C K); check via eigenvalues
lhs=sum(u4.values())*Z4; rhs=(2*Z4-sum(w4[i]*mp.e**(-w4[i]) for i in range(4)))/2
assert abs(lhs-rhs)<=mp.mpf('1e-30'), "A4 edge-sum identity failed"
print("gates: E8 orbit invariance ok; A4 edge-sum identity ok",flush=True)
rows=[]; ps=[]; count=0; ties=0; unin=0
for n,e in folds:
    d0,match=defect_from_R(n,e); G=nx.Graph(e)
    rec={"n":n,"edges":e,"defect":list(d0)}
    for beta in BETAS:
        u=heat_u(n,e,beta); a=analyse(G,u,d0); rec[f"beta={beta}"]=a
        if beta==1:
            if a["informative"]:
                ps.append(mp.mpf(2)/a["m"]); count+=a["distinguished"]; ties+=a["tie"]
            else: unin+=1
    # cut edges (the other two lifts) recorded
    rows.append(rec); print(f"n={n} defect {d0} m={rec['beta=1.0']['m']} " + (f"rank {rec['beta=1.0']['rank']}/{rec['beta=1.0']['m']} distinguished={rec['beta=1.0']['distinguished']}" if rec['beta=1.0']['informative'] else "uninformative"),flush=True)
q99=pb_quantile(ps); T1=count>q99
print(f"T1: distinguished {count}/{len(ps)} informative (uninformative {unin}, ties {ties}); 99% null quantile {q99}; PASS={T1}",flush=True)
# T2: null calibration on degree-matched random trees with designated edges
cps=[]; ccount=0; cun=0
for i,(n,e) in enumerate(folds):
    degseq=[d for _,d in sorted(nx.Graph(e).degree())]
    for j in range(5):
        ce=prufer_tree(degseq,seed=1000*i+j); Gc=nx.Graph(ce)
        rng=random.Random(7000*i+j); des=tuple(sorted(rng.choice(ce)))
        uc=heat_u(n,ce,mp.mpf(1)); a=analyse(Gc,uc,des)
        if a["informative"]: cps.append(mp.mpf(2)/a["m"]); ccount+=a["distinguished"]
        else: cun+=1
cq=pb_quantile(cps); T2=ccount<=cq
print(f"T2 (null calibration): controls distinguished {ccount}/{len(cps)} (uninformative {cun}); 99% quantile {cq}; PASS={T2}",flush=True)
res={"registration":"phase7","folds":rows,"T1":{"count":count,"informative":len(ps),"quantile99":q99,"pass":T1,"ties":ties},
     "T2":{"count":ccount,"informative":len(cps),"quantile99":cq,"pass":T2},
     "T4_beta_robustness":{str(b):sum(1 for r in rows if r[f"beta={b}"]["informative"] and r[f"beta={b}"]["distinguished"]) for b in BETAS}}
json.dump(res,open(RES/'exp_16_defect_dynamics.json','w'),indent=1,default=str)
print("T4 (beta robustness) distinguished counts:",res["T4_beta_robustness"],flush=True); print("SCORE DONE",flush=True)
