#!/usr/bin/env python3
"""exp_17: Phase 7b (SEALED) — the Pi-asymmetry profile. For each strict fold: u_e = K_ab/tr K
(K = exp(-beta C), mpmath 40 digits); Pi-pairs {e, Pi(e)} of non-defect edges; asymmetry
a = |u_e - u_Pi(e)| / (u_e + u_Pi(e)); distance d from the pair to the defect; Spearman rho(a, d);
per-fold permutation null over reference edges. T1: folds with p<=0.10 vs Bin(18,0.1) 99% quantile
(5). T2: peak pair at distance <=1 vs Poisson-binomial null. T3 recorded."""
import json, sys, sympy as sp, networkx as nx, mpmath as mp
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
    sign={v:int(S5R[v,v]) for v in range(n)}
    E={tuple(sorted(x)) for x in e}
    d=[Ed for Ed in E if tuple(sorted((match[Ed[0]],match[Ed[1]]))) not in E]
    assert len(d)==1; return d[0], match, sign
def heat_u(n,e,beta):
    C=mp.matrix(n,n)
    for i in range(n): C[i,i]=2
    for a,b in e: C[a,b]=C[b,a]=-1
    w,V=mp.eigsy(C); K=V*mp.diag([mp.e**(-beta*w[i]) for i in range(n)])*V.T; Z=sum(K[i,i] for i in range(n))
    return {tuple(sorted((a,b))):K[a,b]/Z for a,b in e}
def avg_ranks(vals):
    order=sorted(range(len(vals)),key=lambda i:vals[i]); ranks=[0]*len(vals); i=0
    while i<len(order):
        j=i
        while j+1<len(order) and abs(vals[order[j+1]]-vals[order[i]])<=TOL: j+=1
        r=(i+j)/2+1
        for k2 in range(i,j+1): ranks[order[k2]]=r
        i=j+1
    return ranks
def spearman(x,y):
    rx=avg_ranks(x); ry=avg_ranks(y); n=len(x)
    mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry)); den=(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**0.5
    return num/den if den>0 else 0.0
def pair_distance(G,pair,ref):
    dist={}
    for v in set(pair[0])|set(pair[1]):
        dist[v]=min(nx.shortest_path_length(G,v,r) for r in ref)
    return min(dist.values())
def pb_quantile(ps,q=0.99):
    dist=[1.0]
    for p in ps:
        new=[0.0]*(len(dist)+1)
        for i,d in enumerate(dist): new[i]+=d*(1-p); new[i+1]+=d*p
        dist=new
    cdf=0
    for k,d in enumerate(dist):
        cdf+=d
        if cdf>=q: return k
    return len(dist)-1
folds=[]
for f in ('explore_r15_matching_n12.json','explore_r15_matching_n16.json'):
    for x in json.load(open(RES/f)): folds.append((x["n"],[tuple(v) for v in x["edges"]]))
folds.append((16,[(0,9),(1,0),(1,2),(1,6),(1,8),(2,3),(3,4),(4,5),(6,7),(9,10),(9,14),(10,11),(11,12),(12,13),(14,15)]))
# gate: E8 pairs
e8=[(i,i+1) for i in range(6)]+[(2,7)]; d8,m8,_=defect_from_R(8,e8); assert d8==(2,3), d8
rows=[]; T1c={str(b):0 for b in BETAS}; T2c={str(b):0 for b in BETAS}; T2ps=[]; inf=0
for n,e in folds:
    d0,match,sign=defect_from_R(n,e); G=nx.Graph(e); E={tuple(sorted(x)) for x in e}
    pairs=[]; seen=set()
    for ed in E:
        if ed==d0: continue
        im=tuple(sorted((match[ed[0]],match[ed[1]]))); key=frozenset((ed,im))
        if key in seen: continue
        seen.add(key); pairs.append((ed,im))
    informative=len(pairs)>=5; rec={"n":n,"edges":e,"defect":list(d0),"pairs":len(pairs),"informative":informative}
    if informative: inf+=1
    dists=[pair_distance(G,pr,d0) for pr in pairs]
    p_near=sum(1 for d in dists if d<=1)/len(pairs)
    if informative: T2ps.append(p_near)
    for beta in BETAS:
        u=heat_u(n,e,beta)
        a=[abs(u[x]-u[y])/(u[x]+u[y]) for x,y in pairs]
        rho=spearman(a,[mp.mpf(d) for d in dists])
        # permutation null over reference edges
        rhos=[]
        for ref in E:
            dr=[pair_distance(G,pr,ref) for pr in pairs]; rhos.append(spearman(a,[mp.mpf(d) for d in dr]))
        pval=sum(1 for r in rhos if r<=rho+1e-15)/len(rhos)
        peak=pairs[max(range(len(a)),key=lambda i:a[i])]; peak_d=pair_distance(G,peak,d0)
        # T3: which sheet carries more occupation (sheet 0 = both endpoints sign +1)
        sheet0=[x for x,y in pairs if sign[x[0]]==1 and sign[x[1]]==1]
        sgn=sum((u[x]-u[tuple(sorted((match[x[0]],match[x[1]])))]) for x in sheet0)
        rec[f"beta={beta}"]={"rho":float(rho),"p":pval,"peak_dist":peak_d,"sheet0_minus_sheet1":float(sgn)}
        if informative:
            T1c[str(beta)]+=(pval<=0.10); T2c[str(beta)]+=(peak_d<=1)
    b1=rec["beta=1.0"]
    print(f"n={n} defect {d0} pairs {len(pairs)} rho={b1['rho']:+.3f} p={b1['p']:.3f} peak_d={b1['peak_dist']} {'' if informative else '(uninformative)'}",flush=True)
    rows.append(rec)
q1=5; q2=pb_quantile(T2ps)
tests={"informative":inf,"T1_count":T1c["1.0"],"T1_quantile":q1,"T1":T1c["1.0"]>q1,
       "T2_count":T2c["1.0"],"T2_quantile":q2,"T2":T2c["1.0"]>q2,"T1_by_beta":T1c,"T2_by_beta":T2c}
json.dump({"registration":"phase7b","rows":rows,"tests":tests},open(RES/'exp_17_pi_asymmetry.json','w'),indent=1,default=str)
print("TESTS:",tests,flush=True); print("SCORE DONE",flush=True)
