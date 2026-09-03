#!/usr/bin/env python3
"""exp_18: Phase 7c (SEALED 4ad88e40) — the branch fingerprint at n=20. 47 evaluable strict folds
(defects/matchings from r19). u_e = K_ab/tr K, K = exp(-beta C), mpmath 40 digits. T1: peak
asymmetry pair at distance 0 from the defect, count vs q0=22 (Poisson-binomial 99%). T2: profile
Spearman rho(a,d) vs per-fold permutation null over reference edges, count of p<=0.10 vs 10
(Bin(47,0.1) 99%). T3 recorded: sheet sign, beta robustness."""
import json, networkx as nx, mpmath as mp
from pathlib import Path
mp.mp.dps=40; TOL=mp.mpf('1e-25'); RES=Path(__file__).parent.parent/"results"
BETAS=[mp.mpf(1),mp.mpf('0.5'),mp.mpf(2)]; Q0=22; QB=10
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
    rx=avg_ranks(x); ry=avg_ranks(y); n=len(x); mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry)); den=(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**0.5
    return num/den if den>0 else 0.0
def pair_distance(G,pair,ref):
    return min(min(nx.shortest_path_length(G,v,x) for x in ref) for v in set(pair[0])|set(pair[1]))
folds=json.load(open(RES/'explore_r19_n20_defects.json'))
rows=[]; T1={str(b):0 for b in BETAS}; T2={str(b):0 for b in BETAS}; sheet={str(b):0 for b in BETAS}
for x in folds:
    n=20; e=[tuple(v) for v in x["edges"]]; G=nx.Graph(e); m={int(k):v for k,v in x["matching"].items()}
    sign={int(k):v for k,v in x["sign"].items()}; d0=tuple(x["defect"])
    E={tuple(sorted(v)) for v in e}; pairs=[]; seen=set()
    for ed in E:
        if ed==d0: continue
        im=tuple(sorted((m[ed[0]],m[ed[1]]))); key=frozenset((ed,im))
        if key in seen: continue
        seen.add(key); pairs.append((ed,im))
    dists=[pair_distance(G,pr,d0) for pr in pairs]
    rec={"det":x["det"],"defect":list(d0)}
    for beta in BETAS:
        u=heat_u(n,e,beta); a=[abs(u[p_]-u[q_])/(u[p_]+u[q_]) for p_,q_ in pairs]
        peak=max(range(len(a)),key=lambda i:a[i]); peak_d=dists[peak]
        rho=spearman(a,[mp.mpf(d) for d in dists])
        rhos=[spearman(a,[mp.mpf(pair_distance(G,pr,ref)) for pr in pairs]) for ref in E]
        pval=sum(1 for r_ in rhos if r_<=rho+1e-15)/len(rhos)
        sheet0=[p_ for p_,q_ in pairs if sign[p_[0]]==1 and sign[p_[1]]==1]
        sgn=sum(u[p_]-u[tuple(sorted((m[p_[0]],m[p_[1]])))] for p_ in sheet0)
        rec[f"beta={beta}"]={"rho":float(rho),"p":pval,"peak_dist":peak_d,"sheet0_minus_sheet1":float(sgn)}
        T1[str(beta)]+=(peak_d==0); T2[str(beta)]+=(pval<=0.10); sheet[str(beta)]+=(sgn>0)
    b1=rec["beta=1.0"]; rows.append(rec)
    print(f"det={x['det']:>6} peak_d={b1['peak_dist']} rho={b1['rho']:+.3f} p={b1['p']:.3f}",flush=True)
tests={"T1_count":T1["1.0"],"T1_q0":Q0,"T1":T1["1.0"]>Q0,"T2_count":T2["1.0"],"T2_q":QB,"T2":T2["1.0"]>QB,
       "T1_by_beta":T1,"T2_by_beta":T2,"sheet0_greater_by_beta":sheet,"folds":len(folds)}
json.dump({"registration":"4ad88e40","rows":rows,"tests":tests},open(RES/'exp_18_branch_fingerprint.json','w'),indent=1,default=str)
print("TESTS:",tests,flush=True); print("SCORE DONE",flush=True)
