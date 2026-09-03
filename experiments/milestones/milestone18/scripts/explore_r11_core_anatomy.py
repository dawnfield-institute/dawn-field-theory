#!/usr/bin/env python3
"""explore_r11 (exploration): what the core does to a fold. On the 12 registered-domain core folds:
(1) the OFF-CORE reflection R_off = P_off - sigma(P_off) (gauge-free): does its sign split restore
    the two-component + diagram-halves law on all 12, including the det -464 exception?
(2) leakage anatomy: (I-P)BP splits into four orthogonal blocks by left/right factor
    (off->off, core-conj->off, off->core, core-conj->core). Is off->off exactly 2/5 on all 12,
    with the 28/45 excess of dets -44/-284 confined to the core blocks?
Also records tr(R_off D) and the R_off diagonal values."""
import sys, json, sympy as sp, networkx as nx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from foldlaws import *   # promoted from exp_12_part2 on 2026-09-02
census, golden, diag = load_context(14)
rs=lambda x: sp.radsimp(sp.expand(x))
diag2={}
for k in range(2,8):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr()); diag2.setdefault(sp.expand(q*q.subs(s5,-s5)),(E,pos))
def frob2(X): return sp.nsimplify(sp.expand(sum(x**2 for x in X)))
out=[]
for r in golden:
    n=r["n"]; e=[tuple(x) for x in r["edges"]]; p=charpoly(n,e)
    if p not in diag or grade_of(r)!="core": continue
    q,M=diag[p]; Ps,C,standin=projector_for(n,e,q)
    if standin: continue
    invs=[invariants(P_,n,e) for P_ in Ps]
    if not all(sp.simplify(invs[0][i]-invs[-1][i])==0 for i in range(3)): continue   # domain only
    # rebuild the pieces
    rat=rational_core_factors(p); q_off=q
    for g in rat:
        b,ex=g.as_base_exp(); q_off=sp.cancel(q_off/b**(ex//2))
    P_off=bezout_proj(C,sp.expand(q_off))
    Qc=sp.zeros(n,n); lams=[]
    for g in rat:
        b,ex=g.as_base_exp()
        for lam in sp.solve(b,t):
            lams.append(lam); V=sp.Matrix.hstack(*(C-lam*sp.eye(n)).nullspace()); Qc+=simp(V*(V.T*V).inv()*V.T)
    P_off=simp(P_off*(sp.eye(n)-Qc))
    P=Ps[0]; P_v=simp(P-P_off)                    # the chosen golden core line(s)
    # (1) off-core reflection structure
    R_off=simp(P_off-sigma(P_off)); d=[rs(R_off[v,v]) for v in range(n)]
    vals=sorted(set(str(x) for x in d))
    copy=[v for v in range(n) if d[v].evalf()>0]; conj=[v for v in range(n) if d[v].evalf()<0]
    zero=[v for v in range(n) if d[v]==0]
    G=nx.Graph(e); cut=[x for x in e if (x[0] in copy)!=(x[1] in copy)]
    cc=nx.number_connected_components(G.subgraph(copy)) if copy else 0
    sizes=sorted(len(c) for c in nx.connected_components(G.subgraph(conj))) if conj else []
    E,pos=diag2[p]; DG=nx.Graph(E); DG.remove_edge(*E[pos]); halves=sorted(len(c) for c in nx.connected_components(DG))
    trRoffD=sp.nsimplify(sp.expand((R_off*sp.diag(*[G.degree(v) for v in range(n)])).trace()))
    # (2) leakage anatomy: orthogonal block decomposition of (I-P)B P
    A=sp.zeros(n,n)
    for i,j in e: A[i,j]=A[j,i]=1
    D=sp.diag(*[sum(A[i,k] for k in range(n)) for i in range(n)]); B=2*sp.eye(n)-D
    sPoff=simp(sigma(P_off)); Qconj=simp(Qc-P_v)
    blocks={"off->off":frob2(simp(sPoff*B*P_off)),"coreconj->off":frob2(simp(Qconj*B*P_off)),
            "off->core":frob2(simp(sPoff*B*P_v)),"coreconj->core":frob2(simp(Qconj*B*P_v))}
    total=frob2(simp((sp.eye(n)-P)*B*P))
    ok_sum=sp.simplify(sum(blocks.values())-total)==0
    print(f"n={n} det={r['det']:>5}  R_off diag values {vals}  zeros {len(zero)} (on core support)")
    print(f"    sign split: copy {len(copy)}v ({cc} comp) | conj comps {sizes} | cut {len(cut)} | halves {halves} match={sizes==halves}  tr(R_off D)={trRoffD}")
    print(f"    leak blocks: {{ {', '.join(f'{k}: {v}' for k,v in blocks.items())} }} total={total} sum-check={ok_sum}")
    out.append({"n":n,"det":r["det"],"edges":e,"Roff_diag_values":vals,"zeros":zero,"copy_comps":cc,
        "conj_sizes":sizes,"cut":len(cut),"halves":halves,"halves_match":sizes==halves,
        "trRoffD":str(trRoffD),"leak_blocks":{k:str(v) for k,v in blocks.items()},"leak_total":str(total),"sum_check":ok_sum})
json.dump(out,open(RES/'explore_r11_core_anatomy.json','w'),indent=1)
print("DONE")
