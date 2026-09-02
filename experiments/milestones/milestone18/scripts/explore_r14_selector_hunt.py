#!/usr/bin/env python3
"""explore_r14 (exploration): what selects the clean off-core regime? For all 61 core-grade Galois
folds at n<=14, compute structural features of the core and correlate with r13's law flags:
 F1 B-invariance of the core space (B*Qc == Qc*B, B = 2I - D)
 F2 D-invariance (D*Qc == Qc*D)  [equivalent to F1; recorded for the check]
 F3 core support: size, induced components, contains-a-leaf
 F4 Qc diagonal: distinct mass values, uniform-on-support
 F5 per-eigenvalue kernel dimensions
Also records the actual tr(R_off D) when the trace law fails (deviation spectrum)."""
import sys, json, sympy as sp, networkx as nx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from foldlaws import *   # promoted from exp_12_part2 on 2026-09-02
census, golden, diag = load_context(14)
rs=lambda x: sp.radsimp(sp.expand(x))
r13={ (r["n"],str(r["edges"])): r for r in json.load(open(RES/'explore_r13_offcore_universal.json'))["rows"] }
out=[]
for r in golden:
    n=r["n"]; e=[tuple(x) for x in r["edges"]]; p=charpoly(n,e)
    if p not in diag or grade_of(r)!="core": continue
    key=(n,str(r["edges"])); flags=r13.get(key) or r13.get((n,str([list(x) for x in e])))
    if flags is None: continue
    q,M=diag[p]; C=cart(n,e); rat=rational_core_factors(p); q_off=q
    for g in rat:
        b,ex=g.as_base_exp(); q_off=sp.cancel(q_off/b**(ex//2))
    P_off=bezout_proj(C,sp.expand(q_off))
    Qc=sp.zeros(n,n); kdims={}
    for g in rat:
        b,ex=g.as_base_exp()
        for lam in sp.solve(b,t):
            ns=(C-lam*sp.eye(n)).nullspace(); kdims[str(lam)]=len(ns)
            V=sp.Matrix.hstack(*ns); Qc+=simp(V*(V.T*V).inv()*V.T)
    Qc=Qc.applyfunc(sp.radsimp)
    G=nx.Graph(e); D=sp.diag(*[G.degree(v) for v in range(n)]); B=2*sp.eye(n)-D
    F1=simp(B*Qc-Qc*B)==sp.zeros(n,n); F2=simp(D*Qc-Qc*D)==sp.zeros(n,n)
    supp=[v for v in range(n) if Qc[v,v]!=0]
    sub=G.subgraph(supp); comps=nx.number_connected_components(sub) if supp else 0
    leaves=[v for v in supp if G.degree(v)==1]
    masses=sorted(set(str(rs(Qc[v,v])) for v in supp))
    P_off2=simp(P_off*(sp.eye(n)-Qc)); R_off=simp(P_off2-sigma(P_off2))
    trv=sp.nsimplify(sp.expand((R_off*D).trace()))
    out.append({"n":n,"det":r["det"],"edges":e,"core":",".join(str(g) for g in rat),
        "F1_Binv":F1,"F2_Dinv":F2,"supp":len(supp),"supp_comps":comps,"supp_leaves":len(leaves),
        "masses":masses,"kdims":kdims,"trRoffD":str(trv),
        "ii":flags["ii"],"iii":flags["iii"],"iv":flags["iv"],"v":flags["v"]})
    print(f"n={n} det={r['det']:>6} F1:{int(F1)} supp {len(supp)}({comps}c,{len(leaves)}lf) masses {masses} ii:{int(flags['ii'])} iii:{int(flags['iii'])} iv:{int(flags['iv'])}",flush=True)
json.dump(out,open(RES/'explore_r14_selector_hunt.json','w'),indent=1)
# correlation table
def cont(feat,lawkey):
    a=sum(1 for r in out if feat(r) and r[lawkey]); b=sum(1 for r in out if feat(r) and not r[lawkey])
    c=sum(1 for r in out if not feat(r) and r[lawkey]); d=sum(1 for r in out if not feat(r) and not r[lawkey])
    return f"{a:>2}/{b:<2} vs {c:>2}/{d:<2}"
print("\nfeature -> law contingency (feature-true pass/fail vs feature-false pass/fail):")
for name,f in [("F1 B-invariant core",lambda r:r["F1_Binv"]),("supp connected",lambda r:r["supp_comps"]<=1),
               ("all-leaf support",lambda r:r["supp_leaves"]==r["supp"]),("uniform mass",lambda r:len(r["masses"])==1),
               ("mass includes 1/2",lambda r:"1/2" in r["masses"])]:
    print(f"  {name:>22}: ii {cont(f,'ii')} | iii {cont(f,'iii')} | iv {cont(f,'iv')}")
print("DONE",flush=True)
