#!/usr/bin/env python3
"""exp_14: Phase 5 (SEALED) — the off-core instrument and the selector, out of sample, at n=16.
Folds: the 80 partnered core-grade Galois folds from exp_13's census. Per fold: P_off, Qc, R_off
by construction; Lemma-1 identity checked before evaluation. T1 Clean <=> (B-invariant OR uniform
mass); T2 selector => trace; T3 selector => structure (some-partner halves); T4 mixed blocks
recorded; T5 layer order (Clean => trace AND structure)."""
import json, sys, sympy as sp, networkx as nx
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import bezout_proj, simp, cart as _cart
t=sp.Symbol('t'); s5=sp.sqrt(5); phi=(1+s5)/2; RES=Path(__file__).parent.parent/"results"
def cart(n,e): return _cart(n,[tuple(x) for x in e])
def sigma(M): return M.applyfunc(lambda x: sp.expand(x.subs(s5,-s5)))
rs=lambda x: sp.radsimp(sp.expand(x))
def frob2(X): return sp.nsimplify(sp.expand(sum(x**2 for x in X)))
# all one-5 partners at k=8, multi-map: p -> list of (q, edges, pos)
partners={}
for T in nx.nonisomorphic_trees(8):
    E=list(T.edges())
    for pos in range(len(E)):
        M=2*sp.eye(8)
        for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
        q=sp.expand(M.charpoly(t).as_expr()); tgt=sp.expand(q*q.subs(s5,-s5))
        partners.setdefault(tgt,[]).append((q,E,pos))
r13=json.load(open(RES/'exp_13_n16.json'))
folds=[x["edges"] for x in r13["T5"]]
print(f"{len(folds)} partnered core folds",flush=True)
out=[]; n=16
for e in folds:
    e=[tuple(v) for v in e]; C=cart(n,e); p=sp.expand(C.charpoly(t).as_expr())
    plist=partners[p]
    qs=[]
    for q,_,_ in plist:
        if not any(sp.expand(q-q2)==0 for q2 in qs): qs.append(q)
    q=qs[0]  # distinct q groupings beyond one are declared
    fQ=sp.factor(p); rat=[g for g in sp.Mul.make_args(sp.factor(p,extension=s5)) if g.has(t) and not g.has(s5)]
    q_off=q
    for g in rat:
        b,ex=g.as_base_exp(); q_off=sp.cancel(q_off/b**(ex//2))
    P_off=bezout_proj(C,sp.expand(q_off))
    Qc=sp.zeros(n,n)
    for g in rat:
        b,ex=g.as_base_exp()
        for lam in sp.solve(b,t):
            V=sp.Matrix.hstack(*(C-lam*sp.eye(n)).nullspace()); Qc+=simp(V*(V.T*V).inv()*V.T)
    Qc=Qc.applyfunc(sp.radsimp); P_off=simp(P_off*(sp.eye(n)-Qc)); R_off=simp(P_off-sigma(P_off))
    lem1=(simp(P_off+sigma(P_off)-(sp.eye(n)-Qc))==sp.zeros(n,n)) and (simp(P_off*P_off-P_off)==sp.zeros(n,n))
    G=nx.Graph(e); D=sp.diag(*[G.degree(v) for v in range(n)]); B=2*sp.eye(n)-D
    binv=simp(B*Qc-Qc*B)==sp.zeros(n,n)
    supp=[v for v in range(n) if Qc[v,v]!=0]
    uni=len(set(str(rs(Qc[v,v])) for v in supp))==1 if supp else True
    sel=binv or uni
    vertex=all(sp.simplify((s5*R_off[v,v])**2-(1-Qc[v,v])**2)==0 for v in range(n))
    leak_oo=frob2(simp(sigma(P_off)*B*P_off))
    clean=vertex and sp.simplify(leak_oo-sp.Rational(2,5))==0
    trace=sp.simplify(sp.expand((R_off*D).trace())-2*s5/5)==0
    d=[rs(R_off[v,v]) for v in range(n)]
    zeros=[v for v in range(n) if d[v]==0]
    copy=[v for v in range(n) if sp.re(d[v].evalf())>1e-12]; conj=[v for v in range(n) if sp.re(d[v].evalf())<-1e-12]
    cut=[x for x in e if (x[0] in copy)!=(x[1] in copy)]
    cc=nx.number_connected_components(G.subgraph(copy)) if copy else 0
    sizes=sorted(len(c) for c in nx.connected_components(G.subgraph(conj))) if conj else []
    halves_ok=False
    for _,E2,pos2 in plist:
        DG=nx.Graph(E2); DG.remove_edge(*E2[pos2])
        if sizes==sorted(len(c) for c in nx.connected_components(DG)): halves_ok=True; break
    struct=(cc==1 and len(cut)==2 and halves_ok and not zeros)
    out.append({"edges":e,"det":int(C.det()),"lem1":lem1,"binv":binv,"uniform":uni,"selector":sel,
        "vertex":vertex,"leak_oo":str(leak_oo),"clean":clean,"trace":trace,"struct":struct,
        "zeros":zeros,"q_groupings":len(qs)})
    print(f"det={out[-1]['det']:>7} lem1:{int(lem1)} sel:{int(sel)}(B:{int(binv)} U:{int(uni)}) clean:{int(clean)} trace:{int(trace)} struct:{int(struct)}",flush=True)
tests={"lemma1_all":all(r["lem1"] for r in out),
 "T1_selector_iff_clean":all(r["clean"]==r["selector"] for r in out),
 "T2_selector_implies_trace":all(r["trace"] for r in out if r["selector"]),
 "T3_selector_implies_structure":all(r["struct"] for r in out if r["selector"]),
 "T5_layer_order":all((r["trace"] and r["struct"]) for r in out if r["clean"]),
 "clean_count":sum(r["clean"] for r in out),"selector_count":sum(r["selector"] for r in out),
 "multi_grouping_declared":sum(1 for r in out if r["q_groupings"]>1)}
json.dump({"registration":"phase5","rows":out,"tests":tests},open(RES/'exp_14_n16.json','w'),indent=1,default=str)
print("TESTS:",tests,flush=True); print("SCORE DONE",flush=True)
