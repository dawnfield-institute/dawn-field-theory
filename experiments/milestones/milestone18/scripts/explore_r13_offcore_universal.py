#!/usr/bin/env python3
"""explore_r13 (exploration): are the off-core laws UNIVERSAL? R_off = P_off - sigma(P_off) needs
no gauge, no conic, no multiplicity restriction — it exists for every core-grade Galois fold,
including all folds exp_12 declared out of domain (gauge-dependent cores, multiplicity > 2,
quadratic rational cores). For every Galois fold at n <= 14, checks:
 (i) off-ledger identity P_off + sigma(P_off) = I - Qc  (Qc = full rational-core projector,
     every root of every rational factor, extensions included)
 (ii) modulated vertex law sqrt5*R_off,vv = +-(1 - Qc_vv)
 (iii) trace law tr(R_off D) = 2/sqrt5
 (iv) sign-split structure: copy connected, cut 2, conjugate two components = diagram halves
 (v) off->off leakage block ||sigma(P_off) B P_off||^2 = 2/5"""
import sys, json, sympy as sp, networkx as nx
sys.argv=['x','14']
src=open(__file__.replace('explore_r13_offcore_universal.py','exp_12_part2_fold_laws.py')).read().split('res={"registration"')[0]
exec(src)
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
out=[]; tally={"i":0,"ii":0,"iii":0,"iv":0,"v":0,"total":0}
for r in golden:
    n=r["n"]; e=[tuple(x) for x in r["edges"]]; p=charpoly(n,e)
    if p not in diag or grade_of(r)!="core": continue
    q,M=diag[p]
    C=cart(n,e); rat=rational_core_factors(p); q_off=q
    for g in rat:
        b,ex=g.as_base_exp(); q_off=sp.cancel(q_off/b**(ex//2))
    P_off=bezout_proj(C,sp.expand(q_off))
    Qc=sp.zeros(n,n)
    for g in rat:
        b,ex=g.as_base_exp()
        for lam in sp.solve(b,t):
            V=sp.Matrix.hstack(*(C-lam*sp.eye(n)).nullspace()); Qc+=simp(V*(V.T*V).inv()*V.T)
    Qc=Qc.applyfunc(sp.radsimp); P_off=simp(P_off*(sp.eye(n)-Qc)); R_off=simp(P_off-sigma(P_off))
    ok_i=sp.simplify(P_off+sigma(P_off)-(sp.eye(n)-Qc))==sp.zeros(n,n)
    ok_ii=all(sp.simplify((sp.sqrt(5)*R_off[v,v])**2-(1-Qc[v,v])**2)==0 for v in range(n))
    G=nx.Graph(e); D=sp.diag(*[G.degree(v) for v in range(n)]); B=2*sp.eye(n)-D
    ok_iii=sp.simplify(sp.expand((R_off*D).trace())-2*sp.sqrt(5)/5)==0
    d=[rs(R_off[v,v]) for v in range(n)]
    copy=[v for v in range(n) if sp.re(d[v].evalf())>1e-12]; conj=[v for v in range(n) if sp.re(d[v].evalf())<-1e-12]
    cut=[x for x in e if (x[0] in copy)!=(x[1] in copy)]
    cc=nx.number_connected_components(G.subgraph(copy)) if copy else 0
    sizes=sorted(len(c) for c in nx.connected_components(G.subgraph(conj))) if conj else []
    E,pos=diag2[p]; DG=nx.Graph(E); DG.remove_edge(*E[pos]); halves=sorted(len(c) for c in nx.connected_components(DG))
    ok_iv=(cc==1 and len(cut)==2 and sizes==halves)
    ok_v=sp.simplify(frob2(simp(sigma(P_off)*B*P_off))-sp.Rational(2,5))==0
    core_desc=",".join(str(g) for g in rat)
    print(f"n={n} det={r['det']:>6} core[{core_desc}] i:{ok_i} ii:{ok_ii} iii:{ok_iii} iv:{ok_iv} v:{ok_v}",flush=True)
    out.append({"n":n,"det":r["det"],"edges":e,"core":core_desc,"i":ok_i,"ii":ok_ii,"iii":ok_iii,"iv":ok_iv,"v":ok_v,
                "conj_sizes":sizes,"halves":halves})
    tally["total"]+=1
    for k_,v_ in (("i",ok_i),("ii",ok_ii),("iii",ok_iii),("iv",ok_iv),("v",ok_v)): tally[k_]+=bool(v_)
print("TALLY:",tally,flush=True)
json.dump({"tally":tally,"rows":out},open(RES/'explore_r13_offcore_universal.json','w'),indent=1)
print("DONE",flush=True)
