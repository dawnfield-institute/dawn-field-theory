#!/usr/bin/env python3
"""explore_r12 (exploration): the modulated vertex law. On every registered-domain core fold,
sqrt5 * R_off,vv = +-(1 - Qc_vv) at EVERY vertex (Qc = rational-core projector) — the strict
vertex law (|R_vv| = 1/sqrt5) is the Qc = 0 case. Core-mass values are uniform per fold:
1/2 (leaf-difference kernel pair) on ten folds, 1/3 (three-vertex kernel vectors) on exactly the
two folds whose mixed core blocks leak (dets -44, -284) — the leak selector is the kernel geometry."""
import sys, json, sympy as sp
sys.argv=['x','14']
src=open(__file__.replace('explore_r12_modulated_vertex_law.py','exp_12_part2_fold_laws.py')).read().split('res={"registration"')[0]
exec(src)
rs=lambda x: sp.radsimp(sp.expand(x))
out=[]; allok=True
for r in golden:
    n=r["n"]; e=[tuple(x) for x in r["edges"]]; p=charpoly(n,e)
    if p not in diag or grade_of(r)!="core": continue
    q,M=diag[p]; Ps,C,standin=projector_for(n,e,q)
    if standin: continue
    invs=[invariants(P_,n,e) for P_ in Ps]
    if not all(sp.simplify(invs[0][i]-invs[-1][i])==0 for i in range(3)): continue
    rat=rational_core_factors(p); q_off=q
    for g in rat:
        b,ex=g.as_base_exp(); q_off=sp.cancel(q_off/b**(ex//2))
    P_off=bezout_proj(C,sp.expand(q_off))
    Qc=sp.zeros(n,n)
    for g in rat:
        b,ex=g.as_base_exp()
        for lam in sp.solve(b,t):
            V=sp.Matrix.hstack(*(C-lam*sp.eye(n)).nullspace()); Qc+=simp(V*(V.T*V).inv()*V.T)
    P_off=simp(P_off*(sp.eye(n)-Qc)); R_off=simp(P_off-sigma(P_off))
    ok=all(sp.simplify((sp.sqrt(5)*R_off[v,v])**2-(1-Qc[v,v])**2)==0 for v in range(n))
    qvals=sorted(set(str(rs(Qc[v,v])) for v in range(n) if Qc[v,v]!=0))
    print(f"n={n} det={r['det']:>5}  identity holds at all vertices: {ok}  core-mass values: {qvals}")
    out.append({"n":n,"det":r["det"],"edges":e,"identity":ok,"core_mass_values":qvals,
                "Qc_diag":[str(rs(Qc[v,v])) for v in range(n)]})
    allok&=ok
print(f"IDENTITY sqrt5*R_off,vv = +-(1 - Qc_vv): {len(out)} folds, all pass: {allok}")
json.dump(out,open(RES/'explore_r12_modulated_vertex_law.json','w'),indent=1)
print("DONE")
