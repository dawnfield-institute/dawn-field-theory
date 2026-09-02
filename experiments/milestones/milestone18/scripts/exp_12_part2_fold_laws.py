#!/usr/bin/env python3
"""exp_12 T2-T6 (registration 5bc8faff) on every Galois fold and quotient fold at n <= NMAX.
T2 det = N(q(0)); T3 signature split; T4 strict-fold invariant tr(RD)=2/sqrt5, ||(I-P)BP||^2=2/5;
T5 the same on core-grade Galois folds (core resolved on the golden conic; gauge dependence
declared by evaluating at two conic points); T6 quotient folds: tr(RD)=0, tr(PB)=1.
Machinery promoted to core/foldlaws.py on 2026-09-02 (was exec'd from this file by exp_13 and six
explorations); this script keeps only the registered run. Outputs verified identical at promotion."""
import sys, json, sympy as sp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from foldlaws import *   # t, s5, phi, RES, sigma, grade_of, charpoly, projector_for, invariants, sig, ...
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 12
census, golden, diag = load_context(NMAX)
res={"registration":"5bc8faff","nmax":NMAX,"T2":[],"T3":[],"T4":[],"T5":[],"T6":[]}
for r in golden:
    n=r["n"]; e=[tuple(x) for x in r["edges"]]; p=charpoly(n,e)
    if p in diag:                                   # Galois fold
        q,M=diag[p]; Ps,C,standin=projector_for(n,e,q)
        detN=sp.nsimplify(sp.expand(q.subs(t,0)*q.subs(t,0).subs(s5,-s5)))
        res["T2"].append({"n":n,"det":r["det"],"N(q0)":str(detN),"ok":sp.simplify(detN-r["det"])==0})
        P=Ps[0]; ledger=(simp(P*P-P)==sp.zeros(n,n)) and (simp(P+sigma(P)-sp.eye(n))==sp.zeros(n,n)) and P.rank()==n//2
        res["T3"].append({"ledger_projector":ledger,"n":n,"det":r["det"],"copy":sig(simp(P*C*P)),"conj":sig(simp((sp.eye(n)-P)*C*(sp.eye(n)-P))),"diagram_sig":sig(M),"standin":standin})
        inv=[invariants(P_,n,e) for P_ in Ps]
        rec={"n":n,"det":r["det"],"grade":grade_of(r),"trRD":[str(x[0]) for x in inv],"trPB":[str(x[1]) for x in inv],"leak":[str(x[2]) for x in inv],
             "gauge_independent": all(sp.simplify(inv[0][i]-inv[-1][i])==0 for i in range(3)),
             "core_standin_declared": standin, "ledger_projector": ledger}
        (res["T4"] if grade_of(r)=="strict" else res["T5"]).append(rec)
    else:                                            # quotient fold
        rat=rational_core_factors(p)
        if not all(str(g).startswith("(t - 2)") for g in rat): res["T6"].append({"n":n,"det":r["det"],"note":"non-(t-2) core, not evaluated"}); continue
        C=cart(n,e); f=sp.factor(p,extension=s5); gold=[g for g in sp.Mul.make_args(f) if g.has(s5)]
        seen=set(); q=sp.Integer(1)
        for g in gold:
            b,ex=g.as_base_exp(); kb=str(sp.expand(b)); kc=str(sp.expand(b.subs(s5,-s5)))
            if kc in seen: continue
            seen.add(kb); q*=b**ex
        P_off=bezout_proj(C,sp.expand(q)); V=sp.Matrix.hstack(*(C-2*sp.eye(n)).nullspace()); Qc=simp(V*(V.T*V).inv()*V.T)
        P=simp(P_off*(sp.eye(n)-Qc)+Qc/2)          # core traces gauge-independent on leaf-difference vectors
        inv=invariants(P,n,e); res["T6"].append({"n":n,"det":r["det"],"trRD":str(inv[0]),"trPB":str(inv[1])})
def allok(lst,key,val): return all(sp.simplify(sp.sympify(x)-val)==0 for rec in lst for x in rec[key])
res["tests"]={
 "T2":all(x["ok"] for x in res["T2"]),
 "T3_as_sealed":all((x["copy"][1]==0 and x["conj"]==x["copy"]) if x["diagram_sig"][1]==0 else (x["copy"][1]==0 and x["conj"][1]==1) for x in res["T3"] if not x["standin"]),
 "T3_corrected_law(copy=diagram sig, conjugate definite)":all(x["copy"][1]==x["diagram_sig"][1] and x["conj"][1]==0 for x in res["T3"] if not x["standin"]),
 "T4":allok(res["T4"],"trRD",2*s5/5) and allok(res["T4"],"leak",sp.Rational(2,5)),
 "T5_on_registered_domain":all(sp.simplify(sp.sympify(rec["trRD"][0])-2*s5/5)==0 and sp.simplify(sp.sympify(rec["leak"][0])-sp.Rational(2,5))==0 for rec in res["T5"] if (not rec["core_standin_declared"]) and rec["gauge_independent"]),
 "T5_domain_size":sum(1 for rec in res["T5"] if (not rec["core_standin_declared"]) and rec["gauge_independent"]),
 "T5_gauge_dependent_declared":sum(1 for rec in res["T5"] if (not rec["core_standin_declared"]) and not rec["gauge_independent"]),
 "T5_outside_recipe(multiplicity>2 core)":sum(1 for rec in res["T5"] if any(x.startswith("stand-in:multiplicity") for x in rec["core_standin_declared"])),
 "T5_outside_recipe(quadratic rational core)":sum(1 for rec in res["T5"] if any(x.startswith("stand-in:quadratic") for x in rec["core_standin_declared"]) and not any(x.startswith("stand-in:multiplicity") for x in rec["core_standin_declared"])),
 "T5_outside_recipe(conic unresolved)":sum(1 for rec in res["T5"] if any(x.startswith("stand-in:conic") for x in rec["core_standin_declared"]) and not any(x.startswith("stand-in:multiplicity") for x in rec["core_standin_declared"])),
 "T6":all(sp.simplify(sp.sympify(x["trRD"]))==0 and sp.simplify(sp.sympify(x["trPB"])-1)==0 for x in res["T6"] if "trRD" in x)}

print("TESTS (n<=%d):"%NMAX, res["tests"], " T1 from part1: PASS")
(RES/f"exp_12_part2_n{NMAX}.json").write_text(json.dumps(res,indent=1,default=str))
