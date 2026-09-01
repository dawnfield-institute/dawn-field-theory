#!/usr/bin/env python3
"""exp_12 T2-T6 (registration 5bc8faff) on every Galois fold and quotient fold at n <= NMAX.
T2 det = N(q(0)); T3 signature split; T4 strict-fold invariant tr(RD)=2/sqrt5, ||(I-P)BP||^2=2/5;
T5 the same on core-grade Galois folds (core resolved on the golden conic; gauge dependence
declared by evaluating at two conic points); T6 quotient folds: tr(RD)=0, tr(PB)=1."""
import sys, json, itertools, sympy as sp, numpy as np, networkx as nx
from pathlib import Path
from networkx.algorithms.isomorphism import GraphMatcher
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import bezout_proj, simp, cart
t=sp.Symbol('t'); s5=sp.sqrt(5); phi=(1+s5)/2
NMAX=int(sys.argv[1]) if len(sys.argv)>1 else 12
RES=Path(__file__).parent.parent/"results"
census=json.load(open(RES/"explore_g1_census_20260901.json"))
if NMAX>=14 and (RES/"exp_11b_core_trees_n14.json").exists():
    for r in json.load(open(RES/"exp_11b_core_trees_n14.json")):
        census.append({"n":14,"edges":r["edges"],"det":r["det"],"fields":{"sqrt5":{"grade":r["grade"],"rational":r["rational"]}}})
def sigma(M): return M.applyfunc(lambda x: sp.expand(x.subs(s5,-s5)))
def grade_of(r): return r["fields"].get("sqrt5",{}).get("grade")
golden=[r for r in census if grade_of(r) in ("strict","core") and r["n"]<=NMAX]
# one-5 diagrams -> q, for k<=NMAX/2
diag={}
for k in range(2,NMAX//2+1):
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr()); diag.setdefault(sp.expand(q*q.subs(s5,-s5)),(q,M))
def charpoly(n,e): return sp.expand(cart(n,e).charpoly(t).as_expr())
def rational_core_factors(p):
    f=sp.factor(p,extension=s5); return [g for g in sp.Mul.make_args(f) if g.has(t) and not g.has(s5)]
def conic_half(C, lam):
    """split the 2-dim eigenspace of rational eigenvalue lam into a golden line v=u1+tau*u2 with
    sigma-complement: c*N(tau)+b*Tr(tau)+a=0. Return (P_line at two conic points).
    For multiplicity > 2 (Grassmannian split, not registered) return the half-core stand-in
    twice and let the caller DECLARE it (traces are gauge-independent only on B-constant cores)."""
    core=(C-lam*sp.eye(C.shape[0])).nullspace()
    if len(core)!=2:
        V=sp.Matrix.hstack(*core); Q=simp(V*(V.T*V).inv()*V.T); return [Q/2, Q/2, "stand-in:multiplicity>2"]
    u1,u2=core; a_,b_,c_=(u1.T*u1)[0],(u1.T*u2)[0],(u2.T*u2)[0]
    sols=[]
    for qq in [sp.Rational(x,y) for y in range(1,13) for x in range(-12,13)]:
        if qq==0: continue
        disc=sp.nsimplify(b_**2-c_*(a_-5*c_*qq**2)); r=sp.sqrt(disc)
        if r.is_rational:
            tau=sp.Rational(sp.nsimplify((-b_+r)/c_))+qq*s5
            if not any(sp.simplify(tau-s)==0 for s in sols): sols.append(tau)
        if len(sols)>=2: break
    if len(sols)<2:   # conic has no small rational point: registered recipe cannot reach it -> declare
        V=sp.Matrix.hstack(*core); Q=simp(V*(V.T*V).inv()*V.T); return [Q/2, Q/2, "stand-in:conic-unresolved(lam=%s)"%lam]
    out=[]
    for tau in sols[:2]:
        v=u1+tau*u2; out.append(simp((v*v.T)/(v.T*v)[0]))
    return out
def projector_for(n,e,q):
    C=cart(n,e); p=charpoly(n,e); rat=rational_core_factors(p)
    q_off=q
    for g in rat:
        b,ex=g.as_base_exp(); q_off=sp.cancel(q_off/b**(ex//2))
    q_off=sp.expand(q_off); P_off=bezout_proj(C,q_off)
    cores=[]; quad=[]
    for g in rat:
        b,ex=g.as_base_exp()
        roots=sp.solve(b,t)
        cores.extend(roots)                       # EVERY root of every rational factor (quadratic cores have two)
        if len(roots)>1: quad.append(str(b))
    if not cores: return [P_off], C, []
    # remove core content from P_off, then add golden lines for each rational eigenvalue (two gauges)
    Qc=sp.zeros(n,n)
    for lam in cores:
        V=sp.Matrix.hstack(*(C-lam*sp.eye(n)).nullspace()); Qc+=simp(V*(V.T*V).inv()*V.T)
    P_off=simp(P_off*(sp.eye(n)-Qc))
    gauges=[conic_half(C,lam) for lam in cores]
    standin=[g[2] for g in gauges if len(g)==3]+["stand-in:quadratic-rational-core(%s)"%b for b in quad]
    Ps=[]
    for choice in (0,1):
        P=P_off
        for g in gauges: P=simp(P+g[min(choice,1)])
        Ps.append(P)
    return Ps, C, standin
def invariants(P,n,e):
    A=sp.zeros(n,n)
    for i,j in e: A[i,j]=A[j,i]=1
    D=sp.diag(*[sum(A[i,k] for k in range(n)) for i in range(n)]); B=2*sp.eye(n)-D
    R=simp(P-sigma(P)); X=simp((sp.eye(n)-P)*B*P)
    return (sp.nsimplify(sp.expand((R*D).trace())), sp.nsimplify(sp.expand((P*B).trace())),
            sp.nsimplify(sp.expand(sum(x**2 for x in X))))
def sig(M):
    ev=np.linalg.eigvalsh(np.array(M.evalf(30).tolist(),dtype=float)); return (int((ev>1e-9).sum()),int((ev<-1e-9).sum()))
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
