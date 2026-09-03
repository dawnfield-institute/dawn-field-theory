#!/usr/bin/env python3
"""exp_05: Galois blindness and the golden probe. Registration: Phase 2 seal 06073227.
Scored prediction = T2 only. T1 theorem-verification; T3/T4 (S) confirmations."""
import sys, json, numpy as np, sympy as sp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent/"core"))
from ledger import projector, sigma, simp, cart, EDGES
from scipy.linalg import expm
t=sp.Symbol('t'); s5=sp.sqrt(5)
res={"experiment_id":"exp_05_galois_blindness","registration":"06073227","tests":{}}

def M_of(n, edges, s):
    A=np.zeros((n,n))
    for i,j in edges: A[i,j]=A[j,i]=1
    D=np.diag(A.sum(1)); return (D-A)+s*(2*np.eye(n)-D)
def B_of(n, edges):
    A=np.zeros((n,n))
    for i,j in edges: A[i,j]=A[j,i]=1
    return 2*np.eye(n)-np.diag(A.sum(1))

data={}
for nm in ("A4","D6","E8"):
    C,P=projector(nm); n=C.shape[0]
    Pn=np.array(P.evalf(30).tolist(),dtype=float); data[nm]=(n,P,Pn)

# T1 (S): for a RATIONAL probe v, the copy occupancies ||Pv||^2 and ||(I-P)v||^2 are Galois
# conjugates — a rational observer cannot label the copies. Exact.
t1={}
for nm,(n,P,Pn) in data.items():
    v=sp.Matrix([sp.Rational(k+1,3) for k in range(n)])            # rational probe
    occ=sp.expand((v.T*P*v)[0]); occ_c=sp.expand((v.T*(sp.eye(n)-P)*v)[0])
    t1[nm]=bool(sp.simplify(occ.subs(s5,-s5)-occ_c)==0)
res["tests"]["T1"]={**t1,"pass":all(t1.values()),"class":"S"}

# T2 (SCORED): golden-probe leakage is FIRST order in |s-1| with exact coefficient (I-P)BP != 0
t2={}
for nm,(n,P,Pn) in data.items():
    e=EDGES[nm][1]; Bs=sp.Matrix(B_of(n,e).astype(int).tolist())
    X=simp((sp.eye(n)-P)*Bs*P); nrm2=sp.expand(sum(x**2 for x in X))
    nonzero = sp.simplify(nrm2)!=0
    in_field = (not sp.nsimplify(nrm2).has(sp.I)) and all(a.is_rational for a in sp.Poly(sp.expand(nrm2.subs(s5,sp.Symbol('S'))),sp.Symbol('S')).all_coeffs())
    # numeric order of leakage near s=1 (tau small so expm is well-conditioned)
    tt=0.3; deltas=np.array([1e-3,2e-3,5e-3,1e-2,2e-2,5e-2])
    leak=[]
    for d in deltas:
        M=M_of(n,e,1+d); L=(np.eye(n)-Pn)@expm(-tt*M)@Pn; leak.append(np.linalg.norm(L))
    slope=np.polyfit(np.log(deltas),np.log(leak),1)[0]
    is_rational = bool(sp.nsimplify(nrm2).is_rational)
    t2[nm]={"||(I-P)BP||^2":str(sp.nsimplify(nrm2)),"nonzero":bool(nonzero),"coef_in_Q(sqrt5)":bool(in_field),
            "coef_rational":is_rational,"leak_order_slope":round(float(slope),3),"first_order":bool(abs(slope-1)<0.1)}
# SEALED T2 text (06073227): "Fails if leakage is second-order ... OR THE COEFFICIENT IS RATIONAL."
# The first run's pass logic omitted the rationality clause. Scoring against the seal:
res["tests"]["T2"]={**t2,
    "pass":all(v["nonzero"] and v["first_order"] and not v["coef_rational"] for v in t2.values()),
    "class":"PREDICTION",
    "note":"physics clauses (first-order, nonzero) hold on all three; the rationality clause FAILS — and is provably unwinnable: ||(I-P)BP||_F^2 = ||PB(I-P)||_F^2 by transposition, hence sigma-fixed, hence rational (T1). The sealed clause contradicted T1."}

# T3 (S): zero-mean isotropic noise — sigma-copy share of injected power = tr(P Cov)/tr(Cov)
t3={}; rng=np.random.default_rng(1)
for nm,(n,P,Pn) in data.items():
    Cov=np.eye(n)-np.ones((n,n))/n; exact=float(np.trace(Pn@Cov)/np.trace(Cov))
    eps=rng.normal(size=(20000,n)); eps-=eps.mean(1,keepdims=True)
    mc=float(np.mean(np.sum((eps@Pn.T)**2,1))/np.mean(np.sum(eps**2,1)))
    t3[nm]={"exact":round(exact,4),"monte_carlo":round(mc,4),"ok":abs(exact-mc)<0.01}
res["tests"]["T3"]={**t3,"pass":all(v["ok"] for v in t3.values()),"class":"S"}

# T4 (S): rational probe under RATIONAL polynomial dynamics (I - dt M)^k: retention in the copy
# and leakage into the conjugate copy are Galois conjugates at every s. Exact.
t4={}
for nm,(n,P,Pn) in data.items():
    e=EDGES[nm][1]; ok=True
    for sv in (sp.Rational(1,2), sp.Rational(1), sp.Rational(3,2)):
        Ms=sp.Matrix(cart(n,e))*sv + (sp.Matrix(M_of(n,e,0.0).astype(int).tolist()))*(1-sv)
        U=(sp.eye(n)-sp.Rational(1,100)*Ms)**8
        v=sp.Matrix([sp.Rational(k+1,3) for k in range(n)]); w=U*v
        a=sp.expand((w.T*P*w)[0]); b=sp.expand((w.T*(sp.eye(n)-P)*w)[0])
        if sp.simplify(a.subs(s5,-s5)-b)!=0: ok=False
    t4[nm]=ok
res["tests"]["T4"]={**t4,"pass":all(t4.values()),"class":"S"}
res["score"]=sum(v["pass"] for v in res["tests"].values())
print(json.dumps(res["tests"],indent=1,default=str)); print("SCORE",res["score"],"/4  (scored prediction: T2)")
Path(__file__).parent.parent.joinpath("results","exp_05_galois_20260901.json").write_text(json.dumps(res,indent=1,default=str))
