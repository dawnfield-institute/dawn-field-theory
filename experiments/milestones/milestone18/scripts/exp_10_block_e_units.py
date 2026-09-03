#!/usr/bin/env python3
"""exp_10 (Block E): re-derivations as units. Registered in the Phase 2 seal."""
import json, sympy as sp
from pathlib import Path
s5=sp.sqrt(5); phi=(1+s5)/2; psi=(1-s5)/2
N=lambda x: sp.simplify(sp.expand(x)*sp.expand(x).subs(s5,-s5))
res={"experiment_id":"exp_10_block_e_units","tests":{}}
# T1: exp_37's 1/phi^4 is a unit of norm +1 (even power -> split side, not ramified)
n1=N(phi**-4); res["tests"]["T1"]={"N(phi^-4)":str(n1),"value":float(phi**-4),"pass":n1==1}
# T2: Baxter z_c = phi^5 = F5 phi + F4, norm -1
zc=(11+5*s5)/2; n2=N(zc)
res["tests"]["T2"]={"zc==phi^5":sp.simplify(zc-phi**5)==0,"zc==5phi+3":sp.simplify(zc-(5*phi+3))==0,
                    "N":str(n2),"pass":sp.simplify(zc-phi**5)==0 and n2==-1}
# T3: Lucas = traces of Q^n (rational), Fibonacci = Delta readout (phi^n-psi^n)/sqrt5, n=1..30
Q=sp.Matrix([[1,1],[1,0]]); ok=True; Qn=sp.eye(2)
for n in range(1,31):
    Qn=Qn*Q
    L=sp.simplify(phi**n+psi**n); F=sp.simplify((phi**n-psi**n)/s5)
    if not (Qn.trace()==L and Qn[0,1]==F and L.is_Integer and F.is_Integer): ok=False; break
res["tests"]["T3"]={"n_range":"1..30","trace=Lucas, offdiag=Fibonacci, both integers":ok,"pass":ok}
# T4: Delta-ratio rationality: F_m/F_n in Q; unpaired phi -> irrational (alpha's phi-bearing factor)
F=sp.fibonacci
sin2=sp.Rational(F(4),F(7)); alpha_factor=sp.Rational(F(3),F(4)*F(10))/phi
res["tests"]["T4"]={"sin2thetaW=F4/F7":str(sin2),"rational":sin2.is_rational,
                    "alpha_factor F3/(F4*phi*F10) rational?":bool(sp.nsimplify(alpha_factor).is_rational),
                    "pass":bool(sin2.is_rational and not sp.nsimplify(alpha_factor).is_rational)}
res["score"]=sum(t["pass"] for t in res["tests"].values())
print(json.dumps(res["tests"],indent=1,default=str)); print("SCORE",res["score"],"/4")
Path(__file__).parent.parent.joinpath("results","exp_10_block_e_20260901.json").write_text(json.dumps(res,indent=1,default=str))
