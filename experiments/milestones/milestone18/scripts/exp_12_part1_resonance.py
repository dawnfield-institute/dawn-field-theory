#!/usr/bin/env python3
"""exp_12 T1 (registration 5bc8faff): field-resonance law on A13, A14, D13, D14 and the affine
trees D~n (n=5..8), E~6, E~7, E~8. Predicted: finite trees pair over Q(sqrt d) iff sqrt d in
Q(zeta_2h); affine trees are never strict (their spectrum contains 0); grades recorded."""
import json, sympy as sp
from pathlib import Path
t=sp.Symbol('t')
def cart(n,e):
    C=2*sp.eye(n)
    for i,j in e: C[i,j]=C[j,i]=-1
    return C
def grade(p,dd):
    sd=sp.sqrt(dd); f=sp.factor(p,extension=sd); facs=[g for g in sp.Mul.make_args(f) if g.has(t)]
    gold=[g for g in facs if g.has(sd)]; rat=[g for g in facs if not g.has(sd)]
    if not gold: return "-"
    if not rat: return "strict"
    return "core" if all(g.as_base_exp()[1]%2==0 for g in rat) else "partial"
def pred(m):
    return [dd for dd in (2,3,5,7,13,15) if m%dd==0 and (dd%4==1 or m%(4*dd)==0)]
path=lambda n:[(i,i+1) for i in range(n-1)]
Dn=lambda n:[(i,i+1) for i in range(n-2)]+[(n-3,n-1)]
finite={"A13":(path(13),14),"A14":(path(14),15),"D13":(Dn(13),24),"D14":(Dn(14),26)}
affine={"D~5":[(0,1),(1,2),(2,3),(2,4),(1,5)],"D~6":[(0,1),(1,2),(2,3),(3,4),(3,5),(1,6)],
        "D~7":[(0,1),(1,2),(2,3),(3,4),(4,5),(4,6),(1,7)],"D~8":[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(5,7),(1,8)],
        "E~6":[(0,1),(1,2),(2,3),(3,4),(2,5),(5,6)],"E~7":[(i,i+1) for i in range(6)]+[(3,7)],
        "E~8":[(i,i+1) for i in range(7)]+[(2,8)]}
res={"registration":"5bc8faff","finite":{},"affine":{}}
ok=True
print("finite:  tree  h  predicted  observed(sqrt2,3,5,7,13,15)  law?")
for nm,(e,h) in finite.items():
    n=max(max(x) for x in e)+1; p=sp.expand(cart(n,e).charpoly(t).as_expr())
    obs={dd:grade(p,dd) for dd in (2,3,5,7,13,15)}; pr=pred(2*h)
    holds=all((obs[dd]!='-')==(dd in pr) for dd in obs); ok&=holds
    res["finite"][nm]={"h":h,"predicted":pr,"observed":obs,"law":holds}
    print(f"  {nm:<4} {h:>3}  {str(pr):<12} {[obs[dd] for dd in (2,3,5,7,13,15)]}  {holds}")
print("affine (recorded; predicted never strict):")
noaff=True
for nm,e in affine.items():
    n=max(max(x) for x in e)+1; p=sp.expand(cart(n,e).charpoly(t).as_expr())
    obs={dd:grade(p,dd) for dd in (2,3,5,7,13,15)}; s=any(v=="strict" for v in obs.values()); noaff&=(not s)
    res["affine"][nm]={"observed":obs}; print(f"  {nm:<4} {[obs[dd] for dd in (2,3,5,7,13,15)]}")
res["T1"]={"finite_law":ok,"affine_never_strict":noaff,"pass":ok and noaff}
print("T1:",res["T1"])
Path(__file__).parent.parent.joinpath("results","exp_12_part1_resonance.json").write_text(json.dumps(res,indent=1,default=str))
