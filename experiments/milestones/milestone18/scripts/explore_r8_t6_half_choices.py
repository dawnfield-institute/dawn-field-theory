#!/usr/bin/env python3
"""explore_r8 (exploration): for every quotient-fold candidate at n=14 with pure (t-2)^m core, evaluate
tr(RD) and tr(PB) for EVERY choice of golden half (one factor from each sigma-pair). Tells whether a
T6 failure is the seal's under-specification (some half gives 0) or genuine (no half gives 0)."""
import sys, json, itertools, sympy as sp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from foldlaws import *   # promoted from exp_12_part2 on 2026-09-02
census, golden, diag = load_context(14)
out=[]
for r in golden:
    n=r["n"]; e=[tuple(x) for x in r["edges"]]; p=charpoly(n,e)
    if n!=14 or p in diag: continue
    rat=rational_core_factors(p)
    if not all(str(g).startswith("(t - 2)") for g in rat): continue
    C=cart(n,e); f=sp.factor(p,extension=s5); gold=[g for g in sp.Mul.make_args(f) if g.has(s5)]
    pairs=[]; seen=set()
    for g in gold:
        b,ex=g.as_base_exp(); kb=str(sp.expand(b)); kc=str(sp.expand(b.subs(s5,-s5)))
        if kb in seen or kc in seen: continue
        seen.add(kb); seen.add(kc); pairs.append((b**ex, sp.expand(b.subs(s5,-s5))**ex))
    V=sp.Matrix.hstack(*(C-2*sp.eye(n)).nullspace()); Qc=simp(V*(V.T*V).inv()*V.T)
    vals=[]
    for choice in itertools.product((0,1),repeat=len(pairs)):
        if choice[0]==1: continue                      # overall swap is the same split
        q=sp.Integer(1)
        for c,(a,b) in zip(choice,pairs): q*=(a if c==0 else b)
        P_off=bezout_proj(C,sp.expand(q)); P=simp(P_off*(sp.eye(n)-Qc)+Qc/2)
        inv=invariants(P,n,e); vals.append((choice,str(inv[0]),str(inv[1])))
    print(f"det={r['det']:>7} pairs={len(pairs)} core={[str(g) for g in rat]}")
    for v in vals: print("    choice",v[0],"trRD",v[1],"trPB",v[2])
    out.append({"det":r["det"],"edges":e,"pairs":len(pairs),"core":[str(g) for g in rat],"choices":vals})
json.dump(out,open(RES/'explore_r8_t6_half_choices_n14.json','w'),indent=1,default=str)
print("DONE")
