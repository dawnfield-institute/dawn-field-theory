#!/usr/bin/env python3
"""explore_r19: recover the matching Pi, signs S and the defect edge from R for every evaluable
partnered strict fold at n=20 (exp_15 T3 rows), and save them — the input for Phase 7c's pre-seal
null computation (lesson 8) and for any dynamics test at 20."""
import json, sys, sympy as sp, networkx as nx, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import cart, simp, sigma, bezout_proj, t
s5=sp.sqrt(5); phi=(1+s5)/2; RES=Path(__file__).parent.parent/"results"
K=sp.QQ.algebraic_field(s5)
partners={}
for T in nx.nonisomorphic_trees(10):
    E=list(T.edges())
    for pos in range(len(E)):
        M=2*sp.eye(10)
        for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
        q=sp.expand(M.charpoly(t).as_expr()); partners.setdefault(str(sp.expand(q*q.subs(s5,-s5))),[]).append(q)
r=json.load(open(RES/'exp_15_n20.json')); rows=[x for x in r["T3"] if x.get("ok") is not None]
out=[]; t0=time.time()
for x in rows:
    e=[tuple(v) for v in x["edges"]]; n=20; C=cart(n,e); p=str(sp.expand(C.charpoly(t).as_expr()))
    q=None
    for qc in partners[p]:
        if sp.gcd(sp.Poly(qc,t,domain=K),sp.Poly(sp.expand(qc.subs(s5,-s5)),t,domain=K)).degree()==0: q=qc; break
    R=simp(bezout_proj(C,q)); R=simp(R-sigma(R)); S5R=simp(s5*R).applyfunc(sp.nsimplify)
    match={v:[w for w in range(n) if w!=v and S5R[v,w]!=0][0] for v in range(n)}
    sign={v:int(S5R[v,v]) for v in range(n)}
    E={tuple(sorted(v)) for v in e}
    d=[Ed for Ed in E if tuple(sorted((match[Ed[0]],match[Ed[1]]))) not in E]
    assert len(d)==1
    out.append({"edges":e,"det":int(C.det()),"defect":list(d[0]),"matching":match,"sign":sign})
    print(f"fold {len(out)}/{len(rows)} det={out[-1]['det']} defect={d[0]} [{time.time()-t0:.0f}s]",flush=True)
json.dump(out,open(RES/'explore_r19_n20_defects.json','w'),indent=1)
print("DONE",flush=True)
