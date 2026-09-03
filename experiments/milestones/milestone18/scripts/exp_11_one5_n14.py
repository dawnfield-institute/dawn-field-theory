#!/usr/bin/env python3
"""exp_11: the one-5 prediction at n=14. Registration 5bc8faff (sealed). Exhaustive.
T1 every 7-node one-5 tree diagram has a 14-vertex tree parent; T2 no two-5 linear diagram does;
T3 every parent is core-grade; T4 no strict sqrt5-golden 14-tree exists."""
import json, time, sympy as sp, networkx as nx
from pathlib import Path
t=sp.Symbol('t'); s5=sp.sqrt(5); phi=(1+s5)/2
OUT=Path(__file__).parent.parent/"results"/"exp_11_one5_n14.json"
def cart(n,e):
    C=2*sp.eye(n)
    for i,j in e: C[i,j]=C[j,i]=-1
    return C
def grade(p):
    fQ=sp.factor(p); facsQ=[g for g in sp.Mul.make_args(fQ) if g.has(t)]
    for g in facsQ:
        b,e=g.as_base_exp()
        if sp.degree(b,t)%2==1 and sp.degree(b,t)>1: return "none"
    f=sp.factor(p,extension=s5); facs=[g for g in sp.Mul.make_args(f) if g.has(t)]
    gold=[g for g in facs if g.has(s5)]; rat=[g for g in facs if not g.has(s5)]
    if not gold: return "none"
    if not rat: return "strict"
    return "core" if all(g.as_base_exp()[1]%2==0 for g in rat) else "partial"
def diagrams(k):
    out={}
    for T in nx.nonisomorphic_trees(k):
        E=list(T.edges())
        for pos in range(len(E)):
            M=2*sp.eye(k)
            for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
            q=sp.expand(M.charpoly(t).as_expr()); tgt=sp.expand(q*q.subs(s5,-s5))
            out.setdefault(tgt,{"q":str(q),"shape":sorted(map(list,E)),"bond":list(E[pos])})
    return out
def linear(labels):
    k=len(labels)+1; M=2*sp.eye(k)
    for i,m in enumerate(labels): M[i,i+1]=M[i+1,i]=(-phi if m==5 else -1)
    q=sp.expand(M.charpoly(t).as_expr()); return sp.expand(q*q.subs(s5,-s5))
# ---- self-test on n=8 (E8 <-> [3,3,5], cat8 <-> [3,5,3]) ----
idx8={sp.expand(cart(8,list(T.edges())).charpoly(t).as_expr()):sorted(map(list,T.edges())) for T in nx.nonisomorphic_trees(8)}
d4=diagrams(4); assert sum(1 for tg in d4 if tg in idx8)==3, "self-test: 3 one-5 4-node diagrams must have 8-vertex parents"
assert grade(sp.expand(cart(8,[(i,i+1) for i in range(6)]+[(2,7)]).charpoly(t).as_expr()))=="strict", "self-test: E8 strict"
assert grade(sp.expand(cart(6,[(0,1),(1,2),(2,3),(3,4),(3,5)]).charpoly(t).as_expr()))=="core", "self-test: D6 core"
print("self-tests passed", flush=True)
# ---- the run ----
t0=time.time(); n=14; idx={}; grades={}; count=0
for T in nx.nonisomorphic_trees(n):
    e=sorted(map(list,T.edges())); p=sp.expand(cart(n,[tuple(x) for x in e]).charpoly(t).as_expr())
    idx[p]=e; grades[p]=grade(p); count+=1
    if count%300==0: print(f"  {count} trees  [{time.time()-t0:.0f}s]", flush=True)
print(f"enumerated {count} trees on 14 vertices [{time.time()-t0:.0f}s]", flush=True)
one5=diagrams(7)
two5=[[5,3,3,3,3,5],[3,5,3,3,3,5],[3,3,5,3,3,5],[5,3,3,3,5,3],[3,5,3,3,5,3]]
parents={tg:idx.get(tg) for tg in one5}
t1=all(v is not None for v in parents.values())
t2=all(linear(l) not in idx for l in two5)
t3=all(grades[tg]=="core" for tg,v in parents.items() if v is not None)
strict=[e for p,e in idx.items() if grades[p]=="strict"]
t4=len(strict)==0
res={"registration":"5bc8faff","n":14,"trees":count,"one5_diagrams":len(one5),
     "orphans":[one5[tg] for tg,v in parents.items() if v is None],
     "two5_with_parent":[l for l in two5 if linear(l) in idx],
     "parent_grades":{str(one5[tg]["q"])[:40]:grades[tg] for tg,v in parents.items() if v is not None},
     "strict_trees":strict,"grade_counts":{g:sum(1 for x in grades.values() if x==g) for g in ("strict","core","partial","none")},
     "folds":[{"edges":v,"diagram":one5[tg]} for tg,v in parents.items() if v is not None],
     "tests":{"T1":t1,"T2":t2,"T3":t3,"T4":t4},"score":sum([t1,t2,t3,t4])}
OUT.write_text(json.dumps(res,indent=1,default=str))
print(json.dumps(res["tests"]),"SCORE",res["score"],"/4"); print("grade counts:",res["grade_counts"]); print("DONE",flush=True)
