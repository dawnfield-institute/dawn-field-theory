#!/usr/bin/env python3
"""exp_13: Phase 4 (registration SEALED, commit ade32b0a) at n=16.
T1 one-5 conjecture at k=8; T2 strict trees are folds; T3 strict invariant; T4 vertex law +
two-component structure; T5 trace law on registered-domain core folds; T5b component relations;
T6 sector classification of non-Galois golden trees (trivial orbit partition = UNEXPLAINED = fail)."""
import json, time, sys, sympy as sp, networkx as nx
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import bezout_proj, simp, cart as _cart
t=sp.Symbol('t'); s5=sp.sqrt(5); phi=(1+s5)/2
RES=Path(__file__).parent.parent/"results"; OUT=RES/"exp_13_n16.json"
def cart(n,e): return _cart(n,[tuple(x) for x in e])
def sigma(M): return M.applyfunc(lambda x: sp.expand(x.subs(s5,-s5)))
def charpoly(n,e): return sp.expand(cart(n,e).charpoly(t).as_expr())
def grade(p):  # exp_11's instrument, verbatim
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
            out.setdefault(tgt,{"q":q,"edges":E,"pos":pos})
    return out
src12=open(str(Path(__file__).parent/"exp_12_part2_fold_laws.py")).read()
exec(src12[src12.index("def rational_core_factors"):src12.index("def sig(")])   # rational_core_factors, conic_half, projector_for, invariants
def orbits(G):
    def canon(root):
        def rec(v,parent): return "("+"".join(sorted(rec(w,v) for w in G[v] if w!=parent))+")"
        return rec(root,None)
    key={v:canon(v) for v in G}; seen={}
    for v in sorted(G): seen.setdefault(key[v],[]).append(v)
    return list(seen.values())
def restricted_charpoly(C,V):
    G=V.T*V; M=V.T*C*V
    return sp.expand(sp.cancel((M-t*G).det()/G.det()))
def golden_bases(p):
    return sorted(str(sp.expand(g.as_base_exp()[0])) for g in sp.Mul.make_args(sp.factor(p,extension=s5)) if g.has(s5))
def sector_check(n,e):
    """(is_pure_quotient, sperp_pairs, unexplained)"""
    G=nx.Graph([tuple(x) for x in e]); orb=orbits(G)
    if len(orb)==n: return (False, False, True)
    S=sp.Matrix.hstack(*[sp.Matrix([1 if v in c else 0 for v in range(n)]) for c in orb])
    Sperp=sp.Matrix.hstack(*S.T.nullspace())
    p=charpoly(n,e)
    gS=golden_bases(restricted_charpoly(cart(n,e),S)); gT=golden_bases(p)
    gP=golden_bases(restricted_charpoly(cart(n,e),Sperp)) if Sperp.shape[1] else []
    if not gP: return (gS==gT, True, False)
    paired=all(any(sp.expand(sp.sympify(b).subs(s5,-s5)-sp.sympify(b2))==0 for b2 in gP) for b in gP)
    return (False, paired, False)
def fold_structure(n,e,P,dgm):
    R=simp(P-sigma(P)); d=[sp.radsimp(sp.expand(R[v,v])) for v in range(n)]
    vertex_law=all(sp.simplify(x**2-sp.Rational(1,5))==0 for x in d)
    zeros=[v for v in range(n) if d[v]==0]
    copy=[v for v in range(n) if d[v].evalf()>0]; conj=[v for v in range(n) if d[v].evalf()<0]
    G=nx.Graph([tuple(x) for x in e]); cut=[x for x in e if (x[0] in copy)!=(x[1] in copy)]
    cc=nx.number_connected_components(G.subgraph(copy)) if copy else 0
    sizes=sorted(len(c) for c in nx.connected_components(G.subgraph(conj))) if conj else []
    DG=nx.Graph(dgm["edges"]); DG.remove_edge(*dgm["edges"][dgm["pos"]])
    halves=sorted(len(c) for c in nx.connected_components(DG))
    return {"vertex_law":vertex_law,"zeros":zeros,"copy_comps":cc,"conj_comps":len(sizes),
            "conj_sizes":sizes,"cut":len(cut),"halves":halves,"halves_match":sizes==halves,
            "copy_connected":cc==1,"relations":(len(sizes)==cc+1 and len(cut)==2*cc)}
# ---- self-tests (sealed) ----
assert grade(charpoly(8,[(i,i+1) for i in range(6)]+[(2,7)]))=="strict", "E8 strict"
assert grade(charpoly(6,[(0,1),(1,2),(2,3),(3,4),(3,5)]))=="core", "D6 core"
assert grade(charpoly(5,[(i,i+1) for i in range(4)]))=="none", "A5 none"
d4=diagrams(4); e8=[(i,i+1) for i in range(6)]+[(2,7)]; p8=charpoly(8,e8)
assert p8 in d4, "E8 has a 4-node one-5 partner"
P8=bezout_proj(cart(8,e8),d4[p8]["q"]); inv=invariants(P8,8,[tuple(x) for x in e8])
assert sp.simplify(inv[0]-2*s5/5)==0 and sp.simplify(inv[2]-sp.Rational(2,5))==0, "E8 invariants"
st=fold_structure(8,e8,P8,d4[p8])
assert st["vertex_law"] and st["copy_connected"] and st["halves_match"], "E8 structure"
cen=json.load(open(RES/"explore_g1_census_20260901.json"))
q6=[r for r in cen if r["n"]==6 and r["det"]==-16][0]
assert sector_check(6,q6["edges"])[0], "n=6 det -16 pure quotient fold"
print("self-tests passed",flush=True)
# ---- stage 1: census at 16 ----
t0=time.time(); n=16; idx={}; grades={}; count=0
for T in nx.nonisomorphic_trees(n):
    e=sorted(map(list,T.edges())); p=charpoly(n,e)
    idx[p]=e; grades[p]=grade(p); count+=1
    if count%500==0: print(f"  {count} trees  [{time.time()-t0:.0f}s]",flush=True)
print(f"enumerated {count} trees on 16 vertices [{time.time()-t0:.0f}s]",flush=True)
gc={g:sum(1 for x in grades.values() if x==g) for g in ("strict","core","partial","none")}
print("grade counts:",gc,flush=True)
# ---- stage 2 ----
one5=diagrams(8); parents={tg:idx.get(tg) for tg in one5}
t1=all(v is not None for v in parents.values())
strict=[p for p in idx if grades[p]=="strict"]
t2=all(p in one5 for p in strict)
T3rows=[];T4rows=[];T5rows=[];T5brows=[];T6rows=[];checks=[]
for p in strict:
    if p not in one5: continue
    e=idx[p]; C=cart(n,e); q=one5[p]["q"]; P=bezout_proj(C,q)
    ledger=(simp(P*P-P)==sp.zeros(n,n)) and (simp(P+sigma(P)-sp.eye(n))==sp.zeros(n,n)) and P.rank()==8
    inv=invariants(P,n,[tuple(x) for x in e]); st=fold_structure(n,e,P,one5[p])
    detq=sp.nsimplify(sp.expand(q.subs(t,0)*q.subs(t,0).subs(s5,-s5)))
    checks.append({"det_law":sp.simplify(detq-C.det())==0})
    T3rows.append({"edges":e,"ledger":ledger,"trRD":str(inv[0]),"leak":str(inv[2]),
        "ok":ledger and sp.simplify(inv[0]-2*s5/5)==0 and sp.simplify(inv[2]-sp.Rational(2,5))==0})
    T4rows.append({"edges":e,**{k:str(v) if not isinstance(v,(bool,int,list)) else v for k,v in st.items()},
        "ok":st["vertex_law"] and st["copy_connected"] and st["conj_comps"]==2 and st["halves_match"] and not st["zeros"]})
    print(f"strict fold det={C.det()}: T3 {T3rows[-1]['ok']} T4 {T4rows[-1]['ok']}",flush=True)
for p in idx:
    if grades[p]!="core" or p not in one5: continue
    e=idx[p]; q=one5[p]["q"]
    Ps,C,standin=projector_for(n,[tuple(x) for x in e],q)
    if standin: T5rows.append({"edges":e,"declared":standin}); continue
    invs=[invariants(P_,n,[tuple(x) for x in e]) for P_ in Ps]
    gi=all(sp.simplify(invs[0][i]-invs[-1][i])==0 for i in range(3))
    if not gi: T5rows.append({"edges":e,"declared":["gauge-dependent"]}); continue
    ledger=(simp(Ps[0]*Ps[0]-Ps[0])==sp.zeros(n,n)) and (simp(Ps[0]+sigma(Ps[0])-sp.eye(n))==sp.zeros(n,n))
    st=fold_structure(n,e,Ps[0],one5[p])
    T5rows.append({"edges":e,"ledger":ledger,"trRD":str(invs[0][0]),"leak_recorded":str(invs[0][2]),
        "ok":ledger and sp.simplify(invs[0][0]-2*s5/5)==0})
    T5brows.append({"edges":e,"copy_comps":st["copy_comps"],"conj_comps":st["conj_comps"],"cut":st["cut"],
        "halves_recorded":st["halves_match"],"ok":st["relations"] and not st["zeros"]})
    print(f"core fold det={cart(n,e).det()}: T5 {T5rows[-1]['ok']} T5b {T5brows[-1]['ok']}",flush=True)
for p in idx:
    if grades[p] not in ("strict","core") or p in one5: continue
    e=idx[p]; pure,paired,unexplained=sector_check(n,e)
    T6rows.append({"edges":e,"det":str(cart(n,e).det()),"pure_quotient":pure,"sperp_pairs":paired,"unexplained":unexplained,
        "ok":(not unexplained) and paired})
    print(f"non-Galois golden det={T6rows[-1]['det']}: pure={pure} paired={paired} unexplained={unexplained}",flush=True)
dom=[r for r in T5rows if "ok" in r]
tests={"T1":t1,"T2":t2,
 "T3":all(r["ok"] for r in T3rows),"T4":all(r["ok"] for r in T4rows),
 "T5":all(r["ok"] for r in dom),"T5_domain":len(dom),"T5_declared":len(T5rows)-len(dom),
 "T5b":all(r["ok"] for r in T5brows),
 "T6":all(r["ok"] for r in T6rows),
 "strict_count":len(strict),"det_law_checks":all(c["det_law"] for c in checks)}
res={"registration":"ade32b0a","n":16,"trees":count,"grade_counts":gc,
 "orphans":[str(one5[tg]["q"])[:60] for tg,v in parents.items() if v is None],
 "T3":T3rows,"T4":T4rows,"T5":T5rows,"T5b":T5brows,"T6":T6rows,"tests":tests}
OUT.write_text(json.dumps(res,indent=1,default=str))
print("TESTS:",{k:v for k,v in tests.items()},flush=True); print("SCORE DONE",flush=True)
