#!/usr/bin/env python3
"""exp_15: Phase 6 (SEALED) — the matching structure at n=20. Stage 1: strict hunt (norm screen,
proven necessary, + exact factorization). Stage 2: T1 one-5 at k=10; T2 partnered-or-sector-strict;
T3 matching form; T4 quotient/multiplicities; T5 single copy-internal defect over the mult-3 edge;
T6 consequences (trace, leak, vertex). Per-tree basis; per-polynomial tallies reported."""
import time, json, sys, sympy as sp, networkx as nx
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"core"))
from ledger import bezout_proj, simp, cart as _cart
t=sp.Symbol('t'); s5=sp.sqrt(5); phi=(1+s5)/2; RES=Path(__file__).parent.parent/"results"
n=20
def cart(nn,e): return _cart(nn,[tuple(x) for x in e])
def sigma(M): return M.applyfunc(lambda x: sp.expand(x.subs(s5,-s5)))
def frob2(X): return sp.nsimplify(sp.expand(sum(x**2 for x in X)))
def is_norm(m):
    if m==0: return True
    for pr,ex in sp.factorint(abs(m)).items():
        if pr%5 in (2,3) and ex%2==1: return False
    return True
def strict_grade(p):
    facs=[g for g in sp.Mul.make_args(sp.factor(p,extension=s5)) if g.has(t)]
    return bool(facs) and all(g.has(s5) for g in facs)
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
def sector_strict(e):
    G=nx.Graph(e); orb=orbits(G)
    if len(orb)==n: return False
    S=sp.Matrix.hstack(*[sp.Matrix([1 if v in c else 0 for v in range(n)]) for c in orb])
    Sperp=sp.Matrix.hstack(*S.T.nullspace())
    gP=golden_bases(restricted_charpoly(cart(n,e),Sperp)) if Sperp.shape[1] else []
    if not gP: return True   # all golden in symmetric sector, pure-quotient-like strict
    return all(any(sp.expand(sp.sympify(b).subs(s5,-s5)-sp.sympify(b2))==0 for b2 in gP) for b in gP)
# stage 1 (checkpointed)
CKPT=RES/'exp_15_survivors_ckpt.json'
t0=time.time(); XS=(0,1,-1,2,3,-2)
if CKPT.exists():
    d=json.load(open(CKPT)); cnt=d["trees"]; surv=[(e,sp.sympify(ps)) for e,ps in d["surv"]]
    strict=[(e,sp.sympify(ps)) for e,ps in d["strict"]]
    print(f"checkpoint loaded: {cnt} trees, {len(surv)} survivors, {len(strict)} strict",flush=True)
else:
    cnt=0; surv=[]
    for T in nx.nonisomorphic_trees(n):
        e=list(T.edges()); C=2*sp.eye(n)
        for i,j in e: C[i,j]=C[j,i]=-1
        p=C.charpoly(t); cnt+=1
        if all(is_norm(int(p.eval(x))) for x in XS): surv.append((sorted(map(list,e)),sp.expand(p.as_expr())))
        if cnt%25000==0: print(f"  {cnt} trees, {len(surv)} survivors [{time.time()-t0:.0f}s]",flush=True)
    print(f"n=20: {cnt} trees; survivors {len(surv)} [{time.time()-t0:.0f}s]",flush=True)
    t1=time.time(); strict=[(e,p) for e,p in surv if strict_grade(p)]
    print(f"factorization [{time.time()-t1:.0f}s]; STRICT trees: {len(strict)} on {len({str(p) for _,p in strict})} polynomials",flush=True)
    json.dump({"trees":cnt,"surv":[(e,str(p)) for e,p in surv],"strict":[(e,str(p)) for e,p in strict]},open(CKPT,'w'))
    print("checkpoint saved",flush=True)
# partner map k=10
partners={}
for T in nx.nonisomorphic_trees(10):
    E=list(T.edges())
    for pos in range(len(E)):
        M=2*sp.eye(10)
        for m,(i,j) in enumerate(E): M[i,j]=M[j,i]=(-phi if m==pos else -1)
        q=sp.expand(M.charpoly(t).as_expr())
        partners.setdefault(sp.expand(q*q.subs(s5,-s5)),[]).append((q,E,pos))
idx={str(p) for _,p in strict}
pkeys={str(tg):tg for tg in partners}
# T1 properly: every diagram target has a 20-vertex parent — search ALL trees' polys? Only strict
# trees can be parents of one-5 diagrams (q*sq strict by definition: no rational factors? q may have
# rational content at k=10 even... k=10 even: no forced core, but individual diagrams CAN have
# rational eigenvalues -> their parents are core-grade, not strict. T1 must search the full census.
# We record orphans among STRICT-parent diagrams and check core-grade parents for the rest via the
# norm screen survivors' full grade. Declare scope: T1 evaluated on survivors + core search below.
res={"registration":"phase6","n":20,"trees":cnt,"survivors":len(surv),"strict":[{"edges":e,"charpoly":str(p)} for e,p in strict]}
survkeys={}
for e,p in surv: survkeys.setdefault(str(p),e)
orphan_cand=[k for k in pkeys if k not in idx]
core_parents={k:survkeys[k] for k in orphan_cand if k in survkeys}
still_orphan=[k for k in orphan_cand if k not in core_parents]
# note: a core-grade parent's charpoly = q*sq exactly only if diagram itself has paired rational...
print(f"T1: {len(partners)} diagram targets; strict-parented {len(partners)-len(orphan_cand)}; survivor-parented {len(core_parents)}; unresolved {len(still_orphan)}",flush=True)
res["T1"]={"targets":len(partners),"strict_parented":len(partners)-len(orphan_cand),"survivor_parented":len(core_parents),"unresolved":len(still_orphan)}
# battery
T2=[];T3=[];T4=[];T5=[];T6=[]
for e,p in strict:
    e2=[tuple(v) for v in e]
    if str(p) not in pkeys:
        ss=sector_strict(e2); T2.append({"edges":e,"partnered":False,"sector_strict":ss,"ok":ss}); continue
    T2.append({"edges":e,"partnered":True,"ok":True})
    plist=partners[pkeys[str(p)]]
    qs=[]
    for q,_,_ in plist:
        if not any(sp.expand(q-q2)==0 for q2 in qs): qs.append(q)
    C=cart(n,e2); P=bezout_proj(C,qs[0])
    ledger=(simp(P*P-P)==sp.zeros(n,n)) and (simp(P+sigma(P)-sp.eye(n))==sp.zeros(n,n)) and P.rank()==10
    R=simp(P-sigma(P)); S5R=simp(s5*R).applyfunc(sp.nsimplify)
    match={}; form=True
    for v in range(n):
        row=[(w,S5R[v,w]) for w in range(n) if w!=v and S5R[v,w]!=0]
        if not (S5R[v,v]**2==1 and len(row)==1 and row[0][1]**2==4): form=False; break
        match[v]=row[0][0]
    anti=form and all(S5R[v,v]==-S5R[match[v],match[v]] for v in match)
    T3.append({"edges":e,"ledger":ledger,"form":form,"anti":anti,"q_groupings":len(qs),"ok":ledger and form and anti})
    if not (form and anti):
        T4.append({"edges":e,"ok":False,"note":"no matching form"}); T5.append({"edges":e,"ok":False,"note":"no matching form"})
    else:
        pairs=sorted({frozenset((v,match[v])) for v in match},key=sorted); pid={fs:i for i,fs in enumerate(pairs)}
        mult={}
        for a,b in e2:
            pa=pid[[fs for fs in pairs if a in fs][0]]; pb=pid[[fs for fs in pairs if b in fs][0]]
            if pa!=pb: mult[tuple(sorted((pa,pb)))]=mult.get(tuple(sorted((pa,pb))),0)+1
        m3=[k for k,v in mult.items() if v==3]; QG=nx.Graph(list(mult.keys()))
        quot=False; bond=False
        for q_,E_,pos_ in plist:
            gm=nx.algorithms.isomorphism.GraphMatcher(QG,nx.Graph(E_))
            for iso in gm.isomorphisms_iter():
                quot=True
                if len(m3)==1 and any(tuple(sorted((iso[a],iso[b])))==tuple(sorted(E_[pos_])) for a,b in m3): bond=True; break
            if bond: break
        multpat=sorted(mult.values())==[2]*(9-1)+[3]
        T4.append({"edges":e,"quotient_iso":quot,"mult_pattern":multpat,"bond_is_mult3":bond,"ok":quot and multpat and bond})
        E3={tuple(sorted(x)) for x in e2}
        defects=[Ed for Ed in E3 if tuple(sorted((match[Ed[0]],match[Ed[1]]))) not in E3]
        single=len(defects)==1
        copyint=single and all(S5R[v,v]==1 for v in defects[0])
        over=False
        if single:
            a,b=defects[0]
            proj=tuple(sorted((pid[[fs for fs in pairs if a in fs][0]],pid[[fs for fs in pairs if b in fs][0]])))
            over=len(m3)==1 and proj==m3[0]
        T5.append({"edges":e,"single":single,"copy_internal":copyint,"over_mult3":over,"ok":single and copyint and over})
    G=nx.Graph(e2); D=sp.diag(*[G.degree(v) for v in range(n)]); B=2*sp.eye(n)-D
    tr_ok=sp.simplify(sp.expand((R*D).trace())-2*s5/5)==0
    lk=frob2(simp((sp.eye(n)-P)*B*P)); lk_ok=sp.simplify(lk-sp.Rational(2,5))==0
    vx=all(sp.simplify(R[v,v]**2-sp.Rational(1,5))==0 for v in range(n))
    T6.append({"edges":e,"trace":tr_ok,"leak":str(lk),"leak_ok":lk_ok,"vertex":vx,"ok":tr_ok and lk_ok and vx})
    print(f"strict fold: T3 {T3[-1]['ok']} T4 {T4[-1]['ok'] if T4 else '-'} T5 {T5[-1]['ok'] if T5 else '-'} T6 {T6[-1]['ok']}",flush=True)
res.update({"T2":T2,"T3":T3,"T4":T4,"T5":T5,"T6":T6})
res["tests"]={"T1_zero_orphans":len(still_orphan)==0,
 "T2":all(x["ok"] for x in T2),"T3":all(x["ok"] for x in T3),"T4":all(x["ok"] for x in T4),
 "T5":all(x["ok"] for x in T5),"T6":all(x["ok"] for x in T6),
 "strict_trees":len(strict),"strict_polys":len({str(p) for _,p in strict}),
 "vacuous":len(strict)==0}
json.dump(res,open(RES/'exp_15_n20.json','w'),indent=1,default=str)
print("TESTS:",res["tests"],flush=True); print("SCORE DONE",flush=True)
