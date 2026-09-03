#!/usr/bin/env python3
"""explore_r16: fast golden-grader prototype for the n=20 tier. Strategy: integer charpoly once,
cheap rational-factor screens first, the expensive Q(sqrt5) factorization only on survivors.
Validated against exp_11's known n=14 grade counts (per distinct charpoly: strict 0, core 50,
partial 153, none 2520) before any use at 20. Prints stage timings for the n=20 estimate."""
import time, sys, sympy as sp, networkx as nx
t=sp.Symbol('t'); s5=sp.sqrt(5)
n=int(sys.argv[1]) if len(sys.argv)>1 else 14
def charpoly_int(A):
    # A: adjacency sets; Cartan charpoly via sympy on an int Matrix (Berkowitz) — baseline
    return sp.expand((2*sp.eye(len(A))-sp.Matrix(len(A),len(A),lambda i,j: 1 if j in A[i] else 0)).charpoly(t).as_expr())
def grade_fast(p):
    fQ=sp.factor_list(p)[1]
    gold_cand=[]; rat_even=True; any_gold=False
    for b,ex in fQ:
        d=sp.degree(b,t)
        if d%2==1 and d>1: return "none"          # odd-degree irrational factor cannot sigma-pair
        if d==1:
            if ex%2==1: rat_even=False
            continue
        # even-degree rational-irreducible factor: does it split over Q(sqrt5)?
        gold_cand.append((b,ex))
    if not gold_cand: return "none"
    # expensive step only here
    f=sp.factor(p,extension=s5); facs=[g for g in sp.Mul.make_args(f) if g.has(t)]
    gold=[g for g in facs if g.has(s5)]; rat=[g for g in facs if not g.has(s5)]
    if not gold: return "none"
    if not rat: return "strict"
    return "core" if all(g.as_base_exp()[1]%2==0 for g in rat) else "partial"
t0=time.time(); polys={}; cnt=0; t_char=0.0
for T in nx.nonisomorphic_trees(n):
    A={v:set(T[v]) for v in T}
    ta=time.time(); p=charpoly_int(A); t_char+=time.time()-ta
    polys.setdefault(p,None); cnt+=1
print(f"n={n}: {cnt} trees, {len(polys)} distinct charpolys  [{time.time()-t0:.0f}s total, {t_char:.0f}s in charpoly]",flush=True)
t1=time.time(); grades={}
t_screen=0.0; t_ext=0.0; ext_calls=0
for p in polys:
    fQ=sp.factor_list(p)[1]
    quick=None
    if all(sp.degree(b,t)==1 for b,_ in fQ): quick="none"
    elif any(sp.degree(b,t)%2==1 and sp.degree(b,t)>1 for b,_ in fQ): quick="none"
    if quick: grades[p]=quick; continue
    ext_calls+=1; grades[p]=grade_fast(p)
gc={g:sum(1 for x in grades.values() if x==g) for g in ("strict","core","partial","none")}
print(f"grading: {time.time()-t1:.0f}s, extension factorizations needed: {ext_calls}/{len(polys)}",flush=True)
print("grade counts (distinct charpolys):",gc,flush=True)
if n==14:
    expect={"strict":0,"core":50,"partial":153,"none":2520}
    print("MATCHES exp_11 known answer:",gc==expect,flush=True)
print("DONE",flush=True)
