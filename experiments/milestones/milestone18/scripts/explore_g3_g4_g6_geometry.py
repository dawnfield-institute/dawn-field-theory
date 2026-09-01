#!/usr/bin/env python3
"""Panel G parts 3/4/6 (EXPLORING): signature decomposition of C over the golden copies,
fold-partner search (does the copy's polynomial come from a golden Coxeter-type diagram?),
and Galois structure of the hyperbolic hits' rational quartics."""
import sys, json, itertools, numpy as np, sympy as sp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent/"core"))
from ledger import bezout_proj, simp, sigma, cart, projector
t=sp.Symbol('t'); s5=sp.sqrt(5); phi=(1+s5)/2
out={}

TREES={"A4":(4,[(0,1),(1,2),(2,3)]),"D6":(6,[(0,1),(1,2),(2,3),(3,4),(3,5)]),
       "E8":(8,[(i,i+1) for i in range(6)]+[(2,7)]),"cat8":(8,[(0,6),(2,1),(5,3),(6,1),(1,3),(3,4),(4,7)])}

def golden_pairs(C):
    p=sp.expand(C.charpoly(t).as_expr()); f=sp.factor(p,extension=s5)
    facs=[g for g in sp.Mul.make_args(f) if g.has(s5)]
    pairs=[]; used=set()
    for g in facs:
        b,e=g.as_base_exp(); key=str(sp.expand(b))
        if key in used: continue
        cb=sp.expand(b.subs(s5,-s5)); used.add(key); used.add(str(cb)); pairs.append((sp.expand(b),cb,e))
    return p,pairs

def signature(M, tol=1e-9):
    ev=np.linalg.eigvalsh(np.array(M.evalf(30).tolist(),dtype=float))
    return (int((ev>tol).sum()), int((ev<-tol).sum()), int((abs(ev)<=tol).sum()))

# ---------- G3: signature split ----------
print("G3 — signature of C on the copy vs its conjugate")
for nm,(n,e) in TREES.items():
    C=cart(n,e); p,pairs=golden_pairs(C)
    out[nm]={"sig_C":signature(C),"det":int(sp.Poly(p,t).all_coeffs()[-1]*(-1)**n),"choices":[]}
    # q choices: one member from each conjugate pair (up to global sigma)
    choices=[]
    for bits in itertools.product([0,1], repeat=len(pairs)):
        if bits and bits[0]==1: continue                      # fix global sigma
        q=sp.Integer(1)
        for (b,cb,ex),bit in zip(pairs,bits): q*= (cb if bit else b)**ex
        choices.append(sp.expand(q))
    for q in choices:
        if nm=="D6":
            C,P=projector("D6")                                 # gauge-resolved core
        else:
            P=bezout_proj(C,q)
        Pc=simp(P*C*P); Pcc=simp((sp.eye(n)-P)*C*(sp.eye(n)-P))
        rec={"q":str(q),"sig_on_copy":signature(Pc),"sig_on_conjugate":signature(Pcc),
             "complement_identity":bool(simp(sigma(P)-(sp.eye(n)-P))==sp.zeros(n,n))}
        out[nm]["choices"].append(rec)
        print(f"  {nm}: C {out[nm]['sig_C']} det {out[nm]['det']} | q={str(q)[:38]:<38} copy {rec['sig_on_copy']} conj {rec['sig_on_conjugate']} sigma(P)=I-P {rec['complement_identity']}")
        if nm=="D6": break

# ---------- G4: fold-partner search ----------
print("\nG4 — fold partner: a k-node symmetric matrix, diag 2, golden bonds, with charpoly = q ?")
BONDS=[0, -1, -phi, -(phi-1), -phi**2, -s5, -2]
def search(q, k):
    pairs_idx=list(itertools.combinations(range(k),2))
    qp=sp.Poly(sp.expand(q),t); target=[complex(r) for r in qp.nroots()]
    tv=sorted(x.real for x in target)
    found=[]
    for w in itertools.product(range(len(BONDS)), repeat=len(pairs_idx)):
        if all(x==0 for x in w): continue
        M=np.full((k,k),2.0)
        Ms=2*sp.eye(k)
        for (i,j),b in zip(pairs_idx,w):
            M[i,j]=M[j,i]=float(BONDS[b]); Ms[i,j]=Ms[j,i]=BONDS[b]
        for i in range(k): M[i,i]=2.0
        ev=sorted(np.linalg.eigvalsh(M))
        if np.allclose(ev,tv,atol=1e-7):
            if sp.simplify(sp.expand(Ms.charpoly(t).as_expr())-sp.expand(q))==0:
                found.append([[str(Ms[i,j]) for j in range(k)] for i in range(k)])
                if len(found)>=3: break
    return found
for nm,(n,e) in TREES.items():
    C=cart(n,e); p,pairs=golden_pairs(C)
    k=n//2
    facsQ=[g for g in sp.Mul.make_args(sp.factor(p)) if g.has(t) and not sp.factor(g,extension=s5).has(s5)]
    core=sp.Integer(1)
    for g in facsQ:
        b,ex=g.as_base_exp(); core*=b**(ex//2)
    for rec in out[nm]["choices"][:2]:
        q=sp.expand(sp.sympify(rec["q"])*core); rec["q_padded"]=str(q); f=search(q,k)
        rec["fold_partners"]=f
        print(f"  {nm}: q={str(q)[:34]:<34} -> {len(f)} golden {k}-node partner(s) found" + (f"; first: {f[0]}" if f else ""))

# ---------- G6: Galois structure of the rational quartics ----------
print("\nG6 — Galois groups / discriminants of the Q-irreducible factors")
from sympy.polys.numberfields.galoisgroups import galois_group
for nm,(n,e) in TREES.items():
    C=cart(n,e); p=sp.expand(C.charpoly(t).as_expr())
    facsQ=[g for g in sp.Mul.make_args(sp.factor(p)) if g.has(t)]
    out[nm]["galois"]=[]
    for g in facsQ:
        b,ex=g.as_base_exp(); d=sp.degree(b,t)
        disc=sp.discriminant(b,t)
        gg=None
        if 2<=d<=6:
            try: gg=str(galois_group(sp.Poly(b,t))[0])
            except Exception as ex_: gg=f"n/a ({type(ex_).__name__})"
        rec={"factor":str(b),"degree":int(d),"mult":int(ex),"disc":str(sp.factorint(disc)) if disc!=0 else "0","galois":gg}
        out[nm]["galois"].append(rec)
        print(f"  {nm}: deg {d} x{ex}  disc={rec['disc']}  Gal={gg}")
Path(__file__).parent.parent.joinpath("results","explore_g3g4g6_20260901.json").write_text(json.dumps(out,indent=1,default=str))
