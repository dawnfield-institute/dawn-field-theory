"""R2 (EXPLORING): boundary knife-edge — golden structure along M(s) = (1-s)L + sC.
R3: the conjugate-ledger landscape across all ADE Cartans."""
import sympy as sp
t, s = sp.symbols('t s'); phi = (1+sp.sqrt(5))/2

def mats(n, edges):
    A = sp.zeros(n, n)
    for i, j in edges: A[i, j] = A[j, i] = 1
    D = sp.diag(*[sum(A[i, k] for k in range(n)) for i in range(n)])
    L = D - A; C = 2*sp.eye(n) - A
    return L, C

def shape(M):
    f = sp.factor(sp.expand(M.charpoly(t).as_expr()), extension=sp.sqrt(5))
    degs = []
    for g in sp.Mul.make_args(f):
        if g.has(t):
            b, e = g.as_base_exp()
            degs += [sp.degree(b, t)] * int(e)
    gold = any(g.has(sp.sqrt(5)) for g in sp.Mul.make_args(f))
    return sorted(degs), gold

DIAGRAMS = {
  "path4 (A4)":  (4, [(0,1),(1,2),(2,3)]),
  "path5 (A5)":  (5, [(0,1),(1,2),(2,3),(3,4)]),
  "D6-tree":     (6, [(0,1),(1,2),(2,3),(3,4),(3,5)]),
  "E8-tree":     (8, [(i,i+1) for i in range(6)]+[(2,7)]),
}
print("R2 — factor shape over Q(sqrt5) [golden marked *] along s = 0, 1/4, 1/2, 3/4, 1")
for name, (n, e) in DIAGRAMS.items():
    L, C = mats(n, e)
    row = []
    for sv in [0, sp.Rational(1,4), sp.Rational(1,2), sp.Rational(3,4), 1]:
        M = (1-sv)*L + sv*C
        degs, gold = shape(M)
        row.append(("*" if gold else " ") + str(degs))
    print(f"  {name:<12} " + " | ".join(f"{r:<14}" for r in row))

print("\nR3 — the conjugate-ledger landscape (Cartan matrices; q*sigma(q) with H-factor?)")
H3C = sp.Matrix([[2,-1,0],[-1,2,-phi],[0,-phi,2]])
H4C = sp.Matrix([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-phi],[0,0,-phi,2]])
H2C = sp.Matrix([[2,-phi],[-phi,2]])
def cart(n, edges):
    C = sp.zeros(n, n)
    for i in range(n): C[i, i] = 2
    for i, j in edges: C[i, j] = C[j, i] = -1
    return C
def conj_ledger(Cm, Hm):
    p = sp.expand(Cm.charpoly(t).as_expr())
    q = sp.expand(Hm.charpoly(t).as_expr())
    qs = sp.expand(q.subs(sp.sqrt(5), -sp.sqrt(5)))
    return sp.simplify(sp.expand(q*qs) - p) == 0
print("  A4 = charpoly(H2)*sigma(charpoly(H2)):", conj_ledger(cart(4,[(0,1),(1,2),(2,3)]), H2C))
print("  D6 = charpoly(H3)*sigma(charpoly(H3)):", conj_ledger(cart(6,[(0,1),(1,2),(2,3),(3,4),(3,5)]), H3C))
print("  E8 = charpoly(H4)*sigma(charpoly(H4)):", conj_ledger(cart(8,[(i,i+1) for i in range(6)]+[(2,7)]), H4C))
print("  negative family (no H-partner) — golden content of Cartan charpoly over Q(sqrt5):")
for name, n, e in [("A5",5,[(0,1),(1,2),(2,3),(3,4)]), ("D4",4,[(0,1),(1,2),(1,3)]),
                   ("D5",5,[(0,1),(1,2),(2,3),(2,4)]), ("E6",6,[(0,1),(1,2),(2,3),(3,4),(2,5)]),
                   ("E7",7,[(i,i+1) for i in range(5)]+[(2,6)])]:
    degs, gold = shape(cart(n, e))
    print(f"    {name}: shape {degs}, golden = {gold}")
