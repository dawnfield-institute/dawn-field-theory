"""E-B (EXPLORING): the symmetry content of the self-dual point.
(a) Is spec M(1) self-dual (lambda <-> 4-lambda) universally?
(b) For E8: does the duality act WITHIN each Galois copy (Klein group Z2 x Z2)?
(c) Does the duality-collapsed operator (M(1)-2I)^2 = A^2 carry PURE golden pairing
    on folding diagrams and none on controls?"""
import numpy as np, sympy as sp
t = sp.Symbol('t')

def mats(n, edges):
    A = np.zeros((n, n))
    for i, j in edges: A[i, j] = A[j, i] = 1
    return A

TREES = {
  "A4-path": [(0,1),(1,2),(2,3)],
  "D6-tree": [(0,1),(1,2),(2,3),(3,4),(3,5)],
  "E8-tree": [(i,i+1) for i in range(6)]+[(2,7)],
  "D8-tree": [(i,i+1) for i in range(6)]+[(5,7)],
  "E7-tree": [(i,i+1) for i in range(5)]+[(2,6)],
  "cat8":    [(0,1),(1,2),(2,3),(1,4),(2,5),(0,6),(3,7)],
  "rand8b":  [(0,1),(0,2),(2,3),(2,4),(4,5),(4,6),(6,7)],
}
print("(a) self-duality of spec at s=1 (spec == 4 - spec as multisets):")
for nm, e in TREES.items():
    n = max(max(p) for p in e)+1
    A = mats(n, e); ev = np.sort(np.linalg.eigvalsh(2*np.eye(n)-A))
    sym = np.allclose(np.sort(4-ev), ev)
    print(f"   {nm:<8} {sym}")

print("\n(b) E8 Klein structure: duality m<->30-m maps each Galois quadruple to itself:")
H4m, CJm = {1,11,19,29}, {7,13,17,23}
print("   H4 copy closed under m->30-m:", {30-m for m in H4m} == H4m,
      "| conjugate copy closed:", {30-m for m in CJm} == CJm)
print("   => spectral symmetry group at s=1: golden diagrams Z2 x Z2 (duality x sigma),")
print("      generic trees Z2 (duality only). Goldenness = the SECOND involution.")

print("\n(c) duality-collapsed channel (A^2): golden pairing survives, alone?")
for nm, e in TREES.items():
    n = max(max(p) for p in e)+1
    A = sp.zeros(n, n)
    for i, j in e: A[i, j] = A[j, i] = 1
    f = sp.factor(sp.expand((A*A).charpoly(t).as_expr()), extension=sp.sqrt(5))
    golden = any(g.has(sp.sqrt(5)) for g in sp.Mul.make_args(f))
    print(f"   {nm:<8} A^2 charpoly golden over Q(sqrt5): {golden}")
