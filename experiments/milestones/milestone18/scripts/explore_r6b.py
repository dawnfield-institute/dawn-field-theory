"""R6b (EXPLORING): dense exact map of golden boundary points, paths n=4,5 and E8-tree.
Rational s with denominator <= 10 in [-1, 2]. Exact factorization at every point."""
import sympy as sp
from fractions import Fraction
t = sp.Symbol('t')

def golden_at(n_verts, edges, sv):
    A = sp.zeros(n_verts, n_verts)
    for i, j in edges: A[i, j] = A[j, i] = 1
    D = sp.diag(*[sum(A[i, k] for k in range(n_verts)) for i in range(n_verts)])
    M = (D - A) + sp.Rational(sv.numerator, sv.denominator) * (2*sp.eye(n_verts) - D)
    f = sp.factor(sp.expand(M.charpoly(t).as_expr()), extension=sp.sqrt(5))
    return any(g.has(sp.sqrt(5)) for g in sp.Mul.make_args(f))

grid = sorted({Fraction(p, q) for q in range(1, 11) for p in range(-q, 2*q + 1)})
print(f"grid: {len(grid)} exact rationals in [-1, 2]")
cases = {"path4": (4, [(0,1),(1,2),(2,3)]),
         "path5": (5, [(0,1),(1,2),(2,3),(3,4)]),
         "E8-tree": (8, [(i,i+1) for i in range(6)]+[(2,7)])}
for name, (n, e) in cases.items():
    hits = [sv for sv in grid if golden_at(n, e, sv)]
    print(f"  {name:<8} golden s-values: {[str(h) for h in hits]}")
