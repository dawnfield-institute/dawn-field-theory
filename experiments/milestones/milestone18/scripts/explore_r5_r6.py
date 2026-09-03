"""R5 (EXPLORING): what does the imbalance involution R do to E8's roots?
R6: are s=1 (and s=0 for path5) the ONLY golden boundary conditions? Complete solve."""
import sys, numpy as np, sympy as sp
sys.path.insert(0,'/Users/petergroom/repos/core_workspace/worktrees/dft-m18-founding/experiments/milestones/milestone18/core')
from folding import e8_roots, E8_SIMPLE, coxeter_element, eigenplane_basis

print("R5 — the golden reflection R vs the E8 root system")
W8 = coxeter_element(E8_SIMPLE)
B4 = eigenplane_basis(W8, [1, 11], 30)
P = B4.T @ B4
R = 2*P - np.eye(8)                      # = P - sigma(P)
roots = e8_roots()
img = roots @ R.T
key = {tuple(np.round(v, 8)) for v in roots}
hits = sum(tuple(np.round(v, 8)) in key for v in img)
norms = np.unique(np.round(np.linalg.norm(img, axis=1), 9))
print(f"  R(roots) that are still roots: {hits}/240   (|Rx| preserved: norms {norms})")
print("  -> R is an isometry of R^8 that does NOT belong to the Weyl/automorphism group:")
print("     it maps E8 onto a golden twin lattice. The ledger swap is invisible from")
print("     inside the root system — it acts on the FIELD, not the lattice." if hits==0 else "     UNEXPECTED — investigate")

print("\nR6 — ALL golden boundary conditions of the 4-path (complete, not sampled)")
t, s = sp.symbols('t s', real=True)
a, b, g, d = sp.symbols('alpha beta gamma delta', real=True)
# M(s): path4, interior diag 2, leaf diag 1+s
M = sp.Matrix([[1+s, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1+s]])
p = sp.expand(M.charpoly(t).as_expr())
q  = t**2 + (a + b*sp.sqrt(5))*t + (g + d*sp.sqrt(5))
qs = t**2 + (a - b*sp.sqrt(5))*t + (g - d*sp.sqrt(5))
prod = sp.expand(q*qs)
eqs = [sp.Eq(sp.expand(prod.coeff(t, k) - p.coeff(t, k)), 0) for k in range(4)]
sols = sp.solve(eqs, [a, b, g, d, s], dict=True)
golden_s = sorted({sp.simplify(so[s]) for so in sols if so.get(b, 0) != 0 or so.get(d, 0) != 0})
all_s    = sorted({sp.simplify(so[s]) for so in sols})
print("  s-values where charpoly = q*sigma(q):", all_s)
print("  ...with GENUINE golden content (beta or delta nonzero):", golden_s)
# also: golden LINEAR factor (a single eigenvalue in Q(sqrt5)\Q): p(a+b*sqrt5)=0, b!=0
x, y = sp.symbols('x y', rational=True)
expr = sp.expand(p.subs(t, x + y*sp.sqrt(5)))
rat  = sp.expand(expr.subs(sp.sqrt(5), 0) + expr.subs(sp.sqrt(5), 0))/2
c0 = sp.expand((expr + expr.subs(sp.sqrt(5), -sp.sqrt(5)))/2)
c1 = sp.expand((expr - expr.subs(sp.sqrt(5), -sp.sqrt(5)))/(2*sp.sqrt(5)))
lin = sp.solve([sp.Eq(c0, 0), sp.Eq(c1, 0)], [x, y, s], dict=True)
lin_golden = [ (so.get(x), so.get(y), sp.simplify(so[s])) for so in lin if so.get(y,0) != 0 ]
print("  golden single-eigenvalue solutions (x + y*sqrt5, s):", lin_golden if lin_golden else "none")
