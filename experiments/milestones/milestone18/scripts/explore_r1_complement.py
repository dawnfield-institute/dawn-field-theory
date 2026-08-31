"""R1 (EXPLORING): Is Galois conjugation = complementation on the folding projector?
Claim to test: sigma(P) = I - P, where P projects onto the H4 subspace of E8
and sigma is the Q(sqrt5) conjugation applied entrywise."""
import sys, numpy as np, sympy as sp
sys.path.insert(0,'/Users/petergroom/repos/core_workspace/worktrees/dft-m18-founding/experiments/milestones/milestone18/core')
from folding import e8_roots, E8_SIMPLE, coxeter_element, eigenplane_basis, PHI

W8 = coxeter_element(E8_SIMPLE)
B4 = eigenplane_basis(W8, [1, 11], 30)
P = B4.T @ B4

# recognize every entry exactly as (a + b*sqrt5)/c, then conjugate symbolically
s5f = 5**0.5
def rec(x, maxden=64, tol=1e-9):
    for c in range(1, maxden+1):
        y = x*c
        for b in range(-3*c, 3*c+1):
            a = round(y - b*s5f)
            if abs(a + b*s5f - y) < tol*c: return sp.Rational(a,c) + sp.Rational(b,c)*sp.sqrt(5)
    return None
Ps = sp.Matrix(8, 8, lambda i, j: rec(float(P[i, j])))
assert all(Ps[i,j] is not None for i in range(8) for j in range(8)), "recognition failed"
sigmaP = Ps.subs(sp.sqrt(5), -sp.sqrt(5))

print("EXACT CHECKS (sympy, no floats):")
print("  P is idempotent (P^2 = P):        ", sp.simplify(Ps*Ps - Ps) == sp.zeros(8,8))
print("  sigma(P) = I - P:                 ", sp.simplify(sigmaP - (sp.eye(8) - Ps)) == sp.zeros(8,8))
print("  => P + sigma(P) = I  (ledger: P + A = C, C = identity)")
print("  => P * sigma(P) = 0:              ", sp.simplify(Ps*sigmaP) == sp.zeros(8,8))
R = Ps - sigmaP
print("  R = P - sigma(P) is pure-golden (rational part 0 — the Delta channel):",
      all(sp.simplify(R[i,j] + R[i,j].subs(sp.sqrt(5),-sp.sqrt(5))) == 0 for i in range(8) for j in range(8)))
print("  R^2 = I (the imbalance operator is an involution):", sp.simplify(R*R - sp.eye(8)) == sp.zeros(8,8))
print("  tr(P) = ", sp.simplify(sp.trace(Ps)), "  (4 = half of 8: the split is exactly half the identity)")
print()
print("Andy's Vieta, lifted to operators:")
print("  numbers:    phi + psi = 1,     phi * psi = -1")
print("  projectors: P + sigmaP = I,    P * sigmaP = 0,    (P - sigmaP)^2 = I")
print()
# Same identity for A4 -> H2
a4s = np.array([[1,-1,0,0,0],[0,1,-1,0,0],[0,0,1,-1,0],[0,0,0,1,-1]], float)
W4 = coxeter_element(a4s); B2 = eigenplane_basis(W4, [1], 5)
P4 = B2.T @ B2
P4s = sp.Matrix(5, 5, lambda i, j: rec(float(P4[i, j])))
ok = all(P4s[i,j] is not None for i in range(5) for j in range(5))
if ok:
    s4 = P4s.subs(sp.sqrt(5), -sp.sqrt(5))
    # A4 lives in the 4D subspace of R^5 orthogonal to (1,1,1,1,1): 'identity' there is
    # the projector onto that subspace, not eye(5)
    ones = sp.ones(5,1)/sp.sqrt(5)
    Isub = sp.eye(5) - sp.Matrix(5,5, lambda i,j: sp.Rational(1,5))
    print("A4 -> H2 (in the 4D root space of R^5):")
    print("  sigma(P) = I_sub - P:", sp.simplify(s4 - (Isub - P4s)) == sp.zeros(5,5))
else:
    print("A4: entry recognition failed (report honestly)")
