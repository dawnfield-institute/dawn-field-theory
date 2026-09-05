"""Lane 3: what selects gamma = 1/phi in the construction theorem?
The parent's only free parameter is the DEFECT edge weight w (all other lifts are forced by the
cover). Redo r17 Theorem 1's decoupling argument with defect weight w and see which field appears."""
import sympy as sp
t, g, w = sp.symbols('t gamma w')

# --- symbolic: the invariance condition with a weight-w defect edge
# 5-bond lifts: (A,0)-(B,0) weight w [defect], (A,0)-(B,1) and (A,1)-(B,0) weight 1.
# (B,0) -> w(A,0) + (A,1);  (B,1) -> (A,0).  So (B,0)+g(B,1) -> (w+g)(A,0) + (A,1).
# In W_g = span{(v,0)+g(v,1)} iff (w+g, 1) proportional to (1, g): mu = w+g and 1 = mu*g.
cond = sp.expand(g * (w + g) - 1)
print("invariance condition:", sp.Eq(cond, 0), "  <-- r17 has w=1:", sp.Eq(cond.subs(w, 1), 0))
disc = sp.discriminant(sp.Poly(cond, g))
print("discriminant in gamma:", sp.factor(disc), " => the fold's field is Q(sqrt(w^2+4))")
for wv in range(0, 7):
    d = int(disc.subs(w, wv)); roots = sp.solve(cond.subs(w, wv), g)
    sq = sp.sqrt(d); nice = sp.nsimplify(roots[0]) if roots else None
    print(f"   w={wv}: disc={d:<3} field=Q(sqrt{d})  gamma={[sp.radsimp(r) for r in roots]}"
          f"{'   <== GOLDEN, simply-laced' if wv == 1 else ('   (degenerate: rational)' if d == 4 else '')}")

# --- numeric check: build parent(H2, e*) with defect weight w and factor the charpoly
def parent_matrix(wv):
    """H2 = 2 nodes joined by the 5-bond. parent = 4 nodes; at w=1 this is A4."""
    A0, B0, A1, B1 = 0, 1, 2, 3
    C = sp.eye(4) * 2
    for i, j, wt in [(A0, B0, wv), (A0, B1, 1), (A1, B0, 1)]:
        C[i, j] = C[j, i] = -wt
    return C
print("\nparent(H2) charpoly factorisation by defect weight:")
for wv in [0, 1, 2, 3]:
    C = parent_matrix(wv); p = sp.expand(C.charpoly(t).as_expr())
    d = wv * wv + 4
    K = sp.QQ.algebraic_field(sp.sqrt(d)) if not sp.sqrt(d).is_Integer else sp.QQ
    fac = sp.factor_list(p, extension=sp.sqrt(d)) if not sp.sqrt(d).is_Integer else sp.factor_list(p)
    degs = sorted(sp.degree(f, t) for f, _ in fac[1])
    print(f"   w={wv}: p = {sp.factor(p)}")
    print(f"        over Q(sqrt{d}): factor degrees {degs}  -> {'SPLITS into conjugate halves' if degs == [2,2] else 'does not split into two halves'}")
# confirm w=1 reproduces A4 and the golden halves
C = parent_matrix(1); p = sp.expand(C.charpoly(t).as_expr())
PHI = (1 + sp.sqrt(5)) / 2
H2q = sp.expand((t - (2 - PHI)) * (t - (2 + PHI)))
sq = sp.expand(H2q.subs(sp.sqrt(5), -sp.sqrt(5)))
print("\n  w=1 check: charpoly(parent) == q*sigma(q) ?", sp.simplify(p - sp.expand(H2q * sq)) == 0)
print("  and parent(H2) at w=1 is the A4 path:", sorted(C[i, j] for i in range(4) for j in range(i+1, 4)))
