"""Exact sigma-ledger projectors (Bezout construction; D6 core by golden gauge)."""
import sympy as sp
t = sp.Symbol('t'); PHI = (1+sp.sqrt(5))/2; K = sp.QQ.algebraic_field(sp.sqrt(5))

class DegenerateFoldError(ValueError):
    """q and sigma(q) share a root: no Bezout projector exists (core-grade or repeated pair). Callers declare, never score."""

def cart(n, edges):
    C = sp.zeros(n, n)
    for i in range(n): C[i, i] = 2
    for i, j in edges: C[i, j] = C[j, i] = -1
    return C
def simp(M): return M.applyfunc(lambda x: sp.radsimp(sp.expand(x)))
def sigma(M): return M.applyfunc(lambda x: sp.expand(x.subs(sp.sqrt(5), -sp.sqrt(5))))
def _polyval(coeffs, C):
    P = sp.zeros(*C.shape)
    for c in coeffs: P = simp(P*C + c*sp.eye(C.shape[0]))
    return P
def bezout_proj(C, q):
    sq = sp.expand(q.subs(sp.sqrt(5), -sp.sqrt(5)))
    Q, SQ = sp.Poly(q, t, domain=K), sp.Poly(sq, t, domain=K)
    a, b, g = sp.gcdex(Q, SQ)
    if g.degree() != 0:
        raise DegenerateFoldError(f"gcd(q, sigma q) has degree {g.degree()}")
    pp = (b*SQ)/g.all_coeffs()[0]
    return _polyval([sp.expand(c.as_expr()) for c in sp.Poly(pp.as_expr(), t).all_coeffs()], C)

EDGES = {"A4": (4, [(0,1),(1,2),(2,3)]), "D6": (6, [(0,1),(1,2),(2,3),(3,4),(3,5)]),
         "E8": (8, [(i,i+1) for i in range(6)]+[(2,7)])}
H2q = sp.expand((t-(2-PHI))*(t-(2+PHI)))
H3 = sp.Matrix([[2,-1,0],[-1,2,-PHI],[0,-PHI,2]])
H4 = sp.Matrix([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-PHI],[0,0,-PHI,2]])

def projector(name, tau=6-3*sp.sqrt(5)):
    """Exact H-copy projector for A4/D6/E8 (D6 needs a golden core gauge; default = exp_06's)."""
    n, e = EDGES[name]; C = cart(n, e)
    if name == "A4": return C, bezout_proj(C, H2q)
    if name == "E8": return C, bezout_proj(C, sp.expand(H4.charpoly(t).as_expr()))
    q_off = sp.expand(sp.cancel(sp.expand(H3.charpoly(t).as_expr())/(t-2)))
    P_off = bezout_proj(C, q_off)
    core = (C - 2*sp.eye(6)).nullspace(); V = sp.Matrix.hstack(*core)
    Qc = simp(V*(V.T*V).inv()*V.T); P_off = simp(P_off*(sp.eye(6)-Qc))
    v = core[0] + tau*core[1]; Pv = simp((v*v.T)/(v.T*v)[0])
    return C, simp(P_off + Pv)
