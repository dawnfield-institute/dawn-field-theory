"""R4 (EXPLORING): scope-check the Complement Identity.
Conjecture: for ANY rational symmetric matrix whose spectrum splits into Q(sqrt5)-conjugate
halves, the projector onto either half satisfies sigma(P) = I - P automatically.
Proof sketch: P = f(M) for the Lagrange interpolation polynomial f selecting the half;
f's coefficients lie in Q(sqrt5); entrywise sigma commutes with polynomials in rational M,
and sigma(f) selects the conjugate half, so sigma(P) = complement projector. Verify on
instances, including the SMALLEST one."""
import sympy as sp
t = sp.Symbol('t')

def proj_half(M, half):
    """Exact spectral projector onto the eigenvalues in `half` via Lagrange interpolation."""
    evs = list(M.eigenvals())
    f = sp.Integer(0)
    for lam in half:
        term = sp.Integer(1)
        for mu in evs:
            if sp.simplify(mu - lam) != 0:
                term = term * (t - mu) / (lam - mu)
        f = f + term
    f = sp.expand(sp.simplify(f))
    P = sp.zeros(*M.shape)
    for k, c in enumerate(sp.Poly(f, t).all_coeffs()[::-1]):
        P = P + c * (M**k)
    return P.applyfunc(lambda x: sp.simplify(sp.radsimp(x)))

def check(M, name):
    evs = list(M.eigenvals())
    golden = [e for e in evs if sp.nsimplify(e).has(sp.sqrt(5))]
    half = []
    for e in golden:
        if all(sp.simplify(e - h) != 0 and sp.simplify(e - h.subs(sp.sqrt(5), -sp.sqrt(5))) != 0 for h in half):
            half.append(e)
    P = proj_half(M, half)
    sP = P.subs(sp.sqrt(5), -sp.sqrt(5))
    comp = sp.simplify(sP - (sp.eye(M.shape[0]) - P)) == sp.zeros(*M.shape)
    print(f"  {name}: sigma(P) = I - P: {comp}   (half = {[sp.nsimplify(h) for h in half]})")
    return P

print("SMALLEST INSTANCE — Andy's own Q-matrix (Fibonacci, his section 2.9):")
Q = sp.Matrix([[1, 1], [1, 0]])   # spectrum {phi, psi}: a conjugate pair
P = check(Q, "Q = [[1,1],[1,0]]")
print("    P (projector onto the phi-eigenspace):"); sp.pprint(sp.simplify(P))

print("\nRANDOM rational-symmetric instance with golden spectrum (premise check):")
# build one: block-diagonal in a rotated rational frame is hard to keep rational; instead
# take a rational symmetric matrix and CHECK whether its golden eigenvalues obey the law.
M = sp.Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
evs = list(M.eigenvals())
print("  M eigenvalues:", [sp.nsimplify(e) for e in evs])
try:
    check(M, "M 3x3")
except Exception as ex:
    print("  (", ex, ")")

print("\nA4 CARTAN as instance (should reproduce tonight's identity in root coordinates):")
C4 = sp.Matrix(4, 4, lambda i, j: 2 if i == j else (-1 if abs(i - j) == 1 else 0))
P4 = check(C4, "Cartan(A4)")
print("\nCONCLUSION IF ALL TRUE: the Complement Identity is a GENERAL law of rational")
print("operators with conjugate-split spectra. The foldings' special content is Result 2:")
print("WHICH diagrams have such spectra — exactly {A4, D6, E8}.")
