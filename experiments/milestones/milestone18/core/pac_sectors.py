"""Milestone 18 core — the exact SECTOR ROUTE for the fold certificate on complete PAC binary trees
(Block D draft §2.7). The Cartan of the depth-d tree splits under Aut into sqrt2-weighted radial paths:
one radial sector (path of d+1 nodes, multiplicity 1) and, for every internal node x at level l, an
antisymmetric sector (path of d-l nodes, multiplicity 2^l). Level-diagonal operators (D, B = 2I - D)
act on each sector as k x k diagonals, so tr(R·D), the leaks and the vertex multiset of the full tree
are sums of k x k computations (k <= d+1), each done by the Bezout projector on the small path Cartan
over Q(sqrt2, sqrt5). Gated against the full-tree Bezout route at d = 3, 4, 5 before use."""
import sympy as sp
from ledger import DegenerateFoldError
t = sp.Symbol('t'); s5 = sp.sqrt(5); r2 = sp.sqrt(2)
K2 = sp.QQ.algebraic_field(sp.sqrt(2), sp.sqrt(5))


def sig(e): return sp.expand(e.subs(s5, -s5))


def path_cartan(k):
    C = 2 * sp.eye(k)
    for i in range(k - 1): C[i, i + 1] = C[i + 1, i] = -r2
    return C


def small_projector(Ck, q):
    """Projector onto ker q(Ck) as a polynomial in Ck, over Q(sqrt2, sqrt5). Returns None if q has no root in this sector."""
    p = sp.expand(Ck.charpoly(t).as_expr()); k = Ck.shape[0]
    Q = sp.Poly(q, t, domain=K2); P = sp.Poly(p, t, domain=K2)
    g = sp.gcd(P, Q)
    if g.degree() == 0: return None
    qs = sp.Poly(g.as_expr(), t, domain=K2)                       # the part of q realized in this sector
    CF, rem = P.div(qs); assert rem.is_zero
    a, b, gg = sp.gcdex(qs, CF)
    if gg.degree() != 0: raise DegenerateFoldError("sector: q shares a root with its cofactor")
    poly = sp.Poly(sp.expand(((b * CF) / gg.all_coeffs()[0]).as_expr()), t)
    Pm = sp.zeros(k, k)
    for c in poly.all_coeffs(): Pm = (Pm * Ck + sp.expand(sp.sympify(c)) * sp.eye(k)).applyfunc(lambda x: sp.radsimp(sp.expand(x)))
    return Pm


def sectors(d):
    """[(k, multiplicity, level_of_root, degrees_by_depth)] — degrees of the tree vertices at each depth j of the sector.
    Radial sector: j = 0..d (root degree 2, internal 3, leaves 1). Antisymmetric sector at level l: j = 1..d-l, levels l+j."""
    out = []
    out.append((d + 1, 1, None, [2 if j == 0 else (3 if j < d else 1) for j in range(d + 1)]))
    for l in range(d):
        k = d - l; out.append((k, 2 ** l, l, [3 if l + j < d else 1 for j in range(1, k + 1)]))
    return out


def pac_certificate(d, q):
    """tr(R·D), off->off leak, total leak, vertex multiset {R_vv^2} for the golden pair q on the depth-d tree."""
    trRD = sp.Integer(0); leak_oo = sp.Integer(0); leak_tot = sp.Integer(0)
    n = 2 ** (d + 1) - 1
    # vertex R_vv: a vertex at level L gets 2^-L * (R_rad)_LL from the radial sector and, for each ancestor level l < L,
    # 2^-(L-l) * (R_l)_{jj} with j = L - l from that ancestor's antisymmetric sector (all vertices at a level are alike).
    Rdiag = {}
    for k, m, l, degs in sectors(d):
        Ck = path_cartan(k); P = small_projector(Ck, q); sP = small_projector(Ck, sig(q))
        if P is None and sP is None: continue
        if P is None: P = sp.zeros(k, k)
        if sP is None: sP = sp.zeros(k, k)
        R = (P - sP).applyfunc(lambda x: sp.radsimp(sp.expand(x)))
        Dk = sp.diag(*degs); Bk = 2 * sp.eye(k) - Dk
        trRD += m * (R * Dk).trace()
        X = (sP * Bk * P).applyfunc(lambda x: sp.radsimp(sp.expand(x))); leak_oo += m * sum(x ** 2 for x in X)
        Y = ((sp.eye(k) - P) * Bk * P).applyfunc(lambda x: sp.radsimp(sp.expand(x))); leak_tot += m * sum(y ** 2 for y in Y)
        Rdiag[(k, l)] = [R[j, j] for j in range(k)]
    vertex = {}
    for L in range(d + 1):
        val = sp.Rational(1, 2 ** L) * Rdiag.get((d + 1, None), [0] * (d + 1))[L] if (d + 1, None) in Rdiag else sp.Integer(0)
        for l in range(L):
            key = (d - l, l)
            if key in Rdiag: val += sp.Rational(1, 2 ** (L - l)) * Rdiag[key][L - l - 1]
        vertex[L] = sp.radsimp(sp.expand(val))
    vsq = sorted({str(sp.nsimplify(sp.radsimp(sp.expand(v ** 2)))) for v in vertex.values()})
    return dict(trRD=sp.nsimplify(sp.radsimp(sp.expand(trRD))), leak_oo=sp.nsimplify(sp.radsimp(sp.expand(leak_oo))),
                leak_total=sp.nsimplify(sp.radsimp(sp.expand(leak_tot))), vertex_sq=vsq, vertex_by_level={L: str(v) for L, v in vertex.items()}, n=n)
