"""Milestone 18 core — the fold certificate (Block D instrument; spec: journals/2026-09-02_blockD_registration_DRAFT.md §2).

Everything exact over Q(sqrt5). The spectral projector onto ker q(C) is built as a POLYNOMIAL in C
from the Bezout identity of q against its cofactor p/q — no nullspaces, no conic, no radicals — so
rational cores are killed automatically (Lemma 1 of the provenance journal is a consequence, not an
input). Gated (scripts/explore_d0_certificate_gates.py) against the exp_14 recipe on D6 and the
exp_13 core folds, and against KA1–KA7, before any object is scored.
"""
import itertools, sympy as sp, networkx as nx
from ledger import simp, cart, DegenerateFoldError
t = sp.Symbol('t'); s5 = sp.sqrt(5); phi = (1 + s5) / 2
K = sp.QQ.algebraic_field(sp.sqrt(5))

__all__ = ["t", "s5", "phi", "K", "sig", "sigma", "grade", "sector_projector_dm", "golden_pairs", "rational_cores", "halves", "sector_projector",
           "certificate", "class_sectors", "is_regular", "evaluate"]


def sig(e): return sp.expand(e.subs(s5, -s5))
def sigma(M): return M.applyfunc(lambda x: sp.expand(x.subs(s5, -s5)))


# ---- grade (ported verbatim from scripts/explore_g1_census_exhaustive.py, d = 5) -------------------
def grade(p, d=5):
    """(grade, golden_factor_list, rational_factor_list, norm_signature) over Q(sqrt d): strict = complete
    pairing, core = rational factors all to even multiplicity, partial = some golden content, none."""
    sd = sp.sqrt(d)
    facsQ = [g for g in sp.Mul.make_args(sp.factor(p)) if g.has(t)]
    for g in facsQ:
        b, e = g.as_base_exp()
        if sp.degree(b, t) % 2 == 1 and sp.degree(b, t) > 1:
            return "none", [], [], []
    facs = [g for g in sp.Mul.make_args(sp.factor(p, extension=sd)) if g.has(t)]
    gold = [g for g in facs if g.has(sd)]; rat = [g for g in facs if not g.has(sd)]
    if not gold:
        return "none", [], [str(r) for r in rat], []
    sigs = []; seen = set()
    for g in gold:
        b, e = g.as_base_exp(); c0 = sp.Poly(b, t).all_coeffs()[-1]
        nrm = sp.simplify(sp.expand(c0 * c0.subs(sd, -sd)))
        key = str(sp.expand(b)); conj = str(sp.expand(b.subs(sd, -sd)))
        if conj not in seen:
            seen.add(key); sigs.append(str(nrm))
    if not rat:
        return "strict", [str(g) for g in gold], [], sigs
    even = all((g.as_base_exp()[1] % 2 == 0) for g in rat)
    return ("core" if even else "partial"), [str(g) for g in gold], [str(r) for r in rat], sigs


def golden_pairs(p):
    """sigma-conjugate irreducible golden factors of p over Q(sqrt5), with multiplicity: [(g, sigma g, m)]."""
    facs = {}
    for g in sp.Mul.make_args(sp.factor(p, extension=s5)):
        if g.has(t) and g.has(s5):
            b, e = g.as_base_exp(); facs[str(sp.expand(b))] = (sp.expand(b), int(e))
    pairs, used = [], set()
    for key, (b, m) in facs.items():
        if key in used: continue
        ck = str(sig(b))
        if ck in facs and ck != key:
            pairs.append((b, facs[ck][0], min(m, facs[ck][1]))); used |= {key, ck}
        else:
            pairs.append((b, None, m)); used.add(key)      # sigma-fixed golden factor (should not occur)
    return pairs


def rational_cores(p):
    """Rational irreducible factors of p with multiplicity: [(r, m)]."""
    return [(sp.expand(g.as_base_exp()[0]), int(g.as_base_exp()[1]))
            for g in sp.Mul.make_args(sp.factor(p, extension=s5)) if g.has(t) and not g.has(s5)]


def halves(p):
    """All golden halves q (products of one factor from each conjugate pair, to full multiplicity), up to
    complement. Rational factors are NOT included (they are cores). Returns [] if no golden content."""
    pairs = [(g, sg, m) for g, sg, m in golden_pairs(p) if sg is not None]
    if not pairs: return []
    out = []
    for choice in itertools.product([0, 1], repeat=len(pairs) - 1):
        q = pairs[0][0] ** pairs[0][2]
        for c, (g, sg, m) in zip(choice, pairs[1:]):
            q *= (g if c == 0 else sg) ** m
        out.append(sp.expand(q))
    return out


def sector_projector(C, q, p):
    """The spectral projector onto ker q(C), as a polynomial in C: with a*q + b*(p/q) = 1 over Q(sqrt5)[t],
    P = b(C)*(p/q)(C). Kills every other eigenspace (including rational cores) automatically.
    Raises DegenerateFoldError when q and its cofactor share a root (repeated golden pair)."""
    Q = sp.Poly(q, t, domain=K)
    CF, rem = sp.Poly(sp.expand(p), t, domain=K).div(Q)          # exact division over Q(sqrt5)
    if not rem.is_zero:
        raise ValueError("q does not divide p over Q(sqrt5)")
    a, b, g = sp.gcdex(Q, CF)
    if g.degree() != 0:
        raise DegenerateFoldError(f"gcd(q, p/q) has degree {g.degree()}")
    g0 = g.all_coeffs()[0]
    poly = sp.Poly(sp.expand(((b * CF) / g0).as_expr()), t)
    n = C.shape[0]; P = sp.zeros(n, n)
    for c in poly.all_coeffs():
        P = simp(P * C + sp.expand(sp.sympify(c)) * sp.eye(n))
    return P


def is_regular(D):
    d = [D[i, i] for i in range(D.shape[0])]
    return all(x == d[0] for x in d)


def certificate(C, D, P):
    """tr(R·D), off->off leak ||sigma(P) B P||_F^2, total leak ||(I-P) B P||_F^2, vertex multiset {R_vv^2},
    with R = P - sigma(P), B = 2I - D. Exact."""
    n = C.shape[0]; B = 2 * sp.eye(n) - D
    R = simp(P - sigma(P))
    trRD = sp.nsimplify(sp.expand((R * D).trace()))
    X = simp(sigma(P) * B * P); leak_oo = sp.nsimplify(sp.expand(sum(x ** 2 for x in X)))
    Y = simp((sp.eye(n) - P) * B * P); leak_tot = sp.nsimplify(sp.expand(sum(y ** 2 for y in Y)))
    vv = sorted({str(sp.nsimplify(sp.expand(R[v, v] ** 2))) for v in range(n)})
    return dict(trRD=trRD, leak_oo=leak_oo, leak_total=leak_tot, vertex_sq=vv, R=R)


def class_sectors(p):
    """Classify each golden pair up to uniform bond scaling s (bond magnitude = coordinate; class = relation).
    With u = t - 2: H2-type iff pair = u^2 - s^2*phi^2, H3-type iff u^2 - s^2*(1 + phi^2),
    H4-type iff u^4 - s^2*(2 + phi^2)*u^2 + s^4*phi^2, with s^2 in Q+. Returns [(pair, class, s2)]."""
    u = sp.Symbol('u'); out = []
    for g, sg, m in golden_pairs(p):
        cls, s2, member = "unclassified", None, None
        for cand in ([g] if sg is None else [g, sg]):        # a PAIR is classified by whichever member carries phi (not its conjugate)
            gu = sp.Poly(sp.expand(cand.subs(t, u + 2)), u)
            if gu.degree() == 1:                                   # H2-type splits over Q(sqrt5): u^2 - s^2 phi^2 = (u - s phi)(u + s phi)
                alpha = -gu.all_coeffs()[1]
                r = sp.nsimplify(sp.simplify((alpha / phi) ** 2))
                if r.is_rational and r > 0: cls, s2, member = "H2", r, cand; break
                continue
            if gu.degree() == 2 and gu.all_coeffs()[1] == 0:
                kappa = -gu.all_coeffs()[2]
                for name, ref in (("H2", phi ** 2), ("H3", 1 + phi ** 2)):
                    r = sp.nsimplify(sp.simplify(kappa / ref))
                    if r.is_rational and r > 0: cls, s2, member = name, r, cand; break
            elif gu.degree() == 4 and gu.all_coeffs()[1] == 0 and gu.all_coeffs()[3] == 0:
                c2, c0 = gu.all_coeffs()[2], gu.all_coeffs()[4]
                r = sp.nsimplify(sp.simplify(-c2 / (2 + phi ** 2)))
                if r.is_rational and r > 0 and sp.simplify(c0 - r ** 2 * phi ** 2) == 0: cls, s2, member = "H4", r, cand
            if cls != "unclassified": break
        out.append((member if member is not None else g, cls, s2))
    return out


def evaluate(n, edges, q=None):
    """Full evaluation of one operator: grade, cores, halves, per-half certificate (some-partner semantics),
    guards. Never called on Block D objects before the seal."""
    C = cart(n, [tuple(e) for e in edges]); p = sp.expand(C.charpoly(t).as_expr())
    G = nx.Graph([tuple(e) for e in edges]); G.add_nodes_from(range(n))
    D = sp.diag(*[G.degree(v) for v in range(n)])
    gr = grade(p)[0]
    rec = dict(n=n, grade=gr, regular=is_regular(D), cores=[(str(r), m) for r, m in rational_cores(p)],
               classes=[(str(g), c, str(s)) for g, c, s in class_sectors(p)], halves=[])
    if rec["regular"]:
        rec["blind"] = True; return rec
    hs = [q] if q is not None else halves(p)
    for h in hs:
        try:
            P = sector_projector(C, h, p); cert = certificate(C, D, P); cert.pop("R")
            cert = {k: (str(v) if not isinstance(v, list) else v) for k, v in cert.items()}
            cert["carrier"] = (sp.simplify(sp.sympify(cert["trRD"]) - 2 * s5 / 5) == 0 and sp.sympify(cert["leak_oo"]) == sp.Rational(2, 5))
            cert["q"] = str(h); rec["halves"].append(cert)
        except DegenerateFoldError as ex:
            rec["halves"].append(dict(q=str(h), declared=str(ex)))
    rec["carrier_some_half"] = any(h.get("carrier") for h in rec["halves"])
    return rec


def sector_projector_dm(C, q, p):
    """Same projector as sector_projector, evaluated with sympy DomainMatrix arithmetic over Q(sqrt5)
    (Horner on the field matrix; exact; several times faster at n ~ 100). Returns a sympy Matrix.
    Gated against sector_projector (explore_d0d) before use on the sealed objects."""
    from sympy.polys.matrices import DomainMatrix
    n = C.shape[0]
    Q = sp.Poly(q, t, domain=K)
    CF, rem = sp.Poly(sp.expand(p), t, domain=K).div(Q)
    if not rem.is_zero:
        raise ValueError("q does not divide p over Q(sqrt5)")
    a, b, g = sp.gcdex(Q, CF)
    if g.degree() != 0:
        raise DegenerateFoldError(f"gcd(q, p/q) has degree {g.degree()}")
    g0 = g.all_coeffs()[0]
    poly = sp.Poly(sp.expand(((b * CF) / g0).as_expr()), t, domain=K)
    Cd = DomainMatrix.from_Matrix(C).convert_to(K); I = DomainMatrix.eye(n, K); P = DomainMatrix.zeros((n, n), K)
    for c in poly.all_coeffs():
        P = P * Cd + I * K.convert(c)
    return P.to_Matrix().applyfunc(sp.expand)
