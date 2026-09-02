"""Milestone 18 core — the fold-law machinery.

Promoted on 2026-09-02 from `scripts/exp_12_part2_fold_laws.py`, which `exp_13` and the
explorations r7b, r8, r11, r12, r13, r14 had been `exec`-ing as a library (split on a results-dict
literal). Behaviour is byte-identical to the promoted source; exp_12's n ≤ 12 outputs were diffed
against the committed results at promotion.

Exports: sigma, grade_of, charpoly, load_census, golden_of, one5_diagrams, load_context,
rational_core_factors, conic_half, projector_for, invariants, sig — plus the shared symbols
t, s5, phi, RES and the ledger primitives bezout_proj, simp, cart.
"""
import json, sympy as sp, numpy as np, networkx as nx
from pathlib import Path
from ledger import bezout_proj, simp, cart

t = sp.Symbol('t'); s5 = sp.sqrt(5); phi = (1 + s5) / 2
RES = Path(__file__).parent.parent / "results"

__all__ = ["t", "s5", "phi", "RES", "bezout_proj", "simp", "cart", "sp", "np", "nx", "json",
           "sigma", "grade_of", "charpoly", "load_census", "golden_of", "one5_diagrams",
           "load_context", "rational_core_factors", "conic_half", "projector_for", "invariants", "sig"]


def sigma(M):
    """Entrywise Galois conjugation sqrt5 -> -sqrt5."""
    return M.applyfunc(lambda x: sp.expand(x.subs(s5, -s5)))


def grade_of(r):
    return r["fields"].get("sqrt5", {}).get("grade")


def charpoly(n, e):
    return sp.expand(cart(n, e).charpoly(t).as_expr())


def load_census(NMAX):
    """The exhaustive n <= 13 census, plus exp_11b's core-grade trees at n = 14 when asked for."""
    census = json.load(open(RES / "explore_g1_census_20260901.json"))
    if NMAX >= 14 and (RES / "exp_11b_core_trees_n14.json").exists():
        for r in json.load(open(RES / "exp_11b_core_trees_n14.json")):
            census.append({"n": 14, "edges": r["edges"], "det": r["det"],
                           "fields": {"sqrt5": {"grade": r["grade"], "rational": r["rational"]}}})
    return census


def golden_of(census, NMAX):
    return [r for r in census if grade_of(r) in ("strict", "core") and r["n"] <= NMAX]


def one5_diagrams(NMAX):
    """One-5 Coxeter diagrams for k <= NMAX//2, keyed by q*sigma(q); value (q, Gram matrix).
    First placement wins per key (cospectral placements exist — see exp_13 T4)."""
    diag = {}
    for k in range(2, NMAX // 2 + 1):
        for T in nx.nonisomorphic_trees(k):
            E = list(T.edges())
            for pos in range(len(E)):
                M = 2 * sp.eye(k)
                for m, (i, j) in enumerate(E):
                    M[i, j] = M[j, i] = (-phi if m == pos else -1)
                q = sp.expand(M.charpoly(t).as_expr())
                diag.setdefault(sp.expand(q * q.subs(s5, -s5)), (q, M))
    return diag


def load_context(NMAX):
    """(census, golden, diag) exactly as exp_12_part2 built them at module level."""
    census = load_census(NMAX)
    return census, golden_of(census, NMAX), one5_diagrams(NMAX)


def rational_core_factors(p):
    f = sp.factor(p, extension=s5)
    return [g for g in sp.Mul.make_args(f) if g.has(t) and not g.has(s5)]


def conic_half(C, lam):
    """split the 2-dim eigenspace of rational eigenvalue lam into a golden line v=u1+tau*u2 with
    sigma-complement: c*N(tau)+b*Tr(tau)+a=0. Return (P_line at two conic points).
    For multiplicity > 2 (Grassmannian split, not registered) return the half-core stand-in
    twice and let the caller DECLARE it (traces are gauge-independent only on B-constant cores)."""
    core = (C - lam * sp.eye(C.shape[0])).nullspace()
    if len(core) != 2:
        V = sp.Matrix.hstack(*core); Q = simp(V * (V.T * V).inv() * V.T)
        return [Q / 2, Q / 2, "stand-in:multiplicity>2"]
    u1, u2 = core
    a_, b_, c_ = (u1.T * u1)[0], (u1.T * u2)[0], (u2.T * u2)[0]
    sols = []
    for qq in [sp.Rational(x, y) for y in range(1, 13) for x in range(-12, 13)]:
        if qq == 0:
            continue
        disc = sp.nsimplify(b_ ** 2 - c_ * (a_ - 5 * c_ * qq ** 2)); r = sp.sqrt(disc)
        if r.is_rational:
            tau = sp.Rational(sp.nsimplify((-b_ + r) / c_)) + qq * s5
            if not any(sp.simplify(tau - s) == 0 for s in sols):
                sols.append(tau)
        if len(sols) >= 2:
            break
    if len(sols) < 2:   # conic has no small rational point: registered recipe cannot reach it -> declare
        V = sp.Matrix.hstack(*core); Q = simp(V * (V.T * V).inv() * V.T)
        return [Q / 2, Q / 2, "stand-in:conic-unresolved(lam=%s)" % lam]
    out = []
    for tau in sols[:2]:
        v = u1 + tau * u2; out.append(simp((v * v.T) / (v.T * v)[0]))
    return out


def projector_for(n, e, q):
    """Bezout projector for the diagram polynomial q with the rational core removed and re-added as
    golden lines at two conic points. Returns (Ps, C, standin_labels)."""
    C = cart(n, e); p = charpoly(n, e); rat = rational_core_factors(p)
    q_off = q
    for g in rat:
        b, ex = g.as_base_exp(); q_off = sp.cancel(q_off / b ** (ex // 2))
    q_off = sp.expand(q_off); P_off = bezout_proj(C, q_off)
    cores = []; quad = []
    for g in rat:
        b, ex = g.as_base_exp()
        roots = sp.solve(b, t)
        cores.extend(roots)                       # EVERY root of every rational factor (quadratic cores have two)
        if len(roots) > 1:
            quad.append(str(b))
    if not cores:
        return [P_off], C, []
    Qc = sp.zeros(n, n)
    for lam in cores:
        V = sp.Matrix.hstack(*(C - lam * sp.eye(n)).nullspace()); Qc += simp(V * (V.T * V).inv() * V.T)
    P_off = simp(P_off * (sp.eye(n) - Qc))
    gauges = [conic_half(C, lam) for lam in cores]
    standin = [g[2] for g in gauges if len(g) == 3] + ["stand-in:quadratic-rational-core(%s)" % b for b in quad]
    Ps = []
    for choice in (0, 1):
        P = P_off
        for g in gauges:
            P = simp(P + g[min(choice, 1)])
        Ps.append(P)
    return Ps, C, standin


def invariants(P, n, e):
    """(tr(RD), tr(PB), ||(I-P) B P||_F^2) with D the degree matrix and B = 2I - D."""
    A = sp.zeros(n, n)
    for i, j in e:
        A[i, j] = A[j, i] = 1
    D = sp.diag(*[sum(A[i, k] for k in range(n)) for i in range(n)]); B = 2 * sp.eye(n) - D
    R = simp(P - sigma(P)); X = simp((sp.eye(n) - P) * B * P)
    return (sp.nsimplify(sp.expand((R * D).trace())), sp.nsimplify(sp.expand((P * B).trace())),
            sp.nsimplify(sp.expand(sum(x ** 2 for x in X))))


def sig(M):
    """Signature (positives, negatives) of a symmetric sympy matrix, numerically at 30 digits."""
    ev = np.linalg.eigvalsh(np.array(M.evalf(30).tolist(), dtype=float))
    return (int((ev > 1e-9).sum()), int((ev < -1e-9).sum()))
