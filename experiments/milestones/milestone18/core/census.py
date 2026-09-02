"""Milestone 18 core — the strict-fold census pipeline.

Promoted on 2026-09-02 from `scripts/explore_r16b_strict_hunt.py`, the stage-1 instrument of exp_15
(Phase 6, n = 20). The norm screen is a PROVEN necessary condition: p strict ⇒ p = q·σ(q) ⇒ p(x) =
N(q(x)) is a ℚ(√5)-norm for every rational x, and an integer is a norm iff every prime ≡ ±2 (mod 5)
divides it to an even power. It cannot lose a strict tree; survivors get the exact factorization.

Known-answer gate (exp_15 registration): at n = 16 the pipeline returns 14 distinct strict
polynomials on 15 trees (one cospectral pair). Counting basis is declared per call.
"""
import time, sympy as sp, networkx as nx

t = sp.Symbol('t'); s5 = sp.sqrt(5)
SCREEN_POINTS = (0, 1, -1, 2, 3, -2)


def is_norm(m):
    """Is the integer m a norm from Q(sqrt5)? (primes = +-2 mod 5 to even powers)"""
    if m == 0:
        return True
    for pr, ex in sp.factorint(abs(m)).items():
        if pr % 5 in (2, 3) and ex % 2 == 1:
            return False
    return True


def strict_grade(p):
    """True iff every irreducible factor of p over Q(sqrt5) is golden (non-rational)."""
    f = sp.factor(p, extension=s5); facs = [g for g in sp.Mul.make_args(f) if g.has(t)]
    return bool(facs) and all(g.has(s5) for g in facs)


def strict_hunt(n, xs=SCREEN_POINTS, log=None, progress=None):
    """Exhaustive strict-tree hunt on n vertices.
    Returns (tree_count, survivors, strict) with survivors/strict as lists of (edges, charpoly)."""
    t0 = time.time(); cnt = 0; surv = []
    for T in nx.nonisomorphic_trees(n):
        e = list(T.edges()); C = 2 * sp.eye(n)
        for i, j in e:
            C[i, j] = C[j, i] = -1
        p = C.charpoly(t); cnt += 1
        if all(is_norm(int(p.eval(x))) for x in xs):
            surv.append((sorted(map(list, e)), sp.expand(p.as_expr())))
        if progress and cnt % progress == 0 and log:
            log(f"  {cnt} trees, {len(surv)} survivors [{time.time() - t0:.0f}s]")
    if log:
        log(f"n={n}: {cnt} trees; norm-screen survivors: {len(surv)} [{time.time() - t0:.0f}s]")
    t1 = time.time(); strict = [(e, p) for e, p in surv if strict_grade(p)]
    if log:
        log(f"exact factorization on survivors: {time.time() - t1:.0f}s; STRICT trees: {len(strict)} "
            f"on {len({str(p) for _, p in strict})} polynomials")
    return cnt, surv, strict


def known_answer_gate_n16(strict):
    """The sealed gate: 14 distinct strict polynomials / 15 trees at n = 16 (basis declared)."""
    polys = len({str(p) for _, p in strict})
    return polys == 14 and len(strict) == 15, (len(strict), polys)
