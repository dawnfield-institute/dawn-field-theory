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


# ---------------------------------------------------------------------------------------------
# Parallel path (added 2026-09-03 for Phase 8, n = 22/24). Same screen, same factorization, same
# counting basis; trees are enumerated once and dealt to workers in index blocks. Validated
# through known_answer_gate_n16 before use — a parallel census that fails the gate is not a census.
# ---------------------------------------------------------------------------------------------
import itertools, os
from multiprocessing import Pool


def _screen_block(args):
    """Worker: screen a block of trees given as edge lists; return survivors (edges, charpoly str)."""
    n, block, xs = args
    out = []
    for e in block:
        C = 2 * sp.eye(n)
        for i, j in e:
            C[i, j] = C[j, i] = -1
        p = C.charpoly(t)
        if all(is_norm(int(p.eval(x))) for x in xs):
            out.append((sorted(map(list, e)), str(sp.expand(p.as_expr()))))
    return out


def _grade_one(ps):
    return ps, strict_grade(sp.sympify(ps))


def strict_hunt_parallel(n, xs=SCREEN_POINTS, workers=None, block=2000, log=None, progress=50):
    """Exhaustive strict-tree hunt on n vertices, parallel over index blocks of the tree enumeration.
    Returns (tree_count, survivors, strict) exactly as strict_hunt does (edges, charpoly as sympy expr)."""
    workers = workers or max(1, os.cpu_count() - 1)
    t0 = time.time(); cnt = 0; surv = []
    gen = (list(T.edges()) for T in nx.nonisomorphic_trees(n))

    def blocks():
        while True:
            b = list(itertools.islice(gen, block))
            if not b:
                return
            yield (n, b, xs)
    with Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_screen_block, blocks(), chunksize=1), 1):
            surv.extend(res); cnt_b = block  # exact count is reconciled below from the enumeration
            if log and k % progress == 0:
                log(f"  ~{k * block} trees, {len(surv)} survivors [{time.time() - t0:.0f}s]")
    # exact tree count from the enumeration itself (cheap: enumeration is ~10 us/tree)
    cnt = sum(1 for _ in nx.nonisomorphic_trees(n))
    if log:
        log(f"n={n}: {cnt} trees; norm-screen survivors: {len(surv)} [{time.time() - t0:.0f}s]")
    t1 = time.time()
    polys = sorted({ps for _, ps in surv})
    with Pool(workers) as pool:
        grade = dict(pool.map(_grade_one, polys, chunksize=4))
    strict = [(e, sp.sympify(ps)) for e, ps in surv if grade[ps]]
    if log:
        log(f"exact factorization on {len(polys)} survivor polynomials: {time.time() - t1:.0f}s; STRICT trees: "
            f"{len(strict)} on {len({str(p) for _, p in strict})} polynomials")
    return cnt, surv, strict


# ---------------------------------------------------------------------------------------------
# Fast exact one-5 partner map (added 2026-09-03 for Phase 8). The diagram polynomial of a tree D
# with one golden bond e* = {A, B} (weight -phi) expands along that bond with INTEGER charpolys:
#   q = det(tI - Gram) = q_del - phi^2 * q_cut = (q_del - q_cut) - phi * q_cut,
# where q_del = charpoly of D with the bond deleted (two components) and q_cut = charpoly of
# D with both endpoints deleted. Both are integer-matrix charpolys (fast); no symbolic phi.
# Validated against foldlaws.one5_diagrams (the symbolic map) at k <= 8 before use.
# ---------------------------------------------------------------------------------------------

def _int_charpoly(edges, nodes):
    idx = {v: i for i, v in enumerate(nodes)}; m = len(nodes)
    if m == 0:
        return sp.Integer(1)
    C = 2 * sp.eye(m)
    for a, b in edges:
        if a in idx and b in idx:
            C[idx[a], idx[b]] = C[idx[b], idx[a]] = -1
    return sp.expand(C.charpoly(t).as_expr())


def one5_partners_fast(k, phi_sym=None):
    """All one-5 tree diagrams on k nodes: dict key str(q*sigma(q)) -> list of (q, E, pos).
    q is exact in Z[phi][t] (phi = (1+sqrt5)/2). Same key semantics as foldlaws.one5_diagrams but
    keeps EVERY placement per polynomial (cospectral placements exist)."""
    phi = phi_sym if phi_sym is not None else (1 + s5) / 2
    out = {}
    for T in nx.nonisomorphic_trees(k):
        E = list(T.edges()); nodes = list(T.nodes())
        for pos, (A, B) in enumerate(E):
            Edel = [e for m, e in enumerate(E) if m != pos]
            q_del = _int_charpoly(Edel, nodes)
            q_cut = _int_charpoly(Edel, [v for v in nodes if v not in (A, B)])
            q = sp.expand(q_del - (phi + 1) * q_cut)          # phi^2 = phi + 1
            key = str(sp.expand(q * q.subs(s5, -s5)))
            out.setdefault(key, []).append((q, E, pos))
    return out
