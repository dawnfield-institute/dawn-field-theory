"""Milestone 18 core — signed graphs (Block F, exp_20; registration journals/2026-09-07_blockF_registration.md).

A signed graph is a symmetric adjacency with entries in {0, ±1}. Switching by s ∈ {±1}^V is A ↦ diag(s)·A·diag(s);
the spectrum is switching-invariant, and on a unicyclic graph the class is decided by the sign product round the
cycle (Zaslavsky): +1 BALANCED (switching-equivalent to the unsigned graph), −1 TWISTED (the Möbius class). The
Cartan channel is C = 2I − A; the balanced class IS the ordinary Cartan matrix, so every tree result of this
milestone is the balanced case (a tree has no cycle, hence one class).

Grading over Q(√d) is exp_12 part-1's ('strict' / 'core' / 'partial' / '-'), computed per Q-irreducible factor —
identical by unique factorization — and it does NOT carry certificate.grade's odd-degree screen: an odd-degree
Q-irreducible factor never splits over a quadratic field (it would need two conjugate halves), so it is a rational
factor, but it does not make the whole polynomial golden-free (balanced C_28 over √7 is the counterexample).

M15's frame-holonomy instrument is IMPORTED from milestone15/core/representative.py, never re-implemented: its
complement frames slice with np.ix_ and diagonalise with eigh, so a signed adjacency needs no change. The one
M15 helper that assumes 0/1 entries (cycle_basis_single strips leaves by row sums) is always called on the
UNSIGNED adjacency, before twisting.
"""
import math, sys
from pathlib import Path
import numpy as np, sympy as sp, networkx as nx
from sympy.polys.matrices import DomainMatrix

HERE = Path(__file__).parent
_M15 = HERE.parent.parent / "milestone15" / "core"
if str(_M15) not in sys.path:
    sys.path.insert(0, str(_M15))
from representative import edge_transport, build_cycle, cycle_basis_single   # noqa: E402  (M15's instrument)

t = sp.Symbol('t')
FIELDS = (2, 3, 5, 6, 7, 13, 15)
# OEIS A001429: connected unicyclic graphs on n unlabelled vertices.
A001429 = {3: 1, 4: 2, 5: 5, 6: 13, 7: 33, 8: 89, 9: 240, 10: 657, 11: 1806, 12: 5026, 13: 13999, 14: 39260}

__all__ = ["t", "FIELDS", "A001429", "cond", "signed_cycle", "switch", "class_sign", "twist", "cartan",
           "charpoly_exact", "grade_by_factor", "pairing_fields", "no_integer_eigenvalue", "unicyclic_graphs",
           "adjacency", "holonomy_matrix", "holonomy_invariants", "predicted_cycle_grade",
           "edge_transport", "build_cycle", "cycle_basis_single"]


# ---- the objects ------------------------------------------------------------------------------------------
def cond(d):
    """Conductor of Q(√d), d squarefree: d if d ≡ 1 (mod 4), else 4d. √d ∈ Q(ζ_N) iff cond(d) | N."""
    return d if d % 4 == 1 else 4 * d


def signed_cycle(n, eps=1):
    """Cycle C_n, all edges +1 except (n−1, 0) carrying eps ∈ {+1, −1}: eps = +1 balanced, −1 twisted."""
    A = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = A[j, i] = 1.0
    A[n - 1, 0] = A[0, n - 1] = float(eps)
    return A


def switch(A, s):
    """Switching by s ∈ {±1}^V: diag(s) A diag(s)."""
    s = np.asarray(s, dtype=float)
    return (s[:, None] * A) * s[None, :]


def class_sign(A, cycle):
    """Sign product round an ordered vertex cycle (+1 balanced, −1 twisted)."""
    p = 1.0
    m = len(cycle)
    for i in range(m):
        p *= A[cycle[i], cycle[(i + 1) % m]]
    return int(round(p))


def twist(A, cycle):
    """Negate one cycle edge (cycle[0]–cycle[1]); every choice is switching-equivalent."""
    B = A.copy()
    u, v = cycle[0], cycle[1]
    B[u, v] = -B[u, v]
    B[v, u] = -B[v, u]
    return B


def cartan(A):
    return 2.0 * np.eye(A.shape[0]) - A


def adjacency(G, n):
    return nx.to_numpy_array(G, nodelist=range(n), dtype=float)


# ---- exact arithmetic ---------------------------------------------------------------------------------------
def charpoly_exact(A):
    """Exact characteristic polynomial of the Cartan matrix 2I − A over Z (DomainMatrix), as an expression in t.
    Gated against sympy's Matrix.charpoly in explore_f0 (KA-8) before use."""
    n = A.shape[0]
    rows = [[sp.ZZ(int(round(2.0 * (i == j) - A[i, j]))) for j in range(n)] for i in range(n)]
    coeffs = DomainMatrix(rows, (n, n), sp.ZZ).charpoly()        # highest degree first
    return sp.Poly([int(c) for c in coeffs], t).as_expr()


def grade_by_factor(p, d):
    """exp_12 part-1 grading over Q(√d), per Q-irreducible factor. Returns (grade, golden, rational) with
    golden/rational as lists of (factor, multiplicity); grade ∈ {'-', 'strict', 'core', 'partial'}:
    '-' no factor splits; 'strict' every factor splits; 'core' non-splitting factors all to even multiplicity;
    'partial' otherwise. An odd-degree factor is rational without trying the extension (it cannot split)."""
    sd = sp.sqrt(d)
    gold, rat = [], []
    _, facs = sp.factor_list(p, t)
    for f, e in facs:
        if sp.degree(f, t) == 0:
            continue
        if sp.degree(f, t) % 2 == 1:
            rat.append((f, int(e)))
            continue
        parts = [g for g in sp.Mul.make_args(sp.factor(f, extension=sd)) if g.has(t)]
        (gold if any(g.has(sd) for g in parts) else rat).append((f, int(e)))
    if not gold:
        return "-", [], rat
    if not rat:
        return "strict", gold, []
    return ("core" if all(e % 2 == 0 for _, e in rat) else "partial"), gold, rat


def pairing_fields(p, fields=FIELDS):
    """The set of d over which p has golden content (grade ≠ '-')."""
    return {d for d in fields if grade_by_factor(p, d)[0] != "-"}


def no_integer_eigenvalue(C, tol=1e-9):
    """True iff no eigenvalue of the (integer, symmetric) matrix C is an integer — a rational eigenvalue of a monic
    integer polynomial is an integer, so this is 'no rational eigenvalue'. Numeric prefilter; exact grading after."""
    w = np.linalg.eigvalsh(C)
    return bool(np.all(np.abs(w - np.round(w)) > tol))


# ---- the pure signed cycle: the theorem table (KA-6) ------------------------------------------------------------
def predicted_cycle_grade(n, eps, d):
    """Theorem 1 (registration §Theorems): the grade of the signed cycle C_n^eps over Q(√d).
    Balanced: '-' unless cond(d) | n, then 'partial' (simple zero mode). Twisted: '-' unless cond(d) | 2n; then
    'strict' iff d = 2 and 4 | n; 'partial' if n odd (the simple root 4); else 'core' (every eigenvalue doubled)."""
    if eps == 1:
        return "partial" if n % cond(d) == 0 else "-"
    if (2 * n) % cond(d) != 0:
        return "-"
    if d == 2 and n % 4 == 0:
        return "strict"
    return "partial" if n % 2 == 1 else "core"


# ---- the first non-tree class ---------------------------------------------------------------------------------
def unicyclic_graphs(n):
    """All connected unicyclic graphs on n vertices up to isomorphism: every tree plus one non-edge, deduplicated by
    Weisfeiler–Lehman hash then exact isomorphism. The count is asserted against OEIS A001429 (KA-7): an enumerator
    that misses a graph is not a census. Nodes are 0..n−1."""
    buckets, out = {}, []
    for T in nx.nonisomorphic_trees(n):
        edges = {frozenset(e) for e in T.edges()}
        for i in range(n):
            for j in range(i + 1, n):
                if frozenset((i, j)) in edges:
                    continue
                G = T.copy()
                G.add_edge(i, j)
                h = nx.weisfeiler_lehman_graph_hash(G, iterations=4)
                if any(nx.is_isomorphic(G, H) for H in buckets.get(h, [])):
                    continue
                buckets.setdefault(h, []).append(G)
                out.append(G)
    assert len(out) == A001429[n], f"unicyclic count at n={n}: {len(out)} ≠ A001429 {A001429[n]}"
    return out


# ---- M15's frame holonomy on a signed adjacency ---------------------------------------------------------------------
def holonomy_matrix(A, cycle, k):
    """The loop of representative.cycle_holonomy, returning the holonomy matrix itself plus the per-edge transport
    dets (reflections) and the minimum eigengap (M15's degeneracy guard). Adjacent pairs only, as in M15."""
    frames, H, gap, dets = {}, np.eye(k), np.inf, []
    m = len(cycle)
    for i in range(m):
        u, v = cycle[i], cycle[(i + 1) % m]
        T, g = edge_transport(A, u, v, k, frames)
        gap = min(gap, g)
        dets.append(int(np.sign(np.linalg.det(T))))
        H = T @ H
    return H, dets, float(gap)


def holonomy_invariants(H):
    """Conjugation invariants only (Theorem 2 relates the classes up to a diagonal ±1 conjugation)."""
    eig = np.linalg.eigvals(H)
    k = H.shape[0]
    return {"angles": sorted(float(a) for a in np.abs(np.angle(eig))),
            "det": float(np.linalg.det(H)),
            "deficit": float(np.linalg.norm(H - np.eye(k)))}
