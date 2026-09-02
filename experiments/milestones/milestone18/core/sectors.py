"""Milestone 18 core — automorphism sectors of a tree.

Promoted on 2026-09-02 from `scripts/exp_13_phase4_n16.py` / `exp_15_phase6_n20.py` (T6) and
`explore_r9_mixed_trees.py`. Vertex orbits of a TREE are exactly the classes of the AHU rooted
canonical form (no automorphism enumeration — the n = 14 double stars have ~10⁶ automorphisms).
The symmetric sector S is the span of orbit indicators; the golden content of C on S is the
quotient fold, and on S⊥ it is a Galois fold of swapped subtrees (journal 2026-09-02_exp15_outcomes.md).
"""
import sympy as sp, networkx as nx
from ledger import cart as _cart

t = sp.Symbol('t'); s5 = sp.sqrt(5)


def cart(n, e):
    return _cart(n, [tuple(x) for x in e])


def charpoly(n, e):
    return sp.expand(cart(n, e).charpoly(t).as_expr())


def orbits(G):
    """Automorphism orbits of a tree via AHU canonical rooted forms (exact for trees)."""
    def canon(root):
        def rec(v, parent):
            return "(" + "".join(sorted(rec(w, v) for w in G[v] if w != parent)) + ")"
        return rec(root, None)
    key = {v: canon(v) for v in G}; seen = {}
    for v in sorted(G):
        seen.setdefault(key[v], []).append(v)
    return list(seen.values())


def restricted_charpoly(C, V):
    """Characteristic polynomial of C on the subspace with basis matrix V: det(VᵀCV − t·VᵀV)/det(VᵀV)."""
    G = V.T * V; M = V.T * C * V
    return sp.expand(sp.cancel((M - t * G).det() / G.det()))


def golden_bases(p):
    return sorted(str(sp.expand(g.as_base_exp()[0])) for g in sp.Mul.make_args(sp.factor(p, extension=s5)) if g.has(s5))


def sector_check(n, e):
    """(is_pure_quotient, sperp_pairs, unexplained). A tree with trivial orbit partition is
    UNEXPLAINED (fails), never vacuously a quotient fold (Phase 4 pre-seal sharpening)."""
    G = nx.Graph([tuple(x) for x in e]); orb = orbits(G)
    if len(orb) == n:
        return (False, False, True)
    S = sp.Matrix.hstack(*[sp.Matrix([1 if v in c else 0 for v in range(n)]) for c in orb])
    Sperp = sp.Matrix.hstack(*S.T.nullspace())
    p = charpoly(n, e)
    gS = golden_bases(restricted_charpoly(cart(n, e), S)); gT = golden_bases(p)
    gP = golden_bases(restricted_charpoly(cart(n, e), Sperp)) if Sperp.shape[1] else []
    if not gP:
        return (gS == gT, True, False)
    paired = all(any(sp.expand(sp.sympify(b).subs(s5, -s5) - sp.sympify(b2)) == 0 for b2 in gP) for b in gP)
    return (False, paired, False)
