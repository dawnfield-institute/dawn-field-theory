"""Milestone 18 core — the matching structure of a strict fold.

Promoted on 2026-09-02 from `scripts/explore_r15_matching_structure.py`, `explore_r15c_defect_side.py`
and `explore_r19_n20_defects.py` (the inputs of exp_16–exp_18). On a strict fold the reflection
R = P − σ(P) has the form √5·R = S + 2Π (S = diag(±1), Π a perfect matching, SΠ = −ΠS); this module
reads S, Π and the defect edge off R, and builds the Π-quotient with its edge multiplicities.
Proved on construction parents (journal 2026-09-02_r17_construction_theorem.md); observed 68/68.
"""
import sympy as sp, networkx as nx
from ledger import bezout_proj, simp

s5 = sp.sqrt(5)


def sigma(M):
    return M.applyfunc(lambda x: sp.expand(x.subs(s5, -s5)))


def reflection(C, q):
    """R = P − σ(P) for the Bezout projector of the diagram polynomial q (strict case)."""
    P = bezout_proj(C, q)
    return simp(P - sigma(P))


def matching_form(R):
    """Test √5·R = S + 2Π. Returns (form_ok, match, sign): match[v] = Π(v); sign[v] = S_vv.
    form_ok is False when any row is not (diagonal ±1, exactly one off-diagonal ±2, zeros)."""
    n = R.shape[0]
    S5R = simp(s5 * R).applyfunc(sp.nsimplify)
    match = {}; sign = {}
    for v in range(n):
        row = [(w, S5R[v, w]) for w in range(n) if w != v and S5R[v, w] != 0]
        if not (S5R[v, v] ** 2 == 1 and len(row) == 1 and row[0][1] ** 2 == 4):
            return False, {}, {}
        match[v] = row[0][0]; sign[v] = int(S5R[v, v])
    return True, match, sign


def anticommute(match, sign):
    """SΠ = −ΠS: matched vertices carry opposite signs."""
    return all(sign[v] == -sign[match[v]] for v in match)


def defect_edges(edges, match):
    """Edges whose Π-image is a non-edge (exactly one on every strict fold known)."""
    E = {tuple(sorted(x)) for x in edges}
    return [ed for ed in E if tuple(sorted((match[ed[0]], match[ed[1]]))) not in E]


def quotient_multiplicities(edges, match):
    """The Π-quotient as a multigraph: {(pair_i, pair_j): multiplicity}, and the pair index."""
    pairs = sorted({frozenset((v, match[v])) for v in match}, key=sorted)
    pid = {fs: i for i, fs in enumerate(pairs)}
    mult = {}
    for a, b in edges:
        pa = pid[[fs for fs in pairs if a in fs][0]]; pb = pid[[fs for fs in pairs if b in fs][0]]
        if pa != pb:
            mult[tuple(sorted((pa, pb)))] = mult.get(tuple(sorted((pa, pb))), 0) + 1
    return mult, pairs, pid


def sign_split_structure(edges, sign):
    """Copy/conjugate sides by sign, their components, and the cut."""
    copy = [v for v in sign if sign[v] > 0]; conj = [v for v in sign if sign[v] < 0]
    G = nx.Graph([tuple(x) for x in edges])
    cut = [x for x in edges if (x[0] in copy) != (x[1] in copy)]
    cc = nx.number_connected_components(G.subgraph(copy)) if copy else 0
    sizes = sorted(len(c) for c in nx.connected_components(G.subgraph(conj))) if conj else []
    return {"copy": copy, "conj": conj, "cut": cut, "copy_components": cc, "conj_sizes": sizes}
