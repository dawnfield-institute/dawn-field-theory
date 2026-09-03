#!/usr/bin/env python3
"""explore_r21b (exploration): verify the rational reduction of the Bezout problem on every Galois half of every strict tree:
q = q0 - phi*q1 (q1 = (sigma q - q)/sqrt5, q0 = q + phi*q1), v = a + sqrt5*c with
a*(2q0 - q1) + 5c*q1 = 1 over Q[t]; then 5b = (5c)*(2q0 - q1) + 5a*q1.
Checks: (i) this 5b equals the K-Bezout 5b; (ii) den(5b) | 2^{deg q1} * Res(q0, q1);
(iii) 5 | den(5b) only if 5 | Res(q0, q1)."""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import sympy as sp
sys.path.insert(0, str(Path(__file__).parent))
import explore_r21_poly_integrality as m
from foldlaws import t, s5, phi, cart
K = sp.QQ.algebraic_field(sp.sqrt(5))

def one_half(p, q):
    sq = m.sig(q)
    q1 = sp.expand((sq - q) / s5); q0 = sp.expand(q + phi * q1)
    assert not q1.has(s5) and not q0.has(s5)
    A = sp.expand(2 * q0 - q1)
    PA, P1 = sp.Poly(A, t, domain=sp.QQ), sp.Poly(q1, t, domain=sp.QQ)
    if P1.is_zero:
        return dict(skip="q1=0")
    a, c5, g = sp.gcdex(PA, P1)          # a*A + c5*q1 = g
    if g.degree() != 0:
        return dict(skip="gcd(A,q1) nonconstant")
    g0 = g.all_coeffs()[0]; a = a / g0; c5 = c5 / g0
    fiveb_rat = sp.Poly(sp.expand((c5 * PA + 5 * a * P1).as_expr()), t)
    r = m.check(p, q)
    if r["den_5b"] is None:
        return dict(skip="K-half degenerate")
    # K-Bezout 5b recomputed for comparison
    Q, SQ = sp.Poly(q, t, domain=K), sp.Poly(sq, t, domain=K); aa, bb, gg = sp.gcdex(Q, SQ)
    gg0 = sp.expand(gg.all_coeffs()[0].as_expr()); P = sp.Poly(p, t, domain=K)
    Pt = sp.Poly(sp.expand(sp.radsimp((bb * SQ).as_expr() / gg0)), t, domain=K).rem(P)
    R = sp.expand(2 * Pt.as_expr() - 1); odd = sp.expand((R - m.sig(R)) / (2 * s5))
    fiveb_K = sp.Poly(sp.expand(5 * odd), t)
    same = (fiveb_rat - fiveb_K).is_zero
    res01 = sp.Rational(sp.resultant(q0, q1, t))
    den = sp.ilcm(*[sp.Rational(x).q for x in fiveb_K.all_coeffs()])
    bound = sp.Integer(2) ** P1.degree() * abs(res01.p)  # res01 integer if q0,q1 integral
    divides = (bound % den == 0) if res01.q == 1 else None
    return dict(same=bool(same), den_5b=int(den), res01=str(res01), q1_deg=P1.degree(),
                bound_divides=divides, five_in_den=(den % 5 == 0), five_in_res01=(res01.p % 5 == 0),
                q0q1_integral=(all(sp.Rational(x).q == 1 for x in sp.Poly(q0, t).all_coeffs()) and all(sp.Rational(x).q == 1 for x in P1.all_coeffs())))

def worker(job):
    n, e, src, parent = job
    C = cart(n, e); p = sp.expand(C.charpoly(t).as_expr()); hs = m.halves(p)
    out = []
    for q in hs or []:
        try: out.append(one_half(p, q))
        except Exception as ex: out.append(dict(error=repr(ex)))
    return dict(n=n, src=src, parent=parent, halves=out)

if __name__ == "__main__":
    trees = []
    for r in m.load_census(13):
        if r.get("fields", {}).get("sqrt5", {}).get("grade") == "strict": trees.append((r["n"], [tuple(x) for x in r["edges"]], "census", None))
    for r in json.load(open(m.RES / "explore_r16b_strict_n16.json")): trees.append((16, [tuple(x) for x in r["edges"]], "r16b", None))
    e15 = json.load(open(m.RES / "exp_15_n20.json"))
    key = lambda E: tuple(sorted(tuple(sorted(x)) for x in E))
    T3 = {key(r["edges"]): r for r in e15["T3"]}; T4 = {key(r["edges"]): r for r in e15["T4"]}
    for r in e15["strict"]:
        k = key(r["edges"]); trees.append((20, [tuple(x) for x in r["edges"]], "exp15", bool(T3.get(k, {}).get("ledger") and T4.get(k, {}).get("quotient_iso"))))
    from collections import Counter
    tally = Counter(); rows = []
    with Pool(8) as pool:
        for rec in pool.imap_unordered(worker, trees):
            rows.append(rec)
            for h in rec["halves"]:
                if "skip" in h: tally[("skip", h["skip"])] += 1; continue
                if "error" in h: tally[("error",)] += 1; print(h); continue
                tally[("rational==K", h["same"])] += 1
                tally[("den | 2^deg q1 * Res(q0,q1)", h["bound_divides"])] += 1
                tally[("q0,q1 integral", h["q0q1_integral"])] += 1
                tally[("5|den", h["five_in_den"], "5|Res01", h["five_in_res01"])] += 1
                if rec["n"] == 20 and rec["parent"] and len(rec["halves"]) == 1:
                    tally[("parent fold half: den, Res01", h["den_5b"], h["res01"])] += 1
    for k, v in sorted(tally.items(), key=str): print(k, v)
    json.dump(rows, open(m.RES / "explore_r21b_rational_reduction_20260902.json", "w"), indent=1, default=str)
