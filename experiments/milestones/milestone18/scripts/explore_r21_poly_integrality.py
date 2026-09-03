#!/usr/bin/env python3
"""explore_r21 (exploration): the polynomial form of integrality, tested on every strict tree we hold
(n <= 16: census + explore_r16b; n = 20: exp_15) and on EVERY Galois half q of p, not only the fold
half. For each half: the reduced reflection polynomial R(t) = 2P(t) - 1 mod p with P = v*sigma(q)
from the minimal-degree Bezout identity, b = R/sqrt5, the denominators of b and 5b, and the prime
content of Res(q, sigma q). Prompted by an external proof attempt of "5b in Z[t] on every
construction parent" (A. Farmer, 2026-09-02). Finding: FALSE at n = 20 — three construction parents
(every strict law holding) have den(5b) = 3, Res(q, sigma q) = 3^2 * 5^5. See the r21 journal.
Runs in ~2 min on 8 cores; results are append-only."""
import sys, json, itertools
from pathlib import Path
from collections import Counter
import sympy as sp
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "core"))
from foldlaws import t, s5, cart, load_census, one5_diagrams
K = sp.QQ.algebraic_field(sp.sqrt(5))
RES = ROOT / "results"

def sig(e): return sp.expand(e.subs(s5, -s5))

def halves(p):
    """All Galois halves of p = q*sigma(q) with gcd(q,sigma q)=1, up to complement."""
    f = sp.factor(p, extension=s5)
    facs = []
    for g in sp.Mul.make_args(f):
        if g.has(t):
            b, ex = g.as_base_exp(); facs += [sp.expand(b)] * int(ex)
    pairs, used = [], set()
    for i, g in enumerate(facs):
        if i in used: continue
        sg = sig(g)
        j = next((k for k in range(i + 1, len(facs)) if k not in used and sp.expand(facs[k] - sg) == 0), None)
        if j is None:
            return None  # sigma-fixed factor -> not strict
        used |= {i, j}; pairs.append((g, sg))
    out = []
    for choice in itertools.product([0, 1], repeat=len(pairs) - 1):
        q = pairs[0][0]
        for c, pr in zip(choice, pairs[1:]): q *= pr[c]
        out.append(sp.expand(q))
    return out

def split_s5(expr):
    """expr in Q(sqrt5)[t] -> (rational part, sqrt5 part) as expressions."""
    e = sp.expand(expr)
    a = sp.expand((e + sig(e)) / 2); b = sp.expand((e - sig(e)) / (2 * s5))
    return a, b

def check(p, q):
    sq = sig(q)
    Q, SQ = sp.Poly(q, t, domain=K), sp.Poly(sq, t, domain=K)
    a, b, g = sp.gcdex(Q, SQ)
    if g.degree() != 0:
        return dict(declared="gcd(q,sigma q) non-constant (repeated conjugate pair; half is sigma-fixed)", deg_b=None, den_b=None, den_5b=None, fiveb=None, res=None, res_factor="0", res_has_sqrt5=None, bezout_den=None)
    P = sp.Poly(p, t, domain=K)
    g0 = sp.expand(g.all_coeffs()[0].as_expr()) if hasattr(g.all_coeffs()[0], "as_expr") else sp.expand(sp.sympify(g.all_coeffs()[0]))
    Pt = sp.Poly(sp.expand(sp.radsimp((b * SQ).as_expr() / g0)), t, domain=K).rem(P)   # projector polynomial mod p
    Rt = sp.Poly(sp.expand(2 * Pt.as_expr() - 1), t, domain=K).rem(P)                  # reflection polynomial
    Rexpr = sp.expand(Rt.as_expr())
    rat, odd = split_s5(Rexpr)
    assert sp.expand(rat) == 0, "R not sigma-odd"
    b_t = sp.Poly(odd, t)                                    # R = sqrt5 * b(t), b in Q[t]
    fiveb = sp.Poly(sp.expand(5 * odd), t)
    dens = [sp.Rational(c).q for c in b_t.all_coeffs()]
    den_b = sp.ilcm(*dens) if dens else 1
    dens5 = [sp.Rational(c).q for c in fiveb.all_coeffs()]
    den_5b = sp.ilcm(*dens5) if dens5 else 1
    res = sp.resultant(q, sq, t)
    res_rat, res_odd = split_s5(res)
    resN = sp.Rational(res_rat) if res_odd == 0 else sp.Rational(res_odd)  # k odd -> sqrt5 * rational
    return dict(deg_b=b_t.degree(), den_b=int(den_b), den_5b=int(den_5b),
                fiveb=str(fiveb.as_expr()) if b_t.degree() <= 7 else None,
                res=str(res), res_factor=str(sp.factorint(resN)) if resN != 0 else "0",
                res_has_sqrt5=(res_odd != 0),
                bezout_den=int(sp.ilcm(*[sp.Rational(x).q for c in b.all_coeffs() for x in (split_s5(c.as_expr()))])) )

DIAGQ = []

def _init(dq):
    DIAGQ[:] = dq

def worker(job):
    n, e, src, parent = job
    try:
        C = cart(n, e); p = sp.expand(C.charpoly(t).as_expr())
        hs = halves(p)
        if hs is None:
            return dict(n=n, src=src, edges=e, parent=parent, note="sigma-fixed factor; not strict")
        rec = dict(n=n, src=src, edges=e, parent=parent, n_halves=len(hs), halves=[])
        for q in hs:
            is_diag = any(sp.expand(q - dq) == 0 or sp.expand(sig(q) - dq) == 0 for dq in DIAGQ) if n <= 16 else None
            r = check(p, q); r["is_one5_diagram"] = is_diag
            rec["halves"].append(r)
        return rec
    except Exception as ex:
        return dict(n=n, src=src, edges=e, parent=parent, error=repr(ex))

def main():
    trees = []
    census = load_census(13)
    for r in census:
        if r["fields"].get("sqrt5", {}).get("grade") == "strict":
            trees.append((r["n"], [tuple(e) for e in r["edges"]], "census"))
    for r in json.load(open(RES / "explore_r16b_strict_n16.json")):
        trees.append((16, [tuple(e) for e in r["edges"]], "r16b"))
    for r in json.load(open(RES / "exp_15_n20.json"))["strict"]:
        trees.append((20, [tuple(e) for e in r["edges"]], "exp15"))
    # parent flag at n = 20 from exp_15 T2 (one-5 partner exists); diagram labels at n <= 16 from k <= 8 diagrams
    e15 = json.load(open(RES / "exp_15_n20.json"))
    parent20 = {tuple(sorted(tuple(sorted(x)) for x in r["edges"])): bool(r.get("partnered")) for r in e15["T2"]}
    diag = one5_diagrams(16); DIAGQ[:] = [q for (q, M) in diag.values()]
    print("diagrams k<=8:", len(DIAGQ), "| parent flags at 20:", sum(parent20.values()), "/", len(parent20), flush=True)
    out = []; tally = Counter()
    outpath = RES / "explore_r21_poly_integrality_20260902.json"
    from multiprocessing import Pool
    jobs = [(n, e, src, parent20.get(tuple(sorted(tuple(sorted(x)) for x in e))) if n == 20 else None) for n, e, src in trees]
    with Pool(8, initializer=_init, initargs=(list(DIAGQ),)) as pool:
        for rec in pool.imap_unordered(worker, jobs):
            out.append(rec)
            n = rec["n"]
            for h in rec.get("halves", []):
                if h["den_5b"] is None:
                    tally[(n, "declared half")] += 1; continue
                lab = ("diagram-half" if h["is_one5_diagram"] else "other-half") if n <= 16 else ("parent-tree" if rec.get("parent") else "nonparent-tree")
                tally[(n, lab, "5b integral" if h["den_5b"] == 1 else "5b NOT integral")] += 1
            if "halves" in rec:
                print(n, rec["src"], "parent" if rec.get("parent") else "", "halves", rec["n_halves"], [(h["den_b"], h["den_5b"], h["deg_b"], h["res_factor"], h["is_one5_diagram"]) for h in rec["halves"]], flush=True)
            else:
                print(n, rec["src"], rec.get("note") or rec.get("error"), flush=True)
            json.dump(out, open(outpath, "w"), indent=1, default=str)
    print("\nTALLY"); [print(k, v) for k, v in sorted(tally.items(), key=str) if v]

if __name__ == "__main__":
    main()
