#!/usr/bin/env python3
"""exp_19: Phase 8 (SEALED by the commit carrying journals/2026-09-03_phase8_registration.md) — the
census at n = 22 and n = 24. Stage 1: strict hunt (core/census.strict_hunt_parallel: proven-necessary
norm screen + exact factorization). Stage 2: per strict tree — partner lookup (one5_partners_fast at
k = n/2, every placement), Phase 6's T3–T6 battery with the r22 repairs, the fold-half denominators via
the rational reduction (r21), the certificate over ALL Galois halves, and the species flags. T6: zero
orphans at k = 12. Per-tree basis; per-polynomial tallies alongside. Degenerate partners declared.
Usage: python exp_19_phase8_census.py <n> [workers]. Run as a file (macOS spawn)."""
import sys, time, json, itertools
from pathlib import Path
from multiprocessing import Pool
import sympy as sp, networkx as nx
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from ledger import bezout_proj, simp, cart as _cart, DegenerateFoldError
from census import strict_hunt_parallel, one5_partners_fast
from sectors import orbits, restricted_charpoly, golden_bases
t = sp.Symbol('t'); s5 = sp.sqrt(5); phi = (1 + s5) / 2; K = sp.QQ.algebraic_field(sp.sqrt(5))
RES = Path(__file__).parent.parent / "results"

def sig(e): return sp.expand(e.subs(s5, -s5))
def sigma(M): return M.applyfunc(lambda x: sp.expand(x.subs(s5, -s5)))
def frob2(X): return sp.expand(sum(x ** 2 for x in X))

def halves(p):
    """All Galois halves of p (up to complement); None if a sigma-fixed factor exists."""
    facs = []
    for g in sp.Mul.make_args(sp.factor(p, extension=s5)):
        if g.has(t):
            b, ex = g.as_base_exp(); facs += [sp.expand(b)] * int(ex)
    pairs, used = [], set()
    for i, g in enumerate(facs):
        if i in used: continue
        sg = sig(g); j = next((k for k in range(i + 1, len(facs)) if k not in used and sp.expand(facs[k] - sg) == 0), None)
        if j is None: return None
        used |= {i, j}; pairs.append((g, sg))
    out = []
    for choice in itertools.product([0, 1], repeat=len(pairs) - 1):
        q = pairs[0][0]
        for c, pr in zip(choice, pairs[1:]): q *= pr[c]
        out.append(sp.expand(q))
    return out

def denominators(q):
    """Rational reduction (r21): q = q0 - phi*q1; a(2q0-q1) + 5c*q1 = 1; 5b = (5c)(2q0-q1) + 5a*q1.
    Returns dict or {'declared': ...} when gcd(2q0-q1, q1) is non-constant (= gcd(q, sigma q) non-constant)."""
    sq = sig(q); q1 = sp.expand((sq - q) / s5); q0 = sp.expand(q + phi * q1)
    assert not q1.has(s5) and not q0.has(s5)
    A = sp.expand(2 * q0 - q1); PA, P1 = sp.Poly(A, t, domain=sp.QQ), sp.Poly(q1, t, domain=sp.QQ)
    if P1.is_zero: return dict(declared="q1 = 0")
    a, c5, g = sp.gcdex(PA, P1)
    if g.degree() != 0: return dict(declared="gcd non-constant (degenerate half)")
    g0 = g.all_coeffs()[0]; a = a / g0; c5 = c5 / g0
    fiveb = sp.Poly(sp.expand((c5 * PA + 5 * a * P1).as_expr()), t)
    den = int(sp.ilcm(*[sp.Rational(x).q for x in fiveb.all_coeffs()]))
    res01 = sp.Rational(sp.resultant(q0, q1, t)); assert res01.q == 1; res01 = int(res01)
    fac = sp.factorint(abs(res01)) if res01 not in (0,) else {}
    odd_rad = 1
    for pr in fac:
        if pr != 2: odd_rad *= pr
    return dict(res01=res01, den5b=den, odd_part_of_rad=odd_rad, two_in_den=(den % 2 == 0), five_in_res01=(res01 % 5 == 0),
                relation_ok=(den == odd_rad), integral=(den == 1))

def sector_strict(n, e):
    G = nx.Graph(e); orb = orbits(G)
    if len(orb) == n: return False
    S = sp.Matrix.hstack(*[sp.Matrix([1 if v in c else 0 for v in range(n)]) for c in orb])
    Sperp = sp.Matrix.hstack(*S.T.nullspace()) if S.T.nullspace() else None
    gP = golden_bases(restricted_charpoly(_cart(n, e), Sperp)) if Sperp is not None and Sperp.shape[1] else []
    if not gP: return True
    return all(any(sp.expand(sp.sympify(b).subs(s5, -s5) - sp.sympify(b2)) == 0 for b2 in gP) for b in gP)

def worker(job):
    n, e, pstr, plist = job
    e2 = [tuple(x) for x in e]; p = sp.sympify(pstr); C = _cart(n, e2); k = n // 2
    rec = dict(edges=e, charpoly=pstr, partnered=bool(plist), n_placements=len(plist))
    try:
        # --- species flags (all strict trees) ---
        G = nx.Graph(e2); rec["orbits"] = len(orbits(G)); rec["asymmetric"] = (rec["orbits"] == n)
        # --- certificate over all Galois halves (all strict trees) ---
        hs = halves(p); rec["n_halves"] = None if hs is None else len(hs)
        rec["halves"] = []
        if hs:
            for q in hs:
                d = denominators(q); rec["halves"].append(d)
        rec["any_integral_half"] = any(h.get("integral") for h in rec["halves"])
        if not plist:
            rec["sector_strict"] = sector_strict(n, e2); rec["cls"] = "asymmetric" if rec["asymmetric"] else "unpartnered"
            return rec
        # --- fold half: a non-degenerate partner q ---
        qs = []
        for q, _, _ in plist:
            if not any(sp.expand(q - q2) == 0 for q2 in qs): qs.append(q)
        q_good = None
        for qc in qs:
            if sp.gcd(sp.Poly(qc, t, domain=K), sp.Poly(sig(qc), t, domain=K)).degree() == 0: q_good = qc; break
        rec["q_groupings"] = len(qs)
        if q_good is None:
            rec["cls"] = "degenerate-partner"; rec["sector_strict"] = sector_strict(n, e2); return rec
        rec["cls"] = "parent-candidate"
        # --- fold-half denominators (T2, T3) ---
        rec["fold_half"] = denominators(q_good)
        # --- T1 battery (Phase 6 verbatim, r22 repairs) ---
        P = bezout_proj(C, q_good)
        ledger = (simp(P * P - P) == sp.zeros(n, n)) and (simp(P + sigma(P) - sp.eye(n)) == sp.zeros(n, n)) and P.rank() == k
        R = simp(P - sigma(P)); S5R = simp(s5 * R)
        match = {}; form = True
        for v in range(n):
            row = [(w, S5R[v, w]) for w in range(n) if w != v and S5R[v, w] != 0]
            if not (S5R[v, v] ** 2 == 1 and len(row) == 1 and row[0][1] ** 2 == 4): form = False; break
            match[v] = row[0][0]
        anti = form and all(S5R[v, v] == -S5R[match[v], match[v]] for v in match)
        rec.update(ledger=ledger, form=form, anti=anti)
        quot = bond = multpat = single = copyint = over = False
        if form and anti:
            pairs = sorted({frozenset((v, match[v])) for v in match}, key=sorted); pid = {fs: i for i, fs in enumerate(pairs)}
            mult = {}
            for a, b in e2:
                pa = pid[[fs for fs in pairs if a in fs][0]]; pb = pid[[fs for fs in pairs if b in fs][0]]
                if pa != pb: mult[tuple(sorted((pa, pb)))] = mult.get(tuple(sorted((pa, pb))), 0) + 1
            m3 = [kk for kk, v in mult.items() if v == 3]; QG = nx.Graph(list(mult.keys()))
            for q_, E_, pos_ in plist:
                gm = nx.algorithms.isomorphism.GraphMatcher(QG, nx.Graph(E_))
                for iso in gm.isomorphisms_iter():
                    quot = True
                    if len(m3) == 1 and any(tuple(sorted((iso[a], iso[b]))) == tuple(sorted(E_[pos_])) for a, b in m3): bond = True; break
                if bond: break
            multpat = sorted(mult.values()) == [2] * (k - 2) + [3]
            E3 = {tuple(sorted(x)) for x in e2}
            defects = [Ed for Ed in E3 if tuple(sorted((match[Ed[0]], match[Ed[1]]))) not in E3]
            single = len(defects) == 1
            copyint = single and all(S5R[v, v] == 1 for v in defects[0])
            if single:
                a, b = defects[0]
                proj = tuple(sorted((pid[[fs for fs in pairs if a in fs][0]], pid[[fs for fs in pairs if b in fs][0]])))
                over = len(m3) == 1 and proj == m3[0]
        D = sp.diag(*[G.degree(v) for v in range(n)]); B = 2 * sp.eye(n) - D
        tr_ok = sp.simplify(sp.expand((R * D).trace()) - 2 * s5 / 5) == 0
        lk = frob2(simp((sp.eye(n) - P) * B * P)); lk_ok = sp.simplify(lk - sp.Rational(2, 5)) == 0
        vx = all(sp.simplify(R[v, v] ** 2 - sp.Rational(1, 5)) == 0 for v in range(n))
        rec.update(quotient_iso=quot, mult_pattern=multpat, bond_is_mult3=bond, single_defect=single, copy_internal=copyint,
                   over_mult3=over, trace=tr_ok, leak=str(lk), leak_ok=lk_ok, vertex=vx)
        rec["battery_ok"] = all([ledger, form, anti, quot, multpat, bond, single, copyint, over, tr_ok, lk_ok, vx])
        rec["cls"] = "parent" if rec["battery_ok"] else "PARTNERED-NON-CONSTRUCTION"
        return rec
    except DegenerateFoldError as ex:
        rec["cls"] = "degenerate-partner"; rec["declared"] = str(ex); return rec
    except Exception as ex:
        rec["error"] = repr(ex); rec["cls"] = "ERROR"; return rec

def score(n, recs, orphans, k_even):
    parents = [r for r in recs if r.get("cls") == "parent"]; bad = [r for r in recs if r.get("cls") == "PARTNERED-NON-CONSTRUCTION"]
    nonpar = [r for r in recs if r.get("cls") in ("unpartnered", "asymmetric", "degenerate-partner")]
    errs = [r for r in recs if r.get("cls") == "ERROR"]
    fh = [r["fold_half"] for r in recs if r.get("fold_half") and "res01" in r["fold_half"]]
    T1 = dict(evaluable=len(parents) + len(bad), failures=len(bad), ok=(len(bad) == 0 and len(parents) + len(bad) >= 1))
    T2 = dict(fold_halves=len(fh), exceptions=sum(1 for h in fh if h["five_in_res01"]), informative=len(fh) >= 7)
    T2["ok"] = T2["exceptions"] == 0 if T2["informative"] else None
    nonunit = [h for h in fh if abs(h["res01"]) != 1]
    T3 = dict(fold_halves=len(fh), nonunit=len(nonunit), exceptions=sum(1 for h in fh if not h["relation_ok"]),
              two_in_den=sum(1 for h in fh if h["two_in_den"]), res01_values=sorted({h["res01"] for h in fh}), informative=len(nonunit) >= 3)
    T3["ok"] = T3["exceptions"] == 0 if T3["informative"] else None
    T4 = dict(nonparents=len(nonpar), false_positives=sum(1 for r in nonpar if r.get("any_integral_half")),
              sensitivity=f"{sum(1 for r in parents if r.get('any_integral_half'))}/{len(parents)}", informative=len(nonpar) >= 5)
    T4["ok"] = T4["false_positives"] == 0 if T4["informative"] else None
    asym = [r for r in recs if r.get("cls") == "asymmetric"]
    T5 = dict(asymmetric=len(asym), ok=len(asym) >= 1)
    T6 = dict(k=n // 2, in_scope=k_even, orphans=len(orphans), ok=(len(orphans) == 0) if k_even else None)
    return dict(T1=T1, T2=T2, T3=T3, T4=T4, T5=T5, T6=T6, errors=len(errs),
                classes={c: sum(1 for r in recs if r.get("cls") == c) for c in sorted({r.get("cls") for r in recs})},
                strict_trees=len(recs), strict_polys=len({r["charpoly"] for r in recs}), vacuous=len(recs) == 0)

if __name__ == "__main__":
    n = int(sys.argv[1]); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8; ts = time.strftime("%Y%m%d_%H%M%S")
    log = open(RES / f"exp_19_phase8_n{n}_{ts}_log.txt", "w")
    def L(s): print(s, flush=True); log.write(s + "\n"); log.flush()
    t0 = time.time(); L(f"exp_19 Phase 8 n={n} workers={workers} start {ts}")
    cnt, surv, strict = strict_hunt_parallel(n, workers=workers, block=2000, log=L, progress=50)
    L(f"stage 1 done: {cnt} trees, {len(surv)} survivors, {len(strict)} strict on {len({str(p) for _, p in strict})} polys [{time.time()-t0:.0f}s]")
    json.dump(dict(n=n, trees=cnt, survivors=[(e, str(p)) for e, p in surv], strict=[(e, str(p)) for e, p in strict]),
              open(RES / f"exp_19_phase8_n{n}_{ts}_census.json", "w"))
    t1 = time.time(); k = n // 2; partners = one5_partners_fast(k); L(f"partner map k={k}: {len(partners)} targets [{time.time()-t1:.0f}s]")
    jobs = [(n, e, str(p), partners.get(str(sp.expand(p)), [])) for e, p in strict]
    recs = []
    with Pool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(worker, jobs, chunksize=1), 1):
            recs.append(rec); L(f"  [{i}/{len(jobs)}] {rec.get('cls')} partnered={rec.get('partnered')} fold_half={rec.get('fold_half', {}).get('den5b') if rec.get('fold_half') else '-'} halves={rec.get('n_halves')} [{time.time()-t0:.0f}s]")
    strict_keys = {str(sp.expand(p)) for _, p in strict}
    orphans = []
    if k % 2 == 0:
        for key, plist in partners.items():
            q = plist[0][0]
            if sp.gcd(sp.Poly(q, t, domain=K), sp.Poly(sig(q), t, domain=K)).degree() == 0 and key not in strict_keys:
                orphans.append(dict(key=key, diagram=plist[0][1], pos=plist[0][2]))
    tests = score(n, recs, orphans, k % 2 == 0)
    out = dict(registration="phase8 (journals/2026-09-03_phase8_registration.md)", n=n, trees=cnt, survivors=len(surv),
               partner_targets=len(partners), records=recs, orphans=orphans, tests=tests, seconds=round(time.time() - t0, 1))
    json.dump(out, open(RES / f"exp_19_phase8_n{n}_{ts}.json", "w"), indent=1, default=str)
    L("TESTS: " + json.dumps(tests, default=str)); L("SCORE DONE")
