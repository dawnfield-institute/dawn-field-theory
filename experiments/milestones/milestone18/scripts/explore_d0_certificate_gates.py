#!/usr/bin/env python3
"""explore_d0 (instrument validation, Block D): core/certificate.py against the known-answer gates of the
Block D draft §2.8 (KA1–KA7) and the projector-recipe gate (§2.2): the polynomial-in-C sector projector
must equal the exp_14 nullspace recipe EXACTLY on D6 and on the exp_13 core folds. No Block D object
(O1–O4) is touched. Results append-only."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
import sympy as sp, networkx as nx
from certificate import *
from ledger import bezout_proj, simp, cart, EDGES, H2q, H4, H3
from foldlaws import rational_core_factors
RES = Path(__file__).parent.parent / "results"

def dynkin(name):
    A = {"A4": (4, [(0,1),(1,2),(2,3)]), "A5": (5, [(0,1),(1,2),(2,3),(3,4)]), "D5": (5, [(0,1),(1,2),(2,3),(2,4)]),
         "D6": (6, [(0,1),(1,2),(2,3),(3,4),(3,5)]), "E6": (6, [(0,1),(1,2),(2,3),(3,4),(2,5)]),
         "E7": (7, [(0,1),(1,2),(2,3),(3,4),(4,5),(2,6)]), "E8": (8, [(i,i+1) for i in range(6)]+[(2,7)])}
    return A[name]

def exp14_recipe(n, e, q):
    """The sealed exp_14 off-core projector: bezout_proj(C, q_off)·(I − Qc) with Qc from nullspaces."""
    C = cart(n, e); p = sp.expand(C.charpoly(t).as_expr()); rat = rational_core_factors(p)
    q_off = q
    for g in rat:
        b, ex = g.as_base_exp(); q_off = sp.cancel(q_off / b ** (ex // 2))
    P_off = bezout_proj(C, sp.expand(q_off)); Qc = sp.zeros(n, n)
    for g in rat:
        b, ex = g.as_base_exp()
        for lam in sp.solve(b, t):
            V = sp.Matrix.hstack(*(C - lam * sp.eye(n)).nullspace()); Qc += simp(V * (V.T * V).inv() * V.T)
    return simp(P_off * (sp.eye(n) - Qc.applyfunc(sp.radsimp))), sp.expand(q_off), p, C

if __name__ == "__main__":
    t0 = time.time(); out = {}; L = lambda s: print(s, flush=True)
    # KA1: A4 / E8 strict: tr = 2/sqrt5, leak 2/5, |R_vv| = 1/sqrt5, matching form
    for name, q in (("A4", H2q), ("E8", sp.expand(H4.charpoly(t).as_expr()))):
        n, e = dynkin(name); r = evaluate(n, e, q=q); h = r["halves"][0]
        from matching import matching_form
        C = cart(n, e); p = sp.expand(C.charpoly(t).as_expr()); P = sector_projector(C, q, p); R = simp(P - sigma(P))
        ok = r["grade"] == "strict" and h["carrier"] and h["vertex_sq"] == ["1/5"] and matching_form(R)[0]
        out[f"KA1_{name}"] = dict(ok=ok, grade=r["grade"], trRD=h["trRD"], leak_oo=h["leak_oo"], vertex=h["vertex_sq"]); L(f"KA1 {name}: {ok} {out[f'KA1_{name}']}")
    # KA2: D6 core: off-core tr 2/sqrt5, leak_oo 2/5, modulated vertex law sqrt5*R_vv = ±(1 − Qc_vv)
    n, e = dynkin("D6"); q3 = sp.expand(H3.charpoly(t).as_expr()); P14, q_off, p, C = exp14_recipe(n, e, q3)
    Pnew = sector_projector(C, q_off, p); same = (simp(Pnew - P14) == sp.zeros(n, n))
    G = nx.Graph(e); D = sp.diag(*[G.degree(v) for v in range(n)]); cert = certificate(C, D, Pnew)
    Qc = sp.eye(n) - simp(Pnew + sigma(Pnew))          # Lemma 1: P_off + sigma P_off = I − Qc
    mod = all(sp.simplify(s5 * cert["R"][v, v] - (1 - Qc[v, v])) == 0 or sp.simplify(s5 * cert["R"][v, v] + (1 - Qc[v, v])) == 0 for v in range(n))
    B = 2 * sp.eye(n) - D; binv = (simp(B * Qc - Qc * B) == sp.zeros(n, n))
    masses = sorted({str(Qc[v, v]) for v in range(n) if Qc[v, v] != 0}); uniform = len(masses) == 1
    clean = binv or uniform                                  # exp_14 clean-regime rule
    laws = (cert["leak_oo"] == sp.Rational(2, 5)) and mod    # off->off leak law and modulated vertex law
    ok = same and grade(p)[0] == "core" and sp.simplify(cert["trRD"] - 2 * s5 / 5) == 0 and (laws == clean)
    out["KA2_D6"] = dict(ok=ok, projector_equals_exp14=same, grade=grade(p)[0], trRD=str(cert["trRD"]), leak_oo=str(cert["leak_oo"]),
                         modulated_vertex_law=mod, core_B_invariant=binv, core_masses=masses, clean_regime=clean,
                         note="draft KA2 expected leak 2/5 + modulated law on D6; exp_14's rule predicts NOT clean -> laws fail; trace law must hold")
    L(f"KA2 D6: {ok} {out['KA2_D6']}")
    # projector-recipe gate on the exp_13 core folds (n = 16, from the sealed T3 list) via the census partner q
    from census import one5_partners_fast
    partners = one5_partners_fast(8); e13 = json.load(open(RES / "exp_13_n16.json"))["T3"]
    agree_s = tried_s = agree_c = tried_c = 0
    full13 = json.load(open(RES / "exp_13_n16.json")); seen = set()
    cands = []
    for key_, lst in full13.items():
        if isinstance(lst, list):
            for rec in lst:
                if isinstance(rec, dict) and "edges" in rec:
                    k_ = str(sorted(map(tuple, rec["edges"])))
                    if k_ not in seen: seen.add(k_); cands.append(rec["edges"])
    for ed in cands:
        e16 = [tuple(x) for x in ed]; C16 = cart(16, e16); p16 = sp.expand(C16.charpoly(t).as_expr())
        pl = partners.get(str(p16))
        if not pl: continue
        q16 = pl[0][0]; rat = rational_core_factors(p16)
        try:
            if not rat:
                Pb = bezout_proj(C16, q16); Pn = sector_projector(C16, q16, p16); tried_s += 1; agree_s += int(simp(Pn - Pb) == sp.zeros(16, 16))
            else:
                P14b, q_off16, _, _ = exp14_recipe(16, e16, q16); Pn = sector_projector(C16, q_off16, p16); tried_c += 1; agree_c += int(simp(Pn - P14b) == sp.zeros(16, 16))
        except Exception as ex: L(f"  exp_13 fold skipped: {ex!r}")
    out["projector_gate_exp13"] = dict(strict_agree=agree_s, strict_tried=tried_s, core_agree=agree_c, core_tried=tried_c,
                                       ok=(tried_s > 0 and agree_s == tried_s and agree_c == tried_c))
    L(f"projector gate exp_13: strict {agree_s}/{tried_s} (vs bezout_proj), core {agree_c}/{tried_c} (vs exp_14 recipe)")
    # KA3: A5, D5, E6, E7 -> grade none
    ka3 = {}
    for name in ("A5", "D5", "E6", "E7"):
        n_, e_ = dynkin(name); p_ = sp.expand(cart(n_, e_).charpoly(t).as_expr()); ka3[name] = grade(p_)[0]
    out["KA3"] = dict(ok=all(v == "none" for v in ka3.values()), grades=ka3); L(f"KA3: {out['KA3']}")
    # KA6: C5 blind
    n_, e_ = 5, [(i, (i + 1) % 5) for i in range(5)]; r5 = evaluate(n_, e_); out["KA6_C5"] = dict(ok=r5.get("blind", False) and r5["grade"] != "none", grade=r5["grade"], blind=r5.get("blind")); L(f"KA6 C5: {out['KA6_C5']}")
    # KA5: mixed trees det -80 (H3 pair) and det -620 (H2 pair) at n = 14
    r9 = json.load(open(RES / "explore_r9_mixed_trees_n14.json")); ka5 = {}
    for rec in r9:
        e14 = [tuple(x) for x in (rec["edges"] if isinstance(rec["edges"], list) else json.loads(rec["edges"]))]; p14 = sp.expand(cart(14, e14).charpoly(t).as_expr())
        ka5[str(rec["det"])] = [(c, str(s)) for g, c, s in class_sectors(p14)]
    ok5 = any(c == "H3" for c, s in ka5.get("-80", [])) and any(c == "H2" for c, s in ka5.get("-620", []))
    out["KA5"] = dict(ok=ok5, classes=ka5); L(f"KA5: {out['KA5']}")
    # KA7: PAC trees d = 3, 4, 5: grade partial, one conjugate pair (t^2 - 4t + 1 ± sqrt5)^m, class H2 with s^2 = 2
    ka7 = {}
    for d in (3, 4, 5):
        nn = 2 ** (d + 1) - 1; ed = [(i, 2 * i + 1) for i in range(nn) if 2 * i + 1 < nn] + [(i, 2 * i + 2) for i in range(nn) if 2 * i + 2 < nn]
        p_ = sp.expand(cart(nn, ed).charpoly(t).as_expr()); gp = golden_pairs(p_); cl = class_sectors(p_)
        ka7[d] = dict(grade=grade(p_)[0], pairs=[(str(g), m) for g, sg, m in gp], classes=[(c, str(s)) for g, c, s in cl])
        L(f"KA7 d={d}: {ka7[d]} [{time.time()-t0:.0f}s]")
    ok7 = all(v["grade"] == "partial" and len(v["pairs"]) == 1 and v["classes"][0][0] == "H2" and v["classes"][0][1] == "2" for v in ka7.values())
    out["KA7"] = dict(ok=ok7, detail={str(k): v for k, v in ka7.items()})
    # KA4: the four asymmetric strict n = 20 trees: certificate set over halves reproduces the draft's values; leak != 2/5; no form
    e15 = json.load(open(RES / "exp_15_n20.json")); asym = []
    from sectors import orbits
    for rec in e15["T2"]:
        if not rec.get("partnered"):
            G20 = nx.Graph([tuple(x) for x in rec["edges"]])
            if len(orbits(G20)) == 20: asym.append(rec["edges"])
    target = {str(sp.nsimplify(x)) for x in [34*s5/125, 66*s5/125, 458*s5/1235, -458*s5/1235, 6*s5/65, -6*s5/65, -206*s5/247]}
    got = set(); leaks_ok = True; forms = []
    for ed in asym:
        r20 = evaluate(20, [tuple(x) for x in ed])
        for h in r20["halves"]:
            if "trRD" in h:
                got.add(str(sp.nsimplify(sp.sympify(h["trRD"])))); leaks_ok &= (sp.sympify(h["leak_oo"]) != sp.Rational(2, 5))
    out["KA4"] = dict(ok=(len(asym) == 4 and target <= got and leaks_ok), n_asym=len(asym), values=sorted(got), target=sorted(target), leaks_ne_2_5=leaks_ok)
    L(f"KA4: {out['KA4']} [{time.time()-t0:.0f}s]")
    out["ALL"] = all(v.get("ok") for k, v in out.items() if isinstance(v, dict) and "ok" in v)
    json.dump(out, open(RES / f"explore_d0_certificate_gates_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1, default=str)
    L(f"CERTIFICATE GATES: {'PASS' if out['ALL'] else 'FAIL'} [{time.time()-t0:.0f}s]"); L("DONE")
