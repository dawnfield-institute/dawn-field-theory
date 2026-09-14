#!/usr/bin/env python3
"""exp_20: Block F (SEALED by the commit carrying journals/2026-09-07_blockF_registration.md) — the parity law, the
twist and the obstruction on signed connected unicyclic graphs, both switching classes, scored to the sealed text.
  T1 a strict (over √5) signed unicyclic graph exists only at n ≡ 0 (mod 4), in both classes        [n ≤ nmax]
  T2 the twisted class pairs over every field the balanced class pairs over (seven fields)          [n ≤ nmax_fields]
  T3 at n ≡ 2 (mod 4) no graph without an integer Cartan eigenvalue has golden content over √5      [n ≤ nmax]
Pipeline per graph and class: numeric no-integer-eigenvalue prefilter (a rational eigenvalue of a monic integer
polynomial is an integer) → exact charpoly (DomainMatrix, KA-8) → the proven norm screen at SCREEN_POINTS for
strictness (census.is_norm) → per-factor grading (signed.grade_by_factor = exp_12's grade, KA-8). At n ≡ 2 (mod 4)
every prefilter survivor is graded (T3 needs golden content, not strictness). Results append-only, timestamped.
Usage: python exp_20_signed_unicyclic.py --nmax 14 --nmax-fields 10 [--nmin 3] [--tag NAME]. Run as a file."""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, sympy as sp
HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "core"))
from signed import (t, FIELDS, A001429, cartan, charpoly_exact, grade_by_factor, pairing_fields, no_integer_eigenvalue,
                    unicyclic_graphs, adjacency, twist, class_sign, cycle_basis_single)
from census import is_norm, SCREEN_POINTS
RES = ROOT / "results"

ap = argparse.ArgumentParser()
ap.add_argument("--nmin", type=int, default=3); ap.add_argument("--nmax", type=int, default=14)
ap.add_argument("--nmax-fields", type=int, default=10); ap.add_argument("--tag", default="")
args = ap.parse_args()
ts = time.strftime("%Y%m%d_%H%M%S"); name = f"exp_20_signed_unicyclic{('_' + args.tag) if args.tag else ''}_{ts}"
RES.mkdir(exist_ok=True); log = open(RES / f"{name}_log.txt", "w")


def L(s):
    print(s, flush=True); log.write(s + "\n"); log.flush()


out = dict(registration="Block F (journals/2026-09-07_blockF_registration.md)", args=vars(args), sizes={},
           strict=[], t2_exceptions=[], t3_exceptions=[])
CLASSES = ("bal", "tw")
T0 = time.time()

for n in range(args.nmin, args.nmax + 1):
    t0 = time.time(); graphs = unicyclic_graphs(n)       # KA-7 assertion lives inside
    L(f"n={n}: {len(graphs)} unicyclic graphs (A001429 {A001429[n]}) [{time.time() - t0:.0f}s enumeration]")
    rec = {c: dict(graphs=len(graphs), survivors=0, screened=0, strict=0, golden_survivors=0, graded=0) for c in CLASSES}
    t2 = dict(evaluated=0, superset=0, equal=0, exceptions=0)
    for gi, G in enumerate(graphs):
        A0 = adjacency(G, n); cyc = cycle_basis_single(A0)          # on the UNSIGNED adjacency
        mats = {"bal": A0, "tw": twist(A0, cyc)}
        assert class_sign(mats["bal"], cyc) == 1 and class_sign(mats["tw"], cyc) == -1
        polys = {}
        for c in CLASSES:
            A = mats[c]; C = cartan(A); r = rec[c]
            survivor = no_integer_eigenvalue(C)
            need_poly = survivor or n <= args.nmax_fields
            if not need_poly: continue
            p = charpoly_exact(A); polys[c] = p
            if not survivor: continue
            r["survivors"] += 1
            P = sp.Poly(p, t)
            screened = all(is_norm(int(P.eval(x))) for x in SCREEN_POINTS)
            if screened: r["screened"] += 1
            if screened or n % 4 == 2:
                g = grade_by_factor(p, 5)[0]; r["graded"] += 1
                if g == "strict":
                    r["strict"] += 1
                    out["strict"].append(dict(n=n, cls=c, edges=sorted(map(list, G.edges())), cycle=cyc, poly=str(p)))
                    L(f"   STRICT n={n} {c} edges={sorted(G.edges())}")
                if n % 4 == 2 and g != "-":
                    r["golden_survivors"] += 1
                    out["t3_exceptions"].append(dict(n=n, cls=c, edges=sorted(map(list, G.edges())), grade=g, poly=str(p)))
        if n <= args.nmax_fields:
            fb, ft = pairing_fields(polys["bal"]), pairing_fields(polys["tw"]); t2["evaluated"] += 1
            if fb <= ft: t2["superset"] += 1
            if fb == ft: t2["equal"] += 1
            if not fb <= ft:
                t2["exceptions"] += 1
                out["t2_exceptions"].append(dict(n=n, edges=sorted(map(list, G.edges())), bal=sorted(fb), tw=sorted(ft)))
        if (gi + 1) % 500 == 0:
            L(f"   {gi + 1}/{len(graphs)} [{time.time() - t0:.0f}s] survivors {rec['bal']['survivors']}/{rec['tw']['survivors']} "
              f"strict {rec['bal']['strict']}/{rec['tw']['strict']}")
    out["sizes"][n] = dict(classes=rec, t2=t2 if n <= args.nmax_fields else None, seconds=round(time.time() - t0, 1))
    L(f"n={n} done [{time.time() - t0:.0f}s]: " + "; ".join(
        f"{c}: survivors {rec[c]['survivors']} screened {rec[c]['screened']} strict {rec[c]['strict']}"
        + (f" golden-survivors {rec[c]['golden_survivors']}" if n % 4 == 2 else "") for c in CLASSES)
      + (f"; T2 {t2['superset']}/{t2['evaluated']} superset ({t2['equal']} equal, {t2['exceptions']} exceptions)"
         if n <= args.nmax_fields else ""))


def score(out):
    sizes = out["sizes"]; tests = {}
    # T1 — the parity law on the first non-tree class
    strict_at = {n: sum(s["classes"][c]["strict"] for c in CLASSES) for n, s in sizes.items()}
    bad = {n: k for n, k in strict_at.items() if k and n % 4 != 0}
    informative = sum(strict_at.values()) > 0
    tests["T1"] = dict(informative=informative, exceptions=bad, strict_per_n=strict_at,
                       strict_per_class={c: {n: s["classes"][c]["strict"] for n, s in sizes.items()} for c in CLASSES},
                       ok=(not bad) if informative else None)
    # T2 — the twist never loses a field
    ev = sum(s["t2"]["evaluated"] for s in sizes.values() if s["t2"]); ex = len(out["t2_exceptions"])
    tests["T2"] = dict(informative=ev > 0, evaluated=ev, exceptions=ex,
                       equal=sum(s["t2"]["equal"] for s in sizes.values() if s["t2"]), ok=(ex == 0) if ev > 0 else None)
    # T3 — the obstruction is a root
    surv = sum(s["classes"][c]["survivors"] for n, s in sizes.items() if n % 4 == 2 for c in CLASSES)
    gold = sum(s["classes"][c]["golden_survivors"] for n, s in sizes.items() if n % 4 == 2 for c in CLASSES)
    tests["T3"] = dict(informative=surv > 0, survivors=surv, golden_survivors=gold,
                       sizes=[n for n in sizes if n % 4 == 2], ok=(gold == 0) if surv > 0 else None)
    return tests


out["tests"] = score(out); out["seconds"] = round(time.time() - T0, 1)
(RES / f"{name}.json").write_text(json.dumps(out, indent=1, default=str))
L("TESTS: " + json.dumps(out["tests"], default=str))
L(f"wrote results/{name}.json [{out['seconds']}s]")
L("SCORE DONE")
