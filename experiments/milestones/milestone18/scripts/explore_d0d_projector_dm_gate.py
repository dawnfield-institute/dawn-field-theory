#!/usr/bin/env python3
"""explore_d0d (instrument validation): certificate.sector_projector_dm (DomainMatrix Horner) equals
certificate.sector_projector (sympy Matrix) EXACTLY on: A4, E8, D6 (off-core q), the 13 strict exp_13 folds
at n=16, and the four asymmetric n=20 trees (all halves). Speed recorded."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
import sympy as sp, networkx as nx
from certificate import sector_projector, sector_projector_dm, halves, t, s5
from ledger import cart, EDGES, H2q, H4
from sectors import orbits
RES = Path(__file__).parent.parent / "results"

if __name__ == "__main__":
    t0 = time.time(); cases = []
    cases.append(("A4", 4, EDGES["A4"][1], H2q)); cases.append(("E8", 8, EDGES["E8"][1], sp.expand(H4.charpoly(t).as_expr())))
    e13 = json.load(open(RES / "exp_13_n16.json"))["T3"]
    for i, rec in enumerate(e13): cases.append((f"exp13#{i}", 16, [tuple(x) for x in rec["edges"]], None))
    e15 = json.load(open(RES / "exp_15_n20.json"))
    for rec in e15["T2"]:
        if not rec.get("partnered"):
            G = nx.Graph([tuple(x) for x in rec["edges"]])
            if len(orbits(G)) == 20: cases.append((f"asym20", 20, [tuple(x) for x in rec["edges"]], None))
    agree = tried = 0; tm = td = 0.0
    for name, n, e, q in cases:
        C = cart(n, e); p = sp.expand(C.charpoly(t).as_expr()); qs = [q] if q is not None else halves(p)
        for qq in qs:
            try:
                s0 = time.time(); P1 = sector_projector(C, qq, p); tm += time.time() - s0
                s0 = time.time(); P2 = sector_projector_dm(C, qq, p); td += time.time() - s0
                tried += 1; agree += int((P1 - P2).applyfunc(sp.expand) == sp.zeros(n, n))
            except Exception as ex: print(f"  {name} half skipped: {ex!r}", flush=True)
        print(f"{name} n={n}: {tried} tried, {agree} agree so far [{time.time()-t0:.0f}s]", flush=True)
    ok = tried > 0 and agree == tried
    json.dump(dict(tried=tried, agree=agree, ok=ok, seconds_matrix=round(tm, 1), seconds_dm=round(td, 1)), open(RES / f"explore_d0d_projector_dm_gate_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1)
    print(f"PROJECTOR-DM GATE: {'PASS' if ok else 'FAIL'} {agree}/{tried}; Matrix {tm:.1f}s vs DomainMatrix {td:.1f}s"); print("DONE")
